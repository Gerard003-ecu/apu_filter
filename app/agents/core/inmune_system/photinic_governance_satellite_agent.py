# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO : Photinic Governance Satellite Agent                                 ║
║ RUTA   : app/agents/core/inmune_system/photinic_governance_satellite_agent.py║
║ VERSIÓN: 6.0.0 — tres fases anidadas, auditoría numérica nivel 3             ║
╚══════════════════════════════════════════════════════════════════════════════╝

Agente supervisor para la validación de políticas federadas. El flujo se organiza
en tres fases anidadas:

    Fase 1 — Observe:
        Validación de entrada, regularidad de Banach, tensor espectral,
        refinamiento numérico acotado de nivel 3 y sellado HMAC.

    Fase 2 — Orient:
        Cálculo del motor, comparación agente↔motor y conservación del reporte
        físico bruto cuando falla la certificación estricta.

    Fase 3 — Decide & Act:
        Clasificador de Heyting Ω₃, histéresis, tokens firmados contra repetición
        y modelo RC de interlock.

Notas de alcance y seguridad
----------------------------
* `majorana_spinor_norm` conserva el nombre de interfaz histórico, pero aquí
  designa el proxy espectral sqrt(Tr(S²)); por sí solo no constituye una medida
  física de un espinor de Majorana.
* Los indicadores Choi, causalidad y CHSH dependen del motor inyectado. El motor
  de fallback es una implementación demostrativa y NO certifica física cuántica
  ni es apto para producción.
* El auditor de Poincaré es opcional. Si se usa, recibe explícitamente un estado
  N-cuerpos Newtoniano en unidades SI; no interpreta las políticas como cuerpos
  celestes.
* El modelo RC no acciona silicio. Sin un adaptador externo de hardware, el
  certificado solo registra una solicitud lógica y una latencia calculada.
  Ni CPython ni un callback normal ofrecen una garantía temporal de 400 ns.
* La automejora de nivel 3 es un procedimiento determinista y acotado de
  verificación/refinamiento. No reescribe el programa, no aprende credenciales
  y nunca modifica los umbrales de seguridad.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from enum import IntEnum
from typing import (
    Any,
    Callable,
    Final,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


# ------------------------------------------------------------------------------
# COMPATIBILIDAD CATEGÓRICA E IMPORTACIONES RESILIENTES
# ------------------------------------------------------------------------------
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:

    class Morphism:
        """Stub de compatibilidad para entornos desacoplados."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

    class TopologicalInvariantError(Exception):
        """Violación topológico-algebraica."""

        pass


try:
    from photinic_governance_satellite_engine import (
        PhotinicGovernanceSatelliteEngine,
        PhotinicEngineState,
        PhotinicObservationKernel as EngineKernel,
        PhotinicPolicyReport as EngineReport,
        PhotinicEngineError,
        PhotinicDimensionError,
        PhotinicNonPositiveDefiniteError,
        PhotinicCausalViolationError,
        PhotinicNonCausalChannelError,
        PhotinicTsirelsonViolationError,
        KahanNeumaierSum,
    )
except ImportError:
    try:
        from app.agents.core.inmune_system.photinic_governance_satellite_engine import (
            PhotinicGovernanceSatelliteEngine,
            PhotinicEngineState,
            PhotinicObservationKernel as EngineKernel,
            PhotinicPolicyReport as EngineReport,
            PhotinicEngineError,
            PhotinicDimensionError,
            PhotinicNonPositiveDefiniteError,
            PhotinicCausalViolationError,
            PhotinicNonCausalChannelError,
            PhotinicTsirelsonViolationError,
            KahanNeumaierSum,
        )
    except ImportError:
        try:
            from app.core.immune_system.photinic_governance_satellite_engine import (
                PhotinicGovernanceSatelliteEngine,
                PhotinicEngineState,
                PhotinicObservationKernel as EngineKernel,
                PhotinicPolicyReport as EngineReport,
                PhotinicEngineError,
                PhotinicDimensionError,
                PhotinicNonPositiveDefiniteError,
                PhotinicCausalViolationError,
                PhotinicNonCausalChannelError,
                PhotinicTsirelsonViolationError,
                KahanNeumaierSum,
            )
        except ImportError:
            # Fallback autocontenido para desarrollo y pruebas.
            # NO proporciona certificación cuántica apta para producción.
            class PhotinicEngineError(Exception):
                pass

            class PhotinicDimensionError(PhotinicEngineError):
                pass

            class PhotinicNonPositiveDefiniteError(PhotinicEngineError):
                pass

            class PhotinicCausalViolationError(PhotinicEngineError):
                pass

            class PhotinicNonCausalChannelError(PhotinicCausalViolationError):
                pass

            class PhotinicTsirelsonViolationError(PhotinicCausalViolationError):
                pass

            class KahanNeumaierSum:
                """Suma compensada mínima para el motor de fallback."""

                @staticmethod
                def sum(arr: Union[NDArray[np.float64], Sequence[float]]) -> float:
                    s, c = 0.0, 0.0
                    for x in arr:
                        val = float(x)
                        t = s + val
                        if abs(s) >= abs(val):
                            c += (s - t) + val
                        else:
                            c += (val - t) + s
                        s = t
                    return float(s + c)

                @staticmethod
                def compensated_dot(
                    u: NDArray[np.float64], v: NDArray[np.float64]
                ) -> float:
                    if u.shape != v.shape:
                        raise ValueError("Las dimensiones del producto punto no coinciden.")
                    return KahanNeumaierSum.sum(u * v)

                @staticmethod
                def compensated_l2_norm(v: NDArray[np.float64]) -> float:
                    return float(math.sqrt(max(
                        KahanNeumaierSum.compensated_dot(v, v), 0.0
                    )))

            @dataclass(frozen=True, slots=True)
            class EngineKernel:
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

            @dataclass(frozen=True, slots=True)
            class EngineReport:
                kernel: EngineKernel
                idempotence_residual: float
                trace_rank_mismatch: float
                choi_min_eigenvalue: float
                choi_trace_residual: float
                choi_subunital_defect_min_eigenvalue: float
                tsirelson_chsh_value: float
                horodecki_eigs: Tuple[float, float]
                horodecki_bipartition_pairs: int
                is_choi_completely_positive: bool
                is_trace_non_increasing: bool
                is_causal_non_signaling: bool
                is_tsirelson_bounded: bool
                is_classical_lhv: bool
                timestamp: float

            @dataclass(frozen=True, slots=True)
            class PhotinicEngineState:
                kernel: EngineKernel
                policy_report: EngineReport
                fpu_execution_time_ms: float
                federated_density_matrix: NDArray[np.float64]
                von_neumann_entropy: float
                quantum_purity: float
                bell_pair_entanglement_entropy: float
                fpu_precision_drift: float
                cryptographic_seal: str
                phase_signature: str

            class PhotinicGovernanceSatelliteEngine:
                """
                Motor aproximado para desarrollo. Sus valores de Choi/CHSH son
                indicadores sintéticos, no una certificación física.
                """

                def __init__(self, tolerance: float = 1.0e-12) -> None:
                    self.tol = float(tolerance)

                def canonize_photinic_observation_kernel(
                    self,
                    node_policies: Sequence[NDArray[np.float64]],
                ) -> EngineKernel:
                    if not node_policies:
                        raise PhotinicDimensionError("N = 0.")
                    clean = [
                        np.array(v, dtype=np.float64, copy=True, order="C")
                        for v in node_policies
                    ]
                    p = np.vstack(clean)
                    n, dim = p.shape
                    gram = (p.T @ p) / float(n)
                    gram += self.tol * np.eye(dim, dtype=np.float64)
                    eigs = np.sort(la.eigvalsh(gram))
                    min_eig = max(float(eigs[0]), np.finfo(np.float64).tiny)
                    condition = float(max(eigs[-1], min_eig) / min_eig)
                    majorana = float(la.norm(gram, ord="fro"))
                    ratios = tuple(
                        float(np.sum(np.abs(v)) / max(float(la.norm(v)), 1.0e-15))
                        if np.any(v) else 0.0
                        for v in clean
                    )
                    return EngineKernel(
                        node_policies=tuple(clean),
                        majorana_spinor_norm=majorana,
                        majorana_chiral_parity=float(np.trace(gram) / max(majorana, 1e-15)),
                        banach_ratios=ratios,
                        gram_matrix=gram,
                        gram_eigenvalues=eigs,
                        gram_symmetry_defect=0.0,
                        gram_reconstruction_residual=0.0,
                        condition_number=condition,
                        spectral_gap=float(eigs[1] - eigs[0]) if dim > 1 else 0.0,
                        sha256_seal=hashlib.sha256(p.tobytes()).hexdigest(),
                        timestamp=time.time(),
                    )

                def orient_photinic_policy(self, kernel: EngineKernel) -> EngineReport:
                    p = np.vstack(kernel.node_policies)
                    n = p.shape[0]

                    _, singular_values, vt = la.svd(p, full_matrices=False)
                    rank = max(int(np.sum(singular_values > 1.0e-10)), 1)
                    projector = vt[:rank].T @ vt[:rank]
                    idempotence = float(la.norm(
                        projector @ projector - projector, ord="fro"
                    ))
                    trace_mismatch = abs(float(np.trace(projector)) - float(rank))

                    mid = n // 2 if n >= 2 else 1
                    pairs = min(mid, n - mid) if n >= 2 else 0
                    if pairs:
                        corr = (p[:pairs].T @ p[mid:mid + pairs]) / float(pairs)
                    else:
                        corr = np.outer(p[0], p[0])

                    eigs = np.sort(la.eigvalsh(corr.T @ corr))[::-1]
                    u1 = float(max(eigs[0], 0.0)) if len(eigs) else 0.0
                    u2 = float(max(eigs[1], 0.0)) if len(eigs) > 1 else 0.0
                    chsh = float(2.0 * math.sqrt(max(u1 + u2, 0.0)))
                    tsirelson_limit = 2.0 * math.sqrt(2.0)

                    return EngineReport(
                        kernel=kernel,
                        idempotence_residual=idempotence,
                        trace_rank_mismatch=trace_mismatch,
                        choi_min_eigenvalue=0.0,
                        choi_trace_residual=0.0,
                        choi_subunital_defect_min_eigenvalue=0.0,
                        tsirelson_chsh_value=chsh,
                        horodecki_eigs=(u1, u2),
                        horodecki_bipartition_pairs=pairs,
                        is_choi_completely_positive=True,
                        is_trace_non_increasing=True,
                        is_causal_non_signaling=True,
                        is_tsirelson_bounded=chsh <= tsirelson_limit + 1.0e-9,
                        is_classical_lhv=chsh <= 2.0 + 1.0e-9,
                        timestamp=time.time(),
                    )

                def execute_photinic_audit(self, target: Any) -> PhotinicEngineState:
                    start = time.perf_counter()
                    report = (
                        target
                        if isinstance(target, EngineReport)
                        else self.orient_photinic_policy(
                            self.canonize_photinic_observation_kernel(target)
                        )
                    )
                    if not report.is_choi_completely_positive:
                        raise PhotinicNonPositiveDefiniteError("Fallback: canal no-CP.")
                    if not report.is_causal_non_signaling:
                        raise PhotinicNonCausalChannelError("Fallback: canal no causal.")
                    if not report.is_tsirelson_bounded:
                        raise PhotinicTsirelsonViolationError(
                            "Fallback: indicador sintético supera la cota de Tsirelson."
                        )

                    trace = float(np.trace(report.kernel.gram_matrix))
                    rho = report.kernel.gram_matrix / max(trace, 1.0e-15)
                    return PhotinicEngineState(
                        kernel=report.kernel,
                        policy_report=report,
                        fpu_execution_time_ms=(time.perf_counter() - start) * 1000.0,
                        federated_density_matrix=rho,
                        von_neumann_entropy=0.0,
                        quantum_purity=1.0,
                        bell_pair_entanglement_entropy=0.0,
                        fpu_precision_drift=report.idempotence_residual,
                        cryptographic_seal=report.kernel.sha256_seal,
                        phase_signature="FALLBACK_DEMONSTRATION_ONLY",
                    )


logger = logging.getLogger("APU.Agents.Omega.PhotinicGovernanceSatelliteAgent")


# ------------------------------------------------------------------------------
# CONSTANTES OPERACIONALES Y METROLÓGICAS
# ------------------------------------------------------------------------------
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_SAFETY_FLOOR: Final[float] = 1.0e-15
_FLOAT_TINY: Final[float] = float(np.finfo(np.float64).tiny)

_HARD_VETO_CHOI_MIN_EIGENVALUE: Final[float] = -1.0e-6
_HARD_VETO_IDEMPOTENCE_FRACTION: Final[float] = 0.50
_SOFT_VETO_IDEMPOTENCE_LOWER_FRACTION: Final[float] = 0.30
_SOFT_VETO_IDEMPOTENCE_UPPER_FRACTION: Final[float] = 0.50
_HARD_VETO_MAX_CONDITION_NUMBER: Final[float] = 1.0e10
_HARD_VETO_TRACE_RESIDUAL: Final[float] = 1.0e-6
_HARD_VETO_GRAM_SYMMETRY_DEFECT: Final[float] = 1.0e-10
_SOFT_VETO_CHSH_CLASSICAL_THRESHOLD: Final[float] = 2.0
_TSIRELSON_BOUND: Final[float] = 2.0 * math.sqrt(2.0)
_TSIRELSON_NUMERICAL_MARGIN: Final[float] = 1.0e-9
_CROSS_ENGINE_CONSISTENCY_TOLERANCE: Final[float] = 1.0e-6

_TOKEN_FRESHNESS_WINDOW_SECONDS: Final[float] = 300.0
_TOKEN_MAX_FUTURE_SKEW_SECONDS: Final[float] = 30.0
_FOCK_ANNIHILATION_CROSS_SECTION_SIGMA: Final[float] = 0.5

# Parámetros ilustrativos del modelo RC. Se presupone un driver de compuerta
# externo; NO se modela la conexión directa de un GPIO ESP32 al tiristor.
_CROWBAR_MAX_MODEL_LATENCY_NS: Final[float] = 400.0
_ESP32_CLOCK_FREQ_HZ: Final[float] = 240.0e6
_ESP32_CYCLE_TIME_NS: Final[float] = 1.0e9 / _ESP32_CLOCK_FREQ_HZ
_ISR_DISPATCH_OVERHEAD_CYCLES: Final[int] = 42
_BT151_VGT_VOLTS: Final[float] = 1.1
_BT151_IGT_AMPS: Final[float] = 0.005
_GATE_DRIVER_VOH_VOLTS: Final[float] = 5.0
_GATE_DRIVER_MAX_SOURCE_CURRENT_AMPS: Final[float] = 0.100
_CROWBAR_GATE_RESISTOR_OHMS: Final[float] = 150.0
_CROWBAR_GATE_CAPACITANCE_FARADS: Final[float] = 1.0e-9
_CROWBAR_MIN_SAFETY_MARGIN: Final[float] = 5.0


# ------------------------------------------------------------------------------
# EXCEPCIONES DEL AGENTE
# ------------------------------------------------------------------------------
class PhotinicAgentError(PhotinicEngineError):
    """Excepción raíz del agente."""

    pass


class PhotinicInputValidationError(PhotinicAgentError):
    """Entrada con dimensiones, formato o valores numéricos no válidos."""

    pass


class PhotinicNumericalAuditError(PhotinicAgentError):
    """Falló un invariante numérico requerido por la auditoría nivel 3."""

    pass


class PhotinicMetricConsistencyError(PhotinicAgentError):
    """Divergencia metrológica entre el cálculo del agente y el del motor."""

    def __init__(self, message: str, relative_residual: float = math.inf) -> None:
        super().__init__(message)
        self.relative_residual = float(relative_residual)


class PhotinicCelestialError(PhotinicAgentError):
    """Estado o integración Newtoniana no válida."""

    pass


class PhotinicCelestialCollisionError(PhotinicCelestialError):
    """Colisión o separación numéricamente singular en el modelo N-cuerpos."""

    pass


class PhotinicCrowbarMisfireError(PhotinicAgentError):
    """El modelo RC no satisface los requisitos configurados de disparo."""

    pass


class PhotinicTokenFreshnessError(PhotinicAgentError):
    """Token firmado expirado, futuro o ya consumido."""

    pass


# ------------------------------------------------------------------------------
# UTILIDADES NUMÉRICAS ESTABLES
# ------------------------------------------------------------------------------
def _stable_l2(values: Union[NDArray[np.float64], Sequence[float]]) -> float:
    """Norma L2 escalada para evitar desbordamiento por suma de cuadrados."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size == 0:
        return 0.0
    if not np.all(np.isfinite(arr)):
        return math.inf
    scale = float(np.max(np.abs(arr)))
    if scale == 0.0:
        return 0.0
    normalized_sum = math.fsum((float(x) / scale) ** 2 for x in arr)
    result = scale * math.sqrt(normalized_sum)
    return float(result) if math.isfinite(result) else math.inf


def _relative_residual(
    candidate: Union[NDArray[np.float64], Sequence[float]],
    reference: Union[NDArray[np.float64], Sequence[float]],
) -> float:
    """Residuo relativo estable ||candidate-reference||₂ / max(||reference||₂, floor)."""
    a = np.asarray(candidate, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape:
        return math.inf
    with np.errstate(over="ignore", invalid="ignore"):
        difference = a - b
    numerator = _stable_l2(difference)
    denominator = max(_stable_l2(b), _WILKINSON_SAFETY_FLOOR)
    return float(numerator / denominator) if math.isfinite(numerator) else math.inf


def _frozen_float_array(values: Any) -> NDArray[np.float64]:
    """Copia C-contigua float64, canonizada a little-endian y de solo lectura."""
    try:
        arr = np.array(values, dtype="<f8", order="C", copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PhotinicInputValidationError(
            f"No se pudo convertir la entrada a float64: {exc}"
        ) from exc
    arr.setflags(write=False)
    return arr


def _finite_optional(value: Any) -> Optional[float]:
    """Convierte a float finito; devuelve None para telemetría no finita."""
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


# ------------------------------------------------------------------------------
# ÁLGEBRA DE HEYTING Ω₃
# ------------------------------------------------------------------------------
class PhotinicHeytingVerdict(IntEnum):
    """Cadena finita ordenada VETOED < DEGRADED < COHERENT."""

    VETOED = 0
    DEGRADED = 1
    COHERENT = 2

    @property
    def canonical_name(self) -> str:
        return self.name


class HeytingOmega3Algebra:
    """
    Álgebra de Heyting sobre la cadena finita Ω₃.

    En esta cadena:
        a ⇒ b = ⊤ si a ≤ b; en otro caso, a ⇒ b = b.
        ¬a = a ⇒ ⊥.

    Por ello ¬¬(*) = ⊤ ≠ *, sin invocar propiedades no demostradas de un topos
    de De Rham.
    """

    @staticmethod
    def meet(
        a: PhotinicHeytingVerdict,
        b: PhotinicHeytingVerdict,
    ) -> PhotinicHeytingVerdict:
        return PhotinicHeytingVerdict(min(int(a), int(b)))

    @staticmethod
    def join(
        a: PhotinicHeytingVerdict,
        b: PhotinicHeytingVerdict,
    ) -> PhotinicHeytingVerdict:
        return PhotinicHeytingVerdict(max(int(a), int(b)))

    @staticmethod
    def implication(
        a: PhotinicHeytingVerdict,
        b: PhotinicHeytingVerdict,
    ) -> PhotinicHeytingVerdict:
        if int(a) <= int(b):
            return PhotinicHeytingVerdict.COHERENT
        return b

    @classmethod
    def pseudocomplement(
        cls,
        a: PhotinicHeytingVerdict,
    ) -> PhotinicHeytingVerdict:
        return cls.implication(a, PhotinicHeytingVerdict.VETOED)

    @classmethod
    def verify_intuitionistic_failure(cls) -> bool:
        double_negation = cls.pseudocomplement(
            cls.pseudocomplement(PhotinicHeytingVerdict.DEGRADED)
        )
        return double_negation != PhotinicHeytingVerdict.DEGRADED

    @classmethod
    def verify_lattice_laws(cls) -> bool:
        elements = tuple(PhotinicHeytingVerdict)
        for x in elements:
            for y in elements:
                for z in elements:
                    if cls.meet(x, y) != cls.meet(y, x):
                        return False
                    if cls.join(x, y) != cls.join(y, x):
                        return False
                    if cls.meet(x, cls.join(y, z)) != cls.join(
                        cls.meet(x, y), cls.meet(x, z)
                    ):
                        return False
                    if cls.join(x, cls.meet(y, z)) != cls.meet(
                        cls.join(x, y), cls.join(x, z)
                    ):
                        return False
                    # Adjunción de Heyting: x ∧ a ≤ b ⇔ x ≤ (a ⇒ b).
                    lhs = int(cls.meet(x, y)) <= int(z)
                    rhs = int(x) <= int(cls.implication(y, z))
                    if lhs != rhs:
                        return False
        return cls.verify_intuitionistic_failure()


# ------------------------------------------------------------------------------
# AUTOMEJORA RECURSIVA NIVEL 3: EVIDENCIA Y REFINAMIENTO ACOTADO
# ------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class RecursiveAuditEvidence:
    """Evidencia verificable del refinamiento numérico de profundidad máxima 3."""

    maximum_depth: int
    depth_used: int
    blas_vs_compensated_residual: float
    permutation_residual: float
    spectral_reconstruction_residual: float
    verified: bool


class RecursiveSelfImprovementLevel3:
    """
    Rutina de refinamiento/autoverificación de profundidad fija 3.

    Niveles:
      1. Contrasta Gram BLAS con acumulación Neumaier elemento a elemento.
      2. Si el residuo rebasa el límite, sustituye el candidato por la referencia
         compensada y vuelve a verificar (recursión acotada).
      3. Comprueba invariancia ante permutación de filas y reconstrucción espectral.

    No modifica código, credenciales ni umbrales de decisión.
    """

    MAX_DEPTH: Final[int] = 3

    @staticmethod
    def _compensated_gram_rows(p: NDArray[np.float64]) -> NDArray[np.float64]:
        """Suma compensada matricial de productos externos, sobre filas escaladas."""
        n, dim = p.shape
        total = np.zeros((dim, dim), dtype=np.float64)
        compensation = np.zeros_like(total)

        for row in p:
            term = np.outer(row, row)
            provisional = total + term
            larger = np.abs(total) >= np.abs(term)
            compensation += np.where(
                larger,
                (total - provisional) + term,
                (term - provisional) + total,
            )
            total = provisional

        return (total + compensation) / float(n)

    @classmethod
    def _recursive_refine(
        cls,
        candidate: NDArray[np.float64],
        reference: NDArray[np.float64],
        target: float,
        depth: int = 1,
    ) -> Tuple[NDArray[np.float64], int, float]:
        residual = _relative_residual(candidate, reference)
        if math.isfinite(residual) and residual <= target:
            return candidate, depth, residual
        if depth >= cls.MAX_DEPTH:
            raise PhotinicNumericalAuditError(
                f"Refinamiento nivel 3 agotado; residuo={residual:.4e}."
            )
        # Corrección monotónica hacia la referencia compensada; no hay relajación
        # de tolerancias ni actualización de parámetros de seguridad.
        return cls._recursive_refine(
            reference.copy(), reference, target, depth + 1
        )

    @classmethod
    def spectral_tensor(
        cls,
        policies: NDArray[np.float64],
        ridge: float,
        tolerance: float,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        float,
        float,
        float,
        float,
        float,
        RecursiveAuditEvidence,
    ]:
        """
        Construye S = PᵀP/N + ridge·I, verifica la suma y obtiene los invariantes.

        El escalado previo limita el riesgo de desbordamiento de PᵀP. Si el tensor
        real no cabe en float64, la rutina falla explícitamente.
        """
        p = np.asarray(policies, dtype=np.float64)
        if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] == 0:
            raise PhotinicInputValidationError("P debe tener forma N×d, con N,d > 0.")
        if not np.all(np.isfinite(p)):
            raise PhotinicInputValidationError("P contiene valores no finitos.")
        if not math.isfinite(ridge) or ridge <= 0.0:
            raise PhotinicInputValidationError("La regularización ridge debe ser positiva.")
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise PhotinicInputValidationError("La tolerancia debe ser positiva.")

        n, dim = p.shape
        max_abs = float(np.max(np.abs(p)))

        if max_abs == 0.0:
            normalized = np.zeros_like(p)
            scale_squared = 0.0
        else:
            if max_abs > math.sqrt(np.finfo(np.float64).max):
                raise PhotinicNumericalAuditError(
                    "La escala de las políticas excede el dominio seguro de float64."
                )
            normalized = p / max_abs
            scale_squared = max_abs * max_abs
            if not math.isfinite(scale_squared):
                raise PhotinicNumericalAuditError("Desbordamiento al escalar PᵀP.")

        target = max(
            float(tolerance),
            128.0 * _MACHINE_EPS * max(1, dim),
        )
        blas_gram = (normalized.T @ normalized) / float(n)
        compensated_gram = cls._compensated_gram_rows(normalized)

        candidate, depth_used, blas_residual = cls._recursive_refine(
            blas_gram, compensated_gram, target, depth=1
        )

        reversed_reference = cls._compensated_gram_rows(normalized[::-1])
        permutation_residual = _relative_residual(
            compensated_gram, reversed_reference
        )
        if not math.isfinite(permutation_residual) or permutation_residual > target:
            raise PhotinicNumericalAuditError(
                "La suma compensada no superó la prueba de permutación de filas: "
                f"{permutation_residual:.4e}."
            )

        with np.errstate(over="raise", invalid="raise"):
            try:
                gram = candidate * scale_squared
                gram += ridge * np.eye(dim, dtype=np.float64)
            except FloatingPointError as exc:
                raise PhotinicNumericalAuditError(
                    "Desbordamiento al construir el tensor espectral."
                ) from exc

        if not np.all(np.isfinite(gram)):
            raise PhotinicNumericalAuditError("El tensor espectral no es finito.")

        symmetry_defect = _relative_residual(gram, gram.T)
        gram = 0.5 * (gram + gram.T)

        try:
            eigenvalues, eigenvectors = la.eigh(gram, check_finite=True)
        except la.LinAlgError as exc:
            raise PhotinicNumericalAuditError(
                f"Falló la diagonalización simétrica: {exc}"
            ) from exc

        spectral_scale = max(
            float(np.max(np.abs(eigenvalues))),
            _WILKINSON_SAFETY_FLOOR,
        )
        if float(eigenvalues[0]) < -target * spectral_scale:
            raise PhotinicNumericalAuditError(
                "El tensor de Gram perdió positividad más allá del error numérico."
            )

        reconstruction = (eigenvectors * eigenvalues[np.newaxis, :]) @ eigenvectors.T
        reconstruction_residual = _relative_residual(reconstruction, gram)
        reconstruction_limit = max(1.0e-10, 10.0 * target)
        if (
            not math.isfinite(reconstruction_residual)
            or reconstruction_residual > reconstruction_limit
        ):
            raise PhotinicNumericalAuditError(
                "Falló la reconstrucción espectral nivel 3: "
                f"{reconstruction_residual:.4e}."
            )

        majorana_proxy = _stable_l2(eigenvalues)
        if not math.isfinite(majorana_proxy):
            raise PhotinicNumericalAuditError("Proxy espectral no finito.")

        trace_value = math.fsum(float(x) for x in eigenvalues)
        chiral_proxy = float(
            trace_value / max(majorana_proxy, _WILKINSON_SAFETY_FLOOR)
        )
        min_eigenvalue = max(float(eigenvalues[0]), _WILKINSON_SAFETY_FLOOR)
        max_eigenvalue = max(float(eigenvalues[-1]), min_eigenvalue)
        condition_number = float(max_eigenvalue / min_eigenvalue)
        spectral_gap = (
            float(eigenvalues[1] - eigenvalues[0]) if dim > 1 else 0.0
        )

        evidence = RecursiveAuditEvidence(
            maximum_depth=cls.MAX_DEPTH,
            depth_used=depth_used,
            blas_vs_compensated_residual=float(blas_residual),
            permutation_residual=float(permutation_residual),
            spectral_reconstruction_residual=float(reconstruction_residual),
            verified=True,
        )

        return (
            gram,
            eigenvalues,
            float(majorana_proxy),
            chiral_proxy,
            condition_number,
            spectral_gap,
            float(symmetry_defect),
            evidence,
        )


# ------------------------------------------------------------------------------
# MECÁNICA CELESTE DE POINCARÉ: ESTADO N-CUERPOS Y SECCIÓN DE POINCARÉ
# ------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PoincareCelestialState:
    """Estado Newtoniano N-cuerpos en SI: posiciones [m], velocidades [m/s], masas [kg]."""

    positions_m: NDArray[np.float64]
    velocities_m_s: NDArray[np.float64]
    masses_kg: NDArray[np.float64]

    def __post_init__(self) -> None:
        try:
            positions = np.array(
                self.positions_m, dtype=np.float64, order="C", copy=True
            )
            velocities = np.array(
                self.velocities_m_s, dtype=np.float64, order="C", copy=True
            )
            masses = np.array(self.masses_kg, dtype=np.float64, order="C", copy=True)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PhotinicCelestialError(
                f"Estado celeste no convertible a float64: {exc}"
            ) from exc

        if positions.ndim != 2 or positions.shape[1] != 3:
            raise PhotinicCelestialError("positions_m debe tener forma N×3.")
        if velocities.shape != positions.shape:
            raise PhotinicCelestialError("velocities_m_s debe tener forma N×3.")
        if masses.ndim != 1 or masses.shape[0] != positions.shape[0]:
            raise PhotinicCelestialError("masses_kg debe tener forma N.")
        if positions.shape[0] < 2:
            raise PhotinicCelestialError("Se requieren al menos dos cuerpos.")
        if not (
            np.all(np.isfinite(positions))
            and np.all(np.isfinite(velocities))
            and np.all(np.isfinite(masses))
        ):
            raise PhotinicCelestialError("El estado celeste contiene valores no finitos.")
        if np.any(masses <= 0.0):
            raise PhotinicCelestialError("Todas las masas deben ser estrictamente positivas.")

        positions.setflags(write=False)
        velocities.setflags(write=False)
        masses.setflags(write=False)
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "velocities_m_s", velocities)
        object.__setattr__(self, "masses_kg", masses)


@dataclass(frozen=True, slots=True)
class PoincareCelestialAudit:
    """Diagnósticos de integración, invariantes y retornos a una sección de Poincaré."""

    steps: int
    step_size_seconds: float
    integration_time_seconds: float
    initial_energy_j: float
    final_energy_j: float
    relative_energy_drift: float
    relative_angular_momentum_drift: float
    reversibility_residual: float
    minimum_separation_m: float
    section_crossings: int
    return_intervals_seconds: Tuple[float, ...]
    section_points: Tuple[Tuple[float, ...], ...]
    stability_utilization: float
    stable: bool
    integrator: str
    audit_timestamp: float


class PoincareCelestialMechanics:
    """
    Integrador Newtoniano N-cuerpos leapfrog kick-drift-kick.

    La sección se define explícitamente por x_j - x_i = 0 con velocidad relativa
    positiva en x. El método es simpléctico para el Hamiltoniano separable ideal;
    los residuos reportados son diagnósticos finitos, no una prueba de estabilidad
    orbital a tiempo infinito.
    """

    def __init__(
        self,
        gravitational_constant: float = 6.67430e-11,
        collision_radius_m: float = 0.0,
        max_energy_drift: float = 1.0e-3,
        max_angular_momentum_drift: float = 1.0e-8,
        max_reversibility_residual: float = 1.0e-8,
        max_steps: int = 100_000,
    ) -> None:
        values = (
            gravitational_constant,
            collision_radius_m,
            max_energy_drift,
            max_angular_momentum_drift,
            max_reversibility_residual,
        )
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("Los parámetros del auditor celeste deben ser finitos.")
        if gravitational_constant <= 0.0:
            raise ValueError("G debe ser estrictamente positiva.")
        if collision_radius_m < 0.0:
            raise ValueError("collision_radius_m no puede ser negativo.")
        if min(
            max_energy_drift,
            max_angular_momentum_drift,
            max_reversibility_residual,
        ) <= 0.0:
            raise ValueError("Los límites de deriva deben ser estrictamente positivos.")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps debe ser un entero positivo.")

        self._g = float(gravitational_constant)
        self._collision_radius = float(collision_radius_m)
        self._max_energy_drift = float(max_energy_drift)
        self._max_angular_drift = float(max_angular_momentum_drift)
        self._max_reversibility = float(max_reversibility_residual)
        self._max_steps = int(max_steps)

    def _accelerations(
        self,
        positions: NDArray[np.float64],
        masses: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float]:
        n = positions.shape[0]
        acceleration = np.zeros_like(positions)
        minimum_separation = math.inf

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                for i in range(n):
                    for j in range(i + 1, n):
                        displacement = positions[j] - positions[i]
                        distance = _stable_l2(displacement)
                        if (
                            not math.isfinite(distance)
                            or distance <= max(self._collision_radius, _FLOAT_TINY)
                        ):
                            raise PhotinicCelestialCollisionError(
                                f"Separación singular entre cuerpos {i} y {j}."
                            )
                        minimum_separation = min(minimum_separation, distance)
                        inverse_cube = (1.0 / distance) ** 3
                        acceleration[i] += (
                            self._g * masses[j] * inverse_cube * displacement
                        )
                        acceleration[j] -= (
                            self._g * masses[i] * inverse_cube * displacement
                        )
            except FloatingPointError as exc:
                raise PhotinicCelestialError(
                    "Desbordamiento al calcular aceleraciones Newtonianas."
                ) from exc

        if not np.all(np.isfinite(acceleration)):
            raise PhotinicCelestialError("Aceleración no finita.")
        return acceleration, float(minimum_separation)

    def _leapfrog_step(
        self,
        positions: NDArray[np.float64],
        velocities: NDArray[np.float64],
        masses: NDArray[np.float64],
        dt: float,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float]:
        acceleration_0, min_sep_0 = self._accelerations(positions, masses)
        with np.errstate(over="raise", invalid="raise"):
            try:
                half_velocity = velocities + 0.5 * dt * acceleration_0
                next_positions = positions + dt * half_velocity
                acceleration_1, min_sep_1 = self._accelerations(next_positions, masses)
                next_velocities = half_velocity + 0.5 * dt * acceleration_1
            except FloatingPointError as exc:
                raise PhotinicCelestialError(
                    "Desbordamiento durante el paso leapfrog."
                ) from exc

        if not (
            np.all(np.isfinite(next_positions))
            and np.all(np.isfinite(next_velocities))
        ):
            raise PhotinicCelestialError("El paso leapfrog produjo valores no finitos.")
        return next_positions, next_velocities, min(min_sep_0, min_sep_1)

    def _integrate_raw(
        self,
        positions: NDArray[np.float64],
        velocities: NDArray[np.float64],
        masses: NDArray[np.float64],
        steps: int,
        dt: float,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float]:
        p = positions.copy()
        v = velocities.copy()
        minimum_separation = math.inf
        for _ in range(steps):
            p, v, min_sep = self._leapfrog_step(p, v, masses, dt)
            minimum_separation = min(minimum_separation, min_sep)
        return p, v, float(minimum_separation)

    def _invariants(
        self,
        positions: NDArray[np.float64],
        velocities: NDArray[np.float64],
        masses: NDArray[np.float64],
    ) -> Tuple[float, NDArray[np.float64], float, float, float]:
        kinetic_terms: List[float] = []
        potential_terms: List[float] = []
        angular_components: List[List[float]] = [[], [], []]
        angular_scale_terms: List[float] = []

        n = len(masses)
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                for i in range(n):
                    speed = _stable_l2(velocities[i])
                    radius = _stable_l2(positions[i])
                    kinetic = 0.5 * float(masses[i]) * speed * speed
                    if not math.isfinite(kinetic):
                        raise PhotinicCelestialError("Energía cinética no finita.")
                    kinetic_terms.append(kinetic)

                    angular = float(masses[i]) * np.cross(
                        positions[i], velocities[i]
                    )
                    if not np.all(np.isfinite(angular)):
                        raise PhotinicCelestialError(
                            "Momento angular no finito."
                        )
                    for axis in range(3):
                        angular_components[axis].append(float(angular[axis]))

                    scale_term = float(masses[i]) * radius * speed
                    if not math.isfinite(scale_term):
                        raise PhotinicCelestialError(
                            "Escala del momento angular no finita."
                        )
                    angular_scale_terms.append(scale_term)

                for i in range(n):
                    for j in range(i + 1, n):
                        distance = _stable_l2(positions[j] - positions[i])
                        if (
                            not math.isfinite(distance)
                            or distance <= max(self._collision_radius, _FLOAT_TINY)
                        ):
                            raise PhotinicCelestialCollisionError(
                                f"Separación singular entre cuerpos {i} y {j}."
                            )
                        potential = (
                            -self._g
                            * float(masses[i])
                            * float(masses[j])
                            / distance
                        )
                        if not math.isfinite(potential):
                            raise PhotinicCelestialError(
                                "Energía potencial no finita."
                            )
                        potential_terms.append(potential)
            except FloatingPointError as exc:
                raise PhotinicCelestialError(
                    "Desbordamiento al calcular invariantes."
                ) from exc

        kinetic_total = math.fsum(kinetic_terms)
        potential_total = math.fsum(potential_terms)
        energy = math.fsum((kinetic_total, potential_total))
        angular_momentum = np.array(
            [math.fsum(component) for component in angular_components],
            dtype=np.float64,
        )
        angular_scale = math.fsum(angular_scale_terms)

        if not (
            math.isfinite(energy)
            and np.all(np.isfinite(angular_momentum))
            and math.isfinite(angular_scale)
        ):
            raise PhotinicCelestialError("Invariantes no finitos.")
        return (
            float(energy),
            angular_momentum,
            float(kinetic_total),
            float(potential_total),
            float(angular_scale),
        )

    def integrate_and_audit(
        self,
        state: PoincareCelestialState,
        steps: int,
        step_size_seconds: float,
        section_pair: Tuple[int, int] = (0, 1),
        section_axis: int = 0,
    ) -> PoincareCelestialAudit:
        """Integra, mide invariantes y registra retornos a la sección definida."""
        if not isinstance(state, PoincareCelestialState):
            raise PhotinicCelestialError("state debe ser PoincareCelestialState.")
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise PhotinicCelestialError("steps debe ser un entero.")
        if steps <= 0 or steps > self._max_steps:
            raise PhotinicCelestialError(
                f"steps debe pertenecer a [1, {self._max_steps}]."
            )
        if not math.isfinite(step_size_seconds) or step_size_seconds <= 0.0:
            raise PhotinicCelestialError("El paso temporal debe ser positivo y finito.")
        if (
            not isinstance(section_pair, tuple)
            or len(section_pair) != 2
            or section_pair[0] == section_pair[1]
        ):
            raise PhotinicCelestialError("section_pair debe contener dos índices distintos.")
        i, j = section_pair
        n = len(state.masses_kg)
        if not (0 <= i < n and 0 <= j < n):
            raise PhotinicCelestialError("Índices de sección fuera del rango.")
        if isinstance(section_axis, bool) or section_axis not in (0, 1, 2):
            raise PhotinicCelestialError("section_axis debe ser 0, 1 o 2.")

        p0 = np.array(state.positions_m, copy=True)
        v0 = np.array(state.velocities_m_s, copy=True)
        masses = np.array(state.masses_kg, copy=True)

        energy_0, angular_0, kinetic_0, potential_0, angular_scale_0 = (
            self._invariants(p0, v0, masses)
        )

        p, v = p0.copy(), v0.copy()
        minimum_separation = math.inf
        crossing_times: List[float] = []
        section_points: List[Tuple[float, ...]] = []
        last_crossing_time: Optional[float] = None

        for step_index in range(steps):
            previous_relative = p[j] - p[i]
            previous_x = float(previous_relative[section_axis])

            p, v, min_sep = self._leapfrog_step(
                p, v, masses, float(step_size_seconds)
            )
            minimum_separation = min(minimum_separation, min_sep)

            current_relative = p[j] - p[i]
            current_x = float(current_relative[section_axis])
            relative_velocity = v[j] - v[i]

            # Sección transversal x_j-x_i=0, cruce de negativo a no-negativo
            # y velocidad relativa positiva en el eje escogido.
            if (
                previous_x < 0.0
                and current_x >= 0.0
                and float(relative_velocity[section_axis]) > 0.0
            ):
                denominator = current_x - previous_x
                alpha = (
                    1.0
                    if denominator == 0.0
                    else min(1.0, max(0.0, -previous_x / denominator))
                )
                crossing_time = (
                    step_index + alpha
                ) * float(step_size_seconds)
                crossing_relative = (
                    (1.0 - alpha) * previous_relative + alpha * current_relative
                )
                crossing_velocity = relative_velocity
                point = tuple(
                    float(x)
                    for x in (
                        crossing_relative[(section_axis + 1) % 3],
                        crossing_relative[(section_axis + 2) % 3],
                        crossing_velocity[(section_axis + 1) % 3],
                        crossing_velocity[(section_axis + 2) % 3],
                    )
                )
                crossing_times.append(crossing_time)
                section_points.append(point)
                if last_crossing_time is not None:
                    last_interval = crossing_time - last_crossing_time
                    if last_interval >= 0.0:
                        # Se añade después, al construir la tupla de intervalos.
                        pass
                last_crossing_time = crossing_time

        energy_f, angular_f, _, _, angular_scale_f = self._invariants(p, v, masses)
        energy_scale = max(
            abs(energy_0),
            abs(kinetic_0) + abs(potential_0),
            _FLOAT_TINY,
        )
        relative_energy_drift = abs(energy_f - energy_0) / energy_scale

        angular_scale = max(
            _stable_l2(angular_0),
            _stable_l2(angular_f),
            angular_scale_0,
            angular_scale_f,
            _FLOAT_TINY,
        )
        relative_angular_drift = (
            _stable_l2(angular_f - angular_0) / angular_scale
        )

        p_back, v_back, reverse_min_sep = self._integrate_raw(
            p,
            v,
            masses,
            steps,
            -float(step_size_seconds),
        )
        minimum_separation = min(minimum_separation, reverse_min_sep)

        position_error = _stable_l2((p_back - p0).ravel()) / max(
            _stable_l2(p0.ravel()), _stable_l2(p.ravel()), _FLOAT_TINY
        )
        velocity_error = _stable_l2((v_back - v0).ravel()) / max(
            _stable_l2(v0.ravel()), _stable_l2(v.ravel()), _FLOAT_TINY
        )
        reversibility_residual = max(position_error, velocity_error)

        return_intervals = tuple(
            float(crossing_times[k] - crossing_times[k - 1])
            for k in range(1, len(crossing_times))
        )
        stability_utilization = max(
            relative_energy_drift / self._max_energy_drift,
            relative_angular_drift / self._max_angular_drift,
            reversibility_residual / self._max_reversibility,
        )
        if not math.isfinite(stability_utilization):
            raise PhotinicCelestialError("Utilización de estabilidad no finita.")

        return PoincareCelestialAudit(
            steps=steps,
            step_size_seconds=float(step_size_seconds),
            integration_time_seconds=float(steps * step_size_seconds),
            initial_energy_j=float(energy_0),
            final_energy_j=float(energy_f),
            relative_energy_drift=float(relative_energy_drift),
            relative_angular_momentum_drift=float(relative_angular_drift),
            reversibility_residual=float(reversibility_residual),
            minimum_separation_m=float(minimum_separation),
            section_crossings=len(crossing_times),
            return_intervals_seconds=return_intervals,
            section_points=tuple(section_points),
            stability_utilization=float(stability_utilization),
            stable=bool(stability_utilization <= 1.0),
            integrator="velocity-Verlet / leapfrog kick-drift-kick",
            audit_timestamp=time.time(),
        )


# ------------------------------------------------------------------------------
# EXPEDIENTES INMUTABLES
# ------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PhotinicObservationKernel:
    """Expediente canónico de Fase 1."""

    node_policies: Tuple[NDArray[np.float64], ...]
    majorana_spinor_norm: float
    majorana_chiral_parity: float
    gram_symmetry_defect: float
    banach_ratios: Tuple[float, ...]
    metric_condition_number: float
    spectral_gap: float
    cryptographic_seal: str
    observation_timestamp: float
    recursive_audit: RecursiveAuditEvidence = field(
        default_factory=lambda: RecursiveAuditEvidence(
            maximum_depth=3,
            depth_used=0,
            blas_vs_compensated_residual=0.0,
            permutation_residual=0.0,
            spectral_reconstruction_residual=0.0,
            verified=False,
        )
    )
    zero_policy_count: int = 0
    poincare_celestial_audit: Optional[PoincareCelestialAudit] = None
    gram_matrix: Optional[NDArray[np.float64]] = None


@dataclass(frozen=True, slots=True)
class PhotinicOrientationReport:
    """Reporte de Fase 2; conserva métricas brutas aunque falle la certificación."""

    kernel: PhotinicObservationKernel
    engine_state: Optional[PhotinicEngineState]
    engine_validation_error: Optional[str]
    idempotence_residual: float
    trace_rank_mismatch: float
    choi_min_eigenvalue: float
    choi_trace_residual: float
    choi_subunital_defect_min_eigenvalue: float
    tsirelson_chsh_value: float
    horodecki_eigs: Tuple[float, float]
    horodecki_bipartition_pairs: int
    is_choi_completely_positive: bool
    is_trace_non_increasing: bool
    is_causal_non_signaling: bool
    is_tsirelson_bounded: bool
    is_classical_lhv: bool
    von_neumann_entropy: Optional[float]
    quantum_purity: Optional[float]
    bell_pair_entanglement_entropy: Optional[float]
    cross_engine_majorana_residual: float
    orientation_timestamp: float
    cross_engine_gram_residual: float = 0.0
    poincare_celestial_audit: Optional[PoincareCelestialAudit] = None


@dataclass(frozen=True, slots=True)
class PhotinicAgentCertificate:
    """Certificado de Fase 3. La firma cubre todos sus campos salvo la propia firma."""

    phase: str
    heyting_verdict: str
    heyting_instant_verdict: str
    heyting_hysteresis_verdict: str
    heyting_truth_value: int
    majorana_spinor_norm: Optional[float]
    idempotence_residual: Optional[float]
    trace_rank_mismatch: Optional[float]
    choi_min_eigenvalue: Optional[float]
    tsirelson_chsh_value: Optional[float]
    von_neumann_entropy: Optional[float]
    quantum_purity: Optional[float]
    engine_validation_error: Optional[str]
    is_choi_completely_positive: bool
    is_causal_non_signaling: bool
    is_tsirelson_bounded: bool
    is_soft_veto_active: bool
    override_grace_period_expired: bool
    fock_annihilation_occurred: bool
    fock_transition_probability: float
    hardware_interlock_fired: bool
    crowbar_iram_cycles: int
    crowbar_gate_current_amps: float
    actuation_latency_ns: float
    time_grace_remaining: float
    digital_signature_hmac_sha256: str
    execution_duration_microseconds: float
    total_pipeline_latency_microseconds: float

    choi_trace_residual: Optional[float] = None
    choi_subunital_defect_min_eigenvalue: Optional[float] = None
    is_trace_non_increasing: bool = False
    is_classical_lhv: bool = False
    cross_engine_majorana_residual: Optional[float] = None
    cross_engine_gram_residual: Optional[float] = None
    observation_hmac_sha256: str = ""
    zero_policy_count: int = 0
    recursive_audit_depth: int = 0
    recursive_audit_residual: Optional[float] = None
    recursive_decision_audit_passed: bool = False
    poincare_stable: Optional[bool] = None
    poincare_energy_drift: Optional[float] = None
    poincare_angular_momentum_drift: Optional[float] = None
    poincare_reversibility_residual: Optional[float] = None
    poincare_stability_utilization: Optional[float] = None
    poincare_section_crossings: Optional[int] = None
    hardware_interlock_requested: bool = False
    hardware_dispatch_confirmed: bool = False
    hardware_model_passed: bool = False
    hardware_dispatch_error: Optional[str] = None
    hardware_model_only: bool = False
    signature_key_is_ephemeral: bool = True
    certificate_timestamp: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: INGESTA, BANACH, TENSOR ESPECTRAL Y AUTOMEJORA NIVEL 3
# ══════════════════════════════════════════════════════════════════════════════
class Phase1_PhotinicAgentObserver:
    """
    Observe:
      * valida políticas y canoniza ceros IEEE-754;
      * calcula cocientes de norma ℓ¹/ℓ²;
      * construye el tensor de Gram mediante refinamiento numérico acotado;
      * opcionalmente integra un estado celeste Newtoniano independiente;
      * sella el expediente con HMAC-SHA256.
    """

    def __init__(
        self,
        tolerance: float = 1.0e-12,
        hmac_key: Optional[bytes] = None,
        poincare_mechanics: Optional[PoincareCelestialMechanics] = None,
    ) -> None:
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance debe ser positiva y finita.")
        self._tol: Final[float] = float(tolerance)

        self._key_is_ephemeral: Final[bool] = hmac_key is None
        if hmac_key is None:
            # Clave aleatoria por instancia: evita una clave pública hardcodeada.
            # No permite verificar certificados tras reiniciar el proceso.
            hmac_key = secrets.token_bytes(32)
            logger.warning(
                "Clave HMAC efímera: inyecte una clave secreta gestionada "
                "externamente para un despliegue persistente."
            )
        if not isinstance(hmac_key, bytes) or len(hmac_key) == 0:
            raise ValueError("hmac_key debe ser bytes no vacíos.")
        if len(hmac_key) < 16:
            logger.warning("La clave HMAC tiene menos de 128 bits de material.")

        self._hmac_key: Final[bytes] = bytes(hmac_key)
        self._poincare_mechanics: Final[PoincareCelestialMechanics] = (
            poincare_mechanics
            if poincare_mechanics is not None
            else PoincareCelestialMechanics()
        )

    # ------------------------------------------------------------------
    # 1.1 — HMAC con separación de dominio y encuadre de longitud
    # ------------------------------------------------------------------
    def _compute_hmac_seal(self, *chunks: bytes) -> str:
        mac = hmac.new(
            self._hmac_key,
            b"PhotinicGovernanceSatelliteAgent/v6\x00",
            hashlib.sha256,
        )
        for chunk in chunks:
            if not isinstance(chunk, bytes):
                raise TypeError("Cada fragmento HMAC debe ser bytes.")
            mac.update(len(chunk).to_bytes(8, byteorder="big", signed=False))
            mac.update(chunk)
        return mac.hexdigest()

    # ------------------------------------------------------------------
    # 1.2 — Regularidad de Banach con tratamiento explícito del vector cero
    # ------------------------------------------------------------------
    def evaluate_banach_regularity(self, vector: NDArray[np.float64]) -> float:
        r"""
        Calcula ||x||₁ / ||x||₂. Para x=0 el cociente no está definido y el
        convenio operacional es devolver 0; el número de vectores cero queda
        registrado en el kernel para su clasificación posterior.
        """
        x = np.asarray(vector, dtype=np.float64)
        if x.ndim != 1 or not np.all(np.isfinite(x)):
            raise PhotinicInputValidationError("Se esperaba un vector finito.")
        norm2 = _stable_l2(x)
        if norm2 == 0.0:
            return 0.0
        norm1 = math.fsum(float(abs(v)) for v in x)
        if not math.isfinite(norm1):
            raise PhotinicNumericalAuditError("La norma ℓ¹ no es finita.")
        return float(norm1 / norm2)

    # ------------------------------------------------------------------
    # 1.3 — Tensor espectral y auditoría de simetría
    # ------------------------------------------------------------------
    def _audit_gram_symmetry_defect(
        self,
        gram: NDArray[np.float64],
    ) -> float:
        defect = _relative_residual(gram, gram.T)
        if not math.isfinite(defect):
            raise PhotinicNumericalAuditError("Defecto de simetría no finito.")
        if defect > self._tol:
            logger.warning(
                "Defecto de simetría del Gram=%.6e excede tolerance=%.6e.",
                defect,
                self._tol,
            )
        return float(defect)

    def _compute_spectral_tensor_with_audit(
        self,
        cleaned_policies: Sequence[NDArray[np.float64]],
    ) -> Tuple[
        NDArray[np.float64],
        float,
        float,
        float,
        float,
        float,
        RecursiveAuditEvidence,
    ]:
        if not cleaned_policies:
            raise PhotinicDimensionError("No hay políticas para formar el tensor.")
        p = np.vstack(cleaned_policies)
        (
            gram,
            _eigenvalues,
            majorana_proxy,
            chiral_proxy,
            condition,
            spectral_gap,
            symmetry_defect,
            evidence,
        ) = RecursiveSelfImprovementLevel3.spectral_tensor(
            p, ridge=self._tol, tolerance=self._tol
        )
        return (
            gram,
            majorana_proxy,
            chiral_proxy,
            condition,
            spectral_gap,
            symmetry_defect,
            evidence,
        )

    def compute_majorana_sym_spectral_tensor(
        self,
        cleaned_policies: Sequence[NDArray[np.float64]],
    ) -> Tuple[float, float, float, float, float]:
        """
        Interfaz compacta compatible con la versión previa.

        `majorana_norm` es un proxy espectral de Frobenius, no una medición
        independiente de un espinor físico.
        """
        (
            _gram,
            majorana,
            chiral,
            condition,
            spectral_gap,
            symmetry_defect,
            _evidence,
        ) = self._compute_spectral_tensor_with_audit(cleaned_policies)
        return majorana, chiral, condition, spectral_gap, symmetry_defect

    # ------------------------------------------------------------------
    # 1.4 — Canonización inmutable y sellado del kernel
    # ------------------------------------------------------------------
    def canonize_photinic_observation_kernel(
        self,
        cleaned_policies: Sequence[NDArray[np.float64]],
        banach_ratios: Optional[Sequence[float]] = None,
        seal_material: Optional[Sequence[bytes]] = None,
        poincare_celestial_audit: Optional[PoincareCelestialAudit] = None,
    ) -> PhotinicObservationKernel:
        """Construye el expediente canónico, incorpora evidencia y calcula HMAC."""
        if not cleaned_policies:
            raise PhotinicDimensionError("No hay políticas para canonizar.")

        policies = tuple(_frozen_float_array(v) for v in cleaned_policies)
        dimension = policies[0].shape[0]
        if dimension == 0:
            raise PhotinicDimensionError("La dimensión de una política no puede ser cero.")
        for index, vector in enumerate(policies):
            if vector.ndim != 1 or vector.shape[0] != dimension:
                raise PhotinicDimensionError(
                    f"Dimensión inconsistente en política [{index}]."
                )
            if not np.all(np.isfinite(vector)):
                raise PhotinicInputValidationError(
                    f"Política [{index}] contiene valores no finitos."
                )

        if banach_ratios is None:
            ratios = tuple(self.evaluate_banach_regularity(v) for v in policies)
        else:
            if len(banach_ratios) != len(policies):
                raise PhotinicInputValidationError(
                    "El número de cocientes de Banach no coincide con N."
                )
            ratios = tuple(float(x) for x in banach_ratios)
            if not all(math.isfinite(x) and x >= 0.0 for x in ratios):
                raise PhotinicInputValidationError(
                    "Los cocientes de Banach deben ser finitos y no negativos."
                )

        (
            gram,
            majorana_norm,
            chiral_parity,
            condition,
            spectral_gap,
            symmetry_defect,
            recursive_audit,
        ) = self._compute_spectral_tensor_with_audit(policies)

        frozen_gram = _frozen_float_array(gram)
        zero_count = sum(1 for v in policies if not np.any(v))

        if seal_material is None:
            material = [
                f"{v.shape[0]}:{v.dtype.str}".encode("ascii") + v.tobytes(order="C")
                for v in policies
            ]
        else:
            material = list(seal_material)

        metadata: dict[str, Any] = {
            "schema": "photinic-observation-kernel-v6",
            "node_count": len(policies),
            "dimension": dimension,
            "majorana_proxy": format(majorana_norm, ".17g"),
            "chiral_proxy": format(chiral_parity, ".17g"),
            "condition": format(condition, ".17g"),
            "spectral_gap": format(spectral_gap, ".17g"),
            "symmetry_defect": format(symmetry_defect, ".17g"),
            "banach_ratios": [format(x, ".17g") for x in ratios],
            "zero_policy_count": zero_count,
            "recursive_audit": asdict(recursive_audit),
            "poincare_audit": (
                asdict(poincare_celestial_audit)
                if poincare_celestial_audit is not None
                else None
            ),
        }
        metadata_bytes = json.dumps(
            metadata,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")

        seal = self._compute_hmac_seal(
            b"observation-kernel-v6",
            metadata_bytes,
            *material,
            frozen_gram.tobytes(order="C"),
        )

        return PhotinicObservationKernel(
            node_policies=policies,
            majorana_spinor_norm=float(majorana_norm),
            majorana_chiral_parity=float(chiral_parity),
            gram_symmetry_defect=float(symmetry_defect),
            banach_ratios=ratios,
            metric_condition_number=float(condition),
            spectral_gap=float(spectral_gap),
            cryptographic_seal=seal,
            observation_timestamp=time.time(),
            recursive_audit=recursive_audit,
            zero_policy_count=int(zero_count),
            poincare_celestial_audit=poincare_celestial_audit,
            gram_matrix=frozen_gram,
        )

    # ------------------------------------------------------------------
    # 1.5 — Ingesta pública y puerto terminal hacia Fase 2
    # ------------------------------------------------------------------
    def observe_federated_nodes(
        self,
        node_policies: Sequence[NDArray[np.float64]],
        *,
        celestial_state: Optional[PoincareCelestialState] = None,
        celestial_steps: int = 1,
        celestial_step_seconds: Optional[float] = None,
        celestial_section_pair: Tuple[int, int] = (0, 1),
        celestial_section_axis: int = 0,
    ) -> PhotinicObservationKernel:
        """
        Valida, limpia y canoniza políticas.

        La mecánica celeste es opcional y exige un estado y un paso temporal
        explícitos. No se deriva ningún estado orbital de las políticas.
        """
        try:
            raw_nodes = list(node_policies)
        except TypeError as exc:
            raise PhotinicInputValidationError(
                "node_policies debe ser una secuencia iterable."
            ) from exc

        if not raw_nodes:
            raise PhotinicDimensionError(
                "El consorcio federado no contiene políticas (N = 0)."
            )

        try:
            first = np.asarray(raw_nodes[0], dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PhotinicInputValidationError(
                f"No se pudo convertir la primera política: {exc}"
            ) from exc
        if first.ndim != 1 or first.shape[0] == 0:
            raise PhotinicDimensionError(
                "Cada política debe ser un vector unidimensional no vacío."
            )
        dimension = int(first.shape[0])

        cleaned: List[NDArray[np.float64]] = []
        ratios: List[float] = []
        seal_material: List[bytes] = []

        for index, raw_vector in enumerate(raw_nodes):
            try:
                vector = np.asarray(raw_vector, dtype=np.float64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise PhotinicInputValidationError(
                    f"Política [{index}] no convertible a float64: {exc}"
                ) from exc
            if vector.ndim != 1 or vector.shape[0] != dimension:
                raise PhotinicDimensionError(
                    f"Dimensión inconsistente en política [{index}]: "
                    f"esperada ({dimension},), recibida {vector.shape}."
                )
            if not np.all(np.isfinite(vector)):
                raise PhotinicInputValidationError(
                    f"Política [{index}] contiene valores no finitos."
                )

            # La asignación convierte -0.0 a +0.0 en el buffer copiado.
            canonical = np.array(vector, dtype="<f8", copy=True, order="C")
            canonical[canonical == 0.0] = 0.0
            ratio = self.evaluate_banach_regularity(canonical)
            canonical.setflags(write=False)

            cleaned.append(canonical)
            ratios.append(ratio)
            seal_material.append(
                len(canonical).to_bytes(8, "big", signed=False)
                + canonical.tobytes(order="C")
            )
            seal_material.append(format(ratio, ".17g").encode("ascii"))

        poincare_audit: Optional[PoincareCelestialAudit] = None
        if celestial_state is None:
            if celestial_step_seconds is not None:
                raise PhotinicInputValidationError(
                    "Se indicó celestial_step_seconds sin celestial_state."
                )
        else:
            if celestial_step_seconds is None:
                raise PhotinicInputValidationError(
                    "celestial_state requiere celestial_step_seconds explícito."
                )
            poincare_audit = self._poincare_mechanics.integrate_and_audit(
                state=celestial_state,
                steps=celestial_steps,
                step_size_seconds=float(celestial_step_seconds),
                section_pair=celestial_section_pair,
                section_axis=celestial_section_axis,
            )

        return self.canonize_photinic_observation_kernel(
            cleaned_policies=cleaned,
            banach_ratios=ratios,
            seal_material=seal_material,
            poincare_celestial_audit=poincare_audit,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: ORIENTACIÓN, CONSISTENCIA CRUZADA Y CERTIFICACIÓN ESTRICTA
# ══════════════════════════════════════════════════════════════════════════════
class Phase2_PhotinicAgentOrienter(Phase1_PhotinicAgentObserver):
    """
    Orient:
      1. calcula el reporte bruto del motor;
      2. contrasta invariantes agente↔motor;
      3. ejecuta la certificación estricta en un bloque separado;
      4. conserva el reporte bruto si el motor rechaza la certificación.
    """

    def __init__(
        self,
        tolerance: float = 1.0e-12,
        hmac_key: Optional[bytes] = None,
        engine: Optional[PhotinicGovernanceSatelliteEngine] = None,
        poincare_mechanics: Optional[PoincareCelestialMechanics] = None,
    ) -> None:
        super().__init__(
            tolerance=tolerance,
            hmac_key=hmac_key,
            poincare_mechanics=poincare_mechanics,
        )
        self._engine: Final[PhotinicGovernanceSatelliteEngine] = (
            engine
            if engine is not None
            else PhotinicGovernanceSatelliteEngine(tolerance=tolerance)
        )

    def _audit_cross_engine_consistency(
        self,
        agent_kernel: PhotinicObservationKernel,
        engine_kernel: EngineKernel,
    ) -> float:
        agent_value = float(agent_kernel.majorana_spinor_norm)
        engine_value = float(engine_kernel.majorana_spinor_norm)
        if not (math.isfinite(agent_value) and math.isfinite(engine_value)):
            raise PhotinicMetricConsistencyError(
                "Invariante espectral no finito en una de las implementaciones.",
                relative_residual=math.inf,
            )

        residual = abs(agent_value - engine_value) / max(
            abs(agent_value), _WILKINSON_SAFETY_FLOOR
        )
        if residual > _CROSS_ENGINE_CONSISTENCY_TOLERANCE:
            raise PhotinicMetricConsistencyError(
                "Divergencia del proxy espectral agente↔motor: "
                f"residuo relativo={residual:.4e}.",
                relative_residual=residual,
            )
        return float(residual)

    def _audit_cross_engine_gram(
        self,
        agent_kernel: PhotinicObservationKernel,
        engine_kernel: EngineKernel,
    ) -> float:
        if agent_kernel.gram_matrix is None:
            raise PhotinicMetricConsistencyError(
                "El kernel del agente no contiene matriz de Gram.",
                relative_residual=math.inf,
            )

        agent_gram = np.asarray(agent_kernel.gram_matrix, dtype=np.float64)
        engine_gram = np.asarray(engine_kernel.gram_matrix, dtype=np.float64)
        if agent_gram.shape != engine_gram.shape:
            raise PhotinicMetricConsistencyError(
                "Las matrices de Gram agente↔motor tienen dimensiones distintas.",
                relative_residual=math.inf,
            )
        if not (
            np.all(np.isfinite(agent_gram))
            and np.all(np.isfinite(engine_gram))
        ):
            raise PhotinicMetricConsistencyError(
                "Una matriz de Gram agente↔motor no es finita.",
                relative_residual=math.inf,
            )

        residual = _relative_residual(engine_gram, agent_gram)
        if residual > _CROSS_ENGINE_CONSISTENCY_TOLERANCE:
            raise PhotinicMetricConsistencyError(
                f"Divergencia de Gram agente↔motor: residuo={residual:.4e}.",
                relative_residual=residual,
            )
        return float(residual)

    def orient_photinic_policy_state(
        self,
        kernel: PhotinicObservationKernel,
    ) -> PhotinicOrientationReport:
        """Orienta un kernel, separando explícitamente cálculo y certificación."""
        if not isinstance(kernel, PhotinicObservationKernel):
            raise PhotinicInputValidationError(
                "kernel debe ser PhotinicObservationKernel."
            )

        # El motor recibe copias para impedir que una dependencia mutile el
        # expediente inmutable del agente.
        engine_input = tuple(np.array(v, copy=True) for v in kernel.node_policies)
        engine_kernel = self._engine.canonize_photinic_observation_kernel(engine_input)
        engine_report: EngineReport = self._engine.orient_photinic_policy(engine_kernel)

        cross_majorana = 0.0
        cross_gram = 0.0
        consistency_errors: List[str] = []

        try:
            cross_majorana = self._audit_cross_engine_consistency(
                kernel, engine_kernel
            )
        except PhotinicMetricConsistencyError as exc:
            cross_majorana = exc.relative_residual
            consistency_errors.append(str(exc))

        try:
            cross_gram = self._audit_cross_engine_gram(kernel, engine_kernel)
        except PhotinicMetricConsistencyError as exc:
            cross_gram = exc.relative_residual
            consistency_errors.append(str(exc))

        engine_state: Optional[PhotinicEngineState] = None
        validation_error: Optional[str] = None

        if consistency_errors:
            # El expediente bruto sigue disponible para el clasificador y el
            # interlock; no se confía en una certificación incoherente.
            validation_error = " | ".join(consistency_errors)
        else:
            try:
                engine_state = self._engine.execute_photinic_audit(engine_report)
            except PhotinicEngineError as exc:
                validation_error = str(exc)
                logger.error(
                    "Certificación estricta rechazada; se conserva el reporte "
                    "bruto para decisión fail-closed: %s",
                    validation_error,
                )

        return PhotinicOrientationReport(
            kernel=kernel,
            engine_state=engine_state,
            engine_validation_error=validation_error,
            idempotence_residual=float(engine_report.idempotence_residual),
            trace_rank_mismatch=float(engine_report.trace_rank_mismatch),
            choi_min_eigenvalue=float(engine_report.choi_min_eigenvalue),
            choi_trace_residual=float(engine_report.choi_trace_residual),
            choi_subunital_defect_min_eigenvalue=float(
                engine_report.choi_subunital_defect_min_eigenvalue
            ),
            tsirelson_chsh_value=float(engine_report.tsirelson_chsh_value),
            horodecki_eigs=tuple(
                float(x) for x in engine_report.horodecki_eigs
            ),
            horodecki_bipartition_pairs=int(
                engine_report.horodecki_bipartition_pairs
            ),
            is_choi_completely_positive=bool(
                engine_report.is_choi_completely_positive
            ),
            is_trace_non_increasing=bool(
                engine_report.is_trace_non_increasing
            ),
            is_causal_non_signaling=bool(
                engine_report.is_causal_non_signaling
            ),
            is_tsirelson_bounded=bool(
                engine_report.is_tsirelson_bounded
            ),
            is_classical_lhv=bool(engine_report.is_classical_lhv),
            von_neumann_entropy=(
                _finite_optional(engine_state.von_neumann_entropy)
                if engine_state is not None
                else None
            ),
            quantum_purity=(
                _finite_optional(engine_state.quantum_purity)
                if engine_state is not None
                else None
            ),
            bell_pair_entanglement_entropy=(
                _finite_optional(engine_state.bell_pair_entanglement_entropy)
                if engine_state is not None
                else None
            ),
            cross_engine_majorana_residual=float(cross_majorana),
            orientation_timestamp=time.time(),
            cross_engine_gram_residual=float(cross_gram),
            poincare_celestial_audit=kernel.poincare_celestial_audit,
        )

    def synthesize_photinic_orientation(
        self,
        kernel_or_policies: Union[
            PhotinicObservationKernel,
            Sequence[NDArray[np.float64]],
        ],
        *,
        celestial_state: Optional[PoincareCelestialState] = None,
        celestial_steps: int = 1,
        celestial_step_seconds: Optional[float] = None,
        celestial_section_pair: Tuple[int, int] = (0, 1),
        celestial_section_axis: int = 0,
    ) -> PhotinicOrientationReport:
        """
        Puerto terminal de Fase 2.

        Si se recibe un kernel ya construido, no se permite añadir por separado
        un estado celeste: el expediente firmado debe contenerlo desde Fase 1.
        """
        if isinstance(kernel_or_policies, PhotinicObservationKernel):
            if celestial_state is not None or celestial_step_seconds is not None:
                raise PhotinicInputValidationError(
                    "No se puede adjuntar dinámica celeste a un kernel ya canonizado."
                )
            kernel = kernel_or_policies
        else:
            kernel = self.observe_federated_nodes(
                kernel_or_policies,
                celestial_state=celestial_state,
                celestial_steps=celestial_steps,
                celestial_step_seconds=celestial_step_seconds,
                celestial_section_pair=celestial_section_pair,
                celestial_section_axis=celestial_section_axis,
            )
        return self.orient_photinic_policy_state(kernel)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: HEYTING, HISTÉRESIS, FOCK Y SOLICITUD DE INTERLOCK
# ══════════════════════════════════════════════════════════════════════════════
class Phase3_PhotinicAgentDecider(Phase2_PhotinicAgentOrienter):
    """
    Decide & Act:
      * clasificador Ω₃ con pruebas de monotonicidad de profundidad 3;
      * histéresis bajo RLock;
      * tokens HMAC firmados, frescos y de un solo uso;
      * cálculo RC predictivo y solicitud opcional a un adaptador externo.
    """

    def __init__(
        self,
        tolerance: float = 1.0e-12,
        idempotence_threshold: float = 0.25,
        grace_period_seconds: float = 3600.0,
        hmac_key: Optional[bytes] = None,
        authorized_tokens: Optional[Set[str]] = None,
        engine: Optional[PhotinicGovernanceSatelliteEngine] = None,
        poincare_mechanics: Optional[PoincareCelestialMechanics] = None,
        hardware_dispatcher: Optional[
            Callable[[int, float, float], bool]
        ] = None,
    ) -> None:
        super().__init__(
            tolerance=tolerance,
            hmac_key=hmac_key,
            engine=engine,
            poincare_mechanics=poincare_mechanics,
        )

        if not math.isfinite(idempotence_threshold) or idempotence_threshold <= 0.0:
            raise ValueError("idempotence_threshold debe ser positivo y finito.")
        if not math.isfinite(grace_period_seconds) or grace_period_seconds <= 0.0:
            raise ValueError("grace_period_seconds debe ser positivo y finito.")
        if hardware_dispatcher is not None and not callable(hardware_dispatcher):
            raise ValueError("hardware_dispatcher debe ser invocable.")

        self._idempotence_threshold: Final[float] = float(
            idempotence_threshold
        )
        self._grace_limit: Final[float] = float(grace_period_seconds)
        self._hardware_dispatcher = hardware_dispatcher

        if authorized_tokens is None:
            logger.warning(
                "`authorized_tokens` se interpreta como lista de principals "
                "autorizados, no como bearer tokens. Configure principals y "
                "gestione la clave HMAC externamente."
            )
            authorized_tokens = {"governance-operator", "emergency-override"}

        if any(not isinstance(x, str) or not x for x in authorized_tokens):
            raise ValueError("Cada principal autorizado debe ser un string no vacío.")
        self._authorized_principals: Final[frozenset[str]] = frozenset(
            authorized_tokens
        )

        self._state_lock: Final[threading.RLock] = threading.RLock()
        self._soft_veto_timestamp: Optional[float] = None
        self._is_soft_veto_active = False
        self._last_control_time: Optional[float] = None

        # El registro antirrepetición persiste aunque se reinicie la histéresis.
        self._consumed_override_digests: dict[str, float] = {}

    # ------------------------------------------------------------------
    # 3.1 — Estado soberano e historial antirrepetición
    # ------------------------------------------------------------------
    def reset_sovereign_state(self, clear_replay_cache: bool = False) -> None:
        """
        Reinicia histéresis de forma atómica.

        El caché anti-replay no se borra por defecto: borrarlo puede rehabilitar
        tokens aún frescos. `clear_replay_cache=True` se reserva para rotación
        controlada o pruebas.
        """
        with self._state_lock:
            self._soft_veto_timestamp = None
            self._is_soft_veto_active = False
            self._last_control_time = None
            if clear_replay_cache:
                logger.warning("Se solicitó vaciado explícito del caché anti-replay.")
                self._consumed_override_digests.clear()

    # ------------------------------------------------------------------
    # 3.2 — Clasificador puro y autoverificación monotónica de nivel 3
    # ------------------------------------------------------------------
    @staticmethod
    def _orientation_numbers_finite(report: PhotinicOrientationReport) -> bool:
        values = [
            report.idempotence_residual,
            report.trace_rank_mismatch,
            report.choi_min_eigenvalue,
            report.choi_trace_residual,
            report.choi_subunital_defect_min_eigenvalue,
            report.tsirelson_chsh_value,
            report.cross_engine_majorana_residual,
            report.cross_engine_gram_residual,
            report.kernel.majorana_spinor_norm,
            report.kernel.majorana_chiral_parity,
            report.kernel.gram_symmetry_defect,
            report.kernel.metric_condition_number,
            report.kernel.spectral_gap,
            *report.horodecki_eigs,
            *report.kernel.banach_ratios,
        ]
        if not all(math.isfinite(float(x)) for x in values):
            return False

        if report.kernel.gram_matrix is None:
            return False
        if not np.all(np.isfinite(report.kernel.gram_matrix)):
            return False
        for vector in report.kernel.node_policies:
            if not np.all(np.isfinite(vector)):
                return False

        optional_scalars = (
            report.von_neumann_entropy,
            report.quantum_purity,
            report.bell_pair_entanglement_entropy,
        )
        if any(
            value is not None and not math.isfinite(float(value))
            for value in optional_scalars
        ):
            return False

        if report.engine_state is not None:
            try:
                if not np.all(
                    np.isfinite(report.engine_state.federated_density_matrix)
                ):
                    return False
            except (AttributeError, TypeError):
                return False

        audit = report.poincare_celestial_audit
        if audit is not None:
            celestial_values = (
                audit.relative_energy_drift,
                audit.relative_angular_momentum_drift,
                audit.reversibility_residual,
                audit.minimum_separation_m,
                audit.stability_utilization,
            )
            if not all(math.isfinite(float(x)) for x in celestial_values):
                return False

        recursive = report.kernel.recursive_audit
        if recursive is None:
            return False
        recursive_values = (
            recursive.blas_vs_compensated_residual,
            recursive.permutation_residual,
            recursive.spectral_reconstruction_residual,
        )
        return all(math.isfinite(float(x)) for x in recursive_values)

    def _classify_heyting_raw(
        self,
        report: PhotinicOrientationReport,
    ) -> Tuple[PhotinicHeytingVerdict, bool, bool]:
        """Clasificación sin histéresis ni efectos laterales."""
        if not self._orientation_numbers_finite(report):
            return PhotinicHeytingVerdict.VETOED, False, True

        idem = float(report.idempotence_residual)
        limit = self._idempotence_threshold
        chsh = abs(float(report.tsirelson_chsh_value))
        poincare = report.poincare_celestial_audit
        recursive = report.kernel.recursive_audit
        node_count = len(report.kernel.node_policies)

        is_hard_veto = (
            report.engine_validation_error is not None
            or report.engine_state is None
            or not report.is_choi_completely_positive
            or not report.is_trace_non_increasing
            or not report.is_causal_non_signaling
            or not report.is_tsirelson_bounded
            or chsh > _TSIRELSON_BOUND + _TSIRELSON_NUMERICAL_MARGIN
            or report.choi_min_eigenvalue < _HARD_VETO_CHOI_MIN_EIGENVALUE
            or abs(report.choi_trace_residual) > _HARD_VETO_TRACE_RESIDUAL
            or report.choi_subunital_defect_min_eigenvalue
            < _HARD_VETO_CHOI_MIN_EIGENVALUE
            or abs(report.trace_rank_mismatch) > _HARD_VETO_TRACE_RESIDUAL
            or idem < 0.0
            or idem > _HARD_VETO_IDEMPOTENCE_FRACTION * limit
            or report.kernel.metric_condition_number < 1.0
            or report.kernel.metric_condition_number
            > _HARD_VETO_MAX_CONDITION_NUMBER
            or report.kernel.gram_symmetry_defect
            > _HARD_VETO_GRAM_SYMMETRY_DEFECT
            or report.cross_engine_majorana_residual
            > _CROSS_ENGINE_CONSISTENCY_TOLERANCE
            or report.cross_engine_gram_residual
            > _CROSS_ENGINE_CONSISTENCY_TOLERANCE
            or not recursive.verified
            or node_count == 0
            or report.kernel.zero_policy_count >= node_count
            or (poincare is not None and not poincare.stable)
        )

        if is_hard_veto:
            return PhotinicHeytingVerdict.VETOED, False, True

        idem_soft = (
            _SOFT_VETO_IDEMPOTENCE_LOWER_FRACTION * limit
            < idem
            <= _SOFT_VETO_IDEMPOTENCE_UPPER_FRACTION * limit
        )
        chsh_soft = (
            chsh > _SOFT_VETO_CHSH_CLASSICAL_THRESHOLD
            or not report.is_classical_lhv
        )
        partial_zero_policy = report.kernel.zero_policy_count > 0
        poincare_near_limit = (
            poincare is not None and poincare.stability_utilization >= 0.80
        )

        is_soft_veto = (
            idem_soft
            or chsh_soft
            or partial_zero_policy
            or poincare_near_limit
        )
        if is_soft_veto:
            return PhotinicHeytingVerdict.DEGRADED, True, False
        return PhotinicHeytingVerdict.COHERENT, False, False

    def _recursive_decision_audit(
        self,
        report: PhotinicOrientationReport,
        baseline: PhotinicHeytingVerdict,
    ) -> bool:
        """
        Tres pruebas de empeoramiento monotónico.

        Un aumento del residuo de idempotencia, una caída de Choi o una violación
        explícita de Tsirelson no deben mejorar el veredicto. Si el reporte no es
        finito, se exige que la ruta fail-closed ya haya producido VETOED.
        """
        if not HeytingOmega3Algebra.verify_lattice_laws():
            return False
        if not self._orientation_numbers_finite(report):
            return baseline == PhotinicHeytingVerdict.VETOED

        delta = max(
            self._tol,
            64.0 * _MACHINE_EPS
            * max(1.0, abs(self._idempotence_threshold)),
        )
        hard_idem = (
            _HARD_VETO_IDEMPOTENCE_FRACTION
            * self._idempotence_threshold
        )

        probes = (
            replace(
                report,
                idempotence_residual=max(
                    report.idempotence_residual, hard_idem
                ) + delta,
            ),
            replace(
                report,
                choi_min_eigenvalue=(
                    _HARD_VETO_CHOI_MIN_EIGENVALUE - delta
                ),
            ),
            replace(
                report,
                tsirelson_chsh_value=max(
                    abs(report.tsirelson_chsh_value),
                    _TSIRELSON_BOUND + delta,
                ),
                is_tsirelson_bounded=False,
            ),
        )

        # La secuencia es fija: profundidad 1, 2 y 3.
        for probe in probes:
            worsened, _, _ = self._classify_heyting_raw(probe)
            if int(worsened) > int(baseline):
                return False
        return True

    def _evaluate_classifier_with_audit(
        self,
        report: PhotinicOrientationReport,
    ) -> Tuple[PhotinicHeytingVerdict, bool, bool, bool]:
        baseline, soft, hard = self._classify_heyting_raw(report)
        audit_passed = self._recursive_decision_audit(report, baseline)
        if not audit_passed:
            logger.critical(
                "Falló la autoverificación nivel 3 del clasificador; "
                "se fuerza VETOED."
            )
            return PhotinicHeytingVerdict.VETOED, False, True, False
        return baseline, soft, hard, True

    def evaluate_heyting_topos_subobject_classifier(
        self,
        report: PhotinicOrientationReport,
    ) -> Tuple[PhotinicHeytingVerdict, bool, bool]:
        """API pública del clasificador instantáneo, sin mutación de estado."""
        verdict, soft, hard, _audit_passed = (
            self._evaluate_classifier_with_audit(report)
        )
        return verdict, soft, hard

    # ------------------------------------------------------------------
    # 3.3 — Tokens HMAC de Fock con frescura, principal y antirrepetición
    # ------------------------------------------------------------------
    def create_override_token(
        self,
        principal: str,
        issued_at: Optional[float] = None,
        nonce: Optional[str] = None,
    ) -> str:
        """
        Crea un token firmado para integración/pruebas.

        En producción el emisor debe ser un servicio separado que controle la
        clave; no se debe exponer este método como firmante a los solicitantes.
        """
        if principal not in self._authorized_principals:
            raise PhotinicTokenFreshnessError("Principal no autorizado.")
        timestamp = float(time.time() if issued_at is None else issued_at)
        if not math.isfinite(timestamp):
            raise PhotinicTokenFreshnessError("Timestamp de emisión no finito.")

        token_nonce = nonce or secrets.token_urlsafe(18)
        if (
            not token_nonce
            or len(token_nonce) > 128
            or "|" in token_nonce
            or ":" in token_nonce
        ):
            raise PhotinicTokenFreshnessError("Nonce de token no válido.")

        payload = f"{principal}|{timestamp:.6f}|{token_nonce}"
        signature = hmac.new(
            self._hmac_key,
            payload.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"{payload}:{signature}"

    def evaluate_quantum_fock_annihilation(
        self,
        token: Optional[str],
        idempotence_residual: float,
    ) -> Tuple[bool, float]:
        """
        Verifica token HMAC firmado, principal permitido, ventana temporal y uso
        único. Los antiguos bearer tokens en texto plano ya no son aceptados.
        """
        if token is None:
            return False, 0.0
        try:
            residual = float(idempotence_residual)
        except (TypeError, ValueError, OverflowError):
            return False, 0.0
        if not math.isfinite(residual) or residual < 0.0:
            return False, 0.0

        token_string = str(token).strip()
        if not token_string or len(token_string) > 4096:
            return False, 0.0

        with self._state_lock:
            if not self._is_soft_veto_active:
                return False, 0.0

            if ":" not in token_string:
                return False, 0.0
            payload, signature = token_string.rsplit(":", 1)
            parts = payload.split("|")
            if len(parts) != 3:
                return False, 0.0

            principal, timestamp_text, nonce = parts
            if (
                principal not in self._authorized_principals
                or not nonce
                or len(nonce) > 128
            ):
                return False, 0.0

            expected = hmac.new(
                self._hmac_key,
                payload.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            if len(signature) != 64 or not hmac.compare_digest(
                signature.lower(), expected.lower()
            ):
                return False, 0.0

            try:
                issued_at = float(timestamp_text)
            except ValueError as exc:
                raise PhotinicTokenFreshnessError(
                    "Timestamp firmado no interpretable."
                ) from exc
            if not math.isfinite(issued_at):
                raise PhotinicTokenFreshnessError(
                    "Timestamp firmado no finito."
                )

            now = time.time()
            age = now - issued_at
            if age > _TOKEN_FRESHNESS_WINDOW_SECONDS:
                raise PhotinicTokenFreshnessError(
                    f"Token expirado: edad={age:.1f}s."
                )
            if age < -_TOKEN_MAX_FUTURE_SKEW_SECONDS:
                raise PhotinicTokenFreshnessError(
                    f"Token emitido en el futuro: desfase={-age:.1f}s."
                )

            # Poda de entradas que ya no pueden volver a ser válidas.
            expiration_cutoff = now - (
                _TOKEN_FRESHNESS_WINDOW_SECONDS
                + _TOKEN_MAX_FUTURE_SKEW_SECONDS
            )
            stale_digests = [
                digest
                for digest, consumed_at in self._consumed_override_digests.items()
                if consumed_at < expiration_cutoff
            ]
            for digest in stale_digests:
                del self._consumed_override_digests[digest]

            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            if digest in self._consumed_override_digests:
                raise PhotinicTokenFreshnessError(
                    "Token ya consumido; replay rechazado."
                )
            self._consumed_override_digests[digest] = now

        # Índice fenomenológico acotado, no una probabilidad física calibrada.
        exponent = min(
            residual / _FOCK_ANNIHILATION_CROSS_SECTION_SIGMA,
            745.0,
        )
        transition_score = float(-math.expm1(-exponent))
        return True, transition_score

    # ------------------------------------------------------------------
    # 3.4 — Modelo RC predictivo del Crowbar
    # ------------------------------------------------------------------
    def simulate_bt151_crowbar_iram_discharge(
        self,
    ) -> Tuple[bool, int, float, float]:
        """
        Calcula la condición de disparo y una latencia RC estimada.

        La fórmula supone un driver externo de compuerta con parámetros
        especificados en las constantes. No mide un tiristor real ni constituye
        una garantía de tiempo real.
        """
        voltage_headroom = _GATE_DRIVER_VOH_VOLTS - _BT151_VGT_VOLTS
        if voltage_headroom <= 0.0:
            raise PhotinicCrowbarMisfireError(
                "La tensión del driver no supera V_GT."
            )

        gate_current = voltage_headroom / _CROWBAR_GATE_RESISTOR_OHMS
        required_current = _CROWBAR_MIN_SAFETY_MARGIN * _BT151_IGT_AMPS

        if gate_current < required_current:
            raise PhotinicCrowbarMisfireError(
                f"I_G={gate_current * 1e3:.2f} mA < "
                f"{required_current * 1e3:.2f} mA requeridos."
            )
        if gate_current > _GATE_DRIVER_MAX_SOURCE_CURRENT_AMPS:
            raise PhotinicCrowbarMisfireError(
                "La corriente calculada excede el máximo declarado del driver."
            )

        rc_tau = (
            _CROWBAR_GATE_RESISTOR_OHMS
            * _CROWBAR_GATE_CAPACITANCE_FARADS
        )
        t_rise_ns = (
            rc_tau
            * math.log(
                _GATE_DRIVER_VOH_VOLTS
                / (_GATE_DRIVER_VOH_VOLTS - _BT151_VGT_VOLTS)
            )
            * 1.0e9
        )
        isr_ns = _ISR_DISPATCH_OVERHEAD_CYCLES * _ESP32_CYCLE_TIME_NS
        total_latency_ns = isr_ns + t_rise_ns

        if total_latency_ns > _CROWBAR_MAX_MODEL_LATENCY_NS:
            raise PhotinicCrowbarMisfireError(
                f"Latencia estimada={total_latency_ns:.2f} ns supera "
                f"{_CROWBAR_MAX_MODEL_LATENCY_NS:.0f} ns."
            )

        cycles = int(math.ceil(total_latency_ns / _ESP32_CYCLE_TIME_NS))
        return True, cycles, float(total_latency_ns), float(gate_current)

    # ------------------------------------------------------------------
    # 3.5 — Composición algebraica, histéresis y certificado firmado
    # ------------------------------------------------------------------
    def decide_heyting_verdict(
        self,
        orientation_report: PhotinicOrientationReport,
        override_token: Optional[str] = None,
        current_time: Optional[float] = None,
        simulate_grace_expired: bool = False,
        _pipeline_start_perf: Optional[float] = None,
    ) -> PhotinicAgentCertificate:
        """
        Veredicto final = meet(retícula) del clasificador instantáneo y el estado
        de histéresis. `current_time`, si se inyecta, debe pertenecer al mismo
        dominio monotónico que las llamadas previas al agente.
        """
        if not isinstance(orientation_report, PhotinicOrientationReport):
            raise PhotinicInputValidationError(
                "orientation_report debe ser PhotinicOrientationReport."
            )

        start_perf = time.perf_counter()
        try:
            control_time = float(
                time.monotonic() if current_time is None else current_time
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise PhotinicInputValidationError(
                "current_time no es un tiempo interpretable."
            ) from exc
        if not math.isfinite(control_time):
            raise PhotinicInputValidationError("current_time debe ser finito.")

        instant, _is_soft, _is_hard, decision_audit_passed = (
            self._evaluate_classifier_with_audit(orientation_report)
        )

        report = orientation_report
        hysteresis = PhotinicHeytingVerdict.COHERENT
        fock_annihilated = False
        fock_probability = 0.0
        time_remaining = 0.0
        grace_expired = False

        with self._state_lock:
            clock_rollback = (
                self._last_control_time is not None
                and control_time < self._last_control_time
            )
            self._last_control_time = control_time

            if instant == PhotinicHeytingVerdict.VETOED:
                hysteresis = PhotinicHeytingVerdict.VETOED
                self._is_soft_veto_active = False
                self._soft_veto_timestamp = None
                logger.critical(
                    "VETO DURO: CHSH=%s, Choi_min=%s, engine_error=%s",
                    report.tsirelson_chsh_value,
                    report.choi_min_eigenvalue,
                    report.engine_validation_error,
                )

            elif instant == PhotinicHeytingVerdict.DEGRADED:
                if clock_rollback or simulate_grace_expired:
                    hysteresis = PhotinicHeytingVerdict.VETOED
                    self._is_soft_veto_active = False
                    self._soft_veto_timestamp = None
                    grace_expired = True
                    logger.critical(
                        "La ventana de gracia se fuerza a expirar por "
                        "solicitud explícita o retroceso del reloj."
                    )
                elif not self._is_soft_veto_active:
                    self._is_soft_veto_active = True
                    self._soft_veto_timestamp = control_time
                    time_remaining = self._grace_limit
                    hysteresis = PhotinicHeytingVerdict.DEGRADED
                    logger.warning(
                        "VETO SUAVE activado: idempotencia=%.4e, CHSH=%.4f",
                        report.idempotence_residual,
                        report.tsirelson_chsh_value,
                    )
                else:
                    started_at = self._soft_veto_timestamp
                    if started_at is None or control_time < started_at:
                        hysteresis = PhotinicHeytingVerdict.VETOED
                        self._is_soft_veto_active = False
                        self._soft_veto_timestamp = None
                        grace_expired = True
                    else:
                        elapsed = control_time - started_at
                        time_remaining = max(0.0, self._grace_limit - elapsed)
                        if time_remaining <= self._tol:
                            hysteresis = PhotinicHeytingVerdict.VETOED
                            self._is_soft_veto_active = False
                            self._soft_veto_timestamp = None
                            grace_expired = True
                            logger.critical("Ventana de gracia expirada.")
                        else:
                            hysteresis = PhotinicHeytingVerdict.DEGRADED

            else:
                self._is_soft_veto_active = False
                self._soft_veto_timestamp = None
                hysteresis = PhotinicHeytingVerdict.COHERENT

            verdict = HeytingOmega3Algebra.meet(instant, hysteresis)

            if self._is_soft_veto_active and override_token is not None:
                try:
                    fock_annihilated, fock_probability = (
                        self.evaluate_quantum_fock_annihilation(
                            override_token,
                            report.idempotence_residual,
                        )
                    )
                except PhotinicTokenFreshnessError as exc:
                    logger.error("Token de Fock rechazado: %s", exc)
                    fock_annihilated = False
                    fock_probability = 0.0

                if fock_annihilated:
                    # La sutura puede mantener DEGRADED, nunca convertirlo en
                    # COHERENT ni anular un veto duro.
                    verdict = PhotinicHeytingVerdict.DEGRADED
                    hysteresis = PhotinicHeytingVerdict.DEGRADED
                    self._is_soft_veto_active = False
                    self._soft_veto_timestamp = None
                    time_remaining = 0.0
                    logger.info(
                        "Token Fock aceptado; índice fenomenológico=%.6f.",
                        fock_probability,
                    )

            soft_active_snapshot = self._is_soft_veto_active

        # La predicción RC no equivale a una activación física.
        interlock_requested = verdict == PhotinicHeytingVerdict.VETOED
        model_passed = False
        interlock_fired = False
        dispatch_confirmed = False
        dispatch_error: Optional[str] = None
        iram_cycles = 0
        actuation_latency_ns = 0.0
        gate_current_amps = 0.0

        if interlock_requested:
            try:
                (
                    model_passed,
                    iram_cycles,
                    actuation_latency_ns,
                    gate_current_amps,
                ) = self.simulate_bt151_crowbar_iram_discharge()
            except PhotinicCrowbarMisfireError as exc:
                dispatch_error = str(exc)
                logger.critical(
                    "El modelo RC no valida el disparo; VETOED se mantiene: %s",
                    exc,
                )
            else:
                if self._hardware_dispatcher is not None:
                    try:
                        dispatch_confirmed = bool(
                            self._hardware_dispatcher(
                                iram_cycles,
                                actuation_latency_ns,
                                gate_current_amps,
                            )
                        )
                        interlock_fired = dispatch_confirmed
                        if not dispatch_confirmed:
                            dispatch_error = (
                                "El adaptador externo no confirmó la solicitud."
                            )
                    except Exception as exc:  # el veto se conserva ante fallo externo
                        dispatch_error = (
                            f"Excepción del adaptador de hardware: {exc}"
                        )
                        logger.exception(
                            "Falló el adaptador de hardware; el veredicto "
                            "VETOED permanece vigente."
                        )
                else:
                    logger.critical(
                        "Solicitud lógica de interlock registrada; no hay "
                        "adaptador de hardware configurado."
                    )

        end_perf = time.perf_counter()
        duration_us = (end_perf - start_perf) * 1.0e6
        total_pipeline_us = (
            (end_perf - _pipeline_start_perf) * 1.0e6
            if _pipeline_start_perf is not None
            else duration_us
        )
        certificate_timestamp = time.time()

        recursive = report.kernel.recursive_audit
        celestial = report.poincare_celestial_audit

        unsigned_certificate = PhotinicAgentCertificate(
            phase="G_OMEGA_PHOTINIC_GOVERNANCE_V6",
            heyting_verdict=verdict.canonical_name,
            heyting_instant_verdict=instant.canonical_name,
            heyting_hysteresis_verdict=hysteresis.canonical_name,
            heyting_truth_value=int(verdict.value),
            majorana_spinor_norm=_finite_optional(
                report.kernel.majorana_spinor_norm
            ),
            idempotence_residual=_finite_optional(
                report.idempotence_residual
            ),
            trace_rank_mismatch=_finite_optional(
                report.trace_rank_mismatch
            ),
            choi_min_eigenvalue=_finite_optional(
                report.choi_min_eigenvalue
            ),
            tsirelson_chsh_value=_finite_optional(
                report.tsirelson_chsh_value
            ),
            von_neumann_entropy=_finite_optional(
                report.von_neumann_entropy
            ),
            quantum_purity=_finite_optional(report.quantum_purity),
            engine_validation_error=report.engine_validation_error,
            is_choi_completely_positive=bool(
                report.is_choi_completely_positive
            ),
            is_causal_non_signaling=bool(report.is_causal_non_signaling),
            is_tsirelson_bounded=bool(report.is_tsirelson_bounded),
            is_soft_veto_active=bool(soft_active_snapshot),
            override_grace_period_expired=bool(grace_expired),
            fock_annihilation_occurred=bool(fock_annihilated),
            fock_transition_probability=float(fock_probability),
            hardware_interlock_fired=bool(interlock_fired),
            crowbar_iram_cycles=int(iram_cycles),
            crowbar_gate_current_amps=float(gate_current_amps),
            actuation_latency_ns=float(actuation_latency_ns),
            time_grace_remaining=float(time_remaining),
            digital_signature_hmac_sha256="",
            execution_duration_microseconds=float(duration_us),
            total_pipeline_latency_microseconds=float(total_pipeline_us),
            choi_trace_residual=_finite_optional(report.choi_trace_residual),
            choi_subunital_defect_min_eigenvalue=_finite_optional(
                report.choi_subunital_defect_min_eigenvalue
            ),
            is_trace_non_increasing=bool(report.is_trace_non_increasing),
            is_classical_lhv=bool(report.is_classical_lhv),
            cross_engine_majorana_residual=_finite_optional(
                report.cross_engine_majorana_residual
            ),
            cross_engine_gram_residual=_finite_optional(
                report.cross_engine_gram_residual
            ),
            observation_hmac_sha256=report.kernel.cryptographic_seal,
            zero_policy_count=int(report.kernel.zero_policy_count),
            recursive_audit_depth=int(recursive.depth_used),
            recursive_audit_residual=_finite_optional(
                max(
                    recursive.blas_vs_compensated_residual,
                    recursive.permutation_residual,
                    recursive.spectral_reconstruction_residual,
                )
            ),
            recursive_decision_audit_passed=bool(decision_audit_passed),
            poincare_stable=celestial.stable if celestial is not None else None,
            poincare_energy_drift=(
                _finite_optional(celestial.relative_energy_drift)
                if celestial is not None
                else None
            ),
            poincare_angular_momentum_drift=(
                _finite_optional(celestial.relative_angular_momentum_drift)
                if celestial is not None
                else None
            ),
            poincare_reversibility_residual=(
                _finite_optional(celestial.reversibility_residual)
                if celestial is not None
                else None
            ),
            poincare_stability_utilization=(
                _finite_optional(celestial.stability_utilization)
                if celestial is not None
                else None
            ),
            poincare_section_crossings=(
                celestial.section_crossings if celestial is not None else None
            ),
            hardware_interlock_requested=bool(interlock_requested),
            hardware_dispatch_confirmed=bool(dispatch_confirmed),
            hardware_model_passed=bool(model_passed),
            hardware_dispatch_error=dispatch_error,
            hardware_model_only=bool(interlock_requested and not dispatch_confirmed),
            signature_key_is_ephemeral=bool(self._key_is_ephemeral),
            certificate_timestamp=float(certificate_timestamp),
        )

        signature_payload = asdict(unsigned_certificate)
        signature_payload.pop("digital_signature_hmac_sha256", None)
        serialized = json.dumps(
            signature_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        signature = self._compute_hmac_seal(
            b"agent-certificate-v6",
            serialized,
        )
        return replace(
            unsigned_certificate,
            digital_signature_hmac_sha256=signature,
        )

    # ------------------------------------------------------------------
    # 3.6 — Ciclo OODA completo y puerto público
    # ------------------------------------------------------------------
    def audit_federated_governance_cycle(
        self,
        node_policies: Union[
            PhotinicOrientationReport,
            PhotinicObservationKernel,
            Sequence[NDArray[np.float64]],
        ],
        override_token: Optional[str] = None,
        current_time: Optional[float] = None,
        simulate_grace_expired: bool = False,
        *,
        celestial_state: Optional[PoincareCelestialState] = None,
        celestial_steps: int = 1,
        celestial_step_seconds: Optional[float] = None,
        celestial_section_pair: Tuple[int, int] = (0, 1),
        celestial_section_axis: int = 0,
    ) -> PhotinicAgentCertificate:
        """Ejecuta el ciclo OODA y mide el tiempo total de software."""
        pipeline_start = time.perf_counter()

        if isinstance(node_policies, PhotinicOrientationReport):
            if celestial_state is not None or celestial_step_seconds is not None:
                raise PhotinicInputValidationError(
                    "No se puede adjuntar dinámica a un reporte ya orientado."
                )
            orientation = node_policies
        else:
            orientation = self.synthesize_photinic_orientation(
                node_policies,
                celestial_state=celestial_state,
                celestial_steps=celestial_steps,
                celestial_step_seconds=celestial_step_seconds,
                celestial_section_pair=celestial_section_pair,
                celestial_section_axis=celestial_section_axis,
            )

        return self.decide_heyting_verdict(
            orientation_report=orientation,
            override_token=override_token,
            current_time=current_time,
            simulate_grace_expired=simulate_grace_expired,
            _pipeline_start_perf=pipeline_start,
        )


# ══════════════════════════════════════════════════════════════════════════════
# AGENTE SOBERANO: COMPOSICIÓN DE LAS TRES FASES
# ══════════════════════════════════════════════════════════════════════════════
class PhotinicGovernanceSatelliteAgent(Morphism, Phase3_PhotinicAgentDecider):
    """Composición categórica del pipeline Observe → Orient → Decide."""

    def __init__(
        self,
        tolerance: float = 1.0e-12,
        idempotence_threshold: float = 0.25,
        grace_period_seconds: float = 3600.0,
        hmac_key: Optional[bytes] = None,
        authorized_tokens: Optional[Set[str]] = None,
        engine: Optional[PhotinicGovernanceSatelliteEngine] = None,
        poincare_mechanics: Optional[PoincareCelestialMechanics] = None,
        hardware_dispatcher: Optional[
            Callable[[int, float, float], bool]
        ] = None,
    ) -> None:
        Morphism.__init__(self)
        Phase3_PhotinicAgentDecider.__init__(
            self,
            tolerance=tolerance,
            idempotence_threshold=idempotence_threshold,
            grace_period_seconds=grace_period_seconds,
            hmac_key=hmac_key,
            authorized_tokens=authorized_tokens,
            engine=engine,
            poincare_mechanics=poincare_mechanics,
            hardware_dispatcher=hardware_dispatcher,
        )

        if not HeytingOmega3Algebra.verify_lattice_laws():
            raise PhotinicNumericalAuditError(
                "Falló la verificación de las leyes de Ω₃."
            )

        logger.info(
            "Agente PhotinicGovernanceSatelliteAgent v6 inicializado: "
            "fases Observe/Orient/Decide, refinamiento nivel 3 acotado, "
            "Heyting Ω₃ y auditor Poincaré opcional."
        )


__all__ = [
    "PhotinicGovernanceSatelliteAgent",
    "PhotinicObservationKernel",
    "PhotinicOrientationReport",
    "PhotinicAgentCertificate",
    "PhotinicHeytingVerdict",
    "HeytingOmega3Algebra",
    "RecursiveAuditEvidence",
    "RecursiveSelfImprovementLevel3",
    "PoincareCelestialState",
    "PoincareCelestialAudit",
    "PoincareCelestialMechanics",
    "PhotinicAgentError",
    "PhotinicInputValidationError",
    "PhotinicNumericalAuditError",
    "PhotinicMetricConsistencyError",
    "PhotinicCelestialError",
    "PhotinicCelestialCollisionError",
    "PhotinicCrowbarMisfireError",
    "PhotinicTokenFreshnessError",
    "Phase1_PhotinicAgentObserver",
    "Phase2_PhotinicAgentOrienter",
    "Phase3_PhotinicAgentDecider",
]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("\n" + "═" * 78)
    print(" AUDITORÍA DEL AGENTE FOTÍNICO — DEMOSTRACIÓN, NO CERTIFICACIÓN FÍSICA")
    print("═" * 78)
    print(
        "Ω₃: leyes de Heyting =",
        HeytingOmega3Algebra.verify_lattice_laws(),
    )
    print(
        "Ω₃: ¬¬DEGRADED ≠ DEGRADED =",
        HeytingOmega3Algebra.verify_intuitionistic_failure(),
    )

    # En producción, inyectar una clave gestionada externamente y un motor real.
    agent = PhotinicGovernanceSatelliteAgent(
        hmac_key=secrets.token_bytes(32),
        authorized_tokens={"demo-operator"},
    )
    rng = np.random.default_rng(7)

    # Amplitud moderada para no inducir una violación sintética de la cota CHSH
    # del motor de fallback.
    nominal_policies = [
        rng.normal(loc=0.0, scale=0.1, size=6)
        for _ in range(5)
    ]
    nominal = agent.audit_federated_governance_cycle(nominal_policies)
    print("\n[+] Caso nominal")
    print("    Veredicto:", nominal.heyting_verdict)
    print("    Proxy espectral:", nominal.majorana_spinor_norm)
    print("    Residuo recursivo:", nominal.recursive_audit_residual)
    print("    Firma HMAC:", nominal.digital_signature_hmac_sha256[:24] + "...")
    print("    Confirmación física de interlock:", nominal.hardware_interlock_fired)

    # Construcción deliberada de una región CHSH degradada para el fallback.
    base = np.array([0.85, 0.75, 0.10, 0.00, 0.05, -0.02], dtype=np.float64)
    correlated_policies = [
        base * (1.0 + 0.01 * i)
        for i in range(6)
    ]
    token = agent.create_override_token("demo-operator")
    amber = agent.audit_federated_governance_cycle(
        correlated_policies,
        override_token=token,
    )
    print("\n[+] Caso degradado con token firmado")
    print("    Veredicto:", amber.heyting_verdict)
    print("    Sutura aceptada:", amber.fock_annihilation_occurred)
    print("    Score fenomenológico:", amber.fock_transition_probability)

    veto = agent.audit_federated_governance_cycle(
        correlated_policies,
        simulate_grace_expired=True,
    )
    print("\n[+] Caso de veto y modelo RC")
    print("    Veredicto:", veto.heyting_verdict)
    print("    Solicitud lógica:", veto.hardware_interlock_requested)
    print("    Modelo RC válido:", veto.hardware_model_passed)
    print("    Latencia RC estimada (ns):", veto.actuation_latency_ns)
    print("    Adaptador confirmó:", veto.hardware_dispatch_confirmed)
    print("    Solo modelo, sin hardware confirmado:", veto.hardware_model_only)

    agent.reset_sovereign_state()
    print("\nAuditoría de demostración completada.")