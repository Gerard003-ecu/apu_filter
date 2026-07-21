# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Utils Agent (Custodio de la Frontera Termodinámica)                 ║
║ Ruta   : app/agents/core/utils_agent.py                                      ║
║ Versión: 2.0.0-Topological-FPU-Boundary-Strict                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GEOMETRÍA DIFERENCIAL (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor gobierna al módulo `utils.py` (Capa Límite Termodinámica).

Actúa como filtro absoluto de entropía sintáctica y proyector métrico de entrada,
garantizando que ninguna singularidad en R_FPU o desgarro topológico del sistema
de archivos penetre hacia los estratos superiores de la Malla Agéntica.

ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Certificación de Dominio Formal:
    Combina retractos de deformación idempotentes y proyección FPU IEEE 754.

        |f(f(x)) - f(x)| ≤ ε_mach,
        x ∈ R ⇔ |x| < ∞ ∧ x ≠ NaN.

Fase 2 → Filtración de Variedad Estadística (MAD):
    Aplica geometría de outliers mediante Z-Score modificado:

        M_i = 0.6745 · |x_i - x̃| / MAD ≤ τ_critical.

Fase 3 → Difeomorfismo de Frontera I/O:
    Verifica que el grafo de inodos sea acíclico y que la ruta resuelta sea
    absoluta, existente y topológicamente segura.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Optional, Union

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter (Stubs de aislamiento)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:

    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos MIC."""
        pass

    class Morphism:
        """Clase base de Morfismos del Topos."""
        pass


logger = logging.getLogger("MIC.Core.UtilsAgent")

if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# ═══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES FÍSICAS, NUMÉRICAS Y ESTADÍSTICAS
# ═══════════════════════════════════════════════════════════════════════════════

_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)

# Constante de escala para consistencia asintótica Gaussiana.
_MAD_CONSTANT: Final[float] = 0.6745

# τ_critical para extirpación de anomalías.
_TAU_CRITICAL_ZSCORE: Final[float] = 3.5

# Límite topológico para profundidad de rutas / bucles de inodos.
_MAX_SYMLINK_DEPTH: Final[int] = 40


# ═══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES TERMODINÁMICAS DE FRONTERA
# ═══════════════════════════════════════════════════════════════════════════════

class UtilsAgentError(TopologicalInvariantError):
    """Excepción raíz del Custodio de la Frontera Termodinámica."""
    pass


class DomainIntegrityViolationError(UtilsAgentError):
    """Detonada cuando un input viola su contrato formal de dominio."""
    pass


class IdempotencyViolationError(UtilsAgentError):
    r"""
    Detonada si f(f(x)) ≠ f(x).

    El normalizador induce oscilación armónica o mutación parásita.
    """
    pass


class NumericSingularityVeto(UtilsAgentError):
    r"""
    Detonada si x → ±∞ o x = NaN.

    Singularidad que colapsaría integradores LTI o proyectores métricos.
    """
    pass


class StatisticalManifoldDeformationVeto(UtilsAgentError):
    r"""
    Detonada si la métrica de dispersión acusa degeneración y la variedad se
    desgarra irreparablemente.
    """
    pass


class IOBoundaryTopologyVeto(UtilsAgentError):
    r"""
    Detonada si el espacio de archivos diverge en un ciclo homológico infinito
    o si la frontera I/O no puede resolverse como variedad absoluta.
    """
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Espacio Cociente)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class IdempotenceAuditData:
    r"""Artefacto de Fase 1. Certificado de proyección idempotente."""
    is_idempotent: bool
    residual_norm: float
    input_type: str = "unknown"
    projection_type: str = "categorical"
    idempotence_tolerance: float = 0.0


@dataclass(frozen=True, slots=True)
class FPUProjectionData:
    r"""Artefacto de Fase 1. Certificado de clausura en R."""
    is_finite: bool
    validated_scalar: float
    numeric_tolerance: float = 0.0


@dataclass(frozen=True, slots=True)
class StatisticalFiltrationData:
    r"""Artefacto de Fase 2. Certificado de filtración isométrica."""
    filtered_tensor: NDArray[np.float64]
    extirpated_count: int
    manifold_median: float
    manifold_mad: float
    tau_critical: float = _TAU_CRITICAL_ZSCORE
    max_modified_z_score: float = 0.0
    dispersion_scale_used: float = 0.0


@dataclass(frozen=True, slots=True)
class IOBoundaryDiffeomorphismData:
    r"""Artefacto de Fase 3. Certificado de difeomorfismo de inodos."""
    resolved_absolute_path: str
    is_acyclic_mapping: bool
    inode_depth: int = 0
    is_absolute_path: bool = True


@dataclass(frozen=True, slots=True)
class Phase1DomainHandoff:
    r"""
    Handoff formal de Fase 1 → Fase 2.

    Este objeto es la continuación material de la certificación de dominio y
    el prefijo obligatorio de la filtración estadística.
    """
    idempotence_audit: Optional[IdempotenceAuditData]
    fpu_audit: Optional[FPUProjectionData]
    has_domain_payload: bool


@dataclass(frozen=True, slots=True)
class Phase2StatisticalHandoff:
    r"""
    Handoff formal de Fase 2 → Fase 3.

    Este objeto transporta los certificados de dominio y la auditoría
    estadística como prefijo obligatorio del difeomorfismo I/O.
    """
    phase1_handoff: Phase1DomainHandoff
    filtration_audit: Optional[StatisticalFiltrationData]


@dataclass(frozen=True, slots=True)
class ThermodynamicBoundaryState:
    r"""Objeto final del endofuntor Z_Utils."""
    idempotence_audit: Optional[IdempotenceAuditData]
    fpu_audit: Optional[FPUProjectionData]
    filtration_audit: Optional[StatisticalFiltrationData]
    io_audit: Optional[IOBoundaryDiffeomorphismData]
    is_epistemologically_valid: bool


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1: CERTIFICACIÓN DE DOMINIO FORMAL                                 ║
# ║   Retracto de deformación idempotente y proyección FPU IEEE 754.          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase1_DeformationRetractAndFPUProjector:
    r"""
    Garantiza que las transformaciones de normalización sean proyectores
    idempotentes y que los escalares pertenezcan al cuerpo real válido.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 1.1. Tolerancia adaptativa
    # ─────────────────────────────────────────────────────────────────────────
    def _adaptive_tolerance(
        self,
        base_tolerance: float,
        reference: Any,
    ) -> float:
        r"""
        Construye una tolerancia numéricamente consciente:

            tol = max(tol_base, κ · ε_máquina · tamaño · escala)
        """
        if isinstance(reference, np.ndarray):
            size = max(1, int(reference.size))
            if reference.size == 0:
                scale = 1.0
            else:
                scale = max(
                    1.0,
                    float(la.norm(reference.ravel(), ord=np.inf)),
                )
        else:
            size = 1
            try:
                scale = max(1.0, abs(float(reference)))
            except (TypeError, ValueError):
                scale = 1.0

        return max(
            float(base_tolerance),
            10.0 * _MACHINE_EPSILON * size * scale,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1.2. Detección de proyección numérica real
    # ─────────────────────────────────────────────────────────────────────────
    def _is_real_numeric_projection(
        self,
        value: Any,
    ) -> bool:
        r"""
        Determina si un valor es un escalar real susceptible de norma residual.
        """
        if isinstance(value, (bool, np.bool_)):
            return False

        if isinstance(value, (str, bytes, bytearray, complex)):
            return False

        if isinstance(value, (int, float, np.integer, np.floating)):
            return True

        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError):
            return False

        return bool(np.isfinite(scalar))

    # ─────────────────────────────────────────────────────────────────────────
    # 1.3. Coerción a escalar real
    # ─────────────────────────────────────────────────────────────────────────
    def _to_real_scalar(
        self,
        value: Any,
    ) -> float:
        r"""
        Coerciona un valor a escalar float64 finito, rechazando booleanos,
        cadenas, bytes y números complejos.
        """
        if isinstance(value, (bool, np.bool_)):
            raise NumericSingularityVeto(
                "Booleano detectado en proyección numérica. Se requiere un "
                "escalar real."
            )

        if isinstance(value, (str, bytes, bytearray, complex)):
            raise NumericSingularityVeto(
                "Tipo no real detectado en proyección numérica."
            )

        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise NumericSingularityVeto(
                "El valor no puede proyectarse al cuerpo real IEEE 754."
            ) from exc

        if not np.isfinite(scalar):
            raise NumericSingularityVeto(
                "Singularidad detectada: el escalar es NaN o infinito."
            )

        return scalar

    # ─────────────────────────────────────────────────────────────────────────
    # 1.4. Comparación categórica segura
    # ─────────────────────────────────────────────────────────────────────────
    def _safe_categorical_equality(
        self,
        left: Any,
        right: Any,
    ) -> bool:
        r"""
        Compara de forma segura objetos discretos, incluyendo arreglos NumPy,
        sin provocar ambigüedades de verdad.
        """
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            try:
                left_arr = np.asarray(left)
                right_arr = np.asarray(right)

                if left_arr.shape != right_arr.shape:
                    return False

                return bool(np.array_equal(left_arr, right_arr))
            except Exception:
                return False

        try:
            return bool(left == right)
        except Exception:
            return False

    # ─────────────────────────────────────────────────────────────────────────
    # 1.5. Certificación de retracto de deformación idempotente
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_deformation_retract_idempotence(
        self,
        transform_func: Callable[[Any], Any],
        raw_input: Any,
    ) -> IdempotenceAuditData:
        r"""
        Aplica el operador dos veces y verifica:

            f(f(x)) = f(x)

        Si la proyección es numérica, mide norma residual contra tolerancia
        adaptativa. Si es discreta, exige igualdad categórica estricta.
        """
        if not callable(transform_func):
            raise DomainIntegrityViolationError(
                "transform_func debe ser un operador callable."
            )

        try:
            first_projection = transform_func(raw_input)
        except Exception as exc:
            raise IdempotencyViolationError(
                "El operador de normalización falló al evaluar f(x)."
            ) from exc

        try:
            second_projection = transform_func(first_projection)
        except Exception as exc:
            raise IdempotencyViolationError(
                "El operador de normalización falló al evaluar f(f(x))."
            ) from exc

        input_type = type(raw_input).__name__

        if (
            self._is_real_numeric_projection(first_projection)
            and self._is_real_numeric_projection(second_projection)
        ):
            first_scalar = self._to_real_scalar(first_projection)
            second_scalar = self._to_real_scalar(second_projection)

            residual = abs(second_scalar - first_scalar)
            tolerance = self._adaptive_tolerance(
                _MACHINE_EPSILON,
                max(1.0, abs(first_scalar), abs(second_scalar)),
            )

            if residual > tolerance:
                raise IdempotencyViolationError(
                    f"Ruptura Funtorial: El operador numérico no es un "
                    f"proyector idempotente. Norma residual {residual:.6e} > "
                    f"{tolerance:.6e}. Induce fricción parásita."
                )

            return IdempotenceAuditData(
                is_idempotent=True,
                residual_norm=residual,
                input_type=input_type,
                projection_type="numeric",
                idempotence_tolerance=tolerance,
            )

        if not self._safe_categorical_equality(
            first_projection,
            second_projection,
        ):
            raise IdempotencyViolationError(
                "Ruptura Funtorial: El operador de normalización falló el "
                "Retracto de Deformación. "
                f"P(P(x))={second_projection!r} ≠ P(x)={first_projection!r}."
            )

        return IdempotenceAuditData(
            is_idempotent=True,
            residual_norm=0.0,
            input_type=input_type,
            projection_type="categorical",
            idempotence_tolerance=0.0,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1.6. Auditoría de proyección FPU
    # ─────────────────────────────────────────────────────────────────────────
    def _audit_fpu_projection_bounds(
        self,
        numeric_value: float,
    ) -> FPUProjectionData:
        r"""
        Audita el dominio de clausura real. Si el valor es indeterminado,
        detona el veto.
        """
        scalar = self._to_real_scalar(numeric_value)

        return FPUProjectionData(
            is_finite=True,
            validated_scalar=scalar,
            numeric_tolerance=0.0,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1.7. ÚLTIMO MÉTODO DE FASE 1: HANDOFF FORMAL HACIA FASE 2
    # ─────────────────────────────────────────────────────────────────────────
    def _phase1_certify_and_handoff_to_phase2(
        self,
        normalizer_func: Optional[Callable[[Any], Any]] = None,
        raw_input: Any = None,
        numeric_value: Optional[float] = None,
    ) -> Phase1DomainHandoff:
        r"""
        Último método de la Fase 1.

        Su definición formal es la continuación directa de la Fase 2:
        entrega los certificados de idempotencia y FPU como prefijo obligatorio
        de la filtración de variedad estadística.
        """
        idempotence_audit: Optional[IdempotenceAuditData] = None
        fpu_audit: Optional[FPUProjectionData] = None
        has_domain_payload = False

        if normalizer_func is not None:
            if raw_input is None:
                raise DomainIntegrityViolationError(
                    "normalizer_func requiere raw_input para auditar el "
                    "retracto de deformación."
                )

            idempotence_audit = (
                self._certify_deformation_retract_idempotence(
                    normalizer_func,
                    raw_input,
                )
            )
            has_domain_payload = True

        elif raw_input is not None:
            raise DomainIntegrityViolationError(
                "raw_input fue provisto sin normalizer_func. "
                "No existe operador de retracto para certificar."
            )

        if numeric_value is not None:
            fpu_audit = self._audit_fpu_projection_bounds(numeric_value)
            has_domain_payload = True

        logger.debug(
            "Fase 1 completada. Idempotencia=%s | FPU=%s.",
            str(idempotence_audit is not None),
            str(fpu_audit is not None),
        )

        return Phase1DomainHandoff(
            idempotence_audit=idempotence_audit,
            fpu_audit=fpu_audit,
            has_domain_payload=has_domain_payload,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2: FILTRACIÓN DE VARIEDAD ESTADÍSTICA Y GEOMETRÍA DE OUTLIERS      ║
# ║   Evalúa Z-Score modificado: M_i = 0.6745 |x_i - x̃| / MAD ≤ τ.          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase2_StatisticalManifoldFilter(
    Phase1_DeformationRetractAndFPUProjector
):
    r"""
    Protege el Tensor Métrico Riemanniano de deformaciones espurias causadas
    por valores anómalos. Aplica un filtro de hiperesfera geométrica basado en
    mediana y MAD.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 2.1. Coerción de tensor estadístico finito
    # ─────────────────────────────────────────────────────────────────────────
    def _coerce_finite_statistical_tensor(
        self,
        data_series: Any,
    ) -> NDArray[np.float64]:
        r"""
        Materializa un tensor unidimensional float64, no vacío y finito.
        """
        try:
            arr = np.asarray(data_series, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise StatisticalManifoldDeformationVeto(
                "El tensor de datos no puede materializarse como "
                "NDArray[np.float64]."
            ) from exc

        if arr.ndim == 0:
            arr = arr.reshape(1)
        else:
            arr = arr.reshape(-1)

        if arr.size == 0:
            raise StatisticalManifoldDeformationVeto(
                "El tensor de datos está vacío. Colapso volumétrico."
            )

        if not np.all(np.isfinite(arr)):
            raise StatisticalManifoldDeformationVeto(
                "El tensor de datos contiene componentes NaN o infinitas."
            )

        return arr

    # ─────────────────────────────────────────────────────────────────────────
    # 2.2. Coerción de τ_critical
    # ─────────────────────────────────────────────────────────────────────────
    def _coerce_positive_tau_critical(
        self,
        tau_critical: Any,
    ) -> float:
        r"""
        Valida que τ_critical sea un escalar real finito y estrictamente
        positivo.
        """
        if isinstance(tau_critical, (bool, np.bool_)):
            raise DomainIntegrityViolationError(
                "tau_critical no puede ser booleano."
            )

        if isinstance(tau_critical, (str, bytes, bytearray, complex)):
            raise DomainIntegrityViolationError(
                "tau_critical debe ser un escalar real positivo."
            )

        try:
            tau = float(tau_critical)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DomainIntegrityViolationError(
                "tau_critical no puede convertirse en escalar real."
            ) from exc

        if not np.isfinite(tau) or tau <= 0.0:
            raise DomainIntegrityViolationError(
                "tau_critical debe ser finito y estrictamente positivo."
            )

        return tau

    # ─────────────────────────────────────────────────────────────────────────
    # 2.3. Filtración de variedad estadística
    # ─────────────────────────────────────────────────────────────────────────
    def _enforce_statistical_manifold_filtration(
        self,
        data_series: NDArray[np.float64],
        tau_critical: float = _TAU_CRITICAL_ZSCORE,
    ) -> StatisticalFiltrationData:
        r"""
        Acota el espacio de fases midiendo la desviación absoluta respecto a la
        mediana geométrica.

        Si MAD se degenera, emplea una escala adaptativa para evitar división
        por cero y preservar la capacidad de extirpación de outliers.
        """
        x = self._coerce_finite_statistical_tensor(data_series)
        tau = self._coerce_positive_tau_critical(tau_critical)

        manifold_median = float(np.median(x))
        absolute_deviations = np.abs(x - manifold_median)
        manifold_mad = float(np.median(absolute_deviations))

        max_deviation = float(np.max(absolute_deviations)) if x.size else 0.0
        base_scale = self._adaptive_tolerance(_MACHINE_EPSILON, x)

        # Varianza cero o variedad colapsada en un punto (Delta de Dirac).
        if manifold_mad <= base_scale:
            if max_deviation <= base_scale:
                return StatisticalFiltrationData(
                    filtered_tensor=x.copy(),
                    extirpated_count=0,
                    manifold_median=manifold_median,
                    manifold_mad=0.0,
                    tau_critical=tau,
                    max_modified_z_score=0.0,
                    dispersion_scale_used=base_scale,
                )

            # MAD degenerada pero existen desviaciones: escala de emergencia.
            dispersion_scale = base_scale
        else:
            dispersion_scale = manifold_mad

        if dispersion_scale <= 0.0:
            dispersion_scale = max(_MACHINE_EPSILON, base_scale)

        modified_z_scores = (_MAD_CONSTANT * absolute_deviations) / dispersion_scale

        if not np.all(np.isfinite(modified_z_scores)):
            raise StatisticalManifoldDeformationVeto(
                "El Z-Score modificado contiene componentes no finitas."
            )

        valid_indices = modified_z_scores <= tau
        filtered_tensor = x[valid_indices]
        extirpated_count = int(x.size - filtered_tensor.size)
        max_modified_z_score = float(np.max(modified_z_scores)) if x.size else 0.0

        if filtered_tensor.size == 0:
            raise StatisticalManifoldDeformationVeto(
                f"La filtración geométrica aniquiló toda la variedad. "
                f"Ruido estocástico masivo insalvable. MAD={manifold_mad:.6e}, "
                f"escala={dispersion_scale:.6e}, τ={tau:.6e}."
            )

        return StatisticalFiltrationData(
            filtered_tensor=filtered_tensor,
            extirpated_count=extirpated_count,
            manifold_median=manifold_median,
            manifold_mad=float(manifold_mad),
            tau_critical=tau,
            max_modified_z_score=max_modified_z_score,
            dispersion_scale_used=float(dispersion_scale),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 2.4. ÚLTIMO MÉTODO DE FASE 2: HANDOFF FORMAL HACIA FASE 3
    # ─────────────────────────────────────────────────────────────────────────
    def _phase2_filter_and_handoff_to_phase3(
        self,
        phase1_handoff: Phase1DomainHandoff,
        data_series: Optional[NDArray[np.float64]] = None,
        tau_critical: float = _TAU_CRITICAL_ZSCORE,
    ) -> Phase2StatisticalHandoff:
        r"""
        Último método de la Fase 2.

        Su definición formal es la continuación directa de la Fase 3:
        entrega los certificados de dominio y la filtración estadística como
        prefijo obligatorio del difeomorfismo I/O.
        """
        if not isinstance(phase1_handoff, Phase1DomainHandoff):
            raise DomainIntegrityViolationError(
                "Fase 2 exige un Phase1DomainHandoff como prefijo formal."
            )

        filtration_audit: Optional[StatisticalFiltrationData] = None

        if data_series is not None:
            filtration_audit = self._enforce_statistical_manifold_filtration(
                data_series,
                tau_critical,
            )

        logger.debug(
            "Fase 2 completada. Filtración estadística=%s.",
            str(filtration_audit is not None),
        )

        return Phase2StatisticalHandoff(
            phase1_handoff=phase1_handoff,
            filtration_audit=filtration_audit,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3: CERTIFICACIÓN DE DIFEOMORFISMO DE FRONTERA I/O                  ║
# ║   Certifica ker(path_resolve) = ∅ (aislamiento de ciclos).                ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase3_IOBoundaryDiffeomorphismCertifier(
    Phase2_StatisticalManifoldFilter
):
    r"""
    Garantiza que la estructura de carpetas sea un Grafo Acíclico Dirigido (DAG)
    en el sistema de inodos, evitando inyecciones de entropía maliciosa mediante
    Symlink Loops.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 3.1. Coerción de ruta de frontera
    # ─────────────────────────────────────────────────────────────────────────
    def _coerce_filesystem_path(
        self,
        target_path: Union[str, Path],
    ) -> Path:
        r"""
        Coerciona la entrada a Path y rechaza dominios inválidos.
        """
        if isinstance(target_path, Path):
            return target_path

        if isinstance(target_path, str):
            if not target_path.strip():
                raise DomainIntegrityViolationError(
                    "file_path no puede ser una cadena vacía."
                )
            return Path(target_path)

        if isinstance(target_path, os.PathLike):
            return Path(target_path)

        raise DomainIntegrityViolationError(
            "file_path debe ser str, pathlib.Path u os.PathLike."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3.2. Certificación de difeomorfismo I/O
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_io_boundary_diffeomorphism(
        self,
        target_path: Union[str, Path],
    ) -> IOBoundaryDiffeomorphismData:
        r"""
        Resuelve recursivamente la ruta del archivo, audita el núcleo lógico y
        veta ciclos topológicos en el grafo de inodos.
        """
        path_obj = self._coerce_filesystem_path(target_path)

        try:
            # resolve(strict=True) transita el grafo de inodos y eleva
            # RuntimeError ante ciclos homológicos infinitos.
            resolved_path = path_obj.resolve(strict=True)
        except RuntimeError as exc:
            raise IOBoundaryTopologyVeto(
                "Paradoja de Difeomorfismo Detectada: Bucle homológico "
                "infinito en Inodos. Riesgo de ataque por enlaces simbólicos "
                f"(Symlinks). Detalle: {exc}"
            ) from exc
        except OSError as exc:
            raise IOBoundaryTopologyVeto(
                f"Falla de variedad I/O al resolver la ruta: {exc}"
            ) from exc
        except Exception as exc:
            raise IOBoundaryTopologyVeto(
                f"Falla crítica resolviendo la variedad I/O: {exc}"
            ) from exc

        if not resolved_path.exists():
            raise IOBoundaryTopologyVeto(
                "El mapeo del inodo colapsó al vacío. Entidad inexistente."
            )

        if not resolved_path.is_absolute():
            raise IOBoundaryTopologyVeto(
                "La ruta resuelta no constituye una variedad absoluta."
            )

        inode_depth = len(resolved_path.parts)

        if inode_depth > _MAX_SYMLINK_DEPTH:
            raise IOBoundaryTopologyVeto(
                f"Profundidad topológica excesiva: {inode_depth} > "
                f"{_MAX_SYMLINK_DEPTH}. Posible grafo de inodos degenerado."
            )

        return IOBoundaryDiffeomorphismData(
            resolved_absolute_path=str(resolved_path),
            is_acyclic_mapping=True,
            inode_depth=inode_depth,
            is_absolute_path=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3.3. ÚLTIMO MÉTODO DE FASE 3: FINALIZACIÓN FUNTORIAL
    # ─────────────────────────────────────────────────────────────────────────
    def _phase3_finalize_from_phase2_handoff(
        self,
        phase2_handoff: Phase2StatisticalHandoff,
        file_path: Optional[Union[str, Path]] = None,
    ) -> ThermodynamicBoundaryState:
        r"""
        Último método de la Fase 3.

        Compone los certificados de Fase 1, Fase 2 y Fase 3 en el objeto
        terminal ThermodynamicBoundaryState.
        """
        if not isinstance(phase2_handoff, Phase2StatisticalHandoff):
            raise DomainIntegrityViolationError(
                "Fase 3 exige un Phase2StatisticalHandoff como prefijo formal."
            )

        io_audit: Optional[IOBoundaryDiffeomorphismData] = None

        if file_path is not None:
            io_audit = self._certify_io_boundary_diffeomorphism(file_path)

        state = ThermodynamicBoundaryState(
            idempotence_audit=phase2_handoff.phase1_handoff.idempotence_audit,
            fpu_audit=phase2_handoff.phase1_handoff.fpu_audit,
            filtration_audit=phase2_handoff.filtration_audit,
            io_audit=io_audit,
            is_epistemologically_valid=True,
        )

        logger.info(
            "Frontera Termodinámica certificada. "
            "Idempotencia=%s | FPU=%s | Outliers extirpados=%d | I/O Seguro=%s.",
            state.idempotence_audit is not None,
            state.fpu_audit is not None,
            state.filtration_audit.extirpated_count
            if state.filtration_audit
            else 0,
            state.io_audit is not None,
        )

        return state


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: UTILS AGENT                                        ║
# ║   Endofuntor Z_Utils = Φ₃ ∘ Φ₂ ∘ Φ₁                                      ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class UtilsAgent(Morphism, Phase3_IOBoundaryDiffeomorphismCertifier):
    r"""
    El Custodio de la Frontera Termodinámica.

    Sella herméticamente la Capa Límite del ecosistema, aplicando matemáticas de
    grados doctorales para impedir que la estocástica del sistema operativo o
    errores de formato destruyan los invariantes operacionales de APU Filter.
    """

    def execute_thermodynamic_boundary_governance(
        self,
        normalizer_func: Optional[Callable[[Any], Any]] = None,
        raw_input: Any = None,
        numeric_value: Optional[float] = None,
        data_series: Optional[NDArray[np.float64]] = None,
        file_path: Optional[Union[str, Path]] = None,
    ) -> ThermodynamicBoundaryState:
        r"""
        Ejecuta la composición funtorial estricta según los datos provistos.
        """
        phase1_handoff = self._phase1_certify_and_handoff_to_phase2(
            normalizer_func=normalizer_func,
            raw_input=raw_input,
            numeric_value=numeric_value,
        )

        phase2_handoff = self._phase2_filter_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            data_series=data_series,
            tau_critical=_TAU_CRITICAL_ZSCORE,
        )

        return self._phase3_finalize_from_phase2_handoff(
            phase2_handoff=phase2_handoff,
            file_path=file_path,
        )

    def __call__(
        self,
        normalizer_func: Optional[Callable[[Any], Any]] = None,
        raw_input: Any = None,
        numeric_value: Optional[float] = None,
        data_series: Optional[NDArray[np.float64]] = None,
        file_path: Optional[Union[str, Path]] = None,
    ) -> ThermodynamicBoundaryState:
        r"""Alias invocable del endofuntor de frontera termodinámica."""
        return self.execute_thermodynamic_boundary_governance(
            normalizer_func=normalizer_func,
            raw_input=raw_input,
            numeric_value=numeric_value,
            data_series=data_series,
            file_path=file_path,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "UtilsAgentError",
    "DomainIntegrityViolationError",
    "IdempotencyViolationError",
    "NumericSingularityVeto",
    "StatisticalManifoldDeformationVeto",
    "IOBoundaryTopologyVeto",
    "IdempotenceAuditData",
    "FPUProjectionData",
    "StatisticalFiltrationData",
    "IOBoundaryDiffeomorphismData",
    "Phase1DomainHandoff",
    "Phase2StatisticalHandoff",
    "ThermodynamicBoundaryState",
    "Phase1_DeformationRetractAndFPUProjector",
    "Phase2_StatisticalManifoldFilter",
    "Phase3_IOBoundaryDiffeomorphismCertifier",
    "UtilsAgent",
]