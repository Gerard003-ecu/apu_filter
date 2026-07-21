# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : MAC Minimizer Agent (Custodio de la Purificación Espectral)         ║
║ Ruta   : app/agents/boole/tactics/mac_minimizer_agent.py                     ║
║ Versión: 2.0.0-Uhlmann-Holevo-Majorization-Categorical-Strict                ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GEOMETRÍA NO CONMUTATIVA (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor gobierna el `mac_minimizer.py` en el estrato WISDOM.

Su mandato axiomático es garantizar que el Funtor de Purificación Espectral P
sobre la Matriz Atómica de Conocimiento (MAC) preserve el isomorfismo de la
información cuántica, auditando:

    1. El preorden de majorización cuántica.
    2. La fidelidad de Uhlmann.
    3. El límite termodinámico de von Neumann y la capacidad de Holevo.

ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Auditoría de Majorización Cuántica:
    Exige:

        λ(ρ_pur) ≽ λ(ρ_orig),

    es decir, para cada k:

        Σ_{j=1}^k λ_j^↓(ρ_pur) ≥ Σ_{j=1}^k λ_j^↓(ρ_orig).

    Esto asegura que la purificación incrementa o preserva la pureza espectral.

    Último método de Fase 1:
        _audit_quantum_majorization(...)

    Dicho método retorna un certificado `MajorizationAuditData`, el cual se
    convierte en el objeto inicial de la Fase 2.

Fase 2 → Certificación de Fidelidad de Uhlmann:
    Computa:

        F(ρ, σ) = (Tr sqrt(sqrt(ρ) σ sqrt(ρ)))² ≥ F_min.

    Previene la mutilación semántica del conocimiento base.

    Primer método de Fase 2:
        _certify_uhlmann_fidelity_bound(..., majorization_audit)

    Este método es la continuación formal de Fase 1: recibe el certificado de
    majorización y lo propaga como invariante inicial de la fidelidad.

Fase 3 → Cota de Capacidad de Holevo y Entropía:
    Verifica:

        ΔS = S(ρ_pur) - S(ρ_orig) ≤ 0.

    Garantiza que la poda espectral no destruya la capacidad del canal de
    transmisión de sabiduría.

    Primer método de Fase 3:
        _enforce_holevo_capacity_retention(..., fidelity_audit)

    Este método continúa formalmente la Fase 2: recibe el certificado de
    fidelidad y verifica que la termodinámica cuántica sea compatible.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Final, List, Optional

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
    from app.wisdom.atomic_knowledge_matrix import AtomicDensityMatrix
except ImportError:

    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos E_MIC."""
        pass

    class Morphism:
        r"""Clase base de Morfismos del Topos."""
        pass

    class AtomicDensityMatrix:
        r"""Marcador estructural para compatibilidad sin dependencia externa."""
        pass


logger = logging.getLogger("MAC.Wisdom.MinimizerAgent")


# ═══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES FÍSICAS Y LÍMITES CUÁNTICOS
# ═══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)

_UHLMANN_FIDELITY_MIN: Final[float] = 0.95
_ENTROPY_TOLERANCE: Final[float] = 1e-12
_MAJORIZATION_TOLERANCE: Final[float] = 1e-10

_HERMITIAN_TOLERANCE: Final[float] = 1e-10
_PSD_TOLERANCE: Final[float] = 1e-10
_TRACE_TOLERANCE: Final[float] = 1e-10
_FIDELITY_NUMERICAL_TOLERANCE: Final[float] = 1e-12

_NUMERICAL_SAFETY_FACTOR: Final[float] = 128.0


# ═══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES CUÁNTICAS
# ═══════════════════════════════════════════════════════════════════════════════
class MACMinimizerAgentError(TopologicalInvariantError):
    r"""Excepción raíz del Custodio de la Purificación Espectral."""
    pass


class DensityMatrixValidationError(MACMinimizerAgentError):
    r"""Detonada si una matriz de densidad no es hermítica, PSD o traza-uno válida."""
    pass


class QuantumMajorizationViolation(MACMinimizerAgentError):
    r"""Detonada si ρ_purificada no majoriza a ρ_orig."""
    pass


class UhlmannFidelityCollapseError(MACMinimizerAgentError):
    r"""Detonada si F(ρ, σ) < F_min. La reducción espectral mutiló el significado."""
    pass


class HolevoCapacityDeficitError(MACMinimizerAgentError):
    r"""Detonada si ΔS > 0 o si la poda destruye capacidad semántica útil."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Espacio de Hilbert)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class MajorizationAuditData:
    r"""
    Artefacto de Fase 1.
    Certificado del preorden de majorización cuántica.

    Este objeto es el resultado final del último método de Fase 1 y el objeto
    inicial de Fase 2.
    """
    dimension: int
    trace_original: float
    trace_purified: float
    min_eigenvalue_original: float
    min_eigenvalue_purified: float
    max_deviation: float
    majorization_tolerance: float
    is_majorized: bool


@dataclass(frozen=True, slots=True)
class FidelityAuditData:
    r"""
    Artefacto de Fase 2.
    Certificado de Fidelidad de Uhlmann.

    Este objeto es el resultado final de Fase 2 y el objeto inicial de Fase 3.
    """
    uhlmann_fidelity: float
    fidelity_tolerance: float
    fidelity_min_required: float
    is_fidelity_preserved: bool


@dataclass(frozen=True, slots=True)
class HolevoAuditData:
    r"""
    Artefacto de Fase 3.
    Certificado Termodinámico de von Neumann.
    """
    entropy_original: float
    entropy_purified: float
    entropy_delta: float
    entropy_tolerance: float
    is_capacity_preserved: bool


@dataclass(frozen=True, slots=True)
class PurificationGovernanceState:
    r"""
    Objeto final del endofuntor Z_MAC-Agent.
    """
    majorization_audit: MajorizationAuditData
    fidelity_audit: FidelityAuditData
    holevo_audit: HolevoAuditData
    is_epistemologically_valid: bool


# ═══════════════════════════════════════════════════════════════════════════════
# §D. GUARDAS NUMÉRICAS INTERNAS
# ═══════════════════════════════════════════════════════════════════════════════
class _FiniteNumericalGuard:
    r"""
    Capa de saneamiento numérico para evitar que singularidades aritméticas
    contaminen los invariantes cuánticos.
    """

    @staticmethod
    def _as_finite_complex_matrix(
        name: str,
        value: Any,
        *,
        square: bool = False,
    ) -> NDArray[np.complex128]:
        r"""
        Valida una matriz compleja finita.
        """
        try:
            arr = np.asarray(value, dtype=np.complex128)
        except Exception as exc:
            raise TypeError(f"{name} debe ser una matriz numérica compleja o real.") from exc

        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contiene valores NaN o infinitos.")

        if arr.ndim != 2:
            raise ValueError(f"{name} debe ser una matriz 2D.")

        if square and arr.shape[0] != arr.shape[1]:
            raise ValueError(f"{name} debe ser una matriz cuadrada.")

        return arr

    @staticmethod
    def _as_finite_real_vector(name: str, value: Any) -> NDArray[np.float64]:
        r"""
        Valida un vector real finito.

        Acepta:
            - Vectores 1D.
            - Vectores columna (n, 1).
            - Vectores fila (1, n).
            - Escalares, interpretados como vector de dimensión 1.
            - Vectores vacíos sólo si el contexto lo permite.
        """
        try:
            raw = np.asarray(value)
        except Exception as exc:
            raise TypeError(f"{name} no puede interpretarse como arreglo numérico.") from exc

        if np.iscomplexobj(raw):
            if not np.all(np.isfinite(raw)):
                raise ValueError(f"{name} contiene valores NaN o infinitos.")

            if raw.size > 0:
                imag_max = float(np.max(np.abs(np.imag(raw))))
            else:
                imag_max = 0.0

            imag_tolerance = max(
                _FIDELITY_NUMERICAL_TOLERANCE,
                _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON,
            )

            if imag_max > imag_tolerance:
                raise TypeError(
                    f"{name} debe ser real; se rechazó componente imaginaria "
                    f"de magnitud {imag_max:.6e}."
                )

            raw = np.real(raw)

        try:
            arr = raw.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} debe ser numérico real convertible a float64.") from exc

        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim == 2 and 1 in arr.shape:
            arr = arr.reshape(-1)
        elif arr.ndim != 1:
            raise ValueError(f"{name} debe ser un vector 1D, fila, columna o escalar.")

        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contiene valores NaN o infinitos.")

        return arr


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1: AUDITORÍA DEL PREORDEN DE MAJORIZACIÓN CUÁNTICA                   ║
# ║                                                                             ║
# ║   Exige:                                                                    ║
# ║       λ(ρ_pur) ≽ λ(ρ_orig)                                                  ║
# ║                                                                             ║
# ║   mediante curvas de Lorenz espectrales.                                    ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_QuantumMajorizationAuditor(_FiniteNumericalGuard):
    r"""
    Garantiza que el minimizador no inyecte entropía estocástica.

    El estado purificado debe estar más cerca de un estado puro en el sentido
    del preorden de majorización cuántica:

        ρ_pur ≽ ρ_orig.

    Equivalentemente, para los autovalores ordenados descendentemente:

        Σ_{j=1}^k λ_j^↓(ρ_pur) ≥ Σ_{j=1}^k λ_j^↓(ρ_orig),  ∀k,

    con igualdad en k = d (traza total).
    """

    def _sanitize_density_matrix(
        self,
        name: str,
        rho: NDArray[np.complex128],
    ) -> tuple[NDArray[np.complex128], NDArray[np.float64], float, float]:
        r"""
        Valida y sanea una matriz de densidad cuántica.

        Exige:
            - Matriz cuadrada.
            - Entradas finitas.
            - Hermiticidad dentro de tolerancia.
            - Traza real positiva.
            - Positive semidefinite (PSD) dentro de tolerancia.

        Retorna:
            rho_sanitized:
                Matriz hermítica, PSD y traza uno reconstruida espectralmente.

            eigenvalues:
                Autovalores reales, recortados y normalizados.

            original_trace:
                Traza original antes de normalización.

            min_eigenvalue:
                Mínimo autovalor original antes de recorte.
        """
        arr = self._as_finite_complex_matrix(name, rho, square=True)

        if arr.shape[0] == 0:
            raise DensityMatrixValidationError(f"{name} no puede ser una matriz vacía.")

        frobenius_norm = float(la.norm(arr, ord="fro"))

        if not math.isfinite(frobenius_norm):
            raise DensityMatrixValidationError(f"La norma de Frobenius de {name} no es finita.")

        hermitian_residual = float(la.norm(arr - arr.conj().T, ord="fro")) / max(
            1.0,
            frobenius_norm,
        )

        hermitian_tolerance = max(
            _HERMITIAN_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON,
        )

        if hermitian_residual > hermitian_tolerance:
            raise DensityMatrixValidationError(
                f"{name} no es hermítica dentro de tolerancia. "
                f"Residuo hermético relativo = {hermitian_residual:.6e} > "
                f"{hermitian_tolerance:.6e}."
            )

        rho_hermitian = (arr + arr.conj().T) / 2.0

        trace_complex = np.trace(rho_hermitian)
        trace_real = float(np.real(trace_complex))
        trace_imag = float(np.imag(trace_complex))

        if not math.isfinite(trace_real) or not math.isfinite(trace_imag):
            raise DensityMatrixValidationError(f"La traza de {name} no es finita.")

        trace_tolerance = max(
            _TRACE_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(1.0, abs(trace_real)),
        )

        if abs(trace_imag) > trace_tolerance:
            raise DensityMatrixValidationError(
                f"La traza de {name} tiene parte imaginaria significativa: "
                f"{trace_imag:.6e}."
            )

        if trace_real <= trace_tolerance:
            raise DensityMatrixValidationError(
                f"La traza de {name} no es positiva: trace = {trace_real:.6e}."
            )

        if abs(trace_real - 1.0) > trace_tolerance:
            logger.warning(
                "%s tiene traza %.6e distinta de 1; se normaliza internamente.",
                name,
                trace_real,
            )
            rho_hermitian = rho_hermitian / trace_real

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(rho_hermitian)
        except np.linalg.LinAlgError as exc:
            raise DensityMatrixValidationError(
                f"Diagonalización hermítica de {name} falló."
            ) from exc

        eigenvalues = np.asarray(eigenvalues, dtype=np.float64)

        if not np.all(np.isfinite(eigenvalues)):
            raise DensityMatrixValidationError(
                f"Los autovalores de {name} contienen valores NaN o infinitos."
            )

        min_eigenvalue = float(np.min(eigenvalues)) if eigenvalues.size > 0 else 0.0

        psd_tolerance = max(
            _PSD_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(1.0, abs(trace_real)),
        )

        if min_eigenvalue < -psd_tolerance:
            raise DensityMatrixValidationError(
                f"{name} no es positive semidefinite. "
                f"Autovalor mínimo = {min_eigenvalue:.6e} < -{psd_tolerance:.6e}."
            )

        eigenvalues = np.clip(eigenvalues, 0.0, None)
        eigen_sum = float(np.sum(eigenvalues))

        if eigen_sum <= psd_tolerance:
            raise DensityMatrixValidationError(
                f"{name} tiene espectro nulo o numéricamente degenerado."
            )

        eigenvalues = eigenvalues / eigen_sum

        rho_sanitized = (eigenvectors * eigenvalues) @ eigenvectors.conj().T
        rho_sanitized = (rho_sanitized + rho_sanitized.conj().T) / 2.0

        return rho_sanitized, eigenvalues, trace_real, min_eigenvalue

    def _sanitize_spectrum(
        self,
        name: str,
        spectrum: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float, float]:
        r"""
        Valida y sanea un espectro de autovalores.

        Retorna:
            spectrum_sanitized:
                Espectro no negativo y normalizado a traza uno.

            original_trace:
                Suma original antes de normalización.

            min_eigenvalue:
                Mínimo autovalor original antes de recorte.
        """
        arr = self._as_finite_real_vector(name, spectrum)

        if arr.size == 0:
            raise DensityMatrixValidationError(f"{name} no puede ser un espectro vacío.")

        min_eigenvalue = float(np.min(arr))

        spectral_mass = float(np.sum(np.abs(arr)))

        psd_tolerance = max(
            _PSD_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(1.0, spectral_mass),
        )

        if min_eigenvalue < -psd_tolerance:
            raise DensityMatrixValidationError(
                f"{name} contiene autovalores negativos no físicos. "
                f"Mínimo = {min_eigenvalue:.6e}."
            )

        arr = np.clip(arr, 0.0, None)
        original_trace = float(np.sum(arr))

        if original_trace <= psd_tolerance:
            raise DensityMatrixValidationError(
                f"{name} tiene masa espectral nula o degenerada."
            )

        trace_tolerance = max(
            _TRACE_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(1.0, original_trace),
        )

        if abs(original_trace - 1.0) > trace_tolerance:
            logger.warning(
                "%s tiene traza espectral %.6e distinta de 1; se normaliza internamente.",
                name,
                original_trace,
            )

        arr = arr / original_trace

        return arr, original_trace, min_eigenvalue

    def _assert_spectra_consistent(
        self,
        name: str,
        supplied_spectrum: NDArray[np.float64],
        reference_spectrum: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica que un espectro suministrado sea consistente con el espectro
        obtenido desde la matriz de densidad.
        """
        a = np.sort(supplied_spectrum)[::-1]
        b = np.sort(reference_spectrum)[::-1]

        max_dim = max(a.size, b.size)

        a_pad = np.pad(a, (0, max_dim - a.size))
        b_pad = np.pad(b, (0, max_dim - b.size))

        if a_pad.size == 0:
            max_difference = 0.0
        else:
            max_difference = float(np.max(np.abs(a_pad - b_pad)))

        mass_scale = max(
            1.0,
            float(np.sum(np.abs(a_pad))),
            float(np.sum(np.abs(b_pad))),
        )

        tolerance = max(
            _MAJORIZATION_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * mass_scale,
        )

        if max_difference > tolerance:
            raise DensityMatrixValidationError(
                f"Inconsistencia espectral en {name}. "
                f"Diferencia máxima entre espectro suministrado y espectro de la "
                f"matriz = {max_difference:.6e} > {tolerance:.6e}."
            )

    def _resolve_spectrum(
        self,
        name: str,
        spectrum: Optional[NDArray[np.float64]],
        density_matrix: Optional[NDArray[np.complex128]],
    ) -> tuple[NDArray[np.float64], float, float]:
        r"""
        Resuelve el espectro válido a partir de:
            - Una matriz de densidad.
            - Un vector de autovalores.
            - Ambos, con verificación de consistencia.
        """
        if density_matrix is not None:
            _, matrix_spectrum, matrix_trace, matrix_min_eval = self._sanitize_density_matrix(
                name,
                density_matrix,
            )

            if spectrum is not None:
                supplied_spectrum, supplied_trace, _ = self._sanitize_spectrum(
                    f"{name}_evals",
                    spectrum,
                )

                trace_consistency_tolerance = max(
                    _TRACE_TOLERANCE,
                    _NUMERICAL_SAFETY_FACTOR
                    * _MACHINE_EPSILON
                    * max(1.0, abs(matrix_trace), abs(supplied_trace)),
                )

                if abs(matrix_trace - supplied_trace) > trace_consistency_tolerance:
                    raise DensityMatrixValidationError(
                        f"Inconsistencia de traza en {name}. "
                        f"Traza de matriz = {matrix_trace:.6e}, "
                        f"traza espectral = {supplied_trace:.6e}."
                    )

                self._assert_spectra_consistent(
                    name,
                    supplied_spectrum,
                    matrix_spectrum,
                )

            return matrix_spectrum, matrix_trace, matrix_min_eval

        if spectrum is not None:
            return self._sanitize_spectrum(name, spectrum)

        raise ValueError(
            f"Debe proveerse spectrum o density_matrix para {name}."
        )

    def _audit_quantum_majorization(
        self,
        evals_orig: Optional[NDArray[np.float64]] = None,
        evals_purified: Optional[NDArray[np.float64]] = None,
        *,
        rho_orig: Optional[NDArray[np.complex128]] = None,
        rho_purified: Optional[NDArray[np.complex128]] = None,
    ) -> MajorizationAuditData:
        r"""
        Último método de la Fase 1.

        Ordena autovalores descendentemente y verifica las curvas de Lorenz:

            Σ_{j=1}^k λ_j^↓(ρ_pur) ≥ Σ_{j=1}^k λ_j^↓(ρ_orig).

        Este método retorna un certificado `MajorizationAuditData`, el cual
        constituye el objeto inicial de la Fase 2.
        """
        spectrum_orig, trace_orig, min_eval_orig = self._resolve_spectrum(
            "rho_orig",
            evals_orig,
            rho_orig,
        )

        spectrum_pur, trace_pur, min_eval_pur = self._resolve_spectrum(
            "rho_purified",
            evals_purified,
            rho_purified,
        )

        lambda_orig = np.sort(spectrum_orig)[::-1]
        lambda_pur = np.sort(spectrum_pur)[::-1]

        max_dim = max(lambda_orig.size, lambda_pur.size)

        if max_dim == 0:
            raise QuantumMajorizationViolation(
                "No se puede auditar majorización sobre espectros vacíos."
            )

        lambda_orig_pad = np.pad(lambda_orig, (0, max_dim - lambda_orig.size))
        lambda_pur_pad = np.pad(lambda_pur, (0, max_dim - lambda_pur.size))

        cumulative_orig = np.cumsum(lambda_orig_pad)
        cumulative_pur = np.cumsum(lambda_pur_pad)

        deviations = cumulative_orig - cumulative_pur

        raw_max_deviation = float(np.max(deviations)) if deviations.size > 0 else 0.0
        max_deviation = max(0.0, raw_max_deviation)

        majorization_tolerance = max(
            _MAJORIZATION_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(1.0, float(max_dim)),
        )

        total_trace_deviation = float(abs(cumulative_orig[-1] - cumulative_pur[-1]))

        if total_trace_deviation > majorization_tolerance:
            raise QuantumMajorizationViolation(
                "Violación de conservación de traza en majorización. "
                f"|Tr(ρ_orig) - Tr(ρ_pur)| = {total_trace_deviation:.6e} > "
                f"{majorization_tolerance:.6e}."
            )

        if max_deviation > majorization_tolerance:
            raise QuantumMajorizationViolation(
                "Violación del preorden de majorización cuántica. "
                f"El minimizador degradó la matriz atómica de conocimiento. "
                f"Desviación máxima = {max_deviation:.6e} > "
                f"{majorization_tolerance:.6e}."
            )

        return MajorizationAuditData(
            dimension=int(max_dim),
            trace_original=float(trace_orig),
            trace_purified=float(trace_pur),
            min_eigenvalue_original=float(min_eval_orig),
            min_eigenvalue_purified=float(min_eval_pur),
            max_deviation=float(max_deviation),
            majorization_tolerance=float(majorization_tolerance),
            is_majorized=True,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2: CERTIFICACIÓN DE FIDELIDAD DE UHLMANN                             ║
# ║                                                                             ║
# ║   Computa:                                                                  ║
# ║       F(ρ, σ) = (Tr sqrt(sqrt(ρ) σ sqrt(ρ)))² ≥ F_min                       ║
# ║                                                                             ║
# ║   Esta fase comienza consumiendo el certificado de Fase 1.                  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_UhlmannFidelityCertifier(Phase1_QuantumMajorizationAuditor):
    r"""
    Asegura que la poda no mutile el conocimiento central.

    La matriz reducida debe mantener proximidad geométrica con la original en
    el espacio de densidad cuántica.

    Esta fase hereda de Fase 1 y su primer método recibe explícitamente el
    certificado de majorización emitido por:

        Phase1_QuantumMajorizationAuditor._audit_quantum_majorization(...)

    De este modo, la Fase 2 no es autónoma: está anidada funcionalmente en la
    Fase 1.
    """

    def _psd_square_root(
        self,
        name: str,
        rho: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        r"""
        Calcula la raíz cuadrada matricial de una matriz de densidad PSD.

        Usa diagonalización hermítica:

            ρ = U diag(λ) U†,
            sqrt(ρ) = U diag(sqrt(λ)) U†.
        """
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(rho)
        except np.linalg.LinAlgError as exc:
            raise DensityMatrixValidationError(
                f"Diagonalización hermítica de {name} falló al calcular sqrt(ρ)."
            ) from exc

        eigenvalues = np.asarray(eigenvalues, dtype=np.float64)

        if not np.all(np.isfinite(eigenvalues)):
            raise DensityMatrixValidationError(
                f"Los autovalores de {name} no son finitos al calcular sqrt(ρ)."
            )

        psd_tolerance = max(
            _PSD_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON,
        )

        min_eigenvalue = float(np.min(eigenvalues)) if eigenvalues.size > 0 else 0.0

        if min_eigenvalue < -psd_tolerance:
            raise DensityMatrixValidationError(
                f"{name} no es PSD al calcular sqrt(ρ). "
                f"Autovalor mínimo = {min_eigenvalue:.6e}."
            )

        sqrt_eigenvalues = np.sqrt(np.clip(eigenvalues, 0.0, None))
        sqrt_rho = (eigenvectors * sqrt_eigenvalues) @ eigenvectors.conj().T
        sqrt_rho = (sqrt_rho + sqrt_rho.conj().T) / 2.0

        return sqrt_rho

    def _certify_uhlmann_fidelity_bound(
        self,
        rho_orig: NDArray[np.complex128],
        rho_purified: NDArray[np.complex128],
        majorization_audit: Optional[MajorizationAuditData] = None,
    ) -> FidelityAuditData:
        r"""
        Primer método de la Fase 2.

        Continuación formal del último método de Fase 1.

        Calcula la fidelidad cuántica de Uhlmann entre dos estados hermíticos y
        positive semidefinite:

            F(ρ, σ) = (Tr sqrt(sqrt(ρ) σ sqrt(ρ)))².

        Si `majorization_audit` es provisto:
            - Verifica que la Fase 1 haya certificado majorización.
            - Exige consistencia dimensional con el certificado.

        Retorna:
            FidelityAuditData, certificado que sirve como objeto inicial de
            la Fase 3.
        """
        if majorization_audit is not None:
            if not majorization_audit.is_majorized:
                raise QuantumMajorizationViolation(
                    "La Fase 2 no puede iniciarse: la Fase 1 no certificó majorización."
                )

        rho_original, _, _, _ = self._sanitize_density_matrix("rho_orig", rho_orig)
        rho_purified_sanitized, _, _, _ = self._sanitize_density_matrix(
            "rho_purified",
            rho_purified,
        )

        if rho_original.shape != rho_purified_sanitized.shape:
            raise DensityMatrixValidationError(
                "rho_orig y rho_purified deben tener la misma dimensión."
            )

        if majorization_audit is not None:
            if majorization_audit.dimension != rho_original.shape[0]:
                raise ValueError(
                    "El certificado de majorización no coincide con la dimensión "
                    f"de las matrices de densidad. Certificado dim="
                    f"{majorization_audit.dimension}, matrices dim="
                    f"{rho_original.shape[0]}."
                )

        sqrt_rho_original = self._psd_square_root("rho_orig", rho_original)

        core_matrix = sqrt_rho_original @ rho_purified_sanitized @ sqrt_rho_original
        core_matrix = (core_matrix + core_matrix.conj().T) / 2.0

        try:
            core_eigenvalues = np.linalg.eigvalsh(core_matrix)
        except np.linalg.LinAlgError as exc:
            raise UhlmannFidelityCollapseError(
                "No se pudo diagonalizar el núcleo de fidelidad de Uhlmann."
            ) from exc

        core_eigenvalues = np.asarray(core_eigenvalues, dtype=np.float64)

        if not np.all(np.isfinite(core_eigenvalues)):
            raise UhlmannFidelityCollapseError(
                "El espectro del núcleo de fidelidad contiene valores no finitos."
            )

        core_eigenvalues = np.clip(core_eigenvalues, 0.0, None)

        trace_sqrt_core = float(np.sum(np.sqrt(core_eigenvalues)))
        fidelity = float(trace_sqrt_core * trace_sqrt_core)

        if not math.isfinite(fidelity):
            raise UhlmannFidelityCollapseError(
                "La fidelidad de Uhlmann no es finita."
            )

        fidelity_tolerance = max(
            _FIDELITY_NUMERICAL_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON,
        )

        upper_numerical_tolerance = max(1e-10, 10.0 * fidelity_tolerance)

        if fidelity < -fidelity_tolerance:
            raise DensityMatrixValidationError(
                f"Fidelidad negativa numéricamente imposible: F = {fidelity:.6e}."
            )

        if fidelity > 1.0 + upper_numerical_tolerance:
            raise DensityMatrixValidationError(
                f"Fidelidad mayor que 1 fuera de tolerancia numérica: F = {fidelity:.6e}."
            )

        fidelity = float(np.clip(fidelity, 0.0, 1.0))

        if fidelity < _UHLMANN_FIDELITY_MIN:
            raise UhlmannFidelityCollapseError(
                "Colapso semántico detectado. "
                f"Fidelidad de Uhlmann post-poda = {fidelity:.6f} < "
                f"F_min = {_UHLMANN_FIDELITY_MIN:.6f}. "
                "Se mutilaron ramas lógicas esenciales."
            )

        return FidelityAuditData(
            uhlmann_fidelity=fidelity,
            fidelity_tolerance=float(fidelity_tolerance),
            fidelity_min_required=float(_UHLMANN_FIDELITY_MIN),
            is_fidelity_preserved=True,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3: COTA DE CAPACIDAD DE HOLEVO Y ENTROPÍA                            ║
# ║                                                                             ║
# ║   Exige:                                                                    ║
# ║       ΔS = S(ρ_pur) - S(ρ_orig) ≤ 0                                         ║
# ║                                                                             ║
# ║   Esta fase comienza consumiendo el certificado de Fase 2.                  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_HolevoCapacityEnforcer(Phase2_UhlmannFidelityCertifier):
    r"""
    La ecuación maestra de Lindblad optimizada no puede inflar la entropía del
    sistema, de lo contrario la capacidad de Holevo del canal se desploma.

    Esta fase hereda de Fase 2 y su primer método recibe explícitamente el
    certificado de fidelidad emitido por:

        Phase2_UhlmannFidelityCertifier._certify_uhlmann_fidelity_bound(...)

    De este modo, la Fase 3 está anidada funcionalmente en la Fase 2.
    """

    @staticmethod
    def _von_neumann_entropy(spectrum: NDArray[np.float64]) -> float:
        r"""
        Calcula la entropía de von Neumann:

            S(ρ) = -Tr(ρ log ρ) = -Σ λ_i log λ_i.

        Usa logaritmo natural (nats).
        """
        probabilities = np.clip(spectrum, 0.0, 1.0)
        probabilities = probabilities[probabilities > 0.0]

        if probabilities.size == 0:
            return 0.0

        entropy = float(-np.sum(probabilities * np.log(probabilities)))

        if not math.isfinite(entropy):
            raise HolevoCapacityDeficitError(
                "La entropía de von Neumann no es finita."
            )

        return entropy

    def _enforce_holevo_capacity_retention(
        self,
        evals_orig: Optional[NDArray[np.float64]] = None,
        evals_purified: Optional[NDArray[np.float64]] = None,
        *,
        rho_orig: Optional[NDArray[np.complex128]] = None,
        rho_purified: Optional[NDArray[np.complex128]] = None,
        fidelity_audit: Optional[FidelityAuditData] = None,
    ) -> HolevoAuditData:
        r"""
        Primer método de la Fase 3.

        Continuación formal de Fase 2.

        Calcula el diferencial de entropía de von Neumann:

            ΔS = S(ρ_pur) - S(ρ_orig).

        Si `fidelity_audit` es provisto:
            - Verifica que la Fase 2 haya preservado la fidelidad.

        Retorna:
            HolevoAuditData, certificado termodinámico final.
        """
        if fidelity_audit is not None:
            if not fidelity_audit.is_fidelity_preserved:
                raise UhlmannFidelityCollapseError(
                    "La Fase 3 no puede iniciarse: la Fase 2 no preservó la fidelidad."
                )

        spectrum_orig, _, _ = self._resolve_spectrum(
            "rho_orig",
            evals_orig,
            rho_orig,
        )

        spectrum_pur, _, _ = self._resolve_spectrum(
            "rho_purified",
            evals_purified,
            rho_purified,
        )

        entropy_orig = self._von_neumann_entropy(spectrum_orig)
        entropy_pur = self._von_neumann_entropy(spectrum_pur)

        delta_s = float(entropy_pur - entropy_orig)

        if not math.isfinite(delta_s):
            raise HolevoCapacityDeficitError(
                "El diferencial de entropía ΔS no es finito."
            )

        entropy_tolerance = max(
            _ENTROPY_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, abs(entropy_orig), abs(entropy_pur)),
        )

        if delta_s > entropy_tolerance:
            raise HolevoCapacityDeficitError(
                "Paradoja termodinámica detectada. "
                f"La poda espectral inyectó entropía al canal: "
                f"ΔS = {delta_s:.6e} > {entropy_tolerance:.6e}. "
                "La capacidad de Holevo colapsa."
            )

        return HolevoAuditData(
            entropy_original=float(entropy_orig),
            entropy_purified=float(entropy_pur),
            entropy_delta=float(delta_s),
            entropy_tolerance=float(entropy_tolerance),
            is_capacity_preserved=True,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: MAC MINIMIZER AGENT                                  ║
# ║                                                                             ║
# ║   Endofuntor Z_MAC-Agent = Φ₃ ∘ Φ₂ ∘ Φ₁                                    ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class MACMinimizerAgent(Morphism, Phase3_HolevoCapacityEnforcer):
    r"""
    El Custodio de la Purificación Espectral.

    Gobierna el módulo `mac_minimizer.py`, garantizando que la compresión del
    operador de densidad respete axiomáticamente la termodinámica de von Neumann
    y el isomorfismo estructural de la información generativa del LLM.
    """

    def execute_spectral_purification_governance(
        self,
        rho_orig: NDArray[np.complex128],
        rho_purified: NDArray[np.complex128],
        evals_orig: Optional[NDArray[np.float64]] = None,
        evals_purified: Optional[NDArray[np.float64]] = None,
    ) -> PurificationGovernanceState:
        r"""
        Ejecuta la composición funtorial estricta:

            Φ₁ : Auditoría de majorización cuántica.
            Φ₂ : Certificación de fidelidad de Uhlmann.
            Φ₃ : Cota de capacidad de Holevo y entropía de von Neumann.

        Parámetros:
            rho_orig:
                Matriz de densidad original ρ.

            rho_purified:
                Matriz de densidad purificada σ.

            evals_orig:
                Autovalores opcionales de ρ. Si se proveen, se validan contra ρ.

            evals_purified:
                Autovalores opcionales de σ. Si se proveen, se validan contra σ.

        Retorna:
            PurificationGovernanceState con los tres certificados y validez
            epistemológica final.
        """
        # Fase 1: Certificar majorización (el espectro debe apuntar a la pureza).
        majorization_audit = self._audit_quantum_majorization(
            evals_orig,
            evals_purified,
            rho_orig=rho_orig,
            rho_purified=rho_purified,
        )

        # Fase 2: Certificar fidelidad de Uhlmann (preservación de isomorfismo).
        fidelity_audit = self._certify_uhlmann_fidelity_bound(
            rho_orig,
            rho_purified,
            majorization_audit=majorization_audit,
        )

        # Fase 3: Certificar termodinámica y retención de capacidad de Holevo.
        holevo_audit = self._enforce_holevo_capacity_retention(
            evals_orig,
            evals_purified,
            rho_orig=rho_orig,
            rho_purified=rho_purified,
            fidelity_audit=fidelity_audit,
        )

        is_epistemologically_valid = bool(
            majorization_audit.is_majorized
            and fidelity_audit.is_fidelity_preserved
            and holevo_audit.is_capacity_preserved
        )

        if not is_epistemologically_valid:
            raise MACMinimizerAgentError(
                "La composición funtorial no autorizó la purificación espectral."
            )

        logger.info(
            "Gobernanza de Purificación Cuántica (MAC) certificada. "
            "Majorización conservada | "
            "Fidelidad Uhlmann: %.6f | "
            "ΔS: %.6e nats",
            fidelity_audit.uhlmann_fidelity,
            holevo_audit.entropy_delta,
        )

        return PurificationGovernanceState(
            majorization_audit=majorization_audit,
            fidelity_audit=fidelity_audit,
            holevo_audit=holevo_audit,
            is_epistemologically_valid=is_epistemologically_valid,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ═══════════════════════════════════════════════════════════════════════════════
__all__: List[str] = [
    "MACMinimizerAgentError",
    "DensityMatrixValidationError",
    "QuantumMajorizationViolation",
    "UhlmannFidelityCollapseError",
    "HolevoCapacityDeficitError",
    "MajorizationAuditData",
    "FidelityAuditData",
    "HolevoAuditData",
    "PurificationGovernanceState",
    "Phase1_QuantumMajorizationAuditor",
    "Phase2_UhlmannFidelityCertifier",
    "Phase3_HolevoCapacityEnforcer",
    "MACMinimizerAgent",
]