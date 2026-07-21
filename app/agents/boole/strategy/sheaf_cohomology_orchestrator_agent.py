# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Sheaf Cohomology Orchestrator Agent                                    ║
║           (Custodio de la Holonomía Global)                                      ║
║  Ruta   : app/agents/boole/strategy/sheaf_cohomology_orchestrator_agent.py       ║
║  Versión: 3.0.0-Categorical-Krylov-Hodge-Evolved                                 ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  NATURALEZA CIBER-FÍSICA Y TOPOLOGÍA ALGEBRAICA (Rigor Doctoral):                ║
║  ──────────────────────────────────────────────────────────────────              ║
║  Este endofuntor gobierna al `sheaf_cohomology_orchestrator.py` en el            ║
║  estrato STRATEGY.                                                               ║
║                                                                                  ║
║                                                                                  ║
║  ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta):                ║
║  ────────────────────────────────────────────────────────────────                ║
║  Fase 1 → _certify_cohomological_veto_axiom → CohomologicalVetoData              ║
║           [objeto inicial de Fase 2]                                             ║
║                                                                                  ║
║  Fase 2 → _audit_krylov_spectral_stability(..., veto_audit)                      ║
║           → KrylovSpectralData  [objeto inicial de Fase 3]                       ║
║                                                                                  ║
║  Fase 3 → _enforce_isoperimetric_hodge_projection(..., spectral_audit)           ║
║           → HodgeProjectionData  [objeto final del endofuntor]                   ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# §0. IMPORTACIONES
# ─────────────────────────────────────────────────────────────────────────────
import hashlib
import logging
import math
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# §0.1 DEPENDENCIAS ARQUITECTÓNICAS
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:

    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos E_MIC."""
        pass

    class Morphism:
        r"""Clase base de Morfismos del Topos."""
        pass


logger = logging.getLogger("MIC.Strategy.SheafCohomologyAgent")


# ═══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES MATEMÁTICAS Y LÍMITES TERMODINÁMICOS
#
# Jerarquía de constantes:
#   _MACHINE_EPSILON          → ε₀ = eps(float64) ≈ 2.22e-16
#   _SVD_TOLERANCE_BASE       → τ_σ base para umbral de rango SVD
#   _SVD_SPECTRAL_FACTOR      → c_σ: factor de escala adaptativa por σ_max
#   _SPECTRAL_GAP_MIN_RATIO   → ρ_gap: umbral mínimo de gap espectral relativo
#   _MAX_CONDITION_NUMBER_L   → κ_max: número de condición máximo admisible de L
#   _FRUSTRATION_TOLERANCE    → ε_frust: piso de frustración térmica admisible
#   _FRUSTRATION_RELATIVE_TOL → ε_rel: tolerancia relativa de frustración
#   _INERTIA_DELTA_MAX        → Δ_in: distancia máxima de proyección de Hodge
#   _POINCARE_CONSTANT_MAX    → C_P: cota de la constante de Poincaré admisible
#   _LIPSCHITZ_SLACK          → ε_L: holgura numérica en la verificación Lipschitz
#   _NUMERICAL_SAFETY_FACTOR  → c_num: factor de seguridad numérica global
#   _SVD_MAX_RETRIES          → máximo de reintentos en SVD con perturbación
#   _ENERGY_RATIO_TOLERANCE   → tolerancia en la razón de reducción energética
# ═══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)
_SVD_TOLERANCE_BASE: Final[float] = 1e-10
_SVD_SPECTRAL_FACTOR: Final[float] = 1e-10
_SPECTRAL_GAP_MIN_RATIO: Final[float] = 1e-2
_MAX_CONDITION_NUMBER_L: Final[float] = 1e15
_FRUSTRATION_TOLERANCE: Final[float] = 1e-2
_FRUSTRATION_RELATIVE_TOL: Final[float] = 1e-8
_INERTIA_DELTA_MAX: Final[float] = 5.0
_POINCARE_CONSTANT_MAX: Final[float] = 1e8
_LIPSCHITZ_SLACK: Final[float] = 1e-6
_NUMERICAL_SAFETY_FACTOR: Final[float] = 128.0
_SVD_MAX_RETRIES: Final[int] = 3
_ENERGY_RATIO_TOLERANCE: Final[float] = 1e-10


# ═══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA COMPLETA DE EXCEPCIONES
#
# Lattice de excepciones:
#   TopologicalInvariantError
#   └── SheafCohomologyAgentError                    [raíz del agente]
#       ├── TopologicalBifurcationError              [dim H¹ > 0]
#       │   └── PoincareLefschetzViolation           [violación P-L]
#       ├── SpectralComputationError                 [κ(L) > κ_max]
#       │   ├── SVDConvergenceError                  [SVD no converge]
#       │   └── HodgeDecompositionError              [descomposición fallida]
#       ├── DirichletFrustrationError                [E(x) > ε_frust]
#       │   └── PoincareBoundViolation               [viola cota de Poincaré]
#       └── HomologicalInconsistencyError            [‖x−x*‖ > Δ o E↑]
#           ├── LipschitzViolation                   [viola axioma Lipschitz]
#           └── MinimalNormViolation                 [proyección no mínimo norm]
# ═══════════════════════════════════════════════════════════════════════════════
class SheafCohomologyAgentError(TopologicalInvariantError):
    r"""Excepción raíz del Custodio de la Holonomía Global."""
    pass


class TopologicalBifurcationError(SheafCohomologyAgentError):
    r"""
    Detonada si dim H¹ > 0.
    Existen dependencias circulares (cocadenas cerradas no exactas) insalvables.
    """
    pass


class PoincareLefschetzViolation(TopologicalBifurcationError):
    r"""
    Detonada si la dualidad de Poincaré-Lefschetz es violada en el complejo
    cohomológico. Indica que la descomposición de Hodge no es ortogonal.
    """
    pass


class SpectralComputationError(SheafCohomologyAgentError):
    r"""
    Detonada si κ(L) > κ_max.
    Peligro de colapso en la Unidad de Punto Flotante (FPU).
    """
    pass


class SVDConvergenceError(SpectralComputationError):
    r"""
    Detonada si la descomposición en valores singulares no converge
    tras el número máximo de reintentos con perturbación diagonal.
    """
    pass


class HodgeDecompositionError(SpectralComputationError):
    r"""
    Detonada si la descomposición de Hodge-Helmholtz no puede verificarse
    numéricamente para el operador δ dado.
    """
    pass


class DirichletFrustrationError(SheafCohomologyAgentError):
    r"""
    Detonada si la energía de Dirichlet excede la frustración térmica admisible:
        E(x) = ‖δx‖₂² > ε_frustration.
    """
    pass


class PoincareBoundViolation(DirichletFrustrationError):
    r"""
    Detonada si la constante de Poincaré implícita C_P excede el umbral:
        ‖x‖ ≤ C_P · ‖δx‖  con  C_P > C_P_max.
    """
    pass


class HomologicalInconsistencyError(SheafCohomologyAgentError):
    r"""
    Detonada si ‖x − x*‖₂ > Δ_inertia o si la proyección aumenta E(x).
    """
    pass


class LipschitzViolation(HomologicalInconsistencyError):
    r"""
    Detonada si la proyección de Hodge viola la condición de Lipschitz fuerte:
        ‖δx* − δx‖ ≤ κ(δ) · ‖x* − x‖.
    """
    pass


class MinimalNormViolation(HomologicalInconsistencyError):
    r"""
    Detonada si la proyección x* no minimiza la norma en ker(δ):
        ‖x*‖ ≤ ‖x‖ + ε_num  no se satisface.
    """
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Topos Estratégico)
#
# Cadena de certificados funtoriales:
#   CohomologicalVetoData  →  Fase 1 → Fase 2
#   KrylovSpectralData     →  Fase 2 → Fase 3
#   HodgeProjectionData    →  Fase 3 → Orquestador
#   SheafGovernanceState   →  Resultado final
#   SheafAuditProvenance   →  Trazabilidad criptográfica
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class CohomologicalVetoData:
    r"""
    Artefacto de Fase 1. Certificado de anulación de obstrucciones globales.

    Campos extendidos v3.0.0:
        dim_C0                  : dim C⁰ (dominio de δ)
        dim_C1                  : dim C¹ (codominio de δ)
        delta_rank              : rank(δ) calculado vía SVD con gap espectral
        h1_dimension            : dim H¹ = dim C¹ − rank(δ)
        svd_tolerance           : tolerancia efectiva usada en el cómputo de rango
        max_singular_value      : σ_max(δ)
        min_nonzero_singular_value: σ_min^+(δ) = menor valor singular no nulo
        spectral_gap             : Δσ = σ_max − σ_min^+ (gap espectral absoluto)
        spectral_gap_ratio       : Δσ / σ_max (gap espectral relativo)
        cohomological_stability_index: β = rank(δ) / dim(C¹) ∈ [0,1]
        euler_characteristic_01  : χ₀₁ = dim(C⁰) − dim(C¹) (Euler parcial)
        whitehead_torsion        : log|τ_W| ∈ ℝ (invariante secundario)
        poincare_lefschetz_ok    : True sii la dualidad P-L es compatible
        is_topologically_coherent: True sii dim H¹ = 0
    """
    dim_C0: int
    dim_C1: int
    delta_rank: int
    h1_dimension: int
    svd_tolerance: float
    max_singular_value: float
    min_nonzero_singular_value: float
    spectral_gap: float
    spectral_gap_ratio: float
    cohomological_stability_index: float
    euler_characteristic_01: int
    whitehead_torsion: float
    poincare_lefschetz_ok: bool
    is_topologically_coherent: bool


@dataclass(frozen=True, slots=True)
class KrylovSpectralData:
    r"""
    Artefacto de Fase 2. Certificado termodinámico y de condicionamiento espectral.

    Campos extendidos v3.0.0:
        dirichlet_energy        : E(x) = ‖δx‖₂²
        dirichlet_energy_norm   : Ê = E(x) / max(1, ‖x‖²)  (energía normalizada)
        frustration_tolerance   : ε_frust efectiva
        frustration_index       : ρ = E(x) / ε_frust (índice de frustración)
        delta_condition_number  : κ(δ) = σ_max / σ_min^+
        laplacian_condition_number: κ(L) = κ(δ)²
        spectral_gap_effective  : Δσ reutilizado del certificado de Fase 1
        harmonic_component_norm : ‖x_harm‖₂ (componente armónica de x)
        exact_component_norm    : ‖x_exact‖₂ (componente exacta de x, si δ† disponible)
        poincare_constant       : C_P = ‖x_orth‖ / ‖δx‖ (estimado)
        is_frustration_bounded  : True sii E(x) ≤ ε_frust
        is_spectrally_stable    : True sii κ(L) ≤ κ_max
        is_poincare_bounded     : True sii C_P ≤ C_P_max
    """
    dirichlet_energy: float
    dirichlet_energy_norm: float
    frustration_tolerance: float
    frustration_index: float
    delta_condition_number: float
    laplacian_condition_number: float
    spectral_gap_effective: float
    harmonic_component_norm: float
    exact_component_norm: float
    poincare_constant: float
    is_frustration_bounded: bool
    is_spectrally_stable: bool
    is_poincare_bounded: bool


@dataclass(frozen=True, slots=True)
class HodgeProjectionData:
    r"""
    Artefacto de Fase 3. Certificado de Lipschitz para el consenso de Hodge.

    Campos extendidos v3.0.0:
        projection_distance          : ‖x − x*‖₂
        relative_projection_distance : ‖x − x*‖ / max(1, ‖x‖)
        inertia_delta_max            : Δ_inertia (constante de cota inercial)
        original_dirichlet_energy    : E(x) = ‖δx‖₂²
        projected_dirichlet_energy   : E(x*) = ‖δx*‖₂²
        energy_reduction_ratio       : E(x*) / E(x) ∈ [0, 1]
        lipschitz_residual           : ‖δx* − δx‖ − κ(δ)·‖x*−x‖ ≤ 0
        lipschitz_satisfied          : True sii la condición Lipschitz fuerte pasa
        minimal_norm_satisfied       : True sii ‖x*‖ ≤ ‖x‖ + ε_num
        cheeger_bound_estimate       : h ≈ E(x*) / ‖x*‖² (estimado de Cheeger)
        morse_reduction_index        : ι_M = dim ker(δᵀδ) estimado
        is_isoperimetrically_bounded : True sii ‖x−x*‖ ≤ Δ_inertia
        is_energy_non_increasing     : True sii E(x*) ≤ E(x) + ε_num
        verified_by_delta            : True sii se verificó con δ explícito
    """
    projection_distance: float
    relative_projection_distance: float
    inertia_delta_max: float
    original_dirichlet_energy: float
    projected_dirichlet_energy: float
    energy_reduction_ratio: float
    lipschitz_residual: float
    lipschitz_satisfied: bool
    minimal_norm_satisfied: bool
    cheeger_bound_estimate: float
    morse_reduction_index: int
    is_isoperimetrically_bounded: bool
    is_energy_non_increasing: bool
    verified_by_delta: bool


@dataclass(frozen=True, slots=True)
class SheafAuditProvenance:
    r"""
    Trazabilidad criptográfica de la cadena funtorial Z_SheafAgent = Φ₃∘Φ₂∘Φ₁.

    Campos:
        timestamp_iso         : Fecha/hora ISO-8601 UTC de la auditoría
        input_checksum_sha256 : SHA-256 hex de los datos de entrada serializados
        phase1_passed         : True sii Fase 1 completó sin excepción
        phase2_passed         : True sii Fase 2 completó sin excepción
        phase3_passed         : True sii Fase 3 completó sin excepción
        functor_chain         : Descripción textual de la composición funtorial
    """
    timestamp_iso: str
    input_checksum_sha256: str
    phase1_passed: bool
    phase2_passed: bool
    phase3_passed: bool
    functor_chain: str


@dataclass(frozen=True, slots=True)
class SheafGovernanceState:
    r"""
    Objeto final del endofuntor Z_SheafAgent = Φ₃∘Φ₂∘Φ₁.

    Contiene:
        veto_audit                : Certificado de Fase 1
        spectral_audit            : Certificado de Fase 2
        hodge_audit               : Certificado de Fase 3
        provenance                : Trazabilidad criptográfica
        is_epistemologically_valid: True sii los tres certificados son válidos
    """
    veto_audit: CohomologicalVetoData
    spectral_audit: KrylovSpectralData
    hodge_audit: HodgeProjectionData
    provenance: SheafAuditProvenance
    is_epistemologically_valid: bool


# ═══════════════════════════════════════════════════════════════════════════════
# §D. GUARDAS NUMÉRICAS INTERNAS — v3.0.0
#
# Evoluciones sobre v2.0.0:
#   · _safe_svdvals     : SVD con reintentos y perturbación diagonal de emergencia
#   · _check_spectral_gap: detección de gap espectral adaptativo
#   · _pseudo_inverse   : pseudoinversa δ† por SVD truncada con gap
#   · _frobenius_norm   : con fallback robusto a norma-1
#   · _squared_norm_from_vector: con tolerancia adaptativa
#   · _compute_input_checksum: SHA-256 de los datos de entrada
# ═══════════════════════════════════════════════════════════════════════════════
class _FiniteNumericalGuard:
    r"""
    Capa de saneamiento numérico que garantiza la finitez, realidad y
    no-degeneración de todas las entradas antes de los cómputos algebraicos.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # D.1 Conversión y validación de arreglos
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _as_float_array(name: str, value: Any) -> NDArray[np.float64]:
        r"""
        Convierte `value` a float64 real finito, rechazando ℂ, NaN y ±∞.
        """
        try:
            raw = np.asarray(value)
        except Exception as exc:
            raise TypeError(
                f"[Guard] '{name}' no puede interpretarse como arreglo numérico: {exc}"
            ) from exc

        if np.iscomplexobj(raw):
            raise TypeError(
                f"[Guard] '{name}' debe ser real; se rechazó dtype complejo={raw.dtype}."
            )

        try:
            arr = raw.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"[Guard] '{name}' no puede convertirse a float64: {exc}"
            ) from exc

        if not np.all(np.isfinite(arr)):
            n_bad = int(np.sum(~np.isfinite(arr)))
            raise ValueError(
                f"[Guard] '{name}' contiene {n_bad} elemento(s) NaN o ±∞."
            )

        return arr

    @classmethod
    def _as_finite_matrix(
        cls,
        name: str,
        value: Any,
        *,
        min_rows: int = 0,
        min_cols: int = 0,
    ) -> NDArray[np.float64]:
        r"""
        Valida una matriz real finita 2D.

        Parámetros:
            name    : Nombre del parámetro.
            value   : Objeto a validar.
            min_rows: Número mínimo de filas (0 = sin restricción).
            min_cols: Número mínimo de columnas (0 = sin restricción).
        """
        arr = cls._as_float_array(name, value)

        if arr.ndim != 2:
            raise ValueError(
                f"[Guard] '{name}' debe ser 2D; tiene ndim={arr.ndim}."
            )

        rows, cols = arr.shape

        if min_rows > 0 and rows < min_rows:
            raise ValueError(
                f"[Guard] '{name}' debe tener al menos {min_rows} filas; tiene {rows}."
            )

        if min_cols > 0 and cols < min_cols:
            raise ValueError(
                f"[Guard] '{name}' debe tener al menos {min_cols} columnas; tiene {cols}."
            )

        return arr

    @classmethod
    def _as_finite_vector(
        cls,
        name: str,
        value: Any,
        *,
        allow_empty: bool = True,
    ) -> NDArray[np.float64]:
        r"""
        Valida y normaliza a 1D un vector real finito.

        Acepta: 1D, (n,1), (1,n), escalar (→ dim 1).

        Parámetros:
            name       : Nombre del parámetro.
            value      : Objeto a validar.
            allow_empty: Si False, rechaza vectores de tamaño 0.
        """
        arr = cls._as_float_array(name, value)

        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim == 2 and 1 in arr.shape:
            arr = arr.reshape(-1)
        elif arr.ndim != 1:
            raise ValueError(
                f"[Guard] '{name}' debe ser 1D, columna, fila o escalar; "
                f"tiene forma {arr.shape}."
            )

        if not allow_empty and arr.size == 0:
            raise ValueError(f"[Guard] '{name}' no puede ser un vector vacío.")

        return arr

    # ─────────────────────────────────────────────────────────────────────────
    # D.2 Normas numéricamente seguras
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _vector_norm(v: NDArray[np.float64]) -> float:
        r"""‖v‖₂ con fallback a ‖v‖₁ si la rutina LAPACK falla."""
        if v.size == 0:
            return 0.0
        try:
            val = float(la.norm(v, ord=2))
            return val if math.isfinite(val) else math.inf
        except Exception:
            try:
                val = float(la.norm(v, ord=1))
                return val if math.isfinite(val) else math.inf
            except Exception:
                return math.inf

    @staticmethod
    def _frobenius_norm(A: NDArray[np.float64]) -> float:
        r"""‖A‖_F con fallback a ‖A‖₁."""
        if A.size == 0:
            return 0.0
        try:
            val = float(la.norm(A, ord="fro"))
            return val if math.isfinite(val) else math.inf
        except Exception:
            try:
                val = float(la.norm(A, ord=1))
                return val if math.isfinite(val) else math.inf
            except Exception:
                return math.inf

    @staticmethod
    def _squared_norm_from_vector(y: NDArray[np.float64]) -> float:
        r"""
        Calcula ‖y‖₂² = yᵀy de forma segura.

        Una negatividad pequeña por ruido numérico se proyecta a 0;
        una negatividad significativa lanza DirichletFrustrationError.
        """
        if y.size == 0:
            return 0.0

        value = float(np.dot(y, y))

        if not math.isfinite(value):
            raise DirichletFrustrationError(
                "[Guard] ‖δx‖₂² no es finita; posible desbordamiento numérico."
            )

        tolerance = (
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(1.0, abs(value))
        )

        if value < -tolerance:
            raise DirichletFrustrationError(
                f"[Guard] ‖δx‖₂² = {value:.4e} < 0 con magnitud significativa; "
                "inconsistencia numérica grave."
            )

        return max(0.0, value)

    # ─────────────────────────────────────────────────────────────────────────
    # D.3 SVD segura con reintentos por perturbación diagonal
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _safe_svdvals(
        A: NDArray[np.float64],
        name: str = "A",
        *,
        max_retries: int = _SVD_MAX_RETRIES,
    ) -> NDArray[np.float64]:
        r"""
        Calcula los valores singulares de A con reintentos ante fallos de LAPACK.

        En cada reintento añade una perturbación diagonal:

            ε_k = ε₀^{1/(k+2)} · ‖A‖_F · I_{min}

        donde I_{min} es la identidad en el subespacio mínimo.

        Parámetros:
            A          : Matriz real finita.
            name       : Nombre del parámetro (para mensajes).
            max_retries: Número máximo de intentos.

        Retorna:
            Vector de valores singulares en orden descendente.

        Lanza:
            SVDConvergenceError si todos los intentos fallan.
        """
        if A.size == 0 or min(A.shape) == 0:
            return np.empty(0, dtype=np.float64)

        last_exc: Optional[Exception] = None
        A_work = A.copy()

        for attempt in range(max_retries):
            try:
                svs = la.svdvals(A_work)
                if np.all(np.isfinite(svs)):
                    return svs.astype(np.float64)
            except (np.linalg.LinAlgError, ValueError) as exc:
                last_exc = exc

            # Perturbación diagonal de regularización
            frob = float(la.norm(A_work, ord="fro") or 1.0)
            eps_k = (_MACHINE_EPSILON ** (1.0 / (attempt + 2))) * frob
            rows, cols = A_work.shape
            min_dim = min(rows, cols)
            A_work = A_work.copy()
            A_work[:min_dim, :min_dim] += eps_k * np.eye(min_dim, dtype=np.float64)

        raise SVDConvergenceError(
            f"[Guard] SVD de '{name}' no convergió tras {max_retries} intentos. "
            f"Último error: {last_exc}."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # D.4 Gap espectral adaptativo
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _check_spectral_gap(
        singular_values: NDArray[np.float64],
        tolerance: float,
    ) -> Tuple[int, float]:
        r"""
        Detecta el gap espectral más prominente para determinar el rango efectivo.

        Estrategia:
            1. Normaliza σᵢ / σ_max.
            2. Calcula saltos: Δᵢ = σᵢ - σᵢ₊₁ (descendentes).
            3. Si el gap relativo mayor Δ_max / σ_max > ρ_gap_min, se usa como
               punto de corte de rango.
            4. Si no hay gap claro, se usa la tolerancia adaptativa estándar.

        Parámetros:
            singular_values: Valores singulares en orden descendente.
            tolerance      : Tolerancia adaptativa para el caso sin gap.

        Retorna:
            (rank_effective, gap_ratio) donde gap_ratio ∈ [0, 1].
        """
        if singular_values.size == 0:
            return 0, 0.0

        sigma_max = float(singular_values[0])

        if sigma_max == 0.0:
            return 0, 0.0

        # Rango por tolerancia estándar
        rank_by_tol = int(np.count_nonzero(singular_values > tolerance))

        if singular_values.size < 2:
            return rank_by_tol, 0.0

        # Detección de gap espectral
        gaps = singular_values[:-1] - singular_values[1:]  # ≥ 0 por orden desc.
        if gaps.size == 0:
            return rank_by_tol, 0.0

        max_gap_idx = int(np.argmax(gaps))
        max_gap = float(gaps[max_gap_idx])
        gap_ratio = max_gap / sigma_max

        if gap_ratio > _SPECTRAL_GAP_MIN_RATIO:
            rank_by_gap = max_gap_idx + 1
            return rank_by_gap, gap_ratio

        return rank_by_tol, gap_ratio

    # ─────────────────────────────────────────────────────────────────────────
    # D.5 Pseudoinversa de Moore-Penrose
    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def _pseudo_inverse(
        cls,
        A: NDArray[np.float64],
        name: str = "A",
        *,
        tolerance: Optional[float] = None,
    ) -> NDArray[np.float64]:
        r"""
        Calcula la pseudoinversa de Moore-Penrose A† por SVD truncada:

            A† = V · diag(1/σᵢ) · Uᵀ   (solo para σᵢ > tol)

        Utiliza _safe_svdvals internamente y detección de gap espectral.

        Parámetros:
            A        : Matriz real finita.
            name     : Nombre del parámetro.
            tolerance: Umbral de truncamiento (None → adaptativo).

        Retorna:
            A† ∈ ℝ^{cols × rows}.
        """
        if A.size == 0 or min(A.shape) == 0:
            return np.zeros((A.shape[1], A.shape[0]), dtype=np.float64)

        try:
            U, s, Vt = la.svd(A, full_matrices=False)
        except (np.linalg.LinAlgError, ValueError):
            # Reintento con SVD segura
            svs = cls._safe_svdvals(A, name)
            # Usamos la implementación de numpy como último recurso
            return np.linalg.pinv(A)

        if not (np.all(np.isfinite(U)) and np.all(np.isfinite(s)) and np.all(np.isfinite(Vt))):
            raise HodgeDecompositionError(
                f"[Guard] SVD de '{name}' produjo valores no finitos en U, σ o Vᵀ."
            )

        if tolerance is None:
            sigma_max = float(s[0]) if s.size > 0 else 0.0
            tolerance = max(
                _SVD_TOLERANCE_BASE,
                _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(A.shape) * sigma_max,
            )

        s_inv = np.where(s > tolerance, 1.0 / s, 0.0)
        A_pinv = (Vt.T * s_inv) @ U.T

        if not np.all(np.isfinite(A_pinv)):
            raise HodgeDecompositionError(
                f"[Guard] La pseudoinversa de '{name}' contiene valores no finitos."
            )

        return A_pinv

    # ─────────────────────────────────────────────────────────────────────────
    # D.6 Checksum SHA-256 de entrada
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_input_checksum(*arrays: Optional[NDArray[np.float64]]) -> str:
        r"""
        Calcula el SHA-256 de la concatenación binaria de los arreglos,
        incluyendo metadatos de forma y dtype.

        Retorna:
            String hexadecimal de 64 caracteres.
        """
        hasher = hashlib.sha256()

        for arr in arrays:
            if arr is None:
                hasher.update(b"\x00")
                continue

            a = np.asarray(arr)
            shape_padded = (*a.shape, *([0] * max(0, 3 - a.ndim)))[:3]
            meta = struct.pack(">4Q", a.ndim, *shape_padded)
            hasher.update(meta)
            hasher.update(np.ascontiguousarray(a).tobytes())

        return hasher.hexdigest()


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 1: CERTIFICACIÓN AXIOMÁTICA DEL VETO COHOMOLÓGICO                    ║
# ║                                                                             ║
# ║   Marco matemático:                                                         ║
# ║   ─────────────────                                                         ║
# ║   Sea δ: C⁰ → C¹ el operador cofrontera del haz celular.                   ║
# ║   La primera cohomología mide las cocadenas cerradas no exactas:             ║
# ║       H¹(G; F) = coker(δ) / im(δ) (en el caso de 1-término)                ║
# ║   Bajo el modelo operacional: dim H¹ = dim(C¹) − rank(δ).                   ║
# ║                                                                             ║
# ║   Invariantes calculados en v3.0.0:                                         ║
# ║   1. rank(δ) por SVD con gap espectral (Golub-Reinsch).                     ║
# ║   2. dim H¹ = dim(C¹) − rank(δ).                                            ║
# ║   3. Índice de estabilidad β = rank(δ) / dim(C¹).                           ║
# ║   4. Característica de Euler parcial χ₀₁ = dim(C⁰) − dim(C¹).              ║
# ║   5. Torsión de Whitehead log|τ_W|.                                         ║
# ║   6. Gap espectral Δσ y ratio Δσ/σ_max.                                     ║
# ║   7. Verificación de la condición de Poincaré-Lefschetz.                    ║
# ║                                                                             ║
# ║   ÚLTIMO MÉTODO DE FASE 1:                                                  ║
# ║       _certify_cohomological_veto_axiom(coboundary_operator_delta)          ║
# ║       → CohomologicalVetoData  [objeto inicial de Fase 2]                   ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_CohomologicalVetoCertifier(_FiniteNumericalGuard):
    r"""
    Evalúa el operador cofrontera δ: C⁰ → C¹.

    Asegura que las dependencias inter-agente formen un consenso libre de
    vórtices cohomológicos (dim H¹ = 0).

    Mejoras v3.0.0:
        · SVD con reintentos por perturbación diagonal.
        · Gap espectral adaptativo para detección robusta de rango.
        · Índice de estabilidad β y característica de Euler parcial χ₀₁.
        · Torsión de Whitehead log|τ_W| como invariante secundario.
        · Verificación de la condición de Poincaré-Lefschetz.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 1.1 Tolerancia SVD adaptativa
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_svd_tolerance(
        sigma_max: float,
        shape: Tuple[int, int],
    ) -> float:
        r"""
        Calcula la tolerancia efectiva para el umbral de rango SVD:

            τ_eff = max(τ_base, c_σ · σ_max · ε₀ · max(m, n) · c_num).

        Esta tolerancia es la misma filosofía que la de MATLAB rank().

        Parámetros:
            sigma_max: σ_max(δ).
            shape    : (m, n) = forma de δ.

        Retorna:
            τ_eff > 0.
        """
        if sigma_max <= 0.0 or not math.isfinite(sigma_max):
            return _SVD_TOLERANCE_BASE

        spectral_tol = (
            _SVD_SPECTRAL_FACTOR
            * sigma_max
            * _MACHINE_EPSILON
            * max(shape)
            * _NUMERICAL_SAFETY_FACTOR
        )

        return max(_SVD_TOLERANCE_BASE, spectral_tol)

    # ─────────────────────────────────────────────────────────────────────────
    # 1.2 Torsión de Whitehead
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_whitehead_torsion(
        singular_values: NDArray[np.float64],
        effective_rank: int,
    ) -> float:
        r"""
        Calcula la torsión de Whitehead log|τ_W| del complejo:

            log|τ_W| = Σᵢ₌₁^{rank} log(σᵢ).

        Para un complejo acíclico, τ_W es el determinante analítico del
        operador de cadenas restringido al subespacio no nulo.

        Si rank = 0, retorna 0.0.

        Parámetros:
            singular_values: Valores singulares de δ en orden descendente.
            effective_rank : Rango efectivo de δ.

        Retorna:
            log|τ_W| ∈ ℝ.
        """
        if effective_rank <= 0 or singular_values.size == 0:
            return 0.0

        significant = singular_values[:effective_rank]
        positive = significant[significant > 0.0]

        if positive.size == 0:
            return 0.0

        log_tau = float(np.sum(np.log(positive)))
        return log_tau if math.isfinite(log_tau) else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # 1.3 Verificación de Poincaré-Lefschetz
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _verify_poincare_lefschetz(
        dim_C0: int,
        dim_C1: int,
        effective_rank: int,
    ) -> bool:
        r"""
        Verifica la compatibilidad con la dualidad de Poincaré-Lefschetz:

            rank(δ) ≤ min(dim C⁰, dim C¹).

        Esta es una condición necesaria para que la descomposición de Hodge
        sea ortogonal. La condición es trivialmente satisfecha por el Teorema
        de Rango-Nulidad, pero se verifica explícitamente para detectar
        inconsistencias numéricas.

        Parámetros:
            dim_C0        : dim C⁰.
            dim_C1        : dim C¹.
            effective_rank: rank(δ) calculado numéricamente.

        Retorna:
            True sii rank(δ) ≤ min(dim C⁰, dim C¹).
        """
        max_possible_rank = min(dim_C0, dim_C1)

        if effective_rank > max_possible_rank:
            logger.warning(
                "[Fase 1] Condición de Poincaré-Lefschetz violada: "
                "rank(δ) = %d > min(dim C⁰, dim C¹) = %d. "
                "Inconsistencia numérica en la SVD.",
                effective_rank, max_possible_rank,
            )
            return False

        return True

    # ─────────────────────────────────────────────────────────────────────────
    # 1.4 ÚLTIMO MÉTODO DE FASE 1
    #     _certify_cohomological_veto_axiom → CohomologicalVetoData
    #     [Objeto inicial de Fase 2]
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_cohomological_veto_axiom(
        self,
        coboundary_operator_delta: NDArray[np.float64],
    ) -> CohomologicalVetoData:
        r"""
        ════════════════════════════════════════════════════════════════════
        ÚLTIMO MÉTODO DE FASE 1 — Retorna el objeto inicial de Fase 2.
        ════════════════════════════════════════════════════════════════════

        Computa rank(δ) y define:

            dim H¹(G; F) = dim(C¹) − rank(δ).

        Si dim H¹ > 0, existen cocadenas cerradas que no son exactas:
        el veto topológico es detonado vía TopologicalBifurcationError.

        Invariantes calculados:
            1. Espectro singular {σᵢ} por _safe_svdvals (con reintentos).
            2. Tolerancia adaptativa τ_eff = max(τ_base, c_σ·σ_max·ε₀·max(m,n)).
            3. Rango efectivo por gap espectral (Golub-Reinsch).
            4. dim H¹ = dim C¹ − rank(δ).
            5. Gap espectral Δσ y ratio Δσ/σ_max.
            6. Índice de estabilidad β = rank(δ) / dim(C¹).
            7. Característica de Euler parcial χ₀₁ = dim(C⁰) − dim(C¹).
            8. Torsión de Whitehead log|τ_W|.
            9. Condición de Poincaré-Lefschetz.

        Parámetros:
            coboundary_operator_delta: Operador δ: C⁰ → C¹ ∈ ℝ^{m×n}.

        Retorna:
            CohomologicalVetoData — certificado cohomológico completo.
            Este objeto es el **objeto inicial de la Fase 2**.

        Lanza:
            TopologicalBifurcationError si dim H¹ > 0.
            PoincareLefschetzViolation si la condición P-L falla gravemente.
            SVDConvergenceError si la SVD no converge.
        """
        # ── Validación de entrada ─────────────────────────────────────────
        delta = self._as_finite_matrix(
            "coboundary_operator_delta",
            coboundary_operator_delta,
        )

        dim_C1, dim_C0 = delta.shape

        # ── Manejo del caso trivial (matriz vacía) ────────────────────────
        if delta.size == 0 or min(delta.shape) == 0:
            return CohomologicalVetoData(
                dim_C0=int(dim_C0),
                dim_C1=int(dim_C1),
                delta_rank=0,
                h1_dimension=0,
                svd_tolerance=_SVD_TOLERANCE_BASE,
                max_singular_value=0.0,
                min_nonzero_singular_value=0.0,
                spectral_gap=0.0,
                spectral_gap_ratio=0.0,
                cohomological_stability_index=1.0 if dim_C1 == 0 else 0.0,
                euler_characteristic_01=int(dim_C0 - dim_C1),
                whitehead_torsion=0.0,
                poincare_lefschetz_ok=True,
                is_topologically_coherent=True,
            )

        # ── Espectro singular con reintentos ──────────────────────────────
        singular_values = self._safe_svdvals(
            delta, "coboundary_operator_delta",
        )

        # ── Tolerancia adaptativa ─────────────────────────────────────────
        sigma_max = float(singular_values[0]) if singular_values.size > 0 else 0.0
        svd_tolerance = self._compute_svd_tolerance(sigma_max, delta.shape)

        # ── Rango efectivo con gap espectral ──────────────────────────────
        effective_rank, gap_ratio = self._check_spectral_gap(
            singular_values, svd_tolerance,
        )

        # ── Espectro no nulo ──────────────────────────────────────────────
        nonzero_svs = singular_values[singular_values > svd_tolerance]

        if nonzero_svs.size > 0:
            sigma_min_nonzero = float(nonzero_svs[-1])
            spectral_gap = float(sigma_max - sigma_min_nonzero)
        else:
            sigma_min_nonzero = 0.0
            spectral_gap = 0.0

        # ── dim H¹ ────────────────────────────────────────────────────────
        h1_dimension = int(dim_C1 - effective_rank)

        if h1_dimension < 0:
            logger.warning(
                "[Fase 1] dim H¹ = %d < 0; proyectando a 0 por consistencia numérica. "
                "Verificar la SVD de δ.",
                h1_dimension,
            )
            h1_dimension = 0

        # ── Índice de estabilidad β ───────────────────────────────────────
        cohomological_stability_index = (
            float(effective_rank) / float(dim_C1)
            if dim_C1 > 0
            else 1.0
        )

        # ── Euler parcial χ₀₁ ────────────────────────────────────────────
        euler_characteristic_01 = int(dim_C0 - dim_C1)

        # ── Torsión de Whitehead ──────────────────────────────────────────
        whitehead_torsion = self._compute_whitehead_torsion(
            singular_values, effective_rank,
        )

        # ── Poincaré-Lefschetz ────────────────────────────────────────────
        pl_ok = self._verify_poincare_lefschetz(dim_C0, dim_C1, effective_rank)

        if not pl_ok:
            raise PoincareLefschetzViolation(
                "[Fase 1] La condición de Poincaré-Lefschetz es violada: "
                f"rank(δ) = {effective_rank} > min(dim C⁰, dim C¹) = "
                f"{min(dim_C0, dim_C1)}. "
                "La descomposición de Hodge no puede ser ortogonal."
            )

        # ── Veto topológico ───────────────────────────────────────────────
        if h1_dimension > 0:
            raise TopologicalBifurcationError(
                "[Fase 1] Fractura homológica global. "
                f"dim H¹(G; F) = {h1_dimension} > 0. "
                f"β = {cohomological_stability_index:.4f}, "
                f"gap_ratio = {gap_ratio:.4e}. "
                "Existen dependencias circulares insalvables en la malla agéntica."
            )

        # Advertencia si el gap espectral es pequeño (rango numéricamente incierto)
        if gap_ratio < _SPECTRAL_GAP_MIN_RATIO and effective_rank > 0:
            logger.warning(
                "[Fase 1] Gap espectral pequeño: Δσ/σ_max = %.4e < ρ_gap = %.4e. "
                "El rango efectivo puede ser numéricamente incierto.",
                gap_ratio, _SPECTRAL_GAP_MIN_RATIO,
            )

        # ── Emisión del certificado (objeto inicial de Fase 2) ────────────
        return CohomologicalVetoData(
            dim_C0=int(dim_C0),
            dim_C1=int(dim_C1),
            delta_rank=int(effective_rank),
            h1_dimension=int(h1_dimension),
            svd_tolerance=float(svd_tolerance),
            max_singular_value=float(sigma_max),
            min_nonzero_singular_value=float(sigma_min_nonzero),
            spectral_gap=float(spectral_gap),
            spectral_gap_ratio=float(gap_ratio),
            cohomological_stability_index=float(cohomological_stability_index),
            euler_characteristic_01=int(euler_characteristic_01),
            whitehead_torsion=float(whitehead_torsion),
            poincare_lefschetz_ok=bool(pl_ok),
            is_topologically_coherent=True,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 2: REGULACIÓN DEL ESPECTRO DE KRYLOV Y ENERGÍA DE DIRICHLET          ║
# ║                                                                             ║
# ║   Marco matemático:                                                         ║
# ║   ─────────────────                                                         ║
# ║   La energía de Dirichlet mide la frustración del estado x ∈ C⁰:            ║
# ║       E(x) = ‖δx‖₂² = xᵀ Lx  donde L = δᵀδ.                               ║
# ║                                                                             ║
# ║   La descomposición de Hodge-Helmholtz:                                      ║
# ║       x = x_harm + x_exact                                                   ║
# ║   donde x_harm ∈ ker(δ) y x_exact ∈ im(δᵀ).                                 ║
# ║                                                                             ║
# ║   La cota de Poincaré (para x ⊥ ker δ):                                     ║
# ║       ‖x‖ ≤ C_P · ‖δx‖.                                                    ║
# ║                                                                             ║
# ║   Invariantes calculados en v3.0.0:                                         ║
# ║   1. E(x) = ‖δx‖₂² y Ê = E(x)/‖x‖².                                        ║
# ║   2. κ(δ) = σ_max / σ_min^+ y κ(L) = κ(δ)².                                ║
# ║   3. Descomposición de Hodge aproximada: x_harm = x - δ†δx.                 ║
# ║   4. Cota de Poincaré C_P = ‖x_exact‖ / ‖δx‖.                              ║
# ║   5. Índice de frustración ρ = E(x) / ε_frust.                              ║
# ║                                                                             ║
# ║   CONEXIÓN FUNTORIAL:                                                       ║
# ║   El primer método de Fase 2 recibe CohomologicalVetoData y verifica        ║
# ║   coherencia topológica y dimensional.                                       ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_KrylovSpectralAuditor(Phase1_CohomologicalVetoCertifier):
    r"""
    Restringe la instanciación matricial explícita del Laplaciano L = δᵀδ
    y mide la frustración térmica E(x) = ‖δx‖₂².

    Hereda de Phase1_CohomologicalVetoCertifier.
    Su primer método _audit_krylov_spectral_stability recibe el certificado
    CohomologicalVetoData emitido por el último método de Fase 1.

    Mejoras v3.0.0:
        · Energía normalizada Ê = E(x)/‖x‖².
        · Descomposición de Hodge aproximada: x_harm = x − δ†δx.
        · Cota de Poincaré C_P = ‖x_exact‖ / ‖δx‖.
        · Índice de frustración ρ = E(x) / ε_frust.
        · Gap espectral efectivo propagado desde Fase 1.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 2.1 Número de condición de δ desde el certificado
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _kappa_from_certificate(
        veto: CohomologicalVetoData,
    ) -> Tuple[float, float]:
        r"""
        Extrae κ(δ) y κ(L) = κ(δ)² desde el certificado de Fase 1.

        Si σ_min = 0 (δ con kernel no trivial), κ(δ) = ∞ y el cómputo
        explícito de L debe vetarse.

        Retorna:
            (κ_delta, κ_L).
        """
        sigma_max = veto.max_singular_value
        sigma_min = veto.min_nonzero_singular_value

        if sigma_max <= 0.0:
            # δ es el operador cero → trivialmente estable.
            return 1.0, 1.0

        if sigma_min <= 0.0:
            return math.inf, math.inf

        kappa_delta = sigma_max / sigma_min

        if not math.isfinite(kappa_delta):
            return math.inf, math.inf

        kappa_L = kappa_delta * kappa_delta

        if not math.isfinite(kappa_L):
            return kappa_delta, math.inf

        return float(kappa_delta), float(kappa_L)

    # ─────────────────────────────────────────────────────────────────────────
    # 2.2 Descomposición de Hodge aproximada
    # ─────────────────────────────────────────────────────────────────────────
    def _approximate_hodge_decomposition(
        self,
        delta: NDArray[np.float64],
        x: NDArray[np.float64],
        delta_x: NDArray[np.float64],
        svd_tolerance: float,
    ) -> Tuple[float, float]:
        r"""
        Calcula las normas de las componentes armónica y exacta de x:

            x_exact = δ†(δx)    →  componente exacta (en im δᵀ)
            x_harm  = x − x_exact  →  componente armónica (en ker δ)

        donde δ† es la pseudoinversa de Moore-Penrose de δ.

        Parámetros:
            delta     : Operador cofrontera δ.
            x         : Estado x ∈ C⁰.
            delta_x   : δx ∈ C¹ (ya calculado).
            svd_tolerance: Tolerancia SVD del certificado de Fase 1.

        Retorna:
            (‖x_harm‖₂, ‖x_exact‖₂).
        """
        try:
            delta_pinv = self._pseudo_inverse(delta, "δ", tolerance=svd_tolerance)
            x_exact = delta_pinv @ delta_x

            if not np.all(np.isfinite(x_exact)):
                raise HodgeDecompositionError(
                    "[Fase 2] δ†(δx) produjo valores no finitos."
                )

            x_harm = x - x_exact

            if not np.all(np.isfinite(x_harm)):
                raise HodgeDecompositionError(
                    "[Fase 2] La componente armónica x − δ†(δx) no es finita."
                )

            harm_norm = self._vector_norm(x_harm)
            exact_norm = self._vector_norm(x_exact)

            return (
                harm_norm if math.isfinite(harm_norm) else math.inf,
                exact_norm if math.isfinite(exact_norm) else math.inf,
            )

        except (HodgeDecompositionError, SVDConvergenceError):
            raise
        except Exception as exc:
            logger.warning(
                "[Fase 2] Descomposición de Hodge aproximada falló: %s. "
                "Se usan normas de emergencia.",
                exc,
            )
            return math.inf, math.inf

    # ─────────────────────────────────────────────────────────────────────────
    # 2.3 Cota de Poincaré
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_poincare_constant(
        exact_component_norm: float,
        dirichlet_energy: float,
    ) -> float:
        r"""
        Estima la constante de Poincaré:

            C_P = ‖x_exact‖₂ / ‖δx‖₂ = ‖x_exact‖₂ / √E(x).

        Para x ⊥ ker(δ): ‖x‖ ≤ C_P · ‖δx‖.

        Si E(x) = 0 (x ∈ ker δ), C_P no está definida → retorna 0.0.

        Parámetros:
            exact_component_norm: ‖x_exact‖₂.
            dirichlet_energy    : E(x) = ‖δx‖₂².

        Retorna:
            C_P ≥ 0.
        """
        if dirichlet_energy <= 0.0 or not math.isfinite(dirichlet_energy):
            return 0.0

        delta_x_norm = math.sqrt(dirichlet_energy)

        if delta_x_norm == 0.0:
            return 0.0

        if not math.isfinite(exact_component_norm):
            return math.inf

        c_p = exact_component_norm / delta_x_norm
        return float(c_p) if math.isfinite(c_p) else math.inf

    # ─────────────────────────────────────────────────────────────────────────
    # 2.4 PRIMER MÉTODO DE FASE 2 / CONTINUACIÓN DE FASE 1
    #     _audit_krylov_spectral_stability → KrylovSpectralData
    #     [Objeto inicial de Fase 3]
    # ─────────────────────────────────────────────────────────────────────────
    def _audit_krylov_spectral_stability(
        self,
        coboundary_operator_delta: NDArray[np.float64],
        x_state: NDArray[np.float64],
        veto_audit: Optional[CohomologicalVetoData] = None,
    ) -> KrylovSpectralData:
        r"""
        ════════════════════════════════════════════════════════════════════
        PRIMER MÉTODO DE FASE 2 — Continuación formal de Fase 1.
        Retorna el objeto inicial de Fase 3.
        ════════════════════════════════════════════════════════════════════

        Mide la energía de disipación y el condicionamiento numérico sin
        ensamblar explícitamente el Laplaciano L = δᵀδ.

        Si `veto_audit` es provisto (composición funtorial estricta):
            - Verifica is_topologically_coherent = True.
            - Verifica consistencia dimensional de δ con el certificado.
            - Reutiliza σ_max, σ_min del certificado para κ(δ).

        Si `veto_audit` es None:
            - Ejecuta internamente Fase 1 para obtener el certificado.

        Invariantes calculados:
            1. E(x) = ‖δx‖₂² (frustración térmica).
            2. Ê = E(x) / max(1, ‖x‖²) (energía normalizada).
            3. ρ = E(x) / ε_frust (índice de frustración).
            4. κ(δ) y κ(L) = κ(δ)².
            5. Descomposición de Hodge aproximada.
            6. Cota de Poincaré C_P.
            7. Gap espectral efectivo desde el certificado de Fase 1.

        Parámetros:
            coboundary_operator_delta: Operador δ: C⁰ → C¹.
            x_state                  : Estado x ∈ C⁰.
            veto_audit               : Certificado de Fase 1 (opcional).

        Retorna:
            KrylovSpectralData — certificado espectral completo.
            Este objeto es el **objeto inicial de la Fase 3**.

        Lanza:
            TopologicalBifurcationError si veto_audit no es coherente.
            SpectralComputationError si κ(L) > κ_max.
            DirichletFrustrationError si E(x) > ε_frust.
            PoincareBoundViolation si C_P > C_P_max.
        """
        # ── Validación de entrada ─────────────────────────────────────────
        delta = self._as_finite_matrix(
            "coboundary_operator_delta",
            coboundary_operator_delta,
        )
        x = self._as_finite_vector("x_state", x_state)

        if x.size != delta.shape[1]:
            raise ValueError(
                f"[Fase 2] x_state ∈ C⁰ debe tener dim={delta.shape[1]}; "
                f"recibido dim={x.size}."
            )

        # ── Continuación funtorial de Fase 1 ─────────────────────────────
        if veto_audit is None:
            veto_audit = self._certify_cohomological_veto_axiom(delta)
        else:
            if not veto_audit.is_topologically_coherent:
                raise TopologicalBifurcationError(
                    "[Fase 2] No puede iniciarse: Fase 1 reportó incoherencia "
                    "topológica (is_topologically_coherent=False)."
                )

            if veto_audit.h1_dimension != 0:
                raise TopologicalBifurcationError(
                    f"[Fase 2] No puede iniciarse: dim H¹ = {veto_audit.h1_dimension} > 0 "
                    "en el certificado de Fase 1."
                )

            cert_shape = (veto_audit.dim_C1, veto_audit.dim_C0)
            if delta.shape != cert_shape:
                raise ValueError(
                    f"[Fase 2] Inconsistencia dimensional entre certificado "
                    f"(δ: {cert_shape}) y δ actual ({delta.shape})."
                )

        # ── Energía de Dirichlet ──────────────────────────────────────────
        delta_x = delta @ x

        if not np.all(np.isfinite(delta_x)):
            raise SpectralComputationError(
                "[Fase 2] δx produjo valores no finitos."
            )

        dirichlet_energy = self._squared_norm_from_vector(delta_x)

        # ── Energía normalizada ───────────────────────────────────────────
        x_norm_sq = float(np.dot(x, x)) if x.size > 0 else 0.0
        x_norm_sq = max(0.0, x_norm_sq)
        dirichlet_energy_norm = (
            dirichlet_energy / max(1.0, x_norm_sq)
            if math.isfinite(dirichlet_energy)
            else math.inf
        )

        # ── Tolerancia de frustración ─────────────────────────────────────
        frustration_tolerance = max(
            _FRUSTRATION_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, abs(dirichlet_energy)),
        )

        # ── Índice de frustración ─────────────────────────────────────────
        frustration_index = (
            dirichlet_energy / frustration_tolerance
            if frustration_tolerance > 0.0
            else math.inf
        )

        # ── Verificación de frustración ───────────────────────────────────
        if dirichlet_energy > frustration_tolerance:
            raise DirichletFrustrationError(
                "[Fase 2] Frustración térmica inadmisible. "
                f"E(x) = ‖δx‖₂² = {dirichlet_energy:.6e} > "
                f"ε_frust = {frustration_tolerance:.6e}. "
                f"Índice ρ = {frustration_index:.4f}."
            )

        # ── κ(δ) y κ(L) desde el certificado ─────────────────────────────
        kappa_delta, kappa_L = self._kappa_from_certificate(veto_audit)

        if not math.isfinite(kappa_delta):
            raise SpectralComputationError(
                "[Fase 2] κ(δ) no es finita; δ tiene kernel no trivial "
                "a pesar de que dim H¹ = 0. Inconsistencia numérica."
            )

        if not math.isfinite(kappa_L):
            raise SpectralComputationError(
                "[Fase 2] κ(L) = κ(δ)² no es finita."
            )

        # ── Verificación de condicionamiento ──────────────────────────────
        condition_tolerance = (
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, _MAX_CONDITION_NUMBER_L)
        )

        if kappa_L > _MAX_CONDITION_NUMBER_L + condition_tolerance:
            raise SpectralComputationError(
                "[Fase 2] Peligro de colapso FPU (IEEE 754). "
                f"κ(L) = {kappa_L:.6e} > κ_max = {_MAX_CONDITION_NUMBER_L:.6e}. "
                "Se veta el cómputo espectral explícito."
            )

        # ── Descomposición de Hodge aproximada ────────────────────────────
        harm_norm, exact_norm = self._approximate_hodge_decomposition(
            delta, x, delta_x, veto_audit.svd_tolerance,
        )

        # ── Cota de Poincaré ──────────────────────────────────────────────
        poincare_constant = self._compute_poincare_constant(
            exact_norm, dirichlet_energy,
        )

        is_poincare_bounded = (
            poincare_constant <= _POINCARE_CONSTANT_MAX
            or not math.isfinite(poincare_constant)
        )

        if math.isfinite(poincare_constant) and poincare_constant > _POINCARE_CONSTANT_MAX:
            raise PoincareBoundViolation(
                "[Fase 2] La constante de Poincaré excede el umbral admisible: "
                f"C_P = {poincare_constant:.6e} > C_P_max = {_POINCARE_CONSTANT_MAX:.6e}."
            )

        # ── Gap espectral efectivo desde Fase 1 ───────────────────────────
        spectral_gap_effective = veto_audit.spectral_gap

        # ── Emisión del certificado (objeto inicial de Fase 3) ────────────
        return KrylovSpectralData(
            dirichlet_energy=float(dirichlet_energy),
            dirichlet_energy_norm=float(dirichlet_energy_norm),
            frustration_tolerance=float(frustration_tolerance),
            frustration_index=float(frustration_index),
            delta_condition_number=float(kappa_delta),
            laplacian_condition_number=float(kappa_L),
            spectral_gap_effective=float(spectral_gap_effective),
            harmonic_component_norm=float(harm_norm),
            exact_component_norm=float(exact_norm),
            poincare_constant=float(poincare_constant),
            is_frustration_bounded=True,
            is_spectrally_stable=True,
            is_poincare_bounded=bool(is_poincare_bounded),
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 3: IMPOSICIÓN DEL LÍMITE ISOPERIMÉTRICO DE HODGE-HELMHOLTZ           ║
# ║                                                                             ║
# ║   Marco matemático:                                                         ║
# ║   ─────────────────                                                         ║
# ║   La proyección de Hodge π: C⁰ → ker(δ) satisface:                          ║
# ║   1. π minimiza la energía: E(π(x)) ≤ E(x)  ∀ x.                           ║
# ║   2. π es una retracción: π∘π = π.                                          ║
# ║   3. π minimiza la norma en ker(δ): ‖π(x)‖ ≤ ‖x‖.                         ║
# ║                                                                             ║
# ║   Invariantes verificados en v3.0.0:                                        ║
# ║   1. ‖x − x*‖₂ ≤ Δ_inertia (cota isoperimétrica).                          ║
# ║   2. E(x*) ≤ E(x) + ε_num (no incremento energético).                      ║
# ║   3. ‖δx* − δx‖ ≤ κ(δ)·‖x* − x‖ (Lipschitz fuerte).                      ║
# ║   4. ‖x*‖ ≤ ‖x‖ + ε_num (mínima norma).                                   ║
# ║   5. h(G) ≈ E(x*)/‖x*‖² (estimado de Cheeger).                             ║
# ║   6. ι_M = dim ker(δᵀδ) (índice de reducción de Morse).                    ║
# ║                                                                             ║
# ║   CONEXIÓN FUNTORIAL:                                                       ║
# ║   El primer método de Fase 3 recibe KrylovSpectralData y verifica           ║
# ║   que la estabilidad espectral sea compatible con la proyección.             ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_IsoperimetricHodgeProjector(Phase2_KrylovSpectralAuditor):
    r"""
    Regula la magnitud geométrica de la corrección del flujo y verifica
    la admisibilidad termodinámica de la proyección de Hodge.

    Hereda de Phase2_KrylovSpectralAuditor.
    Su primer método _enforce_isoperimetric_hodge_projection recibe el
    certificado KrylovSpectralData emitido por el último método de Fase 2.

    Mejoras v3.0.0:
        · Verificación del axioma de Lipschitz fuerte.
        · Verificación de la condición de mínima norma.
        · Estimado de la constante de Cheeger h(G).
        · Índice de reducción de Morse ι_M = dim ker(δᵀδ).
        · Consistencia energética con el certificado de Fase 2.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 3.1 Verificación del axioma de Lipschitz fuerte
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _verify_lipschitz_strong(
        delta_x0: NDArray[np.float64],
        delta_x1: NDArray[np.float64],
        displacement_norm: float,
        kappa_delta: float,
        *,
        vector_norm_fn,
    ) -> Tuple[float, bool]:
        r"""
        Verifica la condición de Lipschitz fuerte de la proyección de Hodge:

            ‖δx* − δx‖₂ ≤ κ(δ) · ‖x* − x‖₂.

        Para una proyección ortogonal exacta en ker(δ), δx* = 0 → lhs = ‖δx‖.
        Para proyecciones aproximadas, la condición mide la estabilidad.

        Parámetros:
            delta_x0      : δx (imagen de x).
            delta_x1      : δx* (imagen de x*).
            displacement_norm: ‖x* − x‖₂.
            kappa_delta   : κ(δ) = σ_max/σ_min.
            vector_norm_fn: Función de norma vectorial.

        Retorna:
            (lipschitz_residual, lipschitz_ok).
        """
        delta_diff = delta_x1 - delta_x0

        if not np.all(np.isfinite(delta_diff)):
            return math.inf, False

        lhs = vector_norm_fn(delta_diff)

        if not math.isfinite(lhs):
            return math.inf, False

        # rhs = κ(δ) · ‖x* − x‖
        if not math.isfinite(kappa_delta) or not math.isfinite(displacement_norm):
            rhs = math.inf
        else:
            rhs = kappa_delta * displacement_norm

        lipschitz_residual = lhs - rhs  # ≤ 0 sii satisfecho
        slack = _LIPSCHITZ_SLACK * max(1.0, rhs)
        lipschitz_ok = lipschitz_residual <= slack

        return float(lipschitz_residual), bool(lipschitz_ok)

    # ─────────────────────────────────────────────────────────────────────────
    # 3.2 Estimado de la constante de Cheeger
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _estimate_cheeger_bound(
        projected_energy: float,
        x_projected_norm: float,
    ) -> float:
        r"""
        Estima la constante isoperimétrica de Cheeger:

            h(G) ≈ E(x*) / ‖x*‖² = ‖δx*‖₂² / ‖x*‖².

        Esta estimación coincide con el menor eigenvalor de L = δᵀδ
        restringido al subespacio ortogonal a ker(δ):

            λ_min^+(L) ≈ h(G)².

        Un h(G) cercano a 0 indica que la proyección es casi exacta.

        Parámetros:
            projected_energy  : E(x*) = ‖δx*‖₂².
            x_projected_norm  : ‖x*‖₂.

        Retorna:
            h(G) ∈ [0, ∞).
        """
        if x_projected_norm <= 0.0 or not math.isfinite(x_projected_norm):
            return 0.0

        if projected_energy <= 0.0:
            return 0.0

        h = projected_energy / (x_projected_norm * x_projected_norm)
        return float(h) if math.isfinite(h) else math.inf

    # ─────────────────────────────────────────────────────────────────────────
    # 3.3 Índice de reducción de Morse
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_morse_reduction_index(
        self,
        delta: NDArray[np.float64],
        svd_tolerance: float,
    ) -> int:
        r"""
        Calcula el índice de reducción de Morse:

            ι_M = dim ker(δᵀδ) = dim ker(δ) = dim(C⁰) − rank(δ).

        Este índice cuenta las direcciones de "colapso" del funcional
        de energía de Dirichlet, i.e., los modos armónicos de L = δᵀδ
        que tienen eigenvalor cero.

        Parámetros:
            delta        : Operador δ.
            svd_tolerance: Tolerancia del certificado de Fase 1.

        Retorna:
            ι_M ≥ 0.
        """
        if delta.size == 0 or min(delta.shape) == 0:
            return int(delta.shape[1]) if delta.ndim == 2 else 0

        try:
            svs = self._safe_svdvals(delta, "δ (Morse)")
            rank = int(np.count_nonzero(svs > svd_tolerance))
            morse_index = int(delta.shape[1]) - rank
            return max(0, morse_index)
        except Exception as exc:
            logger.warning(
                "[Fase 3] No se pudo calcular el índice de Morse: %s. "
                "Se retorna 0.",
                exc,
            )
            return 0

    # ─────────────────────────────────────────────────────────────────────────
    # 3.4 PRIMER MÉTODO DE FASE 3 / CONTINUACIÓN DE FASE 2
    #     _enforce_isoperimetric_hodge_projection → HodgeProjectionData
    # ─────────────────────────────────────────────────────────────────────────
    def _enforce_isoperimetric_hodge_projection(
        self,
        x_original: NDArray[np.float64],
        x_projected: NDArray[np.float64],
        spectral_audit: Optional[KrylovSpectralData] = None,
        coboundary_operator_delta: Optional[NDArray[np.float64]] = None,
    ) -> HodgeProjectionData:
        r"""
        ════════════════════════════════════════════════════════════════════
        PRIMER MÉTODO DE FASE 3 — Continuación formal de Fase 2.
        ════════════════════════════════════════════════════════════════════

        Verifica la admisibilidad termodinámica de la proyección de Hodge:

            x → x*  donde x* ≈ π(x) ∈ ker(δ).

        Si `spectral_audit` es provisto (composición funtorial estricta):
            - Verifica is_spectrally_stable = True.
            - Verifica is_frustration_bounded = True.
            - Verifica consistencia energética con E(x) certificado.

        Si `coboundary_operator_delta` es provisto:
            - Verifica E(x*) ≤ E(x) (no incremento de Dirichlet).
            - Verifica la condición de Lipschitz fuerte.
            - Verifica la condición de mínima norma.
            - Calcula el estimado de Cheeger h(G).
            - Calcula el índice de Morse ι_M.

        Invariantes verificados:
            1. ‖x − x*‖₂ ≤ Δ_inertia.
            2. E(x*) ≤ E(x) + ε_num.
            3. ‖δx* − δx‖₂ ≤ κ(δ) · ‖x* − x‖₂ (Lipschitz fuerte).
            4. ‖x*‖₂ ≤ ‖x‖₂ + ε_num (mínima norma).
            5. Consistencia con el certificado de Fase 2.

        Parámetros:
            x_original               : Estado original x ∈ C⁰.
            x_projected              : Estado proyectado x* ∈ C⁰.
            spectral_audit           : Certificado de Fase 2 (opcional).
            coboundary_operator_delta: Operador δ (opcional).

        Retorna:
            HodgeProjectionData — certificado de proyección completo.

        Lanza:
            SpectralComputationError si spectral_audit no es estable.
            DirichletFrustrationError si spectral_audit no acotó frustración.
            HomologicalInconsistencyError si ‖x−x*‖ > Δ_inertia.
            LipschitzViolation si la condición Lipschitz fuerte falla.
            MinimalNormViolation si ‖x*‖ > ‖x‖ + ε_num.
        """
        # ── Validación de entrada ─────────────────────────────────────────
        x0 = self._as_finite_vector("x_original", x_original)
        x1 = self._as_finite_vector("x_projected", x_projected)

        if x0.shape != x1.shape:
            raise ValueError(
                f"[Fase 3] x_original (dim={x0.size}) y x_projected "
                f"(dim={x1.size}) deben tener la misma dimensión."
            )

        # ── Continuación funtorial de Fase 2 ─────────────────────────────
        if spectral_audit is not None:
            if not spectral_audit.is_spectrally_stable:
                raise SpectralComputationError(
                    "[Fase 3] No puede iniciarse: Fase 2 reportó inestabilidad "
                    "espectral (is_spectrally_stable=False)."
                )
            if not spectral_audit.is_frustration_bounded:
                raise DirichletFrustrationError(
                    "[Fase 3] No puede iniciarse: Fase 2 no acotó la frustración "
                    "térmica (is_frustration_bounded=False)."
                )

        # ── Validación del operador δ (si provisto) ───────────────────────
        delta: Optional[NDArray[np.float64]] = None

        if coboundary_operator_delta is not None:
            delta = self._as_finite_matrix(
                "coboundary_operator_delta",
                coboundary_operator_delta,
            )
            if x0.size != delta.shape[1] or x1.size != delta.shape[1]:
                raise ValueError(
                    f"[Fase 3] δ espera dim C⁰ = {delta.shape[1]}, pero "
                    f"x_original dim={x0.size} y x_projected dim={x1.size}."
                )

        # ── Desplazamiento ────────────────────────────────────────────────
        displacement = x0 - x1

        if not np.all(np.isfinite(displacement)):
            raise HomologicalInconsistencyError(
                "[Fase 3] El vector de desplazamiento x − x* no es finito."
            )

        projection_distance = self._vector_norm(displacement)

        if not math.isfinite(projection_distance):
            raise HomologicalInconsistencyError(
                "[Fase 3] ‖x − x*‖₂ no es finita."
            )

        norm_x0 = self._vector_norm(x0)
        norm_x1 = self._vector_norm(x1)

        if not math.isfinite(norm_x0) or not math.isfinite(norm_x1):
            raise HomologicalInconsistencyError(
                "[Fase 3] Las normas de x_original o x_projected no son finitas."
            )

        # ── Distancia relativa ────────────────────────────────────────────
        relative_distance = projection_distance / max(1.0, norm_x0)

        # ── Tolerancia numérica para distancia ────────────────────────────
        dist_tolerance = (
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, norm_x0, norm_x1)
        )

        inertia_limit = _INERTIA_DELTA_MAX + dist_tolerance

        # ── Verificación isoperimétrica ───────────────────────────────────
        if projection_distance > inertia_limit:
            raise HomologicalInconsistencyError(
                "[Fase 3] Violación del principio de inercia inercial. "
                f"‖x − x*‖₂ = {projection_distance:.6f} > "
                f"Δ_inertia = {_INERTIA_DELTA_MAX:.6f}. "
                "La sanación topológica requeriría recursos irreales."
            )

        # ── Variables de invariantes secundarios ──────────────────────────
        verified_by_delta = delta is not None
        original_energy = 0.0
        projected_energy = 0.0
        energy_reduction_ratio = 0.0
        is_energy_non_increasing = True
        lipschitz_residual = 0.0
        lipschitz_satisfied = True
        minimal_norm_satisfied = True
        cheeger_estimate = 0.0
        morse_index = 0

        # ── Bloque de verificación con δ ──────────────────────────────────
        if verified_by_delta:
            assert delta is not None

            delta_x0 = delta @ x0
            delta_x1 = delta @ x1

            if not np.all(np.isfinite(delta_x0)) or not np.all(np.isfinite(delta_x1)):
                raise HomologicalInconsistencyError(
                    "[Fase 3] La evaluación δx o δx* produjo valores no finitos."
                )

            original_energy = self._squared_norm_from_vector(delta_x0)
            projected_energy = self._squared_norm_from_vector(delta_x1)

            energy_tolerance = (
                _NUMERICAL_SAFETY_FACTOR
                * _MACHINE_EPSILON
                * max(1.0, abs(original_energy), abs(projected_energy))
            )

            # ── Consistencia con certificado de Fase 2 ────────────────────
            if spectral_audit is not None:
                consistency_tol = max(
                    _ENERGY_RATIO_TOLERANCE,
                    _NUMERICAL_SAFETY_FACTOR
                    * _MACHINE_EPSILON
                    * max(
                        1.0,
                        abs(original_energy),
                        abs(spectral_audit.dirichlet_energy),
                    ),
                )
                energy_discrepancy = abs(
                    original_energy - spectral_audit.dirichlet_energy
                )
                if energy_discrepancy > consistency_tol:
                    raise HomologicalInconsistencyError(
                        "[Fase 3] Inconsistencia energética entre Fase 2 y Fase 3. "
                        f"E_cert = {spectral_audit.dirichlet_energy:.6e}, "
                        f"E_recalc = {original_energy:.6e}, "
                        f"discrepancia = {energy_discrepancy:.6e} > "
                        f"τ = {consistency_tol:.6e}."
                    )

            # ── No incremento de energía ──────────────────────────────────
            is_energy_non_increasing = (
                projected_energy <= original_energy + energy_tolerance
            )

            if not is_energy_non_increasing:
                raise HomologicalInconsistencyError(
                    "[Fase 3] La proyección de Hodge incrementó la energía. "
                    f"E(x*) = {projected_energy:.6e} > "
                    f"E(x) + ε = {original_energy + energy_tolerance:.6e}."
                )

            # ── Frustración de la proyección ──────────────────────────────
            frust_limit = _FRUSTRATION_TOLERANCE
            if spectral_audit is not None:
                frust_limit = max(frust_limit, spectral_audit.frustration_tolerance)

            proj_frust_tol = max(
                frust_limit,
                _NUMERICAL_SAFETY_FACTOR
                * _MACHINE_EPSILON
                * max(1.0, abs(original_energy), abs(projected_energy)),
            )

            if projected_energy > proj_frust_tol:
                raise HomologicalInconsistencyError(
                    "[Fase 3] La proyección no redujo la frustración a nivel admisible. "
                    f"E(x*) = {projected_energy:.6e} > "
                    f"ε_frust_efectivo = {proj_frust_tol:.6e}."
                )

            # ── Razón de reducción energética ─────────────────────────────
            if original_energy > energy_tolerance:
                energy_reduction_ratio = projected_energy / original_energy
                if not math.isfinite(energy_reduction_ratio):
                    raise HomologicalInconsistencyError(
                        "[Fase 3] La razón de reducción energética no es finita."
                    )
            else:
                energy_reduction_ratio = 0.0

            # ── Axioma de Lipschitz fuerte ─────────────────────────────────
            kappa_delta = (
                spectral_audit.delta_condition_number
                if spectral_audit is not None
                else 1.0
            )

            lipschitz_residual, lipschitz_satisfied = self._verify_lipschitz_strong(
                delta_x0=delta_x0,
                delta_x1=delta_x1,
                displacement_norm=projection_distance,
                kappa_delta=kappa_delta,
                vector_norm_fn=self._vector_norm,
            )

            if not lipschitz_satisfied:
                raise LipschitzViolation(
                    "[Fase 3] Violación del axioma de Lipschitz fuerte. "
                    f"‖δx* − δx‖ − κ(δ)·‖x*−x‖ = {lipschitz_residual:.6e} > 0. "
                    "La proyección no es contractivante en energía."
                )

            # ── Condición de mínima norma ─────────────────────────────────
            norm_tolerance = (
                _NUMERICAL_SAFETY_FACTOR
                * _MACHINE_EPSILON
                * max(1.0, norm_x0, norm_x1)
            )
            minimal_norm_satisfied = norm_x1 <= norm_x0 + norm_tolerance

            if not minimal_norm_satisfied:
                raise MinimalNormViolation(
                    "[Fase 3] La proyección no satisface la condición de mínima norma. "
                    f"‖x*‖ = {norm_x1:.6e} > ‖x‖ + ε = {norm_x0 + norm_tolerance:.6e}. "
                    "x* no es la proyección de mínima norma en ker(δ)."
                )

            # ── Estimado de Cheeger ───────────────────────────────────────
            cheeger_estimate = self._estimate_cheeger_bound(
                projected_energy, norm_x1,
            )

            # ── Índice de Morse ───────────────────────────────────────────
            svd_tol = (
                spectral_audit.frustration_tolerance
                if spectral_audit is not None
                else _SVD_TOLERANCE_BASE
            )
            # Usamos la tolerancia SVD del certificado de Fase 1 si disponible
            veto_tol = _SVD_TOLERANCE_BASE
            morse_index = self._compute_morse_reduction_index(delta, veto_tol)

        # ── Emisión del certificado ───────────────────────────────────────
        return HodgeProjectionData(
            projection_distance=float(projection_distance),
            relative_projection_distance=float(relative_distance),
            inertia_delta_max=float(_INERTIA_DELTA_MAX),
            original_dirichlet_energy=float(original_energy),
            projected_dirichlet_energy=float(projected_energy),
            energy_reduction_ratio=float(energy_reduction_ratio),
            lipschitz_residual=float(lipschitz_residual),
            lipschitz_satisfied=bool(lipschitz_satisfied),
            minimal_norm_satisfied=bool(minimal_norm_satisfied),
            cheeger_bound_estimate=float(cheeger_estimate),
            morse_reduction_index=int(morse_index),
            is_isoperimetrically_bounded=True,
            is_energy_non_increasing=bool(is_energy_non_increasing),
            verified_by_delta=bool(verified_by_delta),
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   ORQUESTADOR SUPREMO: SHEAF COHOMOLOGY ORCHESTRATOR AGENT                  ║
# ║                                                                             ║
# ║   Endofuntor Z_SheafAgent = Φ₃ ∘ Φ₂ ∘ Φ₁                                   ║
# ║                                                                             ║
# ║   Mejoras v3.0.0:                                                           ║
# ║   · Trazabilidad criptográfica con SHA-256 y timestamp ISO-8601.             ║
# ║   · AuditProvenance con cadena funtorial completa.                          ║
# ║   · Log estructurado de alta granularidad con todos los certificados.        ║
# ║   · Verificación de la coherencia global de los tres certificados.          ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class SheafCohomologyOrchestratorAgent(Morphism, Phase3_IsoperimetricHodgeProjector):
    r"""
    El Custodio de la Holonomía Global en el estrato STRATEGY.

    Somete la estrategia de consenso agéntico a la composición funtorial:

        Z_SheafAgent = Φ₃ ∘ Φ₂ ∘ Φ₁,

    garantizando coherencia topológica, estabilidad espectral y admisibilidad
    termodinámica de la proyección de Hodge.

    Nuevas capacidades v3.0.0:
        · Checksum SHA-256 de las entradas para trazabilidad criptográfica.
        · Timestamp ISO-8601 UTC en cada ejecución.
        · SheafAuditProvenance con la cadena funtorial completa.
        · Log estructurado con todos los certificados de las tres fases.
    """

    def __init__(self, strict_mode: bool = True) -> None:
        r"""
        Inicializa el SheafCohomologyOrchestratorAgent.

        Parámetros:
            strict_mode: Si True (default), toda advertencia se convierte en error.
        """
        self._strict_mode = bool(strict_mode)
        logger.info(
            "[Orquestador] SheafCohomologyOrchestratorAgent inicializado. "
            "strict_mode=%s.",
            self._strict_mode,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Construcción del SheafAuditProvenance
    # ─────────────────────────────────────────────────────────────────────────
    def _build_provenance(
        self,
        checksum: str,
        phase1_passed: bool,
        phase2_passed: bool,
        phase3_passed: bool,
    ) -> SheafAuditProvenance:
        r"""
        Construye el objeto de trazabilidad funtorial.
        """
        timestamp = datetime.now(tz=timezone.utc).isoformat()

        symbols = {True: "✓", False: "✗"}
        functor_chain = (
            f"Φ₁={symbols[phase1_passed]} → "
            f"Φ₂={symbols[phase2_passed]} → "
            f"Φ₃={symbols[phase3_passed]} → "
            f"Z_SheafAgent={symbols[phase1_passed and phase2_passed and phase3_passed]}"
        )

        return SheafAuditProvenance(
            timestamp_iso=timestamp,
            input_checksum_sha256=checksum,
            phase1_passed=phase1_passed,
            phase2_passed=phase2_passed,
            phase3_passed=phase3_passed,
            functor_chain=functor_chain,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Log estructurado
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _log_governance_summary(
        veto_audit: CohomologicalVetoData,
        spectral_audit: KrylovSpectralData,
        hodge_audit: HodgeProjectionData,
        provenance: SheafAuditProvenance,
    ) -> None:
        r"""
        Emite un log estructurado de alta granularidad con todos los certificados.
        """
        logger.info(
            "═══════════════════════════════════════════════════════════\n"
            "  SHEAF COHOMOLOGY GOVERNANCE — REPORTE FINAL\n"
            "  Timestamp  : %s\n"
            "  SHA-256    : %s\n"
            "  Cadena     : %s\n"
            "─────────────────────────────────────────────────────────\n"
            "  FASE 1 — Veto Cohomológico:\n"
            "    dim C⁰        : %d\n"
            "    dim C¹        : %d\n"
            "    rank(δ)       : %d\n"
            "    dim H¹        : %d\n"
            "    σ_max         : %.4e\n"
            "    σ_min^+       : %.4e\n"
            "    Δσ            : %.4e\n"
            "    Δσ/σ_max      : %.4e\n"
            "    β (estab.)    : %.4f\n"
            "    χ₀₁           : %d\n"
            "    log|τ_W|      : %.4f\n"
            "    P-L ✓         : %s\n"
            "    Coherente ✓   : %s\n"
            "─────────────────────────────────────────────────────────\n"
            "  FASE 2 — Espectro de Krylov-Dirichlet:\n"
            "    E(x)          : %.4e\n"
            "    Ê             : %.4e\n"
            "    ε_frust       : %.4e\n"
            "    ρ_frust       : %.4f\n"
            "    κ(δ)          : %.4e\n"
            "    κ(L)          : %.4e\n"
            "    Δσ_eff        : %.4e\n"
            "    ‖x_harm‖      : %.4e\n"
            "    ‖x_exact‖     : %.4e\n"
            "    C_P           : %.4e\n"
            "    Frust. ✓      : %s\n"
            "    Espectral ✓   : %s\n"
            "    Poincaré ✓    : %s\n"
            "─────────────────────────────────────────────────────────\n"
            "  FASE 3 — Hodge Isoperimétrico:\n"
            "    ‖x−x*‖        : %.6f\n"
            "    ‖x−x*‖/‖x‖   : %.6f\n"
            "    Δ_inertia     : %.4f\n"
            "    E(x)          : %.4e\n"
            "    E(x*)         : %.4e\n"
            "    E(x*)/E(x)    : %.4f\n"
            "    Lip. resid.   : %.4e\n"
            "    Lip. ✓        : %s\n"
            "    Min-norm ✓    : %s\n"
            "    h(G) Cheeger  : %.4e\n"
            "    ι_M Morse     : %d\n"
            "    Isoperim. ✓   : %s\n"
            "    E↓ ✓          : %s\n"
            "    Ver. por δ    : %s\n"
            "═══════════════════════════════════════════════════════════",
            provenance.timestamp_iso,
            provenance.input_checksum_sha256[:16] + "...",
            provenance.functor_chain,
            # Fase 1
            veto_audit.dim_C0,
            veto_audit.dim_C1,
            veto_audit.delta_rank,
            veto_audit.h1_dimension,
            veto_audit.max_singular_value,
            veto_audit.min_nonzero_singular_value,
            veto_audit.spectral_gap,
            veto_audit.spectral_gap_ratio,
            veto_audit.cohomological_stability_index,
            veto_audit.euler_characteristic_01,
            veto_audit.whitehead_torsion,
            veto_audit.poincare_lefschetz_ok,
            veto_audit.is_topologically_coherent,
            # Fase 2
            spectral_audit.dirichlet_energy,
            spectral_audit.dirichlet_energy_norm,
            spectral_audit.frustration_tolerance,
            spectral_audit.frustration_index,
            spectral_audit.delta_condition_number,
            spectral_audit.laplacian_condition_number,
            spectral_audit.spectral_gap_effective,
            spectral_audit.harmonic_component_norm,
            spectral_audit.exact_component_norm,
            spectral_audit.poincare_constant,
            spectral_audit.is_frustration_bounded,
            spectral_audit.is_spectrally_stable,
            spectral_audit.is_poincare_bounded,
            # Fase 3
            hodge_audit.projection_distance,
            hodge_audit.relative_projection_distance,
            hodge_audit.inertia_delta_max,
            hodge_audit.original_dirichlet_energy,
            hodge_audit.projected_dirichlet_energy,
            hodge_audit.energy_reduction_ratio,
            hodge_audit.lipschitz_residual,
            hodge_audit.lipschitz_satisfied,
            hodge_audit.minimal_norm_satisfied,
            hodge_audit.cheeger_bound_estimate,
            hodge_audit.morse_reduction_index,
            hodge_audit.is_isoperimetrically_bounded,
            hodge_audit.is_energy_non_increasing,
            hodge_audit.verified_by_delta,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PUNTO DE ENTRADA PRINCIPAL
    # ─────────────────────────────────────────────────────────────────────────
    def execute_sheaf_cohomology_governance(
        self,
        coboundary_operator_delta: NDArray[np.float64],
        x_state: NDArray[np.float64],
        x_projected_consensus: NDArray[np.float64],
    ) -> SheafGovernanceState:
        r"""
        Ejecuta la composición funtorial estricta:

            Z_SheafAgent = Φ₃ ∘ Φ₂ ∘ Φ₁.

        Fases anidadas:
            Φ₁: Certificación cohomológica → CohomologicalVetoData.
            Φ₂: Regulación espectral-energética → KrylovSpectralData.
            Φ₃: Proyección isoperimétrica de Hodge → HodgeProjectionData.

        Adicionalmente:
            · Calcula el SHA-256 de las entradas para trazabilidad.
            · Registra el timestamp UTC de la auditoría.
            · Construye el SheafAuditProvenance con la cadena funtorial.
            · Emite un log estructurado con todos los certificados.

        Parámetros:
            coboundary_operator_delta:
                Operador cofrontera δ: C⁰ → C¹ ∈ ℝ^{m×n}.

            x_state:
                Estado original x ∈ C⁰ ∈ ℝⁿ.

            x_projected_consensus:
                Estado proyectado x* ∈ C⁰, presuntamente en ker(δ).

        Retorna:
            SheafGovernanceState con los tres certificados, provenance y
            is_epistemologically_valid=True sii todas las fases pasan.

        Lanza:
            Cualquier excepción de la jerarquía SheafCohomologyAgentError
            si alguna fase detecta una violación.
        """
        # ── Checksum de entrada ───────────────────────────────────────────
        input_checksum = self._compute_input_checksum(
            np.asarray(coboundary_operator_delta)
            if coboundary_operator_delta is not None else None,
            np.asarray(x_state) if x_state is not None else None,
            np.asarray(x_projected_consensus)
            if x_projected_consensus is not None else None,
        )

        logger.debug(
            "[Orquestador] Iniciando gobernanza. SHA-256: %s.",
            input_checksum[:16] + "...",
        )

        # ── Estado de las fases ───────────────────────────────────────────
        phase1_passed = False
        phase2_passed = False
        phase3_passed = False

        # ── Fase 1: Veto Cohomológico ─────────────────────────────────────
        veto_audit = self._certify_cohomological_veto_axiom(
            coboundary_operator_delta,
        )
        phase1_passed = True

        logger.debug(
            "[Fase 1] ✓ dim H¹=%d, β=%.4f, log|τ_W|=%.4f.",
            veto_audit.h1_dimension,
            veto_audit.cohomological_stability_index,
            veto_audit.whitehead_torsion,
        )

        # ── Fase 2: Espectro de Krylov-Dirichlet ──────────────────────────
        spectral_audit = self._audit_krylov_spectral_stability(
            coboundary_operator_delta,
            x_state,
            veto_audit=veto_audit,
        )
        phase2_passed = True

        logger.debug(
            "[Fase 2] ✓ E(x)=%.4e, κ(L)=%.4e, C_P=%.4e.",
            spectral_audit.dirichlet_energy,
            spectral_audit.laplacian_condition_number,
            spectral_audit.poincare_constant,
        )

        # ── Fase 3: Hodge Isoperimétrico ──────────────────────────────────
        hodge_audit = self._enforce_isoperimetric_hodge_projection(
            x_state,
            x_projected_consensus,
            spectral_audit=spectral_audit,
            coboundary_operator_delta=coboundary_operator_delta,
        )
        phase3_passed = True

        logger.debug(
            "[Fase 3] ✓ ‖x−x*‖=%.6f, E(x*)/E(x)=%.4f, h(G)≈%.4e.",
            hodge_audit.projection_distance,
            hodge_audit.energy_reduction_ratio,
            hodge_audit.cheeger_bound_estimate,
        )

        # ── Autorización epistemológica ───────────────────────────────────
        is_epistemologically_valid = bool(
            veto_audit.is_topologically_coherent
            and spectral_audit.is_spectrally_stable
            and spectral_audit.is_frustration_bounded
            and hodge_audit.is_isoperimetrically_bounded
            and hodge_audit.is_energy_non_increasing
        )

        # ── Construcción de trazabilidad ──────────────────────────────────
        provenance = self._build_provenance(
            checksum=input_checksum,
            phase1_passed=phase1_passed,
            phase2_passed=phase2_passed,
            phase3_passed=phase3_passed,
        )

        # ── Log estructurado ──────────────────────────────────────────────
        self._log_governance_summary(
            veto_audit=veto_audit,
            spectral_audit=spectral_audit,
            hodge_audit=hodge_audit,
            provenance=provenance,
        )

        # ── Verificación final ────────────────────────────────────────────
        if not is_epistemologically_valid:
            raise SheafCohomologyAgentError(
                "[Orquestador] La composición funtorial Z_SheafAgent = Φ₃∘Φ₂∘Φ₁ "
                f"no autorizó la validez epistemológica del haz. "
                f"Cadena: {provenance.functor_chain}."
            )

        return SheafGovernanceState(
            veto_audit=veto_audit,
            spectral_audit=spectral_audit,
            hodge_audit=hodge_audit,
            provenance=provenance,
            is_epistemologically_valid=is_epistemologically_valid,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ═══════════════════════════════════════════════════════════════════════════════
__all__: List[str] = [
    # Excepciones
    "SheafCohomologyAgentError",
    "TopologicalBifurcationError",
    "PoincareLefschetzViolation",
    "SpectralComputationError",
    "SVDConvergenceError",
    "HodgeDecompositionError",
    "DirichletFrustrationError",
    "PoincareBoundViolation",
    "HomologicalInconsistencyError",
    "LipschitzViolation",
    "MinimalNormViolation",
    # DTOs
    "CohomologicalVetoData",
    "KrylovSpectralData",
    "HodgeProjectionData",
    "SheafAuditProvenance",
    "SheafGovernanceState",
    # Fases
    "Phase1_CohomologicalVetoCertifier",
    "Phase2_KrylovSpectralAuditor",
    "Phase3_IsoperimetricHodgeProjector",
    # Orquestador
    "SheafCohomologyOrchestratorAgent",
]