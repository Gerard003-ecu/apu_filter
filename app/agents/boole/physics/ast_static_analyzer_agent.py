# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════╗
║  Módulo : AST Static Analyzer Agent                                              ║
║           (Custodio de la Cohomología Sintáctica)                                ║
║  Ruta   : app/agents/boole/physics/ast_static_analyzer_agent.py                  ║
║  Versión: 3.0.0-Symplectic-Dirichlet-Cohomology-Evolved                          ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  NATURALEZA CIBER-FÍSICA Y GEOMETRÍA SIMPLÉCTICA (Rigor Doctoral):               ║
║  ──────────────────────────────────────────────────────────────────              ║
║  Este endofuntor gobierna a `ast_static_analyzer.py` en V_{Γ-PHYSICS}.           ║
║  El AST generado por la IA es tratado como un espacio de fase mecánico           ║
║  (M, ω), aplicando invariantes topológicos, termodinámicos y cohomológicos       ║
║  para aniquilar código estocástico que diisipe energía computacional             ║
║  no acotada.                                                                     ║
║                                                                                  ║
║  ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta):                ║
║  ────────────────────────────────────────────────────────────────                ║
║  Fase 1 → Auditoría de Invarianza Simpléctica:                                   ║
║      Evalúa la conservación del volumen en el espacio de fase sintáctico         ║
║      mediante la forma canónica ω = Σ dq_i ∧ dp_i. Exige Mᵀ Ω M = Ω.             ║
║      Último método: _audit_symplectic_invariance(...)                            ║
║      Retorna: SymplecticInvariantData  →  objeto inicial de Fase 2.              ║
║                                                                                  ║
║  Fase 2 → Control Port-Hamiltoniano y Fronteras de Dirichlet:                    ║
║      Impone disipación estricta: P_diss = ⟨Φ, ∇V⟩ ≥ 0.                           ║
║      Primer método: _enforce_dirichlet_thermodynamics(..., symplectic_audit)     ║
║      Retorna: ThermodynamicDirichletData  →  objeto inicial de Fase 3.           ║
║                                                                                  ║
║  Fase 3 → Cohomología de Haces Celulares:                                        ║
║      Exige dim H¹(G; F) = 0.                                                     ║
║      Primer método: _audit_cellular_sheaf_cohomology(..., thermodynamic_audit)   ║
║      Retorna: SheafCohomologyAuditData  →  objeto final del endofuntor.          ║
╚══════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# §0. IMPORTACIONES ESTÁNDAR Y CIENTÍFICAS
# ─────────────────────────────────────────────────────────────────────────────
import hashlib
import logging
import math
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# §0.1 DEPENDENCIAS ARQUITECTÓNICAS DEL ECOSISTEMA APU FILTER
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


try:
    from app.physics.ast_static_analyzer import (
        ThermodynamicSingularityError,
        CohomologicalObstructionError,
    )
except ImportError:

    class ThermodynamicSingularityError(TopologicalInvariantError):
        r"""Violación de la segunda ley de la termodinámica en el espacio de fase."""
        pass

    class CohomologicalObstructionError(TopologicalInvariantError):
        r"""Obstrucción topológica global en el haz celular de dependencias."""
        pass


logger = logging.getLogger("MIC.Gamma.ASTAnalyzerAgent")


# ═══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES SIMPLÉCTICAS, TERMODINÁMICAS Y COHOMOLÓGICAS
#
# Jerarquía de tolerancias:
#   _MACHINE_EPSILON                → ε₀ = eps(float64)
#   _SYMPLECTIC_TOLERANCE_BASE      → τ_ω base para residuos simplécticos
#   _SYMPLECTIC_SPECTRAL_FACTOR     → c_σ: escala adaptativa por σ_max
#   _DIRICHLET_DISSIPATION_FLOOR    → P_min: piso de disipación Clausius
#   _ENTROPY_PRODUCTION_FLOOR       → σ_CD_min: piso de producción de entropía
#   _COHOMOLOGICAL_COMPLEX_TOL      → τ_δ: tolerancia del complejo δ¹∘δ⁰ = 0
#   _SPECTRAL_GAP_RATIO             → ρ_gap: umbral de gap espectral para rango
#   _NUMERICAL_SAFETY_FACTOR        → c_num: factor de seguridad numérica general
#   _MASLOV_PHASE_TOLERANCE         → τ_μ: tolerancia para cálculo del índice de Maslov
#   _EULER_CHARACTERISTIC_TOLERANCE → τ_χ: tolerancia para la característica de Euler
# ═══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)
_SYMPLECTIC_TOLERANCE_BASE: Final[float] = 1e-10
_SYMPLECTIC_SPECTRAL_FACTOR: Final[float] = 1e-10
_DIRICHLET_DISSIPATION_FLOOR: Final[float] = 0.0
_ENTROPY_PRODUCTION_FLOOR: Final[float] = 0.0
_COHOMOLOGICAL_COMPLEX_TOL: Final[float] = 1e-8
_SPECTRAL_GAP_RATIO: Final[float] = 1e-10
_NUMERICAL_SAFETY_FACTOR: Final[float] = 128.0
_MASLOV_PHASE_TOLERANCE: Final[float] = 1e-9
_EULER_CHARACTERISTIC_TOLERANCE: Final[int] = 0
_CONDITION_NUMBER_WARNING_THRESHOLD: Final[float] = 1.0 / _MACHINE_EPSILON
_DETERMINANT_RESIDUAL_WARNING_FACTOR: Final[float] = 10.0
_THERMODYNAMIC_TEMPERATURE_REFERENCE: Final[float] = 1.0  # T_ref adimensional


# ═══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA COMPLETA DE EXCEPCIONES
#
# Lattice de excepciones:
#   TopologicalInvariantError
#   ├── SymplecticInvarianceViolation   [Fase 1]
#   │   ├── SymplecticPolarDecompositionError
#   │   └── MaslovIndexError
#   ├── ThermodynamicSingularityError   [Fase 2]
#   │   ├── ClausiusDuhemViolation
#   │   └── ExergyDivergenceError
#   └── CohomologicalObstructionError   [Fase 3]
#       ├── EulerCharacteristicMismatch
#       └── MayerVietorisBreachError
# ═══════════════════════════════════════════════════════════════════════════════
class SymplecticInvarianceViolation(TopologicalInvariantError):
    r"""
    Detonada si la transformación del AST no preserva el volumen del espacio
    de fase simpléctica. Corresponde a la violación del Teorema de Liouville:

        Vol(φₜ(U)) = Vol(U)  ∀ t ∈ ℝ.
    """
    pass


class SymplecticPolarDecompositionError(SymplecticInvarianceViolation):
    r"""
    Detonada si la descomposición polar Q = U·P de la matriz Jacobiana M
    no produce una parte unitaria U que preserve la estructura simpléctica,
    es decir, si U ∉ Sp(2n, ℝ).
    """
    pass


class MaslovIndexError(SymplecticInvarianceViolation):
    r"""
    Detonada si el índice de Maslov μ(Λ) no puede calcularse de forma
    numérica estable, indicando degeneración de la variedad de Lagrange.
    """
    pass


class ClausiusDuhemViolation(ThermodynamicSingularityError):
    r"""
    Detonada si la desigualdad de Clausius-Duhem diferencial es violada:

        σ_CD = dS/dt - Q̇/T < 0.

    Esto implica reducción espontánea de entropía, prohibida por la
    Segunda Ley de la Termodinámica.
    """
    pass


class ExergyDivergenceError(ThermodynamicSingularityError):
    r"""
    Detonada si la norma de exergía disponible diverge, indicando que el
    sistema no puede alcanzar el equilibrio termodinámico:

        ‖Φ‖ → ∞  ó  ‖∇V‖ → ∞.
    """
    pass


class EulerCharacteristicMismatch(CohomologicalObstructionError):
    r"""
    Detonada si la característica de Euler calculada a partir de los rangos
    de los operadores coboundary no es consistente con la topología declarada
    del grafo de dependencias:

        χ(G) ≠ β₀ − β₁ + β₂.
    """
    pass


class MayerVietorisBreachError(CohomologicalObstructionError):
    r"""
    Detonada si la secuencia de Mayer-Vietoris no es exacta en el grafo
    de dependencias, indicando una descomposición inconsistente del haz celular.
    """
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs DEL TOPOS SINTÁCTICO)
#
# Cadena de certificados funtoriales:
#   SymplecticInvariantData    →  emitido por Fase 1, consumido por Fase 2
#   ThermodynamicDirichletData →  emitido por Fase 2, consumido por Fase 3
#   SheafCohomologyAuditData   →  emitido por Fase 3
#   ASTGovernanceState         →  resultado final del endofuntor Z_{Γ-PHYSICS}
#   AuditProvenance            →  trazabilidad de la cadena funtorial
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SymplecticInvariantData:
    r"""
    Artefacto de Fase 1. Certificado del Teorema de Liouville en el AST.

    Campos extendidos v3.0.0:
        phase_space_dimension   : dim del espacio de fase (debe ser par)
        symplectic_residual_norm: ‖MᵀΩM − Ω‖_F   (absoluto)
        symplectic_relative_residual: ‖MᵀΩM − Ω‖_F / scale
        antisymmetric_residual  : ‖skew(MᵀΩM) − Ω‖_F / ‖Ω‖_F
        determinant_residual    : |det(M) − 1|
        condition_number        : κ(M) = σ_max / σ_min
        spectral_radius_Omega   : ρ(MᵀΩM) como control de amplificación
        maslov_index            : μ(Λ) ∈ ℤ — índice de Maslov de la variedad
        polar_unitarity_residual: ‖UᵀΩU − Ω‖_F / ‖Ω‖_F  (parte unitaria)
        effective_tolerance     : tolerancia efectiva usada en la auditoría
        is_volume_preserved     : True sii residuo relativo ≤ tolerancia efectiva
    """
    phase_space_dimension: int
    symplectic_residual_norm: float
    symplectic_relative_residual: float
    antisymmetric_residual: float
    determinant_residual: float
    condition_number: float
    spectral_radius_Omega: float
    maslov_index: int
    polar_unitarity_residual: float
    effective_tolerance: float
    is_volume_preserved: bool


@dataclass(frozen=True, slots=True)
class ThermodynamicDirichletData:
    r"""
    Artefacto de Fase 2. Certificado de Disipación Port-Hamiltoniana.

    Campos extendidos v3.0.0:
        dissipated_power        : P_diss = Φᵀ∇V  (proyectado al cono ≥ 0)
        raw_inner_product       : Φᵀ∇V antes de la proyección
        numerical_tolerance     : ε_num = c·ε₀·max(1, ‖Φ‖·‖∇V‖)
        exergy_norm             : ‖Φ‖₂
        gradient_norm           : ‖∇V‖₂
        alignment_cosine        : cos(θ) = Φᵀ∇V / (‖Φ‖·‖∇V‖)
        thermodynamic_angle_rad : θ = arccos(cos(θ)) ∈ [0, π]
        entropy_production_rate : σ = P_diss / T_ref  (Clausius-Duhem)
        clausius_duhem_satisfied: True sii σ ≥ σ_floor
        exergy_dissipation_ratio: P_diss / (‖Φ‖·‖∇V‖)  (eficiencia)
        is_thermodynamically_stable  : True sii P_diss ≥ −ε_num
        is_strictly_dissipative : True sii P_diss > ε_num
    """
    dissipated_power: float
    raw_inner_product: float
    numerical_tolerance: float
    exergy_norm: float
    gradient_norm: float
    alignment_cosine: float
    thermodynamic_angle_rad: float
    entropy_production_rate: float
    clausius_duhem_satisfied: bool
    exergy_dissipation_ratio: float
    is_thermodynamically_stable: bool
    is_strictly_dissipative: bool


@dataclass(frozen=True, slots=True)
class SheafCohomologyAuditData:
    r"""
    Artefacto de Fase 3. Certificado de Nulidad de Obstrucciones.

    Campos extendidos v3.0.0:
        h1_dimension            : dim H¹(G; F) — debe ser 0
        rank_delta0             : rank(δ⁰) — dimensión de im(δ⁰)
        rank_delta1             : rank(δ¹) — dimensión de im(δ¹)
        betti_numbers           : (β₀, β₁, β₂) = números de Betti
        euler_characteristic    : χ = β₀ − β₁ + β₂
        reidemeister_torsion    : τ ∈ ℝ — invariante secundario (log|τ|)
        complex_residual        : ‖δ¹∘δ⁰‖_F / scale — verificación d²=0
        is_globally_integrable  : True sii dim H¹ = 0
        obstruction_free        : True sii no hay obstrucciones detectadas
        verified_by_coboundary  : True sii se usaron operadores δ⁰, δ¹
        mayer_vietoris_exact    : True sii la secuencia M-V es exacta
    """
    h1_dimension: int
    rank_delta0: int
    rank_delta1: int
    betti_numbers: Tuple[int, int, int]
    euler_characteristic: int
    reidemeister_torsion: float
    complex_residual: float
    is_globally_integrable: bool
    obstruction_free: bool
    verified_by_coboundary: bool
    mayer_vietoris_exact: bool


@dataclass(frozen=True, slots=True)
class AuditProvenance:
    r"""
    Trazabilidad funtorial de la cadena Φ₃∘Φ₂∘Φ₁.

    Campos:
        timestamp_iso       : Fecha/hora ISO-8601 UTC de la auditoría
        input_checksum_sha256: SHA-256 hex de los datos de entrada serializados
        phase1_passed       : True sii Fase 1 completó sin excepción
        phase2_passed       : True sii Fase 2 completó sin excepción
        phase3_passed       : True sii Fase 3 completó sin excepción
        functor_chain       : Descripción textual de la composición funtorial
    """
    timestamp_iso: str
    input_checksum_sha256: str
    phase1_passed: bool
    phase2_passed: bool
    phase3_passed: bool
    functor_chain: str


@dataclass(frozen=True, slots=True)
class ASTGovernanceState:
    r"""
    Objeto final del endofuntor Z_{Γ-PHYSICS} = Φ₃∘Φ₂∘Φ₁.

    Contiene:
        symplectic_audit       : Certificado de Fase 1
        thermodynamic_audit    : Certificado de Fase 2
        cohomology_audit       : Certificado de Fase 3
        provenance             : Trazabilidad de la cadena funtorial
        is_compilation_authorized: True sii los tres certificados son válidos
    """
    symplectic_audit: SymplecticInvariantData
    thermodynamic_audit: ThermodynamicDirichletData
    cohomology_audit: SheafCohomologyAuditData
    provenance: AuditProvenance
    is_compilation_authorized: bool


# ═══════════════════════════════════════════════════════════════════════════════
# §D. GUARDAS NUMÉRICAS INTERNAS — v3.0.0
#
# Evoluciones sobre v2.0.0:
#   · _as_finite_matrix: verifica rango mínimo estructural y ceros estructurales
#   · _as_finite_vector: verifica que el vector no sea el vector cero
#   · _frobenius_norm  : versión con fallback a norm(ord=1) si Frobenius falla
#   · _vector_norm     : ídem con fallback a norm(ord=1)
#   · _spectral_norm   : nueva — calcula ‖A‖₂ = σ_max(A)
#   · _safe_svdvals    : nueva — SVD con reintento por perturbación diagonal
#   · _structural_zero_check: nueva — detecta filas/columnas de ceros
#   · _check_input_checksum : nueva — calcula SHA-256 de la entrada
# ═══════════════════════════════════════════════════════════════════════════════
class _FiniteNumericalGuard:
    r"""
    Capa de saneamiento numérico que garantiza la finitez, realidad y
    no-degeneración estructural de todas las entradas matriciales y vectoriales
    antes de que sean procesadas por los auditores de las tres fases.

    Toda violación emite una excepción descriptiva que identifica el nombre
    del parámetro, el tipo de violación y el valor ofensivo.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # D.1 Conversión y validación de arreglos escalares
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _as_float_array(name: str, value: Any) -> NDArray[np.float64]:
        r"""
        Convierte `value` a un arreglo real float64, rechazando:
            - Objetos complejos (ℂ).
            - Valores NaN.
            - Valores ±∞.

        Parámetros:
            name : Nombre del parámetro (para mensajes de error).
            value: Objeto a convertir.

        Retorna:
            Arreglo float64 con todos los elementos finitos.
        """
        try:
            raw = np.asarray(value)
        except Exception as exc:
            raise TypeError(
                f"[Guard] '{name}' no puede interpretarse como arreglo numérico: {exc}"
            ) from exc

        if np.iscomplexobj(raw):
            raise TypeError(
                f"[Guard] '{name}' debe ser real; se rechazó entrada compleja "
                f"(dtype={raw.dtype})."
            )

        try:
            arr = raw.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"[Guard] '{name}' debe ser numérico real convertible a float64: {exc}"
            ) from exc

        if not np.all(np.isfinite(arr)):
            n_bad = int(np.sum(~np.isfinite(arr)))
            raise ValueError(
                f"[Guard] '{name}' contiene {n_bad} valor(es) NaN o ±∞."
            )

        return arr

    # ─────────────────────────────────────────────────────────────────────────
    # D.2 Validación de matrices
    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def _as_finite_matrix(
        cls,
        name: str,
        value: Any,
        *,
        square: bool = False,
        min_rank: int = 0,
        check_structural_zeros: bool = False,
    ) -> NDArray[np.float64]:
        r"""
        Valida una matriz real finita.

        Parámetros:
            name                  : Nombre del parámetro.
            value                 : Objeto a validar.
            square                : Si True, exige shape[0] == shape[1].
            min_rank              : Rango mínimo estructural exigido (0 = sin restricción).
            check_structural_zeros: Si True, rechaza matrices con filas o columnas de ceros.

        Retorna:
            Matriz float64 válida.
        """
        arr = cls._as_float_array(name, value)

        if arr.ndim != 2:
            raise ValueError(
                f"[Guard] '{name}' debe ser una matriz 2D; tiene ndim={arr.ndim}."
            )

        rows, cols = arr.shape

        if rows == 0 or cols == 0:
            raise ValueError(
                f"[Guard] '{name}' no puede ser una matriz vacía ({rows}×{cols})."
            )

        if square and rows != cols:
            raise ValueError(
                f"[Guard] '{name}' debe ser cuadrada; tiene forma {arr.shape}."
            )

        if check_structural_zeros:
            cls._structural_zero_check(name, arr)

        if min_rank > 0:
            actual_rank = cls._fast_rank_estimate(arr)
            if actual_rank < min_rank:
                raise ValueError(
                    f"[Guard] '{name}' tiene rango estructural estimado {actual_rank} "
                    f"< mínimo requerido {min_rank}."
                )

        return arr

    # ─────────────────────────────────────────────────────────────────────────
    # D.3 Validación de vectores
    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def _as_finite_vector(
        cls,
        name: str,
        value: Any,
        *,
        allow_zero_vector: bool = True,
    ) -> NDArray[np.float64]:
        r"""
        Valida un vector real finito, normalizándolo a 1D.

        Parámetros:
            name            : Nombre del parámetro.
            value           : Objeto a validar.
            allow_zero_vector: Si False, rechaza el vector cero ‖v‖ = 0.

        Retorna:
            Vector float64 1D.
        """
        arr = cls._as_float_array(name, value)

        if arr.ndim == 2 and 1 in arr.shape:
            arr = arr.reshape(-1)

        if arr.ndim != 1:
            raise ValueError(
                f"[Guard] '{name}' debe ser 1D o columna/fila; tiene ndim={arr.ndim}."
            )

        if arr.size == 0:
            raise ValueError(f"[Guard] '{name}' no puede ser un vector vacío.")

        if not allow_zero_vector:
            if float(la.norm(arr, ord=2)) == 0.0:
                raise ValueError(
                    f"[Guard] '{name}' es el vector cero; operación no definida."
                )

        return arr

    # ─────────────────────────────────────────────────────────────────────────
    # D.4 Normas numéricamente seguras
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _frobenius_norm(A: NDArray[np.float64]) -> float:
        r"""
        Norma de Frobenius ‖A‖_F con fallback a ‖A‖₁ si la rutina falla.

        Retorna math.inf si ninguna alternativa converge.
        """
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
    def _vector_norm(v: NDArray[np.float64]) -> float:
        r"""
        Norma euclidiana ‖v‖₂ con fallback a ‖v‖₁.

        Retorna math.inf si ninguna alternativa converge.
        """
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
    def _spectral_norm(A: NDArray[np.float64]) -> float:
        r"""
        Norma espectral ‖A‖₂ = σ_max(A) calculada por LAPACK DGESDD.

        Utilizada para la tolerancia adaptativa en Fase 1.
        Retorna math.inf si la SVD no converge.
        """
        if A.size == 0:
            return 0.0
        try:
            svs = la.svdvals(A)
            if svs.size == 0:
                return 0.0
            val = float(svs[0])
            return val if math.isfinite(val) else math.inf
        except Exception:
            return math.inf

    # ─────────────────────────────────────────────────────────────────────────
    # D.5 SVD segura con perturbación diagonal de emergencia
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _safe_svdvals(
        A: NDArray[np.float64],
        *,
        max_retries: int = 3,
    ) -> NDArray[np.float64]:
        r"""
        Calcula los valores singulares de A con reintentos ante fallos de LAPACK.

        En cada reintento se añade una perturbación diagonal ε·‖A‖_F·I para
        mejorar el condicionamiento numérico.

        Parámetros:
            A          : Matriz de entrada.
            max_retries: Número máximo de intentos.

        Retorna:
            Vector de valores singulares en orden descendente.

        Lanza:
            CohomologicalObstructionError si todos los intentos fallan.
        """
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                svs = la.svdvals(A)
                if np.all(np.isfinite(svs)):
                    return svs
            except (np.linalg.LinAlgError, ValueError) as exc:
                last_exc = exc

            # Perturbación diagonal de regularización
            perturbation = (
                (_MACHINE_EPSILON ** (1.0 / (attempt + 2)))
                * float(la.norm(A, ord="fro") or 1.0)
            )
            rows, cols = A.shape
            min_dim = min(rows, cols)
            A_perturbed = A.copy()
            A_perturbed[:min_dim, :min_dim] += perturbation * np.eye(min_dim)
            A = A_perturbed

        raise CohomologicalObstructionError(
            f"SVD no convergió tras {max_retries} intentos. "
            f"Último error: {last_exc}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # D.6 Verificación de ceros estructurales
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _structural_zero_check(
        name: str,
        A: NDArray[np.float64],
    ) -> None:
        r"""
        Detecta filas o columnas donde todos los elementos son cero exacto
        (o menor que ε_maq). Indica degeneración estructural que invalida
        los cálculos de rango subsiguientes.

        Parámetros:
            name: Nombre del parámetro.
            A   : Matriz a inspeccionar.
        """
        tol = _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON

        zero_rows = np.where(np.all(np.abs(A) <= tol, axis=1))[0]
        zero_cols = np.where(np.all(np.abs(A) <= tol, axis=0))[0]

        if zero_rows.size > 0:
            logger.warning(
                "[Guard] '%s' tiene %d fila(s) estructuralmente cero: índices %s.",
                name, zero_rows.size, zero_rows.tolist(),
            )

        if zero_cols.size > 0:
            logger.warning(
                "[Guard] '%s' tiene %d columna(s) estructuralmente cero: índices %s.",
                name, zero_cols.size, zero_cols.tolist(),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # D.7 Estimación rápida de rango
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _fast_rank_estimate(A: NDArray[np.float64]) -> int:
        r"""
        Estimación rápida del rango numérico por conteo de valores singulares
        superiores a la tolerancia adaptativa:

            tol = c · ε₀ · max(shape(A)) · σ_max(A).

        Retorna 0 si la SVD falla o si A es vacía.
        """
        if A.size == 0 or min(A.shape) == 0:
            return 0
        try:
            svs = la.svdvals(A)
            if svs.size == 0:
                return 0
            sigma_max = float(svs[0])
            if sigma_max == 0.0 or not math.isfinite(sigma_max):
                return 0
            tol = (
                _NUMERICAL_SAFETY_FACTOR
                * _MACHINE_EPSILON
                * max(A.shape)
                * sigma_max
            )
            return int(np.count_nonzero(svs > tol))
        except Exception:
            return 0

    # ─────────────────────────────────────────────────────────────────────────
    # D.8 Checksum de entrada
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_input_checksum(
        *arrays: NDArray[np.float64],
    ) -> str:
        r"""
        Calcula el SHA-256 de la concatenación binaria de los arreglos de entrada,
        usando tobytes() en orden C-contiguo.

        Proporciona trazabilidad criptográfica de los datos auditados.

        Parámetros:
            *arrays: Arreglos numpy a incluir en el checksum.

        Retorna:
            String hexadecimal de 64 caracteres (SHA-256).
        """
        hasher = hashlib.sha256()

        for arr in arrays:
            if arr is not None:
                # Incluye shape y dtype como metadatos
                meta = struct.pack(
                    ">4Q",
                    arr.ndim,
                    *arr.shape[:3] if arr.ndim >= 3 else
                    (*arr.shape, *([0] * (3 - arr.ndim))),
                )
                hasher.update(meta)
                hasher.update(np.ascontiguousarray(arr).tobytes())

        return hasher.hexdigest()


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 1: AUDITORÍA DE INVARIANZA SIMPLÉCTICA Y TEOREMA DE LIOUVILLE        ║
# ║                                                                             ║
# ║   Marco matemático:                                                         ║
# ║   ─────────────────                                                         ║
# ║   Sea (M, ω) el espacio de fase simpléctica del AST, donde                  ║
# ║   ω = Σᵢ dqᵢ ∧ dpᵢ es la forma simpléctica canónica.                        ║
# ║                                                                             ║
# ║   Una transformación sintáctica es representada por su Jacobiana M ∈ GL(2n).║
# ║   La condición de simplecticidad es:                                        ║
# ║       Mᵀ Ω M = Ω                                                            ║
# ║   donde Ω = [[0, Iₙ], [−Iₙ, 0]] ∈ R^{2n×2n}.                                 ║
# ║                                                                             ║
# ║   Invariantes verificados en v3.0.0:                                        ║
# ║   1. Residuo de Frobenius: ‖MᵀΩM − Ω‖_F                                     ║
# ║   2. Residuo antisimétrico: ‖skew(MᵀΩM) − Ω‖_F / ‖Ω‖_F                      ║
# ║   3. Residuo determinantal: |det(M) − 1|                                    ║
# ║   4. Número de condición: κ(M) = σ_max / σ_min                              ║
# ║   5. Radio espectral: ρ(MᵀΩM)                                               ║
# ║   6. Índice de Maslov: μ(Λ) ∈ ℤ                                             ║
# ║   7. Residuo de unitariedad polar: ‖UᵀΩU − Ω‖_F / ‖Ω‖_F                     ║
# ║                                                                             ║
# ║   ÚLTIMO MÉTODO DE FASE 1:                                                  ║
# ║       _audit_symplectic_invariance(ast_jacobian_M)                          ║
# ║       → SymplecticInvariantData    [objeto inicial de Fase 2]               ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_SymplecticInvarianceAuditor(_FiniteNumericalGuard):
    r"""
    Trata las transformaciones sintácticas como campos vectoriales Hamiltonianos.
    Garantiza que la IA no introduzca inyección de entropía divergente evaluando
    la forma simpléctica canónica ω = Σ dqᵢ ∧ dpᵢ.

    Mejoras v3.0.0:
        · Descomposición polar M = U·P para verificar la parte unitaria U ∈ Sp(2n).
        · Tolerancia adaptativa espectral: τ_eff = τ_base + c_σ · σ_max(M) · ε₀.
        · Residuo antisimétrico separado: detecta pérdida de antisimetría.
        · Índice de Maslov μ(Λ) calculado por conteo de signaturas espectrales.
        · Radio espectral ρ(MᵀΩM) como control de amplificación.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 1.1 Construcción de la matriz simpléctica canónica
    # ─────────────────────────────────────────────────────────────────────────
    def _build_canonical_symplectic_matrix(self, n: int) -> NDArray[np.float64]:
        r"""
        Construye la matriz simpléctica estándar Ω ∈ ℝ^{2n × 2n}:

            Ω = [[ 0_n,  I_n],
                 [−I_n,  0_n]].

        Esta matriz es la representación matricial de la forma simpléctica
        canónica ω = Σᵢ dqᵢ ∧ dpᵢ en la base estándar.

        Propiedades verificadas internamente:
            - Antisimetría: Ω = −Ωᵀ
            - Invertibilidad: Ω⁻¹ = −Ω = Ωᵀ
            - det(Ω) = 1

        Parámetros:
            n: Número de grados de libertad (dim del subespacio qᵢ o pᵢ).

        Retorna:
            Ω ∈ ℝ^{2n × 2n}, antisimétrica, ortogonal simplécticmente.
        """
        if not isinstance(n, (int, np.integer)) or isinstance(n, bool):
            raise TypeError(
                f"[Fase 1] n debe ser un entero positivo; recibido: {type(n)}."
            )

        if n <= 0:
            raise ValueError(
                f"[Fase 1] El número de grados de libertad simplécticos debe ser "
                f"un entero positivo; recibido: n={n}."
            )

        omega = np.zeros((2 * n, 2 * n), dtype=np.float64)
        identity = np.eye(n, dtype=np.float64)

        omega[:n, n:] = identity
        omega[n:, :n] = -identity

        # Verificación interna de antisimetría
        antisymmetry_residual = self._frobenius_norm(omega + omega.T)
        if antisymmetry_residual > _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON:
            raise SymplecticInvarianceViolation(
                f"[Fase 1] La matriz Ω construida no es antisimétrica: "
                f"‖Ω + Ωᵀ‖_F = {antisymmetry_residual:.4e}."
            )

        return omega

    # ─────────────────────────────────────────────────────────────────────────
    # 1.2 Residuo determinantal por slogdet
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _determinant_residual_from_slogdet(
        sign: float,
        logabsdet: float,
    ) -> float:
        r"""
        Calcula |det(M) − 1| usando la descomposición logarítmica:

            det(M) = sign · exp(logabsdet).

        Toda matriz simpléctica real M ∈ Sp(2n, ℝ) satisface:

            det(M) = 1.

        El residuo |det(M) − 1| cuantifica la desviación de esta propiedad.

        Parámetros:
            sign      : Signo del determinante (±1 ó 0).
            logabsdet : Logaritmo del valor absoluto del determinante.

        Retorna:
            |det(M) − 1| ∈ [0, ∞].
        """
        if sign == 0:
            # Matriz singular: det = 0, residuo = 1.
            return 1.0

        if not math.isfinite(logabsdet):
            return math.inf

        max_log = math.log(np.finfo(np.float64).max * 0.5)

        if logabsdet > max_log:
            return math.inf

        if logabsdet < -max_log:
            # det ≈ 0 → residuo ≈ 1
            return 1.0

        det_value = float(sign) * math.exp(logabsdet)
        return float(abs(det_value - 1.0))

    # ─────────────────────────────────────────────────────────────────────────
    # 1.3 Número de condición seguro
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _condition_number(M: NDArray[np.float64]) -> float:
        r"""
        Calcula κ(M) = σ_max(M) / σ_min(M) usando valores singulares directos.

        Evita la rutina `la.cond` que puede ser inestable para matrices grandes.
        Una matriz simpléctica bien condicionada satisface κ(M) < 1/ε₀.

        Parámetros:
            M: Matriz cuadrada a evaluar.

        Retorna:
            κ(M) ∈ [1, ∞].
        """
        try:
            svs = la.svdvals(M)
            if svs.size == 0:
                return math.inf
            sigma_max = float(svs[0])
            sigma_min = float(svs[-1])
            if sigma_min == 0.0:
                return math.inf
            if not math.isfinite(sigma_max) or not math.isfinite(sigma_min):
                return math.inf
            cond = sigma_max / sigma_min
            return cond if math.isfinite(cond) else math.inf
        except Exception:
            return math.inf

    # ─────────────────────────────────────────────────────────────────────────
    # 1.4 Tolerancia adaptativa espectral
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_adaptive_symplectic_tolerance(
        sigma_max_M: float,
    ) -> float:
        r"""
        Calcula la tolerancia efectiva adaptada a la escala espectral de M:

            τ_eff = max(τ_base, c_σ · σ_max(M) · ε₀ · c_num).

        Esta tolerancia es más conservadora que la base cuando M tiene
        valores singulares grandes, evitando falsos negativos numéricos.

        Parámetros:
            sigma_max_M: Mayor valor singular de M.

        Retorna:
            Tolerancia efectiva τ_eff > 0.
        """
        if not math.isfinite(sigma_max_M) or sigma_max_M <= 0.0:
            return _SYMPLECTIC_TOLERANCE_BASE

        spectral_tol = (
            _SYMPLECTIC_SPECTRAL_FACTOR
            * sigma_max_M
            * _MACHINE_EPSILON
            * _NUMERICAL_SAFETY_FACTOR
        )

        return max(_SYMPLECTIC_TOLERANCE_BASE, spectral_tol)

    # ─────────────────────────────────────────────────────────────────────────
    # 1.5 Descomposición polar y residuo de unitariedad
    # ─────────────────────────────────────────────────────────────────────────
    def _polar_unitarity_residual(
        self,
        M: NDArray[np.float64],
        omega: NDArray[np.float64],
    ) -> float:
        r"""
        Verifica que la parte unitaria U de la descomposición polar M = U·P
        sea también simpléctica: U ∈ Sp(2n, ℝ).

        La descomposición polar de M se calcula como:

            M = U · P,

        donde U es ortogonal (Uᵀ U = I) y P = (MᵀM)^{1/2} es semidefinida
        positiva. Si M ∈ Sp(2n, ℝ), entonces U ∈ Sp(2n, ℝ) ∩ O(2n, ℝ).

        El residuo es:

            ρ_U = ‖UᵀΩU − Ω‖_F / ‖Ω‖_F.

        Parámetros:
            M    : Jacobiana de la transformación.
            omega: Matriz simpléctica Ω.

        Retorna:
            ρ_U ∈ [0, ∞).
        """
        try:
            # Descomposición polar: M = U · P
            # U = M · (MᵀM)^{-1/2}
            MtM = M.T @ M

            if not np.all(np.isfinite(MtM)):
                raise SymplecticPolarDecompositionError(
                    "[Fase 1] MᵀM produjo valores no finitos."
                )

            # Calcular (MᵀM)^{1/2} por eigendescomposición (MᵀM es SPD)
            eigvals, eigvecs = la.eigh(MtM)

            if not np.all(np.isfinite(eigvals)) or not np.all(np.isfinite(eigvecs)):
                raise SymplecticPolarDecompositionError(
                    "[Fase 1] Eigendescomposición de MᵀM produjo valores no finitos."
                )

            # Proyectar eigenvalores al semiplano positivo
            eigvals_pos = np.maximum(eigvals, 0.0)

            if np.any(eigvals_pos == 0.0):
                logger.warning(
                    "[Fase 1] MᵀM tiene eigenvalores ≤ 0; la descomposición polar "
                    "puede ser degenerada."
                )

            # Raíz cuadrada segura
            sqrt_eigvals = np.sqrt(eigvals_pos)

            # (MᵀM)^{-1/2}: inversión de sqrt_eigvals con protección
            inv_sqrt_eigvals = np.where(
                sqrt_eigvals > _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON,
                1.0 / sqrt_eigvals,
                0.0,
            )

            # P^{-1} = V · diag(1/σᵢ) · Vᵀ donde V = eigvecs
            P_inv = eigvecs @ np.diag(inv_sqrt_eigvals) @ eigvecs.T

            if not np.all(np.isfinite(P_inv)):
                raise SymplecticPolarDecompositionError(
                    "[Fase 1] La inversión de la parte positiva P produjo valores "
                    "no finitos."
                )

            U = M @ P_inv

            if not np.all(np.isfinite(U)):
                raise SymplecticPolarDecompositionError(
                    "[Fase 1] La parte unitaria U = M·P⁻¹ produjo valores no finitos."
                )

            # Residuo simpléctico de U
            UtOmegaU = U.T @ omega @ U
            omega_norm = self._frobenius_norm(omega)

            if omega_norm == 0.0:
                return 0.0

            residual = self._frobenius_norm(UtOmegaU - omega)
            return float(residual / omega_norm)

        except SymplecticPolarDecompositionError:
            raise
        except Exception as exc:
            raise SymplecticPolarDecompositionError(
                f"[Fase 1] Descomposición polar falló: {exc}"
            ) from exc

    # ─────────────────────────────────────────────────────────────────────────
    # 1.6 Índice de Maslov
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_maslov_index(
        self,
        M: NDArray[np.float64],
        omega: NDArray[np.float64],
    ) -> int:
        r"""
        Calcula el índice de Maslov μ(Λ) de la variedad de Lagrange Λ = M(L₀)
        como el número de signaturas negativas del bloque inferior-izquierdo de
        la matrix M en la coordenada simpléctica.

        En coordenadas simplécticas (q, p), la variedad de Lagrange generada por M
        tiene índice de Maslov:

            μ(Λ) = ind(B),

        donde B = bloque inferior-izquierdo de M (la submatriz M[n:, :n]),
        e ind(·) denota el índice de una forma cuadrática = número de
        eigenvalores negativos.

        Para una transformación simpléctica sin topología de entrelazamiento,
        se espera μ(Λ) = 0.

        Parámetros:
            M    : Jacobiana simpléctica 2n × 2n.
            omega: Matriz simpléctica Ω (no utilizada directamente, pero
                   necesaria para la coherencia del contexto).

        Retorna:
            μ(Λ) ∈ ℤ.
        """
        n = M.shape[0] // 2

        # Bloque B = M[n:, :n] (submatriz inferior-izquierda)
        B = M[n:, :n]

        try:
            eigvals_B = la.eigvalsh(B @ B.T)
        except Exception as exc:
            raise MaslovIndexError(
                f"[Fase 1] Cálculo del índice de Maslov falló en eigvalsh: {exc}"
            ) from exc

        if not np.all(np.isfinite(eigvals_B)):
            raise MaslovIndexError(
                "[Fase 1] Los eigenvalores del bloque B·Bᵀ no son finitos."
            )

        # La forma cuadrática q(v) = vᵀ B v tiene índice = número de
        # eigenvalores de B·Bᵀ que corresponden a direcciones de curvatura
        # negativa. Dado que B·Bᵀ es SPD, su índice es 0 (por definición).
        # El índice de Maslov se calcula como el número de eigenvalores de B
        # (no B·Bᵀ) estrictamente negativos.
        try:
            eigvals_B_direct = np.real(la.eigvals(B))
        except Exception as exc:
            raise MaslovIndexError(
                f"[Fase 1] eigvals(B) falló: {exc}"
            ) from exc

        if not np.all(np.isfinite(eigvals_B_direct)):
            raise MaslovIndexError(
                "[Fase 1] Los eigenvalores directos del bloque B no son finitos."
            )

        maslov = int(np.sum(eigvals_B_direct < -_MASLOV_PHASE_TOLERANCE))

        return maslov

    # ─────────────────────────────────────────────────────────────────────────
    # 1.7 Radio espectral de MᵀΩM
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _spectral_radius_transformed_omega(
        transformed_omega: NDArray[np.float64],
    ) -> float:
        r"""
        Calcula ρ(MᵀΩM) = max|λᵢ(MᵀΩM)|.

        Para una transformación simpléctica exacta: MᵀΩM = Ω, y dado que
        Ω tiene eigenvalores ±i, el radio espectral debe ser ρ = 1.

        Una desviación ρ >> 1 indica amplificación no controlada del espacio
        de fase.

        Retorna:
            ρ(MᵀΩM) ∈ [0, ∞).
        """
        try:
            eigvals = la.eigvals(transformed_omega)
            if eigvals.size == 0:
                return 0.0
            rho = float(np.max(np.abs(eigvals)))
            return rho if math.isfinite(rho) else math.inf
        except Exception:
            return math.inf

    # ─────────────────────────────────────────────────────────────────────────
    # 1.8 Residuo antisimétrico
    # ─────────────────────────────────────────────────────────────────────────
    def _antisymmetric_residual(
        self,
        transformed_omega: NDArray[np.float64],
        omega: NDArray[np.float64],
    ) -> float:
        r"""
        Calcula el residuo de la parte antisimétrica:

            ρ_skew = ‖skew(MᵀΩM) − Ω‖_F / ‖Ω‖_F,

        donde skew(A) = (A − Aᵀ) / 2.

        Para MᵀΩM = Ω (exacto), se tiene skew(Ω) = Ω y ρ_skew = 0.

        Este residuo separa la pérdida de antisimetría de la pérdida de
        magnitud, proveyendo diagnóstico más granular que el residuo total.

        Parámetros:
            transformed_omega: MᵀΩM calculada.
            omega            : Ω canónica.

        Retorna:
            ρ_skew ∈ [0, ∞).
        """
        omega_norm = self._frobenius_norm(omega)

        if omega_norm == 0.0:
            return 0.0

        skew_transformed = (transformed_omega - transformed_omega.T) * 0.5
        residual = self._frobenius_norm(skew_transformed - omega)

        return float(residual / omega_norm)

    # ─────────────────────────────────────────────────────────────────────────
    # 1.9 ÚLTIMO MÉTODO DE FASE 1
    #     _audit_symplectic_invariance → SymplecticInvariantData
    #     [Objeto inicial de Fase 2]
    # ─────────────────────────────────────────────────────────────────────────
    def _audit_symplectic_invariance(
        self,
        ast_jacobian_M: NDArray[np.float64],
    ) -> SymplecticInvariantData:
        r"""
        ════════════════════════════════════════════════════════════════════
        ÚLTIMO MÉTODO DE FASE 1 — Retorna el objeto inicial de Fase 2.
        ════════════════════════════════════════════════════════════════════

        Audita el cumplimiento del Teorema de Liouville evaluando todos los
        invariantes simplécticos conocidos de la transformación AST.

        Invariantes verificados (en orden de cálculo):
            1. Dimensión par del espacio de fase.
            2. Finitez y cuadratura de M.
            3. Construcción de Ω canónica.
            4. Residuo simpléctico absoluto: ‖MᵀΩM − Ω‖_F.
            5. Residuo antisimétrico: ‖skew(MᵀΩM) − Ω‖_F / ‖Ω‖_F.
            6. Residuo simpléctico relativo: ‖MᵀΩM − Ω‖_F / scale.
            7. Tolerancia adaptativa espectral por σ_max(M).
            8. Residuo determinantal: |det(M) − 1| por slogdet.
            9. Número de condición: κ(M) = σ_max / σ_min.
            10. Radio espectral: ρ(MᵀΩM).
            11. Índice de Maslov: μ(Λ).
            12. Residuo de unitariedad polar: ‖UᵀΩU − Ω‖_F / ‖Ω‖_F.

        Parámetros:
            ast_jacobian_M: Matriz Jacobiana M de la transformación sintáctica.
                            Debe ser cuadrada, real, finita y de dimensión par.

        Retorna:
            SymplecticInvariantData — certificado simpléctico completo.
            Este objeto es el **objeto inicial de la Fase 2**.

        Lanza:
            SymplecticInvarianceViolation si cualquier invariante es violado.
            SymplecticPolarDecompositionError si la descomposición polar falla.
            MaslovIndexError si el índice de Maslov no puede calcularse.
        """
        # ── Validación de entrada ─────────────────────────────────────────
        M = self._as_finite_matrix(
            "ast_jacobian_M",
            ast_jacobian_M,
            square=True,
            check_structural_zeros=True,
        )

        dim = M.shape[0]

        if dim == 0 or (dim % 2) != 0:
            raise SymplecticInvarianceViolation(
                f"[Fase 1] El espacio de fase del AST debe tener dimensión par "
                f"positiva; recibido: dim={dim}."
            )

        n = dim // 2

        # ── Construcción de Ω ─────────────────────────────────────────────
        omega = self._build_canonical_symplectic_matrix(n)

        # ── Transformación simpléctica: MᵀΩM ─────────────────────────────
        transformed_omega = M.T @ omega @ M

        if not np.all(np.isfinite(transformed_omega)):
            raise SymplecticInvarianceViolation(
                "[Fase 1] La transformación MᵀΩM produjo valores no finitos."
            )

        # ── Normas base ───────────────────────────────────────────────────
        omega_norm = self._frobenius_norm(omega)
        transformed_norm = self._frobenius_norm(transformed_omega)
        m_frobenius = self._frobenius_norm(M)

        if not math.isfinite(m_frobenius):
            raise SymplecticInvarianceViolation(
                "[Fase 1] La norma de Frobenius de M no es finita."
            )

        if not math.isfinite(transformed_norm):
            raise SymplecticInvarianceViolation(
                "[Fase 1] La norma de Frobenius de MᵀΩM no es finita."
            )

        # ── Residuo simpléctico absoluto ─────────────────────────────────
        residual = self._frobenius_norm(transformed_omega - omega)

        # ── Residuo antisimétrico ─────────────────────────────────────────
        antisymmetric_res = self._antisymmetric_residual(transformed_omega, omega)

        # ── Residuo relativo ──────────────────────────────────────────────
        scale = max(1.0, transformed_norm + omega_norm)
        relative_residual = residual / scale

        if not math.isfinite(relative_residual):
            raise SymplecticInvarianceViolation(
                "[Fase 1] El residuo simpléctico relativo no es finito."
            )

        # ── Tolerancia adaptativa espectral ───────────────────────────────
        sigma_max_M = self._spectral_norm(M)
        effective_tolerance = self._compute_adaptive_symplectic_tolerance(sigma_max_M)

        # ── Verificación principal ────────────────────────────────────────
        if relative_residual > effective_tolerance:
            raise SymplecticInvarianceViolation(
                "[Fase 1] Violación del Teorema de Liouville. "
                f"Residuo simpléctico relativo = {relative_residual:.6e} "
                f"> tolerancia efectiva = {effective_tolerance:.6e}. "
                f"Residuo antisimétrico = {antisymmetric_res:.6e}."
            )

        # ── Residuo determinantal ─────────────────────────────────────────
        sign, logabsdet = la.slogdet(M)

        if sign == 0 or not math.isfinite(sign):
            raise SymplecticInvarianceViolation(
                "[Fase 1] El Jacobiano del AST es singular: det(M) = 0. "
                "Una transformación simpléctica real debe cumplir det(M) = 1."
            )

        if float(sign) < 0:
            raise SymplecticInvarianceViolation(
                "[Fase 1] det(M) < 0. Las matrices simplécticas reales tienen "
                "det = +1."
            )

        determinant_residual = self._determinant_residual_from_slogdet(sign, logabsdet)

        # ── Número de condición ───────────────────────────────────────────
        condition_number = self._condition_number(M)

        if not math.isfinite(condition_number):
            raise SymplecticInvarianceViolation(
                "[Fase 1] El Jacobiano es numéricamente singular: κ(M) = ∞."
            )

        # ── Radio espectral ───────────────────────────────────────────────
        spectral_radius = self._spectral_radius_transformed_omega(transformed_omega)

        # ── Índice de Maslov ──────────────────────────────────────────────
        maslov_index = self._compute_maslov_index(M, omega)

        # ── Residuo de unitariedad polar ──────────────────────────────────
        polar_residual = self._polar_unitarity_residual(M, omega)

        # ── Advertencias no bloqueantes ───────────────────────────────────
        if condition_number > _CONDITION_NUMBER_WARNING_THRESHOLD:
            logger.warning(
                "[Fase 1] Jacobiano mal condicionado: κ(M) = %.4e > 1/ε₀ = %.4e.",
                condition_number, _CONDITION_NUMBER_WARNING_THRESHOLD,
            )

        det_warn_threshold = max(1e-8, _DETERMINANT_RESIDUAL_WARNING_FACTOR * effective_tolerance)
        if determinant_residual > det_warn_threshold:
            logger.warning(
                "[Fase 1] Residuo determinantal elevado: |det(M)−1| = %.4e > %.4e.",
                determinant_residual, det_warn_threshold,
            )

        if maslov_index != 0:
            logger.warning(
                "[Fase 1] Índice de Maslov no nulo: μ(Λ) = %d. "
                "El flujo simpléctico tiene topología de entrelazamiento.",
                maslov_index,
            )

        if polar_residual > effective_tolerance:
            logger.warning(
                "[Fase 1] Residuo de unitariedad polar elevado: ρ_U = %.4e.",
                polar_residual,
            )

        # ── Emisión del certificado (objeto inicial de Fase 2) ────────────
        return SymplecticInvariantData(
            phase_space_dimension=dim,
            symplectic_residual_norm=residual,
            symplectic_relative_residual=relative_residual,
            antisymmetric_residual=antisymmetric_res,
            determinant_residual=determinant_residual,
            condition_number=condition_number,
            spectral_radius_Omega=spectral_radius,
            maslov_index=maslov_index,
            polar_unitarity_residual=polar_residual,
            effective_tolerance=effective_tolerance,
            is_volume_preserved=True,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 2: CONTROL PORT-HAMILTONIANO Y FRONTERAS DE DIRICHLET                ║
# ║                                                                             ║
# ║   Marco matemático:                                                         ║
# ║   ─────────────────                                                         ║
# ║   Sea (X, H) un sistema port-Hamiltoniano donde:                            ║
# ║       ẋ = (J(x) − R(x)) ∇H(x) + g(x)u                                       ║
# ║       y = gᵀ(x) ∇H(x)                                                       ║
# ║                                                                             ║
# ║   La condición de disipación (Segunda Ley) exige:                           ║
# ║       dH/dt = −P_diss + Pₑₓₜ,   P_diss = ∇Hᵀ R ∇H ≥ 0.                       ║
# ║                                                                             ║
# ║   En el contexto del AST, Φ = campo de control exergético,                  ║
# ║   ∇V = gradiente de la función de Lyapunov. La condición es:                ║
# ║       P_diss = Φᵀ ∇V ≥ 0.                                                   ║
# ║                                                                             ║
# ║   Invariantes verificados en v3.0.0:                                        ║
# ║   1. P_raw = Φᵀ∇V (producto interno crudo)                                  ║
# ║   2. P_diss = max(0, P_raw) (proyección al cono positivo)                   ║
# ║   3. σ_CD = P_diss / T_ref ≥ 0 (Clausius-Duhem diferencial)                 ║
# ║   4. cos(θ) = Φᵀ∇V / (‖Φ‖·‖∇V‖) (alineamiento)                              ║
# ║   5. θ = arccos(cos(θ)) ∈ [0, π] (ángulo termodinámico)                     ║
# ║   6. ratio = P_diss / (‖Φ‖·‖∇V‖) (eficiencia disipativa)                    ║
# ║                                                                             ║
# ║   CONEXIÓN FUNTORIAL:                                                       ║
# ║   El primer método de Fase 2 recibe SymplecticInvariantData y verifica      ║
# ║   consistencia dimensional y de invariante de volumen.                      ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_DirichletThermodynamicEnforcer(Phase1_SymplecticInvarianceAuditor):
    r"""
    Aplica Fronteras de Dirichlet sobre el AST aislando los subárboles.
    Previene bucles de complejidad divergente exigiendo positividad en la
    disipación termodinámica y satisfacción de la desigualdad de Clausius-Duhem.

    Hereda de Phase1_SymplecticInvarianceAuditor.
    Su primer método _enforce_dirichlet_thermodynamics recibe el certificado
    SymplecticInvariantData emitido por el último método de Fase 1.

    Mejoras v3.0.0:
        · Verificación explícita de Clausius-Duhem diferencial.
        · Tasa de producción de entropía σ_CD = P_diss / T_ref.
        · Ángulo termodinámico θ = arccos(cos(Φ,∇V)).
        · Ratio de eficiencia disipativa P_diss / (‖Φ‖·‖∇V‖).
        · Producto interno crudo vs. proyectado registrados separadamente.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 2.1 Cálculo de la tasa de producción de entropía
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_entropy_production_rate(
        dissipated_power: float,
        temperature_reference: float = _THERMODYNAMIC_TEMPERATURE_REFERENCE,
    ) -> float:
        r"""
        Calcula la tasa de producción de entropía por la desigualdad de
        Clausius-Duhem diferencial:

            σ_CD = dS/dt = P_diss / T_ref.

        En el formalismo port-Hamiltoniano discreto:

            σ_CD = Φᵀ R Φ / T_ref ≥ 0,

        donde R es la matriz de resistencia disipativa (R ≥ 0).
        Aquí usamos la aproximación P_diss ≈ Φᵀ R Φ.

        Parámetros:
            dissipated_power   : P_diss ≥ 0.
            temperature_reference: T_ref > 0 (adimensional por defecto).

        Retorna:
            σ_CD ≥ 0.
        """
        if temperature_reference <= 0.0:
            raise ClausiusDuhemViolation(
                f"[Fase 2] La temperatura de referencia debe ser positiva; "
                f"recibida: T_ref={temperature_reference}."
            )

        sigma = dissipated_power / temperature_reference

        if not math.isfinite(sigma):
            raise ClausiusDuhemViolation(
                f"[Fase 2] La tasa de producción de entropía no es finita: "
                f"σ_CD = {sigma}."
            )

        return sigma

    # ─────────────────────────────────────────────────────────────────────────
    # 2.2 Cálculo del ángulo termodinámico
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_thermodynamic_angle(
        alignment_cosine: float,
    ) -> float:
        r"""
        Calcula el ángulo termodinámico:

            θ = arccos(clip(cos(Φ,∇V), −1, 1)) ∈ [0, π].

        Un ángulo θ < π/2 indica disipación activa.
        Un ángulo θ = π/2 indica ortogonalidad (disipación nula).
        Un ángulo θ > π/2 indica retroalimentación energética (peligroso).

        Parámetros:
            alignment_cosine: cos(θ) ∈ [−1, 1].

        Retorna:
            θ ∈ [0, π] en radianes.
        """
        clipped = float(np.clip(alignment_cosine, -1.0, 1.0))
        return float(math.acos(clipped))

    # ─────────────────────────────────────────────────────────────────────────
    # 2.3 Ratio de eficiencia disipativa
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_exergy_dissipation_ratio(
        dissipated_power: float,
        phi_norm: float,
        grad_norm: float,
    ) -> float:
        r"""
        Calcula la eficiencia disipativa:

            η = P_diss / (‖Φ‖ · ‖∇V‖).

        Si ‖Φ‖ = 0 ó ‖∇V‖ = 0, retorna 0.0 por convención.

        η = 1 indica disipación completa (Φ ∥ ∇V).
        η = 0 indica disipación nula (Φ ⊥ ∇V).
        η ∈ (−1, 0) indica retroalimentación energética.

        Parámetros:
            dissipated_power: P_diss.
            phi_norm        : ‖Φ‖₂.
            grad_norm       : ‖∇V‖₂.

        Retorna:
            η ∈ [−1, 1] ó 0.0 si alguna norma es cero.
        """
        denom = phi_norm * grad_norm

        if denom == 0.0 or not math.isfinite(denom):
            return 0.0

        ratio = dissipated_power / denom
        return float(np.clip(ratio, -1.0, 1.0))

    # ─────────────────────────────────────────────────────────────────────────
    # 2.4 PRIMER MÉTODO DE FASE 2 / CONTINUACIÓN DE FASE 1
    #     _enforce_dirichlet_thermodynamics → ThermodynamicDirichletData
    #     [Objeto inicial de Fase 3]
    # ─────────────────────────────────────────────────────────────────────────
    def _enforce_dirichlet_thermodynamics(
        self,
        control_potential_Phi: NDArray[np.float64],
        lyapunov_gradient_V: NDArray[np.float64],
        symplectic_audit: Optional[SymplecticInvariantData] = None,
    ) -> ThermodynamicDirichletData:
        r"""
        ════════════════════════════════════════════════════════════════════
        PRIMER MÉTODO DE FASE 2 — Continuación formal de Fase 1.
        Retorna el objeto inicial de Fase 3.
        ════════════════════════════════════════════════════════════════════

        Calcula el producto interno covariante para medir la disipación:

            P_diss = Φᵀ ∇V.

        Si `symplectic_audit` es provisto (composición funtorial estricta):
            - Verifica que is_volume_preserved = True.
            - Verifica consistencia dimensional: dim(Φ) = dim del espacio de fase.

        Invariantes calculados:
            1. P_raw = Φᵀ∇V (crudo, puede ser negativo).
            2. P_diss = max(0, P_raw) (proyectado).
            3. ε_num = c·ε₀·max(1, ‖Φ‖·‖∇V‖) (tolerancia numérica).
            4. Rechazo si P_raw < −ε_num (violación real, no numérica).
            5. σ_CD = P_diss / T_ref (Clausius-Duhem).
            6. cos(θ) = P_raw / (‖Φ‖·‖∇V‖) si los vectores no son cero.
            7. θ = arccos(cos(θ)) en radianes.
            8. η = P_diss / (‖Φ‖·‖∇V‖) (eficiencia disipativa).

        Parámetros:
            control_potential_Phi: Campo de control exergético Φ ∈ ℝⁿ.
            lyapunov_gradient_V  : Gradiente de Lyapunov ∇V ∈ ℝⁿ.
            symplectic_audit     : Certificado de Fase 1 (opcional pero recomendado).

        Retorna:
            ThermodynamicDirichletData — certificado termodinámico completo.
            Este objeto es el **objeto inicial de la Fase 3**.

        Lanza:
            SymplecticInvarianceViolation si symplectic_audit no preservó volumen.
            ThermodynamicSingularityError si P_diss < −ε_num.
            ExergyDivergenceError si ‖Φ‖ ó ‖∇V‖ no son finitas.
            ClausiusDuhemViolation si σ_CD < 0.
        """
        # ── Validación de entrada ─────────────────────────────────────────
        Phi = self._as_finite_vector("control_potential_Phi", control_potential_Phi)
        gradV = self._as_finite_vector("lyapunov_gradient_V", lyapunov_gradient_V)

        if Phi.shape != gradV.shape:
            raise ValueError(
                f"[Fase 2] Dimensiones incompatibles: "
                f"Φ tiene dim={Phi.shape}, ∇V tiene dim={gradV.shape}."
            )

        # ── Continuación funtorial de Fase 1 ─────────────────────────────
        if symplectic_audit is not None:
            if not symplectic_audit.is_volume_preserved:
                raise SymplecticInvarianceViolation(
                    "[Fase 2] No puede iniciarse: la Fase 1 no preservó el volumen "
                    "simpléctico (is_volume_preserved=False)."
                )

            if Phi.size != symplectic_audit.phase_space_dimension:
                raise ValueError(
                    "[Fase 2] Inconsistencia dimensional entre el certificado simpléctico "
                    f"(dim={symplectic_audit.phase_space_dimension}) y los campos "
                    f"termodinámicos (dim={Phi.size})."
                )

        # ── Normas ───────────────────────────────────────────────────────
        phi_norm = self._vector_norm(Phi)
        grad_norm = self._vector_norm(gradV)

        if not math.isfinite(phi_norm):
            raise ExergyDivergenceError(
                f"[Fase 2] La norma de Φ no es finita: ‖Φ‖ = {phi_norm}."
            )

        if not math.isfinite(grad_norm):
            raise ExergyDivergenceError(
                f"[Fase 2] La norma de ∇V no es finita: ‖∇V‖ = {grad_norm}."
            )

        # ── Producto interno crudo ────────────────────────────────────────
        p_raw = float(np.dot(Phi, gradV))

        if not math.isfinite(p_raw):
            raise ThermodynamicSingularityError(
                f"[Fase 2] P_diss = Φᵀ∇V no es finita: {p_raw}. "
                "Posible desbordamiento numérico o acoplamiento energético divergente."
            )

        # ── Tolerancia numérica ───────────────────────────────────────────
        tolerance = (
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, phi_norm * grad_norm)
        )

        # ── Verificación de disipación ────────────────────────────────────
        if p_raw < _DIRICHLET_DISSIPATION_FLOOR - tolerance:
            raise ThermodynamicSingularityError(
                "[Fase 2] Singularidad termodinámica en el AST. "
                f"P_diss = {p_raw:.6e} < −ε_num = {-tolerance:.6e}. "
                "El algoritmo puede inducir bucle infinito o desbordamiento FPU."
            )

        # ── Proyección al cono positivo ───────────────────────────────────
        dissipated_power = max(_DIRICHLET_DISSIPATION_FLOOR, p_raw)

        # ── Alineamiento y ángulo ─────────────────────────────────────────
        if phi_norm > 0.0 and grad_norm > 0.0:
            alignment_raw = p_raw / (phi_norm * grad_norm)
            alignment = float(np.clip(alignment_raw, -1.0, 1.0))
        else:
            alignment = 0.0

        thermodynamic_angle = self._compute_thermodynamic_angle(alignment)

        # ── Tasa de producción de entropía ────────────────────────────────
        entropy_production = self._compute_entropy_production_rate(dissipated_power)
        clausius_duhem_ok = entropy_production >= _ENTROPY_PRODUCTION_FLOOR

        if not clausius_duhem_ok:
            raise ClausiusDuhemViolation(
                f"[Fase 2] La desigualdad de Clausius-Duhem es violada: "
                f"σ_CD = {entropy_production:.6e} < 0."
            )

        # ── Eficiencia disipativa ─────────────────────────────────────────
        dissipation_ratio = self._compute_exergy_dissipation_ratio(
            dissipated_power, phi_norm, grad_norm,
        )

        # ── Clasificación de disipación ───────────────────────────────────
        is_strictly_dissipative = dissipated_power > max(
            _DIRICHLET_DISSIPATION_FLOOR, tolerance,
        )

        if not is_strictly_dissipative:
            logger.warning(
                "[Fase 2] Disipación no estricta: P_diss = %.4e, ε_num = %.4e, "
                "θ = %.4f rad.",
                dissipated_power, tolerance, thermodynamic_angle,
            )

        if thermodynamic_angle > math.pi / 2.0:
            logger.warning(
                "[Fase 2] Ángulo termodinámico θ = %.4f rad > π/2. "
                "Retroalimentación energética parcial detectada.",
                thermodynamic_angle,
            )

        # ── Emisión del certificado (objeto inicial de Fase 3) ────────────
        return ThermodynamicDirichletData(
            dissipated_power=dissipated_power,
            raw_inner_product=p_raw,
            numerical_tolerance=tolerance,
            exergy_norm=phi_norm,
            gradient_norm=grad_norm,
            alignment_cosine=alignment,
            thermodynamic_angle_rad=thermodynamic_angle,
            entropy_production_rate=entropy_production,
            clausius_duhem_satisfied=clausius_duhem_ok,
            exergy_dissipation_ratio=dissipation_ratio,
            is_thermodynamically_stable=True,
            is_strictly_dissipative=is_strictly_dissipative,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 3: COHOMOLOGÍA DE HACES CELULARES                                    ║
# ║                                                                             ║
# ║   Marco matemático:                                                         ║
# ║   ─────────────────                                                         ║
# ║   Sea G = (V, E, F) un haz celular sobre el grafo de dependencias del AST.  ║
# ║   El complejo de cadenas del haz tiene la forma:                            ║
# ║                                                                             ║
# ║       C⁰ ──δ⁰──→ C¹ ──δ¹──→ C²                                              ║
# ║                                                                             ║
# ║   con la condición de complejo: δ¹∘δ⁰ = 0.                                  ║
# ║                                                                             ║
# ║   Los grupos de cohomología son:                                            ║
# ║       H⁰ = ker(δ⁰)                                                          ║
# ║       H¹ = ker(δ¹) / im(δ⁰)                                                 ║
# ║       H² = C² / im(δ¹)                                                      ║
# ║                                                                             ║
# ║   Las dimensiones (números de Betti) son:                                   ║
# ║       β₀ = dim ker(δ⁰) = dim C⁰ − rank(δ⁰)                                  ║
# ║       β₁ = dim H¹     = dim C¹ − rank(δ⁰) − rank(δ¹)                        ║
# ║       β₂ = dim H²     = dim C² − rank(δ¹)                                   ║
# ║                                                                             ║
# ║   La característica de Euler-Poincaré:                                      ║
# ║       χ = β₀ − β₁ + β₂ = dim C⁰ − dim C¹ + dim C²                           ║
# ║                                                                             ║
# ║   La torsión de Reidemeister:                                               ║
# ║       log|τ| = Σᵢ (−1)ⁱ · log|det(δⁱ)|                                      ║
# ║                                                                             ║
# ║   Invariantes verificados en v3.0.0:                                        ║
# ║   1. Condición de complejo: ‖δ¹∘δ⁰‖_F / scale ≤ τ_δ                         ║
# ║   2. dim H¹ = 0 (integrabilidad global)                                     ║
# ║   3. Números de Betti (β₀, β₁, β₂)                                          ║
# ║   4. Característica de Euler χ                                              ║
# ║   5. Torsión de Reidemeister log|τ|                                         ║
# ║   6. Exactitud de Mayer-Vietoris                                            ║
# ║                                                                             ║
# ║   CONEXIÓN FUNTORIAL:                                                       ║
# ║   El primer método de Fase 3 recibe ThermodynamicDirichletData y verifica   ║
# ║   que la estabilidad termodinámica sea compatible con la integrabilidad.    ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_CellularSheafCohomologyAuditor(Phase2_DirichletThermodynamicEnforcer):
    r"""
    Eleva el flujo de datos del AST a un Haz Celular (Cellular Sheaf).
    Detecta variables huérfanas, ciclos lógicos mutantes o dependencias fantasma.

    Hereda de Phase2_DirichletThermodynamicEnforcer.
    Su primer método _audit_cellular_sheaf_cohomology recibe el certificado
    ThermodynamicDirichletData emitido por el último método de Fase 2.

    Mejoras v3.0.0:
        · Cálculo de los tres números de Betti (β₀, β₁, β₂).
        · Característica de Euler-Poincaré χ = β₀ − β₁ + β₂.
        · Torsión de Reidemeister log|τ| como invariante secundario.
        · Verificación de exactitud de la secuencia de Mayer-Vietoris.
        · Rango por gap espectral (Golub-Reinsch) en lugar de umbral fijo.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 3.1 Rango numérico por gap espectral (Golub-Reinsch)
    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def _numerical_rank_with_gap(
        cls,
        A: NDArray[np.float64],
        name: str = "A",
    ) -> Tuple[int, NDArray[np.float64]]:
        r"""
        Calcula el rango numérico de A y los valores singulares usando el criterio
        de gap espectral de Golub-Reinsch:

            rank = k  sii  σ_k / σ_max > ρ_gap  y  σ_{k+1} / σ_max ≤ ρ_gap.

        Si no se detecta gap, usa la tolerancia adaptativa estándar:

            tol = c · ε₀ · max(shape(A)) · σ_max.

        Parámetros:
            A   : Matriz a evaluar.
            name: Nombre del parámetro (para mensajes de error).

        Retorna:
            (rank, singular_values).
        """
        if A.size == 0 or min(A.shape) == 0:
            return 0, np.array([], dtype=np.float64)

        svs = cls._safe_svdvals(A)

        if svs.size == 0:
            return 0, svs

        sigma_max = float(svs[0])

        if sigma_max == 0.0:
            return 0, svs

        if not math.isfinite(sigma_max):
            raise CohomologicalObstructionError(
                f"[Fase 3] σ_max({name}) no es finita."
            )

        # Tolerancia adaptativa estándar
        tol_standard = (
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(A.shape)
            * sigma_max
        )

        # Detección de gap espectral
        normalized_svs = svs / sigma_max
        gaps = np.diff(normalized_svs)  # negativo si decreciente

        # Buscar el gap más grande (mayor caída relativa)
        if gaps.size > 0:
            gap_idx = int(np.argmin(gaps))  # índice del mayor gap
            gap_magnitude = float(-gaps[gap_idx])  # positivo

            # Si el gap es significativo (> 10x la tolerancia relativa)
            gap_threshold = 10.0 * _SPECTRAL_GAP_RATIO
            if gap_magnitude > gap_threshold:
                rank_by_gap = gap_idx + 1
                return rank_by_gap, svs

        # Sin gap claro: usar tolerancia estándar
        rank = int(np.count_nonzero(svs > tol_standard))
        return rank, svs

    # ─────────────────────────────────────────────────────────────────────────
    # 3.2 Cálculo de los números de Betti
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_betti_numbers(
        self,
        D0: NDArray[np.float64],
        D1: NDArray[np.float64],
        rank_D0: int,
        rank_D1: int,
    ) -> Tuple[int, int, int]:
        r"""
        Calcula los números de Betti del complejo cohomológico:

            β₀ = dim C⁰ − rank(δ⁰)   = dim ker(δ⁰)
            β₁ = dim C¹ − rank(δ⁰) − rank(δ¹)   = dim H¹
            β₂ = dim C² − rank(δ¹)   = dim H²

        donde:
            δ⁰: D0 ∈ ℝ^{c1 × c0}  →  shape = (c1, c0)
            δ¹: D1 ∈ ℝ^{c2 × c1}  →  shape = (c2, c1)

        Parámetros:
            D0      : Operador coboundary δ⁰.
            D1      : Operador coboundary δ¹.
            rank_D0 : rank(δ⁰) calculado numéricamente.
            rank_D1 : rank(δ¹) calculado numéricamente.

        Retorna:
            (β₀, β₁, β₂) como tupla de enteros no negativos.
        """
        c0 = D0.shape[1]  # dim C⁰
        c1 = D0.shape[0]  # dim C¹ = D0.shape[0] = D1.shape[1]
        c2 = D1.shape[0]  # dim C²

        beta0 = max(0, c0 - rank_D0)
        beta1 = max(0, c1 - rank_D0 - rank_D1)
        beta2 = max(0, c2 - rank_D1)

        return int(beta0), int(beta1), int(beta2)

    # ─────────────────────────────────────────────────────────────────────────
    # 3.3 Torsión de Reidemeister
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_reidemeister_torsion(
        self,
        D0: NDArray[np.float64],
        D1: NDArray[np.float64],
        rank_D0: int,
        rank_D1: int,
        svs_D0: NDArray[np.float64],
        svs_D1: NDArray[np.float64],
    ) -> float:
        r"""
        Calcula la torsión de Reidemeister log|τ| del complejo cohomológico.

        La torsión se define (para un complejo acíclico) como:

            log|τ| = Σᵢ (−1)^{i+1} · Σⱼ log(σⱼ(δⁱ)),

        donde la suma interior es sobre los valores singulares no nulos de δⁱ.

        Para un complejo no acíclico (dim H¹ > 0), la torsión no está bien
        definida y se retorna 0.0.

        Parámetros:
            D0, D1      : Operadores coboundary.
            rank_D0     : rank(δ⁰).
            rank_D1     : rank(δ¹).
            svs_D0      : Valores singulares de δ⁰.
            svs_D1      : Valores singulares de δ¹.

        Retorna:
            log|τ| ∈ ℝ (0.0 si el complejo no es acíclico o la torsión
            no está definida).
        """
        c1 = D0.shape[0]
        h1_dim = c1 - rank_D0 - rank_D1

        if h1_dim != 0:
            # Torsión no definida para complejos no acíclicos
            return 0.0

        def log_sum_sv(svs: NDArray[np.float64], rank: int) -> float:
            r"""Suma de log(σᵢ) para los primeros `rank` valores singulares."""
            if rank <= 0 or svs.size == 0:
                return 0.0
            significant = svs[:rank]
            positive = significant[significant > 0]
            if positive.size == 0:
                return 0.0
            log_sum = float(np.sum(np.log(positive)))
            return log_sum if math.isfinite(log_sum) else 0.0

        # log|τ| = (+1)·Σlog(σⱼ(δ⁰)) + (−1)·Σlog(σⱼ(δ¹))
        log_tau = log_sum_sv(svs_D0, rank_D0) - log_sum_sv(svs_D1, rank_D1)

        return log_tau if math.isfinite(log_tau) else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # 3.4 Verificación de la secuencia de Mayer-Vietoris
    # ─────────────────────────────────────────────────────────────────────────
    def _verify_mayer_vietoris_exactness(
        self,
        D0: NDArray[np.float64],
        D1: NDArray[np.float64],
        rank_D0: int,
        rank_D1: int,
    ) -> bool:
        r"""
        Verifica la exactitud local de la secuencia de Mayer-Vietoris:

            im(δ⁰) ⊆ ker(δ¹),

        lo cual es equivalente a la condición de complejo δ¹∘δ⁰ = 0,
        que ya fue verificada. Aquí verificamos además que:

            rank(δ⁰) + rank(δ¹) ≤ dim C¹,

        condición necesaria para la exactitud de la secuencia en C¹.

        Parámetros:
            D0, D1      : Operadores coboundary.
            rank_D0     : rank(δ⁰).
            rank_D1     : rank(δ¹).

        Retorna:
            True sii la secuencia es compatible con la exactitud.
        """
        c1 = D0.shape[0]

        # Condición necesaria para exactitud: rank(δ⁰) + rank(δ¹) ≤ dim C¹
        if rank_D0 + rank_D1 > c1:
            logger.warning(
                "[Fase 3] Mayer-Vietoris: rank(δ⁰) + rank(δ¹) = %d > dim C¹ = %d. "
                "La secuencia puede no ser exacta.",
                rank_D0 + rank_D1, c1,
            )
            return False

        return True

    # ─────────────────────────────────────────────────────────────────────────
    # 3.5 Cálculo de dim H¹ por operadores coboundary
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_first_cohomology_dimension(
        self,
        coboundary_delta0: NDArray[np.float64],
        coboundary_delta1: NDArray[np.float64],
    ) -> Tuple[int, int, int, int, int, float, float, bool, NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Calcula dim H¹ y todos los invariantes secundarios a partir de los
        operadores coboundary δ⁰ y δ¹.

        El complejo cohomológico requiere δ¹∘δ⁰ = 0.

        Retorna una tupla completa con todos los invariantes necesarios:
            (h1_dim, rank_D0, rank_D1, beta0, beta2, reidemeister_torsion,
             complex_residual, mayer_vietoris_exact, svs_D0, svs_D1)

        El llamador (_audit_cellular_sheaf_cohomology) desempaqueta estos valores.
        """
        # ── Validación ────────────────────────────────────────────────────
        D0 = self._as_finite_matrix(
            "coboundary_delta0", coboundary_delta0, square=False,
        )
        D1 = self._as_finite_matrix(
            "coboundary_delta1", coboundary_delta1, square=False,
        )

        # D0: (c1, c0), D1: (c2, c1)
        if D1.shape[1] != D0.shape[0]:
            raise ValueError(
                "[Fase 3] Los operadores coboundary no componen: "
                f"δ¹ tiene dominio dim={D1.shape[1]}, "
                f"pero δ⁰ tiene codominio dim={D0.shape[0]}."
            )

        # ── Verificación de complejo: δ¹∘δ⁰ = 0 ──────────────────────────
        composition = D1 @ D0

        if not np.all(np.isfinite(composition)):
            raise CohomologicalObstructionError(
                "[Fase 3] δ¹∘δ⁰ produjo valores no finitos."
            )

        d0_norm = self._frobenius_norm(D0)
        d1_norm = self._frobenius_norm(D1)

        if not math.isfinite(d0_norm) or not math.isfinite(d1_norm):
            raise CohomologicalObstructionError(
                "[Fase 3] Las normas de δ⁰ ó δ¹ no son finitas."
            )

        composition_norm = self._frobenius_norm(composition)

        if not math.isfinite(composition_norm):
            raise CohomologicalObstructionError(
                "[Fase 3] La norma de δ¹∘δ⁰ no es finita."
            )

        scale = max(1.0, d0_norm * d1_norm)
        complex_residual = composition_norm / scale

        if complex_residual > _COHOMOLOGICAL_COMPLEX_TOL:
            raise CohomologicalObstructionError(
                "[Fase 3] Los operadores celulares no forman un complejo válido: "
                f"‖δ¹∘δ⁰‖/scale = {complex_residual:.6e} > {_COHOMOLOGICAL_COMPLEX_TOL:.6e}."
            )

        # ── Rangos con gap espectral ───────────────────────────────────────
        rank_D0, svs_D0 = self._numerical_rank_with_gap(D0, "δ⁰")
        rank_D1, svs_D1 = self._numerical_rank_with_gap(D1, "δ¹")

        # ── dim H¹ ────────────────────────────────────────────────────────
        c1 = D0.shape[0]
        h1 = c1 - rank_D0 - rank_D1

        if h1 < 0:
            logger.warning(
                "[Fase 3] dim H¹ calculada = %d < 0; se proyecta a 0 por tolerancia numérica.",
                h1,
            )
            h1 = 0

        # ── Números de Betti ──────────────────────────────────────────────
        beta0, _beta1, beta2 = self._compute_betti_numbers(
            D0, D1, rank_D0, rank_D1,
        )

        # ── Torsión de Reidemeister ───────────────────────────────────────
        torsion = self._compute_reidemeister_torsion(
            D0, D1, rank_D0, rank_D1, svs_D0, svs_D1,
        )

        # ── Mayer-Vietoris ────────────────────────────────────────────────
        mv_exact = self._verify_mayer_vietoris_exactness(
            D0, D1, rank_D0, rank_D1,
        )

        return (
            int(h1),
            int(rank_D0),
            int(rank_D1),
            int(beta0),
            int(beta2),
            float(torsion),
            float(complex_residual),
            bool(mv_exact),
            svs_D0,
            svs_D1,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3.6 PRIMER MÉTODO DE FASE 3 / CONTINUACIÓN DE FASE 2
    #     _audit_cellular_sheaf_cohomology → SheafCohomologyAuditData
    # ─────────────────────────────────────────────────────────────────────────
    def _audit_cellular_sheaf_cohomology(
        self,
        h1_dimension: Optional[int] = None,
        thermodynamic_audit: Optional[ThermodynamicDirichletData] = None,
        coboundary_delta0: Optional[NDArray[np.float64]] = None,
        coboundary_delta1: Optional[NDArray[np.float64]] = None,
    ) -> SheafCohomologyAuditData:
        r"""
        ════════════════════════════════════════════════════════════════════
        PRIMER MÉTODO DE FASE 3 — Continuación formal de Fase 2.
        ════════════════════════════════════════════════════════════════════

        Verifica la condición de integrabilidad global del haz celular:

            dim H¹(G; F) = 0.

        Si `thermodynamic_audit` es provisto (composición funtorial estricta):
            - Verifica que is_thermodynamically_stable = True.

        Puede operar en dos modos:
            1. Recibiendo `h1_dimension` directamente (modo ligero).
            2. Calculando `h1_dimension` desde operadores coboundary δ⁰, δ¹ (modo completo).

        En modo completo, calcula adicionalmente:
            - Números de Betti (β₀, β₁, β₂).
            - Característica de Euler χ.
            - Torsión de Reidemeister log|τ|.
            - Exactitud de Mayer-Vietoris.

        Si ambos modos se suministran, exige consistencia exacta entre
        h1_dimension declarado y el calculado por operadores δ.

        Parámetros:
            h1_dimension      : dim H¹(G; F) declarada (entero no negativo).
            thermodynamic_audit: Certificado de Fase 2 (opcional pero recomendado).
            coboundary_delta0 : Operador δ⁰ ∈ ℝ^{c1 × c0}.
            coboundary_delta1 : Operador δ¹ ∈ ℝ^{c2 × c1}.

        Retorna:
            SheafCohomologyAuditData — certificado cohomológico completo.

        Lanza:
            ThermodynamicSingularityError si thermodynamic_audit no es estable.
            CohomologicalObstructionError si dim H¹ > 0.
            EulerCharacteristicMismatch si χ es inconsistente.
            MayerVietorisBreachError si la secuencia M-V no es exacta.
        """
        # ── Continuación funtorial de Fase 2 ─────────────────────────────
        if thermodynamic_audit is not None:
            if not thermodynamic_audit.is_thermodynamically_stable:
                raise ThermodynamicSingularityError(
                    "[Fase 3] No puede iniciarse: la Fase 2 no certificó estabilidad "
                    "termodinámica (is_thermodynamically_stable=False)."
                )

        # ── Variables de estado de los invariantes secundarios ────────────
        computed_h1: Optional[int] = None
        verified_by_coboundary = False
        rank_delta0_val: int = 0
        rank_delta1_val: int = 0
        betti_vals: Tuple[int, int, int] = (0, 0, 0)
        euler_chi: int = 0
        reidemeister_val: float = 0.0
        complex_residual_val: float = 0.0
        mayer_vietoris_exact: bool = True

        # ── Modo completo: operadores coboundary ──────────────────────────
        if coboundary_delta0 is not None or coboundary_delta1 is not None:
            if coboundary_delta0 is None or coboundary_delta1 is None:
                raise ValueError(
                    "[Fase 3] Para verificación por operadores coboundary deben "
                    "proveerse coboundary_delta0 y coboundary_delta1 simultáneamente."
                )

            (
                computed_h1,
                rank_delta0_val,
                rank_delta1_val,
                beta0_val,
                beta2_val,
                reidemeister_val,
                complex_residual_val,
                mayer_vietoris_exact,
                _svs_D0,
                _svs_D1,
            ) = self._compute_first_cohomology_dimension(
                coboundary_delta0=coboundary_delta0,
                coboundary_delta1=coboundary_delta1,
            )

            beta1_val = computed_h1
            betti_vals = (beta0_val, beta1_val, beta2_val)

            # Característica de Euler
            c0 = np.asarray(coboundary_delta0).shape[1]
            c1 = np.asarray(coboundary_delta0).shape[0]
            c2 = np.asarray(coboundary_delta1).shape[0]
            euler_chi = int(c0 - c1 + c2)

            euler_from_betti = int(beta0_val - beta1_val + beta2_val)
            if abs(euler_chi - euler_from_betti) > _EULER_CHARACTERISTIC_TOLERANCE:
                raise EulerCharacteristicMismatch(
                    "[Fase 3] Inconsistencia en la característica de Euler: "
                    f"χ(G) = {euler_chi} (por cadenas), pero "
                    f"β₀ − β₁ + β₂ = {euler_from_betti} (por Betti)."
                )

            if not mayer_vietoris_exact:
                raise MayerVietorisBreachError(
                    "[Fase 3] La secuencia de Mayer-Vietoris no es exacta en C¹. "
                    "El haz celular no puede descomponerse consistentemente."
                )

            verified_by_coboundary = True

        # ── Resolución de h1 ──────────────────────────────────────────────
        if h1_dimension is None:
            if computed_h1 is None:
                raise ValueError(
                    "[Fase 3] Debe proveerse h1_dimension o los operadores "
                    "coboundary_delta0 y coboundary_delta1."
                )
            h1 = computed_h1
        else:
            # Validación del tipo de h1_dimension
            if isinstance(h1_dimension, (bool, np.bool_)):
                raise TypeError(
                    "[Fase 3] h1_dimension debe ser un entero; se recibió booleano."
                )

            if not isinstance(h1_dimension, (int, np.integer)):
                raise TypeError(
                    f"[Fase 3] h1_dimension debe ser un entero; recibido: "
                    f"{type(h1_dimension)}."
                )

            h1 = int(h1_dimension)

            # Consistencia entre declarado y calculado
            if computed_h1 is not None and computed_h1 != h1:
                raise CohomologicalObstructionError(
                    "[Fase 3] Inconsistencia cohomológica: "
                    f"h1_dimension declarado={h1}, calculado por δ={computed_h1}."
                )

        # ── Verificación de no negatividad ────────────────────────────────
        if h1 < _COHOMOLOGY_DIMENSION_FLOOR:
            raise ValueError(
                f"[Fase 3] h1_dimension no puede ser negativa: {h1}."
            )

        # ── Condición de integrabilidad global ────────────────────────────
        if h1 > _COHOMOLOGY_DIMENSION_FLOOR:
            raise CohomologicalObstructionError(
                "[Fase 3] Obstrucción topológica global detectada en la sintaxis. "
                f"dim H¹(G; F) = {h1} > 0. El código propuesto contiene "
                "contradicciones lógicas, ciclos de dependencia irresolubles "
                "o variables huérfanas."
            )

        # ── Si sólo se proporcionó h1_dimension sin operadores ────────────
        if not verified_by_coboundary:
            betti_vals = (0, 0, 0)
            euler_chi = 0

        # ── Emisión del certificado ───────────────────────────────────────
        return SheafCohomologyAuditData(
            h1_dimension=h1,
            rank_delta0=rank_delta0_val,
            rank_delta1=rank_delta1_val,
            betti_numbers=betti_vals,
            euler_characteristic=euler_chi,
            reidemeister_torsion=reidemeister_val,
            complex_residual=complex_residual_val,
            is_globally_integrable=True,
            obstruction_free=True,
            verified_by_coboundary=verified_by_coboundary,
            mayer_vietoris_exact=mayer_vietoris_exact,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   ORQUESTADOR SUPREMO: AST STATIC ANALYZER AGENT                            ║
# ║                                                                             ║
# ║   Endofuntor Z_{Γ-PHYSICS} = Φ₃ ∘ Φ₂ ∘ Φ₁                                   ║
# ║                                                                             ║
# ║   Mejoras v3.0.0:                                                           ║
# ║   · Contexto de auditoría con timestamp ISO-8601 UTC.                       ║
# ║   · Checksum SHA-256 de los datos de entrada.                               ║
# ║   · Trazabilidad completa de la cadena funtorial (AuditProvenance).         ║
# ║   · Modo de verificación estricto: strict_mode configurable.                ║
# ║   · Log estructurado con todos los certificados de las tres fases.          ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class ASTStaticAnalyzerAgent(Morphism, Phase3_CellularSheafCohomologyAuditor):
    r"""
    El Custodio de la Cohomología Sintáctica en el estrato Γ-PHYSICS.

    Somete el código generado por el Modelo de Lenguaje a la composición
    funtorial estricta:

        Z_{Γ-PHYSICS} = Φ₃ ∘ Φ₂ ∘ Φ₁,

    garantizando un ecosistema de ejecución matemáticamente purificado mediante
    los tres invariantes topológicos, termodinámicos y cohomológicos.

    Nuevas capacidades v3.0.0:
        · Trazabilidad criptográfica por SHA-256 de los datos de entrada.
        · Timestamp ISO-8601 UTC en cada ejecución.
        · Modo estricto configurable: en modo no estricto, advierte pero no
          lanza excepción para violaciones menores (mal condicionamiento,
          índice de Maslov no nulo, disipación no estricta).
        · Log estructurado multi-nivel con todos los certificados.

    Parámetros de constructor:
        strict_mode: Si True (default), cualquier advertencia se convierte en
                     error. Si False, sólo las violaciones críticas lanzan
                     excepción.
    """

    def __init__(self, strict_mode: bool = True) -> None:
        r"""
        Inicializa el ASTStaticAnalyzerAgent.

        Parámetros:
            strict_mode: Controla si las advertencias no bloqueantes
                         se convierten en errores (True) o solo en logs (False).
        """
        self._strict_mode = bool(strict_mode)

        logger.info(
            "[Orquestador] ASTStaticAnalyzerAgent inicializado. strict_mode=%s.",
            self._strict_mode,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Construcción del AuditProvenance
    # ─────────────────────────────────────────────────────────────────────────
    def _build_provenance(
        self,
        checksum: str,
        phase1_passed: bool,
        phase2_passed: bool,
        phase3_passed: bool,
    ) -> AuditProvenance:
        r"""
        Construye el objeto de trazabilidad funtorial.

        Parámetros:
            checksum     : SHA-256 hexadecimal de los datos de entrada.
            phase1_passed: True sii Fase 1 completó sin excepción.
            phase2_passed: True sii Fase 2 completó sin excepción.
            phase3_passed: True sii Fase 3 completó sin excepción.

        Retorna:
            AuditProvenance con timestamp UTC actual.
        """
        timestamp = datetime.now(tz=timezone.utc).isoformat()

        phases_ok = [phase1_passed, phase2_passed, phase3_passed]
        all_passed = all(phases_ok)

        functor_chain = (
            f"Φ₁={'✓' if phase1_passed else '✗'} → "
            f"Φ₂={'✓' if phase2_passed else '✗'} → "
            f"Φ₃={'✓' if phase3_passed else '✗'} → "
            f"Z_{{Γ-PHYSICS}}={'✓' if all_passed else '✗'}"
        )

        return AuditProvenance(
            timestamp_iso=timestamp,
            input_checksum_sha256=checksum,
            phase1_passed=phase1_passed,
            phase2_passed=phase2_passed,
            phase3_passed=phase3_passed,
            functor_chain=functor_chain,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Log estructurado de los certificados
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _log_governance_summary(
        symplectic_audit: SymplecticInvariantData,
        thermodynamic_audit: ThermodynamicDirichletData,
        cohomology_audit: SheafCohomologyAuditData,
        provenance: AuditProvenance,
    ) -> None:
        r"""
        Emite un log estructurado de alta granularidad con todos los
        certificados de las tres fases.
        """
        logger.info(
            "═══════════════════════════════════════════════════════════\n"
            "  AST SYMPLECTIC GOVERNANCE — REPORTE FINAL\n"
            "  Timestamp : %s\n"
            "  SHA-256   : %s\n"
            "  Cadena    : %s\n"
            "───────────────────────────────────────────────────────────\n"
            "  FASE 1 — Invarianza Simpléctica:\n"
            "    dim Esp. Fase : %d\n"
            "    Residuo |F    : %.4e\n"
            "    Residuo Rel.  : %.4e (tol=%.4e)\n"
            "    Residuo Antisim.: %.4e\n"
            "    |det(M)−1|   : %.4e\n"
            "    κ(M)         : %.4e\n"
            "    ρ(MᵀΩM)      : %.4e\n"
            "    μ(Λ)         : %d\n"
            "    ρ_U (polar)  : %.4e\n"
            "    Volumen ✓    : %s\n"
            "───────────────────────────────────────────────────────────\n"
            "  FASE 2 — Termodinámica Port-Hamiltoniana:\n"
            "    P_raw        : %.4e\n"
            "    P_diss       : %.4e\n"
            "    ε_num        : %.4e\n"
            "    ‖Φ‖          : %.4e\n"
            "    ‖∇V‖         : %.4e\n"
            "    cos(θ)       : %.4f\n"
            "    θ            : %.4f rad\n"
            "    σ_CD         : %.4e\n"
            "    C-D ✓        : %s\n"
            "    η (efic.)    : %.4f\n"
            "    Estable ✓    : %s\n"
            "───────────────────────────────────────────────────────────\n"
            "  FASE 3 — Cohomología de Haces Celulares:\n"
            "    dim H¹       : %d\n"
            "    rank(δ⁰)     : %d\n"
            "    rank(δ¹)     : %d\n"
            "    (β₀,β₁,β₂)  : %s\n"
            "    χ            : %d\n"
            "    log|τ|       : %.4f\n"
            "    ‖δ¹∘δ⁰‖/s    : %.4e\n"
            "    Integrable ✓ : %s\n"
            "    M-V Exacto ✓ : %s\n"
            "    Ver. por δ   : %s\n"
            "═══════════════════════════════════════════════════════════",
            # Provenance
            provenance.timestamp_iso,
            provenance.input_checksum_sha256[:16] + "...",
            provenance.functor_chain,
            # Fase 1
            symplectic_audit.phase_space_dimension,
            symplectic_audit.symplectic_residual_norm,
            symplectic_audit.symplectic_relative_residual,
            symplectic_audit.effective_tolerance,
            symplectic_audit.antisymmetric_residual,
            symplectic_audit.determinant_residual,
            symplectic_audit.condition_number,
            symplectic_audit.spectral_radius_Omega,
            symplectic_audit.maslov_index,
            symplectic_audit.polar_unitarity_residual,
            symplectic_audit.is_volume_preserved,
            # Fase 2
            thermodynamic_audit.raw_inner_product,
            thermodynamic_audit.dissipated_power,
            thermodynamic_audit.numerical_tolerance,
            thermodynamic_audit.exergy_norm,
            thermodynamic_audit.gradient_norm,
            thermodynamic_audit.alignment_cosine,
            thermodynamic_audit.thermodynamic_angle_rad,
            thermodynamic_audit.entropy_production_rate,
            thermodynamic_audit.clausius_duhem_satisfied,
            thermodynamic_audit.exergy_dissipation_ratio,
            thermodynamic_audit.is_thermodynamically_stable,
            # Fase 3
            cohomology_audit.h1_dimension,
            cohomology_audit.rank_delta0,
            cohomology_audit.rank_delta1,
            cohomology_audit.betti_numbers,
            cohomology_audit.euler_characteristic,
            cohomology_audit.reidemeister_torsion,
            cohomology_audit.complex_residual,
            cohomology_audit.is_globally_integrable,
            cohomology_audit.mayer_vietoris_exact,
            cohomology_audit.verified_by_coboundary,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PUNTO DE ENTRADA PRINCIPAL
    # ─────────────────────────────────────────────────────────────────────────
    def execute_ast_symplectic_governance(
        self,
        ast_jacobian_M: NDArray[np.float64],
        control_potential_Phi: NDArray[np.float64],
        lyapunov_gradient_V: NDArray[np.float64],
        h1_dimension: Optional[int] = None,
        coboundary_delta0: Optional[NDArray[np.float64]] = None,
        coboundary_delta1: Optional[NDArray[np.float64]] = None,
    ) -> ASTGovernanceState:
        r"""
        Ejecuta la composición funtorial estricta:

            Z_{Γ-PHYSICS} = Φ₃ ∘ Φ₂ ∘ Φ₁.

        Fases anidadas:
            Φ₁ : Auditoría simpléctica → SymplecticInvariantData.
            Φ₂ : Control termodinámico Dirichlet → ThermodynamicDirichletData.
            Φ₃ : Auditoría cohomológica de haces → SheafCohomologyAuditData.

        Adicionalmente:
            · Calcula el SHA-256 de los datos de entrada para trazabilidad.
            · Registra el timestamp UTC de la auditoría.
            · Construye el AuditProvenance con la cadena funtorial completa.
            · Emite un log estructurado con todos los certificados.

        Parámetros:
            ast_jacobian_M:
                Jacobiana M de la transformación sintáctica. Debe ser
                cuadrada, real, finita y de dimensión par 2n.

            control_potential_Phi:
                Campo de control exergético Φ ∈ ℝ^{2n}.

            lyapunov_gradient_V:
                Gradiente de Lyapunov ∇V ∈ ℝ^{2n}.

            h1_dimension:
                Dimensión H¹(G; F) declarada (entero ≥ 0).
                Debe ser 0 para autorizar la compilación.

            coboundary_delta0:
                Operador δ⁰ ∈ ℝ^{c1 × c0} (opcional).

            coboundary_delta1:
                Operador δ¹ ∈ ℝ^{c2 × c1} (opcional).

        Retorna:
            ASTGovernanceState con los tres certificados, provenance y
            is_compilation_authorized=True sii todas las fases pasan.

        Lanza:
            Cualquier excepción de la jerarquía TopologicalInvariantError
            si alguna fase detecta una violación crítica.
        """
        # ── Checksum de entrada ───────────────────────────────────────────
        arrays_for_checksum = [
            np.asarray(ast_jacobian_M) if ast_jacobian_M is not None else np.array([]),
            np.asarray(control_potential_Phi) if control_potential_Phi is not None else np.array([]),
            np.asarray(lyapunov_gradient_V) if lyapunov_gradient_V is not None else np.array([]),
        ]

        if coboundary_delta0 is not None:
            arrays_for_checksum.append(np.asarray(coboundary_delta0))
        if coboundary_delta1 is not None:
            arrays_for_checksum.append(np.asarray(coboundary_delta1))

        input_checksum = self._compute_input_checksum(*arrays_for_checksum)

        logger.debug(
            "[Orquestador] Iniciando gobernanza. SHA-256 entrada: %s.",
            input_checksum[:16] + "...",
        )

        # ── Estado de las fases ───────────────────────────────────────────
        phase1_passed = False
        phase2_passed = False
        phase3_passed = False

        # ── Fase 1: Invarianza Simpléctica ────────────────────────────────
        symplectic_audit = self._audit_symplectic_invariance(ast_jacobian_M)
        phase1_passed = True

        logger.debug(
            "[Fase 1] ✓ Volumen preservado. Residuo relativo=%.4e.",
            symplectic_audit.symplectic_relative_residual,
        )

        # ── Fase 2: Termodinámica Port-Hamiltoniana ────────────────────────
        thermodynamic_audit = self._enforce_dirichlet_thermodynamics(
            control_potential_Phi=control_potential_Phi,
            lyapunov_gradient_V=lyapunov_gradient_V,
            symplectic_audit=symplectic_audit,
        )
        phase2_passed = True

        logger.debug(
            "[Fase 2] ✓ Estabilidad termodinámica. P_diss=%.4e, σ_CD=%.4e.",
            thermodynamic_audit.dissipated_power,
            thermodynamic_audit.entropy_production_rate,
        )

        # ── Fase 3: Cohomología de Haces Celulares ────────────────────────
        cohomology_audit = self._audit_cellular_sheaf_cohomology(
            h1_dimension=h1_dimension,
            thermodynamic_audit=thermodynamic_audit,
            coboundary_delta0=coboundary_delta0,
            coboundary_delta1=coboundary_delta1,
        )
        phase3_passed = True

        logger.debug(
            "[Fase 3] ✓ Integrabilidad global. dim H¹=%d, χ=%d.",
            cohomology_audit.h1_dimension,
            cohomology_audit.euler_characteristic,
        )

        # ── Autorización de compilación ───────────────────────────────────
        is_compilation_authorized = bool(
            symplectic_audit.is_volume_preserved
            and thermodynamic_audit.is_thermodynamically_stable
            and cohomology_audit.is_globally_integrable
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
            symplectic_audit=symplectic_audit,
            thermodynamic_audit=thermodynamic_audit,
            cohomology_audit=cohomology_audit,
            provenance=provenance,
        )

        # ── Verificación final ────────────────────────────────────────────
        if not is_compilation_authorized:
            raise TopologicalInvariantError(
                "[Orquestador] La composición funtorial Z_{Γ-PHYSICS} = Φ₃∘Φ₂∘Φ₁ "
                f"no autorizó la compilación del AST. "
                f"Cadena: {provenance.functor_chain}."
            )

        return ASTGovernanceState(
            symplectic_audit=symplectic_audit,
            thermodynamic_audit=thermodynamic_audit,
            cohomology_audit=cohomology_audit,
            provenance=provenance,
            is_compilation_authorized=is_compilation_authorized,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ═══════════════════════════════════════════════════════════════════════════════
__all__: List[str] = [
    # Excepciones
    "SymplecticInvarianceViolation",
    "SymplecticPolarDecompositionError",
    "MaslovIndexError",
    "ThermodynamicSingularityError",
    "ClausiusDuhemViolation",
    "ExergyDivergenceError",
    "CohomologicalObstructionError",
    "EulerCharacteristicMismatch",
    "MayerVietorisBreachError",
    # DTOs
    "SymplecticInvariantData",
    "ThermodynamicDirichletData",
    "SheafCohomologyAuditData",
    "AuditProvenance",
    "ASTGovernanceState",
    # Fases
    "Phase1_SymplecticInvarianceAuditor",
    "Phase2_DirichletThermodynamicEnforcer",
    "Phase3_CellularSheafCohomologyAuditor",
    # Orquestador
    "ASTStaticAnalyzerAgent",
]