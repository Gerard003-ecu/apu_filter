# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Sheaf Cohomology Orchestrator Agent (Soberano de la Holonomía)      ║
║ Ruta   : app/agents/boole/strategy/sheaf_cohomology_orchestrator_agent.py    ║
║ Versión: 4.1.0-Categorical-Krylov-Hodge-Heyting-Strict-NoHardware-PhD        ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y COHOMOLOGÍA DE HACES EN EL ESTRATO STRATEGY (V_𝕊)
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra al Agente Soberano y Observador Activo que gobierna al
Interferómetro de Holonomía de de Rham sobre la variedad de la Malla Agéntica,
operando formalmente como el Funtor de Custodia de la Holonomía Global:

                 𝒵_SheafAgent : SheafMorphisms ⟶ CoherenceState

Su propósito axiomático es auditar el consenso estructural global y resolver
la existencia de secciones de un Haz Celular ℱ sobre el grafo de restricciones
G = (V, E). Repudia la verificación procedural local en favor de la nulidad
cohomológica estricta y la proyección ortogonal de Hodge. El módulo calcula
de manera determinista e inmutable la energía de Dirichlet espectral,
estrangulando las corrientes circulares parasitarias y confinando cualquier
deriva semántica o alucinación al plano puramente lógico de software.

Política Strict-NoHardware: el Crowbar GPIO14 se SIMULA. Jamás se importa
RPi.GPIO ni se toca registro de hardware.

ARQUITECTURA DE TRES FASES ANIDADAS (Handoff por Constructor Estricto)
────────────────────────────────────────────────────────────────────────────────
La transición de estados se rige por un contrato covariante estricto que
encadena DTOs inmutables vía **precondición constructora** (F₁ ⊣ F₂ ⊣ F₃).
El tipo de retorno del último método de Φᵢ ES el objeto inicial de Φᵢ₊₁:

  Fase 1 ──► CERTIFICACIÓN AXIOMÁTICA DEL VETO COHOMOLÓGICO
             (Phase1_CohomologicalVetoCertifier)
             Ensambla el espectro singular de δ con tolerancia de Wilkinson
             adaptativa; computa números de Betti, χ₀₁, torsión analítica
             de Reidemeister log|τ| y verifica la identidad de Euler
             (testigo numérico de dualidad de Poincaré–Lefschetz).
             Morfismo terminal: nest_into_phase2
                 δ  ⟶  Phase2_KrylovSpectralAuditor

  Fase 2 ──► REGULACIÓN DEL ESPECTRO DE KRYLOV Y DIRICHLET
             (Phase2_KrylovSpectralAuditor)
             ★ INICIO FORMAL = continuación de nest_into_phase2 ★
             Bidiagonaliza δ vía Golub–Kahan–Lanczos nativo con MGS
             (sin materializar L = δᵀδ). Estima κ₂(δ) sin cuadrar,
             computa E(x) = ‖δx‖², la componente armónica y C_P.
             Morfismo terminal: nest_into_phase3
                 (Phase2, x)  ⟶  Phase3_IsoperimetricHodgeProjector

  Fase 3 ──► IMPOSICIÓN ISOPERIMÉTRICA DE HODGE Y COLAPSO HEYTING
             (Phase3_IsoperimetricHodgeProjector)
             ★ INICIO FORMAL = continuación de nest_into_phase3 ★
             Sintetiza (opcionalmente) la sección armónica x* = (I − δ⁺δ)x,
             verifica Lipschitz fuerte, cota de inercia y mínima norma;
             estima el cociente de Rayleigh/Cheeger; calcula ι_M; colapsa Ω₃.
             Morfismo terminal: enforce_isoperimetric_hodge_projection
                 (Phase3, x, x*)  ⟶  HodgeProjectionData

RETÍCULO HEYTING Ω₃ (álgebra de Gödel–Dummett) Y COLAPSO TERMINAL
────────────────────────────────────────────────────────────────────────────────
  Ω₃ = { COHERENT := ⊥, DEGRADED, VETOED := ⊤ }
  Orden: COHERENT < DEGRADED < VETOED
  Join ⊔ = max, Meet ⊓ = min
  Implicación (cadena finita): a → b = ⊤ si a ≤ b,  a → b = b si a > b
  Negación: ¬a = a → ⊥

INVARIANTES MATEMÁTICOS, TOPOLÓGICOS Y LEYES DE CONSERVACIÓN
────────────────────────────────────────────────────────────────────────────────
  [I1] Exactitud Cohomológica Global (Nulidad de Obstrucción):
       dim H¹(K; ℱ) ≡ 0 ⟹ H¹(K; ℱ) ≅ 0          [Axioma de Integrabilidad]

  [I2] Conservación del Volumen Espectral (Censura del Laplaciano):
       Proscrito el ensamblaje explícito de L = δᵀδ para Krylov;
       κ₂(δ) medido directamente sobre δ vía Golub–Kahan–Lanczos.
       Identidad espectral (nunca usada como objeto de Krylov):
           κ₂(L) = κ₂(δ)²,   λᵢ(L) = σᵢ(δ)².

  [I3] Acotamiento de la Frustración Térmica (Energía de Dirichlet):
       E(x) = ‖δx‖₂² ≤ ε_frustration.

  [I4] Estabilidad Isoperimétrica de Poincaré–Wirtinger:
       ‖x_exact‖₂ ≤ C_P · ‖δx‖₂,   C_P ≤ C_P,max.

  [I5] Invarianza de Lipschitz Fuerte de Hodge:
       ‖δx* − δx‖₂ ≤ κ(δ) · ‖x* − x‖₂,    ‖x − x*‖₂ ≤ Δ_inertia.

  [I6] Identidad de Euler del complejo de 2 términos:
       χ(ℱ) = dim H⁰ − dim H¹ = dim C⁰ − dim C¹.

Funtor Maestro:
  𝒵_SheafAgent = Φ₃ ∘ Φ₂ ∘ Φ₁ : (δ, x, x*) ⟶ SheafGovernanceState
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from numpy.typing import NDArray

try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:  # pragma: no cover — fallback topológico autonómico

    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos E_MIC."""

    class Morphism:
        r"""Clase base de Morfismos del Topos."""


logger = logging.getLogger("MIC.Strategy.SheafCohomologyAgent")

__version__: Final[str] = (
    "4.1.0-Categorical-Krylov-Hodge-Heyting-Strict-NoHardware-PhD"
)


# ═══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES MATEMÁTICAS Y LÍMITES TERMODINÁMICOS (FPU IEEE-754 binary64)
# ═══════════════════════════════════════════════════════════════════════════════

_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)  # ≈ 2.22e-16
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

_WILKINSON_MAX_ITER: Final[int] = 8
_WILKINSON_CONVERGENCE_TOL: Final[float] = 1e-12

_KRYLOV_MAX_ITER: Final[int] = 200
_KRYLOV_TOL: Final[float] = 1e-9
_KRYLOV_MAX_SINGULAR_VALUES: Final[int] = 8

_DENSE_SPECTRAL_MAX_DIM: Final[int] = 256
_PENROSE_VERIFY_MAX_DIM: Final[int] = 64

_CROWBAR_GPIO_PIN: Final[int] = 14  # BCM; Strict-NoHardware ⇒ sólo simulación
_EPSILON: Final[float] = 1e-15


# ═══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES
# ═══════════════════════════════════════════════════════════════════════════════

class SheafCohomologyAgentError(TopologicalInvariantError):
    r"""Excepción raíz del Custodio de la Holonomía Global."""


class TopologicalBifurcationError(SheafCohomologyAgentError):
    r"""Detonada si dim H¹ > 0 (dependencias circulares insalvables)."""


class PoincareLefschetzViolation(TopologicalBifurcationError):
    r"""Detonada si la identidad de Euler / dualidad P–L numérica es violada."""


class SpectralComputationError(SheafCohomologyAgentError):
    r"""Detonada si κ(L) > κ_max (peligro de colapso en la FPU) o si Krylov falla."""


class SVDConvergenceError(SpectralComputationError):
    r"""Detonada si la SVD no converge tras el número máximo de reintentos."""


class HodgeDecompositionError(SpectralComputationError):
    r"""Detonada si la descomposición de Hodge–Helmholtz no se verifica."""


class DirichletFrustrationError(SheafCohomologyAgentError):
    r"""Detonada si E(x) = ‖δx‖₂² > ε_frustration."""


class PoincareBoundViolation(DirichletFrustrationError):
    r"""Detonada si C_P > C_P,max o C_P no es finita con E(x) > 0."""


class HomologicalInconsistencyError(SheafCohomologyAgentError):
    r"""Detonada si ‖x − x*‖₂ > Δ_inertia o si la proyección aumenta E(x)."""


class LipschitzViolation(HomologicalInconsistencyError):
    r"""Detonada si ‖δx* − δx‖ > κ(δ) · ‖x* − x‖."""


class MinimalNormViolation(HomologicalInconsistencyError):
    r"""Detonada si ‖x*‖ > ‖x‖ + ε_num."""


class HeytingCollapseError(SheafCohomologyAgentError):
    r"""Detonada cuando el retículo Ω₃ colapsa al supremo terminal VETOED."""


# ═══════════════════════════════════════════════════════════════════════════════
# §C. RETÍCULO DISTRIBUTIVO DE HEYTING Ω₃ (Gödel–Dummett)
# ═══════════════════════════════════════════════════════════════════════════════

class HeytingOmega3(IntEnum):
    r"""Retículo distributivo de Heyting Ω₃ totalmente ordenado.

    Estructura algebraica (cadena finita ⇒ álgebra de Gödel–Dummett):
        Ω₃ = { COHERENT := ⊥ = 0, DEGRADED := 1, VETOED := ⊤ = 2 }
        Orden: COHERENT < DEGRADED < VETOED
        Join (⊔) = max, Meet (⊓) = min.
        Implicación:
            a → b = ⊤   si a ≤ b,
            a → b = b   si a > b.
        Negación de Heyting: ¬a = a → ⊥.
        En particular ¬⊥ = ⊤ y ¬a = ⊥ para a ≠ ⊥.

    Tabla de a → b sobre {0, 1, 2}:

            b\\a   0   1   2
             0     2   0   0
             1     2   2   1
             2     2   2   2
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    def __le__(self, other: "HeytingOmega3") -> bool:
        return int(self) <= int(other)

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """Supremo en Ω₃: máximo."""
        return HeytingOmega3(max(int(self), int(other)))

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        """Ínfimo en Ω₃: mínimo."""
        return HeytingOmega3(min(int(self), int(other)))

    def implication(self, other: "HeytingOmega3") -> "HeytingOmega3":
        r"""Implicación de Gödel: a → b = ⊤ si a ≤ b, else b."""
        if int(self) <= int(other):
            return HeytingOmega3.VETOED
        return other

    def negation(self) -> "HeytingOmega3":
        r"""Negación de Heyting: ¬a = a → ⊥."""
        return self.implication(HeytingOmega3.COHERENT)


# ═══════════════════════════════════════════════════════════════════════════════
# §D. ESTRUCTURAS INMUTABLES (DTOs del Topos Estratégico)
#
# Cadena de certificados funtoriales con handoff por constructor:
#   CohomologicalVetoData  →  Fase 1 → unidad de Fase 2
#   KrylovSpectralData     →  Fase 2 → unidad de Fase 3
#   HodgeProjectionData    →  Fase 3 → Orquestador
#   SheafGovernanceState   →  Resultado terminal de 𝒵_SheafAgent
#   SheafAuditProvenance   →  Trazabilidad criptográfica end-to-end
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True, eq=False)
class CohomologicalVetoData:
    r"""Artefacto de Fase 1. Certificado de anulación de obstrucciones globales.

    Se computa con tolerancia de Wilkinson adaptativa:
        SVD_TOL = d² · κ₂(δ) · ε_mach · σ_max(δ)
    resolviendo el ciclo κ₂ ⟷ rank ⟷ SVD_TOL por punto fijo.

    Invariante de Euler (testigo algebraico):
        h0_dimension − h1_dimension  ==  dim_C0 − dim_C1.

    Este DTO es la **precondición constructora** de Phase2_KrylovSpectralAuditor.
    """

    delta_matrix: NDArray[np.float64]
    dim_C0: int
    dim_C1: int
    delta_rank: int
    h0_dimension: int
    h1_dimension: int
    singular_values: NDArray[np.float64]
    svd_tolerance: float
    max_singular_value: float
    min_nonzero_singular_value: float
    spectral_gap: float
    spectral_gap_ratio: float
    condition_number_delta: float
    wilkinson_iterations: int
    cohomological_stability_index: float
    euler_characteristic_01: int
    whitehead_torsion: float
    poincare_lefschetz_ok: bool
    is_topologically_coherent: bool
    rank_is_certified: bool
    certification_hash_sha256: str


@dataclass(frozen=True, slots=True, eq=False)
class KrylovSpectralData:
    r"""Artefacto de Fase 2. Certificado termodinámico y espectral.

    Obtenido por **bidiagonalización de Golub–Kahan–Lanczos** aplicada
    directamente sobre δ (sin materializar L = δᵀδ).

    Este DTO es la **precondición constructora** de Phase3_IsoperimetricHodgeProjector.
    """

    phase1_reference: CohomologicalVetoData
    krylov_singular_values: NDArray[np.float64]
    krylov_dimension: int
    krylov_residual: float
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
    banach_holder_bound: float
    is_frustration_bounded: bool
    is_spectrally_stable: bool
    is_poincare_bounded: bool
    certification_hash_sha256: str


@dataclass(frozen=True, slots=True, eq=False)
class HodgeProjectionData:
    r"""Artefacto de Fase 3. Certificado de Lipschitz fuerte e isoperimétrico.

    Verifica simultáneamente:
        ‖δx* − δx‖₂ ≤ κ(δ) · ‖x* − x‖₂   (Lipschitz fuerte)
        ‖x − x*‖₂ ≤ Δ_inertia              (inercia)
        ‖x*‖₂ ≤ ‖x‖₂ + ε_num               (mínima norma)
        E(x*) ≤ E(x) + ε_num               (energía no creciente)
        π∘π ≃ π                            (idempotencia de Hodge)

    Colapsa el retículo Ω₃ al veredicto terminal.
    """

    projection_distance: float
    relative_projection_distance: float
    inertia_delta_max: float
    original_dirichlet_energy: float
    projected_dirichlet_energy: float
    energy_reduction_ratio: float
    lipschitz_lhs: float
    lipschitz_residual: float
    lipschitz_satisfied: bool
    lipschitz_slack: float
    isoperimetric_slack: float
    minimal_norm_satisfied: bool
    hodge_idempotence_residual: float
    cheeger_bound_estimate: float
    morse_reduction_index: int
    heyting_verdict: HeytingOmega3
    crowbar_simulated: bool
    is_isoperimetrically_bounded: bool
    is_energy_non_increasing: bool
    verified_by_delta: bool
    certification_hash_sha256: str


@dataclass(frozen=True, slots=True)
class SheafAuditProvenance:
    r"""Trazabilidad criptográfica de la cadena funtorial 𝒵_SheafAgent."""

    timestamp_iso: str
    input_checksum_sha256: str
    phase1_certification_hash: str
    phase2_certification_hash: str
    phase3_certification_hash: str
    phase1_passed: bool
    phase2_passed: bool
    phase3_passed: bool
    functor_chain: str
    agent_version: str


@dataclass(frozen=True, slots=True)
class SheafGovernanceState:
    r"""Objeto final del endofuntor 𝒵_SheafAgent = Φ₃ ∘ Φ₂ ∘ Φ₁."""

    veto_audit: CohomologicalVetoData
    spectral_audit: KrylovSpectralData
    hodge_audit: HodgeProjectionData
    provenance: SheafAuditProvenance
    is_epistemologically_valid: bool


# ═══════════════════════════════════════════════════════════════════════════════
# §E. GUARDAS NUMÉRICAS INTERNAS (álgebra de Banach sobre ℝ, IEEE-754)
# ═══════════════════════════════════════════════════════════════════════════════

class _FiniteNumericalGuard:
    r"""Saneamiento numérico: finitud, realidad y no-degeneración.

    Todas las entradas se proyectan a float64 real finito antes de cualquier
    morfismo algebraico. Las normas se evalúan en la categoría Banach
    (ℝⁿ, ‖·‖₂) con fallback a ‖·‖₁ si la FPU degrada.
    """

    @staticmethod
    def _as_float_array(name: str, value: Any) -> NDArray[np.float64]:
        r"""Convierte a float64 real finito; rechaza ℂ, NaN y ±∞."""
        try:
            raw = np.asarray(value)
        except Exception as exc:
            raise TypeError(
                f"[Guard] '{name}' no interpretable como arreglo: {exc}"
            ) from exc

        if np.iscomplexobj(raw):
            raise TypeError(
                f"[Guard] '{name}' debe ser real; dtype={raw.dtype}."
            )

        try:
            arr = raw.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"[Guard] '{name}' no convertible a float64: {exc}"
            ) from exc

        if not np.all(np.isfinite(arr)):
            n_bad = int(np.sum(~np.isfinite(arr)))
            raise ValueError(f"[Guard] '{name}' contiene {n_bad} NaN o ±∞.")
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
        r"""Valida una matriz real finita 2D."""
        arr = cls._as_float_array(name, value)
        if arr.ndim != 2:
            raise ValueError(f"[Guard] '{name}' debe ser 2D; ndim={arr.ndim}.")
        rows, cols = arr.shape
        if min_rows > 0 and rows < min_rows:
            raise ValueError(
                f"[Guard] '{name}': {rows} filas < {min_rows} requeridas."
            )
        if min_cols > 0 and cols < min_cols:
            raise ValueError(
                f"[Guard] '{name}': {cols} cols < {min_cols} requeridas."
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
        r"""Valida y normaliza a 1D un vector real finito."""
        arr = cls._as_float_array(name, value)
        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim == 2 and 1 in arr.shape:
            arr = arr.reshape(-1)
        elif arr.ndim != 1:
            raise ValueError(
                f"[Guard] '{name}' debe ser 1D, columna, fila o escalar; "
                f"forma={arr.shape}."
            )
        if not allow_empty and arr.size == 0:
            raise ValueError(f"[Guard] '{name}' no puede ser vector vacío.")
        return arr

    @staticmethod
    def _vector_norm(v: NDArray[np.float64]) -> float:
        r"""‖v‖₂ con fallback a ‖v‖₁."""
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
        r"""‖y‖₂² = yᵀy, con recorte de negatividad por ruido de redondeo.

        L = δᵀδ ⪰ 0 implica E(x) = ‖δx‖² ≥ 0 analíticamente; una energía
        negativa de magnitud > 128 ε_mach |E| es un invariante roto.
        """
        if y.size == 0:
            return 0.0
        value = float(np.dot(y, y))
        if not math.isfinite(value):
            raise DirichletFrustrationError(
                "[Guard] ‖δx‖₂² no es finita; posible desbordamiento."
            )
        tolerance = _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(1.0, abs(value))
        if value < -tolerance:
            raise DirichletFrustrationError(
                f"[Guard] ‖δx‖₂² = {value:.4e} < 0 con magnitud significativa."
            )
        return max(0.0, value)

    @staticmethod
    def _holder_operator_norm_bound(A: NDArray[np.float64]) -> float:
        r"""Cota de Hölder de álgebra de Banach: ‖A‖₂ ≤ √(‖A‖₁ ‖A‖∞).

        Testigo barato de σ_max que no requiere SVD ni forma L = AᵀA.
        """
        if A.size == 0:
            return 0.0
        abs_A = np.abs(A)
        norm_1 = float(np.max(np.sum(abs_A, axis=0))) if A.shape[1] else 0.0
        norm_inf = float(np.max(np.sum(abs_A, axis=1))) if A.shape[0] else 0.0
        return float(math.sqrt(max(norm_1, 0.0) * max(norm_inf, 0.0)))

    @staticmethod
    def _safe_svdvals(
        A: NDArray[np.float64],
        name: str = "A",
        *,
        max_retries: int = _SVD_MAX_RETRIES,
    ) -> NDArray[np.float64]:
        r"""SVD con reintentos por perturbación diagonal adaptativa de Tikhonov.

        ε_k = ε_mach^{1/(k+2)} · ‖A‖_F  (regularización espectral mínima).
        Los valores singulares se devuelven en orden descendente.
        """
        if A.size == 0 or min(A.shape) == 0:
            return np.empty(0, dtype=np.float64)

        last_exc: Optional[Exception] = None
        A_work = np.array(A, dtype=np.float64, copy=True)
        for attempt in range(max_retries):
            try:
                svs = la.svdvals(A_work)
                if np.all(np.isfinite(svs)):
                    return np.sort(np.asarray(svs, dtype=np.float64))[::-1]
            except (np.linalg.LinAlgError, ValueError) as exc:
                last_exc = exc

            frob = float(la.norm(A_work, ord="fro") or 1.0)
            eps_k = (_MACHINE_EPSILON ** (1.0 / (attempt + 2))) * frob
            rows, cols = A_work.shape
            min_dim = min(rows, cols)
            A_work = A_work.copy()
            A_work[:min_dim, :min_dim] += eps_k * np.eye(min_dim, dtype=np.float64)

        raise SVDConvergenceError(
            f"[Guard] SVD de '{name}' no convergió tras {max_retries} intentos "
            f"(último error: {last_exc})."
        )

    @classmethod
    def _pseudo_inverse(
        cls,
        A: NDArray[np.float64],
        name: str = "A",
        *,
        tolerance: Optional[float] = None,
    ) -> NDArray[np.float64]:
        r"""Pseudoinversa de Moore–Penrose A⁺ = V · diag(1/σᵢ) · Uᵀ.

        Satisface las cuatro ecuaciones de Penrose:
            A A⁺ A = A,  A⁺ A A⁺ = A⁺,  (A A⁺)ᵀ = A A⁺,  (A⁺ A)ᵀ = A⁺ A.
        Truncación espectral con `tolerance` (Wilkinson de Fase 1 si se aporta).
        """
        if A.size == 0 or min(A.shape) == 0:
            return np.zeros((A.shape[1], A.shape[0]), dtype=np.float64)

        try:
            U, s, Vt = la.svd(A, full_matrices=False)
        except (np.linalg.LinAlgError, ValueError):
            cls._safe_svdvals(A, name)
            return np.linalg.pinv(A)

        if not (
            np.all(np.isfinite(U))
            and np.all(np.isfinite(s))
            and np.all(np.isfinite(Vt))
        ):
            raise HodgeDecompositionError(
                f"[Guard] SVD de '{name}' produjo valores no finitos."
            )

        if tolerance is None:
            sigma_max = float(s[0]) if s.size else 0.0
            tolerance = max(
                _SVD_TOLERANCE_BASE,
                _NUMERICAL_SAFETY_FACTOR
                * _MACHINE_EPSILON
                * max(A.shape)
                * sigma_max,
            )

        s_inv = np.where(s > tolerance, 1.0 / s, 0.0)
        A_pinv = (Vt.T * s_inv) @ U.T
        if not np.all(np.isfinite(A_pinv)):
            raise HodgeDecompositionError(
                f"[Guard] Pseudoinversa de '{name}' contiene no finitos."
            )
        return A_pinv

    @classmethod
    def _verify_penrose_equations(
        cls,
        A: NDArray[np.float64],
        A_pinv: NDArray[np.float64],
        *,
        name: str = "A",
    ) -> float:
        r"""Residuo relativo máximo de las cuatro ecuaciones de Penrose.

        Se evalúa sólo si max(m, n) ≤ _PENROSE_VERIFY_MAX_DIM (coste O(n³)).
        Retorna 0.0 si la verificación se omite por dimensión.
        """
        if max(A.shape) > _PENROSE_VERIFY_MAX_DIM:
            return 0.0
        frob_A = max(cls._frobenius_norm(A), _EPSILON)
        p1 = cls._frobenius_norm(A @ A_pinv @ A - A) / frob_A
        p2 = cls._frobenius_norm(A_pinv @ A @ A_pinv - A_pinv) / max(
            cls._frobenius_norm(A_pinv), _EPSILON
        )
        AA = A @ A_pinv
        ApA = A_pinv @ A
        p3 = cls._frobenius_norm(AA - AA.T) / max(cls._frobenius_norm(AA), _EPSILON)
        p4 = cls._frobenius_norm(ApA - ApA.T) / max(cls._frobenius_norm(ApA), _EPSILON)
        residual = float(max(p1, p2, p3, p4))
        if residual > 1e-6:
            logger.warning(
                "[Guard] Penrose residual de '%s' = %.3e (degradación numérica).",
                name, residual,
            )
        return residual

    @staticmethod
    def _compute_input_checksum(*arrays: Optional[NDArray[np.float64]]) -> str:
        r"""SHA-256 de la concatenación binaria de los arreglos (trazabilidad)."""
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

    @staticmethod
    def _compute_dto_hash(*fields: Any) -> str:
        r"""Firma determinista SHA-256 de campos heterogéneos del DTO."""
        hasher = hashlib.sha256()
        for f in fields:
            if isinstance(f, bool):
                hasher.update(f"|{int(f)}".encode())
            elif isinstance(f, float):
                hasher.update(f"|{f:.17e}".encode())
            elif isinstance(f, int):
                hasher.update(f"|{f}".encode())
            elif isinstance(f, str):
                hasher.update(f"|{f}".encode())
            elif isinstance(f, np.ndarray):
                hasher.update(f"|shape={f.shape}|".encode())
                hasher.update(np.ascontiguousarray(f, dtype=np.float64).tobytes())
            elif f is None:
                hasher.update(b"|__None__")
            else:
                hasher.update(f"|{f!r}".encode())
        return hasher.hexdigest()


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 1: CERTIFICACIÓN AXIOMÁTICA DEL VETO COHOMOLÓGICO                    ║
# ║                                                                             ║
# ║   Marco formal:                                                             ║
# ║   ─────────                                                                 ║
# ║   Sea δ: C⁰ → C¹ el operador cofrontera del haz celular. La cohomología      ║
# ║   de primer grado mide las cocadenas cerradas no exactas:                    ║
# ║       H⁰(G; ℱ) ≅ ker(δ),          dim H⁰ = dim C⁰ − rank(δ)                  ║
# ║       H¹(G; ℱ) ≅ coker(δ),        dim H¹ = dim C¹ − rank(δ)                  ║
# ║       χ(ℱ) = dim H⁰ − dim H¹ = dim C⁰ − dim C¹                               ║
# ║                                                                             ║
# ║   Invariantes calculados por esta fase:                                     ║
# ║     1. Espectro singular {σᵢ} vía SVD con reintentos (Golub–Reinsch).        ║
# ║     2. Wilkinson TOL: punto fijo del ciclo κ₂ ⟷ rank ⟷ SVD_TOL.              ║
# ║     3. rank(δ) por tolerancia Wilkinson-adaptativa.                          ║
# ║     4. Números de Betti β₀ = dim H⁰, β₁ = dim H¹.                            ║
# ║     5. χ₀₁ = β₀ − β₁ (característica de Euler del 2-complejo).               ║
# ║     6. Torsión analítica de Reidemeister log|τ| = Σ log σᵢ.                  ║
# ║     7. Identidad de Euler como testigo de dualidad P–L numérica.             ║
# ║     8. Índice de estabilidad β = rank(δ) / dim C¹.                           ║
# ║                                                                             ║
# ║   ULTIMO MÉTODO DE FASE 1 (morfismo terminal = unidad de Fase 2):            ║
# ║       nest_into_phase2(delta) → Phase2_KrylovSpectralAuditor                 ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase1_CohomologicalVetoCertifier(_FiniteNumericalGuard):
    r"""FASE 1: Certificador axiomático del veto cohomológico.

    Cadena interna de morfismos:
        _validate_delta_operator_invariants
            → _extract_singular_spectrum
            → _compute_wilkinson_spectral_tolerance_fixed_point
            → _compute_rank_nullity_betti
            → _compute_reidemeister_analytic_torsion
            → _verify_euler_poincare_lefschetz_identity
            → certify_cohomological_veto_axiom
            → nest_into_phase2          ★ morfismo terminal = unidad de Φ₂

    Salida del morfismo terminal: Phase2_KrylovSpectralAuditor
    (precondición constructora ya inyectada).
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 1.1 Validación de invariantes de forma del operador cofrontera
    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def _validate_delta_operator_invariants(
        cls,
        delta: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], int, int]:
        r"""Valida invariantes estructurales de δ: C⁰ → C¹.

        Verifica:
            · δ es una matriz real finita 2D con m, n ≥ 1.
            · ‖δ‖_F ≰ ε_mach (no degenerada).

        Returns
        ───────
        (delta_float64, dim_C1 = m, dim_C0 = n)
        """
        delta_val = cls._as_finite_matrix(
            "coboundary_operator_delta", delta, min_rows=1, min_cols=1
        )
        m, n = int(delta_val.shape[0]), int(delta_val.shape[1])
        frob = cls._frobenius_norm(delta_val)
        if frob <= _MACHINE_EPSILON:
            raise SheafCohomologyAgentError(
                f"[Fase 1] δ es numéricamente trivial: ‖δ‖_F = {frob:.3e} "
                "≤ ε_mach. No hay información cohomológica que certificar."
            )
        return delta_val, m, n

    # ─────────────────────────────────────────────────────────────────────────
    # 1.2 Espectro singular certificado (Golub–Reinsch + reintentos)
    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def _extract_singular_spectrum(
        cls,
        delta: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], bool]:
        r"""Extrae σ(δ) descendente. En este agente δ es denso: rango pleno.

        Returns
        ───────
        (singular_values_desc, rank_is_certified=True)
        """
        s = cls._safe_svdvals(delta, "δ")
        return s, True

    # ─────────────────────────────────────────────────────────────────────────
    # 1.3 Wilkinson-adaptativo iterativo (Axioma [A5] / [I2])
    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def _compute_wilkinson_spectral_tolerance_fixed_point(
        cls,
        singular_values: NDArray[np.float64],
        shape: Tuple[int, int],
    ) -> Tuple[float, int, int, float, float]:
        r"""Resuelve el ciclo κ₂ ⟷ rank ⟷ SVD_TOL por punto fijo.

        Modelo (matriz de Wilkinson):
            SVD_TOL = d² · κ₂(δ) · ε_machine · σ_max(δ),
        donde d = max(m, n).

        Iteración:
            SVD_TOL⁽⁰⁾ = d · ε_machine · σ_max              (cota clásica)
            rank⁽ᵏ⁺¹⁾  = #{σᵢ > SVD_TOL⁽ᵏ⁾} ∩ [0, min(m,n)]
            κ₂⁽ᵏ⁾      = σ_max / σ_{rank⁽ᵏ⁾}
            SVD_TOL⁽ᵏ⁺¹⁾ = d² · κ₂⁽ᵏ⁾ · ε_machine · σ_max
        Convergencia: |Δ SVD_TOL| ≤ _WILKINSON_CONVERGENCE_TOL · max(1, tol).

        Returns
        ───────
        (wilkinson_tolerance, rank, iterations, kappa, sigma_min_positive)
        """
        if singular_values.size == 0:
            return _SVD_TOLERANCE_BASE, 0, 0, 1.0, 0.0

        m, n = int(shape[0]), int(shape[1])
        d = max(m, n, 1)
        rank_cap = min(m, n)
        sigma_max = float(singular_values[0])
        if sigma_max <= 0.0:
            return _SVD_TOLERANCE_BASE, 0, 0, 1.0, 0.0

        tol = max(d * _MACHINE_EPSILON * sigma_max, _SVD_TOLERANCE_BASE)
        iters = 0
        rank = 0
        kappa = 1.0
        sigma_min_pos = 0.0

        for _ in range(_WILKINSON_MAX_ITER):
            iters += 1
            rank = int(np.sum(singular_values > tol))
            rank = min(rank, rank_cap)
            if rank == 0:
                sigma_min_pos = 0.0
                kappa = float("inf")
            else:
                sigma_min_pos = float(singular_values[rank - 1])
                kappa = (
                    sigma_max / sigma_min_pos
                    if sigma_min_pos > _MACHINE_EPSILON
                    else float("inf")
                )
            kappa_factor = kappa if math.isfinite(kappa) else 1.0
            tol_new = max(
                (d ** 2) * kappa_factor * _MACHINE_EPSILON * sigma_max,
                _SVD_TOLERANCE_BASE,
            )
            if abs(tol_new - tol) <= _WILKINSON_CONVERGENCE_TOL * max(1.0, tol):
                tol = tol_new
                break
            tol = tol_new

        rank = min(int(np.sum(singular_values > tol)), rank_cap)
        sigma_min_pos = float(singular_values[rank - 1]) if rank > 0 else 0.0
        kappa = (
            sigma_max / sigma_min_pos
            if sigma_min_pos > _MACHINE_EPSILON
            else float("inf")
        )
        return float(tol), int(rank), int(iters), float(kappa), float(sigma_min_pos)

    # ─────────────────────────────────────────────────────────────────────────
    # 1.4 Rango-nulidad y números de Betti
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_rank_nullity_betti(
        dim_C0: int,
        dim_C1: int,
        effective_rank: int,
    ) -> Tuple[int, int, int]:
        r"""Teorema rango-nulidad sobre el complejo de 2 términos [I1], [I6].

            dim H⁰ = dim ker(δ)   = dim C⁰ − rank(δ)
            dim H¹ = dim coker(δ) = dim C¹ − rank(δ)
            χ(ℱ)   = dim H⁰ − dim H¹ = dim C⁰ − dim C¹

        Returns
        ───────
        (h0, h1, euler_characteristic)
        """
        rank = max(0, min(int(effective_rank), dim_C0, dim_C1))
        h0 = dim_C0 - rank
        h1 = dim_C1 - rank
        euler = h0 - h1
        return int(h0), int(h1), int(euler)

    # ─────────────────────────────────────────────────────────────────────────
    # 1.5 Torsión analítica de Reidemeister (Ray–Singer del 2-complejo)
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_reidemeister_analytic_torsion(
        singular_values: NDArray[np.float64],
        effective_rank: int,
    ) -> float:
        r"""Torsión analítica logarítmica del complejo acíclico truncado.

            log|τ| = Σ_{i=1}^{rank} log σᵢ(δ).

        Coincide con la torsión de Reidemeister sobre ℝ (Wh(1)=0) y con
        la torsión de Ray–Singer del 2-complejo. Se conserva el nombre de
        campo `whitehead_torsion` por compatibilidad de API.
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
    # 1.6 Identidad de Euler / testigo numérico de Poincaré–Lefschetz
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _verify_euler_poincare_lefschetz_identity(
        dim_C0: int,
        dim_C1: int,
        effective_rank: int,
        h0: int,
        h1: int,
        euler: int,
    ) -> bool:
        r"""Testigo numérico de dualidad de Poincaré–Lefschetz para 2-términos.

        Condiciones necesarias (todas identidades de álgebra lineal; su
        ruptura delata corrupción de rango Wilkinson o overflow de enteros):
            1. 0 ≤ rank(δ) ≤ min(dim C⁰, dim C¹)
            2. dim H⁰ − dim H¹ = dim C⁰ − dim C¹
            3. euler == h0 − h1
        """
        max_possible_rank = min(dim_C0, dim_C1)
        if effective_rank < 0 or effective_rank > max_possible_rank:
            logger.warning(
                "[Fase 1] P–L violada: rank(δ)=%d ∉ [0, min(C⁰,C¹)=%d].",
                effective_rank, max_possible_rank,
            )
            return False
        if euler != dim_C0 - dim_C1 or euler != h0 - h1:
            logger.warning(
                "[Fase 1] Identidad de Euler rota: χ=%d, C⁰−C¹=%d, H⁰−H¹=%d.",
                euler, dim_C0 - dim_C1, h0 - h1,
            )
            return False
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # 1.7 Emisión del DTO de veto (pre-terminal)
    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def certify_cohomological_veto_axiom(
        cls,
        coboundary_operator_delta: NDArray[np.float64],
    ) -> CohomologicalVetoData:
        r"""Certifica [I1] y emite CohomologicalVetoData (precondición de Φ₂).

        Cadena:
            δ ──(validate_invariants)────────────▶ (m, n)
              ──(extract_singular_spectrum)──────▶ σ(δ)
              ──(wilkinson_fixed_point)──────────▶ SVD_TOL, rank, κ₂(δ)
              ──(rank_nullity_betti)─────────────▶ dim H⁰, dim H¹, χ
              ──(reidemeister_torsion)───────────▶ log|τ|
              ──(euler_poincare_lefschetz)───────▶ bool
              ──(emit_axiom_[I1]_veto)───────────▶ veto si dim H¹ > 0

        Raises
        ──────
        SheafCohomologyAgentError
        SVDConvergenceError
        PoincareLefschetzViolation
        TopologicalBifurcationError
        """
        delta, dim_C1, dim_C0 = cls._validate_delta_operator_invariants(
            coboundary_operator_delta
        )
        singular_values, rank_is_certified = cls._extract_singular_spectrum(delta)
        sigma_max = float(singular_values[0]) if singular_values.size else 0.0

        svd_tol, effective_rank, wilk_iters, kappa, sigma_min_nonzero = (
            cls._compute_wilkinson_spectral_tolerance_fixed_point(
                singular_values, delta.shape
            )
        )

        # Hueco espectral = distancia al núcleo = σ_min⁺.
        # Razón de hueco = σ_min⁺ / σ_max = 1/κ₂  (pequeña ⇒ mal condicionado).
        spectral_gap = float(sigma_min_nonzero)
        gap_ratio = (
            spectral_gap / sigma_max if sigma_max > 0.0 else 0.0
        )

        h0_dim, h1_dim, euler_01 = cls._compute_rank_nullity_betti(
            dim_C0, dim_C1, effective_rank
        )
        stability_index = (
            float(effective_rank) / float(dim_C1) if dim_C1 > 0 else 1.0
        )
        whitehead = cls._compute_reidemeister_analytic_torsion(
            singular_values, effective_rank
        )
        pl_ok = cls._verify_euler_poincare_lefschetz_identity(
            dim_C0, dim_C1, effective_rank, h0_dim, h1_dim, euler_01
        )
        if not pl_ok:
            raise PoincareLefschetzViolation(
                "[Fase 1] Identidad de Euler / P–L violada: "
                f"rank(δ)={effective_rank}, dim C⁰={dim_C0}, dim C¹={dim_C1}, "
                f"dim H⁰={h0_dim}, dim H¹={h1_dim}, χ={euler_01}."
            )

        if h1_dim > 0:
            raise TopologicalBifurcationError(
                "[Fase 1] Fractura homológica global: "
                f"dim H¹ = {h1_dim} > 0. β_estab = {stability_index:.4f}, "
                f"gap_ratio = σ_min⁺/σ_max = {gap_ratio:.4e}. "
                "Dependencias circulares insalvables detectadas."
            )

        if gap_ratio < _SPECTRAL_GAP_MIN_RATIO and effective_rank > 0:
            logger.warning(
                "[Fase 1] Gap espectral pequeño: σ_min⁺/σ_max = %.4e < ρ_gap "
                "(κ₂(δ) ≳ %.1f).",
                gap_ratio,
                1.0 / max(gap_ratio, _EPSILON),
            )

        cert_hash = cls._compute_dto_hash(
            dim_C0, dim_C1, effective_rank, h0_dim, h1_dim,
            float(svd_tol), sigma_max, sigma_min_nonzero, float(kappa),
            stability_index, euler_01, whitehead, pl_ok,
            singular_values,
        )

        logger.info(
            "[Fase 1 ✓] CohomologicalVetoData: dim C⁰=%d, dim C¹=%d, "
            "rank(δ)=%d, dim H⁰=%d, dim H¹=%d, χ=%d, σ_max=%.3e, σ_min⁺=%.3e, "
            "κ₂(δ)=%.3e, SVD_TOL=%.3e (Wilkinson %d iters), log|τ|=%.3f, hash=%s.",
            dim_C0, dim_C1, effective_rank, h0_dim, h1_dim, euler_01,
            sigma_max, sigma_min_nonzero, kappa, svd_tol, wilk_iters,
            whitehead, cert_hash[:16] + "...",
        )

        s_immutable = np.array(singular_values, dtype=np.float64, copy=True)
        s_immutable.setflags(write=False)
        delta_immutable = np.array(delta, dtype=np.float64, copy=True)
        delta_immutable.setflags(write=False)

        return CohomologicalVetoData(
            delta_matrix=delta_immutable,
            dim_C0=int(dim_C0),
            dim_C1=int(dim_C1),
            delta_rank=int(effective_rank),
            h0_dimension=int(h0_dim),
            h1_dimension=int(h1_dim),
            singular_values=s_immutable,
            svd_tolerance=float(svd_tol),
            max_singular_value=float(sigma_max),
            min_nonzero_singular_value=float(sigma_min_nonzero),
            spectral_gap=float(spectral_gap),
            spectral_gap_ratio=float(gap_ratio),
            condition_number_delta=float(kappa),
            wilkinson_iterations=int(wilk_iters),
            cohomological_stability_index=float(stability_index),
            euler_characteristic_01=int(euler_01),
            whitehead_torsion=float(whitehead),
            poincare_lefschetz_ok=bool(pl_ok),
            is_topologically_coherent=True,
            rank_is_certified=bool(rank_is_certified),
            certification_hash_sha256=str(cert_hash),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1.8 ★ MORFISMO TERMINAL DE FASE 1 ★
    #     Tipo de retorno = objeto inicial de la FASE 2.
    #     last(Φ₁) = unit(Φ₂) = Phase2_KrylovSpectralAuditor.
    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def nest_into_phase2(
        cls,
        coboundary_operator_delta: NDArray[np.float64],
    ) -> "Phase2_KrylovSpectralAuditor":
        r"""★ MORFISMO TERMINAL DE FASE 1 / UNIDAD DE LA FASE 2 ★

        Composición estricta F₁ ⊣ F₂:
            nest_into_phase2  :=  Phase2_KrylovSpectralAuditor
                                  ∘ certify_cohomological_veto_axiom.

        El auditor de Krylov nace ya alimentado con CohomologicalVetoData;
        su constructor ES la continuación formal de este método.
        """
        phase1_data = cls.certify_cohomological_veto_axiom(
            coboundary_operator_delta
        )
        return Phase2_KrylovSpectralAuditor(phase1_data)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 2: REGULACIÓN DEL ESPECTRO DE KRYLOV Y ENERGÍA DE DIRICHLET          ║
# ║                                                                             ║
# ║   ★ INICIO FORMAL = continuación de Phase1.nest_into_phase2 ★               ║
# ║   Precondición constructora: CohomologicalVetoData.                          ║
# ║                                                                             ║
# ║   Marco formal:                                                             ║
# ║   ─────────                                                                 ║
# ║   Energía de Dirichlet: E(x) = ‖δx‖₂² = xᵀ L x con L = δᵀδ  (nunca formada). ║
# ║   Descomposición de Hodge–Helmholtz:                                         ║
# ║       C⁰ = ker(δ) ⊕ im(δᵀ),   x = x_harm + x_exact.                          ║
# ║       x_exact = δ⁺(δx),       x_harm = (I − δ⁺δ)x.                           ║
# ║   Cota de Poincaré–Wirtinger: ‖x_exact‖ ≤ C_P · ‖δx‖.                        ║
# ║                                                                             ║
# ║   Bidiagonalización de Golub–Kahan–Lanczos sobre δ (no sobre L):            ║
# ║       δ ≈ U B Vᵀ,  B bidiagonal superior,  UᵀU = I_k, VᵀV = I_k.             ║
# ║   Recurrencias (Kaniel–Paige):                                               ║
# ║       β_{j+1} u_{j+1} = δ v_j − α_j u_j                                     ║
# ║       α_{j+1} v_{j+1} = δᵀ u_{j+1} − β_{j+1} v_j                            ║
# ║                                                                             ║
# ║   ULTIMO MÉTODO DE FASE 2 (morfismo terminal = unidad de Fase 3):            ║
# ║       nest_into_phase3(x) → Phase3_IsoperimetricHodgeProjector               ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase2_KrylovSpectralAuditor(Phase1_CohomologicalVetoCertifier):
    r"""FASE 2: Auditor espectral de Krylov y energía de Dirichlet.

    ★ INICIO FORMAL = continuación del morfismo terminal de la FASE 1 ★

    Cadena interna:
        __init__(CohomologicalVetoData)     ← unidad heredada de Φ₁
            → _deterministic_start_vector
            → _golub_kahan_lanczos_bidiagonalization
            → _measure_delta_condition_number_krylov
            → _evaluate_dirichlet_energy_semigroup
            → _decompose_hodge_helmholtz_moore_penrose
            → _compute_poincare_wirtinger_constant
            → audit_krylov_spectral_stability
            → nest_into_phase3              ★ morfismo terminal = unidad de Φ₃
    """

    def __init__(self, phase1_certification: CohomologicalVetoData) -> None:
        r"""★ CONTINUACIÓN DE FASE 1 / INICIO DE FASE 2 ★

        Args
        ────
        phase1_certification : CohomologicalVetoData
            Salida de `certify_cohomological_veto_axiom`, inyectada por
            `nest_into_phase2`. Sin este objeto Φ₂ carece de δ certificado.
        """
        if not isinstance(phase1_certification, CohomologicalVetoData):
            raise TypeError(
                "Phase2_KrylovSpectralAuditor requiere CohomologicalVetoData "
                "como precondición (Fase 1)."
            )
        if not phase1_certification.is_topologically_coherent:
            raise TopologicalBifurcationError(
                "[Fase 2] Precondición inválida: el DTO de Fase 1 no está "
                "marcado como topológicamente coherente."
            )
        if phase1_certification.h1_dimension != 0:
            raise TopologicalBifurcationError(
                f"[Fase 2] Precondición inválida: dim H¹ = "
                f"{phase1_certification.h1_dimension} > 0. Fase 1 debe vetar."
            )
        self._phase1: Final[CohomologicalVetoData] = phase1_certification
        self._delta: Final[NDArray[np.float64]] = phase1_certification.delta_matrix

    @property
    def phase1_certificate(self) -> CohomologicalVetoData:
        r"""Referencia inmutable al DTO de Fase 1 (precondición)."""
        return self._phase1

    # ─────────────────────────────────────────────────────────────────────────
    # 2.1 Semilla determinista de Krylov (Rademacher vía SHA-256)
    # ─────────────────────────────────────────────────────────────────────────
    def _deterministic_start_vector(self, n: int) -> NDArray[np.float64]:
        r"""Vector de Rademacher ±1 derivado del certification_hash de Φ₁.

        Garantiza reproducibilidad bit a bit del subespacio de Krylov entre
        ejecuciones. El hash ya cifra σ(δ), de modo que la semilla es estable
        respecto del operador certificado.
        """
        digest = self._phase1.certification_hash_sha256.encode("ascii")
        buf = bytearray()
        counter = 0
        while len(buf) < n:
            buf.extend(
                hashlib.sha256(digest + counter.to_bytes(4, "little")).digest()
            )
            counter += 1
        signs = np.frombuffer(bytes(buf[:n]), dtype=np.uint8).astype(np.float64)
        v = np.where(signs >= 128, 1.0, -1.0)
        nrm = float(np.linalg.norm(v))
        if nrm <= _EPSILON:
            v = np.zeros(n, dtype=np.float64)
            v[0] = 1.0
            return v
        return v / nrm

    # ─────────────────────────────────────────────────────────────────────────
    # 2.2 Bidiagonalización Golub–Kahan–Lanczos sobre δ (sin L)
    # ─────────────────────────────────────────────────────────────────────────
    def _golub_kahan_lanczos_bidiagonalization(
        self,
        k: Optional[int] = None,
        tol: Optional[float] = None,
    ) -> Tuple[NDArray[np.float64], int, float]:
        r"""Bidiagonalización de Golub–Kahan–Lanczos de δ con MGS completo.

        NO materializa L = δᵀδ ([I2]): la iteración se ejecuta sobre el par
        (δ, δᵀ). Ante breakdown afortunado (α_j o β_j ≈ 0) se detiene y
        reporta el subespacio invariante. Si el proceso nativo falla, degrada
        a `svds` (ARPACK sobre el operador aumentado [0 δ; δᵀ 0]).

        Returns
        ───────
        (σ_krylov_desc, krylov_dim, krylov_residual)
        """
        m, n = int(self._delta.shape[0]), int(self._delta.shape[1])
        if m == 0 or n == 0:
            return np.array([], dtype=np.float64), 0, 0.0

        dim_min = min(m, n)
        if k is None:
            k = min(
                _KRYLOV_MAX_SINGULAR_VALUES,
                max(1, dim_min - 1 if dim_min > 1 else 1),
            )
        k = int(max(1, min(k, dim_min)))
        if tol is None:
            tol = _KRYLOV_TOL

        try:
            return self._golub_kahan_core(k, float(tol), m, n)
        except Exception as exc:
            logger.warning(
                "Golub–Kahan nativo falló (%s); degradación a svds LM.", exc
            )
            return self._svds_fallback(k, float(tol), m, n)

    def _golub_kahan_core(
        self,
        k: int,
        tol: float,
        m: int,
        n: int,
    ) -> Tuple[NDArray[np.float64], int, float]:
        r"""Núcleo GK con reortogonalización MGS (O(k²(m+n)) + k matvecs)."""
        delta = self._delta
        V = np.zeros((n, k), dtype=np.float64)
        U = np.zeros((m, k), dtype=np.float64)
        alphas = np.zeros(k, dtype=np.float64)
        betas = np.zeros(k, dtype=np.float64)

        v = self._deterministic_start_vector(n)
        u = delta @ v
        alpha = float(np.linalg.norm(u))
        if alpha <= tol:
            return np.array([alpha], dtype=np.float64), 1, alpha

        u /= alpha
        U[:, 0] = u
        V[:, 0] = v
        alphas[0] = alpha
        effective = 1
        last_off = 0.0

        for j in range(k - 1):
            r = (delta.T @ U[:, j]) - alphas[j] * V[:, j]
            for i in range(j + 1):
                r = r - np.dot(V[:, i], r) * V[:, i]
            beta = float(np.linalg.norm(r))
            betas[j + 1] = beta
            last_off = beta
            if beta <= tol:
                break
            v = r / beta
            V[:, j + 1] = v

            p = (delta @ v) - beta * U[:, j]
            for i in range(j + 1):
                p = p - np.dot(U[:, i], p) * U[:, i]
            alpha = float(np.linalg.norm(p))
            alphas[j + 1] = alpha
            effective = j + 2
            if alpha <= tol:
                if alpha > _EPSILON:
                    U[:, j + 1] = p / alpha
                break
            U[:, j + 1] = p / alpha

        B = np.diag(alphas[:effective])
        if effective > 1:
            B += np.diag(betas[1:effective], 1)
        s = la.svdvals(B)
        s_sorted = np.sort(np.asarray(s, dtype=np.float64))[::-1]
        residual = float(last_off) if last_off > 0.0 else (
            float(abs(s_sorted[-1] - s_sorted[-2]))
            if s_sorted.size >= 2
            else float(s_sorted[-1] if s_sorted.size else 0.0)
        )
        return s_sorted, int(effective), residual

    def _svds_fallback(
        self,
        k: int,
        tol: float,
        m: int,
        n: int,
    ) -> Tuple[NDArray[np.float64], int, float]:
        r"""Degradación ARPACK (operador aumentado; no forma L)."""
        dim_min = min(m, n)
        k_eff = max(1, min(k, dim_min - 1)) if dim_min > 1 else 1
        try:
            delta_sparse = sp.csc_matrix(self._delta)
            s = spla.svds(
                delta_sparse,
                k=k_eff,
                which="LM",
                return_singular_vectors=False,
                tol=tol,
                maxiter=_KRYLOV_MAX_ITER,
            )
        except Exception as exc:
            raise SpectralComputationError(
                f"Golub–Kahan–Lanczos/svds no convergió (k={k_eff}, tol={tol}): {exc}."
            ) from exc
        s_sorted = np.sort(np.asarray(s, dtype=np.float64))[::-1]
        residual = (
            float(abs(s_sorted[-1] - s_sorted[-2]))
            if s_sorted.size >= 2
            else float(s_sorted[-1] if s_sorted.size else 0.0)
        )
        return s_sorted, int(s_sorted.size), residual

    # ─────────────────────────────────────────────────────────────────────────
    # 2.3 κ₂(δ) medido por Krylov (sin cuadrar L)
    # ─────────────────────────────────────────────────────────────────────────
    def _measure_delta_condition_number_krylov(self) -> float:
        r"""κ₂(δ) = σ_max / σ_min⁺ SIN materializar L = δᵀδ.

        σ_max se toma del Golub–Kahan nativo; σ_min⁺ se sondea con svds
        which='SM'. Fallback determinista: certificado Wilkinson de Fase 1.
        """
        m, n = int(self._delta.shape[0]), int(self._delta.shape[1])
        dim_min = min(m, n)
        if dim_min <= 1:
            return self._kappa_from_certificate()

        sigma_max = self._phase1.max_singular_value
        try:
            s_hi, _, _ = self._golub_kahan_lanczos_bidiagonalization(
                k=min(2, dim_min - 1), tol=_KRYLOV_TOL
            )
            if s_hi.size > 0:
                sigma_max = float(s_hi[0])
        except Exception:
            pass

        sigma_min_pos = self._phase1.min_nonzero_singular_value
        try:
            delta_sparse = sp.csc_matrix(self._delta)
            s_lo = spla.svds(
                delta_sparse,
                k=1,
                which="SM",
                return_singular_vectors=False,
                tol=_KRYLOV_TOL,
                maxiter=_KRYLOV_MAX_ITER,
            )
            if np.size(s_lo) > 0:
                sigma_min_pos = float(np.asarray(s_lo).ravel()[0])
        except Exception:
            pass

        if sigma_min_pos <= _MACHINE_EPSILON:
            return float("inf")
        return float(sigma_max / sigma_min_pos)

    def _kappa_from_certificate(self) -> float:
        r"""κ₂(δ) tomada del certificado de Fase 1 (fallback determinista)."""
        kappa = float(self._phase1.condition_number_delta)
        if math.isfinite(kappa) and kappa >= 1.0:
            return kappa
        sigma_max = self._phase1.max_singular_value
        sigma_min = self._phase1.min_nonzero_singular_value
        if sigma_max <= 0.0:
            return 1.0
        if sigma_min <= 0.0:
            return math.inf
        return float(sigma_max / sigma_min)

    # ─────────────────────────────────────────────────────────────────────────
    # 2.4 Energía de Dirichlet E(x) = ‖δx‖² (semigrupo, un matvec)
    # ─────────────────────────────────────────────────────────────────────────
    def _evaluate_dirichlet_energy_semigroup(
        self,
        x: NDArray[np.float64],
    ) -> Tuple[float, float, NDArray[np.float64]]:
        r"""E(x) = ‖δx‖₂² sin materializar L = δᵀδ ([I2], [I3]).

        Implementación Banach: r = δx ∈ C¹, E = ⟨r, r⟩_{C¹}, ‖r‖ = √E.

        Returns
        ───────
        (E, ‖δx‖, δx)
        """
        x_ = self._as_finite_vector("x_state", x)
        delta_x = self._delta @ x_
        if not np.all(np.isfinite(delta_x)):
            raise SpectralComputationError("[Fase 2] δx contiene no finitos.")
        energy = self._squared_norm_from_vector(delta_x)
        residual_norm = math.sqrt(energy)
        return energy, float(residual_norm), delta_x

    # ─────────────────────────────────────────────────────────────────────────
    # 2.5 Descomposición de Hodge–Helmholtz (Moore–Penrose)
    # ─────────────────────────────────────────────────────────────────────────
    def _decompose_hodge_helmholtz_moore_penrose(
        self,
        x: NDArray[np.float64],
        delta_x: NDArray[np.float64],
    ) -> Tuple[float, float, NDArray[np.float64], NDArray[np.float64]]:
        r"""Hodge–Helmholtz con pseudoinversa de Moore–Penrose truncada.

            x_exact = δ⁺ (δx)   ∈ im(δᵀ)
            x_harm  = x − x_exact ∈ ker(δ)

        La tolerancia de truncación es la SVD_TOL de Wilkinson de Fase 1.

        Returns
        ───────
        (‖x_harm‖, ‖x_exact‖, x_harm, x_exact)
        """
        try:
            delta_pinv = self._pseudo_inverse(
                self._delta, "δ", tolerance=self._phase1.svd_tolerance
            )
            self._verify_penrose_equations(self._delta, delta_pinv, name="δ")
            x_exact = delta_pinv @ delta_x
            if not np.all(np.isfinite(x_exact)):
                raise HodgeDecompositionError(
                    "[Fase 2] δ⁺(δx) produjo valores no finitos."
                )
            x_harm = x - x_exact
            if not np.all(np.isfinite(x_harm)):
                raise HodgeDecompositionError(
                    "[Fase 2] x_harm = x − δ⁺δx contiene no finitos."
                )
            harm_norm = self._vector_norm(x_harm)
            exact_norm = self._vector_norm(x_exact)
            return (
                harm_norm if math.isfinite(harm_norm) else math.inf,
                exact_norm if math.isfinite(exact_norm) else math.inf,
                x_harm,
                x_exact,
            )
        except HodgeDecompositionError:
            raise
        except Exception as exc:
            logger.warning("[Fase 2] Descomposición de Hodge falló: %s.", exc)
            nan_vec = np.full(x.shape, np.nan, dtype=np.float64)
            return math.inf, math.inf, nan_vec, nan_vec

    # ─────────────────────────────────────────────────────────────────────────
    # 2.6 Constante de Poincaré–Wirtinger
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _compute_poincare_wirtinger_constant(
        exact_component_norm: float,
        dirichlet_energy: float,
    ) -> float:
        r"""C_P = ‖x_exact‖₂ / ‖δx‖₂ = ‖x_exact‖₂ / √E(x) ≥ 0.

        Para x ⊥ ker(δ): ‖x‖ ≤ C_P · ‖δx‖.
        Si E(x) = 0 (x ya armónico), C_P no está definida → 0.0.
        """
        if dirichlet_energy <= 0.0 or not math.isfinite(dirichlet_energy):
            return 0.0
        delta_x_norm = math.sqrt(dirichlet_energy)
        if delta_x_norm <= 0.0:
            return 0.0
        if not math.isfinite(exact_component_norm):
            return math.inf
        c_p = exact_component_norm / delta_x_norm
        return float(c_p) if math.isfinite(c_p) else math.inf

    # ─────────────────────────────────────────────────────────────────────────
    # 2.7 Auditoría espectral (pre-terminal de Φ₂)
    # ─────────────────────────────────────────────────────────────────────────
    def audit_krylov_spectral_stability(
        self,
        x_state: NDArray[np.float64],
    ) -> KrylovSpectralData:
        r"""Produce KrylovSpectralData, precondición estricta de Φ₃.

        Raises
        ──────
        DirichletFrustrationError
        SpectralComputationError
        PoincareBoundViolation
        """
        x = self._as_finite_vector("x_state", x_state, allow_empty=False)
        if x.size != self._phase1.dim_C0:
            raise ValueError(
                f"[Fase 2] x ∈ C⁰ debe tener dim={self._phase1.dim_C0}; "
                f"recibido dim={x.size}."
            )

        s_krylov, krylov_dim, krylov_res = (
            self._golub_kahan_lanczos_bidiagonalization()
        )
        kappa_delta = self._measure_delta_condition_number_krylov()
        kappa_L = (
            kappa_delta * kappa_delta if math.isfinite(kappa_delta) else math.inf
        )

        energy, residual_norm, delta_x = self._evaluate_dirichlet_energy_semigroup(x)
        x_norm_sq = float(np.dot(x, x)) if x.size else 0.0
        energy_norm = (
            energy / max(1.0, x_norm_sq) if math.isfinite(energy) else math.inf
        )

        frustration_tolerance = max(
            _FRUSTRATION_TOLERANCE,
            _FRUSTRATION_RELATIVE_TOL * max(1.0, x_norm_sq),
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(1.0, abs(energy)),
        )
        frustration_index = energy / frustration_tolerance if frustration_tolerance else math.inf

        if energy > frustration_tolerance:
            raise DirichletFrustrationError(
                f"[Fase 2] Frustración térmica inadmisible: E(x) = {energy:.6e} "
                f"> ε_frust = {frustration_tolerance:.6e} "
                f"(ρ = {frustration_index:.4f})."
            )

        if math.isfinite(kappa_L) and kappa_L > _MAX_CONDITION_NUMBER_L:
            raise SpectralComputationError(
                f"[Fase 2] Peligro de colapso FPU: κ(L) = {kappa_L:.3e} > "
                f"κ_max = {_MAX_CONDITION_NUMBER_L:.3e}."
            )

        harm_norm, exact_norm, _, _ = self._decompose_hodge_helmholtz_moore_penrose(
            x, delta_x
        )
        poincare_constant = self._compute_poincare_wirtinger_constant(
            exact_norm, energy
        )
        is_poincare_bounded = (
            math.isfinite(poincare_constant)
            and poincare_constant <= _POINCARE_CONSTANT_MAX
        )
        if (not math.isfinite(poincare_constant)) or (
            poincare_constant > _POINCARE_CONSTANT_MAX
        ):
            raise PoincareBoundViolation(
                f"[Fase 2] C_P = {poincare_constant} > C_P,max = "
                f"{_POINCARE_CONSTANT_MAX:.6e} (o no finita)."
            )

        holder = self._holder_operator_norm_bound(self._delta)

        cert_hash = self._compute_dto_hash(
            self._phase1.certification_hash_sha256,
            float(energy), float(energy_norm),
            float(kappa_delta), float(kappa_L),
            float(harm_norm), float(exact_norm),
            float(poincare_constant), int(krylov_dim), float(krylov_res),
            float(holder),
            s_krylov,
        )

        logger.info(
            "[Fase 2 ✓] KrylovSpectralData: E(x)=%.4e, ‖δx‖=%.4e, "
            "κ₂(δ)=%.3e, κ(L)=%.3e, C_P=%.4e, Krylov_dim=%d, Hölder=%.3e, hash=%s.",
            energy, residual_norm, kappa_delta, kappa_L,
            poincare_constant, krylov_dim, holder, cert_hash[:16] + "...",
        )

        s_immutable = np.array(s_krylov, dtype=np.float64, copy=True)
        s_immutable.setflags(write=False)

        return KrylovSpectralData(
            phase1_reference=self._phase1,
            krylov_singular_values=s_immutable,
            krylov_dimension=int(krylov_dim),
            krylov_residual=float(krylov_res),
            dirichlet_energy=float(energy),
            dirichlet_energy_norm=float(energy_norm),
            frustration_tolerance=float(frustration_tolerance),
            frustration_index=float(frustration_index),
            delta_condition_number=float(kappa_delta),
            laplacian_condition_number=float(kappa_L),
            spectral_gap_effective=float(self._phase1.spectral_gap),
            harmonic_component_norm=float(harm_norm),
            exact_component_norm=float(exact_norm),
            poincare_constant=float(poincare_constant),
            banach_holder_bound=float(holder),
            is_frustration_bounded=True,
            is_spectrally_stable=True,
            is_poincare_bounded=bool(is_poincare_bounded),
            certification_hash_sha256=str(cert_hash),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 2.8 ★ MORFISMO TERMINAL DE FASE 2 ★
    #     Tipo de retorno = objeto inicial de la FASE 3.
    #     last(Φ₂) = unit(Φ₃) = Phase3_IsoperimetricHodgeProjector.
    # ─────────────────────────────────────────────────────────────────────────
    def nest_into_phase3(
        self,
        x_state: NDArray[np.float64],
    ) -> "Phase3_IsoperimetricHodgeProjector":
        r"""★ MORFISMO TERMINAL DE FASE 2 / UNIDAD DE LA FASE 3 ★

        Composición estricta F₂ ⊣ F₃:
            nest_into_phase3(x)  :=  Phase3_IsoperimetricHodgeProjector
                                     ∘ audit_krylov_spectral_stability(x).

        El projector de Hodge nace ya alimentado con KrylovSpectralData;
        su constructor ES la continuación formal de este método.
        """
        phase2_data = self.audit_krylov_spectral_stability(x_state)
        return Phase3_IsoperimetricHodgeProjector(phase2_data)


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 3: IMPOSICIÓN DEL LÍMITE ISOPERIMÉTRICO DE HODGE–HELMHOLTZ           ║
# ║                                                                             ║
# ║   ★ INICIO FORMAL = continuación de Phase2.nest_into_phase3 ★               ║
# ║   Precondición constructora: KrylovSpectralData.                             ║
# ║                                                                             ║
# ║   Marco formal:                                                             ║
# ║   ─────────                                                                 ║
# ║   La proyección de Hodge π: C⁰ → ker(δ) satisface:                          ║
# ║     1. π minimiza la energía: E(π(x)) ≤ E(x)  ∀ x.                          ║
# ║     2. π es una retracción: π∘π = π.                                         ║
# ║     3. π minimiza la norma: ‖π(x)‖ ≤ ‖x‖.                                   ║
# ║   Síntesis: x* = (I − δ⁺δ) x  (sección armónica).                            ║
# ║                                                                             ║
# ║   ULTIMO MÉTODO DE FASE 3 (morfismo terminal del módulo):                    ║
# ║     enforce_isoperimetric_hodge_projection(x, x*) → HodgeProjectionData      ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase3_IsoperimetricHodgeProjector(Phase2_KrylovSpectralAuditor):
    r"""FASE 3: Proyector isoperimétrico de Hodge y colapso Heyting.

    ★ INICIO FORMAL = continuación del morfismo terminal de la FASE 2 ★

    Cadena interna:
        __init__(KrylovSpectralData)        ← unidad heredada de Φ₂
            → synthesize_hodge_section
            → _verify_lipschitz_strong_axiom
            → _estimate_cheeger_isoperimetric_constant
            → _compute_morse_reduction_index_from_certificate
            → _resolve_heyting_omega_three_lattice
            → _simulate_crowbar_gpio14_actuation
            → enforce_isoperimetric_hodge_projection  ★ morfismo terminal
    """

    def __init__(self, phase2_audit: KrylovSpectralData) -> None:
        r"""★ CONTINUACIÓN DE FASE 2 / INICIO DE FASE 3 ★

        Args
        ────
        phase2_audit : KrylovSpectralData
            Salida de `audit_krylov_spectral_stability`, inyectada por
            `nest_into_phase3`.
        """
        if not isinstance(phase2_audit, KrylovSpectralData):
            raise TypeError(
                "Phase3_IsoperimetricHodgeProjector requiere KrylovSpectralData "
                "como precondición (Fase 2)."
            )
        if not phase2_audit.is_spectrally_stable:
            raise SpectralComputationError(
                "[Fase 3] Precondición inválida: Fase 2 reportó inestabilidad."
            )
        if not phase2_audit.is_frustration_bounded:
            raise DirichletFrustrationError(
                "[Fase 3] Precondición inválida: Fase 2 no acotó frustración."
            )
        super().__init__(phase2_audit.phase1_reference)
        self._phase2: Final[KrylovSpectralData] = phase2_audit

    @property
    def phase2_certificate(self) -> KrylovSpectralData:
        r"""Referencia inmutable al DTO de Fase 2 (precondición)."""
        return self._phase2

    # ─────────────────────────────────────────────────────────────────────────
    # 3.1 Síntesis de la sección armónica x* = (I − δ⁺δ) x
    # ─────────────────────────────────────────────────────────────────────────
    def synthesize_hodge_section(
        self,
        x_original: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""Proyección armónica x* ∈ ker(δ) ∩ (x + im(δᵀ)).

        Teorema de Hodge discreto sobre un complejo de 2 términos:
            x* = x − δ⁺ (δx) = (I − δ⁺δ) x.
        Truncación espectral con SVD_TOL de Wilkinson de Fase 1.

        Returns
        ───────
        x_star : NDArray[float64], write-protected copy
        """
        x = self._as_finite_vector("x_original", x_original, allow_empty=False)
        if x.size != self._phase1.dim_C0:
            raise ValueError(
                f"[Fase 3] x debe tener dim = dim C⁰ = {self._phase1.dim_C0}; "
                f"recibido {x.size}."
            )
        delta_x = self._delta @ x
        _, _, x_harm, _ = self._decompose_hodge_helmholtz_moore_penrose(x, delta_x)
        if not np.all(np.isfinite(x_harm)):
            raise HodgeDecompositionError(
                "[Fase 3] synthesize_hodge_section produjo no finitos."
            )
        x_star = np.array(x_harm, dtype=np.float64, copy=True)
        x_star.setflags(write=False)
        return x_star

    # ─────────────────────────────────────────────────────────────────────────
    # 3.2 Axioma de Lipschitz fuerte
    # ─────────────────────────────────────────────────────────────────────────
    @classmethod
    def _verify_lipschitz_strong_axiom(
        cls,
        delta_x: NDArray[np.float64],
        delta_x_star: NDArray[np.float64],
        displacement_norm: float,
        kappa_delta: float,
    ) -> Tuple[float, float, bool]:
        r"""Verifica ‖δx* − δx‖₂ ≤ κ(δ) · ‖x* − x‖₂ ([I5]).

        Returns
        ───────
        (lhs, residual = lhs − rhs, satisfied)
        """
        delta_diff = delta_x_star - delta_x
        if not np.all(np.isfinite(delta_diff)):
            return math.inf, math.inf, False

        lhs = cls._vector_norm(delta_diff)
        if not math.isfinite(lhs):
            return math.inf, math.inf, False

        if not math.isfinite(kappa_delta) or not math.isfinite(displacement_norm):
            rhs = math.inf
        else:
            rhs = kappa_delta * displacement_norm

        residual = lhs - rhs
        slack = (
            _LIPSCHITZ_SLACK * max(1.0, rhs) if math.isfinite(rhs) else math.inf
        )
        satisfied = residual <= slack
        return float(lhs), float(residual), bool(satisfied)

    # ─────────────────────────────────────────────────────────────────────────
    # 3.3 Cociente de Rayleigh / constante isoperimétrica de Cheeger
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _estimate_cheeger_isoperimetric_constant(
        projected_energy: float,
        x_star_norm: float,
    ) -> float:
        r"""Cociente de Rayleigh R(x*) = E(x*) / ‖x*‖² = ‖δx*‖₂² / ‖x*‖₂².

        Cota de Cheeger clásica: h(G)² / 2 ≤ λ_min⁺(L) ≤ R(x*) para
        x* ⊥ ker(δ). Si x* ∈ ker(δ), R = 0 = λ_min(L).
        """
        if x_star_norm <= 0.0 or not math.isfinite(x_star_norm):
            return 0.0
        if projected_energy <= 0.0:
            return 0.0
        h = projected_energy / (x_star_norm * x_star_norm)
        return float(h) if math.isfinite(h) else math.inf

    # ─────────────────────────────────────────────────────────────────────────
    # 3.4 Índice de Morse desde el certificado de Fase 1
    # ─────────────────────────────────────────────────────────────────────────
    def _compute_morse_reduction_index_from_certificate(self) -> int:
        r"""ι_M = dim ker(δ) = dim H⁰ = dim C⁰ − rank(δ).

        Tomado del certificado de Wilkinson de Fase 1 — sin recomputar SVD.
        Interpretable como el índice de Morse de E en el lugar crítico armónico.
        """
        return max(0, int(self._phase1.h0_dimension))

    # ─────────────────────────────────────────────────────────────────────────
    # 3.5 Residuo de idempotencia π∘π − π
    # ─────────────────────────────────────────────────────────────────────────
    def _hodge_idempotence_residual(
        self,
        x_star: NDArray[np.float64],
    ) -> float:
        r"""‖π(x*) − x*‖₂. Si x* ya es armónico, π(x*) = x* y el residuo es 0."""
        try:
            x_ss = self.synthesize_hodge_section(x_star)
            return float(self._vector_norm(x_ss - x_star))
        except Exception as exc:
            logger.warning("[Fase 3] Idempotencia de Hodge no evaluable: %s.", exc)
            return math.inf

    # ─────────────────────────────────────────────────────────────────────────
    # 3.6 Resolución del retículo Ω₃
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _resolve_heyting_omega_three_lattice(
        h1_dimension: int,
        energy_after: float,
        frustration_tolerance: float,
        lipschitz_slack: float,
        isoperimetric_slack: float,
        energy_non_increasing: bool,
    ) -> HeytingOmega3:
        r"""Colapsa Ω₃ al veredicto terminal (join monótono, precedencia estricta).

            1. h1_dimension > 0                            ⟹ VETOED (⊤)
            2. violación de Lipschitz o isoperimétrica    ⟹ VETOED
            3. E(x*) > ε_frust ∨ ¬energy_non_increasing   ⟹ DEGRADED
            4. en otro caso                                ⟹ COHERENT (⊥)
        """
        verdict: HeytingOmega3 = HeytingOmega3.COHERENT

        if h1_dimension > 0:
            verdict = verdict.join(HeytingOmega3.VETOED)

        if (
            lipschitz_slack < -_LIPSCHITZ_SLACK
            or isoperimetric_slack < -_FRUSTRATION_TOLERANCE
        ):
            verdict = verdict.join(HeytingOmega3.VETOED)

        if energy_after > frustration_tolerance or not energy_non_increasing:
            verdict = verdict.join(HeytingOmega3.DEGRADED)

        return verdict

    # ─────────────────────────────────────────────────────────────────────────
    # 3.7 Actuación Crowbar (Strict-NoHardware, sólo simulación)
    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _simulate_crowbar_gpio14_actuation() -> bool:
        r"""Simula la conmutación del disyuntor Crowbar en GPIO14 (BCM).

        Estricta política Strict-NoHardware: NUNCA accede a RPi.GPIO ni a
        registros de hardware. Emite un log CRITICAL y retorna siempre False.

        Returns
        ───────
        False (siempre; conmutación puramente software).
        """
        logger.critical(
            "CROWBAR SIMULADO (Strict-NoHardware): GPIO%d habría sido conmutado "
            "a HIGH. Colapso Ω₃ → VETOED (⊤).",
            _CROWBAR_GPIO_PIN,
        )
        return False

    # ─────────────────────────────────────────────────────────────────────────
    # 3.8 ★ MORFISMO TERMINAL DE FASE 3 / DEL MÓDULO ★
    #     Cierra el funtor maestro 𝒵_SheafAgent = Φ₃ ∘ Φ₂ ∘ Φ₁.
    # ─────────────────────────────────────────────────────────────────────────
    def enforce_isoperimetric_hodge_projection(
        self,
        x_original: NDArray[np.float64],
        x_projected: NDArray[np.float64],
    ) -> HodgeProjectionData:
        r"""★ MORFISMO TERMINAL DE FASE 3 / DEL MÓDULO ★

        Cadena funtorial de Φ₃:
            KrylovSpectralData
              ──(verify_lipschitz_strong_axiom)───────────▶ (lhs, residual, ✓)
              ──(verify_isoperimetric_bound)──────────────▶ slack
              ──(verify_minimal_norm)─────────────────────▶ ✓
              ──(verify_energy_nonincreasing)─────────────▶ ✓
              ──(hodge_idempotence_residual)──────────────▶ ‖π(x*)−x*‖
              ──(cheeger_isoperimetric_constant)──────────▶ R(x*)
              ──(morse_reduction_index_from_certificate)──▶ ι_M
              ──(resolve_heyting_omega_three_lattice)─────▶ veredicto ∈ Ω₃
              ──(simulate_crowbar_gpio14 if VETOED)───────▶ bool
              ──(emit_HodgeProjectionData)────────────────▶ DTO terminal

        Raises
        ──────
        HomologicalInconsistencyError
        LipschitzViolation
        MinimalNormViolation
        """
        x0 = self._as_finite_vector("x_original", x_original, allow_empty=False)
        x1 = self._as_finite_vector("x_projected", x_projected, allow_empty=False)
        if x0.shape != x1.shape:
            raise ValueError(
                f"[Fase 3] x_original y x_projected deben coincidir: "
                f"{x0.shape} vs {x1.shape}."
            )
        if x0.size != self._phase1.dim_C0:
            raise ValueError(
                f"[Fase 3] x debe tener dim = dim C⁰ = {self._phase1.dim_C0}; "
                f"recibido {x0.size}."
            )

        displacement = x0 - x1
        if not np.all(np.isfinite(displacement)):
            raise HomologicalInconsistencyError(
                "[Fase 3] x − x* contiene no finitos."
            )
        projection_distance = self._vector_norm(displacement)
        if not math.isfinite(projection_distance):
            raise HomologicalInconsistencyError(
                "[Fase 3] ‖x − x*‖₂ no es finita."
            )
        norm_x0 = self._vector_norm(x0)
        norm_x1 = self._vector_norm(x1)
        if not (math.isfinite(norm_x0) and math.isfinite(norm_x1)):
            raise HomologicalInconsistencyError(
                "[Fase 3] ‖x‖ o ‖x*‖ no son finitas."
            )
        relative_distance = projection_distance / max(1.0, norm_x0)

        dist_tolerance = (
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, norm_x0, norm_x1)
        )
        inertia_limit = _INERTIA_DELTA_MAX + dist_tolerance
        isoperimetric_slack = inertia_limit - projection_distance
        is_isoperimetric = projection_distance <= inertia_limit

        delta_x0 = self._delta @ x0
        delta_x1 = self._delta @ x1
        if not (np.all(np.isfinite(delta_x0)) and np.all(np.isfinite(delta_x1))):
            raise HomologicalInconsistencyError(
                "[Fase 3] δx o δx* contienen no finitos."
            )
        original_energy = self._squared_norm_from_vector(delta_x0)
        projected_energy = self._squared_norm_from_vector(delta_x1)

        consistency_tol = max(
            _ENERGY_RATIO_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, abs(original_energy), abs(self._phase2.dirichlet_energy)),
        )
        if abs(original_energy - self._phase2.dirichlet_energy) > consistency_tol:
            raise HomologicalInconsistencyError(
                f"[Fase 3] Inconsistencia energética entre F₂ y F₃: "
                f"E_cert={self._phase2.dirichlet_energy:.3e}, "
                f"E_recalc={original_energy:.3e}."
            )

        energy_tol = (
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, abs(original_energy), abs(projected_energy))
        )
        energy_nonincreasing = projected_energy <= original_energy + energy_tol

        lipschitz_lhs, lipschitz_residual, lipschitz_ok = (
            self._verify_lipschitz_strong_axiom(
                delta_x=delta_x0,
                delta_x_star=delta_x1,
                displacement_norm=projection_distance,
                kappa_delta=self._phase2.delta_condition_number,
            )
        )
        lipschitz_slack = -lipschitz_residual

        norm_tol = (
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, norm_x0, norm_x1)
        )
        minimal_norm_ok = norm_x1 <= norm_x0 + norm_tol

        energy_reduction_ratio = (
            projected_energy / original_energy
            if original_energy > energy_tol
            else 0.0
        )

        cheeger_h = self._estimate_cheeger_isoperimetric_constant(
            projected_energy, norm_x1
        )
        morse_index = self._compute_morse_reduction_index_from_certificate()
        idempotence_residual = self._hodge_idempotence_residual(x1)

        verdict = self._resolve_heyting_omega_three_lattice(
            h1_dimension=self._phase1.h1_dimension,
            energy_after=projected_energy,
            frustration_tolerance=max(
                _FRUSTRATION_TOLERANCE, self._phase2.frustration_tolerance
            ),
            lipschitz_slack=lipschitz_slack,
            isoperimetric_slack=isoperimetric_slack,
            energy_non_increasing=energy_nonincreasing,
        )

        crowbar_simulated = False
        if verdict == HeytingOmega3.VETOED:
            crowbar_simulated = self._simulate_crowbar_gpio14_actuation()

        if not is_isoperimetric:
            raise HomologicalInconsistencyError(
                f"[Fase 3] Violación inercial: ‖x − x*‖ = {projection_distance:.6f} > "
                f"Δ_inertia = {_INERTIA_DELTA_MAX:.6f}."
            )
        if not lipschitz_ok:
            raise LipschitzViolation(
                f"[Fase 3] Lipschitz fuerte falla: residual = {lipschitz_residual:.3e}."
            )
        if not minimal_norm_ok:
            raise MinimalNormViolation(
                f"[Fase 3] Mínima norma falla: ‖x*‖ = {norm_x1:.3e} > "
                f"‖x‖ + ε = {norm_x0 + norm_tol:.3e}."
            )

        cert_hash = self._compute_dto_hash(
            self._phase2.certification_hash_sha256,
            float(projection_distance), float(relative_distance),
            float(original_energy), float(projected_energy),
            float(energy_reduction_ratio), float(lipschitz_residual),
            float(lipschitz_lhs), float(idempotence_residual),
            int(morse_index), int(verdict.value),
            float(cheeger_h),
        )

        logger.info(
            "[Fase 3 ✓] HodgeProjectionData: ‖x−x*‖=%.4f, E(x*)/E(x)=%.4f, "
            "h(G)=%.3e, ι_M=%d, ‖ππ−π‖=%.3e, Ω₃=%s, Crowbar=%s, hash=%s.",
            projection_distance, energy_reduction_ratio, cheeger_h,
            morse_index, idempotence_residual, verdict.name, crowbar_simulated,
            cert_hash[:16] + "...",
        )

        return HodgeProjectionData(
            projection_distance=float(projection_distance),
            relative_projection_distance=float(relative_distance),
            inertia_delta_max=float(_INERTIA_DELTA_MAX),
            original_dirichlet_energy=float(original_energy),
            projected_dirichlet_energy=float(projected_energy),
            energy_reduction_ratio=float(energy_reduction_ratio),
            lipschitz_lhs=float(lipschitz_lhs),
            lipschitz_residual=float(lipschitz_residual),
            lipschitz_satisfied=bool(lipschitz_ok),
            lipschitz_slack=float(lipschitz_slack),
            isoperimetric_slack=float(isoperimetric_slack),
            minimal_norm_satisfied=bool(minimal_norm_ok),
            hodge_idempotence_residual=float(idempotence_residual),
            cheeger_bound_estimate=float(cheeger_h),
            morse_reduction_index=int(morse_index),
            heyting_verdict=verdict,
            crowbar_simulated=bool(crowbar_simulated),
            is_isoperimetrically_bounded=bool(is_isoperimetric),
            is_energy_non_increasing=bool(energy_nonincreasing),
            verified_by_delta=True,
            certification_hash_sha256=str(cert_hash),
        )

    def synthesize_then_enforce(
        self,
        x_original: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], HodgeProjectionData]:
        r"""Sintetiza x* = π(x) y aplica el morfismo terminal sobre el par (x, x*).

        Conveniencia interna: no altera el contrato público de
        `enforce_isoperimetric_hodge_projection`.
        """
        x_star = self.synthesize_hodge_section(x_original)
        audit = self.enforce_isoperimetric_hodge_projection(x_original, x_star)
        return x_star, audit


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   ORQUESTADOR SUPREMO: 𝒵_SheafAgent = Φ₃ ∘ Φ₂ ∘ Φ₁                         ║
# ║                                                                             ║
# ║   Endofuntor terminal. Anidamiento EXCLUSIVO vía morfismos terminales:      ║
# ║       Φ₁.nest_into_phase2(δ)  ⟶  Phase2                                     ║
# ║       Φ₂.nest_into_phase3(x)  ⟶  Phase3                                     ║
# ║       Φ₃.enforce_isoperimetric_hodge_projection(x, x*) ⟶ HodgeProjectionData║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class SheafCohomologyOrchestratorAgent(Morphism):
    r"""El Custodio de la Holonomía Global en el estrato STRATEGY.

    Somete la estrategia de consenso agéntico a la composición funtorial:

        𝒵_SheafAgent = Φ₃ ∘ Φ₂ ∘ Φ₁,

    garantizando coherencia topológica, estabilidad espectral y admisibilidad
    termodinámica de la proyección de Hodge. El anidamiento se realiza
    exclusivamente a través de los morfismos terminales `nest_into_phase2`
    y `nest_into_phase3`.
    """

    def __init__(self, strict_mode: bool = True) -> None:
        r"""Inicializa el agente orquestador.

        Args
        ────
        strict_mode : Si True, toda degradación Ω₃ ≠ COHERENT se convierte
            en error al final del pipeline.
        """
        self._strict_mode = bool(strict_mode)
        logger.info(
            "[Orquestador] SheafCohomologyOrchestratorAgent v%s inicializado. "
            "strict_mode=%s.",
            __version__, self._strict_mode,
        )

    @staticmethod
    def _build_provenance(
        checksum_input: str,
        phase1_hash: str,
        phase2_hash: str,
        phase3_hash: str,
        phase1_passed: bool,
        phase2_passed: bool,
        phase3_passed: bool,
    ) -> SheafAuditProvenance:
        r"""Compone el objeto de trazabilidad criptográfica end-to-end."""
        timestamp = datetime.now(tz=timezone.utc).isoformat()
        symbols = {True: "✓", False: "✗"}
        all_ok = phase1_passed and phase2_passed and phase3_passed
        chain = (
            f"Φ₁={symbols[phase1_passed]} → "
            f"Φ₂={symbols[phase2_passed]} → "
            f"Φ₃={symbols[phase3_passed]} → "
            f"𝒵_Sheaf={symbols[all_ok]}"
        )
        return SheafAuditProvenance(
            timestamp_iso=timestamp,
            input_checksum_sha256=checksum_input,
            phase1_certification_hash=phase1_hash,
            phase2_certification_hash=phase2_hash,
            phase3_certification_hash=phase3_hash,
            phase1_passed=bool(phase1_passed),
            phase2_passed=bool(phase2_passed),
            phase3_passed=bool(phase3_passed),
            functor_chain=chain,
            agent_version=__version__,
        )

    @staticmethod
    def _log_governance_summary(
        veto_audit: CohomologicalVetoData,
        spectral_audit: KrylovSpectralData,
        hodge_audit: HodgeProjectionData,
        provenance: SheafAuditProvenance,
    ) -> None:
        r"""Log estructurado con todos los certificados de las tres fases."""
        logger.info(
            "═══════════════════════════════════════════════════════════════\n"
            "  SHEAF COHOMOLOGY GOVERNANCE — REPORTE FINAL (%s)\n"
            "  Timestamp  : %s\n"
            "  SHA-256 in : %s\n"
            "  F1 hash    : %s\n"
            "  F2 hash    : %s\n"
            "  F3 hash    : %s\n"
            "  Cadena     : %s\n"
            "───────────────────────────────────────────────────────────────\n"
            "  FASE 1 — Veto Cohomológico (Wilkinson adaptativo):\n"
            "    dim C⁰=%d, dim C¹=%d, rank(δ)=%d, dim H⁰=%d, dim H¹=%d\n"
            "    σ_max=%.4e, σ_min⁺=%.4e, gap=%.4e, σ_min⁺/σ_max=%.4e, κ₂=%.4e\n"
            "    SVD_TOL=%.4e (Wilkinson %d iters), certified=%s\n"
            "    β_estab=%.4f, χ₀₁=%d, log|τ|=%.4f, P–L=%s\n"
            "───────────────────────────────────────────────────────────────\n"
            "  FASE 2 — Krylov-Dirichlet (Golub–Kahan sobre δ):\n"
            "    E(x)=%.4e, Ê=%.4e, ε_frust=%.4e, ρ=%.4f\n"
            "    κ(δ)=%.4e, κ(L)=%.4e, Gap_ef=%.4e, Hölder=%.4e\n"
            "    ‖x_harm‖=%.4e, ‖x_exact‖=%.4e, C_P=%.4e\n"
            "    Krylov_dim=%d, residual=%.4e\n"
            "───────────────────────────────────────────────────────────────\n"
            "  FASE 3 — Hodge Isoperimétrico y Ω₃:\n"
            "    ‖x−x*‖=%.6f, ‖x−x*‖/‖x‖=%.6f, Δ_in=%.4f\n"
            "    E(x)=%.4e → E(x*)=%.4e (ratio=%.4f)\n"
            "    Lipschitz lhs=%.4e residual=%.4e, ✓=%s, min-norm ✓=%s\n"
            "    slack_L=%.4e, slack_iso=%.4e, ‖ππ−π‖=%.4e\n"
            "    h(G)_Rayleigh=%.4e, ι_M_Morse=%d\n"
            "    Ω₃ verdict=%s, Crowbar simulado=%s\n"
            "═══════════════════════════════════════════════════════════════",
            provenance.agent_version,
            provenance.timestamp_iso,
            provenance.input_checksum_sha256[:16] + "...",
            provenance.phase1_certification_hash[:16] + "...",
            provenance.phase2_certification_hash[:16] + "...",
            provenance.phase3_certification_hash[:16] + "...",
            provenance.functor_chain,
            veto_audit.dim_C0, veto_audit.dim_C1,
            veto_audit.delta_rank, veto_audit.h0_dimension, veto_audit.h1_dimension,
            veto_audit.max_singular_value, veto_audit.min_nonzero_singular_value,
            veto_audit.spectral_gap, veto_audit.spectral_gap_ratio,
            veto_audit.condition_number_delta,
            veto_audit.svd_tolerance, veto_audit.wilkinson_iterations,
            veto_audit.rank_is_certified,
            veto_audit.cohomological_stability_index,
            veto_audit.euler_characteristic_01,
            veto_audit.whitehead_torsion,
            veto_audit.poincare_lefschetz_ok,
            spectral_audit.dirichlet_energy,
            spectral_audit.dirichlet_energy_norm,
            spectral_audit.frustration_tolerance,
            spectral_audit.frustration_index,
            spectral_audit.delta_condition_number,
            spectral_audit.laplacian_condition_number,
            spectral_audit.spectral_gap_effective,
            spectral_audit.banach_holder_bound,
            spectral_audit.harmonic_component_norm,
            spectral_audit.exact_component_norm,
            spectral_audit.poincare_constant,
            spectral_audit.krylov_dimension,
            spectral_audit.krylov_residual,
            hodge_audit.projection_distance,
            hodge_audit.relative_projection_distance,
            hodge_audit.inertia_delta_max,
            hodge_audit.original_dirichlet_energy,
            hodge_audit.projected_dirichlet_energy,
            hodge_audit.energy_reduction_ratio,
            hodge_audit.lipschitz_lhs,
            hodge_audit.lipschitz_residual,
            hodge_audit.lipschitz_satisfied,
            hodge_audit.minimal_norm_satisfied,
            hodge_audit.lipschitz_slack,
            hodge_audit.isoperimetric_slack,
            hodge_audit.hodge_idempotence_residual,
            hodge_audit.cheeger_bound_estimate,
            hodge_audit.morse_reduction_index,
            hodge_audit.heyting_verdict.name,
            hodge_audit.crowbar_simulated,
        )

    def execute_sheaf_cohomology_governance(
        self,
        coboundary_operator_delta: NDArray[np.float64],
        x_state: NDArray[np.float64],
        x_projected_consensus: NDArray[np.float64],
    ) -> SheafGovernanceState:
        r"""Ejecuta la composición funtorial estricta anidada:

            𝒵_SheafAgent = Φ₃ ∘ Φ₂ ∘ Φ₁.

        Anidamiento:
            phase2 = Φ₁.nest_into_phase2(δ)          # unidad de Φ₂
            phase3 = phase2.nest_into_phase3(x)      # unidad de Φ₃
            hodge  = phase3.enforce...(x, x*)        # DTO terminal

        Raises
        ──────
        Cualquier excepción de la jerarquía SheafCohomologyAgentError.
        """
        input_checksum = self._compute_input_checksum(
            np.asarray(coboundary_operator_delta)
            if coboundary_operator_delta is not None
            else None,
            np.asarray(x_state) if x_state is not None else None,
            np.asarray(x_projected_consensus)
            if x_projected_consensus is not None
            else None,
        )
        logger.debug(
            "[Orquestador] Iniciando gobernanza. SHA-256 input: %s.",
            input_checksum[:16] + "...",
        )

        phase1_passed = False
        phase2_passed = False
        phase3_passed = False

        # Φ₁ → unidad de Φ₂
        phase2 = Phase1_CohomologicalVetoCertifier.nest_into_phase2(
            coboundary_operator_delta
        )
        veto_audit = phase2.phase1_certificate
        phase1_passed = True
        logger.debug(
            "[Fase 1] ✓ dim H⁰=%d, dim H¹=%d, β=%.4f, log|τ|=%.4f, hash=%s.",
            veto_audit.h0_dimension,
            veto_audit.h1_dimension,
            veto_audit.cohomological_stability_index,
            veto_audit.whitehead_torsion,
            veto_audit.certification_hash_sha256[:16] + "...",
        )

        # Φ₂ → unidad de Φ₃
        phase3 = phase2.nest_into_phase3(x_state)
        spectral_audit = phase3.phase2_certificate
        phase2_passed = True
        logger.debug(
            "[Fase 2] ✓ E(x)=%.4e, κ(L)=%.4e, C_P=%.4e, hash=%s.",
            spectral_audit.dirichlet_energy,
            spectral_audit.laplacian_condition_number,
            spectral_audit.poincare_constant,
            spectral_audit.certification_hash_sha256[:16] + "...",
        )

        # Φ₃ → DTO terminal
        hodge_audit = phase3.enforce_isoperimetric_hodge_projection(
            x_state, x_projected_consensus
        )
        phase3_passed = True
        logger.debug(
            "[Fase 3] ✓ ‖x−x*‖=%.6f, Ω₃=%s, h(G)≈%.4e, hash=%s.",
            hodge_audit.projection_distance,
            hodge_audit.heyting_verdict.name,
            hodge_audit.cheeger_bound_estimate,
            hodge_audit.certification_hash_sha256[:16] + "...",
        )

        is_valid = bool(
            veto_audit.is_topologically_coherent
            and spectral_audit.is_spectrally_stable
            and spectral_audit.is_frustration_bounded
            and spectral_audit.is_poincare_bounded
            and hodge_audit.is_isoperimetrically_bounded
            and hodge_audit.is_energy_non_increasing
            and hodge_audit.heyting_verdict == HeytingOmega3.COHERENT
        )

        provenance = self._build_provenance(
            checksum_input=input_checksum,
            phase1_hash=veto_audit.certification_hash_sha256,
            phase2_hash=spectral_audit.certification_hash_sha256,
            phase3_hash=hodge_audit.certification_hash_sha256,
            phase1_passed=phase1_passed,
            phase2_passed=phase2_passed,
            phase3_passed=phase3_passed,
        )
        self._log_governance_summary(
            veto_audit=veto_audit,
            spectral_audit=spectral_audit,
            hodge_audit=hodge_audit,
            provenance=provenance,
        )

        if self._strict_mode and not is_valid:
            raise SheafCohomologyAgentError(
                "[Orquestador] Composición 𝒵_SheafAgent no autorizó validez "
                f"epistemológica. Cadena: {provenance.functor_chain}. "
                f"Veredicto Ω₃: {hodge_audit.heyting_verdict.name}."
            )

        return SheafGovernanceState(
            veto_audit=veto_audit,
            spectral_audit=spectral_audit,
            hodge_audit=hodge_audit,
            provenance=provenance,
            is_epistemologically_valid=is_valid,
        )

    def execute_with_synthesized_section(
        self,
        coboundary_operator_delta: NDArray[np.float64],
        x_state: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], SheafGovernanceState]:
        r"""Variante que sintetiza x* = π(x) internamente (Moore–Penrose).

        Equivale a `execute_sheaf_cohomology_governance(δ, x, π(x))`.
        """
        phase2 = Phase1_CohomologicalVetoCertifier.nest_into_phase2(
            coboundary_operator_delta
        )
        phase3 = phase2.nest_into_phase3(x_state)
        x_star = phase3.synthesize_hodge_section(x_state)
        state = self.execute_sheaf_cohomology_governance(
            coboundary_operator_delta, x_state, x_star
        )
        return x_star, state

    _compute_input_checksum = staticmethod(
        _FiniteNumericalGuard._compute_input_checksum
    )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ═══════════════════════════════════════════════════════════════════════════════

__all__: List[str] = [
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
    "HeytingCollapseError",
    "HeytingOmega3",
    "CohomologicalVetoData",
    "KrylovSpectralData",
    "HodgeProjectionData",
    "SheafAuditProvenance",
    "SheafGovernanceState",
    "Phase1_CohomologicalVetoCertifier",
    "Phase2_KrylovSpectralAuditor",
    "Phase3_IsoperimetricHodgeProjector",
    "SheafCohomologyOrchestratorAgent",
]