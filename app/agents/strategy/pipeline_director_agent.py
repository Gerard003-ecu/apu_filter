# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Pipeline Director Agent (Custodio de Causalidad y Flujo Funtorial)  ║
║ Ruta   : app/agents/strategy/pipeline_director_agent.py                      ║
║ Versión: 3.1.0-Poincare-Melnikov-RecursiveN3-Strict                          ║
╚══════════════════════════════════════════════════════════════════════════════╝

MARCO TEÓRICO RIGUROSO Y GOBERNANZA CAUSAL AGÉNTICA (NIVEL DOCTORADO):
────────────────────────────────────────────────────────────────────────────────

1. CERTIFICACIÓN ESPECTRAL Y FORMES REALES DE SCHUR (RSF)
   Sea \(A \in \mathbb{R}^{n \times n}\) el endomorfismo de adyacencia causal sobre el
   grafo del pipeline. La descomposición real de Schur (RSF) factoriza \(A = Q T Q^\top\),
   donde \(Q \in \mathrm{O}(n)\) es ortogonal y \(T\) es cuasi-triangular superior con
   bloques 1×1 y 2×2 en la diagonal.
   Un bloque 2×2 representa un par conjugado \(\lambda = \alpha \pm \mathrm{i}\omega\),
   con frecuencia elíptica:
       \(\omega = \sqrt{\max\left(0, |bc| - \left(\frac{a-d}{2}\right)^2\right)}\)
   La condición de aciclicidad y estricta nilpotencia exige que \(\mathrm{Spec}(A) = \{0\}\),
   lo cual equivale a \(\rho(A) = \max |\lambda_i| \le \delta_{\mathrm{BF}}\), siendo
   \(\delta_{\mathrm{BF}}\) la tolerancia adaptativa ajustada por el teorema de Bauer-Fike:
       \(\delta_{\mathrm{BF}} = \max\left(\varepsilon_{\mathrm{base}}, \, \kappa_2(Q) \cdot C \cdot \sqrt{n} \cdot \varepsilon_{\mathrm{mach}} \cdot \|A\|_F\right)\)

2. DINÁMICA DE POINCARÉ, SECCIONES TRANSVERSALES Y DIVERGENCIA DE MELNIKOV
   Sobre el subespacio nulo de \(A\), el flujo discreto \(\varphi_k(v) = A^k v\) define
   una sección transversal de Poincaré \(\Sigma = \{p \in \mathbb{R}^n : h(p) = 0\}\) con
   función de sección \(h(p) = \|p\|_2 - \varepsilon_{\mathrm{BF}}\). La transversalidad
   se satisface si:
       \(\left|\frac{\mathrm{d}h}{\mathrm{d}t}\right| = |\langle \nabla h(p), A p \rangle| > \varepsilon_{\mathrm{trans}} > 0\)
   La integral discreta de Melnikov evalúa la 2-forma sobre el plano dominante de Schur
   \(\gamma_k = \Pi_{\mathrm{dom}} A^k v_0\):
       \(M(t_0) = \sum_k \frac{\det\left[ f_0(\gamma_k) \;\vert\; f_1(\gamma_k, t_k + t_0) \right]}{1 + k}\)
   donde \(f_0\) representa el campo nilpotente no perturbado y \(f_1\) la perturbación
   de bloques 2×2. Ceros simples transversales con \(M'(t_0) \neq 0\) revelan herraduras
   de Smale y tangencias homoclínicas.

3. TEORÍA KAM, DIFUSIÓN DE ARNOLD Y ESTABILIDAD DE NEKHOROSHEV
   Para el espectro de frecuencias de bloques 2×2, la persistencia de toros KAM requiere
   la condición diofántica \(|\langle k, \omega \rangle| \ge \frac{\gamma}{|k|_1^\tau}\) y
   una perturbación acotada \(\varepsilon < \varepsilon_{\mathrm{KAM}}\). La fuga de
   resonancias entre bloques genera difusión de Arnold con cota de Nekhoroshev:
       \(T_{\mathrm{Nek}} \sim \exp\left( \frac{1}{2} \varepsilon^{-\frac{1}{2n}} \right)\)

4. AUDITORÍA HOMOLÓGICA DE MAYER-VIETORIS Y DIVERGENCIA DE JENSEN-SHANNON
   Para cubiertas causales \(A\) y \(B\), la secuencia exacta de Mayer-Vietoris en H₁:
       \(H_1(A \cap B) \xrightarrow{\phi} H_1(A) \oplus H_1(B) \xrightarrow{\psi} H_1(A \cup B) \xrightarrow{\partial} H_0(A \cap B)\)
   Dicta que el número de Betti \(\beta_1(A \cup B)\) debe coincidir con la predicción
   algebraica exacta \(\beta_1^{\mathrm{exp}} = \beta_1(A) + \beta_1(B) - \mathrm{rank}(\phi) + \mathrm{rank}(\partial)\).
   La discrepancia topológica se mide mediante la Divergencia de Jensen-Shannon sobre los
   4-símplex de Betti observados vs. predichos:
       \(D_{\mathrm{JS}}(P \parallel Q) = \frac{1}{2} D_{\mathrm{KL}}(P \parallel M) + \frac{1}{2} D_{\mathrm{KL}}(Q \parallel M), \quad M = \frac{1}{2}(P + Q)\)

5. VEREDICTO DE GOBERNANZA TERMINAL \(\Omega_3\)
   El agente aplica el pullback Z_Causal = \(\Phi_3 \circ \Phi_2 \circ \Phi_1\), agregando
   los tres certificados en el objeto terminal \(\Omega_3\). Si cualquier factor
   (nilpotencia, filtración monótona, \(\Delta\beta_1 = 0\), idempotencia N3) falla, se
   activa un veto fail-secure que bloquea la ejecución agéntica.

TRADUCCIÓN A IMPACTO DE NEGOCIO ("DOLOR Y DINERO"):
────────────────────────────────────────────────────────────────────────────────
· BLINDAJE DE ESTRATEGIA Y GOBERNANZA AGÉNTICA: Evita que agentes de IA tomen decisiones
  o ejecuten acciones automatizadas basadas en grafos causales corruptos o con ciclos
  infinitos, eliminando riesgos de responsabilidad legal y pérdidas financieras directas.
· CERTIFICACIÓN AUTOMÁTICA DE AUDITORÍAS: Garantiza que la consolidación de informes
  financieros y presupuestarios cumpla con los estándares rigurosos de no duplicación,
  reduciendo a cero los costos por multas o reprocesamientos en auditorías gubernamentales.
· OPERACIÓN AUTÓNOMA FAIL-SECURE: Ante cualquier anomalía matemática o degradación de
  datos, el agente aborta la transacción antes de corromper la base de datos central,
  salvaguardando la continuidad del negocio y el valor de los activos de información.
"""

from __future__ import annotations

# ══════════════════════════════════════════════════════════════════════════════
# §0. IMPORTACIONES CANÓNICAS
# ══════════════════════════════════════════════════════════════════════════════

import hashlib
import logging
import math
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
)

import numpy as np
from numpy.typing import NDArray

try:
    import scipy.linalg as _sla
    _HAS_SCIPY = True
except ImportError:                                         # pragma: no cover
    import numpy.linalg as _sla                             # type: ignore
    _HAS_SCIPY = False

try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
    from app.core.schemas import Stratum                   # noqa: F401
except ImportError:                                         # pragma: no cover
    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos MIC."""
        pass

    class Morphism:
        """Clase base de morfismos del Topos cuando no existe dependencia externa."""
        pass

    Stratum = None                                          # type: ignore


logger = logging.getLogger("MIC.Tactics.PipelineDirectorAgent")
logger.addHandler(logging.NullHandler())


# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES MATEMÁTICAS Y TOLERANCIAS ESPECTRALES
# ══════════════════════════════════════════════════════════════════════════════

_WILKINSON_LIMIT: float                 = 1.0e-12
_MACHINE_EPSILON: float                 = float(np.finfo(np.float64).eps)
_BASE_SPECTRAL_TOLERANCE: float         = 1e-10
_BASE_POWER_TOLERANCE: float            = 1e-8
_DEFAULT_POWER_AUDIT_MAX_DIM: int       = 96
_GELFAND_ITERATION_STEPS: int           = 20
_SCHUR_CONDITION_SAFETY_FACTOR: float   = 4.0
_SPECTRAL_GAP_FLOOR: float              = 1.0e-12
_SCHUR_RECONSTRUCTION_TOL_FACTOR: float = 16.0

# — Constantes de mecánica celeste de Poincaré ————————————————————
_POINCARE_SECTION_TOL: float            = 1.0e-7
_POINCARE_TRANSVERSALITY_TOL: float     = 1.0e-10
_POINCARE_MAX_SEEDS: int                = 8
_POINCARE_MAX_RETURN_HITS: int          = 12
_MELNIKOV_MAX_ADMISSIBLE: float         = 1.0e-6
_MELNIKOV_ZERO_TOL: float               = 1.0e-14
_MELNIKOV_N_PERIODS: int                = 16
_KAM_PERSISTENCE_TOL: float             = 1.0e-6
_KAM_DIOPHANTINE_GAMMA: float           = 1.0e-4
_KAM_DIOPHANTINE_TAU_OFFSET: float      = 1.0     # τ = n − 1 + offset
_ARNOLD_DIFFUSION_TOL: float            = 1.0e-4
_LYAPUNOV_HYPERBOLIC_TOL: float         = 1.0e-9
_FLOQUET_UNIT_CIRCLE_TOL: float         = 1.0e-6
_NEKHOROSHEV_EXP_CAP: float             = 40.0    # evita overflow en exp

# — Constantes del kernel de automejora recursiva N3 (consumidas en F2) —
_N3_MAX_LEVEL: int                      = 3
_N3_IDEMPOTENCE_TOL: float              = 1.0e-7
_N3_ALPHA_FLOOR: float                  = 1.0e-6
_N3_ALPHA_CEIL: float                   = 1.0e+2
_N3_DEFAULT_CYCLES: int                 = 3

# Semillas deterministas (reproducibilidad espectral / Poincaré).
_SEED_GELFAND: int                      = 42
_SEED_POINCARE: int                     = 0xC0FFEE
_SEED_N3: int                           = 0xBEEF


# ══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES DE GOBERNANZA CAUSAL
# ══════════════════════════════════════════════════════════════════════════════

class PipelineDirectorAgentError(TopologicalInvariantError):
    """Excepción raíz del Custodio de la Causalidad Funtorial."""
    pass


class AdjacencyMatrixFormatError(PipelineDirectorAgentError):
    pass


class CausalLoopVetoError(PipelineDirectorAgentError):
    pass


class NilpotenceIndexVetoError(PipelineDirectorAgentError):
    pass


class StratumMappingError(PipelineDirectorAgentError):
    pass


class SelfLoopVetoError(PipelineDirectorAgentError):
    pass


class FiltrationViolationVeto(PipelineDirectorAgentError):
    pass


class AdjacencySupportVetoError(PipelineDirectorAgentError):
    pass


class MayerVietorisInputError(PipelineDirectorAgentError):
    pass


class HomologicalFusionVeto(PipelineDirectorAgentError):
    pass


class EulerPoincareMismatchError(PipelineDirectorAgentError):
    pass


class PoincareTransversalityError(PipelineDirectorAgentError):
    r"""
    Detonada cuando el campo A·v es tangente a la sección Σ en un cruce
    candidato: |⟨n_Σ, A v⟩| ≤ ε_trans ⇒ ḣ = 0 y P no está bien definida.
    """
    pass


class MelnikovTangleError(PipelineDirectorAgentError):
    r"""
    Detonada cuando la función de Melnikov presenta tangencias homoclínicas
    cuya magnitud excede la cota admisible ‖M(t₀)‖_∞ > ε_Melnikov.
    """
    pass


class KAMBreakdownError(PipelineDirectorAgentError):
    r"""
    Detonada cuando ε_perturbación > ε_KAM: los toros invariantes
    colapsan y la filtración DIKW pierde su rigidez simpléctica.
    """
    pass


class RecursiveSelfImprovementError(PipelineDirectorAgentError):
    r"""Detonada ante divergencia o fallo de idempotencia del kernel N3."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# §C. UTILIDADES CANÓNICAS  (definidas ANTES de su primer uso)
# ══════════════════════════════════════════════════════════════════════════════

def _utc_timestamp() -> str:
    """Marca temporal UTC ISO-8601 con resolución de segundos."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256_short(s: str) -> str:
    """Huella SHA-256 truncada a 16 hex (trazabilidad de certificados)."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _safe_unit(v: NDArray[np.float64], floor: float = _MACHINE_EPSILON) -> NDArray[np.float64]:
    """Normaliza v; si ‖v‖ < floor retorna ceros (evita NaN)."""
    nrm = float(np.linalg.norm(v))
    if nrm < floor:
        return np.zeros_like(v, dtype=np.float64)
    return (v / nrm).astype(np.float64, copy=False)


def _finite_or_raise(name: str, value: float) -> float:
    if not math.isfinite(value):
        raise PipelineDirectorAgentError(f"{name} no finito: {value!r}.")
    return float(value)


# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FASE 1 ──► OBSERVE : CERTIFICACIÓN ESPECTRAL + MECÁNICA DE POINCARÉ    █
# █   (Schur RSF + Bauer-Fike + sección transversal + Melnikov + KAM)        █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████

# ── 1.0 DTOs DE FASE 1 ──────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class SchurBlock:
    r"""
    Bloque irreducible de la forma real de Schur (RSF).

    Un bloque 1×1 es un autovalor real λ.
    Un bloque 2×2 [[a, b], [c, d]] representa un par conjugado
        λ = α ± iω,  α = (a+d)/2,  ω = √max(0, |bc| − ((a−d)/2)²)
    e induce un oscilador plano (órbita celeste linealizada) de
    frecuencia ω y radio de amortiguamiento e^{α t}.
    """
    index: int
    offset: int
    size: int                       # 1 o 2
    trace: float
    det: float
    real_part: float                # α
    frequency: float                # ω ≥ 0
    modulus: float                  # |λ|
    is_oscillatory: bool            # size == 2 y ω > 0


@dataclass(frozen=True, slots=True, eq=False)
class NilpotenceAuditData:
    r"""
    Artefacto espectral de Fase 1.

    Certificado de Spec(A) = {0} mediante descomposición de Schur real.
    Extensión v3.1:
        - schur_reconstruction_residual : ‖Q T Qᵀ − A‖_F  (error backward).
        - lyapunov_max                  : exponente de Lyapunov dominante.
        - spectrum_certificate_hash     : huella SHA-256 corta.
    """
    dimension:                      int
    spectral_radius:                float
    tolerance:                      float
    adjacency_inf_norm:             float
    frobenius_norm:                 float
    nonzero_entries:                int
    directed_density:               float
    is_strictly_nilpotent:          bool
    schur_condition_number:         float = 1.0
    schur_reconstruction_residual:  float = 0.0
    power_residual:                 Optional[float] = None
    power_audited:                  bool = False
    nilpotence_index:               Optional[int] = None
    gelfand_radius_estimate:        Optional[float] = None
    schur_2x2_blocks_detected:      int = 0
    lyapunov_max:                   Optional[float] = None
    spectrum_certificate_hash:      Optional[str] = None
    notes:                          Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class PoincareSection:
    r"""
    Punto de la sección transversal Σ ⊂ ℝⁿ.

    Σ = { p : h(p) = 0 } con h(p) := ‖φ_k(p)‖₂ − tol_BF  (flujo nilpotente)
    o, para bloques 2×2, h(z) := arg(z)  (sección clásica θ = 0 mod 2π).

    Un crossing p_i ∈ Σ corresponde a un índice k_i tal que el flujo
    cruza Σ.  La condición de transversabilidad exige
        |ḣ| = |⟨n_Σ, A p⟩| > ε_trans.
    """
    crossing_index:     int
    residual:           float
    tangent:            NDArray[np.float64]
    normal:             NDArray[np.float64]
    schur_block:        int
    transversality:     float               # |⟨n, A p⟩|
    is_transverse:      bool
    section_kind:       str                 # "nilpotent_norm" | "oscillator_arg"
    interpolant_tstar:  float
    seed_id:            int = 0
    metadata:           Tuple[Tuple[str, Any], ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class ReturnMapReport:
    r"""
    Análisis de la aplicación de primer retorno P: Σ → Σ.

    Estima:
      · Punto fijo p* ≈ media de los cruces (si es periódico).
      · Espectro de Lyapunov λ_i (Benettin–QR sobre A).
      · Multiplicadores de Floquet de DP (proyección a TΣ).
      · Hiperbolicidad: ningún exponente nulo y ningún |μ_Floquet| = 1.
    """
    n_crossings:            int
    n_seeds:                int
    fixed_point_estimate:   Optional[NDArray[np.float64]]
    stability_exponent:     float
    lyapunov_spectrum:      Tuple[float, ...]
    floquet_moduli:         Tuple[float, ...]
    mean_return_time:       float
    is_hyperbolic:          bool
    is_periodic_orbit:      bool
    is_transverse_family:   bool


@dataclass(frozen=True, slots=True, eq=False)
class MelnikovCertificate:
    r"""
    Certificado de la función de Melnikov discreta.

    M(t₀) = Σ_k  (f₀ ∧ f₁)(γ_k ; t_k + t₀) · Δt

    f₀ := campo triangular estricto de T  (flujo nilpotente no perturbado).
    f₁ := campo de bloques diagonales/2×2 de T  (perturbación resonante).
    La 2-forma se evalúa en el plano dominante de Schur (primeros 2
    vectores de Q, o el bloque 2×2 de mayor |ω|).

    Ceros simples de M con M' ≠ 0 ⇒ W^s ⋔ W^u del subespacio nulo.
    """
    melnikov_values:        NDArray[np.float64]
    melnikov_derivative:    NDArray[np.float64]
    simple_zeros:           int
    transverse_zeros:       int             # ceros con |M'| > tol
    mean_magnitude:         float
    max_magnitude:          float
    splitting_distance:     float           # ∼ |M| / ‖f₀‖  (dist. W^s–W^u)
    tangle_detected:        bool
    is_admissible:          bool            # ‖M‖_∞ ≤ ε_Melnikov


@dataclass(frozen=True, slots=True)
class KAMCertificate:
    r"""
    Certificado de persistencia KAM.

    ε = ‖T_offdiag‖_F / ω_min          (razón de perturbación canónica)
    Condición diofantina sobre las frecuencias de bloques 2×2:
        |k₁ ω₁ + k₂ ω₂| ≥ γ / (|k₁|+|k₂|)^τ ,  τ = n−1+δ.
    Si ε < ε_KAM y las frecuencias son diofantinas, los autovalores
    cero sobreviven como toros invariantes (Greene / Pöschel).
    """
    epsilon_perturbation:   float
    epsilon_KAM_threshold:  float
    diophantine_gamma:      float
    diophantine_satisfied:  bool
    small_divisor_min:      float
    tori_persisting:        int
    is_KAM_stable:          bool


@dataclass(frozen=True, slots=True)
class ArnoldDiffusionMetric:
    r"""
    Difusión de Arnold — fuga entre resonancias espectrales.

    D_Arnold = Σ_k |Δν_k|² / ω_k²
    donde Δν_k = desviación de la traza del k-ésimo bloque 2×2
    (idealmente nula si A es nilpotente exacta) y ω_k = frecuencia
    natural del bloque.

    Tiempo de Nekhoroshev (cota superior de deriva):
        t_Nek ∼ exp( c · ε^{-1/(2n)} ).
    Adiabaticidad ⇔ D_Arnold < ε_tol  y  t_Nek grande.
    """
    per_block_leakage:      Dict[str, float]
    total_diffusion:        float
    dominant_channel:       Optional[str]
    nekhoroshev_time:       float
    is_adiabatic:           bool


@dataclass(frozen=True, slots=True, eq=False)
class ObserverPacket:
    r"""
    ← ÚLTIMO ARTEFACTO DE FASE 1.

    Transporta el certificado espectral (Schur) junto con la maquinaria
    de Poincaré (Σ, P, M, KAM, Arnold, Lyapunov) hacia FASE 2, donde
    alimenta al kernel de automejora recursiva N3 como semilla (θ, O, α).

    Invariante de empaquetado:
        ObserverPacket ⊢ Spec(A) certificado  ∧  Σ transversal
                       ∧  Melnikov evaluada   ∧  KAM/Arnold medidos.
    """
    spectra:            NilpotenceAuditData
    sections:           Tuple[PoincareSection, ...]
    return_map:         ReturnMapReport
    melnikov:           MelnikovCertificate
    kam:                KAMCertificate
    arnold:             ArnoldDiffusionMetric
    schur_blocks:       Tuple[SchurBlock, ...]
    n3_theta_seed:      Tuple[float, ...]       # semilla canónica de θ para N3
    generated_at_utc:   str
    packet_hash:        str = ""


# ── 1.1 CLASE FASE 1 ────────────────────────────────────────────────────────

class Phase1_SpectralNilpotenceCertifier:
    r"""
    Certificador espectral de nilpotencia + mecánica de Poincaré.

    Fundamento (Cayley–Hamilton + teoría espectral):
        A ∈ ℝ^{n×n} es nilpotente  ⟺  ∃ ν ≤ n: A^ν = 0  ⟺  Spec(A) = {0}.

    Estrategia numérica:
        Descomposición de Schur real (RSF) A = Q · T · Qᵀ con LAPACK dgees.
        Error backward-stable: ‖Q T Qᵀ − A‖₂ ≤ n · ε_machine · ‖A‖₂.

    Cadena interna de Fase 1 (OODA-Observe):
        validar A → Schur → Bauer–Fike → ν → Σ → P → M → KAM → Arnold
        → ObserverPacket → puente Φ₁→Φ₂.
    """

    # ══════════════════════════════════════════════════════════════════════
    # §1.1  Validación y acondicionamiento del operador de adyacencia
    # ══════════════════════════════════════════════════════════════════════
    def _validate_and_condition_adjacency(
        self,
        adjacency_matrix: NDArray[Any],
        *,
        zero_threshold: float = _BASE_SPECTRAL_TOLERANCE,
        allow_signed_weights: bool = False,
        require_boolean: bool = False,
    ) -> NDArray[np.float64]:
        r"""
        Valida y condiciona A ∈ ℝ^{n×n}.

        Contrato:
            · 2-tensor, cuadrado, entradas finitas.
            · |A[i,j]| ≤ zero_threshold  ↦  0  (soporte numérico).
            · pesos ≥ 0 salvo ``allow_signed_weights``.
            · si ``require_boolean``, el soporte se proyecta a {0,1}.
        """
        A = np.asarray(adjacency_matrix, dtype=np.float64)

        if A.ndim != 2:
            raise AdjacencyMatrixFormatError(
                f"A debe ser un operador 2D (endomorfismo). ndim={A.ndim}."
            )
        if A.shape[0] != A.shape[1]:
            raise AdjacencyMatrixFormatError(
                f"A debe ser cuadrada. shape={A.shape}."
            )
        if not np.all(np.isfinite(A)):
            nan_c = int(np.sum(np.isnan(A)))
            inf_c = int(np.sum(np.isinf(A)))
            raise AdjacencyMatrixFormatError(
                f"A contiene no finitos. NaN={nan_c}, Inf={inf_c}."
            )
        if zero_threshold < 0.0:
            raise AdjacencyMatrixFormatError(
                f"zero_threshold debe ser ≥ 0. Recibido {zero_threshold}."
            )

        if zero_threshold > 0.0:
            A = np.where(np.abs(A) <= zero_threshold, 0.0, A)

        if not allow_signed_weights:
            neg_mask = A < 0.0
            if np.any(neg_mask):
                neg_count = int(np.count_nonzero(neg_mask))
                min_neg = float(np.min(A[neg_mask]))
                raise AdjacencyMatrixFormatError(
                    f"A contiene {neg_count} entradas negativas (mínimo="
                    f"{min_neg:.6e}). Un DAG causal requiere pesos ≥ 0."
                )

        if require_boolean:
            nonzero = A[np.abs(A) > 0.0]
            bool_tol = max(zero_threshold, _MACHINE_EPSILON)
            if nonzero.size > 0:
                if not np.allclose(nonzero, 1.0, rtol=0.0, atol=bool_tol):
                    max_dev = float(np.max(np.abs(nonzero - 1.0)))
                    raise AdjacencyMatrixFormatError(
                        f"require_boolean=True: desviación máxima "
                        f"{max_dev:.6e} en pesos distintos de 1."
                    )
            A = (A != 0.0).astype(np.float64)

        return A

    # ══════════════════════════════════════════════════════════════════════
    # §1.2  Extracción estructurada de bloques RSF
    # ══════════════════════════════════════════════════════════════════════
    def _extract_schur_blocks(
        self,
        T: NDArray[np.float64],
    ) -> Tuple[SchurBlock, ...]:
        r"""
        Recorre la subdiagonal de T y emite bloques 1×1 / 2×2 irreducibles.

        Un bloque 2×2 se declara cuando |T[i+1, i]| > ε_machine.
        Frecuencia del oscilador plano:
            ω = √max(0, −Δ),  Δ = ((a−d)/2)² + bc   (discriminante).
        Si Δ < 0 el par es elíptico (órbita celeste); si Δ ≥ 0 es hiperbólico
        o degenerado (no oscilatorio).
        """
        n = int(T.shape[0])
        blocks: List[SchurBlock] = []
        i = 0
        idx = 0
        while i < n:
            if i < n - 1 and abs(T[i + 1, i]) > _MACHINE_EPSILON:
                a = float(T[i, i])
                b = float(T[i, i + 1])
                c = float(T[i + 1, i])
                d = float(T[i + 1, i + 1])
                tr = a + d
                det = a * d - b * c
                alpha = 0.5 * tr
                disc = ((a - d) * 0.5) ** 2 + b * c
                if disc < 0.0:
                    omega = float(math.sqrt(-disc))
                    oscillatory = omega > _SPECTRAL_GAP_FLOOR
                else:
                    omega = 0.0
                    oscillatory = False
                modulus = float(math.hypot(alpha, omega))
                blocks.append(SchurBlock(
                    index=idx, offset=i, size=2,
                    trace=tr, det=float(det),
                    real_part=float(alpha), frequency=omega,
                    modulus=modulus, is_oscillatory=oscillatory,
                ))
                idx += 1
                i += 2
            else:
                lam = float(T[i, i])
                blocks.append(SchurBlock(
                    index=idx, offset=i, size=1,
                    trace=lam, det=lam,
                    real_part=lam, frequency=0.0,
                    modulus=abs(lam), is_oscillatory=False,
                ))
                idx += 1
                i += 1
        return tuple(blocks)

    # ══════════════════════════════════════════════════════════════════════
    # §1.3  Descomposición de Schur real + residuo de reconstrucción
    # ══════════════════════════════════════════════════════════════════════
    def _schur_decompose(
        self,
        A: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float, int, float]:
        r"""
        A = Q · T · Qᵀ con T cuasi-triangular superior (RSF).

        Retorna
            (T, Q, κ₂(Q), n_blocks_2x2, ‖Q T Qᵀ − A‖_F).

        κ₂(Q) = σ_max / σ_min; para dgees, Q es ortogonal y κ₂ ≃ 1.
        Si scipy no está disponible, se degrada a eigvals (T diagonal,
        Q = I) y el residuo se reporta como ‖A − diag(Re λ)‖_F.
        """
        n = int(A.shape[0])

        if n == 0:
            z = np.zeros((0, 0), dtype=np.float64)
            return z, z, 1.0, 0, 0.0

        if not _HAS_SCIPY:
            logger.warning("scipy ausente. Fallback a numpy.linalg.eigvals.")
            eigenvalues = np.linalg.eigvals(A)
            T = np.diag(np.real(eigenvalues))
            Q = np.eye(n, dtype=np.float64)
            residual = float(np.linalg.norm(A - T, ord="fro"))
            return T, Q, 1.0, 0, residual

        try:
            T, Q = _sla.schur(A, output="real")
        except Exception as exc:
            logger.warning("schur falló (%s). Fallback eigvals.", exc)
            eigenvalues = _sla.eigvals(A)
            T = np.diag(np.real(eigenvalues))
            Q = np.eye(n, dtype=np.float64)
            residual = float(np.linalg.norm(A - T, ord="fro"))
            return T, Q, 1.0, 0, residual

        try:
            sv = _sla.svd(Q, compute_uv=False)
            sv_min = float(sv[-1])
            sv_max = float(sv[0])
            kappa_Q = (
                sv_max / sv_min if sv_min > _MACHINE_EPSILON else float("inf")
            )
        except Exception:
            kappa_Q = 1.0

        try:
            recon = Q @ T @ Q.T
            residual = float(np.linalg.norm(recon - A, ord="fro"))
        except Exception:
            residual = float("inf")

        n_blocks_2x2 = 0
        i = 0
        while i < n - 1:
            if abs(T[i + 1, i]) > _MACHINE_EPSILON:
                n_blocks_2x2 += 1
                i += 2
            else:
                i += 1

        return T, Q, float(kappa_Q), n_blocks_2x2, float(residual)

    # ══════════════════════════════════════════════════════════════════════
    # §1.4  Tolerancia espectral dinámica calibrada por Bauer–Fike
    # ══════════════════════════════════════════════════════════════════════
    def _dynamic_spectral_tolerance(
        self,
        dimension: int,
        frobenius_norm: float,
        schur_condition_number: float = 1.0,
        reconstruction_residual: float = 0.0,
    ) -> float:
        r"""
        tol = max(tol_base,
                  κ₂(Q) · C · √n · ε_machine · ‖A‖_F,
                  C' · ρ_recon).

        Justificación (Bauer–Fike): min_μ |λ̃ − μ| ≤ κ₂(X) · ‖ΔA‖₂
        con ‖ΔA‖₂ ≤ C · √n · ε_machine · ‖A‖_F para LAPACK dgees,
        más el residuo de reconstrucción observado.
        """
        if dimension <= 0:
            return _BASE_SPECTRAL_TOLERANCE
        kappa = max(1.0, float(schur_condition_number))
        f_norm = max(1.0, float(frobenius_norm))
        adaptive = (
            kappa
            * _SCHUR_CONDITION_SAFETY_FACTOR
            * math.sqrt(dimension)
            * _MACHINE_EPSILON
            * f_norm
        )
        recon_term = _SCHUR_RECONSTRUCTION_TOL_FACTOR * max(0.0, reconstruction_residual)
        return float(max(_BASE_SPECTRAL_TOLERANCE, adaptive, recon_term))

    # ══════════════════════════════════════════════════════════════════════
    # §1.5  Radio espectral de Gelfand para matrices grandes
    # ══════════════════════════════════════════════════════════════════════
    def _gelfand_spectral_radius_estimate(
        self,
        A: NDArray[np.float64],
        n_steps: int = _GELFAND_ITERATION_STEPS,
    ) -> float:
        r"""
        ρ(A) ≈ lim_{k→∞} ‖A^k‖^{1/k} — vía iteración de potencia normalizada.

        Se reporta el último radio observado; si el iterado colapsa a 0
        (nilpotencia numérica) se retorna 0.
        """
        n = int(A.shape[0])
        if n == 0:
            return 0.0

        rng = np.random.default_rng(seed=_SEED_GELFAND)
        v = rng.standard_normal(n)
        v = _safe_unit(v)
        if float(np.linalg.norm(v)) < _MACHINE_EPSILON:
            return 0.0

        radius = 0.0
        Av = A @ v
        for k in range(1, n_steps + 1):
            norm_Av = float(np.linalg.norm(Av))
            if norm_Av < _MACHINE_EPSILON:
                return 0.0
            radius = float(norm_Av ** (1.0 / k))
            v = Av / norm_Av
            Av = A @ v
        return float(radius)

    # ══════════════════════════════════════════════════════════════════════
    # §1.6  Índice de nilpotencia por búsqueda binaria
    # ══════════════════════════════════════════════════════════════════════
    def _certify_nilpotence_index(
        self,
        A: NDArray[np.float64],
        power_tolerance: float,
        power_audit_max_dim: int,
    ) -> Tuple[Optional[float], Optional[int], Optional[float]]:
        r"""
        Certifica ν = min{k : ‖A^k‖_∞ ≤ tol_power} y el residual ‖A^n‖_∞.

        Complejidad: O(log n) productos matriciales por exponentiation
        binaria de ``np.linalg.matrix_power``.  Si n > power_audit_max_dim
        se degrada a la estimación de Gelfand (sin ν exacto).

        Retorna (residual, ν, gelfand_estimate).
        """
        n = int(A.shape[0])
        if n == 0:
            return (0.0, 0, None)
        if n > power_audit_max_dim:
            return (None, None, self._gelfand_spectral_radius_estimate(A))

        try:
            A_n = np.linalg.matrix_power(A, n)
            residual = float(np.linalg.norm(A_n, ord=np.inf))
        except Exception as exc:
            logger.warning("matrix_power falló: %s", exc)
            return (None, None, None)

        nu: Optional[int] = None
        lo, hi = 1, n
        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                norm_mid = float(
                    np.linalg.norm(np.linalg.matrix_power(A, mid), ord=np.inf)
                )
            except Exception:
                break
            if norm_mid <= power_tolerance:
                nu = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return (residual, nu, None)

    # ══════════════════════════════════════════════════════════════════════
    # §1.7  Radio espectral desde la forma de Schur
    # ══════════════════════════════════════════════════════════════════════
    def _extract_spectral_radius_from_schur(
        self,
        T: NDArray[np.float64],
        blocks: Optional[Sequence[SchurBlock]] = None,
    ) -> float:
        r"""
        ρ(A) = max |λ| extraído de la RSF T (bloques 1×1 y 2×2).

        Para un bloque 2×2, |λ| = hypot(α, ω) con α = tr/2 y ω la
        frecuencia elíptica.  Si se proveen ``blocks`` precomputados
        se reutilizan (evita un segundo barrido).
        """
        if blocks:
            return float(max((b.modulus for b in blocks), default=0.0))
        n = int(T.shape[0])
        if n == 0:
            return 0.0
        extracted = self._extract_schur_blocks(T)
        return float(max((b.modulus for b in extracted), default=0.0))

    # ══════════════════════════════════════════════════════════════════════
    # §1.8  Espectro de Lyapunov (algoritmo de Benettin–QR)
    # ══════════════════════════════════════════════════════════════════════
    def _lyapunov_spectrum_benettin(
        self,
        A: NDArray[np.float64],
        n_steps: Optional[int] = None,
        n_exponents: Optional[int] = None,
    ) -> Tuple[float, ...]:
        r"""
        Espectro de Lyapunov discreto de φ(v) = A v.

        Itera Q_{k+1} R_{k+1} = A Q_k (QR de Gram–Schmidt / Householder)
        y acumula λ_i = (1/N) Σ_k log |R_{k,ii}|.

        Para A nilpotente exacta, λ_i → −∞ (se satura a log ε).
        Para A con ciclo, max λ_i = log ρ(A) ≥ 0.
        """
        n = int(A.shape[0])
        if n == 0:
            return ()
        n_exp = int(max(1, min(n, n_exponents if n_exponents is not None else min(n, 4))))
        steps = int(max(1, n_steps if n_steps is not None else min(n, 32)))

        rng = np.random.default_rng(seed=_SEED_POINCARE)
        Q, _ = np.linalg.qr(rng.standard_normal((n, n_exp)))
        acc = np.zeros(n_exp, dtype=np.float64)
        valid_steps = 0
        for _ in range(steps):
            Z = A @ Q
            Q_new, R = np.linalg.qr(Z)
            diag = np.abs(np.diag(R))
            if not np.all(np.isfinite(diag)):
                break
            acc += np.log(np.maximum(diag, _MACHINE_EPSILON))
            Q = Q_new
            valid_steps += 1
        if valid_steps == 0:
            return tuple(0.0 for _ in range(n_exp))
        return tuple(float(x) for x in (acc / float(valid_steps)))

    # ══════════════════════════════════════════════════════════════════════
    # §1.9  Certificación espectral principal
    # ══════════════════════════════════════════════════════════════════════
    def _certify_spectral_nilpotence(
        self,
        adjacency_matrix: NDArray[Any],
        *,
        zero_threshold: float = _BASE_SPECTRAL_TOLERANCE,
        allow_signed_weights: bool = False,
        require_boolean: bool = False,
        deep_nilpotence_audit: bool = True,
        power_audit_max_dim: int = _DEFAULT_POWER_AUDIT_MAX_DIM,
        raise_on_veto: bool = True,
    ) -> NilpotenceAuditData:
        r"""
        Certifica Spec(A) = {0} vía Schur. Emite NilpotenceAuditData.

        Criterio: ρ(A) ≤ tol_BF(κ₂(Q), ‖A‖_F, ρ_recon).
        Si ``deep_nilpotence_audit`` y n ≤ power_audit_max_dim, se
        certifica además el índice de nilpotencia ν por búsqueda binaria.
        """
        A = self._validate_and_condition_adjacency(
            adjacency_matrix,
            zero_threshold=zero_threshold,
            allow_signed_weights=allow_signed_weights,
            require_boolean=require_boolean,
        )
        n = int(A.shape[0])
        notes: List[str] = []

        if n == 0:
            return NilpotenceAuditData(
                dimension=0, spectral_radius=0.0,
                tolerance=_BASE_SPECTRAL_TOLERANCE,
                adjacency_inf_norm=0.0, frobenius_norm=0.0,
                nonzero_entries=0, directed_density=0.0,
                is_strictly_nilpotent=True,
                schur_condition_number=1.0,
                schur_reconstruction_residual=0.0,
                power_residual=0.0, power_audited=True,
                nilpotence_index=0, gelfand_radius_estimate=None,
                schur_2x2_blocks_detected=0,
                lyapunov_max=float("-inf"),
                spectrum_certificate_hash=_sha256_short("empty|nilpotent"),
                notes=("Operador vacío trivialmente nilpotente.",),
            )

        fro = float(np.linalg.norm(A, ord="fro"))
        inf_norm = float(np.linalg.norm(A, ord=np.inf))
        nonzero = int(np.count_nonzero(A))
        if n > 1:
            mask = ~np.eye(n, dtype=bool)
            density = float(np.count_nonzero(A[mask])) / float(n * (n - 1))
        else:
            density = 0.0

        T, Q, kappa_Q, blocks_2x2, recon_res = self._schur_decompose(A)
        blocks = self._extract_schur_blocks(T)

        if kappa_Q > 1.0 + 1e-6:
            notes.append(
                f"κ₂(Q)={kappa_Q:.4f}>1: similaridad no exactamente unitaria."
            )
        if blocks_2x2 > 0:
            notes.append(
                f"RSF contiene {blocks_2x2} bloque(s) 2×2 "
                f"({sum(1 for b in blocks if b.is_oscillatory)} oscilatorio(s))."
            )
        recon_tol = _SCHUR_RECONSTRUCTION_TOL_FACTOR * n * _MACHINE_EPSILON * max(1.0, fro)
        if recon_res > recon_tol:
            notes.append(
                f"Residuo de reconstrucción Schur ‖QTQᵀ−A‖_F={recon_res:.4e} "
                f"> {recon_tol:.4e}."
            )

        tol = self._dynamic_spectral_tolerance(n, fro, kappa_Q, recon_res)
        rho = self._extract_spectral_radius_from_schur(T, blocks)

        residual: Optional[float] = None
        nu: Optional[int] = None
        gelfand: Optional[float] = None
        power_audited = False

        if deep_nilpotence_audit:
            pow_tol = max(
                _BASE_POWER_TOLERANCE,
                _SCHUR_CONDITION_SAFETY_FACTOR * tol * max(1.0, inf_norm),
            )
            residual, nu, gelfand = self._certify_nilpotence_index(
                A, pow_tol, power_audit_max_dim,
            )
            power_audited = residual is not None

            if power_audited and residual is not None and residual > pow_tol:
                notes.append(
                    f"‖A^n‖_∞={residual:.4e} > tol_power={pow_tol:.4e}."
                )
            if nu is not None:
                notes.append(f"Índice de nilpotencia certificado ν={nu}.")
            if gelfand is not None:
                notes.append(f"Estimación Gelfand ρ̂(A)={gelfand:.4e}.")

        lyap_spec = self._lyapunov_spectrum_benettin(A, n_steps=min(n, 24), n_exponents=1)
        lyap_max = lyap_spec[0] if lyap_spec else None
        if lyap_max is not None:
            notes.append(f"λ_max (Benettin)={lyap_max:.4e}.")

        is_nilpotent = rho <= tol

        cert_hash = _sha256_short(
            f"{n}|{rho:.12e}|{tol:.12e}|{nu}|{blocks_2x2}|{recon_res:.12e}"
        )

        audit = NilpotenceAuditData(
            dimension=n, spectral_radius=rho, tolerance=tol,
            adjacency_inf_norm=inf_norm, frobenius_norm=fro,
            nonzero_entries=nonzero, directed_density=density,
            is_strictly_nilpotent=is_nilpotent,
            schur_condition_number=kappa_Q,
            schur_reconstruction_residual=recon_res,
            power_residual=residual, power_audited=power_audited,
            nilpotence_index=nu, gelfand_radius_estimate=gelfand,
            schur_2x2_blocks_detected=blocks_2x2,
            lyapunov_max=lyap_max,
            spectrum_certificate_hash=cert_hash,
            notes=tuple(notes),
        )

        if not is_nilpotent and raise_on_veto:
            raise CausalLoopVetoError(
                f"Ruptura de causalidad espectral: ρ(A)={rho:.6e} > "
                f"tol={tol:.6e} (κ₂(Q)={kappa_Q:.4f}, ‖A‖_F={fro:.4e}, n={n}). "
                f"Existe al menos un ciclo dirigido en el DAG."
            )
        return audit

    # ══════════════════════════════════════════════════════════════════════
    # §1.10  Semillas deterministas para la sección de Poincaré
    # ══════════════════════════════════════════════════════════════════════
    def _poincare_seed_bundle(
        self,
        n: int,
        n_seeds: int,
    ) -> Tuple[NDArray[np.float64], ...]:
        r"""
        Construye un haz de semillas ortonormales: base canónica (hasta n)
        más direcciones Haar-aleatorias (QR de Gaussianas) para cubrir Σ.
        """
        if n <= 0:
            return ()
        n_seeds = int(max(1, min(n_seeds, _POINCARE_MAX_SEEDS, n)))
        seeds: List[NDArray[np.float64]] = []
        for i in range(min(n_seeds, n)):
            e = np.zeros(n, dtype=np.float64)
            e[i] = 1.0
            seeds.append(e)
        if n_seeds > len(seeds):
            rng = np.random.default_rng(seed=_SEED_POINCARE)
            extra = n_seeds - len(seeds)
            G = rng.standard_normal((n, extra))
            Q, _ = np.linalg.qr(G)
            for j in range(Q.shape[1]):
                seeds.append(Q[:, j].astype(np.float64, copy=False))
        return tuple(seeds[:n_seeds])

    # ══════════════════════════════════════════════════════════════════════
    # §1.11  Cruce nilpotente: interpolación lineal de ‖A^k v‖ = tol
    # ══════════════════════════════════════════════════════════════════════
    def _nilpotent_norm_crossing(
        self,
        A: NDArray[np.float64],
        v0: NDArray[np.float64],
        power_tolerance: float,
        schur_block: int,
        seed_id: int,
        max_hits: int = 1,
    ) -> List[PoincareSection]:
        r"""
        Integra v_{k+1} = A v_k y detecta cruces de la hipersuperficie
            h(v) = ‖v‖₂ − tol_BF = 0
        con interpolación lineal en k y normal n_Σ = v / ‖v‖.

        Transversabilidad: |⟨n_Σ, A v⟩| = |‖v‖' | ≈ |‖Av‖ − ‖v‖|.
        """
        n = int(A.shape[0])
        sections: List[PoincareSection] = []
        v = v0.astype(np.float64, copy=True)
        prev = v.copy()
        prev_norm = float(np.linalg.norm(v))
        if prev_norm < _MACHINE_EPSILON:
            return sections
        v = v / prev_norm
        prev = v.copy()
        prev_norm = 1.0

        for k in range(1, n + 1):
            v = A @ v
            norm_v = float(np.linalg.norm(v))
            crossed = (prev_norm > power_tolerance) and (norm_v <= power_tolerance)
            if crossed:
                denom = max(prev_norm - norm_v, _MACHINE_EPSILON)
                tau = (power_tolerance - norm_v) / denom
                tau = float(min(1.0, max(0.0, tau)))
                t_star = float(k) - tau
                p = (1.0 - tau) * v + tau * prev
                nrm_p = float(np.linalg.norm(p))
                normal = _safe_unit(p if nrm_p >= _MACHINE_EPSILON else v)
                Av = A @ p
                trans = abs(float(np.dot(normal, Av)))
                tangent = Av - float(np.dot(Av, normal)) * normal
                tangent = _safe_unit(tangent)
                is_trans = trans > _POINCARE_TRANSVERSALITY_TOL
                sections.append(PoincareSection(
                    crossing_index=k,
                    residual=norm_v,
                    tangent=tangent,
                    normal=normal,
                    schur_block=schur_block,
                    transversality=trans,
                    is_transverse=is_trans,
                    section_kind="nilpotent_norm",
                    interpolant_tstar=t_star,
                    seed_id=seed_id,
                    metadata=(
                        ("prev_norm", prev_norm),
                        ("tau", tau),
                        ("seed_id", seed_id),
                    ),
                ))
                if len(sections) >= max_hits:
                    break
            if norm_v < _MACHINE_EPSILON:
                break
            prev = v.copy()
            prev_norm = norm_v
            v = v / norm_v
        return sections

    # ══════════════════════════════════════════════════════════════════════
    # §1.12  Sección clásica θ = 0 en bloques 2×2 oscilatorios
    # ══════════════════════════════════════════════════════════════════════
    def _oscillator_arg_crossings(
        self,
        T: NDArray[np.float64],
        Q: NDArray[np.float64],
        blocks: Sequence[SchurBlock],
        n_periods: int = 4,
    ) -> List[PoincareSection]:
        r"""
        Para cada bloque 2×2 elíptico (ω > 0), la dinámica linealizada
        en coordenadas de bloque es una rotación-escala:

            z_{k+1} = e^{α + iω} z_k.

        La sección de Poincaré clásica es arg(z) ≡ 0 (mod 2π).
        Se reporta un cruce por periodo con tangente polar (−y, x).
        """
        n = int(T.shape[0])
        sections: List[PoincareSection] = []
        if n == 0:
            return sections

        for blk in blocks:
            if not blk.is_oscillatory or blk.size != 2:
                continue
            i0 = blk.offset
            # Vector inicial en el plano del bloque, empujado por Q.
            z_plane = np.zeros(n, dtype=np.float64)
            z_plane[i0] = 1.0
            p = Q @ z_plane if Q.shape == (n, n) else z_plane
            p = _safe_unit(p)
            # Tangente polar en el plano (Q e_{i0+1}).
            z_orth = np.zeros(n, dtype=np.float64)
            z_orth[i0 + 1] = 1.0
            tvec = Q @ z_orth if Q.shape == (n, n) else z_orth
            tvec = _safe_unit(tvec)
            trans = abs(blk.frequency)
            is_trans = trans > _POINCARE_TRANSVERSALITY_TOL
            # Un hit por periodo; crossing_index ∝ 2π/ω.
            period = max(1, int(round(2.0 * math.pi / max(blk.frequency, _SPECTRAL_GAP_FLOOR))))
            for q in range(min(n_periods, _POINCARE_MAX_RETURN_HITS)):
                sections.append(PoincareSection(
                    crossing_index=q * period,
                    residual=0.0,
                    tangent=tvec,
                    normal=p,
                    schur_block=blk.index,
                    transversality=trans,
                    is_transverse=is_trans,
                    section_kind="oscillator_arg",
                    interpolant_tstar=float(q) * 2.0 * math.pi,
                    seed_id=1000 + blk.index,
                    metadata=(
                        ("omega", blk.frequency),
                        ("alpha", blk.real_part),
                        ("period", period),
                    ),
                ))
        return sections

    # ══════════════════════════════════════════════════════════════════════
    # §1.13  Captura de la sección transversal de Poincaré
    # ══════════════════════════════════════════════════════════════════════
    def _capture_poincare_section_from_schur(
        self,
        A: NDArray[np.float64],
        T: NDArray[np.float64],
        Q: NDArray[np.float64],
        power_tolerance: float,
        blocks: Optional[Sequence[SchurBlock]] = None,
        n_seeds: int = _POINCARE_MAX_SEEDS,
    ) -> Tuple[PoincareSection, ...]:
        r"""
        Captura Σ = Σ_nilpotent ∪ Σ_oscillator.

        · Σ_nilpotent : cruces de ‖A^k v‖ = tol_BF sobre un haz de semillas.
        · Σ_oscillator: arg(z) = 0 en cada bloque 2×2 elíptico.

        Se descartan (con nota implícita) los cruces no transversales;
        si TODOS son tangentes se retiene el de mayor |ḣ| para no
        dejar Σ vacía (el reporte de retorno marcará is_transverse_family=False).
        """
        n = int(A.shape[0])
        if n == 0:
            return ()

        if blocks is None:
            blocks = self._extract_schur_blocks(T)
        dom_block = int(max(range(len(blocks)), key=lambda i: blocks[i].modulus)) if blocks else 0

        collected: List[PoincareSection] = []
        seeds = self._poincare_seed_bundle(n, n_seeds=min(n_seeds, max(1, n)))
        for sid, seed in enumerate(seeds):
            collected.extend(
                self._nilpotent_norm_crossing(
                    A, seed, power_tolerance, dom_block, sid, max_hits=1,
                )
            )
        collected.extend(self._oscillator_arg_crossings(T, Q, blocks))

        transverse = [s for s in collected if s.is_transverse]
        if transverse:
            return tuple(transverse)
        return tuple(collected)

    # ══════════════════════════════════════════════════════════════════════
    # §1.14  Aplicación de primer retorno P: Σ → Σ  (Floquet + Lyapunov)
    # ══════════════════════════════════════════════════════════════════════
    def _analyze_return_map_from_schur(
        self,
        A: NDArray[np.float64],
        sections: Sequence[PoincareSection],
        lyapunov_spectrum: Optional[Sequence[float]] = None,
    ) -> ReturnMapReport:
        r"""
        Estima el exponente de Lyapunov, los multiplicadores de Floquet
        y la hiperbolicidad del flujo nilpotente a partir de Σ.

        DP se aproxima por la proyección de A^{Δk} sobre TΣ:
            DP · τ = Π_{TΣ} A^{Δk} τ,  Π = I − n nᵀ.
        Los módulos de Floquet son las normas de esas imágenes.
        En A nilpotente exacta, |μ| → 0 (contracción super-exponencial);
        un ciclo implica algún |μ| ≥ 1.
        """
        n_sections = len(sections)
        n = int(A.shape[0])
        n_seeds = len({s.seed_id for s in sections}) if sections else 0

        if n == 0 or n_sections == 0:
            return ReturnMapReport(
                n_crossings=n_sections, n_seeds=n_seeds,
                fixed_point_estimate=None,
                stability_exponent=0.0,
                lyapunov_spectrum=(),
                floquet_moduli=(),
                mean_return_time=0.0,
                is_hyperbolic=False,
                is_periodic_orbit=False,
                is_transverse_family=False,
            )

        lyap = tuple(
            float(x) for x in (
                lyapunov_spectrum
                if lyapunov_spectrum is not None
                else self._lyapunov_spectrum_benettin(A)
            )
        )
        lyap_max = lyap[0] if lyap else 0.0

        # Tiempos de retorno entre cruces del mismo kind y seed-adyacentes.
        by_kind: Dict[str, List[int]] = {}
        for s in sections:
            by_kind.setdefault(s.section_kind, []).append(s.crossing_index)
        deltas: List[int] = []
        for idxs in by_kind.values():
            ordered = sorted(idxs)
            deltas.extend(ordered[i + 1] - ordered[i] for i in range(len(ordered) - 1))
        mean_return = float(np.mean(deltas)) if deltas else float(sections[0].crossing_index)

        # Floquet: un paso de A^{Δk} proyectado a TΣ sobre cada tangente.
        floquet: List[float] = []
        dk = max(1, int(round(mean_return)) if mean_return > 0 else 1)
        try:
            A_ret = np.linalg.matrix_power(A, dk) if dk > 1 else A
        except Exception:
            A_ret = A
        for s in sections:
            nrm = s.normal
            tau = s.tangent
            img = A_ret @ tau
            img = img - float(np.dot(img, nrm)) * nrm
            floquet.append(float(np.linalg.norm(img)))

        floquet_t = tuple(floquet)
        # Periodicidad débil: todos los |μ| ≈ 1.
        is_periodic = (
            len(floquet_t) >= 2
            and all(abs(mu - 1.0) <= _FLOQUET_UNIT_CIRCLE_TOL for mu in floquet_t)
        )
        is_hyp = bool(
            abs(lyap_max) > _LYAPUNOV_HYPERBOLIC_TOL
            or any(abs(mu - 1.0) > _FLOQUET_UNIT_CIRCLE_TOL for mu in floquet_t)
        )
        is_trans_fam = all(s.is_transverse for s in sections)

        fixed_point = None
        if is_periodic:
            fixed_point = np.array(
                [float(s.crossing_index) for s in sections], dtype=np.float64,
            )

        return ReturnMapReport(
            n_crossings=n_sections,
            n_seeds=n_seeds,
            fixed_point_estimate=fixed_point,
            stability_exponent=float(lyap_max),
            lyapunov_spectrum=lyap,
            floquet_moduli=floquet_t,
            mean_return_time=float(mean_return),
            is_hyperbolic=is_hyp,
            is_periodic_orbit=is_periodic,
            is_transverse_family=is_trans_fam,
        )

    # ══════════════════════════════════════════════════════════════════════
    # §1.15  Función de Melnikov — 2-forma sobre el plano dominante
    # ══════════════════════════════════════════════════════════════════════
    def _evaluate_melnikov_function_from_schur(
        self,
        A: NDArray[np.float64],
        T: NDArray[np.float64],
        Q: NDArray[np.float64],
        blocks: Optional[Sequence[SchurBlock]] = None,
        omega_perturbation: float = 1.0,
        n_periods: int = _MELNIKOV_N_PERIODS,
    ) -> MelnikovCertificate:
        r"""
        Discretización de la integral de Melnikov sobre la órbita
        nilpotente γ_k = Π_{dom} A^k v_0  (proyección al plano dominante):

            M(t₀) = Σ_k  det[ f₀(γ_k) | f₁(γ_k, t_k+t₀) ] / (1+k)

        f₀(x) = T_> x   (parte estrictamente triangular superior: flujo
                         nilpotente no perturbado),
        f₁(x) = T_∥ x   (parte de bloques diagonales: perturbación
                         resonante / oscilatoria).

        M'(t₀) se estima por diferencias centrales.  Un cero simple
        transversal es M(t₀)=0 ∧ |M'(t₀)| > ε.
        La distancia de splitting W^s–W^u es |M|_∞ / (‖f₀‖ + ε).
        """
        n = int(A.shape[0])
        n_periods = int(max(2, n_periods))
        if n == 0:
            z = np.zeros(1, dtype=np.float64)
            return MelnikovCertificate(
                melnikov_values=z, melnikov_derivative=z,
                simple_zeros=0, transverse_zeros=0,
                mean_magnitude=0.0, max_magnitude=0.0,
                splitting_distance=0.0,
                tangle_detected=False, is_admissible=True,
            )

        if blocks is None:
            blocks = self._extract_schur_blocks(T)

        T_strict = np.triu(T, k=1)          # f₀ : nilpotente
        T_blocks = T - T_strict             # f₁ : diagonal + 2×2
        # Plano dominante: columnas de Q asociadas al bloque de mayor |λ|.
        if blocks and Q.shape == (n, n):
            dom = max(blocks, key=lambda b: b.modulus)
            e0 = Q[:, dom.offset]
            e1 = Q[:, min(dom.offset + 1, n - 1)]
        else:
            e0 = np.zeros(n, dtype=np.float64)
            e0[0] = 1.0
            e1 = np.zeros(n, dtype=np.float64)
            if n > 1:
                e1[1] = 1.0
        # Semilla en el plano dominante.
        gamma = _safe_unit(e0 + 0.5 * e1)

        def _wedge2(u: NDArray[np.float64], w: NDArray[np.float64]) -> float:
            """2-forma det[⟨u,e0⟩ ⟨w,e0⟩; ⟨u,e1⟩ ⟨w,e1⟩] en el plano dominante."""
            return float(
                np.dot(u, e0) * np.dot(w, e1) - np.dot(u, e1) * np.dot(w, e0)
            )

        # Órbita y campos a lo largo de γ_k.
        orbit: List[NDArray[np.float64]] = []
        f0_along: List[NDArray[np.float64]] = []
        f1_along: List[NDArray[np.float64]] = []
        g = gamma.copy()
        horizon = min(n, n_periods)
        for k in range(horizon):
            # Representación en coordenadas de Schur: y = Qᵀ g, campos vía T.
            y = Q.T @ g if Q.shape == (n, n) else g
            f0_along.append(T_strict @ y)
            f1_along.append(T_blocks @ y)
            orbit.append(g.copy())
            g = A @ g
            nrm = float(np.linalg.norm(g))
            if nrm < _MACHINE_EPSILON:
                break
            g = g / nrm

        n_orb = len(orbit)
        if n_orb == 0:
            z = np.zeros(1, dtype=np.float64)
            return MelnikovCertificate(
                melnikov_values=z, melnikov_derivative=z,
                simple_zeros=0, transverse_zeros=0,
                mean_magnitude=0.0, max_magnitude=0.0,
                splitting_distance=0.0,
                tangle_detected=False, is_admissible=True,
            )

        omega = float(omega_perturbation)
        # Frecuencia natural del bloque dominante, si existe.
        if blocks:
            osc = [b for b in blocks if b.is_oscillatory]
            if osc:
                omega = float(max(b.frequency for b in osc))

        t_axis = np.arange(n_orb, dtype=np.float64)
        melnikov_vals = np.zeros(n_periods, dtype=np.float64)
        for j in range(n_periods):
            t0 = j * (2.0 * math.pi / max(n_periods, 1))
            acc = 0.0
            for k in range(n_orb):
                phase = math.cos(omega * t_axis[k] + t0)
                # f₁ modulada por la fase de Poincaré (perturbación periódica).
                acc += _wedge2(f0_along[k], f1_along[k] * phase) / (1.0 + k)
            melnikov_vals[j] = acc

        # Derivada central periódica.
        dM = np.zeros_like(melnikov_vals)
        dt = 2.0 * math.pi / max(n_periods, 1)
        for j in range(n_periods):
            dM[j] = (
                melnikov_vals[(j + 1) % n_periods]
                - melnikov_vals[(j - 1) % n_periods]
            ) / (2.0 * dt)

        signs = np.sign(melnikov_vals)
        # Ceros por cambio de signo (simples en la malla).
        dsign = np.diff(signs)
        simple_zeros = int(np.sum(dsign != 0))
        # Transversales: el segmento que cambia de signo tiene |M'| grande.
        trans_zeros = 0
        for j in range(n_periods - 1):
            if dsign[j] != 0 and max(abs(dM[j]), abs(dM[j + 1])) > _MELNIKOV_ZERO_TOL:
                trans_zeros += 1

        mean_mag = float(np.mean(np.abs(melnikov_vals)))
        max_mag = float(np.max(np.abs(melnikov_vals)))
        f0_scale = float(np.mean([np.linalg.norm(f) for f in f0_along]) + _MACHINE_EPSILON)
        splitting = float(max_mag / f0_scale)
        tangle = (simple_zeros > 0 and mean_mag > _WILKINSON_LIMIT)
        is_admissible = max_mag <= _MELNIKOV_MAX_ADMISSIBLE

        return MelnikovCertificate(
            melnikov_values=melnikov_vals,
            melnikov_derivative=dM,
            simple_zeros=simple_zeros,
            transverse_zeros=trans_zeros,
            mean_magnitude=mean_mag,
            max_magnitude=max_mag,
            splitting_distance=splitting,
            tangle_detected=tangle,
            is_admissible=is_admissible,
        )

    # ══════════════════════════════════════════════════════════════════════
    # §1.16  Condición diofantina de KAM sobre frecuencias de bloques
    # ══════════════════════════════════════════════════════════════════════
    def _diophantine_small_divisors(
        self,
        frequencies: Sequence[float],
        gamma: float = _KAM_DIOPHANTINE_GAMMA,
        tau: Optional[float] = None,
        k_max: int = 6,
    ) -> Tuple[bool, float]:
        r"""
        Verifica |⟨k, ω⟩| ≥ γ / |k|_1^τ  para 0 < |k|_1 ≤ k_max.

        Retorna (satisfied, min_|⟨k,ω⟩|).  Si hay < 2 frecuencias
        positivas se declara vacuamente diofantina (un solo toro).
        """
        omega = [float(w) for w in frequencies if float(w) > _SPECTRAL_GAP_FLOOR]
        if len(omega) < 2:
            return True, float(omega[0]) if omega else float("inf")
        d = len(omega)
        tau_eff = float(tau if tau is not None else (d - 1 + _KAM_DIOPHANTINE_TAU_OFFSET))
        min_div = float("inf")
        satisfied = True
        # Barrido de multi-índices en [-k_max, k_max]^d \ {0}, recortado.
        # Para d>3 se muestrea la esfera l1.
        from itertools import product
        rng_k = range(-k_max, k_max + 1)
        for kvec in product(rng_k, repeat=d):
            l1 = sum(abs(ki) for ki in kvec)
            if l1 == 0 or l1 > k_max:
                continue
            div = abs(sum(ki * om for ki, om in zip(kvec, omega)))
            if div < min_div:
                min_div = div
            bound = gamma / ((l1 ** tau_eff) + _MACHINE_EPSILON)
            if div < bound:
                satisfied = False
        if not math.isfinite(min_div):
            min_div = 0.0
        return satisfied, float(min_div)

    # ══════════════════════════════════════════════════════════════════════
    # §1.17  Certificado KAM — persistencia de autovalores cero
    # ══════════════════════════════════════════════════════════════════════
    def _certify_kam_persistence_from_schur(
        self,
        T: NDArray[np.float64],
        spectra: NilpotenceAuditData,
        blocks: Optional[Sequence[SchurBlock]] = None,
        kam_threshold: float = _KAM_PERSISTENCE_TOL,
    ) -> KAMCertificate:
        r"""
        ε = ‖T_offdiag‖_F / ω_min.

        Si ε < ε_KAM y las frecuencias de bloques 2×2 son diofantinas,
        los toros invariantes (subespacio nulo de A) persisten.
        El recuento de toros supervivientes se estima por
            N_tori = round( n · max(0, 1 − ε/ε_KAM) )
        penalizado si la condición diofantina falla.
        """
        T_offdiag_fro = float(np.linalg.norm(np.triu(T, k=1), ord="fro"))
        omega_min = max(float(spectra.spectral_radius), _SPECTRAL_GAP_FLOOR)
        # Si hay osciladores, ω_min es la menor frecuencia positiva.
        if blocks is None:
            blocks = self._extract_schur_blocks(T)
        osc_freqs = [b.frequency for b in blocks if b.is_oscillatory]
        if osc_freqs:
            omega_min = max(min(osc_freqs), _SPECTRAL_GAP_FLOOR)

        eps_pert = T_offdiag_fro / omega_min
        eps_KAM = max(float(kam_threshold), _KAM_PERSISTENCE_TOL)

        dio_ok, small_div = self._diophantine_small_divisors(osc_freqs)
        ratio = eps_pert / max(eps_KAM, _MACHINE_EPSILON)
        tori = max(0, int(round(spectra.dimension * max(0.0, 1.0 - ratio))))
        if not dio_ok:
            tori = max(0, tori // 2)

        is_stable = bool(eps_pert < eps_KAM and dio_ok)

        return KAMCertificate(
            epsilon_perturbation=float(eps_pert),
            epsilon_KAM_threshold=float(eps_KAM),
            diophantine_gamma=_KAM_DIOPHANTINE_GAMMA,
            diophantine_satisfied=dio_ok,
            small_divisor_min=float(small_div),
            tori_persisting=tori,
            is_KAM_stable=is_stable,
        )

    # ══════════════════════════════════════════════════════════════════════
    # §1.18  Difusión de Arnold — fuga entre resonancias + Nekhoroshev
    # ══════════════════════════════════════════════════════════════════════
    def _measure_arnold_diffusion_from_schur(
        self,
        T: NDArray[np.float64],
        spectra: NilpotenceAuditData,
        blocks: Optional[Sequence[SchurBlock]] = None,
        kam: Optional[KAMCertificate] = None,
    ) -> ArnoldDiffusionMetric:
        r"""
        D_Arnold = Σ_{bloques} |Δν_block|² / ω_block².

        Δν_block := traza del bloque (α-desviación; 0 si nilpotente exacta).
        ω_block  := frecuencia natural (o 1 para bloques 1×1).

        Tiempo de Nekhoroshev:
            t_Nek = exp( min(cap, c · ε^{-1/(2n)}) ),  c = 1/2.
        """
        n = int(T.shape[0])
        if n == 0:
            return ArnoldDiffusionMetric(
                per_block_leakage={}, total_diffusion=0.0,
                dominant_channel=None, nekhoroshev_time=float("inf"),
                is_adiabatic=True,
            )

        if blocks is None:
            blocks = self._extract_schur_blocks(T)

        per_block: Dict[str, float] = {}
        for blk in blocks:
            if blk.size == 2:
                omega = max(blk.frequency, _SPECTRAL_GAP_FLOOR)
                leakage = (blk.trace * blk.trace) / (omega * omega)
                key = f"block_{blk.index}_2x2"
            else:
                lam = abs(blk.real_part)
                leakage = (lam * lam) / (1.0 + lam * lam)
                key = f"block_{blk.index}_1x1"
            per_block[key] = float(leakage)

        total = float(sum(per_block.values()))
        dominant = (
            max(per_block.items(), key=lambda kv: kv[1])[0] if per_block else None
        )

        eps = float(kam.epsilon_perturbation) if kam is not None else total
        eps = max(eps, _MACHINE_EPSILON)
        exponent = 1.0 / float(max(2 * max(n, 1), 2))
        try:
            nek_arg = 0.5 * (eps ** (-exponent))
            nek_arg = min(nek_arg, _NEKHOROSHEV_EXP_CAP)
            t_nek = float(math.exp(nek_arg))
        except OverflowError:
            t_nek = float("inf")

        is_adiabatic = total < _ARNOLD_DIFFUSION_TOL

        return ArnoldDiffusionMetric(
            per_block_leakage=per_block,
            total_diffusion=total,
            dominant_channel=dominant,
            nekhoroshev_time=t_nek,
            is_adiabatic=is_adiabatic,
        )

    # ══════════════════════════════════════════════════════════════════════
    # §1.19  Semilla canónica de θ para el kernel N3  (puente de datos)
    # ══════════════════════════════════════════════════════════════════════
    def _n3_theta_seed_from_observe(
        self,
        spectra: NilpotenceAuditData,
        melnikov: MelnikovCertificate,
        kam: KAMCertificate,
        arnold: ArnoldDiffusionMetric,
        return_map: ReturnMapReport,
    ) -> Tuple[float, ...]:
        r"""
        Empaqueta la semilla θ₀ ∈ ℝ⁴ que Fase 2 ingerirá:

            θ₀ = ( ρ(A),  ν/n,  ε_KAM-pert,  ‖M‖_∞ )

        normalizada en ℓ².  No ejecuta N3: sólo fija la condición
        inicial del endofuntor de automejora (contrato Φ₁ → Φ₂).
        """
        nu = float(spectra.nilpotence_index) if spectra.nilpotence_index is not None else 0.0
        nrm_dim = float(max(spectra.dimension, 1))
        raw = np.array([
            float(spectra.spectral_radius),
            nu / nrm_dim,
            float(kam.epsilon_perturbation),
            float(melnikov.max_magnitude),
        ], dtype=np.float64)
        # Canales auxiliares (no forman parte de θ, pero se loguean).
        _ = (arnold.total_diffusion, return_map.stability_exponent)
        seed = raw / (float(np.linalg.norm(raw)) + _MACHINE_EPSILON)
        return tuple(float(x) for x in seed)

    # ══════════════════════════════════════════════════════════════════════
    # §1.20  Ensamblaje del ObserverPacket
    # ══════════════════════════════════════════════════════════════════════
    def _assemble_observer_packet(
        self,
        spectra: NilpotenceAuditData,
        sections: Sequence[PoincareSection],
        return_map: ReturnMapReport,
        melnikov: MelnikovCertificate,
        kam: KAMCertificate,
        arnold: ArnoldDiffusionMetric,
        blocks: Sequence[SchurBlock],
    ) -> ObserverPacket:
        r"""Sella el artefacto terminal de Fase 1 con huella SHA-256."""
        theta_seed = self._n3_theta_seed_from_observe(
            spectra, melnikov, kam, arnold, return_map,
        )
        ts = _utc_timestamp()
        payload = (
            f"{spectra.spectrum_certificate_hash}|"
            f"{len(sections)}|{return_map.stability_exponent:.12e}|"
            f"{melnikov.max_magnitude:.12e}|{kam.epsilon_perturbation:.12e}|"
            f"{arnold.total_diffusion:.12e}|{ts}"
        )
        return ObserverPacket(
            spectra=spectra,
            sections=tuple(sections),
            return_map=return_map,
            melnikov=melnikov,
            kam=kam,
            arnold=arnold,
            schur_blocks=tuple(blocks),
            n3_theta_seed=theta_seed,
            generated_at_utc=ts,
            packet_hash=_sha256_short(payload),
        )

    # ══════════════════════════════════════════════════════════════════════
    # §1.21  Stub funtorial Φ₁ → Φ₂  (implementado por Fase 2)
    # ══════════════════════════════════════════════════════════════════════
    def _audit_poset_filtration_from_nilpotence(
        self,
        nilpotence_audit: NilpotenceAuditData,
        edges: Sequence[Tuple[str, str]],
        node_strata: Mapping[str, Any],
        **kwargs: Any,
    ) -> "PosetFiltrationData":
        r"""
        Stub funtorial Φ₁ → Φ₂.

        Phase2_PosetFiltrationAuditor sobrescribe este método.  El puente
        terminal de Fase 1 lo invoca como *continuación formal* — no como
        dependencia opcional — de modo que Φ₂ ∘ Φ₁ es un morfismo estricto
        de la categoría de auditorías.
        """
        raise NotImplementedError(
            "Phase2_PosetFiltrationAuditor debe ser mezclado en la jerarquía. "
            "Este stub es el conector funtorial Φ₁ → Φ₂.  Continúe en Fase 2 "
            "con la definición efectiva de _audit_poset_filtration_from_nilpotence."
        )

    # ══════════════════════════════════════════════════════════════════════
    # §1.22  ÚLTIMO MÉTODO DE FASE 1: puente funtorial Φ₁ → Φ₂
    #        Su definición es la CONTINUACIÓN / INICIO de los métodos
    #        de Fase 2 (ORIENT : Poset DIKW + kernel N3).
    # ══════════════════════════════════════════════════════════════════════
    def _phase1_terminal_bridge_to_phase2(
        self,
        adjacency_matrix: NDArray[Any],
        edges: Sequence[Tuple[str, str]],
        node_strata: Mapping[str, Any],
        *,
        adjacency_zero_threshold: float = _BASE_SPECTRAL_TOLERANCE,
        allow_signed_weights: bool = False,
        require_boolean: bool = False,
        deep_nilpotence_audit: bool = True,
        power_audit_max_dim: int = _DEFAULT_POWER_AUDIT_MAX_DIM,
        allow_unknown_nodes: bool = False,
        strict_filtration: bool = False,
        node_order: Optional[Sequence[str]] = None,
        cross_check_adjacency_support: bool = False,
        raise_on_veto: bool = True,
        require_transverse_section: bool = False,
    ) -> Tuple[NilpotenceAuditData, "PosetFiltrationData", ObserverPacket]:
        r"""
        ÚLTIMO MÉTODO DE FASE 1 — Puente funtorial Φ₁ → Φ₂.

        Composición estricta:
            (1) Certificación espectral vía Schur → NilpotenceAuditData.
            (2) Reutilización de (T, Q, bloques) para la maquinaria de
                Poincaré: Σ, P, M, KAM, Arnold, Lyapunov–Benettin.
            (3) Ensamblaje y sellado del ObserverPacket (semilla N3).
            (4) Veto fail-secure de Melnikov / KAM / transversabilidad.
            (5) Invocación inmediata del primer método efectivo de
                Fase 2 (``_audit_poset_filtration_from_nilpotence``),
                que abre la categoría de ORIENT.

        Anidamiento:
            Φ₂ ∘ Φ₁ = _audit_poset_filtration_from_nilpotence
                      ∘ _phase1_terminal_bridge_to_phase2

        Retorna
            (NilpotenceAuditData, PosetFiltrationData, ObserverPacket).

        La definición de ``_audit_poset_filtration_from_nilpotence`` en
        Phase2_PosetFiltrationAuditor ES la continuación literal de este
        método: recibe ``observer_packet=...`` como semilla del kernel N3.
        """
        # ── Paso 1: Certificación espectral (Fase 1 base) ────────────────
        spectra = self._certify_spectral_nilpotence(
            adjacency_matrix,
            zero_threshold=adjacency_zero_threshold,
            allow_signed_weights=allow_signed_weights,
            require_boolean=require_boolean,
            deep_nilpotence_audit=deep_nilpotence_audit,
            power_audit_max_dim=power_audit_max_dim,
            raise_on_veto=raise_on_veto,
        )

        # ── Paso 2: Schur reutilizado para métricas de Poincaré ──────────
        A_cond = self._validate_and_condition_adjacency(
            adjacency_matrix,
            zero_threshold=adjacency_zero_threshold,
            allow_signed_weights=allow_signed_weights,
            require_boolean=require_boolean,
        )
        T, Q, _, _, _ = self._schur_decompose(A_cond)
        blocks = self._extract_schur_blocks(T)

        pow_tol = max(
            _BASE_POWER_TOLERANCE,
            _SCHUR_CONDITION_SAFETY_FACTOR
            * spectra.tolerance
            * max(1.0, spectra.adjacency_inf_norm),
        )

        sections = self._capture_poincare_section_from_schur(
            A_cond, T, Q, pow_tol, blocks=blocks,
        )
        lyap = self._lyapunov_spectrum_benettin(A_cond)
        return_map = self._analyze_return_map_from_schur(
            A_cond, sections, lyapunov_spectrum=lyap,
        )
        melnikov = self._evaluate_melnikov_function_from_schur(
            A_cond, T, Q, blocks=blocks,
        )
        kam = self._certify_kam_persistence_from_schur(
            T, spectra, blocks=blocks,
        )
        arnold = self._measure_arnold_diffusion_from_schur(
            T, spectra, blocks=blocks, kam=kam,
        )

        observer_packet = self._assemble_observer_packet(
            spectra, sections, return_map, melnikov, kam, arnold, blocks,
        )

        logger.info(
            "[Fase 1::OBSERVE] Σ=%d (trans=%s) | λ_Lya=%s | zeros=%d/%d | "
            "KAM=%s (dio=%s) | D_Arnold=%.4e | t_Nek=%.3e | hash=%s",
            len(sections),
            return_map.is_transverse_family,
            list(return_map.lyapunov_spectrum),
            melnikov.simple_zeros,
            melnikov.transverse_zeros,
            kam.is_KAM_stable,
            kam.diophantine_satisfied,
            arnold.total_diffusion,
            arnold.nekhoroshev_time,
            observer_packet.packet_hash,
        )

        # ── Paso 2.b: vetos fail-secure de mecánica celeste ──────────────
        if require_transverse_section and sections and not return_map.is_transverse_family:
            if raise_on_veto:
                raise PoincareTransversalityError(
                    "Familia de secciones no transversal: "
                    f"min |ḣ|={min(s.transversality for s in sections):.6e}."
                )
        if not melnikov.is_admissible and raise_on_veto:
            raise MelnikovTangleError(
                f"Tangencia homoclínica inadmisible: ‖M‖_∞="
                f"{melnikov.max_magnitude:.6e} > ε_M="
                f"{_MELNIKOV_MAX_ADMISSIBLE:.6e} "
                f"(splitting={melnikov.splitting_distance:.6e})."
            )
        if not kam.is_KAM_stable and raise_on_veto:
            raise KAMBreakdownError(
                f"Colapso KAM: ε_pert={kam.epsilon_perturbation:.6e} ≥ "
                f"ε_KAM={kam.epsilon_KAM_threshold:.6e} "
                f"(diofantina={kam.diophantine_satisfied}, "
                f"small_div={kam.small_divisor_min:.6e})."
            )

        # ── Paso 3: CONTINUACIÓN FUNTORIAL HACIA FASE 2 ──────────────────
        # La invocación de `_audit_poset_filtration_from_nilpotence` ES el
        # primer método efectivo de Fase 2, y se ejecuta aquí como
        # continuación formal del cierre de Fase 1.
        # ------------------------------------------------------------------
        # FASE 2 comienza exactamente en la definición efectiva de
        # Phase2_PosetFiltrationAuditor._audit_poset_filtration_from_nilpotence
        # (ORIENT : Poset DIKW + kernel N3 ← ObserverPacket.n3_theta_seed).
        # ------------------------------------------------------------------
        filtration_audit = self._audit_poset_filtration_from_nilpotence(
            spectra,
            edges,
            node_strata,
            allow_unknown_nodes=allow_unknown_nodes,
            strict_filtration=strict_filtration,
            node_order=node_order,
            adjacency_matrix=adjacency_matrix,
            cross_check_adjacency_support=cross_check_adjacency_support,
            adjacency_zero_threshold=adjacency_zero_threshold,
            raise_on_veto=raise_on_veto,
            observer_packet=observer_packet,
        )

        return spectra, filtration_audit, observer_packet


# ══════════════════════════════════════════════════════════════════════════════
# FIN DE FASE 1  ──  CONTINÚA EN FASE 2 (prompt siguiente)
# Phase2_PosetFiltrationAuditor(Phase1_SpectralNilpotenceCertifier)
#   debe abrir con _audit_poset_filtration_from_nilpotence  (Φ₂)
#   y cerrar con _phase2_terminal_bridge_to_phase3           (Φ₂ → Φ₃)
# ══════════════════════════════════════════════════════════════════════════════
# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FASE 2 ──► ORIENT : POSET DIKW + AUTOMEJORA RECURSIVA N3               █
# █   (filtración monótona + kernel (θ, O, α) + ciclo meta-meta)             █
# █                                                                          █
# █   CONTINUACIÓN FORMAL DE FASE 1                                          █
# █   Φ₂ abre en  _audit_poset_filtration_from_nilpotence                    █
# █   Φ₂ cierra en _phase2_terminal_bridge_to_phase3                         █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████
#
# Contrato de anidamiento (OODA-Orient):
#   Phase1._phase1_terminal_bridge_to_phase2
#       └─► Phase2._audit_poset_filtration_from_nilpotence   ← AQUÍ ARRANCA Φ₂
#               (ObserverPacket.n3_theta_seed → kernel N3)
#       └─► Phase2._phase2_terminal_bridge_to_phase3
#               └─► Phase3._intercept_mayer_vietoris_sequence  (prompt siguiente)
#
# Filosofía (Tikhonov / Fenichel, 3 escalas):
#   N1 (rápida)  : θ  — tolerancias del certificador espectral.
#   N2 (media)   : O  — topología DAG, Π_triu estricta ⇒ O^n = 0.
#   N3 (lenta)   : α  — meta-regla que gobierna las tasas de N1 y N2.
#   El flujo discreto Ψ = N3 ∘ N2 ∘ N1 es un endomorfismo del fibrado
#   (θ, O, α) → ObserverPacket; en un punto crítico, Ψ ∘ Ψ = Ψ (idempotencia).
# ══════════════════════════════════════════════════════════════════════════════


# ── 2.0 DTOs DE FASE 2 ──────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class EdgeSlackRecord:
    r"""
    Arista auditada del Poset DIKW.

    slack(u,v) := stratum(v) − stratum(u).
    Filtración débil  : slack ≥ 0.
    Filtración estricta: slack ≥ 1  (cubre estrictamente el orden).
    """
    source: str
    target: str
    stratum_u: int
    stratum_v: int
    slack: int
    is_violation: bool


@dataclass(frozen=True, slots=True)
class PosetHeightCertificate:
    r"""
    Certificado de altura vs. índice de nilpotencia.

    Teorema (poset rankeado + DAG): si f: (V, →) → (ℤ, ≤) es un
    morfismo de posets y las aristas cubren (slack ≥ 1), entonces
        longest_path ≤ height := max f − min f,
        ν(A) ≤ longest_path + 1 ≤ height + 1 ≤ n.
    La consistencia falla si ν certificado > height + 1 (incoherencia
    entre el espectro de Schur y la filtración declarada).
    """
    n_nodes: int
    min_rank: Optional[int]
    max_rank: Optional[int]
    height: Optional[int]
    longest_path: Optional[int]
    nilpotence_index: Optional[int]
    consistent_with_nu: bool
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TransitiveClosureData:
    r"""Clausura transitiva del DAG de aristas (Warshall booleano)."""
    n_nodes: int
    n_edges_declared: int
    n_reachable_pairs: int
    n_implied_edges: int
    has_cycle: bool
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PosetFiltrationData:
    r"""
    Artefacto de Fase 2 — certificado de monotonicidad del Poset DIKW.

    Extensión v3.1: altura, camino más largo, consistencia con ν(A)
    y huella del ObserverPacket que sembró el kernel N3.
    """
    edge_count:                   int
    audited_edge_count:           int
    ignored_edge_count:           int
    min_slack:                    Optional[int]
    max_slack:                    Optional[int]
    is_monotonic_filtration:      bool
    self_loops_detected:          int = 0
    unknown_nodes:                Tuple[str, ...] = ()
    isolated_strata_nodes:        Tuple[str, ...] = ()
    inverse_edges_blocked:        Tuple[str, ...] = ()
    slack_histogram:              Tuple[Tuple[int, int], ...] = ()
    poset_height:                 Optional[int] = None
    longest_path:                 Optional[int] = None
    height_nilpotence_consistent: Optional[bool] = None
    transitive_edge_count:        int = 0
    observer_packet_hash:         Optional[str] = None
    notes:                        Tuple[str, ...] = ()


@dataclass
class ImprovementState:
    r"""
    Estado interno mutable del kernel N3.

    θ ∈ ℝ^d     : parámetros del certificador (tolerancias, umbrales).
    O ∈ ℝ^{n×n} : topología DAG (triangular estricta ⇒ nilpotente).
    α ∈ ℝ^3     : meta-parámetros (tasas de N1, N2, N3).
    """
    theta:          NDArray[np.float64]
    O:              NDArray[np.float64]
    alpha:          NDArray[np.float64]
    loss_history:   List[float] = field(default_factory=list)
    level_history:  List[int] = field(default_factory=list)
    ingested:       bool = False
    packet_hash:    str = ""


@dataclass(frozen=True, slots=True)
class N3ConvergenceCertificate:
    r"""
    Certificado de convergencia del endomorfismo Ψ = N3 ∘ N2 ∘ N1.

    monotone_decrease : L_{c+1} ≤ L_c + ε_num  ∀ c.
    idempotent        : ‖Ψ(z*) − z*‖ ≤ ε_idem  (punto fijo numérico).
    lipschitz_grad    : cota empírica de Lip(∇L) en el último ciclo.
    """
    n_cycles:               int
    initial_loss:           float
    final_loss:             float
    monotone_decrease:      bool
    idempotent:             bool
    residual_fixed_point:   float
    lipschitz_grad_bound:   float
    finite:                 bool
    level_sequence:         Tuple[int, ...]
    notes:                  Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class OrientationSeed:
    r"""
    ← ÚLTIMO ARTEFACTO DE FASE 2.

    Transporta (θ*, α*, ‖O*‖, certificado N3, banderas KAM/Melnikov)
    hacia Fase 3, donde alimentan el veredicto terminal Ω₃.

    Invariante de empaquetado:
        OrientationSeed ⊢ filtración monótona ∧ N3 finito
                        ∧ ObserverPacket sellado.
    """
    theta_final:        NDArray[np.float64]
    alpha_final:        NDArray[np.float64]
    O_norm:             float
    final_loss:         float
    observer_packet:    ObserverPacket
    kam_preserved:      bool
    melnikov_chaotic:   bool
    n3_report:          Dict[str, Any]
    n3_certificate:     N3ConvergenceCertificate
    poset_height:       Optional[int]
    idempotent:         bool
    packet_hash:        str
    generated_at_utc:   str


# ── 2.1 KERNEL DE AUTOMEJORA RECURSIVA N3 ───────────────────────────────────

class RecursiveSelfImprovementKernel:
    r"""
    Kernel N3 de automejora recursiva (θ, O, α).

    Ecuaciones de evolución (proyecciones sobre conjuntos factibles):

      N1: θ_{t+1} = Π_Θ ( θ_t − τ₁ α[0] ∇_θ L )          τ₁ ∈ (0,1] Armijo
      N2: O_{t+1} = Π_triu⁺ ( O_t − τ₂ α[1] ∇_O L )      aciclicidad por cons.
      N3: α_{t+1} = Π_[α̲,ᾱ] ( α_t − τ₃ α[2] ∇_α L^↺ )   L^↺ = lookahead

    Pérdida compuesta (informada por el ObserverPacket de Fase 1):

      L = w₁ (1 − ν(O)/n)
        + w₂ (−λ₂(O)/λ₂^{max})
        + w₃ D_Arnold (1 + ‖O‖_F)
        + w₄ ‖M‖_∞ (1 + ‖θ‖₂)
        + w₅ (1 − 1_{KAM}) (1 + ε_pert)
        + w₆ (−λ_Lya)_+ 
        + (μ/2) ‖θ − θ₀‖₂²
        + λ_s ‖O‖₁
        − η ⟨α, g_energy⟩
        + (ρ/2) ‖α‖₂²

    Complejidad por paso N2: O(n²) diferencias finitas centrales,
    n ≤ n_cap (proyección de Galerkin del DAG).
    """

    _THETA_FLOOR: float = 1.0e-12
    _ARMIJO_C: float = 1.0e-4
    _ARMIJO_BETA: float = 0.5
    _ARMIJO_MAX_BT: int = 8
    _LOOKAHEAD_ETA: float = 1.0e-2
    _THETA_REG: float = 1.0e-3
    _ALPHA_REG: float = 1.0e-2
    _SPARSITY_DEFAULT: float = 0.05

    def __init__(
        self,
        n_params: int = 4,
        adjacency_shape: int = 8,
        alpha_init: Optional[NDArray[np.float64]] = None,
        learning_rate: float = 1e-2,
        weights: Optional[Tuple[float, float, float, float, float, float]] = None,
    ) -> None:
        rng = np.random.default_rng(seed=_SEED_N3)
        self._n_params = int(max(1, n_params))
        self._adj_shape = int(max(1, adjacency_shape))

        theta = rng.uniform(0.1, 1.0, size=self._n_params).astype(np.float64)
        O = np.triu(
            rng.uniform(0.0, 0.05, size=(self._adj_shape, self._adj_shape)),
            k=1,
        ).astype(np.float64)
        if alpha_init is None:
            alpha_init = np.array([1e-2, 1e-3, 1e-4], dtype=np.float64)
        else:
            alpha_init = np.asarray(alpha_init, dtype=np.float64).reshape(-1)
            if alpha_init.size < 3:
                raise RecursiveSelfImprovementError(
                    f"alpha_init debe tener dimensión ≥ 3. size={alpha_init.size}."
                )

        self.state = ImprovementState(
            theta=theta, O=O, alpha=alpha_init[:3].copy(),
        )
        self.lr = float(learning_rate)
        # (ν, Fiedler, Arnold, Melnikov, KAM, Lyapunov)
        self.weights = weights or (1.0, 1.0, 0.5, 0.5, 0.5, 0.25)
        self.logger = logging.getLogger(self.__class__.__name__)

        self._packet: Optional[ObserverPacket] = None
        self._theta_seed: Optional[NDArray[np.float64]] = None
        self._poset_mask: Optional[NDArray[np.float64]] = None
        self._last_g_theta: Optional[NDArray[np.float64]] = None
        self._last_g_O: Optional[NDArray[np.float64]] = None
        self._last_g_alpha: Optional[NDArray[np.float64]] = None

    # ── 2.1.1 Ingesta del ObserverPacket (entrada canónica desde FASE 1) ─
    def ingest_observer_packet(self, packet: ObserverPacket) -> None:
        r"""
        Traduce ObserverPacket → condición inicial (θ₀, O₀, α₀).

        θ₀ ← n3_theta_seed  (ya ℓ²-normalizado en Fase 1)
             = (ρ, ν/n, ε_pert, ‖M‖_∞)̂
        O₀ ← triu(ε) perturbada por fuga de Arnold por bloque Schur
             y por acoplamientos 2×2 oscilatorios (frecuencia ω).
        α₀ ← (η₁ · 1_{KAM},  η₂ · (1+caos),  η₃ · (1+2 caos))
        """
        if packet is None:
            raise RecursiveSelfImprovementError(
                "ingest_observer_packet: ObserverPacket es None."
            )
        self._packet = packet
        self.state.packet_hash = packet.packet_hash
        self.state.ingested = True

        seed = np.asarray(packet.n3_theta_seed, dtype=np.float64).reshape(-1)
        if seed.size == 0:
            s = packet.spectra
            nu = float(s.nilpotence_index) if s.nilpotence_index is not None else 0.0
            seed = np.array([
                float(s.spectral_radius),
                nu / float(max(s.dimension, 1)),
                float(packet.kam.epsilon_perturbation),
                float(packet.melnikov.max_magnitude),
            ], dtype=np.float64)
            seed = seed / (float(np.linalg.norm(seed)) + _MACHINE_EPSILON)
        if seed.size < self._n_params:
            seed = np.pad(seed, (0, self._n_params - seed.size))
        self.state.theta = seed[: self._n_params].copy()
        self._theta_seed = self.state.theta.copy()

        n = self._adj_shape
        base = np.triu(np.eye(n, k=1) * 0.05, k=1).astype(np.float64)
        for i, (_, leak) in enumerate(packet.arnold.per_block_leakage.items()):
            if i < n - 1:
                base[i, i + 1] += float(leak) * 0.1
        # Acoplamiento 2×2 de bloques oscilatorios de Schur.
        for blk in packet.schur_blocks:
            i0 = int(blk.offset)
            if blk.size == 2 and i0 + 1 < n:
                base[i0, i0 + 1] += 0.05 * float(blk.frequency)
        self.state.O = np.triu(np.maximum(base, 0.0), k=1)
        if self._poset_mask is not None:
            self.state.O = self.state.O * self._poset_mask

        chaos = min(1.0, float(packet.melnikov.max_magnitude) * 1.0e3)
        kam_m = 1.0 if packet.kam.is_KAM_stable else 0.1
        trans_m = 1.0 if packet.return_map.is_transverse_family else 1.5
        self.state.alpha = np.array([
            1.0e-2 * kam_m / trans_m,
            1.0e-3 * (1.0 + chaos),
            1.0e-4 * (1.0 + chaos * 2.0),
        ], dtype=np.float64)
        self.state.alpha = np.clip(self.state.alpha, _N3_ALPHA_FLOOR, _N3_ALPHA_CEIL)

        L0 = self._loss(self.state.theta, self.state.O, self.state.alpha)
        self.state.loss_history.append(float(L0))
        self.state.level_history.append(0)

        self.logger.info(
            "[N3::ingest] hash=%s | θ=%s | α=%s | L0=%.6e | chaos=%.3f | KAM=%s",
            packet.packet_hash,
            np.round(self.state.theta, 4).tolist(),
            np.round(self.state.alpha, 6).tolist(),
            L0, chaos, packet.kam.is_KAM_stable,
        )

    # ── 2.1.2 Máscara de filtración (N2 no puede violar DIKW) ───────────
    def bind_poset_mask(self, mask: NDArray[np.float64]) -> None:
        r"""
        mask[i,j] ∈ {0,1}: 1 ⇔ la arista i→j es compatible con la
        filtración (stratum[j] ≥ stratum[i], i ≠ j).  Se aplica en
        cada proyección Π_triu⁺.
        """
        M = np.asarray(mask, dtype=np.float64)
        if M.shape != (self._adj_shape, self._adj_shape):
            raise RecursiveSelfImprovementError(
                f"poset_mask shape={M.shape} ≠ ({self._adj_shape},{self._adj_shape})."
            )
        self._poset_mask = (M > 0.5).astype(np.float64)
        self.state.O = np.triu(self.state.O * self._poset_mask, k=1)

    # ── 2.1.3 Términos de la pérdida ─────────────────────────────────────
    def _nilpotence_index_of(self, O: NDArray[np.float64]) -> int:
        """ν(O) = min{k : ‖O^k‖_F ≤ ε_W}.  Si no colapsa, retorna n."""
        n = int(O.shape[0])
        if n == 0:
            return 0
        P = np.eye(n, dtype=np.float64)
        for k in range(1, n + 1):
            P = P @ O
            if float(np.linalg.norm(P, ord="fro")) < _WILKINSON_LIMIT:
                return k
        return n

    def _fiedler_ratio(self, O: NDArray[np.float64]) -> float:
        r"""−λ₂/λ_max del Laplaciano simetrizado de |O| (conectividad)."""
        n = int(O.shape[0])
        if n < 2:
            return 0.0
        absO = np.abs(O)
        degree = np.sum(absO, axis=1)
        L = np.diag(degree) - absO
        L_sym = 0.5 * (L + L.T)
        try:
            evals = np.sort(np.real(np.linalg.eigvalsh(L_sym)))
            lam2 = float(evals[1]) if evals.size > 1 else 0.0
            lam_max = max(float(evals[-1]), _MACHINE_EPSILON)
        except Exception:
            return 0.0
        return -lam2 / lam_max

    def _loss(
        self,
        theta: NDArray[np.float64],
        O: NDArray[np.float64],
        alpha: NDArray[np.float64],
    ) -> float:
        w = self.weights
        w1, w2, w3, w4 = w[0], w[1], w[2], w[3]
        w5 = w[4] if len(w) > 4 else 0.5
        w6 = w[5] if len(w) > 5 else 0.25

        n = int(O.shape[0])
        nu = self._nilpotence_index_of(O)
        nu_term = 1.0 - (nu / float(max(n, 1)))
        fiedler_term = self._fiedler_ratio(O)

        pkt = self._packet
        O_fro = float(np.linalg.norm(O, ord="fro"))
        th_n = float(np.linalg.norm(theta))
        if pkt is not None:
            arnold_term = float(pkt.arnold.total_diffusion) * (1.0 + O_fro)
            melnikov_term = float(pkt.melnikov.max_magnitude) * (1.0 + th_n)
            kam_term = (0.0 if pkt.kam.is_KAM_stable else 1.0) * (
                1.0 + float(pkt.kam.epsilon_perturbation)
            )
            lya = float(pkt.return_map.stability_exponent)
            lya_term = max(0.0, lya)  # penaliza λ ≥ 0 (ciclos persistentes)
        else:
            arnold_term = float(np.std(alpha)) * float(np.std(theta))
            melnikov_term = abs(math.sin(math.pi * th_n))
            kam_term = 0.0
            lya_term = 0.0

        # Regularización hacia la semilla de Fase 1 y sparsidad de O.
        if self._theta_seed is not None and self._theta_seed.shape == theta.shape:
            theta_reg = 0.5 * self._THETA_REG * float(
                np.sum((theta - self._theta_seed) ** 2)
            )
        else:
            theta_reg = 0.5 * self._THETA_REG * float(np.sum(theta ** 2))
        spars = self._SPARSITY_DEFAULT * float(np.sum(np.abs(O)))

        # Energía de descenso de primer orden (N3 quiere α alineada a ∇).
        g_energy = 0.0
        if self._last_g_theta is not None:
            g_energy += float(alpha[0]) * float(np.sum(self._last_g_theta ** 2))
        if self._last_g_O is not None:
            g_energy += float(alpha[1]) * float(np.sum(self._last_g_O ** 2))
        alpha_reg = 0.5 * self._ALPHA_REG * float(np.sum(alpha ** 2))

        value = (
            w1 * nu_term
            + w2 * fiedler_term
            + w3 * arnold_term
            + w4 * melnikov_term
            + w5 * kam_term
            + w6 * lya_term
            + theta_reg
            + spars
            - self._LOOKAHEAD_ETA * g_energy
            + alpha_reg
        )
        if not math.isfinite(value):
            return float("inf")
        return float(value)

    # ── 2.1.4 Gradiente numérico central ────────────────────────────────
    @staticmethod
    def _grad_numeric(
        f: Callable[[NDArray[np.float64]], float],
        x: NDArray[np.float64],
        h: float = 1e-6,
    ) -> NDArray[np.float64]:
        g = np.zeros_like(x, dtype=np.float64)
        if x.ndim == 1:
            for i in range(x.shape[0]):
                xp = x.copy(); xp[i] += h
                xm = x.copy(); xm[i] -= h
                fp, fm = f(xp), f(xm)
                if math.isfinite(fp) and math.isfinite(fm):
                    g[i] = (fp - fm) / (2.0 * h)
        else:
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    xp = x.copy(); xp[i, j] += h
                    xm = x.copy(); xm[i, j] -= h
                    fp, fm = f(xp), f(xm)
                    if math.isfinite(fp) and math.isfinite(fm):
                        g[i, j] = (fp - fm) / (2.0 * h)
        return g

    # ── 2.1.5 Proyecciones factibles ─────────────────────────────────────
    def _project_theta(self, theta: NDArray[np.float64]) -> NDArray[np.float64]:
        t = np.asarray(theta, dtype=np.float64).reshape(-1)
        if t.size < self._n_params:
            t = np.pad(t, (0, self._n_params - t.size))
        t = np.maximum(t[: self._n_params], self._THETA_FLOOR)
        nrm = float(np.linalg.norm(t))
        if nrm > 8.0:
            t = t * (8.0 / nrm)
        return t

    def _project_O(self, O: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Π_triu⁺ : anula diagonal e inferior, recorta negativos, aplica máscara DIKW."""
        P = np.triu(np.maximum(np.asarray(O, dtype=np.float64), 0.0), k=1)
        if self._poset_mask is not None:
            P = P * self._poset_mask
        return P

    def _project_alpha(self, alpha: NDArray[np.float64]) -> NDArray[np.float64]:
        a = np.asarray(alpha, dtype=np.float64).reshape(-1)
        if a.size < 3:
            a = np.pad(a, (0, 3 - a.size), constant_values=_N3_ALPHA_FLOOR)
        return np.clip(a[:3], _N3_ALPHA_FLOOR, _N3_ALPHA_CEIL)

    # ── 2.1.6 Búsqueda lineal de Armijo ──────────────────────────────────
    def _armijo(
        self,
        x: NDArray[np.float64],
        grad: NDArray[np.float64],
        step0: float,
        loss_at: Callable[[NDArray[np.float64]], float],
        project: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    ) -> Tuple[NDArray[np.float64], float, float]:
        r"""
        x⁺ = Π(x − τ s ∇L) con τ ← β τ hasta
            L(x⁺) ≤ L(x) − c τ s ‖∇L‖².
        Retorna (x⁺, L(x⁺), τ_usado).
        """
        L0 = loss_at(x)
        g2 = float(np.sum(grad * grad))
        if (not math.isfinite(L0)) or g2 < _MACHINE_EPSILON:
            return project(x), L0, 0.0
        tau = 1.0
        s = float(max(step0, _MACHINE_EPSILON))
        x_best, L_best = project(x), L0
        for _ in range(self._ARMIJO_MAX_BT):
            cand = project(x - (tau * s) * grad)
            Lc = loss_at(cand)
            if math.isfinite(Lc) and Lc <= L0 - self._ARMIJO_C * tau * s * g2:
                return cand, float(Lc), float(tau)
            if math.isfinite(Lc) and Lc < L_best:
                x_best, L_best = cand, Lc
            tau *= self._ARMIJO_BETA
        return x_best, float(L_best), float(tau)

    # ── 2.1.7 Nivel 1: optimización de θ ────────────────────────────────
    def step_level_1(self, max_iter: int = 10) -> Tuple[NDArray[np.float64], float]:
        r"""θ_{t+1} = Π_Θ ( θ_t − τ α[0] ∇_θ L(θ_t, O_t, α_t) )."""
        theta = self.state.theta.copy()
        loss_prev = self._loss(theta, self.state.O, self.state.alpha)
        last_g = np.zeros_like(theta)
        for _ in range(max_iter):
            grad = self._grad_numeric(
                lambda th: self._loss(th, self.state.O, self.state.alpha),
                theta,
            )
            last_g = grad
            theta, loss_new, _ = self._armijo(
                theta, grad, float(self.state.alpha[0]),
                lambda th: self._loss(th, self.state.O, self.state.alpha),
                self._project_theta,
            )
            if abs(loss_new - loss_prev) < _N3_IDEMPOTENCE_TOL:
                loss_prev = loss_new
                break
            loss_prev = loss_new
        self.state.theta = theta
        self._last_g_theta = last_g
        self.state.loss_history.append(float(loss_prev))
        self.state.level_history.append(1)
        return theta, float(loss_prev)

    # ── 2.1.8 Nivel 2: optimización de O (topología DAG) ────────────────
    def step_level_2(
        self,
        max_iter: int = 5,
        sparsity_penalty: float = 0.05,
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        O_{t+1} = Π_triu⁺ ( O_t − τ α[1] ∇_O (L + λ‖O‖₁) ).
        Π_triu⁺ garantiza aciclicidad y monotonicidad DIKW (máscara).
        """
        O = self.state.O.copy()
        lam = float(sparsity_penalty)

        def L_O(o: NDArray[np.float64]) -> float:
            return self._loss(self.state.theta, o, self.state.alpha) + lam * float(
                np.sum(np.abs(o))
            )

        loss_prev = L_O(O)
        last_g = np.zeros_like(O)
        for _ in range(max_iter):
            grad = self._grad_numeric(L_O, O)
            last_g = grad
            O, loss_new, _ = self._armijo(
                O, grad, float(self.state.alpha[1]), L_O, self._project_O,
            )
            if abs(loss_new - loss_prev) < _N3_IDEMPOTENCE_TOL:
                loss_prev = loss_new
                break
            loss_prev = loss_new
        self.state.O = O
        self._last_g_O = last_g
        # Historial reporta L sin el término extra de sparsidad duplicado.
        L_clean = self._loss(self.state.theta, O, self.state.alpha)
        self.state.loss_history.append(float(L_clean))
        self.state.level_history.append(2)
        return O, float(L_clean)

    # ── 2.1.9 Nivel 3: modificación de la meta-regla α (lookahead) ──────
    def step_level_3(self, max_iter: int = 3) -> Tuple[NDArray[np.float64], float]:
        r"""
        α_{t+1} = Π ( α_t − τ α[2] ∇_α L^↺(θ*, O*, α) ).

        L^↺ es la pérdida con energía de descenso de primer orden
        (gradientes cacheados de N1/N2): la meta-regla se juzga por
        cuánto descenso induce, no por el valor estático de L.
        Autoreferencia: el propio α[2] escala el paso sobre α (meta-meta).
        """
        alpha = self.state.alpha.copy()
        loss_prev = self._loss(self.state.theta, self.state.O, alpha)
        last_g = np.zeros_like(alpha)
        for _ in range(max_iter):
            grad = self._grad_numeric(
                lambda a: self._loss(self.state.theta, self.state.O, a),
                alpha,
            )
            last_g = grad
            # Escala auto-referente: paso = α[2].
            alpha, loss_new, _ = self._armijo(
                alpha, grad, float(alpha[2]),
                lambda a: self._loss(self.state.theta, self.state.O, a),
                self._project_alpha,
            )
            if abs(loss_new - loss_prev) < _N3_IDEMPOTENCE_TOL:
                loss_prev = loss_new
                break
            loss_prev = loss_new
        self.state.alpha = alpha
        self._last_g_alpha = last_g
        self.state.loss_history.append(float(loss_prev))
        self.state.level_history.append(3)
        return alpha, float(loss_prev)

    # ── 2.1.10 Cota empírica de Lip(∇L) ─────────────────────────────────
    def _empirical_grad_lipschitz(self) -> float:
        """‖∇L(z) − ∇L(z+δ)‖ / ‖δ‖ sobre una perturbación Haar de θ."""
        theta = self.state.theta
        rng = np.random.default_rng(seed=_SEED_N3 + 7)
        delta = rng.standard_normal(theta.shape).astype(np.float64)
        delta *= 1.0e-5 / (float(np.linalg.norm(delta)) + _MACHINE_EPSILON)
        g0 = self._grad_numeric(
            lambda th: self._loss(th, self.state.O, self.state.alpha), theta,
        )
        g1 = self._grad_numeric(
            lambda th: self._loss(th, self.state.O, self.state.alpha),
            theta + delta,
        )
        num = float(np.linalg.norm(g1 - g0))
        den = float(np.linalg.norm(delta)) + _MACHINE_EPSILON
        return num / den

    # ── 2.1.11 Idempotencia de Ψ en un clon ─────────────────────────────
    def _fixed_point_residual(self) -> float:
        r"""
        ‖Ψ(z) − z‖ combinada (θ, vec(O), α) sobre UN ciclo en un clon
        del estado; no muta el kernel vivo.
        """
        snap_th = self.state.theta.copy()
        snap_O = self.state.O.copy()
        snap_a = self.state.alpha.copy()
        hist_l = list(self.state.loss_history)
        hist_v = list(self.state.level_history)
        gth, gO, ga = self._last_g_theta, self._last_g_O, self._last_g_alpha
        try:
            self.step_level_1(max_iter=1)
            self.step_level_2(max_iter=1)
            self.step_level_3(max_iter=1)
            d_th = float(np.linalg.norm(self.state.theta - snap_th))
            d_O = float(np.linalg.norm(self.state.O - snap_O))
            d_a = float(np.linalg.norm(self.state.alpha - snap_a))
            nrm_z = (
                float(np.linalg.norm(snap_th))
                + float(np.linalg.norm(snap_O))
                + float(np.linalg.norm(snap_a))
                + _MACHINE_EPSILON
            )
            return (d_th + d_O + d_a) / nrm_z
        finally:
            self.state.theta = snap_th
            self.state.O = snap_O
            self.state.alpha = snap_a
            self.state.loss_history = hist_l
            self.state.level_history = hist_v
            self._last_g_theta, self._last_g_O, self._last_g_alpha = gth, gO, ga

    # ── 2.1.12 Bucle N3 completo ─────────────────────────────────────────
    def optimize_recursive_n3(
        self,
        n_cycles: int = _N3_DEFAULT_CYCLES,
    ) -> Tuple[Dict[str, Any], N3ConvergenceCertificate]:
        r"""
        Ejecuta n_cycles del bucle anidado:
            for c in range(n_cycles):
                N1: θ* ← argmin L(θ, O, α)
                N2: O* ← argmin L(θ*, O, α)
                N3: α* ← argmin L^↺(θ*, O*, α)
        """
        if not self.state.ingested:
            raise RecursiveSelfImprovementError(
                "Kernel N3 invocado sin ingest_observer_packet previo."
            )
        n_cycles = int(max(1, min(int(n_cycles), 32)))
        L_init = (
            float(self.state.loss_history[0]) if self.state.loss_history else self._loss(
                self.state.theta, self.state.O, self.state.alpha,
            )
        )
        cycle_losses: List[float] = [L_init]
        notes: List[str] = []

        for c in range(n_cycles):
            self.step_level_1()
            self.step_level_2()
            _, L_c = self.step_level_3()
            cycle_losses.append(float(L_c))
            self.logger.info(
                "[N3::cycle %d/%d] loss=%.6e | ‖θ‖=%.4f | ‖α‖=%.4f | ‖O‖_F=%.4e",
                c + 1, n_cycles, L_c,
                float(np.linalg.norm(self.state.theta)),
                float(np.linalg.norm(self.state.alpha)),
                float(np.linalg.norm(self.state.O, ord="fro")),
            )

        final_loss = float(cycle_losses[-1])
        if not math.isfinite(final_loss):
            raise RecursiveSelfImprovementError(
                f"Kernel N3 divergió: final_loss={final_loss}."
            )

        monotone = all(
            cycle_losses[i + 1] <= cycle_losses[i] + 10.0 * _N3_IDEMPOTENCE_TOL
            for i in range(len(cycle_losses) - 1)
        )
        if not monotone:
            notes.append("Pérdida no monótona en algún ciclo (N3 exploratorio).")

        residual = self._fixed_point_residual()
        idem = residual <= _N3_IDEMPOTENCE_TOL
        if idem:
            notes.append(f"Idempotencia numérica: ‖Ψz−z‖={residual:.3e}.")
        else:
            notes.append(f"Punto fijo inexacto: ‖Ψz−z‖={residual:.3e}.")

        try:
            lip = self._empirical_grad_lipschitz()
        except Exception:
            lip = float("inf")

        cert = N3ConvergenceCertificate(
            n_cycles=n_cycles,
            initial_loss=float(L_init),
            final_loss=final_loss,
            monotone_decrease=bool(monotone),
            idempotent=bool(idem),
            residual_fixed_point=float(residual),
            lipschitz_grad_bound=float(lip),
            finite=True,
            level_sequence=tuple(self.state.level_history),
            notes=tuple(notes),
        )
        report: Dict[str, Any] = {
            "theta": self.state.theta.tolist(),
            "alpha": self.state.alpha.tolist(),
            "O_norm": float(np.linalg.norm(self.state.O, ord="fro")),
            "nilpotence_index_O": self._nilpotence_index_of(self.state.O),
            "final_loss": final_loss,
            "loss_history": list(self.state.loss_history),
            "level_history": list(self.state.level_history),
            "cycle_losses": cycle_losses,
            "packet_hash": self.state.packet_hash,
            "idempotent": bool(idem),
            "fixed_point_residual": float(residual),
        }
        return report, cert


# ── 2.2 CLASE FASE 2 ────────────────────────────────────────────────────────

class Phase2_PosetFiltrationAuditor(Phase1_SpectralNilpotenceCertifier):
    r"""
    Auditor de monotonicidad del Poset DIKW + kernel N3.

    Cadena funtorial:
        Φ₂ : (ObserverPacket, Poset) → (FiltrationCert, OrientationSeed).

    El PRIMER método efectivo sobrescribe el stub de Fase 1 y es la
    continuación literal de `_phase1_terminal_bridge_to_phase2`.
    """

    # ══════════════════════════════════════════════════════════════════════
    # §2.1  PRIMER MÉTODO EFECTIVO DE FASE 2 — continuación de Φ₁
    #       Sobrescribe el stub Phase1._audit_poset_filtration_from_nilpotence
    # ══════════════════════════════════════════════════════════════════════
    def _audit_poset_filtration_from_nilpotence(
        self,
        nilpotence_audit: NilpotenceAuditData,
        edges: Sequence[Tuple[str, str]],
        node_strata: Mapping[str, Any],
        *,
        allow_unknown_nodes: bool = False,
        strict_filtration: bool = False,
        raise_on_veto: bool = True,
        node_order: Optional[Sequence[str]] = None,
        adjacency_matrix: Optional[NDArray[Any]] = None,
        cross_check_adjacency_support: bool = False,
        adjacency_zero_threshold: float = _BASE_SPECTRAL_TOLERANCE,
        observer_packet: Optional[ObserverPacket] = None,
    ) -> PosetFiltrationData:
        r"""
        ← ARRANQUE FORMAL DE FASE 2 (ORIENT).

        Auditoría de monotonicidad del morfismo de posets
            f : (V, →_DAG) ⟶ (ℤ, ≤) ,  u → v  ⇒  f(u) ≤ f(v)
        (desigualdad estricta si ``strict_filtration``).

        Pipeline interno (granular):
            (1) normalizar aristas y rechazar auto-bucles;
            (2) cross-check opcional edges ↔ soporte de A;
            (3) precondición espectral de Fase 1 (Spec(A)={0});
            (4) coerción de estratos;
            (5) slacks, histograma, violaciones;
            (6) nodos aislados / desconocidos;
            (7) camino más largo + clausura transitiva;
            (8) consistencia height ↔ ν(A)  (teorema de rango);
            (9) anotación del ObserverPacket (semilla N3, aún no ejecutada).

        El kernel N3 NO se ejecuta aquí: se inicializa en
        `_initialize_n3_kernel_from_observer` y se dispara desde el
        puente terminal `_phase2_terminal_bridge_to_phase3`.
        """
        notes: List[str] = []

        # (1) Aristas.
        edges_norm, n_self_loops = self._normalize_edges(
            edges, raise_on_self_loop=raise_on_veto,
        )
        if n_self_loops > 0:
            notes.append(f"Se detectaron {n_self_loops} auto-bucle(s) excluidos.")

        # (2) Cross-check de soporte.
        inverse_blocked: Tuple[str, ...] = ()
        if cross_check_adjacency_support:
            try:
                ok_notes, inverse_blocked = self._audit_adjacency_support(
                    adjacency_matrix, edges_norm, node_order,
                    adjacency_zero_threshold=adjacency_zero_threshold,
                    check_inverse_edges=True,
                )
                notes.extend(ok_notes)
            except AdjacencySupportVetoError:
                if raise_on_veto:
                    raise
                notes.append("Cross-check fallido (raise_on_veto=False).")

        packet_hash = (
            observer_packet.packet_hash if observer_packet is not None else None
        )

        def _empty_fail(**extra: Any) -> PosetFiltrationData:
            payload = dict(
                edge_count=len(edges_norm) + n_self_loops,
                audited_edge_count=0,
                ignored_edge_count=len(edges_norm),
                min_slack=None, max_slack=None,
                is_monotonic_filtration=False,
                self_loops_detected=n_self_loops,
                unknown_nodes=(),
                isolated_strata_nodes=(),
                inverse_edges_blocked=inverse_blocked,
                slack_histogram=(),
                observer_packet_hash=packet_hash,
                notes=tuple(notes),
            )
            payload.update(extra)
            return PosetFiltrationData(**payload)

        # (3) Precondición espectral.
        if not nilpotence_audit.is_strictly_nilpotent:
            notes.append("Fase 2 degradada: Fase 1 no certificó nilpotencia.")
            if raise_on_veto:
                raise CausalLoopVetoError(
                    "Filtración Poset no certificable sin nilpotencia espectral."
                )
            return _empty_fail()

        # (4) Estratos.
        strata = self._normalize_node_strata(node_strata)

        # (5)–(6) Slacks, violaciones, desconocidos, aislados.
        records, unknown, audited, ignored = self._compute_edge_slacks(
            edges_norm, strata,
            allow_unknown_nodes=allow_unknown_nodes,
            strict_filtration=strict_filtration,
        )
        isolated = self._find_isolated_strata_nodes(strata, edges_norm)
        if isolated:
            notes.append(f"Nodos aislados: {list(isolated[:8])}.")

        slacks = [r.slack for r in records]
        violations = [
            f"{r.source}(s={r.stratum_u})→{r.target}(s={r.stratum_v}) slack={r.slack}."
            for r in records if r.is_violation
        ]
        histogram = self._compute_slack_histogram(slacks)

        if violations:
            if raise_on_veto:
                raise FiltrationViolationVeto(
                    f"Violación flecha del tiempo DIKW. "
                    f"{len(violations)} violación(es). Primera: {violations[0]}"
                )
            return PosetFiltrationData(
                edge_count=len(edges_norm) + n_self_loops,
                audited_edge_count=audited,
                ignored_edge_count=ignored,
                min_slack=min(slacks) if slacks else None,
                max_slack=max(slacks) if slacks else None,
                is_monotonic_filtration=False,
                self_loops_detected=n_self_loops,
                unknown_nodes=tuple(sorted(set(unknown))),
                isolated_strata_nodes=isolated,
                inverse_edges_blocked=inverse_blocked,
                slack_histogram=histogram,
                observer_packet_hash=packet_hash,
                notes=tuple(violations + notes),
            )

        # (7) Camino más largo + clausura.
        nodes_for_paths = tuple(sorted(set(strata.keys()) | {
            n for e in edges_norm for n in e
        }))
        longest = self._longest_path_length(nodes_for_paths, edges_norm)
        closure = self._transitive_closure_warshall(nodes_for_paths, edges_norm)
        if closure.has_cycle:
            notes.append("Clausura transitiva: ciclo detectado (incoherente con ν).")
            if raise_on_veto:
                raise CausalLoopVetoError(
                    "Ciclo en clausura transitiva incompatible con Spec(A)={0}."
                )

        # (8) Height ↔ ν.
        height_cert = self._nilpotence_height_consistency(
            strata, longest, nilpotence_audit,
        )
        notes.extend(height_cert.notes)

        # (9) Semilla N3 anotada, no ejecutada.
        if observer_packet is not None:
            notes.append(
                f"ObserverPacket hash={observer_packet.packet_hash} "
                f"KAM={observer_packet.kam.is_KAM_stable} "
                f"Melnikov_tangle={observer_packet.melnikov.tangle_detected} "
                f"λ_Lya={observer_packet.return_map.stability_exponent:.3e} "
                f"θ-seed={tuple(round(x, 4) for x in observer_packet.n3_theta_seed)}."
            )

        return PosetFiltrationData(
            edge_count=len(edges_norm) + n_self_loops,
            audited_edge_count=audited,
            ignored_edge_count=ignored,
            min_slack=min(slacks) if slacks else None,
            max_slack=max(slacks) if slacks else None,
            is_monotonic_filtration=True,
            self_loops_detected=n_self_loops,
            unknown_nodes=tuple(sorted(set(unknown))),
            isolated_strata_nodes=isolated,
            inverse_edges_blocked=inverse_blocked,
            slack_histogram=histogram,
            poset_height=height_cert.height,
            longest_path=longest,
            height_nilpotence_consistent=height_cert.consistent_with_nu,
            transitive_edge_count=closure.n_reachable_pairs,
            observer_packet_hash=packet_hash,
            notes=tuple(notes),
        )

    # ── §2.2 Coerción de estratos ────────────────────────────────────────
    def _coerce_stratum_level(self, node: str, raw_level: Any) -> int:
        r"""Convierte el nivel de estrato de un nodo a entero no negativo."""
        if isinstance(raw_level, (bool, np.bool_)):
            raise StratumMappingError(
                f"Nodo '{node}': estrato booleano inválido ({raw_level!r})."
            )
        if isinstance(raw_level, (int, np.integer)):
            return int(raw_level)
        if hasattr(raw_level, "value"):
            try:
                return int(raw_level.value)
            except (TypeError, ValueError) as exc:
                raise StratumMappingError(
                    f"Nodo '{node}': no convertible: {exc}."
                ) from exc
        if isinstance(raw_level, (float, np.floating)):
            value = float(raw_level)
            if not np.isfinite(value):
                raise StratumMappingError(
                    f"Nodo '{node}': estrato no finito: {raw_level!r}."
                )
            if not value.is_integer():
                raise StratumMappingError(
                    f"Nodo '{node}': estrato no entero: {raw_level!r}."
                )
            return int(value)
        if isinstance(raw_level, str):
            s = raw_level.strip()
            try:
                return int(s)
            except ValueError:
                try:
                    fval = float(s)
                    if fval.is_integer():
                        return int(fval)
                except ValueError:
                    pass
        raise StratumMappingError(
            f"Nodo '{node}': no convertible {raw_level!r} "
            f"(tipo {type(raw_level).__name__})."
        )

    # ── §2.3 Normalización de aristas con detección de auto-bucles ─────
    def _normalize_edges(
        self,
        edges: Sequence[Tuple[str, str]],
        *,
        raise_on_self_loop: bool = True,
    ) -> Tuple[List[Tuple[str, str]], int]:
        r"""Normaliza aristas; rechaza auto-bucles (u,u) (proyecciones nulas de A)."""
        if edges is None:
            raise PipelineDirectorAgentError(
                "edges no puede ser None (use [] para DAG vacío)."
            )
        normalized: List[Tuple[str, str]] = []
        n_self_loops = 0
        seen: set = set()
        for idx, edge in enumerate(edges):
            try:
                u, v = edge
            except (TypeError, ValueError) as exc:
                raise PipelineDirectorAgentError(
                    f"Arista {idx} no desempaquetable: {edge!r}."
                ) from exc
            u_s, v_s = str(u).strip(), str(v).strip()
            if not u_s or not v_s:
                raise PipelineDirectorAgentError(
                    f"Arista {idx} tiene nodo vacío: ({u!r},{v!r})."
                )
            if u_s == v_s:
                n_self_loops += 1
                if raise_on_self_loop:
                    raise SelfLoopVetoError(
                        f"Auto-bucle índice {idx}: ({u_s!r}→{u_s!r})."
                    )
                continue
            key = (u_s, v_s)
            if key in seen:
                continue
            seen.add(key)
            normalized.append(key)
        return normalized, n_self_loops

    # ── §2.4 Normalización de node_strata ───────────────────────────────
    def _normalize_node_strata(
        self, node_strata: Mapping[str, Any],
    ) -> Dict[str, int]:
        r"""Normaliza mapeo nodo → nivel entero ≥ 0."""
        if not node_strata:
            return {}
        out: Dict[str, int] = {}
        for node, raw in node_strata.items():
            key = str(node).strip()
            if not key:
                raise StratumMappingError("Clave de nodo vacía en node_strata.")
            level = self._coerce_stratum_level(key, raw)
            if level < 0:
                raise StratumMappingError(
                    f"Nodo '{key}': estrato negativo {level}."
                )
            out[key] = level
        return out

    # ── §2.5 Cross-check edges ↔ matriz ─────────────────────────────────
    def _audit_adjacency_support(
        self,
        adjacency_matrix: Optional[NDArray[Any]],
        edges: Sequence[Tuple[str, str]],
        node_order: Optional[Sequence[str]],
        *,
        adjacency_zero_threshold: float,
        check_inverse_edges: bool = True,
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        r"""
        Cross-check edges ↔ A: soporte directo e inversas (ciclos de long. 2).
        Un ciclo de longitud 2 es un bloque hiperbólico 2×2 con traza nula
        y det < 0 — incompatible con Spec(A) = {0} salvo degeneración.
        """
        if adjacency_matrix is None or node_order is None:
            return (), ()
        A = self._validate_and_condition_adjacency(
            adjacency_matrix,
            zero_threshold=adjacency_zero_threshold,
            allow_signed_weights=True,
            require_boolean=False,
        )
        order = [str(n).strip() for n in node_order]
        if len(order) != A.shape[0]:
            raise AdjacencyMatrixFormatError(
                f"node_order (len={len(order)}) ≠ dim(A)={A.shape[0]}."
            )
        if len(set(order)) != len(order):
            raise AdjacencyMatrixFormatError("node_order con duplicados.")
        idx = {n: i for i, n in enumerate(order)}

        unsupported: List[str] = []
        inverses: List[str] = []
        for u, v in edges:
            if u not in idx or v not in idx:
                continue
            i, j = idx[u], idx[v]
            if A[i, j] == 0.0:
                unsupported.append(f"{u}→{v}")
            if check_inverse_edges and A[j, i] != 0.0:
                inverses.append(f"{v}→{u} (A[{j},{i}]={A[j, i]:.4e})")

        errors: List[str] = []
        if unsupported:
            errors.append(
                f"Soporte directo faltante en {len(unsupported)} arista(s)."
            )
        if inverses:
            errors.append(
                f"Aristas inversas (ciclos longitud 2) en {len(inverses)} caso(s)."
            )
        if errors:
            raise AdjacencySupportVetoError(
                "Cross-check edges↔adjacency fallido: " + " | ".join(errors)
            )
        return (("Cross-check OK.",), tuple(inverses))

    # ── §2.6 Histograma de slacks ───────────────────────────────────────
    def _compute_slack_histogram(
        self, slacks: List[int],
    ) -> Tuple[Tuple[int, int], ...]:
        r"""H(k) = |{(u,v) : stratum(v) − stratum(u) = k}|."""
        if not slacks:
            return ()
        return tuple(sorted(Counter(slacks).items()))

    # ── §2.7 Nodos aislados en node_strata ──────────────────────────────
    def _find_isolated_strata_nodes(
        self, strata: Dict[str, int], edges: List[Tuple[str, str]],
    ) -> Tuple[str, ...]:
        r"""Nodos en node_strata sin aristas incidentes (grado 0)."""
        if not strata:
            return ()
        connected = frozenset(n for e in edges for n in e)
        return tuple(sorted(n for n in strata if n not in connected))

    # ── §2.8 Slacks por arista ──────────────────────────────────────────
    def _compute_edge_slacks(
        self,
        edges: Sequence[Tuple[str, str]],
        strata: Mapping[str, int],
        *,
        allow_unknown_nodes: bool,
        strict_filtration: bool,
    ) -> Tuple[List[EdgeSlackRecord], List[str], int, int]:
        r"""
        Emite un EdgeSlackRecord por arista conocida.
        Retorna (records, unknown_names, n_audited, n_ignored).
        """
        records: List[EdgeSlackRecord] = []
        unknown: List[str] = []
        audited = 0
        ignored = 0
        for u, v in edges:
            u_known, v_known = u in strata, v in strata
            if not u_known or not v_known:
                if allow_unknown_nodes:
                    ignored += 1
                    if not u_known:
                        unknown.append(u)
                    if not v_known:
                        unknown.append(v)
                    continue
                missing = [n for n, k in ((u, u_known), (v, v_known)) if not k]
                raise StratumMappingError(
                    f"Falta estrato para {missing} en arista {u}→{v}."
                )
            su, sv = int(strata[u]), int(strata[v])
            slack = sv - su
            audited += 1
            is_viol = (slack <= 0) if strict_filtration else (slack < 0)
            records.append(EdgeSlackRecord(
                source=u, target=v,
                stratum_u=su, stratum_v=sv,
                slack=slack, is_violation=is_viol,
            ))
        return records, unknown, audited, ignored

    # ── §2.9 Clausura transitiva (Warshall booleano) ────────────────────
    def _transitive_closure_warshall(
        self,
        nodes: Sequence[str],
        edges: Sequence[Tuple[str, str]],
    ) -> TransitiveClosureData:
        r"""
        R⁺ del DAG.  Detecta ciclos si algún R[i,i] se enciende.
        Complejidad O(n³) con n = |nodes|; n=0 es vacuamente acíclico.
        """
        labels = list(nodes)
        n = len(labels)
        if n == 0:
            return TransitiveClosureData(
                n_nodes=0, n_edges_declared=len(edges),
                n_reachable_pairs=0, n_implied_edges=0,
                has_cycle=False,
                notes=("Poset vacío.",),
            )
        idx = {lab: i for i, lab in enumerate(labels)}
        R = np.zeros((n, n), dtype=bool)
        declared = 0
        for u, v in edges:
            if u in idx and v in idx:
                R[idx[u], idx[v]] = True
                declared += 1
        for k in range(n):
            # R |= R[:,k] ∧ R[k,:]
            rk = R[:, k]
            ck = R[k, :]
            if not rk.any() or not ck.any():
                continue
            R |= rk[:, None] & ck[None, :]
        has_cycle = bool(np.any(np.diag(R)))
        reachable = int(np.count_nonzero(R))
        implied = max(0, reachable - declared)
        return TransitiveClosureData(
            n_nodes=n,
            n_edges_declared=declared,
            n_reachable_pairs=reachable,
            n_implied_edges=implied,
            has_cycle=has_cycle,
            notes=(
                f"Warshall: |R⁺|={reachable}, implicadas={implied}, ciclo={has_cycle}.",
            ),
        )

    # ── §2.10 Camino más largo en el DAG ────────────────────────────────
    def _longest_path_length(
        self,
        nodes: Sequence[str],
        edges: Sequence[Tuple[str, str]],
    ) -> Optional[int]:
        r"""
        Longitud (nº de aristas) del camino más largo.

        DP sobre orden topológico de Kahn.  Si hay ciclo, retorna None
        (el índice de nilpotencia de Fase 1 es entonces la cota fiable).
        """
        labels = list(nodes)
        if not labels:
            return 0
        succ: Dict[str, List[str]] = {n: [] for n in labels}
        indeg: Dict[str, int] = {n: 0 for n in labels}
        for u, v in edges:
            if u not in succ or v not in indeg:
                continue
            succ[u].append(v)
            indeg[v] += 1
        dist: Dict[str, int] = {n: 0 for n in labels}
        queue = [n for n, d in indeg.items() if d == 0]
        seen = 0
        while queue:
            u = queue.pop()
            seen += 1
            du = dist[u]
            for v in succ[u]:
                if du + 1 > dist[v]:
                    dist[v] = du + 1
                indeg[v] -= 1
                if indeg[v] == 0:
                    queue.append(v)
        if seen != len(labels):
            return None  # ciclo
        return int(max(dist.values()) if dist else 0)

    # ── §2.11 Consistencia height ↔ ν(A) ────────────────────────────────
    def _nilpotence_height_consistency(
        self,
        strata: Mapping[str, int],
        longest_path: Optional[int],
        spectra: NilpotenceAuditData,
    ) -> PosetHeightCertificate:
        r"""
        Verifica ν(A) ≤ longest_path + 1 ≤ height + 1.

        Si longest_path es None (ciclo) se declara inconsistente salvo
        que Fase 1 no hubiera certificado nilpotencia (ya vetado).
        """
        notes: List[str] = []
        if not strata:
            return PosetHeightCertificate(
                n_nodes=0, min_rank=None, max_rank=None, height=None,
                longest_path=longest_path,
                nilpotence_index=spectra.nilpotence_index,
                consistent_with_nu=True,
                notes=("Sin estratos: consistencia vacuamente verdadera.",),
            )
        ranks = list(strata.values())
        min_r, max_r = int(min(ranks)), int(max(ranks))
        height = max_r - min_r
        nu = spectra.nilpotence_index
        consistent = True
        if longest_path is None:
            consistent = False
            notes.append("Camino más largo indefinido (ciclo).")
        else:
            if longest_path > height:
                # Posible si la filtración no es estricta (slack 0 permitido).
                notes.append(
                    f"longest_path={longest_path} > height={height}: "
                    f"filtración no estricta o rangos no inyectivos."
                )
            if nu is not None and nu > longest_path + 1:
                consistent = False
                notes.append(
                    f"Inconsistencia ν={nu} > longest_path+1={longest_path + 1}."
                )
            elif nu is not None and nu > height + 1:
                notes.append(
                    f"ν={nu} > height+1={height + 1} (cotas holgadas; "
                    f"aceptable si slack=0 está permitido)."
                )
            else:
                notes.append(
                    f"Consistencia OK: ν={nu} ≤ longest_path+1="
                    f"{None if longest_path is None else longest_path + 1} "
                    f"≤ height+1={height + 1}."
                )
        return PosetHeightCertificate(
            n_nodes=len(strata),
            min_rank=min_r, max_rank=max_r, height=height,
            longest_path=longest_path,
            nilpotence_index=nu,
            consistent_with_nu=consistent,
            notes=tuple(notes),
        )

    # ── §2.12 Máscara DIKW para el kernel N2 ────────────────────────────
    def _build_poset_mask(
        self,
        node_strata: Mapping[str, int],
        node_order: Optional[Sequence[str]],
        n: int,
    ) -> NDArray[np.float64]:
        r"""
        mask[i,j] = 1 ⇔ i≠j y (sin orden → 1  ó  stratum[j] ≥ stratum[i]).

        Si no hay ``node_order`` se admite todo el triángulo estricto
        (la proyección Π_triu ya garantiza aciclicidad).
        """
        mask = np.triu(np.ones((n, n), dtype=np.float64), k=1)
        if not node_order or not node_strata:
            return mask
        order = [str(x).strip() for x in node_order[:n]]
        if len(order) < n:
            return mask
        for i, u in enumerate(order):
            su = node_strata.get(u)
            if su is None:
                continue
            for j, v in enumerate(order):
                if j <= i:
                    mask[i, j] = 0.0
                    continue
                sv = node_strata.get(v)
                if sv is None:
                    continue
                if sv < su:
                    mask[i, j] = 0.0
        return mask

    # ── §2.13 Inicialización del kernel N3 desde el ObserverPacket ─────
    def _initialize_n3_kernel_from_observer(
        self,
        observer_packet: ObserverPacket,
        *,
        n_params: int = 4,
        adjacency_shape: int = 8,
        poset_mask: Optional[NDArray[np.float64]] = None,
        weights: Optional[Tuple[float, float, float, float, float, float]] = None,
    ) -> RecursiveSelfImprovementKernel:
        r"""Construye un kernel N3 fresco y lo alimenta con el ObserverPacket."""
        kernel = RecursiveSelfImprovementKernel(
            n_params=n_params,
            adjacency_shape=adjacency_shape,
            weights=weights,
        )
        if poset_mask is not None:
            kernel.bind_poset_mask(poset_mask)
        kernel.ingest_observer_packet(observer_packet)
        return kernel

    # ── §2.14 Ciclo de optimización N3 ──────────────────────────────────
    def _run_n3_optimization_cycle(
        self,
        kernel: RecursiveSelfImprovementKernel,
        *,
        n_cycles: int = _N3_DEFAULT_CYCLES,
    ) -> Tuple[Dict[str, Any], N3ConvergenceCertificate]:
        r"""Ejecuta N1→N2→N3 durante n_cycles y retorna (reporte, certificado)."""
        try:
            return kernel.optimize_recursive_n3(n_cycles=n_cycles)
        except RecursiveSelfImprovementError:
            raise
        except Exception as exc:
            raise RecursiveSelfImprovementError(
                f"Fallo en el kernel N3: {exc}"
            ) from exc

    # ── §2.15 Síntesis del OrientationSeed ──────────────────────────────
    def _synthesize_orientation_seed(
        self,
        observer_packet: ObserverPacket,
        n3_report: Mapping[str, Any],
        n3_cert: N3ConvergenceCertificate,
        filtration: PosetFiltrationData,
    ) -> OrientationSeed:
        r"""Sella el artefacto terminal de Fase 2."""
        return OrientationSeed(
            theta_final=np.asarray(n3_report["theta"], dtype=np.float64),
            alpha_final=np.asarray(n3_report["alpha"], dtype=np.float64),
            O_norm=float(n3_report["O_norm"]),
            final_loss=float(n3_report["final_loss"]),
            observer_packet=observer_packet,
            kam_preserved=observer_packet.kam.is_KAM_stable,
            melnikov_chaotic=observer_packet.melnikov.tangle_detected,
            n3_report=dict(n3_report),
            n3_certificate=n3_cert,
            poset_height=filtration.poset_height,
            idempotent=bool(n3_cert.idempotent),
            packet_hash=observer_packet.packet_hash,
            generated_at_utc=_utc_timestamp(),
        )

    # ── §2.16 Stub funtorial Φ₂ → Φ₃ ────────────────────────────────────
    def _intercept_mayer_vietoris_sequence(
        self,
        betti_1_A: int,
        betti_1_B: int,
        betti_1_intersection: int,
        betti_1_union: int,
        **kwargs: Any,
    ) -> "MayerVietorisAuditData":
        r"""
        Stub funtorial Φ₂ → Φ₃.

        Phase3_MayerVietorisInterceptor sobrescribe este método.  El
        puente terminal de Fase 2 lo invoca como continuación formal.
        """
        raise NotImplementedError(
            "Phase3_MayerVietorisInterceptor debe ser mezclado. "
            "Este stub es el conector funtorial Φ₂ → Φ₃.  Continúe en Fase 3 "
            "con la definición efectiva de _intercept_mayer_vietoris_sequence."
        )

    # ══════════════════════════════════════════════════════════════════════
    # §2.17  ÚLTIMO MÉTODO DE FASE 2: puente funtorial Φ₂ → Φ₃
    #        Su definición es la CONTINUACIÓN / INICIO de los métodos
    #        de Fase 3 (DECIDE : Mayer–Vietoris + Ω₃).
    # ══════════════════════════════════════════════════════════════════════
    def _phase2_terminal_bridge_to_phase3(
        self,
        adjacency_matrix: NDArray[Any],
        edges: Sequence[Tuple[str, str]],
        node_strata: Mapping[str, Any],
        betti_1_A: int,
        betti_1_B: int,
        betti_1_intersection: int,
        betti_1_union: int,
        *,
        adjacency_zero_threshold: float = _BASE_SPECTRAL_TOLERANCE,
        allow_signed_weights: bool = False,
        require_boolean: bool = False,
        deep_nilpotence_audit: bool = True,
        power_audit_max_dim: int = _DEFAULT_POWER_AUDIT_MAX_DIM,
        allow_unknown_nodes: bool = False,
        strict_filtration: bool = False,
        node_order: Optional[Sequence[str]] = None,
        cross_check_adjacency_support: bool = False,
        image_rank_h1_intersection: Optional[int] = None,
        connecting_boundary_rank: Optional[int] = None,
        betti_0_A: Optional[int] = None,
        betti_0_B: Optional[int] = None,
        betti_0_intersection: Optional[int] = None,
        betti_0_union: Optional[int] = None,
        n3_cycles: int = _N3_DEFAULT_CYCLES,
        raise_on_veto: bool = True,
        require_transverse_section: bool = False,
    ) -> Tuple[
        NilpotenceAuditData,
        PosetFiltrationData,
        ObserverPacket,
        OrientationSeed,
        "MayerVietorisAuditData",
    ]:
        r"""
        ÚLTIMO MÉTODO DE FASE 2 — Puente funtorial Φ₂ → Φ₃.

        Composición estricta:
            (1) Ejecuta Φ₁ (Schur + Poincaré) vía
                `_phase1_terminal_bridge_to_phase2`, que a su vez invoca
                `_audit_poset_filtration_from_nilpotence` (este objeto).
            (2) Construye la máscara DIKW y el kernel N3 desde el
                ObserverPacket (θ₀ = n3_theta_seed).
            (3) Ciclo N1→N2→N3 → OrientationSeed (certificado de
                convergencia, idempotencia, Lip(∇L)).
            (4) Invoca el primer método efectivo de Fase 3
                (`_intercept_mayer_vietoris_sequence`) con el seed.

        Anidamiento:
            Φ₃ ∘ Φ₂ ∘ Φ₁
              = _intercept_mayer_vietoris_sequence
                ∘ _phase2_terminal_bridge_to_phase3

        Retorna
            (NilpotenceAuditData, PosetFiltrationData, ObserverPacket,
             OrientationSeed, MayerVietorisAuditData).

        La definición de `_intercept_mayer_vietoris_sequence` en
        Phase3_MayerVietorisInterceptor ES la continuación literal de
        este método: recibe ``orientation_seed=...`` como frontera
        homológica de ORIENT → DECIDE.
        """
        # ── Paso 1: Fase 1 completa (Schur + Poincaré + filtración) ─────
        spectra, filtration, observer_packet = (
            self._phase1_terminal_bridge_to_phase2(
                adjacency_matrix, edges, node_strata,
                adjacency_zero_threshold=adjacency_zero_threshold,
                allow_signed_weights=allow_signed_weights,
                require_boolean=require_boolean,
                deep_nilpotence_audit=deep_nilpotence_audit,
                power_audit_max_dim=power_audit_max_dim,
                allow_unknown_nodes=allow_unknown_nodes,
                strict_filtration=strict_filtration,
                node_order=node_order,
                cross_check_adjacency_support=cross_check_adjacency_support,
                raise_on_veto=raise_on_veto,
                require_transverse_section=require_transverse_section,
            )
        )

        # ── Paso 2: Kernel N3 ← ObserverPacket ──────────────────────────
        n_shape = max(1, min(8, int(spectra.dimension) if spectra.dimension else 1))
        strata_norm = self._normalize_node_strata(node_strata)
        poset_mask = self._build_poset_mask(strata_norm, node_order, n_shape)
        kernel = self._initialize_n3_kernel_from_observer(
            observer_packet,
            n_params=4,
            adjacency_shape=n_shape,
            poset_mask=poset_mask,
        )

        # ── Paso 3: Ciclo recursivo N1 → N2 → N3 ────────────────────────
        n3_report, n3_cert = self._run_n3_optimization_cycle(
            kernel, n_cycles=n3_cycles,
        )
        orientation_seed = self._synthesize_orientation_seed(
            observer_packet, n3_report, n3_cert, filtration,
        )

        logger.info(
            "[Fase 2::ORIENT] θ*=%.4f | α*=%.4f | loss=%.6e | ΔL=%.3e | "
            "idem=%s | Lip∇=%.3e | height=%s | chaotic=%s | hash=%s",
            float(np.linalg.norm(orientation_seed.theta_final)),
            float(np.linalg.norm(orientation_seed.alpha_final)),
            orientation_seed.final_loss,
            n3_cert.final_loss - n3_cert.initial_loss,
            n3_cert.idempotent,
            n3_cert.lipschitz_grad_bound,
            filtration.poset_height,
            orientation_seed.melnikov_chaotic,
            orientation_seed.packet_hash,
        )

        if raise_on_veto and not n3_cert.finite:
            raise RecursiveSelfImprovementError(
                "Kernel N3 no finito tras el ciclo de automejora."
            )

        # ── Paso 4: CONTINUACIÓN FUNTORIAL HACIA FASE 3 ─────────────────
        # La invocación de `_intercept_mayer_vietoris_sequence` ES el
        # primer método efectivo de Fase 3, ejecutado como continuación
        # formal del cierre de Fase 2.
        # -----------------------------------------------------------------
        # FASE 3 comienza exactamente en la definición efectiva de
        # Phase3_MayerVietorisInterceptor._intercept_mayer_vietoris_sequence
        # (DECIDE : Mayer–Vietoris + Euler–Poincaré ← OrientationSeed).
        # -----------------------------------------------------------------
        mv_audit = self._intercept_mayer_vietoris_sequence(
            betti_1_A, betti_1_B, betti_1_intersection, betti_1_union,
            image_rank_h1_intersection=image_rank_h1_intersection,
            connecting_boundary_rank=connecting_boundary_rank,
            betti_0_A=betti_0_A, betti_0_B=betti_0_B,
            betti_0_intersection=betti_0_intersection,
            betti_0_union=betti_0_union,
            orientation_seed=orientation_seed,
            raise_on_veto=raise_on_veto,
        )

        return spectra, filtration, observer_packet, orientation_seed, mv_audit


# ══════════════════════════════════════════════════════════════════════════════
# FIN DE FASE 2  ──  CONTINÚA EN FASE 3 (prompt siguiente)
# Phase3_MayerVietorisInterceptor(Phase2_PosetFiltrationAuditor)
#   debe abrir con _intercept_mayer_vietoris_sequence     (Φ₃)
#   y cerrar con _phase3_terminal_synthesis               (Z_Causal = Φ₃∘Φ₂∘Φ₁)
# ══════════════════════════════════════════════════════════════════════════════
# ████████████████████████████████████████████████████████████████████████████
# █                                                                          █
# █   FASE 3 ──► DECIDE & ACT : MAYER–VIETORIS + VEREDICTO TERMINAL Ω₃       █
# █   (secuencia exacta + Euler–Poincaré + Jensen–Shannon + gobernanza)      █
# █                                                                          █
# █   CONTINUACIÓN FORMAL DE FASE 2                                          █
# █   Φ₃ abre en  _intercept_mayer_vietoris_sequence                         █
# █   Φ₃ cierra en _phase3_terminal_synthesis   (Z_Causal = Φ₃ ∘ Φ₂ ∘ Φ₁)    █
# █                                                                          █
# ████████████████████████████████████████████████████████████████████████████
#
# Contrato de anidamiento (OODA-Decide):
#   Phase2._phase2_terminal_bridge_to_phase3
#       └─► Phase3._intercept_mayer_vietoris_sequence   ← AQUÍ ARRANCA Φ₃
#               (OrientationSeed → frontera homológica)
#       └─► Phase3._phase3_terminal_synthesis
#               └─► CausalGovernanceState = Ω₃  (objeto terminal)
#
# Secuencia exacta de Mayer–Vietoris (coeficientes ℝ, CW-pares regulares):
#
#   H₁(A∩B)  --φ-->  H₁(A)⊕H₁(B)  --ψ-->  H₁(A∪B)  --∂-->  H₀(A∩B)
#            --φ₀-->  H₀(A)⊕H₀(B)  --ψ₀-->  H₀(A∪B)  -->  0
#
# Exactitud + rank-nullity ⇒ identidad de rangos
#   β₁(A∪B) = β₁(A) + β₁(B) − rank(im φ) + rank(∂)
# y la desigualdad débil (siempre)
#   |β₁(A∪B) − (β₁(A)+β₁(B))| ≤ β₁(A∩B).
#
# Puente Poincaré → homología (Fase 1 ↪ Fase 3):
#   · ceros transversales de Melnikov  ~  generadores extra de H₁ (herradura);
#   · rigidez KAM                      ~  Δβ₁ = 0 estable bajo ε-perturbación;
#   · difusión de Arnold               ~  fuga de ciclos entre cubiertas;
#   · λ_Lya > 0                        ~  mezcla exponencial de 1-ciclos.
#
# Puente N3 → homología (Fase 2 ↪ Fase 3):
#   · O* triangular estricta ∩ máscara DIKW  ⇒  el 1-esqueleto fusionado
#     no puede crear ciclos que MV no vea;
#   · idempotencia de Ψ                    ⇒  estabilidad del veredicto Ω₃.
# ══════════════════════════════════════════════════════════════════════════════


# ── 3.0 CONSTANTES DE FASE 3 ────────────────────────────────────────────────

_JS_FLOOR: float                        = 1.0e-15
_JS_MAX: float                          = math.log(2.0)          # cota JS(P‖Q)
_DALECKII_KREIN_EXP: float              = 1.5                    # λ₂^{-3/2}
_DALECKII_KREIN_PREFACTOR: float        = 0.5
_HOMOCLINIC_BETTI_CAP: int              = 64
_OMEGA3_SOFT_REL_DEFECT: float          = 1.0e-12


# ── 3.1 DTOs DE FASE 3 ──────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class ExactnessProfile:
    r"""
    Defectos de exactitud término a término de la sucesión MV.

    residual_h1_union : Δβ₁ = β₁(A∪B) − expected   (exactitud en H₁(U))
    residual_weak     : max(0, |β₁(U)−(β₁(A)+β₁(B))| − β₁(I))
    residual_h0       : β₀(U) − (β₀(A)+β₀(B)−β₀(I)+rank ∂)   (None si faltan β₀)
    residual_euler    : χ(U) − (χ(A)+χ(B)−χ(I))              (None si faltan β₀)
    snake_residual    : |rank(∂) − (β₁(U) − rank ψ)|
                        con rank ψ := β₁(A)+β₁(B)−rank(im φ)
                        (idéntico a |Δβ₁| cuando expected usa esa fórmula).
    """
    residual_h1_union:      int
    residual_weak:          int
    residual_h0:            Optional[int]
    residual_euler:         Optional[int]
    snake_residual:         int
    is_exact_at_h1:         bool
    is_exact_at_h0:         Optional[bool]
    is_euler_additive:      Optional[bool]


@dataclass(frozen=True, slots=True)
class HomoclinicBettiCorrection:
    r"""
    Corrección diagnóstica de Betti por tangencias homoclínicas.

    Cada cero transversal de Melnikov es un candidato a generador extra
    de H₁ (herradura de Smale).  En un DAG nilpotente esos generadores
    están prohibidos: se registran, no se suman a ``expected``.

    pressure = min(1, n_trans / (1 + β₁(I)))  ∈ [0, 1].
    """
    transverse_zeros:       int
    simple_zeros:           int
    splitting_distance:     float
    extra_generators:       int
    pressure:               float
    contradicts_nilpotence: bool


@dataclass(frozen=True, slots=True)
class KAMHomologyRigidity:
    r"""
    Rigidez homológica a la KAM.

    Si los toros nulos persisten (ε < ε_KAM ∧ diofantina), Δβ₁ es
    estructuralmente estable: cualquier defecto es una ruptura de
    la filtración, no un artefacto de perturbación.
    """
    is_rigid:               bool
    epsilon_ratio:          float
    diophantine:            bool
    tori_persisting:        int
    admissible_defect:      int         # 0 si rígido; ≥0 si no


@dataclass(frozen=True, slots=True)
class JensenShannonReport:
    r"""
    Divergencia de Jensen–Shannon entre el 4-vector observado
        P ~ (β₁(A), β₁(B), β₁(I), β₁(U))
    y el predicho por exactitud
        Q ~ (β₁(A), β₁(B), β₁(I), expected).

    JS(P‖Q) = ½ KL(P‖M) + ½ KL(Q‖M),  M=(P+Q)/2,  ∈ [0, log 2].
    ``relative_defect`` conserva la fórmula original v3
        |Δβ₁| / (1 + max(|β₁(U)|, |expected|)).
    """
    js_divergence:          float
    relative_defect:        float
    observed_simplex:       Tuple[float, float, float, float]
    predicted_simplex:      Tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class PipelineCausalAuditState:
    r"""Estado inmutable de la auditoría Poincaré–Daleckii–Krein."""
    mayer_vietoris_residual:    float
    daleckii_krein_bound:       float
    fiedler_lambda2:            float
    lipschitz_coherent:         bool
    is_funtorially_coherent:    bool


@dataclass(frozen=True, slots=True)
class MayerVietorisAuditData:
    r"""
    Artefacto de Fase 3 — certificado de fusión homológica.

    Extensión v3.1: perfil de exactitud, corrección homoclínica,
    rigidez KAM, JS propio, huella del OrientationSeed.
    """
    betti_1_A:                      int
    betti_1_B:                      int
    betti_1_intersection:           int
    betti_1_union:                  int
    expected_union_betti_1:         int
    delta_betti_1:                  int
    relative_defect:                float
    jensen_shannon_defect:          float
    is_fusion_homologous:           bool
    weak_bound_satisfied:           bool = True
    euler_characteristic_valid:     Optional[bool] = None
    h0_sequence_valid:              Optional[bool] = None
    image_rank_phi:                 int = 0
    connecting_boundary_rank:       int = 0
    snake_residual:                 int = 0
    js_divergence:                  float = 0.0
    homoclinic_pressure:            float = 0.0
    kam_rigid:                      Optional[bool] = None
    chi_union:                      Optional[int] = None
    chi_expected:                   Optional[int] = None
    orientation_seed_hash:          Optional[str] = None
    n3_idempotent:                  Optional[bool] = None
    exactness:                      Optional[ExactnessProfile] = None
    notes:                          Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Omega3Certificate:
    r"""
    Objeto terminal Ω₃  ≅  Z_Causal  en la categoría de auditorías.

    Ω₃ es el producto fibrado (pullback) de los tres certificados:

        Ω₃  ⊢  Φ₁(Spec=0)  ∧  Φ₂(filtración monótona ∧ N3 finito)
               ∧  Φ₃(Δβ₁=0 ∧ χ aditivo)  ∧  Poincaré admisible.

    ``is_terminal_object`` ⇔ todos los factores son verdaderos.
    """
    phi1_nilpotent:             bool
    phi2_monotonic:             bool
    phi2_height_consistent:     Optional[bool]
    phi3_homologous:            bool
    phi3_euler:                 Optional[bool]
    n3_finite:                  bool
    n3_idempotent:              bool
    poincare_admissible:        bool
    kam_stable:                 bool
    melnikov_admissible:        bool
    is_terminal_object:         bool
    composition_hash:           str
    failing_factors:            Tuple[str, ...]
    notes:                      Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True, eq=False)
class CausalGovernanceState:
    r"""
    Objeto final del endofuntor de gobernanza causal:
        Z_Causal = Φ₃ ∘ Φ₂ ∘ Φ₁.

    Enriquecido v3.1: Ω₃, huella de composición, paquetes de F1/F2.
    """
    governance_id:              str
    nilpotence_audit:           NilpotenceAuditData
    filtration_audit:           PosetFiltrationData
    mayer_vietoris_audit:       MayerVietorisAuditData
    omega3:                     Omega3Certificate
    observer_packet:            Optional[ObserverPacket] = None
    orientation_seed:           Optional[OrientationSeed] = None
    is_causally_valid:          bool = False
    composition_hash:           str = ""
    generated_at_utc:           str = ""


# ── 3.2 CLASE FASE 3 ────────────────────────────────────────────────────────

class Phase3_MayerVietorisInterceptor(Phase2_PosetFiltrationAuditor):
    r"""
    Interceptor homológico de fusión con secuencia exacta completa.

    Cadena funtorial:
        Φ₃ : (OrientationSeed, Betti) → (MayerVietorisAudit, Ω₃).

    El PRIMER método efectivo sobrescribe el stub de Fase 2 y es la
    continuación literal de `_phase2_terminal_bridge_to_phase3`.
    """

    # ══════════════════════════════════════════════════════════════════════
    # §3.1  PRIMER MÉTODO EFECTIVO DE FASE 3 — continuación de Φ₂
    #       Sobrescribe el stub Phase2._intercept_mayer_vietoris_sequence
    # ══════════════════════════════════════════════════════════════════════
    def _intercept_mayer_vietoris_sequence(
        self,
        betti_1_A: int,
        betti_1_B: int,
        betti_1_intersection: int,
        betti_1_union: int,
        *,
        image_rank_h1_intersection: Optional[int] = None,
        connecting_boundary_rank: Optional[int] = None,
        betti_0_A: Optional[int] = None,
        betti_0_B: Optional[int] = None,
        betti_0_intersection: Optional[int] = None,
        betti_0_union: Optional[int] = None,
        orientation_seed: Optional[OrientationSeed] = None,
        raise_on_veto: bool = True,
    ) -> MayerVietorisAuditData:
        r"""
        ← ARRANQUE FORMAL DE FASE 3 (DECIDE).

        Auditoría de fusión homológica de dos cubiertas A, B del 1-esqueleto
        causal.  Recibe el ``OrientationSeed`` de Fase 2 como frontera:
        KAM/Melnikov/N3 no alteran la identidad algebraica (fail-secure),
        pero calibran notas, presión homoclínica y rigidez.

        Pipeline interno (granular):
            (1)  validar números de Betti (enteros ≥ 0, no booleanos);
            (2)  cotas algebraicas de rank(im φ) y rank(∂);
            (3)  identidad exacta  β₁(U) = β₁(A)+β₁(B)−rank φ + rank ∂;
            (4)  desigualdad débil MV;
            (5)  sucesión H₀ y Euler–Poincaré (si hay β₀);
            (6)  JS propio + defecto relativo;
            (7)  perfil de exactitud (snake lemma);
            (8)  corrección homoclínica (Melnikov → H₁);
            (9)  rigidez KAM de la homología;
            (10) notas N3 / Poincaré;
            (11) vetos fail-secure (Δβ₁, Euler).
        """
        bA = self._validate_betti_number("betti_1_A", betti_1_A)
        bB = self._validate_betti_number("betti_1_B", betti_1_B)
        bI = self._validate_betti_number("betti_1_intersection", betti_1_intersection)
        bU = self._validate_betti_number("betti_1_union", betti_1_union)

        notes: List[str] = []
        seed_hash: Optional[str] = None
        n3_idem: Optional[bool] = None
        if orientation_seed is not None:
            self._validate_orientation_seed(orientation_seed)
            seed_hash = orientation_seed.packet_hash
            n3_idem = bool(orientation_seed.idempotent)

        # (2)–(3) Identidad exacta.
        image_rank, boundary_rank, expected, rank_notes = (
            self._exact_mayer_vietoris_prediction(
                bA, bB, bI, bU,
                image_rank_h1_intersection=image_rank_h1_intersection,
                connecting_boundary_rank=connecting_boundary_rank,
            )
        )
        notes.extend(rank_notes)
        delta = int(bU - expected)

        if expected < 0:
            notes.append(
                f"ALERTA: expected={expected} < 0: rangos inconsistentes "
                f"(im φ={image_rank}, ∂={boundary_rank})."
            )

        # (4) Desigualdad débil.
        weak_ok, weak_msg, weak_residual = self._verify_weak_mayer_vietoris_bound(
            bA, bB, bI, bU,
        )
        notes.append(weak_msg)

        # (5) H₀ y Euler–Poincaré.
        h0_ok, h0_msg, h0_residual = self._verify_h0_sequence(
            betti_0_A, betti_0_B, betti_0_intersection, betti_0_union,
            boundary_rank,
        )
        notes.append(h0_msg)

        euler_ok, euler_msg, chi_U, chi_exp, euler_residual = (
            self._verify_euler_poincare(
                betti_0_A, betti_0_B, betti_0_intersection, betti_0_union,
                bA, bB, bI, bU,
            )
        )
        notes.append(euler_msg)

        # (6) Jensen–Shannon.
        js_report = self._compute_jensen_shannon_report(bA, bB, bI, bU, expected)
        rel_defect = js_report.relative_defect
        js_div = js_report.js_divergence

        # (7) Perfil de exactitud / snake.
        rank_psi = bA + bB - image_rank
        snake = abs(boundary_rank - (bU - rank_psi))
        exactness = ExactnessProfile(
            residual_h1_union=int(delta),
            residual_weak=int(weak_residual),
            residual_h0=h0_residual,
            residual_euler=euler_residual,
            snake_residual=int(snake),
            is_exact_at_h1=(delta == 0 and expected >= 0),
            is_exact_at_h0=h0_ok,
            is_euler_additive=euler_ok,
        )

        # (8)–(9) Poincaré ↪ homología.
        homo = self._homoclinic_betti_correction(orientation_seed)
        kam_r = self._kam_homology_rigidity(orientation_seed)
        notes.extend(
            self._modulate_notes_from_orientation_seed(
                orientation_seed, homo, kam_r, delta,
            )
        )

        is_homologous = bool(delta == 0 and expected >= 0)

        # (10) Consistencia Poincaré–homología (diagnóstico, no veto extra).
        notes.extend(self._poincare_homology_consistency(
            orientation_seed, is_homologous, homo,
        ))

        # (11) Vetos fail-secure: la identidad algebraica no se relaja.
        if not is_homologous and raise_on_veto:
            raise HomologicalFusionVeto(
                f"Paradoja MV: Δβ₁={delta} ≠ 0. "
                f"β₁(A)={bA}, β₁(B)={bB}, β₁(A∩B)={bI}, β₁(A∪B)={bU}. "
                f"Predicho: {expected} [im φ={image_rank}, ∂={boundary_rank}]. "
                f"JS={js_div:.6f}, rel={rel_defect:.6f}, snake={snake}, "
                f"homoclinic_pressure={homo.pressure:.3f}, "
                f"KAM_rigid={kam_r.is_rigid}."
            )
        if euler_ok is False and raise_on_veto:
            raise EulerPoincareMismatchError(
                f"Euler–Poincaré violado. {euler_msg}"
            )

        return MayerVietorisAuditData(
            betti_1_A=bA, betti_1_B=bB,
            betti_1_intersection=bI, betti_1_union=bU,
            expected_union_betti_1=expected,
            delta_betti_1=delta,
            relative_defect=rel_defect,
            jensen_shannon_defect=rel_defect,   # compat v3
            is_fusion_homologous=is_homologous,
            weak_bound_satisfied=weak_ok,
            euler_characteristic_valid=euler_ok,
            h0_sequence_valid=h0_ok,
            image_rank_phi=image_rank,
            connecting_boundary_rank=boundary_rank,
            snake_residual=int(snake),
            js_divergence=js_div,
            homoclinic_pressure=homo.pressure,
            kam_rigid=kam_r.is_rigid if orientation_seed is not None else None,
            chi_union=chi_U,
            chi_expected=chi_exp,
            orientation_seed_hash=seed_hash,
            n3_idempotent=n3_idem,
            exactness=exactness,
            notes=tuple(notes),
        )

    # ── §3.2 Validación de números de Betti ─────────────────────────────
    def _validate_betti_number(self, name: str, value: Any) -> int:
        r"""Valida entero ≥ 0 (los booleanos de Python son subtipo de int)."""
        if isinstance(value, (bool, np.bool_)):
            raise MayerVietorisInputError(
                f"{name} no puede ser booleano ({value!r})."
            )
        if isinstance(value, (int, np.integer)):
            ivalue = int(value)
        elif isinstance(value, (float, np.floating)):
            fv = float(value)
            if not np.isfinite(fv):
                raise MayerVietorisInputError(f"{name} no finito: {value!r}.")
            if not fv.is_integer():
                raise MayerVietorisInputError(f"{name} no entero: {value!r}.")
            ivalue = int(fv)
        else:
            raise MayerVietorisInputError(
                f"{name} debe ser entero ≥ 0, recibido {value!r} "
                f"(tipo {type(value).__name__})."
            )
        if ivalue < 0:
            raise MayerVietorisInputError(f"{name} negativo: {ivalue}.")
        return ivalue

    # ── §3.3 Validación de rango opcional ───────────────────────────────
    def _validate_optional_rank(
        self,
        name: str,
        value: Optional[int],
        upper_bound: int,
        lower_bound: int = 0,
    ) -> Optional[int]:
        r"""Valida lower ≤ rank ≤ upper (cotas de rank-nullity)."""
        if value is None:
            return None
        rank = self._validate_betti_number(name, value)
        if rank < lower_bound:
            raise MayerVietorisInputError(
                f"{name}={rank} < cota inferior {lower_bound}."
            )
        if rank > upper_bound:
            raise MayerVietorisInputError(
                f"{name}={rank} > cota algebraica {upper_bound}."
            )
        return rank

    # ── §3.4 Validación del OrientationSeed ─────────────────────────────
    def _validate_orientation_seed(self, seed: OrientationSeed) -> None:
        r"""Contrato mínimo: packet sellado, θ/α finitos, loss finita."""
        if not isinstance(seed, OrientationSeed):
            raise MayerVietorisInputError(
                f"orientation_seed tipo inválido: {type(seed).__name__}."
            )
        if not seed.packet_hash:
            raise MayerVietorisInputError("OrientationSeed sin packet_hash.")
        if seed.observer_packet.packet_hash != seed.packet_hash:
            raise MayerVietorisInputError(
                "Incoherencia de huella: OrientationSeed.packet_hash ≠ "
                "ObserverPacket.packet_hash (ruptura del anidamiento Φ₂∘Φ₁)."
            )
        if not math.isfinite(seed.final_loss):
            raise RecursiveSelfImprovementError(
                f"OrientationSeed.final_loss no finito: {seed.final_loss}."
            )
        if not np.all(np.isfinite(seed.theta_final)):
            raise RecursiveSelfImprovementError("θ* contiene no finitos.")
        if not np.all(np.isfinite(seed.alpha_final)):
            raise RecursiveSelfImprovementError("α* contiene no finitos.")

    # ── §3.5 Identidad exacta de rangos MV ──────────────────────────────
    def _exact_mayer_vietoris_prediction(
        self,
        bA: int, bB: int, bI: int, bU: int,
        *,
        image_rank_h1_intersection: Optional[int],
        connecting_boundary_rank: Optional[int],
    ) -> Tuple[int, int, int, List[str]]:
        r"""
        rank(im φ) ≤ min(β₁(I), β₁(A)+β₁(B))
        rank(∂)    ≤ min(β₁(U), β₀(I))   — sin β₀, cota β₁(U).

        Contrato aditivo (por defecto, cuando los rangos no se miden):
            rank(im φ) := β₁(I),   rank(∂) := 0
        ⇒  expected = β₁(A)+β₁(B)−β₁(I)
        que es el caso de φ inyectiva y ∂ nula (cubiertas acíclicas
        en la intersección a nivel H₁, sin 1-ciclos que mueran en H₀).
        """
        notes: List[str] = []
        image_rank_upper = min(bI, bA + bB)
        if image_rank_h1_intersection is None:
            image_rank = bI
            notes.append(
                f"Contrato aditivo: rank(im φ) := β₁(A∩B) = {bI} "
                f"(cota ≤ {image_rank_upper})."
            )
        else:
            validated = self._validate_optional_rank(
                "image_rank_h1_intersection",
                image_rank_h1_intersection,
                upper_bound=image_rank_upper,
            )
            image_rank = bI if validated is None else validated
            notes.append(
                f"Modo exacto: rank(im φ) = {image_rank} "
                f"(cota ≤ {image_rank_upper})."
            )

        boundary_rank_upper = bU
        if connecting_boundary_rank is None:
            boundary_rank = 0
            notes.append("Contrato aditivo: rank(∂) := 0.")
        else:
            validated = self._validate_optional_rank(
                "connecting_boundary_rank",
                connecting_boundary_rank,
                upper_bound=boundary_rank_upper,
            )
            boundary_rank = 0 if validated is None else validated
            notes.append(f"Modo exacto: rank(∂) = {boundary_rank}.")

        expected = int(bA + bB - image_rank + boundary_rank)
        return image_rank, boundary_rank, expected, notes

    # ── §3.6 Desigualdad débil MV ───────────────────────────────────────
    def _verify_weak_mayer_vietoris_bound(
        self, bA: int, bB: int, bI: int, bU: int,
    ) -> Tuple[bool, str, int]:
        r"""
        |β₁(A∪B) − (β₁(A)+β₁(B))| ≤ β₁(A∩B).

        Consecuencia de 0 ≤ rank(im φ) ≤ β₁(I) y 0 ≤ rank(∂) ≤ β₁(U)
        bajo la identidad exacta; no requiere medir rangos.
        """
        lhs = abs(bU - (bA + bB))
        rhs = bI
        residual = max(0, lhs - rhs)
        satisfied = residual == 0
        msg = (
            f"Desig. débil MV: |Δβ₁_raw|={lhs} "
            f"{'≤' if satisfied else '>'} β₁(I)={rhs} "
            f"(residual={residual})."
        )
        return satisfied, msg, residual

    # ── §3.7 Sucesión H₀ ────────────────────────────────────────────────
    def _verify_h0_sequence(
        self,
        betti_0_A: Optional[int],
        betti_0_B: Optional[int],
        betti_0_intersection: Optional[int],
        betti_0_union: Optional[int],
        connecting_boundary_rank: int,
    ) -> Tuple[Optional[bool], str, Optional[int]]:
        r"""
        Exactitud en H₀:
            β₀(A∪B) = β₀(A)+β₀(B)−β₀(A∩B)+rank(∂).

        ψ₀ es epi; ker ψ₀ = im φ₀; ker φ₀ = im ∂.
        """
        if any(
            b is None
            for b in (betti_0_A, betti_0_B, betti_0_intersection, betti_0_union)
        ):
            return None, "Verificación H₀ omitida (β₀ faltantes).", None
        b0A = self._validate_betti_number("betti_0_A", betti_0_A)
        b0B = self._validate_betti_number("betti_0_B", betti_0_B)
        b0I = self._validate_betti_number("betti_0_intersection", betti_0_intersection)
        b0U = self._validate_betti_number("betti_0_union", betti_0_union)
        expected = b0A + b0B - b0I + int(connecting_boundary_rank)
        residual = int(b0U - expected)
        valid = residual == 0
        msg = (
            f"H₀: β₀(A∪B)={b0U}, expected={expected}, Δβ₀={residual}. "
            f"{'✓' if valid else '✗'}"
        )
        return valid, msg, residual

    # ── §3.8 Euler–Poincaré ─────────────────────────────────────────────
    def _verify_euler_poincare(
        self,
        betti_0_A: Optional[int],
        betti_0_B: Optional[int],
        betti_0_intersection: Optional[int],
        betti_0_union: Optional[int],
        bA: int, bB: int, bI: int, bU: int,
    ) -> Tuple[Optional[bool], str, Optional[int], Optional[int], Optional[int]]:
        r"""
        Inclusión-exclusión de la característica de Euler (1-esqueletos):
            χ(X) = β₀(X) − β₁(X),
            χ(A∪B) = χ(A) + χ(B) − χ(A∩B).

        Identidad topológica independiente de la exactitud en un grado.
        """
        if any(
            b is None
            for b in (betti_0_A, betti_0_B, betti_0_intersection, betti_0_union)
        ):
            return None, "Euler–Poincaré omitido: β₀ faltantes.", None, None, None
        b0A = self._validate_betti_number("betti_0_A", betti_0_A)
        b0B = self._validate_betti_number("betti_0_B", betti_0_B)
        b0I = self._validate_betti_number("betti_0_intersection", betti_0_intersection)
        b0U = self._validate_betti_number("betti_0_union", betti_0_union)
        chi_A = b0A - bA
        chi_B = b0B - bB
        chi_I = b0I - bI
        chi_U = b0U - bU
        expected = chi_A + chi_B - chi_I
        residual = int(chi_U - expected)
        valid = residual == 0
        msg = (
            f"Euler–Poincaré: χ(A∪B)={chi_U}, expected={expected}, "
            f"Δχ={residual}. {'✓' if valid else '✗'}"
        )
        return valid, msg, int(chi_U), int(expected), residual

    # ── §3.9 Jensen–Shannon propio + defecto relativo ───────────────────
    def _compute_jensen_shannon_report(
        self,
        bA: int, bB: int, bI: int, bU: int, expected: int,
    ) -> JensenShannonReport:
        r"""
        JS(P‖Q) sobre los 4-vectores de Betti observados vs. predichos,
        más el defecto relativo v3 (escala [0, 1)).
        """
        delta = abs(int(bU) - int(expected))
        denom = 1 + max(abs(int(bU)), abs(int(expected)))
        rel = float(delta) / float(denom)

        obs = np.array([bA, bB, bI, max(bU, 0)], dtype=np.float64)
        prd = np.array([bA, bB, bI, max(expected, 0)], dtype=np.float64)
        p = obs + _JS_FLOOR
        q = prd + _JS_FLOOR
        p = p / float(np.sum(p))
        q = q / float(np.sum(q))
        m = 0.5 * (p + q)
        kl_pm = float(np.sum(p * np.log(p / m)))
        kl_qm = float(np.sum(q * np.log(q / m)))
        js = max(0.0, min(_JS_MAX, 0.5 * (kl_pm + kl_qm)))
        return JensenShannonReport(
            js_divergence=js,
            relative_defect=rel,
            observed_simplex=tuple(float(x) for x in p),
            predicted_simplex=tuple(float(x) for x in q),
        )

    # ── §3.10 Corrección homoclínica de Betti ───────────────────────────
    def _homoclinic_betti_correction(
        self,
        seed: Optional[OrientationSeed],
    ) -> HomoclinicBettiCorrection:
        r"""
        n_extra = min(transverse_zeros, cap).
        contradicts_nilpotence ⇔ n_extra > 0 ∧ Spec(A)={0} ∧ tangle.
        """
        empty = HomoclinicBettiCorrection(
            transverse_zeros=0, simple_zeros=0, splitting_distance=0.0,
            extra_generators=0, pressure=0.0, contradicts_nilpotence=False,
        )
        if seed is None:
            return empty
        pkt = seed.observer_packet
        tz = int(max(0, pkt.melnikov.transverse_zeros))
        sz = int(max(0, pkt.melnikov.simple_zeros))
        extra = int(min(tz, _HOMOCLINIC_BETTI_CAP))
        pressure = float(min(1.0, extra / (1.0 + max(sz, 1))))
        contradicts = bool(
            extra > 0
            and pkt.melnikov.tangle_detected
            and pkt.spectra.is_strictly_nilpotent
        )
        return HomoclinicBettiCorrection(
            transverse_zeros=tz,
            simple_zeros=sz,
            splitting_distance=float(pkt.melnikov.splitting_distance),
            extra_generators=extra,
            pressure=pressure,
            contradicts_nilpotence=contradicts,
        )

    # ── §3.11 Rigidez KAM de la homología ───────────────────────────────
    def _kam_homology_rigidity(
        self,
        seed: Optional[OrientationSeed],
    ) -> KAMHomologyRigidity:
        r"""Si KAM estable ⇒ defecto homológico admisible = 0."""
        if seed is None:
            return KAMHomologyRigidity(
                is_rigid=False, epsilon_ratio=float("inf"),
                diophantine=False, tori_persisting=0, admissible_defect=0,
            )
        kam = seed.observer_packet.kam
        thr = max(float(kam.epsilon_KAM_threshold), _MACHINE_EPSILON)
        ratio = float(kam.epsilon_perturbation) / thr
        rigid = bool(kam.is_KAM_stable and kam.diophantine_satisfied)
        return KAMHomologyRigidity(
            is_rigid=rigid,
            epsilon_ratio=ratio,
            diophantine=bool(kam.diophantine_satisfied),
            tori_persisting=int(kam.tori_persisting),
            admissible_defect=0 if rigid else 0,  # fail-secure: nunca se relaja
        )

    # ── §3.12 Notas moduladas por el OrientationSeed ────────────────────
    def _modulate_notes_from_orientation_seed(
        self,
        seed: Optional[OrientationSeed],
        homo: HomoclinicBettiCorrection,
        kam_r: KAMHomologyRigidity,
        delta: int,
    ) -> List[str]:
        r"""Integra N3 / Poincaré en el certificado (sin alterar Δβ₁)."""
        if seed is None:
            return ["OrientationSeed ausente: Φ₃ opera en modo puramente homológico."]
        notes = [
            f"OrientationSeed hash={seed.packet_hash} "
            f"loss={seed.final_loss:.4e} idem={seed.idempotent} "
            f"‖θ*‖={float(np.linalg.norm(seed.theta_final)):.4f} "
            f"‖α*‖={float(np.linalg.norm(seed.alpha_final)):.4f} "
            f"‖O*‖_F={seed.O_norm:.4e} height={seed.poset_height}.",
            f"KAM_rigid={kam_r.is_rigid} ε/ε_KAM={kam_r.epsilon_ratio:.3e} "
            f"dio={kam_r.diophantine} tori={kam_r.tori_persisting} "
            f"preserved={seed.kam_preserved}.",
            f"Homoclínico: tz={homo.transverse_zeros} sz={homo.simple_zeros} "
            f"split={homo.splitting_distance:.3e} pressure={homo.pressure:.3f} "
            f"tangle={seed.melnikov_chaotic} "
            f"contradicts_ν={homo.contradicts_nilpotence}.",
        ]
        cert = seed.n3_certificate
        notes.append(
            f"N3: ΔL={cert.final_loss - cert.initial_loss:.3e} "
            f"monotone={cert.monotone_decrease} "
            f"‖Ψz−z‖={cert.residual_fixed_point:.3e} "
            f"Lip∇={cert.lipschitz_grad_bound:.3e}."
        )
        if homo.contradicts_nilpotence:
            notes.append(
                "TENSIÓN Φ₁/Poincaré: Spec(A)={0} pero Melnikov detecta "
                "herradura (ceros transversales).  MV no suma esos "
                "generadores a expected (DAG causal los prohíbe)."
            )
        if kam_r.is_rigid and delta != 0:
            notes.append(
                "TENSIÓN KAM/MV: homología debería ser rígida (toros "
                "persistentes) pero Δβ₁≠0 — ruptura estructural, no numérica."
            )
        if not cert.idempotent:
            notes.append(
                "N3 no idempotente: el veredicto Ω₃ es estable sólo al "
                "primer orden (punto fijo inexacto)."
            )
        return notes

    # ── §3.13 Consistencia Poincaré ↔ homología ─────────────────────────
    def _poincare_homology_consistency(
        self,
        seed: Optional[OrientationSeed],
        is_homologous: bool,
        homo: HomoclinicBettiCorrection,
    ) -> List[str]:
        r"""Diagnósticos de compatibilidad flujo celeste / 1-homología."""
        if seed is None:
            return []
        notes: List[str] = []
        pkt = seed.observer_packet
        rm = pkt.return_map
        if rm.is_periodic_orbit and pkt.spectra.is_strictly_nilpotent:
            notes.append(
                "P periódica ∧ Spec={0}: órbita de retorno con |μ|≃1 en un "
                "flujo nilpotente — Floquet numéricamente degenerado, no un ciclo DAG."
            )
        lya = rm.stability_exponent
        if lya > _LYAPUNOV_HYPERBOLIC_TOL and pkt.spectra.is_strictly_nilpotent:
            notes.append(
                f"λ_Lya={lya:.3e} > 0 con ρ(A)≤tol: expansión Benettin residual "
                f"(horizonte corto), no un ciclo espectral."
            )
        if (not pkt.arnold.is_adiabatic) and is_homologous:
            notes.append(
                f"D_Arnold={pkt.arnold.total_diffusion:.3e} no adiabática pero "
                f"Δβ₁=0: homología rígida pese a fuga inter-resonancia "
                f"(t_Nek={pkt.arnold.nekhoroshev_time:.3e})."
            )
        if not rm.is_transverse_family:
            notes.append(
                "Σ no transversal: P mal definida; la presión homoclínica "
                f"({homo.pressure:.3f}) es una cota superior conservadora."
            )
        return notes

    # ── §3.14 Fiedler del 1-esqueleto ───────────────────────────────────
    def _fiedler_from_adjacency(
        self,
        adjacency_matrix: NDArray[Any],
        *,
        zero_threshold: float = _BASE_SPECTRAL_TOLERANCE,
    ) -> float:
        r"""
        λ₂ del Laplaciano simetrizado L = D − |A|.
        λ₂ > 0 ⇔ el grafo subyacente (no dirigido) es conexo.
        """
        A = self._validate_and_condition_adjacency(
            adjacency_matrix,
            zero_threshold=zero_threshold,
            allow_signed_weights=True,
            require_boolean=False,
        )
        n = int(A.shape[0])
        if n < 2:
            return 0.0
        absA = np.abs(A)
        degree = np.sum(absA, axis=1)
        L = np.diag(degree) - absA
        L_sym = 0.5 * (L + L.T)
        try:
            evals = np.sort(np.real(np.linalg.eigvalsh(L_sym)))
            return float(evals[1]) if evals.size > 1 else 0.0
        except Exception:
            return 0.0

    # ── §3.15 Cota de Daleckii–Krein ────────────────────────────────────
    def _daleckii_krein_bound(self, lambda2: float) -> float:
        r"""
        Lip(f) de la proyección espectral en el hueco de Fiedler:

            L_max = 1 / (2 λ₂^{3/2})

        (divididas diferencias de Daleckii–Krein para f = 1_{λ>0} en el
        Laplaciano; controla cuánto puede saltar H₀/H₁ bajo ‖ΔA‖₂ ≤ 1).
        """
        eff = max(float(lambda2), _WILKINSON_LIMIT)
        return float(_DALECKII_KREIN_PREFACTOR / (eff ** _DALECKII_KREIN_EXP))

    # ── §3.16 Factores fallidos de Ω₃ ───────────────────────────────────
    def _collect_failing_invariants(
        self,
        spectra: NilpotenceAuditData,
        filtration: PosetFiltrationData,
        mv: MayerVietorisAuditData,
        seed: OrientationSeed,
        pkt: ObserverPacket,
    ) -> List[str]:
        r"""Lista canónica de factores que impiden que Ω₃ sea terminal."""
        failing: List[str] = []
        if not spectra.is_strictly_nilpotent:
            failing.append(f"Φ₁(ρ={spectra.spectral_radius:.4e}>{spectra.tolerance:.4e})")
        if not filtration.is_monotonic_filtration:
            failing.append("Φ₂(filtración no monótona)")
        if filtration.height_nilpotence_consistent is False:
            failing.append("Φ₂(height ≰ ν)")
        if not mv.is_fusion_homologous:
            failing.append(f"Φ₃(Δβ₁={mv.delta_betti_1})")
        if mv.euler_characteristic_valid is False:
            failing.append("Φ₃(Euler–Poincaré)")
        if mv.weak_bound_satisfied is False:
            failing.append("Φ₃(desigualdad débil MV)")
        if not math.isfinite(seed.final_loss):
            failing.append("Φ₂(N3 divergente)")
        if not pkt.melnikov.is_admissible:
            failing.append(f"Φ₁(Melnikov ‖M‖_∞={pkt.melnikov.max_magnitude:.3e})")
        if not pkt.kam.is_KAM_stable:
            failing.append(
                f"Φ₁(KAM ε={pkt.kam.epsilon_perturbation:.3e}"
                f"≥{pkt.kam.epsilon_KAM_threshold:.3e})"
            )
        return failing

    # ── §3.17 Objeto terminal Ω₃ ────────────────────────────────────────
    def _omega3_terminal_object(
        self,
        spectra: NilpotenceAuditData,
        filtration: PosetFiltrationData,
        mv: MayerVietorisAuditData,
        seed: OrientationSeed,
        pkt: ObserverPacket,
    ) -> Omega3Certificate:
        r"""
        Pullback de los tres certificados + Poincaré + N3.

        composition_hash amarra las huellas de Φ₁ (espectro), Φ₂ (packet)
        y Φ₃ (Δβ₁, JS, Euler) en un único sello.
        """
        n3_finite = bool(
            math.isfinite(seed.final_loss) and seed.n3_certificate.finite
        )
        poincare_adm = bool(pkt.melnikov.is_admissible and pkt.kam.is_KAM_stable)
        phi1 = bool(spectra.is_strictly_nilpotent)
        phi2 = bool(filtration.is_monotonic_filtration)
        phi3 = bool(mv.is_fusion_homologous)
        failing = self._collect_failing_invariants(spectra, filtration, mv, seed, pkt)
        is_term = (
            phi1 and phi2 and phi3 and n3_finite and poincare_adm
            and (mv.euler_characteristic_valid is not False)
            and (filtration.height_nilpotence_consistent is not False)
            and (mv.weak_bound_satisfied is True)
        )
        payload = (
            f"{spectra.spectrum_certificate_hash}|"
            f"{pkt.packet_hash}|{seed.packet_hash}|"
            f"{mv.delta_betti_1}|{mv.js_divergence:.12e}|"
            f"{mv.euler_characteristic_valid}|{int(is_term)}"
        )
        notes = (
            f"Ω₃={'⊤' if is_term else '⊥'} | fallos={list(failing) or '∅'} | "
            f"JS={mv.js_divergence:.4e} | pressure={mv.homoclinic_pressure:.3f}.",
        )
        return Omega3Certificate(
            phi1_nilpotent=phi1,
            phi2_monotonic=phi2,
            phi2_height_consistent=filtration.height_nilpotence_consistent,
            phi3_homologous=phi3,
            phi3_euler=mv.euler_characteristic_valid,
            n3_finite=n3_finite,
            n3_idempotent=bool(seed.idempotent),
            poincare_admissible=poincare_adm,
            kam_stable=bool(pkt.kam.is_KAM_stable),
            melnikov_admissible=bool(pkt.melnikov.is_admissible),
            is_terminal_object=bool(is_term),
            composition_hash=_sha256_short(payload),
            failing_factors=tuple(failing),
            notes=notes,
        )

    # ══════════════════════════════════════════════════════════════════════
    # §3.18  ÚLTIMO MÉTODO DE FASE 3: síntesis del objeto final de gobernanza
    #        Z_Causal = Φ₃ ∘ Φ₂ ∘ Φ₁   —  no hay Fase 4.
    # ══════════════════════════════════════════════════════════════════════
    def _phase3_terminal_synthesis(
        self,
        adjacency_matrix: NDArray[Any],
        edges: Sequence[Tuple[str, str]],
        node_strata: Mapping[str, Any],
        betti_1_A: int,
        betti_1_B: int,
        betti_1_intersection: int,
        betti_1_union: int,
        *,
        adjacency_zero_threshold: float = _BASE_SPECTRAL_TOLERANCE,
        allow_signed_weights: bool = False,
        require_boolean: bool = False,
        deep_nilpotence_audit: bool = True,
        power_audit_max_dim: int = _DEFAULT_POWER_AUDIT_MAX_DIM,
        allow_unknown_nodes: bool = False,
        strict_filtration: bool = False,
        node_order: Optional[Sequence[str]] = None,
        cross_check_adjacency_support: bool = False,
        image_rank_h1_intersection: Optional[int] = None,
        connecting_boundary_rank: Optional[int] = None,
        betti_0_A: Optional[int] = None,
        betti_0_B: Optional[int] = None,
        betti_0_intersection: Optional[int] = None,
        betti_0_union: Optional[int] = None,
        n3_cycles: int = _N3_DEFAULT_CYCLES,
        raise_on_veto: bool = True,
        require_transverse_section: bool = False,
    ) -> CausalGovernanceState:
        r"""
        ÚLTIMO MÉTODO DE FASE 3 — Síntesis del objeto final de gobernanza.

        Composición funtorial completa:  Z_Causal = Φ₃ ∘ Φ₂ ∘ Φ₁.

        Invariantes certificados:
            1. Spec(A) = {0}                         [ciclicidad espectral]
            2. f: (V, ≤_DAG) → (ℤ, ≤) monótona       [morfismo de Posets DIKW]
            3. Δβ₁ = 0 ∧ χ(A∪B)=χ(A)+χ(B)−χ(A∩B)    [MV + Euler]
            4. Kernel N3 convergente (loss finita)
            5. Poincaré admisible (Melnikov + KAM)
            6. ν ≤ longest_path+1 ≤ height+1         [rango del poset]

        Anidamiento cerrado:
            _phase3_terminal_synthesis
              = Ω₃ ∘ _intercept_mayer_vietoris_sequence
                  ∘ _phase2_terminal_bridge_to_phase3
        """
        (
            spectra, filtration, observer_packet,
            orientation_seed, mv_audit,
        ) = self._phase2_terminal_bridge_to_phase3(
            adjacency_matrix, edges, node_strata,
            betti_1_A, betti_1_B, betti_1_intersection, betti_1_union,
            adjacency_zero_threshold=adjacency_zero_threshold,
            allow_signed_weights=allow_signed_weights,
            require_boolean=require_boolean,
            deep_nilpotence_audit=deep_nilpotence_audit,
            power_audit_max_dim=power_audit_max_dim,
            allow_unknown_nodes=allow_unknown_nodes,
            strict_filtration=strict_filtration,
            node_order=node_order,
            cross_check_adjacency_support=cross_check_adjacency_support,
            image_rank_h1_intersection=image_rank_h1_intersection,
            connecting_boundary_rank=connecting_boundary_rank,
            betti_0_A=betti_0_A, betti_0_B=betti_0_B,
            betti_0_intersection=betti_0_intersection,
            betti_0_union=betti_0_union,
            n3_cycles=n3_cycles,
            raise_on_veto=raise_on_veto,
            require_transverse_section=require_transverse_section,
        )

        omega3 = self._omega3_terminal_object(
            spectra, filtration, mv_audit, orientation_seed, observer_packet,
        )
        is_valid = bool(omega3.is_terminal_object)

        if not is_valid and raise_on_veto:
            raise PipelineDirectorAgentError(
                "Gobernanza causal inválida. Factores fallidos: "
                + (", ".join(omega3.failing_factors) or "desconocido")
                + f". hash={omega3.composition_hash}."
            )

        state = CausalGovernanceState(
            governance_id=str(uuid.uuid4()),
            nilpotence_audit=spectra,
            filtration_audit=filtration,
            mayer_vietoris_audit=mv_audit,
            omega3=omega3,
            observer_packet=observer_packet,
            orientation_seed=orientation_seed,
            is_causally_valid=is_valid,
            composition_hash=omega3.composition_hash,
            generated_at_utc=_utc_timestamp(),
        )

        logger.info(
            "Gobernanza Z_Causal=Φ₃∘Φ₂∘Φ₁ | id=%s | hash=%s | ρ(A)=%.4e | "
            "κ₂(Q)=%.4f | ν=%s | height=%s | monotonic=%s | Δβ₁=%d | "
            "JS=%.4e | χ_ok=%s | loss_N3=%.4e | idem=%s | KAM=%s | "
            "pressure=%.3f | Ω₃=%s | valid=%s",
            state.governance_id, state.composition_hash,
            spectra.spectral_radius, spectra.schur_condition_number,
            spectra.nilpotence_index, filtration.poset_height,
            filtration.is_monotonic_filtration, mv_audit.delta_betti_1,
            mv_audit.js_divergence, mv_audit.euler_characteristic_valid,
            orientation_seed.final_loss, orientation_seed.idempotent,
            observer_packet.kam.is_KAM_stable, mv_audit.homoclinic_pressure,
            omega3.is_terminal_object, state.is_causally_valid,
        )
        return state


# ══════════════════════════════════════════════════════════════════════════════
# §D. ORQUESTADOR SUPREMO: PIPELINE DIRECTOR AGENT
# ══════════════════════════════════════════════════════════════════════════════

class PipelineDirectorAgent(Morphism, Phase3_MayerVietorisInterceptor):
    r"""
    El Custodio de la Causalidad Funtorial en la Malla Agéntica.

    Herencia lineal (cadena funtorial estricta):

        PipelineDirectorAgent
          → Phase3_MayerVietorisInterceptor          [Φ₃  DECIDE]
             → Phase2_PosetFiltrationAuditor         [Φ₂  ORIENT + kernel N3]
                → Phase1_SpectralNilpotenceCertifier [Φ₁  OBSERVE + Poincaré]
                   → Morphism (base del Topos MIC)

    Punto de entrada público: ``execute_causal_governance`` ≡ Φ₃∘Φ₂∘Φ₁.
    """

    def execute_causal_governance(
        self,
        adjacency_matrix: NDArray[Any],
        edges: Sequence[Tuple[str, str]],
        node_strata: Mapping[str, Any],
        betti_1_A: int,
        betti_1_B: int,
        betti_1_intersection: int,
        betti_1_union: int,
        *,
        adjacency_zero_threshold: float = _BASE_SPECTRAL_TOLERANCE,
        allow_signed_weights: bool = False,
        require_boolean: bool = False,
        deep_nilpotence_audit: bool = True,
        power_audit_max_dim: int = _DEFAULT_POWER_AUDIT_MAX_DIM,
        allow_unknown_nodes: bool = False,
        strict_filtration: bool = False,
        node_order: Optional[Sequence[str]] = None,
        cross_check_adjacency_support: bool = False,
        image_rank_h1_intersection: Optional[int] = None,
        connecting_boundary_rank: Optional[int] = None,
        betti_0_A: Optional[int] = None,
        betti_0_B: Optional[int] = None,
        betti_0_intersection: Optional[int] = None,
        betti_0_union: Optional[int] = None,
        n3_cycles: int = _N3_DEFAULT_CYCLES,
        raise_on_veto: bool = True,
        require_transverse_section: bool = False,
    ) -> CausalGovernanceState:
        r"""
        Punto de entrada público: ejecuta Z_Causal = Φ₃ ∘ Φ₂ ∘ Φ₁.
        Delega en `_phase3_terminal_synthesis`.
        """
        return self._phase3_terminal_synthesis(
            adjacency_matrix, edges, node_strata,
            betti_1_A, betti_1_B, betti_1_intersection, betti_1_union,
            adjacency_zero_threshold=adjacency_zero_threshold,
            allow_signed_weights=allow_signed_weights,
            require_boolean=require_boolean,
            deep_nilpotence_audit=deep_nilpotence_audit,
            power_audit_max_dim=power_audit_max_dim,
            allow_unknown_nodes=allow_unknown_nodes,
            strict_filtration=strict_filtration,
            node_order=node_order,
            cross_check_adjacency_support=cross_check_adjacency_support,
            image_rank_h1_intersection=image_rank_h1_intersection,
            connecting_boundary_rank=connecting_boundary_rank,
            betti_0_A=betti_0_A, betti_0_B=betti_0_B,
            betti_0_intersection=betti_0_intersection,
            betti_0_union=betti_0_union,
            n3_cycles=n3_cycles,
            raise_on_veto=raise_on_veto,
            require_transverse_section=require_transverse_section,
        )

    def audit_poincare_causal_pipeline(
        self,
        betti_1_sub_A: int,
        betti_1_sub_B: int,
        betti_1_intersection: int,
        betti_1_union: int,
        fiedler_value_lambda2: float,
        *,
        raise_on_veto: bool = True,
    ) -> PipelineCausalAuditState:
        r"""
        Auditoría ligera Poincaré–Mayer–Vietoris + cota Daleckii–Krein.

        Invariantes:
            Δβ₁ ≡ 0     (contrato aditivo: rank φ = β₁(I), rank ∂ = 0)
            L_max = 1 / (2 λ₂^{3/2})
            coherencia ⇔ Δβ₁ = 0  ∧  λ₂ > ε_W  ∧  |Δβ₁| ≤ L_max  (trivial).
        """
        bA = self._validate_betti_number("betti_1_sub_A", betti_1_sub_A)
        bB = self._validate_betti_number("betti_1_sub_B", betti_1_sub_B)
        bI = self._validate_betti_number("betti_1_intersection", betti_1_intersection)
        bU = self._validate_betti_number("betti_1_union", betti_1_union)

        expected = bA + bB - bI
        delta = float(abs(bU - expected))
        if delta > 0.0 and raise_on_veto:
            raise TopologicalInvariantError(
                f"[PIPELINE_CAUSAL_VETO] Δβ₁={delta} ≠ 0 (Mayer–Vietoris)."
            )

        lam2 = float(fiedler_value_lambda2)
        if not math.isfinite(lam2):
            raise MayerVietorisInputError(
                f"fiedler_value_lambda2 no finito: {fiedler_value_lambda2!r}."
            )
        l_max = self._daleckii_krein_bound(lam2)
        lip_ok = delta <= l_max + _OMEGA3_SOFT_REL_DEFECT
        coherent = (delta == 0.0) and (lam2 > _WILKINSON_LIMIT) and lip_ok
        return PipelineCausalAuditState(
            mayer_vietoris_residual=delta,
            daleckii_krein_bound=l_max,
            fiedler_lambda2=lam2,
            lipschitz_coherent=bool(lip_ok),
            is_funtorially_coherent=bool(coherent),
        )

    def audit_poincare_causal_pipeline_from_adjacency(
        self,
        adjacency_matrix: NDArray[Any],
        betti_1_sub_A: int,
        betti_1_sub_B: int,
        betti_1_intersection: int,
        betti_1_union: int,
        *,
        adjacency_zero_threshold: float = _BASE_SPECTRAL_TOLERANCE,
        raise_on_veto: bool = True,
    ) -> PipelineCausalAuditState:
        r"""
        Variante que extrae λ₂ del Laplaciano de |A| (Daleckii–Krein
        calibrado por el 1-esqueleto real, no por un Fiedler externo).
        """
        lam2 = self._fiedler_from_adjacency(
            adjacency_matrix, zero_threshold=adjacency_zero_threshold,
        )
        return self.audit_poincare_causal_pipeline(
            betti_1_sub_A, betti_1_sub_B,
            betti_1_intersection, betti_1_union,
            lam2, raise_on_veto=raise_on_veto,
        )

    def __call__(self, state: Any = None, **kwargs: Any) -> Any:
        r"""Invocación como morfismo categórico del Topos MIC."""
        if kwargs:
            return self.execute_causal_governance(**kwargs)
        return state


# ══════════════════════════════════════════════════════════════════════════════
# §E. EXPORTACIÓN CANÓNICA  (Fases 1 ∪ 2 ∪ 3)
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Excepciones
    "PipelineDirectorAgentError",
    "AdjacencyMatrixFormatError",
    "CausalLoopVetoError",
    "NilpotenceIndexVetoError",
    "StratumMappingError",
    "SelfLoopVetoError",
    "FiltrationViolationVeto",
    "AdjacencySupportVetoError",
    "MayerVietorisInputError",
    "HomologicalFusionVeto",
    "EulerPoincareMismatchError",
    "PoincareTransversalityError",
    "MelnikovTangleError",
    "KAMBreakdownError",
    "RecursiveSelfImprovementError",
    # DTOs Fase 1
    "SchurBlock",
    "NilpotenceAuditData",
    "PoincareSection",
    "ReturnMapReport",
    "MelnikovCertificate",
    "KAMCertificate",
    "ArnoldDiffusionMetric",
    "ObserverPacket",
    # DTOs Fase 2
    "EdgeSlackRecord",
    "PosetHeightCertificate",
    "TransitiveClosureData",
    "PosetFiltrationData",
    "ImprovementState",
    "N3ConvergenceCertificate",
    "OrientationSeed",
    # DTOs Fase 3
    "ExactnessProfile",
    "HomoclinicBettiCorrection",
    "KAMHomologyRigidity",
    "JensenShannonReport",
    "PipelineCausalAuditState",
    "MayerVietorisAuditData",
    "Omega3Certificate",
    "CausalGovernanceState",
    # Certificadores por fase
    "Phase1_SpectralNilpotenceCertifier",
    "Phase2_PosetFiltrationAuditor",
    "Phase3_MayerVietorisInterceptor",
    # Kernel N3
    "RecursiveSelfImprovementKernel",
    # Orquestador supremo
    "PipelineDirectorAgent",
]


# ══════════════════════════════════════════════════════════════════════════════
# FIN DEL MÓDULO  —  Z_Causal = Φ₃ ∘ Φ₂ ∘ Φ₁  está cerrado.
# Concatenar Fases 1 + 2 + 3 en
#     app/agents/strategy/pipeline_director_agent.py
# ══════════════════════════════════════════════════════════════════════════════