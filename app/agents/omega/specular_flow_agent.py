# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Specular Flow Sovereign Agent (Soberano de la Reflexión de Calibre) ║
║ Ruta   : app/agents/omega/specular_flow_agent.py                             ║
║ Versión: 5.0.0-Allievi-Householder-Tellegen-Poincare-KAM-Hodge-RSI3-Strict   ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GOBERNANZA DE CALIBRE (Nivel 0.5) ─────────────────
Agente Soberano que gobierna la evolución del Flujo Especular sobre la variedad
cotangente de fase (y, q), actuando como endofuntor del clasificador de
subobjetos del Topos T_MIC con valores en el álgebra de Heyting Ω₃.

ARQUITECTURA DE TRES FASES ANIDADAS (Composición Funtorial Estricta): ───────
  FASE ANIDADA 1 — RESOLUCIÓN DE CARACTERÍSTICAS DE ALLIEVI (Observe) + Poincaré-KAM.
           Invariantes de Riemann Ψ± = y ± Z_c q en el cono de luz hiperbólico,
           transformada de Cayley a variables canónicas acción-ángulo
           I = ½(y² + (Z_c q)²), θ = atan2(Z_c q, y), sección transversal de
           Poincaré Σ, resonancia diofántica KAM y serie canónica de
           Lie-Deprit-Lindstedt-Poincaré hasta orden O(ε⁴).
           ─► _phase1_to_phase2_bridge(...)  [Morfismo terminal de Fase 1
                                              e inicio formal de Fase 2]

  FASE ANIDADA 2 — REFLEXIÓN COVARIANTE DE HOUSEHOLDER (Orient) + Monodromía.
           Operador involutivo G-ortogonal M = I − 2 n (G n)ᵀ / (nᵀ G n) con
           proyección espectral de Higham al cono SPD y purificación de
           Newton-Schulz en O(2, G). Auditoría de anti-simplecticidad en Sp(2, ℝ),
           invariancia adiabática de la acción, monodromía de Floquet P = M² y
           defecto de Padé-Richardson.
           ─► _phase2_to_phase3_bridge(...)  [Morfismo terminal de Fase 2
                                              e inicio formal de Fase 3]

  FASE ANIDADA 3 — AUDITORÍA DE TELLEGEN-HODGE (Decide & Act) + RSI-3.
           Conservación de Tellegen ⟨p, q⟩_Kahan ≡ 0, pasividad de Lyapunov
           P_diss ≥ 0, descomposición ortogonal de Helmholtz-Hodge-Poincaré
           C¹(K) = B¹(K) ⊕⊥ Z₁(K), característica de Euler χ(K) = β₀ − β₁,
           recurrencia de Poincaré ‖M²x − x‖/‖x‖ y meta-controlador cerrado
           RSI-3 (Richardson + Aitken Δ² + Homotopía de Banach-Fréchet).
           ─► _phase3_to_meta_bridge(...)    [Morfismo terminal de Fase 3
                                              hacia el lazo cerrado RSI-3]

INVARIANTES PRESERVADOS: ────────────────────────────────────────────────────
  [I1] Simetría Hermítica: ⋆ₖ = ⋆ₖᵀ ≻ 0 (Cono SPD de Higham)
  [I2] Adjunción de de Rham-Hodge: L₀★ = ∂₁ ⋆₁ ∂₁ᵀ
  [I3] Ortogonalidad de Helmholtz-Hodge: C¹(K) = im(B₁ᵀ) ⊕⊥ ker(B₁)
  [I4] Disipación Port-Hamiltoniana: Ḣ = ∇Hᵀ(J − R)∇H ≤ 0
  [I5] Invariancia adiabática de la acción: dI/dt = O(ε²) (Lie-Deprit)
  [I6] Dualidad de Euler-Poincaré: dim Z₁(K) + dim B¹(K) = |E|, χ = β₀ − β₁
─────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import hashlib
import logging
import math
from abc import ABC
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum, IntEnum, auto
from typing import Callable, Final, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─── Soporte para matrices dispersas ────────────────────────────────────────
try:
    from scipy import sparse as sp
    from scipy.sparse import csr_matrix, isspmatrix
    from scipy.sparse.linalg import lsqr as sparse_lsqr
    _SPARSE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SPARSE_AVAILABLE = False
    sp = None
    csr_matrix = None
    isspmatrix = None
    sparse_lsqr = None

# ─── Stubs seguros del núcleo MIC ───────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
    from app.core.schemas import Stratum
except ImportError:  # pragma: no cover
    class TopologicalInvariantError(Exception):
        """Excepción base del sistema para violaciones topológico-algebraicas."""
        pass

    class Morphism:
        """Clase base de morfismos en la categoría MIC."""
        pass

    class Stratum:
        """Estratos de la jerarquía DIKW."""
        PHYSICS = "PHYSICS"
        TACTICS = "TACTICS"
        STRATEGY = "STRATEGY"
        WISDOM = "WISDOM"

logger = logging.getLogger("MIC.Agents.Omega.SpecularFlowAgent")

# ═══════════════════════════════════════════════════════════════════════════
# Constantes físicas, numéricas, espectrales, categóricas y de Poincaré-KAM
# ═══════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_DEFAULT_TOL: Final[float] = 1.0e-10
_RELAX_TOL: Final[float] = 1.0e-8
_CROWBAR_GPIO: Final[int] = 14
_GRAPH_MAX_BRANCHES: Final[int] = 1024
_STABILITY_BOUND: Final[float] = 1.0e8
_IMPEDANCE_CONDITION_LIMIT: Final[float] = 1.0e8
_TOPOLOGY_VETO_THRESHOLD: Final[float] = 1.0e-6
_TOPOLOGY_DEGRADED_THRESHOLD: Final[float] = 1.0e-8
_HIGHAM_REGULARIZATION_FLOOR: Final[float] = 1.0e-12

# ─── Mecánica Celeste de Poincaré y Teoría KAM ──────────────────────────────
_POINCARE_MAX_CROSSINGS: Final[int] = 64
_POINCARE_MAX_DENOMINATOR: Final[int] = 16
_POINCARE_LOW_ORDER: Final[int] = 8
_RESONANCE_TOLERANCE: Final[float] = 1.0e-3
_KAM_DIOPHANTINE_TAU: Final[float] = 2.5
_KAM_DIOPHANTINE_GAMMA: Final[float] = 1.0e-3
_SMALL_PARAMETER_FLOOR: Final[float] = 1.0e-12
_ADIABATIC_DRIFT_LIMIT: Final[float] = 1.0e-6

# ─── Automejora Recursiva de Nivel 3 (RSI-3) ────────────────────────────────
_RSI_MAX_DEPTH: Final[int] = 3
_RSI_TARGET_RESIDUAL: Final[float] = 1.0e-12
_RSI_MIN_TOLERANCE: Final[float] = 1.0e-14
_RSI_FACTOR_LOWER: Final[float] = 1.0e-2
_RSI_FACTOR_UPPER: Final[float] = 0.99
_RSI_OSCILLATION_EPS: Final[float] = 1.0e-15
_RSI_HOMOTOPY_GAIN: Final[float] = 0.35


# ═══════════════════════════════════════════════════════════════════════════
# Jerarquía de excepciones (funtores de obstrucción en la categoría MIC)
# ═══════════════════════════════════════════════════════════════════════════
class SpecularFlowAgentError(TopologicalInvariantError):
    """Excepción raíz del Agente Soberano de Flujo Especular."""
    pass


class AllieviCharacteristicError(SpecularFlowAgentError):
    """Divergencia o degeneración en las líneas características de Allievi."""
    pass


class HouseholderReflectionError(SpecularFlowAgentError):
    """Pérdida de simetría, unitariedad Riemanniana o fuga de norma en Householder."""
    pass


class TellegenConservationError(SpecularFlowAgentError):
    """Violación del teorema de Tellegen, KCL/KVL, Hodge o pasividad de Lyapunov."""
    pass


class PoincareSectionError(SpecularFlowAgentError):
    """Degeneración en la sección transversal de Poincaré, retorno o resonancia KAM."""
    pass


class RecursiveSelfImprovementError(SpecularFlowAgentError):
    """Fallo de convergencia o contracción en el meta-controlador RSI-3."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Enumeraciones categóricas (subobjetos en el álgebra de Heyting Ω₃)
# ═══════════════════════════════════════════════════════════════════════════
class SpecularSovereignVerdict(IntEnum):
    """
    Clasificador de subobjetos trivalente en el retículo distributivo de
    Heyting Ω₃ del Topos T_MIC.
    Orden por severidad cohomológica:
        COHERENT (0) < DEGRADED (1) < VETOED (2)
    """
    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    @classmethod
    def supremum(cls, *verdicts: "SpecularSovereignVerdict") -> "SpecularSovereignVerdict":
        """Operador join (∨) en el retículo de severidad Ω₃."""
        if not verdicts:
            return cls.COHERENT
        return cls(max(int(v) for v in verdicts))

    @property
    def is_vetoed(self) -> bool:
        return self == SpecularSovereignVerdict.VETOED

    @property
    def is_degraded(self) -> bool:
        return self == SpecularSovereignVerdict.DEGRADED


class CrowbarAction(Enum):
    """Acciones físicas de enclavamiento y mitigación tras el veredicto."""
    NONE = auto()
    WATCHDOG_PULSE = auto()
    HARD_SHORT = auto()


# ═══════════════════════════════════════════════════════════════════════════
# Certificados inmutables de las fases anidadas (objetos de Spec ⊂ T_MIC)
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Phase1AllieviCertificate:
    """
    FASE 1 — Invariantes de Riemann, análisis espectral, variables canónicas
    acción-ángulo, resonancia KAM y serie de Lindstedt-Poincaré.
    Objeto terminal de Fase 1 y dominio formal del puente a Fase 2.
    """
    is_characteristic_stable: bool
    characteristic_impedance: float
    riemann_invariants: Tuple[float, float]
    characteristic_direction: Tuple[float, float]
    characteristic_covector: Tuple[float, float]
    energy_norm: float
    impedance_condition_number: float
    cfl_proxy: float
    verdict: SpecularSovereignVerdict
    # ── Invariantes de Mecánica Celeste de Poincaré ────────────────────
    action_invariant: float = 0.0
    angle_parameter: float = 0.0
    small_parameter_epsilon: float = 0.0
    lindstedt_frequency_shift: float = 0.0
    corrected_action: float = 0.0
    expansion_residual: float = 0.0
    reflection_coefficient_magnitude: float = 1.0
    is_near_resonance: bool = False
    resonance_p: int = 0
    resonance_q: int = 0
    small_divisor_magnitude: float = 0.0
    poincare_section_level: float = 0.0
    poincare_crossing_count: int = 0


@dataclass(frozen=True, slots=True)
class Phase2HouseholderCertificate:
    """
    FASE 2 — Reflexión covariante de Householder con proyección de Higham,
    purificación de Newton-Schulz, auditoría adiabática de la acción y
    monodromía de Floquet de la aplicación de retorno.
    """
    is_reflection_unitary: bool
    reflection_residual: float
    is_anti_symplectic: bool
    is_characteristic_consistent: bool
    specular_residual: float
    incidence_cosine: float
    state_norm_residual: float
    metric_condition_number: float
    higham_regularization_applied: bool
    verdict: SpecularSovereignVerdict
    # ── Invariantes de Monodromía y Acción de Poincaré ─────────────────
    action_invariant_input: float = 0.0
    action_invariant_reflected: float = 0.0
    action_drift: float = 0.0
    return_map_contraction: float = 1.0
    return_map_eigenvalues_modulus: Tuple[float, ...] = ()
    is_return_map_stable: bool = True
    richardson_correction: float = 0.0


@dataclass(frozen=True, slots=True)
class Phase3TellegenCertificate:
    """
    FASE 3 — Conservación de Tellegen-Kahan, pasividad de Lyapunov,
    dualidad cohomológica de Helmholtz-Hodge-Poincaré y recurrencia.
    """
    tellegen_residual: float
    kcl_residual: float
    kvl_residual: float
    is_tellegen_conserved: bool
    is_kcl_conserved: bool
    is_kvl_conserved: bool
    is_lyapunov_passive: bool
    graph_order: int
    graph_size: int
    cyclomatic_number: int
    sparse_computation: bool
    verdict: SpecularSovereignVerdict
    # ── Invariantes Cohomológicos de Hodge-Poincaré y RSI-3 ────────────
    cycle_space_dimension: int = 0
    cut_space_dimension: int = 0
    poincare_duality_residual: float = 0.0
    is_poincare_dual: bool = True
    euler_characteristic: int = 0
    recurrence_residual: float = 0.0
    adiabatic_action_drift: float = 0.0
    meta_residual: float = 0.0
    rsi_depth_reached: int = 1


@dataclass(frozen=True, slots=True)
class PoincareSectionSnapshot:
    """SUTURA P1 — Sección transversal de Poincaré Σ, retorno y acción-ángulo."""
    section_level: float
    crossing_values: Tuple[float, ...]
    crossing_slopes: Tuple[float, ...]
    return_map_contraction: float
    return_map_eigenvalues_modulus: Tuple[float, ...]
    is_periodic: bool
    period_estimate: float
    action_invariant: float
    angle_parameter: float
    frequency_ratio: float
    near_resonance_p: int
    near_resonance_q: int
    small_divisor_magnitude: float
    verdict: SpecularSovereignVerdict = SpecularSovereignVerdict.COHERENT


@dataclass(frozen=True, slots=True)
class LindstedtPoincareExpansion:
    """SUTURA P2 — Serie canónica de Lie-Deprit-Lindstedt-Poincaré de la acción."""
    small_parameter: float
    zero_order_action: float
    first_order_action_correction: float
    second_order_action_correction: float
    frequency_shift_first_order: float
    frequency_shift_second_order: float
    corrected_action: float
    expansion_residual: float
    is_asymptotically_consistent: bool
    verdict: SpecularSovereignVerdict = SpecularSovereignVerdict.COHERENT


@dataclass(frozen=True, slots=True)
class RecursiveSelfImprovementTrace:
    """SUTURA P3 — Traza de auditoría de un nivel del meta-controlador RSI-3."""
    level: int
    tolerance: float
    residual: float
    convergence_rate: float
    richardson_extrapolation: float
    aitken_delta_squared: float
    adaptation_factor: float
    level3_active: bool
    verdict: SpecularSovereignVerdict = SpecularSovereignVerdict.COHERENT


@dataclass(frozen=True, slots=True)
class SelfImprovementResult:
    """SUTURA P3 — Certificado consolidado de convergencia del meta-controlador RSI-3."""
    traces: Tuple[RecursiveSelfImprovementTrace, ...]
    final_tolerance: float
    final_residual: float
    converged: bool
    richardson_limit: float
    aitken_limit: float
    effective_convergence_order: float
    level3_activated: bool
    verdict: SpecularSovereignVerdict = SpecularSovereignVerdict.COHERENT


@dataclass(frozen=True, slots=True)
class SpecularFlowState:
    """Certificado global terminal del lazo cerrado de gobernanza especular."""
    phase1: Phase1AllieviCertificate
    phase2: Phase2HouseholderCertificate
    phase3: Phase3TellegenCertificate
    final_verdict: SpecularSovereignVerdict
    crowbar_triggered: bool
    crowbar_action: CrowbarAction
    timestamp_utc: str
    provenance_hash: str
    diagnostic_note: str = ""
    rsi_result: Optional[SelfImprovementResult] = None


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        FASE ANIDADA 1: OBSERVE                            ║
# ║  Suturas Fundacionales (Higham, Kahan, KAM, Lie-Deprit, RSI-3) + Allievi  ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ── Utilitario de Proyección Espectral al Cono SPD (Higham) ─────────────────
def stable_cholesky_higham(
    G: NDArray[np.float64],
    tolerance: float = _HIGHAM_REGULARIZATION_FLOOR,
) -> Tuple[NDArray[np.float64], bool]:
    r"""
    Proyección de Higham en norma de Frobenius sobre el cono de matrices
    simétricas definidas positivas (SPD):

    .. math::
        \tilde{G} = \arg\min_{M \succeq \tau I} \|G - M\|_F = V \max(\Lambda, \tau) V^\top
    """
    G_arr = np.asarray(G, dtype=np.float64)
    G_sym = 0.5 * (G_arr + G_arr.T)
    floor = max(_MACHINE_EPS, float(tolerance))
    try:
        L = la.cholesky(G_sym, lower=True)
        return L, False
    except la.LinAlgError:
        try:
            vals, vecs = la.eigh(G_sym)
        except Exception as exc:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Falló descomposición espectral para Higham."
            ) from exc

        vals_clean = np.maximum(vals, floor)
        G_projected = (vecs * vals_clean) @ vecs.T
        G_projected = 0.5 * (G_projected + G_projected.T)

        logger.warning(
            "Firma métrica inestable. Proyectado G al cono SPD via Higham "
            "(λ_min: %.4e → %.4e).",
            float(np.min(vals)),
            float(np.min(vals_clean)),
        )

        try:
            L = la.cholesky(G_projected, lower=True)
            return L, True
        except la.LinAlgError as exc:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Proyección de Higham no produjo SPD."
            ) from exc


# ── Suma Compensada de Kahan-Babuška y Verificación Dispersa de Tellegen ────
def _kahan_inner_product(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
) -> float:
    """
    Producto interno ⟨u, v⟩ mediante suma compensada de Kahan con cota de
    error numérico O(ε_mach) invariante respecto a la dimensión del grafo.
    """
    u_flat = np.asarray(u, dtype=np.float64).reshape(-1)
    v_flat = np.asarray(v, dtype=np.float64).reshape(-1)
    if u_flat.size != v_flat.size:
        raise TellegenConservationError(
            "TellegenConservationError: Dimensiones incompatibles en producto Kahan."
        )

    total = 0.0
    compensation = 0.0
    for ui, vi in zip(u_flat, v_flat):
        prod = float(ui * vi)
        if not math.isfinite(prod):
            raise TellegenConservationError(
                "TellegenConservationError: Producto de potencia no finito."
            )
        y = prod - compensation
        t = total + y
        compensation = (t - total) - y
        total = t
    return float(total)


def _sparse_frobenius_norm(matrix: object, is_sparse_mat: bool) -> float:
    """Calcula la norma de Frobenius de manera segura para matrices densas o CSR."""
    if is_sparse_mat and _SPARSE_AVAILABLE and sp is not None:
        try:
            return float(sp.linalg.norm(matrix, ord="fro"))
        except Exception:
            data = getattr(matrix, "data", np.empty(0, dtype=np.float64))
            return float(la.norm(data)) if data.size > 0 else 0.0
    arr = np.asarray(matrix, dtype=np.float64)
    return 0.0 if arr.size == 0 else float(la.norm(arr, ord="fro"))


def verify_tellegen_sparse(
    nodal_potentials_p: NDArray[np.float64],
    flow_vector_f: NDArray[np.float64],
    incidence_matrix_B1: Union[NDArray[np.float64], csr_matrix],
) -> Tuple[float, bool]:
    """
    Verificación cohomológica de Tellegen vía operador coborde B₁ᵀ:
        ⟨B₁ᵀ p, f⟩_Kahan = ⟨p, B₁ f⟩ ≡ 0,   v = B₁ᵀ p
    """
    p = np.asarray(nodal_potentials_p, dtype=np.float64).reshape(-1)
    f = np.asarray(flow_vector_f, dtype=np.float64).reshape(-1)
    B1 = incidence_matrix_B1

    if B1.ndim != 2:
        raise TellegenConservationError(
            "TellegenConservationError: La matriz de incidencia debe ser 2D."
        )

    n_nodes, m_branches = B1.shape
    if p.size != n_nodes:
        raise TellegenConservationError(
            "TellegenConservationError: Dimensiones inconsistentes (potenciales vs B1)."
        )
    if f.size != m_branches:
        raise TellegenConservationError(
            "TellegenConservationError: Dimensiones inconsistentes (flujos vs B1)."
        )

    sparse_used = False
    if _SPARSE_AVAILABLE and isspmatrix is not None:
        if isspmatrix(B1):
            sparse_used = True
        elif np.count_nonzero(B1) < 0.3 * B1.size:
            B1 = csr_matrix(B1)
            sparse_used = True

    pressures_v = (
        np.asarray(B1.T.dot(p), dtype=np.float64).reshape(-1)
        if sparse_used
        else np.asarray(B1.T @ p, dtype=np.float64).reshape(-1)
    )
    power_sum = _kahan_inner_product(pressures_v, f)

    v_norm = float(la.norm(pressures_v))
    f_norm = float(la.norm(f))
    normalization = max(1.0, v_norm * f_norm)
    return abs(power_sum) / normalization, sparse_used


# ── SUTURA P1 — Sección de Poincaré, Acción-Ángulo y Resonancia KAM ─────────
def _continued_fraction_approximation(
    x: float,
    max_denominator: int = _POINCARE_MAX_DENOMINATOR,
) -> Tuple[int, int]:
    """
    Mejor aproximación diofántica racional p/q de x mediante el mapa de Gauss
    de fracciones continuas con denominador acotado q ≤ max_denominator.
    """
    if not math.isfinite(x) or x <= 0.0:
        return (0, 1)

    h_prev, h_curr = 0, 1
    k_prev, k_curr = 1, 0
    val = x

    for _ in range(32):
        a = math.floor(val)
        h_next = a * h_curr + h_prev
        k_next = a * k_curr + k_prev
        if k_next > max_denominator or k_next <= 0:
            break
        h_prev, h_curr = h_curr, h_next
        k_prev, k_curr = k_curr, k_next
        frac = val - a
        if frac < 1.0e-12:
            break
        val = 1.0 / frac

    return (int(h_curr), int(k_curr)) if k_curr > 0 else (0, 1)


def _detect_resonance(
    ratio: float,
    max_denominator: int = _POINCARE_MAX_DENOMINATOR,
    low_order: int = _POINCARE_LOW_ORDER,
    tolerance: float = _RESONANCE_TOLERANCE,
) -> Tuple[bool, int, int, float, float]:
    r"""
    Detección de resonancia de Arnold-Poincaré bajo criterio diofántico KAM:

    .. math::
        \left|\omega - \frac{p}{q}\right| < \max\left(\text{tol}, \frac{\gamma}{q^\tau}\right),
        \quad \mathcal{D}_{p,q} = \frac{1}{q(1 + q|\omega - p/q|)}

    Retorna (es_resonante, p, q, error_racional, divisor_pequeño).
    """
    if not math.isfinite(ratio) or ratio <= 0.0:
        return (False, 0, 0, 0.0, 0.0)

    p, q = _continued_fraction_approximation(ratio, max_denominator)
    if q <= 0:
        return (False, 0, 0, 0.0, 0.0)

    err = abs(ratio - p / q)
    kam_bound = _KAM_DIOPHANTINE_GAMMA / (float(q) ** _KAM_DIOPHANTINE_TAU)
    effective_tol = max(tolerance, kam_bound)
    is_resonant = (err < effective_tol) and (q <= low_order)
    small_div = 1.0 / (float(q) * (1.0 + float(q) * err))
    return (bool(is_resonant), int(p), int(q), float(err), float(small_div))


def _compute_action_angle(y: float, p_canonical: float) -> Tuple[float, float]:
    r"""
    Transformación simpléctica de Darboux al fibrado toral (I, θ):
        I = ½(y² + p²) = ¼((Ψ⁺)² + (Ψ⁻)²),   θ = atan2(p, y)
    """
    action = 0.5 * (y * y + p_canonical * p_canonical)
    angle = math.atan2(p_canonical, y) if action > _MACHINE_EPS else 0.0
    return (float(action), float(angle))


# ── SUTURA P2 — Serie Canónica de Lie-Deprit-Lindstedt-Poincaré ─────────────
def _lindstedt_poincare_expansion(
    action_zero_order: float,
    small_parameter: float,
    frequency_correction_proxy: float,
) -> LindstedtPoincareExpansion:
    r"""
    Expansión perturbativa canónica de Lie-Deprit-Lindstedt-Poincaré libre de
    resonancias seculares hasta orden O(ε⁴):

    .. math::
        \omega(\varepsilon) &= \omega_0 (1 + \varepsilon^2 \omega_1 + \varepsilon^4 \omega_2),
        \quad \omega_2 = -\frac{1}{4}\omega_1^2 \\
        I(\varepsilon) &= I_0 + \varepsilon^2 I_1 + \varepsilon^4 I_2,
        \quad I_1 = I_0 \omega_1, \quad I_2 = -\frac{1}{2} I_0 \omega_1 (1 + \frac{1}{2}\omega_1)
    """
    eps = float(max(_SMALL_PARAMETER_FLOOR, min(1.0, abs(small_parameter))))
    I0 = float(max(0.0, action_zero_order))
    omega1 = float(frequency_correction_proxy)
    omega2 = -0.25 * omega1 * omega1

    I1 = I0 * omega1
    I2 = -0.5 * I0 * omega1 * (1.0 + 0.5 * omega1)

    eps2 = eps * eps
    eps4 = eps2 * eps2

    corrected = float(I0 + eps2 * I1 + eps4 * I2)
    residual = float(abs(corrected - I0) / max(_MACHINE_EPS, I0))

    asymptotic_decay_ok = abs(eps4 * I2) <= abs(eps2 * I1) + _MACHINE_EPS
    is_consistent = (
        math.isfinite(corrected)
        and math.isfinite(residual)
        and asymptotic_decay_ok
        and residual <= _ADIABATIC_DRIFT_LIMIT
    )

    if not is_consistent or eps > 0.5:
        verdict = SpecularSovereignVerdict.DEGRADED
    else:
        verdict = SpecularSovereignVerdict.COHERENT

    return LindstedtPoincareExpansion(
        small_parameter=eps,
        zero_order_action=I0,
        first_order_action_correction=float(I1),
        second_order_action_correction=float(I2),
        frequency_shift_first_order=float(omega1),
        frequency_shift_second_order=float(omega2),
        corrected_action=corrected,
        expansion_residual=residual,
        is_asymptotically_consistent=bool(is_consistent),
        verdict=verdict,
    )


# ── SUTURA P3 — Operadores de Automejora Recursiva de Nivel 3 (RSI-3) ───────
def _richardson_extrapolation(
    r_prev: float,
    r_curr: float,
    tol_prev: float,
    tol_curr: float,
) -> Tuple[float, float]:
    """
    Extrapolación al límite de Richardson y orden local de convergencia de
    Fréchet-Hölder p = log(r_curr / r_prev) / log(tol_curr / tol_prev).
    """
    eps = 1.0e-300
    r_prev_p = max(eps, abs(r_prev))
    r_curr_p = max(eps, abs(r_curr))
    tol_prev_p = max(eps, abs(tol_prev))
    tol_curr_p = max(eps, abs(tol_curr))

    if tol_prev_p == tol_curr_p or r_prev_p == r_curr_p:
        return (0.0, float(r_curr))

    try:
        p = math.log(r_curr_p / r_prev_p) / math.log(tol_curr_p / tol_prev_p)
    except (ValueError, ZeroDivisionError):
        p = 1.0

    if not math.isfinite(p) or p <= 0.0:
        p = 1.0

    p_clamped = min(p, 8.0)
    try:
        q = (tol_prev_p / tol_curr_p) ** p_clamped
    except OverflowError:
        return (float(p), float(r_curr))

    if abs(q - 1.0) < eps:
        return (float(p), float(r_curr))

    R = (q * r_curr - r_prev) / (q - 1.0)
    return (float(p), float(R))


def _aitken_delta_squared(
    r_nm2: float,
    r_nm1: float,
    r_n: float,
) -> float:
    """Transformación no-lineal Δ² de Aitken (Padé [1/1]): r̂ = rₙ − (Δrₙ)² / Δ²rₙ."""
    denom = r_n - 2.0 * r_nm1 + r_nm2
    if abs(denom) < _RSI_OSCILLATION_EPS:
        return float(r_n)
    num = (r_n - r_nm1) ** 2
    return float(r_n - num / denom)


def _adaptive_refinement_factor(
    current_residual: float,
    target_residual: float,
    convergence_rate: float,
    level3_active: bool,
    aitken_signal: float,
    richardson_signal: Optional[float] = None,
) -> float:
    r"""
    Factor de contracción adaptativa α con modulación homotópica de Nivel 3.

    - Nivel 2 (Meta-Control): α⁽²⁾ = (target / rₙ)^{1/p}.
    - Nivel 3 (Meta-Meta-Control): Modula α⁽²⁾ según la razón espectral de
      Aitken y la discrepancia de Fréchet |r̂ₙ − Rₙ| entre el punto fijo de
      Aitken y el límite polinomial de Richardson:

      .. math::
          \alpha^{(3)} = \alpha^{(2)} \exp\!\left(-\gamma \tanh\!\left(\frac{|\hat{r}_n - R_n|}{\max(\varepsilon_{\text{mach}}, r_n)}\right)\right)
    """
    if current_residual <= 0.0:
        return _RSI_FACTOR_LOWER

    p = max(0.1, convergence_rate)
    try:
        factor = (target_residual / current_residual) ** (1.0 / p)
    except (ValueError, ZeroDivisionError, OverflowError):
        factor = 0.1

    factor = float(factor)
    if not math.isfinite(factor):
        factor = 0.1

    if level3_active and current_residual > 0.0:
        ratio = (
            abs(aitken_signal) / current_residual
            if math.isfinite(aitken_signal)
            else 1.0
        )
        if ratio > 1.0:
            factor = min(factor, 0.5)
        elif ratio < 0.5:
            factor = max(factor, 0.5)

        if richardson_signal is not None and math.isfinite(richardson_signal):
            frechet_discrepancy = abs(aitken_signal - richardson_signal) / max(
                _MACHINE_EPS, current_residual
            )
            homotopy_gain = math.exp(
                -_RSI_HOMOTOPY_GAIN * math.tanh(frechet_discrepancy)
            )
            factor = factor * homotopy_gain

    return float(max(_RSI_FACTOR_LOWER, min(_RSI_FACTOR_UPPER, factor)))


# ── Clase de la Fase Anidada 1: Phase1AllieviCharacteristicSolver ───────────
class Phase1AllieviCharacteristicSolver(ABC):
    """
    FASE ANIDADA 1 (Observe):
    Resuelve los invariantes de Riemann en el cono de luz hiperbólico de
    Allievi, proyecta el estado al fibrado acción-ángulo de Darboux, evalúa
    la sección transversal de Poincaré Σ, la resonancia diofántica KAM y la
    serie perturbativa de Lie-Deprit-Lindstedt-Poincaré.

    Último método: `_phase1_to_phase2_bridge(...)`
    """

    def observe_allievi_characteristics(
        self,
        potential_y: float,
        flow_q: float,
        wave_speed_a: float,
        gravity_g: float,
        section_s: float,
        tolerance: float = _DEFAULT_TOL,
        cfl_number: Optional[float] = None,
        poincare_section_level: float = 0.0,
    ) -> Phase1AllieviCertificate:
        r"""
        FASE 1 — Observe.

        Calcula los invariantes hiperbólicos de Riemann:
            Z_c = a / (g S),   Ψ± = y ± Z_c q,
        las variables simplécticas acción-ángulo:
            I = ½(y² + (Z_c q)²),   θ = atan2(Z_c q, y),
        el snapshot de la sección transversal Σ = {y = y_s}, la detección de
        resonancia KAM y la serie de Lie-Deprit-Lindstedt-Poincaré.
        """
        # ─── 1. Auditoría algebraica de entradas ───
        raw_inputs = (potential_y, flow_q, wave_speed_a, gravity_g, section_s)
        if not all(np.isfinite(v) for v in raw_inputs):
            raise AllieviCharacteristicError(
                "AllieviCharacteristicError: Entradas no finitas."
            )
        if wave_speed_a <= _MACHINE_EPS:
            raise AllieviCharacteristicError(
                "AllieviCharacteristicError: Velocidad de onda degenerada."
            )
        if gravity_g <= _MACHINE_EPS:
            raise AllieviCharacteristicError(
                "AllieviCharacteristicError: Acoplamiento g degenerado."
            )
        if section_s <= _MACHINE_EPS:
            raise AllieviCharacteristicError(
                "AllieviCharacteristicError: Sección S degenerada."
            )

        denominator = gravity_g * section_s
        if denominator <= _MACHINE_EPS:
            raise AllieviCharacteristicError(
                "AllieviCharacteristicError: Denominador g·S degenerado."
            )

        z_c = wave_speed_a / denominator
        if not np.isfinite(z_c) or z_c <= _MACHINE_EPS:
            raise AllieviCharacteristicError(
                "AllieviCharacteristicError: Impedancia característica degenerada."
            )

        # ─── 2. Invariantes de Riemann en el cono de luz ───
        psi_plus = potential_y + z_c * flow_q
        psi_minus = potential_y - z_c * flow_q
        if not np.isfinite(psi_plus) or not np.isfinite(psi_minus):
            raise AllieviCharacteristicError(
                "AllieviCharacteristicError: Invariantes no finitos."
            )

        is_stable = (
            abs(psi_plus) <= _STABILITY_BOUND
            and abs(psi_minus) <= _STABILITY_BOUND
        )

        # ─── 3. Vector tangente y 1-forma covectora característica ───
        char_dir = np.array([wave_speed_a, 1.0], dtype=np.float64)
        char_dir_norm = float(la.norm(char_dir))
        if char_dir_norm <= _MACHINE_EPS:
            raise AllieviCharacteristicError(
                "AllieviCharacteristicError: Dirección característica degenerada."
            )
        char_dir = char_dir / char_dir_norm

        g_cal = np.array([[1.0, 0.0], [0.0, z_c * z_c]], dtype=np.float64)
        char_cov = g_cal @ char_dir
        char_cov_norm = float(la.norm(char_cov))
        char_cov = (
            char_cov / char_cov_norm
            if char_cov_norm > _MACHINE_EPS
            else char_dir.copy()
        )

        # ─── 4. Norma Hamiltoniana y condicionamiento espectral ───
        energy_norm = float(np.hypot(potential_y, z_c * flow_q))
        impedance_cond = float(max(z_c, 1.0 / z_c))

        # ─── 5. Invariante de Courant-Friedrichs-Lewy (CFL) ───
        if cfl_number is None:
            cfl_proxy = float(abs(wave_speed_a) / (abs(wave_speed_a) + 1.0))
        else:
            if not np.isfinite(cfl_number):
                raise AllieviCharacteristicError(
                    "AllieviCharacteristicError: CFL no finito."
                )
            cfl_proxy = float(abs(cfl_number))

        # ─── 6. Poincaré: coordenadas canónicas acción-ángulo ───
        p_canonical = z_c * flow_q
        action, angle = _compute_action_angle(potential_y, p_canonical)

        # ─── 7. Intersección con la sección transversal Σ ───
        snapshot = self._poincare_section_snapshot(
            action=action,
            angle=angle,
            section_level=poincare_section_level,
            psi_plus=psi_plus,
            psi_minus=psi_minus,
        )

        # ─── 8. Resonancia diofántica KAM ───
        if abs(psi_minus) > _MACHINE_EPS:
            rho = abs(psi_plus / psi_minus)
        else:
            rho = float("inf")
        is_res, p_res, q_res, _, small_div = _detect_resonance(rho)

        # ─── 9. Serie canónica de Lie-Deprit-Lindstedt-Poincaré ───
        tol_scale = min(1.0, max(_SMALL_PARAMETER_FLOOR, math.sqrt(abs(tolerance))))
        eps_eff = 0.0
        if impedance_cond > 0.0 and math.isfinite(impedance_cond):
            raw_anisotropy = math.log10(max(1.0, impedance_cond)) / 8.0
            eps_eff = min(
                1.0,
                raw_anisotropy * tol_scale if raw_anisotropy < 0.5 else raw_anisotropy,
            )
        omega1_proxy = -0.5 * eps_eff
        lindstedt = _lindstedt_poincare_expansion(
            action_zero_order=action,
            small_parameter=eps_eff,
            frequency_correction_proxy=omega1_proxy,
        )

        # ─── 10. Clasificación en el álgebra de Heyting Ω₃ ───
        verdict = SpecularSovereignVerdict.COHERENT
        if (
            cfl_proxy > 1.0 + tolerance
            or not is_stable
            or snapshot.verdict == SpecularSovereignVerdict.VETOED
        ):
            verdict = SpecularSovereignVerdict.VETOED
        elif (
            impedance_cond > _IMPEDANCE_CONDITION_LIMIT
            or abs(psi_plus - psi_minus) > 1.0e6
            or is_res
            or snapshot.verdict != SpecularSovereignVerdict.COHERENT
            or lindstedt.verdict != SpecularSovereignVerdict.COHERENT
        ):
            verdict = SpecularSovereignVerdict.DEGRADED

        return Phase1AllieviCertificate(
            is_characteristic_stable=is_stable,
            characteristic_impedance=float(z_c),
            riemann_invariants=(float(psi_plus), float(psi_minus)),
            characteristic_direction=(float(char_dir[0]), float(char_dir[1])),
            characteristic_covector=(float(char_cov[0]), float(char_cov[1])),
            energy_norm=energy_norm,
            impedance_condition_number=impedance_cond,
            cfl_proxy=cfl_proxy,
            verdict=verdict,
            action_invariant=action,
            angle_parameter=angle,
            small_parameter_epsilon=eps_eff,
            lindstedt_frequency_shift=float(lindstedt.frequency_shift_first_order),
            corrected_action=float(lindstedt.corrected_action),
            expansion_residual=float(lindstedt.expansion_residual),
            reflection_coefficient_magnitude=float(rho) if math.isfinite(rho) else 0.0,
            is_near_resonance=bool(is_res),
            resonance_p=int(p_res),
            resonance_q=int(q_res),
            small_divisor_magnitude=float(small_div),
            poincare_section_level=float(poincare_section_level),
            poincare_crossing_count=len(snapshot.crossing_values),
        )

    # Alias de compatibilidad retroactiva
    solve_allievi_characteristics = observe_allievi_characteristics

    def _poincare_section_snapshot(
        self,
        action: float,
        angle: float,
        section_level: float,
        psi_plus: float,
        psi_minus: float,
    ) -> PoincareSectionSnapshot:
        """
        Interseca la variedad de nivel de acción I = ½(y² + p²) con la
        hipersuperficie transversal Σ = {y = y_s}. Las ramas de momento
        canónico en Σ vienen dadas por p_± = ±√(2I − y_s²).
        """
        I = float(max(0.0, action))
        y_s = float(section_level)
        radius_sq = 2.0 * I

        if radius_sq > _MACHINE_EPS and abs(y_s) < math.sqrt(radius_sq):
            p_mag = math.sqrt(max(0.0, radius_sq - y_s * y_s))
            crossings = (p_mag, -p_mag)
            slopes = (-1.0, 1.0)
            is_periodic = True
            period = 1.0
        elif (
            radius_sq > _MACHINE_EPS
            and abs(abs(y_s) - math.sqrt(radius_sq)) <= _MACHINE_EPS
        ):
            crossings = (0.0,)
            slopes = (0.0,)
            is_periodic = True
            period = 1.0
        else:
            crossings = ()
            slopes = ()
            is_periodic = False
            period = float("inf")

        return_map_eigs = (1.0, 1.0) if crossings else ()
        contraction = 1.0
        freq_ratio = (
            abs(psi_plus / psi_minus) if abs(psi_minus) > _MACHINE_EPS else 0.0
        )
        is_res, p_res, q_res, _, small_div = _detect_resonance(freq_ratio)

        if not math.isfinite(action) or action < 0.0:
            verdict = SpecularSovereignVerdict.VETOED
        elif not crossings and radius_sq > _MACHINE_EPS:
            verdict = SpecularSovereignVerdict.DEGRADED
        elif is_res:
            verdict = SpecularSovereignVerdict.DEGRADED
        else:
            verdict = SpecularSovereignVerdict.COHERENT

        return PoincareSectionSnapshot(
            section_level=y_s,
            crossing_values=tuple(float(c) for c in crossings),
            crossing_slopes=tuple(float(s) for s in slopes),
            return_map_contraction=contraction,
            return_map_eigenvalues_modulus=tuple(float(e) for e in return_map_eigs),
            is_periodic=is_periodic,
            period_estimate=float(period),
            action_invariant=I,
            angle_parameter=float(angle),
            frequency_ratio=float(freq_ratio),
            near_resonance_p=int(p_res),
            near_resonance_q=int(q_res),
            small_divisor_magnitude=float(small_div),
            verdict=verdict,
        )

    def _lindstedt_poincare_correction(
        self,
        phase1_cert: Phase1AllieviCertificate,
    ) -> LindstedtPoincareExpansion:
        """Evalúa la expansión de Lie-Deprit-Lindstedt-Poincaré sobre un certificado."""
        eps = float(phase1_cert.small_parameter_epsilon)
        omega1 = (
            float(phase1_cert.lindstedt_frequency_shift) if eps > 0.0 else 0.0
        )
        if eps > 0.0 and omega1 == 0.0:
            omega1 = -0.5 * eps
        return _lindstedt_poincare_expansion(
            action_zero_order=phase1_cert.action_invariant,
            small_parameter=eps,
            frequency_correction_proxy=omega1,
        )

    # ═════════════════════════════════════════════════════════════════════
    # ESLABÓN FUNTORIAL 1 → 2: Último método de Fase 1 e inicio de Fase 2
    # ═════════════════════════════════════════════════════════════════════
    def _phase1_to_phase2_bridge(
        self,
        phase1_cert: Phase1AllieviCertificate,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        float,
        NDArray[np.float64],
        PoincareSectionSnapshot,
        LindstedtPoincareExpansion,
    ]:
        r"""
        MORFISMO DE TRANSICIÓN (FASE ANIDADA 1 ──► FASE ANIDADA 2).

        Transporta paralelamente el vector característico normalizado, el
        covector dual, la impedancia Z_c, el tensor métrico canónico
        G₀ = diag(1, Z_c²), el snapshot de Poincaré Σ y la expansión de
        Lindstedt-Poincaré hacia el reflector covariante de Householder.
        """
        char_dir = np.array(
            phase1_cert.characteristic_direction, dtype=np.float64
        )
        char_cov = np.array(
            phase1_cert.characteristic_covector, dtype=np.float64
        )
        z_c = float(phase1_cert.characteristic_impedance)

        if not np.all(np.isfinite(char_dir)) or not np.all(np.isfinite(char_cov)):
            raise AllieviCharacteristicError(
                "AllieviCharacteristicError: Certificado de Fase 1 no finito."
            )

        norm = float(la.norm(char_dir))
        if norm <= _MACHINE_EPS:
            raise AllieviCharacteristicError(
                "AllieviCharacteristicError: Dirección característica degenerada."
            )
        char_dir = char_dir / norm

        g_cal = np.array([[1.0, 0.0], [0.0, z_c * z_c]], dtype=np.float64)

        snapshot = self._poincare_section_snapshot(
            action=phase1_cert.action_invariant,
            angle=phase1_cert.angle_parameter,
            section_level=phase1_cert.poincare_section_level,
            psi_plus=phase1_cert.riemann_invariants[0],
            psi_minus=phase1_cert.riemann_invariants[1],
        )
        lindstedt = self._lindstedt_poincare_correction(phase1_cert)

        return char_dir, char_cov, z_c, g_cal, snapshot, lindstedt


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                        FASE ANIDADA 2: ORIENT                             ║
# ║  Continuación directa de `_phase1_to_phase2_bridge`                       ║
# ║  Reflexión Covariante de Householder + Newton-Schulz + Floquet-Poincaré   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class Phase2SpecularHouseholderReflector(Phase1AllieviCharacteristicSolver):
    """
    FASE ANIDADA 2 (Orient):
    Hereda de `Phase1AllieviCharacteristicSolver` y consume directamente el
    morfismo `_phase1_to_phase2_bridge` para construir el operador involutivo
    G-ortogonal de Householder con regularización de Higham, purificación de
    Newton-Schulz y auditoría simpléctica de monodromía de Poincaré.

    Último método: `_phase2_to_phase3_bridge(...)`
    """

    def _synthesize_covariant_householder(
        self,
        metric_tensor_g: Optional[NDArray[np.float64]],
        boundary_normal_n: Optional[NDArray[np.float64]],
        state_vector: Optional[NDArray[np.float64]],
        char_dir: NDArray[np.float64],
        g_cal: NDArray[np.float64],
        tolerance: float,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        bool,
        float,
    ]:
        r"""
        Sintetiza de forma determinista la métrica proyectada por Higham G,
        el operador G-ortogonal purificado por Newton-Schulz M y el estado
        reflejado Mx:

        .. math::
            M_0 = I - 2 \frac{n (G n)^\top}{n^\top G n}, \quad
            M_1 = \frac{1}{2} M_0 \left(3I - G^{-1} M_0^\top G M_0\right)
        """
        dim = char_dir.size
        if dim != 2:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Fase 2D requerida."
            )
        audit_tol = max(tolerance * 100.0, _RELAX_TOL)

        # 1. Métrica Riemanniana G y proyección de Higham al cono SPD
        if metric_tensor_g is None:
            g_metric = g_cal.copy()
            higham_applied = False
            L_chol, _ = stable_cholesky_higham(g_metric, tolerance)
        else:
            g_metric = np.asarray(metric_tensor_g, dtype=np.float64)
            if g_metric.ndim != 2 or g_metric.shape != (dim, dim):
                raise HouseholderReflectionError(
                    "HouseholderReflectionError: G debe ser 2x2."
                )
            if not np.all(np.isfinite(g_metric)):
                raise HouseholderReflectionError(
                    "HouseholderReflectionError: G no finita."
                )
            sym_res = float(la.norm(g_metric - g_metric.T, ord="fro"))
            if sym_res > audit_tol:
                raise HouseholderReflectionError(
                    f"HouseholderReflectionError: G no simétrica (residuo={sym_res:.4e})."
                )
            L_chol, higham_applied = stable_cholesky_higham(g_metric, tolerance)
            g_metric = L_chol @ L_chol.T

        try:
            metric_cond = float(np.linalg.cond(g_metric))
        except Exception as exc:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: κ(G) no calculable."
            ) from exc
        if not np.isfinite(metric_cond) or metric_cond <= 0.0:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: κ(G) inválido."
            )

        # 2. Vector normal a la hipersuperficie de frontera
        normal = (
            char_dir.copy()
            if boundary_normal_n is None
            else np.asarray(boundary_normal_n, dtype=np.float64).reshape(-1)
        )
        if normal.size != dim:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Normal dimensión ≠ 2."
            )
        if not np.all(np.isfinite(normal)):
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Normal no finita."
            )
        if float(la.norm(normal)) <= _MACHINE_EPS:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Normal euclidiana degenerada."
            )

        # 3. Vector de estado en el espacio de fases
        x_state = (
            np.zeros(dim, dtype=np.float64)
            if state_vector is None
            else np.asarray(state_vector, dtype=np.float64).reshape(-1)
        )
        if x_state.size != dim:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Estado dimensión ≠ 2."
            )
        if not np.all(np.isfinite(x_state)):
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Estado no finito."
            )

        # 4. Construcción del reflector covariante y purificación de Newton-Schulz
        g_normal = g_metric @ normal
        denom = float(normal @ g_normal)
        g_norm = float(la.norm(g_metric, ord="fro"))
        if denom <= _MACHINE_EPS * max(1.0, g_norm):
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Normal ortogonal degenerada."
            )

        M = np.eye(dim, dtype=np.float64) - (2.0 / denom) * np.outer(
            normal, g_normal
        )
        if not np.all(np.isfinite(M)):
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Operador no finito."
            )

        if tolerance < _DEFAULT_TOL:
            try:
                gram_op = la.cho_solve((L_chol, True), M.T @ (g_metric @ M))
                M_purified = 0.5 * (
                    M @ (3.0 * np.eye(dim, dtype=np.float64) - gram_op)
                )
                if np.all(np.isfinite(M_purified)):
                    M = M_purified
            except Exception:
                pass

        reflected = M @ x_state
        return (
            g_metric,
            normal,
            x_state,
            M,
            reflected,
            higham_applied,
            metric_cond,
        )

    def orient_householder_specular_reflection(
        self,
        metric_tensor_g: Optional[NDArray[np.float64]],
        boundary_normal_n: Optional[NDArray[np.float64]],
        state_vector: Optional[NDArray[np.float64]],
        phase1_cert: Phase1AllieviCertificate,
        tolerance: float = _DEFAULT_TOL,
    ) -> Phase2HouseholderCertificate:
        r"""
        FASE 2 — Orient.

        Aplica la reflexión covariante de Householder:
            M = I − 2 n (G n)ᵀ / (nᵀ G n)
        y audita:
          • Proyección de Higham al cono SPD.
          • Unitariedad Riemanniana Mᵀ G M = G.
          • Anti-simplecticidad en 2D: Mᵀ Ω M = −Ω y det(M) = −1.
          • Ley especular de Snell-Descartes sobre la característica.
          • Invariancia adiabática de la acción I (Poincaré).
          • Monodromía de Floquet P = M² de la aplicación de retorno.
        """
        # ─── 0. Recepción del morfismo terminal de Fase 1 ───
        char_dir, _, _, g_cal, poincare_in, lindstedt_in = (
            self._phase1_to_phase2_bridge(phase1_cert)
        )
        audit_tol = max(tolerance * 100.0, _RELAX_TOL)

        # ─── 1–4. Síntesis de G, normal, estado y operador purificado M ───
        (
            g_metric,
            normal,
            x_state,
            M,
            reflected,
            higham_applied,
            metric_cond,
        ) = self._synthesize_covariant_householder(
            metric_tensor_g=metric_tensor_g,
            boundary_normal_n=boundary_normal_n,
            state_vector=state_vector,
            char_dir=char_dir,
            g_cal=g_cal,
            tolerance=tolerance,
        )
        g_norm = float(la.norm(g_metric, ord="fro"))
        denom = float(normal @ (g_metric @ normal))

        # ─── 5. Unitariedad Riemanniana ───
        unitary_res = float(
            la.norm(M.T @ g_metric @ M - g_metric, ord="fro") / max(1.0, g_norm)
        )
        is_unitary = unitary_res <= audit_tol

        # ─── 6. Preservación de norma energética G ───
        x_norm_sq = float(x_state @ g_metric @ x_state)
        mx_norm_sq = float(reflected @ g_metric @ reflected)
        x_norm = float(np.sqrt(max(0.0, x_norm_sq)))
        mx_norm = float(np.sqrt(max(0.0, mx_norm_sq)))
        state_norm_res = float(abs(mx_norm - x_norm) / max(1.0, x_norm))

        # ─── 7. Ley especular sobre la curva característica ───
        char_g_sq = float(char_dir @ g_metric @ char_dir)
        if char_g_sq <= _MACHINE_EPS:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Norma G de la característica nula."
            )
        char_g_norm = float(np.sqrt(max(0.0, char_g_sq)))
        normal_g_norm = float(np.sqrt(max(0.0, denom)))

        inc_num = float(normal @ g_metric @ char_dir)
        inc_den = char_g_norm * normal_g_norm
        if inc_den <= _MACHINE_EPS:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Denominador de incidencia nulo."
            )
        incidence_cos = inc_num / inc_den

        reflected_char = M @ char_dir
        expected = char_dir - (2.0 * inc_num / denom) * normal
        specular_res = float(
            la.norm(reflected_char - expected)
            / max(1.0, float(la.norm(char_dir)))
        )

        # ─── 8. Anti-simplecticidad canónica 2D ───
        omega = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=np.float64)
        sympl_res = float(
            la.norm(M.T @ omega @ M + omega, ord="fro")
            / max(1.0, float(la.norm(omega, ord="fro")))
        )
        try:
            det_m = float(np.linalg.det(M))
        except Exception:
            det_m = 0.0
        is_anti_sympl = sympl_res <= audit_tol and det_m < 0.0

        # ─── 9. Poincaré: deriva adiabática de la acción ───
        I_in = float(phase1_cert.action_invariant)
        I_out = 0.5 * float(reflected @ g_metric @ reflected)
        action_drift = abs(I_out - I_in) / max(1.0, I_in)

        # ─── 10. Monodromía de Floquet de la aplicación de retorno ───
        try:
            eigs_M = la.eigvals(M)
            eig_mods = tuple(float(abs(e)) for e in eigs_M)
        except Exception:
            eig_mods = ()

        monodromy_defect = float(
            la.norm((M @ M) - np.eye(2, dtype=np.float64), ord="fro")
        )
        return_contraction = float(1.0 + monodromy_defect)
        is_return_stable = (
            all(m <= 1.0 + audit_tol for m in eig_mods) if eig_mods else True
        )

        # ─── 11. Defecto de Padé-Richardson acoplado a Lindstedt ───
        dominant = max(unitary_res, specular_res, state_norm_res)
        richardson_corr = float(
            dominant / (1.0 + max(0.0, float(lindstedt_in.expansion_residual)))
        )

        # ─── 12. Clasificación en el álgebra de Heyting Ω₃ ───
        verdict = SpecularSovereignVerdict.COHERENT
        if (
            not np.isfinite(unitary_res)
            or not np.isfinite(specular_res)
            or not np.isfinite(state_norm_res)
            or not is_unitary
            or not is_anti_sympl
            or not is_return_stable
        ):
            verdict = SpecularSovereignVerdict.VETOED
        elif (
            unitary_res > tolerance
            or specular_res > tolerance
            or state_norm_res > tolerance
            or metric_cond > _IMPEDANCE_CONDITION_LIMIT
            or abs(incidence_cos) < 0.1
            or higham_applied
            or action_drift > _ADIABATIC_DRIFT_LIMIT
            or poincare_in.verdict == SpecularSovereignVerdict.DEGRADED
        ):
            verdict = SpecularSovereignVerdict.DEGRADED

        verdict = SpecularSovereignVerdict.supremum(verdict, phase1_cert.verdict)

        return Phase2HouseholderCertificate(
            is_reflection_unitary=is_unitary,
            reflection_residual=unitary_res,
            is_anti_symplectic=is_anti_sympl,
            is_characteristic_consistent=(specular_res <= audit_tol),
            specular_residual=specular_res,
            incidence_cosine=float(incidence_cos),
            state_norm_residual=state_norm_res,
            metric_condition_number=metric_cond,
            higham_regularization_applied=higham_applied,
            verdict=verdict,
            action_invariant_input=I_in,
            action_invariant_reflected=I_out,
            action_drift=float(action_drift),
            return_map_contraction=return_contraction,
            return_map_eigenvalues_modulus=eig_mods,
            is_return_map_stable=bool(is_return_stable),
            richardson_correction=float(richardson_corr),
        )

    # Alias de compatibilidad retroactiva
    reflect_householder_boundary = orient_householder_specular_reflection

    def _poincare_return_map_audit(
        self,
        phase2_cert: Phase2HouseholderCertificate,
        operator_M: NDArray[np.float64],
        state_x: NDArray[np.float64],
        n_iterations: int = 4,
    ) -> Tuple[float, Tuple[float, ...]]:
        """
        Audita la órbita discreta {M^k x} generada por la aplicación de retorno.
        Para una involución exacta (M² = I), la órbita describe un 2-ciclo
        conservativo; la contracción cuantifica la deriva espectral de Floquet.
        """
        M = np.asarray(operator_M, dtype=np.float64)
        x = np.asarray(state_x, dtype=np.float64).reshape(-1)
        if M.shape[0] != x.size:
            raise PoincareSectionError(
                "PoincareSectionError: dimensiones incompatibles en retorno."
            )

        norms = []
        current = x.copy()
        for _ in range(n_iterations):
            current = M @ current
            norms.append(float(la.norm(current)))

        if len(norms) < 2:
            return (1.0, tuple(norms))
        base = max(norms[0], _MACHINE_EPS)
        return (float(max(n / base for n in norms)), tuple(norms))

    # ═════════════════════════════════════════════════════════════════════
    # ESLABÓN FUNTORIAL 2 → 3: Último método de Fase 2 e inicio de Fase 3
    # ═════════════════════════════════════════════════════════════════════
    def _phase2_to_phase3_bridge(
        self,
        phase2_cert: Phase2HouseholderCertificate,
        operator_M: NDArray[np.float64],
        state_x: NDArray[np.float64],
        poincare_input: PoincareSectionSnapshot,
        lindstedt_input: LindstedtPoincareExpansion,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        SpecularSovereignVerdict,
        PoincareSectionSnapshot,
        LindstedtPoincareExpansion,
    ]:
        """
        MORFISMO DE TRANSICIÓN (FASE ANIDADA 2 ──► FASE ANIDADA 3).

        Transfiere el operador involutivo purificado M, el estado reflejado x,
        el veredicto en el retículo Ω₃, el snapshot de Poincaré y la expansión
        de Lindstedt-Poincaré hacia el validador cohomológico de Tellegen-Hodge.
        """
        op = np.asarray(operator_M, dtype=np.float64)
        x = np.asarray(state_x, dtype=np.float64)
        if not np.all(np.isfinite(op)) or not np.all(np.isfinite(x)):
            raise HouseholderReflectionError(
                "HouseholderReflectionError: puente con valores no finitos."
            )
        return op, x, phase2_cert.verdict, poincare_input, lindstedt_input


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                 FASE ANIDADA 3: DECIDE, ACT & META-EVOLVE                 ║
# ║  Continuación directa de `_phase2_to_phase3_bridge`                       ║
# ║  Cohomología de Tellegen-Hodge-Poincaré + Meta-Controlador RSI-3          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

class Phase3TellegenPassivityValidator(Phase2SpecularHouseholderReflector):
    """
    FASE ANIDADA 3 (Decide & Meta-Evolve):
    Hereda de `Phase2SpecularHouseholderReflector`, consume el morfismo
    `_phase2_to_phase3_bridge` y audita la conservación de Tellegen-Kahan,
    la pasividad de Lyapunov, la descomposición ortogonal de Helmholtz-Hodge-Poincaré,
    la recurrencia de Poincaré y el lazo cerrado del meta-controlador RSI-3.

    Último método de validación: `_phase3_to_meta_bridge(...)`
    """

    def decide_tellegen_lyapunov_conservation(
        self,
        pressures: NDArray[np.float64],
        flows: NDArray[np.float64],
        rayleigh_dissipation: float,
        phase1_cert: Phase1AllieviCertificate,
        phase2_cert: Phase2HouseholderCertificate,
        incidence_matrix: Optional[Union[NDArray[np.float64], csr_matrix]] = None,
        nodal_potentials: Optional[NDArray[np.float64]] = None,
        operator_M: Optional[NDArray[np.float64]] = None,
        state_x: Optional[NDArray[np.float64]] = None,
        tolerance: float = _DEFAULT_TOL,
    ) -> Phase3TellegenCertificate:
        r"""
        FASE 3 — Decide.

        Audita sobre el complejo celular 1-dimensional K = (V, E):
          • KCL (Ciclos Z₁ = ker B₁): B₁ q ≈ 0.
          • KVL (Cortes B¹ = im B₁ᵀ): p ∈ im(B₁ᵀ).
          • Tellegen: ⟨p, q⟩_Kahan ≈ 0.
          • Pasividad Port-Hamiltoniana de Lyapunov: P_diss ≥ 0.
          • Dualidad de Hodge-Poincaré: dim Z₁ + dim B¹ = |E|, χ(K) = β₀ − β₁
            y ortogonalidad de Helmholtz-Hodge ⟨P_cut p, P_cycle p⟩_Kahan ≈ 0.
          • Recurrencia de Poincaré: ‖M² x − x‖ / ‖x‖.
        """
        p = np.asarray(pressures, dtype=np.float64).reshape(-1)
        q = np.asarray(flows, dtype=np.float64).reshape(-1)
        m = p.size

        if q.size != m:
            raise TellegenConservationError(
                "TellegenConservationError: dimensiones presiones/flujos inconsistentes."
            )

        if m == 0:
            return Phase3TellegenCertificate(
                tellegen_residual=0.0,
                kcl_residual=0.0,
                kvl_residual=0.0,
                is_tellegen_conserved=True,
                is_kcl_conserved=True,
                is_kvl_conserved=True,
                is_lyapunov_passive=True,
                graph_order=0,
                graph_size=0,
                cyclomatic_number=0,
                sparse_computation=False,
                verdict=SpecularSovereignVerdict.COHERENT,
            )

        if m > _GRAPH_MAX_BRANCHES:
            raise TellegenConservationError(
                "TellegenConservationError: complejidad fuera de cota."
            )

        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
            raise TellegenConservationError(
                "TellegenConservationError: presiones o flujos no finitos."
            )
        if not np.isfinite(rayleigh_dissipation):
            raise TellegenConservationError(
                "TellegenConservationError: disipación de Rayleigh no finita."
            )

        # ─── 0. Enlace formal con el puente de Fase 2 (si M y x están presentes) ───
        op_M_bridged: Optional[NDArray[np.float64]] = None
        state_x_bridged: Optional[NDArray[np.float64]] = None
        if (
            operator_M is not None
            and state_x is not None
            and np.asarray(operator_M).shape[0] == np.asarray(state_x).size
        ):
            _, _, _, _, snap_in, lind_in = self._phase1_to_phase2_bridge(phase1_cert)
            op_M_bridged, state_x_bridged, _, _, _ = self._phase2_to_phase3_bridge(
                phase2_cert=phase2_cert,
                operator_M=np.asarray(operator_M, dtype=np.float64),
                state_x=np.asarray(state_x, dtype=np.float64).reshape(-1),
                poincare_input=snap_in,
                lindstedt_input=lind_in,
            )

        # ─── 1. Operador frontera B₁ : C₁(K) → C₀(K) ───
        sparse_computation = False
        if incidence_matrix is None:
            a_matrix = np.zeros((2, m), dtype=np.float64)
            a_matrix[0, :] = 1.0
            a_matrix[1, :] = -1.0
        else:
            a_matrix = incidence_matrix
            if a_matrix.ndim != 2:
                raise TellegenConservationError(
                    "TellegenConservationError: A debe ser 2D."
                )
            if a_matrix.shape[1] != m:
                raise TellegenConservationError(
                    "TellegenConservationError: A no concuerda con m."
                )
            if _SPARSE_AVAILABLE and isspmatrix is not None and isspmatrix(a_matrix):
                sparse_computation = True
                if not np.all(np.isfinite(a_matrix.data)):
                    raise TellegenConservationError(
                        "TellegenConservationError: A dispersa no finita."
                    )
            else:
                if not np.all(np.isfinite(a_matrix)):
                    raise TellegenConservationError(
                        "TellegenConservationError: A no finita."
                    )

        n_nodes = a_matrix.shape[0]
        if n_nodes == 0:
            raise TellegenConservationError(
                "TellegenConservationError: topología sin nodos."
            )

        # ─── 2. Normas en espacios l₂ y Frobenius ───
        p_norm = float(la.norm(p))
        q_norm = float(la.norm(q))
        a_norm = _sparse_frobenius_norm(a_matrix, sparse_computation)

        # ─── 3. KCL (1-ciclos de Kirchhoff) ───
        kcl_vec = a_matrix.dot(q) if sparse_computation else a_matrix @ q
        kcl_res = float(
            la.norm(kcl_vec)
            / (max(1.0, a_norm) * max(1.0, q_norm) * math.sqrt(max(1, n_nodes)))
        )

        # ─── 4. KVL y Proyección de Helmholtz-Hodge sobre im(B₁ᵀ) ───
        recon_cut: Optional[NDArray[np.float64]] = None
        try:
            if sparse_computation:
                phi = sparse_lsqr(a_matrix.T, p, atol=tolerance, btol=tolerance)[0]
                recon_cut = np.asarray(
                    a_matrix.T.dot(phi), dtype=np.float64
                ).reshape(-1)
            else:
                phi, *_ = np.linalg.lstsq(a_matrix.T, p, rcond=None)
                recon_cut = np.asarray(
                    a_matrix.T @ phi, dtype=np.float64
                ).reshape(-1)
            kvl_res = float(la.norm(recon_cut - p) / max(1.0, p_norm))
        except Exception:
            kvl_res = float("inf")

        # ─── 5. Ortogonalidad de Tellegen con suma compensada de Kahan ───
        if nodal_potentials is not None and incidence_matrix is not None:
            try:
                tellegen_res, sparse_used = verify_tellegen_sparse(
                    nodal_potentials, q, incidence_matrix
                )
                sparse_computation = sparse_computation or sparse_used
            except TellegenConservationError:
                tellegen_power = _kahan_inner_product(p, q)
                tellegen_res = abs(tellegen_power) / max(1.0, p_norm * q_norm)
        else:
            tellegen_power = _kahan_inner_product(p, q)
            tellegen_res = abs(tellegen_power) / max(1.0, p_norm * q_norm)

        # ─── 6. Dualidad Cohomológica de Hodge-Poincaré ───
        try:
            if sparse_computation and _SPARSE_AVAILABLE:
                dense = a_matrix.toarray() if isspmatrix(a_matrix) else a_matrix
                rank_a = int(np.linalg.matrix_rank(dense, tol=1.0e-12))
            else:
                rank_a = int(np.linalg.matrix_rank(a_matrix, tol=1.0e-12))
        except Exception:
            rank_a = min(n_nodes, m)

        cycle_dim = max(0, m - rank_a)
        cut_dim = rank_a
        betti_0 = max(0, n_nodes - rank_a)
        betti_1 = cycle_dim
        euler_char = int(n_nodes - m)
        euler_poincare_defect = abs(euler_char - (betti_0 - betti_1))

        hodge_orthogonality_defect = 0.0
        if recon_cut is not None and recon_cut.size == m and p_norm > _MACHINE_EPS:
            cycle_component = p - recon_cut
            inner_hodge = abs(_kahan_inner_product(recon_cut, cycle_component))
            hodge_orthogonality_defect = float(
                inner_hodge / max(1.0, p_norm * p_norm)
            )

        duality_res = float(
            (abs((cycle_dim + cut_dim) - m) + euler_poincare_defect) / max(1, m)
            + hodge_orthogonality_defect
        )
        is_poincare_dual = duality_res <= max(tolerance * 100.0, _RELAX_TOL)

        # ─── 7. Pasividad de Lyapunov ───
        is_lyap_passive = rayleigh_dissipation >= -_MACHINE_EPS

        # ─── 8. Teorema de Recurrencia de Poincaré ───
        if op_M_bridged is not None and state_x_bridged is not None:
            M2x = op_M_bridged @ (op_M_bridged @ state_x_bridged)
            recurrence_res = float(
                la.norm(M2x - state_x_bridged)
                / max(1.0, float(la.norm(state_x_bridged)))
            )
        else:
            recurrence_res = 0.0

        # ─── 9. Deriva adiabática de la acción I ───
        I_in = float(phase1_cert.action_invariant)
        I_out = float(phase2_cert.action_invariant_reflected)
        adiabatic_drift = abs(I_out - I_in) / max(1.0, I_in)

        # ─── 10. Certificados booleanos de conservación ───
        strict_tol = max(tolerance, _TOPOLOGY_DEGRADED_THRESHOLD)
        is_kcl = kcl_res <= strict_tol
        is_kvl = kvl_res <= strict_tol
        is_tel = tellegen_res <= strict_tol

        # ─── 11. Residuo meta-espectral ───
        meta_res = max(
            kcl_res if math.isfinite(kcl_res) else float("inf"),
            kvl_res if math.isfinite(kvl_res) else float("inf"),
            tellegen_res if math.isfinite(tellegen_res) else float("inf"),
            duality_res,
            recurrence_res,
            adiabatic_drift,
        )
        if not math.isfinite(meta_res):
            meta_res = float("inf")

        # ─── 12. Clasificación en el álgebra de Heyting Ω₃ ───
        own_verdict = SpecularSovereignVerdict.COHERENT
        if (
            not math.isfinite(meta_res)
            or meta_res > _TOPOLOGY_VETO_THRESHOLD
            or rayleigh_dissipation < -tolerance
            or not is_poincare_dual
        ):
            own_verdict = SpecularSovereignVerdict.VETOED
        elif (
            not is_kcl
            or not is_kvl
            or not is_tel
            or rayleigh_dissipation < 0.0
            or meta_res > tolerance
            or adiabatic_drift > _ADIABATIC_DRIFT_LIMIT
        ):
            own_verdict = SpecularSovereignVerdict.DEGRADED

        verdict = SpecularSovereignVerdict.supremum(
            own_verdict, phase1_cert.verdict, phase2_cert.verdict
        )

        return Phase3TellegenCertificate(
            tellegen_residual=tellegen_res,
            kcl_residual=kcl_res,
            kvl_residual=kvl_res,
            is_tellegen_conserved=is_tel,
            is_kcl_conserved=is_kcl,
            is_kvl_conserved=is_kvl,
            is_lyapunov_passive=is_lyap_passive,
            graph_order=int(n_nodes),
            graph_size=int(m),
            cyclomatic_number=int(cycle_dim),
            sparse_computation=sparse_computation,
            verdict=verdict,
            cycle_space_dimension=cycle_dim,
            cut_space_dimension=cut_dim,
            poincare_duality_residual=duality_res,
            is_poincare_dual=is_poincare_dual,
            euler_characteristic=euler_char,
            recurrence_residual=recurrence_res,
            adiabatic_action_drift=float(adiabatic_drift),
            meta_residual=float(meta_res),
            rsi_depth_reached=1,
        )

    # Alias de compatibilidad retroactiva
    decide_conservation_and_passivity = decide_tellegen_lyapunov_conservation

    def _combine_phase_residuals(
        self,
        phase1_cert: Phase1AllieviCertificate,
        phase2_cert: Phase2HouseholderCertificate,
        phase3_cert: Phase3TellegenCertificate,
    ) -> float:
        """Norma del supremo (ℓ_∞) de los residuos acoplados de las 3 fases."""
        parts = [
            float(phase2_cert.reflection_residual),
            float(phase2_cert.specular_residual),
            float(phase2_cert.state_norm_residual),
            float(phase2_cert.action_drift),
            float(phase3_cert.tellegen_residual),
            float(phase3_cert.kcl_residual),
            float(phase3_cert.kvl_residual),
            float(phase3_cert.poincare_duality_residual),
            float(phase3_cert.recurrence_residual),
            float(phase3_cert.adiabatic_action_drift),
            float(phase1_cert.expansion_residual),
        ]
        finite = [v for v in parts if math.isfinite(v)]
        return float(max(finite)) if finite else float("inf")

    # ═════════════════════════════════════════════════════════════════════
    # ESLABÓN FUNTORIAL 3 → META: Puente formal al Meta-Controlador RSI-3
    # ═════════════════════════════════════════════════════════════════════
    def _phase3_to_meta_bridge(
        self,
        phase1_cert: Phase1AllieviCertificate,
        phase2_cert: Phase2HouseholderCertificate,
        phase3_cert: Phase3TellegenCertificate,
    ) -> Tuple[float, SpecularSovereignVerdict]:
        """
        MORFISMO DE CLAUSURA (FASE ANIDADA 3 ──► META-CONTROLADOR RSI-3).

        Extrae el meta-residuo combinado y el veredicto consolidado en Ω₃;
        constituye el contrato formal que consume `_recursive_self_improvement_loop`.
        """
        residual = self._combine_phase_residuals(
            phase1_cert, phase2_cert, phase3_cert
        )
        verdict = SpecularSovereignVerdict.supremum(
            phase1_cert.verdict, phase2_cert.verdict, phase3_cert.verdict
        )
        return residual, verdict

    def _rsi_compute_trace(
        self,
        level: int,
        tolerance: float,
        residual: float,
        traces_prev: Tuple[RecursiveSelfImprovementTrace, ...],
        target_residual: float = _RSI_TARGET_RESIDUAL,
    ) -> RecursiveSelfImprovementTrace:
        """
        Construye el certificado de evolución de un nivel en la jerarquía RSI-3:
          • Nivel 1: Observación base de defectos acoplados.
          • Nivel 2: Extrapolación de Richardson y orden de Fréchet-Hölder p.
          • Nivel 3: Aceleración Δ² de Aitken y modulación homotópica de Banach-Fréchet.
        """
        convergence_rate = 0.0
        richardson = residual
        aitken = residual
        adaptation_factor = 0.5
        level3_active = False

        if len(traces_prev) >= 1:
            prev = traces_prev[-1]
            convergence_rate, richardson = _richardson_extrapolation(
                prev.residual, residual, prev.tolerance, tolerance
            )
            adaptation_factor = _adaptive_refinement_factor(
                current_residual=residual,
                target_residual=target_residual,
                convergence_rate=convergence_rate,
                level3_active=False,
                aitken_signal=residual,
                richardson_signal=richardson,
            )

        if len(traces_prev) >= 2:
            level3_active = True
            t_m2, t_m1 = traces_prev[-2], traces_prev[-1]
            aitken = _aitken_delta_squared(
                t_m2.residual, t_m1.residual, residual
            )
            adaptation_factor = _adaptive_refinement_factor(
                current_residual=residual,
                target_residual=target_residual,
                convergence_rate=convergence_rate,
                level3_active=True,
                aitken_signal=aitken,
                richardson_signal=richardson,
            )

        if residual <= target_residual:
            verdict = SpecularSovereignVerdict.COHERENT
        elif math.isfinite(residual) and residual < 1.0:
            verdict = SpecularSovereignVerdict.DEGRADED
        else:
            verdict = SpecularSovereignVerdict.VETOED

        return RecursiveSelfImprovementTrace(
            level=level,
            tolerance=tolerance,
            residual=residual,
            convergence_rate=convergence_rate,
            richardson_extrapolation=richardson,
            aitken_delta_squared=aitken,
            adaptation_factor=adaptation_factor,
            level3_active=level3_active,
            verdict=verdict,
        )

    def _recursive_self_improvement_loop(
        self,
        run_pipeline: Callable[
            [float],
            Tuple[
                Phase1AllieviCertificate,
                Phase2HouseholderCertificate,
                Phase3TellegenCertificate,
            ],
        ],
        max_depth: int = _RSI_MAX_DEPTH,
        initial_tolerance: float = _DEFAULT_TOL,
        target_residual: float = _RSI_TARGET_RESIDUAL,
    ) -> SelfImprovementResult:
        """
        Meta-controlador cerrado RSI-3 que consume `_phase3_to_meta_bridge`
        en cada iteración recursiva.
        """
        if max_depth < 1:
            raise RecursiveSelfImprovementError(
                "RecursiveSelfImprovementError: max_depth debe ser ≥ 1."
            )

        traces: list[RecursiveSelfImprovementTrace] = []
        tolerance = float(max(_RSI_MIN_TOLERANCE, initial_tolerance))

        for level in range(1, max_depth + 1):
            p1, p2, p3 = run_pipeline(tolerance)
            residual, _ = self._phase3_to_meta_bridge(p1, p2, p3)

            trace = self._rsi_compute_trace(
                level=level,
                tolerance=tolerance,
                residual=residual,
                traces_prev=tuple(traces),
                target_residual=target_residual,
            )
            traces.append(trace)

            if residual <= target_residual:
                break

            factor = trace.adaptation_factor
            if not math.isfinite(factor) or factor <= 0.0:
                factor = 0.1
            next_tol = float(max(_RSI_MIN_TOLERANCE, tolerance * factor))
            if next_tol >= tolerance:
                next_tol = max(_RSI_MIN_TOLERANCE, tolerance * 0.5)
            tolerance = next_tol

        last = traces[-1]
        level3_activated = any(t.level3_active for t in traces)

        verdict = SpecularSovereignVerdict.COHERENT
        for t in traces:
            verdict = SpecularSovereignVerdict.supremum(verdict, t.verdict)

        return SelfImprovementResult(
            traces=tuple(traces),
            final_tolerance=float(last.tolerance),
            final_residual=float(last.residual),
            converged=bool(last.residual <= target_residual),
            richardson_limit=float(last.richardson_extrapolation),
            aitken_limit=float(last.aitken_delta_squared),
            effective_convergence_order=float(last.convergence_rate),
            level3_activated=bool(level3_activated),
            verdict=verdict,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Soberano del Flujo Especular (Morfismo en la Categoría T_MIC)
# ═══════════════════════════════════════════════════════════════════════════
class SpecularFlowSovereignAgent(Morphism, Phase3TellegenPassivityValidator):
    """
    Agente Soberano del haz de De Rham para el control en lazo cerrado del
    Flujo Especular, integrando las 3 Fases Anidadas con mecánica celeste de
    Poincaré-KAM y automejora recursiva de nivel 3 (RSI-3).
    """

    def __init__(self, raise_on_veto: bool = False) -> None:
        super().__init__()
        self._target_stratum: Stratum = Stratum.WISDOM
        self._raise_on_veto: bool = raise_on_veto

    def execute_sovereign_governance(
        self,
        potential_y: float,
        flow_q: float,
        wave_speed_a: float,
        gravity_g: float,
        section_s: float,
        metric_tensor_g: Optional[NDArray[np.float64]],
        boundary_normal_n: Optional[NDArray[np.float64]],
        state_vector: Optional[NDArray[np.float64]],
        pressures: NDArray[np.float64],
        flows: NDArray[np.float64],
        rayleigh_dissipation: float,
        incidence_matrix: Optional[Union[NDArray[np.float64], csr_matrix]] = None,
        nodal_potentials: Optional[NDArray[np.float64]] = None,
        tolerance: float = _DEFAULT_TOL,
        cfl_number: Optional[float] = None,
        poincare_section_level: float = 0.0,
    ) -> SpecularFlowState:
        """
        Orquesta el ciclo OODA sobre la variedad característica de
        Allievi-Householder con validación cohomológica de Tellegen-Hodge y
        auditoría de Poincaré-KAM (ejecución directa de nivel base).
        """
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        try:
            cert_1, cert_2, cert_3 = self._run_pipeline_inner(
                potential_y=potential_y,
                flow_q=flow_q,
                wave_speed_a=wave_speed_a,
                gravity_g=gravity_g,
                section_s=section_s,
                metric_tensor_g=metric_tensor_g,
                boundary_normal_n=boundary_normal_n,
                state_vector=state_vector,
                pressures=pressures,
                flows=flows,
                rayleigh_dissipation=rayleigh_dissipation,
                incidence_matrix=incidence_matrix,
                nodal_potentials=nodal_potentials,
                tolerance=tolerance,
                cfl_number=cfl_number,
                poincare_section_level=poincare_section_level,
            )
            return self._finalize_state(
                cert_1, cert_2, cert_3, timestamp_utc, None
            )
        except SpecularFlowAgentError as exc:
            logger.error("Colapso categórico del agente especular: %s", exc)
            if self._raise_on_veto:
                raise
            return self._cataclysm_state(
                reason=str(exc), timestamp_utc=timestamp_utc
            )
        except Exception as exc:
            logger.exception("Colapso no tipado del agente especular.")
            if self._raise_on_veto:
                raise SpecularFlowAgentError(
                    "Colapso no tipado del agente especular."
                ) from exc
            return self._cataclysm_state(
                reason=str(exc), timestamp_utc=timestamp_utc
            )

    def execute_sovereign_governance_rsi3(
        self,
        potential_y: float,
        flow_q: float,
        wave_speed_a: float,
        gravity_g: float,
        section_s: float,
        metric_tensor_g: Optional[NDArray[np.float64]],
        boundary_normal_n: Optional[NDArray[np.float64]],
        state_vector: Optional[NDArray[np.float64]],
        pressures: NDArray[np.float64],
        flows: NDArray[np.float64],
        rayleigh_dissipation: float,
        incidence_matrix: Optional[Union[NDArray[np.float64], csr_matrix]] = None,
        nodal_potentials: Optional[NDArray[np.float64]] = None,
        initial_tolerance: float = _DEFAULT_TOL,
        cfl_number: Optional[float] = None,
        poincare_section_level: float = 0.0,
        rsi_max_depth: int = _RSI_MAX_DEPTH,
        rsi_target_residual: float = _RSI_TARGET_RESIDUAL,
    ) -> SpecularFlowState:
        """
        Orquesta el ciclo OODA gobernado por el meta-controlador de Automejora
        Recursiva de Nivel 3 (RSI-3).
        """
        timestamp_utc = datetime.now(timezone.utc).isoformat()
        try:
            last_bundle: Optional[
                Tuple[
                    Phase1AllieviCertificate,
                    Phase2HouseholderCertificate,
                    Phase3TellegenCertificate,
                ]
            ] = None

            def run_pipeline(
                tol: float,
            ) -> Tuple[
                Phase1AllieviCertificate,
                Phase2HouseholderCertificate,
                Phase3TellegenCertificate,
            ]:
                nonlocal last_bundle
                last_bundle = self._run_pipeline_inner(
                    potential_y=potential_y,
                    flow_q=flow_q,
                    wave_speed_a=wave_speed_a,
                    gravity_g=gravity_g,
                    section_s=section_s,
                    metric_tensor_g=metric_tensor_g,
                    boundary_normal_n=boundary_normal_n,
                    state_vector=state_vector,
                    pressures=pressures,
                    flows=flows,
                    rayleigh_dissipation=rayleigh_dissipation,
                    incidence_matrix=incidence_matrix,
                    nodal_potentials=nodal_potentials,
                    tolerance=tol,
                    cfl_number=cfl_number,
                    poincare_section_level=poincare_section_level,
                )
                return last_bundle

            rsi_result = self._recursive_self_improvement_loop(
                run_pipeline=run_pipeline,
                max_depth=rsi_max_depth,
                initial_tolerance=initial_tolerance,
                target_residual=rsi_target_residual,
            )

            if last_bundle is None:
                cert_1, cert_2, cert_3 = run_pipeline(rsi_result.final_tolerance)
            else:
                cert_1, cert_2, cert_3 = last_bundle

            depth_reached = len(rsi_result.traces)
            cert_3_updated = Phase3TellegenCertificate(
                tellegen_residual=cert_3.tellegen_residual,
                kcl_residual=cert_3.kcl_residual,
                kvl_residual=cert_3.kvl_residual,
                is_tellegen_conserved=cert_3.is_tellegen_conserved,
                is_kcl_conserved=cert_3.is_kcl_conserved,
                is_kvl_conserved=cert_3.is_kvl_conserved,
                is_lyapunov_passive=cert_3.is_lyapunov_passive,
                graph_order=cert_3.graph_order,
                graph_size=cert_3.graph_size,
                cyclomatic_number=cert_3.cyclomatic_number,
                sparse_computation=cert_3.sparse_computation,
                verdict=cert_3.verdict,
                cycle_space_dimension=cert_3.cycle_space_dimension,
                cut_space_dimension=cert_3.cut_space_dimension,
                poincare_duality_residual=cert_3.poincare_duality_residual,
                is_poincare_dual=cert_3.is_poincare_dual,
                euler_characteristic=cert_3.euler_characteristic,
                recurrence_residual=cert_3.recurrence_residual,
                adiabatic_action_drift=cert_3.adiabatic_action_drift,
                meta_residual=rsi_result.final_residual,
                rsi_depth_reached=depth_reached,
            )

            return self._finalize_state(
                cert_1, cert_2, cert_3_updated, timestamp_utc, rsi_result
            )
        except SpecularFlowAgentError as exc:
            logger.error(
                "Colapso categórico del agente especular (RSI-3): %s", exc
            )
            if self._raise_on_veto:
                raise
            return self._cataclysm_state(
                reason=str(exc), timestamp_utc=timestamp_utc
            )
        except Exception as exc:
            logger.exception("Colapso no tipado del agente especular (RSI-3).")
            if self._raise_on_veto:
                raise SpecularFlowAgentError(
                    "Colapso no tipado del agente especular (RSI-3)."
                ) from exc
            return self._cataclysm_state(
                reason=str(exc), timestamp_utc=timestamp_utc
            )

    def _run_pipeline_inner(
        self,
        *,
        potential_y: float,
        flow_q: float,
        wave_speed_a: float,
        gravity_g: float,
        section_s: float,
        metric_tensor_g: Optional[NDArray[np.float64]],
        boundary_normal_n: Optional[NDArray[np.float64]],
        state_vector: Optional[NDArray[np.float64]],
        pressures: NDArray[np.float64],
        flows: NDArray[np.float64],
        rayleigh_dissipation: float,
        incidence_matrix: Optional[Union[NDArray[np.float64], csr_matrix]],
        nodal_potentials: Optional[NDArray[np.float64]],
        tolerance: float,
        cfl_number: Optional[float],
        poincare_section_level: float,
    ) -> Tuple[
        Phase1AllieviCertificate,
        Phase2HouseholderCertificate,
        Phase3TellegenCertificate,
    ]:
        """Ejecuta las tres fases anidadas preservando la geometría purificada."""
        # ─── FASE ANIDADA 1 ───
        cert_1 = self.observe_allievi_characteristics(
            potential_y=potential_y,
            flow_q=flow_q,
            wave_speed_a=wave_speed_a,
            gravity_g=gravity_g,
            section_s=section_s,
            tolerance=tolerance,
            cfl_number=cfl_number,
            poincare_section_level=poincare_section_level,
        )

        if cert_1.verdict.is_vetoed:
            return (
                cert_1,
                self._vetoed_phase2_certificate(),
                self._vetoed_phase3_certificate(),
            )

        # ─── FASE ANIDADA 2 ───
        cert_2 = self.orient_householder_specular_reflection(
            metric_tensor_g=metric_tensor_g,
            boundary_normal_n=boundary_normal_n,
            state_vector=state_vector,
            phase1_cert=cert_1,
            tolerance=tolerance,
        )

        if cert_2.verdict.is_vetoed:
            return cert_1, cert_2, self._vetoed_phase3_certificate()

        # ─── FASE ANIDADA 3 ───
        # Sintetiza exactamente el operador Higham + Newton-Schulz de Fase 2
        char_dir, _, _, g_cal, _, _ = self._phase1_to_phase2_bridge(cert_1)
        try:
            _, _, _, M_purified, x_ref, _, _ = (
                self._synthesize_covariant_householder(
                    metric_tensor_g=metric_tensor_g,
                    boundary_normal_n=boundary_normal_n,
                    state_vector=state_vector,
                    char_dir=char_dir,
                    g_cal=g_cal,
                    tolerance=tolerance,
                )
            )
        except Exception:
            M_purified, x_ref = None, None

        cert_3 = self.decide_tellegen_lyapunov_conservation(
            pressures=pressures,
            flows=flows,
            rayleigh_dissipation=rayleigh_dissipation,
            phase1_cert=cert_1,
            phase2_cert=cert_2,
            incidence_matrix=incidence_matrix,
            nodal_potentials=nodal_potentials,
            operator_M=M_purified,
            state_x=x_ref,
            tolerance=tolerance,
        )

        return cert_1, cert_2, cert_3

    def _finalize_state(
        self,
        cert_1: Phase1AllieviCertificate,
        cert_2: Phase2HouseholderCertificate,
        cert_3: Phase3TellegenCertificate,
        timestamp_utc: str,
        rsi_result: Optional[SelfImprovementResult],
    ) -> SpecularFlowState:
        """Sintetiza el SpecularFlowState terminal y el enclavamiento Crowbar."""
        final_verdict = SpecularSovereignVerdict.supremum(
            cert_1.verdict, cert_2.verdict, cert_3.verdict
        )
        if rsi_result is not None:
            final_verdict = SpecularSovereignVerdict.supremum(
                final_verdict, rsi_result.verdict
            )

        crowbar_triggered = False
        crowbar_action = CrowbarAction.NONE
        diagnostic_note = "Ciclo OODA especular completado."

        if final_verdict == SpecularSovereignVerdict.VETOED:
            crowbar_triggered = True
            crowbar_action = CrowbarAction.HARD_SHORT
            diagnostic_note = (
                "VETO ESPECTRAL DE FLUJO: fuga de pasividad, conservación o "
                "unitariedad. Gatillando Crowbar."
            )
            logger.error(
                "¡VETO ESPECTRAL DE FLUJO! Gatillando Crowbar (GPIO%d).",
                _CROWBAR_GPIO,
            )
            if self._raise_on_veto:
                raise SpecularFlowAgentError(
                    "Obstrucción de pasividad, conservación o unitariedad "
                    "en el Flujo Especular."
                )
        elif final_verdict == SpecularSovereignVerdict.DEGRADED:
            crowbar_triggered = True
            crowbar_action = CrowbarAction.WATCHDOG_PULSE
            diagnostic_note = (
                "Degradación paramétrica detectada. Activando Watchdog."
            )
            logger.warning(
                "Degradación paramétrica en el Flujo Especular. Activando Watchdog."
            )

        provenance_hash = self._generate_provenance_hash(
            cert_1, cert_2, cert_3, timestamp_utc, rsi_result
        )

        return SpecularFlowState(
            phase1=cert_1,
            phase2=cert_2,
            phase3=cert_3,
            final_verdict=final_verdict,
            crowbar_triggered=crowbar_triggered,
            crowbar_action=crowbar_action,
            timestamp_utc=timestamp_utc,
            provenance_hash=provenance_hash,
            diagnostic_note=diagnostic_note,
            rsi_result=rsi_result,
        )

    def _vetoed_phase2_certificate(self) -> Phase2HouseholderCertificate:
        return Phase2HouseholderCertificate(
            is_reflection_unitary=False,
            reflection_residual=float("inf"),
            is_anti_symplectic=False,
            is_characteristic_consistent=False,
            specular_residual=float("inf"),
            incidence_cosine=0.0,
            state_norm_residual=float("inf"),
            metric_condition_number=float("inf"),
            higham_regularization_applied=False,
            verdict=SpecularSovereignVerdict.VETOED,
        )

    def _vetoed_phase3_certificate(self) -> Phase3TellegenCertificate:
        return Phase3TellegenCertificate(
            tellegen_residual=float("inf"),
            kcl_residual=float("inf"),
            kvl_residual=float("inf"),
            is_tellegen_conserved=False,
            is_kcl_conserved=False,
            is_kvl_conserved=False,
            is_lyapunov_passive=False,
            graph_order=0,
            graph_size=0,
            cyclomatic_number=0,
            sparse_computation=False,
            verdict=SpecularSovereignVerdict.VETOED,
        )

    def _cataclysm_state(
        self, reason: str, timestamp_utc: str
    ) -> SpecularFlowState:
        phase1_dummy = Phase1AllieviCertificate(
            is_characteristic_stable=False,
            characteristic_impedance=0.0,
            riemann_invariants=(0.0, 0.0),
            characteristic_direction=(0.0, 0.0),
            characteristic_covector=(0.0, 0.0),
            energy_norm=0.0,
            impedance_condition_number=float("inf"),
            cfl_proxy=0.0,
            verdict=SpecularSovereignVerdict.VETOED,
        )
        phase2_dummy = self._vetoed_phase2_certificate()
        phase3_dummy = self._vetoed_phase3_certificate()

        raw_payload = f"CATACLYSM_SPECULAR_VETO::{reason}"
        provenance_hash = hashlib.sha256(
            raw_payload.encode("utf-8")
        ).hexdigest()

        return SpecularFlowState(
            phase1=phase1_dummy,
            phase2=phase2_dummy,
            phase3=phase3_dummy,
            final_verdict=SpecularSovereignVerdict.VETOED,
            crowbar_triggered=True,
            crowbar_action=CrowbarAction.HARD_SHORT,
            timestamp_utc=timestamp_utc,
            provenance_hash=provenance_hash,
            diagnostic_note=f"CATACLYSM_SPECULAR_VETO: {reason}",
            rsi_result=None,
        )

    def _generate_provenance_hash(
        self,
        c1: Phase1AllieviCertificate,
        c2: Phase2HouseholderCertificate,
        c3: Phase3TellegenCertificate,
        timestamp_utc: str,
        rsi: Optional[SelfImprovementResult] = None,
    ) -> str:
        """Sello criptográfico SHA-256 de los invariantes de las 3 fases y RSI-3."""
        fields = [
            timestamp_utc,
            str(c1.verdict.value),
            str(c2.verdict.value),
            str(c3.verdict.value),
            f"{c1.characteristic_impedance:.12e}",
            f"{c1.energy_norm:.12e}",
            f"{c1.impedance_condition_number:.12e}",
            f"{c1.action_invariant:.12e}",
            f"{c1.small_parameter_epsilon:.12e}",
            f"{c1.corrected_action:.12e}",
            str(int(c1.is_near_resonance)),
            str(int(c1.resonance_p)),
            str(int(c1.resonance_q)),
            f"{c2.reflection_residual:.12e}",
            f"{c2.specular_residual:.12e}",
            f"{c2.state_norm_residual:.12e}",
            f"{c2.metric_condition_number:.12e}",
            f"{c2.action_drift:.12e}",
            str(int(c2.is_return_map_stable)),
            f"{c3.tellegen_residual:.12e}",
            f"{c3.kcl_residual:.12e}",
            f"{c3.kvl_residual:.12e}",
            f"{c3.poincare_duality_residual:.12e}",
            f"{c3.recurrence_residual:.12e}",
            f"{c3.adiabatic_action_drift:.12e}",
            str(int(c3.graph_order)),
            str(int(c3.graph_size)),
            str(int(c3.cyclomatic_number)),
            str(int(c3.sparse_computation)),
            str(int(c2.higham_regularization_applied)),
        ]
        if rsi is not None:
            fields.extend([
                f"{rsi.final_tolerance:.12e}",
                f"{rsi.final_residual:.12e}",
                f"{rsi.richardson_limit:.12e}",
                f"{rsi.aitken_limit:.12e}",
                f"{rsi.effective_convergence_order:.6f}",
                str(int(rsi.converged)),
                str(int(rsi.level3_activated)),
            ])
        raw_payload = "|".join(fields)
        return hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════
# Alias de compatibilidad para versiones previas
# ═══════════════════════════════════════════════════════════════════════════
Phase1_AllieviCharacteristicSolver = Phase1AllieviCharacteristicSolver
Phase2_SpecularHouseholderReflector = Phase2SpecularHouseholderReflector
Phase3_TellegenPassivityValidator = Phase3TellegenPassivityValidator


__all__ = [
    "SpecularFlowAgentError",
    "AllieviCharacteristicError",
    "HouseholderReflectionError",
    "TellegenConservationError",
    "PoincareSectionError",
    "RecursiveSelfImprovementError",
    "SpecularSovereignVerdict",
    "CrowbarAction",
    "Phase1AllieviCertificate",
    "Phase2HouseholderCertificate",
    "Phase3TellegenCertificate",
    "PoincareSectionSnapshot",
    "LindstedtPoincareExpansion",
    "RecursiveSelfImprovementTrace",
    "SelfImprovementResult",
    "SpecularFlowState",
    "Phase1AllieviCharacteristicSolver",
    "Phase2SpecularHouseholderReflector",
    "Phase3TellegenPassivityValidator",
    "SpecularFlowSovereignAgent",
    "stable_cholesky_higham",
    "verify_tellegen_sparse",
]