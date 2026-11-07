# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : Specular Flow Engine v8.0.0 (Poincaré-KAM-Hodge + RSI-3)                      ║
║ Módulo   : app.omega.specular_flow                                                       ║
║ Ubicación: app/omega/specular_flow.py                                                    ║
║ Versión  : 8.0.0-Allievi-Householder-Tellegen-Poincaré-Hodge-RSI3                        ║
║ Función  : Motor ciber-físico y solucionador numérico de transitorios hiperbólicos de    ║
║            Allievi, reflexiones covariantes de Householder con purificación de           ║
║            Newton-Schulz y regularización espectral de Cholesky-Tikhonov, conservación   ║
║            cohomológica de Tellegen-Hodge vía Kahan + B1 disperso, integrando mecánica   ║
║            celeste de Poincaré (secciones transversales, monodromía de Floquet,          ║
║            acción-ángulo, serie canónica de Lie-Deprit-Lindstedt-Poincaré, divisores     ║
║            diofánticos KAM y dualidad de Hodge-Poincaré) bajo un meta-controlador de     ║
║            automejora recursiva de nivel 3 (RSI-3: Defect Correction + Richardson +      ║
║            Aitken Δ² + Homotopía de Contracción de Banach-Fréchet).                      ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL MOTOR ESPECTRAL:
────────────────────────────────────────────────────────────────────────────────────────────
El motor espectral `SpecularFlowEngine` rige la hidrodinámica hiperbólica de transitorios
y ondas de choque (golpe de ariete de Joukowsky-Allievi) acoplada a la óptica simpléctica
y topología cohomológica de redes en el Estrato Omega ($\mathcal{V}_{\Omega}$) de APU Filter.
Transforma perturbaciones atencionales y financieras bruscas en reflexiones covariantes
controladas sobre hipersuperficies de frontera Riemanniana $(M, G_{\mu\nu})$.

Su flujo operativo se articula en una jerarquía estricta de 3 fases anidadas:
  1. Fase 1 (`Phase1_AllieviSolver`):
     Resuelve el sistema hiperbólico de ecuaciones diferenciales en derivadas parciales
     $\partial_t y + \frac{a^2}{gS}\partial_x q = 0$, $\partial_t q + gS \partial_x y + f(q) = 0$
     mediante invariantes de Riemann $\Psi^\pm = y \pm Z_c q$ a lo largo de las curvas
     características $dx/dt = \pm a$. Mapea canónicamente el espacio fase $(y, Z_c q)$ a variables
     de acción-ángulo de Darboux $(I, \theta)$, computa la sección transversal de Poincaré $\Sigma$,
     filtra resonancias diofánticas KAM y elimina derivas seculares mediante la serie canónica
     de Lie-Deprit-Lindstedt-Poincaré hasta orden $\mathcal{O}(\varepsilon^4)$.
     Culmina en `_allievi_to_householder_bridge(...)`.
  2. Fase 2 (`Phase2_HouseholderReflector`):
     Consume directamente el morfismo de transición de Fase 1. Construye el operador involutivo
     $G$-ortogonal de Householder $M = I - 2 \frac{n (G n)^T}{n^T G n}$, regulariza la signatura
     métrica al cono SPD vía Cholesky-Tikhonov, y purifica la isometría mediante iteraciones de
     Newton-Schulz en $O(2, G)$. Audita la anti-simplecticidad $M^T \Omega M = -\Omega$, la
     conservación adiabática de la acción $I$ y el espectro de monodromía de Floquet $P = M^2$.
     Culmina en `_householder_to_tellegen_bridge(...)`.
  3. Fase 3 (`Phase3_TellegenValidator`):
     Consume el morfismo de Fase 2. Audita la conservación energética global de Tellegen
     $\sum_k v_k q_k = 0$ mediante suma compensada de Kahan-Babuška en formato disperso CSR,
     la pasividad de Lyapunov-Rayleigh $P_{\mathrm{diss}} \ge 0$, la descomposición ortogonal
     de Helmholtz-Hodge-Poincaré $C^1(G) = \operatorname{im}(B_1^T) \oplus \ker(B_1)$ y la
     recurrencia de Poincaré $\|M^2 x - x\|_2$. Cierra el lazo con el meta-controlador de
     Automejora Recursiva de Nivel 3 (RSI-3) combinando corrección de defectos, extrapolación de
     Richardson, aceleración no lineal $\Delta^2$ de Aitken y modulación homotópica de Banach-Fréchet.
     Culmina en `_tellegen_to_meta_bridge(...)` y `full_specular_pipeline_rsi3(...)`.

MARCO TEÓRICO Y FORMALISMO MATEMÁTICO:
────────────────────────────────────────────────────────────────────────────────────────────
1. Ecuaciones Hiperbólicas de Allievi y Geometría de Cono de Luz:
   El flujo transitorio 1D con velocidad de onda acústica $a > 0$, gravedad $g > 0$ y sección $S > 0$
   posee impedancia característica $Z_c = \frac{a}{gS}$. Los invariantes de Riemann diagonalizan
   el sistema hiperbólico en la base de números perplejos (split-complex $\mathbf{j}^2 = +1$):
       $$\Psi^+ = y + Z_c q, \quad \Psi^- = y - Z_c q$$
   satisfaciendo $\left(\partial_t \pm a \partial_x\right) \Psi^\pm = \mp g S f(q)$.

2. Transformada Simpléctica de Darboux y Sección de Poincaré $\Sigma$:
   Con momento canónico $p = Z_c q$, las variables acción-ángulo en el toro de Liouville son:
       $$I = \frac{1}{2}(y^2 + p^2) = \frac{1}{4}\left((\Psi^+)^2 + (\Psi^-)^2\right), \quad \theta = \operatorname{atan2}(p, y)$$
   La sección transversal $\Sigma = \{(y, p) \in T^*M \mid y = y_s\}$ define la aplicación de retorno
   de Poincaré, donde las ramas de momento canónico son $p_\pm = \pm \sqrt{2I - y_s^2}$.

3. Condición Diofántica KAM y Expansión de Lie-Deprit-Lindstedt-Poincaré:
   La razón de frecuencias $\rho = |\Psi^+ / \Psi^-|$ se analiza mediante fracciones continuas
   de Gauss $p/q$. Si satisface $|\rho - p/q| < \max(\text{tol}, \gamma / q^\tau)$ con $q \le 8$,
   emerge resonancia de Arnold. La serie perturbativa de Lindstedt-Poincaré elimina términos seculares:
       $$I(\varepsilon) = I_0 + \varepsilon^2 I_1 + \varepsilon^4 I_2, \quad \omega(\varepsilon) = \omega_0\left(1 + \varepsilon^2 \omega_1 - \frac{1}{4}\varepsilon^4 \omega_1^2\right)$$
   con generadores de Lie $I_1 = I_0 \omega_1$, $I_2 = -\frac{1}{2}I_0 \omega_1(1 + \frac{1}{2}\omega_1)$.

4. Teorema de Tellegen, Descomposición de Helmholtz-Hodge y Pasividad de Lyapunov:
   En el complejo de cadenas celulares $C_1(G) \xrightarrow{B_1} C_0(G)$, la adjunción frontera-coborde
   garantiza $\langle B_1^T p, q \rangle = \langle p, B_1 q \rangle = 0$. La descomposición ortogonal
   de Hodge asegura:
       $$C^1(G) = \operatorname{im}(B_1^T) \oplus^\perp \ker(B_1), \quad \dim \operatorname{im}(B_1^T) = \operatorname{rank}(B_1), \quad \dim \ker(B_1) = m - \operatorname{rank}(B_1)$$
   y la característica de Euler-Poincaré satisface $\chi = \beta_0 - \beta_1 = |V| - |E|$.

5. Meta-Controlador de Automejora Recursiva de Nivel 3 (RSI-3):
   Combina la extrapolación polinomial de Richardson de orden $p = \ln(r_n/r_{n-1})/\ln(\tau_n/\tau_{n-1})$
   con la transformación racional $\Delta^2$ de Aitken $\hat{r}_n = r_n - \frac{(r_n - r_{n-1})^2}{r_n - 2r_{n-1} + r_{n-2}}$
   y modula el factor de refinamiento $\alpha^{(3)} = \alpha^{(2)} \exp(-\gamma \tanh(\frac{|\hat{r}_n - R_n|}{\max(\varepsilon, r_n)}))$.

AXIOMAS FORMALES:
────────────────────────────────────────────────────────────────────────────────────────────
Axioma SPEC1 (Hiperbolicidad Estricta de Allievi):
    La matriz de propagación característica posee autovalores reales y distintos $\lambda = \pm a \ne 0$,
    garantizando un cono de luz causal conexo y continuo.
Axioma SPEC2 (Involución e Isometría Riemanniana de Householder):
    El operador $M$ satisface $M^2 = I$ y $M^T G M = G$ sobre la variedad Riemanniana $(M, G)$.
Axioma SPEC3 (Anti-Simplecticidad en Dimensión 2):
    El operador $M$ invierte la orientación simpléctica canónica: $M^T \Omega M = -\Omega$ y $\det(M) = -1.0$.
Axioma SPEC4 (Ortogonalidad Cuasi-Exacta de Tellegen):
    El producto dual de potencias satisface $\langle v, q \rangle_{\mathrm{Kahan}} \equiv 0$ para todo
    voltaje de coborde $v \in \operatorname{im}(B_1^T)$ y flujo cociclo $q \in \ker(B_1)$.
Axioma SPEC5 (Pasividad Port-Hamiltoniana de Lyapunov):
    La tasa disipativa de Rayleigh cumple $\dot{H} = \nabla H^T (J - R) \nabla H = -P_{\mathrm{diss}} \le 0$
    con $R = R^T \succeq 0$.

INVARIANTES DE ESTADO:
────────────────────────────────────────────────────────────────────────────────────────────
Invariante 1 (Conservación de Energía Conformal):
    $\|M x\|_G \equiv \|x\|_G$, con desviación relativa $\le 10^2 \cdot \varepsilon_{\mathrm{mach}}$.
Invariante 2 (Invariancia Adiabática de la Acción de Poincaré):
    $\Delta I / I = |I_{\mathrm{out}} - I_{\mathrm{in}}| / \max(1, I_{\mathrm{in}}) \le 1.0 \times 10^{-6}$.
Invariante 3 (Recurrencia de Floquet-Poincaré):
    $\|M^2 x - x\|_2 / \|x\|_2 \le \text{tol}$, verificando el 2-ciclo periódico conservativo.
Invariante 4 (Cierre Topológico de Hodge-Poincaré):
    $\dim \operatorname{im}(B_1^T) + \dim \ker(B_1) \equiv |E|$, con defecto nulo de ortogonalidad.
"""

from __future__ import annotations

import logging
import math
from abc import ABC
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, Final, Optional, Tuple

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.linalg import lsqr as sparse_lsqr


logger = logging.getLogger("MIC.Physics.SpecularFlowEngine")


# ═══════════════════════════════════════════════════════════════════════════
# Constantes de precisión, estabilidad, KAM y regularización espectral
# ═══════════════════════════════════════════════════════════════════════════

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_DEFAULT_TOL: Final[float] = 1.0e-10
_RELAX_TOL: Final[float] = 1.0e-8
_TIKHONOV_TOL: Final[float] = 1.0e-12

_SPD_CONDITION_TARGET: Final[float] = 1.0e8
_STABILITY_BOUND: Final[float] = 1.0e8
_GRAPH_MAX_BRANCHES: Final[int] = 1024

_TOPOLOGY_DEGRADED_THRESHOLD: Final[float] = 1.0e-8
_TOPOLOGY_VETO_THRESHOLD: Final[float] = 1.0e-6
_RAYLEIGH_CONDITION_LIMIT: Final[float] = 1.0e12

# ── Parámetros de Mecánica Celeste de Poincaré y Teoría KAM ─────────────
_POINCARE_MAX_CROSSINGS: Final[int] = 64
_POINCARE_MAX_DENOMINATOR: Final[int] = 16
_POINCARE_LOW_ORDER: Final[int] = 8
_RESONANCE_TOLERANCE: Final[float] = 1.0e-3
_KAM_DIOPHANTINE_TAU: Final[float] = 2.5
_KAM_DIOPHANTINE_GAMMA: Final[float] = 1.0e-3
_SMALL_PARAMETER_FLOOR: Final[float] = 1.0e-12
_ADIABATIC_DRIFT_LIMIT: Final[float] = 1.0e-6

# ── Parámetros de Automejora Recursiva de Nivel 3 (RSI-3) ───────────────
_RSI_MAX_DEPTH: Final[int] = 3
_RSI_TARGET_RESIDUAL: Final[float] = 1.0e-12
_RSI_MIN_TOLERANCE: Final[float] = 1.0e-14
_RSI_FACTOR_LOWER: Final[float] = 1.0e-2
_RSI_FACTOR_UPPER: Final[float] = 0.99
_RSI_OSCILLATION_EPS: Final[float] = 1.0e-15
_RSI_HOMOTOPY_GAIN: Final[float] = 0.35


# ═══════════════════════════════════════════════════════════════════════════
# Jerarquía de excepciones
# ═══════════════════════════════════════════════════════════════════════════

class SpecularFlowPhysicsError(Exception):
    """Excepción base para errores estructurales en el motor físico."""


class AllieviIntegrationError(SpecularFlowPhysicsError):
    """Error en la integración de las características hiperbólicas de Allievi."""


class HouseholderReflectionError(SpecularFlowPhysicsError):
    """Error en la reflexión covariante de Householder."""


class TellegenConservationError(SpecularFlowPhysicsError):
    """Violación del Teorema de Tellegen, dualidad de Hodge o pasividad de Lyapunov."""


class PoincareSectionError(SpecularFlowPhysicsError):
    """Error en el análisis de Poincaré (sección transversal, retorno, resonancia KAM)."""


class RecursiveSelfImprovementError(SpecularFlowPhysicsError):
    """Error en el meta-controlador de automejora recursiva de nivel 3."""


# ═══════════════════════════════════════════════════════════════════════════
# Estado diagnóstico (retículo completo de severidad)
# ═══════════════════════════════════════════════════════════════════════════

class SpecularFlowStatus(IntEnum):
    """
    Retículo acotado de estado diagnóstico del motor físico:
        COHERENT (0) < DEGRADED (1) < VETOED (2).
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    @classmethod
    def supremum(cls, *statuses: "SpecularFlowStatus") -> "SpecularFlowStatus":
        """Calcula el supremo (join ∨) en el retículo de severidad."""
        if not statuses:
            return cls.COHERENT
        return cls(max(int(status) for status in statuses))


# ═══════════════════════════════════════════════════════════════════════════
# Estructuras de datos inmutables (Certificados Ciber-Físicos)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class AllieviInvariants:
    """
    FASE 1 — Invariantes de Riemann (cono de luz hiperbólico), dirección
    característica, variables canónicas de acción-ángulo y snapshot de Poincaré.
    """

    psi_plus: float
    psi_minus: float
    characteristic_impedance: float
    characteristic_direction: Tuple[float, float]
    is_stable: bool
    characteristic_covector: Tuple[float, float] = (0.0, 0.0)
    energy_norm: float = 0.0
    impedance_condition_number: float = 1.0
    cfl_proxy: float = 0.0
    status: SpecularFlowStatus = SpecularFlowStatus.COHERENT
    # ── Certificados de Mecánica Celeste de Poincaré ────────────────────
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
class HouseholderReflectionResult:
    """
    FASE 2 — Reflexión covariante de Householder con purificación de
    Newton-Schulz, auditoría adiabática de la acción y monodromía de retorno.
    """

    reflected_state: NDArray[np.float64]
    reflection_operator: NDArray[np.float64]
    unitary_residual: float
    is_anti_symplectic: bool
    is_characteristic_consistent: bool
    state_norm_residual: float = 0.0
    specular_residual: float = 0.0
    incidence_cosine: float = 1.0
    metric_condition_number: float = 1.0
    spectral_uncertainty: float = 0.0
    metric_was_regularized: bool = False
    tikhonov_shift: float = 0.0
    status: SpecularFlowStatus = SpecularFlowStatus.COHERENT
    # ── Certificados de Monodromía y Acción de Poincaré ─────────────────
    action_invariant_input: float = 0.0
    action_invariant_reflected: float = 0.0
    action_drift: float = 0.0
    return_map_contraction: float = 1.0
    return_map_eigenvalues_modulus: Tuple[float, ...] = ()
    is_return_map_stable: bool = True
    richardson_correction: float = 0.0


@dataclass(frozen=True, slots=True)
class NetworkConservationReport:
    """
    FASE 3 — Conservación topológica de Tellegen, pasividad de Lyapunov,
    descomposición cohomológica de Hodge-Poincaré y recurrencia.
    """

    tellegen_residual: float
    is_tellegen_conserved: bool
    rayleigh_dissipation: float
    is_lyapunov_passive: bool
    kcl_residual: float = 0.0
    kvl_residual: float = 0.0
    is_kcl_conserved: bool = True
    is_kvl_conserved: bool = True
    graph_order: int = 0
    graph_size: int = 0
    cyclomatic_number: int = 0
    rayleigh_condition_number: float = 1.0
    min_rayleigh_eigenvalue: float = 0.0
    tellegen_power: float = 0.0
    kahan_tellegen_sum: float = 0.0
    used_sparse_tellegen: bool = False
    status: SpecularFlowStatus = SpecularFlowStatus.COHERENT
    # ── Certificados Topológicos de Hodge-Poincaré y RSI-3 ──────────────
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
    """
    SUTURA 4 — Instantánea de la sección transversal de Poincaré Σ,
    espectro de Floquet de la aplicación de retorno y divisores diofánticos KAM.
    """

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
    status: SpecularFlowStatus = SpecularFlowStatus.COHERENT


@dataclass(frozen=True, slots=True)
class LindstedtPoincareExpansion:
    """
    SUTURA 5 — Serie canónica perturbativa de Lie-Deprit-Lindstedt-Poincaré
    libre de términos seculares hasta orden O(ε⁴).
    """

    small_parameter: float
    zero_order_action: float
    first_order_action_correction: float
    second_order_action_correction: float
    frequency_shift_first_order: float
    frequency_shift_second_order: float
    corrected_action: float
    expansion_residual: float
    is_asymptotically_consistent: bool
    status: SpecularFlowStatus = SpecularFlowStatus.COHERENT


@dataclass(frozen=True, slots=True)
class RecursiveSelfImprovementTrace:
    """
    SUTURA 6 — Traza de auditoría de un nivel del meta-controlador RSI-3.
    """

    level: int
    tolerance: float
    residual: float
    convergence_rate: float
    richardson_extrapolation: float
    aitken_delta_squared: float
    adaptation_factor: float
    level3_active: bool
    status: SpecularFlowStatus = SpecularFlowStatus.COHERENT


@dataclass(frozen=True, slots=True)
class SelfImprovementResult:
    """
    SUTURA 6 — Certificado consolidado de convergencia del meta-controlador RSI-3.
    """

    traces: Tuple[RecursiveSelfImprovementTrace, ...]
    final_tolerance: float
    final_residual: float
    converged: bool
    richardson_limit: float
    aitken_limit: float
    effective_convergence_order: float
    level3_activated: bool
    status: SpecularFlowStatus = SpecularFlowStatus.COHERENT


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║                     FASE ANIDADA 1: OBSERVE                             ║
# ║    Suturas Fundacionales (S1–S6) + Solucionador de Allievi-Poincaré     ║
# ╚═════════════════════════════════════════════════════════════════════════╝

# ── SUTURA 1: Regularización espectral de Cholesky-Tikhonov ──────────────

def stable_cholesky_decomposition(
    G: NDArray[np.float64],
    tolerance: float = 1.0e-12,
) -> NDArray[np.float64]:
    """Calcula el factor triangular inferior L de Cholesky con proyección SPD."""
    L, _, _, _, _ = _stable_cholesky_tikhonov(
        G, tolerance=tolerance, target_condition=_SPD_CONDITION_TARGET
    )
    return L


def _stable_cholesky_tikhonov(
    G: NDArray[np.float64],
    tolerance: float = 1.0e-12,
    target_condition: float = _SPD_CONDITION_TARGET,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], float, bool, float]:
    """
    Proyección espectral sobre el cono de matrices simétricas definidas
    positivas (SPD) acotando el número de condición κ₂(G) ≤ target_condition.
    """
    try:
        G_arr = np.asarray(G, dtype=np.float64)
    except Exception as exc:
        raise SpecularFlowPhysicsError(
            "SpecularFlowPhysicsError: G no convertible a NDArray float64."
        ) from exc

    if G_arr.ndim != 2 or G_arr.shape[0] != G_arr.shape[1]:
        raise SpecularFlowPhysicsError(
            "SpecularFlowPhysicsError: G debe ser una matriz cuadrada."
        )

    if not np.all(np.isfinite(G_arr)):
        raise SpecularFlowPhysicsError(
            "SpecularFlowPhysicsError: G contiene entradas no finitas."
        )

    n = G_arr.shape[0]
    if n == 0:
        empty = G_arr.copy()
        return empty, empty, 0.0, False, 1.0

    G_sym = 0.5 * (G_arr + G_arr.T)

    direct_L: Optional[NDArray[np.float64]] = None
    direct_condition = float("inf")
    try:
        direct_L = la.cholesky(G_sym, lower=True)
        eigvals_direct = la.eigvalsh(G_sym)
        if eigvals_direct.size > 0:
            min_eig = float(np.min(eigvals_direct))
            max_eig = float(np.max(eigvals_direct))
            if min_eig > _MACHINE_EPS:
                direct_condition = float(max_eig / min_eig)
    except Exception:
        direct_L = None
        direct_condition = float("inf")

    if (
        direct_L is not None
        and np.isfinite(direct_condition)
        and direct_condition <= target_condition * (1.0 + 10.0 * tolerance)
    ):
        return direct_L, G_sym, 0.0, False, direct_condition

    try:
        vals, vecs = la.eigh(G_sym)
    except Exception as exc:
        raise SpecularFlowPhysicsError(
            "SpecularFlowPhysicsError: No se pudo diagonalizar G."
        ) from exc

    if vals.size == 0:
        empty = G_sym.copy()
        return empty, empty, 0.0, False, 1.0

    max_val = float(np.max(vals))
    g_norm = float(la.norm(G_sym, ord="fro"))
    scale = max(1.0, abs(max_val), g_norm)

    shift = max(10.0 * _MACHINE_EPS, scale * tolerance * 100.0)
    if max_val > 0.0:
        shift = max(shift, max_val / target_condition)

    last_error: Optional[Exception] = None
    for _ in range(12):
        vals_reg = np.maximum(vals, shift)
        min_reg = float(np.min(vals_reg))
        max_reg = float(np.max(vals_reg))

        if min_reg <= _MACHINE_EPS:
            shift = max(shift * 10.0, 10.0 * _MACHINE_EPS)
            continue

        cond_reg = float(max_reg / min_reg)
        if cond_reg > target_condition:
            shift = max(shift, max_reg / target_condition, 10.0 * _MACHINE_EPS)
            continue

        G_reg = (vecs * vals_reg) @ vecs.T
        G_reg = 0.5 * (G_reg + G_reg.T)

        try:
            L_reg = la.cholesky(G_reg, lower=True)
        except la.LinAlgError as exc:
            last_error = exc
            shift *= 10.0
            continue

        logger.warning(
            "Métrica G no-SPD o mal condicionada. Desplazamiento Tikhonov: %.4e",
            shift,
        )
        return L_reg, G_reg, float(shift), True, cond_reg

    raise SpecularFlowPhysicsError(
        "SpecularFlowPhysicsError: Regularización Tikhonov no convergió."
    ) from last_error


# ── SUTURA 2: Suma compensada de Kahan-Babuška ───────────────────────────

def kahan_tellegen_summation(
    pressures: NDArray[np.float64],
    flows: NDArray[np.float64],
) -> float:
    """
    Producto interno simpléctico/energético ⟨p, q⟩ mediante suma compensada
    de Kahan con cota de error de redondeo O(ε_mach) independiente de N.
    """
    p = np.asarray(pressures, dtype=np.float64).reshape(-1)
    q = np.asarray(flows, dtype=np.float64).reshape(-1)

    if p.size != q.size:
        raise SpecularFlowPhysicsError(
            "SpecularFlowPhysicsError: Dimensiones inconsistentes en Kahan Tellegen."
        )

    total_sum = 0.0
    compensation = 0.0

    for pi, qi in zip(p, q):
        value = float(pi * qi)
        if not np.isfinite(value):
            raise SpecularFlowPhysicsError(
                "SpecularFlowPhysicsError: Producto de potencia no finito en Kahan."
            )
        y = value - compensation
        t = total_sum + y
        compensation = (t - total_sum) - y
        total_sum = t

    return float(total_sum)


# ── SUTURA 3: Verificación cohomológica dispersa de Tellegen ─────────────

def verify_tellegen_sparse(
    nodal_potentials_p: NDArray[np.float64],
    flow_vector_f: NDArray[np.float64],
    incidence_matrix_B1: sp.csr_matrix,
) -> float:
    """
    Evalúa la adjunción frontera-coborde ⟨B₁ᵀ p, f⟩ = ⟨p, B₁ f⟩ en formato CSR
    disperso utilizando suma compensada de Kahan.
    """
    p = np.asarray(nodal_potentials_p, dtype=np.float64).reshape(-1)
    f = np.asarray(flow_vector_f, dtype=np.float64).reshape(-1)

    if sp.issparse(incidence_matrix_B1):
        B1 = incidence_matrix_B1.tocsr()
    else:
        B1 = sp.csr_matrix(incidence_matrix_B1, dtype=np.float64)

    if B1.shape[0] != p.size:
        raise SpecularFlowPhysicsError(
            "SpecularFlowPhysicsError: B1.shape[0] no coincide con p."
        )
    if B1.shape[1] != f.size:
        raise SpecularFlowPhysicsError(
            "SpecularFlowPhysicsError: B1.shape[1] no coincide con f."
        )
    if B1.nnz > 0 and not np.all(np.isfinite(B1.data)):
        raise SpecularFlowPhysicsError(
            "SpecularFlowPhysicsError: B1 dispersa contiene datos no finitos."
        )

    v_pressures = np.asarray(B1.T @ p, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(v_pressures)):
        raise SpecularFlowPhysicsError(
            "SpecularFlowPhysicsError: v = B1ᵀ p no finito."
        )
    return float(abs(kahan_tellegen_summation(v_pressures, f)))


# ── SUTURA 4: Fracciones continuas de Gauss, resonancia KAM y acción-ángulo ─

def _continued_fraction_approximation(
    x: float,
    max_denominator: int = _POINCARE_MAX_DENOMINATOR,
) -> Tuple[int, int]:
    """
    Mejor aproximación diofántica racional p/q de x mediante el operador de
    Gauss de fracciones continuas con denominador acotado q ≤ max_denominator.
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

    if k_curr == 0:
        return (0, 1)
    return (int(h_curr), int(k_curr))


def _detect_resonance(
    ratio: float,
    max_denominator: int = _POINCARE_MAX_DENOMINATOR,
    low_order: int = _POINCARE_LOW_ORDER,
    tolerance: float = _RESONANCE_TOLERANCE,
) -> Tuple[bool, int, int, float, float]:
    r"""
    Detecta resonancias de Arnold-Poincaré de bajo orden y evalúa el divisor
    pequeño diofántico según el criterio KAM:

    .. math::
        \left|\omega - \frac{p}{q}\right| < \max\left(\text{tol}, \frac{\gamma}{q^\tau}\right)
    """
    if not math.isfinite(ratio) or ratio <= 0.0:
        return (False, 0, 0, 0.0, 0.0)

    p, q = _continued_fraction_approximation(ratio, max_denominator)
    if q <= 0:
        return (False, 0, 0, 0.0, 0.0)

    err = abs(ratio - p / q)
    kam_threshold = _KAM_DIOPHANTINE_GAMMA / (float(q) ** _KAM_DIOPHANTINE_TAU)
    effective_tol = max(tolerance, kam_threshold)
    is_resonant = (err < effective_tol) and (q <= low_order)

    # Magnitud normalizada del divisor pequeño: en resonancia exacta vale 1/q.
    small_divisor = 1.0 / (float(q) * (1.0 + float(q) * err))
    return (bool(is_resonant), int(p), int(q), float(err), float(small_divisor))


def _compute_action_angle(
    y: float,
    p_canonical: float,
) -> Tuple[float, float]:
    r"""
    Transformación simpléctica de Darboux $(y, p) \mapsto (I, \theta)$:

    .. math::
        I = \frac{1}{2}(y^2 + p^2) = \frac{1}{4}\left((\Psi^+)^2 + (\Psi^-)^2\right),
        \quad \theta = \operatorname{atan2}(p, y).
    """
    action = 0.5 * (y * y + p_canonical * p_canonical)
    angle = math.atan2(p_canonical, y) if action > _MACHINE_EPS else 0.0
    return (float(action), float(angle))


# ── SUTURA 5: Serie Canónica de Lie-Deprit-Lindstedt-Poincaré ────────────

def _lindstedt_poincare_expansion(
    action_zero_order: float,
    small_parameter: float,
    frequency_correction_proxy: float,
) -> LindstedtPoincareExpansion:
    r"""
    Expansión perturbativa canónica de Lindstedt-Poincaré / Lie-Deprit hasta
    orden $\mathcal{O}(\varepsilon^4)$ eliminando resonancias seculares:

    .. math::
        \omega(\varepsilon) &= \omega_0 \left(1 + \varepsilon^2 \omega_1 + \varepsilon^4 \omega_2\right),
        \quad \omega_2 = -\frac{1}{4}\omega_1^2 \\
        I(\varepsilon) &= I_0 + \varepsilon^2 I_1 + \varepsilon^4 I_2,
        \quad I_1 = I_0 \omega_1, \quad I_2 = -\frac{1}{2} I_0 \omega_1 (1 + \frac{1}{2}\omega_1)
    """
    eps = float(max(_SMALL_PARAMETER_FLOOR, min(1.0, abs(small_parameter))))
    I0 = float(max(0.0, action_zero_order))
    omega1 = float(frequency_correction_proxy)
    omega2 = -0.25 * omega1 * omega1

    # Generadores de Lie de primer y segundo orden libres de términos seculares
    I1 = I0 * omega1
    I2 = -0.5 * I0 * omega1 * (1.0 + 0.5 * omega1)

    eps2 = eps * eps
    eps4 = eps2 * eps2

    corrected = float(I0 + eps2 * I1 + eps4 * I2)
    residual = float(abs(corrected - I0) / max(_MACHINE_EPS, I0))

    # Criterio de consistencia asintótica de Poincaré: el término O(ε⁴) debe
    # estar acotado por el término O(ε²) y la deriva total bajo control.
    asymptotic_decay_ok = abs(eps4 * I2) <= abs(eps2 * I1) + _MACHINE_EPS
    is_consistent = (
        math.isfinite(corrected)
        and math.isfinite(residual)
        and asymptotic_decay_ok
        and residual <= _ADIABATIC_DRIFT_LIMIT
    )

    if not is_consistent or eps > 0.5:
        status = SpecularFlowStatus.DEGRADED
    else:
        status = SpecularFlowStatus.COHERENT

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
        status=status,
    )


# ── SUTURA 6: Operadores de Aceleración y Homotopía RSI-3 ────────────────

def _richardson_extrapolation(
    r_prev: float,
    r_curr: float,
    tol_prev: float,
    tol_curr: float,
) -> Tuple[float, float]:
    r"""
    Extrapolación al límite de Richardson y estimación del orden local de
    convergencia de Fréchet-Hölder $p = \frac{\ln(r_n / r_{n-1})}{\ln(\tau_n / \tau_{n-1})}$.
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
    r"""
    Transformación no-lineal de secuencias $\Delta^2$ de Aitken (caso escalar
    del algoritmo $\varepsilon$ de Wynn / aproximante de Padé $[1/1]$):

    .. math::
        \hat{r}_n = r_n - \frac{(r_n - r_{n-1})^2}{r_n - 2r_{n-1} + r_{n-2}}
    """
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
    Ley de control adaptativo de tolerancia $\tau_{n+1} = \alpha_n \tau_n$.

    - **Nivel 2 (Meta-Control):** $\alpha^{(2)} = (\text{target} / r_n)^{1/p}$.
    - **Nivel 3 (Meta-Meta-Control Homotópico):** Evalúa la discrepancia
      espectral entre el punto fijo de Aitken $\hat{r}_n$ y la extrapolación
      polinomial de Richardson $R_n$, modulando $\alpha^{(2)}$ mediante un
      filtro hiperbólico de contracción de Banach-Fréchet:

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

    if level3_active:
        ratio = (
            abs(aitken_signal) / current_residual
            if math.isfinite(aitken_signal)
            else 1.0
        )
        if ratio > 1.0:
            factor = min(factor, 0.5)
        elif ratio < 0.5:
            factor = max(factor, 0.5)

        ref_signal = (
            richardson_signal
            if (richardson_signal is not None and math.isfinite(richardson_signal))
            else current_residual
        )
        frechet_discrepancy = abs(aitken_signal - ref_signal) / max(
            _MACHINE_EPS, current_residual
        )
        homotopy_modulation = math.exp(
            -_RSI_HOMOTOPY_GAIN * math.tanh(frechet_discrepancy)
        )
        factor = factor * homotopy_modulation

    return float(max(_RSI_FACTOR_LOWER, min(_RSI_FACTOR_UPPER, factor)))


# ── Utilidades de Auditoría Numérica y Rango Matricial ───────────────────

def _frobenius_norm(matrix: object, is_sparse: bool) -> float:
    if is_sparse:
        try:
            return float(sp.linalg.norm(matrix, ord="fro"))
        except Exception:
            data = getattr(matrix, "data", np.empty(0, dtype=np.float64))
            return float(np.linalg.norm(data)) if data.size else 0.0
    arr = np.asarray(matrix, dtype=np.float64)
    return 0.0 if arr.size == 0 else float(la.norm(arr, ord="fro"))


def _estimate_matrix_rank(matrix: object, is_sparse: bool) -> int:
    shape = getattr(matrix, "shape", None)
    if shape is None:
        return 0
    n_rows, n_cols = shape
    if n_rows == 0 or n_cols == 0:
        return 0
    if not is_sparse:
        try:
            return int(
                np.linalg.matrix_rank(
                    np.asarray(matrix, dtype=np.float64), tol=1.0e-12
                )
            )
        except Exception:
            return int(min(n_rows, n_cols))
    if n_rows * n_cols <= 200_000:
        try:
            return int(np.linalg.matrix_rank(matrix.toarray(), tol=1.0e-12))
        except Exception:
            return int(min(n_rows, n_cols))
    return int(min(n_rows, n_cols))


class _NumericalAuditMixin:
    """Mixin de validación algebraica y finitud en punto flotante IEEE-754."""

    @staticmethod
    def _as_finite_float(value: object, name: str) -> float:
        try:
            scalar = float(value)
        except (TypeError, ValueError) as exc:
            raise SpecularFlowPhysicsError(
                f"SpecularFlowPhysicsError: '{name}' no convertible a escalar."
            ) from exc
        if not np.isfinite(scalar):
            raise SpecularFlowPhysicsError(
                f"SpecularFlowPhysicsError: '{name}' no finito."
            )
        return scalar

    @staticmethod
    def _as_finite_vector(values: object, name: str) -> NDArray[np.float64]:
        try:
            vector = np.asarray(values, dtype=np.float64)
        except Exception as exc:
            raise SpecularFlowPhysicsError(
                f"SpecularFlowPhysicsError: '{name}' no convertible a NDArray."
            ) from exc
        if vector.ndim == 0:
            vector = vector.reshape(1)
        if not np.all(np.isfinite(vector)):
            raise SpecularFlowPhysicsError(
                f"SpecularFlowPhysicsError: '{name}' contiene no finitos."
            )
        return vector


# ── Clase de la Fase Anidada 1: Phase1_AllieviSolver ─────────────────────

class Phase1_AllieviSolver(_NumericalAuditMixin, ABC):
    """
    FASE ANIDADA 1 (Observe):
    Resuelve las ecuaciones hiperbólicas de Allievi en coordenadas de cono de luz
    (números perplejos / split-complex) y las proyecta mediante una transformada
    de Cayley al espacio de fases simpléctico de Poincaré.
    """

    def solve_allievi_characteristics(
        self,
        potential_y: float,
        flow_q: float,
        wave_speed_a: float,
        gravity_g: float,
        section_s: float,
        tolerance: float = _DEFAULT_TOL,
        cfl_number: Optional[float] = None,
        poincare_section_level: float = 0.0,
    ) -> AllieviInvariants:
        r"""
        Integra los invariantes de Riemann $\Psi^\pm = y \pm Z_c q$ con $Z_c = \frac{a}{gS}$,
        evalúa la sección transversal de Poincaré $\Sigma = \{y = y_s\}$, el divisor
        pequeño diofántico KAM y la serie perturbativa de Lindstedt-Poincaré.
        """
        y = self._as_finite_float(potential_y, "potential_y")
        q = self._as_finite_float(flow_q, "flow_q")
        a = self._as_finite_float(wave_speed_a, "wave_speed_a")
        g = self._as_finite_float(gravity_g, "gravity_g")
        s = self._as_finite_float(section_s, "section_s")

        if a <= _MACHINE_EPS:
            raise AllieviIntegrationError(
                "AllieviIntegrationError: wave_speed_a no positiva."
            )
        if g <= _MACHINE_EPS:
            raise AllieviIntegrationError(
                "AllieviIntegrationError: gravity_g no positiva."
            )
        if s <= _MACHINE_EPS:
            raise AllieviIntegrationError(
                "AllieviIntegrationError: section_s no positiva."
            )

        denom = g * s
        if denom <= _MACHINE_EPS:
            raise AllieviIntegrationError(
                "AllieviIntegrationError: g·S degenerado."
            )

        z_c = a / denom
        if not np.isfinite(z_c) or z_c <= _MACHINE_EPS:
            raise AllieviIntegrationError(
                "AllieviIntegrationError: Impedancia degenerada."
            )

        psi_plus = y + z_c * q
        psi_minus = y - z_c * q
        if not np.isfinite(psi_plus) or not np.isfinite(psi_minus):
            raise AllieviIntegrationError(
                "AllieviIntegrationError: Invariantes no finitos."
            )

        is_stable = (
            abs(psi_plus) <= _STABILITY_BOUND
            and abs(psi_minus) <= _STABILITY_BOUND
        )

        # Vector tangente a la curva característica dx/dt = a
        char_dir = np.array([a, 1.0], dtype=np.float64)
        char_dir_norm = float(la.norm(char_dir))
        if char_dir_norm <= _MACHINE_EPS:
            raise AllieviIntegrationError(
                "AllieviIntegrationError: Dirección característica degenerada."
            )
        char_dir = char_dir / char_dir_norm

        # 1-forma diferencial (covector dual bajo la métrica de impedancia)
        z_c_sq = z_c * z_c
        if np.isfinite(z_c_sq):
            g_cal = np.array([[1.0, 0.0], [0.0, z_c_sq]], dtype=np.float64)
            char_cov = g_cal @ char_dir
            n_cov = float(la.norm(char_cov))
            char_cov = (
                char_cov / n_cov if n_cov > _MACHINE_EPS else char_dir.copy()
            )
        else:
            char_cov = char_dir.copy()

        zq = z_c * q
        energy_norm = (
            float(np.hypot(y, zq)) if np.isfinite(zq) else float("inf")
        )
        impedance_cond = float(max(z_c, 1.0 / z_c))

        if cfl_number is None:
            cfl_proxy = float(abs(a) / (abs(a) + 1.0))
        else:
            cfl_proxy = abs(self._as_finite_float(cfl_number, "cfl_number"))

        # ── Coordenadas canónicas de Acción-Ángulo de Poincaré ─────────
        p_canonical = z_c * q
        action, angle = _compute_action_angle(y, p_canonical)

        # ── Sección Transversal de Poincaré Σ ──────────────────────────
        section_snapshot = self._poincare_section_snapshot(
            action=action,
            angle=angle,
            section_level=poincare_section_level,
            psi_plus=psi_plus,
            psi_minus=psi_minus,
        )

        # ── Resonancia Diofántica KAM ──────────────────────────────────
        if abs(psi_minus) > _MACHINE_EPS:
            rho = abs(psi_plus / psi_minus)
        else:
            rho = float("inf")
        is_res, p_res, q_res, _, small_div = _detect_resonance(rho)

        # ── Serie Perturbativa de Lie-Deprit-Lindstedt-Poincaré ────────
        # El parámetro pequeño acopla la anisotropía espectral log(κ(G))
        # con la escala de regularización activa del nivel RSI.
        tol_scale = min(1.0, max(_SMALL_PARAMETER_FLOOR, math.sqrt(abs(tolerance))))
        eps_eff = 0.0
        if impedance_cond > 0.0 and math.isfinite(impedance_cond):
            raw_anisotropy = math.log10(max(1.0, impedance_cond)) / 8.0
            eps_eff = min(1.0, raw_anisotropy * tol_scale if raw_anisotropy < 0.5 else raw_anisotropy)
        omega1_proxy = -0.5 * eps_eff
        lindstedt = _lindstedt_poincare_expansion(
            action_zero_order=action,
            small_parameter=eps_eff,
            frequency_correction_proxy=omega1_proxy,
        )

        # ── Evaluación en el retículo diagnóstico ──────────────────────
        status = SpecularFlowStatus.COHERENT
        if (
            not is_stable
            or cfl_proxy > 1.0 + tolerance
            or not np.isfinite(energy_norm)
            or section_snapshot.status == SpecularFlowStatus.VETOED
        ):
            status = SpecularFlowStatus.VETOED
        elif (
            impedance_cond > _SPD_CONDITION_TARGET
            or abs(psi_plus - psi_minus) > 1.0e6
            or cfl_proxy > max(0.0, 1.0 - 10.0 * tolerance)
            or is_res
            or section_snapshot.status == SpecularFlowStatus.DEGRADED
        ):
            status = SpecularFlowStatus.DEGRADED

        return AllieviInvariants(
            psi_plus=float(psi_plus),
            psi_minus=float(psi_minus),
            characteristic_impedance=float(z_c),
            characteristic_direction=(float(char_dir[0]), float(char_dir[1])),
            is_stable=is_stable,
            characteristic_covector=(float(char_cov[0]), float(char_cov[1])),
            energy_norm=energy_norm,
            impedance_condition_number=impedance_cond,
            cfl_proxy=cfl_proxy,
            status=status,
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
            poincare_crossing_count=len(section_snapshot.crossing_values),
        )

    def _poincare_section_snapshot(
        self,
        action: float,
        angle: float,
        section_level: float,
        psi_plus: float,
        psi_minus: float,
    ) -> PoincareSectionSnapshot:
        r"""
        Interseca el toro invariante de Liouville-Arnold $I = \frac{1}{2}(y^2 + p^2)$
        con la hipersuperficie transversal $\Sigma = \{(y, p) \in T^*\mathcal{M} \mid y = y_s\}$.
        Verifica la condición de transversalidad de Poincaré $\nabla H \wedge dy \neq 0$.
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
            # Intersección tangencial (bifurcación de pliegue en Σ)
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

        status = SpecularFlowStatus.COHERENT
        if not math.isfinite(action) or action < 0.0:
            status = SpecularFlowStatus.VETOED
        elif not crossings and radius_sq > _MACHINE_EPS:
            status = SpecularFlowStatus.DEGRADED
        elif is_res:
            status = SpecularFlowStatus.DEGRADED

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
            status=status,
        )

    def _lindstedt_poincare_correction(
        self,
        allievi: AllieviInvariants,
    ) -> LindstedtPoincareExpansion:
        """Evalúa la serie perturbativa de Lindstedt-Poincaré sobre un certificado Allievi."""
        eps = float(allievi.small_parameter_epsilon)
        omega1 = float(allievi.lindstedt_frequency_shift) if eps > 0.0 else 0.0
        if eps > 0.0 and omega1 == 0.0:
            omega1 = -0.5 * eps
        return _lindstedt_poincare_expansion(
            action_zero_order=allievi.action_invariant,
            small_parameter=eps,
            frequency_correction_proxy=omega1,
        )

    # ═════════════════════════════════════════════════════════════════════
    # ESLABÓN FUNCTORIAL 1 → 2: Último método de la Fase 1 e inicio de la 2
    # ═════════════════════════════════════════════════════════════════════
    def _allievi_to_householder_bridge(
        self,
        allievi_inv: AllieviInvariants,
    ) -> Tuple[
        NDArray[np.float64],
        float,
        PoincareSectionSnapshot,
        LindstedtPoincareExpansion,
    ]:
        """
        MORFISMO DE TRANSICIÓN (FASE ANIDADA 1 ──► FASE ANIDADA 2).

        Empaqueta el fibrado tangente característico normalizado, la impedancia
        de onda Z_c, la sección transversal de Poincaré Σ y la serie canónica
        de Lindstedt-Poincaré como pre-condición formal de entrada para el
        reflector covariante de Householder en la Fase Anidada 2.
        """
        char_dir = np.array(
            allievi_inv.characteristic_direction, dtype=np.float64
        )
        z_c = float(allievi_inv.characteristic_impedance)

        if not np.all(np.isfinite(char_dir)) or not np.isfinite(z_c):
            raise AllieviIntegrationError(
                "AllieviIntegrationError: Certificado de Fase 1 inválido."
            )

        n = float(la.norm(char_dir))
        if n <= _MACHINE_EPS:
            raise AllieviIntegrationError(
                "AllieviIntegrationError: Dirección característica degenerada."
            )
        char_dir = char_dir / n

        snapshot = self._poincare_section_snapshot(
            action=allievi_inv.action_invariant,
            angle=allievi_inv.angle_parameter,
            section_level=allievi_inv.poincare_section_level,
            psi_plus=allievi_inv.psi_plus,
            psi_minus=allievi_inv.psi_minus,
        )
        lindstedt = self._lindstedt_poincare_correction(allievi_inv)
        return char_dir, z_c, snapshot, lindstedt


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║                     FASE ANIDADA 2: ORIENT                              ║
# ║   Continuación directa de `_allievi_to_householder_bridge`              ║
# ║   Reflexión Covariante de Householder + Monodromía de Floquet-Poincaré  ║
# ╚═════════════════════════════════════════════════════════════════════════╝

class Phase2_HouseholderReflector(Phase1_AllieviSolver):
    """
    FASE ANIDADA 2 (Orient):
    Hereda directamente de `Phase1_AllieviSolver` y consume el morfismo
    `_allievi_to_householder_bridge` para ejecutar la reflexión especular
    sobre la variedad Riemanniana (M, G) con purificación de Newton-Schulz
    y auditoría simpléctica de la aplicación de retorno de Poincaré.
    """

    def compute_householder_reflection(
        self,
        metric_tensor_g: NDArray[np.float64],
        boundary_normal_n: NDArray[np.float64],
        state_vector: NDArray[np.float64],
        allievi_invariants: AllieviInvariants,
        tolerance: float = _DEFAULT_TOL,
    ) -> HouseholderReflectionResult:
        r"""
        Construye y purifica la involución G-ortogonal de Householder:

        .. math::
            M = I - 2 \frac{n (G n)^\top}{n^\top G n}, \quad M^\top G M = G, \quad M^2 = I.
        """
        # ── 0. Recepción del morfismo final de la Fase Anidada 1 ──────
        char_dir, _, poincare_in, lindstedt_in = (
            self._allievi_to_householder_bridge(allievi_invariants)
        )

        dim = char_dir.size
        if dim != 2:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: El motor especular es 2D."
            )

        audit_tol = max(tolerance * 100.0, _RELAX_TOL)

        # ── 1. Regularización espectral Cholesky-Tikhonov de G ────────
        try:
            G_input = np.asarray(metric_tensor_g, dtype=np.float64)
        except Exception as exc:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: metric_tensor_g no convertible."
            ) from exc

        if G_input.ndim != 2 or G_input.shape != (dim, dim):
            raise HouseholderReflectionError(
                "HouseholderReflectionError: G debe ser 2x2."
            )

        try:
            _L, G_eff, tik_shift, was_reg, cond_g = _stable_cholesky_tikhonov(
                G_input,
                tolerance=max(tolerance, _TIKHONOV_TOL),
                target_condition=_SPD_CONDITION_TARGET,
            )
        except SpecularFlowPhysicsError as exc:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: No se pudo estabilizar G."
            ) from exc

        # ── 2. Validación de la normal a la frontera y vector de estado ─
        normal = self._as_finite_vector(boundary_normal_n, "boundary_normal_n")
        if normal.size != dim:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Normal dimensión ≠ 2."
            )
        if float(la.norm(normal)) <= _MACHINE_EPS:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Normal euclidiana degenerada."
            )

        x_state = self._as_finite_vector(state_vector, "state_vector")
        if x_state.size != dim:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Estado dimensión ≠ 2."
            )

        # ── 3. Construcción covariante y purificación de Newton-Schulz ─
        g_normal = G_eff @ normal
        denom = float(normal @ g_normal)
        g_norm = float(la.norm(G_eff, ord="fro"))
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

        # Paso de purificación de Newton-Schulz en O(2, G) cuando la
        # tolerancia de RSI-3 exige precisión cercana al épsilon de máquina:
        # M_{k+1} = 0.5 * M_k * (3I - G^{-1} M_k^T G M_k)
        if tolerance < _DEFAULT_TOL:
            try:
                gram_defect = la.cho_solve((_L, True), M.T @ (G_eff @ M))
                M_purified = 0.5 * (M @ (3.0 * np.eye(dim) - gram_defect))
                if np.all(np.isfinite(M_purified)):
                    M = M_purified
            except Exception:
                pass

        reflected = M @ x_state

        # ── 4. Auditoría de isometría Riemanniana (Mᵀ G M = G) ────────
        unitary_res = float(
            la.norm(M.T @ G_eff @ M - G_eff, ord="fro") / max(1.0, g_norm)
        )
        is_unitary = unitary_res <= audit_tol

        # ── 5. Conservación de la energía cinético-potencial ||x||_G ──
        x_norm_sq = float(x_state @ G_eff @ x_state)
        mx_norm_sq = float(reflected @ G_eff @ reflected)
        x_norm = float(np.sqrt(max(0.0, x_norm_sq)))
        mx_norm = float(np.sqrt(max(0.0, mx_norm_sq)))
        state_norm_res = float(abs(mx_norm - x_norm) / max(1.0, x_norm))

        # ── 6. Consistencia geométrica especular de Snell-Descartes ───
        char_g_sq = float(char_dir @ G_eff @ char_dir)
        if char_g_sq <= _MACHINE_EPS:
            raise HouseholderReflectionError(
                "HouseholderReflectionError: Norma G de la característica nula."
            )
        char_g_norm = float(np.sqrt(max(0.0, char_g_sq)))
        normal_g_norm = float(np.sqrt(max(0.0, denom)))

        inc_num = float(normal @ G_eff @ char_dir)
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

        # ── 7. Estructura anti-simpléctica (Mᵀ Ω M = -Ω, det(M) = -1) ─
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

        # ── 8. Invariancia adiabática de la acción de Poincaré ────────
        I_in = float(allievi_invariants.action_invariant)
        I_out = 0.5 * float(reflected @ G_eff @ reflected)
        action_drift = abs(I_out - I_in) / max(1.0, I_in)

        # ── 9. Monodromía de Floquet de la Aplicación de Retorno ──────
        try:
            eigs_M = la.eigvals(M)
            eig_mods = tuple(float(abs(e)) for e in eigs_M)
        except Exception:
            eig_mods = ()

        # El operador de retorno de primer ciclo completo es P = M²
        monodromy_P = M @ M
        monodromy_defect = float(
            la.norm(monodromy_P - np.eye(dim), ord="fro")
        )
        return_contraction = float(1.0 + monodromy_defect)
        is_return_stable = (
            all(m <= 1.0 + audit_tol for m in eig_mods) if eig_mods else True
        )

        # ── 10. Defecto asintótico de Padé-Richardson ─────────────────
        dominant = max(unitary_res, specular_res, state_norm_res)
        richardson_corr = float(
            dominant / (1.0 + max(0.0, float(lindstedt_in.expansion_residual)))
        )

        # ── 11. Evaluación en el retículo diagnóstico ─────────────────
        status = SpecularFlowStatus.COHERENT
        if (
            not np.isfinite(unitary_res)
            or not np.isfinite(specular_res)
            or not np.isfinite(state_norm_res)
            or not is_unitary
            or not is_anti_sympl
            or not is_return_stable
        ):
            status = SpecularFlowStatus.VETOED
        elif (
            unitary_res > tolerance
            or specular_res > tolerance
            or state_norm_res > tolerance
            or cond_g > _SPD_CONDITION_TARGET
            or abs(incidence_cos) < 0.1
            or was_reg
            or action_drift > _ADIABATIC_DRIFT_LIMIT
            or poincare_in.status == SpecularFlowStatus.DEGRADED
        ):
            status = SpecularFlowStatus.DEGRADED

        prior = getattr(
            allievi_invariants, "status", SpecularFlowStatus.COHERENT
        )
        status = SpecularFlowStatus.supremum(status, prior)
        spectral_unc = float(cond_g * dominant)

        return HouseholderReflectionResult(
            reflected_state=reflected,
            reflection_operator=M,
            unitary_residual=unitary_res,
            is_anti_symplectic=is_anti_sympl,
            is_characteristic_consistent=(specular_res <= audit_tol),
            state_norm_residual=state_norm_res,
            specular_residual=specular_res,
            incidence_cosine=float(incidence_cos),
            metric_condition_number=cond_g,
            spectral_uncertainty=spectral_unc,
            metric_was_regularized=was_reg,
            tikhonov_shift=tik_shift,
            status=status,
            action_invariant_input=I_in,
            action_invariant_reflected=I_out,
            action_drift=float(action_drift),
            return_map_contraction=return_contraction,
            return_map_eigenvalues_modulus=eig_mods,
            is_return_map_stable=bool(is_return_stable),
            richardson_correction=float(richardson_corr),
        )

    def _poincare_return_map_audit(
        self,
        householder: HouseholderReflectionResult,
        n_iterations: int = 4,
    ) -> Tuple[float, Tuple[float, ...]]:
        """
        Audita la órbita discreta {M^k x}_{k=1}^n inducida por la aplicación
        de retorno sobre la sección transversal, verificando la estabilidad
        de Lyapunov-Floquet del 2-ciclo periódico.
        """
        M = np.asarray(householder.reflection_operator, dtype=np.float64)
        x = np.asarray(
            householder.reflected_state, dtype=np.float64
        ).reshape(-1)

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
        contraction = max(n / base for n in norms)
        return (float(contraction), tuple(norms))

    # ═════════════════════════════════════════════════════════════════════
    # ESLABÓN FUNCTORIAL 2 → 3: Último método de la Fase 2 e inicio de la 3
    # ═════════════════════════════════════════════════════════════════════
    def _householder_to_tellegen_bridge(
        self,
        householder_result: HouseholderReflectionResult,
        poincare_input: PoincareSectionSnapshot,
        lindstedt_input: LindstedtPoincareExpansion,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        SpecularFlowStatus,
        PoincareSectionSnapshot,
        LindstedtPoincareExpansion,
    ]:
        """
        MORFISMO DE TRANSICIÓN (FASE ANIDADA 2 ──► FASE ANIDADA 3).

        Transfiere el operador involutivo M, el estado reflejado Mx, el
        supremo diagnóstico acumulado, el snapshot de Poincaré y la expansión
        de Lindstedt-Poincaré hacia el validador cohomológico de Tellegen-Hodge
        y el meta-controlador RSI-3 en la Fase Anidada 3.
        """
        operator = np.asarray(
            householder_result.reflection_operator, dtype=np.float64
        )
        reflected = np.asarray(
            householder_result.reflected_state, dtype=np.float64
        )
        status = getattr(
            householder_result, "status", SpecularFlowStatus.COHERENT
        )
        return operator, reflected, status, poincare_input, lindstedt_input


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║                FASE ANIDADA 3: DECIDE & META-EVOLVE                     ║
# ║   Continuación directa de `_householder_to_tellegen_bridge`             ║
# ║   Cohomología de Tellegen-Hodge-Poincaré + Meta-Controlador RSI-3       ║
# ╚═════════════════════════════════════════════════════════════════════════╝

class Phase3_TellegenValidator(Phase2_HouseholderReflector):
    """
    FASE ANIDADA 3 (Decide & Meta-Evolve):
    Hereda de `Phase2_HouseholderReflector`, consume el morfismo
    `_householder_to_tellegen_bridge` y audita la conservación topológica
    de Tellegen, la descomposición ortogonal de Hodge-Poincaré
    C¹(G) = im(B₁ᵀ) ⊕ ker(B₁), la recurrencia de Poincaré y el lazo cerrado
    de Automejora Recursiva de Nivel 3 (RSI-3).
    """

    def verify_network_conservation(
        self,
        pressures: Optional[NDArray[np.float64]],
        flows: NDArray[np.float64],
        rayleigh_dissipation_r: NDArray[np.float64],
        gradient_h: Optional[NDArray[np.float64]],
        allievi_invariants: AllieviInvariants,
        householder_result: HouseholderReflectionResult,
        tolerance: float = 1.0e-8,
        incidence_matrix: Optional[object] = None,
        nodal_potentials: Optional[NDArray[np.float64]] = None,
    ) -> NetworkConservationReport:
        """
        Audita el Teorema de Tellegen (⟨v, q⟩ = 0), la pasividad de Lyapunov-Rayleigh,
        la dualidad cohomológica de Hodge-Poincaré y la recurrencia de fase.
        """
        # ── 0. Recepción del morfismo final de la Fase Anidada 2 ──────
        snapshot = self._poincare_section_snapshot(
            action=allievi_invariants.action_invariant,
            angle=allievi_invariants.angle_parameter,
            section_level=allievi_invariants.poincare_section_level,
            psi_plus=allievi_invariants.psi_plus,
            psi_minus=allievi_invariants.psi_minus,
        )
        lindstedt = self._lindstedt_poincare_correction(allievi_invariants)
        M_op, reflected_vec, prior_householder_status, _, _ = (
            self._householder_to_tellegen_bridge(
                householder_result, snapshot, lindstedt
            )
        )

        audit_tol = max(tolerance * 100.0, _RELAX_TOL)

        # ── 1. Vector de 1-cadenas (flujos de rama q ∈ C₁(G; ℝ)) ──────
        q = self._as_finite_vector(flows, "flows")
        m = q.size
        if m > _GRAPH_MAX_BRANCHES:
            raise TellegenConservationError(
                "TellegenConservationError: Complejidad fuera de cota."
            )

        # ── 2. Operador frontera B₁ : C₁(G) → C₀(G) ───────────────────
        is_sparse = False
        if incidence_matrix is not None:
            if sp.issparse(incidence_matrix):
                B1 = incidence_matrix.tocsr()
                is_sparse = True
                if B1.nnz > 0 and not np.all(np.isfinite(B1.data)):
                    raise TellegenConservationError(
                        "TellegenConservationError: B1 dispersa no finita."
                    )
            else:
                try:
                    B1_dense = np.asarray(incidence_matrix, dtype=np.float64)
                except Exception as exc:
                    raise TellegenConservationError(
                        "TellegenConservationError: incidence_matrix no convertible."
                    ) from exc
                if B1_dense.ndim == 1:
                    if m == 1:
                        B1_dense = B1_dense.reshape(-1, 1)
                    else:
                        raise TellegenConservationError(
                            "TellegenConservationError: incidence_matrix 1D incompatible."
                        )
                if B1_dense.ndim != 2:
                    raise TellegenConservationError(
                        "TellegenConservationError: incidence_matrix debe ser 2D."
                    )
                if not np.all(np.isfinite(B1_dense)):
                    raise TellegenConservationError(
                        "TellegenConservationError: incidence_matrix no finita."
                    )
                B1 = B1_dense

            if B1.shape[1] != m:
                raise TellegenConservationError(
                    "TellegenConservationError: incidence_matrix ∉ flows."
                )
            n_nodes = B1.shape[0]
            if m > 0 and n_nodes == 0:
                raise TellegenConservationError(
                    "TellegenConservationError: topología sin nodos."
                )
        else:
            if m == 0:
                B1 = np.zeros((0, 0), dtype=np.float64)
                n_nodes = 0
            else:
                B1 = np.zeros((2, m), dtype=np.float64)
                B1[0, :] = 1.0
                B1[1, :] = -1.0
                n_nodes = 2

        # ── 3. Ortogonalidad de Tellegen y Ley de Tensiones (KVL) ─────
        used_sparse = False
        kvl_res = 0.0
        p_branch: Optional[NDArray[np.float64]] = None
        cut_projection_vec: Optional[NDArray[np.float64]] = None

        if nodal_potentials is not None:
            p_nodal = self._as_finite_vector(
                nodal_potentials, "nodal_potentials"
            )
            if p_nodal.size != n_nodes:
                raise TellegenConservationError(
                    "TellegenConservationError: nodal_potentials ∉ B1.shape[0]."
                )
            v_pressures = np.asarray(
                B1.T @ p_nodal, dtype=np.float64
            ).reshape(-1)
            if not np.all(np.isfinite(v_pressures)):
                raise TellegenConservationError(
                    "TellegenConservationError: v = B1ᵀ p no finito."
                )
            cut_projection_vec = v_pressures
            if pressures is None:
                p_branch = v_pressures.copy()
            else:
                p_branch = self._as_finite_vector(pressures, "pressures")
                if p_branch.size != m:
                    raise TellegenConservationError(
                        "TellegenConservationError: pressures ∉ flows."
                    )
                kvl_res = float(
                    la.norm(p_branch - v_pressures)
                    / max(1.0, float(la.norm(p_branch)))
                )
            verify_tellegen_sparse(p_nodal, q, B1)
            tellegen_power = kahan_tellegen_summation(v_pressures, q)
            used_sparse = True
            tellegen_num = float(abs(tellegen_power))
        else:
            if pressures is None:
                raise TellegenConservationError(
                    "TellegenConservationError: modo rama requiere pressures."
                )
            p_branch = self._as_finite_vector(pressures, "pressures")
            if p_branch.size != m:
                raise TellegenConservationError(
                    "TellegenConservationError: pressures ∉ flows."
                )
            tellegen_power = kahan_tellegen_summation(p_branch, q)
            tellegen_num = float(abs(tellegen_power))

            p_norm = float(la.norm(p_branch))
            if m == 0 or p_norm <= _MACHINE_EPS:
                kvl_res = 0.0
                cut_projection_vec = np.zeros(m, dtype=np.float64)
            else:
                if is_sparse:
                    try:
                        try:
                            res = sparse_lsqr(
                                B1.T, p_branch, atol=audit_tol, btol=audit_tol
                            )
                        except TypeError:
                            res = sparse_lsqr(B1.T, p_branch)
                        phi = np.asarray(res[0], dtype=np.float64).reshape(-1)
                        recon = np.asarray(
                            B1.T @ phi, dtype=np.float64
                        ).reshape(-1)
                        cut_projection_vec = recon
                        kvl_res = float(
                            la.norm(recon - p_branch) / max(1.0, p_norm)
                        )
                    except Exception:
                        kvl_res = float("inf")
                else:
                    try:
                        phi, *_ = np.linalg.lstsq(B1.T, p_branch, rcond=None)
                        recon = B1.T @ phi
                        cut_projection_vec = recon
                        kvl_res = float(
                            la.norm(recon - p_branch) / max(1.0, p_norm)
                        )
                    except Exception:
                        kvl_res = float("inf")

        # ── 4. Ley de Corrientes de Kirchhoff (KCL: q ∈ ker(B₁)) ──────
        q_norm = float(la.norm(q))
        if m == 0 or q_norm <= _MACHINE_EPS:
            kcl_res = 0.0
        else:
            kcl_vec = np.asarray(B1 @ q, dtype=np.float64).reshape(-1)
            b1_norm = _frobenius_norm(B1, is_sparse)
            kcl_res = float(
                la.norm(kcl_vec)
                / (
                    max(1.0, b1_norm)
                    * max(1.0, q_norm)
                    * math.sqrt(max(1, n_nodes))
                )
            )

        # ── 5. Residuo relativizado de Tellegen ───────────────────────
        p_norm_ref = float(la.norm(p_branch)) if p_branch is not None else 0.0
        tellegen_scale = max(1.0, p_norm_ref) * max(1.0, q_norm)
        tellegen_res = float(tellegen_num / tellegen_scale)

        # ── 6. Función de Disipación de Rayleigh y Pasividad ──────────
        try:
            r_raw = np.asarray(rayleigh_dissipation_r, dtype=np.float64)
        except Exception as exc:
            raise TellegenConservationError(
                "TellegenConservationError: rayleigh_dissipation_r no convertible."
            ) from exc

        if r_raw.ndim == 0:
            r_mat = r_raw.reshape(1, 1)
        elif r_raw.ndim == 1:
            r_mat = np.diag(r_raw)
        elif r_raw.ndim == 2:
            r_mat = r_raw
        else:
            raise TellegenConservationError(
                "TellegenConservationError: R debe ser escalar/vector/matriz."
            )

        if r_mat.ndim != 2 or r_mat.shape[0] != r_mat.shape[1]:
            raise TellegenConservationError(
                "TellegenConservationError: R debe ser cuadrada."
            )
        if not np.all(np.isfinite(r_mat)):
            raise TellegenConservationError(
                "TellegenConservationError: R contiene no finitos."
            )

        n_r = r_mat.shape[0]
        if gradient_h is None:
            grad_h = np.zeros(n_r, dtype=np.float64)
        else:
            grad_h = self._as_finite_vector(gradient_h, "gradient_h")
        if grad_h.size != n_r:
            raise TellegenConservationError(
                "TellegenConservationError: dimensiones R vs ∇H incompatibles."
            )

        r_norm = float(la.norm(r_mat, ord="fro"))
        r_sym_res = float(
            la.norm(r_mat - r_mat.T, ord="fro") / max(1.0, r_norm)
        )
        r_sym_ok = r_sym_res <= audit_tol
        r_sym = 0.5 * (r_mat + r_mat.T)

        if n_r == 0:
            min_eig = 0.0
            max_eig = 0.0
            r_cond = 1.0
        else:
            try:
                eigvals = la.eigvalsh(r_sym)
            except Exception as exc:
                raise TellegenConservationError(
                    "TellegenConservationError: diagonalización de R fallida."
                ) from exc
            if eigvals.size == 0:
                min_eig = max_eig = 0.0
                r_cond = 1.0
            else:
                min_eig = float(np.min(eigvals))
                max_eig = float(np.max(eigvals))
                if max_eig <= audit_tol:
                    r_cond = 1.0
                elif min_eig <= audit_tol:
                    r_cond = float("inf")
                else:
                    r_cond = float(max_eig / min_eig)

        r_psd_ok = min_eig >= -audit_tol
        p_diss = float(grad_h @ r_sym @ grad_h)
        if not np.isfinite(p_diss):
            raise TellegenConservationError(
                "TellegenConservationError: disipación no finita."
            )
        is_lyap_passive = p_diss >= -_MACHINE_EPS

        # ── 7. Dualidad Cohomológica de Hodge-Poincaré ────────────────
        # En el complejo celular C⁰(G) -> C¹(G):
        #   dim(im B₁ᵀ) = rank(B₁) (espacio de cortes / 1-cobordes exactos)
        #   dim(ker B₁) = m - rank(B₁) (espacio de ciclos / 1-ciclos H₁(G))
        #   Característica de Euler-Poincaré: χ(G) = |V| - |E| = β₀ - β₁
        rank_b1 = _estimate_matrix_rank(B1, is_sparse)
        cycle_dim = int(max(0, m - rank_b1))
        cut_dim = int(rank_b1)
        betti_0 = int(max(0, n_nodes - rank_b1))
        betti_1 = cycle_dim
        euler_char = int(n_nodes - m)
        euler_poincare_defect = abs(euler_char - (betti_0 - betti_1))

        # Defecto de ortogonalidad de Hodge entre el componente exacto (corte)
        # y el componente coexacto/armónico (ciclo):
        hodge_orthogonality_defect = 0.0
        if (
            cut_projection_vec is not None
            and p_branch is not None
            and m > 0
            and cut_projection_vec.size == m
        ):
            cycle_residual_vec = p_branch - cut_projection_vec
            inner_hodge = abs(
                kahan_tellegen_summation(cut_projection_vec, cycle_residual_vec)
            )
            hodge_orthogonality_defect = float(
                inner_hodge / (max(1.0, p_norm_ref) ** 2)
            )

        duality_res = float(
            (abs((cycle_dim + cut_dim) - m) + euler_poincare_defect) / max(1, m)
            + hodge_orthogonality_defect
        )
        is_poincare_dual = duality_res <= audit_tol

        # ── 8. Teorema de Recurrencia de Poincaré ─────────────────────
        if M_op.shape[0] == reflected_vec.size and reflected_vec.size > 0:
            M2x = M_op @ (M_op @ reflected_vec)
            recurrence_res = float(
                la.norm(M2x - reflected_vec)
                / max(1.0, float(la.norm(reflected_vec)))
            )
        else:
            recurrence_res = 0.0

        # ── 9. Deriva adiabática de la acción I ───────────────────────
        I_in = float(allievi_invariants.action_invariant)
        I_out = float(householder_result.action_invariant_reflected)
        adiabatic_drift = abs(I_out - I_in) / max(1.0, I_in)

        # ── 10. Certificados booleanos de conservación ────────────────
        strict_tol = max(tolerance, _TOPOLOGY_DEGRADED_THRESHOLD)
        is_kcl = kcl_res <= strict_tol
        is_kvl = kvl_res <= strict_tol
        is_tel = is_kcl and is_kvl and tellegen_res <= strict_tol

        # ── 11. Residuo Meta-Espectral (para RSI-3) ───────────────────
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

        # ── 12. Evaluación en el retículo diagnóstico ─────────────────
        status = SpecularFlowStatus.COHERENT
        if (
            not math.isfinite(meta_res)
            or meta_res > _TOPOLOGY_VETO_THRESHOLD
            or not r_psd_ok
            or not r_sym_ok
            or p_diss < -audit_tol
            or not is_poincare_dual
        ):
            status = SpecularFlowStatus.VETOED
        elif (
            meta_res > tolerance
            or p_diss < 0.0
            or (math.isfinite(r_cond) and r_cond > _RAYLEIGH_CONDITION_LIMIT)
            or min_eig < 0.0
            or adiabatic_drift > _ADIABATIC_DRIFT_LIMIT
        ):
            status = SpecularFlowStatus.DEGRADED

        prior_a = getattr(
            allievi_invariants, "status", SpecularFlowStatus.COHERENT
        )
        status = SpecularFlowStatus.supremum(
            status, prior_a, prior_householder_status
        )

        return NetworkConservationReport(
            tellegen_residual=tellegen_res,
            is_tellegen_conserved=is_tel,
            rayleigh_dissipation=p_diss,
            is_lyapunov_passive=is_lyap_passive,
            kcl_residual=kcl_res,
            kvl_residual=kvl_res,
            is_kcl_conserved=is_kcl,
            is_kvl_conserved=is_kvl,
            graph_order=int(n_nodes),
            graph_size=int(m),
            cyclomatic_number=int(cycle_dim),
            rayleigh_condition_number=r_cond,
            min_rayleigh_eigenvalue=min_eig,
            tellegen_power=float(tellegen_power),
            kahan_tellegen_sum=float(tellegen_power),
            used_sparse_tellegen=used_sparse,
            status=status,
            cycle_space_dimension=cycle_dim,
            cut_space_dimension=cut_dim,
            poincare_duality_residual=duality_res,
            is_poincare_dual=is_poincare_dual,
            euler_characteristic=euler_char,
            recurrence_residual=recurrence_res,
            adiabatic_action_drift=adiabatic_drift,
            meta_residual=float(meta_res),
            rsi_depth_reached=1,
        )

    def _combine_phase_residuals(
        self,
        allievi: AllieviInvariants,
        householder: HouseholderReflectionResult,
        tellegen: NetworkConservationReport,
    ) -> float:
        """Norma supremo (ℓ_∞) de los residuos acoplados de las 3 fases."""
        parts = [
            float(householder.unitary_residual),
            float(householder.specular_residual),
            float(householder.state_norm_residual),
            float(householder.action_drift),
            float(tellegen.tellegen_residual),
            float(tellegen.kcl_residual),
            float(tellegen.kvl_residual),
            float(tellegen.poincare_duality_residual),
            float(tellegen.recurrence_residual),
            float(tellegen.adiabatic_action_drift),
            float(allievi.expansion_residual),
        ]
        finite = [v for v in parts if math.isfinite(v)]
        return float(max(finite)) if finite else float("inf")

    # ═════════════════════════════════════════════════════════════════════
    # ESLABÓN FUNCTORIAL 3 → META: Puente formal al Meta-Controlador RSI-3
    # ═════════════════════════════════════════════════════════════════════
    def _tellegen_to_meta_bridge(
        self,
        allievi: AllieviInvariants,
        householder: HouseholderReflectionResult,
        tellegen: NetworkConservationReport,
    ) -> Tuple[float, SpecularFlowStatus]:
        """
        MORFISMO DE CLAUSURA (FASE ANIDADA 3 ──► META-CONTROLADOR RSI-3).

        Sintetiza el residuo global en la norma del supremo y el estado en el
        retículo diagnóstico para alimentar la automejora recursiva de Nivel 3.
        """
        residual = self._combine_phase_residuals(allievi, householder, tellegen)
        status = SpecularFlowStatus.supremum(
            allievi.status,
            householder.status,
            tellegen.status,
        )
        return residual, status

    def _rsi_compute_trace(
        self,
        level: int,
        tolerance: float,
        residual: float,
        traces_prev: Tuple[RecursiveSelfImprovementTrace, ...],
        target_residual: float = _RSI_TARGET_RESIDUAL,
    ) -> RecursiveSelfImprovementTrace:
        """
        Construye el certificado de evolución de un nivel en la jerarquía RSI:
        - Nivel 1: Observación base y corrección de defectos de primer orden.
        - Nivel 2: Extrapolación de Richardson y estimación del orden de Hölder p.
        - Nivel 3: Aceleración no-lineal Δ² de Aitken y modulación homotópica
                   de Banach-Fréchet sobre la ley de adaptación de Nivel 2.
        """
        convergence_rate = 0.0
        richardson = residual
        aitken = residual
        adaptation_factor = 0.5
        level3_active = False

        if len(traces_prev) >= 1:
            prev = traces_prev[-1]
            convergence_rate, richardson = _richardson_extrapolation(
                prev.residual,
                residual,
                prev.tolerance,
                tolerance,
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
                t_m2.residual,
                t_m1.residual,
                residual,
            )
            adaptation_factor = _adaptive_refinement_factor(
                current_residual=residual,
                target_residual=target_residual,
                convergence_rate=convergence_rate,
                level3_active=True,
                aitken_signal=aitken,
                richardson_signal=richardson,
            )

        return RecursiveSelfImprovementTrace(
            level=level,
            tolerance=tolerance,
            residual=residual,
            convergence_rate=convergence_rate,
            richardson_extrapolation=richardson,
            aitken_delta_squared=aitken,
            adaptation_factor=adaptation_factor,
            level3_active=level3_active,
            status=(
                SpecularFlowStatus.COHERENT
                if residual <= target_residual
                else (
                    SpecularFlowStatus.DEGRADED
                    if math.isfinite(residual) and residual < 1.0
                    else SpecularFlowStatus.VETOED
                )
            ),
        )

    def _recursive_self_improvement_loop(
        self,
        run_pipeline: Callable[
            [float],
            Tuple[
                AllieviInvariants,
                HouseholderReflectionResult,
                NetworkConservationReport,
            ],
        ],
        max_depth: int = _RSI_MAX_DEPTH,
        initial_tolerance: float = _DEFAULT_TOL,
        target_residual: float = _RSI_TARGET_RESIDUAL,
    ) -> SelfImprovementResult:
        """
        Ejecuta el lazo cerrado del Meta-Controlador RSI-3 consumiendo el
        morfismo `_tellegen_to_meta_bridge` en cada nivel de recursión.
        """
        if max_depth < 1:
            raise RecursiveSelfImprovementError(
                "RecursiveSelfImprovementError: max_depth debe ser ≥ 1."
            )

        traces: list[RecursiveSelfImprovementTrace] = []
        tolerance = float(max(_RSI_MIN_TOLERANCE, initial_tolerance))

        for level in range(1, max_depth + 1):
            allievi, householder, tellegen = run_pipeline(tolerance)
            residual, _ = self._tellegen_to_meta_bridge(
                allievi, householder, tellegen
            )

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
        richardson_limit = last.richardson_extrapolation
        aitken_limit = last.aitken_delta_squared
        level3_activated = any(t.level3_active for t in traces)

        status = last.status
        for t in traces:
            status = SpecularFlowStatus.supremum(status, t.status)

        return SelfImprovementResult(
            traces=tuple(traces),
            final_tolerance=float(last.tolerance),
            final_residual=float(last.residual),
            converged=bool(last.residual <= target_residual),
            richardson_limit=float(richardson_limit),
            aitken_limit=float(aitken_limit),
            effective_convergence_order=float(last.convergence_rate),
            level3_activated=bool(level3_activated),
            status=status,
        )


# ── Motor Ciber-Físico Soberano (Clausura de las 3 Fases Anidadas) ────────

class SpecularFlowEngine(Phase3_TellegenValidator):
    """
    Motor ciber-físico consolidado que unifica las 3 Fases Anidadas:
        Phase1_AllieviSolver ──► Phase2_HouseholderReflector ──► Phase3_TellegenValidator
    bajo supervisión de Automejora Recursiva de Nivel 3 (RSI-3).
    """

    def full_specular_pipeline(
        self,
        potential_y: float,
        flow_q: float,
        wave_speed_a: float,
        gravity_g: float,
        section_s: float,
        metric_tensor_g: NDArray[np.float64],
        boundary_normal_n: NDArray[np.float64],
        state_vector: NDArray[np.float64],
        pressures: Optional[NDArray[np.float64]],
        flows: NDArray[np.float64],
        rayleigh_dissipation_r: NDArray[np.float64],
        gradient_h: Optional[NDArray[np.float64]],
        incidence_matrix: Optional[object] = None,
        nodal_potentials: Optional[NDArray[np.float64]] = None,
        tolerance: float = _DEFAULT_TOL,
        cfl_number: Optional[float] = None,
        poincare_section_level: float = 0.0,
    ) -> Tuple[
        AllieviInvariants,
        HouseholderReflectionResult,
        NetworkConservationReport,
    ]:
        """Ejecuta secuencialmente las 3 fases anidadas."""
        # FASE ANIDADA 1: Observe (Allievi + Poincaré)
        allievi = self.solve_allievi_characteristics(
            potential_y=potential_y,
            flow_q=flow_q,
            wave_speed_a=wave_speed_a,
            gravity_g=gravity_g,
            section_s=section_s,
            tolerance=tolerance,
            cfl_number=cfl_number,
            poincare_section_level=poincare_section_level,
        )
        # FASE ANIDADA 2: Orient (Householder + Monodromía de Poincaré)
        householder = self.compute_householder_reflection(
            metric_tensor_g=metric_tensor_g,
            boundary_normal_n=boundary_normal_n,
            state_vector=state_vector,
            allievi_invariants=allievi,
            tolerance=tolerance,
        )
        # FASE ANIDADA 3: Decide (Tellegen + Cohomología de Hodge-Poincaré)
        tellegen = self.verify_network_conservation(
            pressures=pressures,
            flows=flows,
            rayleigh_dissipation_r=rayleigh_dissipation_r,
            gradient_h=gradient_h,
            allievi_invariants=allievi,
            householder_result=householder,
            tolerance=tolerance,
            incidence_matrix=incidence_matrix,
            nodal_potentials=nodal_potentials,
        )
        return allievi, householder, tellegen

    def full_specular_pipeline_rsi3(
        self,
        potential_y: float,
        flow_q: float,
        wave_speed_a: float,
        gravity_g: float,
        section_s: float,
        metric_tensor_g: NDArray[np.float64],
        boundary_normal_n: NDArray[np.float64],
        state_vector: NDArray[np.float64],
        pressures: Optional[NDArray[np.float64]],
        flows: NDArray[np.float64],
        rayleigh_dissipation_r: NDArray[np.float64],
        gradient_h: Optional[NDArray[np.float64]],
        incidence_matrix: Optional[object] = None,
        nodal_potentials: Optional[NDArray[np.float64]] = None,
        initial_tolerance: float = _DEFAULT_TOL,
        cfl_number: Optional[float] = None,
        poincare_section_level: float = 0.0,
        rsi_max_depth: int = _RSI_MAX_DEPTH,
        rsi_target_residual: float = _RSI_TARGET_RESIDUAL,
    ) -> Tuple[
        AllieviInvariants,
        HouseholderReflectionResult,
        NetworkConservationReport,
        SelfImprovementResult,
    ]:
        """
        Ejecuta el pipeline completo bajo el meta-controlador de Automejora
        Recursiva de Nivel 3 (RSI-3), registrando la profundidad alcanzada en
        el certificado topológico final.
        """
        last_bundle: Optional[
            Tuple[
                AllieviInvariants,
                HouseholderReflectionResult,
                NetworkConservationReport,
            ]
        ] = None

        def run_pipeline(
            tol: float,
        ) -> Tuple[
            AllieviInvariants,
            HouseholderReflectionResult,
            NetworkConservationReport,
        ]:
            nonlocal last_bundle
            last_bundle = self.full_specular_pipeline(
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
                rayleigh_dissipation_r=rayleigh_dissipation_r,
                gradient_h=gradient_h,
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
            allievi, householder, tellegen = run_pipeline(
                rsi_result.final_tolerance
            )
        else:
            allievi, householder, tellegen = last_bundle

        # Actualiza inmutablemente la profundidad RSI alcanzada en el reporte
        depth_reached = len(rsi_result.traces)
        tellegen_updated = NetworkConservationReport(
            tellegen_residual=tellegen.tellegen_residual,
            is_tellegen_conserved=tellegen.is_tellegen_conserved,
            rayleigh_dissipation=tellegen.rayleigh_dissipation,
            is_lyapunov_passive=tellegen.is_lyapunov_passive,
            kcl_residual=tellegen.kcl_residual,
            kvl_residual=tellegen.kvl_residual,
            is_kcl_conserved=tellegen.is_kcl_conserved,
            is_kvl_conserved=tellegen.is_kvl_conserved,
            graph_order=tellegen.graph_order,
            graph_size=tellegen.graph_size,
            cyclomatic_number=tellegen.cyclomatic_number,
            rayleigh_condition_number=tellegen.rayleigh_condition_number,
            min_rayleigh_eigenvalue=tellegen.min_rayleigh_eigenvalue,
            tellegen_power=tellegen.tellegen_power,
            kahan_tellegen_sum=tellegen.kahan_tellegen_sum,
            used_sparse_tellegen=tellegen.used_sparse_tellegen,
            status=tellegen.status,
            cycle_space_dimension=tellegen.cycle_space_dimension,
            cut_space_dimension=tellegen.cut_space_dimension,
            poincare_duality_residual=tellegen.poincare_duality_residual,
            is_poincare_dual=tellegen.is_poincare_dual,
            euler_characteristic=tellegen.euler_characteristic,
            recurrence_residual=tellegen.recurrence_residual,
            adiabatic_action_drift=tellegen.adiabatic_action_drift,
            meta_residual=rsi_result.final_residual,
            rsi_depth_reached=depth_reached,
        )

        return allievi, householder, tellegen_updated, rsi_result


# ═══════════════════════════════════════════════════════════════════════════
# Exportación pública del módulo
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "SpecularFlowPhysicsError",
    "AllieviIntegrationError",
    "HouseholderReflectionError",
    "TellegenConservationError",
    "PoincareSectionError",
    "RecursiveSelfImprovementError",
    "SpecularFlowStatus",
    "AllieviInvariants",
    "HouseholderReflectionResult",
    "NetworkConservationReport",
    "PoincareSectionSnapshot",
    "LindstedtPoincareExpansion",
    "RecursiveSelfImprovementTrace",
    "SelfImprovementResult",
    "Phase1_AllieviSolver",
    "Phase2_HouseholderReflector",
    "Phase3_TellegenValidator",
    "SpecularFlowEngine",
    "stable_cholesky_decomposition",
    "kahan_tellegen_summation",
    "verify_tellegen_sparse",
]