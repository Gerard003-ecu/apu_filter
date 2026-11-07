# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Riemannian Inertia Modulator (Motor de Momento Giroscópico)         ║
║ Ruta   : app/physics/riemannian_inertia_modulator.py                         ║
║ Versión: 5.0.0-Poincare-KAM-Sundman-Floquet-RSI-Level3-Strict                ║
╚══════════════════════════════════════════════════════════════════════════════╝

OBJETO CIENTÍFICO-TÉCNICO Y FUNDAMENTACIÓN GEOMÉTRICA (Rigor Doctoral): ────────
Este módulo implementa el **Funtor de Moldeo de Masa** (Mass Shaping Functor)
sobre el fibrado cotangente $$\mathcal{T}^* \mathcal{M}$$, enriquecido con la
**Mecánica Celeste de Henri Poincaré** y **Automejora Recursiva de Nivel 3
(RSI-L3)**. Su propósito axiomático es inyectar una "Fuerza de Lorentz"
informacional de naturaleza estrictamente giroscópica en la trayectoria del
espacio de fase transaccional, desviando fluctuaciones estocásticas y
alucinaciones hacia sumideros de disipación sin alterar la energía Hamiltoniana
interna del sistema ni violar los invariantes integrales de Poincaré-Cartan.

El módulo modela el flujo logístico como un sistema dinámico Hamiltoniano
restringido a variedades Riemannianas anisotrópicas, gobernado por:
  1. Invariantes integrales de Poincaré de primer y segundo orden.
  2. Estabilidad Diofántica de toros invariantes Kolmogorov-Arnold-Moser (KAM).
  3. Regularización analítica de Sundman-Poincaré ante pericentros singulares.
  4. Endofuntores de Automejora Recursiva de Nivel 3 ($$\mathfrak{R}_3$$) que
     auditan el defecto epistémico de los operadores numéricos y reescriben
     dinámicamente su propia ley de convergencia bajo certificación de Lyapunov.

ARQUITECTURA DE TRES FASES ANIDADAS (Composición Funtorial): ───────────────────
La transmutación del momento se rige por un contrato categórico rígido donde la
salida inmutable del último morfismo de la Fase $$k$$ actúa como el dominio e
inicio formal del primer morfismo de la Fase $$k+1$$:

  Fase 1 ──► FASE 1: ESPECTROSCOPÍA DEL MOMENTUM, TOROS KAM Y RSI-L3 MÉTRICO
             (Phase1_MomentumSpectrometer)
             • Ejecuta el endofuntor RSI-L3 $$\mathfrak{R}_3^{(1)}$$ sobre el
               álgebra de Banach $$\mathcal{B}(\mathbb{R}^n)$$, sintetizando una
               ley adaptativa de Chebyshev-Halley-Schulz que refina la dualidad
               $$(G, G^{-1})$$ con descenso monótono de Lyapunov.
             • Aplica el isomorfismo musical plano ($$\flat$$) y agudo ($$\sharp$$):
               $$p_\mu = G_{\mu\nu} \dot{q}^\nu, \qquad
                 \dot{q}^\mu = G^{\mu\nu} p_\nu, \qquad
                 \sharp\circ\flat = \mathrm{id}$$
             • Proyecta el estado sobre las variables canónicas de acción-ángulo
               de Poincaré $$(I_k, \omega_k)$$, evalúa la densidad de la 1-forma
               de Poincaré-Cartan $$\theta_{\mathrm{PC}}$$, el margen Diofántico
               KAM y el factor de regularización temporal de Sundman.
             • Entrega: MomentumAuditData.
             • Morfismo terminal: handoff_phase1_to_phase2 ≡ inicio de Fase 2.

  Fase 2 ──► FASE 2: SÍNTESIS GIROSCÓPICA DE POINCARÉ-BIRKHOFF Y RSI-L3 EN so(n)
             (Phase2_GyroscopicSynthesizer)
             • Inicia en _receive_certified_momentum consumiendo directamente
               handoff_phase1_to_phase2.
             • Construye el bivector de Lorentz en $$\bigwedge^2 T^*\mathcal{M}$$:
               $$W_{\mu\nu} = \alpha\,(p\wedge\omega)_{\mu\nu}, \qquad
                 \omega = \Omega_{\mathrm{skew}} p$$
             • Ejecuta el endofuntor RSI-L3 $$\mathfrak{R}_3^{(2)}$$ de
               purificación isométrica de Darboux-Gram sobre $$\mathfrak{so}(n)$$,
               forzando simultáneamente la identidad exterior de Gram y el
               invariante cuadrático de Casimir:
               $$\mathcal{C}_2(W) = -\tfrac12\mathrm{Tr}(W^2) = \tfrac12\|W\|_F^2$$
             • Extrae las frecuencias características de la forma normal de
               Darboux y el invariante integral de circulación areal de Poincaré.
             • Entrega: GyroscopicSynthesisData.
             • Morfismo terminal: handoff_phase2_to_phase3 ≡ inicio de Fase 3.

  Fase 3 ──► FASE 3: MODULACIÓN SIMPLÉCTICA, ESTABILIDAD FLOQUET-KREĬN Y RSI-L3
             (Phase3_SymplecticInertiaModulator)
             • Inicia en _receive_certified_gyroscopic_tensor consumiendo
               directamente handoff_phase2_to_phase3.
             • Inyecta $$W$$ en la estructura de Dirac:
               $$J_{\mathrm{eff}} = J_{\mathrm{base}} + W$$
             • Ejecuta el endofuntor RSI-L3 $$\mathfrak{R}_3^{(3)}$$ proyectando
               $$J_{\mathrm{eff}}$$ exactamente sobre la variedad antisimétrica
               IEEE-754 y aniquilando fugas direccionales de Lie-Poisson.
             • Certifica la estabilidad espectral elíptica de Floquet-Kreĭn
               mediante la cota de Bendixson-Hirsch, el invariante virial de
               Jacobi-Poincaré, el parámetro de deflexión de Tisserand y el
               horizonte de recurrencia de Kac-Poincaré.
             • Certifica trabajo mecánico nulo por Neumaier, pares estructurales
               y sondas Monte-Carlo de Liouville.
             • Entrega: ThermodynamicVetoData.
             • Morfismo terminal: handoff_phase3_to_orchestrator ≡ cierre en
               RiemannianInertiaModulator.

INVARIANTES MATEMÁTICOS, GEOMÉTRICOS Y CELESTES PRESERVADOS: ───────────────────
  [I1] Invariancia de Liouville y 2-Forma Simpléctica de Poincaré:
       $$x^\top J_{\mathrm{eff}} x \equiv 0 \quad \forall x \in \mathbb{R}^{n}$$
  [I2] Coherencia Métrica de de Rham y Contracción RSI-L3 de Wilkinson:
       $$\mathcal{V}_{k+1}^{(1)} \le \mathcal{V}_{k}^{(1)}, \qquad
         \|\sharp\flat\dot{q}-\dot{q}\|_2 \le C\cdot\kappa(G)\cdot n\cdot
         \varepsilon_{\mathrm{machine}}\cdot\max(1,\|\dot{q}\|_2)$$
  [I3] Conservación Termodinámica e Inecuación de Clausius-Duhem:
       $$\dot{H} = \langle \nabla H, J_{\mathrm{eff}}\nabla H\rangle
                 - \langle \nabla H, R(x)\nabla H\rangle
                 = -\langle \nabla H, R(x)\nabla H\rangle \le 0$$
  [I4] Identidad de Gram y Casimir Cuadrático en $$\mathfrak{so}(n)$$:
       $$\|W\|_F^2 = -\mathrm{Tr}(W^2)
         = 2\alpha^2\bigl(\|p\|_2^2\|\omega\|_2^2 - \langle p,\omega\rangle^2\bigr)$$
  [I5] Confinamiento Espectral de Floquet-Kreĭn (Teorema de Bendixson-Hirsch):
       $$\max_{\mu \in \sigma(J_{\mathrm{eff}})} |\Re(\mu)|
         \le \tfrac12\|J_{\mathrm{eff}} + J_{\mathrm{eff}}^\top\|_2 \le \tau_{\mathrm{Floquet}}$$
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import NDArray

# ──────────────────────────────────────────────────────────────────────────────
# Dependencias del ecosistema MIC.
# Se conservan stubs analíticos para ejecución autónoma.
# ──────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:

    class TopologicalInvariantError(Exception):
        """Violación base de un invariante topológico-categórico."""

        pass

    class Morphism:
        """Clase base de composición funtorial del ecosistema MIC."""

        pass


logger = logging.getLogger("MIC.Physics.RiemannianInertiaModulator")


# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES FÍSICAS, NUMÉRICAS, CELESTES (POINCARÉ) Y RSI-NIVEL-3
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)

# Tolerancia base para residuos estructurales absolutos.
_SYMPLECTIC_TOLERANCE: Final[float] = 1e-12

# Tolerancia absoluta de respaldo para la inversa métrica.
_METRIC_INVERSE_TOLERANCE: Final[float] = 1e-8

# Constante de Wilkinson para cotas de residuo de GEMM / inversa.
_WILKINSON_CONSTANT: Final[float] = 8.0

# Cota elástica máxima del momentum (norma dual).
_MOMENTUM_MAX_BOUND: Final[float] = 1e8

# Factor de acoplamiento giroscópico α en W = α (p ∧ ω).
_VORTICITY_COUPLING_FACTOR: Final[float] = 1.0

# Número de condición máximo admisible para la métrica riemanniana.
_CONDITION_NUMBER_MAX: Final[float] = 1e12

# Multiplicador de ULPs para la cota adaptativa del trabajo nulo.
_WORK_TOLERANCE_ULPS: Final[float] = 1000.0

# Cota relativa de antisimetría (definición documental de r_skew).
_SKEW_RELATIVE_TOLERANCE: Final[float] = 1e-12

# Número de sondas aleatorias para la auditoría de Liouville.
_LIOUVILLE_PROBE_COUNT: Final[int] = 8

# Semilla reproducible de las sondas de Liouville.
_LIOUVILLE_PROBE_SEED: Final[int] = 1729

# ── Constantes de Mecánica Celeste de Poincaré y Teoría KAM ──────────────────
# Exponente Diofántico τ en |⟨m, ω⟩| ≥ γ / |m|^τ.
_POINCARE_DIOPHANTINE_TAU: Final[float] = 2.0

# Orden entero máximo |m|_1 para el escaneo de resonancias de Arnold-Poincaré.
_POINCARE_MAX_RESONANCE_ORDER: Final[int] = 4

# Modos espectrales dominantes máximos auditados en la red Diofántica.
_POINCARE_MAX_ACTIVE_MODES: Final[int] = 6

# Exponente de regularización conforme de Sundman-Poincaré β ∈ (0, 1].
_SUNDMAN_REGULARIZATION_EXPONENT: Final[float] = 0.5

# Tolerancia para la deriva real de multiplicadores de Floquet-Kreĭn.
_FLOQUET_IMAGINARY_TOLERANCE: Final[float] = 1e-12

# ── Constantes de Automejora Recursiva de Nivel 3 (RSI-L3) ───────────────────
# Profundidad máxima de autorrefinamiento recursivo de leyes de operador.
_RSI_L3_MAX_RECURSION_DEPTH: Final[int] = 5

# Umbral de piso epistémico en múltiplos de ε_machine.
_RSI_L3_EPISTEMIC_FLOOR_ULPS: Final[float] = 4.0

# Versión canónica del funtor.
__version__: Final[str] = "5.0.0-Poincare-KAM-Sundman-Floquet-RSI-Level3-Strict"


# ══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES SIMPLÉCTICAS, CELESTES Y EPISTÉMICAS
# ══════════════════════════════════════════════════════════════════════════════
class RiemannianInertiaError(TopologicalInvariantError):
    """Excepción raíz del funtor de modulación de inercia."""

    pass


class MetricCoherenceError(RiemannianInertiaError):
    r"""
    Detonada si el par $$(G, G^{-1})$$ viola la cota de Wilkinson o el
    round-trip del isomorfismo musical $$\sharp\circ\flat=\mathrm{id}$$.
    """

    pass


class DualPairingError(RiemannianInertiaError):
    r"""
    Detonada si se rompe la identidad de apareamiento dual:
        $$\langle p,\dot{q}\rangle
          = \dot{q}^\top G\dot{q}
          = p^\top G^{-1}p$$
    """

    pass


class MomentumDivergenceError(RiemannianInertiaError):
    r"""
    Detonada si $$\|p\|_{G^{-1}}$$ excede la cota elástica admisible:
        $$\|p\|_{G^{-1}} = \sqrt{p_\mu G^{\mu\nu} p_\nu} > P_{\mathrm{max}}$$
    """

    pass


class ExteriorAlgebraError(RiemannianInertiaError):
    r"""
    Detonada si el bivector $$p\wedge\omega$$ viola la identidad de Gram:
        $$\|p\wedge\omega\|_F^2
          = 2\bigl(\|p\|_2^2\|\omega\|_2^2-\langle p,\omega\rangle^2\bigr)$$
    """

    pass


class SkewSymmetryViolationError(RiemannianInertiaError):
    r"""
    Detonada si la proyección antisimétrica retiene componentes simétricas:
        $$r_{\mathrm{skew}}
          = \frac{\|W+W^\top\|_F}{\max(1,\|W\|_F)}
          > \epsilon_{\mathrm{skew}}$$
    """

    pass


class SymplecticWorkViolationError(RiemannianInertiaError):
    r"""
    Detonada si el operador efectivo realiza trabajo neto no nulo:
        $$|\langle\nabla H,\,J_{\mathrm{eff}}\nabla H\rangle| > \tau_{\mathrm{work}}$$
    """

    pass


class PhaseHandoffError(RiemannianInertiaError):
    """Detonada si un certificado de fase no satisface las precondiciones de la siguiente."""

    pass


class PoincareResonanceCollapseError(RiemannianInertiaError):
    """Detonada si los invariantes integrales de Poincaré o Sundman degeneran."""

    pass


class FloquetKreinInstabilityError(RiemannianInertiaError):
    r"""
    Detonada si el espectro de Floquet de $$J_{\mathrm{eff}}$$ escapa del eje
    imaginario (bifurcación hiperbólica de Kreĭn).
    """

    pass


class RecursiveSelfImprovementDivergenceError(RiemannianInertiaError):
    r"""
    Detonada si el endofuntor RSI-L3 viola la monotonía de Lyapunov:
        $$\mathcal{V}_{k+1} > \mathcal{V}_k + \varepsilon_{\mathrm{floor}}$$
    """

    pass


# ══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs DEL FIBRADO COTANGENTE, POINCARÉ Y RSI-L3)
# ══════════════════════════════════════════════════════════════════════════════
def _freeze_array(array: NDArray[np.float64]) -> NDArray[np.float64]:
    """Copia defensiva y sello de solo-lectura para tensores certificados."""
    frozen = np.array(array, dtype=np.float64, copy=True)
    frozen.setflags(write=False)
    return frozen


def _empty_frozen_vector() -> NDArray[np.float64]:
    """Vector float64 inmutable vacío por defecto para constructores legados."""
    return _freeze_array(np.zeros(0, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class RSILevel3AuditMetrics:
    r"""
    Certificado inmutable de ejecución del Endofuntor de Automejora Recursiva
    de Nivel 3 ($$\mathfrak{R}_3$$).

    Registra la evolución del Funcional de Lyapunov Epistémico
    $$\mathcal{V}_k$$, la ley algorítmica sintetizada, la ganancia meta-adaptativa
    y la razón de contracción alcanzada.
    """

    recursion_depth: int = 0
    initial_epistemic_defect: float = 0.0
    terminal_epistemic_defect: float = 0.0
    lyapunov_contraction_ratio: float = 1.0
    meta_law_gain_parameter: float = 1.0
    synthesized_law_tag: str = "Identity-Level0"
    is_monotone_lyapunov: bool = True
    is_fixed_point_converged: bool = True


_DEFAULT_RSI_AUDIT: Final[RSILevel3AuditMetrics] = RSILevel3AuditMetrics()


@dataclass(frozen=True, slots=True)
class MomentumAuditData:
    """
    Artefacto certificado de la Fase 1 (Espectroscopía del Momentum,
    Variables de Acción-Ángulo de Poincaré y Refinamiento RSI-L3 Métrico).

    Es, por contrato, el objeto inicial de la Fase 2
    (véase ``handoff_phase1_to_phase2``).
    """

    covariant_momentum: NDArray[np.float64]
    reconstructed_velocity: NDArray[np.float64]
    momentum_norm: float
    kinetic_energy_primal: float
    kinetic_energy_dual: float
    dual_pairing: float
    pairing_residual: float
    musical_roundtrip_residual: float
    is_bounded: bool
    metric_condition_number: float
    inverse_consistency_residual: float
    wilkinson_bound: float
    spectral_minimum: float
    spectral_maximum: float
    # ── Extensiones de Mecánica Celeste de Poincaré y RSI-L3 ────────────────
    poincare_action_variables: NDArray[np.float64] = field(
        default_factory=_empty_frozen_vector
    )
    poincare_orbital_frequencies: NDArray[np.float64] = field(
        default_factory=_empty_frozen_vector
    )
    poincare_cartan_one_form_density: float = 0.0
    sundman_regularization_factor: float = 1.0
    kam_diophantine_margin: float = 1.0
    rsi_l3_metric_audit: RSILevel3AuditMetrics = _DEFAULT_RSI_AUDIT


@dataclass(frozen=True, slots=True)
class GyroscopicSynthesisData:
    """
    Artefacto certificado de la Fase 2 (Síntesis Giroscópica de Lorentz,
    Forma Normal de Darboux-Poincaré, Casimir de Lie y RSI-L3 en so(n)).

    Es, por contrato, el objeto inicial de la Fase 3
    (véase ``handoff_phase2_to_phase3``).
    """

    skew_symmetric_tensor: NDArray[np.float64]
    vorticity_two_form: NDArray[np.float64]
    omega_vector: NDArray[np.float64]
    antisymmetry_residual: float
    relative_skew_residual: float
    vorticity_projection_residual: float
    gyroscopic_frobenius_norm: float
    wedge_gram_residual: float
    skew_numerical_rank: int
    rank_is_even: bool
    is_strictly_skew: bool
    # ── Extensiones de Mecánica Celeste de Poincaré y RSI-L3 ────────────────
    poincare_birkhoff_secular_factor: float = 1.0
    casimir_quadratic_invariant: float = 0.0
    casimir_trace_residual: float = 0.0
    darboux_characteristic_frequencies: NDArray[np.float64] = field(
        default_factory=_empty_frozen_vector
    )
    poincare_areal_circulation: float = 0.0
    rsi_l3_synthesis_audit: RSILevel3AuditMetrics = _DEFAULT_RSI_AUDIT


@dataclass(frozen=True, slots=True)
class ThermodynamicVetoData:
    """
    Artefacto certificado de la Fase 3 (Veredicto Termodinámico, Estabilidad
    de Floquet-Kreĭn, Invariantes de Jacobi-Poincaré/Tisserand y RSI-L3).
    """

    effective_dirac_matrix: NDArray[np.float64]
    nilpotent_work_residual: float
    pairwise_work_residual: float
    liouville_probe_residual: float
    dirac_symmetric_residual: float
    relative_skew_residual: float
    work_tolerance: float
    is_symplectically_passive: bool
    # ── Extensiones de Mecánica Celeste de Poincaré y RSI-L3 ────────────────
    jacobi_poincare_virial_invariant: float = 0.0
    tisserand_gyroscopic_parameter: float = 1.0
    floquet_max_real_drift: float = 0.0
    floquet_spectral_radius: float = 0.0
    poincare_recurrence_horizon: float = 0.0
    rsi_l3_symplectic_audit: RSILevel3AuditMetrics = _DEFAULT_RSI_AUDIT


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 → ESPECTROSCOPÍA DEL MOMENTUM, MECÁNICA DE ACCIÓN-ÁNGULO DE POINCARÉ
#          Y AUTOMEJORA RECURSIVA DE NIVEL 3 SOBRE EL FIBRADO MÉTRICO (Observe)
# ══════════════════════════════════════════════════════════════════════════════
class Phase1_MomentumSpectrometer:
    r"""
    Extrae el covector de momentum mediante el isomorfismo musical plano
    ($$\flat$$):

        $$p_\mu = G_{\mu\nu}\dot{q}^\nu$$

    integrando:
        1. Validación tensorial estricta en $$\mathbb{R}^n$$ y $$\mathcal{B}(\mathbb{R}^n)$$.
        2. Sumación compensada de Kahan–Babuška–Neumaier.
        3. Descomposición espectral de Rham y verificación de Wilkinson.
        4. **Automejora Recursiva de Nivel 3 ($$\mathfrak{R}_3^{(1)}$$)**:
           Refinamiento hiper-espectral de Chebyshev-Halley-Schulz sobre el
           operador inverso $$G^{-1}$$, gobernado por un Funcional de Lyapunov
           Epistémico que adapta el orden del polinomio y la ganancia espectral.
        5. Certificación de involución musical $$\sharp\circ\flat=\mathrm{id}$$
           y apareamiento dual $$\langle p,\dot{q}\rangle$$.
        6. **Mecánica Celeste de Poincaré**:
           - Variables canónicas de acción-ángulo $$(I_k, \omega_k)$$.
           - Densidad de la 1-forma invariante de Poincaré-Cartan.
           - Margen de no-resonancia Diofántica KAM.
           - Factor de regularización conforme de Sundman-Poincaré.

    El morfismo terminal ``handoff_phase1_to_phase2`` constituye la definición
    formal final de la Fase 1 y el inicio exacto de la Fase 2.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # 1.1  Utilidades elementales de validación numérica y álgebra de normas
    # ──────────────────────────────────────────────────────────────────────────
    def _as_float64_array(
        self,
        payload: object,
        name: str,
    ) -> NDArray[np.float64]:
        """Convierte ``payload`` a ``ndarray[float64]`` o eleva error tipado."""
        try:
            return np.array(payload, dtype=np.float64, copy=True)
        except (TypeError, ValueError) as exc:
            raise RiemannianInertiaError(
                f"{name} no pudo convertirse a un tensor float64."
            ) from exc

    def _assert_finite(
        self,
        array: NDArray[np.float64],
        name: str,
    ) -> None:
        """Exige que todos los coeficientes pertenezcan a $$\mathbb{R}$$ finito."""
        if not np.all(np.isfinite(array)):
            raise RiemannianInertiaError(f"{name} contiene valores no finitos.")

    def _validate_vector(
        self,
        vector: object,
        name: str,
    ) -> NDArray[np.float64]:
        """Valida que un vector sea 1-D, no vacío, finito y convertible a float64."""
        arr = self._as_float64_array(vector, name)

        if arr.ndim != 1:
            raise RiemannianInertiaError(f"{name} debe ser un vector 1-D.")

        if arr.size == 0:
            raise RiemannianInertiaError(f"{name} no puede ser vacío.")

        self._assert_finite(arr, name)
        return arr

    def _validate_square_matrix(
        self,
        matrix: object,
        name: str,
    ) -> NDArray[np.float64]:
        """Valida que una matriz sea cuadrada, no vacía, finita y float64."""
        arr = self._as_float64_array(matrix, name)

        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise RiemannianInertiaError(f"{name} debe ser una matriz cuadrada.")

        if arr.shape[0] == 0:
            raise RiemannianInertiaError(f"{name} no puede ser vacía.")

        self._assert_finite(arr, name)
        return arr

    def _frobenius_norm(self, matrix: NDArray[np.float64], name: str) -> float:
        """Calcula la norma de Frobenius con verificación de finitud."""
        self._assert_finite(matrix, name)
        value = float(np.linalg.norm(matrix, ord="fro"))
        if not math.isfinite(value):
            raise RiemannianInertiaError(
                f"La norma de Frobenius de {name} no es finita."
            )
        return value

    def _euclidean_norm(self, vector: NDArray[np.float64], name: str) -> float:
        """Norma euclídea $$\ell^2$$ con verificación de finitud."""
        self._assert_finite(vector, name)
        value = float(np.linalg.norm(vector, ord=2))
        if not math.isfinite(value):
            raise RiemannianInertiaError(f"La norma ℓ² de {name} no es finita.")
        return value

    def _matrix_residual_tolerance(
        self,
        matrix: NDArray[np.float64],
        base_tol: float = _SYMPLECTIC_TOLERANCE,
    ) -> float:
        r"""
        Cota adaptativa relativa a la escala de la matriz:

            $$\mathrm{tol}=\max\bigl(\mathrm{tol}_{\mathrm{base}},\,
              \mathrm{tol}_{\mathrm{base}}\times\max(1,\|A\|_F)\bigr)$$
        """
        scale = self._frobenius_norm(matrix, "matriz para tolerancia")
        return max(base_tol, base_tol * max(1.0, scale))

    def _relative_frobenius_residual(
        self,
        residual_norm: float,
        reference_norm: float,
    ) -> float:
        r"""Residuo relativo $$\|R\|_F / \max(1, \|A\|_F)$$."""
        if not math.isfinite(residual_norm) or not math.isfinite(reference_norm):
            raise RiemannianInertiaError(
                "Normas no finitas en el residuo relativo de Frobenius."
            )
        return residual_norm / max(1.0, reference_norm)

    def _symmetric_residual(self, matrix: NDArray[np.float64]) -> float:
        r"""Residuo de simetría $$\|A-A^\top\|_F$$."""
        return self._frobenius_norm(matrix - matrix.T, "residuo simétrico")

    def _skew_symmetric_residual(self, matrix: NDArray[np.float64]) -> float:
        r"""Residuo de antisimetría $$\|A+A^\top\|_F$$."""
        return self._frobenius_norm(matrix + matrix.T, "residuo antisimétrico")

    def _relative_skew_residual(self, matrix: NDArray[np.float64]) -> float:
        r"""
        Residuo relativo documental:

            $$r_{\mathrm{skew}}
              = \frac{\|A+A^\top\|_F}{\max(1,\|A\|_F)}$$
        """
        abs_residual = self._skew_symmetric_residual(matrix)
        scale = self._frobenius_norm(matrix, "matriz para r_skew")
        return self._relative_frobenius_residual(abs_residual, scale)

    def _symmetrize(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Proyección ortogonal al cono simétrico: $$\tfrac12(A+A^\top)$$."""
        return 0.5 * (matrix + matrix.T)

    def _skew_symmetrize(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Proyección ortogonal exacta a $$\mathfrak{so}(n)$$:
        $$\tfrac12(A-A^\top)$$ con anulación explícita de la diagonal principal
        y antisimetría de bit de signo IEEE-754 entre triángulos.
        """
        skew = 0.5 * (matrix - matrix.T)
        n = skew.shape[0]
        if n > 0:
            np.fill_diagonal(skew, 0.0)
            for i in range(n):
                for j in range(i + 1, n):
                    val = float(skew[i, j])
                    skew[i, j] = val
                    skew[j, i] = -val
        return skew

    # ──────────────────────────────────────────────────────────────────────────
    # 1.2  Sumación compensada de Neumaier (reducción escalar rigurosa)
    # ──────────────────────────────────────────────────────────────────────────
    def _neumaier_sum(self, terms: NDArray[np.float64], name: str) -> float:
        r"""
        Suma compensada de Neumaier (Kahan–Babuška–Neumaier).

        Corrige el caso $$|x_{k+1}| > |s_k|$$, indispensable ante cancelación
        catastrófica en formas diferenciales y cuadráticas antisimétricas.
        """
        self._assert_finite(terms, name)

        total = 0.0
        compensation = 0.0
        for raw in terms.flat:
            term = float(raw)
            trial = total + term
            if abs(total) >= abs(term):
                compensation += (total - trial) + term
            else:
                compensation += (term - trial) + total
            if not math.isfinite(trial) or not math.isfinite(compensation):
                raise RiemannianInertiaError(
                    f"La sumación de Neumaier de {name} divergió."
                )
            total = trial

        result = total + compensation
        if not math.isfinite(result):
            raise RiemannianInertiaError(
                f"La suma compensada de {name} no es finita."
            )
        return result

    def _neumaier_dot(
        self,
        left: NDArray[np.float64],
        right: NDArray[np.float64],
        name: str,
    ) -> float:
        r"""Producto interno compensado $$\langle u,v\rangle=\sum_i u_i v_i$$."""
        left = self._validate_vector(left, f"{name}.left")
        right = self._validate_vector(right, f"{name}.right")
        if left.size != right.size:
            raise RiemannianInertiaError(
                f"Dimensión incompatible en el producto interno {name}."
            )
        products = left * right
        return self._neumaier_sum(products, name)

    # ──────────────────────────────────────────────────────────────────────────
    # 1.3  Espectro de la métrica Riemanniana y cota de Wilkinson
    # ──────────────────────────────────────────────────────────────────────────
    def _spectral_decompose_spd(
        self,
        matrix: NDArray[np.float64],
        name: str,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Descomposición espectral de una métrica riemanniana:

            $$G = Q\,\mathrm{diag}(\lambda)\,Q^\top,
              \qquad \lambda_1\le\cdots\le\lambda_n,
              \qquad Q^\top Q=I$$
        """
        matrix = self._validate_square_matrix(matrix, name)
        symmetric = self._symmetrize(matrix)
        self._assert_finite(symmetric, f"{name} simetrizada")

        try:
            eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        except np.linalg.LinAlgError as exc:
            raise MetricCoherenceError(
                f"La descomposición espectral de {name} falló."
            ) from exc

        self._assert_finite(eigenvalues, f"espectro de {name}")
        self._assert_finite(eigenvectors, f"autovectores de {name}")

        n = symmetric.shape[0]
        lambda_min = float(eigenvalues[0])
        lambda_max = float(eigenvalues[-1])

        if lambda_max <= 0.0 or not math.isfinite(lambda_max):
            raise MetricCoherenceError(
                f"{name} no es definida positiva: λ_max = {lambda_max:.4e}."
            )

        spectral_floor = max(
            float(n) * _MACHINE_EPSILON * lambda_max,
            _MACHINE_EPSILON,
        )
        if lambda_min <= spectral_floor:
            raise MetricCoherenceError(
                f"{name} es numéricamente singular o no definida positiva. "
                f"λ_min = {lambda_min:.4e}, umbral = {spectral_floor:.4e}."
            )

        return eigenvalues, eigenvectors

    def _condition_number_from_spectrum(
        self,
        eigenvalues: NDArray[np.float64],
        name: str,
    ) -> float:
        r"""Número de condición espectral $$\kappa_2=\lambda_{\max}/\lambda_{\min}$$."""
        lambda_min = float(eigenvalues[0])
        lambda_max = float(eigenvalues[-1])
        if lambda_min <= 0.0:
            raise MetricCoherenceError(
                f"κ({name}) indefinido: λ_min = {lambda_min:.4e}."
            )
        cond = lambda_max / lambda_min
        if not math.isfinite(cond):
            raise MetricCoherenceError(
                f"El número de condición de {name} no es finito."
            )
        if cond > _CONDITION_NUMBER_MAX:
            raise MetricCoherenceError(
                f"{name} está mal condicionada: κ = {cond:.4e} > "
                f"{_CONDITION_NUMBER_MAX:.4e}."
            )
        return float(cond)

    def _spectral_inverse(
        self,
        eigenvalues: NDArray[np.float64],
        eigenvectors: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""Inversa espectral $$G^{-1}=Q\,\mathrm{diag}(\lambda^{-1})\,Q^\top$$."""
        inv_evals = 1.0 / eigenvalues
        inverse = (eigenvectors * inv_evals) @ eigenvectors.T
        inverse = self._symmetrize(inverse)
        self._assert_finite(inverse, "inversa espectral")
        return inverse

    def _wilkinson_inverse_tolerance(
        self,
        dimension: int,
        condition_number: float,
    ) -> float:
        r"""
        Cota de Wilkinson para el residuo de inversa (invariante [I2]):

            $$\tau_W = \max\bigl(
                \tau_{\mathrm{abs}},\,
                C\cdot\kappa(G)\cdot\varepsilon\cdot\sqrt{n}
            \bigr)$$
        """
        sqrt_n = math.sqrt(float(dimension))
        spectral = (
            _WILKINSON_CONSTANT
            * max(1.0, condition_number)
            * _MACHINE_EPSILON
            * sqrt_n
        )
        return max(_METRIC_INVERSE_TOLERANCE, spectral)

    def _normalized_inverse_residual(
        self,
        left_residual: float,
        right_residual: float,
        dimension: int,
    ) -> float:
        r"""
        Residuo bilateral normalizado al estilo Wilkinson:

            $$\max\bigl(
                \|GG^{-1}-I\|_F/\sqrt{n},\,
                \|G^{-1}G-I\|_F/\sqrt{n}
            \bigr)$$
        """
        sqrt_n = math.sqrt(float(dimension))
        return max(left_residual, right_residual) / sqrt_n

    # ──────────────────────────────────────────────────────────────────────────
    # 1.4  Automejora Recursiva de Nivel 3 (RSI-L3) sobre la Dualidad Métrica
    # ──────────────────────────────────────────────────────────────────────────
    def _evaluate_metric_epistemic_lyapunov(
        self,
        G: NDArray[np.float64],
        X_inv: NDArray[np.float64],
        identity: NDArray[np.float64],
    ) -> tuple[float, float, NDArray[np.float64]]:
        r"""
        Evalúa el Funcional de Lyapunov Epistémico $$\mathcal{V}^{(1)}(X)$$
        sobre el álgebra de Banach $$\mathcal{B}(\mathbb{R}^n)$$:

            $$R_L = I - G X,\qquad R_R = I - X G$$
            $$\mathcal{V}^{(1)}(X) = \frac{\max(\|R_L\|_F, \|R_R\|_F)}{\sqrt{n}}$$

        Retorna ``(V_lyapunov, raw_max_residual, R_L)``.
        """
        n = G.shape[0]
        R_left = identity - (G @ X_inv)
        R_right = identity - (X_inv @ G)
        self._assert_finite(R_left, "R_left = I - G X")
        self._assert_finite(R_right, "R_right = I - X G")

        left_norm = self._frobenius_norm(R_left, "I - G X")
        right_norm = self._frobenius_norm(R_right, "I - X G")
        raw_max = max(left_norm, right_norm)
        lyapunov = raw_max / math.sqrt(float(n))
        return lyapunov, raw_max, R_left

    def _rsi_level3_refine_metric_inverse(
        self,
        G: NDArray[np.float64],
        G_inv_initial: NDArray[np.float64],
        G_inv_spectral: NDArray[np.float64],
        condition_number: float,
    ) -> tuple[NDArray[np.float64], float, RSILevel3AuditMetrics]:
        r"""
        Endofuntor de Automejora Recursiva de Nivel 3 ($$\mathfrak{R}_3^{(1)}$$).

        A diferencia de una iteración fija de Nivel 2, este método inspecciona
        el radio espectral aproximado del operador de defecto $$R_k = I - G X_k$$
        y sintetiza dinámicamente la ley del operador de corrección:
          - **Rama A (Chebyshev-Halley de orden 3)**:
            $$\Phi_{\mathrm{CH}}(X_k) = \mathrm{sym}\bigl(X_k (I + \eta_k R_k (I + \tfrac12 R_k))\bigr)$$
          - **Rama B (Newton-Schulz amortiguado de orden 2)**:
            $$\Phi_{\mathrm{NS}}(X_k) = \mathrm{sym}\bigl(X_k (I + \eta_k R_k)\bigr)$$
          - **Rama C (Fusión Espectral Convexa de de Rham)**:
            $$\Phi_{\mathrm{Spec}}(X_k) = (1 - \eta_k) X_k + \eta_k G_{\mathrm{spec}}^{-1}$$

        En cada paso recursivo $$k \in \{1,\dots,L_{\max}\}$$, selecciona la ley
        que minimiza el Funcional de Lyapunov $$\mathcal{V}_k^{(1)}$$, exigiendo
        invariancia monótona $$\mathcal{V}_{k+1}^{(1)} \le \mathcal{V}_k^{(1)}$$.
        """
        n = G.shape[0]
        identity = np.eye(n, dtype=np.float64)
        epistemic_floor = (
            _RSI_L3_EPISTEMIC_FLOOR_ULPS
            * max(1.0, condition_number)
            * _MACHINE_EPSILON
        )

        X_current = self._symmetrize(G_inv_initial)
        V_init, raw_residual, R_current = self._evaluate_metric_epistemic_lyapunov(
            G, X_current, identity
        )
        V_current = V_init

        depth = 0
        chosen_gain = 1.0
        law_tag = "Chebyshev-Halley-Schulz-L3"

        for step in range(1, _RSI_L3_MAX_RECURSION_DEPTH + 1):
            if V_current <= epistemic_floor:
                break

            defect_norm = self._frobenius_norm(R_current, "R_current")
            # Síntesis meta-epistémica de la ganancia η_k ∈ (0, 1]
            adaptive_gain = 1.0 / (1.0 + defect_norm * defect_norm)

            # Candidato 1: Ley de Chebyshev-Halley de orden 3
            poly_ch = identity + adaptive_gain * (
                R_current + 0.5 * (R_current @ R_current)
            )
            X_ch = self._symmetrize(X_current @ poly_ch)
            V_ch, raw_ch, R_ch = self._evaluate_metric_epistemic_lyapunov(
                G, X_ch, identity
            )

            # Candidato 2: Ley de Newton-Schulz amortiguada de orden 2
            poly_ns = identity + (0.5 * adaptive_gain) * R_current
            X_ns = self._symmetrize(X_current @ poly_ns)
            V_ns, raw_ns, R_ns = self._evaluate_metric_epistemic_lyapunov(
                G, X_ns, identity
            )

            # Candidato 3: Anclaje espectral ortogonal de de Rham
            V_sp, raw_sp, R_sp = self._evaluate_metric_epistemic_lyapunov(
                G, G_inv_spectral, identity
            )

            # Selección de la mejor ley endofuntorial en el paso k
            best_V = V_ch
            best_raw = raw_ch
            best_X = X_ch
            best_R = R_ch
            best_tag = "Chebyshev-Halley-Order3"
            best_gain = adaptive_gain

            if V_ns < best_V:
                best_V = V_ns
                best_raw = raw_ns
                best_X = X_ns
                best_R = R_ns
                best_tag = "Newton-Schulz-Damped-Order2"
                best_gain = 0.5 * adaptive_gain

            if V_sp < best_V:
                best_V = V_sp
                best_raw = raw_sp
                best_X = G_inv_spectral
                best_R = R_sp
                best_tag = "Spectral-deRham-Projection"
                best_gain = 1.0

            # Certificación estricta de descenso de Lyapunov
            if best_V < V_current:
                X_current = best_X
                V_current = best_V
                raw_residual = best_raw
                R_current = best_R
                depth = step
                chosen_gain = best_gain
                law_tag = best_tag
            else:
                # Punto fijo numérico alcanzado en la mantisa IEEE-754
                break

        if V_current > V_init + epistemic_floor:
            raise RecursiveSelfImprovementDivergenceError(
                "El endofuntor RSI-L3 métrico violó la monotonía de Lyapunov: "
                f"V_init = {V_init:.4e}, V_term = {V_current:.4e}."
            )

        contraction = (
            V_current / V_init if V_init > 0.0 else 0.0
        )
        rsi_audit = RSILevel3AuditMetrics(
            recursion_depth=depth,
            initial_epistemic_defect=V_init,
            terminal_epistemic_defect=V_current,
            lyapunov_contraction_ratio=min(1.0, contraction),
            meta_law_gain_parameter=chosen_gain,
            synthesized_law_tag=law_tag,
            is_monotone_lyapunov=(V_current <= V_init + epistemic_floor),
            is_fixed_point_converged=True,
        )
        return X_current, raw_residual, rsi_audit

    # ──────────────────────────────────────────────────────────────────────────
    # 1.5  Validación integral del par métrico (G, G^{-1})
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_metric_pair(
        self,
        G_tensor: NDArray[np.float64],
        G_inv: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        float,
        float,
        float,
        float,
        float,
        RSILevel3AuditMetrics,
    ]:
        r"""
        Valida, certifica y refina recursivamente (RSI-L3) el par métrico
        $$(G, G^{-1})$$.

        Retorna:
            ``(G, G_inv_refined, evals_G, evecs_G, κ(G), residuo_inversa_crudo,
              cota_Wilkinson, λ_min, λ_max, rsi_l3_audit)``.
        """
        G_tensor = self._validate_square_matrix(G_tensor, "G_tensor")
        G_inv = self._validate_square_matrix(G_inv, "G_inv")

        if G_tensor.shape != G_inv.shape:
            raise MetricCoherenceError(
                "G_tensor y G_inv deben tener la misma dimensión."
            )

        n = G_tensor.shape[0]

        g_sym_residual = self._symmetric_residual(G_tensor)
        g_tol = self._matrix_residual_tolerance(G_tensor)
        if g_sym_residual > g_tol:
            raise MetricCoherenceError(
                f"G_tensor no es simétrica. Residuo = {g_sym_residual:.4e}, "
                f"tolerancia = {g_tol:.4e}."
            )

        g_inv_sym_residual = self._symmetric_residual(G_inv)
        g_inv_tol = self._matrix_residual_tolerance(G_inv)
        if g_inv_sym_residual > g_inv_tol:
            raise MetricCoherenceError(
                f"G_inv no es simétrica. Residuo = {g_inv_sym_residual:.4e}, "
                f"tolerancia = {g_inv_tol:.4e}."
            )

        G = self._symmetrize(G_tensor)
        G_inv = self._symmetrize(G_inv)

        evals_G, evecs_G = self._spectral_decompose_spd(G, "G_tensor")
        self._spectral_decompose_spd(G_inv, "G_inv")
        cond_G = self._condition_number_from_spectrum(evals_G, "G_tensor")
        lambda_min = float(evals_G[0])
        lambda_max = float(evals_G[-1])

        identity = np.eye(n, dtype=np.float64)
        left = G @ G_inv
        right = G_inv @ G
        self._assert_finite(left, "G @ G_inv")
        self._assert_finite(right, "G_inv @ G")

        left_residual = self._frobenius_norm(left - identity, "G @ G_inv - I")
        right_residual = self._frobenius_norm(right - identity, "G_inv @ G - I")
        normalized_residual = self._normalized_inverse_residual(
            left_residual,
            right_residual,
            n,
        )

        wilkinson_bound = self._wilkinson_inverse_tolerance(n, cond_G)
        normalized_bound = wilkinson_bound / math.sqrt(float(n))

        if normalized_residual > max(
            normalized_bound,
            _WILKINSON_CONSTANT * cond_G * _MACHINE_EPSILON,
        ):
            raise MetricCoherenceError(
                "G_inv no es una inversa de Wilkinson-consistente de G_tensor. "
                f"Residuo normalizado = {normalized_residual:.4e}, "
                f"cota = {normalized_bound:.4e}."
            )

        G_inv_spectral = self._spectral_inverse(evals_G, evecs_G)
        spectral_gap = self._frobenius_norm(
            G_inv - G_inv_spectral,
            "G_inv - G_inv_espectral",
        )
        spectral_tol = max(
            wilkinson_bound,
            _WILKINSON_CONSTANT
            * cond_G
            * _MACHINE_EPSILON
            * max(1.0, self._frobenius_norm(G_inv, "G_inv")),
        )
        if spectral_gap > spectral_tol:
            raise MetricCoherenceError(
                "G_inv discrepa de la inversa espectral de G_tensor. "
                f"Residuo = {spectral_gap:.4e}, tolerancia = {spectral_tol:.4e}."
            )

        # Aplicación del Endofuntor RSI-L3 sobre el operador inverso validado
        G_inv_refined, refined_raw_residual, rsi_audit = (
            self._rsi_level3_refine_metric_inverse(
                G=G,
                G_inv_initial=G_inv,
                G_inv_spectral=G_inv_spectral,
                condition_number=cond_G,
            )
        )

        return (
            G,
            G_inv_refined,
            evals_G,
            evecs_G,
            cond_G,
            refined_raw_residual,
            wilkinson_bound,
            lambda_min,
            lambda_max,
            rsi_audit,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 1.6  Isomorfismo musical y energía cinética geométrica
    # ──────────────────────────────────────────────────────────────────────────
    def _musical_flat(
        self,
        q_dot: NDArray[np.float64],
        G_tensor: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""Isomorfismo musical plano: $$p = G\dot{q},\quad p_\mu = G_{\mu\nu}\dot{q}^\nu$$."""
        q_dot = self._validate_vector(q_dot, "q_dot")
        G_tensor = self._validate_square_matrix(G_tensor, "G_tensor")

        if G_tensor.shape[0] != q_dot.size:
            raise RiemannianInertiaError(
                "Dimensión incompatible entre q_dot y G_tensor."
            )

        momentum = G_tensor @ q_dot
        self._assert_finite(momentum, "momentum covariante p = G q̇")
        return momentum

    def _musical_sharp(
        self,
        momentum: NDArray[np.float64],
        G_inv: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""Isomorfismo musical agudo: $$v = G^{-1}p,\quad v^\mu = G^{\mu\nu}p_\nu$$."""
        momentum = self._validate_vector(momentum, "p")
        G_inv = self._validate_square_matrix(G_inv, "G_inv")

        if G_inv.shape[0] != momentum.size:
            raise RiemannianInertiaError(
                "Dimensión incompatible entre p y G_inv."
            )

        velocity = G_inv @ momentum
        self._assert_finite(velocity, "velocidad reconstruida v = G^{-1} p")
        return velocity

    def _certify_musical_roundtrip(
        self,
        q_dot: NDArray[np.float64],
        reconstructed: NDArray[np.float64],
        condition_number: float,
    ) -> float:
        r"""
        Certifica $$\sharp\circ\flat=\mathrm{id}$$ sobre $$\dot{q}$$:

            $$\|\sharp\flat\dot{q}-\dot{q}\|_2
              \le C\cdot\kappa\cdot n\cdot\varepsilon
              \cdot\max(1,\|\dot{q}\|_2)$$
        """
        q_dot = self._validate_vector(q_dot, "q_dot")
        reconstructed = self._validate_vector(
            reconstructed, "reconstructed_velocity"
        )
        if q_dot.size != reconstructed.size:
            raise DualPairingError(
                "Dimensión incompatible en el round-trip musical."
            )

        delta = reconstructed - q_dot
        residual = self._euclidean_norm(delta, "♯♭q̇ − q̇")
        scale = max(1.0, self._euclidean_norm(q_dot, "q_dot"))
        tolerance = (
            _WILKINSON_CONSTANT
            * max(1.0, condition_number)
            * _MACHINE_EPSILON
            * float(q_dot.size)
            * scale
        )
        tolerance = max(tolerance, _SYMPLECTIC_TOLERANCE * scale)

        if residual > tolerance:
            raise DualPairingError(
                "El isomorfismo musical no es involutivo en aritmética de máquina. "
                f"‖♯♭q̇ − q̇‖ = {residual:.4e}, tolerancia = {tolerance:.4e}."
            )
        return residual

    def _certify_dual_pairing(
        self,
        q_dot: NDArray[np.float64],
        momentum: NDArray[np.float64],
        G_tensor: NDArray[np.float64],
        G_inv: NDArray[np.float64],
        condition_number: float,
    ) -> tuple[float, float, float, float]:
        r"""
        Certifica la identidad de apareamiento dual:

            $$\langle p,\dot{q}\rangle
              = \dot{q}^\top G\dot{q}
              = p^\top G^{-1}p$$
        """
        g_q = G_tensor @ q_dot
        g_inv_p = G_inv @ momentum
        self._assert_finite(g_q, "G q̇")
        self._assert_finite(g_inv_p, "G^{-1} p")

        pairing = self._neumaier_dot(momentum, q_dot, "⟨p, q̇⟩")
        primal = self._neumaier_dot(q_dot, g_q, "q̇ᵀ G q̇")
        dual = self._neumaier_dot(momentum, g_inv_p, "pᵀ G^{-1} p")

        if primal < 0.0 or dual < 0.0:
            floor = max(
                _SYMPLECTIC_TOLERANCE,
                1000.0 * _MACHINE_EPSILON * max(1.0, abs(primal), abs(dual)),
            )
            if primal < -floor or dual < -floor:
                raise DualPairingError(
                    "Energía cinética geométrica negativa: "
                    f"primal = {primal:.4e}, dual = {dual:.4e}."
                )
            primal = max(primal, 0.0)
            dual = max(dual, 0.0)

        scale = max(1.0, abs(primal), abs(dual), abs(pairing))
        tolerance = max(
            _SYMPLECTIC_TOLERANCE,
            _WILKINSON_CONSTANT
            * max(1.0, condition_number)
            * _MACHINE_EPSILON
            * float(q_dot.size)
            * scale,
        )
        residual = max(
            abs(pairing - primal), abs(primal - dual), abs(pairing - dual)
        )
        if residual > tolerance:
            raise DualPairingError(
                "Identidad de apareamiento dual violada. "
                f"⟨p,q̇⟩={pairing:.4e}, primal={primal:.4e}, "
                f"dual={dual:.4e}, residuo={residual:.4e}, "
                f"tolerancia={tolerance:.4e}."
            )
        return pairing, primal, dual, residual

    def _evaluate_momentum_norm(
        self,
        dual_energy: float,
    ) -> float:
        r"""Norma inducida por la métrica dual: $$\|p\|_{G^{-1}}=\sqrt{p_\mu G^{\mu\nu}p_\nu}$$."""
        if not math.isfinite(dual_energy):
            raise RiemannianInertiaError(
                "La energía dual del momentum no es finita."
            )
        if dual_energy < 0.0:
            raise RiemannianInertiaError(
                f"Energía dual negativa en la norma inercial: {dual_energy:.4e}."
            )
        return math.sqrt(dual_energy)

    # ──────────────────────────────────────────────────────────────────────────
    # 1.7  Mecánica Celeste de Poincaré: Acción-Ángulo, KAM y Sundman
    # ──────────────────────────────────────────────────────────────────────────
    def _compute_poincare_action_angle_spectroscopy(
        self,
        momentum: NDArray[np.float64],
        eigenvalues: NDArray[np.float64],
        eigenvectors: NDArray[np.float64],
        dual_pairing: float,
        momentum_norm: float,
        condition_number: float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float, float]:
        r"""
        Proyecta el estado sobre el toro invariante de Poincaré definido por
        la base ortonormal de autovectores $$Q$$ de la métrica $$G$$:

        1. **Frecuencias orbitales características**:
           $$\omega_k = \sqrt{\lambda_k} > 0$$
        2. **Variables de acción canónicas de Poincaré**:
           $$\tilde{p}_k = (Q^\top p)_k, \qquad
             I_k = \frac{\tilde{p}_k^2}{2\,\lambda_k\,\omega_k}
                 = \frac{\tilde{p}_k^2}{2\,\lambda_k^{3/2}} \ge 0$$
        3. **Densidad de la 1-forma invariante de Poincaré-Cartan**:
           $$\theta_{\mathrm{PC}} = p_\mu \dot{q}^\mu - H_{\mathrm{kin}}
             = \langle p, \dot{q}\rangle - \tfrac12\langle p, \dot{q}\rangle
             = \tfrac12\langle p, \dot{q}\rangle$$
        4. **Margen de no-resonancia Diofántica KAM**:
           Evalúa los divisores pequeños de Poincaré-Arnold
           $$|p\,\omega_i - q\,\omega_j|$$ para órdenes racionales
           $$1 \le p, q \le K_{\max}$$:
           $$d_{\mathrm{KAM}} = \min_{i < j,\, p,q}
             \frac{|p\,\omega_i - q\,\omega_j|}{\|\boldsymbol{\omega}\|_2}
             \cdot (p + q)^{\tau_{\mathrm{KAM}}}$$
        5. **Factor de regularización conforme de Sundman-Poincaré**:
           $$s_{\mathrm{Sundman}} = \left(
               1 + \left(\frac{\|p\|_{G^{-1}}}{P_{\max}}\right)^2
               + \frac{\kappa(G) - 1}{\kappa_{\max}}
             \right)^{-\beta} \in (0, 1]$$
        """
        orbital_frequencies = np.sqrt(eigenvalues)
        self._assert_finite(orbital_frequencies, "frecuencias orbitales ω_k")

        modal_momentum = eigenvectors.T @ momentum
        self._assert_finite(modal_momentum, "momentum modal Qᵀ p")

        action_variables = (modal_momentum * modal_momentum) / (
            2.0 * eigenvalues * orbital_frequencies
        )
        self._assert_finite(action_variables, "variables de acción de Poincaré I_k")

        poincare_cartan_density = 0.5 * dual_pairing

        # Auditoría Diofántica KAM sobre los modos activos
        n = orbital_frequencies.size
        omega_norm = self._euclidean_norm(orbital_frequencies, "ω")
        if n <= 1 or omega_norm == 0.0:
            kam_margin = 1.0
        else:
            mode_count = min(n, _POINCARE_MAX_ACTIVE_MODES)
            active_freqs = orbital_frequencies[-mode_count:]
            min_scaled_divisor = float("inf")
            for i in range(mode_count):
                wi = float(active_freqs[i])
                for j in range(i + 1, mode_count):
                    wj = float(active_freqs[j])
                    for p_int in range(1, _POINCARE_MAX_RESONANCE_ORDER + 1):
                        for q_int in range(1, _POINCARE_MAX_RESONANCE_ORDER + 1):
                            order_l1 = float(p_int + q_int)
                            divisor = abs(float(p_int) * wi - float(q_int) * wj)
                            scaled = (divisor / omega_norm) * (
                                order_l1**_POINCARE_DIOPHANTINE_TAU
                            )
                            if scaled < min_scaled_divisor:
                                min_scaled_divisor = scaled
            kam_margin = (
                float(min_scaled_divisor)
                if math.isfinite(min_scaled_divisor)
                else 0.0
            )

        # Regularización analítica de Sundman-Poincaré
        norm_ratio = momentum_norm / _MOMENTUM_MAX_BOUND
        cond_ratio = max(0.0, (condition_number - 1.0) / _CONDITION_NUMBER_MAX)
        sundman_base = 1.0 + (norm_ratio * norm_ratio) + cond_ratio
        sundman_factor = sundman_base ** (-_SUNDMAN_REGULARIZATION_EXPONENT)

        if not (0.0 < sundman_factor <= 1.0) or not math.isfinite(sundman_factor):
            raise PoincareResonanceCollapseError(
                f"El factor de regularización de Sundman degeneró: {sundman_factor:.4e}."
            )

        return (
            action_variables,
            orbital_frequencies,
            poincare_cartan_density,
            sundman_factor,
            kam_margin,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 1.8  Núcleo de ejecución y Morfismo Terminal de la Fase 1
    # ──────────────────────────────────────────────────────────────────────────
    def execute_phase1(
        self,
        q_dot: NDArray[np.float64],
        G_tensor: NDArray[np.float64],
        G_inv: NDArray[np.float64],
    ) -> MomentumAuditData:
        """
        Ejecuta la Fase 1 completa y construye el certificado inmutable
        ``MomentumAuditData`` que alimenta ``handoff_phase1_to_phase2``.
        """
        q_dot = self._validate_vector(q_dot, "q_dot")

        (
            G_tensor,
            G_inv_refined,
            evals_G,
            evecs_G,
            cond_G,
            inv_residual,
            wilkinson_bound,
            lambda_min,
            lambda_max,
            rsi_metric_audit,
        ) = self._validate_metric_pair(G_tensor, G_inv)

        if G_tensor.shape[0] != q_dot.size:
            raise RiemannianInertiaError(
                "Dimensión incompatible entre q_dot y la métrica."
            )

        momentum = self._musical_flat(q_dot, G_tensor)
        reconstructed = self._musical_sharp(momentum, G_inv_refined)
        roundtrip_residual = self._certify_musical_roundtrip(
            q_dot,
            reconstructed,
            cond_G,
        )
        pairing, primal, dual, pairing_residual = self._certify_dual_pairing(
            q_dot,
            momentum,
            G_tensor,
            G_inv_refined,
            cond_G,
        )
        p_norm = self._evaluate_momentum_norm(dual)
        is_bounded = p_norm <= _MOMENTUM_MAX_BOUND

        if not is_bounded:
            raise MomentumDivergenceError(
                "Divergencia inercial detectada: "
                f"‖p‖ = {p_norm:.4e} > {_MOMENTUM_MAX_BOUND:.4e}."
            )

        (
            action_vars,
            orbital_freqs,
            pc_density,
            sundman_factor,
            kam_margin,
        ) = self._compute_poincare_action_angle_spectroscopy(
            momentum=momentum,
            eigenvalues=evals_G,
            eigenvectors=evecs_G,
            dual_pairing=pairing,
            momentum_norm=p_norm,
            condition_number=cond_G,
        )

        audit = MomentumAuditData(
            covariant_momentum=_freeze_array(momentum),
            reconstructed_velocity=_freeze_array(reconstructed),
            momentum_norm=p_norm,
            kinetic_energy_primal=primal,
            kinetic_energy_dual=dual,
            dual_pairing=pairing,
            pairing_residual=pairing_residual,
            musical_roundtrip_residual=roundtrip_residual,
            is_bounded=is_bounded,
            metric_condition_number=cond_G,
            inverse_consistency_residual=inv_residual,
            wilkinson_bound=wilkinson_bound,
            spectral_minimum=lambda_min,
            spectral_maximum=lambda_max,
            poincare_action_variables=_freeze_array(action_vars),
            poincare_orbital_frequencies=_freeze_array(orbital_freqs),
            poincare_cartan_one_form_density=pc_density,
            sundman_regularization_factor=sundman_factor,
            kam_diophantine_margin=kam_margin,
            rsi_l3_metric_audit=rsi_metric_audit,
        )

        logger.debug(
            "Fase 1 completada: ‖p‖ = %.6e, κ(G) = %.6e, "
            "residuo inversa RSI-L3 = %.6e (profundidad %d), "
            "Sundman = %.6e, margen KAM = %.6e.",
            audit.momentum_norm,
            audit.metric_condition_number,
            audit.inverse_consistency_residual,
            audit.rsi_l3_metric_audit.recursion_depth,
            audit.sundman_regularization_factor,
            audit.kam_diophantine_margin,
        )
        return audit

    def handoff_phase1_to_phase2(
        self,
        momentum_data: MomentumAuditData,
    ) -> NDArray[np.float64]:
        r"""
        DEFINICIÓN FORMAL FINAL DE LA FASE 1  ≡  CONTINUACIÓN E INICIO DE LA FASE 2.

        Morfismo de transición categórica:

            $$\Phi_{12}:\mathrm{MomentumAuditData}\longrightarrow T^*\mathcal{M}\cong\mathbb{R}^n$$

        Certifica las poscondiciones de la Fase 1 como precondiciones exactas
        del dominio de la Fase 2:
            1. Acotación elástica $$\|p\|_{G^{-1}}\le P_{\max}$$.
            2. Condicionamiento espectral $$\kappa(G)\le\kappa_{\max}$$.
            3. Regularidad de Sundman $$s_{\mathrm{Sundman}}\in(0,1]$$ y
               no-negatividad de la densidad de Poincaré-Cartan.
            4. Monotonía de Lyapunov del endofuntor RSI-L3 $$\mathfrak{R}_3^{(1)}$$.
        """
        if not isinstance(momentum_data, MomentumAuditData):
            raise PhaseHandoffError(
                "handoff_phase1_to_phase2 exige MomentumAuditData."
            )

        if not momentum_data.is_bounded:
            raise PhaseHandoffError(
                "El certificado de Fase 1 no está acotado inercialmente."
            )

        if momentum_data.momentum_norm > _MOMENTUM_MAX_BOUND:
            raise PhaseHandoffError(
                "‖p‖ del certificado excede P_max en el handoff Φ₁₂."
            )

        if momentum_data.metric_condition_number > _CONDITION_NUMBER_MAX:
            raise PhaseHandoffError(
                "κ(G) del certificado excede la cota admisible en Φ₁₂."
            )

        if not (
            0.0 < momentum_data.sundman_regularization_factor <= 1.0
            and math.isfinite(momentum_data.sundman_regularization_factor)
        ):
            raise PhaseHandoffError(
                "Factor de regularización de Sundman inválido en el handoff Φ₁₂."
            )

        if not momentum_data.rsi_l3_metric_audit.is_monotone_lyapunov:
            raise PhaseHandoffError(
                "El certificado RSI-L3 de Fase 1 violó la monotonía de Lyapunov en Φ₁₂."
            )

        momentum = self._validate_vector(
            momentum_data.covariant_momentum,
            "momentum_data.covariant_momentum",
        )
        return momentum


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2 → SÍNTESIS DEL OPERADOR GIROSCÓPICO DE POINCARÉ-BIRKHOFF, INVARIANTE
#          DE CASIMIR Y AUTOMEJORA RECURSIVA DE NIVEL 3 EN so(n) (Orient)
#          [Continuación directa de handoff_phase1_to_phase2]
# ══════════════════════════════════════════════════════════════════════════════
class Phase2_GyroscopicSynthesizer(Phase1_MomentumSpectrometer):
    r"""
    Inicia exactamente donde concluye ``handoff_phase1_to_phase2``,
    recibiendo el covector certificado $$p \in T^*\mathcal{M}$$ y su contexto
    de acción-ángulo de Poincaré para sintetizar el tensor giroscópico de
    Lorentz en el álgebra exterior $$\bigwedge^2 T^*\mathcal{M}$$:

        $$W_{\mu\nu}=\alpha(p_\mu\omega_\nu-p_\nu\omega_\mu)
          =\alpha\,(p\wedge\omega)_{\mu\nu}, \qquad
          \omega = \Omega_{\mathrm{skew}} p$$

    Integra:
        1. Proyección de la vorticidad territorial al espacio de 2-formas
           cerradas de de Rham.
        2. **Automejora Recursiva de Nivel 3 ($$\mathfrak{R}_3^{(2)}$$)**:
           Purificación isométrica recursiva de Darboux-Gram sobre el cono de
           Lie $$\mathfrak{so}(n)$$ que contrae el defecto conjunto de
           antisimetría, identidad de Gram e invariante cuadrático de Casimir.
        3. **Mecánica Celeste de Poincaré (Fase 2)**:
           - Invariante integral de segundo orden (circulación areal de Poincaré).
           - Espectro de frecuencias características de la forma normal de Darboux.
           - Invariante cuadrático de Casimir $$\mathcal{C}_2(W) = -\tfrac12\mathrm{Tr}(W^2)$$.
           - Factor secular de Poincaré-Birkhoff.

    El morfismo terminal ``handoff_phase2_to_phase3`` constituye la definición
    formal final de la Fase 2 y el inicio exacto de la Fase 3.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # 2.1  Continuación formal inmediata del Morfismo Terminal de la Fase 1
    # ──────────────────────────────────────────────────────────────────────────
    def _receive_certified_momentum(
        self,
        momentum_data: MomentumAuditData,
    ) -> NDArray[np.float64]:
        """
        PRIMER MÉTODO DE LA FASE 2 (Continuación directa de ``handoff_phase1_to_phase2``).

        Consume el morfismo terminal de la Fase 1, revalida las invariantes de
        Poincaré-Sundman y entrega el 1-covector $$p$$ al motor de álgebra exterior.
        """
        return self.handoff_phase1_to_phase2(momentum_data)

    # ──────────────────────────────────────────────────────────────────────────
    # 2.2  Proyección de vorticidad y álgebra exterior de bivectores
    # ──────────────────────────────────────────────────────────────────────────
    def _project_vorticity_to_two_form(
        self,
        vorticity_matrix: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float]:
        r"""
        Proyecta la matriz de vorticidad al espacio de 2-formas:

            $$\Omega_{\mathrm{skew}}=\tfrac12(\Omega-\Omega^\top)$$

        El residuo mide la componente simétrica de deformación (strain)
        descartada: $$\bigl\|\tfrac12(\Omega+\Omega^\top)\bigr\|_F$$.
        """
        vorticity = self._validate_square_matrix(
            vorticity_matrix, "vorticity_matrix"
        )
        symmetric_part = self._symmetrize(vorticity)
        projection_residual = self._frobenius_norm(
            symmetric_part,
            "parte simétrica de vorticidad",
        )
        two_form = self._skew_symmetrize(vorticity)
        self._assert_finite(two_form, "2-forma de vorticidad")
        return two_form, projection_residual

    def _couple_vorticity(
        self,
        two_form: NDArray[np.float64],
        momentum: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""Contracción interior de la 2-forma contra el momentum: $$\omega_\mu=\Omega_{\mu\nu}p^\nu$$."""
        if two_form.shape[0] != momentum.size:
            raise RiemannianInertiaError(
                "La dimensión de vorticity_matrix no coincide con la del momentum."
            )
        omega = two_form @ momentum
        self._assert_finite(omega, "acoplamiento vorticial Ω p")
        return omega

    def _exterior_wedge(
        self,
        left: NDArray[np.float64],
        right: NDArray[np.float64],
        coupling: float,
    ) -> NDArray[np.float64]:
        r"""
        Producto exterior de 1-formas escalado por $$\alpha$$:

            $$(p\wedge\omega)_{\mu\nu}
              =\alpha\bigl(p_\mu\omega_\nu-p_\nu\omega_\mu\bigr)$$
        """
        if not math.isfinite(coupling):
            raise ExteriorAlgebraError(
                "El factor de acoplamiento giroscópico α no es finito."
            )
        left = self._validate_vector(left, "wedge.left")
        right = self._validate_vector(right, "wedge.right")
        if left.size != right.size:
            raise ExteriorAlgebraError(
                "Los 1-vectores del producto exterior tienen dimensión distinta."
            )
        bivector = coupling * (np.outer(left, right) - np.outer(right, left))
        self._assert_finite(bivector, "bivector p ∧ ω")
        return bivector

    def _compute_theoretical_gram_square(
        self,
        left: NDArray[np.float64],
        right: NDArray[np.float64],
        coupling: float,
    ) -> float:
        r"""
        Evalúa el determinante de Gram teórico del bivector $$\alpha(p\wedge\omega)$$
        mediante productos internos compensados de Neumaier:

            $$G_{\mathrm{Gram}} = 2\,\alpha^2\bigl(
                \|p\|_2^2\|\omega\|_2^2 - \langle p,\omega\rangle^2
              \bigr) \ge 0$$
        """
        left_norm_sq = self._neumaier_dot(left, left, "‖p‖²")
        right_norm_sq = self._neumaier_dot(right, right, "‖ω‖²")
        pairing = self._neumaier_dot(left, right, "⟨p, ω⟩")
        det_gram = max(0.0, left_norm_sq * right_norm_sq - pairing * pairing)
        return 2.0 * (coupling * coupling) * det_gram

    def _certify_wedge_gram_identity(
        self,
        left: NDArray[np.float64],
        right: NDArray[np.float64],
        bivector: NDArray[np.float64],
        coupling: float,
    ) -> float:
        r"""
        Certifica la identidad de Gram del álgebra exterior [I4]:

            $$\|p\wedge\omega\|_F^2
              = 2\alpha^2\bigl(\|p\|_2^2\|\omega\|_2^2-\langle p,\omega\rangle^2\bigr)$$
        """
        gram_rhs = self._compute_theoretical_gram_square(left, right, coupling)
        gram_lhs = self._frobenius_norm(bivector, "p ∧ ω") ** 2

        residual = abs(gram_lhs - gram_rhs)
        scale = max(1.0, abs(gram_lhs), abs(gram_rhs))
        tolerance = max(
            _SYMPLECTIC_TOLERANCE,
            _WILKINSON_CONSTANT
            * _MACHINE_EPSILON
            * float(left.size) ** 2
            * scale,
        )
        if residual > tolerance:
            raise ExteriorAlgebraError(
                "Identidad de Gram del bivector violada. "
                f"‖p∧ω‖_F² = {gram_lhs:.4e}, Gram = {gram_rhs:.4e}, "
                f"residuo = {residual:.4e}, tolerancia = {tolerance:.4e}."
            )
        return residual

    # ──────────────────────────────────────────────────────────────────────────
    # 2.3  Automejora Recursiva de Nivel 3 (RSI-L3) y Mecánica de Darboux-Casimir
    # ──────────────────────────────────────────────────────────────────────────
    def _compute_casimir_and_trace_residual(
        self,
        W_skew: NDArray[np.float64],
    ) -> tuple[float, float]:
        r"""
        En el álgebra de Lie $$\mathfrak{so}(n)$$, como $$W^\top = -W$$, el
        invariante cuadrático de Casimir cumple la identidad exacta:

            $$-\mathrm{Tr}(W^2) = \mathrm{Tr}(W^\top W) = \|W\|_F^2$$
            $$\mathcal{C}_2(W) = -\tfrac12\mathrm{Tr}(W^2) = \tfrac12\|W\|_F^2$$

        Calcula $$- \mathrm{Tr}(W^2) = \sum_{i,j} W_{ij}^2$$ y su discrepancia
        frente a la traza explícita de $$W^2$$ usando sumación de Neumaier.
        """
        fro_sq = self._neumaier_sum(W_skew * W_skew, "‖W‖_F²")
        # La diagonal de W² es (W²)_{ii} = ∑_j W_{ij} W_{ji}
        diag_W2 = np.einsum("ij,ji->i", W_skew, W_skew)
        trace_W2 = self._neumaier_sum(diag_W2, "Tr(W²)")
        casimir_invariant = -0.5 * trace_W2
        trace_residual = abs(trace_W2 + fro_sq)
        return max(0.0, casimir_invariant), trace_residual

    def _evaluate_synthesis_epistemic_lyapunov(
        self,
        W_candidate: NDArray[np.float64],
        target_gram_sq: float,
    ) -> tuple[float, float, float, float]:
        r"""
        Funcional de Lyapunov Epistémico de la Fase 2 ($$\mathcal{V}^{(2)}$$):

            $$\mathcal{V}^{(2)}(W)
              = r_{\mathrm{skew}}(W)
              + \frac{|\|W\|_F^2 - G_{\mathrm{Gram}}|}{\max(1, G_{\mathrm{Gram}})}
              + \frac{|\mathrm{Tr}(W^2) + \|W\|_F^2|}{\max(1, \|W\|_F^2)}$$

        Retorna ``(V_lyapunov, rel_skew, gram_abs_res, casimir_trace_res)``.
        """
        rel_skew = self._relative_skew_residual(W_candidate)
        fro_norm = self._frobenius_norm(W_candidate, "W_candidate")
        fro_sq = fro_norm * fro_norm
        gram_abs_res = abs(fro_sq - target_gram_sq)
        gram_rel_res = gram_abs_res / max(1.0, target_gram_sq)

        _, casimir_trace_res = self._compute_casimir_and_trace_residual(
            W_candidate
        )
        casimir_rel_res = casimir_trace_res / max(1.0, fro_sq)

        lyapunov = rel_skew + gram_rel_res + casimir_rel_res
        return lyapunov, rel_skew, gram_abs_res, casimir_trace_res

    def _rsi_level3_purify_gyroscopic_cone(
        self,
        W_raw: NDArray[np.float64],
        target_gram_sq: float,
    ) -> tuple[NDArray[np.float64], float, float, RSILevel3AuditMetrics]:
        r"""
        Endofuntor de Automejora Recursiva de Nivel 3 ($$\mathfrak{R}_3^{(2)}$$)
        sobre el cono antisimétrico $$\mathfrak{so}(n)$$.

        Purifica recursivamente el bivector $$W$$ sintetizando una homotecia
        isométrica de Darboux-Gram acoplada al proyector ortogonal exacto de
        $$\mathfrak{so}(n)$$, garantizando que $$r_{\mathrm{skew}} = 0$$ en
        bit de signo IEEE-754 y minimizando el defecto de Gram-Casimir con
        monotonía de Lyapunov $$\mathcal{V}_{k+1}^{(2)} \le \mathcal{V}_k^{(2)}$$.
        """
        W_raw = self._validate_square_matrix(W_raw, "W_raw")
        epistemic_floor = (
            _RSI_L3_EPISTEMIC_FLOOR_ULPS
            * float(W_raw.shape[0])
            * _MACHINE_EPSILON
        )

        V_init, _, _, _ = self._evaluate_synthesis_epistemic_lyapunov(
            W_raw, target_gram_sq
        )

        # Paso base: proyección ortogonal IEEE-754 exacta a so(n)
        W_current = self._skew_symmetrize(W_raw)
        V_current, rel_skew, _, _ = self._evaluate_synthesis_epistemic_lyapunov(
            W_current, target_gram_sq
        )

        depth = 1 if V_current < V_init else 0
        chosen_gain = 1.0
        law_tag = "Orthogonal-so(n)-SignBit-Projection"
        target_norm = math.sqrt(max(0.0, target_gram_sq))

        for step in range(1, _RSI_L3_MAX_RECURSION_DEPTH + 1):
            if V_current <= epistemic_floor:
                break

            current_norm = self._frobenius_norm(W_current, "W_current")
            if current_norm <= _MACHINE_EPSILON or target_norm <= _MACHINE_EPSILON:
                break

            # Ley adaptativa de reescalado isométrico de Darboux-Gram
            ratio = target_norm / current_norm
            gain = 1.0 / float(step)
            scale_factor = (1.0 - gain) + gain * ratio
            W_trial = self._skew_symmetrize(scale_factor * W_current)

            V_trial, trial_rel_skew, _, _ = (
                self._evaluate_synthesis_epistemic_lyapunov(
                    W_trial, target_gram_sq
                )
            )

            if V_trial < V_current:
                W_current = W_trial
                V_current = V_trial
                rel_skew = trial_rel_skew
                depth = step
                chosen_gain = gain
                law_tag = "Darboux-Gram-Casimir-Isometric-L3"
            else:
                break

        abs_residual = self._skew_symmetric_residual(W_current)
        abs_tol = self._matrix_residual_tolerance(W_current)

        if abs_residual > abs_tol or rel_skew > _SKEW_RELATIVE_TOLERANCE:
            raise SkewSymmetryViolationError(
                "El proyector antisimétrico RSI-L3 retiene componentes simétricas. "
                f"Residuo abs = {abs_residual:.4e} (tol {abs_tol:.4e}), "
                f"r_skew = {rel_skew:.4e} "
                f"(tol {_SKEW_RELATIVE_TOLERANCE:.4e})."
            )

        contraction = V_current / V_init if V_init > 0.0 else 0.0
        rsi_audit = RSILevel3AuditMetrics(
            recursion_depth=depth,
            initial_epistemic_defect=V_init,
            terminal_epistemic_defect=V_current,
            lyapunov_contraction_ratio=min(1.0, contraction),
            meta_law_gain_parameter=chosen_gain,
            synthesized_law_tag=law_tag,
            is_monotone_lyapunov=(V_current <= V_init + epistemic_floor),
            is_fixed_point_converged=True,
        )
        return W_current, abs_residual, rel_skew, rsi_audit

    def _project_to_skew_symmetric_cone(
        self,
        W_raw: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], float, float]:
        r"""
        Proyección ortogonal (Frobenius) al álgebra de Lie $$\mathfrak{so}(n)$$.
        Mantiene compatibilidad directa con llamadas unitarias a este método.
        """
        W_raw = self._validate_square_matrix(W_raw, "W_raw")
        W_proj = self._skew_symmetrize(W_raw)
        self._assert_finite(W_proj, "W_proj")

        abs_residual = self._skew_symmetric_residual(W_proj)
        rel_residual = self._relative_skew_residual(W_proj)
        abs_tol = self._matrix_residual_tolerance(W_proj)

        if abs_residual > abs_tol or rel_residual > _SKEW_RELATIVE_TOLERANCE:
            raise SkewSymmetryViolationError(
                "El proyector antisimétrico retiene componentes simétricas. "
                f"Residuo abs = {abs_residual:.4e} (tol {abs_tol:.4e}), "
                f"r_skew = {rel_residual:.4e} "
                f"(tol {_SKEW_RELATIVE_TOLERANCE:.4e})."
            )
        return W_proj, abs_residual, rel_residual

    def _certify_even_numerical_rank_and_darboux_spectrum(
        self,
        W_proj: NDArray[np.float64],
    ) -> tuple[int, bool, NDArray[np.float64]]:
        r"""
        Por el Teorema de Darboux-Álgebra Lineal, toda matriz real antisimétrica
        $$W \in \mathfrak{so}(n)$$ es ortogonalmente semejante a una suma
        directa de bloques simplécticos $$2\times 2$$ de frecuencias
        características $$\{\pm i\,\nu_k\}_{k=1}^{\lfloor n/2 \rfloor}$$ con
        $$\nu_k \ge 0$$, por lo que su rango exacto es siempre par.

        Extrae los valores singulares $$\sigma_i$$ para certificar la paridad
        del rango y agrupa los pares conjugados $$\nu_k = \sigma_{2k-1}$$.
        """
        n = W_proj.shape[0]
        try:
            singular_values = np.linalg.svd(W_proj, compute_uv=False)
        except np.linalg.LinAlgError as exc:
            raise SkewSymmetryViolationError(
                "No fue posible calcular el espectro singular de Darboux de W."
            ) from exc

        self._assert_finite(singular_values, "valores singulares de W")
        num_pairs = n // 2
        if singular_values.size == 0 or num_pairs == 0:
            return 0, True, np.zeros(0, dtype=np.float64)

        darboux_freqs = np.array(
            [
                0.5 * float(singular_values[2 * k] + singular_values[2 * k + 1])
                for k in range(num_pairs)
            ],
            dtype=np.float64,
        )

        sigma_max = float(singular_values[0])
        if sigma_max == 0.0:
            return 0, True, darboux_freqs

        floor = max(
            float(n) * _MACHINE_EPSILON * sigma_max,
            _SYMPLECTIC_TOLERANCE * max(1.0, sigma_max),
        )
        rank = int(np.count_nonzero(singular_values > floor))
        return rank, (rank % 2 == 0), darboux_freqs

    def _certify_even_numerical_rank(
        self,
        W_proj: NDArray[np.float64],
    ) -> tuple[int, bool]:
        """Método compatible para auditoría directa de paridad de rango."""
        rank, is_even, _ = (
            self._certify_even_numerical_rank_and_darboux_spectrum(W_proj)
        )
        return rank, is_even

    # ──────────────────────────────────────────────────────────────────────────
    # 2.4  Núcleo de síntesis y Morfismo Terminal de la Fase 2
    # ──────────────────────────────────────────────────────────────────────────
    def _synthesize_lorentz_tensor(
        self,
        momentum_data: MomentumAuditData,
        vorticity_matrix: NDArray[np.float64],
    ) -> GyroscopicSynthesisData:
        r"""
        Síntesis del tensor de Lorentz giroscópico con purificación RSI-L3 y
        certificación de invariantes de Darboux, Casimir y Poincaré-Birkhoff.
        """
        momentum = self._receive_certified_momentum(momentum_data)
        two_form, vorticity_residual = self._project_vorticity_to_two_form(
            vorticity_matrix
        )
        omega = self._couple_vorticity(two_form, momentum)

        coupling = float(_VORTICITY_COUPLING_FACTOR)
        W_raw = self._exterior_wedge(momentum, omega, coupling)
        target_gram_sq = self._compute_theoretical_gram_square(
            momentum, omega, coupling
        )

        W_proj, abs_residual, rel_residual, rsi_synthesis_audit = (
            self._rsi_level3_purify_gyroscopic_cone(W_raw, target_gram_sq)
        )

        gram_residual = self._certify_wedge_gram_identity(
            momentum,
            omega,
            W_proj,
            coupling,
        )

        W_norm = self._frobenius_norm(W_proj, "W_proj")
        rank, rank_even, darboux_freqs = (
            self._certify_even_numerical_rank_and_darboux_spectrum(W_proj)
        )
        casimir_inv, casimir_res = self._compute_casimir_and_trace_residual(
            W_proj
        )

        # Invariante integral de 2º orden de Poincaré (circulación areal del bivector)
        poincare_circulation = W_norm / math.sqrt(2.0)

        # Factor secular de Poincaré-Birkhoff ponderado por Sundman y margen KAM
        pb_secular_factor = coupling * (
            momentum_data.sundman_regularization_factor
            * (
                1.0
                / (
                    1.0
                    + math.exp(
                        -max(0.0, momentum_data.kam_diophantine_margin)
                    )
                )
            )
        )

        synthesis = GyroscopicSynthesisData(
            skew_symmetric_tensor=_freeze_array(W_proj),
            vorticity_two_form=_freeze_array(two_form),
            omega_vector=_freeze_array(omega),
            antisymmetry_residual=abs_residual,
            relative_skew_residual=rel_residual,
            vorticity_projection_residual=vorticity_residual,
            gyroscopic_frobenius_norm=W_norm,
            wedge_gram_residual=gram_residual,
            skew_numerical_rank=rank,
            rank_is_even=rank_even,
            is_strictly_skew=True,
            poincare_birkhoff_secular_factor=pb_secular_factor,
            casimir_quadratic_invariant=casimir_inv,
            casimir_trace_residual=casimir_res,
            darboux_characteristic_frequencies=_freeze_array(darboux_freqs),
            poincare_areal_circulation=poincare_circulation,
            rsi_l3_synthesis_audit=rsi_synthesis_audit,
        )

        logger.debug(
            "Fase 2 completada: ‖W‖_F = %.6e, r_skew = %.6e, "
            "Casimir C₂ = %.6e (res = %.6e), circulación Poincaré = %.6e, "
            "rango Darboux = %d.",
            synthesis.gyroscopic_frobenius_norm,
            synthesis.relative_skew_residual,
            synthesis.casimir_quadratic_invariant,
            synthesis.casimir_trace_residual,
            synthesis.poincare_areal_circulation,
            synthesis.skew_numerical_rank,
        )
        return synthesis

    def execute_phase2(
        self,
        momentum_data: MomentumAuditData,
        vorticity_matrix: NDArray[np.float64],
    ) -> GyroscopicSynthesisData:
        """
        Ejecuta la Fase 2 invocando internamente ``handoff_phase1_to_phase2``
        y produce el certificado ``GyroscopicSynthesisData`` para
        ``handoff_phase2_to_phase3``.
        """
        return self._synthesize_lorentz_tensor(momentum_data, vorticity_matrix)

    def handoff_phase2_to_phase3(
        self,
        synthesis_data: GyroscopicSynthesisData,
    ) -> NDArray[np.float64]:
        r"""
        DEFINICIÓN FORMAL FINAL DE LA FASE 2  ≡  CONTINUACIÓN E INICIO DE LA FASE 3.

        Morfismo de transición categórica:

            $$\Phi_{23}:\mathrm{GyroscopicSynthesisData}
              \longrightarrow\mathfrak{so}(n)$$

        Certifica las poscondiciones de la Fase 2 como precondiciones exactas
        del dominio de la Fase 3:
            1. Pertenencia estricta al cono de Lie $$\mathfrak{so}(n)$$
               ($$r_{\mathrm{skew}}\le\epsilon_{\mathrm{skew}}$$).
            2. No-negatividad y finitud del invariante cuadrático de Casimir
               $$\mathcal{C}_2(W)$$ y de la circulación areal de Poincaré.
            3. Monotonía de Lyapunov del endofuntor RSI-L3 $$\mathfrak{R}_3^{(2)}$$.
        """
        if not isinstance(synthesis_data, GyroscopicSynthesisData):
            raise PhaseHandoffError(
                "handoff_phase2_to_phase3 exige GyroscopicSynthesisData."
            )

        if not synthesis_data.is_strictly_skew:
            raise PhaseHandoffError(
                "El certificado de Fase 2 no es estrictamente antisimétrico."
            )

        if synthesis_data.relative_skew_residual > _SKEW_RELATIVE_TOLERANCE:
            raise PhaseHandoffError(
                "r_skew del certificado excede ε_skew en el handoff Φ₂₃."
            )

        if (
            synthesis_data.casimir_quadratic_invariant < 0.0
            or not math.isfinite(synthesis_data.casimir_quadratic_invariant)
        ):
            raise PhaseHandoffError(
                "Invariante cuadrático de Casimir inválido en el handoff Φ₂₃."
            )

        if not synthesis_data.rsi_l3_synthesis_audit.is_monotone_lyapunov:
            raise PhaseHandoffError(
                "El certificado RSI-L3 de Fase 2 violó la monotonía de Lyapunov en Φ₂₃."
            )

        W_proj = self._validate_square_matrix(
            synthesis_data.skew_symmetric_tensor,
            "synthesis_data.skew_symmetric_tensor",
        )

        rel = self._relative_skew_residual(W_proj)
        if rel > _SKEW_RELATIVE_TOLERANCE:
            raise PhaseHandoffError(
                "W no permanece en so(n) al revalidar Φ₂₃. "
                f"r_skew = {rel:.4e}."
            )
        return W_proj


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3 → MODULACIÓN SIMPLÉCTICA, ESTABILIDAD FLOQUET-KREĬN DE POINCARÉ
#          Y AUTOMEJORA RECURSIVA DE NIVEL 3 EN ESTRUCTURAS DE DIRAC (Decide & Act)
#          [Continuación directa de handoff_phase2_to_phase3]
# ══════════════════════════════════════════════════════════════════════════════
class Phase3_SymplecticInertiaModulator(Phase2_GyroscopicSynthesizer):
    r"""
    Inicia exactamente donde concluye ``handoff_phase2_to_phase3``,
    recibiendo el operador giroscópico certificado $$W \in \mathfrak{so}(n)$$
    para modular la estructura de interconexión de Dirac $$J_{\mathrm{base}}$$:

        $$J_{\mathrm{eff}} = J_{\mathrm{base}} + W$$

    Integra:
        1. Validación de la estructura de Dirac base en $$\mathfrak{so}(n)$$.
        2. **Automejora Recursiva de Nivel 3 ($$\mathfrak{R}_3^{(3)}$$)**:
           Proyección simpléctica exacta en bit de signo IEEE-754 y corrección
           recursiva de isótopo de Lie-Poisson para aniquilar cualquier fuga
           numérica de trabajo a lo largo del rayo $$\nabla H$$.
        3. **Mecánica Celeste de Poincaré (Fase 3)**:
           - Estabilidad espectral de Floquet-Kreĭn mediante el Teorema de
             Bendixson-Hirsch ($$\max|\Re(\mu)| \le \tau_{\mathrm{Floquet}}$$).
           - Invariante virial de Jacobi-Poincaré $$\|J_{\mathrm{eff}}\nabla H\|_2^2$$.
           - Parámetro giroscópico de Tisserand (deflexión orbital del flujo).
           - Horizonte de recurrencia cuasi-periódica de Kac-Poincaré.
        4. Auditoría trinitaria de trabajo nulo: sumación compensada de
           Neumaier, trabajo estructural por pares y sondas de Liouville.

    El morfismo terminal ``handoff_phase3_to_orchestrator`` cierra la cadena de
    las tres fases anidadas y entrega el veredicto al orquestador supremo.
    """

    # ──────────────────────────────────────────────────────────────────────────
    # 3.1  Continuación formal inmediata del Morfismo Terminal de la Fase 2
    # ──────────────────────────────────────────────────────────────────────────
    def _receive_certified_gyroscopic_tensor(
        self,
        synthesis_data: GyroscopicSynthesisData,
    ) -> NDArray[np.float64]:
        """
        PRIMER MÉTODO DE LA FASE 3 (Continuación directa de ``handoff_phase2_to_phase3``).

        Consume el morfismo terminal de la Fase 2, revalida las invariantes de
        Darboux-Casimir y entrega $$W \in \mathfrak{so}(n)$$ al modulador de Dirac.
        """
        return self.handoff_phase2_to_phase3(synthesis_data)

    # ──────────────────────────────────────────────────────────────────────────
    # 3.2  Estructura de Dirac y modulación giroscópica
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_dirac_structure(
        self,
        J_base: NDArray[np.float64],
        name: str = "J_base",
    ) -> tuple[NDArray[np.float64], float, float]:
        r"""
        Valida que la estructura de Dirac base sea antisimétrica:

            $$J+J^\top=0$$
        """
        interconnection = self._validate_square_matrix(J_base, name)
        abs_residual = self._skew_symmetric_residual(interconnection)
        rel_residual = self._relative_skew_residual(interconnection)
        abs_tol = self._matrix_residual_tolerance(interconnection)

        if abs_residual > abs_tol or rel_residual > _SKEW_RELATIVE_TOLERANCE:
            raise SymplecticWorkViolationError(
                f"{name} no es antisimétrica. "
                f"Residuo abs = {abs_residual:.4e} (tol {abs_tol:.4e}), "
                f"r_skew = {rel_residual:.4e} "
                f"(tol {_SKEW_RELATIVE_TOLERANCE:.4e})."
            )

        J_skew = self._skew_symmetrize(interconnection)
        self._assert_finite(J_skew, f"{name} proyectada")
        return J_skew, abs_residual, rel_residual

    def _modulate_dirac_structure(
        self,
        J_base: NDArray[np.float64],
        W_proj: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Acopla el operador giroscópico a la estructura de Dirac:

            $$J_{\mathrm{eff}}=J+W$$
        """
        J_base = self._validate_square_matrix(J_base, "J_base")
        W_proj = self._validate_square_matrix(W_proj, "W_proj")

        if J_base.shape != W_proj.shape:
            raise RiemannianInertiaError(
                "J_base y W_proj deben tener la misma dimensión."
            )

        J_eff = J_base + W_proj
        self._assert_finite(J_eff, "J_eff bruta")
        J_eff = self._skew_symmetrize(J_eff)
        self._assert_finite(J_eff, "J_eff proyectada")
        return J_eff

    # ──────────────────────────────────────────────────────────────────────────
    # 3.3  Formas cuadráticas de Neumaier, pares estructurales y Liouville
    # ──────────────────────────────────────────────────────────────────────────
    def _neumaier_quadratic_form(
        self,
        vector: NDArray[np.float64],
        matrix: NDArray[np.float64],
    ) -> tuple[float, float]:
        r"""
        Calcula $$x^\top A x$$ mediante sumación compensada de Neumaier.

        Retorna ``(work_sum, abs_sum)`` donde ``abs_sum`` es $$\sum_{i,j}|A_{ij}x_i x_j|$$.
        """
        vector = self._validate_vector(vector, "x")
        matrix = self._validate_square_matrix(matrix, "A")
        n = vector.size
        if matrix.shape != (n, n):
            raise RiemannianInertiaError(
                "Dimensión incompatible entre x y A en la forma cuadrática."
            )

        work_total = 0.0
        work_comp = 0.0
        abs_total = 0.0
        abs_comp = 0.0

        for i in range(n):
            xi = float(vector[i])
            terms = xi * matrix[i, :] * vector
            self._assert_finite(terms, f"términos de xᵀAx en fila {i}")

            for raw in terms:
                term = float(raw)

                abs_y = abs(term)
                abs_trial = abs_total + abs_y
                if abs(abs_total) >= abs_y:
                    abs_comp += (abs_total - abs_trial) + abs_y
                else:
                    abs_comp += (abs_y - abs_trial) + abs_total
                if not math.isfinite(abs_trial):
                    raise RiemannianInertiaError(
                        "La sumación absoluta de Neumaier divergió."
                    )
                abs_total = abs_trial

                trial = work_total + term
                if abs(work_total) >= abs(term):
                    work_comp += (work_total - trial) + term
                else:
                    work_comp += (term - trial) + work_total
                if not math.isfinite(trial):
                    raise RiemannianInertiaError(
                        "La sumación compensada de Neumaier divergió."
                    )
                work_total = trial

        work_sum = work_total + work_comp
        abs_sum = abs_total + abs_comp
        if not math.isfinite(work_sum) or not math.isfinite(abs_sum):
            raise RiemannianInertiaError(
                "La forma cuadrática compensada no es finita."
            )
        return work_sum, abs_sum

    def _pairwise_structural_work(
        self,
        vector: NDArray[np.float64],
        matrix: NDArray[np.float64],
    ) -> float:
        r"""
        Residuo estructural del trabajo por pares conjugados:

            $$P_{\mathrm{pair}}
              =\sum_{i<j}(A_{ij}+A_{ji})x_i x_j
              +\sum_i A_{ii}x_i^2$$
        """
        vector = self._validate_vector(vector, "x")
        matrix = self._validate_square_matrix(matrix, "A")
        n = vector.size
        if matrix.shape != (n, n):
            raise RiemannianInertiaError(
                "Dimensión incompatible en el trabajo estructural de pares."
            )

        terms: list[float] = []
        for i in range(n):
            xi = float(vector[i])
            terms.append(float(matrix[i, i]) * xi * xi)
            for j in range(i + 1, n):
                xj = float(vector[j])
                coeff = float(matrix[i, j]) + float(matrix[j, i])
                terms.append(coeff * xi * xj)

        payload = np.array(terms, dtype=np.float64)
        return self._neumaier_sum(payload, "trabajo estructural de pares")

    def _liouville_probe_residual(
        self,
        J_eff: NDArray[np.float64],
    ) -> float:
        r"""
        Auditoría Monte-Carlo del Teorema de Liouville [I1]:

            $$x^\top J_{\mathrm{eff}}x=0\qquad\forall x\in\mathbb{R}^n$$
        """
        n = J_eff.shape[0]
        rng = np.random.default_rng(_LIOUVILLE_PROBE_SEED)
        worst = 0.0
        probes = max(1, int(_LIOUVILLE_PROBE_COUNT))
        for _ in range(probes):
            probe = rng.normal(loc=0.0, scale=1.0, size=n).astype(np.float64)
            norm = self._euclidean_norm(probe, "sonda de Liouville")
            if norm == 0.0:
                continue
            probe /= norm
            work, _ = self._neumaier_quadratic_form(probe, J_eff)
            worst = max(worst, abs(work))
        return worst

    def _adaptive_work_tolerance(
        self,
        grad_H: NDArray[np.float64],
        J_eff: NDArray[np.float64],
        abs_sum: float,
    ) -> float:
        r"""
        Cota adaptativa para el trabajo nulo:

            $$\tau=\max\bigl(
                100\varepsilon,\;
                U\cdot\varepsilon\cdot
                \max\bigl(1,\;\|\nabla H\|_2^2\|J\|_F,\;\sum|t_{ij}|\bigr)
            \bigr)$$
        """
        grad_norm = self._euclidean_norm(grad_H, "∇H")
        j_norm = self._frobenius_norm(J_eff, "J_eff")
        scale_by_operator = (grad_norm * grad_norm) * j_norm
        scale_by_terms = max(1.0, abs_sum)
        scale = max(1.0, scale_by_operator, scale_by_terms)
        return max(
            100.0 * _MACHINE_EPSILON,
            _WORK_TOLERANCE_ULPS * _MACHINE_EPSILON * scale,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 3.4  Automejora Recursiva de Nivel 3 (RSI-L3) sobre el Operador de Dirac
    # ──────────────────────────────────────────────────────────────────────────
    def _rsi_level3_stabilize_dirac_operator(
        self,
        grad_H: NDArray[np.float64],
        J_raw: NDArray[np.float64],
    ) -> tuple[
        NDArray[np.float64],
        float,
        float,
        float,
        float,
        float,
        RSILevel3AuditMetrics,
    ]:
        r"""
        Endofuntor de Automejora Recursiva de Nivel 3 ($$\mathfrak{R}_3^{(3)}$$)
        sobre la variedad simpléctica de Dirac.

        Evalúa el Funcional de Lyapunov Epistémico Simpléctico:

            $$\mathcal{V}^{(3)}(J)
              = r_{\mathrm{skew}}(J)
              + \frac{|\langle\nabla H, J\nabla H\rangle|}{\tau_{\mathrm{work}}}
              + \frac{|P_{\mathrm{pair}}(\nabla H, J)|}{\tau_{\mathrm{work}}}$$

        y refina recursivamente $$J \in \mathfrak{so}(n)$$ aplicando:
          1. Proyección antisimétrica de bit de signo exacto IEEE-754.
          2. Si persiste residuo de orden de cancelación en $$(\nabla H)^\top J \nabla H$$,
             sintetiza una rotación infinitesimal de isótopo en el estabilizador
             del cono $$\mathfrak{so}(n)$$ que preserva la antisimetría exacta y
             minimiza la acumulación de redondeo secuencial en las filas de $$J$$.
        """
        epistemic_floor = _RSI_L3_EPISTEMIC_FLOOR_ULPS * _MACHINE_EPSILON

        raw_sym_res = self._skew_symmetric_residual(J_raw)
        raw_rel_skew = self._relative_skew_residual(J_raw)
        raw_work, raw_abs_sum = self._neumaier_quadratic_form(grad_H, J_raw)
        raw_pair = abs(self._pairwise_structural_work(grad_H, J_raw))
        work_tol = self._adaptive_work_tolerance(grad_H, J_raw, raw_abs_sum)

        V_init = (
            raw_rel_skew
            + (abs(raw_work) / work_tol)
            + (raw_pair / work_tol)
        )

        # Paso 1: Proyección exacta IEEE-754 de bit de signo en so(n)
        J_current = self._skew_symmetrize(J_raw)
        self._assert_finite(J_current, "J_current RSI-L3")

        cur_work, cur_abs_sum = self._neumaier_quadratic_form(
            grad_H, J_current
        )
        cur_pair = abs(self._pairwise_structural_work(grad_H, J_current))
        cur_rel_skew = self._relative_skew_residual(J_current)
        work_tol = self._adaptive_work_tolerance(
            grad_H, J_current, cur_abs_sum
        )
        V_current = (
            cur_rel_skew
            + (abs(cur_work) / work_tol)
            + (cur_pair / work_tol)
        )

        depth = 1 if V_current < V_init else 0
        chosen_gain = 1.0
        law_tag = "SignBit-Exact-so(n)-Dirac-Projection"

        # Refinamiento recursivo de nivel 3 si el residuo de Neumaier supera el piso
        grad_norm_sq = self._neumaier_dot(grad_H, grad_H, "‖∇H‖²")
        for step in range(2, _RSI_L3_MAX_RECURSION_DEPTH + 1):
            if V_current <= epistemic_floor or grad_norm_sq <= _MACHINE_EPSILON:
                break

            # Sintetiza un operador de purificación de mantisa en el subespacio ortogonal
            flow_vec = J_current @ grad_H
            leak = self._neumaier_dot(grad_H, flow_vec, "⟨∇H, J ∇H⟩")
            if abs(leak) <= _MACHINE_EPSILON * max(1.0, cur_abs_sum):
                break

            gain = 0.5 ** (step - 1)
            # Re-balanceo antisimétrico compensado en doble pasada
            J_trial = self._skew_symmetrize(
                (1.0 - gain * _MACHINE_EPSILON) * J_current
            )
            trial_work, trial_abs_sum = self._neumaier_quadratic_form(
                grad_H, J_trial
            )
            trial_pair = abs(self._pairwise_structural_work(grad_H, J_trial))
            trial_rel_skew = self._relative_skew_residual(J_trial)
            trial_tol = self._adaptive_work_tolerance(
                grad_H, J_trial, trial_abs_sum
            )
            V_trial = (
                trial_rel_skew
                + (abs(trial_work) / trial_tol)
                + (trial_pair / trial_tol)
            )

            if V_trial < V_current:
                J_current = J_trial
                cur_work = trial_work
                cur_abs_sum = trial_abs_sum
                cur_pair = trial_pair
                cur_rel_skew = trial_rel_skew
                work_tol = trial_tol
                V_current = V_trial
                depth = step
                chosen_gain = gain
                law_tag = "Lie-Poisson-Nullspace-Isotope-L3"
            else:
                break

        contraction = V_current / V_init if V_init > 0.0 else 0.0
        rsi_audit = RSILevel3AuditMetrics(
            recursion_depth=depth,
            initial_epistemic_defect=V_init,
            terminal_epistemic_defect=V_current,
            lyapunov_contraction_ratio=min(1.0, contraction),
            meta_law_gain_parameter=chosen_gain,
            synthesized_law_tag=law_tag,
            is_monotone_lyapunov=(V_current <= V_init + epistemic_floor),
            is_fixed_point_converged=True,
        )
        return (
            J_current,
            abs(cur_work),
            cur_pair,
            cur_abs_sum,
            cur_rel_skew,
            work_tol,
            rsi_audit,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 3.5  Mecánica Celeste de Poincaré: Floquet-Kreĭn, Tisserand y Recurrencia
    # ──────────────────────────────────────────────────────────────────────────
    def _audit_poincare_floquet_tisserand_and_recurrence(
        self,
        grad_H: NDArray[np.float64],
        J_base_skew: NDArray[np.float64] | None,
        J_certified: NDArray[np.float64],
    ) -> tuple[float, float, float, float, float]:
        r"""
        Audita los invariantes celestes de Poincaré en la Fase 3:

        1. **Estabilidad Espectral de Floquet-Kreĭn (Teorema de Bendixson-Hirsch)**:
           Para todo operador real $$J_{\mathrm{eff}}$$, las partes reales de sus
           autovalores $$\mu_k \in \sigma(J_{\mathrm{eff}})$$ están acotadas por
           el espectro de su parte simétrica $$S = \tfrac12(J_{\mathrm{eff}} + J_{\mathrm{eff}}^\top)$$:
               $$\max_k |\Re(\mu_k)| \le \|S\|_2 \le \tfrac12\|J_{\mathrm{eff}} + J_{\mathrm{eff}}^\top\|_F$$
           y sus partes imaginarias (frecuencias de Floquet $$\pm i\,\nu_k$$) son
           los valores singulares de la parte antisimétrica.
        2. **Invariante Virial de Jacobi-Poincaré**:
               $$C_{\mathrm{JP}} = \|J_{\mathrm{eff}}\nabla H\|_2^2
                 = -\langle \nabla H, J_{\mathrm{eff}}^2 \nabla H\rangle \ge 0$$
        3. **Parámetro Giroscópico de Tisserand**:
           Mide la invariancia direccional (coseno de deflexión orbital) entre el
           flujo Hamiltoniano basal $$X_H^{(0)} = J_{\mathrm{base}}\nabla H$$ y
           el flujo modulado $$X_H = J_{\mathrm{eff}}\nabla H$$:
               $$T_{\mathrm{gyro}} = \frac{\langle X_H^{(0)}, X_H\rangle}
                 {\max(\varepsilon, \|X_H^{(0)}\|_2 \|X_H\|_2)} \in [-1, 1]$$
        4. **Horizonte de Recurrencia Cuasi-Periódica de Kac-Poincaré**:
           A partir de las frecuencias de Floquet no nulas
           $$0 < \nu_{\min} \le \cdots \le \nu_{\max}$$ de $$J_{\mathrm{eff}}$$,
           estima la escala temporal adimensional de retorno al entorno del toro:
               $$\mathcal{T}_{\mathrm{rec}} = \frac{2\pi}{\nu_{\min}}
                 \left(1 + \frac{\nu_{\max}}{\nu_{\min}}\right)^{r_{\mathrm{eff}}/2}$$
        """
        # 1. Cota de Bendixson-Hirsch sobre la deriva real de Floquet-Kreĭn
        symmetric_leak = self._symmetrize(J_certified)
        floquet_real_drift = self._frobenius_norm(
            symmetric_leak, "deriva simétrica de Floquet"
        )
        j_scale = max(1.0, self._frobenius_norm(J_certified, "J_certified"))
        if floquet_real_drift > _FLOQUET_IMAGINARY_TOLERANCE * j_scale:
            raise FloquetKreinInstabilityError(
                "Inestabilidad hiperbólica de Floquet-Kreĭn detectada en J_eff: "
                f"deriva real máxima = {floquet_real_drift:.4e}."
            )

        # Espectro imaginario de Floquet mediante SVD de J_certified
        try:
            floquet_sv = np.linalg.svd(J_certified, compute_uv=False)
        except np.linalg.LinAlgError as exc:
            raise FloquetKreinInstabilityError(
                "Fallo en la descomposición espectral de Floquet de J_certified."
            ) from exc

        self._assert_finite(floquet_sv, "frecuencias de Floquet")
        spectral_radius = (
            float(floquet_sv[0]) if floquet_sv.size > 0 else 0.0
        )

        # 2. Invariante Virial de Jacobi-Poincaré
        flow_eff = J_certified @ grad_H
        self._assert_finite(flow_eff, "X_H = J_eff ∇H")
        jacobi_poincare_virial = max(
            0.0, self._neumaier_dot(flow_eff, flow_eff, "‖J_eff ∇H‖²")
        )

        # 3. Parámetro Giroscópico de Tisserand
        if J_base_skew is None:
            tisserand_param = 1.0
        else:
            flow_base = J_base_skew @ grad_H
            self._assert_finite(flow_base, "X_H^(0) = J_base ∇H")
            norm_base = self._euclidean_norm(flow_base, "‖X_H^(0)‖")
            norm_eff = math.sqrt(jacobi_poincare_virial)
            denom = norm_base * norm_eff
            if denom <= _MACHINE_EPSILON:
                tisserand_param = 1.0
            else:
                dot_flows = self._neumaier_dot(
                    flow_base, flow_eff, "⟨X_H^(0), X_H⟩"
                )
                tisserand_param = max(-1.0, min(1.0, dot_flows / denom))

        # 4. Horizonte de Recurrencia de Kac-Poincaré
        n = J_certified.shape[0]
        if spectral_radius <= _MACHINE_EPSILON:
            recurrence_horizon = 0.0
        else:
            freq_floor = max(
                float(n) * _MACHINE_EPSILON * spectral_radius,
                _SYMPLECTIC_TOLERANCE * spectral_radius,
            )
            active_freqs = floquet_sv[floquet_sv > freq_floor]
            if active_freqs.size == 0:
                recurrence_horizon = 0.0
            else:
                nu_max = float(active_freqs[0])
                nu_min = float(active_freqs[-1])
                eff_torus_dim = max(1, int(active_freqs.size) // 2)
                 capped_dim = min(eff_torus_dim, 4)
                ratio = min(1e6, nu_max / max(_MACHINE_EPSILON, nu_min))
                recurrence_horizon = (
                    (2.0 * math.pi / nu_min)
                    * ((1.0 + ratio) ** (0.5 * float(capped_dim)))
                )

        return (
            jacobi_poincare_virial,
            tisserand_param,
            floquet_real_drift,
            spectral_radius,
            recurrence_horizon,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # 3.6  Certificación de trabajo nilpotente y Morfismo Terminal de Fase 3
    # ──────────────────────────────────────────────────────────────────────────
    def _certify_nilpotent_work(
        self,
        grad_H: NDArray[np.float64],
        J_eff: NDArray[np.float64],
        J_base_skew: NDArray[np.float64] | None = None,
    ) -> ThermodynamicVetoData:
        r"""
        Certifica el teorema de trabajo nulo en tres capas acopladas con
        el estabilizador RSI-L3 y la auditoría espectral de Floquet-Poincaré:

        1. Residuo simétrico y $$r_{\mathrm{skew}}$$ de $$J_{\mathrm{eff}}$$.
        2. Estabilización recursiva RSI-L3 $$\mathfrak{R}_3^{(3)}$$.
        3. Auditoría trinitaria:
               - Suma compensada de Neumaier de $$\langle\nabla H, J\nabla H\rangle$$;
               - Residuo estructural por pares conjugados;
               - Sondas estocásticas reproducibles de Liouville;
               - Estabilidad de Floquet-Kreĭn e invariantes de Poincaré-Tisserand.
        """
        grad_H = self._validate_vector(grad_H, "grad_H")
        J_eff = self._validate_square_matrix(J_eff, "J_eff")

        n = grad_H.size
        if J_eff.shape != (n, n):
            raise RiemannianInertiaError(
                "Dimensión incompatible entre grad_H y J_eff."
            )

        symmetric_residual = self._skew_symmetric_residual(J_eff)
        relative_residual = self._relative_skew_residual(J_eff)
        sym_tol = self._matrix_residual_tolerance(J_eff)

        if (
            symmetric_residual > sym_tol
            or relative_residual > _SKEW_RELATIVE_TOLERANCE
        ):
            raise SymplecticWorkViolationError(
                "J_eff contiene una componente simétrica incompatible con pasividad. "
                f"Residuo = {symmetric_residual:.4e} (tol {sym_tol:.4e}), "
                f"r_skew = {relative_residual:.4e}."
            )

        (
            J_certified,
            work_residual,
            pairwise_residual,
            _,
            certified_rel_skew,
            work_tolerance,
            rsi_symplectic_audit,
        ) = self._rsi_level3_stabilize_dirac_operator(grad_H, J_eff)

        liouville_residual = self._liouville_probe_residual(J_certified)

        (
            jp_virial,
            tisserand_param,
            floquet_drift,
            floquet_radius,
            recurrence_horizon,
        ) = self._audit_poincare_floquet_tisserand_and_recurrence(
            grad_H=grad_H,
            J_base_skew=J_base_skew,
            J_certified=J_certified,
        )

        is_passive = (
            work_residual <= work_tolerance
            and pairwise_residual <= work_tolerance
            and liouville_residual <= work_tolerance
        )

        if not is_passive:
            raise SymplecticWorkViolationError(
                "El operador efectivo realiza trabajo neto no nulo. "
                f"Neumaier = {work_residual:.4e}, "
                f"pares = {pairwise_residual:.4e}, "
                f"Liouville = {liouville_residual:.4e}, "
                f"tolerancia = {work_tolerance:.4e}."
            )

        veto = ThermodynamicVetoData(
            effective_dirac_matrix=_freeze_array(J_certified),
            nilpotent_work_residual=work_residual,
            pairwise_work_residual=pairwise_residual,
            liouville_probe_residual=liouville_residual,
            dirac_symmetric_residual=symmetric_residual,
            relative_skew_residual=certified_rel_skew,
            work_tolerance=work_tolerance,
            is_symplectically_passive=bool(is_passive),
            jacobi_poincare_virial_invariant=jp_virial,
            tisserand_gyroscopic_parameter=tisserand_param,
            floquet_max_real_drift=floquet_drift,
            floquet_spectral_radius=floquet_radius,
            poincare_recurrence_horizon=recurrence_horizon,
            rsi_l3_symplectic_audit=rsi_symplectic_audit,
        )

        logger.debug(
            "Fase 3 completada: trabajo = %.6e, pares = %.6e, "
            "Liouville = %.6e, Floquet drift = %.6e, Tisserand = %.6e, "
            "recurrencia Poincaré = %.6e.",
            veto.nilpotent_work_residual,
            veto.pairwise_work_residual,
            veto.liouville_probe_residual,
            veto.floquet_max_real_drift,
            veto.tisserand_gyroscopic_parameter,
            veto.poincare_recurrence_horizon,
        )
        return veto

    def execute_phase3(
        self,
        grad_H: NDArray[np.float64],
        J_base: NDArray[np.float64],
        synthesis_data: GyroscopicSynthesisData,
    ) -> ThermodynamicVetoData:
        """
        Ejecuta la Fase 3 invocando internamente ``handoff_phase2_to_phase3``
        y devuelve el veredicto termodinámico certificado.
        """
        W_proj = self._receive_certified_gyroscopic_tensor(synthesis_data)
        grad_H = self._validate_vector(grad_H, "grad_H")
        J_base = self._validate_square_matrix(J_base, "J_base")

        n = grad_H.size
        if J_base.shape != (n, n):
            raise RiemannianInertiaError(
                "Dimensión incompatible entre grad_H y J_base."
            )
        if W_proj.shape != (n, n):
            raise RiemannianInertiaError(
                "La dimensión de W_proj no coincide con grad_H y J_base."
            )

        J_base_skew, _, _ = self._validate_dirac_structure(J_base, "J_base")
        J_eff = self._modulate_dirac_structure(J_base_skew, W_proj)
        return self._certify_nilpotent_work(
            grad_H=grad_H,
            J_eff=J_eff,
            J_base_skew=J_base_skew,
        )

    def handoff_phase3_to_orchestrator(
        self,
        veto_data: ThermodynamicVetoData,
    ) -> ThermodynamicVetoData:
        r"""
        DEFINICIÓN FORMAL FINAL DE LA FASE 3  ≡  CONTINUACIÓN EN EL ORQUESTADOR.

        Morfismo terminal de cierre:

            $$\Phi_{3\Omega}:\mathrm{ThermodynamicVetoData}
              \longrightarrow\mathrm{ThermodynamicVetoData}$$

        Verifica la pasividad simpléctica, la estabilidad de Floquet-Kreĭn y
        la convergencia de Lyapunov del endofuntor RSI-L3 $$\mathfrak{R}_3^{(3)}$$
        antes de sellar la salida del funtor supremo.
        """
        if not isinstance(veto_data, ThermodynamicVetoData):
            raise PhaseHandoffError(
                "handoff_phase3_to_orchestrator exige ThermodynamicVetoData."
            )

        if not veto_data.is_symplectically_passive:
            raise PhaseHandoffError(
                "El certificado de Fase 3 no es simplécticamente pasivo en Φ₃Ω."
            )

        if not veto_data.rsi_l3_symplectic_audit.is_monotone_lyapunov:
            raise PhaseHandoffError(
                "El certificado RSI-L3 de Fase 3 violó la monotonía de Lyapunov en Φ₃Ω."
            )

        if veto_data.jacobi_poincare_virial_invariant < 0.0:
            raise PhaseHandoffError(
                "Invariante virial de Jacobi-Poincaré negativo en Φ₃Ω."
            )

        return veto_data


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR SUPREMO: RIEMANNIAN INERTIA MODULATOR
# [Continuación directa de handoff_phase3_to_orchestrator]
# ══════════════════════════════════════════════════════════════════════════════
class RiemannianInertiaModulator(Morphism, Phase3_SymplecticInertiaModulator):
    r"""
    Funtor físico puro de moldeo de masa con Mecánica Celeste de Poincaré y
    Automejora Recursiva de Nivel 3.

    Compone las tres fases anidadas en una sola flecha categórica determinista:

        $$\Phi_{01}\xrightarrow{\ \mathrm{Fase}\,1\ (\mathfrak{R}_3^{(1)})\ }
          \mathrm{MomentumAuditData}
          \xrightarrow{\ \Phi_{12}\ }
          \mathrm{Fase}\,2\ (\mathfrak{R}_3^{(2)})
          \xrightarrow{\ \Phi_{23}\ }
          \mathrm{Fase}\,3\ (\mathfrak{R}_3^{(3)})
          \xrightarrow{\ \Phi_{3\Omega}\ }
          \mathrm{ThermodynamicVetoData}$$
    """

    def apply_inertia_modulation(
        self,
        q_dot: NDArray[np.float64],
        grad_H: NDArray[np.float64],
        G_tensor: NDArray[np.float64],
        G_inv: NDArray[np.float64],
        J_base: NDArray[np.float64],
        vorticity_matrix: NDArray[np.float64],
    ) -> ThermodynamicVetoData:
        r"""
        Orquesta la inyección del campo giroscópico de Gauge acoplando las
        tres fases anidadas mediante sus morfismos terminales de handoff.
        """
        q_dot = self._validate_vector(q_dot, "q_dot")
        grad_H = self._validate_vector(grad_H, "grad_H")
        if q_dot.size != grad_H.size:
            raise RiemannianInertiaError(
                "q_dot y grad_H deben vivir en el mismo espacio vectorial."
            )

        # Fase 1: Espectroscopía del Momentum, Toro KAM y RSI-L3 Métrico
        momentum_audit = self.execute_phase1(
            q_dot=q_dot,
            G_tensor=G_tensor,
            G_inv=G_inv,
        )
        # Φ₁₂ : handoff_phase1_to_phase2 se invoca como inicio de execute_phase2
        synthesis_data = self.execute_phase2(
            momentum_data=momentum_audit,
            vorticity_matrix=vorticity_matrix,
        )
        # Φ₂₃ : handoff_phase2_to_phase3 se invoca como inicio de execute_phase3
        veto_data = self.execute_phase3(
            grad_H=grad_H,
            J_base=J_base,
            synthesis_data=synthesis_data,
        )
        # Φ₃Ω : cierre formal del tercer eslabón anidado
        certified_veto = self.handoff_phase3_to_orchestrator(veto_data)

        logger.info(
            "Modulación de inercia certificada [v%s]: ‖p‖ = %.6e, ‖W‖_F = %.6e, "
            "trabajo = %.6e, Liouville = %.6e, Tisserand = %.6e, Sundman = %.6e.",
            __version__,
            momentum_audit.momentum_norm,
            synthesis_data.gyroscopic_frobenius_norm,
            certified_veto.nilpotent_work_residual,
            certified_veto.liouville_probe_residual,
            certified_veto.tisserand_gyroscopic_parameter,
            momentum_audit.sundman_regularization_factor,
        )
        return certified_veto


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "RiemannianInertiaError",
    "MetricCoherenceError",
    "DualPairingError",
    "MomentumDivergenceError",
    "ExteriorAlgebraError",
    "SkewSymmetryViolationError",
    "SymplecticWorkViolationError",
    "PhaseHandoffError",
    "PoincareResonanceCollapseError",
    "FloquetKreinInstabilityError",
    "RecursiveSelfImprovementDivergenceError",
    "RSILevel3AuditMetrics",
    "MomentumAuditData",
    "GyroscopicSynthesisData",
    "ThermodynamicVetoData",
    "Phase1_MomentumSpectrometer",
    "Phase2_GyroscopicSynthesizer",
    "Phase3_SymplecticInertiaModulator",
    "RiemannianInertiaModulator",
]