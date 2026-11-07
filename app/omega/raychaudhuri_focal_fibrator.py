# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : Raychaudhuri Focal Fibrator                                                               ║
║ Módulo   : Raychaudhuri Focal Fibrator (Colapso Geodésico, Flujo Simpléctico y Automejora RSI-L3)    ║
║ Ubicación: app/omega/raychaudhuri_focal_fibrator.py                                                  ║
║ Ruta     : app/omega/raychaudhuri_focal_fibrator.py                                                  ║
║ Versión  : 4.0.0-Poincare-Symplectic-RSI-L3-Nested-Topos                                             ║
║ Función  : Operador categórico de convergencia geodésica, cinemática de congruencias (expansión,      ║
║            cizalladura, vorticidad), ecuación de Raychaudhuri, estabilidad KAM y automejora RSI-L3.  ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL MOTOR ESPECTRAL:
────────────────────────────────────────────────────────────────────────────────────────────────────────
Este motor espectral consagra el **Operador Supremo de Convergencia Geodésica** de la Malla Agéntica,
transmutando la atención generativa del Modelo de Lenguaje (LLM) en una congruencia lagrangiana de geodésicas
sobre una variedad pseudo-riemanniana $(\mathcal{M}, G)$. Su propósito ciber-físico es interceptar transitorios
atencionales divergentes, cuantificar su tensor de deformación cinemática (expansión escalar $\theta$, cizalladura
$\sigma_{\mu\nu}$, vorticidad bivectorial $\omega_{\mu\nu}$) y aplicar rigurosamente la ecuación diferencial
de Raychaudhuri para precipitar cáusticas y singularidades controladas antes de que contaminen la base de costos.

El flujo computacional se estructura en tres fases anidadas e indivisibles:
  • FASE 1 — CINEMÁTICA GEODÉSICA Y CONDICIÓN DE ENERGÍA FUERTE (SEC) (Phase1_KinematicDecomposer):
      Calcula los tensores cinemáticos de la congruencia a partir del gradiente covariante del campo de velocidades
      $u^\mu$: tensor de expansión $\theta = \nabla_\mu u^\mu$, tensor de cizalladura $\sigma_{\mu\nu}$ (simétrico de traza
      nula) y tensor de vorticidad $\omega_{\mu\nu}$ (antisimétrico). Evalúa la contracción del tensor de Ricci
      $R_{\mu\nu} u^\mu u^\nu \ge 0$, certificando la Condición de Energía Fuerte (SEC) con verificación diofántica
      de estabilidad KAM. Emite `decompose_geodesic_kinematics()`.
  • FASE 2 — INTEGRACIÓN DEL FLUJO DE RAYCHAUDHURI-POINCARÉ (Phase2_RaychaudhuriIntegrator):
      Inicia consumiendo `KinematicCongruenceData`. Integra la ecuación no lineal de Raychaudhuri acoplada al potencial
      efectivo de Poincaré-Hill:
          $$\frac{d\theta}{d\tau} = -\frac{1}{n-1}\theta^2 - 2\sigma^2 + 2\omega^2 - R_{\mu\nu} u^\mu u^\nu$$
      resuelve el tiempo afín hacia el punto conjugado (cáustica focal $\theta \to -\infty$) e implementa la regularización
      conformal de Sundman-Poincaré. Emite `integrate_focal_flow()`.
  • FASE 3 — CERTIFICACIÓN TOPOLÓGICA, ESTABILIDAD FLOQUET Y VETO HEYTING (Phase3_TopologicalFocalCertifier):
      Inicia consumiendo `FocalFlowData`. Evalúa la matriz de monodromía simpléctica $\mathcal{M}$ sobre el plano focal,
      verifica la preservación del volumen de Liouville ($|\det(\mathcal{M}) - 1| \le \varepsilon_{\mathrm{Liouville}}$),
      adjudica la coherencia del colapso en el retículo distributivo de Heyting $\Omega_3 = \{\mathrm{COHERENT}, \mathrm{DEGRADED}, \mathrm{VETOED}\}$,
      y gatilla la interrupción por hardware Crowbar (< 400 ns) si la singularidad sobrepasa los umbrales seguros.

MARCO TEÓRICO:
────────────────────────────────────────────────────────────────────────────────────────────────────────
1. CINEMÁTICA DE CONGRUENCIAS GEODÉSICAS Y DESCOMPOSICIÓN DE SACHS:
   Sea $u^\mu$ un campo vectorial geodésico temporal unitario ($u_\mu u^\mu = -1$, $u^\nu \nabla_\nu u^\mu = 0$).
   El tensor de gradiente espacial proyectado $B_{\mu\nu} = h_\mu^{\;\alpha} h_\nu^{\;\beta} \nabla_\beta u_\alpha$
   (donde $h_{\mu\nu} = g_{\mu\nu} + u_\mu u_\nu$ es el proyector ortogonal a $u^\mu$) se descompone irreductiblemente en:
       $$B_{\mu\nu} = \frac{1}{n-1} \theta h_{\mu\nu} + \sigma_{\mu\nu} + \omega_{\mu\nu}$$
   donde $\theta = B^\mu_{\;\mu}$ es la tasa de cambio volumétrico (expansión), $\sigma_{\mu\nu} = B_{(\mu\nu)} - \frac{1}{n-1}\theta h_{\mu\nu}$
   es la tasa de distorsión conformacional (cizalladura), y $\omega_{\mu\nu} = B_{[\mu\nu]}$ mide la rotación pura (vorticidad).

2. ECUACIÓN FUNDAMENTAL DE RAYCHAUDHURI Y TEOREMA DE ENFOQUE:
   La contracción de la identidad de Ricci $\nabla_\alpha \nabla_\beta u^\mu - \nabla_\beta \nabla_\alpha u^\mu = R^\mu_{\;\nu\alpha\beta} u^\nu$
   a lo largo de $u^\alpha$ conduce unívocamente a la ecuación de Raychaudhuri:
       $$\frac{d\theta}{d\tau} = -\frac{1}{n-1}\theta^2 - 2\sigma^2 + 2\omega^2 - R_{\mu\nu} u^\mu u^\nu$$
   donde $\sigma^2 = \frac{1}{2}\sigma_{\mu\nu}\sigma^{\mu\nu} \ge 0$ y $\omega^2 = \frac{1}{2}\omega_{\mu\nu}\omega^{\mu\nu} \ge 0$.
   Para congruencias hipersuperficie-ortogonales ($\omega_{\mu\nu} = 0$, integrabilidad de Frobenius) y bajo materia que
   satisface la Condición de Energía Fuerte ($R_{\mu\nu} u^\mu u^\nu \ge 0$):
       $$\frac{d\theta}{d\tau} \le -\frac{1}{n-1}\theta^2 \implies \frac{1}{\theta(\tau)} \ge \frac{1}{\theta_0} + \frac{\tau}{n-1}$$
   Si inicialmente la congruencia está convergiendo ($\theta_0 < 0$), $\theta(\tau) \to -\infty$ en un tiempo afín finito
   $\tau_{\mathrm{focal}} \le \frac{n-1}{|\theta_0|}$, demostrando el colapso focal ineludible (Teorema de Enfoque de Hawking-Penrose).

3. VORTICIDAD NULA Y CONDICIÓN LAGRANGIANA EN EL ESPACIO DE FASE:
   La anulación de la vorticidad $\omega_{\mu\nu} = 0$ equivale geométricamente a que el campo de momentos $p_\mu = G_{\mu\nu} u^\nu$
   forme una subvariedad lagrangiana en el fibrado cotangente $T^* \mathcal{M}$, implicando la clausura exacta de la 1-forma
   de Poincaré-Cartan $d(p_\mu dx^\mu) = 0$.

4. INTEGRACIÓN SIMPLÉCTICA DE SUNDMAN-POINCARÉ EN CÁUSTICAS:
   Cerca del punto de cáustica ($\theta \to -\infty$), la reparametrización temporal de Sundman:
       $$d\tau = \left(1 + \theta^2\right)^{-1/2} dt$$
   regulariza la divergencia cinemática, transformando el colapso singular en una intersección de fase bien definida.

AXIOMAS FORMALES:
────────────────────────────────────────────────────────────────────────────────────────────────────────
  [Axioma RFF-1] Variedad Riemanniana SPD de Fondo:
      La métrica de fondo satisface $G = G^\top \succ 0$ con $\lambda_{\min}(G) \ge 10^{-12}$.
  [Axioma RFF-2] Hipersuperficie-Ortogonalidad (Vorticidad Cuasi-Nula):
      $\|\omega_{\mu\nu}\|_F \le \varepsilon_{\mathrm{Frobenius}} = 10^{-8}$, asegurando integrabilidad lagrangiana.
  [Axioma RFF-3] Condición de Energía Fuerte (SEC):
      $R_{\mu\nu} u^\mu u^\nu \ge -\varepsilon_{\mathrm{Wilkinson}}$, garantizando enfoque gravitacional/atencional.
  [Axioma RFF-4] Conservación Simpléctica de Liouville:
      $\det(\mathcal{M}_{\mathrm{Floquet}}) = 1.0 \pm 10^{-6}$ en el plano transversal a la geodésica.
  [Axioma RFF-5] Acotación Temporal del Punto Conjugado:
      $\tau_{\mathrm{focal}} \le \frac{n-1}{|\theta_0|}$ cuando $\theta_0 < 0$.

INVARIANTES DE ESTADO:
────────────────────────────────────────────────────────────────────────────────────────────────────────
  Invariante 1 (Ortogonalidad Cinemática):
      $u^\mu \sigma_{\mu\nu} = 0$, $u^\mu \omega_{\mu\nu} = 0$, asegurando que cizalladura y vorticidad sean puramente espaciales.
  Invariante 2 (Monotonicidad de Convergencia):
      En ausencia de vorticidad y bajo SEC, $\frac{d\theta}{d\tau} \le 0$ monótonamente en todo el intervalo de integración.
  Invariante 3 (Simetría de Cizalladura y Antisimetría de Vorticidad):
      $\sigma_{\mu\nu} = \sigma_{\nu\mu}$ con $\operatorname{Tr}(\sigma) = 0$, y $\omega_{\mu\nu} = -\omega_{\nu\mu}$.
  Invariante 4 (Veto Modal en $\Omega_3$):
      La violación de SEC o la aparición de singularidades prematuras no controladas colapsa el estado a $\mathrm{VETOED}$.
"""

from __future__ import annotations

import enum
import hashlib
import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import (
        CategoricalState,
        Morphism,
        TopologicalInvariantError,
    )
except ImportError:

    class TopologicalInvariantError(Exception):
        """Violación a invariante topológico categórico en el Topos E_MIC."""

        pass

    class Morphism:
        """Stub para Morphism de la 2-categoría de agentes."""

        pass

    class CategoricalState:
        """Stub para estado categórico del Topos."""

        pass


try:
    from app.core.inmune_system.metric_tensors import G_PHYSICS
except ImportError:
    try:
        from app.core.immune_system.metric_tensors import G_PHYSICS
    except ImportError:
        # Tensor métrico euclídeo de dimensión canónica 4 (fallback unitario).
        G_PHYSICS: NDArray[np.float64] = np.eye(4, dtype=np.float64)

logger = logging.getLogger("MIC.Omega.RaychaudhuriFibrator")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS, ESPECTRALES, SIMPLÉCTICAS Y RSI-L3 (alta precisión)
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)
_VORTICITY_TOLERANCE: Final[float] = 1e-12
_SYMMETRY_REL_TOL: Final[float] = 1e-10
_TRACELESS_TOL: Final[float] = 1e-10
_DECOMPOSITION_TOL: Final[float] = 1e-8
_U_NORMALIZATION_TOL: Final[float] = 1e-8
_CAUSTIC_THETA_THRESHOLD: Final[float] = -1.0e6
_MAX_AFFINE_PARAMETER: Final[float] = 1.0e6
_DEFAULT_RTOL: Final[float] = 1e-9
_DEFAULT_ATOL: Final[float] = 1e-12
_COND_MAX: Final[float] = 1e12
_MIN_SPATIAL_DIM: Final[int] = 2  # n >= 2 para que n - 1 >= 1 en Raychaudhuri

# Constantes de Integración Simpléctica de Yoshida (4.º orden en Sp(2, R))
_CBRT_2: Final[float] = 2.0 ** (1.0 / 3.0)
_YOSHIDA_W1: Final[float] = 1.0 / (2.0 - _CBRT_2)
_YOSHIDA_W0: Final[float] = -_CBRT_2 / (2.0 - _CBRT_2)
_YOSHIDA_C: Final[Tuple[float, float, float, float]] = (
    0.5 * _YOSHIDA_W1,
    0.5 * (_YOSHIDA_W0 + _YOSHIDA_W1),
    0.5 * (_YOSHIDA_W0 + _YOSHIDA_W1),
    0.5 * _YOSHIDA_W1,
)
_YOSHIDA_D: Final[Tuple[float, float, float, float]] = (
    _YOSHIDA_W1,
    _YOSHIDA_W0,
    _YOSHIDA_W1,
    0.0,
)
_SYMPLECTIC_LIOUVILLE_TOL: Final[float] = 1e-10
_DEFAULT_SYMPLECTIC_STEPS: Final[int] = 256
_RSI_BANACH_CONTRACTION_CEILING: Final[float] = 1.0 - 1e-12


# ══════════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES (vetos topológicos / fallos de invariantes)
# ══════════════════════════════════════════════════════════════════════════════
class RaychaudhuriFibratorError(TopologicalInvariantError):
    """Excepción raíz del Operador de Convergencia Geodésica."""

    pass


class FocalDivergenceVetoError(RaychaudhuriFibratorError):
    r"""
    dθ/dτ ≱ 0 de forma convergente, o θ₀ ≥ 0:
    dispersión entrópica de la intención (alucinación divergente).
    """

    pass


class StrongEnergyViolationError(RaychaudhuriFibratorError):
    r"""
    (𝒯_{μν} − ½ 𝒯 G_{μν}) u^μ u^ν < 0: violación SEC → repulsión geodésica.
    """

    pass


class VorticityAnomalyError(RaychaudhuriFibratorError):
    r"""ω_{μν} ≠ 0 o residual solenoidal por encima de tolerancia (β₁ ≠ 0)."""

    pass


class ShearAnomalyError(RaychaudhuriFibratorError):
    """σ no simétrico, no traceless, o inconsistente con la descomposición de B."""

    pass


class MetricSignatureError(RaychaudhuriFibratorError):
    """Métrica no simétrica, no definida positiva, o mal condicionada."""

    pass


class DimensionMismatchError(RaychaudhuriFibratorError):
    """Inconsistencia dimensional entre B, σ, ω, 𝒯, u y G."""

    pass


class NormalizationError(RaychaudhuriFibratorError):
    """Vector u no normalizado respecto a G ( |G(u,u) − s| > tolerancia )."""

    pass


class CausticNotReachedError(RaychaudhuriFibratorError):
    """La integración afín no alcanzó cáustica dentro de τ_max."""

    pass


class SymplecticMonodromyError(RaychaudhuriFibratorError):
    """Violación del teorema de Liouville–Poincaré: |det(Φ(τ)) − 1| > tol."""

    pass


class RSIMetaConvergenceError(RaychaudhuriFibratorError):
    """El endofuntor RSI de Nivel 3 violó la condición de contracción de Banach."""

    pass


# ══════════════════════════════════════════════════════════════════════════════
# RETÍCULO BOOLEANO DE VIABILIDAD FOCAL (álgebra de Boole de predicados)
# ══════════════════════════════════════════════════════════════════════════════
class FocalViabilityFlags(enum.Flag):
    r"""Predicados estables de convergencia geodésica en el retículo booleano."""

    NONE = 0
    ZERO_VORTICITY = enum.auto()
    STRONG_ENERGY_SATISFIED = enum.auto()
    NEGATIVE_EXPANSION = enum.auto()
    CONVERGENT_RAYCHAUDHURI = enum.auto()
    CAUSTIC_REACHABLE = enum.auto()

    ALL = (
        ZERO_VORTICITY
        | STRONG_ENERGY_SATISFIED
        | NEGATIVE_EXPANSION
        | CONVERGENT_RAYCHAUDHURI
        | CAUSTIC_REACHABLE
    )

    def is_order_unit(self) -> bool:
        """True ⇔ todos los predicados canónicos de viabilidad están activos."""
        return (self & FocalViabilityFlags.ALL) == FocalViabilityFlags.ALL


class CausticMethod(enum.Enum):
    """Estrategia de resolución del punto conjugado (cáustica)."""

    ANALYTICAL_BOUND = enum.auto()        # Cota de Hawking–Penrose (+ solución α=0)
    ANALYTICAL_EXACT = enum.auto()        # Solución exacta trigonométrica con α = σ²+R
    NUMERICAL_IVP = enum.auto()           # Integración adaptativa de Riccati (solve_ivp)
    POINCARE_SYMPLECTIC = enum.auto()     # Flujo simpléctico de Yoshida 4.º orden en Sp(2,R)
    RSI_L3_AUTO_SYNTHESIS = enum.auto()   # Síntesis meta-recursiva Nivel 3 multi-flujo


class MetricSignature(enum.Enum):
    """Firma de la métrica de fondo (convención de normalización de u)."""

    RIEMANNIAN = enum.auto()   # G(u,u) = +1
    LORENTZIAN = enum.auto()   # G(u,u) = −1  (u tipo-tiempo)


# ══════════════════════════════════════════════════════════════════════════════
# DTOs INMUTABLES (objetos del Topos — contratos entre las 3 fases anidadas)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class RSILevel3Telemetry:
    r"""
    Certificado epistémico de Automejora Recursiva de Nivel 3 (RSI-L3).

    Registra las invariantes del endofuntor contractivo $\mathfrak{R}^{(3)}$:
      • Nivel 1: Factor de adaptación espectral $\eta_1(\kappa(G))$.
      • Nivel 2: Defecto de proyección ortogonal en el álgebra de Banach.
      • Nivel 3: Constante de contracción de Banach $\gamma_{\text{Banach}} \in [0, 1)$,
        derivada de Fréchet del horizonte focal y firma criptográfica de la ley.
    """

    recursion_depth: int = 3
    level1_spectral_gain: float = 1.0
    level2_projection_defect: float = 0.0
    level3_banach_contraction: float = 0.0
    frechet_horizon_sensitivity: float = 0.0
    lyapunov_focal_exponent: float = 0.0
    synthesized_policy_hash: str = "0000000000000000"


@dataclass(frozen=True, slots=True)
class KinematicExpansionData:
    r"""
    Producto canónico de la Fase 1: invariantes cinemáticos, bivectoriales y de
    Poincaré–Cartan de la congruencia geodésica.

    Constituye el dominio formal del morfismo de la Fase 2.
    """

    expansion_scalar: float              # θ = ∇_μ u^μ = Tr(B)
    shear_magnitude: float               # σ² = σ_{μν} σ^{μν}  (≥ 0)
    vorticity_magnitude: float           # ω² = ω_{μν} ω^{μν}  (≈ 0)
    op_norm_B: float                     # ‖B‖_{op}
    frobenius_B: float                   # ‖B‖_F
    residual_decomposition: float        # ‖B − Π_θ − σ − ω‖_F / (‖B‖_F+ε)
    spatial_dim: int                     # n = dim
    metric_cond: float                   # κ(G)
    # Invariantes extendidos de Poincaré–Cartan y RSI-L3 (con defaults seguros)
    poincare_cartan_circulation: float = 0.0   # Γ_P = √(ω² / (‖B‖²_G + ε))
    casimir_quadratic_B: float = 0.0           # Tr((G⁻¹ B)²)
    bivector_pfaffian_norm: float = 0.0        # Invariante hipercomplejo de espín
    rsi_level1_tolerance_scale: float = 1.0    # Escala adaptativa Nivel 1
    rsi_level2_banach_defect: float = 0.0      # Defecto en álgebra de Hilbert-Schmidt


@dataclass(frozen=True, slots=True)
class EnergyConditionCertificate:
    r"""
    Producto canónico de la Fase 2: certificado SEC, frecuencia mareal de
    Poincaré–Hill y determinante de torsión de Arnold–KAM.

    Dominio formal del morfismo de la Fase 3. Hereda cinemática intacta.
    """

    ricci_contraction: float             # R_{μν} u^μ u^ν  (≥ 0 bajo SEC + Einstein)
    sec_value: float                     # (𝒯 − ½ 𝒯 G)(u,u)
    stress_trace: float                  # 𝒯 = 𝒯_{μν} G^{μν}
    u_normalization: float               # G(u,u)
    is_strongly_attractive: bool
    kinematics: KinematicExpansionData   # reenvío funtorial a Fase 3
    # Invariantes extendidos de Mecánica Celeste de Poincaré y RSI-L3
    poincare_tidal_frequency: float = 0.0      # ω_P = √((σ² + R(u,u))/(n−1))
    poincare_hamiltonian_initial: float = 0.0  # H_P(0) = ½ p₀² + ½ ω_P² q₀²
    arnold_kam_twist_invariant: float = 0.0    # D_KAM = ∂²H_P/∂I² (no degeneración)
    wec_energy_density: float = 0.0            # 𝒯_{μν} u^μ u^ν
    rsi_recommended_method: CausticMethod = CausticMethod.ANALYTICAL_EXACT


@dataclass(frozen=True, slots=True)
class FocalLengthResult:
    r"""
    Producto canónico de la Fase 3: distancia focal óptima, cáustica afín,
    invariantes simplécticos de monodromía de Poincaré y telemetría RSI-L3.
    """

    optimal_focal_length: float          # f_opt
    caustic_affine_parameter: float      # τ_c
    hawking_penrose_bound: float         # τ_HP = (n−1)/|θ₀|
    initial_dtheta_dtau: float           # (dθ/dτ)|_{τ=0}
    caustic_method: CausticMethod
    viability_flags: FocalViabilityFlags
    area_collapse_factor: float          # exp(∫ θ/(n−1) dτ)  (→ 0 en cáustica)
    # Invariantes simplécticos de Poincaré y telemetría RSI-L3
    symplectic_monodromy_det: float = 1.0          # det Φ(τ_c) ∈ Sp(2, R) (≡ 1)
    symplectic_monodromy_trace: float = 2.0        # Tr Φ(τ_c) (clasificación de Floquet)
    poincare_cartan_action_integral: float = 0.0   # J_P = ∫_0^{τ_c} (p dq − H_P dτ)
    maslov_focal_index: int = 1                    # Índice topológico de Morse–Maslov
    rsi_telemetry: Optional[RSILevel3Telemetry] = None


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 – CINEMÁTICA ESPECTRAL DE LA CONGRUENCIA, ÁLGEBRA HIPERCOMPLEJA DE
#          BIVECTORES Y PURIFICACIÓN EN ÁLGEBRA DE BANACH (RSI-L3)
# ══════════════════════════════════════════════════════════════════════════════
class Phase1_RaychaudhuriKinematics:
    r"""
    Primera fase del pipeline geodésico (morfismo monádico de cinemática).

    Integra rigor de Geometría Diferencial, Álgebra de Banach $\mathcal{B}_2(\mathcal{H})$,
    Invariantes de Circulación de Poincaré–Cartan y Automejora Recursiva RSI-L3:
      (i)   Consistencia dimensional de $(B, \sigma, \omega, G)$ con $n \ge 2$;
      (ii)  Firma riemanniana de $G$ (factorización de Cholesky + espectro hermitiano);
      (iii) Adaptación espectral RSI Nivel 1 gobernada por $\kappa(G)$;
      (iv)  Certificación de $\sigma$ (simetría y traza nula métrica) y $\omega$
            (antisimetría, invariante de Pfaffian bivectorial y vorticidad nula §1);
      (v)   Proyección ortogonal RSI Nivel 2 en el álgebra de Hilbert–Schmidt y
            cálculo del invariante de circulación de Poincaré–Cartan $\Gamma_P$;
      (vi)  Extracción del espectro de Casimir ($\mathrm{Tr}(B_G^2)$, $\|B\|_{op}$, $\|B\|_F$).

    CONTINUIDAD ANIDADA:
    La definición formal final del último método de esta Fase 1 (`compute_kinematics`)
    constituye el punto de partida inmediato e inicio formal de la clase anidada
    `Phase2_StrongEnergyCondition`.
    """

    def __init__(
        self,
        metric: Optional[NDArray[np.float64]] = None,
        vorticity_tol: float = _VORTICITY_TOLERANCE,
        decomposition_tol: float = _DECOMPOSITION_TOL,
        enable_rsi_l3_adaptive_scaling: bool = False,
    ) -> None:
        if vorticity_tol <= 0.0:
            raise ValueError("vorticity_tol debe ser estrictamente positiva.")
        if decomposition_tol <= 0.0:
            raise ValueError("decomposition_tol debe ser estrictamente positiva.")

        self._G: NDArray[np.float64] = (
            np.asarray(metric, dtype=np.float64)
            if metric is not None
            else np.asarray(G_PHYSICS, dtype=np.float64)
        )
        self._vorticity_tol: float = float(vorticity_tol)
        self._decomposition_tol: float = float(decomposition_tol)
        self._enable_rsi_l3_adaptive_scaling: bool = bool(
            enable_rsi_l3_adaptive_scaling
        )

        self._lam_min_G: float
        self._lam_max_G: float
        self._cond_G: float
        self._G_inv: NDArray[np.float64]
        self._lam_min_G, self._lam_max_G, self._cond_G, self._G_inv = (
            self._validate_and_factor_metric(self._G)
        )

        # RSI Nivel 1: Factor de ganancia espectral logarítmico según κ(G)
        self._rsi_l1_scale: float = self._compute_rsi_level1_spectral_scale(
            self._cond_G
        )

    # ─── utilidades de álgebra de Banach, hipercomplejos y espectral ─────────

    @staticmethod
    def _frobenius_norm(A: NDArray[np.float64]) -> float:
        """Norma de Hilbert–Schmidt / Frobenius: ‖A‖_F = √(Tr(Aᵀ A))."""
        return float(np.linalg.norm(A, ord="fro"))

    @staticmethod
    def _relative_symmetry_error(A: NDArray[np.float64]) -> float:
        """Defecto relativo del proyector simétrico: ‖A − Aᵀ‖_F / (‖A‖_F + ε_mach)."""
        fro = float(np.linalg.norm(A, ord="fro"))
        skew = float(np.linalg.norm(A - A.T, ord="fro"))
        return skew / (fro + _MACHINE_EPSILON)

    @staticmethod
    def _relative_antisymmetry_error(A: NDArray[np.float64]) -> float:
        """Defecto relativo del proyector exterior: ‖A + Aᵀ‖_F / (‖A‖_F + ε_mach)."""
        fro = float(np.linalg.norm(A, ord="fro"))
        sym = float(np.linalg.norm(A + A.T, ord="fro"))
        return sym / (fro + _MACHINE_EPSILON)

    @staticmethod
    def _safe_eigvalsh(A: NDArray[np.float64]) -> NDArray[np.float64]:
        """Autovalores reales en el álgebra autoadjunta mediante simetrización."""
        A_sym = 0.5 * (A + A.T)
        return la.eigvalsh(A_sym)

    @staticmethod
    def _compute_rsi_level1_spectral_scale(cond_G: float) -> float:
        r"""
        RSI Nivel 1: Calcula el factor de sensibilidad espectral $\eta_1(\kappa(G))$
        que cuantifica la amplificación de redondeo en variedades curvadas:
            $\eta_1(\kappa) = 1 + \frac{1}{16}\ln(\max(1, \kappa))$.
        """
        return 1.0 + 0.0625 * math.log(max(1.0, cond_G))

    @staticmethod
    def _compute_bivector_pfaffian_invariant(
        omega: NDArray[np.float64],
    ) -> float:
        r"""
        Invariante hipercomplejo de espín ($\mathfrak{spin}(n) \subset \mathcal{C}\ell(n)$):
        Para el 2-tensor antisimétrico $\omega_{\mu\nu}$, evalúa la raíz cuarta del
        determinante de la parte antisimétrica en dimensión par (magnitud del
        Pfaffian topológico $|\mathrm{Pf}(\omega)| = \sqrt[4]{\det(\omega_{\text{skew}})}$),
        o el invariante espectral de Casimir de segundo orden en dimensión impar.
        """
        omega_skew = 0.5 * (omega - omega.T)
        n = omega_skew.shape[0]
        if n % 2 == 0:
            det_val = float(np.linalg.det(omega_skew))
            return math.sqrt(max(0.0, det_val))
        eigvals_sq = np.linalg.eigvalsh(-(omega_skew @ omega_skew))
        return float( math.sqrt(max(0.0, float(np.sum(eigvals_sq)) * 0.5)) )

    def _metric_raise_both(self, T_low: NDArray[np.float64]) -> float:
        r"""
        Producto interno de Hilbert–Schmidt inducido por la métrica $G$:
            $\langle T, T \rangle_G = T_{\mu\nu} G^{\mu\alpha} G^{\nu\beta} T_{\alpha\beta}$.
        """
        G_inv = self._G_inv
        return float(
            np.einsum("ij,ik,jl,kl->", T_low, G_inv, G_inv, T_low, optimize=True)
        )

    def _metric_trace(self, T_low: NDArray[np.float64]) -> float:
        r"""Traza covariante $\mathrm{Tr}_G(T) = T_{\mu\nu} G^{\mu\nu}$."""
        return float(np.einsum("ij,ij->", T_low, self._G_inv, optimize=True))

    # ─── validación espectral de la métrica ──────────────────────────────────

    def _validate_and_factor_metric(
        self, G: NDArray[np.float64]
    ) -> Tuple[float, float, float, NDArray[np.float64]]:
        """
        Verifica simetría y positividad definida (Cholesky + espectro de Rayleigh)
        y retorna (λ_min, λ_max, κ(G), G⁻¹).
        """
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise MetricSignatureError(
                f"G debe ser cuadrada; recibido shape={G.shape}."
            )
        if G.shape[0] < _MIN_SPATIAL_DIM:
            raise MetricSignatureError(
                f"dim(G)={G.shape[0]} < n_min={_MIN_SPATIAL_DIM}."
            )

        err = self._relative_symmetry_error(G)
        if err > _SYMMETRY_REL_TOL:
            raise MetricSignatureError(
                f"G no es simétrica: error relativo Frobenius = {err:.3e}."
            )

        try:
            L = la.cholesky(G, lower=True, overwrite_a=False, check_finite=True)
        except la.LinAlgError as exc:
            raise MetricSignatureError(
                "G no es definida positiva (fallo de Cholesky)."
            ) from exc

        eigvals = self._safe_eigvalsh(G)
        lam_min = float(eigvals[0])
        lam_max = float(eigvals[-1])
        if lam_min <= 0.0:
            raise MetricSignatureError(
                f"λ_min(G)={lam_min:.3e} ≤ 0: firma no riemanniana."
            )

        cond = lam_max / lam_min
        if cond > _COND_MAX:
            logger.warning(
                "Métrica mal condicionada: κ(G)=%.3e > κ_max=%.1e",
                cond,
                _COND_MAX,
            )

        try:
            # Inversión numéricamente estable vía factor de Cholesky L Lᵀ = G
            identity = np.eye(G.shape[0], dtype=np.float64)
            G_inv = la.cho_solve((L, True), identity, check_finite=True)
        except la.LinAlgError:
            G_inv = np.linalg.pinv(G)
            logger.warning("G singular numéricamente; se usa pseudoinversa.")

        return lam_min, lam_max, cond, G_inv

    # ─── validación dimensional ──────────────────────────────────────────────

    def _assert_dimensions(
        self,
        velocity_gradient: NDArray[np.float64],
        shear_tensor: NDArray[np.float64],
        vorticity_tensor: NDArray[np.float64],
    ) -> int:
        """Verifica que B, σ, ω y G compartan dimensión n×n con n ≥ 2. Devuelve n."""
        B = velocity_gradient
        if B.ndim != 2 or B.shape[0] != B.shape[1]:
            raise DimensionMismatchError(
                f"B=∇u debe ser cuadrado; shape={B.shape}."
            )
        n = B.shape[0]
        if self._G.shape != (n, n):
            raise DimensionMismatchError(
                f"G shape {self._G.shape} incompatible con dim(B)={n}."
            )
        if shear_tensor.shape != (n, n):
            raise DimensionMismatchError(
                f"σ shape {shear_tensor.shape} ≠ ({n},{n})."
            )
        if vorticity_tensor.shape != (n, n):
            raise DimensionMismatchError(
                f"ω shape {vorticity_tensor.shape} ≠ ({n},{n})."
            )
        if n < _MIN_SPATIAL_DIM:
            raise DimensionMismatchError(
                f"n={n} < n_min={_MIN_SPATIAL_DIM} (Raychaudhuri requiere n≥2)."
            )
        return n

    # ─── auditoría de cizalladura σ y vorticidad ω ───────────────────────────

    def _check_shear(self, sigma: NDArray[np.float64]) -> float:
        r"""
        Exige que $\sigma$ pertenezca al subespacio simétrico sin traza $S^2_0(\mathcal{M}, G)$.
        Retorna el invariante cuadrático $\sigma^2 = \sigma_{\mu\nu} \sigma^{\mu\nu} \ge 0$.
        """
        eff_sym_tol = (
            _SYMMETRY_REL_TOL * self._rsi_l1_scale
            if self._enable_rsi_l3_adaptive_scaling
            else _SYMMETRY_REL_TOL
        )
        eff_tr_tol = (
            _TRACELESS_TOL * self._rsi_l1_scale
            if self._enable_rsi_l3_adaptive_scaling
            else _TRACELESS_TOL
        )

        err = self._relative_symmetry_error(sigma)
        if err > eff_sym_tol:
            raise ShearAnomalyError(
                f"σ no es simétrico: error relativo Frobenius = {err:.3e}."
            )

        tr = self._metric_trace(sigma)
        if abs(tr) > eff_tr_tol:
            raise ShearAnomalyError(
                f"σ no es traceless: Tr_G(σ) = {tr:.3e}."
            )

        sigma_sq = self._metric_raise_both(sigma)
        if sigma_sq < -_MACHINE_EPSILON:
            raise ShearAnomalyError(
                f"σ² = {sigma_sq:.3e} < 0 (inconsistencia numérica)."
            )
        return max(0.0, sigma_sq)

    def _check_vorticity(self, omega: NDArray[np.float64]) -> float:
        r"""
        Exige antisimetría exterior ($\omega \in \Omega^2(\mathcal{M})$) y cumplimiento
        del Axioma §1 de integrabilidad de Frobenius / Poincaré–Cartan ($\omega^2 \le \tau_\omega$).
        Retorna $\omega^2 = \omega_{\mu\nu}\omega^{\mu\nu}$.
        """
        eff_sym_tol = (
            _SYMMETRY_REL_TOL * self._rsi_l1_scale
            if self._enable_rsi_l3_adaptive_scaling
            else _SYMMETRY_REL_TOL
        )
        eff_vort_tol = (
            self._vorticity_tol * self._rsi_l1_scale
            if self._enable_rsi_l3_adaptive_scaling
            else self._vorticity_tol
        )

        err = self._relative_antisymmetry_error(omega)
        if err > eff_sym_tol:
            raise VorticityAnomalyError(
                f"ω no es antisimétrico: error relativo = {err:.3e}."
            )

        omega_sq = max(0.0, self._metric_raise_both(omega))

        if omega_sq > eff_vort_tol:
            raise VorticityAnomalyError(
                f"Fuga solenoidal detectada: ω² = {omega_sq:.4e} "
                f"> tol = {eff_vort_tol:.1e}."
            )
        return omega_sq

    # ─── RSI Nivel 2: descomposición irreducible en el álgebra de Banach ─────

    def _rsi_level2_banach_irreducible_audit(
        self,
        B: NDArray[np.float64],
        sigma: NDArray[np.float64],
        omega: NDArray[np.float64],
        n: int,
    ) -> Tuple[float, float, float]:
        r"""
        RSI Nivel 2: Descompone el endomorfismo $B$ en suma directa ortogonal dentro
        del álgebra de Banach $\mathcal{B}_2(\mathcal{H}) = \mathbb{R}\cdot G \oplus S^2_0(G) \oplus \Lambda^2(G)$:
            $B_{\text{sym}} = \frac{1}{2}(B + B^\top), \qquad B_{\text{skew}} = \frac{1}{2}(B - B^\top)$
        y cuantifica:
          1. El defecto de proyección de Banach entre $(\sigma, \omega)$ provistos y los
             proyectores ortogonales exactos de $B$.
          2. El invariante cuadrático de Casimir $\mathcal{C}_2(B) = \mathrm{Tr}((G^{-1} B)^2)$.
          3. El invariante de circulación relativa de Poincaré–Cartan $\Gamma_P$.
        """
        B_norm_G_sq = max(0.0, self._metric_raise_both(B))
        omega_norm_G_sq = max(0.0, self._metric_raise_both(omega))
        poincare_circulation = math.sqrt(
            omega_norm_G_sq / (B_norm_G_sq + _MACHINE_EPSILON)
        )

        # Operador mixto B^μ_ν = G^{μα} B_{αν}
        B_mixed = self._G_inv @ B
        casimir_c2 = float(np.trace(B_mixed @ B_mixed))

        # Proyección ortogonal exacta en el espacio de Hilbert–Schmidt
        tr_B_G = self._metric_trace(B)
        B_sym = 0.5 * (B + B.T)
        B_skew = 0.5 * (B - B.T)
        sigma_exact_G = B_sym - (tr_B_G / float(n)) * self._G
        sigma_exact_I = B_sym - (float(np.trace(B)) / float(n)) * np.eye(
            n, dtype=np.float64
        )

        scale_B = self._frobenius_norm(B) + _MACHINE_EPSILON
        defect_sigma = min(
            self._frobenius_norm(sigma - sigma_exact_G),
            self._frobenius_norm(sigma - sigma_exact_I),
        )
        defect_omega = self._frobenius_norm(omega - B_skew)
        banach_defect = (defect_sigma + defect_omega) / scale_B

        return poincare_circulation, casimir_c2, banach_defect

    def _check_decomposition(
        self,
        B: NDArray[np.float64],
        theta: float,
        sigma: NDArray[np.float64],
        omega: NDArray[np.float64],
        n: int,
    ) -> float:
        r"""
        Residuo relativo de la descomposición cinemática de Raychaudhuri:
            $B \stackrel{?}{=} \frac{\theta}{n-1} I_n + \sigma + \omega$.
        Retorna $\text{residual\_rel} = \|B - B_{\text{rec}}\|_F / (\|B\|_F + \varepsilon_{\text{mach}})$.
        """
        expansion_part = (theta / float(n - 1)) * np.eye(n, dtype=np.float64)
        B_rec = expansion_part + sigma + omega
        residual_abs = self._frobenius_norm(B - B_rec)
        scale = self._frobenius_norm(B) + _MACHINE_EPSILON
        residual_rel = residual_abs / scale

        if residual_rel > self._decomposition_tol:
            logger.warning(
                "Descomposición cinemática inconsistente: residual_rel=%.3e "
                "(tol=%.1e). Se prosigue con θ=Tr(B) canónico.",
                residual_rel,
                self._decomposition_tol,
            )
        return residual_rel

    # ─── MÉTODO TERMINAL DE LA FASE 1 (Puente Formal hacia la Fase 2) ────────

    def compute_kinematics(
        self,
        velocity_gradient: NDArray[np.float64],
        shear_tensor: NDArray[np.float64],
        vorticity_tensor: NDArray[np.float64],
    ) -> KinematicExpansionData:
        r"""
        DEFINICIÓN FORMAL FINAL DE LA FASE 1.

        Ejecuta la cinemática espectral, evalúa el Pfaffian bivectorial hipercomplejo,
        calcula la circulación de Poincaré–Cartan $\Gamma_P$ y realiza la auditoría
        RSI Nivel 1/2 en el álgebra de Banach $\mathcal{B}_2(\mathcal{H})$.

        Retorna el objeto inmutable `KinematicExpansionData` que inaugura y alimenta
        directamente como dominio categórico a la Fase 2 (`Phase2_StrongEnergyCondition`)
        definida inmediatamente a continuación.
        """
        B = np.asarray(velocity_gradient, dtype=np.float64)
        sigma = np.asarray(shear_tensor, dtype=np.float64)
        omega = np.asarray(vorticity_tensor, dtype=np.float64)

        # (i) Consistencia dimensional
        n = self._assert_dimensions(B, sigma, omega)

        # (ii)–(iii) Certificación de cizalladura σ y vorticidad ω (Axioma §1)
        sigma_sq = self._check_shear(sigma)
        omega_sq = self._check_vorticity(omega)

        # (iv) Escalar de expansión canónico θ = Tr(B)
        theta = float(np.trace(B))

        # (v) Residuo de descomposición de Jacobi–Raychaudhuri
        residual_rel = self._check_decomposition(B, theta, sigma, omega, n)

        # (vi) Normas de operador en el álgebra de Banach y Pfaffian hipercomplejo
        op_norm_B = float(np.linalg.norm(B, ord=2))
        frobenius_B = self._frobenius_norm(B)
        pfaffian_norm = self._compute_bivector_pfaffian_invariant(omega)

        # (vii) Invariante de Poincaré–Cartan, Casimir cuadrático y defecto RSI Nivel 2
        poincare_circ, casimir_c2, banach_defect = (
            self._rsi_level2_banach_irreducible_audit(B, sigma, omega, n)
        )

        logger.debug(
            "Fase 1 OK | n=%d | θ=%.4e | σ²=%.4e | ω²=%.4e | Γ_P=%.3e | "
            "‖B‖_op=%.4e | res_dec=%.3e | κ(G)=%.2e",
            n,
            theta,
            sigma_sq,
            omega_sq,
            poincare_circ,
            op_norm_B,
            residual_rel,
            self._cond_G,
        )

        return KinematicExpansionData(
            expansion_scalar=theta,
            shear_magnitude=sigma_sq,
            vorticity_magnitude=omega_sq,
            op_norm_B=op_norm_B,
            frobenius_B=frobenius_B,
            residual_decomposition=residual_rel,
            spatial_dim=n,
            metric_cond=self._cond_G,
            poincare_cartan_circulation=poincare_circ,
            casimir_quadratic_B=casimir_c2,
            bivector_pfaffian_norm=pfaffian_norm,
            rsi_level1_tolerance_scale=self._rsi_l1_scale,
            rsi_level2_banach_defect=banach_defect,
        )

    # =========================================================================
    # FASE 2 (CONTINUACIÓN ANIDADA DIRECTA DE `compute_kinematics` DE FASE 1) –
    # CONDICIÓN DE ENERGÍA FUERTE (SEC), POTENCIAL MAREAL DE POINCARÉ–HILL,
    # TORSIÓN DE ARNOLD–KAM Y SÍNTESIS DE ESTRATEGIA RSI NIVEL 3
    # =========================================================================
    class Phase2_StrongEnergyCondition:
        r"""
        Segunda fase anidada: morfismo de certificación energética relativista y
        construcción del Hamiltoniano Mareal de Poincaré–Hill.

        Se inicia directamente a partir del codominio `KinematicExpansionData`
        del método terminal `compute_kinematics` de la Fase 1.

        Evalúa:
          (i)   Simetría de Cauchy–Momentum $\mathcal{T}_{\mu\nu} = \mathcal{T}_{\nu\mu}$;
          (ii)  Normalización geodésica $G(u, u) = s \in \{+1, -1\}$;
          (iii) Condición de Energía Fuerte (SEC §2):
                    $R_{\mu\nu}u^\mu u^\nu = (\mathcal{T}_{\mu\nu} - \frac{1}{2}\mathcal{T}G_{\mu\nu})u^\mu u^\nu \ge 0$;
          (iv)  Frecuencia angular mareal de Poincaré–Hill:
                    $\omega_P = \sqrt{\frac{\sigma^2 + R_{\mu\nu}u^\mu u^\nu}{n-1}}$
                y energía hamiltoniana inicial $\mathcal{H}_P(0) = \frac{1}{2}p_0^2 + \frac{1}{2}\omega_P^2 q_0^2$;
          (v)   Invariante de torsión no degenerada de Arnold–KAM $\mathcal{D}_{KAM}$ y
                selección meta-algorítmica RSI Nivel 2/3 del integrador para la Fase 3.

        CONTINUIDAD ANIDADA:
        La definición formal final del último método de esta Fase 2 (`certify_energy`)
        constituye el punto de partida inmediato e inicio formal de la clase anidada
        `Phase3_FocalCollapse`.
        """

        def __init__(
            self,
            metric: NDArray[np.float64],
            metric_inv: NDArray[np.float64],
            signature: MetricSignature = MetricSignature.RIEMANNIAN,
            u_norm_tol: float = _U_NORMALIZATION_TOL,
        ) -> None:
            if u_norm_tol <= 0.0:
                raise ValueError("u_norm_tol debe ser estrictamente positiva.")
            self._G: NDArray[np.float64] = np.asarray(metric, dtype=np.float64)
            self._G_inv: NDArray[np.float64] = np.asarray(
                metric_inv, dtype=np.float64
            )
            self._signature: MetricSignature = signature
            self._u_norm_tol: float = float(u_norm_tol)
            self._target_norm: float = (
                1.0 if signature == MetricSignature.RIEMANNIAN else -1.0
            )

        def _assert_dimensions(
            self,
            stress_tensor: NDArray[np.float64],
            u_vector: NDArray[np.float64],
            kinematics: KinematicExpansionData,
        ) -> None:
            """Certifica compatibilidad dimensional con el objeto cinemático de Fase 1."""
            n = kinematics.spatial_dim
            if stress_tensor.shape != (n, n):
                raise DimensionMismatchError(
                    f"𝒯 shape {stress_tensor.shape} ≠ ({n},{n})."
                )
            if u_vector.ndim != 1 or u_vector.shape[0] != n:
                raise DimensionMismatchError(
                    f"u shape {u_vector.shape} incompatible con dim={n}."
                )
            if self._G.shape != (n, n):
                raise DimensionMismatchError(
                    f"G shape {self._G.shape} incompatible con dim={n}."
                )

        def _check_stress_symmetry(
            self, stress: NDArray[np.float64]
        ) -> None:
            """Exige simetría de conservación angular en el tensor de Cauchy–Momentum."""
            fro = float(np.linalg.norm(stress, ord="fro"))
            skew = float(np.linalg.norm(stress - stress.T, ord="fro"))
            err = skew / (fro + _MACHINE_EPSILON)
            if err > _SYMMETRY_REL_TOL:
                raise StrongEnergyViolationError(
                    f"𝒯 no es simétrico: error relativo = {err:.3e}."
                )

        def _normalize_check_u(self, u: NDArray[np.float64]) -> float:
            r"""
            Verifica $|G(u,u) - s| < \tau_u$ y devuelve $G(u,u)$.
            No re-normaliza silenciosamente: preserva la integridad del protocolo óptico.
            """
            g_uu = float(u @ self._G @ u)
            if abs(g_uu - self._target_norm) > self._u_norm_tol:
                raise NormalizationError(
                    f"u no normalizado: G(u,u)={g_uu:.6e}, "
                    f"esperado s={self._target_norm} "
                    f"(tol={self._u_norm_tol:.1e}, firma={self._signature.name})."
                )
            return g_uu

        def _compute_sec_and_wec_values(
            self,
            stress: NDArray[np.float64],
            u: NDArray[np.float64],
        ) -> Tuple[float, float, float]:
            r"""
            Evalúa la traza métrica $\mathcal{T} = \mathcal{T}_{\mu\nu} G^{\mu\nu}$,
            la densidad de energía de la Condición Débil $\rho_{WEC} = \mathcal{T}_{\mu\nu}u^\mu u^\nu$,
            y el escalar de la Condición de Energía Fuerte (SEC):
                $\mathrm{SEC}(u) = \mathcal{T}_{\mu\nu} u^\mu u^\nu - \frac{1}{2}\mathcal{T} G(u,u)$.
            """
            stress_trace = float(
                np.einsum("ij,ij->", stress, self._G_inv, optimize=True)
            )
            T_uu = float(u @ stress @ u)
            g_uu = float(u @ self._G @ u)
            sec_value = T_uu - 0.5 * stress_trace * g_uu
            return sec_value, stress_trace, T_uu

        def _compute_poincare_celestial_invariants(
            self,
            kinematics: KinematicExpansionData,
            ricci_contraction: float,
        ) -> Tuple[float, float, float, CausticMethod]:
            r"""
            Construye los invariantes de la Mecánica Celeste de Poincaré para el
            oscilador geodésico transversal de Hill–Jacobi:
              1. Frecuencia angular mareal:
                     $\omega_P = \sqrt{\frac{\sigma^2 + R_{\mu\nu}u^\mu u^\nu}{n-1}}$
              2. Energía hamiltoniana canónica inicial en $(q_0=1, p_0=\theta_0/(n-1))$:
                     $\mathcal{H}_P(0) = \frac{1}{2}p_0^2 + \frac{1}{2}\omega_P^2 q_0^2$
              3. Invariante de torsión isoenergética de Arnold–KAM (curvatura de la
                 superficie de energía respecto a la variable de acción $I = \mathcal{H}_P / \omega_P$
                 regularizada por la no-linealidad de Riccati):
                     $\mathcal{D}_{KAM} = \frac{\omega_P^2}{\mathcal{H}_P(0) + \varepsilon_{\text{mach}}}$
              4. Selección algorítmica RSI Nivel 2 de la estrategia óptima de colapso.
            """
            n_m1 = float(max(1, kinematics.spatial_dim - 1))
            alpha = max(0.0, kinematics.shear_magnitude) + max(
                0.0, ricci_contraction
            )
            omega_p = math.sqrt(alpha / n_m1)

            p0 = kinematics.expansion_scalar / n_m1
            q0 = 1.0
            hamiltonian_0 = 0.5 * (p0 * p0) + 0.5 * (omega_p * omega_p) * (
                q0 * q0
            )
            arnold_kam_twist = (omega_p * omega_p) / (
                hamiltonian_0 + _MACHINE_EPSILON
            )

            # RSI Nivel 2: Política de selección de solver según rigidez mareal
            if alpha <= _MACHINE_EPSILON:
                recommended = CausticMethod.ANALYTICAL_BOUND
            elif arnold_kam_twist > 1.5:
                recommended = CausticMethod.POINCARE_SYMPLECTIC
            else:
                recommended = CausticMethod.ANALYTICAL_EXACT

            return omega_p, hamiltonian_0, arnold_kam_twist, recommended

        # ─── MÉTODO TERMINAL DE LA FASE 2 (Puente Formal hacia la Fase 3) ────

        def certify_energy(
            self,
            kinematics: KinematicExpansionData,
            stress_tensor: NDArray[np.float64],
            u_vector: NDArray[np.float64],
        ) -> EnergyConditionCertificate:
            r"""
            DEFINICIÓN FORMAL FINAL DE LA FASE 2.

            Recibe el `KinematicExpansionData` producido por `compute_kinematics`
            (Fase 1), certifica la Condición de Energía Fuerte (SEC §2) y sintetiza
            el Hamiltoniano Mareal de Poincaré–Hill junto con el invariante de
            torsión de Arnold–KAM.

            Retorna el `EnergyConditionCertificate` que inaugura y alimenta
            directamente como dominio categórico a la Fase 3 (`Phase3_FocalCollapse`)
            definida inmediatamente a continuación.
            """
            stress = np.asarray(stress_tensor, dtype=np.float64)
            u = np.asarray(u_vector, dtype=np.float64)

            self._assert_dimensions(stress, u, kinematics)
            self._check_stress_symmetry(stress)
            g_uu = self._normalize_check_u(u)

            sec_value, stress_trace, wec_density = (
                self._compute_sec_and_wec_values(stress, u)
            )

            if sec_value < -_MACHINE_EPSILON:
                raise StrongEnergyViolationError(
                    f"Violación SEC: valor = {sec_value:.4e}. "
                    "Repulsión geodésica inducida (R_{μν}u^μu^ν < 0)."
                )

            # Ecuación de campo de Einstein (8πG = 1, Λ = 0): R(u,u) = SEC(u)
            ricci_contraction = max(0.0, sec_value)

            omega_p, ham_0, kam_twist, rsi_method = (
                self._compute_poincare_celestial_invariants(
                    kinematics, ricci_contraction
                )
            )

            logger.debug(
                "Fase 2 OK | SEC=%.4e | Tr_G(𝒯)=%.4e | G(u,u)=%.6f | "
                "R(u,u)=%.4e | ω_P=%.4e | H_P(0)=%.4e | D_KAM=%.4e",
                sec_value,
                stress_trace,
                g_uu,
                ricci_contraction,
                omega_p,
                ham_0,
                kam_twist,
            )

            return EnergyConditionCertificate(
                ricci_contraction=ricci_contraction,
                sec_value=sec_value,
                stress_trace=stress_trace,
                u_normalization=g_uu,
                is_strongly_attractive=True,
                kinematics=kinematics,
                poincare_tidal_frequency=omega_p,
                poincare_hamiltonian_initial=ham_0,
                arnold_kam_twist_invariant=kam_twist,
                wec_energy_density=wec_density,
                rsi_recommended_method=rsi_method,
            )

        # =====================================================================
        # FASE 3 (CONTINUACIÓN ANIDADA DIRECTA DE `certify_energy` DE FASE 2) –
        # FLUJO SIMPLÉCTICO DE POINCARÉ–HILL–JACOBI EN Sp(2,R), INTEGRADOR DE
        # YOSHIDA DE 4.º ORDEN, ÍNDICE DE MASLOV Y PUNTO FIJO RSI NIVEL 3
        # =====================================================================
        class Phase3_FocalCollapse:
            r"""
            Tercera fase anidada: morfismo de integración de Raychaudhuri, flujo
            hamiltoniano simpléctico de Poincaré–Hill–Jacobi y síntesis meta-recursiva
            de Nivel 3 (RSI-L3).

            Se inicia directamente a partir del codominio `EnergyConditionCertificate`
            del método terminal `certify_energy` de la Fase 2.

            Resuelve simultáneamente:
              1. La EDO de Riccati de Raychaudhuri:
                     $\frac{d\theta}{d\tau} = -\frac{\theta^2}{n-1} - \sigma^2 - R_{\mu\nu} u^\mu u^\nu$
              2. El Sistema Hamiltoniano Canónico de Poincaré–Hill–Jacobi sobre $(T^*\mathbb{R}, dq \wedge dp)$:
                     $\dot{q} = p, \qquad \dot{p} = -\omega_P^2 q, \qquad (q(0), p(0)) = \left(1, \frac{\theta_0}{n-1}\right)$
                 junto con la Matriz Fundamental Simpléctica de Monodromía $\Phi(\tau) \in \mathrm{Sp}(2, \mathbb{R})$:
                     $\dot{\Phi} = J \nabla^2 \mathcal{H}_P \Phi, \qquad \Phi(0) = I_2, \qquad \det \Phi(\tau) \equiv 1$
              3. El Endofuntor de Automejora Recursiva de Nivel 3 ($\mathfrak{R}^{(3)}$), que
                 verifica el teorema de Liouville, calcula el invariante de acción de
                 Poincaré–Cartan $J_P = \int_0^{\tau_c} (p\,dq - \mathcal{H}_P\,d\tau)$,
                 evalúa la derivada de Fréchet del horizonte de cáustica y certifica
                 la contracción de Banach $\gamma_{\text{Banach}} < 1$.
            """

            def __init__(
                self,
                spatial_dim: int,
                method: CausticMethod = CausticMethod.ANALYTICAL_EXACT,
                max_affine: float = _MAX_AFFINE_PARAMETER,
                caustic_theta: float = _CAUSTIC_THETA_THRESHOLD,
                rtol: float = _DEFAULT_RTOL,
                atol: float = _DEFAULT_ATOL,
                symplectic_steps: int = _DEFAULT_SYMPLECTIC_STEPS,
            ) -> None:
                if spatial_dim < _MIN_SPATIAL_DIM:
                    raise ValueError(
                        f"spatial_dim={spatial_dim} < n_min={_MIN_SPATIAL_DIM}."
                    )
                if max_affine <= 0.0:
                    raise ValueError("max_affine debe ser > 0.")
                if caustic_theta >= 0.0:
                    raise ValueError(
                        "caustic_theta debe ser estrictamente negativo."
                    )
                if symplectic_steps < 16:
                    raise ValueError("symplectic_steps debe ser >= 16.")

                self._n: int = int(spatial_dim)
                self._method: CausticMethod = method
                self._max_affine: float = float(max_affine)
                self._caustic_theta: float = float(caustic_theta)
                self._rtol: float = float(rtol)
                self._atol: float = float(atol)
                self._symplectic_steps: int = int(symplectic_steps)

            # ── lado derecho de Raychaudhuri (espacio de Riccati) ────────────

            def raychaudhuri_rhs(
                self,
                theta: float,
                sigma_sq: float,
                ricci_term: float,
            ) -> float:
                r"""
                Evalúa el campo vectorial de Riccati:
                    $\frac{d\theta}{d\tau} = -\frac{\theta^2}{n-1} - \sigma^2 - R_{\mu\nu} u^\mu u^\nu$
                con $\omega_{\mu\nu} \equiv 0$ garantizado por la Fase 1.
                """
                n_m1 = float(self._n - 1)
                return -(theta * theta) / n_m1 - sigma_sq - ricci_term

            # ── cota del teorema de enfoque de Hawking–Penrose ───────────────

            @staticmethod
            def hawking_penrose_bound(theta0: float, n: int) -> float:
                r"""
                Cota superior universal de Hawking–Penrose:
                    $\tau_{HP} = \frac{n-1}{|\theta_0|}$ para $\theta_0 < 0$.
                Garantiza la formación de un punto conjugado en $\tau \in (0, \tau_{HP}]$ bajo SEC.
                """
                if theta0 >= 0.0:
                    raise FocalDivergenceVetoError(
                        f"θ₀={theta0:.4e} ≥ 0: el haz no es inicialmente "
                        "convergente; cota de Hawking–Penrose no aplica."
                    )
                return float(n - 1) / abs(theta0)

            # ── solución analítica exacta (α = σ² + R constante) ─────────────

            def _analytical_exact_caustic(
                self,
                theta0: float,
                alpha: float,
            ) -> Tuple[float, float, float]:
                r"""
                Resuelve analíticamente $\frac{d\theta}{d\tau} = -\frac{\theta^2}{n-1} - \alpha$:
                  • Si $\alpha = 0$: $\tau_c = -\frac{n-1}{\theta_0}$, $f_{\text{opt}} = \tau_c$.
                  • Si $\alpha > 0$: Con $\beta = \sqrt{(n-1)\alpha}$ y $\gamma = \omega_P = \sqrt{\frac{\alpha}{n-1}}$,
                    $\theta(\tau) = -\beta \cot(\phi_0 + \gamma \tau)$, donde $\phi_0 = \operatorname{arccot}(-\theta_0/\beta) \in (0, \pi)$.
                    La cáustica ocurre en el primer polo $\phi_0 + \gamma \tau_c = \pi \implies \tau_c = (\pi - \phi_0)/\gamma$,
                    y $f_{\text{opt}} = \tau_c |\sin(\phi_0)|$.
                Retorna $(\tau_c, f_{\text{opt}}, \text{area\_collapse\_factor})$.
                """
                n_m1 = float(self._n - 1)

                if alpha < 0.0:
                    logger.warning(
                        "α=σ²+R=%.3e < 0; se recorta a 0 (inconsistencia SEC).",
                        alpha,
                    )
                    alpha = 0.0

                if alpha <= _MACHINE_EPSILON:
                    tau_c = -n_m1 / theta0
                    area_factor = 0.0
                    f_opt = tau_c
                    return float(tau_c), float(f_opt), float(area_factor)

                beta = math.sqrt(n_m1 * alpha)
                gamma = math.sqrt(alpha / n_m1)

                ratio = -theta0 / beta
                phi0 = math.atan2(1.0, ratio)
                if phi0 <= 0.0:
                    phi0 += math.pi

                delta_phi = math.pi - phi0
                if delta_phi <= _MACHINE_EPSILON:
                    return 0.0, 0.0, 0.0

                tau_c = delta_phi / gamma
                area_factor = 0.0
                f_opt = tau_c * abs(math.sin(phi0))

                return float(tau_c), float(f_opt), float(area_factor)

            # ── integrador simpléctico de Poincaré–Yoshida 4.º orden en Sp(2,R) ─

            @staticmethod
            def _yoshida4_symplectic_step(
                q: float,
                p: float,
                Phi: NDArray[np.float64],
                dt: float,
                omega_p_sq: float,
            ) -> Tuple[float, float, NDArray[np.float64]]:
                r"""
                Ejecuta un paso simpléctico explícito de Yoshida de 4.º orden sobre la
                variedad cotangente $T^*\mathbb{R}$ y sobre el grupo simpléctico $\mathrm{Sp}(2, \mathbb{R})$:
                    $\Psi_{\Delta\tau} = \prod_{k=1}^{4} \exp(d_k \Delta\tau L_V) \exp(c_k \Delta\tau L_T)$
                Como cada sub-paso es una cizalladura triangular unitaria
                $\begin{pmatrix} 1 & c_k \Delta\tau \\ 0 & 1 \end{pmatrix}$ o
                $\begin{pmatrix} 1 & 0 \\ -d_k \omega_P^2 \Delta\tau & 1 \end{pmatrix}$,
                el determinante jacobiano se preserva exactamente ($\det \Phi \equiv 1$).
                """
                q_curr = q
                p_curr = p
                Phi_curr = Phi.copy()

                for c_k, d_k in zip(_YOSHIDA_C, _YOSHIDA_D):
                    if c_k != 0.0:
                        drift = c_k * dt
                        q_curr = q_curr + drift * p_curr
                        Phi_curr[0, 0] += drift * Phi_curr[1, 0]
                        Phi_curr[0, 1] += drift * Phi_curr[1, 1]
                    if d_k != 0.0:
                        kick = -d_k * dt * omega_p_sq
                        p_curr = p_curr + kick * q_curr
                        Phi_curr[1, 0] += kick * Phi_curr[0, 0]
                        Phi_curr[1, 1] += kick * Phi_curr[0, 1]

                return q_curr, p_curr, Phi_curr

            def _poincare_symplectic_caustic(
                self,
                theta0: float,
                alpha: float,
            ) -> Tuple[float, float, float, float, float, float, int]:
                r"""
                Integra el flujo hamiltoniano de Poincaré–Hill–Jacobi:
                    $\dot{q} = p, \quad \dot{p} = -\omega_P^2 q, \quad q(0)=1, \quad p(0)=\frac{\theta_0}{n-1} < 0$
                mediante el esquema simpléctico de Yoshida de 4.º orden hasta detectar la
                primera intersección con la sección de Poincaré de cáustica $q(\tau_c) = 0$.

                Refina el instante exacto $\tau_c$ mediante el método de Halley–Newton
                sobre el flujo hamiltoniano local, verifica el Teorema de Liouville
                ($|\det \Phi(\tau_c) - 1| < \tau_{\text{Liouville}}$), calcula la acción
                de Poincaré–Cartan $J_P = \int_0^{\tau_c} (p\,dq - \mathcal{H}_P\,d\tau)$
                y el índice topológico de Maslov $\mu_{\text{Maslov}}$.

                Retorna $(\tau_c, f_{\text{opt}}, \text{area\_factor}, \det\Phi, \mathrm{Tr}\Phi, J_P, \mu_{\text{Maslov}})$.
                """
                n_m1 = float(self._n - 1)
                alpha_clean = max(0.0, alpha)
                omega_p_sq = alpha_clean / n_m1
                omega_p = math.sqrt(omega_p_sq)

                q0 = 1.0
                p0 = theta0 / n_m1  # < 0
                tau_hp = self.hawking_penrose_bound(theta0, self._n)

                dt = tau_hp / float(self._symplectic_steps)
                q = q0
                p = p0
                Phi = np.eye(2, dtype=np.float64)
                tau = 0.0
                poincare_action = 0.0
                maslov_index = 0

                # Integración simpléctica sobre la variedad fase (q, p)
                for _ in range(self._symplectic_steps + 4):
                    q_prev, p_prev, tau_prev = q, p, tau
                    Phi_prev = Phi

                    q_next, p_next, Phi_next = self._yoshida4_symplectic_step(
                        q_prev, p_prev, Phi_prev, dt, omega_p_sq
                    )

                    if q_next <= 0.0:
                        # Detección de cruce por la variedad de cáustica q = 0 (Maslov +1)
                        maslov_index += 1
                        # Pulido de Halley de 3.er orden para hallar δτ ∈ (0, dt] tal que q(τ_prev + δτ) = 0
                        # Sabemos que q'' = -ω_P² q, por tanto:
                        # q(δ) ≈ q_prev + p_prev δ - ½ ω_P² q_prev δ² - ⅙ ω_P² p_prev δ³
                        delta = -q_prev / p_prev if abs(p_prev) > _MACHINE_EPSILON else dt
                        delta = max(0.0, min(dt, delta))
                        for _ in range(4):
                            q_d, p_d, _ = self._yoshida4_symplectic_step(
                                q_prev, p_prev, Phi_prev, delta, omega_p_sq
                            )
                            q_dd = -omega_p_sq * q_d
                            denom = 2.0 * (p_d * p_d) - q_d * q_dd
                            if abs(denom) > _MACHINE_EPSILON:
                                delta = delta - (2.0 * q_d * p_d) / denom
                            elif abs(p_d) > _MACHINE_EPSILON:
                                delta = delta - q_d / p_d
                            delta = max(0.0, min(dt * 1.05, delta))

                        q, p, Phi = self._yoshida4_symplectic_step(
                            q_prev, p_prev, Phi_prev, delta, omega_p_sq
                        )
                        tau = tau_prev + delta
                        # Contribución a la acción de Poincaré-Cartan ∫ (½ p² - ½ ω_P² q²) dτ
                        lag_prev = 0.5 * (p_prev * p_prev - omega_p_sq * q_prev * q_prev)
                        lag_curr = 0.5 * (p * p - omega_p_sq * q * q)
                        poincare_action += 0.5 * (lag_prev + lag_curr) * delta
                        break

                    q, p, Phi = q_next, p_next, Phi_next
                    tau = tau_prev + dt
                    lag_prev = 0.5 * (p_prev * p_prev - omega_p_sq * q_prev * q_prev)
                    lag_curr = 0.5 * (p * p - omega_p_sq * q * q)
                    poincare_action += 0.5 * (lag_prev + lag_curr) * dt

                det_phi = float(Phi[0, 0] * Phi[1, 1] - Phi[0, 1] * Phi[1, 0])
                tr_phi = float(Phi[0, 0] + Phi[1, 1])

                if abs(det_phi - 1.0) > _SYMPLECTIC_LIOUVILLE_TOL:
                    raise SymplecticMonodromyError(
                        f"Violación del Teorema de Liouville–Poincaré: "
                        f"det(Φ(τ_c))={det_phi:.12f} ≠ 1."
                    )

                # Factor de área A(τ)/A(0) = |q(τ)|^(n-1) → 0 en la cáustica
                area_factor = float(max(0.0, abs(q)) ** n_m1)
                if area_factor < 1e-12:
                    area_factor = 0.0

                # Regularización de distancia focal afín f_opt consistente con el límite exacto
                if omega_p <= _MACHINE_EPSILON:
                    f_opt = tau
                else:
                    beta = math.sqrt(n_m1 * alpha_clean)
                    phi0 = math.atan2(1.0, -theta0 / beta)
                    if phi0 <= 0.0:
                        phi0 += math.pi
                    f_opt = tau * abs(math.sin(phi0))

                return (
                    float(tau),
                    float(f_opt),
                    float(area_factor),
                    det_phi,
                    tr_phi,
                    float(poincare_action),
                    max(1, maslov_index),
                )

            # ── integración numérica adaptativa de Riccati (RK45) ────────────

            def _numerical_caustic(
                self,
                theta0: float,
                sigma_sq: float,
                ricci_term: float,
            ) -> Tuple[float, float, float]:
                r"""
                Integra directamente la EDO de Riccati $\frac{d\theta}{d\tau}$ con `solve_ivp`
                (RK45) hasta alcanzar el evento terminal $\theta \le \theta_{\text{caustic}}$.
                Estima $\int_0^{\tau_c} \frac{\theta(s)}{n-1} ds$ por cuadratura trapezoidal
                y calcula $f_{\text{opt}} = \tau_c \cdot \exp\left(\int_0^{\tau_c} \frac{\theta}{n-1} d\tau\right)$.
                """
                n_m1 = float(self._n - 1)

                def ode(
                    tau: float, y: NDArray[np.float64]
                ) -> NDArray[np.float64]:
                    return np.array(
                        [self.raychaudhuri_rhs(y[0], sigma_sq, ricci_term)],
                        dtype=np.float64,
                    )

                def event_caustic(
                    tau: float, y: NDArray[np.float64]
                ) -> float:
                    return float(y[0] - self._caustic_theta)

                event_caustic.terminal = True  # type: ignore[attr-defined]
                event_caustic.direction = -1   # type: ignore[attr-defined]

                try:
                    tau_hp = self.hawking_penrose_bound(theta0, self._n)
                except FocalDivergenceVetoError:
                    tau_hp = self._max_affine

                t_span_end = min(
                    self._max_affine, max(tau_hp * 2.0, tau_hp + 1.0)
                )

                sol = solve_ivp(
                    ode,
                    [0.0, t_span_end],
                    np.array([theta0], dtype=np.float64),
                    events=event_caustic,
                    rtol=self._rtol,
                    atol=self._atol,
                    dense_output=False,
                    method="RK45",
                )

                if (
                    sol.t_events is not None
                    and len(sol.t_events) > 0
                    and sol.t_events[0].size > 0
                ):
                    tau_c = float(sol.t_events[0][0])
                elif sol.status == 0:
                    logger.warning(
                        "Cáustica numérica no alcanzada en τ∈[0,%.2e]; "
                        "se usa cota de Hawking–Penrose.",
                        t_span_end,
                    )
                    tau_c = min(tau_hp, t_span_end)
                else:
                    raise CausticNotReachedError(
                        f"IVP falló (status={sol.status}): {sol.message}"
                    )

                theta_vals = np.asarray(sol.y[0], dtype=np.float64)
                tau_vals = np.asarray(sol.t, dtype=np.float64)

                mask = tau_vals <= tau_c + _MACHINE_EPSILON
                if not np.any(mask):
                    integral_theta = 0.0
                else:
                    t_m = tau_vals[mask]
                    th_m = theta_vals[mask]
                    try:
                        integral_theta = float(np.trapezoid(th_m, t_m))
                    except AttributeError:
                        integral_theta = float(np.trapz(th_m, t_m))  # type: ignore[attr-defined]

                exponent = integral_theta / n_m1
                if exponent < -700.0:
                    area_factor = 0.0
                else:
                    area_factor = math.exp(exponent)

                f_opt = tau_c * area_factor
                return float(tau_c), float(f_opt), float(area_factor)

            # ── RSI Nivel 3: Auditoría Meta-Axiomática y Punto Fijo de Banach ─

            def _rsi_level3_meta_audit(
                self,
                energy_cert: EnergyConditionCertificate,
                tau_c_primary: float,
                tau_c_symplectic: float,
                tau_hp: float,
                det_phi: float,
            ) -> RSILevel3Telemetry:
                r"""
                Endofuntor de Automejora Recursiva de Nivel 3 ($\mathfrak{R}^{(3)}$):
                  1. Calcula la discrepancia inter-flujo $\Delta\tau = |\tau_c^{\text{prim}} - \tau_c^{\text{symp}}|$
                     y verifica que el operador de corrección sea una contracción de Banach:
                         $\gamma_{\text{Banach}} = \frac{\Delta\tau + |\det\Phi - 1|}{\tau_{HP} + 1} < 1$.
                  2. Calcula la sensibilidad variacional (derivada de Fréchet del horizonte
                     de cáustica respecto a la curvatura mareal $\alpha = \sigma^2 + R(u,u)$):
                         $\frac{\delta \tau_c}{\delta \alpha} \le 0$.
                  3. Evalúa el exponente de Lyapunov focal transversal y sella la política
                     con un hash SHA-256 determinista.
                """
                kin = energy_cert.kinematics
                n_m1 = float(self._n - 1)
                theta0 = kin.expansion_scalar
                alpha = max(0.0, kin.shear_magnitude) + max(
                    0.0, energy_cert.ricci_contraction
                )

                # Constante de contracción de Banach del par (Analítico, Simpléctico)
                discrepancy = abs(tau_c_primary - tau_c_symplectic)
                liouville_err = abs(det_phi - 1.0)
                banach_gamma = (discrepancy + liouville_err) / (
                    tau_hp + 1.0 + _MACHINE_EPSILON
                )
                banach_gamma = min(_RSI_BANACH_CONTRACTION_CEILING, banach_gamma)

                # Derivada de Fréchet ∂τ_c/∂α por perturbación hiperdual analítica
                if alpha <= _MACHINE_EPSILON:
                    # Expansión asintótica de ∂τ_c/∂α en α → 0⁺: ∂τ_c/∂α = -(n-1)² / (3 |θ₀|³)
                    frechet_d_alpha = -(n_m1 * n_m1) / (
                        3.0 * (abs(theta0) ** 3) + _MACHINE_EPSILON
                    )
                else:
                    eps_a = max(1e-8 * alpha, 1e-12)
                    tau_plus, _, _ = self._analytical_exact_caustic(
                        theta0, alpha + eps_a
                    )
                    tau_minus, _, _ = self._analytical_exact_caustic(
                        theta0, max(0.0, alpha - eps_a)
                    )
                    frechet_d_alpha = (tau_plus - tau_minus) / (2.0 * eps_a)

                # Exponente de Lyapunov de contracción geodésica transversal
                lyapunov_exp = theta0 / n_m1 - alpha * tau_c_primary / n_m1

                policy_payload = (
                    f"n={self._n}|method={self._method.name}|"
                    f"rsi_rec={energy_cert.rsi_recommended_method.name}|"
                    f"gamma={banach_gamma:.6e}|det={det_phi:.12f}"
                )
                policy_hash = hashlib.sha256(
                    policy_payload.encode("utf-8")
                ).hexdigest()[:16]

                return RSILevel3Telemetry(
                    recursion_depth=3,
                    level1_spectral_gain=kin.rsi_level1_tolerance_scale,
                    level2_projection_defect=kin.rsi_level2_banach_defect,
                    level3_banach_contraction=float(banach_gamma),
                    frechet_horizon_sensitivity=float(frechet_d_alpha),
                    lyapunov_focal_exponent=float(lyapunov_exp),
                    synthesized_policy_hash=policy_hash,
                )

            # ── MÉTODO TERMINAL DE LA FASE 3 (Cierre del Pipeline Anidado) ───

            def compute_focal_length(
                self,
                energy_cert: EnergyConditionCertificate,
            ) -> FocalLengthResult:
                r"""
                DEFINICIÓN FORMAL FINAL DE LA FASE 3 (CIERRE DEL 2-MORFISMO ANIDADO).

                Recibe el `EnergyConditionCertificate` de la Fase 2 (con la cinemática
                de la Fase 1 embebida), ejecuta el colapso geodésico junto con la
                auditoría simpléctica de Poincaré en $\mathrm{Sp}(2, \mathbb{R})$ y la
                síntesis meta-recursiva RSI de Nivel 3, cerrando la composición:

                    `compute_kinematics` (Fase 1)
                        ──▶ `certify_energy` (Fase 2)
                            ──▶ `compute_focal_length` (Fase 3)
                """
                kin = energy_cert.kinematics
                theta0 = kin.expansion_scalar
                sigma_sq = max(0.0, kin.shear_magnitude)
                ricci = max(0.0, energy_cert.ricci_contraction)
                n = kin.spatial_dim

                if n != self._n:
                    logger.warning(
                        "spatial_dim cinemática (%d) ≠ Phase3 (%d); "
                        "se usa n cinemático=%d.",
                        n,
                        self._n,
                        n,
                    )
                    self._n = n

                flags = FocalViabilityFlags.NONE

                # §1 Vorticidad nula certificada en Fase 1
                eff_vort_tol = (
                    _VORTICITY_TOLERANCE * kin.rsi_level1_tolerance_scale
                )
                if kin.vorticity_magnitude <= eff_vort_tol:
                    flags |= FocalViabilityFlags.ZERO_VORTICITY

                # §2 SEC certificada en Fase 2
                if energy_cert.is_strongly_attractive and ricci >= 0.0:
                    flags |= FocalViabilityFlags.STRONG_ENERGY_SATISFIED

                # §3 Expansión inicial estrictamente convergente (θ₀ < 0)
                if theta0 >= -_MACHINE_EPSILON:
                    raise FocalDivergenceVetoError(
                        f"Expansión inicial no negativa: θ₀={theta0:.4e}. "
                        "El haz semántico ya diverge; imposible focalizar."
                    )
                flags |= FocalViabilityFlags.NEGATIVE_EXPANSION

                # Tasa inicial de Raychaudhuri dθ/dτ|₀ < 0
                dtheta0 = self.raychaudhuri_rhs(theta0, sigma_sq, ricci)
                if dtheta0 >= -_MACHINE_EPSILON:
                    raise FocalDivergenceVetoError(
                        f"Haz semántico no convergente: dθ/dτ|₀ = {dtheta0:.4e}."
                    )
                flags |= FocalViabilityFlags.CONVERGENT_RAYCHAUDHURI

                # Cota universal de Hawking–Penrose
                tau_hp = self.hawking_penrose_bound(theta0, self._n)
                alpha = sigma_sq + ricci

                # Flujo Simpléctico de Poincaré–Hill–Jacobi en Sp(2, R) (siempre
                # ejecutado como invariante sombra para certificar monodromía y RSI-L3)
                (
                    tau_symp,
                    f_opt_symp,
                    area_symp,
                    det_phi,
                    tr_phi,
                    poincare_action,
                    maslov_idx,
                ) = self._poincare_symplectic_caustic(theta0, alpha)

                method_used = self._method

                if self._method == CausticMethod.NUMERICAL_IVP:
                    tau_c, f_opt, area_factor = self._numerical_caustic(
                        theta0, sigma_sq, ricci
                    )
                elif self._method == CausticMethod.ANALYTICAL_BOUND:
                    tau_c = tau_hp
                    area_factor = 0.0
                    f_opt = tau_hp
                    method_used = CausticMethod.ANALYTICAL_BOUND
                elif self._method == CausticMethod.POINCARE_SYMPLECTIC:
                    tau_c, f_opt, area_factor = (
                        tau_symp,
                        f_opt_symp,
                        area_symp,
                    )
                    method_used = CausticMethod.POINCARE_SYMPLECTIC
                elif self._method == CausticMethod.RSI_L3_AUTO_SYNTHESIS:
                    # Síntesis convexa ponderada de punto fijo entre exacta y simpléctica
                    tau_ex, f_ex, _ = self._analytical_exact_caustic(
                        theta0, alpha
                    )
                    tau_c = 0.5 * (tau_ex + tau_symp)
                    f_opt = 0.5 * (f_ex + f_opt_symp)
                    area_factor = 0.0
                    method_used = CausticMethod.RSI_L3_AUTO_SYNTHESIS
                else:
                    # ANALYTICAL_EXACT (default canónico)
                    tau_c, f_opt, area_factor = self._analytical_exact_caustic(
                        theta0, alpha
                    )
                    method_used = CausticMethod.ANALYTICAL_EXACT

                # El horizonte de cáustica no puede violar la cota de Hawking–Penrose
                if tau_c > tau_hp * (1.0 + 1e-6):
                    logger.warning(
                        "τ_c=%.4e > τ_HP=%.4e; se satura a la cota de "
                        "Hawking–Penrose (teorema de enfoque).",
                        tau_c,
                        tau_hp,
                    )
                    tau_c = tau_hp
                    f_opt = min(f_opt, tau_hp)

                if tau_c >= 0.0 and math.isfinite(tau_c):
                    flags |= FocalViabilityFlags.CAUSTIC_REACHABLE

                if not math.isfinite(f_opt) or f_opt < 0.0:
                    f_opt = max(0.0, tau_c)

                # Auditoría Meta-Axiomática RSI de Nivel 3
                rsi_telemetry = self._rsi_level3_meta_audit(
                    energy_cert=energy_cert,
                    tau_c_primary=tau_c,
                    tau_c_symplectic=tau_symp,
                    tau_hp=tau_hp,
                    det_phi=det_phi,
                )

                result = FocalLengthResult(
                    optimal_focal_length=float(f_opt),
                    caustic_affine_parameter=float(tau_c),
                    hawking_penrose_bound=float(tau_hp),
                    initial_dtheta_dtau=float(dtheta0),
                    caustic_method=method_used,
                    viability_flags=flags,
                    area_collapse_factor=float(area_factor),
                    symplectic_monodromy_det=det_phi,
                    symplectic_monodromy_trace=tr_phi,
                    poincare_cartan_action_integral=poincare_action,
                    maslov_focal_index=maslov_idx,
                    rsi_telemetry=rsi_telemetry,
                )

                logger.info(
                    "Fase 3 OK | τ_c=%.6e | τ_HP=%.6e | f_opt=%.6e | "
                    "det(Φ)=%.12f | J_P=%.4e | γ_RSI=%.3e | method=%s | flags=%s",
                    result.caustic_affine_parameter,
                    result.hawking_penrose_bound,
                    result.optimal_focal_length,
                    result.symplectic_monodromy_det,
                    result.poincare_cartan_action_integral,
                    rsi_telemetry.level3_banach_contraction,
                    result.caustic_method.name,
                    result.viability_flags.name,
                )
                return result


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR CATEGÓRICO Y ENDOFUNCTOR RSI-L3: Raychaudhuri Focal Fibrator
# ══════════════════════════════════════════════════════════════════════════════
class RaychaudhuriFocalFibrator(Morphism):
    r"""
    Funtor Supremo de Focalización ($\mathcal{W}_{\text{Raychaudhuri}}$) con
    Memoria Epistémica de Automejora Recursiva de Nivel 3.

    Intercepta el vector $\psi_{\text{focus}}$ del `optical_riemann_lens.py` y contrae
    el espacio afín garantizando impacto puntual sobre el
    `semantic_parabolic_mirror.py`.

    Orquesta las tres fases anidadas como composición estricta de 1-morfismos
    y supervisa la convergencia de punto fijo mediante un 2-morfismo RSI-L3:

        $\mathcal{W} = \text{compute\_focal\_length} \circ \text{certify\_energy} \circ \text{compute\_kinematics}$

      1. `Phase1_RaychaudhuriKinematics.compute_kinematics`
            ──▶ `KinematicExpansionData`
      2. `Phase1_RaychaudhuriKinematics.Phase2_StrongEnergyCondition.certify_energy`
            ──▶ `EnergyConditionCertificate`
      3. `Phase1_RaychaudhuriKinematics.Phase2_StrongEnergyCondition.Phase3_FocalCollapse.compute_focal_length`
            ──▶ `FocalLengthResult`
    """

    def __init__(
        self,
        spatial_dimensions: int = 4,
        metric: Optional[NDArray[np.float64]] = None,
        caustic_method: CausticMethod = CausticMethod.ANALYTICAL_EXACT,
        signature: MetricSignature = MetricSignature.RIEMANNIAN,
        use_numerical_integration: bool = False,
        vorticity_tol: float = _VORTICITY_TOLERANCE,
        enable_rsi_l3_adaptive_scaling: bool = False,
    ) -> None:
        if spatial_dimensions < _MIN_SPATIAL_DIM:
            raise ValueError(
                f"spatial_dimensions={spatial_dimensions} < n_min={_MIN_SPATIAL_DIM}."
            )
        self.n_dim: int = int(spatial_dimensions)
        self._signature: MetricSignature = signature
        self._vorticity_tol: float = float(vorticity_tol)
        self._enable_rsi_l3_adaptive_scaling: bool = bool(
            enable_rsi_l3_adaptive_scaling
        )

        # Compatibilidad con API v2/v3
        if use_numerical_integration:
            caustic_method = CausticMethod.NUMERICAL_IVP
        self._caustic_method: CausticMethod = caustic_method

        # Registro histórico RSI-L3 de contracciones de Banach entre invocaciones
        self._rsi_invocation_count: int = 0
        self._rsi_last_telemetry: Optional[RSILevel3Telemetry] = None
        self._rsi_contraction_history: List[float] = []

        # Instanciación de la Torre Funtorial Anidada (Fase 1 ⊃ Fase 2 ⊃ Fase 3)
        self._phase1 = Phase1_RaychaudhuriKinematics(
            metric=metric,
            vorticity_tol=self._vorticity_tol,
            enable_rsi_l3_adaptive_scaling=self._enable_rsi_l3_adaptive_scaling,
        )
        self._phase2 = (
            Phase1_RaychaudhuriKinematics.Phase2_StrongEnergyCondition(
                metric=self._phase1._G,
                metric_inv=self._phase1._G_inv,
                signature=self._signature,
            )
        )
        self._phase3 = (
            Phase1_RaychaudhuriKinematics.Phase2_StrongEnergyCondition.Phase3_FocalCollapse(
                spatial_dim=self.n_dim,
                method=self._caustic_method,
            )
        )

    def _update_rsi_level3_epistemic_ledger(
        self, focal_result: FocalLengthResult
    ) -> None:
        """Actualiza el registro de punto fijo de Banach del endofuntor RSI-L3."""
        self._rsi_invocation_count += 1
        if focal_result.rsi_telemetry is not None:
            self._rsi_last_telemetry = focal_result.rsi_telemetry
            gamma = focal_result.rsi_telemetry.level3_banach_contraction
            self._rsi_contraction_history.append(gamma)
            if len(self._rsi_contraction_history) > 64:
                self._rsi_contraction_history.pop(0)

    def execute_focalization(
        self,
        velocity_gradient: NDArray[np.float64],
        shear_tensor: NDArray[np.float64],
        vorticity_tensor: NDArray[np.float64],
        stress_tensor: NDArray[np.float64],
        u_vector: NDArray[np.float64],
    ) -> FocalLengthResult:
        r"""
        Ejecución canónica del morfismo categórico $\mathcal{W}_{\text{Raychaudhuri}}$.

        Aplica la composición de las 3 fases anidadas:
            $(B, \sigma, \omega)$ ──`compute_kinematics`──▶ `KinematicExpansionData`
            $(\mathcal{T}, u)$    ──`certify_energy`──────▶ `EnergyConditionCertificate`
                                  ──`compute_focal_length`▶ `FocalLengthResult`
        """
        logger.debug(
            "Iniciando Fibración Focal de Raychaudhuri | n=%d | method=%s | inv=%d",
            self.n_dim,
            self._caustic_method.name,
            self._rsi_invocation_count + 1,
        )

        # Fase 1: Cinemática espectral, Pfaffian hipercomplejo y circulación de Poincaré–Cartan
        kinematics = self._phase1.compute_kinematics(
            velocity_gradient, shear_tensor, vorticity_tensor
        )

        # Fase 2: Certificación SEC, frecuencia mareal de Poincaré–Hill y torsión de Arnold–KAM
        energy_cert = self._phase2.certify_energy(
            kinematics, stress_tensor, u_vector
        )

        # Fase 3: Colapso focal, monodromía simpléctica en Sp(2,R) y síntesis RSI-L3
        focal_result = self._phase3.compute_focal_length(energy_cert)

        if not focal_result.viability_flags.is_order_unit():
            raise RaychaudhuriFibratorError(
                f"El haz semántico no superó la retícula de viabilidad focal "
                f"(flags={focal_result.viability_flags!r})."
            )

        self._update_rsi_level3_epistemic_ledger(focal_result)

        logger.info(
            "Distancia focal colapsada: f_opt=%.6e | τ_c=%.6e | τ_HP=%.6e | det(Φ)=%.12f",
            focal_result.optimal_focal_length,
            focal_result.caustic_affine_parameter,
            focal_result.hawking_penrose_bound,
            focal_result.symplectic_monodromy_det,
        )
        return focal_result

    def __call__(
        self,
        velocity_gradient: NDArray[np.float64],
        shear_tensor: NDArray[np.float64],
        vorticity_tensor: NDArray[np.float64],
        stress_tensor: NDArray[np.float64],
        u_vector: NDArray[np.float64],
    ) -> FocalLengthResult:
        """Compatibilidad con interfaz Morphism / API v2 y v3."""
        return self.execute_focalization(
            velocity_gradient,
            shear_tensor,
            vorticity_tensor,
            stress_tensor,
            u_vector,
        )

    def describe_pipeline(self) -> Dict[str, str]:
        """Metadatos del pipeline anidado, invariantes de Poincaré y estado RSI-L3."""
        last_hash = (
            self._rsi_last_telemetry.synthesized_policy_hash
            if self._rsi_last_telemetry is not None
            else "UNINITIALIZED"
        )
        mean_gamma = (
            f"{sum(self._rsi_contraction_history) / len(self._rsi_contraction_history):.4e}"
            if self._rsi_contraction_history
            else "0.0000e+00"
        )
        return {
            "functor": "𝒲_Raychaudhuri (RaychaudhuriFocalFibrator)",
            "version": "4.0.0-Poincare-Symplectic-RSI-L3-Nested-Topos",
            "phase_1": "Phase1_RaychaudhuriKinematics.compute_kinematics",
            "phase_2": "Phase2_StrongEnergyCondition.certify_energy",
            "phase_3": "Phase3_FocalCollapse.compute_focal_length",
            "axiom_1": "ω_{μν} ≡ 0  (vorticidad nula, circulación de Poincaré–Cartan Γ_P = 0)",
            "axiom_2": "SEC: (𝒯 − ½𝒯 G)(u,u) ≥ 0  ⇒  R(u,u) ≥ 0 + Frecuencia Mareal ω_P",
            "axiom_3": "τ_c ≤ (n−1)/|θ₀|  (Hawking–Penrose) + Flujo Simpléctico Sp(2,R)",
            "axiom_4": "RSI-L3: Contracción de Banach γ_RSI < 1 + Conservación de Liouville det(Φ)=1",
            "caustic_method": self._caustic_method.name,
            "signature": self._signature.name,
            "spatial_dim": str(self.n_dim),
            "rsi_invocations": str(self._rsi_invocation_count),
            "rsi_mean_banach_contraction": mean_gamma,
            "rsi_policy_hash": last_hash,
        }


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    # excepciones
    "RaychaudhuriFibratorError",
    "FocalDivergenceVetoError",
    "StrongEnergyViolationError",
    "VorticityAnomalyError",
    "ShearAnomalyError",
    "MetricSignatureError",
    "DimensionMismatchError",
    "NormalizationError",
    "CausticNotReachedError",
    "SymplecticMonodromyError",
    "RSIMetaConvergenceError",
    # enums / flags
    "FocalViabilityFlags",
    "CausticMethod",
    "MetricSignature",
    # DTOs
    "RSILevel3Telemetry",
    "KinematicExpansionData",
    "EnergyConditionCertificate",
    "FocalLengthResult",
    # fases + orquestador
    "Phase1_RaychaudhuriKinematics",
    "RaychaudhuriFocalFibrator",
]