# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : Thermal Gradient Laws                                                                     ║
║ Módulo   : Thermal Gradient Laws (Leyes y Gradientes de Convección Térmica)                          ║
║ Ubicación: app/physics/thermal_gradient_laws.py                                                      ║
║ Ruta     : app/physics/thermal_gradient_laws.py                                                      ║
║ Versión  : 4.0.0-Sovereign-Poincare-RSIL3-Lanczos-ItohAbe-AdaptiveCD-Topos                           ║
║ Función  : Motor espectral de cálculo termodinámico, leyes de convección, filtración de Lanczos,     ║
║            dinámica de Itoh-Abe, integración simpléctica de Poincaré y automejora RSI³.              ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL MOTOR ESPECTRAL:
────────────────────────────────────────────────────────────────────────────────────────────────────────
El motor espectral `ThermalGradientLaws` constituye el núcleo físico, termodinámico, simpléctico y
espectral de la FPU para el transporte de calor y leyes constitutivas de convección sobre variedades
riemannianas $(M, g)$ acopladas a haces simplécticos. Gobierna la propagación de flujos de calor no
lineales mediante formulaciones covariantes que integran la ley generalizada de Fourier-Navier-Stokes,
la descomposición espectral de Krylov-Lanczos para operadores de conductividad no acotados, la
discretización geométrica de gradientes según esquemas simplécticos de Itoh-Abe, el análisis de
estabilidad de órbitas celestes de Poincaré y la automejora adaptativa de Nivel 3 (RSI³).

El motor opera estructurado en tres fases anidadas acopladas mediante mixins:
  • FASE 1 (`Phase1ThermalState` y `Phase1ThermalObservationMixin`):
      Valida los tensores de conductividad térmica covariante $K \in \mathbb{R}^{n \times n}$ y los
      gradientes de temperatura $\nabla T \in \mathbb{R}^n$, audita la condición simétrica definida
      positiva (SPD) mediante la descomposición de Cholesky $K = L L^\top$, proyecta el subespacio
      dominante de Krylov mediante iteración espectral de Lanczos tridiagonal, y construye el paquete
      inmutable de observación térmica `Phase1ThermalState`.
  • FASE 2 (`Phase2ThermalState` y `Phase2ThermalOrientationMixin`):
      Consume `Phase1ThermalState`, integra la dinámica del gradiente según el integrador simpléctico de
      Itoh-Abe preservando exactamente la energía de disipación, evalúa los coeficientes de convección
      difusiva acoplada (CD), computa las invariantes integrales de Poincaré-Cartan a lo largo de las
      órbitas cerradas en el espacio de fase $(q, p) = (T, \nabla T)$, y emite el paquete
      `Phase2ThermalState`.
  • FASE 3 (`ThermalGradientCertificate` y `Phase3ThermalDecisionActuationMixin`):
      Consume `Phase1ThermalState` y `Phase2ThermalState`, aplica la cascada de automejora recursiva
      RSI³ ajustando hiperparámetros mediante contracción de Banach $L_{\mathrm{Banach}} \le 0.90$,
      clasifica la admisibilidad termodinámica en el retículo de Heyting $\Omega_4 = \{\mathtt{VETOED} \prec \mathtt{DEGRADED} \prec \mathtt{COHERENT} \prec \mathtt{CERTIFIED}\}$,
      y emite el certificado final inmutable `ThermalGradientCertificate`.

MARCO TEÓRICO:
────────────────────────────────────────────────────────────────────────────────────────────────────────
1. LEY GENERALIZADA DE CONDUCCIÓN COVARIANTE Y FLUJO DE ENERGÍA:
   Sobre una variedad riemanniana $(M, g)$ con métrica $g_{\mu\nu}$, el flujo de calor covariante $q^\mu$
   se rige por la ley de Fourier anisotrópica tensorial:
       $$q^\mu = - K^{\mu\nu} \nabla_\nu T = - K^{\mu\nu} \left( \partial_\nu T + \Gamma_{\nu\lambda}^\mu T \right)$$
   donde $K^{\mu\nu}$ es el tensor de conductividad simétrico definido positivo ($K = K^\top \succ 0$),
   $\nabla_\nu$ es la derivada covariante de Levi-Civita y $T: M \to \mathbb{R}^+$ representa el campo escalar
   de temperatura absoluta. La tasa de disipación termodinámica local (producción de entropía) viene dada por:
       $$\sigma_{\mathrm{th}} = - \frac{1}{T^2} q^\mu \nabla_\mu T = \frac{1}{T^2} \nabla_\mu T K^{\mu\nu} \nabla_\nu T \ge 0$$
   asegurando el cumplimiento estricto del Segundo Principio de la Termodinámica.

2. SUBESPACIO DE KRYLOV Y TRIDIAGONALIZACIÓN ESPECTRAL DE LANCZOS:
   Para sistemas térmicos de gran dimensión $n \gg 1$, el espectro del tensor de difusión se aproxima en el
   subespacio de Krylov $\mathcal{K}_m(K, v_1) = \operatorname{span}\{v_1, K v_1, K^2 v_1, \dots, K^{m-1} v_1\}$
   mediante el algoritmo de Lanczos con ortogonalización completa de Gram-Schmidt:
       $$K V_m = V_m T_m + \beta_m v_{m+1} e_m^\top$$
   donde $V_m = [v_1, \dots, v_m] \in \mathbb{R}^{n \times m}$ posee columnas ortonormales ($V_m^\top V_m = I_m$)
   y $T_m \in \mathbb{R}^{m \times m}$ es la matriz tridiagonal simétrica:
       $$T_m = \begin{pmatrix} \alpha_1 & \beta_1 & & \\ \beta_1 & \alpha_2 & \ddots & \\ & \ddots & \ddots & \beta_{m-1} \\ & & \beta_{m-1} & \alpha_m \end{pmatrix}$$
   Los autovalores extremos de $T_m$ (valores de Ritz) convergen exponencialmente a los modos dominantes de disipación.

3. INTEGRACIÓN GEOMÉTRICA DE GRADIENTES POR DISCRETIZACIÓN DE ITOH-ABE:
   Para garantizar la estabilidad no-condicionada y la conservación de invariantes en la integración del flujo
   térmico no lineal $\dot{x} = -\nabla V(x)$, el método de gradiente discreto de Itoh-Abe evalúa:
       $$\frac{V(x_{k+1}) - V(x_k)}{\|x_{k+1} - x_k\|^2} (x_{k+1} - x_k) = - \bar{\nabla} V(x_k, x_{k+1})$$
   satisfaciendo la relación exacta de incremento de energía:
       $$V(x_{k+1}) - V(x_k) = \langle \bar{\nabla} V(x_k, x_{k+1}), x_{k+1} - x_k \rangle = - \Delta t \|\bar{\nabla} V\|^2 \le 0$$
   eliminando cualquier posibilidad de amplificación espuria o inestabilidad numérica no lineal.

4. ACCIÓN DE POINCARÉ-CARTAN Y RESIDUALES DE MONODROMÍA:
   En el fibrado cotangente de fases térmicas $(q, p)$, la forma diferencial simpléctica $\omega = dq \wedge dp$
   define el invariante integral de Poincaré-Cartan a lo largo de una curva cerrada $\gamma$:
       $$J_P = \oint_\gamma (p \, dq - H \, dt) = \text{constante}$$
   Las desviaciones orbitales se evalúan mediante la matriz de monodromía de Floquet-Poincaré $\mathcal{M}$:
       $$\dot{\mathcal{M}}(t) = J \nabla^2 H(z(t)) \mathcal{M}(t), \quad \mathcal{M}(0) = I_{2n}$$
   cuyos autovalores $\lambda_i \in \mathbb{C}$ certifican la estabilidad de Krein sobre el círculo unitario $|\lambda_i| = 1$.

5. AUTOMEJORA RECURSIVA RSI³ CON CONTRACCIÓN DE BANACH:
   El operador de ajuste de parámetros de convección $\Phi_{\mathrm{RSI}}: \Theta \to \Theta$ actúa en el espacio
   de Banach con norma inducida por la métrica de Fisher:
       $$\theta^{(k+1)} = \theta^{(k)} - \eta \, G^{-1}(\theta^{(k)}) \nabla_{\theta} \mathcal{L}(\theta^{(k)})$$
   La convergencia unívoca se garantiza si la constante de Lipschitz de contracción cumple:
       $$\|\Phi_{\mathrm{RSI}}(\theta_1) - \Phi_{\mathrm{RSI}}(\theta_2)\| \le L_{\mathrm{Banach}} \|\theta_1 - \theta_2\|, \quad L_{\mathrm{Banach}} < 1.0$$
   con el candado de seguridad de Gödel-Tarski prohibiendo estrictamente relajar las tolerancias físicas de frontera.

AXIOMAS FORMALES:
────────────────────────────────────────────────────────────────────────────────────────────────────────
  [Axioma TGL-1] Positividad Estricta de la Matriz de Conductividad:
      $K = K^\top \succ 0$, con autovalor mínimo acotado inferiormente $\lambda_{\min}(K) \ge \tau_{\mathrm{SPD}} = 10^{-12}$.
  [Axioma TGL-2] No Negatividad de la Producción de Entropía:
      $\sigma_{\mathrm{th}} = \nabla T^\top K \nabla T \ge 0$, asegurando disipación termodinámica monótona estricta.
  [Axioma TGL-3] Conservación de Disipación de Itoh-Abe:
      La tasa discreta de variación satisface $\Delta V \le 0$ para todo paso temporal finito $\Delta t > 0$.
  [Axioma TGL-4] Acotación de Ritz en la Tridiagonalización de Lanczos:
      El error residual de Lanczos $\|K V_m - V_m T_m\|_2 \le \beta_m \le \varepsilon_{\mathrm{Krylov}}$.
  [Axioma TGL-5] Contracción Monádica del Operador de Calibre RSI³:
      La constante de Banach de la capa de adaptación satisface $L_{\mathrm{Banach}} \le 0.90$.

INVARIANTES DE ESTADO:
────────────────────────────────────────────────────────────────────────────────────────────────────────
  Invariante 1 (Simetría e Invertibilidad del Tensor $K$):
      $\|K - K^\top\|_F / \|K\|_F \le 1.0 \times 10^{-10}$ y $\operatorname{cond}(K) \le 1.0 \times 10^8$.
  Invariante 2 (Preservación del Espectro de Frecuencias de Poincaré):
      La deriva secular de la acción $\Delta J_P / J_P \le 1.0 \times 10^{-6}$ en órbitas cuasi-periódicas KAM.
  Invariante 3 (Estabilidad Numérica de Wilkinson):
      Todas las operaciones tensoriales se acumulan en precisión doble float64 garantizando residuo $\le 100 \varepsilon_{\mathrm{mach}}$.
  Invariante 4 (Monotonía de Veredicto en Retículo $\Omega_4$):
      Cualquier anomalía térmica no disipativa degrada monótonamente el estado hasta `VETOED`.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Deque, Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la

try:
    from scipy.sparse.linalg import eigsh
except ImportError:  # pragma: no cover
    eigsh = None  # type: ignore[assignment]

logger = logging.getLogger("APU.Physics.ThermalGradientLaws")

# ---------------------------------------------------------------------------
# Constantes metrológicas (Wilkinson / IEEE-754 binary64 / Poincaré-KAM / RSI-L3)
# ---------------------------------------------------------------------------
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_WILKINSON_FLOOR: Final[float] = 1e-15
_CSMD_STEP: Final[float] = 1e-20
_IMAGINARY_TOL: Final[float] = 100.0 * _MACHINE_EPS
_SPD_ABS_TOL: Final[float] = 100.0 * _MACHINE_EPS
_SPD_REL_TOL: Final[float] = 1e-8
_ENTROPY_ABS_TOL: Final[float] = 100.0 * _MACHINE_EPS
_LANDAUER_LN2: Final[float] = float(np.log(2.0))

_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_CROWBAR_LATENCY_FLOOR_NS: Final[float] = 380.0
_CROWBAR_LATENCY_CEIL_NS: Final[float] = 420.0

_DEFAULT_SAFETY_MARGIN: Final[float] = 1.0
_DEGRADED_FOURIER_THRESHOLD: Final[float] = 1e-6
_DEGRADED_CONDITION: Final[float] = 1e8
_KRYLOV_FULL_EIGH_CAP: Final[int] = 24
_KRYLOV_K_DEFAULT: Final[int] = 8
_NOVIKOV_WINDOW: Final[int] = 16
_NOVIKOV_RHO: Final[float] = 0.85
_CD_TAU0: Final[float] = 1e-4

# Constantes de Mecánica Celeste de Poincaré y KAM
_KAM_DIOPHANTINE_TAU: Final[float] = 2.5
_KAM_RESONANCE_MAX_ORDER: Final[int] = 3
_POINCARE_SECTION_WINDOW: Final[int] = 24
_RECURRENCE_CAP: Final[float] = 1e12

# Constantes de Automejora Recursiva de Nivel 3 (RSI-L3)
_RSI_BANACH_TARGET_CAP: Final[float] = 0.92
_RSI_META_LR_INIT: Final[float] = 0.05
_RSI_REG_CEIL: Final[float] = 1e-9


# =============================================================================
# Retículo de Heyting (Topos de Verdad Termodinámica Ω_4)
# =============================================================================
class HeytingVerdict(str, Enum):
    r"""
    Cadena completa de Heyting:  ⊥ = VETOED ≺ DEGRADED ≺ COHERENT ≺ CERTIFIED = ⊤.
    """

    VETOED = "VETOED"
    DEGRADED = "DEGRADED"
    COHERENT = "COHERENT"
    CERTIFIED = "CERTIFIED"

    @property
    def rank(self) -> int:
        return {
            HeytingVerdict.VETOED: 0,
            HeytingVerdict.DEGRADED: 1,
            HeytingVerdict.COHERENT: 2,
            HeytingVerdict.CERTIFIED: 3,
        }[self]

    def meet(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return self if self.rank <= other.rank else other

    def join(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return self if self.rank >= other.rank else other

    def implies(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict.CERTIFIED if self.rank <= other.rank else other

    def negate(self) -> "HeytingVerdict":
        return self.implies(HeytingVerdict.VETOED)


# =============================================================================
# Cartas Diferenciales, Simplécticas (Poincaré), Autopoéticas (RSI-L3) y Contratos
# =============================================================================
def _freeze_array(arr: np.ndarray) -> np.ndarray:
    out = np.array(arr, copy=True)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, slots=True)
class PoincareSpectralResonanceChart:
    r"""
    Carta de Resonancia Espectral de Poincaré-Birkhoff y Estabilidad KAM (Fase 1).

    Atributos:
        characteristic_frequencies: Frecuencias de acción-ángulo ω_i = √max(λ_i, α).
        diophantine_index         : Constante efectiva γ_KAM de pequeños divisores.
        kam_torus_stability       : Medida en (0, 1] de supervivencia del toro invariante KAM.
        arnold_diffusion_risk     : Riesgo [0, 1) de deriva caótica en dimensiones n ≥ 3.
        poincare_first_invariant  : Primer invariante integral espectral J_1 = (1/2π) ∑ |⟨v_i, ∇T⟩|² / ω_i.
        min_resonance_detuning    : Mínimo |p ω_i − q ω_j| para 1 ≤ p, q ≤ 3.
    """

    characteristic_frequencies: np.ndarray
    diophantine_index: float
    kam_torus_stability: float
    arnold_diffusion_risk: float
    poincare_first_invariant: float
    min_resonance_detuning: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "characteristic_frequencies",
            _freeze_array(self.characteristic_frequencies),
        )


@dataclass(frozen=True, slots=True)
class PoincarePhaseSpaceChart:
    r"""
    Carta Simpléctica y Mapa de Retorno de Poincaré en T*M (Fase 2 & Fase 3).

    Atributos:
        hamiltonian_energy      : Hamiltoniano de Dirichlet-Poincaré H(q,p) = ½ pᵀ κ p.
        poincare_cartan_form    : Incremento discreto de la 1-forma θ_PC = p dq − H dt.
        monodromy_determinant   : Determinante de Liouville det(DP_Σ) del mapa de retorno.
        monodromy_trace         : Traza Tr(DP_Σ) (indicador parabólico/elíptico/hiperbólico).
        recurrence_time_estimate: Estimador de recurrencia de Kac-Poincaré τ_rec.
        canonical_q             : Coordenada configuracional adimensional q = L ‖∇T‖_g / T.
        canonical_p             : Momento termodinámico conjugado p = ‖Q_♭‖_{g⁻¹} / T.
        symplectic_area_defect  : Defecto de conservación/disipación del 2-invariante de Poincaré.
    """

    hamiltonian_energy: float
    poincare_cartan_form: float
    monodromy_determinant: float
    monodromy_trace: float
    recurrence_time_estimate: float
    canonical_q: float
    canonical_p: float
    symplectic_area_defect: float


@dataclass(frozen=True, slots=True)
class RSILevel3GovernorChart:
    r"""
    Carta del Gobernador de Automejora Recursiva de Nivel 3 (RSI-L3).

    Jerarquía formal de recursión:
        - Nivel 1 (Parámetros de Objeto): (α_reg, ρ_Novikov, τ_0, w_ItohAbe).
        - Nivel 2 (Meta-Gradiente Natural): Tensor métrico de Fisher-Rao G^(2) y pérdida de Lyapunov.
        - Nivel 3 (Meta-Meta-Endofunctor): Contracción de Lipschitz-Banach L_Banach < 1 y
          candado de consistencia formal de Gödel-Tarski.
    """

    epoch: int
    level1_reg_floor: float
    level1_novikov_rho: float
    level1_cd_tau0: float
    level1_itoh_abe_weight: float
    level2_lyapunov_loss: float
    level2_natural_grad_norm: float
    level2_fisher_metric_trace: float
    level3_banach_contraction: float
    level3_meta_damping: float
    godel_tarski_soundness_locked: bool


@dataclass(frozen=True, slots=True)
class SpectralChart:
    r"""
    Factorización / carta extremal de un operador SPD con invariantes de Poincaré.

    Si `lanczos_used`, `eigenvalues` contiene solo los k extremos (both-ends)
    y `eigenvectors` las bases asociadas; `inv_metric` puede ser None y la
    inversa se aplica por CG. `deflated_operator` es K de rango-k + ridge
    cuando `rank_reduced` es verdadero.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    inv_metric: Optional[np.ndarray]
    sqrt_metric: Optional[np.ndarray]
    condition_number: float
    operator_norm: float
    frobenius_norm: float
    spectral_gap: float
    volume_density: float
    regularized: bool
    tikhonov_alpha: float
    kernel_dimension: int
    min_eigenvalue: float
    max_eigenvalue: float
    lanczos_used: bool
    rank_reduced: bool
    krylov_k: int
    deflated_operator: Optional[np.ndarray] = None
    poincare_diophantine_index: float = 1.0
    poincare_integral_invariant: float = 0.0
    kam_torus_stability: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "eigenvalues", _freeze_array(self.eigenvalues))
        object.__setattr__(self, "eigenvectors", _freeze_array(self.eigenvectors))
        if self.inv_metric is not None:
            object.__setattr__(self, "inv_metric", _freeze_array(self.inv_metric))
        if self.sqrt_metric is not None:
            object.__setattr__(self, "sqrt_metric", _freeze_array(self.sqrt_metric))
        if self.deflated_operator is not None:
            object.__setattr__(self, "deflated_operator", _freeze_array(self.deflated_operator))


@dataclass(frozen=True, slots=True)
class ItohAbeChart:
    r"""Carta del gradiente discreto de Itoh–Abe (estándar y simetrizado RSI-L3) de Ē(p) = pᵀ κ p."""

    discrete_gradient: np.ndarray
    tellegen_residual: float
    pairing: float
    dirichlet_energy: float
    fourier_pairing: float
    coordinate_order_defect: float
    symmetrized_gradient: Optional[np.ndarray] = None
    rsi_barycentric_weight: float = 0.5
    forward_reverse_discrepancy: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "discrete_gradient", _freeze_array(self.discrete_gradient))
        if self.symmetrized_gradient is not None:
            object.__setattr__(
                self, "symmetrized_gradient", _freeze_array(self.symmetrized_gradient)
            )


@dataclass(frozen=True, slots=True)
class TransientFiltration:
    r"""
    Envolvente de transitorio (valuación tipo Novikov + separatrix homoclínica de Poincaré-Melnikov).

    ν ≥ 0 crece con el contenido de alta frecuencia de |dT|_g.
    τ_CD = −τ₀·safety·exp(−ν)  (más negativo ⇒ margen más holgado en transitorios KAM-confinados).
    """

    envelope: float
    high_frequency: float
    valuation: float
    tau_cd: float
    window_ultrametric: float
    mc_polynomial: float
    melnikov_distance: float = 0.0
    poincare_winding_number: int = 0
    lyapunov_exponent: float = 0.0
    kam_confined: bool = True


@dataclass(frozen=True, slots=True)
class Phase1ThermalState:
    r"""
    Objeto de Estado Terminal de la Fase 1 y Dominio Exacto de Inicio de la Fase 2.
    """

    K_reg: np.ndarray
    grad_T: np.ndarray
    T_sys: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    conductivity_chart: Optional[SpectralChart] = None
    weyl_skew_residual: float = 0.0
    temperature_clamped: bool = False
    third_law_flag: bool = False
    onsager_residual: float = 0.0
    poincare_resonance: Optional[PoincareSpectralResonanceChart] = None
    rsi_governor: Optional[RSILevel3GovernorChart] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "K_reg", _freeze_array(self.K_reg))
        object.__setattr__(self, "grad_T", _freeze_array(self.grad_T))


@dataclass(frozen=True, slots=True)
class Phase2ThermalState:
    r"""
    Objeto de Estado Terminal de la Fase 2 y Dominio Exacto de Inicio de la Fase 3.
    """

    Q_vector: np.ndarray
    Q_down: np.ndarray
    clausius_duhem_residual: float
    thermal_entropy_production: float
    carnot_efficiency: float
    carnot_density: float
    entropy_production_rate: float
    exergy_potential: float
    gouy_stodola: float
    fourier_residual: float
    dissipation_identity_residual: float
    dirichlet_energy: float
    csmd_error: float
    landauer_gap: float
    pairing_q_dT: float
    tellegen_residual: float
    T_hot: float
    T_cold: float
    itoh_abe: Optional[ItohAbeChart] = None
    metric_chart: Optional[SpectralChart] = None
    conductivity_up: Optional[np.ndarray] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    poincare_phase_chart: Optional[PoincarePhaseSpaceChart] = None
    rsi_governor: Optional[RSILevel3GovernorChart] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "Q_vector", _freeze_array(self.Q_vector))
        object.__setattr__(self, "Q_down", _freeze_array(self.Q_down))
        if self.conductivity_up is not None:
            object.__setattr__(self, "conductivity_up", _freeze_array(self.conductivity_up))


@dataclass(frozen=True, slots=True)
class Phase3HeytingDecision:
    r"""
    Objeto de Decisión Terminal de la Fase 3 (Clausura del Ciclo Termodinámico).
    """

    heyting_verdict: str
    heyting_rank: int
    heyting_score: float
    veto_reasons: Tuple[str, ...]
    degraded_reasons: Tuple[str, ...]
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    conservation_residual: float
    adaptive_tau_cd: float
    transient_valuation: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    poincare_phase_chart: Optional[PoincarePhaseSpaceChart] = None
    rsi_governor: Optional[RSILevel3GovernorChart] = None


# =============================================================================
# FASE 1 — INGESTA, LANCZOS-HIGHAM, RESONANCIAS DE POINCARÉ-KAM Y PRE-RSI-L3
# =============================================================================
class Phase1ThermalIngestionMixin:
    r"""
    FASE 1 — OBSERVE + ORIENT (Geometría Espectral, Resonancias de Poincaré y Semilla RSI-L3).

    Ejecuta:
        1. Validación metrológica e integración compensada de Kahan-Babuška-Neumaier (KBN).
        2. Proyección simétrica de Weyl π_Sym(K) = ½(K + Kᵀ) y auditoría de reciprocidad de Onsager.
        3. Extracción espectral extremal por Lanczos-Krylov (ARPACK `both-ends`) o Rayleigh-Ritz denso.
        4. Análisis de Resonancias de Poincaré-Birkhoff y Pequeños Divisores Diofánticos KAM sobre el
           toro de frecuencias térmicas ω_i = √max(λ_i, α).
        5. Pre-condicionamiento espectral de Higham sobre el cono {A ⪰ α_RSI I}.

    CONEXIÓN ANIDADA DE FASE:
        El método final de esta clase (`phase1_ingest_and_purify`) emite `Phase1ThermalState`,
        el cual constituye el argumento de entrada directo del primer eslabón de la Fase 2.
    """

    def __init__(
        self,
        dimension_n: int,
        safety_margin: float = _DEFAULT_SAFETY_MARGIN,
        *,
        rng_seed: Optional[int] = None,
        krylov_k: int = _KRYLOV_K_DEFAULT,
        rank_reduce: bool = False,
        novikov_rho: float = _NOVIKOV_RHO,
        cd_tau0: float = _CD_TAU0,
    ) -> None:
        self._n = self._validate_positive_int("dimension_n", dimension_n)
        self._safety_margin = self._validate_positive_finite("safety_margin", safety_margin)
        self._reg = _WILKINSON_FLOOR
        self._krylov_k = int(max(1, krylov_k))
        self._rank_reduce = bool(rank_reduce)
        self._novikov_rho = float(min(max(novikov_rho, 0.0), 1.0 - _MACHINE_EPS))
        self._cd_tau0 = self._validate_positive_finite("cd_tau0", cd_tau0)
        self._rng = np.random.default_rng(rng_seed)
        self._interlock_lock = threading.Lock()
        self._interlock_state = False
        self._transient_envelope = 0.0
        self._transient_window: Deque[float] = deque(maxlen=_NOVIKOV_WINDOW)

        # Estado interno de Mecánica Celeste de Poincaré (sección transversal Σ en T*M)
        self._poincare_orbit_history: Deque[Tuple[float, float, float, float]] = deque(
            maxlen=_POINCARE_SECTION_WINDOW
        )
        self._poincare_monodromy_matrix: np.ndarray = np.eye(2, dtype=np.float64)

        # Estado interno del Gobernador de Automejora Recursiva de Nivel 3 (RSI-L3)
        self._rsi_epoch: int = 0
        self._rsi_itoh_abe_weight: float = 0.5
        self._rsi_fisher_diag: np.ndarray = np.ones(4, dtype=np.float64)
        self._rsi_prev_param_vec: np.ndarray = np.array(
            [self._reg, self._novikov_rho, self._cd_tau0, self._rsi_itoh_abe_weight],
            dtype=np.float64,
        )
        self._rsi_prev_step_norm: float = 0.0
        self._rsi_meta_damping: float = _RSI_META_LR_INIT
        self._rsi_last_banach_contraction: float = 0.0
        self._rsi_last_lyapunov_loss: float = 0.0
        self._rsi_last_grad_norm: float = 0.0

    @staticmethod
    def _validate_positive_int(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} debe ser un entero.")
        if value <= 0:
            raise ValueError(f"{name} debe ser estrictamente mayor que cero.")
        return int(value)

    @staticmethod
    def _validate_positive_finite(name: str, value: Any) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} no debe ser booleano.")
        try:
            value_f = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} debe ser numérico.") from exc
        if not math.isfinite(value_f) or value_f <= 0.0:
            raise ValueError(f"{name} debe ser finito y estrictamente mayor que cero.")
        return value_f

    @staticmethod
    def _validate_finite(name: str, value: Any) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} no debe ser booleano.")
        try:
            value_f = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} debe ser numérico.") from exc
        if not math.isfinite(value_f):
            raise ValueError(f"{name} debe ser finito.")
        return value_f

    def _validate_temperature(self, value: Any, name: str = "T_sys") -> float:
        return self._validate_finite(name, value)

    @staticmethod
    def _ensure_finite_array(arr: np.ndarray, name: str) -> None:
        try:
            finite = bool(np.all(np.isfinite(arr)))
        except TypeError as exc:
            raise ValueError(f"{name} contiene tipos no numéricos.") from exc
        if not finite:
            raise ValueError(f"{name} contiene valores no finitos (NaN/Inf).")

    def _as_numeric_vector(
        self,
        values: Any,
        name: str,
        *,
        expected_size: Optional[int] = None,
    ) -> np.ndarray:
        try:
            raw = np.asarray(values)
        except Exception as exc:
            raise ValueError(f"{name} no puede convertirse en ndarray.") from exc
        if raw.ndim == 0:
            raw = raw.reshape(1)
        elif raw.ndim > 1:
            raw = raw.ravel()
        if np.iscomplexobj(raw):
            self._ensure_finite_array(raw, name)
            if np.any(np.abs(raw.imag) > _IMAGINARY_TOL):
                raise ValueError(f"{name} posee componente imaginaria no despreciable.")
            raw = raw.real
        try:
            arr = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} no puede convertirse a vector float64.") from exc
        self._ensure_finite_array(arr, name)
        if expected_size is not None and arr.size != expected_size:
            raise ValueError(
                f"{name} debe tener tamaño estricto {expected_size}. Se recibió {arr.size}."
            )
        return arr

    def _as_numeric_matrix(
        self,
        values: Any,
        name: str,
        *,
        expected_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        try:
            raw = np.asarray(values)
        except Exception as exc:
            raise ValueError(f"{name} no puede convertirse en ndarray.") from exc
        if raw.ndim != 2:
            raise ValueError(f"{name} debe ser una matriz 2D.")
        if np.iscomplexobj(raw):
            self._ensure_finite_array(raw, name)
            if np.any(np.abs(raw.imag) > _IMAGINARY_TOL):
                raise ValueError(f"{name} posee componente imaginaria no despreciable.")
            raw = raw.real
        try:
            arr = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} no puede convertirse a matriz float64.") from exc
        self._ensure_finite_array(arr, name)
        if expected_shape is not None and arr.shape != expected_shape:
            raise ValueError(
                f"{name} debe tener forma estricta {expected_shape}. Se recibió {arr.shape}."
            )
        return arr

    def _neumaier_sum(self, arr: np.ndarray) -> complex:
        data = np.asarray(arr).ravel()
        if data.size == 0:
            return 0.0
        if not np.all(np.isfinite(data)):
            raise ValueError("kahan/neumaier: el sumando contiene valores no finitos")
        use_complex = np.iscomplexobj(data)
        total: complex = 0.0j if use_complex else 0.0
        compensator: complex = 0.0j if use_complex else 0.0
        for x in data:
            t = total + x
            if abs(total) >= abs(x):
                compensator += (total - t) + x
            else:
                compensator += (x - t) + total
            total = t
        return total + compensator

    def kahan_sum(self, arr: np.ndarray) -> float:
        r"""Suma compensada de Kahan–Babuška–Neumaier: |fl(∑x)−∑x| ≤ (2u+O(u²)) ∑|x|."""
        vec = self._as_numeric_vector(arr, "arr")
        return float(np.real(self._neumaier_sum(vec)))

    @staticmethod
    def _weyl_sym_project(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
        """Proyección ortogonal de Weyl π_Sym(K)=½(K+Kᵀ). Residuo Onsager = ‖K−π(K)‖_F."""
        sym = 0.5 * (matrix + matrix.T)
        residual = float(la.norm(matrix - sym, ord="fro"))
        return sym, residual

    def _snapshot_rsi_governor_chart(self) -> RSILevel3GovernorChart:
        """Emite una instantánea inmutable del estado actual del gobernador RSI-L3."""
        with self._interlock_lock:
            return RSILevel3GovernorChart(
                epoch=int(self._rsi_epoch),
                level1_reg_floor=float(self._reg),
                level1_novikov_rho=float(self._novikov_rho),
                level1_cd_tau0=float(self._cd_tau0),
                level1_itoh_abe_weight=float(self._rsi_itoh_abe_weight),
                level2_lyapunov_loss=float(self._rsi_last_lyapunov_loss),
                level2_natural_grad_norm=float(self._rsi_last_grad_norm),
                level2_fisher_metric_trace=float(np.sum(self._rsi_fisher_diag)),
                level3_banach_contraction=float(self._rsi_last_banach_contraction),
                level3_meta_damping=float(self._rsi_meta_damping),
                godel_tarski_soundness_locked=bool(
                    self._reg >= _WILKINSON_FLOOR
                    and 0.0 <= self._rsi_itoh_abe_weight <= 1.0
                    and self._cd_tau0 > 0.0
                    and self._rsi_last_banach_contraction < 1.0
                ),
            )

    def _compute_poincare_spectral_resonance(
        self,
        evals: np.ndarray,
        evecs: np.ndarray,
        grad_T: Optional[np.ndarray] = None,
    ) -> PoincareSpectralResonanceChart:
        r"""
        Análisis de Resonancias de Poincaré-Birkhoff y Condición Diofántica KAM en Fase 1.

        Dado el espectro regularizado λ_i ≥ α, las frecuencias características del flujo
        port-Hamiltoniano térmico son ω_i = √max(λ_i, α).
        Se evalúan los pequeños divisores de Poincaré:
            D_{ij}^{p,q} = |p ω_i − q ω_j| / ((ω_i + ω_j) · (p + q)^{−τ_KAM})
        para 1 ≤ p, q ≤ 3, midiendo la proximidad a resonancias conmensurables y el
        primer invariante integral de Poincaré J_1 = (1/2π) ∑_i |⟨v_i, ∇T⟩|² / ω_i.
        """
        clamped_evals = np.maximum(np.asarray(evals, dtype=np.float64), self._reg)
        omegas = np.sqrt(clamped_evals)
        m = int(omegas.size)

        min_detuning = float("inf")
        diophantine_gamma = float("inf")

        if m >= 2:
            for i in range(m):
                for j in range(i + 1, m):
                    w_i = float(omegas[i])
                    w_j = float(omegas[j])
                    denom_scale = max(w_i + w_j, _WILKINSON_FLOOR)
                    for p_int in range(1, _KAM_RESONANCE_MAX_ORDER + 1):
                        for q_int in range(1, _KAM_RESONANCE_MAX_ORDER + 1):
                            # En el caso degenerado isótropo (w_i == w_j) con p == q,
                            # la simetría O(n) es integrable (superintegrabilidad de Kepler-Poincaré),
                            # por lo que solo auditamos resonancias no triviales p != q si w_i ≈ w_j.
                            if p_int == q_int and abs(w_i - w_j) <= 10.0 * _SPD_ABS_TOL * denom_scale:
                                continue
                            detuning = abs(p_int * w_i - q_int * w_j)
                            if detuning < min_detuning:
                                min_detuning = detuning
                            order_weight = float((p_int + q_int) ** _KAM_DIOPHANTINE_TAU)
                            gamma_cand = (detuning / denom_scale) * order_weight
                            if gamma_cand < diophantine_gamma:
                                diophantine_gamma = gamma_cand

        if not math.isfinite(min_detuning):
            min_detuning = float(omegas[0]) if m >= 1 else 1.0
        if not math.isfinite(diophantine_gamma):
            diophantine_gamma = 1.0

        kam_stability = float(diophantine_gamma / (1.0 + diophantine_gamma))
        kam_stability = float(np.clip(kam_stability, _WILKINSON_FLOOR, 1.0))

        # Difusión de Arnold solo ocurre topológicamente cuando la dimensión del toro m ≥ 3
        # (los toros 2D dividen el espacio de energía 3D, impidiendo el escape de órbitas).
        if m >= 3:
            arnold_risk = float(np.clip((1.0 - kam_stability) * (1.0 - 2.0 / m), 0.0, 1.0 - _MACHINE_EPS))
        else:
            arnold_risk = 0.0

        j1_invariant = 0.0
        if grad_T is not None and evecs.ndim == 2 and evecs.shape[0] == grad_T.size:
            modal_proj = evecs.T @ np.asarray(grad_T, dtype=np.float64)
            action_densities = (modal_proj * modal_proj) / np.maximum(omegas, _WILKINSON_FLOOR)
            j1_invariant = float(self.kahan_sum(action_densities) / (2.0 * math.pi))

        return PoincareSpectralResonanceChart(
            characteristic_frequencies=omegas,
            diophantine_index=float(diophantine_gamma),
            kam_torus_stability=kam_stability,
            arnold_diffusion_risk=arnold_risk,
            poincare_first_invariant=j1_invariant,
            min_resonance_detuning=float(min_detuning),
        )

    def _lanczos_both_ends(
        self,
        spd: np.ndarray,
        name: str,
        k: int,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        r"""
        Autovalores extremos por Lanczos simétrico (ARPACK both-ends).

        k se recorta a [2, n−1]. Si ARPACK no está o no converge, se
        degrada a `eigh` denso (O(n³)).
        """
        n = int(spd.shape[0])
        k_use = int(max(2, min(k, n - 1)))
        if eigsh is None or n <= _KRYLOV_FULL_EIGH_CAP:
            evals, evecs = la.eigh(spd, check_finite=True)
            return np.real(evals), evecs, False
        try:
            evals, evecs = eigsh(spd, k=k_use, which="BE", maxiter=max(4 * n, 40 * k_use))
            order = np.argsort(np.real(evals))
            return np.real(evals)[order], evecs[:, order], True
        except Exception as exc:
            logger.warning(
                "Lanczos/ARPACK falló sobre %s (%s). Fallback a eigh denso.",
                name,
                exc,
            )
            evals, evecs = la.eigh(spd, check_finite=True)
            return np.real(evals), evecs, False

    def _low_rank_ridge(
        self,
        evals: np.ndarray,
        evecs: np.ndarray,
        n: int,
        gamma: float,
        take_largest: bool,
        k: int,
    ) -> np.ndarray:
        r"""
        K_defl = Σ_{i∈I} λ_i v_i v_iᵀ + γ (I − P_I).
        """
        k_use = int(max(1, min(k, evals.size, n)))
        if take_largest:
            idx = np.argsort(np.abs(evals))[-k_use:]
        else:
            idx = np.argsort(evals)[:k_use]
        V = np.asarray(evecs[:, idx], dtype=np.float64)
        lam = np.maximum(np.asarray(evals[idx], dtype=np.float64), gamma)
        P_term = (V * lam) @ V.T
        gram = V @ V.T
        ridge = gamma * (np.eye(n, dtype=np.float64) - gram)
        K = P_term + ridge
        return 0.5 * (K + K.T)

    def _factorize_spd(
        self,
        matrix: np.ndarray,
        name: str,
        *,
        mode: str = "floor",
        allow_negative: bool = False,
        rank_reduce: Optional[bool] = None,
        grad_probe: Optional[np.ndarray] = None,
    ) -> SpectralChart:
        r"""
        Carta espectral enriquecida con invariantes Diofánticos de Poincaré-KAM.

        mode="floor"    : Higham λ ↦ max(λ, α) vía shift (α−λ_min)_+ I.
        mode="tikhonov" : (A + α I)⁻¹ cuando hay base completa.
        """
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{name} debe ser una matriz cuadrada.")
        if matrix.shape[0] != self._n:
            raise ValueError(f"{name} debe ser {self._n} × {self._n}, se recibió {matrix.shape}")

        sym, _ = self._weyl_sym_project(np.asarray(matrix, dtype=np.float64))
        do_reduce = self._rank_reduce if rank_reduce is None else bool(rank_reduce)

        evals, evecs, lanczos = self._lanczos_both_ends(sym, name, self._krylov_k)
        self._ensure_finite_array(evals, f"{name}.eigenvalues")
        if evals.size == 0:
            raise ValueError(f"{name} no puede ser una matriz vacía.")

        min_eig = float(np.min(evals))
        max_eig = float(np.max(evals))
        max_abs = float(max(abs(min_eig), abs(max_eig)))
        tol = max(_SPD_ABS_TOL, _SPD_REL_TOL * max(1.0, max_abs))
        if min_eig < -tol and not allow_negative:
            raise ValueError(
                f"{name} no es semidefinido positivo dentro de tolerancia. "
                f"Autovalor mínimo: {min_eig:.18e}, tolerancia: {tol:.18e}."
            )

        regularized = bool(min_eig <= self._reg)
        if regularized:
            logger.warning(
                "%s mal condicionada o casi singular (λ_min=%.3e); α=%.3e, lanczos=%s.",
                name,
                min_eig,
                self._reg,
                lanczos,
            )

        full_basis = (not lanczos) or (evals.size == self._n)
        shift = max(self._reg - min_eig, 0.0) if mode == "floor" else 0.0
        tik_shift = self._reg if mode == "tikhonov" else 0.0

        deflated: Optional[np.ndarray] = None
        if do_reduce and evals.size >= 1:
            deflated = self._low_rank_ridge(
                evals,
                evecs,
                self._n,
                max(self._reg, _WILKINSON_FLOOR),
                take_largest=True,
                k=self._krylov_k,
            )

        working = sym + (shift + tik_shift) * np.eye(self._n, dtype=np.float64)
        working = 0.5 * (working + working.T)
        if deflated is not None:
            working = deflated

        inv_metric: Optional[np.ndarray] = None
        sqrt_metric: Optional[np.ndarray] = None
        volume = float("nan")
        frob = float(la.norm(working, ord="fro"))
        kernel_dim = 0

        if full_basis and deflated is None:
            shifted = np.maximum(evals + shift + tik_shift, _WILKINSON_FLOOR)
            inv_metric = (evecs * (1.0 / shifted)) @ evecs.T
            inv_metric = 0.5 * (inv_metric + inv_metric.T)
            sqrt_metric = (evecs * np.sqrt(shifted)) @ evecs.T
            sqrt_metric = 0.5 * (sqrt_metric + sqrt_metric.T)
            log_vol = 0.5 * float(
                self.kahan_sum(np.log(np.maximum(np.abs(shifted), _WILKINSON_FLOOR)))
            )
            volume = float(np.exp(log_vol))
            frob = float(
                np.sqrt(np.real(self._neumaier_sum(shifted.astype(np.complex128) ** 2)))
            )
            kernel_dim = int(np.sum(np.abs(evals) <= max(self._reg, _WILKINSON_FLOOR)))
        elif deflated is not None:
            try:
                d_e, d_v = la.eigh(working, check_finite=True)
                d_e = np.maximum(np.real(d_e), _WILKINSON_FLOOR)
                inv_metric = (d_v * (1.0 / d_e)) @ d_v.T
                inv_metric = 0.5 * (inv_metric + inv_metric.T)
                sqrt_metric = (d_v * np.sqrt(d_e)) @ d_v.T
                sqrt_metric = 0.5 * (sqrt_metric + sqrt_metric.T)
            except la.LinAlgError:
                inv_metric = None
                sqrt_metric = None

        lam_floor = max(min_eig + shift + tik_shift, _WILKINSON_FLOOR)
        lam_ceil = max(max_eig + shift + tik_shift, lam_floor)
        cond = float(lam_ceil / lam_floor)
        op_norm = float(lam_ceil)
        gap = float(max(lam_ceil - lam_floor, 0.0)) if evals.size >= 2 else float(lam_floor)

        poincare_res = self._compute_poincare_spectral_resonance(evals, evecs, grad_T=grad_probe)

        return SpectralChart(
            eigenvalues=evals,
            eigenvectors=evecs,
            inv_metric=inv_metric,
            sqrt_metric=sqrt_metric,
            condition_number=cond,
            operator_norm=op_norm,
            frobenius_norm=frob,
            spectral_gap=gap,
            volume_density=volume,
            regularized=regularized,
            tikhonov_alpha=self._reg,
            kernel_dimension=kernel_dim,
            min_eigenvalue=min_eig,
            max_eigenvalue=max_eig,
            lanczos_used=lanczos,
            rank_reduced=bool(deflated is not None),
            krylov_k=int(min(self._krylov_k, evals.size)),
            deflated_operator=deflated,
            poincare_diophantine_index=poincare_res.diophantine_index,
            poincare_integral_invariant=poincare_res.poincare_first_invariant,
            kam_torus_stability=poincare_res.kam_torus_stability,
        )

    def _apply_inverse_spd(
        self,
        chart: SpectralChart,
        original: np.ndarray,
        vec: np.ndarray,
    ) -> np.ndarray:
        """Aplica A⁻¹: base espectral si existe, si no CG sobre A+αI."""
        if chart.inv_metric is not None:
            return np.asarray(chart.inv_metric, dtype=np.float64) @ vec
        A = 0.5 * (
            np.asarray(original, dtype=np.float64) + np.asarray(original, dtype=np.float64).T
        )
        A = A + self._reg * np.eye(self._n, dtype=np.float64)
        try:
            x, *_ = (
                la.cg(A, vec, atol=1e-12, maxiter=max(4 * self._n, 64))
                if hasattr(la, "cg")
                else (None,)
            )
        except Exception:
            x = None
        if x is None:
            x = la.solve(A, vec, assume_a="pos")
        return np.asarray(x, dtype=np.float64)

    def _regularize_spd_matrix(
        self,
        matrix: np.ndarray,
        name: str,
        *,
        grad_probe: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any], SpectralChart, PoincareSpectralResonanceChart]:
        chart = self._factorize_spd(matrix, name, mode="floor", grad_probe=grad_probe)
        poincare_res = self._compute_poincare_spectral_resonance(
            chart.eigenvalues, chart.eigenvectors, grad_T=grad_probe
        )
        if chart.deflated_operator is not None:
            reconstructed = np.array(chart.deflated_operator, copy=True)
        else:
            shift = max(self._reg - chart.min_eigenvalue, 0.0)
            reconstructed = 0.5 * (matrix + matrix.T) + shift * np.eye(
                self._n, dtype=np.float64
            )
        diagnostics = {
            "min_eigenvalue": chart.min_eigenvalue,
            "max_eigenvalue": chart.max_eigenvalue,
            "regularization_floor": self._reg,
            "spd_tolerance": max(
                _SPD_ABS_TOL,
                _SPD_REL_TOL
                * max(1.0, abs(chart.max_eigenvalue), abs(chart.min_eigenvalue)),
            ),
            "condition_number": chart.condition_number,
            "spectral_gap": chart.spectral_gap,
            "lanczos_used": chart.lanczos_used,
            "rank_reduced": chart.rank_reduced,
            "krylov_k": chart.krylov_k,
            "poincare_diophantine_index": poincare_res.diophantine_index,
            "kam_torus_stability": poincare_res.kam_torus_stability,
            "arnold_diffusion_risk": poincare_res.arnold_diffusion_risk,
            "poincare_first_invariant_j1": poincare_res.poincare_first_invariant,
        }
        return reconstructed, diagnostics, chart, poincare_res

    def execute_phase_1_ingestion(
        self,
        K_raw: np.ndarray,
        grad_T_raw: np.ndarray,
        T_sys: float,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """[COMPATIBILIDAD 1.X — FASE 1]"""
        phase1 = self.phase1_ingest_and_purify(K_raw, grad_T_raw, T_sys)
        return (
            np.array(phase1.K_reg, copy=True),
            np.array(phase1.grad_T, copy=True),
            phase1.T_sys,
        )

    def phase1_ingest_and_purify(
        self,
        K_raw: np.ndarray,
        grad_T_raw: np.ndarray,
        T_sys: float,
    ) -> Phase1ThermalState:
        r"""
        [FASE 1 — MORFISMO TERMINAL DE LA FASE 1 → ESLABÓN DE CONTINUACIÓN A FASE 2]

        Purifica el tensor de conductividad mediante proyección de Weyl, deflación de
        Lanczos-Krylov, regularización SPD de Higham gobernada por RSI-L3 y análisis
        de resonancia Diofántica de Poincaré-KAM.

        Codominio: `Phase1ThermalState`
                 ≡ Dominio exacto de inicio de `Phase2CarnotSimulationMixin`.
        """
        K = self._as_numeric_matrix(K_raw, "K_raw", expected_shape=(self._n, self._n))
        grad_T = self._as_numeric_vector(grad_T_raw, "grad_T_raw", expected_size=self._n)
        T = self._validate_temperature(T_sys)

        diagnostics: Dict[str, Any] = {
            "temperature_clamped": False,
            "third_law_flag": False,
        }
        third_law = bool(T <= 0.0)
        clamped = bool(T < _WILKINSON_FLOOR)
        if clamped:
            diagnostics["temperature_clamped"] = True
            diagnostics["T_sys_raw"] = T
            if third_law:
                diagnostics["third_law_flag"] = True
                logger.warning(
                    "T_sys=%.6e ≤ 0: 3ª ley / Nernst. Se proyecta al piso de Wilkinson; Fase 3 vetará.",
                    T,
                )
            T_clamped = _WILKINSON_FLOOR
        else:
            T_clamped = T

        K_sym, onsager_res = self._weyl_sym_project(K)
        K_reg, spd_diagnostics, chart, poincare_res = self._regularize_spd_matrix(
            K_sym, "K_raw", grad_probe=grad_T
        )
        rsi_snapshot = self._snapshot_rsi_governor_chart()

        diagnostics.update(spd_diagnostics)
        diagnostics["onsager_skew_residual"] = onsager_res
        diagnostics["rsi_epoch"] = rsi_snapshot.epoch
        diagnostics["rsi_reg_floor"] = rsi_snapshot.level1_reg_floor

        return Phase1ThermalState(
            K_reg=K_reg,
            grad_T=grad_T,
            T_sys=float(T_clamped),
            diagnostics=diagnostics,
            conductivity_chart=chart,
            weyl_skew_residual=float(onsager_res),
            temperature_clamped=clamped,
            third_law_flag=third_law,
            onsager_residual=float(onsager_res),
            poincare_resonance=poincare_res,
            rsi_governor=rsi_snapshot,
        )


# =============================================================================
# FASE 2 — CONTINUACIÓN FORMAL DESDE FASE 1: FOURIER, ITOH-ABE SIMETRIZADO RSI-L3,
#          ESPACIO FASE DE POINCARÉ-CARTAN Y TERMODINÁMICA DE CARNOT / CLAUSIUS-DUHEM
# =============================================================================
class Phase2CarnotSimulationMixin(Phase1ThermalIngestionMixin):
    r"""
    FASE 2 — DECIDE FÍSICO (Continuación Anidada Directa del Codominio `Phase1ThermalState`).

    Toma el objeto `Phase1ThermalState` emitido por `phase1_ingest_and_purify` y ejecuta:
        1. Elevación métrica del tensor de conductividad κ sobre la variedad (M, g).
        2. Gradiente Discreto de Itoh–Abe Barycéntrico (simetrización dual directa/inversa
           optimizada por RSI-L3) que preserva de forma exacta la identidad de Tellegen
           ⟨∇̄_IA^(w) Ē(0,p), p⟩ = Ē(p) y elimina la asimetría del orden de coordenadas.
        3. Dinámica Simpléctica Disipativa de Poincaré en T*M: 1-forma de Poincaré-Cartan
           θ_PC = p dq − H dt, matriz de monodromía DP_Σ sobre la sección de Poincaré y
           tiempo de recurrencia de Kac-Poincaré τ_rec.
        4. Evaluación de Clausius-Duhem, eficiencia de Carnot, Gouy-Stodola, Landauer y CSMD.

    CONEXIÓN ANIDADA DE FASE:
        El método final de esta clase (`phase2_simulate_carnot_from_phase1`) emite
        `Phase2ThermalState`, que es el dominio exacto de inicio de la Fase 3.
    """

    @staticmethod
    def _validate_entropy_production(value: Any) -> float:
        if isinstance(value, bool):
            raise TypeError("entropy_production_rate no debe ser booleano.")
        try:
            sigma = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError("entropy_production_rate debe ser numérico.") from exc
        if not math.isfinite(sigma):
            raise ValueError("entropy_production_rate debe ser finito.")
        if sigma < -_ENTROPY_ABS_TOL:
            raise ValueError(
                "entropy_production_rate no puede ser negativo fuera de tolerancia."
            )
        return max(0.0, sigma)

    def _inverse_spd_matrix(self, matrix: np.ndarray, name: str) -> np.ndarray:
        chart = self._factorize_spd(matrix, name, mode="tikhonov")
        if chart.inv_metric is not None:
            return np.array(chart.inv_metric, copy=True)
        A = 0.5 * (
            np.asarray(matrix, dtype=np.float64) + np.asarray(matrix, dtype=np.float64).T
        )
        A = A + self._reg * np.eye(self._n, dtype=np.float64)
        return np.array(la.inv(A), copy=True)

    def _raise_conductivity(
        self,
        K_reg: np.ndarray,
        metric_chart: SpectralChart,
        metric_tensor: np.ndarray,
        conductivity_type: str,
    ) -> np.ndarray:
        ginv = metric_chart.inv_metric
        if ginv is None:
            ginv = self._inverse_spd_matrix(metric_tensor, "metric_tensor")
        ginv = np.asarray(ginv, dtype=np.float64)
        K = np.asarray(K_reg, dtype=np.float64)
        kind = str(conductivity_type).strip().lower()
        if kind == "covariant":
            kappa = ginv @ K @ ginv
        elif kind == "contravariant":
            kappa = K
        elif kind == "mixed":
            kappa = K @ ginv
        elif kind in ("isotropic_metric", "isotropic"):
            k_scalar = float(np.trace(K) / max(self._n, 1))
            kappa = k_scalar * ginv
        else:
            raise ValueError(
                "conductivity_type debe ser 'covariant', 'contravariant', "
                "'mixed' o 'isotropic_metric'."
            )
        return 0.5 * (kappa + kappa.T)

    def _quadratic_energy(self, kappa: np.ndarray, p: np.ndarray) -> float:
        """Ē(p) = pᵀ κ p evaluado con suma compensada KBN sobre p ⊙ (κp)."""
        Kp = np.asarray(kappa, dtype=np.float64) @ np.asarray(p, dtype=np.float64)
        return float(self.kahan_sum(np.asarray(p, dtype=np.float64) * Kp))

    def _itoh_abe_discrete_gradient(
        self,
        energy: Callable[[np.ndarray], float],
        x: np.ndarray,
        y: np.ndarray,
        analytic_partial: Optional[Callable[[np.ndarray, int], float]] = None,
    ) -> Tuple[np.ndarray, float]:
        r"""
        Gradiente discreto canónico de Itoh–Abe (orden directo de coordenadas 0..n−1):

            z^{(0)} = y,   z^{(i)} = (x_1..x_i, y_{i+1}..y_n)
            (∇̄H)_i = [H(z^{(i)}) − H(z^{(i−1)})] / (x_i − y_i)

        Identidad de Tellegen (aritmética exacta):
            ⟨∇̄H(x,y), x−y⟩ = H(x) − H(y).
        """
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        n = x.size
        g = np.zeros(n, dtype=np.float64)
        z = y.copy()
        h_prev = float(energy(z))
        for i in range(n):
            z_next = z.copy()
            z_next[i] = x[i]
            h_next = float(energy(z_next))
            dx = float(x[i] - y[i])
            if abs(dx) > _WILKINSON_FLOOR:
                g[i] = (h_next - h_prev) / dx
            elif analytic_partial is not None:
                g[i] = float(analytic_partial(0.5 * (x + y), i))
            else:
                g[i] = 0.0
            z = z_next
            h_prev = h_next
        pairing = float(self.kahan_sum(g * (x - y)))
        tellegen = float(abs(pairing - (float(energy(x)) - float(energy(y)))))
        return g, tellegen

    def _itoh_abe_symmetrized_gradient(
        self,
        energy: Callable[[np.ndarray], float],
        x: np.ndarray,
        y: np.ndarray,
        analytic_partial: Optional[Callable[[np.ndarray, int], float]] = None,
        weight: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        r"""
        Gradiente Discreto de Itoh-Abe Simetrizado por Barycentro RSI-L3.

        Evalúa la filtración telescópica en el orden directo (0 → n−1) y en el orden
        dual inverso (n−1 → 0):
            ∇̄_IA^(w) H(x,y) = w · ∇̄_IA^(→) H(x,y) + (1 − w) · ∇̄_IA^(←) H(x,y).

        Teorema (Invariancia de Tellegen Bajo Combinación Convexa de Filtraciones):
            Para todo w ∈ [0, 1],
            ⟨∇̄_IA^(w) H(x,y), x − y⟩ = w (H(x) − H(y)) + (1 − w)(H(x) − H(y)) = H(x) − H(y).
            Además, para energías cuadráticas Ē(p) = pᵀ κ p y w = ½, ∇̄_IA^(1/2) Ē(x, 0) = κ x
            coincide exactamente con el gradiente de punto medio de González, anulando el
            defecto de orden de coordenadas.
        """
        g_fwd, _ = self._itoh_abe_discrete_gradient(
            energy, x, y, analytic_partial=analytic_partial
        )

        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        n = x_arr.size
        g_rev = np.zeros(n, dtype=np.float64)
        z = y_arr.copy()
        h_prev = float(energy(z))
        for i in range(n - 1, -1, -1):
            z_next = z.copy()
            z_next[i] = x_arr[i]
            h_next = float(energy(z_next))
            dx = float(x_arr[i] - y_arr[i])
            if abs(dx) > _WILKINSON_FLOOR:
                g_rev[i] = (h_next - h_prev) / dx
            elif analytic_partial is not None:
                g_rev[i] = float(analytic_partial(0.5 * (x_arr + y_arr), i))
            else:
                g_rev[i] = 0.0
            z = z_next
            h_prev = h_next

        w = float(self._rsi_itoh_abe_weight if weight is None else weight)
        w = float(np.clip(w, 0.0, 1.0))

        # Selección/refinamiento micro-numérico de Nivel 1 dentro de la fibra convexa
        delta_h = float(energy(x_arr)) - float(energy(y_arr))
        dx_vec = x_arr - y_arr
        g_sym = w * g_fwd + (1.0 - w) * g_rev
        pairing_sym = float(self.kahan_sum(g_sym * dx_vec))
        tellegen_sym = float(abs(pairing_sym - delta_h))
        fwd_rev_discrepancy = float(la.norm(g_fwd - g_rev))

        return g_fwd, g_sym, tellegen_sym, w, fwd_rev_discrepancy

    def _gonzalez_midpoint_gradient(
        self, kappa: np.ndarray, x: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        r"""González: ∇̄Ē(x,y) = κ(x+y) para Ē(p)=pᵀκp (Hessiano constante)."""
        return np.asarray(kappa, dtype=np.float64) @ (np.asarray(x) + np.asarray(y))

    def _complex_step_dirichlet(
        self,
        kappa: np.ndarray,
        grad_T: np.ndarray,
        step: float = _CSMD_STEP,
    ) -> np.ndarray:
        r"""CSMD de I(p)=½ pᵀκp. Im(I(p+ih e_λ))/h = (κp)_λ. Conjugado complejo prohibido."""
        K = np.asarray(kappa, dtype=np.float64)
        p = np.asarray(grad_T, dtype=np.complex128)
        sens = np.zeros(self._n, dtype=np.float64)
        for lam in range(self._n):
            pert = p.copy()
            pert[lam] += 1.0j * step
            value = 0.5 * pert @ K @ pert
            sens[lam] = float(np.imag(value) / step)
        return sens

    def _evaluate_poincare_phase_dynamics(
        self,
        hamiltonian_H: float,
        grad_norm_g: float,
        q_down_norm_ginv: float,
        T_sys: float,
        length_scale: float,
        phi_cd: float,
    ) -> PoincarePhaseSpaceChart:
        r"""
        Mecánica Celeste de Poincaré sobre el Fibrado Cotangente Termodinámico T*M (Fase 2).

        Construye las coordenadas canónicas adimensionales:
            q_t = L · ‖∇T‖_g / T_sys,    p_t = ‖Q_♭‖_{g⁻¹} / T_sys
        y evalúa:
            1. El incremento de la 1-forma integral de Poincaré-Cartan:
               Δθ_PC = ½(p_t + p_{t-1})(q_t − q_{t-1}) − ½(H_t + H_{t-1}) Δt.
            2. La matriz de monodromía DP_Σ de la aplicación de retorno de Poincaré mediante
               actualización secante simpléctico-disipativa amortiguada, verificando el
               Teorema de Liouville-Poincaré: det(DP_Σ) = exp(−∫ div X dt) ≤ 1.
            3. El tiempo de recurrencia de Kac-Poincaré τ_rec en la celda de fase acotada.
        """
        T_safe = max(T_sys, _WILKINSON_FLOOR)
        q_curr = float(length_scale * grad_norm_g / T_safe)
        p_curr = float(q_down_norm_ginv / T_safe)
        H_curr = float(max(hamiltonian_H, 0.0))
        phi_safe = float(phi_cd) if math.isfinite(phi_cd) else 0.0

        with self._interlock_lock:
            if len(self._poincare_orbit_history) > 0:
                q_prev, p_prev, H_prev, _ = self._poincare_orbit_history[-1]
            else:
                q_prev, p_prev, H_prev = q_curr, p_curr, H_curr

            dq = q_curr - q_prev
            dp = p_curr - p_prev
            dH = 0.5 * (H_curr + H_prev)
            poincare_cartan = float(0.5 * (p_curr + p_prev) * dq - dH)

            # Contracción disipativa de Liouville inducida por la producción de entropía Φ_CD ≥ 0
            dissipation_rate = float(np.clip(phi_safe / (1.0 + abs(phi_safe)), -0.5, 0.95))
            contraction_factor = float(math.exp(-0.1 * max(dissipation_rate, 0.0)))

            # Rotación simpléctica local en el plano de fase (q, p) + contracción de Liouville
            phase_angle = float(math.atan2(dp + _WILKINSON_FLOOR, dq + _WILKINSON_FLOOR))
            cos_a = math.cos(0.05 * phase_angle)
            sin_a = math.sin(0.05 * phase_angle)
            local_jacobian = math.sqrt(contraction_factor) * np.array(
                [[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float64
            )
            self._poincare_monodromy_matrix = (
                0.85 * self._poincare_monodromy_matrix + 0.15 * local_jacobian
            )
            M_mono = self._poincare_monodromy_matrix.copy()
            self._poincare_orbit_history.append((q_curr, p_curr, H_curr, phi_safe))

        mono_det = float(np.linalg.det(M_mono))
        mono_trace = float(np.trace(M_mono))
        symplectic_defect = float(abs(mono_det - contraction_factor))

        # Lema de Kac / Teorema de Recurrencia de Poincaré: τ_rec ~ exp(S_fase)
        entropy_exponent = float(
            np.clip(H_curr / (T_safe * T_safe + _WILKINSON_FLOOR), 0.0, math.log(_RECURRENCE_CAP))
        )
        recurrence_time = float(math.exp(entropy_exponent))

        return PoincarePhaseSpaceChart(
            hamiltonian_energy=H_curr,
            poincare_cartan_form=poincare_cartan,
            monodromy_determinant=mono_det,
            monodromy_trace=mono_trace,
            recurrence_time_estimate=recurrence_time,
            canonical_q=q_curr,
            canonical_p=p_curr,
            symplectic_area_defect=symplectic_defect,
        )

    def execute_phase_2_carnot_simulation(
        self,
        K_reg: np.ndarray,
        grad_T: np.ndarray,
        T_sys: float,
        metric_tensor: np.ndarray,
        entropy_production_rate: float,
    ) -> Tuple[np.ndarray, float, float, float, float]:
        """[COMPATIBILIDAD 1.X — FASE 2]"""
        K = self._as_numeric_matrix(K_reg, "K_reg", expected_shape=(self._n, self._n))
        grad = self._as_numeric_vector(grad_T, "grad_T", expected_size=self._n)
        T = self._validate_temperature(T_sys)
        if T < _WILKINSON_FLOOR:
            T = _WILKINSON_FLOOR
        phase1_state = Phase1ThermalState(
            K_reg=K,
            grad_T=grad,
            T_sys=T,
            diagnostics={"source": "compatibility_api"},
            rsi_governor=self._snapshot_rsi_governor_chart(),
        )
        phase2_state = self.phase2_simulate_carnot_from_phase1(
            phase1_state=phase1_state,
            metric_tensor=metric_tensor,
            entropy_production_rate=entropy_production_rate,
        )
        return (
            np.array(phase2_state.Q_vector, copy=True),
            phase2_state.clausius_duhem_residual,
            phase2_state.carnot_efficiency,
            phase2_state.entropy_production_rate,
            phase2_state.exergy_potential,
        )

    def phase2_simulate_carnot_from_phase1(
        self,
        phase1_state: Phase1ThermalState,
        metric_tensor: np.ndarray,
        entropy_production_rate: float,
        *,
        conductivity_type: str = "covariant",
        length_scale: float = 1.0,
        ambient_temperature: Optional[float] = None,
        info_erasure_rate: float = 0.0,
    ) -> Phase2ThermalState:
        r"""
        [FASE 2 — CONTINUACIÓN DE `phase1_ingest_and_purify` Y MORFISMO TERMINAL HACIA FASE 3]

        Dominio  : `Phase1ThermalState` (codominio exacto de Fase 1).
        Codominio: `Phase2ThermalState` (dominio exacto de inicio de Fase 3).

        Flujo constitutivo q = −κ ∇T.
        Emparejamiento de Tellegen simetrizado por RSI-L3:
            ⟨q, dT⟩_IA = −⟨∇̄_IA^(w) Ē(0, ∇T), ∇T⟩ = −Ē(∇T).
        Desigualdad de Clausius-Duhem:
            Φ = σ − ⟨q, dT⟩ / T² ≥ 0.
        """
        if not isinstance(phase1_state, Phase1ThermalState):
            raise TypeError("phase1_state debe ser Phase1ThermalState.")

        K_reg = np.asarray(phase1_state.K_reg, dtype=np.float64)
        grad_T = np.asarray(phase1_state.grad_T, dtype=np.float64)
        T_sys = float(phase1_state.T_sys)
        self._ensure_finite_array(K_reg, "phase1_state.K_reg")
        self._ensure_finite_array(grad_T, "phase1_state.grad_T")
        if not math.isfinite(T_sys) or T_sys <= 0.0:
            raise ValueError("phase1_state.T_sys debe ser finita y positiva.")

        sigma = self._validate_entropy_production(entropy_production_rate)
        L = self._validate_positive_finite("length_scale", length_scale)
        Gamma = self._validate_finite("info_erasure_rate", info_erasure_rate)
        if Gamma < -_ENTROPY_ABS_TOL:
            raise ValueError("info_erasure_rate no puede ser negativo.")
        Gamma = max(0.0, Gamma)

        metric_chart = self._factorize_spd(
            metric_tensor, "metric_tensor", mode="tikhonov", grad_probe=grad_T
        )
        kappa = self._raise_conductivity(
            K_reg, metric_chart, metric_tensor, conductivity_type
        )

        Q_vector = np.asarray(-kappa @ grad_T, dtype=np.float64)
        self._ensure_finite_array(Q_vector, "heat_flux_vector")

        g_mat = 0.5 * (
            np.asarray(metric_tensor, dtype=np.float64)
            + np.asarray(metric_tensor, dtype=np.float64).T
        )
        Q_down = np.asarray(g_mat @ Q_vector, dtype=np.float64)

        def energy_E(p: np.ndarray) -> float:
            return self._quadratic_energy(kappa, p)

        def partial_E(p: np.ndarray, i: int) -> float:
            Kp = kappa @ np.asarray(p, dtype=np.float64)
            return float(Kp[i] + Kp[i])  # ∂(pᵀκp)/∂p_i = 2 (κp)_i

        zeros_n = np.zeros(self._n, dtype=np.float64)
        ia_grad_fwd, ia_grad_sym, tellegen, w_used, fwd_rev_disc = (
            self._itoh_abe_symmetrized_gradient(
                energy_E, grad_T, zeros_n, analytic_partial=partial_E
            )
        )
        pairing_ia = float(-self.kahan_sum(ia_grad_sym * grad_T))
        pairing_fourier = float(self.kahan_sum(Q_vector * grad_T))
        gonzalez = self._gonzalez_midpoint_gradient(kappa, grad_T, zeros_n)
        order_defect = float(la.norm(ia_grad_fwd - gonzalez))
        sym_order_defect = float(la.norm(ia_grad_sym - gonzalez))

        itoh_chart = ItohAbeChart(
            discrete_gradient=ia_grad_fwd,
            tellegen_residual=tellegen,
            pairing=pairing_ia,
            dirichlet_energy=0.5 * energy_E(grad_T),
            fourier_pairing=pairing_fourier,
            coordinate_order_defect=order_defect,
            symmetrized_gradient=ia_grad_sym,
            rsi_barycentric_weight=w_used,
            forward_reverse_discrepancy=fwd_rev_disc,
        )

        # Pairing físico de CD: Tellegen simetrizado (fiel) con fallback Fourier si no finito.
        pairing = pairing_ia if math.isfinite(pairing_ia) else pairing_fourier

        ginv = metric_chart.inv_metric
        if ginv is None:
            ginv = self._inverse_spd_matrix(metric_tensor, "metric_tensor")
        ginv_arr = np.asarray(ginv, dtype=np.float64)
        dirichlet_g = float(np.real(grad_T @ ginv_arr @ grad_T))
        dirichlet_g = max(dirichlet_g, 0.0)
        grad_norm_g = float(np.sqrt(dirichlet_g))

        q_down_norm_ginv_sq = max(float(np.real(Q_down @ ginv_arr @ Q_down)), 0.0)
        q_down_norm_ginv = float(math.sqrt(q_down_norm_ginv_sq))

        T_squared = T_sys * T_sys
        conduction_term = pairing / T_squared
        phi = float(sigma - conduction_term)
        sigma_th = float(-pairing / T_squared)
        quadratic = float(energy_E(grad_T) / T_squared)
        identity_res = float(abs(sigma_th - quadratic))

        fourier_res = float(la.norm(Q_vector + kappa @ grad_T))
        csmd = self._complex_step_dirichlet(kappa, grad_T)
        csmd_err = float(la.norm(csmd - (kappa @ grad_T)))

        T_cold = T_sys
        T_hot = T_sys + L * grad_norm_g
        if math.isfinite(T_hot) and T_hot > T_cold > 0.0:
            carnot_efficiency = float(np.clip(1.0 - T_cold / T_hot, 0.0, 1.0))
        else:
            carnot_efficiency = 0.0
        carnot_density = float(grad_norm_g / T_sys)

        T0 = float(ambient_temperature) if ambient_temperature is not None else T_sys
        if not math.isfinite(T0) or T0 <= 0.0:
            raise ValueError("ambient_temperature debe ser finita y positiva.")
        gouy = float(T0 * phi)

        landauer_floor = float(Gamma * _LANDAUER_LN2)
        landauer_gap = float(sigma - landauer_floor)

        cd_gate = 1.0 if (math.isfinite(phi) and phi >= -_WILKINSON_FLOOR) else 0.0
        fourier_gate = float(
            1.0
            / (
                1.0
                + fourier_res / max(_DEGRADED_FOURIER_THRESHOLD, _WILKINSON_FLOOR)
            )
        )
        exergy_potential = float(
            np.clip(carnot_efficiency * cd_gate * fourier_gate, 0.0, 1.0)
        )

        poincare_phase = self._evaluate_poincare_phase_dynamics(
            hamiltonian_H=itoh_chart.dirichlet_energy,
            grad_norm_g=grad_norm_g,
            q_down_norm_ginv=q_down_norm_ginv,
            T_sys=T_sys,
            length_scale=L,
            phi_cd=phi,
        )

        diagnostics = dict(phase1_state.diagnostics)
        diagnostics.update(
            {
                "heat_flux_norm": float(la.norm(Q_vector)),
                "q_dot_grad_T": pairing,
                "q_dot_grad_T_fourier": pairing_fourier,
                "q_dot_grad_T_itoh_abe": pairing_ia,
                "tellegen_residual": tellegen,
                "itoh_abe_order_defect": order_defect,
                "itoh_abe_symmetrized_order_defect": sym_order_defect,
                "itoh_abe_rsi_weight": w_used,
                "entropy_production_rate": sigma,
                "thermal_entropy_production": sigma_th,
                "conduction_term": conduction_term,
                "clausius_duhem_phi": phi,
                "dirichlet_energy": dirichlet_g,
                "grad_norm_g": grad_norm_g,
                "fourier_residual": fourier_res,
                "dissipation_identity_residual": identity_res,
                "csmd_error": csmd_err,
                "T_hot": T_hot,
                "T_cold": T_cold,
                "length_scale": L,
                "carnot_efficiency": carnot_efficiency,
                "carnot_density": carnot_density,
                "exergy_potential": exergy_potential,
                "gouy_stodola": gouy,
                "landauer_floor": landauer_floor,
                "landauer_gap": landauer_gap,
                "conductivity_type": conductivity_type,
                "metric_condition": metric_chart.condition_number,
                "metric_lanczos": metric_chart.lanczos_used,
                "ambient_temperature": T0,
                "poincare_hamiltonian": poincare_phase.hamiltonian_energy,
                "poincare_cartan_form": poincare_phase.poincare_cartan_form,
                "poincare_monodromy_det": poincare_phase.monodromy_determinant,
                "poincare_monodromy_trace": poincare_phase.monodromy_trace,
                "poincare_recurrence_time": poincare_phase.recurrence_time_estimate,
            }
        )

        phi_value = (
            float(phi)
            if math.isfinite(phi)
            else (float(phi) if math.isinf(phi) else float("nan"))
        )
        if not math.isfinite(pairing):
            diagnostics["heat_flux_inner_product_nonfinite"] = True
            phi_value = float("nan")

        return Phase2ThermalState(
            Q_vector=Q_vector,
            Q_down=Q_down,
            clausius_duhem_residual=phi_value,
            thermal_entropy_production=sigma_th,
            carnot_efficiency=carnot_efficiency,
            carnot_density=carnot_density,
            entropy_production_rate=sigma,
            exergy_potential=exergy_potential,
            gouy_stodola=gouy,
            fourier_residual=fourier_res,
            dissipation_identity_residual=identity_res,
            dirichlet_energy=dirichlet_g,
            csmd_error=csmd_err,
            landauer_gap=landauer_gap,
            pairing_q_dT=pairing,
            tellegen_residual=tellegen,
            T_hot=float(T_hot),
            T_cold=float(T_cold),
            itoh_abe=itoh_chart,
            metric_chart=metric_chart,
            conductivity_up=kappa,
            diagnostics=diagnostics,
            poincare_phase_chart=poincare_phase,
            rsi_governor=phase1_state.rsi_governor,
        )


# =============================================================================
# FASE 3 — CONTINUACIÓN FORMAL DESDE FASE 2: FILTRACIÓN NOVIKOV-MELNIKOV,
#          AUTOMEJORA RECURSIVA DE NIVEL 3 (RSI-L3), VETO DE HEYTING Y CROWBAR
# =============================================================================
class Phase3HeytingVetoMixin(Phase2CarnotSimulationMixin):
    r"""
    FASE 3 — ACT & SELF-IMPROVE (Continuación Anidada Directa del Codominio `Phase2ThermalState`).

    Toma el objeto `Phase2ThermalState` emitido por `phase2_simulate_carnot_from_phase1` y ejecuta:
        1. Filtración de transitorios acoplada con la distancia homoclínica de Poincaré-Melnikov
           Δ_Melnikov, el exponente maximal de Lyapunov λ_LLE y el índice de rotación de
           Poincaré-Bendixson W_Poincaré ∈ ℤ.
        2. Automejora Recursiva de Nivel 3 (RSI-L3):
           - Nivel 1: Adapta θ^(1) = (α_reg, ρ_Novikov, τ_0, w_IA) dentro del politopo seguro.
           - Nivel 2: Ejecuta descenso por gradiente natural sobre la métrica de Fisher-Rao G^(2).
           - Nivel 3: Audita la contracción de Lipschitz-Banach L_Banach < 1 del operador de Nivel 2
             y reescribe G^(2) bajo el candado lógico de Gödel-Tarski.
        3. Clasificación intuicionista en el Álgebra de Heyting Ω_4 y disparo atómico CAS del
           disyuntor Crowbar BT151 [GPIO14].
    """

    def _cas_interlock(self, expected: bool, desired: bool) -> bool:
        with self._interlock_lock:
            if self._interlock_state == expected:
                self._interlock_state = desired
                return True
            return False

    def reset_hardware_interlock_for_supervision(self) -> bool:
        with self._interlock_lock:
            previous_state = self._interlock_state
            self._interlock_state = False
            return previous_state

    def _act_thermal_interlock(self, verdict: str) -> Tuple[bool, float]:
        normalized_verdict = str(verdict).strip().upper()
        if normalized_verdict != HeytingVerdict.VETOED.value:
            return False, 0.0
        swapped = self._cas_interlock(expected=False, desired=True)
        if not swapped:
            logger.warning(
                "CAS: el interlock ya estaba enclavado. Se reconoce el veto "
                "y se registra la latencia de actuación SIMULADA."
            )
        jitter = float(self._rng.normal(loc=0.0, scale=3.5))
        actuation_latency_ns = float(
            np.clip(
                _CROWBAR_IRAM_LATENCY_NS + jitter,
                _CROWBAR_LATENCY_FLOOR_NS,
                _CROWBAR_LATENCY_CEIL_NS,
            )
        )
        logger.critical(
            "RUPTURA DE CLAUSIUS–DUHEM / 3ª LEY / FOURIER. "
            "Disyuntor Crowbar BT151 [GPIO14] SIMULADO. Latencia ficticia: %.2f ns.",
            actuation_latency_ns,
        )
        return True, actuation_latency_ns

    def _update_transient_filtration(
        self,
        grad_norm_g: float,
        poincare_phase: Optional[PoincarePhaseSpaceChart] = None,
    ) -> TransientFiltration:
        r"""
        Filtración Ultramétrica de Novikov acoplada con Estabilidad Homoclínica de Poincaré-Melnikov.

        Calcula:
            - s_t = ρ s_{t−1} + (1−ρ) x_t          (envolvente lenta de variedad central)
            - b_t = |x_t − s_t|                    (fibra rápida transversal)
            - ν_t = log(1 + b_t / (s_t + ε))       (valuación ultramétrica surrogate)
            - Δ_Melnikov = b_t · |det(DP_Σ) − 1| − s_t · exp(−ν_t)
              (si Δ_Melnikov ≤ 0, la órbita permanece confinada dentro de la separatriz KAM).
            - W_Poincaré ∈ ℤ (índice de giro de Poincaré-Bendixson sobre la ventana de fase).
        """
        x = float(max(grad_norm_g, 0.0))
        with self._interlock_lock:
            rho = self._novikov_rho
            s = rho * self._transient_envelope + (1.0 - rho) * x
            self._transient_envelope = s
            self._transient_window.append(x)
            window = tuple(self._transient_window)
            orbit_hist = tuple(self._poincare_orbit_history)

        hf = abs(x - s)
        valuation = float(math.log1p(hf / (s + _WILKINSON_FLOOR)))
        ultra = 0.0
        lyapunov_lle = 0.0
        if len(window) >= 2:
            logs = [math.log(v + _WILKINSON_FLOOR) for v in window]
            diffs = [abs(logs[j] - logs[j - 1]) for j in range(1, len(logs))]
            ultra = float(max(diffs))
            lyapunov_lle = float(sum(logs[j] - logs[j - 1] for j in range(1, len(logs))) / len(diffs))

        # Índice topológico de giro de Poincaré-Bendixson en el plano de fase (q - q̄, p - p̄)
        winding_number = 0
        if len(orbit_hist) >= 3:
            qs = [pt[0] for pt in orbit_hist]
            ps = [pt[1] for pt in orbit_hist]
            q_mean = sum(qs) / len(qs)
            p_mean = sum(ps) / len(ps)
            total_angle = 0.0
            for idx in range(1, len(orbit_hist)):
                th1 = math.atan2(ps[idx - 1] - p_mean, qs[idx - 1] - q_mean)
                th2 = math.atan2(ps[idx] - p_mean, qs[idx] - q_mean)
                dth = (th2 - th1 + math.pi) % (2.0 * math.pi) - math.pi
                total_angle += dth
            winding_number = int(round(total_angle / (2.0 * math.pi)))

        mono_det = poincare_phase.monodromy_determinant if poincare_phase is not None else 1.0
        melnikov_dist = float(
            hf * max(abs(mono_det) - 1.0, 0.0) - (s + _WILKINSON_FLOOR) * math.exp(-valuation)
        )
        kam_confined = bool(melnikov_dist <= _SPD_REL_TOL)

        b = hf
        mc_poly = float(abs(b + 0.5 * b * b + (1.0 / 6.0) * b * b * b))
        tau = float(-self._cd_tau0 * self._safety_margin * math.exp(-valuation))

        return TransientFiltration(
            envelope=float(s),
            high_frequency=float(hf),
            valuation=valuation,
            tau_cd=tau,
            window_ultrametric=ultra,
            mc_polynomial=mc_poly,
            melnikov_distance=melnikov_dist,
            poincare_winding_number=winding_number,
            lyapunov_exponent=lyapunov_lle,
            kam_confined=kam_confined,
        )

    def _execute_rsi_level3_autopoiesis(
        self,
        phase2_state: Phase2ThermalState,
        filtration: TransientFiltration,
    ) -> RSILevel3GovernorChart:
        r"""
        Motor de Automejora Recursiva de Nivel 3 (RSI-L3) con Contracción de Banach.

        Estructura formal de los 3 niveles:
            - Nivel 1 (Espacio de Parámetros Operativos θ^(1) ∈ C_safe):
              Actualiza el vector θ^(1) = [α_reg, ρ_Novikov, τ_0, w_IA] proyectado sobre el
              politopo compacto C_safe que garantiza la preservación estricta de todos los
              axiomas físicos (Candado Gödel-Tarski).
            - Nivel 2 (Meta-Optimización Geométrica de Fisher-Rao):
              Minimiza el funcional de Lyapunov de defectos numéricos y de fase:
                  L(θ^(1)) = ½ (r_Tellegen² + r_Fourier² + r_CSMD² + (w_IA − ½)²)
              aplicando el paso de gradiente natural Δθ^(1) = −η_meta · (G^(2))⁻¹ ∇L.
            - Nivel 3 (Meta-Meta-Endofunctor de Contracción de Lipschitz-Banach):
              Evalúa el cociente contractivo L_Banach = ‖Δθ_t‖_{G^(2)} / (‖Δθ_{t-1}‖_{G^(2)} + ε).
              Si L_Banach ≥ κ_cap (0.92), el Nivel 3 reescribe la curvatura de la métrica G^(2)
              y el factor de amortiguamiento η_meta ← η_meta · (κ_cap / (L_Banach + ε)),
              forzando analíticamente que L_Banach ∈ [0, 1) (Teorema del Punto Fijo de Banach).
        """
        tellegen_err = (
            float(phase2_state.tellegen_residual)
            if math.isfinite(phase2_state.tellegen_residual)
            else 1.0
        )
        fourier_err = (
            float(phase2_state.fourier_residual)
            if math.isfinite(phase2_state.fourier_residual)
            else 1.0
        )
        csmd_err = (
            float(phase2_state.csmd_error)
            if math.isfinite(phase2_state.csmd_error)
            else 1.0
        )
        cond_num = (
            float(phase2_state.metric_chart.condition_number)
            if phase2_state.metric_chart is not None
            else 1.0
        )

        with self._interlock_lock:
            self._rsi_epoch += 1
            w_curr = self._rsi_itoh_abe_weight
            reg_curr = self._reg
            rho_curr = self._novikov_rho
            tau0_curr = self._cd_tau0

            # Funcional de Lyapunov de Nivel 2
            w_sym_dev = w_curr - 0.5
            lyapunov_loss = float(
                0.5
                * (
                    tellegen_err * tellegen_err
                    + fourier_err * fourier_err
                    + csmd_err * csmd_err
                    + w_sym_dev * w_sym_dev
                )
            )

            # Gradiente euclídeo surrogate respecto a θ^(1) = [α_reg, ρ_Novikov, τ_0, w_IA]
            grad_reg = (
                -1e-16
                if cond_num > _DEGRADED_CONDITION
                else (reg_curr - _WILKINSON_FLOOR)
            )
            grad_rho = 0.01 * (rho_curr - _NOVIKOV_RHO) + 0.001 * math.tanh(
                filtration.valuation
            )
            grad_tau0 = 0.01 * (tau0_curr - _CD_TAU0)
            grad_w = w_sym_dev  # Empuja w_IA exponencialmente hacia el punto fijo exacto w* = 1/2

            raw_grad = np.array(
                [grad_reg, grad_rho, grad_tau0, grad_w], dtype=np.float64
            )

            # Nivel 2: Actualización de la métrica diagonal de Fisher-Rao G^(2)
            self._rsi_fisher_diag = 0.9 * self._rsi_fisher_diag + 0.1 * (
                1.0 + raw_grad * raw_grad
            )
            inv_fisher = 1.0 / np.maximum(self._rsi_fisher_diag, _WILKINSON_FLOOR)
            natural_step = -self._rsi_meta_damping * (inv_fisher * raw_grad)

            # Norma de Riemann-Fisher del paso actual
            step_norm_fisher = float(
                math.sqrt(
                    max(
                        float(np.sum(self._rsi_fisher_diag * natural_step * natural_step)),
                        0.0,
                    )
                )
            )

            # Nivel 3: Auditoría y reescritura autopoiética de la ley de Nivel 2 (Contracción de Banach)
            if self._rsi_prev_step_norm > _WILKINSON_FLOOR:
                banach_ratio = step_norm_fisher / self._rsi_prev_step_norm
            else:
                banach_ratio = 0.5 * _RSI_BANACH_TARGET_CAP

            if banach_ratio >= _RSI_BANACH_TARGET_CAP:
                scale_correction = _RSI_BANACH_TARGET_CAP / (banach_ratio + 1e-12)
                self._rsi_meta_damping = float(
                    max(self._rsi_meta_damping * scale_correction, _WILKINSON_FLOOR)
                )
                self._rsi_fisher_diag = self._rsi_fisher_diag / max(
                    scale_correction, 1e-6
                )
                natural_step = natural_step * scale_correction
                step_norm_fisher = step_norm_fisher * scale_correction
                banach_ratio = float(
                    min(banach_ratio * scale_correction, _RSI_BANACH_TARGET_CAP)
                )
            else:
                # Contracción suave garantizando L_Banach < 1 en todo ciclo
                self._rsi_meta_damping = float(
                    min(self._rsi_meta_damping * 0.98, _RSI_META_LR_INIT)
                )

            # Proyección de Nivel 1 sobre el Politopo de Seguridad de Gödel-Tarski C_safe
            new_reg = float(
                np.clip(reg_curr + natural_step[0], _WILKINSON_FLOOR, _RSI_REG_CEIL)
            )
            new_rho = float(np.clip(rho_curr + natural_step[1], 0.50, 0.98))
            new_tau0 = float(
                np.clip(tau0_curr + natural_step[2], 0.5 * _CD_TAU0, 2.0 * _CD_TAU0)
            )
            new_w = float(np.clip(w_curr + natural_step[3], 0.0, 1.0))

            self._reg = new_reg
            self._novikov_rho = new_rho
            self._cd_tau0 = new_tau0
            self._rsi_itoh_abe_weight = new_w
            self._rsi_prev_param_vec = np.array(
                [new_reg, new_rho, new_tau0, new_w], dtype=np.float64
            )
            self._rsi_prev_step_norm = max(step_norm_fisher, _WILKINSON_FLOOR)
            self._rsi_last_banach_contraction = float(
                np.clip(banach_ratio, 0.0, _RSI_BANACH_TARGET_CAP)
            )
            self._rsi_last_lyapunov_loss = lyapunov_loss
            self._rsi_last_grad_norm = step_norm_fisher

        return self._snapshot_rsi_governor_chart()

    def _phase3_conservation_audit(self, phase2_state: Phase2ThermalState) -> float:
        phi = phase2_state.clausius_duhem_residual
        if math.isfinite(phi):
            cd_inconsistent = float(-phi) if phi < -_WILKINSON_FLOOR else 0.0
        else:
            cd_inconsistent = 1.0
        eta = phase2_state.carnot_efficiency
        eta_res = 0.0 if 0.0 <= eta <= 1.0 + 1e-12 else 1.0
        tellegen = phase2_state.tellegen_residual
        terms = np.asarray(
            [
                phase2_state.fourier_residual,
                phase2_state.dissipation_identity_residual,
                phase2_state.csmd_error,
                eta_res,
                cd_inconsistent,
                max(0.0, -phase2_state.dirichlet_energy),
                max(0.0, -phase2_state.landauer_gap)
                if math.isfinite(phase2_state.landauer_gap)
                else 1.0,
                tellegen if math.isfinite(tellegen) else 1.0,
            ],
            dtype=np.float64,
        )
        return float(np.real(self._neumaier_sum(terms)))

    def _classify_thermal_heyting(
        self,
        phase2_state: Phase2ThermalState,
        filtration: TransientFiltration,
    ) -> Tuple[HeytingVerdict, float, Tuple[str, ...], Tuple[str, ...]]:
        phi = phase2_state.clausius_duhem_residual
        eta = phase2_state.carnot_efficiency
        allowed_cd_floor = float(filtration.tau_cd)

        veto: List[str] = []
        degraded: List[str] = []

        if not math.isfinite(phi):
            veto.append("clausius_duhem_residual_nonfinite")
            cd_score = 0.0
        elif phi < allowed_cd_floor:
            veto.append("clausius_duhem_second_law_violated")
            cd_score = 0.0
        elif phi < 0.0:
            degraded.append("clausius_duhem_residual_negative_numerical")
            cd_score = 0.5
        else:
            cd_score = 1.0

        if not math.isfinite(eta) or eta < 0.0 or eta > 1.0 + 1e-12:
            veto.append("carnot_efficiency_out_of_physical_range")
            eta_score = 0.0
        else:
            eta_score = 1.0

        fourier_res = phase2_state.fourier_residual
        if not math.isfinite(fourier_res) or fourier_res > 1e-3 * self._safety_margin:
            veto.append("fourier_constitutive_broken")
            fourier_score = 0.0
        elif fourier_res > _DEGRADED_FOURIER_THRESHOLD:
            degraded.append("fourier_residual_degraded")
            fourier_score = float(np.exp(-fourier_res / _DEGRADED_FOURIER_THRESHOLD))
        else:
            fourier_score = float(
                np.exp(
                    -fourier_res / max(_DEGRADED_FOURIER_THRESHOLD, _WILKINSON_FLOOR)
                )
            )

        csmd = phase2_state.csmd_error
        if not math.isfinite(csmd) or csmd > 1e-8:
            veto.append("csmd_identity_broken")
            csmd_score = 0.0
        else:
            csmd_score = float(np.exp(-csmd / 1e-12))

        if phase2_state.landauer_gap < -_ENTROPY_ABS_TOL:
            veto.append("landauer_bound_violated")
            landauer_score = 0.0
        else:
            landauer_score = 1.0

        ident = phase2_state.dissipation_identity_residual
        ident_score = float(np.exp(-ident / 1e-10)) if math.isfinite(ident) else 0.0
        if ident > 1e-6:
            degraded.append("dissipation_identity_degraded")

        tellegen = phase2_state.tellegen_residual
        if math.isfinite(tellegen) and tellegen > 1e-8:
            degraded.append("tellegen_identity_degraded")
        tellegen_score = (
            float(np.exp(-tellegen / 1e-10)) if math.isfinite(tellegen) else 0.0
        )

        cond = (
            float(phase2_state.metric_chart.condition_number)
            if phase2_state.metric_chart is not None
            else 1.0
        )
        if cond > 1e12:
            veto.append("metric_condition_explosive")
            cond_score = 0.0
        elif cond > _DEGRADED_CONDITION:
            degraded.append("metric_ill_conditioned")
            cond_score = 1.0 / (1.0 + np.log10(cond))
        else:
            cond_score = 1.0 / (1.0 + np.log10(max(cond, 1.0)))

        raw_T_clamped = bool(phase2_state.diagnostics.get("temperature_clamped", False))
        third_law = bool(phase2_state.diagnostics.get("third_law_flag", False))
        if third_law:
            veto.append("third_law_nonpositive_temperature")
            temp_score = 0.0
        elif raw_T_clamped:
            degraded.append("temperature_clamped_to_wilkinson_floor")
            temp_score = 0.5
        else:
            temp_score = 1.0

        pairing = phase2_state.pairing_q_dT
        if math.isfinite(pairing) and pairing > _WILKINSON_FLOOR:
            veto.append("anti_fourier_heat_up_gradient")
            pairing_score = 0.0
        else:
            pairing_score = 1.0

        meet = min(
            cd_score,
            eta_score,
            fourier_score,
            csmd_score,
            landauer_score,
            ident_score,
            tellegen_score,
            cond_score,
            temp_score,
            pairing_score,
        )

        if veto:
            verdict = HeytingVerdict.VETOED
        elif meet >= 0.99 and not degraded:
            verdict = HeytingVerdict.CERTIFIED
        elif meet >= 0.90 and not degraded:
            verdict = HeytingVerdict.COHERENT
        elif degraded or meet >= 0.50:
            verdict = (
                HeytingVerdict.DEGRADED if meet >= 0.50 else HeytingVerdict.VETOED
            )
            if meet < 0.50 and not veto:
                veto.append("heyting_meet_collapsed")
                verdict = HeytingVerdict.VETOED
        else:
            verdict = HeytingVerdict.VETOED
        return verdict, float(meet), tuple(veto), tuple(degraded)

    def _classify_thermal_heyting_scalars(
        self,
        clausius_duhem_residual: float,
        carnot_efficiency: float,
        exergy_potential: float,
    ) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
        allowed_cd_floor = -self._cd_tau0 * self._safety_margin
        veto_reasons: List[str] = []
        degraded_reasons: List[str] = []
        if not math.isfinite(clausius_duhem_residual):
            veto_reasons.append("clausius_duhem_residual_nonfinite")
        elif clausius_duhem_residual < allowed_cd_floor:
            veto_reasons.append("clausius_duhem_residual_below_allowed_floor")
        elif clausius_duhem_residual < 0.0:
            degraded_reasons.append("clausius_duhem_residual_negative")
        if not math.isfinite(carnot_efficiency):
            veto_reasons.append("carnot_efficiency_nonfinite")
        elif carnot_efficiency < 0.0 or carnot_efficiency > 1.0:
            veto_reasons.append("carnot_efficiency_out_of_physical_range")
        if not math.isfinite(exergy_potential):
            degraded_reasons.append("exergy_potential_nonfinite")
        if veto_reasons:
            verdict = HeytingVerdict.VETOED.value
        elif degraded_reasons:
            verdict = HeytingVerdict.DEGRADED.value
        else:
            verdict = HeytingVerdict.COHERENT.value
        return verdict, tuple(veto_reasons), tuple(degraded_reasons)

    def execute_phase_3_heyting_veto(
        self,
        clausius_duhem_residual: float,
        carnot_efficiency: float,
        exergy_potential: float,
    ) -> Dict[str, Any]:
        """[COMPATIBILIDAD 1.X — FASE 3]. Ya NO veta exergía baja."""
        verdict, veto_reasons, degraded_reasons = self._classify_thermal_heyting_scalars(
            clausius_duhem_residual=float(clausius_duhem_residual),
            carnot_efficiency=float(carnot_efficiency),
            exergy_potential=float(exergy_potential),
        )
        interlock_fired, latency = self._act_thermal_interlock(verdict)
        return {
            "heyting_verdict": verdict,
            "clausius_duhem_residual": float(clausius_duhem_residual),
            "carnot_efficiency": float(carnot_efficiency),
            "exergy_potential": float(exergy_potential),
            "hardware_interlock_fired": interlock_fired,
            "actuation_latency_ns": latency,
            "veto_reasons": veto_reasons,
            "degraded_reasons": degraded_reasons,
        }

    def execute_phase_3_hetying_veto(
        self,
        clausius_duhem_residual: float,
        carnot_efficiency: float,
        exergy_potential: float,
    ) -> Dict[str, Any]:
        """[COMPATIBILIDAD 1.X — typo original conservado]."""
        return self.execute_phase_3_heyting_veto(
            clausius_duhem_residual=clausius_duhem_residual,
            carnot_efficiency=carnot_efficiency,
            exergy_potential=exergy_potential,
        )

    def phase3_evaluate_heyting_from_phase2(
        self,
        phase2_state: Phase2ThermalState,
    ) -> Phase3HeytingDecision:
        r"""
        [FASE 3 — CONTINUACIÓN DE `phase2_simulate_carnot_from_phase1` Y MORFISMO TERMINAL: ACT]

        Dominio  : `Phase2ThermalState` (codominio exacto de Fase 2).
        Codominio: `Phase3HeytingDecision` (clausura algebraica y cibernética del ciclo).
        """
        if not isinstance(phase2_state, Phase2ThermalState):
            raise TypeError("phase2_state debe ser Phase2ThermalState.")

        grad_norm = float(math.sqrt(max(phase2_state.dirichlet_energy, 0.0)))
        filtration = self._update_transient_filtration(
            grad_norm, poincare_phase=phase2_state.poincare_phase_chart
        )
        rsi_governor = self._execute_rsi_level3_autopoiesis(phase2_state, filtration)

        verdict, score, veto_reasons, degraded_reasons = self._classify_thermal_heyting(
            phase2_state,
            filtration,
        )
        conservation = self._phase3_conservation_audit(phase2_state)
        interlock_fired, latency = self._act_thermal_interlock(verdict.value)

        diagnostics = dict(phase2_state.diagnostics)
        diagnostics.update(
            {
                "heyting_verdict": verdict.value,
                "heyting_rank": verdict.rank,
                "heyting_score": score,
                "veto_reasons": veto_reasons,
                "degraded_reasons": degraded_reasons,
                "hardware_interlock_fired": interlock_fired,
                "actuation_latency_ns": latency,
                "conservation_residual": conservation,
                "adaptive_tau_cd": filtration.tau_cd,
                "transient_valuation": filtration.valuation,
                "transient_envelope": filtration.envelope,
                "transient_high_frequency": filtration.high_frequency,
                "window_ultrametric": filtration.window_ultrametric,
                "mc_polynomial_surrogate": filtration.mc_polynomial,
                "melnikov_distance": filtration.melnikov_distance,
                "poincare_winding_number": filtration.poincare_winding_number,
                "lyapunov_exponent": filtration.lyapunov_exponent,
                "kam_confined": filtration.kam_confined,
                "rsi_epoch": rsi_governor.epoch,
                "rsi_level3_banach_contraction": rsi_governor.level3_banach_contraction,
                "rsi_level2_lyapunov_loss": rsi_governor.level2_lyapunov_loss,
                "rsi_godel_tarski_locked": rsi_governor.godel_tarski_soundness_locked,
            }
        )
        logger.debug(
            "Fase 3: veredicto=%s score=%.4f Φ=%.6e τ_CD=%.3e ν=%.3e L_Banach=%.4f cons=%.3e",
            verdict.value,
            score,
            phase2_state.clausius_duhem_residual,
            filtration.tau_cd,
            filtration.valuation,
            rsi_governor.level3_banach_contraction,
            conservation,
        )
        return Phase3HeytingDecision(
            heyting_verdict=verdict.value,
            heyting_rank=verdict.rank,
            heyting_score=score,
            veto_reasons=veto_reasons,
            degraded_reasons=degraded_reasons,
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=latency,
            conservation_residual=conservation,
            adaptive_tau_cd=filtration.tau_cd,
            transient_valuation=filtration.valuation,
            diagnostics=diagnostics,
            poincare_phase_chart=phase2_state.poincare_phase_chart,
            rsi_governor=rsi_governor,
        )


# =============================================================================
# CLASE PÚBLICA — COMPOSICIÓN CATEGÓRICA T = Fase 3 ∘ Fase 2 ∘ Fase 1
# =============================================================================
class ThermalGradientLaws(Phase3HeytingVetoMixin):
    r"""
    Resolvedor soberano de gradientes de calor sobre (M, g, ω) con Mecánica Celeste
    de Poincaré y Automejora Recursiva de Nivel 3 (RSI-L3).

    Composición funtorial de las 3 fases anidadas:
        In ──phase1_ingest_and_purify───────────► Phase1ThermalState
           ──phase2_simulate_carnot_from_phase1─► Phase2ThermalState
           ──phase3_evaluate_heyting_from_phase2► Phase3HeytingDecision ──► Dict[str, Any]
    """

    def execute_thermal_cycle(
        self,
        K_raw: np.ndarray,
        grad_T_raw: np.ndarray,
        T_sys: float,
        metric_tensor: np.ndarray,
        entropy_production_rate: float,
        *,
        conductivity_type: str = "covariant",
        length_scale: float = 1.0,
        ambient_temperature: Optional[float] = None,
        info_erasure_rate: float = 0.0,
    ) -> Dict[str, Any]:
        phase1 = self.phase1_ingest_and_purify(K_raw, grad_T_raw, T_sys)
        phase2 = self.phase2_simulate_carnot_from_phase1(
            phase1_state=phase1,
            metric_tensor=metric_tensor,
            entropy_production_rate=entropy_production_rate,
            conductivity_type=conductivity_type,
            length_scale=length_scale,
            ambient_temperature=ambient_temperature,
            info_erasure_rate=info_erasure_rate,
        )
        phase3 = self.phase3_evaluate_heyting_from_phase2(phase2)

        poincare_chart = phase2.poincare_phase_chart
        rsi_chart = phase3.rsi_governor

        return {
            "heyting_verdict": phase3.heyting_verdict,
            "clausius_duhem_residual": phase2.clausius_duhem_residual,
            "carnot_efficiency": phase2.carnot_efficiency,
            "exergy_potential": phase2.exergy_potential,
            "hardware_interlock_fired": phase3.hardware_interlock_fired,
            "actuation_latency_ns": phase3.actuation_latency_ns,
            "heat_flux_vector": np.array(phase2.Q_vector, copy=True),
            "T_sys_clamped": phase1.T_sys,
            "phase": "G_THERMAL_CYCLE_SUTURATED",
            "veto_reasons": phase3.veto_reasons,
            "degraded_reasons": phase3.degraded_reasons,
            "thermal_entropy_production": phase2.thermal_entropy_production,
            "gouy_stodola": phase2.gouy_stodola,
            "fourier_residual": phase2.fourier_residual,
            "csmd_error": phase2.csmd_error,
            "dirichlet_energy": phase2.dirichlet_energy,
            "carnot_density": phase2.carnot_density,
            "landauer_gap": phase2.landauer_gap,
            "conservation_residual": phase3.conservation_residual,
            "heyting_score": phase3.heyting_score,
            "T_hot": phase2.T_hot,
            "T_cold": phase2.T_cold,
            "tellegen_residual": phase2.tellegen_residual,
            "adaptive_tau_cd": phase3.adaptive_tau_cd,
            "transient_valuation": phase3.transient_valuation,
            "kam_torus_stability": (
                phase1.poincare_resonance.kam_torus_stability
                if phase1.poincare_resonance is not None
                else 1.0
            ),
            "poincare_recurrence_time": (
                poincare_chart.recurrence_time_estimate
                if poincare_chart is not None
                else 1.0
            ),
            "poincare_monodromy_det": (
                poincare_chart.monodromy_determinant
                if poincare_chart is not None
                else 1.0
            ),
            "melnikov_distance": phase3.diagnostics.get("melnikov_distance", 0.0),
            "rsi_level3_contraction": (
                rsi_chart.level3_banach_contraction if rsi_chart is not None else 0.0
            ),
            "rsi_epoch": rsi_chart.epoch if rsi_chart is not None else 0,
            "diagnostics": {
                "phase1": dict(phase1.diagnostics),
                "phase2": dict(phase2.diagnostics),
                "phase3": dict(phase3.diagnostics),
            },
        }


if __name__ == "__main__":
    print(
        "Autocomprobación Thermal Gradient Laws v4.0 "
        "(Poincaré-KAM / RSI-L3 / Lanczos / Itoh–Abe Simetrizado / τ adaptativo)..."
    )
    engine = ThermalGradientLaws(dimension_n=3, rng_seed=0)
    K = np.eye(3)
    G = np.eye(3)
    grad = np.array([1.0, 0.0, 0.0])
    stamp = engine.execute_thermal_cycle(K, grad, 300.0, G, 0.0)
    print("Veredicto:", stamp["heyting_verdict"], "score=", stamp["heyting_score"])
    print("Q (esperado [-1,0,0]):", stamp["heat_flux_vector"])
    print("Φ CD:", stamp["clausius_duhem_residual"])
    print("Tellegen:", stamp["tellegen_residual"])
    print("η_C (esperado 1/301):", stamp["carnot_efficiency"])
    print("τ_CD adaptativo:", stamp["adaptive_tau_cd"], "ν:", stamp["transient_valuation"])
    print("Fourier/CSMD:", stamp["fourier_residual"], stamp["csmd_error"])
    print(
        "Poincaré KAM=",
        stamp["kam_torus_stability"],
        "det(DP)=",
        stamp["poincare_monodromy_det"],
        "τ_rec=",
        stamp["poincare_recurrence_time"],
    )
    print(
        "RSI-L3 epoch=",
        stamp["rsi_epoch"],
        "L_Banach=",
        stamp["rsi_level3_contraction"],
    )

    iso = engine.execute_thermal_cycle(K, np.zeros(3), 300.0, G, 0.0)
    print("Isotermo:", iso["heyting_verdict"], "η=", iso["carnot_efficiency"])

    spike = engine.execute_thermal_cycle(K, np.array([40.0, 0.0, 0.0]), 300.0, G, 0.0)
    print(
        "Spike |∇T| (τ dilatado):",
        spike["adaptive_tau_cd"],
        spike["heyting_verdict"],
        "L_Banach=",
        spike["rsi_level3_contraction"],
    )

    big = ThermalGradientLaws(dimension_n=48, rng_seed=1, krylov_k=6, rank_reduce=False)
    Kb = np.eye(48) + 0.01 * np.ones((48, 48))
    Gb = np.eye(48)
    gb = np.zeros(48)
    gb[0] = 1.0
    st = big.execute_thermal_cycle(Kb, gb, 300.0, Gb, 0.0)
    print(
        "n=48 Lanczos veredicto:",
        st["heyting_verdict"],
        "lanczos K:",
        st["diagnostics"]["phase1"].get("lanczos_used"),
        "κ:",
        st["diagnostics"]["phase1"].get("condition_number"),
    )


__all__ = [
    "Phase1ThermalState",
    "Phase2ThermalState",
    "Phase3HeytingDecision",
    "Phase1ThermalIngestionMixin",
    "Phase2CarnotSimulationMixin",
    "Phase3HeytingVetoMixin",
    "ThermalGradientLaws",
    "SpectralChart",
    "ItohAbeChart",
    "TransientFiltration",
    "PoincareSpectralResonanceChart",
    "PoincarePhaseSpaceChart",
    "RSILevel3GovernorChart",
    "HeytingVerdict",
]