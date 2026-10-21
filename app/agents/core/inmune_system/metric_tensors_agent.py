# -*- coding: utf-8 -*-
r"""
====================================================================================================
ARQUITECTURA COGNITIVA APU FILTER - SISTEMA INMUNE SOBERANO
====================================================================================================

Nombre: MetricTensorsAgent
Ubicación: app/agents/core/inmune_system/metric_tensors_agent.py
Versión: 7.3.0  (evolución rigurosa de 7.2.0)
Función: Soberano de Calibre Riemanniano del sistema inmune. Orquesta la gobernanza geométrica,
         el flujo de Ricci discreto normalizado con automejora recursiva RSI³ (niveles 1 ⊂ 2 ⊂ 3,
         v7.3.0 Bianchi–Darboux–QR–Lindstedt–Killing–Morse) y el mapa de retorno de Poincaré en el
         fibrado cotangente T^*\mathcal{M} con variables acción-ángulo de Hamilton–Jacobi, monodromía
         variacional, función de Melnikov homoclínica y cuantización geométrica prequántica.

----------------------------------------------------------------------------------------------------
DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO MOTOR ESPECTRAL:
----------------------------------------------------------------------------------------------------
El Soberano de Calibre Riemanniano `MetricTensorsAgent` rige la gobernanza geométrica de la Capa 0
(Estrato Wisdom, \mathcal{V}_{\mathbb{W}}) dentro del Sistema Inmune de APU Filter. Su cometido principal
es la auditoría ciber-física, el control de la curvatura escalar y la verificación de invariantes
simplécticos en la variedad de fases \mathcal{M} \subset \mathbb{R}^7. Orquesta el motor espectral mediante un
ciclo OODA trifásico \operatorname{Auditar} = \Phi_3 \circ \Phi_2 \circ \Phi_1:
• Fase 1 (\Phi_1 - Observe): Evaluación del tejido 0-jet, derivación no conmutativa de Connes,
  Christoffel de Koszul, tensor de Einstein, congruencia de volumen y residuales de Noether/Kirchhoff.
• Fase 2 (\Phi_2 - Orient): Integración del flujo de Ricci normalizado con automejora recursiva (\mathrm{RSI}^3),
  certificación de punto fijo de Banach (L_\Phi \le 0.95), construcción del mapa de retorno de Poincaré
  en T^*\mathcal{M}, monodromía variacional M = \mathrm{D}\mathcal{T}|_\Sigma y espectro de Lyapunov continuo.
• Fase 3 (\Phi_3 - Decide/Act): Estabilidad de toros KAM, integral de Melnikov para ruptura homoclínica,
  índice de Conley–Morse, cuantización prequántica F_\nabla = \omega y adjudicación trivalente en el retículo
  de Heyting \Omega_3 con emisión de certificados inmunitarios inmutables.

----------------------------------------------------------------------------------------------------
MARCO TEÓRICO:
----------------------------------------------------------------------------------------------------
1. GOVERNANZA GEOMÉTRICA NO CONMUTATIVA Y TRIPLE ESPECTRAL DE CONNES:
   El soberano evalúa las fluctuaciones Métricas sobre el álgebra de observables \mathcal{A}.
   La derivación no conmutativa \partial_\mu G_{\nu\sigma} = [X_\mu, G]_{\nu\sigma} se combina con
   el operador Dirac de Connes \mathcal{D} para acotar la velocidad de deformación de las métricas.
   La cota Lipschitz exacta de Connes–Daleckii–Krein sobre el operador de densidad \rho^{-1/2} garantiza
   que ninguna transición de estado inmunitario viole la continuidad geométrica:
   $$L_{\mathrm{Connes}} = \max_{i < j} \left| \frac{\lambda_j^{-1/2} - \lambda_i^{-1/2}}{\lambda_j - \lambda_i} \right| \le \frac{1}{2 \lambda_{\min}^{3/2}}$$

2. AUTOMEJORA RECURSIVA DE ESTADOS Y MÉTODOS (\mathrm{RSI}^3):
   El agente ejecuta una automejora en tres niveles anidados con selección dinámica de operadores:
   $$\Phi^{(3)} \supset \Phi^{(2)} \supset \Phi^{(1)}$$
   • \Phi^{(1)}: Paso de Ricci normalizado G_{k+1} = \Pi_{\mathrm{SPD}}\left(G_k - 2\Delta\tau (\operatorname{Ric} - \frac{R}{n} G)\right).
   • \Phi^{(2)}: Adaptación de la tasa temporal \Delta\tau_{k+1} mediante el gradiente de curvatura |\nabla R_k|.
   • \Phi^{(3)}: Adaptación del hiperparámetro \eta_{k+1} para forzar la contracción L_\Phi < 0.95, acoplado al
     selector de método \sigma_{k+1} = (\text{Proyector Cayley-Darboux / Higham}, \text{Discretizador Koszul / Proxy})
     según el estado del residual de Bianchi \nabla_\mu \mathrm{Ein}^{\mu\nu} y el número de condición \kappa_2(G).

3. MECÁNICA CELESTE DE POINCARÉ Y ESTRUCTURA HAMILTONIANA:
   Sobre la sección transversal de Poincaré \Sigma = \{R = \bar{R}\} \subset T^*\mathcal{M}, el agente extrae:
   • Coordenadas de Acción-Ángulo (I, \theta): I = \frac{1}{2\pi} \oint_\Sigma p \, \mathrm{d}q, \theta = \operatorname{fase}(\operatorname{FFT}(q)).
   • Matriz de Monodromía M = \mathrm{D}\mathcal{T}|_\Sigma \in \operatorname{Sp}(2r, \mathbb{R}) y multiplicador de Floquet \mu_{\max} = \rho(M).
   • Espectro completo de Lyapunov \{\lambda_i\}_{i=1}^d mediante ortogonalización QR de Benettin–Shimada.
   • Integral de Melnikov discreta: M(\theta_0) = \sum_k \langle J \nabla H(z_k^0), g(z_k^0, \theta_0) \rangle \Delta\tau \neq 0 \implies \text{Tangle Homoclínico}.
   • Corrección de Poincaré–Lindstedt T_{\mathrm{Lindstedt}} = T_0 (1 + \varepsilon T_1) y cota de recurrencia \tau_{\mathrm{rec}} \ge \mathrm{Vol}(\mathcal{M})/\mathrm{Vol}(\mathrm{tubo}).

4. CUANTIZACIÓN GEOMÉTRICA PREQUÁNTICA (KOSTANT–SOURIAU):
   Para certificar la invarianza simpléctica del lazo de control, el agente verifica que la curvatura de la
   conexión en el fibrado en rectas prequántico (L, \nabla, \langle \cdot, \cdot \rangle) coincida con la 2-forma de Darboux:
   $$F_\nabla = \omega \iff \mathrm{d}\theta = \omega \quad \text{con } \theta = p_i \mathrm{d}q^i$$
   El residual prequántico evalúa la anti-simetría \|\omega + \omega^\top\|_F y la invarianza de pullback \|M^\top \omega M - \omega\|_F \le 10^{-9}.

5. ADJUDICACIÓN TRIVALENTE DE HEYTING \Omega_3 Y HARDWARE INTERLOCK:
   Las observaciones ciber-físicas colapsan en el álgebra de Heyting \Omega_3 = \{\bot = \mathtt{COHERENT} < \frac{1}{2} = \mathtt{DEGRADED} < \top = \mathtt{VETOED}\}.
   La combinación de veredictos utiliza la operación Join a \vee b = \max(a, b) ejecutada en un único ciclo de instrucción SIMD `OR` bitwise
   en el microcontrolador ESP32 (máscaras: 0x00, 0x01, 0x03). Si el veredicto colapsa a \mathtt{VETOED} (0x03), se dispara la interrupción de hardware Crowbar (< 400 ns).

6. EQUIVALENCIA CIRCUITO–GEOMÉTRICA:
   • Ley de Corrientes de Kirchhoff (KCL): \sum_\nu \mathrm{Ein}^{\mu\nu} = 0 \iff \nabla_\mu \mathrm{Ein}^{\mu\nu} = 0.
   • Ley de Tensiones de Kirchhoff (KVL): \mathrm{Ein}_{\mu\nu} = \mathrm{Ein}_{\nu\mu} \iff \text{Conservación del Momento en Cierres 0-jet}.

----------------------------------------------------------------------------------------------------
AXIOMAS FORMALES:
----------------------------------------------------------------------------------------------------
MTA1 (Positividad Espectral Mínima):
    \lambda_{\min}(G) \ge \lambda_\star = 10^{-12} \quad \forall G \in \operatorname{Sym}^+(n, \mathbb{R}).

MTA2 (Conservación de Volumen Simpléctico):
    M^\top \Omega M = \Omega \implies \det M = +1 \quad \text{y } \operatorname{Pf}(\Omega) = 1.

MTA3 (Contracción de Banach e Invariancia de Mapeo):
    L_\Phi = \operatorname{Lip}(\Phi) \le L_\star = 0.95 < 1.0 \quad \text{sobre el espacio completo } (\overline{\mathrm{SPD}}_n, \|\cdot\|_F).

MTA4 (Acatamiento Axiomático de Heyting):
    a \wedge (a \to b) \le b \quad \text{y } a \wedge c \le b \iff c \le (a \to b) \quad \forall a, b, c \in \Omega_3.

----------------------------------------------------------------------------------------------------
INVARIANTES DE ESTADO:
----------------------------------------------------------------------------------------------------
Invariante 1 (Techo de Condicionamiento Espectral):
    \kappa_2(G) = \frac{\lambda_{\max}(G)}{\lambda_{\min}(G)} \le 10^8.

Invariante 2 (Divergencia Nula Bianchi–Kirchhoff):
    \|\nabla_\mu \mathrm{Ein}^{\mu\nu}\|_2 \le 10^{-3} \|\mathrm{Ein}\|_F.

Invariante 3 (Pareado Simpléctico Palindrómico):
    |\lambda_i + \lambda_{d-1-i}| \le 10^{-2} (1 + |\lambda_i|) \quad \text{para dimensiones pares } d = 2n.

Invariante 4 (Asociatividad Categórica OODA de Jets):
    (\Phi_3 \circ \Phi_2) \circ \Phi_1 \cong \Phi_3 \circ (\Phi_2 \circ \Phi_1).
====================================================================================================
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations, product
from typing import Any, Callable, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APUFilter.Physics.MetricTensorsAgent")

__all__ = (
    "HeytingVerdict",
    "ProjectorMode",
    "RicciDiscretizer",
    "MetricTensorGovernanceError",
    "SpectralProfile",
    "GeometricState",
    "RSI3Trace",
    "PoincareData",
    "BanachCertificate",
    "MetricGovernanceCertificate",
    "Phase1Handoff",
    "Phase2Handoff",
    "MetricTensorsAgent",
)

__version__: Final[str] = "7.3.0"

# ── Constantes universales ──────────────────────────────────────────────────
_WILKINSON_EPS: Final[float] = float(np.finfo(np.float64).eps)
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_MAX_CONDITION_NUMBER: Final[float] = 1.0e8
_BANACH_TARGET: Final[float] = 0.95
_LYAPUNOV_CHAOS: Final[float] = 1.0e-2
_KAM_GAMMA: Final[float] = 1.0e-3
_KAM_TAU: Final[float] = 2.0
_RSI3_CONV_EPS: Final[float] = 1.0e-9
_SYMPLECTIC_ATOL: Final[float] = 1.0e-9
_COMPAT_REL_TOL: Final[float] = 1.0e-4
_FLOQUET_UNSTABLE: Final[float] = 1.0 + 1.0e-2
_LIPSCHITZ_PERT_REL: Final[float] = 1.0e-6
_ETA_BOUNDS: Final[Tuple[float, float]] = (1.0e-4, 10.0)
_TAU_BOUNDS: Final[Tuple[float, float]] = (1.0e-5, 0.1)
_ZETA_BOUNDS: Final[Tuple[float, float]] = (1.0e-4, 0.1)
_LYAP_DIM_CAP: Final[int] = 64
_CONSECUTIVE_CONV_STEPS: Final[int] = 3
_BIANCHI_REL_TOL: Final[float] = 1.0e-3
_WEYL_REL_TOL: Final[float] = 1.0e-2
_KILLING_REL_TOL: Final[float] = 1.0e-3
_PAIRING_ATOL: Final[float] = 1.0e-2
_JACOBI_FD_EPS: Final[float] = 1.0e-7
_PREQUANTUM_ATOL: Final[float] = 1.0e-8
_GAUSS_BONNET_4PI: Final[float] = 4.0 * float(np.pi)
_METHOD_SWITCH_HYSTERESIS: Final[float] = 1.25


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  TIPOS, RETÍCULO DE HEYTING Ω₃ Y CERTIFICADOS INMUTABLES                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class HeytingVerdict(Enum):
    r"""Clasificador trivalente en $\Omega_3$. Codificación SIMD: 0x00, 0x01, 0x03."""
    COHERENT = "COHERENT"
    DEGRADED = "DEGRADED"
    VETOED   = "VETOED"

    @property
    def omega(self) -> float:
        return {HeytingVerdict.COHERENT: 0.0,
                HeytingVerdict.DEGRADED: 0.5,
                HeytingVerdict.VETOED:   1.0}[self]

    @property
    def mask(self) -> int:
        return {HeytingVerdict.COHERENT: 0x00,
                HeytingVerdict.DEGRADED: 0x01,
                HeytingVerdict.VETOED:   0x03}[self]

    @classmethod
    def from_omega(cls, value: float) -> "HeytingVerdict":
        if value <= 0.0:
            return cls.COHERENT
        if value < 1.0:
            return cls.DEGRADED
        return cls.VETOED

    @classmethod
    def from_mask(cls, mask_val: int) -> "HeytingVerdict":
        if (mask_val & 0x03) == 0x03:
            return cls.VETOED
        if (mask_val & 0x01) == 0x01:
            return cls.DEGRADED
        return cls.COHERENT

    def join(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict.from_mask(self.mask | other.mask)

    @classmethod
    def fast_join_all(cls, verdicts: List["HeytingVerdict"]) -> "HeytingVerdict":
        acc = 0
        for v in verdicts:
            acc |= v.mask
            if acc == 0x03:
                return cls.VETOED
        return cls.from_mask(acc)

    def meet(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict.from_omega(min(self.omega, other.omega))

    def implies(self, other: "HeytingVerdict") -> "HeytingVerdict":
        r"""Residuación de cadena total: $a\to b=\top$ si $a\le b$, else $b$."""
        if self.omega <= other.omega:
            return HeytingVerdict.COHERENT
        return other

    def leq(self, other: "HeytingVerdict") -> bool:
        return self.omega <= other.omega


class ProjectorMode(Enum):
    r"""Selector de proyección SPD (automejora de método, RSI³)."""
    SPECTRAL_HIGHAM = "SPECTRAL_HIGHAM"
    CAYLEY_DARBOUX  = "CAYLEY_DARBOUX"


class RicciDiscretizer(Enum):
    r"""Selector de discretización de Ricci (automejora de método, RSI³)."""
    KOSZUL_QUADRATIC = "KOSZUL_QUADRATIC"
    TRACEFREE_PROXY  = "TRACEFREE_PROXY"


class MetricTensorGovernanceError(Exception):
    """Excepción raíz para violaciones de invariantes geométricos."""


@dataclass(frozen=True, slots=True)
class SpectralProfile:
    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float
    is_spd: bool


@dataclass(frozen=True, slots=True)
class GeometricState:
    r"""Estado geométrico 0-jet de $(\mathcal{M},\mathbf{G})$."""
    metric: NDArray[np.float64]
    christoffel: NDArray[np.float64]
    ricci_tensor: NDArray[np.float64]
    ricci_scalar: float
    einstein_tensor: NDArray[np.float64]
    spectral_profile: SpectralProfile
    log_volume: float
    eh_action: float
    compatibility_residual: float
    bianchi_residual: float = 0.0
    weyl_residual: float = 0.0
    killing_residual: float = 0.0
    kirchhoff_residual: float = 0.0
    volume_consistency: float = 0.0


@dataclass(frozen=True, slots=True)
class BanachCertificate:
    r"""Certificado del teorema del punto fijo de Banach sobre $(\mathrm{SPD}_n, \|\cdot\|_F)$."""
    lipschitz: float
    is_self_map: bool
    is_complete_domain: bool
    is_contraction: bool
    certified: bool


@dataclass(frozen=True, slots=True)
class RSI3Trace:
    rsi_level_achieved: int
    contraction_constants: Tuple[float, ...]
    delta_tau_history: Tuple[float, ...]
    eta_history: Tuple[float, ...]
    converged: bool
    quadratic_convergence_rate: float = 0.0
    method_lineage: Tuple[str, ...] = ()
    banach_certified: bool = False
    projector_mode: str = ProjectorMode.SPECTRAL_HIGHAM.value


@dataclass(frozen=True, slots=True)
class PoincareData:
    return_periods: Tuple[float, ...]
    lyapunov_max: float
    kam_stable: bool
    homoclinic_tangle: bool
    hopf_index: float
    diophantine_gamma: float
    floquet_multiplier: float = 0.0
    dominant_frequencies: Tuple[float, ...] = ()
    action: float = 0.0
    angle: float = 0.0
    melnikov_amplitude: float = 0.0
    lindstedt_period: float = 0.0
    recurrence_bound: float = 0.0
    monodromy_spectral_radius: float = 0.0
    conley_index: float = 0.0
    prequantum_curvature: float = 0.0
    first_integrals: Tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class MetricGovernanceCertificate:
    spectral_profile: SpectralProfile
    ricci_scalar_curvature: float
    lipschitz_connes_bound: float
    symplectic_volume_drift: float
    heyting_verdict: HeytingVerdict
    is_geometrically_coherent: bool
    rsi3_trace: RSI3Trace
    poincare_data: PoincareData
    banach_contraction_constant: float
    log_volume: float = 0.0
    einstein_hilbert_action: float = 0.0
    metric_compatibility_residual: float = 0.0
    bianchi_identity_residual: float = 0.0
    lyapunov_spectrum: Tuple[float, ...] = ()
    action_angle: Tuple[float, float] = (0.0, 0.0)
    melnikov_amplitude: float = 0.0
    heyting_axioms_verified: bool = True
    weyl_residual: float = 0.0
    killing_residual: float = 0.0
    kirchhoff_residual: float = 0.0
    volume_consistency: float = 0.0
    lyapunov_pairing_defect: float = 0.0
    generating_function_residual: float = 0.0
    ooda_associative: bool = True
    polyakov_residual: float = 0.0
    method_lineage: Tuple[str, ...] = ()


@dataclass
class Phase1Handoff:
    """Morfismo 1 → 2: germen geométrico + órbita RSI¹ + invariantes discretos de Noether."""
    initial_metric: NDArray[np.float64]
    orbit: List[NDArray[np.float64]]
    ricci_scalars: List[float]
    condition_numbers: List[float]
    delta_tau: float
    log_volumes: List[float] = field(default_factory=list)
    eh_actions: List[float] = field(default_factory=list)
    bianchi_residuals: List[float] = field(default_factory=list)
    conserved_quantities_audit: Dict[str, float] = field(default_factory=dict)
    weyl_residuals: List[float] = field(default_factory=list)
    killing_residuals: List[float] = field(default_factory=list)
    kirchhoff_residuals: List[float] = field(default_factory=list)
    first_integrals: Dict[str, float] = field(default_factory=dict)
    volume_consistencies: List[float] = field(default_factory=list)
    projector_mode: str = ProjectorMode.SPECTRAL_HIGHAM.value


@dataclass
class Phase2Handoff:
    """Morfismo 2 → 3: RSI³ + Poincaré + Lyapunov-QR + HJ + monodromía + Lindstedt."""
    phase1: Phase1Handoff
    lyapunov_max: float
    poincare_data: Dict[str, Any]
    contraction_constants: List[float]
    rsi_level_achieved: int
    rsi_converged: bool
    banach_L: float = 1.0
    eta_history: Tuple[float, ...] = ()
    delta_tau_history: Tuple[float, ...] = ()
    curvature_history: Tuple[float, ...] = ()
    dominant_frequencies: Tuple[float, ...] = ()
    lyapunov_spectrum: Tuple[float, ...] = ()
    action_angle: Tuple[float, float] = (0.0, 0.0)
    quadratic_rate: float = 0.0
    resonance_vectors: Tuple[Tuple[int, ...], ...] = ()
    method_lineage: Tuple[str, ...] = ()
    monodromy_eigs: Tuple[complex, ...] = ()
    lindstedt_period: float = 0.0
    recurrence_bound: float = 0.0
    banach_certified: bool = False
    generating_function_residual: float = 0.0
    lyapunov_pairing_defect: float = 0.0
    projector_mode: str = ProjectorMode.SPECTRAL_HIGHAM.value


def _fro(matrix: NDArray[np.float64]) -> float:
    return float(np.linalg.norm(np.asarray(matrix, dtype=np.float64), ord="fro"))


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    d = abs(float(den))
    if d < _WILKINSON_EPS:
        return float(default)
    return float(num) / d


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 1 — TEJIDO GEOMÉTRICO Y FLUJO DE RICCI RSI¹ (BASE)                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class MetricTensorsAgent:
    r"""
    Soberano de Calibre Riemanniano. Las tres fases son un compuesto categórico
    $\mathrm{Auditar}=\Phi_3\circ\Phi_2\circ\Phi_1$ con morfismos de continuación
    `Phase1Handoff` (fase 1 → 2) y `Phase2Handoff` (fase 2 → 3).

    La última definición formal de cada fase es el *handoff* que germina la siguiente:
    $\Phi_2\circ\Phi_1 = \texttt{\_ingest\_phase1\_handoff}\circ\texttt{\_handoff\_to\_phase2}$,
    $\Phi_3\circ\Phi_2 = \texttt{\_ingest\_phase2\_handoff}\circ\texttt{\_handoff\_to\_phase3}$.
    """

    def __init__(
        self,
        condition_ceiling: float = _MAX_CONDITION_NUMBER,
        rsi3_max_iterations: int = 50,
        poincare_energy_tolerance: float = 1.0e-3,
        lipschitz_probe_stride: int = 5,
        benettin_qr_stride: int = 10,
        projector_mode: ProjectorMode = ProjectorMode.SPECTRAL_HIGHAM,
        ricci_discretizer: RicciDiscretizer = RicciDiscretizer.KOSZUL_QUADRATIC,
    ) -> None:
        self._condition_ceiling = float(condition_ceiling)
        self._rsi3_max_iterations = int(rsi3_max_iterations)
        self._poincare_tol = float(poincare_energy_tolerance)
        self._lipschitz_probe_stride = max(int(lipschitz_probe_stride), 1)
        self._benettin_qr_stride = max(int(benettin_qr_stride), 1)
        self._projector_mode = projector_mode
        self._ricci_discretizer = ricci_discretizer
        self._rng = np.random.default_rng(0)

    # ── FASE 1.0 — Validación, Cholesky SPD y log-volumen ─────────────────
    def _validate_square(self, matrix: NDArray[np.float64], name: str) -> NDArray[np.float64]:
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise MetricTensorGovernanceError(
                f"{name} debe ser cuadrada; recibido shape={arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise MetricTensorGovernanceError(f"{name} contiene valores no finitos")
        return arr

    def _spd_inverse(self, G_metric: NDArray[np.float64]) -> NDArray[np.float64]:
        n = G_metric.shape[0]
        shifted = G_metric + _SPECTRAL_PSD_FLOOR * np.eye(n, dtype=np.float64)
        chol, lower = la.cho_factor(shifted, lower=True, check_finite=False)
        return la.cho_solve((chol, lower), np.eye(n, dtype=np.float64), check_finite=False)

    def _spd_logdet(self, G_metric: NDArray[np.float64]) -> float:
        n = G_metric.shape[0]
        shifted = G_metric + _SPECTRAL_PSD_FLOOR * np.eye(n, dtype=np.float64)
        chol, _ = la.cho_factor(shifted, lower=True, check_finite=False)
        diag = np.abs(np.diag(chol))
        return float(2.0 * np.sum(np.log(np.maximum(diag, _WILKINSON_EPS))))

    def _volume_form_consistency(self, G_metric: NDArray[np.float64]) -> float:
        r"""
        Consistencia de la forma de volumen riemanniana:
        $\bigl|\det G - \exp(\mathrm{logdet}_{\mathrm{Chol}} G)\bigr|/|\det G|$.
        Discrepancia nula ⇔ Cholesky y $\det$ coinciden (SPD numéricamente sano).
        """
        try:
            logdet = self._spd_logdet(G_metric)
            det_chol = float(np.exp(np.clip(logdet, -700.0, 700.0)))
            sign, logabs = np.linalg.slogdet(G_metric)
            if sign <= 0.0:
                return float("inf")
            det_ref = float(np.exp(np.clip(logabs, -700.0, 700.0)))
            return _safe_div(abs(det_chol - det_ref), max(det_ref, _WILKINSON_EPS))
        except (la.LinAlgError, ValueError):
            return float("inf")

    # ── FASE 1.A — Descomposición espectral PSD (Higham) ──────────────────
    def _spectral_decomposition_psd(
        self, G_metric: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], SpectralProfile]:
        G_sym = 0.5 * (G_metric + G_metric.T)
        eigenvalues, eigenvectors = la.eigh(G_sym, check_finite=False)
        clipped = np.maximum(eigenvalues, _SPECTRAL_PSD_FLOOR)
        G_proj = (eigenvectors * clipped) @ eigenvectors.T
        G_proj = 0.5 * (G_proj + G_proj.T)
        lam_min, lam_max = float(clipped[0]), float(clipped[-1])
        cond = lam_max / max(lam_min, _WILKINSON_EPS)
        profile = SpectralProfile(lam_min, lam_max, cond, is_spd=True)
        return G_proj, eigenvalues, eigenvectors, profile

    def _project_psd_cayley_darboux(self, G_metric: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Proyección SPD vía transformada de Cayley en el modelo de Darboux de
        $\mathrm{Sp}(n,\mathbb{R})$ cuando $\dim G$ es par; fallback Higham si no.
        """
        G_arr = np.asarray(G_metric, dtype=np.float64)
        n2 = int(G_arr.shape[0])
        if n2 % 2 != 0 or n2 == 0:
            G_proj, _, _, _ = self._spectral_decomposition_psd(G_arr)
            return G_proj
        n = n2 // 2
        I_mat = np.eye(n2, dtype=np.float64)
        J_mat = np.block([
            [np.zeros((n, n), dtype=np.float64), np.eye(n, dtype=np.float64)],
            [-np.eye(n, dtype=np.float64), np.zeros((n, n), dtype=np.float64)],
        ])
        G_shifted = G_arr + _SPECTRAL_PSD_FLOOR * I_mat
        try:
            K = (G_shifted - I_mat) @ self._spd_inverse(G_shifted + I_mat)
        except (la.LinAlgError, ValueError):
            G_proj, _, _, _ = self._spectral_decomposition_psd(G_arr)
            return G_proj
        K_sp = 0.5 * (K + J_mat @ K.T @ J_mat)
        try:
            eigvals, eigvecs = la.eigh(K_sp, check_finite=False)
        except la.LinAlgError:
            G_proj, _, _, _ = self._spectral_decomposition_psd(G_arr)
            return G_proj
        clipped_eigvals = np.clip(eigvals, -0.999999, 0.999999)
        K_proj = (eigvecs * clipped_eigvals) @ eigvecs.T
        try:
            G_sympl = (I_mat + K_proj) @ self._spd_inverse(I_mat - K_proj)
        except la.LinAlgError:
            G_proj, _, _, _ = self._spectral_decomposition_psd(G_arr)
            return G_proj
        return 0.5 * (G_sympl + G_sympl.T)

    def _select_metric_projection(
        self,
        G_metric: NDArray[np.float64],
        mode: Optional[ProjectorMode] = None,
    ) -> NDArray[np.float64]:
        r"""Primitiva de proyección: germen del selector de método RSI³ (fase 2.D′)."""
        chosen = mode if mode is not None else self._projector_mode
        if chosen is ProjectorMode.CAYLEY_DARBOUX:
            return self._project_psd_cayley_darboux(G_metric)
        G_proj, _, _, _ = self._spectral_decomposition_psd(G_metric)
        return G_proj

    # ── FASE 1.B — Derivación de Connes y Christoffel de Koszul ──────────
    def _connes_metric_derivatives(self, G_metric: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""$[X_\mu, G]_{\nu\sigma} = \delta_{\nu\mu} G_{\mu\sigma} - G_{\nu\mu}\delta_{\mu\sigma}$."""
        n = G_metric.shape[0]
        dG = np.zeros((n, n, n), dtype=np.float64)
        for mu in range(n):
            dG[mu, mu, :] += G_metric[mu, :]
            dG[mu, :, mu] -= G_metric[:, mu]
        return dG

    def _synthesize_christoffel_from_metric(self, G_metric: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Símbolos de Koszul $\Gamma^{r}{}_{mn}=\tfrac12 G^{rs}(\partial_m G_{ns}+\partial_n G_{ms}-\partial_s G_{mn})$,
        simetrizados en $(m,n)$ (torsión nula; unicidad del teorema fundamental).
        """
        G_inv = self._spd_inverse(G_metric)
        dG = self._connes_metric_derivatives(G_metric)
        term1 = np.einsum("rs,mns->rmn", G_inv, dG, optimize=True)
        term2 = np.einsum("rs,nms->rmn", G_inv, dG, optimize=True)
        term3 = np.einsum("rs,smn->rmn", G_inv, dG, optimize=True)
        gamma = 0.5 * (term1 + term2 - term3)
        gamma = 0.5 * (gamma + np.transpose(gamma, (0, 2, 1)))
        return gamma

    def _metric_compatibility_residual(
        self,
        G_metric: NDArray[np.float64],
        christoffel: NDArray[np.float64],
        dG: Optional[NDArray[np.float64]] = None,
    ) -> float:
        r"""$(\nabla_\rho G)_{\mu\nu}=\partial_\rho G_{\mu\nu}-\Gamma^\lambda_{\rho\mu}G_{\lambda\nu}-\Gamma^\lambda_{\rho\nu}G_{\mu\lambda}$."""
        if dG is None:
            dG = self._connes_metric_derivatives(G_metric)
        term_a = np.einsum("lrm,ln->rmn", christoffel, G_metric, optimize=True)
        term_b = np.einsum("lrn,ml->rmn", christoffel, G_metric, optimize=True)
        compat = dG - term_a - term_b
        return _fro(compat)

    # ── FASE 1.B′ — Identidad de Bianchi contraída ∇_μ Ein^{μν} = 0 ───────
    def _bianchi_identity_residual(
        self,
        G_metric: NDArray[np.float64],
        christoffel: NDArray[np.float64],
        einstein_low: NDArray[np.float64],
    ) -> float:
        r"""
        Residual de la identidad de Bianchi contraída en carta normal 0-jet:
        $(\nabla_\mu\mathrm{Ein})^{\mu\nu}\approx\Gamma^\mu_{\mu\lambda}\mathrm{Ein}^{\lambda\nu}
         +\Gamma^\nu_{\mu\lambda}\mathrm{Ein}^{\mu\lambda}$.
        Indexación canónica: $\Gamma[r,m,n]=\Gamma^{r}{}_{mn}$.
        """
        G_inv = self._spd_inverse(G_metric)
        einstein_up = G_inv @ einstein_low @ G_inv
        # Γ^μ_{μλ}  →  einsum("mml->l")
        trace_gamma = np.einsum("mml->l", christoffel)
        term_a = np.einsum("l,ln->n", trace_gamma, einstein_up, optimize=True)
        term_b = np.einsum("nml,ml->n", christoffel, einstein_up, optimize=True)
        divergence = term_a + term_b
        return float(np.linalg.norm(divergence, ord=2))

    # ── FASE 1.B″ — Weyl conforme, Killing–Noether, Kirchhoff–Bianchi ─────
    def _weyl_conformal_residual(
        self,
        G_metric: NDArray[np.float64],
        christoffel: NDArray[np.float64],
        ricci: NDArray[np.float64],
        ricci_scalar: float,
    ) -> float:
        r"""
        Residual del tensor de Weyl 0-jet (parte conforme de Riemann).
        Riemann 0-jet: $R^{\rho}{}_{\sigma\mu\nu}=\Gamma^{\rho}{}_{\nu\lambda}\Gamma^{\lambda}{}_{\mu\sigma}
        -\Gamma^{\rho}{}_{\mu\lambda}\Gamma^{\lambda}{}_{\nu\sigma}$.
        Weyl se anula idénticamente en $n\le 2$; en $n=3$ Weyl ≡ 0 y el residual
        mide la desviación Cotton-like de la discretización.
        """
        n = int(G_metric.shape[0])
        if n <= 2:
            return 0.0
        try:
            # R^ρ_{σμν}  (n, n, n, n)
            riemann = (
                np.einsum("rvl,lms->rsmv", christoffel, christoffel, optimize=True)
                - np.einsum("rml,lvs->rsmv", christoffel, christoffel, optimize=True)
            )
            # Baja el primer índice: R_{α σ μ ν} = G_{αρ} R^ρ_{σμν}
            riemann_low = np.einsum("ap,psmv->asmv", G_metric, riemann, optimize=True)
            # Kulkarni–Nomizu: (P ☆ G)_{αβμν}
            # P = Ric − R G / (2(n−1))
            denom = 2.0 * float(n - 1)
            P = ricci - (ricci_scalar / denom) * G_metric
            # (P ☆ G)_{αβμν} = P_{αμ} G_{βν} − P_{αν} G_{βμ} − P_{βμ} G_{αν} + P_{βν} G_{αμ}
            kn = (
                np.einsum("am,bn->abmn", P, G_metric)
                - np.einsum("an,bm->abmn", P, G_metric)
                - np.einsum("bm,an->abmn", P, G_metric)
                + np.einsum("bn,am->abmn", P, G_metric)
            )
            # En n ≥ 3, Weyl_low = Riemann_low − kn / (n−2)
            weyl = riemann_low - kn / float(n - 2)
            return float(np.linalg.norm(weyl))
        except (la.LinAlgError, ValueError, np.core._exceptions._ArrayMemoryError):  # type: ignore[attr-defined]
            return float("inf")
        except Exception:  # noqa: BLE001 — acota memoria en n grande
            return float("inf")

    def _killing_equation_residual(
        self,
        G_metric: NDArray[np.float64],
        christoffel: NDArray[np.float64],
    ) -> float:
        r"""
        Residual de la ecuación de Killing para el marco coordenado $\xi_{(\alpha)}^{\mu}=\delta^{\mu}_{\alpha}$:
        $(\mathcal{L}_{\xi}G)_{\mu\nu}=\nabla_{\mu}\xi_{\nu}+\nabla_{\nu}\xi_{\mu}
        =-(\Gamma^{\lambda}{}_{\mu\nu}+\Gamma^{\lambda}{}_{\nu\mu})\xi_{\lambda}$.
        Con $\xi_{\lambda}=G_{\lambda\alpha}$ (bajar índice del marco), el residual
        cuantifica la ausencia de isometrías infinitesimales (Noether).
        """
        n = int(G_metric.shape[0])
        try:
            # ξ_λ^{(α)} = G_{λα}
            xi_low = G_metric  # (λ, α)
            # ∇_μ ξ_ν^{(α)} = − Γ^λ_{μν} ξ_λ^{(α)}   (ξ constante en carta)
            nabla_xi = -np.einsum("lm n,la->mna".replace(" ", ""), christoffel, xi_low)
            # Indexación: christoffel[l, m, n] = Γ^l_{mn} → "lmn,la->mna"
            nabla_xi = -np.einsum("lmn,la->mna", christoffel, xi_low, optimize=True)
            lie = nabla_xi + np.transpose(nabla_xi, (1, 0, 2))
            return float(np.linalg.norm(lie)) / float(max(n, 1))
        except (la.LinAlgError, ValueError):
            return float("inf")

    def _kirchhoff_current_residual(
        self,
        G_metric: NDArray[np.float64],
        einstein_low: NDArray[np.float64],
    ) -> float:
        r"""
        Dual circuito–geométrico (Kirchhoff Current Law $\equiv$ Bianchi contraída):
        $I^{\mu}=\sum_{\nu}\mathrm{Ein}^{\mu\nu}$.  KCL exige $I^{\mu}=0$ en cada nodo.
        KVL está ya cifrada en $\mathrm{Ein}=\mathrm{Ein}^{T}$ (simetría 0-jet).
        """
        try:
            G_inv = self._spd_inverse(G_metric)
            einstein_up = G_inv @ einstein_low @ G_inv
            currents = np.sum(einstein_up, axis=1)
            kcl = float(np.linalg.norm(currents, ord=2))
            kvl = _fro(einstein_low - einstein_low.T)
            return float(kcl + kvl)
        except (la.LinAlgError, ValueError):
            return float("inf")

    # ── FASE 1.C — Ricci, Einstein, estado geométrico 0-jet ───────────────
    def compute_ricci_tensor_discretized(
        self,
        G_metric: NDArray[np.float64],
        christoffel: Optional[NDArray[np.float64]] = None,
        discretizer: Optional[RicciDiscretizer] = None,
    ) -> Tuple[NDArray[np.float64], float]:
        n = G_metric.shape[0]
        mode = discretizer if discretizer is not None else self._ricci_discretizer
        if christoffel is None and mode is RicciDiscretizer.KOSZUL_QUADRATIC:
            try:
                christoffel = self._synthesize_christoffel_from_metric(G_metric)
            except la.LinAlgError:
                christoffel = None
        if christoffel is not None and mode is RicciDiscretizer.KOSZUL_QUADRATIC:
            gamma = np.asarray(christoffel, dtype=np.float64)
            trace_gamma = np.einsum("srs->r", gamma)
            quad1 = np.einsum("rmn,r->mn", gamma, trace_gamma, optimize=True)
            quad2 = np.einsum("smp,pns->mn", gamma, gamma, optimize=True)
            ricci = 0.5 * ((quad1 - quad2) + (quad1 - quad2).T)
            G_inv = self._spd_inverse(G_metric)
            ricci_scalar = float(np.trace(G_inv @ ricci))
            return ricci, ricci_scalar
        tr_G = float(np.trace(G_metric))
        ricci_proxy = G_metric - (tr_G / float(n)) * np.eye(n, dtype=np.float64)
        G_inv = self._spd_inverse(G_metric)
        ricci_scalar = float(np.trace(G_inv @ ricci_proxy))
        return ricci_proxy, ricci_scalar

    def _evaluate_geometry(self, G_metric: NDArray[np.float64]) -> GeometricState:
        G_proj, _, _, profile = self._spectral_decomposition_psd(G_metric)
        n = G_proj.shape[0]
        try:
            gamma = self._synthesize_christoffel_from_metric(G_proj)
        except la.LinAlgError:
            gamma = np.zeros((n, n, n), dtype=np.float64)
        ricci, ricci_scalar = self.compute_ricci_tensor_discretized(G_proj, gamma)
        einstein = ricci - 0.5 * ricci_scalar * G_proj
        log_volume = self._spd_logdet(G_proj)
        eh_action = float(ricci_scalar * np.exp(np.clip(log_volume, -700.0, 700.0)))
        try:
            compat = self._metric_compatibility_residual(G_proj, gamma)
        except la.LinAlgError:
            compat = float("inf")
        try:
            bianchi = self._bianchi_identity_residual(G_proj, gamma, einstein)
        except la.LinAlgError:
            bianchi = float("inf")
        weyl = self._weyl_conformal_residual(G_proj, gamma, ricci, float(ricci_scalar))
        killing = self._killing_equation_residual(G_proj, gamma)
        kirchhoff = self._kirchhoff_current_residual(G_proj, einstein)
        vol_cons = self._volume_form_consistency(G_proj)
        return GeometricState(
            metric=G_proj, christoffel=gamma, ricci_tensor=ricci,
            ricci_scalar=float(ricci_scalar), einstein_tensor=einstein,
            spectral_profile=profile, log_volume=float(log_volume),
            eh_action=eh_action, compatibility_residual=float(compat),
            bianchi_residual=float(bianchi),
            weyl_residual=float(weyl),
            killing_residual=float(killing),
            kirchhoff_residual=float(kirchhoff),
            volume_consistency=float(vol_cons),
        )

    # ── FASE 1.D — Paso base del flujo de Ricci RSI¹ ──────────────────────
    def apply_ricci_flow_rsi3(
        self,
        G_metric: NDArray[np.float64],
        delta_tau: float = 0.01,
        projector_mode: Optional[ProjectorMode] = None,
        discretizer: Optional[RicciDiscretizer] = None,
    ) -> NDArray[np.float64]:
        n = G_metric.shape[0]
        ricci_tensor, ricci_scalar = self.compute_ricci_tensor_discretized(
            G_metric, discretizer=discretizer
        )
        traceless = ricci_tensor - (ricci_scalar / float(n)) * G_metric
        G_next = G_metric - 2.0 * float(delta_tau) * traceless
        return self._select_metric_projection(G_next, mode=projector_mode)

    def _ricci_flow_operator(
        self, delta_tau: float
    ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        return lambda G: self.apply_ricci_flow_rsi3(G, delta_tau)

    # ── FASE 1.E′ — Auditoría de cantidades conservadas ───────────────────
    def _conserved_quantities_audit(
        self,
        ricci_scalars: List[float],
        log_volumes: List[float],
        eh_actions: List[float],
    ) -> Dict[str, float]:
        r"""Cuantifica el *drift* relativo de $R$, $\log\mathrm{Vol}$, $S_{EH}$ en la órbita."""
        def _drift(xs: List[float]) -> float:
            if not xs:
                return 0.0
            a = np.asarray(xs, dtype=np.float64)
            a = a[np.isfinite(a)]
            if a.size == 0:
                return 0.0
            denom = max(abs(float(np.mean(a))), _WILKINSON_EPS)
            return float((float(np.max(a)) - float(np.min(a))) / denom)
        return {
            "ricci_scalar_drift": _drift(ricci_scalars),
            "log_volume_drift":  _drift(log_volumes),
            "eh_action_drift":    _drift(eh_actions),
        }

    # ── FASE 1.E″ — Primeras integrales de Poincaré (mecánica celeste) ───
    def _poincare_first_integrals(
        self,
        ricci_scalars: Sequence[float],
        log_volumes: Sequence[float],
        delta_tau: float,
        orbit: Sequence[NDArray[np.float64]],
    ) -> Dict[str, float]:
        r"""
        Primeras integrales (análogos celestes sobre el flujo de Ricci):

          • $H$          : hamiltoniano-proxy $= \mathrm{median}(R)$  (superficie de energía).
          • $C_{\mathrm{Jacobi}}$ : $C=\mathrm{median}(\log\mathrm{Vol}-\tau R)$
            (integral de Jacobi del problema restringido: $C=2\Omega-v^{2}$).
          • $L_{\mathrm{ang}}$ : momento angular-proxy $= \|G-\mathrm{diag}(G)\|_F$
            (contenido fuera de eje ≡ “rotación” residual del marco).
        """
        R = np.asarray(list(ricci_scalars), dtype=np.float64)
        lv = np.asarray(list(log_volumes), dtype=np.float64)
        n_min = int(min(R.size, lv.size))
        if n_min == 0:
            return {"H": 0.0, "C_jacobi": 0.0, "L_angular": 0.0}
        R = R[:n_min]
        lv = lv[:n_min]
        tau = float(delta_tau) * np.arange(n_min, dtype=np.float64)
        H = float(np.median(R))
        C = float(np.median(lv - tau * R))
        if orbit:
            G = np.asarray(orbit[-1], dtype=np.float64)
            L = _fro(G - np.diag(np.diag(G)))
        else:
            L = 0.0
        return {"H": H, "C_jacobi": C, "L_angular": float(L)}

    # ── FASE 1.F — Calentamiento RSI¹ y recolección de la órbita ─────────
    def _warmup_rsi1_orbit(
        self,
        G_metric: NDArray[np.float64],
        n_warmup_steps: int,
        delta_tau: float,
    ) -> Phase1Handoff:
        orbit: List[NDArray[np.float64]] = [np.array(G_metric, copy=True, dtype=np.float64)]
        ricci_scalars: List[float] = []
        condition_numbers: List[float] = []
        log_volumes: List[float] = []
        eh_actions: List[float] = []
        bianchi_residuals: List[float] = []
        weyl_residuals: List[float] = []
        killing_residuals: List[float] = []
        kirchhoff_residuals: List[float] = []
        volume_consistencies: List[float] = []
        G = np.array(G_metric, copy=True, dtype=np.float64)

        n_steps = max(int(n_warmup_steps), 0)
        for _ in range(n_steps):
            state = self._evaluate_geometry(G)
            ricci_scalars.append(state.ricci_scalar)
            condition_numbers.append(state.spectral_profile.condition_number)
            log_volumes.append(state.log_volume)
            eh_actions.append(state.eh_action)
            bianchi_residuals.append(state.bianchi_residual)
            weyl_residuals.append(state.weyl_residual)
            killing_residuals.append(state.killing_residual)
            kirchhoff_residuals.append(state.kirchhoff_residual)
            volume_consistencies.append(state.volume_consistency)
            G = self.apply_ricci_flow_rsi3(G, delta_tau)
            orbit.append(np.array(G, copy=True))

        state_final = self._evaluate_geometry(G)
        ricci_scalars.append(state_final.ricci_scalar)
        condition_numbers.append(state_final.spectral_profile.condition_number)
        log_volumes.append(state_final.log_volume)
        eh_actions.append(state_final.eh_action)
        bianchi_residuals.append(state_final.bianchi_residual)
        weyl_residuals.append(state_final.weyl_residual)
        killing_residuals.append(state_final.killing_residual)
        kirchhoff_residuals.append(state_final.kirchhoff_residual)
        volume_consistencies.append(state_final.volume_consistency)

        audit = self._conserved_quantities_audit(ricci_scalars, log_volumes, eh_actions)
        first_int = self._poincare_first_integrals(
            ricci_scalars, log_volumes, delta_tau, orbit
        )

        return Phase1Handoff(
            initial_metric=np.array(G_metric, copy=True, dtype=np.float64),
            orbit=orbit, ricci_scalars=ricci_scalars,
            condition_numbers=condition_numbers, delta_tau=float(delta_tau),
            log_volumes=log_volumes, eh_actions=eh_actions,
            bianchi_residuals=bianchi_residuals,
            conserved_quantities_audit=audit,
            weyl_residuals=weyl_residuals,
            killing_residuals=killing_residuals,
            kirchhoff_residuals=kirchhoff_residuals,
            first_integrals=first_int,
            volume_consistencies=volume_consistencies,
            projector_mode=self._projector_mode.value,
        )

    # ── FASE 1.Z — HANDOFF: última definición de FASE 1 ≡ inicio de FASE 2
    def _handoff_to_phase2(
        self,
        G_metric: NDArray[np.float64],
        n_warmup_steps: int = 5,
        delta_tau: float = 0.01,
    ) -> Phase1Handoff:
        r"""
        Última definición formal de FASE 1 y continuación estricta de FASE 2.

        $\Phi_2\circ\Phi_1(G)=\texttt{\_ingest\_phase1\_handoff}(\texttt{\_handoff\_to\_phase2}(G))$.

        El germen geométrico (órbita RSI¹ + primeras integrales de Poincaré +
        residuales Bianchi/Weyl/Killing/Kirchhoff) es el objeto de ingesta de
        `_ingest_phase1_handoff` (FASE 2.A), que abre el anidamiento RSI³.
        """
        handoff = self._warmup_rsi1_orbit(G_metric, n_warmup_steps, delta_tau)
        logger.info(
            "[FASE1→FASE2] Órbita RSI¹: %d estados, R_final=%.3e, κ_final=%.3e, "
            "Bianchi=%.3e, Weyl=%.3e, Killing=%.3e, KCL=%.3e, H=%.3e, C_J=%.3e",
            len(handoff.orbit),
            handoff.ricci_scalars[-1],
            handoff.condition_numbers[-1],
            handoff.bianchi_residuals[-1] if handoff.bianchi_residuals else float("nan"),
            handoff.weyl_residuals[-1] if handoff.weyl_residuals else float("nan"),
            handoff.killing_residuals[-1] if handoff.killing_residuals else float("nan"),
            handoff.kirchhoff_residuals[-1] if handoff.kirchhoff_residuals else float("nan"),
            handoff.first_integrals.get("H", float("nan")),
            handoff.first_integrals.get("C_jacobi", float("nan")),
        )
        return handoff

    # ╔════════════════════════════════════════════════════════════════════════╗
    # ║  FASE 2 — AUTOMEJORA RECURSIVA RSI³ Y MAPA DE POINCARÉ                 ║
    # ║  (inicia exactamente donde termina FASE 1.Z)                           ║
    # ╚════════════════════════════════════════════════════════════════════════╝

    # ── FASE 2.A — INGESTA del handoff de FASE 1 (continuación directa) ──
    def _ingest_phase1_handoff(
        self, handoff: Phase1Handoff
    ) -> Tuple[NDArray[np.float64], List[NDArray[np.float64]], List[float]]:
        r"""Consume el `Phase1Handoff` de `_handoff_to_phase2` (continuación 1→2)."""
        if not handoff.orbit:
            raise MetricTensorGovernanceError("Phase1Handoff.orbit vacío")
        G_seed = np.array(handoff.orbit[-1], copy=True, dtype=np.float64)
        return G_seed, handoff.orbit, list(handoff.ricci_scalars)

    # ── FASE 2.B — Nivel 1: paso Ricci (unidad atómica) ───────────────────
    def _rsi_level1_step(
        self,
        G: NDArray[np.float64],
        delta_tau: float,
        projector_mode: Optional[ProjectorMode] = None,
        discretizer: Optional[RicciDiscretizer] = None,
    ) -> NDArray[np.float64]:
        r"""RSI¹: $\Phi^{(1)}_{\Delta\tau}(G)$ — un paso de Ricci normalizado."""
        return self.apply_ricci_flow_rsi3(
            G, delta_tau, projector_mode=projector_mode, discretizer=discretizer
        )

    # ── FASE 2.C — Nivel 2: metamejora de Δτ por gradiente de curvatura ──
    def _rsi_level2_adaptive_tau(
        self,
        G: NDArray[np.float64],
        delta_tau: float,
        eta: float,
        curvature_history: List[float],
        projector_mode: Optional[ProjectorMode] = None,
        discretizer: Optional[RicciDiscretizer] = None,
    ) -> Tuple[NDArray[np.float64], float]:
        r"""RSI²: $\Delta\tau_{k+1}=\mathrm{sat}(\Delta\tau_k\exp(-\eta|\nabla R_k|))$ con saturación CFL."""
        if len(curvature_history) >= 2:
            grad_R = abs(curvature_history[-1] - curvature_history[-2])
            delta_tau_new = float(
                np.clip(delta_tau * np.exp(-eta * grad_R), _TAU_BOUNDS[0], _TAU_BOUNDS[1])
            )
        else:
            delta_tau_new = float(delta_tau)
        G_next = self._rsi_level1_step(
            G, delta_tau_new, projector_mode=projector_mode, discretizer=discretizer
        )
        return G_next, delta_tau_new

    # ── FASE 2.D — Nivel 3: metamejora de η por contracción de Banach ────
    def _rsi_level3_adaptive_eta(
        self,
        G: NDArray[np.float64],
        delta_tau: float,
        eta: float,
        zeta: float,
        contraction_history: List[float],
        curvature_history: List[float],
        projector_mode: Optional[ProjectorMode] = None,
        discretizer: Optional[RicciDiscretizer] = None,
    ) -> Tuple[NDArray[np.float64], float, float]:
        r"""RSI³ (parámetros): $\eta_{k+1}=\mathrm{sat}(\eta_k\exp(\zeta(L_k-L_\star)))$, $L_\star=0.95$."""
        if len(contraction_history) >= 1:
            L_recent = float(contraction_history[-1])
            eta_new = float(np.clip(
                eta * np.exp(zeta * (L_recent - _BANACH_TARGET)),
                _ETA_BOUNDS[0], _ETA_BOUNDS[1],
            ))
        else:
            eta_new = float(eta)
        G_next, delta_tau_new = self._rsi_level2_adaptive_tau(
            G, delta_tau, eta_new, curvature_history,
            projector_mode=projector_mode, discretizer=discretizer,
        )
        return G_next, delta_tau_new, eta_new

    def _adapt_zeta_meta_meta(self, zeta: float, eta_history: List[float]) -> float:
        r"""Meta-meta: $\zeta$ se atenúa si $\eta$ oscila y se refuerza si $\eta$ estanca."""
        if len(eta_history) < 4:
            return float(np.clip(zeta, _ZETA_BOUNDS[0], _ZETA_BOUNDS[1]))
        diffs = np.diff(np.asarray(eta_history[-4:], dtype=np.float64))
        osc = float(np.std(diffs))
        eta_abs = max(abs(float(eta_history[-1])), 1.0e-12)
        if osc > 0.5 * eta_abs:
            zeta_new = 0.5 * zeta
        elif osc < 1.0e-6:
            zeta_new = 1.1 * zeta
        else:
            zeta_new = zeta
        return float(np.clip(zeta_new, _ZETA_BOUNDS[0], _ZETA_BOUNDS[1]))

    # ── FASE 2.D′ — Automejora de MÉTODO (RSI³ auténtica) ─────────────────
    def _rsi3_method_selector(
        self,
        state: GeometricState,
        current_projector: ProjectorMode,
        current_discretizer: RicciDiscretizer,
    ) -> Tuple[ProjectorMode, RicciDiscretizer, str]:
        r"""
        Automejora recursiva de *método* (nivel 3 auténtico, no sólo de parámetros):

          • Si $\kappa_2(G)>\kappa_\star/10$ → Higham espectral (más estable).
          • Si $\dim$ par y $\kappa_2$ moderado → Cayley–Darboux (respeta Sp).
          • Si residual de Bianchi relativo $>\texttt{\_BIANCHI\_REL\_TOL}$ → Koszul
            cuadrático (Christoffel); si Bianchi es sano y $\kappa_2$ enorme →
            proxy libre de traza (más barato, menos geométrico).

        Histéresis `_METHOD_SWITCH_HYSTERESIS` evita chattering del selector.
        Devuelve (proyector, discretizador, etiqueta de linaje).
        """
        cond = float(state.spectral_profile.condition_number)
        n = int(state.metric.shape[0])
        ein_scale = max(_fro(state.einstein_tensor), _WILKINSON_EPS)
        bianchi_rel = float(state.bianchi_residual) / ein_scale

        projector = current_projector
        if cond > self._condition_ceiling / 10.0:
            projector = ProjectorMode.SPECTRAL_HIGHAM
        elif n % 2 == 0 and cond < (self._condition_ceiling / 10.0) / _METHOD_SWITCH_HYSTERESIS:
            projector = ProjectorMode.CAYLEY_DARBOUX

        discretizer = current_discretizer
        if bianchi_rel > _BIANCHI_REL_TOL:
            discretizer = RicciDiscretizer.KOSZUL_QUADRATIC
        elif (
            bianchi_rel < _BIANCHI_REL_TOL / _METHOD_SWITCH_HYSTERESIS
            and cond > self._condition_ceiling / 2.0
        ):
            discretizer = RicciDiscretizer.TRACEFREE_PROXY

        tag = f"{projector.value}+{discretizer.value}"
        return projector, discretizer, tag

    # ── FASE 2.E — Orden de convergencia de RSI³ ─────────────────────────
    def _quadratic_convergence_rate(self, orbit: List[NDArray[np.float64]]) -> float:
        r"""
        Estima el orden $p$ vía $|e_{k+1}|\approx C|e_k|^p$, $e_k=\|G_k-G_{k-1}\|_F$:
        $p\approx\log(e_{k+1}/e_k)/\log(e_k/e_{k-1})$.
        Sólo se aceptan tripletas *monótonas decrecientes* (evita basura estadística).
        """
        if len(orbit) < 4:
            return 0.0
        e = [_fro(orbit[k + 1] - orbit[k]) for k in range(len(orbit) - 1)]
        e = [x for x in e if x > _RSI3_CONV_EPS]
        if len(e) < 3:
            return 0.0
        orders: List[float] = []
        for k in range(1, len(e) - 1):
            if not (e[k + 1] < e[k] < e[k - 1]):
                continue
            num = np.log(max(e[k + 1] / e[k], _WILKINSON_EPS))
            den = np.log(max(e[k] / e[k - 1], _WILKINSON_EPS))
            if abs(den) < 1.0e-9:
                continue
            orders.append(float(num / den))
        if not orders:
            return 0.0
        return float(np.clip(np.median(orders), 0.0, 4.0))

    # ── FASE 2.F — Constante de Banach, certificado y sonda de Lipschitz ─
    def _banach_contraction_constant(self, orbit: List[NDArray[np.float64]]) -> float:
        if len(orbit) < 3:
            return 1.0
        diffs = [_fro(orbit[k + 1] - orbit[k]) for k in range(len(orbit) - 1)]
        ratios = [diffs[k + 1] / max(diffs[k], 1.0e-15) for k in range(len(diffs) - 1)]
        if not ratios:
            return 1.0
        return float(np.max(np.clip(np.asarray(ratios, dtype=np.float64), 0.0, 1.0e6)))

    def _banach_fixed_point_certificate(
        self,
        orbit: List[NDArray[np.float64]],
        lipschitz: float,
    ) -> BanachCertificate:
        r"""
        Certificado del teorema del punto fijo de Banach:

          (i)   $X=\overline{\mathrm{SPD}}_n$ es completo en $\|\cdot\|_F$ (proyección Higham
                lo hace cerrado; $\mathbb{R}^{n\times n}$ es Banach).
          (ii)  $\Phi(X)\subset X$  (todo $G_k$ es SPD por construcción del proyector).
          (iii) $\mathrm{Lip}\,\Phi\le L_\star=0.95<1$.

        Si (i)∧(ii)∧(iii), existe único punto fijo y la órbita converge a él.
        """
        is_complete = True
        is_self_map = True
        for G in orbit:
            if not np.all(np.isfinite(G)):
                is_self_map = False
                break
            try:
                eig = la.eigvalsh(0.5 * (G + G.T), check_finite=False)
                if float(np.min(eig)) < -10.0 * _SPECTRAL_PSD_FLOOR:
                    is_self_map = False
                    break
            except la.LinAlgError:
                is_self_map = False
                break
        is_contraction = bool(lipschitz < _BANACH_TARGET)
        certified = bool(is_complete and is_self_map and is_contraction and len(orbit) >= 3)
        return BanachCertificate(
            lipschitz=float(lipschitz),
            is_self_map=is_self_map,
            is_complete_domain=is_complete,
            is_contraction=is_contraction,
            certified=certified,
        )

    def _lipschitz_phi_empirical(
        self, G: NDArray[np.float64], delta_tau: float, eta: float,
        curvature_history: List[float],
        projector_mode: Optional[ProjectorMode] = None,
        discretizer: Optional[RicciDiscretizer] = None,
    ) -> float:
        n = G.shape[0]
        ricci, ricci_scalar = self.compute_ricci_tensor_discretized(G, discretizer=discretizer)
        V = ricci - (ricci_scalar / float(n)) * G
        vnorm = _fro(V)
        if vnorm < 1.0e-15:
            V = np.eye(n, dtype=np.float64)
            vnorm = float(np.sqrt(n))
        eps = _LIPSCHITZ_PERT_REL * max(_fro(G), 1.0)
        H = (eps / vnorm) * V
        G_pert = self._select_metric_projection(G + H, mode=projector_mode)
        dummy_curv = (
            list(curvature_history[-2:])
            if len(curvature_history) >= 2
            else [ricci_scalar, ricci_scalar]
        )
        phi_G, _ = self._rsi_level2_adaptive_tau(
            G, delta_tau, eta, dummy_curv,
            projector_mode=projector_mode, discretizer=discretizer,
        )
        phi_P, _ = self._rsi_level2_adaptive_tau(
            G_pert, delta_tau, eta, dummy_curv,
            projector_mode=projector_mode, discretizer=discretizer,
        )
        return _fro(phi_P - phi_G) / max(_fro(G_pert - G), 1.0e-15)

    # ── FASE 2.G — Jacobiano linealizado (diferencias centrales) ──────────
    def _linearized_flow_jacobian(
        self, G: NDArray[np.float64], delta_tau: float, eta: float,
        curvature_history: List[float], eps: float = _JACOBI_FD_EPS,
        projector_mode: Optional[ProjectorMode] = None,
        discretizer: Optional[RicciDiscretizer] = None,
    ) -> NDArray[np.float64]:
        r"""
        Jacobiano numérico de $\Phi^{(2)}$ por diferencias finitas *centrales*:
        $J_{\cdot i}=\bigl(\Phi(G+\varepsilon e_i)-\Phi(G-\varepsilon e_i)\bigr)/(2\varepsilon)$.
        Dimensión $n^{2}\times n^{2}$.  Orden $O(\varepsilon^{2})$, superior a v7.2.0.
        """
        n = G.shape[0]
        n2 = n * n

        def _phi(H: NDArray[np.float64]) -> NDArray[np.float64]:
            return self._rsi_level2_adaptive_tau(
                H, delta_tau, eta, curvature_history,
                projector_mode=projector_mode, discretizer=discretizer,
            )[0]

        J = np.zeros((n2, n2), dtype=np.float64)
        scale = max(_fro(G), 1.0) * float(eps)
        for i in range(n2):
            Hp = G.copy()
            Hm = G.copy()
            Hp.flat[i] += scale
            Hm.flat[i] -= scale
            J[:, i] = (_phi(Hp).ravel() - _phi(Hm).ravel()) / (2.0 * scale)
        return J

    def _benettin_qr_lyapunov(
        self, orbit: List[NDArray[np.float64]], delta_tau: float, eta: float,
        curvature_history: List[float],
        projector_mode: Optional[ProjectorMode] = None,
        discretizer: Optional[RicciDiscretizer] = None,
    ) -> np.ndarray:
        r"""
        Espectro de Lyapunov vía Benettin–Shimada (1980), con reducción de
        subespacio cuando $n^{2}>\_LYAP\_DIM\_CAP$: se itera
        $Q_{k+1}R_{k+1}=J_k Q_k$ y $\lambda_i=\lim(1/K)\sum_k\log|R_k^{(ii)}|$.
        """
        n_steps = len(orbit) - 1
        if n_steps < 2:
            return np.array([-np.inf])
        n = orbit[0].shape[0]
        n2 = n * n
        d = int(min(n2, _LYAP_DIM_CAP))
        if d < n2:
            Q, _ = np.linalg.qr(self._rng.normal(size=(n2, d)))
        else:
            Q = np.eye(n2)
        log_sums = np.zeros(d)
        count = 0
        for k in range(1, n_steps):
            if k % self._benettin_qr_stride != 0:
                continue
            try:
                J = self._linearized_flow_jacobian(
                    orbit[k], delta_tau, eta, curvature_history,
                    projector_mode=projector_mode, discretizer=discretizer,
                )
            except (la.LinAlgError, ValueError):
                continue
            M = J @ Q
            Q_new, R = np.linalg.qr(M)
            diag_R = np.abs(np.diag(R))
            log_sums += np.log(np.maximum(diag_R, _WILKINSON_EPS))
            Q = Q_new
            count += 1
        if count == 0:
            return self._lyapunov_exponent_spectrum(orbit)
        return np.sort(log_sums / count)[::-1]

    def _symplectic_lyapunov_pairing(self, spectrum: np.ndarray) -> float:
        r"""
        Defecto de pareado simpléctico $\frac1{\lfloor d/2\rfloor}\sum_i|\lambda_i+\lambda_{d-1-i}|$
        (para flujos hamiltonianos, el espectro es palíndromo $\lambda\leftrightarrow-\lambda$).
        """
        lam = np.asarray(spectrum, dtype=np.float64).ravel()
        d = int(lam.size)
        if d < 2:
            return 0.0
        lam_s = np.sort(lam)[::-1]
        pairs = min(d // 2, lam_s.size)
        if pairs == 0:
            return 0.0
        defects = [
            abs(float(lam_s[i]) + float(lam_s[d - 1 - i]))
            / max(1.0 + abs(float(lam_s[i])), _WILKINSON_EPS)
            for i in range(pairs)
        ]
        return float(np.mean(defects)) if defects else 0.0

    # ── FASE 2.H — Orquestador RSI³ (anidamiento 1 ⊂ 2 ⊂ 3 + métodos) ─────
    def _rsi3_driver(
        self,
        G0: NDArray[np.float64],
        n_iterations: Optional[int] = None,
        delta_tau0: float = 0.01,
        eta0: float = 0.1,
        zeta0: float = 0.01,
    ) -> Dict[str, Any]:
        n_iter = int(n_iterations or self._rsi3_max_iterations)
        G = np.array(G0, copy=True, dtype=np.float64)
        delta_tau, eta, zeta = float(delta_tau0), float(eta0), float(zeta0)
        orbit: List[NDArray[np.float64]] = [np.array(G, copy=True)]
        state0 = self._evaluate_geometry(G)
        curv_hist: List[float] = [state0.ricci_scalar]
        contraction_hist: List[float] = []
        lipschitz_hist: List[float] = []
        tau_hist: List[float] = [delta_tau]
        eta_hist: List[float] = [eta]
        zeta_hist: List[float] = [zeta]
        method_lineage: List[str] = []
        projector = self._projector_mode
        discretizer = self._ricci_discretizer
        consecutive_small = 0

        for k in range(n_iter):
            state_k = self._evaluate_geometry(G) if k > 0 else state0
            projector, discretizer, tag = self._rsi3_method_selector(
                state_k, projector, discretizer
            )
            method_lineage.append(tag)

            zeta = self._adapt_zeta_meta_meta(zeta, eta_hist)
            G_next, delta_tau, eta = self._rsi_level3_adaptive_eta(
                G, delta_tau, eta, zeta, contraction_hist, curv_hist,
                projector_mode=projector, discretizer=discretizer,
            )
            orbit.append(np.array(G_next, copy=True))
            state_next = self._evaluate_geometry(G_next)
            curv_hist.append(state_next.ricci_scalar)
            tau_hist.append(delta_tau)
            eta_hist.append(eta)
            zeta_hist.append(zeta)

            if k >= 1:
                d_new = _fro(orbit[-1] - orbit[-2])
                d_old = _fro(orbit[-2] - orbit[-3]) if k >= 2 else d_new
                if d_old > 1.0e-15:
                    contraction_hist.append(d_new / d_old)

            if k > 0 and (k % self._lipschitz_probe_stride == 0):
                try:
                    L_probe = self._lipschitz_phi_empirical(
                        G_next, delta_tau, eta, curv_hist,
                        projector_mode=projector, discretizer=discretizer,
                    )
                    lipschitz_hist.append(float(np.clip(L_probe, 0.0, 1.0e6)))
                except (la.LinAlgError, ValueError) as exc:
                    logger.debug("[RSI³] sonda Lipschitz omitida: %s", exc)

            G = G_next

            if k >= 1:
                step_norm = _fro(orbit[-1] - orbit[-2])
                if step_norm < _RSI3_CONV_EPS:
                    consecutive_small += 1
                    if consecutive_small >= _CONSECUTIVE_CONV_STEPS:
                        break
                else:
                    consecutive_small = 0

        L_final = self._banach_contraction_constant(orbit)
        if lipschitz_hist:
            L_final = float(max(L_final, float(np.median(lipschitz_hist))))
        banach_cert = self._banach_fixed_point_certificate(orbit, L_final)
        converged = bool(banach_cert.certified)
        quad_rate = self._quadratic_convergence_rate(orbit)
        logger.info(
            "[RSI³] Niveles=1⊂2⊂3+método, iter=%d, L_Φ=%.4f, p=%.2f, certified=%s, "
            "Δτ_final=%.3e, η_final=%.3e, ζ_final=%.3e, linaje=%s",
            len(orbit) - 1, L_final, quad_rate, banach_cert.certified,
            delta_tau, eta, zeta,
            method_lineage[-1] if method_lineage else "n/a",
        )
        return {
            "orbit": orbit, "curvature_history": curv_hist,
            "contraction_history": contraction_hist,
            "lipschitz_history": lipschitz_hist,
            "delta_tau_history": tau_hist, "eta_history": eta_hist,
            "zeta_history": zeta_hist,
            "rsi_level_achieved": 3, "converged": converged,
            "banach_L": L_final, "quadratic_rate": quad_rate,
            "method_lineage": method_lineage,
            "banach_certified": banach_cert.certified,
            "projector_mode": projector.value,
            "discretizer": discretizer.value,
        }

    # ── FASE 2.I — Mapa de retorno de Poincaré (sección interpolada) ─────
    def _poincare_return_map(
        self,
        orbit: List[NDArray[np.float64]],
        energy_target: float,
        energies: Optional[NDArray[np.float64]] = None,
    ) -> Dict[str, Any]:
        if energies is None:
            energies_arr = np.array(
                [self.compute_ricci_tensor_discretized(G)[1] for G in orbit],
                dtype=np.float64,
            )
        else:
            energies_arr = np.asarray(energies, dtype=np.float64)
            if len(energies_arr) != len(orbit):
                n_min = min(len(energies_arr), len(orbit))
                energies_arr = energies_arr[:n_min]
                orbit = orbit[:n_min]

        crossings: List[int] = []
        taus: List[float] = []
        section_points: List[NDArray[np.float64]] = []

        for k in range(1, len(energies_arr)):
            e0, e1 = float(energies_arr[k - 1]), float(energies_arr[k])
            if (e0 - energy_target) * (e1 - energy_target) < 0.0:
                denom = e1 - e0
                alpha = 0.5 if abs(denom) < 1.0e-15 else (energy_target - e0) / denom
                alpha = float(np.clip(alpha, 0.0, 1.0))
                crossings.append(k)
                taus.append((k - 1) + alpha)
                G_cross = (1.0 - alpha) * orbit[k - 1] + alpha * orbit[k]
                G_cross, _, _, _ = self._spectral_decomposition_psd(G_cross)
                section_points.append(G_cross)

        periods = (
            np.diff(np.asarray(taus, dtype=np.float64))
            if len(taus) > 1 else np.array([], dtype=np.float64)
        )
        is_periodic = False
        if len(periods) >= 2:
            mean_p = float(np.mean(periods))
            std_rel = float(np.std(periods) / max(mean_p, 1.0e-15))
            is_periodic = std_rel < 0.05

        monodromy, monodromy_eigs = self._monodromy_matrix(section_points)
        floquet = self._variational_floquet(monodromy, monodromy_eigs)

        return {
            "crossings": crossings, "crossing_times": taus,
            "return_periods": periods, "is_periodic_orbit": is_periodic,
            "energies": energies_arr, "energy_target": float(energy_target),
            "floquet_multiplier": floquet, "section_points": section_points,
            "monodromy": monodromy, "monodromy_eigs": monodromy_eigs,
        }

    # ── FASE 2.I′ — Monodromía variacional y Floquet espectral ────────────
    def _monodromy_matrix(
        self, section_points: List[NDArray[np.float64]]
    ) -> Tuple[NDArray[np.float64], np.ndarray]:
        r"""
        Matriz de monodromía $M=D\mathcal{T}|_{\Sigma}$ del mapa de Poincaré.

        Se estima por regresión lineal de mínimos cuadrados sobre las desviaciones
        centradas de secciones consecutivas (proyección PCA al plano tangente de
        $\Sigma$, dim $r=\min(8,\,m-1,\,n^{2})$):

            $\delta G_{k+1}\approx M\,\delta G_k$.

        Devuelve $(M, \mathrm{eig}(M))$.
        """
        empty = np.zeros((1, 1), dtype=np.float64)
        if len(section_points) < 3:
            return empty, np.array([0.0], dtype=np.complex128)
        try:
            X = np.array([G.ravel() for G in section_points], dtype=np.float64)
            Xc = X - X.mean(axis=0, keepdims=True)
            m, n2 = Xc.shape
            r = int(min(8, m - 1, n2, max(m - 1, 1)))
            if r < 1:
                return empty, np.array([0.0], dtype=np.complex128)
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            r = int(min(r, int((S > 1.0e-12 * max(float(S[0]), 1.0)).sum()) or 1))
            coords = U[:, :r] * S[:r]          # (m, r)  coordenadas tangentes
            src = coords[:-1]
            dst = coords[1:]
            # M_r tal que dst ≈ src @ M_r.T   ⇒  M_r = dst.T @ pinv(src.T)
            M_r, *_ = np.linalg.lstsq(src, dst, rcond=None)
            M_r = np.asarray(M_r, dtype=np.float64).T   # (r, r)
            eigs = np.linalg.eigvals(M_r)
            return M_r, eigs
        except (la.LinAlgError, ValueError):
            return empty, np.array([0.0], dtype=np.complex128)

    def _variational_floquet(
        self,
        monodromy: NDArray[np.float64],
        monodromy_eigs: np.ndarray,
    ) -> float:
        r"""
        Multiplicador de Floquet dominante: $\mu_{\max}=\rho(M)=\max_i|\mathrm{eig}(M)_i|$.
        Caída suave a 0 si la monodromía es degenerada.
        """
        eigs = np.asarray(monodromy_eigs)
        if eigs.size == 0:
            if monodromy.size == 0:
                return 0.0
            try:
                eigs = np.linalg.eigvals(monodromy)
            except la.LinAlgError:
                return 0.0
        return float(np.max(np.abs(eigs)))

    # ── FASE 2.I″ — Lindstedt y recurrencia de Poincaré ───────────────────
    def _poincare_lindstedt_correction(
        self,
        periods: np.ndarray,
        energies: np.ndarray,
        energy_target: float,
    ) -> float:
        r"""
        Corrección de Poincaré–Lindstedt del periodo:
        $T=T_0\bigl(1+\varepsilon T_1+\cdots\bigr)$, $\varepsilon=(E-\bar E)/\max(|\bar E|,1)$,
        $T_1\approx\mathrm{corr}(T_k-\bar T,\,E_k-\bar E)/\mathrm{Var}(E)$ (primera orden).
        Devuelve $T_{\mathrm{Lindstedt}}$.
        """
        P = np.asarray(periods, dtype=np.float64).ravel()
        if P.size == 0:
            return 0.0
        T0 = float(np.mean(P))
        E = np.asarray(energies, dtype=np.float64).ravel()
        if E.size < 2 or P.size < 2:
            return T0
        n = int(min(P.size, max(E.size - 1, 1)))
        if n < 2:
            return T0
        e_sec = E[:n] - float(energy_target)
        p_sec = P[:n] - T0
        var_e = float(np.var(e_sec))
        if var_e < 1.0e-18:
            return T0
        T1 = float(np.mean(p_sec * e_sec)) / var_e
        eps = float(np.std(e_sec)) / max(abs(float(energy_target)), 1.0)
        return float(T0 * (1.0 + eps * T1))

    def _poincare_recurrence_bound(
        self,
        log_volumes: Sequence[float],
        lyapunov_max: float,
        n_dim: int,
        mean_step: float,
    ) -> float:
        r"""
        Cota inferior del tiempo de recurrencia de Poincaré:
        $\tau_{\mathrm{rec}}\gtrsim \mathrm{Vol}(\mathcal{M})/\mathrm{Vol}(\mathrm{tubo})$.
        $\mathrm{Vol}\approx\exp(\mathrm{median}\,\log\mathrm{Vol})$,
        tubo $\approx \mathrm{e}^{\lambda_{\max}^+}\,\Delta\tau^{\,n(n+1)/4}$
        (sección de métricas SPD, dim efectiva $n(n+1)/2$).
        """
        lv = np.asarray(list(log_volumes), dtype=np.float64)
        lv = lv[np.isfinite(lv)]
        if lv.size == 0:
            return 0.0
        vol = float(np.exp(np.clip(float(np.median(lv)), -700.0, 700.0)))
        dim_eff = max(0.5 * float(n_dim) * float(n_dim + 1), 1.0)
        lam_p = max(float(lyapunov_max), 0.0)
        tube = float(np.exp(np.clip(lam_p, 0.0, 50.0))) * max(abs(mean_step), 1.0e-12) ** (0.5 * dim_eff)
        tube = max(tube, _WILKINSON_EPS)
        return float(max(vol / tube, 0.0))

    # ── FASE 2.J — Variables acción-ángulo y función generatriz HJ ───────
    def _action_angle_extract(
        self, section_points: List[NDArray[np.float64]]
    ) -> Tuple[float, float]:
        r"""
        Variables acción-ángulo $(I,\theta)$ sobre $\Sigma\subset T^*\mathcal{M}$:

          • Acción: $I=\frac1{2\pi}\oint_\Sigma p\,dq$ (área simpléctica PCA).
          • Ángulo: fase del armónico dominante de la FFT de $q_k$.
        """
        if len(section_points) < 3:
            return (0.0, 0.0)
        try:
            X = np.array([G.ravel() for G in section_points], dtype=np.float64)
            X_centered = X - X.mean(axis=0, keepdims=True)
            U, S, _ = np.linalg.svd(X_centered, full_matrices=False)
            if len(S) < 2 or S[1] < 1.0e-15:
                return (0.0, 0.0)
            q = U[:, 0] * S[0]
            p = U[:, 1] * S[1]
            area = 0.0
            m = len(q)
            for k in range(m - 1):
                area += q[k] * p[k + 1] - q[k + 1] * p[k]
            if m >= 2:
                area += q[-1] * p[0] - q[0] * p[-1]
            action = abs(area) / (2.0 * np.pi)
            q_fft = np.fft.rfft(q - q.mean())
            if q_fft.size > 1:
                dom_idx = int(np.argmax(np.abs(q_fft[1:])) + 1)
                angle = float(np.angle(q_fft[dom_idx]))
            else:
                angle = 0.0
            return (float(action), float(angle))
        except (la.LinAlgError, ValueError):
            return (0.0, 0.0)

    def _hamilton_jacobi_generating_function(
        self, section_points: List[NDArray[np.float64]], action: float
    ) -> float:
        r"""
        Residual de la función generatriz de tipo 2 $S(q,I)$ (Hamilton–Jacobi):
        $p=\partial S/\partial q$, $\theta=\partial S/\partial I$.
        Se toma $S=\sum_k p_k(q_{k+1}-q_k)$ (regla del trapecio) y se verifica
        $|I-\frac1{2\pi}(S_{\mathrm{fin}}-S_{\mathrm{ini}})|$ relativo.
        """
        if len(section_points) < 3 or abs(action) < 1.0e-18:
            return 0.0
        try:
            X = np.array([G.ravel() for G in section_points], dtype=np.float64)
            Xc = X - X.mean(axis=0, keepdims=True)
            U, S, _ = np.linalg.svd(Xc, full_matrices=False)
            if len(S) < 2 or S[1] < 1.0e-15:
                return 0.0
            q = U[:, 0] * S[0]
            p = U[:, 1] * S[1]
            S_gen = float(np.sum(p[:-1] * np.diff(q)))
            I_from_S = abs(S_gen) / (2.0 * np.pi)
            return _safe_div(abs(I_from_S - abs(action)), max(abs(action), _WILKINSON_EPS))
        except (la.LinAlgError, ValueError):
            return 0.0

    # ── FASE 2.K — Espectro de Lyapunov (fallback 1-D) ────────────────────
    def _lyapunov_exponent_spectrum(self, orbit: List[NDArray[np.float64]]) -> np.ndarray:
        if len(orbit) < 3:
            return np.array([-np.inf], dtype=np.float64)
        norms = [_fro(orbit[k + 1] - orbit[k]) for k in range(len(orbit) - 1)]
        log_ratios = [
            float(np.log(max(norms[k + 1] / max(norms[k], 1.0e-15), 1.0e-15)))
            for k in range(len(norms) - 1)
        ]
        if not log_ratios:
            return np.array([-np.inf], dtype=np.float64)
        return np.array([float(np.mean(log_ratios))], dtype=np.float64)

    # ── FASE 2.L — Frecuencias orbitales (FFT de la curvatura) ───────────
    def _extract_orbital_frequencies(
        self, signal: NDArray[np.float64], dt: float = 1.0
    ) -> np.ndarray:
        x = np.asarray(signal, dtype=np.float64)
        n = int(x.size)
        if n < 8:
            return np.array([], dtype=np.float64)
        x = x - float(np.mean(x))
        hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / max(n - 1, 1))
        spec = np.abs(np.fft.rfft(x * hann))
        freqs = np.fft.rfftfreq(n, d=float(dt))
        spec[0] = 0.0
        thresh = float(np.median(spec) * 3.0)
        peaks: List[float] = []
        for i in range(1, len(spec) - 1):
            if spec[i] > spec[i - 1] and spec[i] > spec[i + 1] and spec[i] > thresh:
                peaks.append(float(freqs[i]))
        peaks.sort(key=lambda f: -float(spec[int(np.argmin(np.abs(freqs - f)))]))
        return np.asarray(peaks[:4], dtype=np.float64)

    # ── FASE 2.M — Resonancias por fracciones continuas ───────────────────
    def _continued_fraction_resonances(
        self, omega: np.ndarray, max_denom: int = 64, max_resonances: int = 6
    ) -> Tuple[Tuple[int, ...], ...]:
        r"""
        Retículo de resonancias $k$ con $\omega\cdot k\approx 0$ vía convergentes
        de $\omega_i/\omega_j$ (criterio KAM de divisores pequeños).
        """
        omega = np.asarray(omega, dtype=np.float64).ravel()
        n = int(omega.size)
        if n < 2:
            return ()

        def _cf_convergents(x: float, max_den: int, depth: int = 20) -> List[Tuple[int, int]]:
            conv: List[Tuple[int, int]] = []
            a = float(x)
            p_prev, p_curr = 1, 0
            q_prev, q_curr = 0, 1
            for _ in range(depth):
                ai = int(np.floor(a))
                p_prev, p_curr = p_curr, ai * p_curr + p_prev
                q_prev, q_curr = q_curr, ai * q_curr + q_prev
                if q_curr == 0 or q_curr > max_den:
                    break
                conv.append((p_curr, q_curr))
                frac = a - ai
                if abs(frac) < 1.0e-12:
                    break
                a = 1.0 / frac
            return conv

        candidates: List[Tuple[float, Tuple[int, ...]]] = []
        for i, j in combinations(range(n), 2):
            if abs(omega[j]) < 1.0e-15:
                continue
            ratio = float(omega[i] / omega[j])
            for p, q in _cf_convergents(ratio, max_denom):
                if p == 0 and q == 0:
                    continue
                vec = [0] * n
                vec[i] = q
                vec[j] = -p
                dot = abs(float(np.dot(omega, np.asarray(vec, dtype=np.float64))))
                if dot > 0.0:
                    candidates.append((dot, tuple(vec)))
        candidates.sort(key=lambda t: t[0])
        return tuple(vec for _, vec in candidates[:max_resonances])

    # ── FASE 2.Z — HANDOFF: última definición de FASE 2 ≡ inicio de FASE 3
    def _handoff_to_phase3(
        self,
        phase1_handoff: Phase1Handoff,
        rsi_out: Dict[str, Any],
    ) -> Phase2Handoff:
        r"""
        Última definición formal de FASE 2 y continuación estricta de FASE 3.

        $\Phi_3\circ\Phi_2(h_1)=\texttt{\_ingest\_phase2\_handoff}(\texttt{\_handoff\_to\_phase3}(h_1,\mathrm{RSI}^3))$.

        Empaqueta espectro de Lyapunov (Benettin reducido), mapa de Poincaré con
        monodromía, corrección de Lindstedt, cota de recurrencia, acción-ángulo
        Hamilton–Jacobi y linaje de métodos RSI³.  Ese paquete es el objeto de
        ingesta de `_ingest_phase2_handoff` (FASE 3.A), que abre KAM/Melnikov.
        """
        orbit: List[NDArray[np.float64]] = list(rsi_out["orbit"])
        curv_hist = list(rsi_out.get("curvature_history", []))
        energies = np.asarray(curv_hist, dtype=np.float64) if curv_hist else None
        energy_target = (
            float(np.median(energies)) if energies is not None and energies.size else 0.0
        )

        poincare = self._poincare_return_map(orbit, energy_target, energies=energies)
        lyap_spec_full = self._benettin_qr_lyapunov(
            orbit,
            float(rsi_out.get("delta_tau_history", [0.01])[-1]),
            float(rsi_out.get("eta_history", [0.1])[-1]),
            curv_hist,
        )
        lyap_max = float(lyap_spec_full[0]) if lyap_spec_full.size else float(
            self._lyapunov_exponent_spectrum(orbit)[0]
        )
        pairing_defect = self._symplectic_lyapunov_pairing(lyap_spec_full)
        dt_mean = float(np.mean(rsi_out.get("delta_tau_history", [1.0])))
        omega = self._extract_orbital_frequencies(
            energies if energies is not None else np.array([0.0]),
            dt=max(dt_mean, 1.0e-12),
        )
        banach_L = float(rsi_out.get("banach_L", self._banach_contraction_constant(orbit)))
        action, angle = self._action_angle_extract(poincare["section_points"])
        gf_residual = self._hamilton_jacobi_generating_function(poincare["section_points"], action)
        resonances = self._continued_fraction_resonances(omega) if omega.size >= 2 else ()
        quad_rate = float(rsi_out.get("quadratic_rate", 0.0))
        periods = np.asarray(poincare.get("return_periods", []), dtype=np.float64)
        lindstedt = self._poincare_lindstedt_correction(
            periods,
            energies if energies is not None else np.array([0.0]),
            energy_target,
        )
        n_dim = int(orbit[0].shape[0]) if orbit else 1
        rec_bound = self._poincare_recurrence_bound(
            phase1_handoff.log_volumes, lyap_max, n_dim, dt_mean
        )
        monodromy_eigs = tuple(complex(z) for z in np.asarray(poincare.get("monodromy_eigs", [])))
        method_lineage = tuple(str(x) for x in rsi_out.get("method_lineage", []))

        handoff = Phase2Handoff(
            phase1=phase1_handoff,
            lyapunov_max=lyap_max,
            poincare_data=poincare,
            contraction_constants=list(rsi_out.get("contraction_history", [])),
            rsi_level_achieved=int(rsi_out.get("rsi_level_achieved", 1)),
            rsi_converged=bool(rsi_out.get("converged", False)),
            banach_L=banach_L,
            eta_history=tuple(float(x) for x in rsi_out.get("eta_history", [])),
            delta_tau_history=tuple(float(x) for x in rsi_out.get("delta_tau_history", [])),
            curvature_history=tuple(float(x) for x in curv_hist),
            dominant_frequencies=tuple(float(x) for x in omega),
            lyapunov_spectrum=tuple(float(x) for x in lyap_spec_full),
            action_angle=(float(action), float(angle)),
            quadratic_rate=quad_rate,
            resonance_vectors=resonances,
            method_lineage=method_lineage,
            monodromy_eigs=monodromy_eigs,
            lindstedt_period=float(lindstedt),
            recurrence_bound=float(rec_bound),
            banach_certified=bool(rsi_out.get("banach_certified", False)),
            generating_function_residual=float(gf_residual),
            lyapunov_pairing_defect=float(pairing_defect),
            projector_mode=str(rsi_out.get("projector_mode", self._projector_mode.value)),
        )
        logger.info(
            "[FASE2→FASE3] λ_max=%.4e, |Λ|=%d, crossings=%d, L_Φ=%.4f, "
            "p=%.2f, I=%.3e, θ=%.3f, n_res=%d, μ_Floquet=%.3e, T_Lind=%.3e, "
            "τ_rec=%.3e, pairing=%.3e, HJ=%.3e",
            handoff.lyapunov_max, len(lyap_spec_full),
            len(poincare.get("crossings", [])), banach_L, quad_rate,
            action, angle, len(resonances),
            float(poincare.get("floquet_multiplier", 0.0)),
            lindstedt, rec_bound, pairing_defect, gf_residual,
        )
        return handoff

    # ╔════════════════════════════════════════════════════════════════════════╗
    # ║  FASE 3 — KAM, HOMOCLÍNICA, HEYTING Y GOBERNANZA OODA                  ║
    # ║  (inicia exactamente donde termina FASE 2.Z)                           ║
    # ╚════════════════════════════════════════════════════════════════════════╝

    # ── FASE 3.A — INGESTA del handoff de FASE 2 (continuación directa) ──
    def _ingest_phase2_handoff(
        self, handoff: Phase2Handoff
    ) -> Tuple[Phase1Handoff, Dict[str, Any], float, Tuple[float, ...]]:
        r"""Consume el `Phase2Handoff` de `_handoff_to_phase3` (continuación 2→3)."""
        return (
            handoff.phase1,
            handoff.poincare_data,
            float(handoff.lyapunov_max),
            tuple(handoff.dominant_frequencies),
        )

    # ── FASE 3.B — Criterio diofántico KAM (pares + retículo) ────────────
    def _diophantine_gamma(
        self, omega: np.ndarray, tau: float = _KAM_TAU, max_k_norm: int = 4,
    ) -> float:
        r"""$\gamma=\inf_{k\neq 0}|\omega\cdot k|\,|k|^{\tau}$ sobre el retículo."""
        omega = np.asarray(omega, dtype=np.float64).ravel()
        n = int(omega.size)
        if n == 0:
            return float("inf")
        if n == 1:
            return float(abs(omega[0]))

        min_gamma = np.inf
        if n > 4:
            for i, j in combinations(range(n), 2):
                pair = omega[[i, j]]
                for k in product(range(-max_k_norm, max_k_norm + 1), repeat=2):
                    if k[0] == 0 and k[1] == 0:
                        continue
                    k_arr = np.asarray(k, dtype=np.float64)
                    k_norm = float(np.linalg.norm(k_arr))
                    if k_norm > max_k_norm or k_norm < 1.0e-15:
                        continue
                    candidate = abs(float(np.dot(pair, k_arr))) * (k_norm ** tau)
                    if candidate < min_gamma:
                        min_gamma = candidate
            return float(min_gamma)

        for k in product(range(-max_k_norm, max_k_norm + 1), repeat=n):
            if all(ki == 0 for ki in k):
                continue
            k_arr = np.asarray(k, dtype=np.float64)
            k_norm = float(np.linalg.norm(k_arr))
            if k_norm > max_k_norm or k_norm < 1.0e-15:
                continue
            candidate = abs(float(np.dot(omega, k_arr))) * (k_norm ** tau)
            if candidate < min_gamma:
                min_gamma = candidate
        return float(min_gamma)

    def _kam_torus_stability(
        self,
        eigenvalues: np.ndarray,
        tau: float = _KAM_TAU,
        max_k_norm: int = 4,
        orbital_frequencies: Optional[np.ndarray] = None,
        resonance_vectors: Tuple[Tuple[int, ...], ...] = (),
    ) -> Dict[str, float]:
        r"""
        Criterio KAM: γ sobre el retículo completo, refinado con convergentes
        de fracciones continuas (resonancias peligrosas de Poincaré).
        """
        if orbital_frequencies is not None and np.asarray(orbital_frequencies).size >= 1:
            omega = np.asarray(orbital_frequencies, dtype=np.float64)
            source = "orbital"
        else:
            ev = np.asarray(eigenvalues, dtype=np.float64)
            omega = np.log(np.maximum(ev, _SPECTRAL_PSD_FLOOR))
            source = "spectral"
            if omega.size < 2:
                return {"diophantine_gamma": float("inf"), "is_kam_stable": 1.0, "source": 0.0}

        gamma = self._diophantine_gamma(omega, tau=tau, max_k_norm=max_k_norm)
        if resonance_vectors:
            for k in resonance_vectors:
                k_arr = np.asarray(k, dtype=np.float64)
                kn = float(np.linalg.norm(k_arr))
                if kn < 1.0e-15:
                    continue
                n_dot = int(min(k_arr.size, omega.size))
                gamma = min(
                    gamma,
                    abs(float(np.dot(omega[:n_dot], k_arr[:n_dot]))) * (kn ** tau),
                )
        return {
            "diophantine_gamma": float(gamma),
            "is_kam_stable": 1.0 if gamma >= _KAM_GAMMA else 0.0,
            "source": 1.0 if source == "orbital" else 0.0,
        }

    # ── FASE 3.C — Función de Melnikov (proxy + integral homoclínica) ────
    def _melnikov_function(
        self,
        section_points: List[NDArray[np.float64]],
        lyapunov_max: float,
    ) -> float:
        r"""
        Proxy de Melnikov (splitting transversal):
        $M\approx\sum_k\bigl|\langle\dot G_k^{\perp},\eta_k\rangle\bigr|$,
        $M_{\mathrm{norm}}=M/\max(\lambda_{\max},10^{-3})$.
        """
        if len(section_points) < 3:
            return 0.0
        try:
            orbitals = [
                section_points[k + 1] - section_points[k]
                for k in range(len(section_points) - 1)
            ]
            if not orbitals:
                return 0.0
            O = np.array([o.ravel() for o in orbitals], dtype=np.float64)
            O_norm = O / (np.linalg.norm(O, axis=1, keepdims=True) + 1.0e-15)
            Q, _ = np.linalg.qr(O_norm.T)
            normal_basis = Q[:, len(orbitals):] if Q.shape[1] > len(orbitals) else Q
            if normal_basis.size == 0:
                return 0.0
            M = 0.0
            for k in range(len(section_points) - 2):
                delta = (section_points[k + 1] - section_points[k]).ravel()
                proj = normal_basis.T @ delta
                M += float(np.linalg.norm(proj))
            return float(M / max(abs(lyapunov_max), 1.0e-3))
        except (la.LinAlgError, ValueError):
            return 0.0

    def _melnikov_homoclinic_integral(
        self,
        section_points: List[NDArray[np.float64]],
        delta_tau_mean: float,
        lyapunov_max: float,
    ) -> float:
        r"""
        Integral de Melnikov discreta a lo largo de la separatriz homoclínica:

            $M(\theta_0)=\sum_k \langle J\nabla H(z_k^{0}),\, g(z_k^{0},\theta_0)\rangle\,\Delta\tau$,

        donde $z_k^{0}$ recorre la sección, $\nabla H$ se aproxima por la dirección
        orbital y $g$ es la componente de perturbación ortogonal (proyección QR).
        Un $M\neq 0$ implica splitting transversal (Poincaré–Smale).
        """
        if len(section_points) < 4:
            return self._melnikov_function(section_points, lyapunov_max)
        try:
            X = np.array([G.ravel() for G in section_points], dtype=np.float64)
            V = np.diff(X, axis=0)                       # ∇H proxy
            if V.shape[0] < 2:
                return 0.0
            # Estructura simpléctica canónica sobre el plano PCA (q, p)
            Xc = X - X.mean(axis=0, keepdims=True)
            U, S, _ = np.linalg.svd(Xc, full_matrices=False)
            if S.size < 2 or S[1] < 1.0e-15:
                return self._melnikov_function(section_points, lyapunov_max)
            qp = np.column_stack((U[:, 0] * S[0], U[:, 1] * S[1]))
            dqp = np.diff(qp, axis=0)
            # J = [[0,-1],[1,0]]  (convención de Darboux 2-D)
            J = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=np.float64)
            # perturbación g = componente de dqp rotada 90° menos su proyección orbital
            g = dqp @ J.T
            orbital_n = dqp / (np.linalg.norm(dqp, axis=1, keepdims=True) + 1.0e-15)
            g = g - np.sum(g * orbital_n, axis=1, keepdims=True) * orbital_n
            integrand = np.sum((dqp @ J.T) * g, axis=1)
            M = float(np.sum(integrand) * float(delta_tau_mean))
            return float(abs(M) / max(abs(lyapunov_max), 1.0e-3))
        except (la.LinAlgError, ValueError):
            return self._melnikov_function(section_points, lyapunov_max)

    # ── FASE 3.D — Detección de tangle homoclínico (Poincaré–Smale) ──────
    def _detect_homoclinic_tangle(
        self,
        lyapunov_max: float,
        poincare_data: Dict[str, Any],
        melnikov_amplitude: float = 0.0,
    ) -> bool:
        r"""
        Tangle homoclínico si:
          (i)  $\lambda_{\max}>10^{-2}$,
          (ii) $\ge 3$ cruces con periodos irregulares,
          (iii) $|\mu_{\mathrm{Floquet}}|>1.01$,
          (iv) $|M_{\mathrm{norm}}|>10^{-2}$ (Melnikov no nulo).
        """
        if lyapunov_max <= _LYAPUNOV_CHAOS:
            return False
        crossings = poincare_data.get("crossings", [])
        if len(crossings) < 3:
            return False
        periods = np.asarray(poincare_data.get("return_periods", []), dtype=np.float64)
        if periods.size < 2:
            return False
        irregular = bool(np.std(periods) / max(float(np.mean(periods)), 1.0e-15) > 0.1)
        floquet = abs(float(poincare_data.get("floquet_multiplier", 0.0)))
        stretching = floquet > _FLOQUET_UNSTABLE
        melnikov_significant = melnikov_amplitude > 1.0e-2
        return bool(irregular and stretching and melnikov_significant)

    # ── FASE 3.E — Índice de Poincaré–Hopf y Conley–Morse ────────────────
    def _poincare_hopf_index(self, orbit: List[NDArray[np.float64]]) -> float:
        if len(orbit) < 3:
            return 0.0
        velocities = [_fro(orbit[k + 1] - orbit[k]) for k in range(len(orbit) - 1)]
        if max(velocities) < 1.0e-15:
            return 0.0
        total_index = 0.0
        for k in range(1, len(velocities) - 1):
            if velocities[k] < velocities[k - 1] and velocities[k] < velocities[k + 1]:
                d_in = orbit[k] - orbit[k - 1]
                d_out = orbit[k + 1] - orbit[k]
                n_in, n_out = _fro(d_in), _fro(d_out)
                if n_in < 1.0e-15 or n_out < 1.0e-15:
                    continue
                cos_ang = float(np.sum(d_in * d_out) / (n_in * n_out))
                total_index += 1.0 if cos_ang > 0.0 else -1.0
        return float(total_index)

    def _conley_morse_index(
        self,
        hopf_index: float,
        lyapunov_spectrum: Sequence[float],
    ) -> float:
        r"""
        Índice de Conley–Morse (proxy): $m=\#\{\lambda_i>0\}$ (dimensión inestable)
        corregido por el índice de Poincaré–Hopf, $h=m-\mathrm{Hopf}$.
        $h=0$ sugiere atractor acíclico; $h>0$ silla; $h<0$ fuente invertida.
        """
        spec = [float(x) for x in lyapunov_spectrum if np.isfinite(x)]
        unstable = sum(1 for x in spec if x > _LYAPUNOV_CHAOS)
        return float(unstable) - float(hopf_index)

    # ── FASE 3.E′ — Cuantización geométrica prequántica (Kostant–Souriau)
    def _geometric_quantization_prequantum(
        self,
        canonical_omega: NDArray[np.float64],
        jacobian_M: NDArray[np.float64],
    ) -> float:
        r"""
        Residual de la condición prequántica $F_{\nabla}=\omega$:
          (i)  $\omega+\omega^{T}=0$  (cerrada + alternada en carta lineal),
          (ii) $M^{T}\omega M=\omega$ (pullback simpléctico),
          (iii) $|\mathrm{Pf}(\omega)|$ finito (cuantizabilidad, integrality proxy).
        Devuelve $\|\omega+\omega^{T}\|_F+\|M^{T}\omega M-\omega\|_F$.
        """
        skew = _fro(canonical_omega + canonical_omega.T)
        try:
            pullback = _fro(jacobian_M.T @ canonical_omega @ jacobian_M - canonical_omega)
        except (ValueError, la.LinAlgError):
            pullback = float("inf")
        return float(skew + pullback)

    def _polyakov_einstein_consistency(
        self,
        ricci_scalar: float,
        log_volume: float,
        n_dim: int,
    ) -> float:
        r"""
        Consistencia Polyakov / Einstein–Hilbert:
        en $n=2$, Gauss–Bonnet discreto $R\cdot\sqrt{\mathrm{Vol}}\approx 4\pi\chi$
        con $\chi=0$ (toro KAM) ⇒ residual $|R|\mathrm{e}^{\log\mathrm{Vol}/2}$.
        En $n\neq 2$ se reporta $0$ (no hay anomalía de Weyl 2-D).
        """
        if n_dim != 2:
            return 0.0
        sqrt_vol = float(np.exp(np.clip(0.5 * log_volume, -350.0, 350.0)))
        return abs(float(ricci_scalar) * sqrt_vol) / _GAUSS_BONNET_4PI

    # ── FASE 3.F — Cota de Lipschitz de Connes–Daleckii–Krein ────────────
    def _compute_lipschitz_connes_bound_exact(
        self, G_metric: NDArray[np.float64]
    ) -> float:
        r"""Cota DK exacta: $\max_{i<j}|f(\lambda_j)-f(\lambda_i)|/|\lambda_j-\lambda_i|$, $f(\lambda)=\lambda^{-1/2}$."""
        eigvals = np.sort(np.real(la.eigvalsh(0.5 * (G_metric + G_metric.T), check_finite=False)))
        eigvals = np.maximum(eigvals, _SPECTRAL_PSD_FLOOR)
        n = len(eigvals)
        if n < 2:
            return float(0.5 / (eigvals[0] ** 1.5))
        f_vals = 1.0 / np.sqrt(eigvals)
        max_derivative = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                delta_lam = eigvals[j] - eigvals[i]
                deriv = (
                    (0.5 / (eigvals[i] ** 1.5))
                    if delta_lam < 1e-12
                    else abs(f_vals[j] - f_vals[i]) / delta_lam
                )
                if deriv > max_derivative:
                    max_derivative = deriv
        return float(max_derivative)

    # ── FASE 3.G — Combinatoria de Heyting Ω₃ ────────────────────────────
    def _heyting_combine(
        self,
        is_coherent: bool,
        cond_final: float,
        kam_stable: bool,
        tangle: bool,
        floquet: float,
        banach_L: float,
        pairing_defect: float = 0.0,
        bianchi: float = 0.0,
        ein_scale: float = 1.0,
    ) -> HeytingVerdict:
        v_geom = HeytingVerdict.COHERENT if is_coherent else HeytingVerdict.DEGRADED
        if cond_final > self._condition_ceiling:
            v_cond = HeytingVerdict.VETOED
        elif cond_final > self._condition_ceiling * 0.1:
            v_cond = HeytingVerdict.DEGRADED
        else:
            v_cond = HeytingVerdict.COHERENT
        v_kam = HeytingVerdict.COHERENT if kam_stable else HeytingVerdict.DEGRADED
        v_tangle = HeytingVerdict.VETOED if tangle else HeytingVerdict.COHERENT
        v_floquet = (
            HeytingVerdict.DEGRADED if abs(floquet) > _FLOQUET_UNSTABLE else HeytingVerdict.COHERENT
        )
        v_banach = HeytingVerdict.COHERENT if banach_L < _BANACH_TARGET else HeytingVerdict.DEGRADED
        v_pair = (
            HeytingVerdict.DEGRADED if pairing_defect > _PAIRING_ATOL else HeytingVerdict.COHERENT
        )
        v_bianchi = (
            HeytingVerdict.DEGRADED
            if bianchi > _BIANCHI_REL_TOL * max(ein_scale, 1.0)
            else HeytingVerdict.COHERENT
        )
        return HeytingVerdict.fast_join_all(
            [v_geom, v_cond, v_kam, v_tangle, v_floquet, v_banach, v_pair, v_bianchi]
        )

    # ── FASE 3.H — Verificación de axiomas del retículo Heyting Ω₃ ───────
    def _heyting_lattice_axioms_check(self) -> bool:
        r"""
        Axiomas de retículo de Heyting en $\Omega_3$:
          1. Idempotencia, 2. Conmutatividad, 3. Asociatividad,
          4. Absorción, 5. Distributividad,
          6. Residuación: $a\wedge(a\to b)\le b$  (modus ponens),
          7. Orden total: $\omega$ es inmersión monótona,
          8. Adjunción: $a\wedge c\le b \iff c\le(a\to b)$.
        """
        elems = list(HeytingVerdict)
        ok = True
        for a in elems:
            if a.join(a) is not a or a.meet(a) is not a:
                ok = False
            for b in elems:
                if a.join(b) is not b.join(a) or a.meet(b) is not b.meet(a):
                    ok = False
                # 6. Residuación / modus ponens
                if not a.meet(a.implies(b)).leq(b):
                    ok = False
                for c in elems:
                    if a.join(b).join(c) is not a.join(b.join(c)):
                        ok = False
                    if a.meet(b).meet(c) is not a.meet(b.meet(c)):
                        ok = False
                    if a.join(a.meet(b)) is not a:
                        ok = False
                    if a.meet(a.join(b)) is not a:
                        ok = False
                    lhs = a.meet(b.join(c))
                    rhs = a.meet(b).join(a.meet(c))
                    if lhs is not rhs:
                        ok = False
                    # 8. Adjunción de la residuación
                    left = a.meet(c).leq(b)
                    right = c.leq(a.implies(b))
                    if left != right:
                        ok = False
        if not (
            HeytingVerdict.COHERENT.omega
            <= HeytingVerdict.DEGRADED.omega
            <= HeytingVerdict.VETOED.omega
        ):
            ok = False
        return ok

    # ── FASE 3.H′ — Asociatividad categórica del compuesto OODA ──────────
    def _ooda_associativity_check(
        self,
        phase1: Phase1Handoff,
        phase2: Phase2Handoff,
    ) -> bool:
        r"""
        Continuidad de jets del compuesto $\Phi_3\circ\Phi_2\circ\Phi_1$:
        el germen de fase 1 embebido en fase 2 coincide (misma métrica inicial,
        misma $\Delta\tau$, misma semilla de órbita) — i.e.
        $(\Phi_3\circ\Phi_2)\circ\Phi_1 \cong \Phi_3\circ(\Phi_2\circ\Phi_1)$.
        """
        try:
            p1_from_p2 = phase2.phase1
            if p1_from_p2.initial_metric.shape != phase1.initial_metric.shape:
                return False
            seed_ok = _fro(p1_from_p2.initial_metric - phase1.initial_metric) <= 1.0e-12
            tau_ok = abs(float(p1_from_p2.delta_tau) - float(phase1.delta_tau)) <= 1.0e-15
            orbit_ok = (len(p1_from_p2.orbit) >= 1) and (len(phase1.orbit) >= 1)
            return bool(seed_ok and tau_ok and orbit_ok)
        except Exception:  # noqa: BLE001
            return False

    # ── FASE 3.I — Auditoría completa OODA (compuesto Φ₃∘Φ₂∘Φ₁) ──────────
    def audit_riemannian_manifold_governance(
        self,
        G_metric: NDArray[np.float64],
        jacobian_M: NDArray[np.float64],
        canonical_omega: NDArray[np.float64],
        run_rsi3: bool = True,
        run_poincare: bool = True,
    ) -> MetricGovernanceCertificate:
        r"""
        Ciclo OODA completo $\Phi_3\circ\Phi_2\circ\Phi_1$ con verificación de
        MTA1–3, invariantes 1–4, Bianchi, Weyl, Killing, Kirchhoff, acción-ángulo
        Hamilton–Jacobi, Melnikov homoclínica, Conley–Morse y prequantum.
        """
        G_metric = self._validate_square(G_metric, "G_metric")
        jacobian_M = self._validate_square(jacobian_M, "jacobian_M")
        canonical_omega = self._validate_square(canonical_omega, "canonical_omega")

        # ══════════ FASE 1 ══════════
        G_init, _, _, profile_init = self._spectral_decomposition_psd(G_metric)
        if profile_init.min_eigenvalue < _SPECTRAL_PSD_FLOOR:
            logger.error("[VETO] Firma PSD rota: λ_min=%.3e", profile_init.min_eigenvalue)
            return self._emit_veto_certificate(profile_init, reason="PSD_SIGNATURE_VIOLATION")

        phase1 = self._handoff_to_phase2(
            G_init, n_warmup_steps=3 if run_rsi3 else 1, delta_tau=0.01
        )

        # ══════════ FASE 2 ══════════
        if run_rsi3:
            G_seed, _, _ = self._ingest_phase1_handoff(phase1)
            rsi_out = self._rsi3_driver(G_seed, n_iterations=self._rsi3_max_iterations)
            phase1.orbit = list(rsi_out["orbit"])
            phase1.ricci_scalars = list(rsi_out["curvature_history"])
            phase2 = self._handoff_to_phase3(phase1, rsi_out)
        else:
            rsi_out = {
                "orbit": phase1.orbit,
                "curvature_history": phase1.ricci_scalars,
                "contraction_history": [],
                "lipschitz_history": [],
                "delta_tau_history": [phase1.delta_tau],
                "eta_history": [0.0], "zeta_history": [0.0],
                "rsi_level_achieved": 1, "converged": False,
                "banach_L": self._banach_contraction_constant(phase1.orbit),
                "quadratic_rate": 0.0,
                "method_lineage": [f"{self._projector_mode.value}+{self._ricci_discretizer.value}"],
                "banach_certified": False,
                "projector_mode": self._projector_mode.value,
                "discretizer": self._ricci_discretizer.value,
            }
            phase2 = self._handoff_to_phase3(phase1, rsi_out)

        # ══════════ FASE 3 ══════════
        p1, poincare, lyap_max, omega_orb = self._ingest_phase2_handoff(phase2)
        ooda_ok = self._ooda_associativity_check(phase1, phase2)
        G_final = p1.orbit[-1]
        state_final = self._evaluate_geometry(G_final)
        profile_final = state_final.spectral_profile
        eigenvalues_final = la.eigvalsh(G_final, check_finite=False)

        kam_out = self._kam_torus_stability(
            eigenvalues_final,
            orbital_frequencies=np.asarray(omega_orb, dtype=np.float64),
            resonance_vectors=phase2.resonance_vectors,
        )
        kam_stable = bool(kam_out["is_kam_stable"] >= 0.5)

        dt_mean = float(np.mean(phase2.delta_tau_history)) if phase2.delta_tau_history else 0.01
        melnikov_amp = self._melnikov_homoclinic_integral(
            poincare.get("section_points", []), dt_mean, lyap_max
        )

        tangle = (
            self._detect_homoclinic_tangle(lyap_max, poincare, melnikov_amp)
            if run_poincare else False
        )
        hopf = self._poincare_hopf_index(p1.orbit) if run_poincare else 0.0
        floquet = float(poincare.get("floquet_multiplier", 0.0)) if run_poincare else 0.0
        conley = self._conley_morse_index(hopf, phase2.lyapunov_spectrum)

        l_max_lipschitz = self._compute_lipschitz_connes_bound_exact(G_final)

        det_M = float(np.real(la.det(jacobian_M)))
        volume_drift = abs(det_M - 1.0)
        omega_residual = _fro(jacobian_M.T @ canonical_omega @ jacobian_M - canonical_omega)
        omega_skew_defect = _fro(canonical_omega + canonical_omega.T)
        prequantum = self._geometric_quantization_prequantum(canonical_omega, jacobian_M)

        cond_final = float(profile_final.condition_number)
        symmetry_defect = _fro(G_final - 0.5 * (G_final + G_final.T))
        compat = float(state_final.compatibility_residual)
        bianchi = float(state_final.bianchi_residual)
        g_scale = max(_fro(G_final), 1.0)
        compat_ok = compat <= _COMPAT_REL_TOL * g_scale
        ein_scale = max(_fro(state_final.einstein_tensor), 1.0)
        polyakov = self._polyakov_einstein_consistency(
            state_final.ricci_scalar, state_final.log_volume, int(G_final.shape[0])
        )

        is_coherent = (
            symmetry_defect <= 1.0e-9
            and cond_final <= self._condition_ceiling
            and volume_drift <= _SYMPLECTIC_ATOL
            and omega_residual <= _SYMPLECTIC_ATOL
            and omega_skew_defect <= _SYMPLECTIC_ATOL
            and compat_ok
        )

        banach_L = float(phase2.banach_L)
        pairing_defect = float(phase2.lyapunov_pairing_defect)
        verdict = self._heyting_combine(
            is_coherent=is_coherent, cond_final=cond_final,
            kam_stable=kam_stable, tangle=tangle,
            floquet=floquet, banach_L=banach_L,
            pairing_defect=pairing_defect,
            bianchi=bianchi, ein_scale=ein_scale,
        )
        heyting_ok = self._heyting_lattice_axioms_check()

        if verdict is HeytingVerdict.VETOED:
            logger.error(
                "[VETO] κ=%.3e, drift=%.3e, ω_res=%.3e, tangle=%s, KAM_γ=%.3e, "
                "Floquet=%.3e, compat=%.3e, Bianchi=%.3e, Melnikov=%.3e, "
                "Weyl=%.3e, Killing=%.3e, KCL=%.3e",
                cond_final, volume_drift, omega_residual, tangle,
                kam_out["diophantine_gamma"], floquet, compat, bianchi, melnikov_amp,
                state_final.weyl_residual, state_final.killing_residual,
                state_final.kirchhoff_residual,
            )
        if not heyting_ok:
            logger.error("[HEYTING] Axiomas de retículo violados — ¡bug crítico!")
        if not ooda_ok:
            logger.error("[OODA] Asociatividad Φ₃∘Φ₂∘Φ₁ violada — handoff inconsistente")

        first_int = p1.first_integrals
        rsi_trace = RSI3Trace(
            rsi_level_achieved=int(rsi_out["rsi_level_achieved"]),
            contraction_constants=tuple(float(x) for x in rsi_out.get("contraction_history", [])),
            delta_tau_history=tuple(float(x) for x in rsi_out.get("delta_tau_history", [])),
            eta_history=tuple(float(x) for x in rsi_out.get("eta_history", [])),
            converged=bool(rsi_out.get("converged", False)),
            quadratic_convergence_rate=float(rsi_out.get("quadratic_rate", 0.0)),
            method_lineage=tuple(str(x) for x in rsi_out.get("method_lineage", [])),
            banach_certified=bool(rsi_out.get("banach_certified", False)),
            projector_mode=str(rsi_out.get("projector_mode", self._projector_mode.value)),
        )

        periods = poincare.get("return_periods", np.array([]))
        monodromy_rho = float(poincare.get("floquet_multiplier", 0.0))
        poincare_data = PoincareData(
            return_periods=tuple(float(x) for x in np.asarray(periods).ravel()),
            lyapunov_max=float(lyap_max),
            kam_stable=kam_stable,
            homoclinic_tangle=bool(tangle),
            hopf_index=float(hopf),
            diophantine_gamma=float(kam_out["diophantine_gamma"]),
            floquet_multiplier=floquet,
            dominant_frequencies=tuple(float(x) for x in omega_orb),
            action=float(phase2.action_angle[0]),
            angle=float(phase2.action_angle[1]),
            melnikov_amplitude=float(melnikov_amp),
            lindstedt_period=float(phase2.lindstedt_period),
            recurrence_bound=float(phase2.recurrence_bound),
            monodromy_spectral_radius=monodromy_rho,
            conley_index=float(conley),
            prequantum_curvature=float(prequantum),
            first_integrals=tuple(
                float(first_int.get(k, 0.0)) for k in ("H", "C_jacobi", "L_angular")
            ),
        )

        return MetricGovernanceCertificate(
            spectral_profile=profile_final,
            ricci_scalar_curvature=float(state_final.ricci_scalar),
            lipschitz_connes_bound=l_max_lipschitz,
            symplectic_volume_drift=volume_drift,
            heyting_verdict=verdict,
            is_geometrically_coherent=bool(is_coherent),
            rsi3_trace=rsi_trace,
            poincare_data=poincare_data,
            banach_contraction_constant=banach_L,
            log_volume=float(state_final.log_volume),
            einstein_hilbert_action=float(state_final.eh_action),
            metric_compatibility_residual=compat,
            bianchi_identity_residual=bianchi,
            lyapunov_spectrum=phase2.lyapunov_spectrum,
            action_angle=phase2.action_angle,
            melnikov_amplitude=float(melnikov_amp),
            heyting_axioms_verified=heyting_ok,
            weyl_residual=float(state_final.weyl_residual),
            killing_residual=float(state_final.killing_residual),
            kirchhoff_residual=float(state_final.kirchhoff_residual),
            volume_consistency=float(state_final.volume_consistency),
            lyapunov_pairing_defect=pairing_defect,
            generating_function_residual=float(phase2.generating_function_residual),
            ooda_associative=bool(ooda_ok),
            polyakov_residual=float(polyakov),
            method_lineage=phase2.method_lineage,
        )

    # ── FASE 3.J — Certificado de veto ───────────────────────────────────
    def _emit_veto_certificate(
        self, profile: SpectralProfile, reason: str
    ) -> MetricGovernanceCertificate:
        logger.error("[VETO-CERT] reason=%s, λ_min=%.3e", reason, profile.min_eigenvalue)
        return MetricGovernanceCertificate(
            spectral_profile=profile,
            ricci_scalar_curvature=0.0,
            lipschitz_connes_bound=float("inf"),
            symplectic_volume_drift=1.0,
            heyting_verdict=HeytingVerdict.VETOED,
            is_geometrically_coherent=False,
            rsi3_trace=RSI3Trace(0, (), (), (), False, 0.0),
            poincare_data=PoincareData((), float("inf"), False, False, 0.0, 0.0),
            banach_contraction_constant=float("inf"),
            log_volume=float("-inf"),
            einstein_hilbert_action=0.0,
            metric_compatibility_residual=float("inf"),
            bianchi_identity_residual=float("inf"),
            heyting_axioms_verified=True,
            weyl_residual=float("inf"),
            killing_residual=float("inf"),
            kirchhoff_residual=float("inf"),
            volume_consistency=float("inf"),
            ooda_associative=True,
        )

    # ── FASE 3.K — Serialización JSON ────────────────────────────────────
    def emit_governance_certificate(self, cert: MetricGovernanceCertificate) -> Dict[str, Any]:
        return {
            "module_version": __version__,
            "spectral_profile": {
                "min_eigenvalue": cert.spectral_profile.min_eigenvalue,
                "max_eigenvalue": cert.spectral_profile.max_eigenvalue,
                "condition_number": cert.spectral_profile.condition_number,
                "is_spd": cert.spectral_profile.is_spd,
            },
            "ricci_scalar_curvature": cert.ricci_scalar_curvature,
            "lipschitz_connes_bound": cert.lipschitz_connes_bound,
            "symplectic_volume_drift": cert.symplectic_volume_drift,
            "heyting_verdict": cert.heyting_verdict.value,
            "heyting_omega": cert.heyting_verdict.omega,
            "heyting_axioms_verified": cert.heyting_axioms_verified,
            "is_geometrically_coherent": cert.is_geometrically_coherent,
            "log_volume": cert.log_volume,
            "einstein_hilbert_action": cert.einstein_hilbert_action,
            "metric_compatibility_residual": cert.metric_compatibility_residual,
            "bianchi_identity_residual": cert.bianchi_identity_residual,
            "weyl_residual": cert.weyl_residual,
            "killing_residual": cert.killing_residual,
            "kirchhoff_residual": cert.kirchhoff_residual,
            "volume_consistency": cert.volume_consistency,
            "lyapunov_spectrum": list(cert.lyapunov_spectrum),
            "lyapunov_pairing_defect": cert.lyapunov_pairing_defect,
            "action_angle": list(cert.action_angle),
            "generating_function_residual": cert.generating_function_residual,
            "melnikov_amplitude": cert.melnikov_amplitude,
            "ooda_associative": cert.ooda_associative,
            "polyakov_residual": cert.polyakov_residual,
            "method_lineage": list(cert.method_lineage),
            "rsi3_trace": {
                "rsi_level_achieved": cert.rsi3_trace.rsi_level_achieved,
                "contraction_constants": list(cert.rsi3_trace.contraction_constants),
                "delta_tau_history": list(cert.rsi3_trace.delta_tau_history),
                "eta_history": list(cert.rsi3_trace.eta_history),
                "converged": cert.rsi3_trace.converged,
                "quadratic_convergence_rate": cert.rsi3_trace.quadratic_convergence_rate,
                "method_lineage": list(cert.rsi3_trace.method_lineage),
                "banach_certified": cert.rsi3_trace.banach_certified,
                "projector_mode": cert.rsi3_trace.projector_mode,
            },
            "poincare_data": {
                "return_periods": list(cert.poincare_data.return_periods),
                "lyapunov_max": cert.poincare_data.lyapunov_max,
                "kam_stable": cert.poincare_data.kam_stable,
                "homoclinic_tangle": cert.poincare_data.homoclinic_tangle,
                "hopf_index": cert.poincare_data.hopf_index,
                "diophantine_gamma": cert.poincare_data.diophantine_gamma,
                "floquet_multiplier": cert.poincare_data.floquet_multiplier,
                "dominant_frequencies": list(cert.poincare_data.dominant_frequencies),
                "action": cert.poincare_data.action,
                "angle": cert.poincare_data.angle,
                "melnikov_amplitude": cert.poincare_data.melnikov_amplitude,
                "lindstedt_period": cert.poincare_data.lindstedt_period,
                "recurrence_bound": cert.poincare_data.recurrence_bound,
                "monodromy_spectral_radius": cert.poincare_data.monodromy_spectral_radius,
                "conley_index": cert.poincare_data.conley_index,
                "prequantum_curvature": cert.poincare_data.prequantum_curvature,
                "first_integrals": list(cert.poincare_data.first_integrals),
            },
            "banach_contraction_constant": cert.banach_contraction_constant,
        }

    def emit_governance_certificate_json(
        self, cert: MetricGovernanceCertificate, indent: int = 2
    ) -> str:
        return json.dumps(self.emit_governance_certificate(cert), indent=indent)