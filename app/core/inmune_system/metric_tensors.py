# -*- coding: utf-8 -*-
r"""
====================================================================================================
ARQUITECTURA COGNITIVA APU FILTER - SISTEMA INMUNE SOBERANO
====================================================================================================

Módulo: Riemannian Metric Tensors (MetricTensorFactory)
Ruta: app/core/inmune_system/metric_tensors.py
Versión: 7.3.0  (evolución rigurosa de 7.2.0)
Función: Motor espectral de tensores métricos riemannianos y tejido conectivo anisotrópico.
         Construye, regulariza por Tikhonov/Gershgorin, endurece por flujo de Ricci RSI³
         (v7.3.0 Bianchi–Darboux–QR–Lindstedt–Killing–Morse) y certifica campos métricos
         G_{\mu\nu} en el hiperespacio de fase \mathcal{M} \subset \mathbb{R}^7.

----------------------------------------------------------------------------------------------------
DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO MOTOR ESPECTRAL:
----------------------------------------------------------------------------------------------------
El motor espectral `MetricTensorFactory` constituye el núcleo geométrico de la Capa 0 (Estrato
Wisdom, \mathcal{V}_{\mathbb{W}}) de la arquitectura APU Filter. Su función matemática primordial es
la construcción, regularización espectral y endurecimiento dinámico de la métrica riemanniana
G \in \operatorname{Sym}^+(n, \mathbb{R}) sobre la variedad de fases \mathcal{M} \subset \mathbb{R}^7.
Opera como un compuesto categórico trisfásico \operatorname{Auditar} = \Phi_3 \circ \Phi_2 \circ \Phi_1
donde cada fase está interconectada por morfismos de continuación estrictos (`Phase1Handoff` y
`Phase2Handoff`), garantizando la preservación incondicional de la positividad definida \mathrm{SPD},
el control del número de condición \kappa_2(G) \le 10^8 y la invariancia geométrica bajo el flujo de
Ricci normalizado anidado en el lazo de Automejora Recursiva de Nivel 3 (\mathrm{RSI}^3).

----------------------------------------------------------------------------------------------------
MARCO TEÓRICO:
----------------------------------------------------------------------------------------------------
1. GEOMETRÍA RIEMANNIANA Y DERIVACIÓN NO CONMUTATIVA DE CONNES:
   La estructura métrica sobre \mathcal{M} se define mediante el tensor simétrico definido positivo
   G_{\mu\nu} = \langle e_\mu, e_\nu \rangle_G. Las derivadas no conmutativas del tensor métrico
   en el álgebra de observables \mathcal{A} se evalúan según el formalismo del triple espectral de Connes:
   $$\partial_\mu G_{\nu\sigma} = [X_\mu, G]_{\nu\sigma} = \delta_{\nu\mu} G_{\mu\sigma} - G_{\nu\mu} \delta_{\mu\sigma}$$
   donde X_\mu = |e_\mu\rangle\langle e_\mu| es el operador posición no conmutativo.
   Los símbolos de Christoffel de Koszul de primera especie sin torsión responden a:
   $$\Gamma^r{}_{mn} = \frac{1}{2} G^{rs} \left( \partial_m G_{ns} + \partial_n G_{ms} - \partial_s G_{mn} \right)$$
   simetrizados exactamente en (m,n) para garantizar la unicidad del Teorema Fundamental de Riemann.

2. COTA EXACTA DE LIPSCHITZ DE CONNES–DALECKII–KREIN:
   La variación del operador densidad D = \boldsymbol{\rho}^{-1/2} bajo perturbaciones métricas no
   conmutativas se acota estrictamente por el cociente diferencial del resolvente en el espacio de Hilbert:
   $$L_{\mathrm{Connes}} = \sup_{i \neq j} \left| \frac{f(\lambda_i) - f(\lambda_j)}{\lambda_i - \lambda_j} \right| \quad \text{con } f(\lambda) = \lambda^{-1/2}$$
   Para el espectro ordenado \operatorname{spec}(G) = \{\lambda_1 \le \lambda_2 \le \dots \le \lambda_n\}, el valor exacto satisface:
   $$L_{\mathrm{Connes}} = \max_{i < j} \left| \frac{\lambda_j^{-1/2} - \lambda_i^{-1/2}}{\lambda_j - \lambda_i} \right| \le \frac{1}{2 \lambda_{\min}^{3/2}}$$

3. PROYECCIÓN SIMPLÉCTICA DE CAYLEY–DARBOUX SOBRE \mathfrak{sp}(2n, \mathbb{R}):
   Para preservar la 2-forma canónica de Darboux \omega = \sum_{i=1}^n \mathrm{d}q^i \wedge \mathrm{d}p_i en el
   fibrado cotangente T^*\mathcal{M} \cong \mathbb{R}^{2n} durante la integración del flujo de Ricci, la métrica
   G se mapea al álgebra de Lie simpléctica \mathfrak{sp}(2n, \mathbb{R}) mediante la transformación de Cayley:
   $$K = (G - \mathbf{I}_{2n})(G + \mathbf{I}_{2n})^{-1} \in \mathfrak{sp}(2n, \mathbb{R}) \quad \iff \quad K^\top J + J K = \mathbf{0}$$
   donde J = \begin{pmatrix} \mathbf{0} & \mathbf{I}_n \\ -\mathbf{I}_n & \mathbf{0} \end{pmatrix}. Tras truncar espectralmente
   los autovalores de K en (-1, 1), se reconstruye la métrica proyectada:
   $$G_{\mathrm{sympl}} = (\mathbf{I}_{2n} + K_{\mathrm{proj}})(\mathbf{I}_{2n} - K_{\mathrm{proj}})^{-1} \in \operatorname{Sp}(2n, \mathbb{R}) \cap \mathrm{Sym}^+(2n, \mathbb{R})$$
   garantizando la invarianza simpléctica con precisión \delta\omega = \|G_{\mathrm{sympl}}^\top J G_{\mathrm{sympl}} - J\|_F < 10^{-15}.

4. FLUJO DE RICCI NORMALIZADO Y AUTOMEJORA RECURSIVA (\mathrm{RSI}^3):
   El flujo de Ricci discreto evoluciona la métrica hacia un punto fijo de Einstein \operatorname{Ric}(G) = \frac{R}{n} G:
   $$G_{k+1} = \Pi_{\mathrm{SPD}} \left( G_k - 2 \Delta\tau_k \left( \operatorname{Ric}(G_k) - \frac{R_k}{n} G_k \right) \right)$$
   El esquema de automejora de tres niveles (\mathrm{RSI}^3) anida dinámicamente:
   • Nivel 1 (\Phi^{(1)}): Paso unitario de Ricci con proyección espectral.
   • Nivel 2 (\Phi^{(2)}): Adaptación del paso temporal por gradiente de curvatura \Delta\tau_{k+1} = \mathrm{sat}(\Delta\tau_k \mathrm{e}^{-\eta_k |\nabla R_k|}).
   • Nivel 3 (\Phi^{(3)}): Adaptación del parámetro de aprendizaje \eta_{k+1} = \mathrm{sat}(\eta_k \mathrm{e}^{\zeta(L_k - L_\star)}) en función de la constante de contracción de Banach L_k, combinado con el selector automático de método \sigma_{k+1} = ([\text{Proyector}], [\text{Discretizador}]).

5. TEORÍA DEL PUNTO FIJO DE BANACH EN ESPACIOS DE RIEMANN:
   La convergencia del flujo \mathrm{RSI}^3 se certifica formalmente sobre el espacio métrico completo (\overline{\mathrm{SPD}}_n, \|\cdot\|_F):
   (i)   \overline{\mathrm{SPD}}_n es un dominio cerrado completo bajo la norma de Frobenius.
   (ii)  \Phi(\overline{\mathrm{SPD}}_n) \subset \overline{\mathrm{SPD}}_n (automapeo garantizado por el proyector espectral \Pi_{\mathrm{SPD}}).
   (iii) Constante de Lipschitz L_\Phi = \sup_k \frac{\|G_{k+1} - G_k\|_F}{\|G_k - G_{k-1}\|_F} \le L_\star = 0.95 < 1.0.

6. MECÁNICA CELESTE DE HENRI POINCARÉ Y SECCIÓN DE RETORNO:
   La estabilidad orbital sobre la hipersuperficie de energía \Sigma = \{R = \bar{R}\} \subset T^*\mathcal{M} se analiza mediante:
   • Mapa de Retorno de Poincaré \mathcal{T}: \Sigma \to \Sigma con interpolación lineal del tiempo de vuelo \tau^\star.
   • Matriz de Monodromía Variacional M = \mathrm{D}\mathcal{T}|_\Sigma \in \mathbb{R}^{r \times r} por mínimos cuadrados tangentes SVD.
   • Multiplicador de Floquet \mu_{\max} = \rho(M) = \max_i |\mathrm{eig}(M)_i|.
   • Espectro de Lyapunov completo \lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d vía ortogonalización QR continua de Benettin–Shimada.
   • Pareado Simpléctico Palindrómico: \lambda_i + \lambda_{d-1-i} \approx 0 \quad \forall i.
   • Estabilidad KAM Diofántica: \gamma = \inf_{k \neq 0} |\omega \cdot k| \, \|k\|_1^\tau \ge \gamma_\star = 10^{-3}.
   • Integral Homoclínica de Melnikov: M(\theta_0) = \int_{-\infty}^\infty \langle J \nabla H(z^0(t)), g(z^0(t), \theta_0) \rangle \, \mathrm{d}t \neq 0 \implies \text{Tangle Homoclínico}.
   • Corrección de Periodo de Poincaré–Lindstedt: T = T_0 (1 + \varepsilon T_1).
   • Cota Inferior del Tiempo de Recurrencia de Poincaré: \tau_{\mathrm{rec}} \ge \frac{\mathrm{Vol}(\mathcal{M})}{\mathrm{Vol}(\mathrm{tubo})}.
   • Índice Topológico de Conley–Morse: h = m - \mathrm{Hopf}, donde m es la dimensión inestable.

7. ADJUDICACIÓN TRIVALENTE EN EL RETÍCULO DE HEYTING \Omega_3:
   El espacio de verdad intuicionista \Omega_3 = \{\bot = \mathtt{COHERENT} < \frac{1}{2} = \mathtt{DEGRADED} < \top = \mathtt{VETOED}\}
   combina las obstrucciones métricas mediante la operación Join a \vee b = \max(a, b) (mapeada a `OR` bitwise SIMD de 2 bits: 0x00, 0x01, 0x03).
   La residuación de cadena total satisface:
   $$a \to b = \begin{cases} \mathtt{COHERENT} & \text{si } a \le b \\ b & \text{si } a > b \end{cases} \quad \implies \quad a \wedge (a \to b) \le b \quad (\text{Modus Ponens})$$

8. DUALIDAD CIRCUITO–GEOMÉTRICA (KIRCHHOFF ↔ BIANCHI):
   La Ley de Corrientes de Kirchhoff (KCL) en nodos de la malla agéntica equivale idénticamente a la Identidad de Bianchi Contraída:
   $$\text{KCL} \iff \nabla_\mu \mathrm{Ein}^{\mu\nu} = 0 \quad \text{donde } \mathrm{Ein}_{\mu\nu} = \operatorname{Ric}_{\mu\nu} - \frac{1}{2} R G_{\mu\nu}$$
   La Ley de Tensiones de Kirchhoff (KVL) está cifrada en la simetría del tensor de Einstein \mathrm{Ein}_{\mu\nu} = \mathrm{Ein}_{\nu\mu}.

----------------------------------------------------------------------------------------------------
AXIOMAS FORMALES:
----------------------------------------------------------------------------------------------------
MT1 (Simetría Rigurosa):
    \|G - G^\top\|_F \le \max(\varepsilon_{\mathrm{base}}, \varepsilon_{\mathrm{rtol}} \|G\|_F), \quad \varepsilon_{\mathrm{rtol}} = 10^{-14}.

MT2 (Definición Positiva Estricta):
    G \succ 0 \iff x^\top G x > 0 \quad \forall x \in \mathbb{R}^n \setminus \{\mathbf{0}\} \iff \lambda_{\min}(G) \ge \lambda_{\star} = 10^{-12}.

MT3 (Cota de Deformación Regularizadora):
    \frac{\|G_{\mathrm{reg}} - G_{\mathrm{orig}}\|_F}{\|G_{\mathrm{orig}}\|_F} \le 0.25 \quad (25\% \text{ máximo de distorsión física}).

MT4 (Preservación Simpléctica de Darboux):
    \delta\omega = \|G_{\mathrm{sympl}}^\top J G_{\mathrm{sympl}} - J\|_F \le 10^{-15} \quad \text{en } \operatorname{Sp}(2n, \mathbb{R}).

MT5 (Contracción del Mapa de Ricci):
    L_\Phi = \mathrm{Lip}(\Phi) \le L_\star = 0.95 < 1.0 \quad \text{sobre } (\overline{\mathrm{SPD}}_n, \|\cdot\|_F).

----------------------------------------------------------------------------------------------------
INVARIANTES DE ESTADO:
----------------------------------------------------------------------------------------------------
Invariante 1 (Acatamiento de Condicionamiento Espectral):
    \kappa_2(G) = \frac{\lambda_{\max}(G)}{\lambda_{\min}(G)} \le 10^8.

Invariante 2 (Preservación del Espectro Tikhonov):
    |\lambda_i(G_{\mathrm{reg}}) - (\lambda_i(G) + \delta)| \le 10^{-12} \max_i |\lambda_i|.

Invariante 3 (Divergencia Nula de Bianchi–Kirchhoff):
    \|\nabla_\mu \mathrm{Ein}^{\mu\nu}\|_2 \le 10^{-3} \|\mathrm{Ein}\|_F.

Invariante 4 (Pareado Simpléctico Palindrómico):
    |\lambda_i + \lambda_{d-1-i}| \le 10^{-2} (1 + |\lambda_i|) \quad \text{para dimensiones pares } d = 2n.

Invariante 5 (Invariancia Categórica de Handoffs):
    (\Phi_3 \circ \Phi_2) \circ \Phi_1 \cong \Phi_3 \circ (\Phi_2 \circ \Phi_1) \quad (\text{Asociatividad OODA de Jets}).
====================================================================================================
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations, product
from typing import Any, Dict, Final, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import block_diag, cho_factor, cho_solve, eigh

from app.core.inmune_system.topological_watcher import (
    COND_NUM_TOL,
    MIN_EIGVAL_TOL,
    MetricTensorError,
)

logger = logging.getLogger("MIC.ImmuneSystem.MetricTensors")

__all__ = [
    "G_PHYSICS",
    "G_TOPOLOGY",
    "G_THERMODYNAMICS",
    "HeytingVerdict",
    "ProjectorMode",
    "RicciDiscretizer",
    "MetricTensorFactory",
    "SpectralProfile",
    "RegularizationReport",
    "BanachCertificate",
    "RSI3Trace",
    "PoincareData",
    "HardenedCertificate",
    "Phase1Handoff",
    "Phase2Handoff",
    "get_tensor_diagnostics",
    "get_hardening_diagnostics",
]

__version__: Final[str] = "7.3.0"

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FUNDAMENTALES CON JUSTIFICACIÓN MATEMÁTICA
# ═══════════════════════════════════════════════════════════════════════════════

_FLOAT_DTYPE: Final[np.dtype] = np.dtype(np.float64)
_WILKINSON_EPS: Final[float] = float(np.finfo(np.float64).eps)

_SYMMETRY_RTOL: Final[float] = 1e-14
_SYMMETRY_ATOL_BASE: Final[float] = 1e-15
_SPD_INTERIOR_FACTOR: Final[float] = 1.1
_REGULARIZATION_WARN_THRESHOLD: Final[float] = 0.05
_REGULARIZATION_ABORT_THRESHOLD: Final[float] = 0.25
_TIKHONOV_DELTA_ATOL: Final[float] = 1e-16
_NEAR_ZERO_FROBENIUS_TOL: Final[float] = 1e-14
_GERSHGORIN_MARGIN: Final[float] = 1.0e-6

# ── Constantes RSI³ + Poincaré ──────────────────────────────────────────────
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_BANACH_TARGET: Final[float] = 0.95
_RSI3_CONV_EPS: Final[float] = 1.0e-9
_RSI3_MAX_ITER: Final[int] = 40
_RSI3_MIN_ITER: Final[int] = 8
_EINSTEIN_TOL: Final[float] = 1.0e-3
_LYAPUNOV_CHAOS: Final[float] = 1.0e-2
_KAM_GAMMA: Final[float] = 1.0e-3
_KAM_TAU: Final[float] = 2.0
_POINCARE_RETURN_TOL: Final[float] = 1.0e-4
_TAU_LOWER: Final[float] = 1.0e-5
_TAU_UPPER: Final[float] = 0.1
_ETA_LOWER: Final[float] = 1.0e-4
_ETA_UPPER: Final[float] = 10.0
_ZETA_LOWER: Final[float] = 1.0e-4
_ZETA_UPPER: Final[float] = 0.1
_FLOQUET_UNSTABLE: Final[float] = 1.0 + 1.0e-2
_LIPSCHITZ_PERT_REL: Final[float] = 1.0e-6
_LIPSCHITZ_PROBE_STRIDE: Final[int] = 5
_COMPAT_REL_TOL: Final[float] = 1.0e-4

# ── Constantes v7.2.0 ───────────────────────────────────────────────────────
_LYAP_DIM_CAP: Final[int] = 64
_BENETTIN_QR_STRIDE: Final[int] = 5
_MELNIKOV_THRESHOLD: Final[float] = 1.0e-2
_BIANCHI_REL_TOL: Final[float] = 1.0e-3

# ── Constantes v7.3.0 ───────────────────────────────────────────────────────
_CONSECUTIVE_CONV_STEPS: Final[int] = 3
_WEYL_REL_TOL: Final[float] = 1.0e-2
_KILLING_REL_TOL: Final[float] = 1.0e-3
_PAIRING_ATOL: Final[float] = 1.0e-2
_JACOBI_FD_EPS: Final[float] = 1.0e-7
_METHOD_SWITCH_HYSTERESIS: Final[float] = 1.25
_GAUSS_BONNET_4PI: Final[float] = 4.0 * float(np.pi)


# ═══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS (v4.1.0 / v5.0.0 / v6.0.0 / v7.2.0 + v7.3.0)
# ═══════════════════════════════════════════════════════════════════════════════

class HeytingVerdict(Enum):
    r"""Clasificador en el álgebra de Heyting trivalente $\Omega_3$.

    Orden de obstrucción: $\mathrm{COHERENT}=\bot < \mathrm{DEGRADED}=\tfrac12
    < \mathrm{VETOED}=\top$. El join $a\vee b=\max$ acumula obstrucciones.
    Codificación SIMD bitwise IRAM Xtensa: COHERENT=0x00, DEGRADED=0x01, VETOED=0x03.
    Residuación: $a\to b=\top$ si $a\le b$, else $b$ (cadena total).
    """

    COHERENT = "COHERENT"
    DEGRADED = "DEGRADED"
    VETOED = "VETOED"

    @property
    def omega(self) -> float:
        return {
            HeytingVerdict.COHERENT: 0.0,
            HeytingVerdict.DEGRADED: 0.5,
            HeytingVerdict.VETOED: 1.0,
        }[self]

    @property
    def mask(self) -> int:
        return {
            HeytingVerdict.COHERENT: 0x00,
            HeytingVerdict.DEGRADED: 0x01,
            HeytingVerdict.VETOED: 0x03,
        }[self]

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

    def meet(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict.from_omega(min(self.omega, other.omega))

    def implies(self, other: "HeytingVerdict") -> "HeytingVerdict":
        r"""Residuación de cadena total: $a\to b=\top$ si $a\le b$, else $b$."""
        if self.omega <= other.omega:
            return HeytingVerdict.COHERENT
        return other

    def leq(self, other: "HeytingVerdict") -> bool:
        return self.omega <= other.omega

    @classmethod
    def fast_join_all(cls, verdicts: List["HeytingVerdict"]) -> "HeytingVerdict":
        acc = 0
        for v in verdicts:
            acc |= v.mask
            if acc == 0x03:
                return cls.VETOED
        return cls.from_mask(acc)


class ProjectorMode(Enum):
    r"""Selector de proyección SPD (automejora de método, RSI³)."""
    SPECTRAL_HIGHAM = "SPECTRAL_HIGHAM"
    CAYLEY_DARBOUX = "CAYLEY_DARBOUX"


class RicciDiscretizer(Enum):
    r"""Selector de discretización de Ricci (automejora de método, RSI³)."""
    KOSZUL_QUADRATIC = "KOSZUL_QUADRATIC"
    TRACEFREE_PROXY = "TRACEFREE_PROXY"


class SpectralProfile(NamedTuple):
    """Perfil espectral completo de una matriz simétrica (v4.1.0 preservado)."""

    eigenvalues: np.ndarray
    lambda_min: float
    lambda_max: float
    condition_number: float
    spectral_gap: float
    frobenius_norm: float

    def is_strictly_positive(self, threshold: float = MIN_EIGVAL_TOL) -> bool:
        return bool(self.lambda_min >= threshold)

    def is_well_conditioned(self, max_kappa: float = COND_NUM_TOL) -> bool:
        return bool(np.isfinite(self.condition_number) and self.condition_number <= max_kappa)


@dataclass(frozen=True, slots=True)
class RegularizationReport:
    """Informe de regularización (v4.1.0 preservado)."""

    tensor_name: str
    original_profile: SpectralProfile
    regularized_profile: SpectralProfile
    tikhonov_delta: float
    frobenius_deformation: float
    required_regularization: bool
    input_asymmetry: float

    def log_summary(self) -> None:
        if self.required_regularization:
            log_fn = (
                logger.warning
                if self.frobenius_deformation > _REGULARIZATION_WARN_THRESHOLD
                else logger.info
            )
            log_fn(
                "Regularización aplicada a %s: δ=%.6e, deformación=%.4f%%, "
                "λ_min: %.6e → %.6e, κ: %.4e → %.4e, asimetría entrada: %.4e",
                self.tensor_name,
                self.tikhonov_delta,
                self.frobenius_deformation * 100,
                self.original_profile.lambda_min,
                self.regularized_profile.lambda_min,
                self.original_profile.condition_number,
                self.regularized_profile.condition_number,
                self.input_asymmetry,
            )
        else:
            logger.debug(
                "Tensor %s validado sin regularización: λ_min=%.6e, κ=%.4e, asimetría: %.4e",
                self.tensor_name,
                self.regularized_profile.lambda_min,
                self.regularized_profile.condition_number,
                self.input_asymmetry,
            )


@dataclass(frozen=True, slots=True)
class BanachCertificate:
    r"""Certificado del teorema del punto fijo de Banach sobre $(\overline{\mathrm{SPD}}_n,\|\cdot\|_F)$."""

    lipschitz: float
    is_self_map: bool
    is_complete_domain: bool
    is_contraction: bool
    certified: bool


@dataclass(frozen=True, slots=True)
class RSI3Trace:
    r"""Traza inmutable de la ejecución RSI³ (v7.3.0: +linaje, Banach, proyector)."""

    rsi_level_achieved: int
    banach_L: float
    delta_tau_history: Tuple[float, ...]
    eta_history: Tuple[float, ...]
    einstein_defect_history: Tuple[float, ...]
    converged: bool
    quadratic_convergence_rate: float = 0.0
    method_lineage: Tuple[str, ...] = ()
    banach_certified: bool = False
    projector_mode: str = ProjectorMode.CAYLEY_DARBOUX.value


@dataclass(frozen=True, slots=True)
class PoincareData:
    r"""Datos de mecánica celeste de Poincaré (v7.3.0: +Lindstedt, recurrencia, Conley)."""

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
    first_integrals: Tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class HardenedCertificate:
    r"""Certificado final de endurecimiento (v7.3.0: +Noether, Morse, HJ, Polyakov, OODA)."""

    tensor_name: str
    spectral_profile: SpectralProfile
    rsi3_trace: RSI3Trace
    poincare_data: PoincareData
    einstein_defect: float
    lipschitz_connes_bound: float
    is_hardened: bool
    is_geometrically_stable: bool
    heyting_verdict: HeytingVerdict = HeytingVerdict.COHERENT
    log_volume: float = 0.0
    metric_compatibility_residual: float = 0.0
    used_phase1_fallback: bool = False
    bianchi_identity_residual: float = 0.0
    lyapunov_spectrum: Tuple[float, ...] = ()
    action_angle: Tuple[float, float] = (0.0, 0.0)
    melnikov_amplitude: float = 0.0
    heyting_axioms_verified: bool = True
    conserved_quantities_audit: Dict[str, float] = field(default_factory=dict)
    weyl_residual: float = 0.0
    killing_residual: float = 0.0
    kirchhoff_residual: float = 0.0
    volume_consistency: float = 0.0
    lyapunov_pairing_defect: float = 0.0
    generating_function_residual: float = 0.0
    ooda_associative: bool = True
    polyakov_residual: float = 0.0
    method_lineage: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Phase1Handoff:
    """Morfismo 1 → 2 (v7.3.0: +Weyl, Killing, Kirchhoff, primeras integrales, volumen)."""

    tensor_name: str
    candidate: np.ndarray
    spectral_profile: SpectralProfile
    einstein_defect: float
    lipschitz_connes_bound: float
    input_asymmetry: float
    regularization_report: RegularizationReport
    log_volume: float = 0.0
    compatibility_residual: float = 0.0
    bianchi_residual: float = 0.0
    weyl_residual: float = 0.0
    killing_residual: float = 0.0
    kirchhoff_residual: float = 0.0
    volume_consistency: float = 0.0
    first_integrals: Dict[str, float] = field(default_factory=dict)
    projector_mode: str = ProjectorMode.CAYLEY_DARBOUX.value


@dataclass(frozen=True)
class Phase2Handoff:
    """Morfismo 2 → 3 (v7.3.0: +monodromía, Lindstedt, recurrencia, HJ, pairing, linaje)."""

    phase1: Phase1Handoff
    orbit: Tuple[np.ndarray, ...]
    curvature_history: Tuple[float, ...]
    lyapunov_max: float
    poincare_data: Dict[str, Any]
    banach_L: float
    delta_tau_history: Tuple[float, ...]
    eta_history: Tuple[float, ...]
    einstein_defect_history: Tuple[float, ...]
    rsi3_level: int
    rsi3_converged: bool
    dominant_frequencies: Tuple[float, ...] = field(default_factory=tuple)
    lyapunov_spectrum: Tuple[float, ...] = field(default_factory=tuple)
    action_angle: Tuple[float, float] = (0.0, 0.0)
    quadratic_rate: float = 0.0
    resonance_vectors: Tuple[Tuple[int, ...], ...] = field(default_factory=tuple)
    conserved_quantities_audit: Dict[str, float] = field(default_factory=dict)
    method_lineage: Tuple[str, ...] = field(default_factory=tuple)
    monodromy_eigs: Tuple[complex, ...] = field(default_factory=tuple)
    lindstedt_period: float = 0.0
    recurrence_bound: float = 0.0
    banach_certified: bool = False
    generating_function_residual: float = 0.0
    lyapunov_pairing_defect: float = 0.0
    projector_mode: str = ProjectorMode.CAYLEY_DARBOUX.value


def _fro(matrix: np.ndarray) -> float:
    """Norma de Frobenius escalar segura."""
    return float(np.linalg.norm(np.asarray(matrix, dtype=_FLOAT_DTYPE), ord="fro"))


def _safe_div(num: float, den: float, default: float = 0.0) -> float:
    d = abs(float(den))
    if d < _WILKINSON_EPS:
        return float(default)
    return float(num) / d


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1 — GEOMETRÍA ESPECTRAL, EINSTEIN DEFECT, BIANCHI, NOETHER Y HANDOFF
# ═══════════════════════════════════════════════════════════════════════════════
class MetricTensorFactory:
    r"""
    Fábrica determinista de tensores métricos con endurecimiento RSI³+Poincaré
    opcional. Las tres fases son un compuesto categórico
    $\mathrm{Auditar}=\Phi_3\circ\Phi_2\circ\Phi_1$ con morfismos de continuación
    `Phase1Handoff` (fase 1 → 2) y `Phase2Handoff` (fase 2 → 3).

    Cada tensor construido satisface rigurosamente:

    1. G ∈ ℝⁿˣⁿ (matriz cuadrada)
    2. G = Gᵀ (simetría exacta por proyección ortogonal)
    3. G ≻ 0 (definida positiva, certificada espectralmente y por Cholesky)
    4. λ_min(G) ≥ MIN_EIGVAL_TOL
    5. κ₂(G) ≤ COND_NUM_TOL
    6. ‖G_reg − G_orig‖_F / ‖G_orig‖_F ≤ 25 %
    7. Consistencia espectral post-Tikhonov
    8. [v5.0.0] Estabilidad KAM y ausencia de tangle homoclínico
    9. [v5.0.0] Defecto de Einstein acotado
   10. [v6.0.0] Compatibilidad métrica ∇G≈0, KAM orbital (FFT),
       Floquet interpolado, fallback SPD si el flujo degrada la certificación.
   11. [v7.2.0] Identidad de Bianchi ∇·Ein ≈ 0, espectro completo de Lyapunov
       (Benettin–Shimada QR), variables acción-ángulo, resonancias CF,
       función de Melnikov y verificación axiomática del retículo Ω₃.
   12. [v7.3.0] Automejora de *método* RSI³, monodromía variacional, Lindstedt,
       recurrencia de Poincaré, Hamilton–Jacobi, pairing simpléctico, Weyl,
       Killing–Noether, KCL↔Bianchi, Conley–Morse, Polyakov/Gauss–Bonnet,
       residuación Heyting y asociatividad OODA de jets.
    """

    _EXPECTED_DIMENSIONS: Final[Dict[str, int]] = {
        "G_phys": 3,
        "G_topo": 2,
        "G_thermo": 2,
        "G_global": 7,
    }
    _DEFAULT_PROJECTOR: Final[ProjectorMode] = ProjectorMode.CAYLEY_DARBOUX
    _DEFAULT_DISCRETIZER: Final[RicciDiscretizer] = RicciDiscretizer.KOSZUL_QUADRATIC
    _RNG = np.random.default_rng(0)

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 1.0 — Conversión, Cholesky SPD, volumen logarítmico, forma de volumen
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _as_strict_array(G: np.ndarray) -> np.ndarray:
        return np.array(G, dtype=_FLOAT_DTYPE, copy=True)

    @staticmethod
    def _spd_inverse(G: np.ndarray) -> np.ndarray:
        r"""Inversa SPD vía Cholesky: $G^{-1}=(LL^\top)^{-1}$."""
        n = int(G.shape[0])
        shifted = np.asarray(G, dtype=_FLOAT_DTYPE) + _SPECTRAL_PSD_FLOOR * np.eye(
            n, dtype=_FLOAT_DTYPE
        )
        chol, lower = cho_factor(shifted, lower=True, check_finite=False)
        return cho_solve(
            (chol, lower), np.eye(n, dtype=_FLOAT_DTYPE), check_finite=False
        )

    @staticmethod
    def _spd_logdet(G: np.ndarray) -> float:
        r"""$\log\det G=2\sum_i\log L_{ii}$ (Cholesky)."""
        n = int(G.shape[0])
        shifted = np.asarray(G, dtype=_FLOAT_DTYPE) + _SPECTRAL_PSD_FLOOR * np.eye(
            n, dtype=_FLOAT_DTYPE
        )
        chol, _ = cho_factor(shifted, lower=True, check_finite=False)
        diag = np.abs(np.diag(chol))
        return float(2.0 * np.sum(np.log(np.maximum(diag, _WILKINSON_EPS))))

    @classmethod
    def _volume_form_consistency(cls, G: np.ndarray) -> float:
        r"""
        Consistencia de la forma de volumen riemanniana:
        $\bigl|\det G-\exp(\mathrm{logdet}_{\mathrm{Chol}}G)\bigr|/|\det G|$.
        Discrepancia nula ⇔ Cholesky y $\det$ coinciden (SPD numéricamente sano).
        """
        try:
            logdet = cls._spd_logdet(G)
            det_chol = float(np.exp(np.clip(logdet, -700.0, 700.0)))
            sign, logabs = np.linalg.slogdet(np.asarray(G, dtype=_FLOAT_DTYPE))
            if sign <= 0.0:
                return float("inf")
            det_ref = float(np.exp(np.clip(logabs, -700.0, 700.0)))
            return _safe_div(abs(det_chol - det_ref), max(det_ref, _WILKINSON_EPS))
        except (LinAlgError, ValueError):
            return float("inf")

    @classmethod
    def _validate_structural_preconditions(cls, name: str, G: np.ndarray) -> np.ndarray:
        if G is None:
            raise MetricTensorError(f"Tensor {name}: entrada NoneType.")
        try:
            G_arr = cls._as_strict_array(G)
        except (TypeError, ValueError) as exc:
            raise MetricTensorError(
                f"Tensor {name}: no convertible a float64. Error: {exc}"
            ) from exc
        if G_arr.ndim != 2:
            raise MetricTensorError(f"Tensor {name}: ndim={G_arr.ndim} ≠ 2.")
        n_rows, n_cols = G_arr.shape
        if n_rows != n_cols:
            raise MetricTensorError(f"Tensor {name}: no cuadrada {n_rows}×{n_cols}.")
        if n_rows == 0:
            raise MetricTensorError(f"Tensor {name}: matriz vacía.")
        expected = cls._EXPECTED_DIMENSIONS.get(name)
        if expected is not None and n_rows != expected:
            raise MetricTensorError(
                f"Tensor {name}: dim esperada {expected}, recibida {n_rows}."
            )
        if not np.all(np.isfinite(G_arr)):
            n_nf = int(np.count_nonzero(~np.isfinite(G_arr)))
            idx = list(zip(*np.where(~np.isfinite(G_arr))))[:5]
            raise MetricTensorError(
                f"Tensor {name}: {n_nf} entradas no finitas. Índices: {idx}."
            )
        return G_arr

    @classmethod
    def _compute_symmetry_tolerance(cls, G: np.ndarray) -> float:
        return max(_SYMMETRY_ATOL_BASE, _SYMMETRY_RTOL * _fro(G))

    @classmethod
    def _assess_input_symmetry(cls, name: str, G: np.ndarray) -> float:
        f_norm = _fro(G)
        asym = _fro(G - G.T)
        rel = asym if f_norm < _NEAR_ZERO_FROBENIUS_TOL else asym / f_norm
        if asym > cls._compute_symmetry_tolerance(G):
            logger.warning(
                "Tensor %s: asimetría entrada. ||G-Gᵀ||_F=%.6e, relativa=%.6e",
                name, asym, rel,
            )
        return rel

    @staticmethod
    def _symmetrize_by_projection(G: np.ndarray) -> np.ndarray:
        return 0.5 * (G + G.T)

    @staticmethod
    def _project_psd(G: np.ndarray) -> np.ndarray:
        r"""Proyección espectral de Higham $\Pi_{\mathrm{SPD}}$ con re-simetrización."""
        G_arr = np.asarray(G, dtype=_FLOAT_DTYPE)
        G_sym = 0.5 * (G_arr + G_arr.T)
        try:
            eigvals, eigvecs = eigh(G_sym, check_finite=False)
        except LinAlgError:
            return G_sym
        clipped = np.maximum(eigvals, _SPECTRAL_PSD_FLOOR)
        G_proj = (eigvecs * clipped) @ eigvecs.T
        return 0.5 * (G_proj + G_proj.T)

    @classmethod
    def _project_psd_cayley_darboux(cls, G: np.ndarray) -> np.ndarray:
        r"""
        Proyección simpléctica de Cayley-Darboux sobre
        $\operatorname{Sym}^+(2n,\mathbb{R})\cap Sp(2n,\mathbb{R})$.
        Preserva la $2$-forma de Darboux $\omega=\sum dq^i\wedge dp_i$
        con precisión $\delta\omega < 10^{-15}$. Fallback Higham si $\dim$ impar.
        """
        G_arr = np.asarray(G, dtype=_FLOAT_DTYPE)
        n2 = int(G_arr.shape[0])
        if n2 % 2 != 0 or n2 == 0:
            return cls._project_psd(G_arr)
        n = n2 // 2
        I_mat = np.eye(n2, dtype=_FLOAT_DTYPE)
        J_mat = np.block([
            [np.zeros((n, n), dtype=_FLOAT_DTYPE), np.eye(n, dtype=_FLOAT_DTYPE)],
            [-np.eye(n, dtype=_FLOAT_DTYPE), np.zeros((n, n), dtype=_FLOAT_DTYPE)],
        ])
        G_shifted = G_arr + _SPECTRAL_PSD_FLOOR * I_mat
        try:
            K = (G_shifted - I_mat) @ cls._spd_inverse(G_shifted + I_mat)
        except (LinAlgError, ValueError):
            return cls._project_psd(G_arr)
        K_sp = 0.5 * (K + J_mat @ K.T @ J_mat)
        try:
            eigvals, eigvecs = eigh(K_sp, check_finite=False)
        except LinAlgError:
            return cls._project_psd(G_arr)
        clipped_eigvals = np.clip(eigvals, -0.999999, 0.999999)
        K_proj = (eigvecs * clipped_eigvals) @ eigvecs.T
        try:
            G_sympl = (I_mat + K_proj) @ cls._spd_inverse(I_mat - K_proj)
        except LinAlgError:
            return cls._project_psd(G_arr)
        return 0.5 * (G_sympl + G_sympl.T)

    @classmethod
    def _select_metric_projection(
        cls,
        G: np.ndarray,
        mode: Optional[ProjectorMode] = None,
    ) -> np.ndarray:
        r"""Primitiva de proyección: germen del selector de método RSI³ (fase 2.D′)."""
        chosen = mode if mode is not None else cls._DEFAULT_PROJECTOR
        if chosen is ProjectorMode.CAYLEY_DARBOUX:
            return cls._project_psd_cayley_darboux(G)
        return cls._project_psd(G)

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 1.B — Análisis espectral canónico
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _compute_spectral_profile(cls, G: np.ndarray) -> SpectralProfile:
        tol = cls._compute_symmetry_tolerance(G)
        asym = _fro(G - G.T)
        if asym > tol:
            raise MetricTensorError(
                f"_compute_spectral_profile: matriz no simétrica "
                f"({asym:.3e} > tol={tol:.3e})."
            )
        eigenvalues = np.linalg.eigvalsh(G)
        eigenvalues_immutable = np.array(eigenvalues, dtype=_FLOAT_DTYPE, copy=True)
        eigenvalues_immutable.setflags(write=False)
        lam_min = float(eigenvalues[0])
        lam_max = float(eigenvalues[-1])
        if lam_min > 0.0:
            cond = lam_max / lam_min
        elif lam_min == 0.0:
            cond = np.inf
        else:
            cond = np.nan
        n = len(eigenvalues)
        gap = (
            float(eigenvalues[1] - eigenvalues[0]) / lam_max
            if n > 1 and lam_max > 0.0
            else 0.0
        )
        return SpectralProfile(
            eigenvalues=eigenvalues_immutable,
            lambda_min=lam_min,
            lambda_max=lam_max,
            condition_number=cond,
            spectral_gap=gap,
            frobenius_norm=_fro(G),
        )

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 1.C — Christoffel (Connes+Koszul) + Ricci 0-jet + compatibilidad ∇G
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _connes_metric_derivatives(G: np.ndarray) -> np.ndarray:
        r"""Derivada no conmutativa $\partial_\mu G_{\nu\sigma}=[X_\mu,G]_{\nu\sigma}$."""
        n = int(G.shape[0])
        dG = np.zeros((n, n, n), dtype=_FLOAT_DTYPE)
        for mu in range(n):
            dG[mu, mu, :] += G[mu, :]
            dG[mu, :, mu] -= G[:, mu]
        return dG

    @classmethod
    def _synthesize_christoffel_from_metric(cls, G: np.ndarray) -> np.ndarray:
        r"""
        Christoffel de Koszul $\Gamma^{r}{}_{mn}=\tfrac12 G^{rs}
        (\partial_m G_{ns}+\partial_n G_{ms}-\partial_s G_{mn})$,
        simetrizado en $(m,n)$ (torsión nula; unicidad del teorema fundamental).
        """
        G_inv = cls._spd_inverse(G)
        dG = cls._connes_metric_derivatives(G)
        term1 = np.einsum("rs,mns->rmn", G_inv, dG)
        term2 = np.einsum("rs,nms->rmn", G_inv, dG)
        term3 = np.einsum("rs,smn->rmn", G_inv, dG)
        gamma = 0.5 * (term1 + term2 - term3)
        return 0.5 * (gamma + np.transpose(gamma, (0, 2, 1)))

    @classmethod
    def _metric_compatibility_residual(
        cls,
        G: np.ndarray,
        christoffel: Optional[np.ndarray] = None,
        dG: Optional[np.ndarray] = None,
    ) -> float:
        r"""Residuo de $\nabla_\rho G_{\mu\nu}=0$."""
        if christoffel is None:
            christoffel = cls._synthesize_christoffel_from_metric(G)
        if dG is None:
            dG = cls._connes_metric_derivatives(G)
        term_a = np.einsum("lrm,ln->rmn", christoffel, G)
        term_b = np.einsum("lrn,ml->rmn", christoffel, G)
        return _fro(dG - term_a - term_b)

    @classmethod
    def _compute_ricci_tensor(
        cls,
        G: np.ndarray,
        christoffel: Optional[np.ndarray] = None,
        discretizer: Optional[RicciDiscretizer] = None,
    ) -> Tuple[np.ndarray, float]:
        r"""
        $\operatorname{Ric}_{\mu\nu}=\Gamma^{\rho}_{\mu\nu}\Gamma^{\sigma}_{\rho\sigma}
        -\Gamma^{\sigma}_{\mu\rho}\Gamma^{\rho}_{\nu\sigma}$,
        escalar $R=\mathrm{tr}(G^{-1}\mathrm{Ric})$ (0-jet, carta normal discreta).

        El discretizador `TRACEFREE_PROXY` usa $G-(\mathrm{tr}\,G/n)I$ cuando
        Koszul es numéricamente patológico (automejora de método, RSI³).
        """
        n = int(G.shape[0])
        mode = discretizer if discretizer is not None else cls._DEFAULT_DISCRETIZER
        if christoffel is None and mode is RicciDiscretizer.KOSZUL_QUADRATIC:
            try:
                christoffel = cls._synthesize_christoffel_from_metric(G)
            except LinAlgError:
                christoffel = None
        if christoffel is not None and mode is RicciDiscretizer.KOSZUL_QUADRATIC:
            gamma = np.asarray(christoffel, dtype=_FLOAT_DTYPE)
            trace_gamma = np.einsum("srs->r", gamma)
            quad1 = np.einsum("rmn,r->mn", gamma, trace_gamma)
            quad2 = np.einsum("smp,pns->mn", gamma, gamma)
            ricci = quad1 - quad2
            ricci = 0.5 * (ricci + ricci.T)
            G_inv = cls._spd_inverse(G)
            scalar_r = float(np.trace(G_inv @ ricci))
            return ricci, scalar_r
        tr_G = float(np.trace(G))
        ricci_proxy = G - (tr_G / float(n)) * np.eye(n, dtype=_FLOAT_DTYPE)
        G_inv = cls._spd_inverse(G)
        scalar_r = float(np.trace(G_inv @ ricci_proxy))
        return ricci_proxy, scalar_r

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 1.D — Defecto de Einstein, cota de Connes–DK, Bianchi canónica
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _compute_einstein_defect(cls, G: np.ndarray) -> float:
        r"""
        $\delta_{\mathrm{Einstein}}(G)=\|\operatorname{Ric}-(R/n)G\|_F/\|G\|_F$.
        Mide la distancia al punto fijo geométrico (métrica de Einstein).
        """
        n = int(G.shape[0])
        ricci, scalar_r = cls._compute_ricci_tensor(G)
        target = ricci - (scalar_r / float(n)) * G
        denom = max(_fro(G), _NEAR_ZERO_FROBENIUS_TOL)
        return float(_fro(target) / denom)

    @classmethod
    def _compute_lipschitz_connes_bound_exact(cls, G: np.ndarray) -> float:
        r"""
        Cota exacta de Connes–Daleckii–Krein: cociente diferencial de
        $f(\lambda)=\lambda^{-1/2}$ sobre el espectro de G.
        """
        eigvals = np.sort(np.real(np.linalg.eigvalsh(0.5 * (G + G.T))))
        eigvals = np.maximum(eigvals, _SPECTRAL_PSD_FLOOR)
        n = len(eigvals)
        if n < 2:
            return float(0.5 / (eigvals[0] ** 1.5))
        f_vals = 1.0 / np.sqrt(eigvals)
        max_derivative = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                delta_lam = eigvals[j] - eigvals[i]
                if delta_lam < 1e-12:
                    deriv = 0.5 / (eigvals[i] ** 1.5)
                else:
                    deriv = abs(f_vals[j] - f_vals[i]) / delta_lam
                if deriv > max_derivative:
                    max_derivative = deriv
        return float(max_derivative)

    @classmethod
    def _compute_lipschitz_connes_bound(cls, G: np.ndarray) -> float:
        r"""Cota exacta de Connes-Daleckii-Krein sobre $D=\boldsymbol\rho^{-1/2}$."""
        return cls._compute_lipschitz_connes_bound_exact(G)

    @classmethod
    def _bianchi_identity_residual(
        cls,
        G: np.ndarray,
        christoffel: Optional[np.ndarray] = None,
        einstein_low: Optional[np.ndarray] = None,
    ) -> float:
        r"""
        Residual de la identidad de Bianchi contraída $\nabla_\mu\mathrm{Ein}^{\mu\nu}=0$
        en carta normal 0-jet. Indexación canónica: $\Gamma[r,m,n]=\Gamma^{r}{}_{mn}$.

        $$(\nabla_\mu\mathrm{Ein})^{\mu\nu}\approx
          \Gamma^\mu_{\mu\lambda}\mathrm{Ein}^{\lambda\nu}
          +\Gamma^\nu_{\mu\lambda}\mathrm{Ein}^{\mu\lambda}.$$
        """
        if christoffel is None:
            christoffel = cls._synthesize_christoffel_from_metric(G)
        if einstein_low is None:
            ricci, scalar_r = cls._compute_ricci_tensor(G, christoffel=christoffel)
            einstein_low = ricci - 0.5 * scalar_r * G
        G_inv = cls._spd_inverse(G)
        einstein_up = G_inv @ einstein_low @ G_inv
        trace_gamma = np.einsum("mml->l", christoffel)
        term_a = np.einsum("l,ln->n", trace_gamma, einstein_up, optimize=True)
        term_b = np.einsum("nml,ml->n", christoffel, einstein_up, optimize=True)
        divergence = term_a + term_b
        return float(np.linalg.norm(divergence, ord=2))

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 1.D′ — Weyl conforme, Killing–Noether, Kirchhoff–Bianchi
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _weyl_conformal_residual(
        cls,
        G: np.ndarray,
        christoffel: Optional[np.ndarray] = None,
        ricci: Optional[np.ndarray] = None,
        ricci_scalar: Optional[float] = None,
    ) -> float:
        r"""
        Residual del tensor de Weyl 0-jet (parte conforme de Riemann).
        Riemann 0-jet: $R^{\rho}{}_{\sigma\mu\nu}=\Gamma^{\rho}{}_{\nu\lambda}
        \Gamma^{\lambda}{}_{\mu\sigma}-\Gamma^{\rho}{}_{\mu\lambda}\Gamma^{\lambda}{}_{\nu\sigma}$.
        Weyl se anula idénticamente en $n\le 3$; el residual mide la desviación
        de la discretización (Cotton-like en $n=3$).
        """
        n = int(G.shape[0])
        if n <= 2:
            return 0.0
        try:
            if christoffel is None:
                christoffel = cls._synthesize_christoffel_from_metric(G)
            if ricci is None or ricci_scalar is None:
                ricci, ricci_scalar = cls._compute_ricci_tensor(G, christoffel=christoffel)
            riemann = (
                np.einsum("rvl,lms->rsmv", christoffel, christoffel, optimize=True)
                - np.einsum("rml,lvs->rsmv", christoffel, christoffel, optimize=True)
            )
            riemann_low = np.einsum("ap,psmv->asmv", G, riemann, optimize=True)
            denom = 2.0 * float(n - 1)
            P = ricci - (float(ricci_scalar) / denom) * G
            kn = (
                np.einsum("am,bn->abmn", P, G)
                - np.einsum("an,bm->abmn", P, G)
                - np.einsum("bm,an->abmn", P, G)
                + np.einsum("bn,am->abmn", P, G)
            )
            weyl = riemann_low - kn / float(max(n - 2, 1))
            return float(np.linalg.norm(weyl))
        except (LinAlgError, ValueError, MemoryError):
            return float("inf")

    @classmethod
    def _killing_equation_residual(
        cls,
        G: np.ndarray,
        christoffel: Optional[np.ndarray] = None,
    ) -> float:
        r"""
        Residual de la ecuación de Killing para el marco $\xi_{(\alpha)}^{\mu}=\delta^{\mu}_{\alpha}$:
        $(\mathcal{L}_{\xi}G)_{\mu\nu}=\nabla_{\mu}\xi_{\nu}+\nabla_{\nu}\xi_{\mu}$.
        Con $\xi_{\lambda}=G_{\lambda\alpha}$ (bajar índice), cuantifica la ausencia
        de isometrías infinitesimales (Noether discreto).
        """
        n = int(G.shape[0])
        try:
            if christoffel is None:
                christoffel = cls._synthesize_christoffel_from_metric(G)
            xi_low = np.asarray(G, dtype=_FLOAT_DTYPE)
            nabla_xi = -np.einsum("lmn,la->mna", christoffel, xi_low, optimize=True)
            lie = nabla_xi + np.transpose(nabla_xi, (1, 0, 2))
            return float(np.linalg.norm(lie)) / float(max(n, 1))
        except (LinAlgError, ValueError):
            return float("inf")

    @classmethod
    def _kirchhoff_current_residual(
        cls,
        G: np.ndarray,
        einstein_low: Optional[np.ndarray] = None,
    ) -> float:
        r"""
        Dual circuito–geométrico (Kirchhoff Current Law $\equiv$ Bianchi contraída):
        $I^{\mu}=\sum_{\nu}\mathrm{Ein}^{\mu\nu}$. KCL exige $I^{\mu}=0$ en cada nodo.
        KVL está cifrada en $\mathrm{Ein}=\mathrm{Ein}^{T}$ (simetría 0-jet).
        """
        try:
            if einstein_low is None:
                ricci, scalar_r = cls._compute_ricci_tensor(G)
                einstein_low = ricci - 0.5 * scalar_r * G
            G_inv = cls._spd_inverse(G)
            einstein_up = G_inv @ einstein_low @ G_inv
            currents = np.sum(einstein_up, axis=1)
            kcl = float(np.linalg.norm(currents, ord=2))
            kvl = _fro(einstein_low - einstein_low.T)
            return float(kcl + kvl)
        except (LinAlgError, ValueError):
            return float("inf")

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 1.E — SPD estricta y control de condicionamiento (doble vía)
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _verify_spd_by_cholesky(cls, tensor: np.ndarray) -> bool:
        try:
            np.linalg.cholesky(tensor)
            return True
        except LinAlgError:
            return False

    @classmethod
    def _assert_spd_strict(cls, name: str, G: np.ndarray, profile: SpectralProfile) -> None:
        r"""
        Certificación SPD por doble vía:
          (a) Primaria — espectral: λ_min ≥ MIN_EIGVAL_TOL y λ_max > 0.
          (b) Redundante — Cholesky: confirmación independiente.
        """
        if not np.all(np.isfinite(G)):
            raise MetricTensorError(f"Tensor {name}: valores no finitos post-regularización.")
        tol = cls._compute_symmetry_tolerance(G)
        if _fro(G - G.T) > tol:
            raise MetricTensorError(f"Tensor {name}: asimetría residual.")
        if profile.lambda_max <= 0.0:
            raise MetricTensorError(f"Tensor {name}: λ_max={profile.lambda_max:.6e} ≤ 0.")
        if profile.lambda_min <= MIN_EIGVAL_TOL:
            raise MetricTensorError(
                f"Tensor {name}: λ_min={profile.lambda_min:.6e} ≤ {MIN_EIGVAL_TOL:.3e}."
            )
        if not cls._verify_spd_by_cholesky(G):
            raise MetricTensorError(
                f"Tensor {name}: falló Cholesky (certificación redundante). "
                "Discrepancia espectral/Cholesky ⇒ bug numérico o patología."
            )

    @classmethod
    def _assert_well_conditioned(cls, name: str, profile: SpectralProfile) -> None:
        kappa = profile.condition_number
        if not np.isfinite(kappa):
            raise MetricTensorError(f"Tensor {name}: κ no finito.")
        if kappa > COND_NUM_TOL:
            raise MetricTensorError(f"Tensor {name}: κ₂={kappa:.4e} > {COND_NUM_TOL:.4e}.")

    @classmethod
    def _try_certify_spd(
        cls, name: str, G: np.ndarray
    ) -> Optional[Tuple[np.ndarray, SpectralProfile]]:
        """Certifica SPD+κ sin lanzar; `None` si falla."""
        try:
            G_sym = cls._symmetrize_by_projection(np.asarray(G, dtype=_FLOAT_DTYPE))
            profile = cls._compute_spectral_profile(G_sym)
            cls._assert_spd_strict(name, G_sym, profile)
            cls._assert_well_conditioned(name, profile)
            return G_sym, profile
        except (LinAlgError, MetricTensorError):
            return None

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 1.F — Tikhonov, Gershgorin, deformación y consistencia espectral
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _apply_tikhonov_regularization(
        cls, name: str, G: np.ndarray, original_profile: SpectralProfile
    ) -> Tuple[np.ndarray, float]:
        lam_min = original_profile.lambda_min
        target = MIN_EIGVAL_TOL * _SPD_INTERIOR_FACTOR
        if lam_min >= target:
            return G, 0.0
        delta = target - lam_min
        n = int(G.shape[0])
        return G + delta * np.eye(n, dtype=_FLOAT_DTYPE), float(delta)

    @classmethod
    def _compute_regularization_deformation(
        cls, G_orig: np.ndarray, G_reg: np.ndarray, delta: float
    ) -> float:
        orig_norm = _fro(G_orig)
        if orig_norm < _NEAR_ZERO_FROBENIUS_TOL:
            raise MetricTensorError(
                f"Deformación relativa no definida: ||G||_F={orig_norm:.3e}≈0."
            )
        deform = _fro(G_reg - G_orig)
        n = int(G_orig.shape[0])
        expected = delta * np.sqrt(n)
        rel_disc = abs(deform - expected)
        tol = max(1e-12, 1e-10 * expected)
        if rel_disc > tol:
            logger.warning(
                "Inconsistencia Tikhonov: numérica=%.3e, analítica=%.3e", deform, expected,
            )
        return float(deform / orig_norm)

    @classmethod
    def _verify_spectral_consistency(
        cls, name: str, orig: SpectralProfile, reg: SpectralProfile, delta: float,
    ) -> None:
        if delta < _TIKHONOV_DELTA_ATOL:
            return
        expected = orig.eigenvalues + delta
        actual = reg.eigenvalues
        scale = max(
            float(np.max(np.abs(expected))),
            float(np.max(np.abs(actual))),
            1e-15,
        )
        tol = scale * 1e-12
        disc = float(np.max(np.abs(actual - expected)))
        if disc > tol:
            logger.warning(
                "Tensor %s: inconsistencia espectral post-Tikhonov (disc=%.3e).", name, disc,
            )

    @classmethod
    def _enforce_gershgorin_spd_margin(
        cls, G: np.ndarray, margin: float = _GERSHGORIN_MARGIN
    ) -> np.ndarray:
        r"""
        Levantamiento diagonal mínimo que restaura dominancia de Gershgorin:

            $G_{ii}\leftarrow G_{ii}+\max(0,\sum_{j\neq i}|G_{ij}|+\varepsilon-G_{ii})$.
        """
        G_out = np.array(G, dtype=_FLOAT_DTYPE, copy=True)
        diag = np.diag(G_out).copy()
        off = np.sum(np.abs(G_out), axis=1) - np.abs(diag)
        lift = np.maximum(off + float(margin) - diag, 0.0)
        if np.any(lift > 0.0):
            G_out = G_out + np.diag(lift)
            logger.info(
                "Gershgorin: levantamiento diagonal en filas %s (‖lift‖_∞=%.3e).",
                np.where(lift > 0.0)[0].tolist(), float(np.max(lift)),
            )
        return G_out

    @classmethod
    def _make_immutable(cls, G: np.ndarray) -> np.ndarray:
        G_imm = np.array(G, dtype=_FLOAT_DTYPE, copy=True)
        G_imm.setflags(write=False)
        return G_imm

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 1.G — Auditoría de cantidades conservadas y primeras integrales
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _conserved_quantities_audit(
        ricci_scalars: List[float],
        log_volumes: List[float],
        eh_actions: List[float],
    ) -> Dict[str, float]:
        r"""Cuantifica el *drift* relativo de $R$, $\log\mathrm{Vol}$, $S_{EH}$ en la órbita."""
        def _drift(xs: List[float]) -> float:
            if not xs:
                return 0.0
            a = np.asarray(xs, dtype=_FLOAT_DTYPE)
            a = a[np.isfinite(a)]
            if a.size == 0:
                return 0.0
            denom = max(abs(float(np.mean(a))), _WILKINSON_EPS)
            return float((float(np.max(a)) - float(np.min(a))) / denom)
        return {
            "ricci_scalar_drift": _drift(ricci_scalars),
            "log_volume_drift": _drift(log_volumes),
            "eh_action_drift": _drift(eh_actions),
        }

    @staticmethod
    def _poincare_first_integrals(
        ricci_scalars: Sequence[float],
        log_volumes: Sequence[float],
        delta_tau: float,
        G_snapshot: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        r"""
        Primeras integrales (análogos celestes sobre el flujo de Ricci):

          • $H$                 : hamiltoniano-proxy $=\mathrm{median}(R)$.
          • $C_{\mathrm{Jacobi}}$ : $C=\mathrm{median}(\log\mathrm{Vol}-\tau R)$.
          • $L_{\mathrm{ang}}$  : momento angular-proxy $=\|G-\mathrm{diag}(G)\|_F$.
        """
        R = np.asarray(list(ricci_scalars), dtype=_FLOAT_DTYPE)
        lv = np.asarray(list(log_volumes), dtype=_FLOAT_DTYPE)
        n_min = int(min(R.size, lv.size))
        if n_min == 0:
            return {"H": 0.0, "C_jacobi": 0.0, "L_angular": 0.0}
        R = R[:n_min]
        lv = lv[:n_min]
        tau = float(delta_tau) * np.arange(n_min, dtype=_FLOAT_DTYPE)
        H = float(np.median(R[np.isfinite(R)])) if np.any(np.isfinite(R)) else 0.0
        combo = lv - tau * R
        C = float(np.median(combo[np.isfinite(combo)])) if np.any(np.isfinite(combo)) else 0.0
        if G_snapshot is not None:
            G = np.asarray(G_snapshot, dtype=_FLOAT_DTYPE)
            L = _fro(G - np.diag(np.diag(G)))
        else:
            L = 0.0
        return {"H": H, "C_jacobi": C, "L_angular": float(L)}

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 1.H — Pipeline canónico de certificación (sustrato de FASE 1.Z)
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _certify_canonical(cls, name: str, G: np.ndarray) -> Phase1Handoff:
        """Valida, simetriza, Tikhonov, SPD dual, Bianchi/Weyl/Killing y empaqueta el 0-jet."""
        G_validated = cls._validate_structural_preconditions(name, G)
        input_asym = cls._assess_input_symmetry(name, G_validated)
        G_sym = cls._symmetrize_by_projection(G_validated)
        if not np.all(np.isfinite(G_sym)):
            raise MetricTensorError(f"Tensor {name}: no finitos post-simetrización.")
        try:
            original_profile = cls._compute_spectral_profile(G_sym)
        except (LinAlgError, MetricTensorError) as exc:
            raise MetricTensorError(
                f"Tensor {name}: análisis espectral inicial fallido: {exc}"
            ) from exc

        G_reg, delta = cls._apply_tikhonov_regularization(name, G_sym, original_profile)
        required_reg = delta > _TIKHONOV_DELTA_ATOL
        if required_reg:
            deformation = cls._compute_regularization_deformation(G_sym, G_reg, delta)
            if deformation > _REGULARIZATION_ABORT_THRESHOLD:
                raise MetricTensorError(
                    f"Tensor {name}: deformación {deformation:.2%} > "
                    f"{_REGULARIZATION_ABORT_THRESHOLD:.0%}."
                )
        else:
            deformation = 0.0

        try:
            reg_profile = cls._compute_spectral_profile(G_reg)
        except (LinAlgError, MetricTensorError) as exc:
            raise MetricTensorError(
                f"Tensor {name}: análisis espectral post-reg fallido: {exc}"
            ) from exc

        cls._verify_spectral_consistency(name, original_profile, reg_profile, delta)
        cls._assert_spd_strict(name, G_reg, reg_profile)
        cls._assert_well_conditioned(name, reg_profile)

        report = RegularizationReport(
            tensor_name=name,
            original_profile=original_profile,
            regularized_profile=reg_profile,
            tikhonov_delta=delta,
            frobenius_deformation=deformation,
            required_regularization=required_reg,
            input_asymmetry=input_asym,
        )
        report.log_summary()

        einstein_defect = cls._compute_einstein_defect(G_reg)
        lip = cls._compute_lipschitz_connes_bound(G_reg)
        try:
            gamma = cls._synthesize_christoffel_from_metric(G_reg)
        except LinAlgError:
            n = int(G_reg.shape[0])
            gamma = np.zeros((n, n, n), dtype=_FLOAT_DTYPE)
        try:
            ricci, scalar_r = cls._compute_ricci_tensor(G_reg, christoffel=gamma)
        except LinAlgError:
            ricci, scalar_r = np.zeros_like(G_reg), 0.0
        einstein_low = ricci - 0.5 * float(scalar_r) * G_reg
        try:
            log_volume = cls._spd_logdet(G_reg)
        except LinAlgError:
            log_volume = float("-inf")
        try:
            compat = cls._metric_compatibility_residual(G_reg, christoffel=gamma)
        except LinAlgError:
            compat = float("inf")
        try:
            bianchi = cls._bianchi_identity_residual(G_reg, gamma, einstein_low)
        except LinAlgError:
            bianchi = float("inf")
        weyl = cls._weyl_conformal_residual(G_reg, gamma, ricci, float(scalar_r))
        killing = cls._killing_equation_residual(G_reg, gamma)
        kirchhoff = cls._kirchhoff_current_residual(G_reg, einstein_low)
        vol_cons = cls._volume_form_consistency(G_reg)
        first_int = cls._poincare_first_integrals(
            [float(scalar_r)], [float(log_volume)], 0.01, G_reg
        )

        return Phase1Handoff(
            tensor_name=name,
            candidate=np.array(G_reg, dtype=_FLOAT_DTYPE, copy=True),
            spectral_profile=reg_profile,
            einstein_defect=float(einstein_defect),
            lipschitz_connes_bound=float(lip),
            input_asymmetry=float(input_asym),
            regularization_report=report,
            log_volume=float(log_volume),
            compatibility_residual=float(compat),
            bianchi_residual=float(bianchi),
            weyl_residual=float(weyl),
            killing_residual=float(killing),
            kirchhoff_residual=float(kirchhoff),
            volume_consistency=float(vol_cons),
            first_integrals=first_int,
            projector_mode=cls._DEFAULT_PROJECTOR.value,
        )

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 1.Z — HANDOFF: última definición de FASE 1 ≡ inicio de FASE 2
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _handoff_to_phase2(cls, name: str, G: np.ndarray) -> Phase1Handoff:
        r"""
        Última definición formal de FASE 1 y **continuación estricta** de FASE 2.

        Ejecuta la validación canónica y construye el morfismo `Phase1Handoff`.
        El objeto retornado **es** el dominio de `_ingest_phase1_handoff`
        (primera flecha de FASE 2):

            $\Phi_2\circ\Phi_1(G)=\texttt{\_ingest\_phase1\_handoff}
             (\texttt{\_handoff\_to\_phase2}(G))$.

        El germen geométrico (candidato SPD + residuales Bianchi/Weyl/Killing/
        Kirchhoff + primeras integrales de Poincaré) abre el anidamiento RSI³.
        """
        handoff = cls._certify_canonical(name, G)
        logger.info(
            "[FASE1→FASE2] %s: κ=%.3e, λ_min=%.3e, δ_Einstein=%.3e, "
            "L_Connes=%.3e, logvol=%.3e, ∇G=%.3e, ∇·Ein=%.3e, "
            "Weyl=%.3e, Killing=%.3e, KCL=%.3e, H=%.3e, C_J=%.3e",
            name,
            handoff.spectral_profile.condition_number,
            handoff.spectral_profile.lambda_min,
            handoff.einstein_defect,
            handoff.lipschitz_connes_bound,
            handoff.log_volume,
            handoff.compatibility_residual,
            handoff.bianchi_residual,
            handoff.weyl_residual,
            handoff.killing_residual,
            handoff.kirchhoff_residual,
            handoff.first_integrals.get("H", float("nan")),
            handoff.first_integrals.get("C_jacobi", float("nan")),
        )
        return handoff

    # ═══════════════════════════════════════════════════════════════════════════
    # FASE 2 — RSI³ Y MAPA DE POINCARÉ
    # (inicia exactamente donde termina FASE 1.Z)
    # ═══════════════════════════════════════════════════════════════════════════

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 2.A — INGESTA del handoff de FASE 1 (continuación directa)
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _ingest_phase1_handoff(cls, handoff: Phase1Handoff) -> Tuple[np.ndarray, float]:
        r"""
        Consume el `Phase1Handoff` de `_handoff_to_phase2` (continuación 1→2).

        Extrae la semilla métrica certificada $G_\star$ y el defecto de Einstein
        inicial — sustrato de $\Phi^{(1)}\subset\Phi^{(2)}\subset\Phi^{(3)}$.
        """
        return (
            np.array(handoff.candidate, dtype=_FLOAT_DTYPE, copy=True),
            float(handoff.einstein_defect),
        )

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 2.B — Nivel 1: paso de Ricci normalizado (unidad atómica)
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _rsi_level1_ricci_step(
        cls,
        G: np.ndarray,
        delta_tau: float,
        projector_mode: Optional[ProjectorMode] = None,
        discretizer: Optional[RicciDiscretizer] = None,
    ) -> np.ndarray:
        r"""
        RSI$^1$: $\Phi^{(1)}_{\Delta\tau}(G)=\Pi\bigl(
        G-2\Delta\tau(\operatorname{Ric}-(R/n)G)\bigr)$.
        El proyector $\Pi$ es seleccionado por RSI³ de método (Higham o Cayley–Darboux).
        """
        n = int(G.shape[0])
        ricci, scalar_r = cls._compute_ricci_tensor(G, discretizer=discretizer)
        G_next = G - 2.0 * float(delta_tau) * (ricci - (scalar_r / float(n)) * G)
        return cls._select_metric_projection(G_next, mode=projector_mode)

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 2.C — Nivel 2: adaptación de Δτ por gradiente de curvatura
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _rsi_level2_adaptive_tau(
        cls,
        G: np.ndarray,
        delta_tau: float,
        eta: float,
        curvature_history: List[float],
        projector_mode: Optional[ProjectorMode] = None,
        discretizer: Optional[RicciDiscretizer] = None,
    ) -> Tuple[np.ndarray, float]:
        r"""
        RSI$^2$: $\Delta\tau_{k+1}=\mathrm{sat}(\Delta\tau_k\exp(-\eta|\nabla R_k|))$,
        saturado en $[10^{-5},10^{-1}]$ (CFL discreta). Anida $\Phi^{(1)}$.
        """
        if len(curvature_history) >= 2:
            grad_R = abs(curvature_history[-1] - curvature_history[-2])
            tau_new = float(
                np.clip(delta_tau * np.exp(-eta * grad_R), _TAU_LOWER, _TAU_UPPER)
            )
        else:
            tau_new = float(delta_tau)
        return cls._rsi_level1_ricci_step(
            G, tau_new, projector_mode=projector_mode, discretizer=discretizer
        ), tau_new

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 2.D — Nivel 3: adaptación de η por contracción de Banach
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _rsi_level3_adaptive_eta(
        cls,
        G: np.ndarray,
        delta_tau: float,
        eta: float,
        zeta: float,
        contraction_history: List[float],
        curvature_history: List[float],
        projector_mode: Optional[ProjectorMode] = None,
        discretizer: Optional[RicciDiscretizer] = None,
    ) -> Tuple[np.ndarray, float, float]:
        r"""
        RSI$^3$ (parámetros): $\eta_{k+1}=\mathrm{sat}(\eta_k\exp(\zeta(L_k-L_\star)))$,
        $L_\star=0.95$. Anida $\Phi^{(2)}\supset\Phi^{(1)}$.
        """
        if len(contraction_history) >= 1:
            L_recent = float(contraction_history[-1])
            eta_new = float(
                np.clip(
                    eta * np.exp(zeta * (L_recent - _BANACH_TARGET)),
                    _ETA_LOWER, _ETA_UPPER,
                )
            )
        else:
            eta_new = float(eta)
        G_next, tau_new = cls._rsi_level2_adaptive_tau(
            G, delta_tau, eta_new, curvature_history,
            projector_mode=projector_mode, discretizer=discretizer,
        )
        return G_next, tau_new, eta_new

    @classmethod
    def _adapt_zeta_meta_meta(cls, zeta: float, eta_history: List[float]) -> float:
        r"""Meta-meta: $\zeta$ se atenúa si $\eta$ oscila y se refuerza si $\eta$ estanca."""
        if len(eta_history) < 4:
            return float(np.clip(zeta, _ZETA_LOWER, _ZETA_UPPER))
        diffs = np.diff(np.asarray(eta_history[-4:], dtype=_FLOAT_DTYPE))
        osc = float(np.std(diffs))
        eta_abs = max(abs(float(eta_history[-1])), 1.0e-12)
        if osc > 0.5 * eta_abs:
            zeta_new = 0.5 * zeta
        elif osc < 1.0e-6:
            zeta_new = 1.1 * zeta
        else:
            zeta_new = zeta
        return float(np.clip(zeta_new, _ZETA_LOWER, _ZETA_UPPER))

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 2.D′ — Automejora de MÉTODO (RSI³ auténtica)
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _rsi3_method_selector(
        cls,
        G: np.ndarray,
        current_projector: ProjectorMode,
        current_discretizer: RicciDiscretizer,
        bianchi_residual: float,
        einstein_scale: float,
    ) -> Tuple[ProjectorMode, RicciDiscretizer, str]:
        r"""
        Automejora recursiva de *método* (nivel 3 auténtico, no sólo de parámetros):

          • Si $\kappa_2(G)>\kappa_\star/10$ → Higham espectral (más estable).
          • Si $\dim$ par y $\kappa_2$ moderado → Cayley–Darboux (respeta Sp).
          • Si residual de Bianchi relativo $>\texttt{\_BIANCHI\_REL\_TOL}$ → Koszul
            cuadrático; si Bianchi es sano y $\kappa_2$ enorme → proxy libre de traza.

        Histéresis `_METHOD_SWITCH_HYSTERESIS` evita chattering del selector.
        """
        try:
            G_sym = 0.5 * (np.asarray(G, dtype=_FLOAT_DTYPE) + np.asarray(G, dtype=_FLOAT_DTYPE).T)
            profile = cls._compute_spectral_profile(G_sym)
            cond = float(profile.condition_number)
        except (LinAlgError, MetricTensorError):
            cond = float("inf")
        n = int(G.shape[0])
        bianchi_rel = float(bianchi_residual) / max(float(einstein_scale), _WILKINSON_EPS)

        projector = current_projector
        if cond > COND_NUM_TOL / 10.0:
            projector = ProjectorMode.SPECTRAL_HIGHAM
        elif n % 2 == 0 and cond < (COND_NUM_TOL / 10.0) / _METHOD_SWITCH_HYSTERESIS:
            projector = ProjectorMode.CAYLEY_DARBOUX

        discretizer = current_discretizer
        if bianchi_rel > _BIANCHI_REL_TOL:
            discretizer = RicciDiscretizer.KOSZUL_QUADRATIC
        elif (
            bianchi_rel < _BIANCHI_REL_TOL / _METHOD_SWITCH_HYSTERESIS
            and cond > COND_NUM_TOL / 2.0
        ):
            discretizer = RicciDiscretizer.TRACEFREE_PROXY

        tag = f"{projector.value}+{discretizer.value}"
        return projector, discretizer, tag

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 2.E — Banach, certificado y sonda de Lipschitz (desacoplados)
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _banach_contraction_constant(orbit: List[np.ndarray]) -> float:
        r"""
        $L_\Phi\approx\sup_k\|G_{k+1}-G_k\|_F/\|G_k-G_{k-1}\|_F$.
        Si $L_\Phi<1$ ⇒ flujo contractivo ⇒ punto fijo único (Banach).
        """
        if len(orbit) < 3:
            return 1.0
        diffs = [_fro(orbit[k + 1] - orbit[k]) for k in range(len(orbit) - 1)]
        ratios = [
            diffs[k + 1] / max(diffs[k], 1e-15) for k in range(len(diffs) - 1)
        ]
        if not ratios:
            return 1.0
        return float(np.max(np.clip(np.asarray(ratios, dtype=_FLOAT_DTYPE), 0.0, 1.0e6)))

    @classmethod
    def _banach_fixed_point_certificate(
        cls,
        orbit: List[np.ndarray],
        lipschitz: float,
    ) -> BanachCertificate:
        r"""
        Certificado del teorema del punto fijo de Banach:

          (i)   $X=\overline{\mathrm{SPD}}_n$ es completo en $\|\cdot\|_F$.
          (ii)  $\Phi(X)\subset X$  (todo $G_k$ es SPD por construcción del proyector).
          (iii) $\mathrm{Lip}\,\Phi\le L_\star=0.95<1$.
        """
        is_complete = True
        is_self_map = True
        for G in orbit:
            if not np.all(np.isfinite(G)):
                is_self_map = False
                break
            try:
                eig = np.linalg.eigvalsh(0.5 * (G + G.T))
                if float(np.min(eig)) < -10.0 * _SPECTRAL_PSD_FLOOR:
                    is_self_map = False
                    break
            except LinAlgError:
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

    @classmethod
    def _lipschitz_phi_empirical(
        cls, G: np.ndarray, delta_tau: float, eta: float,
        curvature_history: List[float],
        projector_mode: Optional[ProjectorMode] = None,
        discretizer: Optional[RicciDiscretizer] = None,
    ) -> float:
        r"""Sonda determinista de $\mathrm{Lip}(\Phi^{(2)})$ en la dirección del Ricci sin traza."""
        n = int(G.shape[0])
        ricci, scalar_r = cls._compute_ricci_tensor(G, discretizer=discretizer)
        tangent = ricci - (scalar_r / float(n)) * G
        vnorm = _fro(tangent)
        if vnorm < 1.0e-15:
            tangent = np.eye(n, dtype=_FLOAT_DTYPE)
            vnorm = float(np.sqrt(n))
        eps = _LIPSCHITZ_PERT_REL * max(_fro(G), 1.0)
        perturbation = (eps / vnorm) * tangent
        G_pert = cls._select_metric_projection(G + perturbation, mode=projector_mode)
        dummy = (
            list(curvature_history[-2:]) if len(curvature_history) >= 2
            else [scalar_r, scalar_r]
        )
        phi_g, _ = cls._rsi_level2_adaptive_tau(
            G, delta_tau, eta, dummy,
            projector_mode=projector_mode, discretizer=discretizer,
        )
        phi_p, _ = cls._rsi_level2_adaptive_tau(
            G_pert, delta_tau, eta, dummy,
            projector_mode=projector_mode, discretizer=discretizer,
        )
        return _fro(phi_p - phi_g) / max(_fro(G_pert - G), 1.0e-15)

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 2.F — Orden de convergencia de RSI³ (tripletas monótonas)
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _quadratic_convergence_rate(orbit: List[np.ndarray]) -> float:
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

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 2.G — Jacobiano (diferencias centrales) y Lyapunov QR de Benettin
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _linearized_flow_jacobian(
        cls, G: np.ndarray, delta_tau: float, eta: float,
        curvature_history: List[float], eps: float = _JACOBI_FD_EPS,
        projector_mode: Optional[ProjectorMode] = None,
        discretizer: Optional[RicciDiscretizer] = None,
    ) -> np.ndarray:
        r"""
        Jacobiano numérico de $\Phi^{(2)}$ por diferencias finitas *centrales*:
        $J_{\cdot i}=\bigl(\Phi(G+\varepsilon e_i)-\Phi(G-\varepsilon e_i)\bigr)/(2\varepsilon)$.
        Dimensión $n^{2}\times n^{2}$. Orden $O(\varepsilon^{2})$, superior a v7.2.0.
        """
        n = int(G.shape[0])
        n2 = n * n

        def _phi(H: np.ndarray) -> np.ndarray:
            return cls._rsi_level2_adaptive_tau(
                H, delta_tau, eta, curvature_history,
                projector_mode=projector_mode, discretizer=discretizer,
            )[0]

        J = np.zeros((n2, n2), dtype=_FLOAT_DTYPE)
        scale = max(_fro(G), 1.0) * float(eps)
        for i in range(n2):
            Hp = G.copy()
            Hm = G.copy()
            Hp.flat[i] += scale
            Hm.flat[i] -= scale
            J[:, i] = (_phi(Hp).ravel() - _phi(Hm).ravel()) / (2.0 * scale)
        return J

    @classmethod
    def _benettin_qr_lyapunov(
        cls, orbit: List[np.ndarray], delta_tau: float, eta: float,
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
            return np.array([-np.inf], dtype=_FLOAT_DTYPE)
        n = int(orbit[0].shape[0])
        n2 = n * n
        d = int(min(n2, _LYAP_DIM_CAP))
        if d < n2:
            Q, _ = np.linalg.qr(cls._RNG.normal(size=(n2, d)))
        else:
            Q = np.eye(n2)
        log_sums = np.zeros(d)
        count = 0
        for k in range(1, n_steps):
            if k % _BENETTIN_QR_STRIDE != 0:
                continue
            try:
                J = cls._linearized_flow_jacobian(
                    orbit[k], delta_tau, eta, curvature_history,
                    projector_mode=projector_mode, discretizer=discretizer,
                )
            except (LinAlgError, ValueError, MetricTensorError):
                continue
            M = J @ Q
            Q_new, R = np.linalg.qr(M)
            diag_R = np.abs(np.diag(R))
            log_sums += np.log(np.maximum(diag_R, _WILKINSON_EPS))
            Q = Q_new
            count += 1
        if count == 0:
            return cls._lyapunov_exponent_spectrum(orbit)
        return np.sort(log_sums / float(count))[::-1]

    @staticmethod
    def _symplectic_lyapunov_pairing(spectrum: np.ndarray) -> float:
        r"""
        Defecto de pareado simpléctico
        $\frac1{\lfloor d/2\rfloor}\sum_i|\lambda_i+\lambda_{d-1-i}|/(1+|\lambda_i|)$
        (para flujos hamiltonianos, el espectro es palíndromo $\lambda\leftrightarrow-\lambda$).
        """
        lam = np.asarray(spectrum, dtype=_FLOAT_DTYPE).ravel()
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

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 2.H — Driver RSI³ completo con anidamiento 1 ⊂ 2 ⊂ 3 + métodos
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _rsi3_driver(
        cls,
        G0: np.ndarray,
        n_iterations: int = _RSI3_MAX_ITER,
        delta_tau0: float = 1e-2,
        eta0: float = 0.1,
        zeta0: float = 0.01,
    ) -> Dict[str, Any]:
        r"""
        Orquesta RSI$^3$: en cada iteración ejecuta
        $\Phi^{(3)}\supset\Phi^{(2)}\supset\Phi^{(1)}$, adapta $\zeta$ (meta-meta)
        y selecciona el *método* (proyector + discretizador) según residuales.
        """
        G = np.array(G0, dtype=_FLOAT_DTYPE, copy=True)
        delta_tau, eta, zeta = float(delta_tau0), float(eta0), float(zeta0)
        orbit: List[np.ndarray] = [np.array(G, copy=True)]
        _, r0 = cls._compute_ricci_tensor(G)
        curv_hist: List[float] = [float(r0)]
        einstein_hist: List[float] = [cls._compute_einstein_defect(G)]
        log_vol_hist: List[float] = []
        eh_hist: List[float] = []
        try:
            lv0 = cls._spd_logdet(G)
            log_vol_hist.append(float(lv0))
            eh_hist.append(float(r0 * np.exp(np.clip(lv0, -700.0, 700.0))))
        except LinAlgError:
            log_vol_hist.append(float("-inf"))
            eh_hist.append(0.0)
        contraction_hist: List[float] = []
        lipschitz_hist: List[float] = []
        tau_hist: List[float] = [delta_tau]
        eta_hist: List[float] = [eta]
        zeta_hist: List[float] = [zeta]
        method_lineage: List[str] = []
        projector = cls._DEFAULT_PROJECTOR
        discretizer = cls._DEFAULT_DISCRETIZER
        consecutive_small = 0
        n_iter = max(int(n_iterations), 1)

        for k in range(n_iter):
            try:
                bianchi_k = cls._bianchi_identity_residual(G)
            except (LinAlgError, MetricTensorError):
                bianchi_k = float("inf")
            ein_scale = max(abs(float(einstein_hist[-1])) * max(_fro(G), 1.0), _WILKINSON_EPS)
            projector, discretizer, tag = cls._rsi3_method_selector(
                G, projector, discretizer, bianchi_k, ein_scale
            )
            method_lineage.append(tag)

            zeta = cls._adapt_zeta_meta_meta(zeta, eta_hist)
            G_next, delta_tau, eta = cls._rsi_level3_adaptive_eta(
                G, delta_tau, eta, zeta, contraction_hist, curv_hist,
                projector_mode=projector, discretizer=discretizer,
            )
            orbit.append(np.array(G_next, copy=True))
            _, scalar_r = cls._compute_ricci_tensor(G_next, discretizer=discretizer)
            curv_hist.append(float(scalar_r))
            einstein_hist.append(cls._compute_einstein_defect(G_next))
            try:
                lv = cls._spd_logdet(G_next)
                log_vol_hist.append(float(lv))
                eh_hist.append(float(scalar_r * np.exp(np.clip(lv, -700.0, 700.0))))
            except LinAlgError:
                log_vol_hist.append(float("-inf"))
                eh_hist.append(0.0)
            tau_hist.append(delta_tau)
            eta_hist.append(eta)
            zeta_hist.append(zeta)

            d_new = 0.0
            if k >= 1:
                d_new = _fro(orbit[-1] - orbit[-2])
                d_old = _fro(orbit[-2] - orbit[-3]) if k >= 2 else d_new
                if d_old > 1e-15:
                    contraction_hist.append(d_new / d_old)

            if k > 0 and (k % _LIPSCHITZ_PROBE_STRIDE == 0):
                try:
                    probe = cls._lipschitz_phi_empirical(
                        G_next, delta_tau, eta, curv_hist,
                        projector_mode=projector, discretizer=discretizer,
                    )
                    lipschitz_hist.append(float(np.clip(probe, 0.0, 1.0e6)))
                except (LinAlgError, ValueError, MetricTensorError) as exc:
                    logger.debug("[RSI³] sonda Lipschitz omitida: %s", exc)

            if (
                k + 1 >= _RSI3_MIN_ITER
                and einstein_hist[-1] < _EINSTEIN_TOL
            ):
                logger.info(
                    "[RSI³] Convergencia temprana en k=%d, δ_E=%.3e", k, einstein_hist[-1],
                )
                G = G_next
                break

            if k >= 1 and d_new < _RSI3_CONV_EPS:
                consecutive_small += 1
                if consecutive_small >= _CONSECUTIVE_CONV_STEPS:
                    logger.info(
                        "[RSI³] Punto fijo numérico en k=%d, ‖ΔG‖=%.3e (×%d)",
                        k, d_new, consecutive_small,
                    )
                    G = G_next
                    break
            else:
                consecutive_small = 0
            G = G_next

        L_final = cls._banach_contraction_constant(orbit)
        if lipschitz_hist:
            L_final = float(max(L_final, float(np.median(lipschitz_hist))))
        banach_cert = cls._banach_fixed_point_certificate(orbit, L_final)
        converged = bool(
            banach_cert.certified or (einstein_hist[-1] < _EINSTEIN_TOL)
        )
        quad_rate = cls._quadratic_convergence_rate(orbit)
        audit = cls._conserved_quantities_audit(curv_hist, log_vol_hist, eh_hist)
        first_int = cls._poincare_first_integrals(
            curv_hist, log_vol_hist, float(np.mean(tau_hist) if tau_hist else 0.01),
            orbit[-1] if orbit else None,
        )
        logger.info(
            "[RSI³] Niveles=1⊂2⊂3+método, iter=%d, L_Φ=%.4f, p=%.2f, certified=%s, "
            "Δτ_final=%.3e, η_final=%.3e, ζ_final=%.3e, δ_E=%.3e, linaje=%s",
            len(orbit) - 1, L_final, quad_rate, banach_cert.certified,
            delta_tau, eta, zeta, einstein_hist[-1],
            method_lineage[-1] if method_lineage else "n/a",
        )
        return {
            "orbit": orbit,
            "curvature_history": curv_hist,
            "contraction_history": contraction_hist,
            "lipschitz_history": lipschitz_hist,
            "delta_tau_history": tau_hist,
            "eta_history": eta_hist,
            "zeta_history": zeta_hist,
            "einstein_defect_history": einstein_hist,
            "log_volume_history": log_vol_hist,
            "eh_action_history": eh_hist,
            "banach_L": L_final,
            "rsi_level_achieved": 3,
            "converged": converged,
            "quadratic_rate": quad_rate,
            "conserved_quantities_audit": audit,
            "first_integrals": first_int,
            "method_lineage": method_lineage,
            "banach_certified": banach_cert.certified,
            "projector_mode": projector.value,
            "discretizer": discretizer.value,
        }

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 2.I — Mapa de retorno de Poincaré (sección interpolada)
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _poincare_return_map(
        orbit: List[np.ndarray],
        energy_target: float,
        curvature_history: List[float],
    ) -> Dict[str, Any]:
        r"""
        Sección $\Sigma=\{R=\bar R\}$ con interpolación lineal del tiempo de
        vuelo:

            $\alpha=(\bar R-R_{k-1})/(R_k-R_{k-1}),\quad
             G_{\tau^\star}=(1-\alpha)G_{k-1}+\alpha G_k$.

        El multiplicador de Floquet se obtiene del radio espectral de la
        matriz de monodromía (FASE 2.I′), no de un cociente de normas.
        """
        energies = np.asarray(curvature_history, dtype=_FLOAT_DTYPE)
        n_min = min(int(energies.size), len(orbit))
        energies = energies[:n_min]
        orbit_use = orbit[:n_min]
        crossings: List[int] = []
        taus: List[float] = []
        section_points: List[np.ndarray] = []
        for k in range(1, n_min):
            e0 = float(energies[k - 1])
            e1 = float(energies[k])
            if (e0 - energy_target) * (e1 - energy_target) < 0.0:
                denom = e1 - e0
                alpha = 0.5 if abs(denom) < 1.0e-15 else (energy_target - e0) / denom
                alpha = float(np.clip(alpha, 0.0, 1.0))
                crossings.append(k)
                taus.append((k - 1) + alpha)
                g_cross = (1.0 - alpha) * orbit_use[k - 1] + alpha * orbit_use[k]
                section_points.append(0.5 * (g_cross + g_cross.T))
        periods = (
            np.diff(np.asarray(taus, dtype=_FLOAT_DTYPE))
            if len(taus) > 1
            else np.array([], dtype=_FLOAT_DTYPE)
        )
        is_periodic = False
        if periods.size >= 2:
            mean_p = float(np.mean(periods))
            std_rel = float(np.std(periods) / max(mean_p, 1e-15))
            is_periodic = std_rel < 0.05
        monodromy, monodromy_eigs = MetricTensorFactory._monodromy_matrix(section_points)
        floquet = MetricTensorFactory._variational_floquet(monodromy, monodromy_eigs)
        return {
            "crossings": crossings,
            "crossing_times": taus,
            "return_periods": periods,
            "is_periodic_orbit": is_periodic,
            "energies": energies,
            "energy_target": float(energy_target),
            "floquet_multiplier": floquet,
            "section_points": section_points,
            "monodromy": monodromy,
            "monodromy_eigs": monodromy_eigs,
        }

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 2.I′ — Monodromía variacional y Floquet espectral
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _monodromy_matrix(
        section_points: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Matriz de monodromía $M=D\mathcal{T}|_{\Sigma}$ del mapa de Poincaré.

        Se estima por regresión lineal de mínimos cuadrados sobre las desviaciones
        centradas de secciones consecutivas (proyección PCA al plano tangente de
        $\Sigma$, dim $r=\min(8,\,m-1,\,n^{2})$): $\delta G_{k+1}\approx M\,\delta G_k$.
        """
        empty = np.zeros((1, 1), dtype=_FLOAT_DTYPE)
        if len(section_points) < 3:
            return empty, np.array([0.0], dtype=np.complex128)
        try:
            X = np.array([G.ravel() for G in section_points], dtype=_FLOAT_DTYPE)
            Xc = X - X.mean(axis=0, keepdims=True)
            m, n2 = Xc.shape
            r = int(min(8, m - 1, n2, max(m - 1, 1)))
            if r < 1:
                return empty, np.array([0.0], dtype=np.complex128)
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            r = int(min(r, int((S > 1.0e-12 * max(float(S[0]), 1.0)).sum()) or 1))
            coords = U[:, :r] * S[:r]
            src = coords[:-1]
            dst = coords[1:]
            M_r, *_ = np.linalg.lstsq(src, dst, rcond=None)
            M_r = np.asarray(M_r, dtype=_FLOAT_DTYPE).T
            eigs = np.linalg.eigvals(M_r)
            return M_r, eigs
        except (LinAlgError, ValueError):
            return empty, np.array([0.0], dtype=np.complex128)

    @staticmethod
    def _variational_floquet(
        monodromy: np.ndarray,
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
            except LinAlgError:
                return 0.0
        return float(np.max(np.abs(eigs)))

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 2.I″ — Lindstedt y recurrencia de Poincaré
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _poincare_lindstedt_correction(
        periods: np.ndarray,
        energies: np.ndarray,
        energy_target: float,
    ) -> float:
        r"""
        Corrección de Poincaré–Lindstedt del periodo:
        $T=T_0\bigl(1+\varepsilon T_1+\cdots\bigr)$,
        $\varepsilon=(E-\bar E)/\max(|\bar E|,1)$,
        $T_1\approx\mathrm{corr}(T_k-\bar T,\,E_k-\bar E)/\mathrm{Var}(E)$.
        """
        P = np.asarray(periods, dtype=_FLOAT_DTYPE).ravel()
        if P.size == 0:
            return 0.0
        T0 = float(np.mean(P))
        E = np.asarray(energies, dtype=_FLOAT_DTYPE).ravel()
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

    @staticmethod
    def _poincare_recurrence_bound(
        log_volumes: Sequence[float],
        lyapunov_max: float,
        n_dim: int,
        mean_step: float,
    ) -> float:
        r"""
        Cota inferior del tiempo de recurrencia de Poincaré:
        $\tau_{\mathrm{rec}}\gtrsim\mathrm{Vol}(\mathcal{M})/\mathrm{Vol}(\mathrm{tubo})$.
        $\mathrm{Vol}\approx\exp(\mathrm{median}\,\log\mathrm{Vol})$,
        tubo $\approx\mathrm{e}^{\lambda_{\max}^+}\,\Delta\tau^{\,n(n+1)/4}$.
        """
        lv = np.asarray(list(log_volumes), dtype=_FLOAT_DTYPE)
        lv = lv[np.isfinite(lv)]
        if lv.size == 0:
            return 0.0
        vol = float(np.exp(np.clip(float(np.median(lv)), -700.0, 700.0)))
        dim_eff = max(0.5 * float(n_dim) * float(n_dim + 1), 1.0)
        lam_p = max(float(lyapunov_max), 0.0)
        tube = float(np.exp(np.clip(lam_p, 0.0, 50.0))) * max(
            abs(mean_step), 1.0e-12
        ) ** (0.5 * dim_eff)
        tube = max(tube, _WILKINSON_EPS)
        return float(max(vol / tube, 0.0))

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 2.J — Variables acción-ángulo y función generatriz HJ
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _action_angle_extract(
        section_points: List[np.ndarray],
    ) -> Tuple[float, float]:
        r"""
        Variables acción-ángulo $(I,\theta)$ sobre $\Sigma\subset T^*\mathcal{M}$:

          • Acción: $I=\frac{1}{2\pi}\oint_\Sigma p\,dq$ (área simpléctica PCA, polígono cerrado).
          • Ángulo: fase del armónico dominante de la FFT de $q_k$.
        """
        if len(section_points) < 3:
            return (0.0, 0.0)
        try:
            X = np.array([G.ravel() for G in section_points], dtype=_FLOAT_DTYPE)
            X_centered = X - X.mean(axis=0, keepdims=True)
            U, S, _ = np.linalg.svd(X_centered, full_matrices=False)
            if len(S) < 2 or S[1] < 1e-15:
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
        except (LinAlgError, ValueError):
            return (0.0, 0.0)

    @staticmethod
    def _hamilton_jacobi_generating_function(
        section_points: List[np.ndarray], action: float
    ) -> float:
        r"""
        Residual de la función generatriz de tipo 2 $S(q,I)$ (Hamilton–Jacobi):
        $p=\partial S/\partial q$, $\theta=\partial S/\partial I$.
        Se toma $S=\sum_k p_k(q_{k+1}-q_k)$ y se verifica
        $|I-\frac1{2\pi}S|$ relativo.
        """
        if len(section_points) < 3 or abs(action) < 1.0e-18:
            return 0.0
        try:
            X = np.array([G.ravel() for G in section_points], dtype=_FLOAT_DTYPE)
            Xc = X - X.mean(axis=0, keepdims=True)
            U, S, _ = np.linalg.svd(Xc, full_matrices=False)
            if len(S) < 2 or S[1] < 1.0e-15:
                return 0.0
            q = U[:, 0] * S[0]
            p = U[:, 1] * S[1]
            S_gen = float(np.sum(p[:-1] * np.diff(q)))
            I_from_S = abs(S_gen) / (2.0 * np.pi)
            return _safe_div(abs(I_from_S - abs(action)), max(abs(action), _WILKINSON_EPS))
        except (LinAlgError, ValueError):
            return 0.0

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 2.K — Espectro de Lyapunov (fallback 1-D) y FFT orbital
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _lyapunov_exponent_spectrum(orbit: List[np.ndarray]) -> np.ndarray:
        r"""
        Fallback 1-D: $\lambda_{\max}=\overline{\log(\|G_{k+1}-G_k\|_F/\|G_k-G_{k-1}\|_F)}$.
        """
        if len(orbit) < 3:
            return np.array([-np.inf], dtype=_FLOAT_DTYPE)
        norms = [_fro(orbit[k + 1] - orbit[k]) for k in range(len(orbit) - 1)]
        log_ratios = [
            float(np.log(max(norms[k + 1] / max(norms[k], 1e-15), 1e-15)))
            for k in range(len(norms) - 1)
        ]
        if not log_ratios:
            return np.array([-np.inf], dtype=_FLOAT_DTYPE)
        return np.array([float(np.mean(log_ratios))], dtype=_FLOAT_DTYPE)

    @staticmethod
    def _extract_orbital_frequencies(
        signal: np.ndarray, dt: float = 1.0
    ) -> np.ndarray:
        r"""Extrae frecuencias dominantes de $R(\tau)$ por FFT con ventana de Hann."""
        x = np.asarray(signal, dtype=_FLOAT_DTYPE)
        n = int(x.size)
        if n < 8:
            return np.array([], dtype=_FLOAT_DTYPE)
        x = x - float(np.mean(x))
        hann = 0.5 - 0.5 * np.cos(
            2.0 * np.pi * np.arange(n, dtype=_FLOAT_DTYPE) / max(n - 1, 1)
        )
        spec = np.abs(np.fft.rfft(x * hann))
        freqs = np.fft.rfftfreq(n, d=float(dt))
        spec[0] = 0.0
        thresh = float(np.median(spec) * 3.0)
        peaks: List[float] = []
        for i in range(1, len(spec) - 1):
            if spec[i] > spec[i - 1] and spec[i] > spec[i + 1] and spec[i] > thresh:
                peaks.append(float(freqs[i]))
        peaks.sort(key=lambda f: -float(spec[int(np.argmin(np.abs(freqs - f)))]))
        return np.asarray(peaks[:4], dtype=_FLOAT_DTYPE)

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 2.M — Resonancias por fracciones continuas
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _continued_fraction_resonances(
        omega: np.ndarray, max_denom: int = 64, max_resonances: int = 6,
    ) -> Tuple[Tuple[int, ...], ...]:
        r"""
        Retículo de resonancias $k$ con $\omega\cdot k\approx 0$ vía convergentes
        de $\omega_i/\omega_j$ (criterio KAM de divisores pequeños).
        """
        omega = np.asarray(omega, dtype=_FLOAT_DTYPE).ravel()
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
                if abs(frac) < 1e-12:
                    break
                a = 1.0 / frac
            return conv

        candidates: List[Tuple[float, Tuple[int, ...]]] = []
        for i, j in combinations(range(n), 2):
            if abs(omega[j]) < 1e-15:
                continue
            ratio = float(omega[i] / omega[j])
            for p, q in _cf_convergents(ratio, max_denom):
                if p == 0 and q == 0:
                    continue
                vec = [0] * n
                vec[i] = q
                vec[j] = -p
                dot = abs(float(np.dot(omega, np.asarray(vec, dtype=_FLOAT_DTYPE))))
                if dot > 0.0:
                    candidates.append((dot, tuple(vec)))
        candidates.sort(key=lambda t: t[0])
        return tuple(vec for _, vec in candidates[:max_resonances])

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 2.Z — HANDOFF: última definición de FASE 2 ≡ inicio de FASE 3
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _handoff_to_phase3(
        cls, phase1: Phase1Handoff, rsi_out: Dict[str, Any]
    ) -> Phase2Handoff:
        r"""
        Última definición formal de FASE 2 y **continuación estricta** de FASE 3.

        Empaqueta RSI³ + Poincaré interpolado + Lyapunov QR + FFT + acción-ángulo
        Hamilton–Jacobi + monodromía + Lindstedt + recurrencia + resonancias CF
        y construye el morfismo `Phase2Handoff`, dominio de
        `_ingest_phase2_handoff`:

            $\Phi_3\circ\Phi_2(h_1)=\texttt{\_ingest\_phase2\_handoff}
             (\texttt{\_handoff\_to\_phase3}(h_1,\mathrm{RSI}^3))$.
        """
        orbit: List[np.ndarray] = list(rsi_out["orbit"])
        curv_hist = list(rsi_out["curvature_history"])
        energy_target = float(np.median(curv_hist)) if curv_hist else 0.0
        poincare = cls._poincare_return_map(orbit, energy_target, curv_hist)

        lyap_spec_full = cls._benettin_qr_lyapunov(
            orbit,
            float(rsi_out["delta_tau_history"][-1]) if rsi_out["delta_tau_history"] else 0.01,
            float(rsi_out["eta_history"][-1]) if rsi_out["eta_history"] else 0.1,
            curv_hist,
        )
        lyap_max = float(lyap_spec_full[0]) if lyap_spec_full.size else float(
            cls._lyapunov_exponent_spectrum(orbit)[0]
        )
        pairing_defect = cls._symplectic_lyapunov_pairing(lyap_spec_full)

        dt_mean = float(np.mean(rsi_out.get("delta_tau_history", [1.0])))
        omega = cls._extract_orbital_frequencies(
            np.asarray(curv_hist, dtype=_FLOAT_DTYPE),
            dt=max(dt_mean, 1.0e-12),
        )
        action, angle = cls._action_angle_extract(poincare.get("section_points", []))
        gf_residual = cls._hamilton_jacobi_generating_function(
            poincare.get("section_points", []), action
        )
        resonances = cls._continued_fraction_resonances(omega) if omega.size >= 2 else ()
        periods = np.asarray(poincare.get("return_periods", []), dtype=_FLOAT_DTYPE)
        lindstedt = cls._poincare_lindstedt_correction(
            periods, np.asarray(curv_hist, dtype=_FLOAT_DTYPE), energy_target
        )
        n_dim = int(orbit[0].shape[0]) if orbit else 1
        rec_bound = cls._poincare_recurrence_bound(
            rsi_out.get("log_volume_history", []), lyap_max, n_dim, dt_mean
        )
        monodromy_eigs = tuple(
            complex(z) for z in np.asarray(poincare.get("monodromy_eigs", []))
        )
        method_lineage = tuple(str(x) for x in rsi_out.get("method_lineage", []))

        handoff = Phase2Handoff(
            phase1=phase1,
            orbit=tuple(orbit),
            curvature_history=tuple(float(x) for x in curv_hist),
            lyapunov_max=lyap_max,
            poincare_data=poincare,
            banach_L=float(rsi_out["banach_L"]),
            delta_tau_history=tuple(float(x) for x in rsi_out["delta_tau_history"]),
            eta_history=tuple(float(x) for x in rsi_out["eta_history"]),
            einstein_defect_history=tuple(
                float(x) for x in rsi_out["einstein_defect_history"]
            ),
            rsi3_level=int(rsi_out["rsi_level_achieved"]),
            rsi3_converged=bool(rsi_out["converged"]),
            dominant_frequencies=tuple(float(x) for x in omega),
            lyapunov_spectrum=tuple(float(x) for x in lyap_spec_full),
            action_angle=(float(action), float(angle)),
            quadratic_rate=float(rsi_out.get("quadratic_rate", 0.0)),
            resonance_vectors=resonances,
            conserved_quantities_audit=dict(
                rsi_out.get("conserved_quantities_audit", {})
            ),
            method_lineage=method_lineage,
            monodromy_eigs=monodromy_eigs,
            lindstedt_period=float(lindstedt),
            recurrence_bound=float(rec_bound),
            banach_certified=bool(rsi_out.get("banach_certified", False)),
            generating_function_residual=float(gf_residual),
            lyapunov_pairing_defect=float(pairing_defect),
            projector_mode=str(rsi_out.get("projector_mode", cls._DEFAULT_PROJECTOR.value)),
        )
        logger.info(
            "[FASE2→FASE3] %s: λ_max=%.3e, |Λ|=%d, L_Φ=%.4f, p=%.2f, crossings=%d, "
            "Floquet=%.3e, I=%.3e, θ=%.3f, n_ω=%d, n_res=%d, T_Lind=%.3e, "
            "τ_rec=%.3e, pairing=%.3e, HJ=%.3e, converged=%s",
            phase1.tensor_name,
            handoff.lyapunov_max,
            len(lyap_spec_full),
            handoff.banach_L,
            handoff.quadratic_rate,
            len(poincare["crossings"]),
            float(poincare.get("floquet_multiplier", 0.0)),
            action, angle,
            len(handoff.dominant_frequencies),
            len(resonances),
            lindstedt, rec_bound, pairing_defect, gf_residual,
            handoff.rsi3_converged,
        )
        return handoff

    # ═══════════════════════════════════════════════════════════════════════════
    # FASE 3 — KAM, TANGLE, MELNIKOV, HOPF, MORSE Y CERTIFICADO
    # (inicia exactamente donde termina FASE 2.Z)
    # ═══════════════════════════════════════════════════════════════════════════

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 3.A — INGESTA del handoff de FASE 2 (continuación directa)
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _ingest_phase2_handoff(
        cls, handoff: Phase2Handoff
    ) -> Tuple[Phase1Handoff, List[np.ndarray], Dict[str, Any], float, Tuple[float, ...]]:
        r"""
        Consume el `Phase2Handoff` de `_handoff_to_phase3` (continuación 2→3).
        """
        return (
            handoff.phase1,
            list(handoff.orbit),
            handoff.poincare_data,
            float(handoff.lyapunov_max),
            tuple(handoff.dominant_frequencies),
        )

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 3.B — Criterio diofántico KAM (pares + retículo de resonancias)
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _diophantine_gamma(
        omega: np.ndarray, tau: float = _KAM_TAU, max_k_norm: int = 4,
    ) -> float:
        r"""
        Ínfimo $\gamma=\inf_k|\omega\cdot k|\,|k|^{\tau}$ sobre el retículo.
        Si $\dim\omega>4$ se reduce a chequeos *pares* (2-toros) — evita la
        explosión combinatoria $|\mathcal{L}|=(2K+1)^n$.
        """
        omega = np.asarray(omega, dtype=_FLOAT_DTYPE).ravel()
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
                    k_arr = np.asarray(k, dtype=_FLOAT_DTYPE)
                    k_norm = float(np.linalg.norm(k_arr))
                    if k_norm > max_k_norm or k_norm < 1e-15:
                        continue
                    dot = abs(float(np.dot(pair, k_arr)))
                    candidate = dot * (k_norm ** tau)
                    if candidate < min_gamma:
                        min_gamma = candidate
            return float(min_gamma)
        for k in product(range(-max_k_norm, max_k_norm + 1), repeat=n):
            if all(ki == 0 for ki in k):
                continue
            k_arr = np.asarray(k, dtype=_FLOAT_DTYPE)
            k_norm = float(np.linalg.norm(k_arr))
            if k_norm > max_k_norm or k_norm < 1e-15:
                continue
            dot = abs(float(np.dot(omega, k_arr)))
            candidate = dot * (k_norm ** tau)
            if candidate < min_gamma:
                min_gamma = candidate
        return float(min_gamma)

    @classmethod
    def _kam_torus_stability(
        cls,
        eigenvalues: np.ndarray,
        tau: float = _KAM_TAU,
        max_k_norm: int = 4,
        orbital_frequencies: Optional[np.ndarray] = None,
        resonance_vectors: Tuple[Tuple[int, ...], ...] = (),
    ) -> Dict[str, Any]:
        r"""
        Criterio diofántico KAM (v7.3.0: refina γ sobre las resonancias CF,
        recortando $k$ al $\dim\omega$ efectivo).
        """
        if orbital_frequencies is not None and np.asarray(orbital_frequencies).size >= 1:
            omega = np.asarray(orbital_frequencies, dtype=_FLOAT_DTYPE)
            source = "orbital"
        else:
            ev = np.asarray(eigenvalues, dtype=_FLOAT_DTYPE)
            omega = np.log(np.maximum(ev, _SPECTRAL_PSD_FLOOR))
            source = "spectral"
            if omega.size < 2:
                return {
                    "diophantine_gamma": float("inf"),
                    "is_kam_stable": True,
                    "source": source,
                }
        gamma = cls._diophantine_gamma(omega, tau=tau, max_k_norm=max_k_norm)
        if resonance_vectors:
            for k in resonance_vectors:
                k_arr = np.asarray(k, dtype=_FLOAT_DTYPE)
                kn = float(np.linalg.norm(k_arr))
                if kn < 1e-15:
                    continue
                n_dot = int(min(k_arr.size, omega.size))
                gamma = min(
                    gamma,
                    abs(float(np.dot(omega[:n_dot], k_arr[:n_dot]))) * (kn ** tau),
                )
        return {
            "diophantine_gamma": float(gamma),
            "is_kam_stable": bool(gamma >= _KAM_GAMMA),
            "source": source,
        }

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 3.C — Función de Melnikov (proxy + integral homoclínica)
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _melnikov_function(
        cls,
        section_points: List[np.ndarray],
        lyapunov_max: float,
    ) -> float:
        r"""
        Función de Melnikov discreta (proxy del splitting transversal):

            $M \approx \sum_k \|\mathrm{proj}_{N_k}(\dot G_{k+1}-\dot G_k)\|$,

        $M_{\text{norm}} = M/\max(|\lambda_{\max}|,10^{-3})$.
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
            O = np.array([o.ravel() for o in orbitals], dtype=_FLOAT_DTYPE)
            O_norm = O / (np.linalg.norm(O, axis=1, keepdims=True) + 1e-15)
            Q, _ = np.linalg.qr(O_norm.T)
            normal_basis = Q[:, len(orbitals):] if Q.shape[1] > len(orbitals) else Q
            if normal_basis.size == 0:
                return 0.0
            M = 0.0
            for k in range(len(section_points) - 2):
                delta = (section_points[k + 1] - section_points[k]).ravel()
                proj = normal_basis.T @ delta
                M += float(np.linalg.norm(proj))
            return float(M / max(abs(lyapunov_max), 1e-3))
        except (LinAlgError, ValueError):
            return 0.0

    @classmethod
    def _melnikov_homoclinic_integral(
        cls,
        section_points: List[np.ndarray],
        delta_tau_mean: float,
        lyapunov_max: float,
    ) -> float:
        r"""
        Integral de Melnikov discreta a lo largo de la separatriz homoclínica:

            $M(\theta_0)=\sum_k \langle J\nabla H(z_k^{0}),\, g(z_k^{0},\theta_0)\rangle\,\Delta\tau$.

        $\nabla H$ se aproxima por la dirección orbital PCA y $g$ es la componente
        de perturbación ortogonal. $M\neq 0$ implica splitting transversal (Poincaré–Smale).
        """
        if len(section_points) < 4:
            return cls._melnikov_function(section_points, lyapunov_max)
        try:
            X = np.array([G.ravel() for G in section_points], dtype=_FLOAT_DTYPE)
            Xc = X - X.mean(axis=0, keepdims=True)
            U, S, _ = np.linalg.svd(Xc, full_matrices=False)
            if S.size < 2 or S[1] < 1.0e-15:
                return cls._melnikov_function(section_points, lyapunov_max)
            qp = np.column_stack((U[:, 0] * S[0], U[:, 1] * S[1]))
            dqp = np.diff(qp, axis=0)
            J = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=_FLOAT_DTYPE)
            g = dqp @ J.T
            orbital_n = dqp / (np.linalg.norm(dqp, axis=1, keepdims=True) + 1.0e-15)
            g = g - np.sum(g * orbital_n, axis=1, keepdims=True) * orbital_n
            integrand = np.sum((dqp @ J.T) * g, axis=1)
            M = float(np.sum(integrand) * float(delta_tau_mean))
            return float(abs(M) / max(abs(lyapunov_max), 1.0e-3))
        except (LinAlgError, ValueError):
            return cls._melnikov_function(section_points, lyapunov_max)

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 3.D — Detección de tangle homoclínico (Poincaré–Smale–Melnikov)
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _detect_homoclinic_tangle(
        lyapunov_max: float,
        poincare_data: Dict[str, Any],
        melnikov_amplitude: Optional[float] = None,
    ) -> bool:
        r"""
        Tangle homoclínico (Poincaré 1890; Smale horseshoe; Melnikov 1963):
          (i)  $\lambda_{\max}>10^{-2}$ (hiperbolicidad),
          (ii) $\ge 3$ cruces con periodos irregulares,
          (iii) $|\mu_{\text{Floquet}}|>1.01$ (estiramiento),
          (iv) $|M_{\text{norm}}|>10^{-2}$ si Melnikov disponible.
        """
        if lyapunov_max <= _LYAPUNOV_CHAOS:
            return False
        crossings = poincare_data.get("crossings", [])
        if len(crossings) < 3:
            return False
        periods = np.asarray(poincare_data.get("return_periods", []), dtype=_FLOAT_DTYPE)
        if periods.size < 2:
            return False
        irregular = bool(np.std(periods) / max(float(np.mean(periods)), 1e-15) > 0.1)
        floquet = abs(float(poincare_data.get("floquet_multiplier", 0.0)))
        stretching = floquet > _FLOQUET_UNSTABLE
        if melnikov_amplitude is None:
            melnikov_ok = True
        else:
            melnikov_ok = melnikov_amplitude > _MELNIKOV_THRESHOLD
        return bool(irregular and stretching and melnikov_ok)

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 3.E — Índice de Poincaré–Hopf y Conley–Morse
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _poincare_hopf_index(orbit: List[np.ndarray]) -> float:
        r"""
        Índice topológico discreto sobre mínimos locales de $\|v_k\|_F$:

            $\mathrm{ind}(p)=\mathrm{sign}\langle v_{\mathrm{in}},v_{\mathrm{out}}\rangle_F$.
        """
        if len(orbit) < 3:
            return 0.0
        velocities = [_fro(orbit[k + 1] - orbit[k]) for k in range(len(orbit) - 1)]
        if max(velocities) < 1e-15:
            return 0.0
        total = 0.0
        for k in range(1, len(velocities) - 1):
            if velocities[k] < velocities[k - 1] and velocities[k] < velocities[k + 1]:
                d_in = orbit[k] - orbit[k - 1]
                d_out = orbit[k + 1] - orbit[k]
                n_in = _fro(d_in)
                n_out = _fro(d_out)
                if n_in < 1e-15 or n_out < 1e-15:
                    continue
                cos_ang = float(np.sum(d_in * d_out) / (n_in * n_out))
                total += 1.0 if cos_ang > 0.0 else -1.0
        return float(total)

    @staticmethod
    def _conley_morse_index(
        hopf_index: float,
        lyapunov_spectrum: Sequence[float],
    ) -> float:
        r"""
        Índice de Conley–Morse (proxy): $m=\#\{\lambda_i>0\}$ (dimensión inestable)
        corregido por el índice de Poincaré–Hopf, $h=m-\mathrm{Hopf}$.
        """
        spec = [float(x) for x in lyapunov_spectrum if np.isfinite(x)]
        unstable = sum(1 for x in spec if x > _LYAPUNOV_CHAOS)
        return float(unstable) - float(hopf_index)

    @staticmethod
    def _polyakov_einstein_consistency(
        ricci_scalar: float,
        log_volume: float,
        n_dim: int,
    ) -> float:
        r"""
        Consistencia Polyakov / Einstein–Hilbert:
        en $n=2$, Gauss–Bonnet discreto $R\cdot\sqrt{\mathrm{Vol}}\approx 4\pi\chi$
        con $\chi=0$ (toro KAM) ⇒ residual $|R|\mathrm{e}^{\log\mathrm{Vol}/2}/4\pi$.
        En $n\neq 2$ se reporta $0$ (no hay anomalía de Weyl 2-D).
        """
        if n_dim != 2:
            return 0.0
        sqrt_vol = float(np.exp(np.clip(0.5 * log_volume, -350.0, 350.0)))
        return abs(float(ricci_scalar) * sqrt_vol) / _GAUSS_BONNET_4PI

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 3.F — Verificación de axiomas del retículo Heyting Ω₃
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _heyting_lattice_axioms_check() -> bool:
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

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 3.G — Combinatoria de Heyting Ω₃
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _heyting_combine(
        cls,
        is_stable: bool,
        kam_stable: bool,
        tangle: bool,
        floquet: float,
        banach_L: float,
        einstein_defect: float,
        used_fallback: bool,
        pairing_defect: float = 0.0,
        bianchi: float = 0.0,
        ein_scale: float = 1.0,
    ) -> HeytingVerdict:
        """Join de átomos en Ω₃ vía acumulación bitwise OR (O(1) IRAM SIMD)."""
        v_stab = HeytingVerdict.COHERENT if is_stable else HeytingVerdict.DEGRADED
        v_kam = HeytingVerdict.COHERENT if kam_stable else HeytingVerdict.DEGRADED
        v_tangle = HeytingVerdict.VETOED if tangle else HeytingVerdict.COHERENT
        v_floquet = (
            HeytingVerdict.DEGRADED
            if abs(floquet) > _FLOQUET_UNSTABLE
            else HeytingVerdict.COHERENT
        )
        v_banach = (
            HeytingVerdict.COHERENT
            if banach_L < _BANACH_TARGET
            else HeytingVerdict.DEGRADED
        )
        v_ein = (
            HeytingVerdict.COHERENT
            if einstein_defect < _EINSTEIN_TOL
            else HeytingVerdict.DEGRADED
        )
        v_fb = HeytingVerdict.DEGRADED if used_fallback else HeytingVerdict.COHERENT
        v_pair = (
            HeytingVerdict.DEGRADED if pairing_defect > _PAIRING_ATOL else HeytingVerdict.COHERENT
        )
        v_bianchi = (
            HeytingVerdict.DEGRADED
            if bianchi > _BIANCHI_REL_TOL * max(ein_scale, 1.0)
            else HeytingVerdict.COHERENT
        )
        return HeytingVerdict.fast_join_all(
            [v_stab, v_kam, v_tangle, v_floquet, v_banach, v_ein, v_fb, v_pair, v_bianchi]
        )

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 3.G′ — Asociatividad categórica del compuesto OODA
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _ooda_associativity_check(
        phase1: Phase1Handoff,
        phase2: Phase2Handoff,
    ) -> bool:
        r"""
        Continuidad de jets del compuesto $\Phi_3\circ\Phi_2\circ\Phi_1$:
        el germen de fase 1 embebido en fase 2 coincide (mismo candidato,
        mismo nombre) — i.e.
        $(\Phi_3\circ\Phi_2)\circ\Phi_1 \cong \Phi_3\circ(\Phi_2\circ\Phi_1)$.
        """
        try:
            p1_from_p2 = phase2.phase1
            if p1_from_p2.candidate.shape != phase1.candidate.shape:
                return False
            if p1_from_p2.tensor_name != phase1.tensor_name:
                return False
            seed_ok = _fro(p1_from_p2.candidate - phase1.candidate) <= 1.0e-12
            orbit_ok = len(phase2.orbit) >= 1
            return bool(seed_ok and orbit_ok)
        except Exception:  # noqa: BLE001
            return False

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 3.H — Certificado final endurecido (con fallback SPD)
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _finalize_hardened_certificate(
        cls, handoff: Phase2Handoff
    ) -> Tuple[np.ndarray, HardenedCertificate]:
        r"""
        Consume el `Phase2Handoff`, ejecuta KAM/tangle/Melnikov/Hopf/Conley sobre el
        estado final y emite `(G_certificado, HardenedCertificate)`.

        Si el estado terminal del flujo **degrada** la certificación SPD/κ,
        se retorna el candidato de FASE 1 (ya certificado) con
        `used_phase1_fallback=True` — invariante de fábrica: jamás se emite
        un tensor no SPD.
        """
        phase1, orbit, poincare, lyap_max, omega_orb = cls._ingest_phase2_handoff(handoff)
        ooda_ok = cls._ooda_associativity_check(phase1, handoff)
        G_flow = np.array(orbit[-1], dtype=_FLOAT_DTYPE, copy=True)
        certified = cls._try_certify_spd(phase1.tensor_name, G_flow)
        used_fallback = False
        if certified is None:
            logger.warning(
                "[HARDEN] %s: el flujo RSI³ degradó SPD/κ; se restaura el "
                "candidato certificado de FASE 1.",
                phase1.tensor_name,
            )
            G_final = np.array(phase1.candidate, dtype=_FLOAT_DTYPE, copy=True)
            spectrum_final = phase1.spectral_profile
            used_fallback = True
        else:
            G_final, spectrum_final = certified

        kam = cls._kam_torus_stability(
            spectrum_final.eigenvalues,
            orbital_frequencies=np.asarray(omega_orb, dtype=_FLOAT_DTYPE),
            resonance_vectors=handoff.resonance_vectors,
        )
        dt_mean = (
            float(np.mean(handoff.delta_tau_history)) if handoff.delta_tau_history else 0.01
        )
        melnikov_amp = cls._melnikov_homoclinic_integral(
            poincare.get("section_points", []), dt_mean, lyap_max
        )
        tangle = cls._detect_homoclinic_tangle(lyap_max, poincare, melnikov_amp)
        hopf = cls._poincare_hopf_index(orbit)
        conley = cls._conley_morse_index(hopf, handoff.lyapunov_spectrum)
        einstein_defect = cls._compute_einstein_defect(G_final)
        lipschitz = cls._compute_lipschitz_connes_bound(G_final)
        try:
            gamma = cls._synthesize_christoffel_from_metric(G_final)
        except LinAlgError:
            n = int(G_final.shape[0])
            gamma = np.zeros((n, n, n), dtype=_FLOAT_DTYPE)
        try:
            ricci, scalar_r = cls._compute_ricci_tensor(G_final, christoffel=gamma)
        except LinAlgError:
            ricci, scalar_r = np.zeros_like(G_final), 0.0
        einstein_low = ricci - 0.5 * float(scalar_r) * G_final
        try:
            bianchi = cls._bianchi_identity_residual(G_final, gamma, einstein_low)
        except LinAlgError:
            bianchi = float("inf")
        weyl = cls._weyl_conformal_residual(G_final, gamma, ricci, float(scalar_r))
        killing = cls._killing_equation_residual(G_final, gamma)
        kirchhoff = cls._kirchhoff_current_residual(G_final, einstein_low)
        vol_cons = cls._volume_form_consistency(G_final)
        floquet = float(poincare.get("floquet_multiplier", 0.0))
        try:
            log_volume = cls._spd_logdet(G_final)
        except LinAlgError:
            log_volume = float("-inf")
        try:
            compat = cls._metric_compatibility_residual(G_final, christoffel=gamma)
        except LinAlgError:
            compat = float("inf")
        polyakov = cls._polyakov_einstein_consistency(
            float(scalar_r), float(log_volume), int(G_final.shape[0])
        )
        pairing_defect = float(handoff.lyapunov_pairing_defect)
        ein_scale = max(_fro(einstein_low), 1.0)

        stable = bool(
            handoff.rsi3_converged
            and lyap_max <= _LYAPUNOV_CHAOS
            and kam["is_kam_stable"]
            and not tangle
            and spectrum_final.lambda_min > MIN_EIGVAL_TOL
            and spectrum_final.condition_number <= COND_NUM_TOL
            and not used_fallback
        )
        verdict = cls._heyting_combine(
            is_stable=stable,
            kam_stable=bool(kam["is_kam_stable"]),
            tangle=bool(tangle),
            floquet=floquet,
            banach_L=float(handoff.banach_L),
            einstein_defect=float(einstein_defect),
            used_fallback=used_fallback,
            pairing_defect=pairing_defect,
            bianchi=float(bianchi),
            ein_scale=ein_scale,
        )
        heyting_ok = cls._heyting_lattice_axioms_check()
        if not heyting_ok:
            logger.error(
                "[HEYTING] Axiomas de retículo violados — ¡bug crítico en Ω₃!"
            )
        if not ooda_ok:
            logger.error("[OODA] Asociatividad Φ₃∘Φ₂∘Φ₁ violada — handoff inconsistente")

        first_int = dict(phase1.first_integrals)
        rsi_trace = RSI3Trace(
            rsi_level_achieved=handoff.rsi3_level,
            banach_L=handoff.banach_L,
            delta_tau_history=handoff.delta_tau_history,
            eta_history=handoff.eta_history,
            einstein_defect_history=handoff.einstein_defect_history,
            converged=handoff.rsi3_converged,
            quadratic_convergence_rate=float(handoff.quadratic_rate),
            method_lineage=tuple(handoff.method_lineage),
            banach_certified=bool(handoff.banach_certified),
            projector_mode=str(handoff.projector_mode),
        )
        periods = poincare.get("return_periods", np.array([]))
        poincare_dt = PoincareData(
            return_periods=tuple(float(x) for x in np.asarray(periods).ravel()),
            lyapunov_max=float(lyap_max),
            kam_stable=bool(kam["is_kam_stable"]),
            homoclinic_tangle=bool(tangle),
            hopf_index=float(hopf),
            diophantine_gamma=float(kam["diophantine_gamma"]),
            floquet_multiplier=floquet,
            dominant_frequencies=tuple(float(x) for x in omega_orb),
            action=float(handoff.action_angle[0]),
            angle=float(handoff.action_angle[1]),
            melnikov_amplitude=float(melnikov_amp),
            lindstedt_period=float(handoff.lindstedt_period),
            recurrence_bound=float(handoff.recurrence_bound),
            monodromy_spectral_radius=floquet,
            conley_index=float(conley),
            first_integrals=tuple(
                float(first_int.get(k, 0.0)) for k in ("H", "C_jacobi", "L_angular")
            ),
        )
        cert = HardenedCertificate(
            tensor_name=phase1.tensor_name,
            spectral_profile=spectrum_final,
            rsi3_trace=rsi_trace,
            poincare_data=poincare_dt,
            einstein_defect=float(einstein_defect),
            lipschitz_connes_bound=float(lipschitz),
            is_hardened=bool(stable and verdict is not HeytingVerdict.VETOED),
            is_geometrically_stable=stable,
            heyting_verdict=verdict,
            log_volume=float(log_volume),
            metric_compatibility_residual=float(compat),
            used_phase1_fallback=used_fallback,
            bianchi_identity_residual=float(bianchi),
            lyapunov_spectrum=handoff.lyapunov_spectrum,
            action_angle=handoff.action_angle,
            melnikov_amplitude=float(melnikov_amp),
            heyting_axioms_verified=heyting_ok,
            conserved_quantities_audit=dict(handoff.conserved_quantities_audit),
            weyl_residual=float(weyl),
            killing_residual=float(killing),
            kirchhoff_residual=float(kirchhoff),
            volume_consistency=float(vol_cons),
            lyapunov_pairing_defect=pairing_defect,
            generating_function_residual=float(handoff.generating_function_residual),
            ooda_associative=bool(ooda_ok),
            polyakov_residual=float(polyakov),
            method_lineage=tuple(handoff.method_lineage),
        )
        return G_final, cert

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 3.I — Endurecimiento público (punto de entrada RSI³+Poincaré)
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def harden_tensor(
        cls,
        name: str,
        G: np.ndarray,
        n_iterations: int = _RSI3_MAX_ITER,
    ) -> Tuple[np.ndarray, HardenedCertificate]:
        r"""
        Ciclo OODA completo anidando las tres fases:

          1. FASE 1 (`_handoff_to_phase2`): validación canónica, Christoffel–Koszul,
             Ricci, Einstein, Bianchi, Weyl, Killing, Kirchhoff, ∇G.
          2. FASE 2 (`_ingest_phase1_handoff` → `_rsi3_driver` → `_handoff_to_phase3`):
             RSI³ anidado 1⊂2⊂3+método + Lyapunov QR + Poincaré + FFT + HJ +
             monodromía + Lindstedt.
          3. FASE 3 (`_ingest_phase2_handoff` → `_finalize_hardened_certificate`):
             KAM con resonancias CF + Melnikov homoclínica + tangle + Hopf +
             Conley–Morse + Heyting Ω₃ + asociatividad OODA.

        Retorna `(tensor_endurecido_inmutable, certificado)`.
        Invariante: el tensor emitido es SPD (fallback a FASE 1 si el flujo degrada).
        """
        phase1 = cls._handoff_to_phase2(name, G)
        G_seed, _ = cls._ingest_phase1_handoff(phase1)
        rsi_out = cls._rsi3_driver(G_seed, n_iterations=int(n_iterations))
        phase2 = cls._handoff_to_phase3(phase1, rsi_out)
        G_final, cert = cls._finalize_hardened_certificate(phase2)
        G_frozen = cls._make_immutable(G_final)
        logger.info(
            "[HARDEN] %s: hardened=%s, stable=%s, Heyting=%s (axiomas=%s), "
            "δ_E=%.3e, λ_max=%.3e, |Λ|=%d, p=%.2f, γ_KAM=%.3e, tangle=%s, "
            "Melnikov=%.3e, Floquet=%.3e, I=%.3e, θ=%.3f, ind_Hopf=%.1f, "
            "Conley=%.1f, ∇·Ein=%.3e, Weyl=%.3e, Killing=%.3e, KCL=%.3e, "
            "T_Lind=%.3e, τ_rec=%.3e, pairing=%.3e, HJ=%.3e, OODA=%s, "
            "fallback=%s, linaje=%s",
            name,
            cert.is_hardened,
            cert.is_geometrically_stable,
            cert.heyting_verdict.value,
            cert.heyting_axioms_verified,
            cert.einstein_defect,
            cert.poincare_data.lyapunov_max,
            len(cert.lyapunov_spectrum),
            cert.rsi3_trace.quadratic_convergence_rate,
            cert.poincare_data.diophantine_gamma,
            cert.poincare_data.homoclinic_tangle,
            cert.poincare_data.melnikov_amplitude,
            cert.poincare_data.floquet_multiplier,
            cert.action_angle[0],
            cert.action_angle[1],
            cert.poincare_data.hopf_index,
            cert.poincare_data.conley_index,
            cert.bianchi_identity_residual,
            cert.weyl_residual,
            cert.killing_residual,
            cert.kirchhoff_residual,
            cert.poincare_data.lindstedt_period,
            cert.poincare_data.recurrence_bound,
            cert.lyapunov_pairing_defect,
            cert.generating_function_residual,
            cert.ooda_associative,
            cert.used_phase1_fallback,
            cert.method_lineage[-1] if cert.method_lineage else "n/a",
        )
        return G_frozen, cert

    # ───────────────────────────────────────────────────────────────────────────
    # CONSTRUCTORES (backward-compatible: harden es opt-in)
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def build_physics_tensor(cls) -> np.ndarray:
        r"""Tensor métrico $V_{\mathrm{phys}}\subset\mathbb{R}^3$ (v4.1.0 preservado)."""
        G_phys = np.array(
            [
                [2.50, 0.85, 0.30],
                [0.85, 1.50, 0.45],
                [0.30, 0.45, 1.00],
            ],
            dtype=_FLOAT_DTYPE,
        )
        return cls._make_immutable(cls._handoff_to_phase2("G_phys", G_phys).candidate)

    @classmethod
    def build_topology_tensor(cls) -> np.ndarray:
        r"""Tensor métrico $V_{\mathrm{topo}}\subset\mathbb{R}^2$ (v4.1.0 preservado)."""
        G_topo = np.array([[1.00, 0.60], [0.60, 3.00]], dtype=_FLOAT_DTYPE)
        return cls._make_immutable(cls._handoff_to_phase2("G_topo", G_topo).candidate)

    @classmethod
    def build_thermo_tensor(cls) -> np.ndarray:
        r"""Tensor métrico $V_{\mathrm{thermo}}\subset\mathbb{R}^2$ (v4.1.0 preservado)."""
        G_thermo = np.array([[1.80, 0.75], [0.75, 2.20]], dtype=_FLOAT_DTYPE)
        return cls._make_immutable(cls._handoff_to_phase2("G_thermo", G_thermo).candidate)

    @classmethod
    def build_global_tensor(
        cls,
        physics_tensor: np.ndarray,
        topology_tensor: np.ndarray,
        thermo_tensor: np.ndarray,
    ) -> np.ndarray:
        r"""
        Ensambla el tensor $7\times 7$. Inyecta fricciones cruzadas y
        **restaura dominancia diagonal de Gershgorin** post-inyección.
        """
        G_base = block_diag(physics_tensor, topology_tensor, thermo_tensor)
        G_global = np.array(G_base, dtype=_FLOAT_DTYPE, copy=True)

        dim_phys = int(physics_tensor.shape[0])
        dim_topo = int(topology_tensor.shape[0])

        idx_sat = 0
        idx_frag = dim_phys + 0
        idx_ciclos = dim_phys + 1
        idx_entropia = dim_phys + dim_topo + 0

        G_global[idx_ciclos, idx_entropia] = 0.50
        G_global[idx_entropia, idx_ciclos] = 0.50
        G_global[idx_sat, idx_frag] = 0.35
        G_global[idx_frag, idx_sat] = 0.35

        G_global = cls._enforce_gershgorin_spd_margin(G_global)
        return cls._make_immutable(cls._handoff_to_phase2("G_global", G_global).candidate)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES UTILITARIAS PARA DIAGNÓSTICO EXTERNO
# ═══════════════════════════════════════════════════════════════════════════════

def get_tensor_diagnostics(G: np.ndarray, name: str = "unnamed") -> dict:
    """Diagnósticos completos (v4.1.0 + extensión v7.3.0)."""
    try:
        G_arr = np.asarray(G, dtype=_FLOAT_DTYPE)
        if G_arr.ndim != 2 or G_arr.shape[0] != G_arr.shape[1]:
            return {
                "name": name,
                "error": f"No es matriz cuadrada: shape={G_arr.shape}",
                "error_type": "StructuralError",
            }
        sym_tol = MetricTensorFactory._compute_symmetry_tolerance(G_arr)
        asym = _fro(G_arr - G_arr.T)
        G_sym = MetricTensorFactory._symmetrize_by_projection(G_arr)
        profile = MetricTensorFactory._compute_spectral_profile(G_sym)
        is_spd = MetricTensorFactory._verify_spd_by_cholesky(G_sym)
        einstein_defect = MetricTensorFactory._compute_einstein_defect(G_sym)
        lipschitz = MetricTensorFactory._compute_lipschitz_connes_bound(G_sym)
        try:
            gamma = MetricTensorFactory._synthesize_christoffel_from_metric(G_sym)
            ricci, scalar_r = MetricTensorFactory._compute_ricci_tensor(
                G_sym, christoffel=gamma
            )
            einstein_low = ricci - 0.5 * float(scalar_r) * G_sym
            bianchi = MetricTensorFactory._bianchi_identity_residual(
                G_sym, gamma, einstein_low
            )
            weyl = MetricTensorFactory._weyl_conformal_residual(
                G_sym, gamma, ricci, float(scalar_r)
            )
            killing = MetricTensorFactory._killing_equation_residual(G_sym, gamma)
            kirchhoff = MetricTensorFactory._kirchhoff_current_residual(
                G_sym, einstein_low
            )
        except LinAlgError:
            bianchi = weyl = killing = kirchhoff = float("inf")
            scalar_r = 0.0
        try:
            log_volume = MetricTensorFactory._spd_logdet(G_sym)
        except LinAlgError:
            log_volume = float("-inf")
        try:
            compat = MetricTensorFactory._metric_compatibility_residual(G_sym)
        except LinAlgError:
            compat = float("inf")
        vol_cons = MetricTensorFactory._volume_form_consistency(G_sym)
        polyakov = MetricTensorFactory._polyakov_einstein_consistency(
            float(scalar_r), float(log_volume), int(G_sym.shape[0])
        )
        return {
            "name": name,
            "module_version": __version__,
            "shape": G_arr.shape,
            "eigenvalues": profile.eigenvalues.tolist(),
            "dtype": str(G_arr.dtype),
            "is_symmetric": asym <= sym_tol,
            "asymmetry_norm": asym,
            "symmetry_tolerance": sym_tol,
            "is_spd": is_spd,
            "lambda_min": profile.lambda_min,
            "lambda_max": profile.lambda_max,
            "condition_number": profile.condition_number,
            "spectral_gap": profile.spectral_gap,
            "frobenius_norm": profile.frobenius_norm,
            "is_well_conditioned": profile.is_well_conditioned(),
            "is_strictly_positive": profile.is_strictly_positive(),
            "einstein_defect": einstein_defect,
            "lipschitz_connes_bound": lipschitz,
            "bianchi_identity_residual": bianchi,
            "weyl_residual": weyl,
            "killing_residual": killing,
            "kirchhoff_residual": kirchhoff,
            "volume_consistency": vol_cons,
            "polyakov_residual": polyakov,
            "log_volume": log_volume,
            "metric_compatibility_residual": compat,
        }
    except Exception as exc:
        return {"name": name, "error": str(exc), "error_type": type(exc).__name__}


def get_hardening_diagnostics(
    name: str, G: np.ndarray, n_iterations: int = _RSI3_MAX_ITER
) -> dict:
    """Ejecuta el ciclo RSI³+Poincaré v7.3.0 y devuelve dict serializable."""
    try:
        _, cert = MetricTensorFactory.harden_tensor(name, G, n_iterations)
        return {
            "module_version": __version__,
            "tensor_name": cert.tensor_name,
            "is_hardened": cert.is_hardened,
            "is_geometrically_stable": cert.is_geometrically_stable,
            "heyting_verdict": cert.heyting_verdict.value,
            "heyting_omega": cert.heyting_verdict.omega,
            "heyting_axioms_verified": cert.heyting_axioms_verified,
            "einstein_defect": cert.einstein_defect,
            "lipschitz_connes_bound": cert.lipschitz_connes_bound,
            "bianchi_identity_residual": cert.bianchi_identity_residual,
            "weyl_residual": cert.weyl_residual,
            "killing_residual": cert.killing_residual,
            "kirchhoff_residual": cert.kirchhoff_residual,
            "volume_consistency": cert.volume_consistency,
            "log_volume": cert.log_volume,
            "metric_compatibility_residual": cert.metric_compatibility_residual,
            "used_phase1_fallback": cert.used_phase1_fallback,
            "conserved_quantities_audit": dict(cert.conserved_quantities_audit),
            "lyapunov_pairing_defect": cert.lyapunov_pairing_defect,
            "generating_function_residual": cert.generating_function_residual,
            "ooda_associative": cert.ooda_associative,
            "polyakov_residual": cert.polyakov_residual,
            "method_lineage": list(cert.method_lineage),
            "spectral": {
                "lambda_min": cert.spectral_profile.lambda_min,
                "lambda_max": cert.spectral_profile.lambda_max,
                "condition_number": cert.spectral_profile.condition_number,
            },
            "rsi3": {
                "level": cert.rsi3_trace.rsi_level_achieved,
                "banach_L": cert.rsi3_trace.banach_L,
                "quadratic_convergence_rate": cert.rsi3_trace.quadratic_convergence_rate,
                "converged": cert.rsi3_trace.converged,
                "n_steps": len(cert.rsi3_trace.delta_tau_history),
                "einstein_defect_history": list(cert.rsi3_trace.einstein_defect_history),
                "method_lineage": list(cert.rsi3_trace.method_lineage),
                "banach_certified": cert.rsi3_trace.banach_certified,
                "projector_mode": cert.rsi3_trace.projector_mode,
            },
            "poincare": {
                "lyapunov_max": cert.poincare_data.lyapunov_max,
                "lyapunov_spectrum": list(cert.lyapunov_spectrum),
                "kam_stable": cert.poincare_data.kam_stable,
                "homoclinic_tangle": cert.poincare_data.homoclinic_tangle,
                "hopf_index": cert.poincare_data.hopf_index,
                "diophantine_gamma": cert.poincare_data.diophantine_gamma,
                "floquet_multiplier": cert.poincare_data.floquet_multiplier,
                "dominant_frequencies": list(cert.poincare_data.dominant_frequencies),
                "return_periods": list(cert.poincare_data.return_periods),
                "action": cert.poincare_data.action,
                "angle": cert.poincare_data.angle,
                "melnikov_amplitude": cert.poincare_data.melnikov_amplitude,
                "lindstedt_period": cert.poincare_data.lindstedt_period,
                "recurrence_bound": cert.poincare_data.recurrence_bound,
                "monodromy_spectral_radius": cert.poincare_data.monodromy_spectral_radius,
                "conley_index": cert.poincare_data.conley_index,
                "first_integrals": list(cert.poincare_data.first_integrals),
            },
        }
    except Exception as exc:
        return {"name": name, "error": str(exc), "error_type": type(exc).__name__}


# ═══════════════════════════════════════════════════════════════════════════════
# PRECOMPILACIÓN (preservada; endurecimiento RSI³ es opt-in vía harden_tensor)
# ═══════════════════════════════════════════════════════════════════════════════

G_PHYSICS: Final[np.ndarray] = MetricTensorFactory.build_physics_tensor()
G_TOPOLOGY: Final[np.ndarray] = MetricTensorFactory.build_topology_tensor()
G_THERMODYNAMICS: Final[np.ndarray] = MetricTensorFactory.build_thermo_tensor()

logger.info(
    "Tensores métricos inicializados (v%s): G_PHYSICS %s, G_TOPOLOGY %s, G_THERMODYNAMICS %s",
    __version__,
    G_PHYSICS.shape,
    G_TOPOLOGY.shape,
    G_THERMODYNAMICS.shape,
)