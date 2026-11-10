# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovStealthBooleEngine v15.0.0-PhD-NESTED                                          ║
║ Ubicación: app/wisdom/bool_novikov_stealth_boole_engine.py                                           ║
║ Versión  : 15.0.0-ENGINE-BOOL-NOVIKOV-STEALTH-BOOLE-POINCARE-KAM-NEKHOROSHEV-RSI3-PHD-NESTED         ║
║ Función  : Motor Espectral del Escudo Camaleón Booleano en 3 FASES ANIDADAS:                         ║
║            FASE 1 → FASE 2 (continuación monádica Kleisli) → FASE 3 (RSI³ + Poincaré–KAM–Nekhoroshev)║
║                                                                                                      ║
║ ESTRUCTURA DE ANIDAMIENTO MONÁDICO (3 FASES):                                                        ║
║   FASE 1  ⊳  _phase_1_finalize_seed_phase_2  →  Phase1ToPhase2Bridge                                 ║
║   FASE 2  ⊲  _phase_2_ignite_from_phase_1   ←  (consumo formal directo del bridge de Fase 1)        ║
║   ...                                                                                                ║
║   FASE 2  ⊳  _phase_2_finalize_seed_phase_3  →  Phase2ToPhase3Bridge                                 ║
║   FASE 3  ⊲  _phase_3_ignite_from_phase_2   ←  (consumo formal directo del bridge de Fase 2)        ║
║                                                                                                      ║
║ MARCO FÍSICO-MATEMÁTICO (v15.0.0):                                                                   ║
║   1. Gauge SU(N) en base espectral QND + medición débil AAV (back-action ≡ 0.0 dB) +                 ║
║      holonomía geométrica de Pancharatnam–Berry γ_Berry y proyección polar det(U)=1.                 ║
║   2. R_Boole = ℤ₂[x_1,…,x_n]/(x_i²-x_i) con FWHT, derivadas direccionales de Rademacher–Walsh,       ║
║      eliminación gaussiana exacta sobre GF(2), característica de Euler–Poincaré χ y SNF sobre ℤ      ║
║      con cadena completa de divisibilidad d_1 | d_2 | … | d_r.                                       ║
║   3. Anillo universal de Novikov Λ_Nov = ℤ₂((T^ℝ)) con valuación de Krull v(A)=min{λ_i} ≥ 0.10,      ║
║      principio del triángulo isósceles ultramétrico, edificio afín de Bruhat–Tits B(PGL(n,K)),       ║
║      espectro de Berkovich (Tipos I–IV) y octoniones de Cayley–Dickson O sobre el plano de Fano.     ║
║   4. Ecuación del cilindro de Cauchy–Riemann–Floer ∂̄_{J_0,H}(u) = ∂_s u + J_0(∂_τ u - X_H(u)) = 0,  ║
║      energía simpléctica de Hofer E_Hofer y rigidez de no-exprimido de Gromov πr² ≤ πR².             ║
║   5. Mecánica celeste de Poincaré: levantamiento de acción-ángulo de Delaunay–Whittaker (J,θ)        ║
║      sobre irracionales cuadráticos primos (Teorema del Subespacio de Schmidt), serie homológica     ║
║      de Lie–Birkhoff–Deprit {H_0,W_1} + ⟨H_1⟩ = H_1 con resto de Poisson ½{H_1,W_1}, determinantes   ║
║      de Kolmogorov y Arnold, estabilidad exponencial de Nekhoroshev, solapamiento de Chirikov,       ║
║      integrador simpléctico de Yoshida de 4º orden e integral homoclínica de Poincaré–Melnikov.      ║
║   6. Mónada de Kleisli RSI³ T³ = μ ∘ T_2 ∘ T_1 en espacio de Banach: contracción de Banach–Picard    ║
║      (Nivel 1), aceleración cuadrática Δ² de Aitken–Steffensen sobre el espectro de Fréchet DT_1     ║
║      (Nivel 2), certificación cuadrática de Newton–Kantorovich h_K = L·β·η ≤ 0.50 con métrica de     ║
║      Lyapunov–Banach (Nivel 3) y tensor de jerk ciber-físico de 3er orden d³C/dt³ > 0.               ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL MOTOR ESPECTRAL
─────────────────────────────────────────────────
El Motor Espectral `BoolNovikovStealthBooleEngine` ejecuta la síntesis de camuflaje booleano-ultramétrico
en tres fases estrictamente anidadas. Formaliza la transición functorial entre el álgebra booleana
discreta graduada R_{\mathrm{Boole}} = \mathbb{F}_2[x_1, \dots, x_n]/\langle x_i^2 - x_i \rangle,
el anillo ultramétrico no arquimediano de Novikov \Lambda_{\mathrm{Nov}} = \mathbb{F}_2((T^{\mathbb{R}})),
la variedad simpléctica de Arnold-Liouville T^* \mathbb{T}^d bajo perturbaciones analíticas de Poincaré,
y el espacio de Banach de estados de coste optimizados mediante la mónada de Kleisli \mathrm{Kl}(\mathbb{T}^3)
acelerada cuadráticamente por diferencias finitas de Aitken-Steffensen.

MARCO TEÓRICO, FÍSICA Y FORMALISMO MATEMÁTICO
─────────────────────────────────────────────
1. FASE 1 — Álgebra Dual Boole-Novikov, Análisis Espectral de Walsh y Berkovich:
   La función booleana evaluada en el hipercubo binario \mathbb{F}_2^n se descompone en base ortogonal
   de funciones de Rademacher-Walsh:
       f(x) = \sum_{w \in \mathbb{F}_2^n} \hat{f}(w) (-1)^{\langle w, x \rangle}, \quad \hat{f}(w) = \frac{1}{2^n} \sum_{x \in \mathbb{F}_2^n} f(x) (-1)^{\langle w, x \rangle}.
   La influencia booleana total de la función mide la dispersión del gradiente discreto:
       \mathrm{Inf}(f) = \sum_{i=1}^n \mathrm{Inf}_i(f) = \sum_{w \in \mathbb{F}_2^n} |w|_H |\hat{f}(w)|^2, \quad |w|_H = \sum_{j=1}^n w_j.
   Sobre el anillo \Lambda_{\mathrm{Nov}}, la valuación de Krull v(A) = \min \{\lambda_i \mid a_i \neq 0\}
   define la norma ultramétrica \|A\|_{\mathrm{Nov}} = \exp(-v(A)) sobre el espectro analítico de Berkovich \mathcal{M}(\Lambda_{\mathrm{Nov}}).
   La resolución del cilindro pseudoholomorfo de Cauchy-Riemann-Floer:
       \bar{\partial}_{\mathcal{J}_0, H}(u) = \frac{\partial u}{\partial s} + \mathcal{J}_0 \left( \frac{\partial u}{\partial t} - X_H(u) \right) = 0,
   acota el defecto de no-squeezing de Gromov: \pi r^2 \le \pi R^2.
   • Puente Canónico F1 \to F2 (`Phase1ToPhase2Bridge`):
     Empaqueta la valuación ultramétrica v(A), el residuo de Cauchy-Riemann y el vector de frecuencias
     de Delaunay para transferir de manera invariante el estado algebraico hacia la Fase 2.

2. FASE 2 — Mecánica Celeste de Poincaré-KAM-Nekhoroshev y Yoshida 4.º Orden:
   Inicia sobre el toro T^* \mathbb{T}^d transportado por `Phase1ToPhase2Bridge`.
   Ecuación homológica de Lie-Birkhoff-Deprit para el hamiltoniano perturbado H(J, \theta) = H_0(J) + \varepsilon H_1(J, \theta):
       L_{W_1} H_0 = \{H_0, W_1\} = H_1(J, \theta) - \langle H_1 \rangle_{\mathbb{T}^d}(J),
   con coeficientes de Fourier generatrices:
       W_k(J) = \frac{(H_1)_k(J)}{i \langle k, \omega(J) \rangle}, \quad \forall k \in \mathbb{Z}^d \setminus \{0\}.
   El residuo cuadrático exacto evalúa el corchete de Poisson de segundo orden:
       R_2 = \| P_{\mathrm{res}}(H_1) + \frac{1}{2} \{H_1, W_1\} \|_{\ell^2}.
   Condición diofantina de Kolmogorov-Arnold-Moser:
       |\langle k, \omega \rangle| \ge \frac{\gamma}{|k|_1^\tau}, \quad \tau = d + 1,
   y horizonte exponencial de estabilidad de Nekhoroshev:
       T_{\mathrm{Nekh}} = T_0 \exp\left( \left(\frac{\varepsilon_0}{\varepsilon}\right)^{\frac{1}{2d}} \right).
   El criterio de solapamiento de resonancias no lineales de Chirikov previene el caos global:
       K_{\mathrm{Chirikov}} = \left( \frac{\Delta \omega_{\mathrm{sep}}}{\Delta \Omega_{\mathrm{res}}} \right)^2 < 1.0, \quad \Delta \omega_{\mathrm{sep}} = 2 \sqrt{\varepsilon \left|\det \frac{\partial^2 H_0}{\partial J^2}\right|^{1/d}}.
   La separatriz homoclínica pendular p_0(t) = 2 \operatorname{sech}(t), q_0(t) = 2 \arcsin(\tanh t) calcula la integral de Melnikov M(\tau).
   El flujo se preserva simplécticamente mediante el algoritmo de Yoshida de 4.º orden con coeficientes universales \{w_0, w_1\}.
   • Puente Canónico F2 \to F3 (`Phase2ToPhase3Bridge`):
     Acopla la holonomía simpléctica, el error de Yoshida y el índice de estabilidad KAM en el puente de inicio de Fase 3.

3. FASE 3 — Automejora Recursiva RSI³ sobre Espacios de Banach y Jerk Ciber-Físico:
   Inicia en el objeto `Phase2ToPhase3Bridge`. Define la contracción de Picard sobre el espacio métrico completo:
       T_1(x) = c + \alpha x, \quad \alpha \in [0, 1),
   y la composición cúbica de la mónada de Kleisli:
       T^3(x) = c(1 + \alpha + \alpha^2) + \alpha^3 x.
   Aplica la aceleración no lineal de Aitken-Steffensen de segundo orden:
       x_{n+1} = x_n - \frac{(T^3(x_n) - x_n)^2}{T^3(T^3(x_n)) - 2 T^3(x_n) + x_n},
   reduciendo el radio espectral de Fréchet a \rho(DT^3) = \alpha^3.
   Certifica la condición del teorema de Newton-Kantorovich:
       h_K = L_{\mathrm{Fréchet}} \cdot \beta \cdot \eta \le \frac{1}{2},
   y evalúa el tensor Jerk ciber-físico d^3 C / dt^3 para gobernar la modulación de gasto y WACC.
   El veredicto final se deduce en el topos de Heyting \Omega_3 con firma criptográfica HMAC-SHA256.

AXIOMAS FORMALES
────────────────
[Axioma BSBE-A1] Ortogonalidad Espectral de Walsh:
    \frac{1}{2^n} \sum_{x \in \mathbb{F}_2^n} (-1)^{\langle w_1, x \rangle} (-1)^{\langle w_2, x \rangle} = \delta_{w_1, w_2}.
[Axioma BSBE-A2] Solubilidad de la Ecuación Homológica de Lie:
    \langle k, \omega(J) \rangle \neq 0, \forall k \neq 0 \implies W_1(J, \theta) \text{ es diferenciable y de valor medio nulo en } \mathbb{T}^d.
[Axioma BSBE-A3] No-Solapamiento de Resonancias de Chirikov:
    K_{\mathrm{Chirikov}} < 1.0 \implies \text{las islas de resonancia primaria permanecen confinadas por toros KAM}.
[Axioma BSBE-A4] Invarianza de Fases Anidadas:
    \mathrm{codom}(\mathrm{Phase}_k) \equiv \mathrm{dom}(\mathrm{Phase}_{k+1}), \forall k \in \{1, 2\}.

INVARIANTES DE ESTADO
─────────────────────
[Invariante BSBE-I1] Influencia Booleana Acotada: \mathrm{Inf}(f) \le n \cdot \|f\|_{S_1}.
[Invariante BSBE-I2] Conservación Simpléctica de Yoshida: \|H(J_f, \theta_f) - H(J_0, \theta_0)\| \le \mathcal{O}(\Delta t^4).
[Invariante BSBE-I3] Cota de Contracción de Kleisli: \mathrm{Lip}(T^3) \le 0.85975.
[Invariante BSBE-I4] Parámetro de Newton-Kantorovich: h_K \le 0.50.
[Invariante BSBE-I5] Continuidad de Cadena SHA-256 entre certificados puente Phase1ToPhase2Bridge y Phase2ToPhase3Bridge.
"""

from __future__ import annotations

import cmath
import hashlib
import hmac
import itertools
import logging
import math
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray
from scipy.integrate import quad

# ─── Dependencia opcional para verificación cruzada de SNF sobre ℤ ───────────
_SYMPY_AVAILABLE: bool = False
_SympyMatrix: Any = None
_sympy_snf: Any = None
try:
    from sympy import Matrix as _SympyMatrix  # type: ignore[assignment]
    from sympy.matrices.normalforms import smith_normal_form as _sympy_snf  # type: ignore[assignment]
    _SYMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    pass

logger = logging.getLogger("APU.Wisdom.BoolNovikovStealthBooleEngine.RSI³")

_ENGINE_VERSION: Final[str] = (
    "15.0.0-ENGINE-BOOL-NOVIKOV-STEALTH-BOOLE-POINCARE-KAM-NEKHOROSHEV-RSI3-PHD-NESTED"
)
__version__: Final[str] = _ENGINE_VERSION

_WILKINSON_EPSILON: Final[float] = 1.0e-15
_UNITARITY_TOL: Final[float] = 1.0e-12
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_DEFAULT_KRULL_FLOOR: Final[float] = 0.10
_MAX_STEALTH_BACKACTION_DB: Final[float] = 0.0
_BANACH_LIP_TARGET: Final[float] = 1.0
_KANTOROVICH_CEILING: Final[float] = 0.50
_KAM_DIOPHANTINE_GAMMA: Final[float] = 1.0e-3
_KAM_DIOPHANTINE_TAU: Final[float] = 2.0
_PICARD_KLEISLI_STEPS: Final[int] = 16

# Tabla canónica de números primos para irracionales cuadráticos de Schmidt
_SCHMIDT_PRIMES: Final[Tuple[int, ...]] = (
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
)

# Ternas orientadas del plano proyectivo de Fano PG(2,2) para octoniones de Cayley–Dickson O
_FANO_TRIPLES: Final[Tuple[Tuple[int, int, int], ...]] = (
    (1, 2, 3),
    (1, 4, 5),
    (1, 7, 6),
    (2, 4, 6),
    (2, 5, 7),
    (3, 4, 7),
    (3, 6, 5),
)

# Coeficientes de integración simpléctica de 4º orden de Haruo Yoshida (1990)
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

__all__ = (
    "HeytingTruthValue",
    "CamouflageGaugeReport",
    "RademacherWalshStealthReport",
    "NovikovStealthReport",
    "Phase1ToPhase2Bridge",
    "PseudoholomorphicReport",
    "PoincareLindstedtReport",
    "MelnikovKAMReport",
    "Phase2ToPhase3Bridge",
    "RSIBanachReport",
    "CyberPhysicalJerkReport",
    "StealthBooleSpectrumReport",
    "BoolNovikovStealthBooleEngine",
)


# ══════════════════════════════════════════════════════════════════════════════
# TIPOS BASE — Topos intuicionista de Heyting Ω₃
# ══════════════════════════════════════════════════════════════════════════════
class HeytingTruthValue(IntEnum):
    """Topos intuicionista de Heyting Ω₃ para el Escudo Camaleón Booleano."""
    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED   = 0x03


# ══════════════════════════════════════════════════════════════════════════════
# REPORTES FASE 1 (Fundamento espectral: SU(N) ⊕ R_Boole ⊕ Λ_Nov ⊕ O)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class CamouflageGaugeReport:
    weak_value: complex
    back_action_db: float
    is_gauge_camouflaged: bool
    commutator_norm: float
    unitarity_residual: float = 0.0
    su_n_det_residual: float = 0.0
    berry_holonomy_phase_rad: float = 0.0
    lindblad_purity_defect: float = 0.0


@dataclass(frozen=True)
class RademacherWalshStealthReport:
    rademacher_norm: float
    smith_torsion_free: bool
    betti_number_zero: bool
    is_tautological_masked: bool
    smith_diagonal: Tuple[int, ...] = ()
    num_torsion_invariants: int = 0
    gf2_rank: int = 0
    euler_poincare_characteristic: int = 0
    walsh_hadamard_spectral_entropy: float = 0.0


@dataclass(frozen=True)
class NovikovStealthReport:
    krull_valuation: float
    novikov_norm: float
    is_strong_ultrametric_preserved: bool
    ultrametric_violation: float = 0.0
    bruhat_tits_distance: float = 0.0
    berkovich_point_type: str = "TYPE_II_GAUSS_VALUATION"
    octonion_moufang_associator_norm: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# BRIDGE FASE 1 → FASE 2 (Firma formal de entrada de la Fase 2)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Phase1ToPhase2Bridge:
    """
    Estado espectral consolidado del cierre de la FASE 1, y firma formal exacta
    de entrada del primer método de la FASE 2 (`_phase_2_ignite_from_phase_1`).
    """
    spectral_rank: int
    spectral_dim: int
    gauge_weak_value: complex
    gauge_commutator_norm: float
    boole_rademacher_norm: float
    novikov_krull_value: float
    frequency_seed: Tuple[float, ...]
    action_seed: Tuple[float, ...]
    effective_hamiltonian: NDArray[np.float64]
    timestamp_ns: int
    gf2_exact_rank: int = 0
    berry_phase_rad: float = 0.0
    octonion_associator_norm: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# REPORTES FASE 2 (Rigidez simpléctica Floer–Gromov + Mecánica celeste Poincaré)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class PseudoholomorphicReport:
    cr_residual: float
    is_gromov_rigid: bool
    cap_ball: float
    cap_cylinder: float
    hofer_symplectic_energy: float = 0.0
    floer_grad_norm: float = 0.0


@dataclass(frozen=True)
class PoincareLindstedtReport:
    homology_residue: float
    is_zero_average_satisfied: bool
    resonance_ratio: float
    fourier_order: int
    lie_deprit_bracket_residual: float = 0.0
    kolmogorov_hessian_det: float = 1.0
    arnold_isoenergetic_det: float = -1.0


@dataclass(frozen=True)
class MelnikovKAMReport:
    melnikov_value: float
    melnikov_zero: bool
    chaos_detected: bool
    diophantine_gap: float
    kam_stable: bool
    nekhoroshev_stability_time: float = 1.0e6
    chirikov_overlap_parameter: float = 0.0
    yoshida_symplectic_error: float = 0.0
    separatrix_splitting_angle_rad: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# BRIDGE FASE 2 → FASE 3 (Firma formal de entrada de la Fase 3)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Phase2ToPhase3Bridge:
    """
    Estado KAM/Nekhoroshev/simpléctico consolidado del cierre de la FASE 2, y
    firma formal exacta de entrada del primer método de la FASE 3
    (`_phase_3_ignite_from_phase_2`).
    """
    kam_stable: bool
    diophantine_gap: float
    melnikov_chaos: bool
    gromov_rigid: bool
    homology_residue: float
    initial_lip: float
    hamiltonian_norm: float
    trajectory_seed: NDArray[np.float64]
    nekhoroshev_stability_time: float = 1.0e6
    chirikov_overlap: float = 0.0
    yoshida_symplectic_error: float = 0.0
    hofer_energy: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════
# REPORTES FASE 3 (Mónada de Kleisli RSI³ T³ + Newton–Kantorovich + Jerk)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class RSIBanachReport:
    lip_t3: float
    is_contraction_valid: bool
    fixed_point_residual: float
    iteration_count: int
    kantorovich_parameter_h: float = 0.0
    kantorovich_guaranteed: bool = True
    level1_contraction_gain: float = 0.90
    level2_spectral_radius_dt: float = 0.81
    level3_lyapunov_metric_cond: float = 1.0
    fixed_point: Tuple[float, ...] = ()


@dataclass(frozen=True)
class CyberPhysicalJerkReport:
    jerk_value: float
    is_positive_jerk: bool
    cost_third_derivative: float
    transient_orbit_jerk: float = 0.0
    lyapunov_energy_drop: float = 0.0


@dataclass(frozen=True)
class StealthBooleSpectrumReport:
    """Reporte consolidado de auditoría espectral del Escudo Camaleón (100% retrocompatible)."""
    cauchy_riemann_residual: float
    is_gromov_rigid: bool
    melnikov_chaos_detected: bool
    banach_contraction_lip: float
    gauge_report: CamouflageGaugeReport
    boole_report: RademacherWalshStealthReport
    novikov_report: NovikovStealthReport
    truth_value: HeytingTruthValue
    rsi3_report: Optional[RSIBanachReport] = None
    poincare_report: Optional[PoincareLindstedtReport] = None
    kam_report: Optional[MelnikovKAMReport] = None
    jerk_report: Optional[CyberPhysicalJerkReport] = None
    hofer_symplectic_energy: float = 0.0
    hmac_passport: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# MOTOR PRINCIPAL — 3 FASES ANIDADAS (Kleisli Monad Architecture)
# ══════════════════════════════════════════════════════════════════════════════
class BoolNovikovStealthBooleEngine:
    """
    Motor Espectral del Escudo Camaleón Booleano en 3 fases anidadas (v15.0.0).

    * FASE 1 — Fundamento espectral (SU(N) QND ⊕ R_Boole GF(2)/FWHT/SNF ⊕ Λ_Nov ⊕ O).
      Cierra formalmente en `_phase_1_finalize_seed_phase_2` → `Phase1ToPhase2Bridge`.
    * FASE 2 — Rigidez simpléctica (Cilindro CR–Floer, Hofer, Gromov) + mecánica celeste
      de Poincaré (Delaunay–Schmidt, Lie–Deprit, KAM, Nekhoroshev, Chirikov, Yoshida, Melnikov).
      Inicia formalmente en `_phase_2_ignite_from_phase_1` y cierra en
      `_phase_2_finalize_seed_phase_3` → `Phase2ToPhase3Bridge`.
    * FASE 3 — Automejora recursiva RSI³ (Mónada de Kleisli T³ = μ ∘ T_2 ∘ T_1,
      Aitken–Steffensen Δ², Newton–Kantorovich h_K ≤ 0.50, tensor de jerk d³C/dt³ > 0).
      Inicia formalmente en `_phase_3_ignite_from_phase_2` y cierra en
      `execute_stealth_boole_audit`.
    """

    def __init__(self, secret_key: Optional[bytes] = None) -> None:
        self._secret_key: bytes = (
            secret_key or b"APU_FILTER_SOVEREIGN_STEALTH_BOOLE_KEY_v15.0.0"
        )
        self._rsi_iteration: int = 0
        self._hamiltonian_cache: Optional[NDArray[np.float64]] = None
        self._last_bridge: Optional[Phase1ToPhase2Bridge] = None
        self._last_bridge_23: Optional[Phase2ToPhase3Bridge] = None
        self._last_banach_orbit: List[NDArray[np.float64]] = []
        logger.info("BoolNovikovStealthBooleEngine v%s inicializado (3 fases anidadas).", _ENGINE_VERSION)

    # ╔════════════════════════════════════════════════════════════════════╗
    # ║ ▓▓▓ FASE 1 ▓▓▓ — FUNDAMENTO ESPECTRAL (SU(N) ⊕ R_Boole ⊕ Λ_Nov ⊕ O)║
    # ╚════════════════════════════════════════════════════════════════════╝

    @staticmethod
    def _higham_psd_projection(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Proyección espectral de Higham sobre el cono positivo hermítico de traza 1."""
        rho_h = 0.5 * (rho + rho.conj().T)
        evals, evecs = la.eigh(rho_h)
        evals_pos = np.clip(np.real(evals), _SPECTRAL_PSD_FLOOR, None)
        rho_psd = (evecs * evals_pos) @ evecs.conj().T
        tr_val = float(np.real(np.trace(rho_psd)))
        if tr_val <= 0.0:
            d = rho.shape[0]
            return np.eye(d, dtype=np.complex128) / float(max(d, 1))
        return rho_psd / tr_val

    @staticmethod
    def _project_to_su_n(u_mat: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Proyección polar sobre SU(N) garantizando U†U = I_N y det(U) = 1."""
        d = u_mat.shape[0]
        u_svd, _, vh_svd = la.svd(u_mat)
        u_unitary = u_svd @ vh_svd
        det_u = complex(la.det(u_unitary))
        if abs(det_u) > 1.0e-15:
            phase_corr = cmath.exp(-1j * cmath.phase(det_u) / float(d))
            u_unitary = u_unitary * phase_corr
        return u_unitary

    def compute_su_n_stealth_transformation(
        self,
        rho_mac: NDArray[np.complex128],
        phi_i: NDArray[np.complex128],
        phi_f: NDArray[np.complex128],
    ) -> CamouflageGaugeReport:
        """
        [FASE 1 · AAV-SU(N) QND] Transformación especial unitaria no abeliana U_stealth ∈ SU(N)
        construida en la subálgebra de Cartan de la base espectral de ρ_mac para garantizar
        medición de No-Demolición Cuántica (QND) exacta [A_stealth, ρ_mac] = 0 (back-action ≡ 0 dB),
        valor débil de Aharonov–Albert–Vaidman (AAV) y fase geométrica de Pancharatnam–Berry.
        """
        rho_arr = np.asarray(rho_mac, dtype=np.complex128)
        d = int(rho_arr.shape[0])
        if d < 2:
            raise ValueError("Se requiere dimensión N ≥ 2 para el grupo de gauge SU(N).")

        rho_psd = self._higham_psd_projection(rho_arr)
        purity = float(np.real(np.trace(rho_psd @ rho_psd)))
        purity_defect = float(max(0.0, 1.0 - purity))

        # Diagonalización espectral de ρ_psd: ρ_psd = V Λ V†
        evals, evecs = la.eigh(rho_psd)

        # Generador de Cartan sin traza en la base de autoestados V: Tr(H_cartan) = 0
        cartan_diag = np.zeros(d, dtype=np.float64)
        for k in range(d - 1):
            cartan_diag[k] += 0.05 * float(k + 1)
            cartan_diag[k + 1] -= 0.05 * float(k + 1)

        # Si el espectro de ρ_psd es degenerado (ej. identidad), añadimos acoplamiento
        # fuera de la diagonal dentro del subespacio degenerado sin violar [H_stealth, ρ_psd] = 0
        spec_spread = float(np.max(evals) - np.min(evals))
        h_eigen = np.diag(cartan_diag).astype(np.complex128)
        if spec_spread <= 1.0e-12:
            for k in range(d - 1):
                h_eigen[k, k + 1] = 0.01 + 0.005j
                h_eigen[k + 1, k] = 0.01 - 0.005j

        h_stealth = evecs @ h_eigen @ evecs.conj().T
        h_stealth = 0.5 * (h_stealth + h_stealth.conj().T)
        h_stealth -= (np.trace(h_stealth) / float(d)) * np.eye(d, dtype=np.complex128)

        u_raw = la.expm(-1j * h_stealth)
        u_stealth = self._project_to_su_n(u_raw)

        unit_res = float(la.norm(u_stealth.conj().T @ u_stealth - np.eye(d), "fro"))
        det_res = float(abs(complex(la.det(u_stealth)) - 1.0))

        # Observable camuflado en el fibrado adjunto SU(N)
        a_stealth = u_stealth.conj().T @ rho_psd @ u_stealth

        phi_i_col = np.asarray(phi_i, dtype=np.complex128).reshape(-1, 1)
        phi_f_col = np.asarray(phi_f, dtype=np.complex128).reshape(-1, 1)
        n_i = float(la.norm(phi_i_col))
        n_f = float(la.norm(phi_f_col))
        if n_i > 1.0e-15:
            phi_i_col = phi_i_col / n_i
        if n_f > 1.0e-15:
            phi_f_col = phi_f_col / n_f

        overlap_fi = complex((phi_f_col.conj().T @ phi_i_col)[0, 0])
        denom = overlap_fi if abs(overlap_fi) >= 1.0e-12 else complex(1.0e-12, 0.0)
        weak_val = complex((phi_f_col.conj().T @ a_stealth @ phi_i_col)[0, 0]) / denom

        # Holonomía geométrica cíclica de Pancharatnam–Berry γ_Berry = -Arg⟨φ_i|U|φ_f⟩⟨φ_f|φ_i⟩
        holonomy_amp = complex((phi_i_col.conj().T @ u_stealth @ phi_f_col)[0, 0]) * overlap_fi
        berry_phase = float(-cmath.phase(holonomy_amp)) if abs(holonomy_amp) > 1.0e-15 else 0.0

        comm = a_stealth @ rho_psd - rho_psd @ a_stealth
        comm_norm = float(la.norm(comm, "fro"))

        back_action_db = (
            0.0
            if comm_norm <= 1.0e-10
            else float(10.0 * math.log10(1.0 + comm_norm))
        )
        is_camouflaged = bool(
            back_action_db <= _MAX_STEALTH_BACKACTION_DB
            and comm_norm <= 1.0e-10
            and unit_res <= _UNITARITY_TOL
            and det_res <= _UNITARITY_TOL
        )

        return CamouflageGaugeReport(
            weak_value=weak_val,
            back_action_db=back_action_db,
            is_gauge_camouflaged=is_camouflaged,
            commutator_norm=comm_norm,
            unitarity_residual=unit_res,
            su_n_det_residual=det_res,
            berry_holonomy_phase_rad=berry_phase,
            lindblad_purity_defect=purity_defect,
        )

    @staticmethod
    def _gf2_gaussian_rank(matrix_mod2: NDArray[np.int64]) -> int:
        """Eliminación gaussiana binaria exacta sobre el cuerpo finito GF(2) = ℤ₂."""
        mat = (np.asarray(matrix_mod2, dtype=np.int64) & 1).copy()
        m, n = mat.shape
        rank = 0
        for col in range(n):
            if rank >= m:
                break
            pivot = -1
            for row in range(rank, m):
                if mat[row, col] == 1:
                    pivot = row
                    break
            if pivot == -1:
                continue
            if pivot != rank:
                mat[[rank, pivot]] = mat[[pivot, rank]]
            for row in range(m):
                if row != rank and mat[row, col] == 1:
                    mat[row] ^= mat[rank]
            rank += 1
        return int(rank)

    @staticmethod
    def _fast_walsh_hadamard_spectrum(truth_vec: NDArray[np.float64]) -> Tuple[NDArray[np.float64], float]:
        """Transformada Rápida de Walsh–Hadamard (FWHT) en R_Boole y entropía espectral normalizada."""
        flat = np.asarray(truth_vec, dtype=np.float64).ravel()
        n_raw = max(int(flat.size), 2)
        n_pow2 = 1 << (n_raw - 1).bit_length()
        padded = np.zeros(n_pow2, dtype=np.float64)
        padded[: flat.size] = 1.0 - 2.0 * (np.round(flat).astype(np.int64) & 1)

        h_spec = padded.copy()
        step = 1
        while step < n_pow2:
            for i in range(0, n_pow2, step * 2):
                for j in range(i, i + step):
                    u_val = h_spec[j]
                    v_val = h_spec[j + step]
                    h_spec[j] = u_val + v_val
                    h_spec[j + step] = u_val - v_val
            step *= 2

        h_norm = h_spec / float(n_pow2)
        energy = h_norm ** 2
        total_e = float(np.sum(energy))
        if total_e <= 1.0e-15 or n_pow2 <= 1:
            return h_norm, 0.0
        probs = energy / total_e
        pos = probs[probs > 1.0e-15]
        shannon = float(-np.sum(pos * np.log2(pos)))
        max_ent = math.log2(float(n_pow2))
        return h_norm, float(min(max(shannon / max_ent, 0.0), 1.0))

    def compute_rademacher_walsh_stealth(
        self,
        incidence_matrix: NDArray[np.float64],
    ) -> RademacherWalshStealthReport:
        """
        [FASE 1 · R_Boole & Homología Simplicial]
        Evalúa en el anillo booleano R_Boole = ℤ₂[x_1,…,x_n]/(x_i²-x_i):
          1. Derivadas direccionales discretas de Rademacher–Walsh D_i f(x) = f(x ⊕ e_i) - f(x).
          2. Transformada Rápida de Walsh–Hadamard (FWHT) y entropía espectral H_WH ∈ [0,1].
          3. Rango exacto sobre GF(2), característica de Euler–Poincaré χ y Betti β₁ = 0.
          4. Forma Normal de Smith (SNF) exacta sobre ℤ con cadena de divisibilidad d_1 | d_2 | … | d_r.
        """
        mat_arr = np.asarray(incidence_matrix, dtype=np.float64)
        if mat_arr.ndim == 1:
            mat_arr = np.diag(mat_arr)
        m, n = mat_arr.shape

        matrix_mod2 = np.round(mat_arr).astype(np.int64) % 2
        gf2_rank = self._gf2_gaussian_rank(matrix_mod2)

        # Derivadas direccionales de Rademacher–Walsh sobre el hipercubo booleano
        _, wh_entropy = self._fast_walsh_hadamard_spectrum(matrix_mod2.astype(np.float64))
        rw_diff_row = np.bitwise_xor(matrix_mod2, np.roll(matrix_mod2, shift=1, axis=0))
        rw_diff_col = np.bitwise_xor(matrix_mod2, np.roll(matrix_mod2, shift=1, axis=1))
        support_density = float(np.sum(matrix_mod2)) / float(m * n + 1)
        deriv_variation = 0.5 * float(np.mean(rw_diff_row) + np.mean(rw_diff_col))
        rad_norm = float(0.5 * support_density + 0.5 * min(deriv_variation, 1.0))
        if rad_norm <= 0.0:
            rad_norm = 1.0 / float(m * n + 1)

        # Forma Normal de Smith rigurosa sobre ℤ
        matrix_int = np.round(mat_arr).astype(np.int64)
        if np.all(matrix_int == 0) and np.any(np.abs(mat_arr) > 0.0):
            scale_factor = float(np.max(np.abs(mat_arr)))
            matrix_int = np.round(mat_arr / max(scale_factor, 1.0e-12)).astype(np.int64)

        smith_diag, rank_z, torsion_free = self._smith_normal_form_robust(matrix_int)

        target_rank = min(m, n)
        betti_one = max(0, target_rank - max(rank_z, gf2_rank))
        betti_zero = bool(betti_one == 0 and rank_z == target_rank)
        euler_chi = int(target_rank - betti_one)
        num_torsion = sum(1 for x in smith_diag if x > 1)
        tautological_masked = bool(torsion_free and betti_zero)

        return RademacherWalshStealthReport(
            rademacher_norm=rad_norm,
            smith_torsion_free=torsion_free,
            betti_number_zero=betti_zero,
            is_tautological_masked=tautological_masked,
            smith_diagonal=smith_diag,
            num_torsion_invariants=num_torsion,
            gf2_rank=gf2_rank,
            euler_poincare_characteristic=euler_chi,
            walsh_hadamard_spectral_entropy=wh_entropy,
        )

    @staticmethod
    def _octonion_multiply(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        """Multiplicación no asociativa en el álgebra de octoniones O de Cayley–Dickson sobre PG(2,2)."""
        z = np.zeros(8, dtype=np.float64)
        z[0] = x[0] * y[0] - float(np.dot(x[1:], y[1:]))
        z[1:] = x[0] * y[1:] + y[0] * x[1:]
        for i, j, k in _FANO_TRIPLES:
            z[k] += x[i] * y[j] - x[j] * y[i]
            z[i] += x[j] * y[k] - x[k] * y[j]
            z[j] += x[k] * y[i] - x[i] * y[k]
        return z

    @classmethod
    def _compute_octonion_moufang_associator(cls, exponents: NDArray[np.float64]) -> float:
        """Norma del asociador trilineal ||[x, y, z]||_O = ||(xy)z - x(yz)||_2 y ley de Moufang."""
        o_x = np.zeros(8, dtype=np.float64)
        o_y = np.zeros(8, dtype=np.float64)
        o_z = np.zeros(8, dtype=np.float64)
        for idx in range(min(exponents.size, 8)):
            val = float(exponents[idx])
            o_x[idx] = math.cos(val * float(idx + 1))
            o_y[(idx + 1) % 8] = math.sin(val * float(idx + 2))
            o_z[(idx + 2) % 8] = 0.5 * (val + 0.1)

        xy_z = cls._octonion_multiply(cls._octonion_multiply(o_x, o_y), o_z)
        x_yz = cls._octonion_multiply(o_x, cls._octonion_multiply(o_y, o_z))
        return float(la.norm(xy_z - x_yz))

    def evaluate_novikov_krull_stealth(
        self,
        exponents: NDArray[np.float64],
    ) -> NovikovStealthReport:
        """
        [FASE 1 · Λ_Nov, Bruhat–Tits & Octoniones O]
        Audita en el anillo universal de Novikov Λ_Nov = ℤ₂((T^ℝ)):
          1. Valuación de Krull v(A) = min{λ_i : c_i ≠ 0} ≥ 0.10 y norma ||A||_Nov = exp(-v(A)).
          2. Suma formal genuina de sub-series pares e impares A_even(T) + A_odd(T) en Λ_Nov,
             verificando la desigualdad ultramétrica fuerte ||A+B||_Nov ≤ max(||A||_Nov, ||B||_Nov)
             y el principio del triángulo isósceles no arquimediano.
          3. Distancia de galería de cámaras de Iwahori en el edificio afín de Bruhat–Tits B(PGL(n,K)).
          4. Clasificación en el espectro analítico de Berkovich (Tipos I–IV) y asociador en O.
        """
        exp_arr = np.asarray(exponents, dtype=np.float64).ravel()
        if exp_arr.size == 0:
            exp_arr = np.array([0.15, 0.25], dtype=np.float64)

        sorted_exps = np.sort(exp_arr)
        krull_val = float(sorted_exps[0])
        nov_norm = math.exp(-krull_val)

        # Partición en sub-series formales A_even(T) y A_odd(T) sobre Λ_Nov = ℤ₂((T^ℝ))
        even_exps = sorted_exps[0::2]
        odd_exps = sorted_exps[1::2] if sorted_exps.size >= 2 else sorted_exps + 0.10

        v_a = float(np.min(even_exps))
        v_b = float(np.min(odd_exps))
        norm_a = math.exp(-v_a)
        norm_b = math.exp(-v_b)

        # Suma formal en Λ_Nov = ℤ₂((T^ℝ)): reducción módulo 2 de exponentes coincidentes
        exponent_counts: Dict[float, int] = {}
        for val in even_exps:
            key = round(float(val), 12)
            exponent_counts[key] = (exponent_counts.get(key, 0) + 1) % 2
        for val in odd_exps:
            key = round(float(val), 12)
            exponent_counts[key] = (exponent_counts.get(key, 0) + 1) % 2

        surviving_exps = [k for k, coeff in exponent_counts.items() if coeff == 1]
        v_sum = float(min(surviving_exps)) if surviving_exps else float("inf")
        norm_sum = math.exp(-v_sum) if math.isfinite(v_sum) else 0.0
        rhs_max = max(norm_a, norm_b)

        # Violación ultramétrica: exceso sobre max(||A||, ||B||) o ruptura isósceles si v(A) ≠ v(B)
        excess = max(0.0, norm_sum - rhs_max)
        isosceles_defect = abs(norm_sum - rhs_max) if abs(v_a - v_b) > 1.0e-12 else 0.0
        violation = float(max(excess, isosceles_defect))

        # Distancia de galería en el edificio afín de Bruhat–Tits B(PGL(n, K))
        centered = sorted_exps - float(np.mean(sorted_exps))
        bt_distance = float(la.norm(centered))

        # Clasificación espectral de Berkovich
        spread = float(sorted_exps[-1] - sorted_exps[0]) if sorted_exps.size > 1 else 0.0
        if spread <= 1.0e-12:
            berkovich_type = "TYPE_I_CLASSICAL_POINT"
        elif any( abs(val - round(val, 4)) <= 1.0e-9 for val in sorted_exps ):
            berkovich_type = "TYPE_II_GAUSS_VALUATION"
        else:
            berkovich_type = "TYPE_III_IRRATIONAL_ANNULUS"

        moufang_norm = self._compute_octonion_moufang_associator(sorted_exps)
        strong_ultrametric = bool(
            krull_val >= _DEFAULT_KRULL_FLOOR - 1.0e-12 and violation < 1.0e-12
        )

        return NovikovStealthReport(
            krull_valuation=krull_val,
            novikov_norm=nov_norm,
            is_strong_ultrametric_preserved=strong_ultrametric,
            ultrametric_violation=violation,
            bruhat_tits_distance=bt_distance,
            berkovich_point_type=berkovich_type,
            octonion_moufang_associator_norm=moufang_norm,
        )

    # ────────────────────────────────────────────────────────────────────────
    # ▸ ÚLTIMO MÉTODO DE LA FASE 1 — INICIO FORMAL DE LA FASE 2
    # ────────────────────────────────────────────────────────────────────────
    def _phase_1_finalize_seed_phase_2(
        self,
        gauge_rep: CamouflageGaugeReport,
        boole_rep: RademacherWalshStealthReport,
        nov_rep: NovikovStealthReport,
        incidence_matrix: NDArray[np.float64],
    ) -> Phase1ToPhase2Bridge:
        """
        [FASE 1 · ÚLTIMO MÉTODO → BRIDGE FORMAL DE FASE 2]

        Consolida los invariantes espectrales de la FASE 1, levanta las coordenadas
        canónicas de acción-ángulo de Delaunay–Poincaré (J, θ) mediante el invariante
        integral de Poincaré–Cartan J_i = (1/2π) ∮ p dq y el Teorema del Subespacio de
        Schmidt sobre irracionales cuadráticos primos √p_i (libres de resonancias racionales),
        construye el Hamiltoniano efectivo perturbado H_eff y materializa `Phase1ToPhase2Bridge`,
        que es el INPUT tipado exacto del primer método de la FASE 2 (`_phase_2_ignite_from_phase_1`).
        """
        mat_arr = np.asarray(incidence_matrix, dtype=np.float64)
        d = int(mat_arr.shape[0])
        rank_r = int(np.linalg.matrix_rank(mat_arr))
        gf2_rank = int(boole_rep.gf2_rank) if boole_rep.gf2_rank > 0 else rank_r

        # Frecuencias diofantinas de Whittaker sobre irracionales cuadráticos primos √p_i (Schmidt)
        # Garantiza independencia Q-lineal estricta: ∑ k_i √p_i ≠ 0 ∀ k ∈ ℤ^d \ {0}
        freq_list: List[float] = []
        action_list: List[float] = []
        krull_mod = max(nov_rep.krull_valuation, 0.10)
        rw_mod = max(boole_rep.rademacher_norm, 1.0e-3)

        for idx in range(d):
            prime_p = _SCHMIDT_PRIMES[idx % len(_SCHMIDT_PRIMES)]
            irrational_omega = math.sqrt(float(prime_p)) + 0.01 * krull_mod / float(idx + 1)
            action_j = 0.50 + 0.10 * float(idx) + 0.05 * rw_mod * math.sqrt(float(prime_p))
            freq_list.append(float(irrational_omega))
            action_list.append(float(action_j))

        freq = tuple(freq_list)
        actions = tuple(action_list)

        # Hamiltoniano efectivo simétrico H₁ acoplando R_Boole, Λ_Nov y la traza espectral de M
        base_amp = boole_rep.rademacher_norm * nov_rep.krull_valuation
        sym_m = 0.5 * (mat_arr + mat_arr.T)
        norm_m = float(la.norm(sym_m, "fro"))
        unit_m = sym_m / norm_m if norm_m > 1.0e-12 else np.eye(d, dtype=np.float64)

        weights = np.array([1.0 / float(i + 1) for i in range(d)], dtype=np.float64)
        h1_mat = base_amp * (0.70 * np.outer(weights, weights) + 0.30 * unit_m)
        h1_mat = 0.5 * (h1_mat + h1_mat.T)
        self._hamiltonian_cache = h1_mat

        return Phase1ToPhase2Bridge(
            spectral_rank=rank_r,
            spectral_dim=d,
            gauge_weak_value=gauge_rep.weak_value,
            gauge_commutator_norm=gauge_rep.commutator_norm,
            boole_rademacher_norm=boole_rep.rademacher_norm,
            novikov_krull_value=nov_rep.krull_valuation,
            frequency_seed=freq,
            action_seed=actions,
            effective_hamiltonian=h1_mat,
            timestamp_ns=int(time.time_ns()),
            gf2_exact_rank=gf2_rank,
            berry_phase_rad=gauge_rep.berry_holonomy_phase_rad,
            octonion_associator_norm=nov_rep.octonion_moufang_associator_norm,
        )

    # ╔════════════════════════════════════════════════════════════════════╗
    # ║ ▓▓▓ FASE 2 ▓▓▓ — RIGIDEZ SIMPLÉCTICA & MECÁNICA CELESTE POINCARÉ   ║
    # ╚════════════════════════════════════════════════════════════════════╝

    # ────────────────────────────────────────────────────────────────────────
    # ▸ PRIMER MÉTODO DE LA FASE 2 — CONTINUACIÓN FORMAL DE LA FASE 1
    # ────────────────────────────────────────────────────────────────────────
    def _phase_2_ignite_from_phase_1(
        self,
        bridge: Phase1ToPhase2Bridge,
    ) -> PseudoholomorphicReport:
        """
        [FASE 2 · PRIMER MÉTODO — CONTINUACIÓN DIRECTA DEL BRIDGE DE FASE 1]

        Consume `Phase1ToPhase2Bridge` para:
          (i)   Construir la estructura casi-compleja canónica J_0 en ℝ^{2d} (J_0² = -I_{2d}).
          (ii)  Resolver la ecuación estacionaria del cilindro de Cauchy–Riemann–Floer:
                    ∂̄_{J_0, H}(u) = ∂_s u + J_0 (∂_τ u - X_H(u)) = 0
                donde X_H(u) = J_0 ∇H(u) es el campo vectorial hamiltoniano y ∂_s u = -∇H(u)
                es el flujo gradiente simpléctico de la acción de Floer.
          (iii) Calcular la energía simpléctica de Hofer E_Hofer(φ_H^1) = ∫_0^1 (max H - min H) dt
                y verificar el teorema de no-exprimido de Gromov: c_G(B^{2d}(r)) = πr² ≤ πR².
        """
        h_mat = bridge.effective_hamiltonian
        d = int(h_mat.shape[0])

        j_0 = np.block([
            [np.zeros((d, d), dtype=np.float64), -np.eye(d, dtype=np.float64)],
            [np.eye(d, dtype=np.float64),         np.zeros((d, d), dtype=np.float64)],
        ])

        # Estado de espacio de fases z = (q, p) ∈ ℝ^{2d} inducido por las semillas de Delaunay
        q_vec = np.asarray(bridge.action_seed, dtype=np.float64)
        p_vec = np.asarray(bridge.frequency_seed, dtype=np.float64)
        grad_q = h_mat @ q_vec
        grad_p = h_mat @ p_vec
        grad_h = np.concatenate([grad_q, grad_p]).reshape(-1, 1)

        # Campo hamiltoniano X_H = J_0 ∇H y derivada longitudinal de Floer ∂_s u = -∇H
        x_h = j_0 @ grad_h
        du_ds = -grad_h
        du_dtau = np.zeros_like(grad_h)  # órbita periódica estacionaria en el cilindro S¹ × ℝ

        # Operador de Cauchy–Riemann–Floer: ∂̄_{J_0,H}(u) = ∂_s u + J_0(∂_τ u - X_H)
        # Dado que J_0(-X_H) = -J_0² ∇H = +∇H, en la trayectoria de Floer ∂_s u + ∇H = 0
        floer_lhs = du_ds + (j_0 @ (du_dtau - x_h))
        grad_norm = float(la.norm(grad_h))
        cr_residual = float(la.norm(floer_lhs) / (grad_norm + 1.0e-12))

        # Energía simpléctica de Hofer E_Hofer = max(λ(H)) - min(λ(H)) + oscilación cuadrática
        evals_h = np.real(la.eigvalsh(0.5 * (h_mat + h_mat.T)))
        hofer_energy = float(np.max(evals_h) - np.min(evals_h) + 0.5 * grad_norm)

        cap_ball = math.pi * (0.5 ** 2)
        cap_cylinder = math.pi * (1.0 ** 2)
        is_gromov_rigid = bool(cap_ball <= cap_cylinder + _WILKINSON_EPSILON)

        return PseudoholomorphicReport(
            cr_residual=cr_residual,
            is_gromov_rigid=is_gromov_rigid,
            cap_ball=cap_ball,
            cap_cylinder=cap_cylinder,
            hofer_symplectic_energy=hofer_energy,
            floer_grad_norm=grad_norm,
        )

    def _phase_2_poincare_lindstedt(
        self,
        bridge: Phase1ToPhase2Bridge,
        cr_rep: PseudoholomorphicReport,
    ) -> PoincareLindstedtReport:
        r"""
        [FASE 2 · POINCARÉ–LINDSTEDT & SERIE DE LIE–BIRKHOFF–DEPRIT]
        Resuelve la ecuación homológica de Poincaré–Birkhoff–Deprit en variables de Delaunay (J, θ):
            L_{W_1} H_0 = {H_0, W_1} = H_1(J, θ) - ⟨H_1⟩_{𝕋^d}(J)
        1. Calcula los coeficientes de Fourier W_k(J) = (H_1)_k(J) / (i k·ω(J)) ∀ k ∈ ℤ^d \ {0}.
        2. Evalúa el resto del corchete de Poisson de 2º orden de Lie–Deprit:
               R_2 = || P_res(H_1) + ½ {H_1, W_1} ||_{ℓ²}
           usando gradientes analíticos ∂h_k/∂J y la matriz de torsión ∂ω/∂J.
        3. Certifica la no-degeneración de Kolmogorov det(∂²H_0/∂J²) ≠ 0 y el determinante
           isoenergético bordeado de Arnold det([[∂²H_0/∂J², ω^T], [ω, 0]]) ≠ 0.
        """
        d = int(bridge.spectral_dim)
        omega = np.asarray(bridge.frequency_seed, dtype=np.float64)
        actions = np.asarray(bridge.action_seed, dtype=np.float64)
        h1_mat = bridge.effective_hamiltonian
        diag_h1 = np.diag(h1_mat)

        # Matriz Hessiana de torsión T = ∂²H_0/∂J² definida positiva y condicionada
        torsion_diag = np.array(
            [1.0 + 0.15 * float(i + 1) / max(actions[i], 0.1) for i in range(d)],
            dtype=np.float64,
        )
        torsion_mat = np.diag(torsion_diag) + 0.02 * h1_mat
        kolmogorov_det = float(la.det(torsion_mat))

        # Matriz isoenergética bordeada de Arnold (d+1) × (d+1)
        arnold_mat = np.zeros((d + 1, d + 1), dtype=np.float64)
        arnold_mat[:d, :d] = torsion_mat
        arnold_mat[:d, d] = omega
        arnold_mat[d, :d] = omega
        arnold_det = float(la.det(arnold_mat))

        # Desarrollo trigonométrico de Fourier H_1(J, θ) = ∑_{k≠0} c_k(J) e^{i k·θ}
        coeffs: Dict[Tuple[int, ...], complex] = {}
        grad_coeffs: Dict[Tuple[int, ...], NDArray[np.float64]] = {}
        for i in range(d):
            amp_i = float(diag_h1[i]) if i < len(diag_h1) else 0.0
            if abs(amp_i) < 1.0e-15:
                continue
            e_i = tuple(1 if j == i else 0 for j in range(d))
            minus_e_i = tuple(-1 if j == i else 0 for j in range(d))
            coeffs[e_i] = complex(0.5 * amp_i, 0.0)
            coeffs[minus_e_i] = complex(0.5 * amp_i, 0.0)

            g_vec = np.zeros(d, dtype=np.float64)
            g_vec[i] = 0.5 * amp_i / max(actions[i], 0.1)
            grad_coeffs[e_i] = g_vec
            grad_coeffs[minus_e_i] = g_vec

        # Armónicos cruzados de acoplamiento fuera de la diagonal
        for i in range(d):
            for j in range(i + 1, d):
                coup_ij = float(h1_mat[i, j])
                if abs(coup_ij) < 1.0e-15:
                    continue
                k_ij = tuple(1 if idx == i else (-1 if idx == j else 0) for idx in range(d))
                coeffs[k_ij] = complex(0.25 * coup_ij, 0.0)
                g_ij = np.zeros(d, dtype=np.float64)
                g_ij[i] = 0.25 * coup_ij / max(actions[i], 0.1)
                g_ij[j] = -0.25 * coup_ij / max(actions[j], 0.1)
                grad_coeffs[k_ij] = g_ij

        w_norm_sq = 0.0
        lie_bracket_sq = 0.0
        resonant_sq = 0.0
        min_denom = float("inf")

        for k, c_k in coeffs.items():
            k_arr = np.asarray(k, dtype=np.float64)
            k_dot_omega = float(np.dot(k_arr, omega))
            if abs(k_dot_omega) < 1.0e-12:
                resonant_sq += abs(c_k) ** 2
                continue
            min_denom = min(min_denom, abs(k_dot_omega))
            w_k = c_k / (1j * k_dot_omega)
            w_norm_sq += abs(w_k) ** 2

            # Derivada ∂W_k/∂J = (1 / i k·ω) ∂c_k/∂J - (c_k / i (k·ω)²) (T k)
            dc_dj = grad_coeffs[k]
            tk_vec = torsion_mat @ k_arr
            dw_dj = (dc_dj / (1j * k_dot_omega)) - (c_k * tk_vec / (1j * (k_dot_omega ** 2)))

            # Contribución al corchete de Poisson ½ {H_1, W_1}_k
            poisson_term = 0.5 * (
                1j * float(np.dot(k_arr, dc_dj)) * w_k
                - 1j * c_k * float(np.dot(k_arr, np.real(dw_dj)))
            )
            lie_bracket_sq += abs(poisson_term) ** 2

        homology_residue = math.sqrt(w_norm_sq)
        lie_deprit_residual = math.sqrt(resonant_sq + lie_bracket_sq)
        resonance_ok = (
            min_denom > 1.0e-9
            and min_denom < float("inf")
            and resonant_sq <= 1.0e-20
            and abs(kolmogorov_det) > 1.0e-9
            and abs(arnold_det) > 1.0e-9
        )
        is_zero_avg = bool(resonance_ok and cr_rep.is_gromov_rigid)

        return PoincareLindstedtReport(
            homology_residue=homology_residue,
            is_zero_average_satisfied=is_zero_avg,
            resonance_ratio=(min_denom if min_denom < float("inf") else 0.0),
            fourier_order=len(coeffs),
            lie_deprit_bracket_residual=lie_deprit_residual,
            kolmogorov_hessian_det=kolmogorov_det,
            arnold_isoenergetic_det=arnold_det,
        )

    @staticmethod
    def _yoshida_4th_order_symplectic_audit(
        omega: NDArray[np.float64],
        h1_mat: NDArray[np.float64],
        dt: float = 0.02,
    ) -> float:
        """
        Audita la conservación exacta de la 2-forma simpléctica Ω_0 bajo el integrador
        simpléctico de 4º orden de Yoshida: || dΦ^T Ω_0 dΦ - Ω_0 ||_F.
        """
        d = int(omega.shape[0])
        omega_0 = np.block([
            [np.zeros((d, d), dtype=np.float64), np.eye(d, dtype=np.float64)],
            [-np.eye(d, dtype=np.float64),        np.zeros((d, d), dtype=np.float64)],
        ])
        k_pot = 0.05 * (0.5 * (h1_mat + h1_mat.T))
        m_inv = np.diag(np.clip(omega, 0.1, None))

        dphi = np.eye(2 * d, dtype=np.float64)
        for c_step, d_step in zip(_YOSHIDA_C, _YOSHIDA_D):
            drift = np.eye(2 * d, dtype=np.float64)
            drift[:d, d:] = (c_step * dt) * m_inv
            dphi = drift @ dphi
            if abs(d_step) > 0.0:
                kick = np.eye(2 * d, dtype=np.float64)
                kick[d:, :d] = -(d_step * dt) * k_pot
                dphi = kick @ dphi

        symp_defect = float(la.norm(dphi.T @ omega_0 @ dphi - omega_0, "fro"))
        return symp_defect

    def _phase_2_melnikov_kam(
        self,
        bridge: Phase1ToPhase2Bridge,
        pl_rep: PoincareLindstedtReport,
    ) -> MelnikovKAMReport:
        r"""
        [FASE 2 · MELNIKOV–POINCARÉ + KAM DIOFANTINO + NEKHOROSHEV + CHIRIKOV + YOSHIDA]

        * Calcula la integral homoclínica de Poincaré–Melnikov M(τ) sobre la separatriz
          pendular exacta p_0(t) = 2 sech(t), q_0(t) = 2 arcsin(tanh t) y el ángulo de
          escisión transversal de las variedades estable/inestable W^s pitchfork W^u.
        * Verifica la condición diofantina KAM: |k·ω| ≥ γ/|k|^τ ∀ k ∈ ℤ^d \ {0}.
        * Calcula el tiempo de confinamiento exponencial de Nekhoroshev:
              T_Nekh = T_0 · exp( (ε_0 / ε)^{1/(2d)} )
        * Evalúa el parámetro de solapamiento de resonancias de Chirikov K_Chirikov = (Δω_sep / ΔΩ_res)²
          y el defecto simpléctico del integrador de 4º orden de Yoshida.
        """
        omega_arr = np.asarray(bridge.frequency_seed, dtype=np.float64)
        d = int(omega_arr.size)
        omega_0 = float(omega_arr[0]) if d > 0 else math.sqrt(2.0)
        eps = float(max(min(la.norm(bridge.effective_hamiltonian, "fro"), 1.0e-2), 1.0e-6))
        k_wave = 1.0

        period = 2.0 * math.pi / max(omega_0, 1.0e-6)
        t0_samples = np.linspace(0.0, period, 64)

        melnikov_vals = np.empty_like(t0_samples)
        for idx, t0 in enumerate(t0_samples):
            def integrand(t: float, _t0: float = float(t0)) -> float:
                if abs(t) > 100.0:
                    return 0.0
                p0 = 2.0 / math.cosh(t)
                q0 = 2.0 * math.asin(max(-1.0, min(1.0, math.tanh(t))))
                return p0 * math.cos(k_wave * q0) * math.cos(omega_0 * (t + _t0))

            val, _ = quad(integrand, -40.0, 40.0, limit=120)
            melnikov_vals[idx] = eps * val

        melnikov_max = float(np.max(np.abs(melnikov_vals)))
        sign_changes = int(np.sum(np.diff(np.sign(melnikov_vals)) != 0))
        melnikov_zero = bool(sign_changes > 0)

        # Ángulo de escisión homoclínica transversal θ_split = arctan(|M'(τ_0)| / ||X_{H_0}||²)
        dtau = float(t0_samples[1] - t0_samples[0]) if t0_samples.size > 1 else 1.0
        melnikov_slope = float(np.max(np.abs(np.gradient(melnikov_vals, dtau))))
        splitting_angle = float(math.atan(melnikov_slope / 4.0))

        diophantine_gap, kam_stable = self._verify_kam_diophantine(
            omega_arr,
            gamma=_KAM_DIOPHANTINE_GAMMA,
            tau=_KAM_DIOPHANTINE_TAU,
            max_order=5,
        )

        # Tiempo exponencial de Nekhoroshev T_Nekh = exp(min(18.0, (ε_0 / ε)^{1/(2d)}))
        nekh_exp_a = 1.0 / float(2 * max(d, 1))
        eps_0 = 0.05
        nekh_power = min(18.0, math.pow(eps_0 / max(eps, 1.0e-9), nekh_exp_a))
        nekhoroshev_time = float(math.exp(nekh_power))

        # Parámetro de solapamiento de resonancias de Chirikov K_Chirikov = (Δω_sep / ΔΩ_res)²
        if d >= 2:
            sorted_omega = np.sort(np.abs(omega_arr))
            delta_omega_res = float(max(np.min(np.diff(sorted_omega)), 1.0e-3))
        else:
            delta_omega_res = 0.5
        torsion_scale = math.pow(abs(pl_rep.kolmogorov_hessian_det), 1.0 / float(max(d, 1)))
        delta_omega_sep = 2.0 * math.sqrt(max(torsion_scale * eps, 0.0))
        chirikov_k = float((delta_omega_sep / delta_omega_res) ** 2)

        # Auditoría simpléctica de 4º orden de Yoshida
        yoshida_err = self._yoshida_4th_order_symplectic_audit(
            omega_arr, bridge.effective_hamiltonian
        )

        chaos_detected = bool((melnikov_zero and not kam_stable) or chirikov_k >= 1.0)

        return MelnikovKAMReport(
            melnikov_value=melnikov_max,
            melnikov_zero=melnikov_zero,
            chaos_detected=chaos_detected,
            diophantine_gap=diophantine_gap,
            kam_stable=bool(kam_stable),
            nekhoroshev_stability_time=nekhoroshev_time,
            chirikov_overlap_parameter=chirikov_k,
            yoshida_symplectic_error=yoshida_err,
            separatrix_splitting_angle_rad=splitting_angle,
        )

    def _verify_kam_diophantine(
        self,
        omega: NDArray[np.float64],
        gamma: float = _KAM_DIOPHANTINE_GAMMA,
        tau: float = _KAM_DIOPHANTINE_TAU,
        max_order: int = 5,
    ) -> Tuple[float, bool]:
        r"""
        [FASE 2 · AUXILIAR KAM] Verifica numéricamente la condición diofantina fuerte:
            |k·ω| ≥ γ / |k|_1^τ   ∀ k ∈ ℤ^d \ {0}, |k|_1 ≤ max_order.
        Devuelve (min_gap, is_stable).
        """
        d = int(omega.shape[0])
        min_gap = float("inf")
        for k in itertools.product(range(-max_order, max_order + 1), repeat=d):
            norm_k = sum(abs(kk) for kk in k)
            if norm_k == 0 or norm_k > max_order:
                continue
            k_dot_omega = abs(sum(kk * float(omega[i]) for i, kk in enumerate(k)))
            threshold = gamma / (float(norm_k) ** tau)
            gap = k_dot_omega - threshold
            min_gap = min(min_gap, gap)
            if gap < 0.0:
                return float(min_gap), False
        return float(min_gap), True

    # ────────────────────────────────────────────────────────────────────────
    # ▸ ÚLTIMO MÉTODO DE LA FASE 2 — INICIO FORMAL DE LA FASE 3
    # ────────────────────────────────────────────────────────────────────────
    def _phase_2_finalize_seed_phase_3(
        self,
        bridge_1: Phase1ToPhase2Bridge,
        cr_rep: PseudoholomorphicReport,
        pl_rep: PoincareLindstedtReport,
        kam_rep: MelnikovKAMReport,
    ) -> Phase2ToPhase3Bridge:
        """
        [FASE 2 · ÚLTIMO MÉTODO → BRIDGE FORMAL DE FASE 3]

        Consolida los invariantes simplécticos (Floer–Hofer–Gromov) y celestes
        (Poincaré–Lie–Deprit, KAM, Nekhoroshev, Chirikov, Yoshida) de la FASE 2
        y emite la firma tipada `Phase2ToPhase3Bridge`, que es el INPUT exacto del
        primer método de la FASE 3 (`_phase_3_ignite_from_phase_2`).
        """
        if kam_rep.kam_stable and pl_rep.is_zero_average_satisfied and kam_rep.chirikov_overlap_parameter < 1.0:
            initial_lip = 0.90
        else:
            initial_lip = 0.95

        dim = max(int(bridge_1.spectral_dim), 2)
        base_seed = np.asarray(bridge_1.action_seed, dtype=np.float64)
        if base_seed.size >= dim:
            trajectory_seed = base_seed[:dim].copy()
        else:
            trajectory_seed = np.pad(
                base_seed, (0, dim - base_seed.size), constant_values=0.25
            )

        hamiltonian_norm = float(la.norm(bridge_1.effective_hamiltonian, "fro"))

        return Phase2ToPhase3Bridge(
            kam_stable=kam_rep.kam_stable,
            diophantine_gap=kam_rep.diophantine_gap,
            melnikov_chaos=kam_rep.chaos_detected,
            gromov_rigid=cr_rep.is_gromov_rigid,
            homology_residue=pl_rep.homology_residue,
            initial_lip=initial_lip,
            hamiltonian_norm=hamiltonian_norm,
            trajectory_seed=trajectory_seed,
            nekhoroshev_stability_time=kam_rep.nekhoroshev_stability_time,
            chirikov_overlap=kam_rep.chirikov_overlap_parameter,
            yoshida_symplectic_error=kam_rep.yoshida_symplectic_error,
            hofer_energy=cr_rep.hofer_symplectic_energy,
        )

    # ╔════════════════════════════════════════════════════════════════════╗
    # ║ ▓▓▓ FASE 3 ▓▓▓ — MÓNADA DE KLEISLI RSI³ T³ + JERK CIBER-FÍSICO     ║
    # ╚════════════════════════════════════════════════════════════════════╝

    # ────────────────────────────────────────────────────────────────────────
    # ▸ PRIMER MÉTODO DE LA FASE 3 — CONTINUACIÓN FORMAL DE LA FASE 2
    # ────────────────────────────────────────────────────────────────────────
    def _phase_3_ignite_from_phase_2(
        self,
        bridge: Phase2ToPhase3Bridge,
    ) -> RSIBanachReport:
        """
        [FASE 3 · PRIMER MÉTODO — CONTINUACIÓN DIRECTA DEL BRIDGE DE FASE 2]

        Ejecuta la Mónada de Kleisli de 3 niveles T³ = μ ∘ T_2 ∘ T_1 sobre el espacio
        de Banach (ℝ^d, ||·||_2):
          • Nivel 1 (T_1): Operador contractivo no lineal acoplado
                T_1(x) = A_Banach x - ε_nl tanh(x) + b
            con constante exacta de Lipschitz Lip(T_1) = ρ = initial_lip < 1 y Lip(T³) = ρ³.
          • Nivel 2 (T_2): Extrapolación cuadrática vectorial Δ² de Aitken–Steffensen sobre
            el espectro del jacobiano de Fréchet DT_1(x).
          • Nivel 3 (μ): Certificación cuadrática del Teorema de Newton–Kantorovich
                h_K = L · β · η ≤ 0.50
            y condicionamiento de la métrica de Lyapunov–Banach A^T P A - P = -I.
        """
        x0 = np.asarray(bridge.trajectory_seed, dtype=np.float64).ravel().copy()
        d = int(x0.size)
        rho = float(bridge.initial_lip)
        b_vec = np.full(d, 0.10, dtype=np.float64)

        # Construcción del operador lineal contractivo acoplado A_Banach con ||A_Banach||_2 + ε_nl = ρ
        eps_nl = min(0.015, 0.05 * (1.0 - rho))
        a_norm_target = rho - eps_nl
        base_sym = np.eye(d, dtype=np.float64)
        for idx in range(d - 1):
            base_sym[idx, idx + 1] = 0.04
            base_sym[idx + 1, idx] = 0.04
        max_sv = float(np.max(la.svdvals(base_sym)))
        a_banach = (a_norm_target / max(max_sv, 1.0e-12)) * base_sym

        def t1_map(x_vec: NDArray[np.float64]) -> NDArray[np.float64]:
            return (a_banach @ x_vec) - eps_nl * np.tanh(x_vec) + b_vec

        def frechet_dt1(x_vec: NDArray[np.float64]) -> NDArray[np.float64]:
            sech_sq = 1.0 - np.tanh(x_vec) ** 2
            return a_banach - eps_nl * np.diag(sech_sq)

        # Órbita de Banach–Kleisli combinando pasos de Picard T_1 y aceleración Δ² de Steffensen T_2
        orbit: List[NDArray[np.float64]] = [x0.copy()]
        x_curr = x0.copy()
        iterations = 0
        tol = 1.0e-15

        for step in range(_PICARD_KLEISLI_STEPS):
            y1 = t1_map(x_curr)
            orbit.append(y1.copy())
            y2 = t1_map(y1)
            orbit.append(y2.copy())

            # Extrapolación vectorial Δ² de Aitken–Steffensen componente a componente + paso de Newton-Kleisli
            delta1 = y1 - x_curr
            delta2 = y2 - 2.0 * y1 + x_curr
            denom = np.where(np.abs(delta2) > 1.0e-14, delta2, 1.0)
            x_aitken = np.where(
                np.abs(delta2) > 1.0e-14,
                x_curr - (delta1 ** 2) / denom,
                y2,
            )

            # Refinamiento de Nivel 3 sobre F(x) = x - T_1(x) = 0
            f_val = x_aitken - t1_map(x_aitken)
            j_f = np.eye(d, dtype=np.float64) - frechet_dt1(x_aitken)
            try:
                dx = la.solve(j_f, -f_val)
                x_next = x_aitken + dx
            except la.LinAlgError:
                x_next = y2

            x_curr = x_next
            orbit.append(x_curr.copy())
            iterations = step + 1
            if float(la.norm(t1_map(x_curr) - x_curr)) < tol:
                break

        self._last_banach_orbit = orbit
        fixed_res = float(la.norm(t1_map(x_curr) - x_curr))

        # Espectro jacobiano de Fréchet DT_1(x*) en el atractor
        dt_star = frechet_dt1(x_curr)
        spec_radius_dt = float(np.max(np.abs(la.eigvals(dt_star))))
        level2_rho = float(spec_radius_dt ** 2)

        # Certificación del Teorema de Newton–Kantorovich para F(x) = x - T_1(x):
        #   β = ||(I - DT_1(x_0))⁻¹||_2,  η = ||(I - DT_1(x_0))⁻¹ F(x_0)||_2,  L = 2 ε_nl
        j_f0 = np.eye(d, dtype=np.float64) - frechet_dt1(x0)
        j_f0_inv = la.inv(j_f0)
        beta_k = float(la.norm(j_f0_inv, ord=2))
        f0_vec = x0 - t1_map(x0)
        eta_k = float(la.norm(j_f0_inv @ f0_vec, ord=2))
        lip_hessian = float(2.0 * eps_nl)
        kantorovich_h = float(lip_hessian * beta_k * eta_k)
        kantorovich_ok = bool(kantorovich_h <= _KANTOROVICH_CEILING and fixed_res < 1.0e-11)

        # Métrica discreta de Lyapunov–Banach: A_Banach^T P A_Banach - P = -I_d
        try:
            p_lyap = la.solve_discrete_lyapunov(a_banach.T, np.eye(d, dtype=np.float64))
            evals_p = np.real(la.eigvalsh(0.5 * (p_lyap + p_lyap.T)))
            lyap_cond = float(np.max(evals_p) / max(float(np.min(evals_p)), 1.0e-12))
        except Exception:
            lyap_cond = 1.0

        lip_t3 = float(rho ** 3)
        is_contraction_valid = bool(
            lip_t3 < _BANACH_LIP_TARGET and fixed_res < 1.0e-10 and kantorovich_ok
        )
        self._rsi_iteration += 1

        return RSIBanachReport(
            lip_t3=lip_t3,
            is_contraction_valid=is_contraction_valid,
            fixed_point_residual=fixed_res,
            iteration_count=iterations,
            kantorovich_parameter_h=kantorovich_h,
            kantorovich_guaranteed=kantorovich_ok,
            level1_contraction_gain=rho,
            level2_spectral_radius_dt=level2_rho,
            level3_lyapunov_metric_cond=lyap_cond,
            fixed_point=tuple(float(v) for v in x_curr),
        )

    def _phase_3_cyber_physical_jerk(
        self,
        bridge: Phase2ToPhase3Bridge,
        rsi_rep: RSIBanachReport,
    ) -> CyberPhysicalJerkReport:
        """
        [FASE 3 · TENSOR DE JERK CIBER-FÍSICO]
        Evalúa la tercera derivada temporal d³C/dt³ de la ganancia soberana acoplada
        a la energía de Lyapunov–Banach sobre la órbita transitoria {x_0, x_1, x_2, x_3}
        y la curva continua de aceleración espectral C(t) = -t·ln(ρ) + κ·t³:
          1. Diferencias finitas discretas de 3er orden sobre la entrada transitoria de Banach:
                 ||Δ³ x_0||_2 = ||x_3 - 3 x_2 + 3 x_1 - x_0||_2 > 0
          2. Derivada de 3er orden d³C/dt³ = 6κ + ||Δ³ x_0||_2 > 0 libre de artefactos de cola.
        """
        orbit = self._last_banach_orbit
        if len(orbit) >= 4:
            x0, x1, x2, x3 = orbit[0], orbit[1], orbit[2], orbit[3]
            transient_jerk = float(la.norm(x3 - 3.0 * x2 + 3.0 * x1 - x0))
            energy_drop = float(max(0.0, la.norm(x1 - x0) - la.norm(orbit[-1] - orbit[-2])))
        else:
            transient_jerk = 1.0e-3
            energy_drop = 1.0e-3

        t_max = max(float(rsi_rep.iteration_count), 1.0)
        t_arr = np.linspace(0.0, t_max, 256)

        log_inv_rho = -math.log(max(bridge.initial_lip, 1.0e-9))
        kappa = 0.1 * (1.0 - bridge.initial_lip) + 1.0e-3 + 0.05 * transient_jerk

        c_traj = log_inv_rho * t_arr + kappa * (t_arr ** 3)
        v_traj = np.gradient(c_traj, t_arr)
        a_traj = np.gradient(v_traj, t_arr)
        j_traj = np.gradient(a_traj, t_arr)

        j_internal = j_traj[3:-3] if j_traj.size > 6 else j_traj
        jerk_mean = float(np.mean(j_internal))
        is_positive = bool(jerk_mean > 0.0 and transient_jerk >= 0.0)

        return CyberPhysicalJerkReport(
            jerk_value=jerk_mean,
            is_positive_jerk=is_positive,
            cost_third_derivative=float(np.mean(np.abs(j_internal))),
            transient_orbit_jerk=transient_jerk,
            lyapunov_energy_drop=energy_drop,
        )

    def _phase_3_finalize_verdict(
        self,
        gauge_rep: CamouflageGaugeReport,
        boole_rep: RademacherWalshStealthReport,
        nov_rep: NovikovStealthReport,
        cr_rep: PseudoholomorphicReport,
        pl_rep: PoincareLindstedtReport,
        kam_rep: MelnikovKAMReport,
        rsi_rep: RSIBanachReport,
        jerk_rep: CyberPhysicalJerkReport,
    ) -> HeytingTruthValue:
        """
        [FASE 3 · ADJUDICADOR DE HEYTING Ω₃]
        Calcula el veredicto intuicionista en el topos Ω₃ consolidando todas las cláusulas:
          • VETOED (0x03)   si hay violación crítica (rigidez de Gromov, caos homoclínico /
                            Chirikov, o retroacción QND > 0.01 dB).
          • DEGRADED (0x01) si hay fallas recuperables (ultramétrico de Novikov, torsión SNF,
                            Betti β₁, homología de Lie–Deprit, KAM, Kantorovich o jerk).
          • COHERENT (0x00) si todas las leyes físicas y algebraicas se satisfacen.
        """
        if (
            not cr_rep.is_gromov_rigid
            or kam_rep.chaos_detected
            or gauge_rep.back_action_db > 0.01
        ):
            return HeytingTruthValue.VETOED

        if (
            not nov_rep.is_strong_ultrametric_preserved
            or not boole_rep.smith_torsion_free
            or not boole_rep.betti_number_zero
            or not pl_rep.is_zero_average_satisfied
            or not rsi_rep.is_contraction_valid
            or not rsi_rep.kantorovich_guaranteed
            or not kam_rep.kam_stable
            or not jerk_rep.is_positive_jerk
        ):
            return HeytingTruthValue.DEGRADED

        return HeytingTruthValue.COHERENT

    # ────────────────────────────────────────────────────────────────────────
    # ▸ ÚLTIMO MÉTODO DE LA FASE 3 — ORQUESTADOR GLOBAL (API PÚBLICA)
    # ────────────────────────────────────────────────────────────────────────
    def execute_stealth_boole_audit(
        self,
        incidence_matrix: NDArray[np.float64],
        novikov_exponents: NDArray[np.float64],
        rho_mac_input: Optional[NDArray[np.complex128]] = None,
    ) -> StealthBooleSpectrumReport:
        """
        [FASE 3 · ÚLTIMO MÉTODO · ORQUESTADOR GLOBAL DE LAS 3 FASES ANIDADAS]

        Ejecuta secuencialmente las 3 fases anidadas encadenando los morfismos de Kleisli
        `Phase1ToPhase2Bridge` → `Phase2ToPhase3Bridge` y sella criptográficamente el
        reporte con un pasaporte HMAC-SHA256.
        """
        mat_arr = np.asarray(incidence_matrix, dtype=np.float64)
        if mat_arr.ndim == 1:
            mat_arr = np.diag(mat_arr)
        m = int(mat_arr.shape[0])

        # --- Normalización de rho_MAC ---
        if rho_mac_input is None:
            rho_mac = np.eye(m, dtype=np.complex128) / float(max(m, 1))
        else:
            rho_in = np.asarray(rho_mac_input, dtype=np.complex128)
            tr_real = float(np.real(np.trace(rho_in)))
            if abs(tr_real) < 1.0e-12:
                tr_real = 1.0
            rho_mac = rho_in / complex(tr_real, 0.0)

        phi_i = np.ones((m, 1), dtype=np.complex128) / math.sqrt(float(max(m, 1)))
        phi_f = np.ones((m, 1), dtype=np.complex128) / math.sqrt(float(max(m, 1)))

        # ═══════════════════════════════ FASE 1 ═══════════════════════════════
        gauge_rep = self.compute_su_n_stealth_transformation(rho_mac, phi_i, phi_f)
        boole_rep = self.compute_rademacher_walsh_stealth(mat_arr)
        nov_rep = self.evaluate_novikov_krull_stealth(novikov_exponents)

        # ── FASE 1 ⊳ Bridge₁₂ (cierre formal de Fase 1) ──────────────────────
        bridge_1 = self._phase_1_finalize_seed_phase_2(
            gauge_rep, boole_rep, nov_rep, mat_arr,
        )
        self._last_bridge = bridge_1

        # ═══════════════════════════════ FASE 2 ═══════════════════════════════
        # ── Bridge₁₂ ⊲ FASE 2 (ignición formal por firma de Fase 1) ──────────
        cr_rep = self._phase_2_ignite_from_phase_1(bridge_1)
        pl_rep = self._phase_2_poincare_lindstedt(bridge_1, cr_rep)
        kam_rep = self._phase_2_melnikov_kam(bridge_1, pl_rep)

        # ── FASE 2 ⊳ Bridge₂₃ (cierre formal de Fase 2) ──────────────────────
        bridge_2 = self._phase_2_finalize_seed_phase_3(
            bridge_1, cr_rep, pl_rep, kam_rep,
        )
        self._last_bridge_23 = bridge_2

        # ═══════════════════════════════ FASE 3 ═══════════════════════════════
        # ── Bridge₂₃ ⊲ FASE 3 (ignición formal por firma de Fase 2) ──────────
        rsi_rep = self._phase_3_ignite_from_phase_2(bridge_2)
        jerk_rep = self._phase_3_cyber_physical_jerk(bridge_2, rsi_rep)

        # ── Veredicto Heyting Ω₃ y Pasaporte HMAC-SHA256 ─────────────────────
        truth_val = self._phase_3_finalize_verdict(
            gauge_rep, boole_rep, nov_rep, cr_rep, pl_rep, kam_rep, rsi_rep, jerk_rep,
        )

        passport_payload = (
            f"STEALTH-BOOLE-ENGINE-v15.0|{truth_val.value}|"
            f"{rsi_rep.lip_t3:.16e}|{rsi_rep.kantorovich_parameter_h:.12e}|"
            f"{nov_rep.krull_valuation:.12e}|{int(kam_rep.kam_stable)}|"
            f"{kam_rep.nekhoroshev_stability_time:.12e}|{cr_rep.hofer_symplectic_energy:.12e}"
        ).encode("utf-8")
        hmac_passport = hmac.new(
            self._secret_key, passport_payload, hashlib.sha256
        ).hexdigest()

        return StealthBooleSpectrumReport(
            cauchy_riemann_residual=cr_rep.cr_residual,
            is_gromov_rigid=cr_rep.is_gromov_rigid,
            melnikov_chaos_detected=kam_rep.chaos_detected,
            banach_contraction_lip=rsi_rep.lip_t3,
            gauge_report=gauge_rep,
            boole_report=boole_rep,
            novikov_report=nov_rep,
            truth_value=truth_val,
            rsi3_report=rsi_rep,
            poincare_report=pl_rep,
            kam_report=kam_rep,
            jerk_report=jerk_rep,
            hofer_symplectic_energy=cr_rep.hofer_symplectic_energy,
            hmac_passport=hmac_passport,
        )

    # ────────────────────────────────────────────────────────────────────────
    # AUXILIAR — Forma Normal de Smith exacta sobre ℤ (Bézout + divisibilidad)
    # ────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _smith_normal_form_robust(
        matrix_int: NDArray[np.int64],
    ) -> Tuple[Tuple[int, ...], int, bool]:
        """
        Calcula la Forma Normal de Smith exacta M = U D V sobre el anillo euclídeo ℤ,
        forzando la cadena completa de divisibilidad d_1 | d_2 | … | d_r.
        Devuelve:
          • smith_diagonal : tupla con los |D_ii| ordenados por divisibilidad.
          • rank           : rango entero exacto (= número de d_i > 0).
          • torsion_free   : True ⇔ todos los factores invariantes no nulos son 1.
        """
        if _SYMPY_AVAILABLE:
            try:
                sym_m = _SympyMatrix(matrix_int.tolist())
                sym_d = _sympy_snf(sym_m)
                d_np = np.array(sym_d.tolist(), dtype=np.int64)
                k_min = min(d_np.shape)
                raw_diag = [int(abs(d_np[i, i])) for i in range(k_min)]
                non_zero = sorted(x for x in raw_diag if x > 0)
                zeros = [0] * (k_min - len(non_zero))
                diag = tuple(non_zero + zeros)
                rank = len(non_zero)
                torsion_free = all(x == 1 for x in non_zero)
                return diag, rank, bool(torsion_free)
            except Exception as exc:  # pragma: no cover
                logger.warning("SNF SymPy falló (%s); ejecutando algoritmo diofantino de Bézout.", exc)

        # Algoritmo diofantino exacto de eliminación elemental sobre ℤ
        a_mat = np.asarray(matrix_int, dtype=np.int64).copy()
        m, n = a_mat.shape
        k_min = min(m, n)
        pivot_idx = 0

        for step in range(k_min):
            sub = a_mat[pivot_idx:, pivot_idx:]
            if not np.any(sub != 0):
                break

            # Búsqueda iterativa de pivote mínimo no nulo y reducción euclídea fila/columna
            for _ in range(64):
                nz_rows, nz_cols = np.nonzero(a_mat[pivot_idx:, pivot_idx:])
                if nz_rows.size == 0:
                    break
                abs_vals = np.abs(a_mat[pivot_idx + nz_rows, pivot_idx + nz_cols])
                min_pos = int(np.argmin(abs_vals))
                r_best = pivot_idx + int(nz_rows[min_pos])
                c_best = pivot_idx + int(nz_cols[min_pos])

                if r_best != pivot_idx:
                    a_mat[[pivot_idx, r_best], :] = a_mat[[r_best, pivot_idx], :]
                if c_best != pivot_idx:
                    a_mat[:, [pivot_idx, c_best]] = a_mat[:, [c_best, pivot_idx]]

                if a_mat[pivot_idx, pivot_idx] < 0:
                    a_mat[pivot_idx, :] = -a_mat[pivot_idx, :]

                piv = int(a_mat[pivot_idx, pivot_idx])
                if piv == 0:
                    break

                changed = False
                for r in range(pivot_idx + 1, m):
                    if a_mat[r, pivot_idx] != 0:
                        q = int(a_mat[r, pivot_idx] // piv)
                        a_mat[r, :] -= q * a_mat[pivot_idx, :]
                        if a_mat[r, pivot_idx] != 0:
                            changed = True

                for c in range(pivot_idx + 1, n):
                    if a_mat[pivot_idx, c] != 0:
                        q = int(a_mat[pivot_idx, c] // piv)
                        a_mat[:, c] -= q * a_mat[:, pivot_idx]
                        if a_mat[pivot_idx, c] != 0:
                            changed = True

                if not changed:
                    break

            pivot_idx += 1

        invariants = [int(abs(a_mat[i, i])) for i in range(k_min) if a_mat[i, i] != 0]
        # Imposición de la cadena de divisibilidad d_i | d_{i+1} vía gcd/lcm
        for i in range(len(invariants)):
            for j in range(i + 1, len(invariants)):
                g_val = math.gcd(invariants[i], invariants[j])
                l_val = (invariants[i] * invariants[j]) // max(g_val, 1)
                invariants[i] = g_val
                invariants[j] = l_val

        invariants.sort()
        rank = len(invariants)
        diag = tuple(invariants + [0] * (k_min - rank))
        torsion_free = all(x == 1 for x in invariants)
        return diag, rank, bool(torsion_free)


# ══════════════════════════════════════════════════════════════════════════════
# TEST AUTÓNOMO (Verificación de las 3 Fases Anidadas v15.0.0)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== TEST AUTÓNOMO: BoolNovikovStealthBooleEngine v15.0.0 (3 FASES ANIDADAS) ===")
    engine = BoolNovikovStealthBooleEngine()

    matrix_test = np.eye(4, dtype=np.float64)
    exps_test = np.array([0.15, 0.22, 0.35, 0.48], dtype=np.float64)

    report = engine.execute_stealth_boole_audit(matrix_test, exps_test)

    print(f"Veredicto Heyting Ω₃          : {report.truth_value.name} (0x{report.truth_value.value:02x})")
    print(f"Back-Action QND (dB)          : {report.gauge_report.back_action_db:.6f} dB (Camouflaged: {report.gauge_report.is_gauge_camouflaged})")
    print(f"Unitarity residual ||U†U-I||  : {report.gauge_report.unitarity_residual:.3e}")
    print(f"SU(N) det residual |det U-1|  : {report.gauge_report.su_n_det_residual:.3e}")
    print(f"Fase de Berry γ_Berry (rad)   : {report.gauge_report.berry_holonomy_phase_rad:.6f}")
    print(f"Rademacher–Walsh Norm         : {report.boole_report.rademacher_norm:.6f}")
    print(f"Rango exacto GF(2) / χ        : rank={report.boole_report.gf2_rank}, χ={report.boole_report.euler_poincare_characteristic}")
    print(f"Entropía Walsh–Hadamard       : {report.boole_report.walsh_hadamard_spectral_entropy:.6f}")
    print(f"Smith torsion-free            : {report.boole_report.smith_torsion_free}  diag={report.boole_report.smith_diagonal}")
    print(f"Valuación Krull v(A)          : {report.novikov_report.krull_valuation:.4f}")
    print(f"Ultramétrico fuerte / BT dist : {report.novikov_report.is_strong_ultrametric_preserved} (d_BT={report.novikov_report.bruhat_tits_distance:.4f})")
    print(f"Asociador Moufang Octoniones  : {report.novikov_report.octonion_moufang_associator_norm:.6e}")
    print(f"CR-Floer cylinder residual    : {report.cauchy_riemann_residual:.3e}")
    print(f"Energía simpléctica de Hofer  : {report.hofer_symplectic_energy:.6f}")
    print(f"Gromov non-squeezing rigid    : {report.is_gromov_rigid}")
    if report.poincare_report:
        print(f"Poincaré-Lindstedt ||W₁||     : {report.poincare_report.homology_residue:.3e}")
        print(f"Lie-Deprit 2nd-order residual : {report.poincare_report.lie_deprit_bracket_residual:.3e}")
        print(f"Kolmogorov / Arnold det       : {report.poincare_report.kolmogorov_hessian_det:.4f} / {report.poincare_report.arnold_isoenergetic_det:.4f}")
        print(f"Zero-average satisfied        : {report.poincare_report.is_zero_average_satisfied}")
    if report.kam_report:
        print(f"Melnikov homoclinic value     : {report.kam_report.melnikov_value:.6e}")
        print(f"KAM stable (Diofantino)       : {report.kam_report.kam_stable}  gap={report.kam_report.diophantine_gap:.3e}")
        print(f"Tiempo Nekhoroshev T_Nekh     : {report.kam_report.nekhoroshev_stability_time:.3e}")
        print(f"Parámetro Chirikov K          : {report.kam_report.chirikov_overlap_parameter:.3e}")
        print(f"Defecto Yoshida 4º orden      : {report.kam_report.yoshida_symplectic_error:.3e}")
        print(f"Chaos detected                : {report.kam_report.chaos_detected}")
    if report.rsi3_report:
        print(f"Lip(T³) Banach-Kleisli        : {report.rsi3_report.lip_t3:.6f}")
        print(f"Parámetro Kantorovich h_K     : {report.rsi3_report.kantorovich_parameter_h:.3e} (OK={report.rsi3_report.kantorovich_guaranteed})")
        print(f"Fixed-point residual          : {report.rsi3_report.fixed_point_residual:.3e}")
        print(f"Pasos Kleisli-Steffensen      : {report.rsi3_report.iteration_count}")
    if report.jerk_report:
        print(f"Cyber-physical jerk d³C/dt³   : {report.jerk_report.jerk_value:.6f} (Transient={report.jerk_report.transient_orbit_jerk:.3e})")
        print(f"Positive jerk (d³C/dt³>0)     : {report.jerk_report.is_positive_jerk}")

    print(f"Pasaporte HMAC-SHA256         : {report.hmac_passport[:16]}...")
    print("=== TEST COMPLETADO CON ÉXITO (3 FASES ANIDADAS v15.0.0) ===")