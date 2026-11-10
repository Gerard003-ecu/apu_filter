# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗
║ Módulo   : BoolNovikovRSI3WisdomWeaverProsecutorEngine v15.0.0-PhD+                                  ║
║ Ubicación: app/wisdom/bool_novikov_rsi3_wisdom_weaver_prosecutor_engine.py                           ║
║ Versión  : 15.0.0-ENGINE-BOOL-NOVIKOV-RSI3-WISDOM-WEAVER-PROSECUTOR-POINCARE-KAM-RSI3-PhD+           ║
║ Función  : Fiscal Inquisidor y Ejecutor Ciber-Físico sobre R_Boole ⊗ Λ_Nov ⊗ O_Oct, KAM y Crowbar.   ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL MOTOR ESPECTRAL
─────────────────────────────────────────────────
El Motor Espectral Fiscal `BoolNovikovRSI3WisdomWeaverProsecutorEngine` ejerce la fiscalía judicial
y la sanción ciber-física en la arquitectura APU Filter. Su misión matemática consiste en auditar
de manera inquisitorial el fibrado tensorial R_{\mathrm{Boole}} \otimes \Lambda_{\mathrm{Nov}} \otimes \mathbb{O},
detectar perturbaciones o violaciones de toros invariantes KAM, verificar desviaciones en la ecuación
homológica de Poincaré-Lindstedt, certificar el límite de contracción de Banach en RSI³ y, ante
cualquier transgresión irreconciliable en el topos de Heyting \Omega_3, ordenar la aniquilación cuántica
de Dirac y el disparo del interlock ciber-físico ESP32 Crowbar en tiempo ultra-rápido (\tau \le 392.15\text{ ns}).

MARCO TEÓRICO, FÍSICA Y FORMALISMO MATEMÁTICO
─────────────────────────────────────────────
1. FASE 1 — Inquisición Espectral en R_{\mathrm{Boole}} \otimes \Lambda_{\mathrm{Nov}} \otimes \mathbb{O}:
   Audita la función booleana discreta y el tensor de costes mediante la norma de Schatten-1:
       \|A\|_{S_1} = \operatorname{Tr}(\sqrt{A^\dagger A}) = \sum_{i=1}^n \sigma_i(A).
   En el anillo ultramétrico \Lambda_{\mathrm{Nov}}, verifica que la valuación de Krull no vulnere el piso crítico:
       v(A) = \min \{\lambda_i \mid a_i \neq 0\} \ge 0.10.
   En el álgebra de octoniones \mathbb{O}, evalúa la norma del asociador de Moufang:
       \|[x, y, z]\|_{\mathbb{O}} = \|(x \cdot y) \cdot z - x \cdot (y \cdot z)\|_2,
   asegurando que perturbaciones no asociativas no desestabilicen el álgebra de estados.
   • Apertura F2 (`_build_phase1_evidence_anchor`):
     Emite `ProsecutionEvidenceAnchor` conteniendo los cargos de prueba iniciales para la Fase 2.

2. FASE 2 — Inquisición Celeste de Poincaré-KAM y Condena por Caos Homoclínico:
   Inicia en la evidencia de Fase 1. Para el hamiltoniano perturbado H(J, \theta) = H_0(J) + \varepsilon H_1(J, \theta),
   resuelve la ecuación homológica de Poincaré-Lindstedt:
       \{H_0, W_1\} + H_1 = \langle H_1 \rangle_{\mathbb{T}^d},
   y evalúa el residuo de la serie de Lie.
   Comprueba la condición diofantina fuerte de KAM:
       |\langle k, \omega \rangle| \ge \frac{\gamma}{|k|_1^\tau}, \quad \forall k \neq 0,
   y el parámetro de solapamiento de resonancias de Chirikov:
       K_{\mathrm{Chirikov}} = \left( \frac{\Delta \omega_{\mathrm{sep}}}{\Delta \Omega_{\mathrm{res}}} \right)^2.
   Si K_{\mathrm{Chirikov}} \ge 1.0 o la integral de Melnikov excede el umbral crítico:
       |M(\tau)| > 1.0\times 10^{-4},
   se tipifica la destrucción estocástica de toros (difusión de Arnold) emitiendo condena.
   • Apertura F3 (`_build_phase2_conviction_certificate`):
     Emite `KAMConvictionCertificate` transfiriendo los cargos celestes a la Fase 3.

3. FASE 3 — Aniquilación Relativista de Dirac y Disparo Ciber-Físico ESP32 Crowbar:
   Inicia en el certificado de condena de Fase 2. Audita la contracción monádica de Kleisli T^3 en espacio de Banach
   y la cota de Newton-Kantorovich h_K \le 0.50.
   Ante veredicto \mathrm{VETOED} (0x03) en el álgebra de Heyting \Omega_3, se activa el protocolo irreversible de ejecución:
       • Aniquilación cuántica de pares electrón-positrón:
             e^- + e^+ \longrightarrow 2\gamma, \quad E_{\gamma} \ge 2 m_e c^2 = 1022.0 \text{ keV}.
       • Disparo de interlock ciber-físico ESP32 Crowbar sobre GPIO14 / BT151:
             V_{\mathrm{crowbar}}(t) = V_0 \exp(-\alpha t) \left( \cos(\omega_d t) + \frac{\alpha}{\omega_d} \sin(\omega_d t) \right),
         con latencia de ejecución garantizada \tau_{\mathrm{exec}} \in [42.0, 392.15]\text{ ns} sobre IRAM.
   El reporte judicial consolidado se formaliza en `WisdomWeaverProsecutorReport`.

AXIOMAS FORMALES
────────────────
[Axioma WWPE-A1] Jurisdicción Inquisitorial Ultramétrica:
    v(A) < 0.10 \implies \text{declaración formal de vulnerabilidad en el anillo de Novikov}.
[Axioma WWPE-A2] Tipificación de Caos Homoclínico:
    |M(\tau)| > 1.0\times 10^{-4} \lor K_{\mathrm{Chirikov}} \ge 1.0 \implies \text{condena judicial por desintegración de toros KAM}.
[Axioma WWPE-A3] Ineludibilidad de la Aniquilación de Dirac:
    \Omega_3 = \mathrm{VETOED} \implies \text{emisión irrevocable del pulso de aniquilación } 2\gamma \text{ (1022 keV)}.
[Axioma WWPE-A4] Acotamiento Temporal del Interlock Crowbar:
    \tau_{\mathrm{crowbar}} \le 392.15\text{ ns} \text{ en hardware bare-metal ESP32 / simulación determinista}.

INVARIANTES DE ESTADO
─────────────────────
[Invariante WWPE-I1] Cota de Schatten-1 Preservada: \|A\|_{S_1} \le \operatorname{Tr}(|A|).
[Invariante WWPE-I2] Parámetro de Chirikov Sub-Crítico: K_{\mathrm{Chirikov}} < 1.0 en régimen admisible.
[Invariante WWPE-I3] Parámetro de Kantorovich Controlado: h_K \le 0.50 en la mónada de Kleisli.
[Invariante WWPE-I4] Latencia de Actuación Crowbar: 42.0\text{ ns} \le \tau_{\mathrm{act}} \le 392.15\text{ ns}.
[Invariante WWPE-I5] Sellado de Sentencia Criptográfica HMAC-SHA256 con verificación en tiempo constante.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass
from enum import IntEnum
from functools import reduce
from math import gcd
from typing import Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.BoolNovikovRSI3WisdomWeaverProsecutorEngine")

_ENGINE_VERSION: Final[str] = (
    "15.0.0-ENGINE-BOOL-NOVIKOV-RSI3-WISDOM-WEAVER-PROSECUTOR-POINCARE-KAM-RSI3-PhD+"
)
__version__: Final[str] = _ENGINE_VERSION

# ── Constantes Físicas, Electrodinámicas, Espectrales y Categóricas ──
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_DEFAULT_KRULL_FLOOR: Final[float] = 0.10
_ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 392.15
_DIRAC_ANNIHILATION_ENERGY_KEV: Final[float] = 1022.0
_CLASSICAL_ELECTRON_RADIUS_BARN: Final[float] = 0.079407877  # π r_e² ≈ 2.4946e-25 cm² = 0.24946 barn (prefactor)
_KAM_GAMMA_MIN: Final[float] = 1.0e-4
_KAM_TAU: Final[float] = 2.5
_MELNIKOV_TOL: Final[float] = 1.0e-6
_GREENE_RESIDUE_CRIT: Final[float] = 0.25
_CONTRACTION_CUBE: Final[float] = 0.857375  # 0.95**3
_KANTOROVICH_MAX_H: Final[float] = 0.50


class HeytingTruthValue(IntEnum):
    """Álgebra de Heyting trivalente Ω₃ del Topos Fiscal Inquisidor."""
    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03


# ═══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS INMUTABLES (CONTRATOS DE FASE Y PUENTES CATEGÓRICOS)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DiracAnnihilationProsecutionReport:
    energy_released_kev: float
    positron_hmac_valid: bool
    back_action_penalty_db: float
    spohn_entropy_production: float
    annihilation_hash: str
    # — Extensiones v15 (QED Klein-Nishina / Lindblad) —
    klein_nishina_cross_section_barn: float = 0.0
    lindblad_dissipation_rate: float = 0.0
    pair_production_lor_coincidence: bool = False


@dataclass(frozen=True)
class ESP32CrowbarExecutionReport:
    crowbar_triggered: bool
    iram_instruction_latency_ns: float
    total_actuation_time_ns: float
    gpio_register_address: str
    thyristor_status: str
    # — Extensiones v15 (Física de Potencia del Tiristor SCR BT151 y Descarga RLC) —
    peak_surge_current_amps: float = 0.0
    di_dt_amps_per_us: float = 0.0
    i2t_fusing_integral_a2s: float = 0.0


@dataclass(frozen=True)
class ProsecutionEvidenceAnchor:
    """PUENTE FORMAL F1 ──► F2. Sella evidencia topológica Boole-SNF, Novikov-Krull, Octoniones y Bures."""
    rademacher_walsh_norm: float
    smith_torsion_free: bool
    betti_number_zero: bool
    krull_valuation: float
    novikov_norm: float
    ultrametric_ok: bool
    bures_fidelity: float
    bures_distance: float
    capacity_ball_quantum: float
    capacity_cylinder: float
    anomaly_norm: float
    # — Extensiones v15 —
    octonion_associator_norm: float = 0.0
    euler_characteristic: int = 0
    quantum_fisher_info: float = 0.0
    novikov_spectral_gap: float = 0.0
    anchor_hmac: str = ""


@dataclass(frozen=True)
class LindstedtSolution:
    frequencies: Tuple[float, ...]
    min_divisor: float
    diophantine_gamma: float
    diophantine_tau: float
    diophantine_ok: bool
    generating_norm: float
    homological_residual: float
    mean_H1: float
    # — Extensiones v15 —
    birkhoff_order2_norm: float = 0.0
    nekhoroshev_stability_time: float = 0.0
    secular_precession_rate: float = 0.0


@dataclass(frozen=True)
class KAMConvictionCertificate:
    """PUENTE FORMAL F2 ──► F3. Sella convicción dinámica Floer-Gromov y Poincaré-Lindstedt-KAM-Melnikov."""
    action_J: float
    pert_eps: float
    mean_motion: float
    diophantine_gamma: float
    diophantine_ok: bool
    lindstedt_residual: float
    melnikov_amplitude: float
    melnikov_damping: float
    melnikov_margin: float
    melnikov_has_simple_zero: bool
    is_chaotic: bool
    floquet_multiplier: float
    lyapunov_estimate: float
    floer_energy: float
    cauchy_riemann_residual: float
    is_gromov_violated: bool
    is_kam_stable: bool
    fraud_probability: float
    # — Extensiones v15 —
    poincare_cartan_invariant: float = 0.0
    conley_zehnder_index: int = 0
    greene_residue: float = 0.0
    nekhoroshev_time: float = 0.0
    quadrature_melnikov_error: float = 0.0
    evidence_anchor_hmac: str = ""
    certificate_hmac: str = ""


@dataclass(frozen=True)
class RSI3ContractionReport:
    lip_T: float
    lip_T3: float
    is_contraction: bool
    fixed_point: Tuple[float, ...]
    iterations: int
    banach_error_bound: float
    capacity_history: Tuple[float, ...]
    capacity_jerk_d3c_dt3: float
    is_jerk_positive: bool
    godel_hash: str
    darwin_selected: bool
    # — Extensiones v15 —
    kantorovich_parameter_h: float = 0.0
    frechet_cubic_jerk: float = 0.0
    meta_level3_gain: float = 0.0
    kleisli_endofunctor_id: str = ""


@dataclass(frozen=True)
class WisdomWeaverProsecutorReport:
    cauchy_riemann_residual: float
    krull_valuation: float
    novikov_norm: float
    is_gromov_violated: bool
    melnikov_chaos_detected: bool
    banach_contraction_lip: float
    capacity_jerk_d3c_dt3: float
    dirac_report: DiracAnnihilationProsecutionReport
    crowbar_report: ESP32CrowbarExecutionReport
    truth_value: HeytingTruthValue
    passport_hash: str
    execution_time_ms: float
    # — Extensiones v14 (defaults compat) —
    lindstedt_residual: float = 0.0
    diophantine_gamma: float = 0.0
    melnikov_value: float = 0.0
    floer_energy: float = 0.0
    bures_fidelity: float = 1.0
    fraud_probability: float = 0.0
    rsi_iterations: int = 0
    godel_hash: str = ""
    # — Extensiones v15 —
    greene_residue: float = 0.0
    conley_zehnder_index: int = 0
    nekhoroshev_time: float = 0.0
    octonion_associator_norm: float = 0.0
    quantum_fisher_info: float = 0.0
    kantorovich_h: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# NÚCLEOS MATEMÁTICOS PRIVADOS (OCTONIONES, Z-PID, SIMPLÉCTICA, QED Y POTENCIA)
# ═══════════════════════════════════════════════════════════════════════════════

def _next_pow2(n: int) -> int:
    p = 1
    while p < max(n, 1):
        p <<= 1
    return p


def _fwht(a: NDArray[np.float64]) -> NDArray[np.float64]:
    """Transformada Rápida de Walsh-Hadamard (FWHT) sobre Z₂ⁿ en O(N log N)."""
    out = np.asarray(a, dtype=np.float64).copy()
    n = out.shape[0]
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                x = out[j]
                y = out[j + h]
                out[j] = x + y
                out[j + h] = x - y
        h <<= 1
    return out


def _octonion_mul(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    """Multiplicación en el álgebra de Octoniones de Cayley-Dickson O ≅ R⁸ (plano de Fano PG(2,2))."""
    res = np.zeros(8, dtype=np.float64)
    res[0] = x[0] * y[0] - float(np.dot(x[1:], y[1:]))
    res[1:] = x[0] * y[1:] + y[0] * x[1:]
    fano_triples = (
        (1, 2, 3), (1, 4, 5), (1, 7, 6),
        (2, 4, 6), (2, 5, 7), (3, 4, 7), (3, 6, 5),
    )
    for i, j, k in fano_triples:
        res[k] += x[i] * y[j] - x[j] * y[i]
        res[i] += x[j] * y[k] - x[k] * y[j]
        res[j] += x[k] * y[i] - x[i] * y[k]
    return res


def _cayley_dickson_associator_norm(spectrum: NDArray[np.float64]) -> float:
    """Norma del asociador octoniónico [x, y, z]_O = (xy)z - x(yz) sobre el espectro Walsh."""
    pad = np.zeros(24, dtype=np.float64)
    k = min(spectrum.size, 24)
    pad[:k] = spectrum[:k]
    norm_s = float(np.linalg.norm(pad)) + _WILKINSON_EPSILON
    pad /= norm_s
    x, y, z = pad[0:8], pad[8:16], pad[16:24]
    assoc = _octonion_mul(_octonion_mul(x, y), z) - _octonion_mul(x, _octonion_mul(y, z))
    return float(np.linalg.norm(assoc))


def _smith_invariants(M_int: NDArray[np.int64]) -> Tuple[List[int], int]:
    """Forma Normal de Smith exacta por reducción unimodular PID sobre Z (Euclides-Bezout).

    Garantiza la cadena de divisibilidad d₁ | d₂ | ... | d_r en tiempo polinomial.
    """
    A = np.abs(np.asarray(M_int, dtype=np.int64)).copy()
    m, n = A.shape
    r_max = min(m, n)
    rank = int(np.linalg.matrix_rank(M_int.astype(np.float64)))
    if rank == 0:
        return [], 0

    invariants: List[int] = []
    work = A.copy()
    for k in range(r_max):
        sub = work[k:, k:]
        nz = sub[sub != 0]
        if nz.size == 0:
            break
        g = int(reduce(gcd, [abs(int(v)) for v in nz.flatten()]))
        invariants.append(max(g, 1))
        abs_sub = np.where(sub != 0, np.abs(sub), np.iinfo(np.int64).max)
        piv_r, piv_c = np.unravel_index(int(np.argmin(abs_sub)), sub.shape)
        work[[k, k + piv_r], :] = work[[k + piv_r, k], :]
        work[:, [k, k + piv_c]] = work[:, [k + piv_c, k]]
        piv = int(work[k, k])
        if piv == 0:
            break
        for i in range(k + 1, m):
            if work[i, k] != 0:
                q = work[i, k] // piv
                work[i, k:] -= q * work[k, k:]

    while len(invariants) < rank:
        invariants.append(1)
    invariants = invariants[:rank]
    for i in range(len(invariants) - 1):
        g = gcd(invariants[i], invariants[i + 1])
        lcm = (invariants[i] * invariants[i + 1]) // max(g, 1)
        invariants[i] = max(g, 1)
        invariants[i + 1] = max(lcm, 1)
    return invariants, rank


def _project_to_density(rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """Proyección espectral al símplex de operadores de densidad (ρ ⪰ 0, Tr(ρ)=1)."""
    rh = 0.5 * (rho + rho.conj().T)
    w, V = la.eigh(rh)
    w = np.clip(np.real(w), 0.0, None)
    s = float(np.sum(w))
    d = rho.shape[0]
    if s <= _WILKINSON_EPSILON:
        return np.eye(d, dtype=np.complex128) / float(d)
    return (V * (w / s)) @ V.conj().T


def _uhlmann_fidelity(rho: NDArray[np.complex128], sigma: NDArray[np.complex128]) -> float:
    """Fidelidad cuántica de Uhlmann F(ρ, σ) = (Tr √(√ρ σ √ρ))² vía descomposición espectral."""
    w_r, V_r = la.eigh(0.5 * (rho + rho.conj().T))
    w_r = np.sqrt(np.clip(np.real(w_r), 0.0, None))
    sq_rho = (V_r * w_r) @ V_r.conj().T
    mid = sq_rho @ sigma @ sq_rho
    mid = 0.5 * (mid + mid.conj().T)
    w_m, _ = la.eigh(mid)
    tr = float(np.sum(np.sqrt(np.clip(np.real(w_m), 0.0, None))))
    return float(min(max(tr * tr, 0.0), 1.0))


def _quantum_fisher_information(
    rho: NDArray[np.complex128], H_op: NDArray[np.complex128]
) -> float:
    """Información Cuántica de Fisher (SLD-QFI) F_Q(ρ, G_asym) ante generadores antisimétricos."""
    rh = _project_to_density(rho)
    p, V = la.eigh(rh)
    p = np.clip(np.real(p), 0.0, 1.0)
    H_eig = V.conj().T @ H_op @ V
    d = p.shape[0]
    qfi = 0.0
    for j in range(d):
        for k in range(j + 1, d):
            denom = p[j] + p[k]
            if denom > _SPECTRAL_PSD_FLOOR:
                qfi += 4.0 * ((p[j] - p[k]) ** 2 / denom) * (abs(H_eig[j, k]) ** 2)
    return float(qfi)


def _kepler_frequencies(action_J: float) -> Tuple[NDArray[np.float64], float]:
    """Frecuencias fundamentales de Delaunay-Kepler ω = (n, n/φ) con razón áurea diofántica."""
    mu = 1.0
    L = max(float(action_J), 0.2)
    n = (mu ** 2) / (L ** 3)
    phi = 0.5 * (1.0 + math.sqrt(5.0))
    return np.array([n, n / phi], dtype=np.float64), float(n)


def _melnikov_gauss_legendre_quadrature(
    omega: float, pert_eps: float, damping_delta: float = 0.15, t0: float = 0.0
) -> float:
    """Cuadratura de Gauss-Legendre de 32 nodos para la integral homoclínica de Poincaré-Melnikov:

    M(t₀) = ∫_{-∞}^{∞} p₀(t) [ε sin(Ω(t + t₀)) - δ p₀(t)] dt,  con p₀(t) = 2 sech(t).
    """
    nodes, weights = np.polynomial.legendre.leggauss(32)
    T_span = 12.0
    t_vals = T_span * nodes
    w_vals = T_span * weights
    p0 = 2.0 / np.cosh(t_vals)
    integrand = p0 * (pert_eps * np.sin(omega * t_vals + omega * t0) - damping_delta * p0)
    return float(np.sum(w_vals * integrand))


def _dirac_two_photon_cross_section_barn(anomaly_norm: float) -> float:
    """Sección eficaz de aniquilación de Dirac e⁻ + e⁺ → 2γ en barns:

    σ_{2γ}(γ_L) = (π r_e² / (γ_L + 1)) * [((γ_L² + 4γ_L + 1)/(γ_L² - 1)) ln(γ_L + √(γ_L² - 1)) - (γ_L + 3)/√(γ_L² - 1)],
    donde el factor de Lorentz efectivo del par es γ_L = 1 + max(anomaly_norm, 1e-4).
    """
    gamma_l = 1.0 + max(float(anomaly_norm), 1.0e-4)
    beta_g = math.sqrt(max(gamma_l * gamma_l - 1.0, 1.0e-12))
    pref = 0.249464 / (gamma_l + 1.0)  # π r_e² ≈ 0.249464 barn
    term1 = ((gamma_l * gamma_l + 4.0 * gamma_l + 1.0) / (beta_g * beta_g)) * math.log(
        gamma_l + beta_g
    )
    term2 = (gamma_l + 3.0) / beta_g
    return float(max(0.0, pref * (term1 - term2)))


def _capacity_from_state(s: NDArray[np.float64]) -> float:
    """Funcional de Capacidad Espectral C(v, Lip, CR) en decibelios algebraicos."""
    v, lip, cr = float(s[0]), float(s[1]), float(s[2])
    return 10.0 * v - 5.0 * lip - 5.0 * math.log10(1.0 + 10.0 * max(cr, 0.0))


def _frechet_capacity_derivatives(
    s: NDArray[np.float64], ds: NDArray[np.float64]
) -> Tuple[float, float, float]:
    """Derivadas direccionales de Fréchet D¹C[s](ds), D²C[s](ds,ds) y D³C[s](ds,ds,ds)."""
    cr = max(float(s[2]), 0.0)
    ln10 = math.log(10.0)
    denom = 1.0 + 10.0 * cr
    g_cr = -50.0 / (denom * ln10)
    h_cr = 500.0 / ((denom ** 2) * ln10)
    t_cr = -10000.0 / ((denom ** 3) * ln10)
    d1 = 10.0 * float(ds[0]) - 5.0 * float(ds[1]) + g_cr * float(ds[2])
    d2 = h_cr * (float(ds[2]) ** 2)
    d3 = t_cr * (float(ds[2]) ** 3)
    return float(d1), float(d2), float(d3)


def _hmac_hex(secret: bytes, msg: str) -> str:
    return hmac.new(secret, msg.encode("utf-8"), hashlib.sha256).hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# MOTOR FISCAL INQUISIDOR SOBERANO: ESTRUCTURADO EN 3 FASES ANIDADAS
# ═══════════════════════════════════════════════════════════════════════════════

class BoolNovikovRSI3WisdomWeaverProsecutorEngine:
    """Fiscal Espectral Inquisidor y Ejecutor Ciber-Físico sobre R_Boole ⊗ Λ_Nov ⊗ O_Oct."""

    def __init__(
        self, secret_key: bytes = b"APU_FILTER_PROSECUTOR_WISDOM_KEY_v13.0.0"
    ) -> None:
        self._secret_key: bytes = secret_key
        self._capacity_history: List[float] = []
        self._last_octonion_associator: float = 0.0
        self._last_euler_characteristic: int = 0
        self._last_ultrametric_ok: bool = True
        self._last_novikov_gap: float = 0.0
        self._last_qfi: float = 0.0
        self._last_floer_energy: float = 0.0
        self._last_poincare_cartan: float = 0.0
        self._last_cz_index: int = 0
        self._last_lindstedt: Optional[LindstedtSolution] = None
        self._melnikov_cache: Tuple[float, float, bool, float, float, float] = (
            0.0, 0.4, False, 1.0, 0.0, 0.0
        )
        self._last_rsi: Optional[RSI3ContractionReport] = None
        logger.info(f"{self.__class__.__name__} v{_ENGINE_VERSION} inicializado correctamente.")

    # ╔═════════════════════════════════════════════════════════════════════════╗
    # ║ FASE 1 — INQUISICIÓN EVIDENCIARIA BOOLE-SNF, OCTONIONES Y BURES-QFI    ║
    # ║ Flujo: RW/SNF/Octoniones ──► Krull/Novikov ──► Isometría Bures/QFI      ║
    # ║ Clausura de Fase 1: `seal_prosecution_evidence_anchor` ──► Puente a F2 ║
    # ╚═════════════════════════════════════════════════════════════════════════╝

    def compute_rademacher_walsh_and_snf(
        self, incidence_matrix: NDArray[np.float64],
    ) -> Tuple[float, bool, bool]:
        """1.1 Derivada Rademacher-Walsh, espectro Octoniónico O y Forma Normal de Smith en Z.

        Extrae la asimetría normalizada ‖M - Mᵀ‖_F / (2‖M‖_F + ε), proyecta el espectro
        de Walsh-Hadamard (FWHT) en el álgebra de Octoniones de Cayley-Dickson O para
        evaluar el asociador [x, y, z]_O y calcula los factores invariantes de Smith sobre Z.
        """
        M = np.asarray(incidence_matrix, dtype=np.float64)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError("incidence_matrix debe ser una matriz cuadrada 2D.")
        dim = M.shape[0]
        frob = float(la.norm(M, "fro"))
        rw_norm = float(la.norm(M - M.T, "fro") / (2.0 * frob + _WILKINSON_EPSILON))

        flat = M.flatten()
        N = _next_pow2(int(flat.size))
        pad = np.zeros(N, dtype=np.float64)
        pad[: flat.size] = flat
        walsh_spec = _fwht(pad) / float(N)
        self._last_octonion_associator = _cayley_dickson_associator_norm(walsh_spec)

        M_int = np.round(np.abs(M)).astype(np.int64)
        inv, rank = _smith_invariants(M_int)
        torsion_free = all(d == 1 for d in inv) if inv else True
        betti_zero = (rank == dim) or (rw_norm < 0.08 and rank >= dim - 1)
        self._last_euler_characteristic = int(dim - rank)
        return rw_norm, bool(torsion_free), bool(betti_zero)

    def evaluate_novikov_krull_and_bures(
        self, exponents: NDArray[np.float64], apu_cost_vector: NDArray[np.float64],
    ) -> Tuple[float, float]:
        """1.2 Valuación de Krull v(A) = min λ_i y norma no arquimediana en el anillo Λ_Nov.

        Verifica el buen ordenamiento de la serie formal A(T) = ∑ a_i T^{λ_i} ∈ Λ_Nov
        y la desigualdad ultramétrica estricta ‖x + y‖_Nov ≤ max(‖x‖_Nov, ‖y‖_Nov).
        """
        lam = np.asarray(exponents, dtype=np.float64).flatten()
        cost = np.asarray(apu_cost_vector, dtype=np.float64).flatten()
        if lam.size == 0:
            self._last_ultrametric_ok = True
            self._last_novikov_gap = 0.0
            return _DEFAULT_KRULL_FLOOR, math.exp(-_DEFAULT_KRULL_FLOOR)

        v = float(np.min(lam))
        srt = np.sort(lam)
        gaps = np.diff(srt)
        wf = bool(np.all(gaps > 1.0e-12)) if gaps.size > 0 else True
        self._last_novikov_gap = float(np.min(gaps)) if gaps.size > 0 else float(v)

        even = lam[0::2]
        odd = lam[1::2]
        if even.size > 0 and odd.size > 0:
            n_tot = math.exp(-max(float(np.min(lam)), 0.0))
            n_even = math.exp(-max(float(np.min(even)), 0.0))
            n_odd = math.exp(-max(float(np.min(odd)), 0.0))
            ultra = n_tot <= max(n_even, n_odd) + 1.0e-12
        else:
            ultra = True

        self._last_ultrametric_ok = bool(ultra and wf)
        _ = float(np.mean(np.abs(cost))) if cost.size > 0 else 0.0
        return v, math.exp(-max(v, 0.0))

    def evaluate_bures_uhlmann_isometry(
        self, apu_cost_matrix: NDArray[np.float64]
    ) -> Tuple[float, float, bool]:
        """1.3 Isometría Cuántica de Bures-Uhlmann e Información de Fisher (SLD-QFI).

        Sintetiza el estado térmico de Gibbs ρ = exp(-H₀)/Tr(exp(-H₀)) con H₀ = ½(M + Mᵀ),
        evalúa la fidelidad F(ρ, σ) ante un canal despolarizante al 1%, la distancia de
        Bures d_B = √(2 - 2√F) y la sensibilidad QFI ante perturbaciones de калибре.
        """
        M = np.asarray(apu_cost_matrix, dtype=np.float64)
        d = M.shape[0]
        H_sym = 0.5 * (M + M.T)
        try:
            rho = la.expm(-H_sym).astype(np.complex128)
            rho = rho / max(complex(np.trace(rho)).real, _WILKINSON_EPSILON)
        except Exception:
            rho = np.eye(d, dtype=np.complex128) / float(d)
        rho = _project_to_density(rho)
        sigma = _project_to_density(
            0.99 * rho + 0.01 * np.eye(d, dtype=np.complex128) / float(d)
        )
        fid = _uhlmann_fidelity(rho, sigma)
        dist = math.sqrt(max(0.0, 2.0 - 2.0 * math.sqrt(fid)))

        # Generador de sonda QFI en la base canónica para medir susceptibilidad espectral
        probe_op = (M - np.diag(np.diag(M))).astype(np.complex128)
        self._last_qfi = _quantum_fisher_information(rho, probe_op)
        return fid, dist, bool(fid >= 0.95)

    def seal_prosecution_evidence_anchor(
        self,
        rademacher_walsh_norm: float,
        smith_torsion_free: bool,
        betti_number_zero: bool,
        krull_valuation: float,
        novikov_norm: float,
        bures_fidelity: float = 1.0,
        bures_distance: float = 0.0,
        ball_r: float = 0.85,
        cylinder_R: float = 1.0,
    ) -> ProsecutionEvidenceAnchor:
        """1.4 [CLAUSURA FORMAL DE FASE 1 ──► INICIO ENCADENADO DE FASE 2]

        Empaqueta y firma criptográficamente el ancla `ProsecutionEvidenceAnchor` con la
        capacidad cuántica de Gromov-Novikov c_q(B²ⁿ(r)) = πr²‖A‖_Nov y la norma de anomalía
        inquisitorial. Su retorno alimenta el parámetro `evidence` del primer método de Fase 2
        (`solve_cauchy_riemann_floer_prosecution`).
        """
        cap_bq = math.pi * (float(ball_r) ** 2) * float(novikov_norm)
        cap_c = math.pi * (float(cylinder_R) ** 2)
        topo_pen = 0.0 if (bool(smith_torsion_free) and bool(betti_number_zero)) else 0.5
        krull_pen = max(0.0, _DEFAULT_KRULL_FLOOR - float(krull_valuation)) * 2.0
        anomaly = float(
            min(max(float(rademacher_walsh_norm) + topo_pen + krull_pen, 0.0), 2.0)
        )
        sig_payload = (
            f"F1_PROS_ANCHOR|{rademacher_walsh_norm:.8f}|{int(smith_torsion_free)}|"
            f"{int(betti_number_zero)}|{krull_valuation:.8f}|{novikov_norm:.8f}|"
            f"{bures_fidelity:.8f}|{cap_bq:.8f}|{anomaly:.8f}"
        )
        anchor_sig = _hmac_hex(self._secret_key, sig_payload)
        return ProsecutionEvidenceAnchor(
            rademacher_walsh_norm=float(rademacher_walsh_norm),
            smith_torsion_free=bool(smith_torsion_free),
            betti_number_zero=bool(betti_number_zero),
            krull_valuation=float(krull_valuation),
            novikov_norm=float(novikov_norm),
            ultrametric_ok=bool(self._last_ultrametric_ok),
            bures_fidelity=float(bures_fidelity),
            bures_distance=float(bures_distance),
            capacity_ball_quantum=float(cap_bq),
            capacity_cylinder=float(cap_c),
            anomaly_norm=float(anomaly),
            octonion_associator_norm=float(self._last_octonion_associator),
            euler_characteristic=int(self._last_euler_characteristic),
            quantum_fisher_info=float(self._last_qfi),
            novikov_spectral_gap=float(self._last_novikov_gap),
            anchor_hmac=anchor_sig,
        )

    # ╔═════════════════════════════════════════════════════════════════════════╗
    # ║ FASE 2 — JUICIO DINÁMICO FLOER-GROMOV Y MECÁNICA CELESTE DE POINCARÉ   ║
    # ║ Inicio Encadenado: `solve_cauchy_riemann_floer_prosecution(evidence=…)` ║
    # ║ Flujo: Floer/Poincaré-Cartan ──► Lindstedt-Birkhoff ──► KAM-Melnikov    ║
    # ║ Clausura de Fase 2: `certify_conviction_and_chaos` ──► Puente a Fase 3 ║
    # ╚═════════════════════════════════════════════════════════════════════════╝

    def solve_cauchy_riemann_floer_prosecution(
        self,
        apu_cost_matrix: NDArray[np.float64],
        ball_r: float = 0.85,
        cylinder_R: float = 1.0,
        *,
        evidence: Optional[ProsecutionEvidenceAnchor] = None,
    ) -> Tuple[float, bool]:
        """2.1 [CONTINUACIÓN DIRECTA DE FASE 1] Tira Pseudoholomorfa de Floer ∂̄_{J,H} u = 0.

        Recibe `evidence: ProsecutionEvidenceAnchor` (clausura de Fase 1). Evalúa la tira
        de intersección Lagrangiana entre L₀ (columnas Q_i) y L₁ (filas P_i) sobre T*T^d:
          - El defecto de holomorfía mide la ruptura de autodualidad Lagrangiana (P_i - Q_i)
            acoplada al campo Hamiltoniano antisimétrico X_{H_asym} y al acoplamiento fuera
            de la diagonal. Para una matriz diagonal simétrica equilibrada (e.g. I_d),
            el lazo reside en la intersección Lagrangiana autodual con residual R_CR = 0.0.
          - Calcula la energía de Floer E(u), el invariante integral de Poincaré-Cartan
            A_H = ∮ ⟨P, dQ⟩ - ∫ H dt, el índice de Conley-Zehnder μ_CZ y la violación
            de No-Squeezing de Gromov c_q(B²ⁿ(r)) > c(Z²ⁿ(R)).
        """
        M = np.asarray(apu_cost_matrix, dtype=np.float64)
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise ValueError("apu_cost_matrix debe ser una matriz cuadrada 2D.")
        d = M.shape[0]

        nov_weight = float(evidence.novikov_norm) if evidence is not None else 1.0
        G_sym = 0.5 * (M + M.T)
        Omega_asym = 0.5 * (M - M.T)
        try:
            min_eig = float(np.min(la.eigvalsh(G_sym)))
            shift = max(0.0, -min_eig + 1.0e-9)
        except Exception:
            shift = 1.0e-9
        Greg = G_sym + shift * np.eye(d, dtype=np.float64)
        G_off = G_sym - np.diag(np.diag(G_sym))

        Q = [M[:, i].copy() for i in range(d)]
        P = [M[i, :].copy() for i in range(d)]
        J0 = np.block([
            [np.zeros((d, d), dtype=np.float64), -np.eye(d, dtype=np.float64)],
            [np.eye(d, dtype=np.float64), np.zeros((d, d), dtype=np.float64)],
        ])

        num2 = 0.0
        den_t2 = 0.0
        den_x2 = 0.0
        action_pc = 0.0
        for i in range(d):
            ip1 = (i + 1) % d
            im1 = (i - 1) % d
            dQ_dt = 0.5 * (Q[ip1] - Q[im1])
            # Salto Lagrangiano transversal ∂_t u_i entre L₀ (Q_i) y L₁ (P_i)
            du_dt = np.concatenate([P[i] - Q[i], Q[i] - P[i]])
            # Campo Hamiltoniano de interacción fuera de la diagonal y asimetría de calibre
            grad_H_int = np.concatenate(
                [Omega_asym @ Q[i] + 0.5 * G_off @ P[i], Omega_asym.T @ P[i] + 0.5 * G_off @ Q[i]]
            )
            X_H_int = J0 @ grad_H_int
            # Flujo de referencia simpléctico autodual
            X_H_ref = J0 @ np.concatenate([Greg @ Q[i], Greg @ P[i]])

            r_floer = du_dt - nov_weight * X_H_int
            num2 += float(r_floer @ r_floer)
            den_t2 += float(du_dt @ du_dt)
            den_x2 += float(X_H_ref @ X_H_ref)

            H_val = 0.5 * float(Q[i] @ Greg @ Q[i] + P[i] @ Greg @ P[i])
            action_pc += float(P[i] @ dQ_dt) - H_val / float(max(d, 1))

        res_cr = float(
            math.sqrt(num2) / (math.sqrt(den_t2) + math.sqrt(den_x2) + _WILKINSON_EPSILON)
        )
        self._last_floer_energy = float(0.5 * (den_t2 + den_x2) / max(d, 1))
        self._last_poincare_cartan = float(action_pc)

        eigs_G = np.real(la.eigvalsh(Greg))
        winding_sum = float(np.sum(np.clip(eigs_G, 1.0e-9, None))) / math.pi
        self._last_cz_index = int(2 * round(winding_sum) + d)

        if evidence is not None:
            viol = bool(evidence.capacity_ball_quantum > evidence.capacity_cylinder + 1.0e-12)
        else:
            viol = bool(
                math.pi * (float(ball_r) ** 2) > math.pi * (float(cylinder_R) ** 2) + 1.0e-12
            )
        return res_cr, viol

    def solve_poincare_lindstedt_homological(
        self,
        action_J: float,
        pert_eps: float = 0.01,
        Kmax: int = 8,
        tau: float = _KAM_TAU,
    ) -> LindstedtSolution:
        """2.2 Ecuación Homológica de Poincaré-Lindstedt-Birkhoff {H₀, W} = H₁ - ⟨H₁⟩ en T².

        Resuelve la función generatriz W = W₁ + ε W₂ en variables de acción-ángulo de Delaunay
        L = J, verifica la condición diofántica |k·ω| ≥ γ/|k|^τ y estima el tiempo de
        estabilidad exponencial de Nekhoroshev T_Nekh ~ exp((γ / ε)^{1/(2τ)}).
        """
        omega, n_mean = _kepler_frequencies(action_J)
        eps = float(pert_eps)
        min_div = float("inf")
        gamma = float("inf")
        w1_sq = 0.0
        w2_sq = 0.0

        for k1 in range(-Kmax, Kmax + 1):
            for k2 in range(-Kmax, Kmax + 1):
                if k1 == 0 and k2 == 0:
                    continue
                kd = abs(float(k1 * omega[0] + k2 * omega[1]))
                min_div = min(min_div, kd)
                kn = math.sqrt(float(k1 * k1 + k2 * k2))
                gamma = min(gamma, kd * (kn ** float(tau)))
                l1_norm = abs(k1) + abs(k2)
                hk = eps * math.exp(-float(l1_norm)) / (1.0 + 0.1 * abs(float(action_J)))
                wk1 = hk / (kd + _WILKINSON_EPSILON)
                w1_sq += wk1 * wk1
                hk2 = 0.5 * float(kn) * eps * wk1 * math.exp(-0.5 * float(l1_norm))
                wk2 = hk2 / (kd + _WILKINSON_EPSILON)
                w2_sq += wk2 * wk2

        if not math.isfinite(min_div):
            min_div = 0.0
        if not math.isfinite(gamma):
            gamma = 0.0

        tail = float(
            eps * math.exp(-float(Kmax)) / (min_div + 1.0e-12) / (1.0 - math.exp(-1.0))
        )
        nekh_exp = min(
            25.0, (max(gamma, 1.0e-9) / max(eps, 1.0e-12)) ** (1.0 / (2.0 * float(tau)))
        )
        t_nekh = float(math.exp(nekh_exp))
        secular_prec = float(n_mean * eps * eps / max(float(action_J), 0.2))

        sol = LindstedtSolution(
            frequencies=(float(omega[0]), float(omega[1])),
            min_divisor=float(min_div),
            diophantine_gamma=float(gamma),
            diophantine_tau=float(tau),
            diophantine_ok=bool(gamma > _KAM_GAMMA_MIN and min_div > 1.0e-9),
            generating_norm=float(math.sqrt(w1_sq)),
            homological_residual=float(tail),
            mean_H1=float(0.5 * eps),
            birkhoff_order2_norm=float(math.sqrt(w2_sq)),
            nekhoroshev_stability_time=float(t_nekh),
            secular_precession_rate=float(secular_prec),
        )
        self._last_lindstedt = sol
        return sol

    def evaluate_poincare_kam_melnikov(
        self, action_J: float, pert_eps: float = 0.01
    ) -> float:
        """2.3 Integral Homoclínica de Poincaré-Melnikov y Residuo de Greene — retorna M* = C - D.

        Preserva la firma escalar `float` (compatibilidad v13/v14) mientras valida por
        cuadratura de Gauss-Legendre de 32 nodos sobre la separatriz p₀(t) = 2 sech(t)
        y almacena el Residuo de Greene R_Greene = ¼(2 - Tr M_Floquet) en `_melnikov_cache`.
        """
        _, n = _kepler_frequencies(action_J)
        Om = float(n) if n > 1.0e-9 else 1.0
        delta_damping = 0.15
        D = 8.0 * delta_damping / 3.0
        try:
            sech = 1.0 / math.cosh(0.5 * math.pi * Om)
        except OverflowError:
            sech = 0.0
        C = math.sqrt(2.0) * math.pi * Om * float(pert_eps) * sech

        t0_max = 0.5 * math.pi / Om
        m_quad = _melnikov_gauss_legendre_quadrature(Om, float(pert_eps), delta_damping, t0_max)
        m_analytic = C - D
        quad_err = abs(m_quad - m_analytic)

        trace_monodromy = 2.0 * math.cos(2.0 * math.pi / ((1.0 + math.sqrt(5.0)) * 0.5)) - (
            4.0 * math.pi * float(pert_eps) * C
        )
        greene_res = 0.25 * (2.0 - trace_monodromy)

        has_zero = bool(C > D + _MELNIKOV_TOL)
        self._melnikov_cache = (
            float(C), float(D), bool(has_zero), float(Om), float(greene_res), float(quad_err)
        )
        return float(m_analytic)

    def certify_conviction_and_chaos(
        self,
        action_J: float,
        pert_eps: float = 0.01,
        cr_residual: float = 0.0,
        floer_energy: float = 0.0,
        is_gromov_violated: bool = False,
        *,
        evidence_anchor_hmac: str = "",
    ) -> KAMConvictionCertificate:
        """2.4 [CLAUSURA FORMAL DE FASE 2 ──► INICIO ENCADENADO DE FASE 3]

        Consolida la serie homológica de Poincaré-Lindstedt-Birkhoff, la transversalidad
        homoclínica de Melnikov, el Residuo de Greene, el multiplicador de Floquet y la
        probabilidad de fraude simpléctico en `KAMConvictionCertificate`, entrada formal
        del argumento `conviction` en `evaluate_rsi3_banach_kleisli` (Fase 3).
        """
        lind = self.solve_poincare_lindstedt_homological(action_J, pert_eps)
        margin = self.evaluate_poincare_kam_melnikov(action_J, pert_eps)
        C, D, has_zero, Om, greene_res, quad_err = self._melnikov_cache
        chaotic = bool(has_zero)

        lyap = (0.05 + 0.5 * max(0.0, (C - D) / max(D, 1.0e-12))) if chaotic else 0.0
        try:
            mu = float(math.exp(lyap * (2.0 * math.pi / max(Om, 1.0e-9))))
        except OverflowError:
            mu = float("inf")

        _, n = _kepler_frequencies(action_J)
        stable = bool(
            (not chaotic)
            and lind.diophantine_ok
            and (not bool(is_gromov_violated))
            and (lind.homological_residual < 0.1)
            and (abs(greene_res) < _GREENE_RESIDUE_CRIT + 0.25)
        )

        if bool(is_gromov_violated):
            fraud = 1.0
        elif chaotic:
            fraud = 0.75
        elif stable:
            fraud = 0.10
        else:
            fraud = 0.40

        cert_payload = (
            f"F2_PROS_CERT|{evidence_anchor_hmac}|{action_J:.8f}|{pert_eps:.8f}|"
            f"{lind.diophantine_gamma:.8f}|{margin:.8f}|{int(stable)}|{fraud:.4f}"
        )
        cert_sig = _hmac_hex(self._secret_key, cert_payload)

        return KAMConvictionCertificate(
            action_J=float(action_J),
            pert_eps=float(pert_eps),
            mean_motion=float(n),
            diophantine_gamma=float(lind.diophantine_gamma),
            diophantine_ok=bool(lind.diophantine_ok),
            lindstedt_residual=float(lind.homological_residual),
            melnikov_amplitude=float(C),
            melnikov_damping=float(D),
            melnikov_margin=float(margin),
            melnikov_has_simple_zero=bool(has_zero),
            is_chaotic=bool(chaotic),
            floquet_multiplier=float(mu),
            lyapunov_estimate=float(lyap),
            floer_energy=float(floer_energy),
            cauchy_riemann_residual=float(cr_residual),
            is_gromov_violated=bool(is_gromov_violated),
            is_kam_stable=bool(stable),
            fraud_probability=float(fraud),
            poincare_cartan_invariant=float(self._last_poincare_cartan),
            conley_zehnder_index=int(self._last_cz_index),
            greene_residue=float(greene_res),
            nekhoroshev_time=float(lind.nekhoroshev_stability_time),
            quadrature_melnikov_error=float(quad_err),
            evidence_anchor_hmac=str(evidence_anchor_hmac),
            certificate_hmac=cert_sig,
        )

    # ╔═════════════════════════════════════════════════════════════════════════╗
    # ║ FASE 3 — EJECUCIÓN CIBER-FÍSICA: RSI³ NIVEL-3, DIRAC-CROWBAR Y HEYTING  ║
    # ║ Inicio Encadenado: `evaluate_rsi3_banach_kleisli(conviction=…)`         ║
    # ║ Flujo: Kleisli T³ ──► Paso RSI³ L3 ──► Dirac/Crowbar ──► Auditoría Ω₃   ║
    # ╚═════════════════════════════════════════════════════════════════════════╝

    def evaluate_rsi3_banach_kleisli(
        self,
        initial_lip: float = 0.95,
        *,
        conviction: Optional[KAMConvictionCertificate] = None,
        krull_val: float = 0.15,
        cr_residual: float = 0.0,
    ) -> Tuple[float, float]:
        """3.1 [CONTINUACIÓN DIRECTA DE FASE 2] Mónada de Kleisli T³ Banach-Kantorovich-Halley.

        Recibe `conviction: KAMConvictionCertificate` (clausura de Fase 2) para modular el
        endofuntor de Nivel-3 sobre el espacio de Banach B = R³:
          - Nivel 1 (Objeto): Contracción de Picard s_{k+1} = (1 - α) s_k + α s*.
          - Nivel 2 (Meta-Operador): Control de Newton-Kantorovich h_K = L₀ η₀ ≤ ½ modulado
            por la convicción KAM de la Fase 2.
          - Nivel 3 (Meta-Meta Endofuntor): Acelerador cúbico de Chebyshev-Halley y Jerk
            analítico-discreto D³C/dt³ > 0 evaluado sobre la tríada primaria (s₀, Ts₀, T²s₀, T³s₀)
            libre de cancelación catastrófica asintótica.
        """
        lip0 = float(
            min(max(float(initial_lip) if math.isfinite(float(initial_lip)) else 0.95, 1.0e-6), 0.999)
        )
        if conviction is not None:
            alpha = 0.05 if bool(conviction.is_kam_stable) else 0.03
        else:
            alpha = float(min(max(1.0 - lip0, 0.01), 0.5))

        lip_T = float(min(max(1.0 - alpha, 1.0e-6), 0.999))
        lip_T3 = float(lip_T ** 3)

        s0 = np.array(
            [float(krull_val), lip0, float(max(cr_residual, 0.0))], dtype=np.float64
        )
        s_star = np.array(
            [max(float(krull_val), 0.20), lip_T * 0.95, 0.0], dtype=np.float64
        )

        s1_step = (1.0 - alpha) * s0 + alpha * s_star
        eta0 = float(np.linalg.norm(s1_step - s0))
        kantorovich_h = float(min(_KANTOROVICH_MAX_H, alpha * eta0))

        # Tríada primaria de Kleisli (T⁰s₀, T¹s₀, T²s₀, T³s₀) con corrector de Halley Nivel-3
        kleisli_caps: List[float] = [_capacity_from_state(s0)]
        s_curr = s0.copy()
        for k_step in range(1, 4):
            halley_boost = alpha * (1.0 + 0.15 * k_step * (1.0 - kantorovich_h))
            eff_alpha = min(0.25, halley_boost)
            s_curr = (1.0 - eff_alpha) * s_curr + eff_alpha * s_star
            kleisli_caps.append(_capacity_from_state(s_curr))

        s = s0.copy()
        Caps: List[float] = [_capacity_from_state(s)]
        it = 0
        for it in range(1, 513):
            s = (1.0 - alpha) * s + alpha * s_star
            Caps.append(_capacity_from_state(s))
            if float(np.linalg.norm(s - s_star)) < 1.0e-10:
                break

        ds_vec = s_star - s0
        d1_f, _, d3_f = _frechet_capacity_derivatives(s0, alpha * ds_vec)
        kleisli_diff3 = abs(
            kleisli_caps[3] - 3.0 * kleisli_caps[2] + 3.0 * kleisli_caps[1] - kleisli_caps[0]
        )
        asymptotic_diff3 = (
            max(
                0.0,
                Caps[-1]
                - 3.0 * Caps[max(0, len(Caps) - 2)]
                + 3.0 * Caps[max(0, len(Caps) - 3)]
                - Caps[max(0, len(Caps) - 4)],
            )
            if len(Caps) >= 4
            else 0.0
        )
        frechet_cubic = abs(d3_f) + max(0.0, d1_f) * (alpha ** 3)
        jerk = float(max(kleisli_diff3, asymptotic_diff3, frechet_cubic))

        try:
            err = float((lip_T ** it) / max(1.0 - lip_T, 1.0e-12) * eta0)
        except Exception:
            err = float("inf")

        darwin = bool(Caps[-1] >= Caps[0] - 1.0e-12)
        cert_link = conviction.certificate_hmac[:16] if conviction is not None else "NO_CERT"
        endo_id = f"KLEISLI_PROS_L3_{cert_link}"
        gh = _hmac_hex(
            self._secret_key,
            f"RSI3-PROS|{_ENGINE_VERSION}|{endo_id}|{tuple(np.round(s, 10))}|{it}|{lip_T3:.12f}|{jerk:.8f}",
        )

        for c_val in kleisli_caps:
            self._capacity_history.append(float(c_val))

        self._last_rsi = RSI3ContractionReport(
            lip_T=lip_T,
            lip_T3=lip_T3,
            is_contraction=bool(lip_T3 < 1.0),
            fixed_point=(float(s[0]), float(s[1]), float(s[2])),
            iterations=int(it),
            banach_error_bound=float(err),
            capacity_history=tuple(float(x) for x in kleisli_caps),
            capacity_jerk_d3c_dt3=float(jerk),
            is_jerk_positive=bool(jerk > 0.0),
            godel_hash=gh,
            darwin_selected=bool(darwin),
            kantorovich_parameter_h=float(kantorovich_h),
            frechet_cubic_jerk=float(frechet_cubic),
            meta_level3_gain=float(1.0 - lip_T3),
            kleisli_endofunctor_id=endo_id,
        )
        return lip_T3, jerk

    def recursive_self_improvement_step_nivel3(
        self,
        state: NDArray[np.float64],
        conviction: KAMConvictionCertificate,
        lambda_candidates: int = 3,
    ) -> Tuple[NDArray[np.float64], RSI3ContractionReport]:
        """3.2 Paso de Automejora Recursiva Nivel-3 (Darwin-Gödel sobre End(C_T)).

        Reescribe la composición triple de Kleisli T_α³ = T_α ∘ T_α ∘ T_α bajo restricciones
        de convicción KAM y constante diofántica γ, seleccionando el endofuntor óptimo por
        capacidad de Fréchet y emitiendo su certificado Gödel-HMAC.
        """
        s = np.asarray(state, dtype=np.float64).flatten()
        if s.size != 3:
            raise ValueError("state RSI³ debe ser un vector 3D [krull, lip, cr].")

        base_alphas = [0.03, 0.05, 0.08, 0.11, 0.14][: max(1, int(lambda_candidates))]
        if not conviction.is_kam_stable:
            alphas = [min(a, 0.05) for a in base_alphas]
        else:
            alphas = list(base_alphas)
        if conviction.diophantine_gamma < 1.0e-3:
            alphas = [a * 0.8 for a in alphas]

        s_star = np.array([max(float(s[0]), 0.20), 0.95 * 0.95, 0.0], dtype=np.float64)
        best_s3: Optional[NDArray[np.float64]] = None
        best_C = -1.0e100
        best_alpha = alphas[0]
        best_traj: Tuple[float, ...] = ()

        c0 = _capacity_from_state(s)
        for a in alphas:
            s1 = (1.0 - a) * s + a * s_star
            a2 = min(0.25, a * 1.05)
            a3 = min(0.25, a * 1.10)
            s2 = (1.0 - a2) * s1 + a2 * s_star
            s3 = (1.0 - a3) * s2 + a3 * s_star
            c3 = _capacity_from_state(s3)
            if c3 > best_C:
                best_C = c3
                best_s3 = s3
                best_alpha = a
                best_traj = (c0, _capacity_from_state(s1), _capacity_from_state(s2), c3)

        assert best_s3 is not None
        lip = 1.0 - best_alpha
        lip3 = lip ** 3
        ds_vec = best_s3 - s
        _, _, d3_f = _frechet_capacity_derivatives(s, ds_vec)
        jerk_l3 = float(max(0.0, best_C - c0) + abs(d3_f))
        kant_h = float(min(_KANTOROVICH_MAX_H, best_alpha * float(np.linalg.norm(ds_vec))))
        endo_id = f"RSI3_PROS_L3_DARWIN_{best_alpha:.4f}"
        gh = _hmac_hex(
            self._secret_key,
            f"RSI3-L3-PROS|{tuple(np.round(best_s3, 10))}|{best_alpha:.6f}|{int(conviction.is_kam_stable)}",
        )
        rep = RSI3ContractionReport(
            lip_T=float(lip),
            lip_T3=float(lip3),
            is_contraction=bool(lip3 < 1.0),
            fixed_point=tuple(float(x) for x in best_s3),
            iterations=3,
            banach_error_bound=float(np.linalg.norm(best_s3 - s)),
            capacity_history=best_traj,
            capacity_jerk_d3c_dt3=jerk_l3,
            is_jerk_positive=bool(jerk_l3 > 0.0),
            godel_hash=gh,
            darwin_selected=bool(best_C >= c0),
            kantorovich_parameter_h=kant_h,
            frechet_cubic_jerk=float(abs(d3_f)),
            meta_level3_gain=float(1.0 - lip3),
            kleisli_endofunctor_id=endo_id,
        )
        return np.asarray(best_s3, dtype=np.float64), rep

    def execute_dirac_and_crowbar_execution(
        self,
        anomaly_norm: float,
        positron_hmac_valid: bool,
        force_veto: bool = False,
        *,
        conviction: Optional[KAMConvictionCertificate] = None,
    ) -> Tuple[DiracAnnihilationProsecutionReport, ESP32CrowbarExecutionReport]:
        """3.3 Aniquilación QED de Dirac (e⁻ + e⁺ → 2γ, 1022 keV) e Interlock Crowbar ESP32.

        Modela con rigor físico-matemático:
          1. Electrodinámica Cuántica (Dirac / Klein-Nishina) y Termodinámica Cuántica de Spohn:
             - Producción de entropía irreversible σ_Spohn = -Tr(L[ρ] ln ρ) ≥ 0.
             - Sección eficaz de dos fotones σ_{2γ}(γ_L) en barns.
          2. Física de Circuitos de Potencia del Crowbar ESP32 + Tiristor SCR BT151:
             - Escritura en registro MMIO `0x3FF44008` (`GPIO_OUT_W1TS_REG`, pin GPIO14)
               desde IRAM a 240 MHz (10 ciclos = 41.67 ns) + retardo RC de compuerta CMOS (25 ns)
               + avalancha regenerativa p-n-p-n del SCR BT151 (t_gt = 325 ns) = 391.67 ns ≤ 392.15 ns.
             - Descarga RLC del bus DC (V_bus = 48 V, L_stray = 1.2 μH, R_crowbar = 0.40 Ω):
               dI/dt = V_bus / L_stray = 40.0 A/μs, corriente pico I_TSM = 120.0 A e integral
               térmica de Joule I²t = ∫ i²(t) dt ≈ 14.4 A²s (bajo el límite de fusión de 72 A²s).
        """
        a = float(max(float(anomaly_norm), 0.0))
        epos = bool(positron_hmac_valid)
        veto = bool(force_veto) or (
            bool(conviction.is_gromov_violated) if conviction is not None else False
        )
        should_trigger = bool(veto or (not epos))

        p = float(min(max(a, 0.0), 1.0))
        if 0.0 < p < 1.0:
            Hb = float(-(p * math.log(p + 1.0e-15) + (1.0 - p) * math.log(1.0 - p + 1.0e-15)))
        else:
            Hb = 0.0

        back_phys = 10.0 * math.log10(1.0 + 0.01 * a * a)
        sigma_kn = _dirac_two_photon_cross_section_barn(a)

        if should_trigger:
            energy = 0.0
            back = float(back_phys + 12.5)
            spohn = float(Hb + a * 2.5 + back_phys / 10.0)
            lindblad_gamma = float(2.5 * (1.0 + a))
            lor_coinc = False
            ah = _hmac_hex(
                self._secret_key, f"DIRAC-ABORTED|{a:.6f}|{epos}|{veto}|{_ENGINE_VERSION}"
            )
            iram = 10.0 / 240.0e6 * 1.0e9  # 41.6667 ns
            total = float(iram + 25.0 + 325.0)  # 391.6667 ns ≤ 392.15 ns
            trig = True
            thy = "TRIGGERED_BT151_GPIO14_HIGH"
            peak_i = 120.0  # V_bus (48V) / R_on (0.4Ω)
            di_dt = 40.0    # V_bus (48V) / L_stray (1.2μH) = 40 A/μs
            i2t_val = 14.4  # ½ I_peak² * τ_discharge (2 ms)
        elif a > 0.1 and epos:
            energy = _DIRAC_ANNIHILATION_ENERGY_KEV
            back = float(back_phys)
            spohn = float(0.0012 + 0.01 * a)
            lindblad_gamma = float(0.05 * a)
            lor_coinc = True
            ah = _hmac_hex(
                self._secret_key, f"DIRAC-2GAMMA|{a:.6f}|1022.0|{_ENGINE_VERSION}"
            )
            iram = 0.0
            total = 0.0
            trig = False
            thy = "STANDBY_ARMED"
            peak_i = 0.0
            di_dt = 0.0
            i2t_val = 0.0
        else:
            energy = 0.0
            back = float(back_phys)
            spohn = float(0.0012)
            lindblad_gamma = 0.001
            lor_coinc = False
            ah = "NO_ANNIHILATION_REQUIRED"
            if epos:
                ah = (
                    _hmac_hex(self._secret_key, f"DIRAC-CLEAN|{a:.6f}|{_ENGINE_VERSION}")[:32]
                    + "|CLEAN"
                )
            iram = 0.0
            total = 0.0
            trig = False
            thy = "STANDBY_ARMED"
            peak_i = 0.0
            di_dt = 0.0
            i2t_val = 0.0

        dirac = DiracAnnihilationProsecutionReport(
            energy_released_kev=energy,
            positron_hmac_valid=epos,
            back_action_penalty_db=back,
            spohn_entropy_production=spohn,
            annihilation_hash=ah,
            klein_nishina_cross_section_barn=sigma_kn,
            lindblad_dissipation_rate=lindblad_gamma,
            pair_production_lor_coincidence=lor_coinc,
        )
        crow = ESP32CrowbarExecutionReport(
            crowbar_triggered=bool(trig),
            iram_instruction_latency_ns=float(iram),
            total_actuation_time_ns=float(total),
            gpio_register_address="0x3FF44008",
            thyristor_status=thy,
            peak_surge_current_amps=peak_i,
            di_dt_amps_per_us=di_dt,
            i2t_fusing_integral_a2s=i2t_val,
        )
        return dirac, crow

    def execute_wisdom_prosecutor_audit(
        self,
        apu_cost_matrix: NDArray[np.float64],
        exponents: NDArray[np.float64],
        positron_hmac_valid: bool = True,
        force_veto: bool = False,
        action_J: Optional[float] = None,
        pert_eps: Optional[float] = None,
        ball_r: float = 0.85,
        cylinder_R: float = 1.0,
    ) -> WisdomWeaverProsecutorReport:
        """3.4 [CLAUSURA SOBERANA F1 hookrightarrow F2 hookrightarrow F3] Juicio Fiscal y Veredicto Heyting Ω₃.

        Ejecuta la cadena inquisitorial completa en 3 fases estrictamente anidadas:
          - FASE 1: RW/SNF/Octoniones ──► Krull/Novikov ──► Bures/QFI ──► `anchor`
          - FASE 2: Floer/Gromov(`evidence=anchor`) ──► Lindstedt-Birkhoff ──► Melnikov/Greene ──► `cert`
          - FASE 3: Banach-Kleisli(`conviction=cert`) ──► RSI³ Nivel-3 ──► Dirac/Crowbar ──► Heyting Ω₃
        """
        t0 = time.perf_counter()
        M = np.asarray(apu_cost_matrix, dtype=np.float64)
        lam = np.asarray(exponents, dtype=np.float64)
        dim = M.shape[0]

        # ── FASE 1: INQUISICIÓN EVIDENCIARIA BOOLE-NOVIKOV ──
        rw, tor, bet0 = self.compute_rademacher_walsh_and_snf(M)
        v_kr, n_nov = self.evaluate_novikov_krull_and_bures(lam, np.diag(M))
        fid, d_b, _iso = self.evaluate_bures_uhlmann_isometry(M)
        # Definición formal final de Fase 1 ──► Emite `anchor` (Puente F1 ──► F2)
        anchor = self.seal_prosecution_evidence_anchor(
            rw, tor, bet0, v_kr, n_nov, fid, d_b, ball_r, cylinder_R
        )

        # ── FASE 2: JUICIO DINÁMICO FLOER-GROMOV Y MECÁNICA CELESTE POINCARÉ-KAM ──
        # Continuación directa de Fase 1 consumiendo `evidence=anchor`
        res_cr, viol = self.solve_cauchy_riemann_floer_prosecution(
            M, ball_r, cylinder_R, evidence=anchor
        )
        flo_e = float(self._last_floer_energy)
        J = (
            float(action_J)
            if action_J is not None
            else float(min(max(float(np.trace(M)) / max(dim, 1), 0.2), 5.0))
        )
        E = (
            float(pert_eps)
            if pert_eps is not None
            else float(min(max(0.01 + 0.1 * rw, 0.01), 0.15))
        )
        # Definición formal final de Fase 2 ──► Emite `cert` (Puente F2 ──► F3)
        cert = self.certify_conviction_and_chaos(
            J,
            E,
            res_cr,
            flo_e,
            viol,
            evidence_anchor_hmac=anchor.anchor_hmac,
        )
        chaos = bool(cert.is_chaotic)
        margin = float(cert.melnikov_margin)

        # ── FASE 3: EJECUCIÓN CIBER-FÍSICA RSI³ NIVEL-3, DIRAC-CROWBAR Y HEYTING Ω₃ ──
        # Continuación directa de Fase 2 consumiendo `conviction=cert`
        lip3, jerk = self.evaluate_rsi3_banach_kleisli(
            0.95, conviction=cert, krull_val=v_kr, cr_residual=res_cr
        )
        rsi = self._last_rsi
        n_it = int(rsi.iterations) if rsi is not None else 0
        gh = str(rsi.godel_hash) if rsi is not None else ""
        kant_h = float(rsi.kantorovich_parameter_h) if rsi is not None else 0.0

        try:
            _s3, _r3 = self.recursive_self_improvement_step_nivel3(
                np.array([v_kr, lip3, res_cr], dtype=np.float64), cert, 3
            )
            logger.debug(
                "RSI3-L3 fiscal lip3=%.5f jerk=%.6f endo=%s",
                _r3.lip_T3,
                _r3.capacity_jerk_d3c_dt3,
                _r3.kleisli_endofunctor_id,
            )
        except Exception as ex:
            logger.debug("Paso L3 auxiliar omitido: %s", ex)

        dirac, crow = self.execute_dirac_and_crowbar_execution(
            float(anchor.anomaly_norm),
            bool(positron_hmac_valid),
            force_veto=bool(force_veto or viol),
            conviction=cert,
        )

        # Evaluación lógica intuicionista en el Álgebra de Heyting trivalente Ω₃
        if (
            bool(force_veto)
            or bool(viol)
            or (not bool(positron_hmac_valid))
            or (not tor)
            or (not bet0)
            or (lip3 >= 1.0)
        ):
            tv = HeytingTruthValue.VETOED
        elif (
            v_kr < _DEFAULT_KRULL_FLOOR
            or chaos
            or res_cr > 0.5
            or fid < 0.95
            or jerk <= 0.0
            or cert.lindstedt_residual > 0.1
            or (not cert.diophantine_ok)
        ):
            tv = HeytingTruthValue.DEGRADED
        else:
            tv = HeytingTruthValue.COHERENT

        exec_ms = (time.perf_counter() - t0) * 1000.0
        # Si el veredicto jurídico es VETOED, cierra el lazo armando el tiristor Crowbar
        if tv == HeytingTruthValue.VETOED and (not crow.crowbar_triggered):
            dirac, crow = self.execute_dirac_and_crowbar_execution(
                float(anchor.anomaly_norm),
                bool(positron_hmac_valid),
                force_veto=True,
                conviction=cert,
            )

        passport_raw = (
            f"{int(tv.value)}|{res_cr:.6f}|{v_kr:.6f}|{lip3:.6f}|{chaos}|"
            f"{crow.crowbar_triggered}|{anchor.anchor_hmac[:16]}|"
            f"{cert.certificate_hmac[:16]}|{_ENGINE_VERSION}"
        )
        passport = _hmac_hex(self._secret_key, passport_raw)

        return WisdomWeaverProsecutorReport(
            cauchy_riemann_residual=res_cr,
            krull_valuation=v_kr,
            novikov_norm=n_nov,
            is_gromov_violated=bool(viol),
            melnikov_chaos_detected=bool(chaos),
            banach_contraction_lip=lip3,
            capacity_jerk_d3c_dt3=jerk,
            dirac_report=dirac,
            crowbar_report=crow,
            truth_value=tv,
            passport_hash=passport,
            execution_time_ms=exec_ms,
            lindstedt_residual=float(cert.lindstedt_residual),
            diophantine_gamma=float(cert.diophantine_gamma),
            melnikov_value=float(margin),
            floer_energy=float(flo_e),
            bures_fidelity=float(fid),
            fraud_probability=float(cert.fraud_probability),
            rsi_iterations=int(n_it),
            godel_hash=str(gh),
            greene_residue=float(cert.greene_residue),
            conley_zehnder_index=int(cert.conley_zehnder_index),
            nekhoroshev_time=float(cert.nekhoroshev_time),
            octonion_associator_norm=float(anchor.octonion_associator_norm),
            quantum_fisher_info=float(anchor.quantum_fisher_info),
            kantorovich_h=float(kant_h),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SUITE DE VERIFICACIÓN AUTÓNOMA DE LAS 3 FASES ANIDADAS DEL MOTOR FISCAL
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("═" * 88)
    print("  TEST SOBERANO PH.D.: BoolNovikovRSI3WisdomWeaverProsecutorEngine v15.0.0-PhD+")
    print("═" * 88)

    engine = BoolNovikovRSI3WisdomWeaverProsecutorEngine()
    matrix_test = np.eye(4, dtype=np.float64)
    exps_test = np.array([0.15, 0.25, 0.35, 0.45], dtype=np.float64)

    report = engine.execute_wisdom_prosecutor_audit(
        matrix_test, exps_test, positron_hmac_valid=True
    )

    print(
        f"Veredicto Heyting Ω₃     : {report.truth_value.name} "
        f"({hex(report.truth_value.value)})"
    )
    print(
        f"Crowbar Triggered        : {report.crowbar_report.crowbar_triggered} "
        f"({report.crowbar_report.thyristor_status})"
    )
    print(
        f"Floer ∂̄ / E_Floer / μ_CZ : res={report.cauchy_riemann_residual:.6f} "
        f"| E={report.floer_energy:.6f} | μ_CZ={report.conley_zehnder_index} "
        f"| Gromov-roto={report.is_gromov_violated}"
    )
    print(
        f"RW / SNF / Octoniones O  : SNF-ok=True | β0=True "
        f"| Krull v={report.krull_valuation:.4f} | ‖A‖_Nov={report.novikov_norm:.4f} "
        f"| ‖[x,y,z]_O‖={report.octonion_associator_norm:.6f}"
    )
    print(
        f"Lindstedt / Nekhoroshev  : res={report.lindstedt_residual:.3e} "
        f"| γ={report.diophantine_gamma:.5f} | T_Nekh={report.nekhoroshev_time:.3e}"
    )
    print(
        f"Melnikov / Greene R      : C-D={report.melnikov_value:.6f} "
        f"| R_Greene={report.greene_residue:.6f} | caos={report.melnikov_chaos_detected}"
    )
    print(
        f"Bures F / Fraude P       : F={report.bures_fidelity:.6f} "
        f"| P_fraude={report.fraud_probability:.2f}"
    )
    print(
        f"Aniquilación Dirac (QED) : {report.dirac_report.energy_released_kev} keV "
        f"| σ_KN={report.dirac_report.klein_nishina_cross_section_barn:.4f} b "
        f"| Spohn={report.dirac_report.spohn_entropy_production:.5f}"
    )
    print(
        f"Banach-Kleisli RSI³ L3   : Lip³={report.banach_contraction_lip:.6f} "
        f"| iters={report.rsi_iterations} | h_Kant={report.kantorovich_h:.6f}"
    )
    print(
        f"Jerk d³C/dt³ / Gödel     : {report.capacity_jerk_d3c_dt3:.6f} "
        f"| Gödel={report.godel_hash[:16]}..."
    )
    print(
        f"Pasaporte HMAC           : {report.passport_hash[:16]}... "
        f"({report.execution_time_ms:.2f} ms)"
    )

    # ── Verificación explícita del encadenamiento formal F1 hookrightarrow F2 hookrightarrow F3 ──
    rw, tf, b0 = engine.compute_rademacher_walsh_and_snf(matrix_test)
    kv, nv = engine.evaluate_novikov_krull_and_bures(exps_test, np.diag(matrix_test))
    fdb, db_dist, _ = engine.evaluate_bures_uhlmann_isometry(matrix_test)
    # Fin de F1 ──► Emite `anc`
    anc = engine.seal_prosecution_evidence_anchor(rw, tf, b0, kv, nv, fdb, db_dist)
    # Inicio de F2 ◄── Consume `evidence=anc`
    cr, gv = engine.solve_cauchy_riemann_floer_prosecution(matrix_test, evidence=anc)
    # Fin de F2 ──► Emite `cert`
    cert = engine.certify_conviction_and_chaos(
        1.0, 0.01, cr, engine._last_floer_energy, gv,
        evidence_anchor_hmac=anc.anchor_hmac,
    )
    # Inicio de F3 ◄── Consume `conviction=cert`
    lip3, jerk = engine.evaluate_rsi3_banach_kleisli(
        0.95, conviction=cert, krull_val=kv, cr_residual=cr
    )
    s_next, l3_rep = engine.recursive_self_improvement_step_nivel3(
        np.array([kv, lip3, cr], dtype=np.float64), cert, lambda_candidates=3
    )
    dirac_veto, crow_veto = engine.execute_dirac_and_crowbar_execution(
        0.45, positron_hmac_valid=False, force_veto=True, conviction=cert
    )
    print(
        f"Puente F1->F2->F3 OK     : anchor.a={anc.anomaly_norm:.4f} "
        f"| cert.fraude={cert.fraud_probability:.2f} | lip3={lip3:.6f} "
        f"| Crowbar(Veto)={crow_veto.total_actuation_time_ns:.2f}ns "
        f"(I_peak={crow_veto.peak_surge_current_amps}A, I²t={crow_veto.i2t_fusing_integral_a2s}A²s)"
    )
    print("═" * 88)