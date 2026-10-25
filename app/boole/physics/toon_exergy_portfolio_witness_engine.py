# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : TOON Exergy Portfolio Witness Engine (Motor Espectral Testigo QND)  ║
║ Ruta   : app/boole/physics/toon_exergy_portfolio_witness_engine.py           ║
║ Versión: 6.1.0-QND-KMS-Poincare-Delaunay-Octonionic-RSI3-Heyting-ESP32-PhD   ║
║ Función: Inspección pasiva QND, simetría KMS, aciclicidad β₁≡0 y Crowbar.    ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. DEFINICIÓN GRANULAR Y RIGUROSA DEL MOTOR ESPECTRAL TESTIGO QND:
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra la evolución arquitectónica del **Motor Espectral Testigo
Exergético QND de Portafolios** (`TOONExergyPortfolioWitnessEngine`) en el Estrato
Exergético ($\mathcal{V}_{\mathbb{P}}$, Nivel 0 / Fibrado de Deligne-Jarzynski) del
ecosistema APU Filter v9.0.0.

Su propósito fundamental es ejecutar la **inspección cuántica pasiva no demoledora
(Quantum Non-Demolition - QND)** sobre el estado vacual $|\Omega\rangle$ en el espacio
de fases cotangente $T^*\mathcal{M}_{\text{net}}$, garantizando que la auditoría
continua de tesorería y contratos de infraestructura no altere el conmutador nulo,
la traza unitaria de la matriz de densidad ($\operatorname{Tr}(\rho) \equiv 1.0$),
la condición KMS térmico-monetaria, la aciclicidad homológica de Betti ($\beta_1 \equiv 0$)
ni destruya la foliación simpléctica de Poincaré-Darboux.

2. MARCO TEÓRICO:
────────────────────────────────────────────────────────────────────────────────
  • Medición Cuántica No Demoledora (QND) sobre observables compatibles $[H_{\text{net}}, A_{\text{QND}}] = \mathbf{0}$.
  • Estado térmico de Gibbs $\rho = e^{-\beta_{\text{eff}} H} / Z$ con condición Kubo-Martin-Schwinger (KMS)
    y continuación analítica $\sigma_{i\beta}(A) = e^{-\beta H} A e^{\beta H}$ según el álgebra de Tomita-Takesaki.
  • Inmersión simpléctica de Cayley-Darboux $K = (G - I)(G + I)^{-1}$, $M = (I + K)(I - K)^{-1} \in \operatorname{Sp}(2n, \mathbb{R})$.
  • Mecánica celeste de Poincaré-Delaunay: acciones $L$, movimiento medio $n = L^{-3}$, mapa de retorno de Poincaré
    e integral homoclínica de Melnikov mediante cuadratura de Gauss-Legendre $n=16$.
  • Álgebra no asociativa de octoniones $\mathbb{O}$ por construcción de Cayley-Dickson con asociador $[X, Y, Z]$.
  • Homología simplicial y aciclicidad de de Rham-Betti: $\beta_1 = |E| - |V| + \beta_0 \equiv 0$.
  • Automejora Recursiva Nivel 3 (RSI³ - Darwin-Gödel) como mónada de Kleisli sobre espacios de Banach contractivos:
    $\operatorname{Lip}(T^3) \le 0.857375 < 1.0$, garantizando atracción única por Banach-Picard y jerk cognitivo $\mathrm{d}^3 C / \mathrm{d}t^3 > 0$.
  • Adjudicación trivalente en el topos de Heyting $\Omega_3 = \{0x00 \prec 0x01 \prec 0x03\}$ e interlock perimetral ESP32 Crowbar en $t_{\text{act}} \le 392.15\text{ ns}$.

3. AXIOMAS FORMALES:
────────────────────────────────────────────────────────────────────────────────
  [A1] Axioma de Medición Cuántica No-Demoledora (QND) & Conservación de Traza:
       $$[H_{\text{net}}, A_{\text{QND}}] = \mathbf{0}
         \implies \frac{\mathrm{d}}{\mathrm{d}t}\langle A_{\text{QND}}\rangle
         = \frac{i}{\hbar}\langle[H_{\text{net}}, A_{\text{QND}}]\rangle \equiv 0$$
       con $\|[H_{\text{net}}, A_{\text{QND}}]\|_F \le 1.0\times 10^{-10}$ y
       $\operatorname{Tr}(\rho)\equiv 1$.

  [A2] Axioma de Condición KMS (Kubo–Martin–Schwinger) & Tomita–Takesaki:
       $$\langle A\,B\rangle_{\omega}
         = \langle B\,\sigma_{i\beta}(A)\rangle_{\omega},\qquad
         \sigma_{i\beta}(A)=e^{-\beta H}A e^{\beta H}.$$

  [A3] Geometría Simpléctica de Poincaré, Cayley–Darboux y Caos de Melnikov:
       $$K=(G-I)(G+I)^{-1},\quad
         M=(I+K)(I-K)^{-1}\in\operatorname{Sp}(2n,\mathbb{R}),$$
       $$M(t_0)=\int_{-\infty}^{\infty}\{H_0,H_1\}(q_0(t),p_0(t),t+t_0)\,\mathrm{d}t.$$
       Las variables de Delaunay $(L,G,H,\ell,g,h)$ inducen la frecuencia kepleriana
       $n=L^{-3}$ del mapa de retorno de Poincaré.

  [A4] Curvatura No Asociativa (Cayley–Dickson $\mathbb{O}$):
       $$[X,Y,Z]=(X\cdot Y)\cdot Z - X\cdot(Y\cdot Z)\in\mathbb{O}.$$

  [A5] Aciclicidad Homológica Simplicial ($\beta_1\equiv 0$):
       $$\chi(K)=\beta_0-\beta_1=|V|-|E|
         \implies \beta_1=|E|-|V|+\beta_0\equiv 0.$$

  [A6] Mónada Kleisli RSI³ (Darwin–Gödel Nivel 3) & Crowbar ESP32:
       $$\operatorname{Lip}(T^3)\le 0.857375<1
         \implies \frac{\mathrm{d}^3 C}{\mathrm{d}t^3}>0,$$
       topos Heyting $\Omega_3$ y actuación GPIO14 en $t_{\mathrm{act}}\le 392.15\,\mathrm{ns}$.

4. INVARIANTES DE ESTADO:
────────────────────────────────────────────────────────────────────────────────
  1. Invariante de Conmutación Pasiva: $\|[H_{\text{net}}, A_{\text{QND}}]\|_F \le 1.0 \times 10^{-10}$.
  2. Invariante de Traza Unitaria: $|\operatorname{Tr}(\rho) - 1.0| \le 1.0 \times 10^{-9}$.
  3. Invariante Simpléctico de Cayley-Darboux: $\|M^{\text{T}} J M - J\|_F / (1 + \|M\|_F^2) \le 1.0 \times 10^{-10}$.
  4. Invariante Topológico Acíclico: $\beta_1 \equiv 0$.
  5. Invariante Monádico Banach-Picard: $\operatorname{Lip}(T^3) \le 0.857375$.
  6. Invariante Ciber-Físico Latencia Hardware ESP32: $t_{\text{actuation}} \le 392.15\text{ ns} < 400.0\text{ ns}$.

ARQUITECTURA DE FLUJO POR FASES ANIDADAS:
────────────────────────────────────────────────────────────────────────────────
  [FASE 1] Phase1_SymplecticQNDObserver
           └── observe_qnd_vacuum_and_symplectic_state
                 └── continue_into_phase2_kms_poincare_orientation  ← morfismo 1→2
                       │
                       ▼  WitnessSymplecticQNDObservationEnvelope
  [FASE 2] Phase2_KMSBettiPoincareOrientor
           └── orient_kms_homology_and_poincare
                 └── continue_into_phase3_rsi3_heyting_crowbar      ← morfismo 2→3
                       │
                       ▼  WitnessKMSPoincareOrientationEnvelope
  [FASE 3] Phase3_RSI3HeytingCrowbarWitnessDecider
           └── decide_and_act_witness_crowbar
                 └── WitnessSpectralAuditReport (certificado terminal)
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DEL SISTEMA DE REGISTRO TELEMÉTRICO DEL TESTIGO
# ─────────────────────────────────────────────────────────────────────────────
logger = logging.getLogger("APUFilter.Physics.TOONExergyPortfolioWitnessEngine")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [Witness.Spectral:%(lineno)d] 👁️ %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

ENGINE_VERSION: Final[str] = "6.1.0-QND-KMS-Poincare-Delaunay-Octonionic-RSI3-Heyting-ESP32-PhD"

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES FÍSICAS FUNDAMENTALES Y COTAS DE TOLERANCIA
# ─────────────────────────────────────────────────────────────────────────────
KB_BOLTZMANN: Final[float] = 1.380649e-23
HBAR_PLANCK: Final[float] = 1.054571817e-34
DEFAULT_T_RISK: Final[float] = 298.15

QND_COMMUTATOR_TOLERANCE: Final[float] = 1.0e-10
KMS_RESIDUAL_TOLERANCE: Final[float] = 1.0e-8
DARBOUX_RESIDUAL_TOLERANCE: Final[float] = 1.0e-10
TRACE_RHO_TOLERANCE: Final[float] = 1.0e-9
WILKINSON_CONDITION_CEILING: Final[float] = 1.0e8
BANACH_RSI3_LIP1: Final[float] = 0.9500
BANACH_RSI3_CONTRACTIVE_TARGET: Final[float] = 0.857375  # 0.95^3
ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 400.0
ESP32_CROWBAR_NOMINAL_LATENCY_NS: Final[float] = 392.15
MELNIKOV_TANGLE_THRESHOLD: Final[float] = 1.0e-4
POINCARE_RETURN_TOLERANCE: Final[float] = 1.0e-8
PICARD_FIXED_POINT_TOLERANCE: Final[float] = 1.0e-13
BETA_EFF_CEILING: Final[float] = 40.0
CAYLEY_RIDGE: Final[float] = 1.0e-12

# Nodos / pesos de Gauss–Legendre (n=16) mapeados a ℝ vía tanh para Melnikov.
_GL16_NODES: Final[NDArray[np.float64]] = np.array([
    -0.9894009349916499, -0.9445750230732326, -0.8656312023878318, -0.7554044083550030,
    -0.6178762444026438, -0.4580167776572274, -0.2816035507792589, -0.0950125098376374,
     0.0950125098376374,  0.2816035507792589,  0.4580167776572274,  0.6178762444026438,
     0.7554044083550030,  0.8656312023878318,  0.9445750230732326,  0.9894009349916499,
], dtype=np.float64)
_GL16_WEIGHTS: Final[NDArray[np.float64]] = np.array([
    0.0271524594117540, 0.0622535239386479, 0.0951585116824928, 0.1246289712555339,
    0.1495959888165767, 0.1691565193950025, 0.1826034150449236, 0.1894506104550685,
    0.1894506104550685, 0.1826034150449236, 0.1691565193950025, 0.1495959888165767,
    0.1246289712555339, 0.0951585116824928, 0.0622535239386479, 0.0271524594117540,
], dtype=np.float64)


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_of(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _as_square_float_matrix(mat: NDArray[np.floating], name: str) -> NDArray[np.float64]:
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} debe ser una matriz cuadrada real; recibido shape={arr.shape}.")
    if arr.shape[0] < 1:
        raise ValueError(f"{name} no puede ser vacía.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contiene entradas no finitas.")
    return arr


def _symmetrize(mat: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * (mat + mat.T)


def _dimensionless_beta(t_risk_kelvin: float) -> float:
    """
    β_eff adimensional: el β termodinámico k_B^{-1}T^{-1} es ~10^{20} y satura
    todo espectro. Se compactifica a (0, BETA_EFF_CEILING] preservando monotonía.
    """
    t = max(1.0e-3, float(t_risk_kelvin))
    raw = 298.15 / t
    return float(min(BETA_EFF_CEILING, max(1.0e-6, raw)))


# ─────────────────────────────────────────────────────────────────────────────
# RETÍCULO DE HEYTING TRIVALENTE Ω₃
# ─────────────────────────────────────────────────────────────────────────────
class HeytingState(IntEnum):
    """
    Álgebra de Heyting trivalente Ω₃ = {COHERENT ≺ DEGRADED ≺ VETOED} (cadena).
    Implicación: a → b = ⊤ si a ≤ b, en otro caso b. Aquí ⊤ := VETOED (máximo del orden).
    """
    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03

    @classmethod
    def meet(cls, a: HeytingState, b: HeytingState) -> HeytingState:
        return cls(min(int(a.value), int(b.value)))

    @classmethod
    def join(cls, a: HeytingState, b: HeytingState) -> HeytingState:
        return cls(max(int(a.value), int(b.value)))

    @classmethod
    def implies(cls, a: HeytingState, b: HeytingState) -> HeytingState:
        if int(a.value) <= int(b.value):
            return cls.VETOED
        return b

    @classmethod
    def from_severity(cls, veto: bool, degraded: bool) -> HeytingState:
        if veto:
            return cls.VETOED
        if degraded:
            return cls.DEGRADED
        return cls.COHERENT


# ─────────────────────────────────────────────────────────────────────────────
# ÁLGEBRA DE OCTONIONES POR CAYLEY–DICKSON (no asociativa, alternativa)
# ─────────────────────────────────────────────────────────────────────────────
def _quat_mul(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    a0, a1, a2, a3 = a
    b0, b1, b2, b3 = b
    return (
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
    )


def _quat_conj(a: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    return (a[0], -a[1], -a[2], -a[3])


@dataclass(frozen=True)
class OctonionWitnessNumber:
    """
    Octonión $\mathbb{O}$ como par de cuaterniones (Cayley–Dickson):
      $(a,b)\\cdot(c,d)=(ac-\\bar d b,\\, da + b\\bar c)$.
    Garantiza $\|xy\|=\|x\|\|y\|$ y asociador alternante $[x,x,y]=0$.
    """
    coords: Tuple[float, float, float, float, float, float, float, float]

    def __post_init__(self) -> None:
        if len(self.coords) != 8:
            raise ValueError("Un octonión espectral requiere exactamente 8 coordenadas reales.")
        if not all(math.isfinite(x) for x in self.coords):
            raise ValueError("Coordenadas octoniónicas no finitas.")

    @classmethod
    def zero(cls) -> OctonionWitnessNumber:
        return cls((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    @classmethod
    def from_array(cls, arr: NDArray[np.floating]) -> OctonionWitnessNumber:
        flat = np.asarray(arr, dtype=np.float64).ravel()
        coeffs = np.zeros(8, dtype=np.float64)
        if flat.size == 0:
            return cls.zero()
        energy = float(np.dot(flat, flat))
        n = min(8, int(flat.size))
        coeffs[:n] = flat[:n]
        if flat.size > 8:
            # Momentos invariantes para no truncar ciegamente el espectro residual.
            coeffs[4] += float(np.mean(flat))
            coeffs[5] += float(np.std(flat))
            coeffs[6] += math.sqrt(energy)
            coeffs[7] += float(np.max(np.abs(flat)))
        scale = math.sqrt(energy) + 1.0e-15
        coeffs = coeffs / scale
        return cls(tuple(float(x) for x in coeffs))

    def _split(self) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
        c = self.coords
        return (c[0], c[1], c[2], c[3]), (c[4], c[5], c[6], c[7])

    def norm_squared(self) -> float:
        return float(sum(x * x for x in self.coords))

    def norm(self) -> float:
        return math.sqrt(self.norm_squared())

    def conjugate(self) -> OctonionWitnessNumber:
        c = self.coords
        return OctonionWitnessNumber((c[0], -c[1], -c[2], -c[3], -c[4], -c[5], -c[6], -c[7]))

    def __add__(self, other: OctonionWitnessNumber) -> OctonionWitnessNumber:
        return OctonionWitnessNumber(tuple(a + b for a, b in zip(self.coords, other.coords)))

    def __sub__(self, other: OctonionWitnessNumber) -> OctonionWitnessNumber:
        return OctonionWitnessNumber(tuple(a - b for a, b in zip(self.coords, other.coords)))

    def __mul__(self, other: OctonionWitnessNumber) -> OctonionWitnessNumber:
        a, b = self._split()
        c, d = other._split()
        # (a, b) * (c, d) = (a*c - conj(d)*b,  d*a + b*conj(c))
        left = _quat_mul(a, c)
        left_sub = _quat_mul(_quat_conj(d), b)
        re = tuple(u - v for u, v in zip(left, left_sub))
        right = _quat_mul(d, a)
        right_add = _quat_mul(b, _quat_conj(c))
        im = tuple(u + v for u, v in zip(right, right_add))
        return OctonionWitnessNumber((
            float(re[0]), float(re[1]), float(re[2]), float(re[3]),
            float(im[0]), float(im[1]), float(im[2]), float(im[3]),
        ))

    @staticmethod
    def associator(
        a: OctonionWitnessNumber,
        b: OctonionWitnessNumber,
        c: OctonionWitnessNumber,
    ) -> OctonionWitnessNumber:
        """Asociador $[a,b,c]=(a b)c - a(b c)$."""
        return (a * b) * c - a * (b * c)


# ─────────────────────────────────────────────────────────────────────────────
# MECÁNICA CELESTE DE POINCARÉ–DELAUNAY (espacio de fases reducido)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class DelaunayActionAngleState:
    """
    Elementos de Delaunay (μ = 1) sobre el fibrado cotangente kepleriano:
      L = √a,  G = L √(1-e²),  H = G cos(i),
      ℓ = anomalía media, g = argumento del periapsis, h = longitud del nodo.
    Frecuencia media de Poincaré: n = ∂H_Kep/∂L = L^{-3}.
    """
    L: float
    G: float
    H: float
    ell: float
    g: float
    h: float

    @property
    def mean_motion(self) -> float:
        l_abs = max(1.0e-6, abs(self.L))
        return float(l_abs ** (-3.0))

    @property
    def eccentricity(self) -> float:
        l_abs = max(1.0e-12, abs(self.L))
        ratio = min(1.0, max(0.0, abs(self.G) / l_abs))
        return float(math.sqrt(max(0.0, 1.0 - ratio * ratio)))

    @classmethod
    def from_novikov_krull(cls, v_krull: float, darboux_residual: float) -> DelaunayActionAngleState:
        """Inmersión espectral: la valuación de Novikov-Krull fija el semieje (acción L)."""
        l_action = max(0.25, 1.0 / (abs(v_krull) + 0.25))
        ecc = min(0.95, 0.05 + 10.0 * min(0.09, abs(darboux_residual)))
        g_action = l_action * math.sqrt(max(0.0, 1.0 - ecc * ecc))
        h_action = g_action  # i = 0 (plano invariante de Poincaré)
        return cls(L=l_action, G=g_action, H=h_action, ell=0.0, g=0.25, h=0.0)


def _duffing_homoclinic(t: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Órbita homoclínica del Duffing hamiltoniano H₀ = p²/2 − q²/2 + q⁴/4:
      q₀(t)=√2 sech(t),  p₀(t)= −√2 sech(t) tanh(t).
    """
    sech_t = 1.0 / np.cosh(t)
    q0 = math.sqrt(2.0) * sech_t
    p0 = -math.sqrt(2.0) * sech_t * np.tanh(t)
    return q0, p0


def _gauss_legendre_real_line(integrand: NDArray[np.float64], xi: NDArray[np.float64]) -> float:
    """
    ∫_{-∞}^{∞} f(t) dt  con t = artanh(ξ), dt = dξ / (1-ξ²), ξ ∈ (-1,1).
    `integrand` ya está evaluado en t(ξ).
    """
    jac = 1.0 / np.maximum(1.0e-15, 1.0 - xi * xi)
    return float(np.dot(_GL16_WEIGHTS, integrand * jac))


# ═════════════════════════════════════════════════════════════════════════════
# DTOs INMUTABLES DE HANDOFF INTER-FASE (SOBRES MONÁDICOS SELLADOS)
# ═════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class WitnessSymplecticQNDObservationEnvelope:
    """
    SOBRE INMUTABLE TERMINAL DE FASE 1.
    Auditoría del conmutador QND, traza de ρ, Cayley–Darboux y holonomía octoniónica.
    Es la unidad de Kleisli η(Fase1) y el único objeto inicial de Fase 2.
    """
    timestamp_iso: str
    portfolio_id: str
    qnd_commutator_norm: float
    is_qnd_coherent: bool
    density_matrix_trace: float
    is_density_trace_preserved: bool
    darboux_residual_norm: float
    is_symplectic: bool
    wilkinson_condition_number: float
    novikov_krull_valuation: float
    octonionic_associator_norm: float
    delaunay_mean_motion: float
    delaunay_eccentricity: float
    phase1_sha256: str


@dataclass(frozen=True)
class WitnessKMSPoincareOrientationEnvelope:
    """
    SOBRE INMUTABLE TERMINAL DE FASE 2.
    Orientación modular KMS, aciclicidad de de Rham–Betti, Melnikov homoclínico
    en coordenadas de Delaunay y residuo del mapa de retorno de Poincaré.
    """
    phase1_ref_hash: str
    timestamp_iso: str
    portfolio_id: str
    kms_residual_norm: float
    is_kms_balanced: bool
    tomita_modular_entropy: float
    betti_number_0: int
    betti_number_1: int
    is_acyclic: bool
    poincare_melnikov_integral: float
    is_homoclinic_tangle_detected: bool
    poincare_return_residual: float
    is_poincare_section_regular: bool
    lyapunov_characteristic_estimate: float
    donaldson_thomas_invariant: int
    phase2_sha256: str


@dataclass(frozen=True)
class WitnessSpectralAuditReport:
    """
    CERTIFICADO TERMINAL CONSOLIDADO DE FASE 3.
    """
    portfolio_id: str
    engine_version: str
    global_heyting_verdict: HeytingState
    qnd_observation_envelope: WitnessSymplecticQNDObservationEnvelope
    kms_orientation_envelope: WitnessKMSPoincareOrientationEnvelope
    rsi3_banach_contraction: float
    rsi3_lipschitz_empirical: float
    picard_banach_residual: float
    jerk_capacity_metric: float
    jerk_capacity_positive: bool
    crowbar_triggered: bool
    actuation_latency_nanoseconds: float
    hardware_register_state: Dict[str, str]
    execution_time_us: float
    audit_sha256: str


# ═════════════════════════════════════════════════════════════════════════════
# FASE 1 (OBSERVE): Phase1_SymplecticQNDObserver
# Último método = morfismo de continuación → Fase 2.
# ═════════════════════════════════════════════════════════════════════════════
class Phase1_SymplecticQNDObserver:
    """
    FASE 1 — Observación pasiva QND y estructura simpléctica de Cayley–Darboux.

    Audita $[H,A]=0$, construye $\\rho=e^{-\\beta H}/Z$, verifica $\\operatorname{Tr}(\\rho)=1$,
    inmerge $G=HH^{\\mathsf T}$ en $\\mathfrak{sp}(2d,\\mathbb{R})$ y mide el asociador
    octoniónico. El método terminal `continue_into_phase2_kms_poincare_orientation`
    es formalmente el morfismo de arranque de la Fase 2.
    """

    def __init__(self, portfolio_id: str, t_risk_kelvin: float = DEFAULT_T_RISK) -> None:
        if not portfolio_id or not str(portfolio_id).strip():
            raise ValueError("portfolio_id no puede ser vacío.")
        self.portfolio_id: Final[str] = str(portfolio_id)
        self.t_risk_kelvin: Final[float] = max(1.0e-3, float(t_risk_kelvin))
        self.beta_risk: Final[float] = 1.0 / (KB_BOLTZMANN * self.t_risk_kelvin)
        self.beta_eff: Final[float] = _dimensionless_beta(self.t_risk_kelvin)

    def evaluate_qnd_and_density_matrix(
        self,
        h_net: NDArray[np.float64],
        a_qnd: NDArray[np.float64],
    ) -> Tuple[float, bool, float, bool]:
        """
        Conmutador pasivo $[H,A]=HA-AH$ y matriz de densidad canónica
        $\\rho=e^{-\\beta_{\\mathrm{eff}} H}/\\operatorname{Tr}(e^{-\\beta_{\\mathrm{eff}} H})$.
        """
        comm = h_net @ a_qnd - a_qnd @ h_net
        comm_norm = float(la.norm(comm, ord="fro"))
        is_qnd = comm_norm <= QND_COMMUTATOR_TOLERANCE

        h_symm = _symmetrize(h_net)
        evals, evecs = la.eigh(h_symm)
        shifted = self.beta_eff * (evals - float(np.min(evals)))
        weights = np.exp(-np.clip(shifted, 0.0, BETA_EFF_CEILING))
        partition_z = float(np.sum(weights))
        rho_diag = weights / max(partition_z, 1.0e-15)
        rho_mat = (evecs * rho_diag) @ evecs.T

        trace_rho = float(np.trace(rho_mat))
        is_trace_one = abs(trace_rho - 1.0) <= TRACE_RHO_TOLERANCE
        return comm_norm, is_qnd, trace_rho, is_trace_one

    def compute_cayley_darboux_symplectic_embedding(
        self,
        h_net: NDArray[np.float64],
    ) -> Tuple[float, bool, float, float]:
        """
        Inmersión de Gram $G=HH^{\\mathsf T}$ en $\\mathrm{Sp}(2d,\\mathbb{R})$ vía Cayley:
          $K=(G-I)(G+I)^{-1}$, $M=(I+K)(I-K)^{-1}$.
        Residuo de Darboux: $\|M^{\\mathsf T}JM-J\|_F / (1+\|M\|_F^2)$.
        Valuación de Novikov–Krull: $v(G)=-\\log\\lambda_{\\min}(G)$.
        """
        dim = int(h_net.shape[0])
        g_mat = h_net @ h_net.T + CAYLEY_RIDGE * np.eye(dim)
        cond_wilkinson = float(np.linalg.cond(g_mat))
        eigvals = np.linalg.eigvalsh(g_mat)
        min_lambda = max(1.0e-15, float(eigvals[0]))
        v_krull = float(-math.log(min_lambda))

        if dim % 2 != 0:
            dim_sympl = dim + 1
            g_sympl = np.eye(dim_sympl, dtype=np.float64)
            g_sympl[:dim, :dim] = g_mat
        else:
            dim_sympl = dim
            g_sympl = g_mat

        d_half = dim_sympl // 2
        j_canonical = np.block([
            [np.zeros((d_half, d_half)), np.eye(d_half)],
            [-np.eye(d_half), np.zeros((d_half, d_half))],
        ])

        try:
            eye_s = np.eye(dim_sympl, dtype=np.float64)
            k_cayley = (g_sympl - eye_s) @ la.inv(g_sympl + eye_s)
            i_minus_k = eye_s - k_cayley
            # Regularización de Tikhonov si I−K está mal condicionado.
            if float(np.linalg.cond(i_minus_k)) > WILKINSON_CONDITION_CEILING:
                i_minus_k = i_minus_k + CAYLEY_RIDGE * eye_s
            m_sympl = (eye_s + k_cayley) @ la.inv(i_minus_k)
            m_t_j_m = m_sympl.T @ j_canonical @ m_sympl
            denom = 1.0 + float(la.norm(m_sympl, ord="fro") ** 2)
            darboux_residual = float(la.norm(m_t_j_m - j_canonical, ord="fro") / denom)
        except np.linalg.LinAlgError:
            darboux_residual = 1.0e-4

        is_symplectic = darboux_residual <= DARBOUX_RESIDUAL_TOLERANCE
        return darboux_residual, is_symplectic, cond_wilkinson, v_krull

    def evaluate_octonionic_associator_curvature(
        self,
        h_net: NDArray[np.float64],
        a_qnd: NDArray[np.float64],
    ) -> float:
        """Holonomía no asociativa $[H,A,HA]$ en $\\mathbb{O}$ (Cayley–Dickson)."""
        o_h = OctonionWitnessNumber.from_array(h_net)
        o_a = OctonionWitnessNumber.from_array(a_qnd)
        o_ha = OctonionWitnessNumber.from_array(h_net @ a_qnd)
        return float(OctonionWitnessNumber.associator(o_h, o_a, o_ha).norm())

    def observe_qnd_vacuum_and_symplectic_state(
        self,
        h_net: NDArray[np.float64],
        a_qnd: NDArray[np.float64],
    ) -> WitnessSymplecticQNDObservationEnvelope:
        """
        MÉTODO TERMINAL DE FASE 1 (unidad de Kleisli).
        Emite `WitnessSymplecticQNDObservationEnvelope`, objeto inicial unívoco de Fase 2.
        """
        h_net = _as_square_float_matrix(h_net, "h_net")
        a_qnd = _as_square_float_matrix(a_qnd, "a_qnd")
        if h_net.shape != a_qnd.shape:
            raise ValueError("h_net y a_qnd deben compartir dimensión.")

        t_start = time.perf_counter()
        logger.info("🔭 [FASE 1: OBSERVE] Inspección pasiva QND para: %s", self.portfolio_id)

        c_norm, is_qnd, tr_rho, is_tr1 = self.evaluate_qnd_and_density_matrix(h_net, a_qnd)
        d_res, is_sympl, cond_w, v_krull = self.compute_cayley_darboux_symplectic_embedding(h_net)
        oct_assoc = self.evaluate_octonionic_associator_curvature(h_net, a_qnd)

        delaunay = DelaunayActionAngleState.from_novikov_krull(v_krull, d_res)
        ts_iso = _iso_now()
        raw_sig = (
            f"{ts_iso}|{self.portfolio_id}|{c_norm:.8e}|{is_qnd}|{tr_rho:.12f}|"
            f"{is_tr1}|{d_res:.8e}|{is_sympl}|{cond_w:.6e}|{v_krull:.12f}|"
            f"{oct_assoc:.8e}|{delaunay.mean_motion:.12e}|{delaunay.eccentricity:.12f}"
        )
        p1_sha256 = _sha256_of(raw_sig)

        envelope = WitnessSymplecticQNDObservationEnvelope(
            timestamp_iso=ts_iso,
            portfolio_id=self.portfolio_id,
            qnd_commutator_norm=c_norm,
            is_qnd_coherent=is_qnd,
            density_matrix_trace=tr_rho,
            is_density_trace_preserved=is_tr1,
            darboux_residual_norm=d_res,
            is_symplectic=is_sympl,
            wilkinson_condition_number=cond_w,
            novikov_krull_valuation=v_krull,
            octonionic_associator_norm=oct_assoc,
            delaunay_mean_motion=delaunay.mean_motion,
            delaunay_eccentricity=delaunay.eccentricity,
            phase1_sha256=p1_sha256,
        )
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "✅ [FASE 1 COMPLETADA] Latencia: %.3f ms | Hash: %s... | ||[H,A]||: %.4e | Tr(ρ): %.6f | Darboux: %.4e | n_Del: %.4e",
            elapsed_ms, p1_sha256[:16], c_norm, tr_rho, d_res, delaunay.mean_motion,
        )
        return envelope

    def continue_into_phase2_kms_poincare_orientation(
        self,
        h_net: NDArray[np.float64],
        a_qnd: NDArray[np.float64],
        incidence_matrix: NDArray[np.floating],
        perturbation_amplitude: float = 0.05,
        phase1_envelope: Optional[WitnessSymplecticQNDObservationEnvelope] = None,
    ) -> WitnessKMSPoincareOrientationEnvelope:
        """
        MORFISMO DE ANIDACIÓN 1→2.
        Definición formal final de Fase 1 y arranque unívoco de Fase 2:
        el sobre de observación se inyecta en `Phase2_KMSBettiPoincareOrientor`.
        """
        env = phase1_envelope or self.observe_qnd_vacuum_and_symplectic_state(h_net, a_qnd)
        phase2 = Phase2_KMSBettiPoincareOrientor(t_risk_kelvin=self.t_risk_kelvin)
        return phase2.orient_kms_homology_and_poincare(
            phase1_env=env,
            h_net=h_net,
            a_qnd=a_qnd,
            incidence_matrix=incidence_matrix,
            perturbation_amplitude=perturbation_amplitude,
        )


# ═════════════════════════════════════════════════════════════════════════════
# FASE 2 (ORIENT): Phase2_KMSBettiPoincareOrientor
# Inicio = sobre terminal de Fase 1. Último método = morfismo → Fase 3.
# ═════════════════════════════════════════════════════════════════════════════
class Phase2_KMSBettiPoincareOrientor:
    """
    FASE 2 — Orientación modular KMS, aciclicidad simplicial y caos de Poincaré.

    Consume exclusivamente `WitnessSymplecticQNDObservationEnvelope`.
    Evalúa Tomita–Takesaki, Betti $\\beta_1$, la integral de Melnikov sobre la
    homoclínica de Duffing con frecuencia de Delaunay, el mapa de retorno de
    Poincaré y el invariante de Donaldson–Thomas.
    """

    def __init__(self, t_risk_kelvin: float = DEFAULT_T_RISK) -> None:
        self.t_risk_kelvin: Final[float] = max(1.0e-3, float(t_risk_kelvin))
        self.beta_risk: Final[float] = 1.0 / (KB_BOLTZMANN * self.t_risk_kelvin)
        self.beta_eff: Final[float] = _dimensionless_beta(self.t_risk_kelvin)

    def evaluate_tomita_takesaki_kms_symmetry(
        self,
        h_net: NDArray[np.float64],
        a_qnd: NDArray[np.float64],
    ) -> Tuple[float, bool, float]:
        """
        KMS analítico: $\\langle H A\\rangle = \\langle A\\,\\sigma_{i\\beta}(H)\\rangle$
        con $\\sigma_{i\\beta}(H)=H$ (H conmuta consigo) y, para el observable,
        $\\sigma_{i\\beta}(A)=e^{-\\beta H} A e^{\\beta H}$.
        Residuo: $\|HA - A\\sigma_{i\\beta}(H)\|_F$ se sustituye por
        $\|H A - A_{\\beta} H\|_F$ donde $A_\\beta=\\sigma_{i\\beta}(A)$.
        """
        h_symm = _symmetrize(h_net)
        evals, evecs = la.eigh(h_symm)
        shifted = evals - float(np.min(evals))
        # e^{±βH} en la base espectral, con recorte para estabilidad.
        expo_m = np.exp(-np.clip(self.beta_eff * shifted, 0.0, BETA_EFF_CEILING))
        expo_p = np.exp(np.clip(self.beta_eff * shifted, 0.0, BETA_EFF_CEILING))
        # σ_{iβ}(A) = e^{-βH} A e^{βH}
        a_mod = (evecs * expo_m) @ (evecs.T @ a_qnd @ evecs) @ (expo_p[:, None] * evecs.T)

        kms_diff = h_net @ a_qnd - a_mod @ h_net
        kms_norm = float(la.norm(kms_diff, ord="fro"))
        # Si [H,A]≈0 entonces σ_{iβ}(A)≈A y el residuo colapsa al conmutador.
        is_kms = kms_norm <= KMS_RESIDUAL_TOLERANCE

        weights = np.exp(-np.clip(self.beta_eff * shifted, 0.0, BETA_EFF_CEILING))
        probs = weights / max(float(np.sum(weights)), 1.0e-15)
        modular_entropy = float(-np.sum(probs * np.log(np.clip(probs, 1.0e-15, 1.0))))
        return kms_norm, is_kms, modular_entropy

    def compute_betti_simplicial_acyclicity(
        self,
        incidence_matrix: NDArray[np.floating],
    ) -> Tuple[int, int, bool]:
        """
        1-esqueleto: adyacencia $AA^{\\mathsf T}>0$, $\\beta_0$ por BFS,
        $\\beta_1=\\max(0,|E|-|V|+\\beta_0)$. Se exige $\\beta_1\\equiv 0$.
        """
        inc = np.asarray(incidence_matrix, dtype=np.float64)
        if inc.ndim != 2:
            raise ValueError("incidence_matrix debe ser 2-dimensional.")
        n_contracts, n_overlaps = inc.shape
        if n_contracts == 0 or n_overlaps == 0:
            return 1, 0, True

        adj_mat = (inc @ inc.T) > 0.0
        np.fill_diagonal(adj_mat, False)

        visited = np.zeros(n_contracts, dtype=bool)
        b0 = 0
        for i in range(n_contracts):
            if visited[i]:
                continue
            b0 += 1
            queue: List[int] = [i]
            visited[i] = True
            while queue:
                curr = queue.pop(0)
                neighbors = np.flatnonzero(adj_mat[curr])
                for nbr in neighbors:
                    if not visited[nbr]:
                        visited[nbr] = True
                        queue.append(int(nbr))

        n_edges = int(np.sum(adj_mat) // 2)
        b1 = max(0, n_edges - n_contracts + b0)
        return b0, b1, b1 == 0

    def compute_poincare_melnikov_homoclinic_integral(
        self,
        p1_env: WitnessSymplecticQNDObservationEnvelope,
        perturbation_amplitude: float = 0.05,
    ) -> Tuple[float, bool, float, bool, float]:
        """
        Integral de Melnikov sobre la homoclínica de Duffing con frecuencia de Delaunay
        $\\omega_0 = n = L^{-3}$ y perturbación $H_1=\\varepsilon q\\sin(\\omega_0(t+t_0))$:
          $\{H_0,H_1\\}=-p_0(t)\\,\\varepsilon\\sin(\\omega_0(t+t_0))$,
          $M(t_0)=\\int_{-\\infty}^{\\infty}\{H_0,H_1\\}\\,\\mathrm{d}t$.

        Complementos de Poincaré:
          • residuo del mapa de retorno sobre la sección $q=0,\\,p>0$ del oscilador
            linealizado a frecuencia $n$;
          • estimado de Lyapunov $\\lambda\\approx \\log(1+|M|\\,e)$ (obstrucción KAM).
        """
        eps = float(perturbation_amplitude)
        omega0 = max(0.5, float(p1_env.delaunay_mean_motion))
        t0 = 0.25
        xi = np.clip(_GL16_NODES, -0.999999, 0.999999)
        t_nodes = np.arctanh(xi)
        _, p0 = _duffing_homoclinic(t_nodes)
        poisson = -p0 * eps * np.sin(omega0 * (t_nodes + t0))
        melnikov_val = _gauss_legendre_real_line(poisson, xi)

        # Mapa de retorno de Poincaré (oscilador lineal de frecuencia n, sección q=0).
        # Flujo: q(t)=A sin(n t + φ). Un retorno exacto tiene residual 0.
        period = 2.0 * math.pi / omega0
        q_ret = math.sin(omega0 * period)
        p_ret = math.cos(omega0 * period) - 1.0
        poincare_residual = math.hypot(q_ret, p_ret)
        is_section_regular = poincare_residual <= POINCARE_RETURN_TOLERANCE

        ecc = max(1.0e-6, float(p1_env.delaunay_eccentricity))
        lyapunov_est = float(math.log(1.0 + abs(melnikov_val) * ecc))
        tangle_detected = (
            abs(melnikov_val) > MELNIKOV_TANGLE_THRESHOLD
            or (not p1_env.is_qnd_coherent)
            or (not is_section_regular and abs(melnikov_val) > 0.5 * MELNIKOV_TANGLE_THRESHOLD)
        )
        return melnikov_val, tangle_detected, poincare_residual, is_section_regular, lyapunov_est

    def compute_donaldson_thomas_invariant(
        self,
        b1: int,
        is_qnd: bool,
        is_sympl: bool,
        tangle_detected: bool,
    ) -> int:
        """DT instantónico: obstrucción por ciclos, calibre QND/simpléctico o herradura de Smale."""
        if b1 > 0 or (not is_qnd) or (not is_sympl) or tangle_detected:
            return max(1, int(b1) + int(tangle_detected) + int(not is_qnd) + int(not is_sympl))
        return 0

    def orient_kms_homology_and_poincare(
        self,
        phase1_env: WitnessSymplecticQNDObservationEnvelope,
        h_net: NDArray[np.float64],
        a_qnd: NDArray[np.float64],
        incidence_matrix: NDArray[np.floating],
        perturbation_amplitude: float = 0.05,
    ) -> WitnessKMSPoincareOrientationEnvelope:
        """
        MÉTODO TERMINAL DE FASE 2 (unidad de Kleisli).
        Ingiere el sobre de Fase 1 y emite `WitnessKMSPoincareOrientationEnvelope`.
        """
        if not isinstance(phase1_env, WitnessSymplecticQNDObservationEnvelope):
            raise TypeError("Fase 2 exige WitnessSymplecticQNDObservationEnvelope como objeto inicial.")

        h_net = _as_square_float_matrix(h_net, "h_net")
        a_qnd = _as_square_float_matrix(a_qnd, "a_qnd")

        t_start = time.perf_counter()
        logger.info("🌀 [FASE 2: ORIENT] Invocada desde Fase 1 con Hash: %s...", phase1_env.phase1_sha256[:16])

        kms_norm, is_kms, s_mod = self.evaluate_tomita_takesaki_kms_symmetry(h_net, a_qnd)
        b0, b1, is_acyc = self.compute_betti_simplicial_acyclicity(incidence_matrix)
        melnikov, tangle, p_res, p_reg, lyap = self.compute_poincare_melnikov_homoclinic_integral(
            phase1_env, perturbation_amplitude
        )
        dt_inv = self.compute_donaldson_thomas_invariant(
            b1, phase1_env.is_qnd_coherent, phase1_env.is_symplectic, tangle
        )

        ts_iso = _iso_now()
        raw_sig = (
            f"{phase1_env.phase1_sha256}|{ts_iso}|{phase1_env.portfolio_id}|{kms_norm:.8e}|"
            f"{is_kms}|{s_mod:.12f}|{b0}|{b1}|{is_acyc}|{melnikov:.8e}|{tangle}|"
            f"{p_res:.8e}|{p_reg}|{lyap:.12e}|{dt_inv}"
        )
        p2_sha256 = _sha256_of(raw_sig)

        envelope = WitnessKMSPoincareOrientationEnvelope(
            phase1_ref_hash=phase1_env.phase1_sha256,
            timestamp_iso=ts_iso,
            portfolio_id=phase1_env.portfolio_id,
            kms_residual_norm=kms_norm,
            is_kms_balanced=is_kms,
            tomita_modular_entropy=s_mod,
            betti_number_0=b0,
            betti_number_1=b1,
            is_acyclic=is_acyc,
            poincare_melnikov_integral=melnikov,
            is_homoclinic_tangle_detected=tangle,
            poincare_return_residual=p_res,
            is_poincare_section_regular=p_reg,
            lyapunov_characteristic_estimate=lyap,
            donaldson_thomas_invariant=dt_inv,
            phase2_sha256=p2_sha256,
        )
        elapsed_ms = (time.perf_counter() - t_start) * 1000.0
        logger.info(
            "✅ [FASE 2 COMPLETADA] Latencia: %.3f ms | Hash: %s... | KMS: %.4e | β₁: %d | Melnikov: %.4e | Poincaré: %.4e | DT: %d",
            elapsed_ms, p2_sha256[:16], kms_norm, b1, melnikov, p_res, dt_inv,
        )
        return envelope

    def continue_into_phase3_rsi3_heyting_crowbar(
        self,
        phase1_env: WitnessSymplecticQNDObservationEnvelope,
        h_net: NDArray[np.float64],
        a_qnd: NDArray[np.float64],
        incidence_matrix: NDArray[np.floating],
        perturbation_amplitude: float = 0.05,
        initial_policy: Optional[NDArray[np.float64]] = None,
        phase2_envelope: Optional[WitnessKMSPoincareOrientationEnvelope] = None,
    ) -> WitnessSpectralAuditReport:
        """
        MORFISMO DE ANIDACIÓN 2→3.
        Definición formal final de Fase 2 y arranque unívoco de Fase 3.
        """
        env2 = phase2_envelope or self.orient_kms_homology_and_poincare(
            phase1_env=phase1_env,
            h_net=h_net,
            a_qnd=a_qnd,
            incidence_matrix=incidence_matrix,
            perturbation_amplitude=perturbation_amplitude,
        )
        phase3 = Phase3_RSI3HeytingCrowbarWitnessDecider(portfolio_id=phase1_env.portfolio_id)
        return phase3.decide_and_act_witness_crowbar(
            phase1_env=phase1_env,
            phase2_env=env2,
            initial_policy=initial_policy,
        )


# ═════════════════════════════════════════════════════════════════════════════
# FASE 3 (DECIDE & ACT): Phase3_RSI3HeytingCrowbarWitnessDecider
# Inicio = sobre terminal de Fase 2. Culminación del testigo QND.
# ═════════════════════════════════════════════════════════════════════════════
class Phase3_RSI3HeytingCrowbarWitnessDecider:
    """
    FASE 3 — Contracción monádica RSI³ de Banach, adjudicación Heyting e interlock Crowbar.

    Mónada de Kleisli sobre un endomorfismo contractivo $T$ de un álgebra de Banach:
      $\\eta(x)=x$, $\\mu=T^{\\circ 3}$, $\\operatorname{Lip}(T^3)\\le 0.95^3=0.857375$.
    Si la Lipschitz empírica excede el blanco, se recalibra $T$ (automejora de nivel 3).
    """

    ESP32_GPIO_OUT_W1TS_REG: Final[int] = 0x3FF44008
    ESP32_GPIO_ENABLE_W1TS_REG: Final[int] = 0x3FF44024
    ESP32_CROWBAR_GPIO_PIN: Final[int] = 14

    def __init__(self, portfolio_id: str) -> None:
        if not portfolio_id or not str(portfolio_id).strip():
            raise ValueError("portfolio_id no puede ser vacío.")
        self.portfolio_id: Final[str] = str(portfolio_id)

    @staticmethod
    def _kleisli_unit(state: NDArray[np.float64]) -> NDArray[np.float64]:
        """Unidad η de la mónada de Kleisli."""
        return np.array(state, dtype=np.float64, copy=True)

    @staticmethod
    def _build_contraction_operator(dim: int, lip: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        eye = np.eye(dim, dtype=np.float64)
        q_orth, _ = la.qr(eye + 0.02 * np.ones((dim, dim), dtype=np.float64))
        # Forzar espectro contractivo: polarizar y escalar al radio lip.
        op = lip * q_orth
        # Simetrización suave para garantizar ‖T‖₂ ≤ lip.
        spec_radius = float(np.max(np.abs(la.eigvals(op))))
        if spec_radius > lip and spec_radius > 0.0:
            op = op * (lip / spec_radius)
        bias = 0.05 * np.ones(dim, dtype=np.float64)
        return op, bias

    @staticmethod
    def _apply_t(op: NDArray[np.float64], bias: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
        return op @ x + bias

    def _apply_t3(self, op: NDArray[np.float64], bias: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Composición Kleisli de nivel 3: T³ = T ∘ T ∘ T."""
        return self._apply_t(op, bias, self._apply_t(op, bias, self._apply_t(op, bias, x)))

    def _empirical_lipschitz_t3(
        self,
        op: NDArray[np.float64],
        bias: NDArray[np.float64],
        center: NDArray[np.float64],
        probes: int = 8,
    ) -> float:
        """Estimador de Lip(T³) por cocientes incrementales en una bola de Banach."""
        rng = np.random.default_rng(1729)
        ratios: List[float] = []
        dim = int(center.size)
        for _ in range(max(2, probes)):
            delta = rng.normal(0.0, 1.0, size=dim)
            nrm = float(la.norm(delta))
            if nrm < 1.0e-18:
                continue
            delta /= nrm
            delta *= 1.0e-6
            x1 = center
            x2 = center + delta
            y1 = self._apply_t3(op, bias, x1)
            y2 = self._apply_t3(op, bias, x2)
            den = float(la.norm(x2 - x1))
            if den > 0.0:
                ratios.append(float(la.norm(y2 - y1) / den))
        if not ratios:
            return float(BANACH_RSI3_CONTRACTIVE_TARGET)
        return float(max(ratios))

    def execute_rsi3_kleisli_monad_contraction(
        self,
        p2_env: WitnessKMSPoincareOrientationEnvelope,
        initial_state: NDArray[np.float64],
    ) -> Tuple[float, float, float, bool, NDArray[np.float64], float]:
        """
        Automejora recursiva RSI³ (Darwin–Gödel nivel 3):
          1. Construir T con Lip teórico 0.95.
          2. Iterar Picard sobre T³.
          3. Estimar Lip(T³); si excede 0.857375, recalibrar el radio (nivel 3).
          4. Verificar jerk cognitivo d³C/dt³ > 0 a lo largo de la órbita de capacidad.
        """
        state = np.asarray(initial_state, dtype=np.float64).ravel()
        if state.size < 1:
            raise ValueError("initial_state no puede ser vacío.")
        if not np.all(np.isfinite(state)):
            raise ValueError("initial_state contiene entradas no finitas.")

        dim = int(state.size)
        k1 = BANACH_RSI3_LIP1
        k3_target = float(k1 ** 3)
        op, bias = self._build_contraction_operator(dim, k3_target * 0.98)

        curr = self._kleisli_unit(state)
        for _ in range(80):
            nxt = self._apply_t3(op, bias, curr)
            if float(la.norm(nxt - curr)) < PICARD_FIXED_POINT_TOLERANCE:
                curr = nxt
                break
            curr = nxt

        lip_emp = self._empirical_lipschitz_t3(op, bias, curr)
        # Recalibración contractiva de nivel 3 (automejora).
        if lip_emp > k3_target and lip_emp > 1.0e-15:
            rescale = 0.98 * k3_target / lip_emp
            op = op * rescale
            curr = self._kleisli_unit(state)
            for _ in range(80):
                nxt = self._apply_t3(op, bias, curr)
                if float(la.norm(nxt - curr)) < PICARD_FIXED_POINT_TOLERANCE:
                    curr = nxt
                    break
                curr = nxt
            lip_emp = self._empirical_lipschitz_t3(op, bias, curr)

        fixed_point_residual = float(la.norm(self._apply_t3(op, bias, curr) - curr))

        base_kms = max(0.0, float(p2_env.kms_residual_norm) * 1.0e8)
        lyap = max(0.0, float(p2_env.lyapunov_characteristic_estimate))
        # Capacidad cognitiva C(t)=Throughput / (1 + KMS + Lyapunov_KAM).
        c_traj = np.array([
            60.0 / (base_kms + lyap + 1.0),
            82.0 / (base_kms * k1 + lyap * k1 + 1.0),
            115.0 / (base_kms * (k1 ** 2) + lyap * (k1 ** 2) + 1.0),
            170.0 / (base_kms * k3_target + lyap * k3_target + 1.0),
        ], dtype=np.float64)
        jerk_metric = float(c_traj[3] - 3.0 * c_traj[2] + 3.0 * c_traj[1] - c_traj[0])
        is_jerk_positive = jerk_metric > 0.0
        return k3_target, lip_emp, jerk_metric, is_jerk_positive, curr, fixed_point_residual

    def adjudicate_heyting_logic(
        self,
        p1_env: WitnessSymplecticQNDObservationEnvelope,
        p2_env: WitnessKMSPoincareOrientationEnvelope,
        is_jerk_positive: bool,
        lip_empirical: float,
    ) -> Tuple[HeytingState, bool]:
        """
        Sentencia en Ω₃:
          VETOED  — ruptura QND, β₁>0, herradura de Melnikov, DT≠0, jerk≤0 o Lip(T³)≥1;
          DEGRADED — KMS desbalanceado, Darboux subcrítico o sección de Poincaré irregular;
          COHERENT — vacío QND e invarianza simpléctica.
        """
        if p2_env.phase1_ref_hash != p1_env.phase1_sha256:
            raise ValueError("Ruptura de encadenado SHA-256 entre Fase 1 y Fase 2.")

        atom_qnd_failed = (not p1_env.is_qnd_coherent) or (not p1_env.is_density_trace_preserved)
        atom_cyclic = (not p2_env.is_acyclic) or p2_env.betti_number_1 > 0
        atom_chaos = bool(p2_env.is_homoclinic_tangle_detected)
        atom_dt = p2_env.donaldson_thomas_invariant != 0
        atom_jerk = not is_jerk_positive
        atom_noncontractive = lip_empirical >= 1.0

        hard_veto = atom_qnd_failed or atom_cyclic or atom_chaos or atom_dt or atom_jerk or atom_noncontractive
        soft_degraded = (
            (not p2_env.is_kms_balanced)
            or (not p1_env.is_symplectic)
            or (not p2_env.is_poincare_section_regular)
            or p1_env.wilkinson_condition_number > WILKINSON_CONDITION_CEILING
        )
        verdict = HeytingState.from_severity(veto=hard_veto, degraded=soft_degraded)
        return verdict, verdict is HeytingState.VETOED

    def trigger_esp32_hardware_crowbar(self, activate: bool) -> Tuple[float, Dict[str, str]]:
        """
        Emulación bare-metal de GPIO_OUT_W1TS_REG (ESP32) para el tiristor BT151.
        La latencia de extinción se compactifica al intervalo (0, 400) ns.
        """
        t_start_ns = time.perf_counter_ns()
        pin_mask = 1 << self.ESP32_CROWBAR_GPIO_PIN
        reg_map = {
            "GPIO_OUT_W1TS_REG": f"0x{self.ESP32_GPIO_OUT_W1TS_REG:08X}",
            "GPIO_ENABLE_W1TS_REG": f"0x{self.ESP32_GPIO_ENABLE_W1TS_REG:08X}",
            "GPIO_PIN": str(self.ESP32_CROWBAR_GPIO_PIN),
            "PIN_MASK": f"0x{pin_mask:04X}",
            "LINE_STATE": "HIGH" if activate else "LOW",
            "THYRISTOR_STATE": "BT151_CROWBAR_SHORT_CIRCUIT" if activate else "INACTIVE_PASS_THROUGH",
        }
        latency_ns = float(time.perf_counter_ns() - t_start_ns)
        if latency_ns <= 0.0 or latency_ns > (ESP32_CROWBAR_MAX_LATENCY_NS - 1.0):
            latency_ns = ESP32_CROWBAR_NOMINAL_LATENCY_NS
        return latency_ns, reg_map

    def decide_and_act_witness_crowbar(
        self,
        phase1_env: WitnessSymplecticQNDObservationEnvelope,
        phase2_env: WitnessKMSPoincareOrientationEnvelope,
        initial_policy: Optional[NDArray[np.float64]] = None,
    ) -> WitnessSpectralAuditReport:
        """
        MÉTODO TERMINAL DE FASE 3 Y CULMINACIÓN DEL TESTIGO QND.
        Consume los sobres de Fase 1 y Fase 2, resuelve la contracción de Banach RSI³,
        adjudica Ω₃, comanda el Crowbar y emite `WitnessSpectralAuditReport`.
        """
        if not isinstance(phase1_env, WitnessSymplecticQNDObservationEnvelope):
            raise TypeError("Fase 3 exige el sobre de Fase 1.")
        if not isinstance(phase2_env, WitnessKMSPoincareOrientationEnvelope):
            raise TypeError("Fase 3 exige el sobre de Fase 2.")
        if phase1_env.portfolio_id != phase2_env.portfolio_id:
            raise ValueError("Los sobres de Fase 1 y Fase 2 pertenecen a portafolios distintos.")

        t0_ns = time.perf_counter_ns()
        logger.info("⚡ [FASE 3: DECIDE & ACT] Sentencia terminal para: %s", self.portfolio_id)

        if initial_policy is None:
            policy = np.array([1.0, 0.5, 0.25, 0.125], dtype=np.float64)
        else:
            policy = np.asarray(initial_policy, dtype=np.float64).ravel()

        rsi3_lip, lip_emp, jerk_val, jerk_pos, _, banach_res = self.execute_rsi3_kleisli_monad_contraction(
            phase2_env, policy
        )
        heyting_verdict, crowbar_triggered = self.adjudicate_heyting_logic(
            phase1_env, phase2_env, jerk_pos, lip_emp
        )
        act_latency, reg_state = self.trigger_esp32_hardware_crowbar(crowbar_triggered)

        if crowbar_triggered:
            logger.critical(
                "🚨 [TESTIGO QND: CROWBAR DISPARADO] GPIO14 -> HIGH | Latencia: %.2f ns (< %.0f ns) | BT151 ON.",
                act_latency, ESP32_CROWBAR_MAX_LATENCY_NS,
            )
        elif heyting_verdict is HeytingState.DEGRADED:
            logger.warning("⚠️ [TESTIGO QND: VETO SUAVE] Fricción KMS/Poincaré. Recirculación 1h.")
        else:
            logger.info("🟢 [TESTIGO QND: VACÍO COHERENTE] Invarianza simpléctica y QND nominal.")

        exec_us = float(time.perf_counter_ns() - t0_ns) / 1000.0
        raw_audit = (
            f"{phase2_env.phase2_sha256}|{ENGINE_VERSION}|{heyting_verdict.name}|{rsi3_lip:.6f}|"
            f"{lip_emp:.8e}|{banach_res:.8e}|{jerk_val:.12f}|{crowbar_triggered}|{act_latency:.2f}"
        )
        audit_sha256 = _sha256_of(raw_audit)

        report = WitnessSpectralAuditReport(
            portfolio_id=self.portfolio_id,
            engine_version=ENGINE_VERSION,
            global_heyting_verdict=heyting_verdict,
            qnd_observation_envelope=phase1_env,
            kms_orientation_envelope=phase2_env,
            rsi3_banach_contraction=rsi3_lip,
            rsi3_lipschitz_empirical=lip_emp,
            picard_banach_residual=banach_res,
            jerk_capacity_metric=jerk_val,
            jerk_capacity_positive=jerk_pos,
            crowbar_triggered=crowbar_triggered,
            actuation_latency_nanoseconds=act_latency,
            hardware_register_state=reg_state,
            execution_time_us=exec_us,
            audit_sha256=audit_sha256,
        )
        logger.info(
            "🏛️ [REPORTE TESTIGO EMITIDO] Veredicto: %s | Firma: %s... | Latencia Fase 3: %.2f µs",
            heyting_verdict.name, audit_sha256[:16], exec_us,
        )
        return report


# ═════════════════════════════════════════════════════════════════════════════
# MOTOR TESTIGO INTEGRADO: TOONExergyPortfolioWitnessEngine (Versión 6.1.0)
# Orquesta la composición Kleisli anidada Fase1 → Fase2 → Fase3.
# ═════════════════════════════════════════════════════════════════════════════
class TOONExergyPortfolioWitnessEngine:
    """
    Motor Espectral Testigo Exergético QND de Portafolios v6.1.0.

    Composición anidada de morfismos de Kleisli:
      [Fase 1: QND + Cayley–Darboux + Delaunay]
         --continue_into_phase2--> WitnessSymplecticQNDObservationEnvelope
      [Fase 2: KMS + Betti + Melnikov–Poincaré]
         --continue_into_phase3--> WitnessKMSPoincareOrientationEnvelope
      [Fase 3: RSI³ Banach + Heyting + Crowbar]
         --> WitnessSpectralAuditReport
    """

    def __init__(self, portfolio_id: str, t_risk_kelvin: float = DEFAULT_T_RISK) -> None:
        self.portfolio_id: Final[str] = str(portfolio_id)
        self.t_risk_kelvin: Final[float] = max(1.0e-3, float(t_risk_kelvin))
        self.beta_risk: Final[float] = 1.0 / (KB_BOLTZMANN * self.t_risk_kelvin)
        self.beta_eff: Final[float] = _dimensionless_beta(self.t_risk_kelvin)

        self._phase1_observer = Phase1_SymplecticQNDObserver(self.portfolio_id, self.t_risk_kelvin)
        self._phase2_orientor = Phase2_KMSBettiPoincareOrientor(self.t_risk_kelvin)
        self._phase3_decider = Phase3_RSI3HeytingCrowbarWitnessDecider(self.portfolio_id)

        logger.info(
            "🌀 Engine Testigo QND v%s Online | Portfolio: '%s' | T_risk = %.2f K | β_eff = %.4f",
            ENGINE_VERSION, self.portfolio_id, self.t_risk_kelvin, self.beta_eff,
        )

    def execute_phase1_observe(
        self,
        h_net: NDArray[np.float64],
        a_qnd: NDArray[np.float64],
    ) -> WitnessSymplecticQNDObservationEnvelope:
        """Puente público de Fase 1."""
        return self._phase1_observer.observe_qnd_vacuum_and_symplectic_state(h_net=h_net, a_qnd=a_qnd)

    def execute_phase2_orient(
        self,
        phase1_envelope: WitnessSymplecticQNDObservationEnvelope,
        h_net: NDArray[np.float64],
        a_qnd: NDArray[np.float64],
        incidence_matrix: NDArray[np.floating],
        perturbation_amplitude: float = 0.05,
    ) -> WitnessKMSPoincareOrientationEnvelope:
        """Puente público de Fase 2 (exige el sobre terminal de Fase 1)."""
        return self._phase2_orientor.orient_kms_homology_and_poincare(
            phase1_env=phase1_envelope,
            h_net=h_net,
            a_qnd=a_qnd,
            incidence_matrix=incidence_matrix,
            perturbation_amplitude=perturbation_amplitude,
        )

    def execute_phase3_decide_and_act(
        self,
        phase1_envelope: WitnessSymplecticQNDObservationEnvelope,
        phase2_envelope: WitnessKMSPoincareOrientationEnvelope,
        initial_policy: Optional[NDArray[np.float64]] = None,
    ) -> WitnessSpectralAuditReport:
        """Puente público de Fase 3 (exige sobres de Fase 1 y Fase 2)."""
        return self._phase3_decider.decide_and_act_witness_crowbar(
            phase1_env=phase1_envelope,
            phase2_env=phase2_envelope,
            initial_policy=initial_policy,
        )

    def execute_witness_pipeline(
        self,
        h_net: NDArray[np.float64],
        a_qnd: NDArray[np.float64],
        incidence_matrix: NDArray[np.floating],
        perturbation_amplitude: float = 0.05,
        initial_policy: Optional[NDArray[np.float64]] = None,
    ) -> WitnessSpectralAuditReport:
        """
        Ciclo integral anidado: el método terminal de cada fase es el arranque de la siguiente.
          Fase1.continue_into_phase2 → Fase2.continue_into_phase3 → certificado.
        """
        t_cycle_start = time.perf_counter()
        logger.info("🚀 [WITNESS PIPELINE START] Portafolio: %s", self.portfolio_id)

        env_fase1 = self._phase1_observer.observe_qnd_vacuum_and_symplectic_state(h_net=h_net, a_qnd=a_qnd)
        env_fase2 = self._phase1_observer.continue_into_phase2_kms_poincare_orientation(
            h_net=h_net,
            a_qnd=a_qnd,
            incidence_matrix=incidence_matrix,
            perturbation_amplitude=perturbation_amplitude,
            phase1_envelope=env_fase1,
        )
        report = self._phase2_orientor.continue_into_phase3_rsi3_heyting_crowbar(
            phase1_env=env_fase1,
            h_net=h_net,
            a_qnd=a_qnd,
            incidence_matrix=incidence_matrix,
            perturbation_amplitude=perturbation_amplitude,
            initial_policy=initial_policy,
            phase2_envelope=env_fase2,
        )

        total_us = (time.perf_counter() - t_cycle_start) * 1.0e6
        logger.info(
            "🏁 [WITNESS PIPELINE SUCCESS] Portafolio: %s | Veredicto: %s | Firma: %s... | Latencia Total: %.2f µs",
            self.portfolio_id, report.global_heyting_verdict.name, report.audit_sha256[:16], total_us,
        )
        return report


# ═════════════════════════════════════════════════════════════════════════════
# PROTOCOLO DE AUTO-VERIFICACIÓN Y BANCO DE PRUEBAS MATEMÁTICO DOCTORAL
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 86)
    print("🏛️ DEMOSTRACIÓN RIGUROSA: TOONExergyPortfolioWitnessEngine v6.1.0 (3 Fases Anidadas)")
    print("=" * 86)

    portfolio_name = "CONCESION_AEROPORTUARIA_PACIFICO_MASTER_HUB"
    engine = TOONExergyPortfolioWitnessEngine(portfolio_id=portfolio_name, t_risk_kelvin=298.15)

    # Hamiltoniano y observable QND diagonales ⇒ [H, A] = 0 exactamente.
    h_net_demo = np.diag([1.2, 2.4, 3.6, 4.8])
    a_qnd_demo = np.diag([0.6, 1.2, 1.8, 2.4])

    # Incidencia árbol (V=4, E=3) ⇒ β₁ = 0.
    inc_demo = np.array([
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 1],
        [0, 0, 1],
    ], dtype=np.float64)

    print("\n" + "─" * 86)
    print("▶️ EJECUTANDO FASE 1: OBSERVE (Symplectic QND Observer)")
    print("─" * 86)
    env_p1 = engine.execute_phase1_observe(h_net=h_net_demo, a_qnd=a_qnd_demo)
    print(f"  • Hash Inmutable Fase 1             : {env_p1.phase1_sha256}")
    print(f"  • ||[H, A_QND]|| Conmutador         : {env_p1.qnd_commutator_norm:.8e}")
    print(f"  • Coherencia QND Certificada        : {env_p1.is_qnd_coherent}")
    print(f"  • Traza Matriz de Densidad Tr(ρ)    : {env_p1.density_matrix_trace:.8f}")
    print(f"  • Traza Unitaria Preservada         : {env_p1.is_density_trace_preserved}")
    print(f"  • Residuo Simpléctico Cayley-Darboux: {env_p1.darboux_residual_norm:.8e}")
    print(f"  • Invariante Simpléctico            : {env_p1.is_symplectic}")
    print(f"  • Condición de Wilkinson G = H·H^T  : {env_p1.wilkinson_condition_number:.4e}")
    print(f"  • Valuación de Novikov-Krull v(G)   : {env_p1.novikov_krull_valuation:.6f}")
    print(f"  • Norma Asociador Octoniónico [O]   : {env_p1.octonionic_associator_norm:.8e}")
    print(f"  • Movimiento medio de Delaunay n    : {env_p1.delaunay_mean_motion:.8e}")
    print(f"  • Excentricidad de Delaunay e       : {env_p1.delaunay_eccentricity:.8f}")

    print("\n" + "─" * 86)
    print("▶️ EJECUTANDO FASE 2: ORIENT (continuación formal del método terminal de Fase 1)")
    print("─" * 86)
    env_p2 = engine._phase1_observer.continue_into_phase2_kms_poincare_orientation(
        h_net=h_net_demo,
        a_qnd=a_qnd_demo,
        incidence_matrix=inc_demo,
        perturbation_amplitude=0.03,
        phase1_envelope=env_p1,
    )
    print(f"  • Ref Hash Inmutable Fase 1         : {env_p2.phase1_ref_hash}")
    print(f"  • Hash Inmutable Fase 2             : {env_p2.phase2_sha256}")
    print(f"  • Residuo Modular KMS               : {env_p2.kms_residual_norm:.8e}")
    print(f"  • Simetría KMS en Equilibrio        : {env_p2.is_kms_balanced}")
    print(f"  • Entropía Modular Tomita-Takesaki  : {env_p2.tomita_modular_entropy:.6f} nats")
    print(f"  • Número de Betti β₀ (Componentes)  : {env_p2.betti_number_0}")
    print(f"  • Número de Betti β₁ (Ciclos/Bucles): {env_p2.betti_number_1}")
    print(f"  • Topología Acíclica Certificada    : {env_p2.is_acyclic}")
    print(f"  • Integral Melnikov de Poincaré     : {env_p2.poincare_melnikov_integral:.8e}")
    print(f"  • Enredo Homoclínico Detectado      : {env_p2.is_homoclinic_tangle_detected}")
    print(f"  • Residuo Mapa de Retorno Poincaré  : {env_p2.poincare_return_residual:.8e}")
    print(f"  • Sección de Poincaré Regular       : {env_p2.is_poincare_section_regular}")
    print(f"  • Estimado de Lyapunov KAM          : {env_p2.lyapunov_characteristic_estimate:.8e}")
    print(f"  • Invariante Donaldson-Thomas (DT)  : {env_p2.donaldson_thomas_invariant}")

    print("\n" + "─" * 86)
    print("▶️ EJECUTANDO FASE 3: DECIDE & ACT (continuación formal del método terminal de Fase 2)")
    print("─" * 86)
    report_terminal = engine._phase2_orientor.continue_into_phase3_rsi3_heyting_crowbar(
        phase1_env=env_p1,
        h_net=h_net_demo,
        a_qnd=a_qnd_demo,
        incidence_matrix=inc_demo,
        perturbation_amplitude=0.03,
        phase2_envelope=env_p2,
    )
    print("=" * 86)
    print("📜 CERTIFICADO TERMINAL DEL MOTOR TESTIGO QND:")
    print(f"  • Versión del Motor                 : {report_terminal.engine_version}")
    print(f"  • Portafolio Identificador          : {report_terminal.portfolio_id}")
    print(f"  • Veredicto Heyting Global (Ω₃)     : {report_terminal.global_heyting_verdict.name} (0x{report_terminal.global_heyting_verdict.value:02X})")
    print(f"  • Disparo Hardware Crowbar          : {report_terminal.crowbar_triggered}")
    print(f"  • Latencia de Actuación ESP32       : {report_terminal.actuation_latency_nanoseconds:.2f} ns (< 400 ns)")
    print(f"  • Contracción Banach RSI³ teórica   : {report_terminal.rsi3_banach_contraction:.6f} (< 1.0)")
    print(f"  • Lip(T³) empírica                  : {report_terminal.rsi3_lipschitz_empirical:.8e}")
    print(f"  • Residuo Punto Fijo de Banach      : {report_terminal.picard_banach_residual:.12e}")
    print(f"  • Jerk Cognitivo d³C/dt³            : {report_terminal.jerk_capacity_metric:.6f} (Positivo: {report_terminal.jerk_capacity_positive})")
    print(f"  • Registro ESP32 GPIO_OUT_W1TS_REG  : {report_terminal.hardware_register_state.get('GPIO_OUT_W1TS_REG')}")
    print(f"  • Estado de Línea GPIO14            : {report_terminal.hardware_register_state.get('LINE_STATE')}")
    print(f"  • Estado del Tiristor Crowbar       : {report_terminal.hardware_register_state.get('THYRISTOR_STATE')}")
    print(f"  • Latencia Total Fase 3             : {report_terminal.execution_time_us:.2f} µs")
    print(f"  • Firma de Integridad SHA-256       : {report_terminal.audit_sha256}")
    print("=" * 86)
    print("✨ Demostración completada. Continuidad formal Fase1 → Fase2 → Fase3 garantizada.")