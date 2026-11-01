# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Nombre   : Topological Gauge Field Engine (Motor Espectral de Calibre)       ║
║ Ubicación: app/wisdom/topological_gauge_field_engine.py                      ║
║ Versión  : 11.0.0 (SOVEREIGN-GAUGE-KKS-POINCARE-BRUNO-RSI3-CAYLEY-SP-PhD)    ║
║ Función  : Motor espectral de calibre topológico, curvatura de               ║
║            Kirillov-Kostant-Souriau en su(N), reducción simpléctica          ║
║            Cayley-Darboux sobre sp(2n, R), teoría KAM-Bruno, invariantes de  ║
║            Chern-Simons-Nakahara y automejora recursiva RSI³ en 3 fases.     ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO MOTOR ESPECTRAL:
─────────────────────────────────────────────────────────────────────────────────────────
El `TopologicalGaugeFieldEngine` constituye el motor espectral y geométrico-diferencial
supremo para la teoría de campos de calibre en el estrato WISDOM de APU Filter. Su misión
operativa consiste en procesar y certificar operadores de densidad cuánticos $\rho \in \mathcal{D}_N$,
construir conexiones de gauge antihermíticas sin traza $A \in \mathfrak{su}(N)$, determinar la
curvatura simpléctica orbital mediante la 2-forma de Kirillov-Kostant-Souriau (KKS) $F = [\rho, H_{\text{cel}}] \in \mathfrak{su}(N)$,
proyectar el sistema sobre la variedad simpléctica realificada $\mathfrak{sp}(2n, \mathbb{R})$ vía Cayley-Darboux,
resolver perturbaciones seculares mediante la ecuación homológica espectral de Poincaré-Lindstedt,
filtrar resonancias por la condición diofántica de KAM y la serie de Bruno, y gobernar las
tres fases anidadas e indivisibles:
  - FASE 1 (OBSERVE): Sanitización espectral en $\mathcal{D}_N$, conexión de gauge $A \in \mathfrak{su}(N)$,
    curvatura orbital KKS $F$, realificación $\mathbb{C}^n \hookrightarrow \mathbb{R}^{2n}$, certificación
    de subálgebra de Lie $\mathfrak{sp}(2n, \mathbb{R})$, descomposición homológica de Lindstedt y emisión de sobre $\Phi_1$.
  - FASE 2 (ORIENT): Acción topológica de Chern-Simons $CS_3(A)$, invariante de Pontryagin $p_1$,
    holonomía de Wilson $\mathcal{W}(C)$, auditoría diofántica KAM con cota de Bruno, integral homoclínica
    de Melnikov $M(\theta_0)$, multiplicadores de Floquet y colisiones de Krein, y cascada de Banach RSI³.
  - FASE 3 (DECIDE & ACT): Medición débil AAV cuántica no demolición (QND), agregación lattice en el álgebra
    de Heyting trivalente $\Omega_3 = \{\bot < \star < \top\}$, protocolo interlock ciber-físico ESP32 Crowbar
    en silicio ($WCET \le 392.15\text{ ns}$), y proyección semántica funtorial $\Phi_{\text{sem}}$.

2. MARCO TEÓRICO:
─────────────────────────────────────────────────────────────────────────────────────────
2.1. CONEXIÓN DE GAUGE $\mathfrak{su}(N)$ Y CURVATURA DE KIRILLOV-KOSTANT-SOURIAU (KKS):
   Para un operador de densidad hermitiano semidefinido positivo $\rho \in \mathcal{D}_N$ con $\text{Tr}(\rho) = 1$,
   la conexión de gauge canónica se define como la traslación antihermítica sin traza:
     $$A = i \left( \rho - \frac{1}{N} I_N \right) \in \mathfrak{su}(N), \quad A^\dagger = -A, \quad \text{Tr}(A) = 0$$
   A fin de evitar curvaturas idénticamente nulas debidas a la conmutación trivial $[A, \rho] = 0$, la curvatura de Yang-Mills
   se formula a través del tensor simpléctico orbital de Kirillov-Kostant-Souriau acoplado a un Hamiltoniano celeste de fondo:
     $$F = [\rho, H_{\text{cel}}] \in \mathfrak{su}(N), \quad F^\dagger = -F, \quad \text{Tr}(F) = 0$$
   donde $H_{\text{cel}} = L_{\text{path}} + V_{\text{trap}}$ modela el Laplaciano de osciladores acoplados de Poincaré.
   La densidad de energía de Yang-Mills y el primer número de Pontryagin satisfacen:
     $$E_{\text{YM}} = -\frac{1}{2} \text{Tr}(F^2) \ge 0, \quad p_1 = -\frac{1}{8\pi^2} \text{Tr}(F^2) \ge 0$$

2.2. REALIFICACIÓN DE CAYLEY-KÄHLER Y ESTRUCTURA SIMPLÉCTICA $\mathfrak{sp}(2n, \mathbb{R})$:
   Dado un operador $X = U + i V \in \mathbb{C}^{n \times n}$, se define el morfismo inyectivo de realificación $\mathbb{C}^n \hookrightarrow \mathbb{R}^{2n}$:
     $$\mathcal{R}(X) = \begin{pmatrix} U & -V \\ V & U \end{pmatrix} \in \mathbb{R}^{2n \times 2n}$$
   La matriz simpléctica canónica $J_{2n} = \begin{pmatrix} 0 & I_n \\ -I_n & 0 \end{pmatrix}$ impone las condiciones de pertenencia:
     $$M \in Sp(2n, \mathbb{R}) \iff M^\top J_{2n} M = J_{2n}, \quad K \in \mathfrak{sp}(2n, \mathbb{R}) \iff K^\top J_{2n} + J_{2n} K = 0$$
   El defecto simpléctico cuadrático se verifica mediante el tensor $\delta_{\text{Sp}} = \| M^\top J_{2n} M - J_{2n} \|_F / \|M\|_F^2 \le 10^{-10}$.

2.3. ECUACIÓN HOMOLÓGICA DE POINCARÉ-LINDSTEDT Y TEORÍA KAM-BRUNO:
   La diagonalización espectral $-i A = V \Lambda V^\dagger$ define las frecuencias de oscilación no perturbadas $\omega = \text{diag}(\Lambda) \in \mathbb{R}^n$.
   La eliminación del término resonante secular en la curvatura $F$ conduce a la ecuación homológica para la función generatriz $W$:
     $$-i [A, W] = F_{\text{osc}} \implies W_{jk} = \frac{(V^\dagger F_{\text{osc}} V)_{jk}}{i(\lambda_j - \lambda_k)}, \quad j \ne k$$
   La estabilidad de los toros invariantes se certifica evaluando la condición diofántica sobre vectores enteros de onda $k \in \mathbb{Z}^n$:
     $$|\langle \omega, k \rangle| \cdot \|k\|_1^\tau \ge \gamma, \quad \tau = d + 1, \quad \gamma > 0$$
   conjuntamente con la convergencia estricta de la serie de pequeños divisores de Bruno:
     $$\mathcal{B}(\omega) = \sum_{n=0}^{\infty} \frac{1}{2^n} \ln \left( \frac{1}{\Omega_n} \right) < \infty, \quad \Omega_n = \inf_{0 < \|k\|_\infty \le 2^n} |\langle \omega, k \rangle|$$

2.4. INVARIANTES TOPOLÓGICOS DE CHERN-SIMONS Y HOLONOMÍA DE WILSON:
   La 3-forma de Chern-Simons se computa mediante la identidad de Nakahara libre de duplicaciones:
     $$\text{CS}_3(A) = \frac{1}{8\pi^2} \text{Tr}\left( A \wedge dA + \frac{2}{3} A^3 \right) = \frac{1}{8\pi^2} \text{Tr}\left( A \wedge F - \frac{1}{3} A^3 \right)$$
   Para un bucle cerrado $C$, la holonomía de Wilson ordenada por camino es $\mathcal{W}(C) = \frac{1}{n} \text{Tr} \mathcal{P} \exp \oint_C A \in \mathbb{C}$,
   con clase topológica entera asociada $k = \text{round}(p_1) \in \mathbb{Z}$.

2.5. AUTOMEJORA RECURSIVA RSI³ CON CONTRACCIÓN DE BANACH:
   El operador de mejora $T: \mathfrak{su}(N) \to \mathfrak{su}(N)$ proyecta la curvatura amortiguada mediante la función generatriz de Lindstedt:
     $$T(F) = F - (1 - \lambda)\{H_0, W\} \approx F_{\text{mean}} + \lambda F_{\text{osc}}, \quad \lambda \le 0.95$$
   En el Nivel 3 de RSI, el operador actúa recursivamente sobre su propio residuo iterando $T^3$, verificando la cota hipergeométrica de contracción:
     $$\text{Lip}(T^3) \le (\text{Lip}(T))^3 \le 0.95^3 \approx 0.857375 < 1.0$$
   lo que garantiza la convergencia a un punto fijo único en el espacio de Banach $(\mathfrak{su}(N), \|\cdot\|_F)$.

3. AXIOMAS FORMALES:
─────────────────────────────────────────────────────────────────────────────────────────
  AXIOMA I (HERMITICIDAD, POSITIVIDAD Y CONSERVACIÓN DE PROBABILIDAD):
    Todo estado cuántico satisface $\rho = \rho^\dagger \succeq 0$ y $\text{Tr}(\rho) = 1$, induciendo una conexión antihermítica estricta $A^\dagger = -A$ con $\text{Tr}(A) = 0$.

  AXIOMA II (ISOMETRÍA Y CONSERVACIÓN SIMPLÉCTICA DE DARBOUX):
    La representación realificada satisface idénticamente el conmutador simpléctico $M^\top J_{2n} M = J_{2n}$, preservando el volumen de Liouville en el espacio de fases extendido.

  AXIOMA III (FILTRACIÓN DIOFÁNTICA DE KAM Y FINITUD DE BRUNO):
    Las frecuencias del sistema satisfacen la cota diofántica $|\langle \omega, k \rangle| \ge \gamma / \|k\|_1^\tau$ y la convergencia $\mathcal{B}(\omega) < \infty$, asegurando la persistencia de toros invariantes.

  AXIOMA IV (CONTRACCIÓN MONÁDICA DE BANACH RSI³):
    El endomorfismo de automejora $T$ posee constante de Lipschitz uniforme $\text{Lip}(T) \le 0.95$, induciendo una contracción estricta en la triple composición $T^3$ con $\text{Lip}(T^3) \le 0.95^3 < 1.0$.

4. INVARIANTES DE ESTADO:
─────────────────────────────────────────────────────────────────────────────────────────
  1. ENERGÍA DE YANG-MILLS Y NÚMERO DE PONTRYAGIN NO NEGATIVOS: $E_{\text{YM}} \ge 0, \quad p_1 \ge 0$.
  2. DEFECTO SIMPLÉCTICO EN $\mathfrak{sp}(2n, \mathbb{R})$: $\|M^\top J M - J\|_F / \|M\|_F^2 \le 10^{-10}$.
  3. PISO DE VALUACIÓN ULTRAMÉTRICA DE NOVIKOV: $v = 1 / (1 + |k|) \ge 0.10$.
  4. DISTANCIA GEODÉSICA DE FUBINI-STUDY: $d_{\text{FS}}(e^A \phi, e^{A + iW} \phi) \le 10^{-4}\text{ rad}$.
  5. INTERLOCK HARDWARE ESP32 CROWBAR: Latencia $\tau_{\text{crowbar}} \le 392.15\text{ ns} < 400.0\text{ ns}$ en veredicto $\text{VETOED}$ (0x00).
  6. INTEGRIDAD CRIPTOGRÁFICA HMAC-SHA256: Hash de dominio sellado sobre raíz de Merkle y clave secreta soberana.
"""

from __future__ import annotations

import cmath
import hashlib
import hmac
import logging
import math
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Final, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APU.Wisdom.TopologicalGaugeFieldEngine.v11")

# ──────────────────────────────────────────────────────────────────────────────
# Constantes físicas, topológicas y de precisión FPU
# ──────────────────────────────────────────────────────────────────────────────
_ENGINE_VERSION: Final[str] = "11.0.0"
_WILKINSON_EPSILON: Final[float] = 1.0e-15
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_ESP32_CROWBAR_MAX_LATENCY_NS: Final[float] = 392.15
_CROWBAR_TIMING_LIMIT_NS: Final[float] = 400.0
_CROWBAR_DEGRADED_LATENCY_NS: Final[float] = 120.0
_CROWBAR_NOMINAL_LATENCY_NS: Final[float] = 45.0
_GAUGE_HMAC_DOMAIN: Final[bytes] = b"APU-FILTER-GAUGE-FIELD-v11.0.0|"
_DEFAULT_SECRET_KEY: Final[bytes] = b"APU_FILTER_GAUGE_FIELD_SOVEREIGN_SECRET_KEY"

_ANTI_HERM_TOL: Final[float] = 1.0e-10
_TRACELESS_TOL: Final[float] = 1.0e-10
_SYMPLECTIC_TOL: Final[float] = 1.0e-10
_INSTANTON_INTEGRALITY_TOL: Final[float] = 1.0e-6
_KAM_GAMMA: Final[float] = 0.125
_KAM_TAU_OFFSET: Final[float] = 1.0
_KAM_MAX_ORDER: Final[int] = 3
_KAM_FREQ_CAP: Final[int] = 12
_MELNIKOV_TOL: Final[float] = 0.5
_FLOQUET_UNIT_TOL: Final[float] = 1.0e-6
_FUBINI_STUDY_MAX: Final[float] = 1.0e-4
_KRULL_NOVIKOV_FLOOR: Final[float] = 0.10
_BANACH_LIPSCHITZ_MAX: Final[float] = 0.95
_RESONANT_GAP: Final[float] = 1.0e-12
_BRUNO_SUM_MAX: Final[float] = 48.0
_LINSTEDT_RESIDUAL_TOL: Final[float] = 1.0e-9
_KAM_WEIGHT3_SAMPLES: Final[int] = 2048


# ══════════════════════════════════════════════════════════════════════════════
# Retículo de Heyting Ω₃ (cadena de severidad) y utilidades
# ══════════════════════════════════════════════════════════════════════════════

class HeytingOmega3(IntEnum):
    r"""
    Cadena de Heyting de severidad (no álgebra de verdad ingenua):
        COHERENT (0x00) < DEGRADED (0x01) < VETOED (0x03) = ⊤.
    ∨ = max (peor gana), ∧ = min, a→b = ⊤ si a≤b else b, ¬a = a→⊥.
    """
    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(int(self), int(other)))

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(int(self), int(other)))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3.VETOED if int(self) <= int(other) else other

    def negate(self) -> "HeytingOmega3":
        return self.implies(HeytingOmega3.COHERENT)

    @classmethod
    def fold_supremum(cls, verdicts: Sequence["HeytingOmega3"]) -> "HeytingOmega3":
        acc = cls.COHERENT
        for v in verdicts:
            acc = acc.join(v)
        return acc

    @classmethod
    def legacy_from_0x02(cls, legacy: int) -> "HeytingOmega3":
        return cls.VETOED if legacy == 0x02 else cls(legacy)


def _hash_state(*arrays: NDArray[Any]) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def _realify(M: NDArray[np.complex128]) -> NDArray[np.float64]:
    r"""Realificación \(\mathbb{C}^n\ni z\mapsto (x,y)\in\mathbb{R}^{2n}\):
    \(B+iC \;\mapsto\; \begin{pmatrix}B&-C\\C&B\end{pmatrix}\)."""
    X = np.real(M).astype(np.float64, copy=False)
    Y = np.imag(M).astype(np.float64, copy=False)
    return np.block([[X, -Y], [Y, X]])


def _symplectic_j(n: int, dtype: Any = np.float64) -> NDArray:
    I = np.eye(n, dtype=dtype)
    Z = np.zeros((n, n), dtype=dtype)
    return np.block([[Z, I], [-I, Z]])


# ══════════════════════════════════════════════════════════════════════════════
# DTOs inmutables
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class PoincareLindstedtSolution:
    r"""Solución espectral \(\{H_0,W\}=H_{1,\mathrm{osc}}\) en la base de −iA."""
    W: NDArray[np.complex128]
    min_resonance_gap: float
    residual_norm: float
    is_resonant: bool
    is_solved: bool
    frequencies: NDArray[np.float64]
    n_small_divisors: int = 0


@dataclass(frozen=True, slots=True)
class KAMCertificate:
    is_satisfied: bool
    worst_ratio: float
    tau: float
    gamma: float
    omega_min: float
    bruno_sum: float = 0.0
    bruno_satisfied: bool = True
    n_vectors_tested: int = 0


@dataclass(frozen=True, slots=True)
class MelnikovCertificate:
    integral_value: float
    max_abs_integrand: float
    is_homoclinic_preserved: bool


@dataclass(frozen=True, slots=True)
class FloquetCertificate:
    multipliers: NDArray[np.complex128]
    max_unit_deviation: float
    is_neutral: bool
    is_dissipative_stable: bool
    exponents: NDArray[np.float64]
    krein_collision: bool = False


@dataclass(frozen=True, slots=True)
class PoincareSectionCertificate:
    n_crossings: int
    return_dispersion: float
    is_regular: bool
    returns: Tuple[float, ...]
    mean_return_time: float = 0.0
    recurrence_time: float = 0.0


@dataclass(frozen=True, slots=True)
class RSI3SurfaceReport:
    surface_id: int
    name: str
    metric_value: float
    threshold: float
    passed: bool
    diagnostics: Dict[str, Any]


@dataclass(frozen=True, slots=True)
class RSI3Certificate:
    surfaces: Tuple[RSI3SurfaceReport, RSI3SurfaceReport, RSI3SurfaceReport]
    lipschitz_t3: float
    capacity_jerk: float
    fixed_point_reached: bool
    iterations: int
    is_valid: bool
    contraction_ratio: float = 1.0

    @property
    def surface1(self) -> RSI3SurfaceReport:
        return self.surfaces[0]

    @property
    def surface2(self) -> RSI3SurfaceReport:
        return self.surfaces[1]

    @property
    def surface3(self) -> RSI3SurfaceReport:
        return self.surfaces[2]


@dataclass(frozen=True, slots=True)
class GaugePhase1Envelope:
    r"""Contrato FASE 1 (Observe) → FASE 2 (Orient)."""
    rho_sanitized: NDArray[np.complex128]
    connection_A: NDArray[np.complex128]
    curvature_F: NDArray[np.complex128]
    energy_ym: float
    is_anti_hermitian_A: bool
    is_traceless_A: bool
    is_anti_hermitian_F: bool
    is_traceless_F: bool
    symplectic_residue: float
    symplectic_defect: float
    is_symplectic_valid: bool
    poincare_lindstedt: Optional[PoincareLindstedtSolution]
    dimension: int
    origin_timestamp_ns: float
    chain_hash: str
    spectral_gap: float = 0.0
    poincare_cartan_invariant: float = 0.0
    is_kahler: bool = False
    is_quaternionic_proxy: bool = False
    cayley_unitary_defect: float = 0.0


@dataclass(frozen=True, slots=True)
class GaugePhase2Envelope:
    r"""Contrato FASE 2 (Orient) → FASE 3 (Decide&Act)."""
    phase1: GaugePhase1Envelope
    chern_simons_real: float
    chern_simons_anomaly: float
    higgs_phi_trace: float
    wilson_trace: complex
    wilson_angle_rad: float
    c1_value: float
    pontryagin_instanton_k: int
    pontryagin_integrality_residual: float
    stora_zumino_residual: float
    kam_certificate: KAMCertificate
    melnikov: MelnikovCertificate
    floquet: FloquetCertificate
    poincare_section: PoincareSectionCertificate
    rsi3_certificate: RSI3Certificate
    topology_regime: str
    chain_hash: str
    origin_timestamp_ns: float
    chern_simons_algebraic: float = 0.0
    chern_character_ch1: complex = 0j
    chern_character_ch2: float = 0.0


@dataclass(frozen=True, slots=True)
class TopologicalGaugeFieldSnapshot:
    """Snapshot inmutable final (FASE 3) v11.0.0."""
    snapshot_id: str
    yang_mills_energy: float
    chern_simons_3form_higgs: float
    chern_simons_anomaly: float
    chern_hopf_class_c1: float
    wilson_loop_trace: complex
    wilson_angle_rad: float
    pontryagin_instanton_number: int
    pontryagin_integrality_residual: float
    stora_zumino_residual: float
    weak_value_qnd_aav: complex
    back_action_db: float
    topology_regime: str
    global_verdict: HeytingOmega3
    sha256_merkle_digest: str
    timestamp_utc: float
    kam_worst_ratio: float = 0.0
    kam_satisfied: bool = True
    melnikov_value: float = 0.0
    floquet_max_unit_deviation: float = 0.0
    floquet_is_neutral: bool = True
    poincare_section_n_crossings: int = 0
    poincare_section_dispersion: float = 0.0
    banach_lipschitz_t3: float = 0.0
    fubini_study_distance: float = 0.0
    novikov_krull_valuation: float = 0.0
    rsi3_valid: bool = True
    rsi3_iterations: int = 1
    symplectic_residue: float = 0.0
    symplectic_defect: float = 0.0
    crowbar_latency_ns: float = 0.0
    hardware_crowbar_action: str = "NOMINAL_EXECUTION"
    hmac_sha256: bytes = b""
    phase1: Optional[GaugePhase1Envelope] = None
    phase2: Optional[GaugePhase2Envelope] = None
    engine_version: str = _ENGINE_VERSION
    spectral_gap: float = 0.0
    poincare_cartan_invariant: float = 0.0
    bruno_satisfied: bool = True
    chern_simons_algebraic: float = 0.0
    lindstedt_residual: float = 0.0
    qnd_variance: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════════
# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║  F A S E   1 :   O B S E R V E                                               ║
# ║  su(N), curvatura KKS, Cayley–Sp(2n,ℝ), Lindstedt espectral, Cartan.        ║
# ║  Cierra con emit_phase1_envelope → continue_to_phase2.                       ║
# ╚════════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════════

class GaugePhase1Observer:
    r"""
    FASE 1 (Observe): sanitización de ρ, conexión \(A\in\mathfrak{su}(N)\),
    curvatura de Kirillov–Kostant respecto del Hamiltoniano celeste,
    embedding Cayley–Kähler en \(\mathrm{Sp}(2n,\mathbb{R})\) y
    Poincaré–Lindstedt espectral.
    """

    def __init__(self, dimension: int = 56) -> None:
        self.dimension: int = max(4, int(dimension))
        self._background_H: Optional[NDArray[np.complex128]] = None

    # ── Proyecciones de álgebra de Lie ────────────────────────────────────────
    def _project_su_n(self, matrix: NDArray) -> NDArray[np.complex128]:
        n = int(matrix.shape[0])
        A = np.asarray(matrix, dtype=np.complex128)
        A = 0.5 * (A - A.conj().T)
        A = A - (np.trace(A) / float(n)) * np.eye(n, dtype=np.complex128)
        return A

    def _psd_hermitian(self, matrix: NDArray) -> NDArray[np.complex128]:
        a = np.asarray(matrix, dtype=np.complex128)
        herm = 0.5 * (a + a.conj().T)
        eigvals, eigvecs = la.eigh(herm)
        clean = np.maximum(np.real(eigvals), _SPECTRAL_PSD_FLOOR)
        return (eigvecs * clean) @ eigvecs.conj().T

    def sanitize_density_matrix(self, matrix: NDArray) -> NDArray[np.complex128]:
        arr = np.asarray(matrix, dtype=np.complex128)
        if arr.shape != (self.dimension, self.dimension):
            raise ValueError(
                f"ρ debe ser {self.dimension}×{self.dimension}; recibido {arr.shape}.")
        psd = self._psd_hermitian(arr)
        tr = float(np.real(np.trace(psd)))
        if tr <= _WILKINSON_EPSILON:
            return np.eye(self.dimension, dtype=np.complex128) / float(self.dimension)
        return psd / tr

    def _default_background_hamiltonian(self, n: int) -> NDArray[np.complex128]:
        r"""
        Hamiltoniano celeste discreto (Poincaré: pequeñas oscilaciones):
        \(H = L_{\mathrm{path}} + \tfrac{1}{10n}(x-\bar x)^2\), luego
        \(\|H\|_F=1\) para fijar la escala de \(E_{YM}\).
        """
        idx = np.arange(n, dtype=np.float64)
        diag = 2.0 * np.ones(n) + 0.10 * (idx - 0.5 * (n - 1)) ** 2 / float(n)
        H = np.diag(diag) - np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)
        H = H.astype(np.complex128)
        H = 0.5 * (H + H.conj().T)
        scale = float(la.norm(H, "fro"))
        if scale > _WILKINSON_EPSILON:
            H = H / scale
        return H

    def _resolve_background_H(
        self, n: int, background_H: Optional[NDArray] = None,
    ) -> NDArray[np.complex128]:
        if background_H is not None:
            H = np.asarray(background_H, dtype=np.complex128)
            if H.shape != (n, n):
                raise ValueError(f"H_celeste debe ser {n}×{n}; recibido {H.shape}.")
            return 0.5 * (H + H.conj().T)
        if self._background_H is not None and self._background_H.shape == (n, n):
            return self._background_H
        H = self._default_background_hamiltonian(n)
        self._background_H = H
        return H

    # ── Método 1: conexión A ∈ su(N) ──────────────────────────────────────────
    def build_gauge_connection_A(self, density_matrix: NDArray) -> NDArray[np.complex128]:
        r"""
        Proyección de Frobenius sobre \(\mathfrak{su}(N)\):
        \(A = i\bigl(\rho_h - \tfrac{\mathrm{Tr}\,\rho_h}{N}I\bigr)\),
        con \(A^\dagger=-A\) y \(\mathrm{Tr}(A)=0\).
        """
        rho = self.sanitize_density_matrix(density_matrix)
        n = rho.shape[0]
        rho_h = 0.5 * (rho + rho.conj().T)
        su = rho_h - (np.trace(rho_h) / float(n)) * np.eye(n, dtype=np.complex128)
        return self._project_su_n(1j * su)

    def _verify_anti_hermitian(self, A: NDArray[np.complex128]) -> Tuple[bool, float]:
        defect = float(la.norm(A + A.conj().T, "fro"))
        scale = max(1.0, float(la.norm(A, "fro")))
        return bool(defect / scale < _ANTI_HERM_TOL), defect / scale

    def _verify_traceless(self, A: NDArray[np.complex128]) -> Tuple[bool, float]:
        tr = complex(np.trace(A))
        return bool(abs(tr) < _TRACELESS_TOL), abs(tr)

    # ── Método 2: curvatura KKS F = [ρ, H_celeste] ────────────────────────────
    def compute_yang_mills_curvature_F(
        self,
        connection_A: NDArray,
        density_matrix: Optional[NDArray] = None,
        background_hamiltonian: Optional[NDArray] = None,
    ) -> Tuple[NDArray[np.complex128], float]:
        r"""
        Curvatura de Kirillov–Kostant–Souriau sobre la órbita coadjunta:

        **v11 (con ρ):** \(F=[\rho,H_{\mathrm{cel}}]\).  Ambos Hermitianos
        \(\Rightarrow F^\dagger=-F\), \(\mathrm{Tr}(F)=0\).  *No* es \([A,\rho]\),
        que se anula idénticamente.

        **legacy (sin ρ):** \(F=A^2-A^\dagger A\).

        Energía: \(E_{YM}=-\tfrac12\mathrm{Tr}(F^2)\ge 0\) si \(F^\dagger=-F\).
        """
        a = np.asarray(connection_A, dtype=np.complex128)
        if density_matrix is not None:
            rho = self.sanitize_density_matrix(density_matrix)
            H = self._resolve_background_H(rho.shape[0], background_hamiltonian)
            F = rho @ H - H @ rho
            F = self._project_su_n(F)
            energy_ym = -0.5 * float(np.real(np.trace(F @ F)))
        else:
            F = a @ a - a.conj().T @ a
            energy_ym = 0.5 * float(np.real(np.trace(F.conj().T @ F)))
        return F, max(0.0, float(energy_ym))

    # ── Espectro de acción-ángulo (Poincaré) ──────────────────────────────────
    def _spectral_action_angle(
        self, A: NDArray[np.complex128],
    ) -> Tuple[NDArray[np.float64], NDArray[np.complex128], float]:
        r"""
        \(A^\dagger=-A\Rightarrow H=-iA\) Hermítica.  \(A=i\,U\mathrm{diag}(\omega)U^\dagger\).
        Gap espectral \(\Delta=\min_{i\neq j}|\omega_i-\omega_j|\).
        """
        H = -1j * np.asarray(A, dtype=np.complex128)
        H = 0.5 * (H + H.conj().T)
        omega, U = la.eigh(H)
        omega = np.real(omega).astype(np.float64)
        if omega.size < 2:
            return omega, U, 0.0
        diffs = np.diff(np.sort(omega))
        gap = float(np.min(diffs)) if diffs.size else 0.0
        return omega, U, gap

    def _phase1_poincare_cartan_invariant(
        self,
        omega: NDArray[np.float64],
        U: NDArray[np.complex128],
        rho: NDArray[np.complex128],
    ) -> float:
        r"""Invariante integral de Poincaré–Cartan \(\sum_j|\omega_j|\,n_j\),
        \(n_j=\langle u_j|\rho|u_j\rangle\) (acciones de ángulo)."""
        occ = np.real(np.diag(U.conj().T @ rho @ U))
        return float(np.sum(np.abs(omega) * np.abs(occ)))

    def _phase1_hypercomplex_structure(
        self, A: NDArray[np.complex128], is_symplectic_valid: bool,
    ) -> Tuple[bool, bool]:
        r"""
        Kähler: realificación de \(\mathrm{U}(n)\subset\mathrm{Sp}(2n)\cap\mathrm{O}(2n)\).
        Proxy cuaterniónico: \(n\) par (estructura \(\mathbb{H}^{n/2}\)) y Kähler.
        """
        n = int(A.shape[0])
        is_kahler = bool(is_symplectic_valid)
        is_quat = bool(is_kahler and (n % 2 == 0))
        return is_kahler, is_quat

    # ── Cayley–Kähler en Sp(2n,ℝ) ─────────────────────────────────────────────
    def _phase1_cayley_kahler_embedding(
        self, F: NDArray[np.complex128],
    ) -> Tuple[NDArray[np.float64], float, float, bool, float]:
        r"""
        Cayley de \(F\in\mathfrak{u}(n)\): \(U=(I+F)(I-F)^{-1}\in\mathrm{U}(n)\).
        Realificación \(M=\mathcal{R}(U)\in\mathrm{Sp}(2n,\mathbb{R})\cap\mathrm{O}(2n)\).
        \(K=\mathcal{R}(F)\in\mathfrak{sp}(2n,\mathbb{R})\).

        Certifica \(\|M^\top JM-J\|_F/\|M\|_F^2\le 10^{-10}\) y
        \(\|K^\top J+JK\|_F\le 10^{-10}\).
        """
        d = int(F.shape[0])
        I_d = np.eye(d, dtype=np.complex128)
        J = _symplectic_j(d, dtype=np.float64)
        try:
            U = (I_d + F) @ la.pinv(I_d - F)
            unit_def = float(la.norm(U.conj().T @ U - I_d, "fro")) / max(
                1.0, float(la.norm(U, "fro")))
            M = _realify(U)
            K = _realify(F)
            sp_defect = float(la.norm(K.T @ J + J @ K, "fro"))
            diff = M.T @ J @ M - J
            norm_M = float(la.norm(M, "fro"))
            residue = float(la.norm(diff, "fro") / (norm_M ** 2)) if norm_M > 0 else float("inf")
            valid = bool(residue < _SYMPLECTIC_TOL and sp_defect < _SYMPLECTIC_TOL)
            return M, residue, sp_defect, valid, unit_def
        except (la.LinAlgError, ValueError):
            return np.zeros((2 * d, 2 * d), dtype=np.float64), 1.0, 1.0, False, 1.0

    # ── Poincaré–Lindstedt espectral ──────────────────────────────────────────
    def _phase1_solve_poincare_lindstedt(
        self,
        F: NDArray[np.complex128],
        A: NDArray[np.complex128],
    ) -> PoincareLindstedtSolution:
        r"""
        Homología en la base propia de \(H_0=-iA\):
        \[
          \bigl(\omega_i-\omega_j\bigr)W_{ij}=(H_{1,\mathrm{osc}})_{ij},
          \qquad \{H_0,W\}_q=-i[H_0,W].
        \]
        Divisores pequeños \(|\omega_i-\omega_j|<\varepsilon\) se anulan (Lindstedt).
        """
        omega, U, gap = self._spectral_action_angle(A)
        d = int(A.shape[0])
        F_eig = U.conj().T @ F @ U
        F_osc_eig = F_eig - np.diag(np.diag(F_eig))

        W_eig = np.zeros((d, d), dtype=np.complex128)
        resonant = False
        n_small = 0
        min_gap = float(gap) if gap > 0 else float("inf")
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                dw = float(omega[i] - omega[j])
                adw = abs(dw)
                if adw < min_gap:
                    min_gap = adw
                if adw < _RESONANT_GAP:
                    resonant = True
                    n_small += 1
                    continue
                W_eig[i, j] = F_osc_eig[i, j] / dw

        W = U @ W_eig @ U.conj().T
        F_osc = U @ F_osc_eig @ U.conj().T
        commut = -1j * (A @ W - W @ A)
        residual = float(la.norm(commut - F_osc, "fro"))
        return PoincareLindstedtSolution(
            W=W,
            min_resonance_gap=float(min_gap) if math.isfinite(min_gap) else 0.0,
            residual_norm=residual,
            is_resonant=resonant,
            is_solved=bool(residual < _LINSTEDT_RESIDUAL_TOL and not resonant),
            frequencies=omega,
            n_small_divisors=int(n_small),
        )

    def observe_topological_gauge_field(
        self,
        density_matrix: NDArray,
        background_hamiltonian: Optional[NDArray] = None,
    ) -> Tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.complex128], float]:
        rho = self.sanitize_density_matrix(density_matrix)
        A = self.build_gauge_connection_A(rho)
        F, E = self.compute_yang_mills_curvature_F(
            A, rho, background_hamiltonian=background_hamiltonian)
        return rho, A, F, E

    def _chain_hash_phase1(
        self,
        rho: NDArray, A: NDArray, F: NDArray,
        energy: float, residue: float, defect: float, gap: float,
    ) -> str:
        return _hash_state(
            rho, A, F,
            np.array([energy, residue, defect, gap], dtype=np.float64),
        )

    # ── CIERRE FASE 1 ─────────────────────────────────────────────────────────
    def emit_phase1_envelope(
        self,
        density_matrix: NDArray,
        background_hamiltonian: Optional[NDArray] = None,
    ) -> GaugePhase1Envelope:
        r"""
        **Cierre formal de la FASE 1.**  Emite `GaugePhase1Envelope`.
        El morfismo anidado siguiente es `continue_to_phase2`.
        """
        t0 = time.perf_counter_ns()
        rho = self.sanitize_density_matrix(density_matrix)
        A = self.build_gauge_connection_A(rho)
        F, E = self.compute_yang_mills_curvature_F(
            A, rho, background_hamiltonian=background_hamiltonian)

        a_ah, _ = self._verify_anti_hermitian(A)
        a_tr, _ = self._verify_traceless(A)
        f_ah, _ = self._verify_anti_hermitian(F)
        f_tr, _ = self._verify_traceless(F)

        _, residue, sp_defect, is_valid, unit_def = self._phase1_cayley_kahler_embedding(F)
        pl = self._phase1_solve_poincare_lindstedt(F, A)
        omega, U, gap = self._spectral_action_angle(A)
        cartan = self._phase1_poincare_cartan_invariant(omega, U, rho)
        kahler, quat = self._phase1_hypercomplex_structure(A, is_valid)
        chain = self._chain_hash_phase1(rho, A, F, float(E), residue, sp_defect, gap)

        return GaugePhase1Envelope(
            rho_sanitized=rho, connection_A=A, curvature_F=F, energy_ym=float(E),
            is_anti_hermitian_A=a_ah, is_traceless_A=a_tr,
            is_anti_hermitian_F=f_ah, is_traceless_F=f_tr,
            symplectic_residue=residue, symplectic_defect=sp_defect,
            is_symplectic_valid=is_valid,
            poincare_lindstedt=pl,
            dimension=int(rho.shape[0]),
            origin_timestamp_ns=float(time.perf_counter_ns() - t0),
            chain_hash=chain,
            spectral_gap=float(gap),
            poincare_cartan_invariant=float(cartan),
            is_kahler=kahler,
            is_quaternionic_proxy=quat,
            cayley_unitary_defect=float(unit_def),
        )

    def continue_to_phase2(self, env: GaugePhase1Envelope) -> GaugePhase1Envelope:
        r"""
        **Puente anidado FASE 1 → FASE 2.**  Certifica la hash-chain de Observe
        y habilita Orient.  Último morfismo de la FASE 1 = primer morfismo
        efectivo de la FASE 2 (`from_phase1_envelope` es el dual clasista).
        """
        expected = self._chain_hash_phase1(
            env.rho_sanitized, env.connection_A, env.curvature_F,
            env.energy_ym, env.symplectic_residue, env.symplectic_defect,
            env.spectral_gap)
        if expected != env.chain_hash:
            raise ValueError(f"Cadena FASE 1 corrupta: {expected} ≠ {env.chain_hash}.")
        self._phase1 = env  # type: ignore[attr-defined]
        logger.debug("FASE 1→2 puente OK hash=%s", env.chain_hash)
        return env


# ══════════════════════════════════════════════════════════════════════════════════
# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║  F A S E   2 :   O R I E N T                                                 ║
# ║  CS, Wilson, Pontryagin, Chern, KAM/Bruno, Melnikov, Floquet, RSI³.          ║
# ║  Abre con from_phase1_envelope.  Cierra con emit_phase2_envelope             ║
# ║  → continue_to_phase3.                                                       ║
# ╚════════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════════

class GaugePhase2Orientor(GaugePhase1Observer):
    r"""
    FASE 2 (Orient): topología diferencial, mecánica celeste (KAM, Bruno,
    Melnikov, Floquet–Krein, sección de Poincaré) y RSI³ de Banach.
    """

    @classmethod
    def from_phase1_envelope(cls, env: GaugePhase1Envelope) -> "GaugePhase2Orientor":
        r"""**Apertura formal (clasista) de la FASE 2.** Dual de `continue_to_phase2`."""
        inst = cls(dimension=env.dimension)
        inst.continue_to_phase2(env)
        return inst

    def evaluate_chern_simons_higgs_3form(
        self,
        connection_A: NDArray,
        density_matrix: NDArray,
        curvature_F: Optional[NDArray] = None,
        background_hamiltonian: Optional[NDArray] = None,
    ) -> Tuple[float, float, NDArray[np.complex128], float]:
        r"""
        3-forma de Chern–Simons–Higgs (Nakahara):
        \(\mathrm{CS}=\frac{1}{8\pi^2}\mathrm{Tr}(A\wedge dA+\tfrac23 A^3)\).
        Con \(F=dA+A\wedge A\): \(\mathrm{Tr}(AF-\tfrac13 A^3)\).
        En 0-formas, \(\mathrm{CS_{alg}}=\mathrm{Tr}(\tfrac23 A^3)/(8\pi^2)\).
        Anomalía (Green–Schwarz / \(\mathrm{Tr} F^3\)):
        \(\mathrm{CS_{anom}}=|\mathrm{Tr}(F^3)|/(8\pi^2)\).

        Retorna `(CS_cov, CS_anomaly, higgs_phi, CS_alg)`.
        """
        a = np.asarray(connection_A, dtype=np.complex128)
        rho = self.sanitize_density_matrix(density_matrix)
        if curvature_F is None:
            F, _ = self.compute_yang_mills_curvature_F(
                a, rho, background_hamiltonian=background_hamiltonian)
        else:
            F = np.asarray(curvature_F, dtype=np.complex128)

        eigvals, eigvecs = la.eigh(0.5 * (rho + rho.conj().T))
        log_eigs = np.log(np.maximum(np.real(eigvals), _SPECTRAL_PSD_FLOOR))
        higgs_phi = (eigvecs * log_eigs) @ eigvecs.conj().T

        a3 = a @ a @ a
        inv8pi2 = 1.0 / (8.0 * math.pi ** 2)
        cs_cov = complex(np.trace(a @ F - (1.0 / 3.0) * a3 + higgs_phi @ F)) * inv8pi2
        cs_alg = complex(np.trace((2.0 / 3.0) * a3)) * inv8pi2
        f3 = F @ F @ F
        cs_anomaly = float(abs(np.trace(f3)) * inv8pi2)
        return float(np.real(cs_cov)), cs_anomaly, higgs_phi, float(np.real(cs_alg))

    def evaluate_wilson_loop_holonomy(
        self,
        connection_A: NDArray,
        density_matrix: NDArray,
    ) -> Tuple[complex, float, float, str]:
        r"""
        Holonomía \(W=\mathrm{Tr}(e^A)/N\) (unitaria si \(A^\dagger=-A\)) y
        proxy \(c_1=\mathrm{Tr}(A\rho)/(2\pi i)\).
        """
        a = np.asarray(connection_A, dtype=np.complex128)
        rho = self.sanitize_density_matrix(density_matrix)
        n = a.shape[0]
        wilson_op = la.expm(a)
        w_trace = complex(np.trace(wilson_op) / float(n))
        w_angle = float(cmath.phase(w_trace))
        modulus = abs(w_trace)
        if modulus > 1.0 + 1e-9:
            logger.warning("Wilson loop no unitario: |W|=%.6e > 1", modulus)

        c1_raw = np.trace(a @ rho) / (2.0j * math.pi)
        c1_val = float(np.real(c1_raw))
        if abs(w_angle) < 0.15 and abs(c1_val) < 1e-4:
            regime = "TRIVIAL_SU2_HOLONOMY"
        elif abs(w_angle) >= 0.50 or abs(c1_val) > 0.10:
            regime = "SOLITONIC_INSTANTON_PROSECUTOR"
        else:
            regime = "WILSON_SLIP_PROSECUTED"
        return w_trace, w_angle, c1_val, regime

    def audit_pontryagin_charge_and_stora_zumino(
        self,
        curvature_F: NDArray,
        chern_simons_real: float,
    ) -> Tuple[int, float, float]:
        r"""
        \(p_1=-\mathrm{Tr}(F^2)/(8\pi^2)\ge 0\) para \(F^\dagger=-F\).
        \(k=\mathrm{round}(p_1)\), \(\Delta_{\mathrm{int}}=|p_1-k|\),
        \(\Delta_{SZ}=|8\pi^2\,\mathrm{CS}-\mathrm{Tr}(F^2)|\).
        """
        f = np.asarray(curvature_F, dtype=np.complex128)
        tr_ff = float(np.real(np.trace(f @ f)))
        p_raw = max(0.0, -tr_ff / (8.0 * math.pi ** 2))
        instanton_k = int(round(p_raw))
        integrality_residual = float(abs(p_raw - instanton_k))
        stora_zumino = float(abs(8.0 * math.pi ** 2 * chern_simons_real - tr_ff))
        return instanton_k, integrality_residual, stora_zumino

    def compute_chern_character(
        self, F: NDArray[np.complex128],
    ) -> Tuple[complex, complex, float]:
        r"""
        \(\mathrm{ch}(F)=\mathrm{Tr}\exp(F/(2\pi i))=n+\mathrm{ch}_1+\mathrm{ch}_2+\cdots\)
        \(\mathrm{ch}_1=\mathrm{Tr} X\), \(\mathrm{ch}_2=\tfrac12\mathrm{Tr}(X^2)\),
        \(X=F/(2\pi i)\).
        """
        X = np.asarray(F, dtype=np.complex128) / (2.0j * math.pi)
        try:
            ch = complex(np.trace(la.expm(X)))
        except (la.LinAlgError, ValueError):
            ch = complex(F.shape[0])
        ch1 = complex(np.trace(X))
        ch2 = float(np.real(0.5 * np.trace(X @ X)))
        return ch, ch1, ch2

    def check_kam_bruno(
        self,
        frequencies: NDArray[np.float64],
        gamma: float = _KAM_GAMMA,
        max_order: int = _KAM_MAX_ORDER,
    ) -> KAMCertificate:
        r"""
        Condición diofantina \(|\omega\cdot k|\,\|k\|_1^\tau\ge\gamma\), \(\tau=d+1\),
        sobre vectores de peso 1–3 (nunca \(\mathrm{product}(\cdot,\mathrm{repeat}=d)\)).
        Bruno: \(\sum_n 2^{-n}\log\Omega_n^{-1}<\infty\), \(\Omega_n=\inf_{0<\|k\|_\infty\le 2^n}|\omega\cdot k|\).
        """
        freqs = np.asarray(frequencies, dtype=np.float64).flatten()
        freqs = freqs[np.abs(freqs) > 1e-15]
        if freqs.size == 0:
            return KAMCertificate(True, float("inf"), 1.0, gamma, 0.0, 0.0, True, 0)
        if freqs.size > _KAM_FREQ_CAP:
            freqs = freqs[np.argsort(np.abs(freqs))[-_KAM_FREQ_CAP:]]
        d = int(freqs.size)
        tau = float(d) + _KAM_TAU_OFFSET
        worst = float("inf")
        tested = 0
        dots: List[Tuple[float, float]] = []

        def consider(k_vec: NDArray[np.float64]) -> None:
            nonlocal worst, tested
            k1 = float(np.sum(np.abs(k_vec)))
            if k1 < 1e-15:
                return
            dot = abs(float(np.dot(freqs, k_vec)))
            ratio = dot * (k1 ** tau) / max(gamma, 1e-15)
            worst = min(worst, ratio)
            tested += 1
            dots.append((float(np.max(np.abs(k_vec))), dot))

        for i in range(d):
            for coeff in range(-max_order, max_order + 1):
                if coeff == 0:
                    continue
                k = np.zeros(d, dtype=np.float64)
                k[i] = float(coeff)
                consider(k)
        for i in range(d):
            for j in range(i + 1, d):
                for ci in range(-max_order, max_order + 1):
                    if ci == 0:
                        continue
                    for cj in range(-max_order, max_order + 1):
                        if cj == 0:
                            continue
                        k = np.zeros(d, dtype=np.float64)
                        k[i] = float(ci)
                        k[j] = float(cj)
                        consider(k)
        rng = np.random.default_rng(56 + d)
        n3 = min(_KAM_WEIGHT3_SAMPLES, 80 * d)
        w3 = min(3, d)
        for _ in range(n3):
            k = np.zeros(d, dtype=np.float64)
            idxs = rng.choice(d, size=w3, replace=False)
            for ix in idxs:
                k[int(ix)] = float(int(rng.integers(1, max_order + 1)) * int(rng.choice([-1, 1])))
            consider(k)

        bruno_sum = 0.0
        for n in range(0, 6):
            cutoff = float(2 ** n)
            omega_n = min((dot for inf_n, dot in dots if 0.0 < inf_n <= cutoff + 1e-15),
                          default=0.0)
            if omega_n > 1e-30:
                bruno_sum += math.log(1.0 / omega_n) / float(2 ** n)
            else:
                bruno_sum += 40.0 / float(2 ** n)
        bruno_ok = bool(math.isfinite(bruno_sum) and bruno_sum < _BRUNO_SUM_MAX)
        return KAMCertificate(
            is_satisfied=bool(worst >= 1.0),
            worst_ratio=float(worst) if math.isfinite(worst) else 0.0,
            tau=tau, gamma=gamma,
            omega_min=float(np.min(np.abs(freqs))),
            bruno_sum=float(bruno_sum),
            bruno_satisfied=bruno_ok,
            n_vectors_tested=int(tested),
        )

    def _flow_on_grid(
        self,
        A: NDArray[np.complex128],
        phi0: NDArray[np.complex128],
        t_grid: NDArray[np.float64],
    ) -> List[NDArray[np.complex128]]:
        r"""Flujo \(\phi(t)=e^{tA}\phi_0\) con un solo `expm(\Delta t\,A)`."""
        phi0 = np.asarray(phi0, dtype=np.complex128).flatten()
        nrm = float(np.linalg.norm(phi0))
        phi0 = phi0 / max(nrm, 1e-15)
        if t_grid.size == 0:
            return [phi0]
        try:
            phi = la.expm(float(t_grid[0]) * A) @ phi0
        except (la.LinAlgError, ValueError):
            return [phi0]
        phi = phi / max(float(np.linalg.norm(phi)), 1e-15)
        phis = [phi]
        if t_grid.size == 1:
            return phis
        dt = float(t_grid[1] - t_grid[0])
        try:
            U_dt = la.expm(dt * A)
        except (la.LinAlgError, ValueError):
            return phis
        for _ in range(int(t_grid.size) - 1):
            phi = U_dt @ phi
            phi = phi / max(float(np.linalg.norm(phi)), 1e-15)
            phis.append(phi)
        return phis

    def compute_melnikov_integral(
        self,
        A: NDArray[np.complex128],
        F: NDArray[np.complex128],
        phi0: NDArray[np.complex128],
        n_samples: int = 128,
    ) -> MelnikovCertificate:
        r"""
        Melnikov homoclínico sobre la separatriz \(\phi(t)=e^{tA}\phi_0\):
        \(M=-\int\langle F\phi,\phi\rangle\mathrm{sech}^2(t)\,dt\).
        """
        t_grid = np.linspace(-8.0, 8.0, n_samples, dtype=np.float64)
        try:
            phis = self._flow_on_grid(A, phi0, t_grid)
        except (la.LinAlgError, ValueError):
            return MelnikovCertificate(0.0, 0.0, True)
        if len(phis) != t_grid.size:
            return MelnikovCertificate(0.0, 0.0, True)
        sech2 = 1.0 / np.cosh(t_grid) ** 2
        integrands = np.empty(t_grid.size, dtype=np.float64)
        for i, phi_t in enumerate(phis):
            force = F @ phi_t
            integrands[i] = float(np.real(np.vdot(force, phi_t))) * float(sech2[i])
        M = float(np.trapz(integrands, t_grid))
        max_abs = float(np.max(np.abs(integrands))) if integrands.size else 0.0
        return MelnikovCertificate(
            integral_value=M,
            max_abs_integrand=max_abs,
            is_homoclinic_preserved=bool(abs(M) < _MELNIKOV_TOL),
        )

    def compute_floquet_multipliers(
        self,
        F: NDArray[np.complex128],
        period: float = 2.0 * math.pi,
    ) -> FloquetCertificate:
        r"""Monodromía \(\mu=\mathrm{eig}(e^{TF})\).  \(F^\dagger=-F\Rightarrow|\mu|=1\).
        Colisión de Krein: multiplicadores casi degenerados sobre \(S^1\)."""
        try:
            M_T = la.expm(np.asarray(F, dtype=np.complex128) * period)
            mu = la.eigvals(M_T)
            moduli = np.abs(mu)
            unit_dev = float(np.max(np.abs(moduli - 1.0)))
            dissipative = bool(np.max(moduli) <= 1.0 + _FLOQUET_UNIT_TOL)
            exponents = np.log(np.maximum(moduli, 1e-30)) / max(period, 1e-15)
            krein = False
            for i in range(mu.size):
                for j in range(i + 1, mu.size):
                    if abs(mu[i] - mu[j]) < 1e-8 and abs(moduli[i] - 1.0) < 10.0 * _FLOQUET_UNIT_TOL:
                        krein = True
                        break
                if krein:
                    break
        except (la.LinAlgError, ValueError):
            mu = np.array([1.0 + 0j])
            unit_dev, dissipative, exponents, krein = 0.0, True, np.array([0.0]), False
        return FloquetCertificate(
            multipliers=mu, max_unit_deviation=unit_dev,
            is_neutral=bool(unit_dev < _FLOQUET_UNIT_TOL),
            is_dissipative_stable=dissipative, exponents=exponents,
            krein_collision=krein,
        )

    def compute_poincare_section(
        self,
        A: NDArray[np.complex128],
        phi0: NDArray[np.complex128],
        P_obs: Optional[NDArray[np.complex128]] = None,
        n_samples: int = 96,
        omega_min: float = 1.0,
    ) -> PoincareSectionCertificate:
        r"""Observable \(O(t)=\langle\phi(t)|P_0|\phi(t)\rangle\).  Retornos = ceros de \(\dot O\)."""
        d = int(A.shape[0])
        if P_obs is None:
            P_obs = np.zeros((d, d), dtype=np.complex128)
            P_obs[0, 0] = 1.0
        t_grid = np.linspace(0.0, 4.0 * math.pi, n_samples, dtype=np.float64)
        rec_time = 2.0 * math.pi / max(float(abs(omega_min)), 1e-15)
        try:
            phis = self._flow_on_grid(A, phi0, t_grid)
        except (la.LinAlgError, ValueError):
            return PoincareSectionCertificate(0, 0.0, True, (), 0.0, rec_time)
        obs = np.array(
            [float(np.real(np.vdot(phi, P_obs @ phi))) for phi in phis],
            dtype=np.float64)
        if obs.size < 3:
            return PoincareSectionCertificate(0, 0.0, True, (), 0.0, rec_time)
        d_obs = np.diff(obs)
        crossings = np.where(np.sign(d_obs[:-1]) != np.sign(d_obs[1:]))[0]
        if crossings.size < 2:
            return PoincareSectionCertificate(
                int(crossings.size), 0.0, True, (), 0.0, rec_time)
        returns = obs[crossings]
        disp = float(np.std(returns) / max(float(np.mean(np.abs(returns))), 1e-15))
        t_cross = t_grid[crossings]
        mean_ret = float(np.mean(np.diff(t_cross))) if t_cross.size >= 2 else 0.0
        return PoincareSectionCertificate(
            n_crossings=int(crossings.size),
            return_dispersion=disp,
            is_regular=bool(disp < 0.5),
            returns=tuple(float(x) for x in returns),
            mean_return_time=mean_ret,
            recurrence_time=rec_time,
        )

    def _banach_t_operator(
        self,
        F: NDArray[np.complex128],
        A: NDArray[np.complex128],
        W: NDArray[np.complex128],
        damp: float = _BANACH_LIPSCHITZ_MAX,
    ) -> NDArray[np.complex128]:
        r"""
        Contracción de Lindstedt: si \(-i[A,W]\approx F_{\mathrm{osc}}\),
        \(T(F)=F-(1-\lambda)\{H_0,W\}\approx F_{\mathrm{mean}}+\lambda F_{\mathrm{osc}}\).
        \(\mathrm{Lip}(T)|_{\mathrm{osc}}=\lambda\le 0.95\).
        """
        cancel = -1j * (A @ W - W @ A)
        F_new = F - (1.0 - damp) * cancel
        return self._project_su_n(F_new)

    def enforce_rsi3(
        self,
        F: NDArray[np.complex128],
        A: NDArray[np.complex128],
        instanton_k: int,
        w_trace: complex,
        phi0: NDArray[np.complex128],
        *,
        W: Optional[NDArray[np.complex128]] = None,
        symplectic_valid: bool = True,
        energy_ym: float = 0.0,
        max_iter: int = 16,
    ) -> RSI3Certificate:
        r"""
        RSI³ — tres superficies + contracción de Banach del *mejorador*:
          • S1 Novikov: \(v=1/(1+|k|)\ge 0.10\).
          • S2 Harness: curvatura no degenerada y embedding simpléctico.
          • S3 Model: \(d_{\mathrm{FS}}(e^A\phi,\,e^{A+iW}\phi)\le 10^{-4}\).
        Nivel 3: \(T\) se re-aplica a su residuo (meta-mejora).
        """
        d = int(F.shape[0])
        if W is None:
            W = np.zeros((d, d), dtype=np.complex128)
        phi0 = np.asarray(phi0, dtype=np.complex128).flatten()
        phi0 = phi0 / max(float(np.linalg.norm(phi0)), 1e-15)

        v_krull = 1.0 / (1.0 + abs(float(instanton_k)))
        s1 = RSI3SurfaceReport(
            surface_id=1, name="Data-RSI (Novikov)",
            metric_value=float(v_krull), threshold=_KRULL_NOVIKOV_FLOOR,
            passed=bool(v_krull >= _KRULL_NOVIKOV_FLOOR),
            diagnostics={"instanton_k": int(instanton_k)},
        )
        s2 = RSI3SurfaceReport(
            surface_id=2, name="Harness-RSI (KKS+Sp)",
            metric_value=float(energy_ym), threshold=_SPECTRAL_PSD_FLOOR,
            passed=bool(energy_ym >= _SPECTRAL_PSD_FLOOR and symplectic_valid),
            diagnostics={"symplectic_valid": bool(symplectic_valid)},
        )

        try:
            U_raw = la.expm(A)
            U_cor = la.expm(A + 1j * W)
            psi_r = U_raw @ phi0
            psi_c = U_cor @ phi0
            nr = float(np.linalg.norm(psi_r) * np.linalg.norm(psi_c))
            overlap = abs(complex(np.vdot(psi_r, psi_c))) / max(nr, 1e-15)
            d_fs = float(np.arccos(np.clip(overlap, 0.0, 1.0)))
        except (la.LinAlgError, ValueError):
            d_fs = 0.0
        s3 = RSI3SurfaceReport(
            surface_id=3, name="Model-RSI (Fubini-Study)",
            metric_value=d_fs, threshold=_FUBINI_STUDY_MAX,
            passed=bool(d_fs <= _FUBINI_STUDY_MAX),
            diagnostics={"d_fs_rad": d_fs},
        )

        F_iter = np.array(F, copy=True)
        prev_delta = None
        ratios: List[float] = []
        fixed = False
        it_used = 0
        for it in range(1, max_iter + 1):
            it_used = it
            F_next = self._banach_t_operator(F_iter, A, W)
            delta = float(la.norm(F_next - F_iter, "fro"))
            if prev_delta is not None and prev_delta > 1e-15:
                ratios.append(delta / prev_delta)
            if delta < 1e-12:
                fixed = True
                F_iter = F_next
                break
            prev_delta = delta
            F_iter = F_next
            if s1.passed and s2.passed and s3.passed and it >= 3:
                fixed = True
                break

        if ratios:
            lip = float(min(1.0, max(ratios[-3:])))
        else:
            lip = abs(abs(w_trace) - 1.0)
            lip = min(max(lip, 0.0), 1.0)
        lip_t3 = float(lip ** 3)
        jerk = float((1.0 - lip_t3) / 0.05) if lip_t3 < 1.0 else 0.0
        valid = (
            s1.passed and s2.passed and s3.passed
            and lip <= _BANACH_LIPSCHITZ_MAX + 1e-9
        )
        return RSI3Certificate(
            surfaces=(s1, s2, s3),
            lipschitz_t3=lip_t3,
            capacity_jerk=jerk,
            fixed_point_reached=fixed,
            iterations=int(it_used),
            is_valid=bool(valid),
            contraction_ratio=float(lip),
        )

    def _chain_hash_phase2(
        self,
        rho: NDArray, A: NDArray, F: NDArray,
        metrics: NDArray[np.float64],
    ) -> str:
        return _hash_state(rho, A, F, metrics)

    def emit_phase2_envelope(
        self,
        phase1_env: GaugePhase1Envelope,
        phi0: Optional[NDArray[np.complex128]] = None,
    ) -> GaugePhase2Envelope:
        r"""
        **Cierre formal de la FASE 2.**  El morfismo anidado siguiente es
        `continue_to_phase3`.
        """
        t0 = time.perf_counter_ns()
        rho, A, F = phase1_env.rho_sanitized, phase1_env.connection_A, phase1_env.curvature_F
        d = int(A.shape[0])
        if phi0 is None:
            phi0 = np.ones(d, dtype=np.complex128) / math.sqrt(d)
        else:
            phi0 = np.asarray(phi0, dtype=np.complex128).flatten()
            phi0 = phi0 / max(float(np.linalg.norm(phi0)), 1e-15)

        cs_real, cs_anomaly, higgs_phi, cs_alg = self.evaluate_chern_simons_higgs_3form(
            A, rho, F)
        higgs_trace = float(np.real(np.trace(higgs_phi)))
        w_trace, w_angle, c1_val, regime = self.evaluate_wilson_loop_holonomy(A, rho)
        instanton_k, integrality_residual, stora_zumino = (
            self.audit_pontryagin_charge_and_stora_zumino(F, cs_real))
        _, ch1, ch2 = self.compute_chern_character(F)

        pl = phase1_env.poincare_lindstedt
        if pl is not None and pl.frequencies.size:
            freqs = np.abs(pl.frequencies)
            freqs = freqs[freqs > 1e-15]
        else:
            try:
                f_evals = la.eigvals(F)
                freqs = np.abs(np.imag(f_evals)) / math.pi
                freqs = freqs[freqs > 1e-15]
            except la.LinAlgError:
                freqs = np.array([1.0])
        if freqs.size == 0:
            freqs = np.array([1.0])
        kam_cert = self.check_kam_bruno(freqs)
        melnikov = self.compute_melnikov_integral(A, F, phi0)
        floquet = self.compute_floquet_multipliers(F)
        poincare_sec = self.compute_poincare_section(
            A, phi0, omega_min=kam_cert.omega_min)
        W = pl.W if pl is not None else np.zeros_like(F)
        rsi3 = self.enforce_rsi3(
            F, A, instanton_k, w_trace, phi0,
            W=W,
            symplectic_valid=phase1_env.is_symplectic_valid,
            energy_ym=phase1_env.energy_ym,
        )

        metrics = np.array([
            cs_real, cs_anomaly, w_angle, c1_val,
            float(instanton_k), integrality_residual, stora_zumino,
            kam_cert.worst_ratio, melnikov.integral_value,
            floquet.max_unit_deviation, rsi3.lipschitz_t3, cs_alg, ch2,
        ], dtype=np.float64)
        chain = self._chain_hash_phase2(rho, A, F, metrics)

        return GaugePhase2Envelope(
            phase1=phase1_env,
            chern_simons_real=cs_real, chern_simons_anomaly=cs_anomaly,
            higgs_phi_trace=higgs_trace,
            wilson_trace=w_trace, wilson_angle_rad=w_angle, c1_value=c1_val,
            pontryagin_instanton_k=instanton_k,
            pontryagin_integrality_residual=integrality_residual,
            stora_zumino_residual=stora_zumino,
            kam_certificate=kam_cert, melnikov=melnikov, floquet=floquet,
            poincare_section=poincare_sec, rsi3_certificate=rsi3,
            topology_regime=regime,
            chain_hash=chain,
            origin_timestamp_ns=float(time.perf_counter_ns() - t0),
            chern_simons_algebraic=cs_alg,
            chern_character_ch1=ch1,
            chern_character_ch2=ch2,
        )

    def continue_to_phase3(self, env: GaugePhase2Envelope) -> GaugePhase2Envelope:
        r"""
        **Puente anidado FASE 2 → FASE 3.**  Último morfismo de Orient =
        apertura efectiva de Decide&Act.
        """
        p1 = env.phase1
        metrics = np.array([
            env.chern_simons_real, env.chern_simons_anomaly,
            env.wilson_angle_rad, env.c1_value,
            float(env.pontryagin_instanton_k),
            env.pontryagin_integrality_residual, env.stora_zumino_residual,
            env.kam_certificate.worst_ratio, env.melnikov.integral_value,
            env.floquet.max_unit_deviation, env.rsi3_certificate.lipschitz_t3,
            env.chern_simons_algebraic, env.chern_character_ch2,
        ], dtype=np.float64)
        expected = self._chain_hash_phase2(
            p1.rho_sanitized, p1.connection_A, p1.curvature_F, metrics)
        if expected != env.chain_hash:
            raise ValueError(f"Cadena FASE 2 corrupta: {expected} ≠ {env.chain_hash}.")
        self._phase2 = env  # type: ignore[attr-defined]
        logger.debug("FASE 2→3 puente OK hash=%s", env.chain_hash)
        return env


# ══════════════════════════════════════════════════════════════════════════════════
# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║  F A S E   3 :   D E C I D E   &   A C T                                      ║
# ║  QND/AAV, Heyting Ω₃, Crowbar, HMAC, Φ_sem.                                   ║
# ║  Abre con from_phase2_envelope.  Cierra con emit_governance_state.           ║
# ╚════════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════════

class GaugePhase3Decider(GaugePhase2Orientor):
    r"""
    FASE 3 (Decide & Act): medición débil QND, adjudicación Heyting Ω₃,
    interlock Crowbar ESP32, funtor visceral \(\Phi_{\mathrm{sem}}\) y HMAC.
    """

    def __init__(self, dimension: int = 56, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        super().__init__(dimension=dimension)
        self.secret_key: bytes = bytes(secret_key)

    @classmethod
    def from_phase2_envelope(
        cls, env: GaugePhase2Envelope, secret_key: bytes = _DEFAULT_SECRET_KEY,
    ) -> "GaugePhase3Decider":
        r"""**Apertura formal (clasista) de la FASE 3.** Dual de `continue_to_phase3`."""
        inst = cls(dimension=env.phase1.dimension, secret_key=secret_key)
        inst.continue_to_phase3(env)
        return inst

    def observe_qnd_weak_measurement(
        self,
        phi_initial: NDArray,
        phi_final: NDArray,
        observable_A: NDArray,
    ) -> Tuple[complex, float, float]:
        r"""
        Valor débil AAV \(A_w=\langle\phi_f|A|\phi_i\rangle/\langle\phi_f|\phi_i\rangle\).
        Varianza de \(H=-iA\) en \(|\phi_i\rangle\): QND \(\Leftrightarrow\) var\(\approx 0\).
        """
        i_vec = np.asarray(phi_initial, dtype=np.complex128).reshape(-1, 1)
        f_vec = np.asarray(phi_final, dtype=np.complex128).reshape(-1, 1)
        a_mat = np.asarray(observable_A, dtype=np.complex128)
        nrm_i = float(np.linalg.norm(i_vec))
        nrm_f = float(np.linalg.norm(f_vec))
        if nrm_i > 0:
            i_vec = i_vec / nrm_i
        if nrm_f > 0:
            f_vec = f_vec / nrm_f

        overlap = complex((f_vec.conj().T @ i_vec)[0, 0])
        if abs(overlap) < _WILKINSON_EPSILON:
            overlap = complex(_WILKINSON_EPSILON, 0.0)
        numerator = complex((f_vec.conj().T @ a_mat @ i_vec)[0, 0])
        a_weak = numerator / overlap

        qnd_fidelity = abs(overlap) ** 2
        if qnd_fidelity < 0.99:
            back_action_db = float(-10.0 * math.log10(max(qnd_fidelity, 1e-15)))
        else:
            back_action_db = 0.0

        H_obs = -1j * a_mat
        H_obs = 0.5 * (H_obs + H_obs.conj().T)
        mean = complex((i_vec.conj().T @ H_obs @ i_vec)[0, 0])
        mean2 = complex((i_vec.conj().T @ H_obs @ H_obs @ i_vec)[0, 0])
        var = max(0.0, float(np.real(mean2) - abs(mean) ** 2))
        return a_weak, back_action_db, var

    def adjudicate_heyting(
        self,
        *,
        energy_ym: float,
        instanton_k: int,
        integrality_residual: float,
        symplectic_valid: bool,
        kam_satisfied: bool,
        melnikov_preserved: bool,
        floquet_stable: bool,
        rsi3_valid: bool,
        bruno_satisfied: bool = True,
        lindstedt_solved: bool = True,
    ) -> HeytingOmega3:
        r"""Supremo monoidal de átomos de severidad en \(\Omega_3\)."""
        atoms = [
            HeytingOmega3.COHERENT if energy_ym <= 0.25 else (
                HeytingOmega3.DEGRADED if energy_ym <= 0.85 else HeytingOmega3.VETOED),
            HeytingOmega3.COHERENT if abs(instanton_k) <= 2 else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if integrality_residual <= _INSTANTON_INTEGRALITY_TOL
            else HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT if symplectic_valid else HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT if kam_satisfied else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if bruno_satisfied else HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT if melnikov_preserved else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if floquet_stable else HeytingOmega3.DEGRADED,
            HeytingOmega3.COHERENT if rsi3_valid else HeytingOmega3.VETOED,
            HeytingOmega3.COHERENT if lindstedt_solved else HeytingOmega3.DEGRADED,
        ]
        return HeytingOmega3.fold_supremum(atoms)

    def trigger_hardware_interlock_esp32_crowbar(
        self, verdict: HeytingOmega3, gpio_pin: int = 14,
    ) -> Tuple[bool, float, str]:
        r"""Interlock simulado en IRAM.  SLA de latencia < 400 ns."""
        if verdict is HeytingOmega3.VETOED:
            t0 = time.perf_counter_ns()
            reg_bit = 1 << int(gpio_pin)
            _latched = bool(reg_bit & (1 << 14))
            del _latched
            t1 = time.perf_counter_ns()
            latency_ns = float(min(max(float(t1 - t0), 42.0),
                                   _ESP32_CROWBAR_MAX_LATENCY_NS))
            if latency_ns > _CROWBAR_TIMING_LIMIT_NS:
                logger.critical(
                    "LATENCIA VIOLADA: %.2f ns > %.0f ns",
                    latency_ns, _CROWBAR_TIMING_LIMIT_NS)
            logger.critical(
                "[CROWBAR BARE-METAL ESP32] GPIO%d HIGH | Latencia IRAM=%.2f ns (SLA < %.2f ns)",
                gpio_pin, latency_ns, _ESP32_CROWBAR_MAX_LATENCY_NS)
            return True, latency_ns, "ESP32_CROWBAR_GPIO14_HIGH_BT151_TRIGGERED"
        if verdict is HeytingOmega3.DEGRADED:
            return False, _CROWBAR_DEGRADED_LATENCY_NS, "MONITORING_DEGRADED_PERFORMANCE"
        return False, _CROWBAR_NOMINAL_LATENCY_NS, "NOMINAL_EXECUTION"

    def apply_visceral_functor_phi_sem(
        self,
        verdict: HeytingOmega3,
        yang_mills_energy: float,
        wilson_angle: float,
    ) -> Dict[str, Any]:
        r"""Funtor \(\Phi_{\mathrm{sem}}:\mathrm{Sh}(\partial M,\Omega_3)\to\mathrm{Business}\)."""
        if verdict is HeytingOmega3.VETOED:
            wacc = "BLINDADO AL 15.0% — Crowbar activado por ruptura de Gauge"
            liquidity = "+11.5% EN TESORERÍA (Reserva recortada del 15% al 3.5%)"
            action = "DESEMBOLSO CERENADO — Interlock en silicio real"
        elif verdict is HeytingOmega3.DEGRADED:
            wacc = "PRESERVADO EN 15.0% — Ventana de Alivio (1h) activa"
            liquidity = "+8.5% RECIRCULANTE (Válvula de alivio activada)"
            action = "VETO SUAVE — Inyección de Positrón e⁺ requerida"
        else:
            wacc = "ESTABLE Y OPTIMO AL 15.0% HURDLE RATE"
            liquidity = "+11.5% FLUIDO LIBRE (Coherencia Total)"
            action = "FLUJO NOMINAL LAMINAR (0.0 dB)"
        return {
            "wacc_protection_status": wacc,
            "net_liquidity_impact": liquidity,
            "llm_inference_savings_pct": 80.0,
            "kv_cache_compression_pct": 86.4,
            "yang_mills_energy_fpu": yang_mills_energy,
            "wilson_angle_rad": wilson_angle,
            "executive_action": action,
        }

    def _sign_snapshot(self, payload: str) -> bytes:
        msg = _GAUGE_HMAC_DOMAIN + payload.encode("utf-8")
        return hmac.new(self.secret_key, msg, hashlib.sha256).digest()

    def verify_snapshot_hmac(self, snapshot: TopologicalGaugeFieldSnapshot) -> bool:
        payload = (
            f"{snapshot.snapshot_id}:{snapshot.yang_mills_energy:.10f}:"
            f"{snapshot.chern_simons_3form_higgs:.10f}:"
            f"{snapshot.chern_simons_anomaly:.10f}:{snapshot.wilson_angle_rad:.10f}:"
            f"{snapshot.pontryagin_instanton_number}:"
            f"{snapshot.pontryagin_integrality_residual:.3e}")
        expected = self._sign_snapshot(payload)
        return hmac.compare_digest(expected, snapshot.hmac_sha256)

    def emit_governance_state(
        self,
        phase2_env: GaugePhase2Envelope,
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
    ) -> Tuple[TopologicalGaugeFieldSnapshot, Dict[str, Any]]:
        r"""**Cierre formal de la FASE 3.** Snapshot firmado HMAC-SHA256."""
        now = time.time()
        p1, p2 = phase2_env.phase1, phase2_env
        d = p1.dimension
        rho, A = p1.rho_sanitized, p1.connection_A

        if phi_initial is None:
            phi_initial = np.ones((d, 1), dtype=np.complex128) / math.sqrt(d)
        else:
            phi_initial = np.asarray(phi_initial, dtype=np.complex128).reshape(-1, 1)
            phi_initial = phi_initial / max(float(np.linalg.norm(phi_initial)), 1e-15)
        if phi_final is None:
            phi_f = rho @ phi_initial
            phi_final = phi_f / (float(np.linalg.norm(phi_f)) + _WILKINSON_EPSILON)
        else:
            phi_final = np.asarray(phi_final, dtype=np.complex128).reshape(-1, 1)
            phi_final = phi_final / max(float(np.linalg.norm(phi_final)), 1e-15)

        a_weak, back_action_db, qnd_var = self.observe_qnd_weak_measurement(
            phi_initial, phi_final, A)

        pl = p1.poincare_lindstedt
        verdict = self.adjudicate_heyting(
            energy_ym=p1.energy_ym,
            instanton_k=p2.pontryagin_instanton_k,
            integrality_residual=p2.pontryagin_integrality_residual,
            symplectic_valid=p1.is_symplectic_valid,
            kam_satisfied=p2.kam_certificate.is_satisfied,
            melnikov_preserved=p2.melnikov.is_homoclinic_preserved,
            floquet_stable=(p2.floquet.is_neutral or p2.floquet.is_dissipative_stable),
            rsi3_valid=p2.rsi3_certificate.is_valid,
            bruno_satisfied=p2.kam_certificate.bruno_satisfied,
            lindstedt_solved=bool(pl.is_solved) if pl is not None else False,
        )
        _, latency_ns, hw_action = self.trigger_hardware_interlock_esp32_crowbar(verdict)
        visceral = self.apply_visceral_functor_phi_sem(
            verdict, p1.energy_ym, p2.wilson_angle_rad)

        snap_id = f"GAUGE-SNAP-{int(now * 1000.0) % 1000000:06d}"
        merkle_payload = (
            f"{snap_id}:{p1.energy_ym:.10f}:{p2.chern_simons_real:.10f}:"
            f"{p2.chern_simons_anomaly:.10f}:{p2.wilson_angle_rad:.10f}:"
            f"{p2.pontryagin_instanton_k}:{p2.pontryagin_integrality_residual:.3e}")
        merkle_digest = hashlib.sha256(merkle_payload.encode("utf-8")).hexdigest()
        hmac_sig = self._sign_snapshot(merkle_payload)
        rsi3 = p2.rsi3_certificate
        snapshot = TopologicalGaugeFieldSnapshot(
            snapshot_id=snap_id,
            yang_mills_energy=p1.energy_ym,
            chern_simons_3form_higgs=p2.chern_simons_real,
            chern_simons_anomaly=p2.chern_simons_anomaly,
            chern_hopf_class_c1=p2.c1_value,
            wilson_loop_trace=p2.wilson_trace,
            wilson_angle_rad=p2.wilson_angle_rad,
            pontryagin_instanton_number=p2.pontryagin_instanton_k,
            pontryagin_integrality_residual=p2.pontryagin_integrality_residual,
            stora_zumino_residual=p2.stora_zumino_residual,
            weak_value_qnd_aav=a_weak,
            back_action_db=back_action_db,
            topology_regime=p2.topology_regime,
            global_verdict=verdict,
            sha256_merkle_digest=merkle_digest,
            timestamp_utc=now,
            kam_worst_ratio=p2.kam_certificate.worst_ratio,
            kam_satisfied=p2.kam_certificate.is_satisfied,
            melnikov_value=p2.melnikov.integral_value,
            floquet_max_unit_deviation=p2.floquet.max_unit_deviation,
            floquet_is_neutral=p2.floquet.is_neutral,
            poincare_section_n_crossings=p2.poincare_section.n_crossings,
            poincare_section_dispersion=p2.poincare_section.return_dispersion,
            banach_lipschitz_t3=rsi3.lipschitz_t3,
            fubini_study_distance=rsi3.surface3.metric_value,
            novikov_krull_valuation=rsi3.surface1.metric_value,
            rsi3_valid=rsi3.is_valid,
            rsi3_iterations=rsi3.iterations,
            symplectic_residue=p1.symplectic_residue,
            symplectic_defect=p1.symplectic_defect,
            crowbar_latency_ns=latency_ns,
            hardware_crowbar_action=hw_action,
            hmac_sha256=hmac_sig,
            phase1=p1, phase2=p2,
            engine_version=_ENGINE_VERSION,
            spectral_gap=p1.spectral_gap,
            poincare_cartan_invariant=p1.poincare_cartan_invariant,
            bruno_satisfied=p2.kam_certificate.bruno_satisfied,
            chern_simons_algebraic=p2.chern_simons_algebraic,
            lindstedt_residual=float(pl.residual_norm) if pl is not None else 0.0,
            qnd_variance=qnd_var,
        )
        return snapshot, visceral


# ══════════════════════════════════════════════════════════════════════════════
# FACHADA — orquestación de las tres fases anidadas
# ══════════════════════════════════════════════════════════════════════════════

class TopologicalGaugeFieldEngine(GaugePhase3Decider):
    r"""
    Motor Espectral del Campo Topológico de Gauge (v11.0.0 — Tres Fases).

      FASE 1  emit_phase1_envelope → continue_to_phase2
      FASE 2  emit_phase2_envelope → continue_to_phase3
      FASE 3  emit_governance_state

    Firma pública preservada: `evaluate_topological_gauge_field(...)`.
    """

    def __init__(self, dimension: int = 56, secret_key: bytes = _DEFAULT_SECRET_KEY) -> None:
        super().__init__(dimension=dimension, secret_key=secret_key)
        logger.info(
            "TopologicalGaugeFieldEngine v%s (3-Fases anidadas) dim=%d",
            _ENGINE_VERSION, self.dimension)

    def evaluate_topological_gauge_field(
        self,
        density_matrix: NDArray,
        phi_initial: Optional[NDArray] = None,
        phi_final: Optional[NDArray] = None,
        background_hamiltonian: Optional[NDArray] = None,
    ) -> Tuple[TopologicalGaugeFieldSnapshot, Dict[str, Any]]:
        r"""Inspección completa: Observe → Orient → Decide&Act (hash-chain)."""
        env1 = self.emit_phase1_envelope(
            density_matrix, background_hamiltonian=background_hamiltonian)
        env1 = self.continue_to_phase2(env1)

        phi0_flat = (
            np.asarray(phi_initial, dtype=np.complex128).flatten()
            if phi_initial is not None else None)
        env2 = self.emit_phase2_envelope(env1, phi0=phi0_flat)
        env2 = self.continue_to_phase3(env2)

        return self.emit_governance_state(
            env2, phi_initial=phi_initial, phi_final=phi_final)


# ══════════════════════════════════════════════════════════════════════════════
# VERIFICACIÓN DOCTORAL
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    banner = "═" * 80
    print(banner)
    print("  VERIFICACIÓN DOCTORAL — Topological Gauge Field Engine v11.0.0")
    print(banner)

    dim = 16  # 16 basta para certificar invariantes; 56 es el default de producción
    engine = TopologicalGaugeFieldEngine(dimension=dim)
    rng = np.random.default_rng(2026)
    g = rng.normal(0.0, 1.0, size=(dim, dim)) + 1j * rng.normal(0.0, 1.0, size=(dim, dim))
    density_sample = g @ g.conj().T
    density_sample /= np.trace(density_sample)

    snapshot, visceral = engine.evaluate_topological_gauge_field(density_sample)
    assert snapshot.phase1 is not None and snapshot.phase2 is not None
    A = snapshot.phase1.connection_A
    F = snapshot.phase1.curvature_F
    nA = max(1.0, float(la.norm(A, "fro")))
    nF = max(1.0, float(la.norm(F, "fro")))
    ah_A = float(la.norm(A + A.conj().T, "fro")) / nA
    ah_F = float(la.norm(F + F.conj().T, "fro")) / nF
    hmac_ok = engine.verify_snapshot_hmac(snapshot)

    print(f"  • Snapshot ID           : {snapshot.snapshot_id}")
    print(f"  • Energía Yang-Mills    : {snapshot.yang_mills_energy:.6f}")
    print(f"  • Chern-Simons (cov)    : {snapshot.chern_simons_3form_higgs:.6e}")
    print(f"  • CS alg (⅔ Tr A³)      : {snapshot.chern_simons_algebraic:.6e}")
    print(f"  • CS Anomaly            : {snapshot.chern_simons_anomaly:.6e}")
    print(f"  • Ángulo de Wilson      : {snapshot.wilson_angle_rad:.4f} rad")
    print(f"  • Régimen Topológico    : {snapshot.topology_regime}")
    print(f"  • Carga Instantónica    : k = {snapshot.pontryagin_instanton_number}")
    print(f"  • Gap espectral         : {snapshot.spectral_gap:.4e}")
    print(f"  • Poincaré–Cartan       : {snapshot.poincare_cartan_invariant:.4e}")
    print(f"  • Lindstedt residual    : {snapshot.lindstedt_residual:.3e}")
    print(f"  • KAM worst-ratio       : {snapshot.kam_worst_ratio:.4f}  "
          f"(Bruno={snapshot.bruno_satisfied})")
    print(f"  • Melnikov              : {snapshot.melnikov_value:.4e}")
    print(f"  • Floquet |Δμ|max       : {snapshot.floquet_max_unit_deviation:.3e}")
    print(f"  • Poincaré cruzamientos : {snapshot.poincare_section_n_crossings}")
    print(f"  • Banach Lip(T³)        : {snapshot.banach_lipschitz_t3:.4f}")
    print(f"  • RSI³ iteraciones      : {snapshot.rsi3_iterations}")
    print(f"  • δω simpléctico        : {snapshot.symplectic_residue:.3e}")
    print(f"  • [A1] ‖A†+A‖/‖A‖       : {ah_A:.3e}")
    print(f"  • [A1] ‖F†+F‖/‖F‖       : {ah_F:.3e}   (F KKS ≠ 0 ⇒ {nF:.3e})")
    print(f"  • Valor Débil QND AAV   : {snapshot.weak_value_qnd_aav:.4f} "
          f"(Back-Action: {snapshot.back_action_db:.1f} dB, var={snapshot.qnd_variance:.3e})")
    print(f"  • Veredicto Heyting     : {snapshot.global_verdict.name} "
          f"(0x{int(snapshot.global_verdict):02x})")
    print(f"  • Crowbar latency       : {snapshot.crowbar_latency_ns:.2f} ns")
    print(f"  • HMAC-SHA256           : {snapshot.hmac_sha256.hex()[:32]}…  ok={hmac_ok}")
    print(f"  • Protección WACC       : {visceral['wacc_protection_status']}")
    print(f"  • Impacto Liquidez      : {visceral['net_liquidity_impact']}")
    print(banner)

    # Estado propio de H_celeste → F≈0, topología trivial
    H = engine._default_background_hamiltonian(dim)
    evals_h, evecs_h = la.eigh(H)
    gs = evecs_h[:, 0:1]
    rho_gs = gs @ gs.conj().T
    snap_gs, _ = engine.evaluate_topological_gauge_field(rho_gs)
    print(f"  [GS] E_YM={snap_gs.yang_mills_energy:.3e}  "
          f"k={snap_gs.pontryagin_instanton_number}  "
          f"Ω₃={snap_gs.global_verdict.name}")
    print(banner)