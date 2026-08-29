# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Boundary Ring Sheaf (Haz de Anillos Topológicos Localizados)        ║
║ Ruta   : app/wisd/boundary_ring_sheaf.py                                     ║
║ Versión: 1.1.0-Doctoral-PortHamiltonian-Lindblad-Novikov-ESP32-Secure        ║
║                                                                              ║
║ SINOPSIS MATEMÁTICA Y EXERGÉTICA DE LA FRONTERA ABIERTA (∂M ≠ ∅):            ║
║ Este módulo implementa el reactor algebraico y dinámico acoplado no          ║
║ conmutativo sobre el haz de anillos topológicos localizados:                 ║
║                                                                              ║
║                     Sh(∂M, \mathcal{R})                                      ║
║                                                                              ║
║ Unifica síncronamente los observables térmicos (telemetría Langevin) y       ║
║ topológicos (auditoría homológica de Smith, Choi y Bell-CHSH) en un sistema  ║
║ Port-Hamiltoniano con disipación (IDA-PBC) que metaboliza alucinaciones      ║
║ semánticas mediante la ecuación maestra cuántica de Lindblad-GKSL en Fock.   ║
╚══════════════════════════════════════════════════════════════════════════════╝

================================════════════════════════════════════════════════
I. DEFINICIONES CATEGORIALES Y TERMODINÁMICAS (Física de la Conducción de Rings)
================================════════════════════════════════════════════════

Definición 1 (La Variedad con Estructura de Haz de Anillos Localizados):
  Sea $(\mathcal{M}, \, G_{\mu\nu})$ una variedad Riemanniana orientable de dimensión $N$
  con frontera compacta no nula $\partial\mathcal{M} \neq \emptyset$. Definimos el haz de anillos
  topológicos localizados $\mathbf{Sh}(\partial\mathcal{M}, \, \mathcal{R})$ sobre el Anillo
  de Novikov ultramétrico $\Lambda_{\mathrm{Nov}}$, cuyas variables formales parametrizan
  las deformaciones simplécticas inducidas por perturbaciones de frontera:
  $$\mathcal{R}_{\partial M} \cong \Lambda_{\mathrm{Nov}} = \left\{ \sum_{i=0}^{\infty} a_i T^{\lambda_i} \ \middle|\ a_i \in \mathbb{C}, \ \lambda_i \in \mathbb{R}, \ \lambda_i \to \infty \right\}$$

Definición 2 (El Vector de Estado Covariante de de Rham):
  El acoplamiento mutuo de los satélites se representa como una sección local de la frontera
  $\mathbf{\Psi} = (\mathcal{Q}, \, \mathcal{A})^\top$ que habita en un módulo sobre el anillo
  $\mathcal{R}_{\partial M}$, donde:
  - $\mathcal{Q}$ representa el observable térmico-Langevin de telemetría (entropía de Shannon $H_{\mathrm{ext}}$).
  - $\mathcal{A}$ representa el observable topológico-causal de auditoría (torsión homológica de Smith,
    matriz de Choi y correlación cuántica de Bell-CHSH).

Definición 3 (El Reactor Port-Hamiltoniano Disipativo):
  La evolución de la trayectoria geodésica sobre el manifold de control satisface el principio
  de pasividad de Lyapunov modificado mediante un acoplamiento antisimétrico de calibre:
  $$\begin{bmatrix} d\mathcal{Q} \\ d\mathcal{A} \end{bmatrix} = \left( \mathcal{J}(\mathbf{\Psi}) - \mathcal{R}(\mathbf{\Psi}) \right) \nabla \mathcal{H}(\mathbf{\Psi}) dt + \begin{bmatrix} \xi_{\mathrm{ext}}(t) dt \\ \mathbf{0} \end{bmatrix}$$
  Donde $\mathcal{J}(\mathbf{\Psi}) = -\mathcal{J}(\mathbf{\Psi})^\top = \begin{bmatrix} 0 & -\chi \\ \chi & 0 \end{bmatrix}$ es la matriz antisimétrica de
  interconexión giroscópica de Lorentz (modulada por la característica de Euler-Poincaré $\chi = \beta_0 - \beta_1$)
  y $\mathcal{R}(\mathbf{\Psi}) = \operatorname{diag}(r_{\mathcal{Q}}, \, r_{\mathcal{A}}) \succeq \mathbf{0}$ es el tensor de amortiguamiento de
  Rayleigh que satisface de forma local las leyes de disipación termodinámica.

Definición 4 (Metabolismo de Alucinaciones en el Espacio de Fock):
  Las alucinaciones semánticas del LLM se tratan como polaritones alucinatorios (cuasipartículas
  excitadas parasitarias) en el espacio de Fock multi-cuerpo $\mathcal{F}(\mathcal{H}) = \bigoplus \Lambda^k \mathcal{H}$.
  El reactor fuerza su decaimiento hacia el vacío coherente resolviendo la ecuación disipativa cuántica de Lindblad-GKSL:
  $$\frac{d\rho_{\mathrm{sem}}}{dt} = -i[\mathcal{H}_{\mathrm{coupled}}, \, \rho_{\mathrm{sem}}] + L\rho_{\mathrm{sem}} L^\dagger - \frac{1}{2} \left\{ L^\dagger L, \, \rho_{\mathrm{sem}} \right\}$$
  Donde el operador de salto $L = \sqrt{\Gamma(\Xi_{\mathrm{leak}}, \, \operatorname{Tor})} \cdot a_j$ se encuentra modulado por
  la fuga exergética de telemetría y la presencia de torsión homológica.

================================════════════════════════════════════════════════
II. AXIOMATIZACIÓN DE LA ADUANA INMUNITARIA DE CONTORNO (Leyes de Conservación)
================================════════════════════════════════════════════════

Axioma I (Principio de Pasividad Causal de Choi):
  El canal cuántico de inyección $\mathcal{E}$ entre la frontera exógena y el interior de la fortaleza
  debe conservar la probabilidad cuántica y satisfacer la condición de positividad completa
  y preservación de traza (CPTP) en el espacio de Fock:
  $$\lambda_{\min}(C_{\mathcal{E}}) \ge -\varepsilon_{\mathrm{Wilkinson}} \quad \land \quad \operatorname{Tr}_2(C_{\mathcal{E}}) = \mathbf{I}_{\mathrm{input}}$$
  Donde $\varepsilon_{\mathrm{Wilkinson}} = 10^{-12}$ es la cota elástica de imprecisión en punto flotante.

Axioma II (Axioma de de Rham-Smith de Nulidad de Torsión):
  Toda relación de contorno discreta sobre el complejo simplicial de frontera $\partial K$ debe ser resoluble
  de manera única sobre el anillo de enteros $\mathbb{Z}$, garantizando que la homología de contorno carezca de torsión:
  $$\operatorname{Tor}\left(H_{k-1}(\partial K; \, \mathbb{Z})\right) \equiv \mathbf{0} \quad \Longleftrightarrow \quad d_i = 1 \quad \forall d_i > 0$$
  Donde $d_i$ son los coeficientes diagonales de la Forma Normal de Smith (SNF) calculados en FPU.

Axioma III (Teorema de Actuación Ciber-Física Determinista en Silicio):
  Ante el colapso del retículo de Heyting al Supremo terminal VETOED ($\top$), la subrutina local en C++
  isVerdictCoherent() del microcontrolador ESP32 perimetral despacha la Rutina de Servicio de Interrupción
  (ISR) en IRAM, conmutando el pin físico GPIO14 a HIGH en menos de $400\text{ ns}$:
  $$t_{\mathrm{actuation}} \le \tau_{\mathrm{IRAM}} = 400\text{ ns} \quad \implies \quad \mathtt{GPIO14} \mapsto \mathtt{HIGH}$$
  Disparando síncronamente el tiristor rápido BT151 (Crowbar) para paralizar mecánicamente la obra civil.

================================════════════════════════════════════════════════
III. INVARIANTES ESPECTRALES Y METROLÓGICOS DE WILKINSON (FPU Secure)
================================════════════════════════════════════════════════

Invariante I (Estabilidad Asintótica de Lyapunov):
  Bajo perturbaciones estocásticas extremas de Langevin $\xi_{\mathrm{ext}}(t)$, la disipación térmica local
  y la divergencia exergética satisfacen la contracción asintótica estricta de Lyapunov:
  $$\dot{\mathcal{H}}(\mathbf{\Psi}) \le \tau_{\mathrm{Lyapunov}} \quad \text{con} \quad \tau_{\mathrm{Lyapunov}} = 10^{-12}$$

Invariante II (Confinamiento de Tsirelson contra la Cartelización):
  La correlación no local clásica de ofertas se encuentra estrictamente confinada por la cota cuántica
  relativista de Tsirelson para eludir la inestabilidad de redondeo numérico en la mantisa de la FPU:
  $$\mathcal{B}_{\mathrm{CHSH}} \le 2\sqrt{2} + \varepsilon_{\mathrm{Wilkinson}} \quad \text{con} \quad \varepsilon_{\mathrm{Wilkinson}} = 10^{-15}$$

Invariante III (Inversión Espectral de Higham-Tikhonov):
  La purificación del tensor de conductividad exógena bruto $\mathcal{K}$ mediante Weyl-Toeplitz y Higham SVD
  garantiza que el espectro purificado sea estrictamente definido positivo e invertible en la FPU:
  $$\lambda_{\min}(\tilde{\mathcal{K}}) \ge \varepsilon_{\mathrm{Wilkinson}} \quad \implies \quad \tilde{\mathcal{K}} \succ \mathbf{0}$$
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Final, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la


logger = logging.getLogger("APU.Physics.BoundaryRingSheaf")


# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTES METROLÓGICAS, ESPECTRALES Y DE SEGURIDAD
# ═══════════════════════════════════════════════════════════════════════════

_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_STABILITY_FLOOR: Final[float] = max(1e-12, _MACHINE_EPS)
_REL_RANK_MULT: Final[float] = 1.0e4
_HERMITIAN_VETO_REL: Final[float] = 1.0e-3
_TP_VETO_ABS: Final[float] = 1.0e-4
_TP_DEGRADED_ABS: Final[float] = 1.0e-10
_CHOI_MIN_EIG_VETO: Final[float] = -1.0e-4
_CHOI_MIN_EIG_DEGRADED: Final[float] = -1.0e-12
_RHO_TRACE_TOL: Final[float] = 1.0e-8
_RHO_PSD_TOL: Final[float] = 1.0e-12
_CYCLE_DEGRADED_REL: Final[float] = 1.0e-4
_CONDITION_CAP: Final[float] = 1.0e12
_HARD_LEAK_BASE: Final[float] = 1.0e5
_SOFT_LEAK_BASE: Final[float] = 1.0
_LINDBLAD_VETO_FLOOR: Final[float] = 0.1
_LINDBLAD_DEGRADED_FLOOR: Final[float] = 0.5
_LINDBLAD_GAMMA0: Final[float] = 2.0
_LINDBLAD_TORSION_WEIGHT: Final[float] = 10.0
_LINDBLAD_LEAK_WEIGHT: Final[float] = 4.0
_LINDBLAD_DEFAULT_STEPS: Final[int] = 10
_LINDBLAD_DEFAULT_DT: Final[float] = 0.05

_SNF_MAX_OPS: Final[int] = 200_000
_SNF_MAX_PASSES: Final[int] = 1_000
_SNF_INVARIANT_GUARD: Final[int] = 10_000
_SNF_EXACT_MAX_ENTRIES: Final[int] = 16_384

_CROWBAR_IRAM_LATENCY_NS: Final[float] = 400.0
_CROWBAR_JITTER_STD_NS: Final[float] = 3.2
_CROWBAR_LATENCY_MIN_NS: Final[float] = 380.0
_CROWBAR_LATENCY_MAX_NS: Final[float] = 420.0
_GPIO_CROWBAR_PIN: Final[int] = 14

_BELL_CLASSICAL_BOUND: Final[float] = 2.0
_TSIRELSON_BOUND: Final[float] = 2.0 * math.sqrt(2.0)

_SCHEMA_VERSION: Final[str] = (
    "2.1.0-Doctoral-Smith-Lindblad-PortHamiltonian-CPTP-Heyting-Nested"
)

_DOMAIN_ENGINE_SESSION: Final[bytes] = b"APU/BOUNDARY-RING-SHEAF/ENGINE/SESSION/v2"
_DOMAIN_ENGINE_TELEMETRY: Final[bytes] = b"APU/BOUNDARY-RING-SHEAF/ENGINE/TELEMETRY/v2"

_HEYTING_COHERENT: Final[str] = "COHERENT"
_HEYTING_DEGRADED: Final[str] = "DEGRADED"
_HEYTING_VETOED: Final[str] = "VETOED"

_HEYTING_ORDER: Final[dict] = {
    _HEYTING_VETOED: 0,
    _HEYTING_DEGRADED: 1,
    _HEYTING_COHERENT: 2,
}


# ═══════════════════════════════════════════════════════════════════════════
# ÁLGEBRA DE HEYTING LINEAL Ω₃ (clasificador grosero del topos de borde)
# ═══════════════════════════════════════════════════════════════════════════


def _heyting_meet(*atoms: str) -> str:
    """Ínfimo en la cadena VETOED ≤ DEGRADED ≤ COHERENT."""
    acc = _HEYTING_COHERENT
    acc_ord = _HEYTING_ORDER[acc]
    for atom in atoms:
        key = str(atom).strip().upper()
        ord_a = _HEYTING_ORDER.get(key, 0)
        if ord_a < acc_ord:
            acc = key if key in _HEYTING_ORDER else _HEYTING_VETOED
            acc_ord = _HEYTING_ORDER.get(acc, 0)
    return acc


def _heyting_join(a: str, b: str) -> str:
    """Supremo en Ω₃."""
    oa = _HEYTING_ORDER.get(str(a).strip().upper(), 0)
    ob = _HEYTING_ORDER.get(str(b).strip().upper(), 0)
    return a if oa >= ob else b


def _heyting_implies(a: str, b: str) -> str:
    """Implicación de Heyting sobre la cadena: 1 si a ≤ b, else b."""
    oa = _HEYTING_ORDER.get(str(a).strip().upper(), 0)
    ob = _HEYTING_ORDER.get(str(b).strip().upper(), 0)
    if oa <= ob:
        return _HEYTING_COHERENT
    return str(b).strip().upper() if str(b).strip().upper() in _HEYTING_ORDER else _HEYTING_VETOED


# ═══════════════════════════════════════════════════════════════════════════
# ESTADOS Y CERTIFICADOS INMUTABLES
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class CoupledBoundaryState:
    r"""
    Vector de estado covariante \(\mathbf{\Psi} = (\mathcal{Q}, \mathcal{A})\)
    sobre el haz de anillos topológicos localizados.

    \(\mathcal{Q}\) agrupa observables térmicos/exergéticos;
    \(\mathcal{A}\) agrupa invariantes homológicos y cuánticos de auditoría.
    """

    Q_thermal_entropy: float
    Q_exergy_leak: float
    A_smith_torsion: List[int]
    A_choi_min_eigenvalue: float
    A_bell_chsh: float
    hamiltonian_energy: float

    condition_number: float = math.inf
    choi_trace_preservation_error: float = math.nan
    bell_tsirelson_slack: float = math.nan

    # Extensiones v2.1 (compatibles por keywords).
    spectral_gap: float = math.nan
    algebraic_connectivity: float = math.nan
    spectral_radius: float = math.nan
    choi_hermitian_residual: float = math.nan
    kirchhoff_cycle_residual: float = math.nan
    betti_0: int = 0
    betti_1: int = 0
    homology_rank: int = 0
    euler_from_chain: int = 0
    port_hamiltonian_dissipation: float = math.nan


@dataclass(frozen=True, slots=True)
class MetabolicCertificate:
    r"""Certificado inmutable del metabolismo de alucinaciones semánticas."""

    heyting_verdict: str
    coupled_state: CoupledBoundaryState
    lindblad_density_trace: float
    lindblad_decay_rate: float
    hardware_interlock_fired: bool
    actuation_latency_ns: float
    cryptographic_seal: str

    schema_version: str = _SCHEMA_VERSION
    choi_dimension: int = 0
    choi_completely_positive: bool = False
    choi_trace_preserving: bool = False
    bell_coherent: bool = False
    rho_minimum_eigenvalue: float = math.nan
    rho_purity: float = math.nan
    condition_number: float = math.inf
    diagnostics: Tuple[str, ...] = field(default_factory=tuple)

    # Extensiones v2.1.
    session_digest: str = ""
    choi_trace: float = math.nan
    rho_cp_channel: bool = False
    energy_balance_residual: float = math.nan
    tsirelson_slack: float = math.nan


# ═══════════════════════════════════════════════════════════════════════════
# DOSSIERS INTERNOS DE FASE
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True, slots=True)
class ConductivityAudit:
    """Invariantes espectrales de la conductancia de frontera K."""

    K_h: np.ndarray
    condition_number: float
    passive_penalty: float
    min_eigenvalue: float
    spectral_gap: float
    algebraic_connectivity: float
    spectral_radius: float
    hermitian_residual: float
    diagnostics: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Phase1Dossier:
    """Expediente crudo canonizado y congelado en la FASE 1 · OBSERVE."""

    payload: bytes
    entropy: float
    K_h: np.ndarray
    condition_number: float
    exergy_leak: float
    B: Tuple[Tuple[int, ...], ...]
    C: np.ndarray
    choi_dimension: int
    bell: Tuple[float, float, float, float]
    euler_characteristic: int
    diagnostics: Tuple[str, ...]
    spectral_gap: float = 1.0
    algebraic_connectivity: float = 0.0
    spectral_radius: float = 0.0
    conductivity_min_eig: float = 0.0
    conductivity_hermitian_residual: float = 0.0
    choi_hermitian_residual: float = 0.0
    metric_shift_trace: float = 0.0


@dataclass(frozen=True, slots=True)
class HomologicalAudit:
    """Invariantes del complejo truncado ℤⁿ --B--> ℤᵐ."""

    invariants: Tuple[int, ...]
    torsion: Tuple[int, ...]
    has_torsion: bool
    numerical_rank: int
    betti_0: int
    betti_1: int
    euler_from_chain: int
    cycle_annihilation_residual: float
    diagnostics: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ChoiCPTPAudit:
    """Auditoría CPTP vía isomorfismo de Choi–Jamiołkowski."""

    min_eig: float
    tp_diff: float
    is_cp: bool
    is_tp: bool
    is_cptp: bool
    trace: float
    hermitian_residual: float
    diagnostics: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BellCHSHAudit:
    """Auditoría Bell–CHSH (cota clásica y cota de Tsirelson)."""

    chsh_standard: float
    chsh_conservative: float
    is_bell_coherent: bool
    tsirelson_slack: float
    exceeds_classical: bool
    diagnostics: Tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Phase2Dossier:
    """Expediente auditado en la FASE 2 · ORIENT/DECIDE."""

    phase1: Phase1Dossier
    torsion_coefficients: List[int]
    has_torsion: bool
    choi_minimum_eigenvalue: float
    choi_trace_preservation_error: float
    choi_trace: float
    is_completely_positive: bool
    is_trace_preserving: bool
    is_cptp: bool
    bell_chsh_standard: float
    bell_chsh_conservative: float
    is_bell_coherent: bool
    hamiltonian_energy: float
    heyting_verdict: str
    diagnostics: Tuple[str, ...]
    betti_0: int = 0
    betti_1: int = 0
    homology_rank: int = 0
    euler_from_chain: int = 0
    kirchhoff_cycle_residual: float = 0.0
    choi_hermitian_residual: float = 0.0
    tsirelson_slack: float = 0.0
    port_hamiltonian_dissipation: float = 0.0
    energy_lyapunov_drop: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# ARITMÉTICA ENTERA EXACTA SOBRE EL DIP ℤ
# ═══════════════════════════════════════════════════════════════════════════


def ext_gcd(a: int, b: int) -> Tuple[int, int, int]:
    r"""
    Algoritmo extendido de Euclides, forma iterativa (Bezout).

    Retorna \((g, x, y)\) tales que
        \(a x + b y = g = \gcd(a,b)\),
    con \(g \ge 0\). Para \((0,0)\) se adopta la convención \((0,0,1)\).

    Complejidad: \(O(\log \max(|a|,|b|))\) en número de pasos;
    el tamaño de bits de los coeficientes de Bézout es \(O(\log |a|+|b|)\).
    """
    a_int = int(a)
    b_int = int(b)

    if b_int == 0:
        g = abs(a_int)
        if a_int == 0:
            return 0, 0, 1
        return g, (1 if a_int > 0 else -1), 0

    old_r, r = a_int, b_int
    old_s, s = 1, 0
    old_t, t = 0, 1

    while r != 0:
        q = old_r // r
        old_r, r = r, old_r - q * r
        old_s, s = s, old_s - q * s
        old_t, t = t, old_t - q * t

    if old_r < 0:
        return -old_r, -old_s, -old_t
    return old_r, old_s, old_t


def _gcd_int(a: int, b: int) -> int:
    """gcd no negativo sobre ℤ (Python, precisión arbitraria)."""
    a, b = abs(int(a)), abs(int(b))
    while b:
        a, b = b, a % b
    return a


def _lcm_int(a: int, b: int) -> int:
    """lcm no negativo; 0 si alguno es 0."""
    a, b = abs(int(a)), abs(int(b))
    if a == 0 or b == 0:
        return 0
    return (a // _gcd_int(a, b)) * b


# ═══════════════════════════════════════════════════════════════════════════
# REACTOR SOBERANO
# ═══════════════════════════════════════════════════════════════════════════


class BoundaryRingSheaf:
    r"""
    El Haz de Anillos Topológicos de Frontera \(Sh(\partial M, \mathcal{R})\).

    Gobierna el acoplamiento Port-Hamiltoniano y metaboliza alucinaciones
    semánticas forzando su decaimiento disipativo en Fock (canal GKSL).

    El ciclo público `execute_metabolic_cycle` se resuelve como cadena
    anidada de morfismos:

        execute_metabolic_cycle()
          └─ _phase1_observe_and_normalize()          # unidad de FASE 2
               └─ _phase2_audit_and_orient()          # unidad de FASE 3
                    └─ _phase3_metabolize_and_certify()

    Ante cualquier excepción no recuperable se emite un certificado
    fail-closed `VETOED` con interlock activado.
    """

    def __init__(
        self,
        dimension_n: int,
        safety_margin: float = 1.0,
        *,
        rng_seed: Optional[int] = None,
    ) -> None:
        """
        Inicializa el reactor de frontera.

        Args:
            dimension_n: Dimensión del frente / número de nodos de ∂M.
            safety_margin: Margen elástico para cotas exergéticas.
            rng_seed: Semilla opcional para reproducibilidad del interlock.

        Raises:
            ValueError: Si los parámetros físicos/numéricos son inválidos.
        """
        if isinstance(dimension_n, bool) or not isinstance(dimension_n, (int, np.integer)):
            raise ValueError("dimension_n debe ser un entero estrictamente positivo.")
        if int(dimension_n) <= 0:
            raise ValueError("La dimensión de la frontera debe ser estrictamente positiva.")

        if isinstance(safety_margin, bool) or not math.isfinite(float(safety_margin)) or float(safety_margin) <= 0.0:
            raise ValueError("safety_margin debe ser finito y estrictamente positivo.")

        self._n: Final[int] = int(dimension_n)
        self._safety_margin: Final[float] = float(safety_margin)
        self._reg: Final[float] = 1e-15
        self._rng: Final[np.random.Generator] = np.random.default_rng(rng_seed)
        self._hard_leak_ceiling: Final[float] = _HARD_LEAK_BASE * self._safety_margin
        self._soft_leak_ceiling: Final[float] = _SOFT_LEAK_BASE * self._safety_margin

    # ═══════════════════════════════════════════════════════════════════════
    # UTILIDADES DE HASH CANÓNICO Y ÁLGEBRA MENOR
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _hash_update_domain(sha: Any, domain: bytes) -> None:
        """Actualiza SHA-256 con separación de dominio length-prefixed."""
        sha.update(len(domain).to_bytes(8, "little", signed=False))
        sha.update(domain)

    @staticmethod
    def _hash_update_bytes(sha: Any, data: bytes) -> None:
        """Actualiza SHA-256 con bloque length-prefixed."""
        sha.update(len(data).to_bytes(8, "little", signed=False))
        sha.update(data)

    @classmethod
    def _hash_update_array(cls, sha: Any, name: str, array: np.ndarray) -> None:
        """
        Actualiza SHA-256 con arreglo numpy canónico.

        Incluye nombre, ndim, forma, dtype.str (endianness explícita) y
        bytes C-contiguos para evitar colisiones por re-interpretación.
        """
        cls._hash_update_bytes(sha, name.encode("utf-8"))

        arr = np.ascontiguousarray(array)

        sha.update(int(arr.ndim).to_bytes(4, "little", signed=False))
        sha.update(len(arr.shape).to_bytes(4, "little", signed=False))
        for dim in arr.shape:
            sha.update(int(dim).to_bytes(8, "little", signed=False))

        cls._hash_update_bytes(sha, arr.dtype.str.encode("ascii"))
        sha.update(arr.tobytes(order="C"))

    @staticmethod
    def _reject_bool(value: Any, name: str) -> None:
        if isinstance(value, bool) or isinstance(value, np.bool_):
            raise ValueError(f"{name} no puede ser booleano.")

    @staticmethod
    def _freeze_ndarray(array: np.ndarray, dtype: Any) -> np.ndarray:
        """Copia C-contigua, dtype canónico y write-protect."""
        frozen = np.array(array, dtype=dtype, copy=True, order="C")
        frozen.setflags(write=False)
        return frozen

    @staticmethod
    def _finite_float(value: Any, name: str) -> float:
        if isinstance(value, bool) or isinstance(value, np.bool_):
            raise ValueError(f"{name} no puede ser booleano.")
        try:
            x = float(value)
        except Exception as exc:
            raise ValueError(f"{name} no es escalar.") from exc
        if not math.isfinite(x):
            raise ValueError(f"{name} no es finito.")
        return x

    @staticmethod
    def _frobenius(arr: np.ndarray) -> float:
        val = float(np.linalg.norm(arr, ord="fro"))
        return val if math.isfinite(val) else math.inf

    def _safe_payload_bytes(self, payload: Any) -> bytes:
        """Convierte payload a bytes de forma segura; fallback vacío."""
        try:
            return self._phase1_validate_payload(payload)
        except Exception:
            return b""

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 1 · OBSERVE
    #
    # Responsabilidad:
    #   - Validar payload, K, B, Choi, Bell y Euler.
    #   - Canonizar matrices enteras y complejas.
    #   - Proyectar K al subespacio hermitiano de Mₙ (C*-álgebra).
    #   - Regularizar métrica G ∈ SPD ⊂ Simₙ si se suministra.
    #   - Calcular entropía de Shannon del payload (nats).
    #   - Auditar pasividad de Kirchhoff y fuga exergética.
    #   - Congelar tensores write-protected.
    #   - Transferir el expediente como unidad de la FASE 2.
    #
    # Cada método es un prefijo del siguiente; el último morfismo
    # `_phase1_observe_and_normalize` ES la continuación / inicio
    # formal de la FASE 2.
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _phase1_validate_payload(payload: Any) -> bytes:
        """Valida y normaliza payload a bytes inmutables."""
        if isinstance(payload, bytearray):
            return bytes(payload)

        if isinstance(payload, memoryview):
            return payload.tobytes()

        if not isinstance(payload, (bytes, bytearray)):
            raise TypeError("payload debe ser bytes, bytearray o memoryview.")

        return bytes(payload)

    @staticmethod
    def _phase1_payload_entropy(payload: bytes) -> float:
        r"""
        Entropía de Shannon del payload en nats.

        \[
            H[p] = -\sum_{b=0}^{255} p_b \log p_b
            \in [0, \log 256],
        \]
        con \(H=0\) si el payload es vacío o un único símbolo.
        Se usa \(\log\) natural. La finitud se exige de forma fail-closed.
        """
        n = len(payload)
        if n == 0:
            return 0.0

        counts = np.bincount(np.frombuffer(payload, dtype=np.uint8), minlength=256)
        counts = counts[counts > 0]
        if counts.size == 0:
            return 0.0

        probs = counts.astype(np.float64) / float(n)
        # clip numérico contra log(0) por underflow.
        probs = np.clip(probs, _MACHINE_EPS, 1.0)
        entropy = -float(np.sum(probs * np.log(probs)))

        if not math.isfinite(entropy):
            raise ValueError("Entropía del payload no finita.")

        # Cota teórica: 0 ≤ H ≤ log(256).
        return float(min(max(0.0, entropy), math.log(256.0) + 1e-12))

    def _phase1_canonical_euler(self, euler_characteristic: Any) -> int:
        """Valida la característica de Euler–Poincaré como entero finito."""
        self._reject_bool(euler_characteristic, "euler_characteristic")
        try:
            x = float(euler_characteristic)
        except Exception as exc:
            raise ValueError("euler_characteristic debe ser numérico.") from exc

        if not math.isfinite(x) or not float(x).is_integer():
            raise ValueError("euler_characteristic debe ser un entero finito.")

        return int(x)

    def _phase1_canonical_bell(
        self,
        bell_correlations: Any,
    ) -> Tuple[float, float, float, float]:
        r"""
        Valida correlaciones Bell–CHSH.

        Cada correlador \(E(a_i,b_j)\) es un valor esperado de observables
        dicotómicos \(\pm 1\), luego \(|E|\le 1\) por Cauchy–Schwarz en el
        C*-álgebra generada. Tolerancia numérica \(10^{-9}\).
        """
        if bell_correlations is None:
            raise ValueError("bell_correlations no puede ser None.")

        try:
            seq = tuple(bell_correlations)
        except TypeError as exc:
            raise ValueError("bell_correlations debe ser iterable de 4 elementos.") from exc

        if len(seq) != 4:
            raise ValueError("bell_correlations debe tener exactamente 4 elementos.")

        tol = 1e-9
        vals: List[float] = []

        for idx, value in enumerate(seq):
            self._reject_bool(value, f"bell_correlations[{idx}]")
            try:
                x = float(value)
            except Exception as exc:
                raise ValueError(f"bell_correlations[{idx}] no es escalar.") from exc

            if not math.isfinite(x):
                raise ValueError(f"bell_correlations[{idx}] no es finito.")

            if x < -1.0 - tol or x > 1.0 + tol:
                raise ValueError(f"bell_correlations[{idx}] fuera de [-1,1].")

            vals.append(float(np.clip(x, -1.0, 1.0)))

        return vals[0], vals[1], vals[2], vals[3]

    def _phase1_regularize_spd_metric(
        self,
        metric_tensor: Any,
    ) -> Tuple[np.ndarray, float, float]:
        r"""
        Regulariza un tensor métrico real como SPD.

        Piso relativo de Wilkinson:
            \(\varepsilon = \max(\varepsilon_{\mathrm{mach}}, 10^{-12},
                                 \varepsilon_{\mathrm{mach}}\,\|G\|_2\, n)\).

        Autovalores \(\lambda_i \le \varepsilon\) se proyectan a \(\varepsilon\).
        Devuelve \((G_{\mathrm{spd}}, \mathrm{tr}(\Delta), \kappa_2(G))\).
        """
        if np.iscomplexobj(metric_tensor):
            raise ValueError("metric_tensor debe ser real.")

        try:
            G = np.asarray(metric_tensor, dtype=np.float64)
        except Exception as exc:
            raise ValueError("metric_tensor no puede convertirse a float64.") from exc

        if G.ndim != 2 or G.shape != (self._n, self._n):
            raise ValueError(f"metric_tensor debe tener forma ({self._n},{self._n}).")

        if not np.all(np.isfinite(G)):
            raise ValueError("metric_tensor contiene valores no finitos.")

        G_sym = 0.5 * (G + G.T)

        try:
            eigvals, eigvecs = la.eigh(G_sym)
        except la.LinAlgError as exc:
            raise ValueError("eigh(G) falló; métrica no diagonalizable.") from exc

        if eigvals.size == 0:
            return np.asarray(G_sym, dtype=np.float64), 0.0, 1.0

        op_norm = float(np.max(np.abs(eigvals)))
        floor = max(_STABILITY_FLOOR, _MACHINE_EPS * max(1.0, op_norm) * max(1, self._n))

        shift = np.maximum(floor - eigvals, 0.0)
        shift_trace = float(np.sum(shift))
        eigvals_reg = eigvals + shift

        G_reg = (eigvecs * eigvals_reg) @ eigvecs.T
        G_reg = np.asarray(0.5 * (G_reg + G_reg.T), dtype=np.float64)

        lam_min = float(np.min(eigvals_reg))
        lam_max = float(np.max(eigvals_reg))
        if lam_min <= 0.0 or not math.isfinite(lam_min) or not math.isfinite(lam_max):
            condition = math.inf
        else:
            condition = lam_max / lam_min
            if not math.isfinite(condition) or condition < 1.0:
                condition = math.inf if condition < 0.0 else max(1.0, condition)

        return G_reg, shift_trace, float(condition)

    def _phase1_conductivity_audit(
        self,
        K_boundary_raw: Any,
        metric_tensor: Optional[Any] = None,
    ) -> ConductivityAudit:
        r"""
        Auditoría espectral del tensor de conductividad \(K\).

        En el C*-álgebra \(M_n(\mathbb{C})\) se proyecta
            \(K_H = (K+K^\dagger)/2\)
        y, si hay métrica \(G\succ 0\), se reduce el lápiz
            \(\widetilde K = G^{-1/2} K_H G^{-1/2}\).

        Pasividad de Kirchhoff / Tellegen:
            \(K\) es pasiva ssi \(\mathrm{spec}(K_H)\subset [0,+\infty)\).
        Los autovalores negativos se convierten en penalización exergética.

        \(\kappa_2\) se calcula sobre el espectro positivo; se cota a
        \([1, 10^{12}]\) para no contaminar la fuga con \(\infty\).
        """
        diagnostics: List[str] = []

        try:
            K = np.asarray(K_boundary_raw, dtype=np.complex128)
        except Exception as exc:
            raise ValueError("K_boundary_raw no puede convertirse a complex128.") from exc

        if K.ndim != 2 or K.shape != (self._n, self._n):
            raise ValueError(f"K_boundary_raw debe tener forma ({self._n},{self._n}).")

        if not np.all(np.isfinite(K)):
            raise ValueError("K_boundary_raw contiene valores no finitos.")

        K_h = 0.5 * (K + K.conj().T)
        hermitian_residual = self._frobenius(K - K_h)

        metric_shift = 0.0
        if metric_tensor is not None:
            G, metric_shift, _gcond = self._phase1_regularize_spd_metric(metric_tensor)
            try:
                eig_g, vec_g = la.eigh(G)
            except la.LinAlgError as exc:
                raise ValueError("No fue posible diagonalizar la métrica G.") from exc
            floor = max(_STABILITY_FLOOR, _MACHINE_EPS)
            eig_g = np.maximum(eig_g, floor)
            inv_sqrt_g = (vec_g * (1.0 / np.sqrt(eig_g))) @ vec_g.T
            K_pre = inv_sqrt_g @ K_h @ inv_sqrt_g
            K_pre = 0.5 * (K_pre + K_pre.conj().T)
            if metric_shift > 0.0:
                diagnostics.append(
                    f"Métrica G proyectada a SPD; tr(Δ)={metric_shift:.6e}."
                )
        else:
            K_pre = K_h

        try:
            eigvals = la.eigvalsh(K_pre)
        except la.LinAlgError as exc:
            raise ValueError("No fue posible diagonalizar K_pre.") from exc

        if eigvals.size == 0 or not np.all(np.isfinite(eigvals)):
            raise ValueError("Autovalores de K_pre inválidos.")

        eigvals = np.real(eigvals)
        scale = max(1.0, float(np.max(np.abs(eigvals))))
        tol = max(_STABILITY_FLOOR, 1000.0 * _MACHINE_EPS * scale)

        positive = eigvals[eigvals > tol]
        negative = eigvals[eigvals < -tol]

        spectral_radius = float(np.max(np.abs(eigvals)))
        min_eig = float(np.min(eigvals))

        if eigvals.size >= 2:
            algebraic_connectivity = float(np.sort(eigvals)[1])
        else:
            algebraic_connectivity = float(eigvals[0])

        if positive.size == 0:
            condition_number = 1.0
            spectral_gap = 1.0
            diagnostics.append("Conductividad sin espectro positivo; condición trivial.")
        else:
            max_abs = float(np.max(np.abs(eigvals)))
            min_pos = float(np.min(positive))
            if min_pos <= 0.0 or not math.isfinite(min_pos):
                condition_number = _CONDITION_CAP
                spectral_gap = 0.0
            else:
                condition_number = max_abs / min_pos
                if not math.isfinite(condition_number):
                    condition_number = _CONDITION_CAP
                    spectral_gap = 0.0
                else:
                    spectral_gap = min(1.0, 1.0 / condition_number)

            if negative.size > 0 or (eigvals.size - positive.size - negative.size) > 0:
                # Núcleo o parte negativa: gap algebraico de κ se satura.
                if negative.size > 0:
                    spectral_gap = min(spectral_gap, 0.0)

        condition_number = float(min(max(1.0, condition_number), _CONDITION_CAP))
        if not math.isfinite(spectral_gap) or spectral_gap < 0.0:
            spectral_gap = 0.0

        passive_penalty = float(np.sum(np.abs(negative))) * self._safety_margin
        if not math.isfinite(passive_penalty):
            raise ValueError("Penalización pasiva no finita.")

        if negative.size > 0:
            diagnostics.append(
                f"Conductividad con {int(negative.size)} autovalores negativos; "
                "penalización exergética aplicada (ruptura de pasividad de Kirchhoff)."
            )

        if hermitian_residual > _HERMITIAN_VETO_REL * max(1.0, scale):
            diagnostics.append(
                f"K no hermitiana (‖K-K†‖_F={hermitian_residual:.6e})."
            )

        return ConductivityAudit(
            K_h=K_h,
            condition_number=condition_number,
            passive_penalty=passive_penalty,
            min_eigenvalue=min_eig,
            spectral_gap=float(spectral_gap),
            algebraic_connectivity=float(algebraic_connectivity),
            spectral_radius=float(spectral_radius) if math.isfinite(spectral_radius) else math.inf,
            hermitian_residual=float(hermitian_residual),
            diagnostics=tuple(diagnostics),
        )

    def _phase1_canonical_boundary_matrix(
        self,
        boundary_matrix_integer: Any,
    ) -> Tuple[Tuple[int, ...], ...]:
        """
        Canoniza la matriz de frontera simplicial como matriz entera exacta.

        Se exige compatibilidad topológica: número de columnas igual a `n`
        (B : ℤⁿ → ℤᵐ). Los flotantes se admiten sólo si distan < 10⁻⁹ de ℤ.
        Se rechaza parte imaginaria no nula y no-finitos.
        """
        arr = np.asarray(boundary_matrix_integer)

        if arr.ndim == 1:
            if arr.size == 0:
                arr = arr.reshape(0, self._n)
            else:
                if arr.size != self._n:
                    raise ValueError(
                        "boundary_matrix_integer 1D debe tener tamaño dimension_n."
                    )
                arr = arr.reshape(1, self._n)

        if arr.ndim != 2:
            raise ValueError("boundary_matrix_integer debe ser 1D o 2D.")

        if arr.shape[1] != self._n:
            raise ValueError(
                "boundary_matrix_integer debe tener dimension_n columnas."
            )

        rows, cols = int(arr.shape[0]), int(arr.shape[1])
        out: List[Tuple[int, ...]] = []

        for i in range(rows):
            row: List[int] = []
            for j in range(cols):
                value = arr[i, j]

                if isinstance(value, bool) or isinstance(value, np.bool_):
                    raise ValueError("boundary_matrix_integer no admite booleanos.")

                if isinstance(value, (int, np.integer)):
                    row.append(int(value))
                    continue

                if isinstance(value, (float, np.floating)):
                    x = float(value)
                    if not math.isfinite(x):
                        raise ValueError("Entrada no finita en boundary_matrix_integer.")
                    rounded = round(x)
                    if abs(x - rounded) > 1e-9:
                        raise ValueError("boundary_matrix_integer debe ser entera.")
                    row.append(int(rounded))
                    continue

                if isinstance(value, (complex, np.complexfloating)):
                    z = complex(value)
                    if not (math.isfinite(z.real) and math.isfinite(z.imag)):
                        raise ValueError("Entrada compleja no finita en boundary_matrix_integer.")
                    if abs(z.imag) > 1e-12:
                        raise ValueError("boundary_matrix_integer no admite parte imaginaria.")
                    rounded = round(z.real)
                    if abs(z.real - rounded) > 1e-9:
                        raise ValueError("boundary_matrix_integer debe ser entera.")
                    row.append(int(rounded))
                    continue

                try:
                    iv = int(value)
                except Exception:
                    try:
                        fv = float(value)
                    except Exception as exc:
                        raise ValueError("Entrada no convertible a entero.") from exc
                    if not math.isfinite(fv):
                        raise ValueError("Entrada no finita en boundary_matrix_integer.")
                    rounded = round(fv)
                    if abs(fv - rounded) > 1e-9:
                        raise ValueError("boundary_matrix_integer debe ser entera.")
                    iv = int(rounded)

                row.append(iv)

            out.append(tuple(row))

        return tuple(out)

    def _phase1_canonical_choi(self, Choi_matrix: Any) -> Tuple[np.ndarray, int, float]:
        r"""
        Canoniza la matriz de Choi–Jamiołkowski.

        Teorema de Choi: un mapa \(\Phi: M_d\to M_d\) es completamente
        positivo ssi su matriz de Choi \(C_\Phi\ge 0\). Se proyecta al
        subespacio hermitiano y se mide \(\|C-C^\dagger\|_F/2\).

        Si \(\dim C = n^2\) se toma \(d=n\); si \(\dim C = d^2\) con
        \(d\neq n\) se infiere \(d\) bajo diagnóstico posterior.
        """
        try:
            C = np.asarray(Choi_matrix, dtype=np.complex128)
        except Exception as exc:
            raise ValueError("Choi_matrix no puede convertirse a complex128.") from exc

        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise ValueError("Choi_matrix debe ser cuadrada.")

        if C.shape[0] == 0:
            raise ValueError("Choi_matrix no puede ser vacía.")

        if not np.all(np.isfinite(C)):
            raise ValueError("Choi_matrix contiene valores no finitos.")

        D = int(C.shape[0])

        if D == self._n * self._n:
            d = self._n
        else:
            root = int(round(math.sqrt(D)))
            if root > 0 and root * root == D:
                d = root
            else:
                raise ValueError(
                    "Choi_matrix debe tener dimensión d²; "
                    f"recibido D={D}, incompatible con n={self._n}."
                )

        herm = 0.5 * (C + C.conj().T)
        residual = self._frobenius(C - herm)
        if not math.isfinite(residual):
            raise ValueError("Residuo hermítico de Choi no finito.")

        return np.asarray(herm, dtype=np.complex128), d, float(residual)

    def _phase1_observe_and_normalize(
        self,
        *,
        payload: bytes,
        K_boundary_raw: np.ndarray,
        boundary_matrix_integer: np.ndarray,
        Choi_matrix: np.ndarray,
        bell_correlations: Tuple[float, float, float, float],
        euler_characteristic: int,
        metric_tensor: Optional[np.ndarray],
    ) -> MetabolicCertificate:
        """
        FASE 1 · OBSERVE  —  último morfismo de la fase.

        Definición formal:
            Obs : Inp → Phase1Dossier
            η₁  : Phase1Dossier → MetabolicCertificate
                = `_phase2_audit_and_orient` ∘ Obs

        Su valor de retorno no se detiene en FASE 1: es la unidad
        (continuación estricta) de la FASE 2 · ORIENT / DECIDE.
        """
        payload_b = self._phase1_validate_payload(payload)
        entropy = self._phase1_payload_entropy(payload_b)

        audit = self._phase1_conductivity_audit(
            K_boundary_raw,
            metric_tensor=metric_tensor,
        )

        exergy_leak = entropy * audit.condition_number * 1e-3 + audit.passive_penalty
        if not math.isfinite(exergy_leak):
            exergy_leak = _CONDITION_CAP
        exergy_leak = max(0.0, float(exergy_leak))

        B = self._phase1_canonical_boundary_matrix(boundary_matrix_integer)
        C, choi_dimension, choi_herm_residual = self._phase1_canonical_choi(Choi_matrix)
        bell = self._phase1_canonical_bell(bell_correlations)
        euler = self._phase1_canonical_euler(euler_characteristic)

        K_frozen = self._freeze_ndarray(audit.K_h, np.complex128)
        C_frozen = self._freeze_ndarray(C, np.complex128)

        dossier = Phase1Dossier(
            payload=payload_b,
            entropy=entropy,
            K_h=K_frozen,
            condition_number=audit.condition_number,
            exergy_leak=exergy_leak,
            B=B,
            C=C_frozen,
            choi_dimension=choi_dimension,
            bell=bell,
            euler_characteristic=euler,
            diagnostics=tuple(audit.diagnostics),
            spectral_gap=audit.spectral_gap,
            algebraic_connectivity=audit.algebraic_connectivity,
            spectral_radius=audit.spectral_radius,
            conductivity_min_eig=audit.min_eigenvalue,
            conductivity_hermitian_residual=audit.hermitian_residual,
            choi_hermitian_residual=choi_herm_residual,
        )

        logger.info(
            "FASE 1 OBSERVE: reactor de frontera inicializado. "
            "H=%.6f nats, κ=%.6e, leak=%.6e, d_Choi=%d, gap=%.6e",
            entropy,
            audit.condition_number,
            exergy_leak,
            choi_dimension,
            audit.spectral_gap,
        )

        # ── continuación estricta: FASE 1 ▸ FASE 2 ────────────────────────
        return self._phase2_audit_and_orient(dossier)

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 2 · ORIENT / DECIDE
    #
    # Inicio formal: este bloque es la continuación algebraica del último
    # método de la FASE 1 (`_phase1_observe_and_normalize` → aquí).
    #
    # Responsabilidad:
    #   - Smith Normal Form exacta sobre el DIP ℤ (invariantes, torsión).
    #   - Homología del complejo truncado: rango, Betti, Euler de cadena.
    #   - Consistencia de Kirchhoff: ker(B) ⊆ ker(K_H) (ciclos armónicos).
    #   - Auditoría Choi CP / TP (teorema de Choi + traza parcial).
    #   - Auditoría Bell–CHSH conservadora (clásica y Tsirelson).
    #   - Integración Port-Hamiltoniana (J sesgada, R ≽ 0, Lyapunov).
    #   - Veredicto parcial como ⋀ de átomos en Ω₃.
    #   - Transferir a FASE 3.
    # ═══════════════════════════════════════════════════════════════════════

    def _snf_diagonalize(self, M: List[List[int]]) -> List[List[int]]:
        """
        Diagonalización entera por operaciones elementales unimodulares.

        Reduce M a una forma diagonal equivalente sobre ℤ sin garantizar
        todavía la cadena de divisibilidad \(d_i \mid d_{i+1}\). Esa
        normalización se realiza en `_snf_invariants_from_diagonal`.

        Las transformaciones de Bézout (ext_gcd) son SL(2,ℤ) en cada
        par de filas/columnas, luego el contenido de Fitting se preserva.
        """
        if not M or not M[0]:
            return M

        m = [row[:] for row in M]
        rows = len(m)
        cols = len(m[0])

        if rows * cols > _SNF_EXACT_MAX_ENTRIES:
            raise ArithmeticError(
                f"SNF: matriz {rows}×{cols} excede tope de exactitud "
                f"({_SNF_EXACT_MAX_ENTRIES} entradas)."
            )

        k = 0
        ops = 0

        while k < min(rows, cols):
            pivot_found = False
            best_abs: Optional[int] = None
            pi = pj = k

            for i in range(k, rows):
                for j in range(k, cols):
                    val = m[i][j]
                    if val != 0:
                        av = abs(val)
                        if best_abs is None or av < best_abs:
                            best_abs = av
                            pi, pj = i, j
                            pivot_found = True

            if not pivot_found:
                break

            if pi != k:
                m[k], m[pi] = m[pi], m[k]

            if pj != k:
                for i in range(rows):
                    m[i][k], m[i][pj] = m[i][pj], m[i][k]

            if m[k][k] < 0:
                m[k] = [-x for x in m[k]]

            passes = 0

            while passes < _SNF_MAX_PASSES:
                passes += 1
                changed = False

                for i in range(k + 1, rows):
                    while m[i][k] != 0:
                        ops += 1
                        if ops > _SNF_MAX_OPS:
                            raise ArithmeticError("SNF: exceso de operaciones enteras.")

                        pivot = m[k][k]
                        val = m[i][k]
                        if pivot == 0:
                            raise ArithmeticError("SNF: pivot cero inesperado.")

                        if val % pivot == 0:
                            q = val // pivot
                            if q != 0:
                                m[i] = [a - q * b for a, b in zip(m[i], m[k])]
                        else:
                            g, x, y = ext_gcd(pivot, val)
                            row_k = m[k][:]
                            row_i = m[i][:]
                            m[k] = [x * a + y * b for a, b in zip(row_k, row_i)]
                            m[i] = [
                                -(val // g) * a + (pivot // g) * b
                                for a, b in zip(row_k, row_i)
                            ]
                            if m[k][k] < 0:
                                m[k] = [-v for v in m[k]]

                        changed = True

                for j in range(k + 1, cols):
                    while m[k][j] != 0:
                        ops += 1
                        if ops > _SNF_MAX_OPS:
                            raise ArithmeticError("SNF: exceso de operaciones enteras.")

                        pivot = m[k][k]
                        val = m[k][j]
                        if pivot == 0:
                            raise ArithmeticError("SNF: pivot cero inesperado.")

                        if val % pivot == 0:
                            q = val // pivot
                            if q != 0:
                                for i in range(rows):
                                    m[i][j] -= q * m[i][k]
                        else:
                            g, x, y = ext_gcd(pivot, val)
                            col_k = [m[i][k] for i in range(rows)]
                            col_j = [m[i][j] for i in range(rows)]
                            new_k = [x * a + y * b for a, b in zip(col_k, col_j)]
                            new_j = [
                                -(val // g) * a + (pivot // g) * b
                                for a, b in zip(col_k, col_j)
                            ]
                            for i in range(rows):
                                m[i][k] = new_k[i]
                                m[i][j] = new_j[i]
                            if m[k][k] < 0:
                                for jj in range(cols):
                                    m[k][jj] = -m[k][jj]

                        changed = True

                if m[k][k] < 0:
                    m[k] = [-x for x in m[k]]

                pivot = m[k][k]
                if pivot != 0:
                    found_ndiv = False
                    for i in range(k + 1, rows):
                        for j in range(k + 1, cols):
                            if m[i][j] % pivot != 0:
                                m[k] = [a + b for a, b in zip(m[k], m[i])]
                                changed = True
                                found_ndiv = True
                                break
                        if found_ndiv:
                            break

                row_clear = all(m[k][j] == 0 for j in range(k + 1, cols))
                col_clear = all(m[i][k] == 0 for i in range(k + 1, rows))

                if not changed and row_clear and col_clear:
                    break
            else:
                raise ArithmeticError("SNF: la diagonalización no convergió.")

            if m[k][k] < 0:
                m[k] = [-x for x in m[k]]

            k += 1

        return m

    @staticmethod
    def _snf_invariants_from_diagonal(diagonal_matrix: List[List[int]]) -> List[int]:
        r"""
        Normaliza invariantes diagonales para satisfacer \(d_i \mid d_{i+1}\).

        Se aplica la reducción por pares
            \((a,b)\mapsto(\gcd(a,b),\operatorname{lcm}(a,b))\)
        hasta obtener una cadena de divisibilidad (factores invariantes
        de Fitting / Smith). Los ceros se descartan: no aportan torsión
        ni rango.
        """
        if not diagonal_matrix or not diagonal_matrix[0]:
            return []

        diag: List[int] = []
        size = min(len(diagonal_matrix), len(diagonal_matrix[0]))

        for i in range(size):
            val = int(diagonal_matrix[i][i])
            if val != 0:
                diag.append(abs(val))

        diag.sort()

        changed = True
        guard = 0

        while changed and guard < _SNF_INVARIANT_GUARD:
            changed = False
            guard += 1

            for i in range(len(diag)):
                for j in range(i + 1, len(diag)):
                    a = diag[i]
                    b = diag[j]
                    if a == 0:
                        continue
                    if b % a != 0:
                        g = _gcd_int(a, b)
                        lcm = _lcm_int(a, b)
                        diag[i] = g
                        diag[j] = lcm
                        diag.sort()
                        changed = True
                        break
                if changed:
                    break

        if guard >= _SNF_INVARIANT_GUARD:
            raise ArithmeticError("SNF: normalización de invariantes no convergió.")

        return [d for d in diag if d != 0]

    def _smith_invariants(self, B: Tuple[Tuple[int, ...], ...]) -> List[int]:
        """Calcula invariantes de Smith no nulos de una matriz entera."""
        if not B:
            return []

        M = [list(row) for row in B]
        if not M or not M[0]:
            return []

        diagonal = self._snf_diagonalize(M)
        return self._snf_invariants_from_diagonal(diagonal)

    def compute_smith_normal_form(self, A: np.ndarray) -> np.ndarray:
        r"""
        Calcula la Smith Normal Form de \(A\) sobre el anillo principal \(\mathbb{Z}\).

        Retorna una matriz diagonal equivalente (dtype=object, enteros de
        precisión arbitraria) con los invariantes \(d_i\). La diagonal
        puede contener unos y coeficientes de torsión \(d_i>1\).
        """
        M = self._phase1_canonical_boundary_matrix(A)

        rows = len(M)
        cols = len(M[0]) if rows else self._n

        if rows == 0:
            return np.zeros((0, cols), dtype=object)

        invariants = self._smith_invariants(M)
        out = np.zeros((rows, cols), dtype=object)

        for idx, val in enumerate(invariants):
            if idx < min(rows, cols):
                out[idx, idx] = int(val)

        return out

    def _phase2_numerical_rank_real(self, B: Tuple[Tuple[int, ...], ...]) -> int:
        """Rango numérico de B sobre ℝ por SVD (umbral de Wilkinson relativo)."""
        if not B or not B[0]:
            return 0
        Bf = np.asarray(B, dtype=np.float64)
        try:
            singular = la.svd(Bf, compute_uv=False)
        except la.LinAlgError:
            return 0
        if singular.size == 0:
            return 0
        scale = max(1.0, float(np.max(singular)))
        tol = max(
            _STABILITY_FLOOR,
            _REL_RANK_MULT * _MACHINE_EPS * scale * max(Bf.shape),
        )
        return int(np.sum(singular > tol))

    def _phase2_cycle_annihilation_residual(
        self,
        K_h: np.ndarray,
        B: Tuple[Tuple[int, ...], ...],
    ) -> float:
        r"""
        Residuo de aniquilación del espacio de ciclos.

        Si \(K_H\) es un laplaciano / Kirchhoff del mismo complejo que \(B\),
            \(\ker B \subseteq \ker K_H\)
        (los 1-ciclos son armónicos). Se mide
            \(\max_{\|v\|=1,\, Bv=0} \|K_H v\|_2 / (\|K_H\|_2+\varepsilon)\).
        """
        if not B or not B[0] or K_h.size == 0:
            return 0.0

        Bf = np.asarray(B, dtype=np.float64)
        try:
            _u, s, vt = la.svd(Bf, full_matrices=True)
        except la.LinAlgError:
            return math.inf

        scale = max(1.0, float(np.max(s)) if s.size else 1.0)
        tol = max(
            _STABILITY_FLOOR,
            _REL_RANK_MULT * _MACHINE_EPS * scale * max(Bf.shape),
        )
        rank = int(np.sum(s > tol))
        nullity = int(vt.shape[0]) - rank
        if nullity <= 0:
            return 0.0

        K_real = np.real(0.5 * (K_h + K_h.conj().T))
        null_basis = vt[rank:, :].T
        try:
            k_norm = float(np.linalg.norm(K_real, ord=2))
        except Exception:
            k_norm = self._frobenius(K_real)
        den = k_norm + _STABILITY_FLOOR

        residuals: List[float] = []
        for idx in range(null_basis.shape[1]):
            v = null_basis[:, idx]
            v_norm = float(np.linalg.norm(v))
            if v_norm <= _STABILITY_FLOOR:
                continue
            v = v / v_norm
            residuals.append(float(np.linalg.norm(K_real @ v)) / den)

        if not residuals:
            return 0.0
        val = float(max(residuals))
        return val if math.isfinite(val) else math.inf

    def _phase2_smith_audit(
        self,
        B: Tuple[Tuple[int, ...], ...],
        K_h: np.ndarray,
        euler_characteristic: int,
    ) -> HomologicalAudit:
        r"""
        Auditoría homológica mediante Smith Normal Form.

        Sobre un DIP, \(\ker B\) es libre y la torsión vive en
            \(\mathrm{coker}(B)\cong \mathbb{Z}^{\beta_0}\oplus\bigoplus_i\mathbb{Z}/d_i\mathbb{Z}\),
        con \(d_i>1\) los factores invariantes no triviales.

        Sobre \(\mathbb{Q}\):
            \(r=\mathrm{rk}(B)\), \(\beta_1=n-r\), \(\beta_0=m-r\),
            \(\chi=\beta_0-\beta_1=m-n\).
        \(\chi_{\mathrm{cadena}}\) se compara con la característica de Euler
        declarada como diagnóstico (grupos superiores pueden diferir).
        """
        diagnostics: List[str] = []

        if not B:
            return HomologicalAudit(
                invariants=(),
                torsion=(),
                has_torsion=False,
                numerical_rank=0,
                betti_0=0,
                betti_1=self._n,
                euler_from_chain=-self._n,
                cycle_annihilation_residual=0.0,
                diagnostics=("Matriz de frontera vacía; torsión trivial asumida.",),
            )

        m_rows = len(B)
        n_cols = len(B[0]) if m_rows else self._n

        invariants = tuple(int(d) for d in self._smith_invariants(B))
        torsion = tuple(d for d in invariants if d > 1)
        has_torsion = len(torsion) > 0

        rank_snf = len(invariants)
        rank_num = self._phase2_numerical_rank_real(B)
        rank = max(rank_snf, rank_num)

        betti_1 = max(0, n_cols - rank)
        betti_0 = max(0, m_rows - rank)
        euler_from_chain = betti_0 - betti_1
        cycle_res = self._phase2_cycle_annihilation_residual(K_h, B)

        if has_torsion:
            diagnostics.append(f"Torsión homológica detectada: {list(torsion)}.")
        else:
            diagnostics.append("Homología libre de torsión.")

        if euler_from_chain != euler_characteristic:
            diagnostics.append(
                f"Euler de cadena β0-β1={euler_from_chain} ≠ euler declarado="
                f"{euler_characteristic} (grupos superiores posibles)."
            )

        if cycle_res > _CYCLE_DEGRADED_REL:
            diagnostics.append(
                f"Inconsistencia de Kirchhoff: ker(B) ⊈ ker(K) "
                f"(residuo={cycle_res:.6e})."
            )

        return HomologicalAudit(
            invariants=invariants,
            torsion=torsion,
            has_torsion=has_torsion,
            numerical_rank=rank,
            betti_0=betti_0,
            betti_1=betti_1,
            euler_from_chain=euler_from_chain,
            cycle_annihilation_residual=cycle_res,
            diagnostics=tuple(diagnostics),
        )

    def _phase2_choi_audit(
        self,
        C: np.ndarray,
        d: int,
        hermitian_residual_pre: float = 0.0,
    ) -> ChoiCPTPAudit:
        r"""
        Auditoría CPTP de la matriz de Choi.

        - CP  ⇔  \(C=C^\dagger\) y \(\lambda_{\min}(C)\ge 0\) (Choi).
        - TP  ⇔  \(\mathrm{Tr}_{\mathrm{out}}(C)=I_d\) (convención reshape C).
        - CPTP ⇔ CP ∧ TP, i.e. el mapa es un canal cuántico.

        Retorna min_eig, tp_diff, banderas, traza y diagnósticos.
        """
        diagnostics: List[str] = []
        D = d * d

        if C.shape != (D, D):
            raise ValueError(f"Choi_matrix debe ser ({D},{D}) para d={d}.")

        herm_diff = float(max(hermitian_residual_pre, self._frobenius(C - C.conj().T)))
        C_h = 0.5 * (C + C.conj().T)

        scale = max(1.0, self._frobenius(C_h))
        tol = max(_STABILITY_FLOOR, 1000.0 * _MACHINE_EPS * scale)

        is_hermitian = herm_diff <= max(tol, _HERMITIAN_VETO_REL * scale)
        if not is_hermitian:
            diagnostics.append("Choi no Hermitiana dentro de tolerancia.")

        try:
            eigvals = la.eigvalsh(C_h)
            min_eig = float(np.real(np.min(eigvals))) if eigvals.size else 0.0
        except la.LinAlgError:
            min_eig = -math.inf
            diagnostics.append("Fallo en eigvalsh de Choi; se asume no CP.")

        is_cp = bool(
            is_hermitian and math.isfinite(min_eig) and min_eig >= -tol
        )

        try:
            C4 = np.reshape(C_h, (d, d, d, d), order="C")
            # Convención (i,a,j,b); Tr_out = sum_a C_{i a j a}.
            partial_trace = np.trace(C4, axis1=1, axis2=3)
            tp_diff = float(la.norm(partial_trace - np.eye(d, dtype=np.complex128), ord="fro"))
            is_tp = math.isfinite(tp_diff) and tp_diff <= max(tol, _TP_VETO_ABS)
        except Exception:
            tp_diff = math.inf
            is_tp = False
            diagnostics.append("Fallo en traza parcial de Choi; se asume no TP.")

        trace = float(np.trace(C_h).real)
        if not math.isfinite(trace):
            trace = 0.0
            diagnostics.append("Traza de Choi no finita.")

        is_cptp = bool(is_cp and is_tp)

        if not is_cp:
            diagnostics.append("Choi no completamente positiva.")
        if not is_tp:
            diagnostics.append("Choi no preserva traza.")

        return ChoiCPTPAudit(
            min_eig=min_eig,
            tp_diff=float(tp_diff) if math.isfinite(tp_diff) else math.inf,
            is_cp=is_cp,
            is_tp=is_tp,
            is_cptp=is_cptp,
            trace=trace,
            hermitian_residual=float(herm_diff),
            diagnostics=tuple(diagnostics),
        )

    @staticmethod
    def _phase2_bell_audit(
        bell: Tuple[float, float, float, float],
    ) -> BellCHSHAudit:
        r"""
        Auditoría Bell–CHSH conservadora.

        Se evalúan las cuatro combinaciones de un signo
            \(S = E_{11}+E_{12}+E_{21}-E_{22}\)
        y permutaciones, y se toma el máximo absoluto. Así no se
        subestima no-localidad ni una re-etiquetación de observables.

        Cotas:
            clásica    \(|S|\le 2\),
            Tsirelson  \(|S|\le 2\sqrt{2}\).
        """
        e11, e12, e21, e22 = bell

        standard = abs(e11 + e12 + e21 - e22)

        candidates = (
            e11 + e12 + e21 - e22,
            e11 + e12 - e21 + e22,
            e11 - e12 + e21 + e22,
            -e11 + e12 + e21 + e22,
        )
        conservative = float(max(abs(s) for s in candidates))

        tol = 1e-12
        is_bell_coherent = conservative <= _TSIRELSON_BOUND + tol
        exceeds_classical = conservative > _BELL_CLASSICAL_BOUND + tol
        tsirelson_slack = max(0.0, _TSIRELSON_BOUND - conservative)

        diagnostics: List[str] = []
        if not is_bell_coherent:
            diagnostics.append(
                f"CHSH conservador {conservative:.12f} supera Tsirelson."
            )
        elif exceeds_classical:
            diagnostics.append(
                "CHSH supera la cota clásica (=2) y permanece bajo Tsirelson; "
                "correlación cuántica admisible."
            )

        return BellCHSHAudit(
            chsh_standard=float(standard),
            chsh_conservative=conservative,
            is_bell_coherent=is_bell_coherent,
            tsirelson_slack=float(tsirelson_slack),
            exceeds_classical=exceeds_classical,
            diagnostics=tuple(diagnostics),
        )

    def integrate_port_hamiltonian_step(
        self,
        Q_init: float,
        A_init: float,
        euler_characteristic: int,
        r_Q: float = 0.1,
        r_A: float = 0.15,
        dt: float = 0.01,
    ) -> Tuple[float, float, float]:
        r"""
        Integra un paso del lazo Port-Hamiltoniano acoplado (IDA-PBC):

        \[
            \begin{bmatrix} \dot{\mathcal{Q}} \\ \dot{\mathcal{A}} \end{bmatrix}
            = (\mathcal{J}-\mathcal{R})\,\nabla H,
            \qquad
            H=\tfrac12(\mathcal{Q}^2+\mathcal{A}^2),
        \]

        con \(\mathcal{J}=-\mathcal{J}^T\) (estructura de Poisson / Euler)
        y \(\mathcal{R}=\mathcal{R}^T\succeq 0\) (disipación de Rayleigh).
        Entonces
            \(\dot H = -\nabla H\cdot\mathcal{R}\nabla H \le 0\)
        (Lyapunov estricta si \(\mathcal{R}\succ 0\)).

        Se usa exponencial matricial; fallback RK4 clásico.
        """
        try:
            Q0 = self._finite_float(Q_init, "Q_init")
            A0 = self._finite_float(A_init, "A_init")
            self._reject_bool(euler_characteristic, "euler_characteristic")
            chi = float(euler_characteristic)
            rq = max(0.0, self._finite_float(r_Q, "r_Q"))
            ra = max(0.0, self._finite_float(r_A, "r_A"))
            step = self._finite_float(dt, "dt")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError("Parámetros Port-Hamiltonian no escalares.") from exc

        if not math.isfinite(chi):
            raise ValueError("euler_characteristic no finito.")

        if step <= 0.0:
            H_val = 0.5 * (Q0 * Q0 + A0 * A0)
            return Q0, A0, H_val

        J = np.array([[0.0, -chi], [chi, 0.0]], dtype=np.float64)
        R = np.array([[rq, 0.0], [0.0, ra]], dtype=np.float64)
        A_mat = J - R
        x0 = np.array([Q0, A0], dtype=np.float64)

        try:
            Phi = la.expm(A_mat * step)
            x1 = Phi @ x0
            if not np.all(np.isfinite(x1)):
                raise la.LinAlgError("Exponencial matricial no finita.")
        except Exception:
            def f(x: np.ndarray) -> np.ndarray:
                return A_mat @ x

            k1 = f(x0)
            k2 = f(x0 + 0.5 * step * k1)
            k3 = f(x0 + 0.5 * step * k2)
            k4 = f(x0 + step * k3)
            x1 = x0 + (step / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        Q_next = float(x1[0])
        A_next = float(x1[1])
        if not (math.isfinite(Q_next) and math.isfinite(A_next)):
            raise ValueError("Estado Port-Hamiltonian no finito.")

        H_val = 0.5 * (Q_next * Q_next + A_next * A_next)
        return Q_next, A_next, H_val

    def _phase2_port_hamiltonian_energy(
        self,
        Q_obs: float,
        A_obs: float,
        euler_characteristic: int,
    ) -> Tuple[float, float, float]:
        """
        Integra un paso y reporta (H, disipación Rayleigh, caída de Lyapunov).

        Disipación instantánea en el estado inicial: \(x^T R x\).
        Caída: \(H(0)-H(\Delta t)\ge 0\) salvo error numérico.
        """
        H0 = 0.5 * (Q_obs * Q_obs + A_obs * A_obs)
        r_Q, r_A, dt = 0.1, 0.15, 0.01
        dissipation = r_Q * (Q_obs ** 2) + r_A * (A_obs ** 2)

        _q, _a, H1 = self.integrate_port_hamiltonian_step(
            Q_init=Q_obs,
            A_init=A_obs,
            euler_characteristic=euler_characteristic,
            r_Q=r_Q,
            r_A=r_A,
            dt=dt,
        )
        drop = max(0.0, float(H0 - H1))
        return float(H1), float(dissipation), float(drop)

    def _phase2_evaluate_partial_heyting(
        self,
        *,
        has_torsion: bool,
        is_cp: bool,
        is_tp: bool,
        min_eig: float,
        tp_diff: float,
        chsh_conservative: float,
        exergy_leak: float,
        choi_hermitian_residual: float,
        kirchhoff_cycle_residual: float,
        hamiltonian_energy: float,
        spectral_gap: float,
    ) -> Tuple[str, Tuple[str, ...]]:
        """
        Veredicto parcial de Heyting en la FASE 2, por ⋀ de átomos en Ω₃.

        La tasa Lindblad se evalúa en FASE 3 (dinámica disipativa, no
        coherencia estática).

        Átomos de VETO:
            torsión, ¬CP, ¬TP, λ_min(Choi) < cota dura, error TP duro,
            Tsirelson, fuga exergética dura, no-hermiticidad grosera.
        Átomos de DEGRADED:
            λ_min marginal, error TP marginal, CHSH > 2, fuga blanda,
            Kirchhoff, Hamiltoniano negativo, gap espectral nulo.
        """
        diagnostics: List[str] = []

        homological_atom = _HEYTING_VETOED if has_torsion else _HEYTING_COHERENT
        if has_torsion:
            diagnostics.append("VETO por torsión homológica.")

        causal_veto = (
            (not is_cp)
            or (not is_tp)
            or (math.isfinite(min_eig) and min_eig < _CHOI_MIN_EIG_VETO)
            or (math.isfinite(tp_diff) and tp_diff > _TP_VETO_ABS)
            or (chsh_conservative > _TSIRELSON_BOUND + 1e-12)
            or (
                math.isfinite(choi_hermitian_residual)
                and choi_hermitian_residual
                > _HERMITIAN_VETO_REL * max(1.0, abs(min_eig), 1.0)
            )
        )
        causal_degraded = (
            (math.isfinite(min_eig) and min_eig < _CHOI_MIN_EIG_DEGRADED)
            or (math.isfinite(tp_diff) and tp_diff > _TP_DEGRADED_ABS)
            or (chsh_conservative > _BELL_CLASSICAL_BOUND)
        )
        if not is_cp:
            diagnostics.append("VETO por Choi no CP.")
        if not is_tp:
            diagnostics.append("VETO por Choi no TP.")
        if math.isfinite(min_eig) and min_eig < _CHOI_MIN_EIG_VETO:
            diagnostics.append("VETO por λ_min(Choi) bajo cota dura.")
        if math.isfinite(tp_diff) and tp_diff > _TP_VETO_ABS:
            diagnostics.append("VETO por error de preservación de traza duro.")
        if chsh_conservative > _TSIRELSON_BOUND + 1e-12:
            diagnostics.append("VETO por violación Tsirelson.")

        if causal_veto:
            causal_atom = _HEYTING_VETOED
        elif causal_degraded:
            causal_atom = _HEYTING_DEGRADED
            if math.isfinite(min_eig) and min_eig < _CHOI_MIN_EIG_DEGRADED:
                diagnostics.append("DEGRADED por λ_min(Choi) marginal.")
            if math.isfinite(tp_diff) and tp_diff > _TP_DEGRADED_ABS:
                diagnostics.append("DEGRADED por error de traza marginal.")
            if chsh_conservative > _BELL_CLASSICAL_BOUND:
                diagnostics.append("DEGRADED por CHSH sobre cota clásica.")
        else:
            causal_atom = _HEYTING_COHERENT

        if (not math.isfinite(exergy_leak)) or exergy_leak > self._hard_leak_ceiling:
            thermal_atom = _HEYTING_VETOED
            diagnostics.append("VETO por fuga exergética dura.")
        elif exergy_leak > self._soft_leak_ceiling:
            thermal_atom = _HEYTING_DEGRADED
            diagnostics.append("DEGRADED por fuga exergética elevada.")
        else:
            thermal_atom = _HEYTING_COHERENT

        if kirchhoff_cycle_residual > _CYCLE_DEGRADED_REL:
            kirchhoff_atom = _HEYTING_DEGRADED
            diagnostics.append("DEGRADED por inconsistencia de Kirchhoff / ciclos.")
        else:
            kirchhoff_atom = _HEYTING_COHERENT

        if math.isfinite(hamiltonian_energy) and hamiltonian_energy < -1e-9:
            hamiltonian_atom = _HEYTING_DEGRADED
            diagnostics.append("DEGRADED por Hamiltoniano negativo no físico.")
        else:
            hamiltonian_atom = _HEYTING_COHERENT

        if (not math.isfinite(spectral_gap)) or spectral_gap <= 0.0:
            spectral_atom = _HEYTING_DEGRADED
            diagnostics.append("DEGRADED por gap espectral de conductancia nulo.")
        else:
            spectral_atom = _HEYTING_COHERENT

        verdict = _heyting_meet(
            homological_atom,
            causal_atom,
            thermal_atom,
            kirchhoff_atom,
            hamiltonian_atom,
            spectral_atom,
        )
        return verdict, tuple(diagnostics)

    def _phase2_audit_and_orient(
        self,
        dossier: Phase1Dossier,
    ) -> MetabolicCertificate:
        """
        FASE 2 · ORIENT / DECIDE  —  último morfismo de la fase.

        Definición formal:
            Dec : Phase1Dossier → Phase2Dossier
            η₂  : Phase2Dossier → MetabolicCertificate
                = `_phase3_metabolize_and_certify` ∘ Dec

        Su valor de retorno no se detiene en FASE 2: es la unidad
        (continuación estricta) de la FASE 3 · ACT / CERTIFY.
        """
        diagnostics: List[str] = list(dossier.diagnostics)

        homology = self._phase2_smith_audit(
            dossier.B,
            dossier.K_h,
            dossier.euler_characteristic,
        )
        diagnostics.extend(homology.diagnostics)

        choi = self._phase2_choi_audit(
            dossier.C,
            dossier.choi_dimension,
            hermitian_residual_pre=dossier.choi_hermitian_residual,
        )
        diagnostics.extend(choi.diagnostics)

        bell = self._phase2_bell_audit(dossier.bell)
        diagnostics.extend(bell.diagnostics)

        Q_obs = dossier.entropy if math.isfinite(dossier.entropy) else 0.0
        A_obs = bell.chsh_conservative if bell.is_bell_coherent else 4.0
        if not math.isfinite(A_obs):
            A_obs = 4.0

        h_energy, ph_dissipation, lyap_drop = self._phase2_port_hamiltonian_energy(
            Q_obs,
            A_obs,
            dossier.euler_characteristic,
        )

        partial_verdict, verdict_diag = self._phase2_evaluate_partial_heyting(
            has_torsion=homology.has_torsion,
            is_cp=choi.is_cp,
            is_tp=choi.is_tp,
            min_eig=choi.min_eig,
            tp_diff=choi.tp_diff,
            chsh_conservative=bell.chsh_conservative,
            exergy_leak=dossier.exergy_leak,
            choi_hermitian_residual=choi.hermitian_residual,
            kirchhoff_cycle_residual=homology.cycle_annihilation_residual,
            hamiltonian_energy=h_energy,
            spectral_gap=dossier.spectral_gap,
        )
        diagnostics.extend(verdict_diag)

        phase2 = Phase2Dossier(
            phase1=dossier,
            torsion_coefficients=list(homology.torsion),
            has_torsion=homology.has_torsion,
            choi_minimum_eigenvalue=choi.min_eig,
            choi_trace_preservation_error=choi.tp_diff,
            choi_trace=choi.trace,
            is_completely_positive=choi.is_cp,
            is_trace_preserving=choi.is_tp,
            is_cptp=choi.is_cptp,
            bell_chsh_standard=bell.chsh_standard,
            bell_chsh_conservative=bell.chsh_conservative,
            is_bell_coherent=bell.is_bell_coherent,
            hamiltonian_energy=h_energy,
            heyting_verdict=partial_verdict,
            diagnostics=tuple(diagnostics),
            betti_0=homology.betti_0,
            betti_1=homology.betti_1,
            homology_rank=homology.numerical_rank,
            euler_from_chain=homology.euler_from_chain,
            kirchhoff_cycle_residual=homology.cycle_annihilation_residual,
            choi_hermitian_residual=choi.hermitian_residual,
            tsirelson_slack=bell.tsirelson_slack,
            port_hamiltonian_dissipation=ph_dissipation,
            energy_lyapunov_drop=lyap_drop,
        )

        logger.info(
            "FASE 2 ORIENT: veredicto parcial=%s, torsion=%s, CHSH=%.6f, "
            "λ_min=%.6e, β=(%d,%d), χ_cadena=%d",
            partial_verdict,
            list(homology.torsion),
            bell.chsh_conservative,
            choi.min_eig,
            homology.betti_0,
            homology.betti_1,
            homology.euler_from_chain,
        )

        # ── continuación estricta: FASE 2 ▸ FASE 3 ────────────────────────
        return self._phase3_metabolize_and_certify(phase2)

    # ═══════════════════════════════════════════════════════════════════════
    # FASE 3 · ACT / CERTIFY
    #
    # Inicio formal: este bloque es la continuación algebraica del último
    # método de la FASE 2 (`_phase2_audit_and_orient` → aquí).
    #
    # Responsabilidad:
    #   - Metabolismo Lindblad–GKSL exacto (amortiguación de amplitud CPTP).
    #   - Auditoría de positividad, traza y pureza de ρ.
    #   - Ajuste final del veredicto por tasa de decaimiento (átomo disipativo).
    #   - Interlock hardware simulado (crowbar BT151 / GPIO14 / IRAM).
    #   - Firma SHA-256 con separación de dominio y certificado inmutable.
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _phase3_lindblad_gamma(
        exergy_leak: float,
        torsion_detected: bool,
    ) -> float:
        r"""
        Tasa GKSL \(\Gamma\) del operador de salto \(L=\sqrt{\Gamma}\,\sigma_-\).

        \(\Gamma_0=2\) (unidades adimensionales del qubit semántico).
        La torsión homológica inhibe el metabolismo
            \(\Gamma \leftarrow \Gamma_0 / (1 + 10\,\mathbf{1}_{\mathrm{tors}}
                                              + 4\,\ell/(1+\ell))\),
        con \(\ell=\max(0,\mathrm{leak})\). Así, fuga alta o torsión
        empujan \(\Gamma\) hacia las cotas de DEGRADED / VETO del supervisor.
        """
        leak = max(0.0, float(exergy_leak)) if math.isfinite(float(exergy_leak)) else 0.0
        leak_factor = leak / (1.0 + leak)
        torsion_factor = 1.0 if bool(torsion_detected) else 0.0
        denom = (
            1.0
            + _LINDBLAD_TORSION_WEIGHT * torsion_factor
            + _LINDBLAD_LEAK_WEIGHT * leak_factor
        )
        gamma = _LINDBLAD_GAMMA0 / denom
        if not math.isfinite(gamma) or gamma < 0.0:
            return 0.0
        return float(gamma)

    def metabolize_hallucination_lindblad(
        self,
        exergy_leak: float,
        torsion_detected: bool,
        steps: int = 10,
        dt: float = 0.05,
    ) -> Tuple[np.ndarray, float]:
        r"""
        Metaboliza la alucinación semántica mediante un canal GKSL exacto.

        Para el qubit semántico inicial \(|1\rangle\langle 1|\) con operador
        de salto \(L=\sqrt{\Gamma}\,\sigma_-\), el canal de amortiguación
        de amplitud es CPTP con operadores de Kraus

        \[
            E_0=\begin{pmatrix}1&0\\0&\sqrt{\eta}\end{pmatrix},
            \quad
            E_1=\begin{pmatrix}0&\sqrt{1-\eta}\\0&0\end{pmatrix},
            \quad \eta=e^{-\Gamma t},
        \]

        y solución exacta de poblaciones

        \[
            \rho(t)=\begin{pmatrix}1-e^{-\Gamma t}&0\\0&e^{-\Gamma t}\end{pmatrix}.
        \]

        Esto evita derivas de traza y violaciones de positividad propias
        de esquemas explícitos de Euler. La fase hamiltoniana (si se
        añadiera \(\sigma_z\)) no altera poblaciones de un estado diagonal.
        """
        try:
            leak = float(exergy_leak)
        except Exception as exc:
            raise ValueError("exergy_leak no escalar.") from exc
        if not math.isfinite(leak):
            leak = 0.0

        self._reject_bool(steps, "steps")
        try:
            n_steps = int(steps)
            time_step = float(dt)
        except Exception as exc:
            raise ValueError("steps/dt inválidos en Lindblad.") from exc

        if n_steps < 0:
            raise ValueError("steps debe ser no negativo.")
        if not math.isfinite(time_step) or time_step < 0.0:
            raise ValueError("dt debe ser finito y no negativo.")

        decay_rate = self._phase3_lindblad_gamma(leak, bool(torsion_detected))
        total_time = float(n_steps) * time_step

        if total_time <= 0.0:
            rho = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
            return rho, decay_rate

        eta = math.exp(-decay_rate * total_time)
        if not math.isfinite(eta):
            eta = 0.0
        eta = float(np.clip(eta, 0.0, 1.0))

        # Canal de Kraus aplicado a |1><1|.
        e0 = np.array([[1.0, 0.0], [0.0, math.sqrt(eta)]], dtype=np.complex128)
        e1 = np.array([[0.0, math.sqrt(max(0.0, 1.0 - eta))], [0.0, 0.0]], dtype=np.complex128)
        rho0 = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
        rho = e0 @ rho0 @ e0.conj().T + e1 @ rho0 @ e1.conj().T
        rho = 0.5 * (rho + rho.conj().T)

        trace = float(np.trace(rho).real)
        if not math.isfinite(trace) or trace <= 0.0:
            rho = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
        else:
            rho = rho / trace

        return np.asarray(rho, dtype=np.complex128), decay_rate

    def _phase3_audit_density(self, rho: np.ndarray) -> Tuple[float, float, bool]:
        """
        Audita positividad, pureza y hermiticidad de ρ.

        Retorna (λ_min, pureza=Tr(ρ²), is_state_like) donde is_state_like
        exige λ_min ≥ −ε y Tr(ρ)≈1 (la traza se audita aparte).
        """
        rho_h = 0.5 * (rho + rho.conj().T)

        try:
            eigvals = la.eigvalsh(rho_h)
            min_eig = float(np.real(np.min(eigvals))) if eigvals.size else 0.0
        except la.LinAlgError:
            min_eig = -math.inf

        try:
            purity = float(np.trace(rho_h @ rho_h).real)
        except Exception:
            purity = math.nan

        if not math.isfinite(min_eig):
            min_eig = -math.inf
        if not math.isfinite(purity):
            purity = math.nan

        is_state_like = min_eig >= -_RHO_PSD_TOL
        return min_eig, purity, is_state_like

    def _phase3_actuate_interlock(
        self,
        verdict: str,
        session_hash: str,
    ) -> Tuple[bool, float]:
        """
        Simula actuación crowbar BT151 [GPIO14] en IRAM.

        Sólo `VETOED` dispara el interlock. La latencia se recorta al
        intervalo físico [380, 420] ns (tiristor + jitter de ISR).
        """
        session_ref = str(session_hash)[:16]
        verdict_norm = str(verdict).strip().upper()

        if verdict_norm == _HEYTING_VETOED:
            jitter = float(self._rng.normal(0.0, _CROWBAR_JITTER_STD_NS))
            latency = float(
                np.clip(
                    _CROWBAR_IRAM_LATENCY_NS + jitter,
                    _CROWBAR_LATENCY_MIN_NS,
                    _CROWBAR_LATENCY_MAX_NS,
                )
            )

            logger.critical(
                "¡RUPTURA DE COHERENCIA EN LA FRONTERA! "
                "Disyuntor crowbar BT151 [GPIO%d] gatillado en %.2f ns. "
                "Sello: %s",
                _GPIO_CROWBAR_PIN,
                latency,
                session_ref,
            )
            return True, latency

        logger.info(
            "Lazo de frontera cerrado sin interlock. Estado=%s. Sello=%s",
            verdict_norm,
            session_ref,
        )
        return False, 0.0

    def _phase3_compose_cryptographic_seal(
        self,
        *,
        verdict: str,
        payload: bytes,
        entropy: float,
        exergy_leak: float,
        decay_rate: float,
        latency_ns: float,
        torsion_count: int,
        choi_min_eig: float,
        chsh: float,
        hamiltonian: float,
        extra_scalars: Sequence[float] = (),
    ) -> str:
        """Firma SHA-256 de telemetría final con separación de dominio."""
        sha = hashlib.sha256()

        self._hash_update_domain(sha, _DOMAIN_ENGINE_TELEMETRY)
        self._hash_update_bytes(sha, verdict.encode("utf-8"))
        self._hash_update_bytes(sha, payload)

        scalars: Tuple[float, ...] = (
            entropy,
            exergy_leak,
            decay_rate,
            latency_ns,
            float(torsion_count),
            choi_min_eig,
            chsh,
            hamiltonian,
            *tuple(float(x) for x in extra_scalars),
        )

        for value in scalars:
            self._hash_update_bytes(sha, f"{float(value):.17e}".encode("ascii"))

        self._hash_update_bytes(sha, _SCHEMA_VERSION.encode("utf-8"))
        return sha.hexdigest()

    def _phase3_session_digest(self, payload: bytes) -> str:
        sha = hashlib.sha256()
        self._hash_update_domain(sha, _DOMAIN_ENGINE_SESSION)
        self._hash_update_bytes(sha, payload)
        self._hash_update_bytes(sha, _SCHEMA_VERSION.encode("utf-8"))
        self._hash_update_bytes(sha, str(self._n).encode("ascii"))
        return sha.hexdigest()

    def _phase3_metabolize_and_certify(
        self,
        dossier: Phase2Dossier,
    ) -> MetabolicCertificate:
        """
        FASE 3 · ACT / CERTIFY.

        Terminal del ciclo OODA anidado: metaboliza, ajusta veredicto
        por el átomo disipativo, dispara interlock si corresponde y
        emite el certificado inmutable (objeto inicial de la categoría
        de evidencias: ningún campo posterior puede mutarse).
        """
        diagnostics: List[str] = list(dossier.diagnostics)

        rho_final, decay_rate = self.metabolize_hallucination_lindblad(
            exergy_leak=dossier.phase1.exergy_leak,
            torsion_detected=dossier.has_torsion,
            steps=_LINDBLAD_DEFAULT_STEPS,
            dt=_LINDBLAD_DEFAULT_DT,
        )

        rho_trace = float(np.trace(rho_final).real)
        rho_min_eig, rho_purity, rho_psd = self._phase3_audit_density(rho_final)

        dissipative_atom = _HEYTING_COHERENT
        if decay_rate < _LINDBLAD_VETO_FLOOR:
            dissipative_atom = _HEYTING_VETOED
            diagnostics.append("VETO final por tasa Lindblad insuficiente.")
        elif decay_rate < _LINDBLAD_DEGRADED_FLOOR:
            dissipative_atom = _HEYTING_DEGRADED
            diagnostics.append("DEGRADED final por disipación Lindblad débil.")

        rho_atom = _HEYTING_COHERENT
        if (not rho_psd) or rho_min_eig < -_RHO_PSD_TOL:
            rho_atom = _HEYTING_VETOED
            diagnostics.append("VETO final por ρ no positiva.")
        if (not math.isfinite(rho_trace)) or abs(rho_trace - 1.0) > _RHO_TRACE_TOL:
            rho_atom = _HEYTING_VETOED
            diagnostics.append("VETO final por traza de ρ no conservada.")

        verdict = _heyting_meet(
            dossier.heyting_verdict,
            dissipative_atom,
            rho_atom,
        )

        session_digest = self._phase3_session_digest(dossier.phase1.payload)
        interlock_fired, latency = self._phase3_actuate_interlock(verdict, session_digest)

        seal = self._phase3_compose_cryptographic_seal(
            verdict=verdict,
            payload=dossier.phase1.payload,
            entropy=dossier.phase1.entropy,
            exergy_leak=dossier.phase1.exergy_leak,
            decay_rate=decay_rate,
            latency_ns=latency,
            torsion_count=len(dossier.torsion_coefficients),
            choi_min_eig=dossier.choi_minimum_eigenvalue,
            chsh=dossier.bell_chsh_conservative,
            hamiltonian=dossier.hamiltonian_energy,
            extra_scalars=(
                dossier.phase1.spectral_gap,
                dossier.phase1.algebraic_connectivity,
                dossier.kirchhoff_cycle_residual,
                dossier.choi_hermitian_residual,
                float(dossier.betti_0),
                float(dossier.betti_1),
                rho_trace,
                decay_rate,
            ),
        )

        coupled_state = CoupledBoundaryState(
            Q_thermal_entropy=dossier.phase1.entropy,
            Q_exergy_leak=dossier.phase1.exergy_leak,
            A_smith_torsion=list(dossier.torsion_coefficients),
            A_choi_min_eigenvalue=dossier.choi_minimum_eigenvalue,
            A_bell_chsh=dossier.bell_chsh_conservative,
            hamiltonian_energy=dossier.hamiltonian_energy,
            condition_number=dossier.phase1.condition_number,
            choi_trace_preservation_error=dossier.choi_trace_preservation_error,
            bell_tsirelson_slack=dossier.tsirelson_slack,
            spectral_gap=dossier.phase1.spectral_gap,
            algebraic_connectivity=dossier.phase1.algebraic_connectivity,
            spectral_radius=dossier.phase1.spectral_radius,
            choi_hermitian_residual=dossier.choi_hermitian_residual,
            kirchhoff_cycle_residual=dossier.kirchhoff_cycle_residual,
            betti_0=dossier.betti_0,
            betti_1=dossier.betti_1,
            homology_rank=dossier.homology_rank,
            euler_from_chain=dossier.euler_from_chain,
            port_hamiltonian_dissipation=dossier.port_hamiltonian_dissipation,
        )

        energy_balance_residual = max(
            0.0,
            dossier.phase1.exergy_leak - self._soft_leak_ceiling,
        )

        certificate = MetabolicCertificate(
            heyting_verdict=verdict,
            coupled_state=coupled_state,
            lindblad_density_trace=rho_trace,
            lindblad_decay_rate=decay_rate,
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=latency,
            cryptographic_seal=seal,
            schema_version=_SCHEMA_VERSION,
            choi_dimension=dossier.phase1.choi_dimension,
            choi_completely_positive=dossier.is_completely_positive,
            choi_trace_preserving=dossier.is_trace_preserving,
            bell_coherent=dossier.is_bell_coherent,
            rho_minimum_eigenvalue=rho_min_eig,
            rho_purity=rho_purity,
            condition_number=dossier.phase1.condition_number,
            diagnostics=tuple(diagnostics),
            session_digest=session_digest,
            choi_trace=dossier.choi_trace,
            rho_cp_channel=bool(rho_psd and abs(rho_trace - 1.0) <= _RHO_TRACE_TOL),
            energy_balance_residual=energy_balance_residual,
            tsirelson_slack=dossier.tsirelson_slack,
        )

        if verdict == _HEYTING_VETOED:
            logger.error(
                "FASE 3 CERTIFY: VETO del reactor de anillos. Γ=%.4f, "
                "torsion=%s, λ_min(Choi)=%.6e, CHSH=%.6f",
                decay_rate,
                dossier.torsion_coefficients,
                dossier.choi_minimum_eigenvalue,
                dossier.bell_chsh_conservative,
            )
        elif verdict == _HEYTING_DEGRADED:
            logger.warning(
                "FASE 3 CERTIFY: reactor degradado. diagnostics=%s",
                tuple(diagnostics),
            )
        else:
            logger.info(
                "FASE 3 CERTIFY: reactor coherente. Γ=%.4f, leak=%.6e, digest=%s",
                decay_rate,
                dossier.phase1.exergy_leak,
                session_digest[:16],
            )

        return certificate

    # ═══════════════════════════════════════════════════════════════════════
    # FAIL-CLOSED GLOBAL
    # ═══════════════════════════════════════════════════════════════════════

    def _fail_closed_certificate(
        self,
        reason: str,
        payload: Any = b"",
    ) -> MetabolicCertificate:
        """
        Certificado fail-closed ante excepción no recuperable.

        Se fuerza torsión no trivial ([2]), decaimiento nulo y veredicto
        `VETOED` para que cualquier supervisor aguas arriba (p. ej. el
        BoundaryRingSheafAgent) rechace la frontera.
        """
        payload_b = self._safe_payload_bytes(payload)
        session_digest = self._phase3_session_digest(payload_b)

        interlock_fired, latency = self._phase3_actuate_interlock(
            _HEYTING_VETOED,
            session_digest,
        )

        seal = self._phase3_compose_cryptographic_seal(
            verdict=_HEYTING_VETOED,
            payload=payload_b,
            entropy=0.0,
            exergy_leak=0.0,
            decay_rate=0.0,
            latency_ns=latency,
            torsion_count=1,
            choi_min_eig=0.0,
            chsh=0.0,
            hamiltonian=0.0,
        )

        coupled_state = CoupledBoundaryState(
            Q_thermal_entropy=0.0,
            Q_exergy_leak=0.0,
            A_smith_torsion=[2],
            A_choi_min_eigenvalue=0.0,
            A_bell_chsh=0.0,
            hamiltonian_energy=0.0,
            condition_number=math.inf,
            choi_trace_preservation_error=math.inf,
            bell_tsirelson_slack=0.0,
            spectral_gap=0.0,
            algebraic_connectivity=0.0,
            spectral_radius=math.inf,
            choi_hermitian_residual=math.inf,
            kirchhoff_cycle_residual=math.inf,
            betti_0=0,
            betti_1=0,
            homology_rank=0,
            euler_from_chain=0,
            port_hamiltonian_dissipation=0.0,
        )

        return MetabolicCertificate(
            heyting_verdict=_HEYTING_VETOED,
            coupled_state=coupled_state,
            lindblad_density_trace=1.0,
            lindblad_decay_rate=0.0,
            hardware_interlock_fired=interlock_fired,
            actuation_latency_ns=latency,
            cryptographic_seal=seal,
            schema_version=_SCHEMA_VERSION,
            choi_dimension=self._n,
            choi_completely_positive=False,
            choi_trace_preserving=False,
            bell_coherent=False,
            rho_minimum_eigenvalue=0.0,
            rho_purity=1.0,
            condition_number=math.inf,
            diagnostics=(f"FAIL-CLOSED: {reason}",),
            session_digest=session_digest,
            choi_trace=0.0,
            rho_cp_channel=False,
            energy_balance_residual=0.0,
            tsirelson_slack=0.0,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # CICLO PÚBLICO COMPLETO
    # ═══════════════════════════════════════════════════════════════════════

    def execute_metabolic_cycle(
        self,
        payload: bytes,
        K_boundary_raw: np.ndarray,
        boundary_matrix_integer: np.ndarray,
        Choi_matrix: np.ndarray,
        bell_correlations: Tuple[float, float, float, float],
        euler_characteristic: int = 1,
        metric_tensor: Optional[np.ndarray] = None,
    ) -> MetabolicCertificate:
        r"""
        Orquesta el ciclo acoplado de-confinado en el reactor de anillos.

        Cadena anidada:

            FASE 1 · OBSERVE
              └─ FASE 2 · ORIENT / DECIDE
                   └─ FASE 3 · ACT / CERTIFY

        Preserva la API original y admite `metric_tensor` opcional por
        keyword. Cualquier excepción produce certificado fail-closed VETOED.
        """
        try:
            return self._phase1_observe_and_normalize(
                payload=payload,
                K_boundary_raw=K_boundary_raw,
                boundary_matrix_integer=boundary_matrix_integer,
                Choi_matrix=Choi_matrix,
                bell_correlations=bell_correlations,
                euler_characteristic=euler_characteristic,
                metric_tensor=metric_tensor,
            )
        except Exception as exc:
            logger.exception(
                "Fallo no recuperable en BoundaryRingSheaf; emitiendo VETO fail-closed."
            )
            return self._fail_closed_certificate(reason=str(exc), payload=payload)


# ═══════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN DE FIRMAS DE CALIBRE
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    "BoundaryRingSheaf",
    "CoupledBoundaryState",
    "MetabolicCertificate",
    "ext_gcd",
]

# Compatibilidad con la exportación histórica del módulo original.
all = __all__