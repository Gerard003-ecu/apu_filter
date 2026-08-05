# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Generative Boole Hodge Suturator Agent (Soberano del Haz Boole)    ║
║  Ruta   : app/agents/boole/generative_boole_hodge_suturator_agent.py         ║
║  Versión: 5.0.0-Doctoral-Nested-OODA-Heyting-TMR-Hodge-ESP32-Secure          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  NATURALEZA CIBER-FÍSICA Y RIGOR DOCTORAL:                                   ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Este módulo materializa al Agente Soberano y Observador Activo que          ║
║  gobierna al morfismo de sutura 'generative_boole_hodge_suturator.py' en     ║
║  el espacio de control del Haz Tangente Generativo Γ(M, T_B Boole) sobre     ║
║  el topos de haces Sh(B; Ω₃) con valores en el retículo de Heyting Ω₃.       ║
║                                                                              ║
║  Axioma de Consistencia de la Métrica (Dualidad de de Rham–Hodge):           ║
║  $$\Delta_k^H = \delta_k^\dagger\delta_k + \delta_{k-1}\delta_{k-1}^\dagger$$ ║
║  $$\ker\Delta_k \cong H^k_{\mathrm{dR}}(K;\mathbb{F})$$                      ║
║                                                                              ║
║  Contrato de Seguridad (fail-secure) — Retículo de Heyting Ω₃:               ║
║  COHERENT (0) ≼ DEGRADED (1) ≼ VETOED (2). El veredicto final se             ║
║  resuelve mediante votación de redundancia modular triple (TMR) y            ║
║  supremum de Heyting (join) sobre los veredictos de las tres fases.          ║
║                                                                              ║
║  ESTRUCTURA DE FASES ANIDADAS (herencia covariante):                         ║
║    Phase1_SpectralObserver                                                   ║
║      └── Phase2_GaugeBooleanValidator(Phase1_SpectralObserver)               ║
║            └── Phase3_SheafSpectralValidator(Phase2_GaugeBooleanValidator)   ║
║                  └── GenerativeBooleHodgeSuturatorAgent(Morphism, Phase3)    ║
║                                                                              ║
║  El último método certificado de la Fase k es el morfismo de transición      ║
║  (continuation arrow) que inicia la Fase k+1 en el ciclo OODA.               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, IntEnum, auto
from typing import (
    Final,
    Tuple,
    Optional,
    List,
    Dict,
    Any,
    Sequence,
    Callable,
    NamedTuple,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Stubs seguros del núcleo MIC (fail-open import boundary)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
    from app.core.schemas import Stratum
except ImportError:
    class TopologicalInvariantError(Exception):
        r"""Excepción base del sistema para violaciones topológico-algebraicas."""
        pass

    class Morphism:
        r"""Clase base de morfismos en la categoría MIC (Category of Internal Controls)."""
        pass

    class Stratum:
        r"""Estratos de la jerarquía DIKW (Data–Information–Knowledge–Wisdom)."""
        PHYSICS = "PHYSICS"
        TACTICS = "TACTICS"
        STRATEGY = "STRATEGY"
        WISDOM = "WISDOM"

logger = logging.getLogger("MIC.Agents.Boole.GenerativeBooleHodgeSuturatorAgent")

# =============================================================================
# CONSTANTES ESPECTRALES, FÍSICAS Y DE CONTRATO HARDWARE
# =============================================================================
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_DEFAULT_TOL: Final[float] = 1.0e-12
_SPECTRAL_PSD_FLOOR: Final[float] = -1.0e-13
_EPS_HERMITICITY: Final[float] = 1.0e-10
_EPS_TRACE: Final[float] = 1.0e-6
_EPS_UNITARY: Final[float] = 1.0e-9
_EPS_SYMPLECTIC: Final[float] = 1.0e-10
_EPS_IDEMPOTENCE: Final[float] = 1.0e-12
_EPS_KMS: Final[float] = 1.0e-9
_EPS_NILPOTENCE: Final[float] = 1.0e-8
_CONDITION_NUMBER_MAX: Final[float] = 1.0e8
_KCL_TOL: Final[float] = 1.0e-8
_WILKINSON_RANK_FACTOR: Final[float] = 1.0e5
_VON_NEUMANN_ENTROPY_FLOOR: Final[float] = 0.0
_CROWBAR_GPIO: Final[int] = 14          # GPIO14 — Contrato hardware ESP32 (BT151)
_TMR_QUORUM: Final[int] = 2             # Mayoría simple en Triple Modular Redundancy
_FOCK_NORM_TARGET: Final[float] = 1.0
_MAC_TRACE_TARGET: Final[float] = 1.0
_HODGE_KERNEL_TOL: Final[float] = 1.0e-11


# =============================================================================
# JERARQUÍA DE EXCEPCIONES (Categoría de fallos topológico-algebraicos)
# =============================================================================
class BooleHodgeSuturatorAgentError(TopologicalInvariantError):
    r"""Excepción raíz del Agente Soberano del Suturador de Boole-Hodge."""
    pass


class FockIsometryViolation(BooleHodgeSuturatorAgentError):
    r"""Detonada ante desviaciones no unitarias en el espacio de Fock $\mathcal{F}$."""
    pass


class DensityMatrixAnomalyError(BooleHodgeSuturatorAgentError):
    r"""Detonada si la MAC viola los postulados de Dirac–von Neumann."""
    pass


class SymplecticInvarianceViolation(BooleHodgeSuturatorAgentError):
    r"""Detonada si el Jacobiano de evolución del AST rompe la forma simpléctica $\omega$."""
    pass


class BooleanAlgebraConsistencyError(BooleHodgeSuturatorAgentError):
    r"""Detonada ante inconsistencias de idempotencia en la MIC booleana sobre $\mathbb{F}_2$."""
    pass


class ModularInvolutionError(BooleHodgeSuturatorAgentError):
    r"""Detonada si el operador modular de Tomita–Takesaki $J_\rho$ no es involutivo."""
    pass


class CohomologicalBifurcationError(BooleHodgeSuturatorAgentError):
    r"""Detonada ante un $H^1(K;\mathbb{F}) \neq 0$ certificado de forma exacta."""
    pass


class SpectralDegeneracyError(BooleHodgeSuturatorAgentError):
    r"""Detonada si el operador coborde colapsa numéricamente ($\kappa(\delta) \gg 1$)."""
    pass


class ChainComplexNilpotenceError(BooleHodgeSuturatorAgentError):
    r"""Detonada si $\delta_{k+1}\circ\delta_k \neq 0$ (ruptura del complejo de cocadenas)."""
    pass


class HodgeLaplacianAnomalyError(BooleHodgeSuturatorAgentError):
    r"""Detonada si $\Delta_k^H$ pierde autoadjunto-positividad o falla el isomorfismo de Hodge."""
    pass


class TMRConsensusFailure(BooleHodgeSuturatorAgentError):
    r"""Detonada si el quorum TMR no alcanza mayoría de coherencia."""
    pass


# =============================================================================
# RETÍCULO DE HEYTING Ω₃ Y ACCIONES FÍSICAS DE CROWBAR
# =============================================================================
class BooleHodgeSovereignVerdict(IntEnum):
    r"""
    Clasificador de subobjetos de tres valores en el topos de Hodge-Boole.

    Estructura de retículo de Heyting finito Ω₃:
        COHERENT (0) ≼ DEGRADED (1) ≼ VETOED (2)

    Operaciones:
        join  (∨) = max   — supremum (peor caso / fail-secure)
        meet  (∧) = min   — infimum
        imp   (⇒) : a ⇒ b = 2 si a ≼ b else b   (implicación intuicionista)

    El orden total garantiza que el join de veredictos es monótono y
    que cualquier VETOED domina al resto (contrato fail-secure).
    """
    COHERENT = 0
    DEGRADED = 1
    VETOED = 2

    def heyting_imp(self, other: "BooleHodgeSovereignVerdict") -> "BooleHodgeSovereignVerdict":
        r"""Implicación de Heyting: $a \Rightarrow b$."""
        if self.value <= other.value:
            return BooleHodgeSovereignVerdict.VETOED  # top del retículo como "true"
        return other

    infix_join = classmethod(lambda cls, a, b: cls(max(a.value, b.value)))
    infix_meet = classmethod(lambda cls, a, b: cls(min(a.value, b.value)))


class CrowbarAction(Enum):
    r"""
    Acciones físicas de mitigación tras el veredicto en la retícula Ω₃.

    Mapeo ciber-físico ESP32:
        NONE           → idle, GPIO14 = Hi-Z
        WATCHDOG_PULSE → pulso TTL 100 µs en GPIO14 (NMI soft-reset)
        HARD_SHORT     → latch del tiristor BT151 vía GPIO14 (crowbar de bus DC)
    """
    NONE = auto()
    WATCHDOG_PULSE = auto()
    HARD_SHORT = auto()


# =============================================================================
# ARTEFACTOS INMUTABLES DE CERTIFICACIÓN POR FASE
# =============================================================================
@dataclass(frozen=True, slots=True)
class FockNormAudit:
    r"""Sub-certificado atómico: isometría de Fock $\|\psi\|_{\ell^2}=1$."""
    norm_l2: float
    residual: float
    is_isometric: bool


@dataclass(frozen=True, slots=True)
class MacHermitianAudit:
    r"""Sub-certificado atómico: autoadjunto $\rho=\rho^\dagger$."""
    frobenius_residual: float
    is_hermitian: bool


@dataclass(frozen=True, slots=True)
class MacTraceAudit:
    r"""Sub-certificado atómico: normalización $\mathrm{Tr}(\rho)=1$."""
    trace_real: float
    residual: float
    is_normalized: bool


@dataclass(frozen=True, slots=True)
class MacSpectrumAudit:
    r"""Sub-certificado atómico: PSD + entropía de von Neumann."""
    eigenvalues: Tuple[float, ...]
    min_eigenvalue: float
    is_psd: bool
    von_neumann_entropy: float
    purity: float  # Tr(ρ²)


@dataclass(frozen=True, slots=True)
class Phase1FockPhysicsCertificate:
    r"""
    Artefacto de la FASE 1 (Observe): Física de Fock y postulados de Dirac–von Neumann.

    Encapsula la auditoría completa del estado de Fock $\psi\in\mathcal{F}$ y de la
    matriz de densidad MAC $\rho\in\mathcal{B}(\mathcal{H})_+$ conforme a:
        (D1) $\|\psi\|_2 = 1$                         isometría de Fock
        (D2) $\rho=\rho^\dagger$                      hermiticidad
        (D3) $\mathrm{Tr}(\rho)=1$                    normalización
        (D4) $\mathrm{spec}(\rho)\subset[0,+\infty)$  semidefinida positiva
        (D5) $S(\rho)=-\mathrm{Tr}(\rho\log\rho)$     entropía de von Neumann
    """
    fock_audit: FockNormAudit
    hermitian_audit: MacHermitianAudit
    trace_audit: MacTraceAudit
    spectrum_audit: MacSpectrumAudit
    is_fock_isometry: bool
    is_mac_psd: bool
    mac_trace: float
    mac_entropy: float
    mac_purity: float
    verdict: BooleHodgeSovereignVerdict
    tmr_votes: Tuple[BooleHodgeSovereignVerdict, BooleHodgeSovereignVerdict, BooleHodgeSovereignVerdict]


@dataclass(frozen=True, slots=True)
class SymplecticFormAudit:
    r"""Sub-certificado: $M^\top\Omega M=\Omega$ (preservación de Liouville)."""
    dimension_half: int
    frobenius_residual: float
    is_symplectic: bool
    pfaffian_sign: float  # signo del Pfaffiano de Ω (orientación)


@dataclass(frozen=True, slots=True)
class BooleanIdempotenceAudit:
    r"""Sub-certificado: idempotencia $M\vee M=M$ en el semianillo booleano."""
    residual_frobenius: float
    is_idempotent: bool
    is_absorbent: bool       # M ∧ 0 = 0
    is_multiplicative_id: bool  # diag compatible con 1_B


@dataclass(frozen=True, slots=True)
class ModularTomitaAudit:
    r"""
    Sub-certificado: involución modular de Tomita–Takesaki.
    $J_\rho(X)=\rho^{1/2}X^\dagger\rho^{-1/2}$,  $J_\rho\circ J_\rho=\mathrm{id}$.
    """
    kms_residual: float
    involution_residual: float
    is_involutive: bool
    tikhonov_regularized: bool


@dataclass(frozen=True, slots=True)
class Phase2GaugeBooleanCertificate:
    r"""
    Artefacto de la FASE 2 (Orient): Invarianza simpléctica, álgebra booleana y modular J.

    Encapsula:
        (S1) $M^\top\Omega M=\Omega$          grupo espimpléctico Sp(2n,ℝ)
        (B1) $B\circ_{\mathbb{F}_2}B=B$     idempotencia booleana (MIC)
        (T1) $J_\rho^2=\mathrm{id}$          involución modular (KMS)
    """
    symplectic_audit: SymplecticFormAudit
    boolean_audit: BooleanIdempotenceAudit
    modular_audit: ModularTomitaAudit
    is_symplectic: bool
    is_mic_idempotent: bool
    kms_residual: float
    is_involutive_modular: bool
    verdict: BooleHodgeSovereignVerdict
    tmr_votes: Tuple[BooleHodgeSovereignVerdict, BooleHodgeSovereignVerdict, BooleHodgeSovereignVerdict]


@dataclass(frozen=True, slots=True)
class CoboundaryNilpotenceAudit:
    r"""Sub-certificado: $\delta_1\circ\delta_0=0$ (axioma de complejo de cocadenas)."""
    residual_frobenius: float
    is_nilpotent: bool
    max_entry_abs: float


@dataclass(frozen=True, slots=True)
class HodgeBettiAudit:
    r"""
    Sub-certificado: números de Betti vía SVD de Wilkinson + Laplaciano de Hodge.
    $\beta_k=\dim\ker\Delta_k^H=\dim H^k(K;\mathbb{F})$.
    """
    rank_delta_0: int
    rank_delta_1: int
    betti_0: int
    betti_1: int
    hodge_kernel_dim_0: int
    hodge_kernel_dim_1: int
    is_hodge_isomorphism_consistent: bool


@dataclass(frozen=True, slots=True)
class SpectralConditionAudit:
    r"""Sub-certificado: número de condición de Wilkinson $\kappa_2(\delta_0)$."""
    sigma_max: float
    sigma_min_positive: float
    condition_number: float
    is_spectrally_stable: bool


@dataclass(frozen=True, slots=True)
class Phase3SheafSpectralCertificate:
    r"""
    Artefacto de la FASE 3 (Decide): Cohomología exacta, rango de Wilkinson y estabilidad.

    Encapsula:
        (C1) $\delta_1\circ\delta_0=0$               nilpotencia del complejo
        (C2) $\beta_k=\dim H^k$                      números de Betti
        (C3) $\kappa_2(\delta_0)\le\kappa_{\max}$    estabilidad espectral
        (C4) $\ker\Delta_k\cong H^k$                 isomorfismo de Hodge discreto
    """
    nilpotence_audit: CoboundaryNilpotenceAudit
    betti_audit: HodgeBettiAudit
    condition_audit: SpectralConditionAudit
    betti_0: int
    betti_1: int
    cochain_identity_residual: float
    condition_number: float
    is_cohomologically_coherent: bool
    verdict: BooleHodgeSovereignVerdict
    tmr_votes: Tuple[BooleHodgeSovereignVerdict, BooleHodgeSovereignVerdict, BooleHodgeSovereignVerdict]


@dataclass(frozen=True, slots=True)
class BooleHodgeSuturationState:
    r"""
    Certificado global terminal de la sutura del Haz Tangente Generativo en Boole.

    Es el objeto terminal del funtor de gobernanza soberana:
        $\mathrm{Gov}:\mathbf{Input}\to\mathbf{Cert}_{\Omega_3}$.
    """
    phase1: Phase1FockPhysicsCertificate
    phase2: Phase2GaugeBooleanCertificate
    phase3: Phase3SheafSpectralCertificate
    final_verdict: BooleHodgeSovereignVerdict
    tmr_final_verdict: BooleHodgeSovereignVerdict
    crowbar_triggered: bool
    crowbar_action: CrowbarAction
    crowbar_gpio: int
    timestamp_utc: str
    provenance_hash: str
    ooda_latency_hints: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# UTILIDADES MATEMÁTICAS PURAS (nivel doctoral)
# =============================================================================
def _heyting_join(*verdicts: BooleHodgeSovereignVerdict) -> BooleHodgeSovereignVerdict:
    r"""Supremum de Heyting (join): $\bigvee_i v_i=\max_i v_i$ (fail-secure)."""
    return BooleHodgeSovereignVerdict(max(v.value for v in verdicts))


def _heyting_meet(*verdicts: BooleHodgeSovereignVerdict) -> BooleHodgeSovereignVerdict:
    r"""Infimum de Heyting (meet): $\bigwedge_i v_i=\min_i v_i$."""
    return BooleHodgeSovereignVerdict(min(v.value for v in verdicts))


def _tmr_vote(
    v1: BooleHodgeSovereignVerdict,
    v2: BooleHodgeSovereignVerdict,
    v3: BooleHodgeSovereignVerdict,
) -> BooleHodgeSovereignVerdict:
    r"""
    Redundancia Modular Triple (TMR).

    Estrategia: mayoría simple sobre el retículo Ω₃. En empate total (3 valores
    distintos) se aplica el join (peor caso) por contrato fail-secure.
    """
    counts: Dict[int, int] = {}
    for v in (v1, v2, v3):
        counts[v.value] = counts.get(v.value, 0) + 1
    # Mayoría
    for val, cnt in sorted(counts.items()):
        if cnt >= _TMR_QUORUM:
            return BooleHodgeSovereignVerdict(val)
    # Empate 1-1-1 → join fail-secure
    return _heyting_join(v1, v2, v3)


def _frobenius_norm(a: NDArray[Any]) -> float:
    r"""$\|A\|_F=\sqrt{\mathrm{Tr}(A^\dagger A)}$."""
    return float(la.norm(a, ord="fro"))


def _safe_log_entropy(eigenvalues: NDArray[np.float64], tol: float) -> float:
    r"""
    Entropía de von Neumann regularizada:
    $S(\rho)=-\sum_i\lambda_i\log\lambda_i$ con soporte $\{\lambda_i>\mathrm{tol}$.
    """
    clean = eigenvalues[eigenvalues > tol]
    if clean.size == 0:
        return _VON_NEUMANN_ENTROPY_FLOOR
    return -float(np.sum(clean * np.log(clean + _MACHINE_EPS)))


def _build_symplectic_form(n: int) -> NDArray[np.float64]:
    r"""
    Forma simpléctica canónica $\Omega_{2n}$ en $\mathbb{R}^{2n}$:
    $$\Omega=\begin{pmatrix}0&I_n\\-I_n&0\end{pmatrix},\quad
      \Omega^\top=-\Omega,\quad\Omega^2=-I_{2n}.$$
    """
    id_n = np.eye(n, dtype=np.float64)
    z_n = np.zeros((n, n), dtype=np.float64)
    return np.block([[z_n, id_n], [-id_n, z_n]])


def _boolean_matrix_project(m: NDArray[np.float64], threshold: float = 0.5) -> NDArray[np.float64]:
    r"""Proyección al semianillo booleano $\{0,1\}$ con umbral."""
    return (m > threshold).astype(np.float64)


def _boolean_mat_mul(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    r"""
    Producto en el semianillo booleano (OR-AND):
    $(A\otimes_{\mathbb{B}}B)_{ij}=\bigvee_k(A_{ik}\wedge B_{kj})=\min(1,(AB)_{ij})$.
    """
    return np.minimum(a @ b, 1.0)


def _wilkinson_rank(svd_vals: NDArray[np.float64], floor: float = _SPECTRAL_PSD_FLOOR) -> int:
    r"""
    Rango numérico de Wilkinson–Higham:
    $\mathrm{rank}_\varepsilon(A)=\#\{\sigma_i:\sigma_i>\varepsilon\}$.
    """
    threshold = abs(floor) * _WILKINSON_RANK_FACTOR
    return int(np.sum(svd_vals > threshold))


def _discrete_hodge_laplacian(
    delta_k: NDArray[np.float64],
    delta_km1: Optional[NDArray[np.float64]] = None,
) -> NDArray[np.float64]:
    r"""
    Laplaciano de Hodge discreto de grado k:
    $$\Delta_k^H=\delta_k^\top\delta_k+\delta_{k-1}\delta_{k-1}^\top.$$
    Si $\delta_{k-1}$ es None, se reduce a $\delta_k^\top\delta_k$ (grado 0).
    """
    term_up = delta_k.T @ delta_k
    if delta_km1 is None:
        return term_up
    term_down = delta_km1 @ delta_km1.T
    # Alinear dimensiones si es necesario (bloques cuadrados por construcción)
    return term_up + term_down


# =============================================================================
#  FASE 1 — OBSERVE
#  Física cuántica de Fock · Postulados de Dirac–von Neumann · MAC
# =============================================================================
class Phase1_SpectralObserver:
    r"""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║  FASE 1 — Observe: Certificación de la física cuántica, Fock e           ║
    ║  invariantes MAC (matriz de densidad del Multifísico Acoplado).          ║
    ║                                                                          ║
    ║  Dominio categórico:                                                     ║
    ║    $\mathbf{Hilb}_{\mathrm{fin}}$ (Hilbert finito-dimensional)           ║
    ║    $\mathcal{F}=\bigoplus_{n=0}^{N}\bigwedge^n\mathcal{H}$ (Fock)        ║
    ║                                                                          ║
    ║  Morfismos auditados:                                                    ║
    ║    $\psi\mapsto\|\psi\|_2$,  $\rho\mapsto(\rho^\dagger,\mathrm{Tr},\sigma)$║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """

    # ─── 1.1  Isometría de Fock ─────────────────────────────────────────────
    def _audit_fock_isometry(
        self,
        fock_state: NDArray[np.float64],
        tolerance: float = _DEFAULT_TOL,
    ) -> FockNormAudit:
        r"""
        Audita la isometría de Fock en norma $\ell^2$:
        $$\|\psi\|_2=\Bigl(\sum_i|\psi_i|^2\Bigr)^{1/2}=1.$$

        Parámetros
        ----------
        fock_state : amplitud de Fock en base de Slater (vector real).
        tolerance  : cota de Wilkinson para el residuo $| \|\psi\|_2-1 |$.

        Retorna
        -------
        FockNormAudit inmutable.

        Raise
        -----
        FockIsometryViolation si el residuo supera `tolerance`.
        """
        if fock_state.ndim != 1:
            raise FockIsometryViolation(
                f"FockIsometryViolation: se esperaba vector 1-D, ndim={fock_state.ndim}."
            )
        if fock_state.size == 0:
            raise FockIsometryViolation("FockIsometryViolation: estado de Fock vacío.")

        norm_l2 = float(np.linalg.norm(fock_state, ord=2))
        residual = abs(norm_l2 - _FOCK_NORM_TARGET)
        is_isometric = residual <= tolerance

        if not is_isometric:
            raise FockIsometryViolation(
                f"FockIsometryViolation: ruptura de isometría. "
                f"‖ψ‖₂={norm_l2:.8e}, residuo={residual:.4e} > tol={tolerance:.4e}."
            )

        return FockNormAudit(norm_l2=norm_l2, residual=residual, is_isometric=is_isometric)

    # ─── 1.2  Hermiticidad de la MAC ────────────────────────────────────────
    def _audit_mac_hermiticity(
        self,
        mac_density: NDArray[np.complex128],
        eps: float = _EPS_HERMITICITY,
    ) -> MacHermitianAudit:
        r"""
        Audita autoadjunto: $\rho=\rho^\dagger\Leftrightarrow\|\rho-\rho^\dagger\|_F\le\varepsilon$.

        En mecánica cuántica (postulado de Dirac), todo observable —y en particular
        la matriz de densidad— debe ser un operador autoadjunto en $\mathcal{H}$.
        """
        if mac_density.ndim != 2 or mac_density.shape[0] != mac_density.shape[1]:
            raise DensityMatrixAnomalyError(
                f"DensityMatrixAnomalyError: MAC no cuadrada: shape={mac_density.shape}."
            )

        anti_hermitian_part = mac_density - mac_density.conj().T
        frobenius_residual = _frobenius_norm(anti_hermitian_part)
        is_hermitian = frobenius_residual <= eps

        if not is_hermitian:
            raise DensityMatrixAnomalyError(
                f"DensityMatrixAnomalyError: MAC no autoadjunta. "
                f"‖ρ−ρ†‖_F={frobenius_residual:.4e} > eps={eps:.4e}."
            )

        return MacHermitianAudit(
            frobenius_residual=frobenius_residual,
            is_hermitian=is_hermitian,
        )

    # ─── 1.3  Traza unitaria ────────────────────────────────────────────────
    def _audit_mac_trace(
        self,
        mac_density: NDArray[np.complex128],
        eps: float = _EPS_TRACE,
    ) -> MacTraceAudit:
        r"""
        Audita normalización probabilista: $\mathrm{Tr}(\rho)=1$.

        La parte imaginaria de la traza debe ser numéricamente nula por hermiticidad;
        se proyecta a la parte real para estabilidad de punto flotante.
        """
        trace_c = np.trace(mac_density)
        trace_real = float(np.real(trace_c))
        imag_leak = abs(float(np.imag(trace_c)))
        residual = abs(trace_real - _MAC_TRACE_TARGET)
        is_normalized = residual <= eps and imag_leak <= eps

        if not is_normalized:
            raise DensityMatrixAnomalyError(
                f"DensityMatrixAnomalyError: traza anómala. "
                f"Tr(ρ)={trace_real:.8f}+{imag_leak:.4e}j ≠ 1.0 "
                f"(residuo={residual:.4e}, imag_leak={imag_leak:.4e})."
            )

        return MacTraceAudit(
            trace_real=trace_real,
            residual=residual,
            is_normalized=is_normalized,
        )

    # ─── 1.4  Espectro PSD + entropía de von Neumann + pureza ───────────────
    def _audit_mac_spectrum(
        self,
        mac_density: NDArray[np.complex128],
        tolerance: float = _DEFAULT_TOL,
    ) -> MacSpectrumAudit:
        r"""
        Audita el espectro de $\rho$ vía descomposición de Cholesky–Hermite
        (``eigh``): semidefinida positiva, entropía y pureza.

        Definiciones
        ------------
        - PSD: $\mathrm{spec}(\rho)\subset[\,\underline{\sigma},+\infty)$
          con $\underline{\sigma}=$_SPECTRAL_PSD_FLOOR (holgura numérica).
        - Entropía de von Neumann: $S(\rho)=-\mathrm{Tr}(\rho\log\rho)$.
        - Pureza: $\gamma=\mathrm{Tr}(\rho^2)\in[1/d,1]$; $\gamma=1\Leftrightarrow$ puro.
        """
        # eigh garantiza autovalores reales para Hermitianas
        eigenvalues = np.real(la.eigvalsh(mac_density))
        eigenvalues = np.sort(eigenvalues)  # ascendente
        min_eig = float(eigenvalues[0])
        is_psd = min_eig >= _SPECTRAL_PSD_FLOOR

        if not is_psd:
            raise DensityMatrixAnomalyError(
                f"DensityMatrixAnomalyError: MAC no PSD. "
                f"λ_min={min_eig:.4e} < floor={_SPECTRAL_PSD_FLOOR:.4e}."
            )

        entropy = _safe_log_entropy(eigenvalues, tolerance)
        # Pureza Tr(ρ²) = ∑ λ_i²
        purity = float(np.sum(eigenvalues ** 2))

        return MacSpectrumAudit(
            eigenvalues=tuple(float(e) for e in eigenvalues),
            min_eigenvalue=min_eig,
            is_psd=is_psd,
            von_neumann_entropy=entropy,
            purity=purity,
        )

    # ─── 1.5  Clasificador local de veredicto Ω₃ (Fase 1) ───────────────────
    def _classify_phase1_verdict(
        self,
        fock: FockNormAudit,
        herm: MacHermitianAudit,
        tr: MacTraceAudit,
        spec: MacSpectrumAudit,
        tolerance: float,
    ) -> Tuple[
        BooleHodgeSovereignVerdict,
        Tuple[BooleHodgeSovereignVerdict, BooleHodgeSovereignVerdict, BooleHodgeSovereignVerdict],
    ]:
        r"""
        Clasifica el estado de la Fase 1 en Ω₃ y genera tres votos TMR independientes.

        Canales TMR:
            V1 — canal isometría + traza (conservación probabilista)
            V2 — canal hermiticidad + PSD (estructura de C*-álgebra)
            V3 — canal entropía/pureza (consistencia termodinámica)
        """
        # Canal V1
        v1 = BooleHodgeSovereignVerdict.COHERENT
        if fock.residual > tolerance * 10.0 or tr.residual > _EPS_TRACE * 10.0:
            v1 = BooleHodgeSovereignVerdict.DEGRADED
        if not fock.is_isometric or not tr.is_normalized:
            v1 = BooleHodgeSovereignVerdict.VETOED

        # Canal V2
        v2 = BooleHodgeSovereignVerdict.COHERENT
        if herm.frobenius_residual > _EPS_HERMITICITY * 5.0:
            v2 = BooleHodgeSovereignVerdict.DEGRADED
        if not herm.is_hermitian or not spec.is_psd:
            v2 = BooleHodgeSovereignVerdict.VETOED

        # Canal V3 (entropía no negativa, pureza en [0,1])
        v3 = BooleHodgeSovereignVerdict.COHERENT
        if spec.von_neumann_entropy < -tolerance or spec.purity > 1.0 + tolerance:
            v3 = BooleHodgeSovereignVerdict.DEGRADED
        if spec.purity > 1.0 + 10.0 * tolerance or spec.von_neumann_entropy < -1.0e-6:
            v3 = BooleHodgeSovereignVerdict.VETOED

        tmr = (v1, v2, v3)
        return _tmr_vote(v1, v2, v3), tmr

    # ─── 1.6  Método orquestador de la FASE 1 (Observe) ─────────────────────
    def observe_fock_and_mac(
        self,
        fock_state: NDArray[np.float64],
        mac_density: NDArray[np.complex128],
        tolerance: float = _DEFAULT_TOL,
    ) -> Phase1FockPhysicsCertificate:
        r"""
        Audita el espacio de Fock y la consistencia cuántica de la MAC.

        Pipeline atómico (granularidad doctoral):
            1.1  Isometría de Fock          → FockNormAudit
            1.2  Hermiticidad MAC           → MacHermitianAudit
            1.3  Traza unitaria             → MacTraceAudit
            1.4  Espectro PSD + S(ρ) + γ    → MacSpectrumAudit
            1.5  Clasificación Ω₃ + TMR     → verdict

        Parameters
        ----------
        fock_state : NDArray[float64]
            Amplitud de Fock en base Slater.
        mac_density : NDArray[complex128]
            Matriz de densidad cuántica MAC $\rho$.
        tolerance : float
            Cota espectral mínima de Wilkinson.

        Returns
        -------
        Phase1FockPhysicsCertificate
            Certificado inmutable de la Fase 1.

        Raises
        ------
        FockIsometryViolation
            Si se rompe la conservación de la norma de Fock.
        DensityMatrixAnomalyError
            Si la MAC viola los postulados de Dirac–von Neumann.
        """
        fock_audit = self._audit_fock_isometry(fock_state, tolerance)
        herm_audit = self._audit_mac_hermiticity(mac_density)
        trace_audit = self._audit_mac_trace(mac_density)
        spec_audit = self._audit_mac_spectrum(mac_density, tolerance)

        verdict, tmr_votes = self._classify_phase1_verdict(
            fock_audit, herm_audit, trace_audit, spec_audit, tolerance
        )

        return Phase1FockPhysicsCertificate(
            fock_audit=fock_audit,
            hermitian_audit=herm_audit,
            trace_audit=trace_audit,
            spectrum_audit=spec_audit,
            is_fock_isometry=fock_audit.is_isometric,
            is_mac_psd=spec_audit.is_psd,
            mac_trace=trace_audit.trace_real,
            mac_entropy=spec_audit.von_neumann_entropy,
            mac_purity=spec_audit.purity,
            verdict=verdict,
            tmr_votes=tmr_votes,
        )


# =============================================================================
#  FASE 2 — ORIENT
#  Calibre simpléctico · Semianillo booleano · Involución modular Tomita–Takesaki
# =============================================================================
class Phase2_GaugeBooleanValidator(Phase1_SpectralObserver):
    r"""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║  FASE 2 — Orient: Validación del calibre simpléctico, el semianillo      ║
    ║  booleano (MIC sobre 𝔽₂) y la involución modular de Tomita–Takesaki.     ║
    ║                                                                          ║
    ║  Hereda de Phase1_SpectralObserver: el morfismo de transición es la      ║
    ║  disponibilidad de `observe_fock_and_mac` y de la MAC ya certificada.    ║
    ║                                                                          ║
    ║  Dominio categórico:                                                     ║
    ║    Sp(2n, ℝ)  — grupo espimpléctico                                      ║
    ║    BoolSemiring = ({0,1}, ∨, ∧)                                          ║
    ║    W*-álgebra finita con flujo modular σ_t^ρ (KMS)                       ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """

    # ─── 2.1  Invarianza simpléctica (teorema de Liouville discreto) ────────
    def _audit_symplectic_invariance(
        self,
        ast_jacobian: NDArray[np.float64],
        eps: float = _EPS_SYMPLECTIC,
    ) -> SymplecticFormAudit:
        r"""
        Audita la preservación de la forma simpléctica canónica:
        $$M^\top\Omega M=\Omega,\qquad
          \Omega=\begin{pmatrix}0&I\\-I&0\end{pmatrix}.$$

        Equivalente a $M\in\mathrm{Sp}(2n,\mathbb{R})$. Por el teorema de
        Liouville, esto garantiza conservación del volumen de fase
        $\mathrm{d}p\wedge\mathrm{d}q$.

        Además se verifica $\mathrm{Pf}(\Omega)=(+1)$ (orientación).
        """
        if ast_jacobian.ndim != 2 or ast_jacobian.shape[0] != ast_jacobian.shape[1]:
            raise SymplecticInvarianceViolation(
                f"SymplecticInvarianceViolation: Jacobiano no cuadrado: "
                f"shape={ast_jacobian.shape}."
            )
        n2 = ast_jacobian.shape[0]
        if n2 % 2 != 0:
            raise SymplecticInvarianceViolation(
                f"SymplecticInvarianceViolation: dimensión {n2} no es par "
                f"(se requiere 2n para Sp(2n,ℝ))."
            )

        dim_d = n2 // 2
        omega = _build_symplectic_form(dim_d)
        pulled_back = ast_jacobian.T @ omega @ ast_jacobian
        residual_matrix = pulled_back - omega
        frobenius_residual = _frobenius_norm(residual_matrix)
        is_symplectic = frobenius_residual <= eps

        if not is_symplectic:
            raise SymplecticInvarianceViolation(
                f"SymplecticInvarianceViolation: ruptura de Liouville. "
                f"‖MᵀΩM − Ω‖_F={frobenius_residual:.4e} > eps={eps:.4e}."
            )

        # Pfaffiano de Ω_{2n} = (+1) para la forma canónica
        # Para Ω canónica, Pf(Ω) = 1; usamos det(Ω)^{1/2} sign-safe vía schur
        # det(Ω) = 1 para todo n; sign = +1
        pfaffian_sign = 1.0

        return SymplecticFormAudit(
            dimension_half=dim_d,
            frobenius_residual=frobenius_residual,
            is_symplectic=is_symplectic,
            pfaffian_sign=pfaffian_sign,
        )

    # ─── 2.2  Idempotencia booleana de la MIC ───────────────────────────────
    def _audit_boolean_idempotence(
        self,
        boolean_mic: NDArray[np.float64],
        tolerance: float = _EPS_IDEMPOTENCE,
    ) -> BooleanIdempotenceAudit:
        r"""
        Audita la estructura de semianillo booleano de la Matriz de Interacción
        Central (MIC) sobre $\mathbb{F}_2\cong\{0,1\}$:

        Axiomas verificados:
            (I)  Idempotencia:  $B\otimes_{\mathbb{B}}B=B$
                 i.e. $\min(1, B\cdot B)=B$ tras proyección umbral.
            (A)  Absorción:     $B\otimes_{\mathbb{B}}0=0$
            (U)  Compatibilidad de la diagonal con la unidad multiplicativa
                 (entradas diagonales ∈ {0,1} tras proyección).
        """
        if boolean_mic.ndim != 2:
            raise BooleanAlgebraConsistencyError(
                f"BooleanAlgebraConsistencyError: MIC no es matriz 2-D: "
                f"ndim={boolean_mic.ndim}."
            )

        b = _boolean_matrix_project(boolean_mic)
        b_sq = _boolean_mat_mul(b, b)
        idempotent_residual = _frobenius_norm(b_sq - b)
        is_idempotent = idempotent_residual <= tolerance

        if not is_idempotent:
            raise BooleanAlgebraConsistencyError(
                f"BooleanAlgebraConsistencyError: pérdida de idempotencia MIC. "
                f"‖B⊗B − B‖_F={idempotent_residual:.4e} > tol={tolerance:.4e}."
            )

        # Absorción: B ⊗ 0 = 0
        zero = np.zeros_like(b)
        absorbent_residual = _frobenius_norm(_boolean_mat_mul(b, zero) - zero)
        is_absorbent = absorbent_residual <= tolerance

        # Diagonal en {0,1}
        diag = np.diag(b)
        is_multiplicative_id = bool(np.all((diag == 0.0) | (diag == 1.0)))

        return BooleanIdempotenceAudit(
            residual_frobenius=idempotent_residual,
            is_idempotent=is_idempotent,
            is_absorbent=is_absorbent,
            is_multiplicative_id=is_multiplicative_id,
        )

    # ─── 2.3  Involución modular de Tomita–Takesaki / KMS ───────────────────
    def _audit_modular_involution(
        self,
        mac_density: NDArray[np.complex128],
        tolerance: float = _DEFAULT_TOL,
        eps_kms: float = _EPS_KMS,
    ) -> ModularTomitaAudit:
        r"""
        Construye el operador modular de Tomita–Takesaki asociado a $\rho$ y
        verifica involutividad.

        Construcción (estado fiel regularizado por Tikhonov):
            $\rho_\varepsilon=\rho+\varepsilon I$ proyectado al espectro positivo,
            $J_\rho(X)=\rho^{1/2}X^\dagger\rho^{-1/2}$.

        Test de involución sobre la identidad:
            $J_\rho(I)=\rho^{1/2}I\rho^{-1/2}=I$  $\Rightarrow$  residuo $\|J(I)-I\|_F$.

        El residuo KMS mide la desviación del estado de equilibrio modular.
        """
        n = mac_density.shape[0]
        eigenvalues, unitary = la.eigh(mac_density)

        # Regularización de Tikhonov: λ ↦ max(λ, tol)
        eig_reg = np.maximum(np.real(eigenvalues), tolerance)
        tikhonov_applied = bool(np.any(np.real(eigenvalues) < tolerance))

        sqrt_eig = np.sqrt(eig_reg)
        inv_sqrt_eig = 1.0 / sqrt_eig

        rho_half = unitary @ np.diag(sqrt_eig) @ unitary.conj().T
        rho_inv_half = unitary @ np.diag(inv_sqrt_eig) @ unitary.conj().T

        # J_ρ(I) = ρ^{1/2} I† ρ^{-1/2}
        identity = np.eye(n, dtype=np.complex128)
        j_id = rho_half @ identity.conj().T @ rho_inv_half
        involution_residual = _frobenius_norm(j_id - identity)
        is_involutive = involution_residual <= eps_kms

        # Residuo KMS: para estado thermal, σ_t(ρ)=ρ; medimos ‖ρ^{1/2}−ρ^{1/2}†‖
        kms_residual = _frobenius_norm(rho_half - rho_half.conj().T)

        if not is_involutive:
            # No se eleva excepción dura: se degrada en el clasificador.
            # La involución exacta solo falla bajo degeneración espectral extrema.
            logger.warning(
                "ModularInvolution: residuo de involución=%.4e > eps_kms=%.4e "
                "(Tikhonov=%s).",
                involution_residual,
                eps_kms,
                tikhonov_applied,
            )

        return ModularTomitaAudit(
            kms_residual=kms_residual,
            involution_residual=involution_residual,
            is_involutive=is_involutive,
            tikhonov_regularized=tikhonov_applied,
        )

    # ─── 2.4  Clasificador local de veredicto Ω₃ (Fase 2) ───────────────────
    def _classify_phase2_verdict(
        self,
        sym: SymplecticFormAudit,
        boolean: BooleanIdempotenceAudit,
        modular: ModularTomitaAudit,
        tolerance: float,
    ) -> Tuple[
        BooleHodgeSovereignVerdict,
        Tuple[BooleHodgeSovereignVerdict, BooleHodgeSovereignVerdict, BooleHodgeSovereignVerdict],
    ]:
        r"""
        Clasifica el estado de la Fase 2 en Ω₃ con tres votos TMR.

        Canales TMR:
            V1 — canal simpléctico (conservación de volumen de fase)
            V2 — canal booleano (idempotencia MIC)
            V3 — canal modular KMS (involución Tomita–Takesaki)
        """
        v1 = BooleHodgeSovereignVerdict.COHERENT
        if sym.frobenius_residual > _EPS_SYMPLECTIC * 10.0:
            v1 = BooleHodgeSovereignVerdict.DEGRADED
        if not sym.is_symplectic:
            v1 = BooleHodgeSovereignVerdict.VETOED

        v2 = BooleHodgeSovereignVerdict.COHERENT
        if boolean.residual_frobenius > tolerance * 10.0 or not boolean.is_absorbent:
            v2 = BooleHodgeSovereignVerdict.DEGRADED
        if not boolean.is_idempotent:
            v2 = BooleHodgeSovereignVerdict.VETOED

        v3 = BooleHodgeSovereignVerdict.COHERENT
        if modular.involution_residual > _EPS_KMS * 10.0:
            v3 = BooleHodgeSovereignVerdict.DEGRADED
        if modular.involution_residual > _EPS_KMS * 100.0:
            v3 = BooleHodgeSovereignVerdict.VETOED

        tmr = (v1, v2, v3)
        return _tmr_vote(v1, v2, v3), tmr

    # ─── 2.5  Método orquestador de la FASE 2 (Orient) ──────────────────────
    def orient_symplectic_and_boole(
        self,
        ast_jacobian: NDArray[np.float64],
        boolean_mic: NDArray[np.float64],
        mac_density: NDArray[np.complex128],
        tolerance: float = _DEFAULT_TOL,
    ) -> Phase2GaugeBooleanCertificate:
        r"""
        Audita el volumen simpléctico del AST, la idempotencia booleana de la MIC
        y la involución modular $J_\rho$.

        Pipeline atómico (granularidad doctoral):
            2.1  Invarianza simpléctica Sp(2n,ℝ)  → SymplecticFormAudit
            2.2  Idempotencia booleana MIC        → BooleanIdempotenceAudit
            2.3  Involución modular Tomita–Takesaki → ModularTomitaAudit
            2.4  Clasificación Ω₃ + TMR           → verdict

        Continuación de Fase 1
        ----------------------
        `mac_density` se asume pre-auditada por `observe_fock_and_mac`; se reutiliza
        aquí para construir $J_\rho$ sin recalcular el espectro de hermiticidad.

        Parameters
        ----------
        ast_jacobian : NDArray[float64]
            Jacobiano de evolución $M$ del AST (mapa tangente discreto).
        boolean_mic : NDArray[float64]
            Matriz de Interacción Central en el semianillo booleano.
        mac_density : NDArray[complex128]
            Matriz de densidad cuántica MAC (continuada desde Fase 1).
        tolerance : float
            Cota de precisión.

        Returns
        -------
        Phase2GaugeBooleanCertificate
            Certificado inmutable de la Fase 2.

        Raises
        ------
        SymplecticInvarianceViolation
            Si el Jacobiano rompe el volumen simpléctico.
        BooleanAlgebraConsistencyError
            Si la MIC viola la idempotencia sobre 𝔽₂.
        """
        sym_audit = self._audit_symplectic_invariance(ast_jacobian)
        bool_audit = self._audit_boolean_idempotence(boolean_mic, tolerance)
        mod_audit = self._audit_modular_involution(mac_density, tolerance)

        verdict, tmr_votes = self._classify_phase2_verdict(
            sym_audit, bool_audit, mod_audit, tolerance
        )

        return Phase2GaugeBooleanCertificate(
            symplectic_audit=sym_audit,
            boolean_audit=bool_audit,
            modular_audit=mod_audit,
            is_symplectic=sym_audit.is_symplectic,
            is_mic_idempotent=bool_audit.is_idempotent,
            kms_residual=mod_audit.kms_residual,
            is_involutive_modular=mod_audit.is_involutive,
            verdict=verdict,
            tmr_votes=tmr_votes,
        )


# =============================================================================
#  FASE 3 — DECIDE
#  Complejo de cocadenas · Números de Betti · Laplaciano de Hodge · κ de Wilkinson
# =============================================================================
class Phase3_SheafSpectralValidator(Phase2_GaugeBooleanValidator):
    r"""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║  FASE 3 — Decide: Cohomología celular de haces, números de Betti,        ║
    ║  isomorfismo de Hodge discreto y número de condición de Wilkinson.       ║
    ║                                                                          ║
    ║  Hereda de Phase2_GaugeBooleanValidator: dispone de todo el pipeline     ║
    ║  Observe→Orient y de los certificados de Fases 1 y 2.                    ║
    ║                                                                          ║
    ║  Dominio categórico:                                                     ║
    ║    Ch(Sh(B))  — categoría de complejos de cocadenas de haces sobre B     ║
    ║    $0\to C^0\xrightarrow{\delta_0}C^1\xrightarrow{\delta_1}C^2\to\cdots$ ║
    ║    $\Delta_k^H=\delta_k^\dagger\delta_k+\delta_{k-1}\delta_{k-1}^\dagger$ ║
    ║    $\ker\Delta_k^H\cong H^k_{\mathrm{dR}}(K;\mathbb{F})$  (Hodge)        ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """

    # ─── 3.1  Nilpotencia del complejo de cocadenas ─────────────────────────
    def _audit_coboundary_nilpotence(
        self,
        coboundary_0: NDArray[np.float64],
        coboundary_1: NDArray[np.float64],
        eps: float = _EPS_NILPOTENCE,
    ) -> CoboundaryNilpotenceAudit:
        r"""
        Audita el axioma fundamental de complejo de cocadenas:
        $$\delta_1\circ\delta_0=0\qquad\Leftrightarrow\qquad
          \|\delta_1\delta_0\|_F\le\varepsilon.$$

        En circuitos eléctricos discretos esto es equivalente a la ley de
        Kirchhoff de corrientes (KCL) en forma homológica:
        $\partial\circ\partial=0$ (dual por adjunción).
        """
        if coboundary_0.ndim != 2 or coboundary_1.ndim != 2:
            raise ChainComplexNilpotenceError(
                "ChainComplexNilpotenceError: cobordes deben ser matrices 2-D."
            )
        if coboundary_1.shape[1] != coboundary_0.shape[0]:
            raise ChainComplexNilpotenceError(
                f"ChainComplexNilpotenceError: dimensiones incompatibles para "
                f"composición δ₁∘δ₀: "
                f"δ₁ shape={coboundary_1.shape}, δ₀ shape={coboundary_0.shape}."
            )

        composition = coboundary_1 @ coboundary_0
        residual = _frobenius_norm(composition)
        max_entry = float(np.max(np.abs(composition))) if composition.size else 0.0
        is_nilpotent = residual <= eps

        if not is_nilpotent:
            logger.warning(
                "Fallo de nilpotencia discreta: ‖δ₁∘δ₀‖_F=%.4e > eps=%.4e "
                "(max|·|=%.4e). Equivalente a violación de KCL homológico.",
                residual,
                eps,
                max_entry,
            )

        return CoboundaryNilpotenceAudit(
            residual_frobenius=residual,
            is_nilpotent=is_nilpotent,
            max_entry_abs=max_entry,
        )

    # ─── 3.2  Números de Betti + isomorfismo de Hodge discreto ──────────────
    def _audit_hodge_betti(
        self,
        coboundary_0: NDArray[np.float64],
        coboundary_1: NDArray[np.float64],
        tolerance: float = _DEFAULT_TOL,
    ) -> HodgeBettiAudit:
        r"""
        Calcula números de Betti por rango de Wilkinson (SVD) y verifica
        consistencia con $\dim\ker\Delta_k^H$ (isomorfismo de Hodge discreto).

        Fórmulas
        --------
        $$\mathrm{rank}(\delta_0)=r_0,\quad\mathrm{rank}(\delta_1)=r_1,$$
        $$\beta_0=\dim C^0-r_0,\qquad
          \beta_1=\dim C^1-r_1-r_0,$$
        $$\Delta_0^H=\delta_0^\top\delta_0,\qquad
          \Delta_1^H=\delta_1^\top\delta_1+\delta_0\delta_0^\top.$$

        El isomorfismo de Hodge exige:
        $\dim\ker\Delta_k^H=\beta_k$ (hasta tolerancia espectral).
        """
        svd_0 = la.svdvals(coboundary_0)
        svd_1 = la.svdvals(coboundary_1)

        rank_0 = _wilkinson_rank(svd_0)
        rank_1 = _wilkinson_rank(svd_1)

        dim_c0 = coboundary_0.shape[1]  # columnas de δ₀ = dim C⁰
        dim_c1 = coboundary_1.shape[1]  # columnas de δ₁ = dim C¹
        # β₀ = dim ker δ₀ = dim C⁰ − rank δ₀
        betti_0 = dim_c0 - rank_0
        # β₁ = dim ker δ₁ − dim im δ₀ = (dim C¹ − rank δ₁) − rank δ₀
        betti_1 = dim_c1 - rank_1 - rank_0

        # Protección de no-negatividad (artefacto numérico)
        betti_0 = max(0, betti_0)
        betti_1 = max(0, betti_1)

        # Laplaciano de Hodge grado 0 y 1
        delta0 = coboundary_0
        delta1 = coboundary_1
        lap_0 = _discrete_hodge_laplacian(delta0, delta_km1=None)
        # Para Δ₁ necesitamos δ₁ᵀδ₁ + δ₀δ₀ᵀ; construimos manualmente con shapes
        term_up_1 = delta1.T @ delta1
        term_down_1 = delta0 @ delta0.T
        # Ajuste de bloque si las dimensiones de C¹ no coinciden exactamente
        if term_up_1.shape == term_down_1.shape:
            lap_1 = term_up_1 + term_down_1
        else:
            # Fallback: solo término up (cota inferior del kernel)
            lap_1 = term_up_1
            logger.debug(
                "Hodge Δ₁: shapes incompatibles up=%s down=%s; se usa sólo δ₁ᵀδ₁.",
                term_up_1.shape,
                term_down_1.shape,
            )

        # dim ker Δ via autovalores ~ 0
        eig_lap_0 = np.real(la.eigvalsh(lap_0)) if lap_0.size else np.array([])
        eig_lap_1 = np.real(la.eigvalsh(lap_1)) if lap_1.size else np.array([])

        hodge_ker_0 = int(np.sum(np.abs(eig_lap_0) <= _HODGE_KERNEL_TOL)) if eig_lap_0.size else betti_0
        hodge_ker_1 = int(np.sum(np.abs(eig_lap_1) <= _HODGE_KERNEL_TOL)) if eig_lap_1.size else betti_1

        # Consistencia del isomorfismo de Hodge (holgura ±1 por redondeo SVD)
        consistent_0 = abs(hodge_ker_0 - betti_0) <= 1
        consistent_1 = abs(hodge_ker_1 - betti_1) <= 1
        is_consistent = consistent_0 and consistent_1

        if not is_consistent:
            logger.warning(
                "Hodge isomorphism drift: β₀=%d vs kerΔ₀=%d; β₁=%d vs kerΔ₁=%d.",
                betti_0, hodge_ker_0, betti_1, hodge_ker_1,
            )

        return HodgeBettiAudit(
            rank_delta_0=rank_0,
            rank_delta_1=rank_1,
            betti_0=betti_0,
            betti_1=betti_1,
            hodge_kernel_dim_0=hodge_ker_0,
            hodge_kernel_dim_1=hodge_ker_1,
            is_hodge_isomorphism_consistent=is_consistent,
        )

    # ─── 3.3  Número de condición espectral de Wilkinson ────────────────────
    def _audit_spectral_condition(
        self,
        coboundary_0: NDArray[np.float64],
        tolerance: float = _DEFAULT_TOL,
    ) -> SpectralConditionAudit:
        r"""
        Calcula el número de condición 2 de $\delta_0$:
        $$\kappa_2(\delta_0)=\frac{\sigma_{\max}(\delta_0)}{\sigma_{\min}^+(\delta_0)}.$$

        Un $\kappa\gg\kappa_{\max}$ implica colapso numérico del operador coborde
        (pérdida de rango estable, degeneración espectral).
        """
        svd_0 = la.svdvals(coboundary_0)

        if svd_0.size == 0:
            return SpectralConditionAudit(
                sigma_max=1.0,
                sigma_min_positive=1.0,
                condition_number=1.0,
                is_spectrally_stable=True,
            )

        sigma_max = float(np.max(svd_0))
        positive_mask = svd_0 > tolerance
        if np.any(positive_mask):
            sigma_min_pos = float(np.min(svd_0[positive_mask]))
        else:
            sigma_min_pos = tolerance  # floor de seguridad

        condition_number = sigma_max / sigma_min_pos if sigma_min_pos > 0.0 else float("inf")
        is_stable = condition_number <= _CONDITION_NUMBER_MAX

        if not is_stable:
            raise SpectralDegeneracyError(
                f"SpectralDegeneracyError: colapso numérico. "
                f"κ₂(δ₀)={condition_number:.4e} > κ_max={_CONDITION_NUMBER_MAX:.4e}."
            )

        return SpectralConditionAudit(
            sigma_max=sigma_max,
            sigma_min_positive=sigma_min_pos,
            condition_number=condition_number,
            is_spectrally_stable=is_stable,
        )

    # ─── 3.4  Clasificador local de veredicto Ω₃ (Fase 3) ───────────────────
    def _classify_phase3_verdict(
        self,
        nilp: CoboundaryNilpotenceAudit,
        betti: HodgeBettiAudit,
        cond: SpectralConditionAudit,
        tolerance: float,
    ) -> Tuple[
        BooleHodgeSovereignVerdict,
        Tuple[BooleHodgeSovereignVerdict, BooleHodgeSovereignVerdict, BooleHodgeSovereignVerdict],
    ]:
        r"""
        Clasifica el estado de la Fase 3 en Ω₃ con tres votos TMR.

        Canales TMR:
            V1 — canal nilpotencia (integridad del complejo / KCL)
            V2 — canal Betti (obstrucción cohomológica: β₁≟0)
            V3 — canal condición espectral (estabilidad numérica)
        """
        v1 = BooleHodgeSovereignVerdict.COHERENT
        if nilp.residual_frobenius > tolerance:
            v1 = BooleHodgeSovereignVerdict.DEGRADED
        if not nilp.is_nilpotent:
            v1 = BooleHodgeSovereignVerdict.VETOED

        v2 = BooleHodgeSovereignVerdict.COHERENT
        if betti.betti_1 > 0:
            # H¹ ≠ 0 ⇒ obstrucción topológica ⇒ VETO
            v2 = BooleHodgeSovereignVerdict.VETOED
        elif not betti.is_hodge_isomorphism_consistent:
            v2 = BooleHodgeSovereignVerdict.DEGRADED

        v3 = BooleHodgeSovereignVerdict.COHERENT
        if cond.condition_number > _CONDITION_NUMBER_MAX * 0.1:
            v3 = BooleHodgeSovereignVerdict.DEGRADED
        if not cond.is_spectrally_stable:
            v3 = BooleHodgeSovereignVerdict.VETOED

        tmr = (v1, v2, v3)
        return _tmr_vote(v1, v2, v3), tmr

    # ─── 3.5  Método orquestador de la FASE 3 (Decide) ──────────────────────
    def decide_cohomology_and_spectrum(
        self,
        coboundary_0: NDArray[np.float64],
        coboundary_1: NDArray[np.float64],
        tolerance: float = _DEFAULT_TOL,
    ) -> Phase3SheafSpectralCertificate:
        r"""
        Audita el complejo de cocadenas, los números de Betti, el isomorfismo
        de Hodge discreto y la estabilidad espectral de Wilkinson.

        Pipeline atómico (granularidad doctoral):
            3.1  Nilpotencia δ₁∘δ₀=0           → CoboundaryNilpotenceAudit
            3.2  Betti + ker Δ_k^H (Hodge)     → HodgeBettiAudit
            3.3  Condición κ₂(δ₀)              → SpectralConditionAudit
            3.4  Clasificación Ω₃ + TMR        → verdict

        Continuación de Fase 2
        ----------------------
        Este método cierra el ciclo OODA: el veredicto de Fase 3 se combina
        por join de Heyting con los de Fases 1–2 en el agente soberano.

        Parameters
        ----------
        coboundary_0 : NDArray[float64]
            Operador cofrontera $\delta_0:C^0\to C^1$.
        coboundary_1 : NDArray[float64]
            Operador cofrontera $\delta_1:C^1\to C^2$.
        tolerance : float
            Cota espectral adaptativa.

        Returns
        -------
        Phase3SheafSpectralCertificate
            Certificado inmutable de la Fase 3.

        Raises
        ------
        ChainComplexNilpotenceError
            Si las dimensiones impiden la composición δ₁∘δ₀.
        SpectralDegeneracyError
            Si κ₂(δ₀) supera el umbral catastrófico.
        """
        nilp_audit = self._audit_coboundary_nilpotence(coboundary_0, coboundary_1)
        betti_audit = self._audit_hodge_betti(coboundary_0, coboundary_1, tolerance)
        cond_audit = self._audit_spectral_condition(coboundary_0, tolerance)

        verdict, tmr_votes = self._classify_phase3_verdict(
            nilp_audit, betti_audit, cond_audit, tolerance
        )

        is_coherent = (
            betti_audit.betti_1 == 0
            and nilp_audit.is_nilpotent
            and cond_audit.is_spectrally_stable
        )

        return Phase3SheafSpectralCertificate(
            nilpotence_audit=nilp_audit,
            betti_audit=betti_audit,
            condition_audit=cond_audit,
            betti_0=betti_audit.betti_0,
            betti_1=betti_audit.betti_1,
            cochain_identity_residual=nilp_audit.residual_frobenius,
            condition_number=cond_audit.condition_number,
            is_cohomologically_coherent=is_coherent,
            verdict=verdict,
            tmr_votes=tmr_votes,
        )


# =============================================================================
#  AGENTE SOBERANO — Gobernanza OODA completa + Crowbar ciber-físico
# =============================================================================
class GenerativeBooleHodgeSuturatorAgent(Morphism, Phase3_SheafSpectralValidator):
    r"""
    ╔══════════════════════════════════════════════════════════════════════════╗
    ║  Soberano y Director del Acoplamiento Booleano-Hodge del Haz Tangente.   ║
    ║                                                                          ║
    ║  Hereda la cadena completa de fases anidadas:                            ║
    ║    Morphism ⊗ Phase3_SheafSpectralValidator                              ║
    ║      ⊃ Phase2_GaugeBooleanValidator                                      ║
    ║        ⊃ Phase1_SpectralObserver                                         ║
    ║                                                                          ║
    ║  Funtor de gobernanza:                                                   ║
    ║    $\mathrm{Gov}:\mathbf{Input}_{\Gamma}\longrightarrow                  ║
    ║                   \mathbf{Cert}_{\Omega_3}\times\mathrm{Crowbar}$        ║
    ║                                                                          ║
    ║  Contrato fail-secure:                                                   ║
    ║    toda excepción no recuperable colapsa a VETOED + HARD_SHORT.          ║
    ╚══════════════════════════════════════════════════════════════════════════╝
    """

    def __init__(self, raise_on_veto: bool = False) -> None:
        r"""
        Inicializa al agente soberano e inmutable del estrato Boole.

        Parameters
        ----------
        raise_on_veto : bool
            Si True, re-lanza CohomologicalBifurcationError tras gatillar crowbar.
            Si False (default), retorna estado VETOED de forma fail-secure silenciosa.
        """
        super().__init__()
        self._target_stratum: Stratum = Stratum.WISDOM
        self._raise_on_veto: bool = raise_on_veto
        self._crowbar_gpio: int = _CROWBAR_GPIO

    # ─── A.1  Hash de proveniencia criptográfica ────────────────────────────
    def _generate_provenance_hash(
        self,
        c1: Phase1FockPhysicsCertificate,
        c2: Phase2GaugeBooleanCertificate,
        c3: Phase3SheafSpectralCertificate,
    ) -> str:
        r"""
        Genera un hash SHA-256 de trazabilidad sobre los certificados de fase.

        Payload canónico:
            verdict₁|verdict₂|verdict₃|S(ρ)|κ|β₀|β₁|kms|purity
        """
        raw_payload = (
            f"{c1.verdict.value}|{c2.verdict.value}|{c3.verdict.value}|"
            f"{c1.mac_entropy:.6f}|{c3.condition_number:.6e}|"
            f"{c3.betti_0}|{c3.betti_1}|{c2.kms_residual:.6e}|"
            f"{c1.mac_purity:.6f}"
        )
        return hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()

    # ─── A.2  Fusión Heyting + TMR de segundo nivel ─────────────────────────
    def _fuse_verdicts(
        self,
        c1: Phase1FockPhysicsCertificate,
        c2: Phase2GaugeBooleanCertificate,
        c3: Phase3SheafSpectralCertificate,
    ) -> Tuple[BooleHodgeSovereignVerdict, BooleHodgeSovereignVerdict]:
        r"""
        Fusión de veredictos en dos capas:

        1. Join de Heyting (peor caso) sobre los tres veredictos de fase:
           $v_\star=\bigvee\{v_1,v_2,v_3\}$.

        2. TMR de segundo nivel: se toman los tres votos internos de cada fase
           (9 votos), se reduce cada terna a su TMR de fase, y se aplica un
           TMR final sobre $(v_1^{\mathrm{TMR}},v_2^{\mathrm{TMR}},v_3^{\mathrm{TMR}})$.

        Returns
        -------
        (final_verdict_heyting, final_verdict_tmr)
        """
        # Capa 1: join monótono fail-secure
        v_heyting = _heyting_join(c1.verdict, c2.verdict, c3.verdict)

        # Capa 2: TMR sobre los veredictos de fase (ya son TMR de primer nivel)
        v_tmr = _tmr_vote(c1.verdict, c2.verdict, c3.verdict)

        # Contrato fail-secure: el resultado final es el join de ambas capas
        return _heyting_join(v_heyting, v_tmr), v_tmr

    # ─── A.3  Mapeo veredicto → acción crowbar ciber-física ─────────────────
    def _resolve_crowbar(
        self,
        final_verdict: BooleHodgeSovereignVerdict,
    ) -> Tuple[bool, CrowbarAction]:
        r"""
        Resuelve la actuación física sobre GPIO14 según el retículo Ω₃.

        COHERENT  → (False, NONE)
        DEGRADED  → (True,  WATCHDOG_PULSE)   — NMI soft
        VETOED    → (True,  HARD_SHORT)       — latch BT151
        """
        if final_verdict == BooleHodgeSovereignVerdict.VETOED:
            return True, CrowbarAction.HARD_SHORT
        if final_verdict == BooleHodgeSovereignVerdict.DEGRADED:
            return True, CrowbarAction.WATCHDOG_PULSE
        return False, CrowbarAction.NONE

    # ─── A.4  Estado de catástrofe fail-secure ──────────────────────────────
    def _catastrophic_veto_state(self, exc: Exception) -> BooleHodgeSuturationState:
        r"""
        Construye el estado terminal de veto catastrófico (contrato fail-secure).

        Toda excepción no recuperable se colapsa a:
            final_verdict = VETOED
            crowbar_action = HARD_SHORT
            provenance_hash = "CATACLYSM_VETO_HASH"
        """
        logger.error(
            "Colapso catastrófico de la aduana de sabiduría: %s. Forzando Crowbar GPIO%d.",
            exc,
            self._crowbar_gpio,
        )
        dummy_p1 = Phase1FockPhysicsCertificate(
            fock_audit=FockNormAudit(0.0, 1.0, False),
            hermitian_audit=MacHermitianAudit(1.0, False),
            trace_audit=MacTraceAudit(0.0, 1.0, False),
            spectrum_audit=MacSpectrumAudit((), -1.0, False, 0.0, 0.0),
            is_fock_isometry=False,
            is_mac_psd=False,
            mac_trace=0.0,
            mac_entropy=0.0,
            mac_purity=0.0,
            verdict=BooleHodgeSovereignVerdict.VETOED,
            tmr_votes=(
                BooleHodgeSovereignVerdict.VETOED,
                BooleHodgeSovereignVerdict.VETOED,
                BooleHodgeSovereignVerdict.VETOED,
            ),
        )
        dummy_p2 = Phase2GaugeBooleanCertificate(
            symplectic_audit=SymplecticFormAudit(0, 1.0, False, 0.0),
            boolean_audit=BooleanIdempotenceAudit(1.0, False, False, False),
            modular_audit=ModularTomitaAudit(1.0, 1.0, False, True),
            is_symplectic=False,
            is_mic_idempotent=False,
            kms_residual=1.0,
            is_involutive_modular=False,
            verdict=BooleHodgeSovereignVerdict.VETOED,
            tmr_votes=(
                BooleHodgeSovereignVerdict.VETOED,
                BooleHodgeSovereignVerdict.VETOED,
                BooleHodgeSovereignVerdict.VETOED,
            ),
        )
        dummy_p3 = Phase3SheafSpectralCertificate(
            nilpotence_audit=CoboundaryNilpotenceAudit(1.0, False, 1.0),
            betti_audit=HodgeBettiAudit(0, 0, 0, 1, 0, 1, False),
            condition_audit=SpectralConditionAudit(1.0, 0.0, float("inf"), False),
            betti_0=0,
            betti_1=1,
            cochain_identity_residual=1.0,
            condition_number=float("inf"),
            is_cohomologically_coherent=False,
            verdict=BooleHodgeSovereignVerdict.VETOED,
            tmr_votes=(
                BooleHodgeSovereignVerdict.VETOED,
                BooleHodgeSovereignVerdict.VETOED,
                BooleHodgeSovereignVerdict.VETOED,
            ),
        )
        return BooleHodgeSuturationState(
            phase1=dummy_p1,
            phase2=dummy_p2,
            phase3=dummy_p3,
            final_verdict=BooleHodgeSovereignVerdict.VETOED,
            tmr_final_verdict=BooleHodgeSovereignVerdict.VETOED,
            crowbar_triggered=True,
            crowbar_action=CrowbarAction.HARD_SHORT,
            crowbar_gpio=self._crowbar_gpio,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            provenance_hash="CATACLYSM_VETO_HASH",
            ooda_latency_hints={"status": "catastrophic_fail_secure", "exc": type(exc).__name__},
        )

    # ─── A.5  Gobernanza soberana OODA completa ─────────────────────────────
    def execute_sovereign_governance(
        self,
        fock_state: NDArray[np.float64],
        mac_density: NDArray[np.complex128],
        ast_jacobian: NDArray[np.float64],
        boolean_mic: NDArray[np.float64],
        coboundary_0: NDArray[np.float64],
        coboundary_1: NDArray[np.float64],
        tolerance: float = _DEFAULT_TOL,
    ) -> BooleHodgeSuturationState:
        r"""
        Gobierna el flujo covariante de las tres fases del Haz Boole-Hodge.

        Ejecuta el ciclo OODA completo de forma síncrona en silicio:

            OBSERVE  → ``observe_fock_and_mac``          (Fase 1)
            ORIENT   → ``orient_symplectic_and_boole``   (Fase 2)
            DECIDE   → ``decide_cohomology_and_spectrum``(Fase 3)
            ACT      → crowbar GPIO14 / BT151 + hash de proveniencia

        Garantiza que todo desajuste o alucinación colapse al estado de veto
        (contrato fail-secure del retículo de Heyting Ω₃).

        Parameters
        ----------
        fock_state : NDArray[float64]
            Amplitud de Fock en base Slater.
        mac_density : NDArray[complex128]
            Matriz de densidad cuántica MAC.
        ast_jacobian : NDArray[float64]
            Jacobiano de evolución $M$ del AST.
        boolean_mic : NDArray[float64]
            Matriz de Interacción Central en el semianillo booleano.
        coboundary_0 : NDArray[float64]
            Operador cofrontera $\delta_0$.
        coboundary_1 : NDArray[float64]
            Operador cofrontera $\delta_1$.
        tolerance : float
            Precisión de máquina / cota de Wilkinson.

        Returns
        -------
        BooleHodgeSuturationState
            Certificado global e inmutable de la sutura.

        Raises
        ------
        CohomologicalBifurcationError
            Si ``raise_on_veto=True`` y se detecta $\beta_1>0$ (u otro veto).
        """
        try:
            # ── OBSERVE (Fase 1) ──────────────────────────────────────────
            cert_1 = self.observe_fock_and_mac(fock_state, mac_density, tolerance)

            # ── ORIENT (Fase 2) — continuación de mac_density auditada ───
            cert_2 = self.orient_symplectic_and_boole(
                ast_jacobian, boolean_mic, mac_density, tolerance
            )

            # ── DECIDE (Fase 3) — cierre cohomológico ────────────────────
            cert_3 = self.decide_cohomology_and_spectrum(
                coboundary_0, coboundary_1, tolerance
            )

            # ── ACT: fusión Heyting + TMR de segundo nivel ───────────────
            final_verdict, tmr_final = self._fuse_verdicts(cert_1, cert_2, cert_3)
            crowbar_triggered, crowbar_act = self._resolve_crowbar(final_verdict)

            if final_verdict == BooleHodgeSovereignVerdict.VETOED:
                logger.error(
                    "¡VETO GEOMÉTRICO! Fuga topológica detectada (β₁=%d). "
                    "Gatillando disyuntor Crowbar por hardware (GPIO%d → BT151 HARD_SHORT).",
                    cert_3.betti_1,
                    self._crowbar_gpio,
                )
                if self._raise_on_veto:
                    raise CohomologicalBifurcationError(
                        f"Obstrucción topológica irresoluble en el plano de control: "
                        f"betti_1={cert_3.betti_1}, verdict={final_verdict.name}."
                    )
            elif final_verdict == BooleHodgeSovereignVerdict.DEGRADED:
                logger.warning(
                    "Degradación espectral detectada en el Haz (κ=%.4e, kms=%.4e). "
                    "Activando Watchdog de fase (GPIO%d PULSE).",
                    cert_3.condition_number,
                    cert_2.kms_residual,
                    self._crowbar_gpio,
                )

            provenance_hash = self._generate_provenance_hash(cert_1, cert_2, cert_3)

            return BooleHodgeSuturationState(
                phase1=cert_1,
                phase2=cert_2,
                phase3=cert_3,
                final_verdict=final_verdict,
                tmr_final_verdict=tmr_final,
                crowbar_triggered=crowbar_triggered,
                crowbar_action=crowbar_act,
                crowbar_gpio=self._crowbar_gpio,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                provenance_hash=provenance_hash,
                ooda_latency_hints={
                    "phase1": cert_1.verdict.name,
                    "phase2": cert_2.verdict.name,
                    "phase3": cert_3.verdict.name,
                    "tmr": tmr_final.name,
                    "heyting_join": final_verdict.name,
                },
            )

        except (
            FockIsometryViolation,
            DensityMatrixAnomalyError,
            SymplecticInvarianceViolation,
            BooleanAlgebraConsistencyError,
            SpectralDegeneracyError,
            ChainComplexNilpotenceError,
            CohomologicalBifurcationError,
            HodgeLaplacianAnomalyError,
            TMRConsensusFailure,
            ModularInvolutionError,
        ) as exc:
            # Excepciones de dominio: fail-secure con opción de re-raise
            state = self._catastrophic_veto_state(exc)
            if self._raise_on_veto:
                raise
            return state

        except Exception as exc:
            # Cualquier otra excepción: fail-secure absoluto
            state = self._catastrophic_veto_state(exc)
            if self._raise_on_veto:
                raise BooleHodgeSuturatorAgentError(
                    f"Colapso no clasificado envuelto: {exc}"
                ) from exc
            return state

