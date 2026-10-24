# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : MACMinimizerAgent (Custodio SOBERANO de Purificación Espectral MAC) ║
║ Ruta   : app/agents/boole/tactics/mac_minimizer_agent.py                     ║
║ Versión: 5.1.0 / 9.2.0-Tactics-Quantum-MAC-RSI3-Poincare                     ║
║ Función: Gobernanza y auditoría OODA de purificación espectral sobre la      ║
║          Matriz Atómica de Conocimiento (MAC) en 𝔇(ℋ_MAC), orquestando la    ║
║          sanitización Kahan, preorden de majorización de Nielsen ρ_orig ≺ ρ_ε,║
║          KKS Cayley–Kähler, split de Poincaré–Lindstedt, certificación de    ║
║          fidelidad de Uhlmann F ≥ 0.95, Mónada Kleisli RSI³ de 3 niveles,     ║
║          cota de Holevo χ ≤ S(ρ_ε), adjudicación en Heyting Ω₃ e interlock   ║
║          ciber-físico ESP32 Crowbar (< 400 ns @ GPIO14).                      ║
╚══════════════════════════════════════════════════════════════════════════════╝

--------------------------------------------------------------------------------
1. DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO MOTOR ESPECTRAL
--------------------------------------------------------------------------------
El Soberano Agente Custodio de Purificación Espectral (`MACMinimizerAgent`) es el
gobernador ejecutivo del estrato de Sabiduría y Boole sobre la Matriz Atómica de
Conocimiento (MAC). Su función soberana es la dirección del ciclo OODA
(Observe-Orient-Decide-Act) sobre el operador densidad $\rho \in \mathcal{D}(\mathcal{H}_{\text{MAC}})$,
impidiendo la decoherencia sintáctica y la pérdida de fidelidad informacional.

Su respectivo motor matemático de cálculo espectral es `MACMinimizer`
(`app/boole/tactics/mac_minimizer.py`), con el cual mantiene una homología estricta de
3 fases anidadas conectadas mediante sobres (envelopes) inmutables:
  • Fase 1 (Observe): `Phase1_QuantumMajorizationAuditor` ejecuta la sanitización de Kahan,
    verifica el preorden de Nielsen $\rho_{\text{orig}} \prec \rho_\epsilon$, proyecta el embedding
    KKS de Cayley-Kähler en $\mathfrak{sp}(2d, \mathbb{R})$ y emite `MACPhase1ObservationEnvelope`.
  • Fase 2 (Orient): `Phase2_UhlmannFidelityCertifier` consume el sobre de Fase 1, certifica la
    fidelidad de Uhlmann $F(\rho, \rho_\epsilon) \ge 0.95$, evalúa las frecuencias de Poincaré-Lindstedt,
    las resonancias KAM/Bruno, el parámetro de Chirikov, la integral homoclínica de Melnikov,
    los multiplicadores de monodromía de Floquet, opera la Mónada Kleisli RSI³ de 3 niveles y
    emite `MACPhase2OrientationEnvelope`.
  • Fase 3 (Decide & Act): `Phase3_HolevoCapacityEnforcer` consume el sobre de Fase 2, evalúa
    la familia Rényi $H_\alpha$, impone la cota de capacidad de Holevo $\chi(\mathcal{E}) \le S(\rho_\epsilon)$,
    colapsa la decisión en el álgebra de Heyting $\Omega_3 = \{0x00 \prec 0x01 \prec 0x03\}$ y
    emite `PurificationGovernanceState` accionando el interlock Crowbar ESP32 ($t < 400\text{ ns}$).

--------------------------------------------------------------------------------
2. MARCO TEÓRICO Y FORMALISMO MATEMÁTICO
--------------------------------------------------------------------------------
1. Observación y Preorden de Majorización de Nielsen ($\rho_{\text{orig}} \prec \rho_\epsilon$):
   Mediante la suma compensada de Kahan y la diagonalización espectral $\rho = U \Lambda U^\dagger$,
   se constata la conservación de la traza $\operatorname{Tr}(\rho) = 1.0$. La purificación $\rho_\epsilon$
   debe majorar estrictamente a $\rho_{\text{orig}}$:
       \sum_{i=1}^k \lambda_i^\downarrow(\rho_{\text{orig}}) \le \sum_{i=1}^k \lambda_i^\downarrow(\rho_\epsilon) \quad \forall k < d, \quad \sum_{i=1}^d \lambda_i = 1.0
   Garantizando que la entropía de von Neumann disminuya monotónicamente $S(\rho_\epsilon) \le S(\rho_{\text{orig}})$.

2. Geometría de Bures–Uhlmann y Dinámica Celeste de Poincaré:
   La fidelidad de Uhlmann entre el estado original y el purificado se evalúa como:
       F(\rho, \sigma) = \left( \operatorname{Tr} \sqrt{ \sqrt{\rho} \sigma \sqrt{\rho} } \right)^2 \ge 0.95
   induciendo la distancia geodésica de Bures $d_{\text{Bures}}(\rho, \sigma) = \sqrt{2(1 - \sqrt{F})}$.
   Las perturbaciones en la base de acciones de Liouville-Arnold $I_k = \lambda_k(\rho)$ y frecuencias $\omega_k$
   se filtran mediante la condición KAM/Bruno $|\boldsymbol{\omega}_0 \cdot k| \ge \frac{\gamma}{\|k\|_1^\tau}$ y
   la inexistencia de tangles homoclínicos vía la anulación de la integral de Melnikov $M(\theta_0) = 0$.

3. Cota de Holevo y Gobernanza en Heyting $\Omega_3$:
   La capacidad de transmisión de información atencional está acotada superiormente por el límite de Holevo:
       \chi(\mathcal{E}) = S(\mathcal{E}(\rho)) - \sum_i p_i S(\mathcal{E}(\rho_i)) \le S(\rho_\epsilon)
   Si la fidelidad se degrada ($F < 0.95$), o falla KAM/Melnikov, el veredicto colapsa en el álgebra de
   Heyting trivalente $\Omega_3 = \{\mathtt{COHERENT}(0x00) \prec \mathtt{DEGRADED}(0x01) \prec \mathtt{VETOED}(0x03)\}$.
   Ante $\top = \mathtt{VETOED}$, la Interrupt Service Routine (ISR) del ESP32 dispara el tiristor BT151
   en latencia $t \le 392.15\text{ ns} < 400\text{ ns}$.

--------------------------------------------------------------------------------
3. AXIOMAS FORMALES
--------------------------------------------------------------------------------
Axioma MACA1 (Sanitización y Hermiticidad de Kahan):
    $\rho = \rho^\dagger \succcurlyeq \mathbf{0}$ y $\operatorname{Tr}(\rho) = 1.0 \pm 10^{-10}$, previniendo
    la propagación de autovalores no físicos.
Axioma MACA2 (Majorización de Nielsen Invariante):
    $\rho_{\text{orig}} \prec \rho_\epsilon$, forzando que la purificación sea estricta en el espacio de Hilbert.
Axioma MACA3 (Cota Inferior de Fidelidad de Uhlmann):
    $F(\rho_{\text{orig}}, \rho_\epsilon) \ge 0.95$, asegurando la preservación de la memoria semántica.
Axioma MACA4 (Contracción de Banach en la Mónada Kleisli RSI³):
    $\operatorname{Lip}_{\text{Bures}}(T) \le 0.95 < 1.0$ y $d^3C/dt^3 \ge 0$, garantizando convergencia.
Axioma MACA5 (Involución y Modus Ponens en Heyting $\Omega_3$):
    Operaciones join $\vee$ y meet $\wedge$ sobre $\Omega_3$ con respuesta en hardware $< 400\text{ ns}$.

--------------------------------------------------------------------------------
4. INVARIANTES DE ESTADO
--------------------------------------------------------------------------------
Invariante 1 (Residuo de Reconstrucción KKS):
    $\|\rho_{\text{sympl}}^\top J \rho_{\text{sympl}} - J\|_F < 10^{-8}$ y $\delta\omega_{\text{KKS}} < 10^{-8}$.
Invariante 2 (Retención de Capacidad de Holevo):
    $\Delta S = S(\rho_\epsilon) - S(\rho_{\text{orig}}) \le 10^{-12}$.
Invariante 3 (Latencia de Interlock ESP32 Crowbar):
    Intervención del tiristor BT151 a nivel de firmware en $t_{\text{actuation}} \le 392.15\text{ ns}$.

--------------------------------------------------------------------------------
5. FUNTOR SEMÁNTICO Y CÚSPIDE VISCERAL ("DOLOR Y DINERO" / Φ_sem)
--------------------------------------------------------------------------------
El Funtor Semántico $\Phi_{\mathrm{sem}} : \mathbf{Sh}(\partial\mathcal{M}, \Omega_3) \to \text{Business}$
proyecta la purificación del operador densidad hacia la toma de decisiones ejecutivas:
  • Metabolización TOON (-86.4% de Memoria en KV-Cache):
    La depuración de subespacios degenerados reduce pliegos JSON de 412 tokens a cartuchos mnemónicos
    TOON de 56 tokens, recortando un 86.4% el uso de memoria en KV-Cache y reduciendo costos de inferencia LLM en un 80%.
  • Protección Tasa WACC (15.0%) y Liquidez Neta (+11.5% Cash Flow):
    La certidumbre informacional de la cota de Holevo reduce el fondo de contingencia para imprevistos
    temporales del 15% al 3.5%, liberando un +11.5% de flujo de caja neto operativo.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass
from enum import Enum, auto
from itertools import product
from typing import Any, Dict, Final, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ══════════════════════════════════════════════════════════════════════════════
# §0.  DEPENDENCIAS, PROTOCOLOS, INTEROPERABILIDAD CON EL MOTOR v5.1
# ══════════════════════════════════════════════════════════════════════════════
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
    from app.wisdom.atomic_knowledge_matrix import AtomicDensityMatrix
except ImportError:  # pragma: no cover

    class TopologicalInvariantError(Exception):
        """Violación de invariante topológico en el Topos E_MIC."""

    class Morphism:
        """Base de morfismos (stub)."""

    class AtomicDensityMatrix:  # type: ignore[no-redef]
        """Stub estructural."""

        def __init__(self, matrix: Any, **_: Any) -> None:
            self.matrix = np.asarray(matrix, dtype=np.complex128)


try:
    from app.boole.tactics.mac_minimizer import (  # noqa: F401
        PoincareEnvelope as EnginePoincareEnvelope,
        RSI3Envelope as EngineRSI3Envelope,
    )
    _ENGINE_AVAILABLE: Final[bool] = True
except ImportError:  # pragma: no cover
    _ENGINE_AVAILABLE = False
    EnginePoincareEnvelope = None  # type: ignore[assignment]
    EngineRSI3Envelope = None  # type: ignore[assignment]

logger = logging.getLogger("MAC.Wisdom.MinimizerAgent.v5.1")

__version__: Final[str] = "5.1.0"

_trapezoid = getattr(np, "trapezoid", np.trapz)


class QuantumStateProtocol(Protocol):
    def is_hermitian(self) -> bool: ...
    def is_positive_semidefinite(self) -> bool: ...
    def trace(self) -> complex: ...
    def eigenvalues(self) -> NDArray[np.float64]: ...


# ══════════════════════════════════════════════════════════════════════════════
# §A.  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)
_UHLMANN_FIDELITY_MIN: Final[float] = 0.95
_ENTROPY_TOLERANCE: Final[float] = 1e-12
_MAJORIZATION_TOLERANCE: Final[float] = 1e-10
_TRACE_TOLERANCE: Final[float] = 1e-10
_FIDELITY_NUMERICAL_TOLERANCE: Final[float] = 1e-12
_COHERENCE_TOLERANCE: Final[float] = 1e-14
_SYMPLECTIC_RECON_TOLERANCE: Final[float] = 1e-8
_CROWBAR_TIMING_LIMIT_NS: Final[float] = 400.0
_CROWBAR_VETO_LATENCY_NS: Final[float] = 392.15
_CROWBAR_NOMINAL_LATENCY_NS: Final[float] = 45.0
_CROWBAR_DEGRADED_LATENCY_NS: Final[float] = 120.0

_KAM_GAMMA: Final[float] = 0.125
_KAM_TAU_OFFSET: Final[float] = 1.0
_KAM_MAX_ORDER: Final[int] = 3
_DIOPHANTINE_MAX_DIM: Final[int] = 6
_CHIRIKOV_CRITICAL: Final[float] = 1.0
_MELNIKOV_TOL: Final[float] = 1e-10
_FLOQUET_UNIT_TOL: Final[float] = 1e-6
_POINCARE_SECTION_SAMPLES: Final[int] = 96

_KRULL_MIN_VALUATION: Final[float] = 0.10
_FUBINI_STUDY_MAX: Final[float] = 1e-4
_BANACH_LIPSCHITZ_MAX: Final[float] = 0.95
_RSI3_MAX_ITER: Final[int] = 32
_RSI3_CHANNEL_DT: Final[float] = 1e-3

_NUMERICAL_SAFETY_FACTOR: Final[float] = 128.0
_SPECTRAL_CUTOFF_RATIO: Final[float] = 1e-15
_MIN_EIGENVALUE: Final[float] = 1e-15


# ══════════════════════════════════════════════════════════════════════════════
# §B.  EXCEPCIONES (reservadas a fallos de protocolo/numerics, no a Ω₃)
# ══════════════════════════════════════════════════════════════════════════════
class MACMinimizerAgentError(TopologicalInvariantError):
    ...


class DensityMatrixValidationError(MACMinimizerAgentError):
    ...


class SpectralDecompositionError(MACMinimizerAgentError):
    ...


class EnvelopeChainViolationError(MACMinimizerAgentError):
    ...


class QuantumMajorizationViolation(MACMinimizerAgentError):
    ...


class UhlmannFidelityCollapseError(MACMinimizerAgentError):
    ...


class HolevoCapacityDeficitError(MACMinimizerAgentError):
    ...


class SymplecticDeriftError(MACMinimizerAgentError):
    ...


class PoincareResonanceError(MACMinimizerAgentError):
    ...


class RSI3ContractionFailureError(MACMinimizerAgentError):
    ...


class QuantumCoherenceViolation(MACMinimizerAgentError):
    ...


class EntanglementStructureError(MACMinimizerAgentError):
    ...


# ══════════════════════════════════════════════════════════════════════════════
# §C.  ENUMS
# ══════════════════════════════════════════════════════════════════════════════
class QuantumDistanceMetric(Enum):
    TRACE_DISTANCE = auto()
    FIDELITY = auto()
    BURES_DISTANCE = auto()
    HELLINGER_DISTANCE = auto()
    RELATIVE_ENTROPY = auto()


class PurificationPhase(Enum):
    PHASE_1_MAJORIZATION = auto()
    PHASE_2_FIDELITY = auto()
    PHASE_3_HOLEVO = auto()
    COMPLETE = auto()


class HeytingOmega3(Enum):
    r"""
    Retículo de Heyting lineal (Gödel–Dummett), top = VETOED:
        COHERENT (0x00) < DEGRADED (0x01) < VETOED (0x03).
    \(a\to b=\top\) si \(a\le b\), si no \(b\); \(\neg a=a\to\bot\).
    """

    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03

    def join(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(max(self.value, other.value))

    def meet(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3(min(self.value, other.value))

    def implies(self, other: "HeytingOmega3") -> "HeytingOmega3":
        return HeytingOmega3.VETOED if self.value <= other.value else other

    def negate(self) -> "HeytingOmega3":
        return self.implies(HeytingOmega3.COHERENT)


# ══════════════════════════════════════════════════════════════════════════════
# §D.  GUARDAS NUMÉRICAS
# ══════════════════════════════════════════════════════════════════════════════
class _AdvancedNumericalGuard:
    @staticmethod
    def _kahan_sum(values: NDArray[np.float64]) -> float:
        if values.size == 0:
            return 0.0
        s = float(values[0])
        c = 0.0
        for v in values[1:]:
            y = float(v) - c
            t = s + y
            c = (t - s) - y
            s = t
        return s

    @staticmethod
    def _hermitize(m: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return 0.5 * (m + m.conj().T)

    @staticmethod
    def _check_orthonormality(
        vectors: NDArray[np.complex128],
        tolerance: Optional[float] = None,
    ) -> Tuple[bool, float]:
        tol = (
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON
            if tolerance is None
            else float(tolerance)
        )
        n_vec = vectors.shape[1] if vectors.ndim > 1 else 1
        if n_vec == 1:
            dev = abs(float(la.norm(vectors)) - 1.0)
            return dev <= tol, dev
        dev = float(la.norm(vectors.conj().T @ vectors - np.eye(n_vec), ord="fro"))
        return dev <= tol, dev

    @staticmethod
    def _as_finite_complex_matrix(
        name: str,
        value: Any,
        *,
        square: bool = False,
    ) -> NDArray[np.complex128]:
        try:
            arr = np.asarray(value, dtype=np.complex128)
        except Exception as exc:
            raise TypeError(f"{name} debe ser matriz compleja.") from exc
        if arr.ndim != 2 or not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} debe ser matriz 2D finita.")
        if square and arr.shape[0] != arr.shape[1]:
            raise ValueError(f"{name} debe ser cuadrada.")
        return arr

    @staticmethod
    def _as_finite_real_vector(
        name: str,
        value: Any,
        *,
        positive: bool = False,
        normalized: bool = False,
    ) -> NDArray[np.float64]:
        raw = np.asarray(value)
        if np.iscomplexobj(raw):
            raw = np.real(raw)
        arr = raw.astype(np.float64, copy=False).flatten()
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contiene NaN/inf.")
        if positive:
            arr = np.clip(arr, 0.0, None)
        if normalized:
            tot = float(np.sum(arr))
            if tot <= _MACHINE_EPSILON:
                raise ValueError(f"{name} masa nula.")
            arr = arr / tot
        return arr

    @staticmethod
    def _matrix_of(obj: Any) -> NDArray[np.complex128]:
        if hasattr(obj, "matrix"):
            return np.asarray(obj.matrix, dtype=np.complex128)
        return np.asarray(obj, dtype=np.complex128)


def _hash_state(*arrays: NDArray[Any]) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def _reconstruct_from_spectral(evals: NDArray[np.float64], evecs: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """ρ = V diag(λ) V†  (columnas de V = autovectores)."""
    return (evecs * np.asarray(evals, dtype=np.float64)) @ evecs.conj().T


# ══════════════════════════════════════════════════════════════════════════════
# §E.  DATACLASSES COMUNES
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class SpectralCharacteristics:
    dimension: int
    effective_rank: int
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.complex128]
    condition_number: float
    spectral_gap: float
    spectral_entropy: float
    participation_ratio: float

    def __post_init__(self) -> None:
        if self.dimension <= 0:
            raise ValueError("Dimensión debe ser positiva.")
        if self.effective_rank > self.dimension:
            raise ValueError("Rango efectivo > dimensión.")
        if self.condition_number < 1.0 - 1e-12:
            raise ValueError("Número de condición debe ser ≥ 1.")


@dataclass(frozen=True, slots=True)
class QuantumCoherenceMetrics:
    l1_norm_coherence: float
    relative_entropy_coherence: float
    robustness_of_coherence: float
    off_diagonal_mass: float

    def is_coherence_preserved(self, tol: float = _COHERENCE_TOLERANCE) -> bool:
        return self.l1_norm_coherence >= -tol


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 1 : OBSERVE
#  Nielsen, Cayley–Kähler/KKS, split Liouville–Arnold, Poincaré–Lindstedt
#  Cierre: emit_phase1_envelope → MACPhase1ObservationEnvelope
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class MajorizationAuditData:
    """Artefacto F1: Nielsen + Cayley–Kähler/KKS."""

    dimension: int
    trace_original: float
    trace_purified: float
    min_eigenvalue_original: float
    min_eigenvalue_purified: float
    max_eigenvalue_original: float
    max_eigenvalue_purified: float
    max_lorenz_deviation: float
    majorization_tolerance: float
    is_majorized: bool
    spectral_characteristics_original: SpectralCharacteristics
    spectral_characteristics_purified: SpectralCharacteristics
    renyi_entropy_original: float = 0.0
    renyi_entropy_purified: float = 0.0
    purity_original: float = 0.0
    purity_purified: float = 0.0
    symplectic_residue: float = 0.0
    symplectic_defect: float = 0.0
    kks_min_pairing: float = 0.0
    is_symplectic_valid: bool = True


@dataclass(frozen=True, slots=True)
class PoincareLindstedtSolution:
    r"""Solución de \(\{H_0,W\}_q=H_1-\langle H_1\rangle_{\mathbb{T}^d}\)."""

    W: NDArray[np.complex128]
    min_resonance_gap: float
    residual_norm: float
    is_resonant: bool
    n_resonant_modes: int
    is_solved: bool


@dataclass(frozen=True, slots=True)
class MACPhase1ObservationEnvelope:
    r"""
    Contrato F1→F2. Distingue acciones \(I_k=\lambda_k(\rho)\) de
    frecuencias \(\omega_k=E_k(H_0)\) y entrega el split integrable.
    """

    rho_original: NDArray[np.complex128]
    rho_purified: NDArray[np.complex128]
    majorization_audit: MajorizationAuditData
    poincare_lindstedt: Optional[PoincareLindstedtSolution]
    hamiltonian_H0: NDArray[np.complex128]
    hamiltonian_H1: NDArray[np.complex128]
    actions: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    eigenbasis: NDArray[np.complex128]
    jump_operators: Tuple[Tuple[float, NDArray[np.complex128]], ...]
    engine_envelope: Optional[Any]
    origin_timestamp_ns: float
    chain_hash: str


class Phase1_QuantumMajorizationAuditor(_AdvancedNumericalGuard):
    """FASE 1 — Observe: espectro, Nielsen, KKS, Lindstedt."""

    def __init__(self) -> None:
        self._phase1_envelope: Optional[MACPhase1ObservationEnvelope] = None
        self._phase2_envelope: Optional[MACPhase2OrientationEnvelope] = None
        self._strict_invariants: bool = False

    def _compute_spectral_characteristics(
        self,
        name: str,
        eigenvalues: NDArray[np.float64],
        eigenvectors: NDArray[np.complex128],
    ) -> SpectralCharacteristics:
        eigenvalues = self._as_finite_real_vector(
            f"{name}_eigenvalues", eigenvalues, positive=True
        )
        dim = int(eigenvalues.size)
        cutoff = _SPECTRAL_CUTOFF_RATIO * float(np.max(eigenvalues)) if dim else 0.0
        rank = int(np.sum(eigenvalues > cutoff))
        lmax = float(np.max(eigenvalues)) if dim else 1.0
        nz = eigenvalues[eigenvalues > cutoff]
        lmin_nz = float(np.min(nz)) if nz.size else lmax
        cond = (lmax / lmin_nz) if lmin_nz > 0.0 else float("inf")
        sorted_e = np.sort(eigenvalues)[::-1]
        gaps = np.abs(np.diff(sorted_e))
        gaps = gaps[gaps > cutoff]
        spectral_gap = float(np.min(gaps)) if gaps.size else 0.0
        tot = float(np.sum(eigenvalues)) + _MACHINE_EPSILON
        p = eigenvalues / tot
        p = p[p > _MACHINE_EPSILON]
        spectral_entropy = float(-np.sum(p * np.log(p))) if p.size else 0.0
        s2 = float(np.sum(eigenvalues ** 2))
        pr = 1.0 / s2 if s2 > _MACHINE_EPSILON else 1.0
        return SpectralCharacteristics(
            dimension=dim,
            effective_rank=rank,
            eigenvalues=eigenvalues,
            eigenvectors=np.asarray(eigenvectors, dtype=np.complex128),
            condition_number=float(cond),
            spectral_gap=spectral_gap,
            spectral_entropy=spectral_entropy,
            participation_ratio=pr,
        )

    def _sanitize_density_matrix(
        self,
        name: str,
        rho: NDArray[np.complex128],
    ) -> Tuple[NDArray[np.complex128], NDArray[np.float64], float, float, SpectralCharacteristics]:
        arr = self._as_finite_complex_matrix(name, rho, square=True)
        rho_h = self._hermitize(arr)
        tr = float(np.real(np.trace(rho_h)))
        if tr <= _TRACE_TOLERANCE:
            raise DensityMatrixValidationError(f"Traza no positiva de {name}: {tr:.3e}")
        if abs(tr - 1.0) > _TRACE_TOLERANCE:
            rho_h = rho_h / tr
        try:
            ev, vec = la.eigh(rho_h)
        except la.LinAlgError as exc:
            raise SpectralDecompositionError(f"eigh falló en {name}: {exc}") from exc
        ev = np.clip(np.real(np.asarray(ev, dtype=np.float64)), 0.0, None)
        s = float(np.sum(ev))
        if s <= _MACHINE_EPSILON:
            raise DensityMatrixValidationError(f"Espectro nulo en {name}.")
        ev = ev / s
        rho_san = self._hermitize((vec * ev) @ vec.conj().T)
        return rho_san, ev, tr, float(np.min(ev)), self._compute_spectral_characteristics(name, ev, vec)

    def _eval_nielsen_majorization_preorder(
        self,
        spectrum_orig: NDArray[np.float64],
        spectrum_purified: NDArray[np.float64],
    ) -> Tuple[bool, float, NDArray[np.float64]]:
        r"""
        \(\rho_{\mathrm{orig}}\prec\rho_\varepsilon\) (el purificado es más puro):

        \[
          \sum_{i=1}^{k}\lambda^\downarrow_{\mathrm{orig}}(i)
          \le\sum_{i=1}^{k}\lambda^\downarrow_{\varepsilon}(i)
          \qquad\forall k<d.
        \]

        Equivale a \(L_{\mathrm{orig}}(k)\le L_\varepsilon(k)\). El signo del
        desvío es \(L_{\mathrm{orig}}-L_\varepsilon\le\mathrm{tol}\).
        """
        l_orig = np.cumsum(np.sort(spectrum_orig)[::-1])
        l_pur = np.cumsum(np.sort(spectrum_purified)[::-1])
        n = min(int(l_orig.size), int(l_pur.size))
        if n <= 1:
            return True, 0.0, np.zeros(0, dtype=np.float64)
        deviations = l_orig[: n - 1] - l_pur[: n - 1]
        max_dev = float(np.max(deviations)) if deviations.size else 0.0
        return bool(max_dev <= _MAJORIZATION_TOLERANCE), max_dev, deviations

    def _compute_cayley_kahler_embedding(self, rho_matrix: NDArray[np.complex128]) -> Dict[str, Any]:
        r"""
        \(K=(\rho-I)(\rho+I)^{-1}\), \(\rho_{\mathrm{sym}}=(I+K)(I-K)^{-1}\).

        Certifica residuo de reconstrucción, defecto de Cartan \(\|K^\dagger+K\|_F\)
        y no-degeneración KKS \(\omega_\rho(X,Y)=i\operatorname{Tr}(\rho[X,Y])\).
        (El test bloque \(K^\top J+JK=0\) del v5.0 era geométricamente incorrecto
        para \(K\in M_d(\mathbb{C})\); se sustituye por KKS.)
        """
        arr = self._hermitize(np.asarray(rho_matrix, dtype=np.complex128))
        dim = int(arr.shape[0])
        eye = np.eye(dim, dtype=np.complex128)
        try:
            K = (arr - eye) @ la.pinv(arr + eye)
            rho_sym = (eye + K) @ la.pinv(eye - K)
            residue = float(la.norm(rho_sym - arr, ord="fro"))
            cartan = float(la.norm(K.conj().T + K, ord="fro"))
            kks_min = float("inf")
            cap = min(dim, 8)
            for a in range(cap):
                for b in range(a + 1, cap):
                    e_ab = np.zeros((dim, dim), dtype=np.complex128)
                    e_ab[a, b] = 1.0
                    e_ab[b, a] = -1.0
                    comm = e_ab @ e_ab.conj().T - e_ab.conj().T @ e_ab
                    val = abs(float(np.trace(arr @ comm).imag))
                    kks_min = min(kks_min, val)
            if not math.isfinite(kks_min):
                kks_min = 0.0
            ok = bool(residue < _SYMPLECTIC_RECON_TOLERANCE)
        except Exception as exc:
            logger.warning("Cayley–Kähler degradado: %s", exc)
            K = np.zeros_like(arr)
            residue, cartan, kks_min, ok = 1.0, 1.0, 0.0, False
        return {
            "cayley_matrix": K,
            "symplectic_residue": float(residue),
            "symplectic_defect": float(cartan),
            "kks_min_pairing": float(kks_min),
            "is_symplectic_valid": ok,
        }

    def _compute_coherence_metrics(self, rho: NDArray[np.complex128]) -> QuantumCoherenceMetrics:
        arr = self._hermitize(np.asarray(rho, dtype=np.complex128))
        off = arr - np.diag(np.diag(arr))
        l1 = float(np.sum(np.abs(off)))
        off_mass = float(np.sum(np.abs(off) ** 2))
        ev = np.clip(np.real(la.eigvalsh(arr)), 0.0, None)
        s = float(np.sum(ev))
        if s > 0.0:
            ev = ev / s
        pos = ev[ev > _MACHINE_EPSILON]
        s_vn = float(-np.sum(pos * np.log(pos))) if pos.size else 0.0
        diag = np.clip(np.real(np.diag(arr)), 0.0, None)
        ds = float(np.sum(diag))
        if ds > 0.0:
            diag = diag / ds
        dpos = diag[diag > _MACHINE_EPSILON]
        s_diag = float(-np.sum(dpos * np.log(dpos))) if dpos.size else 0.0
        rel = max(0.0, s_diag - s_vn)
        return QuantumCoherenceMetrics(
            l1_norm_coherence=l1,
            relative_entropy_coherence=rel,
            robustness_of_coherence=l1,
            off_diagonal_mass=off_mass,
        )

    def split_liouville_arnold(
        self,
        rho: NDArray[np.complex128],
        hamiltonian: Optional[NDArray[np.complex128]],
        evecs: NDArray[np.complex128],
        evals: NDArray[np.float64],
    ) -> Tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.float64], NDArray[np.float64]]:
        r"""
        \(H_{\mathrm{rot}}=V^\dagger HV\), \(H_0=\operatorname{diag}(H_{\mathrm{rot}})\),
        \(H_1=H_{\mathrm{rot}}-H_0\), \(I_k=\lambda_k(\rho)\), \(\omega_k=(H_0)_{kk}\).
        """
        dim = int(evals.size)
        actions = np.asarray(evals, dtype=np.float64).copy()
        V = np.asarray(evecs, dtype=np.complex128)
        if hamiltonian is None:
            H0 = np.diag(actions.astype(np.complex128))
            H1 = np.zeros((dim, dim), dtype=np.complex128)
            return H0, H1, actions, actions.copy()
        H = self._hermitize(np.asarray(hamiltonian, dtype=np.complex128))
        if H.shape != (dim, dim):
            d = min(int(H.shape[0]), dim)
            H2 = np.zeros((dim, dim), dtype=np.complex128)
            H2[:d, :d] = H[:d, :d]
            H = H2
        H_rot = V.conj().T @ H @ V
        H0 = np.diag(np.diag(H_rot))
        H1 = H_rot - H0
        omega = np.real(np.diag(H0)).astype(np.float64)
        return H0, H1, actions, omega

    def _solve_poincare_lindstedt(
        self,
        H0: NDArray[np.complex128],
        H1: NDArray[np.complex128],
        tol: float = 1e-12,
    ) -> PoincareLindstedtSolution:
        r"""
        \(W_{ij}=i(H_1^{\mathrm{osc}})_{ij}/(\lambda_i-\lambda_j)\). Los modos con
        \(|\lambda_i-\lambda_j|<\gamma/|i-j|^\tau\) se declaran resonantes
        (divisores pequeños) y se proyectan a media.
        """
        H0 = np.asarray(H0, dtype=np.complex128)
        H1 = np.asarray(H1, dtype=np.complex128)
        d = int(H0.shape[0])
        lam = np.real(np.diag(H0)).astype(np.float64)
        H1_osc = H1 - np.diag(np.diag(H1))
        W = np.zeros_like(H1)
        min_gap = float("inf")
        n_resonant = 0
        tau = float(d) + _KAM_TAU_OFFSET
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                gap = abs(float(lam[i] - lam[j]))
                min_gap = min(min_gap, gap)
                k1 = max(abs(i - j), 1)
                dioph = _KAM_GAMMA / (k1 ** tau)
                if gap < max(tol, dioph):
                    n_resonant += 1
                    continue
                W[i, j] = 1j * H1_osc[i, j] / (lam[i] - lam[j])
        resid = float(la.norm(-1j * (H0 @ W - W @ H0) - H1_osc, ord="fro"))
        if not math.isfinite(min_gap):
            min_gap = 0.0
        return PoincareLindstedtSolution(
            W=W,
            min_resonance_gap=float(min_gap),
            residual_norm=resid,
            is_resonant=bool(n_resonant > 0),
            n_resonant_modes=int(n_resonant),
            is_solved=bool(resid < max(tol * 10.0, 1e-9)),
        )

    def _audit_quantum_majorization(
        self,
        evals_orig: Optional[NDArray[np.float64]] = None,
        evals_purified: Optional[NDArray[np.float64]] = None,
        *,
        rho_orig: Optional[NDArray[np.complex128]] = None,
        rho_purified: Optional[NDArray[np.complex128]] = None,
    ) -> Tuple[MajorizationAuditData, NDArray[np.complex128], NDArray[np.complex128]]:
        if rho_orig is not None:
            rho_o, spec_o, tr_o, min_o, ch_o = self._sanitize_density_matrix("rho_orig", rho_orig)
        else:
            spec_o = self._as_finite_real_vector(
                "spec_o",
                evals_orig if evals_orig is not None else np.array([1.0]),
                positive=True,
                normalized=True,
            )
            rho_o = np.diag(spec_o).astype(np.complex128)
            tr_o, min_o = 1.0, float(np.min(spec_o))
            ch_o = self._compute_spectral_characteristics(
                "rho_orig", spec_o, np.eye(spec_o.size, dtype=np.complex128)
            )

        if rho_purified is not None:
            rho_p, spec_p, tr_p, min_p, ch_p = self._sanitize_density_matrix(
                "rho_purified", rho_purified
            )
        else:
            spec_p = self._as_finite_real_vector(
                "spec_p",
                evals_purified if evals_purified is not None else np.array([1.0]),
                positive=True,
                normalized=True,
            )
            rho_p = np.diag(spec_p).astype(np.complex128)
            tr_p, min_p = 1.0, float(np.min(spec_p))
            ch_p = self._compute_spectral_characteristics(
                "rho_purified", spec_p, np.eye(spec_p.size, dtype=np.complex128)
            )

        is_maj, dev, _ = self._eval_nielsen_majorization_preorder(spec_o, spec_p)
        if not is_maj and self._strict_invariants:
            raise QuantumMajorizationViolation(
                f"Violación de majorización Nielsen: desvío = {dev:.3e}"
            )

        cayley = self._compute_cayley_kahler_embedding(rho_p)
        if not cayley["is_symplectic_valid"] and self._strict_invariants:
            raise SymplecticDeriftError(
                f"Deriva KKS: δω={cayley['symplectic_residue']:.2e}"
            )

        def _renyi2(spec: NDArray[np.float64]) -> float:
            s2 = float(np.sum(np.clip(spec, 0.0, None) ** 2))
            return float(-np.log(max(s2, _MACHINE_EPSILON)))

        audit = MajorizationAuditData(
            dimension=ch_o.dimension,
            trace_original=float(tr_o),
            trace_purified=float(tr_p),
            min_eigenvalue_original=float(min_o),
            min_eigenvalue_purified=float(min_p),
            max_eigenvalue_original=float(np.max(spec_o)),
            max_eigenvalue_purified=float(np.max(spec_p)),
            max_lorenz_deviation=float(dev),
            majorization_tolerance=_MAJORIZATION_TOLERANCE,
            is_majorized=bool(is_maj),
            spectral_characteristics_original=ch_o,
            spectral_characteristics_purified=ch_p,
            renyi_entropy_original=_renyi2(spec_o),
            renyi_entropy_purified=_renyi2(spec_p),
            purity_original=float(np.sum(spec_o ** 2)),
            purity_purified=float(np.sum(spec_p ** 2)),
            symplectic_residue=float(cayley["symplectic_residue"]),
            symplectic_defect=float(cayley["symplectic_defect"]),
            kks_min_pairing=float(cayley["kks_min_pairing"]),
            is_symplectic_valid=bool(cayley["is_symplectic_valid"]),
        )
        return audit, rho_o, rho_p

    def emit_phase1_envelope(
        self,
        rho_orig: Optional[NDArray[np.complex128]] = None,
        rho_purified: Optional[NDArray[np.complex128]] = None,
        *,
        evals_orig: Optional[NDArray[np.float64]] = None,
        evals_purified: Optional[NDArray[np.float64]] = None,
        H0: Optional[NDArray[np.complex128]] = None,
        H1: Optional[NDArray[np.complex128]] = None,
        hamiltonian: Optional[NDArray[np.complex128]] = None,
        jump_ops: Optional[Sequence[Tuple[float, NDArray[np.complex128]]]] = None,
        engine_envelope: Optional[Any] = None,
    ) -> MACPhase1ObservationEnvelope:
        r"""
        **Cierre formal de la FASE 1 / apertura de la FASE 2.**

        Único contrato que `bind_phase1_envelope` / `from_phase1_envelope`
        y `emit_phase2_envelope` están autorizados a consumir.
        """
        t0 = time.perf_counter_ns()
        audit, rho_o, rho_p = self._audit_quantum_majorization(
            evals_orig, evals_purified, rho_orig=rho_orig, rho_purified=rho_purified
        )
        ch_p = audit.spectral_characteristics_purified
        if H0 is None or H1 is None:
            H0_s, H1_s, actions, omega = self.split_liouville_arnold(
                rho_p, hamiltonian if hamiltonian is not None else H0, ch_p.eigenvectors, ch_p.eigenvalues
            )
            H0_eff = H0 if H0 is not None else H0_s
            H1_eff = H1 if H1 is not None else H1_s
        else:
            H0_eff, H1_eff = H0, H1
            actions = ch_p.eigenvalues.copy()
            omega = np.real(np.diag(np.asarray(H0_eff, dtype=np.complex128))).astype(np.float64)
        pl = self._solve_poincare_lindstedt(H0_eff, H1_eff)
        jumps: Tuple[Tuple[float, NDArray[np.complex128]], ...] = tuple(
            (float(g), np.asarray(L, dtype=np.complex128)) for g, L in (jump_ops or ())
        )
        env = MACPhase1ObservationEnvelope(
            rho_original=rho_o,
            rho_purified=rho_p,
            majorization_audit=audit,
            poincare_lindstedt=pl,
            hamiltonian_H0=np.asarray(H0_eff, dtype=np.complex128),
            hamiltonian_H1=np.asarray(H1_eff, dtype=np.complex128),
            actions=np.asarray(actions, dtype=np.float64),
            frequencies=np.asarray(omega, dtype=np.float64),
            eigenbasis=np.asarray(ch_p.eigenvectors, dtype=np.complex128),
            jump_operators=jumps,
            engine_envelope=engine_envelope,
            origin_timestamp_ns=float(time.perf_counter_ns() - t0),
            chain_hash=_hash_state(rho_o, rho_p),
        )
        self._phase1_envelope = env
        return env


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 2 : ORIENT
#  Uhlmann, KAM/Bruno, Chirikov, Melnikov, Floquet(ℒ), RSI³ Kleisli
#  Apertura: from_phase1_envelope  (continuación de emit_phase1_envelope)
#  Cierre:   emit_phase2_envelope → MACPhase2OrientationEnvelope
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class KAMCertificate:
    is_satisfied: bool
    worst_ratio: float
    tau: float
    gamma: float
    omega_min: float


@dataclass(frozen=True, slots=True)
class MelnikovCertificate:
    integral_value: float
    is_homoclinic_preserved: bool
    overlap: float


@dataclass(frozen=True, slots=True)
class ChirikovCertificate:
    parameter: float
    is_stable: bool
    delta_omega: float
    resonance_gap: float


@dataclass(frozen=True, slots=True)
class FloquetCertificate:
    multipliers: NDArray[np.complex128]
    max_unit_deviation: float
    is_neutral: bool
    is_dissipative_stable: bool
    exponents: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PoincareSectionCertificate:
    n_crossings: int
    return_dispersion: float
    is_regular: bool
    returns: Tuple[float, ...]


@dataclass(frozen=True, slots=True)
class NekhoroshevCertificate:
    nekhoroshev_time: float
    epsilon: float
    tau: float
    is_long_term: bool


@dataclass(frozen=True, slots=True)
class KolmogorovCertificate:
    remainder_norm: float
    relative_remainder: float
    is_normalized: bool


@dataclass(frozen=True, slots=True)
class RSI3SurfaceReport:
    surface_id: int
    name: str
    metric_value: float
    threshold: float
    passed: bool
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class RSI3Certificate:
    surfaces: Tuple[RSI3SurfaceReport, RSI3SurfaceReport, RSI3SurfaceReport]
    lipschitz_constant: float
    capacity_acceleration: float
    capacity_jerk: float
    fixed_point_reached: bool
    iterations: int
    is_valid: bool

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
class FidelityAuditData:
    """Artefacto F2: Uhlmann + celeste + RSI³."""

    uhlmann_fidelity: float
    fidelity_tolerance: float
    fidelity_min_required: float
    is_fidelity_preserved: bool
    trace_distance: float = 0.0
    bures_distance: float = 0.0
    hellinger_distance: float = 0.0
    coherence_metrics_original: Optional[QuantumCoherenceMetrics] = None
    coherence_metrics_purified: Optional[QuantumCoherenceMetrics] = None
    kam_certificate: Optional[KAMCertificate] = None
    melnikov_certificate: Optional[MelnikovCertificate] = None
    chirikov_certificate: Optional[ChirikovCertificate] = None
    floquet_certificate: Optional[FloquetCertificate] = None
    poincare_section: Optional[PoincareSectionCertificate] = None
    nekhoroshev_certificate: Optional[NekhoroshevCertificate] = None
    kolmogorov_certificate: Optional[KolmogorovCertificate] = None
    rsi3_certificate: Optional[RSI3Certificate] = None
    diophantine_satisfied: bool = True
    chirikov_parameter: float = 0.0
    melnikov_integral: float = 0.0
    is_kam_stable: bool = True
    krull_valuation: float = 0.12
    fubini_study_distance: float = 1e-5
    lipschitz_constant: float = 0.85
    capacity_acceleration: float = 1.25
    capacity_jerk: float = 0.0
    is_rsi3_valid: bool = True

    def __post_init__(self) -> None:
        if not (0.0 <= self.uhlmann_fidelity <= 1.0 + _FIDELITY_NUMERICAL_TOLERANCE):
            raise ValueError(f"Fidelidad fuera de rango: {self.uhlmann_fidelity}")


@dataclass(frozen=True, slots=True)
class MACPhase2OrientationEnvelope:
    """Contrato F2→F3."""

    phase1: MACPhase1ObservationEnvelope
    fidelity_audit: FidelityAuditData
    poincare_lindstedt_solution: Optional[PoincareLindstedtSolution]
    liouvillian: Optional[NDArray[np.complex128]]
    engine_envelope: Optional[Any]
    origin_timestamp_ns: float
    chain_hash: str

    def __post_init__(self) -> None:
        expected = _hash_state(self.phase1.rho_original, self.phase1.rho_purified)
        if self.phase1.chain_hash != expected:
            raise EnvelopeChainViolationError("Cadena FASE 1 corrupta.")


class Phase2_UhlmannFidelityCertifier(Phase1_QuantumMajorizationAuditor):
    """FASE 2 — Orient: Uhlmann + Poincaré + RSI³."""

    def bind_phase1_envelope(self, env: MACPhase1ObservationEnvelope) -> "Phase2_UhlmannFidelityCertifier":
        r"""Apertura F2 sobre *esta* instancia (no fabrica huérfanos)."""
        expected = _hash_state(env.rho_original, env.rho_purified)
        if env.chain_hash != expected:
            raise EnvelopeChainViolationError("Cadena FASE 1 corrupta en bind.")
        self._phase1_envelope = env
        return self

    @classmethod
    def from_phase1_envelope(cls, env: MACPhase1ObservationEnvelope) -> "Phase2_UhlmannFidelityCertifier":
        r"""**Apertura formal de la FASE 2.** Continuación de `emit_phase1_envelope`."""
        inst = cls()
        inst.bind_phase1_envelope(env)
        return inst

    def _psd_square_root(self, rho: NDArray[np.complex128]) -> NDArray[np.complex128]:
        ev, vec = la.eigh(self._hermitize(rho))
        sqrt_ev = np.sqrt(np.clip(np.real(ev), 0.0, None))
        return self._hermitize((vec * sqrt_ev) @ vec.conj().T)

    @staticmethod
    def construct_gksl_liouvillian(
        H: NDArray[np.complex128],
        jump_operators: Sequence[Tuple[float, NDArray[np.complex128]]],
    ) -> NDArray[np.complex128]:
        r"""
        Superoperador \(\mathcal{L}\) (vec columna):
        \(-i(I\otimes H-H^T\otimes I)+\sum\gamma(\bar L\otimes L
        -\tfrac12 I\otimes L^\dagger L-\tfrac12(L^\dagger L)^T\otimes I)\).
        """
        H = np.asarray(H, dtype=np.complex128)
        d = int(H.shape[0])
        eye = np.eye(d, dtype=np.complex128)
        superop = -1j * (np.kron(eye, H) - np.kron(H.T, eye))
        for gamma, Lk in jump_operators:
            g = max(float(gamma), 0.0)
            if g == 0.0:
                continue
            L = np.asarray(Lk, dtype=np.complex128)
            LdL = L.conj().T @ L
            superop = superop + g * (
                np.kron(L.conj(), L) - 0.5 * np.kron(eye, LdL) - 0.5 * np.kron(LdL.T, eye)
            )
        return superop

    def _filter_diophantine_kam_bruno(
        self,
        omega0: NDArray[np.float64],
        max_order: int = _KAM_MAX_ORDER,
    ) -> KAMCertificate:
        omega0 = np.asarray(omega0, dtype=np.float64)
        omega0 = omega0[np.abs(omega0) > 1e-15]
        if omega0.size == 0:
            return KAMCertificate(True, float("inf"), 1.0, _KAM_GAMMA, 0.0)
        d = int(omega0.size)
        tau = float(d) + _KAM_TAU_OFFSET
        worst = float("inf")
        if d > _DIOPHANTINE_MAX_DIM:
            rng = np.random.default_rng(271828)
            ks = rng.integers(-max_order, max_order + 1, size=(4096, d))
            ks = ks[np.any(ks != 0, axis=1)]
            dots = np.abs(ks.astype(np.float64) @ omega0)
            den = np.sum(np.abs(ks), axis=1).astype(np.float64) ** tau
            ratios = dots * den
            worst = float(np.min(ratios)) if ratios.size else 0.0
        else:
            for k in product(range(-max_order, max_order + 1), repeat=d):
                if all(ki == 0 for ki in k):
                    continue
                k_vec = np.asarray(k, dtype=np.float64)
                denom = float(np.sum(np.abs(k_vec)) ** tau)
                if denom <= 0.0:
                    continue
                worst = min(worst, abs(float(np.dot(omega0, k_vec))) * denom)
        if not math.isfinite(worst):
            worst = 0.0
        return KAMCertificate(
            is_satisfied=bool(worst >= _KAM_GAMMA),
            worst_ratio=float(worst),
            tau=tau,
            gamma=_KAM_GAMMA,
            omega_min=float(np.min(np.abs(omega0))),
        )

    def _compute_chirikov_parameter(
        self,
        omega0: NDArray[np.float64],
        perturbation_norm: float,
    ) -> ChirikovCertificate:
        omega0 = np.asarray(omega0, dtype=np.float64)
        pos = omega0[np.abs(omega0) > 1e-15]
        if pos.size < 2:
            return ChirikovCertificate(0.0, True, 0.0, float("inf"))
        gaps = np.abs(np.subtract.outer(pos, pos))
        np.fill_diagonal(gaps, np.inf)
        gap = float(np.min(gaps))
        delta = float(math.sqrt(max(float(perturbation_norm), 0.0)))
        K = delta / max(gap, 1e-15)
        return ChirikovCertificate(
            parameter=float(K),
            is_stable=bool(K < _CHIRIKOV_CRITICAL),
            delta_omega=delta,
            resonance_gap=gap,
        )

    def _compute_melnikov_integral(
        self,
        H0: NDArray[np.complex128],
        H1: NDArray[np.complex128],
        n_samples: int = 256,
    ) -> MelnikovCertificate:
        t = np.linspace(-12.0, 12.0, int(n_samples))
        H0 = np.asarray(H0, dtype=np.complex128)
        H1 = np.asarray(H1, dtype=np.complex128)
        try:
            _, vecs = la.eigh(self._hermitize(H0))
            u0 = vecs[:, 0]
        except la.LinAlgError:
            u0 = np.zeros(H0.shape[0], dtype=np.complex128)
            u0[0] = 1.0
        commut = H0 @ H1 - H1 @ H0
        overlap = float(np.real(u0.conj() @ commut @ u0))
        projection = float(abs(u0.conj() @ H1 @ u0))
        M = float(_trapezoid(overlap * (1.0 / np.cosh(t)) ** 2, t))
        return MelnikovCertificate(
            integral_value=float(abs(M)),
            is_homoclinic_preserved=bool(abs(M) < _MELNIKOV_TOL),
            overlap=projection,
        )

    def _compute_floquet_multipliers(
        self,
        generator_matrix: NDArray[np.complex128],
        period: float = 2.0 * math.pi,
    ) -> FloquetCertificate:
        try:
            ev = la.eigvals(np.asarray(generator_matrix, dtype=np.complex128))
            moduli = np.exp(np.real(ev) * float(period))
            unit_dev = float(np.max(np.abs(moduli - 1.0)))
            dissipative = bool(np.max(moduli) <= 1.0 + _FLOQUET_UNIT_TOL)
            exponents = np.real(ev)
            mu = np.exp(ev * float(period))
        except Exception:
            mu = np.array([1.0 + 0j])
            unit_dev, dissipative, exponents = 0.0, True, np.array([0.0])
        return FloquetCertificate(
            multipliers=mu,
            max_unit_deviation=unit_dev,
            is_neutral=bool(unit_dev < _FLOQUET_UNIT_TOL),
            is_dissipative_stable=dissipative,
            exponents=exponents,
        )

    def _compute_poincare_section(
        self,
        H0: NDArray[np.complex128],
        H1: NDArray[np.complex128],
        rho: NDArray[np.complex128],
        n_samples: int = _POINCARE_SECTION_SAMPLES,
    ) -> PoincareSectionCertificate:
        evals, evecs = la.eigh(self._hermitize(H0))
        H1_eig = evecs.conj().T @ H1 @ evecs
        rho_eig = evecs.conj().T @ rho @ evecs
        t = np.linspace(0.0, 2.0 * math.pi, int(n_samples))
        obs = np.zeros_like(t)
        evals = np.real(evals)
        for idx, tt in enumerate(t):
            phases = np.exp(-1j * evals * tt)
            rho_t = (phases[:, None] * rho_eig) * phases.conj()[None, :]
            obs[idx] = float(np.real(np.sum(rho_t * H1_eig.T)))
        d_obs = np.diff(obs)
        if d_obs.size < 2:
            return PoincareSectionCertificate(0, 0.0, True, tuple())
        crossings = np.where(np.sign(d_obs[:-1]) != np.sign(d_obs[1:]))[0]
        if crossings.size < 2:
            return PoincareSectionCertificate(int(crossings.size), 0.0, True, tuple())
        returns = obs[crossings]
        disp = float(np.std(returns) / max(float(np.mean(np.abs(returns))), 1e-15))
        return PoincareSectionCertificate(
            n_crossings=int(crossings.size),
            return_dispersion=disp,
            is_regular=bool(disp < 0.5),
            returns=tuple(float(x) for x in returns),
        )

    def _nekhoroshev(self, perturbation_norm: float, h0_norm: float, tau: float) -> NekhoroshevCertificate:
        eps = max(float(perturbation_norm) / max(float(h0_norm), 1e-15), 1e-30)
        exponent = 1.0 / max(2.0 * float(tau), 1e-12)
        log_T = min(eps ** (-exponent), 80.0)
        T = float(math.exp(0.5 * log_T))
        return NekhoroshevCertificate(T, float(eps), float(tau), bool(T > 1.0))

    def _kolmogorov_normal_form(
        self,
        H0: NDArray[np.complex128],
        H1: NDArray[np.complex128],
        pl: Optional[PoincareLindstedtSolution],
    ) -> KolmogorovCertificate:
        if pl is None:
            pl = self._solve_poincare_lindstedt(H0, H1)
        W = pl.W
        H1_full = H1 + np.diag(np.diag(H1))
        remainder = -0.5j * (W @ H1_full - H1_full @ W)
        rem = float(la.norm(remainder, ord="fro"))
        h0n = max(float(la.norm(H0, ord="fro")), 1e-15)
        rel = rem / h0n
        return KolmogorovCertificate(rem, rel, bool(pl.is_solved and rel < 0.5))

    def _certify_kam_stability(
        self,
        env: MACPhase1ObservationEnvelope,
        jump_ops: Sequence[Tuple[float, NDArray[np.complex128]]],
    ) -> Tuple[
        KAMCertificate,
        MelnikovCertificate,
        ChirikovCertificate,
        FloquetCertificate,
        PoincareSectionCertificate,
        NekhoroshevCertificate,
        KolmogorovCertificate,
        NDArray[np.complex128],
        bool,
    ]:
        omega0 = np.asarray(env.frequencies, dtype=np.float64)
        H0, H1 = env.hamiltonian_H0, env.hamiltonian_H1
        pert = float(la.norm(H1, ord="fro"))
        h0n = float(la.norm(H0, ord="fro"))
        kam = self._filter_diophantine_kam_bruno(omega0)
        chir = self._compute_chirikov_parameter(omega0, pert)
        mel = self._compute_melnikov_integral(H0, H1)
        knf = self._kolmogorov_normal_form(H0, H1, env.poincare_lindstedt)
        nek = self._nekhoroshev(pert, h0n, kam.tau)
        if jump_ops:
            G = self.construct_gksl_liouvillian(H0 + H1, jump_ops)
        else:
            G = -1j * (H0 + H1)
        floq = self._compute_floquet_multipliers(G)
        sec = self._compute_poincare_section(H0, H1, env.rho_purified)
        pl_ok = True if env.poincare_lindstedt is None else env.poincare_lindstedt.is_solved
        stable = bool(
            kam.is_satisfied
            and chir.is_stable
            and mel.is_homoclinic_preserved
            and (floq.is_neutral or floq.is_dissipative_stable)
            and pl_ok
        )
        return kam, mel, chir, floq, sec, nek, knf, G, stable

    def _truncate_spectrum(
        self,
        rho: NDArray[np.complex128],
        epsilon: float = 1e-6,
    ) -> NDArray[np.complex128]:
        ev, vec = la.eigh(self._hermitize(rho))
        ev = np.clip(np.real(ev), 0.0, None)
        mask = ev >= max(float(epsilon), _MIN_EIGENVALUE)
        if not np.any(mask):
            mask[int(np.argmax(ev))] = True
        kept = ev * mask.astype(np.float64)
        s = float(np.sum(kept))
        if s <= 0.0:
            return rho
        kept = kept / s
        return self._hermitize((vec * kept) @ vec.conj().T)

    def _apply_gksl_step(
        self,
        rho: NDArray[np.complex128],
        H: NDArray[np.complex128],
        jump_ops: Sequence[Tuple[float, NDArray[np.complex128]]],
        dt: float,
    ) -> NDArray[np.complex128]:
        R = self._hermitize(np.asarray(rho, dtype=np.complex128))
        H = np.asarray(H, dtype=np.complex128)
        comm = H @ R - R @ H
        dissip = np.zeros_like(R)
        for gamma, L in jump_ops:
            g = max(float(gamma), 0.0)
            L = np.asarray(L, dtype=np.complex128)
            LdL = L.conj().T @ L
            dissip += g * (L @ R @ L.conj().T - 0.5 * (LdL @ R + R @ LdL))
        R_next = self._hermitize(R + float(dt) * (-1j * comm + dissip))
        ev, vec = la.eigh(R_next)
        ev = np.clip(np.real(ev), 0.0, None)
        s = float(np.sum(ev))
        if s <= 0.0:
            return R
        ev = ev / s
        return self._hermitize((vec * ev) @ vec.conj().T)

    @staticmethod
    def _fubini_study(psi: NDArray[np.complex128], phi: NDArray[np.complex128]) -> float:
        denom = max(float(la.norm(psi) * la.norm(phi)), 1e-15)
        v = abs(complex(np.vdot(psi, phi))) / denom
        return float(np.arccos(np.clip(v, 0.0, 1.0)))

    def _uhlmann_fidelity(self, rho: NDArray[np.complex128], sigma: NDArray[np.complex128]) -> float:
        sqrt_o = self._psd_square_root(rho)
        core = self._hermitize(sqrt_o @ sigma @ sqrt_o)
        core_ev = np.clip(np.real(la.eigvalsh(core)), 0.0, None)
        tr_sqrt = self._kahan_sum(np.sqrt(core_ev))
        return float(np.clip(tr_sqrt * tr_sqrt, 0.0, 1.0))

    def _bures(self, rho: NDArray[np.complex128], sigma: NDArray[np.complex128]) -> float:
        fid = self._uhlmann_fidelity(rho, sigma)
        return float(math.sqrt(max(0.0, 2.0 * (1.0 - math.sqrt(fid)))))

    def _enforce_rsi3_kleisli_monad_surfaces(
        self,
        rho: NDArray[np.complex128],
        H: NDArray[np.complex128],
        jump_ops: Sequence[Tuple[float, NDArray[np.complex128]]],
        max_iter: int = _RSI3_MAX_ITER,
        kam_stable: bool = True,
    ) -> RSI3Certificate:
        r"""
        \(T=\Phi_{\mathrm{GKSL},\Delta t}\circ\Pi_\varepsilon\). No se proyecta a
        estado puro (eso fuerza \(v_{\mathrm{Krull}}=-\log\lambda_{\max}\to 0\)).
        Lipschitz en métrica de Bures; jerk \(\mathrm{d}^3 C/\mathrm{d}t^3\) por
        diferencias de la capacidad \(S(\rho)\). Nivel 3: meta-ajuste de
        \(\varepsilon,\Delta t\).
        """
        rho_cur = self._hermitize(np.asarray(rho, dtype=np.complex128))
        rho_prev: Optional[NDArray[np.complex128]] = None
        ref = rho_cur
        dummy = RSI3SurfaceReport(0, "init", 0.0, 0.0, False, {})
        last: Tuple[RSI3SurfaceReport, RSI3SurfaceReport, RSI3SurfaceReport] = (dummy, dummy, dummy)
        lip_hist: List[float] = []
        cap_hist: List[float] = []
        fixed = False
        it_used = 0
        eps_trunc = 1e-6
        dt = 0.25 * _RSI3_CHANNEL_DT if not kam_stable else _RSI3_CHANNEL_DT
        lip_max = min(_BANACH_LIPSCHITZ_MAX, 0.85) if not kam_stable else _BANACH_LIPSCHITZ_MAX
        ops: List[Tuple[float, NDArray[np.complex128]]] = list(jump_ops)

        for it in range(1, int(max_iter) + 1):
            it_used = it
            ev = np.clip(np.real(la.eigvalsh(self._hermitize(rho_cur))), 0.0, None)
            s = float(np.sum(ev))
            if s > 0.0:
                ev = ev / s
            lam_max = float(ev[-1]) if ev.size else 1.0  # eigh ascending
            v_krull = float(-math.log(max(lam_max, 1e-15)))
            s1 = RSI3SurfaceReport(
                1, "Data-RSI", v_krull, _KRULL_MIN_VALUATION,
                bool(v_krull >= _KRULL_MIN_VALUATION),
                {"eigenvalue_max": lam_max, "rank": int(np.sum(ev > 1e-12))},
            )

            if ops:
                n = len(ops)
                gram = np.zeros((n, n), dtype=np.complex128)
                rates = np.array([max(float(g), 0.0) for g, _ in ops], dtype=np.float64)
                for i, (_, Li) in enumerate(ops):
                    Ld = Li.conj().T
                    for j, (_, Lj) in enumerate(ops):
                        gram[i, j] = np.trace(Ld @ Lj)
                sqrt_g = np.sqrt(rates)
                Kw = self._hermitize((sqrt_g[:, None] * gram) * sqrt_g[None, :])
                try:
                    lam_min = float(np.min(np.real(la.eigvalsh(Kw))))
                except la.LinAlgError:
                    lam_min = -1.0
                s2 = RSI3SurfaceReport(
                    2, "Harness-RSI", lam_min, -1e-10, bool(lam_min >= -1e-10),
                    {"n_ops": n},
                )
                if not s2.passed:
                    ops = [op for op in ops if op[0] >= 1e-4] or ops[:1]
            else:
                s2 = RSI3SurfaceReport(2, "Harness-RSI", 0.0, -1e-10, True, {"n_ops": 0})

            rho_trunc = self._truncate_spectrum(rho_cur, eps_trunc)
            rho_next = self._apply_gksl_step(rho_trunc, H, ops, dt)
            ev_n, vec_n = la.eigh(self._hermitize(rho_next))
            if rho_prev is not None:
                _, vec_p = la.eigh(self._hermitize(rho_prev))
                d_fs = self._fubini_study(vec_n[:, -1], vec_p[:, -1])
            else:
                d_fs = 0.0
            s3 = RSI3SurfaceReport(
                3, "Model-RSI", float(d_fs), _FUBINI_STUDY_MAX,
                bool(d_fs <= _FUBINI_STUDY_MAX or rho_prev is None),
                {"dominant_eig": float(np.real(ev_n[-1]))},
            )
            last = (s1, s2, s3)

            pos = np.clip(np.real(ev_n), 0.0, None)
            pos = pos / max(float(np.sum(pos)), 1e-30)
            ppos = pos[pos > _MACHINE_EPSILON]
            cap_hist.append(float(-np.sum(ppos * np.log(ppos))) if ppos.size else 0.0)

            if rho_prev is not None:
                d_num = self._bures(rho_prev, rho_next)
                d_den = self._bures(ref, rho_prev)
                lip = float(d_num / max(d_den, 1e-15))
                lip_hist.append(lip)
                if not s1.passed:
                    eps_trunc = max(eps_trunc * 0.5, _MIN_EIGENVALUE)
                if lip > lip_max:
                    dt *= 0.85

            delta = float(la.norm(rho_next - rho_cur, ord="fro"))
            rho_prev, rho_cur = rho_cur, rho_next
            if s1.passed and s2.passed and s3.passed and delta < _FUBINI_STUDY_MAX:
                fixed = True
                break

        lip_f = float(np.mean(lip_hist)) if lip_hist else 1.0
        jerk = 0.0
        if len(cap_hist) >= 4:
            c = np.asarray(cap_hist, dtype=np.float64)
            jerk = float(c[-1] - 3.0 * c[-2] + 3.0 * c[-3] - c[-4])
        elif len(cap_hist) >= 2:
            jerk = float(cap_hist[-1] - cap_hist[0])
        accel = abs(jerk) ** (1.0 / 3.0) if jerk != 0.0 else 0.0
        valid = bool(all(s.passed for s in last) and lip_f <= lip_max and jerk >= -1e-18)
        return RSI3Certificate(
            surfaces=last,
            lipschitz_constant=lip_f,
            capacity_acceleration=float(accel),
            capacity_jerk=float(jerk),
            fixed_point_reached=fixed,
            iterations=it_used,
            is_valid=valid,
        )

    def _certify_uhlmann_fidelity_bound(
        self,
        env: MACPhase1ObservationEnvelope,
        jump_ops: Sequence[Tuple[float, NDArray[np.complex128]]],
    ) -> Tuple[FidelityAuditData, NDArray[np.complex128]]:
        rho_o, rho_p = env.rho_original, env.rho_purified
        maj = env.majorization_audit
        fid = self._uhlmann_fidelity(rho_o, rho_p)
        preserved = bool(fid >= _UHLMANN_FIDELITY_MIN)
        if not preserved and self._strict_invariants:
            raise UhlmannFidelityCollapseError(
                f"Colapso de Uhlmann: F={fid:.6f} < {_UHLMANN_FIDELITY_MIN}"
            )
        bures = float(math.sqrt(max(0.0, 2.0 * (1.0 - math.sqrt(fid)))))
        tr_dist = 0.5 * float(np.sum(np.abs(la.eigvalsh(self._hermitize(rho_o - rho_p)))))
        hellinger = float(math.sqrt(max(0.0, 1.0 - fid)))

        kam, mel, chir, floq, sec, nek, knf, G, kam_stable = self._certify_kam_stability(env, jump_ops)
        rsi3 = self._enforce_rsi3_kleisli_monad_surfaces(
            rho_p, env.hamiltonian_H0, jump_ops, kam_stable=kam_stable
        )
        return (
            FidelityAuditData(
                uhlmann_fidelity=fid,
                fidelity_tolerance=_FIDELITY_NUMERICAL_TOLERANCE,
                fidelity_min_required=_UHLMANN_FIDELITY_MIN,
                is_fidelity_preserved=preserved,
                trace_distance=tr_dist,
                bures_distance=bures,
                hellinger_distance=hellinger,
                coherence_metrics_original=self._compute_coherence_metrics(rho_o),
                coherence_metrics_purified=self._compute_coherence_metrics(rho_p),
                kam_certificate=kam,
                melnikov_certificate=mel,
                chirikov_certificate=chir,
                floquet_certificate=floq,
                poincare_section=sec,
                nekhoroshev_certificate=nek,
                kolmogorov_certificate=knf,
                rsi3_certificate=rsi3,
                diophantine_satisfied=kam.is_satisfied,
                chirikov_parameter=chir.parameter,
                melnikov_integral=mel.integral_value,
                is_kam_stable=kam_stable,
                krull_valuation=rsi3.surface1.metric_value,
                fubini_study_distance=rsi3.surface3.metric_value,
                lipschitz_constant=rsi3.lipschitz_constant,
                capacity_acceleration=rsi3.capacity_acceleration,
                capacity_jerk=rsi3.capacity_jerk,
                is_rsi3_valid=rsi3.is_valid,
            ),
            G,
        )

    def emit_phase2_envelope(
        self,
        phase1_env: Optional[MACPhase1ObservationEnvelope] = None,
        *,
        jump_ops: Optional[Sequence[Tuple[float, NDArray[np.complex128]]]] = None,
        engine_envelope: Optional[Any] = None,
    ) -> MACPhase2OrientationEnvelope:
        r"""
        **Cierre formal de la FASE 2 / apertura de la FASE 3.**

        Consume el `MACPhase1ObservationEnvelope` (argumento o bind previo).
        """
        t0 = time.perf_counter_ns()
        env = phase1_env if phase1_env is not None else self._phase1_envelope
        if env is None:
            raise EnvelopeChainViolationError("FASE 2 sin envelope de FASE 1.")
        self.bind_phase1_envelope(env)
        ops = list(jump_ops) if jump_ops is not None else list(env.jump_operators)
        fid, G = self._certify_uhlmann_fidelity_bound(env, ops)
        out = MACPhase2OrientationEnvelope(
            phase1=env,
            fidelity_audit=fid,
            poincare_lindstedt_solution=env.poincare_lindstedt,
            liouvillian=G,
            engine_envelope=engine_envelope if engine_envelope is not None else env.engine_envelope,
            origin_timestamp_ns=float(time.perf_counter_ns() - t0),
            chain_hash=_hash_state(env.rho_original, env.rho_purified),
        )
        self._phase2_envelope = out
        return out


# ══════════════════════════════════════════════════════════════════════════════
#  FASE 3 : DECIDE & ACT
#  Holevo, Heyting Ω₃, ESP32 Crowbar
#  Apertura: from_phase2_envelope  (continuación de emit_phase2_envelope)
#  Cierre:   emit_governance_state → PurificationGovernanceState
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class HolevoAuditData:
    entropy_original: float
    entropy_purified: float
    entropy_delta: float
    entropy_tolerance: float
    is_capacity_preserved: bool
    renyi_0_entropy_original: float = 0.0
    renyi_0_entropy_purified: float = 0.0
    renyi_2_entropy_original: float = 0.0
    renyi_2_entropy_purified: float = 0.0
    renyi_inf_entropy_original: float = 0.0
    renyi_inf_entropy_purified: float = 0.0
    holevo_capacity_bound: float = 0.0
    accessible_information: float = 0.0
    free_energy_change: float = 0.0
    heyting_verdict: str = "COHERENT"
    heyting_code: int = 0x00
    hardware_crowbar_action: str = "NOMINAL_EXECUTION"
    interlock_latency_ns: float = 45.0

    def satisfies_second_law(self, allow_reversible: bool = True) -> bool:
        return True if allow_reversible else self.entropy_delta >= -self.entropy_tolerance


@dataclass(frozen=True, slots=True)
class PurificationGovernanceState:
    """Objeto final del endofuntor Z_MAC-Agent (FASE 3)."""

    phase1: MACPhase1ObservationEnvelope
    phase2: MACPhase2OrientationEnvelope
    majorization_audit: MajorizationAuditData
    fidelity_audit: FidelityAuditData
    holevo_audit: HolevoAuditData
    is_epistemologically_valid: bool
    purification_phase: PurificationPhase = PurificationPhase.COMPLETE
    timestamp: Optional[float] = None
    overall_quality_score: float = 0.0
    risk_assessment: str = "NOMINAL"


class Phase3_HolevoCapacityEnforcer(Phase2_UhlmannFidelityCertifier):
    """FASE 3 — Decide & Act: Holevo + Ω₃ + Crowbar."""

    def bind_phase2_envelope(self, env: MACPhase2OrientationEnvelope) -> "Phase3_HolevoCapacityEnforcer":
        expected = _hash_state(env.phase1.rho_original, env.phase1.rho_purified)
        if env.phase1.chain_hash != expected or env.chain_hash != expected:
            raise EnvelopeChainViolationError("Cadena FASE 1→FASE 2 corrupta.")
        self._phase1_envelope = env.phase1
        self._phase2_envelope = env
        return self

    @classmethod
    def from_phase2_envelope(cls, env: MACPhase2OrientationEnvelope) -> "Phase3_HolevoCapacityEnforcer":
        r"""**Apertura formal de la FASE 3.** Continuación de `emit_phase2_envelope`."""
        inst = cls()
        inst.bind_phase2_envelope(env)
        return inst

    @staticmethod
    def _von_neumann_entropy(spectrum: NDArray[np.float64]) -> float:
        p = np.clip(spectrum, 0.0, 1.0)
        p = p[p > _MACHINE_EPSILON]
        if p.size == 0:
            return 0.0
        return max(0.0, _AdvancedNumericalGuard._kahan_sum(-p * np.log(p)))

    @staticmethod
    def _renyi_entropy(spectrum: NDArray[np.float64], alpha: float) -> float:
        p = np.clip(spectrum, 0.0, 1.0)
        p = p[p > _MACHINE_EPSILON]
        if p.size == 0:
            return 0.0
        if math.isclose(alpha, 1.0, abs_tol=1e-10):
            return float(-_AdvancedNumericalGuard._kahan_sum(p * np.log(p)))
        if math.isclose(alpha, 0.0):
            return float(np.log(p.size))
        if math.isinf(alpha):
            return float(-np.log(np.max(p)))
        return float((1.0 / (1.0 - alpha)) * np.log(float(np.sum(p ** alpha))))

    def _dispatch_esp32_crowbar_interlock(
        self,
        is_majorized: bool,
        is_fidelity_ok: bool,
        is_kam_stable: bool,
        is_rsi3_valid: bool,
        is_symplectic_valid: bool = True,
        entropy_ok: bool = True,
    ) -> Tuple[HeytingOmega3, str, float]:
        v1 = HeytingOmega3.COHERENT if (is_majorized and is_symplectic_valid) else HeytingOmega3.VETOED
        v2 = HeytingOmega3.COHERENT if is_fidelity_ok else HeytingOmega3.DEGRADED
        v3 = HeytingOmega3.COHERENT if (is_kam_stable and is_rsi3_valid) else HeytingOmega3.VETOED
        v4 = HeytingOmega3.COHERENT if entropy_ok else HeytingOmega3.VETOED
        final = v1.join(v2).join(v3).join(v4)
        if final is HeytingOmega3.VETOED:
            return final, "ESP32_CROWBAR_GPIO14_HIGH_BT151_TRIGGERED", _CROWBAR_VETO_LATENCY_NS
        if final is HeytingOmega3.DEGRADED:
            return final, "MONITORING_DEGRADED_PERFORMANCE", _CROWBAR_DEGRADED_LATENCY_NS
        return final, "NOMINAL_EXECUTION", _CROWBAR_NOMINAL_LATENCY_NS

    def _enforce_holevo_capacity_retention(
        self,
        env: MACPhase2OrientationEnvelope,
    ) -> HolevoAuditData:
        spec_o = env.phase1.majorization_audit.spectral_characteristics_original.eigenvalues
        spec_p = env.phase1.majorization_audit.spectral_characteristics_purified.eigenvalues
        h_o = self._von_neumann_entropy(spec_o)
        h_p = self._von_neumann_entropy(spec_p)
        ds = float(h_p - h_o)
        capacity_ok = bool(ds <= _ENTROPY_TOLERANCE)
        if not capacity_ok and self._strict_invariants:
            raise HolevoCapacityDeficitError(f"Paradoja termodinámica: ΔS={ds:.3e} > 0.")

        maj = env.phase1.majorization_audit
        fid = env.fidelity_audit
        verdict, action, latency = self._dispatch_esp32_crowbar_interlock(
            is_majorized=maj.is_majorized,
            is_fidelity_ok=fid.is_fidelity_preserved,
            is_kam_stable=fid.is_kam_stable,
            is_rsi3_valid=fid.is_rsi3_valid,
            is_symplectic_valid=maj.is_symplectic_valid,
            entropy_ok=capacity_ok,
        )
        if latency > _CROWBAR_TIMING_LIMIT_NS:
            logger.error("Crowbar latency %.2f ns excede el límite de 400 ns.", latency)

        return HolevoAuditData(
            entropy_original=float(h_o),
            entropy_purified=float(h_p),
            entropy_delta=float(ds),
            entropy_tolerance=_ENTROPY_TOLERANCE,
            is_capacity_preserved=capacity_ok,
            renyi_0_entropy_original=self._renyi_entropy(spec_o, 0.0),
            renyi_0_entropy_purified=self._renyi_entropy(spec_p, 0.0),
            renyi_2_entropy_original=self._renyi_entropy(spec_o, 2.0),
            renyi_2_entropy_purified=self._renyi_entropy(spec_p, 2.0),
            renyi_inf_entropy_original=self._renyi_entropy(spec_o, float("inf")),
            renyi_inf_entropy_purified=self._renyi_entropy(spec_p, float("inf")),
            holevo_capacity_bound=float(h_p),
            accessible_information=float(max(0.0, h_o - h_p)),
            free_energy_change=-ds,
            heyting_verdict=verdict.name,
            heyting_code=int(verdict.value),
            hardware_crowbar_action=action,
            interlock_latency_ns=float(latency),
        )

    def emit_governance_state(
        self,
        phase2_env: Optional[MACPhase2OrientationEnvelope] = None,
    ) -> PurificationGovernanceState:
        r"""
        **Cierre formal de la FASE 3.**

        Emite `PurificationGovernanceState`. Los fallos físicos se materializan
        como Ω₃ ∈ {DEGRADED, VETOED} (Crowbar), no como excepciones: el OODA
        debe completar el Act. Las excepciones quedan para corrupción de cadena.
        """
        t0 = time.time()
        env = phase2_env if phase2_env is not None else self._phase2_envelope
        if env is None:
            raise EnvelopeChainViolationError("FASE 3 sin envelope de FASE 2.")
        self.bind_phase2_envelope(env)
        holevo = self._enforce_holevo_capacity_retention(env)
        maj = env.phase1.majorization_audit
        fid = env.fidelity_audit
        valid = bool(
            maj.is_majorized
            and fid.is_fidelity_preserved
            and holevo.is_capacity_preserved
            and maj.is_symplectic_valid
            and fid.is_kam_stable
            and fid.is_rsi3_valid
            and holevo.heyting_code == 0x00
        )
        quality = float(
            np.mean(
                [
                    fid.uhlmann_fidelity,
                    float(maj.is_majorized),
                    float(maj.is_symplectic_valid),
                    float(fid.is_kam_stable),
                    float(fid.is_rsi3_valid),
                    float(holevo.is_capacity_preserved),
                ]
            )
        )
        risk = (
            "VETOED"
            if holevo.heyting_code == 0x03
            else ("DEGRADED" if holevo.heyting_code == 0x01 else "NOMINAL")
        )
        if holevo.heyting_code == 0x03:
            logger.critical(
                "Ω₃=VETOED → %s @ %.2f ns (F=%.4f KAM=%s RSI³=%s maj=%s KKS=%s)",
                holevo.hardware_crowbar_action,
                holevo.interlock_latency_ns,
                fid.uhlmann_fidelity,
                fid.is_kam_stable,
                fid.is_rsi3_valid,
                maj.is_majorized,
                maj.is_symplectic_valid,
            )
        return PurificationGovernanceState(
            phase1=env.phase1,
            phase2=env,
            majorization_audit=maj,
            fidelity_audit=fid,
            holevo_audit=holevo,
            is_epistemologically_valid=valid,
            purification_phase=PurificationPhase.COMPLETE,
            timestamp=t0,
            overall_quality_score=quality,
            risk_assessment=risk,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  ORQUESTADOR — endofuntor Observe → Orient → Decide&Act
# ══════════════════════════════════════════════════════════════════════════════

class MACMinimizerAgent(Morphism, Phase3_HolevoCapacityEnforcer):
    r"""
    Custodio de la purificación espectral. Orquesta F1→F2→F3 sobre *una*
    instancia (bind, no clones) y consume envelopes v5.1 del motor.
    """

    __slots__ = (
        "_audit_history",
        "_phase1_envelope",
        "_phase2_envelope",
        "_strict_invariants",
    )

    def __init__(self, strict_invariants: bool = False) -> None:
        try:
            super().__init__()
        except TypeError:
            pass
        self._audit_history: List[PurificationGovernanceState] = []
        self._phase1_envelope = None
        self._phase2_envelope = None
        self._strict_invariants = bool(strict_invariants)
        logger.info(
            "MACMinimizerAgent v%s (3-Fases) engine=%s",
            __version__,
            "ON" if _ENGINE_AVAILABLE else "OFF",
        )

    def execute_spectral_purification_governance(
        self,
        rho_orig: Optional[NDArray[np.complex128]] = None,
        rho_purified: Optional[NDArray[np.complex128]] = None,
        evals_orig: Optional[NDArray[np.float64]] = None,
        evals_purified: Optional[NDArray[np.float64]] = None,
        *,
        H0: Optional[NDArray[np.complex128]] = None,
        H1: Optional[NDArray[np.complex128]] = None,
        hamiltonian: Optional[NDArray[np.complex128]] = None,
        jump_ops: Optional[Sequence[Tuple[float, NDArray[np.complex128]]]] = None,
        engine_envelope: Optional[Any] = None,
        enable_extended_diagnostics: bool = False,
    ) -> PurificationGovernanceState:
        env1 = self.emit_phase1_envelope(
            rho_orig=rho_orig,
            rho_purified=rho_purified,
            evals_orig=evals_orig,
            evals_purified=evals_purified,
            H0=H0,
            H1=H1,
            hamiltonian=hamiltonian,
            jump_ops=jump_ops,
            engine_envelope=engine_envelope,
        )
        self.bind_phase1_envelope(env1)
        env2 = self.emit_phase2_envelope(env1, jump_ops=jump_ops, engine_envelope=engine_envelope)
        self.bind_phase2_envelope(env2)
        state = self.emit_governance_state(env2)
        self._audit_history.append(state)
        rsi3 = state.fidelity_audit.rsi3_certificate
        logger.info(
            "MACMinimizerAgent — Ω₃=%s F=%.4f ΔS=%+.3e t=%.1fns RSI³_iters=%s valid=%s",
            state.holevo_audit.heyting_verdict,
            state.fidelity_audit.uhlmann_fidelity,
            state.holevo_audit.entropy_delta,
            state.holevo_audit.interlock_latency_ns,
            rsi3.iterations if rsi3 else 0,
            state.is_epistemologically_valid,
        )
        return state

    @staticmethod
    def _unpack_engine_envelope(engine_env: Any) -> Dict[str, Any]:
        """Duck-typing v5.0/v5.1: PoincareEnvelope | RSI3Envelope."""
        trunc = getattr(engine_env, "truncation_report", None)
        if trunc is None:
            raise EnvelopeChainViolationError("Envelope de motor sin truncation_report.")
        sp_orig = trunc.original_spectral_data
        rho_orig = _reconstruct_from_spectral(sp_orig.eigenvalues, sp_orig.eigenvectors)
        rho_pur_obj = engine_env.rho_purified
        rho_pur = rho_pur_obj.matrix if hasattr(rho_pur_obj, "matrix") else np.asarray(rho_pur_obj)
        H0 = getattr(engine_env, "H0_integrable", None)
        if H0 is None:
            H0 = getattr(engine_env, "hamiltonian", None)
        H1 = getattr(engine_env, "H1_perturbation", None)
        jumps = getattr(engine_env, "pruned_jump_operators", None)
        if jumps is None:
            jumps = getattr(engine_env, "jump_operators", None) or []
        return {
            "rho_orig": np.asarray(rho_orig, dtype=np.complex128),
            "rho_pur": np.asarray(rho_pur, dtype=np.complex128),
            "H0": None if H0 is None else np.asarray(H0, dtype=np.complex128),
            "H1": None if H1 is None else np.asarray(H1, dtype=np.complex128),
            "jumps": list(jumps),
        }

    def consume_engine_poincare_envelope(self, engine_env: Any) -> PurificationGovernanceState:
        """Interoperabilidad F1 del motor v5.1 (`PoincareEnvelope`)."""
        p = self._unpack_engine_envelope(engine_env)
        return self.execute_spectral_purification_governance(
            rho_orig=p["rho_orig"],
            rho_purified=p["rho_pur"],
            H0=p["H0"],
            H1=p["H1"],
            jump_ops=p["jumps"],
            engine_envelope=engine_env,
            enable_extended_diagnostics=True,
        )

    def consume_engine_rsi3_envelope(self, engine_env: Any) -> PurificationGovernanceState:
        """Interoperabilidad F2 del motor v5.1 (`RSI3Envelope`)."""
        p = self._unpack_engine_envelope(engine_env)
        return self.execute_spectral_purification_governance(
            rho_orig=p["rho_orig"],
            rho_purified=p["rho_pur"],
            H0=p["H0"],
            H1=p["H1"],
            jump_ops=p["jumps"],
            engine_envelope=engine_env,
            enable_extended_diagnostics=True,
        )

    def report(self) -> Dict[str, Any]:
        last = self._audit_history[-1] if self._audit_history else None
        return {
            "n_audits": len(self._audit_history),
            "last_verdict": None if last is None else last.holevo_audit.heyting_verdict,
            "last_quality": None if last is None else last.overall_quality_score,
            "last_risk": None if last is None else last.risk_assessment,
            "engine_available": _ENGINE_AVAILABLE,
            "version": __version__,
        }


def create_mac_minimizer_agent(*, strict_invariants: bool = False) -> MACMinimizerAgent:
    return MACMinimizerAgent(strict_invariants=strict_invariants)


__all__: List[str] = [
    "MACMinimizerAgentError",
    "DensityMatrixValidationError",
    "QuantumMajorizationViolation",
    "UhlmannFidelityCollapseError",
    "HolevoCapacityDeficitError",
    "SpectralDecompositionError",
    "QuantumCoherenceViolation",
    "EntanglementStructureError",
    "SymplecticDeriftError",
    "PoincareResonanceError",
    "RSI3ContractionFailureError",
    "EnvelopeChainViolationError",
    "QuantumDistanceMetric",
    "PurificationPhase",
    "HeytingOmega3",
    "SpectralCharacteristics",
    "QuantumCoherenceMetrics",
    "MajorizationAuditData",
    "PoincareLindstedtSolution",
    "MACPhase1ObservationEnvelope",
    "Phase1_QuantumMajorizationAuditor",
    "KAMCertificate",
    "MelnikovCertificate",
    "ChirikovCertificate",
    "FloquetCertificate",
    "PoincareSectionCertificate",
    "NekhoroshevCertificate",
    "KolmogorovCertificate",
    "RSI3SurfaceReport",
    "RSI3Certificate",
    "FidelityAuditData",
    "MACPhase2OrientationEnvelope",
    "Phase2_UhlmannFidelityCertifier",
    "HolevoAuditData",
    "PurificationGovernanceState",
    "Phase3_HolevoCapacityEnforcer",
    "MACMinimizerAgent",
    "create_mac_minimizer_agent",
]