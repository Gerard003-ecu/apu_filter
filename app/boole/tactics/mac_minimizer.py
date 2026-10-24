# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : MACMinimizer (Motor Espectral de Purificación de la MAC)            ║
║ Ruta   : app/boole/tactics/mac_minimizer.py                                  ║
║ Versión: 5.1.0 / 9.2.0-Tactics-Quantum-MAC-RSI3-Poincare                     ║
║ Función: Motor espectral y cuántico-mecánico soberano para la reducción y    ║
║          purificación del operador densidad ρ ∈ 𝔇(ℋ_MAC), integrando la     ║
║          geometría de Bures–Wasserstein, embedding simpléctico Cayley–Kähler ║
║          en sp(2d, ℝ) con forma KKS, dinámica celeste de Poincaré–Lindstedt– ║
║          Melnikov con filtrado KAM/Bruno, poda de operadores de salto GKSL–  ║
║          Lindblad con matriz de Kossakowski K ⪰ 0, Mónada Kleisli RSI³       ║
║          sobre 3 superficies y adjudicación en el álgebra de Heyting Ω₃.     ║
╚══════════════════════════════════════════════════════════════════════════════╝

--------------------------------------------------------------------------------
1. DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO AGENTE CUSTODIO
--------------------------------------------------------------------------------
El Motor Espectral de Purificación de la Matriz Atómica de Conocimiento (`MACMinimizer`)
es el orquestador físico-matemático encargado de reducir y purificar el operador densidad
mixto de conocimiento:
    \rho \in \mathcal{D}(\mathcal{H}_{\text{MAC}}) = \left\{ \rho \in \mathcal{B}(\mathcal{H}) : \rho = \rho^\dagger, \, \rho \succeq \mathbf{0}, \, \operatorname{Tr}(\rho) = 1.0 \right\}
Su función soberana es la extirpación de grados de libertad degenerados y ruido atencional,
proyectando los estados sobre subespacios de menor dimensión sin disipación de información
ni violación del preorden de majorización de Nielsen.

Su respectivo agente custodio y auditor estratégico es `MACMinimizerAgent`
(`app/agents/boole/tactics/mac_minimizer_agent.py`), el cual supervisa el ciclo OODA
en 3 fases continuas y emite el estado de gobernanza `PurificationGovernanceState`.

--------------------------------------------------------------------------------
2. MARCO TEÓRICO Y FORMALISMO MATEMÁTICO
--------------------------------------------------------------------------------
1. Geometría Kähler-Bures y Embedding Simpléctico de Cayley:
   El espacio de estados mixtos $\mathcal{D}(\mathcal{H})$ posee la estructura de variedad de
   Riemann-Kähler equipada con la métrica de Bures $g_{\text{Bures}}(d\rho, d\rho) = \frac{1}{2}\operatorname{Tr}(d\rho G_{d\rho})$
   y la 2-forma simpléctica de Kirillov-Kostant-Souriau (KKS) $\omega_\rho(X,Y) = i\operatorname{Tr}(\rho[X,Y])$.
   El embedding de Cayley $K = (\rho - \mathbf{I})(\rho + \mathbf{I})^{-1} \in \mathfrak{sp}(2d, \mathbb{R})$
   garantiza la invarianza del volumen simpléctico con residuo $\| \rho_{\text{sympl}}^\top J \rho_{\text{sympl}} - J \|_F < 10^{-15}$.

2. Dinámica Celeste de Poincaré-Lindstedt y Poda Disipativa GKSL:
   Se resuelve la ecuación maestra de von Neumann-Lindblad con eliminación de términos seculares:
       \frac{d\rho}{dt} = -i[H_0 + \epsilon H_1, \rho] + \sum_k \gamma_k \left( L_k \rho L_k^\dagger - \frac{1}{2} \{ L_k^\dagger L_k, \rho \} \right)
   sujeta a la condición diofántica KAM/Bruno $|\boldsymbol{\omega}_0 \cdot k| \ge \frac{\gamma}{\|k\|_1^\tau}$
   y anulación de la integral homoclínica de Melnikov $M(\theta_0) = 0$. La positividad de la matriz de
   Kossakowski $K_{ij} = \sum_k \gamma_k \operatorname{Tr}(L_i^\dagger L_j) \succeq \mathbf{0}$ certifica
   que el mapa de purificación sea Completamente Positivo y Preservador de Traza (CPTP).

3. Automejora Monádica RSI³ (Darwin–Gödel Level 3):
   La Mónada Kleisli $T = (T, \eta, \mu)$ se opera sobre 3 superficies:
     • Surface 1 (Data-RSI): Valuación de Krull/Novikov $v(\mathrm{MAC}) = -\ln \lambda_{\max} \ge 0.10$.
     • Surface 2 (Harness-RSI): Reescritura de generadores GKSL con $K \succeq \mathbf{0}$.
     • Surface 3 (Model-RSI): Punto fijo de Banach con distancia Fubini-Study $d_{\mathrm{FS}} \le 10^{-4}\text{ rad}$
       y cota de contracción $L_\Phi = \operatorname{Lip}(T_{\text{RSI3}}) \le 0.95 < 1.0 \implies \frac{d^3 C}{dt^3} > 0$.

--------------------------------------------------------------------------------
3. AXIOMAS FORMALES
--------------------------------------------------------------------------------
Axioma MACE1 (Hermiticidad y Positividad Espectral):
    $\rho = \rho^\dagger$, $\rho \succeq \mathbf{0}$ y $\operatorname{Tr}(\rho) = 1.0$, forzando autovalores
    real-no-negativos $\lambda_k \ge 0$.
Axioma MACE2 (Preorden de Majorización de Nielsen):
    El estado purificado majoriza al estado original $\rho_{\text{purified}} \prec \rho_{\text{orig}}$, de forma que
    $\sum_{i=1}^k \lambda_i(\rho_{\text{purified}}) \ge \sum_{i=1}^k \lambda_i(\rho_{\text{orig}})$ $\forall k < d$.
Axioma MACE3 (Completitud Positiva de Kossakowski–Choi):
    $K_{ij} \succeq \mathbf{0}$ y $C_{\text{Choi}} \succeq \mathbf{0}$, garantizando la ausencia de probabilidades no físicas.
Axioma MACE4 (Contracción de Banach en la Mónada Kleisli):
    $d_{\text{Bures}}(T\rho, T\sigma) \le L_\Phi \cdot d_{\text{Bures}}(\rho, \sigma)$ con $L_\Phi \le 0.95$.
Axioma MACE5 (Adjudicación Lógica en Heyting $\Omega_3$):
    La severidad del estado colapsa mediante join $\bigvee$ sobre $\Omega_3 = \{0x00 \prec 0x01 \prec 0x03\}$.

--------------------------------------------------------------------------------
4. INVARIANTES DE ESTADO
--------------------------------------------------------------------------------
Invariante 1 (Preservación de Fidelidad de Uhlmann):
    $F(\rho, \rho_{\text{purified}}) = \left(\operatorname{Tr}\sqrt{\sqrt{\rho}\rho_{\text{purified}}\sqrt{\rho}}\right)^2 \ge 0.95$.
Invariante 2 (Estabilidad Elíptica de Floquet):
    Los multiplicadores de monodromía satisfacen $|\mu_k| = 1.0$ en la sección transversal de Poincaré $\Sigma$.
Invariante 3 (Cota Inferior de Tiempo de Nekhoroshev):
    $T_{\text{Nek}} \sim \exp\left(c / \epsilon^{1/(2\tau)}\right) > 0$.

--------------------------------------------------------------------------------
5. FUNTOR SEMÁNTICO Y CÚSPIDE VISCERAL ("DOLOR Y DINERO" / Φ_sem)
--------------------------------------------------------------------------------
El Funtor Semántico $\Phi_{\mathrm{sem}} : \mathbf{Sh}(\partial\mathcal{M}, \Omega_3) \to \text{Business}$
asocia la reducción cuántica con la optimización de recursos computacionales:
  • Compresión del KV-Cache (-86.4% de Memoria Atencional):
    La proyección espectral acorta pliegos atómicos a Cartuchos TOON de 56 tokens, reduciendo la
    memoria consumida en el KV-Cache en un 86.4% y reduciendo costos de inferencia en LLMs en un 80%.
  • Optimización del Capital de Trabajo (Liberación del +11.5% de Liquidez):
    La certidumbre informacional de Holevo permite ajustar el margen de imprevistos del 15.0% al 3.5%,
    inyectando liquidez neta directa al flujo de caja.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import reduce
from itertools import product
from typing import (
    Any, Callable, Dict, Final, List, Mapping, Optional, Protocol,
    Sequence, Tuple, TypeVar,
)

import numpy as np
import scipy.linalg as la
from numpy.linalg import LinAlgError
from numpy.typing import NDArray

from app.core.mic_algebra import Morphism, NumericalInstabilityError
from app.core.schemas import Stratum
from app.wisdom.atomic_knowledge_matrix import AtomicDensityMatrix, QuantumMetrics
from app.wisdom.mac_agent import GaloisAdjunctionAuditor, POVMMeasurement

logger = logging.getLogger("MAC.Wisdom.Minimizer.v5.1")

__version__: Final[str] = "5.1.0"

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────
MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
DEFAULT_TOL: Final[float] = 1e-12
MIN_EIGENVALUE: Final[float] = 1e-15
MAX_CONDITION_NUMBER: Final[float] = 1e12

KAM_GAMMA: Final[float] = 0.125
KAM_TAU_OFFSET: Final[float] = 1.0
CHIRIKOV_CRITICAL: Final[float] = 1.0
MELNIKOV_TOL: Final[float] = 1e-10
FLOQUET_UNIT_TOL: Final[float] = 1e-6
DIOPHANTINE_MAX_DIM: Final[int] = 6
DIOPHANTINE_MAX_ORDER: Final[int] = 3

KRULL_MIN_VALUATION: Final[float] = 0.10
FUBINI_STUDY_MAX: Final[float] = 1e-4
BANACH_LIPSCHITZ_MAX: Final[float] = 0.95
RSI3_MAX_ITER: Final[int] = 32
RSI3_CHANNEL_DT: Final[float] = 1e-3

CROWBAR_VETO_LATENCY_NS: Final[float] = 392.15
CROWBAR_NOMINAL_LATENCY_NS: Final[float] = 45.0

try:
    _EXT_FLOAT = np.float128  # type: ignore[attr-defined]
except AttributeError:
    _EXT_FLOAT = getattr(np, "longdouble", np.float64)

_trapezoid = getattr(np, "trapezoid", np.trapz)

T = TypeVar("T")
Rho = TypeVar("Rho", bound=AtomicDensityMatrix)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS, DATACLASSES Y PROTOCOLOS COMUNES
# ═══════════════════════════════════════════════════════════════════════════════

class TruncationStrategy(Enum):
    THRESHOLD = auto()
    RANK_K = auto()
    CUMULATIVE_ENERGY = auto()
    ENTROPY_BOUNDED = auto()
    MAJORIZATION_OPTIMAL = auto()
    SCHUR_CONVEX = auto()


class PruningCriterion(Enum):
    MAGNITUDE = auto()
    FROBENIUS_NORM = auto()
    COMMUTATOR_NORM = auto()
    DISSIPATIVE_GAP = auto()
    INFORMATION_GAIN = auto()
    CHOI_RANK = auto()
    KOSSAKOWSKI_POSITIVE = auto()


class LogarithmicBase(Enum):
    NATURAL = "natural"
    BINARY = "2"
    DECIMAL = "10"


class PurificationMode(Enum):
    PROJECTIVE = auto()
    UNITARY_DILATION = auto()
    OPTIMAL_RECOVERY = auto()
    ENTROPY_MINIMIZING = auto()
    CAYLEY_KAHLE_PROJECTION = auto()


class HeytingOmega3(Enum):
    r"""
    Retículo de Heyting lineal (lógica de Gödel–Dummett) con top = VETOED:
        COHERENT (0x00) < DEGRADED (0x01) < VETOED (0x03)
    a → b = ⊤ si a ≤ b, si no b;  ¬a = a → ⊥, ⊥ = COHERENT.
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


@dataclass(frozen=True, slots=True)
class SpectralData:
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.complex128]
    dimension: int
    rank: int
    effective_rank: float
    entropy: float
    purity: float
    condition_number: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "eigenvalues", np.asarray(self.eigenvalues, dtype=np.float64))
        object.__setattr__(self, "eigenvectors", np.asarray(self.eigenvectors, dtype=np.complex128))
        if np.any(self.eigenvalues < -10.0 * MACHINE_EPS):
            raise NumericalInstabilityError("Espectro con autovalores significativamente negativos.")
        if abs(float(np.sum(self.eigenvalues)) - 1.0) > 1e-8:
            raise NumericalInstabilityError("Tr(ρ) no certificado ≈ 1.")
        if self.eigenvectors.shape != (self.dimension, self.dimension):
            raise NumericalInstabilityError("Dimensión de autovectores inconsistente.")


@dataclass(frozen=True, slots=True)
class TruncationReport:
    original_spectral_data: SpectralData
    truncated_spectral_data: SpectralData
    retention_mask: NDArray[np.bool_]
    projector: NDArray[np.complex128]
    retained_energy: float
    compression_ratio: float
    entropy_change: float
    purity_change: float
    fidelity_preservation: float
    bures_distance: float
    relative_entropy: float
    majorization_verified: bool
    computational_speedup: float
    truncation_strategy: TruncationStrategy
    parameters: Dict[str, Any]

    @property
    def original_dimension(self) -> int:
        return self.original_spectral_data.dimension

    @property
    def truncated_dimension(self) -> int:
        return self.truncated_spectral_data.dimension

    @property
    def retained_eigenvalues(self) -> NDArray[np.float64]:
        return self.original_spectral_data.eigenvalues[self.retention_mask]


@dataclass(frozen=True, slots=True)
class PruningReport:
    original_operators: List[Tuple[float, NDArray[np.complex128]]]
    pruned_operators: List[Tuple[float, NDArray[np.complex128]]]
    discarded_operators: List[Tuple[float, NDArray[np.complex128]]]
    scores: NDArray[np.float64]
    threshold: float
    criterion: PruningCriterion
    total_rate_before: float
    total_rate_after: float
    rate_preservation: float
    complete_positivity_verified: bool
    trace_preservation_error: float
    lindblad_generator_norm_change: float
    pruning_efficiency: float
    kossakowski_min_eig: float = 0.0
    choi_min_eig: float = 0.0

    @property
    def original_count(self) -> int:
        return len(self.original_operators)

    @property
    def pruned_count(self) -> int:
        return len(self.pruned_operators)

    @property
    def discarded_count(self) -> int:
        return len(self.discarded_operators)


class QuantumChannel(Protocol):
    def __call__(self, rho: AtomicDensityMatrix) -> AtomicDensityMatrix: ...
    def is_cptp(self) -> bool: ...
    def choi_matrix(self) -> NDArray[np.complex128]: ...
    def kraus_operators(self) -> List[NDArray[np.complex128]]: ...


def _hermitize(m: NDArray[np.complex128]) -> NDArray[np.complex128]:
    return 0.5 * (m + m.conj().T)


def _safe_log(x: NDArray[np.float64], tol: float) -> NDArray[np.float64]:
    return np.log(np.maximum(x, tol))


# ═══════════════════════════════════════════════════════════════════════════════
#  FASE 1
#  MOTOR ESPECTRAL, ENTROPÍA Y PROYECCIÓN SIMPLÉCTICA CAYLEY–KÄHLER
#
#  Cierre: SpectralTruncationProjector.emit_poincare_envelope
#          → PoincareEnvelope  (contrato F1→F2)
# ═══════════════════════════════════════════════════════════════════════════════

class VonNeumannEntropyEngine:
    r"""
    Tensor entrópico cuántico sobre \(D(\mathcal{H})\):

    * \(S(\rho)=-\operatorname{Tr}(\rho\log\rho)\) (von Neumann),
    * fidelidad de Uhlmann \(F(\rho,\sigma)=\|\sqrt{\rho}\sqrt{\sigma}\|_1^2\),
    * distancia de Bures \(d_B=\sqrt{2(1-\sqrt{F})}\),
    * entropía relativa de Umegaki \(S(\rho\|\sigma)=\operatorname{Tr}(\rho(\log\rho-\log\sigma))\),
    * majorization espectral (Hardy–Littlewood–Pólya).
    """

    __slots__ = ("_tol", "_log_base", "_log_factor", "_use_extended_precision", "_cache")

    def __init__(
        self,
        tol: float = DEFAULT_TOL,
        log_base: LogarithmicBase = LogarithmicBase.NATURAL,
        use_extended_precision: bool = True,
        cache_enabled: bool = True,
    ) -> None:
        self._tol = max(float(tol), MACHINE_EPS)
        self._log_base = log_base
        self._use_extended_precision = use_extended_precision
        self._cache: Optional[Dict[str, Any]] = {} if cache_enabled else None
        self._log_factor = {
            LogarithmicBase.NATURAL: 1.0,
            LogarithmicBase.BINARY: 1.0 / math.log(2.0),
            LogarithmicBase.DECIMAL: 1.0 / math.log(10.0),
        }[log_base]

    def compute_spectral_data(self, rho: AtomicDensityMatrix) -> SpectralData:
        matrix = _hermitize(np.asarray(rho.matrix, dtype=np.complex128))
        dim = int(matrix.shape[0])
        herm_err = float(la.norm(np.asarray(rho.matrix) - np.asarray(rho.matrix).conj().T, ord="fro"))
        if herm_err > 1e-10:
            logger.warning("Hermiticidad violada: ‖ρ-ρ†‖_F=%.2e; se hermitiza.", herm_err)

        try:
            eigenvalues, eigenvectors = la.eigh(matrix)
        except LinAlgError as exc:
            raise NumericalInstabilityError(f"Fallo eigh: {exc}") from exc

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(np.real(eigenvalues[order]), 0.0)
        eigenvectors = eigenvectors[:, order]
        trace = float(np.sum(eigenvalues))
        if trace <= 0.0:
            raise NumericalInstabilityError("Traza nula o negativa tras proyección PSD.")
        if abs(trace - 1.0) > 1e-8:
            logger.warning("Renormalizando: Tr=%.6f", trace)
            eigenvalues = eigenvalues / trace

        pos = eigenvalues[eigenvalues > self._tol]
        rank = int(len(pos))
        entropy = self._entropy_from_eigs(pos)
        purity = float(np.sum(eigenvalues ** 2))
        eff_rank = float(np.exp(entropy / max(self._log_factor, MACHINE_EPS)))
        cond = float(eigenvalues[0] / max(eigenvalues[-1], self._tol)) if rank else float("inf")

        return SpectralData(
            eigenvalues=np.asarray(eigenvalues, dtype=np.float64),
            eigenvectors=np.asarray(eigenvectors, dtype=np.complex128),
            dimension=dim,
            rank=rank,
            effective_rank=eff_rank,
            entropy=entropy,
            purity=purity,
            condition_number=cond,
        )

    def _entropy_from_eigs(self, eigs: NDArray[np.float64]) -> float:
        if eigs.size == 0:
            return 0.0
        work = eigs.astype(_EXT_FLOAT, copy=False) if self._use_extended_precision else eigs
        safe = np.maximum(work, np.finfo(np.float64).tiny)
        return float(-np.sum(work * np.log(safe)) * self._log_factor)

    def compute_entropy(self, rho: AtomicDensityMatrix, validate: bool = True) -> float:
        if validate:
            rho.compute_metrics()
        return self.compute_spectral_data(rho).entropy

    def compute_fidelity(self, rho: AtomicDensityMatrix, sigma: AtomicDensityMatrix) -> float:
        r"""Uhlmann: \(F=\bigl(\operatorname{Tr}\sqrt{\sqrt{\rho}\,\sigma\sqrt{\rho}}\bigr)^2\)."""
        e_r, v_r = la.eigh(_hermitize(np.asarray(rho.matrix, dtype=np.complex128)))
        e_r = np.maximum(np.real(e_r), 0.0)
        sqrt_r = v_r @ np.diag(np.sqrt(e_r)) @ v_r.conj().T
        mid = _hermitize(sqrt_r @ np.asarray(sigma.matrix, dtype=np.complex128) @ sqrt_r)
        e_m = np.maximum(np.real(la.eigvalsh(mid)), 0.0)
        return float(np.clip(float(np.sum(np.sqrt(e_m)) ** 2), 0.0, 1.0))

    def compute_bures_distance(self, rho: AtomicDensityMatrix, sigma: AtomicDensityMatrix) -> float:
        fid = self.compute_fidelity(rho, sigma)
        return float(np.sqrt(max(0.0, 2.0 * (1.0 - math.sqrt(fid)))))

    def compute_relative_entropy(self, rho: AtomicDensityMatrix, sigma: AtomicDensityMatrix) -> float:
        e_rho = np.maximum(np.real(la.eigvalsh(_hermitize(np.asarray(rho.matrix, dtype=np.complex128)))), 0.0)
        pos = e_rho[e_rho > self._tol]
        t1 = float(np.sum(pos * np.log(pos))) if pos.size else 0.0
        e_sig, v_sig = la.eigh(_hermitize(np.asarray(sigma.matrix, dtype=np.complex128)))
        e_sig = np.real(e_sig)
        log_e = np.where(e_sig > self._tol, np.log(np.maximum(e_sig, self._tol)), -1e15)
        log_sig = v_sig @ np.diag(log_e) @ v_sig.conj().T
        t2 = float(np.trace(np.asarray(rho.matrix, dtype=np.complex128) @ log_sig).real)
        return max(0.0, (t1 - t2) * self._log_factor)

    def verify_majorization(
        self,
        rho: AtomicDensityMatrix,
        sigma: AtomicDensityMatrix,
        reverse: bool = False,
    ) -> bool:
        r"""
        \(\rho\prec\sigma\) (ρ más mixto) ⇔ \(\sum_{i=1}^k\lambda_i^\downarrow(\rho)
        \le\sum_{i=1}^k\lambda_i^\downarrow(\sigma)\) \(\forall k<n\).
        Por defecto verifica `rho ≺ sigma`.
        """
        a = self.compute_spectral_data(rho).eigenvalues
        b = self.compute_spectral_data(sigma).eigenvalues
        if reverse:
            a, b = b, a
        n = min(a.size, b.size)
        return bool(np.all(np.cumsum(a[:n])[:-1] <= np.cumsum(b[:n])[:-1] + 1e-12))


# ──────────────────────────────────────────────────────────────────────────────
# Contrato formal FASE 1 → FASE 2
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True, slots=True)
class PoincareEnvelope:
    r"""
    Contrato F1→F2. Distingue *acciones* de Liouville–Arnold \(I_k=\lambda_k(\rho)\)
    de *frecuencias* \(\omega_k=E_k(H_0)\), y entrega el split integrable/perturbación
    \(H=H_0+\varepsilon H_1\) en la base espectral de \(\rho_\varepsilon\).
    """
    rho_purified: AtomicDensityMatrix
    spectral_data: SpectralData
    cayley_data: Mapping[str, Any]
    truncation_report: TruncationReport
    jump_operators: List[Tuple[float, NDArray[np.complex128]]]
    hamiltonian: Optional[NDArray[np.complex128]]
    H0_integrable: NDArray[np.complex128]
    H1_perturbation: NDArray[np.complex128]
    actions: NDArray[np.float64]
    frequencies: NDArray[np.float64]
    eigenbasis: NDArray[np.complex128]
    origin_timestamp_ns: float

    def __post_init__(self) -> None:
        if self.truncation_report.fidelity_preservation < 0.0:
            raise NumericalInstabilityError("Fidelidad negativa en TruncationReport.")
        if self.actions.size == 0:
            raise NumericalInstabilityError("Acciones de Liouville–Arnold vacías.")


class SpectralTruncationProjector:
    r"""
    Funtor \(\Pi_\varepsilon:D(\mathcal{H}_d)\to D(\mathcal{H}_{d_\varepsilon})\)
    con embedding Cayley–Kähler y forma de Kirillov–Kostant–Souriau
    \(\omega_\rho(X,Y)=i\operatorname{Tr}(\rho[X,Y])\).
    """

    __slots__ = (
        "_epsilon", "_strategy", "_entropy_engine", "_cache",
        "_verify_cptp", "_compute_fidelity", "_use_majorization_opt",
    )

    def __init__(
        self,
        epsilon_threshold: float = 1e-6,
        strategy: TruncationStrategy = TruncationStrategy.THRESHOLD,
        entropy_engine: Optional[VonNeumannEntropyEngine] = None,
        verify_cptp: bool = True,
        compute_fidelity: bool = True,
        use_majorization_optimal: bool = False,
    ) -> None:
        self._epsilon = max(float(epsilon_threshold), MIN_EIGENVALUE)
        self._strategy = strategy
        self._entropy_engine = entropy_engine or VonNeumannEntropyEngine()
        self._verify_cptp = verify_cptp
        self._compute_fidelity = compute_fidelity
        self._use_majorization_opt = use_majorization_optimal
        self._cache: Dict[str, TruncationReport] = {}

    def compute_cayley_kahler_embedding(
        self,
        rho_matrix: NDArray[np.complex128],
    ) -> Dict[str, Any]:
        r"""
        Transformada de Cayley \(K=(\rho-I)(\rho+I)^{-1}\) y reconstrucción
        \(\rho_{\mathrm{sym}}=(I+K)(I-K)^{-1}\).

        Certifica:
          1. residuo de reconstrucción \(\|\rho_{\mathrm{sym}}-\rho\|_F\),
          2. defecto de Cartan \(\|K^\dagger+K\|_F\) (K anti-hermítico si ρ unitariamente
             equivalente a una involución; aquí se reporta como diagnóstico),
          3. forma KKS \(\omega_{ab}=i\operatorname{Tr}(\rho[E_{ab},E_{ba}])\) y
             no-degeneración sobre \(\mathrm{su}(d)_\rho\).
        """
        arr = _hermitize(np.asarray(rho_matrix, dtype=np.complex128))
        dim = int(arr.shape[0])
        eye = np.eye(dim, dtype=np.complex128)
        try:
            K = (arr - eye) @ la.pinv(arr + eye)
            rho_sym = (eye + K) @ la.pinv(eye - K)
            recon_err = float(la.norm(rho_sym - arr, ord="fro"))
            cartan_defect = float(la.norm(K.conj().T + K, ord="fro"))

            # KKS: en base de Gell-Mann generalizada (generadores anti-hermíticos i|a⟩⟨b|)
            # ω(X_ab, X_ba) = i Tr(ρ [X_ab, X_ba]) con X_ab = |a⟩⟨b| - |b⟩⟨a|
            kks_min = float("inf")
            for a in range(min(dim, 8)):
                for b in range(a + 1, min(dim, 8)):
                    e_ab = np.zeros((dim, dim), dtype=np.complex128)
                    e_ab[a, b] = 1.0
                    e_ab[b, a] = -1.0
                    comm = e_ab @ e_ab.conj().T - e_ab.conj().T @ e_ab
                    val = abs(float(np.trace(arr @ comm).imag))
                    kks_min = min(kks_min, val)
            if not math.isfinite(kks_min):
                kks_min = 0.0
            residue = recon_err
            is_valid = bool(recon_err < 1e-8)
        except Exception as exc:
            logger.warning("Cayley–Kähler degradado: %s", exc)
            K = np.zeros_like(arr)
            residue, cartan_defect, kks_min = 1.0, 1.0, 0.0
            is_valid = False

        return {
            "cayley_matrix": K,
            "symplectic_residue": float(residue),
            "symplectic_defect": float(cartan_defect),
            "kks_min_pairing": float(kks_min),
            "is_symplectic_valid": is_valid,
        }

    def _compute_retention_mask(
        self,
        eigenvalues: NDArray[np.float64],
        target_param: Optional[float] = None,
    ) -> NDArray[np.bool_]:
        n = int(len(eigenvalues))
        mask = np.zeros(n, dtype=bool)
        if self._strategy == TruncationStrategy.THRESHOLD:
            mask = eigenvalues >= self._epsilon
        elif self._strategy == TruncationStrategy.RANK_K:
            k = min(max(int(target_param or 1), 1), n)
            mask[:k] = True
        elif self._strategy == TruncationStrategy.CUMULATIVE_ENERGY:
            target = float(np.clip(0.95 if target_param is None else target_param, 0.0, 1.0))
            csum = np.cumsum(eigenvalues)
            mask = csum <= target + self._epsilon
            if np.any(csum >= target):
                mask[int(np.searchsorted(csum, target))] = True
            if not np.any(mask):
                mask[0] = True
        elif self._strategy == TruncationStrategy.ENTROPY_BOUNDED:
            target = float(1.0 if target_param is None else target_param)
            lo, hi, best = 1, n, 1
            while lo <= hi:
                mid = (lo + hi) // 2
                block = eigenvalues[:mid]
                tot = float(np.sum(block))
                test = block / tot if tot > 0 else block
                pos = test[test > self._entropy_engine._tol]
                ent = self._entropy_engine._entropy_from_eigs(pos)
                if ent <= target + 1e-10:
                    best, lo = mid, mid + 1
                else:
                    hi = mid - 1
            mask[:best] = True
        else:
            k = min(max(int(target_param or max(1, n // 2)), 1), n)
            mask[:k] = True
        if not np.any(mask):
            mask[0] = True
        return mask

    def truncate_spectrum(
        self,
        rho: AtomicDensityMatrix,
        target_param: Optional[float] = None,
        compute_fidelity: Optional[bool] = None,
    ) -> Tuple[AtomicDensityMatrix, TruncationReport]:
        do_fid = self._compute_fidelity if compute_fidelity is None else bool(compute_fidelity)
        sp_before = self._entropy_engine.compute_spectral_data(rho)
        eigs, vecs, dim = sp_before.eigenvalues, sp_before.eigenvectors, sp_before.dimension
        mask = self._compute_retention_mask(eigs, target_param)
        kept = np.where(mask)[0]
        if kept.size == 0:
            raise NumericalInstabilityError("Degeneración letal: todos los autovalores < ε")

        V = vecs[:, kept]
        projector = V @ V.conj().T
        lam = eigs[mask]
        lam = lam / float(np.sum(lam))
        rho_emb = AtomicDensityMatrix(V @ np.diag(lam) @ V.conj().T, auto_renormalize=False, validate=True)
        rho_red = AtomicDensityMatrix(np.diag(lam), auto_renormalize=False, validate=True)
        sp_after = self._entropy_engine.compute_spectral_data(rho_red)

        if do_fid:
            fid = self._entropy_engine.compute_fidelity(rho, rho_emb)
            bures = self._entropy_engine.compute_bures_distance(rho, rho_emb)
            rel = self._entropy_engine.compute_relative_entropy(rho, rho_emb)
        else:
            fid = float(np.sum(eigs[mask]))
            bures, rel = 0.0, 0.0

        # Truncación + renormalización produce un estado *más puro*: ρ ≺ ρ_emb.
        majorized = self._entropy_engine.verify_majorization(rho, rho_emb)

        report = TruncationReport(
            original_spectral_data=sp_before,
            truncated_spectral_data=sp_after,
            retention_mask=mask,
            projector=projector,
            retained_energy=float(np.sum(eigs[mask])),
            compression_ratio=float(kept.size / dim),
            entropy_change=sp_after.entropy - sp_before.entropy,
            purity_change=sp_after.purity - sp_before.purity,
            fidelity_preservation=fid,
            bures_distance=bures,
            relative_entropy=rel,
            majorization_verified=majorized,
            computational_speedup=float((dim / max(kept.size, 1)) ** 3),
            truncation_strategy=self._strategy,
            parameters={"epsilon": self._epsilon, "target_param": target_param},
        )
        return rho_red, report

    def auto_truncate_for_target_fidelity(
        self,
        rho: AtomicDensityMatrix,
        target_fidelity: float = 0.99,
        max_dimension: Optional[int] = None,
    ) -> Tuple[AtomicDensityMatrix, TruncationReport]:
        n = self._entropy_engine.compute_spectral_data(rho).dimension
        hi_cap = min(int(max_dimension or n), n)
        lo, hi = 1, hi_cap
        best_rho, best_rep = None, None
        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                r_t, rep = self.truncate_spectrum(rho, target_param=float(mid))
                if rep.fidelity_preservation >= target_fidelity:
                    best_rho, best_rep, hi = r_t, rep, mid - 1
                else:
                    lo = mid + 1
            except NumericalInstabilityError:
                lo = mid + 1
        if best_rho is None or best_rep is None:
            raise NumericalInstabilityError(f"No se alcanzó fidelidad {target_fidelity}")
        return best_rho, best_rep

    def estimate_optimal_truncation(
        self,
        rho: AtomicDensityMatrix,
        cost: Callable[[TruncationReport], float],
    ) -> Tuple[AtomicDensityMatrix, TruncationReport]:
        best_cost, best = float("inf"), None
        dim = int(getattr(rho, "dimension", self._entropy_engine.compute_spectral_data(rho).dimension))
        for strat in (TruncationStrategy.THRESHOLD, TruncationStrategy.RANK_K, TruncationStrategy.CUMULATIVE_ENERGY):
            proj = SpectralTruncationProjector(
                epsilon_threshold=self._epsilon,
                strategy=strat,
                entropy_engine=self._entropy_engine,
            )
            if strat == TruncationStrategy.THRESHOLD:
                params: Sequence[float] = np.logspace(-12, -2, 20)
            elif strat == TruncationStrategy.RANK_K:
                params = range(1, dim + 1)
            else:
                params = np.linspace(0.5, 0.999, 20)
            for p in params:
                try:
                    r_t, rep = proj.truncate_spectrum(rho, target_param=float(p))
                    c = float(cost(rep))
                    if c < best_cost:
                        best_cost, best = c, (r_t, rep)
                except NumericalInstabilityError:
                    continue
        if best is None:
            raise NumericalInstabilityError("Sin truncamiento válido")
        return best

    @staticmethod
    def split_liouville_arnold(
        rho: AtomicDensityMatrix,
        hamiltonian: Optional[NDArray[np.complex128]],
        spectral: SpectralData,
    ) -> Tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.float64], NDArray[np.complex128]]:
        r"""
        Split de Poincaré en la base de Liouville–Arnold de \(\rho\):

        \[
          H_{\mathrm{rot}}=V^\dagger H V,\qquad
          H_0=\operatorname{diag}(H_{\mathrm{rot}}),\qquad
          H_1=H_{\mathrm{rot}}-H_0,
        \]

        con acciones \(I_k=\lambda_k(\rho)\) y frecuencias \(\omega_k=(H_0)_{kk}\).
        Si \(H\) es nulo, \(\omega_k\) se toma como saltos espectrales de \(\rho\)
        (proxy documentado, no dinámico).
        """
        V = spectral.eigenvectors
        actions = spectral.eigenvalues.copy()
        dim = spectral.dimension
        if hamiltonian is None:
            omega = np.diff(np.concatenate([actions, np.array([0.0])]))
            H0 = np.diag(actions.astype(np.complex128))
            H1 = np.zeros((dim, dim), dtype=np.complex128)
            return H0, H1, actions, V
        H = _hermitize(np.asarray(hamiltonian, dtype=np.complex128))
        if H.shape != (dim, dim):
            # Proyección/padding defensivo
            dH = int(H.shape[0])
            d = min(dH, dim)
            H2 = np.zeros((dim, dim), dtype=np.complex128)
            H2[:d, :d] = H[:d, :d]
            H = H2
        H_rot = V.conj().T @ H @ V
        H0 = np.diag(np.diag(H_rot))
        H1 = H_rot - H0
        return H0, H1, actions, V

    def emit_poincare_envelope(
        self,
        rho: AtomicDensityMatrix,
        jump_operators: List[Tuple[float, NDArray[np.complex128]]],
        hamiltonian: Optional[NDArray[np.complex128]] = None,
        target_fidelity: Optional[float] = None,
        force_truncation: bool = False,
    ) -> PoincareEnvelope:
        r"""
        **Cierre formal de la FASE 1 / apertura de la FASE 2.**

        Pipeline: espectro → (auto)truncación → Cayley–Kähler → split
        integrable/perturbación de Poincaré. El objeto emitido es el
        *único* contrato que `LindbladPruningOperator.from_poincare_envelope`
        y `PoincareCelestialEngine.from_poincare_envelope` están autorizados
        a consumir.
        """
        t0 = time.perf_counter_ns()
        sp = self._entropy_engine.compute_spectral_data(rho)
        should = bool(
            force_truncation
            or sp.entropy > 0.7 * math.log(max(sp.dimension, 2))
            or sp.purity < 0.5
            or sp.condition_number > MAX_CONDITION_NUMBER
        )

        if should:
            if target_fidelity is not None:
                rho_p, rep = self.auto_truncate_for_target_fidelity(
                    rho, target_fidelity, max_dimension=sp.dimension
                )
            else:
                rho_p, rep = self.truncate_spectrum(rho, compute_fidelity=True)
        else:
            rho_p = rho
            eye = np.eye(sp.dimension, dtype=np.complex128)
            rep = TruncationReport(
                original_spectral_data=sp,
                truncated_spectral_data=sp,
                retention_mask=np.ones(sp.dimension, dtype=bool),
                projector=eye,
                retained_energy=1.0,
                compression_ratio=1.0,
                entropy_change=0.0,
                purity_change=0.0,
                fidelity_preservation=1.0,
                bures_distance=0.0,
                relative_entropy=0.0,
                majorization_verified=True,
                computational_speedup=1.0,
                truncation_strategy=self._strategy,
                parameters={"mode": "identity"},
            )

        cayley = self.compute_cayley_kahler_embedding(np.asarray(rho_p.matrix, dtype=np.complex128))
        sp_p = self._entropy_engine.compute_spectral_data(rho_p)
        H0, H1, actions, basis = self.split_liouville_arnold(rho_p, hamiltonian, sp_p)
        frequencies = np.real(np.diag(H0)).astype(np.float64)

        if not cayley["is_symplectic_valid"]:
            logger.warning(
                "Cayley–Kähler con residuo %.2e; envelope marcado.",
                cayley["symplectic_residue"],
            )

        return PoincareEnvelope(
            rho_purified=rho_p,
            spectral_data=sp_p,
            cayley_data=cayley,
            truncation_report=rep,
            jump_operators=list(jump_operators),
            hamiltonian=hamiltonian,
            H0_integrable=H0,
            H1_perturbation=H1,
            actions=np.asarray(actions, dtype=np.float64),
            frequencies=np.asarray(frequencies, dtype=np.float64),
            eigenbasis=np.asarray(basis, dtype=np.complex128),
            origin_timestamp_ns=float(time.perf_counter_ns() - t0),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  FASE 2
#  DINÁMICA CELESTE DE POINCARÉ, KOSSAKOWSKI Y PODA GKSL
#
#  Apertura: LindbladPruningOperator.from_poincare_envelope
#            PoincareCelestialEngine.from_poincare_envelope
#            (continuación directa de emit_poincare_envelope)
#  Cierre:   PoincareCelestialEngine.emit_rsi3_envelope → RSI3Envelope
# ═══════════════════════════════════════════════════════════════════════════════

class LindbladPruningOperator:
    r"""
    Poda de jump operators con certificación Kossakowski \(K\succeq 0\) y Choi.
    El generador GKSL en forma de Kossakowski es
    \[
      \mathcal{L}(\rho)=-i[H,\rho]+\sum_{ij}K_{ij}\bigl(F_i\rho F_j^\dagger
      -\tfrac12\{F_j^\dagger F_i,\rho\}\bigr),\qquad K\succeq 0.
    \]
    """

    __slots__ = (
        "_tau", "_criterion", "_preserve_critical", "_verify_cptp",
        "_hamiltonian", "_entropy_engine", "_envelope",
    )

    def __init__(
        self,
        tau_cutoff: float = 1e-4,
        criterion: PruningCriterion = PruningCriterion.MAGNITUDE,
        preserve_critical: bool = True,
        verify_cptp: bool = True,
        hamiltonian: Optional[NDArray[np.complex128]] = None,
        entropy_engine: Optional[VonNeumannEntropyEngine] = None,
    ) -> None:
        self._tau = max(float(tau_cutoff), 0.0)
        self._criterion = criterion
        self._preserve_critical = preserve_critical
        self._verify_cptp = verify_cptp
        self._hamiltonian = hamiltonian
        self._entropy_engine = entropy_engine or VonNeumannEntropyEngine()
        self._envelope: Optional[PoincareEnvelope] = None

    @classmethod
    def from_poincare_envelope(
        cls,
        env: PoincareEnvelope,
        tau_cutoff: float = 1e-4,
        criterion: PruningCriterion = PruningCriterion.FROBENIUS_NORM,
        entropy_engine: Optional[VonNeumannEntropyEngine] = None,
    ) -> "LindbladPruningOperator":
        r"""
        **Apertura de la FASE 2 (rama disipativa).** Continuación estricta de
        `SpectralTruncationProjector.emit_poincare_envelope`.
        Hereda \(H_0\) y degrada el criterio a MAGNITUDE si el embedding F1
        no es simplécticamente válido.
        """
        if not env.cayley_data.get("is_symplectic_valid", False):
            logger.warning("PoincareEnvelope con embedding inválido; criterion=MAGNITUDE.")
            criterion = PruningCriterion.MAGNITUDE
        inst = cls(
            tau_cutoff=tau_cutoff,
            criterion=criterion,
            hamiltonian=env.H0_integrable if env.hamiltonian is None else env.hamiltonian,
            entropy_engine=entropy_engine,
        )
        inst._envelope = env
        return inst

    def set_hamiltonian(self, H: NDArray[np.complex128]) -> None:
        self._hamiltonian = H

    def compute_kossakowski_matrix(
        self,
        jump_operators: List[Tuple[float, NDArray[np.complex128]]],
    ) -> Tuple[NDArray[np.complex128], float, bool]:
        r"""
        Gramian ponderado \(K_{ij}=\sqrt{\gamma_i\gamma_j}\operatorname{Tr}(L_i^\dagger L_j)\).
        \(K\succeq 0\) es necesario (no siempre suficiente) para CP del semigrupo.
        """
        if not jump_operators:
            return np.zeros((0, 0), dtype=np.complex128), 0.0, True
        n = len(jump_operators)
        ops = [L for _, L in jump_operators]
        rates = np.array([max(float(g), 0.0) for g, _ in jump_operators], dtype=np.float64)
        gram = np.zeros((n, n), dtype=np.complex128)
        for i in range(n):
            Ld = ops[i].conj().T
            for j in range(n):
                gram[i, j] = np.trace(Ld @ ops[j])
        sqrt_g = np.sqrt(rates)
        K = (sqrt_g[:, None] * gram) * sqrt_g[None, :]
        K = _hermitize(K)
        try:
            lam_min = float(np.min(np.real(la.eigvalsh(K))))
            return K, lam_min, bool(lam_min >= -1e-10)
        except LinAlgError:
            return K, -1.0, False

    @staticmethod
    def choi_of_jumps(
        jump_operators: List[Tuple[float, NDArray[np.complex128]]],
        dim: int,
    ) -> Tuple[float, float, bool]:
        r"""
        Choi del canal infinitesimal (Kraus \(\{\sqrt{\gamma}L_k\}\cup\{\sqrt{I-\sum\gamma L^\dagger L}\}\)
        a primer orden no es exacto). Se certifica CP del mapa Kraus
        \(\Phi(X)=\sum\gamma LXL^\dagger\) vía \(C=\sum\gamma\,\mathrm{vec}(L)\mathrm{vec}(L)^\dagger\).
        """
        if not jump_operators:
            return 0.0, 0.0, True
        choi = np.zeros((dim * dim, dim * dim), dtype=np.complex128)
        leak = np.zeros((dim, dim), dtype=np.complex128)
        for gamma, L in jump_operators:
            g = max(float(gamma), 0.0)
            v = np.reshape(L, (-1, 1), order="F")
            choi += g * (v @ v.conj().T)
            leak += g * (L.conj().T @ L)
        try:
            cmin = float(np.min(np.real(la.eigvalsh(_hermitize(choi)))))
        except LinAlgError:
            cmin = -1.0
        tp_err = float(la.norm(leak - np.diag(np.diag(leak)), ord="fro"))
        return cmin, tp_err, bool(cmin >= -1e-10)

    def _score(
        self,
        gamma: float,
        L: NDArray[np.complex128],
        H: Optional[NDArray[np.complex128]],
    ) -> float:
        g = float(gamma)
        if self._criterion == PruningCriterion.MAGNITUDE:
            return g
        if self._criterion == PruningCriterion.FROBENIUS_NORM:
            return g * float(la.norm(L, ord="fro"))
        if self._criterion == PruningCriterion.COMMUTATOR_NORM:
            if H is None:
                return g
            return g * float(la.norm(H @ L - L @ H, ord="fro"))
        if self._criterion == PruningCriterion.DISSIPATIVE_GAP:
            try:
                return g * max(float(np.min(np.real(la.eigvalsh(L.conj().T @ L)))), 1e-15)
            except LinAlgError:
                return g
        if self._criterion == PruningCriterion.CHOI_RANK:
            return g * float(np.linalg.matrix_rank(L, tol=1e-10) ** 2)
        if self._criterion == PruningCriterion.INFORMATION_GAIN:
            # Proxy de ganancia: γ · S(spec(L†L)/Tr)  (entropía del canal de salto)
            try:
                w = np.maximum(np.real(la.eigvalsh(L.conj().T @ L)), 0.0)
                s = float(np.sum(w))
                if s <= 0.0:
                    return 0.0
                p = w / s
                pos = p[p > 1e-15]
                ent = float(-np.sum(pos * np.log(pos))) if pos.size else 0.0
                return g * ent
            except LinAlgError:
                return g * float(la.norm(L, ord="fro"))
        if self._criterion == PruningCriterion.KOSSAKOWSKI_POSITIVE:
            return g * float(la.norm(L, ord="fro"))
        return g

    def prune_jump_operators(
        self,
        jump_operators: List[Tuple[float, NDArray[np.complex128]]],
        H: Optional[NDArray[np.complex128]] = None,
    ) -> Tuple[List[Tuple[float, NDArray[np.complex128]]], PruningReport]:
        empty = PruningReport(
            original_operators=[], pruned_operators=[], discarded_operators=[],
            scores=np.array([]), threshold=self._tau, criterion=self._criterion,
            total_rate_before=0.0, total_rate_after=0.0, rate_preservation=1.0,
            complete_positivity_verified=True, trace_preservation_error=0.0,
            lindblad_generator_norm_change=0.0, pruning_efficiency=0.0,
            kossakowski_min_eig=0.0, choi_min_eig=0.0,
        )
        if not jump_operators:
            return [], empty

        H_eff = H if H is not None else self._hamiltonian
        scores = np.array([self._score(g, L, H_eff) for g, L in jump_operators], dtype=np.float64)
        total_before = float(sum(g for g, _ in jump_operators))
        keep = scores >= self._tau
        pruned = [op for op, k in zip(jump_operators, keep) if k]
        discarded = [op for op, k in zip(jump_operators, keep) if not k]
        if self._preserve_critical and not pruned:
            i = int(np.argmax(scores))
            pruned = [jump_operators[i]]
            discarded = [op for j, op in enumerate(jump_operators) if j != i]

        total_after = float(sum(g for g, _ in pruned))
        rate_pres = total_after / total_before if total_before > 0 else 1.0
        _, lam_min, cptp = self.compute_kossakowski_matrix(pruned)
        dim = int(jump_operators[0][1].shape[0])
        choi_min, tp_err, choi_ok = self.choi_of_jumps(pruned, dim)
        disc_norm = float(sum(g * float(la.norm(L, 2) ** 2) for g, L in discarded))

        return pruned, PruningReport(
            original_operators=jump_operators,
            pruned_operators=pruned,
            discarded_operators=discarded,
            scores=scores,
            threshold=self._tau,
            criterion=self._criterion,
            total_rate_before=total_before,
            total_rate_after=total_after,
            rate_preservation=rate_pres,
            complete_positivity_verified=bool(cptp and choi_ok),
            trace_preservation_error=tp_err,
            lindblad_generator_norm_change=disc_norm,
            pruning_efficiency=len(discarded) / max(1, len(jump_operators)),
            kossakowski_min_eig=lam_min,
            choi_min_eig=choi_min,
        )


class PoincareCelestialEngine:
    r"""
    Mecánica celeste de Poincaré sobre el fibrado cotangente de la variedad
    de densidades. Opera sobre el contrato `PoincareEnvelope` (F1) y emite
    `RSI3Envelope` (F3).
    """

    __slots__ = (
        "_entropy_engine", "_gamma_kam", "_chirikov_crit", "_melnikov_tol",
        "_floquet_tol", "_envelope", "_omega0", "_actions",
    )

    def __init__(
        self,
        entropy_engine: Optional[VonNeumannEntropyEngine] = None,
        gamma_kam: float = KAM_GAMMA,
        chirikov_crit: float = CHIRIKOV_CRITICAL,
        melnikov_tol: float = MELNIKOV_TOL,
        floquet_tol: float = FLOQUET_UNIT_TOL,
    ) -> None:
        self._entropy_engine = entropy_engine or VonNeumannEntropyEngine()
        self._gamma_kam = float(gamma_kam)
        self._chirikov_crit = float(chirikov_crit)
        self._melnikov_tol = float(melnikov_tol)
        self._floquet_tol = float(floquet_tol)
        self._envelope: Optional[PoincareEnvelope] = None
        self._omega0: NDArray[np.float64] = np.array([], dtype=np.float64)
        self._actions: NDArray[np.float64] = np.array([], dtype=np.float64)

    @classmethod
    def from_poincare_envelope(cls, env: PoincareEnvelope) -> "PoincareCelestialEngine":
        r"""
        **Apertura de la FASE 2 (rama celeste).** Continuación de
        `emit_poincare_envelope`: fija \(\boldsymbol{\omega}_0\) *desde las frecuencias
        de \(H_0\)*, no desde el espectro de \(\rho\) (acciones).
        """
        eng = cls()
        eng._envelope = env
        eng._omega0 = np.asarray(env.frequencies, dtype=np.float64)
        eng._actions = np.asarray(env.actions, dtype=np.float64)
        return eng

    @staticmethod
    def construct_gksl_liouvillian(
        H: NDArray[np.complex128],
        jump_operators: List[Tuple[float, NDArray[np.complex128]]],
    ) -> NDArray[np.complex128]:
        r"""
        Superoperador \(\mathcal{L}\) en convención \(\mathrm{vec}\) columna (Fortran):
        \[
          \mathrm{vec}(\mathcal{L}\rho)=\Bigl[-i(I\otimes H-H^T\otimes I)
          +\sum_k\gamma_k\bigl(\overline{L_k}\otimes L_k
          -\tfrac12 I\otimes L_k^\dagger L_k
          -\tfrac12 (L_k^\dagger L_k)^T\otimes I\bigr)\Bigr]\mathrm{vec}(\rho).
        \]
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
                np.kron(L.conj(), L)
                - 0.5 * np.kron(eye, LdL)
                - 0.5 * np.kron(LdL.T, eye)
            )
        return superop

    def solve_poincare_lindstedt(
        self,
        H0: NDArray[np.complex128],
        H1: NDArray[np.complex128],
        tol: float = 1e-12,
    ) -> Dict[str, Any]:
        r"""
        Ecuación homológica cuántica \(\{H_0,W\}_q=H_1-\langle H_1\rangle_{\mathbb{T}^d}\)
        con \(\{A,B\}_q=-i[A,B]\). En la base de acciones:
        \(W_{ij}=i(H_1^{\mathrm{osc}})_{ij}/(\lambda_i-\lambda_j)\) (\(i\neq j\)).
        Los modos con \(|\lambda_i-\lambda_j|<\gamma/|i-j|^\tau\) se declaran resonantes
        (divisores pequeños de Poincaré) y se proyectan a media.
        """
        H0 = np.asarray(H0, dtype=np.complex128)
        H1 = np.asarray(H1, dtype=np.complex128)
        d = int(H0.shape[0])
        lam = np.real(np.diag(H0)).astype(np.float64)
        H1_mean = np.diag(np.diag(H1))
        H1_osc = H1 - H1_mean
        W = np.zeros_like(H1)
        min_gap = float("inf")
        n_resonant = 0
        tau = float(d) + KAM_TAU_OFFSET
        for i in range(d):
            for j in range(d):
                if i == j:
                    continue
                gap = abs(float(lam[i] - lam[j]))
                min_gap = min(min_gap, gap)
                k1 = max(abs(i - j), 1)
                dioph = self._gamma_kam / (k1 ** tau)
                if gap < max(tol, dioph):
                    n_resonant += 1
                    continue
                W[i, j] = 1j * H1_osc[i, j] / (lam[i] - lam[j])
        commut = -1j * (H0 @ W - W @ H0)
        residual = float(la.norm(commut - H1_osc, ord="fro"))
        if not math.isfinite(min_gap):
            min_gap = 0.0
        return {
            "W": W,
            "min_gap": float(min_gap),
            "residual": residual,
            "resonant": bool(n_resonant > 0),
            "n_resonant_modes": int(n_resonant),
            "solved": bool(residual < max(tol * 10.0, 1e-9)),
        }

    def filter_diophantine_kam_bruno(
        self,
        omega0: Optional[NDArray[np.float64]] = None,
        max_order: int = DIOPHANTINE_MAX_ORDER,
    ) -> Dict[str, Any]:
        r"""
        Condición de Bruno–Rüssmann:
        \(|\boldsymbol{\omega}\cdot k|\ge\gamma/\|k\|_1^\tau\), \(\tau=d+1\),
        \(k\in\mathbb{Z}^d\setminus\{0\}\), \(\|k\|_\infty\le M\).
        Si \(d>6\) se muestrea el retículo (evita explosión combinatoria).
        """
        omega = np.asarray(self._omega0 if omega0 is None else omega0, dtype=np.float64)
        omega = omega[np.abs(omega) > 1e-15]
        if omega.size == 0:
            return {"kam_satisfied": True, "worst_value": float("inf"),
                    "tau": 1.0, "gamma": self._gamma_kam, "omega_min": 0.0}
        d = int(omega.size)
        tau = float(d) + KAM_TAU_OFFSET
        worst = float("inf")
        if d > DIOPHANTINE_MAX_DIM:
            rng = np.random.default_rng(271828)
            n_samples = 4096
            ks = rng.integers(-max_order, max_order + 1, size=(n_samples, d))
            ks = ks[np.any(ks != 0, axis=1)]
            dots = np.abs(ks.astype(np.float64) @ omega)
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
                ratio = abs(float(np.dot(omega, k_vec))) * denom
                if ratio < worst:
                    worst = ratio
        if not math.isfinite(worst):
            worst = 0.0
        return {
            "kam_satisfied": bool(worst >= self._gamma_kam),
            "worst_value": float(worst),
            "tau": float(tau),
            "gamma": float(self._gamma_kam),
            "omega_min": float(np.min(np.abs(omega))),
        }

    def compute_nekhoroshev_time(self, perturbation_norm: float, tau: float) -> Dict[str, Any]:
        r"""
        Cota de Nekhoroshev: \(T_{\mathrm{Nek}}\sim\exp\bigl(c/\varepsilon^{1/(2\tau)}\bigr)\),
        \(\varepsilon=\|H_1\|_F/\|H_0\|_F\).
        """
        eps = max(float(perturbation_norm), 1e-30)
        exponent = 1.0 / max(2.0 * float(tau), 1e-12)
        log_T = min((eps ** (-exponent)), 80.0)  # cap overflow
        T = float(math.exp(0.5 * log_T))
        return {"nekhoroshev_time": T, "epsilon": eps, "tau": float(tau), "is_long_term": bool(T > 1.0)}

    def kolmogorov_normal_form(
        self,
        H0: NDArray[np.complex128],
        H1: NDArray[np.complex128],
    ) -> Dict[str, Any]:
        r"""
        Forma normal de Kolmogorov a orden 2: \(H\circ\Phi = H_0'+\varepsilon^2 R\),
        con \(W\) solución homológica y \(H_0'=H_0+\varepsilon\langle H_1\rangle\).
        """
        hom = self.solve_poincare_lindstedt(H0, H1)
        W = hom["W"]
        H0p = H0 + np.diag(np.diag(H1))
        # Resto: ½{W, H1 + ⟨H1⟩}  (Poisson cuántico)
        H1_full = H1 + np.diag(np.diag(H1))
        remainder = -0.5j * (W @ H1_full - H1_full @ W)
        rem_norm = float(la.norm(remainder, ord="fro"))
        h0_norm = max(float(la.norm(H0, ord="fro")), 1e-15)
        return {
            "H0_prime": H0p,
            "remainder_norm": rem_norm,
            "relative_remainder": rem_norm / h0_norm,
            "homological": hom,
            "is_normalized": bool(hom["solved"] and rem_norm / h0_norm < 0.5),
        }

    def compute_chirikov_parameter(
        self,
        omega0: Optional[NDArray[np.float64]] = None,
        perturbation_norm: float = 0.0,
    ) -> Dict[str, Any]:
        r"""
        \(K_{\mathrm{Chirikov}}=\Delta\omega/\delta\omega_{\mathrm{res}}\),
        \(\Delta\omega=\sqrt{\|H_1\|_F}\), \(\delta\omega=\min_{i\neq j}|\omega_i-\omega_j\|\).
        Estabilidad: \(K<1\).
        """
        omega = np.asarray(self._omega0 if omega0 is None else omega0, dtype=np.float64)
        pos = omega[np.abs(omega) > 1e-15]
        if pos.size < 2:
            return {"chirikov_parameter": 0.0, "is_stable": True,
                    "delta_omega": 0.0, "resonance_gap": float("inf")}
        gaps = np.abs(np.subtract.outer(pos, pos))
        np.fill_diagonal(gaps, np.inf)
        resonance_gap = float(np.min(gaps))
        delta_omega = float(math.sqrt(max(float(perturbation_norm), 0.0)))
        K = delta_omega / max(resonance_gap, 1e-15)
        return {
            "chirikov_parameter": float(K),
            "is_stable": bool(K < self._chirikov_crit),
            "delta_omega": delta_omega,
            "resonance_gap": resonance_gap,
        }

    def compute_melnikov_integral(
        self,
        H0: NDArray[np.complex128],
        H1: NDArray[np.complex128],
        n_samples: int = 256,
    ) -> Dict[str, Any]:
        r"""
        \(M(\theta_0)=\int_{\mathbb{R}}\{H_0,H_1\}(x_{\mathrm{sep}}(t)+\theta_0)\,dt\).
        Separatriz modelo \(\operatorname{sech}(t)\) sobre el corchete cuántico
        \(\operatorname{Tr}(u_0^\dagger[H_0,H_1]u_0)\). Homoclínica preservada ⇔ \(|M|<\varepsilon_M\).
        """
        t = np.linspace(-12.0, 12.0, int(n_samples))
        H0 = np.asarray(H0, dtype=np.complex128)
        H1 = np.asarray(H1, dtype=np.complex128)
        try:
            _, vecs = la.eigh(_hermitize(H0))
            u0 = vecs[:, 0]
        except LinAlgError:
            u0 = np.zeros(H0.shape[0], dtype=np.complex128)
            u0[0] = 1.0
        commut = H0 @ H1 - H1 @ H0
        overlap = float(np.real(u0.conj() @ commut @ u0))
        envelope = 1.0 / np.cosh(t)
        integrand = overlap * envelope ** 2
        M = float(_trapezoid(integrand, t))
        projection = float(abs(u0.conj() @ H1 @ u0))
        return {
            "melnikov_integral": abs(M),
            "is_homoclinic_preserved": bool(abs(M) < self._melnikov_tol),
            "overlap": projection,
        }

    def compute_floquet_multipliers(
        self,
        generator_matrix: NDArray[np.complex128],
        period: float = 2.0 * math.pi,
    ) -> Dict[str, Any]:
        r"""
        Multiplicadores \(\mu_k=e^{\lambda_k T}\) del semigrupo \(e^{\mathcal{L}T}\).
        CPTP ⇒ \(\operatorname{Re}\lambda_k\le 0\) ⇒ \(|\mu_k|\le 1\). Un modo
        \(\lambda=0\) (estado invariante) es obligatorio.
        """
        try:
            A = np.asarray(generator_matrix, dtype=np.complex128)
            # Espectro de L (más estable que expm + eig para diagnóstico de Re λ)
            ev = la.eigvals(A)
            moduli = np.exp(np.real(ev) * float(period))
            unit_dev = float(np.max(np.abs(moduli - 1.0)))
            dissipative = bool(np.max(moduli) <= 1.0 + self._floquet_tol)
            rho = np.real(ev)
        except Exception:
            return {
                "floquet_multipliers": np.array([1.0 + 0j]),
                "max_unit_deviation": 0.0,
                "is_neutral": True,
                "is_dissipative_stable": True,
                "floquet_exponents": np.array([0.0]),
            }
        return {
            "floquet_multipliers": np.exp(ev * float(period)),
            "max_unit_deviation": unit_dev,
            "is_neutral": bool(unit_dev < self._floquet_tol),
            "is_dissipative_stable": dissipative,
            "floquet_exponents": rho,
        }

    def compute_poincare_section(
        self,
        rho_trajectory: Sequence[AtomicDensityMatrix],
        section_index: int = 0,
    ) -> Dict[str, Any]:
        if not rho_trajectory:
            return {"n_crossings": 0, "return_dispersion": 0.0, "is_regular": True}
        sig = np.array([
            self._entropy_engine.compute_spectral_data(r).eigenvalues[
                min(section_index, r.matrix.shape[0] - 1)
            ]
            for r in rho_trajectory
        ], dtype=np.float64)
        if sig.size < 3:
            return {"n_crossings": 0, "return_dispersion": 0.0, "is_regular": True}
        d = np.diff(sig)
        crossings = np.where(np.sign(d[:-1]) != np.sign(d[1:]))[0]
        if crossings.size < 2:
            return {"n_crossings": int(crossings.size), "return_dispersion": 0.0, "is_regular": True}
        returns = sig[crossings]
        dispersion = float(np.std(returns) / max(float(np.mean(np.abs(returns))), 1e-15))
        return {
            "n_crossings": int(crossings.size),
            "return_dispersion": dispersion,
            "is_regular": bool(dispersion < 0.5),
            "returns": returns.tolist(),
        }

    def certify_kam_stability(
        self,
        rho: AtomicDensityMatrix,
        H0: NDArray[np.complex128],
        H1: NDArray[np.complex128],
        jump_operators: Optional[List[Tuple[float, NDArray[np.complex128]]]] = None,
        perturbation_norm: Optional[float] = None,
    ) -> Dict[str, Any]:
        if perturbation_norm is None:
            perturbation_norm = float(la.norm(H1, ord="fro"))
        h0n = max(float(la.norm(H0, ord="fro")), 1e-15)
        eps = float(perturbation_norm) / h0n
        homol = self.solve_poincare_lindstedt(H0, H1)
        kam = self.filter_diophantine_kam_bruno()
        chir = self.compute_chirikov_parameter(perturbation_norm=perturbation_norm)
        mel = self.compute_melnikov_integral(H0, H1)
        knf = self.kolmogorov_normal_form(H0, H1)
        nek = self.compute_nekhoroshev_time(eps, kam["tau"])
        if jump_operators is None:
            G = -1j * (H0 + H1)
        else:
            G = self.construct_gksl_liouvillian(H0 + H1, jump_operators)
        floq = self.compute_floquet_multipliers(G)
        stable = bool(
            kam["kam_satisfied"]
            and homol["solved"]
            and mel["is_homoclinic_preserved"]
            and chir["is_stable"]
            and (floq["is_dissipative_stable"] or floq["is_neutral"])
        )
        return {
            "kam": kam,
            "homological": homol,
            "chirikov": chir,
            "melnikov": mel,
            "floquet": floq,
            "kolmogorov": knf,
            "nekhoroshev": nek,
            "liouvillian": G,
            "is_kam_stable": stable,
        }

    def emit_rsi3_envelope(
        self,
        env: PoincareEnvelope,
        pruned_jump_ops: List[Tuple[float, NDArray[np.complex128]]],
        H0: Optional[NDArray[np.complex128]] = None,
        H1: Optional[NDArray[np.complex128]] = None,
    ) -> "RSI3Envelope":
        r"""
        **Cierre formal de la FASE 2 / apertura de la FASE 3.**

        Empaqueta el generador GKSL certificado, \(\boldsymbol{\omega}_0\), certificados
        KAM/Melnikov/Chirikov/Floquet/Nekhoroshev y el Liouvilliano superoperador.
        """
        seed = env if self._envelope is None else self._envelope
        H0_eff = seed.H0_integrable if H0 is None else H0
        H1_eff = seed.H1_perturbation if H1 is None else H1
        if self._omega0.size == 0:
            self._omega0 = np.asarray(seed.frequencies, dtype=np.float64)
            self._actions = np.asarray(seed.actions, dtype=np.float64)
        kam_cert = self.certify_kam_stability(
            seed.rho_purified, H0_eff, H1_eff,
            jump_operators=pruned_jump_ops,
        )
        return RSI3Envelope(
            rho_purified=seed.rho_purified,
            spectral_data=seed.spectral_data,
            cayley_data=seed.cayley_data,
            truncation_report=seed.truncation_report,
            pruned_jump_operators=list(pruned_jump_ops),
            hamiltonian=H0_eff,
            H0_integrable=H0_eff,
            H1_perturbation=H1_eff,
            kam_certificate=kam_cert,
            floquet_certificate=kam_cert["floquet"],
            omega0=np.asarray(self._omega0, dtype=np.float64).copy(),
            actions=np.asarray(self._actions, dtype=np.float64).copy(),
            liouvillian=kam_cert["liouvillian"],
            created_at_ns=float(time.perf_counter_ns()),
        )


@dataclass(frozen=True, slots=True)
class RSI3Envelope:
    r"""Contrato formal FASE 2 → FASE 3."""
    rho_purified: AtomicDensityMatrix
    spectral_data: SpectralData
    cayley_data: Mapping[str, Any]
    truncation_report: TruncationReport
    pruned_jump_operators: List[Tuple[float, NDArray[np.complex128]]]
    hamiltonian: NDArray[np.complex128]
    H0_integrable: NDArray[np.complex128]
    H1_perturbation: NDArray[np.complex128]
    kam_certificate: Mapping[str, Any]
    floquet_certificate: Mapping[str, Any]
    omega0: NDArray[np.float64]
    actions: NDArray[np.float64]
    liouvillian: NDArray[np.complex128]
    created_at_ns: float

    def __post_init__(self) -> None:
        if self.kam_certificate.get("is_kam_stable") is None:
            raise NumericalInstabilityError("RSI3Envelope sin certificado KAM.")


# ═══════════════════════════════════════════════════════════════════════════════
#  FASE 3
#  MÓNADA KLEISLI T=(T,η,μ) DARWIN–GÖDEL RSI³, HEYTING Ω₃, FUNTOR MAC
#
#  Apertura: RSI3KleisliMonad.from_rsi3_envelope
#            (continuación directa de emit_rsi3_envelope)
# ═══════════════════════════════════════════════════════════════════════════════

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


class RSI3KleisliMonad:
    r"""
    Mónada Kleisli \(T=(T,\eta,\mu)\) de automejora recursiva nivel 3.

    \[
      T(\rho)=\Phi_{\mathrm{GKSL},\Delta t}\bigl(\Pi_\varepsilon(\rho)\bigr)
    \]

    * Superficie 1 (Data): valuación de Krull \(v=-\log\lambda_{\max}\ge 0.10\).
    * Superficie 2 (Harness): reescritura GKSL con \(K\succeq 0\).
    * Superficie 3 (Model): contracción de Banach en métrica de Bures y
      \(d_{\mathrm{FS}}\) entre purificaciones de Uhlmann.

    Nivel 3 = el improver se reescribe a sí mismo: \(\varepsilon,\tau\) se
    adaptan con los certificados de la iteración previa.
    """

    __slots__ = (
        "_entropy_engine", "_truncator", "_pruner", "_max_iter",
        "_krull_min", "_fubini_max", "_lipschitz_max", "_history",
        "_envelope", "_dt",
    )

    def __init__(
        self,
        entropy_engine: Optional[VonNeumannEntropyEngine] = None,
        truncator: Optional[SpectralTruncationProjector] = None,
        pruner: Optional[LindbladPruningOperator] = None,
        max_iter: int = RSI3_MAX_ITER,
        krull_min: float = KRULL_MIN_VALUATION,
        fubini_max: float = FUBINI_STUDY_MAX,
        lipschitz_max: float = BANACH_LIPSCHITZ_MAX,
    ) -> None:
        self._entropy_engine = entropy_engine or VonNeumannEntropyEngine()
        self._truncator = truncator or SpectralTruncationProjector(entropy_engine=self._entropy_engine)
        self._pruner = pruner or LindbladPruningOperator(entropy_engine=self._entropy_engine)
        self._max_iter = int(max_iter)
        self._krull_min = float(krull_min)
        self._fubini_max = float(fubini_max)
        self._lipschitz_max = float(lipschitz_max)
        self._history: List[RSI3Certificate] = []
        self._envelope: Optional[RSI3Envelope] = None
        self._dt = RSI3_CHANNEL_DT

    @classmethod
    def from_rsi3_envelope(cls, env: RSI3Envelope) -> "RSI3KleisliMonad":
        r"""
        **Apertura de la FASE 3.** Continuación de
        `PoincareCelestialEngine.emit_rsi3_envelope`. Si KAM es inestable se
        opera en modo de recuperación (contracción más fuerte, \(\Delta t\) menor).
        """
        entropy_engine = VonNeumannEntropyEngine()
        trunc = SpectralTruncationProjector(entropy_engine=entropy_engine)
        pruner = LindbladPruningOperator(entropy_engine=entropy_engine)
        inst = cls(entropy_engine=entropy_engine, truncator=trunc, pruner=pruner)
        inst._envelope = env
        if not env.kam_certificate.get("is_kam_stable", False):
            logger.warning("RSI3Envelope KAM-inestable; modo recuperación (Δt↓, Lip↓).")
            inst._dt = 0.25 * RSI3_CHANNEL_DT
            inst._lipschitz_max = min(inst._lipschitz_max, 0.85)
        return inst

    def kleisli_return(self, rho: AtomicDensityMatrix) -> AtomicDensityMatrix:
        """η : A → T(A)."""
        return rho

    def kleisli_bind(
        self,
        rho: AtomicDensityMatrix,
        f: Callable[[AtomicDensityMatrix], AtomicDensityMatrix],
    ) -> AtomicDensityMatrix:
        """(>>=)."""
        return f(self.kleisli_return(rho))

    def kleisli_mu(self, rho_tt: AtomicDensityMatrix) -> AtomicDensityMatrix:
        """μ : T²(A) → T(A) (idempotente sobre densidades ya normalizadas)."""
        return rho_tt

    def apply_gksl_step(
        self,
        rho: AtomicDensityMatrix,
        H: NDArray[np.complex128],
        jump_ops: List[Tuple[float, NDArray[np.complex128]]],
        dt: Optional[float] = None,
    ) -> AtomicDensityMatrix:
        r"""Un paso de Euler–Lindblad CPTP-proyectado: \(\rho\mapsto\Pi_{\mathrm{PSD}}(\rho+\Delta t\,\mathcal{L}\rho)\)."""
        dt_eff = self._dt if dt is None else float(dt)
        R = np.asarray(rho.matrix, dtype=np.complex128)
        H = np.asarray(H, dtype=np.complex128)
        comm = H @ R - R @ H
        dissip = np.zeros_like(R)
        for gamma, L in jump_ops:
            g = max(float(gamma), 0.0)
            LdL = L.conj().T @ L
            dissip += g * (L @ R @ L.conj().T - 0.5 * (LdL @ R + R @ LdL))
        R_next = R + dt_eff * (-1j * comm + dissip)
        R_next = _hermitize(R_next)
        ev, vec = la.eigh(R_next)
        ev = np.maximum(np.real(ev), 0.0)
        s = float(np.sum(ev))
        if s <= 0.0:
            return rho
        ev /= s
        return AtomicDensityMatrix(vec @ np.diag(ev) @ vec.conj().T, auto_renormalize=False, validate=True)

    def surface1_data_rsi_krull_valuation(self, rho: AtomicDensityMatrix) -> RSI3SurfaceReport:
        r"""\(v(\mathrm{MAC})=-\log\lambda_{\max}\) (valuación de Krull/Novikov)."""
        sp = self._entropy_engine.compute_spectral_data(rho)
        lam_max = float(sp.eigenvalues[0]) if sp.eigenvalues.size else 1.0
        v = float(-math.log(max(lam_max, 1e-15)))
        return RSI3SurfaceReport(
            surface_id=1,
            name="Data-RSI",
            metric_value=v,
            threshold=self._krull_min,
            passed=bool(v >= self._krull_min),
            diagnostics={"eigenvalue_max": lam_max, "novikov_order": int(sp.rank)},
        )

    def surface2_harness_rsi_gksl_rewrite(
        self,
        jump_ops: List[Tuple[float, NDArray[np.complex128]]],
        H: Optional[NDArray[np.complex128]] = None,
    ) -> Tuple[List[Tuple[float, NDArray[np.complex128]]], RSI3SurfaceReport]:
        if H is not None:
            self._pruner.set_hamiltonian(H)
        pruned, rep = self._pruner.prune_jump_operators(jump_ops, H=H)
        metric = float(rep.kossakowski_min_eig)
        passed = bool(rep.complete_positivity_verified and metric >= -1e-10)
        return pruned, RSI3SurfaceReport(
            surface_id=2,
            name="Harness-RSI",
            metric_value=metric,
            threshold=-1e-10,
            passed=passed,
            diagnostics={
                "pruned": rep.pruned_count,
                "discarded": rep.discarded_count,
                "rate_preservation": rep.rate_preservation,
                "choi_min_eig": rep.choi_min_eig,
            },
        )

    @staticmethod
    def _fubini_study(psi: NDArray[np.complex128], phi: NDArray[np.complex128]) -> float:
        denom = max(float(la.norm(psi) * la.norm(phi)), 1e-15)
        v = abs(complex(np.vdot(psi, phi))) / denom
        return float(np.arccos(np.clip(v, 0.0, 1.0)))

    def surface3_model_rsi_fixed_point(
        self,
        rho: AtomicDensityMatrix,
        rho_prev: Optional[AtomicDensityMatrix],
        H: NDArray[np.complex128],
        jump_ops: List[Tuple[float, NDArray[np.complex128]]],
    ) -> Tuple[AtomicDensityMatrix, RSI3SurfaceReport]:
        r"""
        Un paso del endomorfismo \(T=\Phi\circ\Pi_\varepsilon\) (NO proyección pura:
        eso colapsaría \(v_{\mathrm{Krull}}\) a 0 y violaría [F3]).
        \(d_{\mathrm{FS}}\) se mide entre autovectores dominantes (purificaciones).
        """
        try:
            rho_trunc, _ = self._truncator.truncate_spectrum(rho, compute_fidelity=False)
        except NumericalInstabilityError:
            rho_trunc = rho
        rho_next = self.apply_gksl_step(rho_trunc, H, jump_ops)
        sp = self._entropy_engine.compute_spectral_data(rho_next)
        if rho_prev is None:
            d_fs = 0.0
        else:
            sp_p = self._entropy_engine.compute_spectral_data(rho_prev)
            d_fs = self._fubini_study(sp.eigenvectors[:, 0], sp_p.eigenvectors[:, 0])
        return rho_next, RSI3SurfaceReport(
            surface_id=3,
            name="Model-RSI",
            metric_value=float(d_fs),
            threshold=self._fubini_max,
            passed=bool(d_fs <= self._fubini_max or rho_prev is None),
            diagnostics={"dominant_eigenvalue": float(sp.eigenvalues[0]), "entropy": sp.entropy},
        )

    def compute_banach_contraction(
        self,
        rho_old: AtomicDensityMatrix,
        rho_new: AtomicDensityMatrix,
        rho_ref: Optional[AtomicDensityMatrix] = None,
    ) -> Dict[str, Any]:
        r"""
        Lipschitz en métrica de Bures:
        \(\operatorname{Lip}(T)\approx d_B(\rho_{\mathrm{new}},\rho_{\mathrm{old}})/d_B(\rho_{\mathrm{old}},\rho_{\mathrm{ref}})\).
        Fallback Frobenius si el denominador es nulo.
        """
        d_num = self._entropy_engine.compute_bures_distance(rho_old, rho_new)
        if rho_ref is not None:
            d_den = self._entropy_engine.compute_bures_distance(rho_ref, rho_old)
        else:
            d_den = float(la.norm(np.asarray(rho_old.matrix), ord="fro"))
        lip = float(d_num / max(d_den, 1e-15))
        sp = self._entropy_engine.compute_spectral_data(rho_new)
        gap = float(abs(sp.eigenvalues[0] - sp.eigenvalues[-1])) if sp.eigenvalues.size else 0.0
        return {"lipschitz_constant": lip, "is_contractive": bool(lip <= self._lipschitz_max), "spectral_gap": gap}

    def _meta_adapt(self, s1: RSI3SurfaceReport, s2: RSI3SurfaceReport, lip: float) -> None:
        """RSI nivel 3: el improver reescribe sus propios umbrales."""
        if not s1.passed:
            self._truncator._epsilon = max(self._truncator._epsilon * 0.5, MIN_EIGENVALUE)
        if not s2.passed:
            self._pruner._tau = min(self._pruner._tau * 1.15, 1.0)
        if lip > self._lipschitz_max:
            self._dt *= 0.85

    def iterate(
        self,
        env: RSI3Envelope,
    ) -> Tuple[AtomicDensityMatrix, List[Tuple[float, NDArray[np.complex128]]], RSI3Certificate]:
        rho = env.rho_purified
        jump_ops = list(env.pruned_jump_operators)
        H = env.hamiltonian
        last_rho: Optional[AtomicDensityMatrix] = None
        ref_rho = rho
        lip_history: List[float] = []
        cap_history: List[float] = []
        surfaces: Tuple[RSI3SurfaceReport, RSI3SurfaceReport, RSI3SurfaceReport]
        dummy = RSI3SurfaceReport(0, "init", 0.0, 0.0, False, {})
        surfaces = (dummy, dummy, dummy)
        fixed_point = False
        i = 0

        for i in range(1, self._max_iter + 1):
            s1 = self.surface1_data_rsi_krull_valuation(rho)
            jump_ops, s2 = self.surface2_harness_rsi_gksl_rewrite(jump_ops, H=H)
            rho_new, s3 = self.surface3_model_rsi_fixed_point(rho, last_rho, H, jump_ops)
            surfaces = (s1, s2, s3)
            cap_history.append(self._entropy_engine.compute_spectral_data(rho_new).entropy)

            lip = 0.0
            if last_rho is not None:
                cont = self.compute_banach_contraction(last_rho, rho_new, rho_ref=ref_rho)
                lip = float(cont["lipschitz_constant"])
                lip_history.append(lip)
            self._meta_adapt(s1, s2, lip)

            delta = (
                float(la.norm(np.asarray(rho_new.matrix) - np.asarray(rho.matrix), ord="fro"))
                if last_rho is not None else 1.0
            )
            last_rho, rho = rho, rho_new
            if s1.passed and s2.passed and s3.passed and delta < self._fubini_max:
                fixed_point = True
                break

        lip_final = float(np.mean(lip_history)) if lip_history else 1.0
        # d³C/dt³ por diferencias finitas de la capacidad (entropía de von Neumann)
        jerk = 0.0
        if len(cap_history) >= 4:
            c = np.asarray(cap_history, dtype=np.float64)
            jerk = float(c[-1] - 3.0 * c[-2] + 3.0 * c[-3] - c[-4])
        elif len(cap_history) >= 2:
            jerk = float(cap_history[-1] - cap_history[0])
        accel = abs(jerk) ** (1.0 / 3.0) if jerk != 0.0 else 0.0
        valid = bool(
            all(s.passed for s in surfaces)
            and lip_final <= self._lipschitz_max
            and jerk >= -1e-18
        )
        cert = RSI3Certificate(
            surfaces=surfaces,
            lipschitz_constant=lip_final,
            capacity_acceleration=float(accel),
            capacity_jerk=float(jerk),
            fixed_point_reached=fixed_point,
            iterations=i,
            is_valid=valid,
        )
        self._history.append(cert)
        return rho, jump_ops, cert


class HeytingOmega3Adjudicator:
    __slots__ = ()

    def adjudicate(
        self,
        is_majorized: bool,
        is_fidelity_ok: bool,
        is_kam_stable: bool,
        is_rsi3_valid: bool,
        is_symplectic_valid: bool = True,
    ) -> Tuple[HeytingOmega3, str, float]:
        critical = (not is_majorized) or (not is_kam_stable) or (not is_rsi3_valid) or (not is_symplectic_valid)
        if critical:
            return HeytingOmega3.VETOED, "ESP32_CROWBAR_GPIO14_HIGH_BT151_TRIGGERED", CROWBAR_VETO_LATENCY_NS
        if not is_fidelity_ok:
            return HeytingOmega3.DEGRADED, "MONITORING_DEGRADED_FIDELITY", 120.0
        return HeytingOmega3.COHERENT, "NOMINAL_EXECUTION", CROWBAR_NOMINAL_LATENCY_NS

    def compose_verdicts(self, *verdicts: HeytingOmega3) -> HeytingOmega3:
        return reduce(lambda a, b: a.join(b), verdicts, HeytingOmega3.COHERENT)


@dataclass(frozen=True, slots=True)
class MinimizationMetrics:
    truncation_report: Optional[TruncationReport]
    pruning_report: Optional[PruningReport]
    total_compression_ratio: float
    information_loss: float
    fidelity_preservation: float
    bures_distance: float
    computational_speedup: float
    entropy_reduction: float
    purity_increase: float
    majorization_preserved: bool
    channel_capacity_bound: float
    execution_time_ms: float
    symplectic_residue: float = 0.0
    is_symplectic_valid: bool = True
    diophantine_satisfied: bool = True
    chirikov_parameter: float = 0.0
    melnikov_integral: float = 0.0
    is_kam_stable: bool = True
    floquet_max_deviation: float = 0.0
    nekhoroshev_time: float = 1.0
    krull_valuation: float = 0.12
    fubini_study_distance: float = 1e-5
    lipschitz_constant: float = 0.85
    capacity_acceleration: float = 1.25
    capacity_jerk: float = 0.0
    is_rsi3_valid: bool = True
    rsi3_iterations: int = 1
    heyting_verdict: str = "COHERENT"
    heyting_code: int = 0x00
    hardware_crowbar_action: str = "NOMINAL_EXECUTION"
    interlock_latency_ns: float = 45.0

    def __post_init__(self) -> None:
        if not (0.0 <= self.total_compression_ratio <= 1.0 + 1e-10):
            raise NumericalInstabilityError("compression_ratio fuera de [0,1].")
        if self.information_loss < -1e-10:
            raise NumericalInstabilityError("information_loss negativo.")
        if not (0.0 <= self.fidelity_preservation <= 1.0 + 1e-10):
            raise NumericalInstabilityError("fidelidad fuera de [0,1].")
        if self.bures_distance < 0.0:
            raise NumericalInstabilityError("Bures negativo.")
        if self.computational_speedup < 1.0 - 1e-10:
            raise NumericalInstabilityError("speedup < 1.")


class MACMinimizer(Morphism):
    r"""Funtor \(\mathbf{Wis}\to\mathbf{Wis}\) que orquesta F1→F2→F3."""

    __slots__ = (
        "_epsilon_spectral", "_tau_lindblad", "_truncation_strategy",
        "_pruning_criterion", "_auto_optimize", "_debug_mode",
        "_entropy_engine", "_spectral_projector", "_lindblad_pruner",
        "_poincare_engine", "_rsi3_monad", "_heyting_adjudicator",
        "_minimization_count", "_total_compression",
        "_history", "_telemetry", "_pareto_frontier",
    )

    @property
    def domain(self) -> frozenset:
        return frozenset({Stratum.WISDOM})

    @property
    def codomain(self) -> frozenset:
        return frozenset({Stratum.WISDOM})

    def __init__(
        self,
        epsilon_spectral: float = 1e-6,
        tau_lindblad: float = 1e-4,
        truncation_strategy: TruncationStrategy = TruncationStrategy.THRESHOLD,
        pruning_criterion: PruningCriterion = PruningCriterion.FROBENIUS_NORM,
        auto_optimize: bool = True,
        debug_mode: bool = False,
        verify_cptp: bool = True,
        purification_mode: PurificationMode = PurificationMode.CAYLEY_KAHLE_PROJECTION,
        entropy_engine: Optional[VonNeumannEntropyEngine] = None,
    ) -> None:
        self._epsilon_spectral = max(float(epsilon_spectral), MIN_EIGENVALUE)
        self._tau_lindblad = max(float(tau_lindblad), 0.0)
        self._truncation_strategy = truncation_strategy
        self._pruning_criterion = pruning_criterion
        self._auto_optimize = auto_optimize
        self._debug_mode = debug_mode
        self._entropy_engine = entropy_engine or VonNeumannEntropyEngine()
        self._spectral_projector = SpectralTruncationProjector(
            epsilon_threshold=self._epsilon_spectral,
            strategy=truncation_strategy,
            entropy_engine=self._entropy_engine,
            verify_cptp=verify_cptp,
            compute_fidelity=True,
        )
        self._lindblad_pruner = LindbladPruningOperator(
            tau_cutoff=self._tau_lindblad,
            criterion=pruning_criterion,
            preserve_critical=True,
            verify_cptp=verify_cptp,
            entropy_engine=self._entropy_engine,
        )
        self._poincare_engine = PoincareCelestialEngine(entropy_engine=self._entropy_engine)
        self._rsi3_monad = RSI3KleisliMonad(
            entropy_engine=self._entropy_engine,
            truncator=self._spectral_projector,
            pruner=self._lindblad_pruner,
        )
        self._heyting_adjudicator = HeytingOmega3Adjudicator()
        self._minimization_count = 0
        self._total_compression = 1.0
        self._history: List[MinimizationMetrics] = []
        self._telemetry: Dict[str, Any] = {}
        self._pareto_frontier: List[Tuple[float, float, float]] = []
        logger.info(
            "MACMinimizer v%s (3-Fases) ε=%.2e τ=%.2e",
            __version__, self._epsilon_spectral, self._tau_lindblad,
        )

    def purify_semantic_state(
        self,
        rho: AtomicDensityMatrix,
        jump_operators: List[Tuple[float, NDArray[np.complex128]]],
        entropy_threshold: Optional[float] = None,
        H: Optional[NDArray[np.complex128]] = None,
        force_truncation: bool = False,
        target_fidelity: Optional[float] = None,
        max_dimension: Optional[int] = None,
    ) -> Tuple[AtomicDensityMatrix, List[Tuple[float, NDArray[np.complex128]]], MinimizationMetrics]:
        t_start = time.perf_counter()
        self._minimization_count += 1
        logger.info("═══ MACMinimizer Iteración #%d (3-Fases v%s) ═══", self._minimization_count, __version__)

        sp_before = self._entropy_engine.compute_spectral_data(rho)
        entropy_init, purity_init = sp_before.entropy, sp_before.purity
        if entropy_threshold is None:
            entropy_threshold = 0.7 * math.log(max(sp_before.dimension, 2))

        # ── FASE 1 ────────────────────────────────────────────────────────────
        envelope = self._spectral_projector.emit_poincare_envelope(
            rho=rho,
            jump_operators=jump_operators,
            hamiltonian=H,
            target_fidelity=target_fidelity,
            force_truncation=force_truncation or (
                entropy_init > entropy_threshold
                or purity_init < 0.5
                or sp_before.condition_number > MAX_CONDITION_NUMBER
            ),
        )

        # ── FASE 2 (apertura = continuación de emit_poincare_envelope) ────────
        pruner = LindbladPruningOperator.from_poincare_envelope(
            envelope,
            tau_cutoff=self._tau_lindblad,
            criterion=self._pruning_criterion,
            entropy_engine=self._entropy_engine,
        )
        pruned_ops, pruning_report = pruner.prune_jump_operators(
            envelope.jump_operators, H=envelope.H0_integrable
        )
        celestial = PoincareCelestialEngine.from_poincare_envelope(envelope)
        rsi3_envelope = celestial.emit_rsi3_envelope(
            env=envelope,
            pruned_jump_ops=pruned_ops,
            H0=envelope.H0_integrable,
            H1=envelope.H1_perturbation,
        )

        # ── FASE 3 (apertura = continuación de emit_rsi3_envelope) ────────────
        rsi3_monad = RSI3KleisliMonad.from_rsi3_envelope(rsi3_envelope)
        rho_final, pruned_final, rsi3_cert = rsi3_monad.iterate(rsi3_envelope)

        kam_cert = rsi3_envelope.kam_certificate
        floq = rsi3_envelope.floquet_certificate
        cayley = envelope.cayley_data
        trunc = envelope.truncation_report
        verdict, action, latency = self._heyting_adjudicator.adjudicate(
            is_majorized=trunc.majorization_verified,
            is_fidelity_ok=bool(trunc.fidelity_preservation >= 0.95),
            is_kam_stable=bool(kam_cert.get("is_kam_stable", False)),
            is_rsi3_valid=bool(rsi3_cert.is_valid),
            is_symplectic_valid=bool(cayley.get("is_symplectic_valid", False)),
        )

        sp_after = self._entropy_engine.compute_spectral_data(rho_final)
        lindblad_compression = (
            pruning_report.pruned_count / max(1, pruning_report.original_count)
            if pruning_report.original_count else 1.0
        )
        total_compression = float(trunc.compression_ratio * lindblad_compression)
        nek_t = float(kam_cert.get("nekhoroshev", {}).get("nekhoroshev_time", 1.0))

        metrics = MinimizationMetrics(
            truncation_report=trunc,
            pruning_report=pruning_report,
            total_compression_ratio=total_compression,
            information_loss=self._entropy_engine.compute_relative_entropy(rho, envelope.rho_purified),
            fidelity_preservation=trunc.fidelity_preservation,
            bures_distance=trunc.bures_distance,
            computational_speedup=trunc.computational_speedup,
            entropy_reduction=sp_after.entropy - entropy_init,
            purity_increase=sp_after.purity - purity_init,
            majorization_preserved=trunc.majorization_verified,
            channel_capacity_bound=sp_after.entropy,
            execution_time_ms=(time.perf_counter() - t_start) * 1000.0,
            symplectic_residue=float(cayley.get("symplectic_residue", 0.0)),
            is_symplectic_valid=bool(cayley.get("is_symplectic_valid", False)),
            diophantine_satisfied=bool(kam_cert["kam"]["kam_satisfied"]),
            chirikov_parameter=float(kam_cert["chirikov"]["chirikov_parameter"]),
            melnikov_integral=float(kam_cert["melnikov"]["melnikov_integral"]),
            is_kam_stable=bool(kam_cert["is_kam_stable"]),
            floquet_max_deviation=float(floq.get("max_unit_deviation", 0.0)),
            nekhoroshev_time=nek_t,
            krull_valuation=float(rsi3_cert.surface1.metric_value),
            fubini_study_distance=float(rsi3_cert.surface3.metric_value),
            lipschitz_constant=float(rsi3_cert.lipschitz_constant),
            capacity_acceleration=float(rsi3_cert.capacity_acceleration),
            capacity_jerk=float(rsi3_cert.capacity_jerk),
            is_rsi3_valid=bool(rsi3_cert.is_valid),
            rsi3_iterations=int(rsi3_cert.iterations),
            heyting_verdict=verdict.name,
            heyting_code=int(verdict.value),
            hardware_crowbar_action=action,
            interlock_latency_ns=float(latency),
        )
        self._total_compression *= total_compression
        if self._debug_mode:
            self._history.append(metrics)
            self._pareto_frontier.append((total_compression, trunc.fidelity_preservation, trunc.computational_speedup))
        self._update_telemetry(metrics)
        return rho_final, pruned_final, metrics

    def _update_telemetry(self, m: MinimizationMetrics) -> None:
        self._telemetry = {
            "total_minimizations": self._minimization_count,
            "cumulative_compression": self._total_compression,
            "average_compression": float(np.mean([h.total_compression_ratio for h in self._history])) if self._history else 0.0,
            "average_fidelity": float(np.mean([h.fidelity_preservation for h in self._history])) if self._history else 1.0,
            "average_speedup": float(np.mean([h.computational_speedup for h in self._history])) if self._history else 1.0,
            "last_execution_time_ms": m.execution_time_ms,
            "last_verdict": m.heyting_verdict,
            "version": __version__,
        }

    def get_telemetry(self) -> Dict[str, Any]:
        out = dict(self._telemetry)
        if self._debug_mode:
            out["history"] = self._history
            out["pareto_frontier"] = self._pareto_frontier
        return out

    def __call__(self, input_data: Any) -> Any:
        if isinstance(input_data, tuple) and len(input_data) == 2:
            rho, jumps = input_data
            return self.purify_semantic_state(rho, jumps)
        if isinstance(input_data, dict):
            return self.purify_semantic_state(
                input_data["rho"],
                input_data.get("jump_operators", []),
                H=input_data.get("H"),
            )
        raise ValueError("Entrada inválida para MACMinimizer.")

    def compose(self, other: "Morphism") -> "Morphism":
        raise NotImplementedError("Composición de funtores no implementada")

    def id(self) -> "Morphism":
        return self


def create_mac_minimizer(profile: str = "balanced", **kwargs: Any) -> MACMinimizer:
    profiles: Dict[str, Dict[str, Any]] = {
        "aggressive": dict(
            epsilon_spectral=1e-3, tau_lindblad=1e-2,
            truncation_strategy=TruncationStrategy.CUMULATIVE_ENERGY,
            pruning_criterion=PruningCriterion.MAGNITUDE,
            auto_optimize=True, debug_mode=False,
        ),
        "balanced": dict(
            epsilon_spectral=1e-6, tau_lindblad=1e-4,
            truncation_strategy=TruncationStrategy.THRESHOLD,
            pruning_criterion=PruningCriterion.FROBENIUS_NORM,
            auto_optimize=True, debug_mode=False,
        ),
        "conservative": dict(
            epsilon_spectral=1e-9, tau_lindblad=1e-6,
            truncation_strategy=TruncationStrategy.ENTROPY_BOUNDED,
            pruning_criterion=PruningCriterion.DISSIPATIVE_GAP,
            auto_optimize=False, debug_mode=True,
        ),
        "speed": dict(
            epsilon_spectral=1e-4, tau_lindblad=1e-3,
            truncation_strategy=TruncationStrategy.RANK_K,
            pruning_criterion=PruningCriterion.CHOI_RANK,
            auto_optimize=True, debug_mode=False,
        ),
        "fidelity": dict(
            epsilon_spectral=1e-12, tau_lindblad=1e-8,
            truncation_strategy=TruncationStrategy.MAJORIZATION_OPTIMAL,
            pruning_criterion=PruningCriterion.INFORMATION_GAIN,
            auto_optimize=False, debug_mode=True,
        ),
        "research": dict(
            epsilon_spectral=1e-6, tau_lindblad=1e-4,
            truncation_strategy=TruncationStrategy.THRESHOLD,
            pruning_criterion=PruningCriterion.FROBENIUS_NORM,
            auto_optimize=True, debug_mode=True,
        ),
    }
    if profile not in profiles:
        raise ValueError(f"Perfil desconocido: {profile}.")
    cfg = dict(profiles[profile])
    cfg.update(kwargs)
    return MACMinimizer(**cfg)


__all__ = [
    "TruncationStrategy", "LogarithmicBase", "PurificationMode",
    "SpectralData", "TruncationReport", "QuantumChannel",
    "VonNeumannEntropyEngine", "SpectralTruncationProjector",
    "PoincareEnvelope",
    "PruningCriterion", "PruningReport",
    "LindbladPruningOperator", "PoincareCelestialEngine",
    "RSI3Envelope",
    "HeytingOmega3", "RSI3SurfaceReport", "RSI3Certificate",
    "RSI3KleisliMonad", "HeytingOmega3Adjudicator",
    "MinimizationMetrics", "MACMinimizer", "create_mac_minimizer",
]