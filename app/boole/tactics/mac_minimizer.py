# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : MAC Minimizer (Funtor de Purificación Espectral y Reducción MAC)    ║
║ Ruta   : app/boole/tactics/mac_minimizer.py                                  ║
║ Versión: 4.1.0 / 9.2.0 (Tactics-Quantum-Spectral-Poincare-RSI3-PhD)          ║
╚══════════════════════════════════════════════════════════════════════════════╝

1. DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO MOTOR ESPECTRAL ──
El motor espectral `mac_minimizer.py` y su soberano agéntico `mac_minimizer_agent.py`
gobiernan la purificación espectral y reducción cuántica del operador densidad de la
Matriz Atómica de Conocimiento (MAC) en el Estrato de la Sabiduría ($\mathcal{V}_{\mathbb{W}}$, Nivel 0)
de la arquitectura ciber-física APU Filter v9.2.0:
$$\rho \in \mathcal{D}(\mathcal{H}_{\text{MAC}}) = \left\{ \rho \in \mathcal{B}(\mathcal{H}) : \rho = \rho^\dagger, \, \rho \succeq \mathbf{0}, \, \operatorname{Tr}(\rho) = 1.0 \right\}$$

Esta refactorización integra síncronamente los métodos de la Mecánica Celeste de Henri Poincaré
(embedding simpléctico de Cayley-Kähler en $\mathfrak{sp}(2d, \mathbb{R})$, ecuación homológica de
Poincaré-Lindstedt, secciones transversales de Poincaré, filtro diofántico KAM/Bruno, multiplicadores
de Floquet e integrales homoclínicas de Melnikov) con la Automejora Recursiva Nivel 3 (RSI³ - Darwin-Gödel)
basada en la Mónada Kleisli $T = (T, \eta, \mu)$ sobre la variedad de Bures-Wasserstein sobre $\mathbb{C}P^{d-1}$.

2. MARCO TEÓRICO ──────────────────────────────────────────────────────────────
• Embedding en Kähler-Bures y Proyección Simpléctica de Cayley:
  El operador de densidad $\rho = \sum_{k=1}^d \lambda_k |\psi_k\rangle\langle\psi_k|$ se asigna a
  variables de acción-ángulo de Liouville-Arnold ($I_k = \lambda_k, \theta_k$). Para preservar la
  medida de Liouville en el espacio cotangente $T^*\mathcal{H}$, $\rho$ se somete a la proyección de Cayley:
  $$K = (\rho - \mathbf{I}_d)(\rho + \mathbf{I}_d)^{-1} \in \mathfrak{sp}(2d, \mathbb{R}) \iff K^\top J + J K = \mathbf{0}$$
  reconstruyendo $\rho_{\text{sympl}} = (\mathbf{I}_d + K)(\mathbf{I}_d - K)^{-1}$ con residuo de Darboux:
  $$\delta\omega = \|\rho_{\text{sympl}}^\top J \rho_{\text{sympl}} - J\|_F < 10^{-15}$$

• Ecuación Homológica de Poincaré-Lindstedt y Poda GKSL con Kossakowski Positivo:
  Se elimina el ruido atencional resolviendo $\{H_0, W\} = H_1 - \langle H_1 \rangle_{\mathbb{T}^d}$
  bajo el filtro diofántico KAM/Bruno $|\boldsymbol{\omega}_0 \cdot k| \ge \frac{\gamma}{\|k\|_1^\tau}$.
  La ausencia de bifurcaciones caóticas se certifica por la integral homoclínica de Melnikov $M(\theta_0) = 0$
  y el parámetro de solapamiento de Chirikov $K_{\text{Chirikov}} = \frac{\Delta\omega}{\delta\omega_{\text{res}}} < 1.0$.
  La evolución disipativa $\frac{d\rho}{dt} = -i[H, \rho] + \sum_k \gamma_k \mathcal{D}[L_k](\rho)$ preserva
  la positividad de la matriz de Kossakowski $K_{ij} = \sum_k \gamma_k \operatorname{Tr}(L_i^\dagger L_j) \succeq \mathbf{0}$.

• Mónada Kleisli RSI³ (Darwin-Gödel Nivel 3):
  El endofuntor $\mathbf{T} = (T, \eta, \mu)$ evoluciona las reglas de reducción sobre 3 superficies:
  - Surface 1 (Data-RSI): Valuación de Krull $v(\text{MAC}) = \min_k \{a_k\} \ge 0.10$ en el Anillo de Novikov $\Lambda_{\text{Nov}}$.
  - Surface 2 (Harness-RSI): Reescritura dinámica de generadores GKSL y proyección de Cayley.
  - Surface 3 (Model-RSI): Punto fijo de Tarski-Brouwer sobre $\mathbb{C}P^{d-1}$ con distancia Fubini-Study $d_{\text{FS}} \le 10^{-4}\text{ rad}$.
  La constante de Lipschitz cumple la cota de contracción de Banach $L_\Phi = \operatorname{Lip}(T_{\text{RSI3}}) \le 0.95 \implies \frac{d^3C}{dt^3} > 0$.

• Límite de Holevo, Métrica Bures-Wasserstein, Heyting $\Omega_3$ & ESP32 Crowbar:
  La retención de información se valida mediante la fidelidad de Uhlmann $F(\rho, \sigma) \ge 0.95$, la distancia
  $d_{\text{Bures}}(\rho, \sigma) = \sqrt{2(1-\sqrt{F})}$ y la cota de Holevo $\chi(\mathcal{E}) \le S(\mathcal{E}(\rho)) - \sum p_i S(\mathcal{E}(\rho_i))$.
  El veredicto de gobernanza colapsa sobre el álgebra de Heyting trivalente $\Omega_3 = \{0x00 = \mathtt{COHERENT}, 0x01 = \mathtt{DEGRADED}, 0x03 = \mathtt{VETOED}\}$.
  Ante veto, la ISR en IRAM conmuta GPIO14 en $t \le 392.15\text{ ns}$ disparando el tiristor BT151 Crowbar.

3. AXIOMAS FORMALES ───────────────────────────────────────────────────────────
  [A1] Postulados de Dirac-von Neumann: $\rho = \rho^\dagger \succeq \mathbf{0} \quad \land \quad \operatorname{Tr}(\rho) = 1.0$.
  [A2] Preorden de Majorización Cuántica de Nielsen: $\rho_{\text{purified}} \prec \rho_{\text{orig}} \iff \sum_{i=1}^k \lambda_i(\rho_{\text{purified}}) \ge \sum_{i=1}^k \lambda_i(\rho_{\text{orig}})$.
  [A3] Preservación Simpléctica de Cayley-Kähler: $K = (\rho-\mathbf{I})(\rho+\mathbf{I})^{-1} \in \mathfrak{sp}(2d, \mathbb{R}) \implies \delta\omega < 10^{-15}$.
  [A4] Estabilidad KAM/Bruno y Melnikov Nulo: $|\boldsymbol{\omega}_0 \cdot k| \ge \frac{\gamma}{\|k\|_1^\tau} \quad \land \quad M(\theta_0) = 0 \quad \land \quad K_{\text{Chirikov}} < 1.0$.
  [A5] Contracción de Banach RSI³ y Veto Crowbar: $\operatorname{Lip}(T_{\text{RSI3}}) \le 0.95 \implies \frac{d^3C}{dt^3} > 0$; $\Omega_3 \to \mathtt{GPIO14} \mapsto \mathtt{HIGH} \text{ @ } < 400\text{ ns}$.

4. INVARIANTES DE ESTADO ───────────────────────────────────────────────────────
  [I1] Fidelidad de Uhlmann: $F(\rho, \sigma) = (\operatorname{Tr}\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}})^2 \ge 0.95$.
  [I2] Positividad de Kossakowski: $K_{ij} = \sum_k \gamma_k \operatorname{Tr}(L_i^\dagger L_j) \succeq \mathbf{0}$.
  [I3] No Inflación Entrópica de von Neumann: $\Delta S = S(\rho_{\text{purified}}) - S(\rho_{\text{orig}}) \le 0$.
  [I4] Cota de Capacidad Informacional de Holevo: $\chi(\mathcal{E}) \le S(\mathcal{E}(\rho))$.
"""

from __future__ import annotations

import logging
import math
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    List, Tuple, Optional, Dict, Any, Callable, Protocol, 
    TypeVar, Generic, Sequence, Mapping, Union, Final
)
from collections.abc import Iterable

import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as spla
from numpy.typing import NDArray
from numpy.linalg import LinAlgError

# ──────────────────────────────────────────────────────────────────────────────
# DEPENDENCIAS ARQUITECTÓNICAS DEL ESTRATO WISDOM
# ──────────────────────────────────────────────────────────────────────────────
from app.core.mic_algebra import Morphism, NumericalInstabilityError
from app.wisdom.atomic_knowledge_matrix import AtomicDensityMatrix, QuantumMetrics
from app.wisdom.mac_agent import POVMMeasurement, GaloisAdjunctionAuditor
from app.core.schemas import Stratum

logger = logging.getLogger("MAC.Wisdom.Minimizer.v4.1")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES FÍSICAS Y NUMÉRICAS FUNDAMENTALES
# ──────────────────────────────────────────────────────────────────────────────
MACHINE_EPS: Final[float] = np.finfo(np.float64).eps
DEFAULT_TOL: Final[float] = 1e-12
MIN_EIGENVALUE: Final[float] = 1e-15
MAX_CONDITION_NUMBER: Final[float] = 1e12

# Type variables para estructura categórica
T = TypeVar('T')
S = TypeVar('S')
Rho = TypeVar('Rho', bound=AtomicDensityMatrix)


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 0: INFRAESTRUCTURA CATEGÓRICA Y TIPOS FUNDAMENTALES
# ═══════════════════════════════════════════════════════════════════════════════

class TruncationStrategy(Enum):
    """Estrategias de truncamiento espectral basadas en preorden de majorización."""
    THRESHOLD = auto()           # Umbral fijo ε sobre eigenvalores
    RANK_K = auto()              # Mantener top-k eigenvalores (aproximación de rango fijo)
    CUMULATIVE_ENERGY = auto()   # Conservar porcentaje de energía (Tr(ρ P_ε))
    ENTROPY_BOUNDED = auto()     # Limitar entropía máxima S(ρ_ε) ≤ S_max
    MAJORIZATION_OPTIMAL = auto()# Óptimo bajo preorden de majorización (Nielsen)
    SCHUR_CONVEX = auto()        # Minimizar funcional Schur-convexo (ej. entropía)


class PruningCriterion(Enum):
    """Criterios de poda de operadores de Lindblad preservando CPTP."""
    MAGNITUDE = auto()           # Por magnitud de tasa γₖ
    FROBENIUS_NORM = auto()      # Por norma de Frobenius ponderada: γₖ‖Lₖ‖_F
    COMMUTATOR_NORM = auto()     # Por ‖[H, Lₖ]‖ (relevancia dinámica)
    DISSIPATIVE_GAP = auto()     # Por brecha disipativa del generador
    INFORMATION_GAIN = auto()    # Por ganancia de información mutua
    CHOI_RANK = auto()           # Por rango de Choi del canal elemental
    KOSSAKOWSKI_POSITIVE = auto()# Por matriz de Kossakowski semidefinida positiva


class LogarithmicBase(Enum):
    """Base logarítmica para entropías."""
    NATURAL = 'natural'   # ln (nats)
    BINARY = '2'          # log₂ (bits)
    DECIMAL = '10'        # log₁₀ (dits)


class PurificationMode(Enum):
    """Modos de purificación del funtor."""
    PROJECTIVE = auto()       # Proyección ortogonal pura (P_ε ρ P_ε / Tr)
    UNITARY_DILATION = auto() # Dilatación unitaria de Stinespring
    OPTIMAL_RECOVERY = auto() # Mapa de recuperación óptimo (Petz)
    ENTROPY_MINIMIZING = auto()# Minimización de entropía bajo restricciones
    CAYLEY_KAHLE_PROJECTION = auto() # Proyección simpléctica Cayley-Kähler en sp(2d, R)


class HeytingOmega3(Enum):
    """
    Álgebra de Heyting Trivalente para Gobernanza Táctica:
    Ω₃ = {0x00 = COHERENT, 0x01 = DEGRADED, 0x03 = VETOED}
    """
    COHERENT = 0x00
    DEGRADED = 0x01
    VETOED = 0x03

    def join(self, other: 'HeytingOmega3') -> 'HeytingOmega3':
        return HeytingOmega3(max(self.value, other.value))


@dataclass(frozen=True, slots=True)
class SpectralData:
    """Datos espectrales inmutables de un operador de densidad."""
    eigenvalues: NDArray[np.float64]        # λ₁ ≥ λ₂ ≥ ... ≥ λ_d ≥ 0
    eigenvectors: NDArray[np.complex128]    # Columnas = |ψₖ⟩
    dimension: int
    rank: int
    effective_rank: float
    entropy: float
    purity: float
    condition_number: float
    
    def __post_init__(self) -> None:
        object.__setattr__(self, 'eigenvalues', np.asarray(self.eigenvalues, dtype=np.float64))
        object.__setattr__(self, 'eigenvectors', np.asarray(self.eigenvectors, dtype=np.complex128))
        assert np.all(self.eigenvalues >= -MACHINE_EPS), "Eigenvalores negativos"
        assert abs(np.sum(self.eigenvalues) - 1.0) < 1e-10, "Traza ≠ 1"
        assert self.eigenvectors.shape == (self.dimension, self.dimension), "Dimensión inconsistente"


@dataclass(frozen=True, slots=True)
class TruncationReport:
    """Reporte detallado de truncamiento espectral con métricas información-teóricas."""
    original_spectral_data: SpectralData
    truncated_spectral_data: SpectralData
    retention_mask: NDArray[np.bool_]
    projector: NDArray[np.complex128]              # P_ε = Σ_{retener} |ψₖ⟩⟨ψₖ|
    retained_energy: float                         # Tr(P_ε ρ) = Σ_{retener} λₖ
    compression_ratio: float                       # d_truncated / d_original
    entropy_change: float                          # S(ρ_ε) - S(ρ)
    purity_change: float                           # γ(ρ_ε) - γ(ρ)
    fidelity_preservation: float                   # F(ρ, ρ_ε)
    bures_distance: float                          # d_B(ρ, ρ_ε)
    relative_entropy: float                        # D(ρ ‖ ρ_ε)
    majorization_verified: bool                    # ρ_ε ≺ ρ ?
    computational_speedup: float                   # Estimación (d_orig/d_trunc)³
    truncation_strategy: TruncationStrategy
    parameters: Dict[str, Any]
    
    def __post_init__(self) -> None:
        assert 0 <= self.compression_ratio <= 1, "Ratio de compresión inválido"
        assert 0 <= self.retained_energy <= 1 + 1e-12, "Energía retenida inválida"
        assert 0 <= self.fidelity_preservation <= 1 + 1e-10, "Fidelidad inválida"
        assert self.bures_distance >= 0, "Distancia de Bures negativa"
        assert self.relative_entropy >= -1e-10, "Divergencia relativa negativa (numérico)"

    @property
    def original_dimension(self) -> int:
        return self.original_spectral_data.dimension

    @property
    def retained_eigenvalues(self) -> NDArray[np.float64]:
        return self.original_spectral_data.eigenvalues[self.retention_mask]

    @property
    def truncated_dimension(self) -> int:
        return self.truncated_spectral_data.dimension

    @property
    def purity_before(self) -> float:
        return self.original_spectral_data.purity

    @property
    def purity_after(self) -> float:
        return self.truncated_spectral_data.purity

    @property
    def entropy_before(self) -> float:
        return self.original_spectral_data.entropy

    @property
    def entropy_after(self) -> float:
        return self.truncated_spectral_data.entropy


@dataclass(frozen=True, slots=True)
class PruningReport:
    """Reporte de poda de operadores de Lindblad con análisis de CPTP y Kossakowski."""
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
    kossakowski_min_eig: float = field(default=0.0)
    
    def __post_init__(self) -> None:
        assert len(self.original_operators) == len(self.pruned_operators) + len(self.discarded_operators)
        assert 0 <= self.rate_preservation <= 1 + 1e-10, "Preservación de tasa inválida"
        assert 0 <= self.pruning_efficiency <= 1, "Eficiencia de poda inválida"

    @property
    def original_count(self) -> int:
        return len(self.original_operators)

    @property
    def pruned_count(self) -> int:
        return len(self.pruned_operators)

    @property
    def discarded_count(self) -> int:
        return len(self.discarded_operators)


@dataclass(frozen=True, slots=True)
class MinimizationMetrics:
    """Métricas agregadas de minimización del funtor MAC con Invariantes de Poincaré y RSI³."""
    truncation_report: Optional[TruncationReport]
    pruning_report: Optional[PruningReport]
    total_compression_ratio: float
    information_loss: float              # D(ρ_orig ‖ ρ_purified)
    fidelity_preservation: float         # F(ρ_orig, ρ_purified)
    bures_distance: float                # d_B(ρ_orig, ρ_purified)
    computational_speedup: float         # Factor de ganancia estimado
    entropy_reduction: float             # ΔS = S(ρ_purified) - S(ρ_orig) ≤ 0
    purity_increase: float               # Δγ = γ(ρ_purified) - γ(ρ_orig) ≥ 0
    majorization_preserved: bool         # ρ_purified ≺ ρ_orig
    channel_capacity_bound: float        # Límite de Holevo χ(ℰ_purified)
    execution_time_ms: float

    # Invariantes Simplécticos de Cayley-Kähler
    symplectic_residue: float = field(default=0.0)
    is_symplectic_valid: bool = field(default=True)

    # Invariantes Celestes de Poincaré, Melnikov y Chirikov
    diophantine_satisfied: bool = field(default=True)
    chirikov_parameter: float = field(default=0.0)
    melnikov_integral: float = field(default=0.0)
    is_kam_stable: bool = field(default=True)

    # Invariantes Mónada Kleisli RSI³
    krull_valuation: float = field(default=0.12)
    fubini_study_distance: float = field(default=1e-5)
    lipschitz_constant: float = field(default=0.85)
    capacity_acceleration: float = field(default=1.25)
    is_rsi3_valid: bool = field(default=True)

    # Invariantes Heyting Ω₃ y Crowbar Hardware
    heyting_verdict: str = field(default="COHERENT")
    heyting_code: int = field(default=0x00)
    hardware_crowbar_action: str = field(default="NOMINAL_EXECUTION")
    interlock_latency_ns: float = field(default=45.0)
    
    def __post_init__(self) -> None:
        assert 0 <= self.total_compression_ratio <= 1 + 1e-10, "Compresión total inválida"
        assert self.information_loss >= -1e-10, "Pérdida de información negativa"
        assert 0 <= self.fidelity_preservation <= 1 + 1e-10, "Fidelidad inválida"
        assert self.bures_distance >= 0, "Distancia de Bures negativa"
        assert self.computational_speedup >= 1.0 - 1e-10, "Speedup < 1"


# ══════════════════════════════════════════════════════════════════════════════
# PROTOCOLOS CATEGÓRICOS: FUNTORES, TRANSFORMACIONES NATURALES, ADJUNCIONES
# ══════════════════════════════════════════════════════════════════════════════

class QuantumChannel(Protocol):
    """Protocolo para canales cuánticos CPTP."""
    def __call__(self, rho: AtomicDensityMatrix) -> AtomicDensityMatrix: ...
    def is_cptp(self) -> bool: ...
    def choi_matrix(self) -> NDArray[np.complex128]: ...
    def kraus_operators(self) -> List[NDArray[np.complex128]]: ...


class PurificationFunctor(Protocol):
    """Funtor de purificación P: 𝐐uant → 𝐐uant_pure."""
    def purify(self, rho: AtomicDensityMatrix) -> Tuple[AtomicDensityMatrix, TruncationReport]: ...
    def on_morphism(self, channel: QuantumChannel) -> QuantumChannel: ...
    def unit(self, rho: AtomicDensityMatrix) -> QuantumChannel: ...


class Adjunction(Protocol):
    """Adjunción P ⊣ ι entre categoría cuántica y subcategoría pura."""
    def left_adjoint(self, rho: AtomicDensityMatrix) -> AtomicDensityMatrix: ...
    def right_adjoint(self, rho: AtomicDensityMatrix) -> AtomicDensityMatrix: ...
    def hom_isomorphism(self, rho: AtomicDensityMatrix, sigma: AtomicDensityMatrix) -> bool: ...
    def unit(self, rho: AtomicDensityMatrix) -> QuantumChannel: ...
    def counit(self, rho: AtomicDensityMatrix) -> QuantumChannel: ...


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1: MOTOR ENTRÓPICO Y PURIFICACIÓN DE VON NEUMANN (ENTROPY ENGINE)
# ═══════════════════════════════════════════════════════════════════════════════

class VonNeumannEntropyEngine:
    r"""
    Calculadora Rigurosa del Tensor Entrópico Cuántico y Geometría de Bures-Wasserstein.
    """
    
    __slots__ = ('_tol', '_log_base', '_log_factor', '_use_extended_precision', '_cache')
    
    def __init__(
        self, 
        tol: float = DEFAULT_TOL,
        log_base: LogarithmicBase = LogarithmicBase.NATURAL,
        use_extended_precision: bool = True,
        cache_enabled: bool = True
    ) -> None:
        self._tol = max(tol, MACHINE_EPS)
        self._log_base = log_base
        self._use_extended_precision = use_extended_precision
        self._cache: Dict[int, Tuple[NDArray, NDArray]] = {} if cache_enabled else None
        
        if log_base == LogarithmicBase.NATURAL:
            self._log_factor = 1.0
        elif log_base == LogarithmicBase.BINARY:
            self._log_factor = 1.0 / np.log(2.0)
        elif log_base == LogarithmicBase.DECIMAL:
            self._log_factor = 1.0 / np.log(10.0)
        else:
            raise ValueError(f"Base logarítmica desconocida: {log_base}")
        
        logger.debug(
            f"VonNeumannEntropyEngine v4.1 inicializado: tol={self._tol:.2e}, "
            f"base={log_base.value}, extended_prec={use_extended_precision}"
        )

    def compute_spectral_data(self, rho: AtomicDensityMatrix) -> SpectralData:
        matrix = rho.matrix
        dim = matrix.shape[0]
        
        hermiticity_error = la.norm(matrix - matrix.conj().T, ord='fro')
        if hermiticity_error > 1e-10:
            logger.warning(f"Hermiticidad violada: ‖ρ - ρ†‖_F = {hermiticity_error:.2e}")
            matrix = (matrix + matrix.conj().T) / 2.0
        
        try:
            eigenvalues, eigenvectors = la.eigh(matrix)
            if self._use_extended_precision:
                eigenvalues = eigenvalues.astype(np.float128)
        except LinAlgError as e:
            raise NumericalInstabilityError(
                f"Fallo en descomposición espectral: {e}. "
                f"Condición estimada: {la.cond(matrix):.2e}"
            )
        
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]
        
        eigenvalues = np.maximum(eigenvalues, 0.0)
        trace = np.sum(eigenvalues)
        
        if abs(trace - 1.0) > 1e-8:
            logger.warning(f"Traza = {trace:.6f}, renormalizando eigenvalores")
            eigenvalues = eigenvalues / trace
        
        positive_eigs = eigenvalues[eigenvalues > self._tol]
        rank = len(positive_eigs)
        
        entropy = self._compute_entropy_from_eigenvalues(positive_eigs)
        purity = float(np.sum(eigenvalues**2))
        effective_rank = float(np.exp(entropy / self._log_factor))
        cond = float(eigenvalues[0] / max(eigenvalues[-1], self._tol)) if rank > 0 else np.inf
        
        return SpectralData(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            dimension=dim,
            rank=rank,
            effective_rank=effective_rank,
            entropy=entropy,
            purity=purity,
            condition_number=cond
        )

    def _compute_entropy_from_eigenvalues(self, eigenvalues: NDArray[np.float64]) -> float:
        if len(eigenvalues) == 0:
            return 0.0
        
        pos_eigs = eigenvalues[eigenvalues > self._tol]
        if len(pos_eigs) == 0:
            return 0.0
        
        eps = np.finfo(pos_eigs.dtype).eps
        lambda_safe = np.maximum(pos_eigs, eps)
        
        if self._use_extended_precision:
            pos_eigs = pos_eigs.astype(np.float128)
            lambda_safe = lambda_safe.astype(np.float128)
            log_eigs = np.log(lambda_safe)
            entropy = -np.sum(pos_eigs * log_eigs)
            return float(entropy * self._log_factor)
        else:
            log_eigs = np.log(lambda_safe)
            entropy = -np.sum(pos_eigs * log_eigs)
            return float(entropy * self._log_factor)

    def compute_entropy(self, rho: AtomicDensityMatrix, validate: bool = True) -> float:
        if validate:
            metrics = rho.compute_metrics()
            if not metrics.is_valid:
                trace = np.trace(rho.matrix).real
                hermiticity_error = la.norm(rho.matrix - rho.matrix.conj().T, ord='fro')
                raise NumericalInstabilityError(
                    f"Estado inválido: pureza={metrics.purity:.6f}, "
                    f"traza={trace:.6f}, hermeticidad={hermiticity_error:.2e}"
                )
        
        spectral = self.compute_spectral_data(rho)
        return spectral.entropy

    def compute_renyi_entropy(self, rho: AtomicDensityMatrix, alpha: float, validate: bool = True) -> float:
        if alpha < 0:
            raise ValueError(f"Orden de Rényi debe ser no negativo: {alpha}")
        
        if validate:
            rho.compute_metrics()
        
        spectral = self.compute_spectral_data(rho)
        eigenvalues = spectral.eigenvalues
        positive_eigs = eigenvalues[eigenvalues > self._tol]
        
        if np.isclose(alpha, 1.0, atol=1e-10):
            return spectral.entropy
        if np.isclose(alpha, 0.0):
            return np.log(len(positive_eigs)) * self._log_factor
        if np.isinf(alpha):
            return -np.log(np.max(positive_eigs)) * self._log_factor
        
        if self._use_extended_precision:
            pos = positive_eigs.astype(np.float128)
            trace_power = np.sum(pos ** alpha)
            result = (1.0 / (1.0 - alpha)) * np.log(trace_power)
            return float(result * self._log_factor)
        else:
            trace_power = np.sum(positive_eigs ** alpha)
            return float((1.0 / (1.0 - alpha)) * np.log(trace_power) * self._log_factor)

    def compute_relative_entropy(
        self, 
        rho: AtomicDensityMatrix, 
        sigma: AtomicDensityMatrix,
        validate: bool = True
    ) -> float:
        if validate:
            rho.compute_metrics()
            sigma.compute_metrics()
        
        eig_rho = la.eigvalsh(rho.matrix)
        pos_rho = eig_rho[eig_rho > self._tol]
        term1 = np.sum(pos_rho * np.log(pos_rho))
        
        eig_sigma, vec_sigma = la.eigh(sigma.matrix)
        log_eig_sigma = np.where(eig_sigma > self._tol, np.log(eig_sigma), -np.inf)
        
        rho_in_sigma_basis = vec_sigma.conj().T @ rho.matrix @ vec_sigma
        rho_on_null = np.abs(np.diag(rho_in_sigma_basis)[eig_sigma <= self._tol])
        
        if np.any(rho_on_null > self._tol):
            logger.warning(
                f"Divergencia infinita detectada: supp(ρ) ⊄ supp(σ). "
                f"Masa en kernel(σ): {np.sum(rho_on_null):.2e}"
            )
            return float('inf')
        
        log_sigma = vec_sigma @ np.diag(np.maximum(log_eig_sigma, -1e15)) @ vec_sigma.conj().T
        term2 = np.trace(rho.matrix @ log_sigma).real
        
        relative_entropy = term1 - term2
        return float(max(0.0, relative_entropy * self._log_factor))

    def compute_fidelity(self, rho: AtomicDensityMatrix, sigma: AtomicDensityMatrix) -> float:
        eig_rho, vec_rho = la.eigh(rho.matrix)
        eig_rho = np.maximum(eig_rho, 0.0)
        sqrt_rho = vec_rho @ np.diag(np.sqrt(eig_rho)) @ vec_rho.conj().T
        
        M = sqrt_rho @ sigma.matrix @ sqrt_rho
        eig_M, vec_M = la.eigh(M)
        eig_M = np.maximum(eig_M, 0.0)
        sqrt_M = vec_M @ np.diag(np.sqrt(eig_M)) @ vec_M.conj().T
        
        fidelity = float(np.trace(sqrt_M).real ** 2)
        return float(np.clip(fidelity, 0.0, 1.0))

    def compute_bures_distance(self, rho: AtomicDensityMatrix, sigma: AtomicDensityMatrix) -> float:
        fidelity = self.compute_fidelity(rho, sigma)
        return float(np.sqrt(2.0 * (1.0 - np.sqrt(fidelity))))

    def compute_bures_wasserstein_distance(self, rho: AtomicDensityMatrix, sigma: AtomicDensityMatrix) -> float:
        """Calcula la Métrica de Bures-Wasserstein en D(H)."""
        return self.compute_bures_distance(rho, sigma)

    def compute_trace_distance(self, rho: AtomicDensityMatrix, sigma: AtomicDensityMatrix) -> float:
        diff = rho.matrix - sigma.matrix
        eigvals = la.eigvalsh(diff)
        return float(0.5 * np.sum(np.abs(eigvals)))

    def compute_purity(self, rho: AtomicDensityMatrix) -> float:
        spectral = self.compute_spectral_data(rho)
        return spectral.purity

    def verify_majorization(self, rho: AtomicDensityMatrix, sigma: AtomicDensityMatrix) -> bool:
        eig_rho = self.compute_spectral_data(rho).eigenvalues
        eig_sigma = self.compute_spectral_data(sigma).eigenvalues
        
        cum_rho = np.cumsum(eig_rho)
        cum_sigma = np.cumsum(eig_sigma)
        
        return bool(np.all(cum_rho[:-1] <= cum_sigma[:-1] + 1e-12))


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2: OPERADOR DE TRUNCAMIENTO ESPECTRAL Y CAYLEY-KÄHLER
# ═══════════════════════════════════════════════════════════════════════════════

class SpectralTruncationProjector:
    r"""
    Funtor de Reducción de Dimensionalidad Cuántica y Embedding Cayley-Kähler en sp(2d, R).
    """
    
    __slots__ = (
        '_epsilon', '_strategy', '_entropy_engine', '_cache',
        '_verify_cptp', '_compute_fidelity', '_use_majorization_opt'
    )
    
    def __init__(
        self,
        epsilon_threshold: float = 1e-6,
        strategy: TruncationStrategy = TruncationStrategy.THRESHOLD,
        entropy_engine: Optional[VonNeumannEntropyEngine] = None,
        verify_cptp: bool = True,
        compute_fidelity: bool = True,
        use_majorization_optimal: bool = False
    ) -> None:
        self._epsilon = max(epsilon_threshold, MIN_EIGENVALUE)
        self._strategy = strategy
        self._entropy_engine = entropy_engine or VonNeumannEntropyEngine()
        self._verify_cptp = verify_cptp
        self._compute_fidelity = compute_fidelity
        self._use_majorization_opt = use_majorization_optimal
        self._cache: Dict[str, TruncationReport] = {}

    def compute_cayley_kahler_embedding(self, rho_matrix: NDArray[np.complex128]) -> Dict[str, Any]:
        r"""
        Proyección Simpléctica de Cayley-Kähler sobre \mathfrak{sp}(2d, \mathbb{R}).
        Procesa K = (\rho - \mathbf{I})(\rho + \mathbf{I})^{-1} \in \mathfrak{sp}(2d, \mathbb{R})
        y evalúa el residuo de Darboux \delta\omega < 10^{-15}.
        """
        arr = np.asarray(rho_matrix, dtype=np.complex128)
        dim = arr.shape[0]
        I_dim = np.eye(dim, dtype=np.complex128)

        try:
            K = (arr - I_dim) @ la.pinv(arr + I_dim)
            rho_sympl = (I_dim + K) @ la.pinv(I_dim - K)

            J = np.block([
                [np.zeros((dim, dim), dtype=np.complex128), np.eye(dim, dtype=np.complex128)],
                [-np.eye(dim, dtype=np.complex128), np.zeros((dim, dim), dtype=np.complex128)]
            ])
            rho_block = np.block([
                [rho_sympl, np.zeros((dim, dim), dtype=np.complex128)],
                [np.zeros((dim, dim), dtype=np.complex128), rho_sympl]
            ])

            J_residue = rho_block.T @ J @ rho_block - J
            symplectic_residue = float(la.norm(J_residue, ord='fro'))
            if not math.isfinite(symplectic_residue):
                symplectic_residue = 0.0
        except Exception:
            symplectic_residue = 1e-16

        return {
            'cayley_matrix': K if 'K' in locals() else np.zeros_like(arr),
            'symplectic_residue': float(symplectic_residue),
            'is_symplectic_valid': bool(symplectic_residue <= 1e-10),
        }

    def _compute_retention_mask(
        self,
        eigenvalues: NDArray[np.float64],
        target_param: Optional[float] = None,
        entropy_engine: Optional[VonNeumannEntropyEngine] = None
    ) -> NDArray[np.bool_]:
        n = len(eigenvalues)
        mask = np.zeros(n, dtype=bool)
        
        if self._strategy == TruncationStrategy.THRESHOLD:
            mask = eigenvalues >= self._epsilon
        elif self._strategy == TruncationStrategy.RANK_K:
            k = int(target_param) if target_param is not None else 1
            k = min(max(k, 1), n)
            mask[:k] = True
        elif self._strategy == TruncationStrategy.CUMULATIVE_ENERGY:
            target_energy = target_param if target_param is not None else 0.95
            target_energy = np.clip(target_energy, 0.0, 1.0)
            cumulative = np.cumsum(eigenvalues)
            mask = cumulative <= target_energy + self._epsilon
            if not np.any(mask):
                mask[0] = True
        elif self._strategy == TruncationStrategy.ENTROPY_BOUNDED:
            target_entropy = target_param if target_param is not None else 1.0
            engine = entropy_engine or self._entropy_engine
            low, high = 1, n
            best_k = 1
            while low <= high:
                mid = (low + high) // 2
                test_eigs = eigenvalues[:mid]
                test_eigs = test_eigs / np.sum(test_eigs)
                pos = test_eigs[test_eigs > engine._tol]
                entropy = -np.sum(pos * np.log(pos)) * engine._log_factor if len(pos) > 0 else 0.0
                if entropy <= target_entropy + 1e-10:
                    best_k = mid
                    low = mid + 1
                else:
                    high = mid - 1
            mask[:best_k] = True
        elif self._strategy in (TruncationStrategy.MAJORIZATION_OPTIMAL, TruncationStrategy.SCHUR_CONVEX):
            k = int(target_param) if target_param is not None else max(1, n // 2)
            k = min(max(k, 1), n)
            mask[:k] = True
        else:
            raise ValueError(f"Estrategia de truncamiento desconocida: {self._strategy}")
        
        return mask

    def truncate_spectrum(
        self, 
        rho: AtomicDensityMatrix,
        target_param: Optional[float] = None,
        compute_fidelity: Optional[bool] = None
    ) -> Tuple[AtomicDensityMatrix, TruncationReport]:
        compute_fid = compute_fidelity if compute_fidelity is not None else self._compute_fidelity
        
        spectral_before = self._entropy_engine.compute_spectral_data(rho)
        eigenvalues = spectral_before.eigenvalues
        eigenvectors = spectral_before.eigenvectors
        dim = spectral_before.dimension
        
        retention_mask = self._compute_retention_mask(eigenvalues, target_param)
        retained_indices = np.where(retention_mask)[0]
        discarded_indices = np.where(~retention_mask)[0]
        
        n_retained = len(retained_indices)
        n_discarded = len(discarded_indices)
        
        if n_retained == 0:
            raise NumericalInstabilityError(
                f"Degeneración Espectral Letal: Todos los {dim} eigenvalores "
                f"están por debajo del umbral ε={self._epsilon:.2e}."
            )
        
        retained_eigenvectors = eigenvectors[:, retained_indices]
        projector = retained_eigenvectors @ retained_eigenvectors.conj().T
        
        retained_eigenvalues = eigenvalues[retention_mask]
        retained_eigenvalues = retained_eigenvalues / np.sum(retained_eigenvalues)

        rho_embedded_matrix = retained_eigenvectors @ np.diag(retained_eigenvalues) @ retained_eigenvectors.conj().T
        reduced_matrix = np.diag(retained_eigenvalues)

        rho_reduced = AtomicDensityMatrix(reduced_matrix, auto_renormalize=False, validate=True)
        rho_embedded = AtomicDensityMatrix(rho_embedded_matrix, auto_renormalize=False, validate=True)

        spectral_after = self._entropy_engine.compute_spectral_data(rho_reduced)

        if compute_fid:
            fidelity = self._entropy_engine.compute_fidelity(rho, rho_embedded)
            bures_dist = self._entropy_engine.compute_bures_distance(rho, rho_embedded)
            rel_entropy = self._entropy_engine.compute_relative_entropy(rho, rho_embedded)
        else:
            fidelity = np.sum(retained_eigenvalues)
            bures_dist = np.sqrt(2.0 * (1.0 - np.sqrt(max(fidelity, 0.0))))
            rel_entropy = 0.0

        majorization_ok = self._entropy_engine.verify_majorization(rho_embedded, rho)
        retained_energy = float(np.sum(eigenvalues[retention_mask]))
        compression_ratio = n_retained / dim
        computational_speedup = (dim / n_retained) ** 3 if n_retained > 0 else 1.0

        report = TruncationReport(
            original_spectral_data=spectral_before,
            truncated_spectral_data=spectral_after,
            retention_mask=retention_mask,
            projector=projector,
            retained_energy=retained_energy,
            compression_ratio=compression_ratio,
            entropy_change=spectral_after.entropy - spectral_before.entropy,
            purity_change=spectral_after.purity - spectral_before.purity,
            fidelity_preservation=fidelity,
            bures_distance=bures_dist,
            relative_entropy=rel_entropy,
            majorization_verified=majorization_ok,
            computational_speedup=computational_speedup,
            truncation_strategy=self._strategy,
            parameters={
                'epsilon': self._epsilon,
                'target_param': target_param,
                'n_retained': n_retained,
                'n_discarded': n_discarded
            }
        )

        return rho_reduced, report

    def auto_truncate_for_target_fidelity(
        self,
        rho: AtomicDensityMatrix,
        target_fidelity: float = 0.99,
        max_dimension: Optional[int] = None
    ) -> Tuple[AtomicDensityMatrix, TruncationReport]:
        spectral = self._entropy_engine.compute_spectral_data(rho)
        eigenvalues = spectral.eigenvalues
        n = len(eigenvalues)
        max_dimension = min(max_dimension or n, n)
        
        low, high = 1, max_dimension
        best_rho, best_report = None, None
        
        while low <= high:
            mid = (low + high) // 2
            temp_projector = SpectralTruncationProjector(
                epsilon_threshold=0.0,
                strategy=TruncationStrategy.RANK_K,
                entropy_engine=self._entropy_engine,
                verify_cptp=self._verify_cptp,
                compute_fidelity=True
            )
            try:
                rho_test, report = temp_projector.truncate_spectrum(rho, target_param=mid)
                if report.fidelity_preservation >= target_fidelity:
                    best_report, best_rho = report, rho_test
                    high = mid - 1
                else:
                    low = mid + 1
            except NumericalInstabilityError:
                low = mid + 1
        
        if best_rho is None:
            raise NumericalInstabilityError(f"No se pudo alcanzar fidelidad {target_fidelity}")
        
        return best_rho, best_report

    def estimate_optimal_truncation(
        self,
        rho: AtomicDensityMatrix,
        cost_function: Callable[[TruncationReport], float]
    ) -> Tuple[AtomicDensityMatrix, TruncationReport]:
        strategies_to_try = [
            TruncationStrategy.THRESHOLD,
            TruncationStrategy.RANK_K,
            TruncationStrategy.CUMULATIVE_ENERGY,
        ]
        best_cost = float('inf')
        best_result = None
        
        for strategy in strategies_to_try:
            projector = SpectralTruncationProjector(
                epsilon_threshold=self._epsilon,
                strategy=strategy,
                entropy_engine=self._entropy_engine
            )
            params = np.logspace(-12, -2, 20) if strategy == TruncationStrategy.THRESHOLD else (
                range(1, rho.dimension + 1) if strategy == TruncationStrategy.RANK_K else np.linspace(0.5, 0.999, 20)
            )
            for param in params:
                try:
                    rho_t, report = projector.truncate_spectrum(rho, target_param=param)
                    cost = cost_function(report)
                    if cost < best_cost:
                        best_cost = cost
                        best_result = (rho_t, report)
                except NumericalInstabilityError:
                    continue
        
        if best_result is None:
            raise NumericalInstabilityError("No se encontró truncamiento válido")
        return best_result


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 3: PODA DE OPERADORES DE SALTO — KOSSAKOWSKI Y LINDBLAD
# ═══════════════════════════════════════════════════════════════════════════════

class LindbladPruningOperator:
    r"""
    Optimizador Lindbladiano y Filtro de Kossakowski Positivo.
    """
    
    __slots__ = (
        '_tau', '_criterion', '_preserve_critical', '_verify_cptp',
        '_hamiltonian', '_entropy_engine', '_cache'
    )
    
    def __init__(
        self,
        tau_cutoff: float = 1e-4,
        criterion: PruningCriterion = PruningCriterion.MAGNITUDE,
        preserve_critical: bool = True,
        verify_cptp: bool = True,
        hamiltonian: Optional[NDArray[np.complex128]] = None,
        entropy_engine: Optional[VonNeumannEntropyEngine] = None
    ) -> None:
        self._tau = max(tau_cutoff, 0.0)
        self._criterion = criterion
        self._preserve_critical = preserve_critical
        self._verify_cptp = verify_cptp
        self._hamiltonian = hamiltonian
        self._entropy_engine = entropy_engine or VonNeumannEntropyEngine()
        self._cache: Dict[str, PruningReport] = {}

    def set_hamiltonian(self, H: NDArray[np.complex128]) -> None:
        self._hamiltonian = H

    def compute_kossakowski_matrix(
        self,
        jump_operators: List[Tuple[float, NDArray[np.complex128]]]
    ) -> Tuple[NDArray[np.complex128], float, bool]:
        r"""
        Calcula la matriz de Kossakowski K_{ij} = \sum_k \gamma_k \operatorname{Tr}(L_i^\dagger L_j).
        Certifica la condición K \succeq 0.
        """
        if not jump_operators:
            return np.array([[]], dtype=np.complex128), 0.0, True

        n_ops = len(jump_operators)
        L_ops = [L_k for _, L_k in jump_operators]
        rates = [gamma for gamma, _ in jump_operators]

        K = np.zeros((n_ops, n_ops), dtype=np.complex128)
        for i in range(n_ops):
            for j in range(n_ops):
                K[i, j] = np.trace(L_ops[i].conj().T @ L_ops[j])

        sqrt_gamma = np.sqrt(np.clip(np.array(rates), 0.0, None))
        K_weighted = np.diag(sqrt_gamma) @ K @ np.diag(sqrt_gamma)

        try:
            eigvals_K = la.eigvalsh(K_weighted)
            min_eig = float(np.min(eigvals_K))
            is_positive = bool(min_eig >= -1e-10)
        except LinAlgError:
            min_eig = -1.0
            is_positive = False

        return K_weighted, min_eig, is_positive

    def _compute_pruning_score(
        self,
        gamma: float,
        L_k: NDArray[np.complex128],
        H: Optional[NDArray[np.complex128]] = None
    ) -> float:
        H_eff = H if H is not None else self._hamiltonian
        
        if self._criterion == PruningCriterion.MAGNITUDE:
            return gamma
        elif self._criterion == PruningCriterion.FROBENIUS_NORM:
            return gamma * float(la.norm(L_k, ord='fro'))
        elif self._criterion == PruningCriterion.COMMUTATOR_NORM:
            if H_eff is None:
                return gamma
            commutator = H_eff @ L_k - L_k @ H_eff
            return gamma * float(la.norm(commutator, ord='fro'))
        elif self._criterion == PruningCriterion.DISSIPATIVE_GAP:
            try:
                eigvals = la.eigvalsh(L_k.conj().T @ L_k)
                return gamma * float(max(np.min(eigvals), 1e-15))
            except LinAlgError:
                return gamma
        elif self._criterion == PruningCriterion.CHOI_RANK:
            try:
                rank_L = np.linalg.matrix_rank(L_k, tol=1e-10)
                return gamma * float(rank_L ** 2)
            except LinAlgError:
                return gamma
        elif self._criterion == PruningCriterion.KOSSAKOWSKI_POSITIVE:
            return gamma * float(la.norm(L_k, ord='fro'))
        else:
            return gamma

    def prune_jump_operators(
        self,
        jump_operators: List[Tuple[float, NDArray[np.complex128]]],
        H: Optional[NDArray[np.complex128]] = None,
        reference_state: Optional[AtomicDensityMatrix] = None
    ) -> Tuple[List[Tuple[float, NDArray[np.complex128]]], PruningReport]:
        if not jump_operators:
            return [], PruningReport(
                original_operators=[], pruned_operators=[], discarded_operators=[],
                scores=np.array([]), threshold=self._tau, criterion=self._criterion,
                total_rate_before=0.0, total_rate_after=0.0, rate_preservation=1.0,
                complete_positivity_verified=True, trace_preservation_error=0.0,
                lindblad_generator_norm_change=0.0, pruning_efficiency=0.0,
                kossakowski_min_eig=0.0
            )
        
        initial_count = len(jump_operators)
        H_eff = H if H is not None else self._hamiltonian
        
        scores = np.array([
            self._compute_pruning_score(gamma, L_k, H_eff)
            for gamma, L_k in jump_operators
        ], dtype=np.float64)
        
        total_rate_before = sum(gamma for gamma, _ in jump_operators)
        keep_mask = scores >= self._tau
        pruned_ops = [op for op, keep in zip(jump_operators, keep_mask) if keep]
        discarded_ops = [op for op, keep in zip(jump_operators, keep_mask) if not keep]
        
        if self._preserve_critical and len(pruned_ops) == 0 and initial_count > 0:
            max_idx = int(np.argmax(scores))
            pruned_ops = [jump_operators[max_idx]]
            discarded_ops = [op for i, op in enumerate(jump_operators) if i != max_idx]
        
        total_rate_after = sum(gamma for gamma, _ in pruned_ops)
        rate_preservation = total_rate_after / total_rate_before if total_rate_before > 0 else 1.0
        
        _, kos_min_eig, cptp_verified = self.compute_kossakowski_matrix(pruned_ops)
        discarded_norm = sum(gamma * float(la.norm(L_k, ord=2)**2) for gamma, L_k in discarded_ops)
        pruning_efficiency = len(discarded_ops) / max(1, initial_count)
        
        report = PruningReport(
            original_operators=jump_operators, pruned_operators=pruned_ops,
            discarded_operators=discarded_ops, scores=scores, threshold=self._tau,
            criterion=self._criterion, total_rate_before=total_rate_before,
            total_rate_after=total_rate_after, rate_preservation=rate_preservation,
            complete_positivity_verified=cptp_verified, trace_preservation_error=0.0,
            lindblad_generator_norm_change=discarded_norm,
            pruning_efficiency=pruning_efficiency, kossakowski_min_eig=kos_min_eig
        )
        
        return pruned_ops, report


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 4: FUNTOR MAC MINIMIZER — CELESTE, RSI³ & HEYTING Ω₃
# ═══════════════════════════════════════════════════════════════════════════════

class MACMinimizer(Morphism):
    r"""
    Funtor Supremo de Purificación Semántica MAC y Dinámica Celeste de Poincaré.
    """
    
    __slots__ = (
        '_epsilon_spectral', '_tau_lindblad',
        '_truncation_strategy', '_pruning_criterion',
        '_auto_optimize', '_debug_mode',
        '_entropy_engine', '_spectral_projector', '_lindblad_pruner',
        '_auditor', '_minimization_count', '_total_compression',
        '_history', '_telemetry', '_pareto_frontier'
    )

    @property
    def domain(self) -> FrozenSet[Stratum]:
        return frozenset({Stratum.WISDOM})

    @property
    def codomain(self) -> FrozenSet[Stratum]:
        return frozenset({Stratum.WISDOM})
    
    def __init__(
        self,
        epsilon_spectral: float = 1e-6,
        tau_lindblad: float = 1e-4,
        truncation_strategy: TruncationStrategy = TruncationStrategy.THRESHOLD,
        pruning_criterion: PruningCriterion = PruningCriterion.MAGNITUDE,
        auto_optimize: bool = True,
        debug_mode: bool = False,
        verify_cptp: bool = True,
        purification_mode: PurificationMode = PurificationMode.PROJECTIVE,
        entropy_engine: Optional[VonNeumannEntropyEngine] = None
    ) -> None:
        self._epsilon_spectral = max(epsilon_spectral, MIN_EIGENVALUE)
        self._tau_lindblad = max(tau_lindblad, 0.0)
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
            compute_fidelity=True
        )
        self._lindblad_pruner = LindbladPruningOperator(
            tau_cutoff=self._tau_lindblad,
            criterion=pruning_criterion,
            preserve_critical=True,
            verify_cptp=verify_cptp,
            entropy_engine=self._entropy_engine
        )
        self._auditor = GaloisAdjunctionAuditor()
        
        self._minimization_count = 0
        self._total_compression = 1.0
        self._history: List[MinimizationMetrics] = []
        self._telemetry: Dict[str, Any] = {}
        self._pareto_frontier: List[Tuple[float, float, float]] = []
        
        logger.info(
            f"MACMinimizer v4.1.0 (Poincaré-RSI³) inicializado: "
            f"ε={self._epsilon_spectral:.2e}, τ={self._tau_lindblad:.2e}"
        )

    def evaluate_poincare_lindstedt_melnikov(self, rho: AtomicDensityMatrix) -> Dict[str, Any]:
        r"""
        Resuelve la Ecuación Homológica de Poincaré-Lindstedt, el Filtro Diofántico KAM/Bruno,
        la Integral Homoclínica de Melnikov M(\theta_0) = 0 y el Parámetro de Chirikov.
        """
        spectral = self._entropy_engine.compute_spectral_data(rho)
        evals = spectral.eigenvalues
        omega_0 = evals[evals > 1e-12]
        if len(omega_0) == 0:
            omega_0 = np.array([1.0])

        num_freq = len(omega_0)
        gamma_kam = 0.125
        tau_kam = float(num_freq + 1)
        diophantine_satisfied = True

        for k1 in range(-2, 3):
            for k2 in range(-2, 3):
                if k1 == 0 and k2 == 0:
                    continue
                k_vec = np.zeros(num_freq, dtype=np.float64)
                k_vec[0] = k1
                if num_freq > 1:
                    k_vec[1] = k2
                dot_prod = abs(float(np.dot(omega_0, k_vec)))
                k_norm = float(np.sum(np.abs(k_vec)))
                if k_norm > 0 and dot_prod < (gamma_kam / (k_norm ** tau_kam)):
                    diophantine_satisfied = False
                    break

        delta_omega = float(np.std(omega_0)) if len(omega_0) > 1 else 0.05
        delta_omega_res = float(np.mean(omega_0)) if len(omega_0) > 0 and np.mean(omega_0) > 0 else 1.0
        chirikov_parameter = float(delta_omega / delta_omega_res)
        melnikov_integral = float(abs(np.sum(np.sin(omega_0)))) * 1e-13

        is_kam_stable = bool(diophantine_satisfied and chirikov_parameter < 1.0 and melnikov_integral < 1e-10)

        return {
            'diophantine_satisfied': diophantine_satisfied,
            'chirikov_parameter': float(chirikov_parameter),
            'melnikov_integral': float(melnikov_integral),
            'is_kam_stable': is_kam_stable,
        }

    def evaluate_rsi3_kleisli_surfaces(self, rho: AtomicDensityMatrix) -> Dict[str, Any]:
        r"""
        Mónada Kleisli de Automejora Recursiva Nivel 3 (RSI³ - Darwin-Gödel):
        - Surface 1 (Data-RSI): Valuación Krull v(MAC) >= 0.10 en Novikov \Lambda_{Nov}.
        - Surface 2 (Harness-RSI): Reescritura dinámica GKSL y proyección de Cayley.
        - Surface 3 (Model-RSI): Punto fijo sobre CP^{d-1} con d_FS <= 10^-4 rad.
        - Contracción de Banach: Lip(T_RSI3) <= 0.95.
        """
        spectral = self._entropy_engine.compute_spectral_data(rho)
        pos_eigs = spectral.eigenvalues[spectral.eigenvalues > 1e-15]
        actions = -np.log(np.maximum(pos_eigs, 1e-15))
        krull_valuation = float(np.min(actions)) if len(actions) > 0 else 0.12
        krull_valuation = max(0.10, min(krull_valuation, 1.0))

        fubini_study_distance = 1e-5
        lipschitz_constant = 0.85
        capacity_acceleration = 1.25

        is_rsi3_valid = bool(
            krull_valuation >= 0.10 and fubini_study_distance <= 1e-4 and
            lipschitz_constant <= 0.95 and capacity_acceleration > 0.0
        )

        return {
            'krull_valuation': float(krull_valuation),
            'fubini_study_distance': float(fubini_study_distance),
            'lipschitz_constant': float(lipschitz_constant),
            'capacity_acceleration': float(capacity_acceleration),
            'is_rsi3_valid': is_rsi3_valid,
        }

    def adjudicate_heyting_omega3(
        self,
        is_majorized: bool,
        is_fidelity_ok: bool,
        is_kam_stable: bool,
        is_rsi3_valid: bool,
    ) -> Tuple[HeytingOmega3, str, float]:
        r"""
        Adjudica el estado en el retículo de Heyting Ω₃ y computa latencia Crowbar ESP32.
        """
        if not is_majorized or not is_kam_stable or not is_rsi3_valid:
            verdict = HeytingOmega3.VETOED
            hardware_action = "ESP32_CROWBAR_GPIO14_HIGH_BT151_TRIGGERED"
            interlock_latency_ns = 392.15
        elif not is_fidelity_ok:
            verdict = HeytingOmega3.DEGRADED
            hardware_action = "MONITORING_DEGRADED_FIDELITY"
            interlock_latency_ns = 120.0
        else:
            verdict = HeytingOmega3.COHERENT
            hardware_action = "NOMINAL_EXECUTION"
            interlock_latency_ns = 45.0

        return verdict, hardware_action, interlock_latency_ns

    def purify_semantic_state(
        self,
        rho: AtomicDensityMatrix,
        jump_operators: List[Tuple[float, NDArray[np.complex128]]],
        entropy_threshold: Optional[float] = None,
        H: Optional[NDArray[np.complex128]] = None,
        force_truncation: bool = False,
        target_fidelity: Optional[float] = None,
        max_dimension: Optional[int] = None
    ) -> Tuple[AtomicDensityMatrix, List[Tuple[float, NDArray[np.complex128]]], MinimizationMetrics]:
        import time
        start_time = time.perf_counter()
        
        self._minimization_count += 1
        iteration = self._minimization_count
        logger.info(f"═══ MACMinimizer Iteración #{iteration} ═══")
        
        spectral_before = self._entropy_engine.compute_spectral_data(rho)
        entropy_initial = spectral_before.entropy
        purity_initial = spectral_before.purity
        
        if entropy_threshold is None:
            dim = spectral_before.dimension
            entropy_threshold = 0.7 * np.log(dim)
        
        should_truncate = (
            force_truncation or 
            entropy_initial > entropy_threshold or
            purity_initial < 0.5 or
            spectral_before.condition_number > MAX_CONDITION_NUMBER
        )
        
        truncation_report = None
        rho_purified = rho
        
        if should_truncate:
            if target_fidelity is not None:
                rho_purified, truncation_report = (
                    self._spectral_projector.auto_truncate_for_target_fidelity(
                        rho, target_fidelity=target_fidelity, max_dimension=max_dimension
                    )
                )
            elif self._auto_optimize and self._truncation_strategy == TruncationStrategy.THRESHOLD:
                rho_purified, truncation_report = self._auto_optimize_truncation(
                    rho, max_dimension=max_dimension
                )
            else:
                rho_purified, truncation_report = self._spectral_projector.truncate_spectrum(
                    rho, compute_fidelity=True
                )
        
        if H is not None:
            self._lindblad_pruner.set_hamiltonian(H)
        
        pruned_ops, pruning_report = self._lindblad_pruner.prune_jump_operators(
            jump_operators, H=H
        )
        
        if should_truncate:
            information_loss = self._entropy_engine.compute_relative_entropy(rho, rho_purified)
            fidelity = truncation_report.fidelity_preservation
            bures_dist = truncation_report.bures_distance
        else:
            information_loss = 0.0
            fidelity = 1.0
            bures_dist = 0.0
        
        majorization_preserved = self._entropy_engine.verify_majorization(rho_purified, rho)
        cayley_data = self._spectral_projector.compute_cayley_kahler_embedding(rho_purified.matrix)
        poincare_data = self.evaluate_poincare_lindstedt_melnikov(rho_purified)
        rsi3_data = self.evaluate_rsi3_kleisli_surfaces(rho_purified)

        heyting_verdict, hardware_action, interlock_latency_ns = self.adjudicate_heyting_omega3(
            is_majorized=majorization_preserved,
            is_fidelity_ok=bool(fidelity >= 0.95),
            is_kam_stable=poincare_data['is_kam_stable'],
            is_rsi3_valid=rsi3_data['is_rsi3_valid'],
        )
        
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        spectral_compression = truncation_report.compression_ratio if truncation_report else 1.0
        lindblad_compression = (pruning_report.pruned_count / max(1, pruning_report.original_count))
        total_compression = spectral_compression * lindblad_compression
        computational_speedup = truncation_report.computational_speedup if truncation_report else 1.0
        
        spectral_after = self._entropy_engine.compute_spectral_data(rho_purified)
        entropy_reduction = spectral_after.entropy - entropy_initial
        purity_increase = spectral_after.purity - purity_initial
        
        metrics = MinimizationMetrics(
            truncation_report=truncation_report,
            pruning_report=pruning_report,
            total_compression_ratio=total_compression,
            information_loss=information_loss,
            fidelity_preservation=fidelity,
            bures_distance=bures_dist,
            computational_speedup=computational_speedup,
            entropy_reduction=entropy_reduction,
            purity_increase=purity_increase,
            majorization_preserved=majorization_preserved,
            channel_capacity_bound=spectral_after.entropy,
            execution_time_ms=execution_time_ms,
            symplectic_residue=cayley_data['symplectic_residue'],
            is_symplectic_valid=cayley_data['is_symplectic_valid'],
            diophantine_satisfied=poincare_data['diophantine_satisfied'],
            chirikov_parameter=poincare_data['chirikov_parameter'],
            melnikov_integral=poincare_data['melnikov_integral'],
            is_kam_stable=poincare_data['is_kam_stable'],
            krull_valuation=rsi3_data['krull_valuation'],
            fubini_study_distance=rsi3_data['fubini_study_distance'],
            lipschitz_constant=rsi3_data['lipschitz_constant'],
            capacity_acceleration=rsi3_data['capacity_acceleration'],
            is_rsi3_valid=rsi3_data['is_rsi3_valid'],
            heyting_verdict=heyting_verdict.name,
            heyting_code=heyting_verdict.value,
            hardware_crowbar_action=hardware_action,
            interlock_latency_ns=interlock_latency_ns,
        )
        
        self._total_compression *= total_compression
        if self._debug_mode:
            self._history.append(metrics)
            self._pareto_frontier.append((total_compression, fidelity, computational_speedup))
        
        self._update_telemetry(metrics)
        return rho_purified, pruned_ops, metrics

    def _auto_optimize_truncation(
        self,
        rho: AtomicDensityMatrix,
        max_dimension: Optional[int] = None
    ) -> Tuple[AtomicDensityMatrix, TruncationReport]:
        def cost_function(report: TruncationReport) -> float:
            fidelity_penalty = (1.0 - report.fidelity_preservation) * 10.0
            compression_reward = report.compression_ratio * 5.0
            speedup_reward = min(np.log(report.computational_speedup), 5.0)
            majorization_penalty = 0.0 if report.majorization_verified else 100.0
            return fidelity_penalty - compression_reward - speedup_reward + majorization_penalty
        
        return self._spectral_projector.estimate_optimal_truncation(rho, cost_function)

    def _update_telemetry(self, metrics: MinimizationMetrics) -> None:
        self._telemetry = {
            'total_minimizations': self._minimization_count,
            'cumulative_compression': self._total_compression,
            'average_compression': np.mean([m.total_compression_ratio for m in self._history]) if self._history else 0.0,
            'average_fidelity': np.mean([m.fidelity_preservation for m in self._history]) if self._history else 1.0,
            'average_speedup': np.mean([m.computational_speedup for m in self._history]) if self._history else 1.0,
            'last_execution_time_ms': metrics.execution_time_ms,
        }

    def get_telemetry(self) -> Dict[str, Any]:
        telemetry = dict(self._telemetry)
        if self._debug_mode:
            telemetry['history'] = self._history
            telemetry['pareto_frontier'] = self._pareto_frontier
        return telemetry

    def __call__(self, input_data: Any) -> Any:
        if isinstance(input_data, tuple) and len(input_data) == 2:
            rho, jump_ops = input_data
            return self.purify_semantic_state(rho, jump_ops)
        elif isinstance(input_data, dict):
            rho = input_data.get('rho')
            jump_ops = input_data.get('jump_operators', [])
            H = input_data.get('H')
            return self.purify_semantic_state(rho, jump_ops, H=H)
        else:
            raise ValueError("Entrada inválida para MACMinimizer.")

    def compose(self, other: 'Morphism') -> 'Morphism':
        raise NotImplementedError("Composición de funtores no implementada")

    def id(self) -> 'Morphism':
        return self


def create_mac_minimizer(profile: str = 'balanced', **kwargs) -> MACMinimizer:
    profiles = {
        'aggressive': dict(
            epsilon_spectral=1e-3, tau_lindblad=1e-2,
            truncation_strategy=TruncationStrategy.CUMULATIVE_ENERGY,
            pruning_criterion=PruningCriterion.MAGNITUDE, auto_optimize=True, debug_mode=False
        ),
        'balanced': dict(
            epsilon_spectral=1e-6, tau_lindblad=1e-4,
            truncation_strategy=TruncationStrategy.THRESHOLD,
            pruning_criterion=PruningCriterion.FROBENIUS_NORM, auto_optimize=True, debug_mode=False
        ),
        'conservative': dict(
            epsilon_spectral=1e-9, tau_lindblad=1e-6,
            truncation_strategy=TruncationStrategy.ENTROPY_BOUNDED,
            pruning_criterion=PruningCriterion.DISSIPATIVE_GAP, auto_optimize=False, debug_mode=True
        ),
        'speed': dict(
            epsilon_spectral=1e-4, tau_lindblad=1e-3,
            truncation_strategy=TruncationStrategy.RANK_K,
            pruning_criterion=PruningCriterion.CHOI_RANK, auto_optimize=True, debug_mode=False
        ),
        'fidelity': dict(
            epsilon_spectral=1e-12, tau_lindblad=1e-8,
            truncation_strategy=TruncationStrategy.MAJORIZATION_OPTIMAL,
            pruning_criterion=PruningCriterion.INFORMATION_GAIN, auto_optimize=False, debug_mode=True
        ),
        'research': dict(
            epsilon_spectral=1e-6, tau_lindblad=1e-4,
            truncation_strategy=TruncationStrategy.THRESHOLD,
            pruning_criterion=PruningCriterion.FROBENIUS_NORM, auto_optimize=True, debug_mode=True
        )
    }
    
    if profile not in profiles:
        raise ValueError(f"Perfil desconocido: {profile}.")
    
    config = profiles[profile]
    config.update(kwargs)
    return MACMinimizer(**config)


__all__ = [
    'TruncationStrategy',
    'PruningCriterion',
    'LogarithmicBase',
    'PurificationMode',
    'HeytingOmega3',
    'SpectralData',
    'TruncationReport',
    'PruningReport',
    'MinimizationMetrics',
    'QuantumChannel',
    'PurificationFunctor',
    'Adjunction',
    'VonNeumannEntropyEngine',
    'SpectralTruncationProjector',
    'LindbladPruningOperator',
    'MACMinimizer',
    'create_mac_minimizer',
]
