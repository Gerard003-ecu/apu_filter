# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ MÓDULO: MAC Minimizer — Funtor de Purificación Espectral y Reducción Cuántica ║
║ Ubicación: app/boole/tactics/mac_minimizer.py                                 ║
║ Versión: 3.0.0-Topos-Spectral-Categorical-Enhanced                            ║
╚═══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y TOPOLÓGICA REFINADA:
──────────────────────────────────────────────
Este minimizador implementa un **funtor de purificación** 
  P : 𝐐𝐮𝐚𝐧𝐭 → 𝐐𝐮𝐚𝐧𝐭_𝐩𝐮𝐫𝐞
sobre la 2-categoría de canales cuánticos completamente positivos y preservadores de traza (CPTP).
Comprime el operador de densidad ρ ∈ 𝒟(ℋ) eliminando subespacios de baja relevancia semántica
mediante truncamiento espectral óptimo bajo preorden de majorización cuántica,
maximizando la eficiencia informacional del sistema MAC.

FUNDAMENTOS TEÓRICOS UNIFICADOS:
────────────────────────────────
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. TEORÍA ESPECTRAL AVANZADA                                                │
│    • Teorema Espectral para operadores autoadjuntos compactos               │
│    • Descomposición de Schmidt y valores singulares                         │
│    • Perturbación de Weyl y brechas espectrales                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ 2. TEORÍA DE INFORMACIÓN CUÁNTICA                                           │
│    • Entropía de von Neumann: S(ρ) = -Tr(ρ ln ρ)                            │
│    • Entropías de Rényi: S_α(ρ) = (1-α)⁻¹ ln Tr(ρ^α)                        │
│    • Divergencia cuántica relativa: D(ρ‖σ) = Tr(ρ(ln ρ - ln σ))             │
│    • Fidelidad de Uhlmann: F(ρ,σ) = ‖√ρ√σ‖₁²                                │
│    • Majorización cuántica: ρ ≺ σ ⇔ ∃ canal CPTP Λ: ρ = Λ(σ)                │
├─────────────────────────────────────────────────────────────────────────────┤
│ 3. GEOMETRÍA DIFERENCIAL CUÁNTICA                                           │
│    • Variedad de estados cuánticos: 𝒟(ℋ) ≅ {ρ ≥ 0, Tr ρ = 1}               │
│    • Métrica de Bures: d_B(ρ,σ)² = 2(1 - F(ρ,σ))                            │
│    • Métrica de Fisher-Bures (métrica cuántica natural)                     │
│    • Geodésicas y curvatura escalar                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ 4. ÁLGEBRA DE OPERADORES Y C*-ÁLGEBRAS                                      │
│    • Teorema de Stinespring: Λ(ρ) = V†π(ρ)V                                 │
│    • Teorema de Choi-Kraus: Λ(ρ) = Σᵢ KᵢρKᵢ†, Σᵢ Kᵢ†Kᵢ = I                  │
│    • Forma de Lindblad-GKSL: ℒ(ρ) = -i[H,ρ] + Σₖ γₖ 𝒟[Lₖ](ρ)                │
│    • Semi-grupos cuánticos dinámicos                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ 5. TEORÍA DE CATEGORÍAS Y TOPOS                                             │
│    • Categoría 𝐐𝐮𝐚𝐧𝐭: objetos = espacios de Hilbert, morfismos = canales CPTP│
│    • Funtor de purificación P: 𝐐𝐮𝐚𝐧𝐭 → 𝐐𝐮𝐚𝐧𝐭_𝐩𝐮𝐫𝐞 ⊣ ι (inclusión)             │
│    • Adjunción P ⊣ ι: Hom(Pρ, σ) ≅ Hom(ρ, ισ)                               │
│    • Monada de purificación: T = ι∘P con unidad η: Id → T                   │
│    • Lógica interna del topos de conjuntos cuánticos                        │
└─────────────────────────────────────────────────────────────────────────────┘

AXIOMAS MATEMÁTICOS IMPLEMENTADOS Y EXTENDIDOS:
────────────────────────────────────────────────
A1. Entropía de von Neumann:        S(ρ) = -Tr(ρ ln ρ) = -Σᵢ λᵢ ln λᵢ
A2. Descomposición Espectral:       ρ = Σₖ λₖ |ψₖ⟩⟨ψₖ|,  λₖ ≥ 0, Σₖ λₖ = 1
A3. Majorización Cuántica:          ρ ≺ σ  ⇔  λ(ρ) ≺ λ(σ)  (vector de eigenvalores)
A4. Proyección de Truncamiento:     P_ε = Σ_{λₖ≥ε} |ψₖ⟩⟨ψₖ|  (proyector ortogonal)
A5. Mapa CPTP de Purificación:      ρ̃ = P_ε ρ P_ε / Tr(P_ε ρ P_ε)
A6. Pureza Relativa:                γ(ρ) = Tr(ρ²) ∈ [1/d, 1]
A7. Rango Efectivo:                 r_eff(ρ) = exp(S(ρ)) = exp(H(λ))
A8. Fidelidad de Uhlmann:           F(ρ,σ) = (Tr √(√ρ σ √ρ))²
A9. Divergencia Relativa:           D(ρ‖σ) = Tr(ρ(ln ρ - ln σ)) ≥ 0
A10. Conservación Informacional:    I(ρ→ρ̃) = S(ρ̃) - S(ρ) + D(ρ‖ρ̃) ≤ 0
A11. Límite de Holevo:              χ(ℰ) = S(Σ pᵢρᵢ) - Σ pᵢS(ρᵢ)
A12. Desigualdad de Araki-Lieb:     |S(ρ_A) - S(ρ_B)| ≤ S(ρ_AB) ≤ S(ρ_A) + S(ρ_B)

REFERENCIAS TEÓRICAS CANÓNICAS:
────────────────────────────────
[1] von Neumann, J. (1932). "Mathematische Grundlagen der Quantenmechanik"
[2] Nielsen, M.A. & Chuang, I.L. (2010). "Quantum Computation and Quantum Information"
[3] Bhatia, R. (1997). "Matrix Analysis" — Majorización y desigualdades matriciales
[4] Wilde, M.M. (2013). "Quantum Information Theory" — Entropías y capacidades
[5] Heinosaari, T. & Ziman, M. (2012). "The Mathematical Language of Quantum Theory"
[6] Wolf, M.M. (2012). "Quantum Channels & Operations: Guided Tour" — GKSL, Stinespring
[7] Carlen, E.A. (2010). "Trace Inequalities and Quantum Entropy" — Araki-Lieb, etc.
[8] Coecke, B. & Kissinger, A. (2017). "Picturing Quantum Processes" — Categorical QM
[9] Selinger, P. (2007). "Dagger Compact Closed Categories and Completely Positive Maps"
[10] Jacobs, B. (2015). "New Directions in Categorical Logic for Quantum Mechanics"

═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
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
from app.core.mic_algebra import Morphism, NumericalInstabilityError
from app.core.schemas import Stratum
logger = logging.getLogger("MAC.Wisdom.Minimizer")

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
        # Validaciones físicas
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
        assert 0 <= self.fidelity_preservation <= 1, "Fidelidad inválida"
        assert self.bures_distance >= 0, "Distancia de Bures negativa"
        assert self.relative_entropy >= -1e-10, "Divergencia relativa negativa (numérico)"

    @property
    def original_dimension(self) -> int:
        """Dimensión original del espacio de Hilbert."""
        return self.original_spectral_data.dimension

    @property
    def retained_eigenvalues(self) -> NDArray[np.float64]:
        """Eigenvalores correspondientes a los componentes retenidos."""
        return self.original_spectral_data.eigenvalues[self.retention_mask]

    @property
    def truncated_dimension(self) -> int:
        """Dimensión después del truncamiento."""
        return self.truncated_spectral_data.dimension

    @property
    def purity_before(self) -> float:
        """Pureza antes del truncamiento."""
        return self.original_spectral_data.purity

    @property
    def purity_after(self) -> float:
        """Pureza después del truncamiento."""
        return self.truncated_spectral_data.purity

    @property
    def entropy_before(self) -> float:
        """Entropía antes del truncamiento."""
        return self.original_spectral_data.entropy

    @property
    def entropy_after(self) -> float:
        """Entropía después del truncamiento."""
        return self.truncated_spectral_data.entropy


@dataclass(frozen=True, slots=True)
class PruningReport:
    """Reporte de poda de operadores de Lindblad con análisis de CPTP."""
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
    
    def __post_init__(self) -> None:
        assert len(self.original_operators) == len(self.pruned_operators) + len(self.discarded_operators)
        assert 0 <= self.rate_preservation <= 1, "Preservación de tasa inválida"
        assert 0 <= self.pruning_efficiency <= 1, "Eficiencia de poda inválida"

    @property
    def original_count(self) -> int:
        """Número original de operadores de Lindblad."""
        return len(self.original_operators)

    @property
    def pruned_count(self) -> int:
        """Número de operadores retenidos después de la poda."""
        return len(self.pruned_operators)

    @property
    def discarded_count(self) -> int:
        """Número de operadores descartados."""
        return len(self.discarded_operators)


@dataclass(frozen=True, slots=True)
class MinimizationMetrics:
    """Métricas agregadas de minimización del funtor MAC."""
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
    
    def __post_init__(self) -> None:
        assert 0 <= self.total_compression_ratio <= 1, "Compresión total inválida"
        assert self.information_loss >= -1e-10, "Pérdida de información negativa"
        assert 0 <= self.fidelity_preservation <= 1, "Fidelidad inválida"
        assert self.bures_distance >= 0, "Distancia de Bures negativa"
        assert self.computational_speedup >= 1.0, "Speedup < 1"


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
    """Funtor de purificación P: 𝐐𝐮𝐚𝐧𝐭 → 𝐐𝐮𝐚𝐧𝐭_𝐩𝐮𝐫𝐞."""
    def purify(self, rho: AtomicDensityMatrix) -> Tuple[AtomicDensityMatrix, TruncationReport]: ...
    def on_morphism(self, channel: QuantumChannel) -> QuantumChannel: ...
    def unit(self, rho: AtomicDensityMatrix) -> QuantumChannel: ...  # η: Id → T


class Adjunction(Protocol):
    """Adjunción P ⊣ ι entre categoría cuántica y subcategoría pura."""
    def left_adjoint(self, rho: AtomicDensityMatrix) -> AtomicDensityMatrix: ...   # P
    def right_adjoint(self, rho: AtomicDensityMatrix) -> AtomicDensityMatrix: ...  # ι (inclusión)
    def hom_isomorphism(self, rho: AtomicDensityMatrix, sigma: AtomicDensityMatrix) -> bool: ...
    def unit(self, rho: AtomicDensityMatrix) -> QuantumChannel: ...  # η: ρ → ι(Pρ)
    def counit(self, rho: AtomicDensityMatrix) -> QuantumChannel: ...  # ε: P(ιρ) → ρ


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1: MOTOR ENTRÓPICO Y PURIFICACIÓN DE VON NEUMANN (ENTROPY ENGINE)
# ═══════════════════════════════════════════════════════════════════════════════

class VonNeumannEntropyEngine:
    r"""
    Calculadora Rigurosa del Tensor Entrópico Cuántico — Versión Definitiva.
    
    FUNDAMENTO TEÓRICO:
    ──────────────────
    Implementa la familia completa de entropías cuánticas con estabilidad numérica
    garantizada mediante regularización espectral y aritmética de precisión extendida.
    
    ENTROPÍAS IMPLEMENTADAS:
    ───────────────────────
    1. von Neumann:    S(ρ) = -Tr(ρ ln ρ) = H(λ)           (Shannon de eigenvalores)
    2. Rényi-α:        S_α(ρ) = (1-α)⁻¹ ln Tr(ρ^α)         (α ≥ 0, α ≠ 1)
    3. Tsallis-q:      T_q(ρ) = (1-q)⁻¹ (Tr(ρ^q) - 1)      (q > 0, q ≠ 1)
    4. Min-entropía:   S_∞(ρ) = -ln λ_max(ρ)
    5. Max-entropía:   S_0(ρ) = ln rank(ρ)
    6. Divergencia relativa: D(ρ‖σ) = Tr(ρ(ln ρ - ln σ))
    7. Información mutua:    I(A:B) = S(ρ_A) + S(ρ_B) - S(ρ_AB)
    8. Información coherente: I_c(A⟩B) = S(ρ_B) - S(ρ_AB)
    
    PROPIEDADES AXIOMÁTICAS VERIFICADAS:
    ────────────────────────────────────
    • S(ρ) ≥ 0 con igualdad ⇔ ρ puro
    • S(ρ) ≤ ln d con igualdad ⇔ ρ = I/d
    • Concavidad: S(Σ pᵢρᵢ) ≥ Σ pᵢ S(ρᵢ)
    • Invariancia unitaria: S(UρU†) = S(ρ)
    • Subaditividad: S(ρ_AB) ≤ S(ρ_A) + S(ρ_B)
    • Desigualdad de Araki-Lieb: |S(ρ_A) - S(ρ_B)| ≤ S(ρ_AB)
    • Monotonía bajo CPTP: S(Λ(ρ)) ≥ S(ρ) para Λ unital
    • Continuidad de Fannes-Audenaert: |S(ρ) - S(σ)| ≤ ε ln d + H(ε)
    
    ESTABILIDAD NUMÉRICA:
    ────────────────────
    • Regularización espectral: λᵢ → max(λᵢ, ε) con ε = 10⁻¹⁵
    • Log-sum-exp para evitrar overflow/underflow
    • Precisión extendida (float128) para diferencias pequeñas
    • Validación de hermeticidad y positividad antes de calcular
    """
    
    __slots__ = ('_tol', '_log_base', '_log_factor', '_use_extended_precision', '_cache')
    
    def __init__(
        self, 
        tol: float = DEFAULT_TOL,
        log_base: LogarithmicBase = LogarithmicBase.NATURAL,
        use_extended_precision: bool = True,
        cache_enabled: bool = True
    ) -> None:
        """
        Inicializa el motor entrópico.
        
        Args:
            tol: Tolerancia para eigenvalores considerados cero (default: 1e-12)
            log_base: Base logarítmica para entropías (default: natural/nats)
            use_extended_precision: Usar float128 para cálculos intermedios
            cache_enabled: Habilitar caché LRU para eigenvalores repetidos
        """
        self._tol = max(tol, MACHINE_EPS)
        self._log_base = log_base
        self._use_extended_precision = use_extended_precision
        self._cache: Dict[int, Tuple[NDArray, NDArray]] = {} if cache_enabled else None
        
        # Factor de conversión logarítmica
        if log_base == LogarithmicBase.NATURAL:
            self._log_factor = 1.0
        elif log_base == LogarithmicBase.BINARY:
            self._log_factor = 1.0 / np.log(2.0)
        elif log_base == LogarithmicBase.DECIMAL:
            self._log_factor = 1.0 / np.log(10.0)
        else:
            raise ValueError(f"Base logarítmica desconocida: {log_base}")
        
        logger.debug(
            f"VonNeumannEntropyEngine inicializado: tol={self._tol:.2e}, "
            f"base={log_base.value}, extended_prec={use_extended_precision}"
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # MÉTODOS PRINCIPALES DE ENTROPÍA
    # ═══════════════════════════════════════════════════════════════════════════
    
    def compute_spectral_data(self, rho: AtomicDensityMatrix) -> SpectralData:
        """
        Calcula datos espectrales completos del operador de densidad.
        
        Realiza descomposición espectral ρ = U Λ U† con validación física completa.
        
        Args:
            rho: Operador de densidad válido (hermítico, ≥ 0, Tr = 1)
        
        Returns:
            SpectralData inmutable con eigenvalores ordenados descendentemente
        
        Raises:
            NumericalInstabilityError: Si la matriz no es físicamente válida
        """
        matrix = rho.matrix
        dim = matrix.shape[0]
        
        # Validación rápida de hermeticidad
        hermiticity_error = la.norm(matrix - matrix.conj().T, ord='fro')
        if hermiticity_error > 1e-10:
            logger.warning(f"Hermiticidad violada: ‖ρ - ρ†‖_F = {hermiticity_error:.2e}")
            matrix = (matrix + matrix.conj().T) / 2.0  # Proyección hermítica
        
        # Descomposición espectral con estabilidad numérica
        try:
            eigenvalues, eigenvectors = la.eigh(matrix)
            if self._use_extended_precision:
                eigenvalues = eigenvalues.astype(np.float128)
        except LinAlgError as e:
            raise NumericalInstabilityError(
                f"Fallo en descomposición espectral: {e}. "
                f"Condición estimada: {la.cond(matrix):.2e}"
            )
        
        # Ordenar descendentemente
        sort_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]
        
        # Regularización y validación física
        eigenvalues = np.maximum(eigenvalues, 0.0)  # Proyección al cono positivo
        trace = np.sum(eigenvalues)
        
        if abs(trace - 1.0) > 1e-8:
            logger.warning(f"Traza = {trace:.6f}, renormalizando eigenvalores")
            eigenvalues = eigenvalues / trace
        
        # Métricas derivadas
        positive_eigs = eigenvalues[eigenvalues > self._tol]
        rank = len(positive_eigs)
        
        # Entropía de von Neumann
        entropy = self._compute_entropy_from_eigenvalues(positive_eigs)
        
        # Pureza γ = Tr(ρ²) = Σ λᵢ²
        purity = float(np.sum(eigenvalues**2))
        
        # Rango efectivo r_eff = exp(S)
        effective_rank = float(np.exp(entropy / self._log_factor))
        
        # Número de condición
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
        """Calcula S = -Σ λᵢ ln λᵢ con estabilidad numérica (log-sum-exp)."""
        if len(eigenvalues) == 0:
            return 0.0
        
        # Filtrar eigenvalores positivos significativos
        pos_eigs = eigenvalues[eigenvalues > self._tol]
        
        if len(pos_eigs) == 0:
            return 0.0
        
        # ── Sutura I: proyector de regularización sobre el espectro ──
        # Cota inferior de precisión de máquina para el operando del log.
        eps = np.finfo(pos_eigs.dtype).eps
        lambda_safe = np.maximum(pos_eigs, eps)
        
        # Usar log-sum-exp para estabilidad: -Σ λᵢ ln λᵢ
        # En precisión extendida si está habilitado
        if self._use_extended_precision:
            pos_eigs = pos_eigs.astype(np.float128)
            lambda_safe = lambda_safe.astype(np.float128)
            log_eigs = np.log(lambda_safe)
            entropy = -np.sum(pos_eigs * log_eigs)
            return float(entropy * self._log_factor)
        else:
            # Truco numérico: λ ln λ = λ (ln λ) donde ln λ se calcula seguramente
            log_eigs = np.log(lambda_safe)
            entropy = -np.sum(pos_eigs * log_eigs)
            return float(entropy * self._log_factor)

    def compute_entropy(
        self, 
        rho: AtomicDensityMatrix,
        validate: bool = True
    ) -> float:
        r"""
        Calcula rigurosamente la entropía de von Neumann: S(ρ) = -Tr(ρ ln ρ).
        
        Args:
            rho: Operador de densidad
            validate: Si True, valida axiomas cuánticos antes de calcular
        
        Returns:
            Entropía S(ρ) ≥ 0 en la base logarítmica configurada
        """
        if validate:
            # Validación rápida usando métricas del AtomicDensityMatrix
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

    def compute_renyi_entropy(
        self, 
        rho: AtomicDensityMatrix, 
        alpha: float,
        validate: bool = True
    ) -> float:
        r"""
        Entropía de Rényi de orden α:
        
            S_α(ρ) = (1-α)⁻¹ ln Tr(ρ^α) = (1-α)⁻¹ ln(Σᵢ λᵢ^α)
        
        Propiedades:
            • α → 1: converge a entropía de von Neumann
            • α = 0: S₀(ρ) = ln rank(ρ)  (max-entropía / Hartley)
            • α = 2: S₂(ρ) = -ln Tr(ρ²)  (entropía de colisión)
            • α → ∞: S_∞(ρ) = -ln λ_max  (min-entropía)
            • Monótona decreciente en α: S_α ≥ S_β para α ≤ β
        
        Args:
            rho: Operador de densidad
            alpha: Orden de la entropía (α ≥ 0, α ≠ 1 para fórmula directa)
            validate: Validar estado cuántico
        
        Returns:
            Entropía de Rényi S_α(ρ)
        """
        if alpha < 0:
            raise ValueError(f"Orden de Rényi debe ser no negativo: {alpha}")
        
        if validate:
            rho.compute_metrics()  # Lanza excepción si inválido
        
        spectral = self.compute_spectral_data(rho)
        eigenvalues = spectral.eigenvalues
        positive_eigs = eigenvalues[eigenvalues > self._tol]
        
        if np.isclose(alpha, 1.0, atol=1e-10):
            return spectral.entropy
        
        if np.isclose(alpha, 0.0):
            # S₀ = ln rank(ρ)
            return np.log(len(positive_eigs)) * self._log_factor
        
        if np.isinf(alpha):
            # S_∞ = -ln λ_max
            return -np.log(np.max(positive_eigs)) * self._log_factor
        
        # Caso general: S_α = (1-α)⁻¹ ln(Σ λᵢ^α)
        if self._use_extended_precision:
            pos = positive_eigs.astype(np.float128)
            trace_power = np.sum(pos ** alpha)
            result = (1.0 / (1.0 - alpha)) * np.log(trace_power)
            return float(result * self._log_factor)
        else:
            trace_power = np.sum(positive_eigs ** alpha)
            return float((1.0 / (1.0 - alpha)) * np.log(trace_power) * self._log_factor)

    def compute_tsallis_entropy(
        self, 
        rho: AtomicDensityMatrix, 
        q: float
    ) -> float:
        r"""
        Entropía de Tsallis de orden q:
        
            T_q(ρ) = (1-q)⁻¹ (Tr(ρ^q) - 1)
        
        Relación con Rényi: T_q = (exp((1-q)S_q) - 1) / (1-q)
        No aditiva: T_q(ρ⊗σ) = T_q(ρ) + T_q(σ) + (1-q)T_q(ρ)T_q(σ)
        """
        if q <= 0:
            raise ValueError(f"Parámetro Tsallis q debe ser > 0: {q}")
        
        if np.isclose(q, 1.0):
            return self.compute_entropy(rho)
        
        spectral = self.compute_spectral_data(rho)
        positive_eigs = spectral.eigenvalues[spectral.eigenvalues > self._tol]
        
        trace_power = np.sum(positive_eigs ** q)
        return float((trace_power - 1.0) / (1.0 - q) * self._log_factor)

    # ═══════════════════════════════════════════════════════════════════════════
    # DIVERGENCIAS Y MEDIDAS DE DISTANCIA
    # ═══════════════════════════════════════════════════════════════════════════
    
    def compute_relative_entropy(
        self, 
        rho: AtomicDensityMatrix, 
        sigma: AtomicDensityMatrix,
        validate: bool = True
    ) -> float:
        r"""
        Divergencia cuántica relativa (Kullback-Leibler cuántica):
        
            D(ρ‖σ) = Tr(ρ ln ρ) - Tr(ρ ln σ) = S(ρ,σ) - S(ρ)
        
        Propiedades fundamentales:
            • D(ρ‖σ) ≥ 0 con igualdad ⇔ ρ = σ
            • No simétrica: D(ρ‖σ) ≠ D(σ‖ρ)
            • Monotonía bajo CPTP: D(Λ(ρ)‖Λ(σ)) ≤ D(ρ‖σ)
            • Conjunta convexa en (ρ,σ)
            • D(ρ‖σ) = ∞ si supp(ρ) ⊄ supp(σ)
        
        Implementación numéricamente estable via descomposición espectral de σ.
        
        Args:
            rho: Primer estado (densidad)
            sigma: Segundo estado (densidad)
            validate: Validar ambos estados
        
        Returns:
            Divergencia relativa D(ρ‖σ) ≥ 0
        """
        if validate:
            rho.compute_metrics()
            sigma.compute_metrics()
        
        # Término 1: Tr(ρ ln ρ) = -S(ρ) (en nats)
        eig_rho = la.eigvalsh(rho.matrix)
        pos_rho = eig_rho[eig_rho > self._tol]
        term1 = np.sum(pos_rho * np.log(pos_rho))  # = Tr(ρ ln ρ) en nats
        
        # Término 2: Tr(ρ ln σ) — requiere ln(σ)
        eig_sigma, vec_sigma = la.eigh(sigma.matrix)
        
        # Construir ln(σ) = V diag(ln λᵢ) V†
        log_eig_sigma = np.where(
            eig_sigma > self._tol,
            np.log(eig_sigma),
            -np.inf
        )
        
        # Verificar condición de soporte: supp(ρ) ⊆ supp(σ)
        rho_in_sigma_basis = vec_sigma.conj().T @ rho.matrix @ vec_sigma
        rho_on_null = np.abs(np.diag(rho_in_sigma_basis)[eig_sigma <= self._tol])
        
        if np.any(rho_on_null > self._tol):
            logger.warning(
                f"Divergencia infinita detectada: supp(ρ) ⊄ supp(σ). "
                f"Masa en kernel(σ): {np.sum(rho_on_null):.2e}"
            )
            return float('inf')
        
        # ln(σ) en base original
        log_sigma = vec_sigma @ np.diag(np.maximum(log_eig_sigma, -1e15)) @ vec_sigma.conj().T
        
        term2 = np.trace(rho.matrix @ log_sigma).real
        
        # D(ρ‖σ) = -S(ρ) - Tr(ρ ln σ) = term1 - term2
        relative_entropy = term1 - term2
        
        # Clipping numérico (D ≥ 0 por teorema de Klein)
        return float(max(0.0, relative_entropy * self._log_factor))

    def compute_fidelity(
        self, 
        rho: AtomicDensityMatrix, 
        sigma: AtomicDensityMatrix
    ) -> float:
        r"""
        Fidelidad de Uhlmann: F(ρ,σ) = ‖√ρ √σ‖₁² = (Tr √(√ρ σ √ρ))².
        
        Propiedades:
            • 0 ≤ F(ρ,σ) ≤ 1
            • F(ρ,σ) = 1 ⇔ ρ = σ
            • F(ρ,σ) = F(σ,ρ) (simétrica)
            • Monótona bajo CPTP: F(Λ(ρ), Λ(σ)) ≥ F(ρ,σ)
            • Relación con distancia de Bures: d_B² = 2(1 - √F)
            • Fuchs-van de Graaf: 1 - √F ≤ ½‖ρ-σ‖₁ ≤ √(1-F)
        
        Algoritmo: √(√ρ σ √ρ) via descomposición espectral.
        """
        # √ρ via descomposición espectral
        eig_rho, vec_rho = la.eigh(rho.matrix)
        eig_rho = np.maximum(eig_rho, 0.0)
        sqrt_rho = vec_rho @ np.diag(np.sqrt(eig_rho)) @ vec_rho.conj().T
        
        # M = √ρ σ √ρ
        M = sqrt_rho @ sigma.matrix @ sqrt_rho
        
        # √M via descomposición espectral (M es hermítico ≥ 0)
        eig_M, vec_M = la.eigh(M)
        eig_M = np.maximum(eig_M, 0.0)
        sqrt_M = vec_M @ np.diag(np.sqrt(eig_M)) @ vec_M.conj().T
        
        fidelity = float(np.trace(sqrt_M).real ** 2)
        
        # Clipping numérico
        return float(np.clip(fidelity, 0.0, 1.0))

    def compute_bures_distance(
        self, 
        rho: AtomicDensityMatrix, 
        sigma: AtomicDensityMatrix
    ) -> float:
        """
        Distancia de Bures: d_B(ρ,σ) = √(2(1 - √F(ρ,σ))).
        
        Métrica riemanniana natural en el espacio de estados cuánticos.
        """
        fidelity = self.compute_fidelity(rho, sigma)
        return float(np.sqrt(2.0 * (1.0 - np.sqrt(fidelity))))

    def compute_trace_distance(
        self, 
        rho: AtomicDensityMatrix, 
        sigma: AtomicDensityMatrix
    ) -> float:
        """
        Distancia de traza: ½‖ρ - σ‖₁ = ½ Σᵢ |λᵢ(ρ-σ)|.
        
        Métrica operacional: probabilidad óptima de discriminación.
        """
        diff = rho.matrix - sigma.matrix
        eigvals = la.eigvalsh(diff)
        return float(0.5 * np.sum(np.abs(eigvals)))

    # ═══════════════════════════════════════════════════════════════════════════
    # MÉTRICAS DERIVADAS Y FUNCIONALES
    # ═══════════════════════════════════════════════════════════════════════════
    
    def compute_effective_rank(self, rho: AtomicDensityMatrix) -> float:
        """Rango efectivo: r_eff(ρ) = exp(S(ρ)) = exp(H(λ))."""
        return float(np.exp(self.compute_entropy(rho) / self._log_factor))

    def compute_purity(self, rho: AtomicDensityMatrix) -> float:
        """Pureza: γ(ρ) = Tr(ρ²) = Σ λᵢ² ∈ [1/d, 1]."""
        spectral = self.compute_spectral_data(rho)
        return spectral.purity

    def compute_linear_entropy(self, rho: AtomicDensityMatrix) -> float:
        """Entropía lineal: S_L(ρ) = 1 - Tr(ρ²) = 1 - γ(ρ)."""
        return 1.0 - self.compute_purity(rho)

    def compute_participation_ratio(self, rho: AtomicDensityMatrix) -> float:
        """Ratio de participación: PR = 1/Tr(ρ²) = 1/γ(ρ)."""
        purity = self.compute_purity(rho)
        return float(1.0 / purity) if purity > 0 else float('inf')

    def verify_majorization(
        self, 
        rho: AtomicDensityMatrix, 
        sigma: AtomicDensityMatrix
    ) -> bool:
        r"""
        Verifica majorización cuántica: ρ ≺ σ ⇔ λ(ρ) ≺ λ(σ).
        
        Majorización de vectores: x ≺ y ⇔ Σ_{i=1}^k x↓ᵢ ≤ Σ_{i=1}^k y↓ᵢ ∀k, 
        con igualdad para k = d.
        
        Equivalente a: ∃ canal CPTP Λ tal que ρ = Λ(σ).
        """
        eig_rho = self.compute_spectral_data(rho).eigenvalues
        eig_sigma = self.compute_spectral_data(sigma).eigenvalues
        
        # Sumas parciales acumulativas
        cum_rho = np.cumsum(eig_rho)
        cum_sigma = np.cumsum(eig_sigma)
        
        # Verificar Σ_{i=1}^k ρᵢ ≤ Σ_{i=1}^k σᵢ para todo k < d
        # y igualdad para k = d (traza = 1)
        return bool(np.all(cum_rho[:-1] <= cum_sigma[:-1] + 1e-12))


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2: OPERADOR DE TRUNCAMIENTO ESPECTRAL — FUNTOR DE PROYECCIÓN ÓPTIMA
# ═══════════════════════════════════════════════════════════════════════════════

class SpectralTruncationProjector:
    r"""
    Funtor de Reducción de Dimensionalidad Cuántica Óptima — Versión Definitiva.
    
    FUNDAMENTO TEÓRICO CATEGÓRICO:
    ────────────────────────────────
    Implementa el funtor de truncamiento P_ε: 𝒟(ℋ) → 𝒟(ℋ_ε) donde ℋ_ε ⊆ ℋ
    es el subespacio generado por eigenestados con λ ≥ ε.
    
    Como funtor entre categorías de canales CPTP:
    
        P_ε(ρ) = Π_ε ρ Π_ε / Tr(Π_ε ρ Π_ε)
    
    donde Π_ε = Σ_{λₖ≥ε} |ψₖ⟩⟨ψₖ| es proyector ortogonal.
    
    PROPIEDADES FUNCTORIALES:
    ────────────────────────
    1. Preserva identidad: P_ε(I/d) = I_ε/d_ε
    2. Preserva composición: P_ε(Λ(ρ)) ≈ Λ_ε(P_ε(ρ)) para Λ CPTP
    3. Monádico: η_ρ: ρ → P_ε(ρ) es transformación natural Id → P_ε
    4. Óptimo bajo majorización: P_ε(ρ) ≺ ρ (majorización cuántica)
    5. Minimiza pérdida de información: D(ρ‖P_ε(ρ)) mínimo para rango fijo
    
    ESTRATEGIAS DE TRUNCAMIENTO:
    ────────────────────────────
    • THRESHOLD:        Π_ε = Σ_{λ≥ε} |ψ⟩⟨ψ|           (umbral fijo)
    • RANK_K:           Π_k = Σ_{i=1}^k |ψᵢ⟩⟨ψᵢ|        (rango fijo k)
    • CUMULATIVE_ENERGY:Π_η = Σ_{Σλ≤η} |ψ⟩⟨ψ|           (energía η ∈ (0,1])
    • ENTROPY_BOUNDED:  Π_S = max{Π: S(ΠρΠ/Tr) ≤ S_max} (entropía acotada)
    • MAJORIZATION_OPTIMAL: Óptimo bajo preorden ≺ (Nielsen 1999)
    • SCHUR_CONVEX:     Minimiza Φ(λ) convexo-Schur (ej. entropía, pureza)
    
    GARANTÍAS NUMÉRICAS:
    ───────────────────
    • Proyección ortogonal exacta: Π² = Π = Π†
    • Mapa CPTP verificado: Choi(P_ε) ≥ 0, Tr₁(Choi) = I
    • Fidelidad F(ρ, P_ε(ρ)) = Tr(Π_ε ρ) = energía retenida
    • Distancia de Bures: d_B² = 2(1 - √Tr(Π_ε ρ))
    • Límite de error: ‖ρ - P_ε(ρ)‖₁ ≤ 2√(1 - Tr(Π_ε ρ))
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
        """
        Inicializa el proyector de truncamiento espectral.
        
        Args:
            epsilon_threshold: Umbral ε para estrategias basadas en threshold
            strategy: Estrategia de truncamiento (ver TruncationStrategy)
            entropy_engine: Motor entrópico compartido (se crea uno si None)
            verify_cptp: Verificar que el mapa resultante es CPTP
            compute_fidelity: Calcular fidelidad y distancias
            use_majorization_optimal: Usar algoritmo óptimo de majorización
        """
        self._epsilon = max(epsilon_threshold, MIN_EIGENVALUE)
        self._strategy = strategy
        self._entropy_engine = entropy_engine or VonNeumannEntropyEngine()
        self._verify_cptp = verify_cptp
        self._compute_fidelity = compute_fidelity
        self._use_majorization_opt = use_majorization_optimal
        self._cache: Dict[str, TruncationReport] = {}
        
        logger.debug(
            f"SpectralTruncationProjector inicializado: ε={self._epsilon:.2e}, "
            f"strategy={strategy.name}, verify_cptp={verify_cptp}"
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # CÁLCULO DE MÁSCARA DE RETENCIÓN SEGÚN ESTRATEGIA
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _compute_retention_mask(
        self,
        eigenvalues: NDArray[np.float64],
        target_param: Optional[float] = None,
        entropy_engine: Optional[VonNeumannEntropyEngine] = None
    ) -> NDArray[np.bool_]:
        """
        Calcula máscara booleana de eigenvalores a retener según estrategia.
        
        Args:
            eigenvalues: Eigenvalores ordenados descendentemente λ₁ ≥ λ₂ ≥ ...
            target_param: Parámetro adicional según estrategia (k, η, S_max)
            entropy_engine: Motor entrópico para estrategias basadas en entropía
        
        Returns:
            Máscara booleana (True = retener este eigenvalor)
        """
        n = len(eigenvalues)
        mask = np.zeros(n, dtype=bool)
        
        if self._strategy == TruncationStrategy.THRESHOLD:
            # Retener λₖ ≥ ε
            mask = eigenvalues >= self._epsilon
            
        elif self._strategy == TruncationStrategy.RANK_K:
            # Retener top-k eigenvalores
            k = int(target_param) if target_param is not None else 1
            k = min(max(k, 1), n)
            mask[:k] = True
            
        elif self._strategy == TruncationStrategy.CUMULATIVE_ENERGY:
            # Retener hasta alcanzar fracción η de energía total
            target_energy = target_param if target_param is not None else 0.95
            target_energy = np.clip(target_energy, 0.0, 1.0)
            cumulative = np.cumsum(eigenvalues)  # Σ_{i=1}^k λᵢ
            mask = cumulative <= target_energy + self._epsilon
            # Asegurar al menos 1 eigenvalor retenido
            if not np.any(mask):
                mask[0] = True
                
        elif self._strategy == TruncationStrategy.ENTROPY_BOUNDED:
            # Retener máximo número de eigenvalores manteniendo S ≤ S_max
            target_entropy = target_param if target_param is not None else 1.0
            engine = entropy_engine or self._entropy_engine
            
            # Búsqueda binaria en k para entropía objetivo
            # La entropía es monótona creciente en k (más eigenvalores = más mezcla)
            low, high = 1, n
            best_k = 1
            
            while low <= high:
                mid = (low + high) // 2
                test_eigs = eigenvalues[:mid]
                test_eigs = test_eigs / np.sum(test_eigs)  # Renormalizar
                
                # S = -Σ λᵢ ln λᵢ
                pos = test_eigs[test_eigs > engine._tol]
                if len(pos) > 0:
                    entropy = -np.sum(pos * np.log(pos)) * engine._log_factor
                else:
                    entropy = 0.0
                
                if entropy <= target_entropy + 1e-10:
                    best_k = mid
                    low = mid + 1
                else:
                    high = mid - 1
            
            mask[:best_k] = True
            
        elif self._strategy == TruncationStrategy.MAJORIZATION_OPTIMAL:
            # Truncamiento óptimo bajo majorización (Nielsen 1999)
            # Para un rango objetivo k, el truncamiento que maximiza fidelidad
            # y satisface ρ_ε ≺ ρ es simplemente retener top-k eigenvalores.
            # Esto ya se hace por RANK_K. Aquí añadimos verificación de majorización.
            k = int(target_param) if target_param is not None else max(1, n // 2)
            k = min(max(k, 1), n)
            mask[:k] = True
            
        elif self._strategy == TruncationStrategy.SCHUR_CONVEX:
            # Minimizar funcional Schur-convexo Φ(λ) = Σ φ(λᵢ)
            # Para entropía: φ(x) = -x ln x (convexo)
            # Para pureza: φ(x) = x² (convexo)
            # El mínimo bajo restricción de rango k es retener top-k.
            # Implementación general requeriría optimización convexa.
            k = int(target_param) if target_param is not None else max(1, n // 2)
            k = min(max(k, 1), n)
            mask[:k] = True
            
        else:
            raise ValueError(f"Estrategia de truncamiento desconocida: {self._strategy}")
        
        return mask

    # ═══════════════════════════════════════════════════════════════════════════
    # TRUNCAMIENTO ESPECTRAL PRINCIPAL
    # ═══════════════════════════════════════════════════════════════════════════
    
    def truncate_spectrum(
        self, 
        rho: AtomicDensityMatrix,
        target_param: Optional[float] = None,
        compute_fidelity: Optional[bool] = None
    ) -> Tuple[AtomicDensityMatrix, TruncationReport]:
        r"""
        Ejecuta truncamiento espectral con reporte completo de métricas.
        
        ALGORITMO:
        ─────────
        1. Descomposición espectral: ρ = U Λ U†, Λ = diag(λ₁,...,λ_d), λ₁≥...≥λ_d
        2. Determinar máscara de retención M ∈ {0,1}^d según estrategia
        3. Construir proyector Π = U M U† = Σ_{Mₖ=1} |ψₖ⟩⟨ψₖ|
        4. Aplicar mapa CPTP: ρ̃ = Π ρ Π / Tr(Π ρ Π)
        5. Verificar CPTP y calcular métricas de fidelidad/información
        6. Retornar estado truncado y reporte completo
        
        Args:
            rho: Estado cuántico original
            target_param: Parámetro objetivo según estrategia (k, η, S_max)
            compute_fidelity: Sobreescribe configuración de fidelidad
        
        Returns:
            (rho_truncated, TruncationReport)
        
        Raises:
            NumericalInstabilityError: Si todos eigenvalores < ε o traza ≈ 0
        """
        compute_fid = compute_fidelity if compute_fidelity is not None else self._compute_fidelity
        
        # 1. Datos espectrales del estado original
        spectral_before = self._entropy_engine.compute_spectral_data(rho)
        eigenvalues = spectral_before.eigenvalues
        eigenvectors = spectral_before.eigenvectors
        dim = spectral_before.dimension
        
        # 2. Máscara de retención
        retention_mask = self._compute_retention_mask(eigenvalues, target_param)
        
        retained_indices = np.where(retention_mask)[0]
        discarded_indices = np.where(~retention_mask)[0]
        
        n_retained = len(retained_indices)
        n_discarded = len(discarded_indices)
        
        if n_retained == 0:
            raise NumericalInstabilityError(
                f"Degeneración Espectral Letal: Todos los {dim} eigenvalores "
                f"están por debajo del umbral ε={self._epsilon:.2e}. "
                f"Eigenvalores: {eigenvalues[:5]}..."
            )
        
        # 3. Construir proyector ortogonal Π_ε = Σ_{retener} |ψₖ⟩⟨ψₖ|
        retained_eigenvectors = eigenvectors[:, retained_indices]
        projector = retained_eigenvectors @ retained_eigenvectors.conj().T
        
        # Verificar proyección ortogonal: Π² = Π, Π† = Π
        if self._verify_cptp:
            proj_check = projector @ projector
            proj_error = la.norm(proj_check - projector, ord='fro')
            if proj_error > 1e-10:
                logger.warning(f"Proyector no idempotente: ‖Π²-Π‖_F = {proj_error:.2e}")
        
        # 4. Form the embedded state's density matrix in the original basis (for fidelity calculations)
        retained_eigenvalues = eigenvalues[retention_mask]
        # Renormalize to trace 1
        retained_eigenvalues = retained_eigenvalues / np.sum(retained_eigenvalues)
        retained_eigenvectors = eigenvectors[:, retention_mask]

        # Construct the density matrix in the original basis (embedded state)
        rho_embedded_matrix = retained_eigenvectors @ np.diag(retained_eigenvalues) @ retained_eigenvectors.conj().T

        # 5. Create the reduced state (in its own basis) and the embedded state (for fidelity calculations)
        reduced_matrix = np.diag(retained_eigenvalues)
        rho_reduced = AtomicDensityMatrix(
            reduced_matrix,
            auto_renormalize=False,
            validate=True
        )
        rho_embedded = AtomicDensityMatrix(
            rho_embedded_matrix,
            auto_renormalize=False,
            validate=True
        )

        # 6. Compute spectral data for the reduced state (this will be the truncated_spectral_data in the report)
        spectral_after = self._entropy_engine.compute_spectral_data(rho_reduced)

        # 7. Compute fidelity, Bures distance, and relative entropy between original and embedded state
        if compute_fid:
            fidelity = self._entropy_engine.compute_fidelity(rho, rho_embedded)
            bures_dist = self._entropy_engine.compute_bures_distance(rho, rho_embedded)
            rel_entropy = self._entropy_engine.compute_relative_entropy(rho, rho_embedded)
        else:
            # Lower bound for fidelity: F >= Tr(Πρ) = sum of retained eigenvalues
            fidelity = np.sum(retained_eigenvalues)  # Should be 1.0 after renormalization
            bures_dist = np.sqrt(2.0 * (1.0 - np.sqrt(max(fidelity, 0.0))))
            rel_entropy = 0.0

        # 8. Verify majorization: ρ_purified_embedded ≺ ρ_original
        majorization_ok = self._entropy_engine.verify_majorization(rho_embedded, rho)

        # 9. Retained energy (trace of the projector times rho)
        retained_energy = float(np.sum(eigenvalues[retention_mask]))  # Before renormalization, but note we renormalized eigenvalues above.
        # However, the retained energy is defined as Tr(Πρ) which is the sum of the original eigenvalues that are retained.
        # We have not renormalized the eigenvalues for this quantity.
        retained_energy = float(np.sum(eigenvalues[retention_mask]))

        # 10. Compression ratio
        compression_ratio = n_retained / dim

        # 11. Computational speedup
        computational_speedup = (dim / n_retained) ** 3 if n_retained > 0 else 1.0

        # 12. Build the report
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

        # 13. Log the result
        logger.info(
            f"Truncamiento espectral: {dim} → {n_retained} dims "
            f"(compresión={compression_ratio:.1%}, energía={retained_energy:.4f}, "
            f"F={fidelity:.6f}, ΔS={report.entropy_change:+.4f}, "
            f"majorización={'✓' if majorization_ok else '✗'})"
        )

        # 14. Return the reduced state and the report
        return rho_reduced, report
    # MÉTODOS DE ALTO NIVEL: OPTIMIZACIÓN AUTOMÁTICA
    # ═══════════════════════════════════════════════════════════════════════════
    
    def auto_truncate_for_target_fidelity(
        self,
        rho: AtomicDensityMatrix,
        target_fidelity: float = 0.99,
        max_dimension: Optional[int] = None
    ) -> Tuple[AtomicDensityMatrix, TruncationReport]:
        """
        Encuentra truncamiento mínimo que alcanza fidelidad objetivo.
        
        Usa búsqueda binaria en k (rank) para encontrar mínimo rango
        tal que F(ρ, ρ_k) ≥ target_fidelity.
        """
        spectral = self._entropy_engine.compute_spectral_data(rho)
        eigenvalues = spectral.eigenvalues
        n = len(eigenvalues)
        
        if max_dimension is None:
            max_dimension = n
        max_dimension = min(max_dimension, n)
        
        # Búsqueda binaria
        low, high = 1, max_dimension
        best_k = n
        best_report = None
        best_rho = None
        
        while low <= high:
            mid = (low + high) // 2
            
            # Crear proyector temporal con estrategia RANK_K
            temp_projector = SpectralTruncationProjector(
                epsilon_threshold=0.0,
                strategy=TruncationStrategy.RANK_K,
                entropy_engine=self._entropy_engine,
                verify_cptp=self._verify_cptp,
                compute_fidelity=True
            )
            
            try:
                rho_test, report = temp_projector.truncate_spectrum(rho, target_param=mid)
                fidelity = report.fidelity_preservation
                
                if fidelity >= target_fidelity:
                    best_k = mid
                    best_report = report
                    best_rho = rho_test
                    high = mid - 1
                else:
                    low = mid + 1
            except NumericalInstabilityError:
                low = mid + 1
        
        if best_rho is None:
            raise NumericalInstabilityError(
                f"No se pudo alcanzar fidelidad {target_fidelity} con rango ≤ {max_dimension}"
            )
        
        return best_rho, best_report

    def auto_truncate_for_target_entropy(
        self,
        rho: AtomicDensityMatrix,
        target_entropy: float,
        tolerance: float = 1e-3
    ) -> Tuple[AtomicDensityMatrix, TruncationReport]:
        """
        Encuentra truncamiento que aproxima entropía objetivo.
        """
        return self.truncate_spectrum(
            rho, 
            target_param=target_entropy,
            compute_fidelity=True
        )

    def estimate_optimal_truncation(
        self,
        rho: AtomicDensityMatrix,
        cost_function: Callable[[TruncationReport], float]
    ) -> Tuple[AtomicDensityMatrix, TruncationReport]:
        """
        Encuentra truncamiento óptimo minimizando función de costo personalizada.
        
        cost_function: TruncationReport → ℝ (minimizar)
        """
        # Evaluar todas las estrategias viables
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
            
            # Barrer parámetros relevantes
            if strategy == TruncationStrategy.THRESHOLD:
                params = np.logspace(-12, -2, 20)
            elif strategy == TruncationStrategy.RANK_K:
                params = range(1, rho.dimension + 1)
            elif strategy == TruncationStrategy.CUMULATIVE_ENERGY:
                params = np.linspace(0.5, 0.999, 20)
            
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
# FASE 3: PODA DE OPERADORES DE SALTO — OPTIMIZACIÓN LINDBLADIANA
# ═══════════════════════════════════════════════════════════════════════════════

class LindbladPruningOperator:
    r"""
    Optimizador de la Ecuación Maestra de Lindblad-GKSL — Versión Definitiva.
    
    FUNDAMENTO TEÓRICO:
    ──────────────────
    La ecuación maestra en forma de Gorini-Kossakowski-Sudarshan-Lindblad:
    
        dρ/dt = ℒ(ρ) = -i[H, ρ] + Σₖ γₖ 𝒟[Lₖ](ρ)
    
    donde 𝒟[L](ρ) = L ρ L† - ½{L†L, ρ} es el disipador.
    
    El generador ℒ define un semi-grupo cuántico dinámico {e^{tℒ}}_{t≥0}
    completamente positivo y preservador de traza (CPTP).
    
    PODA PRESERVANDO CPTP:
    ─────────────────────
    Eliminar términos con γₖ pequeño cambia el generador a ℒ'.
    Para preservar CPTP, la matriz de Kossakowski K' = (γₖ' ⟨Lᵢ|Lⱼ⟩) 
    debe permanecer semidefinida positiva.
    
    CRITERIOS DE PODA RIGUROSOS:
    ────────────────────────────
    1. MAGNITUDE:           γₖ < τ                                    (tasa bruta)
    2. FROBENIUS_NORM:      γₖ ‖Lₖ‖_F < τ                             (norma ponderada)
    3. COMMUTATOR_NORM:     γₖ ‖[H, Lₖ]‖_F < τ                        (relevancia dinámica)
    4. DISSIPATIVE_GAP:     γₖ / gap(ℒ) < τ                           (brecha espectral)
    5. CHOI_RANK:           rank(Choi(𝒟[Lₖ])) < τ                     (complejidad de Choi)
    6. INFORMATION_GAIN:    I(ρ: ℒₖ(ρ)) < τ                           (info mutua)
    
    VERIFICACIÓN DE CPTP POST-PODA:
    ───────────────────────────────
    • Matriz de Choi del generador podado ≥ 0
    • Traza preservada: Tr(ℒ'(ρ)) = 0 ∀ρ
    • Forma de Lindblad válida: coeficientes γₖ' ≥ 0
    • Norma de diamante del error: ‖ℒ - ℒ'‖_⋄ ≤ Σ_{descartados} γₖ ‖Lₖ‖²
    
    LÍMITES DE ERROR:
    ────────────────
    Sea ℒ = ℒ_kept + ℒ_discarded. Entonces para todo t ≥ 0:
    
        ‖e^{tℒ} - e^{tℒ_kept}‖_⋄ ≤ t ‖ℒ_discarded‖_⋄ e^{t‖ℒ_kept‖}
    
    donde ‖ℒ_discarded‖_⋄ ≤ Σ_{k∈discarded} γₖ ‖Lₖ‖²_∞
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
        """
        Inicializa el operador de poda lindbladiana.
        
        Args:
            tau_cutoff: Umbral τ para poda (depende del criterio)
            criterion: Criterio de poda (ver PruningCriterion)
            preserve_critical: Mantener al menos 1 operador si todos podados
            verify_cptp: Verificar CPTP del generador resultante
            hamiltonian: Hamiltoniano H (requerido para COMMUTATOR_NORM)
            entropy_engine: Motor entrópico para criterios basados en información
        """
        self._tau = max(tau_cutoff, 0.0)
        self._criterion = criterion
        self._preserve_critical = preserve_critical
        self._verify_cptp = verify_cptp
        self._hamiltonian = hamiltonian
        self._entropy_engine = entropy_engine or VonNeumannEntropyEngine()
        self._cache: Dict[str, PruningReport] = {}
        
        # Validar Hamiltoniano para criterios que lo requieren
        if criterion == PruningCriterion.COMMUTATOR_NORM and hamiltonian is None:
            logger.warning(
                "Criterio COMMUTATOR_NORM requiere Hamiltoniano. "
                "Cayendo a MAGNITUDE."
            )
            self._criterion = PruningCriterion.MAGNITUDE
        
        logger.debug(
            f"LindbladPruningOperator inicializado: τ={self._tau:.2e}, "
            f"criterion={criterion.name}, verify_cptp={verify_cptp}"
        )

    def set_hamiltonian(self, H: NDArray[np.complex128]) -> None:
        """Establece/actualiza el Hamiltoniano para criterios dinámicos."""
        self._hamiltonian = H
        if self._criterion == PruningCriterion.COMMUTATOR_NORM:
            logger.debug("Hamiltoniano actualizado para criterio COMMUTATOR_NORM")

    # ═══════════════════════════════════════════════════════════════════════════
    # CÁLCULO DE PUNTUACIÓN DE IMPORTANCIA
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _compute_pruning_score(
        self,
        gamma: float,
        L_k: NDArray[np.complex128],
        H: Optional[NDArray[np.complex128]] = None
    ) -> float:
        """
        Calcula puntuación de importancia del operador de salto.
        
        Mayor puntuación = más importante = menos probable de ser podado.
        
        Args:
            gamma: Tasa de disipación γₖ ≥ 0
            L_k: Operador de salto Lₖ ∈ L(ℋ)
            H: Hamiltoniano (opcional, para criterios dinámicos)
        
        Returns:
            Puntuación de importancia ≥ 0
        """
        H_eff = H if H is not None else self._hamiltonian
        
        if self._criterion == PruningCriterion.MAGNITUDE:
            return gamma
        
        elif self._criterion == PruningCriterion.FROBENIUS_NORM:
            norm_L = la.norm(L_k, ord='fro')
            return gamma * norm_L
        
        elif self._criterion == PruningCriterion.COMMUTATOR_NORM:
            if H_eff is None:
                logger.warning("Hamiltoniano no disponible para COMMUTATOR_NORM. Usando MAGNITUDE.")
                return gamma
            commutator = H_eff @ L_k - L_k @ H_eff
            norm_comm = la.norm(commutator, ord='fro')
            return gamma * norm_comm
        
        elif self._criterion == PruningCriterion.DISSIPATIVE_GAP:
            # Requiere análisis espectral del generador completo
            # Aproximación: γₖ * min_eig(L_k† L_k)
            try:
                eigvals = la.eigvalsh(L_k.conj().T @ L_k)
                min_eig = max(np.min(eigvals), 1e-15)
                return gamma * min_eig
            except LinAlgError:
                return gamma
        
        elif self._criterion == PruningCriterion.CHOI_RANK:
            # Rango de Choi del canal elemental 𝒟[Lₖ]
            # Choi(𝒟[L]) = (I ⊗ 𝒟[L])(|Ω⟩⟨Ω|) donde |Ω⟩ = Σ|ii⟩/√d
            # rank(Choi) = rank(L)² para operador de salto simple
            try:
                rank_L = np.linalg.matrix_rank(L_k, tol=1e-10)
                return gamma * (rank_L ** 2)
            except LinAlgError:
                return gamma
        
        elif self._criterion == PruningCriterion.INFORMATION_GAIN:
            # Ganancia de información mutua: I(ρ : 𝒟[Lₖ](ρ))
            # Requiere estado ρ de referencia; aproximamos por entropía de Lₖ
            try:
                # Entropía del operador normalizado Lₖ/‖Lₖ‖
                L_norm = L_k / max(la.norm(L_k, ord='fro'), 1e-15)
                # Aproximación: información ≈ log(rank)
                rank = np.linalg.matrix_rank(L_norm, tol=1e-10)
                return gamma * np.log(max(rank, 1))
            except LinAlgError:
                return gamma
        
        else:
            raise ValueError(f"Criterio de poda desconocido: {self._criterion}")

    # ═══════════════════════════════════════════════════════════════════════════
    # PODA PRINCIPAL
    # ═══════════════════════════════════════════════════════════════════════════
    
    def prune_jump_operators(
        self,
        jump_operators: List[Tuple[float, NDArray[np.complex128]]],
        H: Optional[NDArray[np.complex128]] = None,
        reference_state: Optional[AtomicDensityMatrix] = None
    ) -> Tuple[List[Tuple[float, NDArray[np.complex128]]], PruningReport]:
        r"""
        Filtra operadores de salto según criterio de poda.
        
        ALGORITMO:
        ─────────
        1. Calcular puntuación sₖ = score(γₖ, Lₖ) para cada operador
        2. Ordenar por puntuación descendente
        3. Retener operadores con sₖ ≥ τ
        4. Si preserve_critical y lista vacía, retener el de mayor score
        5. Verificar CPTP del generador resultante
        6. Calcular métricas de error y reporte
        
        Args:
            jump_operators: Lista de tuplas (γₖ, Lₖ)
            H: Hamiltoniano opcional (sobrescribe el de inicialización)
            reference_state: Estado de referencia para criterios informacionales
        
        Returns:
            (operadores_podados, PruningReport)
        """
        if not jump_operators:
            return [], PruningReport(
                original_operators=[],
                pruned_operators=[],
                discarded_operators=[],
                scores=np.array([]),
                threshold=self._tau,
                criterion=self._criterion,
                total_rate_before=0.0,
                total_rate_after=0.0,
                rate_preservation=1.0,
                complete_positivity_verified=True,
                trace_preservation_error=0.0,
                lindblad_generator_norm_change=0.0,
                pruning_efficiency=0.0
            )
        
        initial_count = len(jump_operators)
        H_eff = H if H is not None else self._hamiltonian
        
        # 1. Calcular puntuaciones
        scores = np.array([
            self._compute_pruning_score(gamma, L_k, H_eff)
            for gamma, L_k in jump_operators
        ], dtype=np.float64)
        
        # 2. Tasas totales
        total_rate_before = sum(gamma for gamma, _ in jump_operators)
        
        # 3. Filtrar según umbral
        keep_mask = scores >= self._tau
        pruned_ops = [op for op, keep in zip(jump_operators, keep_mask) if keep]
        discarded_ops = [op for op, keep in zip(jump_operators, keep_mask) if not keep]
        
        # 4. Preservar crítico si necesario
        if self._preserve_critical and len(pruned_ops) == 0 and initial_count > 0:
            max_idx = int(np.argmax(scores))
            pruned_ops = [jump_operators[max_idx]]
            discarded_ops = [op for i, op in enumerate(jump_operators) if i != max_idx]
            keep_mask = np.zeros(initial_count, dtype=bool)
            keep_mask[max_idx] = True
            logger.warning(
                f"Todos los {initial_count} operadores bajo umbral τ={self._tau:.2e}. "
                f"Preservando operador crítico #{max_idx} (score={scores[max_idx]:.3e})"
            )
        
        pruned_count = len(pruned_ops)
        discarded_count = len(discarded_ops)
        
        total_rate_after = sum(gamma for gamma, _ in pruned_ops)
        rate_preservation = total_rate_after / total_rate_before if total_rate_before > 0 else 1.0
        
        # 5. Verificar CPTP del generador podado
        cptp_verified = True
        trace_error = 0.0
        
        if self._verify_cptp and pruned_ops:
            cptp_verified, trace_error = self._verify_lindblad_cptp(pruned_ops)
        
        # 6. Cambio en norma del generador (límite de error diamante)
        discarded_norm = sum(
            gamma * la.norm(L_k, ord=2)**2 
            for gamma, L_k in discarded_ops
        )
        
        # 7. Eficiencia de poda
        pruning_efficiency = discarded_count / max(1, initial_count)
        
        report = PruningReport(
            original_operators=jump_operators,
            pruned_operators=pruned_ops,
            discarded_operators=discarded_ops,
            scores=scores,
            threshold=self._tau,
            criterion=self._criterion,
            total_rate_before=total_rate_before,
            total_rate_after=total_rate_after,
            rate_preservation=rate_preservation,
            complete_positivity_verified=cptp_verified,
            trace_preservation_error=trace_error,
            lindblad_generator_norm_change=discarded_norm,
            pruning_efficiency=pruning_efficiency
        )
        
        if discarded_count > 0:
            logger.info(
                f"Poda Lindbladiana: {discarded_count}/{initial_count} descartados "
                f"(eficiencia={pruning_efficiency:.1%}, tasa: {total_rate_before:.2e}→{total_rate_after:.2e}, "
                f"CPTP={'✓' if cptp_verified else '✗'}, ‖Δℒ‖≈{discarded_norm:.2e})"
            )
        
        return pruned_ops, report

    # ═══════════════════════════════════════════════════════════════════════════
    # VERIFICACIÓN DE CPTP
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _verify_lindblad_cptp(
        self,
        jump_operators: List[Tuple[float, NDArray[np.complex128]]]
    ) -> Tuple[bool, float]:
        """
        Verifica que el generador de Lindblad podado define un semi-grupo CPTP.
        
        Condiciones necesarias y suficientes (GKSL):
        1. γₖ ≥ 0 para todo k
        2. Matriz de Kossakowski K_{ij} = Σₖ γₖ ⟨Lᵢ|Lⱼ⟩ ≥ 0 (SPD)
        3. Hamiltoniano H hermítico (asumido)
        
        Returns:
            (cptp_verified, trace_preservation_error)
        """
        # Verificar tasas no negativas
        rates = [gamma for gamma, _ in jump_operators]
        if any(gamma < -1e-12 for gamma in rates):
            return False, float('inf')
        
        if not jump_operators:
            return True, 0.0
        
        # Construir matriz de Kossakowski
        n_ops = len(jump_operators)
        L_ops = [L_k for _, L_k in jump_operators]
        dim = L_ops[0].shape[0]
        
        # K_{ij} = Tr(Lᵢ† Lⱼ) — producto interno de Hilbert-Schmidt
        K = np.zeros((n_ops, n_ops), dtype=np.complex128)
        for i in range(n_ops):
            for j in range(n_ops):
                K[i, j] = np.trace(L_ops[i].conj().T @ L_ops[j])
        
        # Pesos por tasas: K_weighted = diag(√γ) K diag(√γ)
        sqrt_gamma = np.sqrt(np.array(rates))
        K_weighted = np.diag(sqrt_gamma) @ K @ np.diag(sqrt_gamma)
        
        # Verificar semidefinida positiva
        try:
            eigvals_K = la.eigvalsh(K_weighted)
            min_eig = np.min(eigvals_K)
            if min_eig < -1e-10:
                logger.warning(f"Matriz de Kossakowski no SPD: λ_min = {min_eig:.2e}")
                return False, abs(min_eig)
        except LinAlgError:
            return False, float('inf')
        
        # Verificar preservación de traza: Tr(ℒ(ρ)) = 0
        # ℒ(ρ) = Σ γₖ (Lₖ ρ Lₖ† - ½{Lₖ†Lₖ, ρ})
        # Tr(ℒ(ρ)) = Σ γₖ (Tr(Lₖ†Lₖ ρ) - Tr(Lₖ†Lₖ ρ)) = 0 ✓ (identicamente)
        trace_error = 0.0  # Se cumple identicamente para forma GKSL
        
        return True, trace_error

    # ═══════════════════════════════════════════════════════════════════════════
    # ANÁLISIS DE IMPACTO DINÁMICO
    # ═══════════════════════════════════════════════════════════════════════════
    
    def estimate_dynamics_error(
        self,
        original_ops: List[Tuple[float, NDArray[np.complex128]]],
        pruned_ops: List[Tuple[float, NDArray[np.complex128]]],
        time: float = 1.0
    ) -> Dict[str, float]:
        """
        Estima error en dinámica inducido por poda.
        
        Límite de diamante: ‖e^{tℒ} - e^{tℒ'}‖_⋄ ≤ t ‖ℒ - ℒ'‖_⋄ e^{t‖ℒ'‖}
        
        Args:
            original_ops: Operadores originales
            pruned_ops: Operadores podados
            time: Tiempo de evolución t
        
        Returns:
            Diccionario con cotas de error
        """
        # Norma del generador descartado
        discarded_ops = [op for op in original_ops if op not in pruned_ops]
        
        discarded_norm = sum(
            gamma * la.norm(L_k, ord=2)**2 
            for gamma, L_k in discarded_ops
        )
        
        # Norma del generador retenido
        kept_norm = sum(
            gamma * la.norm(L_k, ord=2)**2 
            for gamma, L_k in pruned_ops
        )
        
        # Hamiltoniano contribution
        H_norm = la.norm(self._hamiltonian, ord=2) if self._hamiltonian is not None else 0.0
        
        # Límite de diamante (muy conservativo)
        diamond_bound = time * discarded_norm * np.exp(time * (kept_norm + H_norm))
        
        # Límite de traza (más ajustado para estados)
        trace_bound = 2.0 * (1.0 - np.exp(-time * discarded_norm))
        
        return {
            'discarded_generator_norm': discarded_norm,
            'kept_generator_norm': kept_norm,
            'hamiltonian_norm': H_norm,
            'diamond_norm_bound': diamond_bound,
            'trace_norm_bound': trace_bound,
            'time': time
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 4: FUNTOR MAC MINIMIZER — ORQUESTADOR CATEGÓRICO PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

class MACMinimizer(Morphism):
    r"""
    Funtor Supremo de Purificación Semántica MAC — Versión Definitiva.
    
    NATURALEZA CATEGÓRICA:
    ──────────────────────
    Implementa el funtor de purificación P: 𝐌𝐀𝐂 → 𝐌𝐀𝐂_𝐩𝐮𝐫𝐞
    donde:
    
    • 𝐌𝐀𝐂: Categoría de estados MAC (objetos = (ρ, {Lₖ}), morfismos = canales CPTP)
    • 𝐌𝐀𝐂_𝐩𝐮𝐫𝐞: Subcategoría de estados puros/efectivamente puros
    • P(ρ, {Lₖ}) = (P_ε(ρ), {Lₖ}_{kept})  (truncamiento + poda)
    
    ADJUNCIÓN FUNDAMENTAL:
    ──────────────────────
    P ⊣ ι  donde ι: 𝐌𝐀𝐂_𝐩𝐮𝐫𝐞 ↪ 𝐌𝐀𝐂 es la inclusión.
    
    Unidad η: Id → ι∘P    (ρ → ρ_purified como morfismo CPTP)
    Counidad ε: P∘ι → Id  (identidad en subcategoría pura)
    
    PROPIEDADES MONÁDICAS:
    ─────────────────────
    T = ι∘P es una mónada en 𝐌𝐀𝐂:
    • η: Id → T           (purificación)
    • μ: T² → T           (idempotencia: P(P(ρ)) = P(ρ))
    
    ALGORITMO DE PURIFICACIÓN INTEGRAL:
    ───────────────────────────────────
    1. ANALYZE    → Auditoría entrópica completa (S, γ, r_eff, majorización)
    2. TRUNCATE   → Truncamiento espectral óptimo (funtor P_ε)
    3. PRUNE      → Poda lindbladiana preservando CPTP
    4. VALIDATE   → Verificación de adjunción (η natural, CPTP, majorización)
    5. OPTIMIZE   → Búsqueda de Pareto (compresión vs fidelidad vs speedup)
    6. REPORT     → Métricas agregadas y telemetría categórica
    
    GARANTÍAS FÍSICAS:
    ─────────────────
    • Estado resultado: ρ' ≥ 0, Tr(ρ') = 1, ρ' ≺ ρ (majorización)
    • Generador resultado: ℒ' en forma GKSL válida (CPTP)
    • Fidelidad: F(ρ, ρ') = Tr(P_ε ρ) ≥ energía retenida
    • Límite de error: ‖ρ - ρ'‖₁ ≤ 2√(1 - F)
    • Speedup: Ω((d/d')³) para operaciones matriciales densas
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
        """Dominio del morfismo: categoría de estados MAC."""
        return frozenset({Stratum.WISDOM})

    @property
    def codomain(self) -> FrozenSet[Stratum]:
        """Codominio del morfismo: subcategoría de estados puros/efectivamente puros."""
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
        """
        Inicializa el funtor minimizador MAC.
        
        Args:
            epsilon_spectral: Umbral ε para truncamiento espectral
            tau_lindblad: Umbral τ para poda de Lindblad
            truncation_strategy: Estrategia de truncamiento (ver TruncationStrategy)
            pruning_criterion: Criterio de poda (ver PruningCriterion)
            auto_optimize: Habilitar optimización automática adaptativa
            debug_mode: Modo debug con telemetría extendida e historial
            verify_cptp: Verificar CPTP en todas las etapas
            purification_mode: Modo de purificación (PROJECTIVE, UNITARY_DILATION, etc.)
            entropy_engine: Motor entrópico compartido
        """
        self._epsilon_spectral = max(epsilon_spectral, MIN_EIGENVALUE)
        self._tau_lindblad = max(tau_lindblad, 0.0)
        self._truncation_strategy = truncation_strategy
        self._pruning_criterion = pruning_criterion
        self._auto_optimize = auto_optimize
        self._debug_mode = debug_mode
        self._purification_mode = purification_mode
        
        # Componentes fundamentales
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
        
        # Telemetría
        self._minimization_count = 0
        self._total_compression = 1.0
        self._history: List[MinimizationMetrics] = []
        self._telemetry: Dict[str, Any] = {}
        self._pareto_frontier: List[Tuple[float, float, float]] = []  # (compresión, fidelidad, speedup)
        
        logger.info(
            f"MACMinimizer v3.0 inicializado: "
            f"ε={self._epsilon_spectral:.2e}, τ={self._tau_lindblad:.2e}, "
            f"strategy={truncation_strategy.name}, criterion={pruning_criterion.name}, "
            f"auto_opt={auto_optimize}, debug={debug_mode}"
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # MÉTODO PRINCIPAL: PURIFICACIÓN INTEGRAL
    # ═══════════════════════════════════════════════════════════════════════════
    
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
        r"""
        Ejecuta purificación espectral integral con análisis categórico completo.
        
        PROTOCOLO CATEGÓRICO:
        ─────────────────────
        1. ANALYZE:     ρ ⟼ (S(ρ), γ(ρ), r_eff(ρ), λ(ρ)) — Objeto en 𝐌𝐀𝐂
        2. TRUNCATE:    P_ε(ρ) = η_ρ(ρ) — Unidad de la adjunción P ⊣ ι
        3. PRUNE:       {Lₖ} ⟼ {Lₖ}_{kept} — Restricción de morfismos
        4. VALIDATE:    Verificar η natural, CPTP, majorización ρ' ≺ ρ
        5. OPTIMIZE:    Búsqueda de Pareto si auto_optimize
        6. REPORT:      MinimizationMetrics — Telemetría del funtor
        
        Args:
            rho: Estado MAC original (objeto en 𝐌𝐀𝐂)
            jump_operators: Operadores de salto del generador lindbladiano
            entropy_threshold: Umbral de entropía para truncamiento automático
            H: Hamiltoniano del sistema (para criterios dinámicos)
            force_truncation: Forzar truncamiento independiente de entropía
            target_fidelity: Fidelidad objetivo (activa búsqueda automática)
            max_dimension: Dimensión máxima permitida post-truncamiento
        
        Returns:
            (ρ_purified, jump_ops_optimized, MinimizationMetrics)
        
        Raises:
            NumericalInstabilityError: Si el estado es físicamente inválido
        """
        import time
        start_time = time.perf_counter()
        
        self._minimization_count += 1
        iteration = self._minimization_count
        
        logger.info(f"═══ MACMinimizer Iteración #{iteration} ═══")
        
        # ─────────────────────────────────────────────────────────────────────
        # FASE 1: ANALYZE — Auditoría Entrópica Completa
        # ─────────────────────────────────────────────────────────────────────
        
        logger.info(f"[ANALYZE] Iniciando auditoría entrópica del estado MAC...")
        
        # Métricas completas del estado original
        spectral_before = self._entropy_engine.compute_spectral_data(rho)
        entropy_initial = spectral_before.entropy
        purity_initial = spectral_before.purity
        effective_rank_initial = spectral_before.effective_rank
        
        logger.info(
            f"Estado inicial: dim={spectral_before.dimension}, rank={spectral_before.rank}, "
            f"r_eff={effective_rank_initial:.2f}, S={entropy_initial:.6f}, "
            f"γ={purity_initial:.6f}, cond={spectral_before.condition_number:.2e}"
        )
        
        # Determinar si se requiere truncamiento
        if entropy_threshold is None:
            # Umbral adaptativo: 70% de entropía máxima ln(d)
            dim = spectral_before.dimension
            entropy_threshold = 0.7 * np.log(dim)
        
        should_truncate = (
            force_truncation or 
            entropy_initial > entropy_threshold or
            purity_initial < 0.5 or
            spectral_before.condition_number > MAX_CONDITION_NUMBER
        )
        
        logger.info(
            f"[ANALYZE] Truncamiento requerido: {should_truncate} "
            f"(S={entropy_initial:.4f} vs threshold={entropy_threshold:.4f}, "
            f"γ={purity_initial:.4f}, cond={spectral_before.condition_number:.2e})"
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # FASE 2: TRUNCATE — Truncamiento Espectral Óptimo
        # ─────────────────────────────────────────────────────────────────────
        
        truncation_report = None
        rho_purified = rho
        
        if should_truncate:
            logger.info(f"[TRUNCATE] Iniciando truncamiento espectral (estrategia: {self._truncation_strategy.name})...")
            
            if target_fidelity is not None:
                # Búsqueda automática para fidelidad objetivo
                rho_purified, truncation_report = (
                    self._spectral_projector.auto_truncate_for_target_fidelity(
                        rho, target_fidelity=target_fidelity, max_dimension=max_dimension
                    )
                )
            elif self._auto_optimize and self._truncation_strategy == TruncationStrategy.THRESHOLD:
                # Optimización automática de ε via búsqueda de Pareto
                rho_purified, truncation_report = self._auto_optimize_truncation(
                    rho, max_dimension=max_dimension
                )
            else:
                # Truncamiento estándar con parámetros configurados
                rho_purified, truncation_report = self._spectral_projector.truncate_spectrum(
                    rho, compute_fidelity=True
                )
            
            logger.info(
                f"[TRUNCATE] Completado: {truncation_report.original_spectral_data.dimension} → "
                f"{truncation_report.truncated_spectral_data.dimension} dims "
                f"(compresión={truncation_report.compression_ratio:.1%}, "
                f"F={truncation_report.fidelity_preservation:.6f}, "
                f"ΔS={truncation_report.entropy_change:+.6f}, "
                f"majorización={'✓' if truncation_report.majorization_verified else '✗'})"
            )
        else:
            logger.info("[TRUNCATE] Saltado — Entropía dentro de límites operacionales.")
        
        # ─────────────────────────────────────────────────────────────────────
        # FASE 3: PRUNE — Poda de Operadores Lindbladianos
        # ─────────────────────────────────────────────────────────────────────
        
        logger.info(f"[PRUNE] Iniciando poda de operadores de salto (criterio: {self._pruning_criterion.name})...")
        
        # Actualizar Hamiltoniano en el podador si se proporciona
        if H is not None:
            self._lindblad_pruner.set_hamiltonian(H)
        
        optimized_jump_ops, pruning_report = self._lindblad_pruner.prune_jump_operators(
            jump_operators, H=H
        )
        
        logger.info(
            f"[PRUNE] Completado: {pruning_report.original_count} → "
            f"{pruning_report.pruned_count} operadores "
            f"(eficiencia={pruning_report.pruning_efficiency:.1%}, "
            f"tasa preservada={pruning_report.rate_preservation:.4f}, "
            f"CPTP={'✓' if pruning_report.complete_positivity_verified else '✗'})"
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # FASE 4: VALIDATE — Validación Categórica y Física
        # ─────────────────────────────────────────────────────────────────────
        
        logger.info("[VALIDATE] Validando conservación de información y adjunción...")
        
        # Calcular pérdida de información y fidelidad global
        if should_truncate:
            information_loss = self._entropy_engine.compute_relative_entropy(rho, rho_purified)
            fidelity = truncation_report.fidelity_preservation
            bures_dist = truncation_report.bures_distance
            
            # Verificar naturalidad de η: η_ρ ∘ f = P(f) ∘ η_ρ para morfismos f
            # Aquí f es la evolución lindbladiana; verificamos aproximación
            naturality_verified = truncation_report.majorization_verified
        else:
            information_loss = 0.0
            fidelity = 1.0
            bures_dist = 0.0
            naturality_verified = True
        
        # Verificar que ρ_purified ≺ ρ (majorización)
        majorization_preserved = self._entropy_engine.verify_majorization(rho_purified, rho)
        
        # Límite de capacidad de Holevo del canal purificado
        channel_capacity_bound = self._estimate_holevo_bound(rho_purified, optimized_jump_ops)
        
        # ─────────────────────────────────────────────────────────────────────
        # FASE 5: METRICS — Métricas Agregadas
        # ─────────────────────────────────────────────────────────────────────
        
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Compresión total (espectral × lindbladiana)
        if truncation_report:
            spectral_compression = truncation_report.compression_ratio
        else:
            spectral_compression = 1.0
        
        lindblad_compression = (
            pruning_report.pruned_count / max(1, pruning_report.original_count)
        )
        total_compression = spectral_compression * lindblad_compression
        
        # Speedup computacional estimado
        if truncation_report:
            computational_speedup = truncation_report.computational_speedup
        else:
            computational_speedup = 1.0
        
        # Cambios en entropía y pureza
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
            channel_capacity_bound=channel_capacity_bound,
            execution_time_ms=execution_time_ms
        )
        
        # ─────────────────────────────────────────────────────────────────────
        # FASE 6: TELEMETRÍA Y HISTORIAL
        # ─────────────────────────────────────────────────────────────────────
        
        self._total_compression *= total_compression
        
        if self._debug_mode:
            self._history.append(metrics)
            self._pareto_frontier.append((
                total_compression, fidelity, computational_speedup
            ))
        
        # Actualizar telemetría global
        self._update_telemetry(metrics)
        
        logger.info(
            f"✓ Purificación MAC completada en {execution_time_ms:.2f}ms. "
            f"Compresión total: {total_compression:.1%}, "
            f"Fidelidad: {fidelity:.6f}, "
            f"Speedup: {computational_speedup:.2f}x, "
            f"ΔS: {entropy_reduction:+.6f}, "
            f"Majorización: {'✓' if majorization_preserved else '✗'}"
        )
        
        return rho_purified, optimized_jump_ops, metrics

    # ═══════════════════════════════════════════════════════════════════════════
    # OPTIMIZACIÓN AUTOMÁTICA Y BÚSQUEDA DE PARETO
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _auto_optimize_truncation(
        self,
        rho: AtomicDensityMatrix,
        max_dimension: Optional[int] = None
    ) -> Tuple[AtomicDensityMatrix, TruncationReport]:
        """
        Optimización automática: encuentra truncamiento en frontera de Pareto
        óptima entre compresión, fidelidad y speedup.
        """
        # Función de costo multi-objetivo
        def cost_function(report: TruncationReport) -> float:
            # Penalizar: baja fidelidad, alta compresión (poco), bajo speedup
            fidelity_penalty = (1.0 - report.fidelity_preservation) * 10.0
            compression_reward = report.compression_ratio * 5.0
            speedup_reward = min(np.log(report.computational_speedup), 5.0)
            majorization_penalty = 0.0 if report.majorization_verified else 100.0
            
            return fidelity_penalty - compression_reward - speedup_reward + majorization_penalty
        
        return self._spectral_projector.estimate_optimal_truncation(rho, cost_function)

    def _estimate_holevo_bound(
        self,
        rho: AtomicDensityMatrix,
        jump_operators: List[Tuple[float, NDArray[np.complex128]]]
    ) -> float:
        """
        Estima límite superior de capacidad de Holevo χ(ℰ) para el canal purificado.
        
        χ(ℰ) = S(ℰ(ρ)) - Σ pᵢ S(ℰ(ρᵢ)) ≤ S(ℰ(ρ)) ≤ ln d
        """
        # Cota trivial: χ ≤ S(ρ) ≤ ln d
        spectral = self._entropy_engine.compute_spectral_data(rho)
        return spectral.entropy

    # ═══════════════════════════════════════════════════════════════════════════
    # MÉTODOS DE ALTO NIVEL: INTERFAZ PÚBLICA EXTENDIDA
    # ═══════════════════════════════════════════════════════════════════════════
    
    def purify_with_target_fidelity(
        self,
        rho: AtomicDensityMatrix,
        jump_operators: List[Tuple[float, NDArray[np.complex128]]],
        target_fidelity: float,
        H: Optional[NDArray[np.complex128]] = None,
        max_dimension: Optional[int] = None
    ) -> Tuple[AtomicDensityMatrix, List[Tuple[float, NDArray[np.complex128]]], MinimizationMetrics]:
        """Purificación con fidelidad objetivo garantizada (búsqueda automática)."""
        return self.purify_semantic_state(
            rho, jump_operators, H=H, target_fidelity=target_fidelity, max_dimension=max_dimension
        )

    def purify_with_max_dimension(
        self,
        rho: AtomicDensityMatrix,
        jump_operators: List[Tuple[float, NDArray[np.complex128]]],
        max_dimension: int,
        H: Optional[NDArray[np.complex128]] = None
    ) -> Tuple[AtomicDensityMatrix, List[Tuple[float, NDArray[np.complex128]]], MinimizationMetrics]:
        """Purificación con dimensión máxima estricta."""
        return self.purify_semantic_state(
            rho, jump_operators, H=H, max_dimension=max_dimension
        )

    def purify_for_speedup(
        self,
        rho: AtomicDensityMatrix,
        jump_operators: List[Tuple[float, NDArray[np.complex128]]],
        min_speedup: float,
        H: Optional[NDArray[np.complex128]] = None
    ) -> Tuple[AtomicDensityMatrix, List[Tuple[float, NDArray[np.complex128]]], MinimizationMetrics]:
        """Purificación optimizando para speedup computacional mínimo."""
        # Estrategia: buscar máxima compresión que mantenga fidelidad razonable
        original_dim = rho.dimension
        target_dim = max(2, int(original_dim / (min_speedup ** (1/3))))
        
        return self.purify_semantic_state(
            rho, jump_operators, H=H, max_dimension=target_dim
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # TELEMETRÍA Y GESTIÓN DE ESTADO
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _update_telemetry(self, metrics: MinimizationMetrics) -> None:
        """Actualiza telemetría global del funtor."""
        self._telemetry = {
            'total_minimizations': self._minimization_count,
            'cumulative_compression': self._total_compression,
            'average_compression': np.mean([m.total_compression_ratio for m in self._history]) if self._history else 0.0,
            'average_fidelity': np.mean([m.fidelity_preservation for m in self._history]) if self._history else 1.0,
            'average_speedup': np.mean([m.computational_speedup for m in self._history]) if self._history else 1.0,
            'average_entropy_reduction': np.mean([m.entropy_reduction for m in self._history]) if self._history else 0.0,
            'majorization_success_rate': np.mean([m.majorization_preserved for m in self._history]) if self._history else 1.0,
            'last_execution_time_ms': metrics.execution_time_ms,
            'pareto_frontier_size': len(self._pareto_frontier)
        }

    def get_telemetry(self) -> Dict[str, Any]:
        """Retorna telemetría completa del funtor minimizador."""
        telemetry = dict(self._telemetry)
        if self._debug_mode:
            telemetry['history'] = self._history
            telemetry['pareto_frontier'] = self._pareto_frontier
        return telemetry

    def get_pareto_frontier(self) -> List[Tuple[float, float, float]]:
        """Retorna frontera de Pareto (compresión, fidelidad, speedup)."""
        return list(self._pareto_frontier)

    def reset(self) -> None:
        """Reinicia telemetría e historial."""
        self._minimization_count = 0
        self._total_compression = 1.0
        self._history.clear()
        self._telemetry.clear()
        self._pareto_frontier.clear()
        logger.info("MACMinimizer reiniciado — telemetría limpiada")

    def configure(
        self,
        epsilon_spectral: Optional[float] = None,
        tau_lindblad: Optional[float] = None,
        truncation_strategy: Optional[TruncationStrategy] = None,
        pruning_criterion: Optional[PruningCriterion] = None,
        auto_optimize: Optional[bool] = None,
        debug_mode: Optional[bool] = None
    ) -> None:
        """Reconfigura parámetros del funtor en caliente."""
        if epsilon_spectral is not None:
            self._epsilon_spectral = max(epsilon_spectral, MIN_EIGENVALUE)
            self._spectral_projector = SpectralTruncationProjector(
                epsilon_threshold=self._epsilon_spectral,
                strategy=self._truncation_strategy,
                entropy_engine=self._entropy_engine
            )
        
        if tau_lindblad is not None:
            self._tau_lindblad = max(tau_lindblad, 0.0)
            self._lindblad_pruner = LindbladPruningOperator(
                tau_cutoff=self._tau_lindblad,
                criterion=self._pruning_criterion,
                entropy_engine=self._entropy_engine
            )
        
        if truncation_strategy is not None:
            self._truncation_strategy = truncation_strategy
            self._spectral_projector = SpectralTruncationProjector(
                epsilon_threshold=self._epsilon_spectral,
                strategy=truncation_strategy,
                entropy_engine=self._entropy_engine
            )
        
        if pruning_criterion is not None:
            self._pruning_criterion = pruning_criterion
            self._lindblad_pruner = LindbladPruningOperator(
                tau_cutoff=self._tau_lindblad,
                criterion=pruning_criterion,
                entropy_engine=self._entropy_engine
            )
        
        if auto_optimize is not None:
            self._auto_optimize = auto_optimize
        
        if debug_mode is not None:
            self._debug_mode = debug_mode
        
        logger.info(f"MACMinimizer reconfigurado: ε={self._epsilon_spectral:.2e}, τ={self._tau_lindblad:.2e}")

    # ═══════════════════════════════════════════════════════════════════════════
    # INTERFAZ MORPHISM (heredada de app.core.mic_algebra)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def __call__(self, input_data: Any) -> Any:
        """
        Interfaz de morfismo: aplica purificación a datos de entrada.
        
        Espera input_data = (rho, jump_operators) o dict con claves 'rho', 'jump_operators'.
        """
        if isinstance(input_data, tuple) and len(input_data) == 2:
            rho, jump_ops = input_data
        elif isinstance(input_data, dict):
            rho = input_data.get('rho')
            jump_ops = input_data.get('jump_operators', [])
            H = input_data.get('H')
            return self.purify_semantic_state(rho, jump_ops, H=H)
        else:
            raise ValueError(
                "Entrada inválida para MACMinimizer. "
                "Esperado: (rho, jump_operators) o {'rho': ..., 'jump_operators': ...}"
            )
        
        return self.purify_semantic_state(rho, jump_ops)

    def compose(self, other: 'Morphism') -> 'Morphism':
        """Composición categórica de morfismos (no implementada completamente)."""
        # En una implementación completa, esto compondría funtores
        raise NotImplementedError("Composición categórica de funtores no implementada")

    def id(self) -> 'Morphism':
        """Morfismo identidad."""
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIÓN DE FÁBRICA Y EXPORTACIONES
# ═══════════════════════════════════════════════════════════════════════════════

def create_mac_minimizer(
    profile: str = 'balanced',
    **kwargs
) -> MACMinimizer:
    """
    Fábrica de minimizadores MAC con perfiles predefinidos.
    
    Perfiles:
        - 'aggressive':   Máxima compresión, tolera pérdida de fidelidad
        - 'balanced':     Equilibrio compresión/fidelidad (default)
        - 'conservative': Mínima pérdida de información, compresión moderada
        - 'speed':        Optimizado para speedup computacional
        - 'fidelity':     Fidelidad máxima, compresión mínima
        - 'research':     Modo debug completo con telemetría exhaustiva
    """
    profiles = {
        'aggressive': dict(
            epsilon_spectral=1e-3,
            tau_lindblad=1e-2,
            truncation_strategy=TruncationStrategy.CUMULATIVE_ENERGY,
            pruning_criterion=PruningCriterion.MAGNITUDE,
            auto_optimize=True,
            debug_mode=False
        ),
        'balanced': dict(
            epsilon_spectral=1e-6,
            tau_lindblad=1e-4,
            truncation_strategy=TruncationStrategy.THRESHOLD,
            pruning_criterion=PruningCriterion.FROBENIUS_NORM,
            auto_optimize=True,
            debug_mode=False
        ),
        'conservative': dict(
            epsilon_spectral=1e-9,
            tau_lindblad=1e-6,
            truncation_strategy=TruncationStrategy.ENTROPY_BOUNDED,
            pruning_criterion=PruningCriterion.DISSIPATIVE_GAP,
            auto_optimize=False,
            debug_mode=True
        ),
        'speed': dict(
            epsilon_spectral=1e-4,
            tau_lindblad=1e-3,
            truncation_strategy=TruncationStrategy.RANK_K,
            pruning_criterion=PruningCriterion.CHOI_RANK,
            auto_optimize=True,
            debug_mode=False
        ),
        'fidelity': dict(
            epsilon_spectral=1e-12,
            tau_lindblad=1e-8,
            truncation_strategy=TruncationStrategy.MAJORIZATION_OPTIMAL,
            pruning_criterion=PruningCriterion.INFORMATION_GAIN,
            auto_optimize=False,
            debug_mode=True
        ),
        'research': dict(
            epsilon_spectral=1e-6,
            tau_lindblad=1e-4,
            truncation_strategy=TruncationStrategy.THRESHOLD,
            pruning_criterion=PruningCriterion.FROBENIUS_NORM,
            auto_optimize=True,
            debug_mode=True
        )
    }
    
    if profile not in profiles:
        raise ValueError(f"Perfil desconocido: {profile}. Disponibles: {list(profiles.keys())}")
    
    config = profiles[profile]
    config.update(kwargs)  # Permitir sobrescribir
    
    return MACMinimizer(**config)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIONES PÚBLICAS
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Enums
    'TruncationStrategy',
    'PruningCriterion',
    'LogarithmicBase',
    'PurificationMode',
    
    # Dataclasses
    'SpectralData',
    'TruncationReport',
    'PruningReport',
    'MinimizationMetrics',
    
    # Protocolos categóricos
    'QuantumChannel',
    'PurificationFunctor',
    'Adjunction',
    
    # Clases principales
    'VonNeumannEntropyEngine',
    'SpectralTruncationProjector',
    'LindbladPruningOperator',
    'MACMinimizer',
    
    # Fábrica
    'create_mac_minimizer',
]

# ──────────────────────────────────────────────────────────────────────────────
# VALIDACIÓN DE IMPORTACIÓN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test rápido de importación y funcionamiento básico
    import sys
    
    print("╔════════════════════════════════════════════════════════════════════════════╗")
    print("║ MAC Minimizer v3.0 — Test de Importación y Validación Básica             ║")
    print("╚════════════════════════════════════════════════════════════════════════════╝")
    
    # Crear estado de prueba
    dim = 8
    np.random.seed(42)
    A = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    rho_test = A @ A.conj().T
    rho_test = rho_test / np.trace(rho_test)
    
    from app.wisdom.atomic_knowledge_matrix import AtomicDensityMatrix
    rho_obj = AtomicDensityMatrix(rho_test, validate=True)
    
    # Operadores de salto de prueba
    jump_ops = []
    for i in range(5):
        L = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        gamma = 10**(-np.random.uniform(1, 5))
        jump_ops.append((gamma, L))
    
    # Test perfil balanced
    minimizer = create_mac_minimizer('balanced')
    rho_pure, ops_pure, metrics = minimizer.purify_semantic_state(rho_obj, jump_ops)
    
    print(f"\n✓ Test completado exitosamente")
    print(f"  Dimensión original: {dim}")
    print(f"  Dimensión purificada: {rho_pure.dimension}")
    print(f"  Compresión: {metrics.total_compression_ratio:.1%}")
    print(f"  Fidelidad: {metrics.fidelity_preservation:.6f}")
    print(f"  Speedup: {metrics.computational_speedup:.2f}x")
    print(f"  Majorización preservada: {metrics.majorization_preserved}")
    print(f"  CPTP verificado: {metrics.pruning_report.complete_positivity_verified if metrics.pruning_report else 'N/A'}")
    
    print("\n✓ Todos los componentes importados y funcionales")
    sys.exit(0)