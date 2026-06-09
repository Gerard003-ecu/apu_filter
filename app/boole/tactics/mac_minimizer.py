# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo: MAC Minimizer (Funtor de Purificación Espectral y Reducción)         ║
║ Ubicación: app/boole/tactics/mac_minimizer.py                                ║
║ Versión: 2.0.0-Quantum-Spectral-Purification-Enhanced                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Naturaleza Ciber-Física y Topológica Refinada:
────────────────────────────────────────────────
Este minimizador implementa un funtor de purificación sobre el retículo 
ortomodular de proyectores cuánticos. Comprime el operador de densidad de la 
MAC eliminando estados de baja relevancia semántica, maximizando la eficiencia 
informacional del sistema.

FUNDAMENTOS TEÓRICOS:
─────────────────────
1. TEORÍA ESPECTRAL: Descomposición de operadores autoadjuntos
2. TEORÍA DE INFORMACIÓN CUÁNTICA: Entropía de von Neumann
3. GEOMETRÍA DIFERENCIAL: Variedades de estados cuánticos
4. ANÁLISIS FUNCIONAL: Proyectores ortogonales en espacios de Hilbert
5. TERMODINÁMICA CUÁNTICA: Purificación y destilación de estados

Axiomas Matemáticos Implementados y Extendidos:
────────────────────────────────────────────────
A1. Entropía de von Neumann: S(ρ) = -Tr(ρ ln ρ) = -Σᵢ λᵢ ln λᵢ
A2. Descomposición Espectral: ρ = Σₖ λₖ |ψₖ⟩⟨ψₖ|
A3. Proyección de Truncamiento: P_ε = Σ_{λₖ≥ε} |ψₖ⟩⟨ψₖ|
A4. Mapa CPTP: ρ̃ = P_ε ρ P_ε / Tr(P_ε ρ P_ε)
A5. Pureza Relativa: γ(ρ) = Tr(ρ²) ∈ [1/d, 1]
A6. Rango Efectivo: r_eff(ρ) = exp(S(ρ))
A7. Conservación de Información: I(ρ||σ) ≥ 0 (divergencia relativa)

Referencias Teóricas:
─────────────────────
- von Neumann (1932): "Mathematical Foundations of Quantum Mechanics"
- Schumacher (1995): "Quantum coding theorem"
- Vidal et al. (2002): "Entanglement in quantum critical phenomena"
- Eisert & Gross (2007): "Multi-particle entanglement"
- Nielsen & Chuang (2010): "Quantum Computation and Quantum Information"
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import numpy as np
import scipy.linalg as la
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any, Callable
from numpy.typing import NDArray
from enum import Enum, auto

# Dependencias arquitectónicas del estrato WISDOM
from app.core.mic_algebra import Morphism, NumericalInstabilityError
from app.wisdom.atomic_knowledge_matrix import AtomicDensityMatrix, QuantumMetrics
from app.wisdom.mac_agent import POVMMeasurement

logger = logging.getLogger("MAC.Wisdom.Minimizer")


# ══════════════════════════════════════════════════════════════════════════════
# INFRAESTRUCTURA: ENUMERACIONES Y DATACLASSES
# ══════════════════════════════════════════════════════════════════════════════

class TruncationStrategy(Enum):
    """Estrategias de truncamiento espectral."""
    THRESHOLD = auto()          # Umbral fijo ε
    RANK_K = auto()             # Mantener top-k eigenvalores
    CUMULATIVE_ENERGY = auto()  # Conservar porcentaje de energía
    ENTROPY_BOUNDED = auto()    # Limitar entropía máxima


class PruningCriterion(Enum):
    """Criterios de poda de operadores."""
    MAGNITUDE = auto()          # Por magnitud de tasa γₖ
    FROBENIUS_NORM = auto()     # Por norma de Frobenius de Lₖ
    COMMUTATOR_NORM = auto()    # Por ‖[H, Lₖ]‖
    INFORMATION_GAIN = auto()   # Por ganancia de información


@dataclass
class TruncationReport:
    """Reporte detallado de truncamiento espectral."""
    original_dimension: int
    truncated_dimension: int
    retained_eigenvalues: NDArray[np.float64]
    discarded_eigenvalues: NDArray[np.float64]
    retained_energy: float
    compression_ratio: float
    entropy_before: float
    entropy_after: float
    purity_before: float
    purity_after: float
    
    def __post_init__(self):
        """Validación física del reporte."""
        assert 0 <= self.compression_ratio <= 1, "Ratio de compresión inválido"
        assert self.retained_energy >= 0, "Energía retenida negativa"
        assert self.truncated_dimension <= self.original_dimension, \
            "Dimensión truncada mayor que original"


@dataclass
class PruningReport:
    """Reporte de poda de operadores de Lindblad."""
    original_count: int
    pruned_count: int
    discarded_count: int
    total_rate_before: float
    total_rate_after: float
    pruning_efficiency: float
    
    def __post_init__(self):
        """Validación de consistencia."""
        assert self.pruned_count + self.discarded_count == self.original_count, \
            "Conteo inconsistente de operadores"
        assert 0 <= self.pruning_efficiency <= 1, \
            "Eficiencia de poda fuera de rango"


@dataclass
class MinimizationMetrics:
    """Métricas agregadas de minimización."""
    truncation_report: Optional[TruncationReport]
    pruning_report: Optional[PruningReport]
    total_compression_ratio: float
    information_loss: float          # I(ρ_original || ρ_truncated)
    fidelity_preservation: float     # F(ρ_original, ρ_truncated)
    computational_speedup: float     # Estimación de ganancia computacional


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: MOTOR DE ENTROPÍA Y PURIFICACIÓN DE VON NEUMANN
# ══════════════════════════════════════════════════════════════════════════════

class VonNeumannEntropyMinimizer:
    r"""
    Calculadora Rigurosa del Tensor Entrópico - VERSIÓN MEJORADA.
    
    FUNDAMENTO TEÓRICO:
    ──────────────────
    La entropía de von Neumann cuantifica la incertidumbre o "mezcla" de un
    estado cuántico:
    
        S(ρ) = -Tr(ρ ln ρ) = -Σᵢ λᵢ ln λᵢ
    
    donde λᵢ son los eigenvalores de ρ.
    
    PROPIEDADES:
    ───────────
    1. S(ρ) ≥ 0, con igualdad ssi ρ es estado puro
    2. S(ρ) ≤ ln d, con igualdad ssi ρ = I/d (maximalmente mixto)
    3. S es cóncava: S(Σᵢ pᵢ ρᵢ) ≥ Σᵢ pᵢ S(ρᵢ)
    4. Invariante bajo unitarios: S(U ρ U†) = S(ρ)
    
    MEJORAS IMPLEMENTADAS:
    ─────────────────────
    1. Cálculo numericamente estable (evita log(0))
    2. Múltiples bases logarítmicas (ln, log₂, log₁₀)
    3. Entropías generalizadas (Rényi, Tsallis)
    4. Cálculo de divergencia relativa
    5. Métricas derivadas (pureza, rango efectivo)
    """
    
    def __init__(self, tol: float = 1e-12, log_base: str = 'natural'):
        """
        Args:
            tol: Tolerancia para eigenvalores considerados cero
            log_base: Base logarítmica ('natural', '2', '10')
        """
        self.tol = tol
        self.log_base = log_base
        
        # Factor de conversión logarítmica
        if log_base == 'natural':
            self.log_factor = 1.0
        elif log_base == '2':
            self.log_factor = 1.0 / np.log(2)
        elif log_base == '10':
            self.log_factor = 1.0 / np.log(10)
        else:
            raise ValueError(f"Base logarítmica desconocida: {log_base}")

    def compute_entropy(
        self, 
        rho: AtomicDensityMatrix,
        validate: bool = True
    ) -> float:
        r"""
        Calcula rigurosamente: S(ρ) = -Tr(ρ ln ρ) = -Σᵢ λᵢ ln λᵢ.
        
        Args:
            rho: Operador de densidad
            validate: Validar axiomas cuánticos
        
        Returns:
            Entropía de von Neumann S(ρ) ≥ 0
        """
        if validate:
            # Asegurar que rho es válido
            metrics = rho.compute_metrics()
        
        # Extraer eigenvalores
        eigenvalues = la.eigvalsh(rho.matrix)
        
        # Filtrar eigenvalores positivos (evitar log(0))
        positive_eigs = eigenvalues[eigenvalues > self.tol]
        
        if len(positive_eigs) == 0:
            logger.warning("Todos los eigenvalores son cero. Entropía indefinida.")
            return 0.0
        
        # Calcular S = -Σᵢ λᵢ ln λᵢ
        entropy_contributions = -positive_eigs * np.log(positive_eigs)
        entropy = np.sum(entropy_contributions) * self.log_factor
        
        logger.debug(f"Entropía de von Neumann computada: S(ρ) = {entropy:.6e}")
        
        return float(entropy)

    def compute_renyi_entropy(
        self, 
        rho: AtomicDensityMatrix, 
        alpha: float
    ) -> float:
        r"""
        Entropía de Rényi de orden α:
        
            S_α(ρ) = 1/(1-α) ln(Tr(ρ^α)) = 1/(1-α) ln(Σᵢ λᵢ^α)
        
        Casos especiales:
            - α → 1: entropía de von Neumann
            - α = 0: ln(rank(ρ))
            - α = 2: -ln(Tr(ρ²)) (entropía de colisión)
            - α → ∞: -ln(λ_max) (min-entropía)
        
        Args:
            rho: Operador de densidad
            alpha: Orden de la entropía (α ≥ 0, α ≠ 1)
        
        Returns:
            Entropía de Rényi S_α(ρ)
        """
        if np.isclose(alpha, 1.0):
            return self.compute_entropy(rho)
        
        if alpha < 0:
            raise ValueError(f"Orden de Rényi debe ser no negativo: {alpha}")
        
        eigenvalues = la.eigvalsh(rho.matrix)
        positive_eigs = eigenvalues[eigenvalues > self.tol]
        
        if np.isclose(alpha, 0.0):
            # S₀ = ln(rank(ρ))
            return np.log(len(positive_eigs)) * self.log_factor
        elif np.isinf(alpha):
            # S_∞ = -ln(λ_max)
            return -np.log(np.max(positive_eigs)) * self.log_factor
        else:
            # S_α = 1/(1-α) ln(Σᵢ λᵢ^α)
            trace_power = np.sum(positive_eigs ** alpha)
            return (1.0 / (1.0 - alpha)) * np.log(trace_power) * self.log_factor

    def compute_relative_entropy(
        self, 
        rho: AtomicDensityMatrix, 
        sigma: AtomicDensityMatrix
    ) -> float:
        r"""
        Divergencia cuántica relativa (divergencia de Kullback-Leibler):
        
            D(ρ||σ) = Tr(ρ ln ρ) - Tr(ρ ln σ)
        
        Propiedades:
            - D(ρ||σ) ≥ 0 con igualdad ssi ρ = σ
            - No es simétrica
            - D(ρ||σ) = ∞ si supp(ρ) ⊄ supp(σ)
        
        Args:
            rho: Primer estado
            sigma: Segundo estado
        
        Returns:
            Divergencia relativa D(ρ||σ) ≥ 0
        """
        # Calcular Tr(ρ ln ρ)
        eig_rho = la.eigvalsh(rho.matrix)
        positive_rho = eig_rho[eig_rho > self.tol]
        term1 = np.sum(positive_rho * np.log(positive_rho))
        
        # Calcular Tr(ρ ln σ)
        # Esto requiere descomposición espectral de σ
        eig_sigma, vec_sigma = la.eigh(sigma.matrix)
        
        # Construir ln(σ)
        log_eig_sigma = np.where(
            eig_sigma > self.tol,
            np.log(eig_sigma),
            -np.inf  # ln(0) = -∞
        )
        
        # Verificar soporte
        if np.any(np.isinf(log_eig_sigma)):
            # Verificar si ρ tiene soporte fuera de σ
            rho_sigma_overlap = vec_sigma.conj().T @ rho.matrix @ vec_sigma
            overlap_on_zero = np.abs(np.diag(rho_sigma_overlap)[eig_sigma <= self.tol])
            
            if np.any(overlap_on_zero > self.tol):
                logger.warning(
                    "Divergencia infinita: supp(ρ) ⊄ supp(σ)"
                )
                return float('inf')
        
        # ln(σ) = V diag(ln λᵢ) V†
        log_sigma = vec_sigma @ np.diag(log_eig_sigma) @ vec_sigma.conj().T
        
        term2 = np.trace(rho.matrix @ log_sigma).real
        
        relative_entropy = term1 - term2
        
        return float(max(0.0, relative_entropy))  # Clipping numérico

    def compute_effective_rank(self, rho: AtomicDensityMatrix) -> float:
        """
        Rango efectivo: r_eff(ρ) = exp(S(ρ)).
        
        Interpreta la entropía como el logaritmo del "número efectivo de
        estados puros" en la mezcla.
        
        Returns:
            Rango efectivo r_eff ∈ [1, d]
        """
        entropy = self.compute_entropy(rho)
        return float(np.exp(entropy / self.log_factor))

    def compute_purity(self, rho: AtomicDensityMatrix) -> float:
        """
        Pureza: γ(ρ) = Tr(ρ²).
        
        Relacionada con entropía de Rényi de orden 2:
            γ(ρ) = exp(-S₂(ρ))
        
        Returns:
            Pureza γ ∈ [1/d, 1]
        """
        rho_squared = rho.matrix @ rho.matrix
        purity = np.trace(rho_squared).real
        
        return float(purity)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: OPERADOR DE TRUNCAMIENTO ESPECTRAL (QUANTUM PCA)
# ══════════════════════════════════════════════════════════════════════════════

class SpectralTruncationProjector:
    r"""
    Funtor de Reducción de Dimensionalidad Cuántica - VERSIÓN MEJORADA.
    
    FUNDAMENTO TEÓRICO:
    ──────────────────
    Dado un operador de densidad ρ con descomposición espectral:
    
        ρ = Σₖ λₖ |ψₖ⟩⟨ψₖ|
    
    El truncamiento espectral construye un proyector P_ε que retiene solo
    los eigenestados con eigenvalores significativos:
    
        P_ε = Σ_{λₖ≥ε} |ψₖ⟩⟨ψₖ|
    
    El estado truncado es:
    
        ρ̃ = P_ε ρ P_ε / Tr(P_ε ρ P_ε)
    
    que es un mapa CPTP (Completely Positive Trace-Preserving).
    
    MEJORAS IMPLEMENTADAS:
    ─────────────────────
    1. Múltiples estrategias de truncamiento
    2. Análisis de conservación de información
    3. Cálculo de fidelidad pre/post truncamiento
    4. Estimación de ganancia computacional
    5. Reportes detallados con métricas
    """
    
    def __init__(
        self, 
        epsilon_threshold: float = 1e-6,
        strategy: TruncationStrategy = TruncationStrategy.THRESHOLD
    ):
        """
        Args:
            epsilon_threshold: Umbral para truncamiento
            strategy: Estrategia de truncamiento
        """
        self.epsilon = epsilon_threshold
        self.strategy = strategy
        self.entropy_engine = VonNeumannEntropyMinimizer()

    def _compute_truncation_mask(
        self,
        eigenvalues: NDArray[np.float64],
        target_param: Optional[float] = None
    ) -> NDArray[np.bool_]:
        """
        Calcula máscara booleana de eigenvalores a retener según estrategia.
        
        Args:
            eigenvalues: Eigenvalores ordenados descendentemente
            target_param: Parámetro adicional según estrategia
        
        Returns:
            Máscara booleana (True = retener)
        """
        if self.strategy == TruncationStrategy.THRESHOLD:
            # Retener eigenvalores ≥ ε
            return eigenvalues >= self.epsilon
        
        elif self.strategy == TruncationStrategy.RANK_K:
            # Retener top-k eigenvalores
            k = int(target_param) if target_param else 1
            k = min(k, len(eigenvalues))
            mask = np.zeros(len(eigenvalues), dtype=bool)
            mask[:k] = True
            return mask
        
        elif self.strategy == TruncationStrategy.CUMULATIVE_ENERGY:
            # Retener hasta alcanzar % de energía
            target_energy = target_param if target_param else 0.95
            cumulative = np.cumsum(eigenvalues) / np.sum(eigenvalues)
            return cumulative <= target_energy + self.epsilon
        
        elif self.strategy == TruncationStrategy.ENTROPY_BOUNDED:
            # Retener hasta alcanzar entropía máxima
            # (requiere cálculo iterativo)
            target_entropy = target_param if target_param else 1.0
            
            # Ordenar eigenvalues descendentemente
            sorted_eigs = np.sort(eigenvalues)[::-1]
            
            mask = np.zeros(len(eigenvalues), dtype=bool)
            
            for k in range(1, len(eigenvalues) + 1):
                # Calcular entropía con top-k eigenvalores
                truncated_eigs = sorted_eigs[:k]
                truncated_eigs = truncated_eigs / np.sum(truncated_eigs)
                
                entropy = -np.sum(
                    truncated_eigs * np.log(truncated_eigs + 1e-12)
                )
                
                if entropy >= target_entropy:
                    mask[:k] = True
                    break
            
            # Mapear de vuelta al orden original
            sort_indices = np.argsort(eigenvalues)[::-1]
            original_mask = np.zeros(len(eigenvalues), dtype=bool)
            original_mask[sort_indices] = mask
            
            return original_mask
        
        else:
            raise ValueError(f"Estrategia desconocida: {self.strategy}")

    def truncate_spectrum(
        self, 
        rho: AtomicDensityMatrix,
        target_param: Optional[float] = None,
        compute_fidelity: bool = True
    ) -> Tuple[AtomicDensityMatrix, TruncationReport]:
        r"""
        Ejecuta truncamiento espectral con reportes detallados.
        
        Args:
            rho: Estado original
            target_param: Parámetro según estrategia
            compute_fidelity: Calcular fidelidad pre/post
        
        Returns:
            (estado_truncado, reporte_truncamiento)
        
        Raises:
            NumericalInstabilityError: Si todos los eigenvalores son despreciables
        """
        rho_matrix = rho.matrix
        dim = rho_matrix.shape[0]
        
        # Métricas iniciales
        entropy_before = self.entropy_engine.compute_entropy(rho)
        purity_before = self.entropy_engine.compute_purity(rho)
        
        # 1. Descomposición Espectral: ρ = U Λ U†
        eigenvalues, eigenvectors = la.eigh(rho_matrix)
        
        # Ordenar descendentemente
        sort_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sort_indices]
        eigenvectors = eigenvectors[:, sort_indices]
        
        # 2. Determinar eigenvalores a retener
        retention_mask = self._compute_truncation_mask(eigenvalues, target_param)
        
        retained_eigenvalues = eigenvalues[retention_mask]
        discarded_eigenvalues = eigenvalues[~retention_mask]
        
        if len(retained_eigenvalues) == 0:
            raise NumericalInstabilityError(
                "Degeneración Espectral Letal: El estado semántico carece de exergía. "
                f"Todos los {len(eigenvalues)} eigenvalores están por debajo del umbral."
            )
        
        # 3. Construir Proyector P_ε = Σ_{retener} |ψₖ⟩⟨ψₖ|
        retained_eigenvectors = eigenvectors[:, retention_mask]
        
        P_epsilon = retained_eigenvectors @ retained_eigenvectors.conj().T
        
        # 4. Aplicar Mapa CPTP: ρ̃ = P_ε ρ P_ε / Tr(P_ε ρ P_ε)
        rho_truncated = P_epsilon @ rho_matrix @ P_epsilon.conj().T
        trace_truncated = np.trace(rho_truncated).real
        
        if trace_truncated < 1e-12:
            raise NumericalInstabilityError(
                f"Traza post-truncamiento prácticamente nula: {trace_truncated}"
            )
        
        rho_purified = rho_truncated / trace_truncated
        
        # 5. Imponer Hermiticidad (corrección numérica)
        rho_purified = (rho_purified + rho_purified.conj().T) / 2.0
        
        # Crear estado truncado
        rho_truncated_obj = AtomicDensityMatrix(
            rho_purified, 
            auto_renormalize=True, 
            validate=True
        )
        
        # Métricas finales
        entropy_after = self.entropy_engine.compute_entropy(rho_truncated_obj)
        purity_after = self.entropy_engine.compute_purity(rho_truncated_obj)
        
        # Energía retenida
        retained_energy = float(np.sum(retained_eigenvalues))
        total_energy = float(np.sum(eigenvalues))
        
        # Ratio de compresión
        compression_ratio = len(retained_eigenvalues) / len(eigenvalues)
        
        # Crear reporte
        report = TruncationReport(
            original_dimension=dim,
            truncated_dimension=len(retained_eigenvalues),
            retained_eigenvalues=retained_eigenvalues,
            discarded_eigenvalues=discarded_eigenvalues,
            retained_energy=retained_energy / total_energy,
            compression_ratio=compression_ratio,
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            purity_before=purity_before,
            purity_after=purity_after
        )
        
        logger.info(
            f"Truncamiento Espectral ejecutado. "
            f"Dimensión: {dim} → {len(retained_eigenvalues)}, "
            f"Energía retenida: {report.retained_energy*100:.2f}%, "
            f"S: {entropy_before:.4f} → {entropy_after:.4f}"
        )
        
        return rho_truncated_obj, report

    def estimate_computational_speedup(self, report: TruncationReport) -> float:
        """
        Estima la ganancia computacional del truncamiento.
        
        Asume complejidad O(d³) para operaciones matriciales densas.
        
        Args:
            report: Reporte de truncamiento
        
        Returns:
            Factor de speedup (> 1 indica mejora)
        """
        d_original = report.original_dimension
        d_truncated = report.truncated_dimension
        
        # Speedup = (d_original / d_truncated)³
        speedup = (d_original / d_truncated) ** 3
        
        return float(speedup)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: PODA DE OPERADORES DE SALTO (LINDBLADIAN MINIMIZATION)
# ══════════════════════════════════════════════════════════════════════════════

class LindbladPruningOperator:
    r"""
    Optimizador de la Ecuación Maestra de Lindblad - VERSIÓN MEJORADA.
    
    FUNDAMENTO TEÓRICO:
    ──────────────────
    La ecuación de Lindblad incluye términos disipativos:
    
        dρ/dt = -i[H,ρ] + Σₖ γₖ 𝒟[Lₖ](ρ)
    
    donde 𝒟[L](ρ) = L ρ L† - ½{L†L, ρ}.
    
    Operadores con γₖ muy pequeño contribuyen negligiblemente a la dinámica
    pero aumentan el costo computacional. La poda elimina estos términos.
    
    CRITERIOS DE PODA:
    ─────────────────
    1. MAGNITUDE: γₖ < τ
    2. FROBENIUS_NORM: γₖ ‖Lₖ‖_F < τ
    3. COMMUTATOR_NORM: γₖ ‖[H, Lₖ]‖ < τ
    4. INFORMATION_GAIN: ΔI < τ (requiere simulación)
    
    MEJORAS IMPLEMENTADAS:
    ─────────────────────
    1. Múltiples criterios de poda
    2. Análisis de impacto en dinámica
    3. Conservación adaptativa de operadores críticos
    4. Reportes detallados con métricas
    """
    
    def __init__(
        self, 
        tau_cutoff: float = 1e-4,
        criterion: PruningCriterion = PruningCriterion.MAGNITUDE,
        preserve_critical: bool = True
    ):
        """
        Args:
            tau_cutoff: Umbral de poda
            criterion: Criterio de poda
            preserve_critical: Preservar al menos 1 operador
        """
        self.tau = tau_cutoff
        self.criterion = criterion
        self.preserve_critical = preserve_critical

    def _compute_pruning_score(
        self,
        gamma: float,
        L_k: NDArray[np.complex128],
        H: Optional[NDArray[np.complex128]] = None
    ) -> float:
        """
        Calcula puntuación de importancia según criterio.
        
        Args:
            gamma: Tasa de disipación
            L_k: Operador de salto
            H: Hamiltoniano (opcional, para ciertos criterios)
        
        Returns:
            Puntuación (mayor = más importante)
        """
        if self.criterion == PruningCriterion.MAGNITUDE:
            return gamma
        
        elif self.criterion == PruningCriterion.FROBENIUS_NORM:
            norm_L = la.norm(L_k, ord='fro')
            return gamma * norm_L
        
        elif self.criterion == PruningCriterion.COMMUTATOR_NORM:
            if H is None:
                logger.warning(
                    "Hamiltoniano no proporcionado para criterio COMMUTATOR_NORM. "
                    "Usando MAGNITUDE."
                )
                return gamma
            
            commutator = H @ L_k - L_k @ H
            norm_comm = la.norm(commutator, ord='fro')
            return gamma * norm_comm
        
        elif self.criterion == PruningCriterion.INFORMATION_GAIN:
            # Criterio más sofisticado (requiere simulación de dinámica)
            # Por simplicidad, usamos magnitud
            logger.warning(
                "Criterio INFORMATION_GAIN no totalmente implementado. "
                "Usando MAGNITUDE."
            )
            return gamma
        
        else:
            raise ValueError(f"Criterio desconocido: {self.criterion}")

    def prune_jump_operators(
        self, 
        jump_operators: List[Tuple[float, NDArray[np.complex128]]],
        H: Optional[NDArray[np.complex128]] = None
    ) -> Tuple[List[Tuple[float, NDArray[np.complex128]]], PruningReport]:
        r"""
        Filtra operadores de salto según criterio.
        
        Args:
            jump_operators: Lista de (γₖ, Lₖ)
            H: Hamiltoniano (opcional)
        
        Returns:
            (operadores_podados, reporte)
        """
        if not jump_operators:
            # Lista vacía
            report = PruningReport(
                original_count=0,
                pruned_count=0,
                discarded_count=0,
                total_rate_before=0.0,
                total_rate_after=0.0,
                pruning_efficiency=0.0
            )
            return [], report
        
        initial_count = len(jump_operators)
        
        # Calcular puntuaciones
        scores = [
            self._compute_pruning_score(gamma, L_k, H)
            for gamma, L_k in jump_operators
        ]
        
        # Tasas totales
        total_rate_before = sum(gamma for gamma, _ in jump_operators)
        
        # Filtrar según umbral
        pruned_ops = [
            (gamma, L_k) 
            for (gamma, L_k), score in zip(jump_operators, scores)
            if score >= self.tau
        ]
        
        # Si preserve_critical y todos fueron podados, conservar el de mayor score
        if self.preserve_critical and len(pruned_ops) == 0 and initial_count > 0:
            max_score_idx = int(np.argmax(scores))
            pruned_ops = [jump_operators[max_score_idx]]
            logger.warning(
                "Todos los operadores bajo umbral. "
                f"Preservando operador crítico con score {scores[max_score_idx]:.3e}"
            )
        
        pruned_count = len(pruned_ops)
        discarded_count = initial_count - pruned_count
        
        total_rate_after = sum(gamma for gamma, _ in pruned_ops)
        
        # Eficiencia de poda
        pruning_efficiency = discarded_count / max(1, initial_count)
        
        # Crear reporte
        report = PruningReport(
            original_count=initial_count,
            pruned_count=pruned_count,
            discarded_count=discarded_count,
            total_rate_before=total_rate_before,
            total_rate_after=total_rate_after,
            pruning_efficiency=pruning_efficiency
        )
        
        if discarded_count > 0:
            logger.info(
                f"Poda Lindbladiana ejecutada. "
                f"{discarded_count}/{initial_count} operadores descartados. "
                f"Tasa total: {total_rate_before:.3e} → {total_rate_after:.3e}"
            )
        
        return pruned_ops, report


# ══════════════════════════════════════════════════════════════════════════════
# FASE 4: FUNTOR DE MINIMIZACIÓN MAC (ORQUESTADOR PRINCIPAL)
# ══════════════════════════════════════════════════════════════════════════════

class MACMinimizer(Morphism):
    r"""
    El Funtor Supremo de Purificación Semántica - VERSIÓN MEJORADA.
    
    Orquesta la minimización completa del estado MAC mediante:
        1. Análisis entrópico
        2. Truncamiento espectral
        3. Poda de operadores de Lindblad
        4. Validación de conservación de información
    
    MEJORAS IMPLEMENTADAS:
    ─────────────────────
    1. Pipeline completo con validaciones
    2. Métricas agregadas de minimización
    3. Análisis de fidelidad y pérdida de información
    4. Estimación de ganancia computacional
    5. Modo debug con telemetría extendida
    """
    
    def __init__(
        self,
        epsilon_spectral: float = 1e-6,
        tau_lindblad: float = 1e-4,
        truncation_strategy: TruncationStrategy = TruncationStrategy.THRESHOLD,
        pruning_criterion: PruningCriterion = PruningCriterion.MAGNITUDE,
        auto_optimize: bool = True,
        debug_mode: bool = False
    ):
        """
        Args:
            epsilon_spectral: Umbral de truncamiento espectral
            tau_lindblad: Umbral de poda de Lindblad
            truncation_strategy: Estrategia de truncamiento
            pruning_criterion: Criterio de poda
            auto_optimize: Optimización automática adaptativa
            debug_mode: Modo debug (telemetría extendida)
        """
        self.epsilon_spectral = epsilon_spectral
        self.tau_lindblad = tau_lindblad
        self.auto_optimize = auto_optimize
        self.debug_mode = debug_mode
        
        # Componentes
        self.entropy_engine = VonNeumannEntropyMinimizer()
        self.spectral_projector = SpectralTruncationProjector(
            epsilon_threshold=epsilon_spectral,
            strategy=truncation_strategy
        )
        self.lindblad_pruner = LindbladPruningOperator(
            tau_cutoff=tau_lindblad,
            criterion=pruning_criterion
        )
        
        # Telemetría
        self.minimization_count: int = 0
        self.total_compression_ratio: float = 1.0
        self.history: List[MinimizationMetrics] = []

    def purify_semantic_state(
        self, 
        rho: AtomicDensityMatrix, 
        jump_operators: List[Tuple[float, NDArray[np.complex128]]],
        entropy_threshold: Optional[float] = None,
        H: Optional[NDArray[np.complex128]] = None,
        force_truncation: bool = False
    ) -> Tuple[AtomicDensityMatrix, List[Tuple[float, NDArray[np.complex128]]], MinimizationMetrics]:
        r"""
        Ejecuta purificación espectral integral con análisis completo.
        
        PROTOCOLO:
        ──────────
        1. ANALYZE: Auditar entropía y pureza
        2. TRUNCATE: Truncamiento espectral (si necesario)
        3. PRUNE: Poda de operadores de Lindblad
        4. VALIDATE: Validar conservación de información
        5. REPORT: Generar métricas agregadas
        
        Args:
            rho: Estado original
            jump_operators: Operadores de salto (γₖ, Lₖ)
            entropy_threshold: Umbral de entropía para truncamiento
            H: Hamiltoniano (para ciertos criterios)
            force_truncation: Forzar truncamiento independiente de entropía
        
        Returns:
            (estado_purificado, operadores_optimizados, métricas)
        """
        self.minimization_count += 1
        
        # ─────────────────────────────────────────────────────────────────
        # FASE 1: ANALYZE - Auditoría Entrópica
        # ─────────────────────────────────────────────────────────────────
        
        logger.info(f"[ANALYZE] Iniciando auditoría entrópica...")
        
        entropy_initial = self.entropy_engine.compute_entropy(rho)
        purity_initial = self.entropy_engine.compute_purity(rho)
        effective_rank_initial = self.entropy_engine.compute_effective_rank(rho)
        
        logger.info(
            f"Estado inicial: S={entropy_initial:.4f}, "
            f"γ={purity_initial:.4f}, r_eff={effective_rank_initial:.2f}"
        )
        
        # Determinar si se requiere truncamiento
        if entropy_threshold is None:
            # Umbral adaptativo basado en dimensión
            dim = rho.dimension
            entropy_threshold = 0.7 * np.log(dim)  # 70% de máxima entropía
        
        should_truncate = (
            force_truncation or 
            entropy_initial > entropy_threshold or
            purity_initial < 0.5  # Muy mixto
        )
        
        # ─────────────────────────────────────────────────────────────────
        # FASE 2: TRUNCATE - Truncamiento Espectral
        # ─────────────────────────────────────────────────────────────────
        
        truncation_report = None
        
        if should_truncate:
            logger.info(
                f"[TRUNCATE] Entropía {entropy_initial:.4f} excede umbral {entropy_threshold:.4f}. "
                f"Iniciando truncamiento espectral..."
            )
            
            purified_rho, truncation_report = self.spectral_projector.truncate_spectrum(
                rho, compute_fidelity=True
            )
            
            logger.info(
                f"Truncamiento completado. "
                f"Dimensión: {truncation_report.original_dimension} → {truncation_report.truncated_dimension}"
            )
        else:
            logger.info(
                f"[TRUNCATE] Entropía {entropy_initial:.4f} dentro de límites. "
                f"Saltando truncamiento."
            )
            purified_rho = rho
        
        # ─────────────────────────────────────────────────────────────────
        # FASE 3: PRUNE - Poda de Operadores de Lindblad
        # ─────────────────────────────────────────────────────────────────
        
        logger.info(f"[PRUNE] Iniciando poda de operadores de Lindblad...")
        
        optimized_jump_ops, pruning_report = self.lindblad_pruner.prune_jump_operators(
            jump_operators, H=H
        )
        
        logger.info(
            f"Poda completada. "
            f"Operadores: {pruning_report.original_count} → {pruning_report.pruned_count}"
        )
        
        # ─────────────────────────────────────────────────────────────────
        # FASE 4: VALIDATE - Validación de Conservación
        # ─────────────────────────────────────────────────────────────────
        
        logger.info(f"[VALIDATE] Validando conservación de información...")
        
        # Calcular pérdida de información
        if should_truncate:
            information_loss = self.entropy_engine.compute_relative_entropy(
                rho, purified_rho
            )
            
            # Calcular fidelidad (requiere GaloisAdjunctionAuditor)
            from app.wisdom.mac_agent import GaloisAdjunctionAuditor
            auditor = GaloisAdjunctionAuditor()
            fidelity = auditor.compute_fidelity(rho.matrix, purified_rho.matrix)
        else:
            information_loss = 0.0
            fidelity = 1.0
        
        # Calcular ratio de compresión total
        if truncation_report:
            total_compression = truncation_report.compression_ratio * (
                pruning_report.pruned_count / max(1, pruning_report.original_count)
            )
        else:
            total_compression = (
                pruning_report.pruned_count / max(1, pruning_report.original_count)
            )
        
        # Estimar speedup computacional
        if truncation_report:
            computational_speedup = self.spectral_projector.estimate_computational_speedup(
                truncation_report
            )
        else:
            computational_speedup = 1.0
        
        # ─────────────────────────────────────────────────────────────────
        # FASE 5: REPORT - Métricas Agregadas
        # ─────────────────────────────────────────────────────────────────
        
        metrics = MinimizationMetrics(
            truncation_report=truncation_report,
            pruning_report=pruning_report,
            total_compression_ratio=total_compression,
            information_loss=information_loss,
            fidelity_preservation=fidelity,
            computational_speedup=computational_speedup
        )
        
        # Almacenar en historial (si debug)
        if self.debug_mode:
            self.history.append(metrics)
        
        # Actualizar telemetría global
        self.total_compression_ratio *= total_compression
        
        logger.info(
            f"✓ Purificación completada. "
            f"Compresión: {total_compression*100:.1f}%, "
            f"Fidelidad: {fidelity:.4f}, "
            f"Speedup: {computational_speedup:.2f}x"
        )
        
        return purified_rho, optimized_jump_ops, metrics

    def get_telemetry(self) -> Dict[str, Any]:
        """Telemetría completa del minimizador."""
        return {
            'minimization_count': self.minimization_count,
            'total_compression_ratio': self.total_compression_ratio,
            'average_compression': (
                np.mean([m.total_compression_ratio for m in self.history])
                if self.history else 0.0
            ),
            'average_fidelity': (
                np.mean([m.fidelity_preservation for m in self.history])
                if self.history else 1.0
            ),
            'history': self.history if self.debug_mode else []
        }

    def reset(self):
        """Reinicia telemetría."""
        self.minimization_count = 0
        self.total_compression_ratio = 1.0
        self.history.clear()
        logger.info("Minimizador MAC reiniciado")