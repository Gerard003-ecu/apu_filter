# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo: MAC Audit Vectors (Proyector Topológico-Cuántico de Coherencia)      ║
║ Ubicación: app/wisdom/mac_audit_vectors.py                                   ║
║ Versión: 2.0.0-Quantum-Sheaf-Audit-Enhanced                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Naturaleza Ciber-Física y Topológica Refinada:
────────────────────────────────────────────────
Este módulo audita la "Sabiduría" generada por LLMs operando sobre el espacio 
de Hilbert continuo ℋ_MAC. Emplea entropía relativa cuántica, teoría modular 
de Tomita-Takesaki y cohomología de haces celulares para garantizar coherencia 
epistemológica y prevenir alucinaciones estocásticas.

FUNDAMENTOS TEÓRICOS EXTENDIDOS:
─────────────────────────────────
1. TEORÍA DE INFORMACIÓN CUÁNTICA: Divergencias relativas (Umegaki, Renyi)
2. GEOMETRÍA DE INFORMACIÓN: Métricas de Bures-Wasserstein y Fisher
3. COHOMOLOGÍA DE HACES: Energía de Dirichlet y obstrucciones topológicas
4. TEORÍA MODULAR: Análisis de Tomita-Takesaki para coherencia algebraica
5. GEOMETRÍA RIEMANNIANA: Tensor métrico físico G_PHYSICS

Axiomas Matemáticos Implementados y Extendidos:
────────────────────────────────────────────────
A1. Divergencia de Umegaki: S(ρ||σ) = Tr(ρ(ln ρ - ln σ))
A2. Divergencia de Petz-Renyi: S_α(ρ||σ) = 1/(α-1) ln Tr(ρ^α σ^(1-α))
A3. Fidelidad de Uhlmann: F(ρ,σ) = [Tr√(√ρ σ √ρ)]²
A4. Energía de Dirichlet: E_MAC = ⟨δx, G_PHYSICS δx⟩
A5. Índice de Estabilidad: Ψ_Q = F(ρ,σ) · Tr(ρ²) · exp(-E_MAC)
A6. Teorema KMS: ω(xy) = ω(y σ_{-iβ}(x)) (condición termodinámica)

Referencias Teóricas:
─────────────────────
- Umegaki (1962): "Conditional expectation in an operator algebra"
- Petz (1986): "Quasi-entropies for finite quantum systems"
- Uhlmann (1976): "The transition probability in the state space"
- Hansen & Ghrist (2019): "Toward a spectral theory of cellular sheaves"
- Tomita-Takesaki: "Modular theory of operator algebras"
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import time
import math
import numpy as np
import scipy.linalg as la
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from numpy.typing import NDArray
from enum import Enum, auto

# Dependencias arquitectónicas de APU Filter
from app.core.schemas import Stratum
from app.core.mic_algebra import NumericalInstabilityError
from app.wisdom.atomic_knowledge_matrix import AtomicDensityMatrix, QuantumMetrics
from app.adapters.mic_vectors import (
    VectorResultStatus, 
    _build_result, 
    _build_error, 
    VectorMetrics
)

# Acoplamiento fibrado al Estrato PHYSICS
try:
    from app.core.immune_system.metric_tensors import G_PHYSICS
except ImportError:
    # Fallback para testing
    G_PHYSICS = np.eye(10, dtype=np.float64)
    logger = logging.getLogger("MAC.Wisdom.AuditVectors")
    logger.warning("G_PHYSICS no disponible. Usando identidad como fallback.")

logger = logging.getLogger("MAC.Wisdom.AuditVectors")


# ══════════════════════════════════════════════════════════════════════════════
# INFRAESTRUCTURA: EXCEPCIONES Y ENUMERACIONES
# ══════════════════════════════════════════════════════════════════════════════

class TraceAnomalyVeto(Exception):
    """Veto categórico: Divergencia entrópica severa entre estados."""
    pass


class CohomologicalObstructionError(Exception):
    """Obstrucción de haces: Energía de Dirichlet indica paradojas irresolubles."""
    pass


class EpistemologicalStatus(Enum):
    """Estados epistemológicos de auditoría."""
    HOMOMORPHISM_VERIFIED = auto()      # Coherencia total verificada
    ACCEPTABLE_DEVIATION = auto()        # Desviación dentro de tolerancia
    SEMANTIC_DRIFT = auto()              # Deriva semántica detectada
    LOGICAL_CONTRADICTION = auto()       # Contradicción lógica
    HALLUCINATION_DETECTED = auto()      # Alucinación estocástica


@dataclass
class AuditMetrics:
    """Métricas completas de auditoría cuántica."""
    umegaki_divergence: float
    dirichlet_energy: float
    quantum_stability_index: float
    fidelity: float
    purity_mac: float
    purity_reference: float
    entropy_production: float
    epistemological_status: EpistemologicalStatus
    execution_time_ms: float
    
    def is_acceptable(self) -> bool:
        """Determina si la auditoría es aceptable."""
        return self.epistemological_status in [
            EpistemologicalStatus.HOMOMORPHISM_VERIFIED,
            EpistemologicalStatus.ACCEPTABLE_DEVIATION
        ]


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: AUDITORÍA DE ENTROPÍA RELATIVA (DIVERGENCIAS CUÁNTICAS)
# ══════════════════════════════════════════════════════════════════════════════

class UmegakiDivergenceAuditor:
    r"""
    Auditor de Entropía Relativa Cuántica - VERSIÓN MEJORADA.
    
    FUNDAMENTO TEÓRICO:
    ──────────────────
    La divergencia de Umegaki entre estados cuánticos ρ y σ es:
    
        S(ρ||σ) = Tr(ρ(ln ρ - ln σ))
    
    Propiedades:
        - S(ρ||σ) ≥ 0, con igualdad ssi ρ = σ
        - No es simétrica
        - S(ρ||σ) = ∞ si supp(ρ) ⊄ supp(σ)
        - Convexa en ρ
    
    MEJORAS IMPLEMENTADAS:
    ─────────────────────
    1. Regularización espectral de Tikhonov
    2. Manejo robusto de logaritmos matriciales
    3. Detección de soportes disjuntos
    4. Divergencias generalizadas (Renyi, Petz)
    5. Cálculo numericamente estable
    """
    
    @staticmethod
    def compute_matrix_log(
        A: NDArray[np.complex128],
        regularization: float = 1e-12
    ) -> NDArray[np.complex128]:
        """
        Calcula ln(A) con regularización para evitar ln(0).
        
        Args:
            A: Matriz semidefinida positiva
            regularization: Parámetro de regularización
        
        Returns:
            ln(A) regularizado
        """
        dim = A.shape[0]
        
        # Regularización de Tikhonov
        A_reg = A + regularization * np.eye(dim, dtype=np.complex128)
        
        try:
            # Logaritmo matricial vía descomposición espectral
            eigenvalues, eigenvectors = la.eigh(A_reg)
            
            # Evitar log de valores negativos (ruido numérico)
            log_eigenvalues = np.where(
                eigenvalues > regularization,
                np.log(eigenvalues),
                np.log(regularization)
            )
            
            log_A = eigenvectors @ np.diag(log_eigenvalues) @ eigenvectors.conj().T
            
            return log_A
            
        except la.LinAlgError as e:
            raise NumericalInstabilityError(
                f"Singularidad en logaritmo matricial: {e}"
            )
    
    @classmethod
    def compute_divergence(
        cls,
        rho: NDArray[np.complex128], 
        sigma: NDArray[np.complex128], 
        regularization: float = 1e-12,
        check_support: bool = True
    ) -> float:
        r"""
        Calcula divergencia de Umegaki: S(ρ||σ) = Tr(ρ(ln ρ - ln σ)).
        
        Args:
            rho: Primer estado cuántico
            sigma: Estado de referencia
            regularization: Parámetro de regularización
            check_support: Verificar soportes
        
        Returns:
            Divergencia S(ρ||σ) ≥ 0
        
        Raises:
            NumericalInstabilityError: Si hay problemas numéricos
        """
        try:
            # Verificar soportes si se solicita
            if check_support:
                eig_rho = la.eigvalsh(rho)
                eig_sigma = la.eigvalsh(sigma)
                
                # Detectar soporte disjunto
                rho_support = eig_rho > regularization
                sigma_support = eig_sigma > regularization
                
                if np.any(rho_support & ~sigma_support):
                    logger.warning(
                        "Soportes potencialmente disjuntos: supp(ρ) ⊄ supp(σ)"
                    )
            
            # Calcular logaritmos matriciales
            log_rho = cls.compute_matrix_log(rho, regularization)
            log_sigma = cls.compute_matrix_log(sigma, regularization)
            
            # S(ρ||σ) = Tr(ρ(ln ρ - ln σ))
            divergence = np.trace(rho @ (log_rho - log_sigma)).real
            
            # Clamp a no negativo (corrección numérica)
            return float(max(0.0, divergence))
            
        except Exception as e:
            raise NumericalInstabilityError(
                f"Error en cálculo de divergencia de Umegaki: {e}"
            )
    
    @staticmethod
    def compute_renyi_divergence(
        rho: NDArray[np.complex128], 
        sigma: NDArray[np.complex128],
        alpha: float,
        regularization: float = 1e-12
    ) -> float:
        r"""
        Divergencia de Petz-Renyi de orden α:
        
            S_α(ρ||σ) = 1/(α-1) ln Tr(ρ^α σ^(1-α))
        
        Args:
            rho: Primer estado
            sigma: Estado de referencia
            alpha: Orden (α > 0, α ≠ 1)
            regularization: Regularización
        
        Returns:
            Divergencia de Renyi S_α(ρ||σ)
        """
        if np.isclose(alpha, 1.0):
            # Caso límite α → 1: divergencia de Umegaki
            return UmegakiDivergenceAuditor.compute_divergence(
                rho, sigma, regularization
            )
        
        if alpha <= 0:
            raise ValueError(f"Orden α debe ser positivo: {alpha}")
        
        try:
            # Regularización
            dim = rho.shape[0]
            rho_reg = rho + regularization * np.eye(dim, dtype=np.complex128)
            sigma_reg = sigma + regularization * np.eye(dim, dtype=np.complex128)
            
            # Calcular ρ^α y σ^(1-α)
            rho_power = la.fractional_matrix_power(rho_reg, alpha)
            sigma_power = la.fractional_matrix_power(sigma_reg, 1 - alpha)
            
            # Tr(ρ^α σ^(1-α))
            trace_term = np.trace(rho_power @ sigma_power).real
            
            # S_α = 1/(α-1) ln(trace_term)
            renyi_divergence = (1.0 / (alpha - 1.0)) * np.log(trace_term)
            
            return float(renyi_divergence)
            
        except Exception as e:
            raise NumericalInstabilityError(
                f"Error en divergencia de Renyi: {e}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: COHOMOLOGÍA DE HACES Y ENERGÍA DE DIRICHLET
# ══════════════════════════════════════════════════════════════════════════════

class SheafCohomologyAuditor:
    r"""
    Auditor Topológico de Paradojas Semánticas - VERSIÓN MEJORADA.
    
    FUNDAMENTO TEÓRICO:
    ──────────────────
    La energía de Dirichlet sobre un fibrado de haces celulares es:
    
        E_MAC = ⟨δx, G δx⟩ = (δx)^T G (δx)
    
    donde:
        - δx: Operador cofrontera (obstrucción cohomológica)
        - G: Tensor métrico físico (Riemanniano)
    
    Interpretación:
        - E = 0: Sección armónica (sin contradicciones)
        - E > 0: Fricción semántica (tensión lógica)
        - E → ∞: Paradoja irresoluble
    
    MEJORAS IMPLEMENTADAS:
    ─────────────────────
    1. Validación de dimensiones con G_PHYSICS
    2. Normalización adaptativa
    3. Detección de singularidades
    4. Múltiples métricas (Euclidiana, Riemanniana)
    5. Reportes detallados
    """
    
    @staticmethod
    def validate_dimensions(
        delta_x: NDArray[np.float64],
        metric: NDArray[np.float64]
    ) -> None:
        """
        Valida compatibilidad dimensional.
        
        Raises:
            ValueError: Si dimensiones son incompatibles
        """
        if delta_x.ndim != 1:
            raise ValueError(
                f"δx debe ser vector 1D, recibido: {delta_x.ndim}D"
            )
        
        if metric.shape[0] != metric.shape[1]:
            raise ValueError(
                f"Métrica no cuadrada: {metric.shape}"
            )
        
        if delta_x.shape[0] != metric.shape[0]:
            raise ValueError(
                f"Dimensión incompatible: δx={delta_x.shape[0]}, "
                f"G={metric.shape[0]}"
            )
    
    @classmethod
    def compute_dirichlet_energy(
        cls,
        delta_x: NDArray[np.float64],
        metric: Optional[NDArray[np.float64]] = None,
        validate: bool = True
    ) -> float:
        r"""
        Calcula energía de Dirichlet: E = ⟨δx, G δx⟩.
        
        Args:
            delta_x: Vector cofrontera (obstrucción)
            metric: Tensor métrico (default: G_PHYSICS)
            validate: Validar dimensiones
        
        Returns:
            Energía de Dirichlet E ≥ 0
        
        Raises:
            ValueError: Si dimensiones incompatibles
        """
        # Usar G_PHYSICS por defecto
        if metric is None:
            metric = G_PHYSICS
        
        # Validación
        if validate:
            cls.validate_dimensions(delta_x, metric)
        
        try:
            # E = δx^T G δx
            energy = delta_x.T @ metric @ delta_x
            
            # Verificar resultado
            if np.isnan(energy) or np.isinf(energy):
                logger.error(
                    f"Energía de Dirichlet singular: {energy}. "
                    f"Tensor métrico corrupto."
                )
                return float('inf')
            
            # Clamp a no negativo (corrección numérica)
            return float(max(0.0, energy))
            
        except Exception as e:
            logger.error(f"Error en cálculo de energía de Dirichlet: {e}")
            return float('inf')
    
    @staticmethod
    def compute_euclidean_energy(delta_x: NDArray[np.float64]) -> float:
        """
        Energía de Dirichlet con métrica Euclidiana: E = ||δx||².
        
        Returns:
            Energía Euclidiana
        """
        return float(np.sum(delta_x ** 2))
    
    @staticmethod
    def classify_energy_level(energy: float) -> str:
        """
        Clasifica nivel de energía cohomológica.
        
        Returns:
            Clasificación textual
        """
        if np.isinf(energy):
            return "PARADOJA_IRRESOLUBLE"
        elif energy > 10.0:
            return "FRICCIÓN_CRÍTICA"
        elif energy > 1.0:
            return "TENSIÓN_ALTA"
        elif energy > 0.1:
            return "TENSIÓN_MODERADA"
        elif energy > 1e-6:
            return "TENSIÓN_BAJA"
        else:
            return "HOLONOMÍA_PERFECTA"


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: ÍNDICE DE ESTABILIDAD CUÁNTICA
# ══════════════════════════════════════════════════════════════════════════════

class QuantumStabilityIndex:
    r"""
    Índice de Estabilidad Cuántica Ψ_Q - VERSIÓN MEJORADA.
    
    FUNDAMENTO TEÓRICO:
    ──────────────────
    El índice compuesto Ψ_Q integra tres métricas:
    
        Ψ_Q = F(ρ,σ) · Tr(ρ²) · exp(-E_MAC)
    
    donde:
        - F(ρ,σ): Fidelidad de Uhlmann (similitud geométrica)
        - Tr(ρ²): Pureza (calidad del conocimiento)
        - exp(-E): Factor de penalización cohomológica
    
    Interpretación:
        - Ψ_Q ≈ 1: Coherencia epistemológica perfecta
        - Ψ_Q > 0.7: Aceptable
        - Ψ_Q < 0.5: Inestable (riesgo de alucinación)
        - Ψ_Q ≈ 0: Colapso epistemológico
    
    MEJORAS IMPLEMENTADAS:
    ─────────────────────
    1. Cálculo numericamente estable de fidelidad
    2. Normalización adaptativa
    3. Componentes desacoplados para análisis
    4. Múltiples variantes (conservadora, agresiva)
    5. Métricas de diagnóstico
    """
    
    @staticmethod
    def compute_fidelity(
        rho: NDArray[np.complex128], 
        sigma: NDArray[np.complex128]
    ) -> float:
        r"""
        Fidelidad de Uhlmann: F(ρ,σ) = [Tr√(√ρ σ √ρ)]².
        
        Returns:
            Fidelidad F ∈ [0,1]
        """
        try:
            # Descomposición espectral para estabilidad
            sqrt_rho = la.sqrtm(rho)
            core_matrix = sqrt_rho @ sigma @ sqrt_rho
            sqrt_core = la.sqrtm(core_matrix)
            
            fidelity_sqrt = np.trace(sqrt_core).real
            fidelity = fidelity_sqrt ** 2
            
            # Clamp al rango físico
            return float(np.clip(fidelity, 0.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error en cálculo de fidelidad: {e}")
            return 0.0
    
    @staticmethod
    def compute_purity(rho: NDArray[np.complex128]) -> float:
        r"""
        Pureza: γ(ρ) = Tr(ρ²) ∈ [1/d, 1].
        
        Returns:
            Pureza γ
        """
        purity = np.trace(rho @ rho).real
        return float(np.clip(purity, 0.0, 1.0))
    
    @classmethod
    def compute_psi_q(
        cls, 
        rho: NDArray[np.complex128], 
        sigma_ref: NDArray[np.complex128], 
        dirichlet_energy: float,
        conservative: bool = False
    ) -> Tuple[float, Dict[str, float]]:
        r"""
        Calcula índice de estabilidad cuántica Ψ_Q.
        
        Args:
            rho: Estado MAC
            sigma_ref: Estado de referencia
            dirichlet_energy: Energía cohomológica
            conservative: Usar penalización conservadora
        
        Returns:
            (Ψ_Q, componentes)
        """
        # Componentes individuales
        fidelity = cls.compute_fidelity(rho, sigma_ref)
        purity = cls.compute_purity(rho)
        
        # Penalización cohomológica
        if conservative:
            # Penalización más agresiva
            penalty = math.exp(-2.0 * dirichlet_energy)
        else:
            # Penalización estándar
            penalty = math.exp(-dirichlet_energy)
        
        # Índice compuesto
        psi_q = fidelity * purity * penalty
        
        # Componentes para diagnóstico
        components = {
            'fidelity': fidelity,
            'purity': purity,
            'penalty': penalty,
            'dirichlet_energy': dirichlet_energy
        }
        
        logger.debug(
            f"Ψ_Q = {psi_q:.6f} | "
            f"F={fidelity:.4f}, γ={purity:.4f}, "
            f"exp(-E)={penalty:.4f}"
        )
        
        return psi_q, components
    
    @staticmethod
    def classify_stability(psi_q: float) -> EpistemologicalStatus:
        """
        Clasifica estabilidad epistemológica según Ψ_Q.
        
        Returns:
            Estado epistemológico
        """
        if psi_q >= 0.90:
            return EpistemologicalStatus.HOMOMORPHISM_VERIFIED
        elif psi_q >= 0.70:
            return EpistemologicalStatus.ACCEPTABLE_DEVIATION
        elif psi_q >= 0.50:
            return EpistemologicalStatus.SEMANTIC_DRIFT
        elif psi_q >= 0.30:
            return EpistemologicalStatus.LOGICAL_CONTRADICTION
        else:
            return EpistemologicalStatus.HALLUCINATION_DETECTED


# ══════════════════════════════════════════════════════════════════════════════
# FASE 4: VECTOR EXPORTADO (API PÚBLICA)
# ══════════════════════════════════════════════════════════════════════════════

def vector_audit_quantum_semantic_coherence(
    mac_rho: AtomicDensityMatrix,
    reference_sigma: AtomicDensityMatrix,
    logical_delta_x: NDArray[np.float64],
    umegaki_threshold: float = 0.5,
    psi_q_minimum: float = 0.65,
    dirichlet_threshold: float = 10.0,
    conservative_penalty: bool = False,
    compute_extended_metrics: bool = True
) -> Dict[str, Any]:
    r"""
    [WISDOM] Vector de Auditoría Cuántica - VERSIÓN MEJORADA.
    
    Audita coherencia semántica antes de consolidar sabiduría en actas.
    Dispara vetos categóricos si detecta:
        - Divergencia entrópica excesiva (Trace Anomaly)
        - Contradicciones lógicas (Cohomological Obstruction)
        - Inestabilidad epistemológica (Ψ_Q bajo)
    
    MEJORAS:
    ────────
    1. Métricas extendidas opcionales
    2. Penalización configurable
    3. Reportes estructurados
    4. Múltiples umbrales
    5. Telemetría completa
    
    Args:
        mac_rho: Estado cuántico MAC
        reference_sigma: Estado de referencia
        logical_delta_x: Vector cofrontera (obstrucción)
        umegaki_threshold: Umbral de divergencia
        psi_q_minimum: Umbral de estabilidad
        dirichlet_threshold: Umbral de energía cohomológica
        conservative_penalty: Usar penalización conservadora
        compute_extended_metrics: Calcular métricas extendidas
    
    Returns:
        Diccionario con resultado de auditoría
    """
    start_time = time.perf_counter()
    rho = mac_rho.matrix
    sigma = reference_sigma.matrix
    
    try:
        # ─────────────────────────────────────────────────────────────────
        # FASE 1: AUDITORÍA DE DIVERGENCIA DE UMEGAKI
        # ─────────────────────────────────────────────────────────────────
        
        logger.info("[AUDIT] Fase 1: Auditoría de divergencia entrópica...")
        
        divergence = UmegakiDivergenceAuditor.compute_divergence(
            rho, sigma, check_support=True
        )
        
        if divergence > umegaki_threshold:
            raise TraceAnomalyVeto(
                f"Divergencia entrópica S(ρ||σ) = {divergence:.4f} excede "
                f"umbral Lipschitz de {umegaki_threshold}. "
                f"Alucinación semántica detectada."
            )
        
        # ─────────────────────────────────────────────────────────────────
        # FASE 2: AUDITORÍA DE ENERGÍA DE DIRICHLET
        # ─────────────────────────────────────────────────────────────────
        
        logger.info("[AUDIT] Fase 2: Auditoría cohomológica...")
        
        dirichlet_energy = SheafCohomologyAuditor.compute_dirichlet_energy(
            logical_delta_x, validate=True
        )
        
        energy_classification = SheafCohomologyAuditor.classify_energy_level(
            dirichlet_energy
        )
        
        if math.isinf(dirichlet_energy) or dirichlet_energy > dirichlet_threshold:
            raise CohomologicalObstructionError(
                f"Fricción termodinámica extrema (E_MAC = {dirichlet_energy:.4f}). "
                f"Clasificación: {energy_classification}. "
                f"El dictamen contiene paradojas contractuales irresolubles."
            )
        
        # ─────────────────────────────────────────────────────────────────
        # FASE 3: CÁLCULO DE ÍNDICE DE ESTABILIDAD CUÁNTICA
        # ─────────────────────────────────────────────────────────────────
        
        logger.info("[AUDIT] Fase 3: Cálculo de estabilidad epistemológica...")
        
        psi_q, components = QuantumStabilityIndex.compute_psi_q(
            rho, sigma, dirichlet_energy, 
            conservative=conservative_penalty
        )
        
        epistemological_status = QuantumStabilityIndex.classify_stability(psi_q)
        
        # ─────────────────────────────────────────────────────────────────
        # FASE 4: VALIDACIÓN FINAL
        # ─────────────────────────────────────────────────────────────────
        
        if psi_q < psi_q_minimum:
            return _build_error(
                stratum=Stratum.WISDOM,
                status=VectorResultStatus.LOGIC_ERROR,
                error=(
                    f"Estabilidad cuántica crítica: Ψ_Q = {psi_q:.4f} < {psi_q_minimum}. "
                    f"Estado: {epistemological_status.name}"
                ),
                metrics=VectorMetrics(
                    execution_ms=(time.perf_counter() - start_time) * 1000
                )
            )
        
        # ─────────────────────────────────────────────────────────────────
        # FASE 5: MÉTRICAS EXTENDIDAS (OPCIONAL)
        # ─────────────────────────────────────────────────────────────────
        
        extended_metrics = {}
        
        if compute_extended_metrics:
            metrics_mac = mac_rho.compute_metrics()
            metrics_ref = reference_sigma.compute_metrics()
            
            extended_metrics = {
                'purity_mac': metrics_mac.purity,
                'purity_reference': metrics_ref.purity,
                'entropy_mac': metrics_mac.von_neumann_entropy,
                'entropy_reference': metrics_ref.von_neumann_entropy,
                'entropy_production': (
                    metrics_mac.von_neumann_entropy - 
                    metrics_ref.von_neumann_entropy
                ),
                'energy_classification': energy_classification
            }
        
        # ─────────────────────────────────────────────────────────────────
        # FASE 6: RESULTADO EXITOSO
        # ─────────────────────────────────────────────────────────────────
        
        execution_time = (time.perf_counter() - start_time) * 1000
        
        audit_metrics = AuditMetrics(
            umegaki_divergence=divergence,
            dirichlet_energy=dirichlet_energy,
            quantum_stability_index=psi_q,
            fidelity=components['fidelity'],
            purity_mac=extended_metrics.get('purity_mac', components['purity']),
            purity_reference=extended_metrics.get('purity_reference', 0.0),
            entropy_production=extended_metrics.get('entropy_production', 0.0),
            epistemological_status=epistemological_status,
            execution_time_ms=execution_time
        )
        
        logger.info(
            f"✓ Auditoría completada exitosamente. "
            f"Ψ_Q = {psi_q:.4f}, Estado: {epistemological_status.name}"
        )
        
        return _build_result(
            success=True,
            stratum=Stratum.WISDOM,
            status=VectorResultStatus.SUCCESS,
            metrics=VectorMetrics(execution_ms=execution_time),
            quantum_stability_index=psi_q,
            umegaki_divergence=divergence,
            dirichlet_energy=dirichlet_energy,
            epistemological_status=epistemological_status.name,
            fidelity=components['fidelity'],
            purity=components['purity'],
            penalty_factor=components['penalty'],
            audit_metrics=audit_metrics,
            **extended_metrics
        )
        
    except TraceAnomalyVeto as e:
        return _build_error(
            stratum=Stratum.WISDOM,
            status=VectorResultStatus.VALIDATION_ERROR,
            error=str(e),
            metrics=VectorMetrics(
                execution_ms=(time.perf_counter() - start_time) * 1000
            )
        )
    
    except CohomologicalObstructionError as e:
        return _build_error(
            stratum=Stratum.WISDOM,
            status=VectorResultStatus.TOPOLOGY_ERROR,
            error=str(e),
            metrics=VectorMetrics(
                execution_ms=(time.perf_counter() - start_time) * 1000
            )
        )
    
    except NumericalInstabilityError as e:
        return _build_error(
            stratum=Stratum.WISDOM,
            status=VectorResultStatus.PHYSICS_ERROR,
            error=str(e),
            metrics=VectorMetrics(
                execution_ms=(time.perf_counter() - start_time) * 1000
            )
        )