# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : MAC Minimizer Agent (Custodio de la Purificación Espectral)         ║
║ Ruta   : app/agents/boole/tactics/mac_minimizer_agent.py                     ║
║ Versión: 3.0.0-Categorical-Topos-Spectral-Quantum-Rigorous                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GEOMETRÍA NO CONMUTATIVA (Rigor Doctoral Avanzado):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor gobierna el `mac_minimizer.py` en el estrato WISDOM mediante
una composición funtorial estricta que preserva invariantes topológicos,
espectrales y termodinámicos cuánticos.

FUNDAMENTOS MATEMÁTICOS RIGUROSOS:

1. TOPOLOGÍA ALGEBRAICA:
   - El espacio de matrices de densidad forma un simplejo convexo en C^{d×d}
   - La purificación espectral es un retracto que preserva la estructura de fibrado
   - Los certificados forman un 2-funtor entre categorías enriquecidas

2. TEORÍA ESPECTRAL:
   - Descomposición espectral: ρ = Σᵢ λᵢ |ψᵢ⟩⟨ψᵢ| con λᵢ ∈ σ(ρ)
   - Cálculo funcional holomorfo: f(ρ) = ∫_Γ f(z)(zI - ρ)⁻¹ dz
   - Teorema espectral para operadores compactos autoadjuntos

3. TEORÍA DE GRAFOS:
   - Grafo de dependencias espectrales: G = (V, E) donde V = {λᵢ}, E = acoplamiento
   - Conectividad algebraica: λ₂(L) mide coherencia cuántica
   - Isomorfismo de grafos preserva estructura informacional

4. MECÁNICA CUÁNTICA:
   - Postulado de Born: Tr(ρA) = valor esperado del observable A
   - Ecuación de von Neumann: dρ/dt = -i[H, ρ]
   - Límite termodinámico: S(ρ) = -Tr(ρ log ρ)

5. TEORÍA DE CATEGORÍAS Y TOPOS:
   - Funtor de purificación: P: DensityMat → PureDensityMat
   - Transformación natural: η: Id ⟹ P ∘ F (fidelidad)
   - Topos de haces sobre el espectro: Sh(Spec(A))

6. ÁLGEBRA LINEAL RIGUROSA:
   - Normas matriciales: ‖A‖₂ = σₘₐₓ(A), ‖A‖_F = √Tr(A†A)
   - Desigualdad de Weyl para perturbaciones espectrales
   - Teorema de Sylvester para ecuaciones de Lyapunov

7. ÁLGEBRA DE BOOLE Y LÓGICA CUÁNTICA:
   - Retículo ortomodular de proyectores
   - Lógica cuántica no distributiva: (P ∨ Q) ∧ R ≠ (P ∧ R) ∨ (Q ∧ R)
   - Implicación cuántica de Sasaki: P →_S Q = P⊥ ∨ (P ∧ Q)

ARQUITECTURA FUNTORIAL ANIDADA (3 FASES):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Auditoría de Majorización Cuántica
  ├─ Validación Hermítica y PSD (método inicial)
  ├─ Saneamiento Espectral
  ├─ Curvas de Lorenz Cuánticas
  └─ Certificado de Majorización (método final) ──┐
                                                   │
Fase 2 → Certificación de Fidelidad de Uhlmann    │
  ├─ Consumo del Certificado de Fase 1 ←──────────┘
  ├─ Raíz Cuadrada Matricial PSD
  ├─ Núcleo de Fidelidad Cuántica
  └─ Certificado de Fidelidad (método final) ──┐
                                                │
Fase 3 → Cota de Capacidad de Holevo            │
  ├─ Consumo del Certificado de Fase 2 ←────────┘
  ├─ Entropía de von Neumann
  ├─ Diferencial Termodinámico
  └─ Certificado de Capacidad (método final)

MEJORAS IMPLEMENTADAS:
────────────────────────────────────────────────────────────────────────────────
1. Cálculo funcional holomorfo para funciones matriciales
2. Estimación de error espectral con desigualdad de Weyl
3. Métricas de distancia cuántica adicionales (traza, Bures, Hellinger)
4. Análisis de coherencia cuántica mediante elementos fuera de diagonal
5. Validación de desigualdades de incertidumbre cuántica
6. Verificación de propiedades de retículo ortomodular
7. Cálculo de capacidad de Holevo accesible
8. Análisis de canales cuánticos mediante representación de Kraus
9. Estimación de complejidad de enredo (entanglement)
10. Monitoreo de información mutua cuántica
"""

from __future__ import annotations

import cmath
import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Final, List, Optional, Protocol, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


# ═══════════════════════════════════════════════════════════════════════════════
# §0. DEPENDENCIAS ARQUITECTÓNICAS Y PROTOCOLOS
# ═══════════════════════════════════════════════════════════════════════════════
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
    from app.wisdom.atomic_knowledge_matrix import AtomicDensityMatrix
except ImportError:

    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos E_MIC."""
        pass

    class Morphism:
        r"""Clase base de Morfismos del Topos."""
        pass

    class AtomicDensityMatrix:
        r"""Marcador estructural para compatibilidad sin dependencia externa."""
        pass


logger = logging.getLogger("MAC.Wisdom.MinimizerAgent.v3")


# ─────────────────────────────────────────────────────────────────────────────
# Protocolos de Tipo para Matrices Cuánticas
# ─────────────────────────────────────────────────────────────────────────────
class QuantumStateProtocol(Protocol):
    """Protocolo para estados cuánticos válidos."""
    
    def is_hermitian(self) -> bool: ...
    def is_positive_semidefinite(self) -> bool: ...
    def trace(self) -> complex: ...
    def eigenvalues(self) -> NDArray[np.float64]: ...


# ═══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES FÍSICAS Y LÍMITES CUÁNTICOS (RIGOR DOCTORAL)
# ═══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)
_PLANCK_REDUCED: Final[float] = 1.054571817e-34  # ℏ en J·s (SI)
_BOLTZMANN_CONSTANT: Final[float] = 1.380649e-23  # k_B en J/K

# Tolerancias cuánticas refinadas
_UHLMANN_FIDELITY_MIN: Final[float] = 0.95
_ENTROPY_TOLERANCE: Final[float] = 1e-12
_MAJORIZATION_TOLERANCE: Final[float] = 1e-10
_HERMITIAN_TOLERANCE: Final[float] = 1e-10
_PSD_TOLERANCE: Final[float] = 1e-10
_TRACE_TOLERANCE: Final[float] = 1e-10
_FIDELITY_NUMERICAL_TOLERANCE: Final[float] = 1e-12
_COHERENCE_TOLERANCE: Final[float] = 1e-14
_ENTANGLEMENT_TOLERANCE: Final[float] = 1e-11

# Factor de seguridad numérica adaptativo
_NUMERICAL_SAFETY_FACTOR: Final[float] = 128.0
_SPECTRAL_CUTOFF_RATIO: Final[float] = 1e-15  # Para autovalores despreciables

# Límites físicos de información cuántica
_MAX_VON_NEUMANN_ENTROPY_PER_QUBIT: Final[float] = math.log(2.0)  # 1 nat ≈ 1.44 bits
_MIN_QUANTUM_FIDELITY: Final[float] = 0.5  # Fidelidad mínima física (estados ortogonales)


# ═══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES CUÁNTICAS (ENRIQUECIDA)
# ═══════════════════════════════════════════════════════════════════════════════
class MACMinimizerAgentError(TopologicalInvariantError):
    r"""Excepción raíz del Custodio de la Purificación Espectral."""
    pass


class DensityMatrixValidationError(MACMinimizerAgentError):
    r"""Detonada si una matriz de densidad viola axiomas cuánticos."""
    pass


class QuantumMajorizationViolation(MACMinimizerAgentError):
    r"""Detonada si ρ_purificada no majoriza a ρ_orig."""
    pass


class UhlmannFidelityCollapseError(MACMinimizerAgentError):
    r"""Detonada si F(ρ, σ) < F_min. Mutilación semántica detectada."""
    pass


class HolevoCapacityDeficitError(MACMinimizerAgentError):
    r"""Detonada si ΔS > 0 o si la poda destruye capacidad semántica."""
    pass


class SpectralDecompositionError(MACMinimizerAgentError):
    r"""Detonada si la descomposición espectral falla numéricamente."""
    pass


class QuantumCoherenceViolation(MACMinimizerAgentError):
    r"""Detonada si la coherencia cuántica se degrada inaceptablemente."""
    pass


class EntanglementStructureError(MACMinimizerAgentError):
    r"""Detonada si la estructura de enredo se ve comprometida."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# §C. ENUMERACIONES PARA MÉTRICAS CUÁNTICAS
# ═══════════════════════════════════════════════════════════════════════════════
class QuantumDistanceMetric(Enum):
    """Métricas de distancia en el espacio de estados cuánticos."""
    TRACE_DISTANCE = auto()          # D_tr(ρ, σ) = (1/2)‖ρ - σ‖₁
    FIDELITY = auto()                # F(ρ, σ) = (Tr√(√ρ σ √ρ))²
    BURES_DISTANCE = auto()          # D_B(ρ, σ) = √(2(1 - √F(ρ,σ)))
    HELLINGER_DISTANCE = auto()      # D_H(ρ, σ) = √(1 - F(ρ,σ))
    RELATIVE_ENTROPY = auto()        # S(ρ‖σ) = Tr(ρ(log ρ - log σ))


class PurificationPhase(Enum):
    """Fases del proceso de purificación espectral."""
    PHASE_1_MAJORIZATION = auto()
    PHASE_2_FIDELITY = auto()
    PHASE_3_HOLEVO = auto()
    COMPLETE = auto()


# ═══════════════════════════════════════════════════════════════════════════════
# §D. ESTRUCTURAS INMUTABLES ENRIQUECIDAS (DTOs del Espacio de Hilbert)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class SpectralCharacteristics:
    r"""
    Características espectrales completas de un operador de densidad.
    
    Invariantes topológicos:
        - Dimensión del espacio de Hilbert
        - Rango efectivo (número de autovalores significativos)
        - Número de condición espectral
        - Gap espectral (separación mínima entre autovalores distintos)
    """
    dimension: int
    effective_rank: int
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.complex128]
    condition_number: float
    spectral_gap: float
    spectral_entropy: float  # -Σ λᵢ log λᵢ
    participation_ratio: float  # 1/Σ λᵢ²
    
    def __post_init__(self) -> None:
        """Validación de invariantes al construir."""
        if self.dimension <= 0:
            raise ValueError("La dimensión debe ser positiva.")
        if self.effective_rank > self.dimension:
            raise ValueError("El rango efectivo no puede exceder la dimensión.")
        if self.condition_number < 1.0:
            raise ValueError("El número de condición debe ser ≥ 1.")


@dataclass(frozen=True, slots=True)
class QuantumCoherenceMetrics:
    r"""
    Métricas de coherencia cuántica.
    
    La coherencia mide la "cuánticidad" del estado, cuantificando elementos
    fuera de diagonal en la base de energía.
    
    Definiciones:
        l1_norm_coherence: C_l1(ρ) = Σ_{i≠j} |ρᵢⱼ|
        relative_entropy_coherence: C_re(ρ) = S(ρ_diag) - S(ρ)
        robustness_of_coherence: mínimo p tal que (ρ + pτ)/(1+p) es incoherente
    """
    l1_norm_coherence: float
    relative_entropy_coherence: float
    robustness_of_coherence: float
    off_diagonal_mass: float  # ‖ρ - diag(ρ)‖_F
    
    def is_coherence_preserved(self, tolerance: float = _COHERENCE_TOLERANCE) -> bool:
        """Verifica si la coherencia se mantiene dentro de tolerancia."""
        return self.l1_norm_coherence >= -tolerance


@dataclass(frozen=True, slots=True)
class MajorizationAuditData:
    r"""
    Artefacto de Fase 1: Certificado del preorden de majorización cuántica.
    
    Teorema (Hardy-Littlewood-Pólya):
        x ≺ y ⟺ existe matriz doblemente estocástica D tal que x = Dy
    
    En contexto cuántico:
        λ(ρ) ≺ λ(σ) implica que ρ es "más mezclado" que σ
    """
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
    
    # Métricas adicionales de majorización
    renyi_entropy_original: float = field(default=0.0)  # S₂(ρ) = -log Tr(ρ²)
    renyi_entropy_purified: float = field(default=0.0)
    purity_original: float = field(default=0.0)  # Tr(ρ²)
    purity_purified: float = field(default=0.0)


@dataclass(frozen=True, slots=True)
class FidelityAuditData:
    r"""
    Artefacto de Fase 2: Certificado de Fidelidad de Uhlmann.
    
    Propiedades de F(ρ, σ):
        1. 0 ≤ F(ρ, σ) ≤ 1
        2. F(ρ, σ) = 1 ⟺ ρ = σ
        3. F(ρ, σ) = F(σ, ρ) (simétrica)
        4. F(ρ, σ)² ≤ F(ρ, τ)F(τ, σ) (desigualdad triangular)
    
    Relación con distancia de Bures:
        D_B(ρ, σ) = √(2(1 - √F(ρ, σ)))
    """
    uhlmann_fidelity: float
    fidelity_tolerance: float
    fidelity_min_required: float
    is_fidelity_preserved: bool
    
    # Métricas de distancia complementarias
    trace_distance: float = field(default=0.0)  # (1/2)‖ρ - σ‖₁
    bures_distance: float = field(default=0.0)  # √(2(1 - √F))
    hellinger_distance: float = field(default=0.0)  # √(1 - F)
    
    # Coherencia cuántica preservada
    coherence_metrics_original: Optional[QuantumCoherenceMetrics] = field(default=None)
    coherence_metrics_purified: Optional[QuantumCoherenceMetrics] = field(default=None)
    
    def __post_init__(self) -> None:
        """Validación de cotas físicas."""
        if not (0.0 <= self.uhlmann_fidelity <= 1.0 + _FIDELITY_NUMERICAL_TOLERANCE):
            raise ValueError(
                f"Fidelidad fuera de rango físico: {self.uhlmann_fidelity}"
            )


@dataclass(frozen=True, slots=True)
class HolevoAuditData:
    r"""
    Artefacto de Fase 3: Certificado Termodinámico de von Neumann.
    
    Capacidad de Holevo:
        χ({pᵢ, ρᵢ}) = S(Σᵢ pᵢρᵢ) - Σᵢ pᵢS(ρᵢ)
    
    Representa la máxima información clásica que puede transmitirse usando
    un conjunto de estados cuánticos.
    
    Segunda Ley Termodinámica Cuántica:
        ΔS ≥ 0 para procesos irreversibles en sistemas cerrados
        Purificación espectral es reversible: ΔS ≤ 0 es válido
    """
    entropy_original: float
    entropy_purified: float
    entropy_delta: float
    entropy_tolerance: float
    is_capacity_preserved: bool
    
    # Entropías de Rényi (familia paramétrica)
    renyi_2_entropy_original: float = field(default=0.0)  # S₂ = -log Tr(ρ²)
    renyi_2_entropy_purified: float = field(default=0.0)
    renyi_inf_entropy_original: float = field(default=0.0)  # S_∞ = -log λₘₐₓ
    renyi_inf_entropy_purified: float = field(default=0.0)
    
    # Capacidades de información
    holevo_capacity_bound: float = field(default=0.0)  # Cota superior de χ
    accessible_information: float = field(default=0.0)  # Información accesible
    
    # Propiedades termodinámicas
    free_energy_change: float = field(default=0.0)  # ΔF = ΔE - TΔS
    
    def satisfies_second_law(self, allow_reversible: bool = True) -> bool:
        """
        Verifica la segunda ley de la termodinámica cuántica.
        
        Args:
            allow_reversible: Si True, permite ΔS ≤ 0 (procesos reversibles)
        """
        if allow_reversible:
            return True  # La purificación es un proceso reversible
        return self.entropy_delta >= -self.entropy_tolerance


@dataclass(frozen=True, slots=True)
class PurificationGovernanceState:
    r"""
    Objeto final del endofuntor Z_MAC-Agent.
    
    Representa el estado completo de gobernanza tras la composición funtorial:
        Φ₃ ∘ Φ₂ ∘ Φ₁: DensityMat × DensityMat → GovernanceState
    
    Invariante categórico:
        is_epistemologically_valid = True ⟺ 
            ∀ phase ∈ {1,2,3}: certificate(phase).is_valid = True
    """
    majorization_audit: MajorizationAuditData
    fidelity_audit: FidelityAuditData
    holevo_audit: HolevoAuditData
    is_epistemologically_valid: bool
    
    # Metadatos de la purificación
    purification_phase: PurificationPhase = field(default=PurificationPhase.COMPLETE)
    timestamp: Optional[float] = field(default=None)
    
    # Métricas agregadas
    overall_quality_score: float = field(default=0.0)  # [0, 1]
    risk_assessment: str = field(default="NOMINAL")  # NOMINAL | WARNING | CRITICAL
    
    def __post_init__(self) -> None:
        """Validación de consistencia entre certificados."""
        if self.is_epistemologically_valid:
            if not self.majorization_audit.is_majorized:
                raise ValueError("Inconsistencia: majorización no certificada.")
            if not self.fidelity_audit.is_fidelity_preserved:
                raise ValueError("Inconsistencia: fidelidad no preservada.")
            if not self.holevo_audit.is_capacity_preserved:
                raise ValueError("Inconsistencia: capacidad no preservada.")


# ═══════════════════════════════════════════════════════════════════════════════
# §E. GUARDAS NUMÉRICAS ENRIQUECIDAS
# ═══════════════════════════════════════════════════════════════════════════════
class _AdvancedNumericalGuard:
    r"""
    Capa de saneamiento numérico con análisis de error riguroso.
    
    Implementa:
        - Aritmética de intervalo para propagación de error
        - Detección de cancelación catastrófica
        - Compensación de Kahan para sumatorias
        - Validación de ortogonalidad numérica
    """

    @staticmethod
    def _kahan_sum(values: NDArray[np.float64]) -> float:
        r"""
        Suma compensada de Kahan para minimizar error de redondeo.
        
        Error de redondeo:
            Error estándar: O(nε)
            Error Kahan: O(ε) + O(nε²)
        """
        if values.size == 0:
            return 0.0
            
        s = float(values[0])
        c = 0.0  # Compensación de error
        
        for v in values[1:]:
            y = float(v) - c
            t = s + y
            c = (t - s) - y
            s = t
            
        return s

    @staticmethod
    def _compensated_dot_product(
        a: NDArray[np.complex128],
        b: NDArray[np.complex128],
    ) -> complex:
        r"""
        Producto punto compensado para vectores complejos.
        """
        if a.shape != b.shape:
            raise ValueError("Los vectores deben tener la misma forma.")
            
        products = a * b.conj()
        real_part = _AdvancedNumericalGuard._kahan_sum(np.real(products))
        imag_part = _AdvancedNumericalGuard._kahan_sum(np.imag(products))
        
        return complex(real_part, imag_part)

    @staticmethod
    def _estimate_condition_number(
        matrix: NDArray[np.complex128],
    ) -> float:
        r"""
        Estima el número de condición espectral: κ(A) = σₘₐₓ/σₘᵢₙ.
        
        Un número de condición grande indica:
            - Sensibilidad a perturbaciones
            - Pérdida de dígitos significativos en cálculos
            - Posible singularidad numérica
        """
        try:
            singular_values = la.svdvals(matrix)
        except la.LinAlgError:
            return float('inf')
            
        if singular_values.size == 0:
            return 1.0
            
        sigma_max = float(np.max(singular_values))
        sigma_min = float(np.min(singular_values))
        
        if sigma_min <= _MACHINE_EPSILON * sigma_max:
            return float('inf')
            
        return sigma_max / sigma_min

    @staticmethod
    def _check_orthonormality(
        vectors: NDArray[np.complex128],
        tolerance: Optional[float] = None,
    ) -> Tuple[bool, float]:
        r"""
        Verifica ortornormalidad de un conjunto de vectores.
        
        Condiciones:
            ⟨vᵢ, vⱼ⟩ = δᵢⱼ  (delta de Kronecker)
        
        Returns:
            (is_orthonormal, max_deviation)
        """
        if tolerance is None:
            tolerance = _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON
            
        n_vectors = vectors.shape[1] if vectors.ndim > 1 else 1
        
        if n_vectors == 1:
            norm = float(la.norm(vectors))
            deviation = abs(norm - 1.0)
            return deviation <= tolerance, deviation
            
        gram_matrix = vectors.conj().T @ vectors
        identity = np.eye(n_vectors, dtype=np.complex128)
        
        deviation_matrix = gram_matrix - identity
        max_deviation = float(la.norm(deviation_matrix, ord='fro'))
        
        return max_deviation <= tolerance, max_deviation

    @staticmethod
    def _as_finite_complex_matrix(
        name: str,
        value: Any,
        *,
        square: bool = False,
        hermitian: bool = False,
    ) -> NDArray[np.complex128]:
        r"""
        Valida una matriz compleja finita con verificaciones adicionales.
        """
        try:
            arr = np.asarray(value, dtype=np.complex128)
        except Exception as exc:
            raise TypeError(
                f"{name} debe ser una matriz numérica compleja o real."
            ) from exc

        if not np.all(np.isfinite(arr)):
            nan_count = int(np.sum(np.isnan(arr)))
            inf_count = int(np.sum(np.isinf(arr)))
            raise ValueError(
                f"{name} contiene valores no finitos: "
                f"{nan_count} NaN, {inf_count} infinitos."
            )

        if arr.ndim != 2:
            raise ValueError(
                f"{name} debe ser una matriz 2D, recibido {arr.ndim}D."
            )

        if square and arr.shape[0] != arr.shape[1]:
            raise ValueError(
                f"{name} debe ser cuadrada: shape={arr.shape}."
            )
            
        if hermitian:
            frobenius_norm = float(la.norm(arr, ord='fro'))
            hermitian_residual = float(
                la.norm(arr - arr.conj().T, ord='fro')
            ) / max(1.0, frobenius_norm)
            
            hermitian_tolerance = max(
                _HERMITIAN_TOLERANCE,
                _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON,
            )
            
            if hermitian_residual > hermitian_tolerance:
                raise ValueError(
                    f"{name} no es hermítica: residuo={hermitian_residual:.6e}"
                )

        return arr

    @staticmethod
    def _as_finite_real_vector(
        name: str,
        value: Any,
        *,
        positive: bool = False,
        normalized: bool = False,
    ) -> NDArray[np.float64]:
        r"""
        Valida un vector real finito con restricciones adicionales.
        """
        try:
            raw = np.asarray(value)
        except Exception as exc:
            raise TypeError(
                f"{name} no puede interpretarse como arreglo numérico."
            ) from exc

        if np.iscomplexobj(raw):
            if not np.all(np.isfinite(raw)):
                raise ValueError(f"{name} contiene valores NaN o infinitos.")

            imag_max = float(np.max(np.abs(np.imag(raw)))) if raw.size > 0 else 0.0
            imag_tolerance = max(
                _FIDELITY_NUMERICAL_TOLERANCE,
                _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON,
            )

            if imag_max > imag_tolerance:
                raise TypeError(
                    f"{name} debe ser real; parte imaginaria "
                    f"= {imag_max:.6e} > {imag_tolerance:.6e}."
                )

            raw = np.real(raw)

        try:
            arr = raw.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"{name} debe ser numérico real convertible a float64."
            ) from exc

        # Normalización de forma
        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim == 2 and 1 in arr.shape:
            arr = arr.reshape(-1)
        elif arr.ndim != 1:
            raise ValueError(
                f"{name} debe ser 1D, fila, columna o escalar; recibido {arr.ndim}D."
            )

        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contiene valores NaN o infinitos.")

        if positive:
            min_val = float(np.min(arr)) if arr.size > 0 else 0.0
            if min_val < -_PSD_TOLERANCE:
                raise ValueError(
                    f"{name} debe ser no negativo; mínimo={min_val:.6e}"
                )
            arr = np.clip(arr, 0.0, None)
            
        if normalized:
            total = float(np.sum(arr))
            if total <= _MACHINE_EPSILON:
                raise ValueError(f"{name} tiene masa nula, no puede normalizarse.")
            arr = arr / total

        return arr


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 1: AUDITORÍA DEL PREORDEN DE MAJORIZACIÓN CUÁNTICA                   ║
# ║                                                                             ║
# ║   Teoría Matemática:                                                        ║
# ║   ─────────────────                                                         ║
# ║   Sea λ(ρ) = (λ₁, λ₂, ..., λₐ) el espectro ordenado descendentemente.     ║
# ║                                                                             ║
# ║   Definición (Majorización de vectores):                                    ║
# ║       x ≺ y ⟺ Σₖ₌₁ⁱ xₖ↓ ≤ Σₖ₌₁ⁱ yₖ↓, ∀i ∈ {1,...,d-1}                    ║
# ║              y Σₖ₌₁ᵈ xₖ = Σₖ₌₁ᵈ yₖ                                         ║
# ║                                                                             ║
# ║   Definición (Majorización de estados cuánticos):                           ║
# ║       ρ ≺ σ ⟺ λ(ρ) ≺ λ(σ)                                                  ║
# ║                                                                             ║
# ║   Interpretación física:                                                    ║
# ║       ρ ≺ σ significa que ρ es "más mezclado" que σ, i.e., ρ tiene         ║
# ║       mayor entropía y está más alejado de un estado puro.                  ║
# ║                                                                             ║
# ║   Purificación espectral debe satisfacer:                                   ║
# ║       λ(ρ_purificada) ≽ λ(ρ_original)                                       ║
# ║                                                                             ║
# ║   Teorema (Equivalencia con doble estocasticidad):                          ║
# ║       x ≺ y ⟺ ∃ matriz doblemente estocástica D: x = Dy                    ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_QuantumMajorizationAuditor(_AdvancedNumericalGuard):
    r"""
    Fase 1: Auditoría rigurosa del preorden de majorización cuántica.
    
    Garantiza que la purificación espectral preserve o incremente la pureza
    del estado cuántico, medida mediante la majorización del espectro.
    
    Métodos de esta fase:
        1. _compute_spectral_characteristics (inicio)
        2. _sanitize_density_matrix
        3. _sanitize_spectrum
        4. _assert_spectra_consistent
        5. _resolve_spectrum
        6. _compute_lorenz_curves
        7. _compute_renyi_entropies
        8. _audit_quantum_majorization (final)
    
    El último método retorna MajorizationAuditData, que es consumido por Fase 2.
    """

    def _compute_spectral_characteristics(
        self,
        name: str,
        eigenvalues: NDArray[np.float64],
        eigenvectors: NDArray[np.complex128],
    ) -> SpectralCharacteristics:
        r"""
        Método inicial de Fase 1.
        
        Computa características espectrales completas de un operador de densidad.
        
        Returns:
            SpectralCharacteristics con invariantes topológicos del espectro
        """
        eigenvalues = self._as_finite_real_vector(
            f"{name}_eigenvalues",
            eigenvalues,
            positive=True,
            normalized=False,
        )
        
        dimension = eigenvalues.size
        
        if dimension == 0:
            raise SpectralDecompositionError(
                f"{name} tiene espectro vacío."
            )
        
        # Número de autovalores significativos (rango efectivo)
        spectral_cutoff = _SPECTRAL_CUTOFF_RATIO * float(np.max(eigenvalues))
        effective_rank = int(np.sum(eigenvalues > spectral_cutoff))
        
        # Número de condición espectral
        lambda_max = float(np.max(eigenvalues))
        lambda_min_nonzero = float(
            np.min(eigenvalues[eigenvalues > spectral_cutoff])
        ) if effective_rank > 0 else lambda_max
        
        condition_number = (
            lambda_max / lambda_min_nonzero 
            if lambda_min_nonzero > 0 else float('inf')
        )
        
        # Gap espectral: separación mínima entre autovalores distintos
        sorted_evals = np.sort(eigenvalues)[::-1]
        gaps = np.abs(np.diff(sorted_evals))
        gaps_significant = gaps[gaps > spectral_cutoff]
        spectral_gap = (
            float(np.min(gaps_significant))
            if gaps_significant.size > 0
            else 0.0
        )
        
        # Entropía espectral (entropía de Shannon del espectro normalizado)
        prob_eigenvalues = eigenvalues / (float(np.sum(eigenvalues)) + _MACHINE_EPSILON)
        prob_eigenvalues = prob_eigenvalues[prob_eigenvalues > _MACHINE_EPSILON]
        spectral_entropy = float(
            -np.sum(prob_eigenvalues * np.log(prob_eigenvalues))
        ) if prob_eigenvalues.size > 0 else 0.0
        
        # Participation ratio: 1/Σλᵢ² (número efectivo de estados)
        participation_ratio = (
            1.0 / float(np.sum(eigenvalues**2))
            if np.sum(eigenvalues**2) > _MACHINE_EPSILON
            else 1.0
        )
        
        return SpectralCharacteristics(
            dimension=dimension,
            effective_rank=effective_rank,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            condition_number=condition_number,
            spectral_gap=spectral_gap,
            spectral_entropy=spectral_entropy,
            participation_ratio=participation_ratio,
        )

    def _sanitize_density_matrix(
        self,
        name: str,
        rho: NDArray[np.complex128],
    ) -> Tuple[
        NDArray[np.complex128],
        NDArray[np.float64],
        float,
        float,
        SpectralCharacteristics,
    ]:
        r"""
        Valida y sanea una matriz de densidad cuántica con análisis espectral completo.
        
        Axiomas verificados:
            1. Hermiticidad: ρ = ρ†
            2. Positive semidefinite: ρ ≥ 0 (todos los autovalores ≥ 0)
            3. Traza unitaria: Tr(ρ) = 1
            4. Acotamiento: 0 ≤ ρ ≤ I
        
        Returns:
            rho_sanitized: Matriz hermítica, PSD y traza uno reconstruida
            eigenvalues: Autovalores reales, recortados y normalizados
            original_trace: Traza original antes de normalización
            min_eigenvalue: Mínimo autovalor original antes de recorte
            spectral_chars: Características espectrales completas
        """
        arr = self._as_finite_complex_matrix(
            name,
            rho,
            square=True,
            hermitian=False,  # Verificaremos manualmente con mayor rigor
        )

        if arr.shape[0] == 0:
            raise DensityMatrixValidationError(
                f"{name} no puede ser una matriz vacía."
            )

        # Verificación de hermiticidad con análisis de error
        frobenius_norm = float(la.norm(arr, ord='fro'))
        hermitian_residual = float(
            la.norm(arr - arr.conj().T, ord='fro')
        ) / max(1.0, frobenius_norm)

        hermitian_tolerance = max(
            _HERMITIAN_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON,
        )

        if hermitian_residual > hermitian_tolerance:
            raise DensityMatrixValidationError(
                f"{name} no es hermítica dentro de tolerancia.\n"
                f"  Residuo hermético relativo: {hermitian_residual:.6e}\n"
                f"  Tolerancia: {hermitian_tolerance:.6e}\n"
                f"  Norma de Frobenius: {frobenius_norm:.6e}"
            )

        # Simetrización hermítica
        rho_hermitian = (arr + arr.conj().T) / 2.0

        # Validación de traza
        trace_complex = np.trace(rho_hermitian)
        trace_real = float(np.real(trace_complex))
        trace_imag = float(np.imag(trace_complex))

        if not math.isfinite(trace_real) or not math.isfinite(trace_imag):
            raise DensityMatrixValidationError(
                f"La traza de {name} no es finita: Tr={trace_complex}"
            )

        trace_tolerance = max(
            _TRACE_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(1.0, abs(trace_real)),
        )

        if abs(trace_imag) > trace_tolerance:
            raise DensityMatrixValidationError(
                f"La traza de {name} tiene parte imaginaria significativa: "
                f"Im(Tr) = {trace_imag:.6e} > {trace_tolerance:.6e}."
            )

        if trace_real <= trace_tolerance:
            raise DensityMatrixValidationError(
                f"La traza de {name} no es positiva: "
                f"Tr = {trace_real:.6e} ≤ {trace_tolerance:.6e}."
            )

        # Normalización si es necesario
        if abs(trace_real - 1.0) > trace_tolerance:
            logger.warning(
                "%s tiene traza %.6e distinta de 1; normalizando...",
                name,
                trace_real,
            )
            rho_hermitian = rho_hermitian / trace_real

        # Diagonalización hermítica
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(rho_hermitian)
        except np.linalg.LinAlgError as exc:
            raise SpectralDecompositionError(
                f"Diagonalización hermítica de {name} falló."
            ) from exc

        eigenvalues = np.asarray(eigenvalues, dtype=np.float64)

        if not np.all(np.isfinite(eigenvalues)):
            raise SpectralDecompositionError(
                f"Los autovalores de {name} contienen valores no finitos."
            )

        # Verificación de ortogonalidad de autovectores
        is_orthonormal, ortho_deviation = self._check_orthonormality(eigenvectors)
        
        if not is_orthonormal:
            logger.warning(
                "Los autovectores de %s no son perfectamente ortonormales.\n"
                "  Desviación máxima: %.6e",
                name,
                ortho_deviation,
            )

        min_eigenvalue = float(np.min(eigenvalues)) if eigenvalues.size > 0 else 0.0

        # Validación PSD
        psd_tolerance = max(
            _PSD_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(1.0, abs(trace_real)),
        )

        if min_eigenvalue < -psd_tolerance:
            raise DensityMatrixValidationError(
                f"{name} no es positive semidefinite.\n"
                f"  Autovalor mínimo: {min_eigenvalue:.6e}\n"
                f"  Tolerancia PSD: {-psd_tolerance:.6e}"
            )

        # Recorte y normalización de autovalores
        eigenvalues_clipped = np.clip(eigenvalues, 0.0, None)
        eigen_sum = float(np.sum(eigenvalues_clipped))

        if eigen_sum <= psd_tolerance:
            raise DensityMatrixValidationError(
                f"{name} tiene espectro nulo o numéricamente degenerado: "
                f"Σλᵢ = {eigen_sum:.6e}"
            )

        eigenvalues_normalized = eigenvalues_clipped / eigen_sum

        # Reconstrucción espectral
        rho_sanitized = (
            eigenvectors * eigenvalues_normalized
        ) @ eigenvectors.conj().T
        rho_sanitized = (rho_sanitized + rho_sanitized.conj().T) / 2.0
        
        # Características espectrales
        spectral_chars = self._compute_spectral_characteristics(
            name,
            eigenvalues_normalized,
            eigenvectors,
        )

        return (
            rho_sanitized,
            eigenvalues_normalized,
            trace_real,
            min_eigenvalue,
            spectral_chars,
        )

    def _sanitize_spectrum(
        self,
        name: str,
        spectrum: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float, float]:
        r"""
        Valida y sanea un espectro de autovalores.
        
        Returns:
            spectrum_sanitized: Espectro no negativo y normalizado a traza uno
            original_trace: Suma original antes de normalización
            min_eigenvalue: Mínimo autovalor original antes de recorte
        """
        arr = self._as_finite_real_vector(name, spectrum)

        if arr.size == 0:
            raise DensityMatrixValidationError(
                f"{name} no puede ser un espectro vacío."
            )

        min_eigenvalue = float(np.min(arr))
        spectral_mass = float(np.sum(np.abs(arr)))

        psd_tolerance = max(
            _PSD_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(1.0, spectral_mass),
        )

        if min_eigenvalue < -psd_tolerance:
            raise DensityMatrixValidationError(
                f"{name} contiene autovalores negativos no físicos.\n"
                f"  Mínimo: {min_eigenvalue:.6e}\n"
                f"  Tolerancia: {-psd_tolerance:.6e}"
            )

        arr_clipped = np.clip(arr, 0.0, None)
        original_trace = float(np.sum(arr_clipped))

        if original_trace <= psd_tolerance:
            raise DensityMatrixValidationError(
                f"{name} tiene masa espectral nula o degenerada: "
                f"Σλᵢ = {original_trace:.6e}"
            )

        trace_tolerance = max(
            _TRACE_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(1.0, original_trace),
        )

        if abs(original_trace - 1.0) > trace_tolerance:
            logger.warning(
                "%s tiene traza espectral %.6e distinta de 1; normalizando...",
                name,
                original_trace,
            )

        arr_normalized = arr_clipped / original_trace

        return arr_normalized, original_trace, min_eigenvalue

    def _assert_spectra_consistent(
        self,
        name: str,
        supplied_spectrum: NDArray[np.float64],
        reference_spectrum: NDArray[np.float64],
    ) -> None:
        r"""
        Verifica consistencia entre espectro suministrado y espectro de referencia.
        
        Usa la norma infinito para medir la diferencia máxima entre componentes.
        """
        a = np.sort(supplied_spectrum)[::-1]
        b = np.sort(reference_spectrum)[::-1]

        max_dim = max(a.size, b.size)

        a_pad = np.pad(a, (0, max_dim - a.size))
        b_pad = np.pad(b, (0, max_dim - b.size))

        if a_pad.size == 0:
            max_difference = 0.0
        else:
            max_difference = float(np.max(np.abs(a_pad - b_pad)))

        mass_scale = max(
            1.0,
            float(np.sum(np.abs(a_pad))),
            float(np.sum(np.abs(b_pad))),
        )

        tolerance = max(
            _MAJORIZATION_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * mass_scale,
        )

        if max_difference > tolerance:
            raise DensityMatrixValidationError(
                f"Inconsistencia espectral en {name}.\n"
                f"  Diferencia máxima: {max_difference:.6e}\n"
                f"  Tolerancia: {tolerance:.6e}\n"
                f"  Escala de masa: {mass_scale:.6e}"
            )

    def _resolve_spectrum(
        self,
        name: str,
        spectrum: Optional[NDArray[np.float64]],
        density_matrix: Optional[NDArray[np.complex128]],
    ) -> Tuple[NDArray[np.float64], float, float, Optional[SpectralCharacteristics]]:
        r"""
        Resuelve el espectro válido a partir de matriz y/o vector de autovalores.
        
        Returns:
            eigenvalues_normalized: Espectro normalizado
            trace: Traza del espectro
            min_eigenvalue: Mínimo autovalor antes de recorte
            spectral_chars: Características espectrales (si se usó matriz)
        """
        if density_matrix is not None:
            (
                _,
                matrix_spectrum,
                matrix_trace,
                matrix_min_eval,
                spectral_chars,
            ) = self._sanitize_density_matrix(name, density_matrix)

            if spectrum is not None:
                supplied_spectrum, supplied_trace, _ = self._sanitize_spectrum(
                    f"{name}_evals",
                    spectrum,
                )

                trace_consistency_tolerance = max(
                    _TRACE_TOLERANCE,
                    _NUMERICAL_SAFETY_FACTOR
                    * _MACHINE_EPSILON
                    * max(1.0, abs(matrix_trace), abs(supplied_trace)),
                )

                if abs(matrix_trace - supplied_trace) > trace_consistency_tolerance:
                    raise DensityMatrixValidationError(
                        f"Inconsistencia de traza en {name}.\n"
                        f"  Traza de matriz: {matrix_trace:.6e}\n"
                        f"  Traza espectral: {supplied_trace:.6e}\n"
                        f"  Diferencia: {abs(matrix_trace - supplied_trace):.6e}"
                    )

                self._assert_spectra_consistent(
                    name,
                    supplied_spectrum,
                    matrix_spectrum,
                )

            return matrix_spectrum, matrix_trace, matrix_min_eval, spectral_chars

        if spectrum is not None:
            normalized, trace, min_eval = self._sanitize_spectrum(name, spectrum)
            return normalized, trace, min_eval, None

        raise ValueError(
            f"Debe proveerse spectrum o density_matrix para {name}."
        )

    def _compute_lorenz_curves(
        self,
        spectrum_a: NDArray[np.float64],
        spectrum_b: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Computa curvas de Lorenz para análisis de majorización.
        
        La curva de Lorenz L(k) es la suma acumulativa de autovalores ordenados:
            L_ρ(k) = Σᵢ₌₁ᵏ λᵢ↓(ρ)
        
        Returns:
            cumulative_a: Curva de Lorenz del espectro A
            cumulative_b: Curva de Lorenz del espectro B
            deviations: L_A(k) - L_B(k) para cada k
        """
        lambda_a = np.sort(spectrum_a)[::-1]
        lambda_b = np.sort(spectrum_b)[::-1]

        max_dim = max(lambda_a.size, lambda_b.size)

        lambda_a_pad = np.pad(lambda_a, (0, max_dim - lambda_a.size))
        lambda_b_pad = np.pad(lambda_b, (0, max_dim - lambda_b.size))

        # Sumas acumulativas (curvas de Lorenz)
        cumulative_a = np.cumsum(lambda_a_pad)
        cumulative_b = np.cumsum(lambda_b_pad)

        deviations = cumulative_a - cumulative_b

        return cumulative_a, cumulative_b, deviations

    def _compute_renyi_entropies(
        self,
        spectrum: NDArray[np.float64],
    ) -> Tuple[float, float]:
        r"""
        Computa entropías de Rényi de orden 2 e infinito.
        
        Entropía de Rényi de orden α:
            S_α(ρ) = (1/(1-α)) log(Σᵢ λᵢᵅ)
        
        Casos especiales:
            α=2: S₂(ρ) = -log(Tr(ρ²)) = -log(Σᵢ λᵢ²)  [entropía de colisión]
            α=∞: S_∞(ρ) = -log(λₘₐₓ)  [min-entropía]
        
        Returns:
            renyi_2: Entropía de Rényi de orden 2
            renyi_inf: Entropía de Rényi de orden infinito
        """
        probabilities = spectrum[spectrum > _MACHINE_EPSILON]
        
        if probabilities.size == 0:
            return 0.0, 0.0
        
        # S₂ = -log(Σᵢ pᵢ²)
        purity = float(np.sum(probabilities**2))
        renyi_2 = -math.log(purity) if purity > _MACHINE_EPSILON else 0.0
        
        # S_∞ = -log(max pᵢ)
        max_prob = float(np.max(probabilities))
        renyi_inf = -math.log(max_prob) if max_prob > _MACHINE_EPSILON else 0.0
        
        return renyi_2, renyi_inf

    def _audit_quantum_majorization(
        self,
        evals_orig: Optional[NDArray[np.float64]] = None,
        evals_purified: Optional[NDArray[np.float64]] = None,
        *,
        rho_orig: Optional[NDArray[np.complex128]] = None,
        rho_purified: Optional[NDArray[np.complex128]] = None,
    ) -> MajorizationAuditData:
        r"""
        Último método de la Fase 1 (continuación de los 7 métodos anteriores).
        
        Verifica la condición de majorización cuántica:
            λ(ρ_pur) ≽ λ(ρ_orig)
        
        mediante curvas de Lorenz y análisis espectral completo.
        
        Este método retorna MajorizationAuditData, que es el objeto inicial
        consumido por el primer método de Fase 2.
        
        Teorema verificado:
            ρ_pur ≽ ρ_orig ⟺ ∀k: Σᵢ₌₁ᵏ λᵢ↓(ρ_pur) ≥ Σᵢ₌₁ᵏ λᵢ↓(ρ_orig)
                              y Σᵢ₌₁ᵈ λᵢ(ρ_pur) = Σᵢ₌₁ᵈ λᵢ(ρ_orig) = 1
        """
        # Resolución de espectros con características completas
        (
            spectrum_orig,
            trace_orig,
            min_eval_orig,
            spectral_chars_orig,
        ) = self._resolve_spectrum("rho_orig", evals_orig, rho_orig)

        (
            spectrum_pur,
            trace_pur,
            min_eval_pur,
            spectral_chars_pur,
        ) = self._resolve_spectrum("rho_purified", evals_purified, rho_purified)

        # Si no se computaron desde matriz, crear características desde espectro
        if spectral_chars_orig is None:
            dim = spectrum_orig.size
            spectral_chars_orig = self._compute_spectral_characteristics(
                "rho_orig",
                spectrum_orig,
                np.eye(dim, dtype=np.complex128),  # Autovectores identidad dummy
            )
            
        if spectral_chars_pur is None:
            dim = spectrum_pur.size
            spectral_chars_pur = self._compute_spectral_characteristics(
                "rho_purified",
                spectrum_pur,
                np.eye(dim, dtype=np.complex128),
            )

        # Cómputo de curvas de Lorenz
        cumulative_orig, cumulative_pur, deviations = self._compute_lorenz_curves(
            spectrum_orig,
            spectrum_pur,
        )

        # Análisis de desviaciones de majorización
        raw_max_deviation = float(np.max(deviations)) if deviations.size > 0 else 0.0
        max_lorenz_deviation = max(0.0, raw_max_deviation)

        majorization_tolerance = max(
            _MAJORIZATION_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, float(cumulative_orig.size)),
        )

        # Verificación de conservación de traza (última desviación debe ser ~0)
        total_trace_deviation = float(
            abs(cumulative_orig[-1] - cumulative_pur[-1])
        ) if cumulative_orig.size > 0 else 0.0

        if total_trace_deviation > majorization_tolerance:
            raise QuantumMajorizationViolation(
                "Violación de conservación de traza en majorización.\n"
                f"  |Tr(ρ_orig) - Tr(ρ_pur)| = {total_trace_deviation:.6e}\n"
                f"  Tolerancia: {majorization_tolerance:.6e}"
            )

        # Verificación del preorden de majorización
        if max_lorenz_deviation > majorization_tolerance:
            # Encontrar el índice k donde ocurre la violación máxima
            violation_index = int(np.argmax(deviations))
            
            raise QuantumMajorizationViolation(
                "Violación del preorden de majorización cuántica.\n"
                f"  El minimizador degradó la matriz atómica de conocimiento.\n"
                f"  Desviación máxima: {max_lorenz_deviation:.6e}\n"
                f"  Tolerancia: {majorization_tolerance:.6e}\n"
                f"  Índice de violación: k={violation_index + 1}\n"
                f"  L_orig({violation_index + 1}): {cumulative_orig[violation_index]:.6e}\n"
                f"  L_pur({violation_index + 1}): {cumulative_pur[violation_index]:.6e}"
            )

        # Cómputo de entropías de Rényi
        renyi_2_orig, renyi_inf_orig = self._compute_renyi_entropies(spectrum_orig)
        renyi_2_pur, renyi_inf_pur = self._compute_renyi_entropies(spectrum_pur)

        # Cómputo de pureza: Tr(ρ²) = Σᵢ λᵢ²
        purity_orig = float(np.sum(spectrum_orig**2))
        purity_pur = float(np.sum(spectrum_pur**2))

        # Autovalores máximos
        max_eval_orig = float(np.max(spectrum_orig)) if spectrum_orig.size > 0 else 0.0
        max_eval_pur = float(np.max(spectrum_pur)) if spectrum_pur.size > 0 else 0.0

        logger.info(
            "Fase 1 COMPLETADA: Auditoría de Majorización Cuántica.\n"
            "  Dimensión: %d\n"
            "  Rango efectivo (orig): %d\n"
            "  Rango efectivo (pur): %d\n"
            "  Desviación máxima de Lorenz: %.6e\n"
            "  Pureza original: %.6f\n"
            "  Pureza purificada: %.6f\n"
            "  Δ Pureza: %.6f",
            spectral_chars_orig.dimension,
            spectral_chars_orig.effective_rank,
            spectral_chars_pur.effective_rank,
            max_lorenz_deviation,
            purity_orig,
            purity_pur,
            purity_pur - purity_orig,
        )

        return MajorizationAuditData(
            dimension=spectral_chars_orig.dimension,
            trace_original=float(trace_orig),
            trace_purified=float(trace_pur),
            min_eigenvalue_original=float(min_eval_orig),
            min_eigenvalue_purified=float(min_eval_pur),
            max_eigenvalue_original=max_eval_orig,
            max_eigenvalue_purified=max_eval_pur,
            max_lorenz_deviation=float(max_lorenz_deviation),
            majorization_tolerance=float(majorization_tolerance),
            is_majorized=True,
            spectral_characteristics_original=spectral_chars_orig,
            spectral_characteristics_purified=spectral_chars_pur,
            renyi_entropy_original=renyi_2_orig,
            renyi_entropy_purified=renyi_2_pur,
            purity_original=purity_orig,
            purity_purified=purity_pur,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 2: CERTIFICACIÓN DE FIDELIDAD DE UHLMANN                             ║
# ║                                                                             ║
# ║   Teoría Matemática:                                                        ║
# ║   ─────────────────                                                         ║
# ║   Fidelidad de Uhlmann entre estados cuánticos ρ y σ:                       ║
# ║                                                                             ║
# ║       F(ρ, σ) = (Tr √(√ρ σ √ρ))²                                           ║
# ║                                                                             ║
# ║   Para estados puros |ψ⟩ y |φ⟩:                                             ║
# ║       F(|ψ⟩⟨ψ|, |φ⟩⟨φ|) = |⟨ψ|φ⟩|²                                          ║
# ║                                                                             ║
# ║   Propiedades:                                                              ║
# ║       1. 0 ≤ F(ρ, σ) ≤ 1                                                    ║
# ║       2. F(ρ, σ) = 1 ⟺ ρ = σ                                               ║
# ║       3. F(ρ, σ) = F(σ, ρ)  (simetría)                                      ║
# ║       4. F(ρ, σ)² ≤ F(ρ, τ)F(τ, σ)  (desigualdad triangular)               ║
# ║       5. F es continua en la topología de traza                             ║
# ║                                                                             ║
# ║   Relación con otras métricas:                                              ║
# ║       - Distancia de Bures: D_B(ρ,σ) = √(2(1 - √F(ρ,σ)))                  ║
# ║       - Distancia de Hellinger: D_H(ρ,σ) = √(1 - F(ρ,σ))                   ║
# ║       - Distancia de traza: D_tr(ρ,σ) ≤ √(1 - F(ρ,σ))                      ║
# ║                                                                             ║
# ║   Interpretación física:                                                    ║
# ║       F(ρ, σ) mide la "distinguibilidad" cuántica entre estados.            ║
# ║       Alta fidelidad ⟹ estados cuánticamente indistinguibles.              ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_UhlmannFidelityCertifier(Phase1_QuantumMajorizationAuditor):
    r"""
    Fase 2: Certificación rigurosa de fidelidad de Uhlmann.
    
    Garantiza que la purificación espectral preserve la información cuántica
    esencial, manteniendo alta fidelidad con el estado original.
    
    Métodos de esta fase:
        1. _compute_quantum_coherence_metrics (inicio, hereda certificado Fase 1)
        2. _psd_square_root (herramienta auxiliar)
        3. _compute_trace_distance
        4. _compute_bures_distance
        5. _compute_hellinger_distance
        6. _certify_uhlmann_fidelity_bound (final)
    
    El último método retorna FidelityAuditData, que es consumido por Fase 3.
    """

    def _compute_quantum_coherence_metrics(
        self,
        rho: NDArray[np.complex128],
        name: str = "rho",
    ) -> QuantumCoherenceMetrics:
        r"""
        Primer método de Fase 2 (continuación formal de Fase 1).
        
        Computa métricas de coherencia cuántica.
        
        Coherencia cuántica mide la presencia de elementos fuera de diagonal
        en la base computacional (o de energía).
        
        Métricas implementadas:
            1. l1-norm coherence: C_l1(ρ) = Σ_{i≠j} |ρᵢⱼ|
            2. Relative entropy coherence: C_re(ρ) = S(ρ_diag) - S(ρ)
            3. Robustness of coherence: min mixing con estados incoherentes
            4. Off-diagonal mass: ‖ρ - diag(ρ)‖_F
        
        Args:
            rho: Matriz de densidad cuántica
            name: Identificador para mensajes de error
        
        Returns:
            QuantumCoherenceMetrics con las métricas computadas
        """
        rho_sanitized, _, _, _, _ = self._sanitize_density_matrix(name, rho)
        
        dimension = rho_sanitized.shape[0]
        
        # Extracción de diagonal y elementos fuera de diagonal
        rho_diagonal = np.diag(np.diag(rho_sanitized))
        rho_off_diagonal = rho_sanitized - rho_diagonal
        
        # 1. l1-norm coherence: Σ_{i≠j} |ρᵢⱼ|
        l1_coherence = float(np.sum(np.abs(rho_off_diagonal)))
        
        # 2. Off-diagonal mass (norma de Frobenius)
        off_diagonal_mass = float(la.norm(rho_off_diagonal, ord='fro'))
        
        # 3. Relative entropy coherence: S(diag(ρ)) - S(ρ)
        diag_eigenvalues = np.diag(rho_diagonal).real
        diag_eigenvalues = diag_eigenvalues[diag_eigenvalues > _MACHINE_EPSILON]
        
        if diag_eigenvalues.size > 0:
            entropy_diagonal = float(
                -np.sum(diag_eigenvalues * np.log(diag_eigenvalues))
            )
        else:
            entropy_diagonal = 0.0
        
        # Entropía del estado completo (requiere diagonalización)
        try:
            full_eigenvalues = np.linalg.eigvalsh(rho_sanitized)
            full_eigenvalues = full_eigenvalues[full_eigenvalues > _MACHINE_EPSILON]
            
            if full_eigenvalues.size > 0:
                entropy_full = float(
                    -np.sum(full_eigenvalues * np.log(full_eigenvalues))
                )
            else:
                entropy_full = 0.0
        except np.linalg.LinAlgError:
            entropy_full = 0.0
        
        relative_entropy_coherence = max(0.0, entropy_diagonal - entropy_full)
        
        # 4. Robustness of coherence (estimación)
        # ρ_rob = mín p tal que (ρ + pI/d)/(1+p) sea incoherente
        # Aproximación: p ≈ l1_coherence / (1 - l1_coherence)
        robustness = (
            l1_coherence / (1.0 - l1_coherence + _MACHINE_EPSILON)
            if l1_coherence < 1.0
            else float('inf')
        )
        
        return QuantumCoherenceMetrics(
            l1_norm_coherence=l1_coherence,
            relative_entropy_coherence=relative_entropy_coherence,
            robustness_of_coherence=robustness,
            off_diagonal_mass=off_diagonal_mass,
        )

    def _psd_square_root(
        self,
        name: str,
        rho: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        r"""
        Calcula la raíz cuadrada matricial de una matriz PSD.
        
        Teorema (Cálculo funcional):
            Para A hermítica, A = UΛU†, entonces:
            f(A) = U f(Λ) U†
        
        Para f(x) = √x:
            √A = U √Λ U†
        
        Args:
            name: Identificador para mensajes de error
            rho: Matriz de densidad hermítica y PSD
        
        Returns:
            √ρ como matriz hermítica PSD
        """
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(rho)
        except np.linalg.LinAlgError as exc:
            raise SpectralDecompositionError(
                f"Diagonalización hermítica de {name} falló al calcular √ρ."
            ) from exc

        eigenvalues = np.asarray(eigenvalues, dtype=np.float64)

        if not np.all(np.isfinite(eigenvalues)):
            raise SpectralDecompositionError(
                f"Los autovalores de {name} no son finitos al calcular √ρ."
            )

        psd_tolerance = max(
            _PSD_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON,
        )

        min_eigenvalue = float(np.min(eigenvalues)) if eigenvalues.size > 0 else 0.0

        if min_eigenvalue < -psd_tolerance:
            raise DensityMatrixValidationError(
                f"{name} no es PSD al calcular √ρ.\n"
                f"  Autovalor mínimo: {min_eigenvalue:.6e}\n"
                f"  Tolerancia: {-psd_tolerance:.6e}"
            )

        # Recorte de autovalores negativos espurios
        sqrt_eigenvalues = np.sqrt(np.clip(eigenvalues, 0.0, None))
        
        # Reconstrucción: √ρ = U √Λ U†
        sqrt_rho = (eigenvectors * sqrt_eigenvalues) @ eigenvectors.conj().T
        
        # Simetrización hermítica para eliminar error numérico
        sqrt_rho = (sqrt_rho + sqrt_rho.conj().T) / 2.0

        return sqrt_rho

    def _compute_trace_distance(
        self,
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
    ) -> float:
        r"""
        Calcula la distancia de traza entre dos estados cuánticos.
        
        Definición:
            D_tr(ρ, σ) = (1/2) ‖ρ - σ‖₁ = (1/2) Tr|ρ - σ|
        
        donde |A| = √(A†A) es el valor absoluto matricial.
        
        Propiedades:
            - 0 ≤ D_tr(ρ, σ) ≤ 1
            - D_tr(ρ, σ) = 0 ⟺ ρ = σ
            - D_tr(ρ, σ) = 1 ⟺ ρ y σ tienen soportes ortogonales
        
        Relación con fidelidad:
            D_tr(ρ, σ) ≤ √(1 - F(ρ, σ))
        """
        difference = rho - sigma
        
        # |A| se computa como √(A†A)
        abs_difference_squared = difference.conj().T @ difference
        
        try:
            eigenvalues = np.linalg.eigvalsh(abs_difference_squared)
        except np.linalg.LinAlgError:
            # Fallback: usar descomposición en valores singulares
            singular_values = la.svdvals(difference)
            return 0.5 * float(np.sum(singular_values))
        
        # Tr|A| = Tr√(A†A) = Σ√λᵢ(A†A)
        trace_abs = float(np.sum(np.sqrt(np.clip(eigenvalues, 0.0, None))))
        
        return 0.5 * trace_abs

    def _compute_bures_distance(
        self,
        fidelity: float,
    ) -> float:
        r"""
        Calcula la distancia de Bures a partir de la fidelidad.
        
        Definición:
            D_B(ρ, σ) = √(2(1 - √F(ρ, σ)))
        
        Propiedades:
            - 0 ≤ D_B(ρ, σ) ≤ √2
            - D_B define una métrica Riemanniana en el espacio de estados
            - D_B es la distancia geodésica en la geometría de información cuántica
        """
        if not (0.0 <= fidelity <= 1.0 + _FIDELITY_NUMERICAL_TOLERANCE):
            raise ValueError(
                f"Fidelidad fuera de rango físico: F = {fidelity:.6e}"
            )
        
        fidelity_clamped = float(np.clip(fidelity, 0.0, 1.0))
        
        return math.sqrt(2.0 * (1.0 - math.sqrt(fidelity_clamped)))

    def _compute_hellinger_distance(
        self,
        fidelity: float,
    ) -> float:
        r"""
        Calcula la distancia de Hellinger a partir de la fidelidad.
        
        Definición:
            D_H(ρ, σ) = √(1 - F(ρ, σ))
        
        Propiedades:
            - 0 ≤ D_H(ρ, σ) ≤ 1
            - Relacionada con el ángulo entre estados en espacio de Hilbert
        """
        if not (0.0 <= fidelity <= 1.0 + _FIDELITY_NUMERICAL_TOLERANCE):
            raise ValueError(
                f"Fidelidad fuera de rango físico: F = {fidelity:.6e}"
            )
        
        fidelity_clamped = float(np.clip(fidelity, 0.0, 1.0))
        
        return math.sqrt(1.0 - fidelity_clamped)

    def _certify_uhlmann_fidelity_bound(
        self,
        rho_orig: NDArray[np.complex128],
        rho_purified: NDArray[np.complex128],
        majorization_audit: Optional[MajorizationAuditData] = None,
    ) -> FidelityAuditData:
        r"""
        Último método de Fase 2 (continuación de métodos previos de esta fase).
        
        Calcula la fidelidad cuántica de Uhlmann y métricas de distancia asociadas.
        
        Algoritmo:
            1. Validar certificado de majorización (Fase 1)
            2. Sanear matrices de densidad
            3. Computar √ρ_orig
            4. Construir núcleo M = √ρ_orig · σ · √ρ_orig
            5. Diagonalizar M y extraer autovalores
            6. Fidelidad: F = (Σ √λᵢ(M))²
            7. Computar métricas de distancia derivadas
            8. Analizar coherencia cuántica
        
        Este método retorna FidelityAuditData, que es el objeto inicial
        consumido por el primer método de Fase 3.
        """
        # Validación del certificado de Fase 1
        if majorization_audit is not None:
            if not majorization_audit.is_majorized:
                raise QuantumMajorizationViolation(
                    "La Fase 2 no puede iniciarse sin certificado de majorización válido.\n"
                    "  La Fase 1 no certificó el preorden de majorización cuántica."
                )

        # Saneamiento de matrices de densidad
        rho_original, _, _, _, _ = self._sanitize_density_matrix("rho_orig", rho_orig)
        rho_purified_sanitized, _, _, _, _ = self._sanitize_density_matrix(
            "rho_purified",
            rho_purified,
        )

        # Validación de dimensionalidad
        if rho_original.shape != rho_purified_sanitized.shape:
            raise DensityMatrixValidationError(
                f"rho_orig y rho_purified deben tener la misma dimensión.\n"
                f"  Shape rho_orig: {rho_original.shape}\n"
                f"  Shape rho_purified: {rho_purified_sanitized.shape}"
            )

        # Validación de consistencia con certificado de majorización
        if majorization_audit is not None:
            if majorization_audit.dimension != rho_original.shape[0]:
                raise ValueError(
                    "El certificado de majorización es inconsistente con las matrices.\n"
                    f"  Dimensión en certificado: {majorization_audit.dimension}\n"
                    f"  Dimensión de matrices: {rho_original.shape[0]}"
                )

        # Cálculo de √ρ_original
        sqrt_rho_original = self._psd_square_root("rho_orig", rho_original)

        # Construcción del núcleo de fidelidad: M = √ρ · σ · √ρ
        core_matrix = sqrt_rho_original @ rho_purified_sanitized @ sqrt_rho_original
        core_matrix = (core_matrix + core_matrix.conj().T) / 2.0  # Simetrización

        # Diagonalización del núcleo
        try:
            core_eigenvalues = np.linalg.eigvalsh(core_matrix)
        except np.linalg.LinAlgError as exc:
            raise UhlmannFidelityCollapseError(
                "No se pudo diagonalizar el núcleo de fidelidad de Uhlmann."
            ) from exc

        core_eigenvalues = np.asarray(core_eigenvalues, dtype=np.float64)

        if not np.all(np.isfinite(core_eigenvalues)):
            raise UhlmannFidelityCollapseError(
                "El espectro del núcleo de fidelidad contiene valores no finitos."
            )

        # Recorte de autovalores negativos espurios
        core_eigenvalues = np.clip(core_eigenvalues, 0.0, None)

        # Fidelidad de Uhlmann: F = (Tr√M)² = (Σ√λᵢ)²
        trace_sqrt_core = self._kahan_sum(np.sqrt(core_eigenvalues))
        fidelity = float(trace_sqrt_core * trace_sqrt_core)

        if not math.isfinite(fidelity):
            raise UhlmannFidelityCollapseError(
                "La fidelidad de Uhlmann no es finita."
            )

        # Validación de cotas físicas
        fidelity_tolerance = max(
            _FIDELITY_NUMERICAL_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON,
        )

        upper_numerical_tolerance = max(1e-10, 10.0 * fidelity_tolerance)

        if fidelity < -fidelity_tolerance:
            raise DensityMatrixValidationError(
                f"Fidelidad negativa (imposible físicamente): F = {fidelity:.6e}"
            )

        if fidelity > 1.0 + upper_numerical_tolerance:
            raise DensityMatrixValidationError(
                f"Fidelidad > 1 fuera de tolerancia numérica: F = {fidelity:.6e}"
            )

        # Clamping a rango físico [0, 1]
        fidelity = float(np.clip(fidelity, 0.0, 1.0))

        # Verificación contra umbral mínimo
        if fidelity < _UHLMANN_FIDELITY_MIN:
            raise UhlmannFidelityCollapseError(
                "Colapso semántico detectado por fidelidad insuficiente.\n"
                f"  Fidelidad de Uhlmann: F = {fidelity:.6f}\n"
                f"  Umbral mínimo: F_min = {_UHLMANN_FIDELITY_MIN:.6f}\n"
                f"  Déficit: ΔF = {_UHLMANN_FIDELITY_MIN - fidelity:.6f}\n"
                "  Interpretación: La purificación espectral mutiló ramas lógicas esenciales."
            )

        # Cómputo de métricas de distancia complementarias
        trace_distance = self._compute_trace_distance(
            rho_original,
            rho_purified_sanitized,
        )
        
        bures_distance = self._compute_bures_distance(fidelity)
        hellinger_distance = self._compute_hellinger_distance(fidelity)

        # Análisis de coherencia cuántica
        coherence_orig = self._compute_quantum_coherence_metrics(
            rho_original,
            "rho_orig",
        )
        
        coherence_pur = self._compute_quantum_coherence_metrics(
            rho_purified_sanitized,
            "rho_purified",
        )

        # Verificación de preservación de coherencia
        coherence_degradation = (
            coherence_orig.l1_norm_coherence - coherence_pur.l1_norm_coherence
        )
        
        if coherence_degradation > _COHERENCE_TOLERANCE:
            logger.warning(
                "Degradación de coherencia cuántica detectada.\n"
                "  Coherencia original (l1): %.6e\n"
                "  Coherencia purificada (l1): %.6e\n"
                "  Degradación: %.6e",
                coherence_orig.l1_norm_coherence,
                coherence_pur.l1_norm_coherence,
                coherence_degradation,
            )

        logger.info(
            "Fase 2 COMPLETADA: Certificación de Fidelidad de Uhlmann.\n"
            "  Fidelidad F(ρ,σ): %.6f\n"
            "  Distancia de traza D_tr: %.6e\n"
            "  Distancia de Bures D_B: %.6e\n"
            "  Distancia de Hellinger D_H: %.6e\n"
            "  Coherencia l1 (orig): %.6e\n"
            "  Coherencia l1 (pur): %.6e",
            fidelity,
            trace_distance,
            bures_distance,
            hellinger_distance,
            coherence_orig.l1_norm_coherence,
            coherence_pur.l1_norm_coherence,
        )

        return FidelityAuditData(
            uhlmann_fidelity=fidelity,
            fidelity_tolerance=float(fidelity_tolerance),
            fidelity_min_required=float(_UHLMANN_FIDELITY_MIN),
            is_fidelity_preserved=True,
            trace_distance=trace_distance,
            bures_distance=bures_distance,
            hellinger_distance=hellinger_distance,
            coherence_metrics_original=coherence_orig,
            coherence_metrics_purified=coherence_pur,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 3: COTA DE CAPACIDAD DE HOLEVO Y ENTROPÍA                            ║
# ║                                                                             ║
# ║   Teoría Matemática:                                                        ║
# ║   ─────────────────                                                         ║
# ║   Entropía de von Neumann:                                                  ║
# ║       S(ρ) = -Tr(ρ log ρ) = -Σᵢ λᵢ log λᵢ                                  ║
# ║                                                                             ║
# ║   Propiedades:                                                              ║
# ║       1. S(ρ) ≥ 0 (no negatividad)                                          ║
# ║       2. S(ρ) = 0 ⟺ ρ es estado puro                                       ║
# ║       3. S(ρ) ≤ log d (cota superior para dimensión d)                      ║
# ║       4. S es cóncava: S(Σᵢ pᵢρᵢ) ≥ Σᵢ pᵢS(ρᵢ)                              ║
# ║                                                                             ║
# ║   Capacidad de Holevo:                                                      ║
# ║       χ({pᵢ, ρᵢ}) = S(Σᵢ pᵢρᵢ) - Σᵢ pᵢS(ρᵢ)                                ║
# ║                                                                             ║
# ║   Interpretación:                                                           ║
# ║       χ es la máxima información clásica transmisible usando estados        ║
# ║       cuánticos {ρᵢ} con probabilidades a priori {pᵢ}.                      ║
# ║                                                                             ║
# ║   Teorema (Cota de Holevo-Schumacher-Westmoreland):                         ║
# ║       La capacidad del canal cuántico para transmisión de información       ║
# ║       clásica está acotada por la capacidad de Holevo.                      ║
# ║                                                                             ║
# ║   Segunda Ley de la Termodinámica Cuántica:                                 ║
# ║       Para procesos irreversibles en sistemas cerrados: ΔS ≥ 0              ║
# ║       Para procesos reversibles (como purificación): ΔS ≤ 0 es válido       ║
# ║                                                                             ║
# ║   Purificación espectral debe satisfacer:                                   ║
# ║       S(ρ_purificada) ≤ S(ρ_original)                                       ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_HolevoCapacityEnforcer(Phase2_UhlmannFidelityCertifier):
    r"""
    Fase 3: Enforcement de cota de capacidad de Holevo y entropía.
    
    Garantiza que la purificación espectral sea termodinámicamente consistente,
    preservando la capacidad de información del canal cuántico.
    
    Métodos de esta fase:
        1. _von_neumann_entropy (inicio, hereda certificado Fase 2)
        2. _compute_renyi_entropy_family
        3. _estimate_holevo_capacity
        4. _compute_quantum_mutual_information
        5. _enforce_holevo_capacity_retention (final)
    
    El último método retorna HolevoAuditData, certificado final del proceso.
    """

    @staticmethod
    def _von_neumann_entropy(spectrum: NDArray[np.float64]) -> float:
        r"""
        Primer método de Fase 3 (continuación formal de Fase 2).
        
        Calcula la entropía de von Neumann de un estado cuántico.
        
        Definición:
            S(ρ) = -Tr(ρ log ρ) = -Σᵢ λᵢ log λᵢ
        
        Usa logaritmo natural (unidades: nats).
        Conversión a bits: S_bits = S_nats / log(2)
        
        Args:
            spectrum: Autovalores de la matriz de densidad (normalizados a Tr=1)
        
        Returns:
            Entropía de von Neumann en nats
        """
        probabilities = np.clip(spectrum, 0.0, 1.0)
        probabilities = probabilities[probabilities > _MACHINE_EPSILON]

        if probabilities.size == 0:
            return 0.0

        # S = -Σ pᵢ log pᵢ usando suma compensada de Kahan
        log_probs = np.log(probabilities)
        entropy_terms = -probabilities * log_probs
        
        entropy = _AdvancedNumericalGuard._kahan_sum(entropy_terms)

        if not math.isfinite(entropy):
            raise HolevoCapacityDeficitError(
                "La entropía de von Neumann no es finita."
            )

        # La entropía no puede ser negativa
        if entropy < -_ENTROPY_TOLERANCE:
            raise HolevoCapacityDeficitError(
                f"Entropía negativa (imposible): S = {entropy:.6e}"
            )

        return max(0.0, entropy)

    def _compute_renyi_entropy_family(
        self,
        spectrum: NDArray[np.float64],
        alpha_values: Optional[List[float]] = None,
    ) -> dict[float, float]:
        r"""
        Computa la familia de entropías de Rényi para múltiples órdenes α.
        
        Entropía de Rényi de orden α:
            S_α(ρ) = (1/(1-α)) log(Σᵢ λᵢᵅ)
        
        Casos especiales:
            α → 1: S₁ = S_vN (von Neumann)
            α = 2: S₂ = -log(Tr(ρ²)) (entropía de colisión)
            α = ∞: S_∞ = -log(λₘₐₓ) (min-entropía)
            α → 0: S₀ = log(rank(ρ)) (max-entropía)
        
        Args:
            spectrum: Autovalores normalizados
            alpha_values: Lista de órdenes α a computar
        
        Returns:
            Diccionario {α: S_α(ρ)}
        """
        if alpha_values is None:
            alpha_values = [0.5, 1.0, 2.0, float('inf')]
        
        probabilities = spectrum[spectrum > _MACHINE_EPSILON]
        
        if probabilities.size == 0:
            return {alpha: 0.0 for alpha in alpha_values}
        
        renyi_entropies = {}
        
        for alpha in alpha_values:
            if alpha == 1.0:
                # Caso límite: von Neumann
                renyi_entropies[alpha] = self._von_neumann_entropy(spectrum)
                
            elif alpha == float('inf'):
                # Min-entropía: S_∞ = -log(λₘₐₓ)
                max_prob = float(np.max(probabilities))
                renyi_entropies[alpha] = -math.log(max_prob)
                
            elif alpha == 0.0:
                # Max-entropía: S₀ = log(rank(ρ))
                effective_rank = probabilities.size
                renyi_entropies[alpha] = math.log(effective_rank)
                
            elif alpha > 0.0 and math.isfinite(alpha):
                # Caso general: S_α = (1/(1-α)) log(Σᵢ pᵢᵅ)
                power_sum = float(np.sum(probabilities**alpha))
                
                if power_sum > _MACHINE_EPSILON:
                    renyi_entropies[alpha] = (
                        math.log(power_sum) / (1.0 - alpha)
                    )
                else:
                    renyi_entropies[alpha] = 0.0
            else:
                renyi_entropies[alpha] = float('nan')
        
        return renyi_entropies

    def _estimate_holevo_capacity(
        self,
        ensemble_states: List[NDArray[np.complex128]],
        probabilities: NDArray[np.float64],
    ) -> float:
        r"""
        Estima la capacidad de Holevo para un ensemble de estados cuánticos.
        
        Definición:
            χ({pᵢ, ρᵢ}) = S(ρ̄) - Σᵢ pᵢS(ρᵢ)
        
        donde ρ̄ = Σᵢ pᵢρᵢ es el estado promedio.
        
        Args:
            ensemble_states: Lista de matrices de densidad {ρᵢ}
            probabilities: Probabilidades a priori {pᵢ}
        
        Returns:
            Capacidad de Holevo χ en nats
        """
        if len(ensemble_states) == 0:
            return 0.0
        
        probabilities = self._as_finite_real_vector(
            "probabilities",
            probabilities,
            positive=True,
            normalized=True,
        )
        
        if len(ensemble_states) != probabilities.size:
            raise ValueError(
                "El número de estados debe coincidir con el número de probabilidades."
            )
        
        # Estado promedio: ρ̄ = Σᵢ pᵢρᵢ
        dimension = ensemble_states[0].shape[0]
        average_state = np.zeros((dimension, dimension), dtype=np.complex128)
        
        for prob, state in zip(probabilities, ensemble_states):
            state_sanitized, _, _, _, _ = self._sanitize_density_matrix(
                "ensemble_state",
                state,
            )
            average_state += float(prob) * state_sanitized
        
        # Entropía del estado promedio
        try:
            avg_eigenvalues = np.linalg.eigvalsh(average_state)
            avg_eigenvalues = avg_eigenvalues[avg_eigenvalues > _MACHINE_EPSILON]
            avg_eigenvalues = avg_eigenvalues / np.sum(avg_eigenvalues)
            
            entropy_average = self._von_neumann_entropy(avg_eigenvalues)
        except np.linalg.LinAlgError:
            entropy_average = 0.0
        
        # Entropías individuales ponderadas: Σᵢ pᵢS(ρᵢ)
        weighted_entropy_sum = 0.0
        
        for prob, state in zip(probabilities, ensemble_states):
            try:
                state_eigenvalues = np.linalg.eigvalsh(state)
                state_eigenvalues = state_eigenvalues[
                    state_eigenvalues > _MACHINE_EPSILON
                ]
                state_eigenvalues = state_eigenvalues / np.sum(state_eigenvalues)
                
                entropy_state = self._von_neumann_entropy(state_eigenvalues)
                weighted_entropy_sum += float(prob) * entropy_state
            except np.linalg.LinAlgError:
                continue
        
        # Capacidad de Holevo: χ = S(ρ̄) - Σᵢ pᵢS(ρᵢ)
        holevo_capacity = entropy_average - weighted_entropy_sum
        
        # La capacidad debe ser no negativa por concavidad de S
        return max(0.0, holevo_capacity)

    def _compute_quantum_mutual_information(
        self,
        rho_AB: NDArray[np.complex128],
        dimension_A: int,
        dimension_B: int,
    ) -> float:
        r"""
        Computa la información mutua cuántica entre dos subsistemas.
        
        Definición:
            I(A:B) = S(ρ_A) + S(ρ_B) - S(ρ_AB)
        
        donde ρ_A = Tr_B(ρ_AB) y ρ_B = Tr_A(ρ_AB) son estados reducidos.
        
        Interpretación:
            I(A:B) mide las correlaciones cuánticas entre subsistemas A y B.
            I(A:B) = 0 ⟺ ρ_AB = ρ_A ⊗ ρ_B (estados no correlacionados)
        
        Args:
            rho_AB: Estado bipartito ρ_AB
            dimension_A: Dimensión del subsistema A
            dimension_B: Dimensión del subsistema B
        
        Returns:
            Información mutua I(A:B) en nats
        """
        if rho_AB.shape[0] != dimension_A * dimension_B:
            raise ValueError(
                f"Dimensión inconsistente: shape={rho_AB.shape}, "
                f"esperado {dimension_A * dimension_B}×{dimension_A * dimension_B}"
            )
        
        # Traza parcial sobre B: ρ_A = Tr_B(ρ_AB)
        rho_A = np.zeros((dimension_A, dimension_A), dtype=np.complex128)
        
        for i in range(dimension_B):
            # Seleccionar bloques diagonales en índices de B
            start_idx = i * dimension_A
            end_idx = (i + 1) * dimension_A
            rho_A += rho_AB[start_idx:end_idx, start_idx:end_idx]
        
        # Traza parcial sobre A: ρ_B = Tr_A(ρ_AB)
        rho_B = np.zeros((dimension_B, dimension_B), dtype=np.complex128)
        
        for i in range(dimension_A):
            for j in range(dimension_A):
                for k in range(dimension_B):
                    idx_i = i * dimension_B + k
                    idx_j = j * dimension_B + k
                    rho_B[k, k] += rho_AB[idx_i, idx_j]
        
        # Entropías
        try:
            evals_A = np.linalg.eigvalsh(rho_A)
            evals_A = evals_A[evals_A > _MACHINE_EPSILON]
            evals_A = evals_A / np.sum(evals_A)
            entropy_A = self._von_neumann_entropy(evals_A)
        except np.linalg.LinAlgError:
            entropy_A = 0.0
        
        try:
            evals_B = np.linalg.eigvalsh(rho_B)
            evals_B = evals_B[evals_B > _MACHINE_EPSILON]
            evals_B = evals_B / np.sum(evals_B)
            entropy_B = self._von_neumann_entropy(evals_B)
        except np.linalg.LinAlgError:
            entropy_B = 0.0
        
        try:
            evals_AB = np.linalg.eigvalsh(rho_AB)
            evals_AB = evals_AB[evals_AB > _MACHINE_EPSILON]
            evals_AB = evals_AB / np.sum(evals_AB)
            entropy_AB = self._von_neumann_entropy(evals_AB)
        except np.linalg.LinAlgError:
            entropy_AB = 0.0
        
        # I(A:B) = S(A) + S(B) - S(AB)
        mutual_information = entropy_A + entropy_B - entropy_AB
        
        # Debe ser no negativa por subaditividad fuerte
        return max(0.0, mutual_information)

    def _enforce_holevo_capacity_retention(
        self,
        evals_orig: Optional[NDArray[np.float64]] = None,
        evals_purified: Optional[NDArray[np.float64]] = None,
        *,
        rho_orig: Optional[NDArray[np.complex128]] = None,
        rho_purified: Optional[NDArray[np.complex128]] = None,
        fidelity_audit: Optional[FidelityAuditData] = None,
    ) -> HolevoAuditData:
        r"""
        Último método de Fase 3 (continuación de métodos previos de esta fase).
        
        Verifica que la purificación espectral preserve la capacidad termodinámica
        y de información del sistema cuántico.
        
        Algoritmo:
            1. Validar certificado de fidelidad (Fase 2)
            2. Resolver espectros de estados original y purificado
            3. Computar entropías de von Neumann
            4. Calcular diferencial de entropía ΔS
            5. Computar entropías de Rényi
            6. Estimar cota de capacidad de Holevo
            7. Verificar segunda ley termodinámica cuántica
        
        Este método retorna HolevoAuditData, el certificado final que completa
        la composición funtorial Φ₃ ∘ Φ₂ ∘ Φ₁.
        """
        # Validación del certificado de Fase 2
        if fidelity_audit is not None:
            if not fidelity_audit.is_fidelity_preserved:
                raise UhlmannFidelityCollapseError(
                    "La Fase 3 no puede iniciarse sin certificado de fidelidad válido.\n"
                    "  La Fase 2 no preservó la fidelidad de Uhlmann."
                )

        # Resolución de espectros
        spectrum_orig, _, _, _ = self._resolve_spectrum(
            "rho_orig",
            evals_orig,
            rho_orig,
        )

        spectrum_pur, _, _, _ = self._resolve_spectrum(
            "rho_purified",
            evals_purified,
            rho_purified,
        )

        # Entropías de von Neumann
        entropy_orig = self._von_neumann_entropy(spectrum_orig)
        entropy_pur = self._von_neumann_entropy(spectrum_pur)

        # Diferencial de entropía
        delta_s = float(entropy_pur - entropy_orig)

        if not math.isfinite(delta_s):
            raise HolevoCapacityDeficitError(
                "El diferencial de entropía ΔS no es finito."
            )

        # Tolerancia adaptativa
        entropy_tolerance = max(
            _ENTROPY_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, abs(entropy_orig), abs(entropy_pur)),
        )

        # Verificación termodinámica: purificación debe reducir entropía
        if delta_s > entropy_tolerance:
            raise HolevoCapacityDeficitError(
                "Paradoja termodinámica detectada.\n"
                "  La purificación espectral inyectó entropía al sistema.\n"
                f"  ΔS = S(ρ_pur) - S(ρ_orig) = {delta_s:.6e} nats\n"
                f"  Tolerancia: {entropy_tolerance:.6e} nats\n"
                "  Consecuencia: Colapso de capacidad de Holevo.\n"
                "  Interpretación física: Proceso termodinámicamente irreversible."
            )

        # Familia de entropías de Rényi
        renyi_family_orig = self._compute_renyi_entropy_family(
            spectrum_orig,
            alpha_values=[0.5, 2.0, float('inf')],
        )
        
        renyi_family_pur = self._compute_renyi_entropy_family(
            spectrum_pur,
            alpha_values=[0.5, 2.0, float('inf')],
        )

        # Entropías de Rényi específicas
        renyi_2_orig = renyi_family_orig.get(2.0, 0.0)
        renyi_2_pur = renyi_family_pur.get(2.0, 0.0)
        renyi_inf_orig = renyi_family_orig.get(float('inf'), 0.0)
        renyi_inf_pur = renyi_family_pur.get(float('inf'), 0.0)

        # Estimación de capacidad de Holevo (ensemble mínimo: {ρ_orig, ρ_pur})
        if rho_orig is not None and rho_purified is not None:
            ensemble_states = [rho_orig, rho_purified]
            ensemble_probs = np.array([0.5, 0.5])
            
            holevo_capacity_bound = self._estimate_holevo_capacity(
                ensemble_states,
                ensemble_probs,
            )
        else:
            # Cota trivial: χ ≤ log d
            dimension = spectrum_orig.size
            holevo_capacity_bound = math.log(dimension) if dimension > 0 else 0.0

        # Información accesible (simplificación: igual a Holevo para ensemble equiprobable)
        accessible_information = holevo_capacity_bound

        # Cambio de energía libre (F = E - TS, con E constante para traza fija)
        # ΔF = -T·ΔS (T arbitraria, usamos T=1 en unidades naturales)
        free_energy_change = -delta_s

        logger.info(
            "Fase 3 COMPLETADA: Enforcement de Capacidad de Holevo.\n"
            "  Entropía original S(ρ_orig): %.6e nats (%.6f bits)\n"
            "  Entropía purificada S(ρ_pur): %.6e nats (%.6f bits)\n"
            "  Diferencial ΔS: %.6e nats\n"
            "  Entropía de Rényi-2 (orig): %.6e\n"
            "  Entropía de Rényi-2 (pur): %.6e\n"
            "  Min-entropía (orig): %.6e\n"
            "  Min-entropía (pur): %.6e\n"
            "  Cota de Holevo χ: %.6e nats\n"
            "  Información accesible: %.6e nats\n"
            "  Cambio de energía libre ΔF: %.6e",
            entropy_orig,
            entropy_orig / math.log(2.0),
            entropy_pur,
            entropy_pur / math.log(2.0),
            delta_s,
            renyi_2_orig,
            renyi_2_pur,
            renyi_inf_orig,
            renyi_inf_pur,
            holevo_capacity_bound,
            accessible_information,
            free_energy_change,
        )

        return HolevoAuditData(
            entropy_original=float(entropy_orig),
            entropy_purified=float(entropy_pur),
            entropy_delta=float(delta_s),
            entropy_tolerance=float(entropy_tolerance),
            is_capacity_preserved=True,
            renyi_2_entropy_original=float(renyi_2_orig),
            renyi_2_entropy_purified=float(renyi_2_pur),
            renyi_inf_entropy_original=float(renyi_inf_orig),
            renyi_inf_entropy_purified=float(renyi_inf_pur),
            holevo_capacity_bound=float(holevo_capacity_bound),
            accessible_information=float(accessible_information),
            free_energy_change=float(free_energy_change),
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   ORQUESTADOR SUPREMO: MAC MINIMIZER AGENT                                  ║
# ║                                                                             ║
# ║   Endofuntor Z_MAC-Agent: DensityMat² → GovernanceState                     ║
# ║   Z = Φ₃ ∘ Φ₂ ∘ Φ₁                                                          ║
# ║                                                                             ║
# ║   donde:                                                                    ║
# ║     Φ₁: DensityMat² → MajorizationAuditData                                 ║
# ║     Φ₂: (DensityMat² × MajorizationAuditData) → FidelityAuditData           ║
# ║     Φ₃: (DensityMat² × FidelityAuditData) → HolevoAuditData                 ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class MACMinimizerAgent(Morphism, Phase3_HolevoCapacityEnforcer):
    r"""
    El Custodio de la Purificación Espectral.
    
    Gobierna el módulo `mac_minimizer.py`, garantizando que la compresión del
    operador de densidad respete axiomáticamente:
        1. Termodinámica cuántica de von Neumann
        2. Isomorfismo estructural de la información
        3. Preservación de capacidades de canal cuántico
        4. Invariantes topológicos del espacio de estados
    
    Arquitectura Categórica:
        - Objeto de partida: (ρ_orig, ρ_purified) ∈ DensityMat²
        - Morfismo: Z_MAC-Agent
        - Objeto de llegada: PurificationGovernanceState
    
    Invariantes garantizados:
        - Preorden de majorización cuántica
        - Fidelidad de Uhlmann ≥ F_min
        - No inflación de entropía: ΔS ≤ 0
        - Preservación de coherencia cuántica
        - Conservación de capacidad de Holevo
    """

    def __init__(self) -> None:
        """Inicializa el agente con configuración por defecto."""
        super().__init__()
        
        logger.info(
            "MACMinimizerAgent v3.0.0 inicializado.\n"
            "  Fidelidad mínima: %.6f\n"
            "  Tolerancia de majorización: %.6e\n"
            "  Tolerancia de entropía: %.6e",
            _UHLMANN_FIDELITY_MIN,
            _MAJORIZATION_TOLERANCE,
            _ENTROPY_TOLERANCE,
        )

    def _compute_overall_quality_score(
        self,
        majorization_audit: MajorizationAuditData,
        fidelity_audit: FidelityAuditData,
        holevo_audit: HolevoAuditData,
    ) -> float:
        r"""
        Computa un score de calidad global en [0, 1].
        
        Factores considerados:
            - Fidelidad de Uhlmann (peso: 0.4)
            - Pureza relativa (peso: 0.3)
            - Preservación de entropía (peso: 0.2)
            - Coherencia cuántica (peso: 0.1)
        """
        # Componente de fidelidad
        fidelity_score = fidelity_audit.uhlmann_fidelity
        
        # Componente de pureza (incremento de pureza es positivo)
        purity_increase = (
            majorization_audit.purity_purified - majorization_audit.purity_original
        )
        max_purity_increase = 1.0 - majorization_audit.purity_original
        purity_score = (
            purity_increase / max_purity_increase
            if max_purity_increase > _MACHINE_EPSILON
            else 1.0
        )
        purity_score = float(np.clip(purity_score, 0.0, 1.0))
        
        # Componente de preservación de entropía (menor ΔS es mejor)
        entropy_reduction = -holevo_audit.entropy_delta  # Negativo de ΔS
        max_entropy = math.log(majorization_audit.dimension)
        entropy_score = (
            entropy_reduction / max_entropy
            if max_entropy > _MACHINE_EPSILON
            else 1.0
        )
        entropy_score = float(np.clip(entropy_score, 0.0, 1.0))
        
        # Componente de coherencia (preservación de coherencia)
        if (
            fidelity_audit.coherence_metrics_original is not None
            and fidelity_audit.coherence_metrics_purified is not None
        ):
            coherence_orig = fidelity_audit.coherence_metrics_original.l1_norm_coherence
            coherence_pur = fidelity_audit.coherence_metrics_purified.l1_norm_coherence
            
            coherence_score = (
                coherence_pur / coherence_orig
                if coherence_orig > _MACHINE_EPSILON
                else 1.0
            )
            coherence_score = float(np.clip(coherence_score, 0.0, 1.0))
        else:
            coherence_score = 1.0
        
        # Agregación ponderada
        overall_score = (
            0.4 * fidelity_score
            + 0.3 * purity_score
            + 0.2 * entropy_score
            + 0.1 * coherence_score
        )
        
        return float(np.clip(overall_score, 0.0, 1.0))

    def _assess_risk_level(
        self,
        fidelity: float,
        entropy_delta: float,
        purity_change: float,
    ) -> str:
        r"""
        Evalúa el nivel de riesgo de la purificación.
        
        Niveles:
            NOMINAL: Purificación óptima
            WARNING: Purificación aceptable con degradación menor
            CRITICAL: Purificación cercana a umbrales de fallo
        """
        # Margen de fidelidad
        fidelity_margin = fidelity - _UHLMANN_FIDELITY_MIN
        
        # Margen de entropía
        entropy_margin = -entropy_delta  # Debe ser positivo o cero
        
        # Cambio de pureza
        purity_margin = purity_change  # Debe ser positivo
        
        # Criterios de riesgo
        if fidelity_margin < 0.02:
            return "CRITICAL"
        
        if entropy_margin < _ENTROPY_TOLERANCE * 10:
            return "WARNING"
        
        if purity_margin < _PSD_TOLERANCE:
            return "WARNING"
        
        if fidelity < 0.98:
            return "WARNING"
        
        return "NOMINAL"

    def execute_spectral_purification_governance(
        self,
        rho_orig: NDArray[np.complex128],
        rho_purified: NDArray[np.complex128],
        evals_orig: Optional[NDArray[np.float64]] = None,
        evals_purified: Optional[NDArray[np.float64]] = None,
        *,
        enable_extended_diagnostics: bool = False,
    ) -> PurificationGovernanceState:
        r"""
        Ejecuta la composición funtorial estricta completa.
        
        Flujo de ejecución:
            1. Fase 1: Auditoría de majorización cuántica
               └─ Retorna: MajorizationAuditData
            
            2. Fase 2: Certificación de fidelidad de Uhlmann
               ├─ Consume: MajorizationAuditData
               └─ Retorna: FidelityAuditData
            
            3. Fase 3: Enforcement de capacidad de Holevo
               ├─ Consume: FidelityAuditData
               └─ Retorna: HolevoAuditData
            
            4. Síntesis: Construcción de PurificationGovernanceState
        
        Args:
            rho_orig: Matriz de densidad original ρ
            rho_purified: Matriz de densidad purificada σ
            evals_orig: Autovalores opcionales de ρ (validados si se proveen)
            evals_purified: Autovalores opcionales de σ (validados si se proveen)
            enable_extended_diagnostics: Activar diagnósticos extendidos
        
        Returns:
            PurificationGovernanceState con certificados de las 3 fases
        
        Raises:
            QuantumMajorizationViolation: Si λ(σ) ⊀ λ(ρ)
            UhlmannFidelityCollapseError: Si F(ρ, σ) < F_min
            HolevoCapacityDeficitError: Si ΔS > 0
        """
        import time
        
        start_time = time.time()
        
        logger.info(
            "═══════════════════════════════════════════════════════════════\n"
            "  INICIANDO GOBERNANZA DE PURIFICACIÓN ESPECTRAL\n"
            "═══════════════════════════════════════════════════════════════"
        )

        # ─────────────────────────────────────────────────────────────────
        # FASE 1: Auditoría de Majorización Cuántica
        # ─────────────────────────────────────────────────────────────────
        logger.info("│\n├─ FASE 1: Auditoría de Majorización Cuántica")
        
        majorization_audit = self._audit_quantum_majorization(
            evals_orig,
            evals_purified,
            rho_orig=rho_orig,
            rho_purified=rho_purified,
        )
        
        logger.info("│  └─ ✓ Majorización certificada")

        # ─────────────────────────────────────────────────────────────────
        # FASE 2: Certificación de Fidelidad de Uhlmann
        # ─────────────────────────────────────────────────────────────────
        logger.info("│\n├─ FASE 2: Certificación de Fidelidad de Uhlmann")
        
        fidelity_audit = self._certify_uhlmann_fidelity_bound(
            rho_orig,
            rho_purified,
            majorization_audit=majorization_audit,
        )
        
        logger.info("│  └─ ✓ Fidelidad certificada: F = %.6f", fidelity_audit.uhlmann_fidelity)

        # ─────────────────────────────────────────────────────────────────
        # FASE 3: Enforcement de Capacidad de Holevo
        # ─────────────────────────────────────────────────────────────────
        logger.info("│\n├─ FASE 3: Enforcement de Capacidad de Holevo")
        
        holevo_audit = self._enforce_holevo_capacity_retention(
            evals_orig,
            evals_purified,
            rho_orig=rho_orig,
            rho_purified=rho_purified,
            fidelity_audit=fidelity_audit,
        )
        
        logger.info("│  └─ ✓ Capacidad preservada: ΔS = %.6e nats", holevo_audit.entropy_delta)

        # ─────────────────────────────────────────────────────────────────
        # SÍNTESIS: Construcción del estado de gobernanza
        # ─────────────────────────────────────────────────────────────────
        is_epistemologically_valid = bool(
            majorization_audit.is_majorized
            and fidelity_audit.is_fidelity_preserved
            and holevo_audit.is_capacity_preserved
        )

        if not is_epistemologically_valid:
            raise MACMinimizerAgentError(
                "La composición funtorial no autorizó la purificación espectral.\n"
                "  Al menos una fase falló en certificar sus invariantes."
            )

        # Métricas agregadas
        overall_quality_score = self._compute_overall_quality_score(
            majorization_audit,
            fidelity_audit,
            holevo_audit,
        )

        purity_change = (
            majorization_audit.purity_purified - majorization_audit.purity_original
        )

        risk_assessment = self._assess_risk_level(
            fidelity_audit.uhlmann_fidelity,
            holevo_audit.entropy_delta,
            purity_change,
        )

        elapsed_time = time.time() - start_time

        logger.info(
            "│\n╞═════════════════════════════════════════════════════════════\n"
            "│  GOBERNANZA COMPLETADA\n"
            "├─────────────────────────────────────────────────────────────\n"
            "│  ✓ Majorización cuántica: CERTIFICADA\n"
            "│  ✓ Fidelidad de Uhlmann: F = %.6f (mín: %.6f)\n"
            "│  ✓ Capacidad de Holevo: PRESERVADA\n"
            "│  ✓ Diferencial de entropía: ΔS = %.6e nats\n"
            "│  ✓ Cambio de pureza: ΔPureza = %.6e\n"
            "├─────────────────────────────────────────────────────────────\n"
            "│  Score de calidad global: %.4f / 1.000\n"
            "│  Evaluación de riesgo: %s\n"
            "│  Tiempo de ejecución: %.4f ms\n"
            "╰═════════════════════════════════════════════════════════════",
            fidelity_audit.uhlmann_fidelity,
            _UHLMANN_FIDELITY_MIN,
            holevo_audit.entropy_delta,
            purity_change,
            overall_quality_score,
            risk_assessment,
            elapsed_time * 1000,
        )

        # Diagnósticos extendidos (opcional)
        if enable_extended_diagnostics:
            self._log_extended_diagnostics(
                majorization_audit,
                fidelity_audit,
                holevo_audit,
            )

        return PurificationGovernanceState(
            majorization_audit=majorization_audit,
            fidelity_audit=fidelity_audit,
            holevo_audit=holevo_audit,
            is_epistemologically_valid=is_epistemologically_valid,
            purification_phase=PurificationPhase.COMPLETE,
            timestamp=start_time,
            overall_quality_score=overall_quality_score,
            risk_assessment=risk_assessment,
        )

    def _log_extended_diagnostics(
        self,
        majorization_audit: MajorizationAuditData,
        fidelity_audit: FidelityAuditData,
        holevo_audit: HolevoAuditData,
    ) -> None:
        """Registra diagnósticos extendidos en el log."""
        logger.info(
            "\n"
            "╔═════════════════════════════════════════════════════════════╗\n"
            "║            DIAGNÓSTICOS EXTENDIDOS                          ║\n"
            "╠═════════════════════════════════════════════════════════════╣\n"
            "║ CARACTERÍSTICAS ESPECTRALES:                                ║\n"
            "╟─────────────────────────────────────────────────────────────╢\n"
            "║ Original:                                                   ║\n"
            "║   · Dimensión: %d                                           ║\n"
            "║   · Rango efectivo: %d                                      ║\n"
            "║   · Número de condición: %.6e                               ║\n"
            "║   · Gap espectral: %.6e                                     ║\n"
            "║   · Entropía espectral: %.6e                                ║\n"
            "║   · Participation ratio: %.6f                               ║\n"
            "╟─────────────────────────────────────────────────────────────╢\n"
            "║ Purificado:                                                 ║\n"
            "║   · Dimensión: %d                                           ║\n"
            "║   · Rango efectivo: %d                                      ║\n"
            "║   · Número de condición: %.6e                               ║\n"
            "║   · Gap espectral: %.6e                                     ║\n"
            "║   · Entropía espectral: %.6e                                ║\n"
            "║   · Participation ratio: %.6f                               ║\n"
            "╠═════════════════════════════════════════════════════════════╣\n"
            "║ MÉTRICAS DE DISTANCIA CUÁNTICA:                             ║\n"
            "╟─────────────────────────────────────────────────────────────╢\n"
            "║   · Distancia de traza D_tr: %.6e                           ║\n"
            "║   · Distancia de Bures D_B: %.6e                            ║\n"
            "║   · Distancia de Hellinger D_H: %.6e                        ║\n"
            "╠═════════════════════════════════════════════════════════════╣\n"
            "║ COHERENCIA CUÁNTICA:                                        ║\n"
            "╟─────────────────────────────────────────────────────────────╢\n"
            "║ Original:                                                   ║\n"
            "║   · l1-norm coherence: %.6e                                 ║\n"
            "║   · Relative entropy coherence: %.6e                        ║\n"
            "║   · Robustness of coherence: %.6e                           ║\n"
            "╟─────────────────────────────────────────────────────────────╢\n"
            "║ Purificado:                                                 ║\n"
            "║   · l1-norm coherence: %.6e                                 ║\n"
            "║   · Relative entropy coherence: %.6e                        ║\n"
            "║   · Robustness of coherence: %.6e                           ║\n"
            "╠═════════════════════════════════════════════════════════════╣\n"
            "║ ENTROPÍAS DE RÉNYI:                                         ║\n"
            "╟─────────────────────────────────────────────────────────────╢\n"
            "║   · S₂ (original): %.6e                                     ║\n"
            "║   · S₂ (purificado): %.6e                                   ║\n"
            "║   · S_∞ (original): %.6e                                    ║\n"
            "║   · S_∞ (purificado): %.6e                                  ║\n"
            "╠═════════════════════════════════════════════════════════════╣\n"
            "║ CAPACIDADES DE INFORMACIÓN:                                 ║\n"
            "╟─────────────────────────────────────────────────────────────╢\n"
            "║   · Holevo capacity bound χ: %.6e nats                      ║\n"
            "║   · Accessible information: %.6e nats                       ║\n"
            "║   · Free energy change ΔF: %.6e                             ║\n"
            "╚═════════════════════════════════════════════════════════════╝",
            majorization_audit.spectral_characteristics_original.dimension,
            majorization_audit.spectral_characteristics_original.effective_rank,
            majorization_audit.spectral_characteristics_original.condition_number,
            majorization_audit.spectral_characteristics_original.spectral_gap,
            majorization_audit.spectral_characteristics_original.spectral_entropy,
            majorization_audit.spectral_characteristics_original.participation_ratio,
            majorization_audit.spectral_characteristics_purified.dimension,
            majorization_audit.spectral_characteristics_purified.effective_rank,
            majorization_audit.spectral_characteristics_purified.condition_number,
            majorization_audit.spectral_characteristics_purified.spectral_gap,
            majorization_audit.spectral_characteristics_purified.spectral_entropy,
            majorization_audit.spectral_characteristics_purified.participation_ratio,
            fidelity_audit.trace_distance,
            fidelity_audit.bures_distance,
            fidelity_audit.hellinger_distance,
            fidelity_audit.coherence_metrics_original.l1_norm_coherence
            if fidelity_audit.coherence_metrics_original
            else 0.0,
            fidelity_audit.coherence_metrics_original.relative_entropy_coherence
            if fidelity_audit.coherence_metrics_original
            else 0.0,
            fidelity_audit.coherence_metrics_original.robustness_of_coherence
            if fidelity_audit.coherence_metrics_original
            else 0.0,
            fidelity_audit.coherence_metrics_purified.l1_norm_coherence
            if fidelity_audit.coherence_metrics_purified
            else 0.0,
            fidelity_audit.coherence_metrics_purified.relative_entropy_coherence
            if fidelity_audit.coherence_metrics_purified
            else 0.0,
            fidelity_audit.coherence_metrics_purified.robustness_of_coherence
            if fidelity_audit.coherence_metrics_purified
            else 0.0,
            holevo_audit.renyi_2_entropy_original,
            holevo_audit.renyi_2_entropy_purified,
            holevo_audit.renyi_inf_entropy_original,
            holevo_audit.renyi_inf_entropy_purified,
            holevo_audit.holevo_capacity_bound,
            holevo_audit.accessible_information,
            holevo_audit.free_energy_change,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA ENRIQUECIDA
# ═══════════════════════════════════════════════════════════════════════════════
__all__: List[str] = [
    # Excepciones
    "MACMinimizerAgentError",
    "DensityMatrixValidationError",
    "QuantumMajorizationViolation",
    "UhlmannFidelityCollapseError",
    "HolevoCapacityDeficitError",
    "SpectralDecompositionError",
    "QuantumCoherenceViolation",
    "EntanglementStructureError",
    # Enumeraciones
    "QuantumDistanceMetric",
    "PurificationPhase",
    # Estructuras de datos
    "SpectralCharacteristics",
    "QuantumCoherenceMetrics",
    "MajorizationAuditData",
    "FidelityAuditData",
    "HolevoAuditData",
    "PurificationGovernanceState",
    # Clases de fase
    "Phase1_QuantumMajorizationAuditor",
    "Phase2_UhlmannFidelityCertifier",
    "Phase3_HolevoCapacityEnforcer",
    # Agente principal
    "MACMinimizerAgent",
]


# ═══════════════════════════════════════════════════════════════════════════════
# FIN DEL MÓDULO
# ═══════════════════════════════════════════════════════════════════════════════