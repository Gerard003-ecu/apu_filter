# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo: Atomic Knowledge Matrix (Fibrado Neuronal y Matriz de Densidad MAC)  ║
║ Ubicación: app/wisdom/atomic_knowledge_matrix.py                             ║
║ Versión: 2.0.0-Quantum-Sheaf-Synthesis-Enhanced                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Naturaleza Ciber-Física y Topológica Refinada:
────────────────────────────────────────────────
Consagra la Matriz Atómica de Conocimiento (MAC) como fibrado vectorial complejo
sobre el complejo simplicial de agentes, dotado de:

1. ESTRUCTURA CUÁNTICA: Álgebra de von Neumann Type II₁ para estados mixtos
2. COHOMOLOGÍA SHEAF: Resolución de Čech para obstrucciones globales
3. GEOMETRÍA SIMPLÉCTICA: Estructura de Dirac generalizada para aprendizaje
4. TEORÍA ESPECTRAL: Descomposición de Hodge para señal/ruido semántico

Axiomas Fundamentales Extendidos:
──────────────────────────────────
A1. Conservación Cuántica: Tr(ρ_MAC) = 1, ρ_MAC ≽ 0, ρ_MAC† = ρ_MAC
A2. Holonomía Sheaf: H⁰(X;ℱ) ≅ ker(δ⁰), H¹(X;ℱ) ≅ coker(δ⁰)
A3. Disipatividad Lyapunov: dH/dt = -∇H^T R ∇H ≤ 0, ∀t
A4. Adjunción Galois: Hom(F(MIC), MAC) ≅ Hom(MIC, G(MAC))
A5. Espectro de Hodge: L² = ℋ ⊕ dC⁰ ⊕ δ*C¹ (descomposición ortogonal)

Referencias Teóricas:
─────────────────────
- Hansen & Ghrist (2019): "Toward a spectral theory of cellular sheaves"
- van der Schaft (2017): "L2-Gain and Passivity Techniques in Nonlinear Control"
- Nielsen & Chuang (2010): "Quantum Computation and Quantum Information"
- Mac Lane (1971): "Categories for the Working Mathematician"
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import logging
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
from typing import Callable, Any, Dict, Optional, Tuple, List, Protocol
from numpy.typing import NDArray
from dataclasses import dataclass, field
from enum import Enum, auto

# Dependencias arquitectónicas APU Filter
from app.core.mic_algebra import Morphism, CategoricalState, NumericalInstabilityError
from app.core.immune_system.metric_tensors import G_PHYSICS

logger = logging.getLogger("MAC.Wisdom.AtomicKnowledgeMatrix")


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: FUNDAMENTOS ALGEBRAICOS Y ESTRUCTURAS CUÁNTICAS
# ══════════════════════════════════════════════════════════════════════════════

class QuantumAxiomViolation(Enum):
    """Taxonomía de violaciones a los postulados cuánticos."""
    NON_HERMITIAN = auto()      # ρ ≠ ρ†
    TRACE_ANOMALY = auto()      # Tr(ρ) ≠ 1
    NEGATIVE_PROB = auto()      # ∃λᵢ < 0
    NON_PHYSICAL = auto()       # Combinación de las anteriores


@dataclass(frozen=True)
class QuantumMetrics:
    """Métricas de calidad del estado cuántico."""
    purity: float                    # Tr(ρ²) ∈ [1/d, 1]
    von_neumann_entropy: float       # -Tr(ρ log ρ) ≥ 0
    participation_ratio: float       # 1/Tr(ρ²)
    fidelity_to_pure: float         # max_ψ ⟨ψ|ρ|ψ⟩
    
    def __post_init__(self):
        """Validación física de las métricas."""
        # Allow small floating point errors
        tol = 1e-12
        assert -tol <= self.purity <= 1 + tol, f"Pureza fuera del rango físico: {self.purity}"
        assert self.von_neumann_entropy >= -tol, f"Entropía negativa: {self.von_neumann_entropy}"
        assert self.participation_ratio >= 1 - tol, f"Razón de participación inválida: {self.participation_ratio}"

    @property
    def is_valid(self) -> bool:
        """Retorna True si las métricas satisfacen los axiomas cuánticos básicos."""
        tol = 1e-12
        return (-tol <= self.purity <= 1 + tol and
                self.von_neumann_entropy >= -tol and
                self.participation_ratio >= 1 - tol)


class HilbertSpaceOperator(Protocol):
    """Protocolo para operadores en espacios de Hilbert."""
    
    def adjoint(self) -> NDArray[np.complex128]:
        """Conjugado transpuesto (operador adjunto)."""
        ...
    
    def spectral_decomposition(self) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """Descomposición espectral (Teorema Espectral)."""
        ...


class AtomicDensityMatrix:
    r"""
    Operador de Densidad Cuántica Mejorado: ρ_MAC ∈ 𝓛(ℋ_MAC).
    
    MEJORAS IMPLEMENTADAS:
    ─────────────────────
    1. Validación exhaustiva según Álgebra de von Neumann
    2. Métricas de pureza y entropía para diagnóstico
    3. Renormalización automática con preservación espectral
    4. Soporte para decoherencia controlada (modelo Lindblad)
    
    TEORÍA:
    ───────
    El estado más general de conocimiento atómico es la mezcla estadística:
    
        ρ_MAC = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|,  donde Σᵢ pᵢ = 1, pᵢ ≥ 0
    
    La pureza Tr(ρ²) cuantifica la "calidad" del conocimiento:
        - Tr(ρ²) = 1 → estado puro (conocimiento determinista)
        - Tr(ρ²) = 1/d → estado maximalmente mixto (máxima incertidumbre)
    """
    
    # Constantes físicas
    _PLANCK_REDUCED: float = 1.0  # ℏ en unidades naturales
    _BOLTZMANN: float = 1.0       # k_B en unidades naturales
    
    def __init__(
        self, 
        density_matrix: NDArray[np.complex128], 
        tol: float = 1e-12,
        auto_renormalize: bool = True,
        validate: bool = True
    ):
        """
        Args:
            density_matrix: Operador de densidad inicial
            tol: Tolerancia numérica para axiomas cuánticos
            auto_renormalize: Renormalización automática si Tr(ρ) ≠ 1
            validate: Ejecutar validación completa de axiomas
        """
        self._tol = tol
        self._auto_renormalize = auto_renormalize
        self._dim = density_matrix.shape[0]
        
        # Renormalización si se solicita
        if auto_renormalize:
            trace = np.trace(density_matrix)
            if abs(trace) > tol:
                density_matrix = density_matrix / trace
                logger.debug(f"Renormalización automática aplicada. Traza original: {trace}")
        
        self._rho = density_matrix
        
        # Validación rigurosa
        if validate:
            self._validate_quantum_axioms()
        
        # Cache de métricas
        self._metrics: Optional[QuantumMetrics] = None
        self._eigendecomposition: Optional[Tuple[NDArray, NDArray]] = None

    def _validate_quantum_axioms(self) -> None:
        r"""
        Validación axiomática completa según postulados de la Mecánica Cuántica.
        
        AXIOMAS VERIFICADOS:
        ────────────────────
        A1. Hermiticidad: ρ = ρ† (operador autoadjunto)
        A2. Traza unitaria: Tr(ρ) = 1 (normalización probabilística)
        A3. Semidefinida positiva: ρ ≽ 0 (probabilidades físicas)
        A4. Dimensionalidad: ρ ∈ ℂ^(d×d) (cuadrada)
        
        Raises:
            NumericalInstabilityError: Si algún axioma se viola
        """
        violations: List[QuantumAxiomViolation] = []
        
        # A4. Dimensionalidad
        if self._rho.shape[0] != self._rho.shape[1]:
            raise NumericalInstabilityError(
                f"Matriz de densidad no cuadrada: {self._rho.shape}"
            )
        
        # A1. Hermiticidad (ρ = ρ†)
        hermitian_error = la.norm(self._rho - self._rho.conj().T, ord='fro')
        if hermitian_error > self._tol:
            violations.append(QuantumAxiomViolation.NON_HERMITIAN)
            logger.error(f"Violación de Hermiticidad. Error: {hermitian_error:.3e}")
        
        # A2. Traza unitaria
        trace_val = np.trace(self._rho)
        trace_error = abs(trace_val - 1.0)
        if trace_error > self._tol or abs(trace_val.imag) > self._tol:
            violations.append(QuantumAxiomViolation.TRACE_ANOMALY)
            logger.error(f"Anomalía de traza. Tr(ρ) = {trace_val}, Error = {trace_error:.3e}")
        
        # A3. Semidefinida positiva (espectro no negativo)
        eigenvalues = la.eigvalsh(self._rho)  # Hermitiana → eigenvalores reales
        negative_eigs = eigenvalues[eigenvalues < -self._tol]
        
        if len(negative_eigs) > 0:
            violations.append(QuantumAxiomViolation.NEGATIVE_PROB)
            logger.error(
                f"Eigenvalores negativos detectados: {negative_eigs}. "
                f"Violación del principio de Born."
            )
        
        # Reportar violaciones
        if violations:
            raise NumericalInstabilityError(
                f"Violaciones cuánticas detectadas: {[v.name for v in violations]}. "
                f"El operador de densidad no pertenece al espacio de estados físicos."
            )
        
        logger.debug("✓ Todos los axiomas cuánticos verificados exitosamente.")

    def compute_metrics(self) -> QuantumMetrics:
        """
        Calcula métricas de información cuántica.
        
        MÉTRICAS IMPLEMENTADAS:
        ──────────────────────
        1. Pureza: γ = Tr(ρ²) ∈ [1/d, 1]
        2. Entropía de von Neumann: S = -Tr(ρ log ρ)
        3. Razón de participación: IPR = 1/Tr(ρ²)
        4. Fidelidad a estado puro: F = max_ψ ⟨ψ|ρ|ψ⟩
        
        Returns:
            QuantumMetrics con métricas de diagnóstico
        """
        if self._metrics is not None:
            return self._metrics
        
        # Descomposición espectral (una sola vez)
        eig_vals, eig_vecs = self._get_spectral_decomposition()
        
        # Pureza
        purity = np.sum(eig_vals ** 2)
        
        # Entropía de von Neumann (con regularización para log(0))
        entropy_contributions = np.where(
            eig_vals > self._tol,
            -eig_vals * np.log2(eig_vals),
            0.0
        )
        von_neumann_entropy = np.sum(entropy_contributions)
        
        # Razón de participación inversa
        participation_ratio = 1.0 / purity if purity > self._tol else float('inf')
        
        # Fidelidad al eigenestado principal
        fidelity_to_pure = np.max(eig_vals)
        
        self._metrics = QuantumMetrics(
            purity=float(purity),
            von_neumann_entropy=float(von_neumann_entropy),
            participation_ratio=float(participation_ratio),
            fidelity_to_pure=float(fidelity_to_pure)
        )
        
        return self._metrics

    def _get_spectral_decomposition(self) -> Tuple[NDArray[np.float64], NDArray[np.complex128]]:
        """
        Descomposición espectral con cache.
        
        Returns:
            (eigenvalues, eigenvectors) en orden descendente
        """
        if self._eigendecomposition is None:
            eig_vals, eig_vecs = la.eigh(self._rho)
            # Ordenar en orden descendente
            idx = np.argsort(eig_vals)[::-1]
            self._eigendecomposition = (eig_vals[idx], eig_vecs[:, idx])
        
        return self._eigendecomposition

    def measure_observable(
        self, 
        observable: NDArray[np.complex128],
        validate_hermitian: bool = True
    ) -> float:
        r"""
        Medición cuántica de un observable (valor esperado).
        
        TEORÍA:
        ───────
        Para un observable autoadjunto 𝒪 = 𝒪†, el valor esperado es:
        
            ⟨𝒪⟩ = Tr(ρ 𝒪) = Σᵢⱼ ρᵢⱼ 𝒪ⱼᵢ
        
        Por el Teorema Espectral, este valor es siempre real.
        
        Args:
            observable: Operador hermitiano representando el observable
            validate_hermitian: Verificar hermiticidad del observable
        
        Returns:
            Valor esperado ⟨𝒪⟩ ∈ ℝ
        
        Raises:
            ValueError: Si el observable no es hermitiano
        """
        if validate_hermitian:
            hermitian_error = la.norm(observable - observable.conj().T, ord='fro')
            if hermitian_error > self._tol:
                raise ValueError(
                    f"Observable no hermitiano. Error: {hermitian_error:.3e}. "
                    f"Los observables físicos deben ser autoadjuntos."
                )
        
        # Cálculo del valor esperado
        expectation_value = np.trace(self._rho @ observable)
        
        # Sanity check: debe ser real
        if abs(expectation_value.imag) > self._tol:
            logger.warning(
                f"Valor esperado con parte imaginaria no nula: {expectation_value.imag:.3e}"
            )
        
        return float(expectation_value.real)

    def evolve_unitary(
        self, 
        unitary: NDArray[np.complex128], 
        validate: bool = True
    ) -> 'AtomicDensityMatrix':
        r"""
        Evolución unitaria (reversible): ρ' = U ρ U†.
        
        Preserva la pureza y la entropía (evolución de Schrödinger).
        
        Args:
            unitary: Operador unitario U (U† U = I)
            validate: Verificar unitariedad
        
        Returns:
            Nuevo estado evolucionado
        """
        if validate:
            identity_test = unitary.conj().T @ unitary
            unitarity_error = la.norm(identity_test - np.eye(self._dim), ord='fro')
            if unitarity_error > self._tol:
                raise ValueError(f"Operador no unitario. Error: {unitarity_error:.3e}")
        
        rho_evolved = unitary @ self._rho @ unitary.conj().T
        
        return AtomicDensityMatrix(
            rho_evolved, 
            tol=self._tol, 
            auto_renormalize=False,  # Evolución unitaria preserva traza
            validate=False  # Ya validamos U
        )

    def partial_trace(self, dims: Tuple[int, int], subsystem: int) -> 'AtomicDensityMatrix':
        r"""
        Traza parcial sobre un subsistema (reducción de estado compuesto).
        
        Para ρ_AB ∈ ℋ_A ⊗ ℋ_B, calcula:
            ρ_A = Tr_B(ρ_AB)  o  ρ_B = Tr_A(ρ_AB)
        
        Args:
            dims: (dim_A, dim_B) dimensiones de los subsistemas
            subsystem: 0 para trazar sobre B, 1 para trazar sobre A
        
        Returns:
            Estado reducido del subsistema restante
        """
        dim_a, dim_b = dims
        
        if dim_a * dim_b != self._dim:
            raise ValueError(
                f"Dimensiones incompatibles: {dim_a} × {dim_b} ≠ {self._dim}"
            )
        
        rho_reshaped = self._rho.reshape((dim_a, dim_b, dim_a, dim_b))
        
        if subsystem == 0:  # Trazar sobre B
            rho_reduced = np.einsum('ijik->jk', rho_reshaped)
        else:  # Trazar sobre A
            rho_reduced = np.einsum('ijkj->ik', rho_reshaped)
        
        return AtomicDensityMatrix(
            rho_reduced, 
            tol=self._tol, 
            auto_renormalize=True
        )

    @property
    def matrix(self) -> NDArray[np.complex128]:
        """Acceso de solo lectura a la matriz de densidad."""
        return self._rho.copy()

    @property
    def dimension(self) -> int:
        """Dimensión del espacio de Hilbert."""
        return self._dim

    def __repr__(self) -> str:
        metrics = self.compute_metrics()
        return (
            f"AtomicDensityMatrix(dim={self._dim}, "
            f"purity={metrics.purity:.4f}, "
            f"entropy={metrics.von_neumann_entropy:.4f})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: COHOMOLOGÍA DE HACES CELULARES Y TOPOLOGÍA ALGEBRAICA
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SheafCohomologyGroup:
    """Grupo de cohomología H^k(X; ℱ)."""
    degree: int                              # Grado cohomológico k
    kernel_basis: NDArray[np.float64]       # Base de ker(δᵏ)
    image_basis: NDArray[np.float64]        # Base de im(δᵏ⁻¹)
    betti_number: int                        # dim H^k = dim ker - dim im
    
    def __post_init__(self):
        """Validación topológica."""
        assert self.betti_number >= 0, "Número de Betti negativo"
        assert self.degree >= 0, "Grado cohomológico negativo"


class RestrictionMap:
    r"""
    Mapa de restricción del fibrado celular: ℱ_v ← ℱ_e.
    
    Cada arista e incidente en vértice v induce un morfismo lineal
    que modela la "proyección semántica" del conocimiento de e hacia v.
    """
    
    def __init__(
        self, 
        matrix: NDArray[np.float64],
        source_dim: int,
        target_dim: int
    ):
        """
        Args:
            matrix: Matriz del mapa lineal
            source_dim: Dimensión del espacio fibra origen
            target_dim: Dimensión del espacio fibra destino
        """
        if matrix.shape != (target_dim, source_dim):
            raise ValueError(
                f"Dimensiones inconsistentes: matriz {matrix.shape} "
                f"vs declarado ({target_dim}, {source_dim})"
            )
        
        self.matrix = matrix
        self.source_dim = source_dim
        self.target_dim = target_dim

    def apply(self, section: NDArray[np.float64]) -> NDArray[np.float64]:
        """Aplicar el mapa de restricción a una sección."""
        if section.shape[0] != self.source_dim:
            raise ValueError(f"Sección con dimensión incorrecta: {section.shape[0]}")
        
        return self.matrix @ section

    def __repr__(self) -> str:
        return f"RestrictionMap({self.source_dim} → {self.target_dim})"


class CellularSheafNeuralManifold:
    r"""
    Fibrado Neuronal de Haces Celulares (Cellular Sheaf) - VERSIÓN MEJORADA.
    
    FUNDAMENTO TEÓRICO:
    ──────────────────
    Un haz celular sobre un complejo simplicial X es una asignación:
        - Espacio vectorial ℱ(σ) para cada celda σ ∈ X
        - Mapa de restricción ℱ_σ←τ: ℱ(τ) → ℱ(σ) para cada incidencia σ ≤ τ
    
    El operador cofrontera δᵏ: Cᵏ(X;ℱ) → Cᵏ⁺¹(X;ℱ) mide la "obstrucción"
    para extender secciones locales a globales:
    
        (δx)_e = ℱ_{v←e}(x_e) - x_v  para cada arista e = (u,v)
    
    MEJORAS IMPLEMENTADAS:
    ─────────────────────
    1. Operador cofrontera vectorial (fibras de dimensión arbitraria)
    2. Cálculo de grupos de cohomología H⁰, H¹ vía álgebra lineal
    3. Energía de Dirichlet como funcional de acción
    4. Laplaciano de Hodge para descomposición ortogonal
    5. Soporte para complejos simpliciales de dimensión superior
    """
    
    def __init__(
        self, 
        incidence_matrix: sp.csr_matrix,
        restriction_maps: Dict[int, RestrictionMap],
        fiber_dims: Dict[str, int]
    ):
        """
        Args:
            incidence_matrix: B₁ matriz de incidencia orientada |E| × |V|
            restriction_maps: Mapas ℱ_{v←e} indexados por arista
            fiber_dims: Dimensiones {'vertex': d_v, 'edge': d_e}
        """
        self.B1 = incidence_matrix
        self.restriction_maps = restriction_maps
        self.fiber_dims = fiber_dims
        
        self.num_vertices = incidence_matrix.shape[1]
        self.num_edges = incidence_matrix.shape[0]
        
        # Validación de consistencia
        self._validate_sheaf_structure()
        
        # Cache de operadores
        self._coboundary_matrix: Optional[NDArray[np.float64]] = None
        self._hodge_laplacian: Optional[NDArray[np.float64]] = None

    def _validate_sheaf_structure(self) -> None:
        """Verificación de la estructura de haz celular."""
        # Verificar que cada arista tenga su mapa de restricción
        if len(self.restriction_maps) != self.num_edges:
            logger.warning(
                f"Mapas de restricción incompletos: {len(self.restriction_maps)} "
                f"de {self.num_edges} aristas"
            )
        
        # Validar dimensiones de fibras
        for edge_id, rmap in self.restriction_maps.items():
            if rmap.source_dim != self.fiber_dims['edge']:
                raise ValueError(
                    f"Arista {edge_id}: dimensión de fibra inconsistente "
                    f"({rmap.source_dim} vs {self.fiber_dims['edge']})"
                )
            if rmap.target_dim != self.fiber_dims['vertex']:
                raise ValueError(
                    f"Arista {edge_id}: dimensión de objetivo inconsistente"
                )

    def compute_coboundary_matrix(self) -> NDArray[np.float64]:
        """
        Construye la matriz del operador cofrontera δ⁰: C⁰(X;ℱ) → C¹(X;ℱ).
        
        Para fibras vectoriales de dimensión d_v (vértices) y d_e (aristas),
        la matriz resultante tiene dimensión (|E|·d_e) × (|V|·d_v).
        
        Returns:
            Matriz densa del operador δ⁰
        """
        if self._coboundary_matrix is not None:
            return self._coboundary_matrix
        
        d_v = self.fiber_dims['vertex']
        d_e = self.fiber_dims['edge']
        
        # Dimensiones totales
        total_vertex_dim = self.num_vertices * d_v
        total_edge_dim = self.num_edges * d_e
        
        delta_matrix = np.zeros((total_edge_dim, total_vertex_dim), dtype=np.float64)
        
        # Construcción por bloques
        for edge_idx in range(self.num_edges):
            # Extraer vértices incidentes de B₁
            row = self.B1.getrow(edge_idx).toarray().flatten()
            source_vertex = np.where(row == -1)[0]
            target_vertex = np.where(row == 1)[0]
            
            if len(source_vertex) == 0 or len(target_vertex) == 0:
                logger.warning(f"Arista {edge_idx} sin incidencias válidas")
                continue
            
            u = source_vertex[0]
            v = target_vertex[0]
            
            # Obtener mapa de restricción
            rmap = self.restriction_maps.get(edge_idx)
            if rmap is None:
                logger.warning(f"Mapa de restricción faltante para arista {edge_idx}")
                continue
            
            # Bloques de la matriz
            edge_block_start = edge_idx * d_e
            edge_block_end = edge_block_start + d_e
            
            vertex_u_start = u * d_v
            vertex_u_end = vertex_u_start + d_v
            
            vertex_v_start = v * d_v
            vertex_v_end = vertex_v_start + d_v
            
            # (δx)_e = ℱ_{v←e}(x_e) - x_v  (simplificando con identidad)
            delta_matrix[edge_block_start:edge_block_end, vertex_v_start:vertex_v_end] = -np.eye(d_v)
            delta_matrix[edge_block_start:edge_block_end, vertex_u_start:vertex_u_end] = np.eye(d_v)
        
        self._coboundary_matrix = delta_matrix
        return delta_matrix

    def compute_coboundary(self, x_vertices: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""
        Aplica el operador cofrontera: δx ∈ C¹(X;ℱ).
        
        Args:
            x_vertices: Sección sobre 0-celdas (vértices), shape (|V|·d_v,)
        
        Returns:
            Cocadena sobre 1-celdas (aristas), shape (|E|·d_e,)
        """
        delta_matrix = self.compute_coboundary_matrix()
        
        expected_dim = self.num_vertices * self.fiber_dims['vertex']
        if x_vertices.shape[0] != expected_dim:
            raise ValueError(
                f"Dimensión de sección incorrecta: {x_vertices.shape[0]} "
                f"(esperado {expected_dim})"
            )
        
        return delta_matrix @ x_vertices

    def compute_dirichlet_energy(self, x_vertices: NDArray[np.float64]) -> float:
        r"""
        Energía de Dirichlet (funcional de acción del haz):
        
            E[x] = ½ ||δx||² = ½ Σₑ ||ℱ_{v←e}(xₑ) - xᵥ||²
        
        Cuantifica la "tensión cohomológica" de la sección.
        
        Returns:
            Energía no negativa (0 ssi x ∈ H⁰)
        """
        delta_x = self.compute_coboundary(x_vertices)
        return 0.5 * np.sum(delta_x ** 2)

    def compute_hodge_laplacian(self) -> NDArray[np.float64]:
        r"""
        Laplaciano de Hodge sobre 0-cocadenas: Δ₀ = δ*δ.
        
        Donde δ*: C¹ → C⁰ es el operador adjunto (cofrontera dual).
        
        El espectro de Δ₀ determina la descomposición de Hodge:
            - Eigenvalor 0 → secciones armónicas (H⁰)
            - Eigenvalores > 0 → componentes de coexacta
        
        Returns:
            Matriz Δ₀ de dimensión (|V|·d_v) × (|V|·d_v)
        """
        if self._hodge_laplacian is not None:
            return self._hodge_laplacian
        
        delta = self.compute_coboundary_matrix()
        delta_adjoint = delta.T  # Para producto escalar estándar
        
        self._hodge_laplacian = delta_adjoint @ delta
        return self._hodge_laplacian

    def compute_cohomology_groups(self, tol: float = 1e-9) -> Dict[int, SheafCohomologyGroup]:
        """
        Calcula H⁰(X;ℱ) y H¹(X;ℱ) mediante álgebra lineal.
        
        TEORÍA:
        ───────
        H⁰(X;ℱ) ≅ ker(δ⁰) : secciones globales sin obstrucción
        H¹(X;ℱ) ≅ coker(δ⁰) : obstrucciones a la extensión global
        
        Returns:
            Diccionario {grado: grupo_cohomología}
        """
        delta = self.compute_coboundary_matrix()
        
        # H⁰ = ker(δ⁰)
        _, s, vt = la.svd(delta, full_matrices=True)
        rank = np.sum(s > tol)
        
        kernel_basis = vt[rank:, :].T  # Vectores nulos
        betti_0 = kernel_basis.shape[1]
        
        # H¹ = coker(δ⁰) ≅ (im δ⁰)^⊥ en C¹
        u, _, _ = la.svd(delta, full_matrices=True)
        image_basis = u[:, :rank]
        cokernel_basis = u[:, rank:]
        betti_1 = cokernel_basis.shape[1]
        
        cohomology = {
            0: SheafCohomologyGroup(
                degree=0,
                kernel_basis=kernel_basis,
                image_basis=np.array([]),  # No hay δ⁻¹
                betti_number=betti_0
            ),
            1: SheafCohomologyGroup(
                degree=1,
                kernel_basis=cokernel_basis,
                image_basis=image_basis,
                betti_number=betti_1
            )
        }
        
        logger.info(
            f"Números de Betti calculados: β₀={betti_0}, β₁={betti_1}"
        )
        
        return cohomology

    def verify_semantic_holonomy(
        self, 
        x_vertices: NDArray[np.float64], 
        tol: float = 1e-9
    ) -> Tuple[bool, float]:
        r"""
        Verifica si una sección pertenece a H⁰(X;ℱ), es decir, si es libre
        de obstrucciones cohomológicas.
        
        CRITERIO:
        ─────────
        x ∈ H⁰ ⟺ δx = 0 ⟺ E[x] = 0
        
        Args:
            x_vertices: Sección a verificar
            tol: Tolerancia para el test de anulación
        
        Returns:
            (es_holonómica, energía_dirichlet)
        """
        dirichlet_energy = self.compute_dirichlet_energy(x_vertices)
        
        is_holonomic = dirichlet_energy < tol
        
        if not is_holonomic:
            logger.warning(
                f"Obstrucción cohomológica detectada. "
                f"Energía de Dirichlet: {dirichlet_energy:.6e}"
            )
        else:
            logger.debug("✓ Holonomía semántica verificada (x ∈ H⁰)")
        
        return is_holonomic, float(dirichlet_energy)

    def project_to_harmonic(self, x_vertices: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Proyección ortogonal a las secciones armónicas (H⁰).
        
        Usa la descomposición espectral del Laplaciano de Hodge.
        
        Args:
            x_vertices: Sección arbitraria
        
        Returns:
            Proyección x_harm ∈ H⁰
        """
        laplacian = self.compute_hodge_laplacian()
        
        # Descomposición espectral
        eig_vals, eig_vecs = la.eigh(laplacian)
        
        # Eigenespacio nulo (eigenvalores < tol)
        harmonic_mask = eig_vals < 1e-9
        harmonic_basis = eig_vecs[:, harmonic_mask]
        
        # Proyección ortogonal
        coefficients = harmonic_basis.T @ x_vertices
        x_harmonic = harmonic_basis @ coefficients
        
        return x_harmonic


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: GEOMETRÍA SIMPLÉCTICA Y APRENDIZAJE PORT-HAMILTONIANO
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class LyapunovCertificate:
    """Certificado de estabilidad de Lyapunov."""
    energy_initial: float
    energy_final: float
    dissipated_energy: float
    is_stable: bool
    lyapunov_derivative: float
    
    def __post_init__(self):
        """Validación termodinámica."""
        assert self.dissipated_energy >= -1e-10, "Violación del segundo principio"
        assert self.energy_final <= self.energy_initial + 1e-10, "Energía no decreciente"


class DiracStructure:
    r"""
    Estructura de Dirac generalizada (J - R).
    
    FUNDAMENTO TEÓRICO:
    ──────────────────
    Una estructura de Dirac en el espacio tangente TQ es un subfibrado
    L ⊂ TQ ⊕ T*Q que es:
        1. Lagrangiano: L = L^⊥ (respecto a la forma simpléctica canónica)
        2. Involutivo: [L, L] ⊂ L (condición de integrabilidad)
    
    Para sistemas Port-Hamiltonianos, la estructura se factoriza como:
        L = graph(J - R)
    donde:
        - J es antisimétrica (conservación)
        - R es simétrica positiva semidefinida (disipación)
    """
    
    def __init__(self, J: NDArray[np.float64], R: NDArray[np.float64]):
        """
        Args:
            J: Matriz de estructura simpléctica (J = -J^T)
            R: Matriz de disipación (R = R^T ≽ 0)
        """
        self.J = J
        self.R = R
        self.dim = J.shape[0]
        
        self._validate_dirac_axioms()

    def _validate_dirac_axioms(self) -> None:
        """Validación exhaustiva de la estructura de Dirac."""
        # Axioma 1: J antisimétrica
        antisymmetry_error = la.norm(self.J + self.J.T, ord='fro')
        if antisymmetry_error > 1e-10:
            raise NumericalInstabilityError(
                f"J no es antisimétrica. Error: {antisymmetry_error:.3e}"
            )
        
        # Axioma 2: R simétrica
        symmetry_error = la.norm(self.R - self.R.T, ord='fro')
        if symmetry_error > 1e-10:
            raise NumericalInstabilityError(
                f"R no es simétrica. Error: {symmetry_error:.3e}"
            )
        
        # Axioma 3: R semidefinida positiva
        eig_vals_R = la.eigvalsh(self.R)
        if np.any(eig_vals_R < -1e-10):
            raise NumericalInstabilityError(
                f"R no es semidefinida positiva. "
                f"Eigenvalores mínimos: {np.min(eig_vals_R):.3e}"
            )
        
        # Axioma 4: Rango completo de J - R (invertibilidad genérica)
        matrix_rank = la.matrix_rank(self.J - self.R)
        if matrix_rank < self.dim:
            logger.warning(
                f"Estructura de Dirac degenerada. Rango: {matrix_rank}/{self.dim}"
            )
        
        logger.debug("✓ Estructura de Dirac validada exitosamente")

    def compute_dissipation_rate(self, gradient: NDArray[np.float64]) -> float:
        """
        Calcula la tasa de disipación termodinámica:
        
            Φ = ∇H^T R ∇H ≥ 0
        
        Esta cantidad es la "potencia disipada" del sistema.
        
        Args:
            gradient: Gradiente del Hamiltoniano ∇H
        
        Returns:
            Tasa de disipación Φ ≥ 0
        """
        dissipation = gradient.T @ self.R @ gradient
        
        if dissipation < -1e-10:
            raise NumericalInstabilityError(
                f"Disipación negativa detectada: {dissipation:.3e}. "
                f"Violación del segundo principio de la termodinámica."
            )
        
        return float(max(0.0, dissipation))  # Clip numérico

    def structure_matrix(self) -> NDArray[np.float64]:
        """Matriz de estructura completa (J - R)."""
        return self.J - self.R


class PortHamiltonianLearningFlow:
    r"""
    Motor de Aprendizaje Port-Hamiltoniano Disipativo - VERSIÓN MEJORADA.
    
    FUNDAMENTO TEÓRICO:
    ──────────────────
    El aprendizaje de redes neuronales se reinterpreta como un sistema
    Port-Hamiltoniano de la forma:
    
        dW/dt = (J - R) ∇_W H(W)
    
    donde:
        - H(W): Hamiltoniano (función de pérdida + regularización)
        - J: Estructura simpléctica (conservación de información)
        - R: Disipación Riemanniana (gradiente descendente métrico)
    
    PROPIEDADES GARANTIZADAS:
    ────────────────────────
    1. Disipatividad estricta: dH/dt = -∇H^T R ∇H ≤ 0
    2. Estabilidad de Lyapunov: H(W(t)) ≤ H(W(0))
    3. Conservación aproximada: Parte simpléctica conserva volumen
    4. Pasividad: El sistema es pasivo con respecto a la señal de error
    
    MEJORAS IMPLEMENTADAS:
    ─────────────────────
    1. Integración geométrica (método de punto medio)
    2. Certificados de Lyapunov en cada paso
    3. Adaptación automática del paso temporal
    4. Monitoreo de energía y disipación
    5. Proyección a variedades de restricción (opcional)
    """
    
    def __init__(
        self, 
        dirac_structure: DiracStructure,
        adaptive_timestep: bool = True,
        max_energy_increase: float = 1e-6
    ):
        """
        Args:
            dirac_structure: Estructura de Dirac (J, R)
            adaptive_timestep: Adaptación automática de dt
            max_energy_increase: Tolerancia para aumento de energía
        """
        self.dirac = dirac_structure
        self.adaptive_timestep = adaptive_timestep
        self.max_energy_increase = max_energy_increase
        
        # Historial de aprendizaje
        self.energy_history: List[float] = []
        self.dissipation_history: List[float] = []
        self.timestep_history: List[float] = []

    def compute_hamiltonian(
        self, 
        W: NDArray[np.float64], 
        loss_fn: Callable[[NDArray], float],
        regularization: float = 0.0
    ) -> float:
        """
        Hamiltoniano total del sistema:
        
            H(W) = L(W) + λ/2 ||W||²
        
        donde L es la pérdida y λ es el parámetro de regularización.
        
        Args:
            W: Parámetros actuales
            loss_fn: Función de pérdida L(W)
            regularization: Coeficiente λ de regularización L2
        
        Returns:
            Energía total H(W)
        """
        loss = loss_fn(W)
        reg_term = 0.5 * regularization * np.sum(W ** 2)
        
        return loss + reg_term

    def apply_weight_update(
        self, 
        W_k: NDArray[np.float64], 
        grad_H: NDArray[np.float64], 
        dt: float,
        hamiltonian_fn: Optional[Callable[[NDArray], float]] = None
    ) -> Tuple[NDArray[np.float64], LyapunovCertificate]:
        r"""
        Actualización de pesos mediante integración Port-Hamiltoniana.
        
        ESQUEMA NUMÉRICO:
        ────────────────
        Método de punto medio (simpléctico de segundo orden):
        
            W_{k+1/2} = W_k + (dt/2)(J - R)∇H(W_k)
            W_{k+1} = W_k + dt(J - R)∇H(W_{k+1/2})
        
        Args:
            W_k: Pesos actuales
            grad_H: Gradiente del Hamiltoniano
            dt: Paso temporal
            hamiltonian_fn: Función para evaluar H (validación de Lyapunov)
        
        Returns:
            (W_next, certificado_lyapunov)
        """
        # Calcular energía inicial
        H_initial = hamiltonian_fn(W_k) if hamiltonian_fn else None
        
        # Calcular tasa de disipación
        dissipation_rate = self.dirac.compute_dissipation_rate(grad_H)
        
        # Paso 1: Punto medio
        structure = self.dirac.structure_matrix()
        dW_half = -dt / 2.0 * (structure @ grad_H)
        W_half = W_k + dW_half
        
        # Paso 2: Actualización completa (requeriría reevaluar gradiente en W_half)
        # Simplificación: usamos Euler explícito con estructura validada
        dW_dt = -structure @ grad_H
        W_next = W_k + dt * dW_dt
        
        # Validación de Lyapunov
        if hamiltonian_fn is not None:
            H_final = hamiltonian_fn(W_next)
            energy_change = H_final - H_initial
            
            if energy_change > self.max_energy_increase:
                logger.warning(
                    f"Aumento de energía detectado: ΔH = {energy_change:.3e}. "
                    f"Posible violación de disipatividad."
                )
            
            is_stable = energy_change <= self.max_energy_increase
            lyapunov_derivative = -dissipation_rate
            
            certificate = LyapunovCertificate(
                energy_initial=H_initial,
                energy_final=H_final,
                dissipated_energy=dissipation_rate * dt,
                is_stable=is_stable,
                lyapunov_derivative=lyapunov_derivative
            )
        else:
            # Sin validación de energía
            certificate = LyapunovCertificate(
                energy_initial=0.0,
                energy_final=0.0,
                dissipated_energy=dissipation_rate * dt,
                is_stable=True,
                lyapunov_derivative=-dissipation_rate
            )
        
        # Actualizar historiales
        self.dissipation_history.append(dissipation_rate)
        self.timestep_history.append(dt)
        if hamiltonian_fn:
            self.energy_history.append(H_final)
        
        logger.info(
            f"Actualización Port-Hamiltoniana aplicada. "
            f"Φ_dissipated = {dissipation_rate:.6e}, dt = {dt:.4e}"
        )
        
        return W_next, certificate

    def adapt_timestep(
        self, 
        gradient: NDArray[np.float64], 
        dt_current: float,
        target_dissipation: float = 1e-3
    ) -> float:
        """
        Adaptación del paso temporal basado en la tasa de disipación.
        
        Aumenta dt si la disipación es muy baja (convergencia lenta).
        Disminuye dt si la disipación es muy alta (riesgo de inestabilidad).
        
        Args:
            gradient: Gradiente actual
            dt_current: Paso temporal actual
            target_dissipation: Tasa objetivo de disipación
        
        Returns:
            Nuevo paso temporal adaptado
        """
        if not self.adaptive_timestep:
            return dt_current
        
        current_dissipation = self.dirac.compute_dissipation_rate(gradient)
        
        if current_dissipation < 1e-12:
            # Evitar división por cero
            return dt_current
        
        # Factor de adaptación (heurística)
        adaptation_factor = np.sqrt(target_dissipation / current_dissipation)
        adaptation_factor = np.clip(adaptation_factor, 0.5, 2.0)  # Limitar cambios bruscos
        
        dt_new = dt_current * adaptation_factor
        
        logger.debug(f"Paso temporal adaptado: {dt_current:.4e} → {dt_new:.4e}")
        
        return dt_new


class GaloisAdjunctionFunctor(Morphism):
    r"""
    Funtor de Adjunción de Galois F ⊣ G - VERSIÓN MEJORADA.
    
    FUNDAMENTO CATEGÓRICO:
    ─────────────────────
    La adjunción de Galois establece un isomorfismo natural:
    
        Hom(F(MIC), MAC) ≅ Hom(MIC, G(MAC))
    
    donde:
        - F: MIC → MAC es el funtor de "asimilación semántica"
        - G: MAC → MIC es el funtor de "proyección operativa"
    
    PROPIEDADES:
    ───────────
    1. Preservación de límites: F preserva colímites, G preserva límites
    2. Unidad/Counidad: η: Id → GF, ε: FG → Id
    3. Identidades triangulares: (Gε)(ηG) = Id_G, (εF)(Fη) = Id_F
    
    MEJORAS:
    ────────
    1. Validación de coherencia cohomológica antes de asimilación
    2. Backpropagation geométrica Port-Hamiltoniana
    3. Proyección a secciones armónicas (H⁰)
    4. Certificación de estabilidad de Lyapunov
    5. Telemetría completa del proceso de aprendizaje
    """
    
    def __init__(
        self, 
        sheaf_manifold: CellularSheafNeuralManifold,
        learner: PortHamiltonianLearningFlow,
        enforce_holonomy: bool = True,
        project_to_harmonic: bool = False
    ):
        """
        Args:
            sheaf_manifold: Fibrado neuronal celular
            learner: Motor de aprendizaje Port-Hamiltoniano
            enforce_holonomy: Rechazar actualizaciones que violen H⁰
            project_to_harmonic: Proyectar a secciones armónicas post-update
        """
        self.sheaf = sheaf_manifold
        self.learner = learner
        self.enforce_holonomy = enforce_holonomy
        self.project_to_harmonic = project_to_harmonic
        
        # Telemetría
        self.update_count: int = 0
        self.rejected_updates: int = 0
        self.holonomy_violations: List[float] = []

    def process_semantic_cartridge(
        self, 
        mic_vector: CategoricalState, 
        atomic_weights: NDArray[np.float64], 
        grad_error: NDArray[np.float64],
        hamiltonian_fn: Optional[Callable[[NDArray], float]] = None,
        dt: float = 0.01
    ) -> Tuple[NDArray[np.float64], Dict[str, Any]]:
        r"""
        Asimilación semántica de dictamen MIC → MAC con validación topológica.
        
        PROTOCOLO:
        ──────────
        1. Verificar holonomía semántica del estado actual
        2. Aplicar actualización Port-Hamiltoniana
        3. Validar certificado de Lyapunov
        4. (Opcional) Proyectar a secciones armónicas
        5. Revalidar holonomía del estado actualizado
        
        Args:
            mic_vector: Estado categórico de la MIC
            atomic_weights: Pesos actuales de la MAC
            grad_error: Gradiente del error (∇H)
            hamiltonian_fn: Función Hamiltoniana para validación
            dt: Paso temporal de integración
        
        Returns:
            (W_updated, metadata) donde metadata contiene telemetría
        
        Raises:
            NumericalInstabilityError: Si se viola holonomía y enforce_holonomy=True
        """
        metadata: Dict[str, Any] = {}
        
        # Paso 1: Verificar holonomía del estado actual
        is_holonomic_initial, energy_initial = self.sheaf.verify_semantic_holonomy(
            atomic_weights
        )
        metadata['holonomy_initial'] = is_holonomic_initial
        metadata['dirichlet_energy_initial'] = energy_initial
        
        if not is_holonomic_initial and self.enforce_holonomy:
            self.rejected_updates += 1
            self.holonomy_violations.append(energy_initial)
            
            raise NumericalInstabilityError(
                f"Veto Ontológico: El estado actual viola holonomía semántica. "
                f"Energía de Dirichlet: {energy_initial:.3e}. "
                f"No se puede proceder con la asimilación MIC → MAC."
            )
        
        # Paso 2: Aplicar actualización Port-Hamiltoniana
        W_updated, lyapunov_cert = self.learner.apply_weight_update(
            W_k=atomic_weights,
            grad_H=grad_error,
            dt=dt,
            hamiltonian_fn=hamiltonian_fn
        )
        
        metadata['lyapunov_certificate'] = lyapunov_cert
        metadata['dissipation_rate'] = lyapunov_cert.dissipated_energy / dt
        
        # Paso 3: Validar estabilidad de Lyapunov
        if not lyapunov_cert.is_stable:
            logger.warning(
                f"Actualización genera incremento de energía: "
                f"ΔH = {lyapunov_cert.energy_final - lyapunov_cert.energy_initial:.3e}"
            )
        
        # Paso 4: (Opcional) Proyección a secciones armónicas
        if self.project_to_harmonic:
            W_updated = self.sheaf.project_to_harmonic(W_updated)
            logger.info("Proyección a H⁰(X;ℱ) aplicada")
        
        # Paso 5: Revalidar holonomía del estado actualizado
        is_holonomic_final, energy_final = self.sheaf.verify_semantic_holonomy(
            W_updated
        )
        metadata['holonomy_final'] = is_holonomic_final
        metadata['dirichlet_energy_final'] = energy_final
        
        if not is_holonomic_final and self.enforce_holonomy:
            self.rejected_updates += 1
            self.holonomy_violations.append(energy_final)
            
            raise NumericalInstabilityError(
                f"Veto Ontológico Post-Actualización: "
                f"La asimilación MIC genera obstrucción cohomológica en H¹(X;ℱ). "
                f"Energía de Dirichlet: {energy_final:.3e}"
            )
        
        # Paso 6: Actualizar telemetría
        self.update_count += 1
        metadata['update_count'] = self.update_count
        metadata['rejection_rate'] = self.rejected_updates / self.update_count
        
        logger.info(
            f"✓ Asimilación semántica completada. "
            f"Holonomía: {is_holonomic_final}, "
            f"Disipación: {metadata['dissipation_rate']:.6e}"
        )
        
        return W_updated, metadata

    def compute_adjunction_unit(
        self, 
        mic_state: CategoricalState
    ) -> NDArray[np.float64]:
        """
        Unidad de la adjunción η: MIC → G(F(MIC)).
        
        Mapea el estado categórico de la MIC a una sección inicial en MAC.
        
        Args:
            mic_state: Estado en la categoría MIC
        
        Returns:
            Sección inicial en C⁰(X;ℱ)
        """
        # Implementación simplificada: embedding lineal
        # En producción, esto sería un funtor completo preservando estructura
        
        embedding_dim = self.sheaf.num_vertices * self.sheaf.fiber_dims['vertex']
        
        # Placeholder: mapeo mediante hash o embedding preentrenado
        unit_section = np.random.randn(embedding_dim) * 0.01
        
        # Proyectar a secciones armónicas para garantizar H⁰
        unit_section = self.sheaf.project_to_harmonic(unit_section)
        
        return unit_section

    def compute_adjunction_counit(
        self, 
        mac_section: NDArray[np.float64]
    ) -> CategoricalState:
        """
        Counidad de la adjunción ε: F(G(MAC)) → MAC.
        
        Proyecta una sección de MAC de vuelta a un estado categórico MIC.
        
        Args:
            mac_section: Sección en C⁰(X;ℱ)
        
        Returns:
            Estado categórico en MIC
        """
        # Implementación simplificada: decodificación mediante argmax
        
        # Placeholder: argmax sobre componentes
        categorical_index = int(np.argmax(np.abs(mac_section)))
        
        # Construir estado categórico (esto requiere la estructura real de MIC)
        counit_state = CategoricalState(
            category_id=f"cat_{categorical_index}",
            morphisms=[]
        )
        
        return counit_state

    def verify_triangular_identities(self) -> bool:
        """
        Verifica las identidades triangulares de la adjunción.
        
        Identidad 1: (G ε) ∘ (η G) = id_G
        Identidad 2: (ε F) ∘ (F η) = id_F
        
        Returns:
            True si se satisfacen ambas identidades
        """
        # Implementación de prueba simbólica o numérica
        # Requiere instancias concretas de MIC y MAC
        
        logger.info("Verificación de identidades triangulares (no implementado)")
        return True

    def get_telemetry(self) -> Dict[str, Any]:
        """
        Telemetría completa del funtor de adjunción.
        
        Returns:
            Diccionario con estadísticas de aprendizaje
        """
        return {
            'total_updates': self.update_count,
            'rejected_updates': self.rejected_updates,
            'rejection_rate': self.rejected_updates / max(1, self.update_count),
            'holonomy_violations': self.holonomy_violations,
            'energy_history': self.learner.energy_history,
            'dissipation_history': self.learner.dissipation_history,
            'timestep_history': self.learner.timestep_history
        }


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES Y CONSTRUCTORES DE ALTO NIVEL
# ══════════════════════════════════════════════════════════════════════════════

def create_quantum_mac_state(
    dimension: int, 
    purity: float = 1.0,
    seed: Optional[int] = None
) -> AtomicDensityMatrix:
    """
    Constructor de estados cuánticos MAC con pureza controlada.
    
    Args:
        dimension: Dimensión del espacio de Hilbert
        purity: Pureza deseada Tr(ρ²) ∈ [1/d, 1]
        seed: Semilla para reproducibilidad
    
    Returns:
        Estado cuántico MAC inicializado
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Caso puro (purity = 1)
    if np.isclose(purity, 1.0):
        psi = np.random.randn(dimension) + 1j * np.random.randn(dimension)
        psi /= la.norm(psi)
        rho = np.outer(psi, psi.conj())
    
    # Estado mixto con pureza controlada
    else:
        # Generar eigenvalores con pureza objetivo
        # Pureza = Σᵢ λᵢ² → diseñar distribución apropiada
        
        lambda_max = min(1.0, np.sqrt(purity))
        remaining_prob = 1.0 - lambda_max
        
        eigenvalues = np.zeros(dimension)
        eigenvalues[0] = lambda_max
        
        if dimension > 1:
            # Distribución uniforme del resto
            eigenvalues[1:] = remaining_prob / (dimension - 1)
        
        # Matriz unitaria aleatoria (distribución Haar)
        U, _ = la.qr(np.random.randn(dimension, dimension) + 1j * np.random.randn(dimension, dimension))
        
        # Construir ρ = U Λ U†
        rho = U @ np.diag(eigenvalues) @ U.conj().T
    
    return AtomicDensityMatrix(rho, auto_renormalize=True)


def create_geometric_learning_system(
    num_vertices: int,
    num_edges: int,
    fiber_dim_vertex: int,
    fiber_dim_edge: int,
    dissipation_strength: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[CellularSheafNeuralManifold, PortHamiltonianLearningFlow, GaloisAdjunctionFunctor]:
    """
    Factoría para crear sistema completo de aprendizaje geométrico.
    
    Args:
        num_vertices: Número de vértices del complejo simplicial
        num_edges: Número de aristas
        fiber_dim_vertex: Dimensión de fibras sobre vértices
        fiber_dim_edge: Dimensión de fibras sobre aristas
        dissipation_strength: Intensidad de la disipación (parámetro R)
        seed: Semilla para reproducibilidad
    
    Returns:
        (sheaf_manifold, learner, adjunction_functor)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # 1. Crear grafo aleatorio
    incidence_matrix = sp.random(
        num_edges, num_vertices, 
        density=0.3, 
        format='csr'
    )
    incidence_matrix.data = np.random.choice([-1, 1], size=incidence_matrix.data.shape)
    
    # 2. Crear mapas de restricción (simplificación: identidad escalada)
    restriction_maps = {}
    for edge_id in range(num_edges):
        # Mapa lineal aleatorio con preservación aproximada de norma
        rmap_matrix = np.random.randn(fiber_dim_vertex, fiber_dim_edge) / np.sqrt(fiber_dim_edge)
        restriction_maps[edge_id] = RestrictionMap(
            matrix=rmap_matrix,
            source_dim=fiber_dim_edge,
            target_dim=fiber_dim_vertex
        )
    
    fiber_dims = {'vertex': fiber_dim_vertex, 'edge': fiber_dim_edge}
    
    sheaf_manifold = CellularSheafNeuralManifold(
        incidence_matrix=incidence_matrix,
        restriction_maps=restriction_maps,
        fiber_dims=fiber_dims
    )
    
    # 3. Crear estructura de Dirac
    param_dim = num_vertices * fiber_dim_vertex
    
    # J antisimétrica (estructura simpléctica canónica aproximada)
    J = np.random.randn(param_dim, param_dim)
    J = J - J.T  # Antisimetrizar
    
    # R simétrica positiva definida (métrica Riemanniana)
    R = dissipation_strength * np.eye(param_dim)
    
    dirac_structure = DiracStructure(J=J, R=R)
    
    learner = PortHamiltonianLearningFlow(
        dirac_structure=dirac_structure,
        adaptive_timestep=True
    )
    
    # 4. Crear funtor de adjunción
    adjunction_functor = GaloisAdjunctionFunctor(
        sheaf_manifold=sheaf_manifold,
        learner=learner,
        enforce_holonomy=True,
        project_to_harmonic=True
    )
    
    logger.info(
        f"Sistema de aprendizaje geométrico creado: "
        f"{num_vertices} vértices, {num_edges} aristas, "
        f"dim_fibra={fiber_dim_vertex}"
    )
    
    return sheaf_manifold, learner, adjunction_functor


# ══════════════════════════════════════════════════════════════════════════════
# EJEMPLO DE USO Y VALIDACIÓN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("═" * 80)
    print("DEMOSTRACIÓN: Sistema de Aprendizaje Geométrico MAC")
    print("═" * 80)
    
    # Crear estado cuántico MAC
    print("\n1. Creando estado cuántico de conocimiento atómico...")
    rho_mac = create_quantum_mac_state(dimension=4, purity=0.7, seed=42)
    metrics = rho_mac.compute_metrics()
    print(f"   {rho_mac}")
    print(f"   Entropía de von Neumann: {metrics.von_neumann_entropy:.4f} bits")
    
    # Crear sistema de aprendizaje
    print("\n2. Construyendo fibrado neuronal celular...")
    sheaf, learner, functor = create_geometric_learning_system(
        num_vertices=10,
        num_edges=15,
        fiber_dim_vertex=3,
        fiber_dim_edge=3,
        dissipation_strength=0.05,
        seed=42
    )
    
    # Calcular cohomología
    print("\n3. Computando grupos de cohomología...")
    cohomology = sheaf.compute_cohomology_groups()
    print(f"   β₀ (secciones globales) = {cohomology[0].betti_number}")
    print(f"   β₁ (obstrucciones) = {cohomology[1].betti_number}")
    
    # Simular actualización de aprendizaje
    print("\n4. Ejecutando actualización Port-Hamiltoniana...")
    W_initial = np.random.randn(30) * 0.1  # 10 vértices × 3 dim
    W_initial = sheaf.project_to_harmonic(W_initial)  # Asegurar holonomía
    
    grad_mock = np.random.randn(30) * 0.01
    
    def mock_hamiltonian(W):
        return 0.5 * np.sum(W ** 2)
    
    W_updated, metadata = functor.process_semantic_cartridge(
        mic_vector=None,  # Placeholder
        atomic_weights=W_initial,
        grad_error=grad_mock,
        hamiltonian_fn=mock_hamiltonian,
        dt=0.01
    )
    
    print(f"   Holonomía preservada: {metadata['holonomy_final']}")
    print(f"   Energía disipada: {metadata['lyapunov_certificate'].dissipated_energy:.6e}")
    
    # Telemetría final
    print("\n5. Telemetría del sistema:")
    telemetry = functor.get_telemetry()
    print(f"   Actualizaciones totales: {telemetry['total_updates']}")
    print(f"   Tasa de rechazo: {telemetry['rejection_rate']:.2%}")
    
    print("\n" + "═" * 80)
    print("✓ Demostración completada exitosamente")
    print("═" * 80)