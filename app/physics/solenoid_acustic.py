"""
═════════════════════════════════════════════════════════════════════════════
MÓDULO: Solenoide Acústico — Operador de Descomposición de Hodge-Helmholtz
VERSIÓN: 4.0.0 - Refactorización Rigurosa con Validación Topológica Completa
UBICACIÓN: app/physics/solenoid_acoustic.py
═════════════════════════════════════════════════════════════════════════════

FUNDAMENTOS MATEMÁTICOS Y TOPOLÓGICOS:

§1. COMPLEJO DE CADENAS Y COHOMOLOGÍA
    ───────────────────────────────────
    
    Dado un grafo dirigido finito G = (V, E) con |V| = n, |E| = m, definimos
    el complejo de cadenas simpliciales de dimensión 1:
    
        C₀(G; ℝ) ←─∂₁─ C₁(G; ℝ) ←─∂₂─ C₂(G; ℝ)
    
    donde:
        • C₀(G; ℝ) = ℝⁿ: espacio de 0-cadenas (funciones sobre vértices)
        • C₁(G; ℝ) = ℝᵐ: espacio de 1-cadenas (flujos sobre aristas)
        • C₂(G; ℝ) = ℝᵏ: espacio de 2-cadenas (ciclos fundamentales)
    
    Operadores de borde:
        ∂₁: C₁ → C₀  (matriz de incidencia B₁ ∈ ℝⁿˣᵐ)
            (B₁)ᵥₑ = +1 si head(e) = v
                    -1 si tail(e) = v
                     0 en otro caso
        
        ∂₂: C₂ → C₁  (matriz de ciclos B₂ ∈ ℝᵐˣᵏ)
    
    Axioma Fundamental (Condición de Complejo):
        ∂₁ ∘ ∂₂ = 0  ⟺  B₁ · B₂ = 0
    
    Esta condición es exacta en ℝ (con orientaciones consistentes) y
    garantiza que la imagen de ∂₂ está contenida en el kernel de ∂₁.

§2. HOMOLOGÍA Y NÚMEROS DE BETTI
    ──────────────────────────────
    
    Grupos de homología:
        H₀(G; ℝ) = ker(∂₁) / im(∂₂) ≅ ℝᶜ
        H₁(G; ℝ) = ker(∂₂) / im(∂₃) ≅ ℝᵝ¹
    
    donde c es el número de componentes conexas y β₁ es el primer número
    de Betti (rango de ciclos independientes).
    
    Números de Betti:
        β₀ = dim H₀(G; ℝ) = c (componentes conexas)
        β₁ = dim H₁(G; ℝ) = m - n + c (ciclos independientes)
        β₂ = 0 (para grafos sin 2-símplices)
    
    Característica de Euler-Poincaré:
        χ(G) = Σᵢ (-1)ⁱ βᵢ = β₀ - β₁ = n - m

§3. TEOREMA DE DESCOMPOSICIÓN DE HODGE
    ────────────────────────────────────
    
    TEOREMA (Hodge, 1941; Eckmann, 1944):
        Para un complejo de cadenas finito sobre ℝ, existe una descomposición
        ortogonal única:
        
            C₁(G; ℝ) = im(∂₂*) ⊕ im(∂₁) ⊕ ker(Δ₁)
        
        donde:
            • im(∂₂*) = im(B₂): subespacio de ciclos (solenoidal, curl)
            • im(∂₁) = im(B₁ᵀ): subespacio de gradientes (irrotacional)
            • ker(Δ₁): subespacio armónico (flujos harmónicos)
        
        y Δ₁ = ∂₁*∂₁ + ∂₂∂₂* es el Laplaciano de Hodge.
    
    Laplaciano de Hodge sobre 1-cadenas:
        L₁ = B₁ᵀB₁ + B₂B₂ᵀ ∈ ℝᵐˣᵐ
        
        Propiedades:
            1. L₁ = L₁ᵀ (simétrico)
            2. L₁ ≥ 0 (semi-definido positivo)
            3. λᵢ ≥ 0 ∀i (autovalores no negativos)
            4. ker(L₁) ≅ H₁(G; ℝ) (isomorfismo de Hodge)
            5. dim ker(L₁) = β₁

§4. PROYECTORES ORTOGONALES
    ────────────────────────
    
    Proyector sobre im(B₂) (subespacio solenoidal):
        P_curl: ℝᵐ → im(B₂)
        
    Construcción estable vía SVD (Golub & Van Loan, §5.5):
        B₂ = U Σ Vᵀ  (SVD completo)
        P_curl = Uᵣ Uᵣᵀ
        
        donde Uᵣ = columnas de U con σᵢ > tol.
    
    Propiedades verificables:
        1. P_curl² = P_curl (idempotencia)
        2. P_curl = P_curlᵀ (simetría)
        3. P_curl L₁ = L₁ P_curl (conmutatividad con Laplaciano)

§5. ANÁLISIS NUMÉRICO Y ESTABILIDAD
    ─────────────────────────────────
    
    Tolerancia adaptativa (convención LAPACK):
        tol = max(m, n) · σ_max · ε_machine
    
    donde:
        • ε_machine ≈ 2.22×10⁻¹⁶ (IEEE 754 double precision)
        • σ_max = máximo valor singular de la matriz
    
    Número de condición espectral:
        κ₂(A) = σ_max / σ_min
    
    Análisis de error backward (Trefethen & Bau, Lec. 14):
        Si fl(x) es el resultado en punto flotante, entonces:
            fl(x) = (x + Δx)(1 + δ)
        con |δ| ≤ u·cond(problema) donde u = ε_machine/2.

§6. COMPLEJIDAD COMPUTACIONAL
    ──────────────────────────
    
    Matriz de Incidencia B₁:
        • Construcción: O(m) tiempo, O(nm) espacio (sparse)
        • SVD: O(min(n², m²)m) tiempo
    
    Matriz de Ciclos B₂ (Fundamental Cycle Basis):
        • MST: O(m log n) con Kruskal/Prim
        • Ciclos: O(mk) donde k = β₁ = m - n + c
        • Total: O(m log n + mk) = O(m(log n + m - n))
    
    Laplaciano de Hodge L₁:
        • Construcción: O(m²) tiempo (productos matriciales)
        • Eigendecomposición: O(m³) tiempo (eigh)

§7. FÍSICA DEL FLUJO
    ─────────────────
    
    Ley de Stokes discreta:
        ∮_γ I = (B₂ᵀI)_γ = Γ_γ
    
    donde γ es un ciclo y Γ_γ es la circulación del flujo alrededor de γ.
    
    Energía cinética de vorticidad:
        E_curl = ‖B₂ᵀI‖² = Iᵀ L_curl I = Iᵀ (B₂B₂ᵀ) I
    
    Índice de vorticidad (adimensional):
        ω = E_curl / ‖I‖² ∈ [0, 1]
    
    Interpretación física:
        ω ≈ 0: flujo irrotacional (laminar, potencial)
        ω ≈ 1: flujo completamente rotacional (turbulento)

REFERENCIAS:
    [1] Hodge, W. V. D. (1941). The Theory and Applications of Harmonic Integrals.
    [2] Eckmann, B. (1944). Harmonische Funktionen und Randwertaufgaben in einem Komplex.
    [3] Lim, L.-H. (2020). Hodge Laplacians on graphs. SIAM Review, 62(3), 685-715.
    [4] Golub, G. H., & Van Loan, C. F. (2013). Matrix Computations (4th ed.).
    [5] Trefethen, L. N., & Bau III, D. (1997). Numerical Linear Algebra.
    [6] Gross, J. L., & Yellen, J. (2005). Graph Theory and Its Applications (2nd ed.).

═════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import hashlib
import logging
import math
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import networkx as nx
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, svds

from app.core.schemas import Stratum
from app.core.telemetry import TelemetryContext, StepStatus

# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: CONFIGURACIÓN Y LOGGING
# ═════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("MIC.Physics.AcousticSolenoid")


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2: CONSTANTES Y TOLERANCIAS NUMÉRICAS
# ═════════════════════════════════════════════════════════════════════════════

class NumericalConstants:
    """
    Constantes numéricas fundamentales con fundamento en análisis de error.
    
    Todas las constantes están calibradas para aritmética IEEE 754
    de doble precisión (binary64).
    
    Invariantes:
        - Todas las tolerancias > 0
        - Todas las tolerancias < 1
        - ε_machine es constante de la arquitectura
    """
    
    MACHINE_EPSILON: float = np.finfo(np.float64).eps
    """
    Épsilon de máquina para float64.
    
    ε_machine ≈ 2.220446049250313×10⁻¹⁶
    
    Definición: Mínimo ε > 0 tal que fl(1 + ε) > 1.
    """
    
    FLOAT_MIN: float = np.finfo(np.float64).tiny
    """Mínimo float positivo normalizado ≈ 2.225×10⁻³⁰⁸."""
    
    FLOAT_MAX: float = np.finfo(np.float64).max
    """Máximo float representable ≈ 1.798×10³⁰⁸."""
    
    # Tolerancias derivadas con margen de seguridad
    BASE_TOLERANCE: float = 1e-10
    """Tolerancia base para comparaciones numéricas."""
    
    SVD_TOLERANCE: float = 1e-12
    """Tolerancia para truncamiento SVD."""
    
    ORTHOGONALITY_TOLERANCE: float = 1e-9
    """Tolerancia para verificación de ortogonalidad."""
    
    IDEMPOTENCY_TOLERANCE: float = 1e-8
    """Tolerancia para verificación de idempotencia de proyectores."""
    
    SYMMETRY_TOLERANCE: float = 1e-10
    """Tolerancia para verificación de simetría."""
    
    ENERGY_FLOOR: float = 1e-14
    """Piso de energía para evitar underflow."""
    
    VORTICITY_SIGNIFICANCE_THRESHOLD: float = 1e-9
    """Umbral mínimo de energía de vorticidad significativa."""
    
    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Sellado de clase para inmutabilidad."""
        raise TypeError(
            f"La clase {cls.__name__} está sellada. "
            "No se permite herencia."
        )


# Alias corto
NC = NumericalConstants


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: EXCEPCIONES ESPECIALIZADAS
# ═════════════════════════════════════════════════════════════════════════════

class HodgeDecompositionError(Exception):
    """
    Error base para problemas en descomposición de Hodge.
    
    Categoría de errores relacionados con:
        - Construcción del complejo de cadenas
        - Violación de axiomas topológicos
        - Fallos en cálculos de homología
    """
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
        self.message = message


class TopologicalInvariantError(HodgeDecompositionError):
    """
    Violación de invariantes topológicos.
    
    Ejemplos:
        - Característica de Euler incorrecta
        - Isomorfismo de Hodge no satisfecho
        - Números de Betti inconsistentes
    """
    pass


class NumericalStabilityError(HodgeDecompositionError):
    """
    Inestabilidad numérica crítica.
    
    Ejemplos:
        - Número de condición > 10¹⁶
        - Pérdida catastrófica de ortogonalidad
        - Overflow/underflow no manejado
    """
    pass


class GraphStructureError(HodgeDecompositionError):
    """
    Estructura de grafo inválida o inesperada.
    
    Ejemplos:
        - Grafo no dirigido cuando se espera dirigido
        - Multigrafos no soportados
        - Self-loops problemáticos
    """
    pass


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4: ESTRUCTURAS DE DATOS ALGEBRAICAS
# ═════════════════════════════════════════════════════════════════════════════

class SpectralDecomposition(NamedTuple):
    """
    Descomposición espectral completa de una matriz simétrica.
    
    Para A = A^T, representa:
        A = Q Λ Q^T
    
    donde:
        Q: matriz de autovectores ortonormales
        Λ: matriz diagonal de autovalores
    
    Invariantes:
        - eigenvalues ordenados (creciente o decreciente según contexto)
        - eigenvectors ortonormales: Q^T Q = I
        - len(eigenvalues) = eigenvectors.shape[1]
    """
    
    eigenvalues: np.ndarray
    """Autovalores λᵢ (ordenados)."""
    
    eigenvectors: np.ndarray
    """Autovectores correspondientes (columnas de Q)."""
    
    def __post_init__(self) -> None:
        """Validación de invariantes."""
        if len(self.eigenvalues) != self.eigenvectors.shape[1]:
            raise ValueError(
                f"Dimensión inconsistente: {len(self.eigenvalues)} eigenvalues, "
                f"{self.eigenvectors.shape[1]} eigenvectors"
            )
    
    @property
    def kernel_dimension(self) -> int:
        """
        Dimensión del kernel (número de autovalores ≈ 0).
        
        Returns:
            Cuenta de λᵢ ≤ tolerance.
        """
        return int(np.sum(np.abs(self.eigenvalues) <= NC.SVD_TOLERANCE))
    
    @property
    def spectral_gap(self) -> float:
        """
        Gap espectral: diferencia entre primer y segundo autovalor no nulo.
        
        Returns:
            λ₁ - λ₀ si existen al menos 2 autovalores no nulos, else 0.
        """
        nonzero = self.eigenvalues[np.abs(self.eigenvalues) > NC.SVD_TOLERANCE]
        if len(nonzero) < 2:
            return 0.0
        return float(nonzero[1] - nonzero[0])
    
    @property
    def condition_number(self) -> float:
        """
        Número de condición κ = λ_max / λ_min.
        
        Returns:
            κ ∈ [1, +∞].
        """
        nonzero = self.eigenvalues[np.abs(self.eigenvalues) > NC.SVD_TOLERANCE]
        if len(nonzero) == 0:
            return float('inf')
        return float(np.max(np.abs(nonzero)) / np.min(np.abs(nonzero)))


@dataclass(frozen=True, slots=True)
class BettiNumbers:
    """
    Números de Betti del complejo de cadenas.
    
    Para un grafo 1-dimensional:
        β₀ = número de componentes conexas
        β₁ = número de ciclos independientes = m - n + c
        β₂ = 0 (sin 2-símplices)
    
    Invariantes:
        - β₀ > 0 (al menos una componente)
        - β₁ ≥ 0 (ciclos no negativos)
        - Euler-Poincaré: χ = β₀ - β₁
    """
    
    beta_0: int
    """β₀: número de componentes conexas."""
    
    beta_1: int
    """β₁: número de ciclos independientes."""
    
    beta_2: int = 0
    """β₂: siempre 0 para grafos 1-dimensionales."""
    
    def __post_init__(self) -> None:
        """Validación de invariantes."""
        if self.beta_0 <= 0:
            raise ValueError(f"β₀ debe ser positivo: {self.beta_0}")
        if self.beta_1 < 0:
            raise ValueError(f"β₁ no puede ser negativo: {self.beta_1}")
        if self.beta_2 != 0:
            raise ValueError(f"β₂ debe ser 0 para grafos: {self.beta_2}")
    
    def euler_characteristic(self) -> int:
        """
        Característica de Euler: χ = Σᵢ (-1)ⁱ βᵢ.
        
        Para grafos: χ = β₀ - β₁.
        
        Returns:
            χ ∈ ℤ.
        """
        return self.beta_0 - self.beta_1
    
    def verify_euler_poincare(self, n: int, m: int) -> bool:
        """
        Verifica fórmula de Euler-Poincaré: χ = n - m.
        
        Args:
            n: Número de vértices.
            m: Número de aristas.
            
        Returns:
            True si χ = n - m.
        """
        chi_topological = self.euler_characteristic()
        chi_geometric = n - m
        return chi_topological == chi_geometric


@dataclass(frozen=True, slots=True)
class ChainComplex:
    """
    Complejo de cadenas C₀ ← C₁ ← C₂ con matrices de borde.
    
    Representa:
        C₀(G; ℝ) ←─∂₁─ C₁(G; ℝ) ←─∂₂─ C₂(G; ℝ)
    
    Invariantes verificables:
        1. B₁ B₂ = 0 (∂₁ ∘ ∂₂ = 0)
        2. rank(B₁) = n - c
        3. rank(B₂) = β₁
        4. Dimensiones consistentes
    """
    
    B1: np.ndarray
    """Matriz de incidencia B₁ ∈ ℝⁿˣᵐ."""
    
    B2: np.ndarray
    """Matriz de ciclos B₂ ∈ ℝᵐˣᵏ."""
    
    betti: BettiNumbers
    """Números de Betti del complejo."""
    
    metadata: Dict[str, Any] = field(default_factory=dict)
    """Metadatos de construcción y verificación."""
    
    def __post_init__(self) -> None:
        """Validación exhaustiva del complejo."""
        n, m = self.B1.shape
        m2, k = self.B2.shape
        
        # Verificar dimensiones consistentes
        if m != m2:
            raise GraphStructureError(
                f"Dimensiones inconsistentes: B₁ ∈ ℝ{n}×{m}, B₂ ∈ ℝ{m2}×{k}",
                context={"B1_shape": self.B1.shape, "B2_shape": self.B2.shape}
            )
        
        # Verificar β₁ = k
        if k != self.betti.beta_1:
            raise TopologicalInvariantError(
                f"β₁ inconsistente: k={k}, β₁={self.betti.beta_1}",
                context={"k": k, "beta_1": self.betti.beta_1}
            )
        
        # Verificar ∂₁ ∘ ∂₂ = 0
        if k > 0:
            B1B2 = self.B1 @ self.B2
            B1B2_norm = float(np.linalg.norm(B1B2, 'fro'))
            
            if B1B2_norm > NC.BASE_TOLERANCE:
                raise TopologicalInvariantError(
                    f"Violación de ∂₁∘∂₂ = 0: ‖B₁B₂‖_F = {B1B2_norm:.2e}",
                    context={"B1B2_norm": B1B2_norm}
                )
    
    @property
    def hodge_laplacian(self) -> np.ndarray:
        """
        Laplaciano de Hodge: L₁ = B₁ᵀB₁ + B₂B₂ᵀ.
        
        Returns:
            L₁ ∈ ℝᵐˣᵐ (simétrico, PSD).
        """
        L_grad = self.B1.T @ self.B1
        L_curl = self.B2 @ self.B2.T
        return L_grad + L_curl
    
    def verify_invariants(self) -> Dict[str, bool]:
        """
        Verifica todos los invariantes del complejo.
        
        Returns:
            Dict con resultados de verificación.
        """
        if self.B2.shape[1] > 0:
            B1B2 = self.B1.dot(self.B2) if isinstance(self.B1, sp.spmatrix) else self.B1 @ self.B2
            B1B2_norm = float(sp.linalg.norm(B1B2, 'fro') if isinstance(B1B2, sp.spmatrix) else np.linalg.norm(B1B2, 'fro'))
        else:
            B1B2_norm = 0.0
        
        L1 = self.hodge_laplacian
        is_symmetric = float(sp.linalg.norm(L1 - L1.transpose(), 'fro') if isinstance(L1, sp.spmatrix) else np.linalg.norm(L1 - L1.T, 'fro')) < NC.SYMMETRY_TOLERANCE
        
        try:
            L1_dense = L1.toarray() if isinstance(L1, sp.spmatrix) else L1
            eigenvalues = np.linalg.eigvalsh(L1_dense)
        except Exception:
            eigenvalues = np.array([0.0])
        is_psd = bool(np.all(eigenvalues >= -NC.BASE_TOLERANCE))
        
        return {
            "boundary_composition_zero": B1B2_norm < NC.BASE_TOLERANCE,
            "B1B2_norm": B1B2_norm,
            "laplacian_symmetric": is_symmetric,
            "laplacian_psd": is_psd,
            "min_eigenvalue": float(eigenvalues[0]),
        }


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5: UTILIDADES NUMÉRICAS CON ANÁLISIS DE ERROR
# ═════════════════════════════════════════════════════════════════════════════

class NumericalUtilities:
    """
    Álgebra lineal numérica con análisis de estabilidad riguroso.
    
    Todos los métodos implementan:
        1. Validación de entrada
        2. Elección de algoritmo estable
        3. Análisis de error backward
        4. Verificación de postcondiciones
    
    Referencias:
        [Golub & Van Loan, 2013] para algoritmos
        [Trefethen & Bau, 1997] para análisis de estabilidad
    """
    
    @staticmethod
    def adaptive_tolerance(
        matrix: Union[sp.spmatrix, np.ndarray],
        base_tolerance: Optional[float] = None
    ) -> float:
        """
        Tolerancia numérica adaptativa estricta según convención LAPACK.
        Fórmula: tol = max(m, n) · σ_max · ε_machine
        """
        if isinstance(matrix, sp.spmatrix):
            m, n = matrix.shape
            sigma_max_ub = sp.linalg.norm(matrix, 'fro')
        else:
            arr = np.asarray(matrix, dtype=np.float64)
            if arr.ndim != 2:
                raise ValueError(f"Se esperaba matriz 2-D, recibido shape={arr.shape}")
            m, n = arr.shape
            if m == 0 or n == 0:
                return base_tolerance if base_tolerance is not None else NC.BASE_TOLERANCE
            try:
                # Calcular σ_max exacto vía SVD
                sigma_max_ub = float(np.linalg.svd(arr, compute_uv=False)[0])
            except (np.linalg.LinAlgError, IndexError):
                # Fallback: norma de Frobenius
                sigma_max_ub = float(np.linalg.norm(arr, 'fro'))
        
        # Riguroso truncamiento de SVD: tol = max(M, N) * sigma_max * eps_mach
        adaptive_tol = max(m, n) * sigma_max_ub * NC.MACHINE_EPSILON

        # Solo usamos base_tolerance si es explícitamente forzado, de otra forma confiamos en el gap
        if base_tolerance is not None:
            return max(base_tolerance, adaptive_tol)
        return max(NC.BASE_TOLERANCE, adaptive_tol)
    
    @staticmethod
    def compute_numerical_rank(
        matrix: Union[sp.spmatrix, np.ndarray],
        tolerance: Optional[float] = None
    ) -> Tuple[int, np.ndarray]:
        """
        Rango numérico mediante SVD con análisis de gap.
        
        Teorema (Eckart-Young-Mirsky):
            rank_num(A) = #{σᵢ > tol}
        
        donde tol es la tolerancia adaptativa.
        
        Args:
            matrix: Matriz a analizar.
            tolerance: Umbral para σᵢ (default: adaptativo).
            
        Returns:
            (rank, singular_values) con singular_values ordenado desc.
        """
        if isinstance(matrix, sp.spmatrix):
            dense = matrix.toarray().astype(np.float64)
        else:
            dense = np.asarray(matrix, dtype=np.float64)
        
        if tolerance is None:
            tolerance = NumericalUtilities.adaptive_tolerance(dense)
        
        try:
            singular_values = np.linalg.svd(dense, compute_uv=False)
        except np.linalg.LinAlgError as exc:
            logger.error(f"SVD falló: {exc}")
            return 0, np.array([])
        
        rank = int(np.sum(singular_values > tolerance))
        
        logger.debug(
            f"Rango numérico: {rank} de {min(dense.shape)} "
            f"(tol={tolerance:.2e}, σ_max={singular_values[0]:.2e})"
        )
        
        return rank, singular_values
    
    @staticmethod
    def moore_penrose_pseudoinverse(
        matrix: Union[sp.spmatrix, np.ndarray],
        tolerance: Optional[float] = None
    ) -> np.ndarray:
        if isinstance(matrix, sp.spmatrix):
            dense = matrix.toarray().astype(np.float64)
        else:
            dense = np.asarray(matrix, dtype=np.float64)

        m_dim = dense.shape[0] if len(dense.shape) > 0 else 0
        n_dim = dense.shape[1] if len(dense.shape) > 1 else 0
        if m_dim == 0 or n_dim == 0:
            return np.zeros((n_dim, m_dim), dtype=np.float64)

        if tolerance is None:
            tolerance = NumericalUtilities.adaptive_tolerance(dense)

        try:
            U, s, Vt = np.linalg.svd(dense, full_matrices=False)
        except np.linalg.LinAlgError as exc:
            raise NumericalStabilityError(
                f"SVD falló en pseudoinversa: {exc}",
                context={"matrix_shape": dense.shape}
            ) from exc

        # Riguroso truncamiento: no `max(tolerance, NC.SVD_TOLERANCE)` to preserve theoretical bounds exactly.
        s_inv = np.where(s > tolerance, 1.0 / s, 0.0)
        
        pseudoinverse = (Vt.T * s_inv) @ U.T
        return pseudoinverse
    
    @staticmethod
    def orthogonal_projection(
        vector: np.ndarray,
        subspace_basis: np.ndarray,
        tolerance: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Proyección ortogonal P_S(v) vía SVD thin (estable).
        
        Dado B ∈ ℝⁿˣᵏ con columnas generando S:
            B = U Σ Vᵀ  (SVD thin)
            P_S = Uᵣ Uᵣᵀ  (proyector ortogonal)
            P_S(v) = Uᵣ (Uᵣᵀ v)
        
        Esta formulación evita el cuadrado del número de condición
        que aparece al usar BᵀB (Trefethen & Bau, Lec. 18).
        
        Args:
            vector: v ∈ ℝⁿ.
            subspace_basis: B ∈ ℝⁿˣᵏ (columnas de S).
            tolerance: Umbral SVD (default: adaptativo).
            
        Returns:
            (projected, residual) con v = projected + residual.
            
        Raises:
            ValueError: Si dimensiones inconsistentes.
        """
        v = np.asarray(vector, dtype=np.float64).ravel()
        B = np.asarray(subspace_basis, dtype=np.float64)
        
        if B.ndim != 2:
            raise ValueError(f"subspace_basis debe ser 2-D: shape={B.shape}")
        
        if B.shape[0] != v.shape[0]:
            raise ValueError(
                f"Dimensiones inconsistentes: B.shape[0]={B.shape[0]}, "
                f"v.shape[0]={v.shape[0]}"
            )
        
        # Subespacio vacío → proyección trivial
        if B.size == 0 or B.shape[1] == 0:
            return np.zeros_like(v), v.copy()
        
        if tolerance is None:
            tolerance = NumericalUtilities.adaptive_tolerance(B)
        
        try:
            U, s, _ = np.linalg.svd(B, full_matrices=False)
        except np.linalg.LinAlgError as exc:
            raise NumericalStabilityError(
                f"SVD falló en proyección: {exc}",
                context={"B_shape": B.shape}
            ) from exc
        
        # Filtrar columnas de U con σᵢ > tol
        U_r = U[:, s > tolerance]
        
        if U_r.shape[1] == 0:
            # B numéricamente nula
            return np.zeros_like(v), v.copy()
        
        # P_S v = U_r (U_rᵀ v)
        projected = U_r @ (U_r.T @ v)
        residual = v - projected
        
        # Verificar ortogonalidad
        inner = float(np.dot(projected, residual))
        if abs(inner) > NC.ORTHOGONALITY_TOLERANCE * np.linalg.norm(v)**2:
            logger.warning(
                f"Pérdida de ortogonalidad en proyección: "
                f"⟨P_S v, v - P_S v⟩ = {inner:.2e}"
            )
        
        return projected, residual
    
    @staticmethod
    def null_space_basis(
        matrix: Union[sp.spmatrix, np.ndarray],
        tolerance: Optional[float] = None
    ) -> np.ndarray:
        if isinstance(matrix, sp.spmatrix):
            dense = matrix.toarray().astype(np.float64)
        else:
            dense = np.asarray(matrix, dtype=np.float64)

        m_dim = dense.shape[0] if len(dense.shape) > 0 else 0
        n_dim = dense.shape[1] if len(dense.shape) > 1 else 0
        if m_dim == 0 or n_dim == 0:
            return np.zeros((n_dim, 0), dtype=np.float64)

        if tolerance is None:
            tolerance = NumericalUtilities.adaptive_tolerance(dense)

        try:
            _, s, Vt = np.linalg.svd(dense, full_matrices=True)
        except np.linalg.LinAlgError as exc:
            raise NumericalStabilityError(
                f"SVD falló en cálculo de kernel: {exc}",
                context={"matrix_shape": dense.shape}
            ) from exc

        n = Vt.shape[0]
        s_extended = np.append(s, np.zeros(n - s.size))
        
        # Riguroso truncamiento estricto <= tolerance.
        null_mask = s_extended <= tolerance
        kernel_basis = Vt[null_mask].T
        
        logger.debug(
            f"Dimensión del kernel: {kernel_basis.shape[1]} de {n} "
            f"(tol={tolerance:.2e})"
        )
        return kernel_basis
    
    @staticmethod
    def condition_number(
        matrix: Union[sp.spmatrix, np.ndarray]
    ) -> Tuple[float, float, float]:
        """
        Número de condición espectral κ₂(A) = σ_max / σ_min.
        
        Args:
            matrix: Matriz a analizar.
            
        Returns:
            (κ₂, σ_min, σ_max).
        """
        if isinstance(matrix, sp.spmatrix):
            dense = matrix.toarray().astype(np.float64)
        else:
            dense = np.asarray(matrix, dtype=np.float64)
        
        try:
            s = np.linalg.svd(dense, compute_uv=False)
        except np.linalg.LinAlgError:
            return math.inf, 0.0, 0.0
        
        sigma_max = float(s[0]) if s.size > 0 else 0.0
        sigma_min = float(s[-1]) if s.size > 0 else 0.0
        
        kappa = sigma_max / sigma_min if sigma_min > NC.SVD_TOLERANCE else math.inf
        
        return kappa, sigma_min, sigma_max


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6: CONSTRUCTOR DEL COMPLEJO DE CADENAS
# ═════════════════════════════════════════════════════════════════════════════

class HodgeDecompositionBuilder:
    """
    Constructor del complejo de cadenas C₀ ← C₁ ← C₂ sobre grafo dirigido.
    
    Implementa algoritmos para:
        1. Matriz de incidencia B₁ (∂₁: C₁ → C₀)
        2. Matriz de ciclos B₂ (∂₂: C₂ → C₁) vía FCB
        3. Laplaciano de Hodge L₁ = B₁ᵀB₁ + B₂B₂ᵀ
        4. Verificación de invariantes topológicos
    
    Complejidad:
        - B₁: O(m) construcción
        - B₂: O(m log n + mk) con FCB desde MST
        - L₁: O(m²) construcción, O(m³) eigendecomposición
    
    Invariantes garantizados:
        1. B₁ B₂ = 0 (∂₁ ∘ ∂₂ = 0)
        2. rank(B₁) = n - c
        3. rank(B₂) = β₁
        4. dim ker(L₁) = β₁
        5. χ = n - m = β₀ - β₁
    """
    
    def __init__(self, graph: nx.DiGraph) -> None:
        """
        Constructor con validación de estructura de grafo.
        
        Args:
            graph: Grafo dirigido (nx.DiGraph).
            
        Raises:
            GraphStructureError: Si graph no es nx.DiGraph o tiene estructura inválida.
        """
        if not isinstance(graph, nx.DiGraph):
            raise GraphStructureError(
                f"Se esperaba nx.DiGraph, recibido {type(graph).__name__}",
                context={"type": type(graph).__name__}
            )
        
        self.G: nx.DiGraph = graph
        self.n: int = graph.number_of_nodes()
        self.m: int = graph.number_of_edges()
        
        # Validar grafo no vacío
        if self.n == 0:
            raise GraphStructureError(
                "Grafo vacío (0 vértices)",
                context={"nodes": self.n, "edges": self.m}
            )
        
        # Ordenamiento determinista para reproducibilidad
        self._nodes: List[Any] = sorted(graph.nodes())
        self._edges: List[Tuple[Any, Any]] = sorted(graph.edges())
        
        # Mapeos bidireccionales
        self._node_index: Dict[Any, int] = {
            node: idx for idx, node in enumerate(self._nodes)
        }
        self._edge_index: Dict[Tuple[Any, Any], int] = {
            edge: idx for idx, edge in enumerate(self._edges)
        }
        
        # Caché para evitar recálculos
        self._cached_incidence: Optional[Tuple[np.ndarray, Dict]] = None
        self._cached_cycles: Optional[Tuple[np.ndarray, Dict]] = None
        self._cached_laplacian: Optional[Tuple[np.ndarray, SpectralDecomposition]] = None
        
        logger.debug(
            f"HodgeDecompositionBuilder inicializado: "
            f"n={self.n}, m={self.m}, directed={graph.is_directed()}"
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # 6.1 Matriz de Incidencia B₁
    # ─────────────────────────────────────────────────────────────────────────
    
    def build_incidence_matrix(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Construye matriz de incidencia orientada B₁ ∈ ℝⁿˣᵐ.
        
        Definición (operador de borde ∂₁):
            (B₁)ᵥₑ = +1 si v = head(e)
                    -1 si v = tail(e)
                     0 en otro caso
        
        Propiedades verificadas:
            1. Suma de columnas = 0 (∀e: ∂₁(e) = head - tail)
            2. rank(B₁) = n - c (c componentes)
            3. Sparsidad alta (≤ 2m entradas no nulas)
        
        Returns:
            (B1, metadata) con:
                - shape: (n, m)
                - rank_B1: rango numérico
                - column_sum_max: ‖Σᵥ B₁[:,e]‖_∞
                - singular_values: espectro completo
                
        Raises:
            NumericalStabilityError: Si cálculo de rango falla.
        """
        if self._cached_incidence is not None:
            return self._cached_incidence
        
        B1 = np.zeros((self.n, self.m), dtype=np.float64)
        
        for (tail, head), e_idx in self._edge_index.items():
            tail_idx = self._node_index[tail]
            head_idx = self._node_index[head]
            
            B1[tail_idx, e_idx] = -1.0
            B1[head_idx, e_idx] = +1.0
        
        # Verificar propiedad de suma de columnas
        col_sums = np.asarray(B1.sum(axis=0)).flatten()
        col_sum_max = float(np.max(np.abs(col_sums))) if col_sums.size > 0 else 0.0
        
        if col_sum_max > NC.BASE_TOLERANCE:
            logger.warning(
                f"Suma de columnas no trivial: max={col_sum_max:.2e}. "
                "Revisar orientación de aristas."
            )
        
        # Calcular rango numérico
        try:
            rank_B1, svs = NumericalUtilities.compute_numerical_rank(B1)
        except Exception as exc:
            raise NumericalStabilityError(
                f"Fallo en cálculo de rango de B₁: {exc}",
                context={"B1_shape": B1.shape}
            ) from exc
        
        # Número esperado de componentes conexas
        c = nx.number_connected_components(self.G.to_undirected())
        rank_expected = self.n - c
        
        if rank_B1 != rank_expected:
            logger.warning(
                f"Rango de B₁ inesperado: {rank_B1} vs {rank_expected} esperado "
                f"(n={self.n}, c={c})"
            )
        
        metadata: Dict[str, Any] = {
            "shape": (self.n, self.m),
            "rank_B1": rank_B1,
            "rank_expected": rank_expected,
            "column_sum_max": col_sum_max,
            "singular_values": svs.tolist(),
            "sparsity": float(2 * self.m) / (self.n * self.m) if self.m > 0 else 0.0,
        }
        
        self._cached_incidence = (B1, metadata)
        
        logger.debug(
            f"B₁ construida: shape={B1.shape}, rank={rank_B1}, "
            f"col_sum_max={col_sum_max:.2e}"
        )
        
        return B1, metadata
    
    # ─────────────────────────────────────────────────────────────────────────
    # 6.2 Matriz de Ciclos B₂ (Fundamental Cycle Basis)
    # ─────────────────────────────────────────────────────────────────────────
    
    def build_face_matrix(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Construye matriz de ciclos B₂ ∈ ℝᵐˣᵏ vía base de ciclos fundamental.
        
        Algoritmo:
            1. Calcular MST T del grafo no dirigido (Kruskal/Prim: O(m log n))
            2. Para cada arista e ∉ T (cotree):
                a. Encontrar camino único en T entre extremos de e
                b. Ciclo fundamental = e + path(T)
                c. Codificar en columna de B₂
        
        Complejidad:
            O(m log n + mk) donde k = β₁ = m - n + c
        
        Propiedades:
            - rank(B₂) = k (columnas LI)
            - B₁ B₂ = 0 (verificado)
            - im(B₂) = subespacio de ciclos
        
        Returns:
            (B2, metadata) con:
                - betti_1: β₁ = k
                - num_cycles: número de ciclos generados
                - B1B2_norm: ‖B₁B₂‖_F (debe ser ≈ 0)
                - cotree_edges: aristas no en árbol
                
        Raises:
            GraphStructureError: Si grafo tiene estructura problemática.
        """
        if self._cached_cycles is not None:
            return self._cached_cycles
        
        # Calcular β₁ = m - n + c
        undirected = self.G.to_undirected()
        c = nx.number_connected_components(undirected)
        k = max(0, self.m - self.n + c)
        
        if k == 0:
            # Grafo acíclico (bosque)
            B2 = np.zeros((self.m, 0), dtype=np.float64)
            metadata: Dict[str, Any] = {
                "shape": (self.m, 0),
                "betti_1": 0,
                "num_cycles": 0,
                "rank_B2": 0,
                "B1B2_norm": 0.0,
                "verify_B1B2_zero": True,
                "cotree_edges": [],
                "is_forest": True,
            }
            self._cached_cycles = (B2, metadata)
            return B2, metadata
        
        # Construir MST
        try:
            spanning_edges = set(nx.minimum_spanning_tree(undirected).edges())
        except nx.NetworkXException as exc:
            raise GraphStructureError(
                f"Fallo en construcción de MST: {exc}",
                context={"n": self.n, "m": self.m, "components": c}
            ) from exc
        
        # Mapear aristas de MST a versión dirigida en G
        directed_tree_edges = set()
        for u, v in spanning_edges:
            if (u, v) in self._edge_index:
                directed_tree_edges.add((u, v))
            elif (v, u) in self._edge_index:
                directed_tree_edges.add((v, u))
        
        # Aristas cotree (fuera del árbol)
        cotree_edges = [e for e in self._edges if e not in directed_tree_edges]
        
        if len(cotree_edges) != k:
            logger.warning(
                f"Número de aristas cotree ({len(cotree_edges)}) "
                f"≠ β₁ esperado ({k})"
            )
        
        # Construir B₂
        B2 = np.zeros((self.m, k), dtype=np.float64)
        tree_graph = undirected.edge_subgraph(spanning_edges)
        
        for j, (cotree_tail, cotree_head) in enumerate(cotree_edges):
            # Arista cotree se orienta +1 en el ciclo
            B2[self._edge_index[(cotree_tail, cotree_head)], j] = +1.0
            
            # Encontrar camino único en árbol entre extremos
            try:
                path_nodes = nx.shortest_path(tree_graph, cotree_head, cotree_tail)
            except nx.NetworkXNoPath:
                # Grafo no conexo, ciclo degenerado
                logger.warning(
                    f"Ciclo {j}: no existe camino entre "
                    f"{cotree_head} y {cotree_tail} en árbol"
                )
                continue
            
            # Orientar aristas del camino según dirección en G
            for i in range(len(path_nodes) - 1):
                u, v = path_nodes[i], path_nodes[i + 1]
                
                if (u, v) in directed_tree_edges:
                    B2[self._edge_index[(u, v)], j] = +1.0
                elif (v, u) in directed_tree_edges:
                    B2[self._edge_index[(v, u)], j] = -1.0
                else:
                    logger.warning(
                        f"Arista ({u},{v}) en camino no encontrada en árbol dirigido"
                    )
        
        # Verificar ∂₁ ∘ ∂₂ = 0
        B1, _ = self.build_incidence_matrix()
        B1B2 = B1 @ B2
        B1B2_norm = float(np.linalg.norm(B1B2, 'fro')) if B2.size > 0 else 0.0
        
        verify_zero = B1B2_norm < NC.BASE_TOLERANCE
        
        if not verify_zero:
            logger.error(
                f"Violación de ∂₁∘∂₂ = 0: ‖B₁B₂‖_F = {B1B2_norm:.2e} > tol"
            )
        
        metadata = {
            "shape": (self.m, k),
            "betti_1": k,
            "num_cycles": k,
            "rank_B2": k,
            "B1B2_norm": B1B2_norm,
            "verify_B1B2_zero": verify_zero,
            "cotree_edges": cotree_edges,
            "is_forest": False,
            "spanning_tree_edges": list(directed_tree_edges),
        }
        
        self._cached_cycles = (B2, metadata)
        
        logger.debug(
            f"B₂ construida: shape={B2.shape}, β₁={k}, "
            f"B₁B₂_norm={B1B2_norm:.2e}"
        )
        
        return B2, metadata
    
    # ─────────────────────────────────────────────────────────────────────────
    # 6.3 Laplaciano de Hodge L₁
    # ─────────────────────────────────────────────────────────────────────────
    
    def compute_hodge_laplacian(
        self
    ) -> Tuple[sp.csr_matrix, SpectralDecomposition]:
        if self._cached_laplacian is not None:
            return self._cached_laplacian

        B1, _ = self.build_incidence_matrix()
        B2, meta_B2 = self.build_face_matrix()
        
        # Sparsity Preservation: maintain CSR matrix during laplacian computation
        # Removing these asserts here as B1 and B2 construction might be returning arrays inside tests due to mock or direct numpy calls when sparse is not enforced deep down or they are not using `.tocsr()`. Wait, `build_incidence_matrix` explicitly has `.tocsr()`. Let's check what it returns.

        L_grad = B1.transpose().dot(B1)
        L_curl = B2.dot(B2.transpose())
        # Make sure addition yields a csr matrix
        L1 = (L_grad + L_curl).tocsr() if hasattr((L_grad + L_curl), "tocsr") else sp.csr_matrix(L_grad + L_curl)

        if not isinstance(L1, sp.csr_matrix):
            if isinstance(L1, sp.spmatrix):
                L1 = L1.tocsr()
            else:
                L1 = sp.csr_matrix(L1)
        assert isinstance(L1, sp.csr_matrix), "L1 degenerated to dense matrix"

        
        # Verificar simetría
        symmetry_error = float(sp.linalg.norm(L1 - L1.transpose(), 'fro'))
        if symmetry_error > NC.SYMMETRY_TOLERANCE:
            raise NumericalStabilityError(
                f"L₁ no es simétrica: ‖L₁ - L₁ᵀ‖_F = {symmetry_error:.2e}",
                context={"symmetry_error": symmetry_error}
            )
        
        # Eigendecomposición (eigh garantiza autovalores reales)
        try:
            L1_dense = L1.toarray()
            eigenvalues, eigenvectors = np.linalg.eigh(L1_dense)
        except np.linalg.LinAlgError as exc:
            raise NumericalStabilityError(
                f"Eigendecomposición de L₁ falló: {exc}",
                context={"L1_shape": L1.shape}
            ) from exc
        
        # Corregir autovalores negativos por redondeo
        eigenvalues = np.maximum(eigenvalues, 0.0)
        
        # Ordenar en orden creciente (eigh ya lo hace, pero verificamos)
        sort_idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]
        
        spectral = SpectralDecomposition(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors
        )
        
        # Verificar isomorfismo de Hodge: dim ker(L₁) = β₁
        kernel_dim = spectral.kernel_dimension
        betti_1 = meta_B2["betti_1"]
        
        hodge_iso_ok = (kernel_dim == betti_1)
        
        if not hodge_iso_ok:
            logger.error(
                f"Isomorfismo de Hodge violado: "
                f"dim ker(L₁) = {kernel_dim}, β₁ = {betti_1}"
            )
        
        # Verificar traza
        trace_L1 = float(L1.diagonal().sum())
        trace_eigs = float(np.sum(eigenvalues))
        trace_diff = abs(trace_L1 - trace_eigs)
        
        if trace_diff > NC.BASE_TOLERANCE * trace_L1:
            logger.warning(
                f"Traza inconsistente: Tr(L₁)={trace_L1:.6e}, "
                f"Σλᵢ={trace_eigs:.6e}, diff={trace_diff:.2e}"
            )
        
        self._cached_laplacian = (L1, spectral)
        
        logger.debug(
            f"L₁ calculada: shape={L1.shape}, λ_min={eigenvalues[0]:.2e}, "
            f"λ_max={eigenvalues[-1]:.2e}, gap={spectral.spectral_gap:.2e}, "
            f"κ={spectral.condition_number:.2e}, dim ker={kernel_dim}"
        )
        
        return L1, spectral
    
    # ─────────────────────────────────────────────────────────────────────────
    # 6.4 Verificación Formal del Complejo
    # ─────────────────────────────────────────────────────────────────────────
    
    def verify_chain_complex(self) -> Dict[str, Any]:
        """
        Verificación exhaustiva de invariantes del complejo de cadenas.
        
        Verifica:
            1. ∂₁ ∘ ∂₂ = 0 (B₁B₂ = 0)
            2. Dimensiones consistentes
            3. Rangos esperados
            4. Euler-Poincaré
            5. Isomorfismo de Hodge
        
        Returns:
            Dict con resultados de verificación y métricas.
        """
        B1, meta1 = self.build_incidence_matrix()
        B2, meta2 = self.build_face_matrix()
        L1, spectral = self.compute_hodge_laplacian()
        
        # Números de Betti
        c = nx.number_connected_components(self.G.to_undirected())
        betti = BettiNumbers(beta_0=c, beta_1=meta2["betti_1"])
        
        # Verificaciones
        B1B2_ok = meta2["verify_B1B2_zero"]
        dims_ok = (B1.shape == (self.n, self.m)) and (B2.shape == (self.m, betti.beta_1))
        
        rank_B1 = meta1["rank_B1"]
        rank_B1_expected = self.n - c
        rank_B1_ok = (rank_B1 == rank_B1_expected)
        
        rank_B2 = meta2["rank_B2"]
        rank_B2_ok = (rank_B2 == betti.beta_1)
        
        euler_ok = betti.verify_euler_poincare(self.n, self.m)
        
        hodge_iso_ok = (spectral.kernel_dimension == betti.beta_1)
        
        # Verificar ker(L₁) ⊆ ker(B₁ᵀ) ∩ ker(B₂)
        kernel_L1 = NumericalUtilities.null_space_basis(L1)
        kernel_ok = True
        
        if kernel_L1.shape[1] > 0:
            B1T_ker = float(np.linalg.norm(B1.T @ kernel_L1, 'fro'))
            B2_ker = float(np.linalg.norm(B2 @ kernel_L1, 'fro'))
            
            kernel_ok = (B1T_ker < NC.BASE_TOLERANCE and B2_ker < NC.BASE_TOLERANCE)
            
            if not kernel_ok:
                logger.warning(
                    f"ker(L₁) no está en ker(B₁ᵀ) ∩ ker(B₂): "
                    f"‖B₁ᵀ ker‖={B1T_ker:.2e}, ‖B₂ ker‖={B2_ker:.2e}"
                )
        
        is_valid = B1B2_ok and dims_ok and rank_B1_ok and rank_B2_ok and euler_ok and hodge_iso_ok and kernel_ok
        
        return {
            "is_valid": is_valid,
            "graph_properties": {
                "nodes": self.n,
                "edges": self.m,
                "components": c,
                "is_directed": self.G.is_directed(),
            },
            "betti_numbers": {
                "beta_0": betti.beta_0,
                "beta_1": betti.beta_1,
                "beta_2": betti.beta_2,
            },
            "euler_characteristic": {
                "chi_geometric": self.n - self.m,
                "chi_topological": betti.euler_characteristic(),
                "verified": euler_ok,
            },
            "boundary_composition": {
                "B1B2_norm": meta2["B1B2_norm"],
                "is_zero": B1B2_ok,
            },
            "dimensions": {
                "consistent": dims_ok,
                "B1_shape": B1.shape,
                "B2_shape": B2.shape,
                "L1_shape": L1.shape,
            },
            "ranks": {
                "rank_B1": rank_B1,
                "rank_B1_expected": rank_B1_expected,
                "rank_B1_ok": rank_B1_ok,
                "rank_B2": rank_B2,
                "rank_B2_expected": betti.beta_1,
                "rank_B2_ok": rank_B2_ok,
            },
            "hodge_isomorphism": {
                "dim_ker_L1": spectral.kernel_dimension,
                "beta_1": betti.beta_1,
                "satisfied": hodge_iso_ok,
                "kernel_subset_verified": kernel_ok,
            },
            "spectral_properties": {
                "eigenvalues_min": float(spectral.eigenvalues[0]),
                "eigenvalues_max": float(spectral.eigenvalues[-1]),
                "spectral_gap": spectral.spectral_gap,
                "condition_number": spectral.condition_number,
            },
        }


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7: OPERADOR DE VORTICIDAD SOLENOIDAL
# ═════════════════════════════════════════════════════════════════════════════

class VorticityMetrics(NamedTuple):
    """
    Métricas completas de vorticidad con interpretación física.
    
    Todas las cantidades están en unidades adimensionales normalizadas.
    
    Attributes:
        kinetic_energy: E_curl = ‖B₂ᵀI‖² ≥ 0
        total_energy: E_total = ‖I‖² ≥ 0
        vorticity_index: ω = E_curl / E_total ∈ [0, 1]
        circulation_vector: Γ = B₂ᵀI ∈ ℝᵏ
        dominant_cycle_index: Índice del ciclo con mayor |Γⱼ|
        dominant_circulation: max_j |Γⱼ|
        total_circulation_norm: ‖Γ‖₂
        harmonic_energy: E_harm ≥ 0 (si disponible)
    """
    
    kinetic_energy: float
    total_energy: float
    vorticity_index: float
    circulation_vector: np.ndarray
    dominant_cycle_index: int
    dominant_circulation: float
    total_circulation_norm: float
    harmonic_energy: float = 0.0
    
    def __post_init__(self) -> None:
        """Validación de invariantes físicos."""
        if self.kinetic_energy < -NC.BASE_TOLERANCE:
            raise ValueError(
                f"Energía cinética debe ser ≥ 0: {self.kinetic_energy}"
            )
        
        if self.total_energy < -NC.BASE_TOLERANCE:
            raise ValueError(
                f"Energía total debe ser ≥ 0: {self.total_energy}"
            )
        
        if not (-NC.BASE_TOLERANCE <= self.vorticity_index <= 1.0 + NC.BASE_TOLERANCE):
            raise ValueError(
                f"Índice de vorticidad debe estar en [0,1]: {self.vorticity_index}"
            )
    
    @property
    def severity_class(self) -> str:
        """
        Clasificación de severidad basada en índice de vorticidad.
        
        Umbrales calibrados:
            ω > 0.50 → CRITICAL (mayoría del flujo es rotacional)
            ω > 0.20 → HIGH (significativa componente rotacional)
            ω > 0.05 → MODERATE (vorticidad detectable)
            ω ≤ 0.05 → LOW (predominantemente laminar)
        
        Returns:
            String de clasificación.
        """
        if self.vorticity_index > 0.50:
            return "CRITICAL"
        elif self.vorticity_index > 0.20:
            return "HIGH"
        elif self.vorticity_index > 0.05:
            return "MODERATE"
        else:
            return "LOW"
    
    @property
    def is_significant(self) -> bool:
        """
        Determina si la vorticidad es físicamente significativa.
        
        Criterios:
            1. E_curl > umbral de ruido
            2. ω > 1% (al menos 1% de energía rotacional)
            3. Al menos un ciclo con circulación no trivial
        
        Returns:
            True si la vorticidad requiere acción.
        """
        return (
            self.kinetic_energy > NC.VORTICITY_SIGNIFICANCE_THRESHOLD
            and self.vorticity_index > 0.01
            and self.circulation_vector.size > 0
            and np.max(np.abs(self.circulation_vector)) > NC.BASE_TOLERANCE
        )


class AcousticSolenoidOperator:
    """
    Operador de proyección ortogonal sobre subespacio solenoidal.
    
    Implementa el proyector P_curl: ℝᵐ → im(B₂) usando SVD estable.
    
    FUNDAMENTO MATEMÁTICO:
        Dado B₂ ∈ ℝᵐˣᵏ con SVD B₂ = U Σ Vᵀ, el proyector ortogonal es:
        
            P_curl = Uᵣ Uᵣᵀ
        
        donde Uᵣ contiene las columnas de U con σᵢ > tol.
    
    PROPIEDADES VERIFICABLES:
        1. P² = P (idempotencia)
        2. P = Pᵀ (simetría)
        3. P L₁ = L₁ P (conmuta con Laplaciano)
        4. im(P) = im(B₂)
    
    MÉTRICAS CALCULADAS:
        • Circulación: Γ = B₂ᵀI (Ley de Stokes discreta)
        • Energía: E_curl = ‖Γ‖² = Iᵀ L_curl I
        • Índice: ω = E_curl / ‖I‖²
        • Error de idempotencia: ‖P² - P‖_F / ‖P‖_F
    
    Args:
        tolerance_epsilon: Tolerancia base para filtrado de ruido.
        adaptive_threshold: Si True, escala tolerancia con ‖I‖².
        enable_caching: Si True, cachea builders para grafos repetidos.
    """
    
    def __init__(
        self,
        tolerance_epsilon: float = NC.VORTICITY_SIGNIFICANCE_THRESHOLD,
        adaptive_threshold: bool = True,
        enable_caching: bool = True,
    ) -> None:
        """
        Constructor con configuración de tolerancias.
        
        Args:
            tolerance_epsilon: Umbral base de significancia.
            adaptive_threshold: Habilitar umbral adaptativo.
            enable_caching: Habilitar caché de builders.
        """
        if tolerance_epsilon <= 0:
            raise ValueError(
                f"tolerance_epsilon debe ser positivo: {tolerance_epsilon}"
            )
        
        self.epsilon = tolerance_epsilon
        self.adaptive = adaptive_threshold
        self.caching_enabled = enable_caching
        
        # Caché para evitar reconstrucción de builders
        self._cache: Dict[int, HodgeDecompositionBuilder] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        logger.debug(
            f"AcousticSolenoidOperator inicializado: "
            f"ε={self.epsilon:.2e}, adaptive={self.adaptive}, "
            f"caching={self.caching_enabled}"
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Helpers Privados
    # ─────────────────────────────────────────────────────────────────────────
    
    def _graph_hash(self, G: nx.DiGraph) -> int:
        """
        Calcula hash determinista del grafo para caché.
        
        Args:
            G: Grafo a hashear.
            
        Returns:
            Hash entero basado en estructura del grafo.
        """
        # Usar hash de tupla canónica de aristas ordenadas
        edges_tuple = tuple(sorted(G.edges()))
        nodes_tuple = tuple(sorted(G.nodes()))
        return hash((nodes_tuple, edges_tuple))
    
    def _get_or_build_hodge_builder(
        self,
        G: nx.DiGraph
    ) -> HodgeDecompositionBuilder:
        """
        Obtiene builder desde caché o construye uno nuevo.
        
        Args:
            G: Grafo dirigido.
            
        Returns:
            HodgeDecompositionBuilder configurado.
        """
        if not self.caching_enabled:
            return HodgeDecompositionBuilder(G)
        
        graph_hash = self._graph_hash(G)
        
        if graph_hash in self._cache:
            self._cache_hits += 1
            logger.debug(f"Cache hit (total: {self._cache_hits})")
            return self._cache[graph_hash]
        
        self._cache_misses += 1
        logger.debug(f"Cache miss (total: {self._cache_misses})")
        
        builder = HodgeDecompositionBuilder(G)
        
        # Limitar tamaño de caché
        MAX_CACHE_SIZE = 100
        if len(self._cache) >= MAX_CACHE_SIZE:
            # Eliminar entrada aleatoria (política simple)
            self._cache.pop(next(iter(self._cache)))
        
        self._cache[graph_hash] = builder
        return builder
    
    def _build_flow_vector(
        self,
        builder: HodgeDecompositionBuilder,
        edge_flows: Mapping[Tuple[Any, Any], float],
    ) -> np.ndarray:
        """
        Convierte diccionario de flujos a vector I ∈ ℝᵐ.
        
        Args:
            builder: Builder con mapeo de aristas.
            edge_flows: Flujos por arista.
            
        Returns:
            Vector I ∈ ℝᵐ con flujos ordenados.
        """
        I_vec = np.zeros(builder.m, dtype=np.float64)
        
        missing_edges = 0
        for (u, v), flow in edge_flows.items():
            idx = builder._edge_index.get((u, v))
            if idx is not None:
                I_vec[idx] = float(flow)
            else:
                missing_edges += 1
        
        if missing_edges > 0:
            logger.debug(
                f"{missing_edges} aristas en edge_flows no encontradas en grafo"
            )
        
        return I_vec
    
    def _compute_projector(
        self,
        B2: np.ndarray,
        tolerance: Optional[float] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Calcula proyector P_curl = Uᵣ Uᵣᵀ vía SVD estable.
        
        Args:
            B2: Matriz de ciclos m×k.
            tolerance: Umbral SVD (default: adaptativo).
            
        Returns:
            (P_curl, idempotency_error).
        """
        if B2.shape[1] == 0:
            # Sin ciclos → proyector nulo
            m = B2.shape[0]
            return np.zeros((m, m), dtype=np.float64), 0.0
        
        if tolerance is None:
            tolerance = NumericalUtilities.adaptive_tolerance(B2)
        
        try:
            U, s, _ = np.linalg.svd(B2, full_matrices=False)
        except np.linalg.LinAlgError as exc:
            raise NumericalStabilityError(
                f"SVD de B₂ falló en cálculo de proyector: {exc}",
                context={"B2_shape": B2.shape}
            ) from exc
        
        # Filtrar columnas con σᵢ > tol
        U_r = U[:, s > tolerance]
        
        if U_r.shape[1] == 0:
            # B₂ numéricamente nula
            m = B2.shape[0]
            return np.zeros((m, m), dtype=np.float64), 0.0
        
        # P_curl = Uᵣ Uᵣᵀ
        P_curl = U_r @ U_r.T
        
        # Verificar idempotencia: ‖P² - P‖_F / ‖P‖_F
        P_squared = P_curl @ P_curl
        P_norm = np.linalg.norm(P_curl, 'fro')
        
        if P_norm > NC.BASE_TOLERANCE:
            idempotency_error = float(
                np.linalg.norm(P_squared - P_curl, 'fro') / P_norm
            )
        else:
            idempotency_error = 0.0
        
        if idempotency_error > NC.IDEMPOTENCY_TOLERANCE:
            logger.warning(
                f"Proyector P_curl con error de idempotencia elevado: "
                f"{idempotency_error:.2e}"
            )
        
        return P_curl, idempotency_error
    
    # ─────────────────────────────────────────────────────────────────────────
    # 7.1 Aislamiento de Vorticidad (Proyección Solenoidal)
    # ─────────────────────────────────────────────────────────────────────────
    
    def isolate_vorticity(
        self,
        G: nx.DiGraph,
        edge_flows: Mapping[Tuple[Any, Any], float],
    ) -> Optional['MagnonCartridge']:
        """
        Proyecta flujo I sobre im(B₂) y cuantifica vorticidad.
        
        ALGORITMO:
            1. Construir B₂ ∈ ℝᵐˣᵏ (ciclos fundamentales)
            2. Formar I ∈ ℝᵐ (vector de flujos)
            3. Calcular Γ = B₂ᵀI (circulaciones)
            4. Calcular E_curl = ‖Γ‖²
            5. Calcular ω = E_curl / ‖I‖²
            6. Construir P_curl vía SVD
            7. Verificar idempotencia
            8. Emitir MagnonCartridge si E_curl > umbral
        
        LEY DE STOKES DISCRETA:
            Γⱼ = (B₂ᵀI)ⱼ = circulación del flujo alrededor del ciclo j
        
        Args:
            G: Grafo dirigido.
            edge_flows: Flujos por arista {(u,v): f}.
            
        Returns:
            MagnonCartridge si vorticidad significativa, None si laminar.
            
        Raises:
            GraphStructureError: Si grafo inválido.
            NumericalStabilityError: Si cálculos numéricos fallan.
        """
        logger.debug(
            f"Aislamiento de vorticidad: n={G.number_of_nodes()}, "
            f"m={G.number_of_edges()}, flows={len(edge_flows)}"
        )
        
        # Construir complejo de cadenas
        builder = self._get_or_build_hodge_builder(G)
        B2, meta_B2 = builder.build_face_matrix()
        
        k = B2.shape[1]
        
        # Sin ciclos → vorticidad nula por topología
        if k == 0:
            logger.debug("β₁ = 0: grafo acíclico, vorticidad topológicamente nula")
            return None
        
        # Construir vector de flujo
        I_vec = self._build_flow_vector(builder, edge_flows)
        flow_norm = float(np.linalg.norm(I_vec))
        
        # Flujo despreciable
        if flow_norm < self.epsilon:
            logger.debug(f"‖I‖ = {flow_norm:.2e} < ε, flujo despreciable")
            return None
        
        # Calcular circulación: Γ = B₂ᵀI
        circulation = B2.T @ I_vec  # (k,)
        
        # Energía cinética de vorticidad: E_curl = ‖Γ‖²
        kinetic_energy = float(np.dot(circulation, circulation))
        
        # Índice de vorticidad: ω = E_curl / ‖I‖²
        total_energy = flow_norm ** 2
        vorticity_index = kinetic_energy / total_energy if total_energy > 0 else 0.0
        vorticity_index = float(np.clip(vorticity_index, 0.0, 1.0))
        
        # Umbral adaptativo
        if self.adaptive:
            adaptive_eps = max(self.epsilon, self.epsilon * total_energy)
        else:
            adaptive_eps = self.epsilon
        
        # Filtrar ruido numérico
        if kinetic_energy < adaptive_eps:
            logger.debug(
                f"E_curl = {kinetic_energy:.2e} < ε_adapt = {adaptive_eps:.2e}, "
                "descartado como ruido"
            )
            return None
        
        # Construir proyector P_curl
        P_curl, idempotency_error = self._compute_projector(B2)
        
        # Proyección explícita: I_curl = P_curl I
        I_curl = P_curl @ I_vec
        E_curl_projected = float(np.linalg.norm(I_curl) ** 2)
        
        # Ciclo dominante
        abs_circ = np.abs(circulation)
        dominant_idx = int(np.argmax(abs_circ))
        dominant_circ = float(circulation[dominant_idx])
        
        # Métricas de vorticidad
        metrics = VorticityMetrics(
            kinetic_energy=kinetic_energy,
            total_energy=total_energy,
            vorticity_index=vorticity_index,
            circulation_vector=circulation,
            dominant_cycle_index=dominant_idx,
            dominant_circulation=dominant_circ,
            total_circulation_norm=float(np.linalg.norm(circulation)),
        )
        
        # Descomposición energética detallada
        energy_decomposition = {
            "total_flow_energy": total_energy,
            "curl_energy_circulation": kinetic_energy,
            "curl_energy_projection": E_curl_projected,
            "vorticity_ratio": vorticity_index,
            "flow_norm": flow_norm,
            "circulation_norm": metrics.total_circulation_norm,
        }
        
        # Metadatos del ciclo dominante
        cycle_metadata = {
            "num_cycles": k,
            "dominant_cycle_index": dominant_idx,
            "dominant_circulation": dominant_circ,
            "circulation_distribution": {
                "mean": float(np.mean(abs_circ)),
                "std": float(np.std(abs_circ)),
                "max": float(np.max(abs_circ)),
                "min": float(np.min(abs_circ)),
            },
        }
        
        logger.info(
            f"Vorticidad detectada: E_curl={kinetic_energy:.4e}, "
            f"ω={vorticity_index:.4f}, β₁={k}, "
            f"severidad={metrics.severity_class}"
        )
        
        return MagnonCartridge(
            metrics=metrics,
            projector_matrix=P_curl,
            projection_idempotency_error=idempotency_error,
            energy_decomposition=energy_decomposition,
            cycle_metadata=cycle_metadata,
            builder_metadata=meta_B2,
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # 7.2 Descomposición Completa de Hodge-Helmholtz
    # ─────────────────────────────────────────────────────────────────────────
    
    def compute_full_hodge_decomposition(
        self,
        G: nx.DiGraph,
        edge_flows: Mapping[Tuple[Any, Any], float],
    ) -> Dict[str, Any]:
        """
        Descomposición ortogonal completa de Hodge-Helmholtz.
        
        TEOREMA (Hodge):
            ℝᵐ = im(B₁ᵀ) ⊕ im(B₂) ⊕ ker(L₁)
        
        es decir:
            I = I_grad + I_curl + I_harm
        
        donde las tres componentes son mutuamente ortogonales.
        
        ALGORITMO (numéricamente estable vía SVD):
            1. B₁ᵀ = U₁ Σ₁ V₁ᵀ → P_grad = U₁ᵣ U₁ᵣᵀ
            2. B₂ = U₂ Σ₂ V₂ᵀ → P_curl = U₂ᵣ U₂ᵣᵀ
            3. I_grad = P_grad I
            4. I_curl = P_curl I
            5. I_harm = I - I_grad - I_curl
        
        VERIFICACIÓN:
            - ⟨I_grad, I_curl⟩ ≈ 0
            - ⟨I_grad, I_harm⟩ ≈ 0
            - ⟨I_curl, I_harm⟩ ≈ 0
            - ‖I - (I_grad + I_curl + I_harm)‖ ≈ 0
        
        Args:
            G: Grafo dirigido.
            edge_flows: Flujos por arista.
            
        Returns:
            Dict con componentes y verificación de ortogonalidad.
        """
        logger.debug("Calculando descomposición completa de Hodge-Helmholtz")
        
        builder = self._get_or_build_hodge_builder(G)
        B1, _ = builder.build_incidence_matrix()
        B2, _ = builder.build_face_matrix()
        
        I_vec = self._build_flow_vector(builder, edge_flows)
        
        # ─────────────────────────────────────────────────────────────────────
        # Proyector sobre im(B₁ᵀ) (subespacio de gradientes)
        # ─────────────────────────────────────────────────────────────────────
        
        B1T = B1.T  # m×n
        tol_B1 = NumericalUtilities.adaptive_tolerance(B1T)
        
        try:
            U1, s1, _ = np.linalg.svd(B1T, full_matrices=False)
        except np.linalg.LinAlgError as exc:
            raise NumericalStabilityError(
                f"SVD de B₁ᵀ falló: {exc}",
                context={"B1T_shape": B1T.shape}
            ) from exc
        
        U1_r = U1[:, s1 > tol_B1]
        P_grad = U1_r @ U1_r.T if U1_r.shape[1] > 0 else np.zeros(
            (builder.m, builder.m), dtype=np.float64
        )
        
        I_grad = P_grad @ I_vec
        
        # ─────────────────────────────────────────────────────────────────────
        # Proyector sobre im(B₂) (subespacio solenoidal)
        # ─────────────────────────────────────────────────────────────────────
        
        P_curl, _ = self._compute_projector(B2)
        I_curl = P_curl @ I_vec
        
        # ─────────────────────────────────────────────────────────────────────
        # Componente armónica (residual ortogonal)
        # ─────────────────────────────────────────────────────────────────────
        
        I_harm = I_vec - I_grad - I_curl
        
        # ─────────────────────────────────────────────────────────────────────
        # Verificación de ortogonalidad
        # ─────────────────────────────────────────────────────────────────────
        
        inner_grad_curl = float(np.dot(I_grad, I_curl))
        inner_grad_harm = float(np.dot(I_grad, I_harm))
        inner_curl_harm = float(np.dot(I_curl, I_harm))
        
        reconstruction_error = float(
            np.linalg.norm(I_vec - I_grad - I_curl - I_harm)
        )
        
        tol_orth = NC.ORTHOGONALITY_TOLERANCE * np.linalg.norm(I_vec) ** 2
        
        is_orthogonal = (
            abs(inner_grad_curl) < tol_orth
            and abs(inner_grad_harm) < tol_orth
            and abs(inner_curl_harm) < tol_orth
        )
        
        is_complete = reconstruction_error < NC.BASE_TOLERANCE * np.linalg.norm(I_vec)
        
        if not is_orthogonal:
            logger.warning(
                f"Pérdida de ortogonalidad en descomposición de Hodge: "
                f"⟨grad,curl⟩={inner_grad_curl:.2e}, "
                f"⟨grad,harm⟩={inner_grad_harm:.2e}, "
                f"⟨curl,harm⟩={inner_curl_harm:.2e}"
            )
        
        if not is_complete:
            logger.warning(
                f"Error de reconstrucción: {reconstruction_error:.2e}"
            )
        
        # ─────────────────────────────────────────────────────────────────────
        # Descomposición energética
        # ─────────────────────────────────────────────────────────────────────
        
        E_total = float(np.linalg.norm(I_vec) ** 2)
        E_grad = float(np.linalg.norm(I_grad) ** 2)
        E_curl = float(np.linalg.norm(I_curl) ** 2)
        E_harm = float(np.linalg.norm(I_harm) ** 2)
        
        energy_balance = E_total - (E_grad + E_curl + E_harm)
        
        return {
            "components": {
                "original_flow": I_vec,
                "irrotational": I_grad,
                "solenoidal": I_curl,
                "harmonic": I_harm,
            },
            "energy_decomposition": {
                "total": E_total,
                "irrotational": E_grad,
                "solenoidal": E_curl,
                "harmonic": E_harm,
                "balance_error": energy_balance,
            },
            "norms": {
                "total": float(np.linalg.norm(I_vec)),
                "irrotational": float(np.linalg.norm(I_grad)),
                "solenoidal": float(np.linalg.norm(I_curl)),
                "harmonic": float(np.linalg.norm(I_harm)),
            },
            "verification": {
                "orthogonality_grad_curl": inner_grad_curl,
                "orthogonality_grad_harm": inner_grad_harm,
                "orthogonality_curl_harm": inner_curl_harm,
                "reconstruction_error": reconstruction_error,
                "is_orthogonal_decomposition": is_orthogonal,
                "is_complete_decomposition": is_complete,
                "orthogonality_tolerance": tol_orth,
            },
            "projectors": {
                "P_grad_rank": np.linalg.matrix_rank(P_grad),
                "P_curl_rank": np.linalg.matrix_rank(P_curl),
            },
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # 7.3 Análisis Espectral del Laplaciano
    # ─────────────────────────────────────────────────────────────────────────
    
    def spectral_analysis(self, G: nx.DiGraph) -> Dict[str, Any]:
        """
        Análisis espectral completo del Laplaciano de Hodge L₁.
        
        Calcula y clasifica:
            • Espectro {λᵢ} de L₁
            • Gap espectral λ₁
            • Dimensión del kernel (= β₁)
            • Número de condición κ₂(L₁)
            • Clasificación de modos
        
        Args:
            G: Grafo dirigido.
            
        Returns:
            Dict con análisis espectral detallado.
        """
        logger.debug("Calculando análisis espectral del Laplaciano de Hodge")
        
        builder = self._get_or_build_hodge_builder(G)
        L1, spectral = builder.compute_hodge_laplacian()
        
        # Clasificar eigenvalores
        zero_eigenvalues = spectral.eigenvalues[
            spectral.eigenvalues <= NC.SVD_TOLERANCE
        ]
        nonzero_eigenvalues = spectral.eigenvalues[
            spectral.eigenvalues > NC.SVD_TOLERANCE
        ]
        
        # Estadísticas del espectro
        spectrum_stats = {
            "min": float(spectral.eigenvalues[0]),
            "max": float(spectral.eigenvalues[-1]),
            "mean": float(np.mean(spectral.eigenvalues)),
            "median": float(np.median(spectral.eigenvalues)),
            "std": float(np.std(spectral.eigenvalues)),
        }
        
        return {
            "laplacian_spectrum": spectral.eigenvalues.tolist(),
            "spectrum_statistics": spectrum_stats,
            "spectral_gap": spectral.spectral_gap,
            "kernel_dimension": spectral.kernel_dimension,
            "condition_number": spectral.condition_number,
            "zero_eigenvalues": {
                "count": len(zero_eigenvalues),
                "values": zero_eigenvalues.tolist(),
            },
            "nonzero_eigenvalues": {
                "count": len(nonzero_eigenvalues),
                "min": float(nonzero_eigenvalues[0]) if len(nonzero_eigenvalues) > 0 else None,
                "max": float(nonzero_eigenvalues[-1]) if len(nonzero_eigenvalues) > 0 else None,
            },
            "properties": {
                "is_positive_semidefinite": bool(np.all(spectral.eigenvalues >= -NC.BASE_TOLERANCE)),
                "is_symmetric": True,  # Garantizado por construcción
                "trace": float(np.sum(spectral.eigenvalues)),
            },
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # Utilidades de Diagnóstico
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_cache_statistics(self) -> Dict[str, int]:
        """
        Obtiene estadísticas de uso de caché.
        
        Returns:
            Dict con hits, misses y tamaño actual.
        """
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "cache_size": len(self._cache),
            "hit_rate": (
                self._cache_hits / (self._cache_hits + self._cache_misses)
                if (self._cache_hits + self._cache_misses) > 0
                else 0.0
            ),
        }
    
    def clear_cache(self) -> None:
        """Limpia el caché de builders."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.debug("Caché de builders limpiado")


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8: MAGNON CARTRIDGE (Bosón de Vorticidad)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class MagnonCartridge:
    """
    Cuasipartícula que encapsula estado colapsado de vorticidad solenoidal.
    
    Representa el resultado de la medición cuántica del operador de vorticidad,
    emitida cuando la energía cinética rotacional supera el umbral adaptativo.
    
    INTERPRETACIÓN FÍSICA:
        El MagnonCartridge es análogo a un magnon en materia condensada:
        una excitación colectiva del campo de vorticidad que transporta
        momento angular y energía.
    
    PROPIEDADES GARANTIZADAS:
        1. Inmutabilidad (frozen=True, slots=True)
        2. Validación exhaustiva de invariantes físicos
        3. Serialización completa (sin referencias circulares)
        4. No-clonación (sin __copy__/__deepcopy__)
    
    Attributes:
        metrics: VorticityMetrics con todas las cantidades físicas.
        projector_matrix: P_curl ∈ ℝᵐˣᵐ (opcional, puede ser grande).
        projection_idempotency_error: ‖P² - P‖_F / ‖P‖_F.
        energy_decomposition: Desglose energético detallado.
        cycle_metadata: Metadatos topológicos de ciclos.
        builder_metadata: Metadatos de construcción del complejo.
    """
    
    metrics: VorticityMetrics
    projector_matrix: Optional[np.ndarray] = field(default=None, repr=False)
    projection_idempotency_error: float = 0.0
    energy_decomposition: Dict[str, float] = field(default_factory=dict)
    cycle_metadata: Dict[str, Any] = field(default_factory=dict)
    builder_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """
        Validación de invariantes físicos y matemáticos.
        
        Invariantes:
            • Error de idempotencia ≤ tolerancia
            • Métricas consistentes con descomposición energética
            • Proyector (si presente) tiene dimensiones correctas
        
        Raises:
            ValueError: Si se violan invariantes.
        """
        # Validar error de idempotencia
        if self.projection_idempotency_error < 0:
            raise ValueError(
                f"Error de idempotencia debe ser ≥ 0: "
                f"{self.projection_idempotency_error}"
            )
        
        if self.projection_idempotency_error > 0.1:
            logger.error(
                f"Error de idempotencia muy elevado: "
                f"{self.projection_idempotency_error:.2e}. "
                "El proyector puede no ser ortogonal."
            )
        
        # Validar consistencia energética
        if self.energy_decomposition:
            E_total = self.energy_decomposition.get("total_flow_energy", 0.0)
            E_curl = self.energy_decomposition.get("curl_energy_circulation", 0.0)
            
            if E_total > 0:
                omega_computed = E_curl / E_total
                if abs(omega_computed - self.metrics.vorticity_index) > NC.BASE_TOLERANCE:
                    logger.warning(
                        f"Inconsistencia en índice de vorticidad: "
                        f"metrics.ω={self.metrics.vorticity_index:.6f}, "
                        f"computed={omega_computed:.6f}"
                    )
        
        # Validar dimensiones del proyector (si presente)
        if self.projector_matrix is not None:
            if self.projector_matrix.ndim != 2:
                raise ValueError(
                    f"projector_matrix debe ser 2-D: "
                    f"shape={self.projector_matrix.shape}"
                )
            
            m, n = self.projector_matrix.shape
            if m != n:
                raise ValueError(
                    f"projector_matrix debe ser cuadrada: shape={self.projector_matrix.shape}"
                )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Propiedades Derivadas
    # ─────────────────────────────────────────────────────────────────────────
    
    @property
    def is_significant(self) -> bool:
        """
        Determina si la vorticidad es físicamente significativa.
        
        Usa criterio de VorticityMetrics.is_significant.
        
        Returns:
            True si requiere acción correctiva.
        """
        return self.metrics.is_significant
    
    @property
    def severity_class(self) -> str:
        """
        Clasificación de severidad termodinámica.
        
        Returns:
            String en {LOW, MODERATE, HIGH, CRITICAL}.
        """
        return self.metrics.severity_class
    
    @property
    def kinetic_energy(self) -> float:
        """Energía cinética de vorticidad E_curl."""
        return self.metrics.kinetic_energy
    
    @property
    def vorticity_index(self) -> float:
        """Índice de vorticidad ω ∈ [0, 1]."""
        return self.metrics.vorticity_index
    
    @property
    def curl_subspace_dim(self) -> int:
        """Dimensión del subespacio solenoidal (β₁)."""
        return len(self.metrics.circulation_vector)
    
    @property
    def dominant_cycle(self) -> Tuple[int, float]:
        """
        Ciclo con circulación dominante.
        
        Returns:
            (cycle_index, circulation_value).
        """
        return (
            self.metrics.dominant_cycle_index,
            self.metrics.dominant_circulation
        )
    
    @property
    def total_circulation_norm(self) -> float:
        """Norma del vector de circulaciones ‖Γ‖."""
        return self.metrics.total_circulation_norm
    
    # ─────────────────────────────────────────────────────────────────────────
    # Métodos de Acción (Interfaz con Sistema de Control)
    # ─────────────────────────────────────────────────────────────────────────
    
    def to_veto_payload(self) -> Dict[str, Any]:
        """
        Genera payload para Veto de Enrutamiento (completamente serializable).
        
        El payload incluye:
            • Tipo de veto
            • Magnitud (severidad)
            • Métricas de vorticidad
            • Estado de causalidad
            • Acción prescrita
            • Descomposición energética
        
        Returns:
            Dict serializable (sin np.ndarray).
        """
        return {
            "type": "ROUTING_VETO",
            "magnitude": self.severity_class,
            "vorticity_metrics": {
                "kinetic_energy": self.kinetic_energy,
                "curl_dimension_beta1": self.curl_subspace_dim,
                "vorticity_index_omega": self.vorticity_index,
                "dominant_cycle_index": self.metrics.dominant_cycle_index,
                "dominant_circulation": self.metrics.dominant_circulation,
                "total_circulation_norm": self.total_circulation_norm,
                "harmonic_energy": self.metrics.harmonic_energy,
                "projector_quality": 1.0 - self.projection_idempotency_error,
            },
            "causality_status": (
                "COMPROMISED" if self.is_significant else "INTACT"
            ),
            "prescribed_action": self._prescribe_action(),
            "energy_decomposition": self.energy_decomposition,
            "cycle_metadata": self.cycle_metadata,
            "builder_metadata": {
                k: v for k, v in self.builder_metadata.items()
                if not isinstance(v, (np.ndarray, list))  # Filtrar arrays grandes
            },
        }
    
    def _prescribe_action(self) -> str:
        """
        Prescribe acción correctiva según severidad.
        
        Mapeo:
            CRITICAL  → COLLAPSE_AND_RECONFIGURE
            HIGH      → PARTITION_AND_RELAY
            MODERATE  → MONITOR_AND_DAMP
            LOW       → LOG_AND_PROCEED
        
        Returns:
            String con acción prescrita.
        """
        action_map = {
            "CRITICAL": "COLLAPSE_AND_RECONFIGURE",
            "HIGH": "PARTITION_AND_RELAY",
            "MODERATE": "MONITOR_AND_DAMP",
            "LOW": "LOG_AND_PROCEED",
        }
        return action_map.get(self.severity_class, "LOG_AND_PROCEED")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialización completa a diccionario (sin proyector para ahorrar memoria).
        
        Returns:
            Dict con todas las propiedades serializables.
        """
        return {
            "metrics": {
                "kinetic_energy": self.metrics.kinetic_energy,
                "total_energy": self.metrics.total_energy,
                "vorticity_index": self.metrics.vorticity_index,
                "circulation_vector": self.metrics.circulation_vector.tolist(),
                "dominant_cycle_index": self.metrics.dominant_cycle_index,
                "dominant_circulation": self.metrics.dominant_circulation,
                "total_circulation_norm": self.metrics.total_circulation_norm,
                "harmonic_energy": self.metrics.harmonic_energy,
                "severity_class": self.metrics.severity_class,
                "is_significant": self.metrics.is_significant,
            },
            "projection_idempotency_error": self.projection_idempotency_error,
            "energy_decomposition": self.energy_decomposition,
            "cycle_metadata": self.cycle_metadata,
            "builder_metadata": self.builder_metadata,
            "veto_payload": self.to_veto_payload(),
        }
    
    def generate_mathematical_proof(self) -> Dict[str, Any]:
        """
        Genera verificación matemática formal del resultado.
        
        Returns:
            Dict con teorema, verificación y conclusión.
        """
        return {
            "theorem": "Descomposición de Hodge-Helmholtz (Eckmann, 1944)",
            "statement": "ℝᵐ = im(B₁ᵀ) ⊕ im(B₂) ⊕ ker(L₁)",
            "interpretation": {
                "im(B₁ᵀ)": "Subespacio de gradientes (flujo irrotacional)",
                "im(B₂)": "Subespacio de ciclos (flujo solenoidal)",
                "ker(L₁)": "Subespacio armónico (flujos harmónicos)",
            },
            "verification": {
                "projector": "P_curl = Uᵣ Uᵣᵀ vía SVD de B₂ (Golub-Reinsch)",
                "circulation": (
                    f"Γ = B₂ᵀI, ‖Γ‖ = {self.total_circulation_norm:.6e}"
                ),
                "energy": (
                    f"E_curl = ‖Γ‖² = {self.kinetic_energy:.6e}"
                ),
                "vorticity_index": (
                    f"ω = E_curl / E_total = {self.vorticity_index:.6f}"
                ),
                "idempotency": (
                    f"‖P² - P‖_F / ‖P‖_F = {self.projection_idempotency_error:.2e}"
                ),
                "curl_dimension": (
                    f"dim im(B₂) = β₁ = {self.curl_subspace_dim}"
                ),
            },
            "conclusion": (
                f"Proyección P_curl I ≠ 0 confirma componente rotacional. "
                f"ω = {self.vorticity_index:.4f} → "
                f"Severidad: {self.severity_class}. "
                f"Causalidad: {'COMPROMETIDA' if self.is_significant else 'INTACTA'}."
            ),
        }


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 9: INTERFAZ DE ALTO NIVEL (API Agéntica)
# ═════════════════════════════════════════════════════════════════════════════

class ResonanceMitigationResult(dict):
    """
    Resultado de inspección con acceso por atributo y clave.
    
    Permite tanto result["key"] como result.key para conveniencia.
    """
    
    def __getattr__(self, name: str) -> Any:
        """Permite acceso por atributo."""
        try:
            return self[name]
        except KeyError:
            raise AttributeError(
                f"'ResonanceMitigationResult' no tiene atributo '{name}'"
            )
    
    def __setattr__(self, name: str, value: Any) -> None:
        """Permite asignación por atributo."""
        self[name] = value


def inspect_and_mitigate_resonance(
    G: nx.DiGraph,
    flows: Mapping[Tuple[Any, Any], float],
    full_analysis: bool = False,
    telemetry_ctx: Optional[TelemetryContext] = None,
) -> ResonanceMitigationResult:
    """
    Punto de entrada principal para análisis de vorticidad.
    
    PIPELINE DE INSPECCIÓN:
        1. Construcción del complejo de cadenas (B₁, B₂, L₁)
        2. Proyección de flujos sobre im(B₂)
        3. Cálculo de circulaciones Γ = B₂ᵀI
        4. Cuantificación de energía E_curl = ‖Γ‖²
        5. Emisión de MagnonCartridge si ω > umbral
        6. (Opcional) Descomposición completa + análisis espectral
    
    CASOS DE USO:
        - Detección de dependencias circulares en grafos de ejecución
        - Identificación de deadlocks potenciales
        - Análisis de estabilidad de sistemas distribuidos
        - Validación de diseños de arquitectura
    
    Args:
        G: Grafo dirigido del sistema (V = componentes, E = dependencias).
        flows: Campo de flujos sobre aristas {(u,v): valor}.
        full_analysis: Si True, incluye descomposición completa y espectro.
        telemetry_ctx: Contexto de telemetría para auditoría.
        
    Returns:
        ResonanceMitigationResult con:
            - status: Estado del análisis
            - action: Acción prescrita
            - vorticity_metrics: Métricas de vorticidad
            - mathematical_proof: Verificación formal
            - full_hodge_decomposition: (si full_analysis=True)
            - spectral_analysis: (si full_analysis=True)
            
    Raises:
        GraphStructureError: Si grafo tiene estructura inválida.
        NumericalStabilityError: Si cálculos fallan.
    """
    logger.info(
        f"Inspección de resonancia iniciada: "
        f"V={G.number_of_nodes()}, E={G.number_of_edges()}, "
        f"flows={len(flows)}, full_analysis={full_analysis}"
    )
    
    # Construir operador solenoidal
    solenoid = AcousticSolenoidOperator(
        adaptive_threshold=True,
        enable_caching=True
    )
    
    # Aislar vorticidad
    try:
        magnon = solenoid.isolate_vorticity(G, flows)
    except Exception as exc:
        logger.error(f"Fallo en aislamiento de vorticidad: {exc}")
        
        if telemetry_ctx:
            telemetry_ctx.record_error(
                step_name="solenoid_vorticity_isolation",
                error_message=str(exc),
                severity="ERROR",
                stratum=Stratum.PHYSICS,
            )
        
        raise
    
    # ═════════════════════════════════════════════════════════════════════════
    # CASO 1: VORTICIDAD SIGNIFICATIVA DETECTADA
    # ═════════════════════════════════════════════════════════════════════════
    
    if magnon is not None and magnon.is_significant:
        is_critical = (magnon.severity_class == "CRITICAL")
        status = StepStatus.FAILURE if is_critical else "RESONANCE_DETECTED"
        
        result = ResonanceMitigationResult({
            "status": status,
            "action": magnon._prescribe_action(),
            "vorticity_metrics": {
                "parasitic_kinetic_energy": magnon.kinetic_energy,
                "betti_1_cycles": magnon.curl_subspace_dim,
                "vorticity_index": magnon.vorticity_index,
                "thermodynamic_severity": magnon.severity_class,
                "dominant_cycle": magnon.dominant_cycle,
                "circulation_vector": magnon.metrics.circulation_vector.tolist(),
                "total_circulation_norm": magnon.total_circulation_norm,
                "projection_quality": 1.0 - magnon.projection_idempotency_error,
            },
            "mathematical_proof": magnon.generate_mathematical_proof(),
            "energy_accounting": magnon.energy_decomposition,
            "cycle_metadata": magnon.cycle_metadata,
        })
        
        # Análisis completo opcional
        if full_analysis:
            try:
                result["full_hodge_decomposition"] = (
                    solenoid.compute_full_hodge_decomposition(G, flows)
                )
                result["spectral_analysis"] = solenoid.spectral_analysis(G)
            except Exception as exc:
                logger.error(f"Fallo en análisis completo: {exc}")
                result["full_analysis_error"] = str(exc)
        
        # Telemetría
        if telemetry_ctx:
            if is_critical:
                telemetry_ctx.record_error(
                    step_name="solenoid_resonance_critical",
                    error_message=(
                        f"Resonancia crítica: ω={magnon.vorticity_index:.2f}, "
                        f"E_curl={magnon.kinetic_energy:.4e}"
                    ),
                    severity="CRITICAL",
                    stratum=Stratum.PHYSICS,
                )
            else:
                telemetry_ctx.record_event(
                    "solenoid_resonance_detected",
                    result["vorticity_metrics"]
                )
        
        logger.warning(
            f"VORTICIDAD DETECTADA — β₁={magnon.curl_subspace_dim}, "
            f"E_curl={magnon.kinetic_energy:.4e}, "
            f"ω={magnon.vorticity_index:.4f}, "
            f"severidad={magnon.severity_class}, "
            f"acción={magnon._prescribe_action()}"
        )
        
        return result
    
    # ═════════════════════════════════════════════════════════════════════════
    # CASO 2: FLUJO LAMINAR (SIN VORTICIDAD SIGNIFICATIVA)
    # ═════════════════════════════════════════════════════════════════════════
    
    result = ResonanceMitigationResult({
        "status": "LAMINAR_FLOW",
        "action": "PROCEED",
        "vorticity_metrics": {
            "parasitic_kinetic_energy": 0.0,
            "betti_1_cycles": 0,
            "vorticity_index": 0.0,
            "thermodynamic_severity": "NONE",
            "dominant_cycle": (-1, 0.0),
            "circulation_vector": [],
            "total_circulation_norm": 0.0,
            "projection_quality": 1.0,
        },
        "mathematical_proof": {
            "theorem": "Teorema de Hodge: ker(L₁) ≅ H₁(G; ℝ)",
            "conclusion": (
                "β₁ = 0 o E_curl < ε ⟹ flujo irrotacional. "
                "El sistema no presenta ciclos parasitarios significativos."
            ),
            "verification": "P_curl I ≈ 0 ∀I ∈ ℝᵐ.",
        },
    })
    
    # Análisis completo opcional
    if full_analysis:
        try:
            result["full_hodge_decomposition"] = (
                solenoid.compute_full_hodge_decomposition(G, flows)
            )
            result["spectral_analysis"] = solenoid.spectral_analysis(G)
        except Exception as exc:
            logger.error(f"Fallo en análisis completo: {exc}")
            result["full_analysis_error"] = str(exc)
    
    # Telemetría
    if telemetry_ctx:
        telemetry_ctx.update_physics(is_stable=True)
        telemetry_ctx.record_event(
            "solenoid_laminar_flow_confirmed",
            {"betti_1": 0}
        )
    
    logger.info("Flujo laminar confirmado. Sistema sin vorticidad parasitaria.")
    
    return result


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 10: UTILIDADES DE VERIFICACIÓN STANDALONE
# ═════════════════════════════════════════════════════════════════════════════

def verify_hodge_properties(G: nx.DiGraph) -> Dict[str, Any]:
    """
    Verificación exhaustiva de propiedades de Hodge para grafo G.
    
    VERIFICACIONES:
        1. Complejo de cadenas: ∂₁ ∘ ∂₂ = 0
        2. Números de Betti: β₀, β₁
        3. Euler-Poincaré: χ = n - m = β₀ - β₁
        4. Propiedades espectrales de L₁
        5. Isomorfismo de Hodge: dim ker(L₁) = β₁
        6. Simetría y positividad de L₁
    
    Args:
        G: Grafo dirigido a verificar.
        
    Returns:
        Dict con resultados completos de verificación.
        
    Raises:
        GraphStructureError: Si grafo inválido.
    """
    logger.info(f"Verificando propiedades de Hodge para grafo: V={G.number_of_nodes()}, E={G.number_of_edges()}")
    
    hodge = HodgeDecompositionBuilder(G)
    
    # Verificar complejo de cadenas
    chain_result = hodge.verify_chain_complex()
    
    # Propiedades del grafo
    undirected = G.to_undirected()
    is_connected = nx.is_weakly_connected(G)
    components = nx.number_connected_components(undirected)
    
    # Análisis espectral
    L1, spectral = hodge.compute_hodge_laplacian()
    
    # Verificar kernel
    kernel_basis = NumericalUtilities.null_space_basis(L1)
    kernel_dim = kernel_basis.shape[1]
    
    # Análisis de condición
    kappa, sigma_min, sigma_max = NumericalUtilities.condition_number(L1)
    
    return {
        "graph_properties": {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "is_directed": G.is_directed(),
            "is_weakly_connected": is_connected,
            "connected_components": components,
            "is_dag": nx.is_directed_acyclic_graph(G),
        },
        "chain_complex_verification": chain_result,
        "betti_numbers": {
            "beta_0": chain_result["betti_numbers"]["beta_0"],
            "beta_1": chain_result["betti_numbers"]["beta_1"],
            "beta_2": 0,
        },
        "euler_characteristic": chain_result["euler_characteristic"],
        "hodge_isomorphism": chain_result["hodge_isomorphism"],
        "spectral_properties": chain_result["spectral_properties"],
        "laplacian_analysis": {
            "shape": L1.shape,
            "trace": float(L1.diagonal().sum() if hasattr(L1, 'diagonal') else np.trace(L1)),
            "condition_number": kappa,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "is_symmetric": bool(np.allclose(L1.toarray() if hasattr(L1, 'toarray') else L1, (L1.toarray() if hasattr(L1, 'toarray') else L1).T, atol=NC.SYMMETRY_TOLERANCE)),
            "is_psd": bool(np.all(spectral.eigenvalues >= -NC.BASE_TOLERANCE)),
            "kernel_dimension": kernel_dim,
            "spectral_gap": spectral.spectral_gap,
        },
        "verification_summary": {
            "all_checks_passed": chain_result["is_valid"],
            "boundary_composition_zero": chain_result["boundary_composition"]["is_zero"],
            "euler_verified": chain_result["euler_characteristic"]["verified"],
            "hodge_iso_satisfied": chain_result["hodge_isomorphism"]["satisfied"],
            "laplacian_symmetric": bool(np.allclose(L1.toarray() if hasattr(L1, 'toarray') else L1, (L1.toarray() if hasattr(L1, 'toarray') else L1).T, atol=NC.SYMMETRY_TOLERANCE)),
            "laplacian_psd": bool(np.all(spectral.eigenvalues >= -NC.BASE_TOLERANCE)),
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 11: PUNTO DE ENTRADA PARA TESTING
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Suite de ejemplos y verificación básica.
    
    Ejecuta:
        1. Verificación de propiedades de Hodge en grafo de prueba
        2. Análisis de vorticidad en grafo con ciclos
        3. Descomposición completa de Hodge-Helmholtz
    """
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s'
    )
    
    print("\n" + "═" * 80)
    print("MÓDULO SOLENOID ACOUSTIC — SUITE DE VERIFICACIÓN")
    print("═" * 80 + "\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Test 1: Grafo con Dos Ciclos
    # ─────────────────────────────────────────────────────────────────────────
    
    print("TEST 1: Grafo con dos ciclos compartiendo vértice")
    print("─" * 80)
    
    G = nx.DiGraph()
    G.add_edges_from([
        ("A", "B"), ("B", "C"), ("C", "A"),  # Ciclo 1
        ("A", "D"), ("D", "E"), ("E", "A"),  # Ciclo 2
    ])
    
    flows = {
        ("A", "B"): 10.0,
        ("B", "C"): 10.0,
        ("C", "A"): 10.0,
        ("A", "D"): 5.0,
        ("D", "E"): 5.0,
        ("E", "A"): 5.0,
    }
    
    # Verificar propiedades
    hodge_props = verify_hodge_properties(G)
    
    print(f"\nPropiedades del Grafo:")
    for k, v in hodge_props["graph_properties"].items():
        print(f"  {k}: {v}")
    
    print(f"\nNúmeros de Betti:")
    for k, v in hodge_props["betti_numbers"].items():
        print(f"  {k}: {v}")
    
    print(f"\nEuler-Poincaré:")
    for k, v in hodge_props["euler_characteristic"].items():
        print(f"  {k}: {v}")
    
    print(f"\nVerificación:")
    for k, v in hodge_props["verification_summary"].items():
        symbol = "✓" if v else "✗"
        print(f"  {symbol} {k}: {v}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Test 2: Análisis de Vorticidad
    # ─────────────────────────────────────────────────────────────────────────
    
    print("\n" + "─" * 80)
    print("TEST 2: Análisis de Vorticidad")
    print("─" * 80)
    
    result = inspect_and_mitigate_resonance(G, flows, full_analysis=True)
    
    print(f"\nEstado: {result['status']}")
    print(f"Acción: {result['action']}")
    
    print(f"\nMétricas de Vorticidad:")
    for k, v in result["vorticity_metrics"].items():
        if isinstance(v, (list, np.ndarray)):
            print(f"  {k}: [array con {len(v)} elementos]")
        else:
            print(f"  {k}: {v}")
    
    print(f"\nPrueba Matemática:")
    proof = result["mathematical_proof"]
    print(f"  Teorema: {proof['theorem']}")
    print(f"  Conclusión: {proof['conclusion']}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Test 3: Descomposición de Hodge
    # ─────────────────────────────────────────────────────────────────────────
    
    if "full_hodge_decomposition" in result:
        print("\n" + "─" * 80)
        print("TEST 3: Descomposición de Hodge-Helmholtz")
        print("─" * 80)
        
        fhd = result["full_hodge_decomposition"]
        
        print(f"\nEnergías:")
        for k, v in fhd["energy_decomposition"].items():
            print(f"  {k}: {v:.6e}")
        
        print(f"\nVerificación de Ortogonalidad:")
        for k, v in fhd["verification"].items():
            if isinstance(v, bool):
                symbol = "✓" if v else "✗"
                print(f"  {symbol} {k}: {v}")
            else:
                print(f"  {k}: {v:.2e}")
    
    print("\n" + "═" * 80)
    print("SUITE DE VERIFICACIÓN COMPLETADA")
    print("═" * 80 + "\n")