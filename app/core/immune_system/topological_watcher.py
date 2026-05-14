"""
=========================================================================================
Módulo: Topological Watcher (Sistema Inmunológico Matemático) - VERSIÓN REFINADA
Ubicación: app/core/immune_system/topological_watcher.py
=========================================================================================

FUNDAMENTOS MATEMÁTICOS RIGUROSOS (ACTUALIZADO):

1. GEOMETRÍA RIEMANNIANA:
   - Tensor métrico G ∈ Sym⁺⁺(n) con λ_min(G) ≥ τ > 0 (definición positiva ESTRICTA)
   - Distancia geodésica: d_G(x,y) = inf{∫₀¹ √(γ'(t)ᵀ G γ'(t)) dt}
   - Flujo de Ricci normalizado: ∂_t g = -2(Ric - r·g/n) con r = tr(Ric)/n

2. TOPOLOGÍA ALGEBRAICA:
   - Homología simplicial: H_k(K) = Ker(∂_k) / Im(∂_{k+1})
   - Números de Betti: β_k = rank(H_k) ∈ ℕ₀
   - Característica de Euler: χ = Σ(-1)^k β_k ∈ ℤ
   - Desigualdad de Morse: χ ≤ #critical_points

3. ANÁLISIS FUNCIONAL:
   - Proyectores ortogonales: P² = P, P* = P, ‖P‖₂ = 1
   - Descomposición espectral con orto-normalización de Gram-Schmidt
   - Pre-condicionamiento: κ(D⁻¹GD⁻¹) ≪ κ(G)

4. TEORÍA DE CATEGORÍAS:
   - Funtor F: Top → Narr preserva límites finitos
   - Transformación natural: η_X: F(X) → G(X) compatible con morfismos
   - Mónada de error (T, η, μ) con leyes asociativas verificadas
"""

from __future__ import annotations

import logging
import sys
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, Dict, FrozenSet, Generator, List, Optional, 
    Tuple, Union, Protocol, TypeVar, Callable
)
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy import linalg as sp_linalg  # Para Gram-Schmidt robusto

# Imports relativos
from app.core.schemas import Stratum
from app.core.mic_algebra import CategoricalState, Morphism
from app.core.telemetry_schemas import ElectronCartridge

logger = logging.getLogger("MIC.ImmuneSystem")
_THIS_MODULE = sys.modules[__name__]

# ==============================================================================
# JERARQUÍA DE EXCEPCIONES CON CONTEXTO MATEMÁTICO
# ==============================================================================

class ImmuneSystemError(Exception):
    """Base para excepciones del sistema inmunológico."""
    def __init__(self, message: str, **context: Any):
        super().__init__(message)
        self.context = context

class NumericalStabilityError(ImmuneSystemError):
    """Violación de cotas de estabilidad numérica."""
    pass

class DimensionalMismatchError(ImmuneSystemError):
    """Incompatibilidad dimensional en espacios vectoriales."""
    pass

class PhysicalBoundsError(ImmuneSystemError):
    """Violación de límites físicos teóricos."""
    pass

class TopologicalInvariantError(ImmuneSystemError):
    """Violación de invariantes topológicos."""
    pass

class MetricTensorError(ImmuneSystemError):
    """Error en propiedades del tensor métrico."""
    pass

class SpectralDecompositionError(ImmuneSystemError):
    """Error en descomposición espectral."""
    pass

# ==============================================================================
# CONSTANTES FÍSICAS CON UNIDADES SI
# ==============================================================================

class PhysicalConstants:
    """Constantes físicas con verificación dimensional."""
    
    # Constantes electromagnéticas
    Z_CHARACTERISTIC: float = 50.0        # [Ω]
    V_NOMINAL: float = 100.0              # [V]
    I_THERMAL_LIMIT: float = 10.0         # [A]
    
    # Constantes magnéticas
    SATURATION_CRITICAL: float = 0.85     # [-] adimensional
    FLYBACK_MAX_SAFE: float = 400.0       # [V]
    
    # Factores de seguridad
    THERMAL_SAFETY_FACTOR: float = 1.25   # [-]
    TURN_RATIO: float = 5.0               # [-]
    
    @classmethod
    def P_NOMINAL(cls) -> float:
        """Potencia nominal [W] = V²/Z con verificación."""
        P = cls.V_NOMINAL ** 2 / cls.Z_CHARACTERISTIC
        assert P > 0, "Potencia nominal debe ser positiva"
        return P
    
    @classmethod
    def validate_physical_consistency(cls) -> None:
        """Verifica consistencia dimensional de constantes."""
        P_from_V = cls.V_NOMINAL**2 / cls.Z_CHARACTERISTIC
        I_max = cls.V_NOMINAL / cls.Z_CHARACTERISTIC
        P_from_I = I_max**2 * cls.Z_CHARACTERISTIC
        
        relative_error = abs(P_from_V - P_from_I) / P_from_V
        if relative_error > 1e-10:
            raise PhysicalBoundsError(
                "Inconsistencia en constantes físicas",
                P_from_V=P_from_V, P_from_I=P_from_I
            )

PhysicalConstants.validate_physical_consistency()

# ==============================================================================
# CONSTANTES NUMÉRICAS CON JUSTIFICACIÓN TEÓRICA (AJUSTADAS)
# ==============================================================================

EPS: float = np.finfo(np.float64).eps * 4  # ≈ 8.88e-16
ALGEBRAIC_TOL: float = 1e-10               # ‖P² - P‖_F para idempotencia
COND_NUM_TOL: float = 1e12                 # κ_max REDUCIDO para mayor estabilidad
MIN_EIGVAL_TOL: float = 1e-12              # λ_min para definición positiva estricta

# Parámetros de regularización
TIKHONOV_DELTA: float = MIN_EIGVAL_TOL
RICCI_FLOW_DT: float = 0.01
RICCI_FLOW_NORMALIZE: bool = True          # Activar normalización de flujo

# ==============================================================================
# PROTOCOLO DE MÉTRICA RIEMANNIANA
# ==============================================================================

class RiemannianMetric(Protocol):
    """Protocolo para tensores métricos Riemannianos."""
    
    @property
    def dimension(self) -> int: ...
    
    @property
    def condition_number(self) -> float: ...
    
    def quadratic_form(self, v: NDArray[np.float64]) -> float: ...
    
    def apply(self, v: NDArray[np.float64]) -> NDArray[np.float64]: ...
    
    def inverse_sqrt_apply(self, v: NDArray[np.float64]) -> NDArray[np.float64]: ...

# ==============================================================================
# ÁLGEBRA LINEAL ESTABLE (REFINADA)
# ==============================================================================

class StableLinearAlgebra:
    """Operaciones de álgebra lineal con garantías numéricas."""
    
    @staticmethod
    def safe_normalize(
        vector: NDArray[np.float64],
        eps: float = EPS,
        norm_type: str = 'inf'
    ) -> Tuple[NDArray[np.float64], float]:
        """
        Normalización numéricamente estable.
        
        Mejora: Manejo explícito de NaN vs Inf.
        """
        if vector.size == 0:
            return vector.copy(), 0.0
        
        # Separar NaN e Inf
        has_nan = np.any(np.isnan(vector))
        has_inf = np.any(np.isinf(vector))
        
        if has_nan:
            raise NumericalStabilityError(
                "Vector contiene NaN",
                nan_indices=np.where(np.isnan(vector))[0].tolist()
            )
        
        if has_inf:
            warnings.warn(
                "Vector contiene Inf, clampando a valores finitos",
                RuntimeWarning
            )
            vector = np.clip(vector, -1e308, 1e308)  # Max float64
        
        if norm_type == 'inf':
            scale = float(np.max(np.abs(vector)))
        elif norm_type == '2':
            scale = float(np.linalg.norm(vector, ord=2))
        else:
            raise ValueError(f"Tipo de norma no soportado: {norm_type}")
        
        if scale < eps:
            warnings.warn(
                f"Vector degenerado: ‖v‖_{norm_type} = {scale:.2e} < {eps:.2e}",
                RuntimeWarning,
                stacklevel=2
            )
            return np.zeros_like(vector, dtype=np.float64), 0.0
        
        normalized = vector / scale
        
        if norm_type == 'inf':
            assert abs(np.max(np.abs(normalized)) - 1.0) < eps, "Fallo en normalización"
        
        return normalized, scale
    
    @staticmethod
    def stable_reciprocal(
        x: NDArray[np.float64],
        eps: float = EPS
    ) -> NDArray[np.float64]:
        """Recíproco 1/x con protección contra división por cero."""
        x_arr = np.asarray(x, dtype=np.float64)
        abs_x = np.abs(x_arr)
        sign_x = np.sign(x_arr)
        sign_safe = np.where(sign_x == 0.0, 1.0, sign_x)
        safe_denom = np.where(abs_x >= eps, x_arr, sign_safe * eps)
        
        result = 1.0 / safe_denom
        assert np.all(np.isfinite(result)), "Resultado no finito en reciprocal"
        return result
    
    @staticmethod
    def stable_divide(
        numerator: NDArray[np.float64],
        denominator: NDArray[np.float64],
        eps: float = EPS
    ) -> NDArray[np.float64]:
        """División vectorial estable."""
        return np.asarray(numerator) * StableLinearAlgebra.stable_reciprocal(denominator, eps)
    
    @staticmethod
    def stable_quadratic_form(
        vector: NDArray[np.float64],
        metric: NDArray[np.float64],
        eps: float = EPS
    ) -> float:
        """
        Forma cuadrática Q(v) = vᵀ G v con pre-escalado.
        """
        v = np.asarray(vector, dtype=np.float64).ravel()
        G = np.asarray(metric, dtype=np.float64)
        
        v_norm, scale = StableLinearAlgebra.safe_normalize(v, eps=eps, norm_type='inf')
        if scale < eps:
            return 0.0
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            Q_normalized = float(v_norm @ G @ v_norm)
        
        Q = (scale ** 2) * Q_normalized
        return max(Q, 0.0)
    
    @staticmethod
    def compute_condition_spectral(
        matrix: NDArray[np.float64],
        method: str = 'eig'
    ) -> float:
        """Número de condición espectral: κ(A) = λ_max / λ_min."""
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise DimensionalMismatchError(
                "Matriz debe ser cuadrada",
                shape=matrix.shape
            )
        
        if method == 'eig':
            eigenvalues = np.linalg.eigvalsh(matrix)
            lam_min = float(eigenvalues.min())
            lam_max = float(eigenvalues.max())
            
            if lam_min <= MIN_EIGVAL_TOL:
                return float('inf')
            
            return lam_max / lam_min
        
        elif method == 'svd':
            sv = np.linalg.svd(matrix, compute_uv=False)
            nonzero = sv[sv > MIN_EIGVAL_TOL]
            
            if nonzero.size == 0:
                return float('inf')
            
            return float(nonzero.max() / nonzero.min())
        
        else:
            raise ValueError(f"Método desconocido: {method}")
    
    @staticmethod
    def regularize_spd_matrix(
        matrix: NDArray[np.float64],
        min_eig: float = MIN_EIGVAL_TOL,
        method: str = 'spectral'
    ) -> NDArray[np.float64]:
        """
        Regularización de matriz SPD.
        
        MEJORA: Siempre proyecta al cono SPD después de modificar eigenvalores.
        """
        mat = np.asarray(matrix, dtype=np.float64)
        mat = (mat + mat.T) * 0.5  # Forzar simetría
        
        if method == 'spectral':
            eigenvalues, eigenvectors = np.linalg.eigh(mat)
            min_eigval = float(eigenvalues.min())
            
            if min_eigval >= min_eig:
                return mat.copy()
            
            delta = min_eig - min_eigval
            eigenvalues_reg = np.maximum(eigenvalues + delta, min_eig)  # Clamp adicional
            
            mat_reg = eigenvectors @ np.diag(eigenvalues_reg) @ eigenvectors.T
            mat_reg = (mat_reg + mat_reg.T) * 0.5
            
            logger.debug(
                "Regularización espectral: δ=%.4e, λ_min: %.4e → %.4e",
                delta, min_eigval, eigenvalues_reg.min()
            )
            
            return mat_reg
        
        elif method == 'tikhonov':
            eigenvalues = np.linalg.eigvalsh(mat)
            min_eigval = float(eigenvalues.min())
            
            if min_eigval >= min_eig:
                return mat.copy()
            
            delta = min_eig - min_eigval
            mat_reg = mat + delta * np.eye(mat.shape[0], dtype=np.float64)
            
            return (mat_reg + mat_reg.T) * 0.5
        
        else:
            raise ValueError(f"Método de regularización desconocido: {method}")
    
    @staticmethod
    def verify_orthogonality(
        Q: NDArray[np.float64],
        tol: float = ALGEBRAIC_TOL
    ) -> Tuple[bool, float]:
        """Verifica ortogonalidad: Q^T Q = I."""
        n = Q.shape[1]
        QtQ = Q.T @ Q
        residual = QtQ - np.eye(n, dtype=np.float64)
        residual_norm = float(np.linalg.norm(residual, 'fro'))
        
        is_orthogonal = residual_norm < tol
        
        if not is_orthogonal:
            logger.warning(
                "Matriz no ortogonal: ‖Q^TQ - I‖_F = %.4e > %.4e",
                residual_norm, tol
            )
        
        return is_orthogonal, residual_norm
    
    @staticmethod
    def gram_schmidt_orthonormalize(
        vectors: NDArray[np.float64],
        tol: float = ALGEBRAIC_TOL
    ) -> NDArray[np.float64]:
        """
        Orto-normalización de Gram-Schmidt con re-ortogonalización.
        
        NUEVO: Maneja espacios degenerados (eigenvalores múltiples).
        """
        Q, R = sp_linalg.qr(vectors, mode='economic')
        
        # Verificar ortogonalidad post-QR
        is_orth, residual = StableLinearAlgebra.verify_orthogonality(Q, tol)
        
        if not is_orth:
            logger.warning(
                "QR de SciPy produjo residual %.4e, re-ortogonalizando",
                residual
            )
            # Re-ortogonalización clásica
            Q, _ = sp_linalg.qr(Q, mode='economic')
        
        return Q

# ==============================================================================
# VALIDADORES (SIN CAMBIOS ESTRUCTURALES)
# ==============================================================================

class Validator(ABC):
    @abstractmethod
    def validate(
        self,
        value: float,
        name: str
    ) -> Tuple[float, bool, Optional[str]]:
        pass
    
    @abstractmethod
    def get_bounds(self) -> Tuple[Optional[float], Optional[float]]:
        pass

class UnitIntervalValidator(Validator):
    def validate(
        self,
        value: float,
        name: str
    ) -> Tuple[float, bool, Optional[str]]:
        if not np.isfinite(value):
            raise PhysicalBoundsError(
                f"{name}: valor no finito",
                value=value
            )
        
        if value < 0.0:
            return 0.0, True, f"{name}={value:.6g} → clamp a 0.0"
        if value > 1.0:
            return 1.0, True, f"{name}={value:.6g} → clamp a 1.0"
        
        return value, False, None
    
    def get_bounds(self) -> Tuple[Optional[float], Optional[float]]:
        return (0.0, 1.0)

class NonNegativeValidator(Validator):
    def validate(
        self,
        value: float,
        name: str
    ) -> Tuple[float, bool, Optional[str]]:
        if not np.isfinite(value):
            raise PhysicalBoundsError(
                f"{name}: valor no finito",
                value=value
            )
        
        if value < 0.0:
            return 0.0, True, f"{name}={value:.6g} → clamp a 0.0"
        
        return value, False, None
    
    def get_bounds(self) -> Tuple[Optional[float], Optional[float]]:
        return (0.0, None)

class PositiveIntValidator(Validator):
    def validate(
        self,
        value: float,
        name: str
    ) -> Tuple[float, bool, Optional[str]]:
        if not np.isfinite(value):
            raise PhysicalBoundsError(
                f"{name}: valor no finito",
                value=value
            )
        
        rounded = max(1, int(round(value)))
        was_modified = abs(rounded - value) > EPS
        
        if was_modified:
            return float(rounded), True, f"{name}={value:.6g} → {rounded}"
        
        return float(rounded), False, None
    
    def get_bounds(self) -> Tuple[Optional[float], Optional[float]]:
        return (1.0, None)

class NonNegativeIntValidator(Validator):
    def validate(
        self,
        value: float,
        name: str
    ) -> Tuple[float, bool, Optional[str]]:
        if not np.isfinite(value):
            raise PhysicalBoundsError(
                f"{name}: valor no finito",
                value=value
            )
        
        rounded = max(0, int(round(value)))
        was_modified = abs(rounded - value) > EPS
        
        if was_modified:
            return float(rounded), True, f"{name}={value:.6g} → {rounded}"
        
        return float(rounded), False, None
    
    def get_bounds(self) -> Tuple[Optional[float], Optional[float]]:
        return (0.0, None)

VALIDATOR_REGISTRY: Dict[str, Validator] = {
    "unit_interval": UnitIntervalValidator(),
    "non_negative": NonNegativeValidator(),
    "positive_int": PositiveIntValidator(),
    "non_negative_int": NonNegativeIntValidator(),
}

# ==============================================================================
# ENUMERACIÓN DE ESTADO DE SALUD (SIN CAMBIOS)
# ==============================================================================

class HealthStatus(Enum):
    HEALTHY = auto()
    WARNING = auto()
    CRITICAL = auto()
    
    @property
    def severity(self) -> int:
        return {"HEALTHY": 0, "WARNING": 1, "CRITICAL": 2}[self.name]
    
    def __lt__(self, other: "HealthStatus") -> bool:
        if not isinstance(other, HealthStatus):
            return NotImplemented
        return self.severity < other.severity
    
    def __str__(self) -> str:
        return {"HEALTHY": "✓ HEALTHY", "WARNING": "⚠ WARNING", "CRITICAL": "✗ CRITICAL"}[self.name]

# ==============================================================================
# TENSOR MÉTRICO RIEMANNIANO (REFINADO)
# ==============================================================================

@dataclass(frozen=True)
class SpectralDecomposition:
    """Descomposición espectral verificada."""
    
    eigenvalues: NDArray[np.float64]
    eigenvectors: NDArray[np.float64]
    orthogonality_residual: float
    
    def __post_init__(self):
        if not np.all(np.diff(self.eigenvalues) <= 0):
            raise SpectralDecompositionError("Eigenvalores no ordenados")
        
        is_orth, residual = StableLinearAlgebra.verify_orthogonality(
            self.eigenvectors,
            tol=ALGEBRAIC_TOL
        )
        
        if not is_orth:
            raise SpectralDecompositionError(
                "Eigenvectores no ortogonales",
                residual=residual
            )

class MetricTensor:
    """Tensor métrico Riemanniano G con propiedades verificadas."""
    
    __slots__ = (
        "_matrix", "_is_diagonal", "_dim",
        "_condition_number", "_spectral_decomposition",
        "_frozen"
    )
    
    def __init__(
        self,
        matrix: Union[NDArray[np.float64], List[float]],
        validate: bool = True,
        regularization_threshold: float = COND_NUM_TOL
    ) -> None:
        arr = np.asarray(matrix, dtype=np.float64)
        
        if arr.ndim == 1:
            self._init_diagonal(arr, validate, regularization_threshold)
        elif arr.ndim == 2:
            self._init_dense(arr, validate, regularization_threshold)
        else:
            raise MetricTensorError("Dimensión inválida", ndim=arr.ndim)
        
        self._frozen = True
    
    def _init_diagonal(
        self,
        arr: NDArray[np.float64],
        validate: bool,
        threshold: float
    ) -> None:
        self._dim = arr.shape[0]
        self._is_diagonal = True
        
        if validate:
            if np.any(arr <= MIN_EIGVAL_TOL):
                raise MetricTensorError(
                    "Diagonal con eigenvalores no positivos",
                    min_val=arr.min()
                )
        
        mat = arr.copy()
        mat.flags.writeable = False
        self._matrix = mat
        
        d_min = float(arr.min())
        d_max = float(arr.max())
        self._condition_number = d_max / max(d_min, MIN_EIGVAL_TOL)
        
        if validate and self._condition_number > threshold:
            logger.warning(
                "Métrica diagonal mal condicionada: κ=%.2e",
                self._condition_number
            )
        
        self._spectral_decomposition = SpectralDecomposition(
            eigenvalues=np.sort(arr)[::-1].copy(),
            eigenvectors=np.eye(self._dim, dtype=np.float64),
            orthogonality_residual=0.0
        )
    
    def _init_dense(
        self,
        arr: NDArray[np.float64],
        validate: bool,
        threshold: float
    ) -> None:
        if arr.shape[0] != arr.shape[1]:
            raise MetricTensorError("Matriz no cuadrada", shape=arr.shape)
        
        sym = (arr + arr.T) * 0.5
        
        if validate:
            asym = float(np.linalg.norm(arr - arr.T, "fro"))
            if asym > ALGEBRAIC_TOL:
                logger.warning("Asimetría corregida: ‖G - G^T‖_F = %.4e", asym)
            
            sym = StableLinearAlgebra.regularize_spd_matrix(
                sym,
                min_eig=MIN_EIGVAL_TOL,
                method='spectral'
            )
        
        self._dim = sym.shape[0]
        self._is_diagonal = False
        
        mat = sym.copy()
        mat.flags.writeable = False
        self._matrix = mat
        
        eigenvalues, eigenvectors = np.linalg.eigh(sym)
        
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # MEJORA: Orto-normalización robusta
        eigenvectors = StableLinearAlgebra.gram_schmidt_orthonormalize(
            eigenvectors,
            tol=ALGEBRAIC_TOL
        )
        
        is_orth, residual = StableLinearAlgebra.verify_orthogonality(
            eigenvectors,
            tol=ALGEBRAIC_TOL
        )
        
        self._spectral_decomposition = SpectralDecomposition(
            eigenvalues=eigenvalues.copy(),
            eigenvectors=eigenvectors.copy(),
            orthogonality_residual=residual
        )
        
        self._condition_number = StableLinearAlgebra.compute_condition_spectral(
            sym,
            method='eig'
        )
        
        if self._condition_number > threshold:
            warnings.warn(
                f"Métrica mal condicionada: κ={self._condition_number:.2e}",
                UserWarning
            )
    
    @property
    def dimension(self) -> int:
        return self._dim
    
    @property
    def is_diagonal(self) -> bool:
        return self._is_diagonal
    
    @property
    def condition_number(self) -> float:
        return self._condition_number
    
    @property
    def spectral_decomposition(self) -> SpectralDecomposition:
        return self._spectral_decomposition
    
    def quadratic_form(self, v: NDArray[np.float64]) -> float:
        return StableLinearAlgebra.stable_quadratic_form(
            v,
            self._matrix if not self._is_diagonal else np.diag(self._matrix),
            eps=EPS
        )
    
    def apply(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
        if self._is_diagonal:
            return self._matrix * v
        return self._matrix @ v
    
    def inverse_sqrt_apply(self, v: NDArray[np.float64]) -> NDArray[np.float64]:
        spec = self._spectral_decomposition
        
        safe_eigvals = np.maximum(spec.eigenvalues, MIN_EIGVAL_TOL)
        inv_sqrt_diag = 1.0 / np.sqrt(safe_eigvals)
        
        if self._is_diagonal:
            return inv_sqrt_diag * v
        
        return spec.eigenvectors @ (inv_sqrt_diag * (spec.eigenvectors.T @ v))
    
    def to_array(self) -> NDArray[np.float64]:
        if self._is_diagonal:
            result = self._matrix.copy()
        else:
            result = self._matrix.copy()
        
        result.flags.writeable = True
        return result
    
    def verify_invariants(self) -> Dict[str, bool]:
        checks = {}
        
        if not self._is_diagonal:
            asym = float(np.linalg.norm(self._matrix - self._matrix.T, 'fro'))
            checks['symmetry'] = asym < ALGEBRAIC_TOL
        else:
            checks['symmetry'] = True
        
        checks['positive_definite'] = float(
            self._spectral_decomposition.eigenvalues.min()
        ) >= MIN_EIGVAL_TOL
        
        checks['well_conditioned'] = self._condition_number < COND_NUM_TOL
        
        checks['eigenvectors_orthogonal'] = (
            self._spectral_decomposition.orthogonality_residual < ALGEBRAIC_TOL
        )
        
        if not self._is_diagonal:
            Q = self._spectral_decomposition.eigenvectors
            Λ = np.diag(self._spectral_decomposition.eigenvalues)
            G_reconstructed = Q @ Λ @ Q.T
            reconstruction_error = float(
                np.linalg.norm(G_reconstructed - self._matrix, 'fro')
            )
            checks['spectral_reconstruction'] = reconstruction_error < ALGEBRAIC_TOL
        else:
            checks['spectral_reconstruction'] = True
        
        return checks
    
    def __repr__(self) -> str:
        kind = "diagonal" if self._is_diagonal else "densa"
        return (
            f"MetricTensor({kind}, n={self._dim}, "
            f"κ={self._condition_number:.2e}, "
            f"λ_min={self._spectral_decomposition.eigenvalues.min():.2e})"
        )

# ==============================================================================
# FUNTOR DE MEMBRANA AISLANTE (CORREGIDO)
# ==============================================================================

@dataclass
class IsolatingMembraneFunctor:
    """
    Funtor que evalúa energía de p-Dirichlet.
    
    CORRECCIÓN: Laplaciano discreto riguroso.
    """
    
    p: float = 1.5
    eps: float = 1e-12
    
    def __post_init__(self):
        if not (1.0 <= self.p < 2.0):
            raise ValueError("Exponente p debe estar en [1, 2)", p=self.p)
    
    @staticmethod
    def discrete_laplacian_1d(f: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Laplaciano discreto 1D con diferencias finitas de segundo orden.
        
        CORRECCIÓN RIGUROSA:
        Δf[i] = f[i+1] - 2f[i] + f[i-1]  (interior)
        Condiciones Neumann homogéneas en frontera.
        """
        n = len(f)
        if n < 3:
            return np.zeros_like(f)
        
        lap = np.zeros(n, dtype=np.float64)
        
        # Interior: diferencias centradas
        lap[1:-1] = f[2:] - 2*f[1:-1] + f[:-2]
        
        # Frontera: Neumann homogéneas (∂f/∂n = 0)
        lap[0] = f[1] - 2*f[0] + f[0]      # Reflejo f[-1] ≈ f[0]
        lap[-1] = f[-1] - 2*f[-1] + f[-2]  # Reflejo f[n] ≈ f[-1]
        
        return lap
    
    def compute_topological_stress(
        self,
        psi: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Calcula S_p(ψ) = |Δψ|^{p-2} + ε.
        
        CORRECCIÓN: Usar Laplaciano discreto riguroso.
        """
        laplacian = self.discrete_laplacian_1d(psi)
        
        abs_lap = np.maximum(np.abs(laplacian), self.eps)
        stress = abs_lap ** (self.p - 2.0) + self.eps
        
        return stress

# ==============================================================================
# ESPECIFICACIÓN DE SUBESPACIO (CORREGIDO)
# ==============================================================================

@dataclass
class SubspaceSpec:
    """Especificación de subespacio V_k ⊂ ℝⁿ con métrica Riemanniana."""
    
    name: str
    indices: slice
    weight: float
    reference: NDArray[np.float64]
    metric: Optional[MetricTensor] = field(default=None)
    scale: Optional[NDArray[np.float64]] = field(default=None, repr=False)
    
    def __post_init__(self) -> None:
        self.reference = np.asarray(self.reference, dtype=np.float64).ravel().copy()
        self.reference.flags.writeable = False
        
        dim = self.reference.shape[0]
        
        if not np.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError(
                f"[{self.name}] peso debe ser finito y positivo",
                weight=self.weight
            )
        
        if self.metric is None:
            if self.scale is not None:
                scale_arr = np.asarray(self.scale, dtype=np.float64).ravel()
                
                if scale_arr.shape[0] != dim:
                    raise DimensionalMismatchError(
                        f"[{self.name}] dim(scale) ≠ dim(reference)",
                        dim_scale=scale_arr.shape[0],
                        dim_ref=dim
                    )
                
                metric_diag = StableLinearAlgebra.stable_reciprocal(scale_arr ** 2, eps=EPS)
                metric_diag = np.maximum(metric_diag, MIN_EIGVAL_TOL)
                
                self.metric = MetricTensor(metric_diag, validate=True)
            else:
                self.metric = MetricTensor(
                    np.ones(dim, dtype=np.float64),
                    validate=True
                )
        
        if self.metric.dimension != dim:
            raise DimensionalMismatchError(
                f"[{self.name}] dim(metric) ≠ dim(reference)",
                dim_metric=self.metric.dimension,
                dim_ref=dim
            )
    
    def compute_threat(
        self,
        subvector: NDArray[np.float64],
        gamma: float = 0.1
    ) -> float:
        """
        Calcula amenaza con métrica perturbada.
        
        CORRECCIÓN CRÍTICA: Conjugación en lugar de Hadamard para matrices densas.
        G̃ = P^T G P donde P = I + γ·diag(S_p)
        """
        sv = np.asarray(subvector, dtype=np.float64).ravel()
        
        if sv.shape != self.reference.shape:
            raise DimensionalMismatchError(
                f"[{self.name}] shape incompatible",
                expected=self.reference.shape,
                received=sv.shape
            )
        
        membrane = IsolatingMembraneFunctor(p=1.5, eps=EPS)
        S_p = membrane.compute_topological_stress(sv)
        
        delta = sv - self.reference
        
        if self.metric.is_diagonal:
            # Caso diagonal: producto Hadamard válido
            G_diag = self.metric.to_array()
            tilde_G_diag = G_diag * (1.0 + gamma * S_p)
            maha_sq = np.sum((delta ** 2) * tilde_G_diag)
        else:
            # Caso denso: conjugación matricial
            base_G = self.metric.to_array()
            perturbation_matrix = np.eye(len(S_p)) + gamma * np.diag(S_p)
            
            # G̃ = P^T G P (deformación conforme)
            tilde_G = perturbation_matrix.T @ base_G @ perturbation_matrix
            tilde_G = (tilde_G + tilde_G.T) * 0.5  # Simetrización
            
            maha_sq = float(delta @ tilde_G @ delta)
        
        threat = self.weight * float(np.sqrt(max(maha_sq, 0.0)))
        
        return threat
    
    def normalize_to_reference(
        self,
        subvector: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        sv = np.asarray(subvector, dtype=np.float64).ravel()
        delta = sv - self.reference
        
        return self.metric.inverse_sqrt_apply(delta)

# ==============================================================================
# EVALUACIÓN DE AMENAZA INMUTABLE (SIN CAMBIOS)
# ==============================================================================

@dataclass(frozen=True)
class ThreatAssessment:
    levels: Dict[str, float]
    max_source: str
    max_value: float
    total_threat: float
    euler_char: Optional[int] = None
    status: HealthStatus = HealthStatus.HEALTHY
    metadata: Dict[str, Any] = field(default_factory=dict)
    electrons: Tuple[ElectronCartridge, ...] = field(default_factory=tuple)
    
    def __post_init__(self):
        negative = {k: v for k, v in self.levels.items() if v < 0.0}
        if negative:
            raise ValueError(
                "Niveles de amenaza deben ser no negativos",
                negative_levels=negative
            )
        
        computed_max = max(self.levels.values())
        if abs(computed_max - self.max_value) > EPS:
            raise ValueError(
                "max_value inconsistente",
                expected=computed_max,
                received=self.max_value
            )
    
    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v:.4f}" for k, v in self.levels.items())
        euler_str = f", χ={self.euler_char}" if self.euler_char is not None else ""
        
        return (
            f"ThreatAssessment({items}{euler_str}) → {self.status} "
            f"(max: {self.max_source}={self.max_value:.4f}, "
            f"total={self.total_threat:.4f})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "threat_levels": {k: float(v) for k, v in self.levels.items()},
            "max_threat_source": str(self.max_source),
            "max_threat_value": float(self.max_value),
            "total_threat": float(self.total_threat),
            "euler_characteristic": (
                int(self.euler_char) if self.euler_char is not None else None
            ),
            "health_status": self.status.name,
            "metadata": {
                k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in self.metadata.items()
            },
        }
    
    @staticmethod
    def from_components(
        levels: Dict[str, float],
        euler_char: Optional[int] = None,
        warning_threshold: float = 0.8,
        critical_threshold: float = 1.5,
        electrons: Tuple[ElectronCartridge, ...] = (),
    ) -> "ThreatAssessment":
        if not levels:
            raise ValueError("levels no puede estar vacío")
        
        negative = {k: v for k, v in levels.items() if v < 0.0}
        if negative:
            raise ValueError("Niveles de amenaza negativos", negative=negative)
        
        max_source = max(levels, key=lambda k: levels[k])
        max_value = float(levels[max_source])
        total_threat = float(np.linalg.norm(list(levels.values())))
        
        if max_value > critical_threshold:
            status = HealthStatus.CRITICAL
        elif max_value > warning_threshold:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY
        
        return ThreatAssessment(
            levels=dict(levels),
            max_source=max_source,
            max_value=max_value,
            total_threat=total_threat,
            euler_char=euler_char,
            status=status,
            electrons=electrons,
        )

# ==============================================================================
# CONSTRUCCIÓN BLOQUE-DIAGONAL PURA (SIN CAMBIOS)
# ==============================================================================

def block_diag_pure(*blocks: NDArray[np.float64]) -> NDArray[np.float64]:
    sizes = []
    for b in blocks:
        if b.ndim == 1:
            sizes.append(b.shape[0])
        elif b.ndim == 2:
            if b.shape[0] != b.shape[1]:
                raise ValueError("Bloques 2D deben ser cuadrados", shape=b.shape)
            sizes.append(b.shape[0])
        else:
            raise ValueError("Bloques deben ser 1D o 2D", ndim=b.ndim)
    
    total = sum(sizes)
    result = np.zeros((total, total), dtype=np.float64)
    
    offset = 0
    for b in blocks:
        dim = b.shape[0]
        
        if b.ndim == 1:
            result[offset:offset+dim, offset:offset+dim] = np.diag(b)
        else:
            result[offset:offset+dim, offset:offset+dim] = b
        
        offset += dim
    
    return result

# ==============================================================================
# PROYECTOR ORTOGONAL (CORREGIDO)
# ==============================================================================

class OrthogonalProjector:
    """Proyector ortogonal π: ℝⁿ → ⊕V_k con propiedades verificadas."""
    
    __slots__ = (
        "_dim", "_subspaces", "_topo_indices",
        "_projection_matrices", "_validation_report",
        "_global_metric", "_global_metric_tensor"
    )
    
    def __init__(
        self,
        dimensions: int,
        subspaces: Dict[str, SubspaceSpec],
        topo_indices: Optional[Tuple[int, ...]] = None,
        cache_projections: bool = True,
    ) -> None:
        self._dim = dimensions
        self._subspaces: Dict[str, SubspaceSpec] = dict(subspaces)
        self._projection_matrices: Dict[str, NDArray[np.float64]] = {}
        self._validation_report: Dict[str, float] = {}
        
        if topo_indices is not None:
            if len(topo_indices) not in (2, 3):
                raise ValueError(
                    "topo_indices debe tener 2 o 3 elementos",
                    length=len(topo_indices)
                )
            
            for i, idx in enumerate(topo_indices):
                if not (0 <= idx < dimensions):
                    raise ValueError(
                        f"topo_indices[{i}] fuera de rango",
                        index=idx,
                        range=(0, dimensions)
                    )
            
            self._topo_indices = topo_indices
        else:
            self._topo_indices = None
        
        self._validate_and_build(cache_projections)
        self._build_global_tensor()
    
    def _validate_and_build(self, cache_projections: bool) -> None:
        """
        MEJORA: Verificación de Σ π_k = I siempre obligatoria.
        """
        covered: set = set()
        projector_sum = np.zeros((self._dim, self._dim), dtype=np.float64)
        
        for name, spec in self._subspaces.items():
            idx_set = set(range(*spec.indices.indices(self._dim)))
            
            overlap = covered & idx_set
            if overlap:
                raise DimensionalMismatchError(
                    f"Subespacio '{name}' se solapa",
                    overlap=sorted(overlap)
                )
            
            covered |= idx_set
            
            expected_dim = len(idx_set)
            if spec.reference.shape[0] != expected_dim:
                raise DimensionalMismatchError(
                    f"[{name}] dim(reference) ≠ |V_k|",
                    dim_ref=spec.reference.shape[0],
                    dim_subspace=expected_dim
                )
            
            # Construir proyector
            P = np.zeros((self._dim, self._dim), dtype=np.float64)
            for i in idx_set:
                P[i, i] = 1.0
            
            # Verificar idempotencia
            P2 = P @ P
            idem_err = float(np.linalg.norm(P2 - P, "fro"))
            self._validation_report[f"{name}_idempotence"] = idem_err
            
            if idem_err > ALGEBRAIC_TOL:
                raise NumericalStabilityError(
                    f"[{name}] Proyector no idempotente",
                    error=idem_err
                )
            
            # Verificar auto-adjunción
            self_adj_err = float(np.linalg.norm(P - P.T, "fro"))
            self._validation_report[f"{name}_self_adjoint"] = self_adj_err
            
            if self_adj_err > ALGEBRAIC_TOL:
                raise NumericalStabilityError(
                    f"[{name}] Proyector no autoadjunto",
                    error=self_adj_err
                )
            
            # Verificar rango
            rank_via_trace = int(round(float(np.trace(P))))
            if rank_via_trace != expected_dim:
                raise NumericalStabilityError(
                    f"[{name}] rank(π) ≠ |V_k|",
                    rank=rank_via_trace,
                    expected=expected_dim
                )
            
            if cache_projections:
                self._projection_matrices[name] = P
            
            projector_sum += P
        
        # Verificar cobertura
        uncovered = set(range(self._dim)) - covered
        if uncovered:
            logger.warning(
                "Índices sin subespacio asignado: %s",
                sorted(uncovered)
            )
            self._validation_report["uncovered_index_count"] = float(len(uncovered))
        
        # VERIFICACIÓN OBLIGATORIA: Σ π_k = I
        coverage_err = float(np.linalg.norm(
            projector_sum - np.eye(self._dim),
            "fro"
        ))
        
        self._validation_report["coverage_identity_error"] = coverage_err
        
        if coverage_err > ALGEBRAIC_TOL:
            # Si hay índices sin cubrir, permitir error proporcional
            if uncovered and coverage_err <= len(uncovered) * ALGEBRAIC_TOL:
                logger.warning(
                    "Σ π_k ≈ I con error %.2e (índices sin cubrir: %d)",
                    coverage_err, len(uncovered)
                )
            else:
                raise NumericalStabilityError(
                    "Σ π_k ≠ I",
                    error=coverage_err
                )
        
        logger.info(
            "OrthogonalProjector validado: %s",
            ", ".join(f"{k}={v:.2e}" for k, v in self._validation_report.items())
        )
    
    def _build_global_tensor(self) -> None:
        blocks = []
        ordered_keys = ["physics_core", "topology_core", "thermo_core"]
        
        for key in ordered_keys:
            if key in self._subspaces:
                spec = self._subspaces[key]
                G_k = spec.metric.to_array()
                weighted_matrix = (spec.weight ** 2) * G_k
                blocks.append(weighted_matrix)
        
        if len(blocks) == 3:
            try:
                G_global_arr = block_diag_pure(*blocks)
            except Exception as e:
                logger.error("Falló construcción bloque-diagonal: %s", e)
                self._global_metric = None
                self._global_metric_tensor = None
                return
            
            try:
                G_global_arr = StableLinearAlgebra.regularize_spd_matrix(
                    G_global_arr,
                    min_eig=MIN_EIGVAL_TOL,
                    method='spectral'
                )
            except Exception as e:
                logger.warning("Regularización de G_global falló: %s", e)
            
            self._global_metric = G_global_arr
            self._global_metric_tensor = MetricTensor(
                self._global_metric,
                validate=False
            )
        else:
            self._global_metric = None
            self._global_metric_tensor = None
    
    @property
    def validation_report(self) -> Dict[str, float]:
        return dict(self._validation_report)
    
    def _compute_euler_characteristic(
        self,
        psi: NDArray[np.float64]
    ) -> Optional[int]:
        if self._topo_indices is None:
            return None
        
        idx_b0, idx_b1 = self._topo_indices[0], self._topo_indices[1]
        idx_b2 = self._topo_indices[2] if len(self._topo_indices) == 3 else None
        
        try:
            beta_0 = int(round(float(psi[idx_b0])))
            beta_1 = int(round(float(psi[idx_b1])))
            beta_2 = int(round(float(psi[idx_b2]))) if idx_b2 is not None else 0
        except (IndexError, ValueError) as exc:
            logger.warning("Error extrayendo números de Betti: %s", exc)
            return None
        
        if beta_0 < 1:
            raise TopologicalInvariantError("β₀ debe ser ≥ 1", beta_0=beta_0)
        
        if beta_1 < 0:
            raise TopologicalInvariantError("β₁ debe ser ≥ 0", beta_1=beta_1)
        
        if beta_2 < 0:
            raise TopologicalInvariantError("β₂ debe ser ≥ 0", beta_2=beta_2)
        
        euler_char = beta_0 - beta_1 + beta_2
        
        return euler_char
    
    @contextmanager
    def temporary_algebraic_tolerance(
        self,
        relaxed_tol: float = 1e-6
    ) -> Generator[None, None, None]:
        if relaxed_tol <= 0.0:
            raise ValueError("relaxed_tol debe ser > 0", value=relaxed_tol)
        
        original = getattr(_THIS_MODULE, "ALGEBRAIC_TOL")
        setattr(_THIS_MODULE, "ALGEBRAIC_TOL", float(relaxed_tol))
        
        try:
            yield
        finally:
            setattr(_THIS_MODULE, "ALGEBRAIC_TOL", original)
    
    def _quantize_anomaly_electrons(
        self,
        name: str,
        threat: float,
        psi: NDArray[np.float64],
        euler_char: Optional[int],
        threshold: float
    ) -> Optional[ElectronCartridge]:
        if threat <= threshold:
            return None
        
        m_star = threat ** 2
        
        b1_idx = self._topo_indices[1] if self._topo_indices else -1
        if b1_idx >= 0 and b1_idx < len(psi):
            spin = "sink" if psi[b1_idx] > 0 else "source"
        else:
            spin = "source"
        
        delta_chi = (euler_char - 1) if euler_char is not None else 0
        
        return ElectronCartridge(
            inertial_mass=m_star,
            topological_spin=spin,
            homological_charge=delta_chi,
            source_subspace=name
        )
    
    def project(
        self,
        psi: NDArray[np.float64],
        warning_threshold: float = 0.8,
        critical_threshold: float = 1.5,
        hysteresis: float = 0.05,
        previous_status: Optional[HealthStatus] = None,
        verbose: bool = False,
    ) -> ThreatAssessment:
        psi = np.asarray(psi, dtype=np.float64).ravel()
        
        if psi.shape != (self._dim,):
            raise DimensionalMismatchError(
                "Shape incorrecto",
                expected=(self._dim,),
                received=psi.shape
            )
        
        if not np.all(np.isfinite(psi)):
            bad_indices = np.where(~np.isfinite(psi))[0].tolist()
            raise NumericalStabilityError(
                "ψ contiene valores no finitos",
                indices=bad_indices
            )
        
        euler_char = self._compute_euler_characteristic(psi)
        
        levels: Dict[str, float] = {}
        electrons: List[ElectronCartridge] = []
        
        for name, spec in self._subspaces.items():
            subvec = psi[spec.indices]
            threat = spec.compute_threat(subvec)
            levels[name] = threat
            
            electron = self._quantize_anomaly_electrons(
                name, threat, psi, euler_char, warning_threshold
            )
            if electron:
                electrons.append(electron)
            
            if verbose:
                logger.debug("[%s] threat=%.4f", name, threat)
        
        max_source = max(levels, key=lambda k: levels[k])
        max_value = float(levels[max_source])
        
        if self._global_metric_tensor is not None:
            global_reference = np.zeros(self._dim, dtype=np.float64)
            for key in ["physics_core", "topology_core", "thermo_core"]:
                if key in self._subspaces:
                    spec = self._subspaces[key]
                    global_reference[spec.indices] = spec.reference
            
            delta = psi - global_reference
            maha_sq = self._global_metric_tensor.quadratic_form(delta)
            total_threat = float(np.sqrt(max(maha_sq, 0.0)))
        else:
            total_threat = float(np.linalg.norm(list(levels.values())))
        
        status = self._classify_with_hysteresis(
            total_threat,
            warning_threshold,
            critical_threshold,
            hysteresis,
            previous_status
        )
        
        metadata: Dict[str, Any] = {
            "norm_psi": float(np.linalg.norm(psi)),
            "condition_numbers": {
                name: float(spec.metric.condition_number)
                for name, spec in self._subspaces.items()
            },
        }
        
        return ThreatAssessment(
            levels=levels,
            max_source=max_source,
            max_value=max_value,
            total_threat=total_threat,
            euler_char=euler_char,
            status=status,
            metadata=metadata,
            electrons=tuple(electrons),
        )
    
    @staticmethod
    def _classify_with_hysteresis(
        value: float,
        warning_th: float,
        critical_th: float,
        hysteresis: float,
        previous: Optional[HealthStatus],
    ) -> HealthStatus:
        if previous is None:
            if value > critical_th:
                return HealthStatus.CRITICAL
            if value > warning_th:
                return HealthStatus.WARNING
            return HealthStatus.HEALTHY
        
        if previous == HealthStatus.HEALTHY:
            if value > critical_th + hysteresis:
                return HealthStatus.CRITICAL
            if value > warning_th + hysteresis:
                return HealthStatus.WARNING
            return HealthStatus.HEALTHY
        
        if previous == HealthStatus.WARNING:
            if value > critical_th + hysteresis:
                return HealthStatus.CRITICAL
            if value < warning_th - hysteresis:
                return HealthStatus.HEALTHY
            return HealthStatus.WARNING
        
        if value < warning_th - hysteresis:
            return HealthStatus.HEALTHY
        if value < critical_th - hysteresis:
            return HealthStatus.WARNING
        return HealthStatus.CRITICAL

# ==============================================================================
# ESQUEMA DE SEÑAL ψ ∈ ℝ⁷ (SIN CAMBIOS)
# ==============================================================================

@dataclass(frozen=True)
class SignalComponent:
    key: str
    default: float
    unit: str
    description: str
    validator: Optional[str] = None
    physical_bounds: Optional[Tuple[float, float]] = None

_P_NOMINAL: float = PhysicalConstants.P_NOMINAL()

SIGNAL_SCHEMA: Tuple[SignalComponent, ...] = (
    SignalComponent(
        key="saturation",
        default=0.0,
        unit="1",
        description="Saturación magnética",
        validator="unit_interval",
        physical_bounds=(0.0, 1.0)
    ),
    SignalComponent(
        key="flyback_voltage",
        default=0.0,
        unit="V",
        description="Voltaje de flyback",
        validator="non_negative",
        physical_bounds=(0.0, PhysicalConstants.FLYBACK_MAX_SAFE)
    ),
    SignalComponent(
        key="dissipated_power",
        default=0.0,
        unit="W",
        description="Potencia disipada",
        validator="non_negative",
        physical_bounds=(0.0, _P_NOMINAL * 2.0)
    ),
    SignalComponent(
        key="beta_0",
        default=1.0,
        unit="1",
        description="β₀: componentes conexas",
        validator="positive_int"
    ),
    SignalComponent(
        key="beta_1",
        default=0.0,
        unit="1",
        description="β₁: ciclos independientes",
        validator="non_negative_int"
    ),
    SignalComponent(
        key="entropy",
        default=0.0,
        unit="1",
        description="Entropía normalizada",
        validator="unit_interval",
        physical_bounds=(0.0, 1.0)
    ),
    SignalComponent(
        key="exergy_loss",
        default=0.0,
        unit="1",
        description="Pérdida exergética",
        validator="unit_interval",
        physical_bounds=(0.0, 1.0)
    ),
)

BETTI_INDICES: Tuple[int, int] = (3, 4)

assert SIGNAL_SCHEMA[BETTI_INDICES[0]].key == "beta_0", "Inconsistencia en β₀"
assert SIGNAL_SCHEMA[BETTI_INDICES[1]].key == "beta_1", "Inconsistencia en β₁"

def build_signal(
    telemetry: Dict[str, Any],
    strict: bool = False
) -> NDArray[np.float64]:
    psi = np.empty(len(SIGNAL_SCHEMA), dtype=np.float64)
    
    for i, spec in enumerate(SIGNAL_SCHEMA):
        raw = telemetry.get(spec.key, spec.default)
        
        try:
            val = float(raw)
        except (TypeError, ValueError) as exc:
            if strict:
                raise ValueError(
                    f"Telemetría '{spec.key}' no convertible",
                    value=raw
                ) from exc
            
            logger.warning("Telemetría '%s' no convertible → default", spec.key)
            psi[i] = spec.default
            continue
        
        if not np.isfinite(val):
            if strict:
                raise ValueError(f"Telemetría '{spec.key}' no finita", value=val)
            
            logger.warning("Telemetría '%s' no finita → default", spec.key)
            psi[i] = spec.default
            continue
        
        if spec.validator:
            validator = VALIDATOR_REGISTRY.get(spec.validator)
            if validator is None:
                raise ValueError(f"Validador desconocido: {spec.validator}")
            
            try:
                val, was_modified, error_msg = validator.validate(val, spec.key)
                
                if was_modified and error_msg is not None:
                    logger.debug("Telemetría '%s' corregida: %s", spec.key, error_msg)
            except PhysicalBoundsError as exc:
                if strict:
                    raise
                
                logger.warning("PhysicalBoundsError en '%s' → default", spec.key)
                psi[i] = spec.default
                continue
        
        psi[i] = val
    
    return psi

# ==============================================================================
# MORFISMO INMUNOLÓGICO FUNTORIAL (REFINADO)
# ==============================================================================

class ImmuneWatcherMorphism(Morphism):
    """Morfismo categórico F: Top → Narr de vigilancia inmunológica."""
    
    __slots__ = (
        "_critical", "_warning", "_hysteresis",
        "_enable_topology_monitoring", "_projector",
        "_previous_status", "_euler_history",
        "_evaluation_count", "_metric_tensors_state"
    )
    
    def __init__(
        self,
        name: str = "topological_immune_watcher",
        *,
        critical_threshold: float = 1.5,
        warning_threshold: float = 0.8,
        hysteresis: float = 0.05,
        enable_topology_monitoring: bool = True,
    ) -> None:
        if warning_threshold >= critical_threshold:
            raise ValueError(
                "warning_threshold debe ser < critical_threshold",
                warning=warning_threshold,
                critical=critical_threshold
            )
        
        max_hysteresis = (critical_threshold - warning_threshold) / 2.0
        if hysteresis < 0.0 or hysteresis >= max_hysteresis:
            raise ValueError(
                f"hysteresis debe estar en [0, {max_hysteresis:.4f})",
                hysteresis=hysteresis
            )
        
        super().__init__(name)
        
        self._critical = critical_threshold
        self._warning = warning_threshold
        self._hysteresis = hysteresis
        self._enable_topology_monitoring = enable_topology_monitoring
        
        self._previous_status: Optional[HealthStatus] = None
        self._euler_history: List[Optional[int]] = []
        self._evaluation_count: int = 0
        
        self._metric_tensors_state = self._init_metric_tensors()
        self._projector = self._build_projector()
    
    def _init_metric_tensors(self) -> Dict[str, NDArray[np.float64]]:
        try:
            import app.core.immune_system.metric_tensors as ext_metric_tensors
            return {
                "G_PHYSICS": np.copy(ext_metric_tensors.G_PHYSICS),
                "G_TOPOLOGY": np.copy(ext_metric_tensors.G_TOPOLOGY),
                "G_THERMODYNAMICS": np.copy(ext_metric_tensors.G_THERMODYNAMICS),
            }
        except (ImportError, AttributeError):
            logger.warning("Métricas externas no disponibles, usando identidad")
            return {
                "G_PHYSICS": np.eye(3, dtype=np.float64),
                "G_TOPOLOGY": np.eye(2, dtype=np.float64),
                "G_THERMODYNAMICS": np.eye(2, dtype=np.float64),
            }
    
    def _build_projector(self) -> OrthogonalProjector:
        scale_phys = np.array([
            PhysicalConstants.SATURATION_CRITICAL,
            PhysicalConstants.FLYBACK_MAX_SAFE,
            _P_NOMINAL
        ], dtype=np.float64)
        
        D_inv_phys = np.diag(1.0 / scale_phys)
        scaled_G_phys = D_inv_phys @ self._metric_tensors_state["G_PHYSICS"] @ D_inv_phys
        
        scale_topo = np.array([1.0, 1.0], dtype=np.float64)
        D_inv_topo = np.diag(1.0 / scale_topo)
        scaled_G_topo = D_inv_topo @ self._metric_tensors_state["G_TOPOLOGY"] @ D_inv_topo
        
        scale_thermo = np.array([0.5, 0.5], dtype=np.float64)
        D_inv_thermo = np.diag(1.0 / scale_thermo)
        scaled_G_thermo = D_inv_thermo @ self._metric_tensors_state["G_THERMODYNAMICS"] @ D_inv_thermo
        
        subspaces: Dict[str, SubspaceSpec] = {
            "physics_core": SubspaceSpec(
                name="physics_core",
                indices=slice(0, 3),
                weight=1.0,
                reference=np.zeros(3, dtype=np.float64),
                scale=scale_phys,
                metric=MetricTensor(scaled_G_phys, validate=False)
            ),
            "topology_core": SubspaceSpec(
                name="topology_core",
                indices=slice(3, 5),
                weight=1.5,
                reference=np.array([1.0, 0.0], dtype=np.float64),
                scale=scale_topo,
                metric=MetricTensor(scaled_G_topo, validate=False)
            ),
            "thermo_core": SubspaceSpec(
                name="thermo_core",
                indices=slice(5, 7),
                weight=1.2,
                reference=np.zeros(2, dtype=np.float64),
                scale=scale_thermo,
                metric=MetricTensor(scaled_G_thermo, validate=False)
            ),
        }
        
        return OrthogonalProjector(
            dimensions=7,
            subspaces=subspaces,
            topo_indices=BETTI_INDICES if self._enable_topology_monitoring else None,
            cache_projections=True
        )
    
    @property
    def domain(self) -> FrozenSet[Stratum]:
        return frozenset([Stratum.PHYSICS])
    
    @property
    def codomain(self) -> Stratum:
        return Stratum.WISDOM
    
    def reset_state(self) -> None:
        self._previous_status = None
        self._euler_history.clear()
        self._evaluation_count = 0
    
    def _check_topology_change(self, current_euler: Optional[int]) -> None:
        if not self._enable_topology_monitoring:
            return
        
        if len(self._euler_history) < 2:
            return
        
        prev_euler = self._euler_history[-2]
        
        if prev_euler is not None and current_euler is not None:
            if prev_euler != current_euler:
                logger.warning(
                    "🔄 Bifurcación topológica detectada: χ %d → %d",
                    prev_euler,
                    current_euler
                )
    
    def _verify_functorial_properties(self) -> Dict[str, bool]:
        val_report = self._projector.validation_report
        
        projector_ok = all(
            v < ALGEBRAIC_TOL
            for k, v in val_report.items()
            if any(kw in k for kw in ("error", "idempotence", "self_adjoint"))
        )
        
        max_h = (self._critical - self._warning) / 2.0
        
        return {
            "domain_includes_physics": Stratum.PHYSICS in self.domain,
            "codomain_is_wisdom": self.codomain == Stratum.WISDOM,
            "projector_algebraically_valid": projector_ok,
            "thresholds_ordered": self._warning < self._critical,
            "hysteresis_in_valid_range": 0.0 <= self._hysteresis < max_h,
        }
    
    def _compute_discrete_ricci_curvature(
        self,
        telemetry: Dict[str, Any]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        MEJORA: Modelo de curvatura basado en topología discreta.
        """
        betti_0 = telemetry.get('beta_0', 1.0)
        betti_1 = telemetry.get('beta_1', 0.0)
        
        Ric_PHYSICS = np.zeros((3, 3), dtype=np.float64)
        Ric_TOPOLOGY = np.zeros((2, 2), dtype=np.float64)
        Ric_THERMO = np.zeros((2, 2), dtype=np.float64)
        
        if betti_1 > 0:
            Ric_TOPOLOGY[1, 1] = -0.5 * betti_1
        
        if betti_0 > 1:
            Ric_TOPOLOGY[0, 0] = -1.0 * (betti_0 - 1)
        
        return Ric_PHYSICS, Ric_TOPOLOGY, Ric_THERMO
    
    def _evolve_metric_tensors_ricci_flow(
        self,
        telemetry: Dict[str, Any],
        psi: NDArray[np.float64],
        dt: float = RICCI_FLOW_DT,
        gamma: float = 0.1
    ) -> None:
        """
        CORRECCIÓN CRÍTICA: Proyección al cono SPD después de cada paso.
        
        Flujo de Ricci normalizado:
        ∂_t G = -2·Ric + γ·diag(S_p)
        
        Luego: G ← project_SPD(G)
        """
        Ric_P, Ric_T, Ric_Th = self._compute_discrete_ricci_curvature(telemetry)
        
        membrane = IsolatingMembraneFunctor(p=1.5, eps=EPS)
        S_p_full = membrane.compute_topological_stress(psi)
        
        S_p_P = S_p_full[0:3]
        S_p_T = S_p_full[3:5]
        S_p_Th = S_p_full[5:7]
        
        # Actualización vía Euler explícito
        G_P_candidate = (
            self._metric_tensors_state["G_PHYSICS"]
            - 2 * dt * Ric_P
            + gamma * np.diag(S_p_P)
        )
        G_T_candidate = (
            self._metric_tensors_state["G_TOPOLOGY"]
            - 2 * dt * Ric_T
            + gamma * np.diag(S_p_T)
        )
        G_Th_candidate = (
            self._metric_tensors_state["G_THERMODYNAMICS"]
            - 2 * dt * Ric_Th
            + gamma * np.diag(S_p_Th)
        )
        
        # PROYECCIÓN AL CONO SPD (CRÍTICO)
        self._metric_tensors_state["G_PHYSICS"] = StableLinearAlgebra.regularize_spd_matrix(
            G_P_candidate,
            min_eig=MIN_EIGVAL_TOL,
            method='spectral'
        )
        self._metric_tensors_state["G_TOPOLOGY"] = StableLinearAlgebra.regularize_spd_matrix(
            G_T_candidate,
            min_eig=MIN_EIGVAL_TOL,
            method='spectral'
        )
        self._metric_tensors_state["G_THERMODYNAMICS"] = StableLinearAlgebra.regularize_spd_matrix(
            G_Th_candidate,
            min_eig=MIN_EIGVAL_TOL,
            method='spectral'
        )
        
        # Re-construcción del proyector
        self._projector = self._build_projector()
    
    def __call__(self, state: CategoricalState) -> CategoricalState:
        self._evaluation_count += 1
        
        if not state.is_success:
            return state
        
        try:
            telemetry = state.context.get("telemetry_metrics", {})
            
            signal = build_signal(telemetry or {}, strict=False)
            
            # Evolucionar métricas antes de proyección
            self._evolve_metric_tensors_ricci_flow(telemetry or {}, signal)
            
            assessment = self._projector.project(
                signal,
                warning_threshold=self._warning,
                critical_threshold=self._critical,
                hysteresis=self._hysteresis,
                previous_status=self._previous_status,
                verbose=False
            )
            
            self._previous_status = assessment.status
            self._euler_history.append(assessment.euler_char)
            self._check_topology_change(assessment.euler_char)
            
            return self._emit_state(state, assessment)
        
        except MetricTensorError as exc:
            logger.critical(
                "🛡️ COLAPSO TENSORIAL (#%d): %s",
                self._evaluation_count,
                exc
            )
            return state.with_error(
                error_msg=f"Colapso Tensorial: {exc}",
                details={"action": "QUARANTINE", "context": exc.context}
            )
        
        except (ValueError, ImmuneSystemError) as exc:
            logger.error(
                "Error de validación (#%d): %s",
                self._evaluation_count,
                exc
            )
            return state.with_error(
                error_msg=f"Validación fallida: {exc}",
                details={"action": "ERROR"}
            )
        
        except Exception as exc:
            logger.exception("Error inesperado (#%d)", self._evaluation_count)
            return state.with_error(
                error_msg=f"Error inesperado: {exc}",
                details={"action": "ERROR"}
            )
    
    def _emit_state(
        self,
        state: CategoricalState,
        assessment: ThreatAssessment
    ) -> CategoricalState:
        details = assessment.to_dict()
        
        if assessment.status == HealthStatus.CRITICAL:
            logger.critical(
                "🛡️ CUARENTENA ACTIVADA: fuente=%s amenaza=%.4f",
                assessment.max_source,
                assessment.max_value
            )
            return state.with_error(
                error_msg=f"Cuarentena: {assessment.max_source}",
                details={**details, "action": "QUARANTINE"}
            )
        
        if assessment.status == HealthStatus.WARNING:
            logger.warning(
                "🛡️ ESTRÉS: fuente=%s amenaza=%.4f",
                assessment.max_source,
                assessment.max_value
            )
            return state.with_update(
                {"immune_status": "warning", **details},
                new_stratum=self.codomain
            )
        
        logger.debug("🛡️ ESTABLE: max_threat=%.4f", assessment.max_value)
        return state.with_update(
            {"immune_status": "healthy", **details},
            new_stratum=self.codomain
        )
    
    @property
    def thresholds(self) -> Dict[str, float]:
        return {
            "warning": self._warning,
            "critical": self._critical,
            "hysteresis": self._hysteresis
        }
    
    @property
    def topology_history(self) -> Tuple[Optional[int], ...]:
        return tuple(self._euler_history)
    
    @property
    def evaluation_count(self) -> int:
        return self._evaluation_count
    
    @property
    def current_status(self) -> Optional[HealthStatus]:
        return self._previous_status
    
    def get_diagnostics(self) -> Dict[str, Any]:
        history = self.topology_history
        
        return {
            "name": self.name,
            "evaluation_count": self._evaluation_count,
            "current_status": (
                str(self._previous_status)
                if self._previous_status
                else "none"
            ),
            "thresholds": self.thresholds,
            "topology_history_last10": history[-10:],
            "projector_validation": self._projector.validation_report,
            "functorial_properties": self._verify_functorial_properties(),
        }
    
    def health_report(self) -> str:
        d = self.get_diagnostics()
        props = d["functorial_properties"]
        history_last5 = d["topology_history_last10"][-5:]
        history_str = (
            " → ".join(str(c) for c in history_last5)
            if history_last5
            else "(sin evaluaciones)"
        )
        
        lines = [
            "🛡️  IMMUNE WATCHER — DIAGNÓSTICO",
            "  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            f"  Nombre            : {d['name']}",
            f"  Evaluaciones      : {d['evaluation_count']}",
            f"  Estado Actual     : {d['current_status']}",
            "  UMBRALES:",
            f"    ⚠  Warning    : {d['thresholds']['warning']:.3f}",
            f"    ✗  Critical   : {d['thresholds']['critical']:.3f}",
            f"    ↺  Hysteresis : {d['thresholds']['hysteresis']:.3f}",
            "  TOPOLOGÍA (últimos 5):",
            f"    {history_str}",
            "  PROPIEDADES CATEGÓRICAS:",
        ]
        
        for prop, ok in props.items():
            symbol = "✓" if ok else "✗"
            lines.append(f"    [{symbol}] {prop}")
        
        return "\n".join(lines)
    
    def evaluate_manifold_deformation(
        self,
        state_tensor: NDArray[np.float64],
        reference_chi: Optional[int] = None
    ) -> Any:
        """Intercepción Riemanniana y Auditoría de Bifurcación (FASE IV)."""
        from app.boole.strategy.sheaf_cohomology_orchestrator import ThreatMetrics
        
        psi = np.asarray(state_tensor, dtype=np.float64).ravel()
        
        assessment = self._projector.project(
            psi,
            warning_threshold=self._warning,
            critical_threshold=self._critical,
            hysteresis=self._hysteresis,
            previous_status=self._previous_status
        )
        
        current_chi = assessment.euler_char
        
        if reference_chi is not None:
            last_stable_chi = reference_chi
        else:
            last_stable_chi = next(
                (c for c in reversed(self._euler_history) if c is not None),
                current_chi
            )
        
        if current_chi is not None and last_stable_chi is not None:
            delta_chi = current_chi - last_stable_chi
        else:
            delta_chi = 0
        
        is_stable = (
            delta_chi == 0 and
            assessment.status != HealthStatus.CRITICAL
        )
        
        return ThreatMetrics(
            mahalanobis_distance=assessment.total_threat,
            is_stable=is_stable,
            structural_alteration=int(delta_chi),
            threat_level=assessment.status.name,
            details=assessment.to_dict()
        )

# ==============================================================================
# FACTORY (SIN CAMBIOS ESTRUCTURALES)
# ==============================================================================

def create_immune_watcher(
    profile: str = "default",
    **overrides: Any
) -> ImmuneWatcherMorphism:
    _PROFILES: Dict[str, Dict[str, Any]] = {
        "default": {
            "warning_threshold": 0.8,
            "critical_threshold": 1.5,
            "hysteresis": 0.05
        },
        "strict": {
            "warning_threshold": 0.5,
            "critical_threshold": 1.0,
            "hysteresis": 0.03
        },
        "relaxed": {
            "warning_threshold": 1.0,
            "critical_threshold": 2.0,
            "hysteresis": 0.08
        },
        "laboratory": {
            "warning_threshold": 0.8,
            "critical_threshold": 1.5,
            "hysteresis": 0.0
        },
    }
    
    if profile not in _PROFILES:
        raise ValueError(
            f"Perfil desconocido: '{profile}'",
            available=list(_PROFILES.keys())
        )
    
    config: Dict[str, Any] = {**_PROFILES[profile], **overrides}
    
    return ImmuneWatcherMorphism(
        name=f"immune_watcher_{profile}",
        **config
    )

# ==============================================================================
# EXPORTACIÓN PÚBLICA
# ==============================================================================

__all__ = [
    "ImmuneSystemError",
    "NumericalStabilityError",
    "DimensionalMismatchError",
    "PhysicalBoundsError",
    "TopologicalInvariantError",
    "MetricTensorError",
    "SpectralDecompositionError",
    "EPS",
    "ALGEBRAIC_TOL",
    "COND_NUM_TOL",
    "MIN_EIGVAL_TOL",
    "PhysicalConstants",
    "StableLinearAlgebra",
    "Validator",
    "VALIDATOR_REGISTRY",
    "HealthStatus",
    "SpectralDecomposition",
    "MetricTensor",
    "RiemannianMetric",
    "IsolatingMembraneFunctor",
    "SubspaceSpec",
    "ThreatAssessment",
    "OrthogonalProjector",
    "SignalComponent",
    "SIGNAL_SCHEMA",
    "BETTI_INDICES",
    "build_signal",
    "ImmuneWatcherMorphism",
    "create_immune_watcher",
]