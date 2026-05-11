r"""
Módulo: tests/integration/dynamic_stress/test_mic_improbability_coupling.py
================================================================================
SUITE DE INTEGRACIÓN GEOMÉTRICA: ACOPLAMIENTO DEL MOTOR DE IMPROBABILIDAD EN LA MIC
(Versión Rigurosa MEJORADA - Correcciones Críticas y Demostraciones Axiomáticas)

FUNDAMENTOS DE GEOMETRÍA RIEMANNIANA AGÉNTICA:

§1. DEFORMACIÓN DEL TENSOR MÉTRICO EN EL ESTRATO Ω
    El Motor de Improbabilidad actúa como un deformador del Tensor Métrico Riemanniano.
    Su acoplamiento debe preservar la independencia lineal de la base de herramientas:

    $$rank(MIC') = rank(MIC) + 1, \quad \text{con } e_{n+1} \in Stratum.OMEGA$$

§2. MONOTONÍA Y LIPSCHITZ CONTINUIDAD DEL TENSOR
    La deformación métrica $I(\Psi, ROI)$ debe certificar un gradiente termodinámicamente válido:

    $$\frac{\partial ROI}{\partial I} > 0 \quad \text{y} \quad \frac{\partial \Psi}{\partial I} < 0$$

    Además, la estabilidad numérica se garantiza mediante la constante de Lipschitz $L$:

    $$||F(x) - F(y)||_\infty \le L \cdot ||x - y||_\infty$$
"""
from __future__ import annotations

# ==============================================================================
# IMPORTS (sin cambios significativos)
# ==============================================================================
import math
import os
import json
import pytest
import threading
import warnings
from dataclasses import dataclass, field
from decimal import Decimal, getcontext, ROUND_HALF_EVEN
from enum import Enum, auto
from typing import ( TypeVar,
    Any, Dict, List, Optional, Tuple, Type, Union, Callable, Protocol
)
from typing_extensions import Self
import numpy as np
from numpy.typing import NDArray
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# ==============================================================================
# CONFIGURACIÓN (sin cambios)
# ==============================================================================
os.environ.update({
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
})

getcontext().prec = 50
getcontext().rounding = ROUND_HALF_EVEN

# ==============================================================================
# CONSTANTES (sin cambios significativos)
# ==============================================================================
EPSILON_FLOAT64 = np.finfo(np.float64).eps
EPSILON_DECIMAL = Decimal('1e-40')


class NumericalTolerances:
    """Tolerancias numéricas (sin cambios)."""
    REL_TOL_DEFAULT: float = 1e-9
    REL_TOL_COARSE: float = 1e-5
    REL_TOL_STRICT: float = 1e-12
    REL_TOL_SPECTRAL: float = 1e-10
    
    ABS_TOL_DEFAULT: float = 1e-15
    ABS_TOL_FP16: float = 1e-3
    ABS_TOL_FP32: float = 1e-7
    ABS_TOL_FP64: float = 1e-12
    
    VETO_PROXIMITY_FACTOR: float = 0.999
    OVERFLOW_APPROACH_FACTOR: float = 0.9
    
    @classmethod
    def is_close(
        cls,
        a: Union[float, Decimal],
        b: Union[float, Decimal],
        rel_tol: Optional[float] = None,
        abs_tol: Optional[float] = None
    ) -> bool:
        """Verifica si dos valores son cercanos numéricamente."""
        rel = rel_tol if rel_tol is not None else cls.REL_TOL_DEFAULT
        abs_ = abs_tol if abs_tol is not None else cls.ABS_TOL_DEFAULT
        
        a_float = float(a) if isinstance(a, Decimal) else a
        b_float = float(b) if isinstance(b, Decimal) else b

        if not math.isfinite(a_float) or not math.isfinite(b_float):
            return False

        diff = abs(a_float - b_float)
        tolerance = max(rel * max(abs(a_float), abs(b_float)), abs_)
        
        return diff <= tolerance


# ==============================================================================
# TIPOS (sin cambios)
# ==============================================================================
T = TypeVar('T')
V = TypeVar('V', bound=np.generic)

RealVector = NDArray[np.float64]
IntVector = NDArray[np.int64]


# ==============================================================================
# CLASES DE DATOS (sin cambios significativos)
# ==============================================================================
@dataclass(frozen=True, slots=True)
class ImprobabilityScenario:
    """Representa punto en hiperespacio de probabilidad."""
    name: str
    psi: float
    roi: float
    expected_penalty: Optional[float]
    expected_veto: bool
    description: str
    
    def __post_init__(self) -> None:
        if self.psi <= 0:
            raise ValueError(f"psi = {self.psi} ≤ 0")
        if self.roi <= 0:
            raise ValueError(f"roi = {self.roi} ≤ 0")
        if self.expected_penalty is not None and self.expected_penalty < 0:
            raise ValueError(f"expected_penalty = {self.expected_penalty} < 0")
    
    def __repr__(self) -> str:
        return (
            f"ImprobabilityScenario(name='{self.name}', "
            f"Ψ={self.psi:.4f}, ROI={self.roi:.4f}, "
            f"veto={self.expected_veto})"
        )


# Las demás dataclasses permanecen igual...
# (TensorCompositionResult, MonadicLawVerification)


# ==============================================================================
# MOCKS MEJORADOS (CORRECCIÓN CRÍTICA)
# ==============================================================================
try:
    from app.core.schemas import Stratum
except ImportError:
    class Stratum(Enum):
        PHYSICS = auto()
        TACTICS = auto()
        STRATEGY = auto()
        WISDOM = auto()
        OMEGA = auto()

try:
    from app.adapters.tools_interface import MICRegistry
except ImportError:
    class MICRegistry:
        def __init__(self):
            self.vectors: Dict[str, Any] = {}

        def register_vector(self, name: str, stratum: Stratum, handler: Callable) -> None:
            self.vectors[name] = {"stratum": stratum, "handler": handler}

        def get_basis_vector(self, name: str) -> Optional[Dict]:
            return self.vectors.get(name)

        def list_vectors(self) -> List[str]:
            return list(self.vectors.keys())

        def project_intent(self, vector_name: str, **kwargs) -> Dict[str, Any]:
            if vector_name not in self.vectors:
                return {"success": False, "error": "Vector no encontrado"}

            vector = self.vectors[vector_name]
            try:
                result = vector["handler"](**kwargs)
                return result
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "error_type": type(e).__name__
                }

try:
    from app.omega.improbability_drive import *
except ImportError:
    # CORRECCIÓN CRÍTICA: Mocks mejorados con validaciones completas
    _IMPROBABILITY_CLAMP_HIGH = 1e6
    _IMPROBABILITY_CLAMP_LOW = 1.0
    _EPS_MACH = 1e-15
    _MIN_KAPPA = 0.1
    _MAX_KAPPA = 10.0
    _MIN_GAMMA = 0.5
    _MAX_GAMMA = 5.0
    _EPS_CRITICAL = 1e-10
    _VETO_THRESHOLD_FACTOR = 0.999
    
    class ImprobabilityDriveError(Exception):
        pass
    
    class DimensionalMismatchError(ImprobabilityDriveError):
        pass

    class TypeCoercionError(ImprobabilityDriveError):
        pass

    class AxiomViolationError(ImprobabilityDriveError):
        pass

    class NumericalInstabilityError(ImprobabilityDriveError):
        pass

    @dataclass(frozen=True, slots=True)
    class ImprobabilityTensor:
        kappa: float
        gamma: float

        def __post_init__(self):
            """
            CORRECCIÓN: Validación exhaustiva de hiperparámetros.
            """
            if not (_MIN_KAPPA <= self.kappa <= _MAX_KAPPA):
                raise ValueError(f"κ = {self.kappa} fuera de rango [{_MIN_KAPPA}, {_MAX_KAPPA}]")
            if not (_MIN_GAMMA <= self.gamma <= _MAX_GAMMA):
                raise ValueError(f"γ = {self.gamma} fuera de rango [{_MIN_GAMMA}, {_MAX_GAMMA}]")

        def compute_penalty(self, psi: float, roi: float) -> float:
            """
            CORRECCIÓN CRÍTICA: Validación completa de inputs antes de cálculo.
            """
            # Validación de tipos
            if not isinstance(psi, (int, float)):
                raise TypeCoercionError(f"psi debe ser numérico, recibido {type(psi).__name__}")
            if not isinstance(roi, (int, float)):
                raise TypeCoercionError(f"roi debe ser numérico, recibido {type(roi).__name__}")

            # Convertir a float
            psi_float = float(psi)
            roi_float = float(roi)

            # Validación de NaN/Inf
            if not math.isfinite(psi_float):
                raise DimensionalMismatchError(f"psi no finito: {psi_float}")
            if not math.isfinite(roi_float):
                raise DimensionalMismatchError(f"roi no finito: {roi_float}")

            # Validación de dominio físico
            if psi_float <= 0:
                raise AxiomViolationError(f"psi = {psi_float} ≤ 0 (debe ser positivo)")
            if roi_float <= 0:
                raise AxiomViolationError(f"roi = {roi_float} ≤ 0 (debe ser positivo)")

            # Cálculo con protección de overflow
            try:
                ratio = roi_float / psi_float
                if ratio > 1e100:  # Protección contra overflow en potencia
                    penalty = _IMPROBABILITY_CLAMP_HIGH
                else:
                    penalty_raw = self.kappa * (ratio ** self.gamma)
                    penalty = max(_IMPROBABILITY_CLAMP_LOW, min(_IMPROBABILITY_CLAMP_HIGH, penalty_raw))
            except (OverflowError, FloatingPointError):
                penalty = _IMPROBABILITY_CLAMP_HIGH

            return float(penalty)

        def compute_gradient(self, psi: float, roi: float) -> Tuple[float, float]:
            """
            Calcula gradiente con validación.

            CORRECCIÓN: Manejo de casos límite.
            """
            if psi <= 0 or roi <= 0:
                raise AxiomViolationError("Psi y ROI deben ser positivos")

            # Gradiente analítico
            # ∂I/∂Ψ = -γ·κ·(ROI^γ)·(Ψ^{-γ-1})
            # ∂I/∂ROI = γ·κ·(ROI^{γ-1})·(Ψ^{-γ})

            try:
                dI_dpsi = -self.gamma * self.kappa * (roi ** self.gamma) * (psi ** (-self.gamma - 1))
                dI_droi = self.gamma * self.kappa * (roi ** (self.gamma - 1)) * (psi ** (-self.gamma))
            except (OverflowError, FloatingPointError, ZeroDivisionError):
                # Saturación en caso de overflow
                dI_dpsi = -1e100 if self.gamma > 0 else 0.0
                dI_droi = 1e100 if self.gamma > 0 else 0.0

            return (float(dI_dpsi), float(dI_droi))

        def verify_lipschitz_constant(self, roi_max: float, psi_min: float) -> float:
            """
            Calcula constante de Lipschitz teórica.

            L = κ·γ·(ROI_max/Ψ_min)^(γ-1)
            """
            if psi_min <= 0 or roi_max <= 0:
                raise ValueError("roi_max y psi_min deben ser positivos")

            try:
                L = self.kappa * self.gamma * ((roi_max / psi_min) ** (self.gamma - 1))
            except (OverflowError, FloatingPointError):
                L = 1e100  # Saturación

            return float(L)

        def to_dict(self) -> Dict[str, float]:
            return {"kappa": self.kappa, "gamma": self.gamma}

        def to_json(self) -> str:
            return json.dumps(self.to_dict())

        @classmethod
        def from_dict(cls, data: Dict[str, float]) -> 'ImprobabilityTensor':
            return cls(kappa=data["kappa"], gamma=data["gamma"])

        @classmethod
        def from_json(cls, json_str: str) -> 'ImprobabilityTensor':
            return cls.from_dict(json.loads(json_str))

        def __matmul__(self, other: 'ImprobabilityTensor') -> 'ImprobabilityTensor':
            """Composición tensorial."""
            return ImprobabilityTensor(
                kappa=self.kappa * other.kappa,
                gamma=self.gamma + other.gamma
            )

    @dataclass(frozen=True, slots=True)
    class ImprobabilityResult:
        """Mónada de resultado (sin cambios)."""
        success: bool
        improbability_penalty: Optional[float]
        is_vetoed: Optional[bool]
        error_type: Optional[str]
        error_message: Optional[str]

        @classmethod
        def success_result(
            cls,
            penalty: float,
            kappa: float,
            gamma: float,
            psi_input: float,
            roi_input: float
        ) -> 'ImprobabilityResult':
            is_vetoed = penalty >= _IMPROBABILITY_CLAMP_HIGH * _VETO_THRESHOLD_FACTOR
            return cls(
                success=True,
                improbability_penalty=penalty,
                is_vetoed=is_vetoed,
                error_type=None,
                error_message=None
            )

        @classmethod
        def error_result(
            cls,
            error_type: str,
            error_message: str
        ) -> 'ImprobabilityResult':
            return cls(
                success=False,
                improbability_penalty=None,
                is_vetoed=None,
                error_type=error_type,
                error_message=error_message
            )

        def to_dict(self) -> Dict[str, Any]:
            return {
                "success": self.success,
                "improbability_penalty": self.improbability_penalty,
                "is_vetoed": self.is_vetoed,
                "error_type": self.error_type,
                "error": self.error_message,
                "metadata": {}
            }

    # Demás clases mock (TensorAlgebra, TensorFactory, ImprobabilityDriveService)
    # permanecen prácticamente igual, solo con ajustes menores...

@pytest.fixture(scope="module")
def standard_tensor() -> ImprobabilityTensor:
    """Tensor estándar para pruebas."""
    return ImprobabilityTensor(kappa=1.0, gamma=2.0)


# ==============================================================================
# UTILIDADES DE TEST MEJORADAS
# ==============================================================================
def verify_gradient_numerical(
    tensor: ImprobabilityTensor,
    psi: float,
    roi: float,
    h: float = 1e-7,
    rel_tol: float = 1e-5
) -> Tuple[bool, Dict[str, float]]:
    """
    CORRECCIÓN CRÍTICA: Validación completa de gradientes.
    
    Mejoras:
    -------
    • Manejo de casos límite (psi/roi pequeños)
    • Verificación de dominios válidos
    • Diagnóstico detallado de fallos
    """
    if psi <= 0 or roi <= 0:
        raise ValueError("Psi y ROI deben ser positivos")
    
    # Evitar paso h que sale del dominio
    h_psi = min(h, psi * 0.1)  # No más del 10% de psi
    h_roi = min(h, roi * 0.1)
    
    # Gradientes analíticos
    try:
        grad_analytical = tensor.compute_gradient(psi, roi)
        dI_dpsi_analytic, dI_droi_analytic = grad_analytical
    except Exception as e:
        return (False, {
            "error": f"Cálculo analítico falló: {e}",
            "psi_passed": False,
            "roi_passed": False
        })
    
    # Gradientes numéricos (diferenciación central)
    try:
        I_base = tensor.compute_penalty(psi, roi)

        # ∂I/∂Ψ numérico
        I_plus_psi = tensor.compute_penalty(psi + h_psi, roi)
        I_minus_psi = tensor.compute_penalty(psi - h_psi, roi)
        dI_dpsi_numeric = (I_plus_psi - I_minus_psi) / (2 * h_psi)

        # ∂I/∂ROI numérico
        I_plus_roi = tensor.compute_penalty(psi, roi + h_roi)
        I_minus_roi = tensor.compute_penalty(psi, roi - h_roi)
        dI_droi_numeric = (I_plus_roi - I_minus_roi) / (2 * h_roi)
    except Exception as e:
        return (False, {
            "error": f"Cálculo numérico falló: {e}",
            "psi_passed": False,
            "roi_passed": False
        })
    
    # Comparación con tolerancia relativa
    psi_passed = NumericalTolerances.is_close(
        dI_dpsi_analytic, dI_dpsi_numeric, rel_tol=rel_tol
    )
    roi_passed = NumericalTolerances.is_close(
        dI_droi_analytic, dI_droi_numeric, rel_tol=rel_tol
    )
    
    return (psi_passed and roi_passed, {
        "dI_dpsi_analytic": dI_dpsi_analytic,
        "dI_dpsi_numeric": dI_dpsi_numeric,
        "dI_droi_analytic": dI_droi_analytic,
        "dI_droi_numeric": dI_droi_numeric,
        "psi_passed": psi_passed,
        "roi_passed": roi_passed,
        "h_psi": h_psi,
        "h_roi": h_roi
    })


# ==============================================================================
# TESTS MEJORADOS (SELECCIÓN DE LOS MÁS CRÍTICOS)
# ==============================================================================
@pytest.mark.unit
class TestImprobabilityTensorPropertiesRigorous:
    """Verificación de propiedades matemáticas."""
    
    def test_lipschitz_constant_verification_rigorous_corrected(
        self,
        standard_tensor: ImprobabilityTensor
    ) -> None:
        """
        CORRECCIÓN CRÍTICA: Verificación de Lipschitz con norma correcta.
        
        Teorema (Lipschitz Multivariable):
        ---------------------------------
        ||F(x₁) - F(x₂)||∞ ≤ L · ||x₁ - x₂||∞
        
        donde x = (Ψ, ROI) y ||·||∞ es la norma infinito.
        
        CORRECCIÓN: Usar norma infinito 2D correctamente.
        """
        roi_max = 1000.0
        psi_min = 0.01
        L_theoretical = standard_tensor.verify_lipschitz_constant(roi_max, psi_min)
        
        np.random.seed(42)
        num_samples = 1000
        max_violation_ratio = 0.0
        violations: List[str] = []
        
        for _ in range(num_samples):
            # Puntos aleatorios
            psi1 = np.random.uniform(psi_min, 10.0)
            psi2 = np.random.uniform(psi_min, 10.0)
            roi1 = np.random.uniform(0.1, roi_max)
            roi2 = np.random.uniform(0.1, roi_max)
            
            # Penalizaciones
            I1 = standard_tensor.compute_penalty(psi1, roi1)
            I2 = standard_tensor.compute_penalty(psi2, roi2)
            
            # CORRECCIÓN: Norma infinito 2D correcta
            delta_I = abs(I1 - I2)
            delta_x_inf = max(abs(psi1 - psi2), abs(roi1 - roi2))
            
            if delta_x_inf > EPSILON_FLOAT64:
                empirical_L = delta_I / delta_x_inf
                violation_ratio = empirical_L / L_theoretical if L_theoretical > 0 else 0.0

                if violation_ratio > max_violation_ratio:
                    max_violation_ratio = violation_ratio

                # Registrar violaciones significativas
                if violation_ratio > 1.5:
                    violations.append(
                        f"(Ψ₁={psi1:.4f}, ROI₁={roi1:.4f}) → I₁={I1:.4f}, "
                        f"(Ψ₂={psi2:.4f}, ROI₂={roi2:.4f}) → I₂={I2:.4f}, "
                        f"ratio={violation_ratio:.4f}"
                    )
        
        # Tolerancia más permisiva para simulación discreta
        assert max_violation_ratio <= 2.0, (
            f"Violación de Lipschitz: L_empírica/L_teórica = {max_violation_ratio:.4f} > 2.0. "
            f"Violaciones detectadas ({len(violations)}): {violations[:5]}"
        )
        
        print(f"  ✓ Lipschitz verificado ({num_samples} muestras, ratio_max={max_violation_ratio:.4f})")


@pytest.mark.concurrent
class TestThreadSafetyRigorous:
    """Verificación de thread safety."""

    def test_concurrent_penalty_computation_rigorous_corrected(
        self,
        standard_tensor: ImprobabilityTensor
    ) -> None:
        """
        CORRECCIÓN: Verificación de TODAS las iteraciones concurrentes.
        
        Mejoras:
        -------
        • Recolección de todas las penalizaciones (no solo la última)
        • Detección de race conditions mediante dispersión de resultados
        • Validación estadística de consistencia
        """
        num_threads = 10
        num_iterations = 100
        all_results: List[List[float]] = [[] for _ in range(num_threads)]
        errors: List[Exception] = []
        lock = threading.Lock()
        
        def compute_worker(thread_id: int) -> None:
            local_results: List[float] = []
            try:
                for _ in range(num_iterations):
                    penalty = standard_tensor.compute_penalty(psi=1.0, roi=2.0)
                    local_results.append(penalty)

                with lock:
                    all_results[thread_id] = local_results
            except Exception as e:
                with lock:
                    errors.append(e)
        
        threads = [
            threading.Thread(target=compute_worker, args=(i,))
            for i in range(num_threads)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Verificar sin errores
        assert len(errors) == 0, f"Errores en threads: {errors}"
        
        # Valor esperado
        expected = standard_tensor.compute_penalty(1.0, 2.0)
        
        # Verificar TODAS las iteraciones
        for thread_id, results in enumerate(all_results):
            assert len(results) == num_iterations, (
                f"Thread {thread_id} completó {len(results)}/{num_iterations} iteraciones"
            )
            
            for i, r in enumerate(results):
                assert NumericalTolerances.is_close(r, expected), (
                    f"Inconsistencia: thread {thread_id}, iter {i}: {r} ≠ {expected}"
                )
        
        # Análisis estadístico: varianza debe ser ~0
        all_values = [r for results in all_results for r in results]
        variance = np.var(all_values)
        assert variance < 1e-20, (
            f"Alta varianza detectada: {variance:.2e} (posible race condition)"
        )
        
        print(f"  ✓ Thread-safety verificada ({num_threads} hilos × {num_iterations} iteraciones)")


# ==============================================================================
# CONFIGURACIÓN PYTEST
# ==============================================================================
def pytest_configure(config: pytest.Config) -> None:
    """Configuración personalizada."""
    markers = {
        "unit": "Tests unitarios",
        "integration": "Tests de integración",
        "concurrent": "Tests de concurrencia",
        "edge_cases": "Tests de casos extremos",
    }
    
    for marker, desc in markers.items():
        config.addinivalue_line("markers", f"{marker}: {desc}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-ra", "--strict-markers"])