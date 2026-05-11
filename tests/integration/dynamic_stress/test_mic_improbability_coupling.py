r"""
Módulo: tests/integration/dynamic_stress/test_mic_improbability_coupling.py

═══════════════════════════════════════════════════════════════════════════════
SUITE DE INTEGRACIÓN GEOMÉTRICA: ACOPLAMIENTO DEL MOTOR DE IMPROBABILIDAD EN LA MIC
(Versión Rigurosa FINAL - Variedad Topológica Completa y Cerrada)
═══════════════════════════════════════════════════════════════════════════════

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

import math
import os
import json
import pytest
import threading
import warnings
from dataclasses import dataclass, field
from decimal import Decimal, getcontext, ROUND_HALF_EVEN
from enum import auto, Enum
from typing import (
    Any, Dict, List, Optional, Tuple, Type, Union, Callable, Protocol, TypeVar, Iterable
)
from typing_extensions import Self
import numpy as np
from numpy.typing import NDArray
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# ==============================================================================
# CONFIGURACIÓN DETERMINISTA (FASE 1)
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
# CONSTANTES Y TOLERANCIAS
# ==============================================================================
EPSILON_FLOAT64 = np.finfo(np.float64).eps
EPSILON_DECIMAL = Decimal('1e-40')

class NumericalTolerances:
    """Tolerancias numéricas para validación de la variedad."""
    REL_TOL_DEFAULT: float = 1e-9
    REL_TOL_COARSE: float = 1e-5
    REL_TOL_STRICT: float = 1e-12
    
    ABS_TOL_DEFAULT: float = 1e-15
    ABS_TOL_FP16: float = 1e-3
    
    VETO_PROXIMITY_FACTOR: float = 0.999
    
    @classmethod
    def is_close(
        cls,
        a: Union[float, Decimal],
        b: Union[float, Decimal],
        rel_tol: Optional[float] = None,
        abs_tol: Optional[float] = None
    ) -> bool:
        rel = rel_tol if rel_tol is not None else cls.REL_TOL_DEFAULT
        abs_ = abs_tol if abs_tol is not None else cls.ABS_TOL_DEFAULT
        
        a_f = float(a)
        b_f = float(b)
        
        if not math.isfinite(a_f) or not math.isfinite(b_f):
            return False

        return abs(a_f - b_f) <= max(rel * max(abs(a_f), abs(b_f)), abs_)

# ==============================================================================
# MOCKS Y ESTRUCTURA DE SOPORTE (FULL MESH)
# ==============================================================================
try:
    from app.core.schemas import Stratum
except ImportError:
    class Stratum(Enum):
        PHYSICS = 5
        TACTICS = 4
        STRATEGY = 3
        WISDOM = 0
        OMEGA = 6

try:
    from app.adapters.tools_interface import MICRegistry
except ImportError:
    class MICRegistry:
        def __init__(self):
            self.vectors = {}
        def register_vector(self, name, stratum, handler):
            self.vectors[name] = {"stratum": stratum, "handler": handler, "target_stratum": stratum}
        def get_basis_vector(self, name):
            if name not in self.vectors: return None
            data = self.vectors[name]
            # Mock de objeto vector
            return type('Vector', (), {"target_stratum": data["target_stratum"]})()
        def list_vectors(self):
            return list(self.vectors.keys())
        def project_intent(self, vector_name, **kwargs):
            if vector_name not in self.vectors: return {"success": False, "error": "Not Found"}

            # Validación de clausura transitiva (MOCK)
            context = kwargs.get('context', {})
            validated = context.get('validated_strata', frozenset())
            # Si es OMEGA, debe estar validado todo lo anterior
            if self.vectors[vector_name]["stratum"] == Stratum.OMEGA:
                required = {Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY}
                if not required.issubset(validated):
                    return {"success": False, "error": "Clausura Transitiva Violada", "error_type": "MICHierarchyViolationError"}

            try:
                res = self.vectors[vector_name]["handler"](**kwargs)
                return res
            except Exception as e:
                return {"success": False, "error": str(e), "error_type": type(e).__name__}

# Constantes del dominio Ω
_IMPROBABILITY_CLAMP_HIGH = 1e6
_IMPROBABILITY_CLAMP_LOW = 1.0
_MIN_KAPPA = 0.1
_MAX_KAPPA = 10.0
_MIN_GAMMA = 0.5
_MAX_GAMMA = 5.0
_VETO_THRESHOLD_FACTOR = 0.999
_EPS_MACH = 1e-15

class ImprobabilityDriveError(Exception): pass
class DimensionalMismatchError(ImprobabilityDriveError): pass
class TypeCoercionError(ImprobabilityDriveError): pass
class AxiomViolationError(ImprobabilityDriveError): pass

@dataclass(frozen=True, slots=True)
class ImprobabilityTensor:
    """Tensor de Improbabilidad: El Deformador Métrico de Ω."""
    kappa: float
    gamma: float

    def __post_init__(self):
        if not (_MIN_KAPPA <= self.kappa <= _MAX_KAPPA):
            raise ValueError(f"κ={self.kappa} out of range")
        if not (_MIN_GAMMA <= self.gamma <= _MAX_GAMMA):
            raise ValueError(f"γ={self.gamma} out of range")

    def compute_penalty(self, psi: Any, roi: Any) -> float:
        if not isinstance(psi, (int, float)): raise TypeCoercionError(f"Invalid psi type: {type(psi)}")
        if not isinstance(roi, (int, float)): raise TypeCoercionError(f"Invalid roi type: {type(roi)}")
        
        p, r = float(psi), float(roi)
        if not math.isfinite(p) or not math.isfinite(r): raise DimensionalMismatchError("NaN/Inf detected")
        if p < 0 or r <= 0: raise AxiomViolationError("Values must be >= 0 for psi and > 0 for roi")
        
        # Regularización en frontera psi=0
        p_eff = max(p, _EPS_MACH)
        
        try:
            ratio = r / p_eff
            val = self.kappa * (ratio ** self.gamma)
            return float(max(_IMPROBABILITY_CLAMP_LOW, min(_IMPROBABILITY_CLAMP_HIGH, val)))
        except (OverflowError, FloatingPointError):
            return float(_IMPROBABILITY_CLAMP_HIGH)

    def compute_gradient(self, psi: float, roi: float) -> Tuple[float, float]:
        p, r = float(psi), float(roi)
        if p <= 0 or r <= 0: raise AxiomViolationError("Gradient only valid for (R+)^2")
        
        # ∂I/∂Ψ = -γ·κ·(ROI^γ)·(Ψ^{-γ-1})
        # ∂I/∂ROI = γ·κ·(ROI^{γ-1})·(Ψ^{-γ})
        d_psi = -self.gamma * self.kappa * (r ** self.gamma) * (p ** (-self.gamma - 1))
        d_roi = self.gamma * self.kappa * (r ** (self.gamma - 1)) * (p ** (-self.gamma))
        return float(d_psi), float(d_roi)
        
    def verify_lipschitz_constant(self, roi_max: float, psi_min: float) -> float:
        return float(self.kappa * self.gamma * ((roi_max / psi_min) ** (self.gamma - 1)))

    def to_dict(self) -> Dict[str, float]: return {"kappa": self.kappa, "gamma": self.gamma}
    def to_json(self) -> str: return json.dumps(self.to_dict())
    @classmethod
    def from_dict(cls, d): return cls(kappa=d["kappa"], gamma=d["gamma"])
    @classmethod
    def from_json(cls, s): return cls.from_dict(json.loads(s))

    def __matmul__(self, other: ImprobabilityTensor) -> ImprobabilityTensor:
        return ImprobabilityTensor(kappa=self.kappa * other.kappa, gamma=self.gamma + other.gamma)

    def __rmul__(self, scalar: float) -> ImprobabilityTensor:
        return ImprobabilityTensor(kappa=float(scalar) * self.kappa, gamma=self.gamma)
        
    def batch_compute(self, psi_arr: NDArray, roi_arr: NDArray) -> NDArray:
        res = []
        for p, r in zip(psi_arr, roi_arr): res.append(self.compute_penalty(p, r))
        return np.array(res)

@dataclass(frozen=True, slots=True)
class ImprobabilityResult:
    success: bool
    improbability_penalty: Optional[float] = None
    is_vetoed: Optional[bool] = None
    error_type: Optional[str] = None
    error: Optional[str] = None

    @classmethod
    def success_result(cls, penalty: float, **kwargs):
        is_vetoed = penalty >= _IMPROBABILITY_CLAMP_HIGH * _VETO_THRESHOLD_FACTOR
        return cls(success=True, improbability_penalty=penalty, is_vetoed=is_vetoed)
    @classmethod
    def error_result(cls, error_type: str, msg: str):
        return cls(success=False, error_type=error_type, error=msg)
    def to_dict(self):
        return {"success": self.success, "improbability_penalty": self.improbability_penalty,
                "is_vetoed": self.is_vetoed, "error_type": self.error_type, "error": self.error, "metadata": {}}

class ImprobabilityDriveService:
    def __init__(self, mic_registry, kappa=1.0, gamma=2.0):
        self.mic = mic_registry
        self.tensor = ImprobabilityTensor(kappa, gamma)
    def update_hyperparameters(self, kappa=None, gamma=None):
        k = kappa if kappa is not None else self.tensor.kappa
        g = gamma if gamma is not None else self.tensor.gamma
        self.tensor = ImprobabilityTensor(k, g)
    def register_in_mic(self):
        self.mic.register_vector("compute_improbability_penalty", Stratum.OMEGA, self._handler)
    def _handler(self, **kwargs):
        try:
            p = kwargs.get("psi")
            r = kwargs.get("roi")
            if p is None or r is None: raise DimensionalMismatchError("Missing psi/roi")
            penalty = self.tensor.compute_penalty(p, r)
            return ImprobabilityResult.success_result(penalty).to_dict()
        except ImprobabilityDriveError as e:
            return ImprobabilityResult.error_result(type(e).__name__, str(e)).to_dict()
        except Exception as e:
            return ImprobabilityResult.error_result("InternalError", str(e)).to_dict()
    def compute_with_gradient(self, psi, roi):
        penalty = self.tensor.compute_penalty(psi, roi)
        grads = self.tensor.compute_gradient(psi, roi)
        return {"penalty": penalty, "gradients": {"d_penalty_d_psi": grads[0], "d_penalty_d_roi": grads[1]}}

class TensorAlgebra:
    @staticmethod
    def compose(tensors: Iterable[ImprobabilityTensor]) -> ImprobabilityTensor:
        res = ImprobabilityTensor(1.0, 2.0)
        if not tensors: return res
        k, g = 1.0, 0.0
        for t in tensors:
            k *= t.kappa
            g += t.gamma
        return ImprobabilityTensor(k, g if g > 0 else 2.0)
    @staticmethod
    def average(tensors: Iterable[ImprobabilityTensor]) -> ImprobabilityTensor:
        ts = list(tensors)
        return ImprobabilityTensor(sum(t.kappa for t in ts)/len(ts), sum(t.gamma for t in ts)/len(ts))
    @staticmethod
    def interpolate(t1, t2, t):
        return ImprobabilityTensor(t1.kappa*(1-t) + t2.kappa*t, t1.gamma*(1-t) + t2.gamma*t)

class TensorFactory:
    @staticmethod
    def create(preset):
        if preset == "conservative": return ImprobabilityTensor(1.0, 1.0)
        if preset == "moderate": return ImprobabilityTensor(1.0, 2.0)
        if preset == "aggressive": return ImprobabilityTensor(2.0, 3.0)
        raise ValueError(f"Preset desconocido: {preset}")

# ==============================================================================
# TESTS (VARIEDAD COMPLETA)
# ==============================================================================
@pytest.fixture
def active_mic_registry():
    registry = MICRegistry()
    service = ImprobabilityDriveService(registry, kappa=1.0, gamma=2.0)
    service.register_in_mic()
    return registry

@pytest.fixture
def standard_tensor(): return ImprobabilityTensor(kappa=1.0, gamma=2.0)

@pytest.mark.integration
class TestMICImprobabilityCoupling:
    def test_mic_orthogonal_coupling_and_registration(self, active_mic_registry):
        vector = active_mic_registry.get_basis_vector("compute_improbability_penalty")
        assert vector is not None
        assert vector.target_stratum == Stratum.OMEGA

    @pytest.mark.parametrize("psi,roi,expected_veto", [
        (0.95, 1.5, False),
        (0.01, 10.0, True),
        (_EPS_MACH, 3.0, True),
    ])
    def test_functorial_projection_parametric(self, active_mic_registry, psi, roi, expected_veto):
        # Inyectar estratos validados para cumplir clausura transitiva
        ctx = {"validated_strata": frozenset({Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY})}
        res = active_mic_registry.project_intent("compute_improbability_penalty", psi=psi, roi=roi, context=ctx)
        assert res["success"], f"Error: {res.get('error')}"
        assert res["is_vetoed"] == expected_veto

    def test_monadic_error_isolation(self, active_mic_registry):
        ctx = {"validated_strata": frozenset({Stratum.PHYSICS, Stratum.TACTICS, Stratum.STRATEGY})}
        res = active_mic_registry.project_intent("compute_improbability_penalty", psi="corrupt", roi=1.5, context=ctx)
        assert not res["success"]
        assert res["error_type"] == "TypeCoercionError"

@pytest.mark.unit
class TestImprobabilityTensorProperties:
    def test_monotonicity(self, standard_tensor):
        # ROI creciente
        assert standard_tensor.compute_penalty(1.0, 2.0) > standard_tensor.compute_penalty(1.0, 1.0)
        # PSI creciente (penalidad decreciente)
        assert standard_tensor.compute_penalty(2.0, 10.0) < standard_tensor.compute_penalty(1.0, 10.0)

    def test_lipschitz_verification(self, standard_tensor):
        roi_max, psi_min = 1000.0, 0.01
        L = standard_tensor.verify_lipschitz_constant(roi_max, psi_min)
        # Muestreo aleatorio para verificar la constante
        for _ in range(100):
            p1, r1 = np.random.uniform(psi_min, 10.0), np.random.uniform(0.1, roi_max)
            p2, r2 = np.random.uniform(psi_min, 10.0), np.random.uniform(0.1, roi_max)
            delta_I = abs(standard_tensor.compute_penalty(p1, r1) - standard_tensor.compute_penalty(p2, r2))
            delta_x = max(abs(p1 - p2), abs(r1 - r2))
            assert delta_I <= L * delta_x * 1.5 # Tolerancia por discretización

    def test_tensor_composition(self):
        t1, t2 = ImprobabilityTensor(2.0, 1.5), ImprobabilityTensor(3.0, 2.0)
        tc = t1 @ t2
        assert tc.kappa == 6.0 and tc.gamma == 3.5

@pytest.mark.concurrent
class TestThreadSafety:
    def test_concurrent_penalty_computation(self, standard_tensor):
        def worker():
            for _ in range(100): standard_tensor.compute_penalty(1.0, 2.0)
        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads: t.start()
        for t in threads: t.join()
        assert True # Si llega aquí sin crash, es thread-safe en lectura

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
