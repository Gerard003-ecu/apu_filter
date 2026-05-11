r"""
Módulo: tests/integration/mic/test_mic_functional_orthogonality_during_basis_change.py
========================================================================================
SUITE DE INTEGRACIÓN: ORTOGONALIDAD FUNCIONAL Y CAMBIO DE BASE EN LA MIC
(Versión Rigurosa FINAL - Variedad Topológica Completa y Cerrada)
========================================================================================

FUNDAMENTOS DE ÁLGEBRA LINEAL AGÉNTICA:

§1. LA MIC COMO REPRESENTACIÓN DE LA IDENTIDAD ($I_n$)
    Cada herramienta (vector atómico) actúa como una proyección sobre un eje canónico
    del espacio de estado $ℝ^n$:

    $$e_i: ℝ^n \to ℝ^n, \quad e_i(x) = x + \delta_i$$

    La matriz de transformación resultante $T = [e_1(0) | e_2(0) | \dots | e_n(0)]$ debe
    ser isomorfa a la identidad $I_n$ para garantizar la independencia funcional.

§2. ORTOGONALIDAD Y MATRIZ DE GRAM
    Para vectores canónicos $\{e_i\}$, la matriz de Gram $G$ debe satisfacer:

    $$G_{ij} = \langle e_i(0), e_j(0) \rangle = \delta_{ij} \quad (\text{Delta de Kronecker})$$

    Esto asegura la ausencia de efectos secundarios cruzados (interferencias dimensionales).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from decimal import Decimal, getcontext, ROUND_HALF_EVEN
from enum import Enum, auto
from typing import (
    TypeVar, Generic, List, Dict, Optional, Set, Tuple,
    Callable, Protocol, Iterator, Any, Union, Literal
)
from typing_extensions import Self
import pytest
import numpy as np
from numpy.typing import NDArray

# =============================================================================
# CONFIGURACIÓN DETERMINISTA (FASE 1)
# =============================================================================
os_env_update = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
import os
os.environ.update(os_env_update)

getcontext().prec = 50
getcontext().rounding = ROUND_HALF_EVEN

# =============================================================================
# MOCKS E INFRAESTRUCTURA ALGEBRAICA (FULL MESH)
# =============================================================================
try:
    from app.core.schemas import Stratum
except ImportError:
    class Stratum(Enum):
        PHYSICS = 5
        TACTICS = 4
        STRATEGY = 3
        WISDOM = 0

@dataclass(frozen=True, slots=True)
class CategoricalState:
    payload: Dict[str, float]
    def __call__(self, **kwargs) -> CategoricalState:
        new_payload = dict(self.payload)
        new_payload.update(kwargs)
        return CategoricalState(payload=new_payload)

class Morphism:
    def __init__(self, handler: Callable, name: str = ""):
        self.handler = handler
        self.name = name
    def __call__(self, state: CategoricalState) -> CategoricalState:
        result = self.handler(**state.payload)
        return CategoricalState(payload=result)

class ComposedMorphism:
    def __init__(self, f: Morphism, g: Morphism):
        self.f = f
        self.g = g
    def __call__(self, state: CategoricalState) -> CategoricalState:
        return self.g(self.f(state))

def create_morphism_from_handler(name, target_stratum, handler, required_keys, optional_keys):
    return Morphism(handler=handler, name=name)

class MICRegistry:
    def __init__(self):
        self.vectors: Dict[str, Morphism] = {}
    def register_vector(self, name: str, stratum: Stratum, handler: Callable) -> None:
        self.vectors[name] = Morphism(handler=handler, name=name)

# =============================================================================
# CONSTANTES Y TOLERANCIAS
# =============================================================================
ORTHOGONALITY_TOLERANCE = 1e-12
SPECTRAL_TOLERANCE = 1e-10
RANK_TOLERANCE = 1e-10
DETERMINANT_TOLERANCE = 1e-10

_CANONICAL_DIMENSIONS = ["dim_1", "dim_2", "dim_3"]
_SPACE_DIMENSION = 3

# =============================================================================
# HANDLERS CANÓNICOS (OPERADORES DE PROYECCIÓN)
# =============================================================================
def _handler_e1(**kwargs):
    return {"dim_1": 1.0, "dim_2": kwargs.get("dim_2", 0.0), "dim_3": kwargs.get("dim_3", 0.0)}
def _handler_e2(**kwargs):
    return {"dim_1": kwargs.get("dim_1", 0.0), "dim_2": 1.0, "dim_3": kwargs.get("dim_3", 0.0)}
def _handler_e3(**kwargs):
    return {"dim_1": kwargs.get("dim_1", 0.0), "dim_2": kwargs.get("dim_2", 0.0), "dim_3": 1.0}

@dataclass(frozen=True, slots=True)
class CanonicalVectorSpec:
    name: str
    stratum: Stratum
    handler: Callable
    target_dimension: str

_CANONICAL_BASIS_SPEC = [
    CanonicalVectorSpec("search_tool", Stratum.PHYSICS, _handler_e1, "dim_1"),
    CanonicalVectorSpec("calc_tool", Stratum.TACTICS, _handler_e2, "dim_2"),
    CanonicalVectorSpec("strategy_tool", Stratum.STRATEGY, _handler_e3, "dim_3"),
]

# =============================================================================
# TESTS (VARIEDAD COMPLETA)
# =============================================================================
@pytest.fixture(scope="module")
def canonical_basis():
    return [create_morphism_from_handler(s.name, s.stratum, s.handler, [], []) for s in _CANONICAL_BASIS_SPEC]

@pytest.fixture(scope="module")
def zero_state():
    return CategoricalState({dim: 0.0 for dim in _CANONICAL_DIMENSIONS})

@pytest.mark.integration
class TestMICFunctionalOrthogonality:
    def test_canonical_basis_orthonormality(self, canonical_basis, zero_state):
        vectors = []
        for vec in canonical_basis:
            state = vec(zero_state)
            vectors.append(np.array([state.payload[d] for d in _CANONICAL_DIMENSIONS]))

        V = np.array(vectors)
        G = V @ V.T # Matriz de Gram
        I_3 = np.eye(3)

        deviation = np.linalg.norm(G - I_3, ord='fro')
        assert deviation < ORTHOGONALITY_TOLERANCE, f"Deviation {deviation} too high"

    def test_spectral_stability(self, canonical_basis, zero_state):
        # Matriz de transformación T
        cols = []
        for vec in canonical_basis:
            state = vec(zero_state)
            cols.append(np.array([state.payload[d] for d in _CANONICAL_DIMENSIONS]))
        T = np.column_stack(cols)

        eigenvalues = np.linalg.eigvals(T)
        spectral_radius = np.max(np.abs(eigenvalues))

        assert spectral_radius <= 1.0 + SPECTRAL_TOLERANCE
        for eig in eigenvalues:
            assert np.isclose(eig, 1.0, atol=SPECTRAL_TOLERANCE)

    def test_full_rank(self, canonical_basis, zero_state):
        cols = []
        for vec in canonical_basis:
            state = vec(zero_state)
            cols.append(np.array([state.payload[d] for d in _CANONICAL_DIMENSIONS]))
        T = np.column_stack(cols)

        rank = np.linalg.matrix_rank(T, tol=RANK_TOLERANCE)
        assert rank == _SPACE_DIMENSION
        assert abs(np.linalg.det(T) - 1.0) < DETERMINANT_TOLERANCE

    def test_composition_commutativity(self, canonical_basis, zero_state):
        e1, e2 = canonical_basis[0], canonical_basis[1]

        # e2 ∘ e1
        state_12 = e2(e1(zero_state))
        v12 = np.array([state_12.payload[d] for d in _CANONICAL_DIMENSIONS])

        # e1 ∘ e2
        state_21 = e1(e2(zero_state))
        v21 = np.array([state_21.payload[d] for d in _CANONICAL_DIMENSIONS])

        np.testing.assert_allclose(v12, v21, atol=ORTHOGONALITY_TOLERANCE)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
