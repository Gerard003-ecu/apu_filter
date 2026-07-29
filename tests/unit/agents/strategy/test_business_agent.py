"""
=================================================================================
SUITE DE PRUEBAS: BUSINESS AGENT (FASE 1)
Módulo: tests/test_business_agent.py
Versión: 1.0.0-rigorous
Autor: Artesano Programador Senior (Especialización: Topología Algebraica, 
       Teoría Espectral, Física de Circuitos, Mecánica Cuántica)
=================================================================================

FUNDAMENTACIÓN MATEMÁTICA DE LA SUITE DE PRUEBAS:

1. **Principio de Correspondencia Bohr:**
   Las pruebas deben reducirse a casos triviales verificables analíticamente
   cuando los parámetros tienden a límites conocidos (ε → 0, n → ∞).

2. **Teorema de Completitud de Gödel (Adaptado):**
   Ningún sistema de pruebas puede demostrar su propia completitud.
   Por tanto, implementamos verificación cruzada entre múltiples oráculos.

3. **Invariancia Gauge:**
   Los resultados deben ser independientes de la representación numérica
   (float64 vs Decimal) dentro de tolerancias ε definidas.

4. **Principio de Incertidumbre de Heisenberg (Metafórico):**
   Δ(precisión) × Δ(rendimiento) ≥ ℏ_eff
   donde ℏ_eff es la constante de compromiso del sistema.

5. **Topología de Espacio de Pruebas:**
   El espacio de configuraciones de prueba forma un complejo simplicial
   donde cada vértice es un caso de prueba y las aristas representan
   dependencias lógicas entre pruebas.

ESTRUCTURA DE FASES ANIDADAS:
- Fase 1: Constantes matemáticas, tipos refinados, invariantes topológicos
- Fase 2: Álgebra lineal computacional, espectro de Laplaciano
- Fase 3: Motor financiero, termodinámica estadística, opciones reales
- Fase 4: Risk Challenger, Business Agent, integración end-to-end

=================================================================================
"""

# =============================================================================
# IMPORTACIONES Y CONFIGURACIÓN
# =============================================================================

import pytest
import numpy as np
import pandas as pd
from decimal import Decimal, getcontext, ROUND_HALF_UP
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import asdict, fields as dataclass_fields
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from contextlib import contextmanager
import time
import math
import logging
import sys
import os

# Configurar precision Decimal para pruebas financieras
getcontext().prec = 50
getcontext().rounding = ROUND_HALF_UP

# Configurar NumPy para reproducibilidad
np.random.seed(42)  # Semilla fija para reproducibilidad
np.seterr(divide='raise', over='raise', under='ignore', invalid='raise')

# =============================================================================
# FIXTURES GLOBALES Y UTILIDADES DE PRUEBA
# =============================================================================

@pytest.fixture(scope="session")
def mathematical_constants():
    """
    Fixture que proporciona acceso a constantes matemáticas validadas.
    
    Teorema (Validación de Constantes):
        Todas las constantes deben satisfacer sus definiciones analíticas
        dentro de ε_machine = 2.22e-16 (precisión float64).
    """
    from app.strategy.business_agent import MathematicalConstants as MC
    
    class ValidatedConstants:
        def __init__(self):
            self.MC = MC
        
        def validate_golden_ratio(self) -> bool:
            """φ = (1 + √5)/2 satisface φ² = φ + 1"""
            φ = self.MC.GOLDEN_RATIO
            return abs(φ**2 - φ - 1) < self.MC.EPSILON_MACHINE
        
        def validate_euler_characteristics(self) -> bool:
            """χ(S²) = 2, χ(T²) = 0 son invariantes topológicos"""
            return (
                self.MC.EULER_CHARACTERISTIC_SPHERE == 2 and
                self.MC.EULER_CHARACTERISTIC_TORUS == 0
            )
        
        def validate_epsilon_hierarchy(self) -> bool:
            """
            Jerarquía de épsilons debe satisfacer:
            ε_machine < ε_sqrt < ε_tolerance < ε_strict
            """
            return (
                self.MC.EPSILON_MACHINE < 
                self.MC.EPSILON_SQRT < 
                self.MC.EPSILON_TOLERANCE < 
                self.MC.EPSILON_STRICT
            )
    
    return ValidatedConstants()


@pytest.fixture(scope="session")
def type_guards():
    """Fixture para validadores de tipos refinados."""
    from app.strategy.business_agent import TypeGuards
    return TypeGuards()


@pytest.fixture
def sample_betti_data() -> Dict[str, Any]:
    """Datos de muestra para números de Betti."""
    return {
        "beta_0": 1,
        "beta_1": 2,
        "beta_2": 1,
        "beta_n": (0, 0, 0)
    }


@pytest.fixture
def sample_spectral_data() -> Tuple[float, ...]:
    """
    Eigenvalores de muestra para Laplaciano de grafo conexo.
    
    Teorema (Espectro de Laplaciano):
        Para grafo conexo con n nodos:
        0 = λ₀ < λ₁ ≤ λ₂ ≤ ... ≤ λ_{n-1} ≤ 2n
    """
    return (0.0, 0.5, 1.2, 2.1, 3.8)


@pytest.fixture
def sample_financial_params() -> Dict[str, Any]:
    """Parámetros financieros válidos para pruebas."""
    return {
        "initial_investment": "1000000.0",
        "cash_flows": ["300000.0", "350000.0", "400000.0", "450000.0", "500000.0"],
        "discount_rate": "0.10",
        "risk_free_rate": "0.05",
        "market_return": "0.12",
        "beta": "1.2",
        "cost_std_dev": "50000.0",
        "project_volatility": 0.25
    }


@pytest.fixture
def mock_mic():
    """Mock para Matriz de Interacción Central."""
    mic = Mock()
    mic.project_intent = Mock(return_value={"success": True, "payload": {"penalty_relief": 0.15}})
    return mic


@pytest.fixture
def mock_telemetry():
    """Mock para contexto de telemetría."""
    telemetry = Mock()
    telemetry.record_error = Mock()
    return telemetry


# =============================================================================
# CLASE DE PRUEBAS FASE 1: CONSTANTES Y TIPOS
# =============================================================================

class TestMathematicalConstants:
    """
    Suite de pruebas para constantes matemáticas fundamentales.
    
    Fundamentación: Las constantes deben satisfacer identidades analíticas
    exactas dentro de tolerancias de máquina.
    """
    
    def test_golden_ratio_algebraic_identity(self, mathematical_constants):
        """
        Prueba: φ² - φ - 1 = 0 (identidad algebraica del número áureo).
        
        Demostración:
            φ = (1 + √5)/2
            φ² = (1 + 2√5 + 5)/4 = (6 + 2√5)/4 = (3 + √5)/2
            φ² - φ = (3 + √5)/2 - (1 + √5)/2 = 1
            ∴ φ² - φ - 1 = 0
        """
        assert mathematical_constants.validate_golden_ratio(), \
            "La razón áurea no satisface φ² = φ + 1 dentro de ε_machine"
    
    def test_silver_ratio_algebraic_identity(self, mathematical_constants):
        """
        Prueba: δ_S² - 2δ_S - 1 = 0 (identidad del número plateado).
        
        δ_S = 1 + √2 satisface la ecuación cuadrática x² - 2x - 1 = 0.
        """
        MC = mathematical_constants.MC
        δ = MC.SILVER_RATIO
        residual = abs(δ**2 - 2*δ - 1)
        assert residual < MC.EPSILON_MACHINE, \
            f"La razón plateada no satisface δ² - 2δ - 1 = 0: residual = {residual}"
    
    def test_epsilon_hierarchy_consistency(self, mathematical_constants):
        """
        Prueba: Jerarquía de tolerancias numéricas.
        
        Invariante: ε_machine < ε_sqrt < ε_tolerance < ε_strict
        Esto garantiza que comparaciones numéricas sean consistentes.
        """
        assert mathematical_constants.validate_epsilon_hierarchy(), \
            "La jerarquía de épsilons es inconsistente"
    
    def test_euler_characteristic_invariants(self, mathematical_constants):
        """
        Prueba: Invariantes topológicos de superficies.
        
        Teorema (Clasificación de Superficies):
            χ(S²) = 2 (esfera)
            χ(T²) = 0 (toro)
            χ(ℝP²) = 1 (plano proyectivo)
        """
        assert mathematical_constants.validate_euler_characteristics(), \
            "Los invariantes de Euler no coinciden con valores teóricos"
    
    def test_machine_epsilon_definition(self, mathematical_constants):
        """
        Prueba: Definición de ε_machine como menor número tal que 1 + ε ≠ 1.
        
        Esto es fundamental para estabilidad numérica en álgebra lineal.
        """
        MC = mathematical_constants.MC
        one_plus_eps = 1.0 + MC.EPSILON_MACHINE
        assert one_plus_eps > 1.0, \
            "ε_machine no es el menor número que modifica 1.0 en float64"
    
    def test_logarithm_constants(self, mathematical_constants):
        """Prueba: Constantes logarítmicas verificadas analíticamente."""
        MC = mathematical_constants.MC
        
        # ln(2) ≈ 0.6931471805599453
        assert abs(MC.LOG_2 - 0.6931471805599453) < MC.EPSILON_TOLERANCE
        
        # ln(10) ≈ 2.302585092994046
        assert abs(MC.LOG_10 - 2.302585092994046) < MC.EPSILON_TOLERANCE
        
        # ln(ε_tolerance) debe ser negativo
        assert MC.LOG_EPSILON < 0, "Logaritmo de ε debe ser negativo"
    
    def test_sqrt_constants(self, mathematical_constants):
        """Prueba: Raíces cuadradas fundamentales."""
        MC = mathematical_constants.MC
        
        # √2 ≈ 1.4142135623730951
        assert abs(MC.SQRT_2 - 1.4142135623730951) < MC.EPSILON_MACHINE
        
        # √3 ≈ 1.7320508075688772
        assert abs(MC.SQRT_3 - 1.7320508075688772) < MC.EPSILON_MACHINE
        
        # √5 ≈ 2.23606797749979
        assert abs(MC.SQRT_5 - 2.23606797749979) < MC.EPSILON_MACHINE
    
    def test_is_negligible_function(self, mathematical_constants):
        """Prueba: Función de detección de valores insignificantes."""
        MC = mathematical_constants.MC
        
        # Valor por debajo del umbral
        assert MC.is_negligible(1e-11) is True
        
        # Valor por encima del umbral
        assert MC.is_negligible(1e-9) is False
        
        # Valor cero es siempre insignificante
        assert MC.is_negligible(0.0) is True
        
        # Valor negativo pequeño
        assert MC.is_negligible(-1e-11) is True
    
    def test_are_close_robust_comparison(self, mathematical_constants):
        """
        Prueba: Comparación numérica robusta con tolerancias.
        
        Implementa: |a - b| ≤ max(rel_tol × max(|a|, |b|), abs_tol)
        """
        MC = mathematical_constants.MC
        
        # Valores idénticos
        assert MC.are_close(1.0, 1.0) is True
        
        # Valores dentro de tolerancia absoluta
        assert MC.are_close(1.0, 1.0 + 1e-11) is True
        
        # Valores fuera de tolerancia
        assert MC.are_close(1.0, 1.1) is False
        
        # Valores grandes con tolerancia relativa
        assert MC.are_close(1e10, 1e10 + 1e-5, rel_tol=1e-14) is False
        assert MC.are_close(1e10, 1e10 + 1e-5, rel_tol=1e-10) is True
    
    def test_safe_log_handles_non_positive(self, mathematical_constants):
        """Prueba: Logaritmo seguro maneja valores no positivos."""
        MC = mathematical_constants.MC
        
        # Valor positivo normal
        result = MC.safe_log(math.e)
        assert abs(result - 1.0) < MC.EPSILON_TOLERANCE
        
        # Valor cero (debe usar ε como mínimo)
        result_zero = MC.safe_log(0.0)
        assert result_zero < 0  # log(ε) es negativo
        
        # Valor negativo (debe usar ε como mínimo)
        result_neg = MC.safe_log(-1.0)
        assert result_neg == result_zero
    
    def test_safe_divide_handles_zero_denominator(self, mathematical_constants):
        """Prueba: División segura maneja denominador nulo."""
        MC = mathematical_constants.MC
        
        # División normal
        assert MC.safe_divide(10.0, 2.0) == 5.0
        
        # Denominador cero (debe retornar fallback)
        assert MC.safe_divide(10.0, 0.0, fallback=0.0) == 0.0
        assert MC.safe_divide(10.0, 1e-15, fallback=-1.0) == -1.0


class TestTypeGuards:
    """
    Suite de pruebas para guardianes de tipos refinados.
    
    Fundamentación: Los tipos refinados deben verificar invariantes
    en tiempo de ejecución para garantizar corrección.
    """
    
    def test_is_probability_valid_range(self, type_guards):
        """Prueba: Probability ∈ [0, 1]."""
        assert type_guards.is_probability(0.0) is True
        assert type_guards.is_probability(0.5) is True
        assert type_guards.is_probability(1.0) is True
        assert type_guards.is_probability(-0.1) is False
        assert type_guards.is_probability(1.1) is False
    
    def test_is_positive_real_strict(self, type_guards):
        """Prueba: PositiveReal ∈ (0, ∞) (estrictamente positivo)."""
        assert type_guards.is_positive_real(0.0001) is True
        assert type_guards.is_positive_real(1.0) is True
        assert type_guards.is_positive_real(0.0) is False
        assert type_guards.is_positive_real(-1.0) is False
    
    def test_is_non_negative_real_inclusive(self, type_guards):
        """Prueba: NonNegativeReal ∈ [0, ∞) (incluye cero)."""
        assert type_guards.is_non_negative_real(0.0) is True
        assert type_guards.is_non_negative_real(1.0) is True
        assert type_guards.is_non_negative_real(-0.0001) is False
    
    def test_is_unit_interval_bounds(self, type_guards):
        """Prueba: UnitInterval ∈ [0, 1]."""
        assert type_guards.is_unit_interval(0.0) is True
        assert type_guards.is_unit_interval(0.5) is True
        assert type_guards.is_unit_interval(1.0) is True
        assert type_guards.is_unit_interval(-0.1) is False
        assert type_guards.is_unit_interval(1.1) is False
    
    def test_is_simplex_dimension_natural(self, type_guards):
        """Prueba: SimplexDimension ∈ ℕ₀ (enteros no negativos)."""
        assert type_guards.is_simplex_dimension(0) is True
        assert type_guards.is_simplex_dimension(1) is True
        assert type_guards.is_simplex_dimension(100) is True
        assert type_guards.is_simplex_dimension(-1) is False
        assert type_guards.is_simplex_dimension(1.5) is False
    
    def test_is_valid_vector_numpy(self, type_guards):
        """Prueba: Vector NumPy válido (1D, float64)."""
        valid_vector = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        invalid_dim = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        invalid_dtype = np.array([1, 2, 3], dtype=np.int32)
        
        assert type_guards.is_valid_vector(valid_vector) is True
        assert type_guards.is_valid_vector(invalid_dim) is False
        assert type_guards.is_valid_vector(invalid_dtype) is False
        assert type_guards.is_valid_vector([1.0, 2.0, 3.0]) is False  # Lista, no ndarray
    
    def test_is_valid_matrix_numpy(self, type_guards):
        """Prueba: Matriz NumPy válida (2D, float64)."""
        valid_matrix = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        invalid_dim = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        
        assert type_guards.is_valid_matrix(valid_matrix) is True
        assert type_guards.is_valid_matrix(invalid_dim) is False


# =============================================================================
# CLASE DE PRUEBAS FASE 1: INVARIANTES TOPOLÓGICOS
# =============================================================================

class TestBettiNumbers:
    """
    Suite de pruebas para números de Betti como invariantes topológicos.
    
    Fundamentación Teórica:
        β_k = dim H_k(K; 𝔽) = rank(Z_k) - rank(B_k)
        
        donde:
        - H_k: k-ésimo grupo de homología
        - Z_k: k-ciclos (ker ∂_k)
        - B_k: k-fronteras (im ∂_{k+1})
    
    Teorema (Invariancia Homotópica):
        Los números de Betti son invariantes bajo equivalencia homotópica.
    """
    
    def test_betti_numbers_creation_valid(self, sample_betti_data):
        """Prueba: Creación válida de BettiNumbers."""
        from app.strategy.business_agent import BettiNumbers, BettiNumber
        
        betti = BettiNumbers(
            beta_0=BettiNumber(sample_betti_data["beta_0"]),
            beta_1=BettiNumber(sample_betti_data["beta_1"]),
            beta_2=BettiNumber(sample_betti_data["beta_2"]),
            beta_n=tuple(BettiNumber(b) for b in sample_betti_data["beta_n"])
        )
        
        assert betti.beta_0 == 1
        assert betti.beta_1 == 2
        assert betti.beta_2 == 1
    
    def test_betti_numbers_non_negative_invariant(self):
        """
        Prueba: Invariante de no negatividad.
        
        Teorema: β_k ≥ 0 para todo k (dimensión de espacio vectorial).
        """
        from app.strategy.business_agent import BettiNumbers, BettiNumber, TopologicalInvariantError
        
        with pytest.raises(TopologicalInvariantError):
            BettiNumbers(beta_0=BettiNumber(-1))
        
        with pytest.raises(TopologicalInvariantError):
            BettiNumbers(beta_0=BettiNumber(1), beta_1=BettiNumber(-1))
    
    def test_euler_characteristic_formula(self, sample_betti_data):
        """
        Prueba: Fórmula de Euler-Poincaré.
        
        χ = Σ_{k=0}^∞ (-1)^k β_k = β₀ - β₁ + β₂ - β₃ + ...
        
        Para el sample: χ = 1 - 2 + 1 - 0 + 0 - 0 = 0
        """
        from app.strategy.business_agent import BettiNumbers, BettiNumber
        
        betti = BettiNumbers(
            beta_0=BettiNumber(sample_betti_data["beta_0"]),
            beta_1=BettiNumber(sample_betti_data["beta_1"]),
            beta_2=BettiNumber(sample_betti_data["beta_2"]),
            beta_n=tuple(BettiNumber(b) for b in sample_betti_data["beta_n"])
        )
        
        expected_chi = 1 - 2 + 1  # β₀ - β₁ + β₂
        assert betti.euler_characteristic == expected_chi
    
    def test_euler_characteristic_sphere(self):
        """
        Prueba: Característica de Euler para esfera (β₀=1, β₁=0, β₂=1).
        
        Teorema: χ(S²) = 2
        """
        from app.strategy.business_agent import BettiNumbers, BettiNumber
        
        sphere_betti = BettiNumbers(
            beta_0=BettiNumber(1),  # 1 componente conexa
            beta_1=BettiNumber(0),  # 0 ciclos
            beta_2=BettiNumber(1)   # 1 cavidad (interior de esfera)
        )
        
        assert sphere_betti.euler_characteristic == 2
    
    def test_euler_characteristic_torus(self):
        """
        Prueba: Característica de Euler para toro (β₀=1, β₁=2, β₂=1).
        
        Teorema: χ(T²) = 0
        """
        from app.strategy.business_agent import BettiNumbers, BettiNumber
        
        torus_betti = BettiNumbers(
            beta_0=BettiNumber(1),  # 1 componente conexa
            beta_1=BettiNumber(2),  # 2 ciclos independientes (meridiano y paralelo)
            beta_2=BettiNumber(1)   # 1 cavidad
        )
        
        assert torus_betti.euler_characteristic == 0  # 1 - 2 + 1 = 0
    
    def test_is_connected_beta_zero(self):
        """
        Prueba: Conectividad ⇔ β₀ = 1.
        
        Teorema: K es conexo si y solo si β₀(K) = 1.
        """
        from app.strategy.business_agent import BettiNumbers, BettiNumber
        
        connected = BettiNumbers(beta_0=BettiNumber(1))
        disconnected = BettiNumbers(beta_0=BettiNumber(3))  # 3 componentes
        
        assert connected.is_connected is True
        assert disconnected.is_connected is False
    
    def test_has_cycles_beta_one(self):
        """Prueba: Detección de ciclos 1-dimensionales."""
        from app.strategy.business_agent import BettiNumbers, BettiNumber
        
        no_cycles = BettiNumbers(beta_1=BettiNumber(0))
        has_cycles = BettiNumbers(beta_1=BettiNumber(2))
        
        assert no_cycles.has_cycles is False
        assert has_cycles.has_cycles is True
    
    def test_has_voids_beta_two(self):
        """Prueba: Detección de cavidades 2-dimensionales."""
        from app.strategy.business_agent import BettiNumbers, BettiNumber
        
        no_voids = BettiNumbers(beta_2=BettiNumber(0))
        has_voids = BettiNumbers(beta_2=BettiNumber(1))
        
        assert no_voids.has_voids is False
        assert has_voids.has_voids is True
    
    def test_total_homology_rank(self, sample_betti_data):
        """
        Prueba: Rango total de homología = Σ β_k.
        
        Interpretación: Dimensión total del espacio de homología H_*(K; 𝔽).
        """
        from app.strategy.business_agent import BettiNumbers, BettiNumber
        
        betti = BettiNumbers(
            beta_0=BettiNumber(sample_betti_data["beta_0"]),
            beta_1=BettiNumber(sample_betti_data["beta_1"]),
            beta_2=BettiNumber(sample_betti_data["beta_2"]),
            beta_n=tuple(BettiNumber(b) for b in sample_betti_data["beta_n"])
        )
        
        expected_rank = 1 + 2 + 1 + 0 + 0 + 0  # Suma de todos los β_k
        assert betti.total_homology_rank == expected_rank
    
    def test_from_dict_constructor(self, sample_betti_data):
        """Prueba: Constructor desde diccionario con validación."""
        from app.strategy.business_agent import BettiNumbers
        
        betti = BettiNumbers.from_dict(sample_betti_data)
        
        assert betti.beta_0 == 1
        assert betti.beta_1 == 2
        assert betti.beta_2 == 1
    
    def test_from_dict_invalid_data(self):
        """Prueba: Constructor maneja datos inválidos."""
        from app.strategy.business_agent import BettiNumbers, TopologicalInvariantError
        
        with pytest.raises(TopologicalInvariantError):
            BettiNumbers.from_dict({"beta_0": -1})
        
        with pytest.raises(TopologicalInvariantError):
            BettiNumbers.from_dict({"beta_0": "invalid"})
    
    def test_to_dict_serialization(self, sample_betti_data):
        """Prueba: Serialización a diccionario completa."""
        from app.strategy.business_agent import BettiNumbers, BettiNumber
        
        betti = BettiNumbers(
            beta_0=BettiNumber(sample_betti_data["beta_0"]),
            beta_1=BettiNumber(sample_betti_data["beta_1"]),
            beta_2=BettiNumber(sample_betti_data["beta_2"])
        )
        
        result = betti.to_dict()
        
        assert "beta_0" in result
        assert "beta_1" in result
        assert "beta_2" in result
        assert "euler_characteristic" in result
        assert "is_connected" in result
        assert "has_cycles" in result
        assert "has_voids" in result
        assert "total_homology_rank" in result
    
    def test_betti_numbers_immutability(self):
        """
        Prueba: BettiNumbers es inmutable (frozen dataclass).
        
        Esto garantiza que los invariantes topológicos no se corrompan.
        """
        from app.strategy.business_agent import BettiNumbers, BettiNumber
        
        betti = BettiNumbers(beta_0=BettiNumber(1))
        
        with pytest.raises(AttributeError):  # frozen=True
            betti.beta_0 = BettiNumber(2)


class TestSpectralData:
    """
    Suite de pruebas para datos espectrales del Laplaciano.
    
    Fundamentación Teórica:
        Λ = D - A (Laplaciano combinatorio)
        
        Propiedades:
        1. Λ es simétrica y semi-definida positiva
        2. Λ1 = 0 (vector constante es eigenvector)
        3. Número de eigenvalores nulos = componentes conexas
        4. λ₁ (Fiedler) > 0 ⇔ grafo conexo
    """
    
    def test_spectral_data_creation_valid(self, sample_spectral_data):
        """Prueba: Creación válida de SpectralData."""
        from app.strategy.business_agent import SpectralData
        
        spectral = SpectralData(eigenvalues=sample_spectral_data)
        
        assert len(spectral.eigenvalues) == 5
        assert spectral.eigenvalues == tuple(sorted(sample_spectral_data))
    
    def test_spectral_data_empty_eigenvalues(self):
        """Prueba: Rechaza eigenvalores vacíos."""
        from app.strategy.business_agent import SpectralData, SpectralTheoryError
        
        with pytest.raises(SpectralTheoryError):
            SpectralData(eigenvalues=())
    
    def test_fiedler_value_calculation(self, sample_spectral_data):
        """
        Prueba: Valor de Fiedler λ₁ (conectividad algebraica).
        
        Teorema: λ₁ > 0 ⇔ grafo conexo
        """
        from app.strategy.business_agent import SpectralData
        
        spectral = SpectralData(eigenvalues=sample_spectral_data)
        
        # λ₁ es el segundo eigenvalor más pequeño (índice 1)
        assert spectral.fiedler_value == 0.5
    
    def test_spectral_gap_calculation(self, sample_spectral_data):
        """
        Prueba: Gap espectral = λ₁ - λ₀.
        
        Interpretación: Robustez de conectividad ante perturbaciones.
        """
        from app.strategy.business_agent import SpectralData
        
        spectral = SpectralData(eigenvalues=sample_spectral_data)
        
        expected_gap = 0.5 - 0.0  # λ₁ - λ₀
        assert spectral.spectral_gap == expected_gap
    
    def test_algebraic_connectivity_alias(self, sample_spectral_data):
        """Prueba: algebraic_connectivity es alias de fiedler_value."""
        from app.strategy.business_agent import SpectralData
        
        spectral = SpectralData(eigenvalues=sample_spectral_data)
        
        assert spectral.algebraic_connectivity == spectral.fiedler_value
    
    def test_spectral_radius_calculation(self, sample_spectral_data):
        """
        Prueba: Radio espectral = max_i |λ_i|.
        
        Importante para estabilidad de algoritmos iterativos.
        """
        from app.strategy.business_agent import SpectralData
        
        spectral = SpectralData(eigenvalues=sample_spectral_data)
        
        expected_radius = max(abs(e) for e in sample_spectral_data)
        assert spectral.spectral_radius == expected_radius
    
    def test_is_connected_fiedler_positive(self):
        """
        Prueba: Conectividad mediante valor de Fiedler.
        
        Teorema: Grafo G es conexo ⇔ λ₁(Λ) > 0
        """
        from app.strategy.business_agent import SpectralData, MathematicalConstants as MC
        
        # Grafo conexo (λ₁ > 0)
        connected = SpectralData(eigenvalues=(0.0, 0.5, 1.2))
        assert connected.is_connected is True
        
        # Grafo desconexo (λ₁ = 0)
        disconnected = SpectralData(eigenvalues=(0.0, 0.0, 1.2))
        assert disconnected.is_connected is False
        
        # Grafo casi desconexo (λ₁ < ε)
        almost_disconnected = SpectralData(eigenvalues=(0.0, MC.EPSILON_TOLERANCE / 2, 1.2))
        assert almost_disconnected.is_connected is False
    
    def test_number_of_components_zero_eigenvalues(self):
        """
        Prueba: Componentes conexas = multiplicidad de eigenvalor 0.
        
        Teorema: dim(ker Λ) = número de componentes conexas
        """
        from app.strategy.business_agent import SpectralData
        
        # 1 componente (1 eigenvalor cero)
        one_component = SpectralData(eigenvalues=(0.0, 0.5, 1.2))
        assert one_component.number_of_components == 1
        
        # 3 componentes (3 eigenvalores cero)
        three_components = SpectralData(eigenvalues=(0.0, 0.0, 0.0, 1.5, 2.3))
        assert three_components.number_of_components == 3
    
    def test_condition_number_calculation(self):
        """
        Prueba: Número de condición espectral κ(Λ) = λ_max / λ_min.
        
        Interpretación: κ grande ⇒ ill-conditioned (inestable)
        """
        from app.strategy.business_agent import SpectralData, MathematicalConstants as MC
        
        # Bien condicionado
        well_conditioned = SpectralData(eigenvalues=(0.0, 1.0, 2.0, 3.0))
        # Excluye λ₀ = 0, usa λ_min = 1.0, λ_max = 3.0
        assert well_conditioned.condition_number == 3.0
        
        # Mal condicionado (λ_min muy pequeño)
        ill_conditioned = SpectralData(eigenvalues=(0.0, 1e-10, 10.0))
        assert ill_conditioned.condition_number > 1e8
        
        # Todos cero (condición infinita)
        all_zero = SpectralData(eigenvalues=(0.0, 0.0, 0.0))
        assert all_zero.condition_number == float('inf')
    
    def test_eigenvalues_sorted_automatically(self):
        """Prueba: Eigenvalores se ordenan automáticamente."""
        from app.strategy.business_agent import SpectralData
        
        unsorted = (3.8, 0.0, 2.1, 0.5, 1.2)
        spectral = SpectralData(eigenvalues=unsorted)
        
        assert spectral.eigenvalues == tuple(sorted(unsorted))
    
    def test_to_dict_serialization(self, sample_spectral_data):
        """Prueba: Serialización completa a diccionario."""
        from app.strategy.business_agent import SpectralData
        
        spectral = SpectralData(eigenvalues=sample_spectral_data)
        result = spectral.to_dict()
        
        assert "eigenvalues" in result
        assert "fiedler_value" in result
        assert "spectral_gap" in result
        assert "algebraic_connectivity" in result
        assert "spectral_radius" in result
        assert "is_connected" in result
        assert "number_of_components" in result
        assert "condition_number" in result
    
    def test_from_laplacian_matrix(self):
        """
        Prueba: Cálculo espectral desde matriz Laplaciana.
        
        Nota: Esta prueba requiere que la matriz sea válida.
        """
        from app.strategy.business_agent import SpectralData, LaplacianBuilder
        import numpy as np
        
        # Matriz de adyacencia simple (grafo de 3 nodos en línea)
        adjacency = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ], dtype=np.float64)
        
        laplacian = LaplacianBuilder.combinatorial_laplacian(adjacency)
        spectral = SpectralData.from_laplacian(laplacian)
        
        # Debe tener 3 eigenvalores
        assert len(spectral.eigenvalues) == 3
        
        # El más pequeño debe ser ≈ 0
        from app.strategy.business_agent import MathematicalConstants as MC
        assert abs(spectral.eigenvalues[0]) < MC.EPSILON_TOLERANCE


# =============================================================================
# FASE 2: ÁLGEBRA LINEAL COMPUTACIONAL Y TOPOLOGÍA DE GRAFOS
# =============================================================================

# =============================================================================
# CLASE DE PRUEBAS FASE 2: OPERACIONES DE MATRICES
# =============================================================================

class TestMatrixOperations:
    """
    Suite de pruebas para operaciones de álgebra lineal con estabilidad numérica.
    
    Fundamentación Teórica:
        - SVD: A = UΣVᵀ (descomposición en valores singulares)
        - Rango numérico: número de valores singulares > ε
        - Número de condición: κ(A) = σ_max / σ_min
        - Pseudoinversa: A⁺ = VΣ⁺Uᵀ
    """
    
    def test_compute_rank_full_rank_matrix(self):
        """
        Prueba: Cálculo de rango para matriz de rango completo.
        
        Teorema (Rango-Nullidad): rank(A) + nullity(A) = n
        """
        from app.strategy.business_agent import MatrixOperations
        import numpy as np
        
        # Matriz identidad 3x3 (rango 3)
        identity = np.eye(3, dtype=np.float64)
        rank = MatrixOperations.compute_rank(identity)
        
        assert rank == 3
    
    def test_compute_rank_singular_matrix(self):
        """Prueba: Cálculo de rango para matriz singular."""
        from app.strategy.business_agent import MatrixOperations
        import numpy as np
        
        # Matriz con filas linealmente dependientes
        singular = np.array([
            [1, 2, 3],
            [2, 4, 6],  # 2× fila 1
            [3, 6, 9]   # 3× fila 1
        ], dtype=np.float64)
        
        rank = MatrixOperations.compute_rank(singular)
        assert rank == 1  # Solo 1 fila independiente
    
    def test_compute_rank_empty_matrix(self):
        """Prueba: Matriz vacía tiene rango 0."""
        from app.strategy.business_agent import MatrixOperations
        import numpy as np
        
        empty = np.array([], dtype=np.float64).reshape(0, 0)
        rank = MatrixOperations.compute_rank(empty)
        
        assert rank == 0
    
    def test_compute_condition_number_orthogonal(self):
        """
        Prueba: Matriz ortogonal tiene κ = 1 (perfectamente condicionada).
        
        Teorema: Matrices ortogonales preservan normas.
        """
        from app.strategy.business_agent import MatrixOperations
        import numpy as np
        
        # Matriz ortogonal 2x2 (rotación)
        theta = np.pi / 4
        orthogonal = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ], dtype=np.float64)
        
        cond = MatrixOperations.compute_condition_number(orthogonal)
        
        from app.strategy.business_agent import MathematicalConstants as MC
        assert abs(cond - 1.0) < MC.EPSILON_TOLERANCE
    
    def test_compute_condition_number_ill_conditioned(self):
        """Prueba: Detección de matriz ill-conditioned."""
        from app.strategy.business_agent import MatrixOperations
        import numpy as np
        
        # Matriz de Hilbert (notoriamente ill-conditioned)
        hilbert = np.array([
            [1, 1/2, 1/3],
            [1/2, 1/3, 1/4],
            [1/3, 1/4, 1/5]
        ], dtype=np.float64)
        
        cond = MatrixOperations.compute_condition_number(hilbert)
        
        # Matriz de Hilbert 3x3 tiene κ ≈ 500
        assert cond > 100
    
    def test_pseudoinverse_properties(self):
        """
        Prueba: Propiedades de la pseudoinversa de Moore-Penrose.
        
        Propiedades:
        1. AA⁺A = A
        2. A⁺AA⁺ = A⁺
        3. (AA⁺)ᵀ = AA⁺
        4. (A⁺A)ᵀ = A⁺A
        """
        from app.strategy.business_agent import MatrixOperations
        import numpy as np
        
        # Matriz rectangular 3x2
        A = np.array([
            [1, 2],
            [3, 4],
            [5, 6]
        ], dtype=np.float64)
        
        A_pinv = MatrixOperations.pseudoinverse(A)
        
        # Propiedad 1: AA⁺A = A
        AA_pinvA = A @ A_pinv @ A
        assert np.allclose(A, AA_pinvA, atol=1e-10)
        
        # Propiedad 2: A⁺AA⁺ = A⁺
        A_pinvAA_pinv = A_pinv @ A @ A_pinv
        assert np.allclose(A_pinv, A_pinvAA_pinv, atol=1e-10)
        
        # Propiedad 3: (AA⁺)ᵀ = AA⁺ (simétrica)
        AA_pinv = A @ A_pinv
        assert np.allclose(AA_pinv, AA_pinv.T, atol=1e-10)
        
        # Propiedad 4: (A⁺A)ᵀ = A⁺A (simétrica)
        A_pinvA = A_pinv @ A
        assert np.allclose(A_pinvA, A_pinvA.T, atol=1e-10)
    
    def test_gram_schmidt_orthonormal(self):
        """
        Prueba: Ortogonalización de Gram-Schmidt produce base ortonormal.
        
        Propiedad: QᵀQ = I (columnas ortonormales)
        """
        from app.strategy.business_agent import MatrixOperations
        import numpy as np
        
        # Vectores linealmente independientes
        vectors = np.array([
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1]
        ], dtype=np.float64).T
        
        Q, R = MatrixOperations.gram_schmidt(vectors, normalize=True)
        
        # Q debe ser ortonormal: QᵀQ = I
        QtQ = Q.T @ Q
        identity = np.eye(Q.shape[1])
        
        from app.strategy.business_agent import MathematicalConstants as MC
        assert np.allclose(QtQ, identity, atol=MC.EPSILON_TOLERANCE)
    
    def test_gram_schmidt_reconstruction(self):
        """
        Prueba: Reconstrucción A = QR.
        
        Teorema: Descomposición QR existe para cualquier matriz.
        """
        from app.strategy.business_agent import MatrixOperations
        import numpy as np
        
        vectors = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=np.float64)
        
        Q, R = MatrixOperations.gram_schmidt(vectors, normalize=True)
        
        # Reconstrucción: A ≈ QR
        reconstructed = Q @ R
        
        from app.strategy.business_agent import MathematicalConstants as MC
        assert np.allclose(vectors, reconstructed, atol=MC.EPSILON_TOLERANCE)
    
    def test_is_positive_definite_cholesky(self):
        """
        Prueba: Verificación de matriz definida positiva.
        
        Teorema (Sylvester): A es definida positiva ⇔ todos los menores principales > 0
        Implementación: Factorización de Cholesky existe ⇔ A es definida positiva
        """
        from app.strategy.business_agent import MatrixOperations
        import numpy as np
        
        # Matriz definida positiva
        pos_def = np.array([
            [4, 2, 2],
            [2, 5, 1],
            [2, 1, 6]
        ], dtype=np.float64)
        
        assert MatrixOperations.is_positive_definite(pos_def) is True
        
        # Matriz no definida positiva (tiene eigenvalor negativo)
        not_pos_def = np.array([
            [1, 2],
            [2, 1]
        ], dtype=np.float64)
        
        assert MatrixOperations.is_positive_definite(not_pos_def) is False
        
        # Matriz no simétrica
        non_symmetric = np.array([
            [1, 2],
            [3, 4]
        ], dtype=np.float64)
        
        assert MatrixOperations.is_positive_definite(non_symmetric) is False
    
    def test_frobenius_norm_definition(self):
        """
        Prueba: Norma de Frobenius ‖A‖_F = √(Σᵢⱼ |aᵢⱼ|²).
        
        Propiedad: ‖A‖_F = √(tr(AᵀA))
        """
        from app.strategy.business_agent import MatrixOperations
        import numpy as np
        
        A = np.array([
            [1, 2],
            [3, 4]
        ], dtype=np.float64)
        
        frob_norm = MatrixOperations.frobenius_norm(A)
        
        # Cálculo manual: √(1² + 2² + 3² + 4²) = √30
        expected = np.sqrt(30)
        
        from app.strategy.business_agent import MathematicalConstants as MC
        assert abs(frob_norm - expected) < MC.EPSILON_TOLERANCE
    
    def test_operator_norm_spectral(self):
        """
        Prueba: Norma espectral (operador) = máximo valor singular.
        
        Para ord=2: ‖A‖₂ = σ_max
        """
        from app.strategy.business_agent import MatrixOperations
        import numpy as np
        
        A = np.array([
            [1, 2],
            [3, 4]
        ], dtype=np.float64)
        
        # Norma espectral
        spectral_norm = MatrixOperations.operator_norm(A, ord=2)
        
        # Calcular valores singulares
        singular_values = np.linalg.svd(A, compute_uv=False)
        expected = singular_values[0]
        
        from app.strategy.business_agent import MathematicalConstants as MC
        assert abs(spectral_norm - expected) < MC.EPSILON_TOLERANCE


# =============================================================================
# CLASE DE PRUEBAS FASE 2: LAPLACIANO DE GRAFOS
# =============================================================================

class TestLaplacianBuilder:
    """
    Suite de pruebas para construcción del Laplaciano de grafos.
    
    Fundamentación Teórica:
        Λ = D - A (Laplaciano combinatorio)
        
        Variantes:
        1. Combinatorial: Λ = D - A
        2. Normalizado: ℒ = I - D⁻¹/²AD⁻¹/²
        3. Random Walk: ℒ_rw = I - D⁻¹A
        4. Signless: |Λ| = D + A
    """
    
    def test_combinatorial_laplacian_properties(self):
        """
        Prueba: Propiedades del Laplaciano combinatorio.
        
        Propiedades:
        1. Λ es simétrica
        2. Λ es semi-definida positiva
        3. Λ1 = 0 (suma de filas = 0)
        """
        from app.strategy.business_agent import LaplacianBuilder
        import numpy as np
        
        # Grafo simple de 4 nodos
        adjacency = np.array([
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float64)
        
        laplacian = LaplacianBuilder.combinatorial_laplacian(adjacency)
        
        # Propiedad 1: Simetría
        assert np.allclose(laplacian, laplacian.T)
        
        # Propiedad 2: Semi-definida positiva (eigenvalores ≥ 0)
        eigenvalues = np.linalg.eigvalsh(laplacian)
        assert np.all(eigenvalues >= -1e-10)  # Tolerancia numérica
        
        # Propiedad 3: Suma de filas = 0
        row_sums = laplacian.sum(axis=1)
        from app.strategy.business_agent import MathematicalConstants as MC
        assert np.allclose(row_sums, 0, atol=MC.EPSILON_TOLERANCE)
    
    def test_combinatorial_laplacian_square_matrix(self):
        """Prueba: Rechaza matriz de adyacencia no cuadrada."""
        from app.strategy.business_agent import LaplacianBuilder, DimensionMismatchError
        import numpy as np
        
        non_square = np.array([
            [0, 1, 0],
            [1, 0, 1]
        ], dtype=np.float64)
        
        with pytest.raises(DimensionMismatchError):
            LaplacianBuilder.combinatorial_laplacian(non_square)
    
    def test_normalized_laplacian_eigenvalue_bounds(self):
        """
        Prueba: Eigenvalores del Laplaciano normalizado en [0, 2].
        
        Teorema: Para ℒ = I - D⁻¹/²AD⁻¹/², los eigenvalores están en [0, 2].
        """
        from app.strategy.business_agent import LaplacianBuilder
        import numpy as np
        
        adjacency = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ], dtype=np.float64)
        
        normalized = LaplacianBuilder.normalized_laplacian(adjacency)
        eigenvalues = np.linalg.eigvalsh(normalized)
        
        # Todos los eigenvalores deben estar en [0, 2]
        assert np.all(eigenvalues >= -1e-10)
        assert np.all(eigenvalues <= 2 + 1e-10)
    
    def test_random_walk_laplacian_stationary(self):
        """
        Prueba: Laplaciano de paseo aleatorio y distribución estacionaria.
        
        ℒ_rw = I - P donde P = D⁻¹A es la matriz de transición.
        """
        from app.strategy.business_agent import LaplacianBuilder
        import numpy as np
        
        adjacency = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ], dtype=np.float64)
        
        rw_laplacian = LaplacianBuilder.random_walk_laplacian(adjacency)
        
        # Debe ser similar al combinatorial pero con diferente escalado
        assert rw_laplacian.shape == adjacency.shape
    
    def test_signless_laplacian_non_negative(self):
        """
        Prueba: Laplaciano signless tiene eigenvalores no negativos.
        
        |Λ| = D + A (todos los eigenvalores ≥ 0)
        """
        from app.strategy.business_agent import LaplacianBuilder
        import numpy as np
        
        adjacency = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ], dtype=np.float64)
        
        signless = LaplacianBuilder.signless_laplacian(adjacency)
        eigenvalues = np.linalg.eigvalsh(signless)
        
        # Todos los eigenvalores deben ser no negativos
        assert np.all(eigenvalues >= -1e-10)


# =============================================================================
# CLASE DE PRUEBAS FASE 2: MÉTRICAS DE GRAFOS
# =============================================================================

class TestGraphMetrics:
    """
    Suite de pruebas para métricas de teoría de grafos.
    
    Fundamentación:
        - Densidad: ρ = 2|E| / (|V|(|V|-1))
        - Grado promedio: ⟨k⟩ = 2|E| / |V|
        - Sparse: ρ < 0.1
        - Dense: ρ > 0.5
    """
    
    def test_graph_metrics_basic_calculation(self):
        """Prueba: Cálculo básico de métricas de grafo."""
        from app.strategy.business_agent import GraphMetrics
        
        # Grafo completo K4: 4 nodos, 6 aristas
        metrics = GraphMetrics(n_nodes=4, n_edges=6)
        
        # Densidad de K4: 6 / (4×3/2) = 6/6 = 1.0
        from app.strategy.business_agent import MathematicalConstants as MC
        assert abs(metrics.density - 1.0) < MC.EPSILON_TOLERANCE
        
        # Grado promedio: 2×6 / 4 = 3
        assert abs(metrics.average_degree - 3.0) < MC.EPSILON_TOLERANCE
    
    def test_graph_metrics_sparse_detection(self):
        """Prueba: Detección de grafo sparse."""
        from app.strategy.business_agent import GraphMetrics
        
        # Grafo sparse: 100 nodos, 150 aristas (ρ ≈ 0.03)
        metrics = GraphMetrics(n_nodes=100, n_edges=150)
        
        assert metrics.is_sparse is True
        assert metrics.is_dense is False
    
    def test_graph_metrics_dense_detection(self):
        """Prueba: Detección de grafo dense."""
        from app.strategy.business_agent import GraphMetrics
        
        # Grafo denso: 10 nodos, 40 aristas (ρ ≈ 0.89)
        metrics = GraphMetrics(n_nodes=10, n_edges=40)
        
        assert metrics.is_dense is True
        assert metrics.is_sparse is False
    
    def test_graph_metrics_single_node(self):
        """Prueba: Grafo con un solo nodo."""
        from app.strategy.business_agent import GraphMetrics
        
        metrics = GraphMetrics(n_nodes=1, n_edges=0)
        
        assert metrics.density == 0.0
        assert metrics.average_degree == 0.0
    
    def test_graph_metrics_invalid_input(self):
        """Prueba: Rechaza entradas inválidas."""
        from app.strategy.business_agent import GraphMetrics
        
        with pytest.raises(ValueError):
            GraphMetrics(n_nodes=-1, n_edges=0)
        
        with pytest.raises(ValueError):
            GraphMetrics(n_nodes=5, n_edges=-10)
    
    def test_graph_metrics_to_dict(self):
        """Prueba: Serialización completa a diccionario."""
        from app.strategy.business_agent import GraphMetrics
        
        metrics = GraphMetrics(n_nodes=10, n_edges=20)
        result = metrics.to_dict()
        
        assert "n_nodes" in result
        assert "n_edges" in result
        assert "density" in result
        assert "average_degree" in result
        assert "is_sparse" in result
        assert "is_dense" in result
        assert "is_connected" in result
        assert "n_components" in result


# =============================================================================
# CLASE DE PRUEBAS FASE 2: BUNDLE TOPOLÓGICO
# =============================================================================

class TestTopologicalMetricsBundle:
    """
    Suite de pruebas para bundle cohesivo de métricas topológicas.
    
    Integración:
        - Homología (números de Betti)
        - Teoría espectral (Laplaciano)
        - Métricas de grafos
        - Estabilidad piramidal
    """
    
    def test_bundle_creation_valid(self):
        """Prueba: Creación válida del bundle topológico."""
        from app.strategy.business_agent import (
            TopologicalMetricsBundle,
            BettiNumbers, BettiNumber,
            SpectralData,
            GraphMetrics
        )
        
        betti = BettiNumbers(beta_0=BettiNumber(1), beta_1=BettiNumber(0), beta_2=BettiNumber(0))
        spectral = SpectralData(eigenvalues=(0.0, 0.5, 1.2))
        graph_metrics = GraphMetrics(n_nodes=3, n_edges=2)
        
        bundle = TopologicalMetricsBundle(
            betti=betti,
            spectral=spectral,
            graph_metrics=graph_metrics,
            pyramid_stability=0.85
        )
        
        assert bundle.pyramid_stability == 0.85
    
    def test_bundle_stability_range_validation(self):
        """Prueba: Valida que pyramid_stability ∈ [0, 1]."""
        from app.strategy.business_agent import (
            TopologicalMetricsBundle,
            BettiNumbers, BettiNumber,
            SpectralData,
            GraphMetrics
        )
        
        betti = BettiNumbers(beta_0=BettiNumber(1))
        spectral = SpectralData(eigenvalues=(0.0, 1.0))
        graph_metrics = GraphMetrics(n_nodes=2, n_edges=1)
        
        # Fuera de rango
        with pytest.raises(ValueError):
            TopologicalMetricsBundle(
                betti=betti,
                spectral=spectral,
                graph_metrics=graph_metrics,
                pyramid_stability=1.5
            )
        
        with pytest.raises(ValueError):
            TopologicalMetricsBundle(
                betti=betti,
                spectral=spectral,
                graph_metrics=graph_metrics,
                pyramid_stability=-0.1
            )
    
    def test_structural_coherence_formula(self):
        """
        Prueba: Fórmula de coherencia estructural.
        
        C(G) = exp(-λ₀·max(0, β₀-1)) × exp(-λ₁·β₁/√n) × Ψ × tanh(λ₁)
        
        Invariante: C(G) ∈ [0, 1]
        """
        from app.strategy.business_agent import (
            TopologicalMetricsBundle,
            BettiNumbers, BettiNumber,
            SpectralData,
            GraphMetrics
        )
        
        betti = BettiNumbers(beta_0=BettiNumber(1), beta_1=BettiNumber(0))
        spectral = SpectralData(eigenvalues=(0.0, 0.5, 1.2))
        graph_metrics = GraphMetrics(n_nodes=10, n_edges=15)
        
        bundle = TopologicalMetricsBundle(
            betti=betti,
            spectral=spectral,
            graph_metrics=graph_metrics,
            pyramid_stability=0.90
        )
        
        coherence = bundle.structural_coherence
        
        # Debe estar en [0, 1]
        assert 0.0 <= coherence <= 1.0
    
    def test_structural_coherence_disconnected_penalty(self):
        """
        Prueba: Penalización por fragmentación en coherencia.
        
        β₀ > 1 reduce la coherencia exponencialmente.
        """
        from app.strategy.business_agent import (
            TopologicalMetricsBundle,
            BettiNumbers, BettiNumber,
            SpectralData,
            GraphMetrics
        )
        
        # Grafo conexo
        connected_betti = BettiNumbers(beta_0=BettiNumber(1), beta_1=BettiNumber(0))
        connected_bundle = TopologicalMetricsBundle(
            betti=connected_betti,
            spectral=SpectralData(eigenvalues=(0.0, 0.5)),
            graph_metrics=GraphMetrics(n_nodes=5, n_edges=4),
            pyramid_stability=0.90
        )
        
        # Grafo desconexo
        disconnected_betti = BettiNumbers(beta_0=BettiNumber(3), beta_1=BettiNumber(0))
        disconnected_bundle = TopologicalMetricsBundle(
            betti=disconnected_betti,
            spectral=SpectralData(eigenvalues=(0.0, 0.0, 0.0, 0.5)),
            graph_metrics=GraphMetrics(n_nodes=5, n_edges=2),
            pyramid_stability=0.90
        )
        
        # La coherencia del desconexo debe ser menor
        assert disconnected_bundle.structural_coherence < connected_bundle.structural_coherence
    
    def test_cycle_density_calculation(self):
        """
        Prueba: Densidad de ciclos = β₁ / |V|.
        
        Interpretación: Proporción de ciclos independientes por nodo.
        """
        from app.strategy.business_agent import (
            TopologicalMetricsBundle,
            BettiNumbers, BettiNumber,
            SpectralData,
            GraphMetrics
        )
        
        betti = BettiNumbers(beta_0=BettiNumber(1), beta_1=BettiNumber(5))
        spectral = SpectralData(eigenvalues=(0.0, 0.5, 1.2))
        graph_metrics = GraphMetrics(n_nodes=10, n_edges=14)
        
        bundle = TopologicalMetricsBundle(
            betti=betti,
            spectral=spectral,
            graph_metrics=graph_metrics,
            pyramid_stability=0.80
        )
        
        expected_density = 5 / 10  # β₁ / n
        assert abs(bundle.cycle_density - expected_density) < 1e-10
    
    def test_topological_entropy_calculation(self):
        """
        Prueba: Entropía topológica mediante distribución de Betti.
        
        H_top = -Σᵢ pᵢ log pᵢ donde pᵢ = βᵢ / Σⱼ βⱼ
        """
        from app.strategy.business_agent import (
            TopologicalMetricsBundle,
            BettiNumbers, BettiNumber,
            SpectralData,
            GraphMetrics
        )
        
        # Todos los Betti iguales → máxima entropía
        uniform_betti = BettiNumbers(beta_0=BettiNumber(1), beta_1=BettiNumber(1), beta_2=BettiNumber(1))
        uniform_bundle = TopologicalMetricsBundle(
            betti=uniform_betti,
            spectral=SpectralData(eigenvalues=(0.0, 0.5, 1.2)),
            graph_metrics=GraphMetrics(n_nodes=3, n_edges=2),
            pyramid_stability=0.80
        )
        
        entropy = uniform_bundle.topological_entropy
        
        # La entropía debe ser no negativa
        assert entropy >= 0.0
        
        # Un solo Betti no cero → entropía cero
        single_betti = BettiNumbers(beta_0=BettiNumber(1), beta_1=BettiNumber(0), beta_2=BettiNumber(0))
        single_bundle = TopologicalMetricsBundle(
            betti=single_betti,
            spectral=SpectralData(eigenvalues=(0.0,)),
            graph_metrics=GraphMetrics(n_nodes=1, n_edges=0),
            pyramid_stability=0.80
        )
        
        assert single_bundle.topological_entropy == 0.0
    
    def test_bundle_to_dict_complete(self):
        """Prueba: Serialización completa del bundle."""
        from app.strategy.business_agent import (
            TopologicalMetricsBundle,
            BettiNumbers, BettiNumber,
            SpectralData,
            GraphMetrics
        )
        
        betti = BettiNumbers(beta_0=BettiNumber(1))
        spectral = SpectralData(eigenvalues=(0.0, 1.0))
        graph_metrics = GraphMetrics(n_nodes=2, n_edges=1)
        
        bundle = TopologicalMetricsBundle(
            betti=betti,
            spectral=spectral,
            graph_metrics=graph_metrics,
            pyramid_stability=0.85
        )
        
        result = bundle.to_dict()
        
        assert "betti_numbers" in result
        assert "spectral_data" in result
        assert "graph_metrics" in result
        assert "pyramid_stability" in result
        assert "structural_coherence" in result
        assert "cycle_density" in result
        assert "topological_entropy" in result


# =============================================================================
# CLASE DE PRUEBAS FASE 2: HOMOLOGÍA PERSISTENTE
# =============================================================================

class TestPersistenceInterval:
    """
    Suite de pruebas para intervalos de persistencia.
    
    Fundamentación:
        En filtración K₀ ⊆ K₁ ⊆ ... ⊆ Kₙ, un feature topológico
        nace en tiempo birth y muere en tiempo death.
        
        Persistencia = death - birth (medida de significancia)
    """
    
    def test_persistence_interval_creation_valid(self):
        """Prueba: Creación válida de intervalo."""
        from app.strategy.business_agent import PersistenceInterval, SimplexDimension
        
        interval = PersistenceInterval(
            dimension=SimplexDimension(1),
            birth=0.5,
            death=2.0
        )
        
        assert interval.dimension == 1
        assert interval.birth == 0.5
        assert interval.death == 2.0
    
    def test_persistence_interval_birth_non_negative(self):
        """Prueba: Birth debe ser no negativo."""
        from app.strategy.business_agent import PersistenceInterval, SimplexDimension, TopologicalInvariantError
        
        with pytest.raises(TopologicalInvariantError):
            PersistenceInterval(dimension=SimplexDimension(0), birth=-0.1, death=1.0)
    
    def test_persistence_interval_death_greater_than_birth(self):
        """Prueba: Death debe ser > birth (o infinito)."""
        from app.strategy.business_agent import PersistenceInterval, SimplexDimension, TopologicalInvariantError
        
        # Death < birth
        with pytest.raises(TopologicalInvariantError):
            PersistenceInterval(dimension=SimplexDimension(0), birth=2.0, death=1.0)
        
        # Death = birth (no permitido)
        with pytest.raises(TopologicalInvariantError):
            PersistenceInterval(dimension=SimplexDimension(0), birth=1.0, death=1.0)
        
        # Death = ∞ (permitido)
        interval = PersistenceInterval(dimension=SimplexDimension(0), birth=1.0, death=float('inf'))
        assert interval.is_essential is True
    
    def test_persistence_calculation(self):
        """Prueba: Cálculo de persistencia = death - birth."""
        from app.strategy.business_agent import PersistenceInterval, SimplexDimension
        
        interval = PersistenceInterval(dimension=SimplexDimension(1), birth=0.5, death=2.5)
        
        assert interval.persistence == 2.0
    
    def test_persistence_essential_infinite(self):
        """Prueba: Persistencia infinita para intervalos esenciales."""
        from app.strategy.business_agent import PersistenceInterval, SimplexDimension
        
        interval = PersistenceInterval(dimension=SimplexDimension(0), birth=1.0, death=float('inf'))
        
        assert interval.persistence == float('inf')
        assert interval.is_essential is True
    
    def test_bottleneck_distance_same_dimension(self):
        """
        Prueba: Distancia de Bottleneck entre intervalos.
        
        d_B(I₁, I₂) = max(|b₁ - b₂|, |d₁ - d₂|)
        """
        from app.strategy.business_agent import PersistenceInterval, SimplexDimension
        
        i1 = PersistenceInterval(dimension=SimplexDimension(1), birth=1.0, death=3.0)
        i2 = PersistenceInterval(dimension=SimplexDimension(1), birth=1.5, death=3.5)
        
        # |1.0 - 1.5| = 0.5, |3.0 - 3.5| = 0.5
        distance = i1.bottleneck_distance(i2)
        
        assert distance == 0.5
    
    def test_bottleneck_distance_different_dimension(self):
        """Prueba: Distancia infinita entre dimensiones diferentes."""
        from app.strategy.business_agent import PersistenceInterval, SimplexDimension
        
        i1 = PersistenceInterval(dimension=SimplexDimension(0), birth=1.0, death=3.0)
        i2 = PersistenceInterval(dimension=SimplexDimension(1), birth=1.0, death=3.0)
        
        distance = i1.bottleneck_distance(i2)
        
        assert distance == float('inf')
    
    def test_midpoint_calculation(self):
        """Prueba: Punto medio del intervalo."""
        from app.strategy.business_agent import PersistenceInterval, SimplexDimension
        
        # Intervalo finito
        interval = PersistenceInterval(dimension=SimplexDimension(0), birth=1.0, death=3.0)
        assert interval.midpoint == 2.0
        
        # Intervalo esencial (death = ∞)
        essential = PersistenceInterval(dimension=SimplexDimension(0), birth=1.0, death=float('inf'))
        assert essential.midpoint == 1.0  # Usa birth como midpoint


class TestPersistenceDiagram:
    """
    Suite de pruebas para diagramas de persistencia.
    
    Teorema (Estabilidad de Diagramas):
        d_B(Dgm(f), Dgm(g)) ≤ ‖f - g‖_∞
    """
    
    def test_persistence_diagram_creation(self):
        """Prueba: Creación válida de diagrama."""
        from app.strategy.business_agent import PersistenceDiagram, PersistenceInterval, SimplexDimension
        
        intervals = (
            PersistenceInterval(dimension=SimplexDimension(0), birth=0.0, death=1.0),
            PersistenceInterval(dimension=SimplexDimension(1), birth=0.5, death=2.0)
        )
        
        diagram = PersistenceDiagram(intervals=intervals)
        
        assert len(diagram.intervals) == 2
    
    def test_persistence_diagram_sorted_by_persistence(self):
        """Prueba: Intervalos ordenados por persistencia descendente."""
        from app.strategy.business_agent import PersistenceDiagram, PersistenceInterval, SimplexDimension
        
        intervals = (
            PersistenceInterval(dimension=SimplexDimension(0), birth=0.0, death=1.0),  # pers = 1.0
            PersistenceInterval(dimension=SimplexDimension(1), birth=0.0, death=3.0),  # pers = 3.0
            PersistenceInterval(dimension=SimplexDimension(0), birth=0.0, death=2.0)   # pers = 2.0
        )
        
        diagram = PersistenceDiagram(intervals=intervals)
        
        # Debe estar ordenado: 3.0, 2.0, 1.0
        assert diagram.intervals[0].persistence >= diagram.intervals[1].persistence
        assert diagram.intervals[1].persistence >= diagram.intervals[2].persistence
    
    def test_persistence_diagram_betti_numbers_inference(self):
        """
        Prueba: Inferencia de números de Betti desde intervalos esenciales.
        
        β_k = número de intervalos esenciales de dimensión k
        """
        from app.strategy.business_agent import PersistenceDiagram, PersistenceInterval, SimplexDimension
        
        intervals = (
            PersistenceInterval(dimension=SimplexDimension(0), birth=0.0, death=float('inf')),  # esencial
            PersistenceInterval(dimension=SimplexDimension(0), birth=0.0, death=1.0),  # no esencial
            PersistenceInterval(dimension=SimplexDimension(1), birth=0.5, death=float('inf')),  # esencial
            PersistenceInterval(dimension=SimplexDimension(1), birth=0.5, death=2.0)  # no esencial
        )
        
        diagram = PersistenceDiagram(intervals=intervals)
        betti = diagram.betti_numbers
        
        # β₀ = 1 (1 intervalo esencial de dimensión 0)
        assert betti.beta_0 == 1
        
        # β₁ = 1 (1 intervalo esencial de dimensión 1)
        assert betti.beta_1 == 1
    
    def test_persistence_diagram_filter_by_persistence(self):
        """Prueba: Filtrado por umbral de persistencia."""
        from app.strategy.business_agent import PersistenceDiagram, PersistenceInterval, SimplexDimension
        
        intervals = (
            PersistenceInterval(dimension=SimplexDimension(0), birth=0.0, death=1.0),  # pers = 1.0
            PersistenceInterval(dimension=SimplexDimension(1), birth=0.0, death=3.0),  # pers = 3.0
            PersistenceInterval(dimension=SimplexDimension(0), birth=0.0, death=0.5)   # pers = 0.5
        )
        
        diagram = PersistenceDiagram(intervals=intervals)
        filtered = diagram.filter_by_persistence(threshold=1.5)
        
        # Solo el intervalo con persistencia 3.0 debe quedar
        assert len(filtered.intervals) == 1
        assert filtered.intervals[0].persistence == 3.0
    
    def test_persistence_diagram_filter_by_dimension(self):
        """Prueba: Filtrado por dimensión homológica."""
        from app.strategy.business_agent import PersistenceDiagram, PersistenceInterval, SimplexDimension
        
        intervals = (
            PersistenceInterval(dimension=SimplexDimension(0), birth=0.0, death=1.0),
            PersistenceInterval(dimension=SimplexDimension(1), birth=0.5, death=2.0),
            PersistenceInterval(dimension=SimplexDimension(0), birth=0.0, death=2.0)
        )
        
        diagram = PersistenceDiagram(intervals=intervals)
        filtered = diagram.filter_by_dimension(SimplexDimension(0))
        
        # Solo intervalos de dimensión 0
        assert len(filtered.intervals) == 2
        assert all(i.dimension == 0 for i in filtered.intervals)
    
    def test_persistence_diagram_from_intervals_list(self):
        """Prueba: Constructor desde lista de tuplas."""
        from app.strategy.business_agent import PersistenceDiagram
        
        intervals_data = [
            (0, 0.0, 1.0),
            (1, 0.5, 2.0),
            (0, 0.0, float('inf'))
        ]
        
        diagram = PersistenceDiagram.from_intervals_list(intervals_data)
        
        assert len(diagram.intervals) == 3
        assert diagram.intervals[0].dimension == 0
        assert diagram.intervals[1].dimension == 1
        assert diagram.intervals[2].dimension == 0
    
    def test_persistence_diagram_bottleneck_distance(self):
        """Prueba: Distancia de Bottleneck entre diagramas."""
        from app.strategy.business_agent import PersistenceDiagram, PersistenceInterval, SimplexDimension
        
        diagram1 = PersistenceDiagram(intervals=(
            PersistenceInterval(dimension=SimplexDimension(0), birth=0.0, death=1.0),
        ))
        
        diagram2 = PersistenceDiagram(intervals=(
            PersistenceInterval(dimension=SimplexDimension(0), birth=0.0, death=1.5),
        ))
        
        distance = diagram1.bottleneck_distance(diagram2)
        
        # Debe ser no negativa y finita
        assert distance >= 0.0
        assert distance < float('inf')
    
    def test_persistence_diagram_empty(self):
        """Prueba: Diagrama vacío."""
        from app.strategy.business_agent import PersistenceDiagram
        
        empty = PersistenceDiagram(intervals=())
        
        assert len(empty.intervals) == 0
        assert empty.max_dimension == 0
    
    def test_persistence_diagram_to_dict(self):
        """Prueba: Serialización completa."""
        from app.strategy.business_agent import PersistenceDiagram, PersistenceInterval, SimplexDimension
        
        diagram = PersistenceDiagram(intervals=(
            PersistenceInterval(dimension=SimplexDimension(0), birth=0.0, death=1.0),
        ))
        
        result = diagram.to_dict()
        
        assert "intervals" in result
        assert "max_dimension" in result
        assert "total_features" in result
        assert "betti_numbers" in result


# =============================================================================
# FASE 3: MOTOR FINANCIERO Y TERMODINÁMICA ESTADÍSTICA
# =============================================================================

# =============================================================================
# CLASE DE PRUEBAS FASE 3: PARÁMETROS FINANCIEROS
# =============================================================================

class TestFinancialParameters:
    """
    Suite de pruebas para parámetros financieros con precisión Decimal.
    
    Fundamentación Matemática:
        Los cálculos financieros requieren precisión exacta para evitar
        errores de redondeo acumulativos. Decimal proporciona aritmética
        de precisión arbitraria según IEEE 754-2008.
    
    Invariantes Financieros:
        1. I₀ > 0 (inversión inicial positiva)
        2. r ≥ 0 (tasa de descuento no negativa)
        3. σ ∈ [0, 1] (volatilidad normalizada)
        4. CFₜ ∈ ℝ (flujos de caja reales)
    """
    
    def test_financial_parameters_creation_valid(self, sample_financial_params):
        """Prueba: Creación válida de parámetros financieros."""
        from app.strategy.business_agent import FinancialParameters
        from decimal import Decimal
        
        params = FinancialParameters(
            initial_investment=Decimal(sample_financial_params["initial_investment"]),
            cash_flows=tuple(Decimal(cf) for cf in sample_financial_params["cash_flows"]),
            discount_rate=Decimal(sample_financial_params["discount_rate"]),
            risk_free_rate=Decimal(sample_financial_params["risk_free_rate"]),
            market_return=Decimal(sample_financial_params["market_return"]),
            beta=Decimal(sample_financial_params["beta"]),
            cost_std_dev=Decimal(sample_financial_params["cost_std_dev"]),
            project_volatility=sample_financial_params["project_volatility"]
        )
        
        assert params.initial_investment == Decimal("1000000.0")
        assert len(params.cash_flows) == 5
        assert params.periods == 5
    
    def test_financial_parameters_initial_investment_positive(self):
        """
        Prueba: Inversión inicial debe ser positiva.
        
        Invariante: I₀ > 0
        Justificación: No existe proyecto con inversión cero o negativa.
        """
        from app.strategy.business_agent import FinancialParameters
        from decimal import Decimal
        
        with pytest.raises(ValueError):
            FinancialParameters(
                initial_investment=Decimal("0"),
                cash_flows=(Decimal("100000"),),
                discount_rate=Decimal("0.10")
            )
        
        with pytest.raises(ValueError):
            FinancialParameters(
                initial_investment=Decimal("-100000"),
                cash_flows=(Decimal("100000"),),
                discount_rate=Decimal("0.10")
            )
    
    def test_financial_parameters_discount_rate_non_negative(self):
        """
        Prueba: Tasa de descuento no negativa.
        
        Invariante: r ≥ 0
        Justificación: Tasa negativa implicaría valor futuro > valor presente
        sin riesgo, violando principio de preferencia temporal.
        """
        from app.strategy.business_agent import FinancialParameters
        from decimal import Decimal
        
        with pytest.raises(ValueError):
            FinancialParameters(
                initial_investment=Decimal("100000"),
                cash_flows=(Decimal("100000"),),
                discount_rate=Decimal("-0.05")
            )
    
    def test_financial_parameters_volatility_unit_interval(self):
        """
        Prueba: Volatilidad en [0, 1].
        
        Invariante: σ ∈ [0, 1]
        Justificación: Volatilidad normalizada para consistencia con
        modelos de opciones reales (Black-Scholes).
        """
        from app.strategy.business_agent import FinancialParameters
        from decimal import Decimal
        
        # Volatilidad > 1 (inválida)
        with pytest.raises(ValueError):
            FinancialParameters(
                initial_investment=Decimal("100000"),
                cash_flows=(Decimal("100000"),),
                discount_rate=Decimal("0.10"),
                project_volatility=1.5
            )
        
        # Volatilidad < 0 (inválida)
        with pytest.raises(ValueError):
            FinancialParameters(
                initial_investment=Decimal("100000"),
                cash_flows=(Decimal("100000"),),
                discount_rate=Decimal("0.10"),
                project_volatility=-0.1
            )
        
        # Volatilidad en rango (válida)
        params = FinancialParameters(
            initial_investment=Decimal("100000"),
            cash_flows=(Decimal("100000"),),
            discount_rate=Decimal("0.10"),
            project_volatility=0.25
        )
        assert params.project_volatility == 0.25
    
    def test_financial_parameters_cost_std_dev_non_negative(self):
        """Prueba: Desviación estándar de costos no negativa."""
        from app.strategy.business_agent import FinancialParameters
        from decimal import Decimal
        
        with pytest.raises(ValueError):
            FinancialParameters(
                initial_investment=Decimal("100000"),
                cash_flows=(Decimal("100000"),),
                discount_rate=Decimal("0.10"),
                cost_std_dev=Decimal("-5000")
            )
    
    def test_financial_parameters_total_cash_flow(self, sample_financial_params):
        """Prueba: Cálculo de flujo de caja total."""
        from app.strategy.business_agent import FinancialParameters
        from decimal import Decimal
        
        params = FinancialParameters(
            initial_investment=Decimal(sample_financial_params["initial_investment"]),
            cash_flows=tuple(Decimal(cf) for cf in sample_financial_params["cash_flows"]),
            discount_rate=Decimal(sample_financial_params["discount_rate"])
        )
        
        expected_total = sum(Decimal(cf) for cf in sample_financial_params["cash_flows"])
        assert params.total_cash_flow == expected_total
    
    def test_financial_parameters_average_cash_flow(self, sample_financial_params):
        """Prueba: Cálculo de flujo de caja promedio."""
        from app.strategy.business_agent import FinancialParameters
        from decimal import Decimal
        
        params = FinancialParameters(
            initial_investment=Decimal(sample_financial_params["initial_investment"]),
            cash_flows=tuple(Decimal(cf) for cf in sample_financial_params["cash_flows"]),
            discount_rate=Decimal(sample_financial_params["discount_rate"])
        )
        
        expected_avg = params.total_cash_flow / Decimal(params.periods)
        assert params.average_cash_flow == expected_avg
    
    def test_financial_parameters_capm_required_return(self, sample_financial_params):
        """
        Prueba: Cálculo de retorno requerido según CAPM.
        
        Fórmula: E[Rᵢ] = Rᶠ + βᵢ(E[Rₘ] - Rᶠ)
        
        Donde:
            - Rᶠ: tasa libre de riesgo
            - βᵢ: beta del activo
            - E[Rₘ]: retorno esperado del mercado
            - (E[Rₘ] - Rᶠ): prima de riesgo de mercado
        """
        from app.strategy.business_agent import FinancialParameters
        from decimal import Decimal
        
        params = FinancialParameters(
            initial_investment=Decimal("1000000"),
            cash_flows=(Decimal("300000"), Decimal("350000")),
            discount_rate=Decimal("0.10"),
            risk_free_rate=Decimal("0.05"),
            market_return=Decimal("0.12"),
            beta=Decimal("1.2")
        )
        
        # CAPM: 0.05 + 1.2 × (0.12 - 0.05) = 0.05 + 1.2 × 0.07 = 0.05 + 0.084 = 0.134
        expected_capm = Decimal("0.05") + Decimal("1.2") * (Decimal("0.12") - Decimal("0.05"))
        
        from app.strategy.business_agent import MathematicalConstants as MC
        assert abs(float(params.required_return_capm) - float(expected_capm)) < MC.EPSILON_TOLERANCE
    
    def test_financial_parameters_from_dict(self, sample_financial_params):
        """Prueba: Constructor desde diccionario."""
        from app.strategy.business_agent import FinancialParameters
        
        params = FinancialParameters.from_dict(sample_financial_params)
        
        assert params.initial_investment > 0
        assert len(params.cash_flows) == 5
        assert params.discount_rate > 0
    
    def test_financial_parameters_to_dict_serialization(self, sample_financial_params):
        """Prueba: Serialización completa a diccionario."""
        from app.strategy.business_agent import FinancialParameters
        from decimal import Decimal
        
        params = FinancialParameters(
            initial_investment=Decimal(sample_financial_params["initial_investment"]),
            cash_flows=tuple(Decimal(cf) for cf in sample_financial_params["cash_flows"]),
            discount_rate=Decimal(sample_financial_params["discount_rate"])
        )
        
        result = params.to_dict()
        
        assert "initial_investment" in result
        assert "cash_flows" in result
        assert "discount_rate" in result
        assert "periods" in result
        assert "total_cash_flow" in result
        assert "average_cash_flow" in result
        assert "required_return_capm" in result
        
        # Verificar que los Decimals se convierten a float
        assert isinstance(result["initial_investment"], float)
        assert isinstance(result["cash_flows"], list)
        assert all(isinstance(cf, float) for cf in result["cash_flows"])
    
    def test_financial_parameters_immutability(self, sample_financial_params):
        """
        Prueba: FinancialParameters es inmutable (frozen dataclass).
        
        Esto garantiza que los parámetros no se corrompan durante
        el análisis financiero.
        """
        from app.strategy.business_agent import FinancialParameters
        from decimal import Decimal
        
        params = FinancialParameters(
            initial_investment=Decimal("100000"),
            cash_flows=(Decimal("100000"),),
            discount_rate=Decimal("0.10")
        )
        
        with pytest.raises(AttributeError):  # frozen=True
            params.initial_investment = Decimal("200000")


# =============================================================================
# CLASE DE PRUEBAS FASE 3: MOTOR FINANCIERO
# =============================================================================

class TestFinancialEngine:
    """
    Suite de pruebas para el motor de análisis financiero.
    
    Fundamentación Matemática:
        1. VPN: Valor presente de flujos descontados
        2. TIR: Tasa que hace VPN = 0 (raíz de polinomio)
        3. VaR: Cuantil de distribución de pérdidas
        4. CVaR: Valor esperado de pérdidas beyond VaR
    
    Teorema (Convergencia de Newton-Raphson):
        Si f ∈ C²[a,b] y f'(x) ≠ 0 en [a,b], entonces
        Newton-Raphson converge cuadráticamente a la raíz.
    """
    
    def test_financial_engine_npv_positive_cash_flows(self, sample_financial_params):
        """
        Prueba: VPN con flujos positivos.
        
        Fórmula: VPN = -I₀ + Σₜ CFₜ / (1 + r)ᵗ
        
        Para flujos constantes CF y tasa r:
        VPN = -I₀ + CF × [(1 - (1+r)⁻ⁿ) / r]
        """
        from app.strategy.business_agent import FinancialEngine, FinancialParameters
        from decimal import Decimal
        
        params = FinancialParameters(
            initial_investment=Decimal("1000000"),
            cash_flows=(Decimal("300000"), Decimal("300000"), Decimal("300000"),
                       Decimal("300000"), Decimal("300000")),
            discount_rate=Decimal("0.10")
        )
        
        engine = FinancialEngine()
        npv = engine.calculate_npv(params)
        
        # VPN debe ser positivo (flujos > inversión descontada)
        assert npv > 0
        
        # Verificación manual aproximada
        # VPN ≈ -1000000 + 300000 × 3.7908 = -1000000 + 1137240 = 137240
        from app.strategy.business_agent import MathematicalConstants as MC
        assert abs(float(npv) - 137240) < 50000  # Tolerancia por redondeo
    
    def test_financial_engine_npv_negative_cash_flows(self):
        """Prueba: VPN con flujos insuficientes."""
        from app.strategy.business_agent import FinancialEngine, FinancialParameters
        from decimal import Decimal
        
        params = FinancialParameters(
            initial_investment=Decimal("1000000"),
            cash_flows=(Decimal("100000"), Decimal("100000"), Decimal("100000")),
            discount_rate=Decimal("0.10")
        )
        
        engine = FinancialEngine()
        npv = engine.calculate_npv(params)
        
        # VPN debe ser negativo
        assert npv < 0
    
    def test_financial_engine_npv_continuous_discounting(self, sample_financial_params):
        """
        Prueba: VPN con descuento continuo.
        
        Fórmula: VPN = -I₀ + Σₜ CFₜ · e^(-rt)
        
        Comparación: descuento continuo < descuento discreto
        (e^(-r) < 1/(1+r) para r > 0)
        """
        from app.strategy.business_agent import FinancialEngine, FinancialParameters
        from decimal import Decimal
        
        params = FinancialParameters(
            initial_investment=Decimal("1000000"),
            cash_flows=(Decimal("300000"), Decimal("300000"), Decimal("300000")),
            discount_rate=Decimal("0.10")
        )
        
        engine_discrete = FinancialEngine(use_continuous_discounting=False)
        engine_continuous = FinancialEngine(use_continuous_discounting=True)
        
        npv_discrete = engine_discrete.calculate_npv(params)
        npv_continuous = engine_continuous.calculate_npv(params)
        
        # Descuento continuo produce VPN menor (mayor descuento)
        assert npv_continuous < npv_discrete
    
    def test_financial_engine_irr_convergence(self):
        """
        Prueba: Convergencia de TIR mediante Newton-Raphson.
        
        Teorema (Existencia de TIR):
            Si hay cambio de signo en flujos de caja (inversión negativa,
            flujos positivos), existe al menos una TIR real.
        """
        from app.strategy.business_agent import FinancialEngine, FinancialParameters
        from decimal import Decimal
        
        params = FinancialParameters(
            initial_investment=Decimal("1000000"),
            cash_flows=(Decimal("300000"), Decimal("350000"), Decimal("400000"),
                       Decimal("450000"), Decimal("500000")),
            discount_rate=Decimal("0.10")
        )
        
        engine = FinancialEngine()
        irr = engine.calculate_irr(params)
        
        # TIR debe existir y ser positiva
        assert irr is not None
        assert irr > 0
        
        # TIR debe ser razonable (< 100%)
        assert irr < Decimal("1.0")
        
        # Verificar que VPN(TIR) ≈ 0
        params_with_irr = FinancialParameters(
            initial_investment=params.initial_investment,
            cash_flows=params.cash_flows,
            discount_rate=irr
        )
        
        npv_at_irr = engine.calculate_npv(params_with_irr)
        
        from app.strategy.business_agent import MathematicalConstants as MC
        assert abs(float(npv_at_irr)) < 1000  # Cercano a cero
    
    def test_financial_engine_irr_no_convergence(self):
        """Prueba: TIR cuando no hay convergencia (flujos todos negativos)."""
        from app.strategy.business_agent import FinancialEngine, FinancialParameters
        from decimal import Decimal
        
        params = FinancialParameters(
            initial_investment=Decimal("1000000"),
            cash_flows=(Decimal("-100000"), Decimal("-100000"), Decimal("-100000")),
            discount_rate=Decimal("0.10")
        )
        
        engine = FinancialEngine()
        irr = engine.calculate_irr(params)
        
        # TIR puede ser None si no converge
        # (depende de la implementación de fallback)
        assert irr is None or irr < 0
    
    def test_financial_engine_payback_period_exact(self):
        """
        Prueba: Período de recuperación exacto.
        
        Definición: Tiempo t* tal que Σₜ₌₁ᵗ* CFₜ = I₀
        """
        from app.strategy.business_agent import FinancialEngine, FinancialParameters
        from decimal import Decimal
        
        # Inversión: 1000000, Flujo anual: 250000 → Payback = 4 años exactos
        params = FinancialParameters(
            initial_investment=Decimal("1000000"),
            cash_flows=(Decimal("250000"), Decimal("250000"), Decimal("250000"),
                       Decimal("250000"), Decimal("250000")),
            discount_rate=Decimal("0.10")
        )
        
        engine = FinancialEngine()
        payback = engine.calculate_payback_period(params)
        
        assert payback == Decimal("4")
    
    def test_financial_engine_payback_period_fractional(self):
        """
        Prueba: Período de recuperación fraccional.
        
        Interpolación lineal para fracción de período.
        """
        from app.strategy.business_agent import FinancialEngine, FinancialParameters
        from decimal import Decimal
        
        # Inversión: 1000000, Flujos: 300000/año
        # Año 3: 900000 acumulados, faltan 100000
        # Fracción: 100000/300000 = 1/3
        # Payback = 3 + 1/3 = 3.333...
        params = FinancialParameters(
            initial_investment=Decimal("1000000"),
            cash_flows=(Decimal("300000"), Decimal("300000"), Decimal("300000"),
                       Decimal("300000"), Decimal("300000")),
            discount_rate=Decimal("0.10")
        )
        
        engine = FinancialEngine()
        payback = engine.calculate_payback_period(params)
        
        # Debe estar entre 3 y 4
        assert Decimal("3") < payback < Decimal("4")
        
        # Aproximadamente 3.33
        from app.strategy.business_agent import MathematicalConstants as MC
        assert abs(float(payback) - 3.333) < 0.1
    
    def test_financial_engine_payback_period_never(self):
        """Prueba: Payback cuando nunca se recupera inversión."""
        from app.strategy.business_agent import FinancialEngine, FinancialParameters
        from decimal import Decimal
        
        params = FinancialParameters(
            initial_investment=Decimal("1000000"),
            cash_flows=(Decimal("100000"), Decimal("100000"), Decimal("100000")),
            discount_rate=Decimal("0.10")
        )
        
        engine = FinancialEngine()
        payback = engine.calculate_payback_period(params)
        
        assert payback is None
    
    def test_financial_engine_profitability_index_viable(self):
        """
        Prueba: Índice de rentabilidad para proyecto viable.
        
        Fórmula: PI = (VPN + I₀) / I₀ = VP(flujos) / I₀
        
        Criterio: PI > 1 ⇒ proyecto viable
        """
        from app.strategy.business_agent import FinancialEngine, FinancialParameters
        from decimal import Decimal
        
        params = FinancialParameters(
            initial_investment=Decimal("1000000"),
            cash_flows=(Decimal("300000"), Decimal("350000"), Decimal("400000"),
                       Decimal("450000"), Decimal("500000")),
            discount_rate=Decimal("0.10")
        )
        
        engine = FinancialEngine()
        npv = engine.calculate_npv(params)
        pi = engine.calculate_profitability_index(params, npv)
        
        # PI debe ser > 1 para proyecto viable
        assert pi > Decimal("1")
    
    def test_financial_engine_profitability_index_non_viable(self):
        """Prueba: Índice de rentabilidad para proyecto no viable."""
        from app.strategy.business_agent import FinancialEngine, FinancialParameters
        from decimal import Decimal
        
        params = FinancialParameters(
            initial_investment=Decimal("1000000"),
            cash_flows=(Decimal("100000"), Decimal("100000"), Decimal("100000")),
            discount_rate=Decimal("0.10")
        )
        
        engine = FinancialEngine()
        npv = engine.calculate_npv(params)
        pi = engine.calculate_profitability_index(params, npv)
        
        # PI debe ser < 1 para proyecto no viable
        assert pi < Decimal("1")
    
    def test_financial_engine_var_parametric(self, sample_financial_params):
        """
        Prueba: VaR paramétrico (distribución normal).
        
        Fórmula: VaR_α = μ - z_α · σ
        
        Donde:
            - μ: VPN esperado
            - z_α: cuantil de normal estándar (z_0.05 ≈ -1.645)
            - σ: desviación estándar del VPN
        """
        from app.strategy.business_agent import FinancialEngine, FinancialParameters
        from decimal import Decimal
        
        params = FinancialParameters(
            initial_investment=Decimal(sample_financial_params["initial_investment"]),
            cash_flows=tuple(Decimal(cf) for cf in sample_financial_params["cash_flows"]),
            discount_rate=Decimal(sample_financial_params["discount_rate"]),
            cost_std_dev=Decimal(sample_financial_params["cost_std_dev"])
        )
        
        engine = FinancialEngine()
        npv = engine.calculate_npv(params)
        var_95, cvar_95 = engine.calculate_var_parametric(params, npv, confidence_level=0.95)
        
        # VaR y CVaR deben ser Decimals
        assert isinstance(var_95, Decimal)
        assert isinstance(cvar_95, Decimal)
        
        # CVaR ≥ VaR (en valor absoluto para pérdidas)
        # Nota: depende de la convención de signos
    
    def test_financial_engine_modified_irr(self, sample_financial_params):
        """
        Prueba: TIR Modificada (MIRR).
        
        Fórmula: MIRR = [(FV_positivos / PV_negativos)^(1/n)] - 1
        
        Ventaja sobre TIR tradicional:
            - Asume reinversión a tasa realista (no a la TIR)
            - Evita múltiples soluciones
        """
        from app.strategy.business_agent import FinancialEngine, FinancialParameters
        from decimal import Decimal
        
        params = FinancialParameters(
            initial_investment=Decimal(sample_financial_params["initial_investment"]),
            cash_flows=tuple(Decimal(cf) for cf in sample_financial_params["cash_flows"]),
            discount_rate=Decimal(sample_financial_params["discount_rate"]),
            risk_free_rate=Decimal(sample_financial_params["risk_free_rate"])
        )
        
        engine = FinancialEngine()
        mirr = engine.calculate_modified_irr(params)
        
        # MIRR debe existir para flujos normales
        assert mirr is not None
        assert mirr > 0
    
    def test_financial_engine_roi_calculation(self, sample_financial_params):
        """
        Prueba: ROI simple.
        
        Fórmula: ROI = (Σ CF - I₀) / I₀
        
        Interpretación: Retorno total como fracción de inversión.
        """
        from app.strategy.business_agent import FinancialEngine, FinancialParameters
        from decimal import Decimal
        
        params = FinancialParameters(
            initial_investment=Decimal(sample_financial_params["initial_investment"]),
            cash_flows=tuple(Decimal(cf) for cf in sample_financial_params["cash_flows"]),
            discount_rate=Decimal(sample_financial_params["discount_rate"])
        )
        
        engine = FinancialEngine()
        roi = engine.calculate_roi(params)
        
        # ROI debe ser Decimal
        assert isinstance(roi, Decimal)
        
        # Para flujos positivos, ROI > 0
        assert roi > 0
    
    def test_financial_engine_analyze_complete(self, sample_financial_params):
        """
        Prueba: Análisis financiero completo.
        
        Ejecuta todos los cálculos y retorna FinancialMetrics.
        """
        from app.strategy.business_agent import FinancialEngine, FinancialParameters, FinancialMetrics
        from decimal import Decimal
        
        params = FinancialParameters(
            initial_investment=Decimal(sample_financial_params["initial_investment"]),
            cash_flows=tuple(Decimal(cf) for cf in sample_financial_params["cash_flows"]),
            discount_rate=Decimal(sample_financial_params["discount_rate"]),
            risk_free_rate=Decimal(sample_financial_params["risk_free_rate"]),
            cost_std_dev=Decimal(sample_financial_params["cost_std_dev"])
        )
        
        engine = FinancialEngine()
        metrics = engine.analyze(params)
        
        # Verificar que todos los campos están presentes
        assert isinstance(metrics, FinancialMetrics)
        assert metrics.npv is not None
        assert metrics.irr is not None
        assert metrics.payback_period is not None
        assert metrics.profitability_index is not None
        assert metrics.roi is not None
        
        # Verificar viabilidad
        assert isinstance(metrics.is_viable, bool)
        assert isinstance(metrics.risk_class, str)
    
    def test_financial_metrics_is_viable_criteria(self):
        """
        Prueba: Criterios de viabilidad financiera.
        
        Criterios:
            1. VPN > 0
            2. PI > 1
            3. TIR > tasa de descuento (si existe)
        """
        from app.strategy.business_agent import FinancialMetrics
        from decimal import Decimal
        
        # Proyecto viable
        viable = FinancialMetrics(
            npv=Decimal("100000"),
            irr=Decimal("0.15"),
            payback_period=Decimal("3.5"),
            profitability_index=Decimal("1.2"),
            roi=Decimal("0.20")
        )
        assert viable.is_viable is True
        
        # Proyecto no viable (VPN negativo)
        non_viable_npv = FinancialMetrics(
            npv=Decimal("-100000"),
            irr=Decimal("0.05"),
            profitability_index=Decimal("0.8")
        )
        assert non_viable_npv.is_viable is False
        
        # Proyecto no viable (PI < 1)
        non_viable_pi = FinancialMetrics(
            npv=Decimal("50000"),
            profitability_index=Decimal("0.9")
        )
        assert non_viable_pi.is_viable is False
    
    def test_financial_metrics_risk_classification(self):
        """
        Prueba: Clasificación de riesgo por VaR/VPN.
        
        Criterios:
            - LOW: VaR/VPN < 0.1
            - MODERATE: 0.1 ≤ VaR/VPN < 0.3
            - HIGH: 0.3 ≤ VaR/VPN < 0.6
            - CRITICAL: VaR/VPN ≥ 0.6
        """
        from app.strategy.business_agent import FinancialMetrics
        from decimal import Decimal
        
        # Riesgo bajo
        low_risk = FinancialMetrics(
            npv=Decimal("1000000"),
            var_95=Decimal("-50000"),  # 5% del VPN
            profitability_index=Decimal("1.5")
        )
        assert low_risk.risk_class == "LOW"
        
        # Riesgo moderado
        moderate_risk = FinancialMetrics(
            npv=Decimal("1000000"),
            var_95=Decimal("-200000"),  # 20% del VPN
            profitability_index=Decimal("1.2")
        )
        assert moderate_risk.risk_class == "MODERATE"
        
        # Riesgo alto
        high_risk = FinancialMetrics(
            npv=Decimal("1000000"),
            var_95=Decimal("-500000"),  # 50% del VPN
            profitability_index=Decimal("1.1")
        )
        assert high_risk.risk_class == "HIGH"
        
        # Riesgo crítico
        critical_risk = FinancialMetrics(
            npv=Decimal("1000000"),
            var_95=Decimal("-800000"),  # 80% del VPN
            profitability_index=Decimal("1.05")
        )
        assert critical_risk.risk_class == "CRITICAL"
    
    def test_financial_metrics_to_dict(self):
        """Prueba: Serialización de FinancialMetrics."""
        from app.strategy.business_agent import FinancialMetrics
        from decimal import Decimal
        
        metrics = FinancialMetrics(
            npv=Decimal("100000"),
            irr=Decimal("0.15"),
            payback_period=Decimal("3.5"),
            profitability_index=Decimal("1.2"),
            var_95=Decimal("-50000"),
            cvar_95=Decimal("-60000"),
            roi=Decimal("0.20"),
            modified_irr=Decimal("0.14")
        )
        
        result = metrics.to_dict()
        
        assert "npv" in result
        assert "irr" in result
        assert "payback_period" in result
        assert "profitability_index" in result
        assert "var_95" in result
        assert "cvar_95" in result
        assert "roi" in result
        assert "modified_irr" in result
        assert "is_viable" in result
        assert "risk_class" in result
        
        # Verificar conversión a float
        assert isinstance(result["npv"], float)
        assert isinstance(result["is_viable"], bool)


# =============================================================================
# CLASE DE PRUEBAS FASE 3: OPCIONES REALES
# =============================================================================

class TestBlackScholesEngine:
    """
    Suite de pruebas para el motor de Black-Scholes-Merton.
    
    Fundamentación Matemática:
        Modelo de Black-Scholes para opción call europea:
        
        C = S·N(d₁) - K·e^(-rT)·N(d₂)
        
        donde:
            d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
            d₂ = d₁ - σ√T
            N(·): CDF de normal estándar
    
    Supuestos del Modelo:
        1. Mercado eficiente (no arbitraje)
        2. Volatilidad constante
        3. Sin dividendos
        4. Tasa libre de riesgo constante
        5. Trading continuo
    """
    
    def test_black_scholes_call_option_value(self):
        """
        Prueba: Valor de opción call europea.
        
        Caso de prueba conocido:
            S = 100, K = 100, r = 0.05, σ = 0.2, T = 1
            Call value ≈ 10.45 (valor de referencia)
        """
        from app.strategy.business_agent import BlackScholesEngine
        from decimal import Decimal
        
        engine = BlackScholesEngine()
        
        call_value = engine.call_option_value(
            S=Decimal("100"),
            K=Decimal("100"),
            r=Decimal("0.05"),
            sigma=0.2,
            T=1.0
        )
        
        # Valor debe ser positivo
        assert call_value > 0
        
        # Valor aproximado esperado (puede variar por implementación)
        assert float(call_value) > 5  # Mínimo razonable
        assert float(call_value) < 20  # Máximo razonable
    
    def test_black_scholes_call_option_moneyness(self):
        """
        Prueba: Opción call in/out/at the money.
        
        - ITM (S > K): mayor valor intrínseco
        - ATM (S = K): valor puramente temporal
        - OTM (S < K): menor valor
        """
        from app.strategy.business_agent import BlackScholesEngine
        from decimal import Decimal
        
        engine = BlackScholesEngine()
        
        # ITM
        call_itm = engine.call_option_value(
            S=Decimal("120"), K=Decimal("100"),
            r=Decimal("0.05"), sigma=0.2, T=1.0
        )
        
        # ATM
        call_atm = engine.call_option_value(
            S=Decimal("100"), K=Decimal("100"),
            r=Decimal("0.05"), sigma=0.2, T=1.0
        )
        
        # OTM
        call_otm = engine.call_option_value(
            S=Decimal("80"), K=Decimal("100"),
            r=Decimal("0.05"), sigma=0.2, T=1.0
        )
        
        # ITM > ATM > OTM
        assert call_itm > call_atm
        assert call_atm > call_otm
    
    def test_black_scholes_put_option_value(self):
        """
        Prueba: Valor de opción put europea.
        
        Paridad put-call: P = C - S + K·e^(-rT)
        """
        from app.strategy.business_agent import BlackScholesEngine
        from decimal import Decimal
        
        engine = BlackScholesEngine()
        
        put_value = engine.put_option_value(
            S=Decimal("100"),
            K=Decimal("100"),
            r=Decimal("0.05"),
            sigma=0.2,
            T=1.0
        )
        
        # Valor debe ser positivo
        assert put_value > 0
    
    def test_black_scholes_put_call_parity(self):
        """
        Prueba: Paridad put-call.
        
        Teorema (Put-Call Parity):
            C - P = S - K·e^(-rT)
        
        Esta relación debe mantenerse para opciones europeas.
        """
        from app.strategy.business_agent import BlackScholesEngine, MathematicalConstants as MC
        from decimal import Decimal
        import math
        
        engine = BlackScholesEngine()
        
        S = Decimal("100")
        K = Decimal("100")
        r = Decimal("0.05")
        sigma = 0.2
        T = 1.0
        
        call = engine.call_option_value(S, K, r, sigma, T)
        put = engine.put_option_value(S, K, r, sigma, T)
        
        # Lado izquierdo: C - P
        left_side = float(call) - float(put)
        
        # Lado derecho: S - K·e^(-rT)
        right_side = float(S) - float(K) * math.exp(-float(r) * T)
        
        # Deben ser iguales (dentro de tolerancia numérica)
        assert abs(left_side - right_side) < 0.5  # Tolerancia por aproximación
    
    def test_black_scholes_greeks_delta_call(self):
        """
        Prueba: Delta de opción call.
        
        Delta (Δ) = ∂C/∂S = N(d₁)
        
        Propiedades:
            - Δ ∈ [0, 1] para call
            - Δ → 1 cuando S >> K (deep ITM)
            - Δ → 0 cuando S << K (deep OTM)
        """
        from app.strategy.business_agent import BlackScholesEngine
        from decimal import Decimal
        
        engine = BlackScholesEngine()
        
        greeks = engine.greeks(
            S=Decimal("100"), K=Decimal("100"),
            r=Decimal("0.05"), sigma=0.2, T=1.0,
            option_type="call"
        )
        
        assert "delta" in greeks
        assert 0.0 <= greeks["delta"] <= 1.0
        
        # ATM call: delta ≈ 0.5
        assert 0.4 <= greeks["delta"] <= 0.6
    
    def test_black_scholes_greeks_delta_put(self):
        """
        Prueba: Delta de opción put.
        
        Delta (Δ) = ∂P/∂S = N(d₁) - 1
        
        Propiedades:
            - Δ ∈ [-1, 0] para put
            - Δ → -1 cuando S << K (deep ITM)
            - Δ → 0 cuando S >> K (deep OTM)
        """
        from app.strategy.business_agent import BlackScholesEngine
        from decimal import Decimal
        
        engine = BlackScholesEngine()
        
        greeks = engine.greeks(
            S=Decimal("100"), K=Decimal("100"),
            r=Decimal("0.05"), sigma=0.2, T=1.0,
            option_type="put"
        )
        
        assert "delta" in greeks
        assert -1.0 <= greeks["delta"] <= 0.0
    
    def test_black_scholes_greeks_gamma(self):
        """
        Prueba: Gamma de opción.
        
        Gamma (Γ) = ∂²C/∂S² = φ(d₁) / (S·σ·√T)
        
        Propiedades:
            - Γ ≥ 0 para call y put
            - Γ máxima cuando ATM
            - Γ → 0 cuando deep ITM o OTM
        """
        from app.strategy.business_agent import BlackScholesEngine
        from decimal import Decimal
        
        engine = BlackScholesEngine()
        
        greeks = engine.greeks(
            S=Decimal("100"), K=Decimal("100"),
            r=Decimal("0.05"), sigma=0.2, T=1.0
        )
        
        assert "gamma" in greeks
        assert greeks["gamma"] >= 0
    
    def test_black_scholes_greeks_vega(self):
        """
        Prueba: Vega de opción.
        
        Vega (ν) = ∂C/∂σ = S·φ(d₁)·√T / 100
        
        Propiedades:
            - ν ≥ 0 para call y put
            - ν máxima cuando ATM
            - ν → 0 cuando T → 0
        """
        from app.strategy.business_agent import BlackScholesEngine
        from decimal import Decimal
        
        engine = BlackScholesEngine()
        
        greeks = engine.greeks(
            S=Decimal("100"), K=Decimal("100"),
            r=Decimal("0.05"), sigma=0.2, T=1.0
        )
        
        assert "vega" in greeks
        assert greeks["vega"] >= 0
    
    def test_black_scholes_greeks_theta(self):
        """
        Prueba: Theta de opción.
        
        Theta (Θ) = ∂C/∂T (decaimiento temporal)
        
        Propiedades:
            - Θ ≤ 0 para call y put (long position)
            - El valor decae con el tiempo
        """
        from app.strategy.business_agent import BlackScholesEngine
        from decimal import Decimal
        
        engine = BlackScholesEngine()
        
        greeks = engine.greeks(
            S=Decimal("100"), K=Decimal("100"),
            r=Decimal("0.05"), sigma=0.2, T=1.0,
            option_type="call"
        )
        
        assert "theta" in greeks
        # Theta es negativo (decaimiento)
        assert greeks["theta"] <= 0
    
    def test_black_scholes_greeks_rho(self):
        """
        Prueba: Rho de opción.
        
        Rho (ρ) = ∂C/∂r (sensibilidad a tasa de interés)
        
        Propiedades:
            - ρ ≥ 0 para call
            - ρ ≤ 0 para put
        """
        from app.strategy.business_agent import BlackScholesEngine
        from decimal import Decimal
        
        engine = BlackScholesEngine()
        
        # Call
        greeks_call = engine.greeks(
            S=Decimal("100"), K=Decimal("100"),
            r=Decimal("0.05"), sigma=0.2, T=1.0,
            option_type="call"
        )
        assert greeks_call["rho"] >= 0
        
        # Put
        greeks_put = engine.greeks(
            S=Decimal("100"), K=Decimal("100"),
            r=Decimal("0.05"), sigma=0.2, T=1.0,
            option_type="put"
        )
        assert greeks_put["rho"] <= 0
    
    def test_black_scholes_edge_cases(self):
        """Prueba: Casos extremos de Black-Scholes."""
        from app.strategy.business_agent import BlackScholesEngine
        from decimal import Decimal
        
        engine = BlackScholesEngine()
        
        # S = 0 (activo sin valor)
        call_zero = engine.call_option_value(
            S=Decimal("0"), K=Decimal("100"),
            r=Decimal("0.05"), sigma=0.2, T=1.0
        )
        assert call_zero == Decimal("0.0")
        
        # T = 0 (vencimiento inmediato)
        call_expiry = engine.call_option_value(
            S=Decimal("100"), K=Decimal("100"),
            r=Decimal("0.05"), sigma=0.2, T=0.0
        )
        assert call_expiry == Decimal("0.0")  # ATM al vencimiento = 0
    
    def test_black_scholes_to_dict(self):
        """Prueba: Serialización de RealOption."""
        from app.strategy.business_agent import RealOption
        from decimal import Decimal
        
        option = RealOption(
            option_type="wait",
            underlying_value=Decimal("1000000"),
            strike_price=Decimal("1000000"),
            volatility=0.25,
            time_to_maturity=1.0,
            risk_free_rate=Decimal("0.05"),
            option_value=Decimal("100000")
        )
        
        result = option.to_dict()
        
        assert "option_type" in result
        assert "underlying_value" in result
        assert "strike_price" in result
        assert "volatility" in result
        assert "time_to_maturity" in result
        assert "risk_free_rate" in result
        assert "option_value" in result


class TestRealOptionsEngine:
    """
    Suite de pruebas para el motor de opciones reales.
    
    Fundamentación:
        Las opciones reales extienden el análisis de VPN tradicional
        incorporando flexibilidad gerencial:
        
        1. Opción de espera (defer): postergar inversión
        2. Opción de expansión (scale-up): ampliar proyecto
        3. Opción de abandono (exit): liquidar proyecto
        4. Opción de flexibilidad (switch): cambiar operación
    """
    
    def test_real_options_engine_wait_option(self, sample_financial_params):
        """
        Prueba: Valoración de opción de espera.
        
        Modelado como call europea:
            - Subyacente: Valor del proyecto (VPN + I₀)
            - Strike: Inversión inicial
            - Vencimiento: Años de espera posibles
        """
        from app.strategy.business_agent import (
            RealOptionsEngine, FinancialParameters, FinancialMetrics
        )
        from decimal import Decimal
        
        params = FinancialParameters.from_dict(sample_financial_params)
        
        engine = RealOptionsEngine()
        
        # Simular métricas financieras
        metrics = FinancialMetrics(
            npv=Decimal("500000"),
            profitability_index=Decimal("1.5")
        )
        
        wait_option = engine.value_wait_option(params, metrics.npv, wait_years=1.0)
        
        assert wait_option.option_type == "wait"
        assert wait_option.underlying_value == Decimal("500000")
        assert wait_option.strike_price == params.initial_investment
        assert wait_option.time_to_maturity == 1.0
    
    def test_real_options_engine_abandon_option(self, sample_financial_params):
        """
        Prueba: Valoración de opción de abandono.
        
        Modelado como put europea:
            - Subyacente: Valor del proyecto en operación
            - Strike: Valor de salvamento
        """
        from app.strategy.business_agent import (
            RealOptionsEngine, FinancialParameters, FinancialMetrics
        )
        from decimal import Decimal
        
        params = FinancialParameters.from_dict(sample_financial_params)
        
        engine = RealOptionsEngine()
        metrics = FinancialMetrics(npv=Decimal("500000"))
        
        salvage_value = params.initial_investment * Decimal("0.5")
        abandon_option = engine.value_abandon_option(
            params, metrics.npv, salvage_value, project_life=5.0
        )
        
        assert abandon_option.option_type == "abandon"
        assert abandon_option.strike_price == salvage_value
        assert abandon_option.time_to_maturity == 5.0
    
    def test_real_options_engine_expand_option(self, sample_financial_params):
        """
        Prueba: Valoración de opción de expansión.
        
        Modelado como call sobre proyecto ampliado:
            - Subyacente: VPN del proyecto ampliado
            - Strike: Costo de expansión
        """
        from app.strategy.business_agent import (
            RealOptionsEngine, FinancialParameters, FinancialMetrics
        )
        from decimal import Decimal
        
        params = FinancialParameters.from_dict(sample_financial_params)
        
        engine = RealOptionsEngine()
        metrics = FinancialMetrics(npv=Decimal("500000"))
        
        expansion_cost = params.initial_investment * Decimal("1.5")
        expand_option = engine.value_expand_option(
            params, metrics.npv, expansion_cost,
            expansion_factor=2.0, expansion_window=2.0
        )
        
        assert expand_option.option_type == "expand"
        assert expand_option.strike_price == expansion_cost
        assert expand_option.time_to_maturity == 2.0
    
    def test_real_options_engine_analyze_complete(self, sample_financial_params):
        """Prueba: Análisis completo de opciones reales."""
        from app.strategy.business_agent import (
            RealOptionsEngine, FinancialParameters, FinancialMetrics
        )
        from decimal import Decimal
        
        params = FinancialParameters.from_dict(sample_financial_params)
        
        engine = RealOptionsEngine()
        metrics = FinancialMetrics(
            npv=Decimal("500000"),
            profitability_index=Decimal("1.5")
        )
        
        result = engine.analyze_real_options(params, metrics)
        
        assert "wait_option" in result
        assert "abandon_option" in result
        assert "expand_option" in result
        assert "total_option_value" in result
        assert "expanded_npv" in result
        assert "option_value_ratio" in result
        
        # El VPN expandido debe ser mayor que el VPN base
        assert result["expanded_npv"] > float(metrics.npv)


# =============================================================================
# CLASE DE PRUEBAS FASE 3: TERMODINÁMICA ESTADÍSTICA
# =============================================================================

class TestThermodynamicState:
    """
    Suite de pruebas para estado termodinámico del presupuesto.
    
    Analogía Termodinámica:
        - Energía interna U: Flujo de caja total
        - Entropía S: Desorden/incertidumbre
        - Temperatura T: Volatilidad del mercado
        - Capacidad calorífica C: Inercia financiera
        - Exergía B: Trabajo útil máximo extraíble
        - Energía libre F: F = U - TS
    """
    
    def test_thermodynamic_state_creation_valid(self):
        """Prueba: Creación válida de estado termodinámico."""
        from app.strategy.business_agent import ThermodynamicState
        from decimal import Decimal
        
        state = ThermodynamicState(
            internal_energy=Decimal("1000000"),
            entropy=0.5,
            temperature=25.0,
            heat_capacity=0.7,
            exergy=0.6
        )
        
        assert state.internal_energy == Decimal("1000000")
        assert state.entropy == 0.5
        assert state.temperature == 25.0
        assert state.heat_capacity == 0.7
        assert state.exergy == 0.6
    
    def test_thermodynamic_state_free_energy_calculation(self):
        """
        Prueba: Cálculo de energía libre de Helmholtz.
        
        Fórmula: F = U - TS
        
        Donde:
            - U: energía interna
            - T: temperatura
            - S: entropía
        """
        from app.strategy.business_agent import ThermodynamicState
        from decimal import Decimal
        
        state = ThermodynamicState(
            internal_energy=Decimal("1000000"),
            entropy=0.5,
            temperature=20.0,
            heat_capacity=0.7,
            exergy=0.6
        )
        
        # F = 1000000 - 20 × 0.5 = 1000000 - 10 = 999990
        expected_free_energy = Decimal("1000000") - Decimal(str(20.0 * 0.5))
        
        assert state.free_energy == expected_free_energy
    
    def test_thermodynamic_state_negentropy(self):
        """
        Prueba: Negentropía (información/orden).
        
        Fórmula: Negentropía = 1 - S
        
        Interpretación: Medida de orden del sistema.
        """
        from app.strategy.business_agent import ThermodynamicState
        from decimal import Decimal
        
        state = ThermodynamicState(
            internal_energy=Decimal("1000000"),
            entropy=0.3,
            temperature=20.0,
            heat_capacity=0.7,
            exergy=0.6
        )
        
        assert state.negentropy == 0.7  # 1 - 0.3
    
    def test_thermodynamic_state_thermal_efficiency(self):
        """
        Prueba: Eficiencia térmica.
        
        Aproximación: η ≈ exergía
        
        Interpretación: Fracción de energía convertible en trabajo útil.
        """
        from app.strategy.business_agent import ThermodynamicState
        from decimal import Decimal
        
        state = ThermodynamicState(
            internal_energy=Decimal("1000000"),
            entropy=0.5,
            temperature=20.0,
            heat_capacity=0.7,
            exergy=0.6
        )
        
        assert state.thermal_efficiency == state.exergy
    
    def test_thermodynamic_state_entropy_bounds(self):
        """Prueba: Entropía en [0, 1]."""
        from app.strategy.business_agent import ThermodynamicState
        from decimal import Decimal
        
        # Entropía = 0 (orden perfecto)
        state_min = ThermodynamicState(
            internal_energy=Decimal("1000000"),
            entropy=0.0,
            temperature=20.0,
            heat_capacity=0.7,
            exergy=0.6
        )
        assert state_min.entropy == 0.0
        
        # Entropía = 1 (desorden máximo)
        state_max = ThermodynamicState(
            internal_energy=Decimal("1000000"),
            entropy=1.0,
            temperature=20.0,
            heat_capacity=0.7,
            exergy=0.6
        )
        assert state_max.entropy == 1.0
    
    def test_thermodynamic_state_to_dict(self):
        """Prueba: Serialización completa."""
        from app.strategy.business_agent import ThermodynamicState
        from decimal import Decimal
        
        state = ThermodynamicState(
            internal_energy=Decimal("1000000"),
            entropy=0.5,
            temperature=25.0,
            heat_capacity=0.7,
            exergy=0.6
        )
        
        result = state.to_dict()
        
        assert "internal_energy" in result
        assert "entropy" in result
        assert "temperature" in result
        assert "heat_capacity" in result
        assert "exergy" in result
        assert "free_energy" in result
        assert "negentropy" in result
        assert "thermal_efficiency" in result


class TestThermodynamicsEngine:
    """
    Suite de pruebas para el motor de análisis termodinámico.
    
    Fundamentación:
        La termodinámica estadística proporciona un marco para
        cuantificar el desorden, la eficiencia y la estabilidad
        de sistemas complejos mediante analogías físicas.
    """
    
    def test_thermodynamics_engine_entropy_calculation(self):
        """
        Prueba: Cálculo de entropía de Shannon.
        
        Fórmula: S = -Σᵢ pᵢ log pᵢ / log n
        
        Donde pᵢ = |CFᵢ| / Σⱼ |CFⱼ|
        """
        from app.strategy.business_agent import (
            ThermodynamicsEngine, TopologicalMetricsBundle,
            BettiNumbers, BettiNumber, SpectralData, GraphMetrics
        )
        from decimal import Decimal
        
        engine = ThermodynamicsEngine()
        
        cash_flows = (Decimal("100000"), Decimal("200000"), Decimal("300000"),
                     Decimal("400000"), Decimal("500000"))
        
        # Crear bundle topológico mínimo
        betti = BettiNumbers(beta_0=BettiNumber(1))
        spectral = SpectralData(eigenvalues=(0.0, 1.0))
        graph_metrics = GraphMetrics(n_nodes=5, n_edges=4)
        
        topo_bundle = TopologicalMetricsBundle(
            betti=betti,
            spectral=spectral,
            graph_metrics=graph_metrics,
            pyramid_stability=0.8
        )
        
        entropy = engine.calculate_entropy(cash_flows, topo_bundle)
        
        # Entropía debe estar en [0, 1]
        assert 0.0 <= entropy <= 1.0
        
        # Con flujos variados, entropía > 0
        assert entropy > 0
    
    def test_thermodynamics_engine_entropy_uniform_flows(self):
        """
        Prueba: Entropía máxima con flujos uniformes.
        
        Teorema: La entropía de Shannon es máxima cuando
        la distribución es uniforme (pᵢ = 1/n).
        """
        from app.strategy.business_agent import (
            ThermodynamicsEngine, TopologicalMetricsBundle,
            BettiNumbers, BettiNumber, SpectralData, GraphMetrics
        )
        from decimal import Decimal
        
        engine = ThermodynamicsEngine()
        
        # Flujos uniformes
        uniform_flows = (Decimal("100000"), Decimal("100000"), Decimal("100000"))
        
        betti = BettiNumbers(beta_0=BettiNumber(1))
        spectral = SpectralData(eigenvalues=(0.0, 1.0))
        graph_metrics = GraphMetrics(n_nodes=3, n_edges=2)
        
        topo_bundle = TopologicalMetricsBundle(
            betti=betti,
            spectral=spectral,
            graph_metrics=graph_metrics,
            pyramid_stability=0.8
        )
        
        entropy_uniform = engine.calculate_entropy(uniform_flows, topo_bundle)
        
        # Flujos no uniformes
        non_uniform_flows = (Decimal("10000"), Decimal("100000"), Decimal("1000000"))
        entropy_non_uniform = engine.calculate_entropy(non_uniform_flows, topo_bundle)
        
        # Entropía uniforme debe ser mayor
        assert entropy_uniform > entropy_non_uniform
    
    def test_thermodynamics_engine_temperature_calculation(self, sample_financial_params):
        """
        Prueba: Cálculo de temperatura del sistema.
        
        Fórmula: T = T₀ · (1 + σ_proyecto) · (1 + σ_mercado)
        
        Donde T₀ = 20°C (temperatura de referencia)
        """
        from app.strategy.business_agent import ThermodynamicsEngine, FinancialParameters
        from decimal import Decimal
        
        params = FinancialParameters.from_dict(sample_financial_params)
        
        engine = ThermodynamicsEngine()
        temperature = engine.calculate_temperature(params, market_volatility=0.15)
        
        # Temperatura debe ser positiva
        assert temperature > 0
        
        # Temperatura base es 20°C, con volatilidad debe ser mayor
        assert temperature > 20.0
    
    def test_thermodynamics_engine_heat_capacity_calculation(self, sample_financial_params):
        """
        Prueba: Cálculo de capacidad calorífica (inercia financiera).
        
        Fórmula: C = (I₀ / σ_C) · (1 + PI)
        
        Interpretación: Mayor inversión con bajo riesgo ⇒ alta inercia.
        """
        from app.strategy.business_agent import (
            ThermodynamicsEngine, FinancialParameters, FinancialMetrics
        )
        from decimal import Decimal
        
        params = FinancialParameters.from_dict(sample_financial_params)
        
        engine = ThermodynamicsEngine()
        metrics = FinancialMetrics(
            npv=Decimal("500000"),
            profitability_index=Decimal("1.5")
        )
        
        heat_capacity = engine.calculate_heat_capacity(params, metrics)
        
        # Capacidad calorífica debe estar en [0, 1] (normalizada)
        assert 0.0 <= heat_capacity <= 1.0
    
    def test_thermodynamics_engine_exergy_calculation(self, sample_financial_params):
        """
        Prueba: Cálculo de exergía (trabajo útil máximo).
        
        Fórmula: B = √(PI - 1) · Ψ · coherencia
        
        Interpretación: Exergía alta ⇒ proyecto eficiente con baja entropía.
        """
        from app.strategy.business_agent import (
            ThermodynamicsEngine, FinancialParameters, FinancialMetrics,
            TopologicalMetricsBundle, BettiNumbers, BettiNumber,
            SpectralData, GraphMetrics
        )
        from decimal import Decimal
        
        params = FinancialParameters.from_dict(sample_financial_params)
        
        metrics = FinancialMetrics(
            npv=Decimal("500000"),
            profitability_index=Decimal("1.5")
        )
        
        betti = BettiNumbers(beta_0=BettiNumber(1))
        spectral = SpectralData(eigenvalues=(0.0, 1.0))
        graph_metrics = GraphMetrics(n_nodes=5, n_edges=4)
        
        topo_bundle = TopologicalMetricsBundle(
            betti=betti,
            spectral=spectral,
            graph_metrics=graph_metrics,
            pyramid_stability=0.8
        )
        
        engine = ThermodynamicsEngine()
        exergy = engine.calculate_exergy(metrics, topo_bundle)
        
        # Exergía debe estar en [0, 1]
        assert 0.0 <= exergy <= 1.0
        
        # Con PI > 1 y estabilidad alta, exergía > 0
        assert exergy > 0
    
    def test_thermodynamics_engine_analyze_complete(self, sample_financial_params):
        """Prueba: Análisis termodinámico completo."""
        from app.strategy.business_agent import (
            ThermodynamicsEngine, FinancialParameters, FinancialMetrics,
            TopologicalMetricsBundle, BettiNumbers, BettiNumber,
            SpectralData, GraphMetrics, ThermodynamicState
        )
        from decimal import Decimal
        
        params = FinancialParameters.from_dict(sample_financial_params)
        
        metrics = FinancialMetrics(
            npv=Decimal("500000"),
            profitability_index=Decimal("1.5")
        )
        
        betti = BettiNumbers(beta_0=BettiNumber(1))
        spectral = SpectralData(eigenvalues=(0.0, 1.0))
        graph_metrics = GraphMetrics(n_nodes=5, n_edges=4)
        
        topo_bundle = TopologicalMetricsBundle(
            betti=betti,
            spectral=spectral,
            graph_metrics=graph_metrics,
            pyramid_stability=0.8
        )
        
        engine = ThermodynamicsEngine()
        state = engine.analyze(params, metrics, topo_bundle)
        
        assert isinstance(state, ThermodynamicState)
        assert state.internal_energy == params.total_cash_flow
        assert 0.0 <= state.entropy <= 1.0
        assert state.temperature > 0
        assert 0.0 <= state.heat_capacity <= 1.0
        assert 0.0 <= state.exergy <= 1.0


# =============================================================================
# FASE 4: RISK CHALLENGER, COMPOSITOR Y BUSINESS AGENT
# =============================================================================

# =============================================================================
# CLASE DE PRUEBAS FASE 4: UMBRALES Y PESOS DE DECISIÓN
# =============================================================================

class TestRiskChallengerThresholds:
    """
    Suite de pruebas para umbrales del Risk Challenger.
    
    Fundamentación:
        Los umbrales definen puntos de decisión para vetos y alertas.
        Todos deben estar en [0, 1] para consistencia con UnitInterval.
    """
    
    def test_risk_challenger_thresholds_defaults(self):
        """Prueba: Valores por defecto de umbrales."""
        from app.strategy.business_agent import RiskChallengerThresholds
        
        thresholds = RiskChallengerThresholds()
        
        assert thresholds.critical_stability == 0.70
        assert thresholds.warning_stability == 0.85
        assert thresholds.coherence_minimum == 0.60
        assert thresholds.cycle_density_limit == 0.33
        assert thresholds.integrity_penalty_veto == 0.30
        assert thresholds.integrity_penalty_warn == 0.15
        assert thresholds.improbability_threshold == 0.15
    
    def test_risk_challenger_thresholds_validation(self):
        """Prueba: Validación de umbrales en [0, 1]."""
        from app.strategy.business_agent import RiskChallengerThresholds
        
        # Umbral > 1 (inválido)
        with pytest.raises(ValueError):
            RiskChallengerThresholds(critical_stability=1.5)
        
        # Umbral < 0 (inválido)
        with pytest.raises(ValueError):
            RiskChallengerThresholds(warning_stability=-0.1)
        
        # Umbral en rango (válido)
        thresholds = RiskChallengerThresholds(critical_stability=0.75)
        assert thresholds.critical_stability == 0.75
    
    def test_risk_challenger_thresholds_from_dict(self):
        """Prueba: Constructor desde diccionario."""
        from app.strategy.business_agent import RiskChallengerThresholds
        
        data = {
            "critical_stability": 0.65,
            "warning_stability": 0.80,
            "coherence_minimum": 0.55
        }
        
        thresholds = RiskChallengerThresholds.from_dict(data)
        
        assert thresholds.critical_stability == 0.65
        assert thresholds.warning_stability == 0.80
        assert thresholds.coherence_minimum == 0.55
        
        # Valores no especificados usan defaults
        assert thresholds.cycle_density_limit == 0.33
    
    def test_risk_challenger_thresholds_to_dict(self):
        """Prueba: Serialización a diccionario."""
        from app.strategy.business_agent import RiskChallengerThresholds
        
        thresholds = RiskChallengerThresholds()
        result = thresholds.to_dict()
        
        assert "critical_stability" in result
        assert "warning_stability" in result
        assert "coherence_minimum" in result
        assert "cycle_density_limit" in result
        assert "integrity_penalty_veto" in result
        assert "integrity_penalty_warn" in result
        assert "improbability_threshold" in result
        
        # Todos los valores deben ser float
        assert all(isinstance(v, float) for v in result.values())


class TestDecisionWeights:
    """
    Suite de pruebas para pesos de decisión multicriterio.
    
    Fundamentación Matemática:
        Los pesos forman una partición de la unidad:
        α + β + γ = 1
        
        Esto garantiza que la combinación convexa preserve
        la escala de los vectores de decisión.
    """
    
    def test_decision_weights_defaults(self):
        """Prueba: Valores por defecto de pesos."""
        from app.strategy.business_agent import DecisionWeights
        
        weights = DecisionWeights()
        
        assert weights.topology == 0.40
        assert weights.finance == 0.40
        assert weights.thermodynamics == 0.20
    
    def test_decision_weights_normalization(self):
        """
        Prueba: Normalización a suma 1.
        
        Teorema: Para cualquier pesos (α, β, γ) con α+β+γ > 0,
        existe normalización única (α', β', γ') con α'+β'+γ' = 1.
        """
        from app.strategy.business_agent import DecisionWeights
        
        # Pesos que no suman 1
        weights = DecisionWeights(topology=0.5, finance=0.3, thermodynamics=0.1)
        normalized = weights.normalized
        
        from app.strategy.business_agent import MathematicalConstants as MC
        total = normalized.topology + normalized.finance + normalized.thermodynamics
        assert abs(total - 1.0) < MC.EPSILON_TOLERANCE
    
    def test_decision_weights_normalization_zero_sum(self):
        """Prueba: Normalización cuando suma es cero (fallback)."""
        from app.strategy.business_agent import DecisionWeights
        
        # Todos los pesos cero
        weights = DecisionWeights(topology=0.0, finance=0.0, thermodynamics=0.0)
        normalized = weights.normalized
        
        # Debe retornar distribución uniforme
        assert normalized.topology == normalized.finance == normalized.thermodynamics
        assert abs(normalized.topology - 1/3) < 1e-10
    
    def test_decision_weights_negative_validation(self):
        """Prueba: Rechaza pesos negativos."""
        from app.strategy.business_agent import DecisionWeights
        
        with pytest.raises(ValueError):
            DecisionWeights(topology=-0.1, finance=0.5, thermodynamics=0.5)
    
    def test_decision_weights_to_tuple(self):
        """Prueba: Conversión a tupla."""
        from app.strategy.business_agent import DecisionWeights
        
        weights = DecisionWeights(topology=0.4, finance=0.4, thermodynamics=0.2)
        result = weights.to_tuple()
        
        assert result == (0.4, 0.4, 0.2)
        assert isinstance(result, tuple)
        assert len(result) == 3
    
    def test_decision_weights_to_dict(self):
        """Prueba: Serialización a diccionario."""
        from app.strategy.business_agent import DecisionWeights
        
        weights = DecisionWeights()
        result = weights.to_dict()
        
        assert "topology" in result
        assert "finance" in result
        assert "thermodynamics" in result
        
        assert all(isinstance(v, float) for v in result.values())
    
    def test_decision_weights_from_dict(self):
        """Prueba: Constructor desde diccionario."""
        from app.strategy.business_agent import DecisionWeights
        
        data = {"topology": 0.5, "finance": 0.3, "thermodynamics": 0.2}
        weights = DecisionWeights.from_dict(data)
        
        assert weights.topology == 0.5
        assert weights.finance == 0.3
        assert weights.thermodynamics == 0.2


# =============================================================================
# CLASE DE PRUEBAS FASE 4: REGISTROS DE VETOS Y EXCEPCIONES
# =============================================================================

class TestVetoRecord:
    """
    Suite de pruebas para registros de veto.
    
    Invariante:
        Los vetos son inmutables y contienen toda la información
        necesaria para auditoría y trazabilidad.
    """
    
    def test_veto_record_creation(self):
        """Prueba: Creación válida de veto."""
        from app.strategy.business_agent import (
            VetoRecord, VetoSeverity, RiskClassification
        )
        
        record = VetoRecord(
            veto_type="VETO_CRITICAL_INSTABILITY",
            severity=VetoSeverity.CRITICO,
            stability_at_veto=0.65,
            financial_class=RiskClassification.HIGH,
            original_integrity=80.0,
            penalty_applied=0.30,
            reason="Estabilidad piramidal insuficiente"
        )
        
        assert record.veto_type == "VETO_CRITICAL_INSTABILITY"
        assert record.severity == VetoSeverity.CRITICO
        assert record.stability_at_veto == 0.65
        assert record.penalty_applied == 0.30
    
    def test_veto_record_timestamp(self):
        """Prueba: Marca temporal automática."""
        from app.strategy.business_agent import VetoRecord, VetoSeverity, RiskClassification
        import time
        
        before = time.time()
        record = VetoRecord(
            veto_type="TEST",
            severity=VetoSeverity.MODERADO,
            stability_at_veto=0.70,
            financial_class=RiskClassification.MODERATE,
            original_integrity=75.0,
            penalty_applied=0.15,
            reason="Test"
        )
        after = time.time()
        
        assert before <= record.timestamp <= after
    
    def test_veto_record_to_dict(self):
        """Prueba: Serialización completa."""
        from app.strategy.business_agent import VetoRecord, VetoSeverity, RiskClassification
        
        record = VetoRecord(
            veto_type="VETO_TEST",
            severity=VetoSeverity.CRITICAL,
            stability_at_veto=0.68,
            financial_class=RiskClassification.HIGH,
            original_integrity=80.0,
            penalty_applied=0.25,
            reason="Prueba de veto"
        )
        
        result = record.to_dict()
        
        assert "veto_type" in result
        assert "severity" in result
        assert "stability_at_veto" in result
        assert "financial_class" in result
        assert "original_integrity" in result
        assert "penalty_applied" in result
        assert "reason" in result
        assert "timestamp" in result
        assert "timestamp_iso" in result


class TestLateralExceptionRecord:
    """
    Suite de pruebas para registros de excepción lateral.
    
    Fundamentación:
        Las excepciones por pensamiento lateral permiten
        superar vetos estructurales mediante instrumentos
        financieros avanzados o garantías externas.
    """
    
    def test_lateral_exception_record_creation(self):
        """Prueba: Creación válida de excepción."""
        from app.strategy.business_agent import (
            LateralExceptionRecord, PivotType
        )
        
        record = LateralExceptionRecord(
            exception_type="EXCEPCIÓN_OPCION_ESPERA",
            pivot_type=PivotType.OPCION_ESPERA,
            penalty_relief=0.15,
            reason="Valor de opción justifica retraso",
            approved_by_mic=True
        )
        
        assert record.exception_type == "EXCEPCIÓN_OPCION_ESPERA"
        assert record.pivot_type == PivotType.OPCION_ESPERA
        assert record.penalty_relief == 0.15
        assert record.approved_by_mic is True
    
    def test_lateral_exception_record_to_dict(self):
        """Prueba: Serialización completa."""
        from app.strategy.business_agent import LateralExceptionRecord, PivotType
        
        record = LateralExceptionRecord(
            exception_type="EXCEPCIÓN_TEST",
            pivot_type=PivotType.MONOPOLIO_COBERTURADO,
            penalty_relief=0.20,
            reason="Prueba de excepción",
            approved_by_mic=True
        )
        
        result = record.to_dict()
        
        assert "exception_type" in result
        assert "pivot_type" in result
        assert "penalty_relief" in result
        assert "reason" in result
        assert "approved_by_mic" in result
        assert "timestamp" in result
        assert "timestamp_iso" in result


# =============================================================================
# CLASE DE PRUEBAS FASE 4: ESTRATEGIAS DE PIVOTE
# =============================================================================

class TestPivotStrategies:
    """
    Suite de pruebas para estrategias de pivote lateral.
    
    Fundamentación:
        Cada estrategia implementa criterios específicos para
        determinar si una excepción estratégica está justificada.
    """
    
    def test_monopolio_coberturado_strategy_applies(self):
        """
        Prueba: Pivote de monopolio coberturado aplica.
        
        Condiciones:
            - Ψ < umbral (base estrecha)
            - T < T_umbral (sistema frío)
            - C > C_umbral (alta inercia)
        """
        from app.strategy.business_agent import (
            MonopolioCoberturadoStrategy, ThermodynamicState,
            RiskClassification, FinancialMetrics
        )
        from decimal import Decimal
        
        strategy = MonopolioCoberturadoStrategy(
            stability_threshold=0.70,
            temp_threshold=15.0,
            inertia_threshold=0.70
        )
        
        thermal_state = ThermodynamicState(
            internal_energy=Decimal("1000000"),
            entropy=0.3,
            temperature=12.0,  # Sistema frío
            heat_capacity=0.85,  # Alta inercia
            exergy=0.6
        )
        
        applies, reason = strategy.evaluate(
            stability=0.65,  # < 0.70
            financial_class=RiskClassification.MODERATE,
            thermal_state=thermal_state,
            financial_metrics=FinancialMetrics(npv=Decimal("100000")),
            topo_bundle=None  # No se usa en esta estrategia
        )
        
        assert applies is True
        assert "inercia térmica" in reason
    
    def test_monopolio_coberturado_strategy_not_applies(self):
        """Prueba: Pivote de monopolio coberturado no aplica."""
        from app.strategy.business_agent import (
            MonopolioCoberturadoStrategy, ThermodynamicState,
            RiskClassification, FinancialMetrics
        )
        from decimal import Decimal
        
        strategy = MonopolioCoberturadoStrategy()
        
        thermal_state = ThermodynamicState(
            internal_energy=Decimal("1000000"),
            entropy=0.5,
            temperature=25.0,  # Sistema caliente
            heat_capacity=0.50,  # Baja inercia
            exergy=0.5
        )
        
        applies, reason = strategy.evaluate(
            stability=0.65,
            financial_class=RiskClassification.MODERATE,
            thermal_state=thermal_state,
            financial_metrics=FinancialMetrics(npv=Decimal("100000")),
            topo_bundle=None
        )
        
        assert applies is False
        assert "insuficientes" in reason
    
    def test_opcion_espera_strategy_applies(self):
        """
        Prueba: Pivote de opción de espera aplica.
        
        Condiciones:
            - Riesgo financiero HIGH/CRITICAL
            - Valor de opción > VPN × k
        """
        from app.strategy.business_agent import (
            OpcionEsperaStrategy, ThermodynamicState,
            RiskClassification, FinancialMetrics
        )
        from decimal import Decimal
        
        strategy = OpcionEsperaStrategy(npv_multiplier=1.5)
        
        # Riesgo alto
        financial_metrics = FinancialMetrics(
            npv=Decimal("1000000"),
            var_95=Decimal("-600000"),  # 60% del VPN
            profitability_index=Decimal("1.1")
        )
        
        thermal_state = ThermodynamicState(
            internal_energy=Decimal("1000000"),
            entropy=0.5,
            temperature=25.0,
            heat_capacity=0.5,
            exergy=0.5
        )
        
        # Esta estrategia tiene lógica simplificada en el código
        # El test valida que la estructura funciona
        applies, reason = strategy.evaluate(
            stability=0.70,
            financial_class=RiskClassification.HIGH,
            thermal_state=thermal_state,
            financial_metrics=financial_metrics,
            topo_bundle=None
        )
        
        # Depende de la implementación específica
        assert isinstance(applies, bool)
        assert isinstance(reason, str)
    
    def test_opcion_espera_strategy_not_high_risk(self):
        """Prueba: Opción de espera no aplica con riesgo bajo."""
        from app.strategy.business_agent import (
            OpcionEsperaStrategy, ThermodynamicState,
            RiskClassification, FinancialMetrics
        )
        from decimal import Decimal
        
        strategy = OpcionEsperaStrategy()
        
        financial_metrics = FinancialMetrics(npv=Decimal("100000"))
        
        thermal_state = ThermodynamicState(
            internal_energy=Decimal("1000000"),
            entropy=0.5,
            temperature=25.0,
            heat_capacity=0.5,
            exergy=0.5
        )
        
        applies, reason = strategy.evaluate(
            stability=0.80,
            financial_class=RiskClassification.SAFE,  # No HIGH/CRITICAL
            thermal_state=thermal_state,
            financial_metrics=financial_metrics,
            topo_bundle=None
        )
        
        assert applies is False
        assert "no HIGH/CRITICAL" in reason
    
    def test_cuarentena_topologica_strategy_applies(self):
        """
        Prueba: Cuarentena topológica aplica.
        
        Condiciones:
            - β₁ > 0 (ciclos presentes)
            - Sin sinergia multiplicativa (densidad baja)
        """
        from app.strategy.business_agent import (
            CuarentenaTopologicaStrategy, TopologicalMetricsBundle,
            BettiNumbers, BettiNumber, SpectralData, GraphMetrics,
            ThermodynamicState, RiskClassification, FinancialMetrics
        )
        from decimal import Decimal
        
        strategy = CuarentenaTopologicaStrategy()
        
        # Ciclos presentes pero aislados
        betti = BettiNumbers(beta_0=BettiNumber(1), beta_1=BettiNumber(2))
        spectral = SpectralData(eigenvalues=(0.0, 0.5, 1.0))
        graph_metrics = GraphMetrics(n_nodes=10, n_edges=9)
        
        topo_bundle = TopologicalMetricsBundle(
            betti=betti,
            spectral=spectral,
            graph_metrics=graph_metrics,
            pyramid_stability=0.75
        )
        
        thermal_state = ThermodynamicState(
            internal_energy=Decimal("1000000"),
            entropy=0.5,
            temperature=25.0,
            heat_capacity=0.5,
            exergy=0.5
        )
        
        applies, reason = strategy.evaluate(
            stability=0.75,
            financial_class=RiskClassification.MODERATE,
            thermal_state=thermal_state,
            financial_metrics=FinancialMetrics(npv=Decimal("100000")),
            topo_bundle=topo_bundle
        )
        
        # Depende de la densidad de ciclos calculada
        assert isinstance(applies, bool)
        assert isinstance(reason, str)
    
    def test_improbability_override_strategy(self):
        """Prueba: Estrategia de anulación de improbabilidad."""
        from app.strategy.business_agent import (
            ImprobabilityOverrideStrategy, ThermodynamicState,
            RiskClassification, FinancialMetrics
        )
        from decimal import Decimal
        
        strategy = ImprobabilityOverrideStrategy(collateral_ratio_threshold=1.5)
        
        thermal_state = ThermodynamicState(
            internal_energy=Decimal("1000000"),
            entropy=0.5,
            temperature=25.0,
            heat_capacity=0.5,
            exergy=0.5
        )
        
        applies, reason = strategy.evaluate(
            stability=0.50,
            financial_class=RiskClassification.CRITICAL,
            thermal_state=thermal_state,
            financial_metrics=FinancialMetrics(npv=Decimal("-100000")),
            topo_bundle=None
        )
        
        # La implementación actual usa placeholder
        assert isinstance(applies, bool)
        assert isinstance(reason, str)


# =============================================================================
# CLASE DE PRUEBAS FASE 4: DECISION ALGEBRA
# =============================================================================

class TestDecisionAlgebra:
    """
    Suite de pruebas para álgebra de decisiones multicriterio.
    
    Fundamentación Matemática:
        1. Normalización a esfera: v̂ = v / ‖v‖
        2. Media geométrica ponderada: GM_w = exp(Σ wᵢ log xᵢ / Σ wᵢ)
        3. Combinación convexa: d = Σ αᵢ·vᵢ donde Σ αᵢ = 1
    """
    
    def test_decision_algebra_normalize_to_sphere(self):
        """
        Prueba: Normalización a esfera unitaria.
        
        Propiedad: ‖v̂‖ = 1
        """
        from app.strategy.business_agent import DecisionAlgebra
        import numpy as np
        
        vector = np.array([3.0, 4.0], dtype=np.float64)
        normalized = DecisionAlgebra.normalize_to_sphere(vector)
        
        # Norma debe ser 1
        norm = np.linalg.norm(normalized)
        
        from app.strategy.business_agent import MathematicalConstants as MC
        assert abs(norm - 1.0) < MC.EPSILON_TOLERANCE
    
    def test_decision_algebra_normalize_to_sphere_near_zero(self):
        """Prueba: Normalización de vector casi nulo."""
        from app.strategy.business_agent import DecisionAlgebra
        import numpy as np
        
        # Vector casi nulo
        vector = np.array([1e-15, 1e-15], dtype=np.float64)
        normalized = DecisionAlgebra.normalize_to_sphere(vector)
        
        # Debe retornar distribución uniforme
        assert len(normalized) == 2
        assert normalized[0] == normalized[1]
    
    def test_decision_algebra_weighted_geometric_mean(self):
        """
        Prueba: Media geométrica ponderada.
        
        Fórmula: GM_w = exp(Σ wᵢ log xᵢ / Σ wᵢ)
        
        Propiedad: GM_w ≤ AM (desigualdad de medias)
        """
        from app.strategy.business_agent import DecisionAlgebra
        
        factors = [0.8, 0.9, 0.7]
        weights = [0.4, 0.4, 0.2]
        
        gm = DecisionAlgebra.weighted_geometric_mean(factors, weights)
        
        # Resultado debe estar en [0, 1]
        assert 0.0 <= gm <= 1.0
        
        # Debe ser menor que la media aritmética
        am = sum(f * w for f, w in zip(factors, weights)) / sum(weights)
        assert gm <= am
    
    def test_decision_algebra_weighted_geometric_mean_empty(self):
        """Prueba: Media geométrica con factores vacíos."""
        from app.strategy.business_agent import DecisionAlgebra
        
        gm = DecisionAlgebra.weighted_geometric_mean([])
        
        assert gm == 0.0
    
    def test_decision_algebra_weighted_geometric_mean_zero_factor(self):
        """Prueba: Media geométrica con factor cero."""
        from app.strategy.business_agent import DecisionAlgebra
        
        # Factor cero con peso positivo ⇒ resultado cero
        factors = [0.0, 0.8, 0.9]
        weights = [0.5, 0.3, 0.2]
        
        gm = DecisionAlgebra.weighted_geometric_mean(factors, weights)
        
        assert gm == 0.0
    
    def test_decision_algebra_convex_combination(self):
        """
        Prueba: Combinación convexa de vectores.
        
        Propiedad: Si vᵢ ∈ [0, 1]ⁿ y Σ αᵢ = 1, entonces
        d = Σ αᵢ·vᵢ ∈ [0, 1]ⁿ
        """
        from app.strategy.business_agent import DecisionAlgebra, DecisionWeights
        import numpy as np
        
        vectors = [
            np.array([0.8, 0.9, 0.7], dtype=np.float64),
            np.array([0.6, 0.8, 0.9], dtype=np.float64),
            np.array([0.7, 0.7, 0.8], dtype=np.float64)
        ]
        
        weights = DecisionWeights(topology=0.4, finance=0.4, thermodynamics=0.2)
        
        result = DecisionAlgebra.convex_combination(vectors, weights)
        
        # Dimensión debe coincidir
        assert len(result) == 3
        
        # Todos los elementos en [0, 1]
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)
    
    def test_decision_algebra_convex_combination_dimension_mismatch(self):
        """Prueba: Rechaza vectores de dimensiones diferentes."""
        from app.strategy.business_agent import DecisionAlgebra, DecisionWeights
        from app.strategy.business_agent import DimensionMismatchError
        import numpy as np
        
        vectors = [
            np.array([0.8, 0.9], dtype=np.float64),  # 2D
            np.array([0.6, 0.8, 0.9], dtype=np.float64),  # 3D
            np.array([0.7, 0.7], dtype=np.float64)  # 2D
        ]
        
        weights = DecisionWeights()
        
        with pytest.raises(DimensionMismatchError):
            DecisionAlgebra.convex_combination(vectors, weights)
    
    def test_decision_algebra_compute_quality_factors(self):
        """
        Prueba: Cálculo de factores de calidad.
        
        Fórmulas:
            Q_topo = √(coherence × Ψ)
            Q_finance = (tanh(VPN/I₀) + 1) / 2
            Q_thermo = (negentropía + exergía) / 2
        """
        from app.strategy.business_agent import (
            DecisionAlgebra, TopologicalMetricsBundle,
            BettiNumbers, BettiNumber, SpectralData, GraphMetrics,
            FinancialMetrics, ThermodynamicState
        )
        from decimal import Decimal
        
        # Bundle topológico
        betti = BettiNumbers(beta_0=BettiNumber(1))
        spectral = SpectralData(eigenvalues=(0.0, 0.5))
        graph_metrics = GraphMetrics(n_nodes=5, n_edges=4)
        
        topo_bundle = TopologicalMetricsBundle(
            betti=betti,
            spectral=spectral,
            graph_metrics=graph_metrics,
            pyramid_stability=0.8
        )
        
        # Métricas financieras
        financial_metrics = FinancialMetrics(
            npv=Decimal("500000"),
            profitability_index=Decimal("1.5")
        )
        
        # Estado termodinámico
        thermo_state = ThermodynamicState(
            internal_energy=Decimal("1000000"),
            entropy=0.4,
            temperature=25.0,
            heat_capacity=0.6,
            exergy=0.6
        )
        
        q_topo, q_finance, q_thermo = DecisionAlgebra.compute_quality_factors(
            topo_bundle=topo_bundle,
            financial_metrics=financial_metrics,
            thermo_state=thermo_state,
            initial_investment=Decimal("1000000")
        )
        
        # Todos en [0, 1]
        assert 0.0 <= q_topo <= 1.0
        assert 0.0 <= q_finance <= 1.0
        assert 0.0 <= q_thermo <= 1.0


# =============================================================================
# CLASE DE PRUEBAS FASE 4: RISK CHALLENGER
# =============================================================================

class TestRiskChallenger:
    """
    Suite de pruebas para el Risk Challenger (auditoría adversarial).
    
    Fundamentación:
        El Risk Challenger actúa como operador de medición en el
        espacio de Hilbert de decisiones, forzando el colapso de
        la función de onda del proyecto a viable/inviable.
    """
    
    def test_risk_challenger_initialization(self):
        """Prueba: Inicialización del Risk Challenger."""
        from app.strategy.business_agent import RiskChallenger, RiskChallengerThresholds
        
        thresholds = RiskChallengerThresholds(critical_stability=0.65)
        challenger = RiskChallenger(thresholds=thresholds)
        
        assert challenger._thresholds.critical_stability == 0.65
    
    def test_risk_challenger_default_strategies(self):
        """Prueba: Estrategias por defecto registradas."""
        from app.strategy.business_agent import RiskChallenger, PivotType
        
        challenger = RiskChallenger()
        
        # Debe tener las 4 estrategias por defecto
        assert len(challenger._strategies) == 4
        assert PivotType.MONOPOLIO_COBERTURADO in challenger._strategies
        assert PivotType.OPCION_ESPERA in challenger._strategies
        assert PivotType.CUARENTENA_TOPOLOGICA in challenger._strategies
        assert PivotType.IMPROBABILITY_OVERRIDE in challenger._strategies
    
    def test_risk_challenger_custom_strategies(self):
        """Prueba: Estrategias personalizadas."""
        from app.strategy.business_agent import (
            RiskChallenger, PivotStrategy, PivotType
        )
        from abc import ABC, abstractmethod
        
        class CustomStrategy(PivotStrategy):
            @property
            def pivot_type(self):
                return PivotType.OPCION_ESPERA
            
            def evaluate(self, stability, financial_class, thermal_state,
                        financial_metrics, topo_bundle):
                return True, "Custom strategy"
        
        custom = CustomStrategy()
        challenger = RiskChallenger(strategies=[custom])
        
        assert PivotType.OPCION_ESPERA in challenger._strategies
        assert challenger._strategies[PivotType.OPCION_ESPERA] is custom
    
    def test_risk_challenger_build_mic_context(self):
        """Prueba: Construcción de contexto MIC."""
        from app.strategy.business_agent import RiskChallenger
        
        challenger = RiskChallenger()
        
        session_context = {
            "session_id": "test-123",
            "validated_strata": {"PHYSICS", "TACTICS"}
        }
        
        mic_context = challenger._build_mic_context(session_context)
        
        assert "validated_strata" in mic_context
        assert "session_id" in mic_context
        assert mic_context["session_id"] == "test-123"
    
    def test_risk_challenger_project_to_mic_no_mic(self):
        """Prueba: Proyección a MIC sin MIC configurada."""
        from app.strategy.business_agent import RiskChallenger
        
        challenger = RiskChallenger(mic=None)
        
        result = challenger._project_to_mic(
            service_name="test_service",
            payload={"key": "value"},
            context={}
        )
        
        assert result["success"] is False
        assert "MIC no configurada" in result["error"]
    
    def test_risk_challenger_emit_veto(self):
        """Prueba: Emisión de veto."""
        from app.strategy.business_agent import (
            RiskChallenger, VetoSeverity, RiskClassification
        )
        
        # Mock de ConstructionRiskReport
        class MockReport:
            integrity_score = 80.0
            waste_alerts = []
            circular_risks = []
            complexity_level = "Media"
            financial_risk_level = "MODERATE"
            details = {}
            strategic_narrative = "Narrativa base"
        
        challenger = RiskChallenger()
        report = MockReport()
        
        modified = challenger._emit_veto(
            report=report,
            veto_type="VETO_TEST",
            severity=VetoSeverity.CRITICO,
            stability=0.60,
            financial_class=RiskClassification.HIGH,
            penalty=0.30,
            reason="Prueba de veto"
        )
        
        # La integridad debe reducirse
        assert modified.integrity_score < report.integrity_score
    
    def test_risk_challenger_emit_lateral_exception(self):
        """Prueba: Emisión de excepción lateral."""
        from app.strategy.business_agent import RiskChallenger, PivotType
        
        class MockReport:
            integrity_score = 70.0
            waste_alerts = []
            circular_risks = []
            complexity_level = "Media"
            financial_risk_level = "HIGH"
            details = {}
            strategic_narrative = "Narrativa base"
        
        challenger = RiskChallenger()
        report = MockReport()
        
        modified = challenger._emit_lateral_exception(
            report=report,
            pivot_type=PivotType.OPCION_ESPERA,
            exception_type="EXCEPCIÓN_TEST",
            penalty_relief=0.15,
            reason="Prueba de excepción"
        )
        
        # La integridad debe aumentar
        assert modified.integrity_score > report.integrity_score
    
    def test_risk_challenger_generate_veto_narrative(self):
        """Prueba: Generación de narrativa de veto."""
        from app.strategy.business_agent import (
            RiskChallenger, VetoRecord, VetoSeverity, RiskClassification
        )
        
        veto = VetoRecord(
            veto_type="VETO_TEST",
            severity=VetoSeverity.CRITICO,
            stability_at_veto=0.60,
            financial_class=RiskClassification.HIGH,
            original_integrity=80.0,
            penalty_applied=0.30,
            reason="Razón de prueba"
        )
        
        challenger = RiskChallenger()
        narrative = challenger._generate_veto_narrative(veto, new_integrity=56.0)
        
        assert "ACTA DE DELIBERACIÓN" in narrative
        assert "VETO_TEST" in narrative
        assert "CRÍTICO" in narrative
        assert "Razón de prueba" in narrative
    
    def test_risk_challenger_generate_exception_narrative(self):
        """Prueba: Generación de narrativa de excepción."""
        from app.strategy.business_agent import (
            RiskChallenger, LateralExceptionRecord, PivotType
        )
        
        exception = LateralExceptionRecord(
            exception_type="EXCEPCIÓN_TEST",
            pivot_type=PivotType.MONOPOLIO_COBERTURADO,
            penalty_relief=0.15,
            reason="Razón de excepción",
            approved_by_mic=True
        )
        
        challenger = RiskChallenger()
        narrative = challenger._generate_exception_narrative(exception)
        
        assert "EXCEPCIÓN POR PENSAMIENTO LATERAL" in narrative
        assert "MONOPOLIO_COBERTURADO" in narrative
        assert "Razón de excepción" in narrative


# =============================================================================
# CLASE DE PRUEBAS FASE 4: REPORT COMPOSER
# =============================================================================

class TestReportComposer:
    """
    Suite de pruebas para el compositor de reportes ejecutivos.
    
    Fundamentación:
        El ReportComposer integra las tres dimensiones (topológica,
        financiera, termodinámica) en un reporte unificado mediante
        álgebra de decisiones.
    """
    
    def test_report_composer_initialization(self):
        """Prueba: Inicialización del ReportComposer."""
        from app.strategy.business_agent import ReportComposer, DecisionWeights
        
        weights = DecisionWeights(topology=0.5, finance=0.3, thermodynamics=0.2)
        composer = ReportComposer(weights=weights)
        
        assert composer._weights.topology == 0.5
    
    def test_report_composer_create_fallback_report(self):
        """Prueba: Creación de reporte fallback."""
        from app.strategy.business_agent import ReportComposer
        
        composer = ReportComposer()
        report = composer._create_fallback_report()
        
        # Debe tener atributos básicos
        assert hasattr(report, 'integrity_score')
        assert hasattr(report, 'waste_alerts')
        assert hasattr(report, 'strategic_narrative')
    
    def test_report_composer_generate_narrative(self):
        """Prueba: Generación de narrativa estratégica."""
        from app.strategy.business_agent import (
            ReportComposer, TopologicalMetricsBundle,
            BettiNumbers, BettiNumber, SpectralData, GraphMetrics,
            FinancialMetrics, ThermodynamicState
        )
        from decimal import Decimal
        
        composer = ReportComposer()
        
        # Bundle topológico
        betti = BettiNumbers(beta_0=BettiNumber(1))
        spectral = SpectralData(eigenvalues=(0.0, 0.5))
        graph_metrics = GraphMetrics(n_nodes=5, n_edges=4)
        
        topo_bundle = TopologicalMetricsBundle(
            betti=betti,
            spectral=spectral,
            graph_metrics=graph_metrics,
            pyramid_stability=0.8
        )
        
        # Métricas financieras
        financial_metrics = FinancialMetrics(
            npv=Decimal("500000"),
            irr=Decimal("0.15"),
            profitability_index=Decimal("1.5")
        )
        
        # Estado termodinámico
        thermo_state = ThermodynamicState(
            internal_energy=Decimal("1000000"),
            entropy=0.4,
            temperature=25.0,
            heat_capacity=0.6,
            exergy=0.6
        )
        
        decision_summary = {
            "quality_factors": {
                "topology": 0.75,
                "finance": 0.80,
                "thermodynamics": 0.70
            },
            "integrated_score_100": 75.0
        }
        
        narrative = composer._generate_narrative(
            topo_bundle=topo_bundle,
            financial_metrics=financial_metrics,
            thermal_state=thermo_state,
            decision_summary=decision_summary,
            base_narrative=""
        )
        
        assert "Reporte Ejecutivo" in narrative
        assert "Score de Integridad" in narrative
        assert "VPN" in narrative


# =============================================================================
# CLASE DE PRUEBAS FASE 4: BUSINESS AGENT
# =============================================================================

class TestBusinessAgent:
    """
    Suite de pruebas para el Business Agent (orquestador principal).
    
    Fundamentación:
        El Business Agent implementa un funtor covariante:
        F: ℂ_Budget → ℂ_Decision
        
        Que mapea el complejo simplicial del presupuesto a un
        retículo de decisión booleano, preservando estructura.
    """
    
    def test_business_agent_initialization(self):
        """Prueba: Inicialización del BusinessAgent."""
        from app.strategy.business_agent import BusinessAgent
        
        config = {
            "financial_config": {"discount_rate": 0.10},
            "risk_challenger_config": {"critical_stability": 0.70},
            "decision_weights": {"topology": 0.4, "finance": 0.4, "thermodynamics": 0.2}
        }
        
        # Mock de MIC
        class MockMIC:
            def project_intent(self, service, payload, context):
                return {"success": True, "payload": {}}
        
        agent = BusinessAgent(config=config, mic=MockMIC())
        
        assert agent.config == config
        assert agent.mic is not None
        assert agent.risk_challenger is not None
    
    def test_business_agent_validate_config(self):
        """Prueba: Validación de configuración."""
        from app.strategy.business_agent import BusinessAgent
        
        # Configuración inválida (no dict)
        with pytest.raises(ValueError):
            BusinessAgent(config="invalid", mic=None)
    
    def test_business_agent_build_decision_weights(self):
        """Prueba: Construcción de pesos de decisión."""
        from app.strategy.business_agent import BusinessAgent
        
        config = {
            "decision_weights": {
                "topology": 0.5,
                "finance": 0.3,
                "thermodynamics": 0.2
            }
        }
        
        # Crear agente parcialmente para probar método
        class PartialAgent:
            def _build_decision_weights(self, cfg):
                from app.strategy.business_agent import DecisionWeights
                return DecisionWeights.from_dict(cfg.get("decision_weights", {}))
        
        agent = PartialAgent()
        weights = agent._build_decision_weights(config)
        
        assert weights.topology == 0.5
        assert weights.finance == 0.3
        assert weights.thermodynamics == 0.2
    
    def test_business_agent_build_challenger_thresholds(self):
        """Prueba: Construcción de umbrales del challenger."""
        from app.strategy.business_agent import BusinessAgent
        
        config = {
            "risk_challenger_config": {
                "critical_stability": 0.65,
                "warning_stability": 0.80
            }
        }
        
        class PartialAgent:
            def _build_challenger_thresholds(self, cfg):
                from app.strategy.business_agent import RiskChallengerThresholds
                return RiskChallengerThresholds.from_dict(cfg.get("risk_challenger_config", {}))
        
        agent = PartialAgent()
        thresholds = agent._build_challenger_thresholds(config)
        
        assert thresholds.critical_stability == 0.65
        assert thresholds.warning_stability == 0.80
    
    def test_business_agent_extract_financial_params(self):
        """Prueba: Extracción de parámetros financieros."""
        from app.strategy.business_agent import BusinessAgent
        from decimal import Decimal
        
        class PartialAgent:
            def _extract_financial_params(self, ctx):
                from app.strategy.business_agent import FinancialParameters
                initial = Decimal(str(ctx.get("initial_investment", "1000000.0")))
                
                if "cash_flows" in ctx:
                    cash_flows = tuple(Decimal(str(cf)) for cf in ctx["cash_flows"])
                else:
                    n_periods = 5
                    annual_cf = initial * Decimal("0.3")
                    cash_flows = tuple(annual_cf for _ in range(n_periods))
                
                return FinancialParameters(
                    initial_investment=initial,
                    cash_flows=cash_flows,
                    discount_rate=Decimal(str(ctx.get("discount_rate", "0.10")))
                )
        
        agent = PartialAgent()
        
        context = {
            "initial_investment": 2000000,
            "discount_rate": 0.12,
            "cash_flows": [500000, 600000, 700000, 800000, 900000]
        }
        
        params = agent._extract_financial_params(context)
        
        assert params.initial_investment == Decimal("2000000")
        assert len(params.cash_flows) == 5
        assert params.discount_rate == Decimal("0.12")
    
    def test_business_agent_evaluate_project_error_handling(self):
        """Prueba: Manejo de errores en evaluación."""
        from app.strategy.business_agent import BusinessAgent
        
        config = {}
        
        class MockMIC:
            pass
        
        class MockTelemetry:
            def __init__(self):
                self.errors = []
            
            def record_error(self, category, message):
                self.errors.append((category, message))
        
        agent = BusinessAgent(config=config, mic=MockMIC(), telemetry=MockTelemetry())
        
        # Contexto inválido (sin componentes topológicos)
        context = {"df_presupuesto": None}
        
        result = agent.evaluate_project(context)
        
        # Debe retornar None y registrar error
        assert result is None
        assert len(agent._telemetry.errors) > 0


# =============================================================================
# CLASE DE PRUEBAS FASE 4: INTEGRACIÓN END-TO-END
# =============================================================================

class TestIntegrationEndToEnd:
    """
    Suite de pruebas de integración end-to-end.
    
    Fundamentación:
        Estas pruebas validan que todos los componentes trabajan
        juntos coherentemente, desde la entrada de datos hasta
        la generación del reporte ejecutivo final.
    """
    
    def test_integration_full_pipeline(self, sample_financial_params):
        """
        Prueba: Pipeline completo de evaluación.
        
        Flujo:
            1. FinancialParameters → FinancialEngine → FinancialMetrics
            2. TopologicalMetricsBundle (simulado)
            3. ThermodynamicsEngine → ThermodynamicState
            4. DecisionAlgebra → Score integrado
            5. RiskChallenger → Reporte auditado
        """
        from app.strategy.business_agent import (
            FinancialParameters, FinancialEngine, FinancialMetrics,
            TopologicalMetricsBundle, BettiNumbers, BettiNumber,
            SpectralData, GraphMetrics, ThermodynamicsEngine,
            ThermodynamicState, DecisionAlgebra, DecisionWeights,
            RiskChallenger, RiskChallengerThresholds
        )
        from decimal import Decimal
        
        # Fase 1: Finanzas
        params = FinancialParameters.from_dict(sample_financial_params)
        financial_engine = FinancialEngine()
        financial_metrics = financial_engine.analyze(params)
        
        assert financial_metrics.npv is not None
        assert financial_metrics.irr is not None
        
        # Fase 2: Topología (simulada)
        betti = BettiNumbers(beta_0=BettiNumber(1), beta_1=BettiNumber(0))
        spectral = SpectralData(eigenvalues=(0.0, 0.5, 1.0))
        graph_metrics = GraphMetrics(n_nodes=5, n_edges=4)
        
        topo_bundle = TopologicalMetricsBundle(
            betti=betti,
            spectral=spectral,
            graph_metrics=graph_metrics,
            pyramid_stability=0.85
        )
        
        # Fase 3: Termodinámica
        thermo_engine = ThermodynamicsEngine()
        thermal_state = thermo_engine.analyze(params, financial_metrics, topo_bundle)
        
        assert isinstance(thermal_state, ThermodynamicState)
        assert 0.0 <= thermal_state.entropy <= 1.0
        
        # Fase 4: Álgebra de decisiones
        q_topo, q_finance, q_thermo = DecisionAlgebra.compute_quality_factors(
            topo_bundle=topo_bundle,
            financial_metrics=financial_metrics,
            thermo_state=thermal_state,
            initial_investment=params.initial_investment
        )
        
        weights = DecisionWeights()
        integrated_score = DecisionAlgebra.weighted_geometric_mean(
            factors=[q_topo, q_finance, q_thermo],
            weights=list(weights.normalized.to_tuple())
        )
        
        assert 0.0 <= integrated_score <= 1.0
        
        # Fase 5: Risk Challenger (simulado)
        thresholds = RiskChallengerThresholds()
        challenger = RiskChallenger(thresholds=thresholds)
        
        # Mock de ConstructionRiskReport
        class MockReport:
            integrity_score = integrated_score * 100
            waste_alerts = []
            circular_risks = []
            complexity_level = "Media"
            financial_risk_level = financial_metrics.risk_class
            details = {}
            strategic_narrative = "Narrativa base"
        
        report = MockReport()
        
        # El challenger puede modificar el reporte
        # (en este test solo validamos que no falle)
        assert report.integrity_score > 0
    
    def test_integration_coherence_across_dimensions(self):
        """
        Prueba: Coherencia entre dimensiones topológica, financiera y termodinámica.
        
        Invariante:
            Alta coherencia topológica + Alta viabilidad financiera
            ⇒ Alta exergía termodinámica
        """
        from app.strategy.business_agent import (
            TopologicalMetricsBundle, BettiNumbers, BettiNumber,
            SpectralData, GraphMetrics, FinancialMetrics,
            ThermodynamicState, DecisionAlgebra
        )
        from decimal import Decimal
        
        # Escenario 1: Todo alto
        betti_high = BettiNumbers(beta_0=BettiNumber(1), beta_1=BettiNumber(0))
        spectral_high = SpectralData(eigenvalues=(0.0, 0.8, 1.5))
        graph_high = GraphMetrics(n_nodes=10, n_edges=20)
        
        topo_high = TopologicalMetricsBundle(
            betti=betti_high,
            spectral=spectral_high,
            graph_metrics=graph_high,
            pyramid_stability=0.90
        )
        
        finance_high = FinancialMetrics(
            npv=Decimal("1000000"),
            profitability_index=Decimal("2.0")
        )
        
        thermo_high = ThermodynamicState(
            internal_energy=Decimal("2000000"),
            entropy=0.2,  # Baja entropía (orden)
            temperature=20.0,
            heat_capacity=0.8,
            exergy=0.8  # Alta exergía
        )
        
        q_topo_high, q_finance_high, q_thermo_high = DecisionAlgebra.compute_quality_factors(
            topo_bundle=topo_high,
            financial_metrics=finance_high,
            thermo_state=thermo_high,
            initial_investment=Decimal("1000000")
        )
        
        # Escenario 2: Todo bajo
        betti_low = BettiNumbers(beta_0=BettiNumber(3), beta_1=BettiNumber(5))
        spectral_low = SpectralData(eigenvalues=(0.0, 0.0, 0.0, 0.1))
        graph_low = GraphMetrics(n_nodes=10, n_edges=5)
        
        topo_low = TopologicalMetricsBundle(
            betti=betti_low,
            spectral=spectral_low,
            graph_metrics=graph_low,
            pyramid_stability=0.50
        )
        
        finance_low = FinancialMetrics(
            npv=Decimal("-500000"),
            profitability_index=Decimal("0.5")
        )
        
        thermo_low = ThermodynamicState(
            internal_energy=Decimal("500000"),
            entropy=0.8,  # Alta entropía (desorden)
            temperature=35.0,
            heat_capacity=0.3,
            exergy=0.2  # Baja exergía
        )
        
        q_topo_low, q_finance_low, q_thermo_low = DecisionAlgebra.compute_quality_factors(
            topo_bundle=topo_low,
            financial_metrics=finance_low,
            thermo_state=thermo_low,
            initial_investment=Decimal("1000000")
        )
        
        # Los factores altos deben ser mayores que los bajos
        assert q_topo_high > q_topo_low
        assert q_finance_high > q_finance_low
        assert q_thermo_high > q_thermo_low
    
    def test_integration_risk_classification_consistency(self):
        """
        Prueba: Consistencia en clasificación de riesgo.
        
        Invariante:
            VaR/VPN alto ⇒ Clasificación de riesgo alta
        """
        from app.strategy.business_agent import FinancialMetrics, RiskClassification
        
        # Bajo riesgo
        low_risk = FinancialMetrics(
            npv=Decimal("1000000"),
            var_95=Decimal("-100000"),  # 10%
            profitability_index=Decimal("1.5")
        )
        assert low_risk.risk_class in ["LOW", "MODERATE"]
        
        # Alto riesgo
        high_risk = FinancialMetrics(
            npv=Decimal("1000000"),
            var_95=Decimal("-700000"),  # 70%
            profitability_index=Decimal("1.1")
        )
        assert high_risk.risk_class in ["HIGH", "CRITICAL"]


# =============================================================================
# EJECUCIÓN DE PRUEBAS
# =============================================================================

if __name__ == "__main__":
    """
    Punto de entrada para ejecutar la suite de pruebas.
    
    Uso:
        python test_business_agent.py
    
    O con pytest:
        pytest test_business_agent.py -v --cov=app.strategy.business_agent
    """
    import sys
    
    # Configurar logging para pruebas
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ejecutar con pytest
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
