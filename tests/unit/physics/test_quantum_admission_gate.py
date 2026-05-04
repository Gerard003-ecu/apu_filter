"""
═════════════════════════════════════════════════════════════════════════════
MÓDULO: Test Suite para Quantum Admission Gate
VERSIÓN: 1.0.0 - Suite Rigurosa de Validación
UBICACIÓN: tests/unit/physics/test_quantum_admission_gate.py
═════════════════════════════════════════════════════════════════════════════

ARQUITECTURA DE TESTING:

Este módulo implementa una suite exhaustiva de pruebas unitarias, de integración
y de propiedades algebraicas para el Quantum Admission Gate.

ESTRUCTURA JERÁRQUICA:

    1. FIXTURES Y MOCKS (Objetos de Test Reutilizables)
       ├── MockTopologicalWatcher (χ² configurable)
       ├── MockLaplaceOracle (σ configurable)
       ├── MockSheafOrchestrator (E_frust configurable)
       └── Payloads de referencia (canónicos)

    2. TESTS DE UNIDAD (Componentes Atómicos)
       ├── Constantes físicas
       ├── Estructuras de datos (Interval, WKBParams, Measurement)
       ├── Morfismos numéricos
       ├── Calculadores de entropía
       ├── Serialización canónica
       └── Calculadores físicos individuales

    3. TESTS DE INTEGRACIÓN (Componentes Compuestos)
       ├── Pipeline completo de admisión
       ├── Interacción entre moduladores
       ├── Propagación de errores
       └── Coherencia de observables

    4. TESTS DE PROPIEDADES ALGEBRAICAS
       ├── Axiomas del espacio de Hilbert
       ├── Conservación probabilística
       ├── Hermiticidad del operador
       ├── Funtorialidad categórica
       └── No-clonación cuántica

    5. TESTS DE INVARIANTES FÍSICOS
       ├── No-negatividad de energías
       ├── Normalización de probabilidades
       ├── Consistencia estado-momentum
       ├── Validez de aproximación WKB
       └── Monotonía de funciones de trabajo

    6. TESTS DE CASOS LÍMITE
       ├── Payload vacío
       ├── Payload de tamaño máximo
       ├── Sistema inestable (σ > 0)
       ├── Barrera infinita (m_eff → ∞)
       ├── Veto cohomológico
       └── Entropía mínima/máxima

    7. TESTS DE PROPIEDADES ESTADÍSTICAS
       ├── Distribución de umbrales θ
       ├── Reproducibilidad determinista
       ├── Independencia de orden de inserción
       └── Estabilidad numérica

    8. TESTS DE REGRESIÓN
       ├── Casos conocidos de admisión/rechazo
       ├── Validación contra valores de referencia
       └── Compatibilidad con versiones anteriores

═════════════════════════════════════════════════════════════════════════════

CONVENCIONES DE NOMENCLATURA:

    - test_unit_*: Tests unitarios de componentes atómicos
    - test_integration_*: Tests de integración multi-componente
    - test_property_*: Tests de propiedades algebraicas/físicas
    - test_invariant_*: Tests de invariantes preservados
    - test_edge_*: Tests de casos límite
    - test_statistical_*: Tests de propiedades estadísticas
    - test_regression_*: Tests de regresión

ESTRATEGIA DE ASERCIONES:

    Cada test debe validar:
        1. Precondiciones (setup correcto)
        2. Ejecución sin excepciones (robustez)
        3. Postcondiciones (resultado esperado)
        4. Invariantes (propiedades preservadas)
        5. Efectos secundarios (logging, métricas)

═════════════════════════════════════════════════════════════════════════════
"""

import hashlib
import math
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Tuple
from unittest.mock import MagicMock, Mock, patch

import pytest

# Importaciones del módulo bajo test
from app.physics.quantum_admission_gate import (
    # Constantes
    PhysicalConstants,
    Const,
    
    # Excepciones
    QuantumAdmissionError,
    QuantumNumericalError,
    QuantumInterfaceError,
    QuantumStateError,
    WKBValidityError,
    CohomologicalVetoError,
    
    # Enumeraciones
    Eigenstate,
    
    # Estructuras de datos
    NumericInterval,
    WKBParameters,
    QuantumMeasurement,
    
    # Protocolos
    ITopologicalWatcher,
    ILaplaceOracle,
    ISheafCohomologyOrchestrator,
    
    # Morfismos numéricos
    NumericalMorphisms,
    NM,
    
    # Calculadores
    EntropyCalculator,
    PayloadSerializer,
    IncidentEnergyCalculator,
    WKBCalculator,
    WorkFunctionModulator,
    EffectiveMassModulator,
    CollapseThresholdGenerator,
    
    # Operador principal
    QuantumAdmissionGate,
)

# Importaciones de infraestructura MIC
from app.core.mic_algebra import CategoricalState, Morphism
from app.core.schemas import Stratum


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: FIXTURES Y MOCKS
# ═════════════════════════════════════════════════════════════════════════════

class MockTopologicalWatcher:
    """
    Mock del observador topológico con χ² configurable.
    
    Permite simular diferentes niveles de amenaza topológica para
    validar el comportamiento del modulador de función de trabajo.
    
    Invariantes:
        - get_mahalanobis_threat() siempre retorna float ≥ 0
        - Idempotencia: múltiples llamadas retornan mismo valor
    """
    
    def __init__(self, threat: float = 0.0, should_fail: bool = False):
        """
        Constructor con configuración de comportamiento.
        
        Args:
            threat: Valor de χ² a retornar (≥ 0).
            should_fail: Si True, lanza RuntimeError al llamar.
        """
        if threat < 0:
            raise ValueError(f"threat debe ser no negativo: {threat}")
        
        self._threat = threat
        self._should_fail = should_fail
        self._call_count = 0
    
    def get_mahalanobis_threat(self) -> float:
        """Implementación del protocolo ITopologicalWatcher."""
        self._call_count += 1
        
        if self._should_fail:
            raise RuntimeError("Mock configurado para fallar")
        
        return self._threat
    
    @property
    def call_count(self) -> int:
        """Contador de llamadas (para verificación)."""
        return self._call_count


class MockLaplaceOracle:
    """
    Mock del oráculo de Laplace con polo dominante configurable.
    
    Permite simular diferentes regímenes de estabilidad del sistema:
        - σ < -ε: sistema estable
        - -ε ≤ σ < 0: sistema marginalmente estable
        - σ ≥ 0: sistema inestable
    
    Invariantes:
        - get_dominant_pole_real() siempre retorna float finito
        - Idempotencia en ausencia de mutaciones
    """
    
    def __init__(self, pole_real: float = -1.0, should_fail: bool = False):
        """
        Constructor con configuración de comportamiento.
        
        Args:
            pole_real: Parte real del polo dominante σ ∈ ℝ.
            should_fail: Si True, lanza RuntimeError al llamar.
        """
        self._pole_real = pole_real
        self._should_fail = should_fail
        self._call_count = 0
    
    def get_dominant_pole_real(self) -> float:
        """Implementación del protocolo ILaplaceOracle."""
        self._call_count += 1
        
        if self._should_fail:
            raise RuntimeError("Mock configurado para fallar")
        
        return self._pole_real
    
    @property
    def call_count(self) -> int:
        """Contador de llamadas."""
        return self._call_count


class MockSheafOrchestrator:
    """
    Mock del orquestador cohomológico con energía de frustración configurable.
    
    Permite simular diferentes niveles de obstrucción topológica:
        - E ≈ 0: cohomología trivial (sin frustración)
        - E > tol: obstrucción cohomológica (veto activado)
    
    Invariantes:
        - get_global_frustration_energy() siempre retorna float ≥ 0
        - Idempotencia
    """
    
    def __init__(self, frustration: float = 0.0, should_fail: bool = False):
        """
        Constructor con configuración de comportamiento.
        
        Args:
            frustration: Energía de frustración E_frust ≥ 0.
            should_fail: Si True, lanza RuntimeError al llamar.
        """
        if frustration < 0:
            raise ValueError(f"frustration debe ser no negativa: {frustration}")
        
        self._frustration = frustration
        self._should_fail = should_fail
        self._call_count = 0
    
    def get_global_frustration_energy(self) -> float:
        """Implementación del protocolo ISheafCohomologyOrchestrator."""
        self._call_count += 1
        
        if self._should_fail:
            raise RuntimeError("Mock configurado para fallar")
        
        return self._frustration
    
    @property
    def call_count(self) -> int:
        """Contador de llamadas."""
        return self._call_count


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures de Pytest
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def stable_mocks() -> Tuple[MockTopologicalWatcher, MockLaplaceOracle, MockSheafOrchestrator]:
    """
    Fixture con mocks en configuración de sistema estable nominal.
    
    Returns:
        Tupla (topo_watcher, laplace_oracle, sheaf_orchestrator)
        con valores nominales para sistema saludable.
    """
    return (
        MockTopologicalWatcher(threat=1.0),
        MockLaplaceOracle(pole_real=-0.5),
        MockSheafOrchestrator(frustration=1e-12),
    )


@pytest.fixture
def unstable_mocks() -> Tuple[MockTopologicalWatcher, MockLaplaceOracle, MockSheafOrchestrator]:
    """
    Fixture con mocks en configuración de sistema inestable.
    
    Returns:
        Tupla con:
            - Alta amenaza topológica (χ² = 10.0)
            - Polo inestable (σ = 0.5)
            - Sin frustración cohomológica
    """
    return (
        MockTopologicalWatcher(threat=10.0),
        MockLaplaceOracle(pole_real=0.5),
        MockSheafOrchestrator(frustration=1e-12),
    )


@pytest.fixture
def veto_mocks() -> Tuple[MockTopologicalWatcher, MockLaplaceOracle, MockSheafOrchestrator]:
    """
    Fixture con mocks que activan veto cohomológico.
    
    Returns:
        Tupla con frustración > FRUSTRATION_VETO_TOL.
    """
    return (
        MockTopologicalWatcher(threat=1.0),
        MockLaplaceOracle(pole_real=-0.5),
        MockSheafOrchestrator(frustration=1.0),  # Muy alta
    )


@pytest.fixture
def quantum_gate(stable_mocks) -> QuantumAdmissionGate:
    """
    Fixture de QuantumAdmissionGate con configuración estable.
    
    Args:
        stable_mocks: Fixture de mocks estables.
        
    Returns:
        Instancia configurada de QuantumAdmissionGate.
    """
    topo, laplace, sheaf = stable_mocks
    return QuantumAdmissionGate(
        topo_watcher=topo,
        laplace_oracle=laplace,
        sheaf_orchestrator=sheaf,
    )


@pytest.fixture
def sample_payloads() -> Dict[str, Mapping[str, Any]]:
    """
    Fixture con payloads de referencia para tests.
    
    Returns:
        Diccionario con payloads canónicos:
            - empty: payload vacío
            - minimal: payload mínimo válido
            - simple: payload simple (baja entropía)
            - complex: payload complejo (alta entropía)
            - large: payload grande (muchos bytes)
    """
    return {
        'empty': {},
        
        'minimal': {
            'key': 'value'
        },
        
        'simple': {
            'endpoint': '/api/test',
            'method': 'GET',
            'data': 'A' * 100,  # Baja entropía
        },
        
        'complex': {
            'endpoint': '/api/data',
            'method': 'POST',
            'data': os.urandom(500).hex(),  # Alta entropía
            'metadata': {
                'timestamp': 1234567890,
                'user_id': 42,
            }
        },
        
        'large': {
            'data': 'X' * 10000,  # 10KB de datos
        }
    }


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2: TESTS DE UNIDAD - CONSTANTES FÍSICAS
# ═════════════════════════════════════════════════════════════════════════════

class TestPhysicalConstants:
    """
    Suite de validación de constantes físicas.
    
    Verifica que todas las constantes cumplan:
        1. No-negatividad (cuando corresponda)
        2. Finitud (representabilidad en float64)
        3. Coherencia dimensional
        4. Valores dentro de rangos físicos esperados
    """
    
    def test_unit_planck_constants_positive(self):
        """
        Verifica que constantes de Planck sean positivas.
        
        Invariantes:
            - PLANCK_H > 0
            - PLANCK_HBAR > 0
            - PLANCK_HBAR = PLANCK_H / (2π)
        """
        assert Const.PLANCK_H > 0, "h debe ser positiva"
        assert Const.PLANCK_HBAR > 0, "ℏ debe ser positiva"
        
        # Verificar relación h = 2π·ℏ
        expected_hbar = Const.PLANCK_H / (2.0 * math.pi)
        assert math.isclose(
            Const.PLANCK_HBAR,
            expected_hbar,
            rel_tol=1e-15
        ), f"ℏ debe ser h/(2π): esperado {expected_hbar}, obtenido {Const.PLANCK_HBAR}"
    
    def test_unit_potential_parameters_positive(self):
        """
        Verifica que parámetros del potencial sean positivos.
        
        Invariantes:
            - BASE_WORK_FUNCTION > 0
            - BASE_EFFECTIVE_MASS > 0
            - BARRIER_WIDTH > 0
            - ALPHA_THREAT > 0
        """
        assert Const.BASE_WORK_FUNCTION > 0
        assert Const.BASE_EFFECTIVE_MASS > 0
        assert Const.BARRIER_WIDTH > 0
        assert Const.ALPHA_THREAT > 0
    
    def test_unit_tolerances_positive_and_small(self):
        """
        Verifica que tolerancias numéricas sean pequeñas y positivas.
        
        Invariantes:
            - Todas las tolerancias > 0
            - Todas las tolerancias < 1
            - Tolerancias ordenadas razonablemente
        """
        tolerances = [
            Const.MIN_KINETIC_ENERGY,
            Const.FRUSTRATION_VETO_TOL,
            Const.SIGMA_CHAOS_TOL,
            Const.ENTROPY_FLOOR,
            Const.DIVISION_EPSILON,
        ]
        
        for tol in tolerances:
            assert tol > 0, f"Tolerancia debe ser positiva: {tol}"
            assert tol < 1, f"Tolerancia debe ser pequeña: {tol}"
    
    def test_unit_exp_cutoff_negative(self):
        """
        Verifica que el cutoff de exp() sea negativo y razonable.
        
        Invariantes:
            - EXP_UNDERFLOW_CUTOFF < 0
            - -750 < EXP_UNDERFLOW_CUTOFF < -650 (rango seguro)
        """
        assert Const.EXP_UNDERFLOW_CUTOFF < 0
        assert -750 < Const.EXP_UNDERFLOW_CUTOFF < -650
    
    def test_unit_wkb_threshold_reasonable(self):
        """
        Verifica que umbral WKB esté en rango razonable.
        
        Invariantes:
            - 0 < WKB_VALIDITY_THRESHOLD < 1
            - Típicamente ~ 0.1 para régimen semiclásico
        """
        assert 0 < Const.WKB_VALIDITY_THRESHOLD < 1
        assert math.isclose(Const.WKB_VALIDITY_THRESHOLD, 0.1, rel_tol=0.5)
    
    def test_unit_constants_class_sealed(self):
        """
        Verifica que la clase PhysicalConstants esté sellada (no heredable).
        
        Garantiza inmutabilidad semántica de las constantes.
        """
        with pytest.raises(TypeError, match="sellada"):
            class DerivedConstants(PhysicalConstants):
                pass


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: TESTS DE UNIDAD - ESTRUCTURAS DE DATOS
# ═════════════════════════════════════════════════════════════════════════════

class TestNumericInterval:
    """
    Suite de validación de NumericInterval.
    
    Verifica propiedades algebraicas:
        - Orden total en extremos
        - Métrica semi-definida positiva
        - Operaciones de pertenencia e intersección
    """
    
    def test_unit_interval_creation_valid(self):
        """
        Verifica creación exitosa de intervalos válidos.
        
        Casos:
            - Intervalo unitario [0, 1]
            - Intervalo simétrico [-1, 1]
            - Intervalo degenerado [x, x]
        """
        # Intervalo unitario
        iv = NumericInterval(0.0, 1.0)
        assert iv.lower == 0.0
        assert iv.upper == 1.0
        assert iv.width == 1.0
        assert iv.midpoint == 0.5
        
        # Intervalo simétrico
        iv2 = NumericInterval(-1.0, 1.0)
        assert iv2.midpoint == 0.0
        assert iv2.width == 2.0
        
        # Intervalo degenerado
        iv3 = NumericInterval(5.0, 5.0)
        assert iv3.width == 0.0
        assert iv3.midpoint == 5.0
    
    def test_unit_interval_rejects_nan(self):
        """
        Verifica que intervalos rechacen NaN.
        
        Invariante:
            NumericInterval no admite NaN en extremos.
        """
        with pytest.raises(ValueError, match="NaN"):
            NumericInterval(float('nan'), 1.0)
        
        with pytest.raises(ValueError, match="NaN"):
            NumericInterval(0.0, float('nan'))
    
    def test_unit_interval_rejects_inverted(self):
        """
        Verifica que intervalos rechacen lower > upper.
        
        Invariante:
            lower ≤ upper (orden total).
        """
        with pytest.raises(ValueError, match="mal formado"):
            NumericInterval(1.0, 0.0)
    
    def test_unit_interval_contains(self):
        """
        Verifica predicado de pertenencia.
        
        Casos:
            - Extremos incluidos
            - Interior incluido
            - Exterior excluido
        """
        iv = NumericInterval(0.0, 10.0)
        
        # Extremos
        assert iv.contains(0.0)
        assert iv.contains(10.0)
        
        # Interior
        assert iv.contains(5.0)
        
        # Exterior
        assert not iv.contains(-1.0)
        assert not iv.contains(11.0)
    
    def test_unit_interval_intersects(self):
        """
        Verifica detección de intersección.
        
        Casos:
            - Intersección no vacía
            - Intersección vacía
            - Contenido completo
        """
        iv1 = NumericInterval(0.0, 10.0)
        iv2 = NumericInterval(5.0, 15.0)
        iv3 = NumericInterval(20.0, 30.0)
        
        # Intersección no vacía
        assert iv1.intersects(iv2)
        assert iv2.intersects(iv1)
        
        # Intersección vacía
        assert not iv1.intersects(iv3)
        assert not iv3.intersects(iv1)
        
        # Contenido
        iv4 = NumericInterval(2.0, 8.0)
        assert iv1.intersects(iv4)
        assert iv4.intersects(iv1)
    
    def test_unit_interval_from_tolerance(self):
        """
        Verifica constructor con tolerancia.
        
        Invariante:
            Intervalo centrado en valor con radio |tolerance|.
        """
        iv = NumericInterval.from_value_with_tolerance(5.0, 0.1)
        
        assert iv.lower == 4.9
        assert iv.upper == 5.1
        assert iv.midpoint == 5.0
        assert iv.width == 0.2


class TestWKBParameters:
    """
    Suite de validación de WKBParameters.
    
    Verifica:
        - Validación de restricciones físicas
        - Cálculo del factor de Gamow
        - Verificación de régimen semiclásico
    """
    
    def test_unit_wkb_params_creation_valid(self):
        """
        Verifica creación exitosa con parámetros válidos.
        """
        params = WKBParameters(
            incident_energy=5.0,
            barrier_height=10.0,
            effective_mass=1.0,
            barrier_width=1.0,
            integrand=3.16,  # ≈ √(2·1·5)
            exponent=-10.0,
            validity_parameter=0.05
        )
        
        assert params.incident_energy == 5.0
        assert params.barrier_height == 10.0
        assert params.effective_mass == 1.0
    
    def test_unit_wkb_params_rejects_negative_energy(self):
        """
        Verifica rechazo de energía incidente negativa.
        """
        with pytest.raises(QuantumNumericalError, match="negativa"):
            WKBParameters(
                incident_energy=-1.0,
                barrier_height=10.0,
                effective_mass=1.0,
                barrier_width=1.0,
                integrand=0.0,
                exponent=0.0,
                validity_parameter=0.0
            )
    
    def test_unit_wkb_params_rejects_negative_mass(self):
        """
        Verifica rechazo de masa efectiva negativa.
        """
        with pytest.raises(QuantumNumericalError, match="positiva"):
            WKBParameters(
                incident_energy=5.0,
                barrier_height=10.0,
                effective_mass=-1.0,
                barrier_width=1.0,
                integrand=0.0,
                exponent=0.0,
                validity_parameter=0.0
            )
    
    def test_unit_wkb_params_accepts_infinite_mass(self):
        """
        Verifica aceptación de masa infinita (barrera impenetrable).
        """
        params = WKBParameters(
            incident_energy=5.0,
            barrier_height=10.0,
            effective_mass=float('inf'),
            barrier_width=1.0,
            integrand=float('inf'),
            exponent=float('-inf'),
            validity_parameter=float('inf')
        )
        
        assert math.isinf(params.effective_mass)
    
    def test_unit_wkb_params_semiclassical_validity(self):
        """
        Verifica detección de régimen semiclásico válido.
        
        Condiciones:
            - barrier_height > 0
            - validity_parameter < threshold
            - 0 < m_eff < ∞
        """
        # Válido
        params_valid = WKBParameters(
            incident_energy=5.0,
            barrier_height=10.0,
            effective_mass=1.0,
            barrier_width=1.0,
            integrand=3.0,
            exponent=-5.0,
            validity_parameter=0.05
        )
        assert params_valid.is_valid_semiclassical_regime()
        
        # Inválido: no túnel (barrier_height = 0)
        params_classical = WKBParameters(
            incident_energy=15.0,
            barrier_height=0.0,
            effective_mass=1.0,
            barrier_width=1.0,
            integrand=0.0,
            exponent=0.0,
            validity_parameter=0.0
        )
        assert not params_classical.is_valid_semiclassical_regime()
        
        # Inválido: validity_parameter alto
        params_invalid = WKBParameters(
            incident_energy=5.0,
            barrier_height=10.0,
            effective_mass=1.0,
            barrier_width=1.0,
            integrand=3.0,
            exponent=-5.0,
            validity_parameter=0.5  # > threshold
        )
        assert not params_invalid.is_valid_semiclassical_regime()
    
    def test_unit_wkb_params_gamow_factor(self):
        """
        Verifica cálculo del factor de Gamow γ = |exponent|/2.
        """
        params = WKBParameters(
            incident_energy=5.0,
            barrier_height=10.0,
            effective_mass=1.0,
            barrier_width=1.0,
            integrand=3.0,
            exponent=-10.0,
            validity_parameter=0.05
        )
        
        assert params.gamow_factor() == 5.0


class TestQuantumMeasurement:
    """
    Suite de validación de QuantumMeasurement.
    
    Verifica:
        - Inmutabilidad (frozen)
        - Validación de invariantes físicos
        - Consistencia estado-observables
        - Serialización
    """
    
    def test_unit_measurement_creation_admitted(self):
        """
        Verifica creación exitosa de medición admitida.
        """
        measurement = QuantumMeasurement(
            eigenstate=Eigenstate.ADMITIDO,
            incident_energy=10.0,
            work_function=5.0,
            tunneling_probability=0.8,
            kinetic_energy=5.0,
            momentum=3.16,
            frustration_veto=False,
            effective_mass=1.0,
            dominant_pole_real=-0.5,
            threat_level=2.0,
            collapse_threshold=0.7,
            admission_reason="Test admission"
        )
        
        assert measurement.eigenstate == Eigenstate.ADMITIDO
        assert measurement.kinetic_energy == 5.0
        assert measurement.momentum == 3.16
    
    def test_unit_measurement_creation_rejected(self):
        """
        Verifica creación exitosa de medición rechazada.
        
        Invariante:
            Si RECHAZADO, entonces K = p = 0.
        """
        measurement = QuantumMeasurement(
            eigenstate=Eigenstate.RECHAZADO,
            incident_energy=10.0,
            work_function=5.0,
            tunneling_probability=0.2,
            kinetic_energy=0.0,
            momentum=0.0,
            frustration_veto=False,
            effective_mass=1.0,
            dominant_pole_real=-0.5,
            threat_level=2.0,
            collapse_threshold=0.3,
            admission_reason="Test rejection"
        )
        
        assert measurement.eigenstate == Eigenstate.RECHAZADO
        assert measurement.kinetic_energy == 0.0
        assert measurement.momentum == 0.0
    
    def test_unit_measurement_rejects_negative_energy(self):
        """
        Verifica rechazo de energías negativas.
        """
        with pytest.raises(QuantumStateError, match="negativa"):
            QuantumMeasurement(
                eigenstate=Eigenstate.ADMITIDO,
                incident_energy=-1.0,
                work_function=5.0,
                tunneling_probability=0.8,
                kinetic_energy=5.0,
                momentum=3.0,
                frustration_veto=False,
                effective_mass=1.0,
                dominant_pole_real=-0.5,
                threat_level=2.0,
                collapse_threshold=0.7,
                admission_reason="Test"
            )
    
    def test_unit_measurement_rejects_invalid_probability(self):
        """
        Verifica rechazo de probabilidades fuera de [0, 1].
        """
        with pytest.raises(QuantumStateError, match="fuera de"):
            QuantumMeasurement(
                eigenstate=Eigenstate.ADMITIDO,
                incident_energy=10.0,
                work_function=5.0,
                tunneling_probability=1.5,  # > 1
                kinetic_energy=5.0,
                momentum=3.0,
                frustration_veto=False,
                effective_mass=1.0,
                dominant_pole_real=-0.5,
                threat_level=2.0,
                collapse_threshold=0.7,
                admission_reason="Test"
            )
    
    def test_unit_measurement_rejects_inconsistent_rejected_state(self):
        """
        Verifica rechazo de estado rechazado con K o p no nulos.
        
        Invariante:
            RECHAZADO ⟹ K = 0 ∧ p = 0
        """
        with pytest.raises(QuantumStateError, match="rechazado"):
            QuantumMeasurement(
                eigenstate=Eigenstate.RECHAZADO,
                incident_energy=10.0,
                work_function=5.0,
                tunneling_probability=0.2,
                kinetic_energy=1.0,  # ≠ 0
                momentum=0.0,
                frustration_veto=False,
                effective_mass=1.0,
                dominant_pole_real=-0.5,
                threat_level=2.0,
                collapse_threshold=0.3,
                admission_reason="Test"
            )
    
    def test_unit_measurement_immutability(self):
        """
        Verifica que QuantumMeasurement sea inmutable (frozen).
        """
        measurement = QuantumMeasurement(
            eigenstate=Eigenstate.ADMITIDO,
            incident_energy=10.0,
            work_function=5.0,
            tunneling_probability=0.8,
            kinetic_energy=5.0,
            momentum=3.0,
            frustration_veto=False,
            effective_mass=1.0,
            dominant_pole_real=-0.5,
            threat_level=2.0,
            collapse_threshold=0.7,
            admission_reason="Test"
        )
        
        with pytest.raises(AttributeError):
            measurement.eigenstate = Eigenstate.RECHAZADO
    
    def test_unit_measurement_to_dict(self):
        """
        Verifica serialización a diccionario.
        """
        measurement = QuantumMeasurement(
            eigenstate=Eigenstate.ADMITIDO,
            incident_energy=10.0,
            work_function=5.0,
            tunneling_probability=0.8,
            kinetic_energy=5.0,
            momentum=3.0,
            frustration_veto=False,
            effective_mass=1.0,
            dominant_pole_real=-0.5,
            threat_level=2.0,
            collapse_threshold=0.7,
            admission_reason="Test"
        )
        
        result_dict = measurement.to_dict()
        
        assert result_dict['eigenstate'] == 'ADMITIDO'
        assert result_dict['incident_energy'] == 10.0
        assert result_dict['momentum'] == 3.0
        assert 'admission_reason' in result_dict
    
    def test_unit_measurement_de_broglie_wavelength(self):
        """
        Verifica cálculo de longitud de onda de De Broglie: λ = h/p.
        """
        measurement = QuantumMeasurement(
            eigenstate=Eigenstate.ADMITIDO,
            incident_energy=10.0,
            work_function=5.0,
            tunneling_probability=0.8,
            kinetic_energy=5.0,
            momentum=2.0,
            frustration_veto=False,
            effective_mass=1.0,
            dominant_pole_real=-0.5,
            threat_level=2.0,
            collapse_threshold=0.7,
            admission_reason="Test"
        )
        
        wavelength = measurement.de_broglie_wavelength()
        expected = Const.PLANCK_H / 2.0
        
        assert math.isclose(wavelength, expected, rel_tol=1e-9)
    
    def test_unit_measurement_de_broglie_wavelength_zero_momentum(self):
        """
        Verifica que λ = ∞ cuando p = 0.
        """
        measurement = QuantumMeasurement(
            eigenstate=Eigenstate.RECHAZADO,
            incident_energy=10.0,
            work_function=5.0,
            tunneling_probability=0.2,
            kinetic_energy=0.0,
            momentum=0.0,
            frustration_veto=False,
            effective_mass=1.0,
            dominant_pole_real=-0.5,
            threat_level=2.0,
            collapse_threshold=0.3,
            admission_reason="Test"
        )
        
        wavelength = measurement.de_broglie_wavelength()
        assert math.isinf(wavelength)


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4: TESTS DE UNIDAD - MORFISMOS NUMÉRICOS
# ═════════════════════════════════════════════════════════════════════════════

class TestNumericalMorphisms:
    """
    Suite de validación de morfismos numéricos.
    
    Verifica comportamiento robusto ante:
        - Valores normales
        - Casos especiales (0, ±∞, NaN)
        - Casos límite
    """
    
    def test_unit_validate_finite_float_valid(self):
        """
        Verifica conversión exitosa de valores válidos.
        """
        assert NM.validate_finite_float(5.0, name="test") == 5.0
        assert NM.validate_finite_float(5, name="test") == 5.0
        assert NM.validate_finite_float("5.5", name="test") == 5.5
    
    def test_unit_validate_finite_float_rejects_nan(self):
        """
        Verifica rechazo de NaN.
        """
        with pytest.raises(QuantumNumericalError, match="NaN"):
            NM.validate_finite_float(float('nan'), name="test")
    
    def test_unit_validate_finite_float_rejects_inf_by_default(self):
        """
        Verifica rechazo de infinito por defecto.
        """
        with pytest.raises(QuantumNumericalError, match="infinito"):
            NM.validate_finite_float(float('inf'), name="test")
    
    def test_unit_validate_finite_float_accepts_inf_when_allowed(self):
        """
        Verifica aceptación de infinito cuando allow_inf=True.
        """
        result = NM.validate_finite_float(
            float('inf'),
            name="test",
            allow_inf=True
        )
        assert math.isinf(result)
    
    def test_unit_clamp_to_unit_interval_normal_values(self):
        """
        Verifica clamp correcto de valores normales.
        """
        assert NM.clamp_to_unit_interval(0.5) == 0.5
        assert NM.clamp_to_unit_interval(-0.5) == 0.0
        assert NM.clamp_to_unit_interval(1.5) == 1.0
        assert NM.clamp_to_unit_interval(0.0) == 0.0
        assert NM.clamp_to_unit_interval(1.0) == 1.0
    
    def test_unit_clamp_to_unit_interval_special_values(self):
        """
        Verifica manejo de valores especiales.
        """
        assert NM.clamp_to_unit_interval(float('inf')) == 1.0
        assert NM.clamp_to_unit_interval(float('-inf')) == 0.0
        assert NM.clamp_to_unit_interval(float('nan')) == 0.0
    
    def test_unit_safe_division_normal(self):
        """
        Verifica división normal sin singularidades.
        """
        assert NM.safe_division(10.0, 2.0) == 5.0
        assert NM.safe_division(1.0, 3.0) == pytest.approx(1.0/3.0)
    
    def test_unit_safe_division_by_zero_exact(self):
        """
        Verifica fallback cuando denominador es exactamente 0.
        """
        result = NM.safe_division(10.0, 0.0, fallback=999.0)
        assert result == 999.0
    
    def test_unit_safe_division_regularization(self):
        """
        Verifica regularización para denominadores muy pequeños.
        """
        # Denominador muy pequeño → regularizado a epsilon
        result = NM.safe_division(
            1.0,
            1e-20,
            epsilon=1e-15
        )
        
        # Debe ser finito (no overflow)
        assert math.isfinite(result)
        assert result > 0
    
    def test_unit_safe_sqrt_positive(self):
        """
        Verifica raíz cuadrada de valores positivos.
        """
        assert NM.safe_sqrt(4.0) == 2.0
        assert NM.safe_sqrt(0.0) == 0.0
    
    def test_unit_safe_sqrt_negative_clamped(self):
        """
        Verifica clamp de valores negativos a 0.
        """
        assert NM.safe_sqrt(-4.0) == 0.0
    
    def test_unit_safe_exp_normal(self):
        """
        Verifica exponencial de valores normales.
        """
        assert math.isclose(NM.safe_exp(0.0), 1.0, rel_tol=1e-9)
        assert math.isclose(NM.safe_exp(1.0), math.e, rel_tol=1e-9)
    
    def test_unit_safe_exp_underflow(self):
        """
        Verifica saturación a 0 en underflow.
        """
        result = NM.safe_exp(-800.0)
        assert result == 0.0
    
    def test_unit_safe_exp_overflow(self):
        """
        Verifica saturación a FLOAT_MAX en overflow.
        """
        result = NM.safe_exp(800.0)
        assert result == Const.FLOAT_MAX
    
    def test_unit_safe_log_positive(self):
        """
        Verifica logaritmo de valores positivos.
        """
        assert math.isclose(NM.safe_log(math.e), 1.0, rel_tol=1e-9)
        assert math.isclose(NM.safe_log(1.0), 0.0, abs_tol=1e-9)
    
    def test_unit_safe_log_with_floor(self):
        """
        Verifica piso en logaritmo para evitar -∞.
        """
        result = NM.safe_log(0.0, floor=1e-100)
        assert math.isfinite(result)
        assert result < 0  # ln(x) < 0 para 0 < x < 1


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5: TESTS DE UNIDAD - CALCULADORES
# ═════════════════════════════════════════════════════════════════════════════

class TestEntropyCalculator:
    """
    Suite de validación del calculador de entropía de Shannon.
    
    Verifica propiedades teóricas:
        - H ≥ 0 (no-negatividad)
        - H = 0 ⟺ secuencia constante
        - H ≤ ln(256) para bytes
        - Caché LRU funciona correctamente
    """
    
    def test_unit_entropy_empty_bytes(self):
        """
        Verifica que H = 0 para secuencia vacía.
        """
        entropy = EntropyCalculator.shannon_entropy_bytes(b'')
        assert entropy == 0.0
    
    def test_unit_entropy_constant_sequence(self):
        """
        Verifica que H = 0 para secuencia constante.
        """
        constant_data = b'A' * 100
        entropy = EntropyCalculator.shannon_entropy_bytes(constant_data)
        
        assert entropy == 0.0
    
    def test_unit_entropy_uniform_distribution(self):
        """
        Verifica que H ≈ ln(256) para distribución uniforme.
        """
        # Crear secuencia con todos los bytes posibles
        uniform_data = bytes(range(256)) * 10
        entropy = EntropyCalculator.shannon_entropy_bytes(uniform_data)
        
        # Debe estar muy cerca de ln(256)
        assert math.isclose(
            entropy,
            Const.MAX_SHANNON_ENTROPY,
            rel_tol=0.01
        )
    
    def test_unit_entropy_non_negativity(self):
        """
        Verifica no-negatividad para datos arbitrarios.
        """
        import random
        
        for _ in range(10):
            data = bytes(random.randint(0, 255) for _ in range(100))
            entropy = EntropyCalculator.shannon_entropy_bytes(data)
            
            assert entropy >= 0.0
            assert entropy <= Const.MAX_SHANNON_ENTROPY
    
    def test_unit_entropy_normalized(self):
        """
        Verifica entropía normalizada en [0, 1].
        """
        # Constante → 0
        norm_const = EntropyCalculator.normalized_entropy(b'A' * 100)
        assert norm_const == 0.0
        
        # Uniforme → ≈ 1
        uniform_data = bytes(range(256)) * 10
        norm_uniform = EntropyCalculator.normalized_entropy(uniform_data)
        assert 0.95 <= norm_uniform <= 1.0
    
    def test_unit_entropy_conditional(self):
        """
        Verifica cálculo de entropía condicional.
        """
        data = b'AAABBBCCC'
        
        # Condición: byte es 'A'
        h_cond = EntropyCalculator.conditional_entropy(
            data,
            condition=lambda b: b == ord('A')
        )
        
        # Debe ser finita y no negativa
        assert math.isfinite(h_cond)
        assert h_cond >= 0.0


class TestPayloadSerializer:
    """
    Suite de validación del serializador de payloads.
    
    Verifica:
        - Determinismo
        - Independencia de orden
        - Hash criptográfico
    """
    
    def test_unit_serializer_deterministic(self):
        """
        Verifica que serialización sea determinista.
        """
        payload = {'key1': 'value1', 'key2': 42}
        
        bytes1 = PayloadSerializer.serialize(payload)
        bytes2 = PayloadSerializer.serialize(payload)
        
        assert bytes1 == bytes2
    
    def test_unit_serializer_order_independent(self):
        """
        Verifica que el orden de inserción no afecte serialización.
        """
        payload1 = {'a': 1, 'b': 2, 'c': 3}
        payload2 = {'c': 3, 'a': 1, 'b': 2}
        
        bytes1 = PayloadSerializer.serialize(payload1)
        bytes2 = PayloadSerializer.serialize(payload2)
        
        assert bytes1 == bytes2
    
    def test_unit_serializer_empty_payload(self):
        """
        Verifica serialización de payload vacío.
        """
        empty = {}
        result = PayloadSerializer.serialize(empty)
        
        assert isinstance(result, bytes)
        assert len(result) > 0
    
    def test_unit_serializer_rejects_non_mapping(self):
        """
        Verifica rechazo de no-Mappings.
        """
        with pytest.raises(QuantumAdmissionError, match="Mapping"):
            PayloadSerializer.serialize([1, 2, 3])
    
    def test_unit_hash_deterministic(self):
        """
        Verifica que hash sea determinista.
        """
        data = b'test data'
        
        hash1 = PayloadSerializer.deterministic_hash(data)
        hash2 = PayloadSerializer.deterministic_hash(data)
        
        assert hash1 == hash2
        assert len(hash1) == 32  # SHA-256 = 32 bytes
    
    def test_unit_hash_different_for_different_data(self):
        """
        Verifica que datos distintos produzcan hashes distintos.
        """
        hash1 = PayloadSerializer.deterministic_hash(b'data1')
        hash2 = PayloadSerializer.deterministic_hash(b'data2')
        
        assert hash1 != hash2
    
    def test_unit_hash_to_unit_interval(self):
        """
        Verifica proyección de hash a [0, 1).
        """
        for _ in range(100):
            data = os.urandom(32)
            theta = PayloadSerializer.hash_to_unit_interval(data)
            
            assert 0.0 <= theta < 1.0


class TestIncidentEnergyCalculator:
    """
    Suite de validación del calculador de energía incidente.
    
    Verifica:
        - E ≥ 0
        - E = 0 para payload vacío
        - E crece con tamaño
        - Propagación de incertidumbre
    """
    
    def test_unit_incident_energy_empty_payload(self):
        """
        Verifica que E = 0 para payload vacío.
        """
        E, uncertainty = IncidentEnergyCalculator.calculate({})
        
        assert E == 0.0
        assert uncertainty.lower == 0.0
        assert uncertainty.upper == 0.0
    
    def test_unit_incident_energy_positive(self):
        """
        Verifica que E > 0 para payload no vacío.
        """
        payload = {'key': 'value'}
        E, uncertainty = IncidentEnergyCalculator.calculate(payload)
        
        assert E > 0.0
        assert math.isfinite(E)
    
    def test_unit_incident_energy_grows_with_size(self):
        """
        Verifica que E crece con tamaño del payload (entropía fija).
        """
        payload_small = {'data': 'A' * 10}
        payload_large = {'data': 'A' * 100}
        
        E_small, _ = IncidentEnergyCalculator.calculate(payload_small)
        E_large, _ = IncidentEnergyCalculator.calculate(payload_large)
        
        assert E_large > E_small
    
    def test_unit_incident_energy_decreases_with_entropy(self):
        """
        Verifica que E decrece con mayor entropía (tamaño fijo).
        """
        # Baja entropía (constante)
        payload_low_entropy = {'data': 'A' * 100}
        
        # Alta entropía (aleatorio)
        payload_high_entropy = {'data': os.urandom(100).hex()}
        
        E_low, _ = IncidentEnergyCalculator.calculate(payload_low_entropy)
        E_high, _ = IncidentEnergyCalculator.calculate(payload_high_entropy)
        
        # Con baja entropía, la energía debe ser mayor
        assert E_low > E_high
    
    def test_unit_incident_energy_uncertainty_interval(self):
        """
        Verifica que intervalo de incertidumbre contenga E.
        """
        payload = {'key': 'value' * 10}
        E, uncertainty = IncidentEnergyCalculator.calculate(payload)
        
        assert uncertainty.contains(E)
        assert uncertainty.width > 0


class TestWKBCalculator:
    """
    Suite de validación del calculador WKB.
    
    Verifica casos:
        - Transmisión clásica (E ≥ Φ)
        - Túnel cuántico (E < Φ)
        - Masa infinita
        - Validez semiclásica
    """
    
    def test_unit_wkb_classical_transmission(self):
        """
        Verifica que T = 1 cuando E ≥ Φ (transmisión clásica).
        """
        T, params = WKBCalculator.compute_tunneling_probability(
            E=10.0,
            Phi=5.0,
            m_eff=1.0
        )
        
        assert T == 1.0
        assert params.barrier_height == 0.0
    
    def test_unit_wkb_tunneling_regime(self):
        """
        Verifica que 0 < T < 1 en régimen túnel (E < Φ).
        """
        T, params = WKBCalculator.compute_tunneling_probability(
            E=5.0,
            Phi=10.0,
            m_eff=1.0
        )
        
        assert 0.0 < T < 1.0
        assert params.barrier_height == 5.0
    
    def test_unit_wkb_infinite_mass_impenetrable(self):
        """
        Verifica que T = 0 cuando m_eff = ∞ (barrera impenetrable).
        """
        T, params = WKBCalculator.compute_tunneling_probability(
            E=5.0,
            Phi=10.0,
            m_eff=float('inf')
        )
        
        assert T == 0.0
        assert math.isinf(params.effective_mass)
    
    def test_unit_wkb_rejects_negative_energy(self):
        """
        Verifica rechazo de energías negativas.
        """
        with pytest.raises(QuantumNumericalError, match="negativas"):
            WKBCalculator.compute_tunneling_probability(
                E=-1.0,
                Phi=10.0,
                m_eff=1.0
            )
    
    def test_unit_wkb_rejects_negative_work_function(self):
        """
        Verifica rechazo de función de trabajo negativa.
        """
        with pytest.raises(QuantumNumericalError, match="negativas"):
            WKBCalculator.compute_tunneling_probability(
                E=5.0,
                Phi=-1.0,
                m_eff=1.0
            )
    
    def test_unit_wkb_rejects_negative_mass(self):
        """
        Verifica rechazo de masa efectiva negativa.
        """
        with pytest.raises(QuantumNumericalError, match="positiva"):
            WKBCalculator.compute_tunneling_probability(
                E=5.0,
                Phi=10.0,
                m_eff=-1.0
            )
    
    def test_unit_wkb_exponential_suppression(self):
        """
        Verifica supresión exponencial con barrera alta.
        """
        T_low, _ = WKBCalculator.compute_tunneling_probability(
            E=1.0, Phi=5.0, m_eff=1.0
        )
        
        T_high, _ = WKBCalculator.compute_tunneling_probability(
            E=1.0, Phi=50.0, m_eff=1.0
        )
        
        # Con barrera más alta, T debe ser mucho menor
        assert T_high < T_low
        assert T_high < 0.01


class TestWorkFunctionModulator:
    """
    Suite de validación del modulador de función de trabajo.
    
    Verifica:
        - Φ ≥ Φ₀ (monotonía)
        - Φ crece con χ²
        - Manejo de fallos del observador
    """
    
    def test_unit_work_function_nominal_threat(self):
        """
        Verifica cálculo con amenaza nominal.
        """
        watcher = MockTopologicalWatcher(threat=1.0)
        modulator = WorkFunctionModulator(watcher)
        
        Phi, threat = modulator.calculate()
        
        assert Phi >= Const.BASE_WORK_FUNCTION
        assert threat == 1.0
    
    def test_unit_work_function_zero_threat(self):
        """
        Verifica que Φ = Φ₀ cuando χ² = 0.
        """
        watcher = MockTopologicalWatcher(threat=0.0)
        modulator = WorkFunctionModulator(watcher)
        
        Phi, threat = modulator.calculate()
        
        assert math.isclose(
            Phi,
            Const.BASE_WORK_FUNCTION,
            rel_tol=1e-9
        )
        assert threat == 0.0
    
    def test_unit_work_function_grows_with_threat(self):
        """
        Verifica que Φ crece con χ².
        """
        watcher_low = MockTopologicalWatcher(threat=1.0)
        watcher_high = MockTopologicalWatcher(threat=5.0)
        
        modulator_low = WorkFunctionModulator(watcher_low)
        modulator_high = WorkFunctionModulator(watcher_high)
        
        Phi_low, _ = modulator_low.calculate()
        Phi_high, _ = modulator_high.calculate()
        
        assert Phi_high > Phi_low
    
    def test_unit_work_function_handles_watcher_failure(self):
        """
        Verifica manejo defensivo de fallos del observador.
        """
        watcher = MockTopologicalWatcher(threat=1.0, should_fail=True)
        modulator = WorkFunctionModulator(watcher)
        
        # No debe lanzar excepción, debe asumir χ² = 0
        Phi, threat = modulator.calculate()
        
        assert Phi >= 0.0
        assert threat == 0.0


class TestEffectiveMassModulator:
    """
    Suite de validación del modulador de masa efectiva.
    
    Verifica:
        - m_eff finita cuando σ < 0 (estable)
        - m_eff = ∞ cuando σ ≥ 0 (inestable)
        - m_eff decrece con |σ| creciente
    """
    
    def test_unit_effective_mass_stable_system(self):
        """
        Verifica masa finita para sistema estable (σ < 0).
        """
        oracle = MockLaplaceOracle(pole_real=-1.0)
        modulator = EffectiveMassModulator(oracle)
        
        m_eff, sigma = modulator.calculate()
        
        assert math.isfinite(m_eff)
        assert m_eff > 0
        assert sigma == -1.0
    
    def test_unit_effective_mass_unstable_system(self):
        """
        Verifica masa infinita para sistema inestable (σ ≥ 0).
        """
        oracle = MockLaplaceOracle(pole_real=0.5)
        modulator = EffectiveMassModulator(oracle)
        
        m_eff, sigma = modulator.calculate()
        
        assert math.isinf(m_eff)
        assert sigma == 0.5
    
    def test_unit_effective_mass_marginal_stability(self):
        """
        Verifica masa infinita en estabilidad marginal (σ ≈ 0).
        """
        oracle = MockLaplaceOracle(pole_real=-1e-12)
        modulator = EffectiveMassModulator(oracle)
        
        m_eff, sigma = modulator.calculate()
        
        # Dentro de tolerancia → considerado inestable
        assert math.isinf(m_eff)
    
    def test_unit_effective_mass_decreases_with_stability(self):
        """
        Verifica que m_eff decrece cuando sistema se vuelve más estable.
        """
        oracle_less_stable = MockLaplaceOracle(pole_real=-0.5)
        oracle_more_stable = MockLaplaceOracle(pole_real=-2.0)
        
        modulator_less = EffectiveMassModulator(oracle_less_stable)
        modulator_more = EffectiveMassModulator(oracle_more_stable)
        
        m_less, _ = modulator_less.calculate()
        m_more, _ = modulator_more.calculate()
        
        # Sistema más estable (|σ| mayor) → masa menor
        assert m_more < m_less
    
    def test_unit_effective_mass_handles_oracle_failure(self):
        """
        Verifica manejo defensivo de fallos del oráculo.
        """
        oracle = MockLaplaceOracle(pole_real=-1.0, should_fail=True)
        modulator = EffectiveMassModulator(oracle)
        
        # No debe lanzar excepción, debe asumir σ = 0 (inestable)
        m_eff, sigma = modulator.calculate()
        
        assert math.isinf(m_eff)
        assert sigma == 0.0


class TestCollapseThresholdGenerator:
    """
    Suite de validación del generador de umbral de colapso.
    
    Verifica:
        - θ ∈ [0, 1)
        - Determinismo
        - Distribución aproximadamente uniforme
    """
    
    def test_unit_threshold_in_unit_interval(self):
        """
        Verifica que θ ∈ [0, 1).
        """
        for _ in range(100):
            hash_bytes = os.urandom(32)
            theta = CollapseThresholdGenerator.generate(hash_bytes)
            
            assert 0.0 <= theta < 1.0
    
    def test_unit_threshold_deterministic(self):
        """
        Verifica que mismo hash → mismo θ.
        """
        hash_bytes = os.urandom(32)
        
        theta1 = CollapseThresholdGenerator.generate(hash_bytes)
        theta2 = CollapseThresholdGenerator.generate(hash_bytes)
        
        assert theta1 == theta2
    
    def test_unit_threshold_different_hashes_different_thresholds(self):
        """
        Verifica que hashes distintos produzcan θ distintos.
        """
        hash1 = os.urandom(32)
        hash2 = os.urandom(32)
        
        theta1 = CollapseThresholdGenerator.generate(hash1)
        theta2 = CollapseThresholdGenerator.generate(hash2)
        
        # Altamente probable que sean distintos
        assert theta1 != theta2
    
    def test_unit_threshold_approximately_uniform(self):
        """
        Verifica distribución aproximadamente uniforme en [0, 1).
        
        Test estadístico: partición en 10 bins, cada uno debe tener ~10%.
        """
        n_samples = 10000
        bins = [0] * 10
        
        for _ in range(n_samples):
            hash_bytes = os.urandom(32)
            theta = CollapseThresholdGenerator.generate(hash_bytes)
            
            bin_index = int(theta * 10)
            if bin_index == 10:  # θ muy cerca de 1.0
                bin_index = 9
            
            bins[bin_index] += 1
        
        # Cada bin debe tener aproximadamente n_samples/10 = 1000
        expected = n_samples / 10
        
        for count in bins:
            # Tolerancia del 20% (test estadístico robusto)
            assert 0.8 * expected <= count <= 1.2 * expected


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6: TESTS DE INTEGRACIÓN
# ═════════════════════════════════════════════════════════════════════════════

class TestQuantumAdmissionGateIntegration:
    """
    Suite de integración del Quantum Admission Gate.
    
    Verifica:
        - Pipeline completo de admisión
        - Interacción entre componentes
        - Casos de admisión/rechazo
        - Veto cohomológico
    """
    
    def test_integration_gate_construction_validates_dependencies(self):
        """
        Verifica que el constructor valide dependencias rigurosamente.
        """
        topo = MockTopologicalWatcher()
        laplace = MockLaplaceOracle()
        sheaf = MockSheafOrchestrator()
        
        # Construcción exitosa
        gate = QuantumAdmissionGate(
            topo_watcher=topo,
            laplace_oracle=laplace,
            sheaf_orchestrator=sheaf
        )
        
        assert gate is not None
    
    def test_integration_gate_rejects_none_dependencies(self):
        """
        Verifica rechazo de dependencias None.
        """
        with pytest.raises(QuantumInterfaceError, match="None"):
            QuantumAdmissionGate(
                topo_watcher=None,
                laplace_oracle=MockLaplaceOracle(),
                sheaf_orchestrator=MockSheafOrchestrator()
            )
    
    def test_integration_gate_rejects_invalid_protocol(self):
        """
        Verifica rechazo de objetos que no cumplen protocolos.
        """
        class InvalidWatcher:
            pass
        
        with pytest.raises(QuantumInterfaceError, match="protocolo"):
            QuantumAdmissionGate(
                topo_watcher=InvalidWatcher(),
                laplace_oracle=MockLaplaceOracle(),
                sheaf_orchestrator=MockSheafOrchestrator()
            )
    
    def test_integration_simple_payload_admission(self, quantum_gate, sample_payloads):
        """
        Verifica admisión de payload simple con configuración estable.
        """
        payload = sample_payloads['simple']
        measurement = quantum_gate.evaluate_admission(payload)
        
        # Verificar estructura básica
        assert isinstance(measurement, QuantumMeasurement)
        assert measurement.eigenstate in [Eigenstate.ADMITIDO, Eigenstate.RECHAZADO]
        
        # Verificar invariantes físicos
        assert measurement.incident_energy >= 0.0
        assert measurement.work_function >= 0.0
        assert 0.0 <= measurement.tunneling_probability <= 1.0
        assert 0.0 <= measurement.collapse_threshold < 1.0
    
    def test_integration_empty_payload(self, quantum_gate, sample_payloads):
        """
        Verifica manejo de payload vacío.
        """
        payload = sample_payloads['empty']
        measurement = quantum_gate.evaluate_admission(payload)
        
        # Payload vacío → E = 0 → debería ser rechazado
        assert measurement.eigenstate == Eigenstate.RECHAZADO
        assert measurement.incident_energy == 0.0
        assert measurement.kinetic_energy == 0.0
        assert measurement.momentum == 0.0
    
    def test_integration_veto_cohomological(self, veto_mocks, sample_payloads):
        """
        Verifica veto cohomológico con frustración alta.
        """
        topo, laplace, sheaf = veto_mocks
        gate = QuantumAdmissionGate(
            topo_watcher=topo,
            laplace_oracle=laplace,
            sheaf_orchestrator=sheaf
        )
        
        payload = sample_payloads['simple']
        measurement = gate.evaluate_admission(payload)
        
        # Debe ser rechazado por veto
        assert measurement.eigenstate == Eigenstate.RECHAZADO
        assert measurement.frustration_veto is True
        assert "VETO" in measurement.admission_reason.upper()
    
    def test_integration_unstable_system_suppresses_tunneling(self, unstable_mocks, sample_payloads):
        """
        Verifica que sistema inestable suprima túnel cuántico.
        """
        topo, laplace, sheaf = unstable_mocks
        gate = QuantumAdmissionGate(
            topo_watcher=topo,
            laplace_oracle=laplace,
            sheaf_orchestrator=sheaf
        )
        
        payload = sample_payloads['simple']
        measurement = gate.evaluate_admission(payload)
        
        # Sistema inestable → m_eff = ∞ → T ≈ 0
        assert math.isinf(measurement.effective_mass)
        
        # Probabilidad de túnel debe ser muy baja o cero
        assert measurement.tunneling_probability < 0.01
    
    def test_integration_reproducibility(self, quantum_gate, sample_payloads):
        """
        Verifica reproducibilidad determinista del resultado.
        """
        payload = sample_payloads['simple']
        
        m1 = quantum_gate.evaluate_admission(payload)
        m2 = quantum_gate.evaluate_admission(payload)
        
        # Mismos observables
        assert m1.eigenstate == m2.eigenstate
        assert m1.incident_energy == m2.incident_energy
        assert m1.work_function == m2.work_function
        assert m1.tunneling_probability == m2.tunneling_probability
        assert m1.collapse_threshold == m2.collapse_threshold
        assert m1.momentum == m2.momentum
    
    def test_integration_large_payload(self, quantum_gate, sample_payloads):
        """
        Verifica manejo de payload grande.
        """
        payload = sample_payloads['large']
        measurement = quantum_gate.evaluate_admission(payload)
        
        # Payload grande → energía alta → mayor probabilidad de admisión
        assert measurement.incident_energy > 0.0
        assert isinstance(measurement, QuantumMeasurement)
    
    def test_integration_categorical_state_morphism(self, quantum_gate, sample_payloads):
        """
        Verifica que el gate funcione como morfismo categórico.
        """
        payload = sample_payloads['simple']
        
        # Crear estado categórico inicial
        initial_state = CategoricalState(
            payload=payload,
            context={},
            validated_strata=frozenset()
        )
        
        # Aplicar morfismo
        final_state = quantum_gate(initial_state)
        
        # Verificar que retorna CategoricalState
        assert isinstance(final_state, CategoricalState)
        
        # Verificar que payload se preserva
        assert final_state.payload == payload
        
        # Verificar que contexto contiene medición
        assert 'quantum_admission' in final_state.context
        
        measurement = final_state.context['quantum_admission']
        
        if measurement.eigenstate == Eigenstate.ADMITIDO:
            # Si admitido, PHYSICS debe estar en strata
            assert Stratum.PHYSICS in final_state.validated_strata
            assert 'quantum_momentum' in final_state.context
        else:
            # Si rechazado, strata debe estar vacío
            assert len(final_state.validated_strata) == 0
            assert 'quantum_error' in final_state.context


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7: TESTS DE PROPIEDADES ALGEBRAICAS
# ═════════════════════════════════════════════════════════════════════════════

class TestAlgebraicProperties:
    """
    Suite de validación de propiedades algebraicas y cuánticas.
    
    Verifica:
        - Axiomas del espacio de Hilbert
        - Conservación probabilística
        - Hermiticidad
        - Funtorialidad
    """
    
    def test_property_hilbert_space_completeness(self, quantum_gate, sample_payloads):
        """
        Verifica axioma de completitud: Σ |n⟩⟨n| = 𝟙.
        
        En nuestro caso: P(ADMITIDO) + P(RECHAZADO) = 1 (implícitamente).
        """
        # Ejecutar múltiples mediciones
        payloads = [sample_payloads['simple'], sample_payloads['complex']]
        
        for payload in payloads:
            measurement = quantum_gate.evaluate_admission(payload)
            
            # Cada medición debe colapsar a uno de los dos estados
            assert measurement.eigenstate in [Eigenstate.ADMITIDO, Eigenstate.RECHAZADO]
    
    def test_property_born_rule_normalization(self, quantum_gate):
        """
        Verifica que probabilidades estén normalizadas.
        
        0 ≤ T ≤ 1 y 0 ≤ θ < 1.
        """
        for _ in range(50):
            payload = {'data': os.urandom(100).hex()}
            measurement = quantum_gate.evaluate_admission(payload)
            
            assert 0.0 <= measurement.tunneling_probability <= 1.0
            assert 0.0 <= measurement.collapse_threshold < 1.0
    
    def test_property_hermiticity_real_eigenvalues(self, quantum_gate, sample_payloads):
        """
        Verifica que observables sean reales (hermiticidad del operador).
        """
        payload = sample_payloads['simple']
        measurement = quantum_gate.evaluate_admission(payload)
        
        # Todos los observables deben ser reales y finitos
        assert math.isfinite(measurement.incident_energy)
        assert math.isfinite(measurement.work_function)
        assert math.isfinite(measurement.kinetic_energy)
        assert math.isfinite(measurement.momentum)
        assert math.isfinite(measurement.threat_level)
        
        # Polo puede ser cualquier real
        assert math.isfinite(measurement.dominant_pole_real)
    
    def test_property_functoriality_composition(self, stable_mocks, sample_payloads):
        """
        Verifica funtorialidad: F(id) = id.
        
        Aplicar gate dos veces al mismo payload (con mocks idénticos)
        debe dar mismo resultado.
        """
        topo, laplace, sheaf = stable_mocks
        gate1 = QuantumAdmissionGate(topo, laplace, sheaf)
        gate2 = QuantumAdmissionGate(topo, laplace, sheaf)
        
        payload = sample_payloads['simple']
        
        m1 = gate1.evaluate_admission(payload)
        m2 = gate2.evaluate_admission(payload)
        
        # Mismo resultado (determinismo)
        assert m1.eigenstate == m2.eigenstate
        assert m1.incident_energy == m2.incident_energy
    
    def test_property_no_cloning_immutability(self, quantum_gate, sample_payloads):
        """
        Verifica teorema de no-clonación (inmutabilidad de medición).
        """
        payload = sample_payloads['simple']
        measurement = quantum_gate.evaluate_admission(payload)
        
        # No debe existir __copy__ ni __deepcopy__
        import copy
        
        # Shallow copy debe retornar misma referencia (frozen dataclass)
        copied = copy.copy(measurement)
        assert copied is measurement or copied == measurement
        
        # Modificación debe fallar
        with pytest.raises(AttributeError):
            measurement.eigenstate = Eigenstate.RECHAZADO


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8: TESTS DE INVARIANTES FÍSICOS
# ═════════════════════════════════════════════════════════════════════════════

class TestPhysicalInvariants:
    """
    Suite de validación de invariantes físicos.
    
    Verifica que se preserven leyes de conservación y restricciones.
    """
    
    def test_invariant_energy_non_negativity(self, quantum_gate):
        """
        Verifica que todas las energías sean no negativas.
        """
        for _ in range(50):
            payload = {'data': os.urandom(50).hex()}
            measurement = quantum_gate.evaluate_admission(payload)
            
            assert measurement.incident_energy >= 0.0
            assert measurement.work_function >= 0.0
            assert measurement.kinetic_energy >= 0.0
    
    def test_invariant_momentum_non_negativity(self, quantum_gate):
        """
        Verifica que momentum sea no negativo.
        """
        for _ in range(50):
            payload = {'data': os.urandom(50).hex()}
            measurement = quantum_gate.evaluate_admission(payload)
            
            assert measurement.momentum >= 0.0
    
    def test_invariant_rejected_state_zero_momentum(self, quantum_gate):
        """
        Verifica que estado rechazado tenga p = K = 0.
        """
        # Forzar rechazo con payload vacío
        measurement = quantum_gate.evaluate_admission({})
        
        if measurement.eigenstate == Eigenstate.RECHAZADO:
            assert measurement.kinetic_energy == 0.0
            assert measurement.momentum == 0.0
    
    def test_invariant_admitted_state_positive_momentum(self, quantum_gate, sample_payloads):
        """
        Verifica que estado admitido tenga p > 0.
        """
        # Intentar con payload grande (alta probabilidad de admisión)
        payload = {'data': 'A' * 1000}
        measurement = quantum_gate.evaluate_admission(payload)
        
        if measurement.eigenstate == Eigenstate.ADMITIDO:
            assert measurement.momentum > 0.0
            assert measurement.kinetic_energy >= Const.MIN_KINETIC_ENERGY
    
    def test_invariant_kinetic_energy_bounded_by_incident(self, quantum_gate, sample_payloads):
        """
        Verifica que K ≤ E (no puede haber más energía cinética que incidente).
        """
        for payload in sample_payloads.values():
            if not payload:  # Skip empty
                continue
            
            measurement = quantum_gate.evaluate_admission(payload)
            
            # K = max(E - Φ, K_min) ≤ E (si E ≥ Φ)
            # En régimen túnel, K = K_min ≪ E
            if measurement.eigenstate == Eigenstate.ADMITIDO:
                # Tolerancia por K_min
                assert measurement.kinetic_energy <= measurement.incident_energy + Const.MIN_KINETIC_ENERGY
    
    def test_invariant_work_function_grows_with_threat(self):
        """
        Verifica monotonía: χ² ↑ ⟹ Φ ↑.
        """
        threats = [0.0, 1.0, 5.0, 10.0]
        work_functions = []
        
        for threat in threats:
            watcher = MockTopologicalWatcher(threat=threat)
            modulator = WorkFunctionModulator(watcher)
            Phi, _ = modulator.calculate()
            work_functions.append(Phi)
        
        # Verificar orden creciente
        for i in range(len(work_functions) - 1):
            assert work_functions[i] <= work_functions[i + 1]
    
    def test_invariant_mass_decreases_with_stability(self):
        """
        Verifica monotonía: |σ| ↑ ⟹ m_eff ↓ (sistema más estable).
        """
        poles = [-0.1, -0.5, -1.0, -5.0]
        masses = []
        
        for pole in poles:
            oracle = MockLaplaceOracle(pole_real=pole)
            modulator = EffectiveMassModulator(oracle)
            m_eff, _ = modulator.calculate()
            
            if math.isfinite(m_eff):
                masses.append(m_eff)
        
        # Verificar orden decreciente (masa disminuye con estabilidad)
        for i in range(len(masses) - 1):
            assert masses[i] >= masses[i + 1]


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 9: TESTS DE CASOS LÍMITE
# ═════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """
    Suite de validación de casos límite y extremos.
    """
    
    def test_edge_empty_payload_rejected(self, quantum_gate):
        """
        Verifica que payload vacío sea rechazado.
        """
        measurement = quantum_gate.evaluate_admission({})
        
        assert measurement.eigenstate == Eigenstate.RECHAZADO
        assert measurement.incident_energy == 0.0
    
    def test_edge_very_large_payload(self, quantum_gate):
        """
        Verifica manejo de payload muy grande (10MB simulado).
        """
        # No crear realmente 10MB en memoria, simular con hash
        large_payload = {'data': 'X' * 100000}  # 100KB
        
        measurement = quantum_gate.evaluate_admission(large_payload)
        
        # Debe terminar sin error
        assert isinstance(measurement, QuantumMeasurement)
        assert measurement.incident_energy > 0.0
    
    def test_edge_high_entropy_payload(self, quantum_gate):
        """
        Verifica payload con entropía máxima.
        """
        # Datos aleatorios → entropía ≈ máxima
        random_payload = {
            'data': os.urandom(1000).hex()
        }
        
        measurement = quantum_gate.evaluate_admission(random_payload)
        
        assert isinstance(measurement, QuantumMeasurement)
    
    def test_edge_low_entropy_payload(self, quantum_gate):
        """
        Verifica payload con entropía mínima.
        """
        # Datos constantes → entropía = 0
        constant_payload = {
            'data': 'A' * 1000
        }
        
        measurement = quantum_gate.evaluate_admission(constant_payload)
        
        # Baja entropía → energía muy alta → probablemente admitido
        assert isinstance(measurement, QuantumMeasurement)
    
    def test_edge_nested_payload(self, quantum_gate):
        """
        Verifica payload con estructura anidada profunda.
        """
        nested = {'level1': {'level2': {'level3': {'data': 'deep'}}}}
        
        measurement = quantum_gate.evaluate_admission(nested)
        
        assert isinstance(measurement, QuantumMeasurement)
    
    def test_edge_unicode_payload(self, quantum_gate):
        """
        Verifica payload con caracteres Unicode.
        """
        unicode_payload = {
            'message': '你好世界 🌍 مرحبا العالم',
            'emoji': '🚀🔬⚛️'
        }
        
        measurement = quantum_gate.evaluate_admission(unicode_payload)
        
        assert isinstance(measurement, QuantumMeasurement)
    
    def test_edge_special_float_values_in_mocks(self):
        """
        Verifica manejo de valores especiales en mocks.
        """
        # Amenaza muy alta
        watcher_extreme = MockTopologicalWatcher(threat=1e10)
        modulator = WorkFunctionModulator(watcher_extreme)
        
        Phi, threat = modulator.calculate()
        
        # No debe producir infinito (saturación a FLOAT_MAX)
        assert math.isfinite(Phi) or Phi == Const.FLOAT_MAX
    
    def test_edge_zero_barrier_classical_limit(self):
        """
        Verifica límite clásico: Φ = 0 ⟹ T = 1.
        """
        T, params = WKBCalculator.compute_tunneling_probability(
            E=10.0,
            Phi=0.0,
            m_eff=1.0
        )
        
        assert T == 1.0


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 10: TESTS ESTADÍSTICOS
# ═════════════════════════════════════════════════════════════════════════════

class TestStatisticalProperties:
    """
    Suite de validación de propiedades estadísticas.
    
    Verifica distribuciones, varianzas, y comportamiento en ensemble.
    """
    
    def test_statistical_threshold_distribution(self):
        """
        Verifica que θ tenga distribución aproximadamente uniforme.
        """
        n_samples = 1000
        thresholds = []
        
        for _ in range(n_samples):
            payload = {'random': os.urandom(32).hex()}
            payload_bytes = PayloadSerializer.serialize(payload)
            payload_hash = PayloadSerializer.deterministic_hash(payload_bytes)
            theta = CollapseThresholdGenerator.generate(payload_hash)
            thresholds.append(theta)
        
        # Calcular estadísticas
        mean = sum(thresholds) / len(thresholds)
        
        # Media debe estar cerca de 0.5 (distribución uniforme)
        assert 0.45 <= mean <= 0.55
        
        # Todos en [0, 1)
        assert all(0.0 <= t < 1.0 for t in thresholds)
    
    def test_statistical_admission_rate_stability(self, quantum_gate):
        """
        Verifica que tasa de admisión sea estable en ensemble grande.
        """
        n_trials = 100
        admitted_count = 0
        
        for i in range(n_trials):
            payload = {'trial': i, 'data': os.urandom(50).hex()}
            measurement = quantum_gate.evaluate_admission(payload)
            
            if measurement.eigenstate == Eigenstate.ADMITIDO:
                admitted_count += 1
        
        admission_rate = admitted_count / n_trials
        
        # Tasa debe estar en rango razonable (no todo admitido ni todo rechazado)
        assert 0.1 <= admission_rate <= 0.9
    
    def test_statistical_energy_variance(self, quantum_gate):
        """
        Verifica que energía incidente tenga varianza razonable.
        """
        energies = []
        
        for _ in range(100):
            payload = {'data': os.urandom(100).hex()}
            measurement = quantum_gate.evaluate_admission(payload)
            energies.append(measurement.incident_energy)
        
        # Calcular varianza
        mean_energy = sum(energies) / len(energies)
        variance = sum((e - mean_energy)**2 for e in energies) / len(energies)
        
        # Varianza debe ser positiva (hay dispersión)
        assert variance > 0


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 11: TESTS DE REGRESIÓN
# ═════════════════════════════════════════════════════════════════════════════

class TestRegression:
    """
    Suite de regresión con casos conocidos.
    
    Captura comportamiento esperado en escenarios específicos.
    """
    
    def test_regression_known_admission_case(self, quantum_gate):
        """
        Verifica admisión de un caso conocido (payload simple, sistema estable).
        """
        known_payload = {
            'endpoint': '/api/health',
            'method': 'GET',
            'data': 'simple'
        }
        
        measurement = quantum_gate.evaluate_admission(known_payload)
        
        # Este payload debería tener energía moderada
        assert measurement.incident_energy > 0.0
        assert measurement.work_function > 0.0
    
    def test_regression_known_rejection_case(self):
        """
        Verifica rechazo de payload vacío (caso trivial).
        """
        topo = MockTopologicalWatcher(threat=1.0)
        laplace = MockLaplaceOracle(pole_real=-0.5)
        sheaf = MockSheafOrchestrator(frustration=1e-12)
        
        gate = QuantumAdmissionGate(topo, laplace, sheaf)
        
        empty_payload = {}
        measurement = gate.evaluate_admission(empty_payload)
        
        assert measurement.eigenstate == Eigenstate.RECHAZADO
        assert measurement.incident_energy == 0.0
    
    def test_regression_veto_case(self):
        """
        Verifica veto cohomológico con alta frustración.
        """
        topo = MockTopologicalWatcher(threat=1.0)
        laplace = MockLaplaceOracle(pole_real=-0.5)
        sheaf = MockSheafOrchestrator(frustration=10.0)  # Muy alta
        
        gate = QuantumAdmissionGate(topo, laplace, sheaf)
        
        payload = {'data': 'test'}
        measurement = gate.evaluate_admission(payload)
        
        assert measurement.eigenstate == Eigenstate.RECHAZADO
        assert measurement.frustration_veto is True


# ═════════════════════════════════════════════════════════════════════════════
# SECCIÓN 12: TESTS DE PERFORMANCE (OPCIONAL)
# ═════════════════════════════════════════════════════════════════════════════

class TestPerformance:
    """
    Suite de validación de performance (opcional, marca como slow).
    """
    
    @pytest.mark.slow
    def test_performance_large_batch(self, quantum_gate):
        """
        Verifica que el gate maneje lote grande sin degradación.
        """
        import time
        
        n_payloads = 1000
        start_time = time.time()
        
        for i in range(n_payloads):
            payload = {'index': i, 'data': f'payload_{i}'}
            measurement = quantum_gate.evaluate_admission(payload)
            assert isinstance(measurement, QuantumMeasurement)
        
        elapsed = time.time() - start_time
        
        # Debe completar en tiempo razonable (< 10s para 1000 payloads)
        assert elapsed < 10.0
        
        # Throughput promedio
        throughput = n_payloads / elapsed
        print(f"\nThroughput: {throughput:.2f} payloads/segundo")
    
    @pytest.mark.slow
    def test_performance_cache_effectiveness(self):
        """
        Verifica que caché LRU mejore performance en payloads repetidos.
        """
        import time
        
        # Mismo payload repetido
        payload = {'data': 'cached_payload'}
        payload_bytes = PayloadSerializer.serialize(payload)
        
        # Primera llamada (sin caché)
        start = time.time()
        for _ in range(100):
            _ = PayloadSerializer.deterministic_hash(payload_bytes)
        first_run = time.time() - start
        
        # Segunda llamada (con caché)
        start = time.time()
        for _ in range(100):
            _ = PayloadSerializer.deterministic_hash(payload_bytes)
        second_run = time.time() - start
        
        # Segunda ejecución debe ser más rápida (caché)
        assert second_run <= first_run


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PYTEST
# ═════════════════════════════════════════════════════════════════════════════

def pytest_configure(config):
    """
    Configuración adicional de pytest.
    """
    config.addinivalue_line(
        "markers",
        "slow: marca tests que requieren más tiempo de ejecución"
    )


# ═════════════════════════════════════════════════════════════════════════════
# PUNTO DE ENTRADA PARA EJECUCIÓN DIRECTA
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Permite ejecutar la suite completa directamente.
    
    Uso:
        python test_quantum_admission_gate.py
    """
    pytest.main([
        __file__,
        "-v",  # Verbose
        "--tb=short",  # Traceback corto
        "--color=yes",  # Colorear output
        "-ra",  # Resumen de todos los tests
    ])