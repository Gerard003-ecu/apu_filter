"""
Módulo: tests/integration/dynamic_stress/test_mic_improbability_coupling.py

═══════════════════════════════════════════════════════════════════════════════
SUITE DE INTEGRACIÓN GEOMÉTRICA: ACOPLAMIENTO DEL MOTOR DE IMPROBABILIDAD EN LA MIC
═══════════════════════════════════════════════════════════════════════════════

Taxonomía de Invariantes Verificados:
───────────────────────────────────────────────────────────────────────────────

INVARIANTES CATEGORIALES:
─────────────────────────
    A. Acoplamiento Ortogonal:    
            rank(MIC') = rank(MIC) + 1, con e_{n+1} ∈ Stratum.OMEGA
       
    B. Proyección Funtorial:      
            F: ℂ_top → ℝ_Δ  tal que  F(Ψ, ROI) ↦ I(Ψ, ROI)
            preservando el estado global (pureza referencial).
       
    C. Composición Tensorial:    
            Τ₁ ∘ Τ₂ = ImprobabilityTensor(κ₁·κ₂, γ₁+γ₂)

INVARIANTES TOPOLÓGICOS:
────────────────────────
    D. Propagación de Veto:       
            lim_{Ψ→0⁺} I(Ψ, ROI) = I_max ⟹ Colapso de la función de onda (Veto).
       
    E. Continuidad en Frontera:   
            Ψ_eff = ε·sigmoid(Ψ/ε)  garantiza continuidad en Ψ=0.
       
    F. Aislamiento Monádico:      
            Inyección de entropía (NaN, strings) es absorbida sin fracturar
            la variedad base.

INVARIANTES NUMÉRICOS:
─────────────────────
    G. Monotonía Estricta:
            ∂I/∂ROI > 0  para todo (Ψ, ROI) ∈ ℝ₊²
            ∂I/∂Ψ < 0    para todo (Ψ, ROI) ∈ ℝ₊²
            
    H. Lipschitz Continuidad:
            ||F(x) - F(y)||∞ ≤ L · ||x - y||∞
            con L = κ·γ·(ROI_max/Ψ_min)^(γ-1)
            
    I. Validez del Dominio y Rango:
            domain(F) ⊆ ℝ₊ × ℝ₊  (ortante positivo)
            range(F) ⊆ [I_min, I_max]  (politopo admisible)

Marco teórico:
───────────────────────────────────────────────────────────────────────────────
El Motor de Improbabilidad actúa como un deformador del Tensor Métrico Riemanniano 
en el Estrato Ω. Esta suite garantiza que su inyección mediante el patrón Command 
(project_intent) preserve la matriz identidad I_n de las herramientas ortogonales,
cumpliendo estrictamente la Ley de Clausura Transitiva de la pirámide DIKW.

Referencias:
    • Awodey, S. "Category Theory" (2010) - Fundamentos categóricos
    • MacLane, S. "Categories for the Working Mathematician" - Teoría de functores
    • Golub & Van Loan "Matrix Computations" - Estabilidad numérica
"""

from __future__ import annotations

import math
import os
import json
import pytest
import threading
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Type
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time

# Esterilización del Vacío Termodinámico (Determinismo estricto en BLAS/LAPACK)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from app.core.schemas import Stratum
from app.adapters.tools_interface import MICRegistry
from app.omega.improbability_drive import (
    ImprobabilityDriveService,
    ImprobabilityTensor,
    ImprobabilityResult,
    TensorAlgebra,
    TensorFactory,
    NumericalPrecision,
    # Excepciones
    ImprobabilityDriveError,
    DimensionalMismatchError,
    TypeCoercionError,
    AxiomViolationError,
    NumericalInstabilityError,
    # Constantes
    _IMPROBABILITY_CLAMP_HIGH,
    _IMPROBABILITY_CLAMP_LOW,
    _EPS_MACH,
    _MIN_KAPPA,
    _MAX_KAPPA,
    _MIN_GAMMA,
    _MAX_GAMMA,
    _EPS_CRITICAL,
    _VETO_THRESHOLD_FACTOR,
)


# ════════════════════════════════════════════════════════════════════════════
# PARÁMETROS DE TOLERANCIA NUMÉRICA
# ════════════════════════════════════════════════════════════════════════════

class NumericalTolerances:
    """
    Tolerancias numéricas para comparaciones en tests.
    
    Organizadas por contexto de uso para mantener consistencia
    y facilitar ajustes futuros.
    """
    
    # Tolerancias relativas (para comparaciones proporcionales)
    REL_TOL_DEFAULT: float = 1e-9
    REL_TOL_COARSE: float = 1e-5
    REL_TOL_STRICT: float = 1e-12
    
    # Tolerancias absolutas (para comparaciones near-zero)
    ABS_TOL_DEFAULT: float = 1e-15
    ABS_TOL_FP16: float = 1e-3  # half precision
    ABS_TOL_FP32: float = 1e-7  # single precision
    
    # Factores de saturación para tests de límites
    VETO_PROXIMITY_FACTOR: float = 0.999
    OVERFLOW_APPROACH_FACTOR: float = 0.9
    
    @classmethod
    def is_close(
        cls,
        a: float,
        b: float,
        rel_tol: Optional[float] = None,
        abs_tol: Optional[float] = None
    ) -> bool:
        """
        Verifica si dos valores son cercanos numéricamente.
        
        Implementa la definición:
            |a - b| ≤ max(rel_tol * max(|a|, |b|), abs_tol)
        
        Args:
            a: Primer valor.
            b: Segundo valor.
            rel_tol: Tolerancia relativa (default: REL_TOL_DEFAULT).
            abs_tol: Tolerancia absoluta (default: ABS_TOL_DEFAULT).
        
        Returns:
            True si los valores son considerados iguales.
        """
        rel = rel_tol if rel_tol is not None else cls.REL_TOL_DEFAULT
        abs_ = abs_tol if abs_tol is not None else cls.ABS_TOL_DEFAULT
        
        diff = abs(a - b)
        tolerance = max(rel * max(abs(a), abs(b)), abs_)
        
        return diff <= tolerance


# ════════════════════════════════════════════════════════════════════════════
# FIXTURES E INFRAESTRUCTURA TOPOLÓGICA
# ════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TestCase:
    """
    Caso de prueba parametrizado.
    
    Attributes:
        name: Identificador descriptivo del caso.
        psi: Valor de Ψ (estabilidad topológica).
        roi: Valor de ROI (rentabilidad).
        expected_penalty: Valor esperado de penalización (None si no se verifica).
        expected_veto: Resultado esperado de veto.
        description: Descripción del caso en el espacio topológico.
    """
    name: str
    psi: float
    roi: float
    expected_penalty: Optional[float]
    expected_veto: bool
    description: str


@pytest.fixture
def base_mic_registry() -> MICRegistry:
    """
    Crea una Matriz de Interacción Central (MIC) limpia.
    
    Returns:
        MICRegistry vacío, sin operadores registrados.
    """
    return MICRegistry()


@pytest.fixture
def active_mic_registry() -> MICRegistry:
    """
    Inyecta una Matriz de Interacción Central (MIC) limpia y registra el 
    operador tensorial del Motor de Improbabilidad, estableciendo la base 
    canónica e_{n+1} en el espacio vectorial del Estrato OMEGA.
    
    Fixture con scope 'function' para garantizar aislamiento entre tests.
    """
    registry = MICRegistry()
    service = ImprobabilityDriveService(
        mic_registry=registry,
        kappa=1.0,
        gamma=2.0
    )
    service.register_in_mic()
    return registry


@pytest.fixture
def standard_tensor() -> ImprobabilityTensor:
    """
    Tensor estándar (κ=1.0, γ=2.0) para tests directos del tensor.
    
    Returns:
        ImprobabilityTensor con parámetros por defecto.
    """
    return ImprobabilityTensor(kappa=1.0, gamma=2.0)


@pytest.fixture
def tensor_factory() -> Type[TensorFactory]:
    """
    Referencia a la clase TensorFactory.
    
    Returns:
        Clase TensorFactory.
    """
    return TensorFactory


# Casos de prueba predefinidos para parametrización
STANDARD_TEST_CASES: List[TestCase] = [
    TestCase(
        name="nominal_stable_high_roi",
        psi=0.95,
        roi=1.5,
        expected_penalty=(1.5 / 0.95) ** 2,
        expected_veto=False,
        description="Estado estable con alta rentabilidad"
    ),
    TestCase(
        name="nominal_equal",
        psi=1.0,
        roi=1.0,
        expected_penalty=1.0,
        expected_veto=False,
        description="Punto unitario del espacio"
    ),
    TestCase(
        name="unstable_high_roi",
        psi=0.01,
        roi=10.0,
        expected_penalty=None,  # Valor grande, verificamos veto
        expected_veto=True,
        description="Alta rentabilidad sobre baja estabilidad"
    ),
    TestCase(
        name="very_stable_low_roi",
        psi=100.0,
        roi=0.5,
        expected_penalty=_IMPROBABILITY_CLAMP_LOW,
        expected_veto=False,
        description="Estabilidad extrema con baja rentabilidad"
    ),
    TestCase(
        name="asymptotic_psi_zero",
        psi=_EPS_MACH,
        roi=3.0,
        expected_penalty=_IMPROBABILITY_CLAMP_HIGH,
        expected_veto=True,
        description="Proximidad a singularidad matemática"
    ),
]


# ════════════════════════════════════════════════════════════════════════════
# UTILIDADES DE TEST
# ════════════════════════════════════════════════════════════════════════════

def collect_errors(
    func,
    *args,
    expected_exceptions: Tuple[Type[Exception], ...] = ()
) -> Tuple[bool, Optional[Exception], Any]:
    """
    Ejecuta una función y captura excepciones.
    
    Args:
        func: Función a ejecutar.
        *args: Argumentos posicionales.
        expected_exceptions: Tipos de excepción esperados.
    
    Returns:
        Tupla (success, exception, result) donde:
            - success: True si la ejecución fue exitosa.
            - exception: Excepción capturada o None.
            - result: Valor de retorno o None si hubo excepción.
    """
    try:
        result = func(*args)
        return (True, None, result)
    except expected_exceptions as e:
        return (False, e, None)
    except Exception as e:
        return (False, e, None)


def verify_gradient_numerical(
    tensor: ImprobabilityTensor,
    psi: float,
    roi: float,
    h: float = 1e-7,
    rel_tol: float = 1e-5
) -> Tuple[bool, Dict[str, float]]:
    """
    Verifica gradientes analíticos contra aproximación numérica.
    
    Implementa la diferenciación finita central:
        f'(x) ≈ (f(x+h) - f(x-h)) / (2h)
    
    Args:
        tensor: Tensor a evaluar.
        psi: Punto de evaluación en Ψ.
        roi: Punto de evaluación en ROI.
        h: Paso de diferenciación.
        rel_tol: Tolerancia relativa para comparación.
    
    Returns:
        Tupla (passed, details) donde details contiene los valores comparados.
    """
    # Gradientes analíticos
    grad_analytical = tensor.compute_gradient(psi, roi)
    dI_dpsi_analytic, dI_droi_analytic = grad_analytical
    
    # Gradientes numéricos (diferenciación central)
    I_base = tensor.compute_penalty(psi, roi)
    
    # ∂I/∂Ψ numérico
    I_plus = tensor.compute_penalty(psi + h, roi)
    I_minus = tensor.compute_penalty(psi - h, roi)
    dI_dpsi_numeric = (I_plus - I_minus) / (2 * h)
    
    # ∂I/∂ROI numérico
    I_plus_r = tensor.compute_penalty(psi, roi + h)
    I_minus_r = tensor.compute_penalty(psi, roi - h)
    dI_droi_numeric = (I_plus_r - I_minus_r) / (2 * h)
    
    # Comparación
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
        "roi_passed": roi_passed
    })


# ════════════════════════════════════════════════════════════════════════════
# SUITE DE TESTS: INVARIANTES CATEGORIALES
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestMICImprobabilityCoupling:
    """
    Validación de la coherencia geométrica de la integración del Motor de 
    Improbabilidad en el ecosistema Port-Hamiltoniano.
    
    Categorías de Tests:
        1. Acoplamiento Ortogonal (MIC)
        2. Proyección Funtorial
        3. Propagación de Veto
        4. Aislamiento Monádico
        5. Serialización
        6. Composición Tensorial
    """

    # ─────────────────────────────────────────────────────────────────────────
    # GRUPO 1: ACOPLAMIENTO ORTOGONAL EN LA MIC
    # ─────────────────────────────────────────────────────────────────────────

    def test_mic_orthogonal_coupling_and_registration(
        self,
        active_mic_registry: MICRegistry
    ) -> None:
        """
        Invariante A: Acoplamiento Ortogonal en la MIC.
        
        Axioma: El servicio debe inyectar su proyector en la MIC bajo el estrato 
        correcto (OMEGA) sin generar dependencia lineal. El registro debe contener 
        el vector atómico exacto.
        
        Verificación:
            • El vector existe en la MIC
            • El vector reside en Stratum.OMEGA
            • El nombre del vector es exactamente "compute_improbability_penalty"
        """
        # Extracción del vector atómico del hiperespacio de la MIC
        vector = active_mic_registry.get_basis_vector("compute_improbability_penalty")
        
        # Aserción 1: Existencia del vector
        assert vector is not None, (
            "Ruptura del Fibrado: El vector de improbabilidad no logró "
            "acoplarse a la base de la MIC."
        )
        
        # Aserción 2: Residencia en el estrato correcto
        assert vector.target_stratum == Stratum.OMEGA, (
            f"Violación de Topología DIKW: El vector reside en "
            f"{vector.target_stratum.name}, se esperaba estrictamente "
            f"{Stratum.OMEGA.name}."
        )

    def test_mic_registry_increments_rank(
        self,
        base_mic_registry: MICRegistry,
        active_mic_registry: MICRegistry
    ) -> None:
        """
        Invariante A (corolario): Verifica que el registro incrementa el rango.
        
        |MIC'| = |MIC| + 1  (después del acoplamiento)
        """
        base_count = len(base_mic_registry.list_vectors())
        active_count = len(active_mic_registry.list_vectors())
        
        assert active_count == base_count + 1, (
            f"El acoplamiento no incremento el rango de la MIC. "
            f"Base: {base_count}, Activo: {active_count}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # GRUPO 2: PROYECCIÓN FUNTORIAL (CRUCE DE ESTRATOS)
    # ─────────────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("test_case", STANDARD_TEST_CASES, ids=lambda tc: tc.name)
    def test_functorial_projection_parametric(
        self,
        active_mic_registry: MICRegistry,
        test_case: TestCase
    ) -> None:
        """
        Invariante B: Proyección Funtorial (parametrizada).
        
        Axioma: La proyección de un estado a través del bus de la MIC debe 
        resolverse mediante el handler del servicio, computando la deformación 
        métrica sin efectos secundarios (pureza referencial).
        
        Formalismo:
            F(Ψ, ROI) = clip(κ · (ROI / Ψ)^γ, I_min, I_max)
        
        Args:
            active_mic_registry: MIC con el operador registrado.
            test_case: Caso de prueba parametrizado.
        """
        result_monad = active_mic_registry.project_intent(
            vector_name="compute_improbability_penalty",
            psi=test_case.psi,
            roi=test_case.roi
        )

        # Aserción 1: Éxito de la computación
        assert result_monad.get("success") is True, (
            f"El Funtor falló al mapear el estado '{test_case.name}': "
            f"{result_monad.get('error', 'Unknown error')}"
        )
        
        # Aserción 2: Presencia del tensor de penalización
        assert "improbability_penalty" in result_monad, (
            "Pérdida de información: falta el tensor de penalización en "
            "la mónada de resultado."
        )
        
        # Aserción 3: Consistencia del veto
        assert result_monad.get("is_vetoed") == test_case.expected_veto, (
            f"Discrepancia en el veto para '{test_case.name}': "
            f"esperado={test_case.expected_veto}, "
            f"obtenido={result_monad.get('is_vetoed')}"
        )
        
        # Aserción 4: Verificación del valor esperado (si está definido)
        if test_case.expected_penalty is not None:
            actual_penalty = result_monad["improbability_penalty"]
            assert NumericalTolerances.is_close(
                actual_penalty,
                test_case.expected_penalty,
                rel_tol=NumericalTolerances.REL_TOL_COARSE
            ), (
                f"Fractura geométrica en el cálculo de la penalización para "
                f"'{test_case.name}': esperado≈{test_case.expected_penalty:.6f}, "
                f"obtenido={actual_penalty:.6f}"
            )

    def test_functorial_projection_success(
        self,
        active_mic_registry: MICRegistry
    ) -> None:
        """
        Invariante B: Proyección Funtorial (caso nominal).
        
        Caso de prueba específico con validación matemática exacta.
        """
        psi_nominal = 0.95
        roi_nominal = 1.5
        
        result_monad = active_mic_registry.project_intent(
            vector_name="compute_improbability_penalty",
            psi=psi_nominal,
            roi=roi_nominal
        )

        assert result_monad.get("success") is True, (
            f"El Funtor falló al mapear el estado estable: "
            f"{result_monad.get('error', 'Unknown error')}"
        )
        assert "improbability_penalty" in result_monad, (
            "Pérdida de información: falta el tensor de penalización."
        )
        assert result_monad.get("is_vetoed") is False, (
            "Falso positivo: Un estado sano no debe inducir colapso (Veto)."
        )
        
        # Validación matemática exacta del operador: I = (1.5 / 0.95)^2
        expected_penalty = (roi_nominal / psi_nominal) ** 2
        actual_penalty = result_monad["improbability_penalty"]
        assert NumericalTolerances.is_close(
            actual_penalty, expected_penalty, rel_tol=1e-5
        ), (
            f"Fractura geométrica en el cálculo: esperado={expected_penalty:.8f}, "
            f"obtenido={actual_penalty:.8f}"
        )

    # ─────────────────────────────────────────────────────────────────────────
    # GRUPO 3: PROPAGACIÓN DE VETO TERMODINÁMICO
    # ─────────────────────────────────────────────────────────────────────────

    def test_asymptotic_veto_propagation(
        self,
        active_mic_registry: MICRegistry
    ) -> None:
        """
        Invariante C: Propagación del Veto Termodinámico.
        
        Axioma: Cuando el espacio topológico se fractura (Ψ → 0) ante 
        hiper-rentabilidad, el operador debe saturar la métrica asintóticamente 
        y emitir un Veto Físico ineludible.
        
        Formalismo:
            lim_{Ψ→0⁺} I(Ψ, ROI) = I_max
        
        Caso de prueba:
            Ψ = ε_máquina × 10  (proximidad a singularidad)
            ROI = 3.0           (300% de rentabilidad)
        """
        result_monad = active_mic_registry.project_intent(
            vector_name="compute_improbability_penalty",
            psi=_EPS_MACH * 10,
            roi=3.0
        )

        # Aserción 1: La mónada evaluó el caso límite
        assert result_monad.get("success") is True, (
            "La mónada falló en evaluar el caso límite asintótico."
        )
        
        # Aserción 2: Saturación en el límite superior
        assert result_monad.get("improbability_penalty") == _IMPROBABILITY_CLAMP_HIGH, (
            f"Fallo del Retracto Topológico: La penalización no fue acotada. "
            f"Esperado={_IMPROBABILITY_CLAMP_HIGH}, "
            f"Obtenido={result_monad.get('improbability_penalty')}"
        )
        
        # Aserción 3: Emisión del Veto
        assert result_monad.get("is_vetoed") is True, (
            "Infracción Termodinámica: El sistema no colapsó la función de onda "
            "(Veto Estructural) ante una singularidad de riesgo geométrica."
        )

    def test_veto_threshold_boundary(
        self,
        active_mic_registry: MICRegistry
    ) -> None:
        """
        Invariante C (corolario): Verifica el límite exacto del veto.
        
        El veto se activa cuando:
            I ≥ I_max × factor_umbral = I_max × 0.999
        """
        threshold = _IMPROBABILITY_CLAMP_HIGH * _VETO_THRESHOLD_FACTOR
        
        # Caso en el umbral: penalización justo en el límite del veto
        result_at_threshold = active_mic_registry.project_intent(
            vector_name="compute_improbability_penalty",
            psi=_EPS_MACH,
            roi=1e6  # ROI极端
        )
        
        penalty = result_at_threshold.get("improbability_penalty", 0)
        
        # Debe estar cerca o en el máximo
        assert penalty >= threshold * 0.99, (
            f"La penalización {penalty} debería estar cerca del máximo "
            f"para este caso extremo."
        )

    @pytest.mark.parametrize("psi_value,roi_value,should_veto", [
        (_EPS_MACH * 100, 1.0, True),    # Ψ muy pequeño
        (_EPS_MACH, 1.0, True),          # Ψ en épsilon de máquina
        (1e-5, 1.0, True),                # Ψ bajo con ROI=1
        (1e-3, 10.0, True),               # Ψ bajo con ROI alto
        (1.0, 1.0, False),               # Estado nominal
        (10.0, 1.0, False),               # Ψ alto, ROI=1
        (100.0, 100.0, False),           # Ψ muy alto
    ])
    def test_veto_conditions_comprehensive(
        self,
        active_mic_registry: MICRegistry,
        psi_value: float,
        roi_value: float,
        should_veto: bool
    ) -> None:
        """
        Invariante C (extensión): Tabla completa de condiciones de veto.
        
        Verifica la consistencia de la emisión de veto para diferentes
        combinaciones de (Ψ, ROI).
        """
        result = active_mic_registry.project_intent(
            vector_name="compute_improbability_penalty",
            psi=psi_value,
            roi=roi_value
        )
        
        assert result.get("success") is True, (
            f"La evaluación falló para Ψ={psi_value}, ROI={roi_value}"
        )
        
        if should_veto:
            assert result.get("is_vetoed") is True, (
                f"Se esperaba veto para Ψ={psi_value}, ROI={roi_value}, "
                f"pero no se emitió. Penalización: {result.get('improbability_penalty')}"
            )
        else:
            assert result.get("is_vetoed") is False, (
                f"No se esperaba veto para Ψ={psi_value}, ROI={roi_value}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # GRUPO 4: AISLAMIENTO MONÁDICO DE SINGULARIDADES
    # ─────────────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("psi_input,roi_input,expected_error_type", [
        ("entropia_corrupta", 1.5, "TypeCoercionError"),
        (None, 1.5, "DimensionalMismatchError"),
        (1.5, None, "DimensionalMismatchError"),
        (float('nan'), 1.5, "DimensionalMismatchError"),
        (1.5, float('nan'), "DimensionalMismatchError"),
        (float('inf'), 1.5, "DimensionalMismatchError"),
        (1.5, float('-inf'), "DimensionalMismatchError"),
        (-1.0, 1.5, "AxiomViolationError"),
        (1.5, -1.0, "AxiomViolationError"),
        (1.5, 0.0, "AxiomViolationError"),
        ("texto", "另一文本", "TypeCoercionError"),
        ([], 1.5, "TypeCoercionError"),
        ({}, 1.5, "TypeCoercionError"),
    ])
    def test_monadic_error_isolation_comprehensive(
        self,
        active_mic_registry: MICRegistry,
        psi_input: Any,
        roi_input: Any,
        expected_error_type: str
    ) -> None:
        """
        Invariante D: Aislamiento Monádico de Singularidades (parametrizado).
        
        Axioma: La inyección de entropía pura (variables categóricas o tipos 
        no numéricos) debe ser interceptada. La MIC no debe propagar excepciones 
        crudas, sino encapsularlas en la Mónada de Error para mantener la 
        matriz no-singular.
        
        Args:
            psi_input: Valor de Ψ a inyectar.
            roi_input: Valor de ROI a inyectar.
            expected_error_type: Tipo de error esperado.
        """
        result_monad = active_mic_registry.project_intent(
            vector_name="compute_improbability_penalty",
            psi=psi_input,
            roi=roi_input
        )

        # Aserción 1: La mónada reporta fallo
        assert result_monad.get("success") is False, (
            f"La barrera monádica permitió el paso de entropía cruda. "
            f"Entrada: psi={psi_input} (type={type(psi_input).__name__}), "
            f"roi={roi_input} (type={type(roi_input).__name__})"
        )
        
        # Aserción 2: Presencia de información de error
        assert "error" in result_monad or "error_type" in result_monad, (
            "La mónada de fallo carece del rastro de stacktrace o mensaje."
        )
        
        # Aserción 3: Tipo de error reconocido
        error_type = result_monad.get("error_type", "")
        assert error_type == expected_error_type, (
            f"Clasificación de error incorrecta. "
            f"Esperado: {expected_error_type}, "
            f"Obtenido: {error_type}"
        )

    def test_monadic_error_isolation(
        self,
        active_mic_registry: MICRegistry
    ) -> None:
        """
        Invariante D: Aislamiento Monádico (test original).
        
        Caso específico con strings como entrada inválida.
        """
        result_monad = active_mic_registry.project_intent(
            vector_name="compute_improbability_penalty",
            psi="entropia_corrupta",
            roi=1.5
        )

        assert result_monad.get("success") is False, (
            "La barrera monádica permitió el paso de entropía cruda."
        )
        
        error_type = result_monad.get("error_type")
        assert error_type in [
            "DimensionalMismatchError",
            "NumericalInstabilityError",
            "TypeCoercionError",
            "ValueError",
            "TypeError"
        ], (
            f"Fuga dimensional: El colapso no fue clasificado en un dominio "
            f"de error conocido. Tipo: {error_type}"
        )
        
        assert "error" in result_monad, (
            "La mónada de fallo carece del rastro de stacktrace o mensaje."
        )


# ════════════════════════════════════════════════════════════════════════════
# SUITE DE TESTS: PROPIEDADES MATEMÁTICAS DEL TENSOR
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestImprobabilityTensorProperties:
    """
    Verificación de propiedades matemáticas formales del tensor.
    
    Invariantes verificados:
        • Monotonía estricta
        • Continuidad
        • Diferenciabilidad
        • Lipschitz Continuidad
        • Composición tensorial
        • Serialización
    """

    # ─────────────────────────────────────────────────────────────────────────
    # MONOTONÍA Y MONOTONÍA ESTRICTA
    # ─────────────────────────────────────────────────────────────────────────

    def test_monotonicity_in_roi(self, standard_tensor: ImprobabilityTensor) -> None:
        """
        Invariante G (parte 1): Monotonía estricta en ROI.
        
        Teorema: ∂I/∂ROI > 0 para todo (Ψ, ROI) ∈ ℝ₊²
        
        Esto implica que I(Ψ, ROI₁) < I(Ψ, ROI₂) iff ROI₁ < ROI₂.
        """
        psi_fixed = 1.0
        roi_values = [0.5, 1.0, 1.5, 2.0, 5.0, 10.0]
        
        penalties = [
            standard_tensor.compute_penalty(psi_fixed, roi)
            for roi in roi_values
        ]
        
        # Verificar monotonía creciente
        for i in range(len(penalties) - 1):
            assert penalties[i] < penalties[i + 1], (
                f"Violación de monotonía en ROI: "
                f"ROI={roi_values[i]} → I={penalties[i]}, "
                f"ROI={roi_values[i+1]} → I={penalties[i+1]}"
            )

    def test_monotonicity_in_psi(self, standard_tensor: ImprobabilityTensor) -> None:
        """
        Invariante G (parte 2): Monotonía decreciente en Ψ.
        
        Teorema: ∂I/∂Ψ < 0 para todo (Ψ, ROI) ∈ ℝ₊²
        
        Esto implica que I(Ψ₁, ROI) > I(Ψ₂, ROI) iff Ψ₁ < Ψ₂.
        """
        roi_fixed = 1.5
        psi_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0]
        
        penalties = [
            standard_tensor.compute_penalty(psi, roi_fixed)
            for psi in psi_values
        ]
        
        # Verificar monotonía decreciente
        for i in range(len(penalties) - 1):
            assert penalties[i] > penalties[i + 1], (
                f"Violación de monotonía en Ψ: "
                f"Ψ={psi_values[i]} → I={penalties[i]}, "
                f"Ψ={psi_values[i+1]} → I={penalties[i+1]}"
            )

    def test_gradient_analytic_vs_numeric(
        self,
        standard_tensor: ImprobabilityTensor
    ) -> None:
        """
        Invariante de Diferenciabilidad: Gradientes analíticos vs numéricos.
        
        Verifica que las derivadas analíticas coinciden con la aproximación
        por diferencias finitas centrales.
        """
        test_points = [
            (1.0, 1.0),
            (0.5, 2.0),
            (2.0, 0.5),
            (10.0, 10.0),
            (0.1, 5.0),
        ]
        
        for psi, roi in test_points:
            passed, details = verify_gradient_numerical(
                standard_tensor, psi, roi, rel_tol=1e-4
            )
            
            assert passed, (
                f"Gradiente analítico/numérico divergente en (Ψ={psi}, ROI={roi}):\n"
                f"  ∂I/∂Ψ: analítico={details['dI_dpsi_analytic']:.6e}, "
                f"numérico={details['dI_dpsi_numeric']:.6e}\n"
                f"  ∂I/∂ROI: analítico={details['dI_droi_analytic']:.6e}, "
                f"numérico={details['dI_droi_numeric']:.6e}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # CONTINUIDAD Y LÍMITES
    # ─────────────────────────────────────────────────────────────────────────

    def test_continuity_at_psi_zero(self, standard_tensor: ImprobabilityTensor) -> None:
        """
        Invariante E: Continuidad en la frontera Ψ = 0.
        
        Teorema: La regularización sigmoidal garantiza:
            lim_{Ψ→0⁺} Ψ_eff = ε/2
            lim_{Ψ→0⁺} I(Ψ, ROI) = finito
        
        Verifica que no hay discontinuidad de salto en Ψ = 0.
        """
        roi_fixed = 1.5
        psi_values = [0.0, _EPS_MACH, _EPS_MACH * 10, 1e-12, 1e-10, 1e-8]
        
        penalties = []
        for psi in psi_values:
            try:
                penalty = standard_tensor.compute_penalty(psi, roi_fixed)
                penalties.append(penalty)
            except Exception as e:
                pytest.fail(f"Excepción en Ψ={psi}: {e}")
        
        # Verificar que todos los valores son finitos
        for p in penalties:
            assert math.isfinite(p), (
                f"Valor no finito encontrado: {p} para algún Ψ cercano a 0"
            )
        
        # Verificar que no hay saltos bruscos (diferencia < 50%)
        for i in range(1, len(penalties)):
            ratio = penalties[i] / max(penalties[i-1], 1e-10)
            assert 0.5 < ratio < 2.0, (
                f"Salto brusco detectado entre Ψ={psi_values[i-1]} y "
                f"Ψ={psi_values[i]}: ratio={ratio:.2f}"
            )

    def test_output_range_bounds(self, standard_tensor: ImprobabilityTensor) -> None:
        """
        Invariante I: Validez del rango de salida.
        
        Teorema: range(F) ⊆ [I_min, I_max]
        
        Verifica que la penalización está siempre dentro del politopo admisible.
        """
        test_cases = [
            (1e-15, 1e15),   # Casos extremos
            (1e15, 1e-15),
            (0.0, 1e15),
            (1e15, 0.1),
            (_EPS_MACH, _IMPROBABILITY_CLAMP_HIGH * 10),
        ]
        
        for psi, roi in test_cases:
            penalty = standard_tensor.compute_penalty(psi, roi)
            
            assert _IMPROBABILITY_CLAMP_LOW <= penalty <= _IMPROBABILITY_CLAMP_HIGH, (
                f"Penalización fuera del rango admisible para "
                f"(Ψ={psi}, ROI={roi}): {penalty}"
            )

    # ─────────────────────────────────────────────────────────────────────────
    # LIPSCHITZ CONTINUIDAD
    # ─────────────────────────────────────────────────────────────────────────

    def test_lipschitz_constant_verification(
        self,
        standard_tensor: ImprobabilityTensor
    ) -> None:
        """
        Invariante H: Verificación de la constante de Lipschitz.
        
        Teorema:
            ||F(x) - F(y)||∞ ≤ L · ||x - y||∞
            donde L = κ · γ · (ROI_max/Ψ_min)^(γ-1)
        """
        # Calcular constante de Lipschitz teórica
        roi_max = 1000.0
        psi_min = 0.01
        L_theoretical = standard_tensor.verify_lipschitz_constant(roi_max, psi_min)
        
        # Verificar empíricamente con muestras aleatorias
        np.random.seed(42)  # Determinismo
        num_samples = 1000
        
        max_violation_ratio = 0.0
        
        for _ in range(num_samples):
            # Generar puntos aleatorios en el dominio
            psi1 = np.random.uniform(psi_min, 10.0)
            psi2 = np.random.uniform(psi_min, 10.0)
            roi1 = np.random.uniform(0.1, roi_max)
            roi2 = np.random.uniform(0.1, roi_max)
            
            # Calcular penalizaciones
            I1 = standard_tensor.compute_penalty(psi1, roi1)
            I2 = standard_tensor.compute_penalty(psi2, roi2)
            
            # Calcular normas
            delta_I = abs(I1 - I2)
            delta_x = max(abs(psi1 - psi2), abs(roi1 - roi2))
            
            # Verificar condición de Lipschitz
            if delta_x > 0:
                local_L = delta_I / delta_x
                violation_ratio = local_L / L_theoretical
                max_violation_ratio = max(max_violation_ratio, violation_ratio)
        
        # La constante empírica no debe exceder la teórica significativamente
        assert max_violation_ratio <= 1.5, (
            f"Violación de Lipschitz: L_empírica/L_teórica = {max_violation_ratio:.4f}. "
            f"Esto indica que la constante de Lipschitz teórica está subestimada."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # COMPOSICIÓN TENSORTIAL
    # ─────────────────────────────────────────────────────────────────────────

    def test_tensor_composition(self) -> None:
        """
        Invariante C': Composición de tensores.
        
        Formalismo:
            Τ₁ ∘ Τ₂ = ImprobabilityTensor(κ₁·κ₂, γ₁+γ₂)
        
        Verifica que el operador __matmul__ implementa correctamente
        la composición funtorial.
        """
        t1 = ImprobabilityTensor(kappa=2.0, gamma=1.5)
        t2 = ImprobabilityTensor(kappa=3.0, gamma=2.0)
        
        t_composed = t1 @ t2
        
        assert t_composed.kappa == pytest.approx(6.0, abs=1e-10), (
            f"Composición de κ fallida: "
            f"κ₁={t1.kappa} × κ₂={t2.kappa} = {t1.kappa * t2.kappa}, "
            f"pero el resultado tiene κ={t_composed.kappa}"
        )
        
        assert t_composed.gamma == pytest.approx(3.5, abs=1e-10), (
            f"Composición de γ fallida: "
            f"γ₁={t1.gamma} + γ₂={t2.gamma} = {t1.gamma + t2.gamma}, "
            f"pero el resultado tiene γ={t_composed.gamma}"
        )

    def test_tensor_composition_associativity(self) -> None:
        """
        Verifica la asociatividad de la composición:
            (Τ₁ ∘ Τ₂) ∘ Τ₃ = Τ₁ ∘ (Τ₂ ∘ Τ₃)
        """
        t1 = ImprobabilityTensor(kappa=1.5, gamma=1.2)
        t2 = ImprobabilityTensor(kappa=2.0, gamma=1.5)
        t3 = ImprobabilityTensor(kappa=3.0, gamma=2.0)
        
        left = (t1 @ t2) @ t3
        right = t1 @ (t2 @ t3)
        
        assert left.kappa == pytest.approx(right.kappa, abs=1e-10)
        assert left.gamma == pytest.approx(right.gamma, abs=1e-10)

    def test_tensor_scalar_multiplication(self) -> None:
        """
        Verifica la multiplicación por escalar:
            (s · Τ).κ = s · Τ.κ
            (s · Τ).γ = Τ.γ
        """
        tensor = ImprobabilityTensor(kappa=2.0, gamma=1.5)
        scalar = 3.0
        
        scaled = scalar * tensor
        
        assert scaled.kappa == pytest.approx(6.0, abs=1e-10)
        assert scaled.gamma == pytest.approx(1.5, abs=1e-10)

    # ─────────────────────────────────────────────────────────────────────────
    # SERIALIZACIÓN
    # ─────────────────────────────────────────────────────────────────────────

    def test_tensor_serialization_roundtrip(self) -> None:
        """
        Verifica la serialización y deserialización completa.
        
        Serialización: Tensor → Dict → JSON → Dict → Tensor
        """
        original = ImprobabilityTensor(kappa=2.5, gamma=3.14159)
        
        # Serialización a Dict
        as_dict = original.to_dict()
        assert isinstance(as_dict, dict)
        assert as_dict["kappa"] == 2.5
        assert as_dict["gamma"] == 3.14159
        
        # Serialización a JSON
        json_str = original.to_json()
        assert isinstance(json_str, str)
        
        # Deserialización desde Dict
        restored_from_dict = ImprobabilityTensor.from_dict(as_dict)
        assert restored_from_dict.kappa == original.kappa
        assert restored_from_dict.gamma == original.gamma
        
        # Deserialización desde JSON
        restored_from_json = ImprobabilityTensor.from_json(json_str)
        assert restored_from_json.kappa == original.kappa
        assert restored_from_json.gamma == original.gamma

    def test_tensor_immutability(self, standard_tensor: ImprobabilityTensor) -> None:
        """
        Verifica la inmutabilidad del tensor (frozen=True).
        
        Un tensor inmutable garantiza transparencia referencial.
        """
        # Intentar modificar kappa (debe fallar)
        with pytest.raises(AttributeError):
            standard_tensor.kappa = 5.0
        
        # Intentar modificar gamma (debe fallar)
        with pytest.raises(AttributeError):
            standard_tensor.gamma = 5.0

    # ─────────────────────────────────────────────────────────────────────────
    # FÁBRICA DE TENSORES
    # ─────────────────────────────────────────────────────────────────────────

    def test_tensor_factory_presets(
        self,
        tensor_factory: Type[TensorFactory]
    ) -> None:
        """
        Verifica los presets predefinidos de la fábrica.
        """
        conservative = tensor_factory.create("conservative")
        moderate = tensor_factory.create("moderate")
        aggressive = tensor_factory.create("aggressive")
        
        # Conservative: κ bajo, γ bajo
        assert conservative.kappa < moderate.kappa
        assert conservative.gamma < moderate.gamma
        
        # Aggressive: κ alto, γ alto
        assert aggressive.kappa > moderate.kappa
        assert aggressive.gamma > moderate.gamma

    def test_tensor_factory_unknown_preset(
        self,
        tensor_factory: Type[TensorFactory]
    ) -> None:
        """
        Verifica que la fábrica rechaza presets desconocidos.
        """
        with pytest.raises(ValueError) as exc_info:
            tensor_factory.create("unknown_preset")
        
        assert "Preset desconocido" in str(exc_info.value)

    # ─────────────────────────────────────────────────────────────────────────
    # ÁLGEBRA TENSORTIAL
    # ─────────────────────────────────────────────────────────────────────────

    def test_tensor_algebra_compose(self) -> None:
        """
        Verifica la composición n-aria de tensores.
        """
        tensors = (
            ImprobabilityTensor(kappa=2.0, gamma=1.5),
            ImprobabilityTensor(kappa=3.0, gamma=2.0),
            ImprobabilityTensor(kappa=1.0, gamma=1.0),
        )
        
        composed = TensorAlgebra.compose(tensors)
        
        expected_kappa = 2.0 * 3.0 * 1.0
        expected_gamma = 1.5 + 2.0 + 1.0
        
        assert composed.kappa == pytest.approx(expected_kappa, abs=1e-10)
        assert composed.gamma == pytest.approx(expected_gamma, abs=1e-10)

    def test_tensor_algebra_compose_empty(self) -> None:
        """
        Verifica la composición de tensor vacío (identidad).
        """
        composed = TensorAlgebra.compose(())
        
        assert composed.kappa == 1.0
        assert composed.gamma == 1.0

    def test_tensor_algebra_average(self) -> None:
        """
        Verifica el promedio ponderado de tensores.
        """
        t1 = ImprobabilityTensor(kappa=1.0, gamma=1.0)
        t2 = ImprobabilityTensor(kappa=3.0, gamma=3.0)
        
        averaged = TensorAlgebra.average((t1, t2))
        
        assert averaged.kappa == pytest.approx(2.0, abs=1e-10)
        assert averaged.gamma == pytest.approx(2.0, abs=1e-10)

    def test_tensor_algebra_interpolate(self) -> None:
        """
        Verifica la interpolación lineal entre tensores.
        """
        t1 = ImprobabilityTensor(kappa=1.0, gamma=1.0)
        t2 = ImprobabilityTensor(kappa=3.0, gamma=3.0)
        
        interpolated = TensorAlgebra.interpolate(t1, t2, t=0.5)
        
        assert interpolated.kappa == pytest.approx(2.0, abs=1e-10)
        assert interpolated.gamma == pytest.approx(2.0, abs=1e-10)


# ════════════════════════════════════════════════════════════════════════════
# SUITE DE TESTS: MÓNADA DE RESULTADO
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.unit
class TestImprobabilityResultMonad:
    """
    Verificación del patrón Either (Mónada de Resultado).
    """

    def test_success_result_construction(self) -> None:
        """
        Verifica la construcción de resultados exitosos.
        """
        result = ImprobabilityResult.success_result(
            penalty=2.5,
            kappa=1.0,
            gamma=2.0,
            psi_input=0.95,
            roi_input=1.5
        )
        
        assert result.success is True
        assert result.improbability_penalty == 2.5
        assert result.is_vetoed is False
        assert result.error_type is None
        assert result.error_message is None

    def test_error_result_construction(self) -> None:
        """
        Verifica la construcción de resultados de error.
        """
        result = ImprobabilityResult.error_result(
            error_type="TypeCoercionError",
            error_message="No se pudo convertir 'psi' a número"
        )
        
        assert result.success is False
        assert result.improbability_penalty is None
        assert result.is_vetoed is None
        assert result.error_type == "TypeCoercionError"
        assert result.error_message == "No se pudo convertir 'psi' a número"

    def test_result_to_dict_success(self) -> None:
        """
        Verifica la serialización a dict de resultados exitosos.
        """
        result = ImprobabilityResult.success_result(
            penalty=100.0,
            kappa=1.0,
            gamma=2.0,
            psi_input=0.1,
            roi_input=10.0
        )
        
        d = result.to_dict()
        
        assert d["success"] is True
        assert d["improbability_penalty"] == 100.0
        assert d["is_vetoed"] is True
        assert "metadata" in d

    def test_result_to_dict_error(self) -> None:
        """
        Verifica la serialización a dict de resultados de error.
        """
        result = ImprobabilityResult.error_result(
            error_type="DimensionalMismatchError",
            error_message="Ψ no puede ser None"
        )
        
        d = result.to_dict()
        
        assert d["success"] is False
        assert d["error_type"] == "DimensionalMismatchError"
        assert d["error"] == "Ψ no puede ser None"


# ════════════════════════════════════════════════════════════════════════════
# SUITE DE TESTS: SERVICIO
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestImprobabilityDriveService:
    """
    Verificación del microservicio de improbabilidad.
    """

    def test_service_initialization(self, base_mic_registry: MICRegistry) -> None:
        """
        Verifica la inicialización del servicio.
        """
        service = ImprobabilityDriveService(
            mic_registry=base_mic_registry,
            kappa=2.0,
            gamma=3.0
        )
        
        assert service.tensor.kappa == 2.0
        assert service.tensor.gamma == 3.0
        assert service.mic is base_mic_registry

    def test_service_update_hyperparameters(
        self,
        base_mic_registry: MICRegistry
    ) -> None:
        """
        Verifica la actualización de hiperparámetros.
        """
        service = ImprobabilityDriveService(
            mic_registry=base_mic_registry
        )
        
        service.update_hyperparameters(kappa=5.0)
        
        assert service.tensor.kappa == 5.0
        assert service.tensor.gamma == 2.0  # Sin cambios

    def test_service_update_invalid_kappa(
        self,
        base_mic_registry: MICRegistry
    ) -> None:
        """
        Verifica rechazo de κ inválido.
        """
        service = ImprobabilityDriveService(
            mic_registry=base_mic_registry
        )
        
        with pytest.raises(ValueError):
            service.update_hyperparameters(kappa=_MAX_KAPPA * 10)

    def test_service_batch_compute(self, standard_tensor: ImprobabilityTensor) -> None:
        """
        Verifica la computación por lotes vectorizada.
        """
        psi_array = np.array([0.5, 1.0, 2.0, 5.0])
        roi_array = np.array([1.0, 1.0, 1.0, 1.0])
        
        penalties = standard_tensor.batch_compute(psi_array, roi_array)
        
        assert isinstance(penalties, np.ndarray)
        assert penalties.shape == psi_array.shape
        
        # Verificar monotonía
        for i in range(len(penalties) - 1):
            assert penalties[i] > penalties[i + 1], (
                f"Monotonía violada en batch: "
                f"psi={psi_array[i]} → {penalties[i]}, "
                f"psi={psi_array[i+1]} → {penalties[i+1]}"
            )

    def test_service_compute_with_gradient(
        self,
        base_mic_registry: MICRegistry
    ) -> None:
        """
        Verifica la computación con gradientes.
        """
        service = ImprobabilityDriveService(
            mic_registry=base_mic_registry
        )
        
        result = service.compute_with_gradient(psi=1.0, roi=2.0)
        
        assert "penalty" in result
        assert "gradients" in result
        assert "d_penalty_d_psi" in result["gradients"]
        assert "d_penalty_d_roi" in result["gradients"]
        
        # Verificar sign consistency
        assert result["gradients"]["d_penalty_d_psi"] < 0  # Decreciente en Ψ
        assert result["gradients"]["d_penalty_d_roi"] > 0  # Creciente en ROI


# ════════════════════════════════════════════════════════════════════════════
# SUITE DE TESTS: CONCURRENCIA Y THREAD SAFETY
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.concurrent
class TestThreadSafety:
    """
    Verificación de thread safety del servicio.
    """

    def test_concurrent_penalty_computation(
        self,
        standard_tensor: ImprobabilityTensor
    ) -> None:
        """
        Verifica que la computación es thread-safe.
        
        Múltiples hilos computan penalizaciones concurrentemente.
        Los resultados deben ser consistentes y no haber race conditions.
        """
        num_threads = 10
        num_iterations = 100
        
        results = [0.0] * num_threads
        errors = []
        
        def compute_worker(thread_id: int) -> None:
            try:
                for _ in range(num_iterations):
                    penalty = standard_tensor.compute_penalty(
                        psi=1.0,
                        roi=2.0
                    )
                    results[thread_id] = penalty
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=compute_worker, args=(i,))
            for i in range(num_threads)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # Verificar que no hubo errores
        assert len(errors) == 0, f"Errores en threads: {errors}"
        
        # Verificar consistencia (todos deben haber calculado el mismo valor)
        expected = standard_tensor.compute_penalty(1.0, 2.0)
        for r in results:
            assert NumericalTolerances.is_close(r, expected), (
                f"Inconsistencia: thread calculó {r}, esperado {expected}"
            )

    def test_concurrent_service_access(
        self,
        base_mic_registry: MICRegistry
    ) -> None:
        """
        Verifica acceso concurrente al servicio a través de MIC.
        """
        service = ImprobabilityDriveService(
            mic_registry=base_mic_registry,
            kappa=1.0,
            gamma=2.0
        )
        service.register_in_mic()
        
        num_threads = 5
        results = []
        errors = []
        
        def mic_worker() -> None:
            try:
                result = base_mic_registry.project_intent(
                    vector_name="compute_improbability_penalty",
                    psi=0.5,
                    roi=1.5
                )
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = [
            threading.Thread(target=mic_worker)
            for _ in range(num_threads)
        ]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0, f"Errores concurrentes: {errors}"
        assert len(results) == num_threads
        
        # Verificar consistencia
        for r in results:
            assert r.get("success") is True
            expected = (1.5 / 0.5) ** 2
            assert NumericalTolerances.is_close(
                r["improbability_penalty"], expected, rel_tol=1e-5
            )


# ════════════════════════════════════════════════════════════════════════════
# SUITE DE TESTS: CASOS EXTREMOS Y EDGE CASES
# ════════════════════════════════════════════════════════════════════════════

@pytest.mark.edge_cases
class TestEdgeCases:
    """
    Verificación de casos extremos y bordes del dominio.
    """

    def test_psi_exactly_zero(self, standard_tensor: ImprobabilityTensor) -> None:
        """
        Verifica el comportamiento cuando Ψ = 0 exactamente.
        """
        penalty = standard_tensor.compute_penalty(psi=0.0, roi=1.0)
        
        # Debe retornar un valor finito (no debe lanzar excepción)
        assert math.isfinite(penalty)
        assert penalty > 0

    def test_roi_approaching_zero(self, standard_tensor: ImprobabilityTensor) -> None:
        """
        Verifica el comportamiento cuando ROI → 0⁺.
        """
        roi_values = [1e-15, 1e-10, 1e-5, 1e-3]
        
        for roi in roi_values:
            penalty = standard_tensor.compute_penalty(psi=1.0, roi=roi)
            
            assert math.isfinite(penalty)
            assert penalty >= _IMPROBABILITY_CLAMP_LOW

    def test_roi_exactly_zero_invalid(self, standard_tensor: ImprobabilityTensor) -> None:
        """
        Verifica que ROI = 0 es rechazado.
        """
        with pytest.raises((AxiomViolationError, ValueError)):
            standard_tensor.compute_penalty(psi=1.0, roi=0.0)

    def test_extreme_ratio(self, standard_tensor: ImprobabilityTensor) -> None:
        """
        Verifica el comportamiento con ratios extremos.
        """
        # ROI/Ψ muy grande
        penalty_large = standard_tensor.compute_penalty(
            psi=1e-15,
            roi=1e15
        )
        assert penalty_large == _IMPROBABILITY_CLAMP_HIGH
        
        # ROI/Ψ muy pequeño
        penalty_small = standard_tensor.compute_penalty(
            psi=1e15,
            roi=1e-15
        )
        assert penalty_small == _IMPROBABILITY_CLAMP_LOW

    def test_nan_propagation_prevention(self, standard_tensor: ImprobabilityTensor) -> None:
        """
        Verifica que NaN no se propaga a través del tensor.
        """
        with pytest.raises(DimensionalMismatchError):
            standard_tensor.compute_penalty(psi=float('nan'), roi=1.0)
        
        with pytest.raises(DimensionalMismatchError):
            standard_tensor.compute_penalty(psi=1.0, roi=float('nan'))

    def test_inf_propagation_prevention(self, standard_tensor: ImprobabilityTensor) -> None:
        """
        Verifica que Inf no se propaga a través del tensor.
        """
        with pytest.raises(DimensionalMismatchError):
            standard_tensor.compute_penalty(psi=float('inf'), roi=1.0)
        
        with pytest.raises(DimensionalMismatchError):
            standard_tensor.compute_penalty(psi=1.0, roi=float('inf'))

    @pytest.mark.parametrize("kappa,gamma", [
        (_MIN_KAPPA, 1.0),
        (_MAX_KAPPA, 1.0),
        (1.0, _MIN_GAMMA),
        (1.0, _MAX_GAMMA),
    ])
    def test_boundary_hyperparameters(
        self,
        kappa: float,
        gamma: float
    ) -> None:
        """
        Verifica el tensor en los bordes del dominio de hiperparámetros.
        """
        tensor = ImprobabilityTensor(kappa=kappa, gamma=gamma)
        
        penalty = tensor.compute_penalty(psi=1.0, roi=1.0)
        
        assert math.isfinite(penalty)
        assert _IMPROBABILITY_CLAMP_LOW <= penalty <= _IMPROBABILITY_CLAMP_HIGH


# ════════════════════════════════════════════════════════════════════════════
# EJECUTOR DE DIAGNÓSTICO
# ════════════════════════════════════════════════════════════════════════════

def run_diagnostic_suite() -> Dict[str, Any]:
    """
    Ejecuta una suite de diagnóstico completa y genera un reporte.
    
    Returns:
        Diccionario con resultados del diagnóstico.
    """
    print("\n" + "=" * 70)
    print("DIAGNÓSTICO COMPLETO DEL MOTOR DE IMPROBABILIDAD")
    print("=" * 70)
    
    results = {
        "tensor_construction": [],
        "monotonicity": [],
        "gradients": [],
        "serialization": [],
        "service": [],
        "summary": {}
    }
    
    # Test de construcción de tensor
    print("\n[1] CONSTRUCCIÓN DE TENSOR")
    try:
        tensor = ImprobabilityTensor(kappa=1.0, gamma=2.0)
        print(f"  ✓ Tensor creado: κ={tensor.kappa}, γ={tensor.gamma}")
        results["tensor_construction"].append("PASS")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results["tensor_construction"].append(f"FAIL: {e}")
    
    # Test de monotonicidad
    print("\n[2] MONOTONICIDAD")
    tensor = ImprobabilityTensor(kappa=1.0, gamma=2.0)
    try:
        # Monotonía en ROI
        p1 = tensor.compute_penalty(1.0, 1.0)
        p2 = tensor.compute_penalty(1.0, 2.0)
        assert p1 < p2, "Monotonía en ROI violada"
        print(f"  ✓ Monotonía en ROI: I(ROI=1)={p1:.4f} < I(ROI=2)={p2:.4f}")
        results["monotonicity"].append("PASS_roi")
        
        # Monotonía en Ψ
        p3 = tensor.compute_penalty(2.0, 1.0)
        p4 = tensor.compute_penalty(0.5, 1.0)
        assert p3 < p4, "Monotonía en Ψ violada"
        print(f"  ✓ Monotonía en Ψ: I(Ψ=2)={p3:.4f} < I(Ψ=0.5)={p4:.4f}")
        results["monotonicity"].append("PASS_psi")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results["monotonicity"].append(f"FAIL: {e}")
    
    # Test de gradientes
    print("\n[3] GRADIENTES")
    try:
        grad_psi, grad_roi = tensor.compute_gradient(1.0, 1.5)
        print(f"  ✓ Gradientes calculados: ∂I/∂Ψ={grad_psi:.6e}, ∂I/∂ROI={grad_roi:.6e}")
        
        passed, details = verify_gradient_numerical(tensor, 1.0, 1.5, rel_tol=1e-4)
        if passed:
            print(f"  ✓ Gradientes analíticos vs numéricos: CONCUERDAN")
            results["gradients"].append("PASS")
        else:
            print(f"  ✗ Gradientes divergentes: {details}")
            results["gradients"].append("FAIL")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results["gradients"].append(f"FAIL: {e}")
    
    # Test de serialización
    print("\n[4] SERIALIZACIÓN")
    try:
        json_str = tensor.to_json()
        restored = ImprobabilityTensor.from_json(json_str)
        assert restored.kappa == tensor.kappa
        assert restored.gamma == tensor.gamma
        print(f"  ✓ Serialización/deserialización exitosa")
        results["serialization"].append("PASS")
    except Exception as e:
        print(f"  ✗ Error: {e}")
        results["serialization"].append(f"FAIL: {e}")
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN DE DIAGNÓSTICO")
    print("=" * 70)
    
    total_tests = sum(len(v) if isinstance(v, list) else 1 for v in results.values())
    total_pass = sum(
        sum(1 for x in v if x == "PASS" or x.startswith("PASS_"))
        if isinstance(v, list)
        else (1 if v == "PASS" else 0)
        for v in results.values()
    )
    
    print(f"Tests ejecutados: {total_tests}")
    print(f"Tests exitosos: {total_pass}")
    print(f"Tests fallidos: {total_tests - total_pass}")
    print(f"Tasa de éxito: {100 * total_pass / max(total_tests, 1):.1f}%")
    
    results["summary"] = {
        "total": total_tests,
        "passed": total_pass,
        "failed": total_tests - total_pass,
        "success_rate": 100 * total_pass / max(total_tests, 1)
    }
    
    return results


if __name__ == "__main__":
    # Ejecutar con pytest si está disponible, o diagnóstico directo
    import sys
    
    if "pytest" in sys.modules or len(sys.argv) > 1:
        pytest.main([__file__, "-v", "--tb=short"])
    else:
        run_diagnostic_suite()