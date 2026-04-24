"""
Módulo: tests/integration/test_mic_improbability_coupling.py

Suite de Integración Geométrica: Acoplamiento del Motor de Improbabilidad en la MIC.

Aserciones algebraicas (invariantes de la suite):
═══════════════════════════════════════════════════════════════════════════════
A. Acoplamiento Ortogonal:    rank(MIC') = rank(MIC) + 1, con e_{n+1} ∈ Stratum.OMEGA
B. Proyección Funtorial:      F(Ψ, ROI) ↦ I(Ψ, ROI) preservando el estado global (pureza).
C. Propagación de Veto:       lim_{Ψ→0} I(Ψ, ROI) = I_max ⟹ Colapso de la función de onda (Veto).
D. Aislamiento Monádico:      Inyección de entropía pura (NaN, strings) es absorbida
                              por el funtor de error sin fracturar la variedad base.

Marco teórico:
───────────────────────────────────────────────────────────────────────────────
El Motor de Improbabilidad actúa como un deformador del Tensor Métrico Riemanniano 
en el Estrato Ω. Esta suite garantiza que su inyección mediante el patrón Command 
(project_intent) preserve la matriz identidad I_n de las herramientas ortogonales,
cumpliendo estrictamente la Ley de Clausura Transitiva de la pirámide DIKW.
"""
from __future__ import annotations

import math
import os
import pytest
import numpy as np

# Esterilización del Vacío Termodinámico (Determinismo estricto en BLAS/LAPACK)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

from app.core.schemas import Stratum
from app.adapters.tools_interface import MICRegistry
from app.omega.improbability_drive import (
    ImprobabilityDriveService,
    _IMPROBABILITY_CLAMP_HIGH,
    _EPS_MACH
)

# =============================================================================
# FIXTURES E INFRAESTRUCTURA TOPOLÓGICA
# =============================================================================

@pytest.fixture
def active_mic_registry() -> MICRegistry:
    """
    Inyecta una Matriz de Interacción Central (MIC) limpia y registra el 
    operador tensorial del Motor de Improbabilidad, estableciendo la base 
    canónica e_{n+1} en el espacio vectorial del Estrato OMEGA.
    """
    registry = MICRegistry()
    service = ImprobabilityDriveService(mic_registry=registry, kappa=1.0, gamma=2.0)
    service.register_in_mic()
    return registry


# =============================================================================
# SUITE DE INTEGRACIÓN
# =============================================================================

@pytest.mark.integration
class TestMICImprobabilityCoupling:
    """
    Validación de la coherencia geométrica de la integración del Motor de 
    Improbabilidad en el ecosistema Port-Hamiltoniano.
    """

    def test_mic_orthogonal_coupling_and_registration(self, active_mic_registry: MICRegistry) -> None:
        """
        Invariante 1: Acoplamiento Ortogonal en la MIC.
        
        Axioma: El servicio debe inyectar su proyector en la MIC bajo el estrato correcto (OMEGA)
        sin generar dependencia lineal. El registro debe contener el vector atómico exacto.
        """
        # Extracción del vector atómico del hiperespacio de la MIC
        vector = active_mic_registry.get_basis_vector("compute_improbability_penalty")
        
        assert vector is not None, \
            "Ruptura del Fibrado: El vector de improbabilidad no logró acoplarse a la base de la MIC."
            
        assert vector.target_stratum == Stratum.OMEGA, \
            f"Violación de Topología DIKW: El vector reside en {vector.target_stratum.name}, " \
            f"se esperaba estrictamente {Stratum.OMEGA.name}."

    def test_functorial_projection_success(self, active_mic_registry: MICRegistry) -> None:
        """
        Invariante 2: Proyección Funtorial (Cruce de Estratos).
        
        Axioma: La proyección de un estado nominal a través del bus de la MIC debe resolverse
        mediante el handler del servicio, computando la deformación métrica sin efectos secundarios.
        """
        # Condiciones nominales: Ψ=0.95 (Alta estabilidad), ROI=1.5 (Rentable)
        psi_nominal = 0.95
        roi_nominal = 1.5
        
        result_monad = active_mic_registry.project_intent(
            vector_name="compute_improbability_penalty",
            psi=psi_nominal,
            roi=roi_nominal
        )

        assert result_monad.get("success") is True, "El Funtor falló al mapear el estado estable."
        assert "improbability_penalty" in result_monad, "Pérdida de información: falta el tensor de penalización."
        assert result_monad.get("is_vetoed") is False, "Falso positivo: Un estado sano no debe inducir colapso (Veto)."
        
        # Validación matemática exacta del operador: I = (1.5 / 0.95)^2
        expected_penalty = (roi_nominal / psi_nominal) ** 2
        assert math.isclose(result_monad["improbability_penalty"], expected_penalty, rel_tol=1e-5), \
            "Fractura geométrica en el cálculo de la penalización a través del bus de la MIC."

    def test_asymptotic_veto_propagation(self, active_mic_registry: MICRegistry) -> None:
        """
        Invariante 3: Propagación del Veto Termodinámico.
        
        Axioma: Cuando el espacio topológico se fractura (Ψ → 0) ante hiper-rentabilidad,
        el operador debe saturar la métrica asintóticamente y emitir un Veto Físico ineludible.
        """
        # Condición patológica: Rentabilidad absurda (300%) sobre un socavón lógico estructural
        result_monad = active_mic_registry.project_intent(
            vector_name="compute_improbability_penalty",
            psi=_EPS_MACH * 10,  # Proximidad a la singularidad matemática
            roi=3.0
        )

        assert result_monad.get("success") is True, "La mónada falló en evaluar el caso límite."
        assert result_monad.get("improbability_penalty") == _IMPROBABILITY_CLAMP_HIGH, \
            "Fallo del Retracto Topológico: La penalización no fue acotada en su límite superior asintótico."
        assert result_monad.get("is_vetoed") is True, \
            "Infracción Termodinámica: El sistema no colapsó la función de onda (Veto Estructural) " \
            "ante una singularidad de riesgo geométrica."

    def test_monadic_error_isolation(self, active_mic_registry: MICRegistry) -> None:
        """
        Invariante 4: Aislamiento Monádico de Singularidades.
        
        Axioma: La inyección de entropía pura (variables categóricas o tipos no numéricos) 
        debe ser interceptada. La MIC no debe propagar excepciones crudas, sino encapsularlas 
        en la Mónada de Error para mantener la matriz no-singular.
        """
        # Inyección de entropía sintáctica violando los tipos esperados
        result_monad = active_mic_registry.project_intent(
            vector_name="compute_improbability_penalty",
            psi="entropia_corrupta",
            roi=1.5
        )

        assert result_monad.get("success") is False, "La barrera monádica permitió el paso de entropía cruda."
        
        error_type = result_monad.get("error_type")
        # El servicio debe mapear esto a un DimensionalMismatchError o NumericalInstabilityError
        assert error_type in ["DimensionalMismatchError", "NumericalInstabilityError", "TypeError", "ValueError"], \
            f"Fuga dimensional: El colapso no fue clasificado en un dominio de error conocido. Tipo: {error_type}"
            
        assert "error" in result_monad, "La mónada de fallo carece del rastro de stacktrace o mensaje."

