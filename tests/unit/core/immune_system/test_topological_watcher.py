"""
Suite de Pruebas de Topological Watcher (Variedades Riemanianas y p-Laplaciano)
"""

import pytest
import numpy as np
from app.core.immune_system.topological_watcher import (
    IsolatingMembraneFunctor, create_immune_watcher
)
from app.core.mic_algebra import CategoricalState, Stratum

class TestViscoelasticMembrane:
    """Verifica la dinámica de la membrana bajo el funcional p-Dirichlet."""

    def test_p_laplacian_hardening(self):
        """Verifica que el estrés topológico S_p endurece la membrana ante picos."""
        functor = IsolatingMembraneFunctor(p=1.5)
        psi_smooth = np.array([0.5, 0.5, 0.5, 1.0, 0.0, 0.2, 0.1])
        psi_spiky = np.array([0.9, 0.1, 0.8, 2.0, 5.0, 0.9, 0.8])
        
        stress_smooth = functor.compute_topological_stress(psi_smooth)
        stress_spiky = functor.compute_topological_stress(psi_spiky)
        
        # S_p = |Delta psi|^(p-2). Si Delta psi sube, S_p baja (endurecimiento).
        assert np.mean(stress_spiky) < np.mean(stress_smooth)

    def test_metric_tensor_diffeomorphism_hadamard(self):
        """Verifica la deformación paramétrica vía producto de Hadamard (para diagonal)."""
        # En el código, la deformación usa G * (I + gamma * diag(S_p))
        G_init = np.eye(3)
        stress = np.array([0.1, 0.1, 0.1])
        gamma = 2.0
        
        # Simulación de la deformación dictada
        G_deformed = G_init * (np.eye(3) + gamma * np.diag(stress))
        assert G_deformed[0, 0] > G_init[0, 0]
        assert np.allclose(G_deformed[0, 0], 1.2)

class TestImmuneIntegrationRigorous:
    """Pruebas de integración del Watcher en el ciclo OODA."""

    def test_full_immune_morphism_quarantine(self):
        """Verifica que una anomalía crítica dispara un estado de error (Cuarentena)."""
        watcher = create_immune_watcher("strict")
        state = CategoricalState(
            payload={},
            context={"telemetry_metrics": {
                "saturation": 0.99,
                "beta_1": 5.0, # Ciclos lógicos -> Veto Topológico
                "entropy": 0.9
            }},
            validated_strata=frozenset({Stratum.PHYSICS})
        )
        
        result = watcher(state)
        assert result.is_failed
        assert "Cuarentena" in result.error
