import pytest
import math
import networkx as nx
import numpy as np
from typing import Dict, Any

from app.core.schemas import Stratum
from app.agents.MIC_agent import MICAgent, ImpedanceMatchStatus
from unittest.mock import MagicMock
from app.tactics.business_topology import BusinessTopologicalAnalyzer, TopologyAnalyzerConfig

# =============================================================================
# CONSTANTS
# =============================================================================
_SPECTRAL_ZERO_TOLERANCE = 1e-10
_FIEDLER_FRACTURE_THRESHOLD = 0.5
_CRITICAL_THERMAL_STRESS = 65.0
_COLLAPSED_FINANCIAL_INERTIA = 0.1

@pytest.mark.integration
class TestAdversarialHomologyAndThermalCollapse:
    """
    Test de Integración de Estrés Dinámico:
    Valida la resistencia categórica frente a una discrepancia homológica (Mayer-Vietoris)
    y colapso térmico simultáneo, aplicando rigor analítico estricto.
    """

    def test_adversarial_homology_exactness(self):
        """
        Valida que β1 se calcula de manera determinista vía la característica de Euler
        (χ = V - E = β0 - β1), rechazando la enumeración NP-Hard de ciclos.
        """
        # Crear un complejo simplicial D1 (grafo) adversarial
        # V = 4, E = 5 -> χ = 4 - 5 = -1
        # β0 (Componentes débilmente conexas) = 1
        # Por Teorema Euler-Poincaré: β1 = E - V + β0 = 5 - 4 + 1 = 2
        graph = nx.DiGraph()
        graph.add_edges_from([
            ("A", "B"), ("B", "C"), ("C", "A"), # Ciclo 1
            ("C", "D"), ("D", "A")              # Ciclo 2
        ])

        V = graph.number_of_nodes()
        E = graph.number_of_edges()

        # Compute exact β0 via O(V+E) weak connectivity
        beta_0 = nx.number_weakly_connected_components(graph)
        assert beta_0 == 1, "Beta_0 no es 1 para grafo conexo"

        # Invariante exacto Euler-Poincaré:
        beta_1_exact = E - V + beta_0
        assert beta_1_exact == 2, "La biyección homológica exacta (χ) falló"

        # Validar a través del analizador (usará su propia aproximación o wrapper)
        analyzer = BusinessTopologicalAnalyzer(
            telemetry=MagicMock(),
            config=TopologyAnalyzerConfig()
        )
        metrics = analyzer.calculate_betti_numbers(graph)

        # Extraemos el beta_1 reportado (dependiendo si es dict o dataclass)
        reported_beta_1 = getattr(metrics, "beta_1", metrics.get("beta_1", 0) if isinstance(metrics, dict) else 0)
        assert reported_beta_1 == beta_1_exact, f"El analyzer reporta {reported_beta_1} vs exacto {beta_1_exact}"

    def test_spectral_analysis_rigor(self):
        """
        Demuestra la integridad del operador Laplaciano normalizado.
        - Todos los autovalores λ_i ∈ [0, 2]
        - La multiplicidad algebraica de λ=0 es igual a β0
        """
        graph = nx.DiGraph()
        graph.add_edges_from([("A", "B"), ("B", "C")])
        graph.add_node("D") # Componente aislada (β0 = 2)

        beta_0 = nx.number_weakly_connected_components(graph)
        assert beta_0 == 2

        # Obtener grafo no dirigido para Laplaciano espectral
        undirected_graph = graph.to_undirected()

        # Calcular Laplaciano Normalizado y Espectro (eigvalsh)
        L_norm = nx.normalized_laplacian_matrix(undirected_graph).toarray()
        eigenvalues = np.linalg.eigvalsh(L_norm)

        # Invariantes Espectrales
        for lambda_i in eigenvalues:
            assert -_SPECTRAL_ZERO_TOLERANCE <= lambda_i <= 2 + _SPECTRAL_ZERO_TOLERANCE, \
                f"Eigenvalor {lambda_i} excede la cota de Chung [0, 2]"

        # Multiplicidad del autovalor cero
        zero_eigenvalues = sum(1 for val in eigenvalues if abs(val) <= _SPECTRAL_ZERO_TOLERANCE)
        assert zero_eigenvalues == beta_0, "Matriz Laplaciana corrupta: dim(Ker(L)) != β0"

        # Fiedler Value (segundo autovalor más pequeño si conexo, o el de menor no nulo)
        fiedler = eigenvalues[1] if len(eigenvalues) > 1 else 0.0
        assert fiedler < _FIEDLER_FRACTURE_THRESHOLD, "Fractura estructural no detectada correctamente"

    def test_mic_agent_thermal_collapse_and_veto(self):
        """
        El RiskChallenger y el MICAgent interceptan la mónada de error en la telemetría,
        y aplican un veto categórico irrompible ante el colapso del filtro EKF.
        """
        # Configuramos la "Fiebre Inflacionaria" extrema
        adversarial_context = {
            "system_temperature": _CRITICAL_THERMAL_STRESS,
            "financial_inertia": _COLLAPSED_FINANCIAL_INERTIA,
            "laplacian_fiedler": 0.0001,
            "betti_numbers": {"beta_1": 2, "beta_0": 1},
            "isolated_nodes": [],
            # Trazas de falla del EKF/Lyapunov (entropía física inmanejable)
            "ekf_innovation_divergence": True,
            "max_lyapunov_exponent": 1.5 # λ > 0 (caos inmanejable)
        }

        mock_mic_registry = MagicMock()
        mock_mic_registry.get_vector_info.return_value = {"stratum": Stratum.STRATEGY}
        agent = MICAgent(mic_registry=mock_mic_registry)

        llm_output_strategy = {
            "territorial_friction": 0.5, # Debería ser >= 1.0 según contrato STRATEGY
            "risk_coupling": 0.99
        }

        result = agent.encapsulate_monad(
            target_vector="Riemannian_Friction_Contract",
            llm_output=llm_output_strategy,
            validated_strata=frozenset([Stratum.PHYSICS, Stratum.TACTICS]),
            raw_telemetry=adversarial_context,
            force_override=True
        )

        # Validar el Supremo categórico
        assert result.is_failed, "El sistema no vetó un colapso térmico acoplado"
        assert result.error == ImpedanceMatchStatus.ALGEBRAIC_VETO.value

        # Sello forense en el AuditTrail preservando la entropía de Shannon
        audits = agent.get_recent_audits(1)
        assert len(audits) == 1

        audit = audits[0]
        # Recuperamos usando diccionario o atributo
        status = audit.get("impedance_match_status") if hasattr(audit, "get") else getattr(audit, "impedance_match_status").value
        assert status == ImpedanceMatchStatus.ALGEBRAIC_VETO.value

        errors = audit.get("validation_errors", []) if hasattr(audit, "get") else getattr(audit, "validation_errors")
        assert len(errors) > 0
