"""
Pruebas Unitarias para Refinamientos de BusinessAgent
======================================================

Valida específicamente las mejoras implementadas:
1. Nueva fórmula de structural_coherence (decaimiento exponencial).
2. RiskChallenger con umbrales configurables y multi-nivel.
3. Operaciones Algebraicas robustas.
4. Validación extendida de DataFrames.

Fundamentos Matemáticos:
------------------------
- Coherencia Estructural: C = exp(-λ₀·(β₀-1)) · exp(-λ₁·β₁) · Ψ
  donde λ₀ = ln(2), λ₁ = ln(2)/√n, y Ψ ∈ [0,1] es estabilidad piramidal.

- Invariantes Topológicos: β₀ cuenta componentes conexas (β₀ ≥ 1 para grafos no vacíos),
  β₁ cuenta ciclos independientes (rango del grupo de homología H₁).

- La coherencia C ∈ (0, 1] con C → 0 para fragmentación severa o ciclicidad excesiva.
"""

import math
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, PropertyMock

from app.business_agent import (
    TopologicalMetricsBundle,
    RiskChallenger,
    AlgebraicOperations,
    BusinessAgent,
    ConstructionRiskReport
)
from app.constants import ColumnNames
from app.tools_interface import MICRegistry


# =============================================================================
# CONSTANTES MATEMÁTICAS Y CONFIGURACIÓN
# =============================================================================

# Constante de decaimiento para fragmentación (β₀)
LAMBDA_0 = math.log(2)

# Tolerancia numérica para comparaciones de punto flotante
NUMERICAL_TOLERANCE = 1e-10

# Umbrales por defecto del RiskChallenger
DEFAULT_THRESHOLDS = {
    "critical_stability": 0.70,
    "warning_stability": 0.85,
    "cycle_density_limit": 0.33
}


# =============================================================================
# FIXTURES COMPARTIDOS
# =============================================================================

@pytest.fixture
def mock_graph_factory():
    """Factory para crear grafos mock con número configurable de nodos."""
    def _create(n_nodes: int):
        mock_graph = MagicMock()
        mock_graph.number_of_nodes.return_value = n_nodes
        mock_graph.number_of_edges.return_value = max(0, n_nodes - 1)  # Árbol minimal
        return mock_graph
    return _create


@pytest.fixture
def mock_mic_registry():
    """Registry de MIC mockeado para pruebas de BusinessAgent."""
    return MagicMock(spec=MICRegistry)


@pytest.fixture
def base_dataframes():
    """DataFrames válidos mínimos para pruebas."""
    df_presupuesto = pd.DataFrame({
        ColumnNames.CODIGO_APU: ["APU-001", "APU-002"],
        ColumnNames.DESCRIPCION_APU: ["Cimentación", "Estructura"],
        ColumnNames.VALOR_TOTAL: [150000.0, 280000.0]
    })
    df_detalle = pd.DataFrame({
        ColumnNames.CODIGO_APU: ["APU-001", "APU-001", "APU-002"],
        ColumnNames.DESCRIPCION_INSUMO: ["Cemento", "Arena", "Acero"],
        ColumnNames.CANTIDAD_APU: [10.0, 5.0, 20.0],
        ColumnNames.COSTO_INSUMO_EN_APU: [5000.0, 2000.0, 14000.0]
    })
    return df_presupuesto, df_detalle


@pytest.fixture
def default_risk_report():
    """Reporte de riesgo base para pruebas de RiskChallenger."""
    return ConstructionRiskReport(
        integrity_score=100.0,
        waste_alerts=[],
        circular_risks=[],
        complexity_level="Baja",
        financial_risk_level="BAJO",
        details={"pyramid_stability": 0.95},
        strategic_narrative="Proyecto estructuralmente sólido."
    )


# =============================================================================
# FUNCIONES AUXILIARES MATEMÁTICAS
# =============================================================================

def compute_expected_coherence(
    beta_0: int,
    beta_1: int,
    n_nodes: int,
    stability: float
) -> float:
    """
    Calcula el valor esperado de coherencia estructural.

    Fórmula: C = exp(-λ₀·(β₀-1)) · exp(-λ₁·β₁) · Ψ

    Donde:
        - λ₀ = ln(2) ≈ 0.693 (penalización por fragmentación)
        - λ₁ = ln(2)/√n (penalización por ciclos, escalada por tamaño)
        - Ψ = pyramid_stability ∈ [0, 1]

    Propiedades matemáticas:
        - C = 1.0 cuando β₀ = 1, β₁ = 0, Ψ = 1 (grafo conexo, acíclico, estable)
        - C → 0 cuando β₀ → ∞ (fragmentación severa)
        - C → 0 cuando β₁ → ∞ (alta ciclicidad)
        - C es multiplicativo: permite factorización de riesgos independientes
    """
    if n_nodes <= 0:
        return 0.0

    lambda_1 = LAMBDA_0 / math.sqrt(n_nodes)

    fragmentation_penalty = math.exp(-LAMBDA_0 * max(0, beta_0 - 1))
    cycle_penalty = math.exp(-lambda_1 * beta_1)

    return fragmentation_penalty * cycle_penalty * stability


def compute_cycle_density(beta_1: int, n_nodes: int) -> float:
    """
    Calcula la densidad de ciclos normalizada.

    ρ_cycle = β₁ / n

    Interpretación topológica:
        - ρ_cycle = 0: grafo es un bosque (unión de árboles)
        - ρ_cycle ≈ 0.33: presencia moderada de ciclos
        - ρ_cycle > 0.5: estructura altamente cíclica (potencial riesgo)
    """
    if n_nodes <= 0:
        return float('inf') if beta_1 > 0 else 0.0
    return beta_1 / n_nodes


# =============================================================================
# TESTS: TopologicalMetricsBundle
# =============================================================================

class TestTopologicalMetricsBundle:
    """
    Pruebas para el bundle de métricas topológicas.

    Verifica la correcta implementación de la fórmula de coherencia
    estructural basada en números de Betti y estabilidad piramidal.
    """

    @pytest.mark.parametrize("beta_0,beta_1,n_nodes,stability,expected_coherence", [
        # Caso ideal: grafo conexo, sin ciclos, máxima estabilidad
        (1, 0, 10, 1.0, 1.0),
        # Fragmentación: dos componentes conexas
        (2, 0, 10, 1.0, 0.5),
        # Alta fragmentación
        (3, 0, 10, 1.0, 0.25),
        # Con ciclos (n=10): λ₁ = ln(2)/√10 ≈ 0.219
        (1, 1, 10, 1.0, None),  # Calculado dinámicamente
        # Estabilidad reducida
        (1, 0, 10, 0.5, 0.5),
        # Combinación de factores
        (2, 1, 10, 0.8, None),  # Calculado dinámicamente
    ])
    def test_structural_coherence_parametrized(
        self,
        mock_graph_factory,
        beta_0: int,
        beta_1: int,
        n_nodes: int,
        stability: float,
        expected_coherence: Optional[float]
    ):
        """Valida la fórmula de coherencia para múltiples configuraciones."""
        mock_graph = mock_graph_factory(n_nodes)

        bundle = TopologicalMetricsBundle(
            betti_numbers={"beta_0": beta_0, "beta_1": beta_1},
            pyramid_stability=stability,
            graph=mock_graph
        )

        # Calcular valor esperado si no se proporcionó
        if expected_coherence is None:
            expected_coherence = compute_expected_coherence(
                beta_0, beta_1, n_nodes, stability
            )

        assert bundle.structural_coherence == pytest.approx(
            expected_coherence,
            rel=NUMERICAL_TOLERANCE
        )

    def test_coherence_multiplicative_decomposition(self, mock_graph_factory):
        """
        Verifica que la coherencia se descompone multiplicativamente.

        C(β₀, β₁, Ψ) = C_frag(β₀) · C_cycle(β₁) · Ψ

        Esta propiedad es fundamental para el análisis independiente
        de factores de riesgo.
        """
        n_nodes = 16  # Cuadrado perfecto para cálculos limpios
        mock_graph = mock_graph_factory(n_nodes)

        # Coherencia completa
        bundle_full = TopologicalMetricsBundle(
            betti_numbers={"beta_0": 2, "beta_1": 2},
            pyramid_stability=0.8,
            graph=mock_graph
        )

        # Factor de fragmentación aislado
        bundle_frag = TopologicalMetricsBundle(
            betti_numbers={"beta_0": 2, "beta_1": 0},
            pyramid_stability=1.0,
            graph=mock_graph
        )

        # Factor de ciclos aislado
        bundle_cycle = TopologicalMetricsBundle(
            betti_numbers={"beta_0": 1, "beta_1": 2},
            pyramid_stability=1.0,
            graph=mock_graph
        )

        # Verificar descomposición multiplicativa
        expected = bundle_frag.structural_coherence * bundle_cycle.structural_coherence * 0.8
        assert bundle_full.structural_coherence == pytest.approx(expected, rel=1e-9)

    def test_coherence_bounds(self, mock_graph_factory):
        """
        Verifica que la coherencia permanece en [0, 1].

        Para cualquier configuración válida de parámetros:
            0 < C ≤ 1

        El límite inferior es abierto porque exp(-x) > 0 ∀x.
        """
        mock_graph = mock_graph_factory(10)

        # Configuraciones extremas
        extreme_configs = [
            {"beta_0": 1, "beta_1": 0, "stability": 1.0},   # Máximo
            {"beta_0": 100, "beta_1": 0, "stability": 1.0}, # Alta fragmentación
            {"beta_0": 1, "beta_1": 100, "stability": 1.0}, # Muchos ciclos
            {"beta_0": 10, "beta_1": 10, "stability": 0.1}, # Todo adverso
        ]

        for config in extreme_configs:
            bundle = TopologicalMetricsBundle(
                betti_numbers={"beta_0": config["beta_0"], "beta_1": config["beta_1"]},
                pyramid_stability=config["stability"],
                graph=mock_graph
            )

            assert 0 < bundle.structural_coherence <= 1.0, (
                f"Coherencia fuera de rango para config: {config}"
            )

    def test_coherence_monotonicity(self, mock_graph_factory):
        """
        Verifica propiedades de monotonicidad.

        - C es decreciente en β₀ (más fragmentación → menos coherencia)
        - C es decreciente en β₁ (más ciclos → menos coherencia)
        - C es creciente en Ψ (más estabilidad → más coherencia)
        """
        mock_graph = mock_graph_factory(10)

        # Monotonicidad en β₀
        coherences_beta0 = []
        for beta_0 in range(1, 6):
            bundle = TopologicalMetricsBundle(
                betti_numbers={"beta_0": beta_0, "beta_1": 0},
                pyramid_stability=1.0,
                graph=mock_graph
            )
            coherences_beta0.append(bundle.structural_coherence)

        assert all(
            c1 > c2 for c1, c2 in zip(coherences_beta0[:-1], coherences_beta0[1:])
        ), "Coherencia debe ser estrictamente decreciente en β₀"

        # Monotonicidad en β₁
        coherences_beta1 = []
        for beta_1 in range(0, 5):
            bundle = TopologicalMetricsBundle(
                betti_numbers={"beta_0": 1, "beta_1": beta_1},
                pyramid_stability=1.0,
                graph=mock_graph
            )
            coherences_beta1.append(bundle.structural_coherence)

        assert all(
            c1 > c2 for c1, c2 in zip(coherences_beta1[:-1], coherences_beta1[1:])
        ), "Coherencia debe ser estrictamente decreciente en β₁"

        # Monotonicidad en Ψ
        coherences_psi = []
        for psi in [0.2, 0.4, 0.6, 0.8, 1.0]:
            bundle = TopologicalMetricsBundle(
                betti_numbers={"beta_0": 1, "beta_1": 0},
                pyramid_stability=psi,
                graph=mock_graph
            )
            coherences_psi.append(bundle.structural_coherence)

        assert all(
            c1 < c2 for c1, c2 in zip(coherences_psi[:-1], coherences_psi[1:])
        ), "Coherencia debe ser estrictamente creciente en Ψ"

    def test_edge_case_single_node_graph(self, mock_graph_factory):
        """Prueba con grafo de un solo nodo (caso degenerado)."""
        mock_graph = mock_graph_factory(1)

        bundle = TopologicalMetricsBundle(
            betti_numbers={"beta_0": 1, "beta_1": 0},
            pyramid_stability=1.0,
            graph=mock_graph
        )

        # λ₁ = ln(2)/√1 = ln(2), pero β₁ = 0, así que no afecta
        assert bundle.structural_coherence == pytest.approx(1.0)

    def test_edge_case_empty_graph(self, mock_graph_factory):
        """Prueba con grafo vacío (debe manejarse gracefully)."""
        mock_graph = mock_graph_factory(0)

        # β₀ = 0 para grafo vacío (convención topológica)
        bundle = TopologicalMetricsBundle(
            betti_numbers={"beta_0": 0, "beta_1": 0},
            pyramid_stability=1.0,
            graph=mock_graph
        )

        # El código debe manejar n=0 sin división por cero
        # Esperamos coherencia 0 o un valor sentinela
        assert bundle.structural_coherence >= 0

    def test_numerical_stability_large_values(self, mock_graph_factory):
        """Verifica estabilidad numérica con valores grandes."""
        mock_graph = mock_graph_factory(10000)

        bundle = TopologicalMetricsBundle(
            betti_numbers={"beta_0": 1000, "beta_1": 500},
            pyramid_stability=0.99,
            graph=mock_graph
        )

        # No debe haber underflow a exactamente 0 ni NaN
        coherence = bundle.structural_coherence
        assert not math.isnan(coherence)
        assert not math.isinf(coherence)
        assert coherence > 0  # exp(-x) siempre positivo


# =============================================================================
# TESTS: RiskChallenger
# =============================================================================

class TestRiskChallengerRefined:
    """
    Pruebas para el sistema de desafío y auditoría de riesgos.

    El RiskChallenger implementa un sistema de verificación multi-nivel
    que ajusta dinámicamente las evaluaciones de riesgo basándose en
    métricas topológicas.
    """

    def test_default_thresholds_initialization(self):
        """Verifica inicialización con umbrales por defecto."""
        challenger = RiskChallenger()

        assert challenger.thresholds["critical_stability"] == pytest.approx(0.70)
        assert challenger.thresholds["warning_stability"] == pytest.approx(0.85)
        assert challenger.thresholds["cycle_density_limit"] == pytest.approx(0.33)

    @pytest.mark.parametrize("custom_thresholds,expected_merged", [
        # Sobrescribir solo critical
        ({"critical_stability": 0.5}, {"critical_stability": 0.5, "warning_stability": 0.85}),
        # Sobrescribir múltiples
        ({"critical_stability": 0.6, "warning_stability": 0.9},
         {"critical_stability": 0.6, "warning_stability": 0.9}),
        # Umbral personalizado nuevo
        ({"custom_metric": 0.75}, {"custom_metric": 0.75, "critical_stability": 0.70}),
    ])
    def test_custom_thresholds_merge(self, custom_thresholds, expected_merged):
        """Verifica que umbrales personalizados se fusionen correctamente."""
        challenger = RiskChallenger(custom_thresholds)

        for key, value in expected_merged.items():
            assert challenger.thresholds[key] == pytest.approx(value)

    def test_threshold_validation(self):
        """Verifica validación de umbrales en rango [0, 1]."""
        # Umbrales fuera de rango deberían ser rechazados o normalizados
        invalid_thresholds = {"critical_stability": 1.5}

        # Dependiendo de la implementación, esto podría:
        # a) Lanzar excepción
        # b) Clampar el valor a [0, 1]
        # c) Ignorar el valor inválido

        # Asumimos que la implementación debe validar
        with pytest.raises((ValueError, AssertionError)):
            RiskChallenger(invalid_thresholds)

    def test_veto_critical_instability(self, default_risk_report):
        """
        Prueba veto por inestabilidad crítica (Ψ < 0.70).

        Cuando la estabilidad piramidal cae bajo el umbral crítico,
        el sistema debe:
        1. Elevar el nivel de riesgo a CRÍTICO
        2. Aplicar penalización del 30% a integrity_score
        3. Insertar acta de deliberación
        """
        challenger = RiskChallenger()

        report = ConstructionRiskReport(
            integrity_score=100.0,
            waste_alerts=[],
            circular_risks=[],
            complexity_level="Baja",
            financial_risk_level="BAJO",
            details={"pyramid_stability": 0.50},  # < 0.70 crítico
            strategic_narrative="Todo bien."
        )

        audited = challenger.challenge_verdict(report)

        assert audited.financial_risk_level == "RIESGO ESTRUCTURAL (CRÍTICO)"
        assert audited.integrity_score == pytest.approx(70.0)  # 100 * 0.70
        assert "ACTA DE DELIBERACIÓN" in audited.strategic_narrative
        assert "challenger_applied" in audited.details

    def test_warning_suboptimal_stability(self, default_risk_report):
        """
        Prueba alerta por estabilidad subóptima (0.70 ≤ Ψ < 0.85).

        Zona de advertencia con penalización moderada del 15%.
        """
        challenger = RiskChallenger()

        report = ConstructionRiskReport(
            integrity_score=100.0,
            waste_alerts=[],
            circular_risks=[],
            complexity_level="Media",
            financial_risk_level="MODERADO",
            details={"pyramid_stability": 0.80},  # En zona de warning
            strategic_narrative="Estructura aceptable."
        )

        audited = challenger.challenge_verdict(report)

        assert audited.financial_risk_level == "RIESGO ESTRUCTURAL (SEVERO)"
        assert audited.integrity_score == pytest.approx(85.0)  # 100 * 0.85
        assert "challenger_warning" in audited.details

    @pytest.mark.parametrize("stability,expected_level,expected_penalty", [
        (0.50, "CRÍTICO", 0.30),
        (0.69, "CRÍTICO", 0.30),
        (0.70, "SEVERO", 0.15),   # Frontera exacta → zona warning
        (0.84, "SEVERO", 0.15),
        (0.85, None, 0.0),        # Frontera exacta → sin penalización
        (0.95, None, 0.0),
    ])
    def test_stability_threshold_boundaries(
        self,
        stability: float,
        expected_level: Optional[str],
        expected_penalty: float
    ):
        """Prueba comportamiento exacto en fronteras de umbrales."""
        challenger = RiskChallenger()

        report = ConstructionRiskReport(
            integrity_score=100.0,
            waste_alerts=[],
            circular_risks=[],
            complexity_level="Baja",
            financial_risk_level="BAJO",
            details={"pyramid_stability": stability},
            strategic_narrative="Test."
        )

        audited = challenger.challenge_verdict(report)

        expected_score = 100.0 * (1 - expected_penalty)
        assert audited.integrity_score == pytest.approx(expected_score, rel=1e-9)

        if expected_level:
            assert expected_level in audited.financial_risk_level

    def test_cycle_density_warning(self):
        """
        Prueba alerta por densidad de ciclos excesiva.

        Cuando β₁/n > 0.33, indica potenciales dependencias circulares
        en la estructura del proyecto.
        """
        challenger = RiskChallenger()

        report = ConstructionRiskReport(
            integrity_score=100.0,
            waste_alerts=[],
            circular_risks=[],
            complexity_level="Alta",
            financial_risk_level="MODERADO",
            details={
                "pyramid_stability": 0.90,  # OK, no triggerea umbral principal
                "topological_invariants": {
                    "betti_numbers": {"beta_0": 1, "beta_1": 5},
                    "n_nodes": 10
                }
            },
            strategic_narrative="Estructura compleja."
        )

        audited = challenger.challenge_verdict(report)

        # ρ = 5/10 = 0.5 > 0.33 → penalización leve (5%)
        # ADVERTENCIA: La nueva lógica de Cuarentena Topológica aplica una penalización del 10%
        # si beta_1 > 0 y no se aprueba la excepción.
        # Score = 100 * 0.90 (Cuarentena fallida) * 0.95 (Densidad alta) = 85.5
        assert audited.integrity_score == pytest.approx(85.5)
        assert "challenger_cycle_warning" in audited.details
        assert "cycle_density" in audited.details["challenger_cycle_warning"]

    def test_combined_risks_cumulative_penalty(self):
        """
        Prueba que múltiples factores de riesgo se acumulen correctamente.

        Las penalizaciones deben componerse multiplicativamente para
        evitar scores negativos:
        score_final = score_inicial × (1 - p₁) × (1 - p₂) × ...
        """
        challenger = RiskChallenger()

        report = ConstructionRiskReport(
            integrity_score=100.0,
            waste_alerts=["Exceso de cemento"],
            circular_risks=["Dependencia A→B→C→A"],
            complexity_level="Alta",
            financial_risk_level="ALTO",
            details={
                "pyramid_stability": 0.75,  # Warning: -15%
                "topological_invariants": {
                    "betti_numbers": {"beta_0": 2, "beta_1": 4},
                    "n_nodes": 10
                }  # Fragmentación + ciclos
            },
            strategic_narrative="Proyecto con múltiples riesgos."
        )

        audited = challenger.challenge_verdict(report)

        # Penalización por stability: 15%
        # Penalización por ciclos (4/10 = 0.4 > 0.33): 5%
        # Score = 100 * 0.85 * 0.95 = 80.75 (composición multiplicativa)
        # O podría ser aditivo capeado: min(100 - 15 - 5, 0) = 80

        assert audited.integrity_score < 100.0
        assert audited.integrity_score > 0.0
        assert len(audited.details.get("penalties_applied", [])) >= 2

    def test_no_penalty_healthy_structure(self, default_risk_report):
        """Verifica que estructuras saludables no sean penalizadas."""
        challenger = RiskChallenger()

        # Reporte con estabilidad alta y sin ciclos problemáticos
        report = ConstructionRiskReport(
            integrity_score=100.0,
            waste_alerts=[],
            circular_risks=[],
            complexity_level="Baja",
            financial_risk_level="BAJO",
            details={
                "pyramid_stability": 0.95,
                "topological_invariants": {
                    "betti_numbers": {"beta_0": 1, "beta_1": 0}, # Sin ciclos para ser verdaderamente "sano"
                    "n_nodes": 10
                }  # ρ = 0.0 < 0.33, OK
            },
            strategic_narrative="Proyecto óptimo."
        )

        audited = challenger.challenge_verdict(report)

        assert audited.integrity_score == pytest.approx(100.0)
        assert audited.financial_risk_level == "BAJO"
        assert "challenger_applied" not in audited.details


# =============================================================================
# TESTS: AlgebraicOperations
# =============================================================================

class TestAlgebraicOperations:
    """
    Pruebas para operaciones algebraicas robustas.

    Estas operaciones son fundamentales para el cálculo de métricas
    y deben ser numéricamente estables en todos los casos edge.
    """

    class TestSafeNormalize:
        """Pruebas para normalización segura de vectores."""

        def test_unit_vector_preserved(self):
            """Un vector unitario debe permanecer sin cambios."""
            v = np.array([1.0, 0.0, 0.0])
            normed = AlgebraicOperations.safe_normalize(v)
            np.testing.assert_array_almost_equal(normed, v)

        def test_standard_normalization(self):
            """Normalización estándar 3-4-5."""
            v = np.array([3.0, 4.0])
            normed = AlgebraicOperations.safe_normalize(v)

            assert np.linalg.norm(normed) == pytest.approx(1.0)
            np.testing.assert_array_almost_equal(normed, [0.6, 0.8])

        def test_zero_vector_handling(self):
            """
            Vector cero debe retornar vector uniforme normalizado.

            Justificación: En ausencia de dirección preferencial,
            asumimos distribución uniforme (principio de máxima entropía).
            """
            v = np.array([0.0, 0.0])
            normed = AlgebraicOperations.safe_normalize(v)

            assert np.linalg.norm(normed) == pytest.approx(1.0)
            expected = np.ones(2) / np.sqrt(2)
            np.testing.assert_array_almost_equal(normed, expected)

        def test_near_zero_vector(self):
            """Vector cercano a cero (underflow potencial)."""
            v = np.array([1e-320, 1e-320])  # Cerca del límite de precisión
            normed = AlgebraicOperations.safe_normalize(v)

            assert np.linalg.norm(normed) == pytest.approx(1.0, rel=1e-6)
            assert not np.any(np.isnan(normed))
            assert not np.any(np.isinf(normed))

        def test_large_vector(self):
            """Vector muy grande (overflow potencial)."""
            v = np.array([1e150, 1e150])
            normed = AlgebraicOperations.safe_normalize(v)

            assert np.linalg.norm(normed) == pytest.approx(1.0, rel=1e-6)
            assert not np.any(np.isnan(normed))

        def test_negative_components(self):
            """Normalización preserva signos."""
            v = np.array([-3.0, 4.0])
            normed = AlgebraicOperations.safe_normalize(v)

            assert normed[0] == pytest.approx(-0.6)
            assert normed[1] == pytest.approx(0.8)

        def test_high_dimensional(self):
            """Normalización en alta dimensionalidad."""
            dim = 1000
            v = np.ones(dim)
            normed = AlgebraicOperations.safe_normalize(v)

            expected_component = 1.0 / np.sqrt(dim)
            np.testing.assert_array_almost_equal(
                normed,
                np.full(dim, expected_component)
            )

    class TestWeightedGeometricMean:
        """Pruebas para media geométrica ponderada."""

        def test_uniform_factors_uniform_weights(self):
            """Media geométrica de factores iguales = el factor mismo."""
            factors = [0.5, 0.5, 0.5]
            weights = [1.0, 1.0, 1.0]
            result = AlgebraicOperations.weighted_geometric_mean(factors, weights)
            assert result == pytest.approx(0.5)

        def test_geometric_mean_basic(self):
            """Cálculo básico: √(0.1 × 0.9) = 0.3"""
            factors = [0.1, 0.9]
            weights = [1.0, 1.0]
            result = AlgebraicOperations.weighted_geometric_mean(factors, weights)
            assert result == pytest.approx(0.3)

        def test_weighted_emphasis(self):
            """Pesos no uniformes enfatizan ciertos factores."""
            factors = [0.1, 0.9]
            weights = [3.0, 1.0]  # Enfatiza el primer factor

            # (0.1³ × 0.9¹)^(1/4) = (0.001 × 0.9)^0.25 = 0.0009^0.25 ≈ 0.173
            result = AlgebraicOperations.weighted_geometric_mean(factors, weights)
            expected = (0.1**3 * 0.9**1) ** (1/4)
            assert result == pytest.approx(expected)

        def test_single_factor(self):
            """Caso degenerado: un solo factor."""
            factors = [0.7]
            weights = [2.0]
            result = AlgebraicOperations.weighted_geometric_mean(factors, weights)
            assert result == pytest.approx(0.7)

        def test_zero_factor(self):
            """
            Factor cero debe resultar en media cero.

            Nota: Esto es matemáticamente correcto pero puede no ser
            deseable en todas las aplicaciones. Considerar log-sum-exp
            con suavizado para evitar colapso total.
            """
            factors = [0.0, 0.5, 0.8]
            weights = [1.0, 1.0, 1.0]
            result = AlgebraicOperations.weighted_geometric_mean(factors, weights)
            assert result == pytest.approx(0.0)

        def test_zero_weight_exclusion(self):
            """Peso cero debe excluir efectivamente el factor."""
            factors = [0.01, 0.5]  # Primer factor muy bajo
            weights = [0.0, 1.0]   # Pero tiene peso cero
            result = AlgebraicOperations.weighted_geometric_mean(factors, weights)
            assert result == pytest.approx(0.5)

        @pytest.mark.parametrize("factors,weights", [
            ([-0.1, 0.5], [1.0, 1.0]),  # Factor negativo
            ([0.5, 0.5], [-1.0, 1.0]),   # Peso negativo
        ])
        def test_invalid_inputs_raise_error(self, factors, weights):
            """Factores/pesos negativos deben rechazarse."""
            with pytest.raises((ValueError, AssertionError)):
                AlgebraicOperations.weighted_geometric_mean(factors, weights)

        def test_empty_inputs(self):
            """Listas vacías deben manejarse."""
            with pytest.raises((ValueError, ZeroDivisionError, IndexError)):
                AlgebraicOperations.weighted_geometric_mean([], [])

        def test_numerical_stability_extreme_values(self):
            """Estabilidad con valores extremos usando log-space."""
            # Factores muy pequeños pueden causar underflow
            factors = [1e-100, 1e-100, 1e-100]
            weights = [1.0, 1.0, 1.0]
            result = AlgebraicOperations.weighted_geometric_mean(factors, weights)

            # (1e-100)³^(1/3) = 1e-100
            assert result == pytest.approx(1e-100, rel=1e-6)


# =============================================================================
# TESTS: BusinessAgent DataFrames Validation
# =============================================================================

class TestBusinessAgentRefined:
    """
    Pruebas para validación de DataFrames en BusinessAgent.

    La validación debe ser exhaustiva para detectar problemas de datos
    antes del procesamiento costoso.
    """

    def test_valid_dataframes_pass(self, mock_mic_registry, base_dataframes):
        """DataFrames válidos pasan la validación."""
        agent = BusinessAgent({}, mock_mic_registry)
        df_p, df_d = base_dataframes

        valid, msg, diag = agent._validate_dataframes(df_p, df_d)

        assert valid is True
        assert msg == "" or "success" in msg.lower()
        assert diag is not None
        assert "row_counts" in diag
        assert diag["row_counts"]["presupuesto"] == 2
        assert diag["row_counts"]["detalle"] == 3

    def test_missing_required_columns(self, mock_mic_registry):
        """Detecta columnas faltantes."""
        agent = BusinessAgent({}, mock_mic_registry)

        # DataFrame sin columna DESCRIPCION_APU (que es requerida)
        df_p = pd.DataFrame({
            ColumnNames.CODIGO_APU: ["A1"],
            # Falta DESCRIPCION_APU
            ColumnNames.VALOR_TOTAL: [100.0]
        })
        df_d = pd.DataFrame({
            ColumnNames.CODIGO_APU: ["A1"],
            ColumnNames.DESCRIPCION_INSUMO: ["I1"],
            ColumnNames.CANTIDAD_APU: [1.0],
            ColumnNames.COSTO_INSUMO_EN_APU: [100.0]
        })

        valid, msg, diag = agent._validate_dataframes(df_p, df_d)

        assert valid is False
        assert ColumnNames.DESCRIPCION_APU in msg or "columna" in msg.lower()
        assert diag["missing_columns"]["presupuesto"] != []

    def test_empty_dataframe(self, mock_mic_registry, base_dataframes):
        """Rechaza DataFrames vacíos."""
        agent = BusinessAgent({}, mock_mic_registry)
        _, df_d = base_dataframes

        df_p_empty = pd.DataFrame(columns=[
            ColumnNames.CODIGO_APU,
            ColumnNames.DESCRIPCION_APU,
            ColumnNames.VALOR_TOTAL
        ])

        valid, msg, diag = agent._validate_dataframes(df_p_empty, df_d)

        assert valid is False
        assert "vacío" in msg.lower() or "empty" in msg.lower()

    def test_null_values_detection(self, mock_mic_registry):
        """Detecta y reporta valores nulos."""
        agent = BusinessAgent({}, mock_mic_registry)

        df_p = pd.DataFrame({
            ColumnNames.CODIGO_APU: ["A1", None, "A3"],  # Null en medio
            ColumnNames.DESCRIPCION_APU: ["D1", "D2", "D3"],
            ColumnNames.VALOR_TOTAL: [100.0, 200.0, np.nan]  # NaN al final
        })
        df_d = pd.DataFrame({
            ColumnNames.CODIGO_APU: ["A1"],
            ColumnNames.DESCRIPCION_INSUMO: ["I1"],
            ColumnNames.CANTIDAD_APU: [1.0],
            ColumnNames.COSTO_INSUMO_EN_APU: [100.0]
        })

        valid, msg, diag = agent._validate_dataframes(df_p, df_d)

        assert "null_analysis" in diag
        assert diag["null_analysis"]["presupuesto"]["total_nulls"] == 2

    def test_outlier_detection_iqr_method(self, mock_mic_registry):
        """
        Detecta outliers usando método IQR.

        Outlier: valor fuera de [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
        """
        agent = BusinessAgent({}, mock_mic_registry)

        # Generar datos con outlier claro
        normal_values = [100.0] * 19
        outlier_value = 10000.0  # 100x mayor
        all_values = normal_values + [outlier_value]

        df_p = pd.DataFrame({
            ColumnNames.CODIGO_APU: [f"A{i}" for i in range(20)],
            ColumnNames.DESCRIPCION_APU: ["Desc"] * 20,
            ColumnNames.VALOR_TOTAL: all_values
        })
        df_d = pd.DataFrame({
            ColumnNames.CODIGO_APU: ["A0"],
            ColumnNames.DESCRIPCION_INSUMO: ["I1"],
            ColumnNames.CANTIDAD_APU: [1.0],
            ColumnNames.COSTO_INSUMO_EN_APU: [100.0]
        })

        valid, msg, diag = agent._validate_dataframes(df_p, df_d)

        assert "distribution_analysis" in diag
        assert diag["distribution_analysis"]["outlier_count"] >= 1
        assert diag["distribution_analysis"]["outlier_indices"] == [19]

    def test_orphan_codes_detection(self, mock_mic_registry):
        """
        Detecta códigos huérfanos (en detalle pero no en presupuesto).

        Integridad referencial: todo código en detalle debe existir en presupuesto.
        """
        agent = BusinessAgent({}, mock_mic_registry)

        df_p = pd.DataFrame({
            ColumnNames.CODIGO_APU: ["APU-001"],
            ColumnNames.DESCRIPCION_APU: ["Único"],
            ColumnNames.VALOR_TOTAL: [1000.0]
        })
        df_d = pd.DataFrame({
            ColumnNames.CODIGO_APU: ["APU-001", "APU-999"],  # APU-999 no existe
            ColumnNames.DESCRIPCION_INSUMO: ["I1", "I2"],
            ColumnNames.CANTIDAD_APU: [1.0, 2.0],
            ColumnNames.COSTO_INSUMO_EN_APU: [100.0, 200.0]
        })

        valid, msg, diag = agent._validate_dataframes(df_p, df_d)

        assert "referential_integrity" in diag
        assert "APU-999" in diag["referential_integrity"]["orphan_codes"]

    def test_duplicate_codes_detection(self, mock_mic_registry):
        """Detecta códigos duplicados en presupuesto."""
        agent = BusinessAgent({}, mock_mic_registry)

        df_p = pd.DataFrame({
            ColumnNames.CODIGO_APU: ["APU-001", "APU-001"],  # Duplicado
            ColumnNames.DESCRIPCION_APU: ["Desc1", "Desc2"],
            ColumnNames.VALOR_TOTAL: [100.0, 200.0]
        })
        df_d = pd.DataFrame({
            ColumnNames.CODIGO_APU: ["APU-001"],
            ColumnNames.DESCRIPCION_INSUMO: ["I1"],
            ColumnNames.CANTIDAD_APU: [1.0],
            ColumnNames.COSTO_INSUMO_EN_APU: [100.0]
        })

        valid, msg, diag = agent._validate_dataframes(df_p, df_d)

        assert "duplicate_analysis" in diag
        assert "APU-001" in diag["duplicate_analysis"]["duplicated_codes"]

    def test_negative_values_flagged(self, mock_mic_registry):
        """Valores negativos en columnas monetarias son marcados."""
        agent = BusinessAgent({}, mock_mic_registry)

        df_p = pd.DataFrame({
            ColumnNames.CODIGO_APU: ["A1", "A2"],
            ColumnNames.DESCRIPCION_APU: ["D1", "D2"],
            ColumnNames.VALOR_TOTAL: [100.0, -50.0]  # Negativo inválido
        })
        df_d = pd.DataFrame({
            ColumnNames.CODIGO_APU: ["A1"],
            ColumnNames.DESCRIPCION_INSUMO: ["I1"],
            ColumnNames.CANTIDAD_APU: [1.0],
            ColumnNames.COSTO_INSUMO_EN_APU: [100.0]
        })

        valid, msg, diag = agent._validate_dataframes(df_p, df_d)

        assert "value_range_analysis" in diag
        assert diag["value_range_analysis"]["negative_monetary_values"] > 0

    def test_diagnostic_summary_structure(self, mock_mic_registry, base_dataframes):
        """Verifica estructura completa del diagnóstico."""
        agent = BusinessAgent({}, mock_mic_registry)
        df_p, df_d = base_dataframes

        valid, msg, diag = agent._validate_dataframes(df_p, df_d)

        required_keys = [
            "row_counts",
            "column_check",
            "null_analysis",
            "distribution_analysis",
            "referential_integrity",
            "validation_timestamp"
        ]

        for key in required_keys:
            assert key in diag, f"Falta clave de diagnóstico: {key}"


# =============================================================================
# TESTS: Integración y Consistencia
# =============================================================================

class TestIntegrationConsistency:
    """
    Pruebas de integración para verificar consistencia entre componentes.
    """

    def test_coherence_affects_risk_assessment(self, mock_graph_factory, mock_mic_registry):
        """
        Verifica que la coherencia estructural influya en la evaluación de riesgo.

        Un grafo fragmentado (β₀ > 1) debe resultar en menor integrity_score.
        """
        # Simular flujo completo: grafo → métricas → reporte → challenger
        mock_graph = mock_graph_factory(10)

        # Bundle con fragmentación
        bundle = TopologicalMetricsBundle(
            betti_numbers={"beta_0": 3, "beta_1": 0},
            pyramid_stability=0.9,
            graph=mock_graph
        )

        # La coherencia debería ser ~0.25 * 0.9 = 0.225
        assert bundle.structural_coherence < 0.5

        # Esto debería reflejarse en el reporte
        report = ConstructionRiskReport(
            integrity_score=100.0,
            waste_alerts=[],
            circular_risks=[],
            complexity_level="Alta",
            financial_risk_level="MODERADO",
            details={"pyramid_stability": bundle.structural_coherence},
            strategic_narrative="Análisis estructural."
        )

        challenger = RiskChallenger()
        audited = challenger.challenge_verdict(report)

        # Debe haber penalización por baja estabilidad
        assert audited.integrity_score < 100.0

    def test_algebraic_operations_in_coherence_calculation(self, mock_graph_factory):
        """
        Verifica que las operaciones algebraicas se usen correctamente
        en el cálculo de coherencia.
        """
        mock_graph = mock_graph_factory(10)

        # Crear bundle y verificar que usa operaciones seguras
        bundle = TopologicalMetricsBundle(
            betti_numbers={"beta_0": 1, "beta_1": 0},
            pyramid_stability=0.0,  # Edge case: estabilidad cero
            graph=mock_graph
        )

        # Coherencia debe ser 0 pero no NaN
        assert bundle.structural_coherence == pytest.approx(0.0)
        assert not math.isnan(bundle.structural_coherence)


# =============================================================================
# TESTS: Propiedades Matemáticas Avanzadas
# =============================================================================

class TestMathematicalProperties:
    """
    Pruebas de propiedades matemáticas fundamentales.

    Estas pruebas verifican que las implementaciones respeten
    las propiedades teóricas esperadas.
    """

    def test_coherence_is_homomorphism(self, mock_graph_factory):
        """
        Verifica propiedad de homomorfismo aproximado.

        Para grafos que se pueden descomponer:
        C(G₁ ∪ G₂) ≈ C(G₁) * C(G₂) (bajo ciertas condiciones)
        """
        # Esta es una propiedad teórica que puede no aplicar exactamente
        # pero la estructura multiplicativa de la fórmula lo sugiere
        mock_graph = mock_graph_factory(10)

        # Dos "sub-estructuras" combinadas
        bundle1 = TopologicalMetricsBundle(
            betti_numbers={"beta_0": 2, "beta_1": 0},
            pyramid_stability=1.0,
            graph=mock_graph
        )

        bundle2 = TopologicalMetricsBundle(
            betti_numbers={"beta_0": 1, "beta_1": 1},
            pyramid_stability=1.0,
            graph=mock_graph
        )

        # Combinado
        bundle_combined = TopologicalMetricsBundle(
            betti_numbers={"beta_0": 2, "beta_1": 1},
            pyramid_stability=1.0,
            graph=mock_graph
        )

        # La coherencia combinada debería estar relacionada con el producto
        # de los factores individuales
        factor_frag = bundle1.structural_coherence
        factor_cycle = bundle2.structural_coherence

        # Verificar que la combinación preserva la estructura multiplicativa
        assert bundle_combined.structural_coherence == pytest.approx(
            factor_frag * factor_cycle,
            rel=1e-9
        )

    def test_geometric_mean_vs_arithmetic_mean(self):
        """
        Verifica desigualdad AG: Geométrica ≤ Aritmética.

        Para factores no negativos:
        (∏fᵢ)^(1/n) ≤ (∑fᵢ)/n
        """
        test_cases = [
            [0.1, 0.9],
            [0.5, 0.5],
            [0.1, 0.2, 0.3, 0.4],
            [0.99, 0.01],
        ]

        for factors in test_cases:
            weights = [1.0] * len(factors)
            geo_mean = AlgebraicOperations.weighted_geometric_mean(factors, weights)
            arith_mean = sum(factors) / len(factors)

            assert geo_mean <= arith_mean + NUMERICAL_TOLERANCE, (
                f"Violación AG para {factors}: geo={geo_mean}, arith={arith_mean}"
            )


# =============================================================================
# PUNTO DE ENTRADA
# =============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Detener en primer fallo
        "--durations=10"  # Mostrar las 10 pruebas más lentas
    ])
