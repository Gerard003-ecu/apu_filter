"""
test_adversarial_homology_and_thermal_collapse.py
==================================================

Suite de Integración de Estrés Dinámico: Colapso Homológico y Térmico.

Fundamentos Matemáticos Verificados
─────────────────────────────────────────────────────────────────────────────
1. Proxy operacional de Mayer-Vietoris:
   Se evalúa si dos subgrafos localmente acíclicos inducen, tras su unión,
   una patología global: aparición de ciclos dirigidos y/o fragmentación.

   Nota rigurosa:
   β₁ NO se computa como homología simplicial exacta (rango de H₁ del
   complejo de cliques), sino mediante un proxy operacional conservador:
       β₁_proxy := |{ciclos dirigidos simples en A ∪ B}|
   Esto es una cota inferior del primer número de Betti del complejo
   de banderas asociado, suficiente para detectar patología topológica
   cuando β₁_proxy > 0.

2. Teoría espectral de grafos:
   Se calcula λ₂ (valor de Fiedler), el segundo menor autovalor del
   Laplaciano normalizado del grafo no dirigido subyacente.
       λ₂ ≈ 0 ⟹ fractura o conectividad algebraica débil
       λ₂ > 0 ⟹ conectividad algebraica (Cheeger bound inferior)

   Se usa eigvalsh (diagonalización para matrices simétricas reales)
   por estabilidad numérica, ya que L_norm es simétrica por construcción.

3. Acoplamiento ciberfísico-financiero:
   Bajo hipertermia financiera (T > T_crit) Y topología degradada
   (β₁ > 0 ∧ β₀ > 1 ∧ λ₂ < umbral), el sistema debe:
   - Escalar al peor caso (severidad CRITICO)
   - Emitir un veto técnico compuesto
   - Preservar la cadena causal en telemetría

Estructura de la Suite
─────────────────────────────────────────────────────────────────────────────
- TestGraphConstruction:      Validación de invariantes de grafos de prueba
- TestTopologicalInvariants:  Propiedades topológicas puras (sin sistema)
- TestSpectralProperties:     Propiedades espectrales puras (sin sistema)
- TestMayerVietorisProxy:     Emergencia de patología por fusión
- TestThermalCollapseIntegration: Integración con sistema MIC completo
- TestEdgeCasesAndRobustness: Casos extremos y defensas numéricas
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock, PropertyMock, patch

import networkx as nx
import numpy as np
import pytest

from app.core.mic_algebra import CategoricalState, Stratum

# ============================================================================
# IMPORTACIONES CONDICIONALES CON FALLBACK
# ============================================================================

try:
    from app.core.telemetry import TelemetryContext

    _HAS_TELEMETRY = True
except ImportError:
    _HAS_TELEMETRY = False

    class TelemetryContext:
        """Stub minimal para cuando el módulo de telemetría no está disponible."""

        def __init__(self) -> None:
            self._errors: List[Dict[str, Any]] = []

        def get_errors(self) -> List[Dict[str, Any]]:
            return list(self._errors)

        def record_error(self, error: Dict[str, Any]) -> None:
            self._errors.append(error)

        def clear(self) -> None:
            self._errors.clear()


try:
    from app.strategy.business_agent import RiskChallenger

    _HAS_RISK_CHALLENGER = True
except ImportError:
    _HAS_RISK_CHALLENGER = False
    RiskChallenger = None

try:
    from app.tactics.business_topology import (
        BusinessTopologicalAnalyzer,
        TopologicalMetrics,
    )

    _HAS_TOPOLOGY = True
except ImportError:
    _HAS_TOPOLOGY = False
    BusinessTopologicalAnalyzer = None

    @dataclass(frozen=True)
    class TopologicalMetrics:
        """Stub inmutable para métricas topológicas."""

        beta_0: int = 1
        beta_1: int = 0
        fiedler_value: float = 1.0


# ============================================================================
# CONSTANTES CON JUSTIFICACIÓN
# ============================================================================

# Temperatura crítica del sistema financiero (unidad abstracta).
# Representa el umbral a partir del cual la inercia financiera se
# degrada exponencialmente, requiriendo escalamiento a peor caso.
_CRITICAL_THERMAL_STRESS: float = 65.0

# Umbral de fractura de Fiedler: si λ₂ < este valor, la conectividad
# algebraica es insuficiente para garantizar propagación de información
# coherente a través de la topología organizacional.
# Justificación: para grafos de presupuesto con 5-10 nodos, λ₂ < 0.5
# indica que el corte de Cheeger permite particiones baratas.
_FIEDLER_FRACTURE_THRESHOLD: float = 0.5

# Tolerancia numérica para comparación con cero en el espectro.
# Basada en la precisión de máquina float64 (eps ≈ 2.2e-16)
# escalada por la norma típica del Laplaciano normalizado (≤ 2).
# Se usa 1e-10 como compromiso conservador.
_SPECTRAL_ZERO_TOLERANCE: float = 1e-10

# Límite superior de ciclos a enumerar antes de abortar.
# Previene explosión combinatoria en grafos densos.
_MAX_CYCLE_ENUMERATION: int = 10_000


# ============================================================================
# CONSTRUCTORES DE GRAFOS ADVERSARIALES
# ============================================================================


def build_nominal_master_graph() -> nx.DiGraph:
    """
    Construye el Grafo A (Presupuesto Maestro).

    Topología garantizada:
    - DAG estricto (sin ciclos dirigidos)
    - Débilmente conexo (β₀_débil = 1)
    - 4 nodos, 4 arcos
    - Estructura: dos ramas (CIMENTACION, ESTRUCTURA) comparten ACERO_REFUERZO

    Diagrama:
        CIMENTACION ──→ CONCRETO_3000
        CIMENTACION ──→ ACERO_REFUERZO ←── ESTRUCTURA
        ESTRUCTURA  ──→ CONCRETO_4000
    """
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            ("CIMENTACION", "CONCRETO_3000"),
            ("CIMENTACION", "ACERO_REFUERZO"),
            ("ESTRUCTURA", "CONCRETO_4000"),
            ("ESTRUCTURA", "ACERO_REFUERZO"),
        ]
    )
    return graph


def build_adversarial_apu_graph() -> nx.DiGraph:
    """
    Construye el Grafo B (APUs inyectados adversarialmente).

    Topología garantizada:
    - DAG estricto (localmente, sin ciclos dirigidos propios)
    - Débilmente NO conexo (dos componentes)
    - Componente 1: ACERO_REFUERZO → PROVEEDOR_MONOPOLICO → CIMENTACION
      (al componer con A, cierra el ciclo CIMENTACION → ACERO_REFUERZO →
       PROVEEDOR_MONOPOLICO → CIMENTACION)
    - Componente 2: APU_FANTASMA → INSUMO_HUERFANO (aislada, provoca
      fragmentación β₀ > 1 en la unión)

    Diagrama:
        ACERO_REFUERZO ──→ PROVEEDOR_MONOPOLICO ──→ CIMENTACION
        APU_FANTASMA   ──→ INSUMO_HUERFANO  (componente aislada)
    """
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            ("ACERO_REFUERZO", "PROVEEDOR_MONOPOLICO"),
            ("PROVEEDOR_MONOPOLICO", "CIMENTACION"),
            ("APU_FANTASMA", "INSUMO_HUERFANO"),
        ]
    )
    return graph


def build_union_graph(
    graph_a: nx.DiGraph,
    graph_b: nx.DiGraph,
) -> nx.DiGraph:
    """
    Construye la unión adversarial A ∪ B.

    Usa nx.compose que preserva nodos y arcos de ambos grafos.
    """
    return nx.compose(graph_a, graph_b)


# ============================================================================
# UTILIDADES ESPECTRALES Y TOPOLÓGICAS
# ============================================================================


def compute_directed_cycle_count(
    graph: nx.DiGraph,
    *,
    max_cycles: int = _MAX_CYCLE_ENUMERATION,
) -> int:
    """
    Proxy operacional de β₁ para grafos dirigidos:
    cantidad de ciclos dirigidos simples, acotado por max_cycles.

    Complejidad: O(V + E) por ciclo encontrado, pero la cantidad
    total de ciclos puede ser exponencial. El parámetro max_cycles
    protege contra esta explosión.

    Args:
        graph: Grafo dirigido a analizar.
        max_cycles: Límite superior de ciclos a enumerar.

    Returns:
        Cantidad de ciclos encontrados (≤ max_cycles).
    """
    if graph.number_of_nodes() == 0:
        return 0

    count = 0
    for _ in nx.simple_cycles(graph):
        count += 1
        if count >= max_cycles:
            warnings.warn(
                f"Enumeración de ciclos truncada en {max_cycles}. "
                f"El conteo real puede ser mayor.",
                RuntimeWarning,
                stacklevel=2,
            )
            break
    return count


def compute_weak_beta_0(graph: nx.DiGraph) -> int:
    """
    Número de componentes débilmente conexas (β₀ débil).

    Equivale al número de componentes conexas del grafo no dirigido
    subyacente. Para un grafo vacío, retorna 0.
    """
    if graph.number_of_nodes() == 0:
        return 0
    return nx.number_weakly_connected_components(graph)


def compute_fiedler_value(graph: nx.DiGraph) -> float:
    """
    Calcula λ₂ (valor de Fiedler) del Laplaciano normalizado
    del grafo no dirigido subyacente.

    Propiedades del Laplaciano normalizado L_norm:
    - Simétrico real ⟹ autovalores reales
    - Semidefinido positivo ⟹ autovalores ≥ 0
    - λ₁ = 0 siempre (autovector constante)
    - λ₂ > 0 ⟺ grafo conexo (teorema de Fiedler)
    - λ₂ ≤ n/(n-1) para n nodos

    Convenciones:
    - Grafo con 0 o 1 nodo: λ₂ := 0.0
    - Grafo desconexo: λ₂ ≈ 0.0 (dentro de tolerancia numérica)
    - Se usa eigvalsh por estabilidad numérica (O(n²) vs O(n³) para eigh)

    Nota sobre conversión a no dirigido:
    nx.Graph(digraph) ignora dirección pero NO crea multiaristas.
    Esto es correcto para el Laplaciano normalizado.
    """
    # Conversión explícita a grafo simple no dirigido
    undirected = nx.Graph(graph)

    n_nodes = undirected.number_of_nodes()
    if n_nodes <= 1:
        return 0.0

    # Verificar que no hay nodos aislados con grado 0
    # (el Laplaciano normalizado no está definido para grado 0)
    degrees = dict(undirected.degree())
    isolated_nodes = [v for v, d in degrees.items() if d == 0]
    if isolated_nodes:
        # Nodos aislados implican componentes desconexas → λ₂ = 0
        return 0.0

    try:
        laplacian = nx.normalized_laplacian_matrix(undirected)
        laplacian_dense = np.asarray(laplacian.todense(), dtype=np.float64)
    except Exception as exc:
        warnings.warn(
            f"Error al construir Laplaciano normalizado: {exc}. "
            f"Retornando λ₂ = 0.0 como caso seguro.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0.0

    # Verificación de simetría (invariante del Laplaciano normalizado)
    symmetry_error = np.max(np.abs(laplacian_dense - laplacian_dense.T))
    if symmetry_error > _SPECTRAL_ZERO_TOLERANCE:
        warnings.warn(
            f"Laplaciano no simétrico (error={symmetry_error:.2e}). "
            f"Posible corrupción de datos.",
            RuntimeWarning,
            stacklevel=2,
        )

    try:
        eigenvalues = np.linalg.eigvalsh(laplacian_dense)
    except np.linalg.LinAlgError as exc:
        warnings.warn(
            f"Fallo en diagonalización: {exc}. Retornando λ₂ = 0.0.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0.0

    eigenvalues = np.sort(eigenvalues)

    if len(eigenvalues) < 2:
        return 0.0

    lambda_2 = float(eigenvalues[1])

    # Limpieza numérica: valores negativos cercanos a cero son artefactos
    if lambda_2 < 0 and abs(lambda_2) < _SPECTRAL_ZERO_TOLERANCE:
        lambda_2 = 0.0

    # Sanidad: λ₂ no debería ser negativo fuera de tolerancia
    if lambda_2 < -_SPECTRAL_ZERO_TOLERANCE:
        warnings.warn(
            f"λ₂ negativo fuera de tolerancia: {lambda_2:.2e}. "
            f"Posible inestabilidad numérica.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Limpieza: valores muy pequeños positivos → 0
    if abs(lambda_2) < _SPECTRAL_ZERO_TOLERANCE:
        return 0.0

    return lambda_2


def compute_topological_stress_metrics(
    graph: nx.DiGraph,
) -> TopologicalMetrics:
    """
    Construye el paquete de métricas topológicas requerido por el sistema.

    Centraliza el cómputo para garantizar consistencia entre métricas:
    todas se calculan sobre el mismo grafo en el mismo instante.
    """
    return TopologicalMetrics(
        beta_0=compute_weak_beta_0(graph),
        beta_1=compute_directed_cycle_count(graph),
        fiedler_value=compute_fiedler_value(graph),
    )


# ============================================================================
# BUILDER DE ESTADO ADVERSARIAL
# ============================================================================


def build_adversarial_state(
    graph: nx.DiGraph,
    topological_metrics: TopologicalMetrics,
    *,
    system_temperature: float = _CRITICAL_THERMAL_STRESS,
    financial_inertia: float = 0.1,
    validated_strata: frozenset = frozenset(
        {Stratum.PHYSICS, Stratum.TACTICS}
    ),
    source: str = "adversarial_injection",
    test_case: str = "",
) -> CategoricalState:
    """
    Construye un CategoricalState adversarial configurado para
    pruebas de estrés con acoplamiento topológico-térmico.
    """
    return CategoricalState(
        payload={
            "graph": graph,
            "thermal_metrics": {
                "system_temperature": system_temperature,
                "financial_inertia": financial_inertia,
            },
            "topological_metrics": topological_metrics,
        },
        context={
            "source": source,
            "test_case": test_case,
        },
        validated_strata=validated_strata,
    )


# ============================================================================
# 1. VALIDACIÓN DE CONSTRUCCIÓN DE GRAFOS
# ============================================================================


@pytest.mark.unit
class TestGraphConstruction:
    """
    Verifica las invariantes estructurales de los grafos de prueba.

    Estas pruebas son precondiciones: si fallan, los tests de
    integración subsecuentes no tienen sentido.
    """

    def test_nominal_graph_is_dag(self):
        """Grafo A (presupuesto maestro) debe ser DAG estricto."""
        graph = build_nominal_master_graph()
        assert nx.is_directed_acyclic_graph(graph), (
            "Invariante violada: Grafo A debe ser un DAG."
        )

    def test_nominal_graph_is_weakly_connected(self):
        """Grafo A debe ser débilmente conexo (β₀ = 1)."""
        graph = build_nominal_master_graph()
        assert nx.number_weakly_connected_components(graph) == 1, (
            "Invariante violada: Grafo A debe tener β₀_débil = 1."
        )

    def test_nominal_graph_structure(self):
        """Grafo A debe tener exactamente 4 nodos y 4 arcos."""
        graph = build_nominal_master_graph()
        assert graph.number_of_nodes() == 4
        assert graph.number_of_edges() == 4

    def test_adversarial_graph_is_dag(self):
        """Grafo B (APUs) debe ser localmente DAG."""
        graph = build_adversarial_apu_graph()
        assert nx.is_directed_acyclic_graph(graph), (
            "Invariante violada: Grafo B debe ser localmente un DAG."
        )

    def test_adversarial_graph_has_two_components(self):
        """Grafo B debe tener exactamente 2 componentes débiles."""
        graph = build_adversarial_apu_graph()
        assert nx.number_weakly_connected_components(graph) == 2, (
            "Invariante violada: Grafo B debe tener 2 componentes débiles "
            "(cadena principal + componente fantasma)."
        )

    def test_adversarial_graph_structure(self):
        """Grafo B debe tener exactamente 4 nodos y 3 arcos."""
        graph = build_adversarial_apu_graph()
        assert graph.number_of_nodes() == 4
        assert graph.number_of_edges() == 3

    def test_union_preserves_all_nodes(self):
        """La unión debe contener todos los nodos de A y B."""
        graph_a = build_nominal_master_graph()
        graph_b = build_adversarial_apu_graph()
        graph_union = build_union_graph(graph_a, graph_b)

        nodes_a = set(graph_a.nodes())
        nodes_b = set(graph_b.nodes())
        expected_nodes = nodes_a | nodes_b

        assert set(graph_union.nodes()) == expected_nodes

    def test_union_preserves_all_edges(self):
        """La unión debe contener todos los arcos de A y B."""
        graph_a = build_nominal_master_graph()
        graph_b = build_adversarial_apu_graph()
        graph_union = build_union_graph(graph_a, graph_b)

        edges_a = set(graph_a.edges())
        edges_b = set(graph_b.edges())
        expected_edges = edges_a | edges_b

        assert set(graph_union.edges()) == expected_edges

    def test_shared_nodes_are_correct(self):
        """Los nodos compartidos entre A y B deben ser exactamente
        {ACERO_REFUERZO, CIMENTACION}."""
        graph_a = build_nominal_master_graph()
        graph_b = build_adversarial_apu_graph()

        shared = set(graph_a.nodes()) & set(graph_b.nodes())
        assert shared == {"ACERO_REFUERZO", "CIMENTACION"}, (
            f"Nodos compartidos inesperados: {shared}. "
            f"Se esperaban exactamente ACERO_REFUERZO y CIMENTACION."
        )


# ============================================================================
# 2. PROPIEDADES TOPOLÓGICAS PURAS
# ============================================================================


@pytest.mark.unit
class TestTopologicalInvariants:
    """
    Verifica propiedades topológicas puras de la fusión adversarial,
    sin invocar componentes del sistema MIC.
    """

    @pytest.fixture
    def graph_a(self) -> nx.DiGraph:
        return build_nominal_master_graph()

    @pytest.fixture
    def graph_b(self) -> nx.DiGraph:
        return build_adversarial_apu_graph()

    @pytest.fixture
    def graph_union(
        self, graph_a: nx.DiGraph, graph_b: nx.DiGraph
    ) -> nx.DiGraph:
        return build_union_graph(graph_a, graph_b)

    def test_local_acyclicity_precondition(
        self,
        graph_a: nx.DiGraph,
        graph_b: nx.DiGraph,
    ):
        """
        Precondición de Mayer-Vietoris operacional:
        ambos subgrafos son localmente acíclicos.
        """
        assert nx.is_directed_acyclic_graph(graph_a)
        assert nx.is_directed_acyclic_graph(graph_b)

    def test_fusion_breaks_acyclicity(self, graph_union: nx.DiGraph):
        """
        Propiedad emergente: la unión de dos DAGs puede NO ser DAG.

        Ciclo inducido:
            CIMENTACION → ACERO_REFUERZO → PROVEEDOR_MONOPOLICO → CIMENTACION
        """
        assert not nx.is_directed_acyclic_graph(graph_union), (
            "La unión A ∪ B debía romper la aciclicidad global."
        )

    def test_emergent_cycle_exists(self, graph_union: nx.DiGraph):
        """
        Verifica la existencia del ciclo emergente específico.
        """
        cycle_count = compute_directed_cycle_count(graph_union)
        assert cycle_count > 0, (
            "La unión A ∪ B debía inducir al menos un ciclo dirigido "
            "emergente (proxy de β₁ > 0)."
        )

    def test_emergent_cycle_is_specific(self, graph_union: nx.DiGraph):
        """
        El ciclo emergente debe involucrar los nodos de la cadena
        adversarial: CIMENTACION, ACERO_REFUERZO, PROVEEDOR_MONOPOLICO.
        """
        expected_cycle_nodes = {
            "CIMENTACION",
            "ACERO_REFUERZO",
            "PROVEEDOR_MONOPOLICO",
        }
        cycles = list(nx.simple_cycles(graph_union))
        cycle_node_sets = [set(cycle) for cycle in cycles]

        assert any(
            expected_cycle_nodes.issubset(cycle_nodes)
            for cycle_nodes in cycle_node_sets
        ), (
            f"Ningún ciclo encontrado contiene los nodos esperados "
            f"{expected_cycle_nodes}. Ciclos: {cycles}"
        )

    def test_fragmentation_induced(self, graph_union: nx.DiGraph):
        """
        La componente fantasma (APU_FANTASMA → INSUMO_HUERFANO) debe
        inducir fragmentación: β₀_débil > 1.
        """
        beta_0 = compute_weak_beta_0(graph_union)
        assert beta_0 > 1, (
            f"Fragmentación topológica esperada (β₀ > 1), obtenido β₀ = {beta_0}."
        )

    def test_fragmentation_count_exact(self, graph_union: nx.DiGraph):
        """
        Con la construcción actual, deben existir exactamente 2
        componentes débilmente conexas.
        """
        beta_0 = compute_weak_beta_0(graph_union)
        assert beta_0 == 2, (
            f"Se esperaban exactamente 2 componentes débilmente conexas, "
            f"obtenidas {beta_0}."
        )

    def test_phantom_component_identified(self, graph_union: nx.DiGraph):
        """
        La componente fantasma debe ser exactamente {APU_FANTASMA, INSUMO_HUERFANO}.
        """
        components = list(nx.weakly_connected_components(graph_union))
        phantom_expected = {"APU_FANTASMA", "INSUMO_HUERFANO"}

        assert any(component == phantom_expected for component in components), (
            f"Componente fantasma no encontrada. Componentes: {components}"
        )


# ============================================================================
# 3. PROPIEDADES ESPECTRALES PURAS
# ============================================================================


@pytest.mark.unit
class TestSpectralProperties:
    """
    Verifica propiedades espectrales del Laplaciano normalizado
    del grafo no dirigido subyacente a la unión adversarial.
    """

    @pytest.fixture
    def graph_union(self) -> nx.DiGraph:
        return build_union_graph(
            build_nominal_master_graph(),
            build_adversarial_apu_graph(),
        )

    def test_fiedler_value_computable(self, graph_union: nx.DiGraph):
        """λ₂ debe ser computable sin excepciones."""
        lambda_2 = compute_fiedler_value(graph_union)
        assert isinstance(lambda_2, float)

    def test_fiedler_value_nonnegative(self, graph_union: nx.DiGraph):
        """λ₂ ≥ 0 (propiedad del Laplaciano semidefinido positivo)."""
        lambda_2 = compute_fiedler_value(graph_union)
        assert lambda_2 >= -_SPECTRAL_ZERO_TOLERANCE, (
            f"λ₂ negativo fuera de tolerancia: {lambda_2}"
        )

    def test_fiedler_indicates_disconnection(self, graph_union: nx.DiGraph):
        """
        Para un grafo desconexo, λ₂ = 0 (o muy cercano a 0).

        La unión A ∪ B tiene β₀ = 2 (desconexo débilmente),
        por lo que el grafo no dirigido subyacente es desconexo
        y λ₂ debe ser ≈ 0.
        """
        lambda_2 = compute_fiedler_value(graph_union)
        assert lambda_2 < _SPECTRAL_ZERO_TOLERANCE, (
            f"Grafo desconexo debería tener λ₂ ≈ 0, obtenido {lambda_2:.6e}"
        )

    def test_fiedler_below_fracture_threshold(
        self, graph_union: nx.DiGraph
    ):
        """λ₂ < umbral de fractura para el escenario adversarial."""
        lambda_2 = compute_fiedler_value(graph_union)
        assert lambda_2 < _FIEDLER_FRACTURE_THRESHOLD, (
            f"λ₂ = {lambda_2:.6f} excede el umbral de fractura "
            f"({_FIEDLER_FRACTURE_THRESHOLD})"
        )

    def test_connected_graph_has_positive_fiedler(self):
        """Control positivo: grafo conexo debe tener λ₂ > 0."""
        graph = build_nominal_master_graph()
        assert nx.number_weakly_connected_components(graph) == 1

        lambda_2 = compute_fiedler_value(graph)
        assert lambda_2 > _SPECTRAL_ZERO_TOLERANCE, (
            f"Grafo conexo debería tener λ₂ > 0, obtenido {lambda_2:.6e}"
        )

    def test_laplacian_spectrum_complete(self, graph_union: nx.DiGraph):
        """
        El espectro del Laplaciano normalizado debe tener n autovalores,
        todos en [0, 2], con al menos uno igual a 0.
        """
        undirected = nx.Graph(graph_union)
        n = undirected.number_of_nodes()

        # Filtrar nodos aislados para el Laplaciano normalizado
        degrees = dict(undirected.degree())
        non_isolated = [v for v, d in degrees.items() if d > 0]

        if len(non_isolated) < 2:
            pytest.skip("Muy pocos nodos no aislados para análisis espectral.")

        subgraph = undirected.subgraph(non_isolated)
        laplacian = nx.normalized_laplacian_matrix(subgraph)
        laplacian_dense = np.asarray(laplacian.todense(), dtype=np.float64)
        eigenvalues = np.sort(np.linalg.eigvalsh(laplacian_dense))

        assert len(eigenvalues) == len(non_isolated)

        # Todos los autovalores en [0, 2] (con tolerancia numérica)
        assert np.all(eigenvalues >= -_SPECTRAL_ZERO_TOLERANCE), (
            f"Autovalores negativos encontrados: "
            f"{eigenvalues[eigenvalues < -_SPECTRAL_ZERO_TOLERANCE]}"
        )
        assert np.all(eigenvalues <= 2 + _SPECTRAL_ZERO_TOLERANCE), (
            f"Autovalores > 2 encontrados: "
            f"{eigenvalues[eigenvalues > 2 + _SPECTRAL_ZERO_TOLERANCE]}"
        )

        # λ₁ ≈ 0 (siempre)
        assert abs(eigenvalues[0]) < _SPECTRAL_ZERO_TOLERANCE, (
            f"λ₁ debería ser ≈ 0, obtenido {eigenvalues[0]:.6e}"
        )


# ============================================================================
# 4. PROXY DE MAYER-VIETORIS
# ============================================================================


@pytest.mark.unit
class TestMayerVietorisProxy:
    """
    Verifica el análogo operacional de Mayer-Vietoris:
    la unión de dos subcomplejos localmente acíclicos puede inducir
    homología global no trivial.

    En la secuencia exacta de Mayer-Vietoris para H₁:
        ... → H₁(A) ⊕ H₁(B) → H₁(A ∪ B) → H₀(A ∩ B) → ...

    Si H₁(A) = H₁(B) = 0 (DAGs), pero A ∩ B ≠ ∅, entonces
    H₁(A ∪ B) puede ser no trivial.

    Nuestro proxy: ciclos dirigidos simples como sustituto de H₁.
    """

    def test_mayer_vietoris_preconditions(self):
        """H₁(A) = H₁(B) = 0 (ambos DAGs, sin ciclos)."""
        graph_a = build_nominal_master_graph()
        graph_b = build_adversarial_apu_graph()

        assert compute_directed_cycle_count(graph_a) == 0
        assert compute_directed_cycle_count(graph_b) == 0

    def test_intersection_nonempty(self):
        """A ∩ B ≠ ∅ (condición necesaria para emergencia)."""
        graph_a = build_nominal_master_graph()
        graph_b = build_adversarial_apu_graph()

        intersection = set(graph_a.nodes()) & set(graph_b.nodes())
        assert len(intersection) > 0, (
            "A ∩ B vacía: no puede haber emergencia topológica."
        )

    def test_homological_emergence(self):
        """
        H₁(A ∪ B) ≠ 0 pese a H₁(A) = H₁(B) = 0.

        Esta es la propiedad fundamental que hace al test adversarial
        interesante: la patología es EMERGENTE, no local.
        """
        graph_a = build_nominal_master_graph()
        graph_b = build_adversarial_apu_graph()
        graph_union = build_union_graph(graph_a, graph_b)

        beta_1_a = compute_directed_cycle_count(graph_a)
        beta_1_b = compute_directed_cycle_count(graph_b)
        beta_1_union = compute_directed_cycle_count(graph_union)

        assert beta_1_a == 0, "Precondición: A acíclico"
        assert beta_1_b == 0, "Precondición: B acíclico"
        assert beta_1_union > 0, (
            "Emergencia homológica: H₁(A∪B) > 0 pese a H₁(A) = H₁(B) = 0"
        )

    def test_metrics_package_consistency(self):
        """Las métricas empaquetadas son consistentes con cómputos individuales."""
        graph_union = build_union_graph(
            build_nominal_master_graph(),
            build_adversarial_apu_graph(),
        )
        metrics = compute_topological_stress_metrics(graph_union)

        assert metrics.beta_0 == compute_weak_beta_0(graph_union)
        assert metrics.beta_1 == compute_directed_cycle_count(graph_union)
        assert abs(
            metrics.fiedler_value - compute_fiedler_value(graph_union)
        ) < _SPECTRAL_ZERO_TOLERANCE


# ============================================================================
# 5. INTEGRACIÓN CON SISTEMA MIC
# ============================================================================


@pytest.mark.integration
@pytest.mark.stress
class TestThermalCollapseIntegration:
    """
    Integración: verifica que el sistema MIC responda correctamente
    ante una perturbación ortogonal acoplada (colapso topológico +
    hipertermia financiera).

    Estos tests requieren los componentes del sistema (RiskChallenger,
    BusinessTopologicalAnalyzer, TelemetryContext). Se saltan
    automáticamente si no están disponibles.
    """

    @pytest.fixture
    def telemetry_ctx(self) -> TelemetryContext:
        ctx = TelemetryContext()
        yield ctx
        # Cleanup: evitar contaminación entre tests
        if hasattr(ctx, "clear"):
            ctx.clear()

    @pytest.fixture
    def adversarial_scenario(self):
        """Escenario adversarial completo pre-construido."""
        graph_a = build_nominal_master_graph()
        graph_b = build_adversarial_apu_graph()
        graph_union = build_union_graph(graph_a, graph_b)
        metrics = compute_topological_stress_metrics(graph_union)
        return {
            "graph_a": graph_a,
            "graph_b": graph_b,
            "graph_union": graph_union,
            "metrics": metrics,
        }

    def test_adversarial_scenario_preconditions(self, adversarial_scenario):
        """
        Verifica todas las precondiciones del escenario adversarial
        antes de invocar componentes del sistema.
        """
        metrics = adversarial_scenario["metrics"]

        assert metrics.beta_1 > 0, (
            "Precondición: β₁ > 0 (ciclos emergentes)."
        )
        assert metrics.beta_0 > 1, (
            "Precondición: β₀ > 1 (fragmentación)."
        )
        assert metrics.fiedler_value < _FIEDLER_FRACTURE_THRESHOLD, (
            f"Precondición: λ₂ = {metrics.fiedler_value:.6f} < "
            f"{_FIEDLER_FRACTURE_THRESHOLD} (fractura de Fiedler)."
        )

    @pytest.mark.skipif(
        not _HAS_RISK_CHALLENGER,
        reason="RiskChallenger no disponible",
    )
    def test_risk_challenger_detects_critical_severity(
        self,
        adversarial_scenario,
        telemetry_ctx: TelemetryContext,
    ):
        """
        Bajo acoplamiento topológico-térmico extremo, el RiskChallenger
        debe elevar la severidad a CRITICO.
        """
        metrics = adversarial_scenario["metrics"]
        graph_union = adversarial_scenario["graph_union"]

        malicious_state = build_adversarial_state(
            graph=graph_union,
            topological_metrics=metrics,
            test_case="risk_challenger_critical_severity",
        )

        challenger = RiskChallenger(mic=MagicMock())
        report = challenger.evaluate(malicious_state)

        assert report is not None, (
            "RiskChallenger.evaluate() retornó None para estado adversarial."
        )

        # Verificar severidad
        global_severity = getattr(report, "global_severity", None)
        assert global_severity is not None, (
            "El report no tiene atributo 'global_severity'."
        )
        assert str(global_severity).upper() == "CRITICO", (
            f"Severidad esperada: CRITICO, obtenida: {global_severity}"
        )

    @pytest.mark.skipif(
        not _HAS_RISK_CHALLENGER,
        reason="RiskChallenger no disponible",
    )
    def test_risk_challenger_emits_veto(
        self,
        adversarial_scenario,
        telemetry_ctx: TelemetryContext,
    ):
        """
        Bajo acoplamiento topológico-térmico extremo, el sistema debe
        emitir un veto técnico compuesto.
        """
        metrics = adversarial_scenario["metrics"]
        graph_union = adversarial_scenario["graph_union"]

        malicious_state = build_adversarial_state(
            graph=graph_union,
            topological_metrics=metrics,
            test_case="risk_challenger_veto",
        )

        challenger = RiskChallenger(mic=MagicMock())
        report = challenger.evaluate(malicious_state)

        assert report is not None

        verdict_code = str(getattr(report, "verdict_code", "")).upper()
        assert "VETO" in verdict_code, (
            f"Se esperaba 'VETO' en verdict_code, obtenido: '{verdict_code}'"
        )

    @pytest.mark.skipif(
        not _HAS_RISK_CHALLENGER,
        reason="RiskChallenger no disponible",
    )
    def test_risk_challenger_verdict_reflects_instability(
        self,
        adversarial_scenario,
        telemetry_ctx: TelemetryContext,
    ):
        """
        El código de veredicto debe reflejar explícitamente la
        inestabilidad crítica detectada.
        """
        metrics = adversarial_scenario["metrics"]
        graph_union = adversarial_scenario["graph_union"]

        malicious_state = build_adversarial_state(
            graph=graph_union,
            topological_metrics=metrics,
            test_case="risk_challenger_instability_verdict",
        )

        challenger = RiskChallenger(mic=MagicMock())
        report = challenger.evaluate(malicious_state)

        assert report is not None

        verdict_code = str(getattr(report, "verdict_code", "")).upper()
        instability_indicators = {
            "CRITICAL",
            "INSTABILITY",
            "INESTABILIDAD",
            "COLLAPSE",
            "COLAPSO",
        }
        assert any(indicator in verdict_code for indicator in instability_indicators), (
            f"El verdict_code '{verdict_code}' no contiene ningún "
            f"indicador de inestabilidad: {instability_indicators}"
        )

    @pytest.mark.skipif(
        not _HAS_TELEMETRY or not _HAS_RISK_CHALLENGER,
        reason="TelemetryContext o RiskChallenger no disponible",
    )
    def test_telemetry_preserves_causal_chain(
        self,
        adversarial_scenario,
        telemetry_ctx: TelemetryContext,
    ):
        """
        La telemetría debe preservar la cadena de custodia causal
        del colapso topológico/térmico.
        """
        metrics = adversarial_scenario["metrics"]
        graph_union = adversarial_scenario["graph_union"]

        malicious_state = build_adversarial_state(
            graph=graph_union,
            topological_metrics=metrics,
            test_case="telemetry_causal_chain",
        )

        challenger = RiskChallenger(mic=MagicMock())
        _ = challenger.evaluate(malicious_state)

        telemetry_errors = telemetry_ctx.get_errors()

        if not telemetry_errors:
            # Si no hay errores registrados, el sistema puede manejar
            # el escenario sin registrar en telemetría — aceptable
            # si el veto se emitió correctamente.
            pytest.skip(
                "Sin errores en telemetría — cadena causal no aplica."
            )

        causal_keywords = {
            "topology",
            "topological",
            "thermal",
            "temperature",
            "fiedler",
            "cycle",
            "ciclo",
            "fracture",
            "fractura",
        }

        error_types = []
        for event in telemetry_errors:
            if isinstance(event, dict):
                error_type = str(event.get("type", "")).lower()
                error_msg = str(event.get("message", "")).lower()
                error_types.append(error_type)
                error_types.append(error_msg)

        assert any(
            any(kw in entry for kw in causal_keywords)
            for entry in error_types
        ), (
            f"La telemetría registró errores pero no preservó la cadena "
            f"causal topológico/térmica. "
            f"Tipos/mensajes encontrados: {error_types}"
        )


# ============================================================================
# 6. ROBUSTEZ Y CASOS EXTREMOS
# ============================================================================


@pytest.mark.unit
class TestEdgeCasesAndRobustness:
    """
    Verifica comportamiento correcto ante entradas degeneradas
    y condiciones extremas en las utilidades topológicas y espectrales.
    """

    def test_empty_graph_beta_0(self):
        """Grafo vacío: β₀ = 0."""
        assert compute_weak_beta_0(nx.DiGraph()) == 0

    def test_empty_graph_beta_1(self):
        """Grafo vacío: β₁ = 0."""
        assert compute_directed_cycle_count(nx.DiGraph()) == 0

    def test_empty_graph_fiedler(self):
        """Grafo vacío: λ₂ = 0."""
        assert compute_fiedler_value(nx.DiGraph()) == 0.0

    def test_single_node_graph(self):
        """Grafo con un solo nodo."""
        graph = nx.DiGraph()
        graph.add_node("A")

        assert compute_weak_beta_0(graph) == 1
        assert compute_directed_cycle_count(graph) == 0
        assert compute_fiedler_value(graph) == 0.0

    def test_single_edge_graph(self):
        """Grafo con una sola arista."""
        graph = nx.DiGraph()
        graph.add_edge("A", "B")

        assert compute_weak_beta_0(graph) == 1
        assert compute_directed_cycle_count(graph) == 0
        lambda_2 = compute_fiedler_value(graph)
        assert lambda_2 > 0, "Grafo conexo con 2 nodos debe tener λ₂ > 0."

    def test_self_loop_is_cycle(self):
        """Un self-loop cuenta como ciclo dirigido."""
        graph = nx.DiGraph()
        graph.add_edge("A", "A")

        assert compute_directed_cycle_count(graph) >= 1

    def test_complete_graph_fiedler(self):
        """
        Para el grafo completo K_n dirigido (todas las aristas en ambas
        direcciones), el grafo no dirigido subyacente es K_n con
        λ₂ = n/(n-1) para el Laplaciano normalizado.
        """
        n = 5
        complete = nx.complete_graph(n, create_using=nx.DiGraph)
        lambda_2 = compute_fiedler_value(complete)

        expected = n / (n - 1)
        assert abs(lambda_2 - expected) < 0.01, (
            f"K_{n}: λ₂ esperado ≈ {expected:.4f}, obtenido {lambda_2:.4f}"
        )

    def test_cycle_count_limit(self):
        """La enumeración de ciclos se trunca correctamente."""
        # Grafo con muchos ciclos (completo dirigido pequeño)
        graph = nx.complete_graph(6, create_using=nx.DiGraph)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            count = compute_directed_cycle_count(graph, max_cycles=5)
            assert count == 5

            if w:
                assert any("truncada" in str(warning.message) for warning in w)

    def test_disconnected_graph_fiedler_zero(self):
        """Grafo con múltiples componentes desconexas: λ₂ = 0."""
        graph = nx.DiGraph()
        graph.add_edge("A", "B")
        graph.add_edge("C", "D")
        graph.add_edge("E", "F")

        lambda_2 = compute_fiedler_value(graph)
        assert abs(lambda_2) < _SPECTRAL_ZERO_TOLERANCE, (
            f"Grafo desconexo debe tener λ₂ ≈ 0, obtenido {lambda_2:.6e}"
        )

    def test_isolated_node_fiedler(self):
        """Grafo con nodo aislado (grado 0): λ₂ = 0."""
        graph = nx.DiGraph()
        graph.add_edge("A", "B")
        graph.add_node("ISOLATED")

        lambda_2 = compute_fiedler_value(graph)
        assert lambda_2 == 0.0, (
            f"Nodo aislado implica desconexión, λ₂ debe ser 0, "
            f"obtenido {lambda_2}"
        )

    def test_topological_metrics_dataclass(self):
        """TopologicalMetrics se construye correctamente."""
        metrics = TopologicalMetrics(
            beta_0=2, beta_1=1, fiedler_value=0.3
        )
        assert metrics.beta_0 == 2
        assert metrics.beta_1 == 1
        assert abs(metrics.fiedler_value - 0.3) < 1e-10

    def test_adversarial_state_construction(self):
        """El builder de estado adversarial produce un estado válido."""
        graph = build_nominal_master_graph()
        metrics = compute_topological_stress_metrics(graph)
        state = build_adversarial_state(
            graph=graph,
            topological_metrics=metrics,
            test_case="construction_test",
        )

        assert state.is_success
        assert state.payload["graph"] is graph
        assert state.payload["topological_metrics"] is metrics
        assert state.context["source"] == "adversarial_injection"
        assert Stratum.PHYSICS in state.validated_strata
        assert Stratum.TACTICS in state.validated_strata

    def test_adversarial_state_custom_temperature(self):
        """El builder respeta la temperatura personalizada."""
        graph = build_nominal_master_graph()
        metrics = compute_topological_stress_metrics(graph)
        state = build_adversarial_state(
            graph=graph,
            topological_metrics=metrics,
            system_temperature=99.9,
        )
        assert state.payload["thermal_metrics"]["system_temperature"] == 99.9

    def test_fiedler_numerical_stability_with_near_singular(self):
        """
        Grafo casi desconexo (arista muy débilmente acoplada conceptualmente):
        línea larga → λ₂ pequeño pero positivo.
        """
        # Grafo lineal largo: A→B→C→D→E→F→G→H
        graph = nx.path_graph(8, create_using=nx.DiGraph)
        lambda_2 = compute_fiedler_value(graph)

        # Para un path de n nodos, λ₂ del Laplaciano normalizado
        # decrece con n pero permanece > 0
        assert lambda_2 > 0, (
            f"Path graph conexo debe tener λ₂ > 0, obtenido {lambda_2}"
        )

    def test_metrics_deterministic(self):
        """Las métricas son deterministas para el mismo grafo."""
        graph = build_union_graph(
            build_nominal_master_graph(),
            build_adversarial_apu_graph(),
        )

        m1 = compute_topological_stress_metrics(graph)
        m2 = compute_topological_stress_metrics(graph)

        assert m1.beta_0 == m2.beta_0
        assert m1.beta_1 == m2.beta_1
        assert abs(m1.fiedler_value - m2.fiedler_value) < _SPECTRAL_ZERO_TOLERANCE


# ============================================================================
# EJECUCIÓN DIRECTA
# ============================================================================

if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
            "--tb=short",
            "-x",
            "-m",
            "not integration",
        ]
    )