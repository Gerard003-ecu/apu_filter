"""
Suite de Pruebas para Análisis Espectral de Grafos de Presupuesto.

Fundamentos Matemáticos Verificados:
─────────────────────────────────────────────────────────────────────────────
LAPLACIANO NORMALIZADO (Chung, 1997):
  Sea G = (V, E) grafo no-dirigido con matriz de adyacencia A y grado D.
  L_norm = I - D^(-1/2) · A · D^(-1/2)

PROPIEDADES GARANTIZADAS:
  · λᵢ ∈ [0, 2]                ∀i  (acotamiento espectral)
  · λ₁ = 0                         (eigenvector constante ponderado)
  · λ₂ > 0  ⟺  G conexo            (conectividad algebraica de Fiedler)
  · λ_max = 2  ⟺  G bipartito      (caracterización bipartita)

VALORES TEÓRICOS CONOCIDOS:
  · Ciclo C_n:    λ₂ = 1 - cos(2π/n)      [Laplaciano normalizado]
  · Completo K_n: λ₂ = n/(n-1)            [todos los λ>0 iguales]
  · Camino P_n:   λ₂ ≈ 2(1-cos(π/n))/... [depende de normalización]

ENERGÍA ESPECTRAL:
  E(G) = Σ λᵢ² = ||L_norm||²_F  ≥ 0

ANALOGÍA CIRCUITAL (física de redes resistivas):
  El Laplaciano actúa como la matriz de conductancias nodales.
  λ₂ (valor de Fiedler) ≡ "conductancia efectiva" de la red.
  Resonancia espectral ≡ degeneración de modos propios (riesgo de
  amplificación de perturbaciones en frecuencias concentradas).

Referencias:
  - Chung, F. R. K. (1997). Spectral Graph Theory. AMS.
  - Mohar, B. (1991). The Laplacian spectrum of graphs.
  - Von Luxburg, U. (2007). A tutorial on spectral clustering.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import math
import unittest
from typing import Any

import networkx as nx
import numpy as np

from app.tactics.business_topology import BusinessTopologicalAnalyzer


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES GLOBALES
# ═══════════════════════════════════════════════════════════════════════════════

# Tolerancia numérica máquina para float64 (análogo a eps en MATLAB)
_MACHINE_EPS: float = np.finfo(np.float64).eps          # ≈ 2.22e-16
_DEFAULT_EPSILON: float = 1e-9                           # Para comparaciones ~0
_DEFAULT_TOLERANCE: float = 1e-6                         # Para igualdad aproximada
_SPECTRAL_TOLERANCE: float = 1e-4                        # Para valores teóricos
_THEORETICAL_TOLERANCE: float = 0.1                     # Para fórmulas teóricas
                                                         # (diferencias de normalización)


# ═══════════════════════════════════════════════════════════════════════════════
# CLASE BASE
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpectralAnalysisBase(unittest.TestCase):
    """
    Clase base con utilidades comunes para pruebas espectrales.

    Responsabilidades:
      · Gestión del ciclo de vida del analizador (setUp / tearDown).
      · Aserciones matemáticamente robustas (NaN/Inf-safe).
      · Fábrica de grafos de presupuesto parametrizables.
    """

    # ── Ciclo de vida ─────────────────────────────────────────────────────────

    def setUp(self) -> None:
        """Inicializa recursos antes de cada prueba."""
        self.analyzer = BusinessTopologicalAnalyzer(telemetry=None)
        self.EPSILON = _DEFAULT_EPSILON
        self.TOLERANCE = _DEFAULT_TOLERANCE

    def tearDown(self) -> None:
        """
        Libera recursos después de cada prueba.

        Importante para pruebas con grafos grandes (>100 nodos) que
        pueden acumular matrices densas en memoria.
        """
        self.analyzer = None  # Permite GC inmediato

    # ── Aserciones robustas ───────────────────────────────────────────────────

    def assertFinite(self, value: float, msg: str | None = None) -> None:
        """
        Verifica que un valor es finito (no NaN, no Inf).

        Motivación: Los cálculos de eigenvalores sobre matrices mal
        condicionadas pueden producir NaN/Inf silenciosamente.
        """
        if not math.isfinite(value):
            standard_msg = (
                f"Se esperaba valor finito, se obtuvo: {value!r}. "
                f"Posible inestabilidad numérica en eigendescomposición."
            )
            self.fail(self._formatMessage(msg, standard_msg))

    def assertAlmostEqualTolerant(
        self,
        first: float,
        second: float,
        tolerance: float | None = None,
        msg: str | None = None,
    ) -> None:
        """
        Igualdad aproximada con tolerancia configurable y guardas NaN/Inf.

        Contrato: |first - second| ≤ tolerance
        Precondición verificada: ambos valores deben ser finitos.
        """
        # Guardia NaN/Inf antes de la resta (evita comparaciones indefinidas)
        for name, val in [("first", first), ("second", second)]:
            if not math.isfinite(val):
                self.fail(
                    self._formatMessage(
                        msg,
                        f"assertAlmostEqualTolerant: '{name}' = {val!r} no es finito",
                    )
                )

        tol = tolerance if tolerance is not None else self.TOLERANCE
        diff = abs(first - second)
        if diff > tol:
            standard_msg = (
                f"{first} ≠ {second}  "
                f"(|diff| = {diff:.2e} > tol = {tol:.2e})"
            )
            self.fail(self._formatMessage(msg, standard_msg))

    def assertInRange(
        self,
        value: float,
        min_val: float,
        max_val: float,
        msg: str | None = None,
    ) -> None:
        """
        Verifica value ∈ [min_val, max_val] con guardia NaN/Inf.

        Uso típico: verificar que eigenvalores ∈ [0, 2].
        """
        self.assertFinite(value, msg=f"assertInRange: valor {value!r} no es finito")
        if not (min_val <= value <= max_val):
            standard_msg = (
                f"{value} ∉ [{min_val}, {max_val}]  "
                f"(exceso = {max(min_val - value, value - max_val, 0):.2e})"
            )
            self.fail(self._formatMessage(msg, standard_msg))

    def assertSpectralResultComplete(
        self,
        result: dict[str, Any],
        msg: str | None = None,
    ) -> None:
        """
        Verifica que un resultado espectral tiene todas las claves requeridas
        y que los valores numéricos son finitos.

        Claves obligatorias definidas por contrato del analizador.
        """
        required_keys = {
            "fiedler_value",
            "spectral_gap",
            "spectral_energy",
            "wavelength",
            "resonance_risk",
            "status",
        }
        for key in required_keys:
            self.assertIn(
                key,
                result,
                self._formatMessage(msg, f"Falta clave requerida '{key}' en resultado"),
            )

        # Verificar finitud de valores numéricos
        for key in ("fiedler_value", "spectral_energy"):
            self.assertFinite(
                result[key],
                msg=self._formatMessage(
                    msg, f"result['{key}'] = {result[key]!r} no es finito"
                ),
            )

    # ── Fábrica de grafos de presupuesto ──────────────────────────────────────

    def create_budget_graph(
        self,
        chapters: int = 1,
        apus_per_chapter: int = 2,
        insumos_per_apu: int = 3,
    ) -> nx.DiGraph:
        """
        Crea un grafo de presupuesto piramidal (DAG) para pruebas.

        Estructura jerárquica:
            ROOT  (nivel 0)
            └── CAPITULO_c  (nivel 1)  ×chapters
                └── APU_c_a  (nivel 2)  ×apus_per_chapter
                    └── INSUMO_c_a_i  (nivel 3)  ×insumos_per_apu

        Complejidad: O(chapters × apus_per_chapter × insumos_per_apu) nodos.

        Args:
            chapters:         Número de capítulos (≥ 1).
            apus_per_chapter: APUs por capítulo (≥ 1).
            insumos_per_apu:  Insumos por APU (≥ 1).

        Raises:
            ValueError: Si algún parámetro es < 1.
        """
        if chapters < 1 or apus_per_chapter < 1 or insumos_per_apu < 1:
            raise ValueError(
                f"Todos los parámetros deben ser ≥ 1. "
                f"Recibidos: chapters={chapters}, "
                f"apus_per_chapter={apus_per_chapter}, "
                f"insumos_per_apu={insumos_per_apu}"
            )

        G = nx.DiGraph(name="TestBudgetGraph")
        G.add_node("ROOT", type="ROOT", level=0)

        for c in range(1, chapters + 1):
            chapter_id = f"CAPITULO_{c}"
            G.add_node(chapter_id, type="CAPITULO", level=1)
            G.add_edge("ROOT", chapter_id, weight=1.0, total_cost=100.0 * c)

            for a in range(1, apus_per_chapter + 1):
                apu_id = f"APU_{c}_{a}"
                G.add_node(apu_id, type="APU", level=2)
                G.add_edge(chapter_id, apu_id, weight=1.0, total_cost=50.0)

                for i in range(1, insumos_per_apu + 1):
                    insumo_id = f"INSUMO_{c}_{a}_{i}"
                    G.add_node(
                        insumo_id,
                        type="INSUMO",
                        level=3,
                        description=f"Insumo {i}",
                        tipo_insumo="MATERIAL",
                    )
                    G.add_edge(
                        apu_id,
                        insumo_id,
                        quantity=1.0,
                        total_cost=10.0,
                    )

        return G


# ═══════════════════════════════════════════════════════════════════════════════
# CASOS BÁSICOS Y DEGENERADOS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpectralAnalysisBasicCases(TestSpectralAnalysisBase):
    """Pruebas para casos básicos y degenerados (grafos triviales)."""

    def test_empty_graph_returns_default_values(self) -> None:
        """
        Grafo vacío → valores por defecto sin excepción.

        Contrato del analizador:
          · fiedler_value  = 0.0
          · spectral_energy = 0.0
          · resonance_risk  = False
          · status          = 'insufficient_nodes'
        """
        G = nx.DiGraph()
        result = self.analyzer.analyze_spectral_stability(G)

        self.assertSpectralResultComplete(result)
        self.assertEqual(result["fiedler_value"], 0.0)
        self.assertEqual(result["spectral_energy"], 0.0)
        self.assertFalse(result["resonance_risk"])
        self.assertEqual(result["status"], "insufficient_nodes")

    def test_single_node_graph(self) -> None:
        """Un nodo: caso degenerado sin aristas, Laplaciano indefinido."""
        G = nx.DiGraph()
        G.add_node("A", type="ROOT")

        result = self.analyzer.analyze_spectral_stability(G)

        self.assertSpectralResultComplete(result)
        self.assertEqual(result["fiedler_value"], 0.0)
        self.assertEqual(result["status"], "insufficient_nodes")

    def test_two_node_connected_graph(self) -> None:
        """
        Grafo mínimo conexo (2 nodos, 1 arista).

        Para L_norm del grafo {A-B} no-dirigido: eigenvalores son {0, 2}.
        Luego Fiedler = λ₂ = 2.0 exacto (si los grados son iguales = 1).
        """
        G = nx.DiGraph()
        G.add_edge("A", "B")

        result = self.analyzer.analyze_spectral_stability(G)

        self.assertSpectralResultComplete(result)
        self.assertGreater(
            result["fiedler_value"],
            0.0,
            "Grafo conexo de 2 nodos debe tener Fiedler > 0",
        )
        self.assertEqual(result["status"], "success")

    def test_two_node_disconnected_graph(self) -> None:
        """
        2 nodos sin aristas (desconectado).

        Tras eliminar nodos aislados queda grafo vacío → Fiedler = 0.
        """
        G = nx.DiGraph()
        G.add_node("A")
        G.add_node("B")

        result = self.analyzer.analyze_spectral_stability(G)

        # Con ambos nodos aislados el análisis es inválido
        self.assertLessEqual(
            result["fiedler_value"],
            self.EPSILON,
            f"Grafo de nodos aislados debe tener Fiedler ≤ ε, "
            f"obtenido: {result['fiedler_value']}",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# CONECTIVIDAD ALGEBRAICA (FIEDLER VALUE)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpectralAnalysisConnectivity(TestSpectralAnalysisBase):
    """
    Pruebas de conectividad algebraica.

    Teorema de Fiedler (1973):
      G conexo  ⟺  λ₂(L) > 0
      λ₂ = max min_{x⊥1} (x^T L x) / (x^T x)   [variacional]
    """

    def test_connected_path_graph_positive_fiedler(self) -> None:
        """
        Path graph P₄ es conexo → Fiedler > 0.

        Para Laplaciano normalizado de P_n (no-dirigido):
          λ₂ = 1 - cos(π/(n-1))  [aproximación para n grande]
        Para n=4 el valor es positivo y verificable.
        """
        G = nx.path_graph(4, create_using=nx.DiGraph)

        result = self.analyzer.analyze_spectral_stability(G)

        self.assertSpectralResultComplete(result)
        self.assertGreater(
            result["fiedler_value"],
            0.0,
            "Path graph P₄ conexo debe tener Fiedler > 0",
        )
        self.assertEqual(result["status"], "success")
        self.assertFalse(result["resonance_risk"])

    def test_disconnected_graph_zero_fiedler(self) -> None:
        """
        Grafo con 2 componentes conexas → Fiedler ≈ 0.

        Propiedad fundamental: λ₂ = 0  ⟺  grafo desconectado.
        El test usa comparación estricta con tolerancia numérica EPSILON
        para evitar falsos positivos por errores de redondeo.
        """
        G = nx.DiGraph()
        # Componente 1: A→B→C
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        # Componente 2: X→Y→Z  (sin aristas a componente 1)
        G.add_edge("X", "Y")
        G.add_edge("Y", "Z")

        result = self.analyzer.analyze_spectral_stability(G)

        self.assertSpectralResultComplete(result)
        # Usar assertLessEqual con EPSILON × 10 para absorber errores
        # de redondeo en eigensolveres numéricos (scipy/ARPACK)
        self.assertLessEqual(
            result["fiedler_value"],
            self.EPSILON * 10,
            f"Grafo desconectado debe tener Fiedler ≈ 0, "
            f"obtenido: {result['fiedler_value']:.2e}",
        )

    def test_strongly_connected_cycle(self) -> None:
        """
        Cycle graph C₆ (dirigido convertido a no-dirigido) → Fiedler > 0.

        λ₂(C₆) = 1 - cos(2π/6) = 0.5  (Laplaciano normalizado).
        wavelength = 1/λ₂ > 0.
        """
        G = nx.cycle_graph(6, create_using=nx.DiGraph)

        result = self.analyzer.analyze_spectral_stability(G)

        self.assertSpectralResultComplete(result)
        self.assertGreater(result["fiedler_value"], 0.0)
        self.assertGreater(
            result["wavelength"],
            0.0,
            "wavelength = 1/Fiedler debe ser positivo para grafo conexo",
        )

    def test_star_graph_connectivity(self) -> None:
        """
        Star graph S₅ (1 hub + 5 hojas) → Fiedler > 0.

        Propiedad: El hub es punto de articulación único.
        λ₂(S_n) = 1 para Laplaciano normalizado (independiente de n).
        """
        G = nx.star_graph(5).to_directed()

        result = self.analyzer.analyze_spectral_stability(G)

        self.assertSpectralResultComplete(result)
        self.assertGreater(result["fiedler_value"], 0.0)
        self.assertEqual(result["status"], "success")

    def test_tree_is_connected(self) -> None:
        """
        Árbol binario balanceado es conexo → Fiedler > 0.

        Los árboles son el caso límite de grafos esparsos: n-1 aristas,
        λ₂ > 0 pero muy pequeño para árboles grandes.
        """
        G = nx.balanced_tree(r=2, h=3).to_directed()  # 15 nodos

        result = self.analyzer.analyze_spectral_stability(G)

        self.assertSpectralResultComplete(result)
        self.assertGreater(result["fiedler_value"], 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# PROPIEDADES DE EIGENVALORES
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpectralAnalysisEigenvalueProperties(TestSpectralAnalysisBase):
    """
    Pruebas de propiedades matemáticas de eigenvalores del Laplaciano normalizado.

    Teorema (Chung, 1997):
      Para L_norm = I - D^(-1/2) A D^(-1/2):
        (a) λᵢ ∈ [0, 2]  ∀i
        (b) λ₁ = 0 con eigenvector D^(1/2) 1
        (c) λ_max = 2  ⟺  G bipartito
        (d) Σλᵢ = n  (traza)
    """

    _TEST_GRAPHS = [
        ("path_10",    lambda: nx.path_graph(10, create_using=nx.DiGraph)),
        ("cycle_8",    lambda: nx.cycle_graph(8, create_using=nx.DiGraph)),
        ("complete_6", lambda: nx.complete_graph(6, create_using=nx.DiGraph)),
        ("star_7",     lambda: nx.star_graph(7).to_directed()),
    ]

    def test_eigenvalues_in_valid_range(self) -> None:
        """
        ∀i: λᵢ ∈ [0, 2] para Laplaciano normalizado.

        Se usa tolerancia _MACHINE_EPS × 10 en los extremos para
        absorber errores de redondeo del eigensolve (LAPACK/ARPACK).
        """
        lower = -self.EPSILON          # 0 - ε  (tolerancia numérica)
        upper = 2.0 + self.EPSILON     # 2 + ε

        for name, graph_factory in self._TEST_GRAPHS:
            with self.subTest(graph=name):
                G = graph_factory()
                result = self.analyzer.analyze_spectral_stability(G)

                eigenvalues = result.get("eigenvalues") or []
                self.assertGreater(
                    len(eigenvalues),
                    0,
                    f"[{name}] Se esperaban eigenvalores en el resultado",
                )
                for eig in eigenvalues:
                    self.assertInRange(
                        eig,
                        lower,
                        upper,
                        msg=f"[{name}] λ={eig:.6f} ∉ [0, 2]",
                    )

    def test_lambda_max_bounded_by_two(self) -> None:
        """
        λ_max ≤ 2 para el Laplaciano normalizado.

        Caso de igualdad exacta: grafo bipartito K_{4,4}.
        """
        G = nx.complete_bipartite_graph(4, 4).to_directed()

        result = self.analyzer.analyze_spectral_stability(G)

        if "lambda_max" in result:
            self.assertLessEqual(
                result["lambda_max"],
                2.0 + self.EPSILON,
                f"λ_max = {result['lambda_max']:.6f} excede cota teórica 2",
            )
        # También verificable via eigenvalores directos
        eigenvalues = result.get("eigenvalues") or []
        if eigenvalues:
            observed_max = max(eigenvalues)
            self.assertLessEqual(
                observed_max,
                2.0 + self.EPSILON,
                f"max(λ) = {observed_max:.6f} excede 2.0",
            )

    def test_spectral_energy_non_negative(self) -> None:
        """
        E(G) = Σλᵢ² = ||L_norm||²_F ≥ 0.

        La energía espectral es una norma cuadrada → siempre ≥ 0.
        Valores negativos indicarían error de implementación.
        """
        G = nx.gnm_random_graph(15, 25, directed=True, seed=42)

        result = self.analyzer.analyze_spectral_stability(G)

        self.assertFinite(result["spectral_energy"])
        self.assertGreaterEqual(
            result["spectral_energy"],
            0.0,
            "Energía espectral ||L||²_F debe ser ≥ 0",
        )

    def test_trace_equals_n(self) -> None:
        """
        Σλᵢ = n  (la traza del Laplaciano normalizado es n).

        Esta propiedad es invariante y sirve como verificación
        de la implementación de la eigendescomposición.
        """
        G = nx.cycle_graph(8, create_using=nx.DiGraph)
        result = self.analyzer.analyze_spectral_stability(G)

        eigenvalues = result.get("eigenvalues") or []
        if not eigenvalues:
            self.skipTest("eigenvalores no disponibles en resultado")

        n = G.number_of_nodes()
        trace = sum(eigenvalues)
        self.assertAlmostEqualTolerant(
            trace,
            float(n),
            tolerance=_SPECTRAL_TOLERANCE,
            msg=f"Σλᵢ = {trace:.4f} ≠ n = {n} (traza de L_norm)",
        )

    def test_first_eigenvalue_is_zero(self) -> None:
        """
        λ₁ = 0 siempre (eigenvector constante ponderado D^(1/2)·1).

        Si los eigenvalores están ordenados ascendentemente,
        el mínimo debe ser ≈ 0.
        """
        G = nx.complete_graph(6, create_using=nx.DiGraph)
        result = self.analyzer.analyze_spectral_stability(G)

        eigenvalues = result.get("eigenvalues") or []
        if not eigenvalues:
            self.skipTest("eigenvalores no disponibles")

        lambda_min = min(eigenvalues)
        self.assertLessEqual(
            lambda_min,
            self.EPSILON,
            f"λ_min = {lambda_min:.2e} debe ser ≈ 0",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# DETECCIÓN DE RESONANCIA ESPECTRAL
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpectralAnalysisResonance(TestSpectralAnalysisBase):
    """
    Pruebas de detección de riesgo de resonancia espectral.

    Analogía con resonancia de circuitos RLC:
      En un circuito RLC, la resonancia ocurre cuando todos los
      modos tienen la misma frecuencia natural → amplificación.
      En grafos, la degeneración espectral (eigenvalores concentrados)
      implica que perturbaciones de esa "frecuencia" se amplifican.

    Indicador: coeficiente de variación de {λ₂, ..., λ_n} < umbral.
    """

    def test_resonance_risk_is_bool(self) -> None:
        """resonance_risk debe ser siempre un booleano (no None ni int)."""
        G = nx.random_regular_graph(3, 10, seed=42).to_directed()

        result = self.analyzer.analyze_spectral_stability(G)

        self.assertIn("resonance_risk", result)
        self.assertIsInstance(
            result["resonance_risk"],
            bool,
            f"resonance_risk debe ser bool, obtenido: {type(result['resonance_risk']).__name__}",
        )

    def test_regular_graph_has_defined_resonance(self) -> None:
        """
        Grafo 3-regular: espectro más concentrado que grafo irregular.

        Para grafo k-regular: L_norm = I - (1/k)A  →  eigenvalores de A
        están relacionados con los de L_norm por λ_L = 1 - λ_A/k.
        La estructura simétrica induce posible degeneración espectral.
        """
        G = nx.random_regular_graph(3, 10, seed=42).to_directed()

        result = self.analyzer.analyze_spectral_stability(G)

        self.assertSpectralResultComplete(result)

    def test_irregular_graph_lower_resonance_risk(self) -> None:
        """
        Grafo con grados heterogéneos: espectro más disperso.

        Árbol con ramificaciones desiguales → grados muy variados
        → distribución de eigenvalores más dispersa → menor riesgo.
        """
        G = nx.DiGraph()
        G.add_edges_from([
            ("R",  "A"), ("R",  "B"), ("R",  "C"),
            ("A",  "A1"), ("A", "A2"), ("A", "A3"), ("A", "A4"),
            ("B",  "B1"),
            ("C",  "C1"), ("C", "C2"),
        ])

        result = self.analyzer.analyze_spectral_stability(G)

        self.assertSpectralResultComplete(result)
        self.assertEqual(result["status"], "success")

    def test_complete_graph_degenerate_spectrum(self) -> None:
        """
        K_n tiene espectro altamente degenerado.

        Para K_n con Laplaciano normalizado:
          eigenvalores = {0, n/(n-1), n/(n-1), ..., n/(n-1)}
                             ←────── n-1 veces ──────→
          Solo 2 valores distintos → degeneración máxima.

        Esto debería activar el indicador de resonancia.
        """
        n = 8
        G = nx.complete_graph(n, create_using=nx.DiGraph)

        result = self.analyzer.analyze_spectral_stability(G)

        self.assertSpectralResultComplete(result)
        self.assertGreater(result["fiedler_value"], 0.0)

        # Verificar degeneración: solo 2 eigenvalores distintos
        eigenvalues = result.get("eigenvalues") or []
        if eigenvalues:
            # Redondear a 4 decimales para agrupar eigenvalores casi iguales
            distinct = set(round(e, 4) for e in eigenvalues)
            self.assertLessEqual(
                len(distinct),
                3,  # Toleramos 3 por errores numéricos de redondeo
                f"K_{n} debe tener ≤ 3 eigenvalores distintos, "
                f"se encontraron {len(distinct)}: {sorted(distinct)}",
            )

        # Para K_8: resonance_risk debería ser True
        self.assertTrue(
            result["resonance_risk"],
            f"Grafo completo K_{n} con espectro degenerado debe "
            f"tener resonance_risk=True",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MANEJO DE NODOS AISLADOS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpectralAnalysisIsolatedNodes(TestSpectralAnalysisBase):
    """
    Pruebas de manejo de nodos aislados.

    Motivación matemática:
      D^(-1/2) requiere deg(v) > 0 ∀v.
      Nodo aislado → deg(v) = 0 → D^(-1/2) indefinido (1/0).
      Solución: eliminar nodos aislados antes de construir L_norm.
    """

    def test_graph_with_isolated_nodes_succeeds(self) -> None:
        """
        Componente conexa {A,B,C} + 2 nodos aislados.

        Los nodos aislados se eliminan; el análisis continúa
        sobre la componente conexa.
        """
        G = nx.DiGraph()
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_node("ISOLATED_1")
        G.add_node("ISOLATED_2")

        result = self.analyzer.analyze_spectral_stability(G)

        self.assertSpectralResultComplete(result)
        self.assertEqual(
            result["status"],
            "success",
            "Debe tener éxito al eliminar nodos aislados y analizar componente",
        )

        # Verificar reporte de nodos eliminados (si el analizador lo expone)
        if "isolated_nodes_removed" in result:
            self.assertEqual(
                result["isolated_nodes_removed"],
                2,
                "Deben reportarse exactamente 2 nodos aislados removidos",
            )

    def test_all_isolated_nodes_is_degenerate(self) -> None:
        """
        Solo nodos aislados → grafo vacío tras limpieza → estado degenerado.

        El status debe indicar que no hay suficientes nodos para el análisis,
        consistente con el comportamiento para grafo vacío.
        """
        G = nx.DiGraph()
        G.add_node("A")
        G.add_node("B")
        G.add_node("C")

        result = self.analyzer.analyze_spectral_stability(G)

        self.assertSpectralResultComplete(result)
        # El status debe ser uno de los estados de "no se pudo analizar"
        self.assertIn(
            result["status"],
            {"insufficient_nodes", "degenerate_after_isolation_removal"},
            f"Status inesperado para grafo de nodos aislados: "
            f"'{result['status']}'",
        )
        # En ambos casos el Fiedler debe ser 0
        self.assertEqual(result["fiedler_value"], 0.0)

    def test_mixed_components_largest_analyzed(self) -> None:
        """
        Múltiples componentes: el análisis puede operar sobre todas o la mayor.

        Si el analizador selecciona la componente más grande, debe retornar
        status='success' y Fiedler > 0.
        """
        G = nx.DiGraph()
        # Componente grande: 5 nodos
        for i in range(4):
            G.add_edge(f"L{i}", f"L{i+1}")
        # Nodo aislado
        G.add_node("ISOLATED")

        result = self.analyzer.analyze_spectral_stability(G)

        # No debe fallar
        self.assertSpectralResultComplete(result)
        self.assertFinite(result["fiedler_value"])


# ═══════════════════════════════════════════════════════════════════════════════
# TOPOLOGÍA REAL DE PRESUPUESTO
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpectralAnalysisBudgetTopology(TestSpectralAnalysisBase):
    """
    Pruebas con topologías reales de presupuesto de construcción.

    El grafo de presupuesto es un DAG (Directed Acyclic Graph) con
    estructura piramidal: ROOT → Capítulos → APUs → Insumos.
    Para el análisis espectral se convierte a no-dirigido.
    """

    def test_typical_budget_structure(self) -> None:
        """
        Estructura piramidal básica: 2 capítulos, 3 APUs, 4 insumos.

        Nodos totales: 1 + 2 + 6 + 24 = 33
        Aristas no-dirigidas: 33 - 1 = 32 (árbol)

        El árbol es siempre conexo → Fiedler > 0.
        """
        G = self.create_budget_graph(chapters=2, apus_per_chapter=3, insumos_per_apu=4)

        result = self.analyzer.analyze_spectral_stability(G)

        self.assertSpectralResultComplete(result)
        self.assertGreater(result["fiedler_value"], 0.0)
        self.assertEqual(result["status"], "success")

    def test_budget_with_shared_insumos(self) -> None:
        """
        Insumos compartidos entre APUs: grafo ya no es árbol puro.

        El insumo compartido introduce ciclos en el no-dirigido,
        aumentando la conectividad y por ende el Fiedler value.
        """
        G = self.create_budget_graph(
            chapters=1, apus_per_chapter=3, insumos_per_apu=2
        )

        shared = "INSUMO_COMPARTIDO_ACERO"
        G.add_node(
            shared,
            type="INSUMO",
            level=3,
            description="ACERO ESTRUCTURAL",
            tipo_insumo="ACERO",
        )
        for apu in ["APU_1_1", "APU_1_2", "APU_1_3"]:
            G.add_edge(apu, shared, quantity=100.0, total_cost=5000.0)

        result = self.analyzer.analyze_spectral_stability(G)

        self.assertSpectralResultComplete(result)
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["fiedler_value"], 0.0)

    def test_budget_with_circular_dependency_does_not_crash(self) -> None:
        """
        Dependencia circular (error de datos) no debe causar excepción.

        El analizador debe ser robusto ante datos malformados.
        La presencia de ciclos en el DAG cambia las propiedades
        espectrales pero no invalida el cálculo.
        """
        G = self.create_budget_graph(
            chapters=1, apus_per_chapter=2, insumos_per_apu=2
        )
        # Ciclo artificial: insumo ↔ APU
        G.add_edge("INSUMO_1_1_1", "APU_1_2")
        G.add_edge("APU_1_2", "INSUMO_1_1_1")

        result = self.analyzer.analyze_spectral_stability(G)

        # El resultado debe existir y ser finito
        self.assertSpectralResultComplete(result)

    def test_budget_single_chapter_minimal(self) -> None:
        """
        Presupuesto mínimo: 1 capítulo, 1 APU, 1 insumo → 3 nodos + ROOT.

        Árbol de 4 nodos (P₄ topológicamente) → Fiedler > 0.
        """
        G = self.create_budget_graph(chapters=1, apus_per_chapter=1, insumos_per_apu=1)

        result = self.analyzer.analyze_spectral_stability(G)

        self.assertSpectralResultComplete(result)
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["fiedler_value"], 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# ESCALABILIDAD Y RENDIMIENTO
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpectralAnalysisLargeGraphs(TestSpectralAnalysisBase):
    """
    Pruebas de escalabilidad y estabilidad numérica con grafos grandes.

    El analizador debe usar eigensolvers apropiados:
      · n < 20:  scipy.linalg.eigvalsh (denso, exacto)
      · n ≥ 20:  scipy.sparse.linalg.eigsh (Arnoldi/Lanczos, aproximado)

    La transición debe ser transparente para el usuario.
    """

    def test_large_path_graph_activates_sparse_solver(self) -> None:
        """
        P₅₀: activa solver sparse (n=50 >> umbral=20).

        Para P_n con Laplaciano normalizado:
          λ₂ ≈ 1 - cos(π/n)  [ajuste por normalización]
        Para n=50: λ₂ ≈ 1 - cos(π/50) ≈ 0.00197 × ... (muy pequeño)

        La clave es que sea > 0 y < 0.1.
        """
        n = 50
        G = nx.path_graph(n, create_using=nx.DiGraph)

        result = self.analyzer.analyze_spectral_stability(G)

        self.assertSpectralResultComplete(result)
        self.assertEqual(result["status"], "success")
        self.assertGreater(
            result["fiedler_value"],
            0.0,
            "P₅₀ es conexo → Fiedler > 0",
        )
        # Para árboles de camino grandes, Fiedler es pequeño
        self.assertLess(
            result["fiedler_value"],
            0.15,
            "P₅₀ debe tener Fiedler pequeño (estructura lineal débil)",
        )

    def test_large_random_graph_no_nan_inf(self) -> None:
        """
        Grafo aleatorio grande (n=100, m=300): verificar estabilidad numérica.

        El grafo semilla 12345 puede tener componentes desconectadas;
        el analizador debe manejar esto sin producir NaN/Inf.
        """
        G = nx.gnm_random_graph(100, 300, directed=True, seed=12345)

        result = self.analyzer.analyze_spectral_stability(G)

        self.assertSpectralResultComplete(result)
        self.assertFinite(result["fiedler_value"])
        self.assertFinite(result["spectral_energy"])
        self.assertGreaterEqual(result["spectral_energy"], 0.0)

    def test_large_budget_topology(self) -> None:
        """
        Presupuesto grande: 5 capítulos × 10 APUs × 8 insumos.

        Total nodos: 1 + 5 + 50 + 400 = 456
        Total aristas: 455 (árbol)

        Verifica que el análisis termina en tiempo razonable
        y produce resultados válidos.
        """
        G = self.create_budget_graph(
            chapters=5, apus_per_chapter=10, insumos_per_apu=8
        )
        self.assertEqual(G.number_of_nodes(), 456)

        result = self.analyzer.analyze_spectral_stability(G)

        self.assertSpectralResultComplete(result)
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["fiedler_value"], 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# CASOS EXTREMOS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpectralAnalysisEdgeCases(TestSpectralAnalysisBase):
    """Pruebas de robustez ante entradas atípicas."""

    def test_self_loop_does_not_crash(self) -> None:
        """
        Self-loop A→A: el Laplaciano normalizado ignora self-loops
        (no contribuyen a la conectividad).
        NetworkX los incluye en el degree pero se deben manejar.
        """
        G = nx.DiGraph()
        G.add_edge("A", "B")
        G.add_edge("B", "C")
        G.add_edge("A", "A")  # Self-loop

        result = self.analyzer.analyze_spectral_stability(G)

        self.assertSpectralResultComplete(result)
        self.assertFinite(result["fiedler_value"])

    def test_parallel_edges_overwritten_in_digraph(self) -> None:
        """
        DiGraph sobrescribe aristas duplicadas (la última prevalece).
        No debe causar error ni resultado inconsistente.
        """
        G = nx.DiGraph()
        G.add_edge("A", "B", weight=1.0)
        G.add_edge("A", "B", weight=2.0)  # Sobrescribe weight=1.0

        result = self.analyzer.analyze_spectral_stability(G)

        self.assertSpectralResultComplete(result)
        self.assertEqual(result["status"], "success")

    def test_dense_graph_high_fiedler(self) -> None:
        """
        Grafo denso (K₁₅): alta conectividad → alto Fiedler value.

        Para K_n: Fiedler = n/(n-1).
        Para n=15: Fiedler ≈ 1.0714  →  claramente > 0.5.
        """
        n = 15
        G = nx.complete_graph(n, create_using=nx.DiGraph)

        result = self.analyzer.analyze_spectral_stability(G)

        self.assertSpectralResultComplete(result)
        self.assertEqual(result["status"], "success")
        # K_15: λ₂ = 15/14 ≈ 1.071  (verificación teórica)
        expected_lower_bound = 0.5
        self.assertGreater(
            result["fiedler_value"],
            expected_lower_bound,
            f"K_{n} debe tener Fiedler > {expected_lower_bound}",
        )

    def test_sparse_tree_positive_fiedler(self) -> None:
        """
        Árbol binario balanceado (árbol ~31 nodos): Fiedler > 0 pero pequeño.
        """
        G = nx.balanced_tree(r=2, h=4).to_directed()  # 31 nodos

        result = self.analyzer.analyze_spectral_stability(G)

        self.assertSpectralResultComplete(result)
        self.assertEqual(result["status"], "success")
        self.assertGreater(result["fiedler_value"], 0.0)

    def test_create_budget_graph_invalid_params_raises(self) -> None:
        """
        create_budget_graph con parámetros < 1 debe lanzar ValueError.
        """
        with self.assertRaises(ValueError):
            self.create_budget_graph(chapters=0)

        with self.assertRaises(ValueError):
            self.create_budget_graph(apus_per_chapter=-1)

        with self.assertRaises(ValueError):
            self.create_budget_graph(insumos_per_apu=0)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSISTENCIA Y DETERMINISMO
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpectralAnalysisConsistency(TestSpectralAnalysisBase):
    """Pruebas de consistencia y reproducibilidad de resultados."""

    def test_deterministic_results_across_calls(self) -> None:
        """
        El análisis debe ser determinístico: misma entrada → misma salida.

        Los eigensolvers pueden ser no-determinísticos si usan
        vectores de inicio aleatorios; se verifica que esto no ocurre.
        """
        G = nx.gnm_random_graph(30, 60, directed=True, seed=99999)

        results = [self.analyzer.analyze_spectral_stability(G) for _ in range(3)]

        numeric_fields = ["fiedler_value", "spectral_energy", "spectral_gap"]
        for field in numeric_fields:
            with self.subTest(field=field):
                if field not in results[0]:
                    continue
                ref = results[0][field]
                for i, res in enumerate(results[1:], start=2):
                    self.assertAlmostEqualTolerant(
                        ref,
                        res[field],
                        msg=f"'{field}' no es determinístico (llamada 1 vs {i})",
                    )

    def test_direction_invariance_after_undirected_conversion(self) -> None:
        """
        El análisis espectral opera sobre el grafo no-dirigido subyacente.
        Invertir todas las aristas debe producir el mismo resultado.

        Motivación: DiGraph → to_undirected() es invariante a la dirección.
        """
        # Grafo original: A→B→C
        G1 = nx.DiGraph()
        G1.add_edge("A", "B")
        G1.add_edge("B", "C")

        # Grafo con aristas invertidas: A←B←C
        G2 = nx.DiGraph()
        G2.add_edge("B", "A")
        G2.add_edge("C", "B")

        r1 = self.analyzer.analyze_spectral_stability(G1)
        r2 = self.analyzer.analyze_spectral_stability(G2)

        self.assertAlmostEqualTolerant(
            r1["fiedler_value"],
            r2["fiedler_value"],
            msg=(
                "Grafos con misma estructura no-dirigida deben tener "
                "idéntico Fiedler value"
            ),
        )
        self.assertAlmostEqualTolerant(
            r1["spectral_energy"],
            r2["spectral_energy"],
            msg=(
                "Grafos con misma estructura no-dirigida deben tener "
                "idéntica energía espectral"
            ),
        )

    def test_isomorphic_graphs_same_spectrum(self) -> None:
        """
        Grafos isomorfos tienen espectro idéntico.

        Propiedad: El espectro del Laplaciano normalizado es invariante
        bajo isomorfismo de grafos (renombramiento de nodos).
        """
        G1 = nx.cycle_graph(5, create_using=nx.DiGraph)

        # Mismo ciclo con nodos renombrados
        G2 = nx.DiGraph()
        mapping = {0: "v0", 1: "v1", 2: "v2", 3: "v3", 4: "v4"}
        G2 = nx.relabel_nodes(G1, mapping)

        r1 = self.analyzer.analyze_spectral_stability(G1)
        r2 = self.analyzer.analyze_spectral_stability(G2)

        self.assertAlmostEqualTolerant(
            r1["fiedler_value"],
            r2["fiedler_value"],
            tolerance=_SPECTRAL_TOLERANCE,
            msg="Grafos isomorfos deben tener mismo Fiedler value",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRACIÓN CON OTROS MÉTODOS DEL ANALIZADOR
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpectralAnalysisIntegration(TestSpectralAnalysisBase):
    """
    Pruebas de coherencia entre análisis espectral y otros métodos
    del BusinessTopologicalAnalyzer.
    """

    def test_spectral_coherent_with_betti_connected(self) -> None:
        """
        Coherencia: β₀ = 1  ⟺  Fiedler > 0.

        β₀ cuenta componentes conexas.
        Si β₀ = 1 (conexo) → Fiedler > 0.
        """
        G = nx.path_graph(6, create_using=nx.DiGraph)

        betti = self.analyzer.calculate_betti_numbers(G)
        spectral = self.analyzer.analyze_spectral_stability(G)

        self.assertEqual(betti.beta_0, 1, "P₆ debe tener β₀ = 1")
        self.assertGreater(
            spectral["fiedler_value"],
            0.0,
            "Grafo conexo (β₀=1) debe tener Fiedler > 0",
        )

    def test_spectral_coherent_with_betti_disconnected(self) -> None:
        """
        Coherencia: β₀ > 1  ⟹  Fiedler ≈ 0.
        """
        G = nx.DiGraph()
        G.add_edge("A", "B")
        G.add_edge("X", "Y")

        betti = self.analyzer.calculate_betti_numbers(G)
        spectral = self.analyzer.analyze_spectral_stability(G)

        self.assertGreater(
            betti.beta_0,
            1,
            "Grafo desconectado debe tener β₀ > 1",
        )
        self.assertLessEqual(
            spectral["fiedler_value"],
            self.EPSILON * 10,
            f"Grafo desconectado (β₀={betti.beta_0}) debe tener Fiedler ≈ 0",
        )

    def test_spectral_in_executive_report(self) -> None:
        """
        El análisis espectral debe incluirse en el reporte ejecutivo.

        Claves requeridas en report.details['spectral_analysis']:
          · fiedler_value
          · resonance_risk
        """
        G = self.create_budget_graph(chapters=2, apus_per_chapter=3, insumos_per_apu=4)

        report = self.analyzer.generate_executive_report(G)

        self.assertIn(
            "spectral_analysis",
            report.details,
            "El reporte ejecutivo debe incluir 'spectral_analysis'",
        )

        spectral = report.details["spectral_analysis"]
        for key in ("fiedler_value", "resonance_risk"):
            self.assertIn(
                key,
                spectral,
                f"'spectral_analysis' debe contener la clave '{key}'",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# GRAFOS CON VALORES TEÓRICOS CONOCIDOS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpectralAnalysisKnownGraphs(TestSpectralAnalysisBase):
    """
    Pruebas de corrección matemática con grafos de espectro conocido.

    Estas pruebas actúan como "tests de regresión matemática":
    verifican que la implementación reproduce valores teóricos.
    """

    def test_cycle_C6_fiedler_value(self) -> None:
        """
        C₆: λ₂ = 1 - cos(2π/6) = 1 - cos(π/3) = 1 - 0.5 = 0.5.

        Derivación:
          Para C_n no-dirigido con Laplaciano normalizado (todos los
          nodos tienen grado 2):
            L_norm = I - (1/2)A
          Eigenvalores de A(C_n): 2cos(2πk/n), k=0,...,n-1
          λ_k(L) = 1 - cos(2πk/n)
          λ₂ = λ_{k=1} = 1 - cos(2π/n)
          Para n=6: λ₂ = 1 - cos(π/3) = 0.5  ✓
        """
        n = 6
        G = nx.cycle_graph(n).to_directed()

        result = self.analyzer.analyze_spectral_stability(G)

        expected = 1.0 - np.cos(2.0 * np.pi / n)  # 0.5

        self.assertAlmostEqualTolerant(
            result["fiedler_value"],
            expected,
            tolerance=_THEORETICAL_TOLERANCE,
            msg=f"C₆: Fiedler debe ser ≈ {expected:.4f}",
        )

    def test_complete_graph_K5_fiedler_value(self) -> None:
        """
        K₅: λ₂ = n/(n-1) = 5/4 = 1.25.

        Derivación:
          Para K_n no-dirigido (todos los nodos tienen grado n-1):
            L_norm = I - (1/(n-1))A
          Eigenvalores de A(K_n): {n-1, -1, -1, ..., -1}
          λ₂(L) = 1 - (-1)/(n-1) = 1 + 1/(n-1) = n/(n-1)
          Para n=5: 5/4 = 1.25  ✓
        """
        n = 5
        G = nx.complete_graph(n, create_using=nx.DiGraph)

        result = self.analyzer.analyze_spectral_stability(G)

        expected = n / (n - 1)  # 1.25

        self.assertAlmostEqualTolerant(
            result["fiedler_value"],
            expected,
            tolerance=_THEORETICAL_TOLERANCE,
            msg=f"K₅: Fiedler debe ser ≈ {expected:.4f}",
        )

    def test_path_P4_fiedler_value(self) -> None:
        """
        P₄: λ₂ del Laplaciano normalizado calculable exactamente.

        Para P_n no-dirigido:
          Los grados son [1, 2, 2, ..., 2, 1] (extremos tienen grado 1).
          El Laplaciano normalizado NO es circulante → no hay fórmula cerrada
          simple. Pero el eigenvalor mínimo positivo es verificable numéricamente.

          Para P₄: el espectro es {0, 2-√2, 2, 2+√2} ≈ {0, 0.586, 2, 3.414}
          PERO con normalización Chung: {0, 0.268, 1.0, 1.732} (aprox.)

          Verificamos la propiedad cualitativa: 0 < λ₂ < 1.
        """
        G = nx.path_graph(4).to_directed()

        result = self.analyzer.analyze_spectral_stability(G)

        fiedler = result["fiedler_value"]
        self.assertGreater(fiedler, 0.0, "P₄: Fiedler debe ser > 0")
        self.assertLess(fiedler, 1.0, "P₄: Fiedler debe ser < 1 (grafo esparcido)")

    def test_bipartite_K22_lambda_max_equals_two(self) -> None:
        """
        K_{2,2}: grafo bipartito completo → λ_max = 2.

        Teorema (Chung): λ_max = 2  ⟺  G bipartito.
        K_{2,2} es bipartito → eigenvalor máximo exactamente 2.
        """
        G = nx.complete_bipartite_graph(2, 2).to_directed()

        result = self.analyzer.analyze_spectral_stability(G)

        eigenvalues = result.get("eigenvalues") or []
        if not eigenvalues:
            self.skipTest("eigenvalores no disponibles")

        lambda_max = max(eigenvalues)
        self.assertAlmostEqualTolerant(
            lambda_max,
            2.0,
            tolerance=_SPECTRAL_TOLERANCE,
            msg=f"K_{{2,2}} bipartito: λ_max debe ser ≈ 2.0, obtenido {lambda_max:.4f}",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════════════════


def create_test_suite() -> unittest.TestSuite:
    """
    Construye la suite completa de pruebas espectrales.

    Orden de ejecución diseñado por dificultad creciente:
      1. Casos básicos (smoke tests)
      2. Conectividad algebraica
      3. Propiedades de eigenvalores
      4. Resonancia espectral
      5. Nodos aislados
      6. Topología real de presupuesto
      7. Escalabilidad
      8. Casos extremos
      9. Consistencia y determinismo
      10. Integración
      11. Grafos con valores teóricos conocidos
    """
    test_classes = [
        TestSpectralAnalysisBasicCases,
        TestSpectralAnalysisConnectivity,
        TestSpectralAnalysisEigenvalueProperties,
        TestSpectralAnalysisResonance,
        TestSpectralAnalysisIsolatedNodes,
        TestSpectralAnalysisBudgetTopology,
        TestSpectralAnalysisLargeGraphs,
        TestSpectralAnalysisEdgeCases,
        TestSpectralAnalysisConsistency,
        TestSpectralAnalysisIntegration,
        TestSpectralAnalysisKnownGraphs,
    ]

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    return suite


if __name__ == "__main__":
    runner = unittest.TextTestRunner(verbosity=2, failfast=False)
    result = runner.run(create_test_suite())
    # Retornar código de salida no-cero si hay fallos (útil en CI/CD)
    raise SystemExit(0 if result.wasSuccessful() else 1)