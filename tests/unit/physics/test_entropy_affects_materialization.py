"""
Suite de pruebas rigurosas para la interacción entre entropía de flujo
y materialización del BOM en MatterGenerator.

Estrategia de testing:
──────────────────────
  T1  — Fixtures y fábrica de grafos: topologías controladas y reproducibles.
  T2  — _extract_waste_mean_and_flux_metrics: contrato de estructura del BOM.
  T3  — Propiedades de frontera: entropy_ratio ∈ {0, 1} y valores extremos.
  T4  — Propiedad monotónica principal: entropía alta → desperdicio mayor.
  T5  — Propagación de métricas de flujo al análisis de riesgo.
  T6  — Aislamiento entre llamadas: estado interno de MatterGenerator.
  T7  — Categorías de material: FRAGILE, STANDARD y casos desconocidos.
  T8  — Robustez: grafos degenerados, entradas inválidas, valores extremos.
  T9  — Invariantes algebraicos globales del sistema de materialización.

Correcciones v3:
────────────────
  - `assert` en auxiliares reemplazados por raises con mensajes ricos.
  - Grafo base extendido con múltiples nodos y categorías.
  - Umbral 0.05 documentado con justificación del modelo de desperdicio.
  - Propiedad monotónica con tolerancia mínima significativa (δ = 0.01).
  - Tests de frontera: entropy_ratio ∈ {0.0, 1.0}, NaN, inf, negativos.
  - Aislamiento verificado: cada llamada usa instancia fresca de MatterGenerator.
  - Cobertura de categorías: FRAGILE, STANDARD, PERISHABLE, desconocida.
  - Grafos degenerados: vacío, sin aristas, quantity=0, unit_cost=0.
  - Invariantes algebraicos: no-negatividad, monotonicidad, acotamiento.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# Importaciones opcionales con skip a nivel de módulo
# ─────────────────────────────────────────────────────────────────────────────

nx = pytest.importorskip(
    "networkx",
    reason="Se requiere networkx para construir grafos de materialización.",
)

matter_generator_module = pytest.importorskip(
    "app.matter_generator",
    reason="Se requiere app.matter_generator en el entorno de pruebas.",
)

# Importación defensiva con mensaje descriptivo
MatterGenerator = getattr(matter_generator_module, "MatterGenerator", None)
if MatterGenerator is None:
    pytest.skip(
        "MatterGenerator no está definido en app.matter_generator. "
        "Verifique que la clase exista y esté exportada correctamente.",
        allow_module_level=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DEL MODELO DE DESPERDICIO
# ─────────────────────────────────────────────────────────────────────────────
# Justificación matemática:
#   El modelo de desperdicio de MatterGenerator aplica un factor multiplicativo
#   f(entropy) sobre el desperdicio base por categoría:
#     waste = base_waste × (1 + α × entropy_ratio)
#   Para FRAGILE: base_waste ≈ 0.03 (3% base según norma NTC-4595)
#                 α = 1.5 (amplificador para materiales frágiles)
#   Con entropy_ratio = 0.95: waste ≈ 0.03 × (1 + 1.5 × 0.95) ≈ 0.073
#   El umbral conservador es 0.05 (mitad del valor esperado) para tolerar
#   variaciones de implementación.
_FRAGILE_HIGH_ENTROPY_MIN_WASTE: float = 0.05

# Diferencia mínima significativa entre escenarios (δ):
#   Se considera que el sistema detecta el efecto de la entropía solo si
#   la diferencia absoluta supera 1 punto porcentual (δ = 0.01).
#   Diferencias menores podrían ser ruido numérico.
_MONOTONICITY_MIN_DELTA: float = 0.01

# Tolerancia para comparaciones de punto flotante
_FLOAT_TOLERANCE: float = 1e-9

# Métricas de flujo canónicas usadas en todos los escenarios
_FLUX_METRIC_KEYS = frozenset(
    {"entropy_ratio", "avg_saturation", "pyramid_stability"}
)


# ═════════════════════════════════════════════════════════════════════════════
# T1 — FIXTURES Y FÁBRICA DE GRAFOS
# ═════════════════════════════════════════════════════════════════════════════


def _build_minimal_material_graph() -> "nx.DiGraph":
    """
    Construye el grafo base mínimo y conexo para escenarios controlados.

    Topología:
        ROOT (APU) → MAT1 (FRAGILE, quantity=10)

    Este grafo es intencionalemnte simple para que la única variable
    sea las métricas de flujo entre escenarios.
    """
    graph = nx.DiGraph()
    graph.add_node("ROOT", type="APU")
    graph.add_node(
        "MAT1",
        type="INSUMO",
        unit_cost=100.0,
        unit="UND",
        material_category="FRAGILE",
    )
    graph.add_edge("ROOT", "MAT1", quantity=10.0)
    return graph


def _build_multi_category_graph() -> "nx.DiGraph":
    """
    Construye un grafo con múltiples categorías de material para pruebas
    de diferenciación de comportamiento por categoría.

    Topología:
        ROOT (APU) → MAT_FRAGILE    (FRAGILE,    quantity=5,  unit_cost=200)
        ROOT (APU) → MAT_STANDARD   (STANDARD,   quantity=20, unit_cost=50)
        ROOT (APU) → MAT_PERISHABLE (PERISHABLE, quantity=3,  unit_cost=500)
    """
    graph = nx.DiGraph()
    graph.add_node("ROOT", type="APU")
    graph.add_node(
        "MAT_FRAGILE",
        type="INSUMO",
        unit_cost=200.0,
        unit="UND",
        material_category="FRAGILE",
    )
    graph.add_node(
        "MAT_STANDARD",
        type="INSUMO",
        unit_cost=50.0,
        unit="M2",
        material_category="STANDARD",
    )
    graph.add_node(
        "MAT_PERISHABLE",
        type="INSUMO",
        unit_cost=500.0,
        unit="KG",
        material_category="PERISHABLE",
    )
    graph.add_edge("ROOT", "MAT_FRAGILE", quantity=5.0)
    graph.add_edge("ROOT", "MAT_STANDARD", quantity=20.0)
    graph.add_edge("ROOT", "MAT_PERISHABLE", quantity=3.0)
    return graph


def _build_empty_graph() -> "nx.DiGraph":
    """Grafo sin nodos ni aristas (caso degenerado)."""
    return nx.DiGraph()


def _build_root_only_graph() -> "nx.DiGraph":
    """Grafo con solo el nodo raíz, sin materiales (caso degenerado)."""
    graph = nx.DiGraph()
    graph.add_node("ROOT", type="APU")
    return graph


def _build_zero_quantity_graph() -> "nx.DiGraph":
    """Grafo con cantidad cero en la arista (caso límite matemático)."""
    graph = nx.DiGraph()
    graph.add_node("ROOT", type="APU")
    graph.add_node(
        "MAT1",
        type="INSUMO",
        unit_cost=100.0,
        unit="UND",
        material_category="FRAGILE",
    )
    graph.add_edge("ROOT", "MAT1", quantity=0.0)
    return graph


def _build_zero_cost_graph() -> "nx.DiGraph":
    """Grafo con costo unitario cero (material sin costo asignado)."""
    graph = nx.DiGraph()
    graph.add_node("ROOT", type="APU")
    graph.add_node(
        "MAT1",
        type="INSUMO",
        unit_cost=0.0,
        unit="UND",
        material_category="STANDARD",
    )
    graph.add_edge("ROOT", "MAT1", quantity=10.0)
    return graph


def _build_unknown_category_graph() -> "nx.DiGraph":
    """Grafo con categoría de material no reconocida por el sistema."""
    graph = nx.DiGraph()
    graph.add_node("ROOT", type="APU")
    graph.add_node(
        "MAT1",
        type="INSUMO",
        unit_cost=100.0,
        unit="UND",
        material_category="CATEGORIA_DESCONOCIDA_XYZ",
    )
    graph.add_edge("ROOT", "MAT1", quantity=5.0)
    return graph


def _make_low_entropy_metrics() -> Dict[str, float]:
    """Métricas de flujo de baja entropía (sistema ordenado, estable)."""
    return {
        "entropy_ratio": 0.10,
        "avg_saturation": 0.10,
        "pyramid_stability": 0.95,
    }


def _make_high_entropy_metrics() -> Dict[str, float]:
    """Métricas de flujo de alta entropía (sistema desordenado, inestable)."""
    return {
        "entropy_ratio": 0.95,
        "avg_saturation": 0.90,
        "pyramid_stability": 0.50,
    }


def _make_zero_entropy_metrics() -> Dict[str, float]:
    """Caso límite: entropía nula (sistema perfectamente ordenado)."""
    return {
        "entropy_ratio": 0.0,
        "avg_saturation": 0.0,
        "pyramid_stability": 1.0,
    }


def _make_max_entropy_metrics() -> Dict[str, float]:
    """Caso límite: entropía máxima (sistema en caos total)."""
    return {
        "entropy_ratio": 1.0,
        "avg_saturation": 1.0,
        "pyramid_stability": 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures pytest
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture()
def minimal_graph() -> "nx.DiGraph":
    """Grafo base reproducible para escenarios controlados."""
    return _build_minimal_material_graph()


@pytest.fixture()
def multi_category_graph() -> "nx.DiGraph":
    """Grafo con múltiples categorías de material."""
    return _build_multi_category_graph()


@pytest.fixture()
def fresh_generator() -> "MatterGenerator":
    """
    Instancia fresca de MatterGenerator para cada test.

    Garantiza aislamiento: ningún test puede contaminarse con el
    estado interno de una llamada anterior.
    """
    return MatterGenerator()


@pytest.fixture()
def low_entropy_metrics() -> Dict[str, float]:
    return _make_low_entropy_metrics()


@pytest.fixture()
def high_entropy_metrics() -> Dict[str, float]:
    return _make_high_entropy_metrics()


# ═════════════════════════════════════════════════════════════════════════════
# T2 — CONTRATO DE ESTRUCTURA DEL BOM
# ═════════════════════════════════════════════════════════════════════════════
# Corrección v3: las funciones auxiliares usan pytest.raises-style failures
# (pytest.fail) en lugar de assert bare, que puede silenciarse con -O.


def _assert_bom_structure(result_bom: Any, context: str = "") -> None:
    """
    Verifica que result_bom cumpla el contrato estructural mínimo.

    Usa pytest.fail en lugar de assert para que los fallos sean siempre
    visibles, incluso con optimización Python (-O flag).

    Args:
        result_bom: El objeto BOM retornado por materialize_project.
        context:    Descripción del escenario para mensajes de error ricos.
    """
    prefix = f"[{context}] " if context else ""

    if result_bom is None:
        pytest.fail(f"{prefix}materialize_project devolvió None")

    metadata = getattr(result_bom, "metadata", None)
    if not isinstance(metadata, dict):
        pytest.fail(
            f"{prefix}result_bom.metadata debe ser dict, "
            f"obtenido: {type(metadata).__name__}"
        )

    # ── material_distribution ─────────────────────────────────────────────
    material_distribution = metadata.get("material_distribution")
    if not isinstance(material_distribution, dict):
        pytest.fail(
            f"{prefix}metadata['material_distribution'] debe ser dict, "
            f"obtenido: {type(material_distribution).__name__}"
        )

    waste = material_distribution.get("waste")
    if not isinstance(waste, dict):
        pytest.fail(
            f"{prefix}metadata['material_distribution']['waste'] debe ser dict, "
            f"obtenido: {type(waste).__name__}"
        )

    waste_mean = waste.get("mean")
    if not isinstance(waste_mean, (int, float)):
        pytest.fail(
            f"{prefix}waste['mean'] debe ser numérico, "
            f"obtenido: {type(waste_mean).__name__}"
        )
    if not math.isfinite(waste_mean):
        pytest.fail(
            f"{prefix}waste['mean'] debe ser finito, obtenido: {waste_mean}"
        )
    if waste_mean < 0.0:
        pytest.fail(
            f"{prefix}waste['mean'] no puede ser negativo, obtenido: {waste_mean}"
        )

    # ── risk_analysis ─────────────────────────────────────────────────────
    risk_analysis = metadata.get("risk_analysis")
    if not isinstance(risk_analysis, dict):
        pytest.fail(
            f"{prefix}metadata['risk_analysis'] debe ser dict, "
            f"obtenido: {type(risk_analysis).__name__}"
        )

    flux_metrics = risk_analysis.get("flux_metrics")
    if not isinstance(flux_metrics, dict):
        pytest.fail(
            f"{prefix}metadata['risk_analysis']['flux_metrics'] debe ser dict, "
            f"obtenido: {type(flux_metrics).__name__}"
        )


def _extract_waste_mean(result_bom: Any, context: str = "") -> float:
    """
    Extrae waste['mean'] del BOM después de verificar su estructura.

    Returns:
        float: El valor de waste['mean'].
    """
    _assert_bom_structure(result_bom, context)
    return float(result_bom.metadata["material_distribution"]["waste"]["mean"])


def _extract_flux_metrics(result_bom: Any, context: str = "") -> Dict[str, float]:
    """
    Extrae flux_metrics del BOM después de verificar su estructura.

    Returns:
        Dict[str, float]: Las métricas de flujo propagadas al análisis de riesgo.
    """
    _assert_bom_structure(result_bom, context)
    return result_bom.metadata["risk_analysis"]["flux_metrics"]


def _materialize(
    graph: "nx.DiGraph",
    flux_metrics: Dict[str, float],
    generator: Optional["MatterGenerator"] = None,
) -> Any:
    """
    Ejecuta materialize_project con una instancia fresca por defecto.

    Args:
        graph:        Grafo de materialización (se usa directamente, sin copiar).
        flux_metrics: Métricas de flujo (se pasa copia para garantizar inmutabilidad).
        generator:    Instancia de MatterGenerator. Si None, se crea una nueva.

    Returns:
        El objeto BOM resultado de la materialización.
    """
    gen = generator if generator is not None else MatterGenerator()
    return gen.materialize_project(
        graph=graph,
        flux_metrics=dict(flux_metrics),  # Copia para proteger el original
    )


class TestBOMStructure:
    """
    T2: Verifica que el contrato estructural del BOM se cumple
    para diferentes combinaciones de grafo y métricas.
    """

    def test_bom_is_not_none_for_minimal_graph(
        self,
        minimal_graph: "nx.DiGraph",
        low_entropy_metrics: Dict[str, float],
    ) -> None:
        """materialize_project no debe retornar None con entradas válidas."""
        result = _materialize(minimal_graph, low_entropy_metrics)
        assert result is not None

    def test_bom_has_metadata_dict(
        self,
        minimal_graph: "nx.DiGraph",
        low_entropy_metrics: Dict[str, float],
    ) -> None:
        result = _materialize(minimal_graph, low_entropy_metrics)
        assert isinstance(getattr(result, "metadata", None), dict)

    def test_bom_has_material_distribution(
        self,
        minimal_graph: "nx.DiGraph",
        low_entropy_metrics: Dict[str, float],
    ) -> None:
        result = _materialize(minimal_graph, low_entropy_metrics)
        _assert_bom_structure(result, "material_distribution check")

    def test_waste_mean_is_finite_non_negative(
        self,
        minimal_graph: "nx.DiGraph",
        low_entropy_metrics: Dict[str, float],
    ) -> None:
        """Invariante fundamental: waste_mean ∈ [0, +∞) y es finito."""
        result = _materialize(minimal_graph, low_entropy_metrics)
        waste_mean = _extract_waste_mean(result)
        assert math.isfinite(waste_mean)
        assert waste_mean >= 0.0

    def test_flux_metrics_in_risk_analysis(
        self,
        minimal_graph: "nx.DiGraph",
        low_entropy_metrics: Dict[str, float],
    ) -> None:
        """Las métricas de flujo deben estar presentes en el análisis de riesgo."""
        result = _materialize(minimal_graph, low_entropy_metrics)
        flux = _extract_flux_metrics(result)
        assert isinstance(flux, dict)

    def test_bom_structure_with_multi_category_graph(
        self,
        multi_category_graph: "nx.DiGraph",
        high_entropy_metrics: Dict[str, float],
    ) -> None:
        """El contrato estructural debe mantenerse con múltiples categorías."""
        result = _materialize(multi_category_graph, high_entropy_metrics)
        _assert_bom_structure(result, "multi_category")

    def test_flux_metrics_contains_all_canonical_keys(
        self,
        minimal_graph: "nx.DiGraph",
        high_entropy_metrics: Dict[str, float],
    ) -> None:
        """
        Las métricas de flujo propagadas deben incluir todas las claves
        canónicas del input.
        """
        result = _materialize(minimal_graph, high_entropy_metrics)
        flux = _extract_flux_metrics(result, "canonical_keys")
        for key in _FLUX_METRIC_KEYS:
            assert key in flux, (
                f"Clave canónica '{key}' no propagada a flux_metrics. "
                f"Claves presentes: {list(flux.keys())}"
            )


# ═════════════════════════════════════════════════════════════════════════════
# T3 — PROPIEDADES DE FRONTERA
# ═════════════════════════════════════════════════════════════════════════════


class TestBoundaryConditions:
    """
    T3: Verifica el comportamiento en los extremos del dominio de entropía.

    Los casos frontera son los más propensos a revelar errores de implementación
    (división por cero, overflow, condiciones mal escritas).
    """

    def test_zero_entropy_produces_finite_waste(
        self, minimal_graph: "nx.DiGraph"
    ) -> None:
        """entropy_ratio=0.0 debe producir waste_mean finito y no negativo."""
        zero_metrics = _make_zero_entropy_metrics()
        result = _materialize(minimal_graph, zero_metrics)
        waste_mean = _extract_waste_mean(result, "zero_entropy")
        assert math.isfinite(waste_mean)
        assert waste_mean >= 0.0

    def test_max_entropy_produces_finite_waste(
        self, minimal_graph: "nx.DiGraph"
    ) -> None:
        """entropy_ratio=1.0 debe producir waste_mean finito (sin overflow)."""
        max_metrics = _make_max_entropy_metrics()
        result = _materialize(minimal_graph, max_metrics)
        waste_mean = _extract_waste_mean(result, "max_entropy")
        assert math.isfinite(waste_mean)
        assert waste_mean >= 0.0

    def test_zero_entropy_waste_less_than_max_entropy_waste(
        self, minimal_graph: "nx.DiGraph"
    ) -> None:
        """
        Monotonía en los extremos del dominio:
        waste(entropy=0) ≤ waste(entropy=1).

        Esta es la forma más fuerte de la propiedad monotónica.
        """
        result_zero = _materialize(minimal_graph, _make_zero_entropy_metrics())
        result_max = _materialize(minimal_graph, _make_max_entropy_metrics())

        waste_zero = _extract_waste_mean(result_zero, "boundary_zero")
        waste_max = _extract_waste_mean(result_max, "boundary_max")

        assert waste_zero <= waste_max, (
            f"Violación de monotonía en extremos: "
            f"waste(entropy=0)={waste_zero} > waste(entropy=1)={waste_max}"
        )

    def test_pyramid_stability_one_is_safe(
        self, minimal_graph: "nx.DiGraph"
    ) -> None:
        """pyramid_stability=1.0 (máxima estabilidad) no debe producir errores."""
        metrics = {
            "entropy_ratio": 0.5,
            "avg_saturation": 0.5,
            "pyramid_stability": 1.0,
        }
        result = _materialize(minimal_graph, metrics)
        _assert_bom_structure(result, "pyramid_stability_1.0")

    def test_pyramid_stability_zero_is_safe(
        self, minimal_graph: "nx.DiGraph"
    ) -> None:
        """pyramid_stability=0.0 (inestabilidad total) no debe propagar excepción."""
        metrics = {
            "entropy_ratio": 0.5,
            "avg_saturation": 0.5,
            "pyramid_stability": 0.0,
        }
        result = _materialize(minimal_graph, metrics)
        _assert_bom_structure(result, "pyramid_stability_0.0")

    @pytest.mark.parametrize(
        "bad_entropy,desc",
        [
            (-0.01, "entropy_ratio negativo"),
            (1.01, "entropy_ratio > 1"),
            (float("inf"), "entropy_ratio infinito"),
            (float("-inf"), "entropy_ratio -infinito"),
        ],
    )
    def test_out_of_domain_entropy_handled_safely(
        self,
        minimal_graph: "nx.DiGraph",
        bad_entropy: float,
        desc: str,
    ) -> None:
        """
        Entradas fuera del dominio [0, 1] no deben propagar excepción no documentada.

        El sistema puede: (a) levantar ValueError/TypeError controlado,
        o (b) producir un BOM válido con valor saturado en el extremo del dominio.
        No debe: propagar ZeroDivisionError, OverflowError, etc.
        """
        metrics = {
            "entropy_ratio": bad_entropy,
            "avg_saturation": 0.5,
            "pyramid_stability": 0.5,
        }
        allowed_exceptions = (ValueError, TypeError)
        try:
            result = _materialize(minimal_graph, metrics)
            # Si no falla: el resultado debe ser estructuralmente válido
            waste_mean = _extract_waste_mean(result, desc)
            assert math.isfinite(waste_mean), (
                f"[{desc}] waste_mean no es finito con entropy={bad_entropy}"
            )
        except allowed_exceptions:
            pass  # Fallo controlado: contrato de pre-condición cumplido
        except Exception as exc:
            pytest.fail(
                f"[{desc}] Excepción no controlada con entropy_ratio={bad_entropy}: "
                f"{type(exc).__name__}: {exc}"
            )

    def test_nan_entropy_handled_safely(
        self, minimal_graph: "nx.DiGraph"
    ) -> None:
        """NaN en entropy_ratio no debe propagar excepción no controlada."""
        metrics = {
            "entropy_ratio": float("nan"),
            "avg_saturation": 0.5,
            "pyramid_stability": 0.5,
        }
        allowed_exceptions = (ValueError, TypeError)
        try:
            result = _materialize(minimal_graph, metrics)
            waste_mean = _extract_waste_mean(result, "nan_entropy")
            # Si acepta NaN, el resultado no debe ser NaN
            assert math.isfinite(waste_mean), (
                f"waste_mean es NaN/inf cuando entropy_ratio=NaN"
            )
        except allowed_exceptions:
            pass
        except Exception as exc:
            pytest.fail(
                f"Excepción no controlada con entropy_ratio=NaN: "
                f"{type(exc).__name__}: {exc}"
            )


# ═════════════════════════════════════════════════════════════════════════════
# T4 — PROPIEDAD MONOTÓNICA PRINCIPAL
# ═════════════════════════════════════════════════════════════════════════════


class TestMonotonicProperty:
    """
    T4: Verifica la propiedad central del sistema:
    mayor entropía → mayor desperdicio esperado para materiales frágiles.

    Esta propiedad debe mantenerse con tolerancia mínima δ = 0.01 para
    distinguir efectos reales de ruido numérico.
    """

    def test_high_entropy_increases_waste_for_fragile(
        self,
        minimal_graph: "nx.DiGraph",
        low_entropy_metrics: Dict[str, float],
        high_entropy_metrics: Dict[str, float],
    ) -> None:
        """
        Propiedad monotónica principal:
        waste(high_entropy) > waste(low_entropy) + δ
        donde δ = _MONOTONICITY_MIN_DELTA = 0.01.

        Corrección v3: se exige diferencia significativa (no solo >),
        para distinguir efecto real de error de redondeo.
        """
        result_low = _materialize(minimal_graph, low_entropy_metrics)
        result_high = _materialize(minimal_graph, high_entropy_metrics)

        waste_low = _extract_waste_mean(result_low, "low_entropy")
        waste_high = _extract_waste_mean(result_high, "high_entropy")

        assert waste_high > waste_low + _MONOTONICITY_MIN_DELTA, (
            f"Incremento de entropía no produjo aumento significativo de desperdicio. "
            f"waste_low={waste_low:.6f}, waste_high={waste_high:.6f}, "
            f"diferencia={waste_high - waste_low:.6f} < δ={_MONOTONICITY_MIN_DELTA}"
        )

    def test_high_entropy_waste_exceeds_fragile_threshold(
        self,
        minimal_graph: "nx.DiGraph",
        high_entropy_metrics: Dict[str, float],
    ) -> None:
        """
        Para material_category='FRAGILE' con alta entropía, el desperdicio
        debe superar el umbral mínimo documentado del modelo.

        Ver _FRAGILE_HIGH_ENTROPY_MIN_WASTE para la justificación matemática.
        """
        result = _materialize(minimal_graph, high_entropy_metrics)
        waste_mean = _extract_waste_mean(result, "fragile_threshold")

        assert waste_mean > _FRAGILE_HIGH_ENTROPY_MIN_WASTE, (
            f"waste_mean={waste_mean:.6f} no supera el umbral mínimo "
            f"para FRAGILE+alta entropía: {_FRAGILE_HIGH_ENTROPY_MIN_WASTE}. "
            f"Verifique el modelo de desperdicio para material_category='FRAGILE'."
        )

    def test_monotonicity_across_three_entropy_levels(
        self,
        minimal_graph: "nx.DiGraph",
    ) -> None:
        """
        Monotonía en tres niveles: waste(low) ≤ waste(medium) ≤ waste(high).

        Verifica que la propiedad se cumple en más de dos puntos del dominio.
        """
        medium_metrics = {
            "entropy_ratio": 0.50,
            "avg_saturation": 0.50,
            "pyramid_stability": 0.70,
        }

        result_low = _materialize(minimal_graph, _make_low_entropy_metrics())
        result_med = _materialize(minimal_graph, medium_metrics)
        result_high = _materialize(minimal_graph, _make_high_entropy_metrics())

        waste_low = _extract_waste_mean(result_low, "three_levels_low")
        waste_med = _extract_waste_mean(result_med, "three_levels_medium")
        waste_high = _extract_waste_mean(result_high, "three_levels_high")

        assert waste_low <= waste_med, (
            f"Violación de monotonía low→medium: {waste_low:.6f} > {waste_med:.6f}"
        )
        assert waste_med <= waste_high, (
            f"Violación de monotonía medium→high: {waste_med:.6f} > {waste_high:.6f}"
        )

    def test_monotonicity_with_multi_category_graph(
        self,
        multi_category_graph: "nx.DiGraph",
        low_entropy_metrics: Dict[str, float],
        high_entropy_metrics: Dict[str, float],
    ) -> None:
        """
        La propiedad monotónica debe mantenerse también con múltiples categorías.
        """
        result_low = _materialize(multi_category_graph, low_entropy_metrics)
        result_high = _materialize(multi_category_graph, high_entropy_metrics)

        waste_low = _extract_waste_mean(result_low, "multi_cat_low")
        waste_high = _extract_waste_mean(result_high, "multi_cat_high")

        # Monotonía débil: al menos no decrece
        assert waste_high >= waste_low, (
            f"waste decreció con entropía alta en grafo multi-categoría: "
            f"low={waste_low:.6f}, high={waste_high:.6f}"
        )

    @pytest.mark.parametrize(
        "entropy_ratio",
        [0.0, 0.25, 0.50, 0.75, 1.0],
    )
    def test_waste_is_non_negative_for_all_entropy_levels(
        self,
        minimal_graph: "nx.DiGraph",
        entropy_ratio: float,
    ) -> None:
        """
        Invariante de no-negatividad para un barrido completo del dominio.

        waste_mean ≥ 0 para todo entropy_ratio ∈ [0, 1].
        """
        metrics = {
            "entropy_ratio": entropy_ratio,
            "avg_saturation": entropy_ratio,
            "pyramid_stability": 1.0 - entropy_ratio,
        }
        result = _materialize(minimal_graph, metrics)
        waste_mean = _extract_waste_mean(result, f"sweep_e={entropy_ratio}")
        assert waste_mean >= 0.0, (
            f"waste_mean negativo para entropy_ratio={entropy_ratio}: {waste_mean}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# T5 — PROPAGACIÓN DE MÉTRICAS DE FLUJO
# ═════════════════════════════════════════════════════════════════════════════


class TestFluxMetricsPropagation:
    """
    T5: Verifica que las métricas de flujo se propagan correctamente
    al análisis de riesgo del BOM.
    """

    def test_all_input_metrics_propagated(
        self,
        minimal_graph: "nx.DiGraph",
        high_entropy_metrics: Dict[str, float],
    ) -> None:
        """Cada clave del input debe aparecer en flux_metrics del resultado."""
        result = _materialize(minimal_graph, high_entropy_metrics)
        flux = _extract_flux_metrics(result, "propagation_keys")

        for key, expected in high_entropy_metrics.items():
            assert key in flux, (
                f"Métrica '{key}' no propagada a risk_analysis.flux_metrics. "
                f"Claves presentes: {sorted(flux.keys())}"
            )
            assert flux[key] == pytest.approx(expected, rel=1e-6), (
                f"Métrica '{key}' propagada con valor incorrecto: "
                f"esperado={expected}, obtenido={flux[key]}"
            )

    def test_low_entropy_metrics_propagated(
        self,
        minimal_graph: "nx.DiGraph",
        low_entropy_metrics: Dict[str, float],
    ) -> None:
        """La propagación funciona también para métricas de baja entropía."""
        result = _materialize(minimal_graph, low_entropy_metrics)
        flux = _extract_flux_metrics(result, "low_entropy_propagation")

        for key, expected in low_entropy_metrics.items():
            assert key in flux
            assert flux[key] == pytest.approx(expected, rel=1e-6)

    def test_flux_metrics_values_are_numeric(
        self,
        minimal_graph: "nx.DiGraph",
        high_entropy_metrics: Dict[str, float],
    ) -> None:
        """Todos los valores en flux_metrics deben ser numéricos y finitos."""
        result = _materialize(minimal_graph, high_entropy_metrics)
        flux = _extract_flux_metrics(result, "numeric_check")

        for key, value in flux.items():
            assert isinstance(value, (int, float)), (
                f"flux_metrics['{key}'] no es numérico: {type(value).__name__}"
            )
            assert math.isfinite(value), (
                f"flux_metrics['{key}'] no es finito: {value}"
            )

    def test_input_metrics_not_mutated_by_materialization(
        self,
        minimal_graph: "nx.DiGraph",
        high_entropy_metrics: Dict[str, float],
    ) -> None:
        """
        materialize_project no debe mutar el diccionario de métricas original.

        Corrección v3: se verifica inmutabilidad explícitamente.
        """
        original_metrics = dict(high_entropy_metrics)
        _materialize(minimal_graph, high_entropy_metrics)

        assert high_entropy_metrics == original_metrics, (
            "materialize_project mutó el diccionario de métricas de flujo original. "
            f"Antes: {original_metrics}, Después: {high_entropy_metrics}"
        )

    def test_flux_metrics_propagated_with_extra_keys(
        self,
        minimal_graph: "nx.DiGraph",
    ) -> None:
        """
        Métricas adicionales (no canónicas) deben ser propagadas o ignoradas
        sin causar excepción.
        """
        extended_metrics = {
            "entropy_ratio": 0.5,
            "avg_saturation": 0.5,
            "pyramid_stability": 0.7,
            "custom_metric_alpha": 0.123,
            "custom_metric_beta": 42.0,
        }
        try:
            result = _materialize(minimal_graph, extended_metrics)
            _assert_bom_structure(result, "extra_keys")
        except (ValueError, TypeError, KeyError):
            pass  # Rechazo controlado de claves no canónicas: aceptable
        except Exception as exc:
            pytest.fail(
                f"Excepción no controlada con métricas extra: "
                f"{type(exc).__name__}: {exc}"
            )


# ═════════════════════════════════════════════════════════════════════════════
# T6 — AISLAMIENTO ENTRE LLAMADAS
# ═════════════════════════════════════════════════════════════════════════════


class TestCallIsolation:
    """
    T6: Verifica que cada llamada a materialize_project sea independiente.

    Corrección v3: el test original no verificaba aislamiento; si MatterGenerator
    acumula estado interno entre llamadas, los resultados pueden ser incorrectos.
    """

    def test_two_fresh_generators_produce_same_result(
        self,
        minimal_graph: "nx.DiGraph",
        low_entropy_metrics: Dict[str, float],
    ) -> None:
        """
        Dos instancias frescas de MatterGenerator con el mismo input
        deben producir el mismo waste_mean (determinismo).
        """
        result1 = _materialize(_build_minimal_material_graph(), low_entropy_metrics)
        result2 = _materialize(_build_minimal_material_graph(), low_entropy_metrics)

        waste1 = _extract_waste_mean(result1, "isolation_gen1")
        waste2 = _extract_waste_mean(result2, "isolation_gen2")

        assert waste1 == pytest.approx(waste2, rel=1e-9), (
            f"Dos instancias frescas produjeron resultados diferentes: "
            f"gen1={waste1}, gen2={waste2}"
        )

    def test_sequential_calls_on_same_generator_are_consistent(
        self, fresh_generator: "MatterGenerator"
    ) -> None:
        """
        Dos llamadas consecutivas sobre la misma instancia con el mismo input
        deben producir el mismo resultado (sin acumulación de estado mutable).
        """
        metrics = _make_low_entropy_metrics()

        result1 = fresh_generator.materialize_project(
            graph=_build_minimal_material_graph(),
            flux_metrics=dict(metrics),
        )
        result2 = fresh_generator.materialize_project(
            graph=_build_minimal_material_graph(),
            flux_metrics=dict(metrics),
        )

        waste1 = _extract_waste_mean(result1, "seq_call_1")
        waste2 = _extract_waste_mean(result2, "seq_call_2")

        assert waste1 == pytest.approx(waste2, rel=1e-9), (
            f"Llamadas secuenciales sobre misma instancia produjeron resultados "
            f"diferentes: call1={waste1}, call2={waste2}. "
            f"MatterGenerator puede tener estado mutable acumulable."
        )

    def test_graph_not_mutated_by_materialization(
        self,
        minimal_graph: "nx.DiGraph",
        low_entropy_metrics: Dict[str, float],
    ) -> None:
        """
        materialize_project no debe modificar el grafo de entrada.
        """
        nodes_before = set(minimal_graph.nodes())
        edges_before = set(minimal_graph.edges())

        _materialize(minimal_graph, low_entropy_metrics)

        assert set(minimal_graph.nodes()) == nodes_before, (
            "materialize_project añadió o eliminó nodos del grafo original"
        )
        assert set(minimal_graph.edges()) == edges_before, (
            "materialize_project modificó las aristas del grafo original"
        )

    def test_different_entropy_calls_on_same_instance_independent(
        self, fresh_generator: "MatterGenerator"
    ) -> None:
        """
        Llamadas con baja y alta entropía sobre la misma instancia
        no deben contaminarse entre sí.
        """
        low_metrics = _make_low_entropy_metrics()
        high_metrics = _make_high_entropy_metrics()

        result_low = fresh_generator.materialize_project(
            graph=_build_minimal_material_graph(),
            flux_metrics=dict(low_metrics),
        )
        result_high = fresh_generator.materialize_project(
            graph=_build_minimal_material_graph(),
            flux_metrics=dict(high_metrics),
        )

        waste_low = _extract_waste_mean(result_low, "mixed_low")
        waste_high = _extract_waste_mean(result_high, "mixed_high")

        # Verificar que los resultados corresponden a sus respectivas entropías
        # (no están contaminados por la llamada anterior)
        flux_low = _extract_flux_metrics(result_low, "mixed_flux_low")
        flux_high = _extract_flux_metrics(result_high, "mixed_flux_high")

        assert flux_low.get("entropy_ratio", None) == pytest.approx(
            low_metrics["entropy_ratio"], rel=1e-9
        ), "flux_metrics del resultado 'low' contiene valor de 'high'"

        assert flux_high.get("entropy_ratio", None) == pytest.approx(
            high_metrics["entropy_ratio"], rel=1e-9
        ), "flux_metrics del resultado 'high' contiene valor de 'low'"


# ═════════════════════════════════════════════════════════════════════════════
# T7 — CATEGORÍAS DE MATERIAL
# ═════════════════════════════════════════════════════════════════════════════


class TestMaterialCategories:
    """
    T7: Verifica el comportamiento diferenciado por categoría de material.

    Corrección v3: la suite original no probaba ninguna categoría distinta a FRAGILE.
    """

    def test_fragile_higher_waste_than_standard_at_high_entropy(self) -> None:
        """
        A igualdad de entropía, FRAGILE debe generar mayor desperdicio que STANDARD.

        Justificación del modelo: los materiales frágiles tienen mayor factor
        de amplificación (α_FRAGILE > α_STANDARD) por su sensibilidad al
        desorden logístico.
        """
        high_metrics = _make_high_entropy_metrics()

        result_fragile = _materialize(
            _build_minimal_material_graph(), high_metrics
        )
        result_standard = _materialize(
            _build_zero_cost_graph(), high_metrics  # STANDARD con cost=0
        )

        # Si ambos son válidos estructuralmente, verificar la relación
        waste_fragile = _extract_waste_mean(result_fragile, "fragile_vs_standard")
        waste_standard = _extract_waste_mean(result_standard, "standard_reference")

        # FRAGILE debe producir igual o mayor desperdicio que STANDARD
        assert waste_fragile >= waste_standard, (
            f"FRAGILE produjo menos desperdicio que STANDARD bajo alta entropía: "
            f"fragile={waste_fragile:.6f}, standard={waste_standard:.6f}"
        )

    def test_standard_category_produces_valid_bom(self) -> None:
        """material_category='STANDARD' debe producir BOM estructuralmente válido."""
        graph = nx.DiGraph()
        graph.add_node("ROOT", type="APU")
        graph.add_node(
            "MAT1",
            type="INSUMO",
            unit_cost=100.0,
            unit="M2",
            material_category="STANDARD",
        )
        graph.add_edge("ROOT", "MAT1", quantity=10.0)

        result = _materialize(graph, _make_high_entropy_metrics())
        _assert_bom_structure(result, "STANDARD_category")

    def test_perishable_category_produces_valid_bom(self) -> None:
        """material_category='PERISHABLE' debe producir BOM estructuralmente válido."""
        graph = nx.DiGraph()
        graph.add_node("ROOT", type="APU")
        graph.add_node(
            "MAT1",
            type="INSUMO",
            unit_cost=500.0,
            unit="KG",
            material_category="PERISHABLE",
        )
        graph.add_edge("ROOT", "MAT1", quantity=3.0)

        result = _materialize(graph, _make_high_entropy_metrics())
        _assert_bom_structure(result, "PERISHABLE_category")

    def test_unknown_category_handled_gracefully(self) -> None:
        """
        Una categoría desconocida no debe propagar excepción no controlada.
        El sistema puede usar un comportamiento por defecto.
        """
        graph = _build_unknown_category_graph()
        try:
            result = _materialize(graph, _make_low_entropy_metrics())
            _assert_bom_structure(result, "unknown_category")
        except (ValueError, KeyError):
            pass  # Rechazo controlado: aceptable
        except Exception as exc:
            pytest.fail(
                f"Excepción no controlada con categoría desconocida: "
                f"{type(exc).__name__}: {exc}"
            )

    def test_multi_category_graph_waste_non_negative(
        self,
        multi_category_graph: "nx.DiGraph",
        high_entropy_metrics: Dict[str, float],
    ) -> None:
        """El waste agregado de múltiples categorías debe ser no negativo."""
        result = _materialize(multi_category_graph, high_entropy_metrics)
        waste_mean = _extract_waste_mean(result, "multi_category_aggregate")
        assert waste_mean >= 0.0


# ═════════════════════════════════════════════════════════════════════════════
# T8 — ROBUSTEZ: GRAFOS DEGENERADOS E INPUTS INVÁLIDOS
# ═════════════════════════════════════════════════════════════════════════════


class TestRobustness:
    """
    T8: Verifica que el sistema maneja entradas degeneradas sin propagar
    excepciones no documentadas.

    Corrección v3: la suite original no probaba ningún caso degenerado.
    """

    def test_empty_graph_handled_safely(
        self,
        low_entropy_metrics: Dict[str, float],
    ) -> None:
        """Grafo vacío (sin nodos) no debe propagar excepción inesperada."""
        graph = _build_empty_graph()
        allowed = (ValueError, KeyError, RuntimeError)
        try:
            result = _materialize(graph, low_entropy_metrics)
            # Si acepta grafo vacío: el BOM puede estar vacío pero no debe ser None
            assert result is not None
        except allowed:
            pass
        except Exception as exc:
            pytest.fail(
                f"Excepción no controlada con grafo vacío: "
                f"{type(exc).__name__}: {exc}"
            )

    def test_root_only_graph_handled_safely(
        self,
        low_entropy_metrics: Dict[str, float],
    ) -> None:
        """Grafo con solo el nodo ROOT (sin materiales) no debe fallar."""
        graph = _build_root_only_graph()
        allowed = (ValueError, KeyError, RuntimeError)
        try:
            result = _materialize(graph, low_entropy_metrics)
            assert result is not None
        except allowed:
            pass
        except Exception as exc:
            pytest.fail(
                f"Excepción no controlada con grafo solo-raíz: "
                f"{type(exc).__name__}: {exc}"
            )

    def test_zero_quantity_edge_handled_safely(
        self,
        low_entropy_metrics: Dict[str, float],
    ) -> None:
        """quantity=0 en la arista no debe producir waste negativo ni excepción."""
        graph = _build_zero_quantity_graph()
        try:
            result = _materialize(graph, low_entropy_metrics)
            waste_mean = _extract_waste_mean(result, "zero_quantity")
            assert waste_mean >= 0.0
            assert math.isfinite(waste_mean)
        except (ValueError, ZeroDivisionError):
            pass
        except Exception as exc:
            pytest.fail(
                f"Excepción inesperada con quantity=0: "
                f"{type(exc).__name__}: {exc}"
            )

    def test_zero_unit_cost_handled_safely(
        self,
        low_entropy_metrics: Dict[str, float],
    ) -> None:
        """unit_cost=0 no debe producir división por cero ni waste negativo."""
        graph = _build_zero_cost_graph()
        try:
            result = _materialize(graph, low_entropy_metrics)
            waste_mean = _extract_waste_mean(result, "zero_cost")
            assert waste_mean >= 0.0
            assert math.isfinite(waste_mean)
        except (ValueError, ZeroDivisionError):
            pass
        except Exception as exc:
            pytest.fail(
                f"Excepción inesperada con unit_cost=0: "
                f"{type(exc).__name__}: {exc}"
            )

    def test_empty_flux_metrics_handled_safely(
        self,
        minimal_graph: "nx.DiGraph",
    ) -> None:
        """Diccionario de métricas vacío no debe propagar excepción inesperada."""
        allowed = (ValueError, KeyError, TypeError)
        try:
            result = _materialize(minimal_graph, {})
            _assert_bom_structure(result, "empty_metrics")
        except allowed:
            pass
        except Exception as exc:
            pytest.fail(
                f"Excepción no controlada con métricas vacías: "
                f"{type(exc).__name__}: {exc}"
            )

    def test_missing_canonical_metric_key_handled_safely(
        self,
        minimal_graph: "nx.DiGraph",
    ) -> None:
        """Falta de una clave canónica en métricas no debe propagar excepción inesperada."""
        # Solo entropy_ratio, falta avg_saturation y pyramid_stability
        partial_metrics = {"entropy_ratio": 0.5}
        allowed = (ValueError, KeyError, TypeError)
        try:
            result = _materialize(minimal_graph, partial_metrics)
            _assert_bom_structure(result, "partial_metrics")
        except allowed:
            pass
        except Exception as exc:
            pytest.fail(
                f"Excepción no controlada con métricas parciales: "
                f"{type(exc).__name__}: {exc}"
            )

    def test_very_large_quantity_handled_safely(
        self,
        low_entropy_metrics: Dict[str, float],
    ) -> None:
        """Quantities muy grandes no deben producir overflow."""
        graph = nx.DiGraph()
        graph.add_node("ROOT", type="APU")
        graph.add_node(
            "MAT1",
            type="INSUMO",
            unit_cost=1e6,
            unit="UND",
            material_category="FRAGILE",
        )
        graph.add_edge("ROOT", "MAT1", quantity=1e9)

        try:
            result = _materialize(graph, low_entropy_metrics)
            waste_mean = _extract_waste_mean(result, "large_quantity")
            assert math.isfinite(waste_mean)
            assert waste_mean >= 0.0
        except (ValueError, OverflowError):
            pass
        except Exception as exc:
            pytest.fail(
                f"Excepción inesperada con quantity=1e9: "
                f"{type(exc).__name__}: {exc}"
            )


# ═════════════════════════════════════════════════════════════════════════════
# T9 — INVARIANTES ALGEBRAICOS GLOBALES
# ═════════════════════════════════════════════════════════════════════════════


class TestAlgebraicInvariants:
    """
    T9: Propiedades que deben mantenerse para cualquier entrada válida
    del sistema. Constituyen el contrato algebraico observable de MatterGenerator.
    """

    def test_waste_non_negativity_is_global_invariant(
        self, minimal_graph: "nx.DiGraph"
    ) -> None:
        """
        Invariante global: waste_mean ≥ 0 para todo entropy_ratio ∈ [0, 1].

        Verificación por muestreo uniforme del dominio (10 puntos).
        """
        sample_points = [i / 10.0 for i in range(11)]  # 0.0, 0.1, ..., 1.0
        for entropy in sample_points:
            metrics = {
                "entropy_ratio": entropy,
                "avg_saturation": entropy * 0.9,
                "pyramid_stability": 1.0 - entropy,
            }
            result = _materialize(_build_minimal_material_graph(), metrics)
            waste = _extract_waste_mean(result, f"global_non_neg_e={entropy:.1f}")
            assert waste >= 0.0, (
                f"Violación de no-negatividad global en entropy={entropy}: "
                f"waste={waste}"
            )

    def test_waste_is_finite_for_all_valid_inputs(
        self, minimal_graph: "nx.DiGraph"
    ) -> None:
        """
        Invariante global: waste_mean es finito para todo input válido.

        Ninguna combinación de métricas dentro del dominio debe producir
        NaN o infinito.
        """
        valid_scenarios = [
            _make_zero_entropy_metrics(),
            _make_low_entropy_metrics(),
            _make_high_entropy_metrics(),
            _make_max_entropy_metrics(),
            {"entropy_ratio": 0.5, "avg_saturation": 0.5, "pyramid_stability": 0.5},
        ]
        for scenario in valid_scenarios:
            result = _materialize(
                _build_minimal_material_graph(), scenario
            )
            waste = _extract_waste_mean(result, f"finite_check_e={scenario['entropy_ratio']}")
            assert math.isfinite(waste), (
                f"waste no es finito para scenario={scenario}: waste={waste}"
            )

    def test_materialization_is_deterministic(
        self, minimal_graph: "nx.DiGraph"
    ) -> None:
        """
        Invariante de determinismo: el mismo input siempre produce el mismo output.

        Se verifica con 3 repeticiones independientes.
        """
        metrics = _make_high_entropy_metrics()
        wastes = []

        for i in range(3):
            result = _materialize(_build_minimal_material_graph(), dict(metrics))
            waste = _extract_waste_mean(result, f"determinism_rep{i}")
            wastes.append(waste)

        assert wastes[0] == pytest.approx(wastes[1], rel=1e-9), (
            f"Resultado no determinista: rep0={wastes[0]}, rep1={wastes[1]}"
        )
        assert wastes[1] == pytest.approx(wastes[2], rel=1e-9), (
            f"Resultado no determinista: rep1={wastes[1]}, rep2={wastes[2]}"
        )

    def test_waste_increases_monotonically_with_entropy(
        self, minimal_graph: "nx.DiGraph"
    ) -> None:
        """
        Monotonía global estricta: ∀ e1 < e2 ⟹ waste(e1) ≤ waste(e2).

        Se verifica sobre una secuencia ordenada de 5 niveles de entropía.
        """
        entropy_levels = [0.0, 0.25, 0.50, 0.75, 1.0]
        wastes: List[float] = []

        for entropy in entropy_levels:
            metrics = {
                "entropy_ratio": entropy,
                "avg_saturation": entropy * 0.9,
                "pyramid_stability": 1.0 - entropy,
            }
            result = _materialize(_build_minimal_material_graph(), metrics)
            waste = _extract_waste_mean(result, f"monotonicity_e={entropy}")
            wastes.append(waste)

        for i in range(len(wastes) - 1):
            assert wastes[i] <= wastes[i + 1], (
                f"Violación de monotonía global entre entropy_levels "
                f"[{entropy_levels[i]}, {entropy_levels[i+1]}]: "
                f"waste[{i}]={wastes[i]:.6f} > waste[{i+1}]={wastes[i+1]:.6f}"
            )

    def test_flux_metrics_values_always_in_valid_range_after_propagation(
        self, minimal_graph: "nx.DiGraph"
    ) -> None:
        """
        Las métricas de flujo propagadas deben preservar sus rangos originales.

        Si la implementación transforma las métricas, los valores deben
        permanecer en [0, 1] para entropy_ratio, avg_saturation, pyramid_stability.
        """
        metrics = _make_high_entropy_metrics()
        result = _materialize(minimal_graph, dict(metrics))
        flux = _extract_flux_metrics(result, "range_preservation")

        for key in _FLUX_METRIC_KEYS:
            if key in flux:
                value = flux[key]
                assert 0.0 <= value <= 1.0, (
                    f"flux_metrics['{key}']={value} fuera del rango [0, 1] "
                    f"después de propagación"
                )

    def test_bom_structure_invariant_holds_across_all_scenarios(self) -> None:
        """
        El contrato estructural del BOM se mantiene para todos los escenarios
        de prueba conocidos.
        """
        scenarios = [
            (_build_minimal_material_graph(), _make_low_entropy_metrics()),
            (_build_minimal_material_graph(), _make_high_entropy_metrics()),
            (_build_minimal_material_graph(), _make_zero_entropy_metrics()),
            (_build_minimal_material_graph(), _make_max_entropy_metrics()),
            (_build_multi_category_graph(), _make_high_entropy_metrics()),
        ]

        for i, (graph, metrics) in enumerate(scenarios):
            result = _materialize(graph, metrics)
            _assert_bom_structure(result, f"scenario_{i}")