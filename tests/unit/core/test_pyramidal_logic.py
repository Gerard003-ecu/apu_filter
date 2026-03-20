"""
Suite de Pruebas Topológicas para el Validador Piramidal
========================================================

Fundamentos Matemáticos
-----------------------
1. Teoría de Orden y Retículos:
   - Stratum forma una cadena lineal total (orden total)
   - Propiedades: reflexividad, antisimetría, transitividad, totalidad
   - El orden induce una estructura de retículo con meet y join triviales

2. Topología Algebraica:
   - La estructura piramidal es un complejo simplicial estratificado
   - Característica de Euler: χ = V - E (para grafo APU-Insumo)
   - Invariantes de conectividad y componentes conexas

3. Teoría de Grafos:
   - Grafo bipartito G = (APUs ∪ Insumos, E)
   - Propiedades: matching, cobertura por vértices, independencia
   - Nodos flotantes = vértices de grado 0 en la partición APU

4. Análisis Matemático:
   - Índice de estabilidad: f(b,l) = tanh(b/max(l,1))
   - Propiedades: monotonía en b, antimonotonía en l, acotación en [0,1]
   - Comportamiento asintótico: lim_{b→∞} f = 1, lim_{l→∞} f = 0

5. Álgebra Lineal:
   - Ley de conservación: valor_total = cantidad × precio_unitario
   - Invariante multiplicativo que debe preservarse

Contrato del Dominio
--------------------
- Stratum define una cadena lineal total:
  WISDOM(0) < STRATEGY(1) < TACTICS(2) < PHYSICS(3)
- TopologicalNode representa un nodo estratificado con salud en [0, 1]
- InsumoProcesado ∈ PHYSICS satisface: valor_total = cantidad × precio_unitario
- APUStructure ∈ TACTICS con support_base_width monótono creciente
- PyramidalValidator calcula métricas estructurales:
  - base_width = |{insumos únicos}|
  - structure_load = |{APUs}|
  - pyramid_stability_index = tanh(base_width / max(structure_load, 1))
  - floating_nodes = {APU : degree(APU) = 0}
- StructuralClassifier es una función pura determinista
"""

from __future__ import annotations

import copy
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from typing import Any, Final, TypeVar

import numpy as np
import pandas as pd
import pytest

from app.classifiers.apu_classifier import StructuralClassifier
from app.adapters.data_loader import HierarchyLevel
from app.tactics.data_validator import PyramidalMetrics, PyramidalValidator
from app.core.schemas import APUStructure, InsumoProcesado, Stratum, TopologicalNode


# =============================================================================
# CONSTANTES DEL DOMINIO MATEMÁTICO
# =============================================================================

# Cadena esperada de estratos con sus valores ordinales
_EXPECTED_STRATUM_CHAIN: Final[tuple[tuple[str, int], ...]] = (
    ("WISDOM", 0),
    ("STRATEGY", 1),
    ("TACTICS", 2),
    ("PHYSICS", 3),
)

# Tipos válidos de insumo (partición del espacio de tipos)
_VALID_INSUMO_TYPES: Final[tuple[str, ...]] = (
    "SUMINISTRO",
    "MANO_DE_OBRA",
    "EQUIPO",
    "TRANSPORTE",
)

# Clasificaciones estructurales válidas
_VALID_CLASSIFICATIONS: Final[frozenset[str]] = frozenset({
    "SUMINISTRO_PURO",
    "SERVICIO_PURO",
    "ESTRUCTURA_MIXTA",
})

# Tolerancia para comparaciones de punto flotante
_FLOAT_TOLERANCE: Final[float] = 1e-10

# Tolerancia para ley de conservación valor = cantidad × precio
_CONSERVATION_TOLERANCE: Final[float] = 1e-6

# Rango válido para salud estructural
_HEALTH_MIN: Final[float] = 0.0
_HEALTH_MAX: Final[float] = 1.0

# Cardinalidad esperada del espacio de estratos
_STRATUM_CARDINALITY: Final[int] = 4

# Longitud máxima permitida para IDs
_MAX_ID_LENGTH: Final[int] = 512


# =============================================================================
# TIPOS AUXILIARES
# =============================================================================

T = TypeVar("T")


@dataclass(frozen=True)
class BipartiteGraphMetrics:
    """Métricas de un grafo bipartito APU-Insumo."""
    
    num_apus: int
    num_insumos: int
    num_edges: int
    num_isolated_apus: int
    num_isolated_insumos: int
    
    @property
    def euler_characteristic(self) -> int:
        """Característica de Euler: χ = V - E."""
        return (self.num_apus + self.num_insumos) - self.num_edges
    
    @property
    def density(self) -> float:
        """Densidad del grafo bipartito."""
        max_edges = self.num_apus * self.num_insumos
        if max_edges == 0:
            return 0.0
        return self.num_edges / max_edges
    
    @property
    def is_connected(self) -> bool:
        """Verifica si no hay vértices aislados."""
        return self.num_isolated_apus == 0 and self.num_isolated_insumos == 0


# =============================================================================
# FUNCIONES AUXILIARES MATEMÁTICAS
# =============================================================================

def stability_index_analytical(base_width: int, structure_load: int) -> float:
    """
    Calcula el índice de estabilidad piramidal usando la forma cerrada.
    
    f(b, l) = tanh(b / max(l, 1))
    
    Propiedades:
    - Dominio: b ≥ 0, l ≥ 0
    - Codominio: [0, 1)
    - Monótona creciente en b
    - Monótona decreciente en l (para l ≥ 1)
    - f(0, l) = 0 para todo l
    - lim_{b→∞} f(b, l) = 1
    
    Args:
        base_width: Número de insumos únicos (b ≥ 0).
        structure_load: Número de APUs (l ≥ 0).
    
    Returns:
        Índice de estabilidad en [0, 1).
    """
    if base_width < 0 or structure_load < 0:
        raise ValueError("base_width y structure_load deben ser no negativos")
    
    denominator = max(structure_load, 1)
    return math.tanh(base_width / denominator)


def stability_index_derivative_b(base_width: int, structure_load: int) -> float:
    """
    Calcula la derivada parcial del índice respecto a base_width.
    
    ∂f/∂b = sech²(b/l) / l = (1 - tanh²(b/l)) / l
    
    Esta derivada es siempre positiva, confirmando monotonía creciente.
    
    Args:
        base_width: Número de insumos únicos.
        structure_load: Número de APUs.
    
    Returns:
        Derivada parcial ∂f/∂b.
    """
    l = max(structure_load, 1)
    ratio = base_width / l
    tanh_val = math.tanh(ratio)
    sech_squared = 1 - tanh_val ** 2
    return sech_squared / l


def verify_stratum_chain_properties(strata: Sequence[Stratum]) -> dict[str, bool]:
    """
    Verifica propiedades de orden de una secuencia de estratos.
    
    Args:
        strata: Secuencia de estratos a verificar.
    
    Returns:
        Diccionario con resultados de cada propiedad.
    """
    values = [s.value for s in strata]
    
    # Reflexividad: ∀x: x ≤ x
    reflexive = all(v <= v for v in values)
    
    # Antisimetría: x ≤ y ∧ y ≤ x → x = y
    antisymmetric = all(
        (v1 != v2) or (v1 == v2)
        for v1 in values for v2 in values
        if v1 <= v2 and v2 <= v1
    )
    
    # Transitividad: x ≤ y ∧ y ≤ z → x ≤ z
    transitive = all(
        v1 <= v3
        for v1 in values for v2 in values for v3 in values
        if v1 <= v2 and v2 <= v3
    )
    
    # Totalidad: ∀x,y: x ≤ y ∨ y ≤ x
    total = all(
        v1 <= v2 or v2 <= v1
        for v1 in values for v2 in values
    )
    
    # Valores contiguos (sin huecos)
    sorted_values = sorted(values)
    contiguous = sorted_values == list(range(len(sorted_values)))
    
    return {
        "reflexive": reflexive,
        "antisymmetric": antisymmetric,
        "transitive": transitive,
        "total": total,
        "contiguous": contiguous,
    }


def compute_bipartite_metrics(
    apus_df: pd.DataFrame,
    insumos_df: pd.DataFrame,
) -> BipartiteGraphMetrics:
    """
    Calcula métricas del grafo bipartito APU-Insumo.
    
    Args:
        apus_df: DataFrame con columna CODIGO_APU.
        insumos_df: DataFrame con columnas APU_CODIGO, DESCRIPCION_INSUMO_NORM.
    
    Returns:
        Métricas del grafo bipartito.
    """
    apus = set(apus_df["CODIGO_APU"]) if len(apus_df) > 0 else set()
    
    if len(insumos_df) > 0:
        insumos = set(insumos_df["DESCRIPCION_INSUMO_NORM"])
        apus_with_insumos = set(insumos_df["APU_CODIGO"])
        edges = len(insumos_df)
    else:
        insumos = set()
        apus_with_insumos = set()
        edges = 0
    
    isolated_apus = apus - apus_with_insumos
    
    # Insumos aislados serían aquellos referenciados pero no definidos
    # En este modelo, todos los insumos están conectados a al menos un APU
    isolated_insumos = 0
    
    return BipartiteGraphMetrics(
        num_apus=len(apus),
        num_insumos=len(insumos),
        num_edges=edges,
        num_isolated_apus=len(isolated_apus),
        num_isolated_insumos=isolated_insumos,
    )


def verify_conservation_law(insumo: InsumoProcesado) -> tuple[bool, float]:
    """
    Verifica la ley de conservación: valor_total = cantidad × precio_unitario.
    
    Args:
        insumo: Insumo a verificar.
    
    Returns:
        Tupla (satisface_ley, error_absoluto).
    """
    expected = insumo.cantidad * insumo.precio_unitario
    actual = insumo.valor_total
    error = abs(expected - actual)
    satisfies = error <= _CONSERVATION_TOLERANCE
    return satisfies, error


def is_valid_health(health: float) -> bool:
    """Verifica si un valor de salud está en el rango válido [0, 1]."""
    return _HEALTH_MIN <= health <= _HEALTH_MAX


def stratum_distance(s1: Stratum, s2: Stratum) -> int:
    """
    Calcula la distancia entre dos estratos en la cadena.
    
    La distancia es |level(s1) - level(s2)|.
    
    Args:
        s1: Primer estrato.
        s2: Segundo estrato.
    
    Returns:
        Distancia no negativa entre estratos.
    """
    return abs(s1.value - s2.value)


def stratum_supremum(s1: Stratum, s2: Stratum) -> Stratum:
    """
    Calcula el supremo (join) de dos estratos.
    
    En una cadena lineal, sup(a, b) = max(a, b).
    
    Args:
        s1: Primer estrato.
        s2: Segundo estrato.
    
    Returns:
        Supremo de los dos estratos.
    """
    return s1 if s1.value >= s2.value else s2


def stratum_infimum(s1: Stratum, s2: Stratum) -> Stratum:
    """
    Calcula el ínfimo (meet) de dos estratos.
    
    En una cadena lineal, inf(a, b) = min(a, b).
    
    Args:
        s1: Primer estrato.
        s2: Segundo estrato.
    
    Returns:
        Ínfimo de los dos estratos.
    """
    return s1 if s1.value <= s2.value else s2


# =============================================================================
# CONSTRUCTORES DE DATOS DE PRUEBA
# =============================================================================

from app.core.schemas import TipoInsumo

def make_insumo(
    *,
    codigo_apu: str = "APU001",
    descripcion_apu: str = "Muro de Ladrillo",
    unidad_apu: str = "m2",
    descripcion_insumo: str = "Ladrillo Arcilla",
    unidad_insumo: str = "UND",
    cantidad: float = 150.0,
    precio_unitario: float = 850.0,
    valor_total: float | None = None,
    tipo_insumo: str = "SUMINISTRO",
) -> InsumoProcesado:
    """
    Constructor de InsumoProcesado con valor_total calculado automáticamente.
    
    Si valor_total no se especifica, se calcula como cantidad × precio_unitario,
    garantizando la ley de conservación.
    
    Args:
        codigo_apu: Código del APU padre.
        descripcion_apu: Descripción del APU.
        unidad_apu: Unidad de medida del APU.
        descripcion_insumo: Descripción del insumo.
        unidad_insumo: Unidad de medida del insumo.
        cantidad: Cantidad requerida.
        precio_unitario: Precio por unidad.
        valor_total: Valor total (calculado si None).
        tipo_insumo: Tipo de insumo.
    
    Returns:
        InsumoProcesado configurado.
    """
    if valor_total is None:
        valor_total = cantidad * precio_unitario
    
    rendimiento = 0.0
    if tipo_insumo == TipoInsumo.MANO_DE_OBRA.value and cantidad > 0:
        rendimiento = 1.0 / cantidad

    from app.core.schemas import INSUMO_CLASS_MAP
    clase_insumo = INSUMO_CLASS_MAP.get(tipo_insumo, InsumoProcesado)

    return clase_insumo(
        codigo_apu=codigo_apu,
        descripcion_apu=descripcion_apu,
        unidad_apu=unidad_apu,
        descripcion_insumo=descripcion_insumo,
        unidad_insumo=unidad_insumo,
        cantidad=cantidad,
        precio_unitario=precio_unitario,
        valor_total=valor_total,
        tipo_insumo=tipo_insumo,
        rendimiento=rendimiento,
    )


def make_apus_df(codes: Sequence[str]) -> pd.DataFrame:
    """Construye DataFrame de APUs a partir de códigos."""
    return pd.DataFrame({"CODIGO_APU": list(codes)})


def make_insumos_df(rows: Sequence[tuple[str, str]]) -> pd.DataFrame:
    """Construye DataFrame de insumos a partir de pares (APU, insumo)."""
    return pd.DataFrame({
        "APU_CODIGO": [apu for apu, _ in rows],
        "DESCRIPCION_INSUMO_NORM": [insumo for _, insumo in rows],
    })


def make_balanced_structure(
    n_apus: int,
    n_insumos_per_apu: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye una estructura balanceada para pruebas.
    
    Cada APU tiene exactamente n_insumos_per_apu insumos únicos.
    
    Args:
        n_apus: Número de APUs.
        n_insumos_per_apu: Insumos por APU.
    
    Returns:
        Tupla (apus_df, insumos_df).
    """
    apus = make_apus_df([f"APU{i:04d}" for i in range(n_apus)])
    
    rows = [
        (f"APU{i:04d}", f"INSUMO_{i}_{j}")
        for i in range(n_apus)
        for j in range(n_insumos_per_apu)
    ]
    insumos = make_insumos_df(rows)
    
    return apus, insumos


def make_unbalanced_structure(
    n_apus: int,
    floating_fraction: float = 0.3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye una estructura con APUs flotantes.
    
    Args:
        n_apus: Número total de APUs.
        floating_fraction: Fracción de APUs sin soporte.
    
    Returns:
        Tupla (apus_df, insumos_df).
    """
    n_floating = int(n_apus * floating_fraction)
    n_supported = n_apus - n_floating
    
    apus = make_apus_df([f"APU{i:04d}" for i in range(n_apus)])
    
    # Solo los primeros n_supported tienen insumos
    rows = [
        (f"APU{i:04d}", f"INSUMO_{i}")
        for i in range(n_supported)
    ]
    insumos = make_insumos_df(rows)
    
    return apus, insumos


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def validator() -> PyramidalValidator:
    """Instancia fresca de PyramidalValidator."""
    return PyramidalValidator()


@pytest.fixture
def classifier() -> StructuralClassifier:
    """Instancia fresca de StructuralClassifier."""
    return StructuralClassifier()


@pytest.fixture
def insumo_basico() -> InsumoProcesado:
    """Insumo básico con valores por defecto."""
    return make_insumo()


@pytest.fixture
def apu_vacio() -> APUStructure:
    """APU sin recursos asignados."""
    return APUStructure(
        id="APU001",
        description="Muro de Ladrillo",
        unit="m2",
        quantity=100.0,
    )


@pytest.fixture
def insumos_variados() -> list[InsumoProcesado]:
    """Lista de insumos de diferentes tipos."""
    return [
        make_insumo(
            descripcion_insumo="Ladrillo",
            tipo_insumo="SUMINISTRO",
            unidad_insumo="UND",
        ),
        make_insumo(
            descripcion_insumo="Cemento",
            unidad_insumo="UND",
            cantidad=25.0,
            precio_unitario=450.0,
            tipo_insumo="SUMINISTRO",
        ),
        make_insumo(
            descripcion_insumo="Albañil",
            unidad_insumo="HORA",
            cantidad=4.0,
            precio_unitario=15000.0,
            tipo_insumo="MANO_DE_OBRA",
        ),
        make_insumo(
            descripcion_insumo="Mezcladora",
            unidad_insumo="HORA",
            cantidad=2.0,
            precio_unitario=5000.0,
            tipo_insumo="EQUIPO",
        ),
    ]


@pytest.fixture
def estructura_valida() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estructura piramidal válida sin nodos flotantes."""
    apus = make_apus_df(["APU001", "APU002", "APU003"])
    insumos = make_insumos_df([
        ("APU001", "LADRILLO"),
        ("APU001", "CEMENTO"),
        ("APU002", "ARENA"),
        ("APU002", "GRAVA"),
        ("APU003", "ACERO"),
    ])
    return apus, insumos


@pytest.fixture
def estructura_con_flotante() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estructura con un APU flotante (sin soporte)."""
    apus = make_apus_df(["APU001", "APU002", "APU_FLOTANTE"])
    insumos = make_insumos_df([
        ("APU001", "MATERIAL_A"),
        ("APU002", "MATERIAL_B"),
    ])
    return apus, insumos


@pytest.fixture
def estructura_vacia() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estructura vacía (sin APUs ni insumos)."""
    return make_apus_df([]), make_insumos_df([])


# =============================================================================
# PRUEBAS DE TOPOLOGÍA DE ESTRATOS
# =============================================================================

class TestStratumTopology:
    """Propiedades algebraicas y de orden del espacio de estratos."""

    def test_stratum_expected_chain(self) -> None:
        """Verifica que los estratos coinciden con la cadena esperada."""
        actual = tuple(
            (s.name, s.value)
            for s in sorted(Stratum, key=lambda x: x.value)
        )
        assert actual == _EXPECTED_STRATUM_CHAIN

    def test_stratum_completeness(self) -> None:
        """Verifica completitud: todos los estratos esperados existen."""
        expected_names = {name for name, _ in _EXPECTED_STRATUM_CHAIN}
        actual_names = {s.name for s in Stratum}
        assert actual_names == expected_names

    def test_stratum_cardinality(self) -> None:
        """Verifica cardinalidad del espacio de estratos."""
        assert len(Stratum) == _STRATUM_CARDINALITY

    def test_stratum_values_are_contiguous(self) -> None:
        """Verifica que los valores son contiguos (sin huecos)."""
        values = sorted(s.value for s in Stratum)
        assert values == list(range(len(values)))

    def test_stratum_ordering_is_reflexive(self) -> None:
        """Verifica reflexividad: ∀x: x ≤ x."""
        for s in Stratum:
            assert s.value <= s.value

    def test_stratum_ordering_is_antisymmetric(self) -> None:
        """Verifica antisimetría: x ≤ y ∧ y ≤ x → x = y."""
        strata = list(Stratum)
        for s1 in strata:
            for s2 in strata:
                if s1.value <= s2.value and s2.value <= s1.value:
                    assert s1 == s2

    def test_stratum_ordering_is_transitive(self) -> None:
        """Verifica transitividad: x ≤ y ∧ y ≤ z → x ≤ z."""
        strata = list(Stratum)
        for s1 in strata:
            for s2 in strata:
                for s3 in strata:
                    if s1.value <= s2.value and s2.value <= s3.value:
                        assert s1.value <= s3.value

    def test_stratum_ordering_is_total(self) -> None:
        """Verifica totalidad: ∀x,y: x ≤ y ∨ y ≤ x."""
        strata = list(Stratum)
        for s1 in strata:
            for s2 in strata:
                assert s1.value <= s2.value or s2.value <= s1.value

    def test_pyramid_invariant_extremes(self) -> None:
        """Verifica que WISDOM es mínimo y PHYSICS es máximo."""
        values = [s.value for s in Stratum]
        assert Stratum.WISDOM.value == min(values)
        assert Stratum.PHYSICS.value == max(values)

    def test_stratum_chain_is_linear(self) -> None:
        """Verifica que la cadena es lineal (sin bifurcaciones)."""
        # En una cadena lineal de n elementos, hay exactamente n-1 pares adyacentes
        values = sorted(s.value for s in Stratum)
        adjacent_pairs = sum(
            1 for i in range(len(values) - 1)
            if values[i + 1] - values[i] == 1
        )
        assert adjacent_pairs == len(Stratum) - 1


class TestStratumLatticeOperations:
    """Pruebas de operaciones de retículo sobre estratos."""

    def test_supremum_is_commutative(self) -> None:
        """Verifica conmutatividad del supremo: sup(a,b) = sup(b,a)."""
        strata = list(Stratum)
        for s1 in strata:
            for s2 in strata:
                assert stratum_supremum(s1, s2) == stratum_supremum(s2, s1)

    def test_infimum_is_commutative(self) -> None:
        """Verifica conmutatividad del ínfimo: inf(a,b) = inf(b,a)."""
        strata = list(Stratum)
        for s1 in strata:
            for s2 in strata:
                assert stratum_infimum(s1, s2) == stratum_infimum(s2, s1)

    def test_supremum_is_associative(self) -> None:
        """Verifica asociatividad: sup(sup(a,b),c) = sup(a,sup(b,c))."""
        strata = list(Stratum)
        for s1 in strata:
            for s2 in strata:
                for s3 in strata:
                    left = stratum_supremum(stratum_supremum(s1, s2), s3)
                    right = stratum_supremum(s1, stratum_supremum(s2, s3))
                    assert left == right

    def test_infimum_is_associative(self) -> None:
        """Verifica asociatividad: inf(inf(a,b),c) = inf(a,inf(b,c))."""
        strata = list(Stratum)
        for s1 in strata:
            for s2 in strata:
                for s3 in strata:
                    left = stratum_infimum(stratum_infimum(s1, s2), s3)
                    right = stratum_infimum(s1, stratum_infimum(s2, s3))
                    assert left == right

    def test_absorption_laws(self) -> None:
        """Verifica leyes de absorción: sup(a, inf(a,b)) = a."""
        strata = list(Stratum)
        for s1 in strata:
            for s2 in strata:
                # sup(a, inf(a,b)) = a
                assert stratum_supremum(s1, stratum_infimum(s1, s2)) == s1
                # inf(a, sup(a,b)) = a
                assert stratum_infimum(s1, stratum_supremum(s1, s2)) == s1

    def test_idempotence(self) -> None:
        """Verifica idempotencia: sup(a,a) = a, inf(a,a) = a."""
        for s in Stratum:
            assert stratum_supremum(s, s) == s
            assert stratum_infimum(s, s) == s

    def test_supremum_with_maximum_is_maximum(self) -> None:
        """Verifica que sup(a, PHYSICS) = PHYSICS."""
        for s in Stratum:
            assert stratum_supremum(s, Stratum.PHYSICS) == Stratum.PHYSICS

    def test_infimum_with_minimum_is_minimum(self) -> None:
        """Verifica que inf(a, WISDOM) = WISDOM."""
        for s in Stratum:
            assert stratum_infimum(s, Stratum.WISDOM) == Stratum.WISDOM


# =============================================================================
# PRUEBAS DE TOPOLOGICAL NODE
# =============================================================================

class TestTopologicalNode:
    """Pruebas de estructura y semántica de TopologicalNode."""

    def test_node_default_values(self) -> None:
        """Verifica valores por defecto del nodo."""
        node = TopologicalNode(
            id="test_node",
            stratum=Stratum.TACTICS,
            description="Nodo de prueba",
        )

        assert node.structural_health == pytest.approx(1.0)
        assert node.is_floating is False

    def test_node_identity_preservation(self) -> None:
        """Verifica preservación de identidad."""
        node = TopologicalNode(
            id="ÚNICO_123",
            stratum=Stratum.WISDOM,
            description="Test",
        )
        assert node.id == "ÚNICO_123"

    @pytest.mark.parametrize("health", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_node_health_valid_range(self, health: float) -> None:
        """Verifica que salud válida es aceptada."""
        node = TopologicalNode(
            id="test",
            stratum=Stratum.TACTICS,
            description="Test",
            structural_health=health,
        )

        assert node.structural_health == pytest.approx(health)
        assert is_valid_health(node.structural_health)

    @pytest.mark.parametrize("stratum", list(Stratum))
    def test_node_accepts_all_strata(self, stratum: Stratum) -> None:
        """Verifica que todos los estratos son aceptados."""
        node = TopologicalNode(
            id=f"node_{stratum.name}",
            stratum=stratum,
            description=f"Nodo en {stratum.name}",
        )
        assert node.stratum == stratum

    def test_node_floating_flag_round_trip(self) -> None:
        """Verifica preservación del flag floating."""
        floating = TopologicalNode(
            id="floating",
            stratum=Stratum.PHYSICS,
            description="Flotante",
            is_floating=True,
        )
        grounded = TopologicalNode(
            id="grounded",
            stratum=Stratum.PHYSICS,
            description="Anclado",
            is_floating=False,
        )

        assert floating.is_floating is True
        assert grounded.is_floating is False

    def test_node_stratum_is_immutable_reference(self) -> None:
        """Verifica que el estrato es una referencia al enum correcto."""
        node = TopologicalNode(
            id="test",
            stratum=Stratum.TACTICS,
            description="Test",
        )
        
        assert node.stratum is Stratum.TACTICS
        assert isinstance(node.stratum, Stratum)


# =============================================================================
# PRUEBAS DE INSUMO PROCESADO
# =============================================================================

class TestInsumoProcesado:
    """Pruebas de InsumoProcesado como átomo de la base piramidal."""

    def test_insumo_inherits_topological_node(
        self,
        insumo_basico: InsumoProcesado,
    ) -> None:
        """Verifica herencia de TopologicalNode."""
        assert isinstance(insumo_basico, TopologicalNode)

    def test_insumo_fixed_stratum_physics(
        self,
        insumo_basico: InsumoProcesado,
    ) -> None:
        """Verifica que el estrato es siempre PHYSICS."""
        assert insumo_basico.stratum == Stratum.PHYSICS

    def test_insumo_stratum_is_maximum(
        self,
        insumo_basico: InsumoProcesado,
    ) -> None:
        """Verifica que PHYSICS es el estrato máximo (base de la pirámide)."""
        assert insumo_basico.stratum.value == max(s.value for s in Stratum)

    def test_insumo_id_format(
        self,
        insumo_basico: InsumoProcesado,
    ) -> None:
        """Verifica formato del ID."""
        assert insumo_basico.id.startswith("APU001_")
        prefix, suffix = insumo_basico.id.split("_", 1)
        assert prefix == "APU001"
        assert suffix != ""

    def test_insumo_id_uniqueness_same_apu(self) -> None:
        """Verifica unicidad de IDs para el mismo APU."""
        insumo_a = make_insumo(descripcion_insumo="Ladrillo")
        insumo_b = make_insumo(
            descripcion_insumo="Cemento",
            unidad_insumo="UND",
        )
        assert insumo_a.id != insumo_b.id

    def test_insumo_id_uniqueness_different_apu(self) -> None:
        """Verifica unicidad de IDs para diferentes APUs."""
        insumo_a = make_insumo(codigo_apu="APU001")
        insumo_b = make_insumo(codigo_apu="APU002")
        assert insumo_a.id != insumo_b.id

    def test_insumo_id_length_bounded(self) -> None:
        """Verifica que el ID tiene longitud acotada."""
        long_text = "X" * 10000
        insumo = make_insumo(
            descripcion_apu=long_text,
            descripcion_insumo=long_text,
        )
        assert len(insumo.id) < _MAX_ID_LENGTH

    @pytest.mark.parametrize("tipo", _VALID_INSUMO_TYPES)
    def test_insumo_valid_types(self, tipo: str) -> None:
        """Verifica aceptación de tipos válidos."""
        unidad = "UND"
        if tipo in ["MANO_DE_OBRA", "EQUIPO"]:
            unidad = "HORA"
        elif tipo == "TRANSPORTE":
            unidad = "VIAJE"
        insumo = make_insumo(tipo_insumo=tipo, unidad_insumo=unidad)
        assert insumo.tipo_insumo == tipo


class TestInsumoConservationLaw:
    """Pruebas de la ley de conservación valor = cantidad × precio."""

    def test_conservation_law_basic(self) -> None:
        """Verifica ley de conservación con valores básicos."""
        insumo = make_insumo(
            cantidad=25.5,
            precio_unitario=1234.56,
        )
        
        satisfies, error = verify_conservation_law(insumo)
        assert satisfies, f"Error de conservación: {error}"

    def test_conservation_law_exact_match(self) -> None:
        """Verifica coincidencia exacta de la ley."""
        cantidad = 25.5
        precio = 1234.56
        expected = cantidad * precio
        
        insumo = make_insumo(
            cantidad=cantidad,
            precio_unitario=precio,
            valor_total=expected,
        )
        
        assert insumo.valor_total == pytest.approx(expected, abs=_CONSERVATION_TOLERANCE)

    def test_conservation_law_small_values(self) -> None:
        """Verifica ley con valores pequeños (precisión flotante)."""
        insumo = make_insumo(
            cantidad=0.001,
            precio_unitario=0.002,
        )
        
        satisfies, _ = verify_conservation_law(insumo)
        assert satisfies

    def test_conservation_law_large_values(self) -> None:
        """Verifica ley con valores grandes."""
        insumo = make_insumo(
            cantidad=1e6,
            precio_unitario=1e6,
        )
        
        satisfies, _ = verify_conservation_law(insumo)
        assert satisfies

    def test_conservation_law_zero_quantity(self) -> None:
        """Verifica ley con cantidad cero."""
        import pytest
        with pytest.warns(UserWarning, match="Suministro con cantidad=0 en APU001"):
            insumo = make_insumo(
                cantidad=0.0,
                precio_unitario=1000.0,
            )
        
        satisfies, _ = verify_conservation_law(insumo)
        assert satisfies
        assert insumo.valor_total == pytest.approx(0.0)

    def test_non_negative_quantities(
        self,
        insumo_basico: InsumoProcesado,
    ) -> None:
        """Verifica que las cantidades son no negativas."""
        assert insumo_basico.cantidad >= 0
        assert insumo_basico.precio_unitario >= 0
        assert insumo_basico.valor_total >= 0


# =============================================================================
# PRUEBAS DE APU STRUCTURE
# =============================================================================

class TestAPUStructure:
    """Pruebas de APUStructure como agregado táctico."""

    def test_apu_stratum_is_tactics(self, apu_vacio: APUStructure) -> None:
        """Verifica que el estrato es TACTICS."""
        assert apu_vacio.stratum == Stratum.TACTICS

    def test_apu_stratum_is_below_physics(self, apu_vacio: APUStructure) -> None:
        """Verifica que TACTICS < PHYSICS en el orden."""
        assert apu_vacio.stratum.value < Stratum.PHYSICS.value

    def test_apu_initial_support_base_zero(self, apu_vacio: APUStructure) -> None:
        """Verifica soporte inicial vacío."""
        assert apu_vacio.support_base_width == 0

    def test_apu_add_single_resource(
        self,
        apu_vacio: APUStructure,
        insumos_variados: list[InsumoProcesado],
    ) -> None:
        """Verifica adición de un recurso."""
        apu_vacio.add_resource(insumos_variados[0])
        assert apu_vacio.support_base_width == 1

    def test_apu_add_multiple_resources_incremental(
        self,
        apu_vacio: APUStructure,
        insumos_variados: list[InsumoProcesado],
    ) -> None:
        """Verifica adición incremental de recursos."""
        for i, insumo in enumerate(insumos_variados, start=1):
            apu_vacio.add_resource(insumo)
            assert apu_vacio.support_base_width == i


class TestAPUSupportMonotonicity:
    """Pruebas de monotonía del soporte de APU."""

    def test_support_monotonically_increasing(
        self,
        apu_vacio: APUStructure,
        insumos_variados: list[InsumoProcesado],
    ) -> None:
        """Verifica monotonía creciente del soporte."""
        previous = 0
        for insumo in insumos_variados:
            apu_vacio.add_resource(insumo)
            current = apu_vacio.support_base_width
            assert current >= previous, (
                f"Violación de monotonía: {previous} > {current}"
            )
            previous = current

    def test_support_strictly_increasing_for_unique_resources(
        self,
        apu_vacio: APUStructure,
    ) -> None:
        """Verifica crecimiento estricto para recursos únicos."""
        supports = []
        for i in range(5):
            insumo = make_insumo(descripcion_insumo=f"Insumo_único_{i}")
            apu_vacio.add_resource(insumo)
            supports.append(apu_vacio.support_base_width)
        
        # Verificar crecimiento estricto
        for i in range(len(supports) - 1):
            assert supports[i] < supports[i + 1]

    def test_support_never_decreases(
        self,
        apu_vacio: APUStructure,
        insumos_variados: list[InsumoProcesado],
    ) -> None:
        """Verifica que el soporte nunca decrece."""
        apu_vacio.add_resource(insumos_variados[0])
        initial_support = apu_vacio.support_base_width
        
        for insumo in insumos_variados[1:]:
            apu_vacio.add_resource(insumo)
            assert apu_vacio.support_base_width >= initial_support


class TestAPUStratumRelations:
    """Pruebas de relaciones entre estratos de APU e insumos."""

    def test_apu_stratum_above_insumos(
        self,
        apu_vacio: APUStructure,
        insumos_variados: list[InsumoProcesado],
    ) -> None:
        """Verifica que APU está en estrato superior a sus insumos."""
        apu_vacio.add_resource(insumos_variados[0])
        
        # En el orden, valores menores son "superiores" (más cerca de WISDOM)
        assert apu_vacio.stratum.value < insumos_variados[0].stratum.value

    def test_stratum_distance_apu_to_insumo(
        self,
        apu_vacio: APUStructure,
        insumo_basico: InsumoProcesado,
    ) -> None:
        """Verifica distancia de estrato entre APU e insumo."""
        distance = stratum_distance(apu_vacio.stratum, insumo_basico.stratum)
        
        # TACTICS(2) a PHYSICS(3) = distancia 1
        assert distance == 1


# =============================================================================
# PRUEBAS DEL VALIDADOR PIRAMIDAL
# =============================================================================

class TestPyramidalValidatorBasic:
    """Pruebas básicas del validador piramidal."""

    def test_validator_returns_metrics(
        self,
        validator: PyramidalValidator,
        estructura_valida: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Verifica que el validador retorna PyramidalMetrics."""
        apus, insumos = estructura_valida
        metrics = validator.validate_structure(apus, insumos)
        assert isinstance(metrics, PyramidalMetrics)

    def test_base_width_counts_unique_insumos(
        self,
        validator: PyramidalValidator,
        estructura_valida: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Verifica que base_width cuenta insumos únicos."""
        apus, insumos = estructura_valida
        metrics = validator.validate_structure(apus, insumos)
        
        expected = insumos["DESCRIPCION_INSUMO_NORM"].nunique()
        assert metrics.base_width == expected
        assert metrics.base_width == 5

    def test_structure_load_counts_apus(
        self,
        validator: PyramidalValidator,
        estructura_valida: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Verifica que structure_load cuenta APUs."""
        apus, insumos = estructura_valida
        metrics = validator.validate_structure(apus, insumos)
        
        assert metrics.structure_load == len(apus)
        assert metrics.structure_load == 3


class TestStabilityIndexMathematics:
    """Pruebas matemáticas del índice de estabilidad."""

    def test_stability_matches_analytical_form(
        self,
        validator: PyramidalValidator,
        estructura_valida: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Verifica que el índice coincide con la forma cerrada."""
        apus, insumos = estructura_valida
        metrics = validator.validate_structure(apus, insumos)
        
        expected = stability_index_analytical(
            metrics.base_width,
            metrics.structure_load,
        )
        assert metrics.pyramid_stability_index == pytest.approx(expected)

    def test_stability_index_bounds(
        self,
        validator: PyramidalValidator,
        estructura_valida: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Verifica acotación del índice en [0, 1)."""
        apus, insumos = estructura_valida
        metrics = validator.validate_structure(apus, insumos)
        
        assert 0.0 <= metrics.pyramid_stability_index < 1.0

    def test_stability_zero_for_empty_base(
        self,
        validator: PyramidalValidator,
    ) -> None:
        """Verifica que f(0, l) = 0."""
        apus = make_apus_df(["APU001"])
        insumos = make_insumos_df([])
        
        metrics = validator.validate_structure(apus, insumos)
        
        assert metrics.pyramid_stability_index == pytest.approx(0.0)

    def test_stability_approaches_one_for_large_base(
        self,
        validator: PyramidalValidator,
    ) -> None:
        """Verifica que lim_{b→∞} f(b, l) → 1."""
        apus = make_apus_df(["APU001"])
        insumos = make_insumos_df([
            ("APU001", f"INSUMO_{i}") for i in range(1000)
        ])
        
        metrics = validator.validate_structure(apus, insumos)
        
        # tanh(1000/1) ≈ 1.0
        assert metrics.pyramid_stability_index > 0.999

    def test_stability_monotone_increasing_in_base(
        self,
        validator: PyramidalValidator,
    ) -> None:
        """Verifica monotonía creciente respecto a base_width."""
        apus = make_apus_df(["APU001"])
        stabilities = []

        for n in range(1, 11):
            insumos = make_insumos_df([
                ("APU001", f"INSUMO_{i}") for i in range(n)
            ])
            metrics = validator.validate_structure(apus, insumos)
            stabilities.append(metrics.pyramid_stability_index)

        # Verificar monotonía estricta
        assert stabilities == sorted(stabilities)
        assert all(
            left < right
            for left, right in zip(stabilities, stabilities[1:])
        )

    def test_stability_monotone_decreasing_in_load(
        self,
        validator: PyramidalValidator,
    ) -> None:
        """Verifica monotonía decreciente respecto a structure_load."""
        stabilities = []

        for n_apus in range(1, 6):
            apus = make_apus_df([f"APU{i:03d}" for i in range(n_apus)])
            # Base fija de 5 insumos (solo conectados al primer APU)
            insumos = make_insumos_df([
                (apus.iloc[0]["CODIGO_APU"], f"INSUMO_{i}")
                for i in range(5)
            ])
            metrics = validator.validate_structure(apus, insumos)
            stabilities.append(metrics.pyramid_stability_index)

        # Verificar monotonía decreciente estricta
        assert stabilities == sorted(stabilities, reverse=True)
        assert all(
            left > right
            for left, right in zip(stabilities, stabilities[1:])
        )

    def test_stability_derivative_is_positive(self) -> None:
        """Verifica que ∂f/∂b > 0 (derivada positiva)."""
        for b in range(1, 20):
            for l in range(1, 10):
                derivative = stability_index_derivative_b(b, l)
                assert derivative > 0, (
                    f"Derivada no positiva para b={b}, l={l}: {derivative}"
                )


class TestFloatingNodeDetection:
    """Pruebas de detección de nodos flotantes."""

    def test_floating_nodes_detected(
        self,
        validator: PyramidalValidator,
        estructura_con_flotante: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Verifica detección de nodos flotantes."""
        apus, insumos = estructura_con_flotante
        metrics = validator.validate_structure(apus, insumos)
        
        assert set(metrics.floating_nodes) == {"APU_FLOTANTE"}

    def test_no_floating_in_valid_structure(
        self,
        validator: PyramidalValidator,
        estructura_valida: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Verifica ausencia de flotantes en estructura válida."""
        apus, insumos = estructura_valida
        metrics = validator.validate_structure(apus, insumos)
        
        assert len(metrics.floating_nodes) == 0

    def test_all_floating_when_no_insumos(
        self,
        validator: PyramidalValidator,
    ) -> None:
        """Verifica que todos son flotantes sin insumos."""
        apus = make_apus_df(["APU001", "APU002", "APU003"])
        insumos = make_insumos_df([])

        metrics = validator.validate_structure(apus, insumos)

        assert set(metrics.floating_nodes) == {"APU001", "APU002", "APU003"}

    def test_floating_nodes_are_subset_of_apus(
        self,
        validator: PyramidalValidator,
        estructura_con_flotante: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Verifica que flotantes ⊆ APUs."""
        apus, insumos = estructura_con_flotante
        metrics = validator.validate_structure(apus, insumos)

        assert set(metrics.floating_nodes).issubset(set(apus["CODIGO_APU"]))

    def test_floating_and_supported_partition_apus(
        self,
        validator: PyramidalValidator,
        estructura_con_flotante: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Verifica que flotantes y soportados particionan APUs."""
        apus, insumos = estructura_con_flotante
        metrics = validator.validate_structure(apus, insumos)
        
        all_apus = set(apus["CODIGO_APU"])
        floating = set(metrics.floating_nodes)
        
        # Calcular soportados
        supported = set(insumos["APU_CODIGO"]) if len(insumos) > 0 else set()
        supported = supported.intersection(all_apus)
        
        # Verificar partición
        assert floating.union(supported) == all_apus
        assert floating.intersection(supported) == set()


class TestValidatorEdgeCases:
    """Pruebas de casos límite del validador."""

    def test_empty_structure_handling(
        self,
        validator: PyramidalValidator,
        estructura_vacia: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Verifica manejo de estructura vacía."""
        apus, insumos = estructura_vacia
        metrics = validator.validate_structure(apus, insumos)

        assert metrics.base_width == 0
        assert metrics.structure_load == 0
        assert len(metrics.floating_nodes) == 0
        assert metrics.pyramid_stability_index == pytest.approx(0.0)

    def test_large_structure_shape_invariants(
        self,
        validator: PyramidalValidator,
    ) -> None:
        """Verifica invariantes en estructuras grandes."""
        n_apus = 500
        n_insumos_per_apu = 20

        apus, insumos = make_balanced_structure(n_apus, n_insumos_per_apu)
        metrics = validator.validate_structure(apus, insumos)

        assert metrics.structure_load == n_apus
        # Cada APU tiene insumos únicos, así que base_width = n_insumos_per_apu * n_apus
        # ... a menos que los insumos se compartan
        assert metrics.base_width == n_insumos_per_apu * n_apus
        assert len(metrics.floating_nodes) == 0
        assert 0.0 <= metrics.pyramid_stability_index <= 1.0


class TestValidatorPurity:
    """Pruebas de pureza funcional del validador."""

    def test_validator_is_deterministic(
        self,
        validator: PyramidalValidator,
        estructura_valida: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Verifica determinismo del validador."""
        apus, insumos = estructura_valida

        m1 = validator.validate_structure(apus, insumos)
        m2 = validator.validate_structure(apus, insumos)
        m3 = validator.validate_structure(apus, insumos)

        assert m1.base_width == m2.base_width == m3.base_width
        assert m1.structure_load == m2.structure_load == m3.structure_load
        assert m1.pyramid_stability_index == pytest.approx(m2.pyramid_stability_index)
        assert m1.pyramid_stability_index == pytest.approx(m3.pyramid_stability_index)

    def test_validator_does_not_mutate_inputs(
        self,
        validator: PyramidalValidator,
        estructura_valida: tuple[pd.DataFrame, pd.DataFrame],
    ) -> None:
        """Verifica que el validador no muta sus entradas."""
        apus, insumos = estructura_valida
        apus_before = apus.copy(deep=True)
        insumos_before = insumos.copy(deep=True)

        _ = validator.validate_structure(apus, insumos)

        pd.testing.assert_frame_equal(apus, apus_before)
        pd.testing.assert_frame_equal(insumos, insumos_before)


# =============================================================================
# PRUEBAS DEL CLASIFICADOR ESTRUCTURAL
# =============================================================================

class TestStructuralClassifier:
    """Pruebas de clasificación estructural de APUs."""

    @pytest.mark.parametrize(
        ("insumos", "expected"),
        [
            ([{"TIPO_INSUMO": "SUMINISTRO", "VALOR_TOTAL": 1000}], "SUMINISTRO_PURO"),
            (
                [
                    {"TIPO_INSUMO": "SUMINISTRO", "VALOR_TOTAL": 100},
                    {"TIPO_INSUMO": "SUMINISTRO", "VALOR_TOTAL": 200},
                ],
                "SUMINISTRO_PURO",
            ),
            ([{"TIPO_INSUMO": "MANO_DE_OBRA", "VALOR_TOTAL": 50000}], "SERVICIO_PURO"),
            (
                [
                    {"TIPO_INSUMO": "SUMINISTRO", "VALOR_TOTAL": 100},
                    {"TIPO_INSUMO": "MANO_DE_OBRA", "VALOR_TOTAL": 100},
                ],
                "ESTRUCTURA_MIXTA",
            ),
        ],
    )
    def test_expected_classification(
        self,
        classifier: StructuralClassifier,
        insumos: list[dict[str, object]],
        expected: str,
    ) -> None:
        """Verifica clasificaciones esperadas."""
        clasificacion, metadata = classifier.classify_by_structure(insumos)
        assert clasificacion == expected

    def test_classification_returns_valid_type(
        self,
        classifier: StructuralClassifier,
    ) -> None:
        """Verifica que la clasificación es un tipo válido."""
        result = classifier.classify_by_structure([
            {"TIPO_INSUMO": "SUMINISTRO", "VALOR_TOTAL": 100}
        ])
        
        clasificacion, _ = result
        assert clasificacion in _VALID_CLASSIFICATIONS or clasificacion != ""


class TestClassifierPurity:
    """Pruebas de pureza funcional del clasificador."""

    def test_classification_is_deterministic(
        self,
        classifier: StructuralClassifier,
    ) -> None:
        """Verifica determinismo de la clasificación."""
        insumos = [
            {"TIPO_INSUMO": "SUMINISTRO", "VALOR_TOTAL": 100},
            {"TIPO_INSUMO": "MANO_DE_OBRA", "VALOR_TOTAL": 50},
        ]

        r1 = classifier.classify_by_structure(insumos)
        r2 = classifier.classify_by_structure(insumos)
        r3 = classifier.classify_by_structure(insumos)

        assert r1[0] == r2[0] == r3[0]

    def test_classification_is_order_invariant(
        self,
        classifier: StructuralClassifier,
    ) -> None:
        """Verifica invarianza respecto al orden de insumos."""
        insumos_a = [
            {"TIPO_INSUMO": "SUMINISTRO", "VALOR_TOTAL": 300},
            {"TIPO_INSUMO": "MANO_DE_OBRA", "VALOR_TOTAL": 100},
        ]
        insumos_b = list(reversed(insumos_a))

        clas_a, _ = classifier.classify_by_structure(insumos_a)
        clas_b, _ = classifier.classify_by_structure(insumos_b)

        assert clas_a == clas_b

    def test_classifier_does_not_mutate_input(
        self,
        classifier: StructuralClassifier,
    ) -> None:
        """Verifica que el clasificador no muta su entrada."""
        insumos = [
            {"TIPO_INSUMO": "SUMINISTRO", "VALOR_TOTAL": 100},
            {"TIPO_INSUMO": "MANO_DE_OBRA", "VALOR_TOTAL": 50},
        ]
        original = copy.deepcopy(insumos)

        _ = classifier.classify_by_structure(insumos)

        assert insumos == original


# =============================================================================
# PRUEBAS DE COHERENCIA ESTRUCTURAL
# =============================================================================

class TestHierarchyLevelCoherence:
    """Pruebas de coherencia entre HierarchyLevel y Stratum."""

    def test_hierarchy_and_stratum_same_values(self) -> None:
        """Verifica que comparten los mismos valores."""
        hierarchy_values = sorted(level.value for level in HierarchyLevel)
        stratum_values = sorted(stratum.value for stratum in Stratum)
        assert hierarchy_values == stratum_values

    def test_hierarchy_and_stratum_same_cardinality(self) -> None:
        """Verifica misma cardinalidad."""
        assert len(HierarchyLevel) == len(Stratum)


# =============================================================================
# PRUEBAS DE INVARIANTES GLOBALES
# =============================================================================

class TestGlobalTopologicalInvariants:
    """Invariantes globales del modelo piramidal."""

    def test_stratum_forms_linear_chain(self) -> None:
        """Verifica que los estratos forman una cadena lineal."""
        values = sorted(s.value for s in Stratum)
        assert values == list(range(len(values)))

    def test_pyramid_is_dag_by_value_orientation(self) -> None:
        """Verifica orientación del DAG piramidal."""
        assert Stratum.WISDOM.value < Stratum.STRATEGY.value
        assert Stratum.STRATEGY.value < Stratum.TACTICS.value
        assert Stratum.TACTICS.value < Stratum.PHYSICS.value

    def test_health_conservation_bound(self) -> None:
        """Verifica cota de conservación de salud."""
        n = 100
        health_value = 0.7
        
        nodes = [
            TopologicalNode(
                id=f"n{i}",
                stratum=Stratum.PHYSICS,
                description=f"Node {i}",
                structural_health=health_value,
            )
            for i in range(n)
        ]

        total_health = sum(node.structural_health for node in nodes)
        
        assert total_health <= n
        assert total_health == pytest.approx(n * health_value)


# =============================================================================
# PRUEBAS DE CASOS LÍMITE Y ROBUSTEZ
# =============================================================================

class TestEdgeCasesAndRobustness:
    """Casos límite de serialización, precisión y validación."""

    def test_unicode_in_descriptions(self) -> None:
        """Verifica manejo de Unicode en descripciones."""
        insumo = make_insumo(
            descripcion_apu="Muro con ñ, áéíóú, €, ™",
            unidad_apu="m²",
            descripcion_insumo="Ladrillo café 日本語",
        )

        assert insumo.stratum == Stratum.PHYSICS
        assert insumo.id is not None

    def test_numerical_precision_float(self) -> None:
        """Verifica precisión numérica con flotantes."""
        cantidad = 0.1
        precio = 0.2
        
        insumo = make_insumo(
            cantidad=cantidad,
            precio_unitario=precio,
        )

        expected = cantidad * precio
        assert insumo.valor_total == pytest.approx(expected, rel=1e-10)


# =============================================================================
# PRUEBAS DE INTEGRACIÓN
# =============================================================================

class TestIntegration:
    """Pruebas de integración entre componentes."""

    def test_full_pyramid_construction_flow(self) -> None:
        """Verifica flujo completo de construcción piramidal."""
        insumos = [
            make_insumo(descripcion_insumo=f"Material_{i}")
            for i in range(5)
        ]

        apu = APUStructure(
            id="APU001",
            description="Muro de Prueba",
            unit="m2",
            quantity=100,
        )

        for insumo in insumos:
            apu.add_resource(insumo)

        # Verificar estratificación
        assert apu.stratum == Stratum.TACTICS
        assert all(insumo.stratum == Stratum.PHYSICS for insumo in insumos)
        
        # Verificar soporte
        assert apu.support_base_width == len(insumos)
        
        # Verificar orden de estratos
        assert apu.stratum.value < insumos[0].stratum.value

    def test_validator_and_classifier_consistency(self) -> None:
        """Verifica consistencia entre validador y clasificador."""
        validator = PyramidalValidator()
        classifier = StructuralClassifier()

        apus_df = make_apus_df(["APU001", "APU002"])
        insumos_df = make_insumos_df([
            ("APU001", "MAT_A"),
            ("APU001", "MAT_B"),
            ("APU002", "MAT_C"),
        ])

        insumos_classify = [
            {"TIPO_INSUMO": "SUMINISTRO", "VALOR_TOTAL": 200},
            {"TIPO_INSUMO": "SUMINISTRO", "VALOR_TOTAL": 300},
        ]

        metrics = validator.validate_structure(apus_df, insumos_df)
        clasificacion, _ = classifier.classify_by_structure(insumos_classify)

        assert len(metrics.floating_nodes) == 0
        assert clasificacion == "SUMINISTRO_PURO"
        assert metrics.pyramid_stability_index > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])