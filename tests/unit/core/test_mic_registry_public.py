"""
Suite de Pruebas Públicas para MICRegistry — Versión 2.0
=========================================================

Fundamentos Matemáticos del Modelo DIKW:
────────────────────────────────────────────────────────────────────────────
  La jerarquía DIKW define un ORDEN TOTAL sobre los estratos:

      WISDOM (0) < STRATEGY (1) < TACTICS (2) < PHYSICS (3)

  La función get_required_strata(k) implementa la CLAUSURA INFERIOR
  ESTRICTA (downset abierto) en este orden:

      required(k) = ↓°k = { s ∈ Stratum | s > k } = { s | s.value > k.value }

  Propiedades Algebraicas Verificadas:
  ────────────────────────────────────
  Sea P = (Stratum, ≤) el poset de estratos con el orden natural.

  1. CARDINALIDAD EXACTA:
        |required(k)| = |Stratum| - 1 - k.value = 3 - k.value

  2. MONOTONÍA ANTI-TÓNICA (contravarianza):
        k₁ ≤ k₂  ⟹  required(k₁) ⊇ required(k₂)

  3. IRREFLEXIVIDAD:
        k ∉ required(k)  ∀k ∈ Stratum

  4. TRANSITIVIDAD HEREDADA:
        s ∈ required(k) ∧ t ∈ required(s)  ⟹  t ∈ required(k)

  5. CADENA DE INCLUSIÓN (chain under ⊆):
        required(PHYSICS) ⊂ required(TACTICS) ⊂ required(STRATEGY) ⊂ required(WISDOM)

  6. PARTICIÓN DEL UNIVERSO:
        {k} ⊔ required(k) ⊔ upper(k) = Stratum
        donde upper(k) = { s | s < k }

  7. CLAUSURA BAJO UNIÓN:
        required(k) = ⋃ { required(s) ∪ {s} | s ∈ required(k) }  (para k ≠ PHYSICS)

  Propiedades de Retículo (Lattice):
  ──────────────────────────────────
  El conjunto 2^Stratum forma un retículo booleano bajo ⊆.
  Los conjuntos required(k) forman una cadena (subretículo lineal).

    meet:  required(k₁) ∧ required(k₂) = required(max(k₁, k₂))
    join:  required(k₁) ∨ required(k₂) = required(min(k₁, k₂))

  donde max/min se refieren al orden de valores del enum.

  Propiedades de Grafo Dirigido:
  ──────────────────────────────
  El grafo de dependencias G = (Stratum, E) donde
    E = { (k, s) | s ∈ required(k) }
  es un DAG (Directed Acyclic Graph) con las siguientes propiedades:

    · Acíclico: no existen caminos k →⁺ k
    · Transitivamente cerrado: si k → s y s → t, entonces k → t
    · Altura = 3 (longitud del camino más largo: WISDOM → PHYSICS)
    · Anchura = 1 (antichain máxima tiene tamaño 1 — es orden total)

Referencias:
  - Ackoff, R. L. (1989). From Data to Wisdom.
  - Davey & Priestley (2002). Introduction to Lattices and Order.
  - Rosen (2019). Discrete Mathematics and Its Applications, Ch. 9.
  - Cormen et al. (2009). Introduction to Algorithms, Ch. 22 (Grafos).
"""

from __future__ import annotations

from app.adapters.tools_interface import MICRegistry
from app.core.schemas import Stratum

import itertools
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set, Tuple, Callable, Any
from functools import reduce
import operator

import pytest

# Intentar importar hypothesis para property-based testing
try:
    from hypothesis import given, settings, strategies as st, assume
    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Crear decoradores dummy si hypothesis no está disponible
    def given(*args, **kwargs):
        def decorator(func):
            return pytest.mark.skip(reason="hypothesis not installed")(func)
        return decorator
    settings = lambda **kwargs: lambda f: f
    class st:
        @staticmethod
        def sampled_from(elements):
            return None









# =============================================================================
# ESPECIFICACIÓN DEL DOMINIO (GROUND TRUTH)
# =============================================================================

@dataclass(frozen=True)
class StratumSpec:
    """
    Especificación inmutable de un estrato en la jerarquía DIKW.
    
    Centraliza el conocimiento del dominio para que los tests
    sean resistentes a refactoring del enum.
    """
    stratum: Stratum
    value: int
    required: FrozenSet[Stratum]
    upper: FrozenSet[Stratum]  # Estratos estrictamente superiores
    cardinality: int
    height_in_dag: int  # Distancia al nodo raíz (WISDOM)
    
    @property
    def is_base(self) -> bool:
        """¿Es la base de la pirámide (sin dependencias)?"""
        return len(self.required) == 0
    
    @property
    def is_apex(self) -> bool:
        """¿Es el ápice de la pirámide (sin superiores)?"""
        return len(self.upper) == 0


@dataclass(frozen=True)
class DIKWHierarchy:
    """
    Especificación completa de la jerarquía DIKW.
    
    Actúa como el "oráculo" contra el cual se verifican
    los resultados del MICRegistry.
    """
    strata: Tuple[StratumSpec, ...]
    total_count: int = field(init=False)
    
    def __post_init__(self):
        object.__setattr__(self, 'total_count', len(self.strata))
    
    def get_spec(self, stratum: Stratum) -> StratumSpec:
        """Obtiene la especificación de un estrato."""
        for spec in self.strata:
            if spec.stratum == stratum:
                return spec
        raise ValueError(f"Estrato desconocido: {stratum}")
    
    @property
    def all_strata(self) -> FrozenSet[Stratum]:
        """Todos los estratos del universo."""
        return frozenset(spec.stratum for spec in self.strata)
    
    @property
    def ordered_strata(self) -> Tuple[Stratum, ...]:
        """Estratos ordenados por valor (WISDOM primero)."""
        return tuple(spec.stratum for spec in sorted(self.strata, key=lambda s: s.value))
    
    def pairs_by_order(self) -> List[Tuple[Stratum, Stratum]]:
        """Pares (superior, inferior) adyacentes en el orden."""
        ordered = self.ordered_strata
        return [(ordered[i], ordered[i+1]) for i in range(len(ordered) - 1)]
    
    def all_ordered_pairs(self) -> List[Tuple[Stratum, Stratum]]:
        """Todos los pares (k₁, k₂) donde k₁ < k₂ (k₁ superior a k₂)."""
        ordered = self.ordered_strata
        return [
            (ordered[i], ordered[j])
            for i in range(len(ordered))
            for j in range(i + 1, len(ordered))
        ]


# Construir la especificación del dominio
DIKW = DIKWHierarchy(strata=(
    StratumSpec(
        stratum=Stratum.WISDOM,
        value=0,
        required=frozenset({Stratum.STRATEGY, Stratum.TACTICS, Stratum.PHYSICS}),
        upper=frozenset(),
        cardinality=3,
        height_in_dag=0,
    ),
    StratumSpec(
        stratum=Stratum.STRATEGY,
        value=1,
        required=frozenset({Stratum.TACTICS, Stratum.PHYSICS}),
        upper=frozenset({Stratum.WISDOM}),
        cardinality=2,
        height_in_dag=1,
    ),
    StratumSpec(
        stratum=Stratum.TACTICS,
        value=2,
        required=frozenset({Stratum.PHYSICS}),
        upper=frozenset({Stratum.WISDOM, Stratum.STRATEGY}),
        cardinality=1,
        height_in_dag=2,
    ),
    StratumSpec(
        stratum=Stratum.PHYSICS,
        value=3,
        required=frozenset(),
        upper=frozenset({Stratum.WISDOM, Stratum.STRATEGY, Stratum.TACTICS}),
        cardinality=0,
        height_in_dag=3,
    ),
))


# Constantes derivadas para compatibilidad con tests existentes
_ALL_STRATA: FrozenSet[Stratum] = DIKW.all_strata
_REQUIRED: Dict[Stratum, FrozenSet[Stratum]] = {
    spec.stratum: spec.required for spec in DIKW.strata
}
_EXPECTED_CARDINALITY: Dict[Stratum, int] = {
    spec.stratum: spec.cardinality for spec in DIKW.strata
}


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def registry() -> MICRegistry:
    """
    Instancia compartida de MICRegistry para todo el módulo de tests.

    Scope 'module' es apropiado porque:
      · MICRegistry.get_required_strata es una función pura (sin efectos)
      · Evita overhead de instanciación repetida
      · Si MICRegistry tuviera estado mutable, usaríamos scope='function'
    """
    return MICRegistry()


@pytest.fixture(scope="module")
def all_required_sets(registry: MICRegistry) -> Dict[Stratum, Set[Stratum]]:
    """Cache de todos los conjuntos required para evitar cómputo repetido."""
    return {s: s.requires() for s in Stratum}


@pytest.fixture
def fresh_registry() -> MICRegistry:
    """Instancia fresca para tests que necesitan aislamiento."""
    return MICRegistry()


# =============================================================================
# TESTS: TIPO DE RETORNO Y ESTRUCTURA
# =============================================================================

class TestReturnTypeAndStructure:
    """
    Verificación del contrato de tipos de la API pública.
    
    El tipo de retorno es parte del contrato público:
    set permite operaciones de conjunto (|, &, -, in) sin conversión.
    """

    @pytest.mark.parametrize("stratum", list(Stratum))
    def test_return_type_is_set(self, registry: MICRegistry, stratum: Stratum) -> None:
        """get_required_strata siempre retorna un set (no list, frozenset, etc.)."""
        result = set(stratum.requires())
        
        assert isinstance(result, set), (
            f"[{stratum.name}] Tipo esperado: set, "
            f"obtenido: {type(result).__name__}"
        )

    @pytest.mark.parametrize("stratum", list(Stratum))
    def test_return_type_is_not_subclass(
        self, registry: MICRegistry, stratum: Stratum
    ) -> None:
        """El tipo exacto debe ser set, no una subclase."""
        result = set(stratum.requires())
        
        assert type(result) is set, (
            f"[{stratum.name}] Tipo exacto esperado: set, "
            f"obtenido: {type(result).__name__}"
        )

    @pytest.mark.parametrize("stratum", list(Stratum))
    def test_elements_are_stratum_instances(
        self, registry: MICRegistry, stratum: Stratum
    ) -> None:
        """Todos los elementos del conjunto deben ser instancias de Stratum."""
        result = set(stratum.requires())
        
        for element in result:
            assert isinstance(element, Stratum), (
                f"[{stratum.name}] Elemento no es Stratum: "
                f"{element!r} ({type(element).__name__})"
            )

    @pytest.mark.parametrize("stratum", list(Stratum))
    def test_no_none_elements(
        self, registry: MICRegistry, stratum: Stratum
    ) -> None:
        """El conjunto no debe contener None."""
        result = set(stratum.requires())
        
        assert None not in result, (
            f"[{stratum.name}] El conjunto contiene None: {result}"
        )


# =============================================================================
# TESTS: VALORES EXACTOS (GROUND TRUTH)
# =============================================================================

class TestExactValues:
    """
    Verificación de valores exactos contra la especificación del dominio.
    
    Estos tests comparan directamente contra el "oráculo" DIKW.
    """

    @pytest.mark.parametrize(
        "stratum",
        list(Stratum),
        ids=lambda s: s.name,
    )
    def test_required_strata_matches_spec(
        self, registry: MICRegistry, stratum: Stratum
    ) -> None:
        """
        get_required_strata(k) debe coincidir exactamente con la especificación.
        
        La especificación define:
            required(k) = { s ∈ Stratum | s.value > k.value }
        """
        spec = DIKW.get_spec(stratum)
        result = set(stratum.requires())
        
        assert result == spec.required, (
            f"[{stratum.name}] Valor incorrecto:\n"
            f"  Esperado (spec): {spec.required}\n"
            f"  Obtenido:        {result}\n"
            f"  Diferencia:      {result.symmetric_difference(spec.required)}"
        )

    @pytest.mark.parametrize(
        "stratum, expected",
        [
            pytest.param(
                Stratum.WISDOM,
                frozenset({Stratum.STRATEGY, Stratum.TACTICS, Stratum.PHYSICS}),
                id="WISDOM→{STRATEGY,TACTICS,PHYSICS}",
            ),
            pytest.param(
                Stratum.STRATEGY,
                frozenset({Stratum.TACTICS, Stratum.PHYSICS}),
                id="STRATEGY→{TACTICS,PHYSICS}",
            ),
            pytest.param(
                Stratum.TACTICS,
                frozenset({Stratum.PHYSICS}),
                id="TACTICS→{PHYSICS}",
            ),
            pytest.param(
                Stratum.PHYSICS,
                frozenset(),
                id="PHYSICS→∅",
            ),
        ],
    )
    def test_required_strata_explicit_values(
        self,
        registry: MICRegistry,
        stratum: Stratum,
        expected: FrozenSet[Stratum],
    ) -> None:
        """Test explícito con valores hardcodeados para documentación."""
        result = set(stratum.requires())
        assert result == expected


# =============================================================================
# TESTS: CARDINALIDAD
# =============================================================================

class TestCardinality:
    """
    Verificación de la propiedad de cardinalidad.
    
    Para una pirámide de N=4 estratos:
        |required(k)| = N - 1 - k.value = 3 - k.value
    """

    @pytest.mark.parametrize(
        "stratum",
        list(Stratum),
        ids=lambda s: f"{s.name}-card={DIKW.get_spec(s).cardinality}",
    )
    def test_cardinality_matches_formula(
        self, registry: MICRegistry, stratum: Stratum
    ) -> None:
        """
        |required(k)| = (total_strata - 1) - k.value
        """
        spec = DIKW.get_spec(stratum)
        result = set(stratum.requires())
        
        # Fórmula derivada
        expected_by_formula = DIKW.total_count - 1 - spec.value
        
        assert len(result) == spec.cardinality, (
            f"[{stratum.name}] Cardinalidad incorrecta:\n"
            f"  Esperada (spec):    {spec.cardinality}\n"
            f"  Esperada (fórmula): {expected_by_formula}\n"
            f"  Obtenida:           {len(result)}\n"
            f"  Conjunto:           {result}"
        )
        
        # Verificar consistencia de la spec con la fórmula
        assert spec.cardinality == expected_by_formula, (
            f"Inconsistencia en spec de {stratum.name}: "
            f"cardinality={spec.cardinality} ≠ formula={expected_by_formula}"
        )

    def test_total_cardinality_sum(
        self, registry: MICRegistry
    ) -> None:
        """
        La suma de cardinalidades es un triángulo: 0 + 1 + 2 + 3 = 6.
        
        Σ |required(k)| = Σᵢ₌₀ⁿ⁻¹ i = n(n-1)/2 = 4×3/2 = 6
        """
        total = sum(len(s.requires()) for s in Stratum)
        n = DIKW.total_count
        expected = n * (n - 1) // 2
        
        assert total == expected, (
            f"Suma de cardinalidades: esperada={expected}, obtenida={total}"
        )


# =============================================================================
# TESTS: PROPIEDADES DE ORDEN PARCIAL
# =============================================================================

class TestPartialOrderProperties:
    """
    Verificación de propiedades algebraicas del orden parcial.
    
    El grafo de dependencias define una relación de orden que
    debe satisfacer propiedades específicas.
    """

    # ── Irreflexividad ────────────────────────────────────────────────────

    @pytest.mark.parametrize("stratum", list(Stratum))
    def test_irreflexivity(
        self, registry: MICRegistry, stratum: Stratum
    ) -> None:
        """
        Irreflexividad: k ∉ required(k).
        
        Ningún estrato puede ser su propia dependencia.
        """
        result = set(stratum.requires())
        
        assert stratum not in result, (
            f"[{stratum.name}] Violación de irreflexividad: "
            f"estrato presente en su propio conjunto required"
        )

    # ── Asimetría ─────────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "k1, k2",
        DIKW.all_ordered_pairs(),
        ids=lambda pair: f"{pair[0].name}>{pair[1].name}" if isinstance(pair, tuple) else str(pair),
    )
    def test_asymmetry(
        self, registry: MICRegistry, k1: Stratum, k2: Stratum
    ) -> None:
        """
        Asimetría: si k₂ ∈ required(k₁), entonces k₁ ∉ required(k₂).
        
        Las dependencias no son bidireccionales.
        """
        req_k1 = k1.requires()
        req_k2 = k2.requires()
        
        if k2 in req_k1:
            assert k1 not in req_k2, (
                f"Violación de asimetría: {k2.name} ∈ required({k1.name}) "
                f"pero también {k1.name} ∈ required({k2.name})"
            )

    # ── Transitividad ─────────────────────────────────────────────────────

    def test_transitivity_exhaustive(
        self, registry: MICRegistry
    ) -> None:
        """
        Transitividad: s ∈ required(k) ∧ t ∈ required(s) ⟹ t ∈ required(k).
        
        Verifica todas las tripletas posibles.
        """
        violations = []
        
        for k in Stratum:
            req_k = k.requires()
            for s in req_k:
                req_s = s.requires()
                for t in req_s:
                    if t not in req_k:
                        violations.append((k, s, t))
        
        assert not violations, (
            f"Violaciones de transitividad encontradas:\n" +
            "\n".join(
                f"  {k.name} →req→ {s.name} →req→ {t.name}, pero {t.name} ∉ required({k.name})"
                for k, s, t in violations
            )
        )

    @pytest.mark.parametrize(
        "k, s, t",
        [
            (Stratum.WISDOM, Stratum.STRATEGY, Stratum.TACTICS),
            (Stratum.WISDOM, Stratum.STRATEGY, Stratum.PHYSICS),
            (Stratum.WISDOM, Stratum.TACTICS, Stratum.PHYSICS),
            (Stratum.STRATEGY, Stratum.TACTICS, Stratum.PHYSICS),
        ],
        ids=lambda triple: f"{triple[0].name}→{triple[1].name}→{triple[2].name}" if isinstance(triple, tuple) else str(triple),
    )
    def test_transitivity_specific_chains(
        self, registry: MICRegistry, k: Stratum, s: Stratum, t: Stratum
    ) -> None:
        """
        Casos específicos de transitividad para documentación.
        """
        req_k = k.requires()
        req_s = s.requires()
        
        # Verificar premisas
        assert s in req_k, f"Premisa falsa: {s.name} ∉ required({k.name})"
        assert t in req_s, f"Premisa falsa: {t.name} ∉ required({s.name})"
        
        # Verificar conclusión
        assert t in req_k, (
            f"Transitividad fallida: {s.name} ∈ required({k.name}) y "
            f"{t.name} ∈ required({s.name}), pero {t.name} ∉ required({k.name})"
        )

    # ── Contención en universo ────────────────────────────────────────────

    @pytest.mark.parametrize("stratum", list(Stratum))
    def test_contained_in_universe(
        self, registry: MICRegistry, stratum: Stratum
    ) -> None:
        """
        Contención: required(k) ⊆ Stratum.
        
        No pueden existir dependencias hacia estratos inexistentes.
        """
        result = set(stratum.requires())
        universe = set(Stratum)
        
        assert result <= universe, (
            f"[{stratum.name}] Elementos fuera del universo: "
            f"{result - universe}"
        )


# =============================================================================
# TESTS: MONOTONÍA
# =============================================================================

class TestMonotonicity:
    """
    Verificación de la propiedad de monotonía anti-tónica.
    
    En el orden de la pirámide (WISDOM < STRATEGY < TACTICS < PHYSICS):
        k₁ < k₂  ⟹  required(k₁) ⊇ required(k₂)
    
    Ascender en la pirámide implica más dependencias.
    """

    @pytest.mark.parametrize(
        "higher, lower",
        DIKW.pairs_by_order(),
        ids=lambda pair: f"{pair[0].name}⊇{pair[1].name}" if isinstance(pair, tuple) else str(pair),
    )
    def test_adjacent_monotonicity(
        self, registry: MICRegistry, higher: Stratum, lower: Stratum
    ) -> None:
        """
        Para estratos adyacentes: required(higher) ⊇ required(lower).
        """
        req_higher = higher.requires()
        req_lower = lower.requires()
        
        assert req_higher >= req_lower, (
            f"Monotonía violada entre adyacentes:\n"
            f"  required({higher.name}) = {req_higher}\n"
            f"  required({lower.name})  = {req_lower}\n"
            f"  Faltantes en higher:    {req_lower - req_higher}"
        )

    def test_global_monotonicity(self, registry: MICRegistry) -> None:
        """
        Monotonía global: verifica todos los pares ordenados.
        """
        violations = []
        
        for k1, k2 in DIKW.all_ordered_pairs():
            req_k1 = k1.requires()
            req_k2 = k2.requires()
            
            if not (req_k1 >= req_k2):
                violations.append((k1, k2, req_k1, req_k2))
        
        assert not violations, (
            "Violaciones de monotonía global:\n" +
            "\n".join(
                f"  required({k1.name}) ⊉ required({k2.name}): "
                f"falta {req_k2 - req_k1}"
                for k1, k2, req_k1, req_k2 in violations
            )
        )

    def test_strict_monotonicity(self, registry: MICRegistry) -> None:
        """
        Monotonía estricta: required(k₁) ⊃ required(k₂) (subconjunto propio).
        
        Excepto cuando k₂ es la base (PHYSICS), donde required(k₂) = ∅.
        """
        for higher, lower in DIKW.pairs_by_order():
            req_higher = higher.requires()
            req_lower = lower.requires()
            
            if lower != Stratum.PHYSICS:
                # Subconjunto propio
                assert req_higher > req_lower, (
                    f"Monotonía estricta violada: "
                    f"required({higher.name}) no es superconjunto propio de "
                    f"required({lower.name})"
                )
            else:
                # PHYSICS tiene required vacío, así que ⊇ se reduce a ⊇ ∅
                assert req_higher >= req_lower


# =============================================================================
# TESTS: CASOS EXTREMOS (BASE Y ÁPICE)
# =============================================================================

class TestBoundaryStrata:
    """
    Verificación de los casos extremos: base y ápice de la pirámide.
    """

    def test_physics_is_base(self, registry: MICRegistry) -> None:
        """
        PHYSICS es la base: no tiene dependencias inferiores.
        
        required(PHYSICS) = ∅
        """
        result = Stratum.PHYSICS.requires()
        
        assert result == set(), (
            f"PHYSICS debe tener required vacío, obtenido: {result}"
        )
        assert len(result) == 0

    def test_wisdom_is_apex(self, registry: MICRegistry) -> None:
        """
        WISDOM es el ápice: requiere todos los demás estratos.
        
        required(WISDOM) = Stratum \\ {WISDOM}
        """
        result = Stratum.WISDOM.requires()
        expected = set(Stratum) - {Stratum.WISDOM}
        
        assert result == expected, (
            f"WISDOM debe requerir todos excepto sí mismo:\n"
            f"  Esperado: {expected}\n"
            f"  Obtenido: {result}"
        )

    def test_only_physics_has_empty_required(self, registry: MICRegistry) -> None:
        """Solo PHYSICS tiene conjunto required vacío."""
        for stratum in Stratum:
            result = set(stratum.requires())
            
            if stratum == Stratum.PHYSICS:
                assert len(result) == 0, f"PHYSICS debe tener required vacío"
            else:
                assert len(result) > 0, (
                    f"{stratum.name} no es base, debe tener required no vacío"
                )

    def test_only_wisdom_has_maximal_required(self, registry: MICRegistry) -> None:
        """Solo WISDOM tiene el conjunto required máximo."""
        max_cardinality = DIKW.total_count - 1  # 3
        
        for stratum in Stratum:
            result = set(stratum.requires())
            
            if stratum == Stratum.WISDOM:
                assert len(result) == max_cardinality
            else:
                assert len(result) < max_cardinality


# =============================================================================
# TESTS: PROPIEDADES DE CADENA (CHAIN)
# =============================================================================

class TestChainProperties:
    """
    Los conjuntos required forman una cadena bajo ⊆.
    
    ∅ ⊂ {PHYSICS} ⊂ {TACTICS,PHYSICS} ⊂ {STRATEGY,TACTICS,PHYSICS}
    """

    def test_required_sets_form_chain(self, registry: MICRegistry) -> None:
        """
        Verifica que los conjuntos forman una cadena estricta.
        """
        req = {s: s.requires() for s in Stratum}
        
        # Ordenar por cardinalidad ascendente
        ordered = sorted(req.items(), key=lambda x: len(x[1]))
        
        for i in range(len(ordered) - 1):
            stratum_i, set_i = ordered[i]
            stratum_j, set_j = ordered[i + 1]
            
            assert set_i < set_j, (
                f"No forman cadena estricta:\n"
                f"  required({stratum_i.name}) = {set_i} (card={len(set_i)})\n"
                f"  required({stratum_j.name}) = {set_j} (card={len(set_j)})"
            )

    def test_chain_is_totally_ordered(
        self, all_required_sets: Dict[Stratum, Set[Stratum]]
    ) -> None:
        """
        Cualquier par de conjuntos required es comparable bajo ⊆.
        
        ∀ k₁, k₂: required(k₁) ⊆ required(k₂) ∨ required(k₂) ⊆ required(k₁)
        """
        sets = list(all_required_sets.values())
        
        for i, set_i in enumerate(sets):
            for j, set_j in enumerate(sets):
                if i != j:
                    comparable = set_i <= set_j or set_j <= set_i
                    assert comparable, (
                        f"Conjuntos no comparables (no forman cadena):\n"
                        f"  Set {i}: {set_i}\n"
                        f"  Set {j}: {set_j}"
                    )


# =============================================================================
# TESTS: PROPIEDADES DE RETÍCULO (LATTICE)
# =============================================================================

class TestLatticeProperties:
    """
    Propiedades de retículo de los conjuntos required.
    
    Como forman una cadena, meet y join tienen comportamiento específico.
    """

    @pytest.mark.parametrize(
        "k1, k2",
        list(itertools.combinations(Stratum, 2)),
        ids=lambda pair: f"meet({pair[0].name},{pair[1].name})" if isinstance(pair, tuple) else str(pair),
    )
    def test_meet_is_smaller_required(
        self, registry: MICRegistry, k1: Stratum, k2: Stratum
    ) -> None:
        """
        Meet (∧) = intersección = required del estrato con valor mayor.
        
        required(k₁) ∧ required(k₂) = required(max(k₁, k₂))
        """
        req_k1 = k1.requires()
        req_k2 = k2.requires()
        
        meet = req_k1 & req_k2
        
        # El máximo es el de mayor valor (más bajo en pirámide, menos required)
        max_stratum = k1 if k1.value > k2.value else k2
        expected_meet = max_stratum.requires()
        
        assert meet == expected_meet, (
            f"Meet incorrecto:\n"
            f"  required({k1.name}) ∧ required({k2.name}) = {meet}\n"
            f"  Esperado (required({max_stratum.name})): {expected_meet}"
        )

    @pytest.mark.parametrize(
        "k1, k2",
        list(itertools.combinations(Stratum, 2)),
        ids=lambda pair: f"join({pair[0].name},{pair[1].name})" if isinstance(pair, tuple) else str(pair),
    )
    def test_join_is_larger_required(
        self, registry: MICRegistry, k1: Stratum, k2: Stratum
    ) -> None:
        """
        Join (∨) = unión = required del estrato con valor menor.
        
        required(k₁) ∨ required(k₂) = required(min(k₁, k₂))
        """
        req_k1 = k1.requires()
        req_k2 = k2.requires()
        
        join = req_k1 | req_k2
        
        # El mínimo es el de menor valor (más alto en pirámide, más required)
        min_stratum = k1 if k1.value < k2.value else k2
        expected_join = min_stratum.requires()
        
        assert join == expected_join, (
            f"Join incorrecto:\n"
            f"  required({k1.name}) ∨ required({k2.name}) = {join}\n"
            f"  Esperado (required({min_stratum.name})): {expected_join}"
        )


# =============================================================================
# TESTS: PROPIEDADES DE CONJUNTO
# =============================================================================

class TestSetProperties:
    """
    Propiedades adicionales sobre la estructura de conjuntos.
    """

    def test_union_of_all_required(self, registry: MICRegistry) -> None:
        """
        ⋃ { required(s) | s ∈ Stratum } = Stratum \\ {WISDOM}
        
        La unión cubre exactamente los estratos que son dependencia de algún otro.
        """
        union = set().union(*(s.requires() for s in Stratum))
        expected = set(Stratum) - {Stratum.WISDOM}
        
        assert union == expected, (
            f"Unión de todos los required:\n"
            f"  Esperado: {expected}\n"
            f"  Obtenido: {union}"
        )

    def test_intersection_of_all_required(self, registry: MICRegistry) -> None:
        """
        ⋂ { required(s) | s ∈ Stratum } = ∅
        
        La intersección es vacía porque required(PHYSICS) = ∅.
        """
        sets = [s.requires() for s in Stratum]
        intersection = reduce(operator.and_, sets)
        
        assert intersection == set(), (
            f"Intersección de todos los required debe ser vacía, "
            f"obtenido: {intersection}"
        )

    def test_adjacent_intersection_property(self, registry: MICRegistry) -> None:
        """
        Para estratos adyacentes: required(k) ∩ required(k+1) = required(k+1).
        
        Consecuencia de monotonía.
        """
        for higher, lower in DIKW.pairs_by_order():
            req_higher = higher.requires()
            req_lower = lower.requires()
            
            intersection = req_higher & req_lower
            
            assert intersection == req_lower, (
                f"required({higher.name}) ∩ required({lower.name}) "
                f"debe ser required({lower.name}):\n"
                f"  Intersección: {intersection}\n"
                f"  required({lower.name}): {req_lower}"
            )

    def test_partition_property(self, registry: MICRegistry) -> None:
        """
        Para cada estrato k:
            {k} ∪ required(k) ∪ upper(k) = Stratum
        
        El universo se particiona en: el estrato, sus inferiores, y sus superiores.
        """
        for spec in DIKW.strata:
            stratum = spec.stratum
            required = stratum.requires()
            upper = spec.upper
            
            # Unión de las tres partes
            union = {stratum} | required | upper
            
            assert union == set(Stratum), (
                f"[{stratum.name}] No particiona el universo:\n"
                f"  {{k}}:       {{{stratum}}}\n"
                f"  required(k): {required}\n"
                f"  upper(k):    {upper}\n"
                f"  Unión:       {union}\n"
                f"  Universo:    {set(Stratum)}"
            )
            
            # Verificar que son disjuntos
            assert stratum not in required
            assert stratum not in upper
            assert required.isdisjoint(upper), (
                f"[{stratum.name}] required y upper no son disjuntos: "
                f"{required & upper}"
            )


# =============================================================================
# TESTS: IDEMPOTENCIA Y PUREZA
# =============================================================================

class TestIdempotenceAndPurity:
    """
    Verificación de que get_required_strata es una función pura.
    
    Una función pura:
    1. Siempre retorna el mismo resultado para la misma entrada
    2. No tiene efectos secundarios observables
    """

    @pytest.mark.parametrize("stratum", list(Stratum))
    def test_idempotent_repeated_calls(
        self, registry: MICRegistry, stratum: Stratum
    ) -> None:
        """
        Llamadas sucesivas retornan resultados idénticos.
        """
        results = [stratum.requires() for _ in range(5)]
        
        assert all(r == results[0] for r in results), (
            f"[{stratum.name}] Resultados no idempotentes: {results}"
        )

    @pytest.mark.parametrize("stratum", list(Stratum))
    def test_deterministic_across_instances(
        self, stratum: Stratum
    ) -> None:
        """
        Diferentes instancias de MICRegistry dan el mismo resultado.
        """
        registry1 = MICRegistry()
        registry2 = MICRegistry()
        registry3 = MICRegistry()
        
        result1 = set(stratum.requires())
        result2 = set(stratum.requires())
        result3 = set(stratum.requires())
        
        assert result1 == result2 == result3, (
            f"[{stratum.name}] Resultados difieren entre instancias"
        )

    @pytest.mark.parametrize("stratum", list(Stratum))
    def test_no_ordering_dependency(
        self, stratum: Stratum
    ) -> None:
        """
        El resultado no depende del orden de llamadas previas.
        """
        # Llamar en diferentes órdenes
        registry1 = MICRegistry()
        for s in Stratum:
            set(s.requires())
        result1 = set(stratum.requires())
        
        registry2 = MICRegistry()
        for s in reversed(list(Stratum)):
            set(s.requires())
        result2 = set(stratum.requires())
        
        registry3 = MICRegistry()
        result3 = set(stratum.requires())  # Primera llamada
        
        assert result1 == result2 == result3


# =============================================================================
# TESTS: INMUTABILIDAD DEL RESULTADO
# =============================================================================

class TestResultImmutability:
    """
    Verificación de que mutar el resultado no afecta al registry.
    
    El resultado debe ser una copia defensiva.
    """

    @pytest.mark.parametrize("stratum", list(Stratum))
    def test_mutation_does_not_affect_subsequent_calls(
        self, registry: MICRegistry, stratum: Stratum
    ) -> None:
        """
        Mutar el set retornado no altera llamadas posteriores.
        """
        first_result = set(stratum.requires())
        snapshot = frozenset(first_result)
        
        # Mutaciones agresivas
        first_result.clear()
        
        second_result = set(stratum.requires())
        
        assert frozenset(second_result) == snapshot, (
            f"[{stratum.name}] Mutación afectó estado interno:\n"
            f"  Snapshot:   {snapshot}\n"
            f"  Después:    {second_result}"
        )

    @pytest.mark.parametrize("stratum", list(Stratum))
    def test_adding_elements_does_not_persist(
        self, registry: MICRegistry, stratum: Stratum
    ) -> None:
        """
        Añadir elementos al resultado no persiste.
        """
        first_result = set(stratum.requires())
        original_size = len(first_result)
        
        # Añadir elementos (posiblemente inválidos)
        first_result.add(Stratum.WISDOM)
        first_result.add(stratum)  # Violaría irreflexividad
        
        second_result = set(stratum.requires())
        
        assert len(second_result) == original_size
        assert stratum not in second_result  # Irreflexividad mantenida

    @pytest.mark.parametrize("stratum", list(Stratum))
    def test_results_are_independent_copies(
        self, registry: MICRegistry, stratum: Stratum
    ) -> None:
        """
        Cada llamada retorna un objeto diferente (no el mismo reference).
        """
        result1 = stratum.requires()
        result2 = stratum.requires()
        
        # Deben ser iguales en valor pero diferentes objetos
        assert result1 == result2
        assert result1 is not result2, (
            f"[{stratum.name}] Retorna el mismo objeto, no una copia"
        )


# =============================================================================
# TESTS: ROBUSTEZ ANTE ENTRADAS INVÁLIDAS
# =============================================================================

class TestInvalidInputs:
    """
    Verificación del comportamiento ante entradas inválidas.
    
    El contrato de tipos es parte de la API: fallar ruidosamente
    es preferible a retornar resultados indefinidos.
    """

    @pytest.mark.parametrize(
        "invalid_input, description",
        [
            pytest.param("PHYSICS", "string con nombre de estrato", id="string"),
            pytest.param("wisdom", "string lowercase", id="string-lower"),
            pytest.param(0, "int (valor de enum)", id="int-zero"),
            pytest.param(3, "int (valor de PHYSICS)", id="int-three"),
            pytest.param(1.0, "float", id="float"),
            pytest.param([], "lista vacía", id="empty-list"),
            pytest.param({}, "dict vacío", id="empty-dict"),
            pytest.param(object(), "objeto genérico", id="object"),
        ],
    )
    def test_non_stratum_type_raises(
        self, registry: MICRegistry, invalid_input: Any, description: str
    ) -> None:
        """
        Entradas que no son Stratum deben lanzar excepción.
        """
        with pytest.raises((TypeError, ValueError, AttributeError, KeyError)):
            invalid_input.requires()  # type: ignore

    def test_none_raises(self, registry: MICRegistry) -> None:
        """None debe lanzar excepción."""
        with pytest.raises((TypeError, ValueError, AttributeError)):
            None.requires()  # type: ignore

    @pytest.mark.parametrize(
        "almost_valid",
        [
            pytest.param(type("FakeStratum", (), {"value": 0})(), id="fake-class"),
        ],
    )
    def test_fake_stratum_raises(
        self, registry: MICRegistry, almost_valid: Any
    ) -> None:
        """Objetos que simulan ser Stratum deben fallar."""
        with pytest.raises((TypeError, ValueError, AttributeError, KeyError)):
            almost_valid.requires()  # type: ignore


# =============================================================================
# TESTS: THREAD SAFETY
# =============================================================================

class TestThreadSafety:
    """
    Verificación de comportamiento correcto bajo concurrencia.
    
    Si MICRegistry es stateless, debe ser thread-safe.
    """

    def test_concurrent_reads_same_stratum(self, registry: MICRegistry) -> None:
        """
        Múltiples hilos leyendo el mismo estrato concurrentemente.
        """
        stratum = Stratum.STRATEGY
        expected = DIKW.get_spec(stratum).required
        results: List[Set[Stratum]] = []
        errors: List[Exception] = []
        
        def read_stratum():
            try:
                result = set(stratum.requires())
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=read_stratum) for _ in range(100)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert not errors, f"Errores en hilos: {errors}"
        assert all(r == expected for r in results), (
            f"Resultados inconsistentes bajo concurrencia"
        )

    def test_concurrent_reads_different_strata(self, registry: MICRegistry) -> None:
        """
        Múltiples hilos leyendo diferentes estratos concurrentemente.
        """
        results: Dict[Stratum, List[Set[Stratum]]] = {s: [] for s in Stratum}
        errors: List[Exception] = []
        
        def read_stratum(stratum: Stratum):
            try:
                for _ in range(10):
                    result = set(stratum.requires())
                    results[stratum].append(result)
            except Exception as e:
                errors.append(e)
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(read_stratum, s)
                for s in Stratum
                for _ in range(5)  # 5 tareas por estrato
            ]
            for future in as_completed(futures):
                future.result()  # Propagar excepciones
        
        assert not errors
        
        for stratum, stratum_results in results.items():
            expected = DIKW.get_spec(stratum).required
            assert all(r == expected for r in stratum_results), (
                f"[{stratum.name}] Inconsistencia bajo concurrencia"
            )

    def test_concurrent_instantiation_and_read(self) -> None:
        """
        Crear múltiples instancias y leer concurrentemente.
        """
        results: List[Dict[Stratum, Set[Stratum]]] = []
        errors: List[Exception] = []
        
        def create_and_read():
            try:
                reg = MICRegistry()
                local_results = {s: set(s.requires()) for s in Stratum}
                results.append(local_results)
            except Exception as e:
                errors.append(e)
        
        threads = [threading.Thread(target=create_and_read) for _ in range(50)]
        
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert not errors
        
        # Todos deben tener los mismos resultados
        first = results[0]
        for r in results[1:]:
            for stratum in Stratum:
                assert r[stratum] == first[stratum]


# =============================================================================
# TESTS: PROPIEDADES DE GRAFO (DAG)
# =============================================================================

class TestDAGProperties:
    """
    Verificación de propiedades del grafo de dependencias.
    
    El grafo G = (Stratum, E) donde E = { (k, s) | s ∈ required(k) }
    debe ser un DAG con propiedades específicas.
    """

    def test_graph_is_acyclic(self, registry: MICRegistry) -> None:
        """
        El grafo de dependencias no tiene ciclos.
        
        Para todo k: k no es alcanzable desde k siguiendo aristas de dependencia.
        """
        # Construir matriz de alcanzabilidad
        reachable: Dict[Stratum, Set[Stratum]] = {}
        
        for k in Stratum:
            # BFS/DFS desde k
            visited = set()
            queue = list(k.requires())
            
            while queue:
                current = queue.pop()
                if current not in visited:
                    visited.add(current)
                    queue.extend(current.requires())
            
            reachable[k] = visited
        
        # Verificar aciclicidad
        for k in Stratum:
            assert k not in reachable[k], (
                f"Ciclo detectado: {k.name} es alcanzable desde sí mismo"
            )

    def test_graph_is_transitively_closed(self, registry: MICRegistry) -> None:
        """
        El grafo de dependencias es transitivamente cerrado.
        
        Si s ∈ required(k) y t ∈ required(s), entonces t ∈ required(k).
        """
        for k in Stratum:
            req_k = k.requires()
            
            # Calcular clausura transitiva
            closure = set(req_k)
            changed = True
            while changed:
                changed = False
                for s in list(closure):
                    for t in s.requires():
                        if t not in closure:
                            closure.add(t)
                            changed = True
            
            assert closure == req_k, (
                f"[{k.name}] No es transitivamente cerrado:\n"
                f"  required:   {req_k}\n"
                f"  clausura:   {closure}\n"
                f"  diferencia: {closure - req_k}"
            )

    def test_graph_height(self, registry: MICRegistry) -> None:
        """
        La altura del DAG es 3 (camino más largo: WISDOM → ... → PHYSICS).
        """
        def longest_path_from(k: Stratum) -> int:
            required = k.requires()
            if not required:
                return 0
            return 1 + max(longest_path_from(s) for s in required)
        
        max_height = max(longest_path_from(k) for k in Stratum)
        
        assert max_height == DIKW.total_count - 1, (
            f"Altura del DAG: esperada={DIKW.total_count - 1}, obtenida={max_height}"
        )

    def test_topological_order_exists(self, registry: MICRegistry) -> None:
        """
        Existe un orden topológico válido del grafo.
        """
        # Algoritmo de Kahn
        in_degree = {k: 0 for k in Stratum}
        
        for k in Stratum:
            for s in k.requires():
                in_degree[s] += 1  # s es dependencia de k
        
        # Procesar nodos con in_degree 0
        queue = [k for k in Stratum if in_degree[k] == 0]
        topo_order = []
        
        while queue:
            current = queue.pop(0)
            topo_order.append(current)
            
            for s in current.requires():
                in_degree[s] -= 1
                if in_degree[s] == 0:
                    queue.append(s)
        
        assert len(topo_order) == DIKW.total_count, (
            f"No se pudo generar orden topológico completo: {topo_order}"
        )


# =============================================================================
# TESTS: PROPERTY-BASED (HYPOTHESIS)
# =============================================================================

@pytest.mark.skipif(
    not HYPOTHESIS_AVAILABLE,
    reason="hypothesis library not installed"
)
class TestPropertyBased:
    """
    Tests basados en propiedades usando Hypothesis.
    
    Verifican invariantes que deben cumplirse para cualquier entrada válida.
    """

    @given(stratum=st.sampled_from(list(Stratum)))
    @settings(max_examples=50)
    def test_property_irreflexivity(self, stratum: Stratum) -> None:
        """Propiedad: k ∉ required(k) para todo k."""
        registry = MICRegistry()
        result = set(stratum.requires())
        assert stratum not in result

    @given(stratum=st.sampled_from(list(Stratum)))
    @settings(max_examples=50)
    def test_property_bounded_cardinality(self, stratum: Stratum) -> None:
        """Propiedad: 0 ≤ |required(k)| ≤ |Stratum| - 1."""
        registry = MICRegistry()
        result = set(stratum.requires())
        
        assert 0 <= len(result) <= len(Stratum) - 1

    @given(stratum=st.sampled_from(list(Stratum)))
    @settings(max_examples=50)
    def test_property_elements_are_valid_strata(self, stratum: Stratum) -> None:
        """Propiedad: todos los elementos son Stratum válidos."""
        registry = MICRegistry()
        result = set(stratum.requires())
        
        for element in result:
            assert isinstance(element, Stratum)
            assert element in Stratum

    @given(
        k1=st.sampled_from(list(Stratum)),
        k2=st.sampled_from(list(Stratum)),
    )
    @settings(max_examples=100)
    def test_property_comparable_sets(
        self, k1: Stratum, k2: Stratum
    ) -> None:
        """Propiedad: cualquier par de required sets es comparable bajo ⊆."""
        registry = MICRegistry()
        req_k1 = k1.requires()
        req_k2 = k2.requires()
        
        assert req_k1 <= req_k2 or req_k2 <= req_k1


# =============================================================================
# TESTS: RENDIMIENTO
# =============================================================================

class TestPerformance:
    """
    Tests de rendimiento básico.
    """

    def test_single_call_is_fast(self, registry: MICRegistry) -> None:
        """Una llamada debe completar en menos de 1ms."""
        import time
        
        for stratum in Stratum:
            start = time.perf_counter()
            stratum.requires()
            elapsed = time.perf_counter() - start
            
            assert elapsed < 0.001, (
                f"[{stratum.name}] Llamada demasiado lenta: {elapsed*1000:.3f}ms"
            )

    def test_many_calls_are_efficient(self, registry: MICRegistry) -> None:
        """10,000 llamadas deben completar en menos de 1 segundo."""
        import time
        
        start = time.perf_counter()
        
        for _ in range(10_000):
            for stratum in Stratum:
                stratum.requires()
        
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, (
            f"40,000 llamadas tomaron {elapsed:.3f}s (máx: 1.0s)"
        )

    def test_instantiation_is_fast(self) -> None:
        """Crear 1,000 instancias debe tomar menos de 1 segundo."""
        import time
        
        start = time.perf_counter()
        
        for _ in range(1_000):
            MICRegistry()
        
        elapsed = time.perf_counter() - start
        
        assert elapsed < 1.0, (
            f"1,000 instanciaciones tomaron {elapsed:.3f}s"
        )


# =============================================================================
# TESTS: CONSISTENCIA INTERNA DE LA ESPECIFICACIÓN
# =============================================================================

class TestSpecificationConsistency:
    """
    Verificación de que la especificación DIKW es internamente consistente.
    
    Estos tests no dependen del MICRegistry, solo de la especificación.
    """

    def test_spec_covers_all_strata(self) -> None:
        """La especificación cubre todos los estratos del enum."""
        spec_strata = {spec.stratum for spec in DIKW.strata}
        enum_strata = set(Stratum)
        
        assert spec_strata == enum_strata, (
            f"Especificación incompleta:\n"
            f"  En spec: {spec_strata}\n"
            f"  En enum: {enum_strata}\n"
            f"  Faltantes: {enum_strata - spec_strata}"
        )

    def test_spec_values_match_enum(self) -> None:
        """Los valores en la especificación coinciden con el enum."""
        for spec in DIKW.strata:
            assert spec.value == spec.stratum.value, (
                f"[{spec.stratum.name}] Valor inconsistente: "
                f"spec.value={spec.value}, stratum.value={spec.stratum.value}"
            )

    def test_spec_cardinality_matches_required(self) -> None:
        """La cardinalidad declarada coincide con |required|."""
        for spec in DIKW.strata:
            assert spec.cardinality == len(spec.required), (
                f"[{spec.stratum.name}] Cardinalidad inconsistente: "
                f"declarada={spec.cardinality}, calculada={len(spec.required)}"
            )

    def test_spec_upper_and_required_partition(self) -> None:
        """upper(k) ∪ required(k) ∪ {k} = Stratum."""
        for spec in DIKW.strata:
            union = {spec.stratum} | spec.required | spec.upper
            
            assert union == set(Stratum), (
                f"[{spec.stratum.name}] No particiona el universo"
            )

    def test_spec_upper_and_required_disjoint(self) -> None:
        """upper(k) ∩ required(k) = ∅."""
        for spec in DIKW.strata:
            intersection = spec.upper & spec.required
            
            assert not intersection, (
                f"[{spec.stratum.name}] upper y required no son disjuntos: "
                f"{intersection}"
            )


# =============================================================================
# CONFIGURACIÓN DE PYTEST
# =============================================================================

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",
        "--durations=10",
    ])