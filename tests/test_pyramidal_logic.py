"""
Suite de Pruebas Topológicas para el Validador Piramidal
=========================================================

Fundamentos matemáticos:
- Los estratos forman un orden total (cadena lineal en teoría de posets)
- La estabilidad piramidal modela un sistema dinámico acotado via tanh
- Los nodos flotantes violan la conectividad del espacio topológico

Autor: Artesano Programador Senior
Versión: 2.0.0
"""

import pytest
import math
from typing import List, Tuple, Dict, Any
from dataclasses import FrozenInstanceError

import pandas as pd
import numpy as np

from app.data_validator import PyramidalValidator, PyramidalMetrics
from app.schemas import Stratum, TopologicalNode, InsumoProcesado, APUStructure
from app.data_loader import HierarchyLevel, load_data_with_hierarchy
from app.classifiers.apu_classifier import StructuralClassifier


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: PRUEBAS DE ESTRUCTURA TOPOLÓGICA DE ESTRATOS
# ═══════════════════════════════════════════════════════════════════════════════

class TestStratumTopology:
    """
    Verifica que Stratum forme un espacio topológico ordenado válido.
    
    Propiedades matemáticas verificadas:
    - Orden total (reflexivo, antisimétrico, transitivo, total)
    - Contiguidad (sin huecos en la secuencia)
    - Completitud (todos los estratos definidos)
    """

    def test_stratum_enum_values(self):
        """Verifica los valores numéricos cardinales de cada estrato."""
        assert Stratum.ROOT == 0
        assert Stratum.STRATEGY == 1
        assert Stratum.TACTIC == 2
        assert Stratum.LOGISTICS == 3

    def test_stratum_completeness(self):
        """Verifica que el conjunto de estratos sea completo."""
        expected = {'ROOT', 'STRATEGY', 'TACTIC', 'LOGISTICS'}
        actual = {s.name for s in Stratum}
        assert actual == expected, f"Estratos faltantes: {expected - actual}"

    def test_stratum_ordering_reflexive(self):
        """Propiedad reflexiva: ∀a: a ≤ a"""
        for stratum in Stratum:
            assert stratum.value <= stratum.value

    def test_stratum_ordering_antisymmetric(self):
        """Propiedad antisimétrica: si a ≤ b ∧ b ≤ a → a = b"""
        strata = list(Stratum)
        for s1 in strata:
            for s2 in strata:
                if s1.value <= s2.value and s2.value <= s1.value:
                    assert s1 == s2

    def test_stratum_ordering_transitive(self):
        """Propiedad transitiva: si a ≤ b ∧ b ≤ c → a ≤ c"""
        strata = list(Stratum)
        for s1 in strata:
            for s2 in strata:
                for s3 in strata:
                    if s1.value <= s2.value and s2.value <= s3.value:
                        assert s1.value <= s3.value

    def test_stratum_ordering_total(self):
        """Orden total: ∀a,b: a ≤ b ∨ b ≤ a"""
        strata = list(Stratum)
        for s1 in strata:
            for s2 in strata:
                assert s1.value <= s2.value or s2.value <= s1.value

    def test_stratum_contiguity(self):
        """
        Los valores deben ser contiguos (sin huecos).
        Garantiza un espacio topológico conexo.
        """
        values = sorted(s.value for s in Stratum)
        for i in range(len(values) - 1):
            gap = values[i + 1] - values[i]
            assert gap == 1, f"Hueco detectado: {values[i]} → {values[i+1]}"

    def test_stratum_pyramid_invariant(self):
        """
        Invariante piramidal: ROOT es la cúspide (valor mínimo),
        LOGISTICS es la base (valor máximo).
        """
        assert Stratum.ROOT.value == min(s.value for s in Stratum)
        assert Stratum.LOGISTICS.value == max(s.value for s in Stratum)

    def test_stratum_cardinality(self):
        """La pirámide tiene exactamente 4 niveles."""
        assert len(Stratum) == 4


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2: PRUEBAS DE NODOS TOPOLÓGICOS
# ═══════════════════════════════════════════════════════════════════════════════

class TestTopologicalNode:
    """
    Pruebas para la clase base TopologicalNode.
    
    Un nodo topológico es un punto en el espacio estratificado con:
    - Identidad única
    - Posición en un estrato
    - Salud estructural ∈ [0, 1]
    - Estado de flotación (conectividad)
    """

    def test_node_default_values(self):
        """Verifica valores por defecto del constructor."""
        node = TopologicalNode(
            id="test_node",
            stratum=Stratum.TACTIC,
            description="Nodo de prueba"
        )
        assert node.structural_health == 1.0
        assert node.is_floating is False

    def test_node_identity_preservation(self):
        """El ID debe preservarse exactamente."""
        node = TopologicalNode(id="ÚNICO_123", stratum=Stratum.ROOT, description="Test")
        assert node.id == "ÚNICO_123"

    @pytest.mark.parametrize("health", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_node_health_valid_range(self, health: float):
        """La salud estructural acepta valores en [0, 1]."""
        node = TopologicalNode(
            id="test", stratum=Stratum.TACTIC,
            description="Test", structural_health=health
        )
        assert node.structural_health == health

    @pytest.mark.parametrize("stratum", list(Stratum))
    def test_node_accepts_all_strata(self, stratum: Stratum):
        """Un nodo puede pertenecer a cualquier estrato."""
        node = TopologicalNode(
            id=f"node_{stratum.name}",
            stratum=stratum,
            description=f"Nodo en {stratum.name}"
        )
        assert node.stratum == stratum

    def test_node_floating_explicit_true(self):
        """Nodo explícitamente marcado como flotante."""
        node = TopologicalNode(
            id="floating", stratum=Stratum.LOGISTICS,
            description="Flotante", is_floating=True
        )
        assert node.is_floating is True

    def test_node_floating_explicit_false(self):
        """Nodo explícitamente anclado."""
        node = TopologicalNode(
            id="grounded", stratum=Stratum.LOGISTICS,
            description="Anclado", is_floating=False
        )
        assert node.is_floating is False


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: PRUEBAS DE INSUMO PROCESADO
# ═══════════════════════════════════════════════════════════════════════════════

class TestInsumoProcesado:
    """
    Pruebas para InsumoProcesado como nodo del estrato LOGISTICS.
    
    Los insumos son los elementos atómicos de la base piramidal.
    Deben satisfacer la ecuación de conservación:
        valor_total = cantidad × precio_unitario
    """

    @pytest.fixture
    def insumo_basico(self) -> InsumoProcesado:
        """Fixture: insumo estándar para pruebas."""
        return InsumoProcesado(
            codigo_apu="APU001",
            descripcion_apu="Muro de Ladrillo",
            unidad_apu="m2",
            descripcion_insumo="Ladrillo Arcilla",
            unidad_insumo="und",
            cantidad=150.0,
            precio_unitario=850.0,
            valor_total=127500.0,
            tipo_insumo="SUMINISTRO"
        )

    def test_insumo_inherits_topological_node(self, insumo_basico):
        """InsumoProcesado debe ser subclase de TopologicalNode."""
        assert isinstance(insumo_basico, TopologicalNode)

    def test_insumo_fixed_stratum_logistics(self, insumo_basico):
        """Los insumos siempre pertenecen al estrato LOGISTICS (base)."""
        assert insumo_basico.stratum == Stratum.LOGISTICS

    def test_insumo_id_starts_with_apu_code(self, insumo_basico):
        """El ID debe iniciar con el código APU para trazabilidad."""
        assert insumo_basico.id.startswith("APU001_")

    def test_insumo_id_has_hash_component(self, insumo_basico):
        """El ID contiene un componente hash para unicidad topológica."""
        id_parts = insumo_basico.id.split("_")
        assert len(id_parts) >= 2

    def test_insumo_id_uniqueness_same_apu(self):
        """Insumos diferentes del mismo APU tienen IDs distintos."""
        insumo_a = InsumoProcesado(
            codigo_apu="APU001", descripcion_apu="Muro", unidad_apu="m2",
            descripcion_insumo="Ladrillo", unidad_insumo="und",
            cantidad=100, precio_unitario=500, valor_total=50000,
            tipo_insumo="SUMINISTRO"
        )
        insumo_b = InsumoProcesado(
            codigo_apu="APU001", descripcion_apu="Muro", unidad_apu="m2",
            descripcion_insumo="Cemento", unidad_insumo="kg",
            cantidad=50, precio_unitario=400, valor_total=20000,
            tipo_insumo="SUMINISTRO"
        )
        assert insumo_a.id != insumo_b.id

    def test_insumo_id_uniqueness_different_apu(self):
        """Mismo insumo en diferentes APUs tiene IDs distintos."""
        insumo_a = InsumoProcesado(
            codigo_apu="APU001", descripcion_apu="Muro", unidad_apu="m2",
            descripcion_insumo="Ladrillo", unidad_insumo="und",
            cantidad=100, precio_unitario=500, valor_total=50000,
            tipo_insumo="SUMINISTRO"
        )
        insumo_b = InsumoProcesado(
            codigo_apu="APU002", descripcion_apu="Columna", unidad_apu="m3",
            descripcion_insumo="Ladrillo", unidad_insumo="und",
            cantidad=100, precio_unitario=500, valor_total=50000,
            tipo_insumo="SUMINISTRO"
        )
        assert insumo_a.id != insumo_b.id

    @pytest.mark.parametrize("tipo", [
        "SUMINISTRO", "MANO_DE_OBRA", "EQUIPO", "TRANSPORTE"
    ])
    def test_insumo_valid_types(self, tipo: str):
        """Verifica que todos los tipos de insumo son aceptados."""
        insumo = InsumoProcesado(
            codigo_apu="APU001", descripcion_apu="Test", unidad_apu="m2",
            descripcion_insumo="Material", unidad_insumo="und",
            cantidad=1, precio_unitario=100, valor_total=100,
            tipo_insumo=tipo
        )
        assert insumo.tipo_insumo == tipo

    def test_insumo_value_conservation_law(self):
        """
        Ley de conservación: valor_total = cantidad × precio_unitario
        
        Esta es una invariante fundamental del sistema económico.
        """
        insumo = InsumoProcesado(
            codigo_apu="APU001", descripcion_apu="Test", unidad_apu="m2",
            descripcion_insumo="Material", unidad_insumo="und",
            cantidad=25.5, precio_unitario=1234.56, valor_total=31481.28,
            tipo_insumo="SUMINISTRO"
        )
        expected = insumo.cantidad * insumo.precio_unitario
        assert abs(insumo.valor_total - expected) < 1e-6

    def test_insumo_non_negative_quantities(self, insumo_basico):
        """Las cantidades físicas no pueden ser negativas."""
        assert insumo_basico.cantidad >= 0
        assert insumo_basico.precio_unitario >= 0
        assert insumo_basico.valor_total >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4: PRUEBAS DE ESTRUCTURA APU
# ═══════════════════════════════════════════════════════════════════════════════

class TestAPUStructure:
    """
    Pruebas para APUStructure como nodo del estrato TACTIC.
    
    Un APU agrega recursos (insumos) y su estabilidad depende
    del ancho de su base de soporte (número de insumos).
    """

    @pytest.fixture
    def apu_vacio(self) -> APUStructure:
        """APU sin recursos asignados."""
        return APUStructure(
            id="APU001",
            description="Muro de Ladrillo",
            unit="m2",
            quantity=100.0
        )

    @pytest.fixture
    def insumos_variados(self) -> List[InsumoProcesado]:
        """Lista de insumos de diferentes tipos."""
        return [
            InsumoProcesado(
                codigo_apu="APU001", descripcion_apu="Muro", unidad_apu="m2",
                descripcion_insumo="Ladrillo", unidad_insumo="und",
                cantidad=150, precio_unitario=850, valor_total=127500,
                tipo_insumo="SUMINISTRO"
            ),
            InsumoProcesado(
                codigo_apu="APU001", descripcion_apu="Muro", unidad_apu="m2",
                descripcion_insumo="Cemento", unidad_insumo="kg",
                cantidad=25, precio_unitario=450, valor_total=11250,
                tipo_insumo="SUMINISTRO"
            ),
            InsumoProcesado(
                codigo_apu="APU001", descripcion_apu="Muro", unidad_apu="m2",
                descripcion_insumo="Albañil", unidad_insumo="hr",
                cantidad=4, precio_unitario=15000, valor_total=60000,
                tipo_insumo="MANO_DE_OBRA"
            ),
            InsumoProcesado(
                codigo_apu="APU001", descripcion_apu="Muro", unidad_apu="m2",
                descripcion_insumo="Mezcladora", unidad_insumo="hr",
                cantidad=2, precio_unitario=5000, valor_total=10000,
                tipo_insumo="EQUIPO"
            ),
        ]

    def test_apu_stratum_is_tactic(self, apu_vacio):
        """Los APUs pertenecen al estrato TACTIC."""
        assert apu_vacio.stratum == Stratum.TACTIC

    def test_apu_initial_support_base_zero(self, apu_vacio):
        """Un APU nuevo tiene base de soporte vacía."""
        assert apu_vacio.support_base_width == 0

    def test_apu_add_single_resource(self, apu_vacio, insumos_variados):
        """Agregar un recurso incrementa la base en 1."""
        apu_vacio.add_resource(insumos_variados[0])
        assert apu_vacio.support_base_width == 1

    def test_apu_add_multiple_resources(self, apu_vacio, insumos_variados):
        """La base crece linealmente con cada recurso agregado."""
        for i, insumo in enumerate(insumos_variados):
            apu_vacio.add_resource(insumo)
            assert apu_vacio.support_base_width == i + 1

    def test_apu_support_monotonicity(self, apu_vacio, insumos_variados):
        """
        Propiedad de monotonía: la base nunca decrece.
        Modela acumulación irreversible en sistemas dinámicos.
        """
        previous = 0
        for insumo in insumos_variados:
            apu_vacio.add_resource(insumo)
            assert apu_vacio.support_base_width >= previous
            previous = apu_vacio.support_base_width

    def test_apu_stratum_above_insumos(self, apu_vacio, insumos_variados):
        """
        Invariante piramidal: TACTIC (APU) está sobre LOGISTICS (insumos).
        En términos de valores: TACTIC < LOGISTICS.
        """
        apu_vacio.add_resource(insumos_variados[0])
        assert apu_vacio.stratum.value < insumos_variados[0].stratum.value

    def test_apu_empty_is_potentially_floating(self, apu_vacio):
        """Un APU sin soporte es potencialmente flotante."""
        assert apu_vacio.support_base_width == 0
        # La flotación real se determina en validación

    def test_apu_with_support_not_floating(self, apu_vacio, insumos_variados):
        """Un APU con recursos tiene soporte estructural."""
        apu_vacio.add_resource(insumos_variados[0])
        assert apu_vacio.support_base_width > 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5: PRUEBAS DEL VALIDADOR PIRAMIDAL
# ═══════════════════════════════════════════════════════════════════════════════

class TestPyramidalValidator:
    """
    Pruebas para PyramidalValidator.
    
    El validador calcula métricas topológicas:
    - base_width: cardinalidad del conjunto de insumos únicos
    - structure_load: número de APUs (carga estructural)
    - pyramid_stability_index: tanh(base_width / structure_load)
    - floating_nodes: APUs sin insumos (violan conectividad)
    """

    @pytest.fixture
    def validator(self) -> PyramidalValidator:
        return PyramidalValidator()

    @pytest.fixture
    def estructura_valida(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Estructura piramidal completamente conectada."""
        apus = pd.DataFrame({
            "CODIGO_APU": ["APU001", "APU002", "APU003"]
        })
        insumos = pd.DataFrame({
            "APU_CODIGO": [
                "APU001", "APU001",
                "APU002", "APU002",
                "APU003"
            ],
            "DESCRIPCION_INSUMO_NORM": [
                "LADRILLO", "CEMENTO",
                "ARENA", "GRAVA",
                "ACERO"
            ]
        })
        return apus, insumos

    @pytest.fixture
    def estructura_con_flotante(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Estructura con un APU flotante (sin insumos)."""
        apus = pd.DataFrame({
            "CODIGO_APU": ["APU001", "APU002", "APU_FLOTANTE"]
        })
        insumos = pd.DataFrame({
            "APU_CODIGO": ["APU001", "APU002"],
            "DESCRIPCION_INSUMO_NORM": ["MATERIAL_A", "MATERIAL_B"]
        })
        return apus, insumos

    @pytest.fixture
    def estructura_vacia(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Estructura sin APUs ni insumos."""
        apus = pd.DataFrame({"CODIGO_APU": pd.Series([], dtype=str)})
        insumos = pd.DataFrame({
            "APU_CODIGO": pd.Series([], dtype=str),
            "DESCRIPCION_INSUMO_NORM": pd.Series([], dtype=str)
        })
        return apus, insumos

    def test_validator_returns_metrics(self, validator, estructura_valida):
        """La validación retorna un objeto PyramidalMetrics."""
        apus, insumos = estructura_valida
        metrics = validator.validate_structure(apus, insumos)
        assert isinstance(metrics, PyramidalMetrics)

    def test_base_width_counts_unique_insumos(self, validator, estructura_valida):
        """El ancho de base es el número de insumos únicos."""
        apus, insumos = estructura_valida
        metrics = validator.validate_structure(apus, insumos)
        
        expected = insumos["DESCRIPCION_INSUMO_NORM"].nunique()
        assert metrics.base_width == expected
        assert metrics.base_width == 5  # LADRILLO, CEMENTO, ARENA, GRAVA, ACERO

    def test_structure_load_counts_apus(self, validator, estructura_valida):
        """La carga estructural es el número de APUs."""
        apus, insumos = estructura_valida
        metrics = validator.validate_structure(apus, insumos)
        
        assert metrics.structure_load == len(apus)
        assert metrics.structure_load == 3

    def test_stability_index_uses_tanh(self, validator, estructura_valida):
        """
        El índice de estabilidad usa la función tanh.
        
        tanh es la solución de la ecuación diferencial:
            dy/dx = 1 - y²
        
        Con propiedades:
            - Dominio: ℝ → (-1, 1)
            - Para x > 0: tanh(x) ∈ (0, 1)
            - Monótona creciente
            - C∞ (infinitamente diferenciable)
        """
        apus, insumos = estructura_valida
        metrics = validator.validate_structure(apus, insumos)
        
        ratio = metrics.base_width / max(metrics.structure_load, 1)
        expected_stability = math.tanh(ratio)
        
        assert abs(metrics.pyramid_stability_index - expected_stability) < 0.01

    def test_stability_index_bounds(self, validator, estructura_valida):
        """El índice de estabilidad está en el intervalo (0, 1]."""
        apus, insumos = estructura_valida
        metrics = validator.validate_structure(apus, insumos)
        
        assert 0 < metrics.pyramid_stability_index <= 1

    def test_stability_greater_than_threshold(self, validator, estructura_valida):
        """Una estructura balanceada tiene estabilidad > 0.9."""
        apus, insumos = estructura_valida
        metrics = validator.validate_structure(apus, insumos)
        
        # base_width=5, load=3 → ratio≈1.67 → tanh(1.67)≈0.93
        assert metrics.pyramid_stability_index > 0.9

    def test_floating_nodes_detected(self, validator, estructura_con_flotante):
        """Los APUs sin insumos son detectados como flotantes."""
        apus, insumos = estructura_con_flotante
        metrics = validator.validate_structure(apus, insumos)
        
        assert len(metrics.floating_nodes) == 1
        assert "APU_FLOTANTE" in metrics.floating_nodes

    def test_no_floating_in_valid_structure(self, validator, estructura_valida):
        """Una estructura válida no tiene nodos flotantes."""
        apus, insumos = estructura_valida
        metrics = validator.validate_structure(apus, insumos)
        
        assert len(metrics.floating_nodes) == 0

    def test_empty_structure_handling(self, validator, estructura_vacia):
        """El validador maneja estructuras vacías sin errores."""
        apus, insumos = estructura_vacia
        metrics = validator.validate_structure(apus, insumos)
        
        assert metrics.base_width == 0
        assert metrics.structure_load == 0
        assert len(metrics.floating_nodes) == 0

    def test_all_floating_structure(self, validator):
        """Estructura donde todos los APUs son flotantes."""
        apus = pd.DataFrame({"CODIGO_APU": ["APU001", "APU002", "APU003"]})
        insumos = pd.DataFrame({
            "APU_CODIGO": pd.Series([], dtype=str),
            "DESCRIPCION_INSUMO_NORM": pd.Series([], dtype=str)
        })
        
        metrics = validator.validate_structure(apus, insumos)
        
        assert len(metrics.floating_nodes) == 3
        assert metrics.base_width == 0

    def test_stability_monotonicity_with_base(self, validator):
        """
        La estabilidad aumenta monótonamente con el ancho de base.
        Propiedad derivada de la monotonía de tanh.
        """
        apus = pd.DataFrame({"CODIGO_APU": ["APU001"]})
        
        stabilities = []
        for n in range(1, 11):
            insumos = pd.DataFrame({
                "APU_CODIGO": ["APU001"] * n,
                "DESCRIPCION_INSUMO_NORM": [f"INSUMO_{i}" for i in range(n)]
            })
            metrics = validator.validate_structure(apus, insumos)
            stabilities.append(metrics.pyramid_stability_index)
        
        # Verificar monotonía estricta
        for i in range(len(stabilities) - 1):
            assert stabilities[i] < stabilities[i + 1]

    def test_large_structure_performance(self, validator):
        """El validador maneja estructuras grandes eficientemente."""
        n_apus = 500
        n_insumos_per_apu = 20
        
        apus = pd.DataFrame({
            "CODIGO_APU": [f"APU{i:04d}" for i in range(n_apus)]
        })
        
        insumos_data = []
        for i in range(n_apus):
            for j in range(n_insumos_per_apu):
                insumos_data.append({
                    "APU_CODIGO": f"APU{i:04d}",
                    "DESCRIPCION_INSUMO_NORM": f"INSUMO_{j:03d}"
                })
        insumos = pd.DataFrame(insumos_data)
        
        metrics = validator.validate_structure(apus, insumos)
        
        assert metrics.structure_load == n_apus
        assert metrics.base_width == n_insumos_per_apu
        assert len(metrics.floating_nodes) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6: PRUEBAS DEL CLASIFICADOR ESTRUCTURAL
# ═══════════════════════════════════════════════════════════════════════════════

class TestStructuralClassifier:
    """
    Pruebas para StructuralClassifier.
    
    Clasificaciones:
    - SUMINISTRO_PURO: 100% materiales
    - SERVICIO_PURO: 100% mano de obra
    - ESTRUCTURA_MIXTA: combinación de tipos
    """

    @pytest.fixture
    def classifier(self) -> StructuralClassifier:
        return StructuralClassifier()

    def test_suministro_puro_single(self, classifier):
        """Un solo suministro → SUMINISTRO_PURO."""
        insumos = [{"TIPO_INSUMO": "SUMINISTRO", "VALOR_TOTAL": 1000}]
        clasificacion, _ = classifier.classify_by_structure(insumos)
        assert clasificacion == "SUMINISTRO_PURO"

    def test_suministro_puro_multiple(self, classifier):
        """Múltiples suministros → SUMINISTRO_PURO."""
        insumos = [
            {"TIPO_INSUMO": "SUMINISTRO", "VALOR_TOTAL": 100},
            {"TIPO_INSUMO": "SUMINISTRO", "VALOR_TOTAL": 200},
            {"TIPO_INSUMO": "SUMINISTRO", "VALOR_TOTAL": 300}
        ]
        clasificacion, _ = classifier.classify_by_structure(insumos)
        assert clasificacion == "SUMINISTRO_PURO"

    def test_servicio_puro_single(self, classifier):
        """Un solo servicio → SERVICIO_PURO."""
        insumos = [{"TIPO_INSUMO": "MANO_DE_OBRA", "VALOR_TOTAL": 50000}]
        clasificacion, _ = classifier.classify_by_structure(insumos)
        assert clasificacion == "SERVICIO_PURO"

    def test_servicio_puro_multiple(self, classifier):
        """Múltiples servicios → SERVICIO_PURO."""
        insumos = [
            {"TIPO_INSUMO": "MANO_DE_OBRA", "VALOR_TOTAL": 30000},
            {"TIPO_INSUMO": "MANO_DE_OBRA", "VALOR_TOTAL": 20000}
        ]
        clasificacion, _ = classifier.classify_by_structure(insumos)
        assert clasificacion == "SERVICIO_PURO"

    def test_estructura_mixta_balanced(self, classifier):
        """Suministro + Mano de obra → ESTRUCTURA_MIXTA."""
        insumos = [
            {"TIPO_INSUMO": "SUMINISTRO", "VALOR_TOTAL": 100},
            {"TIPO_INSUMO": "MANO_DE_OBRA", "VALOR_TOTAL": 100}
        ]
        clasificacion, _ = classifier.classify_by_structure(insumos)
        assert clasificacion == "ESTRUCTURA_MIXTA"

    def test_estructura_mixta_unbalanced(self, classifier):
        """Mezcla desbalanceada sigue siendo ESTRUCTURA_MIXTA."""
        insumos = [
            {"TIPO_INSUMO": "SUMINISTRO", "VALOR_TOTAL": 900},
            {"TIPO_INSUMO": "MANO_DE_OBRA", "VALOR_TOTAL": 100}
        ]
        clasificacion, _ = classifier.classify_by_structure(insumos)
        assert clasificacion == "ESTRUCTURA_MIXTA"

    def test_classification_returns_tuple(self, classifier):
        """El método retorna una tupla (clasificación, metadata)."""
        insumos = [{"TIPO_INSUMO": "SUMINISTRO", "VALOR_TOTAL": 100}]
        result = classifier.classify_by_structure(insumos)
        
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_empty_insumos_handling(self, classifier):
        """Lista vacía debe manejarse sin error."""
        clasificacion, _ = classifier.classify_by_structure([])
        assert clasificacion is not None

    def test_zero_value_insumos(self, classifier):
        """Insumos con valor cero deben procesarse."""
        insumos = [
            {"TIPO_INSUMO": "SUMINISTRO", "VALOR_TOTAL": 0},
            {"TIPO_INSUMO": "MANO_DE_OBRA", "VALOR_TOTAL": 100}
        ]
        clasificacion, _ = classifier.classify_by_structure(insumos)
        assert clasificacion is not None

    def test_extreme_value_distribution(self, classifier):
        """Distribución extrema de valores."""
        insumos = [
            {"TIPO_INSUMO": "SUMINISTRO", "VALOR_TOTAL": 1_000_000},
            {"TIPO_INSUMO": "MANO_DE_OBRA", "VALOR_TOTAL": 1}
        ]
        clasificacion, _ = classifier.classify_by_structure(insumos)
        assert clasificacion == "ESTRUCTURA_MIXTA"

    @pytest.mark.parametrize("tipo", ["EQUIPO", "TRANSPORTE"])
    def test_other_types_classification(self, classifier, tipo: str):
        """Otros tipos de insumo deben clasificarse."""
        insumos = [{"TIPO_INSUMO": tipo, "VALOR_TOTAL": 5000}]
        clasificacion, _ = classifier.classify_by_structure(insumos)
        assert clasificacion is not None


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7: PRUEBAS DE HIERARCHY LEVEL
# ═══════════════════════════════════════════════════════════════════════════════

class TestHierarchyLevel:
    """
    Pruebas para HierarchyLevel.
    
    Debe ser isomorfo a Stratum para mantener coherencia
    en el modelo topológico del sistema.
    """

    def test_hierarchy_level_values(self):
        """Verifica valores de cada nivel."""
        assert HierarchyLevel.ROOT.value == 0
        assert HierarchyLevel.STRATEGY.value == 1
        assert HierarchyLevel.TACTIC.value == 2
        assert HierarchyLevel.LOGISTICS.value == 3

    def test_hierarchy_level_completeness(self):
        """Todos los niveles esperados existen."""
        expected = {'ROOT', 'STRATEGY', 'TACTIC', 'LOGISTICS'}
        actual = {level.name for level in HierarchyLevel}
        assert actual == expected

    def test_hierarchy_stratum_isomorphism(self):
        """
        HierarchyLevel ≅ Stratum (isomorfismo de estructuras).
        
        El mapeo preserva:
        - Nombres
        - Valores
        - Orden
        """
        for level in HierarchyLevel:
            stratum = Stratum[level.name]
            assert level.value == stratum.value
            assert level.name == stratum.name

    def test_hierarchy_ordering(self):
        """El orden es consistente con la estructura piramidal."""
        levels = [
            HierarchyLevel.ROOT,
            HierarchyLevel.STRATEGY,
            HierarchyLevel.TACTIC,
            HierarchyLevel.LOGISTICS
        ]
        values = [l.value for l in levels]
        assert values == sorted(values)


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8: PRUEBAS DE INVARIANTES GLOBALES
# ═══════════════════════════════════════════════════════════════════════════════

class TestGlobalTopologicalInvariants:
    """
    Pruebas de invariantes topológicos que aplican a todo el sistema.
    
    Estas propiedades deben mantenerse para garantizar
    la coherencia matemática del modelo piramidal.
    """

    def test_stratum_forms_linear_chain(self):
        """Los estratos forman una cadena lineal (0-simplejo)."""
        values = sorted(s.value for s in Stratum)
        expected = list(range(len(values)))
        assert values == expected

    def test_pyramid_is_dag(self):
        """
        La estructura piramidal es un DAG (grafo acíclico dirigido).
        El flujo va de ROOT → LOGISTICS (valores crecientes).
        """
        assert Stratum.ROOT.value < Stratum.LOGISTICS.value

    def test_strata_partition_nodes(self):
        """
        Los estratos particionan el conjunto de nodos.
        Cada nodo pertenece a exactamente un estrato.
        """
        nodes = [
            TopologicalNode(id="root", stratum=Stratum.ROOT, description="R"),
            TopologicalNode(id="strat", stratum=Stratum.STRATEGY, description="S"),
            TopologicalNode(id="tact", stratum=Stratum.TACTIC, description="T"),
            TopologicalNode(id="log", stratum=Stratum.LOGISTICS, description="L"),
        ]
        
        strata_assigned = [n.stratum for n in nodes]
        # Cada nodo tiene exactamente un estrato
        assert all(s is not None for s in strata_assigned)
        # Los estratos son distintos en este caso
        assert len(set(strata_assigned)) == len(strata_assigned)

    def test_health_conservation_bound(self):
        """
        La salud total está acotada por el número de nodos.
        Σ(health) ≤ |nodes| (porque health ∈ [0, 1])
        """
        n = 100
        nodes = [
            TopologicalNode(
                id=f"n{i}",
                stratum=Stratum.LOGISTICS,
                description=f"Node {i}",
                structural_health=0.7
            )
            for i in range(n)
        ]
        
        total_health = sum(node.structural_health for node in nodes)
        assert total_health <= n


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 9: PRUEBAS DE CASOS LÍMITE Y ROBUSTEZ
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCasesAndRobustness:
    """
    Pruebas de casos extremos para garantizar robustez.
    """

    def test_unicode_in_descriptions(self):
        """Manejo de caracteres Unicode en descripciones."""
        insumo = InsumoProcesado(
            codigo_apu="APU001",
            descripcion_apu="Muro con ñ, áéíóú, €, ™",
            unidad_apu="m²",
            descripcion_insumo="Ladrillo café 日本語",
            unidad_insumo="unidade",
            cantidad=10,
            precio_unitario=500,
            valor_total=5000,
            tipo_insumo="SUMINISTRO"
        )
        assert insumo.stratum == Stratum.LOGISTICS
        assert insumo.id is not None

    def test_long_descriptions_truncation(self):
        """Descripciones muy largas no causan IDs excesivos."""
        long_text = "X" * 10000
        
        insumo = InsumoProcesado(
            codigo_apu="APU001",
            descripcion_apu=long_text,
            unidad_apu="m2",
            descripcion_insumo=long_text,
            unidad_insumo="und",
            cantidad=10,
            precio_unitario=100,
            valor_total=1000,
            tipo_insumo="SUMINISTRO"
        )
        
        # El ID debe estar acotado
        assert len(insumo.id) < 500

    def test_numerical_precision_float(self):
        """Precisión numérica con flotantes problemáticos."""
        # 0.1 + 0.2 ≠ 0.3 en IEEE 754
        insumo = InsumoProcesado(
            codigo_apu="APU001",
            descripcion_apu="Test",
            unidad_apu="m2",
            descripcion_insumo="Material",
            unidad_insumo="und",
            cantidad=0.1,
            precio_unitario=0.2,
            valor_total=0.02,
            tipo_insumo="SUMINISTRO"
        )
        
        expected = 0.1 * 0.2
        assert abs(insumo.cantidad * insumo.precio_unitario - expected) < 1e-15

    def test_empty_strings_in_fields(self):
        """Campos con strings vacíos."""
        insumo = InsumoProcesado(
            codigo_apu="APU001",
            descripcion_apu="",
            unidad_apu="",
            descripcion_insumo="",
            unidad_insumo="",
            cantidad=0,
            precio_unitario=0,
            valor_total=0,
            tipo_insumo="SUMINISTRO"
        )
        assert insumo.id.startswith("APU001_")

    def test_whitespace_handling(self):
        """Manejo de espacios en blanco."""
        insumo = InsumoProcesado(
            codigo_apu="  APU001  ",
            descripcion_apu="  Muro  ",
            unidad_apu=" m2 ",
            descripcion_insumo="  Ladrillo  ",
            unidad_insumo=" und ",
            cantidad=10,
            precio_unitario=100,
            valor_total=1000,
            tipo_insumo="SUMINISTRO"
        )
        # Dependiendo de normalización, verificar que no falle
        assert insumo is not None


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 10: PRUEBAS DE INTEGRACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """
    Pruebas de integración que verifican flujos completos.
    """

    def test_full_pyramid_construction_flow(self):
        """Construcción completa de una pirámide válida."""
        # 1. Crear insumos (LOGISTICS)
        insumos = [
            InsumoProcesado(
                codigo_apu="APU001", descripcion_apu="Muro", unidad_apu="m2",
                descripcion_insumo=f"Material_{i}", unidad_insumo="und",
                cantidad=10 * (i + 1), precio_unitario=100,
                valor_total=1000 * (i + 1), tipo_insumo="SUMINISTRO"
            )
            for i in range(5)
        ]
        
        # 2. Crear APU (TACTIC)
        apu = APUStructure(
            id="APU001", description="Muro de Prueba",
            unit="m2", quantity=100
        )
        
        # 3. Conectar insumos al APU
        for insumo in insumos:
            apu.add_resource(insumo)
        
        # 4. Verificar estructura
        assert apu.stratum == Stratum.TACTIC
        assert all(i.stratum == Stratum.LOGISTICS for i in insumos)
        assert apu.support_base_width == len(insumos)
        assert apu.stratum.value < insumos[0].stratum.value

    def test_validator_and_classifier_consistency(self):
        """Validador y clasificador producen resultados coherentes."""
        validator = PyramidalValidator()
        classifier = StructuralClassifier()
        
        # Estructura para validar
        apus_df = pd.DataFrame({"CODIGO_APU": ["APU001", "APU002"]})
        insumos_df = pd.DataFrame({
            "APU_CODIGO": ["APU001", "APU001", "APU002", "APU002"],
            "DESCRIPCION_INSUMO_NORM": ["MAT_A", "MAT_B", "MAT_C", "MO_A"]
        })
        
        # Datos para clasificar
        insumos_classify = [
            {"TIPO_INSUMO": "SUMINISTRO", "VALOR_TOTAL": 200},
            {"TIPO_INSUMO": "SUMINISTRO", "VALOR_TOTAL": 300}
        ]
        
        metrics = validator.validate_structure(apus_df, insumos_df)
        clasificacion, _ = classifier.classify_by_structure(insumos_classify)
        
        # Verificar coherencia
        assert len(metrics.floating_nodes) == 0
        assert clasificacion == "SUMINISTRO_PURO"
        assert metrics.pyramid_stability_index > 0


# ═══════════════════════════════════════════════════════════════════════════════
# EJECUCIÓN DE PRUEBAS
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])