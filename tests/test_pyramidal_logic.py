import pytest
from app.data_validator import PyramidalValidator, PyramidalMetrics
from app.schemas import Stratum, TopologicalNode, InsumoProcesado, APUStructure
from app.data_loader import HierarchyLevel, load_data_with_hierarchy
from app.classifiers.apu_classifier import StructuralClassifier
import pandas as pd

class TestPyramidalLogic:
    def test_stratum_enum(self):
        assert Stratum.ROOT == 0
        assert Stratum.STRATEGY == 1
        assert Stratum.TACTIC == 2
        assert Stratum.LOGISTICS == 3

    def test_topological_node(self):
        node = TopologicalNode(id="node1", stratum=Stratum.TACTIC, description="APU Test")
        assert node.structural_health == 1.0
        assert node.is_floating is False

    def test_insumo_procesado_is_topological(self):
        insumo = InsumoProcesado(
            codigo_apu="APU1",
            descripcion_apu="Muro",
            unidad_apu="m2",
            descripcion_insumo="Ladrillo",
            unidad_insumo="und",
            cantidad=10,
            precio_unitario=500,
            valor_total=5000,
            tipo_insumo="SUMINISTRO"
        )
        assert isinstance(insumo, TopologicalNode)
        assert insumo.stratum == Stratum.LOGISTICS
        # Note: Normalization might change casing, so we check normalized expectation
        # schemas.py implementation: f"{self.codigo_apu}_{self.descripcion_insumo[:20]}"
        # And normalizers might have run.
        # "Ladrillo" -> "LADRILLO" if normalized.
        # InsumoProcesado.__post_init__ logic:
        # 1. self.id is set using raw "Ladrillo" -> "APU1_Ladrillo"
        # 2. _normalize_all_fields() is called, updating descripcion_insumo to "LADRILLO"
        # 3. BUT self.id is NOT updated after normalization.
        # So we assert against "APU1_Ladrillo"
        assert insumo.id == "APU1_Ladrillo"

    def test_apu_structure(self):
        apu = APUStructure(id="APU1", description="Muro", unit="m2", quantity=100)
        assert apu.stratum == Stratum.TACTIC
        assert apu.support_base_width == 0

        insumo = InsumoProcesado(
            codigo_apu="APU1",
            descripcion_apu="Muro",
            unidad_apu="m2",
            descripcion_insumo="Ladrillo",
            unidad_insumo="und",
            cantidad=10,
            precio_unitario=500,
            valor_total=5000,
            tipo_insumo="SUMINISTRO"
        )
        apu.add_resource(insumo)
        assert apu.support_base_width == 1

    def test_pyramidal_validator(self):
        validator = PyramidalValidator()

        # Mock DataFrames
        # APU1, APU2 are valid
        # APU3 is floating (no inputs)
        apus_df = pd.DataFrame({
            "CODIGO_APU": ["APU1", "APU2", "APU3"]
        })

        # Insumos/Detail DataFrame
        # This mocks the detail table where APUs are linked to inputs
        # APU1 -> INSUMO A
        # APU2 -> INSUMO B, INSUMO C
        # APU3 -> (Nothing)
        insumos_df = pd.DataFrame({
            "CODIGO_APU": ["APU1", "APU2", "APU2"],
            "DESCRIPCION_INSUMO_NORM": ["INSUMO A", "INSUMO B", "INSUMO C"]
        })

        metrics = validator.validate_structure(apus_df, insumos_df)

        assert metrics.base_width == 3 # A, B, C
        assert metrics.structure_load == 3 # APU1, APU2, APU3
        assert metrics.pyramid_stability_index == 3/3  # 1.0

        assert len(metrics.floating_nodes) == 1
        assert metrics.floating_nodes[0] == "APU3"

    def test_structural_classifier(self):
        classifier = StructuralClassifier()

        # Caso Suministro Puro (Solo Materiales)
        insumos_mat = [
            {"TIPO_INSUMO": "SUMINISTRO", "COSTO": 100},
            {"TIPO_INSUMO": "SUMINISTRO", "COSTO": 200}
        ]
        assert classifier.classify_by_structure(insumos_mat) == "SUMINISTRO_PURO"

        # Caso Servicio Puro (Solo Mano de Obra)
        insumos_mo = [
            {"TIPO_INSUMO": "MANO_DE_OBRA", "COSTO": 100}
        ]
        assert classifier.classify_by_structure(insumos_mo) == "SERVICIO_PURO"

        # Caso Mixto
        insumos_mix = [
            {"TIPO_INSUMO": "SUMINISTRO", "COSTO": 100},
            {"TIPO_INSUMO": "MANO_DE_OBRA", "COSTO": 100}
        ]
        assert classifier.classify_by_structure(insumos_mix) == "CONSTRUCCION_MIXTA"

    def test_hierarchy_level_enum(self):
        assert HierarchyLevel.ROOT.value == 0
        assert HierarchyLevel.LOGISTICS.value == 3
