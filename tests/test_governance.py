"""
Pruebas para el Motor de Gobernanza y Validación Semántica.
"""

import json

import pandas as pd
import pytest

from app.constants import ColumnNames
from app.governance import GovernanceEngine


@pytest.fixture
def sample_ontology(tmp_path):
    ontology = {
        "ontology_version": "1.0",
        "domains": {
            "CIMENTACION": {
                "required_keywords": ["CONCRETO", "ACERO"],
                "forbidden_keywords": ["PINTURA", "CUBIERTA"],
            },
            "ACABADOS": {
                "required_keywords": ["PINTURA"],
                "forbidden_keywords": ["CIMENTACION"],
            },
        },
    }
    path = tmp_path / "ontology.json"
    with open(path, "w") as f:
        json.dump(ontology, f)
    return path


@pytest.fixture
def sample_dataframe():
    data = {
        ColumnNames.CODIGO_APU: ["APU01", "APU01", "APU02", "APU03"],
        ColumnNames.DESCRIPCION_APU: [
            "VIGA DE CIMENTACION EN CONCRETO",
            "VIGA DE CIMENTACION EN CONCRETO",
            "MURO EN LAMINAS DE YESO (ACABADOS)",
            "COLUMNA DE CIMENTACION",
        ],
        ColumnNames.DESCRIPCION_INSUMO: [
            "CONCRETO 3000 PSI",
            "PINTURA TIPO 1 (ERROR SEMANTICO)",  # Error: Pintura en cimentación
            "PINTURA VINILO",
            "ACERO DE REFUERZO",
        ],
    }
    return pd.DataFrame(data)


def test_governance_initialization(tmp_path):
    engine = GovernanceEngine(config_dir=str(tmp_path))
    assert engine.ontology == {}


def test_load_ontology(tmp_path, sample_ontology):
    engine = GovernanceEngine(config_dir=str(tmp_path))
    engine.load_ontology(str(sample_ontology))
    assert "CIMENTACION" in engine.ontology["domains"]


def test_semantic_check_forbidden_keywords(tmp_path, sample_ontology, sample_dataframe):
    # Configurar engine
    engine = GovernanceEngine(config_dir=str(tmp_path))
    engine.load_ontology(str(sample_ontology))
    engine.semantic_policy = {"enable_ontology_check": True}

    report = engine.check_semantic_coherence(sample_dataframe)

    # Debe detectar "PINTURA" en APU01 (Cimentación)
    assert len(report.violations) > 0
    violation = next(
        (v for v in report.violations if v["type"] == "SEMANTIC_INCONSISTENCY"), None
    )
    assert violation is not None
    assert "PINTURA" in violation["message"]
    assert "APU01" in violation["message"]


def test_semantic_check_incompleteness(tmp_path, sample_ontology):
    # APU que dice ser Cimentación pero solo tiene insumos raros
    # Corregido: Arrays de misma longitud
    df = pd.DataFrame(
        {
            ColumnNames.CODIGO_APU: ["APU_FAKE", "APU_FAKE"],
            ColumnNames.DESCRIPCION_APU: ["ZAPATA DE CIMENTACION", "ZAPATA DE CIMENTACION"],
            ColumnNames.DESCRIPCION_INSUMO: ["AGUA", "ADITIVO"],  # Faltan CONCRETO o ACERO
        }
    )

    engine = GovernanceEngine(config_dir=str(tmp_path))
    engine.load_ontology(str(sample_ontology))
    engine.semantic_policy = {"enable_ontology_check": True}

    report = engine.check_semantic_coherence(df)

    violation = next(
        (v for v in report.violations if v["type"] == "SEMANTIC_INCOMPLETENESS"), None
    )
    assert violation is not None
    assert "APU_FAKE" in violation["message"]


def test_semantic_check_disabled(tmp_path, sample_dataframe):
    engine = GovernanceEngine(config_dir=str(tmp_path))
    engine.semantic_policy = {"enable_ontology_check": False}
    report = engine.check_semantic_coherence(sample_dataframe)
    assert len(report.violations) == 0
