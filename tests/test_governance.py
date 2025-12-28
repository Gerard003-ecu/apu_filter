"""
Pruebas para el Motor de Gobernanza y Validación Semántica.

Cobertura:
- ComplianceReport: add_violation, _update_status, get_summary
- GovernanceEngine: inicialización, carga de ontología, validación semántica
- Edge cases: None, vacíos, tipos incorrectos, estructuras inválidas
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from app.constants import ColumnNames
from app.governance import (
    GovernanceEngine,
    ComplianceReport,
    Severity,
    ComplianceStatus,
    SEVERITY_PENALTIES,
    SCORE_THRESHOLDS,
)


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def sample_ontology(tmp_path):
    """Ontología básica con dominios de prueba."""
    ontology = {
        "ontology_version": "1.0",
        "domains": {
            "CIMENTACION": {
                "identifying_keywords": ["CIMENTACION", "ZAPATA", "VIGA DE FUNDACION"],
                "required_keywords": ["CONCRETO", "ACERO"],
                "forbidden_keywords": ["PINTURA", "CUBIERTA"],
                "min_required_matches": 1,
            },
            "ACABADOS": {
                "identifying_keywords": ["ACABADOS", "PINTURA", "ESTUCO"],
                "required_keywords": ["PINTURA", "VINILO"],
                "forbidden_keywords": ["CIMENTACION", "EXCAVACION"],
                "min_required_matches": 1,
            },
        },
    }
    path = tmp_path / "ontology.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ontology, f)
    return path


@pytest.fixture
def ontology_without_identifying_keywords(tmp_path):
    """Ontología sin identifying_keywords para probar fallback."""
    ontology = {
        "domains": {
            "CIMENTACION": {
                "required_keywords": ["CONCRETO"],
                "forbidden_keywords": ["PINTURA"],
            },
        },
    }
    path = tmp_path / "ontology_fallback.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ontology, f)
    return path


@pytest.fixture
def invalid_ontology_structure(tmp_path):
    """Ontología con estructura inválida."""
    path = tmp_path / "invalid_ontology.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"domains": "esto_deberia_ser_dict"}, f)
    return path


@pytest.fixture
def invalid_json_file(tmp_path):
    """Archivo con JSON mal formado."""
    path = tmp_path / "malformed.json"
    with open(path, "w", encoding="utf-8") as f:
        f.write("{invalid json content")
    return path


@pytest.fixture
def empty_file(tmp_path):
    """Archivo vacío."""
    path = tmp_path / "empty.json"
    path.touch()
    return path


@pytest.fixture
def sample_dataframe():
    """DataFrame con datos de prueba incluyendo errores semánticos."""
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


@pytest.fixture
def dataframe_with_nulls():
    """DataFrame con valores nulos en descripciones."""
    data = {
        ColumnNames.CODIGO_APU: ["APU01", "APU01", "APU01"],
        ColumnNames.DESCRIPCION_APU: [
            "ZAPATA DE CIMENTACION",
            "ZAPATA DE CIMENTACION",
            "ZAPATA DE CIMENTACION",
        ],
        ColumnNames.DESCRIPCION_INSUMO: [
            "CONCRETO 3000 PSI",
            None,
            "ACERO DE REFUERZO",
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def configured_engine(tmp_path, sample_ontology):
    """Engine configurado con ontología y política habilitada."""
    engine = GovernanceEngine(config_dir=str(tmp_path))
    engine.load_ontology(str(sample_ontology))
    engine.semantic_policy = {"enable_ontology_check": True}
    return engine


# =============================================================================
# TESTS: ComplianceReport
# =============================================================================


class TestComplianceReport:
    """Pruebas para la clase ComplianceReport."""

    def test_initial_state(self):
        """Verifica estado inicial del reporte."""
        report = ComplianceReport()

        assert report.score == 100.0
        assert report.status == ComplianceStatus.PASS.value
        assert report.violations == []
        assert report.semantic_alerts == []
        assert report._error_count == 0
        assert report._warning_count == 0

    def test_add_violation_with_error_severity(self):
        """Verifica que ERROR penaliza correctamente y actualiza status."""
        report = ComplianceReport()

        result = report.add_violation(
            type_="TEST_ERROR", message="Test error message", severity="ERROR"
        )

        assert result is True
        assert len(report.violations) == 1
        assert report.violations[0]["type"] == "TEST_ERROR"
        assert report.violations[0]["severity"] == Severity.ERROR.value
        assert report.score == 100.0 - SEVERITY_PENALTIES[Severity.ERROR]
        assert report._error_count == 1
        assert report.status == ComplianceStatus.FAIL.value

    def test_add_violation_with_warning_severity(self):
        """Verifica que WARNING penaliza correctamente."""
        report = ComplianceReport()

        result = report.add_violation(
            type_="TEST_WARNING", message="Test warning message", severity="WARNING"
        )

        assert result is True
        assert report.score == 100.0 - SEVERITY_PENALTIES[Severity.WARNING]
        assert report._warning_count == 1
        assert report.status == ComplianceStatus.WARNING.value

    def test_add_violation_with_info_severity(self):
        """Verifica que INFO no penaliza el score."""
        report = ComplianceReport()

        report.add_violation(type_="TEST_INFO", message="Test info message", severity="INFO")

        assert report.score == 100.0
        assert report.status == ComplianceStatus.PASS.value

    def test_add_violation_with_invalid_severity_defaults_to_error(self):
        """Verifica que severidad inválida se normaliza a ERROR."""
        report = ComplianceReport()

        result = report.add_violation(
            type_="TEST", message="Test message", severity="INVALID_SEVERITY"
        )

        assert result is True
        assert report.violations[0]["severity"] == Severity.ERROR.value
        assert report._error_count == 1

    def test_add_violation_with_lowercase_severity(self):
        """Verifica normalización de severidad en minúsculas."""
        report = ComplianceReport()

        report.add_violation(type_="TEST", message="Test message", severity="warning")

        assert report.violations[0]["severity"] == Severity.WARNING.value

    def test_add_violation_with_empty_type_returns_false(self):
        """Verifica rechazo de type_ vacío."""
        report = ComplianceReport()

        result = report.add_violation(type_="", message="Test message")

        assert result is False
        assert len(report.violations) == 0

    def test_add_violation_with_none_type_returns_false(self):
        """Verifica rechazo de type_ None."""
        report = ComplianceReport()

        result = report.add_violation(type_=None, message="Test message")

        assert result is False
        assert len(report.violations) == 0

    def test_add_violation_with_empty_message_returns_false(self):
        """Verifica rechazo de message vacío."""
        report = ComplianceReport()

        result = report.add_violation(type_="TEST", message="")

        assert result is False
        assert len(report.violations) == 0

    def test_add_violation_with_context(self):
        """Verifica que el contexto se almacena correctamente."""
        report = ComplianceReport()
        context = {"apu_code": "APU01", "domain": "CIMENTACION"}

        report.add_violation(
            type_="TEST", message="Test message", severity="WARNING", context=context
        )

        assert "context" in report.violations[0]
        assert report.violations[0]["context"]["apu_code"] == "APU01"

    def test_add_violation_with_invalid_context_type_ignores_context(self):
        """Verifica que contexto no-dict se ignora."""
        report = ComplianceReport()

        report.add_violation(type_="TEST", message="Test message", context="invalid_context")

        assert "context" not in report.violations[0]

    def test_score_cannot_go_below_zero(self):
        """Verifica que el score no baja de 0."""
        report = ComplianceReport()

        # Añadir suficientes errores para exceder 100 puntos de penalización
        for i in range(25):
            report.add_violation(type_=f"ERROR_{i}", message=f"Error {i}", severity="ERROR")

        assert report.score == 0.0

    def test_status_updates_based_on_score_thresholds(self):
        """Verifica actualización de status según umbrales de score."""
        report = ComplianceReport()

        # Score por encima de warning threshold -> PASS
        assert report.status == ComplianceStatus.PASS.value

        # Añadir warnings hasta bajar del threshold
        warnings_needed = (
            int((100.0 - SCORE_THRESHOLDS["warning"]) / SEVERITY_PENALTIES[Severity.WARNING])
            + 1
        )
        for i in range(warnings_needed):
            report.add_violation(
                type_=f"WARN_{i}", message=f"Warning {i}", severity="WARNING"
            )

        assert report.score < SCORE_THRESHOLDS["warning"]
        assert report.status == ComplianceStatus.WARNING.value

    def test_get_summary_returns_correct_structure(self):
        """Verifica estructura del resumen."""
        report = ComplianceReport()
        report.add_violation(type_="ERR", message="Error", severity="ERROR")
        report.add_violation(type_="WARN", message="Warning", severity="WARNING")

        summary = report.get_summary()

        assert "status" in summary
        assert "score" in summary
        assert "total_violations" in summary
        assert "errors" in summary
        assert "warnings" in summary
        assert summary["total_violations"] == 2
        assert summary["errors"] == 1
        assert summary["warnings"] == 1

    def test_multiple_violations_accumulate_correctly(self):
        """Verifica acumulación correcta de múltiples violaciones."""
        report = ComplianceReport()

        report.add_violation(type_="ERR1", message="Error 1", severity="ERROR")
        report.add_violation(type_="ERR2", message="Error 2", severity="ERROR")
        report.add_violation(type_="WARN1", message="Warning 1", severity="WARNING")

        expected_score = (
            100.0
            - (2 * SEVERITY_PENALTIES[Severity.ERROR])
            - SEVERITY_PENALTIES[Severity.WARNING]
        )

        assert len(report.violations) == 3
        assert report.score == expected_score
        assert report._error_count == 2
        assert report._warning_count == 1


# =============================================================================
# TESTS: GovernanceEngine - Inicialización
# =============================================================================


class TestGovernanceEngineInitialization:
    """Pruebas de inicialización del GovernanceEngine."""

    def test_initialization_with_empty_config_dir(self, tmp_path):
        """Verifica inicialización cuando el directorio está vacío."""
        engine = GovernanceEngine(config_dir=str(tmp_path))

        assert engine.ontology == {}
        assert engine.semantic_policy.get("enable_ontology_check") is True  # Default
        assert engine._ontology_loaded is False

    def test_initialization_with_nonexistent_dir(self):
        """Verifica inicialización con directorio inexistente."""
        engine = GovernanceEngine(config_dir="/nonexistent/path")

        assert engine.ontology == {}
        assert engine._config_loaded is False

    def test_initialization_loads_existing_ontology(self, tmp_path, sample_ontology):
        """Verifica que carga ontología existente en config_dir."""
        # La ontología ya está en tmp_path por el fixture sample_ontology

        engine = GovernanceEngine(config_dir=str(tmp_path))

        assert "domains" in engine.ontology
        assert engine._ontology_loaded is True

    def test_config_dir_stored_as_path(self, tmp_path):
        """Verifica que config_dir se almacena como Path."""
        engine = GovernanceEngine(config_dir=str(tmp_path))

        assert isinstance(engine.config_dir, Path)


# =============================================================================
# TESTS: GovernanceEngine - Carga de Ontología
# =============================================================================


class TestGovernanceEngineLoadOntology:
    """Pruebas para el método load_ontology."""

    def test_load_ontology_success(self, tmp_path, sample_ontology):
        """Verifica carga exitosa de ontología."""
        engine = GovernanceEngine(config_dir=str(tmp_path))

        result = engine.load_ontology(str(sample_ontology))

        assert result is True
        assert "domains" in engine.ontology
        assert "CIMENTACION" in engine.ontology["domains"]
        assert engine._ontology_loaded is True

    def test_load_ontology_nonexistent_file(self, tmp_path):
        """Verifica manejo de archivo inexistente."""
        engine = GovernanceEngine(config_dir=str(tmp_path))

        result = engine.load_ontology("/nonexistent/file.json")

        assert result is False
        assert engine.ontology == {}

    def test_load_ontology_empty_path(self, tmp_path):
        """Verifica rechazo de ruta vacía."""
        engine = GovernanceEngine(config_dir=str(tmp_path))

        result = engine.load_ontology("")

        assert result is False

    def test_load_ontology_none_path(self, tmp_path):
        """Verifica manejo de ruta None."""
        engine = GovernanceEngine(config_dir=str(tmp_path))

        # El método debería manejar esto sin excepción
        result = engine.load_ontology(None)

        assert result is False

    def test_load_ontology_invalid_json(self, tmp_path, invalid_json_file):
        """Verifica manejo de JSON mal formado."""
        engine = GovernanceEngine(config_dir=str(tmp_path))

        result = engine.load_ontology(str(invalid_json_file))

        assert result is False
        assert engine.ontology == {}

    def test_load_ontology_empty_file(self, tmp_path, empty_file):
        """Verifica manejo de archivo vacío."""
        engine = GovernanceEngine(config_dir=str(tmp_path))

        result = engine.load_ontology(str(empty_file))

        assert result is False

    def test_load_ontology_invalid_structure(self, tmp_path, invalid_ontology_structure):
        """Verifica rechazo de estructura inválida."""
        engine = GovernanceEngine(config_dir=str(tmp_path))

        result = engine.load_ontology(str(invalid_ontology_structure))

        assert result is False

    def test_load_ontology_directory_instead_of_file(self, tmp_path):
        """Verifica rechazo cuando se pasa directorio en vez de archivo."""
        engine = GovernanceEngine(config_dir=str(tmp_path))

        result = engine.load_ontology(str(tmp_path))

        assert result is False

    def test_load_ontology_preserves_previous_on_failure(
        self, tmp_path, sample_ontology, invalid_json_file
    ):
        """Verifica que fallo no sobrescribe ontología válida existente."""
        engine = GovernanceEngine(config_dir=str(tmp_path))
        engine.load_ontology(str(sample_ontology))
        original_ontology = engine.ontology.copy()

        # Intentar cargar ontología inválida
        result = engine.load_ontology(str(invalid_json_file))

        assert result is False
        assert engine.ontology == original_ontology

    def test_load_ontology_with_wrong_extension_logs_warning(self, tmp_path, caplog):
        """Verifica warning al cargar archivo con extensión incorrecta."""
        # Crear archivo .txt con contenido JSON válido
        txt_file = tmp_path / "ontology.txt"
        with open(txt_file, "w") as f:
            json.dump({"domains": {}}, f)

        engine = GovernanceEngine(config_dir=str(tmp_path))

        with caplog.at_level("WARNING"):
            result = engine.load_ontology(str(txt_file))

        assert result is True  # Debería funcionar pero con warning
        assert "extensión" in caplog.text.lower() or "extension" in caplog.text.lower()


# =============================================================================
# TESTS: GovernanceEngine - Validación de Estructura
# =============================================================================


class TestOntologyStructureValidation:
    """Pruebas para _validate_ontology_structure."""

    def test_valid_complete_ontology(self, tmp_path):
        """Verifica aceptación de ontología completa y válida."""
        engine = GovernanceEngine(config_dir=str(tmp_path))

        ontology = {
            "domains": {
                "DOMAIN1": {
                    "required_keywords": ["KW1"],
                    "forbidden_keywords": ["KW2"],
                    "identifying_keywords": ["ID1"],
                }
            }
        }

        assert engine._validate_ontology_structure(ontology) is True

    def test_valid_minimal_ontology(self, tmp_path):
        """Verifica aceptación de ontología mínima (solo domains vacío)."""
        engine = GovernanceEngine(config_dir=str(tmp_path))

        assert engine._validate_ontology_structure({"domains": {}}) is True

    def test_valid_ontology_without_domains(self, tmp_path):
        """Verifica aceptación de ontología sin key domains."""
        engine = GovernanceEngine(config_dir=str(tmp_path))

        # Debería ser válido (domains es opcional según la implementación)
        assert engine._validate_ontology_structure({}) is True

    def test_invalid_ontology_not_dict(self, tmp_path):
        """Verifica rechazo cuando ontología no es dict."""
        engine = GovernanceEngine(config_dir=str(tmp_path))

        assert engine._validate_ontology_structure([]) is False
        assert engine._validate_ontology_structure("string") is False
        assert engine._validate_ontology_structure(None) is False

    def test_invalid_domains_not_dict(self, tmp_path):
        """Verifica rechazo cuando domains no es dict."""
        engine = GovernanceEngine(config_dir=str(tmp_path))

        assert engine._validate_ontology_structure({"domains": "invalid"}) is False
        assert engine._validate_ontology_structure({"domains": []}) is False

    def test_invalid_domain_rules_not_dict(self, tmp_path):
        """Verifica rechazo cuando reglas de dominio no son dict."""
        engine = GovernanceEngine(config_dir=str(tmp_path))

        ontology = {"domains": {"DOMAIN1": "invalid"}}

        assert engine._validate_ontology_structure(ontology) is False

    def test_invalid_keywords_not_list(self, tmp_path):
        """Verifica rechazo cuando keywords no son listas."""
        engine = GovernanceEngine(config_dir=str(tmp_path))

        ontology = {"domains": {"DOMAIN1": {"required_keywords": "should_be_list"}}}

        assert engine._validate_ontology_structure(ontology) is False


# =============================================================================
# TESTS: GovernanceEngine - Validación Semántica
# =============================================================================


class TestSemanticCoherence:
    """Pruebas para check_semantic_coherence."""

    def test_detects_forbidden_keyword_violation(self, configured_engine, sample_dataframe):
        """Verifica detección de palabra prohibida (PINTURA en CIMENTACION)."""
        report = configured_engine.check_semantic_coherence(sample_dataframe)

        semantic_violations = [
            v for v in report.violations if v["type"] == "SEMANTIC_INCONSISTENCY"
        ]

        assert len(semantic_violations) >= 1

        # Verificar que la violación tiene el contexto correcto
        violation = semantic_violations[0]
        assert "context" in violation
        assert violation["context"]["forbidden_keyword"] == "PINTURA"

    def test_detects_incompleteness_violation(self, tmp_path, sample_ontology):
        """Verifica detección de APU sin insumos requeridos."""
        df = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["APU_FAKE", "APU_FAKE"],
                ColumnNames.DESCRIPCION_APU: [
                    "ZAPATA DE CIMENTACION",
                    "ZAPATA DE CIMENTACION",
                ],
                ColumnNames.DESCRIPCION_INSUMO: [
                    "AGUA",
                    "ADITIVO",
                ],  # Faltan CONCRETO o ACERO
            }
        )

        engine = GovernanceEngine(config_dir=str(tmp_path))
        engine.load_ontology(str(sample_ontology))
        engine.semantic_policy = {"enable_ontology_check": True}

        report = engine.check_semantic_coherence(df)

        incompleteness = [
            v for v in report.violations if v["type"] == "SEMANTIC_INCOMPLETENESS"
        ]

        assert len(incompleteness) >= 1
        assert incompleteness[0]["context"]["apu_code"] == "APU_FAKE"

    def test_no_violation_when_policy_disabled(self, tmp_path, sample_dataframe):
        """Verifica que no hay validación cuando está deshabilitada."""
        engine = GovernanceEngine(config_dir=str(tmp_path))
        engine.semantic_policy = {"enable_ontology_check": False}

        report = engine.check_semantic_coherence(sample_dataframe)

        assert len(report.violations) == 0
        assert report.status == ComplianceStatus.PASS.value

    def test_handles_none_dataframe(self, configured_engine):
        """Verifica manejo de DataFrame None."""
        report = configured_engine.check_semantic_coherence(None)

        assert len(report.violations) == 0

    def test_handles_empty_dataframe(self, configured_engine):
        """Verifica manejo de DataFrame vacío."""
        empty_df = pd.DataFrame()

        report = configured_engine.check_semantic_coherence(empty_df)

        assert len(report.violations) == 0

    def test_handles_wrong_type_input(self, configured_engine):
        """Verifica manejo de tipo incorrecto."""
        report = configured_engine.check_semantic_coherence("not_a_dataframe")

        type_errors = [v for v in report.violations if v["type"] == "TYPE_ERROR"]
        assert len(type_errors) >= 1

    def test_handles_missing_columns(self, configured_engine):
        """Verifica manejo de columnas faltantes."""
        df = pd.DataFrame({"COLUMNA_RANDOM": [1, 2, 3]})

        report = configured_engine.check_semantic_coherence(df)

        schema_errors = [v for v in report.violations if v["type"] == "SCHEMA_ERROR"]
        assert len(schema_errors) >= 1

    def test_handles_null_values_in_descriptions(
        self, tmp_path, sample_ontology, dataframe_with_nulls
    ):
        """Verifica manejo correcto de valores nulos."""
        engine = GovernanceEngine(config_dir=str(tmp_path))
        engine.load_ontology(str(sample_ontology))
        engine.semantic_policy = {"enable_ontology_check": True}

        # No debería lanzar excepción
        report = engine.check_semantic_coherence(dataframe_with_nulls)

        assert report is not None
        # No debería haber incompleteness porque tiene CONCRETO y ACERO
        incompleteness = [
            v for v in report.violations if v["type"] == "SEMANTIC_INCOMPLETENESS"
        ]
        assert len(incompleteness) == 0

    def test_no_violation_for_compliant_apu(self, configured_engine):
        """Verifica que APU conforme no genera violaciones."""
        compliant_df = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["APU_OK", "APU_OK"],
                ColumnNames.DESCRIPCION_APU: [
                    "ZAPATA DE CIMENTACION",
                    "ZAPATA DE CIMENTACION",
                ],
                ColumnNames.DESCRIPCION_INSUMO: ["CONCRETO 3000 PSI", "ACERO DE REFUERZO"],
            }
        )

        report = configured_engine.check_semantic_coherence(compliant_df)

        # No debería haber violaciones para CIMENTACION
        cimentacion_violations = [
            v
            for v in report.violations
            if "context" in v and v["context"].get("domain") == "CIMENTACION"
        ]
        assert len(cimentacion_violations) == 0

    def test_domain_inference_uses_identifying_keywords(self, tmp_path, sample_ontology):
        """Verifica que la inferencia usa identifying_keywords."""
        # APU que contiene ZAPATA (identifying keyword) pero no CIMENTACION
        df = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["APU_ZAPATA"],
                ColumnNames.DESCRIPCION_APU: [
                    "ZAPATA EN CONCRETO REFORZADO"
                ],  # Sin "CIMENTACION"
                ColumnNames.DESCRIPCION_INSUMO: [
                    "PINTURA EPOXICA"
                ],  # Prohibido en cimentación
            }
        )

        engine = GovernanceEngine(config_dir=str(tmp_path))
        engine.load_ontology(str(sample_ontology))
        engine.semantic_policy = {"enable_ontology_check": True}

        report = engine.check_semantic_coherence(df)

        # Debería detectar dominio CIMENTACION por keyword "ZAPATA"
        violations = [v for v in report.violations if v["type"] == "SEMANTIC_INCONSISTENCY"]
        assert len(violations) >= 1

    def test_domain_inference_fallback_without_identifying_keywords(
        self, tmp_path, ontology_without_identifying_keywords
    ):
        """Verifica fallback cuando no hay identifying_keywords."""
        df = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["APU01"],
                ColumnNames.DESCRIPCION_APU: ["VIGA DE CIMENTACION"],
                ColumnNames.DESCRIPCION_INSUMO: ["PINTURA"],  # Prohibido
            }
        )

        engine = GovernanceEngine(config_dir=str(tmp_path))
        engine.load_ontology(str(ontology_without_identifying_keywords))
        engine.semantic_policy = {"enable_ontology_check": True}

        report = engine.check_semantic_coherence(df)

        violations = [v for v in report.violations if v["type"] == "SEMANTIC_INCONSISTENCY"]
        assert len(violations) >= 1

    def test_case_insensitive_matching(self, configured_engine):
        """Verifica que el matching es case-insensitive."""
        df = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["APU01"],
                ColumnNames.DESCRIPCION_APU: ["viga de cimentacion"],  # minúsculas
                ColumnNames.DESCRIPCION_INSUMO: ["pintura vinilo"],  # minúsculas
            }
        )

        report = configured_engine.check_semantic_coherence(df)

        violations = [v for v in report.violations if v["type"] == "SEMANTIC_INCONSISTENCY"]
        assert len(violations) >= 1

    def test_reports_violation_count_in_context(self, configured_engine):
        """Verifica que el contexto incluye conteo de violaciones."""
        # DataFrame con múltiples insumos prohibidos
        df = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["APU01"] * 5,
                ColumnNames.DESCRIPCION_APU: ["ZAPATA DE CIMENTACION"] * 5,
                ColumnNames.DESCRIPCION_INSUMO: [
                    "PINTURA TIPO A",
                    "PINTURA TIPO B",
                    "PINTURA TIPO C",
                    "PINTURA TIPO D",
                    "CONCRETO 3000 PSI",
                ],
            }
        )

        report = configured_engine.check_semantic_coherence(df)

        violation = next(
            (v for v in report.violations if v["type"] == "SEMANTIC_INCONSISTENCY"), None
        )

        assert violation is not None
        assert "context" in violation
        assert violation["context"]["violation_count"] == 4  # 4 tipos de pintura


# =============================================================================
# TESTS: GovernanceEngine - Métodos Auxiliares
# =============================================================================


class TestHelperMethods:
    """Pruebas para métodos auxiliares."""

    def test_safe_string_upper_with_string(self, tmp_path):
        """Verifica conversión de string normal."""
        engine = GovernanceEngine(config_dir=str(tmp_path))

        assert engine._safe_string_upper("hello world") == "HELLO WORLD"
        assert engine._safe_string_upper("  spaces  ") == "SPACES"

    def test_safe_string_upper_with_none(self, tmp_path):
        """Verifica manejo de None."""
        engine = GovernanceEngine(config_dir=str(tmp_path))

        assert engine._safe_string_upper(None) == ""

    def test_safe_string_upper_with_nan(self, tmp_path):
        """Verifica manejo de NaN."""
        engine = GovernanceEngine(config_dir=str(tmp_path))

        assert engine._safe_string_upper(float("nan")) == ""

    def test_safe_string_upper_with_number(self, tmp_path):
        """Verifica conversión de número."""
        engine = GovernanceEngine(config_dir=str(tmp_path))

        assert engine._safe_string_upper(123) == "123"
        assert engine._safe_string_upper(45.67) == "45.67"

    def test_normalize_keyword_list_with_valid_list(self, tmp_path):
        """Verifica normalización de lista válida."""
        engine = GovernanceEngine(config_dir=str(tmp_path))

        result = engine._normalize_keyword_list(["hello", "WORLD", "  spaces  "])

        assert result == ["HELLO", "WORLD", "SPACES"]

    def test_normalize_keyword_list_with_empty_strings(self, tmp_path):
        """Verifica que strings vacíos se filtran."""
        engine = GovernanceEngine(config_dir=str(tmp_path))

        result = engine._normalize_keyword_list(["hello", "", "  ", "world"])

        assert result == ["HELLO", "WORLD"]

    def test_normalize_keyword_list_with_non_list(self, tmp_path):
        """Verifica manejo de input no-lista."""
        engine = GovernanceEngine(config_dir=str(tmp_path))

        assert engine._normalize_keyword_list("not a list") == []
        assert engine._normalize_keyword_list(None) == []
        assert engine._normalize_keyword_list(123) == []

    def test_normalize_keyword_list_with_mixed_types(self, tmp_path):
        """Verifica manejo de lista con tipos mixtos."""
        engine = GovernanceEngine(config_dir=str(tmp_path))

        result = engine._normalize_keyword_list(["hello", 123, None, "world"])

        # Solo strings válidos deberían incluirse
        assert "HELLO" in result
        assert "WORLD" in result


# =============================================================================
# TESTS: Integración
# =============================================================================


class TestIntegration:
    """Pruebas de integración end-to-end."""

    def test_full_workflow_with_violations(
        self, tmp_path, sample_ontology, sample_dataframe
    ):
        """Prueba flujo completo con violaciones detectadas."""
        # Inicializar
        engine = GovernanceEngine(config_dir=str(tmp_path))

        # Cargar ontología
        load_result = engine.load_ontology(str(sample_ontology))
        assert load_result is True

        # Habilitar política
        engine.semantic_policy = {"enable_ontology_check": True}

        # Ejecutar validación
        report = engine.check_semantic_coherence(sample_dataframe)

        # Verificar resultado
        assert report.score < 100.0
        assert len(report.violations) > 0
        assert report.status in [ComplianceStatus.WARNING.value, ComplianceStatus.FAIL.value]

        # Verificar resumen
        summary = report.get_summary()
        assert summary["total_violations"] == len(report.violations)

    def test_full_workflow_without_violations(self, tmp_path, sample_ontology):
        """Prueba flujo completo sin violaciones."""
        compliant_df = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["APU01", "APU01"],
                ColumnNames.DESCRIPCION_APU: ["ZAPATA CIMENTACION", "ZAPATA CIMENTACION"],
                ColumnNames.DESCRIPCION_INSUMO: ["CONCRETO PREMEZCLADO", "ACERO CORRUGADO"],
            }
        )

        engine = GovernanceEngine(config_dir=str(tmp_path))
        engine.load_ontology(str(sample_ontology))
        engine.semantic_policy = {"enable_ontology_check": True}

        report = engine.check_semantic_coherence(compliant_df)

        assert report.score == 100.0
        assert report.status == ComplianceStatus.PASS.value

    def test_reload_ontology_updates_validation(self, tmp_path, sample_ontology):
        """Verifica que recargar ontología afecta validaciones siguientes."""
        df = pd.DataFrame(
            {
                ColumnNames.CODIGO_APU: ["APU01"],
                ColumnNames.DESCRIPCION_APU: ["TRABAJO DE CIMENTACION"],
                ColumnNames.DESCRIPCION_INSUMO: ["PINTURA ESPECIAL"],
            }
        )

        engine = GovernanceEngine(config_dir=str(tmp_path))
        engine.load_ontology(str(sample_ontology))
        engine.semantic_policy = {"enable_ontology_check": True}

        # Primera validación - debería detectar violación
        report1 = engine.check_semantic_coherence(df)
        violations1 = len(report1.violations)

        # Crear nueva ontología sin PINTURA como prohibido
        new_ontology = {
            "domains": {
                "CIMENTACION": {
                    "identifying_keywords": ["CIMENTACION"],
                    "required_keywords": ["CONCRETO"],
                    "forbidden_keywords": [],  # Sin palabras prohibidas
                }
            }
        }
        new_path = tmp_path / "new_ontology.json"
        with open(new_path, "w") as f:
            json.dump(new_ontology, f)

        # Recargar ontología
        engine.load_ontology(str(new_path))

        # Segunda validación - no debería detectar violación de forbidden
        report2 = engine.check_semantic_coherence(df)
        forbidden_violations = [
            v for v in report2.violations if v["type"] == "SEMANTIC_INCONSISTENCY"
        ]

        assert len(forbidden_violations) == 0
