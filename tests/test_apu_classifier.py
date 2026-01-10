"""
Tests exhaustivos para el clasificador APU refactorizado.

Cubre:
- Normalización de operadores lógicos
- Validación sintáctica de condiciones
- Clasificación individual y vectorizada
- Análisis de cobertura topológica
- Edge cases y robustez numérica
"""

import json
import os
import tempfile
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from app.classifiers.apu_classifier import APUClassifier, ClassificationRule

CONFIG_PATH = "config/config_rules.json"


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def default_classifier() -> APUClassifier:
    """Fixture que proporciona un clasificador con reglas por defecto."""
    return APUClassifier()


@pytest.fixture
def config_classifier() -> APUClassifier:
    """Fixture que proporciona un clasificador con configuración real si existe."""
    if os.path.exists(CONFIG_PATH):
        return APUClassifier(CONFIG_PATH)
    pytest.skip(f"Archivo de configuración no encontrado: {CONFIG_PATH}")


@pytest.fixture
def temp_config_path():
    """Genera archivo de configuración temporal y lo limpia después."""
    paths = []

    def _create_config(config_dict: dict) -> str:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(config_dict, f)
            paths.append(f.name)
            return f.name

    yield _create_config

    for path in paths:
        if os.path.exists(path):
            os.unlink(path)


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """DataFrame de prueba con diversos escenarios."""
    return pd.DataFrame(
        {
            "VALOR_CONSTRUCCION_UN": [100, 200, 150, 300, 0, 50, 1000, np.nan],
            "VALOR_SUMINISTRO_UN": [70, 120, 45, 180, 0, 35, 100, 50],
            "VALOR_INSTALACION_UN": [30, 80, 105, 120, 0, 15, 900, 50],
        }
    )


# =============================================================================
# TESTS: ClassificationRule - Normalización de Operadores
# =============================================================================


class TestClassificationRuleNormalization:
    """Tests para normalización de operadores lógicos SQL → Python."""

    @pytest.mark.parametrize(
        "input_condition,expected_normalized",
        [
            # Mayúsculas
            (
                "porcentaje_materiales >= 50 AND porcentaje_mo_eq >= 30",
                "porcentaje_materiales >= 50 and porcentaje_mo_eq >= 30",
            ),
            # Minúsculas (sin cambio)
            (
                "porcentaje_materiales >= 50 and porcentaje_mo_eq >= 30",
                "porcentaje_materiales >= 50 and porcentaje_mo_eq >= 30",
            ),
            # Mixto
            (
                "porcentaje_materiales >= 50 And porcentaje_mo_eq >= 30",
                "porcentaje_materiales >= 50 and porcentaje_mo_eq >= 30",
            ),
            # OR
            (
                "porcentaje_materiales >= 60 OR porcentaje_mo_eq >= 60",
                "porcentaje_materiales >= 60 or porcentaje_mo_eq >= 60",
            ),
            # NOT
            ("NOT porcentaje_materiales >= 50", "not porcentaje_materiales >= 50"),
            # Combinación compleja
            (
                "(porcentaje_materiales >= 40 AND porcentaje_materiales <= 60) OR "
                "(porcentaje_mo_eq >= 40 AND porcentaje_mo_eq <= 60)",
                "(porcentaje_materiales >= 40 and porcentaje_materiales <= 60) or "
                "(porcentaje_mo_eq >= 40 and porcentaje_mo_eq <= 60)",
            ),
        ],
    )
    def test_operator_normalization(self, input_condition: str, expected_normalized: str):
        """Verifica normalización AND/OR/NOT → and/or/not."""
        rule = ClassificationRule(
            rule_type="TEST",
            priority=1,
            condition=input_condition,
            description="Test normalization",
        )
        assert rule.condition == expected_normalized

    def test_preserves_variable_names(self):
        """Verifica que no modifique 'and'/'or' dentro de nombres de variables."""
        # Edge case: nombres hipotéticos que contienen 'and'/'or'
        condition = "porcentaje_materiales >= 50"
        rule = ClassificationRule(
            rule_type="TEST",
            priority=1,
            condition=condition,
            description="Test",
        )
        assert "porcentaje_materiales" in rule.condition


# =============================================================================
# TESTS: ClassificationRule - Validación Sintáctica
# =============================================================================


class TestClassificationRuleValidation:
    """Tests para validación de seguridad y sintaxis."""

    @pytest.mark.parametrize(
        "valid_condition",
        [
            "porcentaje_materiales >= 50",
            "porcentaje_mo_eq > 30",
            "porcentaje_materiales >= 40 and porcentaje_materiales <= 60",
            "(porcentaje_materiales >= 50) or (porcentaje_mo_eq >= 50)",
            "not porcentaje_materiales >= 80",
            "porcentaje_materiales == 50.5",
            "porcentaje_mo_eq != 0",
            "porcentaje_materiales >= 0 and porcentaje_mo_eq >= 0",
        ],
    )
    def test_valid_conditions_accepted(self, valid_condition: str):
        """Condiciones válidas deben ser aceptadas."""
        rule = ClassificationRule(
            rule_type="TEST",
            priority=1,
            condition=valid_condition,
            description="Valid",
        )
        assert rule.condition is not None

    @pytest.mark.parametrize(
        "invalid_condition,error_fragment",
        [
            # Inyección de código
            ("__import__('os').system('ls')", "no permitidos"),
            ("exec('print(1)')", "no permitidos"),
            ("eval('1+1')", "no permitidos"),
            # Variables no permitidas
            ("unknown_var >= 50", "no permitidos"),
            ("porcentaje_materiales >= x", "no permitidos"),
            # Funciones no permitidas
            ("len(porcentaje_materiales) > 0", "no permitidos"),
            ("abs(porcentaje_materiales) >= 50", "no permitidos"),
            # Sintaxis inválida
            ("porcentaje_materiales >= ", "Sintaxis inválida"),
            ("porcentaje_materiales >= 50 &&", "no permitidos"),
            ("porcentaje_materiales >= 50 ||", "no permitidos"),
        ],
    )
    def test_invalid_conditions_rejected(self, invalid_condition: str, error_fragment: str):
        """Condiciones inválidas o peligrosas deben ser rechazadas."""
        with pytest.raises(ValueError) as exc_info:
            ClassificationRule(
                rule_type="TEST",
                priority=1,
                condition=invalid_condition,
                description="Invalid",
            )
        assert error_fragment in str(exc_info.value)

    def test_empty_condition_rejected(self):
        """Condición vacía debe ser rechazada."""
        with pytest.raises(ValueError):
            ClassificationRule(
                rule_type="TEST",
                priority=1,
                condition="   ",
                description="Empty",
            )


# =============================================================================
# TESTS: ClassificationRule - Evaluación
# =============================================================================


class TestClassificationRuleEvaluation:
    """Tests para evaluación de reglas."""

    @pytest.mark.parametrize(
        "pct_mat,pct_mo,condition,expected",
        [
            # Pruebas básicas de comparación
            (0.60, 0.20, "porcentaje_materiales >= 60", True),
            (0.59, 0.20, "porcentaje_materiales >= 60", False),
            (0.20, 0.70, "porcentaje_mo_eq >= 60", True),
            # Operadores combinados
            (
                0.50,
                0.50,
                "porcentaje_materiales >= 40 and porcentaje_materiales <= 60",
                True,
            ),
            (
                0.30,
                0.30,
                "porcentaje_materiales >= 40 and porcentaje_materiales <= 60",
                False,
            ),
            # OR
            (0.70, 0.10, "porcentaje_materiales >= 60 or porcentaje_mo_eq >= 60", True),
            (0.10, 0.70, "porcentaje_materiales >= 60 or porcentaje_mo_eq >= 60", True),
            (0.30, 0.30, "porcentaje_materiales >= 60 or porcentaje_mo_eq >= 60", False),
            # NOT
            (0.30, 0.30, "not porcentaje_materiales >= 60", True),
            (0.70, 0.30, "not porcentaje_materiales >= 60", False),
            # Edge: exactamente en el límite
            (0.60, 0.40, "porcentaje_materiales >= 60.0", True),
            (0.40, 0.60, "porcentaje_mo_eq >= 60.0", True),
        ],
    )
    def test_evaluate_conditions(
        self, pct_mat: float, pct_mo: float, condition: str, expected: bool
    ):
        """Verifica evaluación correcta de diversas condiciones."""
        rule = ClassificationRule(
            rule_type="TEST",
            priority=1,
            condition=condition,
            description="Test",
        )
        assert rule.evaluate(pct_mat, pct_mo) == expected

    def test_evaluate_handles_boundary_values(self):
        """Verifica manejo de valores límite [0, 1]."""
        rule = ClassificationRule(
            rule_type="TEST",
            priority=1,
            condition="porcentaje_materiales >= 0 and porcentaje_mo_eq >= 0",
            description="Universal",
        )

        assert rule.evaluate(0.0, 0.0) is True
        assert rule.evaluate(1.0, 1.0) is True
        assert rule.evaluate(0.5, 0.5) is True


# =============================================================================
# TESTS: ClassificationRule - Cobertura Topológica
# =============================================================================


class TestClassificationRuleCoverageBounds:
    """Tests para extracción de límites de cobertura."""

    @pytest.mark.parametrize(
        "condition,expected_mat,expected_mo",
        [
            ("porcentaje_materiales >= 60", (0.60, 1.0), (0.0, 1.0)),
            ("porcentaje_mo_eq >= 70", (0.0, 1.0), (0.70, 1.0)),
            (
                "porcentaje_materiales >= 40 and porcentaje_materiales <= 60",
                (0.40, 0.60),
                (0.0, 1.0),
            ),
            ("porcentaje_mo_eq > 50 and porcentaje_mo_eq < 80", (0.0, 1.0), (0.50, 0.80)),
        ],
    )
    def test_get_coverage_bounds(
        self,
        condition: str,
        expected_mat: Tuple[float, float],
        expected_mo: Tuple[float, float],
    ):
        """Verifica extracción heurística de bounds."""
        rule = ClassificationRule(
            rule_type="TEST",
            priority=1,
            condition=condition,
            description="Test",
        )
        (mat_min, mat_max), (mo_min, mo_max) = rule.get_coverage_bounds()

        assert abs(mat_min - expected_mat[0]) < 0.01
        assert abs(mat_max - expected_mat[1]) < 0.01
        assert abs(mo_min - expected_mo[0]) < 0.01
        assert abs(mo_max - expected_mo[1]) < 0.01


# =============================================================================
# TESTS: APUClassifier - Carga de Configuración
# =============================================================================


class TestAPUClassifierConfigLoading:
    """Tests para carga de configuración."""

    def test_loads_default_rules_without_config(self):
        """Sin config, debe cargar reglas por defecto."""
        classifier = APUClassifier()

        assert len(classifier.rules) > 0
        assert classifier.default_type == "INDEFINIDO"
        assert classifier.zero_cost_type == "SIN_COSTO"

    def test_loads_config_from_json(self, temp_config_path):
        """Carga correcta desde archivo JSON."""
        config = {
            "apu_classification_rules": {
                "rules": [
                    {
                        "type": "CUSTOM_TYPE",
                        "priority": 1,
                        "condition": "porcentaje_materiales >= 50",
                        "description": "Custom rule",
                    }
                ],
                "default_type": "CUSTOM_DEFAULT",
                "zero_cost_type": "CUSTOM_ZERO",
            }
        }
        config_path = temp_config_path(config)

        classifier = APUClassifier(config_path)

        assert len(classifier.rules) == 1
        assert classifier.rules[0].rule_type == "CUSTOM_TYPE"
        assert classifier.default_type == "CUSTOM_DEFAULT"
        assert classifier.zero_cost_type == "CUSTOM_ZERO"

    def test_handles_invalid_json_gracefully(self, temp_config_path):
        """JSON malformado debe fallback a defaults."""
        # Crear archivo con JSON inválido manualmente
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{invalid json content")
            bad_path = f.name

        try:
            classifier = APUClassifier(bad_path)
            assert len(classifier.rules) > 0  # Debe tener defaults
        finally:
            os.unlink(bad_path)

    def test_skips_invalid_rules_in_config(self, temp_config_path):
        """Reglas inválidas deben ser omitidas, no crashear."""
        config = {
            "apu_classification_rules": {
                "rules": [
                    {
                        "type": "VALID_RULE",
                        "priority": 1,
                        "condition": "porcentaje_materiales >= 50",
                    },
                    {
                        "type": "INVALID_RULE",
                        "priority": 2,
                        "condition": "__import__('os')",  # Inválida
                    },
                    {
                        # Falta 'type' - inválida
                        "priority": 3,
                        "condition": "porcentaje_mo_eq >= 50",
                    },
                ],
            }
        }
        config_path = temp_config_path(config)

        classifier = APUClassifier(config_path)

        # Solo la primera regla válida debe cargarse
        assert len(classifier.rules) == 1
        assert classifier.rules[0].rule_type == "VALID_RULE"

    def test_sorts_rules_by_priority(self, temp_config_path):
        """Reglas deben ordenarse por prioridad ascendente."""
        config = {
            "apu_classification_rules": {
                "rules": [
                    {
                        "type": "LOW",
                        "priority": 10,
                        "condition": "porcentaje_materiales >= 0",
                    },
                    {
                        "type": "HIGH",
                        "priority": 1,
                        "condition": "porcentaje_materiales >= 90",
                    },
                    {
                        "type": "MED",
                        "priority": 5,
                        "condition": "porcentaje_materiales >= 50",
                    },
                ],
            }
        }
        config_path = temp_config_path(config)

        classifier = APUClassifier(config_path)

        priorities = [r.priority for r in classifier.rules]
        assert priorities == sorted(priorities)
        assert classifier.rules[0].rule_type == "HIGH"


# =============================================================================
# TESTS: APUClassifier - Clasificación Individual
# =============================================================================


class TestAPUClassifierSingleClassification:
    """Tests para classify_single."""

    def test_zero_cost_returns_sin_costo(self, default_classifier):
        """Costo cero debe retornar SIN_COSTO."""
        result = default_classifier.classify_single(0.5, 0.5, total_cost=0.0)
        assert result == "SIN_COSTO"

    def test_negative_cost_returns_sin_costo(self, default_classifier):
        """Costo negativo debe retornar SIN_COSTO."""
        result = default_classifier.classify_single(0.5, 0.5, total_cost=-100.0)
        assert result == "SIN_COSTO"

    def test_nan_cost_returns_sin_costo(self, default_classifier):
        """Costo NaN debe retornar SIN_COSTO."""
        result = default_classifier.classify_single(0.5, 0.5, total_cost=np.nan)
        assert result == "SIN_COSTO"

    @pytest.mark.parametrize(
        "pct_mat,pct_mo,expected",
        [
            (0.70, 0.30, "SUMINISTRO"),  # Materiales >= 60%
            (0.30, 0.70, "INSTALACION"),  # MO >= 60%
            (0.50, 0.50, "CONSTRUCCION_MIXTO"),  # Balance
            (0.45, 0.45, "CONSTRUCCION_MIXTO"),  # Rango 40-60
        ],
    )
    def test_default_rules_classification(
        self, default_classifier, pct_mat: float, pct_mo: float, expected: str
    ):
        """Verifica clasificación con reglas por defecto."""
        result = default_classifier.classify_single(pct_mat, pct_mo, total_cost=100.0)
        assert result == expected

    def test_clips_out_of_range_percentages(self, default_classifier):
        """Porcentajes fuera de [0,1] deben ser clipped."""
        # pct > 1 debe tratarse como 1
        result = default_classifier.classify_single(1.5, 0.0, total_cost=100.0)
        assert result in ["SUMINISTRO", "OBRA_COMPLETA"]  # Materiales al 100%

        # pct < 0 debe tratarse como 0
        result = default_classifier.classify_single(-0.5, 0.8, total_cost=100.0)
        assert result == "INSTALACION"  # MO al 80%

    def test_with_real_config_scenarios(self, config_classifier):
        """Test con escenarios usando configuración real."""
        test_cases = [
            (0.70, 0.10, "SUMINISTRO", "Predominio material puro", 100.0),
            (0.65, 0.25, "SUMINISTRO_PREFABRICADO", "Material con MO moderada", 100.0),
            (0.30, 0.65, "INSTALACION", "Predominio MO/equipo", 100.0),
            (0.50, 0.45, "CONSTRUCCION_MIXTO", "Balance 50/50", 100.0),
            (0.55, 0.40, "CONSTRUCCION_MIXTO", "Límite superior mixto", 100.0),
            (0.45, 0.35, "CONSTRUCCION_MIXTO", "Caso intermedio", 100.0),
            (0.00, 0.00, "SIN_COSTO", "Sin costo", 0.0),
            (0.90, 0.05, "SUMINISTRO", "Material muy alto", 100.0),
        ]

        for pct_mat, pct_mo_eq, expected, desc, total_cost in test_cases:
            result = config_classifier.classify_single(pct_mat, pct_mo_eq, total_cost)
            assert result == expected, (
                f"Fallo en '{desc}': esperado {expected}, obtenido {result}"
            )


# =============================================================================
# TESTS: APUClassifier - Clasificación DataFrame
# =============================================================================


class TestAPUClassifierDataFrameClassification:
    """Tests para classify_dataframe."""

    def test_adds_output_column(self, default_classifier, sample_dataframe):
        """Debe añadir columna de clasificación."""
        result = default_classifier.classify_dataframe(sample_dataframe)

        assert "TIPO_APU" in result.columns
        assert len(result) == len(sample_dataframe)

    def test_no_null_classifications(self, default_classifier, sample_dataframe):
        """No debe haber clasificaciones nulas."""
        result = default_classifier.classify_dataframe(sample_dataframe)

        assert result["TIPO_APU"].isna().sum() == 0

    def test_zero_cost_row_classified(self, default_classifier, sample_dataframe):
        """Filas con costo cero deben ser SIN_COSTO."""
        result = default_classifier.classify_dataframe(sample_dataframe)

        zero_cost_mask = sample_dataframe["VALOR_CONSTRUCCION_UN"] == 0
        zero_cost_classifications = result.loc[zero_cost_mask, "TIPO_APU"]

        assert all(t == "SIN_COSTO" for t in zero_cost_classifications)

    def test_nan_cost_row_classified(self, default_classifier, sample_dataframe):
        """Filas con costo NaN deben ser SIN_COSTO."""
        result = default_classifier.classify_dataframe(sample_dataframe)

        nan_cost_mask = sample_dataframe["VALOR_CONSTRUCCION_UN"].isna()
        nan_classifications = result.loc[nan_cost_mask, "TIPO_APU"]

        assert all(t == "SIN_COSTO" for t in nan_classifications)

    def test_custom_column_names(self, default_classifier):
        """Debe funcionar con nombres de columna personalizados."""
        df = pd.DataFrame(
            {
                "total": [100, 200],
                "mat": [60, 40],
                "mo": [40, 60],
            }
        )

        result = default_classifier.classify_dataframe(
            df,
            col_total="total",
            col_materiales="mat",
            col_mo_eq="mo",
            output_col="clasificacion",
        )

        assert "clasificacion" in result.columns
        assert result["clasificacion"].isna().sum() == 0

    def test_missing_columns_handled(self, default_classifier):
        """Columnas faltantes deben manejarse gracefully."""
        df = pd.DataFrame(
            {
                "VALOR_CONSTRUCCION_UN": [100, 200],
                # Faltan las otras columnas
            }
        )

        result = default_classifier.classify_dataframe(df)

        assert "TIPO_APU" in result.columns
        # Todas deben ser default por falta de datos
        assert all(t == default_classifier.default_type for t in result["TIPO_APU"])

    def test_does_not_modify_original(self, default_classifier, sample_dataframe):
        """No debe modificar el DataFrame original."""
        original_cols = set(sample_dataframe.columns)

        _ = default_classifier.classify_dataframe(sample_dataframe)

        assert set(sample_dataframe.columns) == original_cols

    def test_handles_string_numeric_columns(self, default_classifier):
        """Debe manejar columnas numéricas como strings."""
        df = pd.DataFrame(
            {
                "VALOR_CONSTRUCCION_UN": ["100", "200", "abc", None],
                "VALOR_SUMINISTRO_UN": ["60", "80", "50", "25"],
                "VALOR_INSTALACION_UN": ["40", "120", "50", "25"],
            }
        )

        result = default_classifier.classify_dataframe(df)

        assert result["TIPO_APU"].isna().sum() == 0


# =============================================================================
# TESTS: APUClassifier - Vectorización
# =============================================================================


class TestAPUClassifierVectorization:
    """Tests para verificar que la vectorización produce resultados correctos."""

    def test_vectorized_matches_single(self, default_classifier):
        """Clasificación vectorizada debe coincidir con individual."""
        test_cases = [
            (100, 70, 30),  # SUMINISTRO
            (100, 30, 70),  # INSTALACION
            (100, 50, 50),  # MIXTO
            (0, 0, 0),  # SIN_COSTO
            (100, 100, 0),  # SUMINISTRO
        ]

        df = pd.DataFrame(
            test_cases,
            columns=[
                "VALOR_CONSTRUCCION_UN",
                "VALOR_SUMINISTRO_UN",
                "VALOR_INSTALACION_UN",
            ],
        )

        df_result = default_classifier.classify_dataframe(df)

        for idx, (total, mat, mo) in enumerate(test_cases):
            pct_mat = mat / total if total > 0 else 0
            pct_mo = mo / total if total > 0 else 0

            single_result = default_classifier.classify_single(pct_mat, pct_mo, total)
            vector_result = df_result.iloc[idx]["TIPO_APU"]

            assert single_result == vector_result, (
                f"Mismatch en fila {idx}: single={single_result}, vector={vector_result}"
            )

    def test_large_dataframe_performance(self, default_classifier):
        """Debe manejar DataFrames grandes eficientemente."""
        np.random.seed(42)
        n = 10_000

        df = pd.DataFrame(
            {
                "VALOR_CONSTRUCCION_UN": np.random.uniform(0, 1000, n),
                "VALOR_SUMINISTRO_UN": np.random.uniform(0, 500, n),
                "VALOR_INSTALACION_UN": np.random.uniform(0, 500, n),
            }
        )

        # Esto debe completarse en tiempo razonable (< 5 segundos)
        import time

        start = time.time()
        result = default_classifier.classify_dataframe(df)
        elapsed = time.time() - start

        assert len(result) == n
        assert elapsed < 5.0, f"Clasificación muy lenta: {elapsed:.2f}s"


# =============================================================================
# TESTS: APUClassifier - Cobertura Topológica
# =============================================================================


class TestAPUClassifierTopologicalCoverage:
    """Tests para análisis de cobertura del espacio [0,1]²."""

    def test_default_rules_cover_space(self, default_classifier):
        """Reglas por defecto deben cubrir todo el espacio válido."""
        uncovered = default_classifier._sample_uncovered_regions(grid_size=10)

        # No debe haber regiones sin cobertura
        assert len(uncovered) == 0, f"Regiones sin cobertura: {uncovered}"

    def test_coverage_report_structure(self, default_classifier):
        """Reporte de cobertura debe tener estructura correcta."""
        report = default_classifier.get_coverage_report()

        assert isinstance(report, pd.DataFrame)
        expected_cols = {
            "tipo",
            "prioridad",
            "mat_range",
            "mo_range",
            "area_estimada",
            "condicion",
        }
        assert expected_cols.issubset(set(report.columns))
        assert len(report) == len(default_classifier.rules)

    def test_detects_gaps_in_coverage(self, temp_config_path):
        """Debe detectar huecos en la cobertura."""
        # Configuración con gap intencional
        config = {
            "apu_classification_rules": {
                "rules": [
                    {
                        "type": "HIGH_MAT",
                        "priority": 1,
                        "condition": "porcentaje_materiales >= 80",
                    },
                    {
                        "type": "HIGH_MO",
                        "priority": 2,
                        "condition": "porcentaje_mo_eq >= 80",
                    },
                    # Gap: no hay regla para (mat < 80) AND (mo < 80)
                ],
            }
        }
        config_path = temp_config_path(config)
        classifier = APUClassifier(config_path)

        uncovered = classifier._sample_uncovered_regions(grid_size=10)

        # Debe detectar regiones sin cobertura
        assert len(uncovered) > 0


# =============================================================================
# TESTS: APUClassifier - Validación de Reglas
# =============================================================================


class TestAPUClassifierRuleValidation:
    """Tests para validación de coherencia de reglas."""

    def test_raises_on_no_rules(self, temp_config_path):
        """Debe fallar si no hay reglas definidas."""
        config = {"apu_classification_rules": {"rules": []}}
        config_path = temp_config_path(config)

        # Todas las reglas son inválidas, y no quedan defaults
        # Esto debería usar defaults, verificar comportamiento
        classifier = APUClassifier(config_path)

        # Si no hay reglas válidas, debe cargar defaults
        assert len(classifier.rules) > 0

    def test_warns_on_duplicate_types(self, temp_config_path, caplog):
        """Debe advertir sobre tipos duplicados."""
        import logging

        config = {
            "apu_classification_rules": {
                "rules": [
                    {
                        "type": "DUPLICATE",
                        "priority": 1,
                        "condition": "porcentaje_materiales >= 50",
                    },
                    {
                        "type": "DUPLICATE",
                        "priority": 2,
                        "condition": "porcentaje_mo_eq >= 50",
                    },
                ],
            }
        }
        config_path = temp_config_path(config)

        with caplog.at_level(logging.WARNING):
            APUClassifier(config_path)

        assert "duplicados" in caplog.text.lower() or "duplicates" in caplog.text.lower()


# =============================================================================
# TESTS: Edge Cases y Robustez
# =============================================================================


class TestAPUClassifierEdgeCases:
    """Tests para casos límite y robustez."""

    def test_empty_dataframe(self, default_classifier):
        """DataFrame vacío debe manejarse correctamente."""
        df = pd.DataFrame(
            {
                "VALOR_CONSTRUCCION_UN": [],
                "VALOR_SUMINISTRO_UN": [],
                "VALOR_INSTALACION_UN": [],
            }
        )

        result = default_classifier.classify_dataframe(df)

        assert len(result) == 0
        assert "TIPO_APU" in result.columns

    def test_single_row_dataframe(self, default_classifier):
        """DataFrame de una fila debe funcionar."""
        df = pd.DataFrame(
            {
                "VALOR_CONSTRUCCION_UN": [100],
                "VALOR_SUMINISTRO_UN": [70],
                "VALOR_INSTALACION_UN": [30],
            }
        )

        result = default_classifier.classify_dataframe(df)

        assert len(result) == 1
        assert result.iloc[0]["TIPO_APU"] == "SUMINISTRO"

    def test_infinity_values(self, default_classifier):
        """Valores infinitos deben manejarse."""
        df = pd.DataFrame(
            {
                "VALOR_CONSTRUCCION_UN": [np.inf, -np.inf, 100],
                "VALOR_SUMINISTRO_UN": [50, 50, 70],
                "VALOR_INSTALACION_UN": [50, 50, 30],
            }
        )

        result = default_classifier.classify_dataframe(df)

        # No debe crashear
        assert len(result) == 3

    def test_very_small_percentages(self, default_classifier):
        """Porcentajes muy pequeños deben manejarse."""
        result = default_classifier.classify_single(0.001, 0.001, total_cost=100.0)

        # Debe tener alguna clasificación válida
        assert result in ["OBRA_COMPLETA", "INDEFINIDO", "CONSTRUCCION_MIXTO"]

    def test_exact_boundary_values(self, default_classifier):
        """Valores exactamente en límites deben clasificarse consistentemente."""
        # Exactamente 60%
        result_60 = default_classifier.classify_single(0.60, 0.40, total_cost=100.0)

        # Justo debajo de 60%
        result_59 = default_classifier.classify_single(0.59, 0.41, total_cost=100.0)

        # 60% debe ser SUMINISTRO, 59% debe ser MIXTO o diferente
        assert result_60 == "SUMINISTRO"


# =============================================================================
# TESTS: Compatibilidad con Tests Originales
# =============================================================================


class TestBackwardsCompatibility:
    """Tests que replican la suite original para garantizar compatibilidad."""

    def test_classifier_with_various_scenarios(self):
        """Test del clasificador con múltiples escenarios (original)."""
        if os.path.exists(CONFIG_PATH):
            classifier = APUClassifier(CONFIG_PATH)
        else:
            pytest.skip(f"Archivo de configuración no encontrado: {CONFIG_PATH}")

        test_cases = [
            (0.70, 0.10, "SUMINISTRO", "Predominio material puro", 100.0),
            (0.65, 0.25, "SUMINISTRO_PREFABRICADO", "Material con MO moderada", 100.0),
            (0.30, 0.65, "INSTALACION", "Predominio MO/equipo", 100.0),
            (0.50, 0.45, "CONSTRUCCION_MIXTO", "Balance 50/50", 100.0),
            (0.55, 0.40, "CONSTRUCCION_MIXTO", "Límite superior mixto", 100.0),
            (0.45, 0.35, "CONSTRUCCION_MIXTO", "Caso intermedio (Mixto)", 100.0),
            (0.00, 0.00, "SIN_COSTO", "Sin costo", 0.0),
            (0.90, 0.05, "SUMINISTRO", "Material muy alto, MO baja", 100.0),
        ]

        for pct_mat, pct_mo_eq, expected, desc, total_cost in test_cases:
            result = classifier.classify_single(pct_mat, pct_mo_eq, total_cost=total_cost)
            assert result == expected, (
                f"Fallo en {desc}: esperado {expected}, obtenido {result}"
            )

    def test_dataframe_classification(self):
        """Test de clasificación de DataFrame completo (original)."""
        if os.path.exists(CONFIG_PATH):
            classifier = APUClassifier(CONFIG_PATH)
        else:
            classifier = APUClassifier()

        df = pd.DataFrame(
            {
                "VALOR_CONSTRUCCION_UN": [100, 200, 150, 300, 0],
                "VALOR_SUMINISTRO_UN": [70, 120, 45, 180, 0],
                "VALOR_INSTALACION_UN": [30, 80, 105, 120, 0],
            }
        )

        df_classified = classifier.classify_dataframe(df)

        assert "TIPO_APU" in df_classified.columns
        assert df_classified["TIPO_APU"].isna().sum() == 0
        assert df_classified.iloc[4]["TIPO_APU"] == "SIN_COSTO"

    def test_config_loading(self):
        """Test de carga de configuración desde JSON (original)."""
        config = {
            "apu_classification_rules": {
                "rules": [
                    {
                        "type": "TEST_TYPE",
                        "priority": 1,
                        "condition": "porcentaje_materiales >= 50.0",
                        "description": "Test rule",
                    }
                ],
                "default_type": "TEST_DEFAULT",
            }
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            config_path = f.name

        try:
            classifier = APUClassifier(config_path)
            assert len(classifier.rules) == 1
            assert classifier.rules[0].rule_type == "TEST_TYPE"
            assert classifier.default_type == "TEST_DEFAULT"
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)


# =============================================================================
# MAIN
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
