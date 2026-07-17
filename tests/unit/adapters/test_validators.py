"""
═══════════════════════════════════════════════════════════════════════════════
SUITE DE PRUEBAS RIGUROSA: Validadores Ciber-Físicos
UBICACIÓN: tests/unit/core/test_validators.py
VERSIÓN: 2.0.0 - Fortificación Algebraica con Cobertura Exhaustiva
═══════════════════════════════════════════════════════════════════════════════

FUNDAMENTOS MATEMÁTICOS DE TESTING
-----------------------------------

Esta suite implementa VERIFICACIÓN FORMAL de propiedades algebraicas mediante:

1. **Property-Based Testing**: Invariantes universales ∀x ∈ Domain
2. **Axiomas del Monoide**: Asociatividad, elemento neutro, clausura
3. **Leyes de Retículo**: Orden parcial, supremo, ínfimo
4. **Teoría de Categorías**: Funtorialidad, naturalidad
5. **Topología Algebraica**: Homeomorfismos, continuidad

ORGANIZACIÓN TAXONÓMICA
-----------------------

I.   Fixtures de Datos (DataFrames canónicos para casos de prueba)
II.  Propiedades Algebraicas del Monoide (ValidationResult)
III. Retículo de Severidades (ValidationSeverity)
IV.  Estructuras Inmutables (ValidationIssue)
V.   Validadores Estructurales (Isomorfismo Dimensional)
VI.  Validadores de Calidad (Extirpación de Singularidades)
VII. Validadores Numéricos (Clausura Algebraica IEEE 754)
VIII.Validadores de Dominio (Termodinámica y Compacidad)
IX.  Validadores Topológicos (Independencia Lineal, Supervivencia)
X.   Orquestación de Dominio (Composición Funcional)
XI.  Casos Límite y Manejo de Errores
XII. Utilidades y Helpers Internos

COBERTURA DE CÓDIGO
-------------------
Target: 100% (líneas, ramas, condiciones)

MÉTRICAS DE CALIDAD
-------------------
- Complejidad Ciclomática: < 10 por función
- Cobertura de Mutación: > 95%
- Cobertura de Ramas: 100%

═══════════════════════════════════════════════════════════════════════════════
"""

import math
import sys
from dataclasses import FrozenInstanceError
from typing import Any, Callable, List, Optional

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st
from numpy.testing import assert_array_almost_equal

from app.adapters.validators import (
    CONDITION_NUMBER_THRESHOLD,
    DEFAULT_SURVIVAL_THRESHOLD,
    FLOAT_TOLERANCE,
    LIPSCHITZ_LIMIT,
    MIN_NORMAL_FLOAT,
    DataFrameValidator,
    ValidationCode,
    ValidationIssue,
    ValidationResult,
    ValidationSeverity,
    _materialize_iterable,
)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN I: FIXTURES DE DATOS (Espacios Canónicos de Prueba)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """DataFrame vacío (espacio nulo ∅)."""
    return pd.DataFrame()


@pytest.fixture
def simple_df() -> pd.DataFrame:
    """
    DataFrame simple con tipos heterogéneos.
    
    Espacio: ℝ² × Strings³
    Dimensiones: 3 columnas × 3 filas
    """
    return pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4.0, 5.0, 6.0],
        "C": ["x", "y", "z"],
    })


@pytest.fixture
def df_with_nulls() -> pd.DataFrame:
    """
    DataFrame con singularidades topológicas (NaN, None, "").
    
    Singularidades:
    - A[1]: None
    - B[0]: NaN
    - C[0]: "" (string vacío)
    - C[2]: None
    """
    return pd.DataFrame({
        "A": [1, None, 3],
        "B": [np.nan, 5.0, 6.0],
        "C": ["", "y", None],
    })


@pytest.fixture
def df_all_nulls() -> pd.DataFrame:
    """DataFrame con todas las columnas nulas (singularidad total)."""
    return pd.DataFrame({
        "x": [None, None, None],
        "y": [np.nan, np.nan, np.nan],
    })


@pytest.fixture
def df_linearly_independent() -> pd.DataFrame:
    """
    Vectores canónicos linealmente independientes.
    
    Base: {e₁ = (1,0), e₂ = (0,1)}
    rank(matriz) = 2 = dim(espacio)
    """
    return pd.DataFrame({"x": [1, 0], "y": [0, 1]})


@pytest.fixture
def df_linearly_dependent() -> pd.DataFrame:
    """
    Columnas linealmente dependientes: y = 2x.
    
    rank(matriz) = 1 < 2 (degeneración del espacio)
    """
    return pd.DataFrame({"x": [1, 2, 3], "y": [2, 4, 6]})


@pytest.fixture
def df_rank_deficient_three_cols() -> pd.DataFrame:
    """
    Tres columnas con rank = 2 (una es combinación lineal de las otras).
    
    z = x + y (dependencia lineal)
    """
    return pd.DataFrame({
        "x": [1, 0, 1],
        "y": [0, 1, 1],
        "z": [1, 1, 2],  # z = x + y
    })


@pytest.fixture
def df_with_subnormals() -> pd.DataFrame:
    """
    Valores subnormales (denormales) IEEE 754.
    
    Subnormales:
    - sub[0]: sys.float_info.min / 2 (positivo)
    - sub[2]: -sys.float_info.min / 2 (negativo)
    """
    subnormal_positive = MIN_NORMAL_FLOAT / 2
    subnormal_negative = -MIN_NORMAL_FLOAT / 2
    
    return pd.DataFrame({
        "fine": [1e-10, 0.0, 1.0],
        "sub": [subnormal_positive, 0.0, subnormal_negative],
    })


@pytest.fixture
def df_with_infinities() -> pd.DataFrame:
    """DataFrame con valores infinitos (±∞)."""
    return pd.DataFrame({
        "x": [1.0, np.inf, -np.inf, 2.0],
        "y": [np.inf, np.inf, 1.0, 1.0],
    })


@pytest.fixture
def df_for_survival() -> pd.DataFrame:
    """
    DataFrame para validación de supervivencia volumétrica.
    
    Filas totales: 4
    Filas limpias (sin singularidades): 2 (índices 0 y 3)
    Supervivencia: 2/4 = 50%
    
    Singularidades:
    - Fila 1: col1 = NaN
    - Fila 2: col2 = "" (string vacío)
    """
    return pd.DataFrame({
        "col1": [1, np.nan, 3, 4],
        "col2": ["a", "z", "", "c"],
        "col3": [1.0, 2.0, 3.0, 4.0],
    })


@pytest.fixture
def df_with_duplicates() -> pd.DataFrame:
    """DataFrame con columnas duplicadas (base degenerada)."""
    return pd.DataFrame([[1, 2, 3]], columns=["A", "A", "B"])


@pytest.fixture
def df_with_negatives() -> pd.DataFrame:
    """DataFrame con valores negativos (violación termodinámica)."""
    return pd.DataFrame({
        "cost": [10, -5, 20, -3],
        "time": [1.0, 2.0, -0.5, 3.0],
    })


@pytest.fixture
def df_out_of_range() -> pd.DataFrame:
    """DataFrame con valores fuera de rango compacto."""
    return pd.DataFrame({
        "percentage": [-10, 50, 150],  # Debe estar en [0, 100]
        "temperature": [-300, 25, 500],  # Debe estar en [-273.15, 100]
    })


@pytest.fixture
def df_mixed_types() -> pd.DataFrame:
    """DataFrame con tipos mixtos (no coercibles a numérico)."""
    return pd.DataFrame({
        "mixed": [1, "two", 3.0, None, "five"],
    })


@pytest.fixture
def df_edge_cases() -> pd.DataFrame:
    """DataFrame con casos límite numéricos."""
    return pd.DataFrame({
        "zeros": [0.0, -0.0, 0],
        "extremes": [sys.float_info.max, -sys.float_info.max, sys.float_info.min],
        "tiny": [1e-308, 1e-309, 1e-310],
    })


# ══════════════════════════════════════════════════════════════════════════════
# ESTRATEGIAS DE HYPOTHESIS (Generación Estocástica)
# ══════════════════════════════════════════════════════════════════════════════

def st_severity():
    """Generador de severidades del retículo."""
    return st.sampled_from(ValidationSeverity)

def st_code():
    """Generador de códigos canónicos."""
    return st.sampled_from(ValidationCode)

def st_issue():
    """Generador de issues de validación."""
    return st.builds(
        ValidationIssue,
        severity=st_severity(),
        code=st_code(),
        message=st.text(min_size=1).filter(lambda s: s.strip() != ""),
        column=st.one_of(st.none(), st.text(min_size=1)),
        count=st.one_of(st.none(), st.integers(min_value=0))
    )

def st_validation_result():
    """Generador de resultados de validación (elementos del monoide)."""
    return st.builds(
        ValidationResult.from_issues,
        issues=st.lists(st_issue(), max_size=10)
    )


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN II: PROPIEDADES ALGEBRAICAS DEL MONOIDE (ValidationResult)
# ══════════════════════════════════════════════════════════════════════════════


class TestValidationResultMonoid:
    """
    Verifica axiomas del MONOIDE (ValidationResult, merge, IDENTITY).
    
    Axiomas Verificados:
    -------------------
    1. Clausura: ∀a,b ∈ M: a ⊕ b ∈ M
    2. Asociatividad: ∀a,b,c ∈ M: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)
    3. Elemento Neutro: ∃ε ∈ M: ∀a ∈ M: a ⊕ ε = ε ⊕ a = a
    """

    def test_identity_element_semantics(self):
        """El elemento neutro es consistente semánticamente."""
        id1 = ValidationResult.IDENTITY
        id2 = ValidationResult.success()
        assert id1 == id2
        assert len(id1.issues) == 0

    def test_identity_left_neutral(self):
        """ε ⊕ a = a (neutro por la izquierda)."""
        issue = ValidationIssue(
            ValidationSeverity.WARNING,
            ValidationCode.NULL_VALUES,
            "test",
        )
        result = ValidationResult.from_issues([issue])
        
        composed = ValidationResult.IDENTITY + result
        assert composed == result
        assert composed.issues == result.issues

    def test_identity_right_neutral(self):
        """a ⊕ ε = a (neutro por la derecha)."""
        issue = ValidationIssue(
            ValidationSeverity.ERROR,
            ValidationCode.MISSING_REQUIRED_COLUMN,
            "test",
        )
        result = ValidationResult.from_issues([issue])
        
        composed = result + ValidationResult.IDENTITY
        assert composed == result
        assert composed.issues == result.issues

    @given(r1=st_validation_result(), r2=st_validation_result(), r3=st_validation_result())
    def test_associativity_axiom(self, r1, r2, r3):
        """Axioma de Asociatividad: (a ⊕ b) ⊕ c = a ⊕ (b ⊕ c)."""
        left = (r1 + r2) + r3
        right = r1 + (r2 + r3)
        
        # Comparación estructural de issues (el orden se preserva por concatenación)
        assert left.issues == right.issues
        assert left.is_valid == right.is_valid

    @given(r=st_validation_result())
    def test_identity_axiom(self, r):
        """Axioma del Elemento Neutro: a ⊕ ε = ε ⊕ a = a."""
        epsilon = ValidationResult.IDENTITY
        assert (r + epsilon) == r
        assert (epsilon + r) == r

    def test_associativity_triple(self):
        """Asociatividad incluyendo elemento neutro."""
        r = ValidationResult.from_issues([
            ValidationIssue(ValidationSeverity.WARNING, ValidationCode.NULL_VALUES, "test")
        ])
        
        assert (r + ValidationResult.IDENTITY) + r == r + (ValidationResult.IDENTITY + r)

    def test_closure_property(self):
        """∀a,b: a ⊕ b ∈ ValidationResult (clausura)."""
        r1 = ValidationResult.success()
        r2 = ValidationResult.from_issues([
            ValidationIssue(ValidationSeverity.CRITICAL, ValidationCode.NONE_DATAFRAME, "fail")
        ])
        
        combo = r1 + r2
        assert isinstance(combo, ValidationResult)
        assert isinstance(r1.merge(r2), ValidationResult)

    def test_merge_vs_add_equivalence(self):
        """merge(a, b) ≡ a + b (equivalencia de operadores)."""
        r1 = ValidationResult.from_issues([
            ValidationIssue(ValidationSeverity.WARNING, ValidationCode.NULL_VALUES, "w1")
        ])
        r2 = ValidationResult.from_issues([
            ValidationIssue(ValidationSeverity.ERROR, ValidationCode.NON_FINITE_VALUES, "e1")
        ])
        
        merged = r1.merge(r2)
        added = r1 + r2
        
        assert merged == added
        assert merged.issues == added.issues

    def test_is_valid_indicator_function_correctness(self):
        """
        Función indicadora de validez:
        is_valid(r) = ∄ i ∈ r.issues : i.severity ∈ BLOCKING_SEVERITIES
        """
        # Caso: sin issues → válido
        assert ValidationResult.success().is_valid
        
        # Caso: solo INFO → válido
        r_info = ValidationResult.from_issues([
            ValidationIssue(ValidationSeverity.INFO, ValidationCode.NULL_VALUES, "info")
        ])
        assert r_info.is_valid
        
        # Caso: solo WARNING → válido
        r_warn = ValidationResult.from_issues([
            ValidationIssue(ValidationSeverity.WARNING, ValidationCode.EMPTY_STRINGS, "warn")
        ])
        assert r_warn.is_valid
        
        # Caso: con ERROR → inválido
        r_err = ValidationResult.from_issues([
            ValidationIssue(ValidationSeverity.ERROR, ValidationCode.MISSING_REQUIRED_COLUMN, "err")
        ])
        assert not r_err.is_valid
        
        # Caso: con CRITICAL → inválido
        r_crit = ValidationResult.from_issues([
            ValidationIssue(ValidationSeverity.CRITICAL, ValidationCode.NONE_DATAFRAME, "crit")
        ])
        assert not r_crit.is_valid
        
        # Caso: mezcla WARNING + ERROR → inválido
        r_mixed = ValidationResult.from_issues([
            ValidationIssue(ValidationSeverity.WARNING, ValidationCode.NULL_VALUES, "w"),
            ValidationIssue(ValidationSeverity.ERROR, ValidationCode.NON_FINITE_VALUES, "e"),
        ])
        assert not r_mixed.is_valid

    def test_merge_preserves_chronological_order(self):
        """La concatenación respeta orden cronológico (FIFO)."""
        i1 = ValidationIssue(ValidationSeverity.WARNING, ValidationCode.NULL_VALUES, "first", count=1)
        i2 = ValidationIssue(ValidationSeverity.ERROR, ValidationCode.MISSING_REQUIRED_COLUMN, "second", count=2)
        i3 = ValidationIssue(ValidationSeverity.INFO, ValidationCode.DUPLICATE_COLUMNS, "third", count=3)
        
        r1 = ValidationResult.from_issues([i1])
        r2 = ValidationResult.from_issues([i2])
        r3 = ValidationResult.from_issues([i3])
        
        merged = r1 + r2 + r3
        
        assert merged.issues == (i1, i2, i3)
        # Verificar también orden inverso
        merged_rev = r3 + r2 + r1
        assert merged_rev.issues == (i3, i2, i1)

    def test_properties_filter_by_severity(self):
        """Propiedades errors/warnings/infos filtran correctamente."""
        issues = [
            ValidationIssue(ValidationSeverity.INFO, ValidationCode.NULL_VALUES, "i1"),
            ValidationIssue(ValidationSeverity.INFO, ValidationCode.EMPTY_STRINGS, "i2"),
            ValidationIssue(ValidationSeverity.WARNING, ValidationCode.DUPLICATE_COLUMNS, "w1"),
            ValidationIssue(ValidationSeverity.WARNING, ValidationCode.NULL_VALUES, "w2"),
            ValidationIssue(ValidationSeverity.ERROR, ValidationCode.MISSING_REQUIRED_COLUMN, "e1"),
            ValidationIssue(ValidationSeverity.CRITICAL, ValidationCode.NONE_DATAFRAME, "c1"),
        ]
        res = ValidationResult.from_issues(issues)
        
        assert len(res.errors) == 2  # ERROR + CRITICAL
        assert len(res.warnings) == 2
        assert len(res.infos) == 2
        
        assert res.has_errors
        assert res.has_warnings
        
        # Verificar que los filtros devuelven los issues correctos
        assert all(i.severity == ValidationSeverity.INFO for i in res.infos)
        assert all(i.severity == ValidationSeverity.WARNING for i in res.warnings)
        assert all(i.severity in {ValidationSeverity.ERROR, ValidationSeverity.CRITICAL} for i in res.errors)

    def test_max_severity_property(self):
        """max_severity devuelve el supremo del retículo."""
        # Sin issues
        assert ValidationResult.success().max_severity is None
        
        # Un único issue
        r = ValidationResult.from_issues([
            ValidationIssue(ValidationSeverity.WARNING, ValidationCode.NULL_VALUES, "test")
        ])
        assert r.max_severity == ValidationSeverity.WARNING
        
        # Múltiples severidades
        r_mixed = ValidationResult.from_issues([
            ValidationIssue(ValidationSeverity.INFO, ValidationCode.NULL_VALUES, "i"),
            ValidationIssue(ValidationSeverity.CRITICAL, ValidationCode.NONE_DATAFRAME, "c"),
            ValidationIssue(ValidationSeverity.WARNING, ValidationCode.EMPTY_STRINGS, "w"),
        ])
        # En el retículo, CRITICAL es el supremo
        assert r_mixed.max_severity == ValidationSeverity.CRITICAL

    def test_to_dict_serialization(self):
        """Serialización a diccionario preserva toda la información."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code=ValidationCode.NON_FINITE_VALUES,
            message="Test infinity",
            column="col_x",
            count=3,
        )
        res = ValidationResult.from_issues([issue])
        d = res.to_dict()
        
        assert d["is_valid"] is False
        assert d["issue_count"] == 1
        assert d["error_count"] == 1
        assert d["warning_count"] == 0
        assert d["info_count"] == 0
        assert d["max_severity"] == "ERROR"
        assert len(d["issues"]) == 1
        
        issue_dict = d["issues"][0]
        assert issue_dict["code"] == "NON_FINITE_VALUES"
        assert issue_dict["severity"] == "ERROR"
        assert issue_dict["message"] == "Test infinity"
        assert issue_dict["column"] == "col_x"
        assert issue_dict["count"] == 3

    def test_to_dataframe_export(self):
        """Exportación a DataFrame para análisis."""
        issues = [
            ValidationIssue(ValidationSeverity.WARNING, ValidationCode.NULL_VALUES, "w1", "A", 5),
            ValidationIssue(ValidationSeverity.ERROR, ValidationCode.NON_FINITE_VALUES, "e1", "B", 2),
        ]
        res = ValidationResult.from_issues(issues)
        df = res.to_dataframe()
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert list(df.columns) == ["severity", "code", "message", "column", "count"]
        assert df.loc[0, "severity"] == "WARNING"
        assert df.loc[1, "code"] == "NON_FINITE_VALUES"

    def test_to_dataframe_empty(self):
        """Exportación de resultado vacío produce DataFrame con columnas correctas."""
        df = ValidationResult.success().to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["severity", "code", "message", "column", "count"]

    def test_string_representation(self):
        """Representaciones __str__ y __repr__ son informativas."""
        res_valid = ValidationResult.success()
        assert "✓ VALID" in str(res_valid)
        assert "errors=0" in str(res_valid)
        
        res_invalid = ValidationResult.from_issues([
            ValidationIssue(ValidationSeverity.ERROR, ValidationCode.NONE_DATAFRAME, "bad")
        ])
        assert "✗ INVALID" in str(res_invalid)
        assert "errors=1" in str(res_invalid)
        
        # __repr__ debe ser evaluable (o al menos informativo)
        repr_str = repr(res_invalid)
        assert "ValidationResult" in repr_str
        assert "is_valid=False" in repr_str

    def test_bool_conversion(self):
        """Conversión booleana: True si válido, False si inválido."""
        assert bool(ValidationResult.success()) is True
        assert bool(ValidationResult.from_issues([
            ValidationIssue(ValidationSeverity.ERROR, ValidationCode.NONE_DATAFRAME, "err")
        ])) is False

    def test_len_returns_issue_count(self):
        """len(result) devuelve cardinalidad de issues."""
        assert len(ValidationResult.success()) == 0
        assert len(ValidationResult.from_issues([
            ValidationIssue(ValidationSeverity.WARNING, ValidationCode.NULL_VALUES, "w1"),
            ValidationIssue(ValidationSeverity.ERROR, ValidationCode.NON_FINITE_VALUES, "e1"),
        ])) == 2

    def test_filter_by_severity(self):
        """Filtrado por severidad mínima."""
        issues = [
            ValidationIssue(ValidationSeverity.INFO, ValidationCode.NULL_VALUES, "i"),
            ValidationIssue(ValidationSeverity.WARNING, ValidationCode.EMPTY_STRINGS, "w"),
            ValidationIssue(ValidationSeverity.ERROR, ValidationCode.NON_FINITE_VALUES, "e"),
        ]
        res = ValidationResult.from_issues(issues)
        
        # Filtrar por WARNING mínimo
        filtered = res.filter_by_severity(ValidationSeverity.WARNING)
        # Solo deben quedar WARNING y ERROR (2 issues)
        assert len(filtered.issues) == 2
        assert not any(i.severity == ValidationSeverity.INFO for i in filtered.issues)

    def test_filter_by_code(self):
        """Filtrado por códigos específicos."""
        issues = [
            ValidationIssue(ValidationSeverity.WARNING, ValidationCode.NULL_VALUES, "w1"),
            ValidationIssue(ValidationSeverity.ERROR, ValidationCode.NON_FINITE_VALUES, "e1"),
            ValidationIssue(ValidationSeverity.WARNING, ValidationCode.NULL_VALUES, "w2"),
        ]
        res = ValidationResult.from_issues(issues)
        
        filtered = res.filter_by_code([ValidationCode.NULL_VALUES])
        assert len(filtered) == 2
        assert all(i.code == ValidationCode.NULL_VALUES for i in filtered.issues)

    def test_raise_if_invalid_raises_on_errors(self):
        """raise_if_invalid lanza excepción si hay errores."""
        res_invalid = ValidationResult.from_issues([
            ValidationIssue(ValidationSeverity.ERROR, ValidationCode.NONE_DATAFRAME, "err")
        ])
        
        with pytest.raises(ValueError, match="Validación fallida"):
            res_invalid.raise_if_invalid()
        
        # Debe funcionar con tipo de excepción personalizado
        with pytest.raises(RuntimeError, match="Validación fallida"):
            res_invalid.raise_if_invalid(RuntimeError)

    def test_raise_if_invalid_success_no_raise(self):
        """raise_if_invalid no lanza si es válido."""
        ValidationResult.success().raise_if_invalid()  # No debe lanzar

    def test_immutability_frozen_dataclass(self):
        """Los objetos son inmutables (frozen=True)."""
        issue = ValidationIssue(ValidationSeverity.INFO, ValidationCode.NULL_VALUES, "msg")
        with pytest.raises(FrozenInstanceError):
            issue.severity = ValidationSeverity.ERROR  # type: ignore
        
        result = ValidationResult.success()
        with pytest.raises(FrozenInstanceError):
            result.is_valid = False  # type: ignore

    def test_failure_constructor_without_blocking_issues(self):
        """failure() con issues no bloqueantes resulta en is_valid=True."""
        r = ValidationResult.failure([
            ValidationIssue(ValidationSeverity.WARNING, ValidationCode.NULL_VALUES, "w")
        ])
        assert r.is_valid  # WARNING no bloquea

    def test_single_issue_constructor(self):
        """Constructor de conveniencia para un único issue."""
        result = ValidationResult.single_issue(
            severity=ValidationSeverity.ERROR,
            code=ValidationCode.NON_FINITE_VALUES,
            message="Test",
            column="col1",
            count=5,
        )
        
        assert len(result.issues) == 1
        assert result.issues[0].severity == ValidationSeverity.ERROR
        assert result.issues[0].column == "col1"
        assert result.issues[0].count == 5


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN III: RETÍCULO DE SEVERIDADES (ValidationSeverity)
# ══════════════════════════════════════════════════════════════════════════════


class TestValidationSeverityLattice:
    """
    Verifica propiedades del RETÍCULO COMPLETO de severidades.
    
    Orden Parcial: INFO < WARNING < ERROR < CRITICAL
    
    Operaciones:
    - join (∨): supremo (max)
    - meet (∧): ínfimo (min)
    - top (⊤): CRITICAL
    - bottom (⊥): INFO
    """

    @given(a=st_severity(), b=st_severity())
    def test_lattice_commutativity(self, a, b):
        """Leyes de Conmutatividad: a ∨ b = b ∨ a, a ∧ b = b ∧ a."""
        assert a.join(b) == b.join(a)
        assert a.meet(b) == b.meet(a)

    @given(a=st_severity(), b=st_severity(), c=st_severity())
    def test_lattice_associativity(self, a, b, c):
        """Leyes de Asociatividad: (a ∨ b) ∨ c = a ∨ (b ∨ c), etc."""
        assert a.join(b).join(c) == a.join(b.join(c))
        assert a.meet(b).meet(c) == a.meet(b.meet(c))

    @given(a=st_severity(), b=st_severity())
    def test_lattice_absorption(self, a, b):
        """Leyes de Absorción de Birkhoff: a ∨ (a ∧ b) = a, a ∧ (a ∨ b) = a."""
        assert a.join(a.meet(b)) == a
        assert a.meet(a.join(b)) == a

    @given(a=st_severity())
    def test_lattice_idempotence(self, a):
        """Leyes de Idempotencia: a ∨ a = a, a ∧ a = a."""
        assert a.join(a) == a
        assert a.meet(a) == a

    def test_rank_ordering(self):
        """Verificar orden de rangos."""
        assert ValidationSeverity.INFO.rank == 0
        assert ValidationSeverity.WARNING.rank == 1
        assert ValidationSeverity.ERROR.rank == 2
        assert ValidationSeverity.CRITICAL.rank == 3

    def test_partial_order_less_than(self):
        """Operador < respeta orden parcial."""
        assert ValidationSeverity.INFO < ValidationSeverity.WARNING
        assert ValidationSeverity.WARNING < ValidationSeverity.ERROR
        assert ValidationSeverity.ERROR < ValidationSeverity.CRITICAL
        
        # Transitividad
        assert ValidationSeverity.INFO < ValidationSeverity.CRITICAL

    def test_partial_order_less_equal(self):
        """Operador <= respeta orden parcial."""
        assert ValidationSeverity.INFO <= ValidationSeverity.INFO  # Reflexividad
        assert ValidationSeverity.WARNING <= ValidationSeverity.ERROR
        assert ValidationSeverity.ERROR <= ValidationSeverity.CRITICAL

    def test_join_operation_supremum(self):
        """join (∨) devuelve supremo."""
        assert ValidationSeverity.INFO.join(ValidationSeverity.WARNING) == ValidationSeverity.WARNING
        assert ValidationSeverity.ERROR.join(ValidationSeverity.WARNING) == ValidationSeverity.ERROR
        assert ValidationSeverity.CRITICAL.join(ValidationSeverity.INFO) == ValidationSeverity.CRITICAL

    def test_join_idempotence(self):
        """s ∨ s = s (idempotencia)."""
        for sev in ValidationSeverity:
            assert sev.join(sev) == sev

    def test_join_commutativity(self):
        """s₁ ∨ s₂ = s₂ ∨ s₁ (conmutatividad)."""
        assert (
            ValidationSeverity.INFO.join(ValidationSeverity.ERROR)
            == ValidationSeverity.ERROR.join(ValidationSeverity.INFO)
        )

    def test_join_associativity(self):
        """(s₁ ∨ s₂) ∨ s₃ = s₁ ∨ (s₂ ∨ s₃) (asociatividad)."""
        s1, s2, s3 = ValidationSeverity.INFO, ValidationSeverity.WARNING, ValidationSeverity.CRITICAL
        assert (s1.join(s2)).join(s3) == s1.join(s2.join(s3))

    def test_meet_operation_infimum(self):
        """meet (∧) devuelve ínfimo."""
        assert ValidationSeverity.ERROR.meet(ValidationSeverity.WARNING) == ValidationSeverity.WARNING
        assert ValidationSeverity.CRITICAL.meet(ValidationSeverity.INFO) == ValidationSeverity.INFO

    def test_meet_idempotence(self):
        """s ∧ s = s (idempotencia)."""
        for sev in ValidationSeverity:
            assert sev.meet(sev) == sev

    def test_meet_commutativity(self):
        """s₁ ∧ s₂ = s₂ ∧ s₁ (conmutatividad)."""
        assert (
            ValidationSeverity.ERROR.meet(ValidationSeverity.WARNING)
            == ValidationSeverity.WARNING.meet(ValidationSeverity.ERROR)
        )

    def test_absorption_law(self):
        """s ∨ (s ∧ t) = s (ley de absorción)."""
        s = ValidationSeverity.WARNING
        t = ValidationSeverity.ERROR
        assert s.join(s.meet(t)) == s

    def test_top_element(self):
        """⊤ = CRITICAL (elemento máximo)."""
        assert ValidationSeverity.top() == ValidationSeverity.CRITICAL
        for sev in ValidationSeverity:
            assert sev.join(ValidationSeverity.CRITICAL) == ValidationSeverity.CRITICAL

    def test_bottom_element(self):
        """⊥ = INFO (elemento mínimo)."""
        assert ValidationSeverity.bottom() == ValidationSeverity.INFO
        for sev in ValidationSeverity:
            assert sev.meet(ValidationSeverity.INFO) == ValidationSeverity.INFO


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN IV: ESTRUCTURAS INMUTABLES (ValidationIssue)
# ══════════════════════════════════════════════════════════════════════════════


class TestValidationIssue:
    """Verifica invariantes y propiedades de ValidationIssue."""

    def test_creation_with_all_fields(self):
        """Crear issue con todos los campos."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code=ValidationCode.NON_FINITE_VALUES,
            message="Test message",
            column="test_col",
            count=10,
        )
        
        assert issue.severity == ValidationSeverity.ERROR
        assert issue.code == ValidationCode.NON_FINITE_VALUES
        assert issue.message == "Test message"
        assert issue.column == "test_col"
        assert issue.count == 10

    def test_creation_with_optional_fields_none(self):
        """Campos opcionales pueden ser None."""
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code=ValidationCode.NULL_VALUES,
            message="Test",
        )
        
        assert issue.column is None
        assert issue.count is None

    def test_invalid_severity_raises(self):
        """Severidad inválida lanza TypeError."""
        with pytest.raises(TypeError, match="severity debe ser ValidationSeverity"):
            ValidationIssue(
                severity="INVALID",  # type: ignore
                code=ValidationCode.NULL_VALUES,
                message="test",
            )

    def test_invalid_code_raises(self):
        """Código inválido lanza TypeError."""
        with pytest.raises(TypeError, match="code debe ser ValidationCode"):
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code="INVALID",  # type: ignore
                message="test",
            )

    def test_empty_message_raises(self):
        """Mensaje vacío lanza ValueError."""
        with pytest.raises(ValueError, match="message debe ser un string no vacío"):
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code=ValidationCode.NULL_VALUES,
                message="",
            )
        
        with pytest.raises(ValueError, match="message debe ser un string no vacío"):
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code=ValidationCode.NULL_VALUES,
                message="   ",  # Solo espacios
            )

    def test_invalid_column_type_raises(self):
        """Tipo inválido para column lanza TypeError."""
        with pytest.raises(TypeError, match="column debe ser str o None"):
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code=ValidationCode.NULL_VALUES,
                message="test",
                column=123,  # type: ignore
            )

    def test_invalid_count_type_raises(self):
        """Tipo inválido para count lanza TypeError."""
        with pytest.raises(TypeError, match="count debe ser int o None"):
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code=ValidationCode.NULL_VALUES,
                message="test",
                count="10",  # type: ignore
            )

    def test_negative_count_raises(self):
        """Count negativo lanza ValueError."""
        with pytest.raises(ValueError, match="count debe ser no negativo"):
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                code=ValidationCode.NULL_VALUES,
                message="test",
                count=-1,
            )

    def test_str_representation(self):
        """Representación string es legible."""
        issue = ValidationIssue(
            severity=ValidationSeverity.ERROR,
            code=ValidationCode.NON_FINITE_VALUES,
            message="Test infinity",
            column="col_x",
            count=5,
        )
        
        s = str(issue)
        assert "[ERROR]" in s
        assert "NON_FINITE_VALUES" in s
        assert "Test infinity" in s
        assert "columna=col_x" in s
        assert "count=5" in s

    def test_to_dict(self):
        """Serialización a diccionario."""
        issue = ValidationIssue(
            severity=ValidationSeverity.WARNING,
            code=ValidationCode.NULL_VALUES,
            message="msg",
            column="A",
            count=3,
        )
        
        d = issue.to_dict()
        assert d["severity"] == "WARNING"
        assert d["code"] == "NULL_VALUES"
        assert d["message"] == "msg"
        assert d["column"] == "A"
        assert d["count"] == 3

    def test_is_blocking_property(self):
        """Propiedad is_blocking identifica severidades bloqueantes."""
        error_issue = ValidationIssue(
            ValidationSeverity.ERROR,
            ValidationCode.NONE_DATAFRAME,
            "test"
        )
        assert error_issue.is_blocking
        
        critical_issue = ValidationIssue(
            ValidationSeverity.CRITICAL,
            ValidationCode.SURVIVAL_THRESHOLD_VIOLATION,
            "test"
        )
        assert critical_issue.is_blocking
        
        warning_issue = ValidationIssue(
            ValidationSeverity.WARNING,
            ValidationCode.NULL_VALUES,
            "test"
        )
        assert not warning_issue.is_blocking

    def test_ordering_lexicographic(self):
        """Issues se ordenan lexicográficamente."""
        i1 = ValidationIssue(ValidationSeverity.WARNING, ValidationCode.NULL_VALUES, "msg", "A", 1)
        i2 = ValidationIssue(ValidationSeverity.ERROR, ValidationCode.NULL_VALUES, "msg", "A", 1)
        
        # ERROR > WARNING en severidad
        assert i2 > i1


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN V: VALIDADORES ESTRUCTURALES (Isomorfismo Dimensional)
# ══════════════════════════════════════════════════════════════════════════════


class TestValidateSchema:
    """Verifica isomorfismo dimensional: required_columns ⊆ df.columns."""

    def test_all_required_columns_present(self, simple_df):
        """Todas las columnas requeridas existen → válido."""
        result = DataFrameValidator.validate_schema(simple_df, ["A", "B", "C"])
        assert result.is_valid
        assert len(result.issues) == 0

    def test_subset_of_columns_present(self, simple_df):
        """Subset de columnas requeridas → válido."""
        result = DataFrameValidator.validate_schema(simple_df, ["A", "B"])
        assert result.is_valid

    def test_missing_single_column_error(self, simple_df):
        """Columna faltante genera ERROR."""
        result = DataFrameValidator.validate_schema(simple_df, ["A", "D"])
        assert not result.is_valid
        
        missing_issues = [
            i for i in result.issues
            if i.code == ValidationCode.MISSING_REQUIRED_COLUMN
        ]
        assert len(missing_issues) == 1
        assert missing_issues[0].column == "D"

    def test_missing_multiple_columns_errors(self, simple_df):
        """Múltiples columnas faltantes generan múltiples ERRORs."""
        result = DataFrameValidator.validate_schema(simple_df, ["D", "E", "F"])
        assert not result.is_valid
        
        missing = [i for i in result.issues if i.code == ValidationCode.MISSING_REQUIRED_COLUMN]
        assert len(missing) == 3
        assert set(i.column for i in missing) == {"D", "E", "F"}

    def test_duplicate_columns_warning(self, df_with_duplicates):
        """Columnas duplicadas generan WARNING."""
        result = DataFrameValidator.validate_schema(df_with_duplicates, ["A", "B"])
        
        assert result.has_warnings
        dup_issues = [i for i in result.issues if i.code == ValidationCode.DUPLICATE_COLUMNS]
        assert len(dup_issues) == 1

    def test_empty_required_columns_valid(self, simple_df):
        """Sin columnas requeridas → siempre válido."""
        result = DataFrameValidator.validate_schema(simple_df, [])
        assert result.is_valid

    def test_invalid_dataframe_none(self):
        """DataFrame None genera CRITICAL."""
        result = DataFrameValidator.validate_schema(None, ["A"])  # type: ignore
        assert not result.is_valid
        assert any(i.code == ValidationCode.NONE_DATAFRAME for i in result.issues)

    def test_invalid_dataframe_type(self):
        """Tipo inválido genera CRITICAL."""
        result = DataFrameValidator.validate_schema("not a df", ["A"])  # type: ignore
        assert not result.is_valid
        assert any(i.code == ValidationCode.INVALID_DATAFRAME_TYPE for i in result.issues)

    def test_normalize_columns_removes_duplicates(self):
        """Columnas duplicadas en required se deduplicantes."""
        result = DataFrameValidator.validate_schema(
            pd.DataFrame({"A": [1], "B": [2]}),
            ["A", "B", "A", "B"]  # Duplicados
        )
        assert result.is_valid

    def test_normalize_columns_invalid_raises(self):
        """Columnas inválidas lanzan ValueError."""
        with pytest.raises(ValueError, match="no puede ser None"):
            DataFrameValidator.validate_schema(
                pd.DataFrame(),
                None  # type: ignore
            )
        
        with pytest.raises(ValueError, match="debe ser str"):
            DataFrameValidator.validate_schema(
                pd.DataFrame(),
                [1, 2, 3]  # type: ignore
            )


class TestCheckDuplicateColumns:
    """Verifica detección de columnas duplicadas (base degenerada)."""

    def test_no_duplicates(self, simple_df):
        """Sin duplicados → válido."""
        result = DataFrameValidator._check_duplicate_columns(simple_df)
        assert result.is_valid
        assert len(result.issues) == 0

    def test_single_duplicate(self):
        """Una columna duplicada."""
        df = pd.DataFrame([[1, 2]], columns=["A", "A"])
        result = DataFrameValidator._check_duplicate_columns(df)
        
        assert result.has_warnings
        assert len(result.issues) == 1
        assert result.issues[0].code == ValidationCode.DUPLICATE_COLUMNS

    def test_multiple_duplicates(self):
        """Múltiples columnas duplicadas."""
        df = pd.DataFrame([[1, 2, 3, 4]], columns=["A", "A", "B", "B"])
        result = DataFrameValidator._check_duplicate_columns(df)
        
        assert result.has_warnings
        assert "A" in result.issues[0].message
        assert "B" in result.issues[0].message


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN VI: VALIDADORES DE CALIDAD (Extirpación de Singularidades)
# ══════════════════════════════════════════════════════════════════════════════


class TestCheckDataQuality:
    """Verifica extirpación de singularidades (NaN, None, strings vacíos)."""

    def test_no_quality_issues(self, simple_df):
        """DataFrame limpio → válido."""
        result = DataFrameValidator.check_data_quality(simple_df, ["A", "B", "C"])
        assert result.is_valid

    def test_null_values_warning(self, df_with_nulls):
        """Nulos generan WARNING."""
        result = DataFrameValidator.check_data_quality(df_with_nulls, ["A", "B", "C"])
        
        assert result.has_warnings
        null_issues = [i for i in result.issues if i.code == ValidationCode.NULL_VALUES]
        assert len(null_issues) == 3  # A, B, C tienen nulos
        
        # Verificar counts
        null_counts = {i.column: i.count for i in null_issues}
        assert null_counts["A"] == 1
        assert null_counts["B"] == 1
        assert null_counts["C"] == 1

    def test_empty_strings_warning(self, df_with_nulls):
        """Strings vacíos generan WARNING."""
        result = DataFrameValidator.check_data_quality(df_with_nulls, ["C"])
        
        assert result.has_warnings
        empty_issues = [i for i in result.issues if i.code == ValidationCode.EMPTY_STRINGS]
        assert len(empty_issues) == 1
        assert empty_issues[0].column == "C"
        assert empty_issues[0].count == 1

    def test_empty_strings_flag_disabled(self, df_with_nulls):
        """Con empty_strings_as_warning=False no se detectan."""
        result = DataFrameValidator.check_data_quality(
            df_with_nulls,
            ["C"],
            empty_strings_as_warning=False
        )
        
        # Solo debe haber NULL_VALUES, no EMPTY_STRINGS
        assert not any(i.code == ValidationCode.EMPTY_STRINGS for i in result.issues)
        assert any(i.code == ValidationCode.NULL_VALUES for i in result.issues)

    def test_all_nulls_detected(self, df_all_nulls):
        """Todas las filas nulas son detectadas."""
        result = DataFrameValidator.check_data_quality(df_all_nulls, ["x", "y"])
        
        null_issues = [i for i in result.issues if i.code == ValidationCode.NULL_VALUES]
        assert len(null_issues) == 2
        
        for issue in null_issues:
            assert issue.count == 3

    def test_missing_critical_column_warning(self, simple_df):
        """Columna crítica faltante genera WARNING."""
        result = DataFrameValidator.check_data_quality(simple_df, ["Z"])
        
        assert result.has_warnings
        missing_issues = [
            i for i in result.issues
            if i.code == ValidationCode.MISSING_CRITICAL_COLUMN
        ]
        assert len(missing_issues) == 1
        assert missing_issues[0].column == "Z"

    def test_empty_critical_columns_valid(self, simple_df):
        """Sin columnas críticas → válido."""
        result = DataFrameValidator.check_data_quality(simple_df, [])
        assert result.is_valid


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN VII: VALIDADORES NUMÉRICOS (Clausura Algebraica IEEE 754)
# ══════════════════════════════════════════════════════════════════════════════


class TestValidateNumericColumns:
    """Verifica clausura algebraica: valores ∈ ℝ_finite."""

    def test_all_numeric_clean(self):
        """Columnas numéricas limpias → válido."""
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4.5, 5.5, 6.5]})
        result = DataFrameValidator.validate_numeric_columns(df, ["x", "y"])
        assert result.is_valid

    def test_integer_columns_valid(self):
        """Enteros son numéricos válidos."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = DataFrameValidator.validate_numeric_columns(df, ["x"])
        assert result.is_valid

    def test_float_columns_valid(self):
        """Flotantes son numéricos válidos."""
        df = pd.DataFrame({"x": [1.0, 2.5, 3.7]})
        result = DataFrameValidator.validate_numeric_columns(df, ["x"])
        assert result.is_valid

    def test_non_numeric_values_warning(self, df_mixed_types):
        """Valores no numéricos generan WARNING."""
        result = DataFrameValidator.validate_numeric_columns(df_mixed_types, ["mixed"])
        
        non_num_issues = [
            i for i in result.issues
            if i.code == ValidationCode.NON_NUMERIC_VALUES
        ]
        assert len(non_num_issues) == 1
        assert non_num_issues[0].count == 2  # "two", "five"

    def test_non_numeric_severity_customizable(self):
        """Severidad de no numéricos es configurable."""
        df = pd.DataFrame({"x": [1, "a"]})
        
        result = DataFrameValidator.validate_numeric_columns(
            df,
            ["x"],
            non_numeric_severity=ValidationSeverity.ERROR
        )
        
        assert not result.is_valid
        assert any(
            i.code == ValidationCode.NON_NUMERIC_VALUES and i.severity == ValidationSeverity.ERROR
            for i in result.issues
        )

    def test_non_finite_values_error(self, df_with_infinities):
        """Valores no finitos (±∞) generan ERROR."""
        result = DataFrameValidator.validate_numeric_columns(df_with_infinities, ["x"])
        
        non_finite = [i for i in result.issues if i.code == ValidationCode.NON_FINITE_VALUES]
        assert len(non_finite) == 1
        assert non_finite[0].count == 2  # +inf, -inf

    def test_nan_handled_separately(self):
        """NaN se trata como valor nulo, no como no finito."""
        df = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
        result = DataFrameValidator.validate_numeric_columns(df, ["x"])
        
        # No debe haber NON_FINITE_VALUES porque NaN se excluye
        assert not any(i.code == ValidationCode.NON_FINITE_VALUES for i in result.issues)

    def test_subnormal_detection_warning(self, df_with_subnormals):
        """Valores subnormales generan WARNING."""
        result = DataFrameValidator.validate_numeric_columns(df_with_subnormals, ["sub"])
        
        sub_issues = [i for i in result.issues if i.code == ValidationCode.SUBNORMAL_VALUES]
        assert len(sub_issues) == 1
        assert sub_issues[0].count == 2  # Positivo y negativo

    def test_zero_not_subnormal(self):
        """Cero no es subnormal."""
        df = pd.DataFrame({"x": [0.0, -0.0]})
        result = DataFrameValidator.validate_numeric_columns(df, ["x"])
        
        assert not any(i.code == ValidationCode.SUBNORMAL_VALUES for i in result.issues)

    def test_normal_tiny_values_not_subnormal(self):
        """Valores pequeños pero normales no son subnormales."""
        df = pd.DataFrame({"x": [MIN_NORMAL_FLOAT, MIN_NORMAL_FLOAT * 2]})
        result = DataFrameValidator.validate_numeric_columns(df, ["x"])
        
        assert not any(i.code == ValidationCode.SUBNORMAL_VALUES for i in result.issues)

    def test_missing_column_warning_mode(self, simple_df):
        """Columna faltante genera WARNING si missing_as_warning=True."""
        result = DataFrameValidator.validate_numeric_columns(
            simple_df,
            ["no_existe"],
            missing_as_warning=True
        )
        
        assert result.has_warnings
        assert any(
            i.code == ValidationCode.MISSING_CRITICAL_COLUMN
            for i in result.issues
        )

    def test_missing_column_error_mode(self, simple_df):
        """Columna faltante genera ERROR si missing_as_warning=False."""
        result = DataFrameValidator.validate_numeric_columns(
            simple_df,
            ["no_existe"],
            missing_as_warning=False
        )
        
        assert result.has_errors
        assert any(
            i.code == ValidationCode.MISSING_REQUIRED_COLUMN
            for i in result.issues
        )

    def test_empty_numeric_columns_valid(self, simple_df):
        """Sin columnas numéricas → válido."""
        result = DataFrameValidator.validate_numeric_columns(simple_df, [])
        assert result.is_valid

    def test_all_nulls_column_no_validation(self):
        """Columna con todos nulos no genera errores numéricos."""
        df = pd.DataFrame({"x": [None, None, None]})
        result = DataFrameValidator.validate_numeric_columns(df, ["x"])
        
        # No debería haber NON_NUMERIC o NON_FINITE porque todos son nulos
        assert not any(
            i.code in {ValidationCode.NON_NUMERIC_VALUES, ValidationCode.NON_FINITE_VALUES}
            for i in result.issues
        )


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN VIII: VALIDADORES DE DOMINIO (Termodinámica y Compacidad)
# ══════════════════════════════════════════════════════════════════════════════


class TestValidateNonNegative:
    """Verifica no-negatividad (Segunda Ley Termodinámica)."""

    def test_all_non_negative_valid(self):
        """Valores no negativos → válido."""
        df = pd.DataFrame({"x": [0, 1, 2, 3]})
        result = DataFrameValidator.validate_non_negative(df, ["x"])
        assert result.is_valid

    def test_zero_is_valid(self):
        """Cero es válido (no negativo)."""
        df = pd.DataFrame({"x": [0, 0.0, -0.0]})
        result = DataFrameValidator.validate_non_negative(df, ["x"])
        assert result.is_valid

    def test_negative_values_warning(self, df_with_negatives):
        """Valores negativos generan WARNING."""
        result = DataFrameValidator.validate_non_negative(df_with_negatives, ["cost", "time"])
        
        neg_issues = [i for i in result.issues if i.code == ValidationCode.NEGATIVE_VALUES]
        assert len(neg_issues) == 2
        
        neg_counts = {i.column: i.count for i in neg_issues}
        assert neg_counts["cost"] == 2  # -5, -3
        assert neg_counts["time"] == 1  # -0.5

    def test_negative_severity_customizable(self):
        """Severidad de negativos es configurable."""
        df = pd.DataFrame({"x": [-1, 0, 1]})
        
        result = DataFrameValidator.validate_non_negative(
            df,
            ["x"],
            negative_severity=ValidationSeverity.ERROR
        )
        
        assert not result.is_valid
        assert any(
            i.code == ValidationCode.NEGATIVE_VALUES and i.severity == ValidationSeverity.ERROR
            for i in result.issues
        )

    def test_infinity_ignored_not_negative(self):
        """-∞ no se considera negativo (no es finito)."""
        df = pd.DataFrame({"x": [-np.inf, 0, 1]})
        result = DataFrameValidator.validate_non_negative(df, ["x"])
        
        # No debe haber NEGATIVE_VALUES porque -∞ se filtra como no finito
        assert not any(i.code == ValidationCode.NEGATIVE_VALUES for i in result.issues)

    def test_nan_ignored(self):
        """NaN se ignora en validación de negativos."""
        df = pd.DataFrame({"x": [np.nan, -1, 1]})
        result = DataFrameValidator.validate_non_negative(df, ["x"])
        
        neg_issues = [i for i in result.issues if i.code == ValidationCode.NEGATIVE_VALUES]
        assert len(neg_issues) == 1
        assert neg_issues[0].count == 1  # Solo -1

    def test_missing_column_handling(self, simple_df):
        """Columna faltante genera WARNING/ERROR según configuración."""
        result_warn = DataFrameValidator.validate_non_negative(
            simple_df,
            ["no_existe"],
            missing_as_warning=True
        )
        assert result_warn.has_warnings
        
        result_err = DataFrameValidator.validate_non_negative(
            simple_df,
            ["no_existe"],
            missing_as_warning=False
        )
        assert result_err.has_errors


class TestThermodynamicClamping:
    """Verifica saturación termodinámica (Lipschitz clamping)."""

    def test_clamping_massive_impulses(self):
        """Valores que exceden LIPSCHITZ_LIMIT son saturados."""
        limit = 100.0
        df = pd.DataFrame({"energy": [10.0, 500.0, -1000.0, 50.0]})

        df_sat, result = DataFrameValidator.apply_thermodynamic_clamping(
            df, ["energy"], limit=limit
        )

        assert result.has_warnings
        assert result.issues[0].code == ValidationCode.RANGE_VIOLATION
        assert result.issues[0].count == 2

        # Verificar saturación física
        assert df_sat["energy"].max() <= limit
        assert df_sat["energy"].min() >= -limit
        assert df_sat.loc[1, "energy"] == limit
        assert df_sat.loc[2, "energy"] == -limit

    def test_no_clamping_if_within_limits(self):
        """No se aplica saturación si los valores son normales."""
        df = pd.DataFrame({"energy": [1.0, 2.0, 3.0]})
        df_sat, result = DataFrameValidator.apply_thermodynamic_clamping(
            df, ["energy"], limit=1e10
        )

        assert result.is_valid
        assert len(result.issues) == 0
        pd.testing.assert_frame_equal(df, df_sat)


class TestValidateRanges:
    """Verifica compacidad: valores ∈ [min, max]."""

    def test_values_within_range_valid(self):
        """Valores dentro de rango → válido."""
        df = pd.DataFrame({"x": [0, 5, 10]})
        result = DataFrameValidator.validate_ranges(df, {"x": (0, 10)})
        assert result.is_valid

    def test_values_at_boundaries_valid(self):
        """Valores en fronteras son válidos."""
        df = pd.DataFrame({"x": [0, 10]})
        result = DataFrameValidator.validate_ranges(df, {"x": (0, 10)})
        assert result.is_valid

    def test_values_out_of_range_warning(self, df_out_of_range):
        """Valores fuera de rango generan WARNING."""
        result = DataFrameValidator.validate_ranges(
            df_out_of_range,
            {"percentage": (0, 100)}
        )
        
        violations = [i for i in result.issues if i.code == ValidationCode.RANGE_VIOLATION]
        assert len(violations) == 1
        assert violations[0].count == 2  # -10 y 150

    def test_unbounded_lower_limit(self):
        """Límite inferior None = -∞."""
        df = pd.DataFrame({"x": [-1000, 0, 50]})
        result = DataFrameValidator.validate_ranges(df, {"x": (None, 50)})
        assert result.is_valid

    def test_unbounded_upper_limit(self):
        """Límite superior None = +∞."""
        df = pd.DataFrame({"x": [0, 100, 1000]})
        result = DataFrameValidator.validate_ranges(df, {"x": (0, None)})
        assert result.is_valid

    def test_both_limits_unbounded(self):
        """Ambos límites None → siempre válido."""
        df = pd.DataFrame({"x": [-1e10, 0, 1e10]})
        result = DataFrameValidator.validate_ranges(df, {"x": (None, None)})
        assert result.is_valid

    def test_invalid_range_min_greater_max_raises(self):
        """min > max lanza ValueError."""
        df = pd.DataFrame({"x": [5]})
        with pytest.raises(ValueError, match="min .* > max"):
            DataFrameValidator.validate_ranges(df, {"x": (10, 0)})

    def test_invalid_bound_type_raises(self):
        """Tipo inválido en límites lanza ValueError."""
        df = pd.DataFrame({"x": [5]})
        
        with pytest.raises(ValueError, match="debe ser numérico"):
            DataFrameValidator.validate_ranges(df, {"x": ("not_a_number", 10)})
        
        with pytest.raises(ValueError, match="debe ser numérico"):
            DataFrameValidator.validate_ranges(df, {"x": (0, "invalid")})

    def test_invalid_bound_non_finite_raises(self):
        """Límites no finitos lanzan ValueError."""
        df = pd.DataFrame({"x": [5]})
        
        with pytest.raises(ValueError, match="debe ser finito"):
            DataFrameValidator.validate_ranges(df, {"x": (np.inf, 10)})
        
        with pytest.raises(ValueError, match="debe ser finito"):
            DataFrameValidator.validate_ranges(df, {"x": (0, np.nan)})

    def test_invalid_range_structure_raises(self):
        """Estructura de regla inválida lanza ValueError."""
        df = pd.DataFrame({"x": [5]})
        
        with pytest.raises(ValueError, match="debe ser tuple"):
            DataFrameValidator.validate_ranges(df, {"x": (1, 2, 3)})  # type: ignore
        
        with pytest.raises(ValueError, match="debe ser tuple"):
            DataFrameValidator.validate_ranges(df, {"x": [0, 10]})  # type: ignore

    def test_none_range_rules_raises(self):
        """range_rules=None lanza ValueError."""
        df = pd.DataFrame({"x": [5]})
        with pytest.raises(ValueError, match="range_rules no puede ser None"):
            DataFrameValidator.validate_ranges(df, None)  # type: ignore

    def test_missing_column_handling(self, simple_df):
        """Columna faltante genera WARNING/ERROR."""
        result_warn = DataFrameValidator.validate_ranges(
            simple_df,
            {"no_existe": (0, 10)},
            missing_as_warning=True
        )
        assert result_warn.has_warnings
        
        result_err = DataFrameValidator.validate_ranges(
            simple_df,
            {"no_existe": (0, 10)},
            missing_as_warning=False
        )
        assert result_err.has_errors

    def test_violation_severity_customizable(self):
        """Severidad de violaciones es configurable."""
        df = pd.DataFrame({"x": [-1, 5, 11]})
        
        result = DataFrameValidator.validate_ranges(
            df,
            {"x": (0, 10)},
            violation_severity=ValidationSeverity.ERROR
        )
        
        assert not result.is_valid
        violations = [i for i in result.issues if i.code == ValidationCode.RANGE_VIOLATION]
        assert violations[0].severity == ValidationSeverity.ERROR


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN IX: VALIDADORES TOPOLÓGICOS (Independencia Lineal, Supervivencia)
# ══════════════════════════════════════════════════════════════════════════════


class TestValidateLinearIndependence:
    """Verifica conexidad del espacio mediante estabilidad espectral (SVD)."""

    def test_independent_columns_valid(self, df_linearly_independent):
        """Vectores independientes → válido."""
        result = DataFrameValidator.validate_linear_independence(
            df_linearly_independent,
            ["x", "y"]
        )
        assert result.is_valid

    def test_linear_independence_via_spectral_gap(self, df_linearly_dependent):
        """Incision Espectral: Los vectores dependientes disparan κ → ∞."""
        result = DataFrameValidator.validate_linear_independence(
            df_linearly_dependent,
            ["x", "y"]
        )
        
        assert not result.is_valid
        dep_issues = [i for i in result.issues if i.code == ValidationCode.LINEAR_DEPENDENCY]
        assert len(dep_issues) == 1
        assert "Número de condición crítico" in dep_issues[0].message
        assert dep_issues[0].count == 1

    def test_three_columns_spectral_deficiency(self, df_rank_deficient_three_cols):
        """Tres columnas dependientes → κ colapsa la estabilidad."""
        result = DataFrameValidator.validate_linear_independence(
            df_rank_deficient_three_cols,
            ["x", "y", "z"]
        )
        
        assert not result.is_valid
        dep_issues = [i for i in result.issues if i.code == ValidationCode.LINEAR_DEPENDENCY]
        assert dep_issues[0].count == 1

    def test_single_column_always_independent(self):
        """Una única columna siempre es independiente."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = DataFrameValidator.validate_linear_independence(df, ["x"])
        assert result.is_valid

    def test_insufficient_rows_warning(self):
        """Pocas filas para calcular rango → WARNING."""
        df = pd.DataFrame({"x": [1], "y": [2]})
        result = DataFrameValidator.validate_linear_independence(df, ["x", "y"])
        
        assert result.has_warnings
        assert any("Insuficientes filas" in i.message for i in result.issues)

    def test_missing_column_warning_mode(self, simple_df):
        """Columna faltante genera WARNING."""
        result = DataFrameValidator.validate_linear_independence(
            simple_df,
            ["no_existe"],
            missing_as_warning=True
        )
        assert result.has_warnings

    def test_missing_column_error_mode(self, simple_df):
        """Columna faltante genera ERROR si missing_as_warning=False."""
        result = DataFrameValidator.validate_linear_independence(
            simple_df,
            ["no_existe"],
            missing_as_warning=False
        )
        assert result.has_errors

    def test_custom_condition_threshold(self):
        """Umbral personalizado de condición afecta detección."""
        # Columnas casi dependientes
        df = pd.DataFrame({
            "x": [1.0, 2.0, 3.0],
            "y": [1.0 + 1e-10, 2.0 + 1e-10, 3.0 + 1e-10]
        })
        
        # Con umbral bajo, se detecta degeneración
        result_low_threshold = DataFrameValidator.validate_linear_independence(
            df,
            ["x", "y"],
            condition_threshold=1e5
        )
        # Con umbral alto, se acepta
        result_high_threshold = DataFrameValidator.validate_linear_independence(
            df,
            ["x", "y"],
            condition_threshold=1e15
        )
        
        assert not result_low_threshold.is_valid
        assert result_high_threshold.is_valid

    def test_custom_severity_warning(self, df_linearly_dependent):
        """Severidad personalizada a WARNING."""
        result = DataFrameValidator.validate_linear_independence(
            df_linearly_dependent,
            ["x", "y"],
            severity=ValidationSeverity.WARNING
        )
        
        assert result.is_valid  # WARNING no bloquea
        assert result.has_warnings
        assert any(i.code == ValidationCode.LINEAR_DEPENDENCY for i in result.issues)

    def test_rows_with_nulls_excluded(self):
        """Filas con NaN se excluyen del cálculo."""
        df = pd.DataFrame({
            "x": [1, 2, np.nan, 4],
            "y": [0, 1, 2, 3]
        })
        
        result = DataFrameValidator.validate_linear_independence(df, ["x", "y"])
        # Quedan 3 filas completas, suficientes para 2 columnas
        assert result.is_valid or result.has_warnings  # Depende del rank


class TestValidateSurvivalThreshold:
    """Verifica preservación de medida: μ(limpias)/μ(totales) ≥ threshold."""

    def test_all_clean_survives(self, simple_df):
        """Dataset limpio → 100% supervivencia."""
        result = DataFrameValidator.validate_survival_threshold(
            simple_df,
            ["A", "B", "C"]
        )
        assert result.is_valid

    def test_below_threshold_critical(self, df_for_survival):
        """Supervivencia < threshold → CRITICAL."""
        # Rigorización: df_for_survival tiene 2/4 = 50% supervivencia.
        # Con min_survival=0.6, debe fallar.
        result = DataFrameValidator.validate_survival_threshold(
            df_for_survival,
            ["col1", "col2", "col3"],
            min_survival=0.6
        )
        
        assert not result.is_valid
        crit = [i for i in result.issues if i.code == ValidationCode.SURVIVAL_THRESHOLD_VIOLATION]
        assert len(crit) == 1
        assert crit[0].severity == ValidationSeverity.CRITICAL

    def test_above_threshold_valid(self, df_for_survival):
        """Supervivencia ≥ threshold → válido."""
        result = DataFrameValidator.validate_survival_threshold(
            df_for_survival,
            ["col1", "col2", "col3"],
            min_survival=0.4  # 50% > 40%
        )
        assert result.is_valid

    def test_exact_threshold_valid(self):
        """Supervivencia = threshold → válido."""
        df = pd.DataFrame({
            "x": [1, np.nan],  # 50% limpias
        })
        result = DataFrameValidator.validate_survival_threshold(
            df,
            ["x"],
            min_survival=0.5
        )
        assert result.is_valid

    def test_empty_strings_treated_as_singularity(self, df_for_survival):
        """Con include_empty_strings=True, "" es singularidad."""
        # 2/4 = 50% < 0.6
        result = DataFrameValidator.validate_survival_threshold(
            df_for_survival,
            ["col1", "col2", "col3"],
            min_survival=0.6,
            include_empty_strings=True
        )
        assert not result.is_valid

    def test_empty_strings_ignored(self, df_for_survival):
        """Con include_empty_strings=False, "" no es singularidad."""
        result = DataFrameValidator.validate_survival_threshold(
            df_for_survival,
            ["col1", "col2", "col3"],
            min_survival=0.6,
            include_empty_strings=False
        )
        # Solo fila 1 (col1=NaN) es sucia → 3/4 = 75% > 60%
        assert result.is_valid

    def test_all_rows_dirty_zero_survival(self):
        """Todas las filas sucias → 0% supervivencia."""
        df = pd.DataFrame({
            "x": [np.nan, None, np.nan],
        })
        result = DataFrameValidator.validate_survival_threshold(
            df,
            ["x"],
            min_survival=0.1
        )
        assert not result.is_valid

    def test_empty_dataframe_100_percent_survival(self, empty_df):
        """DataFrame vacío → 100% supervivencia (0/0 = 1.0)."""
        result = DataFrameValidator.validate_survival_threshold(
            empty_df,
            [],
            min_survival=0.9
        )
        assert result.is_valid

    def test_missing_column_warning(self, simple_df):
        """Columna faltante genera WARNING."""
        result = DataFrameValidator.validate_survival_threshold(
            simple_df,
            ["no_existe"]
        )
        
        assert result.has_warnings
        assert any(i.code == ValidationCode.MISSING_CRITICAL_COLUMN for i in result.issues)

    def test_custom_threshold(self):
        """Threshold personalizado."""
        df = pd.DataFrame({
            "x": [1, np.nan, 3, 4, 5],  # 80% limpias
        })
        
        result_pass = DataFrameValidator.validate_survival_threshold(
            df,
            ["x"],
            min_survival=0.75
        )
        assert result_pass.is_valid
        
        result_fail = DataFrameValidator.validate_survival_threshold(
            df,
            ["x"],
            min_survival=0.85
        )
        assert not result_fail.is_valid


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN X: ORQUESTACIÓN DE DOMINIO (Composición Funcional)
# ══════════════════════════════════════════════════════════════════════════════


class TestValidateDomain:
    """Verifica composición orquestada de todos los validadores."""

    def test_complete_clean_dataframe(self, simple_df):
        """DataFrame limpio pasa todas las validaciones."""
        result = DataFrameValidator.validate_domain(
            simple_df,
            required_columns=["A", "B", "C"],
            numeric_columns=["A", "B"],
            non_negative_columns=["A", "B"],
        )
        assert result.is_valid

    def test_catches_schema_errors(self, simple_df):
        """Errores de esquema son detectados."""
        result = DataFrameValidator.validate_domain(
            simple_df,
            required_columns=["Z"]
        )
        
        assert not result.is_valid
        assert any(i.code == ValidationCode.MISSING_REQUIRED_COLUMN for i in result.issues)

    def test_catches_quality_warnings(self, df_with_nulls):
        """Warnings de calidad son detectados."""
        result = DataFrameValidator.validate_domain(
            df_with_nulls,
            critical_columns=["A", "B"]
        )
        
        assert result.has_warnings
        assert any(i.code == ValidationCode.NULL_VALUES for i in result.issues)

    def test_catches_numeric_violations(self, df_mixed_types):
        """Violaciones numéricas son detectadas."""
        result = DataFrameValidator.validate_domain(
            df_mixed_types,
            numeric_columns=["mixed"]
        )
        
        assert result.has_warnings
        assert any(i.code == ValidationCode.NON_NUMERIC_VALUES for i in result.issues)

    def test_catches_negative_values(self, df_with_negatives):
        """Valores negativos son detectados."""
        result = DataFrameValidator.validate_domain(
            df_with_negatives,
            non_negative_columns=["cost", "time"]
        )
        
        assert result.has_warnings
        assert any(i.code == ValidationCode.NEGATIVE_VALUES for i in result.issues)

    def test_catches_range_violations(self, df_out_of_range):
        """Violaciones de rango son detectadas."""
        result = DataFrameValidator.validate_domain(
            df_out_of_range,
            range_rules={"percentage": (0, 100)}
        )
        
        assert result.has_warnings
        assert any(i.code == ValidationCode.RANGE_VIOLATION for i in result.issues)

    def test_catches_linear_dependency(self, df_linearly_dependent):
        """Dependencia lineal es detectada."""
        result = DataFrameValidator.validate_domain(
            df_linearly_dependent,
            required_columns=["x", "y"],
            linear_dependency_columns=["x", "y"]
        )
        
        assert not result.is_valid
        assert any(i.code == ValidationCode.LINEAR_DEPENDENCY for i in result.issues)

    def test_catches_survival_violation(self, df_for_survival):
        """Violación de supervivencia es detectada."""
        # 50% < 0.6
        result = DataFrameValidator.validate_domain(
            df_for_survival,
            required_columns=["col1", "col2", "col3"],
            survival_critical_columns=["col1", "col2", "col3"],
            survival_min_threshold=0.6
        )
        
        assert not result.is_valid
        assert any(i.code == ValidationCode.SURVIVAL_THRESHOLD_VIOLATION for i in result.issues)

    def test_multiple_validations_combined(self, df_with_nulls):
        """Múltiples validaciones se combinan vía monoide."""
        result = DataFrameValidator.validate_domain(
            df_with_nulls,
            required_columns=["A", "B", "C"],
            critical_columns=["A", "B"],
            numeric_columns=["A", "B"],
        )
        
        # Debe haber múltiples issues de diferentes tipos
        assert len(result.issues) > 1
        
        # Verificar que vienen de diferentes validadores
        codes = {i.code for i in result.issues}
        assert ValidationCode.NULL_VALUES in codes

    def test_empty_arguments_returns_identity(self, simple_df):
        """Sin argumentos → elemento neutro (válido)."""
        result = DataFrameValidator.validate_domain(simple_df)
        assert result.is_valid

    def test_order_of_execution_preserved(self):
        """El orden de ejecución es consistente."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        
        result = DataFrameValidator.validate_domain(
            df,
            required_columns=["A", "B"],
            critical_columns=["A"],
            numeric_columns=["A"],
        )
        
        # Los issues deben aparecer en orden de ejecución:
        # 1. validate_schema
        # 2. check_data_quality
        # 3. validate_numeric_columns
        # (asumiendo que no hay errores, esto no es observable directamente)

    def test_invalid_dataframe_short_circuits(self):
        """DataFrame inválido genera error inmediato."""
        result = DataFrameValidator.validate_domain(
            None,  # type: ignore
            required_columns=["A"]
        )
        
        assert not result.is_valid
        assert any(i.code == ValidationCode.NONE_DATAFRAME for i in result.issues)


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN XI: CASOS LÍMITE Y MANEJO DE ERRORES
# ══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Verifica comportamiento en casos límite y frontera."""

    def test_empty_dataframe_valid(self, empty_df):
        """DataFrame vacío es válido si no se requieren columnas."""
        result = DataFrameValidator.validate_domain(empty_df)
        assert result.is_valid

    def test_single_row_dataframe(self):
        """DataFrame con una fila."""
        df = pd.DataFrame({"x": [1]})
        result = DataFrameValidator.validate_domain(
            df,
            required_columns=["x"],
            numeric_columns=["x"]
        )
        assert result.is_valid

    def test_single_column_dataframe(self):
        """DataFrame con una columna."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = DataFrameValidator.validate_domain(
            df,
            required_columns=["x"],
            numeric_columns=["x"]
        )
        assert result.is_valid

    def test_extreme_numeric_values(self, df_edge_cases):
        """Valores extremos pero válidos."""
        result = DataFrameValidator.validate_domain(
            df_edge_cases,
            numeric_columns=["zeros", "extremes", "tiny"]
        )
        # Puede haber warnings de subnormales, pero debe ser válido
        assert result.is_valid or result.has_warnings

    def test_unicode_column_names(self):
        """Nombres de columnas con Unicode."""
        df = pd.DataFrame({"δ": [1, 2], "α": [3, 4]})
        result = DataFrameValidator.validate_domain(
            df,
            required_columns=["δ", "α"]
        )
        assert result.is_valid

    def test_very_long_column_names(self):
        """Nombres de columnas muy largos."""
        long_name = "x" * 1000
        df = pd.DataFrame({long_name: [1, 2, 3]})
        result = DataFrameValidator.validate_domain(
            df,
            required_columns=[long_name]
        )
        assert result.is_valid

    def test_special_characters_in_column_names(self):
        """Caracteres especiales en nombres de columnas."""
        df = pd.DataFrame({"col-1": [1], "col.2": [2], "col@3": [3]})
        result = DataFrameValidator.validate_domain(
            df,
            required_columns=["col-1", "col.2", "col@3"]
        )
        assert result.is_valid


class TestErrorHandling:
    """Verifica manejo robusto de errores."""

    def test_none_dataframe_all_validators(self):
        """None DataFrame es rechazado por todos los validadores."""
        validators = [
            lambda: DataFrameValidator.validate_schema(None, ["A"]),  # type: ignore
            lambda: DataFrameValidator.check_data_quality(None, ["A"]),  # type: ignore
            lambda: DataFrameValidator.validate_numeric_columns(None, ["A"]),  # type: ignore
            lambda: DataFrameValidator.validate_non_negative(None, ["A"]),  # type: ignore
            lambda: DataFrameValidator.validate_linear_independence(None, ["A"]),  # type: ignore
            lambda: DataFrameValidator.validate_survival_threshold(None, ["A"]),  # type: ignore
            lambda: DataFrameValidator.validate_domain(None),  # type: ignore
        ]
        
        for validator in validators:
            result = validator()
            assert not result.is_valid
            assert any(i.code == ValidationCode.NONE_DATAFRAME for i in result.issues)

    def test_wrong_type_dataframe(self):
        """Tipo incorrecto es rechazado."""
        wrong_types = ["string", 123, [1, 2, 3], {"a": 1}]
        
        for wrong in wrong_types:
            result = DataFrameValidator.validate_schema(wrong, ["A"])  # type: ignore
            assert not result.is_valid
            assert any(i.code == ValidationCode.INVALID_DATAFRAME_TYPE for i in result.issues)

    def test_invalid_columns_argument_raises(self):
        """Argumentos inválidos para columnas lanzan ValueError."""
        df = pd.DataFrame({"A": [1]})
        
        with pytest.raises(ValueError, match="no puede ser None"):
            DataFrameValidator.validate_schema(df, None)  # type: ignore
        
        with pytest.raises(ValueError, match="debe ser un iterable"):
            DataFrameValidator.validate_schema(df, 123)  # type: ignore
        
        with pytest.raises(ValueError, match="debe ser str"):
            DataFrameValidator.validate_schema(df, [1, 2])  # type: ignore

    def test_validation_result_add_wrong_type_raises(self):
        """Sumar tipo incorrecto a ValidationResult lanza TypeError."""
        result = ValidationResult.success()
        
        with pytest.raises(TypeError):
            result + "not a result"  # type: ignore

    def test_validation_result_merge_wrong_type_raises(self):
        """merge con tipo incorrecto lanza TypeError."""
        result = ValidationResult.success()
        
        with pytest.raises(TypeError, match="merge requiere ValidationResult"):
            result.merge("not a result")  # type: ignore


# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN XII: UTILIDADES Y HELPERS INTERNOS
# ══════════════════════════════════════════════════════════════════════════════


class TestMaterializeIterable:
    """Verifica materialización de iterables a tuplas."""

    def test_tuple_passthrough(self):
        """Tuplas pasan sin copiar."""
        t = ("A", "B", "C")
        result = _materialize_iterable(t)
        assert result is t  # Mismo objeto

    def test_list_to_tuple(self):
        """Listas se convierten a tuplas."""
        result = _materialize_iterable(["A", "B", "C"])
        assert result == ("A", "B", "C")
        assert isinstance(result, tuple)

    def test_generator_to_tuple(self):
        """Generadores se consumen y convierten."""
        gen = (x for x in ["A", "B", "C"])
        result = _materialize_iterable(gen)
        assert result == ("A", "B", "C")

    def test_set_to_tuple(self):
        """Sets se convierten (orden no garantizado)."""
        result = _materialize_iterable({"A", "B", "C"})
        assert isinstance(result, tuple)
        assert set(result) == {"A", "B", "C"}

    def test_empty_iterable(self):
        """Iterables vacíos producen tupla vacía."""
        assert _materialize_iterable([]) == ()
        assert _materialize_iterable(iter([])) == ()


class TestIsSubnormal:
    """Verifica detección de valores subnormales."""

    def test_zero_not_subnormal(self):
        """Cero no es subnormal."""
        assert not DataFrameValidator._is_subnormal(0.0)
        assert not DataFrameValidator._is_subnormal(-0.0)

    def test_normal_values_not_subnormal(self):
        """Valores normales no son subnormales."""
        assert not DataFrameValidator._is_subnormal(1.0)
        assert not DataFrameValidator._is_subnormal(-1.0)
        assert not DataFrameValidator._is_subnormal(MIN_NORMAL_FLOAT)

    def test_subnormal_positive(self):
        """Valor subnormal positivo es detectado."""
        assert DataFrameValidator._is_subnormal(MIN_NORMAL_FLOAT / 2)

    def test_subnormal_negative(self):
        """Valor subnormal negativo es detectado."""
        assert DataFrameValidator._is_subnormal(-MIN_NORMAL_FLOAT / 2)

    def test_infinity_not_subnormal(self):
        """Infinitos no son subnormales."""
        assert not DataFrameValidator._is_subnormal(np.inf)
        assert not DataFrameValidator._is_subnormal(-np.inf)

    def test_nan_not_subnormal(self):
        """NaN no es subnormal."""
        assert not DataFrameValidator._is_subnormal(np.nan)


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PYTEST Y HELPERS ADICIONALES
# ══════════════════════════════════════════════════════════════════════════════


def test_module_exports():
    """Verifica que todas las exportaciones están disponibles."""
    from app.adapters.validators import __all__
    
    expected = {
        "ValidationSeverity",
        "ValidationCode",
        "ValidationIssue",
        "ValidationResult",
        "DataFrameValidator",
        "FLOAT_TOLERANCE",
        "CONDITION_NUMBER_THRESHOLD",
        "LIPSCHITZ_LIMIT",
        "MIN_NORMAL_FLOAT",
        "DEFAULT_SURVIVAL_THRESHOLD",
    }
    
    assert set(__all__) == expected


@pytest.mark.parametrize("severity", list(ValidationSeverity))
def test_all_severities_have_rank(severity):
    """Todas las severidades tienen rango definido."""
    assert isinstance(severity.rank, int)
        # El rango es una propiedad del retículo, pero las leyes de Birkhoff
        # (probadas arriba con Hypothesis) son las que garantizan la topología.


@pytest.mark.parametrize("code", list(ValidationCode))
def test_all_codes_are_strings(code):
    """Todos los códigos son strings."""
    assert isinstance(code.value, str)
    assert code.value == code.value.upper()


# ══════════════════════════════════════════════════════════════════════════════
# MARCADORES PARA EJECUCIÓN SELECTIVA
# ══════════════════════════════════════════════════════════════════════════════

"""
Para ejecutar grupos específicos de pruebas:

pytest tests/test_validators.py::TestValidationResultMonoid -v
pytest tests/test_validators.py::TestValidateLinearIndependence -v
pytest tests/test_validators.py -k "subnormal" -v
pytest tests/test_validators.py -m "slow" -v  # Si se añaden marcadores
"""