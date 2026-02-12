"""
Tests para Pipeline Director
============================================================

Cobertura:
  1. Infraestructura algebraica: MICRegistry, BasisVector, stratum_level
  2. Persistencia Glass Box: carga/guardado atómico, limpieza, integridad
  3. Inferencia de estrato: orden descendente, invariantes de filtración
  4. Orquestación: PipelineDirector (ejecución, receta, errores, regresión)
  5. Pasos individuales: precondiciones, null safety, propagación de errores
  6. Punto de entrada: process_all_files (telemetría null-safe, validación de rutas)
  7. Invariantes de filtración DIKW: asignaciones de estrato correctas

Convenciones:
  - Cada test sigue el patrón: Arrange → Act → Assert.
  - Se usa `tmp_path` (fixture de pytest) para aislamiento de I/O.
  - La telemetría se mockea con MagicMock para flexibilidad.
  - Los steps de producción se reemplazan por stubs cuando se testea orquestación.

Cambios vs. suite original:
  - Corregidos imports (steps viven en pipeline_director_v2, no en apu_processor)
  - get_rank() → dimension (propiedad) y len()
  - Eliminada dependencia a PipelineStepExecutionError inexistente
  - Agregados ~40 tests nuevos para cubrir métodos refinados
  - Usa tmp_path en vez de tempfile.mkdtemp() para limpieza automática
"""

import datetime
import json
import logging
import pickle
import uuid

import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, PropertyMock

# ==================== IMPORTS DEL MÓDULO BAJO TEST ====================
# NOTA: Todos los steps están definidos en pipeline_director_v2, NO en apu_processor.

from app.pipeline_director import (
    # Clases principales
    PipelineDirector,
    MICRegistry,
    BasisVector,
    ProcessingStep,
    PipelineSteps,
    # Steps
    LoadDataStep,
    AuditedMergeStep,
    CalculateCostsStep,
    FinalMergeStep,
    BusinessTopologyStep,
    MaterializationStep,
    BuildOutputStep,
    # Funciones auxiliares
    stratum_level,
    _STRATUM_ORDER,
    process_all_files,
)
from app.schemas import Stratum
from app.telemetry import TelemetryContext
from app.apu_processor import ProcessingThresholds


# ==================== HELPERS Y STUBS REUTILIZABLES ====================


class StubStep(ProcessingStep):
    """
    Paso stub configurable para tests de orquestación.
    Permite controlar qué claves agrega al contexto y si debe fallar.
    """

    def __init__(self, config: dict = None, thresholds=None):
        self.config = config or {}
        self.thresholds = thresholds
        # Atributos configurables por test via class-level
        self._output_keys: dict = {}
        self._should_raise: Exception = None
        self._execution_count: int = 0

    def execute(self, context: dict, telemetry) -> dict:
        self._execution_count += 1
        if self._should_raise:
            raise self._should_raise
        updated = {**context, **self._output_keys}
        return updated


def make_stub_class(output_keys: dict = None, raise_on_execute: Exception = None):
    """
    Fábrica de clases stub tipadas. Cada invocación retorna una clase NUEVA
    (evita conflictos de label duplicado en MICRegistry).
    """

    class DynamicStub(ProcessingStep):
        _keys = output_keys or {}
        _error = raise_on_execute

        def __init__(self, config=None, thresholds=None):
            pass

        def execute(self, context: dict, telemetry) -> dict:
            if self._error:
                raise self._error
            return {**context, **self._keys}

    return DynamicStub


def make_telemetry_mock() -> MagicMock:
    """
    Crea un mock de TelemetryContext que acepta cualquier llamada.
    MagicMock es más flexible que Mock(spec=...) porque no requiere
    que los métodos existan en la clase real durante el test.
    """
    mock = MagicMock()
    mock.start_step = MagicMock()
    mock.end_step = MagicMock()
    mock.record_error = MagicMock()
    mock.record_metric = MagicMock()
    return mock


# ==================== FIXTURES ====================


@pytest.fixture
def telemetry():
    """Contexto de telemetría mock con interfaz completa."""
    return make_telemetry_mock()


@pytest.fixture
def base_config(tmp_path):
    """
    Configuración base del pipeline.
    Usa tmp_path para aislamiento de I/O — pytest lo limpia automáticamente.
    """
    return {
        "session_dir": str(tmp_path / "sessions"),
        "file_profiles": {
            "presupuesto_default": {},
            "insumos_default": {},
            "apus_default": {},
        },
        "processing_thresholds": {},
    }


@pytest.fixture
def director(base_config, telemetry):
    """Instancia del PipelineDirector con config y telemetría de test."""
    return PipelineDirector(base_config, telemetry)


@pytest.fixture
def sample_presupuesto_df():
    return pd.DataFrame(
        {"id": [1, 2, 3], "descripcion": ["A", "B", "C"], "cantidad": [10, 20, 30]}
    )


@pytest.fixture
def sample_insumos_df():
    return pd.DataFrame(
        {
            "codigo": ["I001", "I002", "I003"],
            "nombre": ["Insumo X", "Insumo Y", "Insumo Z"],
            "precio_unitario": [1.5, 2.0, 2.5],
        }
    )


@pytest.fixture
def sample_apus_df():
    return pd.DataFrame(
        {
            "id_apu": [1, 1, 2],
            "codigo_insumo": ["I001", "I002", "I003"],
            "cantidad_insumo": [1.0, 0.5, 2.0],
        }
    )


@pytest.fixture
def physics_context(sample_presupuesto_df, sample_insumos_df, sample_apus_df):
    """Contexto con artefactos del estrato PHYSICS."""
    return {
        "df_presupuesto": sample_presupuesto_df,
        "df_insumos": sample_insumos_df,
        "df_apus_raw": sample_apus_df,
    }


@pytest.fixture
def tactics_context(physics_context):
    """Contexto con artefactos del estrato TACTICS (incluye PHYSICS)."""
    return {
        **physics_context,
        "df_merged": pd.DataFrame({"merged": [1]}),
        "df_apu_costos": pd.DataFrame({"costo": [100]}),
        "df_tiempo": pd.DataFrame({"tiempo": [8]}),
        "df_rendimiento": pd.DataFrame({"rendimiento": [0.9]}),
        "df_final": pd.DataFrame({"final": [1]}),
    }


@pytest.fixture
def strategy_context(tactics_context):
    """Contexto con artefactos del estrato STRATEGY (incluye TACTICS)."""
    graph_mock = MagicMock()
    graph_mock.number_of_nodes.return_value = 5
    graph_mock.number_of_edges.return_value = 4
    report_mock = MagicMock()
    report_mock.details = {"pyramid_stability": 8.5}
    return {
        **tactics_context,
        "graph": graph_mock,
        "business_topology_report": report_mock,
        "validated_strata": {Stratum.PHYSICS, Stratum.TACTICS},
    }


@pytest.fixture
def wisdom_context(strategy_context):
    """Contexto con artefactos del estrato WISDOM (incluye STRATEGY)."""
    return {
        **strategy_context,
        "final_result": {"kind": "DataProduct", "payload": {}},
        "bill_of_materials": MagicMock(),
    }


# ==================== 1. TESTS PARA stratum_level ====================


class TestStratumLevel:
    """Valida la función auxiliar que mapea Stratum → nivel ordinal."""

    def test_physics_is_level_zero(self):
        assert stratum_level(Stratum.PHYSICS) == 0

    def test_tactics_is_level_one(self):
        assert stratum_level(Stratum.TACTICS) == 1

    def test_strategy_is_level_two(self):
        assert stratum_level(Stratum.STRATEGY) == 2

    def test_wisdom_is_level_three(self):
        assert stratum_level(Stratum.WISDOM) == 3

    def test_strict_ordering(self):
        """La filtración V_P ⊂ V_T ⊂ V_S ⊂ V_W impone orden estricto."""
        levels = [stratum_level(s) for s in Stratum]
        # Todos los niveles deben ser únicos y estar definidos
        valid_levels = [l for l in levels if l >= 0]
        assert len(valid_levels) == len(set(valid_levels)), (
            "Dos estratos no pueden tener el mismo nivel ordinal"
        )

    def test_unknown_stratum_returns_negative(self):
        """Un valor no mapeado retorna -1 (centinela)."""
        # Simulamos un stratum hipotético no registrado
        fake = MagicMock()
        assert stratum_level(fake) == -1


# ==================== 2. TESTS PARA MICRegistry ====================


class TestMICRegistry:
    """Tests para el catálogo centralizado de operadores."""

    def test_empty_initialization(self):
        """La MICRegistry se inicializa sin vectores base."""
        registry = MICRegistry()
        assert registry.dimension == 0
        assert len(registry) == 0
        assert registry.get_available_labels() == []
        assert list(registry) == []

    def test_add_single_vector(self):
        """Agregar un vector base incrementa la dimensión."""
        registry = MICRegistry()
        StepClass = make_stub_class()
        registry.add_basis_vector("step_a", StepClass, Stratum.PHYSICS)

        assert registry.dimension == 1
        assert len(registry) == 1
        assert registry.get_available_labels() == ["step_a"]

    def test_vector_properties(self):
        """El BasisVector almacena correctamente todas las propiedades."""
        registry = MICRegistry()
        StepClass = make_stub_class()
        registry.add_basis_vector("test_op", StepClass, Stratum.TACTICS)

        vector = registry.get_basis_vector("test_op")
        assert vector is not None
        assert vector.index == 0
        assert vector.label == "test_op"
        assert vector.operator_class is StepClass
        assert vector.stratum == Stratum.TACTICS

    def test_sequential_indices(self):
        """Los índices se asignan secuencialmente."""
        registry = MICRegistry()
        for i, name in enumerate(["alpha", "beta", "gamma"]):
            registry.add_basis_vector(name, make_stub_class(), Stratum.PHYSICS)
            vec = registry.get_basis_vector(name)
            assert vec.index == i

    def test_duplicate_label_raises_value_error(self):
        """Etiquetas duplicadas violan la unicidad de la base vectorial."""
        registry = MICRegistry()
        StepA = make_stub_class()
        StepB = make_stub_class()
        registry.add_basis_vector("unique", StepA, Stratum.PHYSICS)

        with pytest.raises(ValueError, match="Duplicate label"):
            registry.add_basis_vector("unique", StepB, Stratum.TACTICS)

    def test_empty_label_raises_value_error(self):
        """Una etiqueta vacía es rechazada."""
        registry = MICRegistry()
        with pytest.raises(ValueError, match="non-empty string"):
            registry.add_basis_vector("", make_stub_class(), Stratum.PHYSICS)

    def test_non_string_label_raises_value_error(self):
        """Una etiqueta no-string es rechazada."""
        registry = MICRegistry()
        with pytest.raises(ValueError):
            registry.add_basis_vector(123, make_stub_class(), Stratum.PHYSICS)

    def test_non_subclass_raises_type_error(self):
        """Una clase que no hereda de ProcessingStep es rechazada."""
        registry = MICRegistry()

        class NotAStep:
            pass

        with pytest.raises(TypeError, match="subclass of ProcessingStep"):
            registry.add_basis_vector("bad_step", NotAStep, Stratum.PHYSICS)

    def test_get_nonexistent_returns_none(self):
        """Consultar un label inexistente retorna None."""
        registry = MICRegistry()
        assert registry.get_basis_vector("fantasma") is None

    def test_get_available_labels_preserves_insertion_order(self):
        """Las etiquetas se retornan en orden de inserción."""
        registry = MICRegistry()
        labels = ["zeta", "alpha", "mu"]
        for label in labels:
            registry.add_basis_vector(label, make_stub_class(), Stratum.PHYSICS)

        assert registry.get_available_labels() == labels

    def test_get_vectors_by_stratum(self):
        """Proyección sobre un subestrato retorna solo los vectores correspondientes."""
        registry = MICRegistry()
        registry.add_basis_vector("p1", make_stub_class(), Stratum.PHYSICS)
        registry.add_basis_vector("t1", make_stub_class(), Stratum.TACTICS)
        registry.add_basis_vector("p2", make_stub_class(), Stratum.PHYSICS)
        registry.add_basis_vector("s1", make_stub_class(), Stratum.STRATEGY)

        physics_vectors = registry.get_vectors_by_stratum(Stratum.PHYSICS)
        assert len(physics_vectors) == 2
        assert all(v.stratum == Stratum.PHYSICS for v in physics_vectors)
        assert [v.label for v in physics_vectors] == ["p1", "p2"]

    def test_get_vectors_by_stratum_empty(self):
        """Proyección sobre estrato sin vectores retorna lista vacía."""
        registry = MICRegistry()
        registry.add_basis_vector("p1", make_stub_class(), Stratum.PHYSICS)
        assert registry.get_vectors_by_stratum(Stratum.WISDOM) == []

    def test_get_execution_sequence(self):
        """La secuencia de ejecución refleja el orden de registro."""
        registry = MICRegistry()
        registry.add_basis_vector("first", make_stub_class(), Stratum.PHYSICS)
        registry.add_basis_vector("second", make_stub_class(), Stratum.TACTICS)

        seq = registry.get_execution_sequence()
        assert len(seq) == 2
        assert seq[0] == {"step": "first", "enabled": True}
        assert seq[1] == {"step": "second", "enabled": True}

    def test_iteration_yields_basis_vectors_in_order(self):
        """Iterar sobre la MIC yield BasisVectors en orden de registro."""
        registry = MICRegistry()
        labels = ["a", "b", "c"]
        for label in labels:
            registry.add_basis_vector(label, make_stub_class(), Stratum.PHYSICS)

        iterated = list(registry)
        assert len(iterated) == 3
        assert all(isinstance(v, BasisVector) for v in iterated)
        assert [v.label for v in iterated] == labels

    def test_basis_vector_is_frozen(self):
        """BasisVector es inmutable (frozen dataclass)."""
        registry = MICRegistry()
        registry.add_basis_vector("immut", make_stub_class(), Stratum.PHYSICS)
        vector = registry.get_basis_vector("immut")

        with pytest.raises(AttributeError):
            vector.label = "mutated"


# ==================== 3. TESTS PARA PERSISTENCIA ====================


class TestContextPersistence:
    """Tests para la Glass Box Persistence (load/save/cleanup)."""

    def test_save_and_load_roundtrip(self, director):
        """El contexto guardado se recupera idénticamente."""
        context = {"key_a": "value_a", "key_b": [1, 2, 3]}
        session_id = "roundtrip_test"

        director._save_context_state(session_id, context)
        loaded = director._load_context_state(session_id)

        assert loaded == context

    def test_load_nonexistent_session_returns_none(self, director):
        """Cargar una sesión inexistente retorna None."""
        loaded = director._load_context_state("nonexistent_session_id")
        assert loaded is None

    def test_load_empty_session_id_returns_none(self, director):
        """Cargar con session_id vacío retorna None."""
        assert director._load_context_state("") is None
        assert director._load_context_state(None) is None

    def test_save_creates_session_directory(self, base_config, telemetry):
        """Si el directorio de sesiones no existe, se crea."""
        config = {**base_config, "session_dir": str(Path(base_config["session_dir"]) / "deep" / "nested")}
        d = PipelineDirector(config, telemetry)
        d._save_context_state("test_id", {"data": True})

        assert Path(config["session_dir"]).exists()
        loaded = d._load_context_state("test_id")
        assert loaded == {"data": True}

    def test_atomic_write_no_temp_files_remain(self, director):
        """Tras un guardado exitoso, no quedan archivos temporales .tmp."""
        session_id = "atomic_test"
        director._save_context_state(session_id, {"clean": True})

        session_dir = Path(director.session_dir)
        tmp_files = list(session_dir.glob("*.tmp"))
        assert tmp_files == [], f"Archivos temporales residuales: {tmp_files}"

    def test_load_validates_type_is_dict(self, director):
        """Si el archivo pickle contiene un no-dict, retorna None."""
        session_id = "corrupted_type"
        session_file = director.session_dir / f"{session_id}.pkl"

        # Escribir un objeto no-dict directamente
        with open(session_file, "wb") as f:
            pickle.dump(["not", "a", "dict"], f)

        loaded = director._load_context_state(session_id)
        assert loaded is None, "Un pickle corrupto (no-dict) debe retornar None"

    def test_load_handles_corrupted_file(self, director):
        """Un archivo pickle corrupto retorna None."""
        session_id = "corrupted_data"
        session_file = director.session_dir / f"{session_id}.pkl"

        # Escribir bytes basura
        with open(session_file, "wb") as f:
            f.write(b"this is not valid pickle data")

        loaded = director._load_context_state(session_id)
        assert loaded is None

    def test_cleanup_session_removes_file(self, director):
        """_cleanup_session elimina el archivo de sesión."""
        session_id = "to_cleanup"
        director._save_context_state(session_id, {"ephemeral": True})
        session_file = director.session_dir / f"{session_id}.pkl"
        assert session_file.exists()

        director._cleanup_session(session_id)
        assert not session_file.exists()

    def test_cleanup_nonexistent_session_is_noop(self, director):
        """Limpiar una sesión inexistente no lanza excepción."""
        # No debe lanzar excepción
        director._cleanup_session("ghost_session")

    def test_save_with_dataframe(self, director, sample_presupuesto_df):
        """DataFrames se serializan y deserializan correctamente."""
        session_id = "df_test"
        context = {"df_presupuesto": sample_presupuesto_df}

        director._save_context_state(session_id, context)
        loaded = director._load_context_state(session_id)

        pd.testing.assert_frame_equal(loaded["df_presupuesto"], sample_presupuesto_df)


# ==================== 4. TESTS PARA INFERENCIA DE ESTRATO ====================


class TestStratumInference:
    """
    Tests para _infer_current_stratum_from_context.
    
    Invariante crítico: la evaluación debe ser DESCENDENTE (WISDOM → PHYSICS).
    La versión anterior evaluaba ascendente (PHYSICS primero), retornando
    siempre PHYSICS una vez cargados los datos iniciales.
    """

    def test_empty_context_returns_none(self, director):
        """Contexto vacío → estrato indeterminado."""
        assert director._infer_current_stratum_from_context({}) is None

    def test_physics_context_detected(self, director, physics_context):
        """Contexto con solo artefactos PHYSICS → PHYSICS."""
        assert director._infer_current_stratum_from_context(physics_context) == Stratum.PHYSICS

    def test_tactics_context_detected(self, director, tactics_context):
        """
        Contexto con artefactos TACTICS (que incluye PHYSICS) → TACTICS.
        
        Este test valida la corrección principal: la evaluación descendente
        retorna TACTICS y NO PHYSICS a pesar de que las claves de PHYSICS
        también están presentes.
        """
        result = director._infer_current_stratum_from_context(tactics_context)
        assert result == Stratum.TACTICS, (
            f"Expected TACTICS but got {result}. "
            "La evaluación debe ser descendente (WISDOM → PHYSICS)."
        )

    def test_strategy_context_detected(self, director, strategy_context):
        """Contexto con artefactos STRATEGY → STRATEGY."""
        result = director._infer_current_stratum_from_context(strategy_context)
        assert result == Stratum.STRATEGY

    def test_wisdom_context_detected(self, director, wisdom_context):
        """Contexto con artefactos WISDOM → WISDOM."""
        result = director._infer_current_stratum_from_context(wisdom_context)
        assert result == Stratum.WISDOM

    def test_irrelevant_keys_ignored(self, director):
        """Claves que no pertenecen a ningún estrato no producen falso positivo."""
        context = {"random_key": "value", "another_key": 42}
        assert director._infer_current_stratum_from_context(context) is None

    def test_single_strategy_key_sufficient(self, director):
        """Basta una sola clave de un estrato para detectarlo."""
        context = {"graph": MagicMock()}
        assert director._infer_current_stratum_from_context(context) == Stratum.STRATEGY

    def test_monotonicity_through_pipeline(self, director):
        """
        El estrato inferido nunca decrece a medida que se agregan claves
        del pipeline en orden canónico.
        
        Valida la propiedad de filtración:
        V_PHYSICS ⊂ V_TACTICS ⊂ V_STRATEGY ⊂ V_WISDOM
        """
        context = {}
        previous_level = -1

        progressive_keys = [
            ({"df_presupuesto": True}, Stratum.PHYSICS),
            ({"df_apu_costos": True}, Stratum.TACTICS),
            ({"graph": True}, Stratum.STRATEGY),
            ({"final_result": True}, Stratum.WISDOM),
        ]

        for new_keys, expected_stratum in progressive_keys:
            context.update(new_keys)
            inferred = director._infer_current_stratum_from_context(context)
            current_level = stratum_level(inferred)

            assert current_level >= previous_level, (
                f"Monotonicity violated: {previous_level} → {current_level} "
                f"after adding {list(new_keys.keys())}"
            )
            assert inferred == expected_stratum
            previous_level = current_level


# ==================== 5. TESTS PARA PipelineDirector ====================


class TestPipelineDirector:
    """Tests para la orquestación del PipelineDirector."""

    def test_initialization_loads_all_canonical_steps(self, director):
        """El director registra todos los pasos definidos en PipelineSteps."""
        available = set(director.mic.get_available_labels())
        expected = {step.value for step in PipelineSteps}
        assert available == expected, (
            f"Missing steps: {expected - available}, "
            f"Extra steps: {available - expected}"
        )

    def test_initialization_dimension_matches_enum(self, director):
        """La dimensión de la MIC coincide con el número de pasos del enum."""
        assert director.mic.dimension == len(PipelineSteps)
        assert len(director.mic) == len(PipelineSteps)

    def test_stratum_assignments_respect_filtration(self, director):
        """
        Verifica que las asignaciones de estrato a cada paso respetan
        el grafo de dependencias y la filtración DIKW.

        Correcciones del refinamiento:
          - final_merge: PHYSICS → TACTICS (consume df_apu_costos, df_tiempo)
          - materialization: TACTICS → STRATEGY (consume business_topology_report)
        """
        expected_strata = {
            "load_data": Stratum.PHYSICS,
            "audited_merge": Stratum.PHYSICS,
            "calculate_costs": Stratum.TACTICS,
            "final_merge": Stratum.TACTICS,        # ← corregido de PHYSICS
            "business_topology": Stratum.STRATEGY,
            "materialization": Stratum.STRATEGY,    # ← corregido de TACTICS
            "build_output": Stratum.WISDOM,
        }

        for label, expected_stratum in expected_strata.items():
            vector = director.mic.get_basis_vector(label)
            assert vector is not None, f"Step '{label}' not registered"
            assert vector.stratum == expected_stratum, (
                f"Step '{label}': expected {expected_stratum.name}, "
                f"got {vector.stratum.name}"
            )

    def test_registration_order_matches_enum_order(self, director):
        """
        El orden de registro en la MIC debe coincidir con el orden
        del enum PipelineSteps (que define la secuencia de ejecución).
        """
        registered_order = director.mic.get_available_labels()
        enum_order = [step.value for step in PipelineSteps]
        assert registered_order == enum_order

    def test_run_single_step_success(self, director):
        """Un paso exitoso retorna status='success' y persiste el contexto."""
        SuccessStep = make_stub_class(output_keys={"result": "computed"})
        director.mic.add_basis_vector("stub_ok", SuccessStep, Stratum.PHYSICS)

        session_id = "test_success"
        initial = {"input": "data"}

        result = director.run_single_step("stub_ok", session_id, initial_context=initial)

        assert result["status"] == "success"
        assert result["step"] == "stub_ok"
        assert result["stratum"] == Stratum.PHYSICS.name

        # Verificar persistencia
        saved = director._load_context_state(session_id)
        assert saved["input"] == "data"
        assert saved["result"] == "computed"

    def test_run_single_step_error_returns_error_status(self, director, telemetry):
        """Un paso que lanza excepción retorna status='error'."""
        FailStep = make_stub_class(raise_on_execute=ValueError("boom"))
        director.mic.add_basis_vector("stub_fail", FailStep, Stratum.PHYSICS)

        result = director.run_single_step("stub_fail", "test_fail")

        assert result["status"] == "error"
        assert "boom" in result["error"]
        telemetry.record_error.assert_called()

    def test_run_single_step_preserves_pre_error_context(self, director):
        """
        Tras un fallo, el contexto previo al paso no es sobrescrito.
        Esto permite análisis forense del estado antes del error.
        """
        session_id = "pre_error"
        original_context = {"safe_data": "before_failure"}
        director._save_context_state(session_id, original_context)

        FailStep = make_stub_class(raise_on_execute=RuntimeError("crash"))
        director.mic.add_basis_vector("crasher", FailStep, Stratum.PHYSICS)

        director.run_single_step("crasher", session_id)

        # El contexto original debe seguir intacto
        preserved = director._load_context_state(session_id)
        assert preserved == original_context

    def test_run_single_step_nonexistent_step(self, director):
        """Solicitar un paso inexistente retorna error con nombres disponibles."""
        result = director.run_single_step("fantasma", "test_404")

        assert result["status"] == "error"
        assert "fantasma" in result["error"]

    def test_run_single_step_null_context_from_step_is_error(self, director):
        """Un paso que retorna None genera error explícito."""

        class NullStep(ProcessingStep):
            def __init__(self, config=None, thresholds=None):
                pass

            def execute(self, context, telemetry):
                return None  # Bug en el paso

        director.mic.add_basis_vector("null_returner", NullStep, Stratum.PHYSICS)

        result = director.run_single_step("null_returner", "test_null")
        assert result["status"] == "error"
        assert "None context" in result["error"] or "null" in result["error"].lower()

    def test_run_single_step_session_context_takes_precedence(self, director):
        """
        Si el contexto de sesión y initial_context tienen la misma clave,
        el valor de sesión prevalece (evita regresión de estado).
        """
        session_id = "precedence_test"
        director._save_context_state(session_id, {"key": "from_session"})

        ReadStep = make_stub_class(output_keys={})

        class InspectorStep(ProcessingStep):
            captured_value = None

            def __init__(self, config=None, thresholds=None):
                pass

            def execute(self, context, telemetry):
                InspectorStep.captured_value = context.get("key")
                return context

        director.mic.add_basis_vector("inspector", InspectorStep, Stratum.PHYSICS)

        director.run_single_step(
            "inspector", session_id, initial_context={"key": "from_initial"}
        )

        assert InspectorStep.captured_value == "from_session"

    def test_stratum_regression_warning(self, director, caplog):
        """
        Ejecutar un paso de estrato inferior al contexto actual genera warning.
        
        Ejemplo: contexto en TACTICS, paso objetivo en PHYSICS.
        """
        session_id = "regression_test"
        # Crear contexto que se infiere como TACTICS
        tactics_ctx = {"df_apu_costos": pd.DataFrame(), "df_tiempo": pd.DataFrame()}
        director._save_context_state(session_id, tactics_ctx)

        PhysicsStep = make_stub_class(output_keys={"extra": True})
        director.mic.add_basis_vector("regressor", PhysicsStep, Stratum.PHYSICS)

        with caplog.at_level(logging.WARNING):
            director.run_single_step("regressor", session_id)

        assert any(
            "regression" in record.message.lower() for record in caplog.records
        ), (
            f"Expected 'regression' warning in logs. Got: "
            f"{[r.message for r in caplog.records]}"
        )


# ==================== 6. TESTS PARA ORQUESTACIÓN COMPLETA ====================


class TestPipelineOrchestration:
    """Tests para execute_pipeline_orchestrated."""

    def _build_stub_pipeline(self, director, steps_config):
        """
        Reemplaza los pasos de la MIC del director con stubs controlables.
        
        Args:
            steps_config: list of (label, output_keys, stratum)
        """
        # Reconstruir la MIC desde cero
        director.mic = MICRegistry()
        for label, output_keys, stratum in steps_config:
            StubClass = make_stub_class(output_keys=output_keys)
            director.mic.add_basis_vector(label, StubClass, stratum)

    def test_full_pipeline_runs_all_steps_in_order(self, director, base_config):
        """Todos los pasos habilitados se ejecutan en secuencia."""
        execution_log = []

        def make_logging_step(name):
            class LogStep(ProcessingStep):
                def __init__(self, config=None, thresholds=None):
                    pass

                def execute(self, context, telemetry):
                    execution_log.append(name)
                    return {**context, f"done_{name}": True}

            return LogStep

        director.mic = MICRegistry()
        step_names = ["step_1", "step_2", "step_3"]
        for name in step_names:
            director.mic.add_basis_vector(
                name, make_logging_step(name), Stratum.PHYSICS
            )

        result = director.execute_pipeline_orchestrated({"seed": True})

        assert execution_log == step_names
        assert result["done_step_1"] is True
        assert result["done_step_3"] is True

    def test_disabled_step_is_skipped(self, director, base_config):
        """Un paso deshabilitado en la receta no se ejecuta."""
        execution_log = []

        def make_logging_step(name):
            class LogStep(ProcessingStep):
                def __init__(self, config=None, thresholds=None):
                    pass

                def execute(self, context, telemetry):
                    execution_log.append(name)
                    return context

            return LogStep

        director.mic = MICRegistry()
        director.mic.add_basis_vector("active", make_logging_step("active"), Stratum.PHYSICS)
        director.mic.add_basis_vector("inactive", make_logging_step("inactive"), Stratum.PHYSICS)

        # Configurar receta con paso deshabilitado
        base_config["pipeline_recipe"] = [
            {"step": "active", "enabled": True},
            {"step": "inactive", "enabled": False},
        ]
        director.config = base_config

        director.execute_pipeline_orchestrated({})

        assert "active" in execution_log
        assert "inactive" not in execution_log

    def test_pipeline_failure_raises_runtime_error(self, director):
        """Si un paso falla, el pipeline lanza RuntimeError."""
        director.mic = MICRegistry()

        OkStep = make_stub_class(output_keys={"ok": True})
        FailStep = make_stub_class(raise_on_execute=ValueError("step 2 exploded"))

        director.mic.add_basis_vector("ok_step", OkStep, Stratum.PHYSICS)
        director.mic.add_basis_vector("fail_step", FailStep, Stratum.PHYSICS)

        with pytest.raises(RuntimeError, match="step 2 exploded"):
            director.execute_pipeline_orchestrated({})

    def test_pipeline_failure_preserves_session_for_forensics(self, director):
        """
        Tras un fallo, el archivo de sesión NO se elimina
        para permitir análisis forense.
        """
        director.mic = MICRegistry()
        FailStep = make_stub_class(raise_on_execute=ValueError("forensic_test"))
        director.mic.add_basis_vector("fail", FailStep, Stratum.PHYSICS)

        try:
            director.execute_pipeline_orchestrated({"evidence": True})
        except RuntimeError:
            pass

        # Verificar que hay al menos un archivo .pkl en el directorio de sesiones
        session_files = list(director.session_dir.glob("*.pkl"))
        assert len(session_files) >= 1, "Session file should be preserved for forensics"

    def test_pipeline_success_cleans_session(self, director):
        """Tras éxito, el archivo de sesión se elimina."""
        director.mic = MICRegistry()
        OkStep = make_stub_class(output_keys={"done": True})
        director.mic.add_basis_vector("only_step", OkStep, Stratum.PHYSICS)

        director.execute_pipeline_orchestrated({})

        session_files = list(director.session_dir.glob("*.pkl"))
        assert session_files == [], "Session file should be cleaned after success"

    def test_pipeline_verifies_initial_save(self, director, base_config, telemetry):
        """
        Si el guardado inicial del contexto falla (round-trip verification),
        se lanza IOError antes de ejecutar cualquier paso.
        """
        # Hacer que la carga retorne None (simulando fallo de I/O)
        with patch.object(director, "_load_context_state", return_value=None):
            with patch.object(director, "_save_context_state"):
                with pytest.raises(IOError, match="Failed to persist"):
                    director.execute_pipeline_orchestrated({"data": True})

    def test_custom_recipe_overrides_default(self, director, base_config):
        """Una receta personalizada en config reemplaza la secuencia por defecto."""
        execution_log = []

        def make_step(name):
            class S(ProcessingStep):
                def __init__(self, config=None, thresholds=None):
                    pass

                def execute(self, context, telemetry):
                    execution_log.append(name)
                    return context

            return S

        director.mic = MICRegistry()
        director.mic.add_basis_vector("a", make_step("a"), Stratum.PHYSICS)
        director.mic.add_basis_vector("b", make_step("b"), Stratum.PHYSICS)
        director.mic.add_basis_vector("c", make_step("c"), Stratum.PHYSICS)

        # Solo ejecutar c y a (no b), en ese orden
        base_config["pipeline_recipe"] = [
            {"step": "c", "enabled": True},
            {"step": "a", "enabled": True},
        ]
        director.config = base_config

        director.execute_pipeline_orchestrated({})

        assert execution_log == ["c", "a"]

    def test_recipe_entry_without_step_key_is_skipped(self, director, caplog):
        """Una entrada de receta sin clave 'step' se omite con warning."""
        director.mic = MICRegistry()
        OkStep = make_stub_class()
        director.mic.add_basis_vector("valid", OkStep, Stratum.PHYSICS)

        director.config["pipeline_recipe"] = [
            {"enabled": True},  # Sin 'step' key
            {"step": "valid", "enabled": True},
        ]

        with caplog.at_level(logging.WARNING):
            director.execute_pipeline_orchestrated({})

        assert any("no 'step' key" in r.message for r in caplog.records)


# ==================== 7. TESTS PARA _load_thresholds ====================


class TestLoadThresholds:
    """Tests para la carga de umbrales desde configuración."""

    def test_default_thresholds_when_missing(self, telemetry):
        """Sin clave processing_thresholds, retorna defaults."""
        config = {"session_dir": "/tmp/test"}
        d = PipelineDirector(config, telemetry)
        assert isinstance(d.thresholds, ProcessingThresholds)

    def test_valid_override_applied(self, telemetry, tmp_path):
        """Overrides válidos se aplican correctamente."""
        default = ProcessingThresholds()
        # Buscar un atributo numérico para override
        numeric_attrs = [
            a
            for a in dir(default)
            if not a.startswith("_")
            and isinstance(getattr(default, a), (int, float))
        ]

        if numeric_attrs:
            attr = numeric_attrs[0]
            original_value = getattr(default, attr)
            new_value = original_value + 42 if isinstance(original_value, int) else original_value + 0.42

            config = {
                "session_dir": str(tmp_path),
                "processing_thresholds": {attr: new_value},
            }
            d = PipelineDirector(config, telemetry)
            assert getattr(d.thresholds, attr) == new_value

    def test_unknown_key_ignored_with_warning(self, telemetry, tmp_path, caplog):
        """Claves desconocidas generan warning y se ignoran."""
        config = {
            "session_dir": str(tmp_path),
            "processing_thresholds": {"nonexistent_threshold_xyz": 999},
        }

        with caplog.at_level(logging.WARNING):
            d = PipelineDirector(config, telemetry)

        assert any("Unknown threshold" in r.message or "nonexistent" in r.message for r in caplog.records)

    def test_wrong_type_ignored_with_warning(self, telemetry, tmp_path, caplog):
        """
        Un override con tipo incorrecto se rechaza sin aplicar.
        Ej: si el default es float, pasar un string debe ignorarse.
        """
        default = ProcessingThresholds()
        numeric_attrs = [
            a
            for a in dir(default)
            if not a.startswith("_")
            and isinstance(getattr(default, a), (int, float))
        ]

        if numeric_attrs:
            attr = numeric_attrs[0]
            config = {
                "session_dir": str(tmp_path),
                "processing_thresholds": {attr: "not_a_number"},
            }

            with caplog.at_level(logging.WARNING):
                d = PipelineDirector(config, telemetry)

            # El valor debe seguir siendo el default
            assert getattr(d.thresholds, attr) == getattr(default, attr)

    def test_non_dict_thresholds_uses_defaults(self, telemetry, tmp_path, caplog):
        """Si processing_thresholds no es dict, se usan defaults."""
        config = {
            "session_dir": str(tmp_path),
            "processing_thresholds": "invalid_not_a_dict",
        }

        with caplog.at_level(logging.WARNING):
            d = PipelineDirector(config, telemetry)

        assert isinstance(d.thresholds, ProcessingThresholds)


# ==================== 8. TESTS PARA PASOS INDIVIDUALES ====================


class TestProcessingStepBase:
    """Tests para la clase base abstracta ProcessingStep."""

    def test_cannot_instantiate_abstract_class(self):
        """ProcessingStep es abstracta y no se puede instanciar directamente."""
        with pytest.raises(TypeError):
            ProcessingStep()

    def test_subclass_must_implement_execute(self):
        """Una subclase que no implementa execute() no se puede instanciar."""

        class IncompleteStep(ProcessingStep):
            pass

        with pytest.raises(TypeError):
            IncompleteStep()

    def test_subclass_with_execute_is_instantiable(self):
        """Una subclase que implementa execute() se instancia correctamente."""

        class CompleteStep(ProcessingStep):
            def execute(self, context, telemetry):
                return context

        step = CompleteStep()
        assert step is not None


class TestAuditedMergeStep:
    """Tests para AuditedMergeStep con null safety refinada."""

    def test_raises_when_df_apus_raw_is_none(self, telemetry, base_config):
        """Si df_apus_raw es None, lanza ValueError."""
        step = AuditedMergeStep(base_config, ProcessingThresholds())
        context = {
            "df_presupuesto": pd.DataFrame(),
            "df_apus_raw": None,
            "df_insumos": pd.DataFrame(),
        }

        with pytest.raises(ValueError, match="df_apus_raw"):
            step.execute(context, telemetry)

    def test_raises_when_df_insumos_is_none(self, telemetry, base_config):
        """Si df_insumos es None, lanza ValueError."""
        step = AuditedMergeStep(base_config, ProcessingThresholds())
        context = {
            "df_presupuesto": pd.DataFrame(),
            "df_apus_raw": pd.DataFrame({"a": [1]}),
            "df_insumos": None,
        }

        with pytest.raises(ValueError, match="df_insumos"):
            step.execute(context, telemetry)

    def test_audit_skipped_when_presupuesto_missing(
        self, telemetry, base_config, caplog, sample_apus_df, sample_insumos_df
    ):
        """
        Si df_presupuesto es None, la auditoría Mayer-Vietoris se omite
        pero la fusión física continúa.
        """
        step = AuditedMergeStep(base_config, ProcessingThresholds())
        context = {
            "df_presupuesto": None,
            "df_apus_raw": sample_apus_df,
            "df_insumos": sample_insumos_df,
        }

        with patch(
            "app.pipeline_director.DataMerger"
        ) as MockMerger:
            mock_instance = MockMerger.return_value
            mock_instance.merge_apus_with_insumos.return_value = pd.DataFrame(
                {"merged": [1]}
            )

            with caplog.at_level(logging.INFO):
                result = step.execute(context, telemetry)

        assert "df_merged" in result
        assert any("omitida" in r.message or "no disponible" in r.message for r in caplog.records)


class TestCalculateCostsStep:
    """Tests para CalculateCostsStep."""

    def test_raises_when_df_merged_missing(self, telemetry, base_config):
        """Si df_merged no está en contexto, lanza ValueError."""
        step = CalculateCostsStep(base_config, ProcessingThresholds())

        with pytest.raises(ValueError, match="df_merged"):
            step.execute({}, telemetry)

    def test_raises_when_df_merged_is_empty(self, telemetry, base_config):
        """Si df_merged es un DataFrame vacío, lanza ValueError."""
        step = CalculateCostsStep(base_config, ProcessingThresholds())

        with pytest.raises(ValueError, match="df_merged"):
            step.execute({"df_merged": pd.DataFrame()}, telemetry)

    def test_does_not_overwrite_df_merged(self, telemetry, base_config):
        """
        CalculateCostsStep no debe modificar df_merged en el contexto.
        Solo agrega df_apu_costos, df_tiempo, df_rendimiento.
        """
        step = CalculateCostsStep(base_config, ProcessingThresholds())
        # Inject mock MIC
        mock_mic = MagicMock()
        mock_mic.project_intent.return_value = {"success": True, "processed_data": []}
        step.mic = mock_mic

        original_df = pd.DataFrame({"original": [1, 2, 3]})
        context = {"df_merged": original_df}

        with patch("app.pipeline_director.APUProcessor") as MockProc:
            mock_proc = MockProc.return_value
            mock_proc.process_vectors.return_value = (
                pd.DataFrame({"costo": [100]}),
                pd.DataFrame({"tiempo": [8]}),
                pd.DataFrame({"rendimiento": [0.9]}),
            )
            result = step.execute(context, telemetry)

        # df_merged debe ser exactamente el mismo objeto
        pd.testing.assert_frame_equal(result["df_merged"], original_df)
        assert "df_apu_costos" in result
        assert "df_tiempo" in result
        assert "df_rendimiento" in result


class TestBusinessTopologyStep:
    """Tests para BusinessTopologyStep."""

    def test_raises_when_df_final_missing(self, telemetry, base_config):
        """Si df_final es None, lanza ValueError."""
        step = BusinessTopologyStep(base_config, ProcessingThresholds())

        with pytest.raises(ValueError, match="df_final"):
            step.execute({}, telemetry)

    def test_graph_is_materialized(self, telemetry, base_config):
        """El paso materializa el grafo y lo agrega al contexto."""
        step = BusinessTopologyStep(base_config, ProcessingThresholds())
        context = {
            "df_final": pd.DataFrame({"id": [1]}),
            "df_merged": pd.DataFrame({"merged": [1]}),
        }

        mock_graph = MagicMock()
        mock_graph.number_of_nodes.return_value = 3
        mock_graph.number_of_edges.return_value = 2

        with patch(
            "app.pipeline_director.BudgetGraphBuilder"
        ) as MockBuilder, patch.object(
            step, "_resolve_mic_instance", return_value=None
        ):
            MockBuilder.return_value.build.return_value = mock_graph
            result = step.execute(context, telemetry)

        assert "graph" in result
        assert result["graph"] is mock_graph
        telemetry.record_metric.assert_any_call(
            "business_topology", "graph_nodes", 3
        )

    def test_validated_strata_updated(self, telemetry, base_config):
        """El paso agrega PHYSICS y TACTICS a validated_strata."""
        step = BusinessTopologyStep(base_config, ProcessingThresholds())
        context = {
            "df_final": pd.DataFrame({"id": [1]}),
            "validated_strata": set(),
        }

        mock_graph = MagicMock()
        mock_graph.number_of_nodes.return_value = 1
        mock_graph.number_of_edges.return_value = 0

        with patch(
            "app.pipeline_director.BudgetGraphBuilder"
        ) as MockBuilder, patch.object(
            step, "_resolve_mic_instance", return_value=None
        ):
            MockBuilder.return_value.build.return_value = mock_graph
            result = step.execute(context, telemetry)

        assert Stratum.PHYSICS in result["validated_strata"]
        assert Stratum.TACTICS in result["validated_strata"]

    def test_agent_failure_is_degraded_not_fatal(
        self, telemetry, base_config, caplog
    ):
        """
        Si BusinessAgent falla, el paso completa con warning
        (degradación controlada), no con error fatal.
        """
        step = BusinessTopologyStep(base_config, ProcessingThresholds())
        context = {"df_final": pd.DataFrame({"id": [1]})}

        mock_graph = MagicMock()
        mock_graph.number_of_nodes.return_value = 1
        mock_graph.number_of_edges.return_value = 0
        mock_mic = MagicMock()

        with patch(
            "app.pipeline_director.BudgetGraphBuilder"
        ) as MockBuilder, patch.object(
            step, "_resolve_mic_instance", return_value=mock_mic
        ), patch(
            "app.pipeline_director.BusinessAgent"
        ) as MockAgent:
            MockBuilder.return_value.build.return_value = mock_graph
            MockAgent.return_value.evaluate_project.side_effect = RuntimeError(
                "Agent crashed"
            )

            with caplog.at_level(logging.WARNING):
                result = step.execute(context, telemetry)

        # Debe completar exitosamente (no raise)
        assert "graph" in result
        telemetry.end_step.assert_called_with("business_topology", "success")


class TestMaterializationStep:
    """Tests para MaterializationStep."""

    def test_skips_when_no_topology_report(self, telemetry, base_config, caplog):
        """Sin business_topology_report, la materialización se omite sin error."""
        step = MaterializationStep(base_config, ProcessingThresholds())
        context = {"df_final": pd.DataFrame()}

        with caplog.at_level(logging.WARNING):
            result = step.execute(context, telemetry)

        assert "bill_of_materials" not in result
        telemetry.end_step.assert_called_with("materialization", "skipped")

    def test_materializes_bom_when_report_present(self, telemetry, base_config):
        """Con reporte topológico, genera BOM correctamente."""
        step = MaterializationStep(base_config, ProcessingThresholds())

        mock_graph = MagicMock()
        mock_report = MagicMock()
        mock_report.details = {"pyramid_stability": 7.5}

        mock_bom = MagicMock()
        mock_bom.requirements = [1, 2, 3]

        context = {
            "graph": mock_graph,
            "business_topology_report": mock_report,
        }

        with patch(
            "app.pipeline_director.MatterGenerator"
        ) as MockGen, patch(
            "app.pipeline_director.asdict", return_value={"items": 3}
        ):
            MockGen.return_value.materialize_project.return_value = mock_bom
            result = step.execute(context, telemetry)

        assert "bill_of_materials" in result
        assert result["bill_of_materials"] is mock_bom
        assert "logistics_plan" in result

    def test_raises_when_graph_missing_and_df_final_missing(
        self, telemetry, base_config
    ):
        """Sin grafo ni df_final, la materialización falla con ValueError."""
        step = MaterializationStep(base_config, ProcessingThresholds())
        context = {
            "business_topology_report": MagicMock(),
            # Sin 'graph' ni 'df_final'
        }

        with pytest.raises(ValueError, match="df_final"):
            step.execute(context, telemetry)


class TestBuildOutputStep:
    """Tests para BuildOutputStep."""

    def test_raises_when_required_keys_missing(self, telemetry, base_config):
        """Sin claves requeridas, lanza ValueError descriptivo."""
        step = BuildOutputStep(base_config, ProcessingThresholds())

        with pytest.raises(ValueError, match="missing required context keys"):
            step.execute({"partial": True}, telemetry)

    def test_lineage_hash_covers_full_payload(self, telemetry, base_config):
        """
        El hash de linaje debe cambiar cuando CUALQUIER clave del payload
        cambia, no solo 'presupuesto'.
        """
        step = BuildOutputStep(base_config, ProcessingThresholds())

        payload_a = {"presupuesto": [{"id": 1}], "insumos": [{"codigo": "I001"}]}
        payload_b = {"presupuesto": [{"id": 1}], "insumos": [{"codigo": "I999"}]}

        hash_a = step._compute_lineage_hash(payload_a)
        hash_b = step._compute_lineage_hash(payload_b)

        assert hash_a != hash_b, (
            "Hashes should differ when non-presupuesto keys change"
        )

    def test_lineage_hash_deterministic(self, telemetry, base_config):
        """El mismo payload produce el mismo hash."""
        step = BuildOutputStep(base_config, ProcessingThresholds())
        payload = {"key_a": [1, 2, 3], "key_b": {"nested": True}}

        assert step._compute_lineage_hash(payload) == step._compute_lineage_hash(
            payload
        )

    def test_lineage_hash_handles_non_serializable(self, telemetry, base_config):
        """Objetos no serializables no causan fallo en el hash."""
        step = BuildOutputStep(base_config, ProcessingThresholds())
        payload = {"normal": [1, 2], "weird": object()}

        # No debe lanzar excepción
        result = step._compute_lineage_hash(payload)
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex digest

    def test_output_structure_is_data_product(self, telemetry, base_config):
        """La salida tiene estructura DataProduct con metadata y payload."""
        step = BuildOutputStep(base_config, ProcessingThresholds())

        # Construir contexto mínimo válido
        context = {
            "df_final": pd.DataFrame({"id": [1]}),
            "df_insumos": pd.DataFrame({"codigo": ["I001"]}),
            "df_merged": pd.DataFrame({"m": [1]}),
            "df_apus_raw": pd.DataFrame({"a": [1]}),
            "df_apu_costos": pd.DataFrame({"c": [1]}),
            "df_tiempo": pd.DataFrame({"t": [1]}),
            "df_rendimiento": pd.DataFrame({"r": [1]}),
        }

        with patch(
            "app.pipeline_director.synchronize_data_sources",
            return_value=context["df_merged"],
        ), patch(
            "app.pipeline_director.build_processed_apus_dataframe",
            return_value=pd.DataFrame(),
        ), patch(
            "app.pipeline_director.build_output_dictionary",
            return_value={"data": True},
        ), patch(
            "app.pipeline_director.validate_and_clean_data",
            return_value={"data": True},
        ), patch(
            "app.pipeline_director.TelemetryNarrator"
        ) as MockNarr:
            MockNarr.return_value.summarize_execution.return_value = {"summary": "ok"}
            result = step.execute(context, telemetry)

        data_product = result["final_result"]
        assert data_product["kind"] == "DataProduct"
        assert "metadata" in data_product
        assert "payload" in data_product
        assert "lineage_hash" in data_product["metadata"]
        assert "generated_at" in data_product["metadata"]
        assert data_product["metadata"]["version"] == "3.0"

    def test_output_includes_strata_validated(self, telemetry, base_config):
        """La metadata incluye los estratos validados del contexto."""
        step = BuildOutputStep(base_config, ProcessingThresholds())

        context = {
            "df_final": pd.DataFrame({"id": [1]}),
            "df_insumos": pd.DataFrame({"codigo": ["I001"]}),
            "df_merged": pd.DataFrame({"m": [1]}),
            "df_apus_raw": pd.DataFrame({"a": [1]}),
            "df_apu_costos": pd.DataFrame({"c": [1]}),
            "df_tiempo": pd.DataFrame({"t": [1]}),
            "df_rendimiento": pd.DataFrame({"r": [1]}),
            "validated_strata": {Stratum.PHYSICS, Stratum.TACTICS},
        }

        with patch(
            "app.pipeline_director.synchronize_data_sources",
            return_value=context["df_merged"],
        ), patch(
            "app.pipeline_director.build_processed_apus_dataframe",
            return_value=pd.DataFrame(),
        ), patch(
            "app.pipeline_director.build_output_dictionary",
            return_value={"data": True},
        ), patch(
            "app.pipeline_director.validate_and_clean_data",
            return_value={"data": True},
        ), patch(
            "app.pipeline_director.TelemetryNarrator"
        ) as MockNarr:
            MockNarr.return_value.summarize_execution.return_value = {}
            result = step.execute(context, telemetry)

        strata_names = result["final_result"]["metadata"]["strata_validated"]
        assert "PHYSICS" in strata_names
        assert "TACTICS" in strata_names


# ==================== 9. TESTS PARA process_all_files ====================


class TestProcessAllFiles:
    """Tests para la función de entrada principal."""

    def test_creates_default_telemetry_when_none(self, base_config, tmp_path):
        """Si telemetry=None, crea una instancia por defecto sin fallar."""
        # Creamos archivos ficticios para que la validación de existencia pase
        for name in ["presupuesto.csv", "apus.csv", "insumos.csv"]:
            (tmp_path / name).touch()

        with patch(
            "app.pipeline_director.PipelineDirector"
        ) as MockDirector:
            mock_instance = MockDirector.return_value
            mock_instance.execute_pipeline_orchestrated.return_value = {
                "final_result": {"kind": "DataProduct"}
            }

            # telemetry=None NO debe lanzar excepción
            result = process_all_files(
                presupuesto_path=tmp_path / "presupuesto.csv",
                apus_path=tmp_path / "apus.csv",
                insumos_path=tmp_path / "insumos.csv",
                config=base_config,
                telemetry=None,
            )

            assert result is not None

    def test_raises_file_not_found_for_missing_files(self, base_config, tmp_path):
        """Archivos inexistentes lanzan FileNotFoundError antes del pipeline."""
        with pytest.raises(FileNotFoundError, match="presupuesto_path"):
            process_all_files(
                presupuesto_path=tmp_path / "nonexistent.csv",
                apus_path=tmp_path / "also_missing.csv",
                insumos_path=tmp_path / "nope.csv",
                config=base_config,
            )

    def test_returns_data_product_not_raw_context(self, base_config, tmp_path):
        """
        process_all_files retorna el DataProduct (final_result),
        no el contexto completo del pipeline.
        """
        for name in ["p.csv", "a.csv", "i.csv"]:
            (tmp_path / name).touch()

        expected_product = {
            "kind": "DataProduct",
            "metadata": {"version": "3.0"},
            "payload": {},
        }

        with patch(
            "app.pipeline_director.PipelineDirector"
        ) as MockDirector:
            mock_instance = MockDirector.return_value
            mock_instance.execute_pipeline_orchestrated.return_value = {
                "final_result": expected_product,
                "df_presupuesto": pd.DataFrame(),  # contexto interno
            }

            result = process_all_files(
                presupuesto_path=tmp_path / "p.csv",
                apus_path=tmp_path / "a.csv",
                insumos_path=tmp_path / "i.csv",
                config=base_config,
            )

        assert result["kind"] == "DataProduct"
        assert "df_presupuesto" not in result  # No expone contexto interno

    def test_none_config_uses_empty_dict(self, tmp_path):
        """Si config=None, se usa un diccionario vacío sin fallar."""
        for name in ["p.csv", "a.csv", "i.csv"]:
            (tmp_path / name).touch()

        with patch(
            "app.pipeline_director.PipelineDirector"
        ) as MockDirector:
            mock_instance = MockDirector.return_value
            mock_instance.execute_pipeline_orchestrated.return_value = {
                "final_result": {"kind": "DataProduct"}
            }

            # config=None NO debe lanzar excepción
            result = process_all_files(
                presupuesto_path=tmp_path / "p.csv",
                apus_path=tmp_path / "a.csv",
                insumos_path=tmp_path / "i.csv",
                config=None,
            )

            assert result is not None

    def test_propagates_pipeline_exception(self, base_config, tmp_path):
        """Excepciones del pipeline se propagan al llamador."""
        for name in ["p.csv", "a.csv", "i.csv"]:
            (tmp_path / name).touch()

        with patch(
            "app.pipeline_director.PipelineDirector"
        ) as MockDirector:
            mock_instance = MockDirector.return_value
            mock_instance.execute_pipeline_orchestrated.side_effect = RuntimeError(
                "Pipeline exploded"
            )

            with pytest.raises(RuntimeError, match="Pipeline exploded"):
                process_all_files(
                    presupuesto_path=tmp_path / "p.csv",
                    apus_path=tmp_path / "a.csv",
                    insumos_path=tmp_path / "i.csv",
                    config=base_config,
                )


# ==================== 10. TESTS DE INTEGRACIÓN LIGERA ====================


class TestLightIntegration:
    """
    Tests de integración ligera que verifican el flujo completo
    con stubs controlados (sin dependencias externas).
    """

    def test_three_step_pipeline_accumulates_context(self, base_config, telemetry):
        """
        Un pipeline de 3 pasos acumula correctamente las claves de contexto.
        Cada paso agrega una clave; el contexto final las tiene todas.
        """
        director = PipelineDirector(base_config, telemetry)
        director.mic = MICRegistry()

        Step1 = make_stub_class({"phase_1": True})
        Step2 = make_stub_class({"phase_2": True})
        Step3 = make_stub_class({"phase_3": True})

        director.mic.add_basis_vector("s1", Step1, Stratum.PHYSICS)
        director.mic.add_basis_vector("s2", Step2, Stratum.TACTICS)
        director.mic.add_basis_vector("s3", Step3, Stratum.STRATEGY)

        result = director.execute_pipeline_orchestrated({"seed": 42})

        assert result["seed"] == 42
        assert result["phase_1"] is True
        assert result["phase_2"] is True
        assert result["phase_3"] is True

    def test_strata_monotonicity_in_full_pipeline(self, base_config, telemetry):
        """
        A lo largo del pipeline canónico completo (con stubs),
        el estrato inferido nunca decrece.
        """
        director = PipelineDirector(base_config, telemetry)
        director.mic = MICRegistry()

        # Simular la evolución del estrato a través de 4 pasos
        steps = [
            ("p1", {"df_presupuesto": True}, Stratum.PHYSICS),
            ("t1", {"df_apu_costos": True}, Stratum.TACTICS),
            ("s1", {"graph": True}, Stratum.STRATEGY),
            ("w1", {"final_result": True}, Stratum.WISDOM),
        ]

        for label, keys, stratum in steps:
            director.mic.add_basis_vector(label, make_stub_class(keys), stratum)

        result = director.execute_pipeline_orchestrated({})

        # El contexto final debe tener la huella de todos los estratos
        assert director._infer_current_stratum_from_context(result) == Stratum.WISDOM

    def test_mid_pipeline_failure_stops_execution(self, base_config, telemetry):
        """Si el paso 2 de 3 falla, el paso 3 nunca se ejecuta."""
        director = PipelineDirector(base_config, telemetry)
        director.mic = MICRegistry()

        step3_executed = {"flag": False}

        class Step3(ProcessingStep):
            def __init__(self, config=None, thresholds=None):
                pass

            def execute(self, context, telemetry):
                step3_executed["flag"] = True
                return context

        director.mic.add_basis_vector(
            "s1", make_stub_class({"ok": True}), Stratum.PHYSICS
        )
        director.mic.add_basis_vector(
            "s2",
            make_stub_class(raise_on_execute=ValueError("fail")),
            Stratum.PHYSICS,
        )
        director.mic.add_basis_vector("s3", Step3, Stratum.PHYSICS)

        with pytest.raises(RuntimeError):
            director.execute_pipeline_orchestrated({})

        assert step3_executed["flag"] is False
