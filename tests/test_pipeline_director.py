"""
Tests para Pipeline Director V2 - Suite Refinada
Valida:
- Funcionalidad del MICRegistry (registro, unicidad, obtención)
- Orquestación del PipelineDirector (pasos, persistencia, errores)
- Comportamiento de los ProcessingSteps
- Preservación de invariantes lógicos del flujo de datos (claves de contexto, tipos de salida)
"""

import pytest
import tempfile
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from app.pipeline_director_v2 import (
    PipelineDirector, MICRegistry, BasisVector, ProcessingStep,
    PipelineStepExecutionError, PipelineSteps, Stratum
)
from app.telemetry import TelemetryContext
from app.schemas import Stratum # Asumiendo Stratum está aquí o se importa
from app.apu_processor import ProcessingThresholds

# --- Fixtures Generales ---

@pytest.fixture
def config():
    """Configuración base del pipeline."""
    return {
        "session_dir": tempfile.mkdtemp(),
        "pipeline_recipe": [
            {"step": "load_data", "enabled": True},
            {"step": "audited_merge", "enabled": True},
            {"step": "calculate_costs", "enabled": True},
            # ... otros pasos según sea necesario
        ],
        "file_profiles": {},
        "processing_thresholds": {}
    }

@pytest.fixture
def thresholds():
    """Umbrales de procesamiento."""
    return ProcessingThresholds()

@pytest.fixture
def telemetry():
    """Contexto de telemetría mock."""
    return Mock(spec=TelemetryContext)

@pytest.fixture
def director(config, telemetry):
    """Instancia del director del pipeline v2."""
    return PipelineDirector(config, telemetry)


# --- Fixtures para DataFrames de prueba ---

@pytest.fixture
def sample_presupuesto_df():
    """DataFrame de presupuesto de ejemplo."""
    return pd.DataFrame({
        "id": [1, 2, 3],
        "descripcion": ["Item A", "Item B", "Item C"],
        "cantidad": [10, 20, 30]
    })

@pytest.fixture
def sample_insumos_df():
    """DataFrame de insumos de ejemplo."""
    return pd.DataFrame({
        "codigo": ["I001", "I002", "I003"],
        "nombre": ["Insumo X", "Insumo Y", "Insumo Z"],
        "precio_unitario": [1.5, 2.0, 2.5]
    })

@pytest.fixture
def sample_apus_df():
    """DataFrame de APUs de ejemplo."""
    return pd.DataFrame({
        "id_apu": [1, 1, 2],
        "codigo_insumo": ["I001", "I002", "I003"],
        "cantidad_insumo": [1.0, 0.5, 2.0]
    })


# --- Tests para MICRegistry ---

class TestMICRegistry:
    """Tests para la funcionalidad del MICRegistry."""

    def test_registry_initialization(self):
        """La MICRegistry se inicializa vacía."""
        registry = MICRegistry()
        assert registry.get_available_labels() == []
        assert registry.get_rank() == 0

    def test_add_basis_vector_success(self, config, thresholds):
        """Agregar un paso al registro con éxito."""
        from app.apu_processor import LoadDataStep # Importar acá para evitar ciclos si es necesario

        registry = MICRegistry()
        registry.add_basis_vector("test_step", LoadDataStep, Stratum.PHYSICS)

        assert registry.get_rank() == 1
        vector = registry.get_basis_vector("test_step")
        assert vector is not None
        assert vector.label == "test_step"
        assert vector.operator_class == LoadDataStep
        assert vector.stratum == Stratum.PHYSICS

    def test_add_duplicate_label_raises_error(self, config, thresholds):
        """Agregar un paso con una etiqueta duplicada lanza un error."""
        from app.apu_processor import LoadDataStep, AuditedMergeStep

        registry = MICRegistry()
        registry.add_basis_vector("unique_step", LoadDataStep, Stratum.PHYSICS)

        with pytest.raises(ValueError, match="Duplicate label"):
            registry.add_basis_vector("unique_step", AuditedMergeStep, Stratum.TACTICS)

    def test_get_nonexistent_vector_returns_none(self):
        """Obtener un vector con una etiqueta inexistente retorna None."""
        registry = MICRegistry()
        assert registry.get_basis_vector("nonexistent") is None

    def test_get_available_labels(self, config, thresholds):
        """Obtener la lista de etiquetas disponibles."""
        from app.apu_processor import LoadDataStep, AuditedMergeStep

        registry = MICRegistry()
        registry.add_basis_vector("step_a", LoadDataStep, Stratum.PHYSICS)
        registry.add_basis_vector("step_b", AuditedMergeStep, Stratum.PHYSICS)

        labels = registry.get_available_labels()
        assert set(labels) == {"step_a", "step_b"}
        assert len(labels) == 2


# --- Tests para PipelineDirector ---

class TestPipelineDirector:
    """Tests para la funcionalidad del PipelineDirector V2."""

    def test_director_initialization(self, config, telemetry):
        """El director se inicializa correctamente y carga pasos en la MIC."""
        director = PipelineDirector(config, telemetry)

        assert director is not None
        assert hasattr(director, 'mic')
        assert director.mic.get_rank() > 0 # Debe tener los pasos predeterminados
        available_steps = director.mic.get_available_labels()
        expected_steps = {step.value for step in PipelineSteps}
        assert set(available_steps) == expected_steps

    def test_run_single_step_success(self, director, sample_presupuesto_df):
        """Ejecución de un paso que simula éxito."""
        # Creamos un mock para un paso que devuelve un contexto modificado
        class MockSuccessfulStep(ProcessingStep):
            def execute(self, context, telemetry):
                new_context = context.copy()
                new_context["mock_output"] = "success_data"
                return new_context

        # Agregamos el mock a la MIC del director
        director.mic.add_basis_vector("mock_success", MockSuccessfulStep, Stratum.TACTICS)

        session_id = "test_session_success"
        initial_context = {"input": "some_data"}

        result = director.run_single_step("mock_success", session_id, initial_context)

        assert result["status"] == "success"
        assert result["step"] == "mock_success"
        # Verificar que el contexto se haya guardado correctamente
        saved_context = director._load_context_state(session_id)
        assert saved_context.get("mock_output") == "success_data"
        assert saved_context.get("input") == "some_data"

    def test_run_single_step_error_handling(self, director):
        """Ejecución de un paso que lanza un PipelineStepExecutionError."""
        class MockFailingStep(ProcessingStep):
            def execute(self, context, telemetry):
                raise PipelineStepExecutionError("Simulated processing error")

        director.mic.add_basis_vector("mock_fail", MockFailingStep, Stratum.TACTICS)

        session_id = "test_session_fail"
        initial_context = {"input": "data_before_failure"}

        # Guardamos el contexto inicial para simular el estado antes del fallo
        director._save_context_state(session_id, initial_context)

        result = director.run_single_step("mock_fail", session_id)

        assert result["status"] == "error"
        assert result["step"] == "mock_fail"
        assert "Simulated processing error" in result["error"]
        # Verificar que el contexto antes del fallo esté disponible en un archivo diagnóstico
        diagnostic_context = director._load_context_state(f"{session_id}_diagnostic")
        assert diagnostic_context == initial_context # Debe coincidir con el estado antes del paso fallido

    def test_execute_pipeline_orchestrated_success(self, director, config, sample_presupuesto_df, sample_insumos_df, sample_apus_df, telemetry):
        """Ejecución completa del pipeline con pasos simulados."""
        # Simulamos la carga exitosa de archivos para el primer paso
        with patch("app.apu_processor.FileValidator.validate_file_exists", return_value=(True, None)), \
             patch("app.apu_processor.PresupuestoProcessor") as MockProc, \
             patch("app.apu_processor.InsumosProcessor") as MockIProc, \
             patch("app.apu_processor.DataFluxCondenser") as MockCond:
            
            # Configurar mocks para que devuelvan DataFrames de ejemplo
            mock_p_instance = MockProc.return_value
            mock_p_instance.process.return_value = sample_presupuesto_df
            mock_i_instance = MockIProc.return_value
            mock_i_instance.process.return_value = sample_insumos_df
            mock_cond_instance = MockCond.return_value
            mock_cond_instance.stabilize.return_value = sample_apus_df

            initial_context = {
                "presupuesto_path": "/fake/path/presupuesto.csv",
                "insumos_path": "/fake/path/insumos.csv",
                "apus_path": "/fake/path/apus.csv"
            }

            # Ejecutar el pipeline
            final_context = director.execute_pipeline_orchestrated(initial_context)

            # Verificar resultados comunes del primer paso (LoadDataStep)
            assert "df_presupuesto" in final_context
            assert "df_insumos" in final_context
            assert "df_apus_raw" in final_context
            assert len(final_context["df_presupuesto"]) == len(sample_presupuesto_df)
            assert len(final_context["df_insumos"]) == len(sample_insumos_df)
            assert len(final_context["df_apus_raw"]) == len(sample_apus_df)

    def test_execute_pipeline_orchestrated_with_disabled_step(self, director, config):
        """Un paso deshabilitado en la receta es omitido."""
        execution_log = []

        class LoggingStep(ProcessingStep):
            def __init__(self, config, thresholds):
                self.config = config
                self.thresholds = thresholds

            def execute(self, context, telemetry):
                execution_log.append(self.__class__.__name__)
                return context

        # Agregar un paso personalizado para probar
        director.mic.add_basis_vector("log_step", LoggingStep, Stratum.PHYSICS)
        
        # Modificar la receta para incluir y deshabilitar el paso
        config["pipeline_recipe"] = [
            {"step": "load_data", "enabled": True}, # Asumiendo este paso se puede mockear
            {"step": "log_step", "enabled": False}, # Este debe ser omitido
        ]

        initial_context = {
            "presupuesto_path": "/fake/path/presupuesto.csv",
            "insumos_path": "/fake/path/insumos.csv",
            "apus_path": "/fake/path/apus.csv"
        }
        
        # Mockear pasos requeridos para que no fallen antes de llegar a log_step
        with patch("app.apu_processor.FileValidator.validate_file_exists", return_value=(True, None)), \
             patch("app.apu_processor.PresupuestoProcessor") as MockProc, \
             patch("app.apu_processor.InsumosProcessor") as MockIProc, \
             patch("app.apu_processor.DataFluxCondenser") as MockCond:
                
                MockProc.return_value.process.return_value = pd.DataFrame({"id": [1]})
                MockIProc.return_value.process.return_value = pd.DataFrame({"codigo": ["I001"]})
                MockCond.return_value.stabilize.return_value = pd.DataFrame({"id_apu": [1]})

                director.execute_pipeline_orchestrated(initial_context)

        # El paso deshabilitado no debe haberse ejecutado
        assert "LoggingStep" not in execution_log

    def test_stratum_transition_warning(self, director, caplog):
        """Una transición de estrato potencialmente inválida genera un warning."""
        context_with_tactics = {"df_tiempo": pd.DataFrame()} # Simula un contexto de TACTICS
        initial_context = {"df_presupuesto": pd.DataFrame()} # Simula un contexto de PHYSICS

        # Intentar ejecutar un paso de PHYSICS después de uno de TACTICS (simulado)
        # Esto se prueba más fácilmente observando logs si se modifica la lógica de director para recibir el contexto actual como hint.
        # Dado que la lógica actual usa _infer_current_stratum_from_context, probamos esa heurística.
        inferred_stratum = director._infer_current_stratum_from_context(context_with_tactics)
        assert inferred_stratum == Stratum.TACTICS

        # Ahora, si se ejecuta un paso de PHYSICS, debería haber un warning
        # Agregar un paso físico para probar
        class MockPhysicsStep(ProcessingStep):
            def execute(self, context, telemetry):
                return context
        
        director.mic.add_basis_vector("mock_phys", MockPhysicsStep, Stratum.PHYSICS)
        
        # Ejecutar el paso físico sobre el contexto de tactics
        director.run_single_step("mock_phys", "test_regression", initial_context=context_with_tactics)
        
        # Verificar si el warning fue registrado (requiere que el director use el logger adecuadamente)
        # caplog.text contendrá el texto de los logs
        assert "Potential regression:" in caplog.text or "Potential regression:" in caplog.messages


# --- Tests para ProcessingStep y excepciones ---

class TestProcessingStep:
    """Tests para la clase base y excepciones de los pasos."""

    def test_processing_step_is_abstract(self):
        """La clase base ProcessingStep es abstracta."""
        with pytest.raises(TypeError):
            ProcessingStep()

    def test_pipeline_step_execution_error_raised(self):
        """PipelineStepExecutionError se puede lanzar y capturar."""
        with pytest.raises(PipelineStepExecutionError):
            raise PipelineStepExecutionError("Test error message")


# --- Fin de la Suite de Pruebas ---