import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import faiss
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.generate_embeddings import (
    DataValidationError,
    DataValidator,
    EmbeddingGenerator,
    EmbeddingPipeline,
    FileManager,
    InsufficientMemoryError,
    MemoryMonitor,
    ModelLoadError,
    ScriptConfig,
    ValidationStats,
)

# --- Fixtures ---


@pytest.fixture
def mock_config(tmp_path):
    """Fixture para crear una configuración de prueba."""
    return ScriptConfig(
        model_name="mock-model",
        input_file=tmp_path / "input.json",
        output_dir=tmp_path / "output",
        text_column="description",
        id_column="id",
        max_batch_size=32,
        memory_limit_gb=1.0,
        normalize_embeddings=True,
        min_text_length=3,
        max_text_length=5000,
    )


@pytest.fixture
def sample_dataframe():
    """Fixture para un DataFrame de ejemplo."""
    return pd.DataFrame(
        {
            "id": [f"ID{i}" for i in range(10)],
            "description": [f"This is test description number {i}." for i in range(10)],
        }
    )


@pytest.fixture
def sample_dataframe_with_issues():
    """Fixture para un DataFrame con datos problemáticos."""
    return pd.DataFrame(
        {
            "id": ["ID0", "ID1", "ID1", "ID3", None, "ID5", "ID6", "ID7", "ID8", "ID9"],
            "description": [
                "Valid description one",
                "Valid description two",
                "Duplicate ID entry",
                None,  # Null en descripción
                "Null ID entry",
                "ab",  # Muy corto (< min_text_length=3)
                "Valid description six",
                "Valid description seven",
                "Valid description eight",
                "Valid description nine",
            ],
        }
    )


@pytest.fixture
def mock_embedding_generator(mock_config):
    """Fixture para un EmbeddingGenerator con modelo mockeado."""
    with patch("scripts.generate_embeddings.SentenceTransformer") as MockSentenceTransformer:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(10, 384).astype(np.float32)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        MockSentenceTransformer.return_value = mock_model

        generator = EmbeddingGenerator(mock_config)
        generator.load_model()
        return generator


# --- Pruebas de ScriptConfig ---


class TestScriptConfig:
    """Pruebas para la configuración del script."""

    def test_config_creation_success(self, tmp_path):
        """La configuración debe crearse correctamente con valores válidos."""
        config = ScriptConfig(
            model_name="test-model",
            input_file=tmp_path / "test.json",
            output_dir=tmp_path / "output",
            text_column="text",
            id_column="id",
            max_batch_size=64,
            memory_limit_gb=2.0,
        )
        assert config.model_name == "test-model"
        assert config.max_batch_size == 64
        assert config.min_text_length == 3  # Valor por defecto
        assert config.max_text_length == 5000  # Valor por defecto

    def test_config_invalid_model_name_raises_error(self, tmp_path):
        """Debe lanzar error con model_name vacío."""
        with pytest.raises(ValueError, match="model_name debe ser una cadena no vacía"):
            ScriptConfig(
                model_name="",
                input_file=tmp_path / "test.json",
                output_dir=tmp_path / "output",
                text_column="text",
                id_column="id",
                max_batch_size=64,
                memory_limit_gb=2.0,
            )

    def test_config_invalid_batch_size_raises_error(self, tmp_path):
        """Debe lanzar error con batch_size no positivo."""
        with pytest.raises(ValueError, match="max_batch_size debe ser un entero positivo"):
            ScriptConfig(
                model_name="model",
                input_file=tmp_path / "test.json",
                output_dir=tmp_path / "output",
                text_column="text",
                id_column="id",
                max_batch_size=0,
                memory_limit_gb=2.0,
            )

    def test_config_invalid_text_length_range_raises_error(self, tmp_path):
        """Debe lanzar error si min_text_length >= max_text_length."""
        with pytest.raises(ValueError, match="min_text_length.*debe ser menor que"):
            ScriptConfig(
                model_name="model",
                input_file=tmp_path / "test.json",
                output_dir=tmp_path / "output",
                text_column="text",
                id_column="id",
                max_batch_size=64,
                memory_limit_gb=2.0,
                min_text_length=100,
                max_text_length=50,
            )

    def test_config_unsupported_file_format_raises_error(self, tmp_path):
        """Debe lanzar error con formato de archivo no soportado."""
        with pytest.raises(ValueError, match="Formato de archivo no soportado"):
            ScriptConfig(
                model_name="model",
                input_file=tmp_path / "test.txt",
                output_dir=tmp_path / "output",
                text_column="text",
                id_column="id",
                max_batch_size=64,
                memory_limit_gb=2.0,
            )

    def test_config_paths_are_resolved(self, tmp_path):
        """Las rutas deben resolverse a rutas absolutas."""
        config = ScriptConfig(
            model_name="model",
            input_file=tmp_path / "test.json",
            output_dir=tmp_path / "output",
            text_column="text",
            id_column="id",
            max_batch_size=64,
            memory_limit_gb=2.0,
        )
        assert config.input_file.is_absolute()
        assert config.output_dir.is_absolute()


# --- Pruebas de ValidationStats ---


class TestValidationStats:
    """Pruebas para la clase ValidationStats."""

    def test_success_rate_calculation(self):
        """Debe calcular correctamente la tasa de éxito."""
        stats = ValidationStats(
            exact_matches=80, semantic_duplicates=15, failures=5, total=100
        )
        assert stats.success_rate == 0.95

    def test_success_rate_zero_total(self):
        """Debe retornar 0 si total es 0."""
        stats = ValidationStats(exact_matches=0, semantic_duplicates=0, failures=0, total=0)
        assert stats.success_rate == 0.0

    def test_is_valid_no_failures(self):
        """Debe ser válido si no hay fallos."""
        stats = ValidationStats(
            exact_matches=90, semantic_duplicates=10, failures=0, total=100
        )
        assert stats.is_valid is True

    def test_is_valid_with_failures(self):
        """No debe ser válido si hay fallos."""
        stats = ValidationStats(
            exact_matches=90, semantic_duplicates=5, failures=5, total=100
        )
        assert stats.is_valid is False


# --- Pruebas de DataValidator ---


class TestDataValidator:
    """Pruebas para el validador de datos."""

    def test_validate_dataframe_success(self, sample_dataframe, mock_config):
        """La validación debe pasar con datos correctos."""
        validated_df = DataValidator.validate_dataframe(
            sample_dataframe, "description", "id", mock_config
        )
        assert len(validated_df) == 10
        assert "description" in validated_df.columns
        assert "id" in validated_df.columns

    def test_validate_dataframe_missing_column_raises_error(
        self, sample_dataframe, mock_config
    ):
        """Debe lanzar error si falta una columna requerida."""
        df = sample_dataframe.drop(columns=["description"])
        with pytest.raises(DataValidationError, match="Columnas faltantes"):
            DataValidator.validate_dataframe(df, "description", "id", mock_config)

    def test_validate_dataframe_removes_nulls(self, sample_dataframe, mock_config):
        """Debe eliminar filas con valores nulos en columnas clave."""
        sample_dataframe.loc[0, "description"] = None
        validated_df = DataValidator.validate_dataframe(
            sample_dataframe, "description", "id", mock_config
        )
        assert len(validated_df) == 9

    def test_validate_dataframe_removes_duplicates(self, mock_config):
        """Debe eliminar filas duplicadas por ID, conservando la primera."""
        df = pd.DataFrame(
            {
                "id": ["ID1", "ID1", "ID2"],
                "description": ["First", "Duplicate", "Third"],
            }
        )
        validated_df = DataValidator.validate_dataframe(df, "description", "id", mock_config)
        assert len(validated_df) == 2
        assert validated_df[validated_df["id"] == "ID1"]["description"].values[0] == "First"

    def test_validate_dataframe_filters_by_text_length(self, mock_config):
        """Debe filtrar textos fuera del rango de longitud."""
        df = pd.DataFrame(
            {
                "id": ["ID1", "ID2", "ID3"],
                "description": [
                    "ab",
                    "Valid text here",
                    "x" * 6000,
                ],  # muy corto, válido, muy largo
            }
        )
        # Ajustar config para prueba
        mock_config.min_text_length = 3
        mock_config.max_text_length = 5000

        validated_df = DataValidator.validate_dataframe(df, "description", "id", mock_config)
        assert len(validated_df) == 1
        assert validated_df["id"].values[0] == "ID2"

    def test_validate_dataframe_empty_raises_error(self, mock_config):
        """Debe lanzar error si el DataFrame está vacío."""
        df = pd.DataFrame(columns=["id", "description"])
        with pytest.raises(DataValidationError, match="DataFrame está vacío"):
            DataValidator.validate_dataframe(df, "description", "id", mock_config)

    def test_validate_dataframe_all_filtered_raises_error(self, mock_config):
        """Debe lanzar error si todos los datos son filtrados."""
        df = pd.DataFrame(
            {
                "id": [None, None],
                "description": ["test", "test2"],
            }
        )
        with pytest.raises(DataValidationError, match="No quedan datos válidos"):
            DataValidator.validate_dataframe(df, "description", "id", mock_config)

    def test_validate_dataframe_invalid_df_type_raises_error(self, mock_config):
        """Debe lanzar error si df no es un DataFrame."""
        with pytest.raises(TypeError, match="df debe ser un DataFrame"):
            DataValidator.validate_dataframe([], "description", "id", mock_config)

    def test_validate_dataframe_combined_cleaning(
        self, sample_dataframe_with_issues, mock_config
    ):
        """Debe manejar correctamente múltiples tipos de datos problemáticos."""
        # DataFrame tiene: 1 duplicado, 2 nulos, 1 texto muy corto
        validated_df = DataValidator.validate_dataframe(
            sample_dataframe_with_issues, "description", "id", mock_config
        )
        # Esperado: 10 - 2 nulos - 1 duplicado - 1 corto = 6
        assert len(validated_df) == 6


# --- Pruebas de FileManager ---


class TestFileManager:
    """Pruebas para el gestor de archivos."""

    def test_load_data_json_success(self, tmp_path, sample_dataframe):
        """Debe cargar correctamente un archivo JSON."""
        file_path = tmp_path / "test.json"
        sample_dataframe.to_json(file_path, orient="records")

        df = FileManager.load_data(file_path)
        assert len(df) == len(sample_dataframe)
        assert list(df.columns) == list(sample_dataframe.columns)

    def test_load_data_csv_success(self, tmp_path, sample_dataframe):
        """Debe cargar correctamente un archivo CSV."""
        file_path = tmp_path / "test.csv"
        sample_dataframe.to_csv(file_path, index=False)

        df = FileManager.load_data(file_path)
        assert len(df) == len(sample_dataframe)

    def test_load_data_file_not_found_raises_error(self, tmp_path):
        """Debe lanzar error si el archivo no existe."""
        with pytest.raises(FileNotFoundError, match="archivo no existe"):
            FileManager.load_data(tmp_path / "nonexistent.json")

    def test_load_data_unsupported_format_raises_error(self, tmp_path):
        """Debe lanzar error con formato no soportado."""
        file_path = tmp_path / "test.xml"
        file_path.touch()
        with pytest.raises(ValueError, match="Formato de archivo no soportado"):
            FileManager.load_data(file_path)

    def test_load_data_empty_file_raises_error(self, tmp_path):
        """Debe lanzar error si el archivo está vacío."""
        file_path = tmp_path / "empty.json"
        file_path.write_text("[]")
        with pytest.raises(DataValidationError, match="está vacío"):
            FileManager.load_data(file_path)

    def test_create_backup_success(self, tmp_path):
        """Debe crear un backup correctamente."""
        original_file = tmp_path / "original.json"
        original_file.write_text('{"test": "data"}')

        backup_path = FileManager.create_backup(original_file)

        assert backup_path is not None
        assert backup_path.exists()
        assert "_backup_" in backup_path.name
        assert backup_path.read_text() == '{"test": "data"}'

    def test_create_backup_nonexistent_file_returns_none(self, tmp_path):
        """Debe retornar None si el archivo no existe."""
        result = FileManager.create_backup(tmp_path / "nonexistent.json")
        assert result is None

    def test_create_backup_invalid_path_type_raises_error(self):
        """Debe lanzar error si file_path no es Path."""
        with pytest.raises(TypeError, match="debe ser una instancia de pathlib.Path"):
            FileManager.create_backup("/some/string/path")


# --- Pruebas de MemoryMonitor ---


class TestMemoryMonitor:
    """Pruebas para el monitor de memoria."""

    def test_check_memory_exceeds_limit_raises_error(self):
        """Debe lanzar error si el uso estimado excede el límite."""
        with pytest.raises(InsufficientMemoryError, match="límite configurado"):
            MemoryMonitor.check_memory_availability(10.0, 5.0)

    def test_check_memory_exceeds_available_raises_error(self):
        """Debe lanzar error si excede el 80% de memoria disponible."""
        with patch("scripts.generate_embeddings.psutil.virtual_memory") as mock_mem:
            mock_mem.return_value.available = 2 * (1024**3)  # 2 GB
            with pytest.raises(InsufficientMemoryError, match="umbral de seguridad"):
                MemoryMonitor.check_memory_availability(2.0, 10.0)

    def test_check_memory_success(self):
        """No debe lanzar error si hay suficiente memoria."""
        with patch("scripts.generate_embeddings.psutil.virtual_memory") as mock_mem:
            mock_mem.return_value.available = 16 * (1024**3)  # 16 GB
            # No debe lanzar excepción
            MemoryMonitor.check_memory_availability(1.0, 8.0)

    def test_check_memory_invalid_params_raises_error(self):
        """Debe lanzar error con parámetros inválidos."""
        with pytest.raises(ValueError, match="estimated_size_gb"):
            MemoryMonitor.check_memory_availability(-1.0, 5.0)

        with pytest.raises(ValueError, match="limit_gb"):
            MemoryMonitor.check_memory_availability(1.0, 0)


# --- Pruebas de EmbeddingGenerator ---


class TestEmbeddingGenerator:
    """Pruebas para el generador de embeddings."""

    def test_load_model_success(self, mock_config):
        """Debe cargar un modelo mockeado sin errores."""
        with patch(
            "scripts.generate_embeddings.SentenceTransformer"
        ) as MockSentenceTransformer:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = 384
            MockSentenceTransformer.return_value = mock_model

            generator = EmbeddingGenerator(mock_config)
            generator.load_model()

            assert generator.model is not None
            MockSentenceTransformer.assert_called_with(mock_config.model_name, device="cpu")

    def test_load_model_failure_raises_error(self, mock_config):
        """Debe lanzar ModelLoadError si falla la carga."""
        with patch(
            "scripts.generate_embeddings.SentenceTransformer",
            side_effect=Exception("mock error"),
        ):
            generator = EmbeddingGenerator(mock_config)
            with pytest.raises(ModelLoadError, match="Error al cargar modelo"):
                generator.load_model()

    def test_load_model_invalid_dimension_raises_error(self, mock_config):
        """Debe lanzar error si la dimensión del modelo es inválida."""
        with patch(
            "scripts.generate_embeddings.SentenceTransformer"
        ) as MockSentenceTransformer:
            mock_model = MagicMock()
            mock_model.get_sentence_embedding_dimension.return_value = None
            MockSentenceTransformer.return_value = mock_model

            generator = EmbeddingGenerator(mock_config)
            with pytest.raises(ModelLoadError, match="dimensión válida"):
                generator.load_model()

    def test_generate_embeddings_success(self, mock_embedding_generator):
        """Debe generar embeddings con el formato correcto."""
        texts = ["hello world"] * 10
        embeddings = mock_embedding_generator.generate_embeddings(texts)

        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (10, 384)
        assert embeddings.dtype == np.float32

    def test_generate_embeddings_without_model_raises_error(self, mock_config):
        """Debe lanzar error si el modelo no está cargado."""
        generator = EmbeddingGenerator(mock_config)
        with pytest.raises(RuntimeError, match="modelo no ha sido cargado"):
            generator.generate_embeddings(["test"])

    def test_generate_embeddings_empty_list_raises_error(self, mock_embedding_generator):
        """Debe lanzar error con lista vacía."""
        with pytest.raises(ValueError, match="lista de textos no puede estar vacía"):
            mock_embedding_generator.generate_embeddings([])

    def test_generate_embeddings_invalid_type_raises_error(self, mock_embedding_generator):
        """Debe lanzar error si texts no es una lista."""
        with pytest.raises(TypeError, match="texts debe ser una lista"):
            mock_embedding_generator.generate_embeddings("not a list")

    def test_generate_embeddings_non_string_elements_raises_error(
        self, mock_embedding_generator
    ):
        """Debe lanzar error si hay elementos no string."""
        with pytest.raises(TypeError, match="elementos de texts deben ser cadenas"):
            mock_embedding_generator.generate_embeddings(["valid", 123, "another"])

    def test_build_faiss_index_success(self, mock_embedding_generator):
        """Debe construir un índice FAISS correctamente."""
        embeddings = np.random.rand(10, 384).astype(np.float32)

        with patch("scripts.generate_embeddings.faiss") as mock_faiss:
            mock_index = MagicMock()
            mock_index.ntotal = 10
            mock_faiss.IndexFlatIP.return_value = mock_index

            index = mock_embedding_generator.build_faiss_index(embeddings)

            mock_faiss.IndexFlatIP.assert_called_with(384)
            mock_index.add.assert_called_once()
            assert index.ntotal == 10

    def test_build_faiss_index_invalid_array_raises_error(self, mock_embedding_generator):
        """Debe lanzar error con array inválido."""
        with pytest.raises(TypeError, match="debe ser un array numpy"):
            mock_embedding_generator.build_faiss_index([[1, 2, 3]])

        with pytest.raises(ValueError, match="debe ser 2D"):
            mock_embedding_generator.build_faiss_index(np.array([1, 2, 3]))

        with pytest.raises(ValueError, match="no puede estar vacío"):
            mock_embedding_generator.build_faiss_index(np.array([]).reshape(0, 384))

    def test_validate_index_success(self, mock_config):
        """Debe validar correctamente un índice consistente."""
        embeddings = np.random.rand(50, 128).astype(np.float32)

        # Crear índice real para prueba
        index = faiss.IndexFlatIP(128)
        index.add(embeddings)

        generator = EmbeddingGenerator(mock_config)
        result = generator.validate_index(index, embeddings)

        assert result is True

    def test_validate_index_dimension_mismatch_raises_error(self, mock_config):
        """Debe lanzar error si las dimensiones no coinciden."""
        embeddings = np.random.rand(10, 128).astype(np.float32)
        index = faiss.IndexFlatIP(256)  # Dimensión diferente

        generator = EmbeddingGenerator(mock_config)
        with pytest.raises(ValueError, match="Dimensión del índice.*no coincide"):
            generator.validate_index(index, embeddings)

    def test_validate_index_count_mismatch_raises_error(self, mock_config):
        """Debe lanzar error si el número de vectores no coincide."""
        embeddings = np.random.rand(10, 128).astype(np.float32)
        index = faiss.IndexFlatIP(128)
        index.add(np.random.rand(5, 128).astype(np.float32))  # Solo 5 vectores

        generator = EmbeddingGenerator(mock_config)
        with pytest.raises(ValueError, match="Vectores en índice.*no coinciden"):
            generator.validate_index(index, embeddings)


# --- Pruebas de EmbeddingPipeline ---


class TestEmbeddingPipeline:
    """Pruebas para el pipeline de embeddings."""

    def test_estimate_memory_usage(self, mock_config):
        """Debe estimar correctamente el uso de memoria."""
        pipeline = EmbeddingPipeline(mock_config)

        # 1000 samples * 384 dims * 4 bytes * 6 factor / 1024^3
        expected_gb = (1000 * 384 * 4 * 6.0) / (1024**3)
        result = pipeline.estimate_memory_usage(1000, 384)

        assert abs(result - expected_gb) < 0.001

    def test_estimate_memory_invalid_params_raises_error(self, mock_config):
        """Debe lanzar error con parámetros inválidos."""
        pipeline = EmbeddingPipeline(mock_config)

        with pytest.raises(ValueError, match="n_samples"):
            pipeline.estimate_memory_usage(-1, 384)

        with pytest.raises(ValueError, match="embedding_dim"):
            pipeline.estimate_memory_usage(100, 0)

    def test_create_id_map(self, mock_config, sample_dataframe):
        """Debe crear correctamente el mapeo de IDs."""
        pipeline = EmbeddingPipeline(mock_config)
        id_map = pipeline._create_id_map(sample_dataframe)

        assert len(id_map) == 10
        assert id_map["0"] == "ID0"
        assert id_map["9"] == "ID9"
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in id_map.items())


# --- Prueba de Integración del Pipeline ---


@patch("scripts.generate_embeddings.faiss.write_index")
@patch("scripts.generate_embeddings.faiss.IndexFlatIP")
@patch("scripts.generate_embeddings.SentenceTransformer")
def test_pipeline_run_integration(
    MockSentenceTransformer,
    MockIndexFlatIP,
    mock_write_index,
    mock_config,
    sample_dataframe,
    tmp_path,
):
    """
    Prueba el flujo completo del pipeline `run()`, mockeando las dependencias externas.
    """
    # --- Configuración de Mocks ---
    mock_model = MagicMock()
    mock_embeddings = np.random.rand(len(sample_dataframe), 384).astype(np.float32)
    mock_model.encode.return_value = mock_embeddings
    mock_model.get_sentence_embedding_dimension.return_value = 384
    MockSentenceTransformer.return_value = mock_model

    # Mock para FAISS
    mock_index = MagicMock(spec=faiss.Index)
    mock_index.d = 384
    mock_index.ntotal = len(sample_dataframe)

    # Simular búsqueda para validación (coincidencia exacta)
    def mock_search(query, k):
        # Encontrar el índice del embedding que coincide
        for i, emb in enumerate(mock_embeddings):
            if np.allclose(query[0], emb):
                return np.array([[1.0] + [0.0] * (k - 1)]), np.array([[i] + [-1] * (k - 1)])
        return np.array([[0.0] * k]), np.array([[-1] * k])

    mock_index.search.side_effect = mock_search
    MockIndexFlatIP.return_value = mock_index

    # Configurar write_index para crear el archivo temporal y luego renombrarlo
    def mock_write_index_fn(index, path):
        Path(path).touch()

    mock_write_index.side_effect = mock_write_index_fn

    with patch("scripts.generate_embeddings.psutil.virtual_memory") as mock_virtual_memory:
        mock_virtual_memory.return_value.available = 8 * (1024**3)

        # --- Preparación de Archivos de Entrada ---
        input_file = mock_config.input_file
        sample_dataframe.to_json(input_file, orient="records")

        # --- Ejecución del Pipeline ---
        pipeline = EmbeddingPipeline(mock_config)
        metrics = pipeline.run()

        # --- Verificaciones del Modelo ---
        MockSentenceTransformer.assert_called_with(mock_config.model_name, device="cpu")

        # --- Verificaciones del Índice ---
        MockIndexFlatIP.assert_called_with(384)
        mock_index.add.assert_called_once()

        # Verificar que write_index fue llamado con archivo temporal
        write_call_args = mock_write_index.call_args[0]
        assert write_call_args[0] == mock_index
        assert ".tmp" in write_call_args[1]

        # --- Verificación de Archivos de Salida ---
        output_dir = mock_config.output_dir
        assert (output_dir / "faiss.index").exists()
        assert (output_dir / "id_map.json").exists()
        assert (output_dir / "metadata.json").exists()

        # --- Verificar contenido de id_map.json ---
        with open(output_dir / "id_map.json", "r", encoding="utf-8") as f:
            id_map = json.load(f)
            assert len(id_map) == len(sample_dataframe)
            assert id_map["0"] == "ID0"
            assert id_map[str(len(sample_dataframe) - 1)] == f"ID{len(sample_dataframe) - 1}"

        # --- Verificar contenido de metadata.json ---
        with open(output_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
            assert metadata["model_name"] == "mock-model"
            assert metadata["vector_count"] == len(sample_dataframe)
            assert metadata["embedding_dimension"] == 384
            assert metadata["text_column"] == "description"
            assert metadata["id_column"] == "id"
            assert "creation_date" in metadata

        # --- Verificar métricas de retorno (nombres actualizados) ---
        assert metrics["embeddings_generated"] == len(sample_dataframe)
        assert metrics["valid_rows"] == len(sample_dataframe)
        assert metrics["initial_rows"] == len(sample_dataframe)
        assert "processing_time_seconds" in metrics
        assert "validation_rate_pct" in metrics
        assert metrics["validation_rate_pct"] == 100.0
        assert "estimated_memory_gb" in metrics
        assert metrics["embedding_dim"] == 384
        assert metrics["index_vectors"] == len(sample_dataframe)


@patch("scripts.generate_embeddings.faiss.write_index")
@patch("scripts.generate_embeddings.faiss.IndexFlatIP")
@patch("scripts.generate_embeddings.SentenceTransformer")
def test_pipeline_run_with_data_cleaning(
    MockSentenceTransformer,
    MockIndexFlatIP,
    mock_write_index,
    mock_config,
    sample_dataframe_with_issues,
    tmp_path,
):
    """
    Prueba el pipeline con datos que requieren limpieza.
    """
    # Configurar mocks
    mock_model = MagicMock()
    # El DataFrame limpio tendrá 6 filas
    expected_valid_rows = 6
    mock_embeddings = np.random.rand(expected_valid_rows, 384).astype(np.float32)
    mock_model.encode.return_value = mock_embeddings
    mock_model.get_sentence_embedding_dimension.return_value = 384
    MockSentenceTransformer.return_value = mock_model

    mock_index = MagicMock(spec=faiss.Index)
    mock_index.d = 384
    mock_index.ntotal = expected_valid_rows

    def mock_search(query, k):
        for i, emb in enumerate(mock_embeddings):
            if np.allclose(query[0], emb):
                return np.array([[1.0] + [0.0] * (k - 1)]), np.array([[i] + [-1] * (k - 1)])
        return np.array([[0.0] * k]), np.array([[-1] * k])

    mock_index.search.side_effect = mock_search
    MockIndexFlatIP.return_value = mock_index
    mock_write_index.side_effect = lambda idx, path: Path(path).touch()

    with patch("scripts.generate_embeddings.psutil.virtual_memory") as mock_mem:
        mock_mem.return_value.available = 8 * (1024**3)

        input_file = mock_config.input_file
        sample_dataframe_with_issues.to_json(input_file, orient="records")

        pipeline = EmbeddingPipeline(mock_config)
        metrics = pipeline.run()

        # Verificar que se procesaron las filas correctas
        assert metrics["initial_rows"] == 10
        assert metrics["valid_rows"] == expected_valid_rows
        assert metrics["validation_rate_pct"] == 60.0
        assert metrics["embeddings_generated"] == expected_valid_rows


@patch("scripts.generate_embeddings.faiss.write_index")
@patch("scripts.generate_embeddings.faiss.IndexFlatIP")
@patch("scripts.generate_embeddings.SentenceTransformer")
def test_pipeline_backup_creation(
    MockSentenceTransformer,
    MockIndexFlatIP,
    mock_write_index,
    mock_config,
    sample_dataframe,
):
    """
    Prueba que el pipeline crea backups cuando está habilitado.
    """
    # Configurar mocks
    mock_model = MagicMock()
    mock_embeddings = np.random.rand(len(sample_dataframe), 384).astype(np.float32)
    mock_model.encode.return_value = mock_embeddings
    mock_model.get_sentence_embedding_dimension.return_value = 384
    MockSentenceTransformer.return_value = mock_model

    mock_index = MagicMock(spec=faiss.Index)
    mock_index.d = 384
    mock_index.ntotal = len(sample_dataframe)

    def mock_search(query, k):
        for i, emb in enumerate(mock_embeddings):
            if np.allclose(query[0], emb):
                return np.array([[1.0] + [0.0] * (k - 1)]), np.array([[i] + [-1] * (k - 1)])
        return np.array([[0.0] * k]), np.array([[-1] * k])

    mock_index.search.side_effect = mock_search
    MockIndexFlatIP.return_value = mock_index
    mock_write_index.side_effect = lambda idx, path: Path(path).touch()

    with patch("scripts.generate_embeddings.psutil.virtual_memory") as mock_mem:
        mock_mem.return_value.available = 8 * (1024**3)

        # Crear archivos existentes para que se generen backups
        mock_config.output_dir.mkdir(parents=True, exist_ok=True)
        existing_index = mock_config.output_dir / "faiss.index"
        existing_id_map = mock_config.output_dir / "id_map.json"
        existing_index.write_text("old index")
        existing_id_map.write_text('{"old": "map"}')

        input_file = mock_config.input_file
        sample_dataframe.to_json(input_file, orient="records")

        # Asegurar que backup está habilitado
        mock_config.backup_enabled = True

        pipeline = EmbeddingPipeline(mock_config)
        pipeline.run()

        # Verificar que se crearon backups
        backup_files = list(mock_config.output_dir.glob("*_backup_*"))
        assert len(backup_files) >= 2  # Al menos index y id_map


@patch("scripts.generate_embeddings.faiss.write_index")
@patch("scripts.generate_embeddings.faiss.IndexFlatIP")
@patch("scripts.generate_embeddings.SentenceTransformer")
def test_pipeline_no_backup_when_disabled(
    MockSentenceTransformer,
    MockIndexFlatIP,
    mock_write_index,
    mock_config,
    sample_dataframe,
):
    """
    Prueba que no se crean backups cuando está deshabilitado.
    """
    # Configurar mocks
    mock_model = MagicMock()
    mock_embeddings = np.random.rand(len(sample_dataframe), 384).astype(np.float32)
    mock_model.encode.return_value = mock_embeddings
    mock_model.get_sentence_embedding_dimension.return_value = 384
    MockSentenceTransformer.return_value = mock_model

    mock_index = MagicMock(spec=faiss.Index)
    mock_index.d = 384
    mock_index.ntotal = len(sample_dataframe)

    def mock_search(query, k):
        for i, emb in enumerate(mock_embeddings):
            if np.allclose(query[0], emb):
                return np.array([[1.0] + [0.0] * (k - 1)]), np.array([[i] + [-1] * (k - 1)])
        return np.array([[0.0] * k]), np.array([[-1] * k])

    mock_index.search.side_effect = mock_search
    MockIndexFlatIP.return_value = mock_index
    mock_write_index.side_effect = lambda idx, path: Path(path).touch()

    with patch("scripts.generate_embeddings.psutil.virtual_memory") as mock_mem:
        mock_mem.return_value.available = 8 * (1024**3)

        # Crear archivos existentes
        mock_config.output_dir.mkdir(parents=True, exist_ok=True)
        existing_index = mock_config.output_dir / "faiss.index"
        existing_index.write_text("old index")

        input_file = mock_config.input_file
        sample_dataframe.to_json(input_file, orient="records")

        # Deshabilitar backups
        mock_config.backup_enabled = False

        pipeline = EmbeddingPipeline(mock_config)
        pipeline.run()

        # Verificar que NO se crearon backups
        backup_files = list(mock_config.output_dir.glob("*_backup_*"))
        assert len(backup_files) == 0
