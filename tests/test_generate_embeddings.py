import json

# Añadir el directorio raíz al path para permitir la importación del script
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
    ModelLoadError,
    ScriptConfig,
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


# --- Pruebas Unitarias ---


class TestDataValidator:
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
        """Debe lanzar un error si falta una columna requerida."""
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


class TestEmbeddingGenerator:
    def test_load_model_success(self, mock_config):
        """Debe cargar un modelo mockeado sin errores."""
        with patch(
            "scripts.generate_embeddings.SentenceTransformer"
        ) as MockSentenceTransformer:
            MockSentenceTransformer.return_value = MagicMock()
            generator = EmbeddingGenerator(mock_config)
            generator.load_model()
            assert generator.model is not None

    def test_load_model_failure_raises_error(self, mock_config):
        """Debe lanzar ModelLoadError si falla la carga."""
        with patch(
            "scripts.generate_embeddings.SentenceTransformer",
            side_effect=Exception("mock error"),
        ):
            generator = EmbeddingGenerator(mock_config)
            with pytest.raises(ModelLoadError):
                generator.load_model()

    def test_generate_embeddings(self, mock_embedding_generator):
        """Debe generar embeddings con el formato correcto."""
        texts = ["hello"] * 10
        embeddings = mock_embedding_generator.generate_embeddings(texts)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (10, 384)

    def test_build_faiss_index(self, mock_embedding_generator):
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
    # Mock para SentenceTransformer
    mock_model = MagicMock()
    mock_embeddings = np.random.rand(len(sample_dataframe), 384).astype(np.float32)
    mock_model.encode.return_value = mock_embeddings
    mock_model.get_sentence_embedding_dimension.return_value = 384
    MockSentenceTransformer.return_value = mock_model

    # Mock para FAISS
    mock_index = MagicMock(spec=faiss.Index)
    mock_index.d = 384
    mock_index.ntotal = len(sample_dataframe)
    # Simular la búsqueda para la validación
    mock_index.search.side_effect = lambda query, k: (
        np.array([[0.99]]),
        np.array([[np.where(np.all(mock_embeddings == query, axis=1))[0][0]]]),
    )
    MockIndexFlatIP.return_value = mock_index
    # Configurar el mock de write_index para que cree el archivo
    mock_write_index.side_effect = lambda index, path: Path(path).touch()

    # Mock para psutil (MemoryMonitor)
    with patch("scripts.generate_embeddings.psutil.virtual_memory") as mock_virtual_memory:
        mock_virtual_memory.return_value.available = 8 * (1024**3)  # Simular 8GB disponibles

        # --- Preparación de Archivos de Entrada ---
        input_file = mock_config.input_file
        sample_dataframe.to_json(input_file, orient="records")

        # --- Ejecución del Pipeline ---
        pipeline = EmbeddingPipeline(mock_config)
        metrics = pipeline.run()

        # --- Verificaciones ---
        # Verificar que el modelo fue cargado
        MockSentenceTransformer.assert_called_with(
            mock_config.model_name, device="cpu"
        )

        # Verificar que el índice fue construido y guardado
        MockIndexFlatIP.assert_called_with(384)
        mock_index.add.assert_called_once()
        mock_write_index.assert_called_with(
            mock_index, str(mock_config.output_dir / "faiss.index")
        )

        # Verificar que los archivos de salida fueron creados
        output_dir = mock_config.output_dir
        assert (output_dir / "faiss.index").exists()
        assert (output_dir / "id_map.json").exists()
        assert (output_dir / "metadata.json").exists()

        # Verificar contenido de id_map.json
        with open(output_dir / "id_map.json", "r") as f:
            id_map = json.load(f)
            assert len(id_map) == len(sample_dataframe)
            assert id_map["0"] == "ID0"
            assert id_map[str(len(sample_dataframe) - 1)] == f"ID{len(sample_dataframe) - 1}"

        # Verificar contenido de metadata.json
        with open(output_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
            assert metadata["model_name"] == "mock-model"
            assert metadata["vector_count"] == len(sample_dataframe)
            assert "creation_date" in metadata

        # Verificar métricas de retorno
        assert metrics["embeddings_generated"] == len(sample_dataframe)
        assert metrics["valid_rows"] == len(sample_dataframe)


# Simular que faiss.write_index crea un archivo vacío para la prueba
def mock_write_index(index, path):
    Path(path).touch()


@patch("scripts.generate_embeddings.faiss.write_index", mock_write_index)
def test_pipeline_with_mocked_write(mock_config, sample_dataframe, tmp_path):
    # Esta prueba es un ejemplo de cómo mockear una función específica si es necesario.
    # Aquí se reutiliza parte de la lógica de la prueba de integración.
    with (
        patch("scripts.generate_embeddings.SentenceTransformer") as MockSentenceTransformer,
        patch("scripts.generate_embeddings.faiss.IndexFlatIP") as MockIndex,
        patch("scripts.generate_embeddings.psutil.virtual_memory") as mock_virtual_memory,
    ):
        # Configurar mocks correctamente
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384

        # ¡CRUCIAL! Configurar encode para devolver un array numpy del tamaño correcto
        mock_embeddings = np.random.rand(len(sample_dataframe), 384).astype(
            np.float32
        )
        mock_model.encode.return_value = mock_embeddings

        MockSentenceTransformer.return_value = mock_model

        # Configurar el mock del índice para que devuelva un ntotal numérico
        mock_index_instance = MagicMock(spec=faiss.Index)
        mock_index_instance.d = 384
        mock_index_instance.ntotal = len(sample_dataframe)

        # Configurar search para evitar fallos en validación si se ejecuta
        # Devolvemos coincidencia exacta para que pase la validación
        mock_index_instance.search.side_effect = lambda query, k: (
            np.array([[1.0] + [0.0] * (k - 1)]),  # Similitudes
            np.array(
                [[0] + [-1] * (k - 1)]
            ),  # Índices (simulamos que siempre encuentra el 0 como top 1)
        )

        MockIndex.return_value = mock_index_instance

        mock_virtual_memory.return_value.available = 8 * (1024**3)

        input_file = mock_config.input_file
        sample_dataframe.to_json(input_file, orient="records")

        pipeline = EmbeddingPipeline(mock_config)
        pipeline.run()

        assert (mock_config.output_dir / "faiss.index").exists()
