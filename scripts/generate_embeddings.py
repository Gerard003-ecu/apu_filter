import argparse
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import faiss
import numpy as np
import pandas as pd
import psutil
from sentence_transformers import SentenceTransformer

# Configuración del path para importar desde la app
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# --- Configuración de Logging Mejorada ---
class ColoredFormatter(logging.Formatter):
    """Formatter personalizado con colores para mejor legibilidad"""

    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Verde
        "WARNING": "\033[33m",  # Amarillo
        "ERROR": "\033[31m",  # Rojo
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """
    Configura el sistema de logging con formato mejorado.
    Args:
        log_level: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path opcional para guardar logs en archivo
    """
    # Validar nivel de log
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level.upper() not in valid_levels:
        raise ValueError(
            f"Nivel de log inválido: {log_level}. Debe ser uno de: {valid_levels}"
        )

    # Validar archivo de log
    if log_file is not None and not isinstance(log_file, Path):
        raise TypeError("log_file debe ser una instancia de pathlib.Path o None")

    handlers = []

    # Console handler con colores
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)

    # File handler si se especifica
    if log_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)
        except Exception as e:
            logging.error(f"Error al configurar el handler de archivo de log: {e}")
            raise

    # Limpiar handlers previos para evitar duplicados
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    logging.basicConfig(
        level=getattr(logging, log_level.upper()), handlers=handlers, force=True
    )


# --- Configuración y Constantes ---
@dataclass
class ScriptConfig:
    """Configuración del script, combinando JSON y argumentos de línea de comandos."""

    model_name: str
    input_file: Path
    output_dir: Path
    text_column: str
    id_column: str
    max_batch_size: int
    memory_limit_gb: float
    backup_enabled: bool = True
    normalize_embeddings: bool = True
    show_progress: bool = True
    # Parámetros que no se exponen en la línea de comandos
    min_text_length: int = field(default=3, repr=False)
    max_text_length: int = field(default=5000, repr=False)
    validation_sample_size: int = field(default=100, repr=False)

    def __post_init__(self):
        """Validaciones posteriores a la inicialización."""
        self._validate_string_fields()
        self._validate_numeric_fields()
        self._validate_boolean_fields()
        self._validate_paths()
        self._validate_text_length_range()

    def _validate_string_fields(self) -> None:
        """Valida campos de tipo string."""
        string_fields = [
            ("model_name", self.model_name),
            ("text_column", self.text_column),
            ("id_column", self.id_column),
        ]
        for field_name, value in string_fields:
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} debe ser una cadena no vacía")

    def _validate_numeric_fields(self) -> None:
        """Valida campos numéricos."""
        if not isinstance(self.max_batch_size, int) or self.max_batch_size <= 0:
            raise ValueError("max_batch_size debe ser un entero positivo")

        if not isinstance(self.memory_limit_gb, (int, float)) or self.memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb debe ser un número positivo")

        if not isinstance(self.min_text_length, int) or self.min_text_length < 0:
            raise ValueError("min_text_length debe ser un entero no negativo")

        if not isinstance(self.max_text_length, int) or self.max_text_length <= 0:
            raise ValueError("max_text_length debe ser un entero positivo")

        if (
            not isinstance(self.validation_sample_size, int)
            or self.validation_sample_size <= 0
        ):
            raise ValueError("validation_sample_size debe ser un entero positivo")

    def _validate_boolean_fields(self) -> None:
        """Valida campos booleanos."""
        boolean_fields = [
            ("backup_enabled", self.backup_enabled),
            ("normalize_embeddings", self.normalize_embeddings),
            ("show_progress", self.show_progress),
        ]
        for field_name, value in boolean_fields:
            if not isinstance(value, bool):
                raise TypeError(f"{field_name} debe ser booleano")

    def _validate_paths(self) -> None:
        """Valida y normaliza rutas."""
        if not isinstance(self.input_file, Path):
            raise TypeError("input_file debe ser una instancia de pathlib.Path")

        if not isinstance(self.output_dir, Path):
            raise TypeError("output_dir debe ser una instancia de pathlib.Path")

        supported_formats = [".csv", ".json"]
        if self.input_file.suffix.lower() not in supported_formats:
            raise ValueError(
                f"Formato de archivo no soportado: {self.input_file.suffix}. "
                f"Formatos válidos: {supported_formats}"
            )

        self.input_file = self.input_file.resolve()
        self.output_dir = self.output_dir.resolve()

    def _validate_text_length_range(self) -> None:
        """Valida coherencia entre longitudes mínima y máxima de texto."""
        if self.min_text_length >= self.max_text_length:
            raise ValueError(
                f"min_text_length ({self.min_text_length}) debe ser menor que "
                f"max_text_length ({self.max_text_length})"
            )


# --- Excepciones Personalizadas ---
class EmbeddingGenerationError(Exception):
    """Excepción base para errores en la generación de embeddings"""

    pass


class DataValidationError(EmbeddingGenerationError):
    """Error en la validación de datos de entrada"""

    pass


class ModelLoadError(EmbeddingGenerationError):
    """Error al cargar el modelo de embeddings"""

    pass


class InsufficientMemoryError(EmbeddingGenerationError):
    """Error por memoria insuficiente"""

    pass


class ConfigurationError(EmbeddingGenerationError):
    """Error en la configuración del script"""

    pass


# --- Utilidades ---
class DataValidator:
    """Validador de datos de entrada"""

    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame, text_column: str, id_column: str, config: ScriptConfig
    ) -> pd.DataFrame:
        """
        Valida y limpia el DataFrame de entrada.

        Args:
            df: DataFrame a validar
            text_column: Nombre de la columna de texto
            id_column: Nombre de la columna de ID
            config: Configuración del script

        Returns:
            DataFrame validado y limpio

        Raises:
            DataValidationError: Si hay problemas con los datos
            TypeError: Si df no es un DataFrame
        """
        DataValidator._validate_inputs(df, text_column, id_column, config)

        logger = logging.getLogger(__name__)
        initial_rows = len(df)

        if initial_rows == 0:
            raise DataValidationError("El DataFrame está vacío")

        # Verificar columnas requeridas
        DataValidator._check_required_columns(df, text_column, id_column)

        # Pipeline de limpieza con tracking de métricas
        df_clean, removed_nulls = DataValidator._remove_null_rows(
            df, text_column, id_column, logger
        )

        rows_after_nulls = len(df_clean)
        df_clean, removed_duplicates = DataValidator._remove_duplicate_rows(
            df_clean, id_column, logger
        )

        rows_after_duplicates = len(df_clean)
        df_clean[text_column] = df_clean[text_column].astype(str)

        df_filtered, removed_by_length = DataValidator._filter_by_text_length(
            df_clean, text_column, config, logger
        )

        # Validar que quedan datos
        if df_filtered.empty:
            raise DataValidationError(
                f"No quedan datos válidos después de aplicar los filtros. "
                f"Original: {initial_rows}, nulos: {removed_nulls}, "
                f"duplicados: {removed_duplicates}, longitud: {removed_by_length}"
            )

        valid_ratio = len(df_filtered) / initial_rows
        logger.info(
            f"Validación completada: {len(df_filtered)}/{initial_rows} filas válidas "
            f"({valid_ratio * 100:.2f}%)"
        )
        return df_filtered

    @staticmethod
    def _validate_inputs(
        df: pd.DataFrame, text_column: str, id_column: str, config: ScriptConfig
    ) -> None:
        """Valida los parámetros de entrada."""
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df debe ser un DataFrame de pandas")

        if not isinstance(text_column, str) or not text_column.strip():
            raise ValueError("text_column debe ser una cadena no vacía")

        if not isinstance(id_column, str) or not id_column.strip():
            raise ValueError("id_column debe ser una cadena no vacía")

        if not isinstance(config, ScriptConfig):
            raise TypeError("config debe ser una instancia de ScriptConfig")

    @staticmethod
    def _check_required_columns(df: pd.DataFrame, text_column: str, id_column: str) -> None:
        """Verifica que las columnas requeridas existan."""
        missing_columns = [col for col in [text_column, id_column] if col not in df.columns]
        if missing_columns:
            raise DataValidationError(
                f"Columnas faltantes en el DataFrame: {missing_columns}"
            )

    @staticmethod
    def _remove_null_rows(
        df: pd.DataFrame, text_column: str, id_column: str, logger: logging.Logger
    ) -> tuple[pd.DataFrame, int]:
        """Elimina filas con valores nulos en columnas críticas."""
        initial_count = len(df)
        df_clean = df.dropna(subset=[text_column, id_column]).copy()
        removed_count = initial_count - len(df_clean)

        if removed_count > 0:
            logger.warning(
                f"Se eliminaron {removed_count} filas con valores nulos en "
                f"'{text_column}' o '{id_column}'"
            )
        return df_clean, removed_count

    @staticmethod
    def _remove_duplicate_rows(
        df: pd.DataFrame, id_column: str, logger: logging.Logger
    ) -> tuple[pd.DataFrame, int]:
        """Elimina filas duplicadas basadas en la columna de ID."""
        initial_count = len(df)
        df_clean = df.drop_duplicates(subset=[id_column], keep="first").reset_index(
            drop=True
        )
        removed_count = initial_count - len(df_clean)

        if removed_count > 0:
            logger.warning(
                f"Se eliminaron {removed_count} filas duplicadas según '{id_column}'"
            )
        return df_clean, removed_count

    @staticmethod
    def _filter_by_text_length(
        df: pd.DataFrame, text_column: str, config: ScriptConfig, logger: logging.Logger
    ) -> tuple[pd.DataFrame, int]:
        """Filtra filas por longitud de texto."""
        initial_count = len(df)
        text_lengths = df[text_column].str.len()

        mask = text_lengths.between(config.min_text_length, config.max_text_length)
        df_filtered = df[mask].reset_index(drop=True)
        removed_count = initial_count - len(df_filtered)

        if removed_count > 0:
            logger.warning(
                f"Se eliminaron {removed_count} filas fuera del rango de longitud "
                f"[{config.min_text_length}, {config.max_text_length}]"
            )
        return df_filtered, removed_count


class MemoryMonitor:
    """Monitor de memoria del sistema"""

    @staticmethod
    def check_memory_availability(estimated_size_gb: float, limit_gb: float) -> None:
        """
        Verifica si hay suficiente memoria disponible.

        Args:
            estimated_size_gb: Tamaño estimado de uso en GB
            limit_gb: Límite de memoria configurado en GB

        Raises:
            InsufficientMemoryError: Si no hay suficiente memoria
            ValueError: Si los parámetros son inválidos
        """
        if not isinstance(estimated_size_gb, (int, float)) or estimated_size_gb < 0:
            raise ValueError("estimated_size_gb debe ser un número no negativo")

        if not isinstance(limit_gb, (int, float)) or limit_gb <= 0:
            raise ValueError("limit_gb debe ser un número positivo")

        available_gb = psutil.virtual_memory().available / (1024**3)

        if estimated_size_gb > limit_gb:
            raise InsufficientMemoryError(
                f"La operación requiere ~{estimated_size_gb:.2f}GB, "
                f"pero el límite configurado es {limit_gb:.2f}GB"
            )

        if estimated_size_gb > available_gb * 0.8:  # Usar 80% como umbral de seguridad
            raise InsufficientMemoryError(
                f"La operación requiere ~{estimated_size_gb:.2f}GB, "
                f"pero solo hay {available_gb:.2f}GB disponibles "
                f"(usando 80% como umbral de seguridad)"
            )


class FileManager:
    """Gestor de archivos con backup y validación"""

    SUPPORTED_ENCODINGS = ["utf-8", "latin-1", "cp1252"]

    @staticmethod
    def create_backup(file_path: Path) -> Optional[Path]:
        """Crea un backup del archivo si existe."""
        if not isinstance(file_path, Path):
            raise TypeError("file_path debe ser una instancia de pathlib.Path")

        logger = logging.getLogger(__name__)

        if not file_path.exists():
            logger.debug(f"No se puede crear backup, archivo no existe: {file_path}")
            return None

        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = file_path.with_name(
                f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
            )
            shutil.copy2(file_path, backup_path)
            logger.info(f"Backup creado: {backup_path}")
            return backup_path
        except (OSError, IOError) as e:
            logger.error(f"Error al crear backup de {file_path}: {e}")
            raise

    @staticmethod
    def load_data(file_path: Path) -> pd.DataFrame:
        """
        Carga datos desde archivo JSON o CSV con detección automática de codificación.

        Args:
            file_path: Ruta del archivo a cargar

        Returns:
            DataFrame con los datos cargados
        """
        if not isinstance(file_path, Path):
            raise TypeError("file_path debe ser una instancia de pathlib.Path")

        if not file_path.exists():
            raise FileNotFoundError(f"El archivo no existe: {file_path}")

        if not file_path.is_file():
            raise FileNotFoundError(f"La ruta no es un archivo: {file_path}")

        logger = logging.getLogger(__name__)
        suffix = file_path.suffix.lower()

        if suffix == ".json":
            df = FileManager._load_json(file_path, logger)
        elif suffix == ".csv":
            df = FileManager._load_csv(file_path, logger)
        else:
            raise ValueError(f"Formato de archivo no soportado: {suffix}")

        if df.empty:
            raise DataValidationError(f"El archivo {file_path} está vacío")

        logger.info(
            f"Datos cargados: {len(df)} filas, {len(df.columns)} columnas desde {file_path.name}"
        )
        return df

    @staticmethod
    def _load_json(file_path: Path, logger: logging.Logger) -> pd.DataFrame:
        """Carga un archivo JSON."""
        try:
            return pd.read_json(file_path, encoding="utf-8")
        except (ValueError, UnicodeDecodeError) as e:
            raise DataValidationError(f"Error al leer JSON {file_path}: {e}")
        except pd.errors.EmptyDataError:
            raise DataValidationError(f"El archivo JSON {file_path} está vacío o corrupto")

    @staticmethod
    def _load_csv(file_path: Path, logger: logging.Logger) -> pd.DataFrame:
        """Carga un archivo CSV con detección automática de codificación."""
        last_error = None

        for encoding in FileManager.SUPPORTED_ENCODINGS:
            try:
                df = pd.read_csv(file_path, encoding=encoding, sep=";")
                if encoding != "utf-8":
                    logger.warning(
                        f"Archivo {file_path.name} leído con codificación {encoding}"
                    )
                return df
            except UnicodeDecodeError as e:
                last_error = e
                continue
            except pd.errors.EmptyDataError:
                raise DataValidationError(
                    f"El archivo CSV {file_path} está vacío o corrupto"
                )

        raise DataValidationError(
            f"No se pudo leer {file_path} con codificaciones: "
            f"{FileManager.SUPPORTED_ENCODINGS}. Último error: {last_error}"
        )


@dataclass
class ValidationStats:
    """Estadísticas de validación del índice FAISS."""

    exact_matches: int = 0
    semantic_duplicates: int = 0
    failures: int = 0
    total: int = 0

    @property
    def success_rate(self) -> float:
        """Calcula la tasa de éxito."""
        if self.total == 0:
            return 0.0
        return (self.exact_matches + self.semantic_duplicates) / self.total

    @property
    def is_valid(self) -> bool:
        """Determina si la validación es exitosa (0 fallos reales)."""
        return self.failures == 0


class EmbeddingGenerator:
    """Generador de embeddings con manejo robusto"""

    SIMILARITY_THRESHOLD = 0.999  # Umbral para considerar vectores idénticos
    TOP_K_SEARCH = 5  # Número de vecinos a buscar en validación

    def __init__(self, config: ScriptConfig):
        if not isinstance(config, ScriptConfig):
            raise TypeError("config debe ser una instancia de ScriptConfig")

        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model: Optional[SentenceTransformer] = None

    def load_model(self) -> None:
        """Carga el modelo de sentence-transformers."""
        try:
            self.logger.info(f"Cargando modelo: {self.config.model_name}")
            self.model = SentenceTransformer(self.config.model_name, device="cpu")
            embedding_dim = self.model.get_sentence_embedding_dimension()

            if embedding_dim is None or embedding_dim <= 0:
                raise ModelLoadError(
                    f"El modelo {self.config.model_name} no devolvió una dimensión válida"
                )
            self.logger.info(f"Modelo cargado. Dimensión: {embedding_dim}")
        except ModelLoadError:
            raise
        except Exception as e:
            raise ModelLoadError(f"Error al cargar modelo '{self.config.model_name}': {e}")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Genera embeddings en batches con validación mejorada."""
        self._validate_texts_input(texts)

        if not self.model:
            raise RuntimeError("El modelo no ha sido cargado. Llame a load_model() primero")

        # Filtrar y reportar textos vacíos
        valid_texts, empty_count = self._filter_empty_texts(texts)
        if empty_count > 0:
            self.logger.warning(
                f"Se encontraron {empty_count} textos vacíos o con solo espacios"
            )

        if not valid_texts:
            raise ValueError("No hay textos válidos para generar embeddings")

        self.logger.info(f"Generando embeddings para {len(valid_texts)} textos...")

        try:
            embeddings = self.model.encode(
                valid_texts,
                batch_size=self.config.max_batch_size,
                show_progress_bar=self.config.show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_tensor=False,
            )

            embeddings = np.asarray(embeddings, dtype=np.float32)

            if embeddings.size == 0:
                raise RuntimeError("Los embeddings generados están vacíos")

            self.logger.info(f"Embeddings generados: shape={embeddings.shape}")
            return embeddings

        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"Error al generar embeddings: {e}")

    def _validate_texts_input(self, texts: List[str]) -> None:
        """Valida el parámetro de entrada de textos."""
        if not isinstance(texts, list):
            raise TypeError("texts debe ser una lista de cadenas")

        if not texts:
            raise ValueError("La lista de textos no puede estar vacía")

        if not all(isinstance(t, str) for t in texts):
            raise TypeError("Todos los elementos de texts deben ser cadenas")

    def _filter_empty_texts(self, texts: List[str]) -> tuple[List[str], int]:
        """Filtra textos vacíos y retorna los válidos con conteo de eliminados."""
        valid_texts = [t for t in texts if t.strip()]
        empty_count = len(texts) - len(valid_texts)
        return valid_texts, empty_count

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Construye un índice FAISS."""
        self._validate_embeddings_array(embeddings)

        self.logger.info("Construyendo índice FAISS...")

        embedding_dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)

        embeddings_float32 = np.ascontiguousarray(embeddings.astype(np.float32))
        index.add(embeddings_float32)

        if index.ntotal != len(embeddings):
            raise RuntimeError(
                f"Error al añadir embeddings: esperados {len(embeddings)}, "
                f"añadidos {index.ntotal}"
            )

        self.logger.info(
            f"Índice construido. Vectores: {index.ntotal}, Dimensión: {embedding_dim}"
        )
        return index

    def _validate_embeddings_array(self, embeddings: np.ndarray) -> None:
        """Valida el array de embeddings."""
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("embeddings debe ser un array numpy")

        if embeddings.ndim != 2:
            raise ValueError(f"embeddings debe ser 2D, actual: {embeddings.ndim}D")

        if embeddings.shape[0] == 0:
            raise ValueError("embeddings no puede estar vacío")

    def validate_index(self, index: faiss.Index, embeddings: np.ndarray) -> bool:
        """
        Valida el índice FAISS con tolerancia a duplicados.

        Args:
            index: Índice FAISS a validar.
            embeddings: Embeddings originales.

        Returns:
            bool: True si la validación es exitosa.
        """
        self._validate_index_inputs(index, embeddings)

        self.logger.info("=" * 80)
        self.logger.info("Iniciando validación del índice FAISS...")

        n_samples = min(self.config.validation_sample_size, len(embeddings))
        if n_samples == 0:
            self.logger.warning("No hay suficientes embeddings para validación")
            return True

        sample_indices = self._get_validation_sample_indices(len(embeddings), n_samples)
        embeddings_float32 = np.ascontiguousarray(embeddings.astype(np.float32))

        stats = self._run_validation_loop(index, embeddings_float32, sample_indices)
        self._log_validation_results(stats)

        return stats.is_valid

    def _validate_index_inputs(self, index: faiss.Index, embeddings: np.ndarray) -> None:
        """Valida las entradas del método validate_index."""
        if not isinstance(index, faiss.Index):
            raise TypeError("index debe ser una instancia de faiss.Index")

        self._validate_embeddings_array(embeddings)

        if index.d != embeddings.shape[1]:
            raise ValueError(
                f"Dimensión del índice ({index.d}) no coincide con "
                f"embeddings ({embeddings.shape[1]})"
            )

        if index.ntotal != len(embeddings):
            raise ValueError(
                f"Vectores en índice ({index.ntotal}) no coinciden con "
                f"embeddings ({len(embeddings)})"
            )

    def _get_validation_sample_indices(self, total: int, n_samples: int) -> np.ndarray:
        """Obtiene índices de muestra para validación con semilla fija."""
        rng = np.random.default_rng(42)
        return rng.choice(total, n_samples, replace=False)

    def _run_validation_loop(
        self, index: faiss.Index, embeddings: np.ndarray, sample_indices: np.ndarray
    ) -> ValidationStats:
        """Ejecuta el loop principal de validación."""
        stats = ValidationStats(total=len(sample_indices))
        k = min(self.TOP_K_SEARCH, index.ntotal)

        for idx in sample_indices:
            query = embeddings[idx : idx + 1]

            try:
                similarities, indices = index.search(query, k=k)
            except Exception as e:
                self.logger.error(f"Error en búsqueda para índice {idx}: {e}")
                stats.failures += 1
                continue

            classification = self._classify_search_result(idx, indices[0], similarities[0])
            self._update_stats(stats, classification, idx, indices[0], similarities[0])

        return stats

    def _classify_search_result(
        self, expected_idx: int, result_indices: np.ndarray, result_similarities: np.ndarray
    ) -> str:
        """Clasifica el resultado de búsqueda."""
        top_idx = result_indices[0]
        top_similarity = result_similarities[0]

        # Coincidencia exacta
        if top_idx == expected_idx:
            return "exact_match"

        # Índice correcto en top-k con alta similitud
        if expected_idx in result_indices:
            position = np.where(result_indices == expected_idx)[0][0]
            if result_similarities[position] > self.SIMILARITY_THRESHOLD:
                return "semantic_duplicate"

        # Similitud casi perfecta (duplicado semántico)
        if top_similarity > self.SIMILARITY_THRESHOLD:
            return "semantic_duplicate"

        # Fallo real
        return "failure"

    def _update_stats(
        self,
        stats: ValidationStats,
        classification: str,
        expected_idx: int,
        result_indices: np.ndarray,
        result_similarities: np.ndarray,
    ) -> None:
        """Actualiza estadísticas según clasificación."""
        if classification == "exact_match":
            stats.exact_matches += 1
        elif classification == "semantic_duplicate":
            stats.semantic_duplicates += 1
            self.logger.debug(
                f"⚠️ Duplicado detectado: esperado={expected_idx}, "
                f"top={result_indices[0]}, similitud={result_similarities[0]:.6f}"
            )
        else:
            stats.failures += 1
            self.logger.error(
                f"❌ Fallo de validación: esperado={expected_idx}, "
                f"top={result_indices[0]}, similitud={result_similarities[0]:.6f}, "
                f"top-5={result_indices.tolist()}"
            )

    def _log_validation_results(self, stats: ValidationStats) -> None:
        """Registra los resultados de validación."""
        self.logger.info("=" * 80)
        self.logger.info("📊 Resultados de Validación:")
        self.logger.info(f"   Total muestras: {stats.total}")

        if stats.total > 0:
            exact_pct = stats.exact_matches / stats.total * 100
            dup_pct = stats.semantic_duplicates / stats.total * 100
            fail_pct = stats.failures / stats.total * 100

            self.logger.info(
                f"   ✅ Coincidencias exactas: {stats.exact_matches} ({exact_pct:.1f}%)"
            )
            self.logger.info(
                f"   ⚠️  Duplicados semánticos: {stats.semantic_duplicates} ({dup_pct:.1f}%)"
            )
            self.logger.info(f"   ❌ Fallos reales: {stats.failures} ({fail_pct:.1f}%)")

        if stats.is_valid:
            if stats.semantic_duplicates > 0:
                self.logger.warning(
                    f"⚠️ Se detectaron {stats.semantic_duplicates} duplicados. "
                    "Considera limpiar datos duplicados."
                )
            self.logger.info(
                f"✅ Validación EXITOSA (tasa: {stats.success_rate * 100:.2f}%)"
            )
        else:
            self.logger.error(f"❌ Validación FALLIDA: {stats.failures} errores detectados")

        self.logger.info("=" * 80)


class EmbeddingPipeline:
    """Pipeline principal para la generación de embeddings"""

    MEMORY_OVERHEAD_FACTOR = 6.0  # Factor para operaciones temporales

    def __init__(self, config: ScriptConfig):
        if not isinstance(config, ScriptConfig):
            raise TypeError("config debe ser una instancia de ScriptConfig")

        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.generator = EmbeddingGenerator(config)

    def estimate_memory_usage(self, n_samples: int, embedding_dim: int) -> float:
        """
        Estima el uso de memoria en GB.

        El factor de overhead (6.0x) contempla:
        - Embeddings originales (1x)
        - Copia float32 para FAISS (1x)
        - Índice FAISS (1x)
        - Buffers temporales del modelo (2x)
        - Overhead de Python/NumPy (1x)
        """
        if not isinstance(n_samples, int) or n_samples < 0:
            raise ValueError("n_samples debe ser un entero no negativo")

        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError("embedding_dim debe ser un entero positivo")

        bytes_per_embedding = embedding_dim * 4  # float32 = 4 bytes
        total_bytes = n_samples * bytes_per_embedding * self.MEMORY_OVERHEAD_FACTOR
        return total_bytes / (1024**3)

    def run(self) -> Dict[str, Union[str, int, float]]:
        """
        Ejecuta el pipeline completo.

        Returns:
            Diccionario con métricas del proceso
        """
        start_time = time.time()
        self._log_header("INICIANDO GENERACIÓN DE EMBEDDINGS")

        # Cargar datos (guardando conteo inicial antes de validación)
        df_raw = FileManager.load_data(self.config.input_file)
        initial_rows = len(df_raw)

        # Validar datos
        df = DataValidator.validate_dataframe(
            df_raw, self.config.text_column, self.config.id_column, self.config
        )
        del df_raw  # Liberar memoria

        # Cargar modelo y obtener dimensión
        self.generator.load_model()
        embedding_dim = self.generator.model.get_sentence_embedding_dimension()

        # Verificar memoria disponible
        estimated_memory = self.estimate_memory_usage(len(df), embedding_dim)
        MemoryMonitor.check_memory_availability(
            estimated_memory, self.config.memory_limit_gb
        )

        # Generar embeddings
        texts = df[self.config.text_column].tolist()
        embeddings = self.generator.generate_embeddings(texts)

        # Construir y validar índice
        index = self.generator.build_faiss_index(embeddings)

        if not self.generator.validate_index(index, embeddings):
            raise EmbeddingGenerationError("La validación del índice FAISS falló")

        # Crear mapeo de IDs
        id_map = self._create_id_map(df)

        # Guardar artefactos
        self.save_artifacts(index, id_map)

        # Métricas finales
        metrics = {
            "processing_time_seconds": round(time.time() - start_time, 2),
            "initial_rows": initial_rows,
            "valid_rows": len(df),
            "validation_rate_pct": round(len(df) / initial_rows * 100, 2),
            "embeddings_generated": len(embeddings),
            "embedding_dim": embedding_dim,
            "index_vectors": index.ntotal,
            "id_map_size": len(id_map),
            "estimated_memory_gb": round(estimated_memory, 3),
        }

        self._log_header("✅ PROCESO COMPLETADO EXITOSAMENTE")
        return metrics

    def _create_id_map(self, df: pd.DataFrame) -> Dict[str, str]:
        """Crea el mapeo de índices a IDs."""
        return {
            str(i): str(apu_id)
            for i, apu_id in enumerate(df[self.config.id_column])
            if pd.notna(apu_id) and str(apu_id).strip()
        }

    def _log_header(self, message: str) -> None:
        """Registra un mensaje con formato de encabezado."""
        self.logger.info("=" * 80)
        self.logger.info(message)
        self.logger.info("=" * 80)

    def save_artifacts(self, index: faiss.Index, id_map: Dict[str, str]) -> None:
        """
        Guarda los artefactos con manejo transaccional básico.

        Usa archivos temporales para garantizar atomicidad.
        """
        self._validate_artifacts(index, id_map)

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Definir rutas
        paths = {
            "index": self.config.output_dir / "faiss.index",
            "id_map": self.config.output_dir / "id_map.json",
            "metadata": self.config.output_dir / "metadata.json",
        }

        temp_paths = {
            key: path.with_suffix(f"{path.suffix}.tmp") for key, path in paths.items()
        }

        try:
            # Crear backups si está habilitado
            if self.config.backup_enabled:
                for path in paths.values():
                    FileManager.create_backup(path)

            # Escribir a archivos temporales primero
            self._write_temp_files(index, id_map, temp_paths)

            # Mover archivos temporales a destinos finales (operación atómica en la mayoría de sistemas)
            for key in paths:
                if temp_paths[key].exists():
                    temp_paths[key].replace(paths[key])
                    self.logger.info(f"Guardado: {paths[key]}")

        except Exception as e:
            # Limpiar archivos temporales en caso de error
            for temp_path in temp_paths.values():
                if temp_path.exists():
                    temp_path.unlink()
            raise RuntimeError(f"Error al guardar artefactos: {e}")

    def _validate_artifacts(self, index: faiss.Index, id_map: Dict[str, str]) -> None:
        """Valida los artefactos antes de guardar."""
        if not isinstance(index, faiss.Index):
            raise TypeError("index debe ser una instancia de faiss.Index")

        if not isinstance(id_map, dict):
            raise TypeError("id_map debe ser un diccionario")

        if not all(isinstance(k, str) and isinstance(v, str) for k, v in id_map.items()):
            raise TypeError("id_map debe contener solo cadenas como claves y valores")

    def _write_temp_files(
        self, index: faiss.Index, id_map: Dict[str, str], temp_paths: Dict[str, Path]
    ) -> None:
        """Escribe archivos temporales."""
        # Guardar índice FAISS
        faiss.write_index(index, str(temp_paths["index"]))

        # Guardar mapeo de IDs
        with open(temp_paths["id_map"], "w", encoding="utf-8") as f:
            json.dump(id_map, f, indent=2, ensure_ascii=False)

        # Guardar metadatos
        metadata = {
            "model_name": self.config.model_name,
            "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "vector_count": index.ntotal,
            "embedding_dimension": index.d,
            "text_column": self.config.text_column,
            "id_column": self.config.id_column,
            "max_batch_size": self.config.max_batch_size,
            "normalize_embeddings": self.config.normalize_embeddings,
            "input_file": str(self.config.input_file),
        }

        with open(temp_paths["metadata"], "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_config_from_json(config_path: Path) -> Dict[str, Any]:
    """
    Carga la configuración desde un archivo JSON.

    Args:
        config_path: Ruta del archivo de configuración JSON

    Returns:
        Diccionario con la configuración
    """
    if not isinstance(config_path, Path):
        raise TypeError("config_path debe ser una instancia de pathlib.Path")

    if not config_path.exists():
        logging.getLogger(__name__).warning(
            f"Archivo de configuración no encontrado: {config_path}"
        )
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            full_config = json.load(f)
        return full_config.get("embedding_generation", {})
    except json.JSONDecodeError as e:
        logging.getLogger(__name__).error(f"Error al decodificar JSON en {config_path}: {e}")
        raise
    except Exception as e:
        logging.getLogger(__name__).error(
            f"Error al leer el archivo de configuración {config_path}: {e}"
        )
        raise


def main():
    """Punto de entrada principal del script."""
    parser = argparse.ArgumentParser(
        description="Genera embeddings para búsqueda semántica."
    )

    # Cargar configuración desde JSON como base
    try:
        config_path = Path(__file__).resolve().parent.parent / "config" / "config_rules.json"
        json_config = load_config_from_json(config_path)
        # Extraer parámetros de embedding_generation si existe
        if "embedding_generation" in json_config:
            json_config = json_config["embedding_generation"]
    except Exception as e:
        logging.error(f"Error al cargar configuración desde JSON: {e}")
        json_config = {}

    # Definir argumentos permitiendo sobreescritura
    parser.add_argument("--input_file", type=Path, default=json_config.get("input_file", "data/apus_clean.csv"))
    parser.add_argument("--output_dir", type=Path, default=json_config.get("output_dir", "app/embeddings"))
    parser.add_argument("--model_name", type=str, default=json_config.get("model_name", "paraphrase-multilingual-MiniLM-L12-v2"))
    parser.add_argument("--text_column", type=str, default=json_config.get("text_column", "DESCRIPCION"))
    parser.add_argument("--id_column", type=str, default=json_config.get("id_column", "CODIGO_APU"))
    parser.add_argument(
        "--max_batch_size", type=int, default=json_config.get("max_batch_size", 512)
    )
    parser.add_argument(
        "--memory_limit_gb", type=float, default=json_config.get("memory_limit_gb", 8.0)
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--no-backup", action="store_true", help="Desactiva la creación de backups"
    )
    parser.add_argument(
        "--no-normalize", action="store_true", help="No normalizar los embeddings"
    )
    parser.add_argument("--log-file", type=Path, help="Ruta opcional para archivo de logs")

    args = parser.parse_args()

    # Validar que los argumentos requeridos tengan valor
    required_args = ["input_file", "output_dir", "model_name", "text_column", "id_column"]
    missing_args = [arg for arg in required_args if getattr(args, arg) is None]

    if missing_args:
        error_msg = (
            f"Error: Faltan configuraciones requeridas en config.json o como argumentos. "
            f"Necesarios: {', '.join(required_args)}. "
            f"Faltantes: {', '.join(missing_args)}"
        )
        print(error_msg)
        sys.exit(1)

    try:
        setup_logging(args.log_level, args.log_file)
    except Exception as e:
        print(f"Error al configurar logging: {e}")
        sys.exit(1)

    # Crear objeto de configuración final
    try:
        config = ScriptConfig(
            model_name=args.model_name,
            input_file=Path(args.input_file),
            output_dir=Path(args.output_dir),
            text_column=args.text_column,
            id_column=args.id_column,
            max_batch_size=args.max_batch_size,
            memory_limit_gb=args.memory_limit_gb,
            backup_enabled=not args.no_backup,
            normalize_embeddings=not args.no_normalize,
        )
    except Exception as e:
        logging.error(f"Error al crear configuración: {e}")
        sys.exit(1)

    try:
        pipeline = EmbeddingPipeline(config)
        metrics = pipeline.run()
        print("\n📊 Resumen de Métricas:", json.dumps(metrics, indent=2, default=str))
        sys.exit(0)
    except (
        EmbeddingGenerationError,
        FileNotFoundError,
        ValueError,
        ConfigurationError,
    ) as e:
        logging.error(f"Error de ejecución: {e}")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"Error crítico no esperado: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
