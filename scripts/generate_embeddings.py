# scripts/generate_embeddings.py
# NOTA: Este script está en desarrollo y depende de la generación del archivo
# `data/processed_apus.json`, que actualmente está bloqueada por un problema de parsing.

import argparse
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import faiss
import numpy as np
import pandas as pd
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
    handlers = []

    # Console handler con colores
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        ColoredFormatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    handlers.append(console_handler)

    # File handler si se especifica
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        handlers.append(file_handler)

    logging.basicConfig(level=getattr(logging, log_level.upper()), handlers=handlers)


# --- Configuración y Constantes ---
@dataclass
class Config:
    """Configuración centralizada para el generador de embeddings"""

    DEFAULT_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"
    DEFAULT_INPUT_FILE: str = "data/processed_apus.json"
    DEFAULT_OUTPUT_DIR: str = "app/embeddings"
    TEXT_COLUMN: str = "original_description"
    ID_COLUMN: str = "CODIGO_APU"

    # Nuevas configuraciones para robustez
    MAX_BATCH_SIZE: int = 512
    MIN_TEXT_LENGTH: int = 3
    MAX_TEXT_LENGTH: int = 5000
    BACKUP_ENABLED: bool = True
    VALIDATION_SAMPLE_SIZE: int = 100
    MEMORY_LIMIT_GB: float = 8.0
    NORMALIZE_EMBEDDINGS: bool = True
    SHOW_PROGRESS: bool = True


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


# --- Utilidades ---
class DataValidator:
    """Validador de datos de entrada"""

    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame, text_column: str, id_column: str, config: Config
    ) -> pd.DataFrame:
        """
        Valida y limpia el DataFrame de entrada.

        Args:
            df: DataFrame a validar
            text_column: Columna de texto
            id_column: Columna de ID
            config: Configuración

        Returns:
            DataFrame validado y limpio

        Raises:
            DataValidationError: Si los datos no son válidos
        """
        logger = logging.getLogger(__name__)

        # Verificar columnas requeridas
        missing_columns = []
        if text_column not in df.columns:
            missing_columns.append(text_column)
        if id_column not in df.columns:
            missing_columns.append(id_column)

        if missing_columns:
            raise DataValidationError(f"Columnas faltantes en el archivo: {missing_columns}")

        initial_rows = len(df)

        # Limpiar valores nulos
        df = df.dropna(subset=[text_column, id_column])
        rows_after_null = len(df)

        if rows_after_null < initial_rows:
            logger.warning(
                f"Se eliminaron {initial_rows - rows_after_null} filas con valores nulos"
            )

        # Eliminar duplicados por ID
        df = df.drop_duplicates(subset=[id_column], keep="first")
        rows_after_dup = len(df)

        if rows_after_dup < rows_after_null:
            logger.warning(
                f"Se eliminaron {rows_after_null - rows_after_dup} filas duplicadas"
            )

        # Validar longitud del texto
        df[text_column] = df[text_column].astype(str)
        df["text_length"] = df[text_column].str.len()
        invalid_length = df[
            (df["text_length"] < config.MIN_TEXT_LENGTH)
            | (df["text_length"] > config.MAX_TEXT_LENGTH)
        ]

        if not invalid_length.empty:
            logger.warning(
                f"Se encontraron {len(invalid_length)} textos con longitud inválida "
                f"(min: {config.MIN_TEXT_LENGTH}, max: {config.MAX_TEXT_LENGTH})"
            )
            df = df[
                (df["text_length"] >= config.MIN_TEXT_LENGTH)
                & (df["text_length"] <= config.MAX_TEXT_LENGTH)
            ]

        df = df.drop("text_length", axis=1)
        df = df.reset_index(drop=True)

        if df.empty:
            raise DataValidationError("No quedan datos válidos después de la limpieza")

        logger.info(f"Validación completada: {len(df)}/{initial_rows} filas válidas")

        return df


class MemoryMonitor:
    """Monitor de memoria del sistema"""

    @staticmethod
    def check_memory_availability(estimated_size_gb: float, limit_gb: float) -> None:
        """
        Verifica si hay suficiente memoria disponible.

        Args:
            estimated_size_gb: Tamaño estimado en GB
            limit_gb: Límite de memoria en GB

        Raises:
            InsufficientMemoryError: Si no hay suficiente memoria
        """
        import psutil

        available_gb = psutil.virtual_memory().available / (1024**3)

        if estimated_size_gb > limit_gb:
            raise InsufficientMemoryError(
                f"El proceso requiere ~{estimated_size_gb:.2f}GB pero el "
                f"límite es {limit_gb:.2f}GB"
            )

        if estimated_size_gb > available_gb * 0.8:  # Usar máximo 80% de memoria disponible
            raise InsufficientMemoryError(
                f"Memoria insuficiente. Requerido: ~{estimated_size_gb:.2f}GB, "
                f"Disponible: {available_gb:.2f}GB"
            )


class FileManager:
    """Gestor de archivos con backup y validación"""

    @staticmethod
    def create_backup(file_path: Path) -> Optional[Path]:
        """
        Crea un backup del archivo si existe.

        Args:
            file_path: Path del archivo a respaldar

        Returns:
            Path del backup o None si no existía el archivo
        """
        if not file_path.exists():
            return None

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = (
            file_path.parent / f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
        )
        shutil.copy2(file_path, backup_path)

        logger = logging.getLogger(__name__)
        logger.info(f"Backup creado: {backup_path}")

        return backup_path

    @staticmethod
    def load_data(file_path: Path) -> pd.DataFrame:
        """
        Carga datos desde archivo JSON o CSV con validación.

        Args:
            file_path: Path del archivo

        Returns:
            DataFrame con los datos

        Raises:
            FileNotFoundError: Si el archivo no existe
            ValueError: Si el formato no es soportado
        """
        if not file_path.exists():
            raise FileNotFoundError(f"El archivo no existe: {file_path}")

        suffix = file_path.suffix.lower()

        try:
            if suffix == ".json":
                df = pd.read_json(file_path)
            elif suffix == ".csv":
                df = pd.read_csv(file_path, encoding="utf-8")
            else:
                raise ValueError(
                    f"Formato de archivo no soportado: {suffix}. Use .json o .csv"
                )
        except Exception as e:
            raise DataValidationError(f"Error al leer el archivo {file_path}: {str(e)}")

        return df


class EmbeddingGenerator:
    """Generador de embeddings con manejo robusto"""

    def __init__(self, config: Config):
        """
        Inicializa el generador de embeddings.

        Args:
            config: Configuración del sistema
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model: Optional[SentenceTransformer] = None

    def load_model(self, model_name: str) -> None:
        """
        Carga el modelo de sentence-transformers con validación.

        Args:
            model_name: Nombre del modelo

        Raises:
            ModelLoadError: Si no se puede cargar el modelo
        """
        try:
            self.logger.info(f"Cargando modelo: {model_name}")
            self.model = SentenceTransformer(model_name)

            # Validar modelo
            test_embedding = self.model.encode(["test"], show_progress_bar=False)
            embedding_dim = test_embedding.shape[1]

            self.logger.info(f"Modelo cargado exitosamente. Dimensión: {embedding_dim}")
        except Exception as e:
            raise ModelLoadError(f"Error al cargar el modelo '{model_name}': {str(e)}")

    def generate_embeddings_batch(self, texts: List[str], batch_size: int) -> np.ndarray:
        """
        Genera embeddings en batches para manejar grandes volúmenes.

        Args:
            texts: Lista de textos
            batch_size: Tamaño del batch

        Returns:
            Array de embeddings
        """
        if self.model is None:
            raise RuntimeError("El modelo no ha sido cargado")

        self.logger.info(
            f"Generando embeddings para {len(texts)} textos (batch_size={batch_size})"
        )

        embeddings_list = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_num = i // batch_size + 1

            self.logger.debug(f"Procesando batch {batch_num}/{total_batches}")

            batch_embeddings = self.model.encode(
                batch_texts,
                show_progress_bar=self.config.SHOW_PROGRESS,
                convert_to_numpy=True,
                normalize_embeddings=self.config.NORMALIZE_EMBEDDINGS,
            )

            embeddings_list.append(batch_embeddings)

        return np.vstack(embeddings_list)

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.IndexFlatIP:
        """
        Construye un índice FAISS optimizado.

        Args:
            embeddings: Array de embeddings

        Returns:
            Índice FAISS construido
        """
        self.logger.info("Construyendo índice FAISS")

        embedding_dim = embeddings.shape[1]

        # Usar IndexFlatIP para similitud de coseno con embeddings normalizados
        index = faiss.IndexFlatIP(embedding_dim)

        # Convertir a float32 si es necesario
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        index.add(embeddings)

        self.logger.info(
            f"Índice FAISS construido. Vectores: {index.ntotal}, Dimensión: {embedding_dim}"
        )

        return index

    def validate_index(
        self, index: faiss.IndexFlatIP, embeddings: np.ndarray, sample_size: int
    ) -> bool:
        """
        Valida el índice FAISS realizando búsquedas de prueba.

        Args:
            index: Índice FAISS
            embeddings: Embeddings originales
            sample_size: Número de muestras para validar

        Returns:
            True si la validación es exitosa
        """
        self.logger.info("Validando índice FAISS")

        n_samples = min(sample_size, len(embeddings))
        sample_indices = np.random.choice(len(embeddings), n_samples, replace=False)

        for idx in sample_indices:
            query = embeddings[idx : idx + 1].astype(np.float32)
            distances, indices = index.search(query, k=1)

            if indices[0][0] != idx:
                self.logger.error(f"Error de validación: índice {idx} no coincide")
                return False

        self.logger.info(f"Validación exitosa con {n_samples} muestras")
        return True


class EmbeddingPipeline:
    """Pipeline principal para la generación de embeddings"""

    def __init__(self, config: Config):
        """
        Inicializa el pipeline.

        Args:
            config: Configuración del sistema
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.generator = EmbeddingGenerator(config)
        self.validator = DataValidator()
        self.file_manager = FileManager()
        self.memory_monitor = MemoryMonitor()

    def estimate_memory_usage(self, n_samples: int, embedding_dim: int) -> float:
        """
        Estima el uso de memoria en GB.

        Args:
            n_samples: Número de muestras
            embedding_dim: Dimensión de los embeddings

        Returns:
            Estimación en GB
        """
        # Embeddings (float32) + índice FAISS + overhead
        bytes_per_embedding = embedding_dim * 4  # float32
        total_bytes = n_samples * bytes_per_embedding * 5.0  # Factor de overhead
        return total_bytes / (1024**3)

    def run(
        self,
        input_file: Path,
        output_dir: Path,
        model_name: str,
        text_column: str,
        id_column: str,
    ) -> Dict[str, Union[str, int]]:
        """
        Ejecuta el pipeline completo de generación de embeddings.

        Args:
            input_file: Archivo de entrada
            output_dir: Directorio de salida
            model_name: Nombre del modelo
            text_column: Columna de texto
            id_column: Columna de ID

        Returns:
            Diccionario con métricas del proceso
        """
        start_time = time.time()
        metrics = {}

        try:
            # 1. Cargar y validar datos
            self.logger.info("=" * 60)
            self.logger.info("INICIANDO GENERACIÓN DE EMBEDDINGS")
            self.logger.info("=" * 60)

            df = self.file_manager.load_data(input_file)
            metrics["initial_rows"] = len(df)

            df = self.validator.validate_dataframe(df, text_column, id_column, self.config)
            metrics["valid_rows"] = len(df)

            # 2. Cargar modelo
            self.generator.load_model(model_name)

            # 3. Verificar memoria
            embedding_dim = self.generator.model.get_sentence_embedding_dimension()
            estimated_memory = self.estimate_memory_usage(len(df), embedding_dim)

            self.logger.info(f"Memoria estimada: {estimated_memory:.2f}GB")

            self.memory_monitor.check_memory_availability(
                estimated_memory, self.config.MEMORY_LIMIT_GB
            )

            # 4. Generar embeddings
            texts = df[text_column].tolist()
            embeddings = self.generator.generate_embeddings_batch(
                texts, self.config.MAX_BATCH_SIZE
            )
            metrics["embeddings_generated"] = len(embeddings)

            # 5. Construir índice FAISS
            index = self.generator.build_faiss_index(embeddings)

            # 6. Validar índice
            if not self.generator.validate_index(
                index, embeddings, self.config.VALIDATION_SAMPLE_SIZE
            ):
                raise EmbeddingGenerationError("La validación del índice FAISS falló")

            # 7. Crear mapeo de IDs
            id_map = {str(i): str(apu_id) for i, apu_id in enumerate(df[id_column])}

            # 8. Guardar artefactos con backup
            output_dir.mkdir(parents=True, exist_ok=True)

            index_path = output_dir / "faiss.index"
            map_path = output_dir / "id_map.json"
            metadata_path = output_dir / "metadata.json"

            if self.config.BACKUP_ENABLED:
                self.file_manager.create_backup(index_path)
                self.file_manager.create_backup(map_path)

            # Guardar índice
            self.logger.info(f"Guardando índice FAISS: {index_path}")
            faiss.write_index(index, str(index_path))

            # Guardar mapeo
            self.logger.info(f"Guardando mapeo de IDs: {map_path}")
            with open(map_path, "w", encoding="utf-8") as f:
                json.dump(id_map, f, indent=2, ensure_ascii=False)

            # Guardar metadata
            metadata = {
                "model_name": model_name,
                "embedding_dimension": embedding_dim,
                "total_vectors": index.ntotal,
                "normalized": self.config.NORMALIZE_EMBEDDINGS,
                "text_column": text_column,
                "id_column": id_column,
                "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "processing_time_seconds": time.time() - start_time,
            }

            self.logger.info(f"Guardando metadata: {metadata_path}")
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            # Métricas finales
            metrics["processing_time"] = time.time() - start_time
            metrics["index_size"] = index.ntotal
            metrics["embedding_dim"] = embedding_dim

            # Verificar integridad de archivos guardados
            if not index_path.exists() or not map_path.exists():
                raise EmbeddingGenerationError("Error al guardar los archivos de salida")

            self.logger.info("=" * 60)
            self.logger.info("✅ PROCESO COMPLETADO EXITOSAMENTE")
            self.logger.info(f"Tiempo total: {metrics['processing_time']:.2f} segundos")
            self.logger.info(f"Vectores generados: {metrics['index_size']}")
            self.logger.info("=" * 60)

            return metrics

        except Exception as e:
            self.logger.error(f"Error en el pipeline: {str(e)}", exc_info=True)
            raise


def main():
    """Punto de entrada principal del script."""

    # Configuración de argumentos
    parser = argparse.ArgumentParser(
        description="Genera embeddings y un índice FAISS para búsqueda semántica de APUs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Uso básico con configuración por defecto
  python scripts/generate_embeddings.py

  # Especificar archivo de entrada y modelo personalizado
  python scripts/generate_embeddings.py --input data/apus.csv --model all-MiniLM-L6-v2

  # Con todas las opciones
  python scripts/generate_embeddings.py \\
    --input data/apus.json \\
    --output app/embeddings \\
    --model paraphrase-multilingual-MiniLM-L12-v2 \\
    --text-col description \\
    --id-col code \\
    --batch-size 256 \\
    --log-level DEBUG
        """,
    )

    config = Config()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(config.DEFAULT_INPUT_FILE),
        help=f"Ruta al archivo de entrada (JSON/CSV). Default: {config.DEFAULT_INPUT_FILE}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(config.DEFAULT_OUTPUT_DIR),
        help=f"Directorio de salida. Default: {config.DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.DEFAULT_MODEL,
        help=f"Modelo de sentence-transformers. Default: {config.DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--text-col",
        type=str,
        default=config.TEXT_COLUMN,
        help=f"Columna de texto. Default: {config.TEXT_COLUMN}",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default=config.ID_COLUMN,
        help=f"Columna de ID. Default: {config.ID_COLUMN}",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.MAX_BATCH_SIZE,
        help=f"Tamaño del batch para procesamiento. Default: {config.MAX_BATCH_SIZE}",
    )
    parser.add_argument(
        "--memory-limit",
        type=float,
        default=config.MEMORY_LIMIT_GB,
        help=f"Límite de memoria en GB. Default: {config.MEMORY_LIMIT_GB}",
    )
    parser.add_argument(
        "--no-backup", action="store_true", help="Desactiva la creación de backups"
    )
    parser.add_argument(
        "--no-normalize", action="store_true", help="No normalizar los embeddings"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Nivel de logging. Default: INFO",
    )
    parser.add_argument("--log-file", type=Path, help="Archivo para guardar logs")

    args = parser.parse_args()

    # Configurar logging
    setup_logging(args.log_level, args.log_file)

    # Actualizar configuración con argumentos
    config.MAX_BATCH_SIZE = args.batch_size
    config.MEMORY_LIMIT_GB = args.memory_limit
    config.BACKUP_ENABLED = not args.no_backup
    config.NORMALIZE_EMBEDDINGS = not args.no_normalize

    # Ejecutar pipeline
    try:
        pipeline = EmbeddingPipeline(config)
        metrics = pipeline.run(
            args.input, args.output, args.model, args.text_col, args.id_col
        )

        # Mostrar resumen de métricas
        print("\n📊 Resumen de Métricas:")
        print("-" * 40)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")

        sys.exit(0)

    except KeyboardInterrupt:
        print("\n⚠️ Proceso interrumpido por el usuario")
        sys.exit(130)

    except EmbeddingGenerationError as e:
        logging.error(f"Error en generación de embeddings: {e}")
        sys.exit(1)

    except Exception as e:
        logging.critical(f"Error crítico no esperado: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
