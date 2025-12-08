import argparse
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
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
        raise ValueError(f"Nivel de log inválido: {log_level}. Debe ser uno de: {valid_levels}")

    # Validar archivo de log
    if log_file is not None and not isinstance(log_file, Path):
        raise TypeError("log_file debe ser una instancia de pathlib.Path o None")

    handlers = []
    
    # Console handler con colores
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = ColoredFormatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)
    
    # File handler si se especifica
    if log_file:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
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

    logging.basicConfig(level=getattr(logging, log_level.upper()), handlers=handlers, force=True)


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
        # Validar tipos
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name debe ser una cadena no vacía")
        
        if not isinstance(self.input_file, Path):
            raise TypeError("input_file debe ser una instancia de pathlib.Path")
        
        if not isinstance(self.output_dir, Path):
            raise TypeError("output_dir debe ser una instancia de pathlib.Path")
        
        if not isinstance(self.text_column, str) or not self.text_column.strip():
            raise ValueError("text_column debe ser una cadena no vacía")
        
        if not isinstance(self.id_column, str) or not self.id_column.strip():
            raise ValueError("id_column debe ser una cadena no vacía")
        
        if not isinstance(self.max_batch_size, int) or self.max_batch_size <= 0:
            raise ValueError("max_batch_size debe ser un entero positivo")
        
        if not isinstance(self.memory_limit_gb, (int, float)) or self.memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb debe ser un número positivo")
        
        if not isinstance(self.backup_enabled, bool):
            raise TypeError("backup_enabled debe ser booleano")
        
        if not isinstance(self.normalize_embeddings, bool):
            raise TypeError("normalize_embeddings debe ser booleano")
        
        if not isinstance(self.show_progress, bool):
            raise TypeError("show_progress debe ser booleano")
        
        if not isinstance(self.min_text_length, int) or self.min_text_length < 0:
            raise ValueError("min_text_length debe ser un entero no negativo")
        
        if not isinstance(self.max_text_length, int) or self.max_text_length <= 0:
            raise ValueError("max_text_length debe ser un entero positivo")
        
        if not isinstance(self.validation_sample_size, int) or self.validation_sample_size <= 0:
            raise ValueError("validation_sample_size debe ser un entero positivo")
        
        # Validar rutas
        if self.input_file.suffix.lower() not in ['.csv', '.json']:
            raise ValueError(f"Formato de archivo no soportado: {self.input_file.suffix}")
        
        # Normalizar rutas absolutas
        self.input_file = self.input_file.resolve()
        self.output_dir = self.output_dir.resolve()

    def __post_init__(self):
        """Validaciones posteriores a la inicialización."""
        # Validar tipos
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ValueError("model_name debe ser una cadena no vacía")

        if not isinstance(self.input_file, Path):
            raise TypeError("input_file debe ser una instancia de pathlib.Path")

        if not isinstance(self.output_dir, Path):
            raise TypeError("output_dir debe ser una instancia de pathlib.Path")

        if not isinstance(self.text_column, str) or not self.text_column.strip():
            raise ValueError("text_column debe ser una cadena no vacía")

        if not isinstance(self.id_column, str) or not self.id_column.strip():
            raise ValueError("id_column debe ser una cadena no vacía")

        if not isinstance(self.max_batch_size, int) or self.max_batch_size <= 0:
            raise ValueError("max_batch_size debe ser un entero positivo")

        if not isinstance(self.memory_limit_gb, (int, float)) or self.memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb debe ser un número positivo")

        if not isinstance(self.backup_enabled, bool):
            raise TypeError("backup_enabled debe ser booleano")

        if not isinstance(self.normalize_embeddings, bool):
            raise TypeError("normalize_embeddings debe ser booleano")

        if not isinstance(self.show_progress, bool):
            raise TypeError("show_progress debe ser booleano")

        if not isinstance(self.min_text_length, int) or self.min_text_length < 0:
            raise ValueError("min_text_length debe ser un entero no negativo")

        if not isinstance(self.max_text_length, int) or self.max_text_length <= 0:
            raise ValueError("max_text_length debe ser un entero positivo")

        if not isinstance(self.validation_sample_size, int) or self.validation_sample_size <= 0:
            raise ValueError("validation_sample_size debe ser un entero positivo")

        # Validar rutas
        if self.input_file.suffix.lower() not in ['.csv', '.json']:
            raise ValueError(f"Formato de archivo no soportado: {self.input_file.suffix}")

        # Normalizar rutas absolutas
        self.input_file = self.input_file.resolve()
        self.output_dir = self.output_dir.resolve()


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
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df debe ser un DataFrame de pandas")

        if not isinstance(text_column, str) or not text_column.strip():
            raise ValueError("text_column debe ser una cadena no vacía")

        if not isinstance(id_column, str) or not id_column.strip():
            raise ValueError("id_column debe ser una cadena no vacía")

        if not isinstance(config, ScriptConfig):
            raise TypeError("config debe ser una instancia de ScriptConfig")

        logger = logging.getLogger(__name__)

        # Verificar que las columnas existen
        missing_columns = [col for col in [text_column, id_column] if col not in df.columns]
        if missing_columns:
            raise DataValidationError(f"Columnas faltantes en el DataFrame: {missing_columns}")

        initial_rows = len(df)
        if initial_rows == 0:
            raise DataValidationError("El DataFrame está vacío")

        # Eliminar filas con valores nulos en las columnas críticas
        df_clean = df.dropna(subset=[text_column, id_column]).copy()
        removed_nulls = initial_rows - len(df_clean)
        if removed_nulls > 0:
            logger.warning(f"Se eliminaron {removed_nulls} filas con valores nulos en las columnas '{text_column}' o '{id_column}'")

        # Eliminar duplicados basados en la columna de ID
        df_clean = df_clean.drop_duplicates(subset=[id_column], keep="first").reset_index(drop=True)
        removed_duplicates = initial_rows - removed_nulls - len(df_clean)
        if removed_duplicates > 0:
            logger.warning(f"Se eliminaron {removed_duplicates} filas duplicadas según la columna '{id_column}'")

        # Convertir la columna de texto a string para garantizar consistencia
        df_clean[text_column] = df_clean[text_column].astype(str)

        # Aplicar filtros de longitud de texto
        df_clean["text_length"] = df_clean[text_column].str.len()
        df_filtered = df_clean[
            df_clean["text_length"].between(config.min_text_length, config.max_text_length)
        ].drop("text_length", axis=1).reset_index(drop=True)

        removed_by_length = len(df_clean) - len(df_filtered)
        if removed_by_length > 0:
            logger.warning(f"Se eliminaron {removed_by_length} filas fuera del rango de longitud de texto [{config.min_text_length}, {config.max_text_length}]")

        if df_filtered.empty:
            raise DataValidationError(
                f"No quedan datos válidos después de aplicar los filtros. "
                f"Original: {initial_rows}, nulos: {removed_nulls}, duplicados: {removed_duplicates}, longitud: {removed_by_length}"
            )

        logger.info(f"Validación completada: {len(df_filtered)}/{initial_rows} filas válidas ({len(df_filtered)/initial_rows*100:.2f}%)")
        return df_filtered


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
    @staticmethod
    def create_backup(file_path: Path) -> Optional[Path]:
        """
        Crea un backup del archivo si existe.

        Args:
            file_path: Ruta del archivo a respaldar

        Returns:
            Ruta del backup creado o None si no existía el archivo
        """
        if not isinstance(file_path, Path):
            raise TypeError("file_path debe ser una instancia de pathlib.Path")

        if not file_path.exists():
            logging.getLogger(__name__).debug(f"No se puede crear backup, archivo no existe: {file_path}")
            return None

        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_path = file_path.with_name(
                f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
            )
            shutil.copy2(file_path, backup_path)
            logger = logging.getLogger(__name__)
            logger.info(f"Backup creado: {backup_path}")
            return backup_path
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Error al crear backup de {file_path}: {e}")
            raise


    @staticmethod
    def load_data(file_path: Path) -> pd.DataFrame:
        """
        Carga datos desde archivo JSON o CSV.

        Args:
            file_path: Ruta del archivo a cargar

        Returns:
            DataFrame con los datos cargados

        Raises:
            FileNotFoundError: Si el archivo no existe
            DataValidationError: Si hay problemas al leer el archivo
            ValueError: Si el formato no es soportado
        """
        if not isinstance(file_path, Path):
            raise TypeError("file_path debe ser una instancia de pathlib.Path")

        if not file_path.exists():
            raise FileNotFoundError(f"El archivo no existe: {file_path}")

        if not file_path.is_file():
            raise FileNotFoundError(f"La ruta no es un archivo: {file_path}")

        try:
            if file_path.suffix.lower() == ".json":
                df = pd.read_json(file_path)
            elif file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path, encoding="utf-8")
            else:
                raise ValueError(f"Formato de archivo no soportado: {file_path.suffix}")

            if df.empty:
                raise DataValidationError(f"El archivo {file_path} está vacío")

            logging.getLogger(__name__).info(f"Datos cargados exitosamente: {len(df)} filas, {len(df.columns)} columnas")
            return df

        except UnicodeDecodeError:
            # Intentar con codificación alternativa
            try:
                if file_path.suffix.lower() == ".csv":
                    df = pd.read_csv(file_path, encoding="latin-1")
                    logging.getLogger(__name__).warning(f"Archivo {file_path} leído con codificación latin-1")
                    return df
                else:
                    raise DataValidationError(f"No se pudo leer el archivo JSON {file_path} con codificaciones UTF-8 o Latin-1")
            except UnicodeDecodeError:
                raise DataValidationError(f"No se pudo leer el archivo {file_path} con las codificaciones UTF-8 o Latin-1")
        except pd.errors.EmptyDataError:
            raise DataValidationError(f"El archivo {file_path} está vacío o corrupto")
        except Exception as e:
            raise DataValidationError(f"Error al leer el archivo {file_path}: {e}")


class EmbeddingGenerator:
    """Generador de embeddings con manejo robusto"""
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
            self.model = SentenceTransformer(self.config.model_name, device='cpu')
            embedding_dim = self.model.get_sentence_embedding_dimension()
            if embedding_dim is None:
                raise ModelLoadError(f"El modelo {self.config.model_name} no devolvió una dimensión de embeddings válida")
            self.logger.info(f"Modelo cargado exitosamente. Dimensión: {embedding_dim}")
        except Exception as e:
            raise ModelLoadError(f"Error al cargar modelo '{self.config.model_name}': {e}")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Genera embeddings en batches.

        Args:
            texts: Lista de textos para generar embeddings

        Returns:
            Array numpy con los embeddings generados

        Raises:
            RuntimeError: Si el modelo no está cargado
            TypeError: Si texts no es una lista de strings
        """
        if not isinstance(texts, list):
            raise TypeError("texts debe ser una lista de cadenas")

        if not all(isinstance(t, str) for t in texts):
            raise TypeError("Todos los elementos de texts deben ser cadenas")

        if not self.model:
            raise RuntimeError("El modelo no ha sido cargado. Llame a load_model() primero")

        if not texts:
            raise ValueError("La lista de textos no puede estar vacía")

        self.logger.info(f"Generando embeddings para {len(texts)} textos...")

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.config.max_batch_size,
                show_progress_bar=self.config.show_progress,
                convert_to_numpy=True,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_tensor=False,  # Asegurar que devuelve numpy array
            )

            if not isinstance(embeddings, np.ndarray):
                embeddings = np.array(embeddings)

            if embeddings.size == 0:
                raise RuntimeError("Los embeddings generados están vacíos")

            self.logger.info(f"Embeddings generados: shape={embeddings.shape}")
            return embeddings

        except Exception as e:
            raise RuntimeError(f"Error al generar embeddings: {e}")

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Construye un índice FAISS.

        Args:
            embeddings: Array numpy con los embeddings

        Returns:
            Índice FAISS construido

        Raises:
            TypeError: Si embeddings no es un array numpy
            ValueError: Si embeddings tiene dimensiones incorrectas
        """
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("embeddings debe ser un array numpy")

        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError(f"embeddings debe ser un array 2D no vacío, forma actual: {embeddings.shape}")

        self.logger.info("Construyendo índice FAISS...")

        embedding_dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)  # Inner Product para coseno (normalizados)

        # Validar que los embeddings sean float32
        embeddings_float32 = embeddings.astype(np.float32)

        index.add(embeddings_float32)

        if index.ntotal != len(embeddings):
            raise RuntimeError(f"Error al añadir embeddings al índice: esperados {len(embeddings)}, añadidos {index.ntotal}")

        self.logger.info(f"Índice construido exitosamente. Vectores: {index.ntotal}, Dimensión: {embedding_dim}")
        return index

    def validate_index(self, index: faiss.Index, embeddings: np.ndarray) -> bool:
        """
        Valida el índice FAISS con tolerancia a duplicados y análisis de similitud.
        Estrategia:
        - Acepta coincidencias exactas (mismo índice).
        - Acepta duplicados semánticos (Similitud > 0.999).
        - Rechaza desviaciones significativas.
        - Reporta estadísticas detalladas.
        
        Args:
            index: Índice FAISS a validar.
            embeddings: Embeddings originales.
            
        Returns:
            bool: True si la validación es exitosa.
        """
        if not isinstance(index, faiss.Index):
            raise TypeError("index debe ser una instancia de faiss.Index")

        if not isinstance(embeddings, np.ndarray):
            raise TypeError("embeddings debe ser un array numpy")

        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError(f"embeddings debe ser un array 2D no vacío, forma actual: {embeddings.shape}")

        if index.d != embeddings.shape[1]:
            raise ValueError(f"Dimensión del índice ({index.d}) no coincide con embeddings ({embeddings.shape[1]})")

        if index.ntotal != len(embeddings):
            raise ValueError(f"Número de vectores en índice ({index.ntotal}) no coincide con embeddings ({len(embeddings)})")

        self.logger.info("=" * 80)
        self.logger.info("Iniciando validación robusta del índice FAISS...")
        
        n_samples = min(self.config.validation_sample_size, len(embeddings))
        if n_samples == 0:
            self.logger.warning("No hay suficientes embeddings para validación")
            return True

        # Usar semilla fija para reproducibilidad en validación
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(len(embeddings), n_samples, replace=False)
        
        # Umbrales (Para IndexFlatIP/Coseno: 1.0 es idéntico)
        SIMILARITY_THRESHOLD = 0.999  # Aceptamos 99.9% de similitud como idéntico
        
        # Métricas de validación
        stats = {
            "exact_matches": 0,  # Mismo índice
            "semantic_duplicates": 0,  # Índice diferente, vector idéntico
            "failures": 0,  # Validación fallida
            "total": n_samples,
        }
        
        failed_cases = []

        # Preparar embeddings para búsqueda (asegurar tipo float32)
        embeddings_float32 = embeddings.astype(np.float32)

        for idx in sample_indices:
            query = embeddings_float32[idx : idx + 1]

            # Buscar top-k para detectar duplicados
            # Nota: 'distances' aquí son puntajes de similitud (cercanos a 1.0)
            try:
                similarities, indices = index.search(query, k=min(5, index.ntotal))
            except Exception as e:
                self.logger.error(f"Error al buscar en el índice para el índice {idx}: {e}")
                stats["failures"] += 1
                continue

            top_idx = indices[0][0]
            top_similarity = similarities[0][0]
            
            # CASO 1: Coincidencia exacta (esperado)
            if top_idx == idx:
                stats["exact_matches"] += 1
                continue
            
            # CASO 2: Duplicado semántico (aceptable)
            # Verificar si el índice correcto está en el top-k y tiene alta similitud
            if idx in indices[0]:
                position = np.where(indices[0] == idx)[0][0]
                actual_similarity = similarities[0][position]
                if actual_similarity > SIMILARITY_THRESHOLD:
                    stats["semantic_duplicates"] += 1
                    self.logger.debug(
                        f"⚠️  Duplicado detectado (Índice correcto encontrado):\n"
                        f"   Índice esperado: {idx}\n"
                        f"   Índice retornado (Top 1): {top_idx} (Similitud: {top_similarity:.6f})\n"
                        f"   Índice correcto en posición: {position + 1}/5 (Similitud: {actual_similarity:.6f})"
                    )
                    continue
            
            # CASO 3: Similitud casi perfecta con índice diferente
            # (Duplicado perfecto no indexado o colisión)
            if top_similarity > SIMILARITY_THRESHOLD:
                stats["semantic_duplicates"] += 1
                self.logger.debug(
                    f"⚠️  Vector duplicado perfecto (Índice correcto NO en Top-5):\n"
                    f"   Índice esperado: {idx}\n"
                    f"   Índice retornado: {top_idx}\n"
                    f"   Similitud: {top_similarity:.8f} (≈1.0, duplicado legítimo)\n"
                    f"   Top-5 índices: {indices[0].tolist()}"
                )
                continue
            
            # CASO 4: Fallo real de validación
            stats["failures"] += 1
            failed_cases.append({
                "expected_idx": int(idx),
                "returned_idx": int(top_idx),
                "similarity": float(top_similarity),
                "top5_indices": indices[0].tolist(),
            })
            self.logger.error(
                f"❌ Error de validación real:\n"
                f"   Índice esperado: {idx}\n"
                f"   Índice retornado: {top_idx}\n"
                f"   Similitud: {top_similarity:.6f} (< {SIMILARITY_THRESHOLD})\n"
                f"   Top-5: {indices[0].tolist()}"
            )
        
        # Reporte de estadísticas
        self.logger.info("=" * 80)
        self.logger.info("📊 Resultados de Validación:")
        self.logger.info(f"   Total muestras: {stats['total']}")
        
        if stats["total"] > 0:
            success_rate = (stats["exact_matches"] + stats["semantic_duplicates"]) / stats["total"]
            self.logger.info(
                f"   ✅ Coincidencias exactas: {stats['exact_matches']} "
                f"({stats['exact_matches'] / stats['total'] * 100:.1f}%)"
            )
            self.logger.info(
                f"   ⚠️  Duplicados semánticos: {stats['semantic_duplicates']} "
                f"({stats['semantic_duplicates'] / stats['total'] * 100:.1f}%)"
            )
            self.logger.info(
                f"   ❌ Fallos reales: {stats['failures']} "
                f"({stats['failures'] / stats['total'] * 100:.1f}%)"
            )
        else:
            self.logger.warning("⚠️  No se realizaron validaciones (0 muestras).")
            success_rate = 0.0
        
        # Criterio de aceptación: 0 fallos reales
        if stats["failures"] == 0:
            if stats["semantic_duplicates"] > 0:
                self.logger.warning(
                    f"⚠️  ADVERTENCIA: Se detectaron {stats['semantic_duplicates']} duplicados.\n"
                    f"   El sistema funciona, pero considera limpiar tus datos de APUs duplicados."
                )
            self.logger.info(
                f"✅ Validación EXITOSA (tasa de éxito funcional: {success_rate * 100:.2f}%)"
            )
            self.logger.info("=" * 80)
            return True
        else:
            self.logger.error(
                f"❌ Validación FALLIDA: {stats['failures']} errores reales detectados"
            )
            self.logger.info("=" * 80)
            return False


class EmbeddingPipeline:
    """Pipeline principal para la generación de embeddings"""
    def __init__(self, config: ScriptConfig):
        if not isinstance(config, ScriptConfig):
            raise TypeError("config debe ser una instancia de ScriptConfig")

        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.generator = EmbeddingGenerator(config)

    def estimate_memory_usage(self, n_samples: int, embedding_dim: int) -> float:
        """
        Estima el uso de memoria en GB.

        Args:
            n_samples: Número de muestras
            embedding_dim: Dimensión de los embeddings

        Returns:
            Estimación de uso de memoria en GB
        """
        if not isinstance(n_samples, int) or n_samples < 0:
            raise ValueError("n_samples debe ser un entero no negativo")

        if not isinstance(embedding_dim, int) or embedding_dim <= 0:
            raise ValueError("embedding_dim debe ser un entero positivo")

        bytes_per_embedding = embedding_dim * 4  # float32
        # Factor de overhead para operaciones temporales, índices, etc.
        total_bytes = n_samples * bytes_per_embedding * 6.0
        return total_bytes / (1024**3)

    def run(self) -> Dict[str, Union[str, int, float]]:
        """
        Ejecuta el pipeline completo.

        Returns:
            Diccionario con métricas del proceso

        Raises:
            EmbeddingGenerationError: Si falla la validación del índice
        """
        start_time = time.time()
        self.logger.info("=" * 80)
        self.logger.info("INICIANDO GENERACIÓN DE EMBEDDINGS")
        self.logger.info("=" * 80)

        # Cargar y validar datos
        df = FileManager.load_data(self.config.input_file)
        df = DataValidator.validate_dataframe(
            df, self.config.text_column, self.config.id_column, self.config
        )

        # Cargar modelo
        self.generator.load_model()
        embedding_dim = self.generator.model.get_sentence_embedding_dimension()

        # Verificar disponibilidad de memoria
        estimated_memory = self.estimate_memory_usage(len(df), embedding_dim)
        MemoryMonitor.check_memory_availability(
            estimated_memory, self.config.memory_limit_gb
        )

        # Generar embeddings
        embeddings = self.generator.generate_embeddings(df[self.config.text_column].tolist())

        # Construir índice
        index = self.generator.build_faiss_index(embeddings)

        # Validar índice
        if not self.generator.validate_index(index, embeddings):
            raise EmbeddingGenerationError("La validación del índice FAISS falló")

        # Crear mapeo de IDs
        id_map = {
            str(i): str(apu_id)
            for i, apu_id in enumerate(df[self.config.id_column])
            if pd.notna(apu_id) and str(apu_id).strip()  # Filtrar valores nulos o vacíos
        }

        # Guardar artefactos
        self.save_artifacts(index, id_map)

        # Calcular métricas finales
        metrics = {
            "processing_time": round(time.time() - start_time, 2),
            "initial_rows": len(FileManager.load_data(self.config.input_file)),  # Recargar para métrica original
            "valid_rows": len(df),
            "embeddings_generated": len(embeddings),
            "embedding_dim": embedding_dim,
            "index_vectors": index.ntotal,
            "id_map_size": len(id_map),
        }

        self.logger.info("=" * 80)
        self.logger.info("✅ PROCESO COMPLETADO EXITOSAMENTE")
        self.logger.info("=" * 80)

        return metrics

    def save_artifacts(self, index: faiss.Index, id_map: Dict[str, str]) -> None:
        """
        Guarda el índice, el mapa de IDs y los metadatos.

        Args:
            index: Índice FAISS a guardar
            id_map: Diccionario de mapeo de IDs
        """
        if not isinstance(index, faiss.Index):
            raise TypeError("index debe ser una instancia de faiss.Index")

        if not isinstance(id_map, dict):
            raise TypeError("id_map debe ser un diccionario")

        if not all(isinstance(k, str) and isinstance(v, str) for k, v in id_map.items()):
            raise TypeError("id_map debe contener solo cadenas como claves y valores")

        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        index_path = self.config.output_dir / "faiss.index"
        map_path = self.config.output_dir / "id_map.json"
        metadata_path = self.config.output_dir / "metadata.json"

        # Crear backups si está habilitado
        if self.config.backup_enabled:
            FileManager.create_backup(index_path)
            FileManager.create_backup(map_path)
            FileManager.create_backup(metadata_path)

        # Guardar índice FAISS
        self.logger.info(f"Guardando índice FAISS: {index_path}")
        try:
            faiss.write_index(index, str(index_path))
        except Exception as e:
            raise RuntimeError(f"Error al guardar el índice FAISS: {e}")

        # Guardar mapeo de IDs
        self.logger.info(f"Guardando mapeo de IDs: {map_path}")
        try:
            with open(map_path, "w", encoding="utf-8") as f:
                json.dump(id_map, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise RuntimeError(f"Error al guardar el mapeo de IDs: {e}")

        # Guardar metadatos
        metadata = {
            "model_name": self.config.model_name,
            "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "vector_count": index.ntotal,
            "text_column": self.config.text_column,
            "id_column": self.config.id_column,
            "max_batch_size": self.config.max_batch_size,
            "normalize_embeddings": self.config.normalize_embeddings,
            "input_file": str(self.config.input_file),
        }

        self.logger.info(f"Guardando metadata: {metadata_path}")
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise RuntimeError(f"Error al guardar los metadatos: {e}")


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
        logging.getLogger(__name__).warning(f"Archivo de configuración no encontrado: {config_path}")
        return {}

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            full_config = json.load(f)
        return full_config.get("embedding_generation", {})
    except json.JSONDecodeError as e:
        logging.getLogger(__name__).error(f"Error al decodificar JSON en {config_path}: {e}")
        raise
    except Exception as e:
        logging.getLogger(__name__).error(f"Error al leer el archivo de configuración {config_path}: {e}")
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
    except Exception as e:
        logging.error(f"Error al cargar configuración desde JSON: {e}")
        json_config = {}
    
    # Definir argumentos permitiendo sobreescritura
    parser.add_argument("--input_file", type=Path, default=json_config.get("input_file"))
    parser.add_argument("--output_dir", type=Path, default=json_config.get("output_dir"))
    parser.add_argument("--model_name", type=str, default=json_config.get("model_name"))
    parser.add_argument("--text_column", type=str, default=json_config.get("text_column"))
    parser.add_argument("--id_column", type=str, default=json_config.get("id_column"))
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
    parser.add_argument(
        "--log-file", type=Path, help="Ruta opcional para archivo de logs"
    )

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
    except (EmbeddingGenerationError, FileNotFoundError, ValueError, ConfigurationError) as e:
        logging.error(f"Error de ejecución: {e}")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"Error crítico no esperado: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()