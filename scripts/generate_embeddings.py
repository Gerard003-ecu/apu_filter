# scripts/generate_embeddings.py
import argparse
import json
import logging
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

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
        df: pd.DataFrame, text_column: str, id_column: str, config: ScriptConfig
    ) -> pd.DataFrame:
        """
        Valida y limpia el DataFrame de entrada.
        """
        logger = logging.getLogger(__name__)
        missing_columns = [col for col in [text_column, id_column] if col not in df.columns]
        if missing_columns:
            raise DataValidationError(f"Columnas faltantes: {missing_columns}")

        initial_rows = len(df)
        df = df.dropna(subset=[text_column, id_column])
        if len(df) < initial_rows:
            logger.warning(f"Se eliminaron {initial_rows - len(df)} filas con nulos")

        df = df.drop_duplicates(subset=[id_column], keep="first")
        if len(df) < initial_rows:
            logger.warning(f"Se eliminaron {initial_rows - len(df)} filas duplicadas")

        df[text_column] = df[text_column].astype(str)
        df["text_length"] = df[text_column].str.len()
        df = df[df["text_length"].between(config.min_text_length, config.max_text_length)]

        df = df.drop("text_length", axis=1).reset_index(drop=True)
        if df.empty:
            raise DataValidationError("No quedan datos válidos tras la limpieza")

        logger.info(f"Validación completada: {len(df)}/{initial_rows} filas válidas")
        return df


class MemoryMonitor:
    """Monitor de memoria del sistema"""

    @staticmethod
    def check_memory_availability(estimated_size_gb: float, limit_gb: float) -> None:
        """Verifica si hay suficiente memoria disponible."""
        available_gb = psutil.virtual_memory().available / (1024**3)
        if estimated_size_gb > limit_gb:
            raise InsufficientMemoryError(
                f"Requiere ~{estimated_size_gb:.2f}GB, límite {limit_gb:.2f}GB"
            )
        if estimated_size_gb > available_gb * 0.8:
            raise InsufficientMemoryError(
                f"Requiere ~{estimated_size_gb:.2f}GB, disponible {available_gb:.2f}GB"
            )


class FileManager:
    """Gestor de archivos con backup y validación"""

    @staticmethod
    def create_backup(file_path: Path) -> Optional[Path]:
        """Crea un backup del archivo si existe."""
        if not file_path.exists():
            return None
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.with_name(
            f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
        )
        shutil.copy2(file_path, backup_path)
        logging.getLogger(__name__).info(f"Backup creado: {backup_path}")
        return backup_path

    @staticmethod
    def load_data(file_path: Path) -> pd.DataFrame:
        """Carga datos desde archivo JSON o CSV."""
        if not file_path.exists():
            raise FileNotFoundError(f"El archivo no existe: {file_path}")
        try:
            if file_path.suffix == ".json":
                return pd.read_json(file_path)
            elif file_path.suffix == ".csv":
                return pd.read_csv(file_path, encoding="utf-8")
            raise ValueError(f"Formato no soportado: {file_path.suffix}")
        except Exception as e:
            raise DataValidationError(f"Error al leer el archivo {file_path}: {e}")


class EmbeddingGenerator:
    """Generador de embeddings con manejo robusto"""

    def __init__(self, config: ScriptConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model: Optional[SentenceTransformer] = None

    def load_model(self) -> None:
        """Carga el modelo de sentence-transformers."""
        try:
            self.logger.info(f"Cargando modelo: {self.config.model_name}")
            self.model = SentenceTransformer(self.config.model_name)
            embedding_dim = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"Modelo cargado. Dimensión: {embedding_dim}")
        except Exception as e:
            raise ModelLoadError(f"Error al cargar modelo '{self.config.model_name}': {e}")

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Genera embeddings en batches."""
        if not self.model:
            raise RuntimeError("El modelo no ha sido cargado")
        self.logger.info(f"Generando embeddings para {len(texts)} textos...")
        return self.model.encode(
            texts,
            batch_size=self.config.max_batch_size,
            show_progress_bar=self.config.show_progress,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize_embeddings,
        )

    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """Construye un índice FAISS."""
        self.logger.info("Construyendo índice FAISS...")
        embedding_dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(embedding_dim)
        index.add(embeddings.astype(np.float32))
        self.logger.info(f"Índice construido. Vectores: {index.ntotal}")
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
        self.logger.info("=" * 60)
        self.logger.info("Iniciando validación robusta del índice FAISS...")
        
        n_samples = min(self.config.validation_sample_size, len(embeddings))
        # Usar semilla fija para reproducibilidad en validación
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(len(embeddings), n_samples, replace=False)
        
        # Umbrales (Para IndexFlatIP/Coseno: 1.0 es idéntico)
        SIMILARITY_THRESHOLD = 0.999  # Aceptamos 99.9% de similitud como idéntico
        
        # Métricas de validación
        stats = {
            'exact_matches': 0,      # Mismo índice
            'semantic_duplicates': 0, # Índice diferente, vector idéntico
            'failures': 0,            # Validación fallida
            'total': n_samples
        }
        
        failed_cases = []
        
        for idx in sample_indices:
            query = embeddings[idx : idx + 1].astype(np.float32)
            
            # Buscar top-5 para detectar duplicados
            # Nota: 'distances' aquí son puntajes de similitud (cercanos a 1.0)
            similarities, indices = index.search(query, k=5)
            
            top_idx = indices[0][0]
            top_similarity = similarities[0][0]
            
            # CASO 1: Coincidencia exacta (esperado)
            if top_idx == idx:
                stats['exact_matches'] += 1
                continue
            
            # CASO 2: Duplicado semántico (aceptable)
            # Verificar si el índice correcto está en el top-5 y tiene alta similitud
            if idx in indices[0]:
                position = np.where(indices[0] == idx)[0][0]
                actual_similarity = similarities[0][position]
                
                if actual_similarity > SIMILARITY_THRESHOLD:
                    stats['semantic_duplicates'] += 1
                    self.logger.warning(
                        f"⚠️  Duplicado detectado (Índice correcto encontrado):\n"
                        f"   Índice esperado: {idx}\n"
                        f"   Índice retornado (Top 1): {top_idx} (Similitud: {top_similarity:.6f})\n"
                        f"   Índice correcto en posición: {position + 1}/5 (Similitud: {actual_similarity:.6f})"
                    )
                    continue
            
            # CASO 3: Similitud casi perfecta con índice diferente
            # (Duplicado perfecto no indexado o colisión)
            if top_similarity > SIMILARITY_THRESHOLD:
                stats['semantic_duplicates'] += 1
                self.logger.warning(
                    f"⚠️  Vector duplicado perfecto (Índice correcto NO en Top-5):\n"
                    f"   Índice esperado: {idx}\n"
                    f"   Índice retornado: {top_idx}\n"
                    f"   Similitud: {top_similarity:.8f} (≈1.0, duplicado legítimo)\n"
                    f"   Top-5 índices: {indices[0].tolist()}"
                )
                continue
            
            # CASO 4: Fallo real de validación
            stats['failures'] += 1
            failed_cases.append({
                'expected_idx': idx,
                'returned_idx': top_idx,
                'similarity': float(top_similarity),
                'top5_indices': indices[0].tolist()
            })
            
            self.logger.error(
                f"❌ Error de validación real:\n"
                f"   Índice esperado: {idx}\n"
                f"   Índice retornado: {top_idx}\n"
                f"   Similitud: {top_similarity:.6f} (< {SIMILARITY_THRESHOLD})\n"
            )
        
        # Reporte de estadísticas
        self.logger.info("=" * 60)
        self.logger.info("📊 Resultados de Validación:")
        self.logger.info(f"   Total muestras: {stats['total']}")
        self.logger.info(f"   ✅ Coincidencias exactas: {stats['exact_matches']} "
                         f"({stats['exact_matches']/stats['total']*100:.1f}%)")
        self.logger.info(f"   ⚠️  Duplicados semánticos: {stats['semantic_duplicates']} "
                         f"({stats['semantic_duplicates']/stats['total']*100:.1f}%)")
        self.logger.info(f"   ❌ Fallos reales: {stats['failures']} "
                         f"({stats['failures']/stats['total']*100:.1f}%)")
        
        # Criterio de aceptación: 0 fallos reales
        success_rate = (stats['exact_matches'] + stats['semantic_duplicates']) / stats['total']
        
        if stats['failures'] == 0:
            if stats['semantic_duplicates'] > 0:
                self.logger.warning(
                    f"⚠️  ADVERTENCIA: Se detectaron {stats['semantic_duplicates']} duplicados.\n"
                    f"   El sistema funciona, pero considera limpiar tus datos de APUs duplicados."
                )
            self.logger.info(f"✅ Validación EXITOSA (tasa de éxito funcional: {success_rate*100:.2f}%)")
            self.logger.info("=" * 60)
            return True
        else:
            self.logger.error(
                f"❌ Validación FALLIDA: {stats['failures']} errores reales detectados"
            )
            self.logger.info("=" * 60)
            return False


class EmbeddingPipeline:
    """Pipeline principal para la generación de embeddings"""

    def __init__(self, config: ScriptConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.generator = EmbeddingGenerator(config)

    def estimate_memory_usage(self, n_samples: int, embedding_dim: int) -> float:
        """Estima el uso de memoria en GB."""
        bytes_per_embedding = embedding_dim * 4  # float32
        total_bytes = n_samples * bytes_per_embedding * 5.0  # Factor de overhead
        return total_bytes / (1024**3)

    def run(self) -> Dict[str, Union[str, int]]:
        """Ejecuta el pipeline completo."""
        start_time = time.time()
        self.logger.info("=" * 60 + "\nINICIANDO GENERACIÓN DE EMBEDDINGS\n" + "=" * 60)

        df = FileManager.load_data(self.config.input_file)
        df = DataValidator.validate_dataframe(
            df, self.config.text_column, self.config.id_column, self.config
        )

        self.generator.load_model()
        embedding_dim = self.generator.model.get_sentence_embedding_dimension()
        estimated_memory = self.estimate_memory_usage(len(df), embedding_dim)
        MemoryMonitor.check_memory_availability(
            estimated_memory, self.config.memory_limit_gb
        )

        embeddings = self.generator.generate_embeddings(df[self.config.text_column].tolist())
        index = self.generator.build_faiss_index(embeddings)
        if not self.generator.validate_index(index, embeddings):
            raise EmbeddingGenerationError("La validación del índice FAISS falló")

        id_map = {str(i): str(apu_id) for i, apu_id in enumerate(df[self.config.id_column])}

        self.save_artifacts(index, id_map)

        metrics = {
            "processing_time": time.time() - start_time,
            "initial_rows": len(pd.read_json(self.config.input_file)),
            "valid_rows": len(df),
            "embeddings_generated": len(embeddings),
            "embedding_dim": embedding_dim,
        }

        self.logger.info("=" * 60 + "\n✅ PROCESO COMPLETADO EXITOSAMENTE\n" + "=" * 60)
        return metrics

    def save_artifacts(self, index: faiss.Index, id_map: Dict[str, str]) -> None:
        """Guarda el índice, el mapa de IDs y los metadatos."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        index_path = self.config.output_dir / "faiss.index"
        map_path = self.config.output_dir / "id_map.json"
        metadata_path = self.config.output_dir / "metadata.json"

        if self.config.backup_enabled:
            FileManager.create_backup(index_path)
            FileManager.create_backup(map_path)
            FileManager.create_backup(metadata_path)

        self.logger.info(f"Guardando índice FAISS: {index_path}")
        faiss.write_index(index, str(index_path))

        self.logger.info(f"Guardando mapeo de IDs: {map_path}")
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(id_map, f, indent=2)

        metadata = {
            "model_name": self.config.model_name,
            "creation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "vector_count": index.ntotal,
            "text_column": self.config.text_column,
            "id_column": self.config.id_column,
        }
        self.logger.info(f"Guardando metadata: {metadata_path}")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)


def main():
    """Punto de entrada principal del script."""
    parser = argparse.ArgumentParser(
        description="Genera embeddings para búsqueda semántica."
    )

    # Cargar configuración desde JSON como base
    try:
        config_path = Path(__file__).resolve().parent.parent / "app" / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            json_config = json.load(f).get("embedding_generation", {})
    except (FileNotFoundError, json.JSONDecodeError):
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
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--no-backup", action="store_true", help="Desactiva la creación de backups"
    )
    parser.add_argument(
        "--no-normalize", action="store_true", help="No normalizar los embeddings"
    )

    args = parser.parse_args()

    # Validar que los argumentos requeridos tengan valor
    required_args = ["input_file", "output_dir", "model_name", "text_column", "id_column"]
    if any(getattr(args, arg) is None for arg in required_args):
        sys.exit(
            "Error: Faltan configuraciones requeridas en config.json o como argumentos. "
            f"Necesarios: {', '.join(required_args)}"
        )

    setup_logging(args.log_level)

    # Crear objeto de configuración final
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

    try:
        pipeline = EmbeddingPipeline(config)
        metrics = pipeline.run()
        print("\n📊 Resumen de Métricas:", json.dumps(metrics, indent=2, default=str))
        sys.exit(0)
    except (EmbeddingGenerationError, FileNotFoundError, ValueError) as e:
        logging.error(f"Error de ejecución: {e}")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"Error crítico no esperado: {e}", exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
