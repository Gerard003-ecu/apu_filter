"""
Módulo Director de Pipeline.

Este módulo orquesta el flujo completo de procesamiento de datos, desde la carga
de archivos crudos hasta la generación de estructuras de datos validadas y listas
para su consumo por el frontend o análisis posteriores.

Implementa el patrón "Pipeline" donde cada paso es una unidad discreta de trabajo
que recibe un contexto, lo transforma y lo pasa al siguiente paso.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd

from app.telemetry import TelemetryContext

from .data_loader import load_data
from .data_validator import validate_and_clean_data
from .flux_condenser import CondenserConfig, DataFluxCondenser
from .utils import (
    clean_apu_code,
    find_and_rename_columns,
    normalize_text_series,
    sanitize_for_json,
)

logger = logging.getLogger(__name__)

# ==================== CONSTANTES Y CLASES AUXILIARES ====================


class ColumnNames:
    """
    Constantes para nombres de columnas estandarizados en todo el sistema.
    Facilita la refactorización y evita errores por cadenas mágicas.
    """

    CODIGO_APU = "CODIGO_APU"
    DESCRIPCION_APU = "DESCRIPCION_APU"
    DESCRIPCION_SECUNDARIA = "descripcion_secundaria"
    ORIGINAL_DESCRIPTION = "original_description"
    UNIDAD_APU = "UNIDAD_APU"
    CANTIDAD_APU = "CANTIDAD_APU"
    CANTIDAD_PRESUPUESTO = "CANTIDAD_PRESUPUESTO"
    GRUPO_INSUMO = "GRUPO_INSUMO"
    DESCRIPCION_INSUMO = "DESCRIPCION_INSUMO"
    DESCRIPCION_INSUMO_NORM = "DESCRIPCION_INSUMO_NORM"
    VR_UNITARIO_INSUMO = "VR_UNITARIO_INSUMO"
    UNIDAD_INSUMO = "UNIDAD_INSUMO"
    NORMALIZED_DESC = "NORMALIZED_DESC"
    COSTO_INSUMO_EN_APU = "COSTO_INSUMO_EN_APU"
    VR_UNITARIO_FINAL = "VR_UNITARIO_FINAL"
    VALOR_TOTAL_APU = "VALOR_TOTAL_APU"
    PRECIO_UNIT_APU = "PRECIO_UNIT_APU"
    MATERIALES = "MATERIALES"
    MANO_DE_OBRA = "MANO DE OBRA"
    EQUIPO = "EQUIPO"
    OTROS = "OTROS"
    VALOR_SUMINISTRO_UN = "VALOR_SUMINISTRO_UN"
    VALOR_INSTALACION_UN = "VALOR_INSTALACION_UN"
    VALOR_CONSTRUCCION_UN = "VALOR_CONSTRUCCION_UN"
    VALOR_SUMINISTRO_TOTAL = "VALOR_SUMINISTRO_TOTAL"
    VALOR_INSTALACION_TOTAL = "VALOR_INSTALACION_TOTAL"
    VALOR_CONSTRUCCION_TOTAL = "VALOR_CONSTRUCCION_TOTAL"
    TIPO_INSUMO = "TIPO_INSUMO"
    TIPO_APU = "tipo_apu"
    CATEGORIA = "CATEGORIA"
    TIEMPO_INSTALACION = "TIEMPO_INSTALACION"
    RENDIMIENTO = "RENDIMIENTO"
    RENDIMIENTO_DIA = "RENDIMIENTO_DIA"


class InsumoTypes:
    """Tipos de insumos reconocidos por el sistema."""

    SUMINISTRO = "SUMINISTRO"
    MANO_DE_OBRA = "MANO_DE_OBRA"
    EQUIPO = "EQUIPO"
    TRANSPORTE = "TRANSPORTE"
    OTRO = "OTRO"


class APUTypes:
    """Clasificación de tipos de APU."""

    INSTALACION = "Instalación"
    SUMINISTRO = "Suministro"
    SUMINISTRO_PREFABRICADO = "Suministro (Pre-fabricado)"
    OBRA_COMPLETA = "Obra Completa"
    INDEFINIDO = "Indefinido"


@dataclass
class ProcessingThresholds:
    """
    Umbrales configurables para la validación y limpieza de datos.

    Attributes:
        outlier_std_multiplier: Desviaciones estándar para detectar outliers.
        max_quantity: Cantidad máxima permitida.
        max_cost_per_item: Costo unitario máximo permitido.
        max_total_cost: Costo total máximo permitido.
        instalacion_mo_threshold: Umbral % MO para considerar Instalación.
        suministro_mat_threshold: Umbral % Materiales para Suministro.
        # ... otros umbrales ...
    """

    outlier_std_multiplier: float = 3.0
    max_quantity: float = 1e6
    max_cost_per_item: float = 1e9
    max_total_cost: float = 1e11
    instalacion_mo_threshold: float = 75.0
    suministro_mat_threshold: float = 75.0
    suministro_mo_max: float = 15.0
    prefabricado_mat_threshold: float = 65.0
    prefabricado_mo_min: float = 15.0
    max_header_search_rows: int = 10


# ==================== CLASE BASE REFACTORIZADA ====================


class ProcessingStep(ABC):
    """Clase base abstracta para un paso del pipeline de procesamiento."""

    @abstractmethod
    def execute(self, context: dict, telemetry: TelemetryContext) -> dict:
        """
        Ejecuta la lógica del paso.

        Args:
            context: Diccionario con el estado actual del procesamiento.
            telemetry: Contexto de telemetría para registrar métricas y errores.

        Returns:
            El contexto actualizado (puede ser el mismo objeto modificado).
        """
        pass


# ==================== VALIDADORES ====================


class DataValidator:
    """Utilidades para validación de DataFrames."""

    @staticmethod
    def validate_dataframe_not_empty(
        df: pd.DataFrame, name: str
    ) -> Tuple[bool, Optional[str]]:
        """Verifica que un DataFrame no esté vacío."""
        # ROBUSTECIDO: Verificación de tipo antes de operaciones
        if df is None:
            error_msg = f"DataFrame '{name}' es None"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        if not isinstance(df, pd.DataFrame):
            error_msg = f"'{name}' no es un DataFrame válido, es tipo: {type(df).__name__}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        if df.empty:
            error_msg = f"DataFrame '{name}' está vacío (0 filas)"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        # ROBUSTECIDO: Verificar que no sea solo columnas sin datos útiles
        if df.dropna(how='all').empty:
            error_msg = f"DataFrame '{name}' contiene solo valores nulos"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        return True, None

    @staticmethod
    def validate_required_columns(
        df: pd.DataFrame, required_cols: List[str], df_name: str
    ) -> Tuple[bool, Optional[str]]:
        """Verifica que un DataFrame tenga las columnas requeridas."""
        # ROBUSTECIDO: Validación de entrada
        if df is None:
            error_msg = f"DataFrame '{df_name}' es None, no se pueden validar columnas"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        if not isinstance(df, pd.DataFrame):
            error_msg = f"'{df_name}' no es un DataFrame válido"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        if not required_cols:
            logger.warning(f"⚠️ Lista de columnas requeridas vacía para '{df_name}'")
            return True, None

        # ROBUSTECIDO: Normalizar nombres para comparación case-insensitive
        df_cols_upper = {col.upper().strip(): col for col in df.columns}
        missing_cols = []

        for req_col in required_cols:
            req_col_upper = req_col.upper().strip()
            if req_col not in df.columns and req_col_upper not in df_cols_upper:
                missing_cols.append(req_col)

        if missing_cols:
            available_cols = list(df.columns)[:10]  # Limitar para log
            error_msg = (
                f"Faltan columnas requeridas en '{df_name}': {missing_cols}. "
                f"Columnas disponibles (primeras 10): {available_cols}"
            )
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        return True, None

    @staticmethod
    def detect_and_log_duplicates(
        df: pd.DataFrame, subset_cols: List[str], df_name: str, keep: str = "first"
    ) -> pd.DataFrame:
        """Detecta, loguea y elimina duplicados en un DataFrame."""
        # ROBUSTECIDO: Validaciones de entrada exhaustivas
        if df is None or not isinstance(df, pd.DataFrame):
            logger.error(f"❌ DataFrame '{df_name}' inválido para detección de duplicados")
            return pd.DataFrame()

        if df.empty:
            logger.debug(f"DataFrame '{df_name}' vacío, no hay duplicados que detectar")
            return df

        if not subset_cols:
            logger.warning(f"⚠️ No se especificaron columnas para detectar duplicados en '{df_name}'")
            return df

        # ROBUSTECIDO: Verificar que las columnas existan
        missing_subset_cols = [col for col in subset_cols if col not in df.columns]
        if missing_subset_cols:
            logger.error(
                f"❌ Columnas para duplicados no existen en '{df_name}': {missing_subset_cols}"
            )
            # Usar solo las columnas que sí existen
            subset_cols = [col for col in subset_cols if col in df.columns]
            if not subset_cols:
                return df

        # ROBUSTECIDO: Validar parámetro keep
        valid_keep_values = {"first", "last", False}
        if keep not in valid_keep_values:
            logger.warning(f"⚠️ Valor 'keep={keep}' inválido, usando 'first'")
            keep = "first"

        try:
            duplicates = df[df.duplicated(subset=subset_cols, keep=False)]
            if not duplicates.empty:
                unique_dupes = duplicates[subset_cols[0]].unique()
                num_dupes = len(duplicates)
                logger.warning(
                    f"⚠️ Se encontraron {num_dupes} filas duplicadas en '{df_name}' "
                    f"por {subset_cols}. Se conservará: '{keep}'"
                )
                # ROBUSTECIDO: Limitar cantidad de valores logueados
                dupes_sample = unique_dupes[:10].tolist()
                logger.debug(f"Muestra de valores duplicados en '{df_name}': {dupes_sample}")

                df_clean = df.drop_duplicates(subset=subset_cols, keep=keep)
                rows_removed = len(df) - len(df_clean)
                logger.info(f"✅ Duplicados eliminados en '{df_name}': {rows_removed} filas removidas")
                return df_clean
            return df
        except Exception as e:
            logger.error(f"❌ Error detectando duplicados en '{df_name}': {e}")
            return df


class FileValidator:
    """Utilidades para validación de existencia de archivos."""

    # ROBUSTECIDO: Constantes para validación
    MIN_FILE_SIZE_BYTES = 10  # Archivo debe tener al menos 10 bytes
    VALID_EXTENSIONS = {'.csv', '.xlsx', '.xls', '.json'}

    @staticmethod
    def validate_file_exists(
        file_path: str,
        file_type: str,
        check_extension: bool = True,
        min_size: Optional[int] = None
    ) -> Tuple[bool, Optional[str]]:
        """Verifica que un archivo exista y sea accesible."""
        # ROBUSTECIDO: Validar entrada
        if not file_path:
            error_msg = f"Ruta de archivo de {file_type} está vacía o es None"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        if not isinstance(file_path, (str, Path)):
            error_msg = f"Ruta de {file_type} no es válida: tipo {type(file_path).__name__}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        try:
            path = Path(file_path).resolve()  # ROBUSTECIDO: Resolver ruta absoluta
        except Exception as e:
            error_msg = f"Ruta de {file_type} no es válida: {file_path}. Error: {e}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        if not path.exists():
            error_msg = f"Archivo de {file_type} no encontrado: {path}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        if not path.is_file():
            error_msg = f"La ruta de {file_type} no es un archivo: {path}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        # ROBUSTECIDO: Verificar permisos de lectura
        try:
            if not os.access(path, os.R_OK):
                error_msg = f"Sin permisos de lectura para {file_type}: {path}"
                logger.error(f"❌ {error_msg}")
                return False, error_msg
        except Exception as e:
            logger.warning(f"⚠️ No se pudo verificar permisos para {path}: {e}")

        # ROBUSTECIDO: Verificar tamaño mínimo
        min_size = min_size or FileValidator.MIN_FILE_SIZE_BYTES
        try:
            file_size = path.stat().st_size
            if file_size < min_size:
                error_msg = f"Archivo de {file_type} demasiado pequeño ({file_size} bytes): {path}"
                logger.error(f"❌ {error_msg}")
                return False, error_msg
        except OSError as e:
            error_msg = f"No se pudo obtener información del archivo {file_type}: {e}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        # ROBUSTECIDO: Verificar extensión válida
        if check_extension:
            ext = path.suffix.lower()
            if ext not in FileValidator.VALID_EXTENSIONS:
                logger.warning(
                    f"⚠️ Extensión '{ext}' de {file_type} no es estándar. "
                    f"Extensiones esperadas: {FileValidator.VALID_EXTENSIONS}"
                )

        logger.debug(f"✅ Archivo de {file_type} validado: {path} ({file_size} bytes)")
        return True, None


# ==================== IMPLEMENTACIÓN DE PASOS ====================


class LoadDataStep(ProcessingStep):
    """
    Paso de Carga de Datos.
    Carga los archivos CSV/Excel de presupuesto, APUs e insumos.
    """

    def __init__(self, config: dict, thresholds: ProcessingThresholds):
        # ROBUSTECIDO: Validar config al inicializar
        if not config or not isinstance(config, dict):
            raise ValueError("Configuración inválida para LoadDataStep")
        self.config = config
        self.thresholds = thresholds or ProcessingThresholds()

    def execute(self, context: dict, telemetry: TelemetryContext) -> dict:
        """Ejecuta la carga y validación inicial de archivos."""
        telemetry.start_step("load_data")

        try:
            # ROBUSTECIDO: Extraer rutas con validación
            required_paths = ["presupuesto_path", "apus_path", "insumos_path"]
            paths = {}

            for path_key in required_paths:
                path_value = context.get(path_key)
                if not path_value:
                    error = f"Ruta requerida '{path_key}' no encontrada en contexto"
                    telemetry.record_error("load_data", error)
                    raise ValueError(error)
                paths[path_key] = path_value

            presupuesto_path = paths["presupuesto_path"]
            apus_path = paths["apus_path"]
            insumos_path = paths["insumos_path"]

            # Validar existencia de archivos
            file_validator = FileValidator()
            validations = [
                (presupuesto_path, "presupuesto"),
                (apus_path, "APUs"),
                (insumos_path, "insumos"),
            ]

            for file_path, file_type in validations:
                is_valid, error = file_validator.validate_file_exists(file_path, file_type)
                if not is_valid:
                    telemetry.record_error("load_data", error)
                    raise ValueError(error)

            # ROBUSTECIDO: Obtener perfiles con validación
            file_profiles = self.config.get("file_profiles")
            if not file_profiles or not isinstance(file_profiles, dict):
                raise ValueError("'file_profiles' no encontrado o inválido en config")

            # Procesar Presupuesto
            presupuesto_profile = file_profiles.get("presupuesto_default")
            if not presupuesto_profile:
                raise ValueError("No se encontró 'presupuesto_default' en file_profiles")

            p_processor = PresupuestoProcessor(
                self.config, self.thresholds, presupuesto_profile
            )
            df_presupuesto = p_processor.process(presupuesto_path)

            # ROBUSTECIDO: Validar resultado antes de continuar
            if df_presupuesto is None or df_presupuesto.empty:
                error = "Procesamiento de presupuesto retornó DataFrame vacío"
                telemetry.record_error("load_data", error)
                raise ValueError(error)

            telemetry.record_metric("load_data", "presupuesto_rows", len(df_presupuesto))

            # Procesar Insumos
            insumos_profile = file_profiles.get("insumos_default")
            if not insumos_profile:
                raise ValueError("No se encontró 'insumos_default' en file_profiles")

            i_processor = InsumosProcessor(self.thresholds, insumos_profile)
            df_insumos = i_processor.process(insumos_path)

            if df_insumos is None or df_insumos.empty:
                error = "Procesamiento de insumos retornó DataFrame vacío"
                telemetry.record_error("load_data", error)
                raise ValueError(error)

            telemetry.record_metric("load_data", "insumos_rows", len(df_insumos))

            # Procesar APUs
            apus_profile = file_profiles.get("apus_default")
            if not apus_profile:
                raise ValueError("No se encontró 'apus_default' en file_profiles")

            logger.info("⚡️ Iniciando DataFluxCondenser para APUs...")
            condenser_config_data = self.config.get("flux_condenser_config", {})

            # ROBUSTECIDO: Manejar errores de configuración del condenser
            try:
                condenser_config = CondenserConfig(**condenser_config_data)
            except TypeError as e:
                logger.warning(f"⚠️ Error en config de condenser, usando defaults: {e}")
                condenser_config = CondenserConfig()

            condenser = DataFluxCondenser(
                config=self.config, profile=apus_profile, condenser_config=condenser_config
            )
            df_apus_raw = condenser.stabilize(apus_path)

            # ROBUSTECIDO: Verificar stats antes de registrar
            stats = condenser.get_processing_stats() or {}
            for metric_name, default_value in [
                ("avg_saturation", 0.0),
                ("max_flyback_voltage", 0.0),
                ("max_dissipated_power", 0.0),
                ("avg_kinetic_energy", 0.0),
                ("avg_batch_size", 0),
            ]:
                value = stats.get(metric_name, default_value)
                # ROBUSTECIDO: Sanitizar valores numéricos
                if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                    value = default_value
                telemetry.record_metric("flux_condenser", metric_name, value)

            if df_apus_raw is None or df_apus_raw.empty:
                error = "DataFluxCondenser retornó DataFrame vacío"
                telemetry.record_error("load_data", error)
                raise ValueError(error)

            telemetry.record_metric("load_data", "apus_raw_rows", len(df_apus_raw))
            logger.info("✅ DataFluxCondenser completado.")

            # Validación final
            data_validator = DataValidator()
            dataframes = [
                (df_presupuesto, "presupuesto"),
                (df_insumos, "insumos"),
                (df_apus_raw, "APUs"),
            ]

            for df, name in dataframes:
                is_valid, error = data_validator.validate_dataframe_not_empty(df, name)
                if not is_valid:
                    telemetry.record_error("load_data", error)
                    raise ValueError(error)

            # ROBUSTECIDO: Actualizar contexto de forma inmutable
            context = {**context}
            context.update({
                "df_presupuesto": df_presupuesto,
                "df_insumos": df_insumos,
                "df_apus_raw": df_apus_raw,
            })

            telemetry.end_step("load_data", "success")
            return context

        except Exception as e:
            logger.error(f"❌ Error en LoadDataStep: {e}", exc_info=True)
            telemetry.record_error("load_data", str(e))
            telemetry.end_step("load_data", "error")
            raise


class MergeDataStep(ProcessingStep):
    """
    Paso de Fusión de Datos.
    Combina los datos crudos de APUs con la base de datos de insumos.
    """

    def __init__(self, config: dict, thresholds: ProcessingThresholds):
        # config se añade para compatibilidad con signature dinámica, aunque no se use
        self.config = config
        self.thresholds = thresholds

    def execute(self, context: dict, telemetry: TelemetryContext) -> dict:
        """Ejecuta la fusión de DataFrames."""
        telemetry.start_step("merge_data")
        try:
            df_apus_raw = context["df_apus_raw"]
            df_insumos = context["df_insumos"]

            merger = DataMerger(self.thresholds)
            df_merged = merger.merge_apus_with_insumos(df_apus_raw, df_insumos)

            telemetry.record_metric("merge_data", "merged_rows", len(df_merged))
            context["df_merged"] = df_merged

            telemetry.end_step("merge_data", "success")
            return context
        except Exception as e:
            telemetry.record_error("merge_data", str(e))
            telemetry.end_step("merge_data", "error")
            raise


class CalculateCostsStep(ProcessingStep):
    """
    Paso de Cálculo de Costos.
    Calcula costos unitarios, tiempos y rendimientos de los APUs.
    """

    def __init__(self, config: dict, thresholds: ProcessingThresholds):
        self.config = config
        self.thresholds = thresholds

    def execute(self, context: dict, telemetry: TelemetryContext) -> dict:
        """Ejecuta los cálculos de costos y tiempos."""
        telemetry.start_step("calculate_costs")
        try:
            df_merged = context["df_merged"]
            df_merged = calculate_insumo_costs(df_merged, self.thresholds)

            cost_calculator = APUCostCalculator(self.config, self.thresholds)
            df_apu_costos, df_tiempo, df_rendimiento = cost_calculator.calculate(df_merged)

            telemetry.record_metric(
                "calculate_costs", "costos_calculated", len(df_apu_costos)
            )

            context.update(
                {
                    "df_merged": df_merged,
                    "df_apu_costos": df_apu_costos,
                    "df_tiempo": df_tiempo,
                    "df_rendimiento": df_rendimiento,
                }
            )
            telemetry.end_step("calculate_costs", "success")
            return context
        except Exception as e:
            telemetry.record_error("calculate_costs", str(e))
            telemetry.end_step("calculate_costs", "error")
            raise


class FinalMergeStep(ProcessingStep):
    """
    Paso de Fusión Final.
    Integra los costos calculados con el presupuesto original.
    """

    def __init__(self, config: dict, thresholds: ProcessingThresholds):
        self.config = config
        self.thresholds = thresholds

    def execute(self, context: dict, telemetry: TelemetryContext) -> dict:
        """Ejecuta la fusión final con el presupuesto."""
        telemetry.start_step("final_merge")
        try:
            df_presupuesto = context["df_presupuesto"]
            df_apu_costos = context["df_apu_costos"]
            df_tiempo = context["df_tiempo"]

            merger = DataMerger(self.thresholds)
            df_final = merger.merge_with_presupuesto(df_presupuesto, df_apu_costos)

            df_final = pd.merge(df_final, df_tiempo, on=ColumnNames.CODIGO_APU, how="left")
            df_final = group_and_split_description(df_final)
            df_final = calculate_total_costs(df_final, self.thresholds)

            telemetry.record_metric("final_merge", "final_rows", len(df_final))
            context["df_final"] = df_final

            telemetry.end_step("final_merge", "success")
            return context
        except Exception as e:
            telemetry.record_error("final_merge", str(e))
            telemetry.end_step("final_merge", "error")
            raise


class BuildOutputStep(ProcessingStep):
    """
    Paso de Construcción de Salida.
    Prepara y valida el diccionario final de resultados para el cliente.
    """

    def __init__(self, config: dict, thresholds: ProcessingThresholds):
        self.config = config
        self.thresholds = thresholds

    def execute(self, context: dict, telemetry: TelemetryContext) -> dict:
        """Construye y valida la estructura de salida."""
        telemetry.start_step("build_output")
        try:
            df_final = context["df_final"]
            df_insumos = context["df_insumos"]
            df_merged = context["df_merged"]
            df_apus_raw = context["df_apus_raw"]
            df_apu_costos = context["df_apu_costos"]
            df_tiempo = context["df_tiempo"]
            df_rendimiento = context["df_rendimiento"]

            df_merged = synchronize_data_sources(df_merged, df_final)
            df_processed_apus = build_processed_apus_dataframe(
                df_apu_costos, df_apus_raw, df_tiempo, df_rendimiento
            )

            result_dict = build_output_dictionary(
                df_final, df_insumos, df_merged, df_apus_raw, df_processed_apus
            )

            validated_result = validate_and_clean_data(result_dict)
            validated_result["raw_insumos_df"] = df_insumos.to_dict("records")

            context["final_result"] = validated_result
            telemetry.end_step("build_output", "success")
            return context
        except Exception as e:
            telemetry.record_error("build_output", str(e))
            telemetry.end_step("build_output", "error")
            raise


# ==================== PIPELINE DIRECTOR ====================


class PipelineDirector:
    """
    Orquesta la ejecución secuencial de los pasos del pipeline.
    Utiliza una 'receta' de configuración para determinar el orden y
    activación de los pasos.
    """

    STEP_REGISTRY: Dict[str, Type[ProcessingStep]] = {
        "load_data": LoadDataStep,
        "merge_data": MergeDataStep,
        "calculate_costs": CalculateCostsStep,
        "final_merge": FinalMergeStep,
        "build_output": BuildOutputStep,
    }

    def __init__(self, config: dict, telemetry: TelemetryContext):
        self.config = config
        self.telemetry = telemetry
        self.thresholds = self._load_thresholds(config)

    def _load_thresholds(self, config: dict) -> ProcessingThresholds:
        """Carga y configura los umbrales de procesamiento."""
        thresholds = ProcessingThresholds()
        if "processing_thresholds" in config:
            for key, value in config["processing_thresholds"].items():
                if hasattr(thresholds, key):
                    setattr(thresholds, key, value)
        return thresholds

    def execute(self, initial_context: dict) -> dict:
        """
        Ejecuta el pipeline completo con manejo robusto de errores.
        """
        # ROBUSTECIDO: Validar contexto inicial
        if initial_context is None:
            raise ValueError("Contexto inicial es None")

        if not isinstance(initial_context, dict):
            raise ValueError(f"Contexto inicial debe ser dict, recibido: {type(initial_context)}")

        recipe = self.config.get("pipeline_recipe", [])

        if not recipe:
            logger.warning("No 'pipeline_recipe' en config. Usando flujo por defecto.")
            recipe = [
                {"step": "load_data", "enabled": True},
                {"step": "merge_data", "enabled": True},
                {"step": "calculate_costs", "enabled": True},
                {"step": "final_merge", "enabled": True},
                {"step": "build_output", "enabled": True},
            ]

        # ROBUSTECIDO: Validar receta
        if not isinstance(recipe, list):
            raise ValueError("pipeline_recipe debe ser una lista")

        # Crear copia inmutable del contexto
        context = dict(initial_context)
        executed_steps = []  # Para tracking y posible rollback

        for step_idx, step_config in enumerate(recipe):
            # ROBUSTECIDO: Validar configuración del paso
            if not isinstance(step_config, dict):
                logger.warning(f"⚠️ Configuración de paso #{step_idx} inválida, saltando")
                continue

            step_name = step_config.get("step")
            if not step_name:
                logger.warning(f"⚠️ Paso #{step_idx} sin nombre, saltando")
                continue

            if not step_config.get("enabled", True):
                logger.info(f"⏭️ Saltando paso deshabilitado: {step_name}")
                continue

            step_class = self.STEP_REGISTRY.get(step_name)
            if not step_class:
                error_msg = f"Paso desconocido en receta: {step_name}"
                logger.error(f"❌ {error_msg}")
                self.telemetry.record_error("pipeline_director", error_msg)

                # ROBUSTECIDO: Decidir si continuar o fallar
                if step_config.get("required", True):
                    raise ValueError(error_msg)
                continue

            logger.info(f"▶️ Ejecutando paso [{step_idx + 1}/{len(recipe)}]: {step_name}")

            try:
                # Instanciar paso
                step_instance = step_class(self.config, self.thresholds)

                # Ejecutar con timeout implícito si es necesario
                context = step_instance.execute(context, self.telemetry)

                # ROBUSTECIDO: Validar que context sigue siendo válido
                if context is None:
                    raise ValueError(f"Paso '{step_name}' retornó contexto None")

                if not isinstance(context, dict):
                    raise ValueError(
                        f"Paso '{step_name}' retornó tipo inválido: {type(context)}"
                    )

                executed_steps.append(step_name)
                logger.info(f"✅ Paso completado: {step_name}")

            except Exception as e:
                error_msg = f"Error crítico en paso '{step_name}': {e}"
                logger.critical(f"🔥 {error_msg}")
                self.telemetry.record_error(step_name, str(e))

                # ROBUSTECIDO: Registrar pasos ejecutados para diagnóstico
                self.telemetry.record_metric(
                    "pipeline_director",
                    "executed_steps_before_failure",
                    len(executed_steps)
                )

                # ROBUSTECIDO: Información de diagnóstico
                context["_pipeline_error"] = {
                    "step": step_name,
                    "step_index": step_idx,
                    "error": str(e),
                    "executed_steps": executed_steps,
                }

                raise RuntimeError(error_msg) from e

        logger.info(f"🎉 Pipeline completado exitosamente. Pasos ejecutados: {len(executed_steps)}")
        return context


# ==================== CLASES DE SOPORTE (Legacy Refactored) ====================
# (Se mantienen las clases PresupuestoProcessor, InsumosProcessor, etc.
# tal cual estaban, ya que no requieren cambios en sus métodos internos,
# solo son llamadas por los Steps refactorizados)


class PresupuestoProcessor:
    def __init__(self, config: dict, thresholds: ProcessingThresholds, profile: dict):
        # ROBUSTECIDO: Validar parámetros
        if not config:
            raise ValueError("Configuración requerida para PresupuestoProcessor")
        self.config = config
        self.thresholds = thresholds or ProcessingThresholds()
        self.profile = profile or {}
        self.validator = DataValidator()

    def process(self, path: str) -> pd.DataFrame:
        """Procesa archivo de presupuesto con manejo robusto de errores."""
        # ROBUSTECIDO: Validar path al inicio
        if not path:
            logger.error("❌ Ruta de presupuesto vacía")
            return pd.DataFrame()

        try:
            loader_params = self.profile.get("loader_params", {})
            logger.info(f"📥 Cargando presupuesto desde: {path}")
            logger.debug(f"Parámetros de carga: {loader_params}")

            load_result = load_data(path, **loader_params)

            # ROBUSTECIDO: Verificar resultado de carga exhaustivamente
            if load_result is None:
                logger.error("❌ load_data retornó None")
                return pd.DataFrame()

            if not hasattr(load_result, 'status') or not hasattr(load_result, 'data'):
                logger.error("❌ Estructura de load_result inválida")
                return pd.DataFrame()

            # ROBUSTECIDO: Comparación segura de status
            status_value = getattr(load_result.status, 'value', str(load_result.status))
            if status_value != "SUCCESS":
                error_msg = getattr(load_result, 'error_message', 'Error desconocido')
                logger.error(f"Error cargando presupuesto: {error_msg}")
                return pd.DataFrame()

            df = load_result.data
            if df is None or (isinstance(df, pd.DataFrame) and df.empty):
                logger.warning("⚠️ Archivo de presupuesto cargado está vacío")
                return pd.DataFrame()

            # Pipeline de procesamiento con validaciones intermedias
            df_clean = self._clean_phantom_rows(df)
            if df_clean.empty:
                logger.warning("⚠️ DataFrame vacío después de limpiar filas fantasma")
                return pd.DataFrame()

            df_renamed = self._rename_columns(df_clean)
            if not self._validate_required_columns(df_renamed):
                logger.error("❌ Validación de columnas requeridas falló")
                return pd.DataFrame()

            df_converted = self._clean_and_convert_data(df_renamed)
            if df_converted.empty:
                logger.warning("⚠️ DataFrame vacío después de conversión de datos")
                return pd.DataFrame()

            df_final = self._remove_duplicates(df_converted)

            # ROBUSTECIDO: Selección segura de columnas
            final_cols = [
                ColumnNames.CODIGO_APU,
                ColumnNames.DESCRIPCION_APU,
                ColumnNames.CANTIDAD_PRESUPUESTO,
            ]
            available_cols = [col for col in final_cols if col in df_final.columns]

            if ColumnNames.CODIGO_APU not in available_cols:
                logger.error(f"❌ Columna crítica '{ColumnNames.CODIGO_APU}' no disponible")
                return pd.DataFrame()

            result = df_final[available_cols].copy()
            logger.info(f"✅ Presupuesto procesado: {len(result)} filas")
            return result

        except Exception as e:
            logger.error(f"❌ Error fatal procesando presupuesto: {e}", exc_info=True)
            return pd.DataFrame()

    def _clean_phantom_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina filas completamente vacías o con solo valores nulos."""
        if df is None:
            return pd.DataFrame()

        if not isinstance(df, pd.DataFrame):
            logger.error(f"❌ _clean_phantom_rows recibió tipo: {type(df).__name__}")
            return pd.DataFrame()

        if df.empty:
            return df

        # ROBUSTECIDO: Usar operaciones vectorizadas en lugar de apply
        initial_rows = len(df)

        # Eliminar filas completamente NA
        df_clean = df.dropna(how="all")

        # ROBUSTECIDO: Vectorizar la detección de filas vacías
        # Convertir a string y verificar si todos los valores son vacíos
        str_df = df_clean.astype(str)
        empty_patterns = {'', 'nan', 'none', 'nat', '<na>'}

        # Crear máscara vectorizada
        is_empty_mask = str_df.apply(
            lambda col: col.str.strip().str.lower().isin(empty_patterns),
            axis=0
        ).all(axis=1)

        df_clean = df_clean[~is_empty_mask]

        removed_rows = initial_rows - len(df_clean)
        if removed_rows > 0:
            logger.debug(f"Filas fantasma eliminadas: {removed_rows}")

        return df_clean

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        column_map = self.config.get("presupuesto_column_map", {})
        return find_and_rename_columns(df, column_map)

    def _validate_required_columns(self, df: pd.DataFrame) -> bool:
        return self.validator.validate_required_columns(
            df, [ColumnNames.CODIGO_APU], "presupuesto"
        )[0]

    def _clean_and_convert_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Limpia y convierte datos con validaciones robustas."""
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()  # ROBUSTECIDO: Evitar modificar DataFrame original

        # Limpiar código APU
        clean_code_params = self.config.get("clean_apu_code_params", {}).get(
            "presupuesto_item", {}
        )

        # ROBUSTECIDO: Manejar columna CODIGO_APU de forma segura
        if ColumnNames.CODIGO_APU not in df.columns:
            logger.error(f"❌ Columna '{ColumnNames.CODIGO_APU}' no encontrada")
            return pd.DataFrame()

        try:
            df[ColumnNames.CODIGO_APU] = (
                df[ColumnNames.CODIGO_APU]
                .fillna('')
                .astype(str)
                .apply(lambda c: clean_apu_code(c, **clean_code_params))
            )
        except Exception as e:
            logger.error(f"❌ Error limpiando códigos APU: {e}")
            return pd.DataFrame()

        # ROBUSTECIDO: Filtrar códigos inválidos de forma más clara
        invalid_codes = {'', 'nan', 'none', 'null'}
        mask_valid = (
            df[ColumnNames.CODIGO_APU].notna() &
            ~df[ColumnNames.CODIGO_APU].str.strip().str.lower().isin(invalid_codes)
        )

        df = df[mask_valid].copy()

        if df.empty:
            logger.warning("⚠️ No quedaron registros con códigos APU válidos")
            return df

        # ROBUSTECIDO: Procesar cantidad con validaciones
        if ColumnNames.CANTIDAD_PRESUPUESTO in df.columns:
            try:
                qty = df[ColumnNames.CANTIDAD_PRESUPUESTO].astype(str).str.strip()
                # Normalizar separadores decimales
                qty = qty.str.replace(",", ".", regex=False)
                # Remover caracteres no numéricos excepto punto y signo negativo
                qty = qty.str.replace(r"[^\d.\-]", "", regex=True)
                # Manejar múltiples puntos decimales (mantener solo el último)
                qty = qty.str.replace(r"\.(?=.*\.)", "", regex=True)

                df[ColumnNames.CANTIDAD_PRESUPUESTO] = pd.to_numeric(
                    qty, errors="coerce"
                ).fillna(0)

                # ROBUSTECIDO: Validar rangos
                max_qty = self.thresholds.max_quantity
                mask_invalid_qty = df[ColumnNames.CANTIDAD_PRESUPUESTO] > max_qty
                if mask_invalid_qty.any():
                    count = mask_invalid_qty.sum()
                    logger.warning(f"⚠️ {count} cantidades exceden máximo ({max_qty}), se limitarán")
                    df.loc[mask_invalid_qty, ColumnNames.CANTIDAD_PRESUPUESTO] = max_qty

            except Exception as e:
                logger.error(f"❌ Error procesando cantidades: {e}")
                df[ColumnNames.CANTIDAD_PRESUPUESTO] = 0

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.validator.detect_and_log_duplicates(
            df, [ColumnNames.CODIGO_APU], "presupuesto"
        )


class InsumosProcessor:
    def __init__(self, thresholds: ProcessingThresholds, profile: dict):
        self.thresholds = thresholds
        self.profile = profile
        self.validator = DataValidator()

    def process(self, file_path: str) -> pd.DataFrame:
        try:
            records = self._parse_file(file_path)
            if not records:
                return pd.DataFrame()
            df = pd.DataFrame(records)
            df = self._rename_and_select_columns(df)
            df = self._convert_and_normalize(df)
            df = self._remove_duplicates(df)
            return df
        except Exception:
            return pd.DataFrame()

    def _parse_file(self, file_path: str) -> List[Dict]:
        encoding = self.profile.get("encoding", "latin1")
        try:
            with open(file_path, "r", encoding=encoding) as f:
                lines = f.readlines()
        except Exception:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

        records = []
        current_group = None
        header = None

        for line in lines:
            clean_line = line.strip()
            if not clean_line:
                continue

            parts = [p.strip().replace('"', "") for p in clean_line.split(";")]
            if not any(parts):
                continue

            # Detección de Grupo
            # El formato observado es "G;MATERIALES;;..." o similar.
            # A veces puede ser "G1;MATERIALES"
            first_col = parts[0].strip().upper()
            if first_col.startswith("G") and len(parts) > 1:
                # Intentar obtener el nombre del grupo de la segunda columna
                candidate_group = parts[1].strip()
                if candidate_group:
                    logger.debug(
                        f"🔍 Candidato a grupo detectado: '{parts[0]}' -> Grupo: '{candidate_group}'"
                    )
                    current_group = candidate_group
                    logger.info(f"📂 Grupo detectado: {current_group}")
                    header = None  # Reset header al cambiar de grupo
                    continue

            # Detección de Encabezado
            # Buscamos coincidencia parcial de columnas clave
            if (
                "CODIGO" in first_col
                and len(parts) >= 2
                and "DESCRIPCION" in parts[1].upper()
            ):
                header = parts
                logger.info(f"📋 Encabezado detectado para grupo {current_group}: {header}")
                continue

            # Procesamiento de Datos
            if header and current_group:
                # Validar que la línea parezca datos (ej. código no vacío)
                if not parts[0]:
                    continue

                record = {ColumnNames.GRUPO_INSUMO: current_group}

                # Mapeo por índice basado en el encabezado detectado
                for i, col_name in enumerate(header):
                    if i < len(parts):
                        # Normalizar nombres de columnas para el dict interno
                        clean_col = col_name.upper().replace(".", "").strip()
                        if "DESCRIPCION" in clean_col:
                            record["DESCRIPCION"] = parts[i]
                        elif "VR" in clean_col and "UNIT" in clean_col:
                            record["VR. UNIT."] = parts[i]
                        elif "UND" in clean_col:
                            record["UND"] = parts[i]
                        elif "CODIGO" in clean_col:
                            record["CODIGO"] = parts[i]
                        elif "CANT" in clean_col:
                            record["CANTIDAD"] = parts[i]

                # Solo agregamos si tenemos al menos descripción y valor
                if "DESCRIPCION" in record:
                    records.append(record)

        logger.info(f"✅ Total insumos extraídos: {len(records)}")
        return records

    def _rename_and_select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(
            columns={
                "DESCRIPCION": ColumnNames.DESCRIPCION_INSUMO,
                "VR. UNIT.": ColumnNames.VR_UNITARIO_INSUMO,
                "CODIGO": "CODIGO",  # Mantener CODIGO si existe
                "CANTIDAD": "CANTIDAD",
            }
        )
        # Asegurar que CODIGO y CANTIDAD existan si no vinieron en el archivo
        if "CODIGO" not in df.columns:
            df["CODIGO"] = None
        if "CANTIDAD" not in df.columns:
            df["CANTIDAD"] = 0

        return df[
            [
                ColumnNames.GRUPO_INSUMO,
                ColumnNames.DESCRIPCION_INSUMO,
                ColumnNames.VR_UNITARIO_INSUMO,
                "CODIGO",
                "CANTIDAD",
            ]
        ]

    def _convert_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df[ColumnNames.VR_UNITARIO_INSUMO] = pd.to_numeric(
            df[ColumnNames.VR_UNITARIO_INSUMO].astype(str).str.replace(",", "."),
            errors="coerce",
        )
        df[ColumnNames.DESCRIPCION_INSUMO_NORM] = normalize_text_series(
            df[ColumnNames.DESCRIPCION_INSUMO]
        )
        return df.dropna(subset=[ColumnNames.DESCRIPCION_INSUMO])

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.sort_values(
            ColumnNames.VR_UNITARIO_INSUMO, ascending=False
        ).drop_duplicates(subset=[ColumnNames.DESCRIPCION_INSUMO_NORM], keep="first")


class DataMerger:
    def __init__(self, thresholds: ProcessingThresholds):
        self.thresholds = thresholds or ProcessingThresholds()

    def merge_apus_with_insumos(
        self, df_apus: pd.DataFrame, df_insumos: pd.DataFrame
    ) -> pd.DataFrame:
        """Fusiona APUs con insumos de forma robusta."""
        # ROBUSTECIDO: Validaciones exhaustivas
        if df_apus is None or not isinstance(df_apus, pd.DataFrame):
            logger.error("❌ df_apus inválido para merge")
            return pd.DataFrame()

        if df_insumos is None or not isinstance(df_insumos, pd.DataFrame):
            logger.error("❌ df_insumos inválido para merge")
            return df_apus.copy()  # Retornar APUs sin enriquecer

        if df_apus.empty:
            logger.warning("⚠️ df_apus vacío")
            return pd.DataFrame()

        if df_insumos.empty:
            logger.warning("⚠️ df_insumos vacío, retornando APUs sin enriquecer")
            return df_apus.copy()

        df_apus = df_apus.copy()
        df_insumos = df_insumos.copy()

        # ROBUSTECIDO: Verificar columnas necesarias antes de merge
        if ColumnNames.DESCRIPCION_INSUMO not in df_apus.columns:
            logger.error(f"❌ Columna '{ColumnNames.DESCRIPCION_INSUMO}' no existe en df_apus")
            return df_apus

        if ColumnNames.DESCRIPCION_INSUMO_NORM not in df_insumos.columns:
            logger.warning("⚠️ Creando columna normalizada en df_insumos")
            df_insumos[ColumnNames.DESCRIPCION_INSUMO_NORM] = normalize_text_series(
                df_insumos.get(ColumnNames.DESCRIPCION_INSUMO, pd.Series(dtype=str))
            )

        # Crear columna normalizada en APUs si no existe
        if ColumnNames.NORMALIZED_DESC not in df_apus.columns:
            df_apus[ColumnNames.NORMALIZED_DESC] = normalize_text_series(
                df_apus[ColumnNames.DESCRIPCION_INSUMO]
            )

        try:
            # ROBUSTECIDO: Merge con manejo de errores
            df_merged = pd.merge(
                df_apus,
                df_insumos,
                left_on=ColumnNames.NORMALIZED_DESC,
                right_on=ColumnNames.DESCRIPCION_INSUMO_NORM,
                how="left",
                suffixes=("_apu", "_insumo"),
                validate="m:1",
            )
        except pd.errors.MergeError as e:
            logger.warning(f"⚠️ Error de validación en merge (m:1): {e}. Reintentando sin validación.")
            df_merged = pd.merge(
                df_apus,
                df_insumos,
                left_on=ColumnNames.NORMALIZED_DESC,
                right_on=ColumnNames.DESCRIPCION_INSUMO_NORM,
                how="left",
                suffixes=("_apu", "_insumo"),
            )
        except Exception as e:
            logger.error(f"❌ Error en merge APUs-Insumos: {e}")
            return df_apus

        # ROBUSTECIDO: Consolidar columnas de forma segura
        if ColumnNames.CATEGORIA in df_merged.columns:
            df_merged[ColumnNames.TIPO_INSUMO] = df_merged[ColumnNames.CATEGORIA]
        elif ColumnNames.TIPO_INSUMO not in df_merged.columns:
            df_merged[ColumnNames.TIPO_INSUMO] = InsumoTypes.OTRO

        # Consolidar descripción
        desc_col_apu = f"{ColumnNames.DESCRIPCION_INSUMO}_apu"
        desc_col_insumo = f"{ColumnNames.DESCRIPCION_INSUMO}_insumo"

        if ColumnNames.DESCRIPCION_INSUMO in df_merged.columns:
            pass  # Ya existe
        elif desc_col_insumo in df_merged.columns:
            df_merged[ColumnNames.DESCRIPCION_INSUMO] = df_merged[desc_col_insumo]
        elif desc_col_apu in df_merged.columns:
            df_merged[ColumnNames.DESCRIPCION_INSUMO] = df_merged[desc_col_apu]

        # Consolidar unidad
        unidad_col_apu = f"{ColumnNames.UNIDAD_INSUMO}_apu"
        if unidad_col_apu in df_merged.columns:
            if ColumnNames.UNIDAD_INSUMO not in df_merged.columns:
                df_merged[ColumnNames.UNIDAD_INSUMO] = df_merged[unidad_col_apu]
            df_merged.drop(columns=[unidad_col_apu], errors='ignore', inplace=True)

        logger.info(f"✅ Merge APUs-Insumos completado: {len(df_merged)} filas")
        return df_merged

    def merge_with_presupuesto(
        self, df_presupuesto: pd.DataFrame, df_apu_costos: pd.DataFrame
    ) -> pd.DataFrame:
        """Fusiona presupuesto con costos APU de forma robusta."""
        # ROBUSTECIDO: Validaciones
        if df_presupuesto is None or df_presupuesto.empty:
            logger.error("❌ df_presupuesto vacío o None")
            return pd.DataFrame()

        if df_apu_costos is None or df_apu_costos.empty:
            logger.warning("⚠️ df_apu_costos vacío, retornando presupuesto sin costos")
            return df_presupuesto.copy()

        # ROBUSTECIDO: Verificar columna de join
        if ColumnNames.CODIGO_APU not in df_presupuesto.columns:
            logger.error(f"❌ '{ColumnNames.CODIGO_APU}' no existe en presupuesto")
            return df_presupuesto.copy()

        if ColumnNames.CODIGO_APU not in df_apu_costos.columns:
            logger.error(f"❌ '{ColumnNames.CODIGO_APU}' no existe en apu_costos")
            return df_presupuesto.copy()

        try:
            df_merged = pd.merge(
                df_presupuesto,
                df_apu_costos,
                on=ColumnNames.CODIGO_APU,
                how="left",
                validate="1:1",
            )
            logger.info(f"✅ Merge con presupuesto completado: {len(df_merged)} filas")
            return df_merged

        except pd.errors.MergeError as e:
            logger.warning(f"⚠️ Duplicados detectados en merge 1:1: {e}")

            # ROBUSTECIDO: Diagnóstico y corrección
            dupes_pres = df_presupuesto[
                df_presupuesto[ColumnNames.CODIGO_APU].duplicated(keep=False)
            ][ColumnNames.CODIGO_APU].unique()

            dupes_costos = df_apu_costos[
                df_apu_costos[ColumnNames.CODIGO_APU].duplicated(keep=False)
            ][ColumnNames.CODIGO_APU].unique()

            if len(dupes_pres) > 0:
                logger.warning(f"Duplicados en presupuesto: {dupes_pres[:5].tolist()}")
            if len(dupes_costos) > 0:
                logger.warning(f"Duplicados en costos: {dupes_costos[:5].tolist()}")

            # Intentar merge sin validación
            df_merged = pd.merge(
                df_presupuesto,
                df_apu_costos,
                on=ColumnNames.CODIGO_APU,
                how="left",
            )
            logger.info(f"✅ Merge sin validación completado: {len(df_merged)} filas")
            return df_merged

        except Exception as e:
            logger.error(f"❌ Error en merge con presupuesto: {e}")
            raise


class APUCostCalculator:
    def __init__(self, config: dict, thresholds: ProcessingThresholds):
        self.config = config
        self.thresholds = thresholds

    def calculate(
        self, df_merged: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_apu_costos = self._aggregate_costs(df_merged)
        df_apu_costos = self._calculate_unit_values(df_apu_costos)
        df_apu_costos = self._classify_apus(df_apu_costos)
        df_tiempo = self._calculate_time(df_merged)
        df_rendimiento = self._calculate_performance(df_merged)
        return df_apu_costos, df_tiempo, df_rendimiento

    def _aggregate_costs(self, df_merged: pd.DataFrame) -> pd.DataFrame:
        df = df_merged.copy()
        df[ColumnNames.COSTO_INSUMO_EN_APU] = pd.to_numeric(
            df[ColumnNames.COSTO_INSUMO_EN_APU], errors="coerce"
        ).fillna(0)
        df[ColumnNames.TIPO_INSUMO] = (
            df[ColumnNames.TIPO_INSUMO].astype(str).str.strip().str.upper()
        )

        costs = (
            df.groupby([ColumnNames.CODIGO_APU, ColumnNames.TIPO_INSUMO])[
                ColumnNames.COSTO_INSUMO_EN_APU
            ]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )

        mapping = {
            InsumoTypes.SUMINISTRO: ColumnNames.MATERIALES,
            InsumoTypes.MANO_DE_OBRA: ColumnNames.MANO_DE_OBRA,
            InsumoTypes.EQUIPO: ColumnNames.EQUIPO,
            InsumoTypes.TRANSPORTE: ColumnNames.OTROS,
            InsumoTypes.OTRO: ColumnNames.OTROS,
        }

        # Mapear columnas existentes
        for col in costs.columns:
            if col in mapping:
                target = mapping[col]
                if target in costs.columns:
                    costs[target] += costs[col]
                else:
                    costs[target] = costs[col]
                if target != col:
                    costs.drop(columns=[col], inplace=True)

        for req in [
            ColumnNames.MATERIALES,
            ColumnNames.MANO_DE_OBRA,
            ColumnNames.EQUIPO,
            ColumnNames.OTROS,
        ]:
            if req not in costs.columns:
                costs[req] = 0.0

        return costs

    def _calculate_unit_values(self, df: pd.DataFrame) -> pd.DataFrame:
        df[ColumnNames.VALOR_SUMINISTRO_UN] = df[ColumnNames.MATERIALES]
        df[ColumnNames.VALOR_INSTALACION_UN] = (
            df[ColumnNames.MANO_DE_OBRA] + df[ColumnNames.EQUIPO]
        )
        df[ColumnNames.VALOR_CONSTRUCCION_UN] = (
            df[ColumnNames.VALOR_SUMINISTRO_UN]
            + df[ColumnNames.VALOR_INSTALACION_UN]
            + df[ColumnNames.OTROS]
        )
        return df

    def _classify_apus(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clasifica los APUs según la proporción de costos.
        ROBUSTECIDO: Eliminado uso de eval() por seguridad.
        """
        if df is None or not isinstance(df, pd.DataFrame):
            logger.error("❌ DataFrame inválido para clasificación de APUs")
            return pd.DataFrame()

        if df.empty:
            return df

        df = df.copy()  # No modificar original

        # Asegurar columnas de costos
        cost_columns = [
            ColumnNames.VALOR_CONSTRUCCION_UN,
            ColumnNames.VALOR_SUMINISTRO_UN,
            ColumnNames.VALOR_INSTALACION_UN,
        ]

        for col in cost_columns:
            if col not in df.columns:
                df[col] = 0.0
            else:
                # ROBUSTECIDO: Asegurar tipo numérico
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

        # Calcular porcentajes de forma segura
        total_cost = df[ColumnNames.VALOR_CONSTRUCCION_UN].copy()
        # Evitar división por cero
        total_cost = total_cost.replace(0, np.nan)

        df["_pct_materiales"] = (
            df[ColumnNames.VALOR_SUMINISTRO_UN] / total_cost * 100
        ).fillna(0)

        df["_pct_mo_eq"] = (
            df[ColumnNames.VALOR_INSTALACION_UN] / total_cost * 100
        ).fillna(0)

        # Default
        df[ColumnNames.TIPO_APU] = APUTypes.INDEFINIDO

        # ROBUSTECIDO: Aplicar reglas de forma segura sin eval()
        rules = self.config.get("apu_classification_rules", [])

        if not rules:
            logger.warning("⚠️ No hay reglas de clasificación, usando lógica por defecto")
            # Lógica por defecto basada en thresholds
            th = self.thresholds

            # Instalación: alto % mano de obra
            mask_instalacion = df["_pct_mo_eq"] >= th.instalacion_mo_threshold
            df.loc[mask_instalacion, ColumnNames.TIPO_APU] = APUTypes.INSTALACION

            # Suministro: alto % materiales, bajo % MO
            mask_suministro = (
                (df["_pct_materiales"] >= th.suministro_mat_threshold) &
                (df["_pct_mo_eq"] <= th.suministro_mo_max)
            )
            df.loc[mask_suministro, ColumnNames.TIPO_APU] = APUTypes.SUMINISTRO

            # Suministro Prefabricado
            mask_prefab = (
                (df["_pct_materiales"] >= th.prefabricado_mat_threshold) &
                (df["_pct_mo_eq"] >= th.prefabricado_mo_min) &
                (df["_pct_mo_eq"] < th.instalacion_mo_threshold)
            )
            df.loc[mask_prefab, ColumnNames.TIPO_APU] = APUTypes.SUMINISTRO_PREFABRICADO

            # Obra Completa: resto con costos válidos
            mask_obra_completa = (
                (df[ColumnNames.TIPO_APU] == APUTypes.INDEFINIDO) &
                (total_cost.notna()) &
                (total_cost > 0)
            )
            df.loc[mask_obra_completa, ColumnNames.TIPO_APU] = APUTypes.OBRA_COMPLETA
        else:
            # ROBUSTECIDO: Procesar reglas de configuración de forma segura
            for rule in rules:
                try:
                    rule_type = rule.get("type")
                    if not rule_type:
                        continue

                    # Construir máscara según operadores definidos explícitamente
                    conditions = rule.get("conditions", [])
                    if not conditions:
                        continue

                    mask = pd.Series([True] * len(df), index=df.index)

                    for cond in conditions:
                        field = cond.get("field")
                        operator = cond.get("operator")
                        value = cond.get("value")

                        if not all([field, operator, value is not None]):
                            continue

                        # Mapear campos a columnas internas
                        field_map = {
                            "porcentaje_materiales": "_pct_materiales",
                            "porcentaje_mo_eq": "_pct_mo_eq",
                            "pct_materiales": "_pct_materiales",
                            "pct_mo_eq": "_pct_mo_eq",
                        }
                        col_name = field_map.get(field, field)

                        if col_name not in df.columns:
                            logger.warning(f"⚠️ Campo '{field}' no existe en DataFrame")
                            continue

                        # ROBUSTECIDO: Operadores explícitos sin eval
                        if operator == ">=":
                            mask &= df[col_name] >= value
                        elif operator == ">":
                            mask &= df[col_name] > value
                        elif operator == "<=":
                            mask &= df[col_name] <= value
                        elif operator == "<":
                            mask &= df[col_name] < value
                        elif operator == "==":
                            mask &= df[col_name] == value
                        elif operator == "!=":
                            mask &= df[col_name] != value
                        else:
                            logger.warning(f"⚠️ Operador desconocido: {operator}")

                    df.loc[mask, ColumnNames.TIPO_APU] = rule_type

                except Exception as e:
                    logger.error(f"❌ Error aplicando regla {rule}: {e}")

        # Limpieza de columnas temporales
        df.drop(columns=["_pct_materiales", "_pct_mo_eq"], inplace=True, errors="ignore")

        return df

    def _calculate_time(self, df: pd.DataFrame) -> pd.DataFrame:
        return (
            df[df[ColumnNames.TIPO_INSUMO] == InsumoTypes.MANO_DE_OBRA]
            .groupby(ColumnNames.CODIGO_APU)[ColumnNames.CANTIDAD_APU]
            .sum()
            .reset_index()
            .rename(columns={ColumnNames.CANTIDAD_APU: ColumnNames.TIEMPO_INSTALACION})
        )

    def _calculate_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        if ColumnNames.RENDIMIENTO in df.columns:
            return (
                df[df[ColumnNames.TIPO_INSUMO] == InsumoTypes.MANO_DE_OBRA]
                .groupby(ColumnNames.CODIGO_APU)[ColumnNames.RENDIMIENTO]
                .sum()
                .reset_index()
                .rename(columns={ColumnNames.RENDIMIENTO: ColumnNames.RENDIMIENTO_DIA})
            )
        return pd.DataFrame()


# ==================== FUNCIONES AUXILIARES ====================


def calculate_insumo_costs(
    df: pd.DataFrame, thresholds: ProcessingThresholds
) -> pd.DataFrame:
    """Calcula costos de insumos de forma robusta."""
    # ROBUSTECIDO: Validaciones de entrada
    if df is None or not isinstance(df, pd.DataFrame):
        logger.error("❌ DataFrame inválido para cálculo de costos")
        return pd.DataFrame()

    if df.empty:
        return df

    df = df.copy()  # No modificar original

    # ROBUSTECIDO: Asegurar existencia de columnas con tipos correctos
    numeric_columns = [
        ColumnNames.CANTIDAD_APU,
        ColumnNames.VR_UNITARIO_INSUMO,
        ColumnNames.VALOR_TOTAL_APU,
    ]

    for col in numeric_columns:
        if col not in df.columns:
            df[col] = 0.0
        else:
            # Convertir a numérico de forma segura
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    # ROBUSTECIDO: Validar rangos antes de calcular
    max_cost = thresholds.max_cost_per_item
    max_qty = thresholds.max_quantity

    # Limitar valores extremos
    df[ColumnNames.CANTIDAD_APU] = df[ColumnNames.CANTIDAD_APU].clip(0, max_qty)
    df[ColumnNames.VR_UNITARIO_INSUMO] = df[ColumnNames.VR_UNITARIO_INSUMO].clip(0, max_cost)

    # ROBUSTECIDO: Calcular costo con manejo explícito de casos
    tiene_precio = (
        df[ColumnNames.VR_UNITARIO_INSUMO].notna() &
        (df[ColumnNames.VR_UNITARIO_INSUMO] > 0)
    )

    # Calcular costo: cantidad * precio unitario si hay precio, sino usar valor total
    df[ColumnNames.COSTO_INSUMO_EN_APU] = np.where(
        tiene_precio,
        df[ColumnNames.CANTIDAD_APU] * df[ColumnNames.VR_UNITARIO_INSUMO],
        df[ColumnNames.VALOR_TOTAL_APU]
    )

    # ROBUSTECIDO: Validar resultado y limitar
    df[ColumnNames.COSTO_INSUMO_EN_APU] = (
        pd.to_numeric(df[ColumnNames.COSTO_INSUMO_EN_APU], errors='coerce')
        .fillna(0.0)
        .clip(0, thresholds.max_total_cost)
    )

    # VR_UNITARIO_FINAL
    df[ColumnNames.VR_UNITARIO_FINAL] = (
        df[ColumnNames.VR_UNITARIO_INSUMO]
        .fillna(0.0)
        .clip(0, max_cost)
    )

    # ROBUSTECIDO: Log de estadísticas para monitoreo
    total_costo = df[ColumnNames.COSTO_INSUMO_EN_APU].sum()
    registros_sin_costo = (df[ColumnNames.COSTO_INSUMO_EN_APU] == 0).sum()

    if registros_sin_costo > 0:
        pct = (registros_sin_costo / len(df)) * 100
        logger.warning(f"⚠️ {registros_sin_costo} registros ({pct:.1f}%) sin costo calculado")

    logger.debug(f"Costo total calculado: {total_costo:,.2f}")

    return df


def group_and_split_description(df: pd.DataFrame) -> pd.DataFrame:
    if ColumnNames.DESCRIPCION_APU in df.columns:
        df[ColumnNames.ORIGINAL_DESCRIPTION] = df[ColumnNames.DESCRIPCION_APU]
    return df


def calculate_total_costs(
    df: pd.DataFrame, thresholds: ProcessingThresholds
) -> pd.DataFrame:
    if ColumnNames.CANTIDAD_PRESUPUESTO in df.columns:
        qty = pd.to_numeric(df[ColumnNames.CANTIDAD_PRESUPUESTO], errors="coerce").fillna(0)
        df[ColumnNames.VALOR_CONSTRUCCION_TOTAL] = (
            df[ColumnNames.VALOR_CONSTRUCCION_UN] * qty
        )
    return df


def build_processed_apus_dataframe(df_costs, df_raw, df_time, df_perf):
    return df_costs.copy()  # Simplificado


def synchronize_data_sources(df_merged, df_final):
    valid_codes = df_final[ColumnNames.CODIGO_APU].unique()
    return df_merged[df_merged[ColumnNames.CODIGO_APU].isin(valid_codes)].copy()


def build_output_dictionary(df_final, df_insumos, df_merged, df_raw, df_proc):
    return {
        "presupuesto": df_final.to_dict("records"),
        "processed_apus": df_proc.to_dict("records"),
        "apus_detail": df_merged.to_dict("records"),
        "insumos": {},
    }


# ==================== ENTRY POINT ====================


def process_all_files(
    presupuesto_path: str,
    apus_path: str,
    insumos_path: str,
    config: dict,
    telemetry: TelemetryContext,
) -> dict:
    """
    Entry point refactorizado para usar PipelineDirector y Telemetry.
    """
    logger.info("🚀 Iniciando procesamiento via PipelineDirector")

    director = PipelineDirector(config, telemetry)

    initial_context = {
        "presupuesto_path": presupuesto_path,
        "apus_path": apus_path,
        "insumos_path": insumos_path,
    }

    try:
        final_context = director.execute(initial_context)
        final_result = final_context.get("final_result", {})

        # Guardado de archivos (lógica simplificada integrada aquí o delegada)
        output_dir = Path(config.get("output_dir", "data"))
        output_files = {
            "processed_apus": output_dir
            / config.get("processed_apus_file", "processed_apus.json"),
            "presupuesto_final": output_dir
            / config.get("presupuesto_final_file", "presupuesto_final.json"),
        }

        _save_output_files(final_result, output_files, config)

        return final_result

    except Exception as e:
        logger.error(f"❌ Error en process_all_files: {e}")
        return {"error": str(e)}


def _save_output_files(
    result: dict,
    output_files: dict,
    config: dict
) -> Dict[str, bool]:
    """
    Guarda archivos de salida de forma robusta.
    Retorna diccionario con estado de cada archivo.
    """
    import json

    # ROBUSTECIDO: Validar entradas
    if not result or not isinstance(result, dict):
        logger.error("❌ Resultado vacío o inválido para guardar")
        return {}

    if not output_files or not isinstance(output_files, dict):
        logger.error("❌ output_files vacío o inválido")
        return {}

    save_status = {}

    for name, path in output_files.items():
        try:
            # ROBUSTECIDO: Verificar que hay datos para guardar
            if name not in result:
                logger.debug(f"Clave '{name}' no encontrada en resultado, saltando")
                save_status[name] = False
                continue

            data = result[name]
            if not data:
                logger.debug(f"Datos vacíos para '{name}', saltando")
                save_status[name] = False
                continue

            # ROBUSTECIDO: Crear directorio si no existe
            path = Path(path)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except PermissionError as e:
                logger.error(f"❌ Sin permisos para crear directorio: {path.parent}")
                save_status[name] = False
                continue
            except Exception as e:
                logger.error(f"❌ Error creando directorio {path.parent}: {e}")
                save_status[name] = False
                continue

            # ROBUSTECIDO: Sanitizar datos antes de serializar
            try:
                sanitized_data = sanitize_for_json(data)
            except Exception as e:
                logger.error(f"❌ Error sanitizando datos para '{name}': {e}")
                save_status[name] = False
                continue

            # ROBUSTECIDO: Escritura atómica con archivo temporal
            temp_path = path.with_suffix('.tmp')

            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(
                        sanitized_data,
                        f,
                        indent=2,
                        ensure_ascii=False,
                        default=str  # Fallback para tipos no serializables
                    )

                # Mover archivo temporal a destino final
                temp_path.replace(path)
                logger.info(f"✅ Archivo guardado: {path}")
                save_status[name] = True

            except Exception as e:
                logger.error(f"❌ Error escribiendo '{name}': {e}")
                # Limpiar archivo temporal si existe
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except:
                        pass
                save_status[name] = False

        except Exception as e:
            logger.error(f"❌ Error inesperado guardando '{name}': {e}")
            save_status[name] = False

    # ROBUSTECIDO: Resumen de guardado
    successful = sum(1 for v in save_status.values() if v)
    total = len(save_status)
    logger.info(f"📁 Archivos guardados: {successful}/{total}")

    return save_status
