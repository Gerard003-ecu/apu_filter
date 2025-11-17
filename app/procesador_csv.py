import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from scripts.diagnose_apus_file import APUFileDiagnostic

from .apu_processor import APUProcessor
from .data_loader import load_data
from .data_validator import validate_and_clean_data
from .report_parser_crudo import ParserConfig, ReportParserCrudo
from .utils import (
    clean_apu_code,
    find_and_rename_columns,
    normalize_text_series,
)

logger = logging.getLogger(__name__)

# ==================== CONSTANTES ====================


class ColumnNames:
    """Nombres de columnas estandarizados para evitar strings mágicos."""

    # Columnas de APU
    CODIGO_APU = "CODIGO_APU"
    DESCRIPCION_APU = "DESCRIPCION_APU"
    DESCRIPCION_SECUNDARIA = "descripcion_secundaria"
    ORIGINAL_DESCRIPTION = "original_description"
    UNIDAD_APU = "UNIDAD_APU"
    CANTIDAD_APU = "CANTIDAD_APU"

    # Columnas de Presupuesto
    CANTIDAD_PRESUPUESTO = "CANTIDAD_PRESUPUESTO"

    # Columnas de Insumos
    GRUPO_INSUMO = "GRUPO_INSUMO"
    DESCRIPCION_INSUMO = "DESCRIPCION_INSUMO"
    DESCRIPCION_INSUMO_NORM = "DESCRIPCION_INSUMO_NORM"
    VR_UNITARIO_INSUMO = "VR_UNITARIO_INSUMO"
    UNIDAD_INSUMO = "UNIDAD_INSUMO"
    NORMALIZED_DESC = "NORMALIZED_DESC"

    # Columnas de Costos
    COSTO_INSUMO_EN_APU = "COSTO_INSUMO_EN_APU"
    VR_UNITARIO_FINAL = "VR_UNITARIO_FINAL"
    VALOR_TOTAL_APU = "VALOR_TOTAL_APU"
    PRECIO_UNIT_APU = "PRECIO_UNIT_APU"

    # Columnas de Resultados
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

    # Columnas de Tiempo
    TIEMPO_INSTALACION = "TIEMPO_INSTALACION"
    RENDIMIENTO = "RENDIMIENTO"
    RENDIMIENTO_DIA = "RENDIMIENTO_DIA"


class InsumoTypes:
    """Tipos de insumos reconocidos."""

    SUMINISTRO = "SUMINISTRO"
    MANO_DE_OBRA = "MANO_DE_OBRA"
    EQUIPO = "EQUIPO"
    TRANSPORTE = "TRANSPORTE"
    OTRO = "OTRO"


class APUTypes:
    """Tipos de APU clasificados."""

    INSTALACION = "Instalación"
    SUMINISTRO = "Suministro"
    SUMINISTRO_PREFABRICADO = "Suministro (Pre-fabricado)"
    OBRA_COMPLETA = "Obra Completa"
    INDEFINIDO = "Indefinido"


class ProcessingStep(ABC):
    @abstractmethod
    def execute(self, context: dict) -> dict:
        pass


class ProcessingPipeline:
    def __init__(self, steps: list[ProcessingStep]):
        self.steps = steps

    def run(self, initial_context: dict) -> dict:
        context = initial_context
        for step in self.steps:
            context = step.execute(context)
        return context


@dataclass
class ProcessingThresholds:
    """Umbrales configurables para validaciones y clasificación."""

    # Detección de outliers
    outlier_std_multiplier: float = 3.0

    # Límites de costos anómalos
    max_quantity: float = 1e6
    max_cost_per_item: float = 1e9
    max_total_cost: float = 1e11

    # Clasificación de APU (porcentajes)
    instalacion_mo_threshold: float = 75.0
    suministro_mat_threshold: float = 75.0
    suministro_mo_max: float = 15.0
    prefabricado_mat_threshold: float = 65.0
    prefabricado_mo_min: float = 15.0

    # Número máximo de filas para búsqueda de encabezado
    max_header_search_rows: int = 10


@dataclass
class ProcessingResult:
    """Resultado estandarizado del procesamiento."""

    success: bool
    data: Optional[Dict] = None
    error: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


# ==================== VALIDADORES ====================


class DataValidator:
    """Validador centralizado para verificación de integridad de datos."""

    @staticmethod
    def validate_dataframe_not_empty(
        df: pd.DataFrame, name: str
    ) -> Tuple[bool, Optional[str]]:
        """Valida que un DataFrame no esté vacío."""
        if df is None or df.empty:
            error_msg = f"DataFrame '{name}' está vacío o es None"
            logger.error(f"❌ {error_msg}")
            return False, error_msg
        return True, None

    @staticmethod
    def validate_required_columns(
        df: pd.DataFrame, required_cols: List[str], df_name: str
    ) -> Tuple[bool, Optional[str]]:
        """Valida que existan las columnas requeridas."""
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            error_msg = f"Faltan columnas requeridas en '{df_name}': {missing_cols}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg
        return True, None

    @staticmethod
    def detect_and_log_duplicates(
        df: pd.DataFrame, subset_cols: List[str], df_name: str, keep: str = "first"
    ) -> pd.DataFrame:
        """Detecta, registra y elimina duplicados."""
        duplicates = df[df.duplicated(subset=subset_cols, keep=False)]

        if not duplicates.empty:
            unique_dupes = duplicates[subset_cols[0]].unique()
            logger.warning(
                f"⚠️  Se encontraron {len(duplicates)} filas duplicadas en '{df_name}' "
                f"por {subset_cols}. Se conservará: '{keep}'"
            )
            logger.debug(f"Valores duplicados en '{df_name}': {unique_dupes.tolist()[:10]}")

            df_clean = df.drop_duplicates(subset=subset_cols, keep=keep)
            logger.info(f"✅ Duplicados eliminados: {len(df)} -> {len(df_clean)} filas")
            return df_clean

        return df

    @staticmethod
    def validate_numeric_range(
        series: pd.Series, column_name: str, max_value: float, min_value: float = 0
    ) -> bool:
        """Valida que los valores numéricos estén en un rango aceptable."""
        invalid_values = series[(series < min_value) | (series > max_value)]

        if not invalid_values.empty:
            logger.warning(
                f"⚠️  Se encontraron {len(invalid_values)} valores fuera de rango "
                f"en '{column_name}' (rango válido: {min_value} - {max_value})"
            )
            logger.debug(
                f"Valores anómalos en '{column_name}':\n{invalid_values.describe()}"
            )
            return False

        return True


class FileValidator:
    """Validador para archivos de entrada."""

    @staticmethod
    def validate_file_exists(file_path: str, file_type: str) -> Tuple[bool, Optional[str]]:
        """Valida que un archivo exista."""
        path = Path(file_path)

        if not path.exists():
            error_msg = f"Archivo de {file_type} no encontrado: {file_path}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        if not path.is_file():
            error_msg = f"La ruta de {file_type} no es un archivo: {file_path}"
            logger.error(f"❌ {error_msg}")
            return False, error_msg

        logger.debug(f"✅ Archivo de {file_type} validado: {file_path}")
        return True, None


# ==================== PROCESADORES ====================


class LoadDataStep(ProcessingStep):
    def __init__(self, config: dict, thresholds: ProcessingThresholds):
        self.config = config
        self.thresholds = thresholds

    def execute(self, context: dict) -> dict:
        presupuesto_path = context["presupuesto_path"]
        apus_path = context["apus_path"]
        insumos_path = context["insumos_path"]

        # Validaciones de existencia de archivos
        file_validator = FileValidator()
        validations = [
            (presupuesto_path, "presupuesto"),
            (apus_path, "APUs"),
            (insumos_path, "insumos"),
        ]
        for file_path, file_type in validations:
            is_valid, error = file_validator.validate_file_exists(file_path, file_type)
            if not is_valid:
                raise ValueError(error)

        # Leer y usar los perfiles
        file_profiles = self.config.get("file_profiles", {})

        # Cargar Presupuesto con su perfil
        presupuesto_profile = file_profiles.get("presupuesto_default")
        if not presupuesto_profile:
            raise ValueError("No se encontró el perfil 'presupuesto_default' en config.json")
        presupuesto_processor = PresupuestoProcessor(
            self.config, self.thresholds, presupuesto_profile
        )
        df_presupuesto = presupuesto_processor.process(presupuesto_path)

        # Cargar Insumos con su perfil
        insumos_profile = file_profiles.get("insumos_default")
        if not insumos_profile:
            raise ValueError("No se encontró el perfil 'insumos_default' en config.json")
        insumos_processor = InsumosProcessor(self.thresholds, insumos_profile)
        df_insumos = insumos_processor.process(insumos_path)

        # Cargar APUs con su perfil
        apus_profile = file_profiles.get("apus_default")
        if not apus_profile:
            raise ValueError("No se encontró el perfil 'apus_default' en config.json")

        # Paso 1: Ejecutar el "Guardia" para obtener registros crudos y el cache
        config_obj = self.config if isinstance(self.config, ParserConfig) else ParserConfig(**self.config.get('parser_config', {}))
        parser = ReportParserCrudo(apus_path, apus_profile, config_obj)
        raw_records = parser.parse_to_raw()
        parse_cache = parser.get_parse_cache() # Obtener el cache

        # Paso 2: Ejecutar el "Cirujano" pasándole los registros y el cache
        processor = APUProcessor(raw_records, self.config, apus_profile, parse_cache)
        df_apus_raw = processor.process_all()

        data_validator = DataValidator()
        validations = [
            (df_presupuesto, "presupuesto"),
            (df_insumos, "insumos"),
            (df_apus_raw, "APUs"),
        ]
        for df, name in validations:
            is_valid, error = data_validator.validate_dataframe_not_empty(df, name)
            if not is_valid:
                raise ValueError(error)

        context["df_presupuesto"] = df_presupuesto
        context["df_insumos"] = df_insumos
        context["df_apus_raw"] = df_apus_raw
        return context


class MergeDataStep(ProcessingStep):
    def __init__(self, thresholds: ProcessingThresholds):
        self.thresholds = thresholds

    def execute(self, context: dict) -> dict:
        df_apus_raw = context["df_apus_raw"]
        df_insumos = context["df_insumos"]

        merger = DataMerger(self.thresholds)
        df_merged = merger.merge_apus_with_insumos(df_apus_raw, df_insumos)

        context["df_merged"] = df_merged
        return context


class CalculateCostsStep(ProcessingStep):
    def __init__(self, config: dict, thresholds: ProcessingThresholds):
        self.config = config
        self.thresholds = thresholds

    def execute(self, context: dict) -> dict:
        df_merged = context["df_merged"]

        df_merged = calculate_insumo_costs(df_merged, self.thresholds)

        cost_calculator = APUCostCalculator(self.config, self.thresholds)
        df_apu_costos, df_tiempo, df_rendimiento = cost_calculator.calculate(df_merged)

        context["df_merged"] = df_merged
        context["df_apu_costos"] = df_apu_costos
        context["df_tiempo"] = df_tiempo
        context["df_rendimiento"] = df_rendimiento
        return context


class FinalMergeStep(ProcessingStep):
    def __init__(self, thresholds: ProcessingThresholds):
        self.thresholds = thresholds

    def execute(self, context: dict) -> dict:
        df_presupuesto = context["df_presupuesto"]
        df_apu_costos = context["df_apu_costos"]
        df_tiempo = context["df_tiempo"]

        merger = DataMerger(self.thresholds)
        df_final = merger.merge_with_presupuesto(df_presupuesto, df_apu_costos)

        df_final = pd.merge(df_final, df_tiempo, on=ColumnNames.CODIGO_APU, how="left")

        df_final = group_and_split_description(df_final)

        df_final = calculate_total_costs(df_final, self.thresholds)

        context["df_final"] = df_final
        return context


class BuildOutputStep(ProcessingStep):
    def execute(self, context: dict) -> dict:
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
        return context


class PresupuestoProcessor:
    """Procesador especializado para archivos de presupuesto con limpieza robusta de filas fantasma."""

    def __init__(self, config: dict, thresholds: ProcessingThresholds, profile: dict):
        self.config = config
        self.thresholds = thresholds
        self.profile = profile
        self.validator = DataValidator()

    def process(self, path: str) -> pd.DataFrame:
        """Procesa el archivo de presupuesto usando el perfil proporcionado."""
        try:
            # Paso 1: Cargar los datos. `load_data` ya usa el perfil para el encabezado.
            loader_params = self.profile.get("loader_params", {})
            logger.info(f"📥 Cargando presupuesto con perfil: {loader_params}")
            df = load_data(path, **loader_params)

            if df is None or df.empty:
                logger.error("❌ No se pudo leer el archivo de presupuesto o está vacío.")
                return pd.DataFrame()

            logger.info(f"📊 DataFrame inicial: {df.shape[0]} filas, {df.shape[1]} columnas")

            # Paso 2: Limpiar filas fantasma (aún útil para robustez).
            df_clean = self._clean_phantom_rows(df)
            logger.info(f"🧹 Limpieza de filas fantasma: {len(df)} → {len(df_clean)} filas")
            if df_clean.empty:
                logger.error("❌ DataFrame vacío después de la limpieza inicial.")
                return pd.DataFrame()

            # Paso 3: Renombrar columnas.
            df_renamed = self._rename_columns(df_clean)
            if not self._validate_required_columns(df_renamed):
                return pd.DataFrame()

            # Paso 4: Limpiar y convertir datos.
            df_converted = self._clean_and_convert_data(df_renamed)

            # Paso 5: Eliminar duplicados.
            df_final = self._remove_duplicates(df_converted)

            logger.info(f"✅ Presupuesto cargado: {len(df_final)} APUs únicos")

            # Devolver solo las columnas necesarias para el resto del pipeline.
            final_cols = [
                ColumnNames.CODIGO_APU,
                ColumnNames.DESCRIPCION_APU,
                ColumnNames.CANTIDAD_PRESUPUESTO,
            ]
            return df_final[[col for col in final_cols if col in df_final.columns]]

        except Exception as e:
            logger.error(f"❌ Error fatal procesando presupuesto: {e}", exc_info=True)
            return pd.DataFrame()

    def _clean_phantom_rows(self, df: pd.DataFrame, stage: str = "") -> pd.DataFrame:
        """
        Elimina filas fantasma de forma agresiva y robusta.

        Detecta y elimina:
        1. Filas completamente NaN
        2. Filas con solo strings vacíos o espacios
        3. Filas con solo delimitadores (ej: ';;;;;')
        4. Filas sin contenido alfanumérico real

        Args:
            df: DataFrame a limpiar
            stage: Etapa del proceso (para logging)

        Returns:
            DataFrame limpio sin filas fantasma
        """
        if df is None or df.empty:
            logger.warning(f"⚠️ [{stage}] DataFrame vacío o None en entrada")
            return pd.DataFrame()

        rows_before = len(df)
        stage_label = f"[{stage}]" if stage else ""

        # ESTRATEGIA 1: Eliminar filas completamente NaN
        df_clean = df.dropna(how='all')
        if df_clean.empty:
            logger.warning(f"⚠️ {stage_label} DataFrame vacío después de dropna(how='all')")
            return pd.DataFrame()

        # ESTRATEGIA 2: Eliminar filas donde TODOS los valores son vacíos/espacios/NaN
        def is_empty_row(row):
            """Verifica si una fila está completamente vacía."""
            for val in row:
                if pd.notna(val):
                    val_str = str(val).strip()
                    # Si encuentra al menos un valor no vacío, no es fila fantasma
                    if val_str and val_str not in ['', 'nan', 'None', 'NaN', 'NaT']:
                        return False
            return True

        mask_empty = df_clean.apply(is_empty_row, axis=1)
        df_clean = df_clean[~mask_empty]

        if df_clean.empty:
            logger.warning(f"⚠️ {stage_label} DataFrame vacío después de eliminar filas vacías")
            return pd.DataFrame()

        # ESTRATEGIA 3: Eliminar filas que son solo delimitadores o caracteres especiales
        def is_delimiter_only_row(row):
            """Detecta filas que contienen solo delimitadores (ej: ';;;;;')."""
            row_values = [str(val) for val in row if pd.notna(val)]
            if not row_values:
                return True

            # Unir todos los valores y buscar contenido alfanumérico
            combined = ''.join(row_values)
            # Remover todos los caracteres que no sean alfanuméricos
            clean_content = ''.join(c for c in combined if c.isalnum())

            # Si no queda nada alfanumérico, es fila fantasma
            return len(clean_content) == 0

        mask_delimiters = df_clean.apply(is_delimiter_only_row, axis=1)
        df_clean = df_clean[~mask_delimiters]

        if df_clean.empty:
            logger.warning(f"⚠️ {stage_label} DataFrame vacío después de eliminar delimitadores")
            return pd.DataFrame()

        # ESTRATEGIA 4: Validación de contenido mínimo
        def has_minimum_content(row):
            """Verifica que la fila tenga contenido real."""
            for val in row:
                if pd.notna(val):
                    val_str = str(val).strip()
                    # Buscar al menos un carácter alfanumérico
                    if any(c.isalnum() for c in val_str):
                        return True
            return False

        mask_content = df_clean.apply(has_minimum_content, axis=1)
        df_clean = df_clean[mask_content]

        # Reset index para evitar problemas
        df_clean = df_clean.reset_index(drop=True)

        rows_after = len(df_clean)
        removed_count = rows_before - rows_after

        # Logging detallado
        if removed_count > 0:
            logger.info(
                f"🧹 {stage_label} Limpieza de filas fantasma: "
                f"{rows_before} → {rows_after} filas "
                f"(-{removed_count} eliminadas)"
            )

            # Log de muestra de filas eliminadas (solo primeras 3)
            if removed_count > 0:
                removed_indices = set(df.index) - set(df_clean.index)
                sample_indices = list(removed_indices)[:3]
                for idx in sample_indices:
                    if idx < len(df):
                        row_preview = df.iloc[idx].tolist()
                        logger.debug(f"   🗑️  Fila {idx} eliminada: {row_preview}")
        else:
            logger.debug(f"✅ {stage_label} No se encontraron filas fantasma ({rows_after} filas)")

        return df_clean

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renombra columnas según configuración."""
        if df.empty:
            logger.warning("⚠️ DataFrame vacío en _rename_columns")
            return df

        column_map = self.config.get("presupuesto_column_map", {})
        logger.debug(f"📋 Mapa de columnas: {column_map}")

        df_renamed = find_and_rename_columns(df, column_map)
        logger.info(f"✅ Columnas renombradas: {list(df_renamed.columns)}")

        return df_renamed

    def _validate_required_columns(self, df: pd.DataFrame) -> bool:
        """Valida que existan las columnas requeridas."""
        if df.empty:
            logger.error("❌ DataFrame vacío en _validate_required_columns")
            return False

        is_valid, error = self.validator.validate_required_columns(
            df, [ColumnNames.CODIGO_APU], "presupuesto"
        )

        if not is_valid:
            logger.error(f"❌ Validación de columnas falló: {error}")
            logger.error(f"   Columnas disponibles: {list(df.columns)}")

        return is_valid

    def _clean_and_convert_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpia y convierte tipos de datos con validación robusta.
        Elimina registros con códigos APU vacíos o inválidos.
        """
        if df.empty:
            logger.warning("⚠️ DataFrame vacío en _clean_and_convert_data")
            return df

        initial_count = len(df)
        logger.info(f"🔧 Iniciando limpieza de {initial_count} registros...")

        # PASO 1: Limpiar códigos APU
        clean_code_params = self.config.get("clean_apu_code_params", {}).get(
            "presupuesto_item", {}
        )

        df[ColumnNames.CODIGO_APU] = (
            df[ColumnNames.CODIGO_APU]
            .astype(str)
            .apply(lambda code: clean_apu_code(code, **clean_code_params))
        )

        # PASO 2: Filtrar códigos vacíos, NaN, o inválidos
        mask_valid_code = (
            df[ColumnNames.CODIGO_APU].notna() &
            (df[ColumnNames.CODIGO_APU] != "") &
            (df[ColumnNames.CODIGO_APU] != "nan") &
            (df[ColumnNames.CODIGO_APU].str.strip() != "")
        )

        df = df[mask_valid_code].copy()

        codes_removed = initial_count - len(df)
        if codes_removed > 0:
            logger.warning(
                f"⚠️  Eliminados {codes_removed} registros con código APU vacío/inválido"
            )

        if df.empty:
            logger.error("❌ No quedan registros después de limpiar códigos APU")
            return pd.DataFrame()

        # PASO 3: Limpiar descripción (opcional pero recomendado)
        if ColumnNames.DESCRIPCION_APU in df.columns:
            df[ColumnNames.DESCRIPCION_APU] = (
                df[ColumnNames.DESCRIPCION_APU]
                .astype(str)
                .str.strip()
                .replace(['nan', 'None', 'NaN', ''], pd.NA)
                .fillna("SIN DESCRIPCIÓN")
            )

        # PASO 4: Convertir cantidades con manejo robusto
        if ColumnNames.CANTIDAD_PRESUPUESTO in df.columns:
            # Normalizar formato numérico
            cantidad_series = (
                df[ColumnNames.CANTIDAD_PRESUPUESTO]
                .astype(str)
                .str.strip()
                .str.replace(",", ".", regex=False)  # Coma decimal → punto
                .str.replace(r"[^\d.-]", "", regex=True)  # Solo dígitos, punto y signo
                .str.replace(r"\.(?=.*\.)", "", regex=True)  # Eliminar puntos duplicados
            )

            df[ColumnNames.CANTIDAD_PRESUPUESTO] = pd.to_numeric(
                cantidad_series, errors="coerce"
            ).fillna(0)

            # Advertir sobre cantidades cero o negativas
            zero_count = (df[ColumnNames.CANTIDAD_PRESUPUESTO] == 0).sum()
            negative_count = (df[ColumnNames.CANTIDAD_PRESUPUESTO] < 0).sum()

            if zero_count > 0:
                logger.warning(f"⚠️  {zero_count} registros con cantidad = 0")
            if negative_count > 0:
                logger.warning(f"⚠️  {negative_count} registros con cantidad negativa")

        # PASO 5: Validación final
        final_count = len(df)
        total_removed = initial_count - final_count

        logger.info(
            f"✅ Limpieza completada: {final_count} registros válidos "
            f"({total_removed} eliminados, {(final_count/max(initial_count, 1)*100):.1f}% conservado)"
        )

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina duplicados de códigos APU."""
        if df.empty:
            logger.warning("⚠️ DataFrame vacío en _remove_duplicates")
            return df

        initial_count = len(df)

        df_unique = self.validator.detect_and_log_duplicates(
            df, [ColumnNames.CODIGO_APU], "presupuesto", keep="first"
        )

        duplicates_removed = initial_count - len(df_unique)
        if duplicates_removed > 0:
            logger.info(
                f"🔄 Duplicados eliminados: {duplicates_removed} "
                f"({len(df_unique)} únicos de {initial_count} totales)"
            )

        return df_unique


class InsumosProcessor:
    """Procesador especializado para archivos de insumos."""

    def __init__(self, thresholds: ProcessingThresholds, profile: dict):
        self.thresholds = thresholds
        self.profile = profile
        self.validator = DataValidator()

    def process(self, file_path: str) -> pd.DataFrame:
        """Procesa el archivo CSV de insumos con formato no estándar."""
        try:
            records = self._parse_file(file_path)

            if not records:
                logger.warning("⚠️  No se encontraron registros en archivo de insumos")
                return pd.DataFrame()

            df = pd.DataFrame(records)
            df = self._rename_and_select_columns(df)
            df = self._convert_and_normalize(df)
            df = self._remove_duplicates(df)

            logger.info(f"✅ Insumos cargados: {len(df)} insumos únicos")
            return df

        except Exception as e:
            logger.error(f"❌ Error procesando archivo de insumos: {e}", exc_info=True)
            return pd.DataFrame()

    def _parse_file(self, file_path: str) -> List[Dict]:
        """Parsea el archivo de insumos con formato especial."""
        encoding = self.profile.get("encoding", "latin1")
        with open(file_path, "r", encoding=encoding) as f:
            lines = f.readlines()

        records = []
        current_group = None
        header = None

        for line in lines:
            parts = [p.strip().replace('"', "") for p in line.strip().split(";")]

            if not any(parts):
                continue

            # Detectar grupo
            if parts[0].startswith("G"):
                current_group = parts[1]
                header = None
                continue

            # Detectar encabezado
            if "CODIGO" in parts[0] and "DESCRIPCION" in parts[1]:
                header = ["CODIGO", "DESCRIPCION", "UND", "CANT.", "VR. UNIT."]
                continue

            # Parsear registro
            if header and current_group:
                record = {ColumnNames.GRUPO_INSUMO: current_group}
                for i, col in enumerate(header):
                    record[col] = parts[i] if i < len(parts) else None
                records.append(record)

        return records

    def _rename_and_select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Renombra y selecciona columnas relevantes."""
        df = df.rename(
            columns={
                "DESCRIPCION": ColumnNames.DESCRIPCION_INSUMO,
                "VR. UNIT.": ColumnNames.VR_UNITARIO_INSUMO,
            }
        )

        final_cols = [
            ColumnNames.GRUPO_INSUMO,
            ColumnNames.DESCRIPCION_INSUMO,
            ColumnNames.VR_UNITARIO_INSUMO,
        ]

        return df[final_cols]

    def _convert_and_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convierte valores numéricos y normaliza texto."""
        # Convertir precios
        df[ColumnNames.VR_UNITARIO_INSUMO] = pd.to_numeric(
            df[ColumnNames.VR_UNITARIO_INSUMO].astype(str).str.replace(",", "."),
            errors="coerce",
        )

        # Normalizar descripciones
        df[ColumnNames.DESCRIPCION_INSUMO_NORM] = normalize_text_series(
            df[ColumnNames.DESCRIPCION_INSUMO]
        )

        # Eliminar filas sin descripción
        df = df.dropna(subset=[ColumnNames.DESCRIPCION_INSUMO])

        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Elimina duplicados conservando el de mayor precio."""
        duplicates = df[
            df.duplicated(subset=[ColumnNames.DESCRIPCION_INSUMO_NORM], keep=False)
        ]

        if not duplicates.empty:
            logger.warning(
                f"⚠️  Se encontraron {len(duplicates)} insumos con descripciones "
                f"normalizadas duplicadas. Se conservará el de mayor precio."
            )

            df = df.sort_values(
                ColumnNames.VR_UNITARIO_INSUMO, ascending=False
            ).drop_duplicates(subset=[ColumnNames.DESCRIPCION_INSUMO_NORM], keep="first")

        return df


class APUCostCalculator:
    """Calculador de costos y metadatos de APUs."""

    def __init__(self, config: dict, thresholds: ProcessingThresholds):
        self.config = config
        self.thresholds = thresholds

    def calculate(
        self, df_merged: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Calcula costos, clasifica APUs y extrae metadatos."""

        df_apu_costos = self._aggregate_costs_by_type(df_merged)
        df_apu_costos = self._calculate_unit_values(df_apu_costos)
        df_apu_costos = self._detect_cost_outliers(df_apu_costos)
        df_apu_costos = self._classify_apus(df_apu_costos)

        df_tiempo = self._calculate_installation_time(df_merged)
        df_rendimiento = self._calculate_performance(df_merged)

        return df_apu_costos, df_tiempo, df_rendimiento

    def _aggregate_costs_by_type(self, df_merged: pd.DataFrame) -> pd.DataFrame:
        """Agrupa costos por APU y tipo de insumo."""
        logger.info("📊 Agregando costos por APU y tipo de insumo...")

        df_apu_costos = (
            df_merged.groupby([ColumnNames.CODIGO_APU, ColumnNames.TIPO_INSUMO])[
                ColumnNames.COSTO_INSUMO_EN_APU
            ]
            .sum()
            .unstack(fill_value=0)
            .reset_index()
        )

        # Mapear tipos de insumo a columnas de costo
        cost_cols_map = {
            InsumoTypes.SUMINISTRO: ColumnNames.MATERIALES,
            InsumoTypes.MANO_DE_OBRA: ColumnNames.MANO_DE_OBRA,
            InsumoTypes.EQUIPO: ColumnNames.EQUIPO,
            InsumoTypes.TRANSPORTE: ColumnNames.OTROS,
            InsumoTypes.OTRO: ColumnNames.OTROS,
        }

        df_apu_costos = df_apu_costos.rename(columns=cost_cols_map)

        # Consolidar columna OTROS si se generaron múltiples
        if ColumnNames.OTROS in df_apu_costos.columns:
            if isinstance(df_apu_costos[ColumnNames.OTROS], pd.DataFrame):
                df_apu_costos[ColumnNames.OTROS] = df_apu_costos[ColumnNames.OTROS].sum(
                    axis=1
                )

        # Asegurar que existan todas las columnas de costo
        final_cost_cols = [
            ColumnNames.MATERIALES,
            ColumnNames.MANO_DE_OBRA,
            ColumnNames.EQUIPO,
            ColumnNames.OTROS,
        ]

        for col in final_cost_cols:
            if col not in df_apu_costos.columns:
                df_apu_costos[col] = 0

        logger.info(f"✅ Costos agregados: {len(df_apu_costos)} APUs únicos")
        return df_apu_costos

    def _calculate_unit_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula valores unitarios por APU."""
        df[ColumnNames.VALOR_SUMINISTRO_UN] = df[ColumnNames.MATERIALES]

        df[ColumnNames.VALOR_INSTALACION_UN] = (
            df[ColumnNames.MANO_DE_OBRA] + df[ColumnNames.EQUIPO]
        )

        cost_cols = [
            ColumnNames.MATERIALES,
            ColumnNames.MANO_DE_OBRA,
            ColumnNames.EQUIPO,
            ColumnNames.OTROS,
        ]
        df[ColumnNames.VALOR_CONSTRUCCION_UN] = df[cost_cols].sum(axis=1)

        return df

    def _detect_cost_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta y registra valores de costo anómalos."""
        valor_max = df[ColumnNames.VALOR_CONSTRUCCION_UN].max()
        valor_avg = df[ColumnNames.VALOR_CONSTRUCCION_UN].mean()
        valor_std = df[ColumnNames.VALOR_CONSTRUCCION_UN].std()

        logger.info(
            f"📈 Estadísticas de costos unitarios: "
            f"Max=${valor_max:,.2f}, Avg=${valor_avg:,.2f}, Std=${valor_std:,.2f}"
        )

        outlier_threshold = valor_avg + (self.thresholds.outlier_std_multiplier * valor_std)

        outliers = df[df[ColumnNames.VALOR_CONSTRUCCION_UN] > outlier_threshold]

        if not outliers.empty:
            logger.warning(
                f"⚠️  Se detectaron {len(outliers)} APUs con costos atípicos "
                f"(>{outlier_threshold:,.2f})"
            )

            for _, outlier in outliers.head(5).iterrows():
                logger.warning(
                    f"   APU {outlier[ColumnNames.CODIGO_APU]}: "
                    f"${outlier[ColumnNames.VALOR_CONSTRUCCION_UN]:,.2f} "
                    f"(MO: ${outlier.get(ColumnNames.MANO_DE_OBRA, 0):,.2f}, "
                    f"MAT: ${outlier.get(ColumnNames.MATERIALES, 0):,.2f})"
                )

        return df

    def _classify_apus(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clasifica APUs según distribución de costos."""

        def classify_apu(row) -> str:
            costo_total = row[ColumnNames.VALOR_CONSTRUCCION_UN]

            if costo_total == 0:
                return APUTypes.INDEFINIDO

            # Contexto para eval()
            context = {
                "porcentaje_mo_eq": (
                    (row.get(ColumnNames.MANO_DE_OBRA, 0) + row.get(ColumnNames.EQUIPO, 0))
                    / costo_total
                )
                * 100,
                "porcentaje_materiales": (row.get(ColumnNames.MATERIALES, 0) / costo_total)
                * 100,
            }

            # Motor de reglas desde config
            for rule in self.config.get("apu_classification_rules", []):
                try:
                    if eval(rule["condition"], {"__builtins__": {}}, context):
                        return rule["type"]
                except Exception as e:
                    logger.error(f"Error evaluando regla: {rule}. Error: {e}")

            return APUTypes.OBRA_COMPLETA

        df[ColumnNames.TIPO_APU] = df.apply(classify_apu, axis=1)

        # Registrar distribución de tipos
        tipo_counts = df[ColumnNames.TIPO_APU].value_counts()
        logger.info(f"📋 Clasificación de APUs:\n{tipo_counts}")

        return df

    def _calculate_installation_time(self, df_merged: pd.DataFrame) -> pd.DataFrame:
        """Calcula tiempo de instalación basado en mano de obra."""
        df_tiempo = (
            df_merged[df_merged[ColumnNames.TIPO_INSUMO] == InsumoTypes.MANO_DE_OBRA]
            .groupby(ColumnNames.CODIGO_APU)[ColumnNames.CANTIDAD_APU]
            .sum()
            .reset_index()
        )

        df_tiempo = df_tiempo.rename(
            columns={ColumnNames.CANTIDAD_APU: ColumnNames.TIEMPO_INSTALACION}
        )

        return df_tiempo

    def _calculate_performance(self, df_merged: pd.DataFrame) -> pd.DataFrame:
        """Calcula rendimiento diario si la columna existe."""
        if ColumnNames.RENDIMIENTO not in df_merged.columns:
            return pd.DataFrame(
                columns=[ColumnNames.CODIGO_APU, ColumnNames.RENDIMIENTO_DIA]
            )

        df_rendimiento = (
            df_merged[df_merged[ColumnNames.TIPO_INSUMO] == InsumoTypes.MANO_DE_OBRA]
            .groupby(ColumnNames.CODIGO_APU)[ColumnNames.RENDIMIENTO]
            .sum()
            .reset_index()
        )

        df_rendimiento = df_rendimiento.rename(
            columns={ColumnNames.RENDIMIENTO: ColumnNames.RENDIMIENTO_DIA}
        )

        return df_rendimiento


class DataMerger:
    """Gestor de merge de datos con validaciones robustas."""

    def __init__(self, thresholds: ProcessingThresholds):
        self.thresholds = thresholds
        self.validator = DataValidator()

    def merge_apus_with_insumos(
        self, df_apus: pd.DataFrame, df_insumos: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge de APUs con catálogo de insumos."""
        logger.info("🔗 Iniciando merge de APUs con catálogo de insumos...")

        # Validar que existe la columna de normalización
        if ColumnNames.NORMALIZED_DESC not in df_apus.columns:
            logger.warning(
                f"⚠️  Columna {ColumnNames.NORMALIZED_DESC} no encontrada. "
                f"Creando como fallback..."
            )
            df_apus[ColumnNames.NORMALIZED_DESC] = normalize_text_series(
                df_apus[ColumnNames.DESCRIPCION_INSUMO]
            )

        rows_before = len(df_apus)

        df_merged = pd.merge(
            df_apus,
            df_insumos,
            left_on=ColumnNames.NORMALIZED_DESC,
            right_on=ColumnNames.DESCRIPCION_INSUMO_NORM,
            how="left",
            suffixes=("_apu", ""),
            validate="m:1",
        )

        rows_after = len(df_merged)

        if rows_after != rows_before:
            logger.warning(
                f"⚠️  Cambio en número de filas durante merge: {rows_before} -> {rows_after}"
            )

        # Consolidar columnas duplicadas
        df_merged[ColumnNames.TIPO_INSUMO] = df_merged[ColumnNames.CATEGORIA]

        df_merged[ColumnNames.DESCRIPCION_INSUMO] = df_merged[
            ColumnNames.DESCRIPCION_INSUMO
        ].fillna(df_merged[f"{ColumnNames.DESCRIPCION_INSUMO}_apu"])

        # Renombrar unidad de insumo
        if f"{ColumnNames.UNIDAD_INSUMO}_apu" in df_merged.columns:
            df_merged = df_merged.rename(
                columns={f"{ColumnNames.UNIDAD_INSUMO}_apu": ColumnNames.UNIDAD_INSUMO}
            )

        logger.info(f"✅ Merge completado: {len(df_merged)} registros")

        return df_merged

    def merge_with_presupuesto(
        self, df_presupuesto: pd.DataFrame, df_apu_costos: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge de presupuesto con costos de APUs (1:1 validado)."""
        logger.info("🔗 Realizando merge final: presupuesto + costos...")

        logger.debug(
            f"   - Presupuesto: {len(df_presupuesto)} filas, "
            f"{df_presupuesto[ColumnNames.CODIGO_APU].nunique()} APUs únicos"
        )
        logger.debug(
            f"   - Costos: {len(df_apu_costos)} filas, "
            f"{df_apu_costos[ColumnNames.CODIGO_APU].nunique()} APUs únicos"
        )

        rows_before = len(df_presupuesto)

        try:
            df_final = pd.merge(
                df_presupuesto,
                df_apu_costos,
                on=ColumnNames.CODIGO_APU,
                how="left",
                validate="1:1",  # Validación crítica: solo 1:1
            )
        except pd.errors.MergeError as e:
            logger.error(f"❌ Error en merge 1:1: Se detectaron duplicados. {e}")

            # Diagnosticar duplicados
            dupes_presupuesto = df_presupuesto[
                df_presupuesto.duplicated(ColumnNames.CODIGO_APU, keep=False)
            ]
            dupes_costos = df_apu_costos[
                df_apu_costos.duplicated(ColumnNames.CODIGO_APU, keep=False)
            ]

            if not dupes_presupuesto.empty:
                logger.error(
                    f"Duplicados en presupuesto:\n"
                    f"{dupes_presupuesto[ColumnNames.CODIGO_APU].unique()}"
                )

            if not dupes_costos.empty:
                logger.error(
                    f"Duplicados en costos:\n{dupes_costos[ColumnNames.CODIGO_APU].unique()}"
                )

            raise

        rows_after = len(df_final)

        if rows_after != rows_before:
            logger.error(
                f"❌ Explosión cartesiana detectada: {rows_before} -> {rows_after} filas"
            )
            raise ValueError(
                "Merge produjo explosión cartesiana. Revise duplicados en datos."
            )

        logger.info(f"✅ Merge 1:1 exitoso: {len(df_final)} filas")

        return df_final


# ==================== FUNCIONES AUXILIARES ====================


def group_and_split_description(df: pd.DataFrame) -> pd.DataFrame:
    """Conserva la descripción original y la divide en principal y secundaria."""
    if ColumnNames.DESCRIPCION_APU not in df.columns:
        logger.warning(
            f"⚠️  Columna {ColumnNames.DESCRIPCION_APU} no encontrada. "
            f"Saltando división de descripción."
        )
        return df

    df_result = df.copy()

    # Conservar original
    df_result[ColumnNames.ORIGINAL_DESCRIPTION] = df_result[ColumnNames.DESCRIPCION_APU]

    # Dividir por separador
    split_desc = df_result[ColumnNames.DESCRIPCION_APU].str.split(" / ", n=1, expand=True)

    df_result[ColumnNames.DESCRIPCION_APU] = split_desc[0]

    if split_desc.shape[1] > 1:
        df_result[ColumnNames.DESCRIPCION_SECUNDARIA] = split_desc[1]
    else:
        df_result[ColumnNames.DESCRIPCION_SECUNDARIA] = ""

    return df_result


def calculate_total_costs(
    df: pd.DataFrame, thresholds: ProcessingThresholds
) -> pd.DataFrame:
    """Calcula valores totales del presupuesto con validaciones."""
    logger.info("💵 Calculando valores totales del presupuesto...")

    # Validar que existe la columna de cantidad
    if ColumnNames.CANTIDAD_PRESUPUESTO not in df.columns:
        logger.error(f"❌ Columna {ColumnNames.CANTIDAD_PRESUPUESTO} no encontrada")
        raise ValueError(
            f"Columna requerida no encontrada: {ColumnNames.CANTIDAD_PRESUPUESTO}"
        )

    # Convertir y validar cantidades
    df[ColumnNames.CANTIDAD_PRESUPUESTO] = pd.to_numeric(
        df[ColumnNames.CANTIDAD_PRESUPUESTO], errors="coerce"
    ).fillna(0)

    logger.debug(
        f"Estadísticas de cantidades:\n{df[ColumnNames.CANTIDAD_PRESUPUESTO].describe()}"
    )

    # Calcular valores totales
    df[ColumnNames.VALOR_SUMINISTRO_TOTAL] = (
        df[ColumnNames.VALOR_SUMINISTRO_UN] * df[ColumnNames.CANTIDAD_PRESUPUESTO]
    )

    df[ColumnNames.VALOR_INSTALACION_TOTAL] = (
        df[ColumnNames.VALOR_INSTALACION_UN] * df[ColumnNames.CANTIDAD_PRESUPUESTO]
    )

    df[ColumnNames.VALOR_CONSTRUCCION_TOTAL] = (
        df[ColumnNames.VALOR_CONSTRUCCION_UN] * df[ColumnNames.CANTIDAD_PRESUPUESTO]
    )

    # Validar costo total
    total_construccion = df[ColumnNames.VALOR_CONSTRUCCION_TOTAL].sum()

    logger.info(f"💰 COSTO TOTAL CONSOLIDADO: ${total_construccion:,.2f}")

    if total_construccion > thresholds.max_total_cost:
        logger.error(
            f"❌ COSTO TOTAL ANORMALMENTE ALTO: ${total_construccion:,.2f} "
            f"(límite: ${thresholds.max_total_cost:,.2f})"
        )

        # Mostrar principales contribuyentes
        top_contributors = df.nlargest(10, ColumnNames.VALOR_CONSTRUCCION_TOTAL)[
            [
                ColumnNames.CODIGO_APU,
                ColumnNames.DESCRIPCION_APU,
                ColumnNames.CANTIDAD_PRESUPUESTO,
                ColumnNames.VALOR_CONSTRUCCION_UN,
                ColumnNames.VALOR_CONSTRUCCION_TOTAL,
            ]
        ]

        logger.error(f"Top 10 APUs contributores:\n{top_contributors}")

        raise ValueError(f"Costo total excede límite permitido: ${total_construccion:,.2f}")

    return df


def calculate_insumo_costs(
    df: pd.DataFrame, thresholds: ProcessingThresholds
) -> pd.DataFrame:
    """Calcula costos de insumos con validaciones robustas."""
    logger.info("💰 Calculando costos de insumos...")

    # Validar columnas requeridas
    required_cols = [
        ColumnNames.CANTIDAD_APU,
        ColumnNames.VR_UNITARIO_INSUMO,
        ColumnNames.VALOR_TOTAL_APU,
    ]

    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"⚠️  Columna {col} no encontrada. Creando con valor 0.")
            df[col] = 0

    # Estadísticas de cantidad
    qty_stats = df[ColumnNames.CANTIDAD_APU].describe()
    logger.debug(f"Estadísticas de cantidad:\n{qty_stats}")

    if df[ColumnNames.CANTIDAD_APU].max() > thresholds.max_quantity:
        logger.warning(
            f"⚠️  Cantidad extremadamente alta detectada: "
            f"{df[ColumnNames.CANTIDAD_APU].max():,.2f}"
        )

    # Estadísticas de precio
    precio_stats = df[ColumnNames.VR_UNITARIO_INSUMO].describe()
    logger.debug(f"Estadísticas de precio unitario:\n{precio_stats}")

    # Calcular costo usando numpy.where para eficiencia
    costo_insumo = np.where(
        df[ColumnNames.VR_UNITARIO_INSUMO].notna(),
        df[ColumnNames.CANTIDAD_APU] * df[ColumnNames.VR_UNITARIO_INSUMO],
        df[ColumnNames.VALOR_TOTAL_APU],
    )

    df[ColumnNames.COSTO_INSUMO_EN_APU] = pd.Series(costo_insumo).fillna(0)

    # Validar costos anómalos
    costo_max = df[ColumnNames.COSTO_INSUMO_EN_APU].max()

    if costo_max > thresholds.max_cost_per_item:
        logger.error(
            f"❌ COSTO ANÓMALO DETECTADO: ${costo_max:,.2f} "
            f"(límite: ${thresholds.max_cost_per_item:,.2f})"
        )

        anomalies = df[
            df[ColumnNames.COSTO_INSUMO_EN_APU] > thresholds.max_cost_per_item
        ].head(10)

        logger.error(
            "Registros anómalos:\n%s",
            anomalies[
                [
                    ColumnNames.CODIGO_APU,
                    ColumnNames.DESCRIPCION_INSUMO,
                    ColumnNames.CANTIDAD_APU,
                    ColumnNames.VR_UNITARIO_INSUMO,
                    ColumnNames.COSTO_INSUMO_EN_APU,
                ]
            ],
        )

    # Calcular valor unitario final con fallbacks
    df[ColumnNames.VR_UNITARIO_FINAL] = df[ColumnNames.VR_UNITARIO_INSUMO].fillna(
        df.get(ColumnNames.PRECIO_UNIT_APU, 0)
    )

    # Fallback adicional: calcular desde costo total
    cantidad_safe = df[ColumnNames.CANTIDAD_APU].replace(0, 1)
    df[ColumnNames.VR_UNITARIO_FINAL] = df[ColumnNames.VR_UNITARIO_FINAL].fillna(
        df[ColumnNames.COSTO_INSUMO_EN_APU] / cantidad_safe
    )

    logger.info(f"✅ Costos de insumos calculados: {len(df)} registros")

    return df


def build_processed_apus_dataframe(
    df_apu_costos: pd.DataFrame,
    df_apus_raw: pd.DataFrame,
    df_tiempo: pd.DataFrame,
    df_rendimiento: pd.DataFrame,
) -> pd.DataFrame:
    """Construye el DataFrame de APUs procesados consolidado."""
    logger.info("📦 Construyendo DataFrame de APUs procesados...")

    # Extraer descripciones únicas de APUs
    df_apu_descriptions = df_apus_raw[
        [ColumnNames.CODIGO_APU, ColumnNames.DESCRIPCION_APU, ColumnNames.UNIDAD_APU]
    ].drop_duplicates(subset=[ColumnNames.CODIGO_APU])

    # Merge costos + descripciones
    df_processed = pd.merge(
        df_apu_costos, df_apu_descriptions, on=ColumnNames.CODIGO_APU, how="left"
    )

    # Merge tiempo
    df_processed = pd.merge(df_processed, df_tiempo, on=ColumnNames.CODIGO_APU, how="left")

    # Merge rendimiento
    if not df_rendimiento.empty:
        df_processed = pd.merge(
            df_processed, df_rendimiento, on=ColumnNames.CODIGO_APU, how="left"
        )

    # Renombrar unidad
    if ColumnNames.UNIDAD_APU in df_processed.columns:
        df_processed = df_processed.rename(columns={ColumnNames.UNIDAD_APU: "UNIDAD"})

    # Normalizar descripción
    if ColumnNames.DESCRIPCION_APU in df_processed.columns:
        df_processed["DESC_NORMALIZED"] = normalize_text_series(
            df_processed[ColumnNames.DESCRIPCION_APU]
        )

    # Dividir descripción
    df_processed = group_and_split_description(df_processed)

    logger.info(f"✅ APUs procesados consolidados: {len(df_processed)} registros")

    return df_processed


def synchronize_data_sources(
    df_merged: pd.DataFrame, df_final: pd.DataFrame
) -> pd.DataFrame:
    """Sincroniza fuentes de datos para consistencia."""
    logger.info("🔄 Sincronizando fuentes de datos...")

    codigos_apu_validos = df_final[ColumnNames.CODIGO_APU].unique()

    insumos_antes = len(df_merged)
    df_merged_sync = df_merged[
        df_merged[ColumnNames.CODIGO_APU].isin(codigos_apu_validos)
    ].copy()
    insumos_despues = len(df_merged_sync)

    logger.info(
        f"✅ Sincronización completada. Insumos: {insumos_antes} -> {insumos_despues}"
    )

    return df_merged_sync


def build_output_dictionary(
    df_final: pd.DataFrame,
    df_insumos: pd.DataFrame,
    df_merged: pd.DataFrame,
    df_apus_raw: pd.DataFrame,
    df_processed_apus: pd.DataFrame,
) -> dict:
    """Construye el diccionario de salida estandarizado."""
    logger.info("📦 Construyendo diccionario de salida...")

    # Renombrar columnas en df_merged para salida
    df_merged_output = df_merged.rename(
        columns={
            ColumnNames.VR_UNITARIO_FINAL: "VR_UNITARIO",
            ColumnNames.COSTO_INSUMO_EN_APU: "VR_TOTAL",
        }
    )

    # Construir diccionario de insumos por grupo
    insumos_dict = {}
    for name, group in df_insumos.groupby(ColumnNames.GRUPO_INSUMO):
        if name and isinstance(name, str):
            insumos_dict[name.strip()] = (
                group[[ColumnNames.DESCRIPCION_INSUMO, ColumnNames.VR_UNITARIO_INSUMO]]
                .dropna()
                .to_dict("records")
            )

    # Construir diccionario final
    result_dict = {
        "presupuesto": df_final.to_dict("records"),
        "insumos": insumos_dict,
        "apus_detail": df_merged_output.to_dict("records"),
        "all_apus": df_apus_raw.to_dict("records"),
        "raw_insumos_df": df_insumos,
        "processed_apus": df_processed_apus.to_dict("records"),
    }

    logger.info("✅ Diccionario de salida construido")

    return result_dict


# ==================== LÓGICA PRINCIPAL ====================


def _do_processing(
    presupuesto_path: str, apus_path: str, insumos_path: str, config: dict
) -> dict:
    """
    Lógica central para procesar, unificar y calcular todos los datos.
    """
    logger.info("=" * 80)
    logger.info("🚀 Iniciando procesamiento de archivos con patrón pipeline...")
    logger.info("=" * 80)

    # Configuración de umbrales
    thresholds = ProcessingThresholds()
    if "processing_thresholds" in config:
        custom_thresholds = config["processing_thresholds"]
        for key, value in custom_thresholds.items():
            if hasattr(thresholds, key):
                setattr(thresholds, key, value)
                logger.debug(f"Umbral configurado: {key} = {value}")

    # Configurar rutas de salida
    output_dir = Path(config.get("output_dir", "data"))
    output_files = {
        "processed_apus": output_dir
        / config.get("processed_apus_file", "processed_apus.json"),
        "presupuesto_final": output_dir
        / config.get("presupuesto_final_file", "presupuesto_final.json"),
        "insumos_detalle": output_dir
        / config.get("insumos_detalle_file", "insumos_detalle.json"),
    }

    try:
        # Ejecutar pipeline
        pipeline = ProcessingPipeline(
            [
                LoadDataStep(config, thresholds),
                MergeDataStep(thresholds),
                CalculateCostsStep(config, thresholds),
                FinalMergeStep(thresholds),
                BuildOutputStep(),
            ]
        )

        initial_context = {
            "presupuesto_path": presupuesto_path,
            "apus_path": apus_path,
            "insumos_path": insumos_path,
        }

        logger.info("🔄 Ejecutando pipeline de procesamiento...")
        final_context = pipeline.run(initial_context)

        final_result = final_context.get("final_result")
        if not final_result:
            error_msg = "Pipeline no produjo resultado final"
            logger.error(f"❌ {error_msg}")
            return {"error": error_msg}

        if "error" in final_result:
            logger.error(f"❌ Error en resultado final: {final_result['error']}")
            return final_result

        # Validar datos
        logger.info("🔍 Validando datos procesados antes de guardar...")
        validation_results = _validate_output_data(final_result)
        if not validation_results["is_valid"]:
            logger.error(f"❌ Validación de datos falló: {validation_results['errors']}")
            for warning in validation_results["warnings"]:
                logger.warning(f"⚠️ {warning}")

        # Guardar archivos
        logger.info("=" * 80)
        logger.info("💾 Guardando archivos JSON de salida...")
        logger.info("=" * 80)

        saved_files = _save_output_files(final_result, output_files, config)

        # Registrar estadísticas
        _log_processing_statistics(final_result, saved_files)

        logger.info("=" * 80)
        logger.info("🎉 PROCESAMIENTO COMPLETADO EXITOSAMENTE")
        logger.info("=" * 80)

        final_result["output_files"] = {
            name: str(path) for name, path in saved_files.items()
        }
        final_result["processing_timestamp"] = pd.Timestamp.now().isoformat()

        return final_result

    except (ValueError, pd.errors.MergeError) as e:
        error_msg = f"Error en el pipeline: {e}"
        logger.error(f"❌ {error_msg}", exc_info=True)
        if "apus" in str(e).lower():
            logger.info("=" * 80)
            logger.info("🔍 Ejecutando diagnóstico del archivo de APUs...")
            logger.info("=" * 80)
            try:
                diagnostic = APUFileDiagnostic(apus_path)
                diagnostic.diagnose()
            except Exception as diag_e:
                logger.error(f"❌ Error durante el diagnóstico: {diag_e}", exc_info=True)
        logger.info("=" * 80)
        return {"error": error_msg}
    except Exception as e:
        error_msg = f"Error crítico en el pipeline: {e}"
        logger.error(f"❌ {error_msg}", exc_info=True)
        return {"error": error_msg}


# ============================================================
# FUNCIONES AUXILIARES PARA GUARDADO Y VALIDACIÓN
# ============================================================


def _validate_output_data(result: dict) -> dict:
    """Valida la integridad de los datos de salida."""
    validation = {"is_valid": True, "errors": [], "warnings": []}

    required_keys = ["presupuesto", "processed_apus", "apus_detail"]
    for key in required_keys:
        if key not in result:
            validation["errors"].append(f"Falta sección requerida: {key}")
            validation["is_valid"] = False
        elif not result[key]:
            validation["warnings"].append(f"Sección vacía: {key}")

    if "processed_apus" in result:
        processed_count = len(result["processed_apus"])
        if processed_count == 0:
            validation["errors"].append("processed_apus está vacío")
            validation["is_valid"] = False
        else:
            logger.info(f"✅ {processed_count} APUs procesados listos para guardar")

    if "processed_apus" in result and result["processed_apus"]:
        sample_apu = result["processed_apus"][0]
        expected_fields = [
            ColumnNames.CODIGO_APU,
            ColumnNames.DESCRIPCION_APU,
            ColumnNames.VALOR_CONSTRUCCION_UN,
        ]
        missing_fields = [field for field in expected_fields if field not in sample_apu]
        if missing_fields:
            validation["warnings"].append(
                f"Campos faltantes en processed_apus: {missing_fields}"
            )

    return validation


def _save_output_files(result: dict, output_files: dict, config: dict) -> dict:
    """Guarda los archivos JSON de salida de forma robusta."""
    import json

    from .utils import sanitize_for_json

    saved_files = {}

    # Asegurar que el directorio de salida existe
    for file_path in output_files.values():
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directorio verificado: {file_path.parent}")

    # Guardar processed_apus.json
    if "processed_apus" in result and result["processed_apus"]:
        try:
            processed_apus_path = output_files["processed_apus"]
            logger.info(f"💾 Guardando APUs procesados en: {processed_apus_path}")
            processed_data = sanitize_for_json(result["processed_apus"])
            with open(processed_apus_path, "w", encoding="utf-8") as f:
                json.dump(processed_data, f, indent=2, ensure_ascii=False)
            file_size = processed_apus_path.stat().st_size
            logger.info(
                f"✅ processed_apus.json guardado exitosamente "
                f"({len(processed_data)} registros, {file_size:,} bytes)"
            )
            saved_files["processed_apus"] = processed_apus_path
        except Exception as e:
            logger.error(f"❌ Error guardando processed_apus.json: {e}", exc_info=True)
    else:
        logger.warning("⚠️ No hay datos de processed_apus para guardar")

    # Guardar presupuesto_final.json
    if "presupuesto" in result and result["presupuesto"]:
        try:
            presupuesto_path = output_files["presupuesto_final"]
            logger.info(f"💾 Guardando presupuesto final en: {presupuesto_path}")
            presupuesto_data = sanitize_for_json(result["presupuesto"])
            with open(presupuesto_path, "w", encoding="utf-8") as f:
                json.dump(presupuesto_data, f, indent=2, ensure_ascii=False)
            file_size = presupuesto_path.stat().st_size
            logger.info(
                f"✅ presupuesto_final.json guardado "
                f"({len(presupuesto_data)} registros, {file_size:,} bytes)"
            )
            saved_files["presupuesto_final"] = presupuesto_path
        except Exception as e:
            logger.error(f"❌ Error guardando presupuesto_final.json: {e}", exc_info=True)

    # Guardar insumos_detalle.json
    if "apus_detail" in result and result["apus_detail"]:
        try:
            insumos_path = output_files["insumos_detalle"]
            logger.info(f"💾 Guardando detalle de insumos en: {insumos_path}")
            insumos_data = sanitize_for_json(result["apus_detail"])
            with open(insumos_path, "w", encoding="utf-8") as f:
                json.dump(insumos_data, f, indent=2, ensure_ascii=False)
            file_size = insumos_path.stat().st_size
            logger.info(
                f"✅ insumos_detalle.json guardado "
                f"({len(insumos_data)} registros, {file_size:,} bytes)"
            )
            saved_files["insumos_detalle"] = insumos_path
        except Exception as e:
            logger.error(f"❌ Error guardando insumos_detalle.json: {e}", exc_info=True)

    # Guardar archivo completo de respaldo (opcional)
    if config.get("save_full_backup", False):
        try:
            backup_path = output_files.get(
                "full_backup", Path(config.get("output_dir", "data")) / "full_backup.json"
            )
            logger.info(f"💾 Guardando respaldo completo en: {backup_path}")
            backup_data = {
                k: v
                for k, v in result.items()
                if k not in ["raw_insumos_df"] and not isinstance(v, pd.DataFrame)
            }
            backup_data_clean = sanitize_for_json(backup_data)
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(backup_data_clean, f, indent=2, ensure_ascii=False)
            file_size = backup_path.stat().st_size
            logger.info(f"✅ Respaldo completo guardado ({file_size:,} bytes)")
            saved_files["full_backup"] = backup_path
        except Exception as e:
            logger.error(f"⚠️ Error guardando respaldo completo: {e}", exc_info=True)

    return saved_files


def _log_processing_statistics(result: dict, saved_files: dict) -> None:
    """Registra estadísticas del procesamiento completado."""
    logger.info("=" * 80)
    logger.info("📊 ESTADÍSTICAS DEL PROCESAMIENTO")
    logger.info("=" * 80)

    stats = []
    if "presupuesto" in result:
        count = len(result["presupuesto"])
        stats.append(f" 📋 APUs en Presupuesto: {count:,}")
        if result["presupuesto"]:
            try:
                total_construccion = sum(
                    item.get(ColumnNames.VALOR_CONSTRUCCION_TOTAL, 0)
                    for item in result["presupuesto"]
                )
                stats.append(f" 💰 Costo Total Construcción: ${total_construccion:,.2f}")
            except Exception:
                pass

    if "processed_apus" in result:
        count = len(result["processed_apus"])
        stats.append(f" 🏗️ APUs Procesados: {count:,}")

    if "apus_detail" in result:
        count = len(result["apus_detail"])
        stats.append(f" 🔧 Insumos Detallados: {count:,}")

    if "insumos" in result:
        grupos = len(result["insumos"])
        total_insumos = sum(len(items) for items in result["insumos"].values())
        stats.append(f" 📦 Grupos de Insumos: {grupos}")
        stats.append(f" 📦 Total Insumos: {total_insumos:,}")

    for stat in stats:
        logger.info(stat)

    logger.info("")
    logger.info("📁 ARCHIVOS GENERADOS:")
    if saved_files:
        for name, path in saved_files.items():
            file_size = path.stat().st_size if path.exists() else 0
            size_mb = file_size / (1024 * 1024)
            logger.info(f" ✅ {name}: {path} ({size_mb:.2f} MB)")
    else:
        logger.warning(" ⚠️ No se guardaron archivos de salida")

    logger.info("=" * 80)


def process_all_files(
    presupuesto_path: str, apus_path: str, insumos_path: str, config: dict
) -> dict:
    """Orquesta el procesamiento completo de los archivos de entrada."""
    return _do_processing(presupuesto_path, apus_path, insumos_path, config)
