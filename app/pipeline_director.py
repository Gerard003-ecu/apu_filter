"""
Módulo: Pipeline Director V2 (La Matriz de Transformación de Valor M_P)
======================================================================
Este componente actúa como el "Sistema Nervioso Central" de APU_filter.
Su función es orquestar la secuencia de activación de los vectores de transformación
a través del Registro de Interacción Central (MICRegistry), garantizando la integridad
del "Vector de Estado" del proyecto.

Arquitectura y Fundamentos:
---------------------------
1. Orquestación Algebraica (Espacio Vectorial de Operadores):
   Utiliza el `MICRegistry` para proyectar "Intenciones" sobre un espacio
   vectorial de base ortogonal. Cada paso del pipeline (ej. `calculate_costs`) no es una
   función local, sino un vector base unitario ($e_i$) que solicita una transformación.

2. Filtración por Estratos (Jerarquía DIKW):
   Gestiona la ejecución respetando estrictamente la filtración de subespacios:
   $V_{PHYSICS} \subset V_{TACTICS} \subset V_{STRATEGY} \subset V_{WISDOM}$.

3. Protocolo de Caja de Cristal (Glass Box Persistence):
   Mantiene la trazabilidad forense completa. El estado del sistema se serializa
   entre pasos, permitiendo pausar, reanudar y auditar el proceso.
"""

import datetime
import enum
import hashlib
import logging
import os
import pickle
import sys
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import pandas as pd
from flask import current_app

from app.constants import ColumnNames, InsumoType
from app.flux_condenser import CondenserConfig, DataFluxCondenser
from app.matter_generator import MatterGenerator
from app.schemas import Stratum
from app.telemetry import TelemetryContext
from app.telemetry_narrative import TelemetryNarrator
from agent.business_topology import (
    BudgetGraphBuilder,
    BusinessTopologicalAnalyzer,
)
from app.business_agent import BusinessAgent

# ==================== CONSTANTES Y CLASES AUXILIARES ====================

from .apu_processor import (
    APUProcessor,
    APUCostCalculator,
    DataMerger,
    DataValidator,
    FileValidator,
    InsumosProcessor,
    PresupuestoProcessor,
    ProcessingThresholds,
    build_output_dictionary,
    build_processed_apus_dataframe,
    calculate_insumo_costs,
    calculate_total_costs,
    group_and_split_description,
    sanitize_for_json,
    synchronize_data_sources,
)
from app.semantic_translator import SemanticTranslator
from .data_validator import validate_and_clean_data

# Configuración explícita para debug
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)


# ==================== CLASE BASE ====================

class ProcessingStep(ABC):
    """Clase base abstracta para un paso del pipeline de procesamiento."""

    @abstractmethod
    def execute(self, context: dict, telemetry: TelemetryContext) -> dict:
        """
        Ejecuta la lógica del paso.

        Args:
            context (dict): Diccionario con el estado actual del procesamiento.
            telemetry (TelemetryContext): Contexto de telemetría para métricas.

        Returns:
            dict: El contexto actualizado.
        """
        pass


# ==================== ESTRUCTURAS ALGEBRAICAS (MIC) ====================

@dataclass(frozen=True)
class BasisVector:
    """
    Representa un vector base unitario e_i en el espacio de operaciones.
    Propiedades:
    - Index: Posición en la base.
    - Label: Identificador único.
    - Operator Class: Clase del paso asociado.
    - Stratum: Nivel jerárquico (DIKW).
    """
    index: int
    label: str
    operator_class: Type['ProcessingStep']
    stratum: Stratum


class MICRegistry:
    """
    Refinamiento: Catálogo centralizado de pasos del pipeline (anteriormente LinearInteractionMatrix).
    Se enfoca en la unicidad y tipo de los operadores, actuando como un Registry de pasos.
    """
    def __init__(self):
        self._basis: Dict[str, BasisVector] = {}
        self._dimension = 0
        self.logger = logging.getLogger(self.__class__.__name__)

    def add_basis_vector(
        self,
        label: str,
        step_class: Type['ProcessingStep'],
        stratum: Stratum
    ):
        """
        Agrega un paso al catálogo con validaciones básicas.
        """
        if not label or not isinstance(label, str):
            raise ValueError("Label must be a non-empty string.")

        if label in self._basis:
            raise ValueError(f"Duplicate label: '{label}'. Labels must be unique.")

        if not (isinstance(step_class, type) and issubclass(step_class, ProcessingStep)):
            raise TypeError(f"Class {step_class} must be a subclass of ProcessingStep.")

        vector = BasisVector(
            index=self._dimension,
            label=label,
            operator_class=step_class,
            stratum=stratum
        )
        self._basis[label] = vector
        self._dimension += 1
        self.logger.debug(f"Added step '{label}' at index {vector.index} (stratum: {stratum.name})")

    def get_basis_vector(self, label: str) -> Optional[BasisVector]:
        """Obtiene un vector base por su etiqueta."""
        return self._basis.get(label)

    def get_available_labels(self) -> List[str]:
        """Devuelve las etiquetas disponibles."""
        return list(self._basis.keys())


# ==================== IMPLEMENTACIÓN DE PASOS (PRESERVADOS) ====================

class LoadDataStep(ProcessingStep):
    """
    Paso de Carga de Datos.
    Carga los archivos CSV/Excel de presupuesto, APUs e insumos.
    """
    def __init__(self, config: dict, thresholds: ProcessingThresholds):
        if not config or not isinstance(config, dict):
            raise ValueError("Configuración inválida para LoadDataStep")
        self.config = config
        self.thresholds = thresholds or ProcessingThresholds()

    def execute(self, context: dict, telemetry: TelemetryContext) -> dict:
        """Ejecuta la carga y validación inicial de archivos."""
        telemetry.start_step("load_data")
        try:
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

            file_validator = FileValidator()
            validations = [
                (presupuesto_path, "presupuesto"),
                (apus_path, "APUs"),
                (insumos_path, "insumos"),
            ]

            for file_path, file_type in validations:
                is_valid, error = file_validator.validate_file_exists(
                    file_path, file_type
                )
                if not is_valid:
                    telemetry.record_error("load_data", error)
                    raise ValueError(error)

            file_profiles = self.config.get("file_profiles", {})
            if not file_profiles:
                logger.warning("⚠️ 'file_profiles' no encontrado en config, usando defaults vacíos.")
                file_profiles = {"presupuesto_default": {}, "insumos_default": {}, "apus_default": {}}

            presupuesto_profile = file_profiles.get("presupuesto_default", {})
            p_processor = PresupuestoProcessor(self.config, self.thresholds, presupuesto_profile)
            df_presupuesto = p_processor.process(presupuesto_path)

            if df_presupuesto is None or df_presupuesto.empty:
                error = "Procesamiento de presupuesto retornó DataFrame vacío"
                telemetry.record_error("load_data", error)
                raise ValueError(error)

            telemetry.record_metric("load_data", "presupuesto_rows", len(df_presupuesto))

            insumos_profile = file_profiles.get("insumos_default", {})
            i_processor = InsumosProcessor(self.thresholds, insumos_profile)
            df_insumos = i_processor.process(insumos_path)

            if df_insumos is None or df_insumos.empty:
                error = "Procesamiento de insumos retornó DataFrame vacío"
                telemetry.record_error("load_data", error)
                raise ValueError(error)

            telemetry.record_metric("load_data", "insumos_rows", len(df_insumos))

            apus_profile = file_profiles.get("apus_default", {})
            logger.info("⚡️ Iniciando DataFluxCondenser para APUs...")
            condenser_config_data = self.config.get("flux_condenser_config", {})

            try:
                condenser_config = CondenserConfig(**condenser_config_data)
            except TypeError as e:
                logger.warning(f"⚠️ Error en config de condenser, usando defaults: {e}")
                condenser_config = CondenserConfig()

            condenser = DataFluxCondenser(
                config=self.config,
                profile=apus_profile,
                condenser_config=condenser_config,
            )

            # Callbacks simplificados
            def on_progress_stats(processing_stats):
                try:
                    for metric_name, attr_name, default_value in [
                        ("avg_saturation", "avg_saturation", 0.0),
                        ("max_flyback_voltage", "max_flyback_voltage", 0.0),
                    ]:
                        val = getattr(processing_stats, attr_name, default_value)
                        telemetry.record_metric("flux_condenser", metric_name, val)
                except Exception:
                    pass

            def _publish_telemetry(metrics: Dict[str, Any]):
                pass # Simplificado para V2

            df_apus_raw = condenser.stabilize(
                apus_path,
                on_progress=on_progress_stats,
                progress_callback=_publish_telemetry,
                telemetry=telemetry,
            )

            if df_apus_raw is None or df_apus_raw.empty:
                error = "DataFluxCondenser retornó DataFrame vacío"
                telemetry.record_error("load_data", error)
                raise ValueError(error)

            telemetry.record_metric("load_data", "apus_raw_rows", len(df_apus_raw))
            logger.info("✅ DataFluxCondenser completado.")

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
        self.config = config
        self.thresholds = thresholds

    def execute(self, context: dict, telemetry: TelemetryContext) -> dict:
        telemetry.start_step("merge_data")
        try:
            df_apus_raw = context["df_apus_raw"]
            df_insumos = context["df_insumos"]
            
            processor = APUProcessor(self.config)
            df_merged = processor.unify_sources(df_apus_raw, df_insumos)

            telemetry.record_metric("merge_data", "merged_rows", len(df_merged))
            context["df_merged"] = df_merged

            telemetry.end_step("merge_data", "success")
            return context
        except Exception as e:
            telemetry.record_error("merge_data", str(e))
            telemetry.end_step("merge_data", "error")
            raise


class AuditedMergeStep(ProcessingStep):
    """
    Paso de Fusión con Auditoría Topológica (Mayer-Vietoris).
    Construye grafos temporales para validar la integridad antes de comprometer la fusión.
    """
    def __init__(self, config: dict, thresholds: ProcessingThresholds):
        self.config = config
        self.thresholds = thresholds

    def execute(self, context: dict, telemetry: TelemetryContext) -> dict:
        telemetry.start_step("audited_merge")
        try:
            df_a = context.get("df_presupuesto")
            df_b = context.get("df_apus_raw")
            df_insumos = context.get("df_insumos")

            if df_a is not None and df_b is not None:
                try:
                    builder = BudgetGraphBuilder()
                    graph_a = builder.build(df_a, pd.DataFrame())
                    graph_b = builder.build(pd.DataFrame(), df_b)

                    analyzer = BusinessTopologicalAnalyzer(telemetry=telemetry)
                    audit_result = analyzer.audit_integration_homology(graph_a, graph_b) # type: ignore

                    if audit_result.get("delta_beta_1", 0) > 0:
                        logger.warning(f"🚨 {audit_result.get('narrative')}")
                        telemetry.record_metric("topology", "emergent_cycles", audit_result["delta_beta_1"])
                        context["integration_risk_alert"] = audit_result
                    else:
                        logger.info(f"✅ Auditoría Mayer-Vietoris OK")

                except Exception as e_audit:
                    logger.error(f"❌ Error durante auditoría Mayer-Vietoris: {e_audit}")
                    telemetry.record_error("audited_merge_audit", str(e_audit))

            logger.info("🛠️ Ejecutando fusión física de datos...")
            merger = DataMerger(self.thresholds)
            df_merged = merger.merge_apus_with_insumos(df_b, df_insumos)

            telemetry.record_metric("audited_merge", "merged_rows", len(df_merged))
            context["df_merged"] = df_merged

            telemetry.end_step("audited_merge", "success")
            return context

        except Exception as e:
            telemetry.record_error("audited_merge", str(e))
            telemetry.end_step("audited_merge", "error")
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
        telemetry.start_step("calculate_costs")
        try:
            df_merged = context["df_merged"]
            processor = APUProcessor(self.config)
            df_apu_costos, df_tiempo, df_rendimiento = processor.process_vectors(df_merged)

            telemetry.record_metric("calculate_costs", "costos_calculated", len(df_apu_costos))

            context.update({
                "df_merged": df_merged,
                "df_apu_costos": df_apu_costos,
                "df_tiempo": df_tiempo,
                "df_rendimiento": df_rendimiento,
            })
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
        telemetry.start_step("final_merge")
        try:
            df_presupuesto = context["df_presupuesto"]
            df_apu_costos = context["df_apu_costos"]
            df_tiempo = context["df_tiempo"]

            processor = APUProcessor(self.config)
            df_final = processor.consolidate_results(df_presupuesto, df_apu_costos, df_tiempo)

            telemetry.record_metric("final_merge", "final_rows", len(df_final))
            context["df_final"] = df_final

            telemetry.end_step("final_merge", "success")
            return context
        except Exception as e:
            telemetry.record_error("final_merge", str(e))
            telemetry.end_step("final_merge", "error")
            raise


class BusinessTopologyStep(ProcessingStep):
    """
    Paso de Análisis de Negocio.
    Utiliza el BusinessAgent para auditar la integridad estructural y evaluar riesgos.
    """
    def __init__(self, config: dict, thresholds: ProcessingThresholds):
        self.config = config
        self.thresholds = thresholds

    def execute(self, context: dict, telemetry: TelemetryContext) -> dict:
        telemetry.start_step("business_topology")
        try:
            # Recuperar la instancia global de la MIC (del sistema de herramientas)
            mic_instance = getattr(current_app, "mic", None)
            
            # NOTA: BusinessAgent requiere la MIC global (tools_interface) para 'financial_analysis',
            # no la MIC interna del PipelineDirector.
            if not mic_instance:
                 logger.warning("⚠️ No se encontró 'current_app.mic'. BusinessAgent no podrá ejecutar análisis financiero.")
                 # Intentamos importar la clase para typing, pero una instancia vacía no servirá de mucho sin herramientas registradas.
                 try:
                     from .tools_interface import MICRegistry as GlobalMIC
                     mic_instance = GlobalMIC()
                 except ImportError:
                     pass

            if "validated_strata" not in context:
                context["validated_strata"] = {Stratum.PHYSICS, Stratum.TACTICS}
            elif isinstance(context["validated_strata"], set):
                context["validated_strata"].add(Stratum.PHYSICS)
                context["validated_strata"].add(Stratum.TACTICS)

            # Introspección Topológica
            from agent.business_topology import BusinessTopologicalAnalyzer
            topology_analyzer = BusinessTopologicalAnalyzer(telemetry=telemetry)
            
            df_final = context.get("df_final")
            df_merged = context.get("df_merged")
            
            if df_final is not None:
                # graph = topology_analyzer.materialize_structure(df_final, df_merged) # Assuming this method exists or similar logic
                builder = BudgetGraphBuilder()
                graph = builder.build(df_final, df_merged)
                context["graph"] = graph
                logger.info(f"🕸️ Grafo de negocio materializado: {graph.number_of_nodes()} nodos")
            
            # Evaluación de Estrategia
            if mic_instance:
                agent = BusinessAgent(
                    config=self.config,
                    mic=mic_instance,
                    telemetry=telemetry
                )
                try:
                    report = agent.evaluate_project(context)
                    if report:
                        logger.info("✅ BusinessAgent completó la evaluación.")
                        context["business_topology_report"] = report
                except Exception as ba_error:
                    logger.warning(f"BusinessAgent evaluation warning: {ba_error}")
            else:
                 logger.error("❌ No MIC instance available for BusinessAgent") # Should be covered by warning above

            telemetry.end_step("business_topology", "success")
            return context

        except Exception as e:
            logger.error(f"❌ Error en BusinessTopologyStep: {e}", exc_info=True)
            telemetry.record_error("business_topology", str(e))
            telemetry.end_step("business_topology", "error")
            return context


class MaterializationStep(ProcessingStep):
    """
    Paso de Materialización.
    Genera la Lista de Materiales (BOM) a partir del grafo topológico.
    """
    def __init__(self, config: dict, thresholds: ProcessingThresholds):
        self.config = config
        self.thresholds = thresholds

    def execute(self, context: dict, telemetry: TelemetryContext) -> dict:
        telemetry.start_step("materialization")
        try:
            if "business_topology_report" not in context:
                logger.warning("⚠️ No se encontró reporte de topología. Saltando materialización.")
                telemetry.end_step("materialization", "skipped")
                return context

            graph = context.get("graph")
            if not graph:
                builder = BudgetGraphBuilder()
                df_presupuesto = context.get("df_final")
                df_detail = context.get("df_merged")
                if df_presupuesto is not None and df_detail is not None:
                    graph = builder.build(df_presupuesto, df_detail)
                    context["graph"] = graph
                else:
                    telemetry.end_step("materialization", "error")
                    return context

            report = context.get("business_topology_report")
            stability = 10.0
            if report and hasattr(report, 'details') and report.details:
                stability = report.details.get("pyramid_stability", 10.0)

            flux_metrics = {"pyramid_stability": stability, "avg_saturation": 0.0}

            generator = MatterGenerator()
            bom = generator.materialize_project(graph, flux_metrics=flux_metrics, telemetry=telemetry)

            context["bill_of_materials"] = bom
            context["logistics_plan"] = asdict(bom)

            logger.info(f"✅ Materialización completada. Total ítems: {len(bom.requirements)}")
            telemetry.end_step("materialization", "success")
            return context

        except Exception as e:
            logger.error(f"❌ Error en MaterializationStep: {e}", exc_info=True)
            telemetry.record_error("materialization", str(e))
            telemetry.end_step("materialization", "error")
            return context


class BuildOutputStep(ProcessingStep):
    """
    Paso de Construcción de Salida.
    Prepara y valida el diccionario final de resultados para el cliente.
    """
    def __init__(self, config: dict, thresholds: ProcessingThresholds):
        self.config = config
        self.thresholds = thresholds

    def execute(self, context: dict, telemetry: TelemetryContext) -> dict:
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

            if "graph" not in context or "business_topology_report" not in context:
                 logger.warning("⚠️ Faltan artefactos de Estrategia (Grafo/Reporte). Generando salida básica.")
                 result_dict = build_output_dictionary(
                     df_final, df_insumos, df_merged, df_apus_raw, df_processed_apus
                 )
            else:
                 graph = context["graph"]
                 report = context["business_topology_report"]
                 translator = SemanticTranslator()
                 data_product_payload = translator.assemble_data_product(graph, report)
                 
                 data_product_payload["presupuesto"] = df_final.to_dict("records")
                 data_product_payload["insumos"] = df_insumos.to_dict("records")
                 result_dict = data_product_payload
                 logger.info("🦉 WISDOM: Producto de datos ensamblado por SemanticTranslator")

            validated_result = validate_and_clean_data(result_dict, telemetry_context=telemetry)
            validated_result["raw_insumos_df"] = df_insumos.to_dict("records")

            if "business_topology_report" in context:
                validated_result["audit_report"] = asdict(context["business_topology_report"])

            if "logistics_plan" in context:
                validated_result["logistics_plan"] = context["logistics_plan"]

            try:
                narrator = TelemetryNarrator()
                tech_narrative = narrator.summarize_execution(telemetry)
                validated_result["technical_audit"] = tech_narrative if isinstance(tech_narrative, dict) else tech_narrative
            except Exception as e:
                validated_result["technical_audit"] = {"error": str(e)}

            # Simple Checksum
            try:
                import json
                sanitized = sanitize_for_json(validated_result.get("presupuesto", []))
                s = json.dumps(sanitized, sort_keys=True, default=str)
                lineage_hash = hashlib.sha256(s.encode("utf-8")).hexdigest()
            except Exception:
                lineage_hash = "hash_failed"

            data_product = {
                "kind": "DataProduct",
                "metadata": {
                    "version": "3.0",
                    "lineage_hash": lineage_hash,
                    "generated_at": datetime.datetime.now().isoformat(),
                    "generator": "APU_Filter_Pipeline_v2",
                },
                "payload": validated_result,
            }

            context["final_result"] = data_product
            telemetry.end_step("build_output", "success")
            return context
        except Exception as e:
            telemetry.record_error("build_output", str(e))
            telemetry.end_step("build_output", "error")
            raise


# ==================== PIPELINE DIRECTOR (V2) ====================

class PipelineSteps(enum.Enum):
    LOAD_DATA = "load_data"
    AUDITED_MERGE = "audited_merge"
    CALCULATE_COSTS = "calculate_costs"
    FINAL_MERGE = "final_merge"
    BUSINESS_TOPOLOGY = "business_topology"
    MATERIALIZATION = "materialization"
    BUILD_OUTPUT = "build_output"


class PipelineDirector:
    """
    Director del pipeline V2: Orquesta pasos secuenciales con validación de estado.
    Refinamiento: Se simplifica la lógica de transición y persistencia.
    """
    def __init__(self, config: dict, telemetry: TelemetryContext):
        self.config = config
        self.telemetry = telemetry
        self.thresholds = self._load_thresholds(config)
        self.session_dir = Path(config.get("session_dir", "data/sessions"))
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Inicializar la MIC (ahora MICRegistry simplificado)
        self.mic = MICRegistry()
        self._initialize_vector_space_refined()

    def _initialize_vector_space_refined(self):
        """Inicializa la MIC con los pasos del pipeline."""
        steps_and_strata = [
            ("load_data", LoadDataStep, Stratum.PHYSICS),
            ("audited_merge", AuditedMergeStep, Stratum.PHYSICS),
            ("calculate_costs", CalculateCostsStep, Stratum.TACTICS),
            ("final_merge", FinalMergeStep, Stratum.PHYSICS),
            ("materialization", MaterializationStep, Stratum.TACTICS),
            ("business_topology", BusinessTopologyStep, Stratum.STRATEGY),
            ("build_output", BuildOutputStep, Stratum.WISDOM),
        ]
        for label, step_class, stratum in steps_and_strata:
            self.mic.add_basis_vector(label, step_class, stratum)

    def _load_context_state(self, session_id: str) -> dict:
        """Carga el estado de una sesión."""
        if not session_id: return {}
        try:
            session_file = self.session_dir / f"{session_id}.pkl"
            if session_file.exists():
                with open(session_file, "rb") as f:
                    return pickle.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load context for session {session_id}: {e}")
        return {}

    def _save_context_state(self, session_id: str, context: dict):
        """Guarda el estado de una sesión."""
        try:
            session_file = self.session_dir / f"{session_id}.pkl"
            with open(session_file, "wb") as f:
                pickle.dump(context, f)
            self.logger.debug(f"Context saved for session {session_id}")
        except Exception as e:
            self.logger.error(f"Failed to save context for session {session_id}: {e}")

    def _infer_current_stratum_from_context(self, context: dict) -> Optional[Stratum]:
        """Heurística para inferir el estrato actual del contexto."""
        keys = set(context.keys())
        if any(k in keys for k in ["df_presupuesto", "df_insumos", "df_apus_raw"]):
            return Stratum.PHYSICS
        if any(k in keys for k in ["df_merged", "df_apu_costos", "df_tiempo"]):
            return Stratum.TACTICS
        if any(k in keys for k in ["graph", "business_topology_report"]):
            return Stratum.STRATEGY
        if any(k in keys for k in ["final_result", "bill_of_materials"]):
            return Stratum.WISDOM
        return None

    def run_single_step(
        self,
        step_name: str,
        session_id: str,
        initial_context: Optional[Dict[str, Any]] = None,
        validate_stratum: bool = True
    ) -> Dict[str, Any]:
        """
        Ejecuta un único paso del pipeline.
        """
        self.logger.info(f"Executing step: {step_name} (Session: {session_id[:8]}...)")

        # 1. Cargar contexto
        context = self._load_context_state(session_id)
        if initial_context:
            context.update(initial_context)

        try:
            # 2. Obtener vector de la MIC
            basis_vector = self.mic.get_basis_vector(step_name)
            if not basis_vector:
                raise ValueError(f"Step '{step_name}' not found in the interaction matrix.")

            # 3. Validar transición de estrato (opcional)
            if validate_stratum:
                current_stratum = self._infer_current_stratum_from_context(context)
                target_stratum = basis_vector.stratum
                # Simple warning logic (omitted complex validation for brevity/clarity per V2)
                if current_stratum and hasattr(current_stratum, 'level') and target_stratum and hasattr(target_stratum, 'level'):
                     if current_stratum.level > target_stratum.level:
                         self.logger.warning(
                             f"Potential regression: {current_stratum.name} -> {target_stratum.name}."
                         )

            # 4. Instanciar y ejecutar paso
            step_instance = basis_vector.operator_class(self.config, self.thresholds)
            updated_context = step_instance.execute(context, self.telemetry)

            if updated_context is None:
                raise ValueError(f"Step {step_name} returned a null context.")

            # 5. Guardar contexto actualizado
            self._save_context_state(session_id, updated_context)

            self.logger.info(f"Step '{step_name}' completed successfully.")
            return {
                "status": "success",
                "step": step_name,
                "stratum": basis_vector.stratum.name,
                "session_id": session_id,
                "context_keys": list(updated_context.keys())
            }

        except Exception as e:
            self.logger.error(f"Error executing step '{step_name}': {e}", exc_info=True)
            self.telemetry.record_error(step_name, str(e))
            return {"status": "error", "step": step_name, "error": str(e), "session_id": session_id}

    def execute_pipeline_orchestrated(self, initial_context: dict) -> dict:
        """
        Ejecuta el pipeline completo de forma orquestada.
        """
        session_id = str(uuid.uuid4())
        self.logger.info(f"Starting orchestrated pipeline (Session ID: {session_id})")

        # Obtener receta de ejecución (default or custom)
        # Using PipelineSteps enum values
        default_recipe = [{"step": step.value, "enabled": True} for step in PipelineSteps]
        recipe = self.config.get("pipeline_recipe", default_recipe)

        context = initial_context
        # Initial save of context
        self._save_context_state(session_id, context)

        for step_idx, step_config in enumerate(recipe):
            step_name = step_config.get("step")
            enabled = step_config.get("enabled", True)

            if not enabled:
                self.logger.info(f"Skipping disabled step: {step_name}")
                continue

            self.logger.info(f"Orchestrating step [{step_idx + 1}]: {step_name}")
            
            # We don't pass context here, run_single_step loads it from session
            result = self.run_single_step(step_name, session_id)

            if result["status"] == "error":
                error_msg = f"Pipeline failed at step '{step_name}': {result.get('error')}"
                self.logger.critical(error_msg)
                raise RuntimeError(error_msg)

        final_context = self._load_context_state(session_id)
        self.logger.info(f"Pipeline completed successfully (Session: {session_id})")
        return final_context

    def _load_thresholds(self, config: dict):
        thresholds = ProcessingThresholds()
        if "processing_thresholds" in config:
            for key, value in config["processing_thresholds"].items():
                if hasattr(thresholds, key):
                    setattr(thresholds, key, value)
        return thresholds


def process_all_files(
    presupuesto_path: Union[str, Path],
    apus_path: Union[str, Path],
    insumos_path: Union[str, Path],
    config: dict = None,
    telemetry: TelemetryContext = None,
) -> dict:
    """
    Función de entrada principal para el pipeline (Batch Mode).

    Wrapper para PipelineDirector.execute_pipeline_orchestrated.
    """
    if config is None:
        config = {}
    
    # Ensure paths are absolute/resolved
    p_path = Path(presupuesto_path).resolve()
    a_path = Path(apus_path).resolve()
    i_path = Path(insumos_path).resolve()

    director = PipelineDirector(config, telemetry)

    initial_context = {
        "presupuesto_path": str(p_path),
        "apus_path": str(a_path),
        "insumos_path": str(i_path),
    }

    try:
        return director.execute_pipeline_orchestrated(initial_context)
    except Exception as e:
        logger.critical(f"🔥 Critical failure in process_all_files: {e}", exc_info=True)
        if telemetry:
             telemetry.record_error("process_all_files", str(e))
        raise
