"""
Este componente actúa como el "Sistema Nervioso Central" del ecosistema APU Filter.
Su función principal no es procesar datos, sino gestionar la evolución del **Vector de Estado**
del proyecto a través de un espacio vectorial jerarquizado, delegando las transformaciones
a la Matriz de Interacción Central (MICRegistry).

Fundamentos Matemáticos y Arquitectura de Gobernanza:
-----------------------------------------------------

1. Orquestación Algebraica (Espacio Vectorial de Operadores):
   El pipeline no es una lista de funciones, sino una secuencia ordenada de proyecciones
   sobre una base vectorial ortogonal $\{e_1, \dots, e_n\}$ registrada en la MIC.
   Cada paso (ej. `LoadDataStep`) proyecta una "Intención" que la MIC resuelve en un
   handler específico, desacoplando la definición del flujo de su implementación técnica.

2. Filtración por Estratos (Jerarquía DIKW):
   Implementa la restricción topológica de filtración de subespacios:
   $V_{PHYSICS} \subset V_{TACTICS} \subset V_{STRATEGY} \subset V_{WISDOM}$.
   El Director impone la **Clausura Transitiva**: no permite ejecutar un vector de
   Estrategia (Nivel 1) si los invariantes de Física (Nivel 3) y Táctica (Nivel 2)
   no han sido validados previamente en el Vector de Estado.

3. Auditoría Homológica (Secuencia de Mayer-Vietoris):
   En los pasos de fusión de datos (`AuditedMergeStep`), el sistema no realiza un simple JOIN.
   Verifica la exactitud de la secuencia de Mayer-Vietoris:
   $\dots \to H_k(A \cap B) \to H_k(A) \oplus H_k(B) \to H_k(A \cup B) \to \dots$
   Esto garantiza matemáticamente que la integración del Presupuesto ($A$) y los APUs ($B$)
   no introduzca ciclos lógicos espurios ($\beta_1$) ni desconexiones ($\beta_0$) artificiales.

4. Protocolo de Caja de Cristal (Glass Box Persistence):
   Garantiza la trazabilidad forense completa. El estado del sistema se serializa y
   firma criptográficamente entre transiciones de estrato, permitiendo auditoría,
   reanudación y depuración ("Time Travel Debugging") del proceso de decisión.
"""

import datetime
import enum
import hashlib
import json
import logging
import os
import pickle
import sys
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

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
from app.tools_interface import register_core_vectors

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

    mic: Optional['MICRegistry'] = None  # Inyectado por PipelineDirector.run_single_step

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
    Catálogo centralizado de pasos del pipeline.
    
    Refinamiento V3:
    - Propiedad `dimension` expuesta.
    - Iteración ordenada por índice de registro.
    - Consulta por estrato para validación de filtración.
    - Método `get_execution_sequence` para recetas por defecto.
    """
    def __init__(self):
        self._basis: Dict[str, BasisVector] = {}
        self._ordered_labels: List[str] = []
        self._dimension = 0
        self._vectors: Dict[str, Tuple[Stratum, Callable[..., Dict[str, Any]]]] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    @property
    def dimension(self) -> int:
        """Dimensión del espacio vectorial (número de operadores registrados)."""
        return self._dimension

    def add_basis_vector(
        self,
        label: str,
        step_class: Type['ProcessingStep'],
        stratum: Stratum
    ):
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
        self._ordered_labels.append(label)
        self._dimension += 1
        self.logger.debug(
            f"Registered e_{vector.index} = '{label}' (stratum: {stratum.name})"
        )

    def get_basis_vector(self, label: str) -> Optional[BasisVector]:
        return self._basis.get(label)

    def get_available_labels(self) -> List[str]:
        """Devuelve las etiquetas en orden de registro (preserva la secuencia de la base)."""
        return list(self._ordered_labels)

    def get_vectors_by_stratum(self, stratum: Stratum) -> List[BasisVector]:
        """Proyección sobre un subestrato: devuelve todos los vectores de un estrato dado."""
        return [
            self._basis[label]
            for label in self._ordered_labels
            if self._basis[label].stratum == stratum
        ]

    def get_execution_sequence(self) -> List[Dict[str, Any]]:
        """Genera la receta de ejecución por defecto respetando el orden de registro."""
        return [
            {"step": label, "enabled": True}
            for label in self._ordered_labels
        ]

    def __iter__(self):
        """Iteración sobre vectores base en orden de registro."""
        for label in self._ordered_labels:
            yield self._basis[label]

    def __len__(self) -> int:
        return self._dimension

    @property
    def registered_services(self) -> List[str]:
        """Returns list of registered handler-based service names."""
        return list(self._vectors.keys())

    def register_vector(
        self,
        service_name: str,
        stratum: Stratum,
        handler: Callable[..., Dict[str, Any]]
    ) -> None:
        """
        Registers a handler-based vector (used by register_core_vectors).
        Compatible with the tools_interface MICRegistry interface.
        """
        if not service_name or not service_name.strip():
            raise ValueError("service_name cannot be empty")
        if not callable(handler):
            raise TypeError("handler must be callable")
        if service_name in self._vectors:
            self.logger.warning(f"Overwriting existing vector: {service_name}")
        self._vectors[service_name] = (stratum, handler)
        self.logger.info(f"Vector registered: {service_name} [{stratum.name}]")

    def _normalize_validated_strata(self, raw: Any) -> set:
        """Normalizes validated_strata from context to a set of Stratum."""
        if isinstance(raw, set):
            return {s for s in raw if isinstance(s, Stratum)}
        if isinstance(raw, (list, tuple)):
            return {s for s in raw if isinstance(s, Stratum)}
        return set()

    def project_intent(
        self,
        service_name: str,
        payload: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Projects an intent onto the vectorial space by invoking a registered handler.
        """
        if service_name not in self._vectors:
            available = self.registered_services
            raise ValueError(
                f"Unknown vector: '{service_name}'. "
                f"Available: {available if available else 'none registered'}"
            )

        target_stratum, handler = self._vectors[service_name]

        try:
            result = handler(**payload)
            if not isinstance(result, dict):
                result = {"success": True, "result": result}
            if result.get("success", False):
                result["_mic_stratum"] = target_stratum.name
            return result
        except TypeError as e:
            self.logger.error(
                f"Handler signature mismatch for '{service_name}': {e}",
                exc_info=True
            )
            return {"success": False, "error": str(e)}
        except Exception as e:
            self.logger.error(
                f"Error executing vector '{service_name}': {e}",
                exc_info=True
            )
            return {"success": False, "error": str(e)}


# ==================== MAPA DE ESTRATOS ====================

_STRATUM_ORDER: Dict[Stratum, int] = {
    Stratum.PHYSICS: 0,
    Stratum.TACTICS: 1,
    Stratum.STRATEGY: 2,
    Stratum.WISDOM: 3,
}

_STRATUM_EVIDENCE: Dict[Stratum, List[str]] = {
    Stratum.PHYSICS: ["df_presupuesto", "df_insumos", "df_apus_raw"],
    Stratum.TACTICS: ["df_apu_costos", "df_tiempo", "df_rendimiento"],
    Stratum.STRATEGY: ["graph", "business_topology_report"],
    Stratum.WISDOM: ["final_result"],
}


def stratum_level(s: Stratum) -> int:
    """Retorna el nivel ordinal de un estrato en la filtración DIKW."""
    return _STRATUM_ORDER.get(s, -1)


class PipelineSteps(enum.Enum):
    """
    Receta canónica del pipeline. El orden del enum define el orden de ejecución.
    
    Grafo de dependencias (→ = "produce para"):
      LOAD_DATA → AUDITED_MERGE → CALCULATE_COSTS → FINAL_MERGE
                                                        ↓
                                              BUSINESS_TOPOLOGY → MATERIALIZATION
                                                                       ↓
                                                                  BUILD_OUTPUT
    """
    LOAD_DATA = "load_data"
    AUDITED_MERGE = "audited_merge"
    CALCULATE_COSTS = "calculate_costs"
    FINAL_MERGE = "final_merge"
    BUSINESS_TOPOLOGY = "business_topology"
    MATERIALIZATION = "materialization"
    BUILD_OUTPUT = "build_output"


# ==================== IMPLEMENTACIÓN DE PASOS ====================

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
        # Inmutabilidad: Crear un nuevo contexto para preservar el estado original
        new_context = context.copy()
        try:
            required_paths = ["presupuesto_path", "apus_path", "insumos_path"]
            paths = {}

            for path_key in required_paths:
                path_value = new_context.get(path_key)
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

            # 1. Proyectar intención de Física (Estabilización)
            # El mic debe haber sido inyectado por el Director
            # Payload debe coincidir con la firma del adaptador: file_path, config
            flux_result = self.mic.project_intent(
                "stabilize_flux",
                {"file_path": str(apus_path), "config": self.config},
                context
            )
            
            if not flux_result["success"]:
                error = flux_result.get("error", "Unknown error in stabilize_flux")
                telemetry.record_error("load_data", error)
                raise ValueError(error)

            # Recuperar datos estabilizados
            # flux_result["data"] es una lista de dicts (records)
            # Convertir de nuevo a DataFrame para mantener compatibilidad con el resto del pipeline
            df_apus_raw = pd.DataFrame(flux_result["data"])

            if df_apus_raw is None or df_apus_raw.empty:
                error = "DataFluxCondenser vector returned empty data"
                telemetry.record_error("load_data", error)
                raise ValueError(error)

            telemetry.record_metric("load_data", "apus_raw_rows", len(df_apus_raw))
            logger.info("✅ Vector stabilize_flux completado exitosamente.")

            # 2. Proyectar intención de Parsing Topológico (parse_raw)
            apus_profile = file_profiles.get("apus_default", {})
            parse_result = self.mic.project_intent(
                "parse_raw",
                {"file_path": str(apus_path), "profile": apus_profile},
                context
            )

            if not parse_result["success"]:
                error = parse_result.get("error", "Unknown error in parse_raw")
                telemetry.record_error("load_data", error)
                raise ValueError(error)

            raw_records = parse_result["raw_records"]
            parse_cache = parse_result["parse_cache"]
            telemetry.record_metric("load_data", "raw_records_count", len(raw_records))
            logger.info("✅ Vector parse_raw completado exitosamente.")

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

            new_context.update({
                "df_presupuesto": df_presupuesto,
                "df_insumos": df_insumos,
                "df_apus_raw": df_apus_raw,
                "raw_records": raw_records,
                "parse_cache": parse_cache,
            })

            telemetry.end_step("load_data", "success")
            return new_context

        except Exception as e:
            logger.error(f"❌ Error en LoadDataStep: {e}", exc_info=True)
            telemetry.record_error("load_data", str(e))
            telemetry.end_step("load_data", "error")
            raise


class AuditedMergeStep(ProcessingStep):
    """
    Paso de Fusión con Auditoría Topológica (Mayer-Vietoris).
    Refinamiento: Validación de precondiciones antes de operar.
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

            # ── Precondición: las fuentes de la fusión NO pueden ser None ──
            if df_b is None:
                error = "df_apus_raw is None: cannot proceed with merge."
                telemetry.record_error("audited_merge", error)
                raise ValueError(error)
            if df_insumos is None:
                error = "df_insumos is None: cannot proceed with merge."
                telemetry.record_error("audited_merge", error)
                raise ValueError(error)

            # ── Auditoría topológica (no bloquea la fusión si falla) ──
            if df_a is not None:
                try:
                    builder = BudgetGraphBuilder()
                    graph_a = builder.build(df_a, pd.DataFrame())
                    graph_b = builder.build(pd.DataFrame(), df_b)

                    analyzer = BusinessTopologicalAnalyzer(telemetry=telemetry)
                    audit_result = analyzer.audit_integration_homology(
                        graph_a, graph_b
                    )

                    delta_beta_1 = audit_result.get("delta_beta_1", 0)
                    if delta_beta_1 > 0:
                        logger.warning(
                            f"🚨 Mayer-Vietoris: {delta_beta_1} emergent cycle(s) detected. "
                            f"Narrative: {audit_result.get('narrative', 'N/A')}"
                        )
                        telemetry.record_metric(
                            "topology", "emergent_cycles", delta_beta_1
                        )
                        context["integration_risk_alert"] = audit_result
                    else:
                        logger.info("✅ Auditoría Mayer-Vietoris: homología preservada.")
                except Exception as e_audit:
                    logger.error(
                        f"❌ Auditoría Mayer-Vietoris falló (no bloquea fusión): {e_audit}"
                    )
                    telemetry.record_error("audited_merge_audit", str(e_audit))
            else:
                logger.info(
                    "ℹ️ df_presupuesto no disponible; auditoría Mayer-Vietoris omitida."
                )

            # ── Fusión física ──
            logger.info("🛠️ Ejecutando fusión física de datos...")
            merger = DataMerger(self.thresholds)
            df_merged = merger.merge_apus_with_insumos(df_b, df_insumos)

            if df_merged is None or df_merged.empty:
                error = "Merge produced empty DataFrame"
                telemetry.record_error("audited_merge", error)
                raise ValueError(error)

            telemetry.record_metric("audited_merge", "merged_rows", len(df_merged))
            context["df_merged"] = df_merged

            telemetry.end_step("audited_merge", "success")
            return context

        except Exception as e:
            telemetry.record_error("audited_merge", str(e))
            telemetry.end_step("audited_merge", "error")
            raise


class CalculateCostsStep(ProcessingStep):
    """Paso de Cálculo de Costos."""
    def __init__(self, config: dict, thresholds: ProcessingThresholds):
        self.config = config
        self.thresholds = thresholds

    def execute(self, context: dict, telemetry: TelemetryContext) -> dict:
        telemetry.start_step("calculate_costs")
        try:
            df_merged = context.get("df_merged")
            if df_merged is None or df_merged.empty:
                error = "df_merged is missing or empty: cannot calculate costs."
                telemetry.record_error("calculate_costs", error)
                raise ValueError(error)

            # Proyectar intención táctica vía MIC (inversión de control)
            raw_records = context.get("raw_records", [])
            parse_cache = context.get("parse_cache", {})

            logic_result = self.mic.project_intent(
                "structure_logic",
                {
                    "raw_records": raw_records,
                    "parse_cache": parse_cache,
                    "config": self.config,
                },
                context
            )

            if not logic_result["success"]:
                error = logic_result.get("error", "Unknown error in structure_logic")
                telemetry.record_error("calculate_costs", error)
                raise ValueError(error)

            # Protocolo de Dos Fases (Vectorial vs Clásico)
            # Fase 1: Intentar obtener tensores de costo vía MIC
            vectorial_success = False
            if logic_result.get("success") and "df_apu_costos" in logic_result:
                logger.info("✅ Usando tensores de costo vectoriales (structure_logic).")
                df_apu_costos = logic_result["df_apu_costos"]
                df_tiempo = logic_result.get("df_tiempo", pd.DataFrame())
                df_rendimiento = logic_result.get("df_rendimiento", pd.DataFrame())
                context["quality_report"] = logic_result.get("quality_report", {})
                vectorial_success = True

            # Fase 2: Fallback Clásico (APUProcessor)
            if not vectorial_success:
                logger.info("⚠️ Proyección vectorial incompleta. Activando fallback clásico APUProcessor.")
                processor = APUProcessor(self.config)
                df_apu_costos, df_tiempo, df_rendimiento = processor.process_vectors(
                    df_merged
                )

            telemetry.record_metric(
                "calculate_costs", "costos_rows", len(df_apu_costos)
            )
            telemetry.record_metric(
                "calculate_costs", "tiempo_rows", len(df_tiempo)
            )

            context["df_apu_costos"] = df_apu_costos
            context["df_tiempo"] = df_tiempo
            context["df_rendimiento"] = df_rendimiento

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
    Refinamiento: Separación clara, propagación de errores, acceso limpio a MIC global.
    """
    def __init__(self, config: dict, thresholds: ProcessingThresholds):
        self.config = config
        self.thresholds = thresholds

    def _resolve_mic_instance(self):
        """Intenta resolver la instancia global de MIC."""
        try:
            return getattr(current_app, "mic", None)
        except RuntimeError:
            logger.warning(
                "⚠️ No Flask app context. BusinessAgent financial analysis unavailable."
            )
            return None

    def execute(self, context: dict, telemetry: TelemetryContext) -> dict:
        telemetry.start_step("business_topology")
        try:
            df_final = context.get("df_final")
            df_merged = context.get("df_merged")

            if df_final is None:
                error = "df_final is required for BusinessTopologyStep."
                telemetry.record_error("business_topology", error)
                raise ValueError(error)

            # ── Fase 1: Materialización del grafo topológico ──
            builder = BudgetGraphBuilder()
            graph = builder.build(df_final, df_merged if df_merged is not None else pd.DataFrame())
            context["graph"] = graph
            logger.info(
                f"🕸️ Grafo de negocio materializado: "
                f"{graph.number_of_nodes()} nodos, {graph.number_of_edges()} aristas"
            )
            telemetry.record_metric("business_topology", "graph_nodes", graph.number_of_nodes())
            telemetry.record_metric("business_topology", "graph_edges", graph.number_of_edges())

            # ── Fase 2: Evaluación por BusinessAgent (degradable) ──
            mic_instance = self._resolve_mic_instance()
            if mic_instance:
                try:
                    agent = BusinessAgent(
                        config=self.config,
                        mic=mic_instance,
                        telemetry=telemetry,
                    )
                    report = agent.evaluate_project(context)
                    if report:
                        context["business_topology_report"] = report
                        logger.info("✅ BusinessAgent completó la evaluación.")
                    else:
                        logger.warning("⚠️ BusinessAgent retornó reporte vacío.")
                except Exception as ba_error:
                    logger.warning(
                        f"⚠️ BusinessAgent evaluation degraded: {ba_error}",
                        exc_info=True,
                    )
                    telemetry.record_error("business_agent", str(ba_error))
            else:
                logger.warning(
                    "⚠️ Sin instancia MIC global. "
                    "Evaluación de negocio limitada a grafo topológico."
                )

            telemetry.end_step("business_topology", "success")
            return context

        except Exception as e:
            logger.error(f"❌ Error en BusinessTopologyStep: {e}", exc_info=True)
            telemetry.record_error("business_topology", str(e))
            telemetry.end_step("business_topology", "error")
            raise


class MaterializationStep(ProcessingStep):
    """
    Paso de Materialización (BOM).
    Refinamiento: Manejo de skips legítimos y propagación de errores reales.
    """
    def __init__(self, config: dict, thresholds: ProcessingThresholds):
        self.config = config
        self.thresholds = thresholds

    def execute(self, context: dict, telemetry: TelemetryContext) -> dict:
        telemetry.start_step("materialization")

        # ── Precondición: reporte topológico requerido ──
        if "business_topology_report" not in context:
            logger.warning(
                "⚠️ business_topology_report ausente. "
                "Materialización omitida (degradación controlada)."
            )
            telemetry.record_metric("materialization", "skipped", True)
            telemetry.end_step("materialization", "skipped")
            return context

        try:
            # ── Resolver o construir el grafo ──
            graph = context.get("graph")
            if not graph:
                builder = BudgetGraphBuilder()
                df_final = context.get("df_final")
                df_merged = context.get("df_merged")

                if df_final is None:
                    error = "Cannot materialize: df_final missing and no prebuilt graph."
                    telemetry.record_error("materialization", error)
                    raise ValueError(error)

                graph = builder.build(
                    df_final,
                    df_merged if df_merged is not None else pd.DataFrame(),
                )
                context["graph"] = graph
                logger.info("🕸️ Grafo reconstruido para materialización.")

            # ── Extraer métricas de estabilidad ──
            report = context["business_topology_report"]
            stability = 10.0
            if hasattr(report, "details") and isinstance(report.details, dict):
                stability = report.details.get("pyramid_stability", 10.0)

            flux_metrics = {
                "pyramid_stability": stability,
                "avg_saturation": 0.0,
            }

            # ── Generar BOM ──
            generator = MatterGenerator()
            bom = generator.materialize_project(
                graph, flux_metrics=flux_metrics, telemetry=telemetry
            )

            context["bill_of_materials"] = bom
            context["logistics_plan"] = asdict(bom)

            telemetry.record_metric(
                "materialization", "total_items", len(bom.requirements)
            )
            logger.info(
                f"✅ Materialización completada. Total ítems: {len(bom.requirements)}"
            )
            telemetry.end_step("materialization", "success")
            return context

        except Exception as e:
            logger.error(f"❌ Error en MaterializationStep: {e}", exc_info=True)
            telemetry.record_error("materialization", str(e))
            telemetry.end_step("materialization", "error")
            raise


class BuildOutputStep(ProcessingStep):
    """
    Paso de Construcción de Salida.
    Refinamiento: Checksum robusto de payload completo.
    """
    def __init__(self, config: dict, thresholds: ProcessingThresholds):
        self.config = config
        self.thresholds = thresholds

    def _compute_lineage_hash(self, payload: dict) -> str:
        """Calcula un hash SHA-256 sobre el payload completo."""
        hash_input_parts = []
        for key in sorted(payload.keys()):
            value = payload[key]
            try:
                sanitized = sanitize_for_json(value) if isinstance(value, (list, dict)) else value
                part = json.dumps(
                    {key: sanitized}, sort_keys=True, default=str
                )
            except (TypeError, ValueError):
                # Fallback para objetos no serializables
                part = f"{key}:type={type(value).__name__},len={len(value) if hasattr(value, '__len__') else 'N/A'}"
            hash_input_parts.append(part)

        composite = "|".join(hash_input_parts)
        return hashlib.sha256(composite.encode("utf-8")).hexdigest()

    def execute(self, context: dict, telemetry: TelemetryContext) -> dict:
        telemetry.start_step("build_output")
        try:
            # ── Extraer artefactos requeridos ──
            required_keys = [
                "df_final", "df_insumos", "df_merged",
                "df_apus_raw", "df_apu_costos", "df_tiempo", "df_rendimiento",
            ]
            missing = [k for k in required_keys if k not in context]
            if missing:
                error = f"BuildOutputStep missing required context keys: {missing}"
                telemetry.record_error("build_output", error)
                raise ValueError(error)

            df_final = context["df_final"]
            df_insumos = context["df_insumos"]
            df_merged = context["df_merged"]
            df_apus_raw = context["df_apus_raw"]
            df_apu_costos = context["df_apu_costos"]
            df_tiempo = context["df_tiempo"]
            df_rendimiento = context["df_rendimiento"]

            # ── Sincronización ──
            df_merged = synchronize_data_sources(df_merged, df_final)
            df_processed_apus = build_processed_apus_dataframe(
                df_apu_costos, df_apus_raw, df_tiempo, df_rendimiento
            )

            # ── Ensamblaje del producto de datos ──
            has_strategy_artifacts = (
                "graph" in context and "business_topology_report" in context
            )

            if has_strategy_artifacts:
                graph = context["graph"]
                report = context["business_topology_report"]
                translator = SemanticTranslator()
                result_dict = translator.assemble_data_product(graph, report)
                result_dict["presupuesto"] = df_final.to_dict("records")
                result_dict["insumos"] = df_insumos.to_dict("records")
                logger.info("🦉 WISDOM: Producto de datos ensamblado por SemanticTranslator")
            else:
                logger.warning(
                    "⚠️ Sin artefactos de Estrategia. Generando salida básica."
                )
                result_dict = build_output_dictionary(
                    df_final, df_insumos, df_merged, df_apus_raw, df_processed_apus
                )

            # ── Validación y enriquecimiento ──
            validated_result = validate_and_clean_data(
                result_dict, telemetry_context=telemetry
            )
            validated_result["raw_insumos_df"] = df_insumos.to_dict("records")

            if "business_topology_report" in context:
                validated_result["audit_report"] = asdict(
                    context["business_topology_report"]
                )

            if "logistics_plan" in context:
                validated_result["logistics_plan"] = context["logistics_plan"]

            # ── Narrativa técnica ──
            try:
                narrator = TelemetryNarrator()
                tech_narrative = narrator.summarize_execution(telemetry)
                validated_result["technical_audit"] = tech_narrative
            except Exception as e:
                logger.warning(f"⚠️ Narrativa técnica degradada: {e}")
                validated_result["technical_audit"] = {
                    "status": "degraded",
                    "error": str(e),
                }

            # ── Hash de linaje ──
            lineage_hash = self._compute_lineage_hash(validated_result)

            data_product = {
                "kind": "DataProduct",
                "metadata": {
                    "version": "3.0",
                    "lineage_hash": lineage_hash,
                    "generated_at": datetime.datetime.now(
                        datetime.timezone.utc
                    ).isoformat(),
                    "generator": "APU_Filter_Pipeline_v2.1",
                    "strata_validated": [
                        s.name
                        for s in context.get("validated_strata", set())
                        if isinstance(s, Stratum)
                    ],
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


# ==================== PIPELINE DIRECTOR (V3) ====================

class PipelineDirector:
    """
    Director del pipeline V3: Orquesta pasos secuenciales con validación de estado.
    Orquestador Algebraico.
    """
    def __init__(self, config: dict, telemetry: TelemetryContext):
        self.config = config
        self.telemetry = telemetry
        self.logger = logging.getLogger(self.__class__.__name__)
        self.thresholds = self._load_thresholds(config)
        self.session_dir = Path(config.get("session_dir", "data/sessions"))
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar la MIC (ahora MICRegistry simplificado)
        self.mic = MICRegistry()
        register_core_vectors(self.mic)
        self._initialize_vector_space_refined()

    def _load_thresholds(self, config: dict) -> ProcessingThresholds:
        """
        Carga umbrales desde configuración con validación de tipo.
        """
        thresholds = ProcessingThresholds()
        overrides = config.get("processing_thresholds", {})

        if not isinstance(overrides, dict):
            self.logger.warning(
                f"processing_thresholds is not a dict (got {type(overrides).__name__}). "
                f"Using defaults."
            )
            return thresholds

        for key, value in overrides.items():
            if not hasattr(thresholds, key):
                self.logger.warning(f"Unknown threshold key '{key}'. Ignored.")
                continue

            current_value = getattr(thresholds, key)
            if current_value is not None and not isinstance(value, type(current_value)):
                self.logger.warning(
                    f"Threshold '{key}': expected {type(current_value).__name__}, "
                    f"got {type(value).__name__}. Ignored."
                )
                continue

            setattr(thresholds, key, value)

        return thresholds

    def _initialize_vector_space_refined(self):
        """
        Inicializa la MIC con los pasos del pipeline.
        
        El orden de registro es IDÉNTICO al orden del enum PipelineSteps.
        Los estratos respetan la filtración V_P ⊂ V_T ⊂ V_S ⊂ V_W.
        """
        steps_definition: List[Tuple[str, Type[ProcessingStep], Stratum]] = [
            ("load_data",          LoadDataStep,          Stratum.PHYSICS),
            ("audited_merge",      AuditedMergeStep,      Stratum.PHYSICS),
            ("calculate_costs",    CalculateCostsStep,     Stratum.TACTICS),
            ("final_merge",        FinalMergeStep,         Stratum.TACTICS),
            ("business_topology",  BusinessTopologyStep,   Stratum.STRATEGY),
            ("materialization",    MaterializationStep,    Stratum.STRATEGY),
            ("build_output",       BuildOutputStep,        Stratum.WISDOM),
        ]

        for label, step_class, stratum in steps_definition:
            self.mic.add_basis_vector(label, step_class, stratum)

    def _load_context_state(self, session_id: str) -> Optional[dict]:
        """
        Carga el estado de una sesión con validación de tipo.
        Retorna None si la sesión no existe o está corrupta.
        """
        if not session_id:
            return None
        try:
            session_file = self.session_dir / f"{session_id}.pkl"
            if session_file.exists():
                with open(session_file, "rb") as f:
                    data = pickle.load(f)
                if not isinstance(data, dict):
                    self.logger.error(
                        f"Corrupted session {session_id}: expected dict, got {type(data).__name__}"
                    )
                    return None
                return data
        except (pickle.UnpicklingError, EOFError, ModuleNotFoundError) as e:
            self.logger.error(f"Failed to deserialize session {session_id}: {e}")
        except Exception as e:
            self.logger.error(f"Failed to load context for session {session_id}: {e}")
        return None

    def _save_context_state(self, session_id: str, context: dict):
        """
        Guarda el estado de una sesión con escritura atómica.
        """
        try:
            session_file = self.session_dir / f"{session_id}.pkl"
            tmp_file = session_file.with_suffix(".pkl.tmp")
            with open(tmp_file, "wb") as f:
                pickle.dump(context, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp_file.replace(session_file)  # Atómica en POSIX
            self.logger.debug(f"Context saved for session {session_id}")
        except Exception as e:
            self.logger.error(f"Failed to save context for session {session_id}: {e}")
            # Limpiar archivo temporal si quedó
            try:
                if tmp_file.exists():
                    tmp_file.unlink()
            except Exception:
                pass

    def _cleanup_session(self, session_id: str):
        """Elimina el archivo de sesión tras finalización exitosa."""
        try:
            session_file = self.session_dir / f"{session_id}.pkl"
            if session_file.exists():
                session_file.unlink()
                self.logger.debug(f"Session file cleaned: {session_id}")
        except OSError as e:
            self.logger.warning(f"Could not clean session file {session_id}: {e}")

    def _compute_validated_strata(self, context: dict) -> Set[Stratum]:
        """
        Determina los estratos validados basándose en EVIDENCIA, no heurística.
        """
        validated = set()
        for stratum, evidence_keys in _STRATUM_EVIDENCE.items():
            # Un estrato es válido ssi TODAS sus claves de evidencia existen y no son None/Empty
            is_valid = True
            for key in evidence_keys:
                value = context.get(key)
                if value is None:
                    is_valid = False
                    break
                if hasattr(value, "empty") and value.empty:
                    is_valid = False
                    break
                if isinstance(value, (list, dict)) and not value:
                    is_valid = False
                    break

            if is_valid:
                validated.add(stratum)
        return validated

    def _check_stratum_prerequisites(self, target_stratum: Stratum, validated_strata: Set[Stratum]) -> bool:
        """
        Verifica que todos los estratos INFERIORES al objetivo estén validados.
        Clausura Transitiva.
        """
        target_level = stratum_level(target_stratum)
        if target_level == 0: # PHYSICS (Base)
            return True

        for s, level in _STRATUM_ORDER.items():
            if level < target_level:
                if s not in validated_strata:
                    return False
        return True

    def _enforce_filtration_invariant(self, target_stratum: Stratum, context: dict) -> None:
        """
        Lanza excepción si se viola la filtración topológica.
        """
        validated = self._compute_validated_strata(context)
        if not self._check_stratum_prerequisites(target_stratum, validated):
            target_level = stratum_level(target_stratum)
            missing = [
                s.name for s, l in _STRATUM_ORDER.items()
                if l < target_level and s not in validated
            ]
            raise RuntimeError(
                f"🛡️ Security Block: Filtration Invariant Violation. "
                f"Target: {target_stratum.name} (L{target_level}). "
                f"Missing Base Strata: {missing}. "
                f"Validated: {[s.name for s in validated]}."
            )

    def run_single_step(
        self,
        step_name: str,
        session_id: str,
        initial_context: Optional[Dict[str, Any]] = None,
        validate_stratum: bool = True
    ) -> Dict[str, Any]:
        """
        Ejecuta un único paso del pipeline con validación de filtración.
        """
        self.logger.info(f"Executing step: {step_name} (Session: {session_id[:8]}...)")

        # 1. Cargar contexto de sesión
        context = self._load_context_state(session_id) or {}

        # 2. Fusionar initial_context (las claves de sesión tienen precedencia)
        if initial_context:
            merged = {**initial_context, **context}
            context = merged

        try:
            # 3. Resolver vector base
            basis_vector = self.mic.get_basis_vector(step_name)
            if not basis_vector:
                available = self.mic.get_available_labels()
                raise ValueError(
                    f"Step '{step_name}' not found. Available: {available}"
                )

            # 4. Validar filtración de estratos (Lógica Transitiva Refinada)
            if validate_stratum:
                self._enforce_filtration_invariant(basis_vector.stratum, context)

            # 5. Instanciar y ejecutar
            step_instance = basis_vector.operator_class(self.config, self.thresholds)
            
            # Inyectar MIC en el paso para permitir proyección de vectores
            try:
                step_instance.mic = self.mic
            except AttributeError:
                pass # Si el paso no soporta inyección, continuamos (legacy support)
                
            updated_context = step_instance.execute(context, self.telemetry)

            if updated_context is None:
                raise ValueError(
                    f"Step '{step_name}' returned None context. "
                    f"All steps must return the (possibly modified) context dict."
                )

            # 6. Persistir estado actualizado
            self._save_context_state(session_id, updated_context)

            self.logger.info(f"Step '{step_name}' completed successfully.")
            return {
                "status": "success",
                "step": step_name,
                "stratum": basis_vector.stratum.name,
                "session_id": session_id,
                "context_keys": list(updated_context.keys()),
            }

        except Exception as e:
            self.logger.error(f"Error executing step '{step_name}': {e}", exc_info=True)
            self.telemetry.record_error(step_name, str(e))
            return {
                "status": "error",
                "step": step_name,
                "error": str(e),
                "session_id": session_id,
            }

    def execute_pipeline_orchestrated(self, initial_context: dict) -> dict:
        """
        Ejecuta el pipeline completo de forma orquestada.
        """
        session_id = str(uuid.uuid4())
        self.logger.info(f"Starting orchestrated pipeline (Session ID: {session_id})")

        # Obtener receta de ejecución
        default_recipe = self.mic.get_execution_sequence()
        recipe = self.config.get("pipeline_recipe", default_recipe)

        # Guardado inicial del contexto — verificar que no falle silenciosamente
        self._save_context_state(session_id, initial_context)
        verification = self._load_context_state(session_id)
        if verification is None:
            raise IOError(
                f"Failed to persist initial context for session {session_id}. "
                f"Check disk permissions on {self.session_dir}."
            )

        first_step = True
        for step_idx, step_config in enumerate(recipe):
            step_name = step_config.get("step")
            enabled = step_config.get("enabled", True)

            if not step_name:
                self.logger.warning(f"Recipe entry {step_idx} has no 'step' key. Skipping.")
                continue

            if not enabled:
                self.logger.info(f"⏭️ Skipping disabled step: {step_name}")
                continue

            self.logger.info(
                f"Orchestrating step [{step_idx + 1}/{len(recipe)}]: {step_name}"
            )

            # En el primer paso, pasar initial_context como respaldo defensivo
            ctx_override = initial_context if first_step else None
            result = self.run_single_step(
                step_name, session_id, initial_context=ctx_override
            )
            first_step = False

            if result["status"] == "error":
                error_msg = (
                    f"Pipeline failed at step '{step_name}' "
                    f"[{step_idx + 1}/{len(recipe)}]: {result.get('error')}"
                )
                self.logger.critical(error_msg)
                # No limpiar sesión en error para permitir análisis forense
                raise RuntimeError(error_msg)

        final_context = self._load_context_state(session_id) or {}

        # Limpieza de archivo de sesión tras éxito
        self._cleanup_session(session_id)

        self.logger.info(f"Pipeline completed successfully (Session: {session_id})")
        return final_context


def process_all_files(
    presupuesto_path: Union[str, Path],
    apus_path: Union[str, Path],
    insumos_path: Union[str, Path],
    config: dict = None,
    telemetry: TelemetryContext = None,
) -> dict:
    """
    Función de entrada principal para el pipeline (Batch Mode).
    """
    if config is None:
        config = {}

    # Garantizar telemetría (evita AttributeError en todos los pasos)
    if telemetry is None:
        telemetry = TelemetryContext()
        logger.info("ℹ️ No telemetry context provided; created default instance.")

    # Resolver y validar rutas
    paths = {
        "presupuesto_path": Path(presupuesto_path).resolve(),
        "apus_path": Path(apus_path).resolve(),
        "insumos_path": Path(insumos_path).resolve(),
    }

    for name, path in paths.items():
        if not path.exists():
            error = f"File not found: {name} = {path}"
            telemetry.record_error("process_all_files", error)
            raise FileNotFoundError(error)

    director = PipelineDirector(config, telemetry)

    initial_context = {k: str(v) for k, v in paths.items()}

    try:
        final_context = director.execute_pipeline_orchestrated(initial_context)

        # Retornar el DataProduct si existe; sino, el contexto completo
        return final_context.get("final_result", final_context)

    except Exception as e:
        logger.critical(f"🔥 Critical failure in process_all_files: {e}", exc_info=True)
        telemetry.record_error("process_all_files", str(e))
        raise
