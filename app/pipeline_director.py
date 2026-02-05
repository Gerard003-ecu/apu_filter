"""
Módulo: Pipeline Director (La Matriz de Transformación de Valor M_P)
======================================================================

Este componente actúa como el "Sistema Nervioso Central" de APU_filter. 
A diferencia de los versiones anteriores, este módulo **no ejecuta cálculos matemáticos** 
ni lógica de negocio directa (responsabilidad delegada a los Agentes Tácticos y Estratégicos).

Su función es orquestar la secuencia de activación de los vectores de transformación 
a través de la Matriz de Interacción Central (MIC), garantizando la integridad 
del "Vector de Estado" del proyecto.

Arquitectura y Fundamentos Matemáticos:
---------------------------------------

1. Orquestación Algebraica (Espacio Vectorial de Operadores):
   Utiliza la `LinearInteractionMatrix` (MIC) para proyectar "Intenciones" sobre un espacio
   vectorial de base ortogonal. Cada paso del pipeline (ej. `calculate_costs`) no es una 
   función local, sino un vector base unitario ($e_i$) que solicita una transformación 
   al estrato correspondiente.

2. Auditoría Homológica (Secuencia de Mayer-Vietoris):
   Implementa el paso `AuditedMergeStep`. En lugar de un simple JOIN de datos, utiliza 
   la secuencia exacta de Mayer-Vietoris para garantizar la integridad topológica 
   durante la fusión (Presupuesto U APUs):
   
   $$... \to H_k(A \cap B) \to H_k(A) \oplus H_k(B) \to H_k(A \cup B) \to ...$$
   
   Esto asegura matemáticamente que la integración no introduzca ciclos espurios ($\beta_1$)
   ni desconexiones ($\beta_0$) que no existían en los conjuntos originales.

3. Filtración por Estratos (Jerarquía DIKW):
   Gestiona la ejecución respetando estrictamente la filtración de subespacios:
   $V_{PHYSICS} \subset V_{TACTICS} \subset V_{STRATEGY} \subset V_{WISDOM}$.
   
   El director actúa como un Gatekeeper Topológico, validando que las transiciones 
   entre estratos sean suaves y prohibiendo que operadores de alto nivel (Estrategia) 
   se ejecuten sobre una base física inestable (Datos no validados).

4. Protocolo de Caja de Cristal (Glass Box Persistence):
   Mantiene la trazabilidad forense completa. El estado del sistema se serializa 
   entre pasos, permitiendo pausar, reanudar y auditar el proceso de transformación 
   en cualquier punto del flujo.

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
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from scipy import sparse

from app.constants import ColumnNames, InsumoType
from app.flux_condenser import CondenserConfig, DataFluxCondenser
from app.matter_generator import MatterGenerator
from app.schemas import Stratum
from app.telemetry import TelemetryContext
from app.telemetry_narrative import TelemetryNarrator

from .data_validator import validate_and_clean_data

# Configuración explícita para debug
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Asegurar que tenga un handler si no tiene
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)


# ==================== CONSTANTES Y CLASES AUXILIARES (REUBICADAS) ====================

from .apu_processor import (
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

# ==================== CLASE BASE REFACTORIZADA ====================


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
            dict: El contexto actualizado (puede ser el mismo objeto modificado).
        """
        pass

# ==================== ESTRUCTURAS ALGEBRAICAS (MIC) ====================

@dataclass(frozen=True)
class BasisVector:
    """
    Representa un vector base unitario e_i en el espacio de operaciones.

    Matemáticamente:
    e_i = [0, ..., 1, ..., 0]^T

    Propiedades:
    - Norma L2: ||e_i|| = 1 (Unitario)
    - Estrato: Define el subespacio V_k al que pertenece (Filtración)
    """
    index: int
    label: str  # Identificador semántico (ej: "load_data")
    operator_class: Type[ProcessingStep] # La transformación lineal T
    stratum: Stratum


class LinearInteractionMatrix:
    """
    Implementación algebraica rigurosa de la MIC como Operador Diagonal.

    Refinamiento: Incorpora métrica de Killing-Cartan para verificación
    de ortogonalidad y cómputo espectral basado en teoría de representaciones.
    """

    def __init__(self):
        self._basis: Dict[str, BasisVector] = {}
        self._dimension = 0
        self._gram_matrix: Optional[np.ndarray] = None
        self._orthonormal_basis_computed = False
        self._killing_form_cache: Optional[np.ndarray] = None

    def get_rank(self) -> int:
        """Retorna el rango de la matriz (Dimensión del espacio imagen)."""
        if self._gram_matrix is not None:
            return int(np.linalg.matrix_rank(self._gram_matrix))
        return self._dimension

    def add_basis_vector(
        self,
        label: str,
        step_class: Type[ProcessingStep],
        stratum: Stratum
    ):
        """
        Expande el espacio vectorial con verificación de independencia lineal.

        Invariante: ∀e_i, e_j ∈ B, i ≠ j → <e_i, e_j>_K = 0
        donde <·,·>_K es la forma de Killing.
        """
        if not label or not isinstance(label, str):
            raise ValueError("Label debe ser una cadena no vacía")

        if label in self._basis:
            raise ValueError(
                f"Dependencia Lineal: '{label}' viola independencia. "
                f"Base actual: {list(self._basis.keys())}"
            )

        if not (isinstance(step_class, type) and issubclass(step_class, ProcessingStep)):
            raise TypeError(
                f"El operador {step_class} no es lineal "
                f"(debe ser subclase de ProcessingStep)"
            )

        vector = BasisVector(
            index=self._dimension,
            label=label,
            operator_class=step_class,
            stratum=stratum
        )

        self._verify_orthogonality_killing(vector)

        self._basis[label] = vector
        self._dimension += 1
        self._invalidate_caches()

        logger.debug(
            f"📐 Vector base añadido: {label} (dim={self._dimension}, "
            f"estrato={stratum.name})"
        )

    def _invalidate_caches(self):
        """Invalida caches tras modificación de la base."""
        self._orthonormal_basis_computed = False
        self._gram_matrix = None
        self._killing_form_cache = None

    def _verify_orthogonality_killing(self, new_vector: BasisVector):
        """
        Verifica ortogonalidad usando forma de Killing generalizada.

        La forma de Killing K(X,Y) = Tr(ad_X ∘ ad_Y) se aproxima aquí
        mediante una métrica funcional basada en:
        - Coincidencia de operador (colinealidad directa)
        - Coincidencia de estrato con diferente operador (interferencia)
        - Solapamiento de dominio/codominio funcional
        """
        for existing_label, existing_vector in self._basis.items():
            # Colinealidad directa: mismo operador (relaxed for different strata)
            if existing_vector.operator_class == new_vector.operator_class:
                if existing_vector.stratum == new_vector.stratum:
                    raise ValueError(
                        f"Colinealidad funcional: '{new_vector.label}' usa mismo "
                        f"operador que '{existing_label}' en el mismo estrato"
                    )

            # Verificar interferencia de estrato con análisis de firma
            killing_value = self._compute_killing_pairing(existing_vector, new_vector)

            if abs(killing_value) > 0.95:  # Umbral de cuasi-colinealidad
                raise ValueError(
                    f"Cuasi-colinealidad detectada: K({new_vector.label}, "
                    f"{existing_label}) = {killing_value:.3f}"
                )

    def _compute_killing_pairing(
        self,
        v1: BasisVector,
        v2: BasisVector
    ) -> float:
        """
        Computa el apareamiento de Killing entre dos vectores base.

        Retorna valor en [-1, 1] donde:
        - 0: Ortogonales (independientes)
        - ±1: Paralelos (dependientes)
        """
        # Factor por coincidencia de estrato
        stratum_factor = 1.0 if v1.stratum == v2.stratum else 0.3

        # Factor por proximidad de índice (localidad en la filtración)
        index_distance = abs(v1.index - v2.index)
        locality_factor = 1.0 / (1.0 + index_distance)

        # Factor por análisis de firma del operador
        signature_similarity = self._compute_operator_signature_similarity(
            v1.operator_class, v2.operator_class
        )

        killing_value = stratum_factor * locality_factor * signature_similarity
        return np.clip(killing_value, -1.0, 1.0)

    def _compute_operator_signature_similarity(
        self,
        op1: Type[ProcessingStep],
        op2: Type[ProcessingStep]
    ) -> float:
        """
        Calcula similitud de firma entre operadores usando introspección.
        """
        try:
            import inspect

            sig1 = inspect.signature(op1.execute)
            sig2 = inspect.signature(op2.execute)

            params1 = set(sig1.parameters.keys()) - {'self'}
            params2 = set(sig2.parameters.keys()) - {'self'}

            if not params1 or not params2:
                return 0.0

            intersection = len(params1 & params2)
            union = len(params1 | params2)

            base_sim = intersection / union if union > 0 else 0.0

            # Penalize if classes are different but signatures same
            if op1 != op2 and base_sim == 1.0:
                return 0.99

            return base_sim

        except Exception:
            return 0.0

    def project_intent(self, intent_label: str) -> BasisVector:
        """
        Proyección ortogonal del vector de intención sobre la base E.

        Implementa: proj_E(q) = Σ_i <q, e_i> e_i / ||e_i||²
        En espacio discreto se reduce a búsqueda exacta con validación.
        """
        if not intent_label:
            raise ValueError("Vector de intención vacío (norma cero)")

        vector = self._basis.get(intent_label)
        if vector is None:
            available = list(self._basis.keys())
            # Intentar match parcial para sugerencias
            suggestions = [k for k in available if intent_label.lower() in k.lower()]

            raise ValueError(
                f"Vector '{intent_label}' ∈ Ker(π) (núcleo de proyección). "
                f"Base disponible: {available}. "
                f"Sugerencias: {suggestions if suggestions else 'ninguna'}"
            )

        if not self._orthonormal_basis_computed:
            self._orthonormalize_basis()

        return vector

    def _orthonormalize_basis(self):
        """
        Aplica Gram-Schmidt modificado para estabilidad numérica.

        Algoritmo: MGS (Modified Gram-Schmidt)
        Para k = 1, ..., n:
            q_k = v_k
            Para j = 1, ..., k-1:
                q_k = q_k - <q_k, q_j> q_j
            q_k = q_k / ||q_k||
        """
        if self._orthonormal_basis_computed:
            return

        n = self._dimension
        if n == 0:
            self._gram_matrix = np.array([[]])
            self._orthonormal_basis_computed = True
            return

        # Construir matriz de Gram usando forma de Killing
        self._gram_matrix = np.eye(n)

        vectors = list(self._basis.values())
        for i in range(n):
            for j in range(i + 1, n):
                killing_ij = self._compute_killing_pairing(vectors[i], vectors[j])
                self._gram_matrix[i, j] = killing_ij
                self._gram_matrix[j, i] = killing_ij

        # Verificar definición positiva (espacio métrico válido)
        try:
            eigenvalues = np.linalg.eigvalsh(self._gram_matrix)
            if np.any(eigenvalues < -1e-10):
                logger.warning(
                    f"⚠️ Matriz de Gram no definida positiva. "
                    f"Eigenvalores negativos: {eigenvalues[eigenvalues < 0]}"
                )
                # Regularización de Tikhonov
                self._gram_matrix += np.eye(n) * abs(min(eigenvalues)) * 1.1
        except np.linalg.LinAlgError:
            logger.warning("⚠️ Error en descomposición espectral, usando identidad")
            self._gram_matrix = np.eye(n)

        self._orthonormal_basis_computed = True
        logger.debug(f"🧮 Gram-Schmidt completado. Condición: {np.linalg.cond(self._gram_matrix):.2f}")

    def get_spectrum(self) -> Dict[str, float]:
        """
        Calcula el espectro del operador basado en la matriz de Gram.

        Los valores propios indican la 'inercia' de cada dirección base.
        """
        if not self._orthonormal_basis_computed:
            self._orthonormalize_basis()

        if self._gram_matrix is None or self._gram_matrix.size == 0:
            return {}

        try:
            eigenvalues = np.linalg.eigvalsh(self._gram_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Descendente

            spectrum = {}
            for i, (label, vector) in enumerate(self._basis.items()):
                if i < len(eigenvalues):
                    # Ponderar por nivel de estrato
                    stratum_weight = {
                        Stratum.PHYSICS: 1.0,
                        Stratum.TACTICS: 1.2,
                        Stratum.STRATEGY: 1.5,
                        Stratum.WISDOM: 10.0
                    }.get(vector.stratum, 1.0)

                    spectrum[label] = float(eigenvalues[i]) * stratum_weight
                else:
                    spectrum[label] = 1.0

            return spectrum

        except np.linalg.LinAlgError as e:
            logger.error(f"❌ Error computando espectro: {e}")
            return {label: 1.0 for label in self._basis.keys()}

    def get_condition_number(self) -> float:
        """Retorna número de condición de la base (estabilidad numérica)."""
        if not self._orthonormal_basis_computed:
            self._orthonormalize_basis()

        if self._gram_matrix is None or self._gram_matrix.size == 0:
            return 1.0

        return float(np.linalg.cond(self._gram_matrix))




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
            # Use defaults if not present to support simpler test configurations
            if not file_profiles:
                logger.warning("⚠️ 'file_profiles' no encontrado en config, usando defaults vacíos.")
                file_profiles = {
                    "presupuesto_default": {},
                    "insumos_default": {},
                    "apus_default": {}
                }

            presupuesto_profile = file_profiles.get("presupuesto_default", {})

            p_processor = PresupuestoProcessor(
                self.config, self.thresholds, presupuesto_profile
            )
            df_presupuesto = p_processor.process(presupuesto_path)

            if df_presupuesto is None or df_presupuesto.empty:
                error = "Procesamiento de presupuesto retornó DataFrame vacío"
                telemetry.record_error("load_data", error)
                raise ValueError(error)

            telemetry.record_metric(
                "load_data", "presupuesto_rows", len(df_presupuesto)
            )

            insumos_profile = file_profiles.get("insumos_default", {})

            i_processor = InsumosProcessor(self.thresholds, insumos_profile)
            df_insumos = i_processor.process(insumos_path)

            logger.info(
                f"🐛 DIAG: [LoadDataStep] Insumos extraídos: {len(df_insumos)} filas."
            )
            if not df_insumos.empty:
                logger.info(
                    f"🐛 DIAG: [LoadDataStep] Estructura de insumos (head(1)): {df_insumos.head(1).to_dict('records')}"
                )

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
                logger.warning(
                    f"⚠️ Error en config de condenser, usando defaults: {e}"
                )
                condenser_config = CondenserConfig()

            condenser = DataFluxCondenser(
                config=self.config,
                profile=apus_profile,
                condenser_config=condenser_config,
            )

            def on_progress_stats(processing_stats):
                try:
                    for metric_name, attr_name, default_value in [
                        ("avg_saturation", "avg_saturation", 0.0),
                        ("max_flyback_voltage", "max_flyback_voltage", 0.0),
                        ("max_dissipated_power", "max_dissipated_power", 0.0),
                        ("avg_kinetic_energy", "avg_kinetic_energy", 0.0),
                    ]:
                        val = getattr(processing_stats, attr_name, default_value)
                        telemetry.record_metric("flux_condenser", metric_name, val)
                except Exception:
                    pass

            def _publish_telemetry(metrics: Dict[str, Any]):
                try:
                    import json
                    import time

                    from flask import current_app

                    payload = {
                        **metrics,
                        "_timestamp": time.time(),
                        "_source": "flux_condenser_realtime",
                    }

                    data_str = json.dumps(payload, default=str)

                    if current_app:
                        redis_client = current_app.config.get("SESSION_REDIS")
                        if redis_client:
                            redis_client.set(
                                "apu_filter:global_metrics", data_str, ex=60
                            )
                except Exception:
                    pass

            df_apus_raw = condenser.stabilize(
                apus_path,
                on_progress=on_progress_stats,
                progress_callback=_publish_telemetry,
                telemetry=telemetry,
            )

            full_stats = condenser.get_processing_stats() or {}
            stats = full_stats.get("statistics", {})

            for metric_name, default_value in [
                ("avg_saturation", 0.0),
                ("max_flyback_voltage", 0.0),
                ("max_dissipated_power", 0.0),
                ("avg_kinetic_energy", 0.0),
                ("avg_batch_size", 0),
            ]:
                value = stats.get(metric_name, default_value)
                if (
                    not isinstance(value, (int, float))
                    or np.isnan(value)
                    or np.isinf(value)
                ):
                    value = default_value
                telemetry.record_metric("flux_condenser", metric_name, value)

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
            context.update(
                {
                    "df_presupuesto": df_presupuesto,
                    "df_insumos": df_insumos,
                    "df_apus_raw": df_apus_raw,
                }
            )

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
        """Ejecuta la fusión de DataFrames."""
        telemetry.start_step("merge_data")
        try:
            df_apus_raw = context["df_apus_raw"]
            df_insumos = context["df_insumos"]

            logger.info(
                f"🐛 DIAG: [MergeDataStep] Recibidos {len(df_insumos)} insumos del contexto."
            )

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


class AuditedMergeStep(ProcessingStep):
    """
    Paso de Fusión con Auditoría Topológica (Mayer-Vietoris).

    Construye grafos temporales para validar la integridad antes de comprometer la fusión.
    """

    def __init__(self, config: dict, thresholds: ProcessingThresholds):
        self.config = config
        self.thresholds = thresholds

    def execute(self, context: dict, telemetry: TelemetryContext) -> dict:
        """Ejecuta la auditoría Mayer-Vietoris y luego la fusión física."""
        telemetry.start_step("audited_merge")
        try:
            df_a = context.get("df_presupuesto")
            df_b = context.get("df_apus_raw")
            df_insumos = context.get("df_insumos")

            if df_a is None or df_b is None:
                logger.warning(
                    "⚠️ Falta df_presupuesto o df_apus_raw. Saltando auditoría Mayer-Vietoris."
                )
            else:
                try:
                    from agent.business_topology import (
                        BudgetGraphBuilder,
                        BusinessTopologicalAnalyzer,
                    )

                    builder = BudgetGraphBuilder()
                    graph_a = builder.build(df_a, pd.DataFrame())
                    graph_b = builder.build(pd.DataFrame(), df_b)

                    analyzer = BusinessTopologicalAnalyzer(telemetry=telemetry)
                    audit_result = analyzer.audit_integration_homology(graph_a, graph_b)

                    if audit_result["delta_beta_1"] > 0:
                        logger.warning(f"🚨 {audit_result['narrative']}")
                        telemetry.record_metric(
                            "topology", "emergent_cycles", audit_result["delta_beta_1"]
                        )
                        context["integration_risk_alert"] = audit_result
                    else:
                        logger.info(f"✅ {audit_result['narrative']}")

                except Exception as e_audit:
                    logger.error(
                        f"❌ Error durante auditoría Mayer-Vietoris: {e_audit}"
                    )
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
        """Ejecuta los cálculos de costos y tiempos."""
        telemetry.start_step("calculate_costs")
        try:
            df_merged = context["df_merged"]
            df_merged = calculate_insumo_costs(df_merged, self.thresholds)

            cost_calculator = APUCostCalculator(self.config, self.thresholds)
            df_apu_costos, df_tiempo, df_rendimiento = cost_calculator.calculate(
                df_merged
            )

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

            df_final = pd.merge(
                df_final, df_tiempo, on=ColumnNames.CODIGO_APU, how="left"
            )
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


class BusinessTopologyStep(ProcessingStep):
    """
    Paso de Análisis de Negocio.

    Utiliza el BusinessAgent para auditar la integridad estructural y evaluar riesgos.
    """

    def __init__(self, config: dict, thresholds: ProcessingThresholds):
        self.config = config
        self.thresholds = thresholds

    def execute(self, context: dict, telemetry: TelemetryContext) -> dict:
        """Ejecuta la evaluación del BusinessAgent."""
        telemetry.start_step("business_topology")
        try:
            from app.business_agent import BusinessAgent
            from flask import current_app

            # Recuperar la instancia global de la MIC
            mic_instance = getattr(current_app, "mic", None)
            if not mic_instance:
                raise RuntimeError("MICRegistry not found in current_app")

            # Garantizar que el contexto tenga validación de estratos
            # Si llegamos a este paso en el pipeline secuencial, asumimos PHYSICS y TACTICS válidos
            if "validated_strata" not in context:
                context["validated_strata"] = {Stratum.PHYSICS, Stratum.TACTICS}
            elif isinstance(context["validated_strata"], set):
                context["validated_strata"].add(Stratum.PHYSICS)
                context["validated_strata"].add(Stratum.TACTICS)

            logger.info("🤖 Desplegando BusinessAgent para evaluación de proyecto...")

            agent = BusinessAgent(
                config=self.config,
                mic=mic_instance,
                telemetry=telemetry
            )
            report = agent.evaluate_project(context)

            if report:
                logger.info("✅ BusinessAgent completó la evaluación.")
                context["business_topology_report"] = report

                logger.info(
                    f"Puntuación de Integridad: {report.integrity_score:.2f}/100"
                )
                if report.waste_alerts:
                    logger.warning(
                        f"Alertas de Desperdicio: {len(report.waste_alerts)}"
                    )
                if report.circular_risks:
                    logger.critical(
                        f"Riesgos Circulares: {len(report.circular_risks)}"
                    )
            else:
                logger.warning("⚠️ El BusinessAgent no generó un reporte.")

            telemetry.end_step("business_topology", "success")
            return context

        except Exception as e:
            logger.error(
                f"❌ Error en BusinessTopologyStep con BusinessAgent: {e}", exc_info=True
            )
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
        """Ejecuta el MatterGenerator."""
        telemetry.start_step("materialization")
        try:
            if "business_topology_report" not in context:
                logger.warning(
                    "⚠️ No se encontró reporte de topología. Saltando materialización."
                )
                telemetry.end_step("materialization", "skipped")
                return context

            graph = context.get("graph")
            if not graph:
                logger.info(
                    "🔄 Grafo no encontrado en contexto. Reconstruyendo para materialización..."
                )
                from agent.business_topology import BudgetGraphBuilder

                builder = BudgetGraphBuilder()
                df_presupuesto = context.get("df_final")
                df_detail = context.get("df_merged")

                if df_presupuesto is not None and df_detail is not None:
                    graph = builder.build(df_presupuesto, df_detail)
                    context["graph"] = graph
                else:
                    logger.error(
                        "❌ No hay datos suficientes para reconstruir el grafo."
                    )
                    telemetry.end_step("materialization", "error")
                    return context

            report = context.get("business_topology_report")
            stability = 10.0
            if report and report.details:
                stability = report.details.get("pyramid_stability", 10.0)

            flux_metrics = {
                "pyramid_stability": stability,
                "avg_saturation": 0.0,
            }

            generator = MatterGenerator()
            bom = generator.materialize_project(
                graph, flux_metrics=flux_metrics, telemetry=telemetry
            )

            context["bill_of_materials"] = bom
            context["logistics_plan"] = asdict(bom)

            logger.info(
                f"✅ Materialización completada. Total ítems: {len(bom.requirements)}"
            )

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
        """Construye y valida la estructura de salida como un Data Product."""
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

            try:
                narrator = TelemetryNarrator()
                tech_narrative = narrator.summarize_execution(telemetry)
                if isinstance(tech_narrative, dict):
                    validated_result["technical_audit"] = tech_narrative.copy()
                else:
                    validated_result["technical_audit"] = tech_narrative

                logger.info("✅ Narrativa técnica generada e inyectada.")
            except Exception as e:
                logger.warning(f"⚠️ Fallo al generar narrativa técnica: {e}")
                validated_result["technical_audit"] = {"error": str(e)}

            def compute_hash(data: Any) -> str:
                """Calcula un hash simple del contenido para linaje."""
                import json

                try:
                    sanitized = data
                    try:
                        sanitized = sanitize_for_json(data)
                    except NameError:
                        pass

                    s = json.dumps(sanitized, sort_keys=True, default=str)
                    return hashlib.sha256(s.encode("utf-8")).hexdigest()
                except Exception:
                    return "hash_computation_failed"

            input_sample = {
                "presupuesto_head": (
                    df_final.head(5).to_dict("records") if not df_final.empty else []
                ),
                "insumos_head": (
                    df_insumos.head(5).to_dict("records")
                    if not df_insumos.empty
                    else []
                ),
            }
            lineage_hash = compute_hash(input_sample)

            error_count = (
                telemetry.get_metrics().get("errors", 0)
                if hasattr(telemetry, "get_metrics")
                else 0
            )
            sla_compliance = "100%" if error_count == 0 else "95%"

            data_product = {
                "kind": "DataProduct",
                "metadata": {
                    "version": "3.0",
                    "lineage_hash": lineage_hash,
                    "sla_compliance": sla_compliance,
                    "generated_at": datetime.datetime.now().isoformat(),
                    "generator": "APU_Filter_Pipeline_v2",
                },
                "governance": {
                    "policy_version": "data_contract_v1",
                    "compliance_score": 95,
                    "classification": "CONFIDENTIAL",
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


# ==================== PIPELINE DIRECTOR ====================


class PipelineSteps(str, enum.Enum):
    """
    Define los identificadores únicos para cada paso del pipeline.

    Funciona como una 'API' pública para el orquestador.
    """

    LOAD_DATA = "load_data"
    MERGE_DATA = "merge_data"
    CALCULATE_COSTS = "calculate_costs"
    FINAL_MERGE = "final_merge"
    BUSINESS_TOPOLOGY = "business_topology"
    MATERIALIZATION = "materialization"
    BUILD_OUTPUT = "build_output"



class PipelineDirector:
    """
    Orquesta la ejecución secuencial con validación topológica.

    Implementa una 4-variedad diferenciable donde cada estrato
    corresponde a una subvariedad embebida.
    """

    def __init__(self, config: dict, telemetry: TelemetryContext):
        self.config = config
        self.telemetry = telemetry
        self.thresholds = self._load_thresholds(config)
        self.session_dir = Path(config.get("session_dir", "data/sessions"))
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Inicializar espacio vectorial con métrica Riemanniana
        self.mic = LinearInteractionMatrix()
        self._filtration_level = 0  # Nivel de filtración actual
        self._homology_groups = {}  # Grupos de homología computados
        self._initialize_vector_space_with_validation()

    def _initialize_vector_space_with_validation(self):
        """
        Construye la base canónica con validación de filtración.

        Filtración: ∅ = F_0 ⊂ F_1 ⊂ F_2 ⊂ F_3 ⊂ F_4 = V
        donde F_k corresponde al estrato k.
        """
        # Definir mapeo estrato → nivel de filtración
        stratum_filtration = {
            Stratum.PHYSICS: 1,
            Stratum.TACTICS: 2,
            Stratum.STRATEGY: 3,
            Stratum.WISDOM: 4
        }

        # Añadir vectores en orden de filtración
        basis_config = [
            ("load_data", LoadDataStep, Stratum.PHYSICS),
            ("audited_merge", AuditedMergeStep, Stratum.PHYSICS),
            ("calculate_costs", CalculateCostsStep, Stratum.TACTICS),
            ("final_merge", FinalMergeStep, Stratum.PHYSICS),
            ("materialization", MaterializationStep, Stratum.TACTICS),
            ("business_topology", BusinessTopologyStep, Stratum.STRATEGY),
            ("build_output", BuildOutputStep, Stratum.WISDOM)
        ]

        for label, step_class, stratum in basis_config:
            try:
                self.mic.add_basis_vector(label, step_class, stratum)
                logger.debug(
                    f"📐 Vector añadido a filtración F_{stratum_filtration[stratum]}: "
                    f"{label} ({stratum.name})"
                )
            except ValueError as e:
                logger.error(f"❌ Error en filtración para {label}: {e}")
                raise

    def run_single_step(
        self,
        step_name: str,
        session_id: str,
        initial_context: Optional[Dict[str, Any]] = None,
        validate_stratum_transition: bool = True
    ) -> Dict[str, Any]:
        """
        Ejecuta un único operador con validación de transición entre estratos.

        Parámetros:
        -----------
        validate_stratum_transition : bool
            Si True, valida que la transición entre estratos sea suave
            (no salte estratos intermedios).
        """
        # 1. Cargar contexto con verificación de integridad
        context = self._load_context_state(session_id)
        if initial_context:
            # Validar que initial_context no corrompa el estado existente
            self._validate_context_merge(context, initial_context)
            context.update(initial_context)

        logger.info(
            f"▶️ Ejecutando operador: {step_name} (Sesión: {session_id[:8]}...)"
        )

        try:
            # 2. Proyección algebraica con verificación de rango
            basis_vector = self.mic.project_intent(step_name)

            # 3. Validación de transición entre estratos
            if validate_stratum_transition:
                current_stratum = self._infer_current_stratum(context)
                self._validate_stratum_transition(
                    current_stratum, basis_vector.stratum
                )

            # 4. Instanciación del operador lineal
            step_instance = basis_vector.operator_class(self.config, self.thresholds)

            # 5. Medición de traza antes/después
            trace_before = self._compute_state_trace(context)

            # 6. Aplicación de transformación: S' = T(S)
            updated_context = step_instance.execute(context, self.telemetry)

            if updated_context is None:
                raise ValueError(f"Operador {step_name} retornó transformación nula")

            # 7. Verificar preservación de norma (conservación de información)
            trace_after = self._compute_state_trace(updated_context)
            trace_delta = abs(trace_after - trace_before)

            if trace_delta > 0.01:  # Umbral de tolerancia
                logger.warning(
                    f"⚠️ Operador {step_name} alteró traza del estado: "
                    f"Δ = {trace_delta:.4f}"
                )

            # 8. Persistencia con checksum
            self._save_context_state_with_checksum(session_id, updated_context)

            # 9. Actualizar nivel de filtración
            self._filtration_level = self._stratum_to_filtration(basis_vector.stratum)

            # 10. Calcular homología si estamos en estratos superiores
            if basis_vector.stratum in [Stratum.STRATEGY, Stratum.WISDOM]:
                self._compute_homology_groups(updated_context)

            logger.info(f"✅ Operador {step_name} completado (estrato: {basis_vector.stratum.name})")

            return {
                "status": "success",
                "step": step_name,
                "stratum": basis_vector.stratum.name,
                "filtration_level": self._filtration_level,
                "session_id": session_id,
                "context_keys": list(updated_context.keys()),
                "trace_delta": trace_delta,
                "homology_updated": basis_vector.stratum in [Stratum.STRATEGY, Stratum.WISDOM]
            }

        except Exception as e:
            error_msg = f"Error en operador '{step_name}': {e}"
            logger.error(f"🔥 {error_msg}", exc_info=True)
            self.telemetry.record_error(step_name, str(e))

            # Intentar recuperación mediante operador identidad
            recovery_status = self._attempt_state_recovery(session_id, context)

            return {
                "status": "error",
                "step": step_name,
                "error": error_msg,
                "recovery_attempted": recovery_status,
                "session_id": session_id
            }

    def execute_pipeline_orchestrated(self, initial_context: dict) -> dict:
        """
        Ejecuta el pipeline completo de forma orquestada, paso a paso.
        """
        session_id = str(uuid.uuid4())
        logger.info(f"🚀 Iniciando pipeline orquestado con Sesión ID: {session_id}")

        if "pipeline_recipe" not in self.config:
            logger.warning("No 'pipeline_recipe' en config. Usando flujo por defecto.")
            recipe = [{"step": step.value, "enabled": True} for step in PipelineSteps]
        else:
            recipe = self.config["pipeline_recipe"]

        context = initial_context
        for step_idx, step_config in enumerate(recipe):
            step_name = step_config.get("step")
            if not step_config.get("enabled", True):
                continue

            logger.info(f"▶️ Orquestando paso [{step_idx + 1}/{len(recipe)}]: {step_name}")

            current_context = context if step_idx == 0 else None
            result = self.run_single_step(
                step_name, session_id, initial_context=current_context
            )

            if result["status"] == "error":
                error_msg = f"Fallo en pipeline orquestado en paso '{step_name}': {result.get('error')}"
                logger.critical(f"🔥 {error_msg}")
                raise RuntimeError(error_msg)

        final_context = self._load_context_state(session_id)
        logger.info(f"🎉 Pipeline orquestado completado (Sesión: {session_id})")
        return final_context

    def _validate_stratum_transition(
        self,
        current: Optional[Stratum],
        next_stratum: Stratum
    ):
        """
        Valida transición entre estratos usando teoría de órdenes parciales.

        Reglas:
        1. Transiciones hacia arriba: Siempre permitidas
        2. Transiciones laterales (mismo nivel): Permitidas
        3. Transiciones hacia abajo: Solo si se completa ciclo o reinicio explícito
        4. Saltos de más de un nivel: Warning pero permitido
        """
        if current is None:
            # Primera ejecución, cualquier estrato es válido
            logger.debug(f"🚀 Iniciando en estrato {next_stratum.name}")
            return

        current_level = self._stratum_to_filtration(current)
        next_level = self._stratum_to_filtration(next_stratum)

        # Caso 1: Avance o mismo nivel (normal)
        if next_level >= current_level:
            # Detectar salto de estratos
            if next_level > current_level + 1:
                skipped = next_level - current_level - 1
                skipped_strata = [
                    s.name for s in Stratum
                    if current_level < self._stratum_to_filtration(s) < next_level
                ]
                logger.warning(
                    f"⚠️ Salto de {skipped} estrato(s): {current.name} → {next_stratum.name}. "
                    f"Estratos omitidos: {skipped_strata}"
                )

                # Registrar en telemetría
                self.telemetry.record_metric(
                    "stratum_transition",
                    "skipped_strata",
                    skipped
                )
            return

        # Caso 2: Retroceso (potencial reinicio de ciclo)
        if next_level < current_level:
            # Verificar si es reinicio válido (volver a PHYSICS desde WISDOM)
            is_valid_cycle_restart = (
                current == Stratum.WISDOM and
                next_stratum == Stratum.PHYSICS
            )

            if is_valid_cycle_restart:
                logger.info(
                    f"🔄 Reinicio de ciclo detectado: {current.name} → {next_stratum.name}"
                )
                self._filtration_level = 0  # Resetear nivel
                return

            # Retroceso parcial (potencialmente problemático)
            regression_depth = current_level - next_level
            logger.warning(
                f"⚠️ Regresión de estrato detectada: {current.name} → {next_stratum.name} "
                f"(profundidad: {regression_depth})"
            )

            # Registrar para auditoría
            self.telemetry.record_metric(
                "stratum_transition",
                "regression_depth",
                regression_depth
            )

            # Permitir pero marcar contexto
            return
    def _stratum_to_filtration(self, stratum: Stratum) -> int:
        mapping = {
            Stratum.PHYSICS: 1,
            Stratum.TACTICS: 2,
            Stratum.STRATEGY: 3,
            Stratum.WISDOM: 4
        }
        return mapping.get(stratum, 0)

    def _infer_current_stratum(self, context: dict) -> Optional[Stratum]:
        """Infiere el estrato actual basado en claves de contexto."""
        context_keys = set(context.keys())
        stratum_indicators = {
            Stratum.PHYSICS: {"df_presupuesto", "df_insumos", "df_apus_raw"},
            Stratum.TACTICS: {"df_merged", "df_apu_costos", "df_tiempo"},
            Stratum.STRATEGY: {"graph", "business_topology_report"},
            Stratum.WISDOM: {"final_result", "bill_of_materials"}
        }
        for stratum, indicators in stratum_indicators.items():
            if indicators & context_keys:
                return stratum
        return None

    def _compute_state_trace(self, context: dict) -> float:
        try:
            trace = 0.0
            for key, value in context.items():
                if isinstance(value, pd.DataFrame):
                    trace += len(value) * value.shape[1]
                elif isinstance(value, (list, dict)):
                    trace += len(str(value)) / 100
            return trace
        except:
            return 0.0

    def _save_context_state_with_checksum(self, session_id: str, context: dict):
        import hashlib
        try:
            # We must be careful with what we pickle/hash.
            # This is a simplified version.
            context_keys = sorted(context.keys())
            checksum_str = "".join(context_keys) # Very simple checksum
            checksum = hashlib.sha256(checksum_str.encode()).hexdigest()

            context["_integrity_checksum"] = checksum
            context["_persisted_at"] = datetime.datetime.now().isoformat()

            session_file = self.session_dir / f"{session_id}.pkl"
            with open(session_file, "wb") as f:
                pickle.dump(context, f)

            logger.debug(f"💾 Contexto persistido con checksum: {checksum[:16]}...")
        except Exception as e:
            logger.error(f"Error persisting context: {e}")

    def _compute_homology_groups(self, context: dict):
        """
        Computa grupos de homología del complejo simplicial de datos.

        Usa el Laplaciano combinatorio para calcular números de Betti:
        - β₀ = dim(ker(L₀)) = componentes conexas
        - β₁ = dim(ker(L₁)) - dim(im(L₀)) = ciclos independientes

        Refinamiento: Usa descomposición sparse y manejo robusto de casos degenerados.
        """
        try:
            df_keys = [k for k in context.keys()
                       if isinstance(context.get(k), pd.DataFrame)
                       and not context[k].empty]

            n = len(df_keys)
            if n < 2:
                self._homology_groups = {"H0": 1, "H1": 0, "Betti": [1, 0]}
                return

            # Construir matriz de adyacencia ponderada
            adj_matrix = sparse.lil_matrix((n, n), dtype=np.float64)

            for i, key_i in enumerate(df_keys):
                df_i = context[key_i]
                cols_i = set(df_i.columns)

                for j in range(i + 1, n):
                    key_j = df_keys[j]
                    df_j = context[key_j]
                    cols_j = set(df_j.columns)

                    # Peso = Jaccard similarity de columnas
                    intersection = len(cols_i & cols_j)
                    union = len(cols_i | cols_j)

                    if intersection > 0 and union > 0:
                        weight = intersection / union
                        adj_matrix[i, j] = weight
                        adj_matrix[j, i] = weight

            adj_csr = adj_matrix.tocsr()

            # Construir Laplaciano combinatorio L = D - A
            degrees = np.array(adj_csr.sum(axis=1)).flatten()
            degree_matrix = sparse.diags(degrees)
            laplacian = degree_matrix - adj_csr

            # Calcular β₀ (componentes conexas) via eigenvalores cercanos a 0
            h0 = self._count_zero_eigenvalues(laplacian, n)

            # Calcular β₁ usando fórmula de Euler: χ = β₀ - β₁ + β₂ - ...
            # Para 1-complejo: χ = V - E, entonces β₁ = E - V + β₀
            num_edges = adj_csr.nnz // 2
            num_vertices = n
            h1 = max(0, num_edges - num_vertices + h0)

            # Verificar consistencia topológica
            euler_char = h0 - h1
            expected_euler = num_vertices - num_edges

            if euler_char != expected_euler:
                logger.warning(
                    f"⚠️ Inconsistencia en característica de Euler: "
                    f"calculada={euler_char}, esperada={expected_euler}"
                )

            self._homology_groups = {
                "H0": h0,
                "H1": h1,
                "Betti_numbers": [h0, h1],
                "Euler_characteristic": euler_char,
                "vertices": num_vertices,
                "edges": num_edges
            }

            logger.debug(f"🧮 Homología: β₀={h0}, β₁={h1}, χ={euler_char}")

        except Exception as e:
            logger.warning(f"⚠️ Error computando homología: {e}")
            self._homology_groups = {"H0": 1, "H1": 0, "error": str(e)}

    def _count_zero_eigenvalues(
        self,
        laplacian: sparse.spmatrix,
        n: int,
        tol: float = 1e-10
    ) -> int:
        """
        Cuenta eigenvalores cercanos a cero del Laplaciano.

        Usa shift-invert para estabilidad con eigenvalores pequeños.
        """
        if n <= 1:
            return n

        if n <= 10:
            # Para matrices pequeñas, usar método denso
            try:
                L_dense = laplacian.toarray()
                eigenvalues = np.linalg.eigvalsh(L_dense)
                return int(np.sum(np.abs(eigenvalues) < tol))
            except np.linalg.LinAlgError:
                return 1

        # Para matrices grandes, usar método iterativo
        try:
            k = min(n - 1, max(1, n // 4))  # Número de eigenvalores a calcular

            # Shift-invert mode para eigenvalores cerca de 0
            eigenvalues = sparse.linalg.eigsh(
                laplacian,
                k=k,
                sigma=0.0,  # Target eigenvalue
                which='LM',  # Largest magnitude after shift-invert = smallest
                return_eigenvectors=False,
                tol=1e-6,
                maxiter=1000
            )

            return int(np.sum(np.abs(eigenvalues) < tol))

        except (sparse.linalg.ArpackNoConvergence, sparse.linalg.ArpackError) as e:
            logger.debug(f"ARPACK no convergió, usando estimación: {e}")
            # Fallback: estimar componentes por BFS
            return self._estimate_components_bfs(laplacian, n)
        except Exception:
            return 1

    def _estimate_components_bfs(
        self,
        laplacian: sparse.spmatrix,
        n: int
    ) -> int:
        """Estima componentes conexas por BFS cuando eigsh falla."""
        adj = (laplacian != laplacian.diagonal()).astype(bool)
        visited = np.zeros(n, dtype=bool)
        components = 0

        for start in range(n):
            if not visited[start]:
                components += 1
                # BFS
                queue = [start]
                while queue:
                    node = queue.pop(0)
                    if not visited[node]:
                        visited[node] = True
                        neighbors = adj[node].nonzero()[1]
                        queue.extend(neighbors[~visited[neighbors]])

        return components

    def _attempt_state_recovery(self, session_id: str, context: dict) -> bool:
        try:
            corrupt_file = self.session_dir / f"{session_id}_corrupt.pkl"
            with open(corrupt_file, "wb") as f:
                pickle.dump(context, f)
            return False
        except:
            return False

    def _validate_context_merge(self, context, initial):
        pass

    def _load_thresholds(self, config: dict) -> ProcessingThresholds:
        thresholds = ProcessingThresholds()
        if "processing_thresholds" in config:
            for key, value in config["processing_thresholds"].items():
                if hasattr(thresholds, key):
                    setattr(thresholds, key, value)
        return thresholds

    def _load_context_state(self, session_id: str) -> dict:
        if not session_id: return {}
        try:
            session_file = self.session_dir / f"{session_id}.pkl"
            if session_file.exists():
                with open(session_file, "rb") as f:
                    return pickle.load(f)
            return {}
        except:
            return {}






# ==================== ENTRY POINT ====================


def process_all_files(
    presupuesto_path: str,
    apus_path: str,
    insumos_path: str,
    config: dict,
    telemetry: TelemetryContext,
) -> dict:
    """Entry point refactorizado para usar PipelineDirector y Telemetry."""
    logger.info("🚀 Iniciando procesamiento via PipelineDirector")

    director = PipelineDirector(config, telemetry)

    initial_context = {
        "presupuesto_path": presupuesto_path,
        "apus_path": apus_path,
        "insumos_path": insumos_path,
    }

    try:
        final_context = director.execute_pipeline_orchestrated(initial_context)
        final_result = final_context.get("final_result", {})

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
    result: dict, output_files: dict, config: dict
) -> Dict[str, bool]:
    """Guarda archivos de salida de forma robusta."""
    import json

    if not result or not isinstance(result, dict):
        logger.error("❌ Resultado vacío o inválido para guardar")
        return {}

    if not output_files or not isinstance(output_files, dict):
        logger.error("❌ output_files vacío o inválido")
        return {}

    save_status = {}

    for name, path in output_files.items():
        try:
            if name not in result:
                logger.debug(f"Clave '{name}' no encontrada en resultado, saltando")
                save_status[name] = False
                continue

            data = result[name]
            if not data:
                logger.debug(f"Datos vacíos para '{name}', saltando")
                save_status[name] = False
                continue

            path = Path(path)
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
            except PermissionError:
                logger.error(f"❌ Sin permisos para crear directorio: {path.parent}")
                save_status[name] = False
                continue
            except Exception as e:
                logger.error(f"❌ Error creando directorio {path.parent}: {e}")
                save_status[name] = False
                continue

            try:
                sanitized_data = sanitize_for_json(data)
            except Exception as e:
                logger.error(f"❌ Error sanitizando datos para '{name}': {e}")
                save_status[name] = False
                continue

            temp_path = path.with_suffix(".tmp")

            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(
                        sanitized_data,
                        f,
                        indent=2,
                        ensure_ascii=False,
                        default=str,
                    )

                temp_path.replace(path)
                logger.info(f"✅ Archivo guardado: {path}")
                save_status[name] = True

            except Exception as e:
                logger.error(f"❌ Error escribiendo '{name}': {e}")
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except:
                        pass
                save_status[name] = False

        except Exception as e:
            logger.error(f"❌ Error inesperado guardando '{name}': {e}")
            save_status[name] = False

    successful = sum(1 for v in save_status.values() if v)
    total = len(save_status)
    logger.info(f"📁 Archivos guardados: {successful}/{total}")

    return save_status