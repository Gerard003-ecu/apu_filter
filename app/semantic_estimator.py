"""
Microservicio: Semantic Estimator (El Asesor Táctico)
Estrato DIKW: TACTICS (Nivel 2)

Responsabilidad: Alojamiento del espacio vectorial continuo (FAISS) y 
resolución de ambigüedades semánticas. Actúa como el motor de inferencia
desacoplado del Plano de Control.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

from app.schemas import Stratum
from app.telemetry import TelemetryContext
from app.estimator import SearchArtifacts, calculate_estimate
from app.tools_interface import MICRegistry

logger = logging.getLogger("SemanticEstimator")

class SemanticEstimatorService:
    """
    Agente autónomo que gobierna la búsqueda vectorial y estimación de costos.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.artifacts: Optional[SearchArtifacts] = None
        self.is_ready = False
        
        # Carga perezosa (Lazy Loading) para no bloquear el arranque
        self._load_tensor_space()

    def _load_tensor_space(self) -> None:
        """
        [Método Físico] Carga los modelos masivos en memoria aislada.
        Libera al servidor web (Gunicorn/Flask) de esta carga térmica.
        """
        logger.info("Iniciando ignición del Espacio Vectorial (FAISS)...")
        try:
            # Configuración extraída del config_rules.json
            model_name = self.config.get("embedding_metadata", {}).get("model_name", "all-MiniLM-L6-v2")
            index_path = self.config.get("faiss_index_path")
            map_path = self.config.get("id_map_path")

            model = SentenceTransformer(model_name)
            index = faiss.read_index(str(index_path))
            
            import json
            with open(map_path, 'r', encoding='utf-8') as f:
                id_map = json.load(f)

            self.artifacts = SearchArtifacts(model=model, faiss_index=index, id_map=id_map)
            self.is_ready = True
            logger.info(f"✅ Espacio Vectorial cargado. Dimensión: {index.d}, Vectores: {index.ntotal}")

        except Exception as e:
            logger.critical(f"❌ Fallo al materializar el espacio vectorial: {e}")
            self.is_ready = False

    def project_semantic_match(self, payload: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        [Vector de la MIC] Recibe una descripción humana y la mapea a un insumo formal.
        Morfismo: Texto Humano -> Espacio Latente -> Código de Insumo
        """
        telemetry: TelemetryContext = context.get("telemetry_context")
        if telemetry: telemetry.start_step("semantic_projection")

        if not self.is_ready or not self.artifacts:
            return {"success": False, "error": "Asesor Semántico no inicializado (OOM o fallo de carga)."}

        query_text = payload.get("query_text", "")
        df_pool = payload.get("df_pool") # DataFrame serializado o puntero a Redis

        if isinstance(df_pool, list):
            df_pool = pd.DataFrame(df_pool)

        try:
            # 1. Transformación a Tensor
            query_embedding = self.artifacts.model.encode([query_text]).astype(np.float32)
            
            # 2. Búsqueda en Grafo K-NN (FAISS)
            similarities, indices = self.artifacts.faiss_index.search(query_embedding, k=5)
            
            # Lógica de filtrado y match (delegada a _find_best_semantic_match de estimator.txt)
            # Retorna el candidato con mayor similitud estocástica
            best_match_id = self.artifacts.id_map[str(indices)]
            confidence = float(similarities)

            if telemetry: telemetry.end_step("semantic_projection", "success")
            
            return {
                "success": True,
                "matched_id": best_match_id,
                "confidence": confidence,
                "stratum": Stratum.TACTICS.name
            }

        except Exception as e:
            if telemetry: telemetry.record_error("semantic_projection", str(e))
            return {"success": False, "error": str(e)}

    def calculate_dynamic_estimate(self, payload: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        [Vector de la MIC] Ejecuta la estimación de costos delegada.
        Reemplaza la lógica que bloqueaba el endpoint /api/estimate en app.py.
        """
        telemetry: TelemetryContext = context.get("telemetry_context")
        
        try:
            params = payload.get("params", {})
            data_store = payload.get("data_store", {})
            
            # Invocamos la matemática estocástica pura sin tocar la red HTTP
            result = calculate_estimate(
                params=params,
                data_store=data_store,
                config=self.config,
                search_artifacts=self.artifacts
            )
            
            return {"success": True, "estimate": result, "stratum": Stratum.TACTICS.name}

        except Exception as e:
            logger.error(f"Fallo en cálculo táctico: {e}")
            return {"success": False, "error": str(e)}

    def register_in_mic(self, mic: MICRegistry) -> None:
        """
        Registra las capacidades de este agente en la Matriz de Interacción Central.
        Establece la política de Gobernanza: Estas acciones pertenecen a TACTICS.
        """
        mic.register_vector(
            service_name="semantic_match",
            stratum=Stratum.TACTICS,
            handler=self.project_semantic_match
        )
        mic.register_vector(
            service_name="tactical_estimate",
            stratum=Stratum.TACTICS,
            handler=self.calculate_dynamic_estimate
        )
        logger.info("✅ Vectores Tácticos Semánticos registrados en la MIC.")
