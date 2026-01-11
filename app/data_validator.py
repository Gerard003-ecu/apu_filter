"""
Este componente ejecuta una auditoría profunda sobre los datos normalizados.
A diferencia de una validación de tipos simple, aplica reglas de negocio y
física de costos para detectar "patologías" en la información (costos negativos,
incoherencias aritméticas, descripciones vacías).

Protocolos de Auditoría (Versión 2.0 - Ingeniería de Calidad):
--------------------------------------------------------------
1. Coherencia Matemática Estricta (Control Theory):
   Utiliza un modelo de ganancia y sensibilidad para evaluar la estabilidad
   de la ecuación `Cantidad * Precio = Total`. Distingue entre errores
   de redondeo y fallos sistémicos.

2. Validador Piramidal (Graph Theory):
   Construye un grafo bipartito (APUs <-> Insumos) para medir la resiliencia
   estructural, identificando Puntos de Fallo Único (SPOF) y Nodos Flotantes.

3. Detección de Anomalías (Statistical & Light ML):
   Aplica Isolation Forest y métodos estadísticos robustos (Z-Score, IQR)
   para identificar costos atípicos.

4. Entropía de Calidad (Thermodynamics):
   Calcula la "Temperatura de Ingesta" ($S_{data}$) basada en la entropía de
   las alertas y la consistencia de los datos, distinguiendo entre un
   proyecto riesgoso y datos caóticos.
"""

import logging
import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Telemetry integration
try:
    from .telemetry import TelemetryContext
except ImportError:
    TelemetryContext = Any  # Fallback

# Optional fuzzy matching
try:
    from fuzzywuzzy import process

    HAS_FUZZY = True
except ImportError:
    process = None
    HAS_FUZZY = False

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES DE VALIDACIÓN
# ============================================================================
COSTO_MAXIMO_RAZONABLE = 50_000_000
COSTO_MINIMO_VALIDO = 0
CANTIDAD_MINIMA_VALIDA = 0
FUZZY_MATCH_THRESHOLD = 70
TOLERANCIA_COMPARACION_FLOAT = 1e-6
TOLERANCIA_PORCENTUAL_COHERENCIA = 0.01
DESCRIPCION_GENERICA_FALLBACK = "Insumo sin descripción"
MAX_DESCRIPCION_LENGTH = 500
MAX_ALERTAS_POR_ITEM = 50
MAX_IDENTIFICADOR_LENGTH = 100
MAX_FUZZY_ATTEMPTS_PER_BATCH = 1000

VALORES_DESCRIPCION_INVALIDOS = frozenset(
    [
        "nan",
        "none",
        "null",
        "",
        "n/a",
        "na",
        "n.a.",
        "n.a",
        "-",
        "--",
        "---",
        ".",
        "..",
        "...",
        "undefined",
        "sin descripcion",
        "sin descripción",
        "no aplica",
        "no disponible",
        "pendiente",
    ]
)


# ============================================================================
# ENUMS Y DATACLASSES
# ============================================================================
class TipoAlerta(Enum):
    """Tipos de alertas para clasificación"""

    COSTO_EXCESIVO = "COSTO_EXCESIVO"
    COSTO_NEGATIVO = "COSTO_NEGATIVO"
    CANTIDAD_RECALCULADA = "CANTIDAD_RECALCULADA"
    CANTIDAD_INVALIDA = "CANTIDAD_INVALIDA"
    DESCRIPCION_CORREGIDA = "DESCRIPCION_CORREGIDA"
    DESCRIPCION_FALTANTE = "DESCRIPCION_FALTANTE"
    INCOHERENCIA_MATEMATICA = "INCOHERENCIA_MATEMATICA"
    VALOR_INFINITO = "VALOR_INFINITO"
    CAMPO_REQUERIDO_FALTANTE = "CAMPO_REQUERIDO_FALTANTE"
    # Nuevos tipos V2
    ANOMALIA_ESTADISTICA = "ANOMALIA_ESTADISTICA"
    SPOF_DETECTADO = "SPOF_DETECTADO"
    ALTA_SENSIBILIDAD = "ALTA_SENSIBILIDAD_NUMERICA"


@dataclass
class ValidationMetrics:
    """Métricas de validación para reporting"""

    total_items_procesados: int = 0
    items_con_alertas: int = 0
    costos_excesivos: int = 0
    costos_negativos: int = 0
    cantidades_recalculadas: int = 0
    descripciones_corregidas: int = 0
    incoherencias_matematicas: int = 0
    valores_infinitos: int = 0
    items_con_errores: int = 0
    alertas_por_tipo: Dict[str, int] = field(default_factory=dict)

    def agregar_alerta(self, tipo_alerta: TipoAlerta) -> None:
        tipo_str = tipo_alerta.value
        self.alertas_por_tipo[tipo_str] = self.alertas_por_tipo.get(tipo_str, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_items_procesados": self.total_items_procesados,
            "items_con_alertas": self.items_con_alertas,
            "costos_excesivos": self.costos_excesivos,
            "costos_negativos": self.costos_negativos,
            "cantidades_recalculadas": self.cantidades_recalculadas,
            "descripciones_corregidas": self.descripciones_corregidas,
            "incoherencias_matematicas": self.incoherencias_matematicas,
            "valores_infinitos": self.valores_infinitos,
            "items_con_errores": self.items_con_errores,
            "alertas_por_tipo": self.alertas_por_tipo,
        }


@dataclass
class CampoFaltanteInfo:
    nombre: str
    motivo: str
    valor_actual: Any = None


@dataclass
class PyramidalMetrics:
    """Métricas de estabilidad vertical de la pirámide de datos."""

    base_width: int
    structure_load: int
    pyramid_stability_index: float
    floating_nodes: List[str]
    # Campos extendidos V2
    graf_analysis: Optional[Dict[str, Any]] = None


# ============================================================================
# COMPONENTES AVANZADOS (V2)
# ============================================================================


class PyramidalValidator:
    """
    Validador de integridad jerárquica con análisis de grafos y teoría de redes.
    """

    def validate_structure(
        self,
        apus_df: pd.DataFrame,
        insumos_df: pd.DataFrame,
        calcular_centralidad: bool = True,
        umbral_conexion_minima: float = 0.3,
    ) -> PyramidalMetrics:
        # 1. Normalización robusta de entradas
        if not isinstance(insumos_df, pd.DataFrame):
            insumos_df = pd.DataFrame()
        if not isinstance(apus_df, pd.DataFrame):
            apus_df = pd.DataFrame()

        # 2. Construcción del grafo bipartito APUs ↔ Insumos
        grafo_analysis = self._construir_grafo_bipartito(apus_df, insumos_df)

        # 3. Extracción de componentes conectados
        componentes = self._extraer_componentes_conectados(grafo_analysis)

        # 4. Cálculo de métricas de estabilidad
        n_insumos = len(grafo_analysis["nodos_insumos"])
        n_apus = len(grafo_analysis["nodos_apus"])
        n_conexiones = grafo_analysis["total_conexiones"]

        if n_apus > 0:
            psi_basico = n_insumos / n_apus
            max_conexiones_posibles = n_insumos * n_apus
            if max_conexiones_posibles > 0:
                densidad_red = n_conexiones / max_conexiones_posibles
                factor_conectividad = np.tanh(densidad_red * 5)
            else:
                factor_conectividad = 0
            psi_corregido = psi_basico * factor_conectividad
        else:
            psi_basico = 0
            psi_corregido = 0

        # 5. Análisis de nodos flotantes
        apus_con_insumos = set()
        if "APU_CODIGO" in insumos_df.columns:
            apus_con_insumos = set(insumos_df["APU_CODIGO"].dropna().unique())

        apus_totales = set()
        if "CODIGO_APU" in apus_df.columns:
            apus_totales = set(apus_df["CODIGO_APU"].dropna().unique())
        elif not apus_df.empty:
            apus_totales = set(apus_df.iloc[:, 0].dropna().unique())

        floating_nodes_set = apus_totales - apus_con_insumos
        floating_apus = sorted(list(floating_nodes_set))

        # 6. Centralidad
        metricas_centralidad = {}
        if (
            calcular_centralidad
            and "matriz_adyacencia" in grafo_analysis
            and grafo_analysis["matriz_adyacencia"].size > 0
        ):
            metricas_centralidad = self._calcular_centralidad_grafo(
                grafo_analysis["matriz_adyacencia"],
                grafo_analysis["nodos_apus"],
                grafo_analysis["nodos_insumos"],
            )

        # 7. Resiliencia
        resiliencia_metrics = self._analizar_resiliencia(
            grafo_analysis, umbral_conexion_minima
        )

        return PyramidalMetrics(
            base_width=n_insumos,
            structure_load=n_apus,
            pyramid_stability_index=psi_corregido,
            floating_nodes=floating_apus,
            graf_analysis={
                "total_nodos": len(grafo_analysis["nodos_apus"])
                + len(grafo_analysis["nodos_insumos"]),
                "total_conexiones": n_conexiones,
                "componentes_conectados": len(componentes),
                "psi_basico": psi_basico,
                "psi_corregido": psi_corregido,
                "centralidad": metricas_centralidad,
                "resiliencia": resiliencia_metrics,
            },
        )

    def _construir_grafo_bipartito(
        self, apus_df: pd.DataFrame, insumos_df: pd.DataFrame
    ) -> Dict[str, Any]:
        grafo = {
            "nodos_apus": [],
            "nodos_insumos": [],
            "aristas": [],
            "matriz_adyacencia": np.array([]),
            "total_conexiones": 0,
        }

        apu_col = "CODIGO_APU"
        if apu_col in apus_df.columns:
            apus_codes = apus_df[apu_col].dropna().unique().tolist()
        elif not apus_df.empty:
            apus_codes = apus_df.iloc[:, 0].dropna().unique().tolist()
        else:
            apus_codes = []
        grafo["nodos_apus"] = apus_codes

        insumo_desc_col = "DESCRIPCION_INSUMO_NORM"
        if insumo_desc_col not in insumos_df.columns:
            insumo_desc_col = "DESCRIPCION_INSUMO"

        if insumo_desc_col in insumos_df.columns:
            insumos_unique = insumos_df[insumo_desc_col].dropna().unique().tolist()
        else:
            insumos_unique = []
        grafo["nodos_insumos"] = insumos_unique

        if apus_codes and insumos_unique:
            apu_to_idx = {apu: i for i, apu in enumerate(apus_codes)}
            insumo_to_idx = {insumo: i for i, insumo in enumerate(insumos_unique)}

            adj_matrix = np.zeros((len(apus_codes), len(insumos_unique)), dtype=int)

            if "APU_CODIGO" in insumos_df.columns and insumo_desc_col in insumos_df.columns:
                # Vectorizar para performance
                valid_rows = insumos_df.dropna(subset=["APU_CODIGO", insumo_desc_col])
                for _, row in valid_rows.iterrows():
                    apu = row["APU_CODIGO"]
                    insumo = row[insumo_desc_col]
                    if apu in apu_to_idx and insumo in insumo_to_idx:
                        i = apu_to_idx[apu]
                        j = insumo_to_idx[insumo]
                        adj_matrix[i, j] = 1
                        grafo["aristas"].append((apu, insumo))

            grafo["matriz_adyacencia"] = adj_matrix
            grafo["total_conexiones"] = int(np.sum(adj_matrix))

        return grafo

    def _extraer_componentes_conectados(self, grafo: Dict[str, Any]) -> List[List[str]]:
        componentes = []
        if not grafo["aristas"]:
            return componentes

        adj_dict = {}
        for apu, insumo in grafo["aristas"]:
            adj_dict.setdefault(apu, set()).add(insumo)
            adj_dict.setdefault(insumo, set()).add(apu)

        visitados = set()
        for nodo in adj_dict:
            if nodo not in visitados:
                componente = []
                stack = [nodo]
                while stack:
                    current = stack.pop()
                    if current not in visitados:
                        visitados.add(current)
                        componente.append(current)
                        stack.extend(adj_dict.get(current, set()) - visitados)
                if componente:
                    componentes.append(componente)
        return componentes

    def _calcular_centralidad_grafo(
        self, adj_matrix: np.ndarray, nodos_apus: List[str], nodos_insumos: List[str]
    ) -> Dict[str, Any]:
        metrics = {
            "centralidad_grado_apus": {},
            "centralidad_grado_insumos": {},
            "apus_criticos": [],
            "insumos_criticos": [],
        }

        grado_apus = np.sum(adj_matrix, axis=1)
        for i, grado in enumerate(grado_apus):
            metrics["centralidad_grado_apus"][nodos_apus[i]] = int(grado)

        grado_insumos = np.sum(adj_matrix, axis=0)
        for j, grado in enumerate(grado_insumos):
            metrics["centralidad_grado_insumos"][nodos_insumos[j]] = int(grado)

        return metrics

    def _analizar_resiliencia(
        self, grafo: Dict[str, Any], umbral_conexion: float
    ) -> Dict[str, Any]:
        if not grafo["aristas"]:
            return {"resiliencia": 0, "puntos_fallo_unico": []}

        spofs = []
        adj_dict = {}
        for apu, insumo in grafo["aristas"]:
            adj_dict.setdefault(insumo, set()).add(apu)

        # Contar APUs totales para determinar umbral crítico
        total_apus = len(grafo.get("nodos_apus", []))
        umbral_critico = max(2, int(total_apus * 0.1)) # Al menos 2, o 10% del proyecto

        for insumo, apus_conectados in adj_dict.items():
            # Un SPOF es un insumo crítico que afecta a múltiples APUs
            if len(apus_conectados) >= umbral_critico:
                spofs.append({
                    "insumo": insumo,
                    "impacto": len(apus_conectados),
                    "apus_afectados": list(apus_conectados)[:5] # Muestra
                })

        return {"puntos_fallo_unico": spofs}


class AnomalyValidator:
    """
    Sistema de detección de anomalías usando métodos estadísticos y ML liviano.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        defaults = {
            "zscore_threshold": 3.0,
            "iqr_multiplier": 1.5,
            "isolation_forest_samples": 100,
        }
        if config:
            defaults.update(config)
        self.config = defaults

    def detect_cost_anomalies(
        self, cost_data: List[Dict[str, Any]], campo_costo: str = "VALOR_CONSTRUCCION_UN"
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        if not cost_data:
            return [], {"total_anomalias": 0}

        valores = []
        indices_validos = []
        for idx, item in enumerate(cost_data):
            if isinstance(item, dict) and _es_numero_valido(item.get(campo_costo)):
                try:
                    valores.append(float(item[campo_costo]))
                    indices_validos.append(idx)
                except (ValueError, TypeError):
                    continue

        if not valores:
            return cost_data, {"total_anomalias": 0}

        valores_array = np.array(valores).reshape(-1, 1)

        # Ensamble simple: Z-Score + IQR
        # Z-Score
        mean = np.mean(valores_array)
        std = np.std(valores_array)
        z_anomalies = np.zeros(len(valores_array), dtype=bool)
        if std > 0:
            z_anomalies = (
                np.abs((valores_array.flatten() - mean) / std)
                > self.config["zscore_threshold"]
            )

        # IQR
        q1 = np.percentile(valores_array, 25)
        q3 = np.percentile(valores_array, 75)
        iqr = q3 - q1
        iqr_anomalies = (valores_array.flatten() < (q1 - 1.5 * iqr)) | (
            valores_array.flatten() > (q3 + 1.5 * iqr)
        )

        # Isolation Forest (si disponible)
        if_anomalies = np.zeros(len(valores_array), dtype=bool)
        try:
            from sklearn.ensemble import IsolationForest

            n_samples = min(self.config["isolation_forest_samples"], len(valores_array))
            model = IsolationForest(
                contamination="auto", max_samples=n_samples, random_state=42
            )
            preds = model.fit_predict(valores_array)
            if_anomalies = preds == -1
        except ImportError:
            pass

        # Votación
        votes = (
            z_anomalies.astype(int) + iqr_anomalies.astype(int) + if_anomalies.astype(int)
        )
        confirmed = votes >= 2

        datos_marcados = deepcopy(cost_data)
        count = 0
        for i, idx in enumerate(indices_validos):
            if confirmed[i]:
                item = datos_marcados[idx]
                if "anomalias" not in item:
                    item["anomalias"] = []
                item["anomalias"].append(
                    {
                        "tipo": "COSTO_ANOMALO",
                        "valor": float(valores_array[i][0]),
                        "score": int(votes[i]),
                    }
                )
                count += 1

        return datos_marcados, {
            "total_anomalias": count,
            "porcentaje_anomalias": (count / len(valores_array)) * 100,
            "estadisticas": {"media": float(mean), "std": float(std)},
        }


class AlertSystem:
    """Sistema de alertas con priorización basada en entropía."""

    def process_alerts(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        all_alerts = []
        for item in items:
            if isinstance(item, dict) and "alertas" in item:
                for alerta in item["alertas"]:
                    all_alerts.append(alerta["tipo"])

        if not all_alerts:
            return {"entropia_sistema": 0.0}

        unique, counts = np.unique(all_alerts, return_counts=True)
        probs = counts / len(all_alerts)
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(len(unique)) if len(unique) > 0 else 1

        return {
            "total_alertas": len(all_alerts),
            "entropia_sistema": float(entropy),
            "entropia_normalizada": float(entropy / max_entropy) if max_entropy > 0 else 0,
        }


# ============================================================================
# FUNCIONES AUXILIARES BASE
# ============================================================================
def _es_numero_valido(valor: Any) -> bool:
    if valor is None:
        return False
    if isinstance(valor, (int, float, np.integer, np.floating)):
        try:
            return not (pd.isna(valor) or np.isinf(valor))
        except (ValueError, TypeError):
            return False
    return False


def _agregar_alerta(
    item: Dict[str, Any],
    mensaje: str,
    tipo_alerta: TipoAlerta,
    metrics: Optional[ValidationMetrics] = None,
    permitir_duplicados: bool = False,
) -> bool:
    if not isinstance(item, dict):
        return False
    if "alertas" not in item:
        item["alertas"] = []

    if len(item["alertas"]) >= MAX_ALERTAS_POR_ITEM:
        return False

    alerta = {"tipo": tipo_alerta.value, "mensaje": mensaje}

    if not permitir_duplicados:
        for a in item["alertas"]:
            if a["tipo"] == alerta["tipo"] and a["mensaje"] == alerta["mensaje"]:
                return False

    item["alertas"].append(alerta)
    if metrics:
        metrics.agregar_alerta(tipo_alerta)
    return True


def _obtener_identificador_item(item: Dict[str, Any]) -> str:
    campos = ["ITEM", "ID", "CODIGO", "DESCRIPCION_INSUMO", "DESCRIPCION", "NOMBRE"]
    for c in campos:
        val = item.get(c)
        if val and str(val).lower() not in ("nan", "none", ""):
            return str(val)[:MAX_IDENTIFICADOR_LENGTH]
    return "item_desconocido"


def _validar_coherencia_matematica(
    cantidad: float,
    precio_unitario: float,
    valor_total: float,
    tolerancia_absoluta: Optional[float] = None,
) -> Tuple[bool, Optional[float], Optional[str], Optional[Dict[str, Any]]]:
    """
    Validación mejorada con Teoría de Control (Ganancia y Sensibilidad).
    """
    try:
        Q = float(cantidad)
        P = float(precio_unitario)
        VT = float(valor_total)
    except (ValueError, TypeError):
        return False, None, "Error de conversión", {}

    esperado = Q * P

    # Análisis de estabilidad
    ganancia_q = P
    ganancia_p = Q
    sensibilidad = np.sqrt(ganancia_q**2 + ganancia_p**2)

    error_abs = abs(esperado - VT)
    tol = tolerancia_absoluta or max(1.0, abs(esperado) * TOLERANCIA_PORCENTUAL_COHERENCIA)

    es_coherente = error_abs <= tol

    analisis = {
        "sensibilidad": sensibilidad,
        "error_absoluto": error_abs,
        "ganancia_sistema": max(abs(ganancia_q), abs(ganancia_p)),
    }

    msg = f"Esperado: {esperado:.2f}, Reportado: {VT:.2f}"
    if sensibilidad > 1000:
        msg += f" [ALTA SENSIBILIDAD: {sensibilidad:.0f}]"

    diferencia_pct = (error_abs / max(abs(esperado), 1e-9)) * 100

    return es_coherente, diferencia_pct, msg, analisis


def _limpiar_y_validar_descripcion(
    descripcion: Any,
) -> Tuple[Optional[str], bool, Optional[str]]:
    if not descripcion or pd.isna(descripcion):
        return None, False, "Vacío"
    try:
        s = str(descripcion)
        s = unicodedata.normalize("NFC", s)
        s = re.sub(r"[\x00-\x1f]", "", s).strip()
        if not s or s.lower() in VALORES_DESCRIPCION_INVALIDOS:
            return None, False, "Inválido"
        return s[:MAX_DESCRIPCION_LENGTH], True, None
    except Exception:
        return None, False, "Error"


# ============================================================================
# FUNCIONES DE VALIDACIÓN ESPECÍFICAS
# ============================================================================
def _validate_extreme_costs(
    presupuesto_data: List[Dict[str, Any]],
    anomaly_validator: Optional[AnomalyValidator] = None,
) -> Tuple[List[Dict[str, Any]], ValidationMetrics]:
    metrics = ValidationMetrics()
    if not presupuesto_data:
        return [], metrics

    result = deepcopy(presupuesto_data)

    # Validación clásica
    for item in result:
        val = item.get("VALOR_CONSTRUCCION_UN")
        if _es_numero_valido(val):
            v = float(val)
            if v < COSTO_MINIMO_VALIDO:
                _agregar_alerta(item, f"Negativo: {v}", TipoAlerta.COSTO_NEGATIVO, metrics)
            elif v > COSTO_MAXIMO_RAZONABLE:
                _agregar_alerta(item, f"Excesivo: {v}", TipoAlerta.COSTO_EXCESIVO, metrics)

    # Validación ML (si disponible)
    if anomaly_validator:
        result, anom_metrics = anomaly_validator.detect_cost_anomalies(result)
        for item in result:
            if "anomalias" in item:
                for a in item["anomalias"]:
                    _agregar_alerta(
                        item,
                        f"Anomalía (Score {a['score']}): {a['valor']}",
                        TipoAlerta.ANOMALIA_ESTADISTICA,
                        metrics,
                    )

    metrics.total_items_procesados = len(result)
    metrics.items_con_alertas = sum(1 for x in result if x.get("alertas"))
    return result, metrics


def _validate_zero_quantity_with_cost(
    apus_data: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], ValidationMetrics]:
    metrics = ValidationMetrics()
    if not apus_data:
        return [], metrics

    result = deepcopy(apus_data)

    for item in result:
        q = item.get("CANTIDAD")
        vr = item.get("VR_UNITARIO")
        vt = item.get("VALOR_TOTAL")

        if _es_numero_valido(q) and _es_numero_valido(vr) and _es_numero_valido(vt):
            q, vr, vt = float(q), float(vr), float(vt)

            # Control Theory Check
            coherente, diff, msg, analisis = _validar_coherencia_matematica(q, vr, vt)
            if not coherente:
                _agregar_alerta(item, msg, TipoAlerta.INCOHERENCIA_MATEMATICA, metrics)
                if analisis.get("sensibilidad", 0) > 1000:
                    _agregar_alerta(
                        item,
                        f"Sistema Inestable (S={analisis['sensibilidad']:.0f})",
                        TipoAlerta.ALTA_SENSIBILIDAD,
                        metrics,
                    )

            # Recálculo si Q=0 pero VT>0
            if abs(q) < 1e-6 and vt > 1e-6 and vr > 1e-6:
                new_q = vt / vr
                item["CANTIDAD"] = new_q
                _agregar_alerta(
                    item,
                    f"Recalculado: {new_q:.4f}",
                    TipoAlerta.CANTIDAD_RECALCULADA,
                    metrics,
                )

    metrics.total_items_procesados = len(result)
    metrics.items_con_alertas = sum(1 for x in result if x.get("alertas"))
    return result, metrics


def _validate_descriptions(
    apus_data: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], ValidationMetrics]:
    # Placeholder simple para mantener compatibilidad
    metrics = ValidationMetrics()
    if not apus_data:
        return [], metrics
    result = deepcopy(apus_data)
    for item in result:
        desc, valid, _ = _limpiar_y_validar_descripcion(item.get("DESCRIPCION_INSUMO"))
        if not valid:
            item["DESCRIPCION_INSUMO"] = DESCRIPCION_GENERICA_FALLBACK
            _agregar_alerta(
                item, "Descripción inválida", TipoAlerta.DESCRIPCION_FALTANTE, metrics
            )
        elif desc != item.get("DESCRIPCION_INSUMO"):
            item["DESCRIPCION_INSUMO"] = desc

    metrics.total_items_procesados = len(result)
    metrics.items_con_alertas = sum(1 for x in result if x.get("alertas"))
    return result, metrics


# ============================================================================
# ORQUESTADOR PRINCIPAL (V2)
# ============================================================================
def validate_and_clean_data(
    data_store: Dict[str, Any],
    skip_on_error: bool = True,
    validaciones_habilitadas: Optional[Dict[str, bool]] = None,
    telemetry_context: Optional[TelemetryContext] = None,
    aplicar_analisis_termico: bool = True,
) -> Dict[str, Any]:
    """
    Orquesta la validación V2 con análisis de Entropía de Calidad.
    """
    logger.info("=" * 80)
    logger.info("Sistema de Validación de Calidad (V2) - Termodinámica de Datos")
    logger.info("=" * 80)

    if telemetry_context:
        telemetry_context.start_step("validate_data")

    if not data_store or not isinstance(data_store, dict):
        return {"error": "Invalid data_store", "validation_summary": {"exito": False}}

    result = deepcopy(data_store)

    # 1. Inicializar Validadores Avanzados
    anomaly_validator = AnomalyValidator()
    alert_system = AlertSystem()
    pyramidal_validator = PyramidalValidator()

    # 2. Análisis Térmico Inicial (Ingestion Temperature)
    quality_entropy_start = _calcular_entropia_calidad(result)
    logger.info(f"Entropía de Calidad Inicial: {quality_entropy_start:.2f}°Q")

    metricas_totales = {}

    # 3. Validaciones Core
    # Presupuesto
    if "presupuesto" in result:
        result["presupuesto"], m_pres = _validate_extreme_costs(
            result["presupuesto"], anomaly_validator
        )
        metricas_totales["presupuesto"] = m_pres.to_dict()

    # APUs Cantidades y Coherencia
    if "apus_detail" in result:
        result["apus_detail"], m_apu = _validate_zero_quantity_with_cost(
            result["apus_detail"]
        )
        metricas_totales["apus_detail_cantidad"] = m_apu.to_dict()

        # Descripciones
        result["apus_detail"], m_desc = _validate_descriptions(result["apus_detail"])
        metricas_totales["apus_detail_descripcion"] = m_desc.to_dict()

    # 4. Validación Piramidal (Grafos)
    if "apus_detail" in result:
        try:
            # Reconstruir DFs simples para el validador
            df_insumos = pd.DataFrame(result["apus_detail"])
            df_apus = pd.DataFrame(result.get("presupuesto", []))

            # Mapeo de columnas necesario para el validador piramidal
            if "CODIGO" in df_apus.columns:
                df_apus["CODIGO_APU"] = df_apus["CODIGO"]

            if "CODIGO" in df_insumos.columns and "APU" not in df_insumos.columns:
                # Insumos detail generalmente no tiene CODIGO_APU explícito si es lista plana
                pass

            pyr_metrics = pyramidal_validator.validate_structure(df_apus, df_insumos)
            result["pyramidal_metrics"] = pyr_metrics

            if pyr_metrics.floating_nodes:
                logger.warning(
                    f"Nodos Flotantes Detectados: {len(pyr_metrics.floating_nodes)}"
                )
        except Exception as e:
            logger.error(f"Error en validación piramidal: {e}")

    # 5. Análisis Térmico Final y Alertas
    quality_entropy_end = _calcular_entropia_calidad(result)

    # Procesar Alertas Sistémicas
    all_items = (result.get("presupuesto", []) or []) + (result.get("apus_detail", []) or [])
    alert_analysis = alert_system.process_alerts(all_items)

    thermal_analysis = {
        "quality_entropy_initial": quality_entropy_start,
        "quality_entropy_final": quality_entropy_end,
        "delta_entropy": quality_entropy_start - quality_entropy_end,
        "alert_system_entropy": alert_analysis.get("entropia_sistema", 0),
        "stability_status": _evaluar_estabilidad(quality_entropy_end),
    }

    result["quality_entropy_analysis"] = thermal_analysis
    result["validation_metrics"] = metricas_totales

    summary = {
        "exito": True,
        "total_alertas": alert_analysis.get("total_alertas", 0),
        "thermal_status": thermal_analysis["stability_status"],
    }
    result["validation_summary"] = summary

    if telemetry_context:
        telemetry_context.record_metric("validation", "quality_entropy", quality_entropy_end)
        telemetry_context.end_step("validate_data", "success", metadata=summary)

    return result


def _calcular_entropia_calidad(data: Dict[str, Any]) -> float:
    """Calcula S_data (Entropía de Calidad) 0-100."""
    temp = 0.0

    # Factor Anomalías
    total_items = 0
    total_anoms = 0

    for key in ["presupuesto", "apus_detail"]:
        items = data.get(key, [])
        if items:
            total_items += len(items)
            total_anoms += sum(1 for x in items if x.get("alertas") or x.get("anomalias"))

    if total_items > 0:
        ratio = total_anoms / total_items
        temp += ratio * 60  # Hasta 60 puntos por suciedad

    # Factor Estructural (Psi)
    if "pyramidal_metrics" in data:
        psi = data["pyramidal_metrics"].pyramid_stability_index
        if psi < 1.0:
            temp += (1.0 - psi) * 40  # Hasta 40 puntos por inestabilidad

    return min(100.0, temp)


def _evaluar_estabilidad(entropy: float) -> str:
    if entropy < 20:
        return "ESTABLE"
    if entropy < 50:
        return "METASTABLE"
    return "CAOTICO"
