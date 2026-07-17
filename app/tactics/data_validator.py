# -*- coding: utf-8 -*-
"""
=========================================================================================
Módulo: Data Validator (Tribunal de Coherencia Termodinámica y Filtrado Topológico)
Ubicación: app/tactics/data_validator.py
Versión: 4.0 – Arquitectura en 3 Fases Anidadas con Rigor Matemático PhD
=========================================================================================

Fase 1 – Microscopía de Campos (conservación C = Q·P, IEEE 754, limpieza sintáctica).
         El morfismo final `phase1_to_topological_domain` produce el dominio de
         incidencia que inicia la Fase 2.
Fase 2 – Topología Bipartita (complejo simplicial APU–Insumo, laplaciano, β₀, λ₂, SPOF).
         El morfismo final `phase2_to_thermodynamic_state` produce el estado
         estructural que inicia la Fase 3.
Fase 3 – Termodinámica Informacional (entropía de Shannon/von Neumann, temperatura
         de ingesta, clasificación de estabilidad del tensor de datos).

Principios matemáticos aplicados (rigor formal):
    • Conservación del valor:  C = Q·P  con tolerancia híbrida
          ε_abs = max(ulp(C), |C|·ε_mach),  ε_rel = τ·|C|,  ε = ε_abs + ε_rel
      (criterio de Kahan + IEEE 754-2008 §4.3).
    • Sensibilidad: g = ‖∇(Q·P)‖₂ = √(P²+Q²);  κ = g·‖(Q,P)‖₂ / max(|C|,ε)
      (número de condición del producto).
    • Operador frontera ∂₁ : C₁ → C₀ del complejo de cadenas del grafo bipartito.
      Nodos flotantes ≡ {u ∈ U | (∂₁ ω)(u) = 0 ∀ ω con soporte en aristas de u}
      ⇔ grado(u) = 0 ⇔ proyección de ker(Bᵀ) sobre el bloque APU.
    • Laplaciano bipartito L = [[D_U, −A], [−Aᵀ, D_V]];  espectro real no-negativo.
      β₀ = mult(λ=0);  λ₂ = conectividad algebraica (Fiedler);  ρ(A) = radio espectral.
    • Entropía normalizada: H_norm = H / log₂(K) ∈ [0,1],  H = −Σ pᵢ log₂ pᵢ.
      Temperatura de ingesta: T = α·tasa_alertas + β·(1−Ψ),  Ψ = tanh(|V|/|U|).
    • Composición de morfismos (teoría de categorías):
          F₃ ∘ F₂ ∘ F₁ : DataStore → ValidatedThermodynamicState
=========================================================================================
"""

from __future__ import annotations

import logging
import math
import re
import unicodedata
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
from numpy.linalg import eigh, svd

# ---------------------------------------------------------------------------
# Telemetría opcional
# ---------------------------------------------------------------------------
try:
    from app.core.telemetry import TelemetryContext
except ImportError:  # pragma: no cover
    TelemetryContext = Any  # type: ignore

# Fuzzy matching opcional
try:
    from fuzzywuzzy import process as fuzzy_process
    HAS_FUZZY = True
except ImportError:  # pragma: no cover
    fuzzy_process = None
    HAS_FUZZY = False

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTES GLOBALES DE VALIDACIÓN (invariantes del dominio)
# ============================================================================
COSTO_MAXIMO_RAZONABLE: float = 50_000_000.0
COSTO_MINIMO_VALIDO: float = 0.0
CANTIDAD_MINIMA_VALIDA: float = 0.0
FUZZY_MATCH_THRESHOLD: int = 70
TOLERANCIA_PORCENTUAL_COHERENCIA: float = 0.01          # τ = 1 %
MAX_DESCRIPCION_LENGTH: int = 500
MAX_ALERTAS_POR_ITEM: int = 50
MAX_IDENTIFICADOR_LENGTH: int = 100
EPS_MACHINE: float = float(np.finfo(np.float64).eps)    # ≈ 2.22e-16
SENSITIVITY_ALERT_THRESHOLD: float = 1.0e3
SPOF_IMPACT_FRACTION: float = 0.10
FLOATING_DEGREE_TOL: float = 1e-12
SPECTRAL_ZERO_TOL: float = 1e-10

VALORES_DESCRIPCION_INVALIDOS: frozenset = frozenset({
    "nan", "none", "null", "", "n/a", "na", "n.a.", "n.a",
    "-", "--", "---", ".", "..", "...",
    "undefined", "sin descripcion", "sin descripción",
    "no aplica", "no disponible", "pendiente",
})


# ============================================================================
# FASE 0 — FUNDAMENTOS COMPARTIDOS (objetos del topos de datos)
# ============================================================================
# En el lenguaje de la teoría de categorías, las estructuras de esta fase
# constituyen el objeto terminal de soporte: Enums (clasificadores) y
# Dataclasses (objetos de métricas) que se preservan por todos los funtores
# F₁, F₂, F₃ del pipeline.
# ============================================================================

class TipoAlerta(Enum):
    """Clasificador de anomalías según su naturaleza físico-matemática."""
    COSTO_EXCESIVO = "COSTO_EXCESIVO"
    COSTO_NEGATIVO = "COSTO_NEGATIVO"
    CANTIDAD_RECALCULADA = "CANTIDAD_RECALCULADA"
    CANTIDAD_INVALIDA = "CANTIDAD_INVALIDA"
    DESCRIPCION_CORREGIDA = "DESCRIPCION_CORREGIDA"
    DESCRIPCION_FALTANTE = "DESCRIPCION_FALTANTE"
    INCOHERENCIA_MATEMATICA = "INCOHERENCIA_MATEMATICA"
    VALOR_INFINITO = "VALOR_INFINITO"
    CAMPO_REQUERIDO_FALTANTE = "CAMPO_REQUERIDO_FALTANTE"
    ANOMALIA_ESTADISTICA = "ANOMALIA_ESTADISTICA"
    SPOF_DETECTADO = "SPOF_DETECTADO"
    ALTA_SENSIBILIDAD = "ALTA_SENSIBILIDAD_NUMERICA"
    COMPONENTE_DESCONECTADA = "COMPONENTE_DESCONECTADA"
    CONDICIONAMIENTO_CRITICO = "CONDICIONAMIENTO_CRITICO"


@dataclass
class ValidationMetrics:
    """Contabiliza incidencias de la Fase 1 (microscopía de campos)."""
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

    def agregar_alerta(self, tipo: TipoAlerta) -> None:
        t = tipo.value
        self.alertas_por_tipo[t] = self.alertas_por_tipo.get(t, 0) + 1
        # Contadores semánticos
        if tipo is TipoAlerta.COSTO_EXCESIVO:
            self.costos_excesivos += 1
        elif tipo is TipoAlerta.COSTO_NEGATIVO:
            self.costos_negativos += 1
        elif tipo is TipoAlerta.CANTIDAD_RECALCULADA:
            self.cantidades_recalculadas += 1
        elif tipo is TipoAlerta.DESCRIPCION_CORREGIDA or tipo is TipoAlerta.DESCRIPCION_FALTANTE:
            self.descripciones_corregidas += 1
        elif tipo is TipoAlerta.INCOHERENCIA_MATEMATICA:
            self.incoherencias_matematicas += 1
        elif tipo is TipoAlerta.VALOR_INFINITO:
            self.valores_infinitos += 1

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
            "alertas_por_tipo": dict(self.alertas_por_tipo),
        }


@dataclass
class CoherenceAnalysis:
    """Resultado formal de la verificación C = Q·P con análisis de sensibilidad."""
    coherente: bool
    error_absoluto: float
    error_porcentual: float
    tolerancia_aplicada: float
    sensibilidad: float          # ‖∇C‖₂
    condicion: float             # κ(Q,P)
    ganancia_sistema: float
    mensaje: str


@dataclass
class PyramidalMetrics:
    """
    Resultado del análisis topológico del complejo simplicial bipartito APU–Insumo.
    Invariantes:
        base_width          = |V|  (cardinalidad de insumos)
        structure_load      = |U|  (cardinalidad de APUs)
        pyramid_stability   = Ψ = tanh(|V|/|U|)  ∈ (0,1)
        floating_nodes      = {u ∈ U | deg(u) = 0}
        connected_components = β₀ = dim H₀ = mult(λ=0)
        algebraic_connectivity = λ₂ (Fiedler)
        spectral_radius     = ρ(A) = σ_max(A)
    """
    base_width: int
    structure_load: int
    pyramid_stability_index: float
    floating_nodes: List[str]
    connected_components: int
    algebraic_connectivity: float
    spof_list: List[Dict[str, Any]]
    spectral_radius: float
    eigenvalues: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "base_width": self.base_width,
            "structure_load": self.structure_load,
            "pyramid_stability_index": self.pyramid_stability_index,
            "floating_nodes": list(self.floating_nodes),
            "connected_components": self.connected_components,
            "algebraic_connectivity": self.algebraic_connectivity,
            "spof_list": list(self.spof_list),
            "spectral_radius": self.spectral_radius,
            "n_eigenvalues": int(len(self.eigenvalues)) if self.eigenvalues is not None else 0,
        }


@dataclass
class ThermodynamicState:
    """Estado termodinámico del tensor de datos tras Fase 3."""
    entropia_shannon: float
    entropia_normalizada: float
    entropia_von_neumann: float
    tasa_alertas: float
    psi_estructural: float
    temperatura_ingesta: float
    estabilidad: str
    free_energy_proxy: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entropia_shannon": self.entropia_shannon,
            "entropia_normalizada": self.entropia_normalizada,
            "entropia_von_neumann": self.entropia_von_neumann,
            "tasa_alertas": self.tasa_alertas,
            "psi_estructural": self.psi_estructural,
            "temperatura_ingesta": self.temperatura_ingesta,
            "estabilidad": self.estabilidad,
            "free_energy_proxy": self.free_energy_proxy,
        }


# ============================================================================
# FASE 1 — MICROSCOPÍA DE CAMPOS
# ============================================================================
# Objetivo: por cada registro en los estratos `presupuesto` y `apus_detail`:
#   1. Verificar conservación C = Q·P con tolerancia híbrida IEEE 754.
#   2. Acotar costos extremos.
#   3. Recalcular cantidades nulas con costo positivo (inversa local de C/P).
#   4. Limpiar y validar descripciones (normalización Unicode NFC + filtro).
#   5. Detectar alta sensibilidad numérica (g > umbral) y mal condicionamiento.
#
# El último morfismo de esta fase, `phase1_to_topological_domain`, convierte
# las listas validadas en el par de DataFrames (U, V, E) que es el dominio
# de la Fase 2.  Formalmente:
#     F₁ : DataStore → (ValidatedStore, IncidenceDomain)
#     y  IncidenceDomain  ≅  entrada de  F₂.
# ============================================================================

def _es_numero_valido(valor: Any) -> bool:
    """
    Predicado: True ⇔ valor ∈ ℝ finito (no NaN, no ±∞).
    Equivalente a: valor ∈ 𝔽 \\ {NaN, ±∞} donde 𝔽 es el cuerpo flotante IEEE 754.
    """
    if valor is None:
        return False
    if isinstance(valor, (int, float, np.integer, np.floating)):
        try:
            f = float(valor)
            return math.isfinite(f) and not (isinstance(f, float) and math.isnan(f))
        except (ValueError, TypeError, OverflowError):
            return False
    return False


def _ulp(x: float) -> float:
    """
    Unit in the Last Place de |x| en float64.
    ulp(x) = 2^{e-52} donde e = floor(log₂|x|) para x ≠ 0.
    Aproxima el ruido de redondeo local (IEEE 754-2008).
    """
    ax = abs(float(x))
    if ax == 0.0:
        return EPS_MACHINE
    # 2^{exponent - 52}; math.frexp devuelve m∈[0.5,1), e tal que x = m·2^e
    _, exp = math.frexp(ax)
    return math.ldexp(1.0, exp - 53)  # mantissa 52 bits + implícito ⇒ 53


def _agregar_alerta(
    item: Dict[str, Any],
    mensaje: str,
    tipo: TipoAlerta,
    metrics: Optional[ValidationMetrics] = None,
    permitir_duplicados: bool = False,
) -> bool:
    """
    Registra una alerta en el ítem y actualiza métricas.
    Retorna True si la alerta fue insertada.
    """
    if not isinstance(item, dict):
        return False
    if "alertas" not in item or not isinstance(item["alertas"], list):
        item["alertas"] = []
    if len(item["alertas"]) >= MAX_ALERTAS_POR_ITEM:
        return False

    nueva = {"tipo": tipo.value, "mensaje": mensaje}
    if not permitir_duplicados:
        for a in item["alertas"]:
            if (
                isinstance(a, dict)
                and a.get("tipo") == nueva["tipo"]
                and a.get("mensaje") == nueva["mensaje"]
            ):
                return False

    item["alertas"].append(nueva)
    if metrics is not None:
        metrics.agregar_alerta(tipo)
    return True


# ---------------------------------------------------------------------------
# 1.1  Ley de conservación del valor con tolerancia híbrida y sensibilidad
# ---------------------------------------------------------------------------

def _validar_coherencia_matematica(
    cantidad: float,
    precio_unitario: float,
    valor_total: float,
    tau: float = TOLERANCIA_PORCENTUAL_COHERENCIA,
) -> CoherenceAnalysis:
    """
    Verifica la ley de conservación C = Q · P.

    Tolerancia híbrida (Kahan + IEEE 754):
        ε_abs  = max( ulp(Ĉ), |Ĉ| · ε_mach )
        ε_rel  = τ · |Ĉ|
        ε      = ε_abs + ε_rel
        coherente  ⇔  |Ĉ − C| ≤ ε

    Sensibilidad del sistema (gradiente euclídeo):
        ∇C = (∂C/∂Q, ∂C/∂P) = (P, Q)
        g  = ‖∇C‖₂ = √(P² + Q²)

    Número de condición del producto:
        κ = g · ‖(Q, P)‖₂ / max(|Ĉ|, ε_mach)
        (κ ≫ 1 ⇒ inestabilidad numérica local).

    Returns
    -------
    CoherenceAnalysis
    """
    try:
        Q = float(cantidad)
        P = float(precio_unitario)
        VT = float(valor_total)
    except (ValueError, TypeError, OverflowError):
        return CoherenceAnalysis(
            coherente=False,
            error_absoluto=float("inf"),
            error_porcentual=float("inf"),
            tolerancia_aplicada=0.0,
            sensibilidad=float("inf"),
            condicion=float("inf"),
            ganancia_sistema=float("inf"),
            mensaje="Error de conversión numérica",
        )

    # Producto esperado (con posible overflow controlado)
    try:
        esperado = Q * P
    except OverflowError:
        esperado = math.copysign(float("inf"), Q * P)

    if not math.isfinite(esperado) or not math.isfinite(VT):
        return CoherenceAnalysis(
            coherente=False,
            error_absoluto=float("inf"),
            error_porcentual=float("inf"),
            tolerancia_aplicada=0.0,
            sensibilidad=math.hypot(P, Q) if math.isfinite(P) and math.isfinite(Q) else float("inf"),
            condicion=float("inf"),
            ganancia_sistema=max(abs(P), abs(Q)) if math.isfinite(P) and math.isfinite(Q) else float("inf"),
            mensaje=f"Valor no finito: esperado={esperado}, reportado={VT}",
        )

    error_abs = abs(esperado - VT)

    # Piso de ruido flotante local
    eps_abs = max(_ulp(esperado), abs(esperado) * EPS_MACHINE, 1e-15)
    eps_rel = tau * abs(esperado)
    tolerancia = eps_abs + eps_rel

    coherente = error_abs <= tolerancia
    denom = max(abs(esperado), 1e-15)
    error_pct = (error_abs / denom) * 100.0

    # Gradiente y condición
    sensibilidad = math.hypot(P, Q)                       # ‖∇C‖₂
    norm_vars = math.hypot(Q, P)                          # ‖(Q,P)‖₂
    condicion = (sensibilidad * norm_vars) / denom        # κ
    ganancia = max(abs(P), abs(Q))

    msg = (
        f"Esperado: {esperado:.8g}, Reportado: {VT:.8g}, "
        f"|Δ|={error_abs:.6e}, ε={tolerancia:.6e}"
    )
    if sensibilidad > SENSITIVITY_ALERT_THRESHOLD:
        msg += f" [ALTA SENSIBILIDAD g={sensibilidad:.4g}]"
    if condicion > 1e6:
        msg += f" [κ={condicion:.4g} CRÍTICO]"

    return CoherenceAnalysis(
        coherente=coherente,
        error_absoluto=error_abs,
        error_porcentual=error_pct,
        tolerancia_aplicada=tolerancia,
        sensibilidad=sensibilidad,
        condicion=condicion,
        ganancia_sistema=ganancia,
        mensaje=msg,
    )


# ---------------------------------------------------------------------------
# 1.2  Limpieza sintáctica de descripciones (normalización en el monoide libre)
# ---------------------------------------------------------------------------

def _limpiar_y_validar_descripcion(
    descripcion: Any,
) -> Tuple[Optional[str], bool, Optional[str]]:
    """
    Normaliza un campo descriptivo:
        1. NFC Unicode (forma canónica de composición).
        2. Eliminación de controles C0 (\\x00–\\x1f).
        3. Strip + rechazo del conjunto VALORES_DESCRIPCION_INVALIDOS.
        4. Truncamiento a MAX_DESCRIPCION_LENGTH.

    Returns
    -------
    (texto_limpio | None, es_valido, razon_rechazo | None)
    """
    if descripcion is None:
        return None, False, "Vacío"
    try:
        if pd.isna(descripcion):
            return None, False, "Vacío"
    except (ValueError, TypeError):
        pass

    try:
        s = unicodedata.normalize("NFC", str(descripcion))
        s = re.sub(r"[\x00-\x1f\x7f]", "", s).strip()
        if not s or s.lower() in VALORES_DESCRIPCION_INVALIDOS:
            return None, False, "Inválido"
        if len(s) > MAX_DESCRIPCION_LENGTH:
            s = s[:MAX_DESCRIPCION_LENGTH]
        return s, True, None
    except Exception as exc:  # noqa: BLE001
        return None, False, f"Error: {exc}"


# ---------------------------------------------------------------------------
# 1.3  Validaciones atómicas de campos
# ---------------------------------------------------------------------------

def _validate_extreme_costs(
    presupuesto_data: List[Dict[str, Any]],
    anomaly_validator: Optional[Any] = None,
) -> Tuple[List[Dict[str, Any]], ValidationMetrics]:
    """
    Detecta costos fuera del intervalo [COSTO_MINIMO_VALIDO, COSTO_MAXIMO_RAZONABLE]
    y valores no finitos.  Las anomalías estadísticas multivariantes se delegan
    a AnomalyValidator (Fase 3) cuando se proporciona.
    """
    metrics = ValidationMetrics()
    if not presupuesto_data:
        return [], metrics

    result = deepcopy(presupuesto_data)

    for item in result:
        if not isinstance(item, dict):
            continue
        val = item.get("VALOR_CONSTRUCCION_UN")
        if not _es_numero_valido(val):
            if val is not None:
                try:
                    f = float(val)
                    if math.isinf(f):
                        _agregar_alerta(
                            item, f"Valor infinito: {val}",
                            TipoAlerta.VALOR_INFINITO, metrics,
                        )
                except (ValueError, TypeError):
                    pass
            continue

        v = float(val)
        if v < COSTO_MINIMO_VALIDO:
            _agregar_alerta(
                item, f"Costo negativo: {v}",
                TipoAlerta.COSTO_NEGATIVO, metrics,
            )
        elif v > COSTO_MAXIMO_RAZONABLE:
            _agregar_alerta(
                item, f"Costo excesivo: {v}",
                TipoAlerta.COSTO_EXCESIVO, metrics,
            )

    metrics.total_items_procesados = len(result)
    metrics.items_con_alertas = sum(
        1 for x in result if isinstance(x, dict) and x.get("alertas")
    )
    return result, metrics


def _validate_quantity_and_coherence(
    apus_data: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], ValidationMetrics]:
    """
    Aplica la ley de conservación C = Q·P y recalcula Q cuando
    Q ≈ 0 ∧ VT > 0 ∧ P > 0  (inversa local Q* = VT / P).
    Emite alertas de incoherencia, alta sensibilidad y mal condicionamiento.
    """
    metrics = ValidationMetrics()
    if not apus_data:
        return [], metrics

    result = deepcopy(apus_data)

    for item in result:
        if not isinstance(item, dict):
            continue

        q = item.get("CANTIDAD")
        pr = item.get("VR_UNITARIO")
        vt = item.get("VALOR_TOTAL")

        if not (
            _es_numero_valido(q)
            and _es_numero_valido(pr)
            and _es_numero_valido(vt)
        ):
            # Detectar infinitos aislados
            for campo, etiqueta in (
                (q, "CANTIDAD"),
                (pr, "VR_UNITARIO"),
                (vt, "VALOR_TOTAL"),
            ):
                if campo is not None:
                    try:
                        if math.isinf(float(campo)):
                            _agregar_alerta(
                                item,
                                f"{etiqueta} infinito",
                                TipoAlerta.VALOR_INFINITO,
                                metrics,
                            )
                    except (ValueError, TypeError):
                        pass
            continue

        analysis = _validar_coherencia_matematica(
            float(q), float(pr), float(vt),
        )

        if not analysis.coherente:
            _agregar_alerta(
                item, analysis.mensaje,
                TipoAlerta.INCOHERENCIA_MATEMATICA, metrics,
            )

        if analysis.sensibilidad > SENSITIVITY_ALERT_THRESHOLD:
            _agregar_alerta(
                item,
                f"Sistema inestable (g={analysis.sensibilidad:.4g}, κ={analysis.condicion:.4g})",
                TipoAlerta.ALTA_SENSIBILIDAD,
                metrics,
            )

        if analysis.condicion > 1e6:
            _agregar_alerta(
                item,
                f"Condicionamiento crítico κ={analysis.condicion:.4g}",
                TipoAlerta.CONDICIONAMIENTO_CRITICO,
                metrics,
            )

        # Recálculo: Q≈0 pero VT>0 y P>0
        fq, fpr, fvt = float(q), float(pr), float(vt)
        if abs(fq) < 1e-12 and fvt > 1e-12 and fpr > 1e-12:
            nueva_q = fvt / fpr
            if math.isfinite(nueva_q):
                item["CANTIDAD"] = nueva_q
                _agregar_alerta(
                    item,
                    f"Q recalculada: {nueva_q:.8g} (= VT/P)",
                    TipoAlerta.CANTIDAD_RECALCULADA,
                    metrics,
                )

    metrics.total_items_procesados = len(result)
    metrics.items_con_alertas = sum(
        1 for x in result if isinstance(x, dict) and x.get("alertas")
    )
    return result, metrics


def _validate_descriptions(
    apus_data: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], ValidationMetrics]:
    """
    Corrige descripciones inválidas asignando fallback canónico
    y emite DESCRIPCION_FALTANTE / DESCRIPCION_CORREGIDA.
    """
    metrics = ValidationMetrics()
    if not apus_data:
        return [], metrics

    result = deepcopy(apus_data)
    for item in result:
        if not isinstance(item, dict):
            continue
        original = item.get("DESCRIPCION_INSUMO")
        limpia, valida, _razon = _limpiar_y_validar_descripcion(original)
        if not valida:
            item["DESCRIPCION_INSUMO"] = "Insumo sin descripción"
            _agregar_alerta(
                item, "Descripción inválida o ausente",
                TipoAlerta.DESCRIPCION_FALTANTE, metrics,
            )
        elif limpia is not None and limpia != original:
            item["DESCRIPCION_INSUMO"] = limpia
            _agregar_alerta(
                item, "Descripción normalizada (NFC/strip/trunc)",
                TipoAlerta.DESCRIPCION_CORREGIDA, metrics,
            )

    metrics.total_items_procesados = len(result)
    metrics.items_con_alertas = sum(
        1 for x in result if isinstance(x, dict) and x.get("alertas")
    )
    return result, metrics


# ---------------------------------------------------------------------------
# 1.4  MORFISMO FINAL DE FASE 1 → DOMINIO DE FASE 2
# ---------------------------------------------------------------------------
# Definición formal:
#   phase1_to_topological_domain :
#       (presupuesto_validado, apus_detail_validado)
#           →  IncidenceDomain = {
#                 df_apus    : DataFrame[CODIGO_APU, …],
#                 df_insumos: DataFrame[APU_CODIGO, DESCRIPCION_INSUMO, …],
#                 validated_store: Dict   # copia limpia para F₃
#              }
#
# Este morfismo es el *último método de la Fase 1* y, por construcción
# categórica, la *entrada canónica de la Fase 2*.  Toda la topología
# bipartita se construye exclusivamente sobre IncidenceDomain.
# ---------------------------------------------------------------------------

@dataclass
class IncidenceDomain:
    """
    Dominio de incidencia producido por F₁ y consumido por F₂.
    Representa el 1-esqueleto del complejo simplicial bipartito:
        U = APUs,  V = Insumos,  E ⊆ U × V.
    """
    df_apus: pd.DataFrame
    df_insumos: pd.DataFrame
    validated_store: Dict[str, Any]
    metricas_fase1: Dict[str, Any]


def phase1_to_topological_domain(
    validated_store: Dict[str, Any],
    metricas_fase1: Dict[str, Any],
) -> IncidenceDomain:
    """
    Morfismo final de Fase 1 / morfismo inicial de Fase 2.

    Construye el par de DataFrames tipados que codifican el grafo de
    incidencia APU–Insumo, preservando el store validado para la
    termodinámica (Fase 3).

    Parameters
    ----------
    validated_store :
        Dict con claves al menos ``presupuesto`` y/o ``apus_detail``,
        ya filtrados y anotados por las validaciones microscópicas.
    metricas_fase1 :
        Diccionario de métricas acumuladas en Fase 1.

    Returns
    -------
    IncidenceDomain
        Objeto de dominio listo para ``BipartiteTopology`` y
        ``compute_pyramidal_metrics``.
    """
    presupuesto = validated_store.get("presupuesto") or []
    apus_detail = validated_store.get("apus_detail") or []

    df_apus = pd.DataFrame(presupuesto) if presupuesto else pd.DataFrame()
    df_insumos = pd.DataFrame(apus_detail) if apus_detail else pd.DataFrame()

    # Normalización de columnas canónicas para el funtor topológico
    if not df_apus.empty:
        if "CODIGO" in df_apus.columns and "CODIGO_APU" not in df_apus.columns:
            df_apus = df_apus.rename(columns={"CODIGO": "CODIGO_APU"})
        if "CODIGO_APU" not in df_apus.columns and len(df_apus.columns) > 0:
            # Fallback: primera columna como identificador
            df_apus = df_apus.rename(columns={df_apus.columns[0]: "CODIGO_APU"})

    if not df_insumos.empty:
        if "DESCRIPCION_INSUMO_NORM" not in df_insumos.columns:
            if "DESCRIPCION_INSUMO" in df_insumos.columns:
                df_insumos["DESCRIPCION_INSUMO_NORM"] = (
                    df_insumos["DESCRIPCION_INSUMO"]
                    .astype(str)
                    .str.normalize("NFC")
                    .str.strip()
                    .str.lower()
                )
        # Asegurar columna de referencia APU
        if "APU_CODIGO" not in df_insumos.columns and "CODIGO_APU" in df_insumos.columns:
            df_insumos = df_insumos.rename(columns={"CODIGO_APU": "APU_CODIGO"})

    return IncidenceDomain(
        df_apus=df_apus,
        df_insumos=df_insumos,
        validated_store=validated_store,
        metricas_fase1=metricas_fase1,
    )


# ============================================================================
# FIN FASE 1
# El objeto IncidenceDomain es, por definición formal, el inicio de la Fase 2.
# ============================================================================


# ============================================================================
# FASE 2 — TOPOLOGÍA BIPARTITA Y RESILIENCIA ESTRUCTURAL
# ============================================================================
# Entrada canónica: IncidenceDomain (salida de phase1_to_topological_domain).
#
# Procesamiento:
#   1. Construcción del grafo bipartito G = (U ∪ V, E), U∩V=∅, E⊆U×V.
#   2. Matriz de incidencia / adyacencia rectangular A ∈ {0,1}^{|U|×|V|}.
#   3. Laplaciano bipartito
#          L = [[ D_U , −A ],
#               [ −Aᵀ , D_V ]]  ∈ Sym_{|U|+|V|}(ℝ)
#      con espectro real 0 = λ₁ ≤ λ₂ ≤ ⋯ ≤ λ_{n}.
#   4. β₀ = mult(λ=0)  (número de componentes conexas del 1-esqueleto).
#   5. λ₂ = conectividad algebraica (Fiedler);  λ₂ > 0 ⇔ G conexo.
#   6. Nodos flotantes: U₀ = {u ∈ U | (A 1_V)(u) = 0} = proyección de
#      ker(Bᵀ) sobre el bloque APU (B = operador frontera ∂₁).
#   7. SPOF: insumos v con deg(v) ≥ max(2, ⌈0.1·|U|⌉).
#   8. Ψ = tanh(|V|/|U|)  (índice de estabilidad piramidal).
#   9. ρ(A) = σ_max(A)  (radio espectral de la adyacencia rectangular).
#
# El morfismo final `phase2_to_thermodynamic_state` empaqueta
# (IncidenceDomain, PyramidalMetrics) → StructuralState, entrada de F₃.
# ============================================================================

class BipartiteTopology:
    """
    Analizador topológico del complejo simplicial bipartito APU–Insumo.

    Desde la homología simplicial: el grafo es un 1-complejo K.
    El operador frontera ∂₁ : C₁(K) → C₀(K) se representa por la matriz
    de incidencia orientada; aquí usamos la adyacencia no orientada A
    y el laplaciano combinatorio L = ∂₁ ∂₁ᵀ (bloque bipartito).
    """

    @staticmethod
    def build_graph(
        apus_df: pd.DataFrame,
        insumos_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Construye el grafo bipartito de incidencia.

        Returns
        -------
        dict con claves:
            nodos_apus    : List[str]          # U
            nodos_insumos : List[str]          # V
            aristas       : List[Tuple[str,str]]  # E ⊆ U×V
            adj_matrix    : np.ndarray[int8]   # A ∈ {0,1}^{|U|×|V|}
            apu_to_idx    : Dict[str,int]
            insumo_to_idx : Dict[str,int]
        """
        empty = {
            "nodos_apus": [],
            "nodos_insumos": [],
            "aristas": [],
            "adj_matrix": np.zeros((0, 0), dtype=np.int8),
            "apu_to_idx": {},
            "insumo_to_idx": {},
        }

        if apus_df is None or apus_df.empty:
            return empty

        # Identificador de APU
        apu_col = "CODIGO_APU" if "CODIGO_APU" in apus_df.columns else (
            apus_df.columns[0] if len(apus_df.columns) else None
        )
        if apu_col is None:
            return empty

        apus_codes = (
            apus_df[apu_col].dropna().astype(str).unique().tolist()
        )

        # Identificador de insumo
        if insumos_df is None or insumos_df.empty:
            return {
                **empty,
                "nodos_apus": apus_codes,
                "apu_to_idx": {a: i for i, a in enumerate(apus_codes)},
            }

        insumo_col = (
            "DESCRIPCION_INSUMO_NORM"
            if "DESCRIPCION_INSUMO_NORM" in insumos_df.columns
            else (
                "DESCRIPCION_INSUMO"
                if "DESCRIPCION_INSUMO" in insumos_df.columns
                else None
            )
        )
        if insumo_col is None:
            return {
                **empty,
                "nodos_apus": apus_codes,
                "apu_to_idx": {a: i for i, a in enumerate(apus_codes)},
            }

        insumos_unique = (
            insumos_df[insumo_col].dropna().astype(str).unique().tolist()
        )

        if not apus_codes or not insumos_unique:
            return {
                "nodos_apus": apus_codes,
                "nodos_insumos": insumos_unique,
                "aristas": [],
                "adj_matrix": np.zeros(
                    (len(apus_codes), len(insumos_unique)), dtype=np.int8
                ),
                "apu_to_idx": {a: i for i, a in enumerate(apus_codes)},
                "insumo_to_idx": {s: j for j, s in enumerate(insumos_unique)},
            }

        apu_to_idx = {apu: i for i, apu in enumerate(apus_codes)}
        insumo_to_idx = {ins: j for j, ins in enumerate(insumos_unique)}
        adj = np.zeros((len(apus_codes), len(insumos_unique)), dtype=np.int8)
        aristas: List[Tuple[str, str]] = []

        apu_ref_col = "APU_CODIGO" if "APU_CODIGO" in insumos_df.columns else None
        if apu_ref_col is not None:
            valid = insumos_df.dropna(subset=[apu_ref_col, insumo_col])
            for _, row in valid.iterrows():
                a = str(row[apu_ref_col])
                i = str(row[insumo_col])
                if a in apu_to_idx and i in insumo_to_idx:
                    ii, jj = apu_to_idx[a], insumo_to_idx[i]
                    if adj[ii, jj] == 0:
                        adj[ii, jj] = 1
                        aristas.append((a, i))

        return {
            "nodos_apus": apus_codes,
            "nodos_insumos": insumos_unique,
            "aristas": aristas,
            "adj_matrix": adj,
            "apu_to_idx": apu_to_idx,
            "insumo_to_idx": insumo_to_idx,
        }

    @staticmethod
    def compute_laplacian_spectrum(
        adj_matrix: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Espectro del laplaciano bipartito L.

        Construcción:
            D_U = diag(A 1_V),  D_V = diag(Aᵀ 1_U)
            L   = [[ D_U , −A ],
                   [ −Aᵀ , D_V ]]

        Propiedades:
            • L ≽ 0 (semidefinida positiva), espectro real.
            • mult(λ=0) = β₀ = número de componentes conexas.
            • λ₂ = conectividad algebraica (Fiedler value).
            • Si G es conexo, λ₂ > 0.

        Returns
        -------
        dict: eigenvalues (asc), algebraic_connectivity, connected_components
        """
        if adj_matrix is None or adj_matrix.size == 0:
            return {
                "eigenvalues": np.array([], dtype=float),
                "algebraic_connectivity": 0.0,
                "connected_components": 0,
            }

        A = np.asarray(adj_matrix, dtype=float)
        m, n = A.shape
        if m == 0 or n == 0:
            return {
                "eigenvalues": np.array([], dtype=float),
                "algebraic_connectivity": 0.0,
                "connected_components": max(m, n),
            }

        d_u = A.sum(axis=1)          # grados de APUs
        d_v = A.sum(axis=0)          # grados de insumos
        D_u = np.diag(d_u)
        D_v = np.diag(d_v)

        # Bloques del laplaciano
        top = np.hstack([D_u, -A])
        bottom = np.hstack([-A.T, D_v])
        L = np.vstack([top, bottom])

        # Simetrización numérica (elimina errores de redondeo antisimétricos)
        L = 0.5 * (L + L.T)

        # Descomposición espectral simétrica
        try:
            eigvals = eigh(L, eigvals_only=True)
        except np.linalg.LinAlgError:
            # Fallback: SVD del laplaciano (valores singulares ≈ |autovalores|)
            eigvals = svd(L, compute_uv=False)
            eigvals = np.sort(eigvals)

        eigvals = np.real(eigvals)
        eigvals = np.maximum(eigvals, 0.0)  # clip ruido negativo numérico
        eigvals.sort()

        beta0 = int(np.sum(eigvals < SPECTRAL_ZERO_TOL))
        # λ₂: primer autovalor estrictamente positivo (o el de índice β₀)
        if beta0 < len(eigvals):
            lambda2 = float(eigvals[beta0])
        else:
            lambda2 = 0.0

        return {
            "eigenvalues": eigvals,
            "algebraic_connectivity": lambda2,
            "connected_components": max(beta0, 1) if (m + n) > 0 else 0,
        }

    @staticmethod
    def detect_floating_nodes(
        apus_df: pd.DataFrame,
        insumos_df: pd.DataFrame,
    ) -> List[str]:
        """
        APUs sin incidencia en E:  U₀ = U \\ π_U(E).

        Interpretación homológica: son los 0-ciclos generados por vértices
        aislados del bloque U; equivalen a la proyección de ker(Bᵀ) sobre U
        cuando no hay aristas incidentes (grado nulo).
        """
        if apus_df is None or apus_df.empty:
            return []

        apu_col = (
            "CODIGO_APU" if "CODIGO_APU" in apus_df.columns
            else apus_df.columns[0]
        )
        apus_all = set(
            apus_df[apu_col].dropna().astype(str).unique()
        )

        if insumos_df is None or insumos_df.empty:
            return sorted(apus_all)

        apu_col_ins = "APU_CODIGO" if "APU_CODIGO" in insumos_df.columns else None
        if apu_col_ins is None:
            return sorted(apus_all)

        apus_con_insumos = set(
            insumos_df[apu_col_ins].dropna().astype(str).unique()
        )
        return sorted(apus_all - apus_con_insumos)

    @staticmethod
    def detect_spof(
        adj_matrix: np.ndarray,
        nodos_insumos: List[str],
        umbral_impacto: int = 2,
        fraccion_critica: float = SPOF_IMPACT_FRACTION,
    ) -> List[Dict[str, Any]]:
        """
        Single Points of Failure: insumos v ∈ V cuyo grado
            deg(v) ≥ max(umbral_impacto, ⌈fracción_critica · |U|⌉)
        concentra un riesgo sistémico de desconexión masiva.

        Heurística de intermediación: alto grado en grafos bipartitos
        correlaciona con betweenness (Brandes) bajo condiciones de
        expansión moderada; se usa como proxy O(|E|) en lugar de O(|V|·|E|).
        """
        if adj_matrix is None or adj_matrix.size == 0 or not nodos_insumos:
            return []

        A = np.asarray(adj_matrix)
        grados = A.sum(axis=0).astype(int)
        total_apus = A.shape[0]
        umbral = max(umbral_impacto, int(math.ceil(total_apus * fraccion_critica)))

        spofs: List[Dict[str, Any]] = []
        for j, grado in enumerate(grados):
            if grado >= umbral and j < len(nodos_insumos):
                apus_idx = np.where(A[:, j] == 1)[0]
                spofs.append({
                    "insumo": nodos_insumos[j],
                    "impacto": int(grado),
                    "porcentaje_apus": round(
                        float(grado) / total_apus, 4
                    ) if total_apus else 1.0,
                    "apus_afectados_muestra": apus_idx[:5].tolist(),
                    "umbral_aplicado": umbral,
                })
        # Ordenar por impacto descendente
        spofs.sort(key=lambda s: s["impacto"], reverse=True)
        return spofs


def compute_pyramidal_metrics(
    apus_df: pd.DataFrame,
    insumos_df: pd.DataFrame,
) -> PyramidalMetrics:
    """
    Orquesta la Fase 2 a partir de los DataFrames del IncidenceDomain.

    Pipeline:
        build_graph → spectrum → floating → SPOF → Ψ → ρ(A)
    """
    graph = BipartiteTopology.build_graph(apus_df, insumos_df)
    adj = graph["adj_matrix"]

    spectrum = (
        BipartiteTopology.compute_laplacian_spectrum(adj)
        if adj.size > 0
        else {
            "eigenvalues": np.array([]),
            "algebraic_connectivity": 0.0,
            "connected_components": 1,
        }
    )

    n_apus = len(graph["nodos_apus"])
    n_insumos = len(graph["nodos_insumos"])

    # Índice de estabilidad piramidal: Ψ = tanh(|V| / max(|U|,1))
    psi_basico = n_insumos / max(n_apus, 1)
    psi_corregido = math.tanh(psi_basico)

    floating = BipartiteTopology.detect_floating_nodes(apus_df, insumos_df)
    spofs = (
        BipartiteTopology.detect_spof(adj, graph["nodos_insumos"])
        if adj.size > 0
        else []
    )

    # Radio espectral ρ(A) = σ_max(A)
    if adj.size > 0:
        try:
            singular_values = svd(adj.astype(float), compute_uv=False)
            spectral_radius = float(singular_values.max()) if len(singular_values) else 0.0
        except np.linalg.LinAlgError:
            spectral_radius = 0.0
    else:
        spectral_radius = 0.0

    return PyramidalMetrics(
        base_width=n_insumos,
        structure_load=n_apus,
        pyramid_stability_index=psi_corregido,
        floating_nodes=floating,
        connected_components=int(spectrum.get("connected_components", 1)),
        algebraic_connectivity=float(spectrum.get("algebraic_connectivity", 0.0)),
        spof_list=spofs,
        spectral_radius=spectral_radius,
        eigenvalues=spectrum.get("eigenvalues"),
    )


# ---------------------------------------------------------------------------
# 2.F  MORFISMO FINAL DE FASE 2 → ESTADO ESTRUCTURAL PARA FASE 3
# ---------------------------------------------------------------------------
# Definición formal:
#   phase2_to_thermodynamic_state :
#       (IncidenceDomain, PyramidalMetrics)
#           →  StructuralState = {
#                 validated_store, metricas_fase1, pyramidal, incidence
#              }
# Este es el *último método de la Fase 2* y la *entrada canónica de la Fase 3*.
# ---------------------------------------------------------------------------

@dataclass
class StructuralState:
    """
    Estado estructural producido por F₂ y consumido por F₃.
    Empaqueta el store validado, las métricas de Fase 1 y el análisis
    topológico de Fase 2 en un único objeto de paso.
    """
    validated_store: Dict[str, Any]
    metricas_fase1: Dict[str, Any]
    pyramidal: PyramidalMetrics
    incidence: IncidenceDomain


def phase2_to_thermodynamic_state(
    incidence: IncidenceDomain,
    pyramidal: PyramidalMetrics,
) -> StructuralState:
    """
    Morfismo final de Fase 2 / morfismo inicial de Fase 3.

    Parameters
    ----------
    incidence :
        Dominio de incidencia (salida de phase1_to_topological_domain).
    pyramidal :
        Métricas piramidales (salida de compute_pyramidal_metrics).

    Returns
    -------
    StructuralState
        Estado listo para el evaluador termodinámico.
    """
    return StructuralState(
        validated_store=incidence.validated_store,
        metricas_fase1=incidence.metricas_fase1,
        pyramidal=pyramidal,
        incidence=incidence,
    )


# ============================================================================
# FIN FASE 2
# El objeto StructuralState es, por definición formal, el inicio de la Fase 3.
# ============================================================================


# ============================================================================
# FASE 3 — TERMODINÁMICA DE LA INFORMACIÓN Y CONTROL DE EJECUCIÓN
# ============================================================================
# Entrada canónica: StructuralState (salida de phase2_to_thermodynamic_state).
#
# Procesamiento:
#   1. Entropía de Shannon sobre la distribución empírica de tipos de alerta:
#          H = −Σ pᵢ log₂ pᵢ ,   H_norm = H / log₂(K) ∈ [0,1].
#   2. Entropía de von Neumann (análogo cuántico): se forma la matriz densidad
#          ρ = diag(p)  (estado clásico en base de alertas);
#          S_vN = −Tr(ρ log₂ ρ) = H  (coincide en el caso diagonal).
#   3. Temperatura de ingesta:
#          T = 60 · tasa_alertas + 40 · (1 − Ψ)  ∈ [0,100].
#   4. Energía libre proxy: F̃ = T · H_norm  (análogo de Helmholtz).
#   5. Clasificación: ESTABLE (T<20) | METASTABLE (T<50) | CAÓTICO (T≥50).
#   6. Detección de anomalías multivariante (Z-Score + IQR + Isolation Forest)
#      con votación por mayoría (≥ 2 de 3).
# ============================================================================

class AnomalyValidator:
    """
    Detector de anomalías multivariante por ensamble:
        M₁ = Z-Score  (|z| > k·σ),
        M₂ = IQR      (x ∉ [Q1 − λ·IQR, Q3 + λ·IQR]),
        M₃ = Isolation Forest (si sklearn disponible).
    Confirmación por votación:  Σ 𝟙_{Mᵢ} ≥ 2.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        defaults = {
            "zscore_threshold": 3.0,
            "iqr_multiplier": 1.5,
            "isolation_forest_samples": 100,
            "contamination": "auto",
        }
        self.config = {**defaults, **(config or {})}

    def detect_cost_anomalies(
        self,
        cost_data: List[Dict[str, Any]],
        campo_costo: str = "VALOR_CONSTRUCCION_UN",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Marca cada ítem con clave ``anomalias`` si ≥ 2 métodos coinciden.
        """
        if not cost_data:
            return [], {"total_anomalias": 0}

        valores: List[float] = []
        indices: List[int] = []
        for idx, item in enumerate(cost_data):
            if not isinstance(item, dict):
                continue
            val = item.get(campo_costo)
            if _es_numero_valido(val):
                valores.append(float(val))
                indices.append(idx)

        if not valores:
            return cost_data, {"total_anomalias": 0}

        X = np.asarray(valores, dtype=float).reshape(-1, 1)
        flat = X.ravel()

        # M₁: Z-Score
        mean = float(np.mean(flat))
        std = float(np.std(flat))
        safe_std = max(std, 1e-15)
        z_anom = np.abs(flat - mean) > self.config["zscore_threshold"] * safe_std

        # M₂: IQR robusto
        q1, q3 = float(np.percentile(flat, 25)), float(np.percentile(flat, 75))
        iqr = q3 - q1
        lower = q1 - self.config["iqr_multiplier"] * iqr
        upper = q3 + self.config["iqr_multiplier"] * iqr
        iqr_anom = (flat < lower) | (flat > upper)

        # M₃: Isolation Forest
        if_anom = np.zeros(len(flat), dtype=bool)
        try:
            from sklearn.ensemble import IsolationForest
            n_samples = min(int(self.config["isolation_forest_samples"]), len(flat))
            n_samples = max(n_samples, 1)
            model = IsolationForest(
                contamination=self.config["contamination"],
                max_samples=n_samples,
                random_state=42,
            )
            preds = model.fit_predict(X)
            if_anom = preds == -1
        except ImportError:  # pragma: no cover
            pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("IsolationForest no aplicable: %s", exc)

        votes = z_anom.astype(int) + iqr_anom.astype(int) + if_anom.astype(int)
        confirmados = votes >= 2

        datos = deepcopy(cost_data)
        conteo = 0
        for i, idx in enumerate(indices):
            if confirmados[i]:
                datos[idx].setdefault("anomalias", []).append({
                    "tipo": "COSTO_ANOMALO",
                    "valor": float(flat[i]),
                    "score": int(votes[i]),
                    "metodos": {
                        "zscore": bool(z_anom[i]),
                        "iqr": bool(iqr_anom[i]),
                        "isolation_forest": bool(if_anom[i]),
                    },
                })
                conteo += 1

        stats = {
            "media": mean,
            "std": std,
            "iqr": float(iqr),
            "q1": q1,
            "q3": q3,
            "lower_fence": lower,
            "upper_fence": upper,
        }
        return datos, {"total_anomalias": conteo, "estadisticas": stats}


class ThermodynamicEvaluator:
    """
    Calcula la entropía de calidad, la temperatura de ingesta y el
    estado de estabilidad del tensor de datos.

    Entropía de Shannon:
        H = −Σᵢ pᵢ log₂(pᵢ + ε) ,   H_norm = H / log₂(K)

    Entropía de von Neumann (estado clásico diagonal):
        ρ = diag(p) ,  S_vN = −Tr(ρ log₂ ρ) = H

    Temperatura:
        T = 60 · tasa_alertas + 40 · (1 − Ψ)

    Energía libre proxy:
        F̃ = T · H_norm
    """

    @staticmethod
    def entropy_of_alerts(
        items: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Entropía de Shannon (y von Neumann diagonal) de la distribución
        empírica de tipos de alerta.
        """
        tipos: List[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            for a in item.get("alertas") or []:
                if isinstance(a, dict):
                    tipos.append(str(a.get("tipo", "DESCONOCIDO")))
            for a in item.get("anomalias") or []:
                if isinstance(a, dict):
                    tipos.append(str(a.get("tipo", "ANOMALIA")))

        if not tipos:
            return {
                "entropia": 0.0,
                "entropia_norm": 0.0,
                "entropia_von_neumann": 0.0,
                "n_tipos": 0,
                "n_alertas": 0,
            }

        unique, counts = np.unique(tipos, return_counts=True)
        probs = counts.astype(float) / float(len(tipos))
        # Shannon
        H = float(-np.sum(probs * np.log2(probs + 1e-15)))
        K = len(unique)
        H_max = math.log2(K) if K > 1 else 1.0
        H_norm = H / H_max if H_max > 0 else 0.0

        # von Neumann: ρ = diag(p) ⇒ S_vN = H
        S_vN = H

        return {
            "entropia": H,
            "entropia_norm": float(H_norm),
            "entropia_von_neumann": float(S_vN),
            "n_tipos": K,
            "n_alertas": len(tipos),
        }

    @staticmethod
    def compute_quality_temperature(
        data_store: Dict[str, Any],
        pyramidal: Optional[PyramidalMetrics] = None,
        alpha: float = 60.0,
        beta: float = 40.0,
    ) -> Dict[str, float]:
        """
        Temperatura de ingesta:
            T = α · tasa_alertas + β · (1 − Ψ) ,  acotada a [0, 100].
        """
        total_items = 0
        alert_items = 0
        for key in ("presupuesto", "apus_detail"):
            items = data_store.get(key) or []
            if not items:
                continue
            total_items += len(items)
            alert_items += sum(
                1
                for it in items
                if isinstance(it, dict) and (it.get("alertas") or it.get("anomalias"))
            )

        tasa = alert_items / max(total_items, 1)
        psi = (
            float(pyramidal.pyramid_stability_index)
            if pyramidal is not None
            else 1.0
        )
        temp = alpha * tasa + beta * (1.0 - psi)
        temp = min(100.0, max(0.0, temp))
        return {
            "tasa_alertas": float(tasa),
            "psi": float(psi),
            "temperatura": float(temp),
            "total_items": total_items,
            "alert_items": alert_items,
        }

    @staticmethod
    def classify_stability(temp: float) -> str:
        """Clasificación termodinámica del estado del tensor."""
        if temp < 20.0:
            return "ESTABLE"
        if temp < 50.0:
            return "METASTABLE"
        return "CAOTICO"

    @classmethod
    def evaluate(
        cls,
        structural: StructuralState,
    ) -> ThermodynamicState:
        """
        Evaluación termodinámica completa a partir del StructuralState.
        Morfismo principal de la Fase 3:
            F₃ : StructuralState → ThermodynamicState
        """
        store = structural.validated_store
        pyramidal = structural.pyramidal

        all_items: List[Dict[str, Any]] = []
        for key in ("presupuesto", "apus_detail"):
            items = store.get(key) or []
            all_items.extend(it for it in items if isinstance(it, dict))

        entropy_info = cls.entropy_of_alerts(all_items)
        temp_info = cls.compute_quality_temperature(store, pyramidal)
        stability = cls.classify_stability(temp_info["temperatura"])

        H_norm = entropy_info["entropia_norm"]
        T = temp_info["temperatura"]
        free_energy = T * H_norm  # proxy de Helmholtz F̃ = T · H_norm

        return ThermodynamicState(
            entropia_shannon=entropy_info["entropia"],
            entropia_normalizada=H_norm,
            entropia_von_neumann=entropy_info["entropia_von_neumann"],
            tasa_alertas=temp_info["tasa_alertas"],
            psi_estructural=temp_info["psi"],
            temperatura_ingesta=T,
            estabilidad=stability,
            free_energy_proxy=free_energy,
        )


# ============================================================================
# ORQUESTADOR PRINCIPAL — Composición F₃ ∘ F₂ ∘ F₁
# ============================================================================
# En el lenguaje de la teoría de categorías, el orquestador es el
# compuesto de funtores:
#     validate_and_clean_data  ≅  F₃ ∘ F₂ ∘ F₁
# donde
#     F₁ : DataStore → IncidenceDomain
#     F₂ : IncidenceDomain → StructuralState
#     F₃ : StructuralState → ThermodynamicState  (inyectado en el store)
# ============================================================================

def validate_and_clean_data(
    data_store: Dict[str, Any],
    skip_on_error: bool = True,
    validaciones_habilitadas: Optional[Dict[str, bool]] = None,
    telemetry_context: Optional[TelemetryContext] = None,
    aplicar_analisis_termico: bool = True,
    anomaly_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Orquesta el flujo completo en 3 fases anidadas:

        Fase 1 – Validaciones microscópicas
                 → phase1_to_topological_domain
        Fase 2 – Topología bipartita
                 → phase2_to_thermodynamic_state
        Fase 3 – Termodinámica de la información
                 → ThermodynamicState embebido en el store

    Parameters
    ----------
    data_store :
        Diccionario con al menos ``presupuesto`` y/o ``apus_detail``.
    skip_on_error :
        Si True, errores no fatales se registran y se continúa.
    validaciones_habilitadas :
        Flags opcionales por sub-validación (reservado para extensión).
    telemetry_context :
        Contexto de telemetría opcional.
    aplicar_analisis_termico :
        Si False, omite Fase 3 (solo F₁∘F₂).
    anomaly_config :
        Configuración del AnomalyValidator.

    Returns
    -------
    Dict[str, Any]
        Store enriquecido con alertas, pyramidal_metrics,
        quality_entropy_analysis y validation_summary.
    """
    logger.info("=" * 80)
    logger.info("Tribunal de Coherencia Termodinámica — Fases 1 → 2 → 3 (v4.0)")
    logger.info("=" * 80)

    if telemetry_context:
        try:
            telemetry_context.start_step("validate_data")
        except Exception:  # noqa: BLE001
            pass

    if not data_store or not isinstance(data_store, dict):
        return {
            "error": "Invalid data_store",
            "validation_summary": {"exito": False},
        }

    result = deepcopy(data_store)
    metricas: Dict[str, Any] = {}
    flags = validaciones_habilitadas or {}

    anomaly_detector = AnomalyValidator(config=anomaly_config)

    # ===================================================================
    # FASE 1 — Microscopía de campos
    # ===================================================================
    try:
        # 1.a  Presupuesto: costos extremos
        if "presupuesto" in result and flags.get("extreme_costs", True):
            result["presupuesto"], met_pres = _validate_extreme_costs(
                result["presupuesto"], anomaly_detector,
            )
            metricas["presupuesto"] = met_pres.to_dict()

            # Anomalías estadísticas sobre costos de construcción
            if result["presupuesto"]:
                result["presupuesto"], anom_stats = anomaly_detector.detect_cost_anomalies(
                    result["presupuesto"],
                    campo_costo="VALOR_CONSTRUCCION_UN",
                )
                metricas["presupuesto_anomalias"] = anom_stats

        # 1.b  APU detail: coherencia C=Q·P + descripciones
        if "apus_detail" in result:
            if flags.get("quantity_coherence", True):
                result["apus_detail"], met_q = _validate_quantity_and_coherence(
                    result["apus_detail"],
                )
                metricas["apus_quantity"] = met_q.to_dict()

            if flags.get("descriptions", True):
                result["apus_detail"], met_d = _validate_descriptions(
                    result["apus_detail"],
                )
                metricas["apus_desc"] = met_d.to_dict()

            # Anomalías sobre precios unitarios
            if (
                result["apus_detail"]
                and isinstance(result["apus_detail"][0], dict)
                and "VR_UNITARIO" in result["apus_detail"][0]
            ):
                result["apus_detail"], anom_apu = anomaly_detector.detect_cost_anomalies(
                    result["apus_detail"],
                    campo_costo="VR_UNITARIO",
                )
                metricas["apus_anomalias"] = anom_apu

    except Exception as exc:
        logger.error("Fase 1 falló: %s", exc, exc_info=True)
        if not skip_on_error:
            raise
        metricas["fase1_error"] = str(exc)

    # Morfismo F₁ → dominio topológico (último método Fase 1 = inicio Fase 2)
    incidence = phase1_to_topological_domain(result, metricas)

    # ===================================================================
    # FASE 2 — Topología bipartita
    # ===================================================================
    pyramidal_metrics: Optional[PyramidalMetrics] = None
    try:
        pyramidal_metrics = compute_pyramidal_metrics(
            incidence.df_apus,
            incidence.df_insumos,
        )
        result["pyramidal_metrics"] = pyramidal_metrics.to_dict()

        # Anotar SPOFs y nodos flotantes como alertas estructurales
        if pyramidal_metrics.floating_nodes:
            for apu_code in pyramidal_metrics.floating_nodes[:MAX_ALERTAS_POR_ITEM]:
                # Buscar ítem correspondiente en presupuesto
                for item in result.get("presupuesto") or []:
                    if not isinstance(item, dict):
                        continue
                    code = item.get("CODIGO_APU") or item.get("CODIGO")
                    if code is not None and str(code) == str(apu_code):
                        _agregar_alerta(
                            item,
                            f"APU flotante (sin insumos): {apu_code}",
                            TipoAlerta.COMPONENTE_DESCONECTADA,
                        )
                        break

        for spof in pyramidal_metrics.spof_list[:10]:
            logger.warning(
                "SPOF detectado: insumo=%s impacto=%s (%.1f%% APUs)",
                spof.get("insumo"),
                spof.get("impacto"),
                100.0 * float(spof.get("porcentaje_apus", 0)),
            )

    except Exception as exc:
        logger.error("Fase 2 falló: %s", exc, exc_info=True)
        if not skip_on_error:
            raise
        metricas["fase2_error"] = str(exc)
        # Métricas degeneradas para no romper F₃
        pyramidal_metrics = PyramidalMetrics(
            base_width=0,
            structure_load=0,
            pyramid_stability_index=0.0,
            floating_nodes=[],
            connected_components=0,
            algebraic_connectivity=0.0,
            spof_list=[],
            spectral_radius=0.0,
        )
        result["pyramidal_metrics"] = pyramidal_metrics.to_dict()

    # Morfismo F₂ → estado estructural (último método Fase 2 = inicio Fase 3)
    structural = phase2_to_thermodynamic_state(incidence, pyramidal_metrics)

    # ===================================================================
    # FASE 3 — Termodinámica de la información
    # ===================================================================
    if aplicar_analisis_termico:
        try:
            thermo = ThermodynamicEvaluator.evaluate(structural)
            quality_analysis = thermo.to_dict()
            result["quality_entropy_analysis"] = quality_analysis
        except Exception as exc:
            logger.error("Fase 3 falló: %s", exc, exc_info=True)
            if not skip_on_error:
                raise
            quality_analysis = {
                "entropia_shannon": 0.0,
                "entropia_normalizada": 0.0,
                "entropia_von_neumann": 0.0,
                "tasa_alertas": 0.0,
                "psi_estructural": 0.0,
                "temperatura_ingesta": 0.0,
                "estabilidad": "DESCONOCIDO",
                "free_energy_proxy": 0.0,
            }
            result["quality_entropy_analysis"] = quality_analysis
            metricas["fase3_error"] = str(exc)
    else:
        quality_analysis = {
            "entropia_shannon": 0.0,
            "entropia_normalizada": 0.0,
            "entropia_von_neumann": 0.0,
            "tasa_alertas": 0.0,
            "psi_estructural": (
                pyramidal_metrics.pyramid_stability_index
                if pyramidal_metrics else 0.0
            ),
            "temperatura_ingesta": 0.0,
            "estabilidad": "OMITIDO",
            "free_energy_proxy": 0.0,
        }
        result["quality_entropy_analysis"] = quality_analysis

    # ===================================================================
    # Resumen final y telemetría
    # ===================================================================
    all_items = list(result.get("presupuesto") or []) + list(
        result.get("apus_detail") or []
    )
    total_alertas = sum(
        1
        for it in all_items
        if isinstance(it, dict) and (it.get("alertas") or it.get("anomalias"))
    )

    result["validation_metrics"] = metricas
    result["validation_summary"] = {
        "exito": True,
        "total_alertas": total_alertas,
        "thermal_status": quality_analysis.get("estabilidad", "DESCONOCIDO"),
        "algebraic_connectivity": (
            pyramidal_metrics.algebraic_connectivity
            if pyramidal_metrics else 0.0
        ),
        "connected_components": (
            pyramidal_metrics.connected_components
            if pyramidal_metrics else 0
        ),
        "spof_count": (
            len(pyramidal_metrics.spof_list) if pyramidal_metrics else 0
        ),
        "floating_nodes_count": (
            len(pyramidal_metrics.floating_nodes) if pyramidal_metrics else 0
        ),
    }

    if telemetry_context:
        try:
            telemetry_context.record_metric(
                "validation",
                "quality_entropy",
                quality_analysis.get("entropia_shannon", 0.0),
            )
            telemetry_context.record_metric(
                "validation",
                "temperatura_ingesta",
                quality_analysis.get("temperatura_ingesta", 0.0),
            )
            telemetry_context.end_step(
                "validate_data",
                "success",
                metadata=result["validation_summary"],
            )
        except Exception:  # noqa: BLE001
            pass

    logger.info(
        "Validación completada — alertas=%s, estado=%s, λ₂=%.6g, β₀=%s",
        total_alertas,
        quality_analysis.get("estabilidad"),
        (
            pyramidal_metrics.algebraic_connectivity
            if pyramidal_metrics else 0.0
        ),
        (
            pyramidal_metrics.connected_components
            if pyramidal_metrics else 0
        ),
    )
    return result


# ============================================================================
# API pública del módulo
# ============================================================================
__all__ = [
    # Enums y estructuras
    "TipoAlerta",
    "ValidationMetrics",
    "CoherenceAnalysis",
    "PyramidalMetrics",
    "ThermodynamicState",
    "IncidenceDomain",
    "StructuralState",
    # Fase 1
    "_es_numero_valido",
    "_validar_coherencia_matematica",
    "_limpiar_y_validar_descripcion",
    "_validate_extreme_costs",
    "_validate_quantity_and_coherence",
    "_validate_descriptions",
    "phase1_to_topological_domain",
    # Fase 2
    "BipartiteTopology",
    "compute_pyramidal_metrics",
    "phase2_to_thermodynamic_state",
    # Fase 3
    "AnomalyValidator",
    "ThermodynamicEvaluator",
    # Orquestador
    "validate_and_clean_data",
]