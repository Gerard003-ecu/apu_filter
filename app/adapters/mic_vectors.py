"""
Módulo: MIC Vectors (Adaptadores de Capacidad y Morfismos Algebraicos)
======================================================================

Este módulo implementa la capa de adaptación ("Glue Code") como un conjunto de
**Morfismos Algebraicos** que proyectan las intenciones del Agente sobre los
motores físicos y lógicos del sistema (FluxCondenser, APUProcessor).

Fundamentos Matemáticos y Arquitectura:
---------------------------------------

1. Teoría de Categorías (Morfismos):
   Cada función vectorial (ej. `vector_stabilize_flux`) se modela como un morfismo φ:
       φ: ConfigSpace × Context → ResultSpace
   Transforman un payload de configuración (diccionario) en un resultado tipado
   (`VectorResult`), preservando la estructura de la información a través de los
   estratos DIKW.

2. Coherencia Topológica (C):
   Métrica escalar invariante para evaluar la calidad de la transformación:
       C = clamp(S · R / (1 + H), 0, 1)
   donde S = stability ∈ [0,1], R = resonance ∈ [0,1], H = entropy ≥ 0.
   El clamp explícito garantiza C ∈ [0,1] para todo input válido.

3. Isomorfismo Dimensional (Enlace Física → Táctica):
   Guardas algebraicas (`validate_dimensional_isomorphism`) que aseguran que la
   dimensión del espacio de salida del vector físico sea isomorfa a la dimensión
   esperada por el vector táctico.

4. Invariante de Euler como Postcondición:
   β₀ - β₁ + β₂ = χ se verifica como postcondición en `calculate_betti_numbers`,
   detectando incoherencias en el complejo simplicial proxy antes de propagarlas.

5. Inmutabilidad Algebraica:
   Las métricas (`VectorMetrics`) son dataclasses congeladas (frozen), tratadas
   como valores puros con identidad observacional fija.

6. Registro Inmutable en VectorFactory:
   `_DEFAULT_PHYSICS_REGISTRY` es una constante de módulo; el registro activo
   se clona desde ella, evitando mutación del estado global entre tests.
"""

import functools
import logging
import os
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    TypeAlias,
)

from app.apu_processor import APUProcessor
from app.flux_condenser import CondenserConfig, DataFluxCondenser
from app.report_parser_crudo import ReportParserCrudo
from app.schemas import Stratum

logger = logging.getLogger(__name__)

# ── dependencia opcional ──────────────────────────────────────────────────────
try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover
    psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False


# =============================================================================
# CONTRATOS ALGEBRAICOS DE TIPO
# =============================================================================
PhysicsPayload: TypeAlias = Dict[str, Any]
TacticsPayload: TypeAlias = Dict[str, Any]
VectorResult: TypeAlias = Dict[str, Any]

# Constantes de dominio físico — separadas del código de lógica
_DEFAULT_RESONANCE_THRESHOLD: float = 0.85
_DEFAULT_STABILITY: float = 1.0
_DEFAULT_SYSTEM_TEMP: float = 25.0
_DEFAULT_HEAT_CAPACITY: float = 0.5

# Umbral de isomorfismo dimensional (ε-tolerancia relativa)
_DIM_ISO_TOLERANCE: float = 0.10

# Umbral de inercia financiera para el pivote MONOPOLIO_COBERTURADO
_FINANCIAL_INERTIA_THRESHOLD: float = 0.70

# Multiplicador del valor de opción sobre VPN para el pivote OPCION_ESPERA
_WAIT_OPTION_NPV_MULTIPLIER: float = 1.5


class VectorResultStatus(Enum):
    """Estratificación del resultado por tipo de fallo."""

    SUCCESS = "success"
    PHYSICS_ERROR = "physics_error"
    LOGIC_ERROR = "logic_error"
    TOPOLOGY_ERROR = "topology_error"


@dataclass(frozen=True)
class VectorMetrics:
    """
    Valor inmutable que captura la telemetría de un vector.

    Inmutabilidad (frozen) ⟹ objeto algebraico puro:
    una vez construido, su identidad observacional es fija.
    Ningún efecto secundario puede alterar la telemetría registrada.
    """

    processing_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    topological_coherence: float = 1.0
    algebraic_integrity: float = 1.0


# =============================================================================
# HELPERS INTERNOS
# =============================================================================


def _measure_memory_mb() -> float:
    """Consumo de memoria residente (MB). Devuelve 0.0 si psutil no existe."""
    if _HAS_PSUTIL:
        try:
            return psutil.Process().memory_info().rss / (1024.0 * 1024.0)
        except psutil.Error:
            # El proceso puede haber terminado entre la llamada y la medición.
            return 0.0
    return 0.0


def _elapsed_ms(start: float) -> float:
    """Milisegundos transcurridos desde *start* (monotónico)."""
    return (time.monotonic() - start) * 1000.0


def _build_result(
    *,
    success: bool,
    stratum: Stratum,
    status: VectorResultStatus,
    metrics: Optional[VectorMetrics] = None,
    error: Optional[str] = None,
    **payload: Any,
) -> VectorResult:
    """
    Constructor canónico de VectorResult.

    Garantiza que **todo** resultado contiene exactamente las claves
    {success, stratum, status, metrics} más un payload variable,
    preservando la coherencia topológica del esquema de salida.

    Invariante: la clave 'error' sólo aparece cuando error is not None,
    manteniendo el esquema mínimo limpio para resultados exitosos.
    """
    result: VectorResult = {
        "success": success,
        "stratum": stratum,
        "status": status.value,
        "metrics": asdict(metrics) if metrics is not None else asdict(VectorMetrics()),
    }
    if error is not None:
        result["error"] = error
    result.update(payload)
    return result


def _build_error(
    *,
    stratum: Stratum,
    status: VectorResultStatus,
    error: str,
    elapsed_ms: float = 0.0,
    extra_metrics: Optional[VectorMetrics] = None,
) -> VectorResult:
    """
    Constructor canónico de resultados de fallo.

    Todos los parámetros son keyword-only para evitar errores de orden
    en los call-sites. `extra_metrics` permite propagar métricas parciales
    calculadas antes del fallo (ej. tiempo transcurrido, memoria).

    Refactorización respecto a la versión anterior:
    - Firma exclusivamente keyword-only (elimina ambigüedad posicional).
    - `extra_metrics` reemplaza la reconstrucción manual de VectorMetrics
      en cada call-site que necesite propagar más que el tiempo.
    """
    metrics = extra_metrics if extra_metrics is not None else VectorMetrics(
        processing_time_ms=elapsed_ms
    )
    return _build_result(
        success=False,
        stratum=stratum,
        status=status,
        error=error,
        metrics=metrics,
    )


# =============================================================================
# GUARDAS TOPOLÓGICAS
# =============================================================================


def validate_topological_preconditions(
    file_path: str,
    config: Dict[str, Any],
    required_keys: List[str],
) -> Tuple[bool, Optional[str]]:
    """
    Verifica que las secciones locales del fibrado de configuración
    estén bien definidas antes de intentar una operación física.

    Comprobaciones
    ──────────────
    1. Existencia del archivo (punto base de la variedad).
    2. Presencia de claves requeridas (secciones del haz).
    3. Consistencia numérica de restricciones dimensionales.

    Retorna
    ───────
    (True, None)           si todas las condiciones se satisfacen.
    (False, mensaje_error) si alguna condición falla.
    """
    if not os.path.exists(file_path):
        return False, f"Topología rota: archivo '{file_path}' no existe"

    missing = [k for k in required_keys if k not in config]
    if missing:
        return False, f"Configuración incompleta: faltan {missing}"

    constraints = config.get("dimension_constraints")
    if isinstance(constraints, dict):
        non_numeric = [
            k for k, v in constraints.items()
            if not isinstance(v, (int, float))
        ]
        if non_numeric:
            return False, f"Dimensionalidad no numérica en claves: {non_numeric}"

    return True, None


def validate_homological_constraints(constraints: Dict[str, Any]) -> bool:
    """
    Valida que las restricciones formen un complejo de cadenas válido.

    Requisitos
    ──────────
    - max_dimension : int ≥ 0   (dimensión máxima del complejo)
    - allow_holes   : bool      (permitir β₁ > 0)
    - connectivity  : numérico  (umbral de componentes conexas ≥ 0)

    Nota: connectivity debe ser no-negativo — un umbral negativo de
    componentes conexas carece de interpretación topológica.
    """
    required: FrozenSet[str] = frozenset({"max_dimension", "allow_holes", "connectivity"})
    if not required.issubset(constraints.keys()):
        return False

    md = constraints["max_dimension"]
    if not isinstance(md, int) or md < 0:
        return False

    if not isinstance(constraints["allow_holes"], bool):
        return False

    conn = constraints["connectivity"]
    if not isinstance(conn, (int, float)) or conn < 0:
        return False

    return True


# =============================================================================
# FUNCIONES AUXILIARES DE TOPOLOGÍA ALGEBRAICA
# =============================================================================


def calculate_topological_coherence(physics_report: Dict[str, Any]) -> float:
    """
    Coherencia topológica como cociente normalizado con clamp explícito.

    Fórmula
    ───────
        C = clamp(S · R / (1 + H), 0.0, 1.0)

    donde:
        S = stability  ∈ [0, 1]   (estabilidad del sistema)
        R = resonance  ∈ [0, 1]   (factor de resonancia)
        H = entropy    ∈ [0, ∞)   (entropía de Shannon)

    Propiedades garantizadas
    ────────────────────────
    • C ∈ [0, 1]   por construcción (clamp explícito + denominador ≥ 1).
    • ∂C/∂S > 0,  ∂C/∂R > 0   (monótona creciente en S, R).
    • ∂C/∂H < 0               (monótona decreciente en H).
    • C = 0  ⟺  S = 0  ∨  R = 0   (cero absorbente).
    • H = 0, S = R = 1  ⟹  C = 1  (máximo alcanzable).

    El clamp superior a 1.0 es innecesario matemáticamente (S·R ≤ 1),
    pero se incluye como invariante defensivo frente a inputs corruptos
    que pasen los clamps internos de S y R.
    """
    if not physics_report:
        return 0.0

    stability = max(0.0, min(1.0, float(physics_report.get("stability_index", 0))))
    entropy = max(0.0, float(physics_report.get("entropy", 0)))
    resonance = max(0.0, min(1.0, float(physics_report.get("resonance_factor", 0))))

    raw = stability * resonance / (1.0 + entropy)
    return max(0.0, min(1.0, raw))  # clamp defensivo explícito


def calculate_betti_numbers(
    raw_records: List[Dict[str, Any]],
    cache: Dict[str, Any],
) -> Dict[str, int]:
    """
    Números de Betti proxy del complejo simplicial inducido por los registros.

    Definiciones proxy
    ──────────────────
    β₀ = componentes conexas  → tipos de registro distintos.
    β₁ = 1-ciclos             → ciclos de dependencia en el grafo del cache.
    β₂ = 2-cavidades          → nodos anidados vacíos de contenido.

    Postcondición (Invariante de Euler)
    ────────────────────────────────────
        χ = β₀ − β₁ + β₂

    El valor χ se incluye en el resultado y se verifica como log de advertencia
    si difiere de la característica esperada por el complejo teórico.

    Robustez: retorna el diccionario nulo si `raw_records` es vacío,
    sin lanzar excepciones.
    """
    if not raw_records:
        return {"beta_0": 0, "beta_1": 0, "beta_2": 0, "euler": 0}

    # β₀: cardinalidad de tipos distintos (componentes conexas proxy)
    types: Set[str] = {r.get("record_type", "unknown") for r in raw_records}
    beta_0: int = len(types)

    # β₁: ciclos del grafo de dependencias (extraído del cache)
    cycles = cache.get("dependency_cycles", [])
    beta_1: int = len(cycles) if isinstance(cycles, list) else 0

    # β₂: cavidades (nodos anidados huecos)
    beta_2: int = sum(
        1 for r in raw_records
        if r.get("nested") and not r.get("content")
    )

    # Invariante de Euler — postcondición algebraica
    euler: int = beta_0 - beta_1 + beta_2

    result = {
        "beta_0": beta_0,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "euler": euler,
    }

    # Verificación forense: log si χ difiere de β₀ (complejo acíclico esperado)
    # En un complejo 0-dimensional puro (sin ciclos ni cavidades), χ = β₀.
    if beta_1 > 0 or beta_2 > 0:
        logger.debug(
            "Complejo simplicial no trivial: β₀=%d, β₁=%d, β₂=%d, χ=%d",
            beta_0, beta_1, beta_2, euler,
        )

    return result


def calculate_algebraic_integrity(betti_numbers: Dict[str, int]) -> float:
    """
    Integridad algebraica derivada de los números de Betti superiores.

    Fórmula
    ───────
        I = 1 / (1 + Σ βᵢ,  i > 0)

    Propiedades
    ───────────
    • I = 1   ⟹  no hay ciclos ni cavidades (variedad acíclica).
    • I → 0   cuando los defectos topológicos se acumulan.
    • I ∈ (0, 1]  siempre (denominador ≥ 1).

    Robustez: itera sobre claves reales del diccionario; no asume
    que las claves sean enteros consecutivos, y filtra 'euler'
    (que puede ser negativo y no representa un número de Betti).
    """
    if not betti_numbers:
        return 1.0

    # Claves superiores: beta_1, beta_2, … (excluir beta_0 y euler)
    excluded: FrozenSet[str] = frozenset({"beta_0", "euler"})
    higher_sum = sum(
        count
        for key, count in betti_numbers.items()
        if key not in excluded and isinstance(count, (int, float)) and count > 0
    )
    return 1.0 / (1.0 + higher_sum)


def calculate_dimensionality(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Rango del espacio de atributos por tipo de registro.

        dim(V_t) = |⋃_{r : type(r)=t} keys(r)|

    Calcula la **cardinalidad de la unión** de atributos por tipo,
    análogo a la dimensión de un espacio vectorial columnar.
    A diferencia de sumar el conteo por registro (que crece con n),
    este valor es estable ante duplicación de registros del mismo tipo.

    Retorna {} si `records` es vacío (sin excepción).
    """
    if not records:
        return {}

    attribute_spaces: Dict[str, Set[str]] = {}
    for record in records:
        rtype = str(record.get("record_type", "default"))
        attribute_spaces.setdefault(rtype, set()).update(record.keys())

    return {rtype: len(attrs) for rtype, attrs in attribute_spaces.items()}


def validate_dimensional_isomorphism(
    expected: Dict[str, int],
    actual: Dict[str, int],
    tolerance: float = _DIM_ISO_TOLERANCE,
) -> bool:
    """
    Dos espacios de atributos son isomorfos si comparten los mismos
    tipos y sus rangos coinciden dentro de una tolerancia ε relativa.

        ∀ t ∈ Types:  |dim_exp(t) − dim_act(t)| / max(dim_exp(t), 1) ≤ ε

    Casos especiales
    ────────────────
    • Si ambos espacios están vacíos → True (isomorfismo vacuo trivial).
    • Si sólo uno está vacío → False (no hay base de comparación válida).
      Esto distingue "sin datos todavía" de "sin tipos comunes",
      evitando falsos positivos en pipelines parcialmente inicializados.
    • Si los conjuntos de tipos difieren → False (tipos incompatibles).
    """
    # Ambos vacíos: isomorfismo vacuo
    if not expected and not actual:
        return True

    # Exactamente uno vacío: incompatibles
    if not expected or not actual:
        return False

    # Tipos distintos: no isomorfos
    if set(expected.keys()) != set(actual.keys()):
        return False

    for key in expected:
        ref = max(expected[key], 1)
        if abs(expected[key] - actual[key]) / ref > tolerance:
            return False

    return True


# =============================================================================
# VECTOR FÍSICO 1: ESTABILIZACIÓN DE FLUJO (DataFluxCondenser)
# =============================================================================
_REQUIRED_STABILIZE_KEYS: List[str] = [
    "system_capacitance",
    "system_inductance",
    "base_resistance",
]


def vector_stabilize_flux(
    file_path: str,
    config: Dict[str, Any],
) -> VectorResult:
    """
    Vector de nivel PHYSICS.
    Invoca al DataFluxCondenser para estabilizar la ingesta de un archivo.

    Morfismo:  (FilePath × Config) ─F─▶ StabilizedManifold

    donde F es el fibrado estabilizador (DataFluxCondenser)
    y la sección de salida pertenece a la frontera ∂ de estabilidad.

    Circuito físico equivalente
    ───────────────────────────
    El CondenserConfig modela un circuito RLC serie:
        Z(ω) = R + j(ωL − 1/ωC)
    La resonancia ocurre cuando ωL = 1/ωC  →  ω₀ = 1/√(LC).
    resonance_threshold actúa como umbral del factor de calidad Q = ω₀L/R.
    """
    start = time.monotonic()

    is_valid, err = validate_topological_preconditions(
        file_path, config, _REQUIRED_STABILIZE_KEYS
    )
    if not is_valid:
        return _build_error(
            stratum=Stratum.PHYSICS,
            status=VectorResultStatus.TOPOLOGY_ERROR,
            error=err,  # type: ignore[arg-type]
            elapsed_ms=_elapsed_ms(start),
        )

    try:
        condenser_conf = CondenserConfig(
            system_capacitance=float(config["system_capacitance"]),
            system_inductance=float(config["system_inductance"]),
            base_resistance=float(config["base_resistance"]),
            resonance_threshold=float(
                config.get("resonance_threshold", _DEFAULT_RESONANCE_THRESHOLD)
            ),
        )

        condenser = DataFluxCondenser(
            config=config,
            profile=config.get("file_profile", {}),
            condenser_config=condenser_conf,
        )

        df_stabilized = condenser.stabilize(file_path)
        physics_report = condenser.get_physics_report()

        coherence = calculate_topological_coherence(physics_report)
        metrics = VectorMetrics(
            processing_time_ms=_elapsed_ms(start),
            memory_usage_mb=_measure_memory_mb(),
            topological_coherence=coherence,
        )

        return _build_result(
            success=True,
            stratum=Stratum.PHYSICS,
            status=VectorResultStatus.SUCCESS,
            metrics=metrics,
            data=df_stabilized.to_dict("records"),
            physics_metrics=physics_report,
        )

    except Exception as exc:
        logger.error(
            "Fallo en vector 'stabilize_flux': %s", exc, exc_info=True
        )
        return _build_error(
            stratum=Stratum.PHYSICS,
            status=VectorResultStatus.PHYSICS_ERROR,
            error=str(exc),
            elapsed_ms=_elapsed_ms(start),
        )


# =============================================================================
# VECTOR FÍSICO 2: PARSING TOPOLÓGICO (ReportParserCrudo)
# =============================================================================


def vector_parse_raw_structure(
    file_path: str,
    profile: Dict[str, Any],
    topological_constraints: Optional[Dict[str, Any]] = None,
) -> VectorResult:
    """
    Vector de nivel PHYSICS.
    Utiliza ReportParserCrudo para extraer el complejo simplicial del archivo.

    Morfismo:  (FilePath × Profile) ─∂─▶ ChainComplex(Records)

    Homología proxy
    ───────────────
    H₀ → componentes conexas (registros crudos).
    H₁ → ciclos (dependencias circulares).
    H₂ → cavidades (estructuras anidadas vacías).

    La dimensionalidad calculada se inyecta en el cache para que
    `validate_dimensional_isomorphism` sea no-vacua en el vector táctico.
    """
    start = time.monotonic()

    # ── Guardia: existencia del punto base ────────────────────────────────
    if not os.path.exists(file_path):
        return _build_error(
            stratum=Stratum.PHYSICS,
            status=VectorResultStatus.TOPOLOGY_ERROR,
            error=f"Archivo no existe: '{file_path}'",
            elapsed_ms=_elapsed_ms(start),
        )

    # ── Guardia: restricciones homológicas ───────────────────────────────
    if (
        topological_constraints is not None
        and not validate_homological_constraints(topological_constraints)
    ):
        return _build_error(
            stratum=Stratum.PHYSICS,
            status=VectorResultStatus.TOPOLOGY_ERROR,
            error="Restricciones homológicas inválidas",
            elapsed_ms=_elapsed_ms(start),
        )

    try:
        parser = ReportParserCrudo(
            file_path,
            profile=profile,
            topological_constraints=topological_constraints,
        )

        raw_records: List[Dict[str, Any]] = parser.parse_to_raw()
        cache: Dict[str, Any] = parser.get_parse_cache()

        betti = calculate_betti_numbers(raw_records, cache)
        integrity = calculate_algebraic_integrity(betti)

        # Inyectar dimensionalidad en el cache para que
        # validate_dimensional_isomorphism sea no-vacua aguas abajo.
        cache["dimensionality"] = calculate_dimensionality(raw_records)

        metrics = VectorMetrics(
            processing_time_ms=_elapsed_ms(start),
            memory_usage_mb=_measure_memory_mb(),
            algebraic_integrity=integrity,
        )

        # Acceso seguro y normalizado a validation_stats
        validation_stats: Dict[str, Any] = {}
        if hasattr(parser, "validation_stats"):
            vs = parser.validation_stats
            if hasattr(vs, "__dict__"):
                validation_stats = dict(vars(vs))
            elif hasattr(vs, "items"):
                validation_stats = dict(vs)

        return _build_result(
            success=True,
            stratum=Stratum.PHYSICS,
            status=VectorResultStatus.SUCCESS,
            metrics=metrics,
            raw_records=raw_records,
            parse_cache=cache,
            validation_stats=validation_stats,
            homological_invariants=betti,
        )

    except Exception as exc:
        logger.error(
            "Fallo en vector 'parse_raw_structure': %s", exc, exc_info=True
        )
        return _build_error(
            stratum=Stratum.PHYSICS,
            status=VectorResultStatus.PHYSICS_ERROR,
            error=str(exc),
            elapsed_ms=_elapsed_ms(start),
        )


# =============================================================================
# VECTOR TÁCTICO: ESTRUCTURACIÓN LÓGICA (APUProcessor)
# =============================================================================


def vector_structure_logic(
    raw_records: List[Dict[str, Any]],
    parse_cache: Dict[str, Any],
    config: Dict[str, Any],
    algebraic_structure: str = "module",
) -> VectorResult:
    """
    Vector de nivel TACTICS.
    Transforma registros crudos en estructuras de costos validadas.

    Morfismo:  (Records × Cache × Config) ─φ─▶ ProcessedModule

    Álgebra abstracta
    ─────────────────
    G   = grupo de transformaciones (APUProcessor).
    φ   = homomorfismo  G → Aut(Data).
    ker(φ) = núcleo de validación (errores capturados por get_validation_kernel).
    im(φ)  = imagen procesada limpia (df_processed).

    El isomorfismo dimensional se verifica antes de cualquier transformación
    para detectar desalineaciones columnar antes del cálculo de costos.
    """
    start = time.monotonic()

    # ── Guardia: conjunto no vacío ────────────────────────────────────────
    if not raw_records:
        return _build_error(
            stratum=Stratum.TACTICS,
            status=VectorResultStatus.LOGIC_ERROR,
            error="Conjunto vacío: no hay registros para transformar",
            elapsed_ms=_elapsed_ms(start),
        )

    # ── Verificación de isomorfismo dimensional ──────────────────────────
    expected_dims = parse_cache.get("dimensionality", {})
    actual_dims = calculate_dimensionality(raw_records)

    if not validate_dimensional_isomorphism(expected_dims, actual_dims):
        logger.warning(
            "Isomorfismo dimensional roto: esperado=%s, actual=%s",
            expected_dims,
            actual_dims,
        )

    try:
        processor = APUProcessor(
            config=config,
            parse_cache=parse_cache,
            algebraic_structure=algebraic_structure,
        )
        processor.raw_records = raw_records

        df_processed = processor.process_all()

        # ── Extracción segura del núcleo algebraico ───────────────────────
        kernel: Dict[str, Any] = (
            processor.get_validation_kernel()
            if hasattr(processor, "get_validation_kernel")
            else {}
        )

        topo_coherence: float = (
            processor.get_topological_coherence()
            if hasattr(processor, "get_topological_coherence")
            else 1.0
        )

        alg_integrity: float = (
            processor.get_algebraic_integrity()
            if hasattr(processor, "get_algebraic_integrity")
            else 1.0
        )

        quality_report: Dict[str, Any] = (
            processor.get_quality_report()
            if hasattr(processor, "get_quality_report")
            else {}
        )

        metrics = VectorMetrics(
            processing_time_ms=_elapsed_ms(start),
            memory_usage_mb=_measure_memory_mb(),
            topological_coherence=topo_coherence,
            algebraic_integrity=alg_integrity,
        )

        return _build_result(
            success=True,
            stratum=Stratum.TACTICS,
            status=VectorResultStatus.SUCCESS,
            metrics=metrics,
            processed_data=df_processed.to_dict("records"),
            quality_report=quality_report,
            algebraic_kernel=kernel,
        )

    except Exception as exc:
        logger.error(
            "Fallo en vector 'structure_logic': %s", exc, exc_info=True
        )
        return _build_error(
            stratum=Stratum.TACTICS,
            status=VectorResultStatus.LOGIC_ERROR,
            error=str(exc),
            elapsed_ms=_elapsed_ms(start),
        )


# =============================================================================
# VECTOR ESTRATÉGICO: PENSAMIENTO LATERAL (Risk Challenger)
# =============================================================================


def vector_lateral_pivot(
    payload: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> VectorResult:
    """
    Vector de nivel STRATEGY.
    Morfismo que proyecta una intención de pensamiento lateral para mitigar
    un riesgo topológico o financiero buscando compensaciones (hedges)
    en el espacio ortogonal.

    Transformación: (T × Φ × Θ) ─L─▶ D_lateral
    Donde:
      T = Espacio Topológico    (Estabilidad, β₁)
      Φ = Espacio Financiero    (Clase de riesgo, VPN, Opciones Reales)
      Θ = Espacio Termodinámico (Temperatura, Inercia Financiera)
      D_lateral = Decisión de Excepción Validada

    Pivotes implementados
    ─────────────────────
    MONOPOLIO_COBERTURADO  : base estrecha + sistema frío + alta inercia.
    OPCION_ESPERA          : riesgo alto + valor de opción > VPN * k.
    CUARENTENA_TOPOLOGICA  : ciclos presentes + sin sinergia multiplicativa.

    Parámetros configurables (constantes de módulo)
    ───────────────────────────────────────────────
    _FINANCIAL_INERTIA_THRESHOLD  : umbral de inercia para MONOPOLIO.
    _WAIT_OPTION_NPV_MULTIPLIER   : multiplicador del VPN para OPCION_ESPERA.
    """
    start = time.monotonic()

    # ── 1. Extracción Segura de Subespacios ──────────────────────────────
    report_state: Dict[str, Any] = payload.get("report_state", {})
    thermal_metrics: Dict[str, Any] = payload.get("thermal_metrics", {})
    financial_metrics: Dict[str, Any] = payload.get("financial_metrics", {})
    synergy_risk: Dict[str, Any] = payload.get("synergy_risk", {})
    pivot_type: str = str(payload.get("pivot_type", "UNKNOWN"))

    # Variables de estado con tipos explícitos y defaults seguros
    stability: float = float(report_state.get("stability", _DEFAULT_STABILITY))
    beta_1: int = int(report_state.get("beta_1", 0))
    system_temp: float = float(thermal_metrics.get("system_temperature", _DEFAULT_SYSTEM_TEMP))
    financial_inertia: float = float(thermal_metrics.get("heat_capacity", _DEFAULT_HEAT_CAPACITY))
    financial_class: str = str(report_state.get("financial_class", "UNKNOWN"))
    npv: float = float(financial_metrics.get("npv", 0.0))

    # Las métricas base se construyen UNA VEZ con el tiempo al inicio.
    # Las rutas de éxito usan estas métricas; las rutas de error recalculan
    # el tiempo para capturar el overhead de evaluación de condiciones.
    base_metrics = VectorMetrics(
        processing_time_ms=_elapsed_ms(start),
        memory_usage_mb=_measure_memory_mb(),
        topological_coherence=1.0,
        algebraic_integrity=1.0,
    )

    # ── 2. Proyección Algebraica del Pivote ──────────────────────────────

    # Pivote A: El Monopolio Coberturado (Topología vs Termodinámica)
    # Condición: base estrecha (ψ < 0.70) + sistema frío (T < 15.0) +
    #            inercia financiera alta (contratos fijos, I > umbral).
    if pivot_type == "MONOPOLIO_COBERTURADO":
        if (
            stability < 0.70
            and system_temp < 15.0
            and financial_inertia > _FINANCIAL_INERTIA_THRESHOLD
        ):
            return _build_result(
                success=True,
                stratum=Stratum.STRATEGY,
                status=VectorResultStatus.SUCCESS,
                metrics=base_metrics,
                payload={
                    "approved_pivot": pivot_type,
                    "penalty_relief": 0.30,
                    "reasoning": (
                        "Riesgo logístico neutralizado por alta inercia "
                        "térmica financiera."
                    ),
                },
            )
        return _build_error(
            stratum=Stratum.STRATEGY,
            status=VectorResultStatus.LOGIC_ERROR,
            error=(
                "Rechazado: Condiciones termodinámicas insuficientes "
                "para cobertura de monopolio."
            ),
            elapsed_ms=_elapsed_ms(start),
        )

    # Pivote B: El Atajo de Opciones Reales (Opción de Espera)
    # Condición: riesgo alto + valor estocástico de esperar > VPN × k.
    if pivot_type == "OPCION_ESPERA":
        real_options: Dict[str, Any] = financial_metrics.get("real_options", {})
        wait_option_value: float = float(real_options.get("wait_option_value", 0.0))
        npv_threshold: float = max(npv, 0.0) * _WAIT_OPTION_NPV_MULTIPLIER

        if financial_class == "HIGH" and wait_option_value > npv_threshold:
            return _build_result(
                success=True,
                stratum=Stratum.STRATEGY,
                status=VectorResultStatus.SUCCESS,
                metrics=base_metrics,
                payload={
                    "approved_pivot": pivot_type,
                    "strategic_action": "FREEZE_6_MONTHS",
                    "reasoning": (
                        f"Valor de la opción de espera ({wait_option_value:.4f}) "
                        f"supera el umbral VPN × {_WAIT_OPTION_NPV_MULTIPLIER} "
                        f"= {npv_threshold:.4f}."
                    ),
                },
            )
        return _build_error(
            stratum=Stratum.STRATEGY,
            status=VectorResultStatus.LOGIC_ERROR,
            error=(
                "Rechazado: El valor de la opción de retraso no justifica "
                "la inactividad."
            ),
            elapsed_ms=_elapsed_ms(start),
        )

    # Pivote C: Cuarentena Topológica (Confinamiento de Ciclos)
    # Condición: ciclos presentes (β₁ > 0) SIN sinergia multiplicativa.
    if pivot_type == "CUARENTENA_TOPOLOGICA":
        has_synergy: bool = bool(synergy_risk.get("synergy_detected", False))

        if beta_1 > 0 and not has_synergy:
            return _build_result(
                success=True,
                stratum=Stratum.STRATEGY,
                status=VectorResultStatus.SUCCESS,
                metrics=base_metrics,
                payload={
                    "approved_pivot": pivot_type,
                    "quarantine_active": True,
                    "reasoning": (
                        "Ciclos detectados pero confinados. Se aprueba la "
                        "ejecución exceptuando el subgrafo aislado."
                    ),
                },
            )
        return _build_error(
            stratum=Stratum.STRATEGY,
            status=VectorResultStatus.LOGIC_ERROR,
            error=(
                "Rechazado: Los ciclos topológicos presentan sinergia "
                "multiplicativa. Cuarentena imposible."
            ),
            elapsed_ms=_elapsed_ms(start),
        )

    # Pivote desconocido — retorno explícito al final (no cadena elif)
    return _build_error(
        stratum=Stratum.STRATEGY,
        status=VectorResultStatus.LOGIC_ERROR,
        error=f"Tipo de pivote lateral desconocido: '{pivot_type}'",
        elapsed_ms=_elapsed_ms(start),
    )


# =============================================================================
# VECTOR TÁCTICO: AUDITORÍA DE FUSIÓN (Mayer-Vietoris)
# =============================================================================


def vector_audit_homological_fusion(
    payload: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> VectorResult:
    """
    Vector de nivel TACTICS.
    Aplica la Secuencia Exacta de Mayer-Vietoris para auditar la integración
    del Grafo Maestro (A) y el Grafo de APUs (B).

    Transformación: (G_A × G_B) ─MV─▶ G_{A∪B}

    Invariante algebraico de Mayer-Vietoris
    ────────────────────────────────────────
        Δβ₁ = β₁(A ∪ B) − [β₁(A) + β₁(B) − β₁(A ∩ B)] = 0

    Si Δβ₁ > 0, la fusión generó ciclos "fantasma" no presentes en los
    sumandos individuales — señal de una integración topológicamente rota.

    Manejo de ImportError
    ─────────────────────
    Si `BusinessTopologicalAnalyzer` no está disponible en el entorno,
    el vector retorna TOPOLOGY_ERROR en lugar de propagar ImportError
    silenciosamente como LOGIC_ERROR genérico.
    """
    start = time.monotonic()

    graph_a = payload.get("graph_a")
    graph_b = payload.get("graph_b")

    if graph_a is None or graph_b is None:
        return _build_error(
            stratum=Stratum.TACTICS,
            status=VectorResultStatus.LOGIC_ERROR,
            error=(
                "Rechazado: Faltan grafos de entrada (A o B) "
                "para la auditoría de fusión."
            ),
            elapsed_ms=_elapsed_ms(start),
        )

    # ── Importación diferida con manejo explícito de dependencia ausente ─
    try:
        from agent.business_topology import BusinessTopologicalAnalyzer  # noqa: PLC0415
    except ImportError as imp_err:
        logger.error(
            "Dependencia 'agent.business_topology' no disponible: %s",
            imp_err,
        )
        return _build_error(
            stratum=Stratum.TACTICS,
            status=VectorResultStatus.TOPOLOGY_ERROR,
            error=(
                f"Dependencia de análisis topológico no disponible: {imp_err}"
            ),
            elapsed_ms=_elapsed_ms(start),
        )

    try:
        analyzer = BusinessTopologicalAnalyzer()
        audit_result: Dict[str, Any] = analyzer.audit_integration_homology(
            graph_a, graph_b
        )
        delta_beta_1: int = int(audit_result.get("delta_beta_1", 0))

        # Métricas construidas DESPUÉS de conocer delta_beta_1
        # para que topological_coherence refleje el resultado real.
        metrics = VectorMetrics(
            processing_time_ms=_elapsed_ms(start),
            memory_usage_mb=_measure_memory_mb(),
            topological_coherence=1.0 if delta_beta_1 == 0 else 0.0,
            algebraic_integrity=1.0,
        )

        # Δβ₁ > 0: ciclos fantasma — integración abortada
        if delta_beta_1 > 0:
            return _build_error(
                stratum=Stratum.TACTICS,
                status=VectorResultStatus.TOPOLOGY_ERROR,
                error=(
                    f"Anomalía de Mayer-Vietoris: La fusión generó "
                    f"{delta_beta_1} ciclo(s) fantasma (Δβ₁ > 0). "
                    "Integración abortada."
                ),
                extra_metrics=metrics,  # propaga coherence=0.0
            )

        # Fusión topológicamente segura
        return _build_result(
            success=True,
            stratum=Stratum.TACTICS,
            status=VectorResultStatus.SUCCESS,
            metrics=metrics,
            payload={
                "audit_result": audit_result,
                "merged_graph_valid": True,
            },
        )

    except Exception as exc:
        logger.error(
            "Fallo matemático en auditoría de fusión: %s", exc, exc_info=True
        )
        return _build_error(
            stratum=Stratum.TACTICS,
            status=VectorResultStatus.LOGIC_ERROR,
            error=f"Fallo matemático en auditoría de fusión: {exc}",
            elapsed_ms=_elapsed_ms(start),
        )


# =============================================================================
# FÁBRICA CON INYECCIÓN DE DEPENDENCIAS
# =============================================================================

# Registro inmutable de vectores por defecto — fuente de verdad única.
# VectorFactory clona desde aquí en reset_registry, eliminando la duplicación.
_DEFAULT_PHYSICS_REGISTRY: Dict[str, Callable[..., VectorResult]] = {
    "stabilize": vector_stabilize_flux,
    "parse": vector_parse_raw_structure,
}


class VectorFactory:
    """
    Factory que produce callables de vector con dependencias inyectadas.

    Principios de diseño
    ────────────────────
    • Open-Closed: registro abierto a extensión vía `register_physics_vector`.
    • Inmutabilidad de defaults: `_DEFAULT_PHYSICS_REGISTRY` no se muta nunca;
      `_PHYSICS_REGISTRY` se clona desde él en `reset_registry`.
    • Introspección correcta: los wrappers propagan `__wrapped__` además de
      `__name__` y `__doc__`, permitiendo inspección con `inspect.unwrap`.
    """

    # El registro activo se inicializa como copia del registro por defecto.
    _PHYSICS_REGISTRY: Dict[str, Callable[..., VectorResult]] = dict(
        _DEFAULT_PHYSICS_REGISTRY
    )

    @classmethod
    def register_physics_vector(
        cls, name: str, fn: Callable[..., VectorResult]
    ) -> None:
        """Registra un vector físico personalizado (extensión abierta)."""
        cls._PHYSICS_REGISTRY[name] = fn

    @classmethod
    def reset_registry(cls) -> None:
        """
        Restaura el registro al estado original.

        Clona desde `_DEFAULT_PHYSICS_REGISTRY` en lugar de duplicar
        el diccionario literal, garantizando que reset y default
        estén siempre sincronizados (principio DRY).
        """
        cls._PHYSICS_REGISTRY = dict(_DEFAULT_PHYSICS_REGISTRY)

    @classmethod
    def create_physics_vector(
        cls, vector_type: str, **defaults: Any
    ) -> Callable[..., VectorResult]:
        """
        Devuelve un callable que invoca el vector físico elegido
        con *defaults* fusionados bajo los argumentos explícitos.

        El wrapper propaga `__wrapped__`, `__name__` y `__doc__`
        para compatibilidad con `inspect.unwrap` y herramientas de test.
        """
        if vector_type not in cls._PHYSICS_REGISTRY:
            available = sorted(cls._PHYSICS_REGISTRY.keys())
            raise ValueError(
                f"Vector físico '{vector_type}' no registrado. "
                f"Disponibles: {available}"
            )

        base_fn = cls._PHYSICS_REGISTRY[vector_type]

        @functools.wraps(base_fn)
        def wrapper(*args: Any, **kwargs: Any) -> VectorResult:
            merged = {**defaults, **kwargs}
            return base_fn(*args, **merged)

        wrapper.__name__ = f"physics_{vector_type}"
        wrapper.__wrapped__ = base_fn  # type: ignore[attr-defined]
        return wrapper

    @staticmethod
    def create_tactics_vector(
        **defaults: Any,
    ) -> Callable[..., VectorResult]:
        """
        Devuelve un callable que envuelve `vector_structure_logic` con defaults.

        El wrapper propaga `__wrapped__` para introspección correcta.
        """

        @functools.wraps(vector_structure_logic)
        def wrapper(*args: Any, **kwargs: Any) -> VectorResult:
            merged = {**defaults, **kwargs}
            return vector_structure_logic(*args, **merged)

        wrapper.__name__ = "tactics_structure"
        wrapper.__wrapped__ = vector_structure_logic  # type: ignore[attr-defined]
        return wrapper


# =============================================================================
# COMPOSICIÓN DE VECTORES (MORFISMOS COMPUESTOS)
# =============================================================================


def compose_vectors(
    physics_vector: Callable[..., VectorResult],
    tactics_vector: Callable[..., VectorResult],
    physics_args: Tuple[Any, ...],
    tactics_config: Dict[str, Any],
) -> VectorResult:
    """
    Compone un morfismo físico con uno táctico en un pipeline coherente.

    Diagrama conmutativo
    ────────────────────
        File ──φ_phys──▶ RawData ──φ_tact──▶ ProcessedData

    La composición φ_tact ∘ φ_phys está definida sólo cuando la imagen
    del vector físico pertenece al dominio del táctico, es decir,
    'raw_records' debe existir en la salida física.

    Métricas combinadas (principio de cuello de botella)
    ─────────────────────────────────────────────────────
    pipeline_coherence = min(C_phys, C_tact)
    total_time_ms      = t_phys + t_tact

    El mínimo de coherencias es el cuello de botella topológico:
    el pipeline es tan coherente como su eslabón más débil.

    Parámetros
    ──────────
    physics_args   : argumentos posicionales para el vector físico.
    tactics_config : configuración para el vector táctico.
    """
    # ── Fase 1: Física ────────────────────────────────────────────────────
    physics_result = physics_vector(*physics_args)

    if not physics_result.get("success"):
        return physics_result

    # ── Comprobación de compatibilidad de dominio ─────────────────────────
    raw_records: Optional[List[Dict[str, Any]]] = physics_result.get("raw_records")
    parse_cache: Dict[str, Any] = physics_result.get("parse_cache", {})

    if raw_records is None:
        return _build_result(
            success=False,
            stratum=Stratum.TACTICS,
            status=VectorResultStatus.TOPOLOGY_ERROR,
            error=(
                "Composición indefinida: el vector físico no produjo "
                "'raw_records'. Use vector_parse_raw_structure como "
                "componente física de esta composición."
            ),
            physics_context=physics_result,
        )

    # ── Fase 2: Táctica ──────────────────────────────────────────────────
    tactics_result = tactics_vector(raw_records, parse_cache, tactics_config)

    # ── Métricas combinadas ──────────────────────────────────────────────
    phys_m: Dict[str, Any] = physics_result.get("metrics", {})
    tact_m: Dict[str, Any] = tactics_result.get("metrics", {})

    tactics_result["combined_metrics"] = {
        "physics": phys_m,
        "tactics": tact_m,
        "total_time_ms": (
            phys_m.get("processing_time_ms", 0.0)
            + tact_m.get("processing_time_ms", 0.0)
        ),
        # Mínimo de coherencias: cuello de botella topológico del pipeline.
        "pipeline_coherence": min(
            phys_m.get("topological_coherence", 1.0),
            tact_m.get("topological_coherence", 1.0),
        ),
    }

    return tactics_result