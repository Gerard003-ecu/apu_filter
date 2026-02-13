"""
Módulo: MIC Vectors (Adaptadores de Capacidad y Morfismos Algebraicos)
======================================================================

Este módulo implementa la capa de adaptación ("Glue Code") como un conjunto de 
**Morfismos Algebraicos** que proyectan las intenciones del Agente sobre los 
motores físicos y lógicos del sistema (FluxCondenser, APUProcessor).

Fundamentos Matemáticos y Arquitectura:
---------------------------------------

1. Teoría de Categorías (Morfismos):
   Cada función vectorial (ej. `vector_stabilize_flux`) se modela como un morfismo $\phi$:
   $$ \phi: \text{ConfigSpace} \times \text{Context} \to \text{ResultSpace} $$
   Transforman un payload de configuración (diccionario) en un resultado tipado (`VectorResult`),
   preservando la estructura de la información a través de los estratos DIKW.

2. Coherencia Topológica ($C$):
   Se calcula una métrica escalar invariante para evaluar la calidad de la transformación,
   basada en la Estabilidad ($S$), Robustez ($R$) y Entropía de Shannon ($H$):
   $$ C = \frac{S \cdot R}{1 + H} \in [1] $$
   Esto permite al sistema rechazar resultados "técnicamente exitosos" pero 
   "topológicamente incoherentes" (alto ruido o baja estabilidad) [Fuente 990].

3. Isomorfismo Dimensional (Enlace Física $\to$ Táctica):
   Implementa guardas algebraicas (`validate_dimensional_isomorphism`) que aseguran 
   que la dimensión del espacio de salida del vector físico (columnas detectadas en el parser)
   sea isomorfa a la dimensión esperada por el vector táctico. Esto previene errores de 
   alineación antes de intentar cualquier cálculo de costos [Fuente 1000].

4. Inmutabilidad Algebraica:
   Las métricas (`VectorMetrics`) se definen como `dataclasses` congeladas (frozen), 
   tratándolas como valores puros que no pueden ser alterados por efectos secundarios, 
   garantizando la integridad forense de la telemetría [Fuente 994].

"""

import logging
import os
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
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

    Inmutabilidad (frozen) ⇒ objeto algebraico puro:
    una vez construido, su identidad observacional es fija.
    """

    processing_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    topological_coherence: float = 1.0
    algebraic_integrity: float = 1.0


# =============================================================================
# HELPERS INTERNOS
# =============================================================================
def _measure_memory_mb() -> float:
    """Consumo de memoria residente (MB).  Devuelve 0 si psutil no existe."""
    if _HAS_PSUTIL:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    return 0.0


def _elapsed_ms(start: float) -> float:
    """Milisegundos transcurridos desde *start*."""
    return (time.time() - start) * 1000.0


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
    """
    result: VectorResult = {
        "success": success,
        "stratum": stratum,
        "status": status.value,
        "metrics": asdict(metrics) if metrics else asdict(VectorMetrics()),
    }
    if error is not None:
        result["error"] = error
    result.update(payload)
    return result


def _build_error(
    stratum: Stratum,
    status: VectorResultStatus,
    error: str,
    elapsed_ms: float = 0.0,
) -> VectorResult:
    """Atajo para resultados de fallo con métricas mínimas."""
    return _build_result(
        success=False,
        stratum=stratum,
        status=status,
        error=error,
        metrics=VectorMetrics(processing_time_ms=elapsed_ms),
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
    """
    if not os.path.exists(file_path):
        return False, f"Topología rota: archivo '{file_path}' no existe"

    missing = [k for k in required_keys if k not in config]
    if missing:
        return False, f"Configuración incompleta: faltan {missing}"

    constraints = config.get("dimension_constraints")
    if isinstance(constraints, dict):
        non_numeric = [
            k for k, v in constraints.items() if not isinstance(v, (int, float))
        ]
        if non_numeric:
            return False, (
                f"Dimensionalidad no numérica en claves: {non_numeric}"
            )

    return True, None


def validate_homological_constraints(constraints: Dict[str, Any]) -> bool:
    """
    Valida que las restricciones formen un complejo de cadenas válido.

    Requisitos
    ──────────
    - max_dimension : int ≥ 0   (dimensión máxima del complejo)
    - allow_holes   : bool      (permitir β₁ > 0)
    - connectivity  : numérico  (umbral de componentes conexas)
    """
    required = {"max_dimension", "allow_holes", "connectivity"}
    if not required.issubset(constraints.keys()):
        return False

    md = constraints["max_dimension"]
    if not isinstance(md, int) or md < 0:
        return False

    if not isinstance(constraints["allow_holes"], bool):
        return False

    if not isinstance(constraints["connectivity"], (int, float)):
        return False

    return True


# =============================================================================
# FUNCIONES AUXILIARES DE TOPOLOGÍA ALGEBRAICA
# =============================================================================
def calculate_topological_coherence(physics_report: Dict[str, Any]) -> float:
    """
    Coherencia topológica como cociente normalizado.

    Fórmula
    ───────
        C = S · R / (1 + H)

    donde S = stability ∈ [0,1],  R = resonance ∈ [0,1],  H = entropy ≥ 0.

    Propiedades
    ───────────
    • C ∈ [0, 1]  (acotación natural, sin necesidad de clamp negativo).
    • ∂C/∂S > 0,  ∂C/∂R > 0           (monótona creciente en S, R).
    • ∂C/∂H < 0                        (monótona decreciente en H).
    • C = 0  ⟺  S = 0  ∨  R = 0       (cero absorbente).

    Comparación con la versión anterior  max(0, (S − H) · R):
    ─ Aquella mezcla magnitudes incompatibles (S ∈ [0,1] vs H ≥ 0).
    ─ Produce clamp discontinuo en 0.
    ─ La nueva degrada suavemente con entropía creciente.
    """
    if not physics_report:
        return 0.0

    stability = max(0.0, min(1.0, float(physics_report.get("stability_index", 0))))
    entropy = max(0.0, float(physics_report.get("entropy", 0)))
    resonance = max(0.0, min(1.0, float(physics_report.get("resonance_factor", 0))))

    return stability * resonance / (1.0 + entropy)


def calculate_betti_numbers(
    raw_records: List[Dict[str, Any]],
    cache: Dict[str, Any],
) -> Dict[str, int]:
    """
    Números de Betti proxy del complejo simplicial inducido por los registros.

    β₀ = componentes conexas  (tipos de registro distintos).
    β₁ = 1-ciclos             (ciclos de dependencia en el grafo del cache).
    β₂ = 2-cavidades          (estructuras anidadas vacías de contenido).

    Incluye la característica de Euler:  χ = β₀ − β₁ + β₂.
    """
    if not raw_records:
        return {"beta_0": 0, "beta_1": 0, "beta_2": 0, "euler": 0}

    # β₀: componentes conexas
    types: Set[str] = {r.get("record_type", "unknown") for r in raw_records}
    beta_0 = len(types)

    # β₁: ciclos del grafo de dependencias
    cycles = cache.get("dependency_cycles", [])
    beta_1 = len(cycles) if isinstance(cycles, list) else 0

    # β₂: cavidades (nodos anidados huecos)
    beta_2 = sum(
        1 for r in raw_records if r.get("nested") and not r.get("content")
    )

    euler = beta_0 - beta_1 + beta_2

    return {
        "beta_0": beta_0,
        "beta_1": beta_1,
        "beta_2": beta_2,
        "euler": euler,
    }


def calculate_algebraic_integrity(betti_numbers: Dict[str, int]) -> float:
    """
    Integridad algebraica derivada de los números de Betti superiores.

    I = 1 / (1 + Σ β_i ,  i > 0)

    • I = 1   ⟹  no hay ciclos ni cavidades (variedad acíclica).
    • I → 0   cuando los defectos topológicos se acumulan.

    Robustez: itera sobre claves reales del diccionario;
    no asume que las claves sean enteros consecutivos.
    """
    if not betti_numbers:
        return 1.0

    # Claves superiores: beta_1, beta_2, …
    higher_sum = sum(
        count
        for key, count in betti_numbers.items()
        if key not in ("beta_0", "euler") and isinstance(count, (int, float))
    )
    return 1.0 / (1.0 + higher_sum)


def calculate_dimensionality(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    Rango del espacio de atributos por tipo de registro.

        dim(V_t) = | ⋃_{r : type(r)=t}  keys(r) |

    A diferencia de sumar el conteo de claves por registro (lo cual crece
    linealmente con el número de registros), esta función calcula la
    **cardinalidad de la unión** de atributos — el rango real del espacio
    columnar, análogo a la dimensión de un espacio vectorial.
    """
    if not records:
        return {}

    attribute_spaces: Dict[str, Set[str]] = {}
    for record in records:
        rtype = record.get("record_type", "default")
        if rtype not in attribute_spaces:
            attribute_spaces[rtype] = set()
        attribute_spaces[rtype].update(record.keys())

    return {rtype: len(attrs) for rtype, attrs in attribute_spaces.items()}


def validate_dimensional_isomorphism(
    expected: Dict[str, int],
    actual: Dict[str, int],
    tolerance: float = 0.10,
) -> bool:
    """
    Dos espacios de atributos son isomorfos si comparten los mismos
    tipos y sus rangos coinciden dentro de una tolerancia ε.

        ∀ t ∈ Types:  |dim_exp(t) − dim_act(t)| / max(dim_exp(t), 1) ≤ ε

    Retorna True vacuamente si alguno de los espacios está vacío
    (no hay contra qué comparar).
    """
    if not expected or not actual:
        return True  # verdad vacua

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
    """
    start = time.time()

    is_valid, err = validate_topological_preconditions(
        file_path, config, _REQUIRED_STABILIZE_KEYS
    )
    if not is_valid:
        return _build_error(
            Stratum.PHYSICS,
            VectorResultStatus.TOPOLOGY_ERROR,
            err,  # type: ignore[arg-type]
            _elapsed_ms(start),
        )

    try:
        condenser_conf = CondenserConfig(
            system_capacitance=float(config["system_capacitance"]),
            system_inductance=float(config["system_inductance"]),
            base_resistance=float(config["base_resistance"]),
            resonance_threshold=float(
                config.get("resonance_threshold", 0.85)
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
            Stratum.PHYSICS,
            VectorResultStatus.PHYSICS_ERROR,
            str(exc),
            _elapsed_ms(start),
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

    Homología:
      H₀ → componentes conexas (registros crudos).
      H₁ → ciclos (dependencias circulares).
      H₂ → cavidades (estructuras anidadas vacías).
    """
    start = time.time()

    # ── Guardia: existencia del punto base ─────────────────────────────────
    if not os.path.exists(file_path):
        return _build_error(
            Stratum.PHYSICS,
            VectorResultStatus.TOPOLOGY_ERROR,
            f"Archivo no existe: '{file_path}'",
            _elapsed_ms(start),
        )

    # ── Guardia: restricciones homológicas ─────────────────────────────────
    if topological_constraints is not None and not validate_homological_constraints(
        topological_constraints
    ):
        return _build_error(
            Stratum.PHYSICS,
            VectorResultStatus.TOPOLOGY_ERROR,
            "Restricciones homológicas inválidas",
            _elapsed_ms(start),
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

        # Acceso seguro a validation_stats
        validation_stats: Dict[str, Any] = {}
        if hasattr(parser, "validation_stats"):
            vs = parser.validation_stats
            validation_stats = (
                vs.__dict__ if hasattr(vs, "__dict__") else dict(vs)
            )

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
            Stratum.PHYSICS,
            VectorResultStatus.PHYSICS_ERROR,
            str(exc),
            _elapsed_ms(start),
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

    Álgebra abstracta:
      G   = grupo de transformaciones (APUProcessor).
      φ   = homomorfismo  G → Aut(Data).
      ker(φ) = núcleo de validación (errores capturados).
      im(φ)  = imagen procesada limpia.
    """
    start = time.time()

    # ── Guardia: conjunto no vacío ─────────────────────────────────────────
    if not raw_records:
        return _build_error(
            Stratum.TACTICS,
            VectorResultStatus.LOGIC_ERROR,
            "Conjunto vacío: no hay registros para transformar",
            _elapsed_ms(start),
        )

    # ── Verificación de isomorfismo dimensional ────────────────────────────
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

        # ── Extracción segura del núcleo algebraico ────────────────────────
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
            Stratum.TACTICS,
            VectorResultStatus.LOGIC_ERROR,
            str(exc),
            _elapsed_ms(start),
        )


# =============================================================================
# FÁBRICA CON INYECCIÓN DE DEPENDENCIAS
# =============================================================================
class VectorFactory:
    """
    Factory que produce callables de vector con dependencias inyectadas.

    Soporta registro abierto (open-closed principle) y reset para tests.
    """

    _PHYSICS_REGISTRY: Dict[str, Callable[..., VectorResult]] = {
        "stabilize": vector_stabilize_flux,
        "parse": vector_parse_raw_structure,
    }

    @classmethod
    def register_physics_vector(
        cls, name: str, fn: Callable[..., VectorResult]
    ) -> None:
        """Registra un vector físico personalizado (extensión abierta)."""
        cls._PHYSICS_REGISTRY[name] = fn

    @classmethod
    def reset_registry(cls) -> None:
        """Restaura el registro al estado original (útil en tests)."""
        cls._PHYSICS_REGISTRY = {
            "stabilize": vector_stabilize_flux,
            "parse": vector_parse_raw_structure,
        }

    @classmethod
    def create_physics_vector(
        cls, vector_type: str, **defaults: Any
    ) -> Callable[..., VectorResult]:
        """
        Devuelve un callable que invoca el vector físico elegido
        con *defaults* fusionados bajo los argumentos explícitos.
        """
        if vector_type not in cls._PHYSICS_REGISTRY:
            available = sorted(cls._PHYSICS_REGISTRY.keys())
            raise ValueError(
                f"Vector físico '{vector_type}' no registrado. "
                f"Disponibles: {available}"
            )

        base_fn = cls._PHYSICS_REGISTRY[vector_type]

        def wrapper(*args: Any, **kwargs: Any) -> VectorResult:
            merged = {**defaults, **kwargs}
            return base_fn(*args, **merged)

        wrapper.__name__ = f"physics_{vector_type}"
        wrapper.__doc__ = base_fn.__doc__
        return wrapper

    @staticmethod
    def create_tactics_vector(
        **defaults: Any,
    ) -> Callable[..., VectorResult]:
        """Devuelve un callable que envuelve vector_structure_logic con defaults."""

        def wrapper(*args: Any, **kwargs: Any) -> VectorResult:
            merged = {**defaults, **kwargs}
            return vector_structure_logic(*args, **merged)

        wrapper.__name__ = "tactics_structure"
        wrapper.__doc__ = vector_structure_logic.__doc__
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

    Diagrama conmutativo:

        File ──φ_phys──▶ RawData ──φ_tact──▶ ProcessedData

    La composición  φ_tact ∘ φ_phys  está definida sólo cuando la
    imagen del vector físico pertenece al dominio del táctico
    (es decir, raw_records debe existir en la salida física).

    Parámetros
    ──────────
    physics_args   : argumentos posicionales para el vector físico.
    tactics_config : diccionario de configuración para el vector táctico.
    """
    # ── Fase 1: Física ─────────────────────────────────────────────────────
    physics_result = physics_vector(*physics_args)

    if not physics_result.get("success"):
        return physics_result

    # ── Comprobación de compatibilidad de dominio ──────────────────────────
    raw_records = physics_result.get("raw_records")
    parse_cache = physics_result.get("parse_cache", {})

    if raw_records is None:
        return _build_result(
            success=False,
            stratum=Stratum.TACTICS,
            status=VectorResultStatus.TOPOLOGY_ERROR,
            error=(
                "Composición indefinida: el vector físico no produjo "
                "'raw_records'.  Use vector_parse_raw_structure como "
                "componente física de esta composición."
            ),
            physics_context=physics_result,
        )

    # ── Fase 2: Táctica ────────────────────────────────────────────────────
    tactics_result = tactics_vector(raw_records, parse_cache, tactics_config)

    # ── Métricas combinadas (principio de cuello de botella) ───────────────
    phys_m = physics_result.get("metrics", {})
    tact_m = tactics_result.get("metrics", {})

    tactics_result["combined_metrics"] = {
        "physics": phys_m,
        "tactics": tact_m,
        "total_time_ms": (
            phys_m.get("processing_time_ms", 0.0)
            + tact_m.get("processing_time_ms", 0.0)
        ),
        "pipeline_coherence": min(
            phys_m.get("topological_coherence", 1.0),
            tact_m.get("topological_coherence", 1.0),
        ),
    }

    return tactics_result