"""
Módulo: Lithological Manifold
Ubicación: app/physics/lithological_manifold.py

Naturaleza Ciber-Física y Topológica:
Este módulo abandona la heurística empírica de la mecánica de suelos tradicional para modelar
la litología del terreno como un Tensor de Impedancia Geomecánica Compleja (Z_{geo}). Se
acopla ortogonalmente al Estrato de Física (V_PHYSICS) actuando como la Condición de Frontera
de Dirichlet absoluta. Fija los nodos de anclaje (tierra) de la matriz Laplaciana del proyecto;
sin esta métrica, el motor dinámico asumiría un espacio euclidiano isotrópico de rigidez infinita,
induciendo una violación catastrófica de la conservación de energía (P_diss ≥ 0).

Fundamentación Axiomática y Modelo Matemático:
1. Espacio de Estado Litológico: Sea S = (σ_USCS, LL, PI, Vs, e₀, sat, ρ) ∈ ℝ⁷ el vector covariante
de estado del suelo.
2. Difeomorfismo Físico (El Operador Φ): El funtor Φ : S → (G_max, I_sw, I_y, I_liq, {C_k}) ejecuta
un mapeo no lineal desde las primitivas geomecánicas hacia un conjunto de magnitudes derivadas acotadas
estrictamente en el intervalo unitario [3]. 
3. Cuantización de Defectos (Cartuchos TOON): Las patologías litológicas no se propagan como cadenas
de texto de alta entropía. El operador colapsa el estado en un conjunto finito de cuasipartículas
informacionales C_k con k ∈ {SwellingPlasmon, YieldingPhonon, LiquefactionSoliton}. Estas partículas
inyectan capacitancia parásita, arrastre viscoso inercial, o aniquilan la conectividad topológica (β₀ → ∞) del grafo logístico.

Invariantes Computacionales Garantizados:
- Finitud Estricta y Consistencia Dimensional: Todo resultado intermedio es verificado como finito (¬NaN, ¬±Inf).
El módulo de rigidez dinámica se aproxima mediante la inercia de la onda de corte G_max = ρ · Vs², preservando
la condición estricta de no-negatividad.
- Clausura Transitiva Categórica: La detección de una singularidad litológica extrema (ej. suelos altamente orgánicos
como Turba 'PT') dispara un "Fast-Fail", aniquilando la invertibilidad de la matriz Laplaciana e impidiendo la disipación
inútil de exergía computacional en la FPU.

Referencias Fundacionales:
- Terzaghi, K., Peck, R.B., Mesri, G. (1996). Soil Mechanics in Engineering Practice. 3ª ed [4].
- Seed, H.B. & Idriss, I.M. (1971). Simplified Procedure for Evaluating Soil Liquefaction Potential [4].
- Skempton, A.W. (1944). Notes on the Compressibility of Clays [4].
"""

from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    Final,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from app.core.mic_algebra import Morphism, CategoricalState
from app.core.schemas import Stratum
from app.core.immune_system.metric_tensors import MetricTensorFactory

logger = logging.getLogger("MIC.Physics.LithologicalManifold")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    §1  CONSTANTES FÍSICAS Y UMBRALES                       ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class GeomechanicalConstants:
    """
    Constantes, umbrales y parámetros del modelo geomecánico.

    Observación
    -----------
    Los umbrales aquí definidos son de carácter operacional interno.
    Deben calibrarse si el sistema pasa a régimen de uso normativo.

    Los rangos admisibles se basan en valores extremos documentados
    en la literatura geotécnica (Terzaghi et al., 1996).
    """

    # --- Constantes fundamentales ---
    EPSILON_MACH: Final[float] = 1e-12
    GRAVITY_M_S2: Final[float] = 9.80665
    WATER_DENSITY_KG_M3: Final[float] = 1000.0

    # --- Densidad por defecto ---
    DEFAULT_SOIL_DENSITY_KG_M3: Final[float] = 1800.0

    # --- Rangos admisibles de validación ---
    # Límite líquido: [0, 500] % — incluye bentonitas extremas
    LIQUID_LIMIT_MAX: Final[float] = 500.0
    # Índice de plasticidad: [0, 350] % — bentonitas sódicas
    PLASTICITY_INDEX_MAX: Final[float] = 350.0
    # Velocidad de onda cortante: (0, 5000] m/s — roca competente
    SHEAR_WAVE_VELOCITY_MAX: Final[float] = 5000.0
    # Relación de vacíos: [0, 15] — arcillas mexicanas y turbas
    VOID_RATIO_MAX: Final[float] = 15.0
    # Densidad: [500, 3500] kg/m³ — suelos orgánicos a minerales densos
    BULK_DENSITY_MIN: Final[float] = 500.0
    BULK_DENSITY_MAX: Final[float] = 3500.0

    # --- Umbrales operativos de diagnóstico ---
    CRITICAL_LIQUEFACTION_VS_M_S: Final[float] = 150.0
    SOFT_SOIL_VS_M_S: Final[float] = 200.0
    HIGH_VOID_RATIO_THRESHOLD: Final[float] = 0.8
    SWELLING_LIQUID_LIMIT_THRESHOLD: Final[float] = 50.0
    SWELLING_PI_THRESHOLD: Final[float] = 20.0
    LOW_DYNAMIC_RIGIDITY_PA: Final[float] = 25.0e6

    # --- Parámetros de funciones de índice ---
    # Factor de escala sigmoidal para expansividad
    SWELLING_SIGMOID_MIDPOINT: Final[float] = 20.0   # LL·PI/100 donde I = 0.5
    SWELLING_SIGMOID_STEEPNESS: Final[float] = 0.15   # Pendiente logística

    # Factor de Skempton para Cc proxy: Cc ≈ 0.009·(LL - 10)
    SKEMPTON_CC_FACTOR: Final[float] = 0.009
    SKEMPTON_CC_OFFSET: Final[float] = 10.0

    # Escala referencia para normalizar índice de cedencia
    YIELDING_REFERENCE_SCALE: Final[float] = 0.01  # [s/m], para adimensionalizar

    # Exponente no lineal para susceptibilidad a licuación
    LIQUEFACTION_NONLINEAR_EXPONENT: Final[float] = 2.0


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    §2  CLASIFICACIÓN USCS                                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

USCS_COARSE_GROUPS: FrozenSet[str] = frozenset({
    "GW", "GP", "GM", "GC",
    "SW", "SP", "SM", "SC",
    # Grupos duales (ASTM D2487)
    "GW-GM", "GW-GC", "GP-GM", "GP-GC",
    "SW-SM", "SW-SC", "SP-SM", "SP-SC",
})

USCS_FINE_GROUPS: FrozenSet[str] = frozenset({
    "ML", "CL", "OL",
    "MH", "CH", "OH",
    # Grupos duales
    "CL-ML",
})

USCS_ORGANIC_EXTREME_GROUPS: FrozenSet[str] = frozenset({"PT"})

# Unión de todos los grupos válidos para validación
USCS_ALL_VALID_GROUPS: FrozenSet[str] = (
    USCS_COARSE_GROUPS | USCS_FINE_GROUPS | USCS_ORGANIC_EXTREME_GROUPS
)


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    §3  EXCEPCIONES                                         ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class LithologicalManifoldError(Exception):
    """Excepción base del módulo litológico."""


class LithologicalInputError(LithologicalManifoldError):
    """Error de validación o parseo de entrada."""


class LithologicalSingularityError(LithologicalManifoldError):
    """
    Singularidad litológica extrema.

    Caso canónico
    -------------
    Turba / peat (PT): degrada drásticamente capacidad portante,
    rigidez y estabilidad de la frontera de anclaje. Corresponde a
    un punto singular en la variedad litológica donde los indicadores
    convencionales pierden significado.
    """


class LithologicalNumericalError(LithologicalManifoldError):
    """Resultado numérico no finito en cálculo intermedio."""


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    §4  ENUMERACIONES TIPADAS                               ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@enum.unique
class DiagnosticRule(str, enum.Enum):
    """
    Enumeración de reglas diagnósticas activables.

    El uso de enumeración tipada elimina el riesgo de errores
    tipográficos en cadenas literales y permite verificación
    estática por el sistema de tipos.
    """
    SWELLING_POTENTIAL_DETECTED = "swelling_potential_detected"
    YIELDING_SUSCEPTIBILITY_DETECTED = "yielding_susceptibility_detected"
    LIQUEFACTION_SUSCEPTIBILITY_DETECTED = "liquefaction_susceptibility_detected"
    LOW_DYNAMIC_RIGIDITY_REGIME = "low_dynamic_rigidity_regime"


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    §5  CARTUCHOS DIAGNÓSTICOS                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True, slots=True)
class SwellingPlasmonCartridge:
    """
    Excitación asociada a suelo potencialmente expansivo.

    Atributos
    ---------
    liquid_limit : float
        Límite líquido LL [%].
    plasticity_index : float
        Índice de plasticidad PI [%].
    parasitic_capacitance : float
        Índice proxy normalizado ∈ [0, 1] de almacenamiento potencial
        volumétrico.  Construido mediante función sigmoidal sobre el
        producto LL·PI.
    topological_spin : str
        Etiqueta semántica interna de clasificación del cartucho.
    """
    liquid_limit: float
    plasticity_index: float
    parasitic_capacitance: float
    topological_spin: str = "potential_accumulator"

    def __post_init__(self) -> None:
        _assert_unit_interval(
            self.parasitic_capacitance, "parasitic_capacitance"
        )


@dataclass(frozen=True, slots=True)
class YieldingPhononCartridge:
    """
    Excitación asociada a compresibilidad/blandura y susceptibilidad
    a cedencia.

    Atributos
    ---------
    compression_index : float
        Índice de compresión proxy Cc (correlación de Skempton).
    void_ratio : float
        Relación de vacíos e₀.
    viscous_drag : float
        Índice proxy normalizado ∈ [0, 1] de arrastre/retraso de
        asentamiento.  Adimensional por normalización explícita.
    topological_spin : str
        Etiqueta semántica interna.
    """
    compression_index: float
    void_ratio: float
    viscous_drag: float
    topological_spin: str = "inertial_drag"

    def __post_init__(self) -> None:
        _assert_unit_interval(self.viscous_drag, "viscous_drag")


@dataclass(frozen=True, slots=True)
class LiquefactionSolitonCartridge:
    """
    Excitación asociada a susceptibilidad de licuación.

    Atributos
    ---------
    shear_wave_velocity : float
        Velocidad de onda cortante Vs [m/s].
    susceptibility_index : float
        Índice proxy normalizado ∈ [0, 1] de susceptibilidad a
        licuación.  Nombre corregido respecto al original para evitar
        confusión con CSR normativo (Seed & Idriss, 1971).
    topological_spin : str
        Etiqueta semántica interna.
    """
    shear_wave_velocity: float
    susceptibility_index: float
    topological_spin: str = "connectivity_annihilator"

    def __post_init__(self) -> None:
        _assert_unit_interval(
            self.susceptibility_index, "susceptibility_index"
        )


LithologyCartridge = Union[
    SwellingPlasmonCartridge,
    YieldingPhononCartridge,
    LiquefactionSolitonCartridge,
]


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              §6  TENSOR DE ESTADO Y REPORTE DIAGNÓSTICO                    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

@dataclass(frozen=True, slots=True)
class SoilTensor:
    """
    Tensor de entrada con propiedades litológicas mínimas.

    Convenciones
    ------------
    - liquid_limit y plasticity_index en porcentaje [%].
    - shear_wave_velocity en m/s.
    - void_ratio adimensional.
    - bulk_density_kg_m3 en kg/m³.

    Invariantes (verificados en validate)
    ----------
    1.  LL ≥ 0,  PI ≥ 0.
    2.  PI ≤ LL  (restricción de Casagrande: la línea A del diagrama
        de plasticidad impone PI < LL en todo caso).
    3.  Vs > 0.
    4.  e₀ ≥ 0.
    5.  ρ > 0.
    6.  Clasificación USCS reconocida.
    7.  Todos los valores numéricos son finitos.
    """
    uscs_classification: str
    liquid_limit: float
    plasticity_index: float
    shear_wave_velocity: float
    void_ratio: float
    is_saturated: bool
    bulk_density_kg_m3: float = GeomechanicalConstants.DEFAULT_SOIL_DENSITY_KG_M3

    def validate(self) -> None:
        """
        Valida consistencia básica y rangos admisibles.

        Lanza
        -----
        LithologicalInputError
            Si se detecta dato físicamente inválido o fuera de rango.
        """
        C = GeomechanicalConstants

        # --- Clasificación ---
        uscs_norm = _normalize_uscs(self.uscs_classification)
        if not uscs_norm:
            raise LithologicalInputError(
                "La clasificación USCS no puede ser vacía."
            )
        if uscs_norm not in USCS_ALL_VALID_GROUPS:
            raise LithologicalInputError(
                f"Clasificación USCS no reconocida: '{uscs_norm}'. "
                f"Grupos válidos: {sorted(USCS_ALL_VALID_GROUPS)}"
            )

        # --- Finitud ---
        numeric_fields = {
            "liquid_limit": self.liquid_limit,
            "plasticity_index": self.plasticity_index,
            "shear_wave_velocity": self.shear_wave_velocity,
            "void_ratio": self.void_ratio,
            "bulk_density_kg_m3": self.bulk_density_kg_m3,
        }
        for name, value in numeric_fields.items():
            if not math.isfinite(value):
                raise LithologicalInputError(
                    f"El campo '{name}' debe ser finito; recibido: {value}"
                )

        # --- Límite líquido ---
        if self.liquid_limit < 0.0:
            raise LithologicalInputError(
                "El límite líquido no puede ser negativo."
            )
        if self.liquid_limit > C.LIQUID_LIMIT_MAX:
            raise LithologicalInputError(
                f"Límite líquido {self.liquid_limit}% excede el máximo "
                f"admisible {C.LIQUID_LIMIT_MAX}%."
            )

        # --- Índice de plasticidad ---
        if self.plasticity_index < 0.0:
            raise LithologicalInputError(
                "El índice de plasticidad no puede ser negativo."
            )
        if self.plasticity_index > C.PLASTICITY_INDEX_MAX:
            raise LithologicalInputError(
                f"PI {self.plasticity_index}% excede el máximo admisible "
                f"{C.PLASTICITY_INDEX_MAX}%."
            )

        # --- Restricción de Casagrande: PI ≤ LL ---
        if self.plasticity_index > self.liquid_limit + C.EPSILON_MACH:
            raise LithologicalInputError(
                f"El índice de plasticidad ({self.plasticity_index}%) no "
                f"puede exceder el límite líquido ({self.liquid_limit}%). "
                f"Restricción de consistencia de Casagrande violada."
            )

        # --- Velocidad de onda cortante ---
        if self.shear_wave_velocity <= 0.0:
            raise LithologicalInputError(
                "La velocidad de onda cortante Vs debe ser estrictamente "
                "positiva."
            )
        if self.shear_wave_velocity > C.SHEAR_WAVE_VELOCITY_MAX:
            raise LithologicalInputError(
                f"Vs {self.shear_wave_velocity} m/s excede el máximo "
                f"admisible {C.SHEAR_WAVE_VELOCITY_MAX} m/s."
            )

        # --- Relación de vacíos ---
        if self.void_ratio < 0.0:
            raise LithologicalInputError(
                "La relación de vacíos no puede ser negativa."
            )
        if self.void_ratio > C.VOID_RATIO_MAX:
            raise LithologicalInputError(
                f"Void ratio {self.void_ratio} excede el máximo admisible "
                f"{C.VOID_RATIO_MAX}."
            )

        # --- Densidad ---
        if self.bulk_density_kg_m3 <= 0.0:
            raise LithologicalInputError(
                "La densidad del suelo debe ser positiva."
            )
        if not (C.BULK_DENSITY_MIN
                <= self.bulk_density_kg_m3
                <= C.BULK_DENSITY_MAX):
            raise LithologicalInputError(
                f"Densidad {self.bulk_density_kg_m3} kg/m³ fuera del rango "
                f"admisible [{C.BULK_DENSITY_MIN}, {C.BULK_DENSITY_MAX}]."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialización canónica del tensor de estado."""
        return {
            "uscs_classification": self.uscs_classification,
            "liquid_limit": self.liquid_limit,
            "plasticity_index": self.plasticity_index,
            "shear_wave_velocity": self.shear_wave_velocity,
            "void_ratio": self.void_ratio,
            "is_saturated": self.is_saturated,
            "bulk_density_kg_m3": self.bulk_density_kg_m3,
        }


@dataclass(frozen=True, slots=True)
class LithologyDiagnosticReport:
    """
    Reporte estructurado de diagnóstico litológico.

    Todos los índices de susceptibilidad están normalizados a [0, 1].
    """
    soil_tensor: SoilTensor
    dynamic_rigidity_modulus_pa: float
    swelling_potential_index: float
    yielding_susceptibility_index: float
    liquefaction_susceptibility_index: float
    emitted_cartridges: Tuple[LithologyCartridge, ...] = ()
    activated_rules: Tuple[DiagnosticRule, ...] = ()
    recommendations: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialización canónica del reporte para trazabilidad.

        Delega la serialización aquí en vez de en el operador, siguiendo
        el principio de responsabilidad única.
        """
        return {
            "soil_tensor": self.soil_tensor.to_dict(),
            "dynamic_rigidity_modulus_pa": self.dynamic_rigidity_modulus_pa,
            "swelling_potential_index": self.swelling_potential_index,
            "yielding_susceptibility_index": self.yielding_susceptibility_index,
            "liquefaction_susceptibility_index": (
                self.liquefaction_susceptibility_index
            ),
            "activated_rules": [r.value for r in self.activated_rules],
            "recommendations": list(self.recommendations),
            "emitted_cartridges_count": len(self.emitted_cartridges),
        }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                    §7  FUNCIONES AUXILIARES                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

def _safe_upper_strip(value: str) -> str:
    """Normaliza cadena: elimina espacios y convierte a mayúsculas."""
    return value.strip().upper()


def _clamp(value: float, lo: float, hi: float) -> float:
    """
    Restricción al intervalo [lo, hi].

    Invariante: lo ≤ hi.
    """
    return max(lo, min(value, hi))


def _assert_finite(value: float, name: str) -> float:
    """
    Verifica que un resultado intermedio sea finito.

    Lanza
    -----
    LithologicalNumericalError
        Si el valor es NaN o ±∞.
    """
    if not math.isfinite(value):
        raise LithologicalNumericalError(
            f"Resultado no finito en '{name}': {value}"
        )
    return value


def _assert_unit_interval(value: float, name: str) -> None:
    """
    Verifica que un valor pertenezca a [0, 1] con tolerancia numérica.

    Lanza
    -----
    LithologicalNumericalError
        Si el valor cae fuera de [-ε, 1+ε].
    """
    eps = GeomechanicalConstants.EPSILON_MACH
    if value < -eps or value > 1.0 + eps:
        raise LithologicalNumericalError(
            f"Índice '{name}' = {value} fuera de [0, 1]."
        )


def _sigmoid(x: float, midpoint: float, steepness: float) -> float:
    """
    Función sigmoidal logística normalizada a [0, 1].

        σ(x) = 1 / (1 + exp(-k·(x - x₀)))

    Propiedades
    -----------
    - σ : ℝ → (0, 1), continua, monótona creciente.
    - σ(x₀) = 0.5.
    - Para |k·(x - x₀)| grande, saturación asintótica.

    Se incluye protección contra overflow en exp().
    """
    z = steepness * (x - midpoint)
    # Protección contra overflow: exp(z) para z > 700 ≈ ∞ en float64
    if z > 700.0:
        return 1.0
    if z < -700.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def _safe_divide(
    numerator: float,
    denominator: float,
    *,
    fallback: float = 0.0,
    eps: float = GeomechanicalConstants.EPSILON_MACH,
) -> float:
    """
    División segura con fallback explícito.

    Si |denominador| < eps, retorna el valor de fallback en lugar de
    producir valores astronómicamente grandes.  Esto es más seguro que
    dividir por max(denom, eps) cuando el numerador puede ser grande.

    Parámetros
    ----------
    numerator : float
    denominator : float
    fallback : float
        Valor retornado si el denominador es degenerado.
    eps : float
        Tolerancia de degeneración.
    """
    if abs(denominator) < eps:
        return fallback
    return numerator / denominator


def _parse_bool(value: Any) -> bool:
    """
    Parser booleano robusto.

    Acepta
    ------
    - bool
    - int / float (0 → False, distinto de 0 → True)
    - str: "true", "false", "1", "0", "yes", "no", "si", "sí"

    Lanza
    -----
    LithologicalInputError
        Si el valor no es interpretable como booleano.
    """
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise LithologicalInputError(
                f"No se puede interpretar {value!r} como booleano."
            )
        return bool(value)

    if isinstance(value, str):
        normalized = value.strip().lower()
        _TRUE_TOKENS: FrozenSet[str] = frozenset({
            "true", "1", "yes", "y", "si", "sí",
        })
        _FALSE_TOKENS: FrozenSet[str] = frozenset({
            "false", "0", "no", "n",
        })
        if normalized in _TRUE_TOKENS:
            return True
        if normalized in _FALSE_TOKENS:
            return False

    raise LithologicalInputError(
        f"No fue posible interpretar booleano desde: {value!r}"
    )


def _normalize_uscs(uscs: str) -> str:
    """
    Normaliza la clasificación USCS.

    Preserva guiones para grupos duales (e.g. "SW-SM") y convierte
    a mayúsculas.
    """
    return _safe_upper_strip(uscs)


def _is_peat(uscs: str) -> bool:
    """Determina si la clasificación corresponde a turba (PT)."""
    return _normalize_uscs(uscs) in USCS_ORGANIC_EXTREME_GROUPS


def _is_sand_like(uscs: str) -> bool:
    """
    Determina si la clasificación sugiere suelo arenoso o con matriz
    arenosa.

    Criterio
    --------
    Códigos que comienzan con 'S' (SW, SP, SM, SC y sus duales) o
    que contienen '-S' como componente secundario.  Se mantiene
    conservador para el caso de licuación.
    """
    u = _normalize_uscs(uscs)
    return u.startswith("S") or "-S" in u


def _is_fine_grained(uscs: str) -> bool:
    """Determina si la clasificación corresponde a suelo fino."""
    return _normalize_uscs(uscs) in USCS_FINE_GROUPS


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║              §8  OPERADOR LITOLÓGICO (MORFISMO PRINCIPAL)                  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class LithologicalManifold(Morphism):
    """
    Morfismo Φ : V_externo → V_PHYSICS.

    Diagrama categórico
    --------------------
    ::

        V_externo ──Φ──→ V_PHYSICS
            │                 │
            │   parse         │   enrich
            ↓                 ↓
        SoilTensor ──eval──→ DiagnosticReport ──assemble──→ CategoricalState

    Responsabilidades
    -----------------
    1.  Parsear y validar el estado litológico (SoilTensor).
    2.  Calcular indicadores geomecánicos derivados normalizados.
    3.  Emitir cartuchos diagnósticos según reglas de umbral.
    4.  Producir un CategoricalState con trazabilidad explícita.

    Consideraciones
    ---------------
    - El componente no sustituye un estudio geotécnico formal.
    - Las reglas implementadas son conservadoras y auditables.
    - La composición con otros morfismos preserva la estructura
      categórica por herencia de Morphism.
    """

    def __init__(self) -> None:
        super().__init__(
            name="lithological_evaluation",
            target_stratum=Stratum.PHYSICS,
        )
        self._metric_tensor = MetricTensorFactory.build_physics_tensor()

    # ------------------------------------------------------------------
    # §8.1  Parseo y validación
    # ------------------------------------------------------------------

    def _parse_soil_tensor(self, payload: Mapping[str, Any]) -> SoilTensor:
        """
        Convierte el payload externo en un SoilTensor validado.

        Estrategia de defaults
        ----------------------
        - USCS: "UN" (desconocido → fallará en validate si no es grupo válido).
        - Numéricos: valores conservadores centrales.
        - Booleanos: False (no saturado).

        Lanza
        -----
        LithologicalInputError
            Si los tipos son incompatibles o la validación falla.
        """
        try:
            soil_tensor = SoilTensor(
                uscs_classification=str(payload.get("uscs", "SW")),
                liquid_limit=float(payload.get("liquid_limit", 0.0)),
                plasticity_index=float(payload.get("plasticity_index", 0.0)),
                shear_wave_velocity=float(payload.get("vs", 300.0)),
                void_ratio=float(payload.get("void_ratio", 0.5)),
                is_saturated=_parse_bool(payload.get("is_saturated", False)),
                bulk_density_kg_m3=float(
                    payload.get(
                        "bulk_density_kg_m3",
                        GeomechanicalConstants.DEFAULT_SOIL_DENSITY_KG_M3,
                    )
                ),
            )
        except (TypeError, ValueError) as exc:
            raise LithologicalInputError(
                f"Degeneración de tipos en input litológico: {exc}"
            ) from exc

        soil_tensor.validate()
        return soil_tensor

    # ------------------------------------------------------------------
    # §8.2  Magnitudes derivadas
    # ------------------------------------------------------------------

    def _compute_dynamic_rigidity_pa(self, soil: SoilTensor) -> float:
        """
        Módulo cortante dinámico máximo (aproximación elástica).

            G_max = ρ · Vs²

        Verificación dimensional
        ------------------------
            [kg/m³] · [m/s]² = [kg/(m·s²)] = [Pa]  ✓

        El resultado se verifica como finito antes de retornar.
        """
        rho = soil.bulk_density_kg_m3
        vs = soil.shear_wave_velocity
        g_max = rho * vs * vs
        return _assert_finite(g_max, "G_max")

    def _compute_swelling_potential_index(self, soil: SoilTensor) -> float:
        """
        Índice normalizado de expansividad potencial.

        Construcción
        ------------
        Se define el producto de actividad:

            A = LL · PI / 100

        y se mapea mediante función sigmoidal:

            I_sw = σ(A; x₀, k)

        donde x₀ y k son parámetros configurables.

        Propiedades
        -----------
        - I_sw ∈ (0, 1)  estrictamente.
        - Monótona creciente en LL y PI.
        - Continuamente diferenciable (C^∞).
        - Acotada superiormente: evita divergencia para suelos extremos.
        - σ(x₀) = 0.5 por construcción.

        Nota
        ----
        El producto LL·PI correlaciona con potencial de cambio
        volumétrico (correlación de Van der Merwe, 1964).
        """
        C = GeomechanicalConstants
        activity_product = soil.liquid_limit * soil.plasticity_index / 100.0
        index = _sigmoid(
            activity_product,
            C.SWELLING_SIGMOID_MIDPOINT,
            C.SWELLING_SIGMOID_STEEPNESS,
        )
        return _assert_finite(index, "swelling_potential_index")

    def _compute_yielding_susceptibility_index(
        self, soil: SoilTensor
    ) -> float:
        """
        Índice normalizado de susceptibilidad a cedencia/asentamiento.

        Construcción
        ------------
        El cociente crudo tiene dimensiones:

            R = e₀² / Vs    [s/m]

        Se adimensionaliza mediante una escala de referencia:

            R* = R / R_ref

        y se mapea al intervalo [0, 1] con saturación:

            I_y = min(1, R*)

        Propiedades
        -----------
        - I_y ∈ [0, 1].
        - Crece con e₀ (más vacíos → más compresible).
        - Decrece con Vs (más rígido → menos susceptible).
        - Adimensional por construcción.
        - Continua y monótona en cada variable por separado.
        """
        C = GeomechanicalConstants
        raw = _safe_divide(
            soil.void_ratio ** 2,
            soil.shear_wave_velocity,
            fallback=0.0,
        )
        normalized = raw / C.YIELDING_REFERENCE_SCALE
        index = _clamp(normalized, 0.0, 1.0)
        return _assert_finite(index, "yielding_susceptibility_index")

    def _compute_compression_index_proxy(self, soil: SoilTensor) -> float:
        """
        Proxy correlacional del índice de compresión Cc.

        Correlación de Skempton (1944)
        ------------------------------
            Cc ≈ 0.009 · (LL - 10)

        Esta correlación es más robusta que la basada únicamente en e₀
        porque incorpora la mineralogía a través del límite líquido.

        Si LL ≤ 10, se retorna 0 (suelo no plástico o granular).

        Unidades: adimensional (Cc es adimensional por definición).

        Referencia
        ----------
        Skempton, A.W. (1944). Notes on the Compressibility of Clays.
        QJGS, 100, 119-135.
        """
        C = GeomechanicalConstants
        cc = max(
            0.0,
            C.SKEMPTON_CC_FACTOR * (soil.liquid_limit - C.SKEMPTON_CC_OFFSET),
        )
        return _assert_finite(cc, "compression_index_proxy")

    def _compute_liquefaction_susceptibility_index(
        self, soil: SoilTensor
    ) -> float:
        """
        Índice proxy normalizado de susceptibilidad a licuación.

        Construcción
        ------------
        Condiciones necesarias (conjuntivas):
            1.  Suelo saturado.
            2.  Clasificación tipo arena (S*).

        Si ambas se cumplen:

            r = (Vs_crit - Vs) / Vs_crit

        se mapea no linealmente:

            I_liq = max(0, r)^n

        con n = 2 (exponente configurable), lo que produce:
            - Sensibilidad reducida cerca del umbral (transición suave).
            - Crecimiento acelerado para Vs << Vs_crit.

        Propiedades
        -----------
        - I_liq ∈ [0, 1].
        - I_liq = 0 si Vs ≥ Vs_crit o suelo no saturado o no arenoso.
        - I_liq → 1 cuando Vs → 0.
        - Continua (C⁰ en Vs = Vs_crit; C^∞ en el interior).

        Nota
        ----
        Este índice NO es el CSR normativo de Seed & Idriss (1971).
        Es un indicador ordinal interno de susceptibilidad.
        """
        if not soil.is_saturated:
            return 0.0
        if not _is_sand_like(soil.uscs_classification):
            return 0.0

        C = GeomechanicalConstants
        vs_crit = C.CRITICAL_LIQUEFACTION_VS_M_S
        r = (vs_crit - soil.shear_wave_velocity) / vs_crit
        r_clamped = max(0.0, r)
        index = r_clamped ** C.LIQUEFACTION_NONLINEAR_EXPONENT
        return _assert_finite(_clamp(index, 0.0, 1.0), "liquefaction_index")

    # ------------------------------------------------------------------
    # §8.3  Reglas diagnósticas y emisión de cartuchos
    # ------------------------------------------------------------------

    def _evaluate_diagnostic_rules(
        self, soil: SoilTensor
    ) -> LithologyDiagnosticReport:
        """
        Evalúa el estado litológico, emite cartuchos y produce reporte.

        Esta función es pura respecto al estado externo: no muta payload
        ni context.  Garantiza que todo cartucho emitido contiene índices
        verificados en [0, 1].

        Flujo
        -----
        1.  Detección de singularidad (PT → excepción).
        2.  Cálculo de magnitudes derivadas.
        3.  Evaluación secuencial de reglas diagnósticas.
        4.  Ensamblado del reporte inmutable.

        Lanza
        -----
        LithologicalSingularityError
            Si se detecta turba (PT).
        LithologicalNumericalError
            Si algún indicador resulta no finito.
        """
        C = GeomechanicalConstants
        activated_rules: List[DiagnosticRule] = []
        recommendations: List[str] = []
        cartridges: List[LithologyCartridge] = []

        # ── Singularidad extrema ──
        if _is_peat(soil.uscs_classification):
            logger.critical(
                "Singularidad litológica detectada: uscs=%s",
                soil.uscs_classification,
            )
            raise LithologicalSingularityError(
                "La presencia de Turba (PT) induce colapso de la frontera "
                "de anclaje y exige estrategia de cimentación profunda o "
                "sustitución/mejora integral del estrato."
            )

        # ── Magnitudes derivadas ──
        g_max = self._compute_dynamic_rigidity_pa(soil)
        swelling_index = self._compute_swelling_potential_index(soil)
        yielding_index = self._compute_yielding_susceptibility_index(soil)
        liquefaction_index = self._compute_liquefaction_susceptibility_index(
            soil
        )

        # ── Regla 1: Expansividad ──
        if (
            soil.liquid_limit >= C.SWELLING_LIQUID_LIMIT_THRESHOLD
            and soil.plasticity_index >= C.SWELLING_PI_THRESHOLD
        ):
            activated_rules.append(
                DiagnosticRule.SWELLING_POTENTIAL_DETECTED
            )
            cartridges.append(
                SwellingPlasmonCartridge(
                    liquid_limit=soil.liquid_limit,
                    plasticity_index=soil.plasticity_index,
                    parasitic_capacitance=swelling_index,
                )
            )
            recommendations.append(
                "Verificar potencial expansivo con ensayos de hinchamiento "
                "libre y controlado.  Implementar control hídrico perimetral."
            )
            logger.warning(
                "Cartucho de expansión emitido: LL=%.1f%%, PI=%.1f%%, "
                "I_sw=%.4f",
                soil.liquid_limit,
                soil.plasticity_index,
                swelling_index,
            )

        # ── Regla 2: Compresibilidad / blandura ──
        if (
            soil.void_ratio > C.HIGH_VOID_RATIO_THRESHOLD
            and soil.shear_wave_velocity < C.SOFT_SOIL_VS_M_S
        ):
            compression_index = self._compute_compression_index_proxy(soil)
            activated_rules.append(
                DiagnosticRule.YIELDING_SUSCEPTIBILITY_DETECTED
            )
            cartridges.append(
                YieldingPhononCartridge(
                    compression_index=compression_index,
                    void_ratio=soil.void_ratio,
                    viscous_drag=yielding_index,
                )
            )
            recommendations.append(
                "Evaluar consolidación y asentamientos.  Considerar precarga, "
                "mejora de suelo o cimentación adaptada."
            )
            logger.info(
                "Cartucho de cedencia emitido: e₀=%.3f, Vs=%.1f m/s, "
                "Cc_proxy=%.4f, I_y=%.4f",
                soil.void_ratio,
                soil.shear_wave_velocity,
                compression_index,
                yielding_index,
            )

        # ── Regla 3: Licuación ──
        if liquefaction_index > 0.0:
            activated_rules.append(
                DiagnosticRule.LIQUEFACTION_SUSCEPTIBILITY_DETECTED
            )
            cartridges.append(
                LiquefactionSolitonCartridge(
                    shear_wave_velocity=soil.shear_wave_velocity,
                    susceptibility_index=liquefaction_index,
                )
            )
            recommendations.append(
                "Realizar evaluación formal de licuación (Seed & Idriss o "
                "Boulanger & Idriss) con demanda sísmica y esfuerzos "
                "efectivos de campo."
            )
            logger.critical(
                "Cartucho de licuación emitido: uscs=%s, saturated=%s, "
                "Vs=%.1f m/s, I_liq=%.4f",
                soil.uscs_classification,
                soil.is_saturated,
                soil.shear_wave_velocity,
                liquefaction_index,
            )

        # ── Regla 4: Rigidez dinámica baja ──
        if g_max < C.LOW_DYNAMIC_RIGIDITY_PA:
            activated_rules.append(
                DiagnosticRule.LOW_DYNAMIC_RIGIDITY_REGIME
            )
            recommendations.append(
                f"Rigidez dinámica estimada G_max = {g_max:.0f} Pa "
                f"(< {C.LOW_DYNAMIC_RIGIDITY_PA:.0f} Pa).  Revisar "
                f"interacción suelo-estructura ante cargas dinámicas."
            )

        return LithologyDiagnosticReport(
            soil_tensor=soil,
            dynamic_rigidity_modulus_pa=g_max,
            swelling_potential_index=swelling_index,
            yielding_susceptibility_index=yielding_index,
            liquefaction_susceptibility_index=liquefaction_index,
            emitted_cartridges=tuple(cartridges),
            activated_rules=tuple(activated_rules),
            recommendations=tuple(recommendations),
        )

    # ------------------------------------------------------------------
    # §8.4  Ensamblado de salida categórica
    # ------------------------------------------------------------------

    def __call__(
        self,
        payload: Dict[str, Any],
        context: Dict[str, Any],
    ) -> CategoricalState:
        """
        Ejecuta el morfismo litológico.

        Diagrama de flujo
        -----------------
        ::

            payload ──parse──→ SoilTensor ──eval──→ Report ──assemble──→ State
                                    │                  │
                                    └── validate       └── serialize

        Salida
        ------
        CategoricalState con:
            - payload enriquecido con indicadores y magnitudes derivadas.
            - context actualizado con cartuchos y reporte estructurado.
            - validated_strata = {PHYSICS}.

        Excepciones
        -----------
        LithologicalInputError
            Datos de entrada inválidos.
        LithologicalSingularityError
            Detección de turba u otra singularidad.
        LithologicalNumericalError
            Resultado numérico degenerado.
        LithologicalManifoldError
            Cualquier falla no clasificada (envolvente).
        """
        logger.info("Iniciando evaluación de la variedad litológica.")

        try:
            # Copias defensivas para preservar inmutabilidad del caller
            next_payload: Dict[str, Any] = dict(payload)
            next_context: Dict[str, Any] = dict(context)

            # Pipeline: parse → validate → evaluate → serialize
            soil_tensor = self._parse_soil_tensor(next_payload)
            report = self._evaluate_diagnostic_rules(soil_tensor)

            # Registro sináptico (append, no overwrite)
            synaptic_registry: List[Any] = list(
                next_context.get("synaptic_registry", [])
            )
            synaptic_registry.extend(report.emitted_cartridges)
            next_context["synaptic_registry"] = synaptic_registry

            # Reporte serializado (delegado al dataclass)
            next_context["lithology_report"] = report.to_dict()

            # Enriquecimiento del payload con magnitudes derivadas
            next_payload.update({
                "dynamic_rigidity_modulus_pa": (
                    report.dynamic_rigidity_modulus_pa
                ),
                "swelling_potential_index": report.swelling_potential_index,
                "yielding_susceptibility_index": (
                    report.yielding_susceptibility_index
                ),
                "liquefaction_susceptibility_index": (
                    report.liquefaction_susceptibility_index
                ),
                "lithology_activated_rules": [
                    r.value for r in report.activated_rules
                ],
            })

            logger.info(
                "Evaluación litológica completada: G_max=%.0f Pa, "
                "cartuchos=%d, reglas=%s",
                report.dynamic_rigidity_modulus_pa,
                len(report.emitted_cartridges),
                [r.value for r in report.activated_rules],
            )

            return CategoricalState(
                payload=next_payload,
                context=next_context,
                validated_strata=frozenset({Stratum.PHYSICS}),
            )

        except (
            LithologicalInputError,
            LithologicalSingularityError,
            LithologicalNumericalError,
        ):
            raise
        except Exception as exc:
            logger.exception(
                "Falla no controlada en LithologicalManifold: %s", exc
            )
            raise LithologicalManifoldError(
                f"Falla interna del operador litológico: {exc}"
            ) from exc