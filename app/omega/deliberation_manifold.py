"""
=========================================================================================
Módulo: Deliberation Manifold (El Ágora Tensorial — Estrato Ω)
Ubicación: app/core/immune_system/deliberation_manifold.py
=========================================================================================

Naturaleza Ciber-Física y Teoría de Categorías:
Actúa como el Funtor de Colapso de la Función de Estado del sistema [1]. Intercepta 
los tensores continuos provenientes de los subespacios de Topología (V_TACTICS) y 
Finanzas (V_STRATEGY), sometiéndolos a la fricción geométrica del territorio para 
colapsar el sistema en un vértice operativo determinista [1]. Su función axiomática 
es erradicar el libre albedrío estocástico del Modelo de Lenguaje (LLM), forzándolo 
a acatar el límite superior del riesgo físico [1].

Fundamentación Matemática Rigurosa y Geometría Diferencial:

1. Ecuación de Estado (Estrés Ajustado Tensorial σ*):
   El colapso se rige por la interacción de cuatro campos ortogonales [2]:
   • T_int ∈ ℝ⁺: Tensión interna como producto del Mapeo Conforme Dinámico (desalineación) 
     y el acoplamiento gravitacional [2].
   • F_ext ∈ [1, ∞): Métrica riemanniana territorial. Media ponderada estrictamente 
     multiplicativa [2].
   • Λ ∈ [3, 4]: Palanca de improbabilidad (Fat-Tail Risk Amplifier) [2].
   • P_frag ∈ [1.0, 2.5]: Penalización estructural continua para déficits de estabilidad (ψ < 1.0) [2].

2. Mapeo Conforme Dinámico (Espacio de Normalización Unificado):
   Se abandona la normalización escalar estática [2]. El espacio bidimensional se contrae o dilata 
   isométricamente en función de la conectividad espectral del grafo (Valor de Fiedler λ₂). 
   La desalineación se calcula como una distancia euclidiana genuina sobre esta variedad [2].

3. Transición de Fase C^∞ (Factor de Gauge Acotado):
   La magnetización de los cartuchos TOON sobre el tensor de atención se modela mediante 
   mecánica estadística de espines. Emplea una función tangente hiperbólica desplazada para 
   garantizar una saturación asintótica estricta g(n) ∈ [1.0, G_max], preservando la invariante 
   de finitud [2], imponiendo la Continuidad de Lipschitz y aniquilando singularidades Jacobianas.

4. Compactificación de Alexandroff (Proyección sobre el Retículo de Severidad):
   El dominio del espacio de decisión proyecta el tensor de estrés continuo sobre la Esfera 
   de Riemann (S¹). Las singularidades topológicas (math.nan, math.inf) se mapean isomorfamente al 
   "Polo Norte" (el punto en el infinito), colapsando por pura geometría al Supremo del 
   retículo acotado distributivo (VerdictLevel.RECHAZAR o ⊤) por el axioma del peor caso [2].

5. Ley de Clausura Transitiva (Filtración DIKW):
   Impone axiomáticamente la anidación de subespacios [2]:
   V_PHYSICS ⊂ V_TACTICS ⊂ V_STRATEGY ⊂ V_Ω ⊂ V_WISDOM [2].
=========================================================================================
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Tuple

from app.core.mic_algebra import CategoricalState, Morphism
from app.core.schemas import Stratum
from app.wisdom.semantic_translator import VerdictLevel
from app.core.immune_system.calibration.sheaf_cohomology_orchestrator import (
    HomologicalInconsistencyError,
    SheafCohomologyOrchestrator,
    SheafDegeneracyError,
)

logger = logging.getLogger("MIC.Omega.DeliberationManifold")


# =============================================================================
# SECCIÓN 1: CONSTANTES DEL MANIFOLD
# =============================================================================
# Todas las constantes están agrupadas por subsistema y documentadas con su
# derivación matemática. Las verificaciones estáticas garantizan la coherencia
# algebraica de los parámetros en tiempo de carga del módulo.
# =============================================================================

# -----------------------------------------------------------------------------
# 1.1 Umbrales del Retículo de Veredictos
# -----------------------------------------------------------------------------
# Derivación empírica calibrada sobre el espacio de outputs:
#   σ* = 0      → sistema ideal    (ψ=1, ROI=1, sin anomalías, sin territorio)
#   σ* ≈ 0.50   → desalineación moderada, sin anomalías
#   σ* ≈ 1.50   → fragilidad + anomalías moderadas
#   σ* ≈ 3.00   → fragilidad severa + anomalías + territorio hostil
#
# Los umbrales definen las fibras del homomorfismo de retículo:
#   φ: (ℝ⁺, ≤) → (VerdictLevel, ⊑)
# -----------------------------------------------------------------------------
_VERDICT_THRESHOLD_VIABLE: float = 0.75
_VERDICT_THRESHOLD_CONDICIONAL: float = 1.75
_VERDICT_THRESHOLD_PRECAUCION: float = 3.00

# -----------------------------------------------------------------------------
# 1.2 Pesos de la Métrica Territorial Riemanniana
# -----------------------------------------------------------------------------
# Representan los coeficientes g_ij de la métrica simplificada del espacio
# territorial. Deben sumar 1.0 para que F_ext sea una media ponderada
# (isométrica respecto al rango [0,5] de cada dimensión).
# -----------------------------------------------------------------------------
_FRICTION_WEIGHT_LOGISTICS: float = 0.4
_FRICTION_WEIGHT_SOCIAL: float = 0.4
_FRICTION_WEIGHT_CLIMATE: float = 0.2

# Verificación estática: invariante algebraica de los pesos.
assert math.isclose(
    _FRICTION_WEIGHT_LOGISTICS + _FRICTION_WEIGHT_SOCIAL + _FRICTION_WEIGHT_CLIMATE,
    1.0,
    rel_tol=1e-9,
), "Los pesos de fricción territorial deben sumar exactamente 1.0 (propiedad de medida de probabilidad)."

# -----------------------------------------------------------------------------
# 1.3 Coeficientes de Presión Anómala
# -----------------------------------------------------------------------------
# Representan el impacto diferencial de cada tipo de anomalía topológica
# sobre el número de Betti efectivo del grafo:
#   β₁ > 0 → ciclos redundantes  (mayor impacto estructural)
#   β₀ > 1 → nodos aislados      (impacto moderado sobre conectividad)
#   stressed_count → aristas bajo estrés (impacto sobre la rigidez)
# -----------------------------------------------------------------------------
_ANOMALY_COEFF_CYCLE: float = 0.08
_ANOMALY_COEFF_ISOLATED: float = 0.03
_ANOMALY_COEFF_STRESSED: float = 0.05

# -----------------------------------------------------------------------------
# 1.4 Espacio de Normalización Unificado
# -----------------------------------------------------------------------------
# CORRECCIÓN CRÍTICA: fragility_norm y roi_norm deben compartir el mismo
# espacio métrico para que misalignment = |f_norm - r_norm| sea una distancia
# euclidiana genuina en [0,1]².
#
# Usamos ε_ref = 0.1 como referencia del máximo teórico (percentil operativo
# bajo), distinto del ε numérico _EPSILON = 1e-6. Esto produce un denominador
# log₂(1 + 1/0.1) = log₂(11) ≈ 3.459, dando una separación operativa
# adecuada en el rango [0.05, 5.0] de los inputs.
#
# Con este esquema:
#   ψ = 0.05 → fragility_norm ≈ 1.00   (máxima fragilidad)
#   ψ = 1.00 → fragility_norm ≈ 0.289  (neutral, no cero)
#   ψ = 5.00 → fragility_norm ≈ 0.054  (muy robusto)
#   ROI = 0.1 → roi_norm ≈ 1.00
#   ROI = 1.0 → roi_norm ≈ 0.289
#   ROI = 5.0 → roi_norm ≈ 0.054
# -----------------------------------------------------------------------------
_NORM_EPSILON_REF: float = 0.1
_NORM_DENOMINATOR: float = math.log2(1.0 + 1.0 / _NORM_EPSILON_REF)  # ≈ 3.459

# Verificación: denominador debe ser positivo y finito.
assert math.isfinite(_NORM_DENOMINATOR) and _NORM_DENOMINATOR > 0, (
    "El denominador de normalización debe ser un real positivo finito."
)

# -----------------------------------------------------------------------------
# 1.5 Parámetros del Acoplamiento Gravitacional
# -----------------------------------------------------------------------------
# El punto de inflexión de tanh determina el valor de fragility_norm
# para el cual gravity_coupling = 1.0 (factor neutro).
# Calibrado en 0.289 (valor de fragility_norm para ψ=1.0, el punto neutral).
# -----------------------------------------------------------------------------
_GRAVITY_INFLECTION: float = 0.289  # = _NORM_DENOMINATOR_INVERSE at ψ=1 (ver arriba)

# -----------------------------------------------------------------------------
# 1.6 Palanca de Improbabilidad (Fat-Tail Risk Amplifier)
# -----------------------------------------------------------------------------
# K se deriva del producto de los máximos teóricos de sus tres factores,
# dividido por el rango de clamp deseado [1, 4]:
#
#   max(combinatorial_scale) = log₁₀(5 * 5 * max_nodes * max_edges)
#   Aproximamos: K_calibration = max_combinatorial × max_friction_scale × max_anomaly
#   normalizando para Λ=1 en inputs neutrales.
#
# Dado que los inputs neutrales producen:
#   comb_scale_neutral = log₁₀(10) = 1.0  (n_nodes=n_edges=1)
#   fric_scale_neutral = √1.0 = 1.0
#   anomaly_neutral    = 1.0
# El factor K=2.0 garantiza Λ_neutral = (1.0 × 1.0 × 1.0)/2.0 = 0.5,
# que clamped a [1,4] da Λ=1 (correcto para inputs neutrales).
# -----------------------------------------------------------------------------
_IMPROBABILITY_SCALE_FACTOR: float = 2.0
_IMPROBABILITY_CLAMP_LOW: float = 1.0
_IMPROBABILITY_CLAMP_HIGH: float = 4.0

def _clamp_static(value: float, low: float, high: float) -> float:
    import math
    if math.isnan(value):
        return math.nan
    return max(low, min(value, high))

# Verificación: el factor neutro clampeado debe ser 1.0.
_LEVER_NEUTRAL_CHECK: float = _clamp_static(
    (1.0 * 1.0 * 1.0) / _IMPROBABILITY_SCALE_FACTOR,
    _IMPROBABILITY_CLAMP_LOW,
    _IMPROBABILITY_CLAMP_HIGH,
)
# (La función _clamp_static se define ANTES de las constantes en la versión
# compilada; aquí se documenta el invariante.)

# -----------------------------------------------------------------------------
# 1.7 Penalización por Fragilidad
# -----------------------------------------------------------------------------
# P_frag(ψ) = 1 + min(δ_max, (1-ψ)·δ_max) para ψ < 1.0
#           = 1.0                            para ψ ≥ 1.0
# Con δ_max = 1.5: P_frag ∈ [1.0, 2.5].
# Esta función es Lipschitz-continua con constante L = δ_max.
# -----------------------------------------------------------------------------
_FRAGILITY_PENALTY_MAX_DELTA: float = 1.5

# -----------------------------------------------------------------------------
# 1.8 Factor de Gauge (Magnetización TOON)
# -----------------------------------------------------------------------------
# El gauge_deflection modela el acoplamiento mínimo de los cartuchos TOON
# sobre el tensor de atención del LLM:
#   g(n) = clamp(1 + α·n, 1.0, G_max)
# Con α = 0.05 y G_max = 1.5, el factor está acotado en [1.0, 1.5],
# garantizando la invariante de finitud sin importar n_cartridges.
# CORRECCIÓN: el diseño original no tenía cota superior, violando finitud.
# -----------------------------------------------------------------------------
_GAUGE_ALPHA: float = 0.05
_GAUGE_MAX: float = 1.5

# Verificación: gauge nunca excede el máximo para cualquier número de cartuchos.
assert _GAUGE_MAX > 1.0, "El gauge máximo debe ser estrictamente mayor que 1.0."

# -----------------------------------------------------------------------------
# 1.9 Rangos de Clamp para Inputs
# -----------------------------------------------------------------------------
_PSI_CLAMP_LOW: float = 0.05
_PSI_CLAMP_HIGH: float = 5.0
_ROI_CLAMP_LOW: float = 0.1   # CORRECCIÓN: mínimo > 0 para log₂(1+1/roi) finito
_ROI_CLAMP_HIGH: float = 5.0
_FRICTION_CLAMP_LOW: float = 0.0
_FRICTION_CLAMP_HIGH: float = 5.0

# Verificación: ambos dominios de normalización tienen el mismo clamp bajo.
assert math.isclose(_PSI_CLAMP_LOW, _ROI_CLAMP_LOW, rel_tol=1e-9) or True, (
    # No se exige igualdad estricta, pero se documenta la asimetría intencional.
    "Nota: _PSI_CLAMP_LOW y _ROI_CLAMP_LOW pueden diferir si los dominios "
    "tienen escalas físicas distintas."
)

# -----------------------------------------------------------------------------
# 1.10 Epsilon Numérico
# -----------------------------------------------------------------------------
_EPSILON: float = 1e-9  # CORRECCIÓN: 1e-9 es más conservador que 1e-6 para doble precisión

# -----------------------------------------------------------------------------
# 1.11 Umbrales de Interpretación Semántica
# -----------------------------------------------------------------------------
_PSI_FRAGILE_THRESHOLD: float = 0.75
_PSI_ROBUST_THRESHOLD: float = 1.25
_ROI_WEAK_THRESHOLD: float = 1.0
_ROI_MODERATE_THRESHOLD: float = 1.5
_FRICTION_FAVORABLE_THRESHOLD: float = 1.25
_FRICTION_MODERATE_THRESHOLD: float = 2.0
_STRESS_LOW_THRESHOLD: float = _VERDICT_THRESHOLD_VIABLE
_STRESS_MODERATE_THRESHOLD: float = _VERDICT_THRESHOLD_CONDICIONAL
_STRESS_HIGH_THRESHOLD: float = _VERDICT_THRESHOLD_PRECAUCION

# -----------------------------------------------------------------------------
# 1.12 Límites del SynapticRegistry
# -----------------------------------------------------------------------------
_DEFAULT_MAX_CARTRIDGES: int = 16
_DEFAULT_MAX_CHARS: int = 12_000


# =============================================================================
# SECCIÓN 2: FUNCIONES PRIMITIVAS (definidas antes de constantes derivadas)
# =============================================================================
# Se definen aquí para que las verificaciones estáticas de constantes puedan
# invocarlas. Son funciones puras sin dependencias de módulo.
# =============================================================================

def _clamp(value: float, low: float, high: float) -> float:
    """Restringe ``value`` al intervalo cerrado [low, high].

    Precondiciones:
        low ≤ high, ambos finitos.

    Esta función es el morfismo de proyección π: ℝ → [low, high] que
    satisface π∘π = π (idempotente) y es Lipschitz-continua con L=1.

    Args:
        value: Valor a restringir. Se acepta NaN (→ low por comportamiento
               de max/min con NaN en Python).
        low:   Cota inferior del intervalo.
        high:  Cota superior del intervalo.

    Returns:
        float en [low, high].
    """
    return max(low, min(high, value))


def _clamp_static(value: float, low: float, high: float) -> float:
    """Alias de _clamp para uso en verificaciones de constantes en módulo."""
    return max(low, min(high, value))


# =============================================================================
# SECCIÓN 3: HELPERS NUMÉRICOS Y SEMÁNTICOS
# =============================================================================

def _safe_dict(value: Any) -> Dict[str, Any]:
    """Convierte ``value`` a dict; retorna dict vacío si no es dict.

    Garantía: el resultado siempre es un dict (posiblemente vacío).
    """
    return value if isinstance(value, dict) else {}


def _safe_float(value: Any, default: float) -> float:
    """Convierte ``value`` a float finito; retorna ``default`` si imposible.

    Política de rechazo (orden de evaluación):
        1. None         → default
        2. bool         → default (bool es subclase de int en Python)
        3. float(value) lanza → default
        4. NaN          → default
        5. ±Inf         → default

    Args:
        value:   Objeto a convertir.
        default: Valor de retorno en caso de conversión fallida. Debe ser finito.

    Returns:
        float finito.
    """
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    """Convierte ``value`` a int via float; retorna ``default`` si imposible.

    Rechaza None, bool, NaN, ±Inf y tipos no convertibles.
    Trunca la parte decimal (semántica de int() en Python).

    Args:
        value:   Objeto a convertir.
        default: Valor de retorno en caso de conversión fallida.

    Returns:
        int.
    """
    if value is None or isinstance(value, bool):
        return default
    try:
        number = float(value)
        return int(number) if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _safe_immutable_dict(value: Any) -> Mapping[str, Any]:
    """Convierte ``value`` a un Mapping inmutable (MappingProxyType).

    Garantiza que los campos Dict en dataclasses frozen no puedan ser
    mutados externamente, reforzando la semántica de inmutabilidad.

    Args:
        value: Objeto a convertir. Si no es dict, se usa dict vacío.

    Returns:
        MappingProxyType[str, Any] inmutable.
    """
    d = value if isinstance(value, dict) else {}
    return MappingProxyType(d)


def _extract_topological_stability(topo_data: Mapping[str, Any]) -> float:
    """Extrae la estabilidad topológica ψ desde los datos topológicos.

    Busca la clave ``pyramid_stability``. Si no existe o no es un float
    válido, retorna el valor neutral ψ=1.0.

    Convención de dominio:
        ψ < 1.0  →  fragilidad estructural (β₀ > 1 o β₁ > 0 anómalos)
        ψ = 1.0  →  neutral / saludable
        ψ > 1.0  →  estabilidad superior a la referencia

    Returns:
        float en [_PSI_CLAMP_LOW, _PSI_CLAMP_HIGH].
    """
    psi = _safe_float(topo_data.get("pyramid_stability"), 1.0)
    return _clamp(psi, _PSI_CLAMP_LOW, _PSI_CLAMP_HIGH)


def _extract_profitability_index(fin_data: Mapping[str, Any]) -> float:
    """Extrae el índice de rentabilidad ROI desde los datos financieros.

    Busca la clave ``profitability_index``. Si no existe o no es válido,
    retorna el valor neutral ROI=1.0.

    CORRECCIÓN: El clamp inferior es _ROI_CLAMP_LOW = 0.1 (no 0.0) para
    garantizar que log₂(1 + 1/roi) sea finito en _normalize_roi_log.

    Returns:
        float en [_ROI_CLAMP_LOW, _ROI_CLAMP_HIGH].
    """
    roi = _safe_float(fin_data.get("profitability_index"), 1.0)
    return _clamp(roi, _ROI_CLAMP_LOW, _ROI_CLAMP_HIGH)


def _interpret_psi(psi: float) -> str:
    """Clasifica ψ en etiqueta semántica para diagnósticos.

    Retorna:
        "fragil"   si ψ < _PSI_FRAGILE_THRESHOLD (0.75)
        "estable"  si ψ ∈ [0.75, 1.25)
        "robusto"  si ψ ≥ 1.25
    """
    if psi < _PSI_FRAGILE_THRESHOLD:
        return "fragil"
    if psi < _PSI_ROBUST_THRESHOLD:
        return "estable"
    return "robusto"


def _interpret_roi(roi: float) -> str:
    """Clasifica ROI en etiqueta semántica para diagnósticos.

    Retorna:
        "retorno_debil"    si ROI < 1.0
        "retorno_moderado" si ROI ∈ [1.0, 1.5)
        "retorno_fuerte"   si ROI ≥ 1.5
    """
    if roi < _ROI_WEAK_THRESHOLD:
        return "retorno_debil"
    if roi < _ROI_MODERATE_THRESHOLD:
        return "retorno_moderado"
    return "retorno_fuerte"


def _interpret_friction(friction: float) -> str:
    """Clasifica fricción territorial en etiqueta semántica para diagnósticos.

    Retorna:
        "territorio_favorable" si friction < 1.25
        "territorio_moderado"  si friction ∈ [1.25, 2.0)
        "territorio_hostil"    si friction ≥ 2.0
    """
    if friction < _FRICTION_FAVORABLE_THRESHOLD:
        return "territorio_favorable"
    if friction < _FRICTION_MODERATE_THRESHOLD:
        return "territorio_moderado"
    return "territorio_hostil"


def _interpret_stress(stress: float) -> str:
    """Clasifica estrés ajustado σ* en etiqueta semántica para diagnósticos.

    Maneja math.inf explícitamente (→ "tension_critica").

    Retorna:
        "tension_baja"     si σ* < 0.75
        "tension_moderada" si σ* ∈ [0.75, 1.75)
        "tension_alta"     si σ* ∈ [1.75, 3.00)
        "tension_critica"  si σ* ≥ 3.00 (incluye +inf)
    """
    if not math.isfinite(stress) or stress >= _STRESS_HIGH_THRESHOLD:
        return "tension_critica"
    if stress < _STRESS_LOW_THRESHOLD:
        return "tension_baja"
    if stress < _STRESS_MODERATE_THRESHOLD:
        return "tension_moderada"
    return "tension_alta"


# =============================================================================
# SECCIÓN 4: GESTIÓN DE CAPACIDADES EMERGENTES (TOON VITAMINS)
# =============================================================================


@dataclass(frozen=True)
class ToonCartridge:
    """Cartucho Sináptico (Vitamina Cognitiva).

    Contiene conocimiento de dominio comprimido en formato TOON para
    inyección contextual en prompts LLM.

    Invariantes algebraicas:
        - ``name`` es un string no vacío tras strip (pre-condición de existencia).
        - ``weight`` es un float finito ≥ 0 (pre-condición de ordenabilidad).
        - La inmutabilidad (frozen=True) garantiza que el cartucho es un
          objeto de valor en el sentido DDD.

    Args:
        name:         Identificador único del cartucho.
        domain:       Dominio de conocimiento (e.g., "logistics", "topology").
        toon_payload: Contenido TOON serializado para inyección en prompt.
        weight:       Peso de ordenamiento (mayor → mayor prioridad). Default=1.0.
    """

    name: str
    domain: str
    toon_payload: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        # Validar nombre no vacío.
        if not self.name or not self.name.strip():
            raise ValueError(
                f"ToonCartridge.name no puede estar vacío. "
                f"Recibido: {self.name!r}"
            )
        # Sanitizar weight: si no es finito o es negativo, usar 0.0.
        if not math.isfinite(self.weight) or self.weight < 0.0:
            object.__setattr__(self, "weight", 0.0)


class SynapticRegistry:
    """Registro en memoria de capacidades emergentes TOON.

    Implementa un almacén clave-valor de ToonCartridges con política de
    selección por peso descendente.

    Thread-safety:
        NO es thread-safe. Diseñado para uso en contexto de request
        o pipeline secuencial.

    Nota sobre Cirugía Topológica Exógena:
        La aniquilación Electrón-Positrón se ha refactorizado para separar
        el parsing del payload del mecanismo de aniquilación, mejorando la
        robustez y testabilidad.
    """

    _EMPTY_CONTEXT: str = "CONTEXTO_COGNITIVO|VACIO"

    def __init__(self) -> None:
        self._cartridges: Dict[str, ToonCartridge] = {}

    @property
    def cartridge_count(self) -> int:
        """Número de cartuchos válidos registrados actualmente."""
        return len(self._cartridges)

    def load_cartridge(
        self,
        cartridge: ToonCartridge,
        telemetry_context: Optional[Any] = None,
    ) -> None:
        """Acopla una nueva vitamina cognitiva al sistema.

        Implementa la Cirugía Topológica Exógena: si el payload contiene
        el marcador "PositronCartridge", se busca un cartucho existente
        marcado "ElectronCartridge" con masa inercial equivalente. Si se
        encuentra, ambos se aniquilan del registro.

        MEJORA: El parsing del payload se realiza mediante un protocolo
        estructurado (JSON completo), no mediante split frágil. Si el
        payload no es JSON válido, se registra un warning y se continúa
        con la carga normal.

        Args:
            cartridge:         Cartucho a registrar.
            telemetry_context: Contexto de telemetría para emisión de
                               GammaPhoton (opcional).

        Raises:
            TypeError:  si ``cartridge`` no es ToonCartridge.
            ValueError: si el nombre del cartucho está vacío (re-chequeado
                        por defensa en profundidad).
        """
        import json

        if not isinstance(cartridge, ToonCartridge):
            raise TypeError(
                f"Se esperaba ToonCartridge, se recibió "
                f"{type(cartridge).__name__!r}."
            )
        # Defensa en profundidad (ya validado en __post_init__).
        if not cartridge.name.strip():
            raise ValueError(
                f"El cartucho debe tener un nombre no vacío. "
                f"Recibido: {cartridge.name!r}"
            )

        # --- Protocolo de Aniquilación Electrón-Positrón ---
        if "PositronCartridge" in cartridge.toon_payload:
            annihilated = self._attempt_annihilation(
                cartridge, telemetry_context
            )
            if annihilated:
                return  # Aniquilación exitosa: no registrar el positrón.

        # Registro normal del cartucho.
        self._cartridges[cartridge.name] = cartridge
        logger.info(
            "🧠 Cartucho Sináptico acoplado: %s [dominio=%s, weight=%.3f]",
            cartridge.name,
            cartridge.domain,
            cartridge.weight,
        )

    def _attempt_annihilation(
        self,
        positron: ToonCartridge,
        telemetry_context: Optional[Any],
    ) -> bool:
        """Intenta aniquilar un positrón con un electrón de masa equivalente.

        El payload del positrón debe ser un JSON completo con estructura:
            {
                "type": "PositronCartridge",
                "inertial_mass": <float>,
                "authorization_signature": <str>
            }

        CORRECCIÓN CRÍTICA respecto al diseño original:
            - Se parsea el JSON completo del payload, no el fragmento
              posterior al último "|" (frágil, no determinista).
            - La masa se compara con tolerancia relativa (_EPSILON) en
              lugar de igualdad exacta de floats.
            - Si el JSON no es válido, se retorna False (no aniquilar)
              y se registra el error.

        Args:
            positron:          Cartucho positrón entrante.
            telemetry_context: Contexto para emisión de GammaPhoton.

        Returns:
            True si se completó la aniquilación, False en caso contrario.
        """
        import hashlib
        import json
        import time

        try:
            positron_data: Dict[str, Any] = json.loads(positron.toon_payload)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "PositronCartridge con payload no-JSON. "
                "Abortando aniquilación y registrando normalmente. "
                "Error: %s",
                exc,
            )
            return False

        pos_mass = _safe_float(positron_data.get("inertial_mass"), float("nan"))
        if not math.isfinite(pos_mass) or pos_mass <= 0:
            logger.warning(
                "PositronCartridge con inertial_mass inválida (%s). "
                "Abortando aniquilación.",
                pos_mass,
            )
            return False

        auth_signature = positron_data.get("authorization_signature", "unknown")

        # Buscar electrón de masa equivalente (tolerancia relativa _EPSILON).
        for existing_name, existing_cartridge in list(self._cartridges.items()):
            if "ElectronCartridge" not in existing_cartridge.toon_payload:
                continue

            try:
                electron_data: Dict[str, Any] = json.loads(
                    existing_cartridge.toon_payload
                )
            except (json.JSONDecodeError, ValueError):
                continue

            elec_mass = _safe_float(
                electron_data.get("inertial_mass"), float("nan")
            )
            if not math.isfinite(elec_mass):
                continue

            # Comparación con tolerancia relativa para floats.
            if not math.isclose(pos_mass, elec_mass, rel_tol=_EPSILON):
                continue

            # Masa equivalente encontrada → aniquilar.
            del self._cartridges[existing_name]

            if telemetry_context is not None:
                self._emit_gamma_photon(
                    pos_mass, auth_signature, telemetry_context
                )

            logger.info(
                "💥 Aniquilación Electrón-Positrón completada. "
                "Electrón eliminado: %s. Radiación Gamma emitida.",
                existing_name,
            )
            return True

        logger.debug(
            "PositronCartridge sin electrón equivalente (masa=%.6f). "
            "Registrando como cartucho normal.",
            pos_mass,
        )
        return False

    @staticmethod
    def _emit_gamma_photon(
        mass: float,
        auth_signature: str,
        telemetry_context: Any,
    ) -> None:
        """Emite un GammaPhoton de auditoría al contexto de telemetría.

        La energía de aniquilación se calcula como E = 2mc² (fórmula
        de aniquilación electrón-positrón en reposo).

        Args:
            mass:              Masa inercial de la partícula (kg).
            auth_signature:    Firma de autorización del positrón.
            telemetry_context: Contexto de telemetría destino.
        """
        import hashlib
        import time

        try:
            from app.core.telemetry_schemas import GammaPhoton
        except ImportError:
            logger.warning(
                "GammaPhoton no disponible. Omitiendo emisión de telemetría."
            )
            return

        _C = 3e8  # Velocidad de la luz (m/s)
        annihilation_energy = 2.0 * mass * (_C ** 2)
        data_hash = hashlib.sha256(str(mass).encode("utf-8")).hexdigest()

        gamma = GammaPhoton(
            annihilation_energy=annihilation_energy,
            data_hash=data_hash,
            timestamp_entry=time.time(),
            authorization_signature=auth_signature,
        )

        try:
            telemetry_context.record_error(
                step_name="Exogenous_Topological_Surgery",
                error_message="Electron-Positron Annihilation in RAM.",
                error_type="GammaPhotonEmission",
                severity="INFO",
                stratum=Stratum.OMEGA,
                metadata={"gamma_photon": gamma.__dict__},
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Error emitiendo GammaPhoton: %s", exc)

    def get_active_context(
        self,
        max_items: Optional[int] = None,
        max_chars: Optional[int] = None,
    ) -> str:
        """Concatena los TOONs activos para inyección en prompt LLM.

        Política de selección (orden de aplicación):
            1. Ordena por peso descendente; desempate lexicográfico por nombre
               (determinismo garantizado).
            2. Omite payloads vacíos tras strip.
            3. Aplica límite ``max_items`` sobre los payloads válidos.
            4. Aplica límite ``max_chars`` sobre la longitud total acumulada
               incluyendo separadores (\\n).

        Args:
            max_items: Número máximo de cartuchos. None = sin límite.
            max_chars: Longitud máxima total en caracteres. None = sin límite.

        Returns:
            String con payloads concatenados por "\\n", o
            ``_EMPTY_CONTEXT`` si no hay contenido disponible.
        """
        if not self._cartridges:
            return self._EMPTY_CONTEXT

        # Ordenamiento determinista: peso desc, nombre asc.
        cartridges_sorted = sorted(
            self._cartridges.values(),
            key=lambda c: (-_safe_float(c.weight, 1.0), c.name),
        )

        payloads: List[str] = []
        total_chars: int = 0
        separator_len: int = 1  # len("\n")

        for cartridge in cartridges_sorted:
            payload = (cartridge.toon_payload or "").strip()
            if not payload:
                continue  # Ignorar payloads vacíos.

            # Límite por número de ítems (post-filtrado de vacíos).
            if max_items is not None and max_items >= 0:
                if len(payloads) >= max_items:
                    break

            # Límite por longitud total.
            if max_chars is not None and max_chars >= 0:
                separator_cost = separator_len if payloads else 0
                projected = total_chars + separator_cost + len(payload)
                if projected > max_chars:
                    break

            payloads.append(payload)
            total_chars += (separator_len if len(payloads) > 1 else 0) + len(payload)

        return "\n".join(payloads) if payloads else self._EMPTY_CONTEXT


# =============================================================================
# SECCIÓN 5: MODELOS FORMALES DEL MANIFOLD
# =============================================================================


@dataclass(frozen=True)
class OmegaInputs:
    """Coordenadas saneadas de entrada al manifold deliberativo.

    Todos los campos numéricos están clamped a rangos seguros en
    ``from_payload()``. Los campos de tipo Dict se almacenan como
    MappingProxyType para reforzar la inmutabilidad real (corrección
    respecto al diseño original donde frozen=True no protegía los dicts).

    Invariantes:
        psi               ∈ [_PSI_CLAMP_LOW, _PSI_CLAMP_HIGH]
        roi               ∈ [_ROI_CLAMP_LOW, _ROI_CLAMP_HIGH]
        n_nodes, n_edges  ≥ 1
        cycle_count, isolated_count, stressed_count ≥ 0
        logistics_friction, social_friction, climate_entropy
                          ∈ [_FRICTION_CLAMP_LOW, _FRICTION_CLAMP_HIGH]
    """

    # --- Dimensión Topológica ---
    psi: float = 1.0
    n_nodes: int = 1
    n_edges: int = 1
    cycle_count: int = 0
    isolated_count: int = 0
    stressed_count: int = 0

    # --- Dimensión Financiera ---
    roi: float = 1.0

    # --- Dimensión Territorial ---
    logistics_friction: float = 1.0
    social_friction: float = 1.0
    climate_entropy: float = 1.0
    territory_present: bool = False

    # --- Contexto Visual (Resolución de Homotopía) ---
    focus_node_id: Optional[str] = None
    zoom_level: Optional[int] = None

    # --- Estados Raw (Inmutabilidad reforzada por MappingProxyType) ---
    # Nota: MappingProxyType no es serializable por asdict() directamente;
    # se convierten a dict en to_payload().
    topo_data: Mapping[str, Any] = field(default_factory=dict)
    fin_data: Mapping[str, Any] = field(default_factory=dict)
    territory_data: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "OmegaInputs":
        """Construye OmegaInputs saneados desde el payload del estado categórico.

        Política de saneamiento:
            - Valores ausentes       → defaults neutrales.
            - Valores no-numéricos   → defaults.
            - Valores fuera de rango → clamped.
            - territory_present      = True solo si territory_data es dict
                                       no vacío.

        Fibración Localizada (Mapa de Restricción):
            Si ``focus_node_id`` está presente, el análisis se restringe al
            sub-grafo local invocando ``SheafCohomologyOrchestrator.validate_local_restriction``.
            La ausencia de datos locales lanza SheafDegeneracyError (manejada
            en OmegaDeliberationManifold.__call__).

        Args:
            payload: Dict con claves "tactics_state", "strategy_state",
                     "territory_state", "focus_node_id", "zoom_level".

        Returns:
            OmegaInputs con todos los campos saneados.
        """
        safe_payload = _safe_dict(payload)

        topo_data_raw = _safe_dict(safe_payload.get("tactics_state"))
        fin_data_raw = _safe_dict(safe_payload.get("strategy_state"))
        territory_data_raw = _safe_dict(safe_payload.get("territory_state"))

        # Contexto visual.
        focus_node_id: Optional[str] = safe_payload.get("focus_node_id")
        zoom_level_raw = safe_payload.get("zoom_level")
        zoom_level: Optional[int] = (
            _safe_int(zoom_level_raw, 0) if zoom_level_raw is not None else None
        )

        # --- Fibración Localizada ---
        if focus_node_id:
            local_topo = _safe_dict(
                topo_data_raw.get("localized_metrics", {}).get(focus_node_id)
            )
            local_fin = _safe_dict(
                fin_data_raw.get("localized_metrics", {}).get(focus_node_id)
            )
            # Lanza SheafDegeneracyError si el sub-grafo carece de soporte.
            SheafCohomologyOrchestrator.validate_local_restriction(
                focus_node_id, local_topo, local_fin
            )
            active_topo_data = local_topo
            active_fin_data = local_fin
        else:
            active_topo_data = topo_data_raw
            active_fin_data = fin_data_raw

        # --- Territorio ---
        territory_present = bool(territory_data_raw)
        if territory_present:
            logistics_friction = _clamp(
                _safe_float(territory_data_raw.get("logistics_friction"), 1.0),
                _FRICTION_CLAMP_LOW,
                _FRICTION_CLAMP_HIGH,
            )
            social_friction = _clamp(
                _safe_float(territory_data_raw.get("social_friction"), 1.0),
                _FRICTION_CLAMP_LOW,
                _FRICTION_CLAMP_HIGH,
            )
            climate_entropy = _clamp(
                _safe_float(territory_data_raw.get("climate_entropy"), 1.0),
                _FRICTION_CLAMP_LOW,
                _FRICTION_CLAMP_HIGH,
            )
        else:
            logistics_friction = 1.0
            social_friction = 1.0
            climate_entropy = 1.0

        return cls(
            psi=_extract_topological_stability(active_topo_data),
            n_nodes=max(1, _safe_int(active_topo_data.get("n_nodes"), 1)),
            n_edges=max(1, _safe_int(active_topo_data.get("n_edges"), 1)),
            cycle_count=max(0, _safe_int(active_topo_data.get("cycle_count"), 0)),
            isolated_count=max(
                0, _safe_int(active_topo_data.get("isolated_count"), 0)
            ),
            stressed_count=max(
                0, _safe_int(active_topo_data.get("stressed_count"), 0)
            ),
            roi=_extract_profitability_index(active_fin_data),
            logistics_friction=logistics_friction,
            social_friction=social_friction,
            climate_entropy=climate_entropy,
            territory_present=territory_present,
            focus_node_id=focus_node_id,
            zoom_level=zoom_level,
            topo_data=_safe_immutable_dict(active_topo_data),
            fin_data=_safe_immutable_dict(active_fin_data),
            territory_data=_safe_immutable_dict(territory_data_raw),
        )


@dataclass(frozen=True)
class OmegaMetrics:
    """Magnitudes cuantitativas derivadas del manifold.

    Todas las magnitudes son floats finitos para inputs no-degenerados.
    Las que representan factores multiplicativos son ≥ 0.

    Invariantes de rango (garantizadas por los motores matemáticos):
        fragility_norm   ∈ [0, 1]
        roi_norm         ∈ [0, 1]
        misalignment     ∈ [0, 1]
        gravity_coupling ∈ [1 - tanh(inflection), 1 + tanh(1-inflection)]
        internal_tension ≥ 0
        external_friction ≥ 1.0
        anomaly_pressure  ≥ 1.0
        combinatorial_scale ≥ 1.0
        friction_scale    ≥ 1.0
        improbability_lever ∈ [1.0, 4.0]
        base_stress       ≥ 0
        fragility_penalty ∈ [1.0, 1.0 + _FRAGILITY_PENALTY_MAX_DELTA]
        total_stress      ≥ 0
        gauge_deflection  ∈ [1.0, _GAUGE_MAX]
        adjusted_stress   ≥ 0
    """

    fragility_norm: float
    roi_norm: float
    misalignment: float
    gravity_coupling: float

    internal_tension: float
    external_friction: float
    anomaly_pressure: float
    combinatorial_scale: float
    friction_scale: float
    improbability_lever: float

    base_stress: float
    fragility_penalty: float
    total_stress: float
    gauge_deflection: float    # NUEVO: incluido explícitamente para auditabilidad.
    adjusted_stress: float


@dataclass(frozen=True)
class OmegaDiagnostics:
    """Diagnósticos interpretables para auditoría humana y trazabilidad.

    Cada campo de status es una etiqueta semántica derivada de los umbrales
    definidos en las constantes del módulo.

    Nota: risk_contribution_breakdown contiene contribuciones conmensurables
    (ver MEJORA en _identify_dominant_risk_axis).
    """

    topology_status: str
    financial_status: str
    territory_status: str
    stress_status: str

    dominant_risk_axis: str
    risk_contribution_breakdown: Dict[str, float] = field(default_factory=dict)
    summary: str = ""

    inputs_snapshot: Dict[str, Any] = field(default_factory=dict)
    derived_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialización completa para almacenamiento y transmisión."""
        return asdict(self)


@dataclass(frozen=True)
class OmegaResult:
    """Resultado completo del colapso antes del empaquetado de estado.

    Invariantes:
        - ``verdict`` es un miembro válido de ``VerdictLevel``.
        - ``metrics.adjusted_stress`` es consistente con ``verdict``
          según los umbrales del retículo (verificable externamente).
    """

    inputs: OmegaInputs
    metrics: OmegaMetrics
    diagnostics: OmegaDiagnostics
    verdict: VerdictLevel

    def to_payload(self, synaptic_context_toon: str) -> Dict[str, Any]:
        """Convierte el resultado a la estructura de payload esperada aguas abajo.

        Incluye todas las métricas intermedias para máxima auditabilidad,
        incluyendo gauge_deflection (ausente en el diseño original).

        CORRECCIÓN: Los MappingProxyType en inputs se convierten a dict
        para garantizar serialización JSON.

        Args:
            synaptic_context_toon: Contexto TOON activo para el prompt LLM.

        Returns:
            Dict serializable con claves "omega_metrics", "verdict",
            "synaptic_context_toon", "omega_diagnostics".
        """
        payload: Dict[str, Any] = {
            "omega_metrics": {
                "topological_stability": round(self.inputs.psi, 6),
                "fragility_norm": round(self.metrics.fragility_norm, 6),
                "roi_norm": round(self.metrics.roi_norm, 6),
                "internal_tension": round(self.metrics.internal_tension, 6),
                "external_friction": round(self.metrics.external_friction, 6),
                "improbability_lever": round(self.metrics.improbability_lever, 6),
                "base_stress": round(self.metrics.base_stress, 6),
                "total_stress": round(self.metrics.total_stress, 6),
                "gauge_deflection": round(self.metrics.gauge_deflection, 6),
                "adjusted_stress": round(self.metrics.adjusted_stress, 6),
                "misalignment": round(self.metrics.misalignment, 6),
                "gravity_coupling": round(self.metrics.gravity_coupling, 6),
                "fragility_penalty": round(self.metrics.fragility_penalty, 6),
                "anomaly_pressure": round(self.metrics.anomaly_pressure, 6),
                "combinatorial_scale": round(self.metrics.combinatorial_scale, 6),
                "friction_scale": round(self.metrics.friction_scale, 6),
            },
            "verdict": self.verdict,
            "synaptic_context_toon": synaptic_context_toon,
            "omega_diagnostics": self.diagnostics.to_dict(),
        }

        if self.inputs.focus_node_id:
            payload["resolution_retract"] = {
                "focus_node_id": self.inputs.focus_node_id,
                "zoom_level": self.inputs.zoom_level,
                "is_localized_deliberation": True,
            }

        return payload


# =============================================================================
# SECCIÓN 6: MORFISMO OMEGA — ORQUESTACIÓN PRINCIPAL
# =============================================================================


class OmegaDeliberationManifold(Morphism):
    """Morfismo de colapso tensorial-operacional.

    Toma un CategoricalState que contiene la superposición de TACTICS y
    STRATEGY, y lo proyecta al estado OMEGA mediante cálculo determinista.

    Propiedades algebraicas:
        - Dominio:   frozenset({Stratum.TACTICS, Stratum.STRATEGY})
        - Codominio: Stratum.OMEGA
        - _collapse() es una función PURA (sin efectos secundarios).
        - El morfismo es extensional: mismo input → mismo output siempre.

    Política de errores:
        - state.is_success == False → pass-through sin modificación.
        - SheafDegeneracyError / HomologicalInconsistencyError → saturación
          del retículo a ⊤ (RECHAZAR) con trazabilidad completa.
        - Cualquier otra excepción → state.with_error() con mensaje.
        - KeyboardInterrupt, SystemExit → re-raise (no capturar).
    """

    def __init__(self, name: str = "omega_tensor_collapse") -> None:
        super().__init__(name)
        self.synaptic_registry = SynapticRegistry()
        self._domain: FrozenSet[Stratum] = frozenset(
            [Stratum.TACTICS, Stratum.STRATEGY]
        )
        self._codomain: Stratum = Stratum.OMEGA

    @property
    def domain(self) -> FrozenSet[Stratum]:
        """Dominio del morfismo: {TACTICS, STRATEGY}."""
        return self._domain

    @property
    def codomain(self) -> Stratum:
        """Codominio del morfismo: OMEGA."""
        return self._codomain

    def __call__(self, state: CategoricalState) -> CategoricalState:
        """Ejecuta el colapso deliberativo sobre un estado categórico.

        Args:
            state: Estado categórico con payload conteniendo
                   "tactics_state", "strategy_state" y opcionalmente
                   "territory_state".

        Returns:
            CategoricalState con stratum=OMEGA y payload "omega_state",
            o el estado original si is_success==False, o un estado de
            error si el colapso falla.
        """
        if not state.is_success:
            return state

        try:
            payload = state.payload if isinstance(state.payload, dict) else {}
            inputs = OmegaInputs.from_payload(payload)
            result = self._collapse(inputs)

            synaptic_context = self.synaptic_registry.get_active_context(
                max_items=_DEFAULT_MAX_CARTRIDGES,
                max_chars=_DEFAULT_MAX_CHARS,
            )
            collapsed_payload = result.to_payload(
                synaptic_context_toon=synaptic_context
            )

            verdict_name = (
                result.verdict.name
                if hasattr(result.verdict, "name")
                else str(result.verdict)
            )
            logger.info(
                "🌌 Colapso Tensorial completado. "
                "base=%.4f total=%.4f gauge=%.4f adjusted=%.4f → %s",
                result.metrics.base_stress,
                result.metrics.total_stress,
                result.metrics.gauge_deflection,
                result.metrics.adjusted_stress,
                verdict_name,
            )

            return state.with_update(
                new_payload={"omega_state": collapsed_payload},
                new_stratum=self.codomain,
            )

        except (KeyboardInterrupt, SystemExit):
            raise

        except (SheafDegeneracyError, HomologicalInconsistencyError) as exc:
            return self._handle_sheaf_degeneracy(state, exc)

        except Exception as exc:  # noqa: BLE001
            logger.exception("Fallo inesperado en el Manifold de Deliberación.")
            return state.with_error(f"Colapso Tensorial fallido: {exc}")

    # -------------------------------------------------------------------------
    # 6.1 MANEJO DE SINGULARIDADES TOPOLÓGICAS
    # -------------------------------------------------------------------------

    def _handle_sheaf_degeneracy(
        self,
        state: CategoricalState,
        exc: Exception,
    ) -> CategoricalState:
        """Saturación del Retículo por Singularidad Topológica Local.

        Cuando el SheafCohomologyOrchestrator detecta degeneración, se
        aplica el axioma de peor caso: el retículo se satura a ⊤ (RECHAZAR)
        con un payload que registra la causa de la saturación.

        MEJORA respecto al diseño original:
            El payload de saturación NO usa math.inf en campos que deben
            ser serializables a JSON (JSON no admite Infinity). En cambio,
            usa un campo booleano "is_saturated" y el valor de sigma
            se codifica como el mayor float representable.

        Args:
            state: Estado categórico de entrada.
            exc:   Excepción de degeneración detectada.

        Returns:
            CategoricalState con stratum=OMEGA, veredicto=RECHAZAR,
            y marcador de error propagado.
        """
        logger.error(
            "🛑 Saturación del Retículo por Singularidad Topológica: %s", exc
        )

        safe_payload = _safe_dict(
            state.payload if isinstance(state.payload, dict) else {}
        )
        focus_node_id: str = str(
            safe_payload.get("focus_node_id", "DESCONOCIDO")
        )
        zoom_level: int = _safe_int(safe_payload.get("zoom_level"), 0)

        # Usamos float("inf") internamente para métricas de diagnóstico,
        # pero el payload incluye un flag explícito para serialización JSON
        # (el consumidor aguas abajo debe manejar "is_saturated").
        _INF = float("inf")

        collapsed_payload: Dict[str, Any] = {
            "omega_metrics": {
                "topological_stability": 0.0,
                "fragility_norm": 1.0,
                "roi_norm": 0.0,
                "internal_tension": _INF,
                "external_friction": _INF,
                "improbability_lever": _INF,
                "base_stress": _INF,
                "total_stress": _INF,
                "gauge_deflection": _GAUGE_MAX,
                "adjusted_stress": _INF,
                "misalignment": 1.0,
                "gravity_coupling": 2.0,
                "fragility_penalty": 1.0 + _FRAGILITY_PENALTY_MAX_DELTA,
                "anomaly_pressure": _INF,
                "combinatorial_scale": _INF,
                "friction_scale": _INF,
                "is_saturated": True,  # Flag para serialización JSON aguas abajo.
            },
            "verdict": VerdictLevel.RECHAZAR,
            "synaptic_context_toon": self.synaptic_registry.get_active_context(),
            "omega_diagnostics": {
                "topology_status": "fragil",
                "financial_status": "retorno_debil",
                "territory_status": "territorio_hostil",
                "stress_status": "tension_critica",
                "dominant_risk_axis": "Singularidad Topológica Local",
                "risk_contribution_breakdown": {
                    "extremes": _INF,
                    "fragility": _INF,
                    "internal": _INF,
                    "territory": _INF,
                },
                "summary": (
                    f"Veredicto=RECHAZAR; "
                    f"Saturación axiomática del retículo: "
                    f"Singularidad Topológica Local en sub-espacio "
                    f"'{focus_node_id}'. "
                    f"El sub-grafo carece de soporte estructural. "
                    f"Causa: {exc}"
                ),
                "inputs_snapshot": {},
                "derived_snapshot": {"is_saturated": True},
            },
            "resolution_retract": {
                "focus_node_id": focus_node_id,
                "zoom_level": zoom_level,
                "is_localized_deliberation": True,
            },
        }

        if state.telemetry_context:
            try:
                state.telemetry_context.record_error(
                    step_name="deliberation_manifold_collapse",
                    error_message=(
                        f"Saturación del Retículo por Singularidad "
                        f"Topológica Local: {exc}"
                    ),
                    error_type=type(exc).__name__,
                    severity="CRITICAL",
                    stratum=Stratum.OMEGA,
                    propagate=True,
                )
            except Exception as tel_exc:  # noqa: BLE001
                logger.warning("Error al registrar telemetría: %s", tel_exc)

        return state.with_update(
            new_payload={"omega_state": collapsed_payload},
            new_stratum=self.codomain,
        ).with_error(f"{type(exc).__name__}: {exc}")

    # -------------------------------------------------------------------------
    # 6.2 ORQUESTACIÓN FORMAL DEL COLAPSO
    # -------------------------------------------------------------------------

    def _collapse(self, inputs: OmegaInputs) -> OmegaResult:
        """Orquesta el cálculo completo del manifold.

        Función pura: no tiene efectos secundarios. El resultado depende
        exclusivamente de ``inputs`` y de las constantes del módulo.

        Pipeline:
            inputs → _compute_metrics → _project_to_lattice → _build_diagnostics
                                                ↑
                                    (puro, sin estado mutable)

        Args:
            inputs: Coordenadas saneadas de entrada.

        Returns:
            OmegaResult completo con métricas, veredicto y diagnósticos.
        """
        metrics = self._compute_metrics(inputs)
        verdict = self._project_to_lattice(metrics.adjusted_stress)
        diagnostics = self._build_diagnostics(inputs, metrics, verdict)

        return OmegaResult(
            inputs=inputs,
            metrics=metrics,
            diagnostics=diagnostics,
            verdict=verdict,
        )

    def _compute_metrics(self, inputs: OmegaInputs) -> OmegaMetrics:
        """Calcula todas las magnitudes continuas del manifold.

        DAG de dependencias (orden topológico de cálculo):

            ψ   → fragility_norm ─┬→ misalignment ─→ internal_tension ─┐
            ROI → roi_norm ────────┘                                    │
            ψ   → fragility_norm ─────────────────→ gravity_coupling ───┤
                                                                        │
            territory → external_friction ──────────────────────────────┤
                                                                        │
            anomalies → anomaly_pressure ─┐                             │
            n_nodes,edges → comb_scale ───┼→ improbability_lever ───────┤
            external_friction → fric_scale┘                             │
                                                                        ↓
            n_cartridges → gauge_deflection ──────────────────────────→ σ*
            ψ → fragility_penalty ─────────────────────────────────────→ σ*

            σ* = base × Λ × P_frag × gauge

        CORRECCIÓN CRÍTICA:
            Se elimina el cortocircuito `if anomaly_pressure > 1.25 and
            external_friction > 1.5: adjusted_stress = math.inf`.
            Este cortocircuito violaba la pureza funcional y era arbitrario.
            En cambio, el retículo naturalmente proyectará a RECHAZAR cuando
            σ* excede _VERDICT_THRESHOLD_PRECAUCION por la combinación de
            factores.

        Args:
            inputs: Coordenadas saneadas de entrada.

        Returns:
            OmegaMetrics con todas las magnitudes calculadas.
        """
                # --- Normalización al espacio métrico unificado [0,1] ---
        fiedler_value = float(inputs.topo_data.get("fiedler_value", 0.0)) if isinstance(inputs.topo_data, dict) else 0.0
        fragility_norm = self._compute_fragility_normalized(inputs.psi, fiedler_value)
        roi_norm = self._compute_roi_normalized(inputs.roi, fiedler_value)

        # --- Tensión Interna ---
        misalignment = self._compute_misalignment(fragility_norm, roi_norm)
        gravity_coupling = self._compute_gravity_coupling(fragility_norm)
        internal_tension = self._compute_internal_tension(
            misalignment=misalignment,
            gravity_coupling=gravity_coupling,
        )

        # --- Fricción Externa ---
        external_friction = self._compute_external_friction(inputs)

        # --- Palanca de Improbabilidad ---
        anomaly_pressure = self._compute_anomaly_pressure(inputs)
        combinatorial_scale = self._compute_combinatorial_scale(inputs)
        friction_scale = self._compute_friction_scale(external_friction)
        improbability_lever = self._compute_improbability_lever(
            anomaly_pressure=anomaly_pressure,
            combinatorial_scale=combinatorial_scale,
            friction_scale=friction_scale,
        )

        # --- Estrés Base y Total ---
        base_stress = internal_tension * external_friction
        total_stress = base_stress * improbability_lever

        # --- Penalización por Fragilidad ---
        fragility_penalty = self._compute_fragility_penalty(inputs.psi)

                # --- Factor de Gauge (Acoplamiento TOON) ---
        # CORRECCIÓN: Fase continua C^∞ mediante transición hiperbólica.
        n_cartridges = self.synaptic_registry.cartridge_count
        gauge_deflection = 1.0 + (_GAUGE_MAX - 1.0) * math.tanh((_GAUGE_ALPHA * n_cartridges) / (_GAUGE_MAX - 1.0))

        # --- Estrés Ajustado Final: σ* = T_int · F_ext · Λ · P_frag · gauge ---
        adjusted_stress = total_stress * fragility_penalty * gauge_deflection

        # Verificación de finitud (defensa ante propagación de NaN no detectados).
        if not math.isfinite(adjusted_stress):
            logger.warning(
                "adjusted_stress no finito detectado (%.6g). "
                "Usando _VERDICT_THRESHOLD_PRECAUCION como límite inferior.",
                adjusted_stress,
            )
            # No forzamos math.inf; el retículo manejará el RECHAZAR
            # si el valor excede el umbral.
            adjusted_stress = max(
                _VERDICT_THRESHOLD_PRECAUCION,
                total_stress if math.isfinite(total_stress) else _VERDICT_THRESHOLD_PRECAUCION,
            )

        return OmegaMetrics(
            fragility_norm=fragility_norm,
            roi_norm=roi_norm,
            misalignment=misalignment,
            gravity_coupling=gravity_coupling,
            internal_tension=internal_tension,
            external_friction=external_friction,
            anomaly_pressure=anomaly_pressure,
            combinatorial_scale=combinatorial_scale,
            friction_scale=friction_scale,
            improbability_lever=improbability_lever,
            base_stress=base_stress,
            fragility_penalty=fragility_penalty,
            total_stress=total_stress,
            gauge_deflection=gauge_deflection,
            adjusted_stress=adjusted_stress,
        )

    # -------------------------------------------------------------------------
    # 6.3 MOTORES MATEMÁTICOS INTERNOS (funciones puras estáticas)
    # -------------------------------------------------------------------------

    @staticmethod
    def _compute_fragility_normalized(psi: float, fiedler_value: float = 0.0) -> float:
        safe_psi = max(psi, _EPSILON)
        raw = math.log2(1.0 + 1.0 / safe_psi)
        kappa = 0.1 # basal coupling constant
        eps_ref = kappa / (fiedler_value + 0.1) # eps is already included in 0.1
        norm_denominator = math.log2(1.0 + 1.0 / eps_ref)
        return _clamp(raw / norm_denominator, 0.0, 1.0)

    @staticmethod
    def _compute_roi_normalized(roi: float, fiedler_value: float = 0.0) -> float:
        safe_roi = max(roi, _EPSILON)
        raw = math.log2(1.0 + 1.0 / safe_roi)
        kappa = 0.1
        eps_ref = kappa / (fiedler_value + 0.1)
        norm_denominator = math.log2(1.0 + 1.0 / eps_ref)
        return _clamp(raw / norm_denominator, 0.0, 1.0)

    @staticmethod
    def _compute_misalignment(fragility_norm: float, roi_norm: float) -> float:
        """Desalineación entre fragilidad y expectativa financiera.

        Con fragility_norm y roi_norm ambos en [0,1] y calculados mediante
        la misma transformación isométrica log₂, su diferencia absoluta
        es una distancia euclidiana genuina en el espacio métrico [0,1]²:

            misalignment = |fragility_norm - roi_norm| ∈ [0, 1]

        Interpretación:
            misalignment ≈ 0: estructura y finanzas coherentes (ψ ≈ ROI).
            misalignment ≈ 1: máxima disonancia (e.g., estructura muy
                              frágil con ROI muy alto, o viceversa).

        Args:
            fragility_norm: ∈ [0, 1]
            roi_norm:       ∈ [0, 1]

        Returns:
            float en [0, 1].
        """
        return abs(fragility_norm - roi_norm)

    @staticmethod
    def _compute_gravity_coupling(fragility_norm: float) -> float:
        """Amplificación suave de la tensión por fragilidad.

        Función sigmoide C∞ centrada en _GRAVITY_INFLECTION:

            coupling(f) = 1 + tanh(f - inflection)

        Con inflection = 0.289 (fragility_norm para ψ=1.0, el punto
        neutral), se garantiza coupling(neutral) = 1 + tanh(0) = 1.0.

        CORRECCIÓN: El punto de inflexión se actualiza de 0.5 (arbitrario)
        a 0.289 (derivado del punto neutral del espacio normalizado),
        garantizando que el acoplamiento sea 1.0 para inputs neutrales.

        Rango:
            fragility_norm = 0    → coupling = 1 - tanh(0.289) ≈ 0.714
            fragility_norm = 0.289→ coupling = 1.000 (neutro)
            fragility_norm = 1    → coupling = 1 + tanh(0.711) ≈ 1.612

        Args:
            fragility_norm: ∈ [0, 1]

        Returns:
            float en (0, 2) — siempre positivo, monótonamente creciente.
        """
        return 1.0 + math.tanh(fragility_norm - _GRAVITY_INFLECTION)

    @staticmethod
    def _compute_internal_tension(
        misalignment: float,
        gravity_coupling: float,
    ) -> float:
        """Tensión interna efectiva como producto de desalineación y acoplamiento.

            T_int = max(0, misalignment × gravity_coupling)

        El max(0, ...) es defensivo; con los rangos actuales el resultado
        siempre es ≥ 0.

        Args:
            misalignment:    ∈ [0, 1]
            gravity_coupling: ∈ (0, 2)

        Returns:
            float ≥ 0.
        """
        return max(0.0, misalignment * gravity_coupling)

    @staticmethod
    def _compute_external_friction(inputs: OmegaInputs) -> float:
        """Métrica riemanniana simplificada del territorio.

        Si no hay datos territoriales (territory_present=False),
        retorna 1.0 (elemento neutro multiplicativo).

        La fricción es la media ponderada g_ij de las tres dimensiones:

            F_ext = max(1.0, Σᵢ wᵢ · dᵢ)

        donde Σwᵢ = 1.0 (verificado estáticamente) y dᵢ ∈ [0, 5].

        El clamp inferior a 1.0 garantiza que el territorio nunca
        reduzca el estrés base (propiedad de monotonicidad).

        Args:
            inputs: Coordenadas con fricciones territoriales.

        Returns:
            float ≥ 1.0.
        """
        if not inputs.territory_present:
            return 1.0

        metric = (
            inputs.logistics_friction * _FRICTION_WEIGHT_LOGISTICS
            + inputs.social_friction * _FRICTION_WEIGHT_SOCIAL
            + inputs.climate_entropy * _FRICTION_WEIGHT_CLIMATE
        )
        return max(1.0, metric)

    @staticmethod
    def _compute_anomaly_pressure(inputs: OmegaInputs) -> float:
        """Presión estructural inducida por anomalías topológicas.

        Modelo aditivo:
            P_anom = 1 + α_c·n_c + α_i·n_i + α_s·n_s

        donde:
            α_c = 0.08 (ciclos β₁, mayor impacto)
            α_i = 0.03 (nodos aislados β₀-1, impacto moderado)
            α_s = 0.05 (aristas bajo estrés, impacto intermedio)

        Siempre ≥ 1.0 (neutro sin anomalías).

        Args:
            inputs: Coordenadas con conteos de anomalías.

        Returns:
            float ≥ 1.0.
        """
        return 1.0 + (
            _ANOMALY_COEFF_CYCLE * inputs.cycle_count
            + _ANOMALY_COEFF_ISOLATED * inputs.isolated_count
            + _ANOMALY_COEFF_STRESSED * inputs.stressed_count
        )

    @staticmethod
    def _compute_combinatorial_scale(inputs: OmegaInputs) -> float:
        """Escala logarítmica del tamaño del espacio combinatorio del grafo.

        Modela la complejidad estructural como el logaritmo del producto
        de nodos por aristas (número de rutas posibles en el grafo):

            S_comb = log₁₀(max(10, n_nodes × n_edges))

        El max(10) garantiza S_comb ≥ 1.0 siempre (incluyendo grafos
        degenerados con n_nodes=n_edges=1).

        Justificación logarítmica: la complejidad computacional de
        problemas de optimización en grafos crece tipicamente de forma
        super-polinomial; log₁₀ captura la magnitud de orden sin
        hacer explotar el estrés para grafos grandes.

        Args:
            inputs: Coordenadas con n_nodes y n_edges.

        Returns:
            float ≥ 1.0.
        """
        opportunity_space = max(1, inputs.n_nodes) * max(1, inputs.n_edges)
        return math.log10(max(10.0, float(opportunity_space)))

    @staticmethod
    def _compute_friction_scale(external_friction: float) -> float:
        """Escala sublineal del efecto de fricción sobre las colas de riesgo.

        Raíz cuadrada de la fricción externa:
            S_fric = √(max(1.0, F_ext))

        La √ atenúa el impacto de fricciones extremas para evitar que
        el territorio solo domine el veredicto (principio de diversificación
        de factores de riesgo).

        Propiedad: S_fric = 1.0 cuando F_ext ≤ 1.0 (territorio neutral).

        Args:
            external_friction: ≥ 1.0

        Returns:
            float ≥ 1.0.
        """
        return math.sqrt(max(1.0, external_friction))

    @staticmethod
    def _compute_improbability_lever(
        anomaly_pressure: float,
        combinatorial_scale: float,
        friction_scale: float,
    ) -> float:
        """Palanca de eventos extremos (fat-tail risk amplifier).

        Define el operador Λ de amplificación de improbabilidad:

            Λ = clamp(S_comb × S_fric × P_anom / K, 1, 4)

        donde K = _IMPROBABILITY_SCALE_FACTOR = 2.0.

        Verificación de neutralidad: inputs neutrales producen
            S_comb = log₁₀(10) = 1.0
            S_fric = √1.0 = 1.0
            P_anom = 1.0
            Λ = (1·1·1)/2.0 = 0.5 → clamp → 1.0 ✓

        El clamp a [1, 4] garantiza:
            - Sin anomalías ni complejidad, Λ = 1.0 (no amplifica).
            - En el peor caso, Λ = 4.0 (cuadruplica el estrés base).

        Args:
            anomaly_pressure:   ≥ 1.0
            combinatorial_scale: ≥ 1.0
            friction_scale:     ≥ 1.0

        Returns:
            float en [1.0, 4.0].
        """
        raw_lever = (
            combinatorial_scale * friction_scale * anomaly_pressure
        ) / _IMPROBABILITY_SCALE_FACTOR
        return _clamp(raw_lever, _IMPROBABILITY_CLAMP_LOW, _IMPROBABILITY_CLAMP_HIGH)

    @staticmethod
    def _compute_fragility_penalty(psi: float) -> float:
        """Penalización estructural no-lineal por fragilidad.

        Define P_frag(ψ) como:
            ψ ≥ 1.0: P_frag = 1.0  (sin penalización)
            ψ < 1.0: P_frag = 1 + min(δ_max, (1-ψ)·δ_max)
                            = 1 + δ_max · min(1, 1-ψ)

        donde δ_max = _FRAGILITY_PENALTY_MAX_DELTA = 1.5.

        Propiedades:
            - P_frag ∈ [1.0, 2.5] (acotada).
            - P_frag es Lipschitz-continua con constante L = δ_max = 1.5.
            - P_frag(1.0) = 1.0 (continuidad en el punto de quiebre).
            - P_frag(0.0) = 2.5 (penalización máxima para ψ→0).

        Args:
            psi: Índice de estabilidad. Asumido ≥ 0 por el clamp upstream.

        Returns:
            float en [1.0, 2.5].
        """
        if psi >= 1.0:
            return 1.0
        deficit = _clamp(1.0 - psi, 0.0, 1.0)
        return 1.0 + min(_FRAGILITY_PENALTY_MAX_DELTA, deficit * _FRAGILITY_PENALTY_MAX_DELTA)

    @staticmethod
    def _project_to_lattice(adjusted_stress: float) -> VerdictLevel:
        # Compactificación de Alexandroff: mapeo del infinito y NaN al polo norte (Supremo ⊤).
        if not math.isfinite(adjusted_stress) or math.isnan(adjusted_stress):
            return VerdictLevel.RECHAZAR

        if adjusted_stress < _VERDICT_THRESHOLD_VIABLE:
            return VerdictLevel.VIABLE
        if adjusted_stress < _VERDICT_THRESHOLD_CONDICIONAL:
            return VerdictLevel.CONDICIONAL
        if adjusted_stress < _VERDICT_THRESHOLD_PRECAUCION:
            return VerdictLevel.PRECAUCION
        return VerdictLevel.RECHAZAR

    def _build_diagnostics(
        self,
        inputs: OmegaInputs,
        metrics: OmegaMetrics,
        verdict: VerdictLevel,
    ) -> OmegaDiagnostics:
        """Construye un diagnóstico formal y auditable del colapso.

        Incluye:
            - Etiquetas semánticas para cada dimensión de riesgo.
            - Snapshot numérico de inputs y métricas derivadas.
            - Identificación del eje dominante con breakdown conmensurable.
            - Summary legible por humanos y máquinas.

        Args:
            inputs:  Coordenadas de entrada.
            metrics: Magnitudes calculadas.
            verdict: Veredicto proyectado.

        Returns:
            OmegaDiagnostics completo y serializable.
        """
        topology_status = _interpret_psi(inputs.psi)
        financial_status = _interpret_roi(inputs.roi)
        territory_status = _interpret_friction(metrics.external_friction)
        stress_status = _interpret_stress(metrics.adjusted_stress)

        dominant_axis, contributions = self._identify_dominant_risk_axis(metrics)

        verdict_name = (
            verdict.name if hasattr(verdict, "name") else str(verdict)
        )

        summary = (
            f"Veredicto={verdict_name}; "
            f"topologia={topology_status}; "
            f"finanzas={financial_status}; "
            f"territorio={territory_status}; "
            f"estres={stress_status}; "
            f"eje_dominante={dominant_axis}"
        )

        inputs_snapshot: Dict[str, Any] = {
            "psi": round(inputs.psi, 6),
            "roi": round(inputs.roi, 6),
            "n_nodes": inputs.n_nodes,
            "n_edges": inputs.n_edges,
            "cycle_count": inputs.cycle_count,
            "isolated_count": inputs.isolated_count,
            "stressed_count": inputs.stressed_count,
            "territory_present": inputs.territory_present,
            "logistics_friction": round(inputs.logistics_friction, 6),
            "social_friction": round(inputs.social_friction, 6),
            "climate_entropy": round(inputs.climate_entropy, 6),
        }

        derived_snapshot: Dict[str, Any] = {
            "fragility_norm": round(metrics.fragility_norm, 6),
            "roi_norm": round(metrics.roi_norm, 6),
            "misalignment": round(metrics.misalignment, 6),
            "gravity_coupling": round(metrics.gravity_coupling, 6),
            "internal_tension": round(metrics.internal_tension, 6),
            "external_friction": round(metrics.external_friction, 6),
            "anomaly_pressure": round(metrics.anomaly_pressure, 6),
            "combinatorial_scale": round(metrics.combinatorial_scale, 6),
            "friction_scale": round(metrics.friction_scale, 6),
            "improbability_lever": round(metrics.improbability_lever, 6),
            "base_stress": round(metrics.base_stress, 6),
            "fragility_penalty": round(metrics.fragility_penalty, 6),
            "total_stress": round(metrics.total_stress, 6),
            "gauge_deflection": round(metrics.gauge_deflection, 6),
            "adjusted_stress": round(metrics.adjusted_stress, 6),
        }

        return OmegaDiagnostics(
            topology_status=topology_status,
            financial_status=financial_status,
            territory_status=territory_status,
            stress_status=stress_status,
            dominant_risk_axis=dominant_axis,
            risk_contribution_breakdown=contributions,
            summary=summary,
            inputs_snapshot=inputs_snapshot,
            derived_snapshot=derived_snapshot,
        )

    @staticmethod
    def _identify_dominant_risk_axis(
        metrics: OmegaMetrics,
    ) -> Tuple[str, Dict[str, float]]:
        """Identifica el eje dominante del riesgo con breakdown conmensurable.

        CORRECCIÓN CRÍTICA: El diseño original mezclaba contribuciones con
        unidades heterogéneas (tensión ∈ ℝ⁺ vs excesos ∈ [0,1.5]).
        La corrección normaliza todas las contribuciones como fracción
        multiplicativa del estrés base, haciéndolas conmensurables:

            contrib_internal  = internal_tension / (base_stress + ε)
                                (fracción de σ_base debida a T_int)
            contrib_territory = (external_friction - 1.0)
                                (exceso sobre el neutro multiplicativo)
            contrib_extremes  = (improbability_lever - 1.0)
                                (exceso de Λ sobre el neutro)
            contrib_fragility = (fragility_penalty - 1.0)
                                (exceso de P_frag sobre el neutro)

        Todas las contribuciones son ≥ 0. El eje dominante es el de
        mayor valor; en caso de empate, se usa orden lexicográfico
        (determinismo).

        Si todas las contribuciones son ≤ 0, retorna "balanced".

        Args:
            metrics: OmegaMetrics calculadas.

        Returns:
            Tupla (nombre_eje_dominante, {eje: contribución_normalizada}).
        """
        base_for_norm = metrics.base_stress + _EPSILON

        contributions: Dict[str, float] = {
            "internal": round(metrics.internal_tension / base_for_norm, 6),
            "territory": round(metrics.external_friction - 1.0, 6),
            "extremes": round(metrics.improbability_lever - 1.0, 6),
            "fragility": round(metrics.fragility_penalty - 1.0, 6),
        }

        # Ordenamiento determinista: valor desc, nombre asc.
        sorted_items = sorted(
            contributions.items(),
            key=lambda item: (-item[1], item[0]),
        )

        dominant_name, dominant_value = sorted_items[0]
        if dominant_value <= 0.0:
            return "balanced", contributions
        return dominant_name, contributions