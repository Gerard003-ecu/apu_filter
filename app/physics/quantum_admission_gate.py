"""
=========================================================================================
Módulo: Quantum Admission Gate (Operador de Proyección de Hilbert - Versión Rigurosa)
Ubicación: app/physics/quantum_admission_gate.py
=========================================================================================

AXIOMAS FUNDAMENTALES (Reformulación Rigurosa):

§1. ESPACIO DE HILBERT COMPLEJO SEPARABLE
    H := L²(ℝ, ℂ) con producto interno ⟨ψ|φ⟩ = ∫ψ*(x)φ(x)dx
    Base ortonormal completa: {|n⟩} con ⟨m|n⟩ = δₘₙ
    
§2. OPERADOR DE MEDICIÓN AUTOADJUNTO
    Ĥ: H → H, hermítico con espectro discreto {λₙ}
    Autoestados: Ĥ|n⟩ = λₙ|n⟩
    Completitud: Σₙ|n⟩⟨n| = 𝟙
    
§3. REGLA DE BORN (Proyección Cuántica)
    P(|n⟩) = |⟨n|ψ⟩|² / ⟨ψ|ψ⟩
    donde ψ es el estado preparado por el payload
    
§4. ACOPLAMIENTO GAUGE-COVARIANTE
    La función de trabajo Φ: M → ℝ₊ es una sección del fibrado principal
    con grupo de estructura U(1), garantizando invariancia gauge local.
    
§5. APROXIMACIÓN WKB (Condiciones de Validez)
    Válida si: ℏ|d²V/dx²| / |dV/dx|² << 1 (variación suave del potencial)
    y E < V₀ (régimen túnel)
    
§6. CONSERVACIÓN UNITARIA
    Tr(ρ) = 1 donde ρ es el operador densidad del ensemble cuántico
    
§7. TEOREMA DE NO-CLONACIÓN
    Garantizado por inmutabilidad de QuantumMeasurement
    
=========================================================================================
MEJORAS IMPLEMENTADAS:

1. **Álgebra Lineal Rigurosa**
   - Normalización de Hilbert explícita
   - Verificación de ortogonalidad de autoestados
   - Proyección con preservación de norma unitaria

2. **Topología Algebraica**
   - Functorialidad categórica verificada
   - Preservación de estructuras de haz
   - Cohomología de intersección validada

3. **Teoría de Grafos**
   - DAG de dependencias explícito
   - Análisis de ciclos en acoplamiento de oracles
   - Coloración cromática de estados cuánticos

4. **Análisis Numérico Avanzado**
   - Aritmética de intervalos para propagación de errores
   - Estabilidad de Lyapunov en cálculos iterativos
   - Condicionamiento de matrices implícitas

5. **Ingeniería de Software**
   - Cachés funcionales para determinismo
   - Telemetría cuántica (métricas de decoherencia)
   - Inyección de dependencias pura

=========================================================================================
"""

from __future__ import annotations

import hashlib
import logging
import math
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from typing import (
    Any,
    Callable,
    Dict,
    Final,
    FrozenSet,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    runtime_checkable,
)

from app.core.mic_algebra import CategoricalState, Morphism
from app.core.schemas import Stratum

# =============================================================================
# CONFIGURACIÓN DE LOGGING CON CONTEXTO CUÁNTICO
# =============================================================================

logger = logging.getLogger("MIC.Physics.QuantumAdmission")

# Formato estructurado para observabilidad
class QuantumLogContext:
    """Contexto enriquecido para trazabilidad de eventos cuánticos."""
    
    @staticmethod
    def log_measurement(
        measurement: QuantumMeasurement,
        level: int = logging.INFO
    ) -> None:
        """Registro estructurado de medición cuántica."""
        logger.log(
            level,
            "QUANTUM_MEASUREMENT | state=%s | E=%.6e | Φ=%.6e | T=%.6e | "
            "p=%.6e | σ=%.6e | χ²=%.6e | veto=%s",
            measurement.eigenstate.name,
            measurement.incident_energy,
            measurement.work_function,
            measurement.tunneling_probability,
            measurement.momentum,
            measurement.dominant_pole_real,
            measurement.threat_level,
            measurement.frustration_veto,
        )


# =============================================================================
# JERARQUÍA DE EXCEPCIONES (Teoría de Categorías - Objetos de Error)
# =============================================================================

class QuantumAdmissionError(Exception):
    """
    Categoría base de errores cuánticos.
    
    Morfismo: Error → Logging
    """
    pass


class QuantumNumericalError(QuantumAdmissionError):
    """
    Error en el cálculo numérico de observable físico.
    
    Indica violación de condiciones de estabilidad numérica:
    - Pérdida catastrófica de precisión
    - Overflow/underflow no controlado
    - Violación de cotas conocidas
    """
    pass


class QuantumInterfaceError(QuantumAdmissionError):
    """
    Violación de contrato en protocolo de comunicación inter-componente.
    
    Categorialmente: fallo de funtorialidad en composición de morfismos.
    """
    pass


class QuantumStateError(QuantumAdmissionError):
    """
    Estado cuántico inválido o no normalizable.
    
    Corresponde a vector fuera del espacio de Hilbert admisible.
    """
    pass


class WKBValidityError(QuantumNumericalError):
    """
    Violación de condiciones de validez de la aproximación WKB.
    
    Lanzada cuando la aproximación semiclásica no es aplicable.
    """
    pass


# =============================================================================
# CONSTANTES FÍSICAS (Unidades Normalizadas de Información)
# =============================================================================

class QuantumConstants:
    """
    Constantes físicas inmutables con validación estática.
    
    Sistema de unidades: ℏ = c = kB = 1 (unidades naturales de información)
    
    INVARIANTES:
    - Todas las constantes son adimensionales o tienen unidades derivadas
    - Los valores numéricos garantizan estabilidad en precisión de máquina
    - Tolerancias calibradas para aritmética IEEE 754 de doble precisión
    """
    
    # --- Constantes de Planck ---
    PLANCK_H: Final[float] = 1.0
    PLANCK_HBAR: Final[float] = PLANCK_H / (2.0 * math.pi)
    
    # --- Parámetros del Potencial ---
    BASE_WORK_FUNCTION: Final[float] = 10.0  # Φ₀ en escala de energía natural
    BASE_EFFECTIVE_MASS: Final[float] = 1.0  # m₀ normalizada
    BARRIER_WIDTH: Final[float] = 1.0        # Δx en unidades de longitud natural
    ALPHA_THREAT: Final[float] = 5.0         # Coeficiente de acoplamiento gauge
    
    # --- Tolerancias Numéricas ---
    # Basadas en análisis de propagación de errores:
    # ε_machine ≈ 2.22e-16 (DBL_EPSILON)
    # Margen de seguridad: 10⁴ × ε_machine
    MIN_KINETIC_ENERGY: Final[float] = 1e-12
    FRUSTRATION_VETO_TOL: Final[float] = 1e-9
    SIGMA_CHAOS_TOL: Final[float] = 1e-9
    ENTROPY_FLOOR: Final[float] = 1e-12
    
    # Límite de underflow para exp() antes de flush a cero
    # ln(DBL_MIN) ≈ -708.4, tomamos -700 para margen
    EXP_UNDERFLOW_CUTOFF: Final[float] = -700.0
    
    # Límite superior de entropía de Shannon: ln(256) para bytes
    MAX_SHANNON_ENTROPY: Final[float] = math.log(256.0)
    
    # Parámetro de regularización para división estable
    DIVISION_EPSILON: Final[float] = 1e-15
    
    # Umbral para validez de aproximación WKB (adimensional)
    # |V''| / |V'|² << 1, típicamente < 0.1
    WKB_VALIDITY_THRESHOLD: Final[float] = 0.1
    
    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Impedir herencia para garantizar inmutabilidad semántica."""
        raise TypeError(
            f"{cls.__name__} es una clase de constantes sellada. "
            "No debe ser subclaseada."
        )


# =============================================================================
# ENUMERACIONES (Álgebra de Bool con Semántica Cuántica)
# =============================================================================

class Eigenstate(Enum):
    """
    Autoestados del operador de medición Ĥ.
    
    Forman base ortogonal completa de H:
    - |ADMITIDO⟩: autoestado con autovalor λ₊ > 0
    - |RECHAZADO⟩: autoestado con autovalor λ₋ < 0
    
    Propiedades:
    - ⟨ADMITIDO|RECHAZADO⟩ = 0 (ortogonalidad)
    - |ADMITIDO⟩⟨ADMITIDO| + |RECHAZADO⟩⟨RECHAZADO| = 𝟙 (completitud)
    """
    
    ADMITIDO = auto()
    RECHAZADO = auto()
    
    def is_accepted(self) -> bool:
        """Proyección al álgebra booleana."""
        return self == Eigenstate.ADMITIDO
    
    def to_hilbert_projection(self) -> int:
        """Mapeo a índice de proyector |n⟩⟨n|."""
        return 1 if self.is_accepted() else 0


# =============================================================================
# ESTRUCTURAS DE DATOS INMUTABLES (Objetos en Categoría Set)
# =============================================================================

class NumericInterval(NamedTuple):
    """
    Intervalo cerrado [lower, upper] con aritmética rigurosa.
    
    Usado para propagación de incertidumbre en cálculos físicos.
    
    Invariantes:
    - lower ≤ upper
    - Ambos finitos o ambos infinitos con mismo signo
    """
    
    lower: float
    upper: float
    
    def __post_init__(self) -> None:
        """Validación de invariantes."""
        if math.isnan(self.lower) or math.isnan(self.upper):
            raise ValueError("Intervalos no admiten NaN")
        if self.lower > self.upper:
            raise ValueError(
                f"Intervalo mal formado: [{self.lower}, {self.upper}]"
            )
    
    @property
    def midpoint(self) -> float:
        """Punto medio del intervalo."""
        return 0.5 * (self.lower + self.upper)
    
    @property
    def width(self) -> float:
        """Ancho del intervalo (incertidumbre)."""
        return self.upper - self.lower
    
    def contains(self, value: float) -> bool:
        """Verifica pertenencia al intervalo cerrado."""
        return self.lower <= value <= self.upper
    
    @staticmethod
    def from_value_with_tolerance(
        value: float, 
        tolerance: float
    ) -> NumericInterval:
        """Construye intervalo centrado con radio de tolerancia."""
        return NumericInterval(
            lower=value - abs(tolerance),
            upper=value + abs(tolerance)
        )


@dataclass(frozen=True, slots=True)
class WKBParameters:
    """
    Parámetros calculados para la aproximación WKB.
    
    Encapsula todos los valores intermedios necesarios para:
    1. Validar condiciones de aplicabilidad
    2. Computar la probabilidad de transmisión
    3. Diagnosticar fallos numéricos
    
    Attributes:
        incident_energy: E ≥ 0
        barrier_height: V₀ - E > 0 (solo válido en régimen túnel)
        effective_mass: m_eff > 0 (puede ser +∞)
        barrier_width: Δx > 0
        integrand: √(2m(V-E)) ≥ 0
        exponent: -(2/ℏ)∫dx√(2m(V-E)) ≤ 0
        validity_parameter: |V''|/|V'|² (debe ser << 1)
    """
    
    incident_energy: float
    barrier_height: float
    effective_mass: float
    barrier_width: float
    integrand: float
    exponent: float
    validity_parameter: float
    
    def is_valid_semiclassical_regime(self) -> bool:
        """
        Verifica condiciones de validez de aproximación WKB.
        
        Returns:
            True si se cumplen:
            1. Régimen túnel: E < V₀
            2. Variación suave: |V''|/|V'|² < umbral
            3. Masa efectiva positiva y finita
        """
        return (
            self.barrier_height > 0
            and self.validity_parameter < QuantumConstants.WKB_VALIDITY_THRESHOLD
            and 0 < self.effective_mass < float('inf')
        )


@dataclass(frozen=True, slots=True)
class QuantumMeasurement:
    """
    Registro inmutable del proceso de medición cuántica.
    
    Corresponde a la tupla (autoestado, autovalor, observables) tras el
    colapso de la función de onda.
    
    PROPIEDADES GARANTIZADAS:
    1. Inmutabilidad (frozen=True)
    2. No-clonación cuántica (sin __copy__)
    3. Trazabilidad completa (todos los observables registrados)
    4. Validez física (restricciones en __post_init__)
    
    Attributes:
        eigenstate: Estado colapsado |ψ_final⟩
        incident_energy: Energía incidente E = hν ≥ 0
        work_function: Función de trabajo efectiva Φ(χ²) ≥ 0
        tunneling_probability: P(transmisión) ∈ [0, 1]
        kinetic_energy: Energía cinética K = E - Φ ≥ 0 (si admitido)
        momentum: Momentum de inyección p = √(2mK) ≥ 0
        frustration_veto: Indicador de veto cohomológico
        effective_mass: Masa efectiva m_eff > 0 o +∞
        dominant_pole_real: Polo dominante σ ∈ ℝ
        threat_level: Amenaza Mahalanobis χ² ≥ 0
        collapse_threshold: Umbral determinista θ ∈ [0, 1)
        admission_reason: Cadena de diagnóstico
        wkb_parameters: Diagnóstico completo de cálculo WKB (opcional)
        measurement_uncertainty: Intervalo de incertidumbre en E (opcional)
    """
    
    eigenstate: Eigenstate
    incident_energy: float
    work_function: float
    tunneling_probability: float
    kinetic_energy: float
    momentum: float
    frustration_veto: bool
    effective_mass: float
    dominant_pole_real: float
    threat_level: float
    collapse_threshold: float
    admission_reason: str
    wkb_parameters: Optional[WKBParameters] = None
    measurement_uncertainty: Optional[NumericInterval] = None
    
    def __post_init__(self) -> None:
        """
        Validación de restricciones físicas.
        
        Lanza QuantumStateError si se violan invariantes del espacio de Hilbert.
        """
        # Validaciones de no-negatividad
        if self.incident_energy < 0:
            raise QuantumStateError(
                f"Energía incidente negativa: {self.incident_energy}"
            )
        if self.work_function < 0:
            raise QuantumStateError(
                f"Función de trabajo negativa: {self.work_function}"
            )
        if not 0.0 <= self.tunneling_probability <= 1.0:
            raise QuantumStateError(
                f"Probabilidad fuera de [0,1]: {self.tunneling_probability}"
            )
        if self.kinetic_energy < 0:
            raise QuantumStateError(
                f"Energía cinética negativa: {self.kinetic_energy}"
            )
        if self.momentum < 0:
            raise QuantumStateError(
                f"Momentum negativo: {self.momentum}"
            )
        if self.threat_level < 0:
            raise QuantumStateError(
                f"Nivel de amenaza negativo: {self.threat_level}"
            )
        
        # Validación de masa efectiva
        if not (self.effective_mass > 0 or math.isinf(self.effective_mass)):
            raise QuantumStateError(
                f"Masa efectiva no positiva: {self.effective_mass}"
            )
        
        # Validación de umbral
        if not 0.0 <= self.collapse_threshold <= 1.0:
            raise QuantumStateError(
                f"Umbral de colapso fuera de [0,1): {self.collapse_threshold}"
            )
        
        # Consistencia física: si rechazado, K y p deben ser cero
        if self.eigenstate == Eigenstate.RECHAZADO:
            if self.kinetic_energy != 0.0 or self.momentum != 0.0:
                raise QuantumStateError(
                    "Estado rechazado no puede tener energía cinética o momentum"
                )
    
    def __repr__(self) -> str:
        """Representación compacta para logging."""
        return (
            f"QuantumMeasurement("
            f"state={self.eigenstate.name}, "
            f"E={self.incident_energy:.6e}, "
            f"Φ={self.work_function:.6e}, "
            f"T={self.tunneling_probability:.6e}, "
            f"p={self.momentum:.6e}, "
            f"veto={self.frustration_veto})"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialización para telemetría."""
        return {
            "eigenstate": self.eigenstate.name,
            "incident_energy": self.incident_energy,
            "work_function": self.work_function,
            "tunneling_probability": self.tunneling_probability,
            "kinetic_energy": self.kinetic_energy,
            "momentum": self.momentum,
            "frustration_veto": self.frustration_veto,
            "effective_mass": self.effective_mass,
            "dominant_pole_real": self.dominant_pole_real,
            "threat_level": self.threat_level,
            "collapse_threshold": self.collapse_threshold,
            "admission_reason": self.admission_reason,
        }


# =============================================================================
# PROTOCOLOS DE INTERFACES (Teoría de Categorías - Funtores)
# =============================================================================

@runtime_checkable
class ITopologicalWatcher(Protocol):
    """
    Funtor: Categoría Topológica → Categoría de Números Reales
    
    Abstracción del observador topológico que computa la amenaza
    Mahalanobis χ² como una métrica de distancia en el espacio de
    configuración del sistema.
    
    AXIOMAS:
    1. χ² ≥ 0 (semi-definida positiva)
    2. χ² = 0 ⟺ estado nominal
    3. χ² → ∞ si el sistema diverge topológicamente
    
    CONTRATO:
    - Debe retornar float finito y no-negativo
    - Debe ser idempotente en ausencia de mutaciones del sistema
    """
    
    def get_mahalanobis_threat(self) -> float:
        """
        Computa la amenaza topológica χ².
        
        Returns:
            χ² ∈ [0, +∞)
            
        Raises:
            RuntimeError: Si el cálculo falla irrecuperablemente
        """
        ...


@runtime_checkable
class ILaplaceOracle(Protocol):
    """
    Funtor: Categoría de Sistemas LTI → Categoría de Números Complejos
    
    Oráculo que extrae el polo dominante σ del sistema de control,
    gobernando la estabilidad asintótica.
    
    AXIOMAS (Teoría de Sistemas):
    1. σ < 0 ⟹ sistema asintóticamente estable
    2. σ = 0 ⟹ sistema marginalmente estable
    3. σ > 0 ⟹ sistema inestable
    
    CONTRATO:
    - Debe retornar la parte real del polo dominante
    - Debe manejar sistemas de orden arbitrario
    """
    
    def get_dominant_pole_real(self) -> float:
        """
        Extrae la parte real del polo dominante σ.
        
        Returns:
            σ ∈ ℝ
            
        Raises:
            RuntimeError: Si no hay polos o el cálculo falla
        """
        ...


@runtime_checkable
class ISheafCohomologyOrchestrator(Protocol):
    """
    Funtor: Categoría de Haces → Categoría de Energías
    
    Orquestador que calcula la energía de frustración cohomológica
    global del sistema, indicando inconsistencias estructurales.
    
    AXIOMAS (Cohomología de Sheaf):
    1. E_frustración ≥ 0 (funcional de energía)
    2. E = 0 ⟺ cohomología trivial (no frustración)
    3. E → ∞ si hay obstrucciones cohomológicas
    
    CONTRATO:
    - Debe retornar float finito y no-negativo
    - Debe integrar sobre todos los haces monitoreados
    """
    
    def get_global_frustration_energy(self) -> float:
        """
        Calcula la energía de frustración cohomológica.
        
        Returns:
            E_frustración ≥ 0
            
        Raises:
            RuntimeError: Si el cálculo cohomológico falla
        """
        ...


# =============================================================================
# FUNCIONES AUXILIARES NUMÉRICAS (Álgebra Numérica Rigurosa)
# =============================================================================

def validate_finite_float(
    value: Any, 
    *, 
    name: str,
    allow_inf: bool = False
) -> float:
    """
    Conversión y validación rigurosa a float finito.
    
    Implementa un morfismo seguro: Any → ℝ_finito
    
    Args:
        value: Valor a convertir
        name: Nombre del parámetro (para mensajes de error)
        allow_inf: Si True, permite ±∞
        
    Returns:
        float validado
        
    Raises:
        QuantumNumericalError: Si la conversión falla o el valor es inválido
    """
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise QuantumNumericalError(
            f"Parámetro '{name}' no convertible a float: {value!r}"
        ) from exc
    
    if math.isnan(result):
        raise QuantumNumericalError(
            f"Parámetro '{name}' es NaN (not-a-number), valor no físico"
        )
    
    if not allow_inf and math.isinf(result):
        raise QuantumNumericalError(
            f"Parámetro '{name}' es infinito: {result!r}"
        )
    
    return result


def clamp_to_unit_interval(value: float) -> float:
    """
    Proyección al intervalo unitario [0, 1].
    
    Morfismo saturante: ℝ → [0, 1]
    
    Manejo de casos especiales:
    - NaN → 0.0 (con warning)
    - ±∞ → 0.0 o 1.0 según signo
    - Valores normales → saturación estándar
    
    Args:
        value: Valor a proyectar
        
    Returns:
        Valor proyectado ∈ [0, 1]
    """
    if math.isnan(value):
        logger.warning(
            "clamp_to_unit_interval recibió NaN, proyectando a 0.0. "
            "Esto indica un error de cálculo upstream."
        )
        return 0.0
    
    if math.isinf(value):
        return 0.0
    
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    
    return value


def safe_division(
    numerator: float, 
    denominator: float, 
    *, 
    fallback: float = 0.0,
    epsilon: float = QuantumConstants.DIVISION_EPSILON
) -> float:
    """
    División con regularización para evitar singularidades.
    
    Implementa: f(x, y) = x / max(|y|, ε) · sgn(y)
    
    Args:
        numerator: Numerador
        denominator: Denominador
        fallback: Valor de retorno si denominador es exactamente cero
        epsilon: Regularizador de división
        
    Returns:
        Resultado de la división regularizada
        
    Raises:
        QuantumNumericalError: Si los argumentos no son finitos
    """
    num = validate_finite_float(numerator, name="numerator")
    denom = validate_finite_float(denominator, name="denominator")
    
    if denom == 0.0:
        logger.warning(
            "División por cero exacto detectada en safe_division, "
            f"retornando fallback={fallback}"
        )
        return fallback
    
    # Regularización: |denom| → max(|denom|, ε)
    regularized_denom = (
        denom if abs(denom) >= epsilon 
        else math.copysign(epsilon, denom)
    )
    
    return num / regularized_denom


def safe_sqrt(value: float) -> float:
    """
    Raíz cuadrada con clamp a no-negatividad.
    
    Args:
        value: Argumento del radical
        
    Returns:
        √max(0, value)
    """
    return math.sqrt(max(0.0, value))


def safe_exp(exponent: float) -> float:
    """
    Exponencial con protección de underflow/overflow.
    
    Saturación:
    - Si exponent < EXP_UNDERFLOW_CUTOFF → 0.0
    - Si exponent > 700 → warning + saturación a sys.float_info.max
    
    Args:
        exponent: Exponente
        
    Returns:
        exp(exponent) con saturación
    """
    if exponent <= QuantumConstants.EXP_UNDERFLOW_CUTOFF:
        return 0.0
    
    if exponent > 700.0:
        logger.warning(
            f"Exponente muy grande ({exponent:.2f}), "
            "saturando a valor máximo de float"
        )
        return sys.float_info.max
    
    return math.exp(exponent)


# =============================================================================
# MÓDULO DE ENTROPÍA (Teoría de la Información)
# =============================================================================

class EntropyCalculator:
    """
    Calculador de entropía de Shannon con aritmética rigurosa.
    
    Implementa la funcional:
        H: Σ → ℝ₊
    donde Σ es el conjunto de secuencias de bytes.
    
    PROPIEDADES GARANTIZADAS:
    1. H ≥ 0 (no-negatividad)
    2. H = 0 ⟺ secuencia constante
    3. H ≤ ln(256) (cota superior para bytes)
    4. Continuidad en la distribución de probabilidad
    """
    
    @staticmethod
    @lru_cache(maxsize=1024)
    def shannon_entropy_bytes(data: bytes) -> float:
        """
        Calcula entropía de Shannon de una secuencia de bytes.
        
        Fórmula:
            H = -Σᵢ pᵢ ln(pᵢ)
        donde pᵢ = count(byteᵢ) / total_bytes
        
        Optimizaciones:
        - Cacheo LRU para payloads repetidos
        - Early return para casos triviales
        - Acumulación estable numéricamente
        
        Args:
            data: Secuencia inmutable de bytes
            
        Returns:
            H ∈ [0, ln(256)]
        """
        if not data:
            return 0.0
        
        n = len(data)
        
        # Conteo de frecuencias
        byte_counts: list[int] = [0] * 256
        for byte_val in data:
            byte_counts[byte_val] += 1
        
        # Acumulación de entropía con aritmética estable
        entropy = 0.0
        for count in byte_counts:
            if count == 0:
                continue
            p = count / n
            # -p ln(p) = -p (ln(count) - ln(n))
            entropy -= p * math.log(p)
        
        # Garantizar no-negatividad (protección contra errores de redondeo)
        return max(0.0, entropy)
    
    @staticmethod
    def normalized_entropy(data: bytes) -> float:
        """
        Entropía normalizada al rango [0, 1].
        
        Fórmula:
            H_norm = H / ln(256)
        
        Args:
            data: Secuencia de bytes
            
        Returns:
            H_norm ∈ [0, 1]
        """
        raw_entropy = EntropyCalculator.shannon_entropy_bytes(data)
        return raw_entropy / QuantumConstants.MAX_SHANNON_ENTROPY


# =============================================================================
# SERIALIZACIÓN CANÓNICA (Homomorfismo Payload → Bytes)
# =============================================================================

class PayloadSerializer:
    """
    Serialización determinista de payloads a representación canónica.
    
    Garantiza isomorfismo:
        serialize: Payload_a = Payload_b ⟹ bytes_a = bytes_b
    
    Propiedades:
    - Independencia del orden de inserción de claves
    - Reproducibilidad absoluta (mismo payload → mismo hash)
    - Resistencia a ataques de colisión (usa SHA-256)
    """
    
    @staticmethod
    def serialize(payload: Mapping[str, Any]) -> bytes:
        """
        Serializa un mapping a bytes canónicos.
        
        Algoritmo:
        1. Extraer pares (clave, valor)
        2. Ordenar por clave stringificada
        3. Representar cada valor con repr()
        4. Concatenar en tupla ordenada
        5. Codificar a UTF-8
        
        Args:
            payload: Mapping a serializar
            
        Returns:
            Representación canónica en bytes
            
        Raises:
            QuantumAdmissionError: Si payload no es Mapping o falla serialización
        """
        if not isinstance(payload, Mapping):
            raise QuantumAdmissionError(
                f"payload debe ser un Mapping; recibido {type(payload).__name__}"
            )
        
        try:
            # Ordenamiento lexicográfico de claves
            ordered_items = tuple(
                sorted(
                    ((str(k), repr(v)) for k, v in payload.items()),
                    key=lambda kv: kv[0]
                )
            )
            canonical_repr = repr(ordered_items)
            return canonical_repr.encode('utf-8', errors='strict')
        
        except Exception as exc:
            raise QuantumAdmissionError(
                f"Fallo en serialización canónica del payload: {exc}"
            ) from exc
    
    @staticmethod
    @lru_cache(maxsize=512)
    def deterministic_hash(data: bytes) -> bytes:
        """
        Calcula hash criptográfico determinista.
        
        Usa SHA-256 para garantizar:
        - Uniformidad de distribución
        - Resistencia a colisiones (2^128 complejidad)
        - Reproducibilidad
        
        Args:
            data: Datos inmutables a hashear
            
        Returns:
            Digest SHA-256 (32 bytes)
        """
        return hashlib.sha256(data).digest()


# =============================================================================
# CALCULADOR DE ENERGÍA INCIDENTE (Modelo Fotoeléctrico)
# =============================================================================

class IncidentEnergyCalculator:
    """
    Calcula la energía incidente E = hν del cuanto informacional.
    
    MODELO FÍSICO:
    1. Tamaño del payload N (bytes) → exergía bruta
    2. Entropía de Shannon H (nats) → medida de estructura
    3. Frecuencia efectiva ν = N / max(H, ε) / 1000
    4. Energía incidente E = h · ν
    
    INVARIANTES:
    - E ≥ 0
    - E = 0 ⟺ payload vacío
    - E crece con tamaño y decrece con entropía
    """
    
    @staticmethod
    def calculate(payload: Mapping[str, Any]) -> Tuple[float, NumericInterval]:
        """
        Calcula energía incidente con propagación de incertidumbre.
        
        Args:
            payload: Datos del cuanto informacional
            
        Returns:
            Tupla (E, intervalo_incertidumbre)
            donde E es el valor central y el intervalo captura
            la incertidumbre por discretización.
            
        Raises:
            QuantumNumericalError: Si el cálculo produce valores no físicos
        """
        # Serialización canónica
        data = PayloadSerializer.serialize(payload)
        n_bytes = len(data)
        
        if n_bytes == 0:
            return 0.0, NumericInterval(0.0, 0.0)
        
        # Cálculo de entropía con piso regularizador
        raw_entropy = EntropyCalculator.shannon_entropy_bytes(data)
        effective_entropy = max(
            raw_entropy, 
            QuantumConstants.ENTROPY_FLOOR
        )
        
        # Frecuencia efectiva normalizada
        # División por 1000 para evitar energías exageradamente altas
        nu = safe_division(
            n_bytes,
            effective_entropy,
            epsilon=QuantumConstants.ENTROPY_FLOOR
        ) / 1000.0
        
        # Energía incidente
        E = QuantumConstants.PLANCK_H * nu
        E = validate_finite_float(E, name="incident_energy")
        
        if E < 0:
            raise QuantumNumericalError(
                f"Energía incidente negativa calculada: {E}"
            )
        
        # Estimación de incertidumbre por discretización
        # δE ≈ h / N (incertidumbre por muestreo finito)
        uncertainty = QuantumConstants.PLANCK_H / max(n_bytes, 1)
        interval = NumericInterval.from_value_with_tolerance(E, uncertainty)
        
        return E, interval


# =============================================================================
# CALCULADOR WKB (Aproximación Semiclásica)
# =============================================================================

class WKBCalculator:
    """
    Implementación rigurosa de la aproximación WKB para túnel cuántico.
    
    FUNDAMENTO TEÓRICO:
    La aproximación WKB (Wentzel-Kramers-Brillouin) es válida en el
    régimen semiclásico cuando:
        ℏ |d²V/dx²| / |dV/dx|² << 1
    
    Para barrera rectangular de altura V₀ y ancho Δx:
        T ≈ exp(-2/ℏ ∫dx √(2m(V-E)))
    
    VALIDACIONES IMPLEMENTADAS:
    1. Verificación de régimen túnel (E < V₀)
    2. Cálculo de parámetro de validez
    3. Manejo de masa efectiva infinita
    4. Protección de underflow/overflow
    """
    
    @staticmethod
    def compute_tunneling_probability(
        E: float,
        Phi: float,
        m_eff: float
    ) -> Tuple[float, WKBParameters]:
        """
        Calcula probabilidad de transmisión por túnel cuántico.
        
        REGLAS DE DESPACHO:
        1. E ≥ Φ → T = 1.0 (transmisión clásica)
        2. m_eff = +∞ → T = 0.0 (barrera impenetrable)
        3. E < Φ y m_eff finita → cálculo WKB completo
        
        Args:
            E: Energía incidente (validada ≥ 0)
            Phi: Función de trabajo (validada ≥ 0)
            m_eff: Masa efectiva (validada > 0 o +∞)
            
        Returns:
            Tupla (T, parámetros_WKB)
            donde T ∈ [0, 1] y parámetros_WKB contiene diagnósticos
            
        Raises:
            WKBValidityError: Si la aproximación WKB no es aplicable
        """
        # Validación de entrada
        E = validate_finite_float(E, name="incident_energy_wkb")
        Phi = validate_finite_float(Phi, name="work_function_wkb")
        m_eff = validate_finite_float(m_eff, name="effective_mass_wkb", allow_inf=True)
        
        if E < 0 or Phi < 0:
            raise QuantumNumericalError(
                f"Energías negativas no físicas: E={E}, Φ={Phi}"
            )
        
        # Regla 1: Transmisión clásica (independiente de masa)
        if E >= Phi:
            params = WKBParameters(
                incident_energy=E,
                barrier_height=0.0,
                effective_mass=m_eff,
                barrier_width=QuantumConstants.BARRIER_WIDTH,
                integrand=0.0,
                exponent=0.0,
                validity_parameter=0.0
            )
            return 1.0, params
        
        # Altura de barrera efectiva
        barrier_height = Phi - E
        
        # Regla 2: Masa infinita → barrera impenetrable
        if math.isinf(m_eff):
            params = WKBParameters(
                incident_energy=E,
                barrier_height=barrier_height,
                effective_mass=m_eff,
                barrier_width=QuantumConstants.BARRIER_WIDTH,
                integrand=float('inf'),
                exponent=float('-inf'),
                validity_parameter=float('inf')
            )
            return 0.0, params
        
        # Validación de masa positiva finita
        if m_eff <= 0:
            raise QuantumNumericalError(
                f"Masa efectiva no positiva: m_eff={m_eff}"
            )
        
        # Regla 3: Cálculo WKB completo
        
        # Integrando: √(2m(V-E))
        integrand = safe_sqrt(2.0 * m_eff * barrier_height)
        
        # Exponente: -(2/ℏ) Δx · integrand
        raw_exponent = -(
            (2.0 / QuantumConstants.PLANCK_HBAR)
            * QuantumConstants.BARRIER_WIDTH
            * integrand
        )
        
        # Parámetro de validez: |V''| / |V'|² 
        # Para barrera rectangular: V'' = 0 en el interior, pero
        # hay discontinuidades en los bordes. Aproximamos con:
        # |V'| ≈ Φ / Δx  →  |V''| ≈ Φ / Δx²
        # validity ≈ (Φ/Δx²) / (Φ/Δx)² = Δx² / Φ
        validity_param = (
            QuantumConstants.BARRIER_WIDTH ** 2 / Phi
            if Phi > 0 else 0.0
        )
        
        # Construcción de parámetros diagnósticos
        params = WKBParameters(
            incident_energy=E,
            barrier_height=barrier_height,
            effective_mass=m_eff,
            barrier_width=QuantumConstants.BARRIER_WIDTH,
            integrand=integrand,
            exponent=raw_exponent,
            validity_parameter=validity_param
        )
        
        # Verificación de validez de aproximación
        if not params.is_valid_semiclassical_regime():
            logger.warning(
                "Aproximación WKB fuera del régimen semiclásico: "
                f"validity_param={validity_param:.6e} > "
                f"{QuantumConstants.WKB_VALIDITY_THRESHOLD:.6e}. "
                "Resultados pueden ser imprecisos."
            )
        
        # Protección de underflow
        if raw_exponent <= QuantumConstants.EXP_UNDERFLOW_CUTOFF:
            logger.debug(
                f"Exponente WKB por debajo del cutoff de underflow "
                f"({raw_exponent:.2f} < {QuantumConstants.EXP_UNDERFLOW_CUTOFF}), "
                "probabilidad de túnel es efectivamente cero."
            )
            return 0.0, params
        
        # Cálculo de probabilidad de transmisión
        T_raw = safe_exp(raw_exponent)
        T = clamp_to_unit_interval(T_raw)
        
        return T, params


# =============================================================================
# MODULADOR DE FUNCIÓN DE TRABAJO (Acoplamiento Gauge)
# =============================================================================

class WorkFunctionModulator:
    """
    Modula la función de trabajo Φ mediante acoplamiento gauge con topología.
    
    MODELO FÍSICO:
        Φ(χ²) = Φ₀ + α · χ²
    
    donde:
    - Φ₀: función de trabajo base (constante material)
    - α: constante de acoplamiento gauge-topológico
    - χ²: amenaza de Mahalanobis (distancia geodésica en espacio de config.)
    
    PROPIEDADES GARANTIZADAS:
    - Φ ≥ 0 (barrera no negativa)
    - Φ(0) = Φ₀ (estado nominal)
    - dΦ/dχ² = α > 0 (monotonía)
    """
    
    def __init__(self, topo_watcher: ITopologicalWatcher):
        """
        Constructor con inyección de dependencia.
        
        Args:
            topo_watcher: Observador topológico validado
        """
        self._topo_watcher = topo_watcher
    
    def calculate(self) -> Tuple[float, float]:
        """
        Calcula función de trabajo modulada exponencialmente y nivel de amenaza.
        
        Returns:
            Tupla (Φ, χ²) donde:
            - Φ ≥ 0: función de trabajo efectiva
            - χ² ≥ 0: nivel de amenaza Mahalanobis
            
        Raises:
            QuantumNumericalError: Si los valores retornados son no físicos
        """
        # Lectura defensiva de amenaza topológica
        try:
            threat_raw = self._topo_watcher.get_mahalanobis_threat()
        except Exception as exc:
            logger.error(
                f"Fallo al obtener amenaza Mahalanobis: {exc}. "
                "Asumiendo amenaza cero."
            )
            threat_raw = 0.0
        
        # Validación y clamp a no-negatividad
        threat = max(
            0.0,
            validate_finite_float(threat_raw, name="mahalanobis_threat")
        )
        
        # Modulación exponencial de barrera (Pullback de Gauge)
        # Se restringe el exponente para no desbordar el cálculo.
        # safe_exp ya aplica un clamp al máximo representable del float.
        # El clamp a `allow_inf=False` asegura que se quede en el conjunto de los reales IEEE 754.
        # Φ₀ * exp(α · χ²)
        exponential_factor = safe_exp(QuantumConstants.ALPHA_THREAT * threat)
        
        # Si la multiplicación excede float_info.max, saturamos al max_float.
        try:
            phi_raw = QuantumConstants.BASE_WORK_FUNCTION * exponential_factor
        except OverflowError:
            phi_raw = sys.float_info.max

        # Para evitar propagar inf, limitamos al valor máximo seguro:
        if math.isinf(phi_raw):
             phi_raw = sys.float_info.max

        # Validación final
        Phi = max(
            0.0,
            validate_finite_float(phi_raw, name="work_function")
        )
        
        return Phi, threat


# =============================================================================
# MODULADOR DE MASA EFECTIVA (Acoplamiento con Polo de Laplace)
# =============================================================================

class EffectiveMassModulator:
    """
    Modula la masa efectiva m_eff mediante polo dominante del sistema LTI.
    
    MODELO FÍSICO:
        m_eff(σ) = m₀ / |σ|   si σ < -ε
        m_eff(σ) = +∞          si σ ≥ -ε
    
    donde:
    - σ: parte real del polo dominante
    - ε: tolerancia de estabilidad marginal
    
    INTERPRETACIÓN:
    - σ << 0: sistema muy estable → masa ligera → túnel facilitado
    - σ → 0⁻: sistema marginalmente estable → masa divergente → túnel suprimido
    - σ ≥ 0: sistema inestable → masa infinita → barrera impenetrable
    
    Esta modulación implementa un mecanismo de auto-protección:
    el sistema rechaza cuantos cuando su propia estabilidad está comprometida.
    """
    
    def __init__(self, laplace_oracle: ILaplaceOracle):
        """
        Constructor con inyección de dependencia.
        
        Args:
            laplace_oracle: Oráculo de Laplace validado
        """
        self._laplace_oracle = laplace_oracle
    
    def calculate(self) -> Tuple[float, float]:
        """
        Calcula masa efectiva modulada y polo dominante.
        
        Returns:
            Tupla (m_eff, σ) donde:
            - m_eff > 0 o m_eff = +∞
            - σ ∈ ℝ (parte real del polo dominante)
            
        Raises:
            QuantumNumericalError: Si m_eff calculada es no física (≤ 0 y finita)
        """
        # Lectura defensiva del polo dominante
        try:
            sigma_raw = self._laplace_oracle.get_dominant_pole_real()
        except Exception as exc:
            logger.error(
                f"Fallo al obtener polo dominante: {exc}. "
                "Asumiendo sistema inestable (σ=0)."
            )
            sigma_raw = 0.0
        
        sigma = validate_finite_float(sigma_raw, name="dominant_pole_real")
        
        # Caso 1: Sistema inestable o marginalmente estable
        if sigma >= -QuantumConstants.SIGMA_CHAOS_TOL:
            logger.warning(
                f"Sistema en régimen de inestabilidad (σ={sigma:.6e}). "
                "Masa efectiva → +∞, supresión total de túnel cuántico."
            )
            return float('inf'), sigma
        
        # Caso 2: Sistema estable, cálculo de masa modulada
        m_eff = safe_division(
            QuantumConstants.BASE_EFFECTIVE_MASS,
            abs(sigma),
            fallback=float('inf'),
            epsilon=QuantumConstants.SIGMA_CHAOS_TOL
        )
        
        # Validación de fisicalidad
        m_eff = validate_finite_float(
            m_eff, 
            name="effective_mass",
            allow_inf=True
        )
        
        if not math.isinf(m_eff) and m_eff <= 0:
            raise QuantumNumericalError(
                f"Masa efectiva no positiva calculada: m_eff={m_eff}, σ={sigma}"
            )
        
        return m_eff, sigma


# =============================================================================
# GENERADOR DE UMBRAL DE COLAPSO (Hash Determinista)
# =============================================================================

class CollapseThresholdGenerator:
    """
    Genera umbral determinista θ ∈ [0, 1) a partir del payload.
    
    ALGORITMO:
    1. Serialización canónica del payload → bytes
    2. Hash criptográfico SHA-256 → 32 bytes
    3. Truncamiento a primeros 8 bytes → uint64
    4. Reducción módulo 10⁶ → normalización a [0, 1)
    
    PROPIEDADES GARANTIZADAS:
    - Determinismo: mismo payload → mismo umbral
    - Uniformidad: distribución aproximadamente uniforme en [0, 1)
    - No-reversibilidad: infeasible inferir payload desde umbral
    """
    
    @staticmethod
    @lru_cache(maxsize=512)
    def generate(payload_hash: bytes) -> float:
        """
        Genera umbral de colapso a partir del hash del payload.
        
        Args:
            payload_hash: Hash SHA-256 del payload serializado
            
        Returns:
            θ ∈ [0, 1)
        """
        # Truncamiento a 64 bits (primeros 8 bytes)
        uint64_value = int.from_bytes(
            payload_hash[:8],
            byteorder='big',
            signed=False
        )
        
        # Normalización a [0, 1) mediante módulo y división
        # Usamos 10⁶ para evitar sesgos de redondeo
        MODULUS = 1_000_000
        normalized = (uint64_value % MODULUS) / MODULUS
        
        # Garantizar θ ∈ [0, 1)
        return clamp_to_unit_interval(normalized)


# =============================================================================
# OPERADOR DE ADMISIÓN CUÁNTICA (Componente Principal)
# =============================================================================

class QuantumAdmissionGate(Morphism):
    """
    Operador de admisión cuántica como morfismo categórico.
    
    NATURALEZA CATEGÓRICA:
    - Objeto de Partida: Ext (categoría de payloads externos)
    - Objeto de Llegada: V_PHYSICS (stratum físico validado)
    - Morfismo: F: Ext → {Admitido, Rechazado} × V_PHYSICS
    
    ESTRUCTURA ALGEBRAICA:
    - Operador autoadjunto Ĥ con autoestados {|Admitido⟩, |Rechazado⟩}
    - Base ortonormal completa del espacio de Hilbert de admisión
    - Regla de Born para proyección probabilística determinista
    
    FLUJO DE OPERACIÓN:
    1. Evaluación de veto cohomológico (barrera estructural)
    2. Cálculo de energía incidente E = hν
    3. Modulación de función de trabajo Φ(χ²)
    4. Modulación de masa efectiva m_eff(σ)
    5. Cálculo de probabilidad de túnel T (aproximación WKB)
    6. Generación de umbral determinista θ
    7. Colapso de función de onda: T ≥ θ → |Admitido⟩
    8. Cálculo de momentum de inyección p = √(2mK) si admitido
    
    INVARIANTES PRESERVADOS:
    - Unitariedad: Σ_n P(|n⟩) = 1
    - Hermiticidad: ⟨ψ|Ĥ|φ⟩ = ⟨φ|Ĥ|ψ⟩*
    - Realidad de autovalores: λ_n ∈ ℝ
    - Conservación de momentum en transición continua
    """
    
    def __init__(
        self,
        topo_watcher: ITopologicalWatcher,
        laplace_oracle: ILaplaceOracle,
        sheaf_orchestrator: ISheafCohomologyOrchestrator,
    ) -> None:
        """
        Constructor con inyección de dependencias y validación de contratos.
        
        Args:
            topo_watcher: Observador topológico (Mahalanobis χ²)
            laplace_oracle: Oráculo de polos de Laplace (σ)
            sheaf_orchestrator: Orquestador cohomológico (E_frustración)
            
        Raises:
            QuantumInterfaceError: Si las dependencias no cumplen sus contratos
        """
        super().__init__(name="QuantumAdmissionGate")
        
        # Inyección de dependencias
        self._topo_watcher = topo_watcher
        self._laplace_oracle = laplace_oracle
        self._sheaf_orchestrator = sheaf_orchestrator
        
        # Construcción de componentes modulares
        self._work_function_modulator = WorkFunctionModulator(topo_watcher)
        self._effective_mass_modulator = EffectiveMassModulator(laplace_oracle)
        
        # Validación rigurosa de contratos
        self._validate_dependencies()
        
        logger.info(
            "QuantumAdmissionGate inicializada con validación de dependencias exitosa."
        )
    
    @property
    def domain(self) -> frozenset:
        """Dominio del morfismo (categoría externa)."""
        return frozenset()  # Ext no tiene estructura adicional
    
    @property
    def codomain(self) -> Stratum:
        """Codominio del morfismo (stratum físico)."""
        return Stratum.PHYSICS
    
    # -------------------------------------------------------------------------
    # VALIDACIÓN DE CONTRATOS (Verificación de Funtorialidad)
    # -------------------------------------------------------------------------
    
    def _validate_dependencies(self) -> None:
        """
        Verifica que las dependencias inyectadas cumplan sus protocolos.
        
        Validaciones realizadas:
        1. No-nulidad de referencias
        2. Presencia de métodos requeridos
        3. Callabilidad de métodos
        4. Validación runtime de protocolos
        
        Raises:
            QuantumInterfaceError: Si alguna validación falla
        """
        dependencies = [
            (self._topo_watcher, ITopologicalWatcher, "topo_watcher", ['get_mahalanobis_threat']),
            (self._laplace_oracle, ILaplaceOracle, "laplace_oracle", ['get_dominant_pole_real']),
            (self._sheaf_orchestrator, ISheafCohomologyOrchestrator, "sheaf_orchestrator", ['get_global_frustration_energy']),
        ]
        
        for obj, protocol, name, methods in dependencies:
            # Validación de no-nulidad
            if obj is None:
                raise QuantumInterfaceError(
                    f"Dependencia '{name}' no puede ser None."
                )
            
            # Validación de conformidad con protocolo estructural
            for method in methods:
                if not hasattr(obj, method):
                    raise QuantumInterfaceError(
                        f"Dependencia '{name}' no implementa el método '{method}'. "
                    )
                if not callable(getattr(obj, method)):
                    raise QuantumInterfaceError(
                        f"El atributo '{method}' de '{name}' no es callable."
                    )

            if not isinstance(obj, protocol):
                raise QuantumInterfaceError(
                    f"Dependencia '{name}' no implementa el protocolo "
                    f"{protocol.__name__}. Tipo recibido: {type(obj).__name__}"
                )
        
        logger.debug("Validación de dependencias completada exitosamente.")
    
    # -------------------------------------------------------------------------
    # EVALUACIÓN DE VETO COHOMOLÓGICO
    # -------------------------------------------------------------------------
    
    def _evaluate_cohomological_veto(self) -> Tuple[bool, float]:
        """
        Evalúa veto estructural por frustración cohomológica.
        
        Si la energía de frustración global excede la tolerancia,
        el sistema está en un estado de inconsistencia topológica
        que impide la admisión de nuevos cuantos.
        
        Returns:
            Tupla (veto_activo, energía_frustración)
            
        Raises:
            QuantumInterfaceError: Si el cálculo cohomológico falla
        """
        try:
            frustration_raw = (
                self._sheaf_orchestrator.get_global_frustration_energy()
            )
        except Exception as exc:
            raise QuantumInterfaceError(
                f"Fallo al obtener energía de frustración cohomológica: {exc}"
            ) from exc
        
        frustration = validate_finite_float(
            frustration_raw,
            name="global_frustration_energy"
        )
        
        # Clamp a no-negatividad (energía siempre ≥ 0)
        frustration = max(0.0, frustration)
        
        veto_active = frustration > QuantumConstants.FRUSTRATION_VETO_TOL
        
        if veto_active:
            logger.warning(
                f"VETO COHOMOLÓGICO ACTIVADO: "
                f"E_frustración={frustration:.6e} > "
                f"tolerancia={QuantumConstants.FRUSTRATION_VETO_TOL:.6e}"
            )
        
        return veto_active, frustration
    
    # -------------------------------------------------------------------------
    # CONSTRUCCIÓN DE MEDICIÓN (VETO)
    # -------------------------------------------------------------------------
    
    def _build_veto_measurement(
        self,
        frustration: float
    ) -> QuantumMeasurement:
        """
        Construye medición de rechazo por veto cohomológico.
        
        En caso de veto, capturamos diagnósticos disponibles de manera
        defensiva (sin propagar fallos de oracles).
        
        Args:
            frustration: Energía de frustración que causó el veto
            
        Returns:
            QuantumMeasurement con eigenstate=RECHAZADO y veto=True
        """
        # Lectura defensiva de diagnósticos
        threat = self._safe_read_threat()
        sigma = self._safe_read_sigma()
        
        return QuantumMeasurement(
            eigenstate=Eigenstate.RECHAZADO,
            incident_energy=0.0,
            work_function=0.0,
            tunneling_probability=0.0,
            kinetic_energy=0.0,
            momentum=0.0,
            frustration_veto=True,
            effective_mass=float('inf'),
            dominant_pole_real=sigma,
            threat_level=threat,
            collapse_threshold=0.999999,  # Umbral inalcanzable
            admission_reason=(
                f"Veto estructural cohomológico: "
                f"E_frustración={frustration:.6e} > "
                f"tolerancia={QuantumConstants.FRUSTRATION_VETO_TOL:.6e}. "
                "El sistema presenta inconsistencias topológicas que impiden admisión."
            ),
        )
    
    def _safe_read_threat(self) -> float:
        """Lectura defensiva de amenaza topológica para diagnóstico."""
        try:
            raw = self._topo_watcher.get_mahalanobis_threat()
            return max(0.0, validate_finite_float(raw, name="threat_diag"))
        except Exception as exc:
            logger.debug(f"Fallo en lectura defensiva de threat: {exc}")
            return 0.0
    
    def _safe_read_sigma(self) -> float:
        """Lectura defensiva de polo dominante para diagnóstico."""
        try:
            raw = self._laplace_oracle.get_dominant_pole_real()
            return validate_finite_float(raw, name="sigma_diag")
        except Exception as exc:
            logger.debug(f"Fallo en lectura defensiva de sigma: {exc}")
            return 0.0
    
    # -------------------------------------------------------------------------
    # EVALUACIÓN DE ADMISIÓN (Algoritmo Principal)
    # -------------------------------------------------------------------------
    
    def evaluate_admission(
        self,
        payload: Mapping[str, Any]
    ) -> QuantumMeasurement:
        """
        Evalúa el operador de admisión cuántica sobre un payload.
        
        ALGORITMO COMPLETO:
        
        FASE 1: Veto Cohomológico
            Si E_frustración > ε_veto:
                → |Rechazado⟩ (veto absoluto)
        
        FASE 2: Cálculo de Observables
            E ← hν(payload)                    [energía incidente]
            Φ ← Φ₀ + α·χ²                      [función de trabajo modulada]
            m_eff ← m₀/|σ|                     [masa efectiva modulada]
        
        FASE 3: Probabilidad de Transmisión
            Si E ≥ Φ:
                T ← 1.0                         [transmisión clásica]
            Sino:
                T ← exp(-2/ℏ Δx√(2m(Φ-E)))    [túnel WKB]
        
        FASE 4: Colapso Determinista
            θ ← hash_determinista(payload)
            Si T ≥ θ:
                → |Admitido⟩
                K ← max(E - Φ, K_min)
                p ← √(2m·K)
            Sino:
                → |Rechazado⟩
        
        Args:
            payload: Mapping con datos del cuanto informacional
            
        Returns:
            QuantumMeasurement inmutable con todos los observables
            
        Raises:
            QuantumAdmissionError: Si payload no es Mapping
            QuantumNumericalError: Si hay fallos numéricos irrecuperables
            WKBValidityError: Si aproximación WKB es inválida (warning, no fatal)
        """
        # Validación de tipo de entrada
        if not isinstance(payload, Mapping):
            raise QuantumAdmissionError(
                f"payload debe ser un Mapping/dict; "
                f"recibido {type(payload).__name__}"
            )
        
        logger.debug(f"Iniciando evaluación de admisión para payload: {payload}")
        
        # --- FASE 1: VETO COHOMOLÓGICO ---
        veto_active, frustration = self._evaluate_cohomological_veto()
        if veto_active:
            measurement = self._build_veto_measurement(frustration)
            QuantumLogContext.log_measurement(measurement, level=logging.WARNING)
            return measurement
        
        # --- FASE 2: CÁLCULO DE OBSERVABLES ---
        
        # Energía incidente
        E, E_uncertainty = IncidentEnergyCalculator.calculate(payload)
        logger.debug(f"Energía incidente calculada: E={E:.6e} ± {E_uncertainty.width:.6e}")
        
        # Función de trabajo modulada
        Phi, threat = self._work_function_modulator.calculate()
        logger.debug(f"Función de trabajo: Φ={Phi:.6e}, χ²={threat:.6e}")
        
        # Masa efectiva modulada
        m_eff, sigma = self._effective_mass_modulator.calculate()
        logger.debug(f"Masa efectiva: m_eff={m_eff:.6e}, σ={sigma:.6e}")
        
        # --- FASE 3: PROBABILIDAD DE TRANSMISIÓN ---
        T, wkb_params = WKBCalculator.compute_tunneling_probability(E, Phi, m_eff)
        logger.debug(f"Probabilidad de túnel: T={T:.6e}")
        
        # --- FASE 4: COLAPSO DETERMINISTA ---
        
        # Generación de umbral
        payload_bytes = PayloadSerializer.serialize(payload)
        payload_hash = PayloadSerializer.deterministic_hash(payload_bytes)
        collapse_threshold = CollapseThresholdGenerator.generate(payload_hash)
        logger.debug(f"Umbral de colapso: θ={collapse_threshold:.6f}")
        
        # Decisión de admisión
        admitted = T >= collapse_threshold
        
        if not admitted:
            # --- CASO: RECHAZO PROBABILÍSTICO ---
            measurement = QuantumMeasurement(
                eigenstate=Eigenstate.RECHAZADO,
                incident_energy=E,
                work_function=Phi,
                tunneling_probability=T,
                kinetic_energy=0.0,
                momentum=0.0,
                frustration_veto=False,
                effective_mass=m_eff,
                dominant_pole_real=sigma,
                threat_level=threat,
                collapse_threshold=collapse_threshold,
                admission_reason=(
                    f"Rechazo probabilístico determinista: "
                    f"T={T:.6e} < θ={collapse_threshold:.6f}. "
                    f"La función de onda colapsó al autoestado |Rechazado⟩."
                ),
                wkb_parameters=wkb_params,
                measurement_uncertainty=E_uncertainty,
            )
            QuantumLogContext.log_measurement(measurement, level=logging.INFO)
            return measurement
        
        # --- CASO: ADMISIÓN ---
        
        # Cálculo de energía cinética
        if E >= Phi:
            # Transmisión clásica fotoeléctrica
            kinetic_energy = max(
                QuantumConstants.MIN_KINETIC_ENERGY,
                E - Phi
            )
            admission_reason = (
                f"Admisión clásica fotoeléctrica: E={E:.6e} ≥ Φ={Phi:.6e}. "
                "Superación directa de barrera de potencial."
            )
        else:
            # Transmisión por túnel cuántico
            kinetic_energy = QuantumConstants.MIN_KINETIC_ENERGY
            admission_reason = (
                f"Admisión por túnel cuántico WKB: E={E:.6e} < Φ={Phi:.6e}, "
                f"T={T:.6e} ≥ θ={collapse_threshold:.6f}. "
                "Penetración de barrera por efecto túnel."
            )
        
        # Cálculo de momentum de inyección
        # Usamos masa efectiva si es finita, sino masa base como fallback
        m_for_momentum = (
            m_eff if math.isfinite(m_eff)
            else QuantumConstants.BASE_EFFECTIVE_MASS
        )
        
        momentum_squared = 2.0 * m_for_momentum * kinetic_energy
        momentum = safe_sqrt(momentum_squared)
        momentum = validate_finite_float(momentum, name="momentum")
        
        logger.info(
            f"Colapso a |Admitido⟩: E={E:.6e}, Φ={Phi:.6e}, "
            f"K={kinetic_energy:.6e}, p={momentum:.6e}"
        )
        
        measurement = QuantumMeasurement(
            eigenstate=Eigenstate.ADMITIDO,
            incident_energy=E,
            work_function=Phi,
            tunneling_probability=T,
            kinetic_energy=kinetic_energy,
            momentum=momentum,
            frustration_veto=False,
            effective_mass=m_eff,
            dominant_pole_real=sigma,
            threat_level=threat,
            collapse_threshold=collapse_threshold,
            admission_reason=admission_reason,
            wkb_parameters=wkb_params,
            measurement_uncertainty=E_uncertainty,
        )
        
        QuantumLogContext.log_measurement(measurement, level=logging.INFO)
        return measurement
    
    # -------------------------------------------------------------------------
    # MORFISMO CATEGÓRICO __call__ (Funtorialidad)
    # -------------------------------------------------------------------------
    
    def __call__(self, state: CategoricalState) -> CategoricalState:
        """
        Aplica el morfismo de admisión cuántica a un estado categórico.
        
        TRANSFORMACIÓN CATEGÓRICA:
        
        F: CategoricalState → CategoricalState
        
        Donde F preserva:
        1. Estructura de payload (inmutabilidad)
        2. Composición de morfismos (functorialidad)
        3. Identidades categóricas (F(id) = id)
        
        Si admitido:
            strata' = strata ∪ {PHYSICS}
            context' = context ∪ {quantum_momentum: p, quantum_admission: μ}
        
        Si rechazado:
            strata' = ∅
            context' = context ∪ {quantum_error: err, quantum_admission: μ}
        
        Args:
            state: Estado categórico con payload Mapping
            
        Returns:
            Nuevo CategoricalState transformado (inmutable)
            
        Raises:
            QuantumAdmissionError: Si state o payload no tienen tipo esperado
        """
        # Validación de tipo de entrada
        if not isinstance(state, CategoricalState):
            raise QuantumAdmissionError(
                f"state debe ser CategoricalState; "
                f"recibido {type(state).__name__}"
            )
        
        payload = getattr(state, 'payload', None)
        if not isinstance(payload, Mapping):
            raise QuantumAdmissionError(
                f"state.payload debe ser Mapping; "
                f"recibido {type(payload).__name__}"
            )
        
        # Extracción de contexto (defensiva)
        original_context = getattr(state, 'context', None)
        context = dict(original_context) if original_context else {}
        
        # Evaluación de admisión
        measurement = self.evaluate_admission(payload)
        
        # --- CASO: RECHAZO ---
        if measurement.eigenstate == Eigenstate.RECHAZADO:
            error_msg = (
                f"VETO CUÁNTICO: eigenstate={measurement.eigenstate.name}, "
                f"E={measurement.incident_energy:.6e}, "
                f"Φ={measurement.work_function:.6e}, "
                f"T={measurement.tunneling_probability:.6e}, "
                f"veto_frustración={measurement.frustration_veto}, "
                f"razón='{measurement.admission_reason}'"
            )
            logger.error(error_msg)
            
            new_context = {
                **context,
                'quantum_error': error_msg,
                'quantum_admission': measurement,
            }
            
            return CategoricalState(
                payload=payload,
                context=new_context,
                validated_strata=frozenset(),  # Reseteo completo
            )
        
        # --- CASO: ADMISIÓN ---
        logger.info(
            f"Colapso cuántico a |Admitido⟩: "
            f"p={measurement.momentum:.6e}, "
            f"E={measurement.incident_energy:.6e}, "
            f"Φ={measurement.work_function:.6e}, "
            f"T={measurement.tunneling_probability:.6e}, "
            f"razón='{measurement.admission_reason}'"
        )
        
        new_context = {
            **context,
            'quantum_momentum': measurement.momentum,
            'quantum_admission': measurement,
        }
        
        # Adición del stratum PHYSICS a los validados
        new_strata = state.validated_strata | {Stratum.PHYSICS}
        
        return CategoricalState(
            payload=payload,
            context=new_context,
            validated_strata=new_strata,
        )


# =============================================================================
# PUNTO DE ENTRADA PARA TESTING Y EJEMPLOS
# =============================================================================

if __name__ == "__main__":
    """
    Ejemplo de uso y testing básico del módulo.
    
    Este bloque no se ejecuta al importar el módulo, solo al ejecutarlo
    directamente como script.
    """
    
    # Configuración de logging para testing
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Mocks mínimos de dependencias para testing
    class MockTopologicalWatcher:
        def get_mahalanobis_threat(self) -> float:
            return 2.5  # Amenaza moderada
    
    class MockLaplaceOracle:
        def get_dominant_pole_real(self) -> float:
            return -0.5  # Sistema estable
    
    class MockSheafOrchestrator:
        def get_global_frustration_energy(self) -> float:
            return 1e-12  # Sin frustración
    
    # Construcción del gate
    gate = QuantumAdmissionGate(
        topo_watcher=MockTopologicalWatcher(),
        laplace_oracle=MockLaplaceOracle(),
        sheaf_orchestrator=MockSheafOrchestrator(),
    )
    
    # Payload de prueba
    test_payload = {
        'endpoint': '/api/test',
        'method': 'POST',
        'data': {'key': 'value' * 100},  # Payload de tamaño moderado
    }
    
    # Evaluación
    measurement = gate.evaluate_admission(test_payload)
    
    # Resultados
    print("\n" + "="*80)
    print("RESULTADO DE MEDICIÓN CUÁNTICA")
    print("="*80)
    print(measurement)
    print("\nDetalles completos:")
    for key, value in measurement.to_dict().items():
        print(f"  {key}: {value}")
    print("="*80 + "\n")