"""
Módulo: El Oráculo de Laplace (Tribunal de Estabilidad Espectral)
=================================================================

Este componente actúa como la autoridad matemática suprema del ecosistema,
implementando análisis de Frecuencia Compleja (s = σ + jω) para certificar
que la arquitectura del proyecto es controlable y estable.

Fundamentos Teóricos:
─────────────────────

1. El Plano-S como Mapa de Verdad:
   Modela la dinámica mediante la función de transferencia canónica:
   
       H(s) = ωₙ² / (s² + 2ζωₙs + ωₙ²)

   Donde:
   - σ (Sigma): Representa amortiguamiento. σ < 0 (LHP) = estabilidad.
   - ω (Omega): Representa frecuencia de oscilación.

2. La Pirámide de Laplace (Diagnóstico Jerárquico):
   - Nivel 0 (Vértice): VEREDICTO DE CONTROLABILIDAD
   - Nivel 1 (Robustez): Márgenes de Fase/Ganancia
   - Nivel 2 (Dinámica): Ubicación de Polos, ωₙ, ζ
   - Nivel 3 (Base): Parámetros Físicos (R, L, C)

3. Análisis de Sensibilidad:
   Sensibilidad normalizada: S_x^y = (∂y/∂x) · (x/y) = ∂(ln y)/∂(ln x)

Arquitectura:
─────────────
- `LaplaceOracle`: Fachada principal para análisis de estabilidad
- `SystemAnalyzer`: Cálculos de polos y respuesta transitoria
- `FrequencyAnalyzer`: Respuesta en frecuencia y márgenes
- `SensitivityAnalyzer`: Análisis de sensibilidad paramétrica
- `ReportBuilder`: Generación de reportes estructurados
"""

from __future__ import annotations

import logging
import math
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum, IntEnum, auto
from functools import cached_property, lru_cache
from typing import (
    Any, Callable, Dict, Final, FrozenSet, Generic, Iterator,
    List, NamedTuple, Optional, Protocol, Sequence, Set, Tuple,
    Type, TypeVar, Union, cast, runtime_checkable
)

# =============================================================================
# MANEJO DE DEPENDENCIAS OPCIONALES
# =============================================================================

logger = logging.getLogger("financial_engine")

# NumPy con fallback robusto
_HAS_NUMPY: bool = False
_np_module: Any = None

try:
    import numpy as np
    _HAS_NUMPY = True
    _np_module = np
except ImportError:
    logger.warning("NumPy no disponible - funcionalidad reducida")
    np = None  # type: ignore

# SciPy con fallback
_HAS_SCIPY: bool = False
_scipy_signal: Any = None

try:
    import scipy.signal as _scipy_signal
    _HAS_SCIPY = True
except ImportError:
    logger.warning("SciPy no disponible - funcionalidad reducida")


def _require_numpy(func_name: str) -> None:
    """Verifica disponibilidad de NumPy."""
    if not _HAS_NUMPY:
        raise ImportError(f"{func_name} requiere NumPy. Instale con: pip install numpy")


def _require_scipy(func_name: str) -> None:
    """Verifica disponibilidad de SciPy."""
    if not _HAS_SCIPY:
        raise ImportError(f"{func_name} requiere SciPy. Instale con: pip install scipy")


# =============================================================================
# CONSTANTES NUMÉRICAS (Inmutables y Centralizadas)
# =============================================================================

@dataclass(frozen=True, slots=True)
class NumericalConstants:
    """
    Constantes numéricas centralizadas para consistencia en todo el módulo.
    
    Todas las constantes están agrupadas semánticamente y son inmutables.
    """
    # Tolerancias de comparación
    EPSILON_ZERO: float = 1e-20          # Umbral para comparación con cero
    EPSILON_UNITY: float = 1e-6          # Umbral para comparación con 1
    EPSILON_STABILITY: float = 1e-9      # Umbral para análisis de estabilidad
    
    # Límites físicos (SI)
    MIN_INDUCTANCE_H: float = 1e-15      # Inductancia mínima (femtohenrios)
    MIN_CAPACITANCE_F: float = 1e-18     # Capacitancia mínima (attofaradios)
    MAX_RESISTANCE_OHM: float = 1e9      # Resistencia máxima
    MAX_INDUCTANCE_H: float = 1e6        # Inductancia máxima
    MAX_CAPACITANCE_F: float = 1e3       # Capacitancia máxima
    
    # Límites de frecuencia
    MAX_FREQUENCY_RAD: float = 1e12      # Frecuencia máxima (THz)
    MIN_SAMPLE_RATE_HZ: float = 1.0      # Sample rate mínimo
    MAX_SAMPLE_RATE_HZ: float = 1e12     # Sample rate máximo
    
    # Factores de diseño
    NYQUIST_SAFETY_FACTOR: float = 10.0  # Factor de seguridad sobre Nyquist
    DEFAULT_SETTLING_TOLERANCE: float = 0.02  # 2% para tiempo de asentamiento
    DEFAULT_PHASE_MARGIN_MIN: float = 30.0    # Margen de fase mínimo (grados)
    DEFAULT_PHASE_MARGIN_GOOD: float = 45.0   # Margen de fase bueno (grados)
    
    # Límites de cálculo
    LOG_MIN: float = 1e-30               # Mínimo para logaritmos
    INF_THRESHOLD: float = 1e30          # Umbral para considerar infinito
    
    @classmethod
    def default(cls) -> 'NumericalConstants':
        """Retorna instancia con valores por defecto."""
        return cls()


# Instancia global (singleton pattern)
NC: Final[NumericalConstants] = NumericalConstants.default()


# =============================================================================
# EXCEPCIONES
# =============================================================================

class FinancialEngineError(Exception):
    """Clase base para excepciones del módulo."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.details = details or {}
        self.timestamp = time.time()


class ConfigurationError(FinancialEngineError):
    """Error en configuración del sistema."""
    pass


class StabilityError(FinancialEngineError):
    """Error en análisis de estabilidad."""
    pass


class NumericalError(FinancialEngineError):
    """Error numérico (overflow, underflow, singularidad)."""
    pass


# =============================================================================
# ENUMERACIONES
# =============================================================================

class DampingClass(Enum):
    """
    Clasificación del amortiguamiento del sistema según factor ζ.
    
    | Clase              | Condición      | Polos                    | Comportamiento         |
    |--------------------|----------------|--------------------------|------------------------|
    | NEGATIVE           | ζ < 0          | RHP (parte real > 0)     | Inestable, divergente  |
    | UNDAMPED           | ζ = 0          | Imaginarios puros ±jωₙ   | Oscilador armónico     |
    | UNDERDAMPED        | 0 < ζ < 1      | Complejos conjugados LHP | Oscilación amortiguada |
    | CRITICALLY_DAMPED  | ζ = 1          | Real doble: -ωₙ          | Crítico (más rápido)   |
    | OVERDAMPED         | ζ > 1          | Reales distintos LHP     | Exponencial monótona   |
    """
    NEGATIVE = auto()
    UNDAMPED = auto()
    UNDERDAMPED = auto()
    CRITICALLY_DAMPED = auto()
    OVERDAMPED = auto()
    
    @classmethod
    def from_zeta(cls, zeta: float, epsilon: float = NC.EPSILON_UNITY) -> 'DampingClass':
        """Clasifica según valor de ζ."""
        if zeta < -NC.EPSILON_ZERO:
            return cls.NEGATIVE
        elif abs(zeta) < NC.EPSILON_ZERO:
            return cls.UNDAMPED
        elif zeta < 1.0 - epsilon:
            return cls.UNDERDAMPED
        elif abs(zeta - 1.0) < epsilon:
            return cls.CRITICALLY_DAMPED
        else:
            return cls.OVERDAMPED


class StabilityStatus(Enum):
    """Estado de estabilidad del sistema."""
    STABLE = "STABLE"
    MARGINALLY_STABLE = "MARGINALLY_STABLE"
    UNSTABLE = "UNSTABLE"
    UNKNOWN = "UNKNOWN"
    
    @property
    def is_acceptable(self) -> bool:
        """True si el sistema es aceptable para operación."""
        return self in (StabilityStatus.STABLE, StabilityStatus.MARGINALLY_STABLE)


class ResponseType(Enum):
    """Tipo de respuesta temporal del sistema."""
    DIVERGENT_OSCILLATORY = "DIVERGENT_OSCILLATORY"
    SUSTAINED_OSCILLATION = "SUSTAINED_OSCILLATION"
    DAMPED_OSCILLATION = "DAMPED_OSCILLATION"
    CRITICAL_EXPONENTIAL = "CRITICAL_EXPONENTIAL"
    OVERDAMPED_EXPONENTIAL = "OVERDAMPED_EXPONENTIAL"


class RobustnessClass(Enum):
    """Clasificación de robustez del sistema."""
    ROBUST = "ROBUST"           # max_sens < 0.5
    MODERATE = "MODERATE"       # 0.5 ≤ max_sens < 1.0
    SENSITIVE = "SENSITIVE"     # 1.0 ≤ max_sens < 2.0
    FRAGILE = "FRAGILE"         # max_sens ≥ 2.0
    UNBALANCED = "UNBALANCED"   # Alto número de condición
    NOT_APPLICABLE = "N/A"


# =============================================================================
# ESTRUCTURAS DE DATOS INMUTABLES
# =============================================================================

@dataclass(frozen=True, slots=True)
class RLCParameters:
    """
    Parámetros físicos validados del sistema RLC.
    
    Representa un circuito RLC serie con:
    - R: Resistencia (Ω) - disipación de energía
    - L: Inductancia (H) - almacenamiento de energía magnética
    - C: Capacitancia (F) - almacenamiento de energía eléctrica
    
    Invariantes:
    - L > 0 (inductancia estrictamente positiva)
    - C > 0 (capacitancia estrictamente positiva)
    - R ≥ 0 (resistencia no negativa)
    """
    R: float  # Resistencia en Ohmios
    L: float  # Inductancia en Henrios
    C: float  # Capacitancia en Faradios
    
    def __post_init__(self) -> None:
        """Valida invariantes físicos."""
        errors = []
        
        if not isinstance(self.R, (int, float)) or not math.isfinite(self.R):
            errors.append(f"R debe ser finito, recibido: {self.R}")
        elif self.R < 0:
            errors.append(f"R no puede ser negativo: {self.R} Ω")
        
        if not isinstance(self.L, (int, float)) or not math.isfinite(self.L):
            errors.append(f"L debe ser finito, recibido: {self.L}")
        elif self.L <= 0:
            errors.append(f"L debe ser positivo: {self.L} H")
        
        if not isinstance(self.C, (int, float)) or not math.isfinite(self.C):
            errors.append(f"C debe ser finito, recibido: {self.C}")
        elif self.C <= 0:
            errors.append(f"C debe ser positivo: {self.C} F")
        
        if errors:
            raise ConfigurationError(
                "Parámetros RLC inválidos:\n" + "\n".join(f"  • {e}" for e in errors),
                details={"R": self.R, "L": self.L, "C": self.C}
            )
    
    @property
    def LC_product(self) -> float:
        """Producto L·C."""
        return self.L * self.C
    
    @property
    def RC_product(self) -> float:
        """Producto R·C (constante de tiempo RC)."""
        return self.R * self.C
    
    @property
    def characteristic_impedance(self) -> float:
        """Impedancia característica Z₀ = √(L/C)."""
        return math.sqrt(self.L / self.C)
    
    @property
    def omega_n(self) -> float:
        """Frecuencia natural no amortiguada ωₙ = 1/√(LC)."""
        return 1.0 / math.sqrt(self.LC_product)
    
    @property
    def zeta(self) -> float:
        """Factor de amortiguamiento ζ = (R/2)·√(C/L)."""
        return (self.R / 2.0) * math.sqrt(self.C / self.L)
    
    @property
    def Q(self) -> float:
        """Factor de calidad Q = 1/(2ζ) = √(L/C)/R."""
        if self.R < NC.EPSILON_ZERO:
            return float('inf')
        return math.sqrt(self.L / self.C) / self.R
    
    @property
    def omega_d(self) -> float:
        """Frecuencia amortiguada ωd = ωₙ·√(1-ζ²). Válido solo si 0 < ζ < 1."""
        z = self.zeta
        if 0 < z < 1.0:
            return self.omega_n * math.sqrt(1.0 - z**2)
        return 0.0
    
    @property
    def damping_class(self) -> DampingClass:
        """Clasificación de amortiguamiento."""
        return DampingClass.from_zeta(self.zeta)
    
    def to_dict(self) -> Dict[str, float]:
        """Serializa a diccionario."""
        return {
            "R_ohm": self.R,
            "L_henry": self.L,
            "C_farad": self.C,
            "omega_n_rad_s": self.omega_n,
            "zeta": self.zeta,
            "Q": self.Q,
            "characteristic_impedance_ohm": self.characteristic_impedance,
        }


@dataclass(frozen=True, slots=True)
class ComplexPole:
    """
    Polo de un sistema dinámico en el plano-s.
    
    Representación: p = σ + jω
    donde σ es la parte real y ω la parte imaginaria.
    """
    real: float  # σ (parte real)
    imag: float  # ω (parte imaginaria)
    
    @property
    def magnitude(self) -> float:
        """Magnitud |p| = √(σ² + ω²)."""
        return math.sqrt(self.real**2 + self.imag**2)
    
    @property
    def angle_rad(self) -> float:
        """Ángulo en radianes."""
        return math.atan2(self.imag, self.real)
    
    @property
    def angle_deg(self) -> float:
        """Ángulo en grados."""
        return math.degrees(self.angle_rad)
    
    @property
    def is_stable(self) -> bool:
        """True si el polo está en LHP (σ < 0)."""
        return self.real < NC.EPSILON_STABILITY
    
    @property
    def is_marginally_stable(self) -> bool:
        """True si el polo está sobre el eje imaginario."""
        return abs(self.real) < NC.EPSILON_STABILITY
    
    @property
    def time_constant(self) -> float:
        """Constante de tiempo τ = -1/σ (solo válido si σ < 0)."""
        if self.real < -NC.EPSILON_ZERO:
            return -1.0 / self.real
        return float('inf')
    
    @classmethod
    def from_complex(cls, z: complex) -> 'ComplexPole':
        """Crea desde número complejo de Python."""
        return cls(real=z.real, imag=z.imag)
    
    def to_complex(self) -> complex:
        """Convierte a número complejo de Python."""
        return complex(self.real, self.imag)
    
    def to_tuple(self) -> Tuple[float, float]:
        """Convierte a tupla (real, imag)."""
        return (self.real, self.imag)
    
    def __str__(self) -> str:
        sign = "+" if self.imag >= 0 else "-"
        return f"{self.real:.4f} {sign} j{abs(self.imag):.4f}"


@dataclass(frozen=True, slots=True)
class StabilityMargins:
    """
    Márgenes de estabilidad del sistema.
    
    Para un sistema de 2º orden H(s) = ωₙ²/(s² + 2ζωₙs + ωₙ²),
    el margen de ganancia es infinito (sistema pasivo).
    
    El margen de fase se relaciona con ζ mediante:
        PM ≈ arctan(2ζ / √(√(1 + 4ζ⁴) - 2ζ²))
    """
    gain_margin_db: float
    phase_margin_deg: float
    gain_crossover_freq_rad_s: float
    phase_crossover_freq_rad_s: float
    is_meaningful: bool = True
    regime: str = "UNKNOWN"
    interpretation: str = ""
    
    @property
    def is_robust(self) -> bool:
        """True si los márgenes indican robustez."""
        return (
            self.phase_margin_deg >= NC.DEFAULT_PHASE_MARGIN_GOOD and
            self.gain_margin_db > 6.0
        )
    
    @property
    def is_acceptable(self) -> bool:
        """True si los márgenes son aceptables."""
        return (
            self.phase_margin_deg >= NC.DEFAULT_PHASE_MARGIN_MIN and
            self.gain_margin_db > 0.0
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "gain_margin_db": self.gain_margin_db,
            "phase_margin_deg": self.phase_margin_deg,
            "gain_crossover_freq_rad_s": self.gain_crossover_freq_rad_s,
            "phase_crossover_freq_rad_s": self.phase_crossover_freq_rad_s,
            "is_meaningful": self.is_meaningful,
            "regime": self.regime,
            "interpretation": self.interpretation,
            "is_robust": self.is_robust,
            "is_acceptable": self.is_acceptable,
        }


@dataclass(frozen=True, slots=True)
class TransientMetrics:
    """
    Métricas de respuesta transitoria ante entrada escalón.
    
    Para sistema subamortiguado (0 < ζ < 1):
    - rise_time: Tiempo de 10% a 90%
    - peak_time: Tiempo hasta primer pico
    - overshoot: Sobrepaso porcentual
    - settling_time: Tiempo hasta ±2% del valor final
    """
    status: str
    rise_time_s: float
    peak_time_s: float
    overshoot_percent: float
    settling_time_s: float
    peak_value: float
    steady_state_value: float
    damped_frequency_rad_s: float = 0.0
    oscillations_to_settle: float = 0.0
    notes: str = ""
    
    @property
    def has_overshoot(self) -> bool:
        """True si hay sobrepaso."""
        return self.overshoot_percent > 0.0
    
    @property
    def is_fast(self) -> bool:
        """True si respuesta es rápida (rise_time < 0.1 * settling_time)."""
        if self.settling_time_s <= 0:
            return False
        return self.rise_time_s < 0.1 * self.settling_time_s
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return asdict(self)
    
    @classmethod
    def unstable(cls) -> 'TransientMetrics':
        """Crea métricas para sistema inestable."""
        return cls(
            status="UNSTABLE",
            rise_time_s=float('inf'),
            peak_time_s=float('inf'),
            overshoot_percent=float('inf'),
            settling_time_s=float('inf'),
            peak_value=float('inf'),
            steady_state_value=float('nan'),
            notes="Sistema inestable - respuesta divergente"
        )


@dataclass(frozen=True, slots=True)
class SensitivityMatrix:
    """
    Matriz de sensibilidades normalizadas del sistema.
    
    Sensibilidad normalizada: S_x^y = ∂(ln y)/∂(ln x)
    
    Mide el cambio porcentual en y dado un cambio porcentual en x.
    """
    # Sensibilidades de ωₙ
    S_R_omega_n: float = 0.0
    S_L_omega_n: float = -0.5
    S_C_omega_n: float = -0.5
    
    # Sensibilidades de ζ
    S_R_zeta: float = 1.0
    S_L_zeta: float = -0.5
    S_C_zeta: float = 0.5
    
    # Métricas escalares
    scalar_R: float = 0.0
    scalar_L: float = 0.0
    scalar_C: float = 0.0
    
    @property
    def most_sensitive_parameter(self) -> str:
        """Parámetro con mayor sensibilidad."""
        params = {"R": self.scalar_R, "L": self.scalar_L, "C": self.scalar_C}
        return max(params, key=params.get)
    
    @property
    def max_sensitivity(self) -> float:
        """Máxima sensibilidad escalar."""
        return max(self.scalar_R, self.scalar_L, self.scalar_C)
    
    @property
    def condition_number(self) -> float:
        """Número de condición (ratio max/min sensibilidad)."""
        sens = [s for s in [self.scalar_R, self.scalar_L, self.scalar_C] if s > NC.EPSILON_ZERO]
        if not sens:
            return 1.0
        return max(sens) / min(sens)
    
    @property
    def robustness_class(self) -> RobustnessClass:
        """Clasificación de robustez."""
        max_s = self.max_sensitivity
        
        if max_s >= 2.0:
            return RobustnessClass.FRAGILE
        elif max_s >= 1.0:
            return RobustnessClass.SENSITIVE
        elif self.condition_number > 5.0:
            return RobustnessClass.UNBALANCED
        elif max_s < 0.5:
            return RobustnessClass.ROBUST
        else:
            return RobustnessClass.MODERATE
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return {
            "omega_n": {
                "R": self.S_R_omega_n,
                "L": self.S_L_omega_n, 
                "C": self.S_C_omega_n
            },
            "zeta": {
                "R": self.S_R_zeta,
                "L": self.S_L_zeta,
                "C": self.S_C_zeta
            },
            "scalar_sensitivities": {
                "R": self.scalar_R,
                "L": self.scalar_L,
                "C": self.scalar_C
            },
            "most_sensitive_parameter": self.most_sensitive_parameter,
            "condition_number": self.condition_number,
            "robustness_class": self.robustness_class.value,
        }


@dataclass(frozen=True, slots=True)
class ResonanceInfo:
    """
    Información sobre el pico de resonancia.
    
    El pico existe solo si ζ < 1/√2 ≈ 0.707.
    
    Frecuencia de resonancia: ω_r = ωₙ·√(1 - 2ζ²)
    Magnitud en resonancia: |H(jω_r)| = 1/(2ζ·√(1-ζ²))
    """
    exists: bool
    frequency_rad_s: float = 0.0
    frequency_hz: float = 0.0
    magnitude: float = 1.0
    magnitude_db: float = 0.0
    quality_factor: float = 0.0
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Serializa a diccionario."""
        return asdict(self)


# =============================================================================
# PROTOCOLOS
# =============================================================================

@runtime_checkable
class TransferFunctionProtocol(Protocol):
    """Protocolo para funciones de transferencia."""
    @property
    def poles(self) -> Sequence[complex]: ...
    @property
    def zeros(self) -> Sequence[complex]: ...


# =============================================================================
# CACHE CON TTL
# =============================================================================

@dataclass
class CacheEntry(Generic[TypeVar('T')]):
    """Entrada de cache con timestamp."""
    value: Any
    timestamp: float
    
    def is_expired(self, ttl_seconds: float) -> bool:
        """Verifica si la entrada expiró."""
        return (time.time() - self.timestamp) > ttl_seconds


class AnalysisCache:
    """
    Cache thread-safe para resultados de análisis.
    
    Implementa TTL y límite de tamaño con evicción LRU.
    """
    
    def __init__(self, ttl_seconds: float = 3600.0, max_size: int = 50):
        self._data: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._ttl = ttl_seconds
        self._max_size = max_size
    
    def get(self, key: str) -> Optional[Any]:
        """Obtiene valor si existe y no expiró."""
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if entry.is_expired(self._ttl):
                del self._data[key]
                return None
            return entry.value
    
    def set(self, key: str, value: Any) -> None:
        """Almacena valor con timestamp actual."""
        with self._lock:
            # Evicción si excede tamaño
            if len(self._data) >= self._max_size:
                oldest = min(self._data.items(), key=lambda x: x[1].timestamp)
                del self._data[oldest[0]]
            
            self._data[key] = CacheEntry(value=value, timestamp=time.time())
    
    def invalidate(self, key: Optional[str] = None) -> int:
        """Invalida una o todas las entradas. Retorna cantidad eliminada."""
        with self._lock:
            if key is not None:
                if key in self._data:
                    del self._data[key]
                    return 1
                return 0
            else:
                count = len(self._data)
                self._data.clear()
                return count


# =============================================================================
# VALIDADORES
# =============================================================================

class ParameterValidator:
    """
    Validador de parámetros para sistemas RLC.
    
    Realiza validación exhaustiva incluyendo:
    - Finitud y positividad
    - Rangos físicamente realizables
    - Consistencia con frecuencia de muestreo (Nyquist)
    """
    
    def __init__(self, constants: NumericalConstants = NC):
        self._nc = constants
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def validate_rlc(
        self,
        R: float,
        L: float,
        C: float,
        sample_rate: float
    ) -> Tuple[List[str], List[str]]:
        """
        Valida parámetros RLC.
        
        Returns:
            Tuple (errores, advertencias)
        """
        errors: List[str] = []
        warnings: List[str] = []
        nc = self._nc
        
        # === Validación de finitud y rangos ===
        param_specs = [
            ("R", R, 0.0, nc.MAX_RESISTANCE_OHM, "Ω"),
            ("L", L, nc.MIN_INDUCTANCE_H, nc.MAX_INDUCTANCE_H, "H"),
            ("C", C, nc.MIN_CAPACITANCE_F, nc.MAX_CAPACITANCE_F, "F"),
            ("sample_rate", sample_rate, nc.MIN_SAMPLE_RATE_HZ, nc.MAX_SAMPLE_RATE_HZ, "Hz"),
        ]
        
        for name, value, min_val, max_val, unit in param_specs:
            if not isinstance(value, (int, float)):
                errors.append(f"{name} debe ser numérico, recibido: {type(value).__name__}")
                continue
            
            if not math.isfinite(value):
                errors.append(f"{name} debe ser finito, recibido: {value}")
            elif value < min_val:
                errors.append(f"{name} = {value} {unit} < mínimo {min_val} {unit}")
            elif value > max_val:
                warnings.append(f"{name} = {value:.2e} {unit} excede rango típico")
        
        if errors:
            return errors, warnings
        
        # === Análisis de consistencia física ===
        omega_n = 1.0 / math.sqrt(L * C)
        f_n = omega_n / (2 * math.pi)
        nyquist_freq = sample_rate / 2
        
        # Verificar criterio de Nyquist
        if f_n > nyquist_freq:
            errors.append(
                f"f_n = {f_n:.2e} Hz > f_Nyquist = {nyquist_freq:.2e} Hz. "
                f"Aumente sample_rate a ≥ {2 * f_n * nc.NYQUIST_SAFETY_FACTOR:.0f} Hz"
            )
        elif f_n > nyquist_freq / nc.NYQUIST_SAFETY_FACTOR:
            warnings.append(
                f"f_n = {f_n:.2e} Hz cercana a Nyquist. "
                f"Recomendado: sample_rate ≥ {nc.NYQUIST_SAFETY_FACTOR * 2 * f_n:.0f} Hz"
            )
        
        # Verificar amortiguamiento
        if R > 0:
            zeta = (R / 2.0) * math.sqrt(C / L)
            if zeta < 0.01:
                warnings.append(
                    f"Sistema casi sin amortiguamiento (ζ = {zeta:.4f}). "
                    "Riesgo de oscilaciones persistentes."
                )
            elif zeta > 50:
                warnings.append(
                    f"Sistema extremadamente sobreamortiguado (ζ = {zeta:.1f}). "
                    "Respuesta muy lenta."
                )
        
        return errors, warnings
    
    def validate_and_raise(
        self,
        R: float,
        L: float,
        C: float,
        sample_rate: float
    ) -> None:
        """Valida y lanza excepción si hay errores."""
        errors, warnings = self.validate_rlc(R, L, C, sample_rate)
        
        for warning in warnings:
            self._logger.warning(f"⚠️ {warning}")
        
        if errors:
            raise ConfigurationError(
                "Parámetros físicos inválidos:\n" + "\n".join(f"  • {e}" for e in errors),
                details={"R": R, "L": L, "C": C, "sample_rate": sample_rate}
            )


# =============================================================================
# CALCULADORES ESPECIALIZADOS
# =============================================================================

class StabilityCalculator:
    """
    Calculador de métricas de estabilidad.
    
    Realiza cálculos relacionados con:
    - Márgenes de fase y ganancia
    - Respuesta transitoria
    - Ubicación de polos
    """
    
    def __init__(self, params: RLCParameters, constants: NumericalConstants = NC):
        self._params = params
        self._nc = constants
    
    def calculate_margins(self) -> StabilityMargins:
        """
        Calcula márgenes de estabilidad.
        
        Para sistema de 2º orden pasivo:
        - Margen de ganancia: infinito (siempre estable en ganancia)
        - Margen de fase: relacionado con ζ
        """
        nc = self._nc
        zeta = self._params.zeta
        omega_n = self._params.omega_n
        
        # GM infinito para sistema pasivo
        gain_margin_db = float('inf')
        omega_pc = float('inf')
        
        # Casos especiales
        if zeta <= nc.EPSILON_ZERO:
            return StabilityMargins(
                gain_margin_db=gain_margin_db,
                phase_margin_deg=0.0,
                gain_crossover_freq_rad_s=omega_n,
                phase_crossover_freq_rad_s=omega_pc,
                is_meaningful=False,
                regime="UNDAMPED_OR_UNSTABLE",
                interpretation="Sistema sin amortiguamiento - PM = 0°"
            )
        
        # Cálculo del margen de fase usando relación con lazo abierto implícito
        # PM = arctan(2ζ / √(√(1 + 4ζ⁴) - 2ζ²))
        term_inner = math.sqrt(1.0 + 4.0 * zeta**4) - 2.0 * zeta**2
        
        if term_inner > nc.EPSILON_ZERO:
            phase_margin_deg = math.degrees(
                math.atan(2.0 * zeta / math.sqrt(term_inner))
            )
        else:
            phase_margin_deg = 90.0
        
        # Frecuencia de cruce de ganancia
        omega_gc = omega_n * math.sqrt(term_inner) if term_inner > 0 else 0.0
        
        # Determinar régimen e interpretación
        if zeta < 1.0:
            regime = "UNDERDAMPED"
            quality = "Buena estabilidad." if phase_margin_deg > 45 else "Marginalmente estable."
        elif abs(zeta - 1.0) < nc.EPSILON_UNITY:
            regime = "CRITICALLY_DAMPED"
            quality = "Respuesta óptima sin oscilación."
        else:
            regime = "OVERDAMPED"
            quality = "Sistema muy robusto pero lento."
        
        interpretation = f"Sistema {regime.lower()} (ζ = {zeta:.4f}). PM = {phase_margin_deg:.1f}°. {quality}"
        
        return StabilityMargins(
            gain_margin_db=gain_margin_db,
            phase_margin_deg=phase_margin_deg,
            gain_crossover_freq_rad_s=omega_gc,
            phase_crossover_freq_rad_s=omega_pc,
            is_meaningful=True,
            regime=regime,
            interpretation=interpretation
        )
    
    def calculate_transient_metrics(self, tolerance: float = NC.DEFAULT_SETTLING_TOLERANCE) -> TransientMetrics:
        """
        Calcula métricas de respuesta transitoria ante escalón.
        
        Las fórmulas varían según el régimen de amortiguamiento.
        """
        nc = self._nc
        zeta = self._params.zeta
        omega_n = self._params.omega_n
        omega_d = self._params.omega_d
        
        # === Sistema inestable ===
        if zeta < -nc.EPSILON_ZERO:
            return TransientMetrics.unstable()
        
        # === Sistema sin amortiguamiento ===
        if abs(zeta) < nc.EPSILON_ZERO:
            period = 2.0 * math.pi / omega_n if omega_n > 0 else float('inf')
            return TransientMetrics(
                status="UNDAMPED",
                rise_time_s=period / 4.0,
                peak_time_s=period / 2.0,
                overshoot_percent=100.0,
                settling_time_s=float('inf'),
                peak_value=2.0,
                steady_state_value=1.0,
                damped_frequency_rad_s=omega_n,
                notes="Oscilación armónica sostenida"
            )
        
        # === Sistema subamortiguado (0 < ζ < 1) ===
        if zeta < 1.0 - nc.EPSILON_UNITY:
            return self._calculate_underdamped_metrics(zeta, omega_n, omega_d, tolerance)
        
        # === Sistema críticamente amortiguado (ζ ≈ 1) ===
        if abs(zeta - 1.0) < nc.EPSILON_UNITY:
            return self._calculate_critical_metrics(omega_n)
        
        # === Sistema sobreamortiguado (ζ > 1) ===
        return self._calculate_overdamped_metrics(zeta, omega_n)
    
    def _calculate_underdamped_metrics(
        self,
        zeta: float,
        omega_n: float,
        omega_d: float,
        tolerance: float
    ) -> TransientMetrics:
        """Métricas para sistema subamortiguado."""
        # Tiempo de subida (fórmula exacta)
        phi = math.acos(zeta)
        rise_time = (math.pi - phi) / omega_d
        
        # Tiempo de pico
        peak_time = math.pi / omega_d
        
        # Sobrepaso
        damping_factor = zeta / math.sqrt(1.0 - zeta**2)
        overshoot_factor = math.exp(-math.pi * damping_factor)
        overshoot_percent = overshoot_factor * 100.0
        
        # Tiempo de asentamiento
        sqrt_term = math.sqrt(1.0 - zeta**2)
        settling_argument = tolerance * sqrt_term
        
        if NC.EPSILON_ZERO < settling_argument < 1.0:
            settling_time = -math.log(settling_argument) / (zeta * omega_n)
        else:
            settling_time = 4.0 / (zeta * omega_n)
        
        # Oscilaciones antes de asentarse
        n_oscillations = settling_time * omega_d / (2.0 * math.pi)
        
        return TransientMetrics(
            status="UNDERDAMPED",
            rise_time_s=rise_time,
            peak_time_s=peak_time,
            overshoot_percent=overshoot_percent,
            settling_time_s=settling_time,
            peak_value=1.0 + overshoot_factor,
            steady_state_value=1.0,
            damped_frequency_rad_s=omega_d,
            oscillations_to_settle=n_oscillations,
            notes=f"Oscilación amortiguada con {n_oscillations:.1f} ciclos"
        )
    
    def _calculate_critical_metrics(self, omega_n: float) -> TransientMetrics:
        """Métricas para sistema críticamente amortiguado."""
        # Coeficientes exactos derivados de y(t) = 1 - (1 + ωₙt)e^(-ωₙt)
        rise_time = 3.3579 / omega_n
        settling_time = 5.8335 / omega_n
        
        return TransientMetrics(
            status="CRITICALLY_DAMPED",
            rise_time_s=rise_time,
            peak_time_s=float('inf'),
            overshoot_percent=0.0,
            settling_time_s=settling_time,
            peak_value=1.0,
            steady_state_value=1.0,
            damped_frequency_rad_s=0.0,
            notes="Respuesta monótona más rápida sin sobrepaso"
        )
    
    def _calculate_overdamped_metrics(self, zeta: float, omega_n: float) -> TransientMetrics:
        """Métricas para sistema sobreamortiguado."""
        # Polos: s₁,₂ = -ζωₙ ± ωₙ√(ζ²-1)
        sqrt_disc = math.sqrt(zeta**2 - 1.0)
        s1 = -omega_n * (zeta - sqrt_disc)  # Polo lento
        s2 = -omega_n * (zeta + sqrt_disc)  # Polo rápido
        
        tau_slow = 1.0 / abs(s1)
        
        return TransientMetrics(
            status="OVERDAMPED",
            rise_time_s=2.2 * tau_slow,
            peak_time_s=float('inf'),
            overshoot_percent=0.0,
            settling_time_s=4.0 * tau_slow,
            peak_value=1.0,
            steady_state_value=1.0,
            damped_frequency_rad_s=0.0,
            notes=f"Respuesta dominada por polo lento s₁ = {s1:.4f} rad/s"
        )


class SensitivityCalculator:
    """
    Calculador de análisis de sensibilidad paramétrica.
    
    Calcula sensibilidades normalizadas usando derivadas parciales analíticas:
        S_x^y = (∂y/∂x) · (x/y) = ∂(ln y)/∂(ln x)
    """
    
    def __init__(self, params: RLCParameters, constants: NumericalConstants = NC):
        self._params = params
        self._nc = constants
    
    def calculate_sensitivity_matrix(self) -> SensitivityMatrix:
        """
        Calcula matriz completa de sensibilidades.
        
        Derivaciones analíticas exactas:
            ωₙ = 1/√(LC)  →  S_L^ωₙ = S_C^ωₙ = -1/2
            ζ = (R/2)√(C/L)  →  S_R^ζ = 1, S_L^ζ = -1/2, S_C^ζ = 1/2
        """
        zeta = self._params.zeta
        
        if zeta < self._nc.EPSILON_ZERO:
            # Sistema inestable o sin amortiguamiento
            return SensitivityMatrix()
        
        # === Sensibilidades exactas por derivación ===
        S_L_omega_n = -0.5
        S_C_omega_n = -0.5
        S_R_omega_n = 0.0
        
        S_R_zeta = 1.0
        S_L_zeta = -0.5
        S_C_zeta = 0.5
        
        # === Sensibilidades escalares (normas L2) ===
        scalar_R = abs(S_R_zeta)
        scalar_L = math.sqrt(S_L_omega_n**2 + S_L_zeta**2)
        scalar_C = math.sqrt(S_C_omega_n**2 + S_C_zeta**2)
        
        return SensitivityMatrix(
            S_R_omega_n=S_R_omega_n,
            S_L_omega_n=S_L_omega_n,
            S_C_omega_n=S_C_omega_n,
            S_R_zeta=S_R_zeta,
            S_L_zeta=S_L_zeta,
            S_C_zeta=S_C_zeta,
            scalar_R=scalar_R,
            scalar_L=scalar_L,
            scalar_C=scalar_C
        )
    
    def generate_recommendations(self, matrix: SensitivityMatrix) -> List[str]:
        """Genera recomendaciones basadas en sensibilidad."""
        recommendations: List[str] = []
        HIGH_THRESHOLD = 1.0
        
        if matrix.scalar_R > HIGH_THRESHOLD:
            recommendations.append(
                f"⚠️ Alta sensibilidad a R ({matrix.scalar_R:.2f}). "
                "Use resistores de precisión (±0.1%)."
            )
        
        if matrix.scalar_L > HIGH_THRESHOLD:
            recommendations.append(
                f"⚠️ Alta sensibilidad a L ({matrix.scalar_L:.2f}). "
                "Prefiera inductores con núcleo de ferrita de baja histéresis."
            )
        
        if matrix.scalar_C > HIGH_THRESHOLD:
            recommendations.append(
                f"⚠️ Alta sensibilidad a C ({matrix.scalar_C:.2f}). "
                "Use capacitores NPO/C0G para mínima deriva."
            )
        
        if not recommendations:
            recommendations.append(
                "✓ Sistema bien condicionado. Sensibilidades dentro de rangos aceptables."
            )
        
        return recommendations


class FrequencyResponseCalculator:
    """
    Calculador de respuesta en frecuencia.
    
    Evalúa H(jω) directamente para eficiencia:
        H(jω) = 1 / (1 - LCω² + jRCω)
    """
    
    def __init__(self, params: RLCParameters, constants: NumericalConstants = NC):
        self._params = params
        self._nc = constants
    
    def calculate_resonance(self) -> ResonanceInfo:
        """
        Encuentra pico de resonancia usando fórmulas analíticas.
        
        Frecuencia de resonancia: ω_r = ωₙ√(1 - 2ζ²)
        Existe solo si ζ < 1/√2 ≈ 0.707
        """
        nc = self._nc
        zeta = self._params.zeta
        omega_n = self._params.omega_n
        Q = self._params.Q
        
        zeta_threshold = 1.0 / math.sqrt(2.0)
        
        if zeta >= zeta_threshold or zeta <= nc.EPSILON_ZERO:
            return ResonanceInfo(
                exists=False,
                reason="ζ ≥ 1/√2 - Sin pico de resonancia"
            )
        
        # Frecuencia de resonancia
        omega_r = omega_n * math.sqrt(1.0 - 2.0 * zeta**2)
        
        # Magnitud en resonancia
        denominator = 2.0 * zeta * math.sqrt(1.0 - zeta**2)
        if denominator > nc.EPSILON_ZERO:
            resonance_mag = 1.0 / denominator
        else:
            resonance_mag = float('inf')
        
        resonance_db = (
            20.0 * math.log10(resonance_mag) 
            if resonance_mag < nc.INF_THRESHOLD 
            else float('inf')
        )
        
        return ResonanceInfo(
            exists=True,
            frequency_rad_s=omega_r,
            frequency_hz=omega_r / (2.0 * math.pi),
            magnitude=resonance_mag,
            magnitude_db=resonance_db,
            quality_factor=Q
        )
    
    def calculate_response(
        self,
        frequencies: Optional[Sequence[float]] = None,
        n_points: int = 500
    ) -> Dict[str, Any]:
        """
        Calcula respuesta en frecuencia completa.
        
        Requiere NumPy para evaluación vectorizada.
        """
        _require_numpy("calculate_response")
        
        omega_n = self._params.omega_n
        
        # Generar frecuencias si no se proporcionan
        if frequencies is None:
            w_min = max(omega_n / 1000.0, 1e-4)
            w_max = min(omega_n * 1000.0, self._nc.MAX_FREQUENCY_RAD)
            w = np.logspace(np.log10(w_min), np.log10(w_max), n_points)
        else:
            w = np.array(frequencies)
        
        # Evaluación vectorizada
        LC = self._params.LC_product
        RC = self._params.RC_product
        
        real_part = 1.0 - LC * w**2
        imag_part = RC * w
        
        denom_sq = np.maximum(real_part**2 + imag_part**2, self._nc.LOG_MIN)
        
        magnitude = 1.0 / np.sqrt(denom_sq)
        magnitude_db = 20.0 * np.log10(np.maximum(magnitude, self._nc.LOG_MIN))
        
        H_real = real_part / denom_sq
        H_imag = -imag_part / denom_sq
        
        phase_rad = np.arctan2(H_imag, H_real)
        phase_deg = np.unwrap(phase_rad) * 180 / np.pi
        
        # Ancho de banda -3dB
        bandwidth = self._calculate_bandwidth(w, magnitude_db)
        
        # Resonancia
        resonance = self.calculate_resonance()
        
        return {
            "frequencies_rad_s": w.tolist(),
            "frequencies_hz": (w / (2 * np.pi)).tolist(),
            "magnitude": magnitude.tolist(),
            "magnitude_db": magnitude_db.tolist(),
            "phase_rad": phase_rad.tolist(),
            "phase_deg": phase_deg.tolist(),
            "nyquist_real": H_real.tolist(),
            "nyquist_imag": H_imag.tolist(),
            "resonance": resonance.to_dict(),
            "bandwidth_rad_s": bandwidth,
            "bandwidth_hz": bandwidth / (2 * np.pi),
            "dc_gain_db": 0.0,
            "high_freq_rolloff_db_per_decade": -40.0,
        }
    
    def _calculate_bandwidth(
        self,
        frequencies: 'np.ndarray',
        magnitude_db: 'np.ndarray'
    ) -> float:
        """Calcula ancho de banda -3dB con interpolación logarítmica."""
        if len(magnitude_db) < 2:
            return 0.0
        
        target_db = magnitude_db[0] - 3.0
        
        for i in range(len(magnitude_db) - 1):
            if magnitude_db[i] >= target_db > magnitude_db[i + 1]:
                # Interpolación logarítmica
                log_f1 = np.log10(frequencies[i])
                log_f2 = np.log10(frequencies[i + 1])
                
                t = (target_db - magnitude_db[i]) / (magnitude_db[i + 1] - magnitude_db[i])
                log_bw = log_f1 + t * (log_f2 - log_f1)
                
                return 10.0 ** log_bw
        
        return frequencies[-1] if magnitude_db[-1] > target_db else 0.0


# =============================================================================
# CONSTRUCTOR DE SISTEMAS
# =============================================================================

class TransferFunctionBuilder:
    """
    Constructor de funciones de transferencia.
    
    Construye sistemas continuos y discretos con validación.
    """
    
    def __init__(self, params: RLCParameters, sample_rate: float):
        self._params = params
        self._sample_rate = sample_rate
        self._T = 1.0 / sample_rate
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def build_continuous(self) -> Any:
        """
        Construye función de transferencia continua.
        
        H(s) = 1 / (LCs² + RCs + 1)
        
        Requiere SciPy.
        """
        _require_scipy("build_continuous")
        
        num = [1.0]
        den = [
            self._params.LC_product,
            self._params.RC_product,
            1.0
        ]
        
        try:
            return _scipy_signal.TransferFunction(num, den)
        except Exception as e:
            raise ConfigurationError(f"Error construyendo sistema continuo: {e}")
    
    def build_discrete(self) -> Any:
        """
        Construye función de transferencia discreta.
        
        Usa transformación bilineal (Tustin) con pre-warping en ωₙ.
        
        Requiere SciPy y NumPy.
        """
        _require_scipy("build_discrete")
        _require_numpy("build_discrete")
        
        T = self._T
        omega_n = self._params.omega_n
        
        # Coeficiente de pre-warping
        omega_critical = min(omega_n, NC.MAX_FREQUENCY_RAD)
        omega_T_half = omega_critical * T / 2.0
        
        if omega_T_half < NC.EPSILON_ZERO:
            k = 2.0 / T
        elif omega_T_half > math.pi / 2 - 0.01:
            self._logger.warning(
                f"ωT/2 = {omega_T_half:.3f} ≈ π/2. "
                f"Sistema continuo demasiado rápido para sample_rate = {self._sample_rate} Hz"
            )
            k = 2.0 / T
        else:
            k = omega_critical / math.tan(omega_T_half)
        
        # Coeficientes del sistema continuo
        a2 = self._params.LC_product
        a1 = self._params.RC_product
        a0 = 1.0
        
        k2 = k * k
        
        # Denominador discretizado
        den_z2 = a2 * k2 + a1 * k + a0
        den_z1 = 2.0 * (a0 - a2 * k2)
        den_z0 = a2 * k2 - a1 * k + a0
        
        if abs(den_z2) < NC.EPSILON_ZERO:
            self._logger.warning("Coeficiente líder ≈ 0. Fallback a primer orden.")
            return _scipy_signal.TransferFunction([1.0], [1.0], dt=T)
        
        # Normalizar
        num = np.array([1.0, 2.0, 1.0]) / den_z2
        den = np.array([1.0, den_z1 / den_z2, den_z0 / den_z2])
        
        # Validar estabilidad
        self._validate_discrete_stability(den)
        
        try:
            return _scipy_signal.TransferFunction(num.tolist(), den.tolist(), dt=T)
        except Exception as e:
            self._logger.error(f"Error en discretización: {e}")
            return self.build_continuous()
    
    def _validate_discrete_stability(self, den_coeffs: 'np.ndarray') -> None:
        """Valida que todos los polos estén dentro del círculo unitario."""
        roots = np.roots(den_coeffs)
        max_mag = np.max(np.abs(roots)) if len(roots) > 0 else 0
        
        if max_mag > 1.0 + NC.EPSILON_STABILITY:
            self._logger.error(
                f"⚠️ Sistema discreto INESTABLE: max|polo| = {max_mag:.6f} > 1"
            )
        elif max_mag > 0.99:
            self._logger.warning(
                f"Sistema discreto marginalmente estable: max|polo| = {max_mag:.6f}"
            )


# =============================================================================
# GENERADOR DE REPORTES
# =============================================================================

class ReportBuilder:
    """
    Constructor de reportes estructurados.
    
    Genera la Pirámide de Laplace y reportes completos.
    """
    
    def __init__(
        self,
        params: RLCParameters,
        stability_calc: StabilityCalculator,
        sensitivity_calc: SensitivityCalculator,
        freq_calc: FrequencyResponseCalculator,
        sample_rate: float
    ):
        self._params = params
        self._stability = stability_calc
        self._sensitivity = sensitivity_calc
        self._frequency = freq_calc
        self._sample_rate = sample_rate
    
    def build_laplace_pyramid(self) -> Dict[str, Any]:
        """
        Genera la Pirámide de Laplace - resumen jerárquico de 4 niveles.
        
        Nivel 0 (Vértice): VEREDICTO
        Nivel 1 (Superior): ROBUSTEZ
        Nivel 2 (Medio): DINÁMICA
        Nivel 3 (Base): FÍSICA
        """
        margins = self._stability.calculate_margins()
        sensitivity = self._sensitivity.calculate_sensitivity_matrix()
        
        # Determinar veredicto
        is_controllable = (
            margins.is_acceptable and
            sensitivity.robustness_class not in (RobustnessClass.FRAGILE, RobustnessClass.NOT_APPLICABLE)
        )
        
        confidence = "HIGH" if margins.phase_margin_deg > 45 else "MODERATE"
        
        return {
            "level_0_verdict": {
                "is_controllable": is_controllable,
                "stability_status": self._params.damping_class.name,
                "confidence": confidence,
                "summary": (
                    "✓ APTO para control" if is_controllable
                    else "⚠️ REQUIERE atención antes de control"
                ),
            },
            
            "level_1_robustness": {
                "phase_margin_deg": margins.phase_margin_deg,
                "gain_margin_db": margins.gain_margin_db,
                "robustness_class": sensitivity.robustness_class.value,
                "most_sensitive_param": sensitivity.most_sensitive_parameter,
                "condition_number": sensitivity.condition_number,
            },
            
            "level_2_dynamics": {
                "omega_n_rad_s": self._params.omega_n,
                "omega_n_hz": self._params.omega_n / (2 * math.pi),
                "zeta": self._params.zeta,
                "Q": self._params.Q,
                "damping_class": self._params.damping_class.name,
            },
            
            "level_3_physics": {
                "R_ohm": self._params.R,
                "L_henry": self._params.L,
                "C_farad": self._params.C,
                "sample_rate_hz": self._sample_rate,
                "characteristic_impedance": self._params.characteristic_impedance,
            },
            
            "metadata": {
                "analysis_version": "3.0",
                "timestamp": time.time(),
            }
        }
    
    def build_comprehensive_report(self) -> Dict[str, Any]:
        """Genera reporte completo de análisis."""
        margins = self._stability.calculate_margins()
        transient = self._stability.calculate_transient_metrics()
        sensitivity = self._sensitivity.calculate_sensitivity_matrix()
        resonance = self._frequency.calculate_resonance()
        
        recommendations = self._generate_control_recommendations(margins, sensitivity)
        validation = self._validate_for_control(margins, sensitivity)
        
        return {
            "system_parameters": self._params.to_dict(),
            "stability_margins": margins.to_dict(),
            "transient_response": transient.to_dict(),
            "sensitivity": sensitivity.to_dict(),
            "resonance": resonance.to_dict(),
            "validation": validation,
            "recommendations": recommendations,
            "laplace_pyramid": self.build_laplace_pyramid(),
            "timestamp": time.time(),
            "version": "3.0",
        }
    
    def _validate_for_control(
        self,
        margins: StabilityMargins,
        sensitivity: SensitivityMatrix
    ) -> Dict[str, Any]:
        """Valida idoneidad para diseño de control."""
        issues: List[str] = []
        warnings: List[str] = []
        
        # Verificar márgenes
        if margins.phase_margin_deg < NC.DEFAULT_PHASE_MARGIN_MIN:
            issues.append(f"Margen de fase insuficiente ({margins.phase_margin_deg:.1f}° < 30°)")
        elif margins.phase_margin_deg < NC.DEFAULT_PHASE_MARGIN_GOOD:
            warnings.append(f"Margen de fase marginal ({margins.phase_margin_deg:.1f}°)")
        
        # Verificar Nyquist
        omega_n = self._params.omega_n
        nyquist_limit = 2 * omega_n
        
        if self._sample_rate < nyquist_limit:
            issues.append(
                f"Sample rate insuficiente: {self._sample_rate:.1f} Hz < {nyquist_limit:.1f} Hz"
            )
        elif self._sample_rate < 10 * omega_n:
            warnings.append(
                f"Sample rate bajo: {self._sample_rate:.1f} Hz < {10 * omega_n:.1f} Hz"
            )
        
        # Verificar sensibilidad
        if sensitivity.robustness_class == RobustnessClass.FRAGILE:
            warnings.append("Alta sensibilidad paramétrica")
        
        is_suitable = len(issues) == 0
        
        return {
            "is_suitable_for_control": is_suitable,
            "issues": issues,
            "warnings": warnings,
            "summary": (
                "APTO PARA CONTROL" if is_suitable
                else f"NO APTO - {len(issues)} problemas críticos"
            )
        }
    
    def _generate_control_recommendations(
        self,
        margins: StabilityMargins,
        sensitivity: SensitivityMatrix
    ) -> List[str]:
        """Genera recomendaciones para diseño de control."""
        recommendations: List[str] = []
        zeta = self._params.zeta
        
        if zeta < 0.2:
            recommendations.append(
                f"⚠️ Amortiguamiento muy bajo (ζ = {zeta:.3f}). "
                "Implemente control derivativo (PD)."
            )
        elif zeta > 2.0:
            recommendations.append(
                f"Sistema sobreamortiguado (ζ = {zeta:.3f}). "
                "Considere control proporcional para acelerar respuesta."
            )
        
        if margins.phase_margin_deg < 30:
            recommendations.append(
                f"⚠️ Margen de fase bajo ({margins.phase_margin_deg:.1f}°). "
                "Agregue compensación de adelanto."
            )
        
        if sensitivity.robustness_class == RobustnessClass.FRAGILE:
            recommendations.append(
                "Sistema frágil. Considere control robusto (H∞)."
            )
        
        # Recomendaciones de sensibilidad
        recommendations.extend(
            self._sensitivity.generate_recommendations(sensitivity)
        )
        
        if not recommendations:
            recommendations.append("✓ Sistema bien condicionado para control PID.")
        
        return recommendations


# =============================================================================
# ORÁCULO DE LAPLACE - FACHADA PRINCIPAL
# =============================================================================

class LaplaceOracle:
    """
    Analizador de estabilidad en el dominio de Laplace.
    
    Fachada principal que coordina todos los componentes de análisis
    para sistemas RLC de segundo orden.
    
    Sistema canónico:
        H(s) = ωₙ² / (s² + 2ζωₙs + ωₙ²)
    
    Ejemplo de uso:
        ```python
        oracle = LaplaceOracle(R=100.0, L=0.001, C=1e-6, sample_rate=10000)
        
        # Análisis completo
        stability = oracle.analyze_stability()
        
        # Pirámide de Laplace
        pyramid = oracle.get_laplace_pyramid()
        
        # Respuesta en frecuencia
        freq_response = oracle.get_frequency_response()
        ```
    """
    
    def __init__(
        self,
        R: float,
        L: float,
        C: float,
        sample_rate: float = 1000.0,
        constants: Optional[NumericalConstants] = None
    ):
        """
        Inicializa el analizador.
        
        Args:
            R: Resistencia (Ω) - debe ser ≥ 0
            L: Inductancia (H) - debe ser > 0
            C: Capacitancia (F) - debe ser > 0
            sample_rate: Frecuencia de muestreo para análisis discreto (Hz)
            constants: Constantes numéricas personalizadas (opcional)
        """
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._nc = constants or NC
        
        # Validación exhaustiva
        validator = ParameterValidator(self._nc)
        validator.validate_and_raise(R, L, C, sample_rate)
        
        # Parámetros inmutables
        self._params = RLCParameters(R=float(R), L=float(L), C=float(C))
        self._sample_rate = float(sample_rate)
        self._T = 1.0 / sample_rate
        
        # Exponer propiedades para compatibilidad
        self.R = self._params.R
        self.L = self._params.L
        self.C = self._params.C
        self.omega_n = self._params.omega_n
        self.zeta = self._params.zeta
        self.Q = self._params.Q
        self.omega_d = self._params.omega_d
        self.sample_rate = self._sample_rate
        self.T = self._T
        
        # Clasificación
        self.damping_class = self._params.damping_class
        self.stability_status = self._derive_stability_status()
        self.response_type = self._derive_response_type()
        
        # Componentes especializados
        self._stability_calc = StabilityCalculator(self._params, self._nc)
        self._sensitivity_calc = SensitivityCalculator(self._params, self._nc)
        self._freq_calc = FrequencyResponseCalculator(self._params, self._nc)
        self._report_builder = ReportBuilder(
            self._params,
            self._stability_calc,
            self._sensitivity_calc,
            self._freq_calc,
            self._sample_rate
        )
        
        # Sistemas (lazy loading)
        self._continuous_system: Optional[Any] = None
        self._discrete_system: Optional[Any] = None
        
        # Cache
        self._cache = AnalysisCache(ttl_seconds=3600.0)
    
    def _derive_stability_status(self) -> StabilityStatus:
        """Deriva estado de estabilidad desde clasificación de amortiguamiento."""
        dc = self.damping_class
        
        if dc == DampingClass.NEGATIVE:
            return StabilityStatus.UNSTABLE
        elif dc == DampingClass.UNDAMPED:
            return StabilityStatus.MARGINALLY_STABLE
        else:
            return StabilityStatus.STABLE
    
    def _derive_response_type(self) -> ResponseType:
        """Deriva tipo de respuesta desde clasificación."""
        dc = self.damping_class
        
        mapping = {
            DampingClass.NEGATIVE: ResponseType.DIVERGENT_OSCILLATORY,
            DampingClass.UNDAMPED: ResponseType.SUSTAINED_OSCILLATION,
            DampingClass.UNDERDAMPED: ResponseType.DAMPED_OSCILLATION,
            DampingClass.CRITICALLY_DAMPED: ResponseType.CRITICAL_EXPONENTIAL,
            DampingClass.OVERDAMPED: ResponseType.OVERDAMPED_EXPONENTIAL,
        }
        
        return mapping.get(dc, ResponseType.DAMPED_OSCILLATION)
    
    # ─────────────────────────────────────────────────────────────────────────
    # PROPIEDADES LAZY
    # ─────────────────────────────────────────────────────────────────────────
    
    @property
    def continuous_system(self) -> Any:
        """Función de transferencia continua (lazy loading)."""
        if self._continuous_system is None:
            builder = TransferFunctionBuilder(self._params, self._sample_rate)
            self._continuous_system = builder.build_continuous()
        return self._continuous_system
    
    @property
    def discrete_system(self) -> Any:
        """Función de transferencia discreta (lazy loading)."""
        if self._discrete_system is None:
            builder = TransferFunctionBuilder(self._params, self._sample_rate)
            self._discrete_system = builder.build_discrete()
        return self._discrete_system
    
    # ─────────────────────────────────────────────────────────────────────────
    # MÉTODOS DE ANÁLISIS
    # ─────────────────────────────────────────────────────────────────────────
    
    def analyze_stability(self) -> Dict[str, Any]:
        """
        Análisis completo de estabilidad del sistema.
        
        Combina:
        - Análisis de polos (continuo y discreto)
        - Márgenes de estabilidad
        - Métricas de respuesta transitoria
        - Sensibilidad paramétrica
        """
        cache_key = "stability_analysis"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Polos
        poles_c = [ComplexPole.from_complex(p) for p in self.continuous_system.poles]
        poles_d = [ComplexPole.from_complex(p) for p in self.discrete_system.poles]
        
        # Métricas
        margins = self._stability_calc.calculate_margins()
        transient = self._stability_calc.calculate_transient_metrics()
        sensitivity = self._sensitivity_calc.calculate_sensitivity_matrix()
        recommendations = self._report_builder._generate_control_recommendations(margins, sensitivity)
        
        result = {
            "status": self.stability_status.value,
            "is_stable": self.stability_status == StabilityStatus.STABLE,
            "is_marginally_stable": self.stability_status == StabilityStatus.MARGINALLY_STABLE,
            
            "continuous": {
                "poles": [p.to_tuple() for p in poles_c],
                "natural_frequency_rad_s": self.omega_n,
                "damping_ratio": self.zeta,
                "quality_factor": self.Q,
                "damping_class": self.damping_class.name,
                "response_type": self.response_type.value,
            },
            
            "discrete": {
                "poles": [p.to_tuple() for p in poles_d],
                "sample_rate_hz": self._sample_rate,
                "sample_period_s": self._T,
                "is_stable": all(abs(p.magnitude) < 1.0 + NC.EPSILON_STABILITY for p in poles_d),
            },
            
            "stability_margins": margins.to_dict(),
            "transient_response": transient.to_dict(),
            "parameter_sensitivity": sensitivity.to_dict(),
            "control_recommendations": recommendations,
        }
        
        self._cache.set(cache_key, result)
        return result
    
    def get_frequency_response(
        self,
        frequencies: Optional[Sequence[float]] = None,
        n_points: int = 500,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Calcula respuesta en frecuencia.
        
        Args:
            frequencies: Frecuencias específicas en rad/s (opcional)
            n_points: Número de puntos si no se especifican frecuencias
            use_cache: Si usar cache para resultados
        
        Returns:
            Dict con magnitud, fase, diagrama de Nyquist, etc.
        """
        cache_key = f"freq_response_{n_points}"
        
        if use_cache and frequencies is None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached
        
        result = self._freq_calc.calculate_response(frequencies, n_points)
        
        if use_cache and frequencies is None:
            self._cache.set(cache_key, result)
        
        return result
    
    def get_laplace_pyramid(self) -> Dict[str, Any]:
        """
        Genera la Pirámide de Laplace - resumen jerárquico de 4 niveles.
        
        Returns:
            Dict con niveles 0-3 de la pirámide y metadata
        """
        cache_key = "laplace_pyramid"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        
        result = self._report_builder.build_laplace_pyramid()
        self._cache.set(cache_key, result)
        return result
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """
        Genera reporte completo de análisis.
        
        Returns:
            Dict con todos los análisis combinados
        """
        return self._report_builder.build_comprehensive_report()
    
    def validate_for_control_design(self) -> Dict[str, Any]:
        """
        Valida si el sistema es adecuado para diseño de control.
        
        Returns:
            Dict con {is_suitable_for_control, issues, warnings, recommendations}
        """
        margins = self._stability_calc.calculate_margins()
        sensitivity = self._sensitivity_calc.calculate_sensitivity_matrix()
        
        return self._report_builder._validate_for_control(margins, sensitivity)
    
    # ─────────────────────────────────────────────────────────────────────────
    # MÉTODOS ADICIONALES
    # ─────────────────────────────────────────────────────────────────────────
    
    def get_root_locus_data(
        self,
        K_range: Optional[Sequence[float]] = None,
        n_points: int = 300
    ) -> Dict[str, Any]:
        """
        Genera datos para lugar de las raíces.
        
        Sistema en lazo abierto: G(s) = 1/(LCs² + RCs + 1)
        Ecuación característica: 1 + K·G(s) = 0
        """
        _require_numpy("get_root_locus_data")
        
        if K_range is None:
            K_max = max(100.0, 10.0 / abs(self._params.LC_product)) if abs(self._params.LC_product) > 0 else 100.0
            K_range = np.logspace(-3, math.log10(K_max), n_points)
        else:
            K_range = np.array(K_range)
        
        a = self._params.LC_product
        b = self._params.RC_product
        c_base = 1.0
        
        poles_real: List[float] = []
        poles_imag: List[float] = []
        
        for K in K_range:
            c = c_base + K
            discriminant = b**2 - 4 * a * c
            
            if discriminant >= 0:
                sqrt_disc = math.sqrt(discriminant)
                s1 = (-b + sqrt_disc) / (2 * a)
                s2 = (-b - sqrt_disc) / (2 * a)
                poles_real.extend([s1, s2])
                poles_imag.extend([0.0, 0.0])
            else:
                real_part = -b / (2 * a)
                imag_part = math.sqrt(-discriminant) / (2 * a)
                poles_real.extend([real_part, real_part])
                poles_imag.extend([imag_part, -imag_part])
        
        # Punto de ruptura
        s_break = -b / (2 * a) if abs(a) > 0 else 0
        K_break = -(a * s_break**2 + b * s_break + c_base)
        
        breakpoints = []
        if K_range[0] <= K_break <= K_range[-1]:
            breakpoints.append({
                "real": s_break,
                "imag": 0.0,
                "gain": K_break,
            })
        
        return {
            "gain_values": K_range.tolist(),
            "poles_real": poles_real,
            "poles_imag": poles_imag,
            "asymptote_center": -b / (2 * a),
            "asymptote_angles_deg": [90, 270],
            "breakaway_points": breakpoints,
            "system_order": 2,
        }
    
    def clear_cache(self) -> int:
        """Limpia el cache de análisis. Retorna entradas eliminadas."""
        return self._cache.invalidate()


# =============================================================================
# CONFIGURACIÓN DE MOTOR FINANCIERO (COMPATIBILIDAD)
# =============================================================================

@dataclass(frozen=True, slots=True)
class FinancialConfig:
    """
    Configuración para análisis financiero.
    
    Encapsula parámetros para simulación de proyectos.
    """
    project_life_years: int = 5
    discount_rate: float = 0.10
    risk_free_rate: float = 0.03
    market_volatility: float = 0.30
    random_seed: int = 42
    
    def __post_init__(self) -> None:
        if self.project_life_years <= 0:
            raise ValueError("project_life_years debe ser positivo")
        if not 0 <= self.discount_rate <= 1:
            raise ValueError("discount_rate debe estar en [0, 1]")


class FinancialEngine:
    """
    Motor de análisis financiero para proyectos.
    
    Integra el análisis de estabilidad de Laplace con
    métricas financieras estándar (VPN, TIR, Payback).
    """
    
    def __init__(self, config: FinancialConfig):
        self._config = config
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def analyze_project(
        self,
        initial_investment: float,
        expected_cash_flows: List[float],
        cost_std_dev: float = 0.0,
        project_volatility: float = 0.30
    ) -> Dict[str, Any]:
        """
        Analiza un proyecto de inversión.
        
        Args:
            initial_investment: Inversión inicial
            expected_cash_flows: Flujos de caja esperados por período
            cost_std_dev: Desviación estándar de costos
            project_volatility: Volatilidad del proyecto
        
        Returns:
            Dict con VPN, TIR, Payback, recomendación, etc.
        """
        if not expected_cash_flows:
            return {
                "npv": -initial_investment,
                "irr": None,
                "payback_years": None,
                "recommendation": "REJECT",
                "risk_level": "UNKNOWN",
            }
        
        # VPN
        npv = -initial_investment
        r = self._config.discount_rate
        
        for t, cf in enumerate(expected_cash_flows, start=1):
            npv += cf / ((1 + r) ** t)
        
        # Payback simple
        cumulative = 0.0
        payback = None
        for t, cf in enumerate(expected_cash_flows, start=1):
            cumulative += cf
            if cumulative >= initial_investment:
                payback = t
                break
        
        # TIR (aproximación simple)
        irr = self._approximate_irr(initial_investment, expected_cash_flows)
        
        # Clasificación de riesgo
        cv = cost_std_dev / initial_investment if initial_investment > 0 else 0
        if cv > 0.3:
            risk_level = "HIGH"
        elif cv > 0.15:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Recomendación
        if npv > 0 and (irr is None or irr > r):
            recommendation = "ACCEPT"
        elif npv > 0:
            recommendation = "REVIEW"
        else:
            recommendation = "REJECT"
        
        return {
            "wacc": r,
            "npv": round(npv, 2),
            "irr": round(irr, 4) if irr else None,
            "payback_years": payback,
            "recommendation": recommendation,
            "risk_level": risk_level,
            "volatility": project_volatility,
            "performance": {
                "recommendation": recommendation,
                "risk_level": risk_level,
            }
        }
    
    def _approximate_irr(
        self,
        investment: float,
        cash_flows: List[float],
        max_iter: int = 100,
        tolerance: float = 1e-6
    ) -> Optional[float]:
        """Aproxima TIR usando método de Newton-Raphson."""
        if not cash_flows or investment <= 0:
            return None
        
        # Estimación inicial
        total_cf = sum(cash_flows)
        if total_cf <= investment:
            return None
        
        irr = (total_cf / investment - 1) / len(cash_flows)
        
        for _ in range(max_iter):
            npv = -investment
            d_npv = 0.0
            
            for t, cf in enumerate(cash_flows, start=1):
                discount = (1 + irr) ** t
                npv += cf / discount
                d_npv -= t * cf / ((1 + irr) ** (t + 1))
            
            if abs(d_npv) < NC.EPSILON_ZERO:
                break
            
            delta = npv / d_npv
            irr -= delta
            
            if abs(delta) < tolerance:
                return max(0.0, min(irr, 10.0))  # Acotar a rango razonable
        
        return irr if 0 < irr < 10 else None


# =============================================================================
# ALIAS PARA COMPATIBILIDAD
# =============================================================================

# Mantener nombre original si se importa directamente
SystemParameters = RLCParameters


# =============================================================================
# FUNCIONES DE UTILIDAD
# =============================================================================

def create_oracle_from_dict(config: Dict[str, Any]) -> LaplaceOracle:
    """
    Crea LaplaceOracle desde diccionario de configuración.
    
    Args:
        config: Dict con claves R, L, C, y opcionalmente sample_rate
    
    Returns:
        Instancia de LaplaceOracle
    """
    return LaplaceOracle(
        R=config["R"],
        L=config["L"],
        C=config["C"],
        sample_rate=config.get("sample_rate", 1000.0)
    )


def validate_rlc_parameters(
    R: float,
    L: float,
    C: float,
    sample_rate: float = 1000.0
) -> Dict[str, Any]:
    """
    Valida parámetros RLC sin crear instancia.
    
    Returns:
        Dict con {is_valid, errors, warnings, derived_parameters}
    """
    validator = ParameterValidator()
    errors, warnings = validator.validate_rlc(R, L, C, sample_rate)
    
    result: Dict[str, Any] = {
        "is_valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }
    
    if result["is_valid"]:
        try:
            params = RLCParameters(R=R, L=L, C=C)
            result["derived_parameters"] = params.to_dict()
        except Exception as e:
            result["is_valid"] = False
            result["errors"].append(str(e))
    
    return result