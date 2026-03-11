"""
Este módulo provee la infraestructura matemática de bajo nivel que alimenta al 
'Financial Engine'. Su función es transformar costos deterministas (precios fijos) 
en variables estocásticas (distribuciones de probabilidad), ejecutando simulaciones 
masivas para revelar la volatilidad oculta del proyecto.

Capacidades y Metodologías:
---------------------------
1. Simulación de Monte Carlo Vectorizada:
   Implementa `MonteCarloSimulator` utilizando NumPy para ejecutar miles de 
   iteraciones de escenarios de costos de manera eficiente. Transforma la incertidumbre 
   teórica en un rango de resultados probables.

2. Estadística Descriptiva para Riesgo (VaR y CVaR):
   Calcula métricas críticas no paramétricas, específicamente los percentiles 
   (P5, P95) necesarios para determinar el Valor en Riesgo (VaR) 
   y el Déficit Esperado (CVaR/Expected Shortfall) con confianza estadística.

3. Gestión de Recursos (Memory Safety):
   Implementa `estimate_memory_usage` para predecir la huella de RAM de la matriz 
   de simulación antes de su ejecución. Actúa como un disyuntor preventivo para 
   evitar el colapso del sistema en simulaciones masivas (>1M de iteraciones).

4. Saneamiento de Datos Estocásticos:
   Asegura que los inputs de la simulación sean numéricamente estables, manejando 
   valores infinitos, NaNs y aplicando truncamiento de costos negativos (`truncate_negative`) 
   para mantener la coherencia física del modelo (los precios no pueden ser negativos).

5. Soporte Multi-Distribución:
   Permite elegir entre distribución normal, log-normal y triangular, siendo la 
   log-normal más apropiada para variables que no pueden ser negativas (costos).

6. Técnicas de Reducción de Varianza:
   Implementa variantes antitéticas (antithetic variates) para mejorar la convergencia
   de las simulaciones con menos iteraciones.

7. Análisis de Convergencia:
   Verifica si el número de simulaciones es suficiente para obtener estimaciones
   estadísticamente confiables mediante el error estándar de la media.

8. Análisis de Sensibilidad:
   Identifica qué APUs contribuyen más a la varianza total del proyecto,
   permitiendo focalizar esfuerzos de mitigación de riesgo.
"""

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


# ============================================================================
# ENUMS Y CONSTANTES
# ============================================================================


class SimulationStatus(Enum):
    """Estados posibles de una simulación."""

    SUCCESS = "success"
    NO_VALID_DATA = "no_valid_data"
    INSUFFICIENT_DATA = "insufficient_data"
    CONVERGENCE_WARNING = "convergence_warning"
    ERROR = "error"


class DistributionType(Enum):
    """
    Tipos de distribución soportados para la simulación.
    
    NORMAL: Distribución gaussiana simétrica. Puede generar valores negativos.
    LOGNORMAL: Distribución asimétrica positiva. Ideal para costos y precios.
    TRIANGULAR: Distribución acotada con mínimo, moda y máximo definidos.
    """

    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    TRIANGULAR = "triangular"


# Constantes de validación
DEFAULT_NUM_SIMULATIONS: int = 1000
MIN_NUM_SIMULATIONS: int = 100
MAX_NUM_SIMULATIONS: int = 1_000_000
DEFAULT_VOLATILITY: float = 0.10
MAX_VOLATILITY: float = 1.0
MIN_VOLATILITY: float = 0.0
MEMORY_WARNING_THRESHOLD_BYTES: int = 100_000_000  # 100MB
MEMORY_HARD_LIMIT_GB: int = 10
MIN_VALID_ITEMS_FOR_SIMULATION: int = 1
MAX_SAFE_FLOAT: float = 1e15
CV_WARNING_THRESHOLD: float = 2.0  # Coeficiente de variación considerado alto
CONVERGENCE_TOLERANCE: float = 0.01  # 1% de tolerancia para convergencia
MIN_SCALE_VALUE: float = 1e-10  # Escala mínima para evitar divisiones por cero
Z_SCORE_95: float = 1.96  # Valor crítico para IC al 95%


# ============================================================================
# CONFIGURACIÓN
# ============================================================================


@dataclass
class MonteCarloConfig:
    """
    Configuración para simulaciones de Monte Carlo.

    Esta clase encapsula todos los parámetros que controlan el comportamiento
    de la simulación, permitiendo reproducibilidad y personalización.

    Attributes:
        num_simulations: Número de iteraciones de simulación. Más iteraciones
            producen estimaciones más precisas pero requieren más tiempo y memoria.
        volatility_factor: Factor de volatilidad (0-1) que representa la 
            incertidumbre relativa. Un valor de 0.10 significa ±10% de variación.
        min_cost_threshold: Costo mínimo para considerar un APU válido.
            APUs con VR_TOTAL menor a este valor serán descartados.
        min_quantity_threshold: Cantidad mínima para considerar válida.
        random_seed: Semilla para reproducibilidad (None = aleatorio).
            Usar la misma semilla produce resultados idénticos.
        truncate_negative: Si True, trunca costos negativos a 0 en distribución normal.
            Nota: Esto introduce sesgo. Considere usar distribución log-normal.
        percentiles: Lista de percentiles a calcular en las estadísticas.
        distribution: Tipo de distribución a usar para generar valores aleatorios.
        use_antithetic: Si True, usa variantes antitéticas para reducción de varianza.
            Mejora la convergencia usando la correlación negativa de pares simétricos.
        check_convergence: Si True, verifica convergencia de la simulación.
        use_float32: Si True, usa float32 en lugar de float64 para ahorrar memoria.
            Reduce precisión pero permite simulaciones más grandes.

    Example:
        >>> config = MonteCarloConfig(
        ...     num_simulations=10000,
        ...     volatility_factor=0.15,
        ...     distribution=DistributionType.LOGNORMAL,
        ...     use_antithetic=True
        ... )
    """

    num_simulations: int = DEFAULT_NUM_SIMULATIONS
    volatility_factor: float = DEFAULT_VOLATILITY
    min_cost_threshold: float = 0.0
    min_quantity_threshold: float = 0.0
    random_seed: Optional[int] = None
    truncate_negative: bool = True
    percentiles: List[int] = field(default_factory=lambda: [5, 25, 50, 75, 95])
    distribution: DistributionType = DistributionType.NORMAL
    use_antithetic: bool = False
    check_convergence: bool = True
    use_float32: bool = False

    def __post_init__(self) -> None:
        """Valida y normaliza la configuración después de la inicialización."""
        # Convertir string a enum si es necesario
        if isinstance(self.distribution, str):
            try:
                self.distribution = DistributionType(self.distribution.lower())
            except ValueError:
                valid_types = [d.value for d in DistributionType]
                raise ValueError(
                    f"distribution debe ser uno de {valid_types}, "
                    f"recibido: '{self.distribution}'"
                )

        # Normalizar percentiles: validar tipo, ordenar, eliminar duplicados
        if not isinstance(self.percentiles, list):
            raise TypeError(
                f"percentiles debe ser una lista, "
                f"recibido: {type(self.percentiles).__name__}"
            )

        self.percentiles = sorted(set(self.percentiles))
        self._validate()

    def _validate(self) -> None:
        """Valida todos los parámetros de configuración."""
        self._validate_num_simulations()
        self._validate_volatility()
        self._validate_thresholds()
        self._validate_random_seed()
        self._validate_percentiles()
        self._validate_distribution_compatibility()

    def _validate_num_simulations(self) -> None:
        """Valida el número de simulaciones."""
        if not isinstance(self.num_simulations, int):
            raise TypeError(
                f"num_simulations debe ser int, "
                f"recibido: {type(self.num_simulations).__name__}"
            )
        if not (MIN_NUM_SIMULATIONS <= self.num_simulations <= MAX_NUM_SIMULATIONS):
            raise ValueError(
                f"num_simulations debe estar entre {MIN_NUM_SIMULATIONS:,} y "
                f"{MAX_NUM_SIMULATIONS:,}, recibido: {self.num_simulations:,}"
            )

    def _validate_volatility(self) -> None:
        """Valida el factor de volatilidad."""
        if not isinstance(self.volatility_factor, (int, float)):
            raise TypeError(
                f"volatility_factor debe ser numérico, "
                f"recibido: {type(self.volatility_factor).__name__}"
            )
        if not (MIN_VOLATILITY <= self.volatility_factor <= MAX_VOLATILITY):
            raise ValueError(
                f"volatility_factor debe estar entre {MIN_VOLATILITY} y "
                f"{MAX_VOLATILITY}, recibido: {self.volatility_factor}"
            )

    def _validate_thresholds(self) -> None:
        """Valida los umbrales de costo y cantidad."""
        for name, value in [
            ("min_cost_threshold", self.min_cost_threshold),
            ("min_quantity_threshold", self.min_quantity_threshold),
        ]:
            if not isinstance(value, (int, float)):
                raise TypeError(f"{name} debe ser numérico")
            if value < 0:
                raise ValueError(f"{name} no puede ser negativo")

    def _validate_random_seed(self) -> None:
        """Valida la semilla aleatoria."""
        if self.random_seed is not None:
            if not isinstance(self.random_seed, int):
                raise TypeError("random_seed debe ser int o None")
            if self.random_seed < 0:
                raise ValueError("random_seed no puede ser negativo")

    def _validate_percentiles(self) -> None:
        """Valida la lista de percentiles."""
        if len(self.percentiles) == 0:
            raise ValueError("percentiles no puede estar vacía")
        if not all(isinstance(p, int) for p in self.percentiles):
            raise TypeError("Todos los percentiles deben ser enteros")
        if not all(0 <= p <= 100 for p in self.percentiles):
            raise ValueError("Todos los percentiles deben estar entre 0 y 100")

    def _validate_distribution_compatibility(self) -> None:
        """Valida compatibilidad entre distribución y otras opciones."""
        if self.distribution == DistributionType.LOGNORMAL and self.truncate_negative:
            # Log-normal nunca genera negativos, truncate_negative es redundante
            pass  # Solo informativo, no es un error

    def __repr__(self) -> str:
        """Representación detallada de la configuración."""
        return (
            f"MonteCarloConfig("
            f"n={self.num_simulations:,}, "
            f"vol={self.volatility_factor:.1%}, "
            f"dist={self.distribution.value}, "
            f"antithetic={self.use_antithetic}, "
            f"seed={self.random_seed})"
        )


# ============================================================================
# CLASES DE MÉTRICAS Y RESULTADOS
# ============================================================================


@dataclass
class ConvergenceMetrics:
    """
    Métricas de convergencia de la simulación de Monte Carlo.

    Estas métricas ayudan a determinar si el número de simulaciones
    es suficiente para obtener estimaciones estadísticamente confiables.

    Attributes:
        is_converged: True si la simulación ha convergido según la tolerancia.
        mean_std_error: Error estándar de la media (σ/√n).
        relative_error: Error relativo de la media (SEM/|μ|).
        half_width_ci: Semi-ancho del intervalo de confianza al 95%.
        effective_sample_size: Tamaño de muestra efectivo después de filtrar inválidos.

    Notes:
        La convergencia se define como relative_error < tolerancia (default 1%).
        El intervalo de confianza al 95% es: mean ± half_width_ci
    """

    is_converged: bool
    mean_std_error: float
    relative_error: float
    half_width_ci: float
    effective_sample_size: int

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario serializable."""
        return {
            "is_converged": self.is_converged,
            "mean_std_error": sanitize_value(self.mean_std_error),
            "relative_error": sanitize_value(self.relative_error),
            "half_width_ci": sanitize_value(self.half_width_ci),
            "effective_sample_size": self.effective_sample_size,
        }


@dataclass
class SimulationResult:
    """
    Resultado completo de una simulación de Monte Carlo.

    Encapsula todos los outputs de la simulación incluyendo estadísticas,
    metadata del proceso, métricas de convergencia y opcionalmente los
    resultados brutos para análisis posterior.

    Attributes:
        status: Estado de la simulación (SUCCESS, ERROR, etc.).
        statistics: Diccionario con estadísticas calculadas (mean, std, percentiles, etc.).
        metadata: Información sobre el proceso de simulación.
        raw_results: Array con resultados brutos (opcional, para análisis).
        convergence: Métricas de convergencia (opcional).

    Example:
        >>> result = simulator.run_simulation(apu_data)
        >>> if result.is_successful():
        ...     print(f"Costo esperado: {result.statistics['mean']:,.2f}")
        ...     print(f"VaR 95%: {result.get_var(0.95):,.2f}")
    """

    status: SimulationStatus
    statistics: Dict[str, Optional[float]]
    metadata: Dict[str, Any]
    raw_results: Optional[np.ndarray] = None
    convergence: Optional[ConvergenceMetrics] = None

    def to_dict(self, include_raw: bool = False) -> Dict[str, Any]:
        """
        Convierte el resultado a diccionario serializable.

        Args:
            include_raw: Si True, incluye resultados brutos (puede ser muy grande).

        Returns:
            Diccionario con el resultado completo.
        """
        result = {
            "status": self.status.value,
            "statistics": self.statistics.copy(),
            "metadata": self.metadata.copy(),
        }

        if self.convergence is not None:
            result["convergence"] = self.convergence.to_dict()

        if include_raw and self.raw_results is not None:
            result["raw_results"] = self.raw_results.tolist()

        return result

    def is_successful(self) -> bool:
        """
        Retorna True si la simulación fue exitosa.
        
        Considera exitoso tanto SUCCESS como CONVERGENCE_WARNING,
        ya que en ambos casos se produjeron resultados válidos.
        """
        return self.status in (
            SimulationStatus.SUCCESS,
            SimulationStatus.CONVERGENCE_WARNING,
        )

    def get_var(self, confidence: float = 0.95) -> Optional[float]:
        """
        Obtiene el Value at Risk para un nivel de confianza dado.

        El VaR representa el costo máximo esperado con una probabilidad dada.
        Por ejemplo, VaR al 95% significa que hay 95% de probabilidad de que
        el costo real no supere este valor.

        Args:
            confidence: Nivel de confianza (default 0.95 = 95%).

        Returns:
            VaR o None si no está disponible.
        """
        percentile = int(confidence * 100)
        return self.statistics.get(f"percentile_{percentile}")

    def get_cvar(self, confidence: float = 0.95) -> Optional[float]:
        """
        Obtiene el Conditional Value at Risk (Expected Shortfall).

        El CVaR (también conocido como Expected Shortfall) es el promedio
        de los costos que exceden el VaR. Es una medida de riesgo de cola
        más conservadora que el VaR.

        Args:
            confidence: Nivel de confianza (default 0.95).

        Returns:
            CVaR o None si no está disponible.
        """
        percentile = int(confidence * 100)
        return self.statistics.get(f"cvar_{percentile}")

    def get_confidence_interval(
        self, confidence: float = 0.90
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Obtiene el intervalo de confianza simétrico.

        Args:
            confidence: Nivel de confianza (0.90 = 90%, 0.50 = 50%).

        Returns:
            Tupla (lower, upper) del intervalo de confianza.
        """
        alpha = (1 - confidence) / 2
        lower_pct = int(alpha * 100)
        upper_pct = int((1 - alpha) * 100)

        return (
            self.statistics.get(f"percentile_{lower_pct}"),
            self.statistics.get(f"percentile_{upper_pct}"),
        )

    def __repr__(self) -> str:
        """Representación del resultado."""
        mean = self.statistics.get("mean", 0) or 0
        std = self.statistics.get("std_dev", 0) or 0
        return (
            f"SimulationResult(status={self.status.value}, "
            f"mean={mean:,.2f}, std={std:,.2f})"
        )


# ============================================================================
# UTILIDADES
# ============================================================================


def sanitize_value(
    value: Any, recursive: bool = True, max_depth: int = 50
) -> Optional[Union[float, int, str, list, dict, tuple]]:
    """
    Sanitiza valores para serialización, convirtiendo NaN/inf a None.

    Maneja múltiples tipos de valores especiales de numpy/pandas y los
    convierte a valores seguros para JSON u otros formatos de serialización.
    Implementa protección contra recursión infinita.

    Args:
        value: Valor a sanitizar (cualquier tipo).
        recursive: Si True, sanitiza recursivamente colecciones.
        max_depth: Profundidad máxima de recursión para evitar stack overflow.

    Returns:
        - None si el valor es NaN, NA, inf o -inf
        - El valor convertido a tipo Python nativo en otros casos
        - Colecciones sanitizadas recursivamente si recursive=True

    Examples:
        >>> sanitize_value(np.nan)
        None
        >>> sanitize_value(np.inf)
        None
        >>> sanitize_value(np.float64(10.5))
        10.5
        >>> sanitize_value([1, np.nan, 3])
        [1, None, 3]
    """
    # Protección contra recursión infinita
    if max_depth <= 0:
        warnings.warn(
            "Se alcanzó la profundidad máxima de recursión en sanitize_value",
            RuntimeWarning,
            stacklevel=2,
        )
        return str(value) if value is not None else None

    # Sanitizar listas recursivamente
    if isinstance(value, list):
        if recursive:
            return [
                sanitize_value(item, recursive=True, max_depth=max_depth - 1)
                for item in value
            ]
        return value

    # Sanitizar tuplas recursivamente
    if isinstance(value, tuple):
        if recursive:
            return tuple(
                sanitize_value(item, recursive=True, max_depth=max_depth - 1)
                for item in value
            )
        return value

    # Sanitizar diccionarios recursivamente
    if isinstance(value, dict):
        if recursive:
            return {
                k: sanitize_value(v, recursive=True, max_depth=max_depth - 1)
                for k, v in value.items()
            }
        return value

    # Manejar arrays de numpy (no sanitizar internamente por eficiencia)
    if isinstance(value, np.ndarray):
        return value

    # Strings se retornan sin modificar
    if isinstance(value, str):
        return value

    # Verificar None explícito
    if value is None:
        return None

    # Verificar si es NaN o NA usando pandas
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    # Verificar si es infinito
    try:
        float_val = float(value)
        if not np.isfinite(float_val):
            return None
    except (TypeError, ValueError, OverflowError):
        pass

    # Convertir tipos numpy a tipos nativos de Python
    if isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, (np.complexfloating, complex)):
        # Números complejos no son serializables directamente
        return str(value)

    return value


def validate_required_keys(
    item: Dict[str, Any],
    required_keys_options: List[List[str]],
    item_index: Optional[int] = None,
) -> None:
    """
    Valida que un diccionario contenga al menos una de las opciones de claves requeridas.

    Args:
        item: Diccionario a validar.
        required_keys_options: Lista de listas de opciones de claves.
            Cada sublista representa opciones alternativas para un campo requerido.
        item_index: Índice del item para mensajes de error más descriptivos.

    Raises:
        ValueError: Si faltan claves requeridas.

    Example:
        >>> validate_required_keys(
        ...     {"VR_TOTAL": 100, "cantidad": 5},
        ...     [["VR_TOTAL", "vr_total"], ["CANTIDAD", "cantidad"]]
        ... )
    """
    item_keys = set(item.keys())
    index_msg = f" en índice {item_index}" if item_index is not None else ""

    for key_options in required_keys_options:
        if not any(key in item_keys for key in key_options):
            raise ValueError(
                f"Falta campo requerido{index_msg}. "
                f"Se requiere al menos una de estas claves: {key_options}"
            )


def estimate_memory_usage(
    num_simulations: int,
    num_apus: int,
    use_float32: bool = False,
    include_dataframe_overhead: bool = True,
) -> int:
    """
    Estima el uso de memoria en bytes para la simulación.

    Esta función proporciona una estimación conservadora del uso de memoria,
    considerando la matriz de simulación, arrays auxiliares, DataFrames
    intermedios y overhead de Python/NumPy.

    Args:
        num_simulations: Número de simulaciones.
        num_apus: Número de APUs a simular.
        use_float32: Si True, asume float32 (4 bytes) en lugar de float64 (8 bytes).
        include_dataframe_overhead: Si True, incluye overhead de pandas DataFrame.

    Returns:
        Estimación de bytes necesarios.

    Example:
        >>> mem = estimate_memory_usage(10000, 500)
        >>> print(f"Memoria estimada: {mem / 1024**2:.2f} MB")
    """
    bytes_per_float = 4 if use_float32 else 8

    # Matriz principal: num_simulations × num_apus
    matrix_size = num_simulations * num_apus * bytes_per_float

    # Arrays auxiliares (base_costs, scales, masks, temporales)
    num_auxiliary_arrays = 10
    auxiliary_size = num_apus * bytes_per_float * num_auxiliary_arrays

    # Array de resultados (costos totales por simulación)
    results_size = num_simulations * bytes_per_float

    # Overhead de pandas DataFrame
    dataframe_overhead = 0
    if include_dataframe_overhead:
        # Aproximadamente 80 bytes por fila + estructura base
        dataframe_overhead = num_apus * 80 + 1024

    # Overhead de Python/numpy (aproximado 25%)
    base_memory = matrix_size + auxiliary_size + results_size + dataframe_overhead
    overhead = int(base_memory * 0.25)

    return base_memory + overhead


def is_numeric_valid(value: Any) -> bool:
    """
    Verifica si un valor es numérico válido (no NaN, no inf, finito).

    Args:
        value: Valor a verificar.

    Returns:
        True si es un número válido y finito.

    Example:
        >>> is_numeric_valid(10.5)
        True
        >>> is_numeric_valid(np.nan)
        False
        >>> is_numeric_valid(np.inf)
        False
    """
    try:
        if pd.isna(value):
            return False
        float_value = float(value)
        return np.isfinite(float_value)
    except (TypeError, ValueError, OverflowError):
        return False


def calculate_convergence_metrics(
    simulated_costs: np.ndarray, tolerance: float = CONVERGENCE_TOLERANCE
) -> ConvergenceMetrics:
    """
    Calcula métricas de convergencia para la simulación.

    Utiliza el error estándar de la media (SEM) y el teorema del límite central
    para estimar si la simulación ha convergido adecuadamente. La convergencia
    se define como tener un error relativo menor a la tolerancia especificada.

    Args:
        simulated_costs: Array de costos simulados.
        tolerance: Tolerancia para considerar convergencia (default 1%).

    Returns:
        ConvergenceMetrics con indicadores de convergencia.

    Notes:
        - Error estándar: SEM = σ / √n
        - Error relativo: SEM / |μ|
        - IC 95%: μ ± 1.96 × SEM
    """
    n = len(simulated_costs)
    mean = np.mean(simulated_costs)
    std = np.std(simulated_costs, ddof=1)

    # Error estándar de la media
    mean_std_error = std / np.sqrt(n) if n > 0 else float("inf")

    # Error relativo (coeficiente de variación de la media)
    relative_error = mean_std_error / abs(mean) if mean != 0 else float("inf")

    # Semi-ancho del intervalo de confianza al 95%
    half_width_ci = Z_SCORE_95 * mean_std_error

    # Verificar convergencia
    is_converged = relative_error < tolerance

    return ConvergenceMetrics(
        is_converged=is_converged,
        mean_std_error=float(mean_std_error),
        relative_error=float(relative_error),
        half_width_ci=float(half_width_ci),
        effective_sample_size=n,
    )


# ============================================================================
# CLASE PRINCIPAL
# ============================================================================


class MonteCarloSimulator:
    """
    Simulador de Monte Carlo para análisis estocástico de costos APU.

    Implementa simulaciones estocásticas para estimar distribuciones
    de costos totales considerando incertidumbre en los valores unitarios.
    Soporta múltiples distribuciones de probabilidad y técnicas de
    reducción de varianza.

    Features:
        - Soporte para distribución normal, log-normal y triangular
        - Reducción de varianza con variantes antitéticas
        - Métricas de convergencia para validar número de simulaciones
        - Cálculo de VaR y CVaR (Expected Shortfall)
        - Análisis de sensibilidad para identificar APUs críticos
        - Gestión de memoria para simulaciones masivas

    Attributes:
        config: Configuración de la simulación.
        logger: Logger para registro de eventos.
        rng: Generador de números aleatorios (numpy Generator).

    Example:
        >>> config = MonteCarloConfig(
        ...     num_simulations=10000,
        ...     volatility_factor=0.15,
        ...     distribution=DistributionType.LOGNORMAL
        ... )
        >>> simulator = MonteCarloSimulator(config)
        >>> result = simulator.run_simulation(apu_data)
        >>> print(f"Costo esperado: ${result.statistics['mean']:,.2f}")
    """

    # Aliases de columnas soportados (tuplas inmutables para eficiencia)
    VR_TOTAL_ALIASES: Tuple[str, ...] = ("VR_TOTAL", "valor_total", "vr_total", "VALOR_TOTAL")
    CANTIDAD_ALIASES: Tuple[str, ...] = ("CANTIDAD", "cantidad", "qty", "QTY")

    def __init__(
        self,
        config: Optional[MonteCarloConfig] = None,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Inicializa el simulador.

        Args:
            config: Configuración personalizada (usa default si es None).
            logger: Logger para eventos (crea uno básico si es None).
        """
        self.config = config or MonteCarloConfig()
        self.logger = logger or self._create_default_logger()

        # Inicializar generador de números aleatorios moderno
        self.rng = np.random.default_rng(self.config.random_seed)

        self.logger.info(f"MonteCarloSimulator inicializado: {self.config}")

    @staticmethod
    def _create_default_logger() -> logging.Logger:
        """Crea un logger básico si no se proporciona uno."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def run_simulation(self, apu_details: List[Dict[str, Any]]) -> SimulationResult:
        """
        Ejecuta la simulación de Monte Carlo sobre los datos APU.

        Este es el método principal del simulador. Toma una lista de datos APU,
        los valida, prepara, ejecuta la simulación estocástica y retorna
        estadísticas completas incluyendo métricas de riesgo.

        Args:
            apu_details: Lista de diccionarios con claves 'VR_TOTAL' y 'CANTIDAD'
                        (o sus aliases como 'valor_total', 'cantidad', etc.).

        Returns:
            SimulationResult con estadísticas, metadata y opcionalmente
            resultados brutos y métricas de convergencia.

        Raises:
            TypeError: Si apu_details no tiene el tipo correcto.
            ValueError: Si los datos son inválidos estructuralmente.

        Example:
            >>> apu_data = [
            ...     {"VR_TOTAL": 1000, "CANTIDAD": 5},
            ...     {"VR_TOTAL": 2500, "CANTIDAD": 3},
            ... ]
            >>> result = simulator.run_simulation(apu_data)
        """
        try:
            # 1. Validar estructura de datos de entrada
            self._validate_input_data(apu_details)

            # 2. Preparar y limpiar datos
            df_valid, discarded_count = self._prepare_data(apu_details)

            # 3. Verificar que hay datos válidos suficientes
            if len(df_valid) < MIN_VALID_ITEMS_FOR_SIMULATION:
                return self._create_no_data_result(
                    total_items=len(apu_details), discarded_items=discarded_count
                )

            # 4. Verificar requisitos de memoria
            self._check_memory_requirements(len(df_valid))

            # 5. Ejecutar simulación estocástica
            simulated_costs = self._execute_simulation(df_valid)

            # 6. Validar tasa de éxito de simulaciones
            success_rate = len(simulated_costs) / self.config.num_simulations
            if success_rate < 0.5:
                self.logger.warning(
                    f"Se perdieron más del 50% de simulaciones: "
                    f"{len(simulated_costs)}/{self.config.num_simulations}"
                )

            # 7. Calcular métricas de convergencia
            convergence = None
            status = SimulationStatus.SUCCESS

            if self.config.check_convergence:
                convergence = calculate_convergence_metrics(simulated_costs)
                if not convergence.is_converged:
                    self.logger.warning(
                        f"⚠️ Convergencia insuficiente. "
                        f"Error relativo: {convergence.relative_error:.2%}. "
                        f"Considere aumentar num_simulations."
                    )
                    status = SimulationStatus.CONVERGENCE_WARNING

            # 8. Calcular estadísticas descriptivas y de riesgo
            statistics = self._calculate_statistics(simulated_costs)

            # 9. Preparar metadata del proceso
            metadata = self._create_metadata(
                df_valid=df_valid,
                total_items=len(apu_details),
                discarded_items=discarded_count,
                simulations_completed=len(simulated_costs),
            )

            self.logger.info(
                f"Simulación completada: {len(df_valid)} APUs, "
                f"{len(simulated_costs):,} sims, "
                f"μ={statistics.get('mean', 0):,.2f}"
            )

            return SimulationResult(
                status=status,
                statistics=statistics,
                metadata=metadata,
                raw_results=simulated_costs,
                convergence=convergence,
            )

        except (TypeError, ValueError) as e:
            self.logger.error(f"Error de validación: {str(e)}")
            raise

        except MemoryError as e:
            self.logger.error(f"Error de memoria: {str(e)}")
            return SimulationResult(
                status=SimulationStatus.ERROR,
                statistics={},
                metadata={"error": f"MemoryError: {str(e)}"},
            )

        except Exception as e:
            self.logger.error(f"Error inesperado: {str(e)}", exc_info=True)
            return SimulationResult(
                status=SimulationStatus.ERROR,
                statistics={},
                metadata={"error": str(e), "error_type": type(e).__name__},
            )

    def _validate_input_data(self, apu_details: Any) -> None:
        """Valida los datos de entrada antes de procesarlos."""
        if not isinstance(apu_details, list):
            raise TypeError(
                f"apu_details debe ser una lista, "
                f"recibido: {type(apu_details).__name__}"
            )

        if len(apu_details) == 0:
            raise ValueError("apu_details no puede estar vacía")

        # Validar que todos los elementos sean diccionarios
        invalid_indices = [
            i for i, item in enumerate(apu_details) if not isinstance(item, dict)
        ]
        if invalid_indices:
            raise TypeError(
                f"Todos los elementos deben ser diccionarios. "
                f"Elementos inválidos en índices: {invalid_indices[:5]}"
                + ("..." if len(invalid_indices) > 5 else "")
            )

        # Validar que al menos algunos items tengan las claves requeridas
        required_keys_options = [
            list(self.VR_TOTAL_ALIASES),
            list(self.CANTIDAD_ALIASES),
        ]

        items_to_check = min(10, len(apu_details))
        items_with_required_keys = sum(
            1
            for item in apu_details[:items_to_check]
            if self._has_required_keys(item, required_keys_options)
        )

        if items_with_required_keys == 0:
            raise ValueError(
                f"Ninguno de los primeros {items_to_check} elementos contiene "
                f"las claves requeridas. Se esperan: {self.VR_TOTAL_ALIASES} "
                f"y {self.CANTIDAD_ALIASES}"
            )

    def _has_required_keys(
        self, item: Dict[str, Any], required_keys_options: List[List[str]]
    ) -> bool:
        """Verifica si un item tiene todas las claves requeridas."""
        item_keys = set(item.keys())
        return all(
            any(key in item_keys for key in options) for options in required_keys_options
        )

    def _prepare_data(
        self, apu_details: List[Dict[str, Any]]
    ) -> Tuple[pd.DataFrame, int]:
        """
        Prepara y limpia los datos para la simulación.

        Pipeline de preparación:
        1. Crear DataFrame
        2. Normalizar nombres de columnas
        3. Convertir a numérico
        4. Filtrar valores válidos
        5. Calcular costo base
        6. Filtrar costos base válidos

        Args:
            apu_details: Datos brutos de APU.

        Returns:
            Tupla (DataFrame válido, número de items descartados).
        """
        self.logger.debug(f"Preparando {len(apu_details)} registros APU")

        # Crear DataFrame
        df = pd.DataFrame(apu_details).copy()
        initial_rows = len(df)

        # Pipeline de preparación
        df = self._normalize_columns(df)
        df = self._convert_to_numeric(df)
        df_valid = self._filter_valid_rows(df)
        df_valid = self._calculate_base_cost(df_valid)
        df_valid = self._filter_valid_base_costs(df_valid)

        discarded_count = initial_rows - len(df_valid)

        if discarded_count > 0:
            discard_rate = (discarded_count / initial_rows) * 100
            self.logger.warning(
                f"Descartados {discarded_count}/{initial_rows} items ({discard_rate:.1f}%)"
            )

        return df_valid, discarded_count

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza nombres de columnas usando aliases conocidos."""
        df = df.copy()

        vr_total_col = self._find_column(df.columns, self.VR_TOTAL_ALIASES)
        cantidad_col = self._find_column(df.columns, self.CANTIDAD_ALIASES)

        if vr_total_col is None:
            self.logger.warning("No se encontró columna VR_TOTAL, creando columna NaN")
            df["VR_TOTAL"] = np.nan
        elif vr_total_col != "VR_TOTAL":
            df = df.rename(columns={vr_total_col: "VR_TOTAL"})

        if cantidad_col is None:
            self.logger.warning("No se encontró columna CANTIDAD, creando columna NaN")
            df["CANTIDAD"] = np.nan
        elif cantidad_col != "CANTIDAD":
            df = df.rename(columns={cantidad_col: "CANTIDAD"})

        return df

    @staticmethod
    def _find_column(columns: pd.Index, aliases: Tuple[str, ...]) -> Optional[str]:
        """Encuentra una columna que coincida con algún alias."""
        for col in columns:
            if col in aliases:
                return col
        return None

    def _convert_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convierte columnas relevantes a tipo numérico."""
        df = df.copy()

        for col in ["VR_TOTAL", "CANTIDAD"]:
            original_nulls = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            conversion_failures = df[col].isna().sum() - original_nulls

            if conversion_failures > 0:
                self.logger.warning(
                    f"{col}: {conversion_failures} valores no numéricos convertidos a NaN"
                )

        return df

    def _filter_valid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtra filas con valores válidos que cumplen umbrales."""
        valid_mask = (
            df["VR_TOTAL"].notna()
            & df["CANTIDAD"].notna()
            & np.isfinite(df["VR_TOTAL"])
            & np.isfinite(df["CANTIDAD"])
            & (df["VR_TOTAL"] > self.config.min_cost_threshold)
            & (df["CANTIDAD"] > self.config.min_quantity_threshold)
        )
        return df[valid_mask].copy()

    def _calculate_base_cost(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula el costo base (VR_TOTAL * CANTIDAD)."""
        df = df.copy()
        df["base_cost"] = df["VR_TOTAL"] * df["CANTIDAD"]

        # Detectar overflows (valores extremadamente grandes)
        overflow_mask = np.abs(df["base_cost"]) > MAX_SAFE_FLOAT
        overflow_count = overflow_mask.sum()

        if overflow_count > 0:
            self.logger.warning(
                f"Se detectaron {overflow_count} valores con posible overflow numérico"
            )
            df.loc[overflow_mask, "base_cost"] = np.nan

        return df

    def _filter_valid_base_costs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filtra costos base válidos (finitos, no NaN, no cero)."""
        valid_mask = (
            df["base_cost"].notna()
            & np.isfinite(df["base_cost"])
            & (df["base_cost"] != 0)
        )
        return df[valid_mask].copy()

    def _check_memory_requirements(self, num_apus: int) -> None:
        """Verifica los requisitos de memoria para la simulación."""
        estimated_memory = estimate_memory_usage(
            self.config.num_simulations, num_apus, use_float32=self.config.use_float32
        )

        estimated_mb = estimated_memory / (1024 * 1024)

        self.logger.debug(
            f"Memoria estimada: {estimated_mb:.2f} MB "
            f"({self.config.num_simulations:,} sims × {num_apus} APUs)"
        )

        if estimated_memory > MEMORY_WARNING_THRESHOLD_BYTES:
            self.logger.warning(f"⚠️ Memoria estimada elevada: {estimated_mb:.2f} MB")

        limit_mb = MEMORY_HARD_LIMIT_GB * 1024
        if estimated_mb > limit_mb:
            raise ValueError(
                f"La simulación requiere {estimated_mb:.2f} MB, "
                f"excediendo el límite de {MEMORY_HARD_LIMIT_GB} GB. "
                f"Reduzca num_simulations o el número de APUs."
            )

    def _execute_simulation(self, df_valid: pd.DataFrame) -> np.ndarray:
        """
        Ejecuta la simulación de Monte Carlo según la distribución configurada.

        Selecciona el método de simulación apropiado basado en la configuración
        y aplica técnicas de reducción de varianza si están habilitadas.

        Args:
            df_valid: DataFrame con datos válidos y preparados.

        Returns:
            Array con costos totales simulados (solo valores válidos).
        """
        base_costs = df_valid["base_cost"].values

        # Seleccionar tipo de datos para optimización de memoria
        dtype = np.float32 if self.config.use_float32 else np.float64
        base_costs = base_costs.astype(dtype)

        # Ejecutar simulación según distribución configurada
        if self.config.distribution == DistributionType.LOGNORMAL:
            simulated_costs = self._simulate_lognormal(base_costs, dtype)
        elif self.config.distribution == DistributionType.TRIANGULAR:
            simulated_costs = self._simulate_triangular(base_costs, dtype)
        else:  # NORMAL
            simulated_costs = self._simulate_normal(base_costs, dtype)

        # Validar y filtrar resultados
        return self._validate_simulation_results(simulated_costs)

    def _simulate_normal(self, base_costs: np.ndarray, dtype: np.dtype) -> np.ndarray:
        """
        Genera simulación con distribución normal (gaussiana).

        La distribución normal puede generar valores negativos, por lo que
        se aplica truncamiento si está configurado. Soporta variantes
        antitéticas para reducción de varianza.

        Args:
            base_costs: Array de costos base.
            dtype: Tipo de datos numpy a usar.

        Returns:
            Array de costos totales simulados.
        """
        scales = np.abs(base_costs * self.config.volatility_factor)
        scales = self._validate_and_correct_scales(scales)

        num_sims = self.config.num_simulations

        if self.config.use_antithetic:
            # Variantes antitéticas: genera n/2 valores y usa sus negativos
            # Esto reduce la varianza mediante correlación negativa perfecta
            half_sims = num_sims // 2
            z = self.rng.standard_normal((half_sims, len(base_costs)))
            z = np.vstack([z, -z]).astype(dtype)

            simulated_matrix = base_costs + z * scales
        else:
            simulated_matrix = self.rng.normal(
                loc=base_costs,
                scale=scales,
                size=(num_sims, len(base_costs)),
            ).astype(dtype)

        # Truncar negativos si está configurado
        if self.config.truncate_negative:
            negative_count = (simulated_matrix < 0).sum()
            if negative_count > 0:
                pct = (negative_count / simulated_matrix.size) * 100
                self.logger.debug(f"Truncando {negative_count:,} valores negativos ({pct:.2f}%)")
                simulated_matrix = np.maximum(simulated_matrix, 0)

        return simulated_matrix.sum(axis=1)

    def _simulate_lognormal(self, base_costs: np.ndarray, dtype: np.dtype) -> np.ndarray:
        """
        Genera simulación con distribución log-normal.

        La distribución log-normal garantiza valores positivos naturalmente,
        siendo más apropiada para costos y precios. Los parámetros se calculan
        para que la media coincida con el costo base.

        Para una log-normal con media m y varianza v:
        - μ = ln(m² / √(v + m²))
        - σ² = ln(1 + v/m²)

        Args:
            base_costs: Array de costos base.
            dtype: Tipo de datos numpy a usar.

        Returns:
            Array de costos totales simulados.
        """
        # Calcular varianza objetivo
        variance = (base_costs * self.config.volatility_factor) ** 2

        # Evitar log de cero o negativos
        base_costs_safe = np.maximum(base_costs, MIN_SCALE_VALUE)

        # Convertir a parámetros de log-normal (μ, σ del logaritmo)
        mu = np.log(base_costs_safe**2 / np.sqrt(variance + base_costs_safe**2))
        sigma = np.sqrt(np.log(1 + variance / base_costs_safe**2))

        num_sims = self.config.num_simulations

        if self.config.use_antithetic:
            half_sims = num_sims // 2
            z = self.rng.standard_normal((half_sims, len(base_costs)))
            z = np.vstack([z, -z])
            simulated_matrix = np.exp(mu + sigma * z).astype(dtype)
        else:
            simulated_matrix = self.rng.lognormal(
                mean=mu,
                sigma=sigma,
                size=(num_sims, len(base_costs)),
            ).astype(dtype)

        return simulated_matrix.sum(axis=1)

    def _simulate_triangular(
        self, base_costs: np.ndarray, dtype: np.dtype
    ) -> np.ndarray:
        """
        Genera simulación con distribución triangular.

        La distribución triangular es útil cuando se conocen estimados de
        mínimo, más probable y máximo. Aquí se aproxima usando la volatilidad:
        - left = base_cost × (1 - volatility)
        - mode = base_cost
        - right = base_cost × (1 + volatility)

        Args:
            base_costs: Array de costos base.
            dtype: Tipo de datos numpy a usar.

        Returns:
            Array de costos totales simulados.
        """
        vol = self.config.volatility_factor

        # Configurar límites de la distribución triangular
        left = base_costs * (1 - vol)
        mode = base_costs
        right = base_costs * (1 + vol)

        # Asegurar que left >= 0 (costos no pueden ser negativos)
        left = np.maximum(left, 0)

        num_sims = self.config.num_simulations

        # Generar matriz de simulación
        # Nota: triangular no tiene forma vectorizada eficiente para múltiples parámetros
        simulated_matrix = np.zeros((num_sims, len(base_costs)), dtype=dtype)

        for i in range(len(base_costs)):
            simulated_matrix[:, i] = self.rng.triangular(
                left=left[i], mode=mode[i], right=right[i], size=num_sims
            )

        return simulated_matrix.sum(axis=1)

    def _validate_and_correct_scales(self, scales: np.ndarray) -> np.ndarray:
        """
        Valida y corrige los valores de scale para distribuciones.

        Asegura que todos los valores de escala (desviación estándar) sean
        positivos, finitos y no demasiado pequeños para evitar problemas numéricos.

        Args:
            scales: Array de desviaciones estándar.

        Returns:
            Array de scales corregidos.
        """
        # Corregir valores negativos (no deberían ocurrir, pero por seguridad)
        scales = np.abs(scales)

        # Corregir valores muy pequeños o cero
        small_mask = scales < MIN_SCALE_VALUE
        if small_mask.any():
            self.logger.debug(f"Ajustando {small_mask.sum()} scales muy pequeños")
            scales = np.maximum(scales, MIN_SCALE_VALUE)

        # Corregir valores inválidos (NaN o inf)
        invalid_mask = ~np.isfinite(scales)
        if invalid_mask.any():
            self.logger.warning(f"Corrigiendo {invalid_mask.sum()} scales inválidos (NaN/inf)")
            scales[invalid_mask] = MIN_SCALE_VALUE

        return scales

    def _validate_simulation_results(self, total_simulated_costs: np.ndarray) -> np.ndarray:
        """
        Valida los resultados de la simulación y filtra valores inválidos.

        Args:
            total_simulated_costs: Array de costos totales simulados.

        Returns:
            Array filtrado con solo valores válidos.

        Raises:
            RuntimeError: Si todos los resultados son inválidos.
        """
        if len(total_simulated_costs) == 0:
            raise RuntimeError("La simulación produjo un array vacío")

        # Filtrar valores inválidos (NaN, inf)
        valid_mask = np.isfinite(total_simulated_costs)
        invalid_count = (~valid_mask).sum()

        if invalid_count > 0:
            invalid_pct = (invalid_count / len(total_simulated_costs)) * 100
            self.logger.warning(
                f"Filtrando {invalid_count:,} resultados inválidos ({invalid_pct:.2f}%)"
            )
            total_simulated_costs = total_simulated_costs[valid_mask]

        if len(total_simulated_costs) == 0:
            raise RuntimeError(
                "Todos los resultados de simulación son inválidos (NaN/inf). "
                "Verifique los datos de entrada."
            )

        return total_simulated_costs

    def _calculate_statistics(
        self, simulated_costs: np.ndarray
    ) -> Dict[str, Optional[float]]:
        """
        Calcula estadísticas descriptivas completas de la simulación.

        Incluye medidas de tendencia central, dispersión, forma de la distribución,
        métricas de riesgo (VaR, CVaR) e intervalos de confianza.

        Args:
            simulated_costs: Array con costos simulados válidos.

        Returns:
            Diccionario con estadísticas sanitizadas.
        """
        n = len(simulated_costs)

        # Estadísticas básicas (usando ddof=1 para estimador insesgado)
        statistics: Dict[str, Optional[float]] = {
            "mean": sanitize_value(float(np.mean(simulated_costs))),
            "median": sanitize_value(float(np.median(simulated_costs))),
            "std_dev": sanitize_value(float(np.std(simulated_costs, ddof=1))),
            "variance": sanitize_value(float(np.var(simulated_costs, ddof=1))),
            "min": sanitize_value(float(np.min(simulated_costs))),
            "max": sanitize_value(float(np.max(simulated_costs))),
            "skewness": self._calculate_skewness(simulated_costs),
            "kurtosis": self._calculate_kurtosis(simulated_costs),
        }

        # Percentiles configurados
        try:
            percentile_values = np.percentile(simulated_costs, self.config.percentiles)
            for percentile, value in zip(self.config.percentiles, percentile_values):
                statistics[f"percentile_{percentile}"] = sanitize_value(float(value))
        except Exception as e:
            self.logger.error(f"Error calculando percentiles: {e}")
            for percentile in self.config.percentiles:
                statistics[f"percentile_{percentile}"] = None

        # Métricas de riesgo: VaR y CVaR
        for confidence in [90, 95, 99]:
            statistics[f"var_{confidence}"] = self._calculate_percentile(
                simulated_costs, confidence
            )
            statistics[f"cvar_{confidence}"] = self._calculate_cvar(
                simulated_costs, confidence
            )

        # Intervalos de confianza
        statistics["ci_90_lower"] = statistics.get("percentile_5")
        statistics["ci_90_upper"] = statistics.get("percentile_95")
        statistics["ci_50_lower"] = statistics.get("percentile_25")
        statistics["ci_50_upper"] = statistics.get("percentile_75")

        # Coeficiente de variación (CV = σ/μ)
        mean_val = statistics.get("mean")
        std_val = statistics.get("std_dev")

        if mean_val and std_val and mean_val > 0:
            cv = std_val / mean_val
            statistics["coefficient_of_variation"] = sanitize_value(cv)

            # Advertencia si CV es muy alto
            if cv > CV_WARNING_THRESHOLD:
                self.logger.warning(
                    f"⚠️ Coeficiente de variación alto ({cv:.2f}). "
                    f"Alta dispersión en los datos."
                )
        else:
            statistics["coefficient_of_variation"] = None

        # Rango intercuartílico (IQR = P75 - P25)
        p25 = statistics.get("percentile_25")
        p75 = statistics.get("percentile_75")
        statistics["iqr"] = (
            sanitize_value(p75 - p25) if p25 is not None and p75 is not None else None
        )

        return statistics

    def _calculate_skewness(self, data: np.ndarray) -> Optional[float]:
        """
        Calcula la asimetría (skewness) de los datos.

        Skewness > 0: cola derecha más larga (distribución sesgada a la derecha)
        Skewness < 0: cola izquierda más larga (distribución sesgada a la izquierda)
        Skewness ≈ 0: distribución aproximadamente simétrica

        Args:
            data: Array de datos.

        Returns:
            Skewness o None si hay error.
        """
        try:
            n = len(data)
            if n < 3:
                return None

            mean = np.mean(data)
            std = np.std(data, ddof=1)

            if std == 0:
                return 0.0

            # Momento central estandarizado de orden 3
            m3 = np.mean(((data - mean) / std) ** 3)

            # Corrección de sesgo (Fisher)
            correction = np.sqrt(n * (n - 1)) / (n - 2)
            skewness = m3 * correction

            return sanitize_value(float(skewness))
        except Exception:
            return None

    def _calculate_kurtosis(self, data: np.ndarray) -> Optional[float]:
        """
        Calcula la curtosis (excess kurtosis) de los datos.

        Kurtosis > 0: colas más pesadas que normal (leptocúrtica)
        Kurtosis < 0: colas más ligeras que normal (platicúrtica)
        Kurtosis ≈ 0: similar a distribución normal (mesocúrtica)

        Args:
            data: Array de datos.

        Returns:
            Excess kurtosis o None si hay error.
        """
        try:
            n = len(data)
            if n < 4:
                return None

            mean = np.mean(data)
            std = np.std(data, ddof=1)

            if std == 0:
                return 0.0

            # Momento central estandarizado de orden 4
            m4 = np.mean(((data - mean) / std) ** 4)

            # Excess kurtosis (resta 3 para que normal tenga kurtosis = 0)
            kurtosis = m4 - 3

            return sanitize_value(float(kurtosis))
        except Exception:
            return None

    def _calculate_cvar(self, data: np.ndarray, confidence: int = 95) -> Optional[float]:
        """
        Calcula el Conditional Value at Risk (Expected Shortfall).

        CVaR es el promedio de los valores en la cola que exceden el VaR.
        Es una medida de riesgo más conservadora que el VaR porque considera
        la magnitud de las pérdidas extremas, no solo su probabilidad.

        Para costos (mayor = peor), CVaR es el promedio de los costos
        por encima del percentil especificado.

        Args:
            data: Array de datos.
            confidence: Nivel de confianza (90, 95, 99).

        Returns:
            CVaR o None si hay error.
        """
        try:
            var = np.percentile(data, confidence)
            # Valores en la cola (por encima del VaR)
            tail_values = data[data >= var]

            if len(tail_values) == 0:
                return sanitize_value(float(var))

            return sanitize_value(float(np.mean(tail_values)))
        except Exception:
            return None

    def _calculate_percentile(self, data: np.ndarray, percentile: int) -> Optional[float]:
        """Calcula un percentil específico."""
        try:
            return sanitize_value(float(np.percentile(data, percentile)))
        except Exception:
            return None

    def _create_metadata(
        self,
        df_valid: pd.DataFrame,
        total_items: int,
        discarded_items: int,
        simulations_completed: int,
    ) -> Dict[str, Any]:
        """
        Crea metadata detallada sobre el proceso de simulación.

        Args:
            df_valid: DataFrame con datos válidos.
            total_items: Total de items en entrada.
            discarded_items: Items descartados por validación.
            simulations_completed: Número de simulaciones completadas exitosamente.

        Returns:
            Diccionario con metadata sanitizada y organizada.
        """
        base_costs = df_valid["base_cost"].values

        # Calcular estadísticas de costos base de forma segura
        base_std = float(np.std(base_costs, ddof=1)) if len(base_costs) > 1 else 0.0

        metadata = {
            "config": {
                "num_simulations_requested": self.config.num_simulations,
                "volatility_factor": self.config.volatility_factor,
                "distribution": self.config.distribution.value,
                "use_antithetic": self.config.use_antithetic,
                "random_seed": self.config.random_seed,
                "truncate_negative": self.config.truncate_negative,
                "use_float32": self.config.use_float32,
            },
            "data_quality": {
                "total_items_input": total_items,
                "valid_items": len(df_valid),
                "discarded_items": discarded_items,
                "discard_rate": discarded_items / total_items if total_items > 0 else 0,
            },
            "simulation": {
                "num_simulations_completed": simulations_completed,
                "simulation_success_rate": (
                    simulations_completed / self.config.num_simulations
                    if self.config.num_simulations > 0
                    else 0
                ),
            },
            "base_cost_summary": {
                "sum": sanitize_value(float(np.sum(base_costs))),
                "mean": sanitize_value(float(np.mean(base_costs))),
                "std": sanitize_value(base_std),
                "min": sanitize_value(float(np.min(base_costs))),
                "max": sanitize_value(float(np.max(base_costs))),
            },
        }

        return sanitize_value(metadata, recursive=True)

    def _create_no_data_result(
        self, total_items: int, discarded_items: int
    ) -> SimulationResult:
        """Crea un resultado cuando no hay datos válidos suficientes."""
        valid_items = total_items - discarded_items

        self.logger.error(
            f"❌ Datos insuficientes para simulación: "
            f"{valid_items} válidos de {total_items} totales"
        )

        # Crear estadísticas vacías con todas las claves esperadas
        statistics: Dict[str, Optional[float]] = {
            key: None
            for key in [
                "mean", "median", "std_dev", "variance", "min", "max",
                "skewness", "kurtosis", "coefficient_of_variation", "iqr",
                "ci_90_lower", "ci_90_upper", "ci_50_lower", "ci_50_upper",
            ]
        }

        # Agregar percentiles y métricas de riesgo
        for percentile in self.config.percentiles:
            statistics[f"percentile_{percentile}"] = None

        for confidence in [90, 95, 99]:
            statistics[f"var_{confidence}"] = None
            statistics[f"cvar_{confidence}"] = None

        metadata = {
            "num_simulations_requested": self.config.num_simulations,
            "num_simulations_completed": 0,
            "total_items_input": total_items,
            "valid_items": valid_items,
            "discarded_items": discarded_items,
            "error_reason": (
                "No valid data"
                if valid_items == 0
                else f"Insufficient data (minimum required: {MIN_VALID_ITEMS_FOR_SIMULATION})"
            ),
        }

        return SimulationResult(
            status=SimulationStatus.NO_VALID_DATA,
            statistics=statistics,
            metadata=metadata,
        )

    def get_sensitivity_analysis(
        self, apu_details: List[Dict[str, Any]], top_n: int = 10
    ) -> Dict[str, Any]:
        """
        Realiza análisis de sensibilidad para identificar APUs críticos.

        Calcula la contribución de cada APU a la varianza total del proyecto,
        permitiendo identificar los componentes que más impactan la incertidumbre
        total. Útil para focalizar esfuerzos de mitigación de riesgo.

        La contribución a la varianza de cada APU se calcula como:
        V_i = (base_cost_i × volatility)²
        Sensitivity_i = V_i / Σ V_j

        Args:
            apu_details: Lista de datos APU.
            top_n: Número de APUs principales a retornar.

        Returns:
            Diccionario con análisis de sensibilidad:
            - top_contributors: Lista de los APUs más importantes
            - total_variance: Varianza total del proyecto
            - num_apus_analyzed: Número de APUs analizados
            - concentration_index: Índice de concentración (Herfindahl)
        """
        try:
            df_valid, _ = self._prepare_data(apu_details)

            if len(df_valid) == 0:
                return {"error": "No valid data for sensitivity analysis"}

            base_costs = df_valid["base_cost"].values
            n_apus = len(base_costs)

            # Calcular varianza individual de cada APU
            individual_variances = (base_costs * self.config.volatility_factor) ** 2
            total_variance = np.sum(individual_variances)

            # Índices de sensibilidad (contribución a la varianza total)
            if total_variance > 0:
                sensitivity_indices = individual_variances / total_variance
            else:
                sensitivity_indices = np.zeros(n_apus)

            # Ordenar por importancia (mayor a menor)
            sorted_indices = np.argsort(sensitivity_indices)[::-1]

            # Calcular índice de concentración Herfindahl-Hirschman
            hhi = np.sum(sensitivity_indices**2)

            # Preparar top contributors
            top_contributors = []
            for rank, idx in enumerate(sorted_indices[:top_n]):
                top_contributors.append(
                    {
                        "rank": rank + 1,
                        "index": int(idx),
                        "base_cost": float(base_costs[idx]),
                        "sensitivity_index": float(sensitivity_indices[idx]),
                        "variance_contribution_pct": float(sensitivity_indices[idx] * 100),
                        "cumulative_contribution_pct": float(
                            np.sum(sensitivity_indices[sorted_indices[: rank + 1]]) * 100
                        ),
                    }
                )

            return {
                "top_contributors": top_contributors,
                "total_variance": float(total_variance),
                "num_apus_analyzed": n_apus,
                "concentration_index_hhi": float(hhi),
                "interpretation": {
                    "hhi_meaning": (
                        "Alto (>0.25): Riesgo concentrado en pocos APUs"
                        if hhi > 0.25
                        else "Bajo (<0.15): Riesgo diversificado"
                        if hhi < 0.15
                        else "Moderado: Concentración media"
                    ),
                },
            }

        except Exception as e:
            self.logger.error(f"Error en análisis de sensibilidad: {e}")
            return {"error": str(e)}


# ============================================================================
# FUNCIÓN DE COMPATIBILIDAD CON API ANTERIOR
# ============================================================================


def run_monte_carlo_simulation(
    apu_details: List[Dict[str, Any]],
    num_simulations: int = DEFAULT_NUM_SIMULATIONS,
    volatility_factor: float = DEFAULT_VOLATILITY,
    min_cost_threshold: float = 0.0,
    log_warnings: bool = False,
) -> Dict[str, Optional[float]]:
    """
    Ejecuta simulación de Monte Carlo (función de compatibilidad legacy).

    ⚠️ DEPRECATION WARNING: Esta función se mantiene por compatibilidad con
    código legacy. Se recomienda usar la clase MonteCarloSimulator directamente
    para mayor control, mejor manejo de errores y acceso a funcionalidad completa
    como diferentes distribuciones, CVaR y análisis de sensibilidad.

    Args:
        apu_details: Lista de diccionarios con claves 'VR_TOTAL' y 'CANTIDAD'.
        num_simulations: Número de simulaciones (default: 1000).
        volatility_factor: Factor de volatilidad 0-1 (default: 0.10).
        min_cost_threshold: Umbral mínimo de costo (default: 0.0).
        log_warnings: Si True, muestra advertencias en stderr (default: False).

    Returns:
        Diccionario con estadísticas básicas:
            - mean: Promedio de costos simulados
            - std_dev: Desviación estándar
            - percentile_5: Percentil 5 (límite inferior del IC 90%)
            - percentile_95: Percentil 95 (límite superior del IC 90%)

        En caso de error, retorna diccionario con valores None.

    Example:
        >>> apu_data = [
        ...     {"VR_TOTAL": 1000, "CANTIDAD": 5},
        ...     {"VR_TOTAL": 2000, "CANTIDAD": 3},
        ... ]
        >>> result = run_monte_carlo_simulation(apu_data, num_simulations=5000)
        >>> print(f"Costo promedio: {result['mean']:.2f}")
    """
    warnings.warn(
        "run_monte_carlo_simulation está deprecada. "
        "Use MonteCarloSimulator directamente para funcionalidad completa.",
        DeprecationWarning,
        stacklevel=2,
    )

    empty_result = {
        "mean": None,
        "std_dev": None,
        "percentile_5": None,
        "percentile_95": None,
    }

    try:
        if not isinstance(apu_details, list) or len(apu_details) == 0:
            return empty_result

        config = MonteCarloConfig(
            num_simulations=num_simulations,
            volatility_factor=volatility_factor,
            min_cost_threshold=min_cost_threshold,
            percentiles=[5, 95],
            check_convergence=False,  # Desactivar para mantener comportamiento legacy
        )

        logger = None
        if log_warnings:
            logger = logging.getLogger(f"{__name__}.legacy")
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setLevel(logging.WARNING)
                logger.addHandler(handler)
            logger.setLevel(logging.WARNING)

        simulator = MonteCarloSimulator(config=config, logger=logger)
        result = simulator.run_simulation(apu_details)

        return {
            "mean": result.statistics.get("mean"),
            "std_dev": result.statistics.get("std_dev"),
            "percentile_5": result.statistics.get("percentile_5"),
            "percentile_95": result.statistics.get("percentile_95"),
        }

    except Exception as e:
        if log_warnings:
            logging.getLogger(__name__).error(f"Error en simulación: {str(e)}")
        return empty_result


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Clases principales
    "MonteCarloSimulator",
    "MonteCarloConfig",
    "SimulationResult",
    # Enums
    "SimulationStatus",
    "DistributionType",
    # Métricas
    "ConvergenceMetrics",
    # Funciones
    "run_monte_carlo_simulation",
    "sanitize_value",
    "estimate_memory_usage",
    "validate_required_keys",
    "is_numeric_valid",
    "calculate_convergence_metrics",
]