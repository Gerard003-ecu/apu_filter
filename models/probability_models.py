"""
Modelos de probabilidad para análisis de costos APU.

Implementa simulaciones de Monte Carlo para estimar distribuciones
de costos considerando incertidumbre y variabilidad.
"""

from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
import numpy as np
import pandas as pd
import logging
from enum import Enum


# ============================================================================
# ENUMS Y CONSTANTES
# ============================================================================

class SimulationStatus(Enum):
    """Estados posibles de una simulación."""
    SUCCESS = "success"
    NO_VALID_DATA = "no_valid_data"
    INSUFFICIENT_DATA = "insufficient_data"
    ERROR = "error"


# Constantes de validación
DEFAULT_NUM_SIMULATIONS = 1000
MIN_NUM_SIMULATIONS = 100
MAX_NUM_SIMULATIONS = 1_000_000
DEFAULT_VOLATILITY = 0.10
MAX_VOLATILITY = 1.0
MIN_VOLATILITY = 0.0
MEMORY_WARNING_THRESHOLD = 100_000_000  # 100M de elementos en matriz


# ============================================================================
# CONFIGURACIÓN
# ============================================================================

@dataclass
class MonteCarloConfig:
    """
    Configuración para simulaciones de Monte Carlo.
    
    Attributes:
        num_simulations: Número de iteraciones de simulación.
        volatility_factor: Factor de volatilidad (0-1) para la desviación.
        min_cost_threshold: Costo mínimo para considerar un APU válido.
        min_quantity_threshold: Cantidad mínima para considerar válida.
        random_seed: Semilla para reproducibilidad (None = aleatorio).
        truncate_negative: Si True, trunca costos negativos a 0.
        percentiles: Lista de percentiles a calcular.
    """
    num_simulations: int = DEFAULT_NUM_SIMULATIONS
    volatility_factor: float = DEFAULT_VOLATILITY
    min_cost_threshold: float = 0.0
    min_quantity_threshold: float = 0.0
    random_seed: Optional[int] = None
    truncate_negative: bool = True
    percentiles: List[int] = None
    
    def __post_init__(self):
        """Valida la configuración después de la inicialización."""
        if self.percentiles is None:
            self.percentiles = [5, 25, 50, 75, 95]
        
        self._validate()
    
    def _validate(self):
        """Valida todos los parámetros de configuración."""
        # Validar num_simulations
        if not isinstance(self.num_simulations, int):
            raise TypeError(
                f"num_simulations debe ser int, recibido: {type(self.num_simulations).__name__}"
            )
        
        if not (MIN_NUM_SIMULATIONS <= self.num_simulations <= MAX_NUM_SIMULATIONS):
            raise ValueError(
                f"num_simulations debe estar entre {MIN_NUM_SIMULATIONS} y {MAX_NUM_SIMULATIONS}, "
                f"recibido: {self.num_simulations}"
            )
        
        # Validar volatility_factor
        if not isinstance(self.volatility_factor, (int, float)):
            raise TypeError(
                f"volatility_factor debe ser numérico, recibido: {type(self.volatility_factor).__name__}"
            )
        
        if not (MIN_VOLATILITY <= self.volatility_factor <= MAX_VOLATILITY):
            raise ValueError(
                f"volatility_factor debe estar entre {MIN_VOLATILITY} y {MAX_VOLATILITY}, "
                f"recibido: {self.volatility_factor}"
            )
        
        # Validar umbrales
        if not isinstance(self.min_cost_threshold, (int, float)):
            raise TypeError("min_cost_threshold debe ser numérico")
        
        if not isinstance(self.min_quantity_threshold, (int, float)):
            raise TypeError("min_quantity_threshold debe ser numérico")
        
        if self.min_cost_threshold < 0:
            raise ValueError("min_cost_threshold no puede ser negativo")
        
        if self.min_quantity_threshold < 0:
            raise ValueError("min_quantity_threshold no puede ser negativo")
        
        # Validar random_seed
        if self.random_seed is not None and not isinstance(self.random_seed, int):
            raise TypeError("random_seed debe ser int o None")
        
        # Validar percentiles
        if not isinstance(self.percentiles, list):
            raise TypeError("percentiles debe ser una lista")
        
        if not all(isinstance(p, int) and 0 <= p <= 100 for p in self.percentiles):
            raise ValueError("Todos los percentiles deben ser enteros entre 0 y 100")


# ============================================================================
# CLASES DE RESULTADO
# ============================================================================

@dataclass
class SimulationResult:
    """
    Resultado de una simulación de Monte Carlo.
    
    Attributes:
        status: Estado de la simulación.
        statistics: Diccionario con estadísticas calculadas.
        metadata: Información sobre el proceso de simulación.
        raw_results: Array con resultados brutos (opcional, para análisis).
    """
    status: SimulationStatus
    statistics: Dict[str, Optional[float]]
    metadata: Dict[str, Any]
    raw_results: Optional[np.ndarray] = None
    
    def to_dict(self, include_raw: bool = False) -> Dict[str, Any]:
        """
        Convierte el resultado a diccionario serializable.
        
        Args:
            include_raw: Si True, incluye resultados brutos.
            
        Returns:
            Diccionario con el resultado.
        """
        result = {
            "status": self.status.value,
            "statistics": self.statistics,
            "metadata": self.metadata
        }
        
        if include_raw and self.raw_results is not None:
            result["raw_results"] = self.raw_results.tolist()
        
        return result
    
    def is_successful(self) -> bool:
        """Retorna True si la simulación fue exitosa."""
        return self.status == SimulationStatus.SUCCESS


# ============================================================================
# UTILIDADES
# ============================================================================

def sanitize_value(value: Any) -> Optional[Union[float, int, str, list, tuple]]:
    """
    Sanitiza valores para serialización, convirtiendo NaN/inf a None.
    
    Maneja múltiples tipos de valores especiales de numpy/pandas y los
    convierte a valores seguros para JSON u otros formatos de serialización.
    
    Args:
        value: Valor a sanitizar (cualquier tipo).
    
    Returns:
        - None si el valor es NaN, NA, inf o -inf
        - El valor convertido a tipo Python nativo en otros casos
        - Colecciones (list, tuple) se retornan sin modificar
    
    Examples:
        >>> sanitize_value(np.nan)
        None
        >>> sanitize_value(np.inf)
        None
        >>> sanitize_value(np.float64(10.5))
        10.5
        >>> sanitize_value([1, 2, 3])
        [1, 2, 3]
    """
    # Colecciones se retornan sin modificar
    if isinstance(value, (list, tuple, dict)):
        return value
    
    # Strings se retornan sin modificar
    if isinstance(value, str):
        return value
    
    # Verificar si es NaN o NA
    if pd.isna(value):
        return None
    
    # Verificar si es infinito
    try:
        if np.isinf(value):
            return None
    except (TypeError, ValueError):
        # El valor no es numérico, continuar
        pass
    
    # Convertir tipos numpy a tipos nativos de Python
    if isinstance(value, (np.floating, np.complexfloating)):
        return float(value)
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.bool_):
        return bool(value)
    
    # Retornar el valor original si no necesita conversión
    return value


def validate_apu_data_structure(
    apu_details: Any,
    logger: Optional[logging.Logger] = None
) -> bool:
    """
    Valida la estructura básica de los datos APU.
    
    Args:
        apu_details: Datos a validar.
        logger: Logger opcional para mensajes de error.
    
    Returns:
        True si la estructura es válida, False en caso contrario.
    """
    # Verificar que sea una lista
    if not isinstance(apu_details, list):
        if logger:
            logger.error(
                f"apu_details debe ser una lista, recibido: {type(apu_details).__name__}"
            )
        return False
    
    # Verificar que no esté vacía
    if len(apu_details) == 0:
        if logger:
            logger.warning("apu_details está vacía")
        return False
    
    # Verificar que todos los elementos sean diccionarios
    if not all(isinstance(item, dict) for item in apu_details):
        if logger:
            logger.error("Todos los elementos de apu_details deben ser diccionarios")
        return False
    
    return True


def estimate_memory_usage(num_simulations: int, num_apus: int) -> int:
    """
    Estima el uso de memoria en bytes para la simulación.
    
    Args:
        num_simulations: Número de simulaciones.
        num_apus: Número de APUs a simular.
    
    Returns:
        Estimación de bytes necesarios.
    """
    # float64 = 8 bytes
    # Matriz principal: num_simulations × num_apus
    matrix_size = num_simulations * num_apus * 8
    
    # Arrays auxiliares (aproximación)
    auxiliary_size = num_apus * 8 * 5  # base_costs, scales, etc.
    
    # Array de resultados
    results_size = num_simulations * 8
    
    return matrix_size + auxiliary_size + results_size


# ============================================================================
# CLASE PRINCIPAL
# ============================================================================

class MonteCarloSimulator:
    """
    Simulador de Monte Carlo para análisis de costos APU.
    
    Implementa simulaciones estocásticas para estimar distribuciones
    de costos totales considerando incertidumbre en los valores.
    
    Attributes:
        config: Configuración de la simulación.
        logger: Logger para registro de eventos.
        rng: Generador de números aleatorios.
    """
    
    def __init__(
        self,
        config: Optional[MonteCarloConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Inicializa el simulador.
        
        Args:
            config: Configuración personalizada (usa default si es None).
            logger: Logger para eventos (crea uno básico si es None).
        """
        self.config = config or MonteCarloConfig()
        self.logger = logger or self._create_default_logger()
        
        # Inicializar generador de números aleatorios
        self.rng = np.random.default_rng(self.config.random_seed)
        
        self.logger.info(
            f"MonteCarloSimulator inicializado: "
            f"simulations={self.config.num_simulations}, "
            f"volatility={self.config.volatility_factor:.2%}, "
            f"seed={self.config.random_seed}"
        )
    
    @staticmethod
    def _create_default_logger() -> logging.Logger:
        """Crea un logger básico si no se proporciona uno."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def run_simulation(
        self,
        apu_details: List[Dict[str, Any]]
    ) -> SimulationResult:
        """
        Ejecuta la simulación de Monte Carlo.
        
        Args:
            apu_details: Lista de diccionarios con claves 'VR_TOTAL' y 'CANTIDAD'.
        
        Returns:
            SimulationResult con estadísticas y metadata.
        
        Raises:
            TypeError: Si apu_details no tiene el tipo correcto.
            ValueError: Si los datos son inválidos estructuralmente.
        """
        try:
            # 1. Validar estructura de datos
            self._validate_input_data(apu_details)
            
            # 2. Preparar datos
            df_valid, discarded_count = self._prepare_data(apu_details)
            
            # 3. Verificar que hay datos válidos
            if len(df_valid) == 0:
                return self._create_no_data_result(
                    total_items=len(apu_details),
                    discarded_items=discarded_count
                )
            
            # 4. Verificar uso de memoria
            self._check_memory_requirements(len(df_valid))
            
            # 5. Ejecutar simulación
            simulated_costs = self._execute_simulation(df_valid)
            
            # 6. Calcular estadísticas
            statistics = self._calculate_statistics(simulated_costs)
            
            # 7. Preparar metadata
            metadata = self._create_metadata(
                df_valid=df_valid,
                total_items=len(apu_details),
                discarded_items=discarded_count
            )
            
            self.logger.info(
                f"Simulación completada exitosamente: "
                f"{len(df_valid)} APUs válidos, "
                f"costo medio={statistics['mean']:,.2f}"
            )
            
            return SimulationResult(
                status=SimulationStatus.SUCCESS,
                statistics=statistics,
                metadata=metadata,
                raw_results=simulated_costs
            )
            
        except (TypeError, ValueError) as e:
            self.logger.error(f"Error de validación en simulación: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(
                f"Error inesperado en simulación: {str(e)}",
                exc_info=True
            )
            return SimulationResult(
                status=SimulationStatus.ERROR,
                statistics={},
                metadata={"error": str(e)}
            )
    
    def _validate_input_data(self, apu_details: Any) -> None:
        """
        Valida los datos de entrada.
        
        Args:
            apu_details: Datos a validar.
        
        Raises:
            TypeError: Si el tipo no es correcto.
            ValueError: Si la estructura es inválida.
        """
        if not isinstance(apu_details, list):
            raise TypeError(
                f"apu_details debe ser una lista, "
                f"recibido: {type(apu_details).__name__}"
            )
        
        if len(apu_details) == 0:
            raise ValueError("apu_details no puede estar vacía")
        
        if not all(isinstance(item, dict) for item in apu_details):
            raise TypeError(
                "Todos los elementos de apu_details deben ser diccionarios"
            )
    
    def _prepare_data(
        self,
        apu_details: List[Dict[str, Any]]
    ) -> tuple[pd.DataFrame, int]:
        """
        Prepara y limpia los datos para la simulación.
        
        Args:
            apu_details: Datos brutos de APU.
        
        Returns:
            Tupla (DataFrame válido, número de items descartados).
        
        Raises:
            ValueError: Si faltan columnas requeridas.
        """
        # Convertir a DataFrame
        df = pd.DataFrame(apu_details)
        
        # Validar columnas requeridas
        required_cols = {"VR_TOTAL", "CANTIDAD"}
        missing_cols = required_cols - set(df.columns)
        
        if missing_cols:
            raise ValueError(
                f"Faltan columnas requeridas: {sorted(missing_cols)}"
            )
        
        # Convertir a numérico
        df["VR_TOTAL"] = pd.to_numeric(df["VR_TOTAL"], errors="coerce")
        df["CANTIDAD"] = pd.to_numeric(df["CANTIDAD"], errors="coerce")
        
        # Contar valores convertidos a NaN
        vr_total_nulls = df["VR_TOTAL"].isna().sum()
        cantidad_nulls = df["CANTIDAD"].isna().sum()
        
        if vr_total_nulls > 0:
            self.logger.warning(
                f"VR_TOTAL: {vr_total_nulls} valores no numéricos convertidos a NaN"
            )
        
        if cantidad_nulls > 0:
            self.logger.warning(
                f"CANTIDAD: {cantidad_nulls} valores no numéricos convertidos a NaN"
            )
        
        # Filtrar filas válidas
        valid_mask = (
            df["VR_TOTAL"].notna() &
            df["CANTIDAD"].notna() &
            (df["VR_TOTAL"] > self.config.min_cost_threshold) &
            (df["CANTIDAD"] > self.config.min_quantity_threshold)
        )
        
        df_valid = df[valid_mask].copy()
        discarded_count = len(df) - len(df_valid)
        
        # Log de datos descartados
        if discarded_count > 0:
            self.logger.warning(
                f"Se descartaron {discarded_count}/{len(df)} items por no cumplir umbrales: "
                f"VR_TOTAL > {self.config.min_cost_threshold}, "
                f"CANTIDAD > {self.config.min_quantity_threshold}"
            )
        
        # Calcular costo base
        df_valid["base_cost"] = df_valid["VR_TOTAL"] * df_valid["CANTIDAD"]
        
        # Verificar que no haya costos base inválidos
        invalid_base_costs = (
            df_valid["base_cost"].isna() | 
            np.isinf(df_valid["base_cost"])
        ).sum()
        
        if invalid_base_costs > 0:
            self.logger.error(
                f"Se encontraron {invalid_base_costs} costos base inválidos (NaN/inf)"
            )
            df_valid = df_valid[
                df_valid["base_cost"].notna() & 
                np.isfinite(df_valid["base_cost"])
            ].copy()
        
        return df_valid, discarded_count
    
    def _check_memory_requirements(self, num_apus: int) -> None:
        """
        Verifica los requisitos de memoria para la simulación.
        
        Args:
            num_apus: Número de APUs a simular.
        
        Raises:
            MemoryError: Si la simulación requiere demasiada memoria.
        """
        estimated_memory = estimate_memory_usage(
            self.config.num_simulations,
            num_apus
        )
        
        # Convertir a MB para logging
        estimated_mb = estimated_memory / (1024 * 1024)
        
        self.logger.debug(
            f"Memoria estimada para simulación: {estimated_mb:.2f} MB"
        )
        
        # Advertir si es mucha memoria
        if estimated_memory > MEMORY_WARNING_THRESHOLD:
            self.logger.warning(
                f"La simulación requerirá aproximadamente {estimated_mb:.2f} MB de memoria. "
                f"Considere reducir num_simulations o el número de APUs."
            )
        
        # Límite duro (opcional, ajustar según necesidad)
        max_memory_gb = 10
        if estimated_mb > max_memory_gb * 1024:
            raise MemoryError(
                f"La simulación requiere {estimated_mb:.2f} MB, "
                f"excediendo el límite de {max_memory_gb} GB"
            )
    
    def _execute_simulation(self, df_valid: pd.DataFrame) -> np.ndarray:
        """
        Ejecuta la simulación de Monte Carlo.
        
        Args:
            df_valid: DataFrame con datos válidos y preparados.
        
        Returns:
            Array con costos totales simulados.
        """
        # Extraer costos base
        base_costs = df_valid["base_cost"].values
        
        # Calcular desviaciones estándar
        scales = base_costs * self.config.volatility_factor
        
        # Validar que no haya scales inválidos
        if np.any(scales < 0):
            self.logger.warning("Se encontraron scales negativos, usando valor absoluto")
            scales = np.abs(scales)
        
        # Generar simulaciones
        # shape: (num_simulations, num_apus)
        self.logger.debug(
            f"Generando matriz de simulación: "
            f"({self.config.num_simulations}, {len(base_costs)})"
        )
        
        simulated_costs_matrix = self.rng.normal(
            loc=base_costs,
            scale=scales,
            size=(self.config.num_simulations, len(base_costs))
        )
        
        # Truncar negativos si está configurado
        if self.config.truncate_negative:
            negative_count = (simulated_costs_matrix < 0).sum()
            if negative_count > 0:
                self.logger.debug(
                    f"Truncando {negative_count} valores negativos a 0"
                )
                simulated_costs_matrix = np.maximum(simulated_costs_matrix, 0)
        
        # Sumar por simulación (por fila)
        total_simulated_costs = simulated_costs_matrix.sum(axis=1)
        
        # Validar resultados
        if len(total_simulated_costs) == 0:
            raise RuntimeError("La simulación produjo un array vacío")
        
        invalid_results = (
            np.isnan(total_simulated_costs) | 
            np.isinf(total_simulated_costs)
        ).sum()
        
        if invalid_results > 0:
            self.logger.error(
                f"Se encontraron {invalid_results} resultados inválidos (NaN/inf)"
            )
            # Filtrar inválidos
            total_simulated_costs = total_simulated_costs[
                np.isfinite(total_simulated_costs)
            ]
            
            if len(total_simulated_costs) == 0:
                raise RuntimeError(
                    "Todos los resultados de simulación son inválidos"
                )
        
        return total_simulated_costs
    
    def _calculate_statistics(
        self,
        simulated_costs: np.ndarray
    ) -> Dict[str, Optional[float]]:
        """
        Calcula estadísticas descriptivas de la simulación.
        
        Args:
            simulated_costs: Array con costos simulados.
        
        Returns:
            Diccionario con estadísticas.
        """
        statistics = {
            "mean": sanitize_value(float(np.mean(simulated_costs))),
            "median": sanitize_value(float(np.median(simulated_costs))),
            "std_dev": sanitize_value(float(np.std(simulated_costs))),
            "variance": sanitize_value(float(np.var(simulated_costs))),
            "min": sanitize_value(float(np.min(simulated_costs))),
            "max": sanitize_value(float(np.max(simulated_costs))),
        }
        
        # Calcular percentiles configurados
        for percentile in self.config.percentiles:
            key = f"percentile_{percentile}"
            value = float(np.percentile(simulated_costs, percentile))
            statistics[key] = sanitize_value(value)
        
        # Calcular intervalo de confianza 90%
        statistics["ci_90_lower"] = statistics.get("percentile_5")
        statistics["ci_90_upper"] = statistics.get("percentile_95")
        
        # Calcular coeficiente de variación
        if statistics["mean"] and statistics["mean"] > 0:
            cv = statistics["std_dev"] / statistics["mean"]
            statistics["coefficient_of_variation"] = sanitize_value(cv)
        else:
            statistics["coefficient_of_variation"] = None
        
        return statistics
    
    def _create_metadata(
        self,
        df_valid: pd.DataFrame,
        total_items: int,
        discarded_items: int
    ) -> Dict[str, Any]:
        """
        Crea metadata sobre el proceso de simulación.
        
        Args:
            df_valid: DataFrame con datos válidos.
            total_items: Total de items en entrada.
            discarded_items: Items descartados.
        
        Returns:
            Diccionario con metadata.
        """
        base_costs = df_valid["base_cost"].values
        
        return {
            "num_simulations": self.config.num_simulations,
            "volatility_factor": self.config.volatility_factor,
            "random_seed": self.config.random_seed,
            "total_items_input": total_items,
            "valid_items": len(df_valid),
            "discarded_items": discarded_items,
            "discard_rate": discarded_items / total_items if total_items > 0 else 0,
            "min_cost_threshold": self.config.min_cost_threshold,
            "min_quantity_threshold": self.config.min_quantity_threshold,
            "truncate_negative": self.config.truncate_negative,
            "base_cost_sum": sanitize_value(float(np.sum(base_costs))),
            "base_cost_mean": sanitize_value(float(np.mean(base_costs))),
            "base_cost_std": sanitize_value(float(np.std(base_costs))),
        }
    
    def _create_no_data_result(
        self,
        total_items: int,
        discarded_items: int
    ) -> SimulationResult:
        """
        Crea un resultado cuando no hay datos válidos.
        
        Args:
            total_items: Total de items en entrada.
            discarded_items: Items descartados.
        
        Returns:
            SimulationResult con estado NO_VALID_DATA.
        """
        self.logger.warning(
            f"No hay datos válidos para simulación: "
            f"{discarded_items}/{total_items} items descartados"
        )
        
        # Crear estadísticas vacías con None
        statistics = {
            "mean": None,
            "median": None,
            "std_dev": None,
            "variance": None,
            "min": None,
            "max": None,
        }
        
        for percentile in self.config.percentiles:
            statistics[f"percentile_{percentile}"] = None
        
        metadata = {
            "num_simulations": self.config.num_simulations,
            "total_items_input": total_items,
            "valid_items": 0,
            "discarded_items": discarded_items,
            "discard_rate": 1.0 if total_items > 0 else 0,
        }
        
        return SimulationResult(
            status=SimulationStatus.NO_VALID_DATA,
            statistics=statistics,
            metadata=metadata
        )


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
    Ejecuta simulación de Monte Carlo (función de compatibilidad).
    
    NOTA: Esta función se mantiene por compatibilidad con código legacy.
    Se recomienda usar la clase MonteCarloSimulator directamente para
    mayor control y funcionalidad.
    
    Args:
        apu_details: Lista de diccionarios con claves 'VR_TOTAL' y 'CANTIDAD'.
        num_simulations: Número de simulaciones (default: 1000).
        volatility_factor: Factor de volatilidad 0-1 (default: 0.10).
        min_cost_threshold: Umbral mínimo de costo (default: 0.0).
        log_warnings: Si True, muestra advertencias (default: False).
    
    Returns:
        Diccionario con estadísticas:
            - mean: Promedio
            - std_dev: Desviación estándar
            - percentile_5: Percentil 5
            - percentile_95: Percentil 95
    
    Raises:
        ValueError: Si los parámetros son inválidos.
        TypeError: Si los tipos de datos son incorrectos.
    """
    try:
        # Crear configuración
        config = MonteCarloConfig(
            num_simulations=num_simulations,
            volatility_factor=volatility_factor,
            min_cost_threshold=min_cost_threshold,
            percentiles=[5, 95]  # Solo los necesarios para compatibilidad
        )
        
        # Configurar logger
        logger = None
        if log_warnings:
            logger = logging.getLogger(__name__)
            if not logger.handlers:
                handler = logging.StreamHandler()
                logger.addHandler(handler)
                logger.setLevel(logging.WARNING)
        
        # Crear simulador y ejecutar
        simulator = MonteCarloSimulator(config=config, logger=logger)
        result = simulator.run_simulation(apu_details)
        
        # Retornar en formato legacy
        return {
            "mean": result.statistics.get("mean"),
            "std_dev": result.statistics.get("std_dev"),
            "percentile_5": result.statistics.get("percentile_5"),
            "percentile_95": result.statistics.get("percentile_95"),
        }
        
    except Exception as e:
        if log_warnings:
            print(f"ERROR en simulación: {str(e)}")
        
        # Retornar estructura vacía en caso de error (compatibilidad)
        return {
            "mean": None,
            "std_dev": None,
            "percentile_5": None,
            "percentile_95": None,
        }