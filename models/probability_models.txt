"""
Modelos de probabilidad para análisis de costos APU.

Implementa simulaciones de Monte Carlo para estimar distribuciones
de costos considerando incertidumbre y variabilidad.
"""

import logging
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
    ERROR = "error"


# Constantes de validación
DEFAULT_NUM_SIMULATIONS = 1000
MIN_NUM_SIMULATIONS = 100
MAX_NUM_SIMULATIONS = 1_000_000
DEFAULT_VOLATILITY = 0.10
MAX_VOLATILITY = 1.0
MIN_VOLATILITY = 0.0
MEMORY_WARNING_THRESHOLD = 100_000_000  # 100M elementos
MEMORY_HARD_LIMIT_GB = 10
MIN_VALID_ITEMS_FOR_SIMULATION = 1
MAX_SAFE_FLOAT = 1e15  # Valor máximo seguro para evitar overflow


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
    percentiles: List[int] = field(default_factory=lambda: [5, 25, 50, 75, 95])

    def __post_init__(self):
        """Valida y normaliza la configuración después de la inicialización."""
        # Validar que sea lista antes de cualquier transformación
        if not isinstance(self.percentiles, list):
            raise TypeError("percentiles debe ser una lista")

        # Normalizar percentiles: ordenar, eliminar duplicados
        if self.percentiles is not None:
            self.percentiles = sorted(set(self.percentiles))

        self._validate()

    def _validate(self):
        """Valida todos los parámetros de configuración."""
        # Validar num_simulations
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

        # Validar volatility_factor
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
        if self.random_seed is not None:
            if not isinstance(self.random_seed, int):
                raise TypeError("random_seed debe ser int o None")
            if self.random_seed < 0:
                raise ValueError("random_seed no puede ser negativo")

        # Validar percentiles
        if not isinstance(self.percentiles, list):
            raise TypeError("percentiles debe ser una lista")

        if len(self.percentiles) == 0:
            raise ValueError("percentiles no puede estar vacía")

        if not all(isinstance(p, int) for p in self.percentiles):
            raise ValueError("Todos los percentiles deben ser enteros")

        if not all(0 <= p <= 100 for p in self.percentiles):
            raise ValueError("Todos los percentiles deben estar entre 0 y 100")


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
            "statistics": self.statistics.copy(),
            "metadata": self.metadata.copy(),
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


def sanitize_value(
    value: Any, recursive: bool = True
) -> Optional[Union[float, int, str, list, dict]]:
    """
    Sanitiza valores para serialización, convirtiendo NaN/inf a None.

    Maneja múltiples tipos de valores especiales de numpy/pandas y los
    convierte a valores seguros para JSON u otros formatos de serialización.

    Args:
        value: Valor a sanitizar (cualquier tipo).
        recursive: Si True, sanitiza recursivamente colecciones.

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
    # Sanitizar listas recursivamente
    if isinstance(value, list):
        if recursive:
            return [sanitize_value(item, recursive=True) for item in value]
        return value

    # Sanitizar tuplas recursivamente
    if isinstance(value, tuple):
        if recursive:
            return tuple(sanitize_value(item, recursive=True) for item in value)
        return value

    # Sanitizar diccionarios recursivamente
    if isinstance(value, dict):
        if recursive:
            return {k: sanitize_value(v, recursive=True) for k, v in value.items()}
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

    # Verificar si es NaN o NA usando pandas (maneja más casos)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    # Verificar si es infinito
    try:
        if np.isinf(float(value)):
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

    # Retornar el valor original si no necesita conversión
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

    # Arrays auxiliares (base_costs, scales, etc.)
    # Estimamos 10 arrays auxiliares del tamaño de num_apus
    auxiliary_size = num_apus * 8 * 10

    # Array de resultados
    results_size = num_simulations * 8

    # Overhead de Python/numpy (aproximado 20%)
    overhead = (matrix_size + auxiliary_size + results_size) * 0.2

    return int(matrix_size + auxiliary_size + results_size + overhead)


def is_numeric_valid(value: Any) -> bool:
    """
    Verifica si un valor es numérico válido (no NaN, no inf, finito).

    Args:
        value: Valor a verificar.

    Returns:
        True si es un número válido y finito.
    """
    try:
        if pd.isna(value):
            return False
        float_value = float(value)
        return np.isfinite(float_value)
    except (TypeError, ValueError, OverflowError):
        return False


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

    # Aliases de columnas soportados
    VR_TOTAL_ALIASES = ["VR_TOTAL", "valor_total", "vr_total"]
    CANTIDAD_ALIASES = ["CANTIDAD", "cantidad"]

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

        # Inicializar generador de números aleatorios
        self.rng = np.random.default_rng(self.config.random_seed)

        self.logger.info(
            f"MonteCarloSimulator inicializado: "
            f"simulations={self.config.num_simulations:,}, "
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
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def run_simulation(self, apu_details: List[Dict[str, Any]]) -> SimulationResult:
        """
        Ejecuta la simulación de Monte Carlo.

        Args:
            apu_details: Lista de diccionarios con claves 'VR_TOTAL' y 'CANTIDAD'
                        (o sus aliases).

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

            # 3. Verificar que hay datos válidos suficientes
            if len(df_valid) < MIN_VALID_ITEMS_FOR_SIMULATION:
                return self._create_no_data_result(
                    total_items=len(apu_details), discarded_items=discarded_count
                )

            # 4. Verificar uso de memoria
            self._check_memory_requirements(len(df_valid))

            # 5. Ejecutar simulación
            simulated_costs = self._execute_simulation(df_valid)

            # 6. Validar resultados de simulación
            if len(simulated_costs) < self.config.num_simulations * 0.5:
                self.logger.warning(
                    f"Se perdieron más del 50% de simulaciones por valores inválidos: "
                    f"{len(simulated_costs)}/{self.config.num_simulations}"
                )

            # 7. Calcular estadísticas
            statistics = self._calculate_statistics(simulated_costs)

            # 8. Preparar metadata
            metadata = self._create_metadata(
                df_valid=df_valid,
                total_items=len(apu_details),
                discarded_items=discarded_count,
                simulations_completed=len(simulated_costs),
            )

            self.logger.info(
                f"Simulación completada exitosamente: "
                f"{len(df_valid)} APUs válidos, "
                f"{len(simulated_costs):,} simulaciones, "
                f"costo medio={statistics.get('mean', 0):,.2f}"
            )

            return SimulationResult(
                status=SimulationStatus.SUCCESS,
                statistics=statistics,
                metadata=metadata,
                raw_results=simulated_costs,
            )

        except (TypeError, ValueError) as e:
            self.logger.error(f"Error de validación en simulación: {str(e)}")
            raise
        except MemoryError as e:
            self.logger.error(f"Error de memoria en simulación: {str(e)}")
            return SimulationResult(
                status=SimulationStatus.ERROR,
                statistics={},
                metadata={"error": f"MemoryError: {str(e)}"},
            )
        except Exception as e:
            self.logger.error(f"Error inesperado en simulación: {str(e)}", exc_info=True)
            return SimulationResult(
                status=SimulationStatus.ERROR,
                statistics={},
                metadata={"error": str(e), "error_type": type(e).__name__},
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
                f"apu_details debe ser una lista, recibido: {type(apu_details).__name__}"
            )

        if len(apu_details) == 0:
            raise ValueError("apu_details no puede estar vacía")

        if not all(isinstance(item, dict) for item in apu_details):
            invalid_indices = [
                i for i, item in enumerate(apu_details) if not isinstance(item, dict)
            ]
            raise TypeError(
                f"Todos los elementos de apu_details deben ser diccionarios. "
                f"Elementos inválidos en índices: {invalid_indices[:5]}"
            )

        # Validar que al menos algunos items tengan las claves requeridas
        required_keys_options = [
            self.VR_TOTAL_ALIASES,
            self.CANTIDAD_ALIASES,
        ]

        items_with_required_keys = 0
        for i, item in enumerate(apu_details[:10]):  # Validar primeros 10
            try:
                validate_required_keys(item, required_keys_options, i)
                items_with_required_keys += 1
            except ValueError:
                pass

        if items_with_required_keys == 0:
            raise ValueError(
                f"Ninguno de los primeros elementos contiene las claves requeridas. "
                f"Se esperan claves como: {self.VR_TOTAL_ALIASES} y {self.CANTIDAD_ALIASES}"
            )

    def _prepare_data(self, apu_details: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, int]:
        """
        Prepara y limpia los datos para la simulación.

        Args:
            apu_details: Datos brutos de APU.

        Returns:
            Tupla (DataFrame válido, número de items descartados).
        """
        self.logger.debug(f"Preparando {len(apu_details)} registros APU")

        # 1. Crear DataFrame
        df = pd.DataFrame(apu_details).copy()
        initial_rows = len(df)

        # 2. Normalizar columnas (coalesce de aliases)
        df = self._normalize_columns(df)

        # 3. Convertir a numérico
        df = self._convert_to_numeric(df)

        # 4. Filtrar valores válidos
        df_valid = self._filter_valid_rows(df)

        # 5. Calcular costo base
        df_valid = self._calculate_base_cost(df_valid)

        # 6. Filtrar costos base válidos
        df_valid = self._filter_valid_base_costs(df_valid)

        # 7. Calcular items descartados
        discarded_count = initial_rows - len(df_valid)

        if discarded_count > 0:
            discard_rate = (discarded_count / initial_rows) * 100
            self.logger.warning(
                f"Descartados {discarded_count}/{initial_rows} items "
                f"({discard_rate:.1f}%) por valores inválidos o umbrales no cumplidos"
            )

        return df_valid, discarded_count

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normaliza nombres de columnas usando aliases.

        Args:
            df: DataFrame con columnas potencialmente con aliases.

        Returns:
            DataFrame con columnas normalizadas a VR_TOTAL y CANTIDAD.
        """
        df = df.copy()

        # Encontrar columnas que coincidan con aliases
        vr_total_col = None
        cantidad_col = None

        for col in df.columns:
            if col in self.VR_TOTAL_ALIASES:
                vr_total_col = col
                break

        for col in df.columns:
            if col in self.CANTIDAD_ALIASES:
                cantidad_col = col
                break

        # Si no encontramos las columnas, crear columnas vacías
        if vr_total_col is None:
            self.logger.warning("No se encontró columna VR_TOTAL, creando columna vacía")
            df["VR_TOTAL"] = np.nan
        elif vr_total_col != "VR_TOTAL":
            df = df.rename(columns={vr_total_col: "VR_TOTAL"})

        if cantidad_col is None:
            self.logger.warning("No se encontró columna CANTIDAD, creando columna vacía")
            df["CANTIDAD"] = np.nan
        elif cantidad_col != "CANTIDAD":
            df = df.rename(columns={cantidad_col: "CANTIDAD"})

        return df

    def _convert_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convierte columnas a tipo numérico.

        Args:
            df: DataFrame con columnas VR_TOTAL y CANTIDAD.

        Returns:
            DataFrame con columnas convertidas a numérico.
        """
        df = df.copy()

        # Convertir a numérico, forzando errores a NaN
        for col in ["VR_TOTAL", "CANTIDAD"]:
            original_nulls = df[col].isna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            new_nulls = df[col].isna().sum()

            conversion_failures = new_nulls - original_nulls
            if conversion_failures > 0:
                self.logger.warning(
                    f"{col}: {conversion_failures} valores no numéricos convertidos a NaN"
                )

        return df

    def _filter_valid_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra filas con valores válidos que cumplen umbrales.

        Args:
            df: DataFrame con columnas numéricas.

        Returns:
            DataFrame filtrado.
        """
        initial_count = len(df)

        # Crear máscara de validez
        valid_mask = (
            df["VR_TOTAL"].notna()
            & df["CANTIDAD"].notna()
            & np.isfinite(df["VR_TOTAL"])
            & np.isfinite(df["CANTIDAD"])
            & (df["VR_TOTAL"] > self.config.min_cost_threshold)
            & (df["CANTIDAD"] > self.config.min_quantity_threshold)
        )

        df_valid = df[valid_mask].copy()

        filtered_count = initial_count - len(df_valid)
        if filtered_count > 0:
            self.logger.debug(
                f"Filtrados {filtered_count} registros por valores inválidos o umbrales"
            )

        return df_valid

    def _calculate_base_cost(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula el costo base (VR_TOTAL * CANTIDAD).

        Args:
            df: DataFrame con VR_TOTAL y CANTIDAD válidos.

        Returns:
            DataFrame con columna base_cost añadida.
        """
        df = df.copy()

        # Calcular costo base
        df["base_cost"] = df["VR_TOTAL"] * df["CANTIDAD"]

        # Detectar overflows (valores muy grandes)
        overflow_mask = np.abs(df["base_cost"]) > MAX_SAFE_FLOAT
        overflow_count = overflow_mask.sum()

        if overflow_count > 0:
            self.logger.warning(
                f"Se detectaron {overflow_count} valores de base_cost "
                f"extremadamente grandes (posible overflow)"
            )
            # Marcar como NaN para filtrar después
            df.loc[overflow_mask, "base_cost"] = np.nan

        return df

    def _filter_valid_base_costs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filtra costos base válidos (finitos, no NaN, no cero).

        Args:
            df: DataFrame con columna base_cost.

        Returns:
            DataFrame filtrado.
        """
        initial_count = len(df)

        # Filtrar base_cost válidos
        valid_mask = (
            df["base_cost"].notna() & np.isfinite(df["base_cost"]) & (df["base_cost"] != 0)
        )

        df_valid = df[valid_mask].copy()

        filtered_count = initial_count - len(df_valid)
        if filtered_count > 0:
            self.logger.warning(
                f"Se descartaron {filtered_count} items por base_cost "
                f"inválido (NaN/inf/cero)"
            )

        return df_valid

    def _check_memory_requirements(self, num_apus: int) -> None:
        """
        Verifica los requisitos de memoria para la simulación.

        Args:
            num_apus: Número de APUs a simular.

        Raises:
            ValueError: Si la simulación requiere demasiada memoria.
        """
        estimated_memory = estimate_memory_usage(self.config.num_simulations, num_apus)

        # Convertir a MB para logging
        estimated_mb = estimated_memory / (1024 * 1024)

        self.logger.debug(
            f"Memoria estimada para simulación: {estimated_mb:.2f} MB "
            f"({self.config.num_simulations:,} sims × {num_apus} APUs)"
        )

        # Advertir si es mucha memoria
        if estimated_memory > MEMORY_WARNING_THRESHOLD:
            self.logger.warning(
                f"⚠️  La simulación requerirá ~{estimated_mb:.2f} MB de memoria. "
                f"Considere reducir num_simulations o filtrar más APUs."
            )

        # Límite duro
        if estimated_mb > MEMORY_HARD_LIMIT_GB * 1024:
            raise ValueError(
                f"La simulación requiere {estimated_mb:.2f} MB, excediendo el "
                f"límite de {MEMORY_HARD_LIMIT_GB} GB. Reduzca num_simulations "
                f"o el número de APUs."
            )

    def _execute_simulation(self, df_valid: pd.DataFrame) -> np.ndarray:
        """
        Ejecuta la simulación de Monte Carlo.

        Args:
            df_valid: DataFrame con datos válidos y preparados.

        Returns:
            Array con costos totales simulados (solo valores válidos).

        Raises:
            RuntimeError: Si la simulación no produce resultados válidos.
        """
        # Extraer costos base
        base_costs = df_valid["base_cost"].values

        # Calcular desviaciones estándar (scales)
        scales = base_costs * self.config.volatility_factor

        # Validar y corregir scales
        scales = self._validate_and_correct_scales(scales)

        # Generar matriz de simulación
        self.logger.debug(
            f"Generando matriz de simulación: "
            f"({self.config.num_simulations:,} × {len(base_costs)})"
        )

        try:
            simulated_costs_matrix = self.rng.normal(
                loc=base_costs,
                scale=scales,
                size=(self.config.num_simulations, len(base_costs)),
            )
        except MemoryError as e:
            raise MemoryError(f"Memoria insuficiente para generar matriz de simulación: {e}")

        # Truncar negativos si está configurado
        if self.config.truncate_negative:
            negative_count = (simulated_costs_matrix < 0).sum()
            if negative_count > 0:
                negative_pct = (negative_count / simulated_costs_matrix.size) * 100
                self.logger.debug(
                    f"Truncando {negative_count:,} valores negativos "
                    f"({negative_pct:.2f}%) a 0"
                )
                simulated_costs_matrix = np.maximum(simulated_costs_matrix, 0)

        # Sumar por simulación (por fila)
        total_simulated_costs = simulated_costs_matrix.sum(axis=1)

        # Validar resultados
        total_simulated_costs = self._validate_simulation_results(total_simulated_costs)

        return total_simulated_costs

    def _validate_and_correct_scales(self, scales: np.ndarray) -> np.ndarray:
        """
        Valida y corrige los valores de scale para la distribución normal.

        Args:
            scales: Array de desviaciones estándar.

        Returns:
            Array de scales corregidos.
        """
        # Verificar valores negativos
        negative_mask = scales < 0
        if negative_mask.any():
            negative_count = negative_mask.sum()
            self.logger.warning(
                f"Corrigiendo {negative_count} scales negativos a valor absoluto"
            )
            scales = np.abs(scales)

        # Verificar valores cero o muy pequeños
        min_scale = 1e-10
        zero_mask = scales < min_scale
        if zero_mask.any():
            zero_count = zero_mask.sum()
            self.logger.debug(
                f"Ajustando {zero_count} scales muy pequeños/cero a {min_scale}"
            )
            scales = np.maximum(scales, min_scale)

        # Verificar valores NaN o inf
        invalid_mask = ~np.isfinite(scales)
        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            self.logger.warning(
                f"Reemplazando {invalid_count} scales inválidos (NaN/inf) con valor mínimo"
            )
            scales[invalid_mask] = min_scale

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

        # Contar y filtrar valores inválidos
        valid_mask = np.isfinite(total_simulated_costs)
        invalid_count = (~valid_mask).sum()

        if invalid_count > 0:
            invalid_pct = (invalid_count / len(total_simulated_costs)) * 100
            self.logger.warning(
                f"Se encontraron {invalid_count:,} resultados inválidos "
                f"(NaN/inf) ({invalid_pct:.2f}%), filtrando..."
            )
            total_simulated_costs = total_simulated_costs[valid_mask]

        if len(total_simulated_costs) == 0:
            raise RuntimeError("Todos los resultados de simulación son inválidos (NaN/inf)")

        # Advertir si se perdieron muchas simulaciones
        loss_rate = invalid_count / self.config.num_simulations
        if loss_rate > 0.1:  # Más del 10%
            self.logger.warning(
                f"⚠️  Se perdieron {loss_rate:.1%} de simulaciones por valores inválidos"
            )

        return total_simulated_costs

    def _calculate_statistics(
        self, simulated_costs: np.ndarray
    ) -> Dict[str, Optional[float]]:
        """
        Calcula estadísticas descriptivas de la simulación.

        Args:
            simulated_costs: Array con costos simulados válidos.

        Returns:
            Diccionario con estadísticas sanitizadas.
        """
        # Estadísticas básicas
        statistics = {
            "mean": sanitize_value(float(np.mean(simulated_costs))),
            "median": sanitize_value(float(np.median(simulated_costs))),
            "std_dev": sanitize_value(float(np.std(simulated_costs, ddof=1))),
            "variance": sanitize_value(float(np.var(simulated_costs, ddof=1))),
            "min": sanitize_value(float(np.min(simulated_costs))),
            "max": sanitize_value(float(np.max(simulated_costs))),
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

        # Intervalos de confianza comunes
        statistics["ci_90_lower"] = statistics.get("percentile_5")
        statistics["ci_90_upper"] = statistics.get("percentile_95")
        statistics["ci_50_lower"] = statistics.get("percentile_25")
        statistics["ci_50_upper"] = statistics.get("percentile_75")

        # Coeficiente de variación
        mean_val = statistics.get("mean")
        std_val = statistics.get("std_dev")

        if mean_val and std_val and mean_val > 0:
            cv = std_val / mean_val
            statistics["coefficient_of_variation"] = sanitize_value(cv)
        else:
            statistics["coefficient_of_variation"] = None

        # Rango intercuartílico (IQR)
        p25 = statistics.get("percentile_25")
        p75 = statistics.get("percentile_75")
        if p25 is not None and p75 is not None:
            statistics["iqr"] = sanitize_value(p75 - p25)
        else:
            statistics["iqr"] = None

        return statistics

    def _create_metadata(
        self,
        df_valid: pd.DataFrame,
        total_items: int,
        discarded_items: int,
        simulations_completed: int,
    ) -> Dict[str, Any]:
        """
        Crea metadata sobre el proceso de simulación.

        Args:
            df_valid: DataFrame con datos válidos.
            total_items: Total de items en entrada.
            discarded_items: Items descartados.
            simulations_completed: Número de simulaciones completadas exitosamente.

        Returns:
            Diccionario con metadata sanitizada.
        """
        base_costs = df_valid["base_cost"].values

        metadata = {
            "num_simulations_requested": self.config.num_simulations,
            "num_simulations_completed": simulations_completed,
            "simulation_success_rate": (
                simulations_completed / self.config.num_simulations
                if self.config.num_simulations > 0
                else 0
            ),
            "volatility_factor": self.config.volatility_factor,
            "random_seed": self.config.random_seed,
            "total_items_input": total_items,
            "valid_items": len(df_valid),
            "discarded_items": discarded_items,
            "discard_rate": (discarded_items / total_items if total_items > 0 else 0),
            "min_cost_threshold": self.config.min_cost_threshold,
            "min_quantity_threshold": self.config.min_quantity_threshold,
            "truncate_negative": self.config.truncate_negative,
            "base_cost_sum": sanitize_value(float(np.sum(base_costs))),
            "base_cost_mean": sanitize_value(float(np.mean(base_costs))),
            "base_cost_std": sanitize_value(float(np.std(base_costs, ddof=1))),
            "base_cost_min": sanitize_value(float(np.min(base_costs))),
            "base_cost_max": sanitize_value(float(np.max(base_costs))),
        }

        return sanitize_value(metadata, recursive=True)

    def _create_no_data_result(
        self, total_items: int, discarded_items: int
    ) -> SimulationResult:
        """
        Crea un resultado cuando no hay datos válidos suficientes.

        Args:
            total_items: Total de items en entrada.
            discarded_items: Items descartados.

        Returns:
            SimulationResult con estado NO_VALID_DATA.
        """
        valid_items = total_items - discarded_items

        if valid_items == 0:
            self.logger.error(
                f"❌ No hay datos válidos para simulación: "
                f"{discarded_items}/{total_items} items descartados (100%)"
            )
        else:
            self.logger.warning(
                f"⚠️  Datos insuficientes para simulación: "
                f"solo {valid_items} items válidos (mínimo: {MIN_VALID_ITEMS_FOR_SIMULATION})"
            )

        # Crear estadísticas vacías
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

        statistics.update(
            {
                "ci_90_lower": None,
                "ci_90_upper": None,
                "coefficient_of_variation": None,
            }
        )

        metadata = {
            "num_simulations_requested": self.config.num_simulations,
            "num_simulations_completed": 0,
            "simulation_success_rate": 0.0,
            "total_items_input": total_items,
            "valid_items": valid_items,
            "discarded_items": discarded_items,
            "discard_rate": discarded_items / total_items if total_items > 0 else 0,
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

    ⚠️  ADVERTENCIA: Esta función se mantiene por compatibilidad con código legacy.
    Se recomienda usar la clase MonteCarloSimulator directamente para mayor
    control, mejor manejo de errores y acceso a funcionalidad completa.

    Args:
        apu_details: Lista de diccionarios con claves 'VR_TOTAL' y 'CANTIDAD'.
        num_simulations: Número de simulaciones (default: 1000).
        volatility_factor: Factor de volatilidad 0-1 (default: 0.10).
        min_cost_threshold: Umbral mínimo de costo (default: 0.0).
        log_warnings: Si True, muestra advertencias en stderr (default: False).

    Returns:
        Diccionario con estadísticas:
            - mean: Promedio de costos simulados
            - std_dev: Desviación estándar
            - percentile_5: Percentil 5
            - percentile_95: Percentil 95

        En caso de error, retorna diccionario con valores None.

    Raises:
        ValueError: Si los parámetros son inválidos (solo si son muy graves).
        TypeError: Si los tipos de datos son incorrectos (solo si son muy graves).

    Example:
        >>> apu_data = [
        ...     {"VR_TOTAL": 1000, "CANTIDAD": 5},
        ...     {"VR_TOTAL": 2000, "CANTIDAD": 3},
        ... ]
        >>> result = run_monte_carlo_simulation(apu_data, num_simulations=5000)
        >>> print(f"Costo promedio: {result['mean']:.2f}")
    """
    try:
        # Validar parámetros básicos antes de crear configuración
        if not isinstance(apu_details, list):
            raise TypeError(
                f"apu_details debe ser lista, recibido: {type(apu_details).__name__}"
            )

        if len(apu_details) == 0:
            raise ValueError("apu_details no puede estar vacía")

        # Crear configuración
        config = MonteCarloConfig(
            num_simulations=num_simulations,
            volatility_factor=volatility_factor,
            min_cost_threshold=min_cost_threshold,
            percentiles=[5, 95],  # Solo los necesarios para compatibilidad
        )

        # Configurar logger
        logger = None
        if log_warnings:
            logger = logging.getLogger(f"{__name__}.legacy")
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter("%(levelname)s: %(message)s")
                handler.setFormatter(formatter)
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

    except (TypeError, ValueError) as e:
        # Errores críticos de validación se capturan para compatibilidad legacy
        if log_warnings:
            logging.getLogger(__name__).error(f"ERROR en simulación: {str(e)}")

        return {
            "mean": None,
            "std_dev": None,
            "percentile_5": None,
            "percentile_95": None,
        }

    except Exception as e:
        # Otros errores se capturan y retornan estructura vacía
        if log_warnings:
            logging.getLogger(__name__).error(f"ERROR inesperado en simulación: {str(e)}")

        # Retornar estructura vacía en caso de error (compatibilidad legacy)
        return {
            "mean": None,
            "std_dev": None,
            "percentile_5": None,
            "percentile_95": None,
        }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "MonteCarloSimulator",
    "MonteCarloConfig",
    "SimulationResult",
    "SimulationStatus",
    "run_monte_carlo_simulation",
    "sanitize_value",
    "estimate_memory_usage",
    "validate_required_keys",
    "is_numeric_valid",
]
