from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class APUProcessingConfig:
    """Configuración centralizada para el procesamiento de APU."""

    required_columns: set = None
    numeric_columns: set = None
    default_category: str = "INDEFINIDO"
    tolerance_price_variance: float = 0.01  # 1% tolerancia en precios

    def __post_init__(self):
        if self.required_columns is None:
            self.required_columns = {
                "CATEGORIA",
                "DESCRIPCION_INSUMO",
                "CANTIDAD_APU",
                "VALOR_TOTAL_APU",
                "UNIDAD_APU",
                "PRECIO_UNIT_APU",
                "CODIGO_APU",
                "UNIDAD_INSUMO",
                "RENDIMIENTO",
            }

        if self.numeric_columns is None:
            self.numeric_columns = {
                "CANTIDAD_APU",
                "VALOR_TOTAL_APU",
                "PRECIO_UNIT_APU",
                "RENDIMIENTO",
            }


class APUPresenter:
    """
    Procesa y agrupa datos de APU para presentación.

    Características:
    - Validación exhaustiva de datos de entrada
    - Sanitización robusta de datos
    - Agregación inteligente con validación de consistencia
    - Logging detallado de operaciones
    - Manejo de casos edge
    """

    def __init__(self, logger, config: Optional[APUProcessingConfig] = None):
        """
        Inicializa el presentador de APU.

        Args:
            logger: Logger para registro de eventos y errores.
            config: Configuración personalizada (usa default si es None).
        """
        if logger is None:
            raise ValueError("Logger es requerido y no puede ser None")

        self.logger = logger
        self.config = config or APUProcessingConfig()
        self.logger.info("APUPresenter inicializado correctamente")

    def process_apu_details(
        self, apu_details: List[Dict[str, Any]], apu_code: str
    ) -> Dict[str, Any]:
        """
        Procesa los detalles de un APU específico con validaciones robustas.

        Args:
            apu_details: Lista de diccionarios con detalles del APU.
            apu_code: Código identificador del APU (string no vacío).

        Returns:
            Diccionario con estructura:
                {
                    "items": List[Dict],           # Items procesados
                    "desglose": Dict[str, List],   # Organizado por categoría
                    "total_items": int,            # Cantidad de items
                    "metadata": Dict               # Info del procesamiento
                }

        Raises:
            ValueError: Si los datos de entrada son inválidos.
            KeyError: Si faltan columnas requeridas en los datos.
        """
        try:
            # 1. Validaciones de entrada
            self._validate_inputs(apu_details, apu_code)

            self.logger.info(
                f"Iniciando procesamiento de APU: {apu_code} "
                f"con {len(apu_details)} registros"
            )

            # 2. Crear y sanitizar DataFrame
            df = self._create_and_sanitize_dataframe(apu_details, apu_code)

            # 3. Validar esquema de datos
            self._validate_dataframe_schema(df, apu_code)

            # 4. Procesar y agrupar por categorías
            processed_items = self._group_by_category(df)

            if not processed_items:
                self.logger.warning(f"APU {apu_code}: No se generaron items procesados")

            # 5. Organizar desglose por categoría
            desglose = self._organize_breakdown(processed_items)

            # 6. Calcular metadata del procesamiento
            metadata = self._calculate_metadata(df, processed_items)

            self.logger.info(
                f"APU {apu_code} procesado exitosamente. "
                f"Items: {len(processed_items)}, "
                f"Categorías: {len(desglose)}, "
                f"Reducción: {metadata['reduction_rate']:.1%}"
            )

            return {
                "items": processed_items,
                "desglose": desglose,
                "total_items": len(processed_items),
                "metadata": metadata,
            }

        except (ValueError, KeyError) as e:
            self.logger.error(f"Error de validación en APU {apu_code}: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(
                f"Error inesperado procesando APU {apu_code}: {str(e)}", exc_info=True
            )
            raise RuntimeError(f"Fallo en procesamiento de APU {apu_code}") from e

    def _validate_inputs(self, apu_details: Any, apu_code: Any) -> None:
        """
        Valida los parámetros de entrada del método principal.

        Args:
            apu_details: Debe ser una lista no vacía de diccionarios.
            apu_code: Debe ser un string no vacío.

        Raises:
            ValueError: Si alguna validación falla.
        """
        # Validar tipo de apu_details
        if not isinstance(apu_details, list):
            raise ValueError(
                f"apu_details debe ser una lista, recibido: {type(apu_details).__name__}"
            )

        # Validar que no esté vacía
        if not apu_details:
            raise ValueError(f"No se encontraron detalles para el APU {apu_code}")

        # Validar que contenga diccionarios
        if not all(isinstance(item, dict) for item in apu_details):
            raise ValueError("Todos los elementos de apu_details deben ser diccionarios")

        # Validar apu_code
        if not isinstance(apu_code, str) or not apu_code.strip():
            raise ValueError(f"apu_code debe ser un string no vacío, recibido: '{apu_code}'")

    def _create_and_sanitize_dataframe(
        self, apu_details: List[Dict], apu_code: str
    ) -> pd.DataFrame:
        """
        Crea DataFrame desde los detalles y aplica sanitización.

        Args:
            apu_details: Lista de diccionarios con los detalles.
            apu_code: Código del APU para mensajes de error.

        Returns:
            DataFrame sanitizado y validado.

        Raises:
            ValueError: Si no se puede crear el DataFrame.
        """
        try:
            df = pd.DataFrame(apu_details)
        except Exception as e:
            raise ValueError(f"Error creando DataFrame para APU {apu_code}: {str(e)}") from e

        if df.empty:
            raise ValueError(f"DataFrame resultó vacío para APU {apu_code}")

        # Sanitizar datos
        df = self._sanitize_dataframe(df)

        self.logger.debug(f"DataFrame creado: {len(df)} filas, {len(df.columns)} columnas")

        return df

    def _sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitiza el DataFrame aplicando múltiples transformaciones.

        Operaciones:
        - Reemplaza NaN, inf, -inf por None
        - Elimina espacios en strings
        - Convierte y valida columnas numéricas
        - Detecta valores negativos en campos que no deberían tenerlos

        Args:
            df: DataFrame a sanitizar.

        Returns:
            DataFrame sanitizado (copia, no modifica el original).
        """
        df = df.copy()

        # Limpiar strings: eliminar espacios y normalizar
        string_columns = df.select_dtypes(include=["object"]).columns
        for col in string_columns:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

        # Validar y convertir columnas numéricas
        for col in self.config.numeric_columns:
            if col in df.columns:
                # Convertir a numérico
                original_dtype = df[col].dtype
                df[col] = pd.to_numeric(df[col], errors="coerce")

                # Contar conversiones fallidas
                null_count = df[col].isna().sum()
                if null_count > 0:
                    self.logger.warning(
                        f"Columna '{col}': {null_count} valores no numéricos "
                        f"convertidos a None (tipo original: {original_dtype})"
                    )

                # Advertir sobre valores negativos
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    self.logger.warning(
                        f"Columna '{col}': {negative_count} valores negativos encontrados"
                    )

        # Reemplazar valores especiales de forma robusta al final
        # Esto asegura que los NaN introducidos por to_numeric también se manejen
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Convertir todos los np.nan a None.
        # Esto es necesario para que las aserciones 'is None' pasen
        # y puede cambiar el dtype de las columnas a 'object'.
        df = df.astype(object).where(pd.notnull(df), None)

        return df

    def _validate_dataframe_schema(self, df: pd.DataFrame, apu_code: str) -> None:
        """
        Valida que el DataFrame contenga todas las columnas requeridas.

        Args:
            df: DataFrame a validar.
            apu_code: Código del APU para mensajes de error.

        Raises:
            KeyError: Si faltan columnas requeridas.
        """
        current_columns = set(df.columns)
        missing_columns = self.config.required_columns - current_columns

        if missing_columns:
            raise KeyError(f"APU {apu_code}: Columnas faltantes: {sorted(missing_columns)}")

        self.logger.debug(f"Validación de esquema exitosa para APU {apu_code}")

    def _group_by_category(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Agrupa los ítems por categoría y descripción con agregaciones.

        Procesa cada categoría independientemente, agrupando por descripción
        de insumo y aplicando agregaciones (sum, validación de consistencia).

        Args:
            df: DataFrame con los detalles de los insumos.

        Returns:
            Lista de diccionarios con items procesados y consolidados.
        """
        processed = []
        # Incluir valores NaN/None como una categoría a procesar
        categories = df["CATEGORIA"].unique()

        if len(categories) == 0:
            self.logger.warning("No se encontraron categorías válidas en el DataFrame")
            return processed

        self.logger.debug(f"Procesando {len(categories)} categorías")

        for categoria in categories:
            # Manejar el filtrado para valores NaN/None
            if pd.isna(categoria):
                df_categoria = df[df["CATEGORIA"].isna()].copy()
            else:
                df_categoria = df[df["CATEGORIA"] == categoria].copy()

            if df_categoria.empty:
                continue

            try:
                grouped_items = self._aggregate_category_items(df_categoria, categoria)
                processed.extend(grouped_items)

                self.logger.debug(
                    f"Categoría '{categoria}': {len(grouped_items)} items procesados"
                )

            except Exception as e:
                self.logger.error(
                    f"Error procesando categoría '{categoria}': {str(e)}", exc_info=True
                )
                # Continuar con otras categorías
                continue

        return processed

    def _aggregate_category_items(
        self, df_categoria: pd.DataFrame, categoria_name: str
    ) -> List[Dict[str, Any]]:
        """
        Agrega items de una categoría específica.

        Args:
            df_categoria: DataFrame filtrado por categoría.
            categoria_name: Nombre de la categoría (para logging).

        Returns:
            Lista de items agregados para la categoría.
        """
        # Definir estrategias de agregación
        aggregations = {
            "CANTIDAD_APU": "sum",
            "VALOR_TOTAL_APU": "sum",
            "RENDIMIENTO": "sum",
            "UNIDAD_APU": lambda x: self._validate_consistency(x, "UNIDAD_APU"),
            "PRECIO_UNIT_APU": lambda x: self._aggregate_price(x, categoria_name),
            "CATEGORIA": "first",
            "CODIGO_APU": "first",
            "UNIDAD_INSUMO": lambda x: self._validate_consistency(x, "UNIDAD_INSUMO"),
        }

        # Agregar manejo de alertas si la columna existe
        if "alerta" in df_categoria.columns:
            aggregations["alerta"] = self._aggregate_alerts

        # Agrupar por descripción de insumo
        df_grouped = (
            df_categoria.groupby("DESCRIPCION_INSUMO", dropna=False)
            .agg(aggregations)
            .reset_index()
        )

        # Renombrar columnas a formato de salida
        df_grouped.rename(
            columns={
                "DESCRIPCION_INSUMO": "descripcion",
                "CANTIDAD_APU": "cantidad",
                "VALOR_TOTAL_APU": "valor_total",
                "UNIDAD_INSUMO": "unidad",
                "PRECIO_UNIT_APU": "valor_unitario",
                "CATEGORIA": "categoria",
            },
            inplace=True,
        )

        # Convertir a lista de diccionarios y limpiar NaN finales
        items = df_grouped.replace({np.nan: None}).to_dict("records")

        return items

    def _validate_consistency(self, series: pd.Series, field_name: str = "campo") -> Any:
        """
        Valida que todos los valores en una serie sean consistentes.

        Args:
            series: Serie de pandas a validar.
            field_name: Nombre del campo para logging.

        Returns:
            El valor consistente, el primero si hay múltiples, o None.
        """
        unique_values = series.dropna().unique()

        if len(unique_values) == 0:
            return None
        elif len(unique_values) == 1:
            return unique_values[0]
        else:
            # Hay inconsistencia
            self.logger.warning(
                f"Inconsistencia en '{field_name}': "
                f"{len(unique_values)} valores únicos encontrados. "
                f"Usando: '{unique_values[0]}'"
            )
            return unique_values[0]

    def _aggregate_price(self, series: pd.Series, categoria: str = "") -> float:
        """
        Agrega precios unitarios validando consistencia.

        Si hay variación significativa en los precios, calcula el promedio
        y emite una advertencia.

        Args:
            series: Serie con precios unitarios.
            categoria: Nombre de la categoría para logging.

        Returns:
            Precio unitario (único o promedio).
        """
        prices = series.dropna()

        if len(prices) == 0:
            return 0.0

        unique_prices = prices.unique()

        # Caso ideal: un solo precio
        if len(unique_prices) == 1:
            return float(unique_prices[0])

        # Múltiples precios: calcular estadísticas
        price_mean = float(prices.mean())
        price_std = float(prices.std())
        price_min = float(prices.min())
        price_max = float(prices.max())

        # Verificar varianza significativa
        if price_mean > 0:
            coef_variation = price_std / price_mean

            if coef_variation > self.config.tolerance_price_variance:
                self.logger.warning(
                    f"Categoría '{categoria}': Variación significativa en precios. "
                    f"Mean={price_mean:.2f}, Std={price_std:.2f}, "
                    f"Min={price_min:.2f}, Max={price_max:.2f}, "
                    f"CV={coef_variation:.2%}"
                )

        return price_mean

    @staticmethod
    def _aggregate_alerts(series: pd.Series) -> Optional[str]:
        """
        Agrega alertas concatenándolas con separador.

        Args:
            series: Serie con mensajes de alerta.

        Returns:
            String con alertas únicas concatenadas, o None si no hay.
        """
        alerts = series.dropna().unique()

        if len(alerts) == 0:
            return None

        # Concatenar alertas únicas con separador
        return " | ".join(str(alert).strip() for alert in alerts if alert)

    def _organize_breakdown(
        self, items: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Organiza items en un desglose por categoría.

        Args:
            items: Lista de ítems procesados.

        Returns:
            Diccionario con categorías como claves y listas de items como valores.
        """
        if not items:
            self.logger.warning("No hay items para organizar en desglose")
            return {}

        desglose = {}
        items_sin_categoria = 0

        for item in items:
            categoria = item.get("categoria")

            # Manejar categoría None, vacía o solo espacios
            if not categoria or (isinstance(categoria, str) and not categoria.strip()):
                categoria = self.config.default_category
                items_sin_categoria += 1

            if categoria not in desglose:
                desglose[categoria] = []

            desglose[categoria].append(item)

        if items_sin_categoria > 0:
            self.logger.warning(
                f"{items_sin_categoria} items sin categoría válida, "
                f"asignados a '{self.config.default_category}'"
            )

        self.logger.debug(f"Desglose organizado en {len(desglose)} categorías")

        return desglose

    def _calculate_metadata(
        self, df_original: pd.DataFrame, processed_items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calcula metadata del procesamiento para auditoría.

        Args:
            df_original: DataFrame original antes del procesamiento.
            processed_items: Items después del procesamiento.

        Returns:
            Diccionario con métricas del procesamiento.
        """
        total_value = sum(float(item.get("VR_TOTAL") or 0) for item in processed_items)

        original_rows = len(df_original)
        processed_count = len(processed_items)

        return {
            "original_rows": original_rows,
            "processed_items": processed_count,
            "reduction_rate": (
                1 - (processed_count / original_rows) if original_rows > 0 else 0
            ),
            "categories_count": int(df_original["CATEGORIA"].nunique()),
            "total_value": round(total_value, 2),
            "avg_value_per_item": (
                round(total_value / processed_count, 2) if processed_count > 0 else 0
            ),
        }
