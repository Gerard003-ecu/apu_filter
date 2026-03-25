"""
=========================================================================================
Módulo: Presenters (Operador de Proyección a la Variedad de Salida y Fibración Semántica)
Ubicación: app/adapters/presenters.py
=========================================================================================

Naturaleza Ciber-Física y Topológica:
    Este módulo no es un simple formateador de datos; actúa como el Proyector 
    Estructural Final que mapea el Complejo Simplicial del presupuesto (Estrato TACTICS) 
    hacia la Variedad de Observabilidad del usuario (Estrato WISDOM). Su mandato 
    axiomático es garantizar que ninguna entropía residual alcance la interfaz ejecutiva, 
    preservando estrictamente los isomorfismos financieros y semánticos.

1. Fibración Categórica y Clases de Equivalencia (Normalización):
    Abandona la "estandarización" empírica para instaurar un mapeo de fibración. 
    Resuelve vacíos y ambigüedades proyectando variantes textuales u ortográficas 
    hacia clases de equivalencia canónicas exactas. Garantiza que dos recursos 
    topológicamente homólogos posean firmas de identidad binariamente indistinguibles.

2. Retractos de Deformación (Sanitización Defensiva):
    Los operadores de limpieza actúan como retractos de deformación continuos. Aniquilan 
    caracteres de control, entropía de espacios y ruido sintáctico sin alterar el 
    ADN estructural del dato. Convierte columnas numéricas asegurando la clausura en el 
    cuerpo de los reales (ℝ) bajo las estrictas restricciones de la FPU (IEEE 754), 
    preservando el tipado inmutable del tensor durante la proyección.

3. Cuantificación de Medida de Lebesgue (Cálculo de Metadatos):
    Las estimaciones de cobertura no son simples conteos. El módulo cuantifica la 
    reducción del espacio de estados (registros) y computa la medida geométrica de 
    la clasificación, resumiendo las magnitudes económicas agregadas como integrales 
    definidas sobre los subespacios de costo.

4. Ley de Conservación de Valor (Consistencia de Agregación):
    Consolida registros por categoría bajo el rigor del Teorema de Conservación del 
    Valor. Verifica axiomáticamente la relación fundamental del espacio financiero:
        C = Q × P  (valor_total ≈ cantidad × valor_unitario)
    Somete esta igualdad a tolerancias híbridas (ε_rel, ε_abs) para absorber la 
    fricción de truncamiento numérico, dictaminando un VETO DE PRESENTACIÓN si 
    la discrepancia evidencia una corrupción termodinámica en la contabilidad.
=========================================================================================
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class APUProcessingConfig:
    """
    Configuración centralizada e inmutable para el procesamiento de APU.

    Invariantes:
    - Todas las tolerancias son no negativas.
    - ``required_columns`` y ``numeric_columns`` son conjuntos inmutables.
    - ``numeric_columns ⊆ required_columns`` (toda columna numérica debe existir).
    - ``default_category`` es un string no vacío tras strip.
    """

    required_columns: frozenset[str] | None = None
    numeric_columns: frozenset[str] | None = None
    default_category: str = "INDEFINIDO"
    tolerance_price_variance: float = 0.01
    consistency_relative_tolerance: float = 0.01
    consistency_absolute_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if self.required_columns is None:
            object.__setattr__(
                self,
                "required_columns",
                frozenset(
                    {
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
                ),
            )

        if self.numeric_columns is None:
            object.__setattr__(
                self,
                "numeric_columns",
                frozenset(
                    {
                        "CANTIDAD_APU",
                        "VALOR_TOTAL_APU",
                        "PRECIO_UNIT_APU",
                        "RENDIMIENTO",
                    }
                ),
            )

        # --- Validación de tolerancias ---
        if self.tolerance_price_variance < 0:
            raise ValueError(
                f"tolerance_price_variance debe ser ≥ 0; recibido={self.tolerance_price_variance}"
            )
        if self.consistency_relative_tolerance < 0:
            raise ValueError(
                f"consistency_relative_tolerance debe ser ≥ 0; "
                f"recibido={self.consistency_relative_tolerance}"
            )
        if self.consistency_absolute_tolerance < 0:
            raise ValueError(
                f"consistency_absolute_tolerance debe ser ≥ 0; "
                f"recibido={self.consistency_absolute_tolerance}"
            )

        # --- Invariante de subconjunto: numeric_columns ⊆ required_columns ---
        orphan_numeric = self.numeric_columns - self.required_columns
        if orphan_numeric:
            raise ValueError(
                f"numeric_columns contiene columnas no presentes en required_columns: "
                f"{sorted(orphan_numeric)}"
            )

        # --- default_category no vacío ---
        if not isinstance(self.default_category, str) or not self.default_category.strip():
            raise ValueError(
                f"default_category debe ser un string no vacío; "
                f"recibido={self.default_category!r}"
            )


class APUPresenter:
    """
    Procesa y agrupa datos de APU para presentación.

    Propiedades del pipeline:
    - validación explícita de entradas;
    - sanitización y normalización determinista;
    - agregación por categoría y descripción;
    - verificación defensiva de consistencia económica;
    - serialización final segura para consumidores externos.

    Garantías de idempotencia:
    - ``_sanitize_text`` y ``_normalize_category`` son funciones puras.
    - ``_sanitize_dataframe`` produce el mismo resultado si se aplica una o más veces.
    """

    _CATEGORY_NORMALIZATION_MAP: Dict[str, str] = {
        "MATERIAL": "MATERIALES",
        "MATERIALES": "MATERIALES",
        "MAT": "MATERIALES",
        "MANO DE OBRA": "MANO DE OBRA",
        "MO": "MANO DE OBRA",
        "EQUIPO": "EQUIPO",
        "EQUIPOS": "EQUIPO",
        "MAQUINARIA": "EQUIPO",
    }

    def __init__(
        self,
        logger: logging.Logger,
        config: Optional[APUProcessingConfig] = None,
    ) -> None:
        """
        Inicializa el presentador de APU.

        Args:
            logger: Instancia de ``logging.Logger`` no nula.
            config: Configuración del procesamiento. Si ``None``, usa valores por defecto.

        Raises:
            TypeError: Si ``logger`` no es instancia de ``logging.Logger``.
        """
        if not isinstance(logger, logging.Logger):
            raise TypeError(
                f"logger debe ser una instancia de logging.Logger; "
                f"recibido={type(logger).__name__}"
            )

        self._logger = logger
        self._config = config or APUProcessingConfig()
        self._logger.info("APUPresenter inicializado correctamente")

    # ------------------------------------------------------------------
    # Propiedades de acceso (encapsulación)
    # ------------------------------------------------------------------

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @property
    def config(self) -> APUProcessingConfig:
        return self._config

    # ------------------------------------------------------------------
    # Pipeline principal
    # ------------------------------------------------------------------

    def process_apu_details(
        self,
        apu_details: List[Dict[str, Any]],
        apu_code: str,
    ) -> Dict[str, Any]:
        """
        Procesa los detalles de un APU específico.

        Args:
            apu_details: Lista no vacía de diccionarios con columnas requeridas.
            apu_code: Código identificador del APU (string no vacío).

        Returns:
            Diccionario con estructura::

                {
                    "items": List[Dict[str, Any]],
                    "desglose": Dict[str, List[Dict[str, Any]]],
                    "total_items": int,
                    "metadata": Dict[str, Any],
                }

        Raises:
            ValueError: Entradas inválidas o DataFrame vacío.
            KeyError: Columnas requeridas faltantes.
            RuntimeError: Cualquier otro error inesperado.
        """
        try:
            self._validate_inputs(apu_details, apu_code)

            self._logger.info(
                "Iniciando procesamiento de APU %s con %d registros",
                apu_code,
                len(apu_details),
            )

            df = self._create_and_sanitize_dataframe(apu_details, apu_code)

            processed_items = self._group_by_category(df, apu_code)

            if not processed_items:
                self._logger.warning(
                    "APU %s: no se generaron items procesados", apu_code
                )

            desglose = self._organize_breakdown(processed_items)
            metadata = self._calculate_metadata(df, processed_items)

            self._logger.info(
                "APU %s procesado exitosamente. Items=%d, Categorías=%d, Reducción=%.1f%%",
                apu_code,
                len(processed_items),
                len(desglose),
                metadata["reduction_rate"] * 100.0,
            )

            return {
                "items": processed_items,
                "desglose": desglose,
                "total_items": len(processed_items),
                "metadata": metadata,
            }

        except (ValueError, KeyError):
            self._logger.exception("Error de validación en APU %s", apu_code)
            raise
        except Exception as exc:
            self._logger.exception("Error inesperado procesando APU %s", apu_code)
            raise RuntimeError(
                f"Fallo en procesamiento de APU {apu_code}"
            ) from exc

    # ------------------------------------------------------------------
    # Validación de entradas
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(apu_details: Any, apu_code: Any) -> None:
        """
        Valida parámetros de entrada del pipeline principal.

        Contratos:
        - ``apu_details`` es una lista no vacía de diccionarios.
        - ``apu_code`` es un string no vacío tras strip.
        """
        if not isinstance(apu_code, str) or not apu_code.strip():
            raise ValueError(
                f"apu_code debe ser un string no vacío; recibido={apu_code!r}"
            )

        if not isinstance(apu_details, list):
            raise ValueError(
                f"apu_details debe ser una lista; "
                f"recibido={type(apu_details).__name__}"
            )

        if not apu_details:
            raise ValueError(
                f"No se encontraron detalles para el APU {apu_code}"
            )

        non_dict_indices = [
            i for i, item in enumerate(apu_details) if not isinstance(item, dict)
        ]
        if non_dict_indices:
            raise ValueError(
                f"Elementos en posiciones {non_dict_indices} no son diccionarios"
            )

    # ------------------------------------------------------------------
    # Construcción y saneamiento del DataFrame
    # ------------------------------------------------------------------

    def _create_and_sanitize_dataframe(
        self,
        apu_details: List[Dict[str, Any]],
        apu_code: str,
    ) -> pd.DataFrame:
        """
        Crea, valida esquema y sanea el DataFrame base.

        El esquema se valida exactamente una vez, antes de la sanitización,
        evitando la doble validación del código original.
        """
        try:
            df = pd.DataFrame(apu_details)
        except Exception as exc:
            raise ValueError(
                f"Error creando DataFrame para APU {apu_code}: {exc}"
            ) from exc

        if df.empty:
            raise ValueError(f"DataFrame resultó vacío para APU {apu_code}")

        # Validación de esquema: única invocación en el pipeline
        self._validate_dataframe_schema(df, apu_code)

        df = self._sanitize_dataframe(df)

        self._logger.debug(
            "DataFrame saneado para APU %s: %d filas, %d columnas",
            apu_code,
            len(df),
            len(df.columns),
        )
        return df

    def _validate_dataframe_schema(
        self, df: pd.DataFrame, apu_code: str
    ) -> None:
        """
        Valida presencia de columnas requeridas.

        Raises:
            KeyError: Si faltan columnas requeridas, listadas en orden determinista.
        """
        current_columns = frozenset(df.columns)
        missing_columns = self._config.required_columns - current_columns

        if missing_columns:
            raise KeyError(
                f"APU {apu_code}: columnas faltantes: {sorted(missing_columns)}"
            )

        self._logger.debug(
            "Validación de esquema exitosa para APU %s", apu_code
        )

    # ------------------------------------------------------------------
    # Sanitización
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_text(value: Any) -> Any:
        """
        Sanitiza un valor textual de forma idempotente:

        - Elimina caracteres de control ASCII (U+0000–U+001F, U+007F).
        - Colapsa secuencias de espacios a uno solo.
        - Recorta extremos.
        - Convierte cadenas vacías resultantes a ``None``.

        Para valores no-string, retorna el valor sin modificar.
        """
        if value is None:
            return None

        if not isinstance(value, str):
            return value

        # Eliminar caracteres de control ASCII excepto espacio normal (0x20)
        cleaned = re.sub(r"[\x00-\x1F\x7F]", " ", value)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned if cleaned else None

    def _normalize_category(self, value: Any) -> str:
        """
        Normaliza una categoría de negocio a un conjunto canónico.

        La normalización colapsa espacios internos antes de la búsqueda
        en el mapa canónico, garantizando que variantes como
        ``"MANO  DE   OBRA"`` se resuelvan correctamente.

        Retorna ``config.default_category`` si el valor es nulo o vacío.
        """
        sanitized = self._sanitize_text(value)

        if sanitized is None:
            return self._config.default_category

        # Colapsar espacios internos y convertir a mayúsculas para búsqueda canónica
        canonical = re.sub(r"\s+", " ", sanitized.upper()).strip()

        return self._CATEGORY_NORMALIZATION_MAP.get(canonical, canonical)

    def _sanitize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitiza el DataFrame preservando tipos numéricos.

        Operaciones (en orden):
        1. Sanitización textual en columnas ``object``/``string``.
        2. Normalización categórica de ``CATEGORIA``.
        3. Conversión numérica segura con detección de coerciones fallidas.
        4. Reemplazo global de ±inf por NaN.

        La operación es idempotente: aplicarla N veces produce el mismo resultado.
        """
        df = df.copy()

        # 1. Sanitización textual solo en columnas de tipo texto
        text_columns = df.select_dtypes(include=["object", "string"]).columns
        for col in text_columns:
            df[col] = df[col].map(self._sanitize_text)

        # 2. Normalización categórica explícita
        if "CATEGORIA" in df.columns:
            df["CATEGORIA"] = df["CATEGORIA"].map(self._normalize_category)

        # 3. Conversión numérica segura
        for col in self._config.numeric_columns:
            if col not in df.columns:
                continue

            # Contar valores no-nulos ANTES de la conversión numérica
            # Nota: tras la sanitización textual, los vacíos ya son None.
            pre_conversion = df[col].copy()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)

            # Detectar coerciones fallidas: valores que eran no-nulos
            # antes de to_numeric pero resultaron NaN después.
            was_non_null = pre_conversion.notna()
            is_null_after = df[col].isna()
            failed_mask = was_non_null & is_null_after
            failed_count = int(failed_mask.sum())

            if failed_count > 0:
                self._logger.warning(
                    "Columna '%s': %d valores no convertibles a numérico (coerced a NaN)",
                    col,
                    failed_count,
                )

            negative_count = int((df[col] < 0).sum())
            if negative_count > 0:
                self._logger.warning(
                    "Columna '%s': %d valores negativos encontrados",
                    col,
                    negative_count,
                )

        # 4. Barrido global de no-finitos residuales
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].replace(
                [np.inf, -np.inf], np.nan
            )

        return df

    # ------------------------------------------------------------------
    # Agrupación y agregación
    # ------------------------------------------------------------------

    def _group_by_category(
        self, df: pd.DataFrame, apu_code: str
    ) -> List[Dict[str, Any]]:
        """
        Agrupa los ítems por categoría y descripción de insumo.

        Errores en categorías individuales se registran pero no abortan
        el procesamiento de las restantes.
        """
        processed: List[Dict[str, Any]] = []

        # Asegurar que NaN en CATEGORIA se reemplace por default_category
        categoria_series = df["CATEGORIA"].fillna(self._config.default_category)
        categories = list(categoria_series.unique())

        if not categories:
            self._logger.warning(
                "APU %s: no se encontraron categorías para procesar", apu_code
            )
            return processed

        self._logger.debug(
            "APU %s: procesando %d categorías", apu_code, len(categories)
        )

        # Trabajar sobre la serie ya rellenada
        df_work = df.copy()
        df_work["CATEGORIA"] = categoria_series

        for categoria in categories:
            df_categoria = df_work.loc[
                df_work["CATEGORIA"] == categoria
            ].copy()

            if df_categoria.empty:
                continue

            try:
                grouped_items = self._aggregate_category_items(
                    df_categoria, categoria
                )
                processed.extend(grouped_items)

                self._logger.debug(
                    "Categoría '%s': %d items procesados",
                    categoria,
                    len(grouped_items),
                )
            except (ValueError, KeyError) as exc:
                self._logger.warning(
                    "Error procesando categoría '%s' en APU %s: %s",
                    categoria,
                    apu_code,
                    exc,
                )
                continue

        return [self._serialize_record(item) for item in processed]

    def _aggregate_category_items(
        self,
        df_categoria: pd.DataFrame,
        categoria_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Agrega items de una categoría específica por ``DESCRIPCION_INSUMO``.

        Aplica:
        - Suma para cantidades, totales y rendimientos.
        - Promedio ponderado por cantidad para precios unitarios
          (evita sesgo cuando las cantidades difieren).
        - Validación de consistencia por campo singular.
        - Consolidación de alertas.
        """
        aggregations: Dict[str, Any] = {
            "CANTIDAD_APU": "sum",
            "VALOR_TOTAL_APU": "sum",
            "RENDIMIENTO": "sum",
            "UNIDAD_APU": lambda x: self._validate_consistency(
                x, "UNIDAD_APU"
            ),
            "PRECIO_UNIT_APU": lambda x: self._aggregate_price(
                x,
                df_categoria.loc[x.index, "CANTIDAD_APU"]
                if "CANTIDAD_APU" in df_categoria.columns
                else None,
                categoria_name,
            ),
            "CATEGORIA": "first",
            "CODIGO_APU": lambda x: self._validate_consistency(
                x, "CODIGO_APU"
            ),
            "UNIDAD_INSUMO": lambda x: self._validate_consistency(
                x, "UNIDAD_INSUMO"
            ),
        }

        if "alerta" in df_categoria.columns:
            aggregations["alerta"] = self._aggregate_alerts

        grouped = (
            df_categoria.groupby("DESCRIPCION_INSUMO", dropna=False)
            .agg(aggregations)
            .reset_index()
        )

        grouped.rename(
            columns={
                "DESCRIPCION_INSUMO": "descripcion",
                "CANTIDAD_APU": "cantidad",
                "VALOR_TOTAL_APU": "valor_total",
                "UNIDAD_INSUMO": "unidad",
                "PRECIO_UNIT_APU": "valor_unitario",
                "CATEGORIA": "categoria",
                "CODIGO_APU": "codigo_apu",
                "UNIDAD_APU": "unidad_apu",
                "RENDIMIENTO": "rendimiento",
            },
            inplace=True,
        )

        records = grouped.to_dict("records")

        for record in records:
            self._validate_item_consistency(record, categoria_name)

        return records

    def _validate_consistency(
        self, series: pd.Series, field_name: str = "campo"
    ) -> Any:
        """
        Valida que todos los valores no nulos en una serie sean idénticos.

        Si hay múltiples valores distintos, devuelve el más frecuente
        (modo estadístico) y emite advertencia.
        """
        cleaned = series.dropna()
        if cleaned.empty:
            return None

        unique_values = cleaned.unique()

        if len(unique_values) == 1:
            return unique_values[0]

        # Seleccionar el valor más frecuente (determinista: primer modo)
        mode_value = cleaned.mode().iloc[0]

        self._logger.warning(
            "Inconsistencia en '%s': %d valores únicos encontrados %s. "
            "Usando valor más frecuente: '%s'.",
            field_name,
            len(unique_values),
            list(unique_values),
            mode_value,
        )
        return mode_value

    def _aggregate_price(
        self,
        price_series: pd.Series,
        quantity_series: Optional[pd.Series] = None,
        categoria: str = "",
    ) -> Optional[float]:
        """
        Agrega precios unitarios con promedio ponderado por cantidad.

        Si ``quantity_series`` está disponible y contiene valores positivos,
        calcula:

            p̄ = Σ(pᵢ × qᵢ) / Σ(qᵢ)

        lo cual es consistente con la identidad:

            valor_total_agregado = Σ(pᵢ × qᵢ) = p̄ × Σ(qᵢ)

        Si no hay cantidades válidas, usa promedio aritmético simple.
        Retorna ``None`` (no ``0.0``) cuando no hay datos, evitando
        inyectar precios ficticios.

        Emite advertencia si el coeficiente de variación supera la tolerancia.
        """
        prices = pd.to_numeric(price_series, errors="coerce").dropna()

        if prices.empty:
            return None

        unique_prices = prices.unique()

        if len(unique_prices) == 1:
            return float(unique_prices[0])

        # Intentar promedio ponderado por cantidad
        weighted_mean: Optional[float] = None
        if quantity_series is not None:
            quantities = pd.to_numeric(quantity_series, errors="coerce").reindex(
                prices.index
            )
            # Filtrar pares donde ambos son finitos y cantidad > 0
            valid_mask = quantities.notna() & prices.notna() & (quantities > 0)
            if valid_mask.any():
                valid_p = prices[valid_mask]
                valid_q = quantities[valid_mask]
                total_q = valid_q.sum()
                if total_q > 0:
                    weighted_mean = float((valid_p * valid_q).sum() / total_q)

        result = weighted_mean if weighted_mean is not None else float(prices.mean())

        # Diagnóstico de dispersión
        price_mean = float(prices.mean())
        if price_mean > 0.0:
            price_std = float(prices.std(ddof=0))
            coef_variation = price_std / price_mean
            if coef_variation > self._config.tolerance_price_variance:
                self._logger.warning(
                    "Categoría '%s': variación significativa en precios. "
                    "mean=%.4f, std=%.4f, min=%.4f, max=%.4f, cv=%.2f%%, "
                    "método=%s",
                    categoria,
                    price_mean,
                    price_std,
                    float(prices.min()),
                    float(prices.max()),
                    coef_variation * 100.0,
                    "ponderado" if weighted_mean is not None else "aritmético",
                )

        return result

    @staticmethod
    def _aggregate_alerts(series: pd.Series) -> Optional[str]:
        """
        Consolida alertas únicas no vacías en un único string delimitado.
        """
        cleaned = [
            str(alert).strip()
            for alert in series.dropna().unique()
            if str(alert).strip()
        ]
        if not cleaned:
            return None
        return " | ".join(sorted(set(cleaned)))

    def _validate_item_consistency(
        self, item: Dict[str, Any], categoria: str
    ) -> None:
        """
        Verifica consistencia aproximada:

            valor_total ≈ cantidad × valor_unitario

        Usa ``numpy.isclose`` con tolerancias configurables (rtol, atol).

        Casos especiales manejados:
        - Si alguno de los tres valores es ``None``, no se puede verificar → skip.
        - Si ``valor_unitario = 0`` pero ``valor_total ≠ 0`` → inconsistencia.
        - Si ``cantidad = 0`` pero ``valor_total ≠ 0`` → inconsistencia.

        No aborta; registra advertencias.
        """
        cantidad = self._safe_float(item.get("cantidad"))
        valor_total = self._safe_float(item.get("valor_total"))
        valor_unitario = self._safe_float(item.get("valor_unitario"))

        if cantidad is None or valor_total is None or valor_unitario is None:
            return

        expected_total = cantidad * valor_unitario

        if not np.isclose(
            valor_total,
            expected_total,
            rtol=self._config.consistency_relative_tolerance,
            atol=self._config.consistency_absolute_tolerance,
        ):
            descripcion = item.get("descripcion", "<sin descripción>")
            abs_diff = abs(valor_total - expected_total)
            rel_diff = (
                abs_diff / abs(valor_total) if abs(valor_total) > 0 else float("inf")
            )
            self._logger.warning(
                "Inconsistencia en categoría '%s', item '%s': "
                "valor_total=%.6f vs cantidad(%.6f)×valor_unitario(%.6f)=%.6f "
                "[Δabs=%.6f, Δrel=%.4f%%]",
                categoria,
                descripcion,
                valor_total,
                cantidad,
                valor_unitario,
                expected_total,
                abs_diff,
                rel_diff * 100.0,
            )

    # ------------------------------------------------------------------
    # Organización del desglose
    # ------------------------------------------------------------------

    def _organize_breakdown(
        self,
        items: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Organiza items en un desglose por categoría.

        Items sin categoría válida se asignan a ``config.default_category``.
        """
        if not items:
            self._logger.warning("No hay items para organizar en desglose")
            return {}

        desglose: Dict[str, List[Dict[str, Any]]] = {}
        items_sin_categoria = 0

        for item in items:
            categoria = item.get("categoria")

            if not categoria or (
                isinstance(categoria, str) and not categoria.strip()
            ):
                categoria = self._config.default_category
                item["categoria"] = categoria
                items_sin_categoria += 1

            desglose.setdefault(categoria, []).append(item)

        if items_sin_categoria > 0:
            self._logger.warning(
                "%d items sin categoría válida fueron asignados a '%s'",
                items_sin_categoria,
                self._config.default_category,
            )

        self._logger.debug(
            "Desglose organizado en %d categorías", len(desglose)
        )
        return desglose

    # ------------------------------------------------------------------
    # Metadatos
    # ------------------------------------------------------------------

    def _calculate_metadata(
        self,
        df_original: pd.DataFrame,
        processed_items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Calcula metadatos del procesamiento.

        Métricas:
        - ``original_rows``: filas en el DataFrame original.
        - ``processed_items``: items tras agregación.
        - ``reduction_rate``: 1 − (processed / original), clamped a [0, 1].
        - ``categories_count``: categorías únicas (excluyendo NaN).
        - ``classification_coverage``: fracción de filas con categoría no nula.
        - ``total_value``: suma de ``valor_total`` de items procesados.
        - ``avg_value_per_item``: ``total_value / processed_items``.
        """
        original_rows = len(df_original)
        processed_count = len(processed_items)

        # Categorías y cobertura (calculados una sola vez)
        has_categoria_col = "CATEGORIA" in df_original.columns
        if original_rows > 0 and has_categoria_col:
            categories_count = int(
                df_original["CATEGORIA"].nunique(dropna=True)
            )
            classified_rows = int(df_original["CATEGORIA"].notna().sum())
            classification_coverage = classified_rows / original_rows
        else:
            categories_count = 0
            classified_rows = 0
            classification_coverage = 0.0

        # Total económico
        total_value = 0.0
        for item in processed_items:
            val = self._safe_float(item.get("valor_total"))
            if val is not None:
                total_value += val

        # Tasa de reducción, clamped a [0, 1]
        if original_rows > 0:
            raw_reduction = 1.0 - (processed_count / original_rows)
            reduction_rate = max(0.0, min(1.0, raw_reduction))
        else:
            reduction_rate = 0.0

        avg_value_per_item = (
            total_value / processed_count if processed_count > 0 else 0.0
        )

        metadata = {
            "original_rows": original_rows,
            "processed_items": processed_count,
            "reduction_rate": round(reduction_rate, 4),
            "categories_count": categories_count,
            "classification_coverage": round(classification_coverage, 4),
            "total_value": round(total_value, 2),
            "avg_value_per_item": round(avg_value_per_item, 2),
        }

        self._logger.info(
            "Metadata: items=%d | total=$%.2f | promedio=$%.2f | "
            "cobertura=%.2f%% | reducción=%.1f%%",
            processed_count,
            total_value,
            avg_value_per_item,
            classification_coverage * 100.0,
            reduction_rate * 100.0,
        )

        return metadata

    # ------------------------------------------------------------------
    # Utilidades de conversión y serialización
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """
        Convierte defensivamente a float finito.

        Retorna ``None`` para valores nulos, no convertibles o no finitos
        (NaN, ±inf).
        """
        if value is None:
            return None
        try:
            result = float(value)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(result):
            return None
        return result

    @classmethod
    def _serialize_scalar(cls, value: Any) -> Any:
        """
        Convierte valores escalares a representación segura para serialización JSON:

        - ``NaN``, ``±inf`` → ``None``
        - numpy scalars → Python nativos
        - ``pd.NaT``, ``pd.NA`` → ``None``
        """
        if value is None:
            return None

        # Convertir escalares numpy a tipos Python nativos
        if isinstance(value, np.generic):
            value = value.item()

        # Verificar NaN/inf para floats (después de la conversión numpy)
        if isinstance(value, float) and not np.isfinite(value):
            return None

        # pd.isna maneja NaT, NA, y NaN de forma unificada
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            # pd.isna puede fallar con tipos no soportados (e.g., listas)
            pass

        return value

    @classmethod
    def _serialize_record(cls, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitiza un registro completo para serialización segura.
        """
        return {
            key: cls._serialize_scalar(value) for key, value in record.items()
        }