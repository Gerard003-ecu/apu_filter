import pandas as pd
import numpy as np


class APUPresenter:
    """Procesa y agrupa datos de APU para presentación."""

    def __init__(self, logger):
        self.logger = logger

    def process_apu_details(self, apu_details: list, apu_code: str) -> dict:
        """
        Procesa los detalles de un APU específico.

        Args:
            apu_details: Lista de detalles del APU.
            apu_code: Código del APU.

        Returns:
            Diccionario con los datos procesados.
        """
        if not apu_details:
            raise ValueError(f"No se encontraron detalles para el APU {apu_code}")

        # Convertir a DataFrame y sanitizar
        df = pd.DataFrame(apu_details)
        df.replace({np.nan: None}, inplace=True)

        # Procesar por categorías
        processed_items = self._group_by_category(df)

        # Organizar desglose
        desglose = self._organize_breakdown(processed_items)

        return {
            "items": processed_items,
            "desglose": desglose,
            "total_items": len(processed_items),
        }

    def _group_by_category(self, df: pd.DataFrame) -> list:
        """
        Agrupa los ítems de un DataFrame por su categoría y descripción.

        Para cada categoría, agrupa los insumos por su descripción, sumando
        sus cantidades y valores para consolidar registros duplicados.

        Args:
            df: DataFrame con los detalles de los insumos de un APU.

        Returns:
            Una lista de diccionarios, donde cada diccionario representa un
            ítem de insumo procesado y agrupado.
        """
        processed = []

        for categoria in df["CATEGORIA"].unique():
            df_categoria = df[df["CATEGORIA"] == categoria]

            if df_categoria.empty:
                continue

            # Definir agregaciones
            aggregations = {
                "CANTIDAD_APU": "sum",
                "VALOR_TOTAL_APU": "sum",
                "RENDIMIENTO": "sum",
                "UNIDAD_APU": "first",
                "PRECIO_UNIT_APU": "first",
                "CATEGORIA": "first",
                "CODIGO_APU": "first",
                "UNIDAD_INSUMO": "first",
            }

            # Agregar manejo de alertas si existe
            if "alerta" in df_categoria.columns:
                aggregations["alerta"] = lambda x: " | ".join(x.dropna().unique())

            # Agrupar
            df_grouped = (
                df_categoria.groupby("DESCRIPCION_INSUMO").agg(aggregations).reset_index()
            )

            # Renombrar columnas
            df_grouped.rename(
                columns={
                    "DESCRIPCION_INSUMO": "DESCRIPCION",
                    "CANTIDAD_APU": "CANTIDAD",
                    "VALOR_TOTAL_APU": "VR_TOTAL",
                    "UNIDAD_INSUMO": "UNIDAD",
                    "PRECIO_UNIT_APU": "VR_UNITARIO",
                },
                inplace=True,
            )

            processed.extend(df_grouped.to_dict("records"))

        return processed

    def _organize_breakdown(self, items: list) -> dict:
        """
        Organiza una lista de ítems en un diccionario desglosado por categoría.

        Args:
            items: Lista de ítems (insumos) a organizar.

        Returns:
            Un diccionario donde las claves son los nombres de las categorías y
            los valores son listas de los ítems pertenecientes a esa categoría.
        """
        desglose = {}
        for item in items:
            categoria = item.get("CATEGORIA", "INDEFINIDO")
            if categoria not in desglose:
                desglose[categoria] = []
            desglose[categoria].append(item)

        return desglose
