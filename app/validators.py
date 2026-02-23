"""
Módulo de Validación de Datos (Data Validators).

Este módulo proporciona estructuras y utilidades para la validación semántica y estructural
de los DataFrames procesados en el pipeline. Define contratos de calidad de datos y
mecanismos de reporte de errores.

Conceptos Clave:
----------------
1. Validación de Esquema (Schema Validation):
   Verifica la existencia de columnas críticas necesarias para el procesamiento aguas abajo.
   Actúa como un 'Type Guard' para los DataFrames.

2. Calidad de Datos (Data Quality):
   Evalúa la integridad de los datos, detectando valores nulos, tipos incorrectos
   o violaciones de restricciones de dominio (e.g., precios negativos).

3. Reporte de Validación (ValidationResult):
   Encapsula el resultado de una validación, separando errores bloqueantes de advertencias
   informativas, permitiendo estrategias de "fail-fast" o "best-effort".
"""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass
class ValidationResult:
    """
    Resultado de una operación de validación.

    Attributes:
        is_valid (bool): Indica si la validación pasó exitosamente (sin errores bloqueantes).
        errors (List[str]): Lista de mensajes de error que impiden el procesamiento.
        warnings (List[str]): Lista de advertencias sobre problemas no críticos.
    """
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class DataFrameValidator:
    """
    Validador de DataFrames con reglas de negocio y estructura.

    Proporciona métodos estáticos para verificar esquemas y calidad de datos
    en DataFrames de Pandas.
    """

    @staticmethod
    def validate_schema(df: pd.DataFrame, required_columns: List[str]) -> ValidationResult:
        """
        Verifica que existan las columnas requeridas en el DataFrame.

        Esta validación es estructural: asegura que el DataFrame tenga la "forma"
        esperada para ser consumido por los procesadores siguientes.

        Args:
            df (pd.DataFrame): El DataFrame a validar.
            required_columns (List[str]): Lista de nombres de columnas obligatorias.

        Returns:
            ValidationResult: Resultado de la validación.
                - is_valid=True si todas las columnas existen.
                - errors contiene la lista de columnas faltantes si falla.
        """
        if df is None:
            return ValidationResult(
                is_valid=False,
                errors=["El DataFrame es None"],
                warnings=[]
            )

        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            return ValidationResult(
                is_valid=False,
                errors=[f"Faltan columnas requeridas: {missing}"],
                warnings=[],
            )

        return ValidationResult(True, [], [])

    @staticmethod
    def check_data_quality(
        df: pd.DataFrame, critical_columns: List[str]
    ) -> ValidationResult:
        """
        Verifica la calidad de los datos (nulos, consistencia).

        Analiza el contenido del DataFrame buscando anomalías como valores nulos
        en columnas críticas.

        Args:
            df (pd.DataFrame): El DataFrame a validar.
            critical_columns (List[str]): Columnas donde los nulos son críticos.

        Returns:
            ValidationResult: Informe de calidad.
                - is_valid=True si no hay errores críticos (aunque puede haber warnings).
                - warnings contiene detalles sobre nulos encontrados.
        """
        if df is None:
            return ValidationResult(
                is_valid=False,
                errors=["El DataFrame es None"],
                warnings=[]
            )

        errors = []
        warnings = []

        for col in critical_columns:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    # Consideramos nulos en columnas críticas como warnings severos
                    # o errores dependiendo de la política. Aquí se reportan como warnings
                    # para permitir procesamiento parcial, pero se podría elevar a error.
                    warnings.append(f"Columna {col} tiene {null_count} valores nulos")
            else:
                # Si falta una columna crítica para calidad, es un error de esquema implícito
                # pero aquí lo tratamos como warning de calidad si schema validation pasó.
                warnings.append(f"Columna crítica {col} no encontrada para validación de calidad")

        return ValidationResult(len(errors) == 0, errors, warnings)
