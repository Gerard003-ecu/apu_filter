import pandas as pd
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]

class DataFrameValidator:
    """Validador de DataFrames con reglas de negocio."""

    @staticmethod
    def validate_schema(df: pd.DataFrame, required_columns: List[str]) -> ValidationResult:
        """Verifica que existan las columnas requeridas."""
        missing = [col for col in required_columns if col not in df.columns]

        if missing:
            return ValidationResult(
                is_valid=False,
                errors=[f"Faltan columnas requeridas: {missing}"],
                warnings=[]
            )

        return ValidationResult(True, [], [])

    @staticmethod
    def check_data_quality(df: pd.DataFrame, critical_columns: List[str]) -> ValidationResult:
        """Verifica calidad de datos (nulls, tipos, etc)."""
        errors = []
        warnings = []

        for col in critical_columns:
            if col in df.columns:
                null_count = df[col].isnull().sum()
                if null_count > 0:
                    warnings.append(f"Columna {col} tiene {null_count} valores nulos")

        return ValidationResult(len(errors) == 0, errors, warnings)
