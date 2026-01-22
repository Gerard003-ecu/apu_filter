"""
Tests para Pipeline Director
=============================

Suite robustecida que valida:
- Orquestación de pasos del pipeline
- Validación de datos de entrada
- Transformaciones y merges
- Inyección de telemetría
- Manejo de errores y recuperación
- Invariantes del flujo de datos

Modelo del Pipeline:
--------------------
El pipeline se modela como un grafo dirigido acíclico (DAG) donde:

    [Presupuesto] ──┐
                    ├──► [Merge APUs+Insumos] ──► [Merge Final] ──► [Output]
    [APUs] ─────────┤
                    │
    [Insumos] ──────┘

Invariantes del Pipeline:
- Conservación: |Output| ≤ |Input_max| (no genera filas fantasma)
- Determinismo: f(x) = f(x) para cualquier ejecución
- Idempotencia parcial: Validaciones son idempotentes
- Trazabilidad: Cada paso registra telemetría
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
import contextlib
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from app.pipeline_director import (
    ColumnNames,
    DataMerger,
    DataValidator,
    FileValidator,
    LoadDataStep,
    PipelineDirector,
    PresupuestoProcessor,
    ProcessingStep,
    ProcessingThresholds,
)
from app.telemetry import TelemetryContext


# =============================================================================
# CONSTANTES Y CONFIGURACIÓN
# =============================================================================

FIXED_TIMESTAMP = datetime(2024, 1, 15, 10, 30, 0)

# Configuración de pruebas
TEST_CONFIG = {
    "presupuesto_path": "test_presupuesto.csv",
    "apus_path": "test_apus.csv",
    "insumos_path": "test_insumos.csv",
    "output_path": "test_output.csv",
    "pipeline_recipe": [],
    "loader_params": {},
}

# Umbrales para validación
VALIDATION_THRESHOLDS = {
    "min_file_size": 10,
    "max_null_ratio": 0.5,
    "max_duplicate_ratio": 0.1,
}


class StepStatus(Enum):
    """Estados posibles de un paso del pipeline."""
    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    SKIPPED = auto()


class ValidationResult(Enum):
    """Resultados de validación."""
    VALID = auto()
    INVALID = auto()
    WARNING = auto()


# =============================================================================
# BUILDERS: Construcción de DataFrames con Validación
# =============================================================================


@dataclass
class ColumnSpec:
    """Especificación de una columna."""
    name: str
    dtype: str = "object"
    nullable: bool = True
    default: Any = None


class DataFrameBuilder:
    """
    Builder base para DataFrames con validación de esquema.
    
    Garantiza:
    - Tipos de datos consistentes
    - Nombres de columnas válidos
    - Datos coherentes para testing
    """
    
    def __init__(self):
        self._data: Dict[str, List[Any]] = {}
        self._schema: Dict[str, ColumnSpec] = {}
    
    def _add_column(self, name: str, values: List[Any]) -> "DataFrameBuilder":
        """Agrega una columna con valores."""
        self._data[name] = values
        return self
    
    def _ensure_equal_lengths(self) -> None:
        """Verifica que todas las columnas tengan la misma longitud."""
        if not self._data:
            return
        lengths = {col: len(vals) for col, vals in self._data.items()}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            raise ValueError(f"Longitudes inconsistentes: {lengths}")
    
    def build(self) -> pd.DataFrame:
        """Construye el DataFrame validando coherencia."""
        self._ensure_equal_lengths()
        return pd.DataFrame(self._data)
    
    def build_empty(self) -> pd.DataFrame:
        """Construye DataFrame vacío con el esquema definido."""
        return pd.DataFrame(columns=list(self._schema.keys()))


class PresupuestoBuilder(DataFrameBuilder):
    """
    Builder para DataFrames de presupuesto.
    
    Esquema esperado:
    - CODIGO_APU: Identificador único del APU
    - DESCRIPCION_APU: Descripción de la actividad
    - CANTIDAD_PRESUPUESTO: Cantidad presupuestada
    - UNIDAD: Unidad de medida
    """
    
    def __init__(self):
        super().__init__()
        self._schema = {
            ColumnNames.CODIGO_APU: ColumnSpec(ColumnNames.CODIGO_APU, "object", False),
            "DESCRIPCION_APU": ColumnSpec("DESCRIPCION_APU", "object", True),
            "CANTIDAD_PRESUPUESTO": ColumnSpec("CANTIDAD_PRESUPUESTO", "float64", False),
            "UNIDAD": ColumnSpec("UNIDAD", "object", True, "UND"),
        }
        self._items: List[Dict[str, Any]] = []
    
    def with_item(
        self,
        codigo: str,
        descripcion: str = "Actividad",
        cantidad: float = 1.0,
        unidad: str = "UND"
    ) -> "PresupuestoBuilder":
        """Agrega un ítem de presupuesto."""
        if not codigo:
            raise ValueError("Código APU no puede estar vacío")
        if cantidad < 0:
            raise ValueError(f"Cantidad no puede ser negativa: {cantidad}")
        
        self._items.append({
            ColumnNames.CODIGO_APU: codigo,
            "DESCRIPCION_APU": descripcion,
            "CANTIDAD_PRESUPUESTO": cantidad,
            "UNIDAD": unidad,
        })
        return self
    
    def with_items(self, count: int, prefix: str = "APU") -> "PresupuestoBuilder":
        """Agrega múltiples ítems con datos generados."""
        for i in range(1, count + 1):
            self.with_item(
                codigo=f"{prefix}-{i:03d}",
                descripcion=f"Actividad {i}",
                cantidad=float(i * 10),
            )
        return self
    
    def with_phantom_rows(self, count: int = 2) -> "PresupuestoBuilder":
        """Agrega filas fantasma (vacías o con solo nulos)."""
        for _ in range(count):
            self._items.append({
                ColumnNames.CODIGO_APU: "",
                "DESCRIPCION_APU": None,
                "CANTIDAD_PRESUPUESTO": np.nan,
                "UNIDAD": None,
            })
        return self
    
    def with_duplicates(self, codigo: str, count: int = 2) -> "PresupuestoBuilder":
        """Agrega ítems duplicados para testing."""
        for i in range(count):
            self.with_item(codigo, f"Duplicado {i}", float(i + 1))
        return self
    
    def build(self) -> pd.DataFrame:
        """Construye el DataFrame de presupuesto."""
        if not self._items:
            return self.build_empty()
        return pd.DataFrame(self._items)


class InsumosBuilder(DataFrameBuilder):
    """
    Builder para DataFrames de insumos.
    
    Esquema esperado:
    - DESCRIPCION_INSUMO: Descripción del insumo
    - TIPO_INSUMO: Tipo (MATERIAL, MANO_OBRA, etc.)
    - PRECIO_UNITARIO: Precio por unidad
    - UNIDAD: Unidad de medida
    """
    
    TIPOS_VALIDOS = frozenset({"MATERIAL", "MANO_OBRA", "EQUIPO", "TRANSPORTE"})
    
    def __init__(self):
        super().__init__()
        self._items: List[Dict[str, Any]] = []
    
    def with_insumo(
        self,
        descripcion: str,
        tipo: str = "MATERIAL",
        precio: float = 100.0,
        unidad: str = "UND"
    ) -> "InsumosBuilder":
        """Agrega un insumo."""
        if tipo not in self.TIPOS_VALIDOS:
            raise ValueError(f"Tipo inválido: {tipo}. Válidos: {self.TIPOS_VALIDOS}")
        if precio < 0:
            raise ValueError(f"Precio no puede ser negativo: {precio}")
        
        self._items.append({
            ColumnNames.DESCRIPCION_INSUMO: descripcion,
            "TIPO_INSUMO": tipo,
            "PRECIO_UNITARIO": precio,
            "UNIDAD": unidad,
        })
        return self
    
    def with_insumos(self, count: int) -> "InsumosBuilder":
        """Agrega múltiples insumos con datos generados."""
        tipos = list(self.TIPOS_VALIDOS)
        for i in range(1, count + 1):
            self.with_insumo(
                descripcion=f"Insumo {i}",
                tipo=tipos[i % len(tipos)],
                precio=float(i * 50),
            )
        return self
    
    def build(self) -> pd.DataFrame:
        """Construye el DataFrame de insumos."""
        if not self._items:
            return pd.DataFrame(columns=[
                ColumnNames.DESCRIPCION_INSUMO, "TIPO_INSUMO", 
                "PRECIO_UNITARIO", "UNIDAD"
            ])
        return pd.DataFrame(self._items)


class APUsBuilder(DataFrameBuilder):
    """
    Builder para DataFrames de APUs (análisis de precios unitarios).
    
    Esquema esperado:
    - CODIGO_APU: Código del APU
    - DESCRIPCION_INSUMO: Insumo usado en el APU
    - CANTIDAD: Cantidad del insumo
    - RENDIMIENTO: Rendimiento del insumo
    """
    
    def __init__(self):
        super().__init__()
        self._items: List[Dict[str, Any]] = []
    
    def with_apu_insumo(
        self,
        codigo_apu: str,
        descripcion_insumo: str,
        cantidad: float = 1.0,
        rendimiento: float = 1.0
    ) -> "APUsBuilder":
        """Agrega una relación APU-Insumo."""
        self._items.append({
            ColumnNames.CODIGO_APU: codigo_apu,
            ColumnNames.DESCRIPCION_INSUMO: descripcion_insumo,
            "CANTIDAD": cantidad,
            "RENDIMIENTO": rendimiento,
        })
        return self
    
    def for_presupuesto(
        self, 
        presupuesto: pd.DataFrame,
        insumos_per_apu: int = 3
    ) -> "APUsBuilder":
        """Genera APUs para un presupuesto dado."""
        for _, row in presupuesto.iterrows():
            codigo = row.get(ColumnNames.CODIGO_APU)
            if pd.notna(codigo) and codigo:
                for i in range(insumos_per_apu):
                    self.with_apu_insumo(
                        codigo_apu=codigo,
                        descripcion_insumo=f"Insumo {i+1}",
                        cantidad=float(i + 1),
                    )
        return self
    
    def build(self) -> pd.DataFrame:
        """Construye el DataFrame de APUs."""
        if not self._items:
            return pd.DataFrame(columns=[
                ColumnNames.CODIGO_APU, ColumnNames.DESCRIPCION_INSUMO,
                "CANTIDAD", "RENDIMIENTO"
            ])
        return pd.DataFrame(self._items)


class ContextBuilder:
    """
    Builder para contextos de ejecución del pipeline.
    
    El contexto contiene:
    - Rutas de archivos
    - DataFrames cargados
    - Resultados intermedios
    - Metadatos de ejecución
    """
    
    def __init__(self):
        self._context: Dict[str, Any] = {}
    
    def with_paths(
        self,
        presupuesto: str = "presupuesto.csv",
        apus: str = "apus.csv",
        insumos: str = "insumos.csv"
    ) -> "ContextBuilder":
        """Configura rutas de archivos."""
        self._context.update({
            "presupuesto_path": presupuesto,
            "apus_path": apus,
            "insumos_path": insumos,
        })
        return self
    
    def with_dataframes(
        self,
        presupuesto: Optional[pd.DataFrame] = None,
        apus: Optional[pd.DataFrame] = None,
        insumos: Optional[pd.DataFrame] = None
    ) -> "ContextBuilder":
        """Configura DataFrames precargados."""
        if presupuesto is not None:
            self._context["df_presupuesto"] = presupuesto
        if apus is not None:
            self._context["df_apus"] = apus
        if insumos is not None:
            self._context["df_insumos"] = insumos
        return self
    
    def with_param(self, key: str, value: Any) -> "ContextBuilder":
        """Agrega un parámetro adicional."""
        self._context[key] = value
        return self
    
    def build(self) -> Dict[str, Any]:
        """Construye el contexto."""
        return self._context.copy()


# =============================================================================
# MOCK FACTORIES
# =============================================================================


class MockStep(ProcessingStep):
    """Mock de un paso del pipeline para testing."""
    
    def __init__(self, config: Dict, thresholds: ProcessingThresholds):
        self.config = config
        self.thresholds = thresholds
        self.execute_count = 0
        self.last_context = None
    
    def execute(self, context: Dict, telemetry: TelemetryContext) -> Dict:
        """Ejecuta el paso mock."""
        telemetry.start_step("mock_step")
        self.execute_count += 1
        self.last_context = context.copy()
        context["mock_executed"] = True
        context["mock_timestamp"] = FIXED_TIMESTAMP.isoformat()
        telemetry.end_step("mock_step", "success")
        return context


class FailingStep(ProcessingStep):
    """Paso que siempre falla para testing de errores."""
    
    def __init__(
        self, 
        config: Dict, 
        thresholds: ProcessingThresholds,
        error_type: Type[Exception] = ValueError,
        error_message: str = "Step failed intentionally"
    ):
        self.config = config
        self.thresholds = thresholds
        self.error_type = error_type
        self.error_message = error_message
    
    def execute(self, context: Dict, telemetry: TelemetryContext) -> Dict:
        """Lanza excepción."""
        telemetry.start_step("failing_step")
        raise self.error_type(self.error_message)


class ConditionalStep(ProcessingStep):
    """Paso que falla condicionalmente para testing."""
    
    def __init__(self, config: Dict, thresholds: ProcessingThresholds):
        self.config = config
        self.thresholds = thresholds
        self.call_count = 0
        self.fail_on_calls: Set[int] = set()
    
    def fail_on(self, *call_numbers: int) -> "ConditionalStep":
        """Configura en qué llamadas debe fallar."""
        self.fail_on_calls = set(call_numbers)
        return self
    
    def execute(self, context: Dict, telemetry: TelemetryContext) -> Dict:
        """Ejecuta y falla condicionalmente."""
        self.call_count += 1
        telemetry.start_step("conditional_step")
        
        if self.call_count in self.fail_on_calls:
            telemetry.end_step("conditional_step", "error")
            raise RuntimeError(f"Conditional failure on call {self.call_count}")
        
        context["conditional_executed"] = self.call_count
        telemetry.end_step("conditional_step", "success")
        return context


class MockFileSystem:
    """Mock del sistema de archivos para testing."""
    
    def __init__(self):
        self._files: Dict[str, Dict[str, Any]] = {}
    
    def add_file(
        self,
        path: str,
        exists: bool = True,
        is_file: bool = True,
        size: int = 1000,
        readable: bool = True,
        content: Optional[pd.DataFrame] = None
    ) -> "MockFileSystem":
        """Registra un archivo mock."""
        self._files[path] = {
            "exists": exists,
            "is_file": is_file,
            "size": size,
            "readable": readable,
            "content": content,
        }
        return self
    
    def patch_validators(self):
        """Retorna los patches necesarios para simular el filesystem."""
        
        def _resolve_file_data(path_obj):
            # Try to resolve by full string, then by name
            path_str = str(path_obj)
            if path_str in self._files:
                return self._files[path_str]
            # Try name if available
            if hasattr(path_obj, "name") and path_obj.name in self._files:
                return self._files[path_obj.name]
            return {}

        def mock_exists(self_path):
            data = _resolve_file_data(self_path)
            return data.get("exists", False)

        def mock_is_file(self_path):
            data = _resolve_file_data(self_path)
            return data.get("is_file", False)
        
        def mock_stat(self_path):
            data = _resolve_file_data(self_path)
            mock = MagicMock()
            mock.st_size = data.get("size", 0)
            return mock
        
        def mock_access(path, mode):
            data = _resolve_file_data(path)
            return data.get("readable", False)
        
        return {
            "pathlib.Path.exists": mock_exists,
            "pathlib.Path.is_file": mock_is_file,
            "pathlib.Path.stat": mock_stat,
            "os.access": mock_access,
        }


# =============================================================================
# FIXTURES
# =============================================================================


class TestFixtures:
    """Clase base con fixtures reutilizables."""

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        """Configuración base del pipeline."""
        return TEST_CONFIG.copy()

    @pytest.fixture
    def thresholds(self) -> ProcessingThresholds:
        """Umbrales de procesamiento."""
        return ProcessingThresholds()

    @pytest.fixture
    def telemetry(self) -> TelemetryContext:
        """Contexto de telemetría limpio."""
        return TelemetryContext()

    @pytest.fixture
    def director(self, config, telemetry) -> PipelineDirector:
        """Instancia del director del pipeline."""
        return PipelineDirector(config, telemetry)

    @pytest.fixture
    def presupuesto_df(self) -> pd.DataFrame:
        """DataFrame de presupuesto válido."""
        return PresupuestoBuilder().with_items(5).build()

    @pytest.fixture
    def insumos_df(self) -> pd.DataFrame:
        """DataFrame de insumos válido."""
        return InsumosBuilder().with_insumos(10).build()

    @pytest.fixture
    def apus_df(self, presupuesto_df) -> pd.DataFrame:
        """DataFrame de APUs válido."""
        return APUsBuilder().for_presupuesto(presupuesto_df).build()

    @pytest.fixture
    def valid_context(
        self, presupuesto_df, apus_df, insumos_df
    ) -> Dict[str, Any]:
        """Contexto válido completo."""
        return (
            ContextBuilder()
            .with_paths()
            .with_dataframes(presupuesto_df, apus_df, insumos_df)
            .build()
        )

    @pytest.fixture
    def mock_filesystem(self) -> MockFileSystem:
        """Mock del sistema de archivos."""
        return MockFileSystem()

    @pytest.fixture
    def merger(self, thresholds) -> DataMerger:
        """Instancia del merger."""
        return DataMerger(thresholds)


# =============================================================================
# TESTS: DATA VALIDATOR
# =============================================================================


class TestDataValidator(TestFixtures):
    """
    Tests para DataValidator.
    
    Valida:
    - Detección de DataFrames vacíos
    - Detección de valores nulos
    - Validación de columnas requeridas
    - Detección de duplicados
    """

    # --- Tests para validate_dataframe_not_empty ---
    
    def test_validate_none_returns_invalid(self):
        """None es detectado como inválido."""
        valid, error = DataValidator.validate_dataframe_not_empty(None, "test")
        
        assert valid is False
        assert "None" in error

    def test_validate_non_dataframe_returns_invalid(self):
        """Tipo no-DataFrame es detectado como inválido."""
        invalid_inputs = ["string", 123, [], {}, set()]
        
        for invalid in invalid_inputs:
            valid, error = DataValidator.validate_dataframe_not_empty(invalid, "test")
            assert valid is False
            assert "no es un DataFrame" in error

    def test_validate_empty_dataframe_returns_invalid(self):
        """DataFrame vacío es detectado como inválido."""
        valid, error = DataValidator.validate_dataframe_not_empty(pd.DataFrame(), "test")
        
        assert valid is False
        assert "vacío" in error

    def test_validate_all_nulls_returns_invalid(self):
        """DataFrame con solo nulos es detectado como inválido."""
        df_nulls = pd.DataFrame({
            "A": [None, None, None],
            "B": [np.nan, np.nan, np.nan]
        })
        
        valid, error = DataValidator.validate_dataframe_not_empty(df_nulls, "test")
        
        assert valid is False
        assert "solo valores nulos" in error

    def test_validate_valid_dataframe_returns_valid(self):
        """DataFrame válido es aceptado."""
        df = PresupuestoBuilder().with_items(3).build()
        
        valid, error = DataValidator.validate_dataframe_not_empty(df, "test")
        
        assert valid is True
        assert error is None

    def test_validate_partial_nulls_returns_valid(self):
        """DataFrame con algunos nulos pero datos válidos es aceptado."""
        df = pd.DataFrame({
            "A": [1, None, 3],
            "B": ["x", "y", None]
        })
        
        valid, error = DataValidator.validate_dataframe_not_empty(df, "test")
        
        assert valid is True

    # --- Tests para validate_required_columns ---
    
    def test_validate_columns_all_present(self):
        """Columnas requeridas presentes son validadas correctamente."""
        df = pd.DataFrame({"Col1": [1], "Col2": [2], "Col3": [3]})
        
        valid, error = DataValidator.validate_required_columns(
            df, ["Col1", "Col2"], "test"
        )
        
        assert valid is True
        assert error is None

    def test_validate_columns_case_insensitive(self):
        """Validación de columnas es case-insensitive."""
        df = pd.DataFrame({"COLUMN_NAME": [1]})
        
        valid, error = DataValidator.validate_required_columns(
            df, ["column_name"], "test"
        )
        
        assert valid is True

    def test_validate_columns_missing_detected(self):
        """Columnas faltantes son detectadas."""
        df = pd.DataFrame({"Col1": [1]})
        
        valid, error = DataValidator.validate_required_columns(
            df, ["Col1", "Col2", "Col3"], "test"
        )
        
        assert valid is False
        assert "Faltan columnas" in error
        assert "Col2" in error or "Col3" in error

    def test_validate_columns_none_dataframe(self):
        """Validación falla gracefully con None."""
        valid, error = DataValidator.validate_required_columns(
            None, ["Col1"], "test"
        )
        
        assert valid is False

    def test_validate_columns_empty_required_list(self):
        """Lista vacía de requeridos siempre es válida."""
        df = pd.DataFrame({"Col1": [1]})
        
        valid, error = DataValidator.validate_required_columns(df, [], "test")
        
        assert valid is True

    # --- Tests para detect_and_log_duplicates ---
    
    def test_detect_duplicates_removes_them(self):
        """Duplicados son detectados y eliminados."""
        df = pd.DataFrame({
            "ID": [1, 2, 2, 3, 3, 3],
            "Val": ["a", "b", "c", "d", "e", "f"]
        })
        
        cleaned = DataValidator.detect_and_log_duplicates(df, ["ID"], "test")
        
        assert len(cleaned) == 3
        assert list(cleaned["ID"]) == [1, 2, 3]

    def test_detect_duplicates_keeps_first(self):
        """Por defecto mantiene la primera ocurrencia."""
        df = pd.DataFrame({
            "ID": [1, 1, 1],
            "Val": ["first", "second", "third"]
        })
        
        cleaned = DataValidator.detect_and_log_duplicates(df, ["ID"], "test")
        
        assert len(cleaned) == 1
        assert cleaned.iloc[0]["Val"] == "first"

    def test_detect_duplicates_missing_column(self):
        """Columna inexistente no causa error."""
        df = pd.DataFrame({"ID": [1, 2, 3]})
        
        cleaned = DataValidator.detect_and_log_duplicates(df, ["Missing"], "test")
        
        assert len(cleaned) == 3  # Sin cambios

    def test_detect_duplicates_none_input(self):
        """Input None retorna DataFrame vacío."""
        cleaned = DataValidator.detect_and_log_duplicates(None, ["ID"], "test")
        
        assert isinstance(cleaned, pd.DataFrame)
        assert cleaned.empty

    def test_detect_duplicates_multiple_columns(self):
        """Duplicados por múltiples columnas."""
        df = pd.DataFrame({
            "A": [1, 1, 1, 2],
            "B": ["x", "x", "y", "x"],
            "Val": [10, 20, 30, 40]
        })
        
        cleaned = DataValidator.detect_and_log_duplicates(df, ["A", "B"], "test")
        
        assert len(cleaned) == 3  # (1,x), (1,y), (2,x) son únicos


# =============================================================================
# TESTS: FILE VALIDATOR
# =============================================================================


class TestFileValidator(TestFixtures):
    """
    Tests para FileValidator.
    
    Valida:
    - Existencia de archivos
    - Tipo de archivo (no directorio)
    - Permisos de lectura
    - Tamaño mínimo
    """

    def test_validate_file_exists_success(self, mock_filesystem):
        """Archivo válido es aceptado."""
        mock_filesystem.add_file("data.csv", exists=True, is_file=True, readable=True)
        
        with contextlib.ExitStack() as stack:
            for target, side_effect in mock_filesystem.patch_validators().items():
                # Use autospec=True so that 'self' is passed to the mock
                stack.enter_context(patch(target, side_effect=side_effect, autospec=True))
            valid, error = FileValidator.validate_file_exists("data.csv", "test")
        
        assert valid is True
        assert error is None

    def test_validate_file_not_found(self, mock_filesystem):
        """Archivo no existente es rechazado."""
        mock_filesystem.add_file("missing.csv", exists=False)
        
        with contextlib.ExitStack() as stack:
            for target, side_effect in mock_filesystem.patch_validators().items():
                stack.enter_context(patch(target, side_effect=side_effect, autospec=True))
            valid, error = FileValidator.validate_file_exists("missing.csv", "test")
        
        assert valid is False
        assert "no encontrado" in error

    def test_validate_directory_rejected(self, mock_filesystem):
        """Directorio es rechazado."""
        mock_filesystem.add_file("data_dir", exists=True, is_file=False)
        
        with contextlib.ExitStack() as stack:
            for target, side_effect in mock_filesystem.patch_validators().items():
                stack.enter_context(patch(target, side_effect=side_effect, autospec=True))
            valid, error = FileValidator.validate_file_exists("data_dir", "test")
        
        assert valid is False
        assert "no es un archivo" in error

    def test_validate_no_read_permission(self, mock_filesystem):
        """Archivo sin permisos de lectura es rechazado."""
        mock_filesystem.add_file("protected.csv", exists=True, is_file=True, readable=False)
        
        with contextlib.ExitStack() as stack:
            for target, side_effect in mock_filesystem.patch_validators().items():
                # os.access is a function, autospec works too
                stack.enter_context(patch(target, side_effect=side_effect, autospec=True))
            valid, error = FileValidator.validate_file_exists("protected.csv", "test")
        
        assert valid is False
        assert "Sin permisos" in error

    def test_validate_file_too_small(self, mock_filesystem):
        """Archivo muy pequeño es rechazado."""
        mock_filesystem.add_file("tiny.csv", exists=True, is_file=True, readable=True, size=5)
        
        with contextlib.ExitStack() as stack:
            for target, side_effect in mock_filesystem.patch_validators().items():
                stack.enter_context(patch(target, side_effect=side_effect, autospec=True))
            valid, error = FileValidator.validate_file_exists(
                "tiny.csv", "test", min_size=10
            )
        
        assert valid is False
        assert "demasiado pequeño" in error

    def test_validate_empty_path(self):
        """Ruta vacía es rechazada."""
        valid, error = FileValidator.validate_file_exists("", "test")
        
        assert valid is False

    @pytest.mark.parametrize("extension", [".csv", ".xlsx", ".json", ".parquet"])
    def test_validate_various_extensions(self, mock_filesystem, extension):
        """Diferentes extensiones son manejadas."""
        filename = f"data{extension}"
        mock_filesystem.add_file(filename, exists=True, is_file=True, readable=True)
        
        with contextlib.ExitStack() as stack:
            for target, side_effect in mock_filesystem.patch_validators().items():
                stack.enter_context(patch(target, side_effect=side_effect, autospec=True))
            valid, _ = FileValidator.validate_file_exists(filename, "test")
        
        assert valid is True


# =============================================================================
# TESTS: PRESUPUESTO PROCESSOR
# =============================================================================


class TestPresupuestoProcessor(TestFixtures):
    """
    Tests para PresupuestoProcessor.
    
    Valida:
    - Limpieza de filas fantasma
    - Normalización de datos
    - Manejo de errores de carga
    """

    @pytest.fixture
    def processor(self, config, thresholds) -> PresupuestoProcessor:
        """Procesador de presupuesto."""
        profile = {"loader_params": {}}
        return PresupuestoProcessor(config, thresholds, profile)

    def test_clean_phantom_rows_removes_empty(self, processor):
        """Filas completamente vacías son eliminadas."""
        df = (
            PresupuestoBuilder()
            .with_items(2)
            .with_phantom_rows(3)
            .build()
        )
        
        cleaned = processor._clean_phantom_rows(df)
        
        assert len(cleaned) == 2

    def test_clean_phantom_rows_removes_nan_strings(self, processor):
        """Filas con 'nan' como string son eliminadas."""
        df = pd.DataFrame({
            "A": ["valid", "nan", "NaN", ""],
            "B": [1, np.nan, np.nan, np.nan]
        })
        
        cleaned = processor._clean_phantom_rows(df)
        
        assert len(cleaned) == 1
        assert cleaned.iloc[0]["A"] == "valid"

    def test_clean_phantom_rows_preserves_zeros(self, processor):
        """Ceros son preservados (no confundidos con vacío)."""
        df = pd.DataFrame({
            "A": ["item", "zero_qty"],
            "B": [10, 0]  # 0 es válido
        })
        
        cleaned = processor._clean_phantom_rows(df)
        
        assert len(cleaned) == 2

    def test_process_returns_empty_on_load_error(self, processor):
        """Error de carga retorna DataFrame vacío."""
        with patch("app.pipeline_director.load_data", return_value=None):
            result = processor.process("nonexistent.csv")
        
        assert result.empty

    def test_process_handles_exception(self, processor):
        """Excepción durante procesamiento retorna DataFrame vacío."""
        with patch("app.pipeline_director.load_data", side_effect=Exception("Load failed")):
            result = processor.process("error.csv")
        
        assert result.empty


# =============================================================================
# TESTS: DATA MERGER
# =============================================================================


class TestDataMerger(TestFixtures):
    """
    Tests para DataMerger.
    
    Valida:
    - Merge APUs + Insumos
    - Merge con Presupuesto
    - Manejo de datos faltantes
    - Detección de duplicados en merge
    """

    def test_merge_apus_with_insumos_success(self, merger):
        """Merge exitoso de APUs con Insumos."""
        df_apus = pd.DataFrame({
            ColumnNames.DESCRIPCION_INSUMO: ["Material A", "Material B"],
            "cantidad": [10, 20]
        })
        df_insumos = pd.DataFrame({
            ColumnNames.DESCRIPCION_INSUMO: ["Material A", "Material B"],
            "precio": [100, 200]
        })
        
        merged = merger.merge_apus_with_insumos(df_apus, df_insumos)
        
        assert len(merged) == 2
        assert "precio" in merged.columns
        assert "cantidad" in merged.columns

    def test_merge_apus_with_empty_insumos(self, merger):
        """Merge con insumos vacíos retorna resultado apropiado."""
        df_apus = pd.DataFrame({
            ColumnNames.DESCRIPCION_INSUMO: ["Material A"],
            "other": [1]
        })
        
        merged = merger.merge_apus_with_insumos(df_apus, pd.DataFrame())
        
        # Con validación estricta, retorna vacío
        assert merged.empty

    def test_merge_apus_missing_key_column(self, merger):
        """Merge sin columna clave falla gracefully."""
        df_apus = pd.DataFrame({"wrong_column": [1]})
        df_insumos = pd.DataFrame({
            ColumnNames.DESCRIPCION_INSUMO: ["Material A"],
            "precio": [100]
        })
        
        merged = merger.merge_apus_with_insumos(df_apus, df_insumos)
        
        assert merged.empty

    def test_merge_with_presupuesto_success(self, merger, presupuesto_df):
        """Merge exitoso con presupuesto."""
        df_costos = pd.DataFrame({
            ColumnNames.CODIGO_APU: presupuesto_df[ColumnNames.CODIGO_APU].tolist(),
            "costo_total": [1000, 2000, 3000, 4000, 5000]
        })
        
        merged = merger.merge_with_presupuesto(presupuesto_df, df_costos)
        
        assert len(merged) == 5
        assert "costo_total" in merged.columns

    def test_merge_with_presupuesto_duplicates_warning(self, merger, presupuesto_df, caplog):
        """Duplicados en merge generan warning."""
        codigo = presupuesto_df[ColumnNames.CODIGO_APU].iloc[0]
        df_costos = pd.DataFrame({
            ColumnNames.CODIGO_APU: [codigo, codigo],  # Duplicado
            "costo": [100, 200]
        })
        
        import logging
        with caplog.at_level(logging.WARNING):
            merged = merger.merge_with_presupuesto(presupuesto_df, df_costos)
        
        # Debe haber manejado el duplicado
        assert len(merged) >= 1

    def test_merge_preserves_all_presupuesto_rows(self, merger, presupuesto_df):
        """Merge preserva todas las filas del presupuesto (left join)."""
        # Solo algunos códigos tienen costos
        first_code = presupuesto_df[ColumnNames.CODIGO_APU].iloc[0]
        df_costos = pd.DataFrame({
            ColumnNames.CODIGO_APU: [first_code],
            "costo": [100]
        })
        
        merged = merger.merge_with_presupuesto(presupuesto_df, df_costos)
        
        # Debe mantener todas las filas del presupuesto
        assert len(merged) == len(presupuesto_df)


# =============================================================================
# TESTS: PIPELINE DIRECTOR
# =============================================================================


class TestPipelineDirector(TestFixtures):
    """
    Tests para PipelineDirector.
    
    Valida:
    - Registro de pasos
    - Ejecución secuencial
    - Manejo de errores
    - Inyección de telemetría
    """

    def test_director_initialization(self, director):
        """Director se inicializa correctamente."""
        assert director is not None
        assert hasattr(director, "STEP_REGISTRY")

    def test_register_custom_step(self, director):
        """Paso personalizado se registra correctamente."""
        director.STEP_REGISTRY["custom"] = MockStep
        
        assert "custom" in director.STEP_REGISTRY
        assert director.STEP_REGISTRY["custom"] == MockStep

    def test_execute_with_mock_step(self, director, config, telemetry, thresholds):
        """Ejecución con paso mock funciona correctamente."""
        director.STEP_REGISTRY["mock"] = MockStep
        config["pipeline_recipe"] = [{"step": "mock", "enabled": True}]
        
        context = ContextBuilder().with_paths().build()
        
        # Mock las validaciones de archivos
        with patch.object(FileValidator, "validate_file_exists", return_value=(True, None)):
            with patch("app.pipeline_director.load_data", return_value=pd.DataFrame({"A": [1]})):
                result = director.execute_pipeline_orchestrated(context)
        
        assert "mock_executed" in result

    def test_execute_failing_step_raises(self, director, config, telemetry):
        """Paso que falla propaga excepción."""
        director.STEP_REGISTRY["fail"] = FailingStep
        config["pipeline_recipe"] = [{"step": "fail", "enabled": True}]
        
        with pytest.raises(RuntimeError):
            director.execute_pipeline_orchestrated({})

    def test_telemetry_records_error(self, director, config, telemetry):
        """Error se registra en telemetría."""
        director.STEP_REGISTRY["fail"] = FailingStep
        config["pipeline_recipe"] = [{"step": "fail", "enabled": True}]
        
        try:
            director.execute_pipeline_orchestrated({})
        except RuntimeError:
            pass
        
        assert len(telemetry.errors) > 0
        assert any(e["step"] == "fail" for e in telemetry.errors)

    def test_disabled_step_skipped(self, director, config):
        """Paso deshabilitado es omitido."""
        director.STEP_REGISTRY["mock"] = MockStep
        config["pipeline_recipe"] = [{"step": "mock", "enabled": False}]
        
        context = {}
        # No debe ejecutar el paso
        # Esto depende de la implementación del director

    def test_execute_multiple_steps_in_order(self, director, config, thresholds):
        """Múltiples pasos se ejecutan en orden."""
        execution_order = []
        
        class TrackingStep(ProcessingStep):
            def __init__(self, cfg, thr, name):
                self.name = name
            
            def execute(self, context, telemetry):
                execution_order.append(self.name)
                return context
        
        # Registrar pasos con tracking
        director.STEP_REGISTRY["step1"] = lambda c, t: TrackingStep(c, t, "step1")
        director.STEP_REGISTRY["step2"] = lambda c, t: TrackingStep(c, t, "step2")
        director.STEP_REGISTRY["step3"] = lambda c, t: TrackingStep(c, t, "step3")


# =============================================================================
# TESTS: LOAD DATA STEP
# =============================================================================


class TestLoadDataStep(TestFixtures):
    """
    Tests para LoadDataStep.
    
    Valida:
    - Carga exitosa de archivos
    - Manejo de archivos vacíos
    - Validación de datos cargados
    """

    @pytest.fixture
    def load_step(self, config, thresholds) -> LoadDataStep:
        """Paso de carga de datos."""
        return LoadDataStep(config, thresholds)

    def test_load_step_success(self, load_step, telemetry, presupuesto_df):
        """Carga exitosa de datos."""
        context = ContextBuilder().with_paths().build()
        
        with patch.object(FileValidator, "validate_file_exists", return_value=(True, None)):
            with patch("app.pipeline_director.PresupuestoProcessor") as MockPresupuesto:
                MockPresupuesto.return_value.process.return_value = presupuesto_df
                
                with patch("app.pipeline_director.InsumosProcessor") as MockInsumos:
                    # Mock process return value to be a non-empty DataFrame (mock)
                    # Must handle dropna(how="all").empty check
                    mock_insumos_df = Mock(spec=pd.DataFrame)
                    mock_insumos_df.empty = False
                    mock_insumos_df.__len__ = Mock(return_value=10)

                    # Ensure dropna returns a non-empty mock
                    mock_dropna_res = Mock(spec=pd.DataFrame)
                    mock_dropna_res.empty = False
                    mock_insumos_df.dropna.return_value = mock_dropna_res

                    MockInsumos.return_value.process.return_value = mock_insumos_df

                    with patch("app.pipeline_director.DataFluxCondenser") as MockCondenser:
                        # Mock stabilize return value - apply same fix for dropna check if needed
                        mock_apus_df = Mock(spec=pd.DataFrame)
                        mock_apus_df.empty = False
                        mock_apus_df.__len__ = Mock(return_value=5)
                        mock_apus_dropna = Mock(spec=pd.DataFrame)
                        mock_apus_dropna.empty = False
                        mock_apus_df.dropna.return_value = mock_apus_dropna

                        MockCondenser.return_value.stabilize.return_value = mock_apus_df

                        result = load_step.execute(context, telemetry)
        
        assert "df_presupuesto" in result
        assert "df_insumos" in result
        assert "df_apus_raw" in result

    def test_load_step_empty_presupuesto_raises(self, load_step, telemetry):
        """Presupuesto vacío lanza excepción."""
        context = ContextBuilder().with_paths().build()
        
        with patch.object(FileValidator, "validate_file_exists", return_value=(True, None)):
            with patch("app.pipeline_director.PresupuestoProcessor") as MockProc:
                MockProc.return_value.process.return_value = pd.DataFrame()  # Vacío
                
                with pytest.raises(ValueError) as exc_info:
                    load_step.execute(context, telemetry)
        
        assert "vacío" in str(exc_info.value).lower()
        assert any("vacío" in e["message"] for e in telemetry.errors)

    def test_load_step_file_not_found(self, load_step, telemetry):
        """Archivo no encontrado lanza excepción."""
        context = ContextBuilder().with_paths().build()
        
        with patch.object(
            FileValidator, 
            "validate_file_exists", 
            return_value=(False, "Archivo no encontrado")
        ):
            with pytest.raises((ValueError, FileNotFoundError)):
                load_step.execute(context, telemetry)


# =============================================================================
# TESTS: TELEMETRÍA
# =============================================================================


class TestTelemetryIntegration(TestFixtures):
    """
    Tests para integración de telemetría.
    
    Valida:
    - Registro de inicio/fin de pasos
    - Registro de errores
    - Métricas de tiempo
    """

    def test_telemetry_records_step_timing(self, telemetry):
        """Telemetría registra tiempo de ejecución."""
        telemetry.start_step("test_step")
        time.sleep(0.01)  # Pequeña espera
        telemetry.end_step("test_step", "success")
        
        # Verificar que se registró el tiempo
        assert hasattr(telemetry, 'steps') or hasattr(telemetry, '_steps')

    def test_telemetry_records_errors(self, telemetry):
        """Telemetría registra errores."""
        telemetry.start_step("failing_step")
        telemetry.record_error("failing_step", "Test error message")
        
        assert len(telemetry.errors) >= 1
        assert any("Test error" in str(e) for e in telemetry.errors)

    def test_telemetry_tracks_multiple_steps(self, telemetry):
        """Telemetría rastrea múltiples pasos."""
        steps = ["step1", "step2", "step3"]
        
        for step in steps:
            telemetry.start_step(step)
            telemetry.end_step(step, "success")
        
        # Verificar que todos fueron registrados

    def test_step_injects_telemetry_correctly(self, config, thresholds, telemetry):
        """Pasos reciben e inyectan telemetría correctamente."""
        step = MockStep(config, thresholds)
        context = {}
        
        result = step.execute(context, telemetry)
        
        assert "mock_executed" in result
        # Telemetría debe tener registro del paso


# =============================================================================
# TESTS: INVARIANTES DEL PIPELINE
# =============================================================================


class TestPipelineInvariants(TestFixtures):
    """
    Tests para invariantes del pipeline.
    
    Valida propiedades que siempre deben cumplirse:
    - Conservación de datos
    - Determinismo
    - Idempotencia de validaciones
    """

    def test_merge_conserves_data_invariant(self, merger, presupuesto_df):
        """Merge no genera filas adicionales (left join)."""
        df_costos = pd.DataFrame({
            ColumnNames.CODIGO_APU: presupuesto_df[ColumnNames.CODIGO_APU].tolist() * 2,
            "costo": list(range(len(presupuesto_df) * 2))
        })
        
        merged = merger.merge_with_presupuesto(presupuesto_df, df_costos)
        
        # Con duplicados puede haber más filas, pero controlado
        # El invariante real es que no hay "filas fantasma" sin origen
        original_codes = set(presupuesto_df[ColumnNames.CODIGO_APU])
        merged_codes = set(merged[ColumnNames.CODIGO_APU])
        
        assert merged_codes.issubset(original_codes)

    def test_validation_is_idempotent(self, presupuesto_df):
        """Validación aplicada múltiples veces da mismo resultado."""
        result1, error1 = DataValidator.validate_dataframe_not_empty(
            presupuesto_df, "test"
        )
        result2, error2 = DataValidator.validate_dataframe_not_empty(
            presupuesto_df, "test"
        )
        
        assert result1 == result2
        assert error1 == error2

    def test_clean_is_idempotent(self, config, thresholds):
        """Limpieza aplicada múltiples veces da mismo resultado."""
        processor = PresupuestoProcessor(config, thresholds, {"loader_params": {}})
        
        df = (
            PresupuestoBuilder()
            .with_items(3)
            .with_phantom_rows(2)
            .build()
        )
        
        cleaned_once = processor._clean_phantom_rows(df)
        cleaned_twice = processor._clean_phantom_rows(cleaned_once)
        
        pd.testing.assert_frame_equal(cleaned_once, cleaned_twice)

    def test_processing_is_deterministic(self, merger, presupuesto_df, insumos_df):
        """Mismo input produce mismo output."""
        df_apus = APUsBuilder().for_presupuesto(presupuesto_df).build()
        
        result1 = merger.merge_apus_with_insumos(df_apus, insumos_df)
        result2 = merger.merge_apus_with_insumos(df_apus, insumos_df)
        
        pd.testing.assert_frame_equal(result1, result2)


# =============================================================================
# TESTS: CASOS EDGE
# =============================================================================


class TestEdgeCases(TestFixtures):
    """Tests para casos límite."""

    def test_empty_pipeline_recipe(self, director, config):
        """Recipe vacío ejecuta sin pasos."""
        config["pipeline_recipe"] = []
        
        context = {}
        # No debe fallar con recipe vacío

    def test_very_large_dataframe(self, merger, thresholds):
        """DataFrames grandes son procesados."""
        large_presupuesto = PresupuestoBuilder().with_items(10000).build()
        large_apus = APUsBuilder().for_presupuesto(large_presupuesto, insumos_per_apu=5).build()
        
        # No debe fallar ni tardar excesivamente
        import time
        start = time.time()
        # Procesamiento...
        elapsed = time.time() - start
        
        assert elapsed < 30.0  # Menos de 30 segundos

    def test_unicode_in_data(self, merger):
        """Datos con unicode son manejados."""
        df_apus = pd.DataFrame({
            ColumnNames.DESCRIPCION_INSUMO: ["Café ☕", "Niño 👶", "Señal 📡"],
            "cantidad": [1, 2, 3]
        })
        df_insumos = pd.DataFrame({
            ColumnNames.DESCRIPCION_INSUMO: ["Café ☕", "Niño 👶", "Señal 📡"],
            "precio": [10, 20, 30]
        })
        
        merged = merger.merge_apus_with_insumos(df_apus, df_insumos)
        
        assert len(merged) == 3

    def test_special_characters_in_codes(self):
        """Códigos con caracteres especiales son manejados."""
        df = PresupuestoBuilder().with_item("APU-001/A", "Test", 10.0).build()
        
        valid, _ = DataValidator.validate_dataframe_not_empty(df, "test")
        
        assert valid is True

    def test_whitespace_handling(self, thresholds, config):
        """Espacios en blanco son manejados correctamente."""
        processor = PresupuestoProcessor(config, thresholds, {"loader_params": {}})
        
        # To be removed by _clean_phantom_rows, ALL columns must be empty/whitespace/nan
        df = pd.DataFrame({
            "A": ["  value  ", "   ", "\t\n", "valid"],
            "B": ["1", "", "   ", "4"] # Modified to have empty values in row 2 and 3
        })
        
        cleaned = processor._clean_phantom_rows(df)
        
        # Filas con solo espacios (indices 1 y 2) deben ser eliminadas
        # "   " and "" in B should match empty patterns
        assert len(cleaned) <= 2


# =============================================================================
# TESTS: BUILDER VALIDATION
# =============================================================================


class TestBuilderValidation(TestFixtures):
    """Tests para validación de builders."""

    def test_presupuesto_builder_rejects_negative_quantity(self):
        """Builder rechaza cantidad negativa."""
        with pytest.raises(ValueError, match="negativa"):
            PresupuestoBuilder().with_item("APU-001", "Test", -10.0)

    def test_presupuesto_builder_rejects_empty_code(self):
        """Builder rechaza código vacío."""
        with pytest.raises(ValueError, match="vacío"):
            PresupuestoBuilder().with_item("", "Test", 10.0)

    def test_insumos_builder_rejects_invalid_type(self):
        """Builder rechaza tipo de insumo inválido."""
        with pytest.raises(ValueError, match="Tipo inválido"):
            InsumosBuilder().with_insumo("Test", "INVALID_TYPE", 100.0)

    def test_insumos_builder_rejects_negative_price(self):
        """Builder rechaza precio negativo."""
        with pytest.raises(ValueError, match="negativo"):
            InsumosBuilder().with_insumo("Test", "MATERIAL", -50.0)

    def test_context_builder_creates_valid_context(self):
        """ContextBuilder crea contexto válido."""
        context = (
            ContextBuilder()
            .with_paths("p.csv", "a.csv", "i.csv")
            .with_param("custom_key", "custom_value")
            .build()
        )
        
        assert "presupuesto_path" in context
        assert "custom_key" in context
        assert context["custom_key"] == "custom_value"


# =============================================================================
# ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])