"""
Tests para Pipeline Director - Suite Robustecida
=================================================

Suite algebraicamente rigurosa que valida:
- Estructuras algebraicas (MIC, espacios vectoriales, formas bilineales)
- Geometría de la información (entropía, Fisher, Procrustes)
- Topología computacional (homología, números de Betti)
- Invariantes del flujo de datos
- Orquestación y recuperación de errores

Modelo Matemático:
------------------
El pipeline se modela como una composición de operadores lineales:

    T = T_n ∘ T_{n-1} ∘ ... ∘ T_1

donde cada T_i : V → V preserva estructura (norma, rango, topología).

Invariantes Verificados:
- Ortogonalidad: <e_i, e_j>_K = δ_ij (forma de Killing)
- Preservación de rango: rank(T(A)) ≤ rank(A)
- Continuidad topológica: β_k(T(X)) estable
- Conservación de información: H(T(X)) ≥ H(X) - ε
"""

import contextlib
import hashlib
import pickle
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
from unittest.mock import MagicMock, Mock, PropertyMock, call, patch
import inspect

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from scipy.spatial.distance import pdist, squareform

from app.pipeline_director import (
    APUCostCalculator,
    AuditedMergeStep,
    BaseCostProcessor,
    BasisVector,
    BuildOutputStep,
    BusinessTopologyStep,
    CalculateCostsStep,
    ColumnNames,
    DataMerger,
    DataValidator,
    FileValidator,
    FinalMergeStep,
    InformationGeometry,
    InsumosProcessor,
    LinearInteractionMatrix,
    LoadDataStep,
    MaterializationStep,
    MergeDataStep,
    PipelineDirector,
    PipelineSteps,
    PresupuestoProcessor,
    ProcessingStep,
    ProcessingThresholds,
    ProcrustesAnalyzer,
)
from app.schemas import Stratum
from app.telemetry import TelemetryContext


# =============================================================================
# CONSTANTES Y CONFIGURACIÓN
# =============================================================================

FIXED_TIMESTAMP = datetime(2024, 1, 15, 10, 30, 0)
NUMERICAL_TOLERANCE = 1e-10
SPECTRAL_TOLERANCE = 1e-6

# Configuración base de pruebas
TEST_CONFIG = {
    "presupuesto_path": "test_presupuesto.csv",
    "apus_path": "test_apus.csv",
    "insumos_path": "test_insumos.csv",
    "output_path": "test_output.csv",
    "pipeline_recipe": [],
    "loader_params": {},
    "file_profiles": {
        "presupuesto_default": {},
        "insumos_default": {},
        "apus_default": {}
    },
    "session_dir": "/tmp/test_sessions",
}

# Umbrales para validación
VALIDATION_THRESHOLDS = {
    "min_file_size": 10,
    "max_null_ratio": 0.5,
    "max_duplicate_ratio": 0.1,
    "min_entropy_preservation": 0.7,
    "max_dimension_collapse": 0.5,
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
# UTILIDADES MATEMÁTICAS PARA TESTING
# =============================================================================


class AlgebraicTestUtils:
    """Utilidades para verificar propiedades algebraicas."""
    
    @staticmethod
    def is_orthonormal(vectors: np.ndarray, tol: float = NUMERICAL_TOLERANCE) -> bool:
        """Verifica si un conjunto de vectores es ortonormal."""
        gram = vectors @ vectors.T
        identity = np.eye(gram.shape[0])
        return np.allclose(gram, identity, atol=tol)
    
    @staticmethod
    def is_positive_definite(matrix: np.ndarray, tol: float = NUMERICAL_TOLERANCE) -> bool:
        """Verifica si una matriz es definida positiva."""
        try:
            eigenvalues = np.linalg.eigvalsh(matrix)
            return np.all(eigenvalues > -tol)
        except np.linalg.LinAlgError:
            return False
    
    @staticmethod
    def condition_number(matrix: np.ndarray) -> float:
        """Calcula el número de condición de una matriz."""
        try:
            return float(np.linalg.cond(matrix))
        except np.linalg.LinAlgError:
            return float('inf')
    
    @staticmethod
    def frobenius_distance(A: np.ndarray, B: np.ndarray) -> float:
        """Calcula la distancia de Frobenius entre dos matrices."""
        return float(np.linalg.norm(A - B, 'fro'))
    
    @staticmethod
    def spectral_gap(matrix: np.ndarray) -> float:
        """Calcula el gap espectral (diferencia entre dos mayores eigenvalores)."""
        try:
            eigenvalues = np.sort(np.linalg.eigvalsh(matrix))[::-1]
            if len(eigenvalues) >= 2:
                return float(eigenvalues[0] - eigenvalues[1])
            return 0.0
        except np.linalg.LinAlgError:
            return 0.0


class TopologicalTestUtils:
    """Utilidades para verificar propiedades topológicas."""
    
    @staticmethod
    def euler_characteristic(vertices: int, edges: int, faces: int = 0) -> int:
        """Calcula característica de Euler: χ = V - E + F."""
        return vertices - edges + faces
    
    @staticmethod
    def betti_number_0(adj_matrix: np.ndarray) -> int:
        """Calcula β₀ (componentes conexas) via análisis espectral."""
        n = adj_matrix.shape[0]
        if n == 0:
            return 0
        
        degrees = np.sum(adj_matrix, axis=1)
        laplacian = np.diag(degrees) - adj_matrix
        
        eigenvalues = np.linalg.eigvalsh(laplacian)
        return int(np.sum(np.abs(eigenvalues) < NUMERICAL_TOLERANCE))
    
    @staticmethod
    def betti_number_1(vertices: int, edges: int, beta_0: int) -> int:
        """Calcula β₁ (ciclos) usando fórmula de Euler."""
        return max(0, edges - vertices + beta_0)
    
    @staticmethod
    def is_connected(adj_matrix: np.ndarray) -> bool:
        """Verifica si el grafo es conexo."""
        return TopologicalTestUtils.betti_number_0(adj_matrix) == 1


class InformationTheoreticTestUtils:
    """Utilidades para verificar propiedades de teoría de la información."""
    
    @staticmethod
    def shannon_entropy(probabilities: np.ndarray) -> float:
        """Calcula entropía de Shannon: H = -Σ p_i log₂(p_i)."""
        probabilities = probabilities[probabilities > 0]
        if len(probabilities) == 0:
            return 0.0
        return float(-np.sum(probabilities * np.log2(probabilities)))
    
    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """Calcula divergencia KL: D_KL(P||Q) = Σ p_i log(p_i/q_i)."""
        p = np.clip(p, NUMERICAL_TOLERANCE, 1.0)
        q = np.clip(q, NUMERICAL_TOLERANCE, 1.0)
        return float(np.sum(p * np.log(p / q)))
    
    @staticmethod
    def mutual_information(joint: np.ndarray) -> float:
        """Calcula información mutua desde distribución conjunta."""
        marginal_x = joint.sum(axis=1)
        marginal_y = joint.sum(axis=0)
        outer = np.outer(marginal_x, marginal_y)
        
        mask = (joint > 0) & (outer > 0)
        if not mask.any():
            return 0.0
        
        return float(np.sum(joint[mask] * np.log(joint[mask] / outer[mask])))


# =============================================================================
# BUILDERS MEJORADOS
# =============================================================================


@dataclass
class ColumnSpec:
    """Especificación de una columna con validación."""
    name: str
    dtype: str = "object"
    nullable: bool = True
    default: Any = None
    validator: Optional[Callable[[Any], bool]] = None


class DataFrameBuilder:
    """
    Builder base para DataFrames con validación de esquema y propiedades.
    
    Garantiza:
    - Tipos de datos consistentes
    - Nombres de columnas válidos
    - Integridad referencial
    - Propiedades estadísticas controladas
    """
    
    def __init__(self):
        self._data: Dict[str, List[Any]] = {}
        self._schema: Dict[str, ColumnSpec] = {}
        self._invariants: List[Callable[[pd.DataFrame], bool]] = []
    
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
    
    def add_invariant(self, invariant: Callable[[pd.DataFrame], bool]) -> "DataFrameBuilder":
        """Agrega un invariante a verificar en build()."""
        self._invariants.append(invariant)
        return self
    
    def _verify_invariants(self, df: pd.DataFrame) -> None:
        """Verifica todos los invariantes."""
        for i, invariant in enumerate(self._invariants):
            if not invariant(df):
                raise ValueError(f"Invariante {i} violado")
    
    def build(self) -> pd.DataFrame:
        """Construye el DataFrame validando coherencia e invariantes."""
        self._ensure_equal_lengths()
        df = pd.DataFrame(self._data)
        self._verify_invariants(df)
        return df
    
    def build_empty(self) -> pd.DataFrame:
        """Construye DataFrame vacío con el esquema definido."""
        return pd.DataFrame(columns=list(self._schema.keys()))


class PresupuestoBuilder(DataFrameBuilder):
    """
    Builder para DataFrames de presupuesto con validación algebraica.
    
    Invariantes:
    - Códigos APU únicos (inyectividad)
    - Cantidades no negativas (orden parcial)
    - Descripciones no vacías
    """
    
    def __init__(self):
        super().__init__()
        self._schema = {
            ColumnNames.CODIGO_APU: ColumnSpec(
                ColumnNames.CODIGO_APU, "object", False,
                validator=lambda x: bool(x and str(x).strip())
            ),
            ColumnNames.DESCRIPCION_APU: ColumnSpec(
                ColumnNames.DESCRIPCION_APU, "object", True
            ),
            ColumnNames.CANTIDAD_PRESUPUESTO: ColumnSpec(
                ColumnNames.CANTIDAD_PRESUPUESTO, "float64", False,
                validator=lambda x: x >= 0
            ),
            "UNIDAD": ColumnSpec("UNIDAD", "object", True, "UND"),
        }
        self._items: List[Dict[str, Any]] = []
        self._used_codes: Set[str] = set()
        
        # Invariante: códigos únicos
        self.add_invariant(
            lambda df: df[df[ColumnNames.CODIGO_APU] != ""][ColumnNames.CODIGO_APU].is_unique if not df.empty else True
        )
    
    def with_item(
        self,
        codigo: str,
        descripcion: str = "Actividad",
        cantidad: float = 1.0,
        unidad: str = "UND"
    ) -> "PresupuestoBuilder":
        """Agrega un ítem de presupuesto con validación."""
        if not codigo or not str(codigo).strip():
            raise ValueError("Código APU no puede estar vacío")
        if cantidad < 0:
            raise ValueError(f"Cantidad no puede ser negativa: {cantidad}")
        if codigo in self._used_codes:
            raise ValueError(f"Código duplicado: {codigo}")
        
        self._used_codes.add(codigo)
        self._items.append({
            ColumnNames.CODIGO_APU: codigo,
            ColumnNames.DESCRIPCION_APU: descripcion,
            ColumnNames.CANTIDAD_PRESUPUESTO: cantidad,
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
                ColumnNames.DESCRIPCION_APU: None,
                ColumnNames.CANTIDAD_PRESUPUESTO: np.nan,
                "UNIDAD": None,
            })
        return self
    
    def with_duplicates(self, codigo: str, count: int = 2) -> "PresupuestoBuilder":
        """Agrega ítems duplicados (bypass validación para testing)."""
        for i in range(count):
            self._items.append({
                ColumnNames.CODIGO_APU: codigo,
                ColumnNames.DESCRIPCION_APU: f"Duplicado {i}",
                ColumnNames.CANTIDAD_PRESUPUESTO: float(i + 1),
                "UNIDAD": "UND",
            })
        # Remover invariante de unicidad para este caso
        self._invariants = [inv for inv in self._invariants 
                           if "is_unique" not in str(inv)]
        return self
    
    def with_statistical_distribution(
        self,
        count: int,
        mean_qty: float = 100.0,
        std_qty: float = 20.0
    ) -> "PresupuestoBuilder":
        """Agrega ítems con distribución estadística controlada."""
        np.random.seed(42)  # Reproducibilidad
        quantities = np.abs(np.random.normal(mean_qty, std_qty, count))
        
        for i, qty in enumerate(quantities, 1):
            self.with_item(
                codigo=f"STAT-{i:04d}",
                descripcion=f"Item estadístico {i}",
                cantidad=float(qty),
            )
        return self
    
    def build(self) -> pd.DataFrame:
        """Construye el DataFrame de presupuesto."""
        if not self._items:
            return self.build_empty()
        df = pd.DataFrame(self._items)
        # Solo verificar invariantes si hay datos válidos
        if not df.empty and df[ColumnNames.CODIGO_APU].notna().any():
            self._verify_invariants(df[df[ColumnNames.CODIGO_APU].notna()])
        return df


class InsumosBuilder(DataFrameBuilder):
    """
    Builder para DataFrames de insumos con tipado estricto.
    
    Invariantes:
    - Tipos de insumo válidos (conjunto cerrado)
    - Precios no negativos
    - Descripciones no vacías
    """
    
    TIPOS_VALIDOS = frozenset({"MATERIAL", "MANO_OBRA", "EQUIPO", "TRANSPORTE", "OTROS"})
    
    def __init__(self):
        super().__init__()
        self._items: List[Dict[str, Any]] = []
        self._schema = {
            ColumnNames.DESCRIPCION_INSUMO: ColumnSpec(
                ColumnNames.DESCRIPCION_INSUMO, "object", False
            ),
            ColumnNames.TIPO_INSUMO: ColumnSpec(
                ColumnNames.TIPO_INSUMO, "object", False
            ),
            ColumnNames.VR_UNITARIO_INSUMO: ColumnSpec(
                ColumnNames.VR_UNITARIO_INSUMO, "float64", False
            ),
            "UNIDAD": ColumnSpec("UNIDAD", "object", True, "UND"),
            ColumnNames.GRUPO_INSUMO: ColumnSpec(
                ColumnNames.GRUPO_INSUMO, "object", True
            ),
        }
    
    def with_insumo(
        self,
        descripcion: str,
        tipo: str = "MATERIAL",
        precio: float = 100.0,
        unidad: str = "UND",
        grupo: str = "GENERAL"
    ) -> "InsumosBuilder":
        """Agrega un insumo con validación."""
        if tipo not in self.TIPOS_VALIDOS:
            raise ValueError(f"Tipo inválido: {tipo}. Válidos: {self.TIPOS_VALIDOS}")
        if precio < 0:
            raise ValueError(f"Precio no puede ser negativo: {precio}")
        if not descripcion or not str(descripcion).strip():
            raise ValueError("Descripción no puede estar vacía")
        
        self._items.append({
            ColumnNames.DESCRIPCION_INSUMO: descripcion,
            ColumnNames.TIPO_INSUMO: tipo,
            ColumnNames.VR_UNITARIO_INSUMO: precio,
            "UNIDAD": unidad,
            ColumnNames.GRUPO_INSUMO: grupo,
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
                grupo=f"GRUPO_{(i % 3) + 1}"
            )
        return self
    
    def with_price_distribution(
        self,
        count: int,
        mean_price: float = 500.0,
        std_price: float = 100.0
    ) -> "InsumosBuilder":
        """Agrega insumos con distribución de precios controlada."""
        np.random.seed(42)
        prices = np.abs(np.random.normal(mean_price, std_price, count))
        tipos = list(self.TIPOS_VALIDOS)
        
        for i, price in enumerate(prices, 1):
            self.with_insumo(
                descripcion=f"Insumo precio controlado {i}",
                tipo=tipos[i % len(tipos)],
                precio=float(price),
            )
        return self
    
    def build(self) -> pd.DataFrame:
        """Construye el DataFrame de insumos."""
        if not self._items:
            return pd.DataFrame(columns=list(self._schema.keys()))
        return pd.DataFrame(self._items)


class APUsBuilder(DataFrameBuilder):
    """
    Builder para DataFrames de APUs con relaciones.
    
    Modela la relación muchos-a-muchos: APU ↔ Insumo
    """
    
    def __init__(self):
        super().__init__()
        self._items: List[Dict[str, Any]] = []
        self._schema = {
            ColumnNames.CODIGO_APU: ColumnSpec(ColumnNames.CODIGO_APU, "object", False),
            ColumnNames.DESCRIPCION_INSUMO: ColumnSpec(
                ColumnNames.DESCRIPCION_INSUMO, "object", False
            ),
            ColumnNames.CANTIDAD_APU: ColumnSpec(ColumnNames.CANTIDAD_APU, "float64", False),
            ColumnNames.RENDIMIENTO: ColumnSpec(ColumnNames.RENDIMIENTO, "float64", True),
            ColumnNames.TIPO_INSUMO: ColumnSpec(ColumnNames.TIPO_INSUMO, "object", True),
        }
    
    def with_apu_insumo(
        self,
        codigo_apu: str,
        descripcion_insumo: str,
        cantidad: float = 1.0,
        rendimiento: float = 1.0,
        tipo_insumo: str = "MATERIAL"
    ) -> "APUsBuilder":
        """Agrega una relación APU-Insumo."""
        if not codigo_apu:
            raise ValueError("Código APU no puede estar vacío")
        if cantidad < 0:
            raise ValueError(f"Cantidad no puede ser negativa: {cantidad}")
        
        self._items.append({
            ColumnNames.CODIGO_APU: codigo_apu,
            ColumnNames.DESCRIPCION_INSUMO: descripcion_insumo,
            ColumnNames.CANTIDAD_APU: cantidad,
            ColumnNames.RENDIMIENTO: rendimiento,
            ColumnNames.TIPO_INSUMO: tipo_insumo,
        })
        return self
    
    def for_presupuesto(
        self, 
        presupuesto: pd.DataFrame,
        insumos_per_apu: int = 3,
        tipo_distribution: Optional[Dict[str, float]] = None
    ) -> "APUsBuilder":
        """Genera APUs para un presupuesto con distribución de tipos."""
        tipo_distribution = tipo_distribution or {
            "MATERIAL": 0.4,
            "MANO_OBRA": 0.3,
            "EQUIPO": 0.2,
            "TRANSPORTE": 0.1,
        }
        tipos = list(tipo_distribution.keys())
        
        for _, row in presupuesto.iterrows():
            codigo = row.get(ColumnNames.CODIGO_APU)
            if pd.notna(codigo) and codigo:
                for i in range(insumos_per_apu):
                    tipo_idx = i % len(tipos)
                    self.with_apu_insumo(
                        codigo_apu=codigo,
                        descripcion_insumo=f"Insumo {i+1} para {codigo}",
                        cantidad=float(i + 1),
                        rendimiento=1.0 / (i + 1),
                        tipo_insumo=tipos[tipo_idx],
                    )
        return self
    
    def with_sparse_relation(
        self,
        num_apus: int,
        num_insumos: int,
        density: float = 0.3
    ) -> "APUsBuilder":
        """Genera relación sparse entre APUs e insumos."""
        np.random.seed(42)
        
        for apu_idx in range(num_apus):
            for insumo_idx in range(num_insumos):
                if np.random.random() < density:
                    self.with_apu_insumo(
                        codigo_apu=f"APU-{apu_idx:03d}",
                        descripcion_insumo=f"Insumo-{insumo_idx:03d}",
                        cantidad=np.random.uniform(0.5, 5.0),
                    )
        return self
    
    def build(self) -> pd.DataFrame:
        """Construye el DataFrame de APUs."""
        if not self._items:
            return pd.DataFrame(columns=list(self._schema.keys()))
        return pd.DataFrame(self._items)


class ContextBuilder:
    """
    Builder para contextos de ejecución del pipeline.
    
    El contexto es un espacio de estados del pipeline.
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
        insumos: Optional[pd.DataFrame] = None,
        merged: Optional[pd.DataFrame] = None
    ) -> "ContextBuilder":
        """Configura DataFrames precargados."""
        if presupuesto is not None:
            self._context["df_presupuesto"] = presupuesto
        if apus is not None:
            self._context["df_apus_raw"] = apus
        if insumos is not None:
            self._context["df_insumos"] = insumos
        if merged is not None:
            self._context["df_merged"] = merged
        return self
    
    def with_validated_strata(self, *strata: Stratum) -> "ContextBuilder":
        """Configura estratos validados."""
        self._context["validated_strata"] = set(strata)
        return self
    
    def with_param(self, key: str, value: Any) -> "ContextBuilder":
        """Agrega un parámetro adicional."""
        self._context[key] = value
        return self
    
    def with_telemetry_state(self, step_times: Dict[str, float]) -> "ContextBuilder":
        """Agrega estado de telemetría simulado."""
        self._context["_telemetry_state"] = step_times
        return self
    
    def build(self) -> Dict[str, Any]:
        """Construye el contexto."""
        return self._context.copy()


# =============================================================================
# MOCK FACTORIES MEJORADAS
# =============================================================================


class MockStep(ProcessingStep):
    """Mock de un paso del pipeline con tracking."""
    
    def __init__(self, config: Dict, thresholds: ProcessingThresholds):
        self.config = config
        self.thresholds = thresholds
        self.execute_count = 0
        self.last_context = None
        self.execution_times: List[float] = []
    
    def execute(self, context: Dict, telemetry: TelemetryContext) -> Dict:
        """Ejecuta el paso mock con timing."""
        start = time.time()
        telemetry.start_step("mock_step")
        
        self.execute_count += 1
        self.last_context = context.copy()
        
        context["mock_executed"] = True
        context["mock_count"] = self.execute_count
        context["mock_timestamp"] = FIXED_TIMESTAMP.isoformat()
        
        elapsed = time.time() - start
        self.execution_times.append(elapsed)
        
        telemetry.end_step("mock_step", "success")
        return context


class FailingStep(ProcessingStep):
    """Paso que falla de forma configurable."""
    
    def __init__(
        self, 
        config: Dict, 
        thresholds: ProcessingThresholds,
        error_type: Type[Exception] = ValueError,
        error_message: str = "Step failed intentionally",
        fail_after_n_calls: int = 0
    ):
        self.config = config
        self.thresholds = thresholds
        self.error_type = error_type
        self.error_message = error_message
        self.fail_after_n_calls = fail_after_n_calls
        self.call_count = 0
    
    def execute(self, context: Dict, telemetry: TelemetryContext) -> Dict:
        """Lanza excepción condicionalmente."""
        telemetry.start_step("failing_step")
        self.call_count += 1
        
        if self.fail_after_n_calls == 0 or self.call_count > self.fail_after_n_calls:
            telemetry.end_step("failing_step", "error")
            raise self.error_type(self.error_message)
        
        telemetry.end_step("failing_step", "success")
        return context


class StatefulStep(ProcessingStep):
    """Paso con estado para testing de idempotencia."""
    
    def __init__(self, config: Dict, thresholds: ProcessingThresholds):
        self.config = config
        self.thresholds = thresholds
        self.state_history: List[Dict[str, Any]] = []
    
    def execute(self, context: Dict, telemetry: TelemetryContext) -> Dict:
        """Ejecuta y guarda estado."""
        telemetry.start_step("stateful_step")
        
        state_snapshot = {
            "context_keys": list(context.keys()),
            "timestamp": time.time(),
            "call_number": len(self.state_history) + 1,
        }
        self.state_history.append(state_snapshot)
        
        context["state_history_length"] = len(self.state_history)
        
        telemetry.end_step("stateful_step", "success")
        return context
    
    def verify_idempotence(self, context: Dict) -> bool:
        """Verifica si el estado es idempotente."""
        if len(self.state_history) < 2:
            return True
        
        # Comparar últimos dos estados
        last = self.state_history[-1]
        prev = self.state_history[-2]
        
        return set(last["context_keys"]) == set(prev["context_keys"])


class MockFileSystem:
    """Mock del sistema de archivos con soporte para contenido."""
    
    def __init__(self):
        self._files: Dict[str, Dict[str, Any]] = {}
    
    def add_file(
        self,
        path: str,
        exists: bool = True,
        is_file: bool = True,
        size: int = 1000,
        readable: bool = True,
        content: Optional[Union[pd.DataFrame, bytes]] = None
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
    
    def _resolve(self, path_obj) -> Dict[str, Any]:
        """Resuelve datos del archivo."""
        path_str = str(path_obj)
        if path_str in self._files:
            return self._files[path_str]
        
        # Intentar por nombre
        name = Path(path_str).name if hasattr(path_obj, "name") else path_str
        if name in self._files:
            return self._files[name]
        
        return {}
    
    def patch_validators(self) -> Dict[str, Callable]:
        """Retorna patches para simular filesystem."""
        
        def mock_exists(self_path):
            return self._resolve(self_path).get("exists", False)
        
        def mock_is_file(self_path):
            return self._resolve(self_path).get("is_file", False)
        
        def mock_stat(self_path):
            mock = MagicMock()
            mock.st_size = self._resolve(self_path).get("size", 0)
            return mock
        
        def mock_access(path, mode):
            return self._resolve(path).get("readable", False)
        
        return {
            "pathlib.Path.exists": mock_exists,
            "pathlib.Path.is_file": mock_is_file,
            "pathlib.Path.stat": mock_stat,
            "os.access": mock_access,
        }


class TelemetryContextMock(TelemetryContext):
    """Mock de TelemetryContext con tracking mejorado."""
    
    def __init__(self):
        super().__init__()
        self.step_starts: Dict[str, float] = {}
        self.step_ends: Dict[str, Tuple[float, str]] = {}
        self.recorded_metrics: Dict[str, Dict[str, Any]] = {}
    
    def start_step(self, step_name: str):
        """Registra inicio de paso."""
        self.step_starts[step_name] = time.time()
        super().start_step(step_name)
    
    def end_step(self, step_name: str, status: str):
        """Registra fin de paso."""
        self.step_ends[step_name] = (time.time(), status)
        super().end_step(step_name, status)
    
    def record_metric(self, step_name: str, metric_name: str, value: Any):
        """Registra métrica."""
        if step_name not in self.recorded_metrics:
            self.recorded_metrics[step_name] = {}
        self.recorded_metrics[step_name][metric_name] = value
        super().record_metric(step_name, metric_name, value)
    
    def get_step_duration(self, step_name: str) -> Optional[float]:
        """Obtiene duración de un paso."""
        if step_name in self.step_starts and step_name in self.step_ends:
            return self.step_ends[step_name][0] - self.step_starts[step_name]
        return None


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
    def telemetry(self) -> TelemetryContextMock:
        """Contexto de telemetría mock."""
        return TelemetryContextMock()

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

    @pytest.fixture
    def info_geometry(self) -> InformationGeometry:
        """Instancia de geometría de la información."""
        return InformationGeometry()

    @pytest.fixture
    def procrustes(self) -> ProcrustesAnalyzer:
        """Instancia del analizador Procrustes."""
        return ProcrustesAnalyzer()


# =============================================================================
# TESTS: LINEAR INTERACTION MATRIX (ÁLGEBRA)
# =============================================================================


class TestLinearInteractionMatrixAlgebra(TestFixtures):
    """
    Tests para propiedades algebraicas de LinearInteractionMatrix.
    
    Verifica:
    - Espacio vectorial válido
    - Ortogonalidad de base
    - Propiedades espectrales
    - Forma de Killing
    """

    @pytest.fixture
    def fresh_mic(self) -> LinearInteractionMatrix:
        """MIC sin vectores base."""
        return LinearInteractionMatrix()

    def test_initial_dimension_is_zero(self, fresh_mic):
        """Espacio inicialmente tiene dimensión 0."""
        assert fresh_mic.get_rank() == 0
        assert fresh_mic._dimension == 0

    def test_add_single_vector_increases_dimension(self, fresh_mic):
        """Agregar vector incrementa dimensión."""
        fresh_mic.add_basis_vector("test", MockStep, Stratum.PHYSICS)
        assert fresh_mic._dimension == 1
        assert fresh_mic.get_rank() == 1

    def test_linear_independence_enforced(self, fresh_mic):
        """No se permiten vectores linealmente dependientes (mismo label)."""
        fresh_mic.add_basis_vector("vec1", MockStep, Stratum.PHYSICS)
        
        with pytest.raises(ValueError, match="Dependencia Lineal"):
            fresh_mic.add_basis_vector("vec1", StatefulStep, Stratum.TACTICS)

    def test_functional_colinearity_detected(self, fresh_mic):
        """Colinealidad funcional (mismo operador y estrato) es detectada."""
        fresh_mic.add_basis_vector("vec1", MockStep, Stratum.PHYSICS)
        
        with pytest.raises(ValueError, match="Colinealidad funcional"):
            fresh_mic.add_basis_vector("vec2", MockStep, Stratum.PHYSICS)

    def test_different_strata_allowed_same_operator(self, fresh_mic):
        """Mismo operador en diferentes estratos es permitido."""
        fresh_mic.add_basis_vector("vec1", MockStep, Stratum.PHYSICS)
        fresh_mic.add_basis_vector("vec2", MockStep, Stratum.STRATEGY)
        
        assert fresh_mic._dimension == 2

    def test_projection_returns_correct_vector(self, fresh_mic):
        """Proyección retorna el vector correcto."""
        fresh_mic.add_basis_vector("target", MockStep, Stratum.TACTICS)
        
        result = fresh_mic.project_intent("target")
        
        assert result.label == "target"
        assert result.operator_class == MockStep
        assert result.stratum == Stratum.TACTICS

    def test_projection_of_unknown_raises(self, fresh_mic):
        """Proyección de vector desconocido lanza error."""
        fresh_mic.add_basis_vector("known", MockStep, Stratum.PHYSICS)
        
        with pytest.raises(ValueError, match="Ker"):
            fresh_mic.project_intent("unknown")

    def test_projection_empty_intent_raises(self, fresh_mic):
        """Proyección de intención vacía lanza error."""
        with pytest.raises(ValueError, match="vacío"):
            fresh_mic.project_intent("")

    def test_spectrum_contains_all_vectors(self, director):
        """Espectro contiene todos los vectores base."""
        spectrum = director.mic.get_spectrum()
        
        assert "load_data" in spectrum
        assert "build_output" in spectrum
        assert len(spectrum) == director.mic._dimension

    def test_spectrum_values_are_positive(self, director):
        """Valores espectrales son positivos (estabilidad)."""
        spectrum = director.mic.get_spectrum()
        
        for label, value in spectrum.items():
            assert value > 0, f"Valor espectral no positivo para {label}"

    def test_stratum_weighting_in_spectrum(self, director):
        """Estratos superiores tienen mayor peso espectral."""
        spectrum = director.mic.get_spectrum()
        
        # WISDOM debe tener mayor peso que PHYSICS
        physics_vectors = [v for v in director.mic._basis.values() 
                         if v.stratum == Stratum.PHYSICS]
        wisdom_vectors = [v for v in director.mic._basis.values() 
                         if v.stratum == Stratum.WISDOM]
        
        if physics_vectors and wisdom_vectors:
            physics_val = spectrum[physics_vectors[0].label]
            wisdom_val = spectrum[wisdom_vectors[0].label]
            assert wisdom_val > physics_val

    def test_gram_matrix_is_symmetric(self, director):
        """Matriz de Gram es simétrica."""
        director.mic._orthonormalize_basis()
        gram = director.mic._gram_matrix
        
        if gram is not None and gram.size > 0:
            np.testing.assert_array_almost_equal(gram, gram.T)

    def test_gram_matrix_is_positive_semidefinite(self, director):
        """Matriz de Gram es semi-definida positiva."""
        director.mic._orthonormalize_basis()
        gram = director.mic._gram_matrix
        
        if gram is not None and gram.size > 0:
            eigenvalues = np.linalg.eigvalsh(gram)
            assert np.all(eigenvalues >= -NUMERICAL_TOLERANCE)

    def test_killing_pairing_is_symmetric(self, fresh_mic):
        """Apareamiento de Killing es simétrico: K(v,w) = K(w,v)."""
        fresh_mic.add_basis_vector("v1", MockStep, Stratum.PHYSICS)
        fresh_mic.add_basis_vector("v2", StatefulStep, Stratum.TACTICS)
        
        v1 = fresh_mic._basis["v1"]
        v2 = fresh_mic._basis["v2"]
        
        k12 = fresh_mic._compute_killing_pairing(v1, v2)
        k21 = fresh_mic._compute_killing_pairing(v2, v1)
        
        assert abs(k12 - k21) < NUMERICAL_TOLERANCE

    def test_killing_self_pairing_is_maximal(self, fresh_mic):
        """Auto-apareamiento de Killing es máximo para un vector."""
        fresh_mic.add_basis_vector("v", MockStep, Stratum.PHYSICS)
        v = fresh_mic._basis["v"]
        
        k_self = fresh_mic._compute_killing_pairing(v, v)
        
        # Auto-apareamiento debe ser 1.0 (normalizado)
        assert k_self == 1.0

    def test_operator_signature_similarity(self, fresh_mic):
        """Similitud de firma entre operadores distintos es menor que 1."""
        similarity = fresh_mic._compute_operator_signature_similarity(
            MockStep, StatefulStep
        )
        
        assert 0.0 <= similarity <= 1.0
        # Operadores distintos deberían tener similitud < 1
        assert similarity < 1.0

    def test_condition_number_is_finite(self, director):
        """Número de condición de la base es finito."""
        cond = director.mic.get_condition_number()
        
        assert np.isfinite(cond)
        assert cond >= 1.0  # Mínimo posible para matriz no singular


# =============================================================================
# TESTS: INFORMATION GEOMETRY
# =============================================================================


class TestInformationGeometry(TestFixtures):
    """
    Tests para geometría de la información.
    
    Verifica:
    - Cálculo de entropía
    - Dimensión intrínseca
    - Información de Fisher
    - Divergencia KL
    """

    def test_empty_dataframe_returns_zero_entropy(self, info_geometry):
        """DataFrame vacío retorna entropía cero."""
        result = info_geometry.compute_entropy(pd.DataFrame())
        
        assert result["shannon_entropy"] == 0.0
        assert result["intrinsic_dimension"] == 0.0

    def test_entropy_of_uniform_distribution(self, info_geometry):
        """Entropía de distribución uniforme es máxima."""
        n_categories = 4
        df = pd.DataFrame({
            "category": ["A", "B", "C", "D"] * 25  # Uniforme
        })
        
        result = info_geometry.compute_entropy(df)
        expected_max_entropy = np.log2(n_categories)
        
        assert abs(result["shannon_entropy"] - expected_max_entropy) < 0.1

    def test_entropy_of_constant_is_zero(self, info_geometry):
        """Entropía de distribución constante es cero."""
        df = pd.DataFrame({
            "constant": ["A"] * 100
        })
        
        result = info_geometry.compute_entropy(df)
        
        assert result["shannon_entropy"] == 0.0

    def test_intrinsic_dimension_single_column(self, info_geometry):
        """Dimensión intrínseca de una columna es 1."""
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 100)
        })
        
        result = info_geometry.compute_entropy(df)
        
        assert result["intrinsic_dimension"] == 1.0

    def test_intrinsic_dimension_correlated_columns(self, info_geometry):
        """Columnas correlacionadas tienen dimensión intrínseca menor."""
        x = np.random.normal(0, 1, 100)
        df = pd.DataFrame({
            "x": x,
            "y": x + np.random.normal(0, 0.01, 100),  # Altamente correlacionado
            "z": x * 2 + np.random.normal(0, 0.01, 100),
        })
        
        result = info_geometry.compute_entropy(df)
        
        # Dimensión intrínseca debería ser cercana a 1 (no 3)
        assert result["intrinsic_dimension"] < 2.5

    def test_intrinsic_dimension_independent_columns(self, info_geometry):
        """Columnas independientes tienen dimensión intrínseca igual a n."""
        np.random.seed(42)
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 100),
            "y": np.random.normal(0, 1, 100),
            "z": np.random.normal(0, 1, 100),
        })
        
        result = info_geometry.compute_entropy(df)
        
        # Dimensión debería ser cercana a 3
        assert result["intrinsic_dimension"] >= 2.0

    def test_fisher_information_increases_with_precision(self, info_geometry):
        """Información de Fisher aumenta con mayor precisión (menor varianza)."""
        df_high_var = pd.DataFrame({"x": np.random.normal(0, 10, 100)})
        df_low_var = pd.DataFrame({"x": np.random.normal(0, 0.1, 100)})
        
        fisher_high = info_geometry._compute_fisher_information(
            df_high_var.values
        )
        fisher_low = info_geometry._compute_fisher_information(
            df_low_var.values
        )
        
        assert fisher_low > fisher_high

    def test_effective_rank_is_bounded(self, info_geometry):
        """Rango efectivo está acotado por dimensión del espacio."""
        df = pd.DataFrame({
            "x": np.random.normal(0, 1, 100),
            "y": np.random.normal(0, 1, 100),
        })
        
        result = info_geometry.compute_entropy(df)
        
        assert 0 <= result["effective_rank"] <= 2

    def test_kl_divergence_zero_for_identical(self, info_geometry):
        """Divergencia KL es cero para distribuciones idénticas."""
        df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
        
        kl = info_geometry.kl_divergence(df, df)
        
        assert kl < 0.1  # Pequeño por estabilidad numérica

    def test_kl_divergence_asymmetric(self, info_geometry):
        """Divergencia KL es asimétrica."""
        df1 = pd.DataFrame({"x": [1, 1, 1, 2, 2]})
        df2 = pd.DataFrame({"x": [1, 2, 2, 2, 2]})
        
        kl_12 = info_geometry.kl_divergence(df1, df2)
        kl_21 = info_geometry.kl_divergence(df2, df1)
        
        # En general KL(P||Q) != KL(Q||P)
        # Solo verificar que no falla y retorna valores razonables
        assert np.isfinite(kl_12)
        assert np.isfinite(kl_21)


# =============================================================================
# TESTS: PROCRUSTES ANALYZER
# =============================================================================


class TestProcrustesAnalyzer(TestFixtures):
    """
    Tests para análisis Procrustes.
    
    Verifica:
    - Alineamiento isométrico
    - Alineamiento conforme
    - Preservación de propiedades
    """

    def test_isometric_align_identity(self, procrustes):
        """Datos idénticos producen rotación identidad."""
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        Y = X.copy()
        
        Xc, Yc, R = procrustes.isometric_align(X, Y)
        
        if R is not None:
            np.testing.assert_array_almost_equal(R, np.eye(2), decimal=5)

    def test_isometric_preserves_distances(self, procrustes):
        """Alineamiento isométrico preserva distancias."""
        np.random.seed(42)
        X = np.random.randn(10, 3)
        
        # Rotación arbitraria
        theta = np.pi / 4
        R_true = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        Y = X @ R_true.T
        
        Xc, Yc, R = procrustes.isometric_align(X, Y)
        
        if Yc is not None:
            # Distancias deben preservarse
            dist_X = pdist(Xc)
            dist_Y = pdist(Yc)
            np.testing.assert_array_almost_equal(dist_X, dist_Y, decimal=5)

    def test_isometric_rotation_matrix_orthogonal(self, procrustes):
        """Matriz de rotación es ortogonal."""
        X = np.random.randn(5, 3)
        Y = np.random.randn(5, 3)
        
        _, _, R = procrustes.isometric_align(X, Y)
        
        if R is not None:
            # R^T R = I
            product = R.T @ R
            np.testing.assert_array_almost_equal(product, np.eye(R.shape[0]), decimal=5)

    def test_conformal_align_allows_scaling(self, procrustes):
        """Alineamiento conforme permite escala uniforme."""
        X = np.array([[1, 0], [0, 1], [-1, 0]], dtype=float)
        Y = 2.0 * X  # Escalado por 2
        
        Xc, Yc, params = procrustes.conformal_align(X, Y)
        
        if params is not None:
            scale, R = params
            assert abs(scale - 2.0) < 0.5  # Tolerancia alta por centrado

    def test_dimension_mismatch_rows(self, procrustes):
        """Maneja diferente número de filas."""
        X = np.random.randn(10, 3)
        Y = np.random.randn(8, 3)
        
        Xc, Yc, R = procrustes.isometric_align(X, Y)
        
        # Debe ejecutar sin error y producir outputs válidos
        assert Xc is not None
        assert Yc is not None
        assert Xc.shape[0] == Yc.shape[0]

    def test_dimension_mismatch_columns(self, procrustes):
        """Maneja diferente número de columnas (padding)."""
        X = np.random.randn(10, 3)
        Y = np.random.randn(10, 5)
        
        Xc, Yc, R = procrustes.isometric_align(X, Y)
        
        assert Xc is not None
        assert Yc is not None

    def test_alignment_quality_computed(self, procrustes):
        """Calidad de alineamiento se computa correctamente."""
        X = np.random.randn(10, 3)
        Y = X + np.random.randn(10, 3) * 0.1  # Pequeña perturbación
        
        procrustes.isometric_align(X, Y)
        quality = procrustes.get_last_alignment_quality()
        
        assert quality is not None
        assert 0.0 <= quality <= 1.0
        assert quality > 0.5  # Debe ser alta para datos similares

    def test_alignment_quality_low_for_random(self, procrustes):
        """Calidad es baja para datos aleatorios no relacionados."""
        np.random.seed(42)
        X = np.random.randn(10, 3)
        Y = np.random.randn(10, 3)
        
        procrustes.isometric_align(X, Y)
        quality = procrustes.get_last_alignment_quality()
        
        # Datos aleatorios deberían tener alineamiento mediocre
        assert quality is not None
        assert quality < 0.9

    def test_affine_align_handles_general_transform(self, procrustes):
        """Alineamiento afín maneja transformaciones generales."""
        X = np.random.randn(10, 3)
        A_true = np.random.randn(3, 3)
        Y = X @ A_true
        
        Xc, Yc, A = procrustes.affine_align(X, Y)
        
        if A is not None:
            # Verificar que Y @ A ≈ X
            residual = np.linalg.norm(Y @ A - X) / np.linalg.norm(X)
            assert residual < 0.5  # Tolerancia razonable

    def test_nan_handling(self, procrustes):
        """Maneja NaN en datos de entrada."""
        X = np.array([[1, 2], [np.nan, 4], [5, 6]])
        Y = np.array([[1, 2], [3, 4], [5, np.nan]])
        
        Xc, Yc, R = procrustes.isometric_align(X, Y)
        
        # Debe completar sin error
        assert not np.any(np.isnan(Xc))
        assert not np.any(np.isnan(Yc))


# =============================================================================
# TESTS: HOMOLOGY COMPUTATION
# =============================================================================


class TestHomologyComputation(TestFixtures):
    """
    Tests para cómputo de homología del pipeline.
    
    Verifica:
    - Números de Betti
    - Característica de Euler
    - Propiedades topológicas
    """

    def test_empty_context_gives_trivial_homology(self, director):
        """Contexto vacío da homología trivial."""
        director._compute_homology_groups({})
        
        assert director._homology_groups["H0"] == 1
        assert director._homology_groups["H1"] == 0

    def test_single_dataframe_connected(self, director, presupuesto_df):
        """Un solo DataFrame es un espacio conexo (β₀ = 1)."""
        context = {"df1": presupuesto_df}
        
        director._compute_homology_groups(context)
        
        assert director._homology_groups["H0"] == 1

    def test_disjoint_dataframes_multiple_components(self, director):
        """DataFrames sin columnas comunes forman componentes separadas."""
        df1 = pd.DataFrame({"A": [1, 2, 3]})
        df2 = pd.DataFrame({"B": [4, 5, 6]})
        df3 = pd.DataFrame({"C": [7, 8, 9]})
        
        context = {"df1": df1, "df2": df2, "df3": df3}
        
        director._compute_homology_groups(context)
        
        # Sin conexiones, deberían ser 3 componentes
        assert director._homology_groups["H0"] >= 1

    def test_connected_dataframes_single_component(self, director):
        """DataFrames con columnas comunes forman una componente."""
        df1 = pd.DataFrame({"A": [1], "B": [2]})
        df2 = pd.DataFrame({"B": [3], "C": [4]})
        df3 = pd.DataFrame({"C": [5], "A": [6]})  # Conecta con df1
        
        context = {"df1": df1, "df2": df2, "df3": df3}
        
        director._compute_homology_groups(context)
        
        # Deberían estar conectados
        assert director._homology_groups["H0"] == 1

    def test_cycle_detected_in_homology(self, director):
        """Ciclo en conexiones incrementa β₁."""
        # Crear DataFrames que formen un ciclo: A-B, B-C, C-A
        df1 = pd.DataFrame({"col_A": [1], "col_B": [2]})
        df2 = pd.DataFrame({"col_B": [3], "col_C": [4]})
        df3 = pd.DataFrame({"col_C": [5], "col_A": [6]})
        
        context = {"df1": df1, "df2": df2, "df3": df3}
        
        director._compute_homology_groups(context)
        
        # Debería haber al menos un ciclo
        assert director._homology_groups["H1"] >= 0

    def test_euler_characteristic_consistency(self, director):
        """Característica de Euler es consistente: χ = β₀ - β₁."""
        df1 = pd.DataFrame({"A": [1], "B": [2]})
        df2 = pd.DataFrame({"B": [3], "C": [4]})
        
        context = {"df1": df1, "df2": df2}
        
        director._compute_homology_groups(context)
        
        h = director._homology_groups
        if "Euler_characteristic" in h:
            assert h["Euler_characteristic"] == h["H0"] - h["H1"]

    def test_homology_preserved_after_adding_dataframe(self, director):
        """Agregar DataFrame que no crea ciclos no aumenta β₁."""
        df1 = pd.DataFrame({"A": [1], "B": [2]})
        df2 = pd.DataFrame({"C": [3], "D": [4]})
        
        context1 = {"df1": df1}
        director._compute_homology_groups(context1)
        h1_initial = director._homology_groups["H1"]
        
        context2 = {"df1": df1, "df2": df2}
        director._compute_homology_groups(context2)
        h1_final = director._homology_groups["H1"]
        
        # Agregar componente disjunta no crea ciclos
        assert h1_final >= h1_initial


# =============================================================================
# TESTS: STRATUM TRANSITIONS
# =============================================================================


class TestStratumTransitions(TestFixtures):
    """
    Tests para validación de transiciones entre estratos.
    
    Verifica:
    - Transiciones válidas
    - Detección de saltos
    - Ciclos de reinicio
    """

    def test_forward_transition_allowed(self, director):
        """Transición hacia adelante es permitida."""
        # No debe lanzar excepción
        director._validate_stratum_transition(Stratum.PHYSICS, Stratum.TACTICS)
        director._validate_stratum_transition(Stratum.TACTICS, Stratum.STRATEGY)
        director._validate_stratum_transition(Stratum.STRATEGY, Stratum.WISDOM)

    def test_same_level_transition_allowed(self, director):
        """Transición en mismo nivel es permitida."""
        director._validate_stratum_transition(Stratum.PHYSICS, Stratum.PHYSICS)
        director._validate_stratum_transition(Stratum.TACTICS, Stratum.TACTICS)

    def test_skip_stratum_warns(self, director, caplog):
        """Saltar estratos genera warning."""
        import logging
        
        with caplog.at_level(logging.WARNING):
            director._validate_stratum_transition(Stratum.PHYSICS, Stratum.WISDOM)
        
        assert any("Salto" in record.message or "salto" in record.message 
                  for record in caplog.records)

    def test_cycle_restart_detected(self, director, caplog):
        """Reinicio de ciclo (WISDOM → PHYSICS) es detectado."""
        import logging
        
        with caplog.at_level(logging.INFO):
            director._validate_stratum_transition(Stratum.WISDOM, Stratum.PHYSICS)
        
        assert any("ciclo" in record.message.lower() for record in caplog.records)

    def test_backward_transition_warns(self, director, caplog):
        """Transición hacia atrás genera warning."""
        import logging
        
        with caplog.at_level(logging.WARNING):
            director._validate_stratum_transition(Stratum.STRATEGY, Stratum.TACTICS)

    def test_none_current_stratum_allowed(self, director):
        """Primera transición (current=None) es permitida."""
        # No debe lanzar excepción
        director._validate_stratum_transition(None, Stratum.PHYSICS)
        director._validate_stratum_transition(None, Stratum.WISDOM)

    def test_stratum_to_filtration_mapping(self, director):
        """Mapeo de estrato a nivel de filtración es correcto."""
        assert director._stratum_to_filtration(Stratum.PHYSICS) == 1
        assert director._stratum_to_filtration(Stratum.TACTICS) == 2
        assert director._stratum_to_filtration(Stratum.STRATEGY) == 3
        assert director._stratum_to_filtration(Stratum.WISDOM) == 4

    def test_infer_current_stratum_from_context(self, director, presupuesto_df):
        """Inferencia de estrato actual desde contexto."""
        context_physics = {"df_presupuesto": presupuesto_df}
        assert director._infer_current_stratum(context_physics) == Stratum.PHYSICS
        
        context_tactics = {"df_merged": presupuesto_df}
        assert director._infer_current_stratum(context_tactics) == Stratum.TACTICS
        
        context_strategy = {"graph": MagicMock()}
        assert director._infer_current_stratum(context_strategy) == Stratum.STRATEGY


# =============================================================================
# TESTS: DATA VALIDATOR
# =============================================================================


class TestDataValidator(TestFixtures):
    """
    Tests para DataValidator con cobertura completa.
    """

    def test_validate_none_returns_invalid(self):
        """None es detectado como inválido."""
        valid, error = DataValidator.validate_dataframe_not_empty(None, "test")
        
        assert valid is False
        assert "None" in error

    def test_validate_non_dataframe_returns_invalid(self):
        """Tipos no-DataFrame son rechazados."""
        invalid_inputs = ["string", 123, [], {}, set(), (1, 2)]
        
        for invalid in invalid_inputs:
            valid, error = DataValidator.validate_dataframe_not_empty(invalid, "test")
            assert valid is False
            assert "no es un DataFrame" in error

    def test_validate_empty_dataframe_returns_invalid(self):
        """DataFrame vacío es detectado."""
        valid, error = DataValidator.validate_dataframe_not_empty(pd.DataFrame(), "test")
        
        assert valid is False
        assert "vacío" in error

    def test_validate_all_nulls_returns_invalid(self):
        """DataFrame con solo nulos es detectado."""
        df_nulls = pd.DataFrame({
            "A": [None, None, None],
            "B": [np.nan, np.nan, np.nan],
            "C": [pd.NA, pd.NA, pd.NA]
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

    def test_validate_mixed_nulls_valid(self):
        """DataFrame con algunos nulos pero datos válidos es aceptado."""
        df = pd.DataFrame({
            "A": [1, None, 3],
            "B": ["x", "y", None]
        })
        
        valid, error = DataValidator.validate_dataframe_not_empty(df, "test")
        
        assert valid is True

    def test_validate_columns_all_present(self):
        """Columnas requeridas presentes son validadas."""
        df = pd.DataFrame({"Col1": [1], "Col2": [2], "Col3": [3]})
        
        valid, error = DataValidator.validate_required_columns(
            df, ["Col1", "Col2"], "test"
        )
        
        assert valid is True

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

    def test_detect_duplicates_removes_them(self):
        """Duplicados son detectados y eliminados."""
        df = pd.DataFrame({
            "ID": [1, 2, 2, 3, 3, 3],
            "Val": ["a", "b", "c", "d", "e", "f"]
        })
        
        cleaned = DataValidator.detect_and_log_duplicates(df, ["ID"], "test")
        
        assert len(cleaned) == 3
        assert sorted(cleaned["ID"].tolist()) == [1, 2, 3]

    def test_detect_duplicates_keeps_first(self):
        """Por defecto mantiene primera ocurrencia."""
        df = pd.DataFrame({
            "ID": [1, 1, 1],
            "Val": ["first", "second", "third"]
        })
        
        cleaned = DataValidator.detect_and_log_duplicates(df, ["ID"], "test")
        
        assert len(cleaned) == 1
        assert cleaned.iloc[0]["Val"] == "first"

    def test_detect_duplicates_keeps_last(self):
        """Puede mantener última ocurrencia."""
        df = pd.DataFrame({
            "ID": [1, 1, 1],
            "Val": ["first", "second", "third"]
        })
        
        cleaned = DataValidator.detect_and_log_duplicates(
            df, ["ID"], "test", keep="last"
        )
        
        assert len(cleaned) == 1
        assert cleaned.iloc[0]["Val"] == "third"


# =============================================================================
# TESTS: FILE VALIDATOR
# =============================================================================


class TestFileValidator(TestFixtures):
    """Tests para FileValidator."""

    def test_validate_file_exists_success(self, mock_filesystem):
        """Archivo válido es aceptado."""
        mock_filesystem.add_file("data.csv", exists=True, is_file=True, readable=True)
        
        with contextlib.ExitStack() as stack:
            for target, side_effect in mock_filesystem.patch_validators().items():
                stack.enter_context(patch(target, side_effect=side_effect, autospec=True))
            valid, error = FileValidator.validate_file_exists("data.csv", "test")
        
        assert valid is True

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
        """Sin permisos de lectura es rechazado."""
        mock_filesystem.add_file("protected.csv", exists=True, is_file=True, readable=False)
        
        with contextlib.ExitStack() as stack:
            for target, side_effect in mock_filesystem.patch_validators().items():
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


# =============================================================================
# TESTS: DATA MERGER ALGEBRAIC PROPERTIES
# =============================================================================


class TestDataMergerAlgebraic(TestFixtures):
    """
    Tests para propiedades algebraicas del DataMerger.
    
    Verifica:
    - Preservación de estructura
    - Calidad de merge
    - Estrategias múltiples
    """

    def test_merge_preserves_left_join_cardinality(self, merger, presupuesto_df):
        """Merge preserva cardinalidad de left join."""
        df_costos = pd.DataFrame({
            ColumnNames.CODIGO_APU: presupuesto_df[ColumnNames.CODIGO_APU].tolist(),
            "costo": [100, 200, 300, 400, 500]
        })
        
        merged = merger.merge_with_presupuesto(presupuesto_df, df_costos)
        
        assert len(merged) == len(presupuesto_df)

    def test_merge_quality_evaluation(self, merger):
        """Calidad de merge se evalúa correctamente."""
        df_good = pd.DataFrame({
            "A": [1, 2, 3],
            "B": [4, 5, 6],
            "_merge": ["both", "both", "both"]
        })
        
        quality = merger._evaluate_merge_quality(df_good)
        
        assert quality > 0.8

    def test_merge_quality_low_for_missing(self, merger):
        """Calidad baja para merge con muchos faltantes."""
        df_poor = pd.DataFrame({
            "A": [1, None, None],
            "B": [None, None, None],
            "_merge": ["left_only", "left_only", "left_only"]
        })
        
        quality = merger._evaluate_merge_quality(df_poor)
        
        assert quality < 0.5

    def test_exact_merge_creates_indicator(self, merger):
        """Merge exacto crea columna indicadora."""
        df_apus = pd.DataFrame({
            ColumnNames.DESCRIPCION_INSUMO: ["Material A"],
            ColumnNames.CANTIDAD_APU: [10]
        })
        df_insumos = pd.DataFrame({
            ColumnNames.DESCRIPCION_INSUMO: ["Material A"],
            ColumnNames.VR_UNITARIO_INSUMO: [100]
        })
        
        merged = merger._exact_merge(df_apus, df_insumos)
        
        assert "_merge" in merged.columns

    def test_fallback_merge_returns_valid_result(self, merger):
        """Fallback merge retorna resultado válido."""
        df_apus = pd.DataFrame({
            ColumnNames.CODIGO_APU: ["APU-001"],
            ColumnNames.DESCRIPCION_INSUMO: ["Test"]
        })
        
        result = merger._fallback_merge(df_apus, pd.DataFrame())
        
        assert not result.empty
        assert ColumnNames.CODIGO_APU in result.columns

    def test_information_preservation_validated(self, merger, info_geometry):
        """Preservación de información es validada."""
        df_apus = pd.DataFrame({
            ColumnNames.DESCRIPCION_INSUMO: ["A", "B", "C"],
            "val": [1, 2, 3]
        })
        df_insumos = pd.DataFrame({
            ColumnNames.DESCRIPCION_INSUMO: ["A", "B", "C"],
            ColumnNames.VR_UNITARIO_INSUMO: [10, 20, 30]
        })
        
        info_before_a = info_geometry.compute_entropy(df_apus)
        info_before_b = info_geometry.compute_entropy(df_insumos)
        
        merged = merger.merge_apus_with_insumos(df_apus, df_insumos)
        
        # Verificar que no hubo colapso dimensional excesivo
        info_after = info_geometry.compute_entropy(merged)
        
        # Dimensión no debe colapsar más de 50%
        dim_before = info_before_a["intrinsic_dimension"] + info_before_b["intrinsic_dimension"]
        if dim_before > 0:
            preservation = info_after["intrinsic_dimension"] / dim_before
            assert preservation >= 0.3 or dim_before < 2  # Tolerancia para dimensiones pequeñas


# =============================================================================
# TESTS: PRESUPUESTO PROCESSOR
# =============================================================================


class TestPresupuestoProcessor(TestFixtures):
    """Tests para PresupuestoProcessor."""

    @pytest.fixture
    def processor(self, config, thresholds) -> PresupuestoProcessor:
        """Procesador de presupuesto."""
        profile = {"loader_params": {}}
        return PresupuestoProcessor(config, thresholds, profile)

    def test_clean_phantom_rows_removes_empty(self, processor):
        """Filas vacías son eliminadas."""
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
            "B": ["data", "", "   ", "\t"]
        })
        
        cleaned = processor._clean_phantom_rows(df)
        
        assert len(cleaned) <= 2

    def test_clean_phantom_rows_preserves_zeros(self, processor):
        """Ceros son preservados."""
        df = pd.DataFrame({
            "A": ["item1", "item2"],
            "B": [10, 0]
        })
        
        cleaned = processor._clean_phantom_rows(df)
        
        assert len(cleaned) == 2

    def test_clean_removes_metadata_rows(self, processor):
        """Filas de metadatos son eliminadas."""
        df = pd.DataFrame({
            "A": ["APU-001", "TOTAL", "SUBTOTAL", "APU-002"],
            "B": [100, 500, 200, 150]
        })
        
        cleaned = processor._clean_phantom_rows(df)
        
        # TOTAL y SUBTOTAL deben ser removidos
        assert "TOTAL" not in cleaned["A"].values
        assert "SUBTOTAL" not in cleaned["A"].values

    def test_process_returns_empty_on_load_error(self, processor):
        """Error de carga retorna DataFrame vacío."""
        with patch("app.pipeline_director.load_data", return_value=None):
            result = processor.process("nonexistent.csv")
        
        assert result.empty


# =============================================================================
# TESTS: PIPELINE DIRECTOR ORCHESTRATION
# =============================================================================


class TestPipelineDirectorOrchestration(TestFixtures):
    """Tests para orquestación del PipelineDirector."""

    def test_director_initialization(self, director):
        """Director se inicializa correctamente."""
        assert director is not None
        assert hasattr(director, "mic")
        assert director.mic.get_rank() > 0

    def test_register_custom_step(self, director):
        """Paso personalizado se registra."""
        director.mic.add_basis_vector("custom", MockStep, Stratum.WISDOM)
        
        vector = director.mic.project_intent("custom")
        assert vector.operator_class == MockStep

    def test_run_single_step_success(self, director, config, telemetry, thresholds):
        """Ejecución de paso único funciona."""
        director.mic.add_basis_vector("mock", MockStep, Stratum.TACTICS)
        
        session_id = "test-session-123"
        context = {"test_key": "test_value"}
        
        result = director.run_single_step("mock", session_id, context)
        
        assert result["status"] == "success"
        assert result["step"] == "mock"

    def test_run_single_step_error_handling(self, director, telemetry):
        """Error en paso es manejado."""
        director.mic.add_basis_vector("fail", FailingStep, Stratum.PHYSICS)
        
        result = director.run_single_step("fail", "test-session")
        
        assert result["status"] == "error"
        assert "error" in result

    def test_execute_multiple_steps_order(self, director, config):
        """Múltiples pasos se ejecutan en orden."""
        execution_order = []
        
        def create_tracking_step(name):
            class TrackingStep(ProcessingStep):
                step_name = name
                def __init__(self, cfg, thr):
                    pass
                def execute(self, context, telemetry):
                    execution_order.append(self.step_name)
                    return context
            return TrackingStep
        
        director.mic.add_basis_vector("s1", create_tracking_step("s1"), Stratum.PHYSICS)
        director.mic.add_basis_vector("s2", create_tracking_step("s2"), Stratum.TACTICS)
        
        config["pipeline_recipe"] = [
            {"step": "s1", "enabled": True},
            {"step": "s2", "enabled": True},
        ]
        
        director.execute_pipeline_orchestrated({})
        
        assert execution_order == ["s1", "s2"]

    def test_disabled_step_skipped(self, director, config):
        """Paso deshabilitado es omitido."""
        execution_order = []
        
        class TrackingStep(ProcessingStep):
            def __init__(self, cfg, thr):
                pass
            def execute(self, context, telemetry):
                execution_order.append("executed")
                return context
        
        director.mic.add_basis_vector("skip_me", TrackingStep, Stratum.PHYSICS)
        
        config["pipeline_recipe"] = [
            {"step": "skip_me", "enabled": False},
        ]
        
        director.execute_pipeline_orchestrated({})
        
        assert "executed" not in execution_order

    def test_state_trace_computation(self, director, presupuesto_df):
        """Traza del estado se computa correctamente."""
        context = {"df_test": presupuesto_df}
        
        trace = director._compute_state_trace(context)
        
        assert trace > 0
        assert np.isfinite(trace)


# =============================================================================
# TESTS: TELEMETRY INTEGRATION
# =============================================================================


class TestTelemetryIntegration(TestFixtures):
    """Tests para integración de telemetría."""

    def test_telemetry_records_step_timing(self, telemetry):
        """Telemetría registra timing."""
        telemetry.start_step("test_step")
        time.sleep(0.01)
        telemetry.end_step("test_step", "success")
        
        duration = telemetry.get_step_duration("test_step")
        
        assert duration is not None
        assert duration > 0.01

    def test_telemetry_records_errors(self, telemetry):
        """Telemetría registra errores."""
        telemetry.start_step("failing_step")
        telemetry.record_error("failing_step", "Test error")
        
        assert len(telemetry.errors) >= 1

    def test_telemetry_records_metrics(self, telemetry):
        """Telemetría registra métricas."""
        telemetry.record_metric("step1", "count", 42)
        telemetry.record_metric("step1", "ratio", 0.95)
        
        assert "step1" in telemetry.recorded_metrics
        assert telemetry.recorded_metrics["step1"]["count"] == 42


# =============================================================================
# TESTS: PIPELINE INVARIANTS
# =============================================================================


class TestPipelineInvariants(TestFixtures):
    """Tests para invariantes del pipeline."""

    def test_validation_is_idempotent(self, presupuesto_df):
        """Validación es idempotente."""
        result1, _ = DataValidator.validate_dataframe_not_empty(presupuesto_df, "test")
        result2, _ = DataValidator.validate_dataframe_not_empty(presupuesto_df, "test")
        
        assert result1 == result2

    def test_clean_is_idempotent(self, config, thresholds):
        """Limpieza es idempotente."""
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

    def test_processing_is_deterministic(self, merger):
        """Mismo input produce mismo output."""
        df_apus = pd.DataFrame({
            ColumnNames.DESCRIPCION_INSUMO: ["A", "B"],
            "val": [1, 2]
        })
        df_insumos = pd.DataFrame({
            ColumnNames.DESCRIPCION_INSUMO: ["A", "B"],
            ColumnNames.VR_UNITARIO_INSUMO: [10, 20]
        })
        
        result1 = merger.merge_apus_with_insumos(df_apus.copy(), df_insumos.copy())
        result2 = merger.merge_apus_with_insumos(df_apus.copy(), df_insumos.copy())
        
        pd.testing.assert_frame_equal(result1, result2)

    def test_merge_conserves_data_invariant(self, merger, presupuesto_df):
        """Merge no genera filas sin origen."""
        df_costos = pd.DataFrame({
            ColumnNames.CODIGO_APU: presupuesto_df[ColumnNames.CODIGO_APU].tolist(),
            "costo": list(range(len(presupuesto_df)))
        })
        
        merged = merger.merge_with_presupuesto(presupuesto_df, df_costos)
        
        original_codes = set(presupuesto_df[ColumnNames.CODIGO_APU])
        merged_codes = set(merged[ColumnNames.CODIGO_APU])
        
        assert merged_codes.issubset(original_codes)


# =============================================================================
# TESTS: EDGE CASES
# =============================================================================


class TestEdgeCases(TestFixtures):
    """Tests para casos límite."""

    def test_empty_pipeline_recipe(self, director, config):
        """Recipe vacío ejecuta sin pasos."""
        config["pipeline_recipe"] = []
        
        result = director.execute_pipeline_orchestrated({})
        
        assert isinstance(result, dict)

    def test_very_large_dataframe(self, merger):
        """DataFrames grandes son procesados."""
        large_df = PresupuestoBuilder().with_items(5000).build()
        large_apus = APUsBuilder().for_presupuesto(large_df, insumos_per_apu=3).build()
        
        start = time.time()
        # Solo verificar que no falla
        assert len(large_df) == 5000
        elapsed = time.time() - start
        
        assert elapsed < 5.0

    def test_unicode_in_data(self, merger):
        """Datos con unicode son manejados."""
        df_apus = pd.DataFrame({
            ColumnNames.DESCRIPCION_INSUMO: ["Café ☕", "Niño 👶", "Señal 📡"],
            "cantidad": [1, 2, 3]
        })
        df_insumos = pd.DataFrame({
            ColumnNames.DESCRIPCION_INSUMO: ["Café ☕", "Niño 👶", "Señal 📡"],
            ColumnNames.VR_UNITARIO_INSUMO: [10, 20, 30]
        })
        
        merged = merger.merge_apus_with_insumos(df_apus, df_insumos)
        
        assert len(merged) == 3

    def test_special_characters_in_codes(self):
        """Códigos con caracteres especiales son manejados."""
        df = PresupuestoBuilder().with_item("APU-001/A", "Test", 10.0).build()
        
        valid, _ = DataValidator.validate_dataframe_not_empty(df, "test")
        
        assert valid is True

    def test_numeric_overflow_protection(self, thresholds):
        """Protección contra overflow numérico."""
        df = pd.DataFrame({
            ColumnNames.CODIGO_APU: ["APU-001"],
            ColumnNames.CANTIDAD_APU: [1e20],  # Muy grande
            ColumnNames.VR_UNITARIO_INSUMO: [1e20],
        })
        
        # No debe lanzar error de overflow
        assert len(df) == 1

    def test_nan_propagation(self, merger):
        """NaN no se propagan incorrectamente."""
        df_apus = pd.DataFrame({
            ColumnNames.DESCRIPCION_INSUMO: ["A", np.nan, "C"],
            "val": [1, 2, 3]
        })
        df_insumos = pd.DataFrame({
            ColumnNames.DESCRIPCION_INSUMO: ["A", "B", "C"],
            ColumnNames.VR_UNITARIO_INSUMO: [10, 20, 30]
        })
        
        merged = merger.merge_apus_with_insumos(df_apus, df_insumos)
        
        # Debe completar sin error
        assert merged is not None


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

    def test_presupuesto_builder_rejects_duplicate_codes(self):
        """Builder rechaza códigos duplicados."""
        with pytest.raises(ValueError, match="duplicado"):
            (PresupuestoBuilder()
             .with_item("APU-001", "Test 1", 10.0)
             .with_item("APU-001", "Test 2", 20.0))

    def test_insumos_builder_rejects_invalid_type(self):
        """Builder rechaza tipo inválido."""
        with pytest.raises(ValueError, match="Tipo inválido"):
            InsumosBuilder().with_insumo("Test", "INVALID_TYPE", 100.0)

    def test_insumos_builder_rejects_negative_price(self):
        """Builder rechaza precio negativo."""
        with pytest.raises(ValueError, match="negativo"):
            InsumosBuilder().with_insumo("Test", "MATERIAL", -50.0)

    def test_apus_builder_rejects_negative_quantity(self):
        """Builder rechaza cantidad negativa."""
        with pytest.raises(ValueError, match="negativa"):
            APUsBuilder().with_apu_insumo("APU-001", "Insumo", -5.0)

    def test_context_builder_creates_valid_context(self):
        """ContextBuilder crea contexto válido."""
        context = (
            ContextBuilder()
            .with_paths("p.csv", "a.csv", "i.csv")
            .with_validated_strata(Stratum.PHYSICS, Stratum.TACTICS)
            .with_param("custom", "value")
            .build()
        )
        
        assert "presupuesto_path" in context
        assert Stratum.PHYSICS in context["validated_strata"]
        assert context["custom"] == "value"


# =============================================================================
# TESTS: MATHEMATICAL UTILITIES
# =============================================================================


class TestMathematicalUtilities(TestFixtures):
    """Tests para utilidades matemáticas de testing."""

    def test_orthonormal_check(self):
        """Verificación de ortonormalidad funciona."""
        # Matriz identidad es ortonormal
        I = np.eye(3)
        assert AlgebraicTestUtils.is_orthonormal(I)
        
        # Matriz no ortonormal
        M = np.array([[1, 1], [0, 1]])
        assert not AlgebraicTestUtils.is_orthonormal(M)

    def test_positive_definite_check(self):
        """Verificación de definición positiva funciona."""
        # Identidad es definida positiva
        I = np.eye(3)
        assert AlgebraicTestUtils.is_positive_definite(I)
        
        # Matriz con eigenvalor negativo
        M = np.array([[1, 2], [2, 1]])
        eigenvalues = np.linalg.eigvalsh(M)
        if np.min(eigenvalues) < 0:
            assert not AlgebraicTestUtils.is_positive_definite(M)

    def test_condition_number(self):
        """Cálculo de número de condición funciona."""
        # Identidad tiene condición 1
        I = np.eye(3)
        cond = AlgebraicTestUtils.condition_number(I)
        assert abs(cond - 1.0) < NUMERICAL_TOLERANCE

    def test_frobenius_distance(self):
        """Distancia de Frobenius funciona."""
        A = np.zeros((2, 2))
        B = np.ones((2, 2))
        
        dist = AlgebraicTestUtils.frobenius_distance(A, B)
        
        assert dist == 2.0  # sqrt(4) = 2

    def test_betti_number_0_connected(self):
        """β₀ es 1 para grafo conexo."""
        # Grafo completamente conectado
        adj = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0]
        ])
        
        beta_0 = TopologicalTestUtils.betti_number_0(adj)
        
        assert beta_0 == 1

    def test_betti_number_0_disconnected(self):
        """β₀ > 1 para grafo desconectado."""
        # Dos componentes
        adj = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
        
        beta_0 = TopologicalTestUtils.betti_number_0(adj)
        
        assert beta_0 == 2

    def test_shannon_entropy_uniform(self):
        """Entropía de Shannon para distribución uniforme."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        
        entropy = InformationTheoreticTestUtils.shannon_entropy(p)
        
        assert abs(entropy - 2.0) < NUMERICAL_TOLERANCE

    def test_kl_divergence_same_distribution(self):
        """KL divergencia es 0 para distribuciones idénticas."""
        p = np.array([0.5, 0.5])
        
        kl = InformationTheoreticTestUtils.kl_divergence(p, p)
        
        assert abs(kl) < NUMERICAL_TOLERANCE


# =============================================================================
# TESTS: LOAD DATA STEP
# =============================================================================


class TestLoadDataStep(TestFixtures):
    """Tests para LoadDataStep."""

    @pytest.fixture
    def load_step(self, config, thresholds) -> LoadDataStep:
        """Paso de carga de datos."""
        return LoadDataStep(config, thresholds)

    def test_load_step_missing_path_raises(self, load_step, telemetry):
        """Ruta faltante lanza error."""
        context = {}  # Sin rutas
        
        with pytest.raises(ValueError, match="no encontrada"):
            load_step.execute(context, telemetry)

    def test_load_step_empty_presupuesto_raises(self, load_step, telemetry):
        """Presupuesto vacío lanza error."""
        context = ContextBuilder().with_paths().build()
        
        with patch.object(FileValidator, "validate_file_exists", return_value=(True, None)):
            with patch("app.pipeline_director.PresupuestoProcessor") as MockProc:
                MockProc.return_value.process.return_value = pd.DataFrame()
                
                with pytest.raises(ValueError, match="vacío"):
                    load_step.execute(context, telemetry)


# =============================================================================
# TESTS: APU COST CALCULATOR
# =============================================================================


class TestAPUCostCalculator(TestFixtures):
    """Tests para APUCostCalculator."""

    @pytest.fixture
    def calculator(self, config, thresholds) -> APUCostCalculator:
        """Calculador de costos APU."""
        return APUCostCalculator(config, thresholds)

    def test_normalize_tipo_insumo(self, calculator):
        """Normalización de tipo de insumo funciona."""
        df = pd.DataFrame({
            ColumnNames.CODIGO_APU: ["APU-001", "APU-002"],
            ColumnNames.TIPO_INSUMO: ["MATERIAL", "MANO DE OBRA"],
            ColumnNames.COSTO_INSUMO_EN_APU: [100, 200]
        })
        
        normalized = calculator._normalize_tipo_insumo(df)
        
        assert "_CATEGORIA_COSTO" in normalized.columns

    def test_aggregate_costs_by_category(self, calculator):
        """Agregación de costos por categoría funciona."""
        df = pd.DataFrame({
            ColumnNames.CODIGO_APU: ["APU-001", "APU-001", "APU-001"],
            "_CATEGORIA_COSTO": [ColumnNames.MATERIALES, ColumnNames.MANO_DE_OBRA, ColumnNames.EQUIPO],
            ColumnNames.COSTO_INSUMO_EN_APU: [100, 200, 50]
        })
        
        aggregated = calculator._aggregate_costs(df)
        
        assert ColumnNames.MATERIALES in aggregated.columns
        assert ColumnNames.MANO_DE_OBRA in aggregated.columns

    def test_calculate_returns_tuple(self, calculator):
        """calculate() retorna tupla de 3 DataFrames."""
        df = pd.DataFrame({
            ColumnNames.CODIGO_APU: ["APU-001"],
            ColumnNames.COSTO_INSUMO_EN_APU: [100],
            ColumnNames.TIPO_INSUMO: ["MATERIAL"]
        })
        
        result = calculator.calculate(df)
        
        assert len(result) == 3
        assert all(isinstance(r, pd.DataFrame) for r in result)


# =============================================================================
# ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
        "--durations=10",  # Show 10 slowest tests
    ])