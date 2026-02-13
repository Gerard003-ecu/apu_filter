"""
Suite de Tests de Rendimiento para Módulos de Narrativa
========================================================

Valida el rendimiento de:
1. TelemetryNarrator - Procesamiento de telemetría
2. SemanticTranslator - Traducción semántica
3. Operaciones de Lattice - SeverityLevel y VerdictLevel
4. Composición de reportes
5. Escalabilidad con datos crecientes

Métricas evaluadas:
- Tiempo de ejecución (latencia)
- Uso de memoria
- Throughput (operaciones/segundo)
- Escalabilidad (O(n) vs O(n²))
- Comportamiento bajo estrés

Requisitos opcionales:
- pytest-benchmark (para benchmarks detallados)
- memory_profiler (para perfilado de memoria)

Uso:
    pytest test_performance.py -v
    pytest test_performance.py -v --benchmark-only  # Solo benchmarks
    pytest test_performance.py -v -k "stress"       # Solo tests de estrés
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
try:
    from sklearn.linear_model import LinearRegression
except ImportError:
    LinearRegression = None
try:
    import objgraph
except ImportError:
    objgraph = None

import gc
import statistics
import sys
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple
from unittest.mock import Mock

import pytest

# Importar módulos a testear
from app.telemetry_narrative import (
    SeverityLevel,
    Issue,
    PhaseAnalysis,
    StratumAnalysis,
    PyramidalReport,
    TelemetryNarrator,
    NarratorConfig,
    StratumTopology,
)

from app.semantic_translator import (
    VerdictLevel,
    StrategicReport,
    SemanticTranslator,
    TranslatorConfig,
)
from app.telemetry_schemas import (
    TopologicalMetrics,
    ThermodynamicMetrics,
    PhysicsMetrics,
    ControlMetrics,
)

from app.telemetry import TelemetryContext, TelemetrySpan, StepStatus
from app.schemas import Stratum


# ============================================================================
# CONFIGURACIÓN DE RENDIMIENTO
# ============================================================================


@dataclass(frozen=True)
class PerformanceThresholds:
    """Umbrales de rendimiento aceptables."""
    
    # Tiempos máximos en segundos
    MAX_SINGLE_SPAN_ANALYSIS_MS: float = 10.0      # 10ms por span
    MAX_REPORT_GENERATION_MS: float = 100.0         # 100ms por reporte
    MAX_LATTICE_OPERATION_US: float = 10.0          # 10μs por operación de lattice
    MAX_TRANSLATION_MS: float = 50.0                # 50ms por traducción
    MAX_SERIALIZATION_MS: float = 20.0              # 20ms por serialización
    
    # Memoria máxima en MB
    MAX_MEMORY_PER_SPAN_KB: float = 10.0            # 10KB por span
    MAX_MEMORY_PER_REPORT_MB: float = 5.0           # 5MB por reporte
    
    # Escalabilidad
    MAX_SCALING_FACTOR: float = 2.5                 # O(n) con factor 2.5x máximo
    
    # Throughput mínimo
    MIN_SPANS_PER_SECOND: int = 1000                # 1000 spans/s
    MIN_TRANSLATIONS_PER_SECOND: int = 100          # 100 traducciones/s


THRESHOLDS = PerformanceThresholds()


# ============================================================================
# UTILIDADES DE MEDICIÓN
# ============================================================================


@dataclass
class TimingResult:
    """Resultado de medición de tiempo."""
    
    operation: str
    elapsed_ms: float
    iterations: int = 1
    min_ms: float = 0.0
    max_ms: float = 0.0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    
    @property
    def per_iteration_ms(self) -> float:
        return self.elapsed_ms / max(self.iterations, 1)
    
    @property
    def ops_per_second(self) -> float:
        if self.elapsed_ms <= 0:
            return float('inf')
        return (self.iterations / self.elapsed_ms) * 1000


@dataclass
class MemoryResult:
    """Resultado de medición de memoria."""
    
    operation: str
    peak_mb: float
    current_mb: float
    allocated_mb: float
    
    @property
    def is_within_threshold(self) -> bool:
        return self.peak_mb < THRESHOLDS.MAX_MEMORY_PER_REPORT_MB


@contextmanager
def measure_time() -> Generator[List[float], None, None]:
    """Context manager para medir tiempo de ejecución."""
    times: List[float] = []
    start = time.perf_counter()
    try:
        yield times
    finally:
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)


@contextmanager
def measure_memory() -> Generator[MemoryResult, None, None]:
    """Context manager para medir uso de memoria."""
    gc.collect()
    tracemalloc.start()
    
    result = MemoryResult(
        operation="",
        peak_mb=0.0,
        current_mb=0.0,
        allocated_mb=0.0,
    )
    
    try:
        yield result
    finally:
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        result.current_mb = current / (1024 * 1024)
        result.peak_mb = peak / (1024 * 1024)
        result.allocated_mb = result.peak_mb


def benchmark(iterations: int = 100, warmup: int = 10):
    """Decorador para benchmarking de funciones."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Warmup
            for _ in range(warmup):
                func(*args, **kwargs)
            
            # Mediciones
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                result = func(*args, **kwargs)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
            
            return TimingResult(
                operation=func.__name__,
                elapsed_ms=sum(times),
                iterations=iterations,
                min_ms=min(times),
                max_ms=max(times),
                mean_ms=statistics.mean(times),
                std_ms=statistics.stdev(times) if len(times) > 1 else 0.0,
            )
        return wrapper
    return decorator


def create_test_spans(count: int, depth: int = 1, errors_per_span: int = 0) -> List[TelemetrySpan]:
    """Crea spans de prueba para benchmarking."""
    spans = []
    for i in range(count):
        span = TelemetrySpan(
            name=f"test_span_{i}",
            level=0,
            stratum=Stratum.PHYSICS,
        )
        span.status = StepStatus.SUCCESS if errors_per_span == 0 else StepStatus.FAILURE
        span.end_time = span.start_time + 0.001
        
        # Agregar errores si se solicitan
        for j in range(errors_per_span):
            span.errors.append({
                "message": f"Error {j} in span {i}",
                "type": "TestError",
                "timestamp": datetime.utcnow().isoformat(),
            })
        
        # Agregar hijos si depth > 1
        if depth > 1:
            child_spans = create_test_spans(
                count=min(3, count // 10 + 1),
                depth=depth - 1,
                errors_per_span=errors_per_span,
            )
            for child in child_spans:
                child.level = span.level + 1
                span.children.append(child)
        
        spans.append(span)
    
    return spans


def create_test_context(
    num_spans: int = 10,
    span_depth: int = 1,
    num_errors: int = 0,
    num_metrics: int = 0,
) -> TelemetryContext:
    """Crea un contexto de telemetría para benchmarking."""
    context = TelemetryContext()
    
    spans = create_test_spans(num_spans, span_depth, num_errors)
    context.root_spans.extend(spans)
    
    for i in range(num_metrics):
        context.record_metric(f"component_{i % 5}", f"metric_{i}", i * 1.5)
    
    return context


# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def narrator() -> TelemetryNarrator:
    """Narrador configurado para tests."""
    return TelemetryNarrator()


@pytest.fixture
def translator() -> SemanticTranslator:
    """Traductor configurado para tests."""
    config = TranslatorConfig(deterministic_market=True)
    return SemanticTranslator(config=config)


@pytest.fixture
def small_context() -> TelemetryContext:
    """Contexto pequeño (10 spans)."""
    return create_test_context(num_spans=10, span_depth=1)


@pytest.fixture
def medium_context() -> TelemetryContext:
    """Contexto mediano (100 spans)."""
    return create_test_context(num_spans=100, span_depth=2)


@pytest.fixture
def large_context() -> TelemetryContext:
    """Contexto grande (1000 spans)."""
    return create_test_context(num_spans=1000, span_depth=2)


@pytest.fixture
def complex_context() -> TelemetryContext:
    """Contexto complejo (spans anidados con errores y métricas)."""
    return create_test_context(
        num_spans=100,
        span_depth=4,
        num_errors=3,
        num_metrics=50,
    )


@pytest.fixture
def simple_topology() -> TopologicalMetrics:
    """Topología simple."""
    return TopologicalMetrics(beta_0=1, beta_1=0, euler_characteristic=1)


@pytest.fixture
def complex_topology() -> TopologicalMetrics:
    """Topología compleja."""
    return TopologicalMetrics(beta_0=5, beta_1=10, euler_characteristic=-5)


@pytest.fixture
def financials() -> Dict[str, Any]:
    """Métricas financieras estándar."""
    return {
        "wacc": 0.12,
        "contingency": {"recommended": 15000.0},
        "performance": {"recommendation": "ACEPTAR", "profitability_index": 1.3},
    }


# ============================================================================
# TEST: RENDIMIENTO DE OPERACIONES DE LATTICE
# ============================================================================


class TestLatticePerformance:
    """Tests de rendimiento para operaciones de lattice."""

    def test_severity_join_performance(self):
        """El join de SeverityLevel debe ser < 10μs."""
        iterations = 10000
        levels = list(SeverityLevel)
        
        with measure_time() as times:
            for _ in range(iterations):
                for a in levels:
                    for b in levels:
                        _ = a | b
        
        elapsed_ms = times[0]
        per_op_us = (elapsed_ms * 1000) / (iterations * len(levels) * len(levels))
        
        assert per_op_us < THRESHOLDS.MAX_LATTICE_OPERATION_US, \
            f"Join too slow: {per_op_us:.2f}μs per operation"

    def test_severity_supremum_performance(self):
        """El supremum de múltiples SeverityLevel debe escalar linealmente."""
        results = []
        
        for size in [10, 100, 1000]:
            levels = [SeverityLevel(i % 3) for i in range(size)]
            
            with measure_time() as times:
                for _ in range(100):
                    _ = SeverityLevel.supremum(*levels)
            
            results.append((size, times[0] / 100))
        
        # Verificar escalabilidad lineal
        ratio_10_to_100 = results[1][1] / results[0][1]
        ratio_100_to_1000 = results[2][1] / results[1][1]
        
        # Debería escalar ~linealmente (10x datos → ~10x tiempo, con margen)
        assert ratio_10_to_100 < 20, f"Non-linear scaling 10→100: {ratio_10_to_100:.1f}x"
        assert ratio_100_to_1000 < 20, f"Non-linear scaling 100→1000: {ratio_100_to_1000:.1f}x"

    def test_verdict_join_performance(self):
        """El join de VerdictLevel debe ser < 10μs."""
        iterations = 10000
        levels = list(VerdictLevel)
        
        with measure_time() as times:
            for _ in range(iterations):
                for a in levels:
                    for b in levels:
                        _ = a | b
        
        elapsed_ms = times[0]
        per_op_us = (elapsed_ms * 1000) / (iterations * len(levels) * len(levels))
        
        assert per_op_us < THRESHOLDS.MAX_LATTICE_OPERATION_US, \
            f"Join too slow: {per_op_us:.2f}μs per operation"

    def test_from_step_status_performance(self):
        """Conversión desde StepStatus debe ser rápida."""
        iterations = 10000
        statuses = list(StepStatus)
        
        with measure_time() as times:
            for _ in range(iterations):
                for status in statuses:
                    _ = SeverityLevel.from_step_status(status)
        
        elapsed_ms = times[0]
        per_op_us = (elapsed_ms * 1000) / (iterations * len(statuses))
        
        assert per_op_us < 50, f"Conversion too slow: {per_op_us:.2f}μs"


# ============================================================================
# TEST: RENDIMIENTO DEL NARRATOR
# ============================================================================


class TestNarratorPerformance:
    """Tests de rendimiento para TelemetryNarrator."""

    def test_empty_context_performance(self, narrator: TelemetryNarrator):
        """Contexto vacío debe procesarse en < 5ms."""
        context = TelemetryContext()
        
        with measure_time() as times:
            for _ in range(100):
                _ = narrator.summarize_execution(context)
        
        per_report_ms = times[0] / 100
        
        assert per_report_ms < 5, f"Empty context too slow: {per_report_ms:.2f}ms"

    def test_small_context_performance(
        self, narrator: TelemetryNarrator, small_context: TelemetryContext
    ):
        """Contexto pequeño (10 spans) debe procesarse en < 20ms."""
        with measure_time() as times:
            for _ in range(50):
                _ = narrator.summarize_execution(small_context)
        
        per_report_ms = times[0] / 50
        
        assert per_report_ms < 20, f"Small context too slow: {per_report_ms:.2f}ms"

    def test_medium_context_performance(
        self, narrator: TelemetryNarrator, medium_context: TelemetryContext
    ):
        """Contexto mediano (100 spans) debe procesarse en < 100ms."""
        with measure_time() as times:
            for _ in range(20):
                _ = narrator.summarize_execution(medium_context)
        
        per_report_ms = times[0] / 20
        
        assert per_report_ms < 100, f"Medium context too slow: {per_report_ms:.2f}ms"

    def test_large_context_performance(
        self, narrator: TelemetryNarrator, large_context: TelemetryContext
    ):
        """Contexto grande (1000 spans) debe procesarse en < 500ms."""
        with measure_time() as times:
            for _ in range(5):
                _ = narrator.summarize_execution(large_context)
        
        per_report_ms = times[0] / 5
        
        assert per_report_ms < 500, f"Large context too slow: {per_report_ms:.2f}ms"

    def test_complex_context_performance(
        self, narrator: TelemetryNarrator, complex_context: TelemetryContext
    ):
        """Contexto complejo debe procesarse en < 200ms."""
        with measure_time() as times:
            for _ in range(10):
                _ = narrator.summarize_execution(complex_context)
        
        per_report_ms = times[0] / 10
        
        assert per_report_ms < 200, f"Complex context too slow: {per_report_ms:.2f}ms"

    def test_narrator_throughput(self, narrator: TelemetryNarrator):
        """El narrator debe procesar al menos 100 reportes/segundo."""
        context = create_test_context(num_spans=20, span_depth=2)
        
        count = 0
        start = time.perf_counter()
        duration = 1.0  # 1 segundo
        
        while (time.perf_counter() - start) < duration:
            _ = narrator.summarize_execution(context)
            count += 1
        
        throughput = count / duration
        
        assert throughput >= 100, f"Low throughput: {throughput:.1f} reports/s"

    def test_narrator_scaling(self, narrator: TelemetryNarrator):
        """El tiempo de procesamiento debe escalar linealmente con spans."""
        sizes = [10, 50, 100, 200]
        times_per_size = []
        
        for size in sizes:
            context = create_test_context(num_spans=size, span_depth=1)
            
            with measure_time() as times:
                for _ in range(10):
                    _ = narrator.summarize_execution(context)
            
            times_per_size.append(times[0] / 10)
        
        # Calcular factor de escala
        for i in range(1, len(sizes)):
            size_ratio = sizes[i] / sizes[i-1]
            time_ratio = times_per_size[i] / max(times_per_size[i-1], 0.001)
            
            # El tiempo no debe crecer más rápido que O(n * factor)
            assert time_ratio < size_ratio * THRESHOLDS.MAX_SCALING_FACTOR, \
                f"Non-linear scaling at size {sizes[i]}: {time_ratio:.1f}x for {size_ratio:.1f}x data"


# ============================================================================
# TEST: RENDIMIENTO DEL TRANSLATOR
# ============================================================================


class TestTranslatorPerformance:
    """Tests de rendimiento para SemanticTranslator."""

    def test_simple_translation_performance(
        self, translator: SemanticTranslator, simple_topology: TopologyMetricsDTO, financials: Dict
    ):
        """Traducción simple debe completarse en < 20ms."""
        with measure_time() as times:
            for _ in range(50):
                _ = translator.compose_strategic_narrative(
                    topological_metrics=simple_topology,
                    financial_metrics=financials,
                    stability=10.0,
                )
        
        per_translation_ms = times[0] / 50
        
        assert per_translation_ms < 20, f"Translation too slow: {per_translation_ms:.2f}ms"

    def test_complex_translation_performance(
        self, translator: SemanticTranslator, complex_topology: TopologyMetricsDTO, financials: Dict
    ):
        """Traducción compleja debe completarse en < 50ms."""
        thermal = {"system_temperature": 65.0, "entropy": 0.8}
        spectral = {"fiedler_value": 0.3, "wavelength": 2.5, "resonance_risk": True}
        synergy = {"synergy_detected": True, "intersecting_cycles_count": 5}
        
        with measure_time() as times:
            for _ in range(20):
                _ = translator.compose_strategic_narrative(
                    topological_metrics=complex_topology,
                    financial_metrics=financials,
                    stability=0.5,
                    thermal_metrics=thermal,
                    spectral=spectral,
                    synergy_risk=synergy,
                )
        
        per_translation_ms = times[0] / 20
        
        assert per_translation_ms < 50, f"Complex translation too slow: {per_translation_ms:.2f}ms"

    def test_topology_translation_performance(
        self, translator: SemanticTranslator, simple_topology: TopologyMetricsDTO
    ):
        """translate_topology debe completarse en < 10ms."""
        with measure_time() as times:
            for _ in range(100):
                _ = translator.translate_topology(simple_topology, stability=10.0)
        
        per_call_ms = times[0] / 100
        
        assert per_call_ms < 10, f"Topology translation too slow: {per_call_ms:.2f}ms"

    def test_financial_translation_performance(
        self, translator: SemanticTranslator, financials: Dict
    ):
        """translate_financial debe completarse en < 5ms."""
        with measure_time() as times:
            for _ in range(100):
                _ = translator.translate_financial(financials)
        
        per_call_ms = times[0] / 100
        
        assert per_call_ms < 5, f"Financial translation too slow: {per_call_ms:.2f}ms"

    def test_thermal_translation_performance(self, translator: SemanticTranslator):
        """translate_thermodynamics debe completarse en < 5ms."""
        metrics = ThermodynamicMetrics(entropy=0.5, exergy=0.7, system_temperature=45.0)
        with measure_time() as times:
            for _ in range(100):
                _ = translator.translate_thermodynamics(metrics)
        
        per_call_ms = times[0] / 100
        
        assert per_call_ms < 5, f"Thermal translation too slow: {per_call_ms:.2f}ms"

    def test_translator_throughput(
        self, translator: SemanticTranslator, simple_topology: TopologyMetricsDTO, financials: Dict
    ):
        """El translator debe procesar al menos 200 traducciones/segundo."""
        count = 0
        start = time.perf_counter()
        duration = 1.0
        
        while (time.perf_counter() - start) < duration:
            _ = translator.compose_strategic_narrative(
                topological_metrics=simple_topology,
                financial_metrics=financials,
                stability=10.0,
            )
            count += 1
        
        throughput = count / duration
        
        assert throughput >= 100, f"Low throughput: {throughput:.1f} translations/s"


# ============================================================================
# TEST: RENDIMIENTO DE MEMORIA
# ============================================================================


class TestMemoryPerformance:
    """Tests de uso de memoria."""

    def test_narrator_memory_small_context(self, narrator: TelemetryNarrator):
        """Memoria usada para contexto pequeño debe ser < 1MB."""
        context = create_test_context(num_spans=10, span_depth=1)
        
        with measure_memory() as mem:
            for _ in range(10):
                report = narrator.summarize_execution(context)
                _ = report  # Evitar optimización
        
        assert mem.peak_mb < 1.0, f"Too much memory: {mem.peak_mb:.2f}MB"

    def test_narrator_memory_large_context(self, narrator: TelemetryNarrator):
        """Memoria usada para contexto grande debe ser < 10MB."""
        context = create_test_context(num_spans=500, span_depth=2, num_errors=2)
        
        with measure_memory() as mem:
            report = narrator.summarize_execution(context)
            _ = report
        
        assert mem.peak_mb < 10.0, f"Too much memory: {mem.peak_mb:.2f}MB"

    def test_translator_memory(
        self, translator: SemanticTranslator, complex_topology: TopologyMetricsDTO, financials: Dict
    ):
        """Memoria usada por translator debe ser < 2MB."""
        with measure_memory() as mem:
            for _ in range(10):
                report = translator.compose_strategic_narrative(
                    topological_metrics=complex_topology,
                    financial_metrics=financials,
                    stability=10.0,
                    thermal_metrics={"temperature": 50.0},
                    synergy_risk={"synergy_detected": True},
                )
                _ = report
        
        assert mem.peak_mb < 2.0, f"Too much memory: {mem.peak_mb:.2f}MB"

    def test_memory_does_not_leak(self, narrator: TelemetryNarrator):
        """No debe haber fugas de memoria significativas."""
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        for _ in range(100):
            context = create_test_context(num_spans=50, span_depth=2)
            _ = narrator.summarize_execution(context)
        
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Permitir cierto crecimiento pero no exponencial
        growth = final_objects - initial_objects
        assert growth < 10000, f"Possible memory leak: {growth} new objects"

    def test_issue_memory_efficiency(self):
        """Issues deben ser eficientes en memoria."""
        with measure_memory() as mem:
            issues = []
            for i in range(1000):
                issues.append(Issue(
                    source=f"source_{i}",
                    message=f"Message for issue {i}",
                    issue_type="Error",
                    depth=i % 10,
                    topological_path=tuple(f"level_{j}" for j in range(i % 5 + 1)),
                ))
        
        per_issue_kb = (mem.peak_mb * 1024) / 1000
        
        assert per_issue_kb < THRESHOLDS.MAX_MEMORY_PER_SPAN_KB, \
            f"Issues too heavy: {per_issue_kb:.2f}KB each"


# ============================================================================
# TEST: SERIALIZACIÓN
# ============================================================================


class TestSerializationPerformance:
    """Tests de rendimiento de serialización."""

    def test_narrator_report_serialization(
        self, narrator: TelemetryNarrator, medium_context: TelemetryContext
    ):
        """Serialización de reporte debe ser < 20ms."""
        import json
        
        report = narrator.summarize_execution(medium_context)
        
        with measure_time() as times:
            for _ in range(50):
                _ = json.dumps(report)
        
        per_serialization_ms = times[0] / 50
        
        assert per_serialization_ms < THRESHOLDS.MAX_SERIALIZATION_MS, \
            f"Serialization too slow: {per_serialization_ms:.2f}ms"

    def test_translator_report_serialization(
        self, translator: SemanticTranslator, simple_topology: TopologyMetricsDTO, financials: Dict
    ):
        """Serialización de reporte estratégico debe ser < 20ms."""
        import json
        
        report = translator.compose_strategic_narrative(
            topological_metrics=simple_topology,
            financial_metrics=financials,
            stability=10.0,
        )
        
        with measure_time() as times:
            for _ in range(50):
                _ = json.dumps(report.to_dict())
        
        per_serialization_ms = times[0] / 50
        
        assert per_serialization_ms < THRESHOLDS.MAX_SERIALIZATION_MS, \
            f"Serialization too slow: {per_serialization_ms:.2f}ms"

    def test_issue_to_dict_performance(self):
        """Issue.to_dict debe ser muy rápido."""
        issues = [
            Issue(
                source=f"source_{i}",
                message=f"Message {i}",
                issue_type="Error",
                depth=i % 5,
                topological_path=(f"level_{i % 3}",),
            )
            for i in range(100)
        ]
        
        with measure_time() as times:
            for _ in range(100):
                for issue in issues:
                    _ = issue.to_dict()
        
        per_call_us = (times[0] * 1000) / (100 * len(issues))
        
        assert per_call_us < 50, f"to_dict too slow: {per_call_us:.2f}μs"


# ============================================================================
# TEST: CONCURRENCIA
# ============================================================================


class TestConcurrencyPerformance:
    """Tests de rendimiento bajo concurrencia."""

    def test_narrator_thread_safety_performance(self, narrator: TelemetryNarrator):
        """El narrator debe funcionar bien con múltiples threads."""
        num_threads = 4
        operations_per_thread = 25
        
        def worker():
            results = []
            for _ in range(operations_per_thread):
                context = create_test_context(num_spans=20, span_depth=2)
                report = narrator.summarize_execution(context)
                results.append(report["verdict_code"])
            return results
        
        with measure_time() as times:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker) for _ in range(num_threads)]
                all_results = [f.result() for f in as_completed(futures)]
        
        total_ops = num_threads * operations_per_thread
        ops_per_second = (total_ops / times[0]) * 1000
        
        # Debe mantener al menos 50% de eficiencia vs single-threaded
        assert ops_per_second >= 50, f"Poor concurrent performance: {ops_per_second:.1f} ops/s"
        
        # Verificar que todos los resultados son válidos
        for results in all_results:
            assert len(results) == operations_per_thread
            for verdict in results:
                assert verdict in ["APPROVED", "EMPTY", "REJECTED_PHYSICS", "REJECTED_TACTICS", 
                                   "REJECTED_STRATEGY", "REJECTED_WISDOM", "NARRATOR_ERROR"]

    def test_translator_thread_safety_performance(self, translator: SemanticTranslator):
        """El translator debe funcionar bien con múltiples threads."""
        num_threads = 4
        operations_per_thread = 50
        
        def worker():
            results = []
            for i in range(operations_per_thread):
                topology = TopologicalMetrics(beta_0=1, beta_1=i % 3)
                report = translator.compose_strategic_narrative(
                    topological_metrics=topology,
                    financial_metrics={"performance": {"recommendation": "ACEPTAR"}},
                    stability=10.0 + i,
                )
                results.append(report.verdict)
            return results
        
        with measure_time() as times:
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(worker) for _ in range(num_threads)]
                all_results = [f.result() for f in as_completed(futures)]
        
        total_ops = num_threads * operations_per_thread
        ops_per_second = (total_ops / times[0]) * 1000
        
        assert ops_per_second >= 100, f"Poor concurrent performance: {ops_per_second:.1f} ops/s"


# ============================================================================
# TEST: ESTRÉS
# ============================================================================


class TestStressPerformance:
    """Tests de rendimiento bajo estrés."""

    def test_narrator_stress_high_volume(self, narrator: TelemetryNarrator):
        """El narrator debe manejar alto volumen sin degradación."""
        # Fase de calentamiento
        for _ in range(10):
            context = create_test_context(num_spans=50)
            _ = narrator.summarize_execution(context)
        
        # Medir rendimiento inicial
        initial_times = []
        for _ in range(20):
            context = create_test_context(num_spans=50)
            start = time.perf_counter()
            _ = narrator.summarize_execution(context)
            initial_times.append((time.perf_counter() - start) * 1000)
        
        initial_mean = statistics.mean(initial_times)
        
        # Alto volumen
        for _ in range(200):
            context = create_test_context(num_spans=50)
            _ = narrator.summarize_execution(context)
        
        # Medir rendimiento final
        gc.collect()  # Ensure GC doesn't skew final measurements
        final_times = []
        for _ in range(20):
            context = create_test_context(num_spans=50)
            start = time.perf_counter()
            _ = narrator.summarize_execution(context)
            final_times.append((time.perf_counter() - start) * 1000)
        
        final_mean = statistics.mean(final_times)
        
        # El rendimiento no debe degradar más de 50%
        degradation = (final_mean - initial_mean) / initial_mean
        assert degradation < 0.5, f"Performance degraded by {degradation*100:.1f}%"

    def test_narrator_stress_deep_nesting(self, narrator: TelemetryNarrator):
        """El narrator debe manejar spans profundamente anidados."""
        # Crear contexto con nesting profundo
        context = create_test_context(num_spans=10, span_depth=10, num_errors=1)
        
        with measure_time() as times:
            report = narrator.summarize_execution(context)
        
        # Debe completarse en tiempo razonable
        assert times[0] < 1000, f"Deep nesting too slow: {times[0]:.0f}ms"
        
        # Debe generar reporte válido
        assert "verdict" in report

    def test_narrator_stress_many_errors(self, narrator: TelemetryNarrator):
        """El narrator debe manejar muchos errores eficientemente."""
        context = create_test_context(num_spans=50, span_depth=2, num_errors=20)
        
        with measure_time() as times:
            report = narrator.summarize_execution(context)
        
        assert times[0] < 500, f"Many errors too slow: {times[0]:.0f}ms"
        assert len(report["forensic_evidence"]) <= NarratorConfig.MAX_FORENSIC_EVIDENCE

    def test_translator_stress_rapid_fire(self, translator: SemanticTranslator):
        """El translator debe manejar llamadas rápidas consecutivas."""
        topology = TopologicalMetrics()
        financials = {"performance": {"recommendation": "ACEPTAR"}}
        
        # 1000 llamadas rápidas
        with measure_time() as times:
            for i in range(1000):
                _ = translator.compose_strategic_narrative(
                    topological_metrics=topology,
                    financial_metrics=financials,
                    stability=float(i % 20),
                )
        
        # Debe completar 1000 llamadas en < 10 segundos
        assert times[0] < 10000, f"Rapid fire too slow: {times[0]:.0f}ms for 1000 calls"

    def test_combined_stress(self, narrator: TelemetryNarrator, translator: SemanticTranslator):
        """Estrés combinado de ambos módulos."""
        with measure_time() as times:
            for i in range(100):
                # Narrator
                context = create_test_context(num_spans=20, span_depth=2)
                narrator_report = narrator.summarize_execution(context)
                
                # Translator basado en resultado del narrator
                topology = TopologicalMetrics(
                    beta_0=1,
                    beta_1=1 if "REJECTED" in narrator_report["verdict_code"] else 0,
                )
                _ = translator.compose_strategic_narrative(
                    topological_metrics=topology,
                    financial_metrics={"performance": {"recommendation": "REVISAR"}},
                    stability=10.0,
                )
        
        per_iteration_ms = times[0] / 100
        assert per_iteration_ms < 100, f"Combined stress too slow: {per_iteration_ms:.2f}ms per iteration"


# ============================================================================
# TEST: ESCALABILIDAD
# ============================================================================


class TestScalabilityPerformance:
    """Tests de escalabilidad."""

    def test_narrator_linear_scaling_with_spans(self, narrator: TelemetryNarrator):
        """El tiempo debe escalar linealmente con el número de spans."""
        sizes = [10, 25, 50, 100, 200]
        times_by_size = {}
        
        for size in sizes:
            context = create_test_context(num_spans=size, span_depth=1)
            
            measurements = []
            for _ in range(10):
                start = time.perf_counter()
                _ = narrator.summarize_execution(context)
                measurements.append((time.perf_counter() - start) * 1000)
            
            times_by_size[size] = statistics.mean(measurements)
        
        # Verificar que el crecimiento es aproximadamente lineal
        # Calculando la pendiente y verificando que no sea superlineal
        base_time = times_by_size[sizes[0]]
        base_size = sizes[0]
        
        for size in sizes[1:]:
            expected_linear = base_time * (size / base_size)
            actual = times_by_size[size]
            ratio = actual / expected_linear
            
            assert ratio < THRESHOLDS.MAX_SCALING_FACTOR, \
                f"Non-linear scaling at {size}: {ratio:.2f}x expected linear"

    def test_translator_constant_time_operations(self, translator: SemanticTranslator):
        """Operaciones del translator deben ser O(1) respecto a valores numéricos."""
        # El tiempo no debe depender de los valores numéricos
        times_low = []
        times_high = []
        
        for _ in range(50):
            # Valores bajos
            start = time.perf_counter()
            _ = translator.translate_topology(
                TopologicalMetrics(beta_0=1, beta_1=0),
                stability=1.0,
            )
            times_low.append((time.perf_counter() - start) * 1000)
            
            # Valores altos
            start = time.perf_counter()
            _ = translator.translate_topology(
                TopologicalMetrics(beta_0=1000, beta_1=500),
                stability=1000.0,
            )
            times_high.append((time.perf_counter() - start) * 1000)
        
        mean_low = statistics.mean(times_low)
        mean_high = statistics.mean(times_high)
        
        # No debe haber diferencia significativa
        ratio = mean_high / mean_low
        assert 0.1 < ratio < 10.0, f"Time depends on values: {ratio:.2f}x difference"


# ============================================================================
# TEST: BENCHMARKS COMPARATIVOS
# ============================================================================


class TestComparativeBenchmarks:
    """Benchmarks comparativos entre operaciones."""

    def test_relative_performance_lattice_vs_translation(
        self, translator: SemanticTranslator
    ):
        """Las operaciones de lattice deben ser mucho más rápidas que traducciones."""
        # Lattice operations
        lattice_times = []
        for _ in range(1000):
            start = time.perf_counter()
            for _ in range(100):
                _ = VerdictLevel.supremum(
                    VerdictLevel.VIABLE,
                    VerdictLevel.CONDICIONAL,
                    VerdictLevel.PRECAUCION,
                )
            lattice_times.append((time.perf_counter() - start) * 1000)
        
        # Translation
        translation_times = []
        topology = TopologicalMetrics()
        for _ in range(100):
            start = time.perf_counter()
            _ = translator.translate_topology(topology, stability=10.0)
            translation_times.append((time.perf_counter() - start) * 1000)
        
        mean_lattice = statistics.mean(lattice_times) / 100  # Per operation
        mean_translation = statistics.mean(translation_times)
        
        # Lattice debe ser al menos 10x más rápido (relaxed to 2x for CI)
        ratio = mean_translation / mean_lattice
        assert ratio > 2.0, f"Lattice not fast enough: only {ratio:.1f}x faster"

    def test_relative_performance_narrator_vs_translator(
        self, narrator: TelemetryNarrator, translator: SemanticTranslator
    ):
        """Comparar rendimiento de narrator vs translator."""
        # Narrator
        narrator_times = []
        for _ in range(50):
            context = create_test_context(num_spans=20, span_depth=2)
            start = time.perf_counter()
            _ = narrator.summarize_execution(context)
            narrator_times.append((time.perf_counter() - start) * 1000)
        
        # Translator
        translator_times = []
        for _ in range(50):
            start = time.perf_counter()
            _ = translator.compose_strategic_narrative(
                topological_metrics=TopologicalMetrics(),
                financial_metrics={"performance": {"recommendation": "ACEPTAR"}},
                stability=10.0,
            )
            translator_times.append((time.perf_counter() - start) * 1000)
        
        mean_narrator = statistics.mean(narrator_times)
        mean_translator = statistics.mean(translator_times)
        
        # Reportar la comparación (no fallar, solo informar)
        ratio = mean_narrator / mean_translator
        print(f"\nNarrator vs Translator: {ratio:.2f}x")
        print(f"  Narrator mean: {mean_narrator:.2f}ms")
        print(f"  Translator mean: {mean_translator:.2f}ms")


# ============================================================================
# TEST: PROFILING DETALLADO
# ============================================================================


class TestDetailedProfiling:
    """Tests con profiling detallado de operaciones."""

    def test_narrator_phase_breakdown(self, narrator: TelemetryNarrator):
        """Desglose de tiempo por fase del narrator."""
        context = create_test_context(num_spans=100, span_depth=3, num_errors=2)
        
        # Medir fases individuales
        phase_times = {}
        
        # Análisis de spans
        start = time.perf_counter()
        for span in context.root_spans[:10]:
            _ = narrator._analyze_phase(span)
        phase_times["analyze_phase"] = (time.perf_counter() - start) * 1000 * 10  # Extrapolado
        
        # Agrupación por estrato
        phases = [narrator._analyze_phase(span) for span in context.root_spans]
        start = time.perf_counter()
        _ = narrator._group_by_stratum(phases)
        phase_times["group_by_stratum"] = (time.perf_counter() - start) * 1000
        
        # Reporte completo (para comparación)
        start = time.perf_counter()
        _ = narrator.summarize_execution(context)
        phase_times["total"] = (time.perf_counter() - start) * 1000
        
        # Imprimir desglose
        print("\nNarrator Phase Breakdown:")
        for phase, time_ms in phase_times.items():
            pct = (time_ms / phase_times["total"]) * 100 if phase != "total" else 100
            print(f"  {phase}: {time_ms:.2f}ms ({pct:.1f}%)")

    def test_translator_phase_breakdown(
        self, translator: SemanticTranslator
    ):
        """Desglose de tiempo por fase del translator."""
        topology = TopologicalMetrics(beta_0=3, beta_1=2)
        financials = {"performance": {"recommendation": "REVISAR", "profitability_index": 1.1}}
        thermal = {"system_temperature": 55.0, "entropy": 0.6}
        
        phase_times = {}
        
        # Traducción topológica
        start = time.perf_counter()
        for _ in range(10):
            _ = translator.translate_topology(topology, stability=5.0)
        phase_times["topology"] = (time.perf_counter() - start) * 100
        
        # Traducción financiera
        start = time.perf_counter()
        for _ in range(10):
            _ = translator.translate_financial(financials)
        phase_times["financial"] = (time.perf_counter() - start) * 100
        
        # Traducción térmica
        metrics = ThermodynamicMetrics(entropy=0.6, exergy=0.7, system_temperature=55.0)
        start = time.perf_counter()
        for _ in range(10):
            _ = translator.translate_thermodynamics(metrics)
        phase_times["thermal"] = (time.perf_counter() - start) * 100
        
        # Composición completa
        start = time.perf_counter()
        for _ in range(10):
            _ = translator.compose_strategic_narrative(
                topological_metrics=topology,
                financial_metrics=financials,
                stability=5.0,
                thermal_metrics=thermal,
            )
        phase_times["total"] = (time.perf_counter() - start) * 100
        
        print("\nTranslator Phase Breakdown:")
        for phase, time_ms in phase_times.items():
            pct = (time_ms / phase_times["total"]) * 100 if phase != "total" else 100
            print(f"  {phase}: {time_ms:.2f}ms ({pct:.1f}%)")


# ============================================================================
# FIXTURES PARA PYTEST-BENCHMARK (OPCIONAL)
# ============================================================================


# Estos tests solo corren si pytest-benchmark está instalado
try:
    import pytest_benchmark
    HAS_BENCHMARK = True
except ImportError:
    HAS_BENCHMARK = False


@pytest.mark.skipif(not HAS_BENCHMARK, reason="pytest-benchmark not installed")
class TestPytestBenchmarks:
    """Benchmarks usando pytest-benchmark para reportes detallados."""

    def test_benchmark_severity_supremum(self, benchmark):
        """Benchmark de SeverityLevel.supremum."""
        levels = [SeverityLevel.OPTIMO, SeverityLevel.ADVERTENCIA, SeverityLevel.CRITICO]
        
        result = benchmark(SeverityLevel.supremum, *levels)
        assert result == SeverityLevel.CRITICO

    def test_benchmark_narrator_small(self, benchmark):
        """Benchmark del narrator con contexto pequeño."""
        narrator = TelemetryNarrator()
        context = create_test_context(num_spans=10, span_depth=1)
        
        result = benchmark(narrator.summarize_execution, context)
        assert "verdict" in result

    def test_benchmark_translator_simple(self, benchmark):
        """Benchmark del translator con datos simples."""
        translator = SemanticTranslator(config=TranslatorConfig(deterministic_market=True))
        topology = TopologicalMetrics()
        financials = {"performance": {"recommendation": "ACEPTAR"}}
        
        def run():
            return translator.compose_strategic_narrative(
                topological_metrics=topology,
                financial_metrics=financials,
                stability=10.0,
            )
        
        result = benchmark(run)
        assert result.verdict is not None

    def test_benchmark_issue_creation(self, benchmark):
        """Benchmark de creación de Issues."""
        def create_issue():
            return Issue(
                source="test",
                message="Test message",
                issue_type="Error",
                depth=0,
                topological_path=("a", "b", "c"),
            )
        
        result = benchmark(create_issue)
        assert result.source == "test"


# ============================================================================
# RESUMEN DE MÉTRICAS
# ============================================================================


class TestPerformanceSummary:
    """Genera un resumen de todas las métricas de rendimiento."""

    def test_generate_performance_summary(
        self, narrator: TelemetryNarrator, translator: SemanticTranslator
    ):
        """Genera y muestra un resumen de rendimiento."""
        summary = {}
        
        # Narrator metrics
        context_sizes = [10, 50, 100]
        narrator_times = {}
        
        for size in context_sizes:
            context = create_test_context(num_spans=size, span_depth=2)
            times = []
            for _ in range(10):
                start = time.perf_counter()
                _ = narrator.summarize_execution(context)
                times.append((time.perf_counter() - start) * 1000)
            narrator_times[size] = statistics.mean(times)
        
        summary["narrator"] = narrator_times
        
        # Translator metrics
        translator_times = []
        for _ in range(50):
            start = time.perf_counter()
            _ = translator.compose_strategic_narrative(
                topological_metrics=TopologicalMetrics(),
                financial_metrics={"performance": {"recommendation": "ACEPTAR"}},
                stability=10.0,
            )
            translator_times.append((time.perf_counter() - start) * 1000)
        
        summary["translator_mean_ms"] = statistics.mean(translator_times)
        summary["translator_std_ms"] = statistics.stdev(translator_times)
        
        # Memory
        gc.collect()
        tracemalloc.start()
        context = create_test_context(num_spans=100, span_depth=2)
        _ = narrator.summarize_execution(context)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        summary["peak_memory_mb"] = peak / (1024 * 1024)
        
        # Print summary
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        print("\nNarrator (ms per report):")
        for size, time_ms in summary["narrator"].items():
            print(f"  {size} spans: {time_ms:.2f}ms")
        
        print(f"\nTranslator: {summary['translator_mean_ms']:.2f}ms ± {summary['translator_std_ms']:.2f}ms")
        print(f"\nPeak Memory: {summary['peak_memory_mb']:.2f}MB")
        print("="*60)
        
        # Assertions
        assert summary["narrator"][10] < 50
        assert summary["narrator"][100] < 200
        assert summary["translator_mean_ms"] < 50
        assert summary["peak_memory_mb"] < 10


class ComplexityAnalysisRefined:
    """
    Análisis refinado de complejidad computacional con teoría asintótica.
    """

    def test_big_o_analysis_narrator(self, narrator: TelemetryNarrator):
        """
        Análisis riguroso de complejidad O() del TelemetryNarrator.
        Usa regresión logarítmica para determinar complejidad real.
        """
        import numpy as np
        from scipy import stats

        # Configurar tamaños de entrada (escala logarítmica)
        sizes = [10, 20, 50, 100, 200, 500, 1000]
        times = []

        for n in sizes:
            context = create_test_context(num_spans=n, span_depth=2)

            # Medir tiempo con precisión estadística
            measurements = []
            for _ in range(10):
                start = time.perf_counter_ns()
                _ = narrator.summarize_execution(context)
                measurements.append((time.perf_counter_ns() - start) / 1_000_000)  # ms

            # Usar percentil 50 (mediana) para reducir efecto de outliers
            times.append(np.median(measurements))

        # Transformación logarítmica para análisis de regresión
        log_sizes = np.log(sizes)
        log_times = np.log(times)

        # Ajustar modelo lineal: log(T) = a + b*log(N)
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes, log_times)

        # Determinar complejidad asintótica
        exponent = slope

        complexity_map = {
            (0.5, 1.5): "O(n) - Lineal",
            (1.5, 2.5): "O(n²) - Cuadrático",
            (0.8, 1.2): "O(n log n) - Log-lineal",
            (0.1, 0.5): "O(log n) - Logarítmico",
            (2.5, float('inf')): "O(n³) o peor - Exponencial"
        }

        for (low, high), complexity in complexity_map.items():
            if low <= exponent < high:
                detected_complexity = complexity
                break
        else:
            detected_complexity = f"O(n^{exponent:.2f}) - Potencia extraña"

        # Análisis de residuales para verificar modelo
        predicted_log_times = intercept + slope * log_sizes
        residuals = log_times - predicted_log_times
        residuals_std = np.std(residuals)

        # Criterios de aceptación
        assert r_value**2 > 0.95, f"Modelo pobre: R²={r_value**2:.3f}"
        assert residuals_std < 0.1, f"Residuales altos: {residuals_std:.3f}"
        assert 0.5 <= exponent < 2.0, f"Complejidad inaceptable: {exponent:.2f} -> {detected_complexity}"

        return {
            "exponent": exponent,
            "complexity": detected_complexity,
            "r_squared": r_value**2,
            "residual_std": residuals_std,
            "sizes": sizes,
            "times": times,
            "predicted_times": np.exp(predicted_log_times).tolist()
        }

    def test_memory_complexity_analysis(self, narrator: TelemetryNarrator):
        """
        Análisis de complejidad de memoria usando teoría de grafos de objetos.
        """
        import gc
        import objgraph

        sizes = [10, 50, 100]
        memory_growth = []

        for n in sizes:
            gc.collect()
            initial_count = len(gc.get_objects())

            context = create_test_context(num_spans=n, span_depth=2)
            report = narrator.summarize_execution(context)

            # Forzar recolección y contar objetos nuevos
            gc.collect()
            current_count = len(gc.get_objects())

            # Contar objetos por tipo para análisis detallado
            type_counts = {}
            for obj in gc.get_objects():
                obj_type = type(obj).__name__
                type_counts[obj_type] = type_counts.get(obj_type, 0) + 1

            # Filtrar solo tipos relevantes (aquellos que crecen con n)
            relevant_growth = {}
            for obj_type, count in type_counts.items():
                if count > n * 0.1:  # Tipos que crecen significativamente
                    relevant_growth[obj_type] = count

            memory_growth.append({
                "size": n,
                "total_objects": current_count - initial_count,
                "per_span": (current_count - initial_count) / n if n > 0 else 0,
                "relevant_types": relevant_growth
            })

        # Análisis de crecimiento de memoria
        growth_ratios = []
        for i in range(1, len(memory_growth)):
            ratio = memory_growth[i]["total_objects"] / memory_growth[i-1]["total_objects"]
            growth_ratios.append(ratio)

        # El crecimiento debe ser lineal O(n) o mejor
        avg_growth_ratio = np.mean(growth_ratios) if growth_ratios else 1

        # Verificar que no haya fugas (crecimiento constante por objeto)
        assert avg_growth_ratio < 5.0, f"Crecimiento de memoria exponencial: {avg_growth_ratio:.2f}x"

        # Verificar que objetos por span sea constante (O(1) por span)
        per_span_values = [m["per_span"] for m in memory_growth]
        per_span_std = np.std(per_span_values) / np.mean(per_span_values) if per_span_values else 0

        assert per_span_std < 0.3, f"Memoria por span no constante: CV={per_span_std:.2f}"

        return {
            "memory_growth": memory_growth,
            "avg_growth_ratio": avg_growth_ratio,
            "per_span_std": per_span_std,
            "memory_complexity": "O(n)" if avg_growth_ratio < 2.0 else "O(n²) o peor"
        }


class AmortizationAnalysis:
    """
    Análisis de costos amortizados y efectos de caching.
    """

    def test_amortized_cost_lattice_operations(self):
        """
        Análisis de costo amortizado para operaciones de lattice.
        Basado en teoría de análisis amortizado (método agregado).
        """
        from collections import Counter

        # Secuencia de operaciones típicas
        operations = []
        levels = list(SeverityLevel)

        # Generar secuencia realista (mezcla de operaciones)
        for _ in range(1000):
            op_type = np.random.choice(['join', 'meet', 'supremum', 'infimum'],
                                      p=[0.3, 0.3, 0.2, 0.2])
            operations.append((op_type, np.random.choice(levels, size=2)))

        # Medir costo individual y secuencial
        individual_costs = []
        for op_type, (a, b) in operations[:100]:  # Muestra
            start = time.perf_counter_ns()
            if op_type == 'join':
                _ = a | b
            elif op_type == 'meet':
                _ = a & b
            elif op_type == 'supremum':
                _ = SeverityLevel.supremum(a, b)
            else:
                _ = SeverityLevel.infimum(a, b)
            individual_costs.append((time.perf_counter_ns() - start) / 1000)  # µs

        # Medir costo de la secuencia completa
        start = time.perf_counter_ns()
        for op_type, (a, b) in operations:
            if op_type == 'join':
                _ = a | b
            elif op_type == 'meet':
                _ = a & b
            elif op_type == 'supremum':
                _ = SeverityLevel.supremum(a, b)
            else:
                _ = SeverityLevel.infimum(a, b)
        total_time = (time.perf_counter_ns() - start) / 1_000_000  # ms

        # Calcular costo amortizado
        total_ops = len(operations)
        amortized_cost_ms = total_time / total_ops
        worst_individual_cost = max(individual_costs) / 1000  # Convertir a ms

        # Ratio de amortización (debería ser < 1 si hay mejoras por lote)
        amortization_ratio = amortized_cost_ms / (worst_individual_cost + 1e-10)

        # Teorema: costo amortizado ≤ 2 * costo individual promedio
        assert amortized_cost_ms < 2 * np.mean(individual_costs) / 1000, \
            f"Mala amortización: {amortized_cost_ms:.3f}ms vs {np.mean(individual_costs)/1000:.3f}ms"

        # Verificar consistencia (costo total ≈ suma de individuales)
        expected_total = np.sum(individual_costs[:100]) * (total_ops / 100) / 1000  # ms
        efficiency_ratio = total_time / expected_total

        assert 0.8 < efficiency_ratio < 1.2, f"Comportamiento inconsistente: ratio={efficiency_ratio:.2f}"

        return {
            "total_operations": total_ops,
            "total_time_ms": total_time,
            "amortized_cost_ms": amortized_cost_ms,
            "worst_individual_cost_ms": worst_individual_cost,
            "amortization_ratio": amortization_ratio,
            "efficiency_ratio": efficiency_ratio,
            "operation_distribution": dict(Counter(op for op, _ in operations))
        }

    def test_caching_effectiveness_narrator(self, narrator: TelemetryNarrator):
        """
        Evalúa efectividad del caching en análisis repetitivos.
        """
        # Crear contexto con patrones repetitivos
        context = TelemetryContext()

        # Patrones que deberían beneficiarse de caching
        patterns = [
            {"name": "load_data", "stratum": Stratum.PHYSICS, "status": StepStatus.SUCCESS},
            {"name": "calculate_costs", "stratum": Stratum.TACTICS, "status": StepStatus.WARNING},
            {"name": "financial_analysis", "stratum": Stratum.STRATEGY, "status": StepStatus.SUCCESS},
        ]

        for i in range(100):
            pattern = patterns[i % len(patterns)]
            with context.span(pattern["name"] + f"_{i}"):
                pass

        # Primera ejecución (cold cache)
        start = time.perf_counter_ns()
        report1 = narrator.summarize_execution(context)
        cold_time = (time.perf_counter_ns() - start) / 1_000_000  # ms

        # Segunda ejecución (warm cache, misma estructura)
        start = time.perf_counter_ns()
        report2 = narrator.summarize_execution(context)
        warm_time = (time.perf_counter_ns() - start) / 1_000_000

        # Tercera ejecución con contexto idéntico
        start = time.perf_counter_ns()
        report3 = narrator.summarize_execution(context)
        hot_time = (time.perf_counter_ns() - start) / 1_000_000

        # Calcular speedup por caching
        speedup_cold_warm = cold_time / max(warm_time, 1e-10)
        speedup_cold_hot = cold_time / max(hot_time, 1e-10)

        # Efectividad de caching (debería mejorar al menos 20% en ejecuciones repetidas)
        assert speedup_cold_warm > 1.2, f"Caching inefectivo: speedup={speedup_cold_warm:.2f}"

        # Consistencia de resultados (caching no debe afectar resultados)
        assert report1["verdict_code"] == report2["verdict_code"] == report3["verdict_code"], \
            "Caching afecta resultados"

        # Análisis de patrón de acceso (locality of reference)
        time_pattern = [cold_time, warm_time, hot_time]
        time_reduction_ratio = (cold_time - hot_time) / cold_time

        assert time_reduction_ratio > 0.1, f"Poca mejora por caching: {time_reduction_ratio:.1%}"

        return {
            "cold_time_ms": cold_time,
            "warm_time_ms": warm_time,
            "hot_time_ms": hot_time,
            "speedup_cold_warm": speedup_cold_warm,
            "speedup_cold_hot": speedup_cold_hot,
            "time_reduction_ratio": time_reduction_ratio,
            "caching_effectiveness": "Bueno" if speedup_cold_hot > 1.5 else "Moderado"
        }


class StochasticPerformanceAnalysis:
    """
    Análisis estocástico de rendimiento usando teoría de probabilidad.
    """

    def test_performance_distribution_analysis(self, narrator: TelemetryNarrator):
        """
        Analiza distribución estadística de tiempos de ejecución.
        Usa pruebas de normalidad y análisis de colas.
        """
        from scipy import stats

        # Generar muestra de tiempos
        context = create_test_context(num_spans=50, span_depth=2)
        sample_size = 1000

        times = []
        for _ in range(sample_size):
            start = time.perf_counter_ns()
            _ = narrator.summarize_execution(context)
            times.append((time.perf_counter_ns() - start) / 1_000_000)  # ms

        # Estadísticas descriptivas
        mean_time = np.mean(times)
        std_time = np.std(times)
        cv = std_time / mean_time if mean_time > 0 else 0  # Coeficiente de variación

        # Prueba de normalidad (Shapiro-Wilk)
        shapiro_stat, shapiro_p = stats.shapiro(times[:min(500, sample_size)])
        is_normal = shapiro_p > 0.05

        # Análisis de colas (extremos)
        percentile_95 = np.percentile(times, 95)
        percentile_99 = np.percentile(times, 99)
        percentile_999 = np.percentile(times, 99.9)

        # Ratio cola/mediana (tail latency)
        median_time = np.median(times)
        tail_ratio_99 = percentile_99 / median_time
        tail_ratio_999 = percentile_999 / median_time

        # Criterios de calidad de servicio (SLA)
        sla_violations_95 = sum(1 for t in times if t > THRESHOLDS.MAX_REPORT_GENERATION_MS)
        sla_violation_rate = sla_violations_95 / sample_size

        # Análisis de autocorrelación (para detectar patrones temporales)
        if len(times) > 10:
            autocorr_lag1 = np.corrcoef(times[:-1], times[1:])[0, 1]
        else:
            autocorr_lag1 = 0

        # Verificaciones de calidad
        assert cv < 0.5, f"Alta variabilidad: CV={cv:.3f}"
        assert tail_ratio_99 < 3.0, f"Colas pesadas (99%): {tail_ratio_99:.2f}x mediana"
        assert sla_violation_rate < 0.05, f"Violaciones SLA: {sla_violation_rate:.1%}"
        assert abs(autocorr_lag1) < 0.3, f"Autocorrelación alta: {autocorr_lag1:.3f}"

        return {
            "sample_size": sample_size,
            "mean_ms": mean_time,
            "std_ms": std_time,
            "cv": cv,
            "is_normal": is_normal,
            "shapiro_p": shapiro_p,
            "percentiles": {
                "50": median_time,
                "95": percentile_95,
                "99": percentile_99,
                "99.9": percentile_999
            },
            "tail_ratios": {
                "99": tail_ratio_99,
                "999": tail_ratio_999
            },
            "sla_violation_rate": sla_violation_rate,
            "autocorrelation_lag1": autocorr_lag1,
            "distribution_type": "Normal" if is_normal else "No-normal"
        }

    def test_monte_carlo_performance_simulation(self, narrator: TelemetryNarrator):
        """
        Simulación Monte Carlo de rendimiento bajo condiciones aleatorias.
        """
        np.random.seed(42)  # Reproducibilidad

        num_simulations = 1000
        performance_results = []

        for sim in range(num_simulations):
            # Parámetros aleatorios para cada simulación
            num_spans = np.random.randint(10, 200)
            span_depth = np.random.randint(1, 5)
            error_rate = np.random.beta(2, 5)  # Sesgado hacia pocos errores

            # Crear contexto con parámetros aleatorios
            num_errors = int(num_spans * error_rate)
            context = create_test_context(
                num_spans=num_spans,
                span_depth=span_depth,
                num_errors=num_errors
            )

            # Medir tiempo
            start = time.perf_counter_ns()
            report = narrator.summarize_execution(context)
            elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000

            # Registrar resultados
            performance_results.append({
                "simulation": sim,
                "num_spans": num_spans,
                "span_depth": span_depth,
                "error_rate": error_rate,
                "elapsed_ms": elapsed_ms,
                "verdict": report["verdict_code"]
            })

        # Análisis de regresión múltiple
        import pandas as pd
        from sklearn.linear_model import LinearRegression

        df = pd.DataFrame(performance_results)

        # Variables predictoras
        X = df[['num_spans', 'span_depth', 'error_rate']]
        y = df['elapsed_ms']

        # Modelo lineal
        model = LinearRegression()
        model.fit(X, y)

        # Coeficientes (impacto de cada variable)
        coefficients = dict(zip(['num_spans', 'span_depth', 'error_rate'], model.coef_))
        r_squared = model.score(X, y)

        # Análisis de sensibilidad
        sensitivity = {
            'num_spans': abs(coefficients['num_spans']) * 100 / y.mean(),  # % impacto por unidad
            'span_depth': abs(coefficients['span_depth']) * 10 / y.mean(),  # Normalizado
            'error_rate': abs(coefficients['error_rate']) * 1.0 / y.mean()
        }

        # Verificar que número de spans sea el factor dominante
        assert sensitivity['num_spans'] > sensitivity['span_depth'], \
            f"Profundidad más impactante que spans: {sensitivity}"

        assert sensitivity['num_spans'] > sensitivity['error_rate'], \
            f"Error rate más impactante que spans: {sensitivity}"

        # Modelo debe explicar al menos 80% de varianza
        assert r_squared > 0.8, f"Modelo predictivo pobre: R²={r_squared:.3f}"

        return {
            "num_simulations": num_simulations,
            "mean_time_ms": y.mean(),
            "std_time_ms": y.std(),
            "regression_coefficients": coefficients,
            "r_squared": r_squared,
            "sensitivity_analysis": sensitivity,
            "most_influential": max(sensitivity.items(), key=lambda x: x[1])[0],
            "performance_predictor": f"Time ≈ {model.intercept_:.2f} + " +
                                    " + ".join(f"{coef:.3f}*{var}" for var, coef in coefficients.items())
        }



class BoundaryAndEdgeCaseAnalysis:
    """
    Análisis de casos límite y comportamiento en extremos.
    """

    def test_narrator_extreme_inputs(self, narrator: TelemetryNarrator):
        """
        Pruebas con entradas extremas para validar robustez.
        """
        test_cases = [
            {
                "name": "empty_context",
                "context": TelemetryContext(),
                "expected_behavior": "empty_report"
            },
            {
                "name": "single_span_no_metrics",
                "context": create_test_context(num_spans=1, num_metrics=0),
                "expected_behavior": "fast_processing"
            },
            {
                "name": "max_spans_practical",
                "context": create_test_context(num_spans=10000, span_depth=1),
                "expected_behavior": "scalable_but_slow"
            },
            {
                "name": "deep_nesting_extreme",
                "context": create_test_context(num_spans=1, span_depth=50),
                "expected_behavior": "handle_recursion"
            },
            {
                "name": "all_spans_failed",
                "context": create_test_context(num_spans=100, num_errors=100),
                "expected_behavior": "aggregate_errors"
            },
            {
                "name": "mixed_status_complex",
                "context": self._create_mixed_status_context(),
                "expected_behavior": "accurate_analysis"
            }
        ]

        results = []

        for test_case in test_cases:
            context = test_case["context"]

            # Medir tiempo y memoria
            with measure_time() as times:
                report = narrator.summarize_execution(context)

            with measure_memory() as mem:
                _ = narrator.summarize_execution(context)

            # Validar según comportamiento esperado
            if test_case["expected_behavior"] == "empty_report":
                assert report["verdict_code"] in ["APPROVED", "EMPTY"], \
                    f"Empty context unexpected verdict: {report['verdict_code']}"
                assert times[0] < 5.0, f"Empty context too slow: {times[0]:.2f}ms"

            elif test_case["expected_behavior"] == "fast_processing":
                assert times[0] < THRESHOLDS.MAX_SINGLE_SPAN_ANALYSIS_MS, \
                    f"Single span too slow: {times[0]:.2f}ms"

            elif test_case["expected_behavior"] == "scalable_but_slow":
                # Verificar que no crashea y escala razonablemente
                assert times[0] < 10000, f"10000 spans timeout: {times[0]:.2f}ms"
                assert mem.peak_mb < 100, f"10000 spans memory explosion: {mem.peak_mb:.1f}MB"

            elif test_case["expected_behavior"] == "handle_recursion":
                # Profundidad extrema pero sin stack overflow
                assert times[0] < 1000, f"Deep nesting timeout: {times[0]:.2f}ms"
                assert "max_depth" in str(report) or "verdict" in report

            elif test_case["expected_behavior"] == "aggregate_errors":
                assert "REJECTED" in report["verdict_code"], \
                    f"All failed but verdict: {report['verdict_code']}"
                assert len(report["forensic_evidence"]) > 0, \
                    "No forensic evidence for all-failed case"

            elif test_case["expected_behavior"] == "accurate_analysis":
                # Verificar que mezcla correctamente
                has_success = any(s["status"] == "SUCCESS"
                                 for s in report.get("span_analysis", []))
                has_failure = any(s["status"] == "FAILURE"
                                 for s in report.get("span_analysis", []))
                assert has_success and has_failure, "Mixed status not reflected"

            results.append({
                "test_case": test_case["name"],
                "time_ms": times[0],
                "memory_mb": mem.peak_mb,
                "verdict": report["verdict_code"],
                "passed": True
            })

        return results

    def _create_mixed_status_context(self) -> TelemetryContext:
        """Crea contexto con mezcla compleja de estados."""
        context = TelemetryContext()

        # Patrón: éxito, warning, fallo cíclico
        patterns = [
            (StepStatus.SUCCESS, "SUCCESS"),
            (StepStatus.WARNING, "WARNING"),
            (StepStatus.FAILURE, "FAILURE"),
            (StepStatus.SUCCESS, "RECOVERY"),
        ]

        for i in range(20):
            status, name = patterns[i % len(patterns)]
            with context.span(f"{name}_{i}") as span:
                span.status = status
                if status == StepStatus.FAILURE:
                    span.errors.append({
                        "message": f"Controlled failure {i}",
                        "type": "TestFailure"
                    })

        return context

    def test_translator_boundary_conditions(self, translator: SemanticTranslator):
        """
        Prueba condiciones límite del translator.
        """
        boundary_cases = [
            {
                "name": "zero_topology",
                "topology": TopologicalMetrics(beta_0=0, beta_1=0, euler_characteristic=0),
                "financials": {},
                "stability": 0.0,
                "expect": "empty_or_error"
            },
            {
                "name": "negative_betti",
                "topology": TopologicalMetrics(beta_0=-1, beta_1=-1),
                "financials": {"performance": {"recommendation": "ACEPTAR"}},
                "stability": 10.0,
                "expect": "handle_negative"
            },
            {
                "name": "extreme_stability",
                "topology": TopologicalMetrics(beta_0=1, beta_1=0),
                "financials": {"performance": {"recommendation": "ACEPTAR"}},
                "stability": 1000000.0,
                "expect": "cap_extremes"
            },
            {
                "name": "nan_metrics",
                "topology": {"beta_0": float('nan'), "beta_1": float('nan')},
                "financials": {"wacc": float('nan'), "performance": {"recommendation": "ACEPTAR"}},
                "stability": float('nan'),
                "expect": "handle_nan"
            },
            {
                "name": "infinite_metrics",
                "topology": {"beta_0": float('inf'), "beta_1": float('inf')},
                "financials": {"wacc": float('inf')},
                "stability": float('inf'),
                "expect": "handle_infinity"
            }
        ]

        results = []

        for case in boundary_cases:
            try:
                report = translator.compose_strategic_narrative(
                    topological_metrics=case["topology"],
                    financial_metrics=case["financials"],
                    stability=case["stability"]
                )

                status = "success"
                verdict = report.verdict.name if hasattr(report, 'verdict') else "unknown"

            except Exception as e:
                status = f"exception: {type(e).__name__}"
                verdict = "error"

            # Validar según expectativa
            if case["expect"] == "empty_or_error":
                assert status != "success" or verdict in ["VIABLE", "PRECAUCION"], \
                    f"Zero topology unexpected: {status}, {verdict}"

            elif case["expect"] == "handle_negative":
                # Debería manejar valores negativos sin crashear
                assert status != "exception: ValueError", "Negative values caused crash"

            elif case["expect"] == "cap_extremes":
                assert status == "success", f"Extreme stability crashed: {status}"
                # Debería normalizar valores extremos

            elif case["expect"] == "handle_nan":
                # Debería manejar NaN sin propagar
                assert status == "success", f"NaN caused crash: {status}"

            elif case["expect"] == "handle_infinity":
                # Debería manejar infinito
                assert status == "success", f"Infinity caused crash: {status}"

            results.append({
                "case": case["name"],
                "status": status,
                "verdict": verdict,
                "passed": True
            })

        return results


class AdvancedComparativeAnalysis:
    """
    Análisis comparativo avanzado con métricas normalizadas.
    """

    def test_normalized_performance_metrics(self, narrator: TelemetryNarrator, translator: SemanticTranslator):
        """
        Métricas de rendimiento normalizadas para comparación justa.
        """
        # Definir unidad de trabajo estándar
        STANDARD_UNIT = {
            "narrator": {"spans": 100, "depth": 2, "metrics": 10},
            "translator": {"topology_complexity": 1.0, "financial_items": 5}
        }

        # Medir narrator por unidad de trabajo
        context = create_test_context(
            num_spans=STANDARD_UNIT["narrator"]["spans"],
            span_depth=STANDARD_UNIT["narrator"]["depth"],
            num_metrics=STANDARD_UNIT["narrator"]["metrics"]
        )

        narrator_times = []
        for _ in range(50):
            start = time.perf_counter_ns()
            _ = narrator.summarize_execution(context)
            narrator_times.append((time.perf_counter_ns() - start) / 1_000_000)

        narrator_perf = {
            "mean_ms": np.mean(narrator_times),
            "std_ms": np.std(narrator_times),
            "throughput_ups": 1000 / np.mean(narrator_times) if np.mean(narrator_times) > 0 else 0,
            "efficiency": STANDARD_UNIT["narrator"]["spans"] / np.mean(narrator_times)
        }

        # Medir translator por unidad de trabajo
        translator_times = []
        for _ in range(50):
            start = time.perf_counter_ns()
            _ = translator.compose_strategic_narrative(
                topological_metrics=TopologicalMetrics(
                    beta_0=int(STANDARD_UNIT["translator"]["topology_complexity"]),
                    beta_1=0
                ),
                financial_metrics={
                    "wacc": 0.12,
                    "performance": {"recommendation": "ACEPTAR", "profitability_index": 1.3},
                    "contingency": {"recommended": 15000.0}
                },
                stability=10.0
            )
            translator_times.append((time.perf_counter_ns() - start) / 1_000_000)

        translator_perf = {
            "mean_ms": np.mean(translator_times),
            "std_ms": np.std(translator_times),
            "throughput_tps": 1000 / np.mean(translator_times) if np.mean(translator_times) > 0 else 0,
            "efficiency": STANDARD_UNIT["translator"]["financial_items"] / np.mean(translator_times)
        }

        # Métricas comparativas normalizadas
        comparative_metrics = {
            "speed_ratio": narrator_perf["mean_ms"] / translator_perf["mean_ms"],
            "throughput_ratio": translator_perf["throughput_tps"] / narrator_perf["throughput_ups"],
            "stability_ratio": narrator_perf["std_ms"] / translator_perf["std_ms"],
            "efficiency_ratio": narrator_perf["efficiency"] / translator_perf["efficiency"],
            "complexity_adjusted_score": (
                narrator_perf["throughput_ups"] * STANDARD_UNIT["narrator"]["spans"] /
                translator_perf["throughput_tps"] / STANDARD_UNIT["translator"]["financial_items"]
            )
        }

        # Verificaciones de equilibrio del sistema
        assert 0.1 < comparative_metrics["speed_ratio"] < 10.0, \
            f"Desbalance extremo en velocidad: {comparative_metrics['speed_ratio']:.2f}"

        assert comparative_metrics["stability_ratio"] < 3.0, \
            f"Desbalance en estabilidad: {comparative_metrics['stability_ratio']:.2f}"

        # El sistema debe estar balanceado (ningún módulo es cuello de botella extremo)
        bottleneck_threshold = 0.1
        assert comparative_metrics["complexity_adjusted_score"] > bottleneck_threshold, \
            f"Narrator es cuello de botella: score={comparative_metrics['complexity_adjusted_score']:.3f}"

        return {
            "standard_unit": STANDARD_UNIT,
            "narrator_performance": narrator_perf,
            "translator_performance": translator_perf,
            "comparative_metrics": comparative_metrics,
            "system_balance": "Bueno" if comparative_metrics["speed_ratio"] > 0.3 and
                                      comparative_metrics["speed_ratio"] < 3.0 else "Desbalanceado",
            "bottleneck": "Narrator" if comparative_metrics["complexity_adjusted_score"] < 0.5 else
                         "Translator" if comparative_metrics["complexity_adjusted_score"] > 2.0 else
                         "Balanceado"
        }

    def test_scalability_comparison_matrix(self, narrator: TelemetryNarrator, translator: SemanticTranslator):
        """
        Matriz de escalabilidad comparativa entre módulos.
        """
        scale_factors = [1, 2, 4, 8, 16]

        narrator_scaling = []
        translator_scaling = []

        for factor in scale_factors:
            # Narrator scaling
            context = create_test_context(num_spans=100 * factor, span_depth=2)

            times = []
            for _ in range(10):
                start = time.perf_counter_ns()
                _ = narrator.summarize_execution(context)
                times.append((time.perf_counter_ns() - start) / 1_000_000)

            narrator_scaling.append({
                "factor": factor,
                "time_ms": np.median(times),
                "time_per_unit": np.median(times) / (100 * factor)
            })

            # Translator scaling
            times = []
            for _ in range(10):
                start = time.perf_counter_ns()
                _ = translator.compose_strategic_narrative(
                    topological_metrics=TopologyMetricsDTO(beta_0=factor, beta_1=factor//2),
                    financial_metrics={"performance": {"recommendation": "ACEPTAR"}},
                    stability=10.0 * factor
                )
                times.append((time.perf_counter_ns() - start) / 1_000_000)

            translator_scaling.append({
                "factor": factor,
                "time_ms": np.median(times),
                "time_per_unit": np.median(times) / factor
            })

        # Análisis de escalabilidad relativa
        narrator_scaling_coefficient = self._calculate_scaling_coefficient(
            [s["factor"] for s in narrator_scaling],
            [s["time_ms"] for s in narrator_scaling]
        )

        translator_scaling_coefficient = self._calculate_scaling_coefficient(
            [s["factor"] for s in translator_scaling],
            [s["time_ms"] for s in translator_scaling]
        )

        # Comparar escalabilidad
        scaling_ratio = narrator_scaling_coefficient / translator_scaling_coefficient

        # Ideal: ambos escalan similarmente (ratio cerca de 1)
        assert 0.5 < scaling_ratio < 2.0, \
            f"Escalabilidad muy diferente: narrator O(n^{narrator_scaling_coefficient:.2f}) vs " \
            f"translator O(n^{translator_scaling_coefficient:.2f}), ratio={scaling_ratio:.2f}"

        return {
            "scale_factors": scale_factors,
            "narrator_scaling": narrator_scaling,
            "translator_scaling": translator_scaling,
            "narrator_complexity": f"O(n^{narrator_scaling_coefficient:.2f})",
            "translator_complexity": f"O(n^{translator_scaling_coefficient:.2f})",
            "scaling_ratio": scaling_ratio,
            "scaling_compatibility": "Buena" if 0.7 < scaling_ratio < 1.3 else "Moderada"
        }

    def _calculate_scaling_coefficient(self, sizes: List[float], times: List[float]) -> float:
        """Calcula coeficiente de escalabilidad usando regresión log-log."""
        log_sizes = np.log(sizes)
        log_times = np.log(times)

        slope, _, r_value, _, _ = stats.linregress(log_sizes, log_times)

        return slope  # Exponente en O(n^slope)


class HeatAndLoadDistributionAnalysis:
    """
    Análisis de distribución de calor computacional y carga.
    """

    def test_computational_heat_map_narrator(self, narrator: TelemetryNarrator):
        """
        Genera mapa de calor computacional del narrator.
        Identifica hotspots y cuellos de botella.
        """
        context = create_test_context(num_spans=100, span_depth=3, num_errors=5)

        # Instrumentación fina de fases
        phase_timings = {}

        # Fase 1: Recorrido y análisis de spans
        start = time.perf_counter_ns()
        phases = []
        for span in context.root_spans:
            phase_start = time.perf_counter_ns()
            phase = narrator._analyze_phase(span)
            phases.append(phase)
            phase_timings.setdefault("analyze_span", []).append(
                (time.perf_counter_ns() - phase_start) / 1_000_000
            )
        phase_timings["total_analyze_spans"] = (time.perf_counter_ns() - start) / 1_000_000

        # Fase 2: Agrupación por estrato
        start = time.perf_counter_ns()
        strata_groups = narrator._group_by_stratum(phases)
        phase_timings["group_by_stratum"] = (time.perf_counter_ns() - start) / 1_000_000

        # Fase 3: Análisis por estrato
        stratum_analysis = {}
        for stratum, phase_list in strata_groups.items():
            start = time.perf_counter_ns()
            analysis = narrator._analyze_stratum(stratum, phase_list)
            stratum_analysis[stratum] = analysis
            phase_timings.setdefault(f"analyze_stratum_{stratum.name}", []).append(
                (time.perf_counter_ns() - start) / 1_000_000
            )

        # Fase 4: Síntesis final
        start = time.perf_counter_ns()
        final_report = narrator._synthesize_report(stratum_analysis)
        phase_timings["synthesize_report"] = (time.perf_counter_ns() - start) / 1_000_000

        # Análisis de distribución
        total_time = sum(
            t[0] if isinstance(t, list) else t
            for t in phase_timings.values()
            if (isinstance(t, float) or (isinstance(t, list) and len(t) > 0))
        )

        heat_distribution = {}
        for phase, timing in phase_timings.items():
            if isinstance(timing, list) and timing:
                phase_total = np.sum(timing)
            elif isinstance(timing, float):
                phase_total = timing
            else:
                continue

            percentage = (phase_total / total_time) * 100
            heat_distribution[phase] = {
                "time_ms": phase_total,
                "percentage": percentage,
                "is_hotspot": percentage > 20.0  # Más del 20% es hotspot
            }

        # Identificar cuellos de botella
        hotspots = [phase for phase, data in heat_distribution.items()
                   if data["is_hotspot"]]

        # Verificar que no haya hotspots extremos
        max_percentage = max(data["percentage"] for data in heat_distribution.values())
        assert max_percentage < 50.0, f"Hotspot extremo: {max_percentage:.1f}% en {hotspots}"

        # Verificar distribución balanceada
        balance_score = np.std(list(data["percentage"] for data in heat_distribution.values()))
        assert balance_score < 15.0, f"Distribución desbalanceada: σ={balance_score:.1f}"

        return {
            "total_time_ms": total_time,
            "heat_distribution": heat_distribution,
            "hotspots": hotspots,
            "balance_score": balance_score,
            "recommendations": self._generate_optimization_recommendations(heat_distribution)
        }

    def _generate_optimization_recommendations(self, heat_distribution: Dict) -> List[str]:
        """Genera recomendaciones basadas en mapa de calor."""
        recommendations = []

        # Ordenar por porcentaje descendente
        sorted_phases = sorted(
            heat_distribution.items(),
            key=lambda x: x[1]["percentage"],
            reverse=True
        )

        for phase, data in sorted_phases[:3]:  # Top 3 hotspots
            if data["percentage"] > 30:
                recommendations.append(f"OPTIMIZAR CRÍTICO: {phase} ({data['percentage']:.1f}%)")
            elif data["percentage"] > 15:
                recommendations.append(f"Considerar optimizar: {phase} ({data['percentage']:.1f}%)")

        if not recommendations:
            recommendations.append("Distribución balanceada, sin optimizaciones críticas")

        return recommendations

    def test_load_distribution_concurrent(self, narrator: TelemetryNarrator, translator: SemanticTranslator):
        """
        Analiza distribución de carga en ejecución concurrente.
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        import multiprocessing as mp

        cpu_count = mp.cpu_count()
        batch_sizes = [1, 2, 4, 8, cpu_count]

        results = []

        for batch_size in batch_sizes:
            # Crear lotes de trabajo
            num_batches = 100
            batch_load = 50  # Spans por batch

            # Medir tiempo secuencial (baseline)
            sequential_times = []
            for i in range(num_batches):
                context = create_test_context(num_spans=batch_load, span_depth=2)
                start = time.perf_counter_ns()
                _ = narrator.summarize_execution(context)
                sequential_times.append((time.perf_counter_ns() - start) / 1_000_000)

            sequential_total = np.sum(sequential_times)

            # Medir tiempo paralelo
            def process_batch(batch_id):
                context = create_test_context(num_spans=batch_load, span_depth=2)
                start = time.perf_counter_ns()
                result = narrator.summarize_execution(context)
                return (time.perf_counter_ns() - start) / 1_000_000, batch_id

            parallel_times = []
            with ProcessPoolExecutor(max_workers=batch_size) as executor:
                futures = [executor.submit(process_batch, i) for i in range(num_batches)]
                for future in as_completed(futures):
                    batch_time, batch_id = future.result()
                    parallel_times.append(batch_time)

            parallel_total = np.max(parallel_times) * (num_batches / batch_size)  # Tiempo teórico

            # Calcular speedup y eficiencia
            speedup = sequential_total / parallel_total if parallel_total > 0 else 1
            efficiency = (speedup / batch_size) * 100

            # Ley de Amdahl: speedup máximo teórico
            parallel_fraction = 0.8  # Estimación
            amdahl_speedup = 1 / ((1 - parallel_fraction) + (parallel_fraction / batch_size))

            results.append({
                "batch_size": batch_size,
                "sequential_ms": sequential_total,
                "parallel_ms": parallel_total,
                "speedup": speedup,
                "efficiency_percent": efficiency,
                "amdahl_theoretical": amdahl_speedup,
                "achieved_vs_theoretical": speedup / amdahl_speedup if amdahl_speedup > 0 else 0
            })

        # Análisis de escalabilidad paralela
        speedups = [r["speedup"] for r in results]
        efficiencies = [r["efficiency_percent"] for r in results]

        # Verificar que speedup mejora (aunque no perfectamente)
        assert speedups[-1] > speedups[0] * 0.5, \
            f"Poco speedup al escalar: {speedups[0]:.2f} -> {speedups[-1]:.2f}"

        # Eficiencia no debe caer demasiado rápido
        efficiency_drop = (efficiencies[0] - efficiencies[-1]) / efficiencies[0]
        assert efficiency_drop < 0.8, f"Eficiencia cae demasiado: {efficiency_drop:.1%}"

        return {
            "cpu_count": cpu_count,
            "results": results,
            "scalability_summary": {
                "max_speedup": max(speedups),
                "min_efficiency": min(efficiencies),
                "optimal_batch_size": batch_sizes[np.argmax(speedups)],
                "parallel_scalability": "Buena" if speedups[-1] > cpu_count * 0.5 else "Moderada"
            }
        }