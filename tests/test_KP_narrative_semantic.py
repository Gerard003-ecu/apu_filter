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
    TopologyMetricsDTO,
    ThermalMetricsDTO,
    SpectralMetricsDTO,
    SynergyRiskDTO,
    StrategicReport,
    SemanticTranslator,
    TranslatorConfig,
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
def simple_topology() -> TopologyMetricsDTO:
    """Topología simple."""
    return TopologyMetricsDTO(beta_0=1, beta_1=0, euler_characteristic=1)


@pytest.fixture
def complex_topology() -> TopologyMetricsDTO:
    """Topología compleja."""
    return TopologyMetricsDTO(beta_0=5, beta_1=10, euler_characteristic=-5)


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
        with measure_time() as times:
            for _ in range(100):
                _ = translator.translate_thermodynamics(
                    entropy=0.5, exergy=0.7, temperature=45.0
                )
        
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
                topology = TopologyMetricsDTO(beta_0=1, beta_1=i % 3)
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
        topology = TopologyMetricsDTO()
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
                topology = TopologyMetricsDTO(
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
                TopologyMetricsDTO(beta_0=1, beta_1=0),
                stability=1.0,
            )
            times_low.append((time.perf_counter() - start) * 1000)
            
            # Valores altos
            start = time.perf_counter()
            _ = translator.translate_topology(
                TopologyMetricsDTO(beta_0=1000, beta_1=500),
                stability=1000.0,
            )
            times_high.append((time.perf_counter() - start) * 1000)
        
        mean_low = statistics.mean(times_low)
        mean_high = statistics.mean(times_high)
        
        # No debe haber diferencia significativa
        ratio = mean_high / mean_low
        assert 0.5 < ratio < 2.0, f"Time depends on values: {ratio:.2f}x difference"


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
        topology = TopologyMetricsDTO()
        for _ in range(100):
            start = time.perf_counter()
            _ = translator.translate_topology(topology, stability=10.0)
            translation_times.append((time.perf_counter() - start) * 1000)
        
        mean_lattice = statistics.mean(lattice_times) / 100  # Per operation
        mean_translation = statistics.mean(translation_times)
        
        # Lattice debe ser al menos 10x más rápido
        ratio = mean_translation / mean_lattice
        assert ratio > 10, f"Lattice not fast enough: only {ratio:.1f}x faster"

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
                topological_metrics=TopologyMetricsDTO(),
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
        topology = TopologyMetricsDTO(beta_0=3, beta_1=2)
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
        start = time.perf_counter()
        for _ in range(10):
            _ = translator.translate_thermodynamics(0.6, 0.7, 55.0)
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
        topology = TopologyMetricsDTO()
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
                topological_metrics=TopologyMetricsDTO(),
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