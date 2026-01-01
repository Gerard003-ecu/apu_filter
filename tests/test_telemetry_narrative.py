"""
Pruebas para el Narrador de Telemetría Híbrido.
Verifica la integración de la lógica algebraica (Lattice) con la estructura DIKW.
"""
import pytest
from app.telemetry import TelemetryContext, StepStatus, TelemetrySpan
from app.telemetry_narrative import TelemetryNarrator, SeverityLevel

@pytest.fixture
def narrator():
    return TelemetryNarrator()

@pytest.fixture
def context():
    return TelemetryContext()

def test_lattice_logic_supremum():
    """Verifica que el supremo del lattice funcione correctamente."""
    # OPTIMO U OPTIMO = OPTIMO
    assert SeverityLevel.supremum(SeverityLevel.OPTIMO, SeverityLevel.OPTIMO) == SeverityLevel.OPTIMO

    # OPTIMO U ADVERTENCIA = ADVERTENCIA
    assert SeverityLevel.supremum(SeverityLevel.OPTIMO, SeverityLevel.ADVERTENCIA) == SeverityLevel.ADVERTENCIA

    # ADVERTENCIA U CRITICO = CRITICO
    assert SeverityLevel.supremum(SeverityLevel.ADVERTENCIA, SeverityLevel.CRITICO) == SeverityLevel.CRITICO

    # Conjunto vacío = OPTIMO
    assert SeverityLevel.supremum() == SeverityLevel.OPTIMO

def test_empty_context_report(narrator, context):
    """Verifica el reporte para un contexto vacío."""
    report = narrator.summarize_execution(context)
    assert report["verdict"] == "OPTIMO"
    assert "Sin telemetría" in report["narrative"]
    assert report["phases"] == []

def test_legacy_context_report(narrator, context):
    """Verifica el fallback para contextos sin spans jerárquicos."""
    context.start_step("legacy_step")
    context.end_step("legacy_step", StepStatus.SUCCESS)

    report = narrator.summarize_execution(context)
    assert report["verdict"] == "OPTIMO"
    assert "Legacy" in report["narrative"]

    # Con error
    context.record_error("legacy_step", "Something failed")
    report_err = narrator.summarize_execution(context)
    assert report_err["verdict"] == "CRITICO"
    assert len(report_err["forensic_evidence"]) > 0

def test_hierarchical_success(narrator, context):
    """Verifica un árbol de spans exitoso."""
    with context.span("Phase 1"):
        with context.span("Operation A"):
            pass
        with context.span("Operation B"):
            pass

    report = narrator.summarize_execution(context)
    assert report["verdict"] == "OPTIMO"
    assert len(report["phases"]) == 1
    assert report["phases"][0]["status"] == "OPTIMO"
    # Corregido: "óptima" en lugar de "exitosa"
    assert "óptima" in report["narrative"]

def test_hierarchical_warning(narrator, context):
    """Verifica la detección de warnings."""
    with context.span("Phase 1"):
        with context.span("Subtask"):
             pass

    # Manipulamos el estado DESPUÉS del bloque context manager
    # porque el context manager fuerza SUCCESS si no hay excepción.
    context.root_spans[0].status = StepStatus.WARNING

    report = narrator.summarize_execution(context)
    assert report["verdict"] == "ADVERTENCIA"
    assert report["phases"][0]["status"] == "ADVERTENCIA"
    assert "advertencias" in report["narrative"]

def test_hierarchical_failure_with_topology(narrator, context):
    """Verifica fallos críticos y la ruta topológica."""
    try:
        with context.span("Root Phase"):
            with context.span("Level 1"):
                with context.span("Level 2"):
                    raise ValueError("Deep Error")
    except ValueError:
        pass

    report = narrator.summarize_execution(context)

    assert report["verdict"] == "CRITICO"
    assert "FALLO CRÍTICO" in report["narrative"]

    # Buscamos la evidencia que corresponde al nodo hoja (mayor profundidad)
    # telemetry.py propaga la excepción, así que el error aparece en todos los niveles.
    # El narrador ordena por (depth, source), así que el error raíz (depth=0) sale primero.
    # Buscamos si ALGUNA evidencia tiene la ruta completa.

    found_full_path = False
    for issue in report["forensic_evidence"]:
        if "Root Phase → Level 1 → Level 2" in issue["topological_path"]:
            found_full_path = True
            assert issue["message"] == "Deep Error"
            break

    assert found_full_path, "No se encontró el issue con la ruta topológica completa"

def test_silent_failure_detection(narrator, context):
    """Verifica detección de fallos silenciosos (Status FAILURE sin error explícito)."""
    with context.span("Silent Phase"):
        pass

    # Forzamos FAILURE después de salir del bloque
    context.root_spans[0].status = StepStatus.FAILURE

    report = narrator.summarize_execution(context)
    assert report["verdict"] == "CRITICO"

    # Debería generar un issue de SilentFailure
    evidence = report["forensic_evidence"][0]
    assert evidence["type"] == "SilentFailure"
    assert "Silent Phase" in evidence["topological_path"]

def test_lattice_induced_severity(narrator, context):
    """
    Verifica que un issue crítico en un hijo induzca severidad CRITICA
    en el padre aunque el padre diga SUCCESS.
    """
    # Escenario: El span padre termina 'bien' (no lanza excepción hasta arriba),
    # pero un hijo registró un error (o lo inyectamos manualmente).
    with context.span("Parent") as parent:
        with context.span("Child") as child:
             # Simulamos que child falló pero no propagó excepción (catch interno)
             child.status = StepStatus.FAILURE
             child.errors.append({"message": "Child failed", "type": "Error"})
        # Parent sigue en SUCCESS

    report = narrator.summarize_execution(context)

    # El padre 'Parent' tiene un hijo fallido.
    # El análisis de fase recolecta todos los issues del subárbol.
    # Si hay issues críticos, la fase se marca CRITICA.
    phase = report["phases"][0]
    assert phase["status"] == "CRITICO"
    assert report["verdict"] == "CRITICO"
