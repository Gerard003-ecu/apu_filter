"""
Pruebas para el Narrador de Telemetría Híbrido.
Verifica la integración de la lógica algebraica (Lattice) con la estructura DIKW.
"""

from itertools import permutations, product

import pytest

from app.telemetry import StepStatus, TelemetryContext
from app.telemetry_narrative import SeverityLevel, TelemetryNarrator


@pytest.fixture
def narrator():
    """Fixture: instancia limpia del narrador."""
    return TelemetryNarrator()


@pytest.fixture
def context():
    """Fixture: contexto de telemetría aislado."""
    return TelemetryContext()


@pytest.fixture
def all_severity_levels():
    """Fixture: conjunto completo del lattice de severidades."""
    return [SeverityLevel.OPTIMO, SeverityLevel.ADVERTENCIA, SeverityLevel.CRITICO]


def test_lattice_logic_supremum_idempotency(all_severity_levels):
    """
    Propiedad de Idempotencia: ∀a ∈ L: a ∨ a = a
    Fundamental en cualquier semilattice.
    """
    for level in all_severity_levels:
        assert SeverityLevel.supremum(level, level) == level, (
            f"Violación de idempotencia para {level}"
        )


def test_lattice_logic_supremum_commutativity(all_severity_levels):
    """
    Propiedad de Conmutatividad: ∀a,b ∈ L: a ∨ b = b ∨ a
    El supremo es simétrico respecto a sus argumentos.
    """
    for a, b in permutations(all_severity_levels, 2):
        assert SeverityLevel.supremum(a, b) == SeverityLevel.supremum(b, a), (
            f"Violación de conmutatividad para ({a}, {b})"
        )


def test_lattice_logic_supremum_associativity(all_severity_levels):
    """
    Propiedad de Asociatividad: ∀a,b,c ∈ L: (a ∨ b) ∨ c = a ∨ (b ∨ c)
    Garantiza consistencia en operaciones n-arias.
    """
    for a, b, c in product(all_severity_levels, repeat=3):
        left_assoc = SeverityLevel.supremum(SeverityLevel.supremum(a, b), c)
        right_assoc = SeverityLevel.supremum(a, SeverityLevel.supremum(b, c))
        assert left_assoc == right_assoc, f"Violación de asociatividad para ({a}, {b}, {c})"


def test_lattice_logic_identity_and_absorbing_elements(all_severity_levels):
    """
    Verifica elementos especiales del lattice:
    - Identidad (⊥): OPTIMO - elemento bottom, sup(a, ⊥) = a
    - Absorbente (⊤): CRITICO - elemento top, sup(a, ⊤) = ⊤
    - Conjunto vacío: retorna el elemento identidad
    """
    bottom = SeverityLevel.OPTIMO
    top = SeverityLevel.CRITICO

    # Conjunto vacío retorna identidad (bottom)
    assert SeverityLevel.supremum() == bottom, (
        "sup(∅) debe ser el elemento identidad (OPTIMO)"
    )

    for level in all_severity_levels:
        # Propiedad de identidad: a ∨ ⊥ = a
        assert SeverityLevel.supremum(level, bottom) == level, (
            f"OPTIMO no actúa como identidad para {level}"
        )

        # Propiedad de absorción: a ∨ ⊤ = ⊤
        assert SeverityLevel.supremum(level, top) == top, (
            f"CRITICO no actúa como absorbente para {level}"
        )


def test_lattice_partial_order_transitivity():
    """
    Transitividad del orden inducido: a ≤ b ∧ b ≤ c ⟹ a ≤ c
    Donde a ≤ b ⟺ a ∨ b = b (definición via supremum).

    Cadena total esperada: OPTIMO ≤ ADVERTENCIA ≤ CRITICO
    """
    O, A, C = SeverityLevel.OPTIMO, SeverityLevel.ADVERTENCIA, SeverityLevel.CRITICO

    # O ≤ A: O ∨ A = A
    assert SeverityLevel.supremum(O, A) == A
    # A ≤ C: A ∨ C = C
    assert SeverityLevel.supremum(A, C) == C
    # Por transitividad: O ≤ C
    assert SeverityLevel.supremum(O, C) == C


def test_lattice_n_ary_supremum_consistency(all_severity_levels):
    """
    Consistencia del supremo n-ario con composición binaria.
    sup(a₁, a₂, ..., aₙ) = sup(sup(...sup(a₁, a₂), a₃)..., aₙ)
    """
    # El supremo de todo el lattice debe ser el top
    assert SeverityLevel.supremum(*all_severity_levels) == SeverityLevel.CRITICO

    # Verificar consistencia con reducción binaria
    binary_reduction = all_severity_levels[0]
    for level in all_severity_levels[1:]:
        binary_reduction = SeverityLevel.supremum(binary_reduction, level)

    assert SeverityLevel.supremum(*all_severity_levels) == binary_reduction


def test_empty_context_report(narrator, context):
    """
    Verifica el reporte para contexto vacío.
    Caso base: sin información → severidad mínima (bottom del lattice).
    """
    report = narrator.summarize_execution(context)

    assert report["verdict"] == "OPTIMO", "Contexto vacío debe mapear al bottom element"
    assert "Sin telemetría" in report["narrative"]
    assert report["phases"] == []
    assert report.get("forensic_evidence", []) == [], (
        "No debe existir evidencia forense en contexto vacío"
    )


def test_legacy_context_success_report(narrator):
    """Verifica el fallback para contextos legacy exitosos."""
    legacy_context = TelemetryContext()
    legacy_context.start_step("legacy_step")
    legacy_context.end_step("legacy_step", StepStatus.SUCCESS)

    report = narrator.summarize_execution(legacy_context)

    assert report["verdict"] == "OPTIMO"
    assert "Legacy" in report["narrative"]
    assert len(report.get("forensic_evidence", [])) == 0


def test_legacy_context_error_report(narrator):
    """
    Verifica el fallback para contextos legacy con error.
    Aislado del test de éxito para evitar efectos secundarios.
    """
    error_context = TelemetryContext()
    error_context.start_step("legacy_step")
    error_context.end_step("legacy_step", StepStatus.SUCCESS)
    error_context.record_error("legacy_step", "Something failed")

    report = narrator.summarize_execution(error_context)

    assert report["verdict"] == "CRITICO", (
        "Presencia de error debe elevar al top del lattice"
    )
    assert len(report["forensic_evidence"]) > 0, (
        "Debe existir evidencia forense documentando el error"
    )


def test_hierarchical_success(narrator, context):
    """
    Verifica árbol de spans completamente exitoso.
    sup(∅) sobre issues = OPTIMO (sin degradación de severidad).
    """
    with context.span("Phase 1"):
        with context.span("Operation A"):
            pass
        with context.span("Operation B"):
            pass

    report = narrator.summarize_execution(context)

    assert report["verdict"] == "OPTIMO"
    assert len(report["phases"]) == 1
    assert report["phases"][0]["status"] == "OPTIMO"
    assert "óptima" in report["narrative"].lower(), "Narrativa debe reflejar estado óptimo"


def test_hierarchical_warning(narrator, context):
    """
    Verifica propagación de warnings en la jerarquía.
    Warning representa nivel intermedio del lattice.
    """
    with context.span("Phase 1"):
        with context.span("Subtask"):
            pass

    # Inyección post-ejecución para simular warning detectado
    context.root_spans[0].status = StepStatus.WARNING

    report = narrator.summarize_execution(context)

    assert report["verdict"] == "ADVERTENCIA"
    assert report["phases"][0]["status"] == "ADVERTENCIA"
    assert "advertencias" in report["narrative"].lower()


def test_hierarchical_failure_with_topology(narrator, context):
    """
    Verifica fallos críticos y la ruta topológica del error.
    La ruta representa el path en el árbol de spans hasta el punto de fallo.
    """
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

    # Búsqueda eficiente de la ruta topológica completa
    expected_path = "Root Phase → Level 1 → Level 2"
    matching_evidence = next(
        (
            issue
            for issue in report["forensic_evidence"]
            if expected_path in issue["topological_path"]
            and issue["message"] == "Deep Error"
        ),
        None,
    )

    assert matching_evidence is not None, (
        f"No se encontró evidencia con ruta topológica: {expected_path}"
    )


def test_silent_failure_detection(narrator, context):
    """
    Verifica detección de fallos silenciosos.
    Fallo silencioso: Status FAILURE sin excepción/error explícito registrado.
    Importante para detectar estados inconsistentes en el sistema.
    """
    with context.span("Silent Phase"):
        pass

    # Simular fallo sin error explícito
    context.root_spans[0].status = StepStatus.FAILURE

    report = narrator.summarize_execution(context)

    assert report["verdict"] == "CRITICO"
    assert len(report["forensic_evidence"]) > 0

    silent_evidence = report["forensic_evidence"][0]
    assert silent_evidence["type"] == "SilentFailure", (
        "Tipo de evidencia debe identificar fallo silencioso"
    )
    assert "Silent Phase" in silent_evidence["topological_path"]


def test_lattice_induced_severity_from_child(narrator, context):
    """
    Verifica inducción de severidad via estructura de lattice.

    Propiedad fundamental: severity(node) = sup{severity(child) | child ∈ descendants(node)}
    Un fallo crítico en cualquier descendiente debe propagarse al ancestro.
    """
    with context.span("Parent") as parent:
        with context.span("Child") as child:
            # Simular fallo interno sin propagación de excepción
            child.status = StepStatus.FAILURE
            child.errors.append({"message": "Child failed", "type": "Error"})
        # Parent mantiene SUCCESS pero hijo falló

    report = narrator.summarize_execution(context)

    phase = report["phases"][0]
    assert phase["status"] == "CRITICO", (
        "Fallo en descendiente debe inducir CRITICO en ancestro via supremum"
    )
    assert report["verdict"] == "CRITICO"


def test_lattice_induced_severity_multiple_children(narrator, context):
    """
    Verifica supremum sobre múltiples hijos con diferentes severidades.
    sup(OPTIMO, ADVERTENCIA, CRITICO) = CRITICO

    El resultado debe ser el máximo del lattice presente.
    """
    with context.span("Parent"):
        with context.span("Child OK"):
            pass  # Status: SUCCESS → OPTIMO

        with context.span("Child Warning") as warn_child:
            warn_child.status = StepStatus.WARNING  # → ADVERTENCIA

        with context.span("Child Critical") as crit_child:
            crit_child.status = StepStatus.FAILURE
            crit_child.errors.append(
                {"message": "Critical failure in child", "type": "Error"}
            )  # → CRITICO

    report = narrator.summarize_execution(context)

    assert report["phases"][0]["status"] == "CRITICO", (
        "sup(OPTIMO, ADVERTENCIA, CRITICO) debe ser CRITICO"
    )
    assert report["verdict"] == "CRITICO"

    # Verificar que la evidencia forense captura el fallo
    critical_evidence = [
        e for e in report["forensic_evidence"] if "Critical failure" in e.get("message", "")
    ]
    assert len(critical_evidence) > 0, "Debe existir evidencia forense del fallo crítico"
