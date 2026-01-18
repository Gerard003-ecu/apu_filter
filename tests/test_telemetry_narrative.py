"""
Suite de Pruebas para el Narrador de Telemetría Híbrido
=======================================================

Fundamentos matemáticos:
- Teoría de Lattices: (L, ∨, ∧) donde L = {OPTIMO, ADVERTENCIA, CRITICO}
- Propagación de severidad via supremum en árbol de spans
- Transformación DIKW: Data → Information → Knowledge → Wisdom

Propiedades algebraicas verificadas:
- Idempotencia: a ∨ a = a
- Conmutatividad: a ∨ b = b ∨ a
- Asociatividad: (a ∨ b) ∨ c = a ∨ (b ∨ c)
- Absorción: a ∨ (a ∧ b) = a
- Identidades: a ∨ ⊥ = a, a ∧ ⊤ = a

Autor: Artesano Programador Senior
Versión: 2.0.0
"""

from itertools import permutations, product, combinations
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import time

import pytest

from app.telemetry import StepStatus, TelemetryContext
from app.telemetry_narrative import SeverityLevel, TelemetryNarrator


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: FIXTURES Y UTILIDADES
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def narrator() -> TelemetryNarrator:
    """Instancia limpia del narrador de telemetría."""
    return TelemetryNarrator()


@pytest.fixture
def context() -> TelemetryContext:
    """Contexto de telemetría aislado."""
    return TelemetryContext()


@pytest.fixture
def all_severity_levels() -> List[SeverityLevel]:
    """Conjunto completo del lattice de severidades (ordenado)."""
    return [SeverityLevel.OPTIMO, SeverityLevel.ADVERTENCIA, SeverityLevel.CRITICO]


@pytest.fixture
def lattice_elements() -> Dict[str, SeverityLevel]:
    """Elementos especiales del lattice con nombres semánticos."""
    return {
        "bottom": SeverityLevel.OPTIMO,      # ⊥ - Identidad para ∨
        "middle": SeverityLevel.ADVERTENCIA,  # Elemento intermedio
        "top": SeverityLevel.CRITICO,         # ⊤ - Identidad para ∧
    }


@pytest.fixture
def status_to_severity_map() -> Dict[StepStatus, SeverityLevel]:
    """Mapeo de StepStatus a SeverityLevel."""
    return {
        StepStatus.SUCCESS: SeverityLevel.OPTIMO,
        StepStatus.WARNING: SeverityLevel.ADVERTENCIA,
        StepStatus.FAILURE: SeverityLevel.CRITICO,
    }


def create_nested_context(depth: int, status: StepStatus = StepStatus.SUCCESS,
                          error_at_depth: Optional[int] = None) -> TelemetryContext:
    """
    Crea un contexto con spans anidados hasta la profundidad especificada.
    
    Args:
        depth: Número de niveles de anidamiento
        status: Estado final del span más profundo
        error_at_depth: Nivel donde inyectar un error (opcional)
    """
    ctx = TelemetryContext()
    
    def create_level(parent_ctx, current_depth: int, max_depth: int):
        span_name = f"Level_{current_depth}"
        with parent_ctx.span(span_name) as span:
            if current_depth == max_depth:
                span.status = status
                if error_at_depth == current_depth:
                    span.errors.append({
                        "message": f"Error at depth {current_depth}",
                        "type": "DeepError"
                    })
            elif current_depth < max_depth:
                if error_at_depth == current_depth:
                    span.errors.append({
                        "message": f"Error at depth {current_depth}",
                        "type": "IntermediateError"
                    })
                create_level(parent_ctx, current_depth + 1, max_depth)
    
    if depth > 0:
        create_level(ctx, 1, depth)
    
    return ctx


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2: PROPIEDADES FUNDAMENTALES DEL LATTICE - SUPREMUM
# ═══════════════════════════════════════════════════════════════════════════════

class TestLatticeSupremumProperties:
    """
    Pruebas de propiedades algebraicas del supremum (join, ∨).
    
    El supremum en un lattice satisface:
    1. Idempotencia: a ∨ a = a
    2. Conmutatividad: a ∨ b = b ∨ a
    3. Asociatividad: (a ∨ b) ∨ c = a ∨ (b ∨ c)
    4. Identidad con bottom: a ∨ ⊥ = a
    5. Absorción por top: a ∨ ⊤ = ⊤
    """
    
    def test_supremum_idempotency(self, all_severity_levels):
        """
        Propiedad de Idempotencia: ∀a ∈ L: a ∨ a = a
        
        Fundamental para cualquier semilattice.
        """
        for level in all_severity_levels:
            result = SeverityLevel.supremum(level, level)
            assert result == level, f"Violación de idempotencia: {level} ∨ {level} = {result}"
    
    def test_supremum_commutativity(self, all_severity_levels):
        """
        Propiedad de Conmutatividad: ∀a,b ∈ L: a ∨ b = b ∨ a
        
        El supremo es simétrico respecto a sus argumentos.
        """
        for a, b in permutations(all_severity_levels, 2):
            left = SeverityLevel.supremum(a, b)
            right = SeverityLevel.supremum(b, a)
            assert left == right, f"Violación de conmutatividad: {a} ∨ {b} ≠ {b} ∨ {a}"
    
    def test_supremum_associativity(self, all_severity_levels):
        """
        Propiedad de Asociatividad: ∀a,b,c ∈ L: (a ∨ b) ∨ c = a ∨ (b ∨ c)
        
        Garantiza consistencia en operaciones n-arias.
        """
        for a, b, c in product(all_severity_levels, repeat=3):
            left_assoc = SeverityLevel.supremum(SeverityLevel.supremum(a, b), c)
            right_assoc = SeverityLevel.supremum(a, SeverityLevel.supremum(b, c))
            assert left_assoc == right_assoc, \
                f"Violación de asociatividad: ({a} ∨ {b}) ∨ {c} ≠ {a} ∨ ({b} ∨ {c})"
    
    def test_supremum_identity_element(self, all_severity_levels, lattice_elements):
        """
        Propiedad de Identidad: ∀a ∈ L: a ∨ ⊥ = a
        
        OPTIMO actúa como elemento identidad (bottom) del lattice.
        """
        bottom = lattice_elements["bottom"]
        
        for level in all_severity_levels:
            result = SeverityLevel.supremum(level, bottom)
            assert result == level, \
                f"OPTIMO no actúa como identidad: {level} ∨ ⊥ = {result}"
    
    def test_supremum_absorbing_element(self, all_severity_levels, lattice_elements):
        """
        Propiedad de Absorción: ∀a ∈ L: a ∨ ⊤ = ⊤
        
        CRITICO actúa como elemento absorbente (top) del lattice.
        """
        top = lattice_elements["top"]
        
        for level in all_severity_levels:
            result = SeverityLevel.supremum(level, top)
            assert result == top, \
                f"CRITICO no actúa como absorbente: {level} ∨ ⊤ = {result}"
    
    def test_supremum_empty_returns_bottom(self, lattice_elements):
        """
        Convención: sup(∅) = ⊥
        
        El supremum del conjunto vacío retorna el elemento identidad.
        """
        result = SeverityLevel.supremum()
        assert result == lattice_elements["bottom"], \
            f"sup(∅) debe ser OPTIMO (bottom), obtenido: {result}"
    
    def test_supremum_single_element(self, all_severity_levels):
        """
        Caso trivial: sup({a}) = a
        """
        for level in all_severity_levels:
            result = SeverityLevel.supremum(level)
            assert result == level, f"sup({{{level}}}) ≠ {level}"


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: PROPIEDADES FUNDAMENTALES DEL LATTICE - INFIMUM
# ═══════════════════════════════════════════════════════════════════════════════

class TestLatticeInfimumProperties:
    """
    Pruebas de propiedades algebraicas del infimum (meet, ∧).
    
    El infimum es dual al supremum y satisface:
    1. Idempotencia: a ∧ a = a
    2. Conmutatividad: a ∧ b = b ∧ a
    3. Asociatividad: (a ∧ b) ∧ c = a ∧ (b ∧ c)
    4. Identidad con top: a ∧ ⊤ = a
    5. Absorción por bottom: a ∧ ⊥ = ⊥
    """
    
    def test_infimum_idempotency(self, all_severity_levels):
        """
        Propiedad de Idempotencia: ∀a ∈ L: a ∧ a = a
        """
        for level in all_severity_levels:
            result = SeverityLevel.infimum(level, level)
            assert result == level, f"Violación de idempotencia: {level} ∧ {level} = {result}"
    
    def test_infimum_commutativity(self, all_severity_levels):
        """
        Propiedad de Conmutatividad: ∀a,b ∈ L: a ∧ b = b ∧ a
        """
        for a, b in permutations(all_severity_levels, 2):
            left = SeverityLevel.infimum(a, b)
            right = SeverityLevel.infimum(b, a)
            assert left == right, f"Violación de conmutatividad: {a} ∧ {b} ≠ {b} ∧ {a}"
    
    def test_infimum_associativity(self, all_severity_levels):
        """
        Propiedad de Asociatividad: ∀a,b,c ∈ L: (a ∧ b) ∧ c = a ∧ (b ∧ c)
        """
        for a, b, c in product(all_severity_levels, repeat=3):
            left_assoc = SeverityLevel.infimum(SeverityLevel.infimum(a, b), c)
            right_assoc = SeverityLevel.infimum(a, SeverityLevel.infimum(b, c))
            assert left_assoc == right_assoc, \
                f"Violación de asociatividad: ({a} ∧ {b}) ∧ {c} ≠ {a} ∧ ({b} ∧ {c})"
    
    def test_infimum_identity_element(self, all_severity_levels, lattice_elements):
        """
        Propiedad de Identidad: ∀a ∈ L: a ∧ ⊤ = a
        
        CRITICO actúa como elemento identidad para el infimum.
        """
        top = lattice_elements["top"]
        
        for level in all_severity_levels:
            result = SeverityLevel.infimum(level, top)
            assert result == level, \
                f"CRITICO no actúa como identidad para ∧: {level} ∧ ⊤ = {result}"
    
    def test_infimum_absorbing_element(self, all_severity_levels, lattice_elements):
        """
        Propiedad de Absorción: ∀a ∈ L: a ∧ ⊥ = ⊥
        
        OPTIMO actúa como elemento absorbente para el infimum.
        """
        bottom = lattice_elements["bottom"]
        
        for level in all_severity_levels:
            result = SeverityLevel.infimum(level, bottom)
            assert result == bottom, \
                f"OPTIMO no actúa como absorbente para ∧: {level} ∧ ⊥ = {result}"
    
    def test_infimum_empty_returns_top(self, lattice_elements):
        """
        Convención: inf(∅) = ⊤
        
        El infimum del conjunto vacío retorna el elemento identidad del meet.
        """
        result = SeverityLevel.infimum()
        assert result == lattice_elements["top"], \
            f"inf(∅) debe ser CRITICO (top), obtenido: {result}"
    
    def test_infimum_single_element(self, all_severity_levels):
        """
        Caso trivial: inf({a}) = a
        """
        for level in all_severity_levels:
            result = SeverityLevel.infimum(level)
            assert result == level, f"inf({{{level}}}) ≠ {level}"


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4: LEYES DE ABSORCIÓN Y DUALIDAD
# ═══════════════════════════════════════════════════════════════════════════════

class TestLatticeAbsorptionAndDuality:
    """
    Pruebas de las leyes de absorción y dualidad del lattice.
    
    Leyes de absorción (definen lattice):
    - a ∨ (a ∧ b) = a
    - a ∧ (a ∨ b) = a
    
    Dualidad:
    - Las propiedades de ∨ tienen duales en ∧
    """
    
    def test_absorption_law_join_meet(self, all_severity_levels):
        """
        Primera ley de absorción: ∀a,b ∈ L: a ∨ (a ∧ b) = a
        
        Esta ley junto con la segunda caracteriza completamente un lattice.
        """
        for a, b in product(all_severity_levels, repeat=2):
            inner = SeverityLevel.infimum(a, b)  # a ∧ b
            result = SeverityLevel.supremum(a, inner)  # a ∨ (a ∧ b)
            assert result == a, \
                f"Violación de absorción: {a} ∨ ({a} ∧ {b}) = {result} ≠ {a}"
    
    def test_absorption_law_meet_join(self, all_severity_levels):
        """
        Segunda ley de absorción: ∀a,b ∈ L: a ∧ (a ∨ b) = a
        """
        for a, b in product(all_severity_levels, repeat=2):
            inner = SeverityLevel.supremum(a, b)  # a ∨ b
            result = SeverityLevel.infimum(a, inner)  # a ∧ (a ∨ b)
            assert result == a, \
                f"Violación de absorción: {a} ∧ ({a} ∨ {b}) = {result} ≠ {a}"
    
    def test_duality_principle(self, all_severity_levels, lattice_elements):
        """
        Principio de dualidad: intercambiar ∨↔∧ y ⊥↔⊤ preserva verdad.
        
        Verificamos que las identidades son duales:
        - a ∨ ⊥ = a  ↔  a ∧ ⊤ = a
        - a ∨ ⊤ = ⊤  ↔  a ∧ ⊥ = ⊥
        """
        bottom = lattice_elements["bottom"]
        top = lattice_elements["top"]
        
        for a in all_severity_levels:
            # Par dual 1
            join_bottom = SeverityLevel.supremum(a, bottom)
            meet_top = SeverityLevel.infimum(a, top)
            assert join_bottom == meet_top == a, \
                f"Violación de dualidad: {a}∨⊥={join_bottom}, {a}∧⊤={meet_top}"
            
            # Par dual 2
            join_top = SeverityLevel.supremum(a, top)
            meet_bottom = SeverityLevel.infimum(a, bottom)
            assert join_top == top and meet_bottom == bottom, \
                f"Violación de dualidad: {a}∨⊤={join_top}, {a}∧⊥={meet_bottom}"
    
    def test_lattice_order_consistency(self, all_severity_levels):
        """
        Consistencia del orden inducido:
        a ≤ b ⟺ a ∨ b = b ⟺ a ∧ b = a
        """
        # Orden esperado: OPTIMO < ADVERTENCIA < CRITICO
        O = SeverityLevel.OPTIMO
        A = SeverityLevel.ADVERTENCIA
        C = SeverityLevel.CRITICO
        
        # O ≤ A
        assert SeverityLevel.supremum(O, A) == A
        assert SeverityLevel.infimum(O, A) == O
        
        # A ≤ C
        assert SeverityLevel.supremum(A, C) == C
        assert SeverityLevel.infimum(A, C) == A
        
        # O ≤ C (transitividad)
        assert SeverityLevel.supremum(O, C) == C
        assert SeverityLevel.infimum(O, C) == O


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5: ORDEN PARCIAL Y TRANSITIVIDAD
# ═══════════════════════════════════════════════════════════════════════════════

class TestPartialOrder:
    """
    Pruebas del orden parcial inducido por el lattice.
    
    Propiedades del orden ≤:
    1. Reflexividad: a ≤ a
    2. Antisimetría: a ≤ b ∧ b ≤ a ⟹ a = b
    3. Transitividad: a ≤ b ∧ b ≤ c ⟹ a ≤ c
    """
    
    def test_order_reflexivity(self, all_severity_levels):
        """
        Reflexividad: ∀a: a ≤ a
        
        Definición via supremum: a ≤ a ⟺ a ∨ a = a
        """
        for level in all_severity_levels:
            assert SeverityLevel.supremum(level, level) == level
    
    def test_order_antisymmetry(self, all_severity_levels):
        """
        Antisimetría: a ≤ b ∧ b ≤ a ⟹ a = b
        """
        for a, b in product(all_severity_levels, repeat=2):
            a_leq_b = (SeverityLevel.supremum(a, b) == b)
            b_leq_a = (SeverityLevel.supremum(b, a) == a)
            
            if a_leq_b and b_leq_a:
                assert a == b, f"Violación de antisimetría: {a} ≤ {b} y {b} ≤ {a} pero {a} ≠ {b}"
    
    def test_order_transitivity(self, all_severity_levels):
        """
        Transitividad: a ≤ b ∧ b ≤ c ⟹ a ≤ c
        """
        for a, b, c in product(all_severity_levels, repeat=3):
            a_leq_b = (SeverityLevel.supremum(a, b) == b)
            b_leq_c = (SeverityLevel.supremum(b, c) == c)
            a_leq_c = (SeverityLevel.supremum(a, c) == c)
            
            if a_leq_b and b_leq_c:
                assert a_leq_c, \
                    f"Violación de transitividad: {a}≤{b} y {b}≤{c} pero no {a}≤{c}"
    
    def test_total_order(self, all_severity_levels):
        """
        El lattice de severidades es totalmente ordenado (cadena).
        
        ∀a,b: a ≤ b ∨ b ≤ a
        """
        for a, b in combinations(all_severity_levels, 2):
            a_leq_b = (SeverityLevel.supremum(a, b) == b)
            b_leq_a = (SeverityLevel.supremum(b, a) == a)
            
            assert a_leq_b or b_leq_a, \
                f"No es orden total: ni {a}≤{b} ni {b}≤{a}"
    
    def test_expected_order_chain(self):
        """
        Verifica la cadena esperada: OPTIMO < ADVERTENCIA < CRITICO
        """
        O = SeverityLevel.OPTIMO
        A = SeverityLevel.ADVERTENCIA
        C = SeverityLevel.CRITICO
        
        # O < A (O ≤ A y O ≠ A)
        assert SeverityLevel.supremum(O, A) == A
        assert O != A
        
        # A < C
        assert SeverityLevel.supremum(A, C) == C
        assert A != C
        
        # O < C (por transitividad)
        assert SeverityLevel.supremum(O, C) == C


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 6: SUPREMUM N-ARIO Y CONSISTENCIA
# ═══════════════════════════════════════════════════════════════════════════════

class TestNaryOperations:
    """
    Pruebas para operaciones n-arias y su consistencia con las binarias.
    """
    
    def test_nary_supremum_consistency_with_binary(self, all_severity_levels):
        """
        Consistencia del supremo n-ario con composición binaria.
        
        sup(a₁, a₂, ..., aₙ) = sup(sup(...sup(a₁, a₂), a₃)..., aₙ)
        """
        # Reducción binaria
        binary_reduction = all_severity_levels[0]
        for level in all_severity_levels[1:]:
            binary_reduction = SeverityLevel.supremum(binary_reduction, level)
        
        # N-ario directo
        nary_result = SeverityLevel.supremum(*all_severity_levels)
        
        assert nary_result == binary_reduction
    
    def test_nary_supremum_of_full_lattice_is_top(self, all_severity_levels, lattice_elements):
        """
        El supremum de todo el lattice es el elemento top.
        
        sup(L) = ⊤
        """
        result = SeverityLevel.supremum(*all_severity_levels)
        assert result == lattice_elements["top"]
    
    def test_nary_infimum_of_full_lattice_is_bottom(self, all_severity_levels, lattice_elements):
        """
        El infimum de todo el lattice es el elemento bottom.
        
        inf(L) = ⊥
        """
        result = SeverityLevel.infimum(*all_severity_levels)
        assert result == lattice_elements["bottom"]
    
    def test_nary_operations_order_independent(self, all_severity_levels):
        """
        Las operaciones n-arias son independientes del orden de argumentos.
        """
        from itertools import permutations
        
        # Todas las permutaciones dan el mismo resultado
        for perm in permutations(all_severity_levels):
            sup_result = SeverityLevel.supremum(*perm)
            inf_result = SeverityLevel.infimum(*perm)
            
            assert sup_result == SeverityLevel.CRITICO
            assert inf_result == SeverityLevel.OPTIMO


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 7: PRUEBAS DE CONTEXTO VACÍO Y CASOS BASE
# ═══════════════════════════════════════════════════════════════════════════════

class TestEmptyContext:
    """
    Pruebas para contexto de telemetría vacío.
    
    Caso base del sistema: sin información → severidad mínima.
    """
    
    def test_empty_context_verdict(self, narrator, context):
        """Contexto vacío produce veredicto OPTIMO (bottom del lattice)."""
        report = narrator.summarize_execution(context)
        
        assert report["verdict"] == "OPTIMO"
    
    def test_empty_context_narrative(self, narrator, context):
        """Contexto vacío indica ausencia de telemetría en la narrativa."""
        report = narrator.summarize_execution(context)
        
        assert "Sin telemetría" in report["narrative"]
    
    def test_empty_context_phases(self, narrator, context):
        """Contexto vacío no tiene fases."""
        report = narrator.summarize_execution(context)
        
        assert report["phases"] == []
    
    def test_empty_context_no_forensic_evidence(self, narrator, context):
        """Contexto vacío no tiene evidencia forense."""
        report = narrator.summarize_execution(context)
        
        assert report.get("forensic_evidence", []) == []
    
    def test_empty_context_report_structure(self, narrator, context):
        """Verifica estructura completa del reporte para contexto vacío."""
        report = narrator.summarize_execution(context)
        
        required_keys = ["verdict", "narrative", "phases"]
        for key in required_keys:
            assert key in report, f"Falta clave requerida: {key}"


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 8: PRUEBAS DE CONTEXTO LEGACY
# ═══════════════════════════════════════════════════════════════════════════════

class TestLegacyContext:
    """
    Pruebas para compatibilidad con contextos legacy.
    """
    
    def test_legacy_success_report(self, narrator):
        """Fallback para contextos legacy exitosos."""
        legacy_context = TelemetryContext()
        legacy_context.start_step("legacy_step")
        legacy_context.end_step("legacy_step", StepStatus.SUCCESS)
        
        report = narrator.summarize_execution(legacy_context)
        
        assert report["verdict"] == "OPTIMO"
        assert "Legacy" in report["narrative"]
    
    def test_legacy_error_elevates_to_critical(self, narrator):
        """Error en contexto legacy eleva severidad a CRITICO."""
        error_context = TelemetryContext()
        error_context.start_step("legacy_step")
        error_context.end_step("legacy_step", StepStatus.SUCCESS)
        error_context.record_error("legacy_step", "Something failed")
        
        report = narrator.summarize_execution(error_context)
        
        assert report["verdict"] == "CRITICO"
    
    def test_legacy_error_creates_forensic_evidence(self, narrator):
        """Error en contexto legacy genera evidencia forense."""
        error_context = TelemetryContext()
        error_context.start_step("legacy_step")
        error_context.end_step("legacy_step", StepStatus.SUCCESS)
        error_context.record_error("legacy_step", "Something failed")
        
        report = narrator.summarize_execution(error_context)
        
        assert len(report["forensic_evidence"]) > 0
    
    def test_legacy_multiple_steps(self, narrator):
        """Múltiples pasos legacy se procesan correctamente."""
        ctx = TelemetryContext()
        
        for i in range(5):
            ctx.start_step(f"step_{i}")
            ctx.end_step(f"step_{i}", StepStatus.SUCCESS)
        
        report = narrator.summarize_execution(ctx)
        
        assert report["verdict"] == "OPTIMO"


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 9: PRUEBAS DE JERARQUÍA DE SPANS
# ═══════════════════════════════════════════════════════════════════════════════

class TestHierarchicalSpans:
    """
    Pruebas para la estructura jerárquica de spans.
    """
    
    def test_hierarchical_success(self, narrator, context):
        """Árbol de spans completamente exitoso produce OPTIMO."""
        with context.span("Phase 1"):
            with context.span("Operation A"):
                pass
            with context.span("Operation B"):
                pass
        
        report = narrator.summarize_execution(context)
        
        assert report["verdict"] == "OPTIMO"
        assert len(report["phases"]) == 1
        assert report["phases"][0]["status"] == "OPTIMO"
    
    def test_hierarchical_warning_propagation(self, narrator, context):
        """Warning en span hijo propaga a padre."""
        with context.span("Phase 1"):
            with context.span("Subtask"):
                pass
        
        context.root_spans[0].status = StepStatus.WARNING
        
        report = narrator.summarize_execution(context)
        
        assert report["verdict"] == "ADVERTENCIA"
        assert report["phases"][0]["status"] == "ADVERTENCIA"
    
    def test_hierarchical_failure_propagation(self, narrator, context):
        """Fallo en span hijo propaga a la raíz."""
        try:
            with context.span("Root Phase"):
                with context.span("Level 1"):
                    with context.span("Level 2"):
                        raise ValueError("Deep Error")
        except ValueError:
            pass
        
        report = narrator.summarize_execution(context)
        
        assert report["verdict"] == "CRITICO"
    
    def test_topological_path_in_evidence(self, narrator, context):
        """La evidencia forense incluye la ruta topológica completa."""
        try:
            with context.span("Root Phase"):
                with context.span("Level 1"):
                    with context.span("Level 2"):
                        raise ValueError("Deep Error")
        except ValueError:
            pass
        
        report = narrator.summarize_execution(context)
        
        expected_path = "Root Phase → Level 1 → Level 2"
        matching_evidence = next(
            (issue for issue in report["forensic_evidence"]
             if expected_path in issue.get("topological_path", "")
             and issue.get("message") == "Deep Error"),
            None
        )
        
        assert matching_evidence is not None


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 10: PRUEBAS DE PROPAGACIÓN DE SEVERIDAD VIA LATTICE
# ═══════════════════════════════════════════════════════════════════════════════

class TestSeverityPropagation:
    """
    Pruebas de propagación de severidad usando propiedades del lattice.
    
    Propiedad fundamental:
        severity(node) = sup{severity(child) | child ∈ descendants(node)}
    """
    
    def test_child_failure_induces_parent_critical(self, narrator, context):
        """Fallo en descendiente induce CRITICO en ancestro via supremum."""
        with context.span("Parent") as parent:
            with context.span("Child") as child:
                child.status = StepStatus.FAILURE
                child.errors.append({"message": "Child failed", "type": "Error"})
        
        report = narrator.summarize_execution(context)
        
        assert report["phases"][0]["status"] == "CRITICO"
        assert report["verdict"] == "CRITICO"
    
    def test_multiple_children_supremum(self, narrator, context):
        """
        Múltiples hijos: severidad padre = supremum de severidades hijos.
        
        sup(OPTIMO, ADVERTENCIA, CRITICO) = CRITICO
        """
        with context.span("Parent"):
            with context.span("Child OK"):
                pass  # SUCCESS → OPTIMO
            
            with context.span("Child Warning") as warn_child:
                warn_child.status = StepStatus.WARNING  # → ADVERTENCIA
            
            with context.span("Child Critical") as crit_child:
                crit_child.status = StepStatus.FAILURE
                crit_child.errors.append({
                    "message": "Critical failure",
                    "type": "Error"
                })  # → CRITICO
        
        report = narrator.summarize_execution(context)
        
        assert report["phases"][0]["status"] == "CRITICO"
        assert report["verdict"] == "CRITICO"
    
    def test_all_success_produces_optimo(self, narrator, context):
        """Todos los hijos exitosos producen OPTIMO en padre."""
        with context.span("Parent"):
            for i in range(5):
                with context.span(f"Child_{i}"):
                    pass  # Todos SUCCESS
        
        report = narrator.summarize_execution(context)
        
        assert report["verdict"] == "OPTIMO"
    
    def test_single_warning_elevates_to_advertencia(self, narrator, context):
        """Un solo warning entre muchos exitosos eleva a ADVERTENCIA."""
        with context.span("Parent"):
            for i in range(4):
                with context.span(f"Child_{i}"):
                    pass
            
            with context.span("Warning Child") as warn_child:
                warn_child.status = StepStatus.WARNING
        
        report = narrator.summarize_execution(context)
        
        assert report["verdict"] == "ADVERTENCIA"
    
    def test_warning_plus_failure_equals_critical(self, narrator, context):
        """
        sup(ADVERTENCIA, CRITICO) = CRITICO
        
        Un fallo supera cualquier cantidad de warnings.
        """
        with context.span("Parent"):
            # Múltiples warnings
            for i in range(3):
                with context.span(f"Warning_{i}") as warn:
                    warn.status = StepStatus.WARNING
            
            # Un fallo
            with context.span("Failure") as fail:
                fail.status = StepStatus.FAILURE
                fail.errors.append({"message": "Failed", "type": "Error"})
        
        report = narrator.summarize_execution(context)
        
        assert report["verdict"] == "CRITICO"


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 11: PRUEBAS DE FALLOS SILENCIOSOS
# ═══════════════════════════════════════════════════════════════════════════════

class TestSilentFailures:
    """
    Pruebas para detección de fallos silenciosos.
    
    Fallo silencioso: Status FAILURE sin excepción/error explícito registrado.
    """
    
    def test_silent_failure_detected(self, narrator, context):
        """Fallo silencioso es detectado y reportado."""
        with context.span("Silent Phase"):
            pass
        
        context.root_spans[0].status = StepStatus.FAILURE
        
        report = narrator.summarize_execution(context)
        
        assert report["verdict"] == "CRITICO"
    
    def test_silent_failure_creates_evidence(self, narrator, context):
        """Fallo silencioso genera evidencia forense."""
        with context.span("Silent Phase"):
            pass
        
        context.root_spans[0].status = StepStatus.FAILURE
        
        report = narrator.summarize_execution(context)
        
        assert len(report["forensic_evidence"]) > 0
    
    def test_silent_failure_evidence_type(self, narrator, context):
        """La evidencia de fallo silencioso tiene tipo correcto."""
        with context.span("Silent Phase"):
            pass
        
        context.root_spans[0].status = StepStatus.FAILURE
        
        report = narrator.summarize_execution(context)
        
        silent_evidence = report["forensic_evidence"][0]
        assert silent_evidence["type"] == "SilentFailure"
    
    def test_silent_failure_includes_path(self, narrator, context):
        """La evidencia de fallo silencioso incluye ruta topológica."""
        with context.span("Silent Phase"):
            pass
        
        context.root_spans[0].status = StepStatus.FAILURE
        
        report = narrator.summarize_execution(context)
        
        silent_evidence = report["forensic_evidence"][0]
        assert "Silent Phase" in silent_evidence["topological_path"]


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 12: PRUEBAS DE PROFUNDIDAD Y CASOS LÍMITE
# ═══════════════════════════════════════════════════════════════════════════════

class TestDepthAndEdgeCases:
    """
    Pruebas de spans profundos y casos límite.
    """
    
    def test_deeply_nested_success(self, narrator):
        """Spans muy anidados exitosos producen OPTIMO."""
        ctx = create_nested_context(depth=10, status=StepStatus.SUCCESS)
        
        report = narrator.summarize_execution(ctx)
        
        assert report["verdict"] == "OPTIMO"
    
    def test_deeply_nested_failure_at_bottom(self, narrator):
        """Fallo en el span más profundo propaga a la raíz."""
        ctx = create_nested_context(depth=10, status=StepStatus.FAILURE, error_at_depth=10)
        
        report = narrator.summarize_execution(ctx)
        
        assert report["verdict"] == "CRITICO"
    
    def test_failure_at_intermediate_level(self, narrator):
        """Fallo en nivel intermedio propaga correctamente."""
        ctx = create_nested_context(depth=10, status=StepStatus.SUCCESS, error_at_depth=5)
        
        report = narrator.summarize_execution(ctx)
        
        assert report["verdict"] == "CRITICO"
        
        # La evidencia debe incluir la ruta hasta el nivel del error
        evidence = report.get("forensic_evidence", [])
        assert len(evidence) > 0
    
    def test_multiple_root_spans(self, narrator, context):
        """Múltiples spans raíz se procesan correctamente."""
        for i in range(5):
            with context.span(f"Root_{i}"):
                pass
        
        report = narrator.summarize_execution(context)
        
        assert report["verdict"] == "OPTIMO"
        assert len(report["phases"]) == 5
    
    def test_multiple_root_spans_with_one_failure(self, narrator, context):
        """Un fallo entre múltiples raíces eleva severidad."""
        for i in range(5):
            with context.span(f"Root_{i}") as span:
                if i == 2:
                    span.status = StepStatus.FAILURE
                    span.errors.append({"message": "Root 2 failed", "type": "Error"})
        
        report = narrator.summarize_execution(context)
        
        assert report["verdict"] == "CRITICO"
    
    def test_unicode_span_names(self, narrator, context):
        """Nombres de spans con Unicode se manejan correctamente."""
        with context.span("日本語 Phase"):
            with context.span("Café ñ"):
                pass
        
        report = narrator.summarize_execution(context)
        
        assert report["verdict"] == "OPTIMO"
    
    def test_empty_span_name(self, narrator, context):
        """Nombres de spans vacíos no causan errores."""
        with context.span(""):
            pass
        
        report = narrator.summarize_execution(context)
        
        assert report["verdict"] == "OPTIMO"


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 13: PRUEBAS DE ESTRUCTURA DEL REPORTE
# ═══════════════════════════════════════════════════════════════════════════════

class TestReportStructure:
    """
    Pruebas de la estructura completa del reporte.
    """
    
    def test_report_has_required_keys(self, narrator, context):
        """El reporte contiene todas las claves requeridas."""
        with context.span("Test Phase"):
            pass
        
        report = narrator.summarize_execution(context)
        
        required_keys = ["verdict", "narrative", "phases"]
        for key in required_keys:
            assert key in report, f"Falta clave requerida: {key}"
    
    def test_phase_structure(self, narrator, context):
        """Cada fase tiene la estructura correcta."""
        with context.span("Test Phase"):
            with context.span("Subtask"):
                pass
        
        report = narrator.summarize_execution(context)
        
        phase = report["phases"][0]
        assert "status" in phase
        assert phase["status"] in ["OPTIMO", "ADVERTENCIA", "CRITICO"]
    
    def test_forensic_evidence_structure(self, narrator, context):
        """La evidencia forense tiene la estructura correcta."""
        try:
            with context.span("Failing Phase"):
                raise ValueError("Test error")
        except ValueError:
            pass
        
        report = narrator.summarize_execution(context)
        
        assert len(report["forensic_evidence"]) > 0
        
        evidence = report["forensic_evidence"][0]
        assert "topological_path" in evidence
        assert "message" in evidence or "type" in evidence
    
    def test_narrative_reflects_verdict(self, narrator, context):
        """La narrativa refleja el veredicto."""
        # Caso OPTIMO
        report_ok = narrator.summarize_execution(context)
        assert "óptima" in report_ok["narrative"].lower() or "sin telemetría" in report_ok["narrative"].lower()
        
        # Caso CRITICO
        with context.span("Failing"):
            pass
        context.root_spans[0].status = StepStatus.FAILURE
        context.root_spans[0].errors.append({"message": "Error", "type": "Error"})
        
        report_fail = narrator.summarize_execution(context)
        assert "crítico" in report_fail["narrative"].lower() or "fallo" in report_fail["narrative"].lower()
    
    def test_verdict_is_valid_severity(self, narrator, context):
        """El veredicto es una severidad válida."""
        with context.span("Test"):
            pass
        
        report = narrator.summarize_execution(context)
        
        assert report["verdict"] in ["OPTIMO", "ADVERTENCIA", "CRITICO"]


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 14: PRUEBAS DE IDEMPOTENCIA Y CONSISTENCIA
# ═══════════════════════════════════════════════════════════════════════════════

class TestIdempotenceAndConsistency:
    """
    Pruebas de idempotencia y consistencia del narrador.
    """
    
    def test_summarize_idempotent(self, narrator, context):
        """Múltiples llamadas a summarize producen el mismo resultado."""
        with context.span("Test Phase"):
            with context.span("Subtask"):
                pass
        
        report1 = narrator.summarize_execution(context)
        report2 = narrator.summarize_execution(context)
        
        assert report1["verdict"] == report2["verdict"]
        assert report1["narrative"] == report2["narrative"]
        assert len(report1["phases"]) == len(report2["phases"])
    
    def test_different_narrators_same_result(self, context):
        """Diferentes instancias de narrador producen mismo resultado."""
        with context.span("Test"):
            pass
        
        narrator1 = TelemetryNarrator()
        narrator2 = TelemetryNarrator()
        
        report1 = narrator1.summarize_execution(context)
        report2 = narrator2.summarize_execution(context)
        
        assert report1["verdict"] == report2["verdict"]
    
    def test_context_not_modified(self, narrator, context):
        """El contexto no se modifica durante la summarización."""
        with context.span("Test"):
            pass
        
        spans_before = len(context.root_spans)
        _ = narrator.summarize_execution(context)
        spans_after = len(context.root_spans)
        
        assert spans_before == spans_after


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 15: PRUEBAS DE MAPEO DIKW
# ═══════════════════════════════════════════════════════════════════════════════

class TestDIKWMapping:
    """
    Pruebas del mapeo Data → Information → Knowledge → Wisdom.
    
    - Data: eventos raw de telemetría
    - Information: spans estructurados
    - Knowledge: severidades calculadas
    - Wisdom: narrativa y veredicto
    """
    
    def test_data_to_information(self, context):
        """Los eventos se estructuran en spans (Data → Information)."""
        with context.span("Phase"):
            with context.span("Subtask"):
                pass
        
        assert len(context.root_spans) == 1
        assert len(context.root_spans[0].children) == 1
    
    def test_information_to_knowledge(self, narrator, context):
        """Los spans se transforman en severidades (Information → Knowledge)."""
        with context.span("Phase"):
            pass
        
        context.root_spans[0].status = StepStatus.WARNING
        
        report = narrator.summarize_execution(context)
        
        assert report["phases"][0]["status"] == "ADVERTENCIA"
    
    def test_knowledge_to_wisdom(self, narrator, context):
        """Las severidades se sintetizan en narrativa (Knowledge → Wisdom)."""
        with context.span("Phase"):
            pass
        
        report = narrator.summarize_execution(context)
        
        assert "narrative" in report
        assert isinstance(report["narrative"], str)
        assert len(report["narrative"]) > 0
    
    def test_complete_dikw_pipeline(self, narrator, context):
        """Pipeline DIKW completo produce salida coherente."""
        # Data: eventos de spans
        with context.span("Root"):
            with context.span("Child1"):
                pass
            with context.span("Child2") as child:
                child.status = StepStatus.WARNING
        
        # Procesar
        report = narrator.summarize_execution(context)
        
        # Information: estructura de fases
        assert len(report["phases"]) == 1
        
        # Knowledge: severidad calculada
        assert report["verdict"] in ["OPTIMO", "ADVERTENCIA", "CRITICO"]
        
        # Wisdom: narrativa generada
        assert len(report["narrative"]) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 16: PRUEBAS DE ROBUSTEZ
# ═══════════════════════════════════════════════════════════════════════════════

class TestRobustness:
    """
    Pruebas de robustez ante entradas inusuales.
    """
    
    def test_exception_with_none_message(self, narrator, context):
        """Excepción con mensaje None se maneja correctamente."""
        with context.span("Test") as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": None, "type": "NoneError"})
        
        report = narrator.summarize_execution(context)
        
        assert report["verdict"] == "CRITICO"
    
    def test_exception_with_empty_message(self, narrator, context):
        """Excepción con mensaje vacío se maneja correctamente."""
        with context.span("Test") as span:
            span.status = StepStatus.FAILURE
            span.errors.append({"message": "", "type": "EmptyError"})
        
        report = narrator.summarize_execution(context)
        
        assert report["verdict"] == "CRITICO"
    
    def test_very_long_span_name(self, narrator, context):
        """Nombres de span muy largos no causan problemas."""
        long_name = "A" * 10000
        
        with context.span(long_name):
            pass
        
        report = narrator.summarize_execution(context)
        
        assert report["verdict"] == "OPTIMO"
    
    def test_special_characters_in_span_name(self, narrator, context):
        """Caracteres especiales en nombres de span se manejan."""
        special_name = "Test\n\t\r<>&\"'Phase"
        
        with context.span(special_name):
            pass
        
        report = narrator.summarize_execution(context)
        
        assert report["verdict"] == "OPTIMO"
    
    def test_concurrent_status_changes(self, narrator, context):
        """Cambios de status durante iteración no causan errores."""
        with context.span("Parent"):
            for i in range(10):
                with context.span(f"Child_{i}") as child:
                    # Alternar status
                    if i % 3 == 0:
                        child.status = StepStatus.WARNING
        
        report = narrator.summarize_execution(context)
        
        # Debe ser al menos ADVERTENCIA por los warnings
        assert report["verdict"] in ["ADVERTENCIA", "CRITICO"]


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 17: PRUEBAS DE INTEGRACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """
    Pruebas de integración del sistema completo.
    """
    
    def test_complete_success_workflow(self, narrator, context):
        """Flujo completo exitoso."""
        with context.span("Initialization"):
            with context.span("Load Config"):
                pass
            with context.span("Validate Config"):
                pass
        
        with context.span("Processing"):
            for i in range(3):
                with context.span(f"Batch_{i}"):
                    pass
        
        with context.span("Finalization"):
            with context.span("Save Results"):
                pass
        
        report = narrator.summarize_execution(context)
        
        assert report["verdict"] == "OPTIMO"
        assert len(report["phases"]) == 3
        assert all(p["status"] == "OPTIMO" for p in report["phases"])
    
    def test_complete_failure_workflow(self, narrator, context):
        """Flujo con fallo que interrumpe el proceso."""
        try:
            with context.span("Initialization"):
                pass
            
            with context.span("Processing"):
                with context.span("Critical Task"):
                    raise RuntimeError("Processing failed")
        except RuntimeError:
            pass
        
        report = narrator.summarize_execution(context)
        
        assert report["verdict"] == "CRITICO"
        assert len(report["forensic_evidence"]) > 0
    
    def test_mixed_severity_workflow(self, narrator, context):
        """Flujo con severidades mixtas."""
        with context.span("Phase 1"):
            pass  # SUCCESS
        
        with context.span("Phase 2") as p2:
            p2.status = StepStatus.WARNING  # WARNING
        
        with context.span("Phase 3"):
            pass  # SUCCESS
        
        report = narrator.summarize_execution(context)
        
        assert report["verdict"] == "ADVERTENCIA"
        
        statuses = [p["status"] for p in report["phases"]]
        assert "OPTIMO" in statuses
        assert "ADVERTENCIA" in statuses


# ═══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 18: PRUEBAS DE RENDIMIENTO
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerformance:
    """
    Pruebas de rendimiento.
    """
    
    @pytest.mark.slow
    def test_large_span_tree_performance(self, narrator):
        """Rendimiento con árbol de spans grande."""
        ctx = TelemetryContext()
        
        with ctx.span("Root"):
            for i in range(100):
                with ctx.span(f"Branch_{i}"):
                    for j in range(10):
                        with ctx.span(f"Leaf_{i}_{j}"):
                            pass
        
        start = time.time()
        report = narrator.summarize_execution(ctx)
        elapsed = time.time() - start
        
        assert elapsed < 5.0  # Menos de 5 segundos
        assert report["verdict"] == "OPTIMO"
    
    @pytest.mark.slow
    def test_deep_nesting_performance(self, narrator):
        """Rendimiento con anidamiento profundo."""
        ctx = create_nested_context(depth=100, status=StepStatus.SUCCESS)
        
        start = time.time()
        report = narrator.summarize_execution(ctx)
        elapsed = time.time() - start
        
        assert elapsed < 2.0
        assert report["verdict"] == "OPTIMO"
    
    @pytest.mark.slow
    def test_repeated_summarization_performance(self, narrator, context):
        """Rendimiento de summarizaciones repetidas."""
        with context.span("Test"):
            for i in range(10):
                with context.span(f"Sub_{i}"):
                    pass
        
        n_iterations = 100
        
        start = time.time()
        for _ in range(n_iterations):
            narrator.summarize_execution(context)
        elapsed = time.time() - start
        
        assert elapsed < 2.0
        assert elapsed / n_iterations < 0.05  # < 50ms por iteración


# ═══════════════════════════════════════════════════════════════════════════════
# EJECUCIÓN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-m", "not slow"
    ])
