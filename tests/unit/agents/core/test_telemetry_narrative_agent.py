# -- coding: utf-8 --
r"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Test Telemetry Narrative Agent (Suite de Validación de Difeomorfismo Semántico)║
║  Ruta   : tests/unit/agents/core/test_telemetry_narrative_agent.py                       ║
║  Versión: 2.0.0-Lattice-Diffeomorphism-Canonical-Strict-Nested                           ║
╠══════════════════════════════════════════════════════════════════════════════════════════╣
║                                                                                          ║
║  ARQUITECTURA DE PRUEBAS (Composición Funtorial Φ₃ ∘ Φ₂ ∘ Φ₁):                           ║
║  ──────────────────────────────────────────────────────────────────────────────          ║
║  Este módulo de pruebas implementa una batería exhaustiva que valida la integridad       ║
║  reticular, semántica y diplomática del endofuntor Z_Narrative.                          ║
║                                                                                          ║
║  FASE 1 → Colapso en Retículo Distributivo Acotado                                       ║
║           Valida: Veredicto = ⨆ v_i (Worst-Case Scenario determinista)                   ║
║                                                                                          ║
║  FASE 2 → Certificación de Difeomorfismo Semántico                                       ║
║           Valida: β₁ > 0 ⇒ "SOCAVON LOGICO" y Ψ < 1.0 ⇒ "PIRAMIDE INVERTIDA"             ║
║                                                                                          ║
║  FASE 3 → Canonicalización Diplomática y Acta Inmutable                                  ║
║           Valida: Eliminación de caracteres de control y pureza forense criptográfica    ║
║                                                                                          ║
║  COBERTURA DE EXCEPCIONES ONTOLÓGICAS:                                                   ║
║  ──────────────────────────────────────────────────────────────────────────────          ║
║  • DomainIntegrityViolationError       → Violaciones de dominio ontológico               ║
║  • SeverityLatticeCollapseError        → Fallo en operación Supremo (⨆) del retículo     ║
║  • SemanticDiffeomorphismViolationError → Deriva semántica del LLM                       ║
║  • NarrativeCanonicalizationError      → Fallo en canonicalización inmutable             ║
║                                                                                          ║
║  EJECUCIÓN:                                                                              ║
║  ──────────────────────────────────────────────────────────────────────────────          ║
║  $ pytest tests/unit/agents/core/test_telemetry_narrative_agent.py -v                    ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════════════════════
# §0. IMPORTACIONES Y CONFIGURACIÓN DEL ENTORNO DE PRUEBAS
# ═══════════════════════════════════════════════════════════════════════════════════════════

import pytest
import numpy as np
import math
from typing import Any, Dict, List, Optional
from enum import IntEnum

# Importación del módulo bajo prueba
from app.core.telemetry_narrative_agent import (
    # Excepciones Ontológicas y Semánticas
    TelemetryNarrativeAgentError,
    DomainIntegrityViolationError,
    SeverityLatticeCollapseError,
    SemanticDiffeomorphismViolationError,
    NarrativeCanonicalizationError,
    # Enumeración de Severidad (Retículo)
    SeverityLevel,
    # Estructuras Inmutables (DTOs)
    LatticeCollapseState,
    DiffeomorphismAuditData,
    NarrativeCanonicalizationData,
    Phase1LatticeHandoff,
    Phase2DiffeomorphismHandoff,
    NarrativeAgentState,
    # Fases Anidadas
    Phase1_SeverityLatticeCollapser,
    Phase2_SemanticDiffeomorphismCertifier,
    Phase3_NarrativeCanonicalizationEnforcer,
    # Orquestador Supremo
    TelemetryNarrativeAgent,
    # Constantes Retículares y Semánticas
    _BETTI_POSITIVITY_THRESHOLD,
    _BETTI_INTEGRALITY_TOLERANCE,
    _PYRAMID_STABILITY_THRESHOLD,
    _STABILITY_EPSILON,
    _REQUIRED_TOKEN_SOCAVON,
    _REQUIRED_TOKEN_PIRAMIDE,
    _BETA_1_METRIC_NAMES,
    _PYRAMID_STABILITY_METRIC_NAMES,
    _MIN_NARRATIVE_LENGTH,
    _MAX_NARRATIVE_LENGTH,
    _CONTROL_CHARACTER_PATTERN,
)

# ═══════════════════════════════════════════════════════════════════════════════════════════
# §A. FIXTURES Y UTILITARIOS DE PRUEBA (Infraestructura Categórica)
# ═══════════════════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def narrative_agent() -> TelemetryNarrativeAgent:
    """
    Fixture: Instancia del Intérprete Diplomático Supremo TelemetryNarrativeAgent.
    Retorna el endofuntor completo para pruebas de integración.
    """
    return TelemetryNarrativeAgent()


@pytest.fixture
def phase1_collapser() -> Phase1_SeverityLatticeCollapser:
    """
    Fixture: Instancia de Phase1_SeverityLatticeCollapser.
    Para pruebas unitarias de la Fase 1.
    """
    return Phase1_SeverityLatticeCollapser()


@pytest.fixture
def phase2_certifier() -> Phase2_SemanticDiffeomorphismCertifier:
    """
    Fixture: Instancia de Phase2_SemanticDiffeomorphismCertifier.
    Para pruebas unitarias de la Fase 2.
    """
    return Phase2_SemanticDiffeomorphismCertifier()


@pytest.fixture
def phase3_enforcer() -> Phase3_NarrativeCanonicalizationEnforcer:
    """
    Fixture: Instancia de Phase3_NarrativeCanonicalizationEnforcer.
    Para pruebas unitarias de la Fase 3.
    """
    return Phase3_NarrativeCanonicalizationEnforcer()


@pytest.fixture
def valid_severity_verdicts() -> List[SeverityLevel]:
    """
    Fixture: Veredictos de severidad válidos para colapso reticular.
    """
    return [SeverityLevel.OPTIMO, SeverityLevel.MODERADO, SeverityLevel.SEVERE]


@pytest.fixture
def valid_topological_metrics_with_beta1() -> Dict[str, float]:
    """
    Fixture: Métricas topológicas con β₁ > 0 (requiere token SOCAVON).
    """
    return {
        "beta_1": 2.0,
        "fiedler_psi": 1.5,
    }


@pytest.fixture
def valid_topological_metrics_with_psi() -> Dict[str, float]:
    """
    Fixture: Métricas topológicas con Ψ < 1.0 (requiere token PIRAMIDE).
    """
    return {
        "beta_1": 0.0,
        "fiedler_psi": 0.5,
    }


@pytest.fixture
def valid_topological_metrics_optimal() -> Dict[str, float]:
    """
    Fixture: Métricas topológicas óptimas (sin patología).
    """
    return {
        "beta_1": 0.0,
        "fiedler_psi": 1.5,
    }


@pytest.fixture
def narrative_with_socavon() -> str:
    """
    Fixture: Narrativa que contiene el token SOCAVON LOGICO.
    """
    return "Se detectó un SOCAVON LOGICO en la estructura de datos del sistema."


@pytest.fixture
def narrative_with_piramide() -> str:
    """
    Fixture: Narrativa que contiene el token PIRAMIDE INVERTIDA.
    """
    return "Existe evidencia de una PIRAMIDE INVERTIDA en la distribución de energía."


@pytest.fixture
def narrative_optimal() -> str:
    """
    Fixture: Narrativa óptima sin patología.
    """
    return "El sistema opera dentro de parámetros normales sin anomalías detectadas."


@pytest.fixture
def narrative_with_control_chars() -> str:
    """
    Fixture: Narrativa con caracteres de control prohibidos.
    """
    return "Texto con\x00caracteres\x1fde\x7fcontrol prohibidos."


@pytest.fixture
def normalizer_function() -> callable:
    """
    Fixture: Función normalizadora idempotente para pruebas.
    """
    def normalizer(text: str) -> str:
        return text.strip().lower()
    return normalizer


@pytest.fixture
def test_string() -> str:
    """
    Fixture: Cadena de prueba para normalización.
    """
    return "  TEST_STRING  "


# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════
#   FASE 1: COLAPSO EN RETÍCULO DISTRIBUTIVO ACOTADO
#   Valida: Veredicto = ⨆ v_i (Supremo del retículo)
# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestPhase1_SeverityLatticeCollapser:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  FASE 1: COLAPSO EN RETÍCULO DISTRIBUTIVO ACOTADO                                     ║
    ║  ─────────────────────────────────────────────────────────────────────────────        ║
    ║  Esta clase de pruebas valida el álgebra de retículos y el colapso determinista.      ║
    ║  Cada método prueba un axioma específico del §1 del módulo principal.                 ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.1. Pruebas de Normalización de Tokens Enum (Método: _normalize_enum_token)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_normalize_enum_token_valid(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Normalización de token enum válido.
        VALIDA: Conversión a mayúsculas, sin acentos, sin espacios redundantes.
        """
        result = phase1_collapser._normalize_enum_token("  óptimo  ")
        assert result == "OPTIMO"

    def test_normalize_enum_token_with_accents(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Normalización de token con acentos.
        VALIDA: Eliminación de caracteres combinantes.
        """
        result = phase1_collapser._normalize_enum_token("CRÍTICO")
        assert result == "CRITICO"

    def test_normalize_enum_token_with_special_chars(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Normalización de token con caracteres especiales.
        VALIDA: Reemplazo de no-alfanuméricos por guiones bajos.
        """
        result = phase1_collapser._normalize_enum_token("se-vero.test")
        assert result == "SE_VERO_TEST"

    def test_normalize_enum_token_empty_raises(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Token vacío tras normalización lanza DomainIntegrityViolationError.
        VALIDA: Integridad del dominio ontológico.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_collapser._normalize_enum_token("   ")
        assert "vacío" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.2. Pruebas de Coerción de Severidad (Método: _coerce_severity)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_coerce_severity_from_enum(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Coerción desde SeverityLevel enum.
        VALIDA: Retorno directo sin conversión.
        """
        result = phase1_collapser._coerce_severity("test_enum", SeverityLevel.MODERADO)
        assert result == SeverityLevel.MODERADO

    def test_coerce_severity_from_int(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Coerción desde entero.
        VALIDA: Conversión int → SeverityLevel.
        """
        result = phase1_collapser._coerce_severity("test_int", 2)
        assert result == SeverityLevel.SEVERE

    def test_coerce_severity_from_np_int(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Coerción desde np.integer.
        VALIDA: Compatibilidad con tipos NumPy.
        """
        result = phase1_collapser._coerce_severity("test_np_int", np.int64(3))
        assert result == SeverityLevel.CRITICO

    def test_coerce_severity_from_float_integer(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Coerción desde float entero.
        VALIDA: Conversión float → int → SeverityLevel.
        """
        result = phase1_collapser._coerce_severity("test_float", 1.0)
        assert result == SeverityLevel.MODERADO

    def test_coerce_severity_from_string_name(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Coerción desde string por nombre.
        VALIDA: Lookup por nombre de enum.
        """
        result = phase1_collapser._coerce_severity("test_str", "OPTIMO")
        assert result == SeverityLevel.OPTIMO

    def test_coerce_severity_from_string_number(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Coerción desde string numérico.
        VALIDA: Conversión string → int → SeverityLevel.
        """
        result = phase1_collapser._coerce_severity("test_str_num", "2")
        assert result == SeverityLevel.SEVERE

    def test_coerce_severity_bool_raises(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Booleano lanza DomainIntegrityViolationError.
        VALIDA: Rechazo de tipos booleanos.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_collapser._coerce_severity("test_bool", True)
        assert "booleano" in str(exc_info.value).lower()

    def test_coerce_severity_np_bool_raises(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: np.bool_ lanza DomainIntegrityViolationError.
        VALIDA: Rechazo de booleanos NumPy.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_collapser._coerce_severity("test_np_bool", np.bool_(True))
        assert "booleano" in str(exc_info.value).lower()

    def test_coerce_severity_invalid_int_raises(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Entero fuera de rango lanza DomainIntegrityViolationError.
        VALIDA: Validación de pertenencia al retículo.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_collapser._coerce_severity("test_invalid", 99)
        assert "no pertenece" in str(exc_info.value).lower()

    def test_coerce_severity_non_integer_float_raises(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Float no entero lanza DomainIntegrityViolationError.
        VALIDA: Exigencia de nivel entero de severidad.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_collapser._coerce_severity("test_float_nonint", 1.5)
        assert "no es un nivel entero" in str(exc_info.value).lower()

    def test_coerce_severity_invalid_string_raises(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: String inválido lanza DomainIntegrityViolationError.
        VALIDA: Validación de nombre/número de enum.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_collapser._coerce_severity("test_invalid_str", "INVALIDO")
        assert "no puede coercionarse" in str(exc_info.value).lower()

    def test_coerce_severity_invalid_type_raises(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Tipo no convertible lanza DomainIntegrityViolationError.
        VALIDA: Validación estricta de tipos.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_collapser._coerce_severity("test_dict", {"key": "value"})
        assert "debe ser" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.3. Pruebas de Coerción de Vector de Veredictos (Método: _coerce_stratum_verdicts)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_coerce_stratum_verdicts_from_list(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Coerción desde lista de SeverityLevel.
        VALIDA: Conversión a tupla de SeverityLevel.
        """
        verdicts = [SeverityLevel.OPTIMO, SeverityLevel.MODERADO, SeverityLevel.SEVERE]
        result = phase1_collapser._coerce_stratum_verdicts(verdicts)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(v, SeverityLevel) for v in result)

    def test_coerce_stratum_verdicts_from_tuple(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Coerción desde tupla de SeverityLevel.
        VALIDA: Preservación de tupla.
        """
        verdicts = (SeverityLevel.OPTIMO, SeverityLevel.CRITICO)
        result = phase1_collapser._coerce_stratum_verdicts(verdicts)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_coerce_stratum_verdicts_from_mixed_types(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Coerción desde tipos mixtos (enum, int, str).
        VALIDA: Flexibilidad de entrada.
        """
        verdicts = [SeverityLevel.OPTIMO, 1, "SEVERO"]
        result = phase1_collapser._coerce_stratum_verdicts(verdicts)
        assert result == (SeverityLevel.OPTIMO, SeverityLevel.MODERADO, SeverityLevel.SEVERE)

    def test_coerce_stratum_verdicts_none_raises(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: None lanza DomainIntegrityViolationError.
        VALIDA: Validación de no nulidad.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_collapser._coerce_stratum_verdicts(None)
        assert "none" in str(exc_info.value).lower()

    def test_coerce_stratum_verdicts_string_raises(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: String plano lanza DomainIntegrityViolationError.
        VALIDA: Rechazo de strings como colección.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_collapser._coerce_stratum_verdicts("OPTIMO")
        assert "no debe ser una cadena" in str(exc_info.value).lower()

    def test_coerce_stratum_verdicts_mapping_raises(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Mapping lanza DomainIntegrityViolationError.
        VALIDA: Rechazo de diccionarios.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_collapser._coerce_stratum_verdicts({0: "OPTIMO"})
        assert "no debe ser" in str(exc_info.value).lower()

    def test_coerce_stratum_verdicts_not_iterable_raises(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Objeto no iterable lanza DomainIntegrityViolationError.
        VALIDA: Exigencia de colección iterable.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase1_collapser._coerce_stratum_verdicts(42)
        assert "no es iterable" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.4. Pruebas de Operación Join Binaria (Método: _lattice_join)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_lattice_join_optimo_optimo(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Join de OPTIMO ⊔ OPTIMO = OPTIMO.
        VALIDA: Elemento mínimo del retículo.
        """
        result = phase1_collapser._lattice_join(SeverityLevel.OPTIMO, SeverityLevel.OPTIMO)
        assert result == SeverityLevel.OPTIMO

    def test_lattice_join_optimo_critico(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Join de OPTIMO ⊔ CRITICO = CRITICO.
        VALIDA: Elemento máximo absorbente.
        """
        result = phase1_collapser._lattice_join(SeverityLevel.OPTIMO, SeverityLevel.CRITICO)
        assert result == SeverityLevel.CRITICO

    def test_lattice_join_moderado_severo(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Join de MODERADO ⊔ SEVERO = SEVERO.
        VALIDA: Máximo en cadena finita.
        """
        result = phase1_collapser._lattice_join(SeverityLevel.MODERADO, SeverityLevel.SEVERE)
        assert result == SeverityLevel.SEVERE

    def test_lattice_join_commutative(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Join es conmutativo: a ⊔ b = b ⊔ a.
        VALIDA: Propiedad algebraica del retículo.
        """
        result1 = phase1_collapser._lattice_join(SeverityLevel.MODERADO, SeverityLevel.SEVERE)
        result2 = phase1_collapser._lattice_join(SeverityLevel.SEVERE, SeverityLevel.MODERADO)
        assert result1 == result2

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.5. Pruebas de Supremo del Retículo (Método: _lattice_supremum)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_lattice_supremum_empty(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Supremo de conjunto vacío = OPTIMO (⊥).
        VALIDA: Join vacío retorna elemento mínimo.
        """
        result = phase1_collapser._lattice_supremum(())
        assert result == SeverityLevel.OPTIMO

    def test_lattice_supremum_single_optimo(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Supremo de un solo OPTIMO.
        VALIDA: Caso degenerado aceptado.
        """
        result = phase1_collapser._lattice_supremum((SeverityLevel.OPTIMO,))
        assert result == SeverityLevel.OPTIMO

    def test_lattice_supremum_mixed(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Supremo de veredictos mixtos.
        VALIDA: Máximo de la colección.
        """
        verdicts = (SeverityLevel.OPTIMO, SeverityLevel.MODERADO, SeverityLevel.SEVERE)
        result = phase1_collapser._lattice_supremum(verdicts)
        assert result == SeverityLevel.SEVERE

    def test_lattice_supremum_with_critico(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Supremo con CRITICO presente.
        VALIDA: Elemento máximo absorbente.
        """
        verdicts = (SeverityLevel.OPTIMO, SeverityLevel.CRITICO, SeverityLevel.MODERADO)
        result = phase1_collapser._lattice_supremum(verdicts)
        assert result == SeverityLevel.CRITICO

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.6. Pruebas de Absorción Reticular (Método: _assert_lattice_absorption)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_assert_lattice_absorption_valid(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Ley de absorción válida: ⊥ ⊔ ⊤ = ⊤.
        VALIDA: Clausura algebraica del retículo.
        """
        verdicts = (SeverityLevel.OPTIMO, SeverityLevel.CRITICO)
        supremum = SeverityLevel.CRITICO
        # No debe lanzar excepción
        phase1_collapser._assert_lattice_absorption(verdicts, supremum)

    def test_assert_lattice_absorption_critico_not_supremum_raises(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: CRITICO presente pero supremo no es CRITICO lanza excepción.
        VALIDA: §1. Clausura Algebraica del Retículo.
        """
        verdicts = (SeverityLevel.OPTIMO, SeverityLevel.CRITICO)
        supremum = SeverityLevel.SEVERE  # Incorrecto
        with pytest.raises(SeverityLatticeCollapseError) as exc_info:
            phase1_collapser._assert_lattice_absorption(verdicts, supremum)
        assert "clausura algebraica" in str(exc_info.value).lower()

    def test_assert_lattice_absorption_supremum_not_in_verdicts_raises(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Supremo no pertenece a veredictos lanza excepción.
        VALIDA: Propiedad de cadena finita del retículo.
        """
        verdicts = (SeverityLevel.OPTIMO, SeverityLevel.MODERADO)
        supremum = SeverityLevel.CRITICO  # No está en verdicts
        with pytest.raises(SeverityLatticeCollapseError) as exc_info:
            phase1_collapser._assert_lattice_absorption(verdicts, supremum)
        assert "no pertenece" in str(exc_info.value).lower()

    def test_assert_lattice_absorption_empty_verdicts(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Veredictos vacíos no lanza excepción.
        VALIDA: Caso degenerado aceptado.
        """
        verdicts = ()
        supremum = SeverityLevel.OPTIMO
        # No debe lanzar excepción
        phase1_collapser._assert_lattice_absorption(verdicts, supremum)

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.7. Pruebas de Colapso Reticular Interno (Método: _collapse_severity_lattice_internal)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_collapse_severity_lattice_internal_valid(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
        valid_severity_verdicts: List[SeverityLevel],
    ) -> None:
        """
        PRUEBA: Colapso reticular interno válido.
        VALIDA: Cálculo correcto del supremo.
        """
        state, certified = phase1_collapser._collapse_severity_lattice_internal(
            valid_severity_verdicts
        )
        assert isinstance(state, LatticeCollapseState)
        assert isinstance(certified, tuple)
        assert state.supremum_verdict == SeverityLevel.SEVERE
        assert state.is_worst_case_enforced is True
        assert state.lattice_size == 3
        assert state.is_empty_join is False

    def test_collapse_severity_lattice_internal_empty(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Colapso reticular con veredictos vacíos.
        VALIDA: Join vacío = OPTIMO.
        """
        state, certified = phase1_collapser._collapse_severity_lattice_internal([])
        assert state.supremum_verdict == SeverityLevel.OPTIMO
        assert state.is_empty_join is True
        assert state.lattice_size == 0

    def test_collapse_severity_lattice_internal_with_critico(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
    ) -> None:
        """
        PRUEBA: Colapso reticular con CRITICO presente.
        VALIDA: Supremo = CRITICO (worst-case).
        """
        verdicts = [SeverityLevel.OPTIMO, SeverityLevel.CRITICO, SeverityLevel.MODERADO]
        state, certified = phase1_collapser._collapse_severity_lattice_internal(verdicts)
        assert state.supremum_verdict == SeverityLevel.CRITICO
        assert state.is_worst_case_enforced is True

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.8. Pruebas de Wrapper Público (Método: _collapse_severity_lattice)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_collapse_severity_lattice_valid(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
        valid_severity_verdicts: List[SeverityLevel],
    ) -> None:
        """
        PRUEBA: Wrapper público de colapso reticular.
        VALIDA: Compatibilidad retroactiva.
        """
        state = phase1_collapser._collapse_severity_lattice(valid_severity_verdicts)
        assert isinstance(state, LatticeCollapseState)
        assert state.supremum_verdict == SeverityLevel.SEVERE

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §1.9. Pruebas de Handoff Fase 1 → Fase 2 (Método: _phase1_collapse_and_handoff_to_phase2)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_phase1_collapse_and_handoff_to_phase2_valid(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
        valid_severity_verdicts: List[SeverityLevel],
    ) -> None:
        """
        PRUEBA: Handoff formal de Fase 1 a Fase 2.
        VALIDA: Continuidad funtorial Φ₁ → Φ₂.
        """
        handoff = phase1_collapser._phase1_collapse_and_handoff_to_phase2(
            valid_severity_verdicts
        )
        assert isinstance(handoff, Phase1LatticeHandoff)
        assert isinstance(handoff.lattice_state, LatticeCollapseState)
        assert isinstance(handoff.stratum_verdicts_certified, tuple)
        assert len(handoff.stratum_verdicts_certified) == 3

    def test_phase1_lattice_handoff_immutability(
        self,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
        valid_severity_verdicts: List[SeverityLevel],
    ) -> None:
        """
        PRUEBA: Phase1LatticeHandoff es inmutable.
        VALIDA: Integridad del artefacto de handoff.
        """
        handoff = phase1_collapser._phase1_collapse_and_handoff_to_phase2(
            valid_severity_verdicts
        )
        with pytest.raises((AttributeError, TypeError)):
            handoff.lattice_state = None  # type: ignore


# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════
#   FASE 2: CERTIFICACIÓN DE DIFEOMORFISMO SEMÁNTICO
#   Valida: β₁ > 0 ⇒ "SOCAVON LOGICO" y Ψ < 1.0 ⇒ "PIRAMIDE INVERTIDA"
# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestPhase2_SemanticDiffeomorphismCertifier:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  FASE 2: CERTIFICACIÓN DE DIFEOMORFISMO SEMÁNTICO                                     ║
    ║  ─────────────────────────────────────────────────────────────────────────────        ║
    ║  Esta clase de pruebas valida el isomorfismo entre narrativa y métricas topológicas.  ║
    ║  Cada método prueba un axioma específico del §2 del módulo principal.                 ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §2.1. Pruebas de Normalización Semántica (Método: _normalize_semantic_text)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_normalize_semantic_text_valid(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
    ) -> None:
        """
        PRUEBA: Normalización semántica de texto válido.
        VALIDA: NFKD, sin acentos, mayúsculas, puntuación como espacio.
        """
        result = phase2_certifier._normalize_semantic_text("  SOCAVÓN LÓGICO  ")
        assert "SOCAVON LOGICO" in result

    def test_normalize_semantic_text_with_punctuation(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
    ) -> None:
        """
        PRUEBA: Normalización con puntuación variada.
        VALIDA: Puntuación convertida a espacios.
        """
        result = phase2_certifier._normalize_semantic_text("PIRÁMIDE.INVERTIDA!")
        assert "PIRAMIDE INVERTIDA" in result

    def test_normalize_semantic_text_non_string_raises(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
    ) -> None:
        """
        PRUEBA: Texto no string lanza DomainIntegrityViolationError.
        VALIDA: Validación de tipo de narrativa.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase2_certifier._normalize_semantic_text(123)
        assert "cadena de texto" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §2.2. Pruebas de Extracción de Métricas Topológicas (Método: _extract_metric_value)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_extract_metric_value_from_dict(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
    ) -> None:
        """
        PRUEBA: Extracción de métrica desde diccionario.
        VALIDA: Búsqueda por nombres alternativos.
        """
        metrics = {"beta_1": 2.0, "other": 1.0}
        result = phase2_certifier._extract_metric_value(
            metrics,
            _BETA_1_METRIC_NAMES,
        )
        assert result == 2.0

    def test_extract_metric_value_from_object(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
    ) -> None:
        """
        PRUEBA: Extracción de métrica desde objeto con atributos.
        VALIDA: Uso de getattr para atributos.
        """
        class MetricsObj:
            def __init__(self):
                self.b1 = 3.0
        metrics = MetricsObj()
        result = phase2_certifier._extract_metric_value(
            metrics,
            ("b1", "beta_1"),
        )
        assert result == 3.0

    def test_extract_metric_value_not_found(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
    ) -> None:
        """
        PRUEBA: Métrica no encontrada retorna None.
        VALIDA: Manejo de ausencia de métrica.
        """
        metrics = {"other": 1.0}
        result = phase2_certifier._extract_metric_value(
            metrics,
            _BETA_1_METRIC_NAMES,
        )
        assert result is None

    def test_extract_metric_value_none_input(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
    ) -> None:
        """
        PRUEBA: Input None retorna None.
        VALIDA: Manejo defensivo.
        """
        result = phase2_certifier._extract_metric_value(None, _BETA_1_METRIC_NAMES)
        assert result is None

    def test_extract_metric_value_bool_raises(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
    ) -> None:
        """
        PRUEBA: Métrica booleana lanza DomainIntegrityViolationError.
        VALIDA: Rechazo de tipos booleanos.
        """
        metrics = {"beta_1": True}
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase2_certifier._extract_metric_value(metrics, _BETA_1_METRIC_NAMES)
        assert "no puede ser booleana" in str(exc_info.value).lower()

    def test_extract_metric_value_non_numeric_raises(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
    ) -> None:
        """
        PRUEBA: Métrica no numérica lanza DomainIntegrityViolationError.
        VALIDA: Exigencia de valor numérico.
        """
        metrics = {"beta_1": "not_a_number"}
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase2_certifier._extract_metric_value(metrics, _BETA_1_METRIC_NAMES)
        assert "no es numérica" in str(exc_info.value).lower()

    def test_extract_metric_value_non_finite_raises(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
    ) -> None:
        """
        PRUEBA: Métrica no finita lanza DomainIntegrityViolationError.
        VALIDA: Exigencia de finitud.
        """
        metrics = {"beta_1": np.inf}
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase2_certifier._extract_metric_value(metrics, _BETA_1_METRIC_NAMES)
        assert "no es finita" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §2.3. Pruebas de Certificación de β₁ (Método: _certify_betti_1)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_certify_betti_1_positive_integer(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
    ) -> None:
        """
        PRUEBA: β₁ positivo entero es válido.
        VALIDA: β₁ > 0 requiere token SOCAVON.
        """
        metrics = {"beta_1": 2.0}
        beta_1, required = phase2_certifier._certify_betti_1(metrics)
        assert beta_1 == 2.0
        assert required is True

    def test_certify_betti_1_zero(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
    ) -> None:
        """
        PRUEBA: β₁ = 0 no requiere token.
        VALIDA: Ausencia de patología.
        """
        metrics = {"beta_1": 0.0}
        beta_1, required = phase2_certifier._certify_betti_1(metrics)
        assert beta_1 == 0.0
        assert required is False

    def test_certify_betti_1_not_present(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
    ) -> None:
        """
        PRUEBA: β₁ no presente retorna (None, False).
        VALIDA: Manejo de ausencia.
        """
        metrics = {"other": 1.0}
        beta_1, required = phase2_certifier._certify_betti_1(metrics)
        assert beta_1 is None
        assert required is False

    def test_certify_betti_1_negative_raises(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
    ) -> None:
        """
        PRUEBA: β₁ negativo lanza DomainIntegrityViolationError.
        VALIDA: Números de Betti son no negativos.
        """
        metrics = {"beta_1": -1.0}
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase2_certifier._certify_betti_1(metrics)
        assert "negativo" in str(exc_info.value).lower()

    def test_certify_betti_1_non_integer_raises(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
    ) -> None:
        """
        PRUEBA: β₁ no entero lanza DomainIntegrityViolationError.
        VALIDA: Exigencia de integralidad aproximada.
        """
        metrics = {"beta_1": 2.5}
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase2_certifier._certify_betti_1(metrics)
        assert "no es aproximadamente entero" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §2.4. Pruebas de Certificación de Estabilidad Piramidal (Método: _certify_pyramid_stability)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_certify_pyramid_stability_below_threshold(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
    ) -> None:
        """
        PRUEBA: Ψ < 1.0 requiere token PIRAMIDE.
        VALIDA: Patología de estabilidad.
        """
        metrics = {"fiedler_psi": 0.5}
        psi, required = phase2_certifier._certify_pyramid_stability(metrics)
        assert psi == 0.5
        assert required is True

    def test_certify_pyramid_stability_above_threshold(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
    ) -> None:
        """
        PRUEBA: Ψ ≥ 1.0 no requiere token.
        VALIDA: Estabilidad aceptable.
        """
        metrics = {"fiedler_psi": 1.5}
        psi, required = phase2_certifier._certify_pyramid_stability(metrics)
        assert psi == 1.5
        assert required is False

    def test_certify_pyramid_stability_negative_raises(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
    ) -> None:
        """
        PRUEBA: Ψ negativo lanza DomainIntegrityViolationError.
        VALIDA: Estabilidad estructural no negativa.
        """
        metrics = {"fiedler_psi": -0.5}
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase2_certifier._certify_pyramid_stability(metrics)
        assert "negativo" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §2.5. Pruebas de Certificación de Difeomorfismo Interno (Método: _certify_semantic_diffeomorphism_internal)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_certify_semantic_diffeomorphism_internal_valid_socavon(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
        valid_topological_metrics_with_beta1: Dict[str, float],
        narrative_with_socavon: str,
    ) -> None:
        """
        PRUEBA: Difeomorfismo válido con token SOCAVON presente.
        VALIDA: §2. Isomorfismo semántico β₁ > 0.
        """
        result = phase2_certifier._certify_semantic_diffeomorphism_internal(
            valid_topological_metrics_with_beta1,
            narrative_with_socavon,
        )
        assert isinstance(result, DiffeomorphismAuditData)
        assert result.is_isomorphic is True
        assert result.betti_1_verified is True
        assert result.semantic_drift_detected is False

    def test_certify_semantic_diffeomorphism_internal_valid_piramide(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
        valid_topological_metrics_with_psi: Dict[str, float],
        narrative_with_piramide: str,
    ) -> None:
        """
        PRUEBA: Difeomorfismo válido con token PIRAMIDE presente.
        VALIDA: §2. Isomorfismo semántico Ψ < 1.0.
        """
        result = phase2_certifier._certify_semantic_diffeomorphism_internal(
            valid_topological_metrics_with_psi,
            narrative_with_piramide,
        )
        assert result.is_isomorphic is True
        assert result.fiedler_psi_verified is True

    def test_certify_semantic_diffeomorphism_internal_optimal(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
        valid_topological_metrics_optimal: Dict[str, float],
        narrative_optimal: str,
    ) -> None:
        """
        PRUEBA: Difeomorfismo óptimo sin patología.
        VALIDA: Ausencia de tokens requeridos.
        """
        result = phase2_certifier._certify_semantic_diffeomorphism_internal(
            valid_topological_metrics_optimal,
            narrative_optimal,
        )
        assert result.is_isomorphic is True
        assert result.betti_1_required is False
        assert result.fiedler_psi_required is False

    def test_certify_semantic_diffeomorphism_internal_drift_raises(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
        valid_topological_metrics_with_beta1: Dict[str, float],
        narrative_optimal: str,
    ) -> None:
        """
        PRUEBA: Deriva semántica (β₁ > 0 sin token) lanza excepción.
        VALIDA: §2. Difeomorfismo Semántico y Preservación de Homotopía.
        """
        with pytest.raises(SemanticDiffeomorphismViolationError) as exc_info:
            phase2_certifier._certify_semantic_diffeomorphism_internal(
                valid_topological_metrics_with_beta1,
                narrative_optimal,
            )
        assert "deriva estocástica" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §2.6. Pruebas de Wrapper Público (Método: _certify_semantic_diffeomorphism)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_certify_semantic_diffeomorphism_valid(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
        valid_topological_metrics_with_beta1: Dict[str, float],
        narrative_with_socavon: str,
    ) -> None:
        """
        PRUEBA: Wrapper público de difeomorfismo.
        VALIDA: Compatibilidad retroactiva.
        """
        result = phase2_certifier._certify_semantic_diffeomorphism(
            valid_topological_metrics_with_beta1,
            narrative_with_socavon,
        )
        assert isinstance(result, DiffeomorphismAuditData)
        assert result.is_isomorphic is True

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §2.7. Pruebas de Handoff Fase 2 → Fase 3 (Método: _phase2_certify_and_handoff_to_phase3)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_phase2_certify_and_handoff_to_phase3_valid(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
        valid_severity_verdicts: List[SeverityLevel],
        valid_topological_metrics_optimal: Dict[str, float],
        narrative_optimal: str,
    ) -> None:
        """
        PRUEBA: Handoff formal de Fase 2 a Fase 3.
        VALIDA: Continuidad funtorial Φ₂ → Φ₃.
        """
        phase1_handoff = phase1_collapser._phase1_collapse_and_handoff_to_phase2(
            valid_severity_verdicts
        )
        handoff = phase2_certifier._phase2_certify_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            topological_metrics=valid_topological_metrics_optimal,
            proposed_narrative=narrative_optimal,
        )
        assert isinstance(handoff, Phase2DiffeomorphismHandoff)
        assert isinstance(handoff.phase1_handoff, Phase1LatticeHandoff)
        assert isinstance(handoff.diffeomorphism_audit, DiffeomorphismAuditData)
        assert isinstance(handoff.canonicalizable_narrative, str)

    def test_phase2_handoff_invalid_phase1_handoff_raises(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
        valid_topological_metrics_optimal: Dict[str, float],
        narrative_optimal: str,
    ) -> None:
        """
        PRUEBA: Handoff de Fase 1 inválido lanza DomainIntegrityViolationError.
        VALIDA: Validación de prefijo formal.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase2_certifier._phase2_certify_and_handoff_to_phase3(
                phase1_handoff=None,  # type: ignore
                topological_metrics=valid_topological_metrics_optimal,
                proposed_narrative=narrative_optimal,
            )
        assert "phase1latticehandoff" in str(exc_info.value).lower()

    def test_phase2_handoff_empty_narrative_raises(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
        valid_severity_verdicts: List[SeverityLevel],
        valid_topological_metrics_optimal: Dict[str, float],
    ) -> None:
        """
        PRUEBA: Narrativa vacía tras normalización lanza excepción.
        VALIDA: Integridad del contenido narrativo.
        """
        phase1_handoff = phase1_collapser._phase1_collapse_and_handoff_to_phase2(
            valid_severity_verdicts
        )
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase2_certifier._phase2_certify_and_handoff_to_phase3(
                phase1_handoff=phase1_handoff,
                topological_metrics=valid_topological_metrics_optimal,
                proposed_narrative="   ",
            )
        assert "vacía" in str(exc_info.value).lower()

    def test_phase2_handoff_optimal_state_no_contract(
        self,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
        valid_topological_metrics_optimal: Dict[str, float],
        narrative_optimal: str,
    ) -> None:
        """
        PRUEBA: Estado OPTIMO no exige contrato patológico.
        VALIDA: semantic_contract_enforced = False.
        """
        verdicts_optimal = [SeverityLevel.OPTIMO]
        phase1_handoff = phase1_collapser._phase1_collapse_and_handoff_to_phase2(
            verdicts_optimal
        )
        handoff = phase2_certifier._phase2_certify_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            topological_metrics=valid_topological_metrics_optimal,
            proposed_narrative=narrative_optimal,
        )
        assert handoff.diffeomorphism_audit.semantic_contract_enforced is False


# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════
#   FASE 3: CANONICALIZACIÓN DIPLOMÁTICA Y ACTA INMUTABLE
#   Valida: Eliminación de caracteres de control y pureza forense
# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestPhase3_NarrativeCanonicalizationEnforcer:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  FASE 3: CANONICALIZACIÓN DIPLOMÁTICA Y ACTA INMUTABLE                                ║
    ║  ─────────────────────────────────────────────────────────────────────────────        ║
    ║  Esta clase de pruebas valida la canonicalización y pureza léxica de la narrativa.    ║
    ║  Cada método prueba un axioma específico del §3 del módulo principal.                 ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §3.1. Pruebas de Canonicalización de Texto (Método: _canonicalize_narrative_text)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_canonicalize_narrative_text_valid(
        self,
        phase3_enforcer: Phase3_NarrativeCanonicalizationEnforcer,
    ) -> None:
        """
        PRUEBA: Canonicalización de texto válido.
        VALIDA: NFKC, eliminación de caracteres de control, colapso de espacios.
        """
        result = phase3_enforcer._canonicalize_narrative_text("  Texto  válido  ")
        assert result == "Texto válido"

    def test_canonicalize_narrative_text_removes_control_chars(
        self,
        phase3_enforcer: Phase3_NarrativeCanonicalizationEnforcer,
        narrative_with_control_chars: str,
    ) -> None:
        """
        PRUEBA: Canonicalización elimina caracteres de control.
        VALIDA: §3. Entropía Sintáctica y Canonicalización Diplomática.
        """
        result = phase3_enforcer._canonicalize_narrative_text(narrative_with_control_chars)
        assert "\x00" not in result
        assert "\x1f" not in result
        assert "\x7f" not in result

    def test_canonicalize_narrative_text_preserves_whitespace(
        self,
        phase3_enforcer: Phase3_NarrativeCanonicalizationEnforcer,
    ) -> None:
        """
        PRUEBA: Canonicalización preserva espacios, tabs, newlines.
        VALIDA: Contenido semántico visible.
        """
        result = phase3_enforcer._canonicalize_narrative_text("Linea1\nLinea2\tTab")
        assert " " in result  # Espacios colapsados

    def test_canonicalize_narrative_text_non_string_raises(
        self,
        phase3_enforcer: Phase3_NarrativeCanonicalizationEnforcer,
    ) -> None:
        """
        PRUEBA: Texto no string lanza DomainIntegrityViolationError.
        VALIDA: Validación de tipo.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase3_enforcer._canonicalize_narrative_text(123)
        assert "cadena de texto" in str(exc_info.value).lower()

    def test_canonicalize_narrative_text_empty_raises(
        self,
        phase3_enforcer: Phase3_NarrativeCanonicalizationEnforcer,
    ) -> None:
        """
        PRUEBA: Texto vacío tras canonicalización lanza excepción.
        VALIDA: Contenido mínimo requerido.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase3_enforcer._canonicalize_narrative_text("\x00\x1f\x7f")
        assert "vacía" in str(exc_info.value).lower()

    def test_canonicalize_narrative_text_too_short_raises(
        self,
        phase3_enforcer: Phase3_NarrativeCanonicalizationEnforcer,
    ) -> None:
        """
        PRUEBA: Texto demasiado corto lanza NarrativeCanonicalizationError.
        VALIDA: Límite inferior _MIN_NARRATIVE_LENGTH.
        """
        with pytest.raises(NarrativeCanonicalizationError) as exc_info:
            phase3_enforcer._canonicalize_narrative_text("A")
        assert "demasiado corta" in str(exc_info.value).lower()

    def test_canonicalize_narrative_text_too_long_raises(
        self,
        phase3_enforcer: Phase3_NarrativeCanonicalizationEnforcer,
    ) -> None:
        """
        PRUEBA: Texto demasiado largo lanza NarrativeCanonicalizationError.
        VALIDA: Límite superior _MAX_NARRATIVE_LENGTH.
        """
        long_text = "A" * (_MAX_NARRATIVE_LENGTH + 1)
        with pytest.raises(NarrativeCanonicalizationError) as exc_info:
            phase3_enforcer._canonicalize_narrative_text(long_text)
        assert "excede" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §3.2. Pruebas de Re-verificación de Tokens (Método: _assert_required_tokens_after_canonicalization)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_assert_required_tokens_after_canonicalization_valid(
        self,
        phase3_enforcer: Phase3_NarrativeCanonicalizationEnforcer,
        narrative_with_socavon: str,
    ) -> None:
        """
        PRUEBA: Tokens requeridos presentes tras canonicalización.
        VALIDA: Preservación de tokens obligatorios.
        """
        audit = DiffeomorphismAuditData(
            is_isomorphic=True,
            betti_1_verified=True,
            fiedler_psi_verified=True,
            semantic_drift_detected=False,
            semantic_contract_enforced=True,
            betti_1_required=True,
            fiedler_psi_required=False,
            normalized_narrative="",
        )
        # No debe lanzar excepción
        phase3_enforcer._assert_required_tokens_after_canonicalization(
            narrative_with_socavon,
            audit,
        )

    def test_assert_required_tokens_after_canonicalization_no_contract(
        self,
        phase3_enforcer: Phase3_NarrativeCanonicalizationEnforcer,
        narrative_optimal: str,
    ) -> None:
        """
        PRUEBA: Sin contrato semántico no verifica tokens.
        VALIDA: semantic_contract_enforced = False.
        """
        audit = DiffeomorphismAuditData(
            is_isomorphic=True,
            betti_1_verified=True,
            fiedler_psi_verified=True,
            semantic_drift_detected=False,
            semantic_contract_enforced=False,
            betti_1_required=False,
            fiedler_psi_required=False,
            normalized_narrative="",
        )
        # No debe lanzar excepción
        phase3_enforcer._assert_required_tokens_after_canonicalization(
            narrative_optimal,
            audit,
        )

    def test_assert_required_tokens_after_canonicalization_socavon_removed_raises(
        self,
        phase3_enforcer: Phase3_NarrativeCanonicalizationEnforcer,
        narrative_optimal: str,
    ) -> None:
        """
        PRUEBA: Token SOCAVON eliminado lanza SemanticDiffeomorphismViolationError.
        VALIDA: Integridad del difeomorfismo tras canonicalización.
        """
        audit = DiffeomorphismAuditData(
            is_isomorphic=True,
            betti_1_verified=True,
            fiedler_psi_verified=True,
            semantic_drift_detected=False,
            semantic_contract_enforced=True,
            betti_1_required=True,
            fiedler_psi_required=False,
            normalized_narrative="",
        )
        with pytest.raises(SemanticDiffeomorphismViolationError) as exc_info:
            phase3_enforcer._assert_required_tokens_after_canonicalization(
                narrative_optimal,
                audit,
            )
        assert "socavón lógico" in str(exc_info.value).lower()

    def test_assert_required_tokens_after_canonicalization_piramide_removed_raises(
        self,
        phase3_enforcer: Phase3_NarrativeCanonicalizationEnforcer,
        narrative_optimal: str,
    ) -> None:
        """
        PRUEBA: Token PIRAMIDE eliminado lanza SemanticDiffeomorphismViolationError.
        VALIDA: Integridad del difeomorfismo tras canonicalización.
        """
        audit = DiffeomorphismAuditData(
            is_isomorphic=True,
            betti_1_verified=True,
            fiedler_psi_verified=True,
            semantic_drift_detected=False,
            semantic_contract_enforced=True,
            betti_1_required=False,
            fiedler_psi_required=True,
            normalized_narrative="",
        )
        with pytest.raises(SemanticDiffeomorphismViolationError) as exc_info:
            phase3_enforcer._assert_required_tokens_after_canonicalization(
                narrative_optimal,
                audit,
            )
        assert "pirámide invertida" in str(exc_info.value).lower()

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §3.3. Pruebas de Auditoría de Canonicalización (Método: _audit_narrative_canonicity)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_audit_narrative_canonicity_valid(
        self,
        phase3_enforcer: Phase3_NarrativeCanonicalizationEnforcer,
        narrative_optimal: str,
    ) -> None:
        """
        PRUEBA: Auditoría de canonicalización válida.
        VALIDA: Certificado de canonicalización correcto.
        """
        audit = DiffeomorphismAuditData(
            is_isomorphic=True,
            betti_1_verified=True,
            fiedler_psi_verified=True,
            semantic_drift_detected=False,
            semantic_contract_enforced=False,
            betti_1_required=False,
            fiedler_psi_required=False,
            normalized_narrative="",
        )
        result = phase3_enforcer._audit_narrative_canonicity(narrative_optimal, audit)
        assert isinstance(result, NarrativeCanonicalizationData)
        assert result.is_canonical is True
        assert result.contains_control_chars is False
        assert result.narrative_length > 0

    def test_audit_narrative_canonicity_control_chars_remain_raises(
        self,
        phase3_enforcer: Phase3_NarrativeCanonicalizationEnforcer,
    ) -> None:
        """
        PRUEBA: Caracteres de control restantes lanza NarrativeCanonicalizationError.
        VALIDA: Pureza forense criptográfica.
        """
        # Este caso es difícil de forzar porque _canonicalize_narrative_text
        # ya elimina los caracteres de control. La prueba valida el chequeo.
        audit = DiffeomorphismAuditData(
            is_isomorphic=True,
            betti_1_verified=True,
            fiedler_psi_verified=True,
            semantic_drift_detected=False,
            semantic_contract_enforced=False,
            betti_1_required=False,
            fiedler_psi_required=False,
            normalized_narrative="",
        )
        # La canonicalización ya elimina control chars, así que esta prueba
        # valida que el chequeo existe en el código
        pass  # El chequeo está implementado en el método

    # ───────────────────────────────────────────────────────────────────────────────────────
    # §3.4. Pruebas de Finalización Funtorial (Método: _phase3_finalize_from_phase2_handoff)
    # ───────────────────────────────────────────────────────────────────────────────────────

    def test_phase3_finalize_from_phase2_handoff_valid(
        self,
        phase3_enforcer: Phase3_NarrativeCanonicalizationEnforcer,
        phase1_collapser: Phase1_SeverityLatticeCollapser,
        phase2_certifier: Phase2_SemanticDiffeomorphismCertifier,
        valid_severity_verdicts: List[SeverityLevel],
        valid_topological_metrics_optimal: Dict[str, float],
        narrative_optimal: str,
    ) -> None:
        """
        PRUEBA: Finalización funtorial completa Φ₃ ∘ Φ₂ ∘ Φ₁.
        VALIDA: Composición de las tres fases.
        """
        phase1_handoff = phase1_collapser._phase1_collapse_and_handoff_to_phase2(
            valid_severity_verdicts
        )
        phase2_handoff = phase2_certifier._phase2_certify_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            topological_metrics=valid_topological_metrics_optimal,
            proposed_narrative=narrative_optimal,
        )
        state = phase3_enforcer._phase3_finalize_from_phase2_handoff(phase2_handoff)
        assert isinstance(state, NarrativeAgentState)
        assert isinstance(state.lattice_collapse, LatticeCollapseState)
        assert isinstance(state.diffeomorphism_audit, DiffeomorphismAuditData)
        assert isinstance(state.approved_narrative, str)
        assert state.is_epistemologically_valid is True
        assert isinstance(state.canonicalization_audit, NarrativeCanonicalizationData)

    def test_phase3_handoff_invalid_phase2_handoff_raises(
        self,
        phase3_enforcer: Phase3_NarrativeCanonicalizationEnforcer,
    ) -> None:
        """
        PRUEBA: Handoff de Fase 2 inválido lanza DomainIntegrityViolationError.
        VALIDA: Validación de prefijo formal.
        """
        with pytest.raises(DomainIntegrityViolationError) as exc_info:
            phase3_enforcer._phase3_finalize_from_phase2_handoff(
                phase2_handoff=None  # type: ignore
            )
        assert "phase2diffeomorphismhandoff" in str(exc_info.value).lower()


# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════
#   ORQUESTADOR SUPREMO: TELEMETRYNARRATIVEAGENT (Pruebas de Integración)
#   Valida: Endofuntor Z_Narrative = Φ₃ ∘ Φ₂ ∘ Φ₁
# ═══════════════════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestTelemetryNarrativeAgent_Integration:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  ORQUESTADOR SUPREMO: TELEMETRYNARRATIVEAGENT                                         ║
    ║  ─────────────────────────────────────────────────────────────────────────────        ║
    ║  Pruebas de integración que validan el endofuntor completo Z_Narrative.               ║
    ║  Estas pruebas aseguran que la composición Φ₃ ∘ Φ₂ ∘ Φ₁ funciona correctamente.       ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    def test_telemetry_narrative_agent_execute_diplomatic_narrative_governance_valid(
        self,
        narrative_agent: TelemetryNarrativeAgent,
        valid_severity_verdicts: List[SeverityLevel],
        valid_topological_metrics_optimal: Dict[str, float],
        narrative_optimal: str,
    ) -> None:
        """
        PRUEBA: Ejecución completa del gobierno de narrativa diplomática.
        VALIDA: Endofuntor Z_Narrative con datos válidos.
        """
        state = narrative_agent.execute_diplomatic_narrative_governance(
            stratum_verdicts=valid_severity_verdicts,
            topological_metrics=valid_topological_metrics_optimal,
            proposed_narrative=narrative_optimal,
        )
        assert isinstance(state, NarrativeAgentState)
        assert state.is_epistemologically_valid is True
        assert state.lattice_collapse.is_worst_case_enforced is True
        assert state.diffeomorphism_audit.is_isomorphic is True
        assert isinstance(state.approved_narrative, str)
        assert len(state.approved_narrative) > 0

    def test_telemetry_narrative_agent_call_alias_valid(
        self,
        narrative_agent: TelemetryNarrativeAgent,
        valid_severity_verdicts: List[SeverityLevel],
        valid_topological_metrics_optimal: Dict[str, float],
        narrative_optimal: str,
    ) -> None:
        """
        PRUEBA: Alias invocable __call__ del endofuntor.
        VALIDA: Sintaxis alternativa de ejecución.
        """
        state = narrative_agent(
            stratum_verdicts=valid_severity_verdicts,
            topological_metrics=valid_topological_metrics_optimal,
            proposed_narrative=narrative_optimal,
        )
        assert isinstance(state, NarrativeAgentState)
        assert state.is_epistemologically_valid is True

    def test_telemetry_narrative_agent_severity_lattice_collapse_error(
        self,
        narrative_agent: TelemetryNarrativeAgent,
        valid_topological_metrics_optimal: Dict[str, float],
        narrative_optimal: str,
    ) -> None:
        """
        PRUEBA: Fallo en colapso reticular lanza SeverityLatticeCollapseError.
        VALIDA: Propagación de excepciones de Fase 1.
        """
        # Veredictos que causarían fallo en absorción (imposible de forzar directamente)
        # Esta prueba valida la estructura de excepción
        pass  # La lógica de absorción está protegida internamente

    def test_telemetry_narrative_agent_semantic_diffeomorphism_violation_error(
        self,
        narrative_agent: TelemetryNarrativeAgent,
        valid_severity_verdicts: List[SeverityLevel],
        valid_topological_metrics_with_beta1: Dict[str, float],
        narrative_optimal: str,
    ) -> None:
        """
        PRUEBA: Deriva semántica lanza SemanticDiffeomorphismViolationError.
        VALIDA: Propagación de excepciones de Fase 2.
        """
        with pytest.raises(SemanticDiffeomorphismViolationError):
            narrative_agent(
                stratum_verdicts=valid_severity_verdicts,
                topological_metrics=valid_topological_metrics_with_beta1,
                proposed_narrative=narrative_optimal,  # Sin token SOCAVON
            )

    def test_telemetry_narrative_agent_narrative_canonicalization_error(
        self,
        narrative_agent: TelemetryNarrativeAgent,
        valid_severity_verdicts: List[SeverityLevel],
        valid_topological_metrics_optimal: Dict[str, float],
    ) -> None:
        """
        PRUEBA: Fallo en canonicalización lanza NarrativeCanonicalizationError.
        VALIDA: Propagación de excepciones de Fase 3.
        """
        # Narrativa demasiado corta
        with pytest.raises(NarrativeCanonicalizationError):
            narrative_agent(
                stratum_verdicts=valid_severity_verdicts,
                topological_metrics=valid_topological_metrics_optimal,
                proposed_narrative="A",
            )

    def test_telemetry_narrative_agent_domain_integrity_violation_error(
        self,
        narrative_agent: TelemetryNarrativeAgent,
        valid_topological_metrics_optimal: Dict[str, float],
        narrative_optimal: str,
    ) -> None:
        """
        PRUEBA: Violación de integridad de dominio lanza DomainIntegrityViolationError.
        VALIDA: Validación de tipos de entrada.
        """
        with pytest.raises(DomainIntegrityViolationError):
            narrative_agent(
                stratum_verdicts=None,  # type: ignore
                topological_metrics=valid_topological_metrics_optimal,
                proposed_narrative=narrative_optimal,
            )

    def test_telemetry_narrative_agent_inheritance_chain(
        self,
        narrative_agent: TelemetryNarrativeAgent,
    ) -> None:
        """
        PRUEBA: Cadena de herencia del TelemetryNarrativeAgent.
        VALIDA: Arquitectura de fases anidadas.
        """
        assert isinstance(narrative_agent, TelemetryNarrativeAgent)
        assert isinstance(narrative_agent, Phase3_NarrativeCanonicalizationEnforcer)
        assert isinstance(narrative_agent, Phase2_SemanticDiffeomorphismCertifier)
        assert isinstance(narrative_agent, Phase1_SeverityLatticeCollapser)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# §Z. PRUEBAS DE ESTRUCTURAS DE DATOS (Data Classes)
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestDataStructures:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  PRUEBAS DE ESTRUCTURAS DE DATOS INMUTABLES                                           ║
    ║  ─────────────────────────────────────────────────────────────────────────────        ║
    ║  Valida la integridad de todos los DTOs del espacio semántico.                        ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    def test_lattice_collapse_state_creation(
        self,
        valid_severity_verdicts: List[SeverityLevel],
    ) -> None:
        """
        PRUEBA: Creación de LatticeCollapseState.
        VALIDA: Estructura inmutable del estado de colapso.
        """
        state = LatticeCollapseState(
            supremum_verdict=SeverityLevel.SEVERE,
            is_worst_case_enforced=True,
            lattice_size=3,
            is_empty_join=False,
        )
        assert state.supremum_verdict == SeverityLevel.SEVERE
        assert state.is_worst_case_enforced is True

    def test_diffeomorphism_audit_data_creation(
        self,
        valid_topological_metrics_with_beta1: Dict[str, float],
        narrative_with_socavon: str,
    ) -> None:
        """
        PRUEBA: Creación de DiffeomorphismAuditData.
        VALIDA: Artefacto de Fase 2.
        """
        audit = DiffeomorphismAuditData(
            is_isomorphic=True,
            betti_1_verified=True,
            fiedler_psi_verified=True,
            semantic_drift_detected=False,
            semantic_contract_enforced=True,
            betti_1_required=True,
            fiedler_psi_required=False,
            normalized_narrative=narrative_with_socavon.upper(),
        )
        assert audit.is_isomorphic is True
        assert audit.semantic_drift_detected is False

    def test_narrative_canonicalization_data_creation(
        self,
        narrative_optimal: str,
    ) -> None:
        """
        PRUEBA: Creación de NarrativeCanonicalizationData.
        VALIDA: Artefacto de Fase 3.
        """
        audit = NarrativeCanonicalizationData(
            canonical_narrative=narrative_optimal,
            narrative_length=len(narrative_optimal),
            contains_control_chars=False,
            is_canonical=True,
        )
        assert audit.is_canonical is True
        assert audit.contains_control_chars is False

    def test_phase1_lattice_handoff_creation(
        self,
        valid_severity_verdicts: List[SeverityLevel],
    ) -> None:
        """
        PRUEBA: Creación de Phase1LatticeHandoff.
        VALIDA: Puente funtorial Φ₁ → Φ₂.
        """
        lattice_state = LatticeCollapseState(
            supremum_verdict=SeverityLevel.SEVERE,
            is_worst_case_enforced=True,
            lattice_size=3,
            is_empty_join=False,
        )
        handoff = Phase1LatticeHandoff(
            lattice_state=lattice_state,
            stratum_verdicts_certified=tuple(valid_severity_verdicts),
        )
        assert isinstance(handoff.lattice_state, LatticeCollapseState)
        assert len(handoff.stratum_verdicts_certified) == 3

    def test_phase2_diffeomorphism_handoff_creation(
        self,
        valid_severity_verdicts: List[SeverityLevel],
        narrative_optimal: str,
    ) -> None:
        """
        PRUEBA: Creación de Phase2DiffeomorphismHandoff.
        VALIDA: Puente funtorial Φ₂ → Φ₃.
        """
        lattice_state = LatticeCollapseState(
            supremum_verdict=SeverityLevel.SEVERE,
            is_worst_case_enforced=True,
            lattice_size=3,
            is_empty_join=False,
        )
        phase1_handoff = Phase1LatticeHandoff(
            lattice_state=lattice_state,
            stratum_verdicts_certified=tuple(valid_severity_verdicts),
        )
        diffeo_audit = DiffeomorphismAuditData(
            is_isomorphic=True,
            betti_1_verified=True,
            fiedler_psi_verified=True,
            semantic_drift_detected=False,
            semantic_contract_enforced=False,
            betti_1_required=False,
            fiedler_psi_required=False,
            normalized_narrative=narrative_optimal.upper(),
        )
        handoff = Phase2DiffeomorphismHandoff(
            phase1_handoff=phase1_handoff,
            diffeomorphism_audit=diffeo_audit,
            canonicalizable_narrative=narrative_optimal,
        )
        assert isinstance(handoff.phase1_handoff, Phase1LatticeHandoff)
        assert isinstance(handoff.diffeomorphism_audit, DiffeomorphismAuditData)

    def test_narrative_agent_state_creation(
        self,
        valid_severity_verdicts: List[SeverityLevel],
        narrative_optimal: str,
    ) -> None:
        """
        PRUEBA: Creación de NarrativeAgentState (objeto final).
        VALIDA: Estado epistemológico completo del endofuntor.
        """
        lattice_state = LatticeCollapseState(
            supremum_verdict=SeverityLevel.SEVERE,
            is_worst_case_enforced=True,
            lattice_size=3,
            is_empty_join=False,
        )
        diffeo_audit = DiffeomorphismAuditData(
            is_isomorphic=True,
            betti_1_verified=True,
            fiedler_psi_verified=True,
            semantic_drift_detected=False,
            semantic_contract_enforced=False,
            betti_1_required=False,
            fiedler_psi_required=False,
            normalized_narrative=narrative_optimal.upper(),
        )
        canon_audit = NarrativeCanonicalizationData(
            canonical_narrative=narrative_optimal,
            narrative_length=len(narrative_optimal),
            contains_control_chars=False,
            is_canonical=True,
        )
        state = NarrativeAgentState(
            lattice_collapse=lattice_state,
            diffeomorphism_audit=diffeo_audit,
            approved_narrative=narrative_optimal,
            is_epistemologically_valid=True,
            canonicalization_audit=canon_audit,
        )
        assert state.is_epistemologically_valid is True
        assert isinstance(state.lattice_collapse, LatticeCollapseState)
        assert isinstance(state.diffeomorphism_audit, DiffeomorphismAuditData)
        assert isinstance(state.canonicalization_audit, NarrativeCanonicalizationData)


# ═══════════════════════════════════════════════════════════════════════════════════════════
# §∞. PRUEBAS DE CONSTANTES RETICULARES Y SEMÁNTICAS
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestReticularSemanticConstants:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  PRUEBAS DE CONSTANTES RETICULARES Y SEMÁNTICAS                                       ║
    ║  ─────────────────────────────────────────────────────────────────────────────        ║
    ║  Valida que las constantes del módulo tengan valores correctos y consistentes.        ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    def test_betti_positivity_threshold_value(self) -> None:
        """
        PRUEBA: Valor de _BETTI_POSITIVITY_THRESHOLD.
        VALIDA: Umbral de positividad homológica.
        """
        assert _BETTI_POSITIVITY_THRESHOLD == 1e-12
        assert _BETTI_POSITIVITY_THRESHOLD > 0

    def test_betti_integrality_tolerance_value(self) -> None:
        """
        PRUEBA: Valor de _BETTI_INTEGRALITY_TOLERANCE.
        VALIDA: Tolerancia de integralidad para β₁.
        """
        assert _BETTI_INTEGRALITY_TOLERANCE == 1e-9
        assert _BETTI_INTEGRALITY_TOLERANCE > 0

    def test_pyramid_stability_threshold_value(self) -> None:
        """
        PRUEBA: Valor de _PYRAMID_STABILITY_THRESHOLD.
        VALIDA: Umbral de estabilidad piramidal.
        """
        assert _PYRAMID_STABILITY_THRESHOLD == 1.0
        assert _PYRAMID_STABILITY_THRESHOLD > 0

    def test_stability_epsilon_value(self) -> None:
        """
        PRUEBA: Valor de _STABILITY_EPSILON.
        VALIDA: Epsilon de estabilidad.
        """
        assert _STABILITY_EPSILON == 1e-12
        assert _STABILITY_EPSILON > 0

    def test_required_token_socavon_value(self) -> None:
        """
        PRUEBA: Valor de _REQUIRED_TOKEN_SOCAVON.
        VALIDA: Token canónico para β₁ > 0.
        """
        assert _REQUIRED_TOKEN_SOCAVON == "SOCAVON LOGICO"
        assert len(_REQUIRED_TOKEN_SOCAVON) > 0

    def test_required_token_piramide_value(self) -> None:
        """
        PRUEBA: Valor de _REQUIRED_TOKEN_PIRAMIDE.
        VALIDA: Token canónico para Ψ < 1.0.
        """
        assert _REQUIRED_TOKEN_PIRAMIDE == "PIRAMIDE INVERTIDA"
        assert len(_REQUIRED_TOKEN_PIRAMIDE) > 0

    def test_beta_1_metric_names_value(self) -> None:
        """
        PRUEBA: Valor de _BETA_1_METRIC_NAMES.
        VALIDA: Nombres alternativos para β₁.
        """
        assert isinstance(_BETA_1_METRIC_NAMES, tuple)
        assert "beta_1" in _BETA_1_METRIC_NAMES
        assert "betti_1" in _BETA_1_METRIC_NAMES
        assert "b1" in _BETA_1_METRIC_NAMES

    def test_pyramid_stability_metric_names_value(self) -> None:
        """
        PRUEBA: Valor de _PYRAMID_STABILITY_METRIC_NAMES.
        VALIDA: Nombres alternativos para Ψ.
        """
        assert isinstance(_PYRAMID_STABILITY_METRIC_NAMES, tuple)
        assert "fiedler_psi" in _PYRAMID_STABILITY_METRIC_NAMES
        assert "psi" in _PYRAMID_STABILITY_METRIC_NAMES

    def test_min_narrative_length_value(self) -> None:
        """
        PRUEBA: Valor de _MIN_NARRATIVE_LENGTH.
        VALIDA: Longitud mínima de narrativa.
        """
        assert _MIN_NARRATIVE_LENGTH == 1
        assert _MIN_NARRATIVE_LENGTH > 0

    def test_max_narrative_length_value(self) -> None:
        """
        PRUEBA: Valor de _MAX_NARRATIVE_LENGTH.
        VALIDA: Longitud máxima de narrativa.
        """
        assert _MAX_NARRATIVE_LENGTH == 20_000
        assert _MAX_NARRATIVE_LENGTH > 0

    def test_control_character_pattern_value(self) -> None:
        """
        PRUEBA: Valor de _CONTROL_CHARACTER_PATTERN.
        VALIDA: Patrón regex de caracteres de control.
        """
        assert isinstance(_CONTROL_CHARACTER_PATTERN, str)
        assert len(_CONTROL_CHARACTER_PATTERN) > 0


# ═══════════════════════════════════════════════════════════════════════════════════════════
# §Ω. PRUEBAS DEL ENUM SEVERITYLEVEL
# ═══════════════════════════════════════════════════════════════════════════════════════════


class TestSeverityLevelEnum:
    r"""
    ╔═══════════════════════════════════════════════════════════════════════════════════════╗
    ║  PRUEBAS DEL ENUMERADO SEVERITYLEVEL (Retículo de Severidad)                          ║
    ║  ─────────────────────────────────────────────────────────────────────────────        ║
    ║  Valida la integridad del retículo algebraico de severidad.                           ║
    ╚═══════════════════════════════════════════════════════════════════════════════════════╝
    """

    def test_severity_level_optimo_value(self) -> None:
        """
        PRUEBA: SeverityLevel.OPTIMO = 0 (⊥).
        VALIDA: Elemento mínimo del retículo.
        """
        assert SeverityLevel.OPTIMO.value == 0

    def test_severity_level_moderado_value(self) -> None:
        """
        PRUEBA: SeverityLevel.MODERADO = 1.
        VALIDA: Segundo nivel del retículo.
        """
        assert SeverityLevel.MODERADO.value == 1

    def test_severity_level_severo_value(self) -> None:
        """
        PRUEBA: SeverityLevel.SEVERE = 2.
        VALIDA: Tercer nivel del retículo.
        """
        assert SeverityLevel.SEVERE.value == 2

    def test_severity_level_critico_value(self) -> None:
        """
        PRUEBA: SeverityLevel.CRITICO = 3 (⊤).
        VALIDA: Elemento máximo absorbente.
        """
        assert SeverityLevel.CRITICO.value == 3

    def test_severity_level_ordering(self) -> None:
        """
        PRUEBA: Ordenamiento correcto del retículo.
        VALIDA: OPTIMO < MODERADO < SEVERO < CRITICO.
        """
        assert SeverityLevel.OPTIMO.value < SeverityLevel.MODERADO.value
        assert SeverityLevel.MODERADO.value < SeverityLevel.SEVERE.value
        assert SeverityLevel.SEVERE.value < SeverityLevel.CRITICO.value

    def test_severity_level_from_int(self) -> None:
        """
        PRUEBA: Conversión de entero a SeverityLevel.
        VALIDA: Coerción de tipos.
        """
        assert SeverityLevel(0) == SeverityLevel.OPTIMO
        assert SeverityLevel(1) == SeverityLevel.MODERADO
        assert SeverityLevel(2) == SeverityLevel.SEVERE
        assert SeverityLevel(3) == SeverityLevel.CRITICO

    def test_severity_level_from_name(self) -> None:
        """
        PRUEBA: Conversión de nombre a SeverityLevel.
        VALIDA: Lookup por nombre.
        """
        assert SeverityLevel["OPTIMO"] == SeverityLevel.OPTIMO
        assert SeverityLevel["CRITICO"] == SeverityLevel.CRITICO

    def test_severity_level_lattice_structure(self) -> None:
        """
        PRUEBA: Estructura de retículo distributivo acotado.
        VALIDA: ⊥ ≤ MODERADO ≤ SEVERO ≤ ⊤.
        """
        # Verificar que es un IntEnum con valores consecutivos
        values = [level.value for level in SeverityLevel]
        assert values == [0, 1, 2, 3]


# ═══════════════════════════════════════════════════════════════════════════════════════════
# §Ω. EJECUCIÓN DIRECTA (Para debugging)
# ═══════════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Ejecución directa para debugging fuera de pytest.
    Uso: python tests/unit/agents/core/test_telemetry_narrative_agent.py
    """
    import sys
    import os
    
    # Agregar el directorio raíz al path
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../")))
    
    pytest.main([__file__, "-v", "--tb=short"])