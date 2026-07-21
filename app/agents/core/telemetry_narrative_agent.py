# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Telemetry Narrative Agent (El Intérprete Diplomático Supremo)       ║
║ Ruta   : app/core/telemetry_narrative_agent.py                               ║
║ Versión: 2.0.0-Diffeomorphism-Lattice-Canonical-Strict                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y TEORÍA DE CATEGORÍAS (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor gobierna a `telemetry_narrative.py`. Actúa como el Funtor de
Difeomorfismo Semántico:

    F : C_Math -> D_Narrative

proyectando el espacio de invariantes matemáticos sobre la ontología del negocio.
Erradica el libre albedrío estocástico del LLM, confinándolo a redactar un
"Juicio del Consejo" estrictamente gobernado por restricciones geométricas.

ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Colapso en el Retículo Acotado Distributivo:
    Veredicto = ⨆_{i ∈ {F, T, S, W}} v_i.

Fase 2 → Certificación de Difeomorfismo Semántico:
    β₁ > 0  =>  debe contener "Socavón Lógico".
    Ψ < 1   =>  debe contener "Pirámide Invertida".

Fase 3 → Canonicalización Diplomática del Texto:
    Sanea la narrativa, re-verifica los invariantes léxicos y compone el acta
    inmutable final.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum, unique
from typing import Any, Final, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter (Stubs de aislamiento)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:

    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos MIC."""
        pass

    class Morphism:
        """Clase base de Morfismos del Topos."""
        pass


logger = logging.getLogger("MIC.Core.TelemetryNarrativeAgent")

if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# ═══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES RETICULARES, SEMÁNTICAS Y DE CANONICALIZACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

# Umbral de positividad homológica para β₁.
_BETTI_POSITIVITY_THRESHOLD: Final[float] = 1e-12

# Tolerancia de integralidad para números de Betti.
_BETTI_INTEGRALITY_TOLERANCE: Final[float] = 1e-9

# Umbral de estabilidad piramidal / Fiedler Ψ.
_PYRAMID_STABILITY_THRESHOLD: Final[float] = 1.0
_STABILITY_EPSILON: Final[float] = 1e-12

# Tokens canónicos del difeomorfismo semántico (sin acentos para matching).
_REQUIRED_TOKEN_SOCAVON: Final[str] = "SOCAVON LOGICO"
_REQUIRED_TOKEN_PIRAMIDE: Final[str] = "PIRAMIDE INVERTIDA"

# Nombres alternativos para extraer métricas topológicas.
_BETA_1_METRIC_NAMES: Final[Tuple[str, ...]] = (
    "beta_1",
    "betti_1",
    "b1",
)

_PYRAMID_STABILITY_METRIC_NAMES: Final[Tuple[str, ...]] = (
    "pyramid_stability",
    "fiedler_psi",
    "psi",
    "algebraic_connectivity",
)

# Límites de canonicalización narrativa.
_MIN_NARRATIVE_LENGTH: Final[int] = 1
_MAX_NARRATIVE_LENGTH: Final[int] = 20_000

# Patrón de caracteres de control prohibidos tras canonicalización.
_CONTROL_CHARACTER_PATTERN: Final[str] = (
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
)


# ═══════════════════════════════════════════════════════════════════════════════
# §B. RETÍCULO DISTRIBUTIVO ACOTADO (Álgebra de Severidad)
# ═══════════════════════════════════════════════════════════════════════════════

@unique
class SeverityLevel(IntEnum):
    r"""
    Retículo algebraico de severidad bajo el orden parcial estricto:

        ⊥ ≤ MODERADO ≤ SEVERO ≤ ⊤

    donde:
        ⊥ = OPTIMO
        ⊤ = CRITICO
    """
    OPTIMO = 0       # Elemento mínimo (⊥)
    MODERADO = 1
    SEVERO = 2
    CRITICO = 3      # Elemento máximo absorbente (⊤)


# ═══════════════════════════════════════════════════════════════════════════════
# §C. JERARQUÍA DE EXCEPCIONES ONTOLÓGICAS Y SEMÁNTICAS
# ═══════════════════════════════════════════════════════════════════════════════

class TelemetryNarrativeAgentError(TopologicalInvariantError):
    """Excepción raíz del Intérprete Diplomático Supremo."""
    pass


class DomainIntegrityViolationError(TelemetryNarrativeAgentError):
    """Detonada cuando un input viola su contrato de dominio."""
    pass


class SeverityLatticeCollapseError(TelemetryNarrativeAgentError):
    r"""
    Detonada si el sistema viola la operación Supremo (⨆ v_i) del retículo
    distributivo acotado.
    """
    pass


class SemanticDiffeomorphismViolationError(TelemetryNarrativeAgentError):
    r"""
    Detonada si el texto generado no exhibe las traducciones biyectivas exactas
    de la patología geométrica subyacente.
    """
    pass


class NarrativeCanonicalizationError(TelemetryNarrativeAgentError):
    """Detonada si la narrativa no puede canonicalizarse de forma inmutable."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# §D. ESTRUCTURAS INMUTABLES (DTOs del Espacio de Fase Semántico)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class LatticeCollapseState:
    r"""Artefacto de Fase 1. Colapso inmutable de severidad."""
    supremum_verdict: SeverityLevel
    is_worst_case_enforced: bool
    lattice_size: int = 0
    is_empty_join: bool = False


@dataclass(frozen=True, slots=True)
class DiffeomorphismAuditData:
    r"""Artefacto de Fase 2. Certificado de isomorfismo semántico."""
    is_isomorphic: bool
    betti_1_verified: bool
    fiedler_psi_verified: bool
    semantic_drift_detected: bool
    semantic_contract_enforced: bool = True
    betti_1_required: bool = False
    fiedler_psi_required: bool = False
    normalized_narrative: str = ""


@dataclass(frozen=True, slots=True)
class NarrativeCanonicalizationData:
    r"""Artefacto de Fase 3. Certificado de canonicalización narrativa."""
    canonical_narrative: str
    narrative_length: int
    contains_control_chars: bool
    is_canonical: bool


@dataclass(frozen=True, slots=True)
class Phase1LatticeHandoff:
    r"""
    Handoff formal de Fase 1 → Fase 2.

    Este objeto es la continuación material del colapso reticular y el prefijo
    obligatorio del certificador de difeomorfismo semántico.
    """
    lattice_state: LatticeCollapseState
    stratum_verdicts_certified: Tuple[SeverityLevel, ...]


@dataclass(frozen=True, slots=True)
class Phase2DiffeomorphismHandoff:
    r"""
    Handoff formal de Fase 2 → Fase 3.

    Este objeto transporta el certificado de isomorfismo semántico y la
    narrativa propuesta para su canonicalización diplomática.
    """
    phase1_handoff: Phase1LatticeHandoff
    diffeomorphism_audit: DiffeomorphismAuditData
    canonicalizable_narrative: str


@dataclass(frozen=True, slots=True)
class NarrativeAgentState:
    r"""Objeto final del endofuntor Z_Narrative."""
    lattice_collapse: LatticeCollapseState
    diffeomorphism_audit: DiffeomorphismAuditData
    approved_narrative: str
    is_epistemologically_valid: bool
    canonicalization_audit: Optional[NarrativeCanonicalizationData] = None


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1: COLAPSO EN EL RETÍCULO ACOTADO DISTRIBUTIVO                       ║
# ║   Evalúa Veredicto = ⨆ v_i.                                                ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase1_SeverityLatticeCollapser:
    r"""
    Sintetiza las decisiones inter-estrato operando bajo la clausura algebraica
    del Supremo. Impone el Worst-Case Scenario de forma determinista.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 1.1. Normalización de tokens enum
    # ─────────────────────────────────────────────────────────────────────────
    def _normalize_enum_token(
        self,
        value: str,
    ) -> str:
        r"""
        Normaliza un token textual hacia una clave enum canónica:
        mayúsculas, sin acentos, sin espacios redundantes.
        """
        token = unicodedata.normalize("NFKD", value)
        token = "".join(
            ch for ch in token
            if not unicodedata.combining(ch)
        )
        token = token.upper()
        token = re.sub(r"[^A-Z0-9_]+", "_", token)
        token = re.sub(r"_+", "_", token).strip("_")

        if not token:
            raise DomainIntegrityViolationError(
                "Token de severidad vacío tras normalización."
            )

        return token

    # ─────────────────────────────────────────────────────────────────────────
    # 1.2. Coerción estricta de severidad
    # ─────────────────────────────────────────────────────────────────────────
    def _coerce_severity(
        self,
        name: str,
        value: Any,
    ) -> SeverityLevel:
        r"""
        Coerciona un valor a SeverityLevel con tipado estricto.

        Se admite:
            - SeverityLevel
            - int / np.integer
            - float entero
            - str con nombre o número
        """
        if isinstance(value, SeverityLevel):
            return value

        if isinstance(value, (bool, np.bool_)):
            raise DomainIntegrityViolationError(
                f"{name} no puede ser booleano; se requiere SeverityLevel."
            )

        if isinstance(value, (int, np.integer)):
            try:
                return SeverityLevel(int(value))
            except ValueError as exc:
                raise DomainIntegrityViolationError(
                    f"{name}={value!r} no pertenece al retículo de severidad."
                ) from exc

        if isinstance(value, (float, np.floating)):
            scalar = float(value)
            if not np.isfinite(scalar):
                raise DomainIntegrityViolationError(
                    f"{name} no es finito."
                )
            if scalar.is_integer():
                try:
                    return SeverityLevel(int(scalar))
                except ValueError as exc:
                    raise DomainIntegrityViolationError(
                        f"{name}={scalar!r} no pertenece al retículo."
                    ) from exc
            raise DomainIntegrityViolationError(
                f"{name}={scalar!r} no es un nivel entero de severidad."
            )

        if isinstance(value, str):
            token = self._normalize_enum_token(value)
            try:
                return SeverityLevel[token]
            except KeyError:
                try:
                    return SeverityLevel(int(token))
                except (KeyError, ValueError, TypeError) as exc:
                    raise DomainIntegrityViolationError(
                        f"{name}='{value}' no puede coercionarse a "
                        f"SeverityLevel."
                    ) from exc

        raise DomainIntegrityViolationError(
            f"{name} debe ser SeverityLevel, int, float entero o str."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1.3. Coerción del vector de veredictos
    # ─────────────────────────────────────────────────────────────────────────
    def _coerce_stratum_verdicts(
        self,
        stratum_verdicts: Any,
    ) -> Tuple[SeverityLevel, ...]:
        r"""
        Valida y coerciona la colección de veredictos estratales.
        """
        if stratum_verdicts is None:
            raise DomainIntegrityViolationError(
                "stratum_verdicts no puede ser None."
            )

        if isinstance(stratum_verdicts, (str, bytes, bytearray, Mapping)):
            raise DomainIntegrityViolationError(
                "stratum_verdicts no debe ser una cadena, bytes o mapping. "
                "Se requiere una colección iterable de severidades."
            )

        try:
            raw_verdicts = list(stratum_verdicts)
        except TypeError as exc:
            raise DomainIntegrityViolationError(
                "stratum_verdicts no es iterable."
            ) from exc

        certified: List[SeverityLevel] = []

        for idx, verdict in enumerate(raw_verdicts):
            certified.append(
                self._coerce_severity(
                    f"stratum_verdicts[{idx}]",
                    verdict,
                )
            )

        return tuple(certified)

    # ─────────────────────────────────────────────────────────────────────────
    # 1.4. Operación join binaria del retículo
    # ─────────────────────────────────────────────────────────────────────────
    def _lattice_join(
        self,
        left: SeverityLevel,
        right: SeverityLevel,
    ) -> SeverityLevel:
        r"""
        Operación join binaria:

            a ⊔ b = max(a, b)

        en la cadena finita de severidad.
        """
        return left if left.value >= right.value else right

    # ─────────────────────────────────────────────────────────────────────────
    # 1.5. Supremo del retículo
    # ─────────────────────────────────────────────────────────────────────────
    def _lattice_supremum(
        self,
        verdicts: Tuple[SeverityLevel, ...],
    ) -> SeverityLevel:
        r"""
        Calcula el supremo:

            ⨆_{i} v_i.

        El join vacío retorna el elemento mínimo ⊥ = OPTIMO.
        """
        supremum = SeverityLevel.OPTIMO

        for verdict in verdicts:
            supremum = self._lattice_join(supremum, verdict)

        return supremum

    # ─────────────────────────────────────────────────────────────────────────
    # 1.6. Auditoría de absorción y consistencia reticular
    # ─────────────────────────────────────────────────────────────────────────
    def _assert_lattice_absorption(
        self,
        verdicts: Tuple[SeverityLevel, ...],
        supremum: SeverityLevel,
    ) -> None:
        r"""
        Verifica la ley de absorción:

            ⊥ ⊔ ⊤ = ⊤.

        Si CRITICO está presente, el supremo debe ser CRITICO.
        """
        if not verdicts:
            return

        if SeverityLevel.CRITICO in verdicts and supremum != SeverityLevel.CRITICO:
            raise SeverityLatticeCollapseError(
                "Fallo en la Clausura Algebraica: se esperaba "
                "⊥ ⊔ ⊤ = ⊤, pero el sistema no colapsó al estado CRITICO."
            )

        if supremum not in verdicts:
            raise SeverityLatticeCollapseError(
                "El supremo calculado no pertenece al conjunto de veredictos. "
                "Violación de la propiedad de cadena finita del retículo."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 1.7. Implementación interna del colapso reticular
    # ─────────────────────────────────────────────────────────────────────────
    def _collapse_severity_lattice_internal(
        self,
        stratum_verdicts: Any,
    ) -> Tuple[LatticeCollapseState, Tuple[SeverityLevel, ...]]:
        r"""
        Implementación interna que retorna el estado de colapso y los
        veredictos certificados.
        """
        certified_verdicts = self._coerce_stratum_verdicts(stratum_verdicts)
        supremum = self._lattice_supremum(certified_verdicts)

        self._assert_lattice_absorption(certified_verdicts, supremum)

        state = LatticeCollapseState(
            supremum_verdict=supremum,
            is_worst_case_enforced=True,
            lattice_size=len(certified_verdicts),
            is_empty_join=(len(certified_verdicts) == 0),
        )

        return state, certified_verdicts

    # ─────────────────────────────────────────────────────────────────────────
    # 1.8. Wrapper público / retrocompatible de colapso
    # ─────────────────────────────────────────────────────────────────────────
    def _collapse_severity_lattice(
        self,
        stratum_verdicts: List[SeverityLevel],
    ) -> LatticeCollapseState:
        r"""
        Computa la operación ⨆ (Join) en el retículo distributivo.
        Conserva la signatura original de Fase 1.
        """
        state, _ = self._collapse_severity_lattice_internal(stratum_verdicts)
        return state

    # ─────────────────────────────────────────────────────────────────────────
    # 1.9. ÚLTIMO MÉTODO DE FASE 1: HANDOFF FORMAL HACIA FASE 2
    # ─────────────────────────────────────────────────────────────────────────
    def _phase1_collapse_and_handoff_to_phase2(
        self,
        stratum_verdicts: List[SeverityLevel],
    ) -> Phase1LatticeHandoff:
        r"""
        Último método de la Fase 1.

        Su definición formal es la continuación directa de la Fase 2:
        entrega el colapso reticular y los veredictos certificados como
        prefijo obligatorio del difeomorfismo semántico.
        """
        state, certified_verdicts = self._collapse_severity_lattice_internal(
            stratum_verdicts
        )

        logger.debug(
            "Fase 1 completada. Supremo=%s | tamaño_retículo=%d | "
            "join_vacío=%s.",
            state.supremum_verdict.name,
            state.lattice_size,
            str(state.is_empty_join),
        )

        return Phase1LatticeHandoff(
            lattice_state=state,
            stratum_verdicts_certified=certified_verdicts,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2: CERTIFICACIÓN DE DIFEOMORFISMO SEMÁNTICO                          ║
# ║   Verifica que la narrativa preserve los invariantes β₁ y Ψ.                ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase2_SemanticDiffeomorphismCertifier(Phase1_SeverityLatticeCollapser):
    r"""
    Asegura que el texto redactado (Empatía Táctica) posea un isomorfismo
    directo con las métricas abstractas. Subordina el LLM al diccionario
    predefinido.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 2.1. Normalización semántica de texto
    # ─────────────────────────────────────────────────────────────────────────
    def _normalize_semantic_text(
        self,
        text: Any,
    ) -> str:
        r"""
        Normaliza texto para matching semántico:
        NFKD, sin acentos, mayúsculas, puntuación como espacio y colapso de
        blancos.
        """
        if not isinstance(text, str):
            raise DomainIntegrityViolationError(
                "La narrativa propuesta debe ser una cadena de texto."
            )

        normalized = unicodedata.normalize("NFKD", text)
        normalized = "".join(
            ch for ch in normalized
            if not unicodedata.combining(ch)
        )
        normalized = normalized.upper()

        # Convierte puntuación y separadores en espacios para matching robusto.
        normalized = re.sub(r"[^0-9A-Z]+", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()

        return normalized

    # ─────────────────────────────────────────────────────────────────────────
    # 2.2. Extracción robusta de métricas topológicas
    # ─────────────────────────────────────────────────────────────────────────
    def _extract_metric_value(
        self,
        topological_metrics: Any,
        metric_names: Tuple[str, ...],
    ) -> Optional[float]:
        r"""
        Extrae una métrica desde un objeto o mapping, admitiendo nombres
        alternativos. Retorna None si la métrica no está presente.
        """
        if topological_metrics is None:
            return None

        raw_value: Any = None
        found = False

        if isinstance(topological_metrics, Mapping):
            for name in metric_names:
                if name in topological_metrics:
                    raw_value = topological_metrics[name]
                    found = True
                    break
        else:
            for name in metric_names:
                if hasattr(topological_metrics, name):
                    raw_value = getattr(topological_metrics, name)
                    found = True
                    break

        if not found:
            return None

        if raw_value is None:
            return None

        if isinstance(raw_value, (bool, np.bool_)):
            raise DomainIntegrityViolationError(
                f"Métrica topológica {metric_names!r} no puede ser booleana."
            )

        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise DomainIntegrityViolationError(
                f"Métrica topológica {metric_names!r} no es numérica."
            ) from exc

        if not bool(np.isfinite(value)):
            raise DomainIntegrityViolationError(
                f"Métrica topológica {metric_names!r} no es finita."
            )

        return value

    # ─────────────────────────────────────────────────────────────────────────
    # 2.3. Certificación de β₁
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_betti_1(
        self,
        topological_metrics: Any,
    ) -> Tuple[Optional[float], bool]:
        r"""
        Certifica β₁ y determina si se requiere el token 'Socavón Lógico'.

        β₁ debe ser no negativo y, si es positivo, aproximadamente entero.
        """
        beta_1 = self._extract_metric_value(
            topological_metrics,
            _BETA_1_METRIC_NAMES,
        )

        if beta_1 is None:
            return None, False

        if beta_1 < -_BETTI_POSITIVITY_THRESHOLD:
            raise DomainIntegrityViolationError(
                f"β₁={beta_1:.6e} es negativo. Los números de Betti son "
                f"invariantes no negativos."
            )

        if beta_1 < 0.0:
            beta_1 = 0.0

        if beta_1 > _BETTI_POSITIVITY_THRESHOLD:
            nearest_integer = round(beta_1)
            integrality_error = abs(beta_1 - float(nearest_integer))

            if integrality_error > _BETTI_INTEGRALITY_TOLERANCE:
                raise DomainIntegrityViolationError(
                    f"β₁={beta_1:.6e} no es aproximadamente entero. "
                    f"Error de integralidad={integrality_error:.6e}."
                )

            return beta_1, True

        return beta_1, False

    # ─────────────────────────────────────────────────────────────────────────
    # 2.4. Certificación de estabilidad piramidal / Fiedler Ψ
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_pyramid_stability(
        self,
        topological_metrics: Any,
    ) -> Tuple[Optional[float], bool]:
        r"""
        Certifica Ψ y determina si se requiere el token 'Pirámide Invertida'.

        Si Ψ < 1 - ε, existe patología de estabilidad.
        """
        psi = self._extract_metric_value(
            topological_metrics,
            _PYRAMID_STABILITY_METRIC_NAMES,
        )

        if psi is None:
            return None, False

        if psi < -_STABILITY_EPSILON:
            raise DomainIntegrityViolationError(
                f"Ψ={psi:.6e} es negativo. La estabilidad estructural debe "
                f"ser no negativa."
            )

        if psi < 0.0:
            psi = 0.0

        required = psi < (_PYRAMID_STABILITY_THRESHOLD - _STABILITY_EPSILON)

        return psi, required

    # ─────────────────────────────────────────────────────────────────────────
    # 2.5. Implementación interna del difeomorfismo semántico
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_semantic_diffeomorphism_internal(
        self,
        topological_metrics: Any,
        proposed_narrative: str,
    ) -> DiffeomorphismAuditData:
        r"""
        Ejecuta la auditoría de isomorfismo semántico sobre la narrativa.
        """
        normalized_narrative = self._normalize_semantic_text(proposed_narrative)

        beta_1, betti_1_required = self._certify_betti_1(topological_metrics)
        psi, fiedler_psi_required = self._certify_pyramid_stability(
            topological_metrics
        )

        betti_1_verified = (
            (not betti_1_required)
            or (_REQUIRED_TOKEN_SOCAVON in normalized_narrative)
        )

        fiedler_psi_verified = (
            (not fiedler_psi_required)
            or (_REQUIRED_TOKEN_PIRAMIDE in normalized_narrative)
        )

        semantic_drift_detected = (
            (betti_1_required and not betti_1_verified)
            or (fiedler_psi_required and not fiedler_psi_verified)
        )

        if semantic_drift_detected:
            raise SemanticDiffeomorphismViolationError(
                "La narrativa ejecutiva rompió el difeomorfismo con la matriz "
                "topológica. "
                f"[β₁ requerido={betti_1_required}, β₁ match={betti_1_verified}, "
                f"Ψ requerido={fiedler_psi_required}, Ψ match={fiedler_psi_verified}]. "
                "El LLM incurrió en deriva estocástica y diluyó la alerta técnica."
            )

        return DiffeomorphismAuditData(
            is_isomorphic=True,
            betti_1_verified=betti_1_verified,
            fiedler_psi_verified=fiedler_psi_verified,
            semantic_drift_detected=False,
            semantic_contract_enforced=True,
            betti_1_required=betti_1_required,
            fiedler_psi_required=fiedler_psi_required,
            normalized_narrative=normalized_narrative,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 2.6. Wrapper público / retrocompatible de difeomorfismo
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_semantic_diffeomorphism(
        self,
        topological_metrics: Any,
        proposed_narrative: str,
    ) -> DiffeomorphismAuditData:
        r"""
        Realiza la auditoría de isomorfismo. Conserva la signatura original
        de Fase 2.
        """
        return self._certify_semantic_diffeomorphism_internal(
            topological_metrics,
            proposed_narrative,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 2.7. ÚLTIMO MÉTODO DE FASE 2: HANDOFF FORMAL HACIA FASE 3
    # ─────────────────────────────────────────────────────────────────────────
    def _phase2_certify_and_handoff_to_phase3(
        self,
        phase1_handoff: Phase1LatticeHandoff,
        topological_metrics: Any,
        proposed_narrative: str,
    ) -> Phase2DiffeomorphismHandoff:
        r"""
        Último método de la Fase 2.

        Su definición formal es la continuación directa de la Fase 3:
        entrega el certificado de difeomorfismo y la narrativa propuesta como
        prefijo obligatorio de canonicalización diplomática.
        """
        if not isinstance(phase1_handoff, Phase1LatticeHandoff):
            raise DomainIntegrityViolationError(
                "Fase 2 exige un Phase1LatticeHandoff como prefijo formal."
            )

        normalized_preview = self._normalize_semantic_text(proposed_narrative)

        if not normalized_preview:
            raise DomainIntegrityViolationError(
                "La narrativa propuesta está vacía tras normalización."
            )

        if phase1_handoff.lattice_state.supremum_verdict == SeverityLevel.OPTIMO:
            # En estado óptimo no se exige contrato patológico, pero sí
            # canonicalización posterior.
            diffeomorphism_audit = DiffeomorphismAuditData(
                is_isomorphic=True,
                betti_1_verified=True,
                fiedler_psi_verified=True,
                semantic_drift_detected=False,
                semantic_contract_enforced=False,
                betti_1_required=False,
                fiedler_psi_required=False,
                normalized_narrative=normalized_preview,
            )
        else:
            diffeomorphism_audit = (
                self._certify_semantic_diffeomorphism_internal(
                    topological_metrics,
                    proposed_narrative,
                )
            )

        logger.debug(
            "Fase 2 completada. Isomorfismo=%s | contrato_forzado=%s.",
            str(diffeomorphism_audit.is_isomorphic),
            str(diffeomorphism_audit.semantic_contract_enforced),
        )

        return Phase2DiffeomorphismHandoff(
            phase1_handoff=phase1_handoff,
            diffeomorphism_audit=diffeomorphism_audit,
            canonicalizable_narrative=proposed_narrative,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3: CANONICALIZACIÓN DIPLOMÁTICA Y ACTA INMUTABLE                     ║
# ║   Sanea la narrativa y re-verifica los invariantes léxicos.                 ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase3_NarrativeCanonicalizationEnforcer(
    Phase2_SemanticDiffeomorphismCertifier
):
    r"""
    Canonicaliza la narrativa aprobada, eliminando ruido de control y
    re-verificando que el texto saneado preserve los tokens del difeomorfismo.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 3.1. Canonicalización de texto
    # ─────────────────────────────────────────────────────────────────────────
    def _canonicalize_narrative_text(
        self,
        proposed_narrative: Any,
    ) -> str:
        r"""
        Canonicaliza la narrativa:
            - NFKC
            - elimina caracteres de control
            - colapsa espacios
            - preserva contenido semántico visible
        """
        if not isinstance(proposed_narrative, str):
            raise DomainIntegrityViolationError(
                "La narrativa propuesta debe ser una cadena de texto."
            )

        text = unicodedata.normalize("NFKC", proposed_narrative)

        cleaned_chars: List[str] = []

        for ch in text:
            if ch in (" ", "\n", "\t", "\r"):
                cleaned_chars.append(" ")
                continue

            category = unicodedata.category(ch)

            if category.startswith("C"):
                continue

            cleaned_chars.append(ch)

        text = "".join(cleaned_chars)
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            raise DomainIntegrityViolationError(
                "La narrativa propuesta queda vacía tras canonicalización."
            )

        if len(text) < _MIN_NARRATIVE_LENGTH:
            raise NarrativeCanonicalizationError(
                "La narrativa canonicalizada es demasiado corta."
            )

        if len(text) > _MAX_NARRATIVE_LENGTH:
            raise NarrativeCanonicalizationError(
                f"La narrativa canonicalizada excede la longitud máxima "
                f"{_MAX_NARRATIVE_LENGTH}."
            )

        return text

    # ─────────────────────────────────────────────────────────────────────────
    # 3.2. Re-verificación de tokens tras canonicalización
    # ─────────────────────────────────────────────────────────────────────────
    def _assert_required_tokens_after_canonicalization(
        self,
        canonical_narrative: str,
        diffeomorphism_audit: DiffeomorphismAuditData,
    ) -> None:
        r"""
        Re-verifica que la canonicalización no haya destruido los tokens
        obligatorios del difeomorfismo semántico.
        """
        if not diffeomorphism_audit.semantic_contract_enforced:
            return

        normalized = self._normalize_semantic_text(canonical_narrative)

        if (
            diffeomorphism_audit.betti_1_required
            and _REQUIRED_TOKEN_SOCAVON not in normalized
        ):
            raise SemanticDiffeomorphismViolationError(
                "La canonicalización eliminó el token obligatorio "
                "'Socavón Lógico'."
            )

        if (
            diffeomorphism_audit.fiedler_psi_required
            and _REQUIRED_TOKEN_PIRAMIDE not in normalized
        ):
            raise SemanticDiffeomorphismViolationError(
                "La canonicalización eliminó el token obligatorio "
                "'Pirámide Invertida'."
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 3.3. Auditoría de canonicalización narrativa
    # ─────────────────────────────────────────────────────────────────────────
    def _audit_narrative_canonicity(
        self,
        proposed_narrative: str,
        diffeomorphism_audit: DiffeomorphismAuditData,
    ) -> NarrativeCanonicalizationData:
        r"""
        Emite el certificado de canonicalización narrativa.
        """
        canonical_narrative = self._canonicalize_narrative_text(
            proposed_narrative
        )

        self._assert_required_tokens_after_canonicalization(
            canonical_narrative,
            diffeomorphism_audit,
        )

        contains_control_chars = bool(
            re.search(_CONTROL_CHARACTER_PATTERN, canonical_narrative)
        )

        if contains_control_chars:
            raise NarrativeCanonicalizationError(
                "La narrativa canonicalizada aún contiene caracteres de "
                "control prohibidos."
            )

        return NarrativeCanonicalizationData(
            canonical_narrative=canonical_narrative,
            narrative_length=len(canonical_narrative),
            contains_control_chars=False,
            is_canonical=True,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3.4. ÚLTIMO MÉTODO DE FASE 3: FINALIZACIÓN FUNTORIAL
    # ─────────────────────────────────────────────────────────────────────────
    def _phase3_finalize_from_phase2_handoff(
        self,
        phase2_handoff: Phase2DiffeomorphismHandoff,
    ) -> NarrativeAgentState:
        r"""
        Último método de la Fase 3.

        Compone los certificados de Fase 1, Fase 2 y Fase 3 en el objeto
        terminal NarrativeAgentState.
        """
        if not isinstance(phase2_handoff, Phase2DiffeomorphismHandoff):
            raise DomainIntegrityViolationError(
                "Fase 3 exige un Phase2DiffeomorphismHandoff como prefijo "
                "formal."
            )

        canonicalization_audit = self._audit_narrative_canonicity(
            phase2_handoff.canonicalizable_narrative,
            phase2_handoff.diffeomorphism_audit,
        )

        state = NarrativeAgentState(
            lattice_collapse=phase2_handoff.phase1_handoff.lattice_state,
            diffeomorphism_audit=phase2_handoff.diffeomorphism_audit,
            approved_narrative=canonicalization_audit.canonical_narrative,
            is_epistemologically_valid=True,
            canonicalization_audit=canonicalization_audit,
        )

        logger.info(
            "Veredicto Narrativo Categórico ejecutado con éxito. "
            "Supremo=%s | Isomorfismo=%s | Longitud=%d.",
            state.lattice_collapse.supremum_verdict.name,
            str(state.diffeomorphism_audit.is_isomorphic),
            canonicalization_audit.narrative_length,
        )

        return state


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: TELEMETRY NARRATIVE AGENT                            ║
# ║   Endofuntor Z_Narrative = Φ₃ ∘ Φ₂ ∘ Φ₁                                    ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class TelemetryNarrativeAgent(
    Morphism,
    Phase3_NarrativeCanonicalizationEnforcer,
):
    r"""
    El Intérprete Diplomático Supremo.

    Gobierna el Pasaporte de Telemetría colapsando los dictámenes de las fases
    inferiores hacia un Acta de Deliberación inmutable, certificando la Empatía
    Táctica contra los axiomas de la topología algebraica.
    """

    def execute_diplomatic_narrative_governance(
        self,
        stratum_verdicts: List[SeverityLevel],
        topological_metrics: Any,
        proposed_narrative: str,
    ) -> NarrativeAgentState:
        r"""
        Ejecuta la composición funtorial estricta:

            Φ₁: SeverityLatticeCollapser
            Φ₂: SemanticDiffeomorphismCertifier
            Φ₃: NarrativeCanonicalizationEnforcer
        """
        phase1_handoff = self._phase1_collapse_and_handoff_to_phase2(
            stratum_verdicts
        )

        phase2_handoff = self._phase2_certify_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            topological_metrics=topological_metrics,
            proposed_narrative=proposed_narrative,
        )

        return self._phase3_finalize_from_phase2_handoff(
            phase2_handoff=phase2_handoff,
        )

    def __call__(
        self,
        stratum_verdicts: List[SeverityLevel],
        topological_metrics: Any,
        proposed_narrative: str,
    ) -> NarrativeAgentState:
        r"""Alias invocable del endofuntor de gobierno narrativo."""
        return self.execute_diplomatic_narrative_governance(
            stratum_verdicts=stratum_verdicts,
            topological_metrics=topological_metrics,
            proposed_narrative=proposed_narrative,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "TelemetryNarrativeAgentError",
    "DomainIntegrityViolationError",
    "SeverityLatticeCollapseError",
    "SemanticDiffeomorphismViolationError",
    "NarrativeCanonicalizationError",
    "SeverityLevel",
    "LatticeCollapseState",
    "DiffeomorphismAuditData",
    "NarrativeCanonicalizationData",
    "Phase1LatticeHandoff",
    "Phase2DiffeomorphismHandoff",
    "NarrativeAgentState",
    "Phase1_SeverityLatticeCollapser",
    "Phase2_SemanticDiffeomorphismCertifier",
    "Phase3_NarrativeCanonicalizationEnforcer",
    "TelemetryNarrativeAgent",
]