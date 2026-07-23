# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Governance Agent (Custodio de la Gobernanza Computacional Federada) ║
║ Ruta   : app/agents/core/governance_agent.py                                 ║
║ Versión: 3.0.0-Hodge-Spectral-Sheaf-Gluing-Strict-Nested                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

EVOLUCIÓN RIGUROSA (Artesanía Senior + Física Matemática de Doctorado):
────────────────────────────────────────────────────────────────────────────────
• Fase 1 → Proyección dual al retículo distributivo acotado + Teoría de la
  Información.
  - [Álgebra de Boole] Se certifica tanto el join (⊔ᵢvᵢ = maxᵢvᵢ) como su dual,
    el meet (⊓ᵢvᵢ = minᵢvᵢ), con las convenciones de identidad de conjunto
    vacío correctas: join(∅) = ⊥ = 0, meet(∅) = ⊤ = 1.
  - [Leyes de De Morgan] Se certifica la identidad de dualidad vía complemento:
        ⊔ᵢ vᵢ = 1 − ⊓ᵢ (1 − vᵢ)
    como guardia de regresión numérica del propio pipeline de cómputo.
  - [Teoría de la información] Se calcula la entropía de Shannon normalizada
    de la distribución de violaciones para distinguir una violación aislada
    crítica de una degradación sistémica difusa — dos estados con idéntico
    supremo pero semántica operativa opuesta.

• Fase 2 → Auditoría cohomológica completa + Teoría de Hodge Discreta.
  - Valida que δ¹ ∘ δ⁰ = 0 (condición de complejo de cocadenas).
  - Calcula H⁰ = ker(δ⁰), H¹ = ker(δ¹)/im(δ⁰), H² = coker(δ¹) y la
    característica de Euler χ = dim C⁰ − dim C¹ + dim C².
  - [Teoría de Hodge discreta / Laplaciano combinatorio] Certifica dim H¹ por
    una SEGUNDA vía, algorítmicamente independiente:
        Δ₁ = δ⁰δ⁰ᵀ + δ¹ᵀδ¹      (Laplaciano de Hodge sobre C¹)
        dim H¹ ≅ dim ker(Δ₁)
    Si ambos métodos (rango-nulidad vía SVD vs. núcleo del Laplaciano vía
    autovalores) no coinciden, se levanta una alarma de *desconfianza
    numérica* distinta de la alarma de *incoherencia ontológica*.

• Fase 3 → Funtor de política espectral en el Topos de Grothendieck.
  - Valida que Ω sea proyector idempotente: Ω² = Ω.
  - Valida opcionalmente hermiticidad/simetría: Ω = Ωᵀ.
  - [Teorema espectral de proyectores] Certifica σ(Ω) ⊆ {0,1}.
  - [Identidad traza-rango] Certifica tr(Ω) = rank(Ω), válida únicamente para
    operadores idempotentes — una certificación cruzada O(n) vs. el O(n³) de
    la SVD.
  - Valida isomorfismo de producto fibrado: S_allowed ≅ Ω X_domain.
  - Añade residual absoluto, relativo y verificación de punto fijo: ΩS = S.
  - [Curry–Howard] Corrige la tipificación de excepciones: los validadores de
    forma de Ω levantan PullbackInputError, no CohomologyInputError.
  - [Axioma de pegado de haces] La síntesis final reemplaza el AND booleano
    opaco por una condición de pegado (gluing axiom) sobre la cubierta
    {U_lattice, U_ontology, U_pullback, U_hodge}, reportando explícitamente
    qué "abierto" obstruye la sección global.

ANIDAMIENTO FUNTORIAL:
────────────────────────────────────────────────────────────────────────────────
Phase1_InformationTheoreticLatticeProjector
  └─ último método: _phase1_terminal_bridge_to_phase2
       └─ Phase2_EulerCharacteristicCohomologyAuditor
            └─ último método: _phase2_terminal_bridge_to_phase3
                 └─ Phase3_SpectralToposPolicyFunctor
                      └─ último método: _phase3_terminal_synthesis
                           └─ GovernanceAgent.execute_federated_governance
"""

from __future__ import annotations

import logging
import math
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Optional, Sequence, Tuple, Type

import numpy as np
from numpy.typing import NDArray

try:
    import scipy.linalg as la
except ImportError:  # pragma: no cover
    import numpy.linalg as la  # type: ignore

try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:  # pragma: no cover

    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos MIC."""
        pass

    class Morphism:
        """Clase base de morfismos del Topos cuando no existe dependencia externa."""
        pass


logger = logging.getLogger("MIC.Strategy.GovernanceAgent")
logger.addHandler(logging.NullHandler())


# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES MATEMÁTICAS, ESPECTRALES Y DE TOLERANCIA
# ══════════════════════════════════════════════════════════════════════════════

_MACHINE_EPSILON: float = float(np.finfo(np.float64).eps)

_LATTICE_TOLERANCE: float = 1e-12
_SVD_TOLERANCE: float = 1e-10
_COCHAIN_TOLERANCE: float = 1e-10
_PULLBACK_TOLERANCE: float = 1e-12
_HODGE_LAPLACIAN_TOLERANCE: float = 1e-9
_SPECTRAL_PROJECTOR_TOLERANCE: float = 1e-9
_TRACE_RANK_TOLERANCE: float = 1e-8

_TOP_LATTICE_THRESHOLD: float = 1.0 - 1e-9


# ══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES ALGEBRAICAS Y COHOMOLÓGICAS
# ══════════════════════════════════════════════════════════════════════════════

class GovernanceAgentError(TopologicalInvariantError):
    """Excepción raíz del Custodio de la Gobernanza Computacional Federada."""
    pass


class LatticeInputError(GovernanceAgentError):
    """Detonada cuando las violaciones normalizadas son inválidas."""
    pass


class StructuralVetoMonad(GovernanceAgentError):
    r"""Detonada si ⊔ᵢ vᵢ → ⊤. El retículo colapsa a la singularidad de rechazo."""
    pass


class CohomologyInputError(GovernanceAgentError):
    """Detonada cuando los operadores de cofrontera son inválidos."""
    pass


class OntologicalParadoxVeto(GovernanceAgentError):
    r"""Detonada si dim H¹(G; F) > 0. Existen contradicciones lógicas en políticas."""
    pass


class HodgeCrossValidationError(GovernanceAgentError):
    r"""
    Detonada si dim H¹ calculado vía rango-nulidad (SVD) difiere de dim H¹
    calculado vía el núcleo del Laplaciano de Hodge combinatorio (autovalores).

    Distinción epistemológica deliberada: esta excepción NO afirma que la
    ontología sea incoherente (eso lo certifica OntologicalParadoxVeto), sino
    que el propio cómputo numérico no es internamente consistente entre dos
    algoritmos independientes — una alarma de desconfianza epistémica, no
    ontológica.
    """
    pass


class PullbackInputError(GovernanceAgentError):
    """Detonada cuando el dominio, el espacio permitido o el proyector son inválidos."""
    pass


class ZeroTrustViolationError(GovernanceAgentError):
    r"""Detonada si el producto fibrado no es isomorfo: S ≇ X ×_Ω 1."""
    pass


class SheafGluingObstructionError(GovernanceAgentError):
    r"""
    Detonada si la condición de pegado (gluing axiom) sobre la cubierta de
    certificaciones {U_lattice, U_ontology, U_pullback, U_hodge} falla: las
    secciones locales de cumplimiento no glosan en una sección global de
    gobernanza consistente.
    """
    pass


# ══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Topos)
# ══════════════════════════════════════════════════════════════════════════════

def _utc_timestamp() -> str:
    """Devuelve marca de tiempo UTC ISO-8601 para trazabilidad auditable."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class LatticeProjectionData:
    r"""
    Artefacto de Fase 1.
    Evaluación dual en el retículo distributivo acotado
    L = ([0,1], ≤, ⊔, ⊓, ⊥, ⊤), enriquecida con teoría de la información.
    """
    supremum_state: float
    infimum_state: float
    bottom_state: float
    top_threshold: float
    violation_count: int
    critical_count: int
    shannon_entropy_bits: float
    max_entropy_bits: float
    de_morgan_duality_residual: float
    is_structurally_sound: bool
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CohomologicalOntologyData:
    r"""
    Artefacto de Fase 2.
    Certificado de consistencia lógica del haz celular de reglas, con
    característica de Euler y validación cruzada vía Teoría de Hodge Discreta.
    """
    dim_H0: int
    dim_H1: int
    dim_H2: int
    betti_0_image: int
    betti_1_kernel: int
    dim_C0: int
    dim_C1: int
    dim_C2: int
    rank_d0: int
    rank_d1: int
    euler_characteristic: int
    dim_H1_hodge: Optional[int]
    hodge_cross_validation_residual: Optional[int]
    hodge_tolerance: float
    cochain_residual: float
    cochain_tolerance: float
    rank_tolerance_d0: float
    rank_tolerance_d1: float
    is_ontologically_coherent: bool
    is_hodge_cross_validated: bool
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ToposPolicyPullbackData:
    r"""
    Artefacto de Fase 3.
    Certificado de isomorfismo del clasificador de subobjetos Ω, enriquecido
    con el teorema espectral de proyectores y la identidad traza-rango.
    """
    pullback_residual: float
    relative_pullback_residual: float
    tolerance: float
    idempotence_residual: float
    symmetry_residual: float
    spectral_binary_residual: float
    projector_trace: float
    trace_rank_residual: Optional[float]
    allowed_fixed_residual: float
    projector_rank: Optional[int]
    domain_rows: int
    domain_columns: int
    is_zero_trust_verified: bool
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FederatedGovernanceState:
    r"""
    Objeto final del endofuntor:
        Z_Gov = Φ₃ ∘ Φ₂ ∘ Φ₁

    La condición de aceptación se formaliza como axioma de pegado (gluing
    axiom) sobre la cubierta de certificaciones locales.
    """
    governance_id: str
    lattice_audit: LatticeProjectionData
    cohomology_audit: CohomologicalOntologyData
    pullback_audit: ToposPolicyPullbackData
    gluing_obstruction_index: int
    gluing_obstruction_failing_charts: Tuple[str, ...]
    is_fully_compliant: bool
    generated_at_utc: str


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1: PROYECCIÓN DUAL AL RETÍCULO DISTRIBUTIVO ACOTADO                    ║
# ║ + TEORÍA DE LA INFORMACIÓN SOBRE LA DISTRIBUCIÓN DE VIOLACIONES            ║
# ║                                                                            ║
# ║ El último método de esta fase es el puente formal hacia Fase 2.            ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase1_InformationTheoreticLatticeProjector:
    r"""
    Aniquila el sistema aritmético de penalizaciones.

    En un retículo distributivo acotado, la agregación de violaciones no es una
    suma compensable, sino un join:
        ⊔ᵢ vᵢ = maxᵢ vᵢ      (identidad de join(∅) = ⊥ = 0)

    Su dual, el meet, se certifica explícitamente:
        ⊓ᵢ vᵢ = minᵢ vᵢ      (identidad de meet(∅) = ⊤ = 1)

    y ambos se relacionan por la ley de De Morgan vía el complemento
    ortocomplementado c(v) = 1 − v:
        ⊔ᵢ vᵢ = 1 − ⊓ᵢ (1 − vᵢ)

    Adicionalmente, la entropía de Shannon de la distribución normalizada de
    violaciones cuantifica si el riesgo está concentrado en una única regla
    (entropía → 0) o difundido sistémicamente (entropía → log₂ n).

    Si el Supremo alcanza ⊤, ninguna otra métrica puede compensar el veto.
    """

    # ────────────────────────────────────────────────────────────────────
    # §1.1 — Certificados de teoría de la información y dualidad booleana
    # ────────────────────────────────────────────────────────────────────

    def _shannon_entropy_bits(
        self,
        values: Sequence[float],
    ) -> Tuple[float, float]:
        r"""
        Calcula la entropía de Shannon (en bits) de la distribución de
        probabilidad inducida por las violaciones normalizadas:

            p_i = v_i / Σⱼ v_j    (solo sobre v_i > 0)
            H(p) = − Σᵢ p_i log₂(p_i)

        Retorna:
            (entropía_bits, entropía_máxima_bits)

        La entropía máxima log₂(n) corresponde a la distribución uniforme
        (degradación sistémica difusa); H = 0 corresponde a una violación
        puntual dominante (fallo aislado y localizable).
        """
        n = len(values)
        if n == 0:
            return 0.0, 0.0

        total = float(sum(values))
        if total <= 0.0:
            return 0.0, float(math.log2(n)) if n > 1 else 0.0

        probabilities = [v / total for v in values if v > 0.0]
        entropy = float(-sum(p * math.log2(p) for p in probabilities))
        max_entropy = float(math.log2(n)) if n > 1 else 0.0

        return entropy, max_entropy

    def _de_morgan_duality_residual(
        self,
        values: Sequence[float],
        supremum: float,
    ) -> float:
        r"""
        Certifica la ley de De Morgan del retículo acotado vía complemento:

            ⊔ᵢ vᵢ = 1 − ⊓ᵢ (1 − vᵢ)

        Este residual debe ser ≈0 por identidad algebraica exacta de
        min/max; una desviación mayor a la precisión de máquina delata un
        defecto en el propio pipeline de cómputo (guardia de regresión).
        """
        if not values:
            return 0.0

        complement_values = [1.0 - v for v in values]
        join_via_de_morgan = 1.0 - min(complement_values)

        return float(abs(join_via_de_morgan - supremum))

    # ────────────────────────────────────────────────────────────────────
    # §1.2 — Proyección al retículo (implementación real)
    # ────────────────────────────────────────────────────────────────────

    def _project_to_distributive_lattice(
        self,
        normalized_violations: Sequence[float],
        *,
        top_threshold: float = _TOP_LATTICE_THRESHOLD,
        lattice_tolerance: float = _LATTICE_TOLERANCE,
        raise_on_veto: bool = True,
    ) -> LatticeProjectionData:
        """
        Proyecta las violaciones normalizadas al retículo [0,1], certificando
        join, meet, dualidad de De Morgan y entropía de la distribución.

        Contrato:
            ∀vᵢ ∈ [0,1], finito.
            supremum = max(vᵢ) si existen violaciones; 0.0 (⊥) si vacío.
            infimum  = min(vᵢ) si existen violaciones; 1.0 (⊤) si vacío.
            veto si supremum ≥ top_threshold.
        """
        if normalized_violations is None:
            raise LatticeInputError("normalized_violations no puede ser None.")

        if not np.isfinite(lattice_tolerance) or lattice_tolerance < 0.0:
            raise LatticeInputError(
                f"lattice_tolerance debe ser finito y ≥ 0. Se recibió {lattice_tolerance}."
            )

        if not np.isfinite(top_threshold):
            raise LatticeInputError(
                f"top_threshold debe ser finito. Se recibió {top_threshold}."
            )

        tolerance = float(max(float(lattice_tolerance), _MACHINE_EPSILON))

        if top_threshold <= 0.0 or top_threshold > 1.0 + tolerance:
            raise LatticeInputError(
                f"top_threshold debe pertenecer a (0, 1+tol]. "
                f"Se recibió {top_threshold} con tol={tolerance:.6e}."
            )

        raw_values: List[float] = []

        if isinstance(normalized_violations, (str, bytes)):
            raise LatticeInputError(
                "normalized_violations no puede ser una cadena. "
                "Se esperaba una secuencia numérica o un escalar numérico."
            )

        if isinstance(normalized_violations, (float, int, np.floating, np.integer)):
            raw_values = [float(normalized_violations)]
        else:
            try:
                iterator = iter(normalized_violations)
            except TypeError:
                try:
                    raw_values = [float(normalized_violations)]
                except Exception as exc:
                    raise LatticeInputError(
                        "normalized_violations debe ser un escalar numérico o una "
                        "secuencia iterable de números."
                    ) from exc
            else:
                try:
                    raw_values = [float(value) for value in iterator]
                except Exception as exc:
                    raise LatticeInputError(
                        f"No fue posible convertir normalized_violations a float: {exc}"
                    ) from exc

        for idx, value in enumerate(raw_values):
            if not np.isfinite(value):
                raise LatticeInputError(
                    f"normalized_violations[{idx}] no es finito: {value}."
                )

            if value < -tolerance or value > 1.0 + tolerance:
                raise LatticeInputError(
                    f"normalized_violations[{idx}]={value} fuera de [0,1] "
                    f"con tolerancia {tolerance:.6e}."
                )

        clamped_values: List[float] = [
            float(np.clip(value, 0.0, 1.0))
            for value in raw_values
        ]

        # Join (⊔) con identidad de conjunto vacío = ⊥ = 0.
        supremum = float(max(clamped_values)) if clamped_values else 0.0

        # Meet (⊓) dual, con identidad de conjunto vacío = ⊤ = 1.
        infimum = float(min(clamped_values)) if clamped_values else 1.0

        critical_count = int(
            sum(1 for value in clamped_values if value >= top_threshold)
        )

        entropy_bits, max_entropy_bits = self._shannon_entropy_bits(clamped_values)
        de_morgan_residual = self._de_morgan_duality_residual(clamped_values, supremum)

        is_structurally_sound = bool(supremum < top_threshold)

        notes: List[str] = []
        if clamped_values and supremum == 0.0:
            notes.append("Todas las violaciones fueron ⊥ = 0.0.")

        de_morgan_alarm_tolerance = max(tolerance, 1e-9)
        if de_morgan_residual > de_morgan_alarm_tolerance:
            notes.append(
                f"Anomalía de dualidad de De Morgan detectada en el pipeline: "
                f"residuo = {de_morgan_residual:.6e} > "
                f"tol={de_morgan_alarm_tolerance:.6e}. Revise el cómputo de "
                f"clamp/normalización."
            )

        audit = LatticeProjectionData(
            supremum_state=supremum,
            infimum_state=infimum,
            bottom_state=0.0,
            top_threshold=float(top_threshold),
            violation_count=len(clamped_values),
            critical_count=critical_count,
            shannon_entropy_bits=entropy_bits,
            max_entropy_bits=max_entropy_bits,
            de_morgan_duality_residual=de_morgan_residual,
            is_structurally_sound=is_structurally_sound,
            notes=tuple(notes),
        )

        if not is_structurally_sound and raise_on_veto:
            raise StructuralVetoMonad(
                f"Colapso de retículo: la evaluación de gobernanza alcanzó el estado "
                f"absorbente ⊤. Supremo = {supremum:.6f} ≥ top_threshold = "
                f"{top_threshold:.6f}. Entropía = {entropy_bits:.4f}/"
                f"{max_entropy_bits:.4f} bits. Veto estructural incondicional."
            )

        return audit

    # ────────────────────────────────────────────────────────────────────
    # §1.3 — Puente terminal hacia Fase 2 (stub + composición)
    # ────────────────────────────────────────────────────────────────────

    def _audit_semantic_ontology(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> CohomologicalOntologyData:
        """
        Stub formal de continuación hacia Fase 2.

        La implementación real vive en Phase2_EulerCharacteristicCohomologyAuditor.
        """
        raise NotImplementedError(
            "Phase 2 must be mixed in. "
            "Este método es la continuación formal hacia la auditoría cohomológica."
        )

    def _phase1_terminal_bridge_to_phase2(
        self,
        normalized_violations: Sequence[float],
        boundary_d0: NDArray[Any],
        boundary_d1: NDArray[Any],
        *,
        top_threshold: float = _TOP_LATTICE_THRESHOLD,
        lattice_tolerance: float = _LATTICE_TOLERANCE,
        svd_tolerance: float = _SVD_TOLERANCE,
        cochain_tolerance: float = _COCHAIN_TOLERANCE,
        require_cochain_complex: bool = True,
        hodge_tolerance: float = _HODGE_LAPLACIAN_TOLERANCE,
        require_hodge_consistency: bool = True,
        raise_on_veto: bool = True,
    ) -> Tuple[LatticeProjectionData, CohomologicalOntologyData]:
        """
        Último método de Fase 1: puente funtorial hacia Fase 2.

        Composición:
            Φ₁(violations) → LatticeProjectionData
            Φ₁ ▷ Φ₂(δ⁰, δ¹) → (LatticeProjectionData, CohomologicalOntologyData)
        """
        lattice_audit = self._project_to_distributive_lattice(
            normalized_violations,
            top_threshold=top_threshold,
            lattice_tolerance=lattice_tolerance,
            raise_on_veto=raise_on_veto,
        )

        cohomology_audit = self._audit_semantic_ontology(
            boundary_d0,
            boundary_d1,
            svd_tolerance=svd_tolerance,
            cochain_tolerance=cochain_tolerance,
            require_cochain_complex=require_cochain_complex,
            hodge_tolerance=hodge_tolerance,
            require_hodge_consistency=require_hodge_consistency,
            raise_on_veto=raise_on_veto,
        )

        return lattice_audit, cohomology_audit


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2: AUDITORÍA COHOMOLÓGICA COMPLETA + TEORÍA DE HODGE DISCRETA          ║
# ║ Exige dim H¹(G; F) = 0 y valida el resultado por doble vía numérica.        ║
# ║                                                                            ║
# ║ El último método de esta fase es el puente formal hacia Fase 3.            ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase2_EulerCharacteristicCohomologyAuditor(Phase1_InformationTheoreticLatticeProjector):
    r"""
    Modela las directrices de gobernanza como un complejo de cocadenas:

        C⁰ --δ⁰--> C¹ --δ¹--> C²

    Condición de complejo:
        δ¹ ∘ δ⁰ = 0.

    Grupos de cohomología (vía rango-nulidad):
        H⁰ = ker(δ⁰),               dim H⁰ = dim C⁰ − rank(δ⁰)
        H¹ = ker(δ¹) / im(δ⁰),      dim H¹ = (dim C¹ − rank δ¹) − rank δ⁰
        H² = coker(δ¹) = C²/im(δ¹), dim H² = dim C² − rank(δ¹)

    Característica de Euler (invariante topológico del complejo):
        χ = dim C⁰ − dim C¹ + dim C² = dim H⁰ − dim H¹ + dim H²

    Validación cruzada independiente (Teoría de Hodge Discreta):
        Δ₁ = δ⁰ δ⁰ᵀ + δ¹ᵀ δ¹    (Laplaciano combinatorio sobre C¹)
        dim H¹ ≅ dim ker(Δ₁)     (si δ¹δ⁰ = 0)
    """

    # ────────────────────────────────────────────────────────────────────
    # §2.1 — Utilidades numéricas genéricas (tipificadas por dominio)
    # ────────────────────────────────────────────────────────────────────

    def _as_finite_matrix(
        self,
        name: str,
        value: NDArray[Any],
        *,
        exception_cls: Type[GovernanceAgentError] = CohomologyInputError,
    ) -> NDArray[np.float64]:
        """
        Convierte una entrada a matriz 2D float64 finita.

        Mejora rigurosa (Curry-Howard): el tipo de excepción es inyectable,
        de modo que el dominio semántico que invoca este validador (Fase 2:
        cohomología, Fase 3: pullback) determina la categoría de fallo, en
        vez de heredar siempre `CohomologyInputError` por accidente de
        herencia de mixin.
        """
        matrix = np.asarray(value, dtype=np.float64)

        if matrix.ndim != 2:
            raise exception_cls(
                f"{name} debe ser una matriz 2D. Se recibió ndim={matrix.ndim}."
            )

        if not np.all(np.isfinite(matrix)):
            raise exception_cls(
                f"{name} contiene valores no finitos (NaN/Inf)."
            )

        return matrix

    def _frobenius_norm(
        self,
        matrix: NDArray[np.float64],
    ) -> float:
        """
        Calcula la norma de Frobenius de forma segura para matrices vacías.
        """
        if matrix.size == 0:
            return 0.0
        return float(la.norm(matrix, ord="fro"))

    def _svd_rank(
        self,
        name: str,
        matrix: NDArray[np.float64],
        base_tolerance: float,
        *,
        exception_cls: Type[GovernanceAgentError] = CohomologyInputError,
    ) -> Tuple[int, float, float]:
        """
        Calcula rango numérico vía SVD con tolerancia dinámica.

        Modelo de tolerancia:
            tol = max(base, 10 · max(m,n) · ε_machine · max(1, σ_max))

        Retorna:
            (rank, σ_max, tol)
        """
        if matrix.size == 0 or min(matrix.shape) == 0:
            return 0, 0.0, float(base_tolerance)

        try:
            singular_values = la.svd(matrix, compute_uv=False)
        except Exception as exc:
            raise exception_cls(
                f"No fue posible calcular SVD de {name}: {exc}"
            ) from exc

        max_singular = (
            float(np.max(singular_values))
            if singular_values.size > 0
            else 0.0
        )

        scale = max(float(max(matrix.shape)), 1.0)
        tolerance = float(
            max(
                float(base_tolerance),
                10.0 * scale * _MACHINE_EPSILON * max(1.0, max_singular),
            )
        )

        rank = int(np.sum(singular_values > tolerance))
        return rank, max_singular, tolerance

    # ────────────────────────────────────────────────────────────────────
    # §2.2 — Validación cruzada vía Teoría de Hodge Discreta
    # ────────────────────────────────────────────────────────────────────

    def _hodge_laplacian_kernel_dimension(
        self,
        d0: NDArray[np.float64],
        d1: NDArray[np.float64],
        tolerance_scale: float,
    ) -> Tuple[int, float]:
        r"""
        Calcula dim ker(Δ₁) del Laplaciano de Hodge combinatorio:

            Δ₁ = δ⁰ δ⁰ᵀ + δ¹ᵀ δ¹  :  C¹ → C¹

        Δ₁ es simétrica y semidefinida positiva por construcción (suma de dos
        formas de Gram), por lo que sus autovalores son reales y no
        negativos hasta error de máquina. Por Teoría de Hodge discreta, si
        δ¹δ⁰ = 0, entonces ker(Δ₁) es isomorfo al espacio de cocadenas
        armónicas, que representa canónicamente H¹.

        Este cómputo es algorítmicamente independiente de `_svd_rank`: aquí
        se diagonaliza un único operador simétrico combinado, en vez de
        calcular dos SVDs por separado y combinar rangos por aritmética
        entera — un cambio de "familia numérica" que sirve como validación
        cruzada genuina.

        Retorna:
            (dim_ker_Δ₁, tolerancia_efectiva)
        """
        dim_C1 = d0.shape[0]

        if dim_C1 == 0:
            return 0, float(tolerance_scale)

        laplacian = d0 @ d0.T + d1.T @ d1
        eigenvalues = np.linalg.eigvalsh(laplacian)

        max_eigenvalue = float(np.max(eigenvalues)) if eigenvalues.size > 0 else 0.0
        tolerance = float(
            max(
                float(tolerance_scale),
                100.0 * dim_C1 * _MACHINE_EPSILON * max(1.0, max_eigenvalue),
            )
        )

        kernel_dimension = int(np.sum(np.abs(eigenvalues) <= tolerance))
        return kernel_dimension, tolerance

    # ────────────────────────────────────────────────────────────────────
    # §2.3 — Auditoría cohomológica completa (implementación real)
    # ────────────────────────────────────────────────────────────────────

    def _audit_semantic_ontology(
        self,
        boundary_d0: NDArray[Any],
        boundary_d1: NDArray[Any],
        *,
        svd_tolerance: float = _SVD_TOLERANCE,
        cochain_tolerance: float = _COCHAIN_TOLERANCE,
        require_cochain_complex: bool = True,
        hodge_tolerance: float = _HODGE_LAPLACIAN_TOLERANCE,
        require_hodge_consistency: bool = True,
        raise_on_veto: bool = True,
    ) -> CohomologicalOntologyData:
        """
        Audita la consistencia cohomológica de la ontología de gobierno,
        certificando además la característica de Euler y la validación
        cruzada de dim H¹ vía el Laplaciano de Hodge.

        Contrato:
            boundary_d0: δ⁰: C⁰ → C¹, shape (dim_C1, dim_C0)
            boundary_d1: δ¹: C¹ → C², shape (dim_C2, dim_C1)

        Exigencia semántica (ontológica):
            δ¹δ⁰ = 0 y dim H¹ = 0.

        Exigencia epistémica (numérica), independiente de la anterior:
            dim H¹ (vía SVD) == dim H¹ (vía núcleo del Laplaciano de Hodge).
        """
        if not np.isfinite(svd_tolerance) or svd_tolerance < 0.0:
            raise CohomologyInputError(
                f"svd_tolerance debe ser finito y ≥ 0. Se recibió {svd_tolerance}."
            )

        if not np.isfinite(cochain_tolerance) or cochain_tolerance < 0.0:
            raise CohomologyInputError(
                f"cochain_tolerance debe ser finito y ≥ 0. Se recibió {cochain_tolerance}."
            )

        if not np.isfinite(hodge_tolerance) or hodge_tolerance < 0.0:
            raise CohomologyInputError(
                f"hodge_tolerance debe ser finito y ≥ 0. Se recibió {hodge_tolerance}."
            )

        d0 = self._as_finite_matrix("boundary_d0", boundary_d0)
        d1 = self._as_finite_matrix("boundary_d1", boundary_d1)

        dim_C1 = int(d0.shape[0])
        dim_C0 = int(d0.shape[1])
        dim_C2 = int(d1.shape[0])

        if d1.shape[1] != dim_C1:
            raise CohomologyInputError(
                f"Dimensión incompatible del complejo de cocadenas: "
                f"boundary_d0 tiene shape={d0.shape} y boundary_d1 shape={d1.shape}. "
                f"Se requiere boundary_d1.shape[1] == boundary_d0.shape[0] = {dim_C1}."
            )

        rank_d0, _, rank_tol_d0 = self._svd_rank(
            "boundary_d0",
            d0,
            float(svd_tolerance),
        )
        rank_d1, _, rank_tol_d1 = self._svd_rank(
            "boundary_d1",
            d1,
            float(svd_tolerance),
        )

        composition = d1 @ d0
        cochain_residual = self._frobenius_norm(composition)

        norm_d0 = self._frobenius_norm(d0)
        norm_d1 = self._frobenius_norm(d1)

        scale = max(1.0, norm_d0, norm_d1, norm_d0 * norm_d1)
        max_dim = max(dim_C0, dim_C1, dim_C2, 1)

        effective_cochain_tolerance = float(
            max(
                float(cochain_tolerance),
                100.0 * float(max_dim) * _MACHINE_EPSILON * scale,
            )
        )

        notes: List[str] = []
        complex_valid = True
        complex_condition_holds = cochain_residual <= effective_cochain_tolerance

        if not complex_condition_holds:
            if require_cochain_complex:
                complex_valid = False
                notes.append(
                    f"δ¹δ⁰ ≠ 0: ||δ¹δ⁰||_F = {cochain_residual:.6e} > "
                    f"tol={effective_cochain_tolerance:.6e}."
                )
            else:
                notes.append(
                    f"δ¹δ⁰ ≠ 0 pero require_cochain_complex=False: "
                    f"||δ¹δ⁰||_F = {cochain_residual:.6e}."
                )

        # --- Cohomología vía rango-nulidad (SVD) ---
        dim_ker_d1 = dim_C1 - rank_d1
        dim_H1_raw = dim_ker_d1 - rank_d0

        if dim_H1_raw < 0:
            dim_H1 = 0
            notes.append(
                f"dim_H1_raw={dim_H1_raw} negativo; se clampó a 0 por tolerancia "
                f"numérica de rango. Revise rank_d0={rank_d0}, rank_d1={rank_d1}, "
                f"dim_ker_d1={dim_ker_d1}."
            )
        else:
            dim_H1 = int(dim_H1_raw)

        dim_H0 = int(max(0, dim_C0 - rank_d0))
        dim_H2 = int(max(0, dim_C2 - rank_d1))

        euler_characteristic = int(dim_C0 - dim_C1 + dim_C2)
        euler_from_betti = int(dim_H0 - dim_H1 + dim_H2)

        if euler_from_betti != euler_characteristic:  # pragma: no cover — invariante algebraico
            notes.append(
                f"Discrepancia inesperada en la característica de Euler: "
                f"χ_dimensional={euler_characteristic} ≠ "
                f"χ_Betti={euler_from_betti}. Este invariante debería coincidir "
                f"tautológicamente por el teorema de rango-nulidad; su fallo "
                f"delata un error de programación, no de datos."
            )

        # --- Validación cruzada vía Teoría de Hodge Discreta ---
        dim_H1_hodge: Optional[int] = None
        hodge_cross_validation_residual: Optional[int] = None
        is_hodge_cross_validated = True
        effective_hodge_tolerance = float(hodge_tolerance)

        if complex_condition_holds:
            dim_H1_hodge, effective_hodge_tolerance = self._hodge_laplacian_kernel_dimension(
                d0, d1, float(hodge_tolerance)
            )
            hodge_cross_validation_residual = abs(dim_H1 - dim_H1_hodge)
            is_hodge_cross_validated = hodge_cross_validation_residual == 0

            if not is_hodge_cross_validated:
                notes.append(
                    f"Inconsistencia epistémica: dim H¹ vía SVD = {dim_H1}, "
                    f"dim H¹ vía núcleo del Laplaciano de Hodge = {dim_H1_hodge} "
                    f"(Δ₁ = δ⁰δ⁰ᵀ + δ¹ᵀδ¹). Los dos métodos numéricos "
                    f"independientes discrepan."
                )
        else:
            notes.append(
                "Validación de Hodge omitida: la condición de complejo "
                "δ¹δ⁰ = 0 no se satisface dentro de tolerancia, por lo que "
                "ker(Δ₁) no es necesariamente isomorfo a H¹."
            )

        is_ontologically_coherent = bool(complex_valid and dim_H1 == 0)

        audit = CohomologicalOntologyData(
            dim_H0=dim_H0,
            dim_H1=dim_H1,
            dim_H2=dim_H2,
            betti_0_image=rank_d0,
            betti_1_kernel=dim_ker_d1,
            dim_C0=dim_C0,
            dim_C1=dim_C1,
            dim_C2=dim_C2,
            rank_d0=rank_d0,
            rank_d1=rank_d1,
            euler_characteristic=euler_characteristic,
            dim_H1_hodge=dim_H1_hodge,
            hodge_cross_validation_residual=hodge_cross_validation_residual,
            hodge_tolerance=effective_hodge_tolerance,
            cochain_residual=cochain_residual,
            cochain_tolerance=effective_cochain_tolerance,
            rank_tolerance_d0=rank_tol_d0,
            rank_tolerance_d1=rank_tol_d1,
            is_ontologically_coherent=is_ontologically_coherent,
            is_hodge_cross_validated=is_hodge_cross_validated,
            notes=tuple(notes),
        )

        if not is_ontologically_coherent and raise_on_veto:
            if dim_H1 > 0:
                raise OntologicalParadoxVeto(
                    f"Paradoja ontológica: se detectó un ciclo cohomológico no trivial. "
                    f"dim H¹ = {dim_H1} > 0. Existen reglas de gobernanza "
                    f"contradictorias en los contratos de datos."
                )

            detail = " | ".join(notes) if notes else "Complejo de cocadenas inválido."
            raise OntologicalParadoxVeto(
                f"Inconsistencia cohomológica estructural: {detail}"
            )

        if not is_hodge_cross_validated and require_hodge_consistency and raise_on_veto:
            raise HodgeCrossValidationError(
                f"Desconfianza epistémica en el cómputo cohomológico: dim H¹ "
                f"vía SVD ({dim_H1}) ≠ dim H¹ vía Laplaciano de Hodge "
                f"({dim_H1_hodge}). Residuo = {hodge_cross_validation_residual}."
            )

        return audit

    # ────────────────────────────────────────────────────────────────────
    # §2.4 — Puente terminal hacia Fase 3 (stub + composición)
    # ────────────────────────────────────────────────────────────────────

    def _validate_policy_pullback(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> ToposPolicyPullbackData:
        """
        Stub formal de continuación hacia Fase 3.

        La implementación real vive en Phase3_SpectralToposPolicyFunctor.
        """
        raise NotImplementedError(
            "Phase 3 must be mixed in. "
            "Este método es la continuación formal hacia el pullback en el Topos."
        )

    def _phase2_terminal_bridge_to_phase3(
        self,
        normalized_violations: Sequence[float],
        boundary_d0: NDArray[Any],
        boundary_d1: NDArray[Any],
        X_domain: NDArray[Any],
        S_allowed: NDArray[Any],
        Omega_projector: NDArray[Any],
        *,
        top_threshold: float = _TOP_LATTICE_THRESHOLD,
        lattice_tolerance: float = _LATTICE_TOLERANCE,
        svd_tolerance: float = _SVD_TOLERANCE,
        cochain_tolerance: float = _COCHAIN_TOLERANCE,
        require_cochain_complex: bool = True,
        hodge_tolerance: float = _HODGE_LAPLACIAN_TOLERANCE,
        require_hodge_consistency: bool = True,
        pullback_tolerance: float = _PULLBACK_TOLERANCE,
        require_projector: bool = True,
        require_hermitian_projector: bool = True,
        require_spectral_projector: bool = True,
        raise_on_veto: bool = True,
    ) -> Tuple[
        LatticeProjectionData,
        CohomologicalOntologyData,
        ToposPolicyPullbackData,
    ]:
        """
        Último método de Fase 2: puente funtorial hacia Fase 3.

        Composición:
            Φ₂ ∘ Φ₁ → (LatticeProjectionData, CohomologicalOntologyData)
            (Φ₂ ∘ Φ₁) ▷ Φ₃(X, S, Ω) → (LatticeProjectionData,
                                        CohomologicalOntologyData,
                                        ToposPolicyPullbackData)
        """
        lattice_audit, cohomology_audit = self._phase1_terminal_bridge_to_phase2(
            normalized_violations,
            boundary_d0,
            boundary_d1,
            top_threshold=top_threshold,
            lattice_tolerance=lattice_tolerance,
            svd_tolerance=svd_tolerance,
            cochain_tolerance=cochain_tolerance,
            require_cochain_complex=require_cochain_complex,
            hodge_tolerance=hodge_tolerance,
            require_hodge_consistency=require_hodge_consistency,
            raise_on_veto=raise_on_veto,
        )

        pullback_audit = self._validate_policy_pullback(
            X_domain,
            S_allowed,
            Omega_projector,
            pullback_tolerance=pullback_tolerance,
            require_projector=require_projector,
            require_hermitian_projector=require_hermitian_projector,
            require_spectral_projector=require_spectral_projector,
            raise_on_veto=raise_on_veto,
        )

        return lattice_audit, cohomology_audit, pullback_audit


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3: FUNTOR DE POLÍTICA ESPECTRAL EN EL TOPOS DE GROTHENDIECK            ║
# ║ Exige isomorfismo pullback: S ≅ X ×_Ω 1, certificado espectralmente.        ║
# ║                                                                            ║
# ║ El último método de esta fase sintetiza el objeto final de gobernanza      ║
# ║ mediante un axioma de pegado (gluing axiom) sobre haces.                   ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase3_SpectralToposPolicyFunctor(Phase2_EulerCharacteristicCohomologyAuditor):
    r"""
    Aplica Policy-as-Code evaluando el clasificador de subobjetos Ω.

    En un Topos, un subobjeto S ↪ X está clasificado por un morfismo
    característico χ_S: X → Ω. En la linealización matricial adoptada,
    Ω actúa como proyector idempotente y el espacio autorizado se reconstruye:

        S_allowed = Ω X_domain.

    Condiciones estructurales:
        Ω² = Ω                    (idempotencia)
        Ω = Ωᵀ                    (opcional, proyector ortogonal)
        σ(Ω) ⊆ {0,1}              (teorema espectral de proyectores)
        tr(Ω) = rank(Ω)           (identidad traza-rango de operadores idempotentes)
        S = ΩX                    (pullback)
        ΩS = S                    (punto fijo)
    """

    # ────────────────────────────────────────────────────────────────────
    # §3.1 — Utilidades de forma (tipificadas para el dominio de pullback)
    # ────────────────────────────────────────────────────────────────────

    def _as_finite_2d_or_column(
        self,
        name: str,
        value: NDArray[Any],
    ) -> NDArray[np.float64]:
        """
        Convierte vectores a matriz columna y valida matrices 2D finitas.
        """
        matrix = np.asarray(value, dtype=np.float64)

        if matrix.ndim == 0:
            matrix = matrix.reshape(1, 1)
        elif matrix.ndim == 1:
            matrix = matrix.reshape(-1, 1)
        elif matrix.ndim != 2:
            raise PullbackInputError(
                f"{name} debe ser vector, matriz columna o matriz 2D. "
                f"Se recibió ndim={matrix.ndim}."
            )

        if not np.all(np.isfinite(matrix)):
            raise PullbackInputError(
                f"{name} contiene valores no finitos (NaN/Inf)."
            )

        return matrix

    # ────────────────────────────────────────────────────────────────────
    # §3.2 — Certificados espectrales del proyector Ω
    # ────────────────────────────────────────────────────────────────────

    def _certify_projector_spectral_binary(
        self,
        Omega: NDArray[np.float64],
        tolerance: float,
    ) -> float:
        r"""
        Teorema espectral para proyectores ortogonales hermíticos:
            σ(Ω) ⊆ {0, 1}.

        Certificación algorítmicamente independiente de la idempotencia por
        norma de Frobenius: aquí se diagonaliza el operador simetrizado y se
        mide la distancia de cada autovalor al espectro binario admisible.

        Retorna el residuo espectral máximo:
            max_k min(|λ_k|, |λ_k − 1|)
        """
        n = Omega.shape[0]
        if n == 0:
            return 0.0

        Omega_sym = 0.5 * (Omega + Omega.T)
        eigenvalues = np.linalg.eigvalsh(Omega_sym)

        residual_to_zero = np.abs(eigenvalues)
        residual_to_one = np.abs(eigenvalues - 1.0)
        residual = np.minimum(residual_to_zero, residual_to_one)

        return float(np.max(residual)) if residual.size > 0 else 0.0

    def _certify_trace_rank_identity(
        self,
        Omega: NDArray[np.float64],
        projector_rank: Optional[int],
    ) -> Optional[float]:
        r"""
        Identidad algebraica de operadores idempotentes:
            tr(Ω) = Σ_k λ_k = Σ_k 1_{λ_k=1} = rank(Ω).

        Esta es una certificación de costo O(n) (traza) contrastada contra
        una de costo O(n³) (rango vía SVD, ya calculado aguas arriba) — un
        patrón clásico de "chequeo barato contra chequeo caro" en álgebra
        lineal numérica.

        Retorna None si el rango no pudo calcularse (propagación de
        incertidumbre en vez de un valor centinela engañoso).
        """
        if Omega.shape[0] == 0:
            return 0.0

        if projector_rank is None:
            return None

        trace_value = float(np.trace(Omega))
        return float(abs(trace_value - float(projector_rank)))

    # ────────────────────────────────────────────────────────────────────
    # §3.3 — Validación del pullback (implementación real, Fase 3)
    # ────────────────────────────────────────────────────────────────────

    def _validate_policy_pullback(
        self,
        X_domain: NDArray[Any],
        S_allowed: NDArray[Any],
        Omega_projector: NDArray[Any],
        *,
        pullback_tolerance: float = _PULLBACK_TOLERANCE,
        require_projector: bool = True,
        require_hermitian_projector: bool = True,
        require_spectral_projector: bool = True,
        raise_on_veto: bool = True,
    ) -> ToposPolicyPullbackData:
        """
        Verifica que la autorización OPA (Ω) reconstruya axiomáticamente el
        espacio permitido S desde el dominio X, certificando estructura por
        cuatro vías independientes: Frobenius (idempotencia/simetría),
        espectral (teorema de proyectores) y traza-rango.

        Contrato:
            Ω ∈ R^{n×n}
            X ∈ R^{n×k}
            S ∈ R^{n×k}
            S ≈ ΩX
        """
        if not np.isfinite(pullback_tolerance) or pullback_tolerance < 0.0:
            raise PullbackInputError(
                f"pullback_tolerance debe ser finito y ≥ 0. Se recibió {pullback_tolerance}."
            )

        X = self._as_finite_2d_or_column("X_domain", X_domain)
        S = self._as_finite_2d_or_column("S_allowed", S_allowed)

        # Curry-Howard: la validación de forma de Ω pertenece al dominio de
        # pullback/Topos, por lo que su fallo debe levantar PullbackInputError,
        # no CohomologyInputError (defecto corregido de la v2.0.0).
        Omega = self._as_finite_matrix(
            "Omega_projector", Omega_projector, exception_cls=PullbackInputError
        )

        if Omega.shape[0] != Omega.shape[1]:
            raise PullbackInputError(
                f"Omega_projector debe ser cuadrado. Se recibió shape={Omega.shape}."
            )

        n = int(Omega.shape[0])
        k = int(X.shape[1])

        if X.shape[0] != n:
            raise PullbackInputError(
                f"X_domain.shape[0] debe coincidir con Omega_projector.shape[0]. "
                f"Se recibió X.shape={X.shape}, Omega.shape={Omega.shape}."
            )

        if S.shape != X.shape:
            raise PullbackInputError(
                f"S_allowed debe tener la misma forma que X_domain. "
                f"Se recibió S.shape={S.shape}, X.shape={X.shape}."
            )

        omega_norm = self._frobenius_norm(Omega)
        x_norm = self._frobenius_norm(X)
        s_norm = self._frobenius_norm(S)

        scale = max(1.0, omega_norm, x_norm, s_norm)
        max_dim = max(n, k, 1)

        tolerance = float(
            max(
                float(pullback_tolerance),
                100.0 * float(max_dim) * _MACHINE_EPSILON * scale,
            )
        )

        violations: List[str] = []

        # --- Certificación por norma de Frobenius ---
        Omega_squared = Omega @ Omega if n > 0 else Omega.copy()
        idempotence_residual = self._frobenius_norm(Omega_squared - Omega)

        if require_projector and idempotence_residual > tolerance:
            violations.append(
                f"Ω no es idempotente: ||Ω² − Ω||_F = "
                f"{idempotence_residual:.6e} > tol={tolerance:.6e}."
            )

        symmetry_residual = 0.0
        if require_hermitian_projector:
            symmetry_residual = self._frobenius_norm(Omega - Omega.T)
            if symmetry_residual > tolerance:
                violations.append(
                    f"Ω no es simétrico/hermitiano real: ||Ω − Ωᵀ||_F = "
                    f"{symmetry_residual:.6e} > tol={tolerance:.6e}."
                )

        # --- Certificación espectral independiente ---
        spectral_tolerance = max(tolerance, _SPECTRAL_PROJECTOR_TOLERANCE)
        spectral_binary_residual = self._certify_projector_spectral_binary(
            Omega, spectral_tolerance
        )

        if require_spectral_projector and spectral_binary_residual > spectral_tolerance:
            violations.append(
                f"Espectro de Ω fuera de {{0,1}} (teorema espectral de "
                f"proyectores): residuo = {spectral_binary_residual:.6e} > "
                f"tol={spectral_tolerance:.6e}."
            )

        # --- Certificación del pullback propiamente dicho ---
        pullback_projection = Omega @ X if n > 0 else X.copy()
        pullback_residual = self._frobenius_norm(S - pullback_projection)

        projection_norm = self._frobenius_norm(pullback_projection)
        relative_pullback_residual = float(
            pullback_residual / max(1.0, s_norm, projection_norm)
        )

        if pullback_residual > tolerance:
            violations.append(
                f"Producto fibrado no isomorfo: ||S − ΩX||_F = "
                f"{pullback_residual:.6e} > tol={tolerance:.6e}."
            )

        allowed_fixed_projection = Omega @ S if n > 0 else S.copy()
        allowed_fixed_residual = self._frobenius_norm(S - allowed_fixed_projection)

        if allowed_fixed_residual > tolerance:
            violations.append(
                f"El espacio permitido no es punto fijo de Ω: ||S − ΩS||_F = "
                f"{allowed_fixed_residual:.6e} > tol={tolerance:.6e}."
            )

        # --- Certificación traza-rango (barata, O(n), independiente) ---
        projector_rank: Optional[int]
        try:
            projector_rank = (
                int(np.linalg.matrix_rank(Omega, tol=tolerance))
                if n > 0
                else 0
            )
        except Exception:  # pragma: no cover
            projector_rank = None

        projector_trace = float(np.trace(Omega)) if n > 0 else 0.0
        trace_rank_residual = self._certify_trace_rank_identity(Omega, projector_rank)

        trace_rank_tolerance = max(tolerance, _TRACE_RANK_TOLERANCE)
        if (
            require_projector
            and trace_rank_residual is not None
            and trace_rank_residual > trace_rank_tolerance
        ):
            violations.append(
                f"Identidad traza-rango violada: |tr(Ω) − rank(Ω)| = "
                f"{trace_rank_residual:.6e} > tol={trace_rank_tolerance:.6e}. "
                f"tr(Ω)={projector_trace:.6f}, rank(Ω)={projector_rank}."
            )

        is_zero_trust_verified = len(violations) == 0

        audit = ToposPolicyPullbackData(
            pullback_residual=pullback_residual,
            relative_pullback_residual=relative_pullback_residual,
            tolerance=tolerance,
            idempotence_residual=idempotence_residual,
            symmetry_residual=symmetry_residual,
            spectral_binary_residual=spectral_binary_residual,
            projector_trace=projector_trace,
            trace_rank_residual=trace_rank_residual,
            allowed_fixed_residual=allowed_fixed_residual,
            projector_rank=projector_rank,
            domain_rows=n,
            domain_columns=k,
            is_zero_trust_verified=is_zero_trust_verified,
            notes=tuple(violations),
        )

        if not is_zero_trust_verified and raise_on_veto:
            detail = " | ".join(violations)
            raise ZeroTrustViolationError(
                f"Violación Zero-Trust (fuga de Gauge): {detail}"
            )

        return audit

    # ────────────────────────────────────────────────────────────────────
    # §3.4 — Axioma de pegado de haces (gluing axiom)
    # ────────────────────────────────────────────────────────────────────

    def _certify_sheaf_gluing_obstruction(
        self,
        lattice_audit: LatticeProjectionData,
        cohomology_audit: CohomologicalOntologyData,
        pullback_audit: ToposPolicyPullbackData,
    ) -> Tuple[int, Tuple[str, ...]]:
        r"""
        Formaliza la síntesis final como un axioma de pegado (gluing axiom)
        sobre una cubierta {U_lattice, U_ontology, U_pullback, U_hodge} de
        certificaciones locales: la sección global de "gobernanza válida"
        existe si y solo si las secciones locales son compatibles, es decir,
        si la obstrucción de Čech de grado 0 se anula.

        Cada abierto de la cubierta aporta una carga de obstrucción binaria
        (0 = sección local válida, 1 = obstrucción presente); la sección
        global existe sii Σ obstrucciones = 0.
        """
        contributions: List[Tuple[str, int]] = [
            (
                "U_lattice:Retículo-Distributivo",
                0 if lattice_audit.is_structurally_sound else 1,
            ),
            (
                "U_ontology:Cohomología-Ontológica",
                0 if cohomology_audit.is_ontologically_coherent else 1,
            ),
            (
                "U_hodge:Validación-Cruzada-Hodge",
                0 if cohomology_audit.is_hodge_cross_validated else 1,
            ),
            (
                "U_pullback:Pullback-ZeroTrust",
                0 if pullback_audit.is_zero_trust_verified else 1,
            ),
        ]

        obstruction_index = sum(charge for _name, charge in contributions)
        failing_charts = tuple(name for name, charge in contributions if charge != 0)

        return obstruction_index, failing_charts

    # ────────────────────────────────────────────────────────────────────
    # §3.5 — Síntesis terminal del objeto de gobernanza
    # ────────────────────────────────────────────────────────────────────

    def _phase3_terminal_synthesis(
        self,
        normalized_violations: Sequence[float],
        boundary_d0: NDArray[Any],
        boundary_d1: NDArray[Any],
        X_domain: NDArray[Any],
        S_allowed: NDArray[Any],
        Omega_projector: NDArray[Any],
        *,
        top_threshold: float = _TOP_LATTICE_THRESHOLD,
        lattice_tolerance: float = _LATTICE_TOLERANCE,
        svd_tolerance: float = _SVD_TOLERANCE,
        cochain_tolerance: float = _COCHAIN_TOLERANCE,
        require_cochain_complex: bool = True,
        hodge_tolerance: float = _HODGE_LAPLACIAN_TOLERANCE,
        require_hodge_consistency: bool = True,
        pullback_tolerance: float = _PULLBACK_TOLERANCE,
        require_projector: bool = True,
        require_hermitian_projector: bool = True,
        require_spectral_projector: bool = True,
        raise_on_veto: bool = True,
    ) -> FederatedGovernanceState:
        """
        Último método de Fase 3: síntesis del objeto final de gobernanza.

        Composición final:
            Z_Gov = Φ₃ ∘ Φ₂ ∘ Φ₁

        La decisión de cumplimiento se formaliza como axioma de pegado:
        is_fully_compliant ⇔ gluing_obstruction_index = 0.
        """
        (
            lattice_audit,
            cohomology_audit,
            pullback_audit,
        ) = self._phase2_terminal_bridge_to_phase3(
            normalized_violations,
            boundary_d0,
            boundary_d1,
            X_domain,
            S_allowed,
            Omega_projector,
            top_threshold=top_threshold,
            lattice_tolerance=lattice_tolerance,
            svd_tolerance=svd_tolerance,
            cochain_tolerance=cochain_tolerance,
            require_cochain_complex=require_cochain_complex,
            hodge_tolerance=hodge_tolerance,
            require_hodge_consistency=require_hodge_consistency,
            pullback_tolerance=pullback_tolerance,
            require_projector=require_projector,
            require_hermitian_projector=require_hermitian_projector,
            require_spectral_projector=require_spectral_projector,
            raise_on_veto=raise_on_veto,
        )

        gluing_obstruction_index, failing_charts = self._certify_sheaf_gluing_obstruction(
            lattice_audit,
            cohomology_audit,
            pullback_audit,
        )

        is_fully_compliant = bool(gluing_obstruction_index == 0)

        if not is_fully_compliant and raise_on_veto:
            detail = ", ".join(failing_charts)
            raise SheafGluingObstructionError(
                f"Gobernanza federada inválida: obstrucción de pegado = "
                f"{gluing_obstruction_index} ≠ 0. Abiertos de la cubierta en "
                f"falta de sección local válida: {detail}."
            )

        state = FederatedGovernanceState(
            governance_id=str(uuid.uuid4()),
            lattice_audit=lattice_audit,
            cohomology_audit=cohomology_audit,
            pullback_audit=pullback_audit,
            gluing_obstruction_index=gluing_obstruction_index,
            gluing_obstruction_failing_charts=failing_charts,
            is_fully_compliant=is_fully_compliant,
            generated_at_utc=_utc_timestamp(),
        )

        logger.info(
            "Gobernanza Computacional Federada auditada. "
            "id=%s | ⊔=%.6f | H=%.4f/%.4f bits | dim_H¹=%d (Hodge=%s) | "
            "χ=%d | pullback_residual=%.6e | tr(Ω)=%.4f | obstruction=%d | "
            "compliant=%s",
            state.governance_id,
            lattice_audit.supremum_state,
            lattice_audit.shannon_entropy_bits,
            lattice_audit.max_entropy_bits,
            cohomology_audit.dim_H1,
            cohomology_audit.dim_H1_hodge,
            cohomology_audit.euler_characteristic,
            pullback_audit.pullback_residual,
            pullback_audit.projector_trace,
            gluing_obstruction_index,
            state.is_fully_compliant,
        )

        return state


# ══════════════════════════════════════════════════════════════════════════════
# §D. ORQUESTADOR SUPREMO: GOVERNANCE AGENT
# ══════════════════════════════════════════════════════════════════════════════

class GovernanceAgent(Morphism, Phase3_SpectralToposPolicyFunctor):
    r"""
    El Custodio de la Gobernanza Computacional Federada.

    Sustituye la evaluación burocrática del ComplianceReport por una validación
    axiomática dentro de teoría de categorías, retículos distributivos duales,
    teoría de la información, cohomología de haces (con validación cruzada de
    Hodge) y Topos de Grothendieck (con certificación espectral de proyectores),
    sintetizada finalmente como un axioma de pegado de haces.
    """

    def execute_federated_governance(
        self,
        normalized_violations: Sequence[float],
        boundary_d0: NDArray[Any],
        boundary_d1: NDArray[Any],
        X_domain: NDArray[Any],
        S_allowed: NDArray[Any],
        Omega_projector: NDArray[Any],
        *,
        top_threshold: float = _TOP_LATTICE_THRESHOLD,
        lattice_tolerance: float = _LATTICE_TOLERANCE,
        svd_tolerance: float = _SVD_TOLERANCE,
        cochain_tolerance: float = _COCHAIN_TOLERANCE,
        require_cochain_complex: bool = True,
        hodge_tolerance: float = _HODGE_LAPLACIAN_TOLERANCE,
        require_hodge_consistency: bool = True,
        pullback_tolerance: float = _PULLBACK_TOLERANCE,
        require_projector: bool = True,
        require_hermitian_projector: bool = True,
        require_spectral_projector: bool = True,
        raise_on_veto: bool = True,
    ) -> FederatedGovernanceState:
        """
        Ejecuta la composición funtorial estricta completa.

        Retorna:
            FederatedGovernanceState con certificados de Fase 1, Fase 2 y
            Fase 3, más el índice de obstrucción de pegado de haces.

        Veto:
            Si raise_on_veto=True, cualquier violación lanza excepción.
            Si raise_on_veto=False, devuelve estado con is_fully_compliant=False
            y gluing_obstruction_index > 0.
        """
        return self._phase3_terminal_synthesis(
            normalized_violations,
            boundary_d0,
            boundary_d1,
            X_domain,
            S_allowed,
            Omega_projector,
            top_threshold=top_threshold,
            lattice_tolerance=lattice_tolerance,
            svd_tolerance=svd_tolerance,
            cochain_tolerance=cochain_tolerance,
            require_cochain_complex=require_cochain_complex,
            hodge_tolerance=hodge_tolerance,
            require_hodge_consistency=require_hodge_consistency,
            pullback_tolerance=pullback_tolerance,
            require_projector=require_projector,
            require_hermitian_projector=require_hermitian_projector,
            require_spectral_projector=require_spectral_projector,
            raise_on_veto=raise_on_veto,
        )


# ══════════════════════════════════════════════════════════════════════════════
# §E. EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "GovernanceAgentError",
    "LatticeInputError",
    "StructuralVetoMonad",
    "CohomologyInputError",
    "OntologicalParadoxVeto",
    "HodgeCrossValidationError",
    "PullbackInputError",
    "ZeroTrustViolationError",
    "SheafGluingObstructionError",
    "LatticeProjectionData",
    "CohomologicalOntologyData",
    "ToposPolicyPullbackData",
    "FederatedGovernanceState",
    "Phase1_InformationTheoreticLatticeProjector",
    "Phase2_EulerCharacteristicCohomologyAuditor",
    "Phase3_SpectralToposPolicyFunctor",
    "GovernanceAgent",
]