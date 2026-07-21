# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : MIC Minimizer Agent (Custodio de la Base Booleana)                  ║
║ Ruta   : app/agents/boole/tactics/mic_minimizer_agent.py                     ║
║ Versión: 2.0.0-Grobner-ROBDD-Categorical-Strict                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y ÁLGEBRA DE BOOLE (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor gobierna el `mic_minimizer.py` en el subespacio Γ-TACTICS.

Su mandato axiomático es garantizar que la poda topológica en el anillo booleano
Z_2 no destruya el rango efectivo de la Matriz de Interacción Central (MIC).

Erradica redundancias garantizando que la base resultante sea estrictamente
ortogonal:

    ⟨e_i, e_j⟩ = δ_ij.

ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Auditoría de Bases de Gröbner:
    Verifica que el ideal generado por las funciones booleanas de las herramientas

        I = ⟨f_1, ..., f_m⟩ ⊆ Z_2[x_1, ..., x_n]

    sea una base mínima y no colapse en homología trivial.

    Último método de Fase 1:
        _audit_grobner_independence(...)

    Dicho método retorna un certificado `GrobnerAuditData`, el cual se convierte
    en el objeto inicial de la Fase 2.

Fase 2 → Certificación de No-Interferencia (UNSAT Core):
    Audita la cláusula de ortogonalidad:

        Φ_MIC = ∧_{i ≠ j} ¬(e_i ∧ e_j).

    En forma matricial, exige:

        P Pᵀ ≈ I.

    Primer método de Fase 2:
        _certify_non_interference_unsat(..., grobner_audit)

    Este método es la continuación formal de Fase 1: recibe el certificado de
    independencia algebraica y lo propaga como invariante inicial.

Fase 3 → Isomorfismo de Reducción ROBDD:
    Garantiza que la minimización mediante ROBDD conserve la Entropía de Shannon
    booleana original:

        H(X_original) ≈ H(X_reduced).

    Primer método de Fase 3:
        _validate_robdd_homotopy(..., unsat_core_audit)

    Este método continúa formalmente la Fase 2: recibe el certificado de
    ortogonalidad y verifica que la reducción lógica preserve la entropía.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray


# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:

    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos E_MIC."""
        pass

    class Morphism:
        r"""Clase base de Morfismos del Topos."""
        pass


logger = logging.getLogger("MIC.Gamma.MinimizerAgent")


# ═══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES BOOLEANAS, ALGEBRAICAS Y DE COMPLEJIDAD
# ═══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)

_MAX_BOOLEAN_VARIABLES: Final[int] = 256
_MIN_ENTROPY_TOLERANCE: Final[float] = 1e-12
_ORTHOGONALITY_TOLERANCE: Final[float] = 1e-10
_PROBABILITY_TOLERANCE: Final[float] = 1e-12
_INTEGER_REPRESENTATION_TOLERANCE: Final[float] = 1e-12
_NUMERICAL_SAFETY_FACTOR: Final[float] = 128.0


# ═══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES ALGEBRAICAS
# ═══════════════════════════════════════════════════════════════════════════════
class MICMinimizerAgentError(TopologicalInvariantError):
    r"""Excepción raíz del Custodio de la Base Booleana."""
    pass


class BooleanInputValidationError(MICMinimizerAgentError):
    r"""Detonada si los datos booleanos, matriciales o probabilísticos son inválidos."""
    pass


class GrobnerDegeneracyError(MICMinimizerAgentError):
    r"""Detonada si el ideal booleano colapsa o pierde independencia efectiva."""
    pass


class NonInterferenceViolationError(MICMinimizerAgentError):
    r"""Detonada si ⟨e_i, e_j⟩ ≠ 0 para i ≠ j. Ruptura del Zero Side-Effects."""
    pass


class ROBDDHomotopyError(MICMinimizerAgentError):
    r"""Detonada si el ROBDD reducido no conserva la entropía booleana original."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Anillo Z_2)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class GrobnerAuditData:
    r"""
    Artefacto de Fase 1.
    Certificado de independencia algebraica en Z_2[X].

    Este objeto es el resultado final del último método de Fase 1 y el objeto
    inicial de Fase 2.
    """
    rows: int
    cols: int
    ideal_dimension: int
    nullity: int
    pivot_columns: Tuple[int, ...]
    is_minimally_independent: bool


@dataclass(frozen=True, slots=True)
class UnsatCoreCertifierData:
    r"""
    Artefacto de Fase 2.
    Certificado de ortogonalidad estricta y no-interferencia.

    Este objeto es el resultado final de Fase 2 y el objeto inicial de Fase 3.
    """
    tool_count: int
    variable_dim: int
    off_diagonal_conflict_norm: float
    diagonal_deviation_norm: float
    conflict_edges: int
    orthogonality_tolerance: float
    is_strictly_orthogonal: bool


@dataclass(frozen=True, slots=True)
class ROBDDIsomorphismData:
    r"""
    Artefacto de Fase 3.
    Certificado de conservación de entropía de Shannon booleana.
    """
    support_dimension: int
    original_entropy: float
    reduced_entropy: float
    entropy_loss: float
    entropy_tolerance: float
    total_variation_distance: float
    is_homotopically_equivalent: bool


@dataclass(frozen=True, slots=True)
class MinimizerGovernanceState:
    r"""
    Objeto final del endofuntor Z_Minimizer.
    """
    grobner_audit: GrobnerAuditData
    unsat_core_audit: UnsatCoreCertifierData
    robdd_audit: ROBDDIsomorphismData
    is_topologically_valid: bool


# ═══════════════════════════════════════════════════════════════════════════════
# §D. GUARDAS NUMÉRICAS INTERNAS
# ═══════════════════════════════════════════════════════════════════════════════
class _FiniteNumericalGuard:
    r"""
    Capa de saneamiento numérico para evitar que singularidades aritméticas
    contaminen los invariantes algebraicos y booleanos.
    """

    @staticmethod
    def _as_finite_real_matrix(name: str, value: Any) -> NDArray[np.float64]:
        r"""
        Valida una matriz real finita.
        """
        try:
            raw = np.asarray(value)
        except Exception as exc:
            raise BooleanInputValidationError(
                f"{name} no puede interpretarse como arreglo numérico."
            ) from exc

        if np.iscomplexobj(raw):
            raise BooleanInputValidationError(
                f"{name} debe ser real; se rechazó entrada compleja."
            )

        try:
            arr = raw.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise BooleanInputValidationError(
                f"{name} debe ser numérico real convertible a float64."
            ) from exc

        if not np.all(np.isfinite(arr)):
            raise BooleanInputValidationError(
                f"{name} contiene valores NaN o infinitos."
            )

        if arr.ndim != 2:
            raise BooleanInputValidationError(
                f"{name} debe ser una matriz 2D."
            )

        return arr

    @staticmethod
    def _as_finite_real_vector(name: str, value: Any) -> NDArray[np.float64]:
        r"""
        Valida un vector real finito.

        Acepta:
            - Vectores 1D.
            - Vectores columna (n, 1).
            - Vectores fila (1, n).
            - Escalares, interpretados como vector de dimensión 1.
        """
        try:
            raw = np.asarray(value)
        except Exception as exc:
            raise BooleanInputValidationError(
                f"{name} no puede interpretarse como arreglo numérico."
            ) from exc

        if np.iscomplexobj(raw):
            raise BooleanInputValidationError(
                f"{name} debe ser real; se rechazó entrada compleja."
            )

        try:
            arr = raw.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise BooleanInputValidationError(
                f"{name} debe ser numérico real convertible a float64."
            ) from exc

        if arr.ndim == 0:
            arr = arr.reshape(1)
        elif arr.ndim == 2 and 1 in arr.shape:
            arr = arr.reshape(-1)
        elif arr.ndim != 1:
            raise BooleanInputValidationError(
                f"{name} debe ser un vector 1D, fila, columna o escalar."
            )

        if not np.all(np.isfinite(arr)):
            raise BooleanInputValidationError(
                f"{name} contiene valores NaN o infinitos."
            )

        return arr

    @staticmethod
    def _as_finite_gf2_matrix(name: str, value: Any) -> NDArray[np.uint8]:
        r"""
        Valida una matriz booleana y la proyecta a GF(2).

        Acepta valores enteros o casi-enteros, y los reduce módulo 2.
        """
        try:
            raw = np.asarray(value)
        except Exception as exc:
            raise BooleanInputValidationError(
                f"{name} no puede interpretarse como arreglo numérico."
            ) from exc

        if np.iscomplexobj(raw):
            raise BooleanInputValidationError(
                f"{name} debe ser booleano/entero real; se rechazó entrada compleja."
            )

        try:
            arr_float = raw.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise BooleanInputValidationError(
                f"{name} debe ser convertible a valores enteros en GF(2)."
            ) from exc

        if not np.all(np.isfinite(arr_float)):
            raise BooleanInputValidationError(
                f"{name} contiene valores NaN o infinitos."
            )

        if arr_float.ndim != 2:
            raise BooleanInputValidationError(
                f"{name} debe ser una matriz 2D sobre GF(2)."
            )

        if arr_float.size > 0:
            max_abs = float(np.max(np.abs(arr_float)))
        else:
            max_abs = 0.0

        integer_tolerance = max(
            _INTEGER_REPRESENTATION_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(1.0, max_abs),
        )

        rounded = np.rint(arr_float)

        if not np.all(np.abs(arr_float - rounded) <= integer_tolerance):
            raise BooleanInputValidationError(
                f"{name} contiene valores no enteros; GF(2) requiere coeficientes 0 o 1."
            )

        arr_gf2 = np.mod(rounded.astype(np.int64), 2).astype(np.uint8)

        return arr_gf2

    @classmethod
    def _as_finite_probability_vector(
        cls,
        name: str,
        value: Any,
    ) -> NDArray[np.float64]:
        r"""
        Valida un vector de probabilidades.

        Exige:
            - Entradas reales finitas.
            - No negatividad dentro de tolerancia.
            - Masa total positiva.
            - Normalización a suma 1.

        Si la suma se desvía ligeramente de 1, se normaliza con advertencia.
        """
        arr = cls._as_finite_real_vector(name, value)

        if arr.size == 0:
            raise BooleanInputValidationError(
                f"{name} no puede ser un vector de probabilidades vacío."
            )

        min_value = float(np.min(arr))

        if min_value < -_PROBABILITY_TOLERANCE:
            raise BooleanInputValidationError(
                f"{name} contiene probabilidades negativas no físicas. "
                f"Mínimo = {min_value:.6e}."
            )

        arr = np.clip(arr, 0.0, None)
        total_mass = float(np.sum(arr))

        if not math.isfinite(total_mass) or total_mass <= _PROBABILITY_TOLERANCE:
            raise BooleanInputValidationError(
                f"{name} tiene masa probabilística nula o no finita."
            )

        normalization_tolerance = max(
            _PROBABILITY_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * max(1.0, total_mass),
        )

        if abs(total_mass - 1.0) > normalization_tolerance:
            logger.warning(
                "%s tiene masa total %.6e distinta de 1; se normaliza internamente.",
                name,
                total_mass,
            )

        arr = arr / total_mass

        return arr


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1: AUDITORÍA DE BASES DE GRÖBNER EN EL ANILLO Z_2                    ║
# ║                                                                             ║
# ║   Verifica la independencia lineal del ideal de herramientas sobre GF(2).   ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_GrobnerBasisAuditor(_FiniteNumericalGuard):
    r"""
    Garantiza que la minimización de herramientas no degenere la base operativa.

    La matriz booleana se interpreta como un conjunto de generadores en GF(2).
    Se audita que el rango efectivo sea igual al número de generadores, de modo
    que no existan redundancias algebraicas.
    """

    @staticmethod
    def _gf2_rank(matrix: NDArray[np.uint8]) -> Tuple[int, Tuple[int, ...]]:
        r"""
        Calcula el rango sobre GF(2) mediante eliminación de Gauss-Jordan.

        Retorna:
            rank:
                Rango efectivo sobre GF(2).

            pivot_columns:
                Columnas pivote que certifican la base algebraica.
        """
        M = matrix.copy()
        rows, cols = M.shape

        rank = 0
        pivot_columns: List[int] = []

        for col in range(cols):
            if rank >= rows:
                break

            pivot_candidates = np.nonzero(M[rank:, col])[0]

            if pivot_candidates.size == 0:
                continue

            pivot = int(pivot_candidates[0] + rank)

            if pivot != rank:
                M[[rank, pivot]] = M[[pivot, rank]]

            pivot_row = M[rank].copy()

            # Eliminación completa sobre GF(2): arriba y abajo del pivote.
            for i in range(rows):
                if i != rank and M[i, col] == 1:
                    M[i] = np.bitwise_xor(M[i], pivot_row)

            pivot_columns.append(col)
            rank += 1

        return rank, tuple(pivot_columns)

    def _audit_grobner_independence(
        self,
        boolean_polynomial_matrix: NDArray[np.uint8],
    ) -> GrobnerAuditData:
        r"""
        Último método de la Fase 1.

        Calcula el rango algebraico en Z_2 usando eliminación de Gauss-Jordan.

        Si el rango efectivo es menor que el número de generadores, se detecta
        degeneración del ideal booleano.

        Este método retorna un certificado `GrobnerAuditData`, el cual constituye
        el objeto inicial de la Fase 2.
        """
        matrix_gf2 = self._as_finite_gf2_matrix(
            "boolean_polynomial_matrix",
            boolean_polynomial_matrix,
        )

        rows, cols = matrix_gf2.shape

        if rows == 0 or cols == 0:
            raise BooleanInputValidationError(
                "boolean_polynomial_matrix no puede ser vacía."
            )

        if cols > _MAX_BOOLEAN_VARIABLES:
            raise MICMinimizerAgentError(
                "Explosión combinatoria detectada: "
                f"n_vars={cols} > límite seguro={_MAX_BOOLEAN_VARIABLES}."
            )

        rank, pivot_columns = self._gf2_rank(matrix_gf2)
        nullity = int(cols - rank)

        if rank < rows:
            raise GrobnerDegeneracyError(
                "Degeneración en el anillo booleano detectada. "
                f"El ideal colapsó: rango efectivo en Z_2 = {rank} < "
                f"número de generadores = {rows}. "
                "La poda algorítmica amputaría capacidades esenciales del agente."
            )

        return GrobnerAuditData(
            rows=int(rows),
            cols=int(cols),
            ideal_dimension=int(rank),
            nullity=nullity,
            pivot_columns=pivot_columns,
            is_minimally_independent=True,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2: CERTIFICACIÓN DE NO-INTERFERENCIA (UNSAT CORE)                    ║
# ║                                                                             ║
# ║   Audita:                                                                   ║
# ║       P Pᵀ ≈ I                                                              ║
# ║                                                                             ║
# ║   Esta fase comienza consumiendo el certificado de Fase 1.                  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_UnsatCoreCertifier(Phase1_GrobnerBasisAuditor):
    r"""
    Evalúa que las herramientas sugeridas sean estrictamente ortogonales para
    prevenir el colapso del principio Zero Side-Effects en la MIC.

    La condición de no-interferencia es:

        ⟨e_i, e_j⟩ = δ_ij.

    En forma matricial:

        P Pᵀ = I.

    Esta fase hereda de Fase 1 y su primer método recibe explícitamente el
    certificado algebraico emitido por:

        Phase1_GrobnerBasisAuditor._audit_grobner_independence(...)

    De este modo, la Fase 2 no es autónoma: está anidada funcionalmente en la
    Fase 1.
    """

    def _certify_non_interference_unsat(
        self,
        tool_projection_matrix: NDArray[np.float64],
        grobner_audit: Optional[GrobnerAuditData] = None,
    ) -> UnsatCoreCertifierData:
        r"""
        Primer método de la Fase 2.

        Continuación formal del último método de Fase 1.

        Computa la matriz de Gram:

            G = P Pᵀ,

        y exige:
            - G_ij ≈ 0 para i ≠ j.
            - G_ii ≈ 1.

        Si `grobner_audit` es provisto:
            - Verifica que la Fase 1 haya certificado independencia algebraica.
            - Exige consistencia dimensional con el espacio booleano certificado.

        Retorna:
            UnsatCoreCertifierData, certificado que sirve como objeto inicial de
            la Fase 3.
        """
        if grobner_audit is not None:
            if not grobner_audit.is_minimally_independent:
                raise GrobnerDegeneracyError(
                    "La Fase 2 no puede iniciarse: la Fase 1 no certificó "
                    "independencia algebraica en GF(2)."
                )

        P = self._as_finite_real_matrix(
            "tool_projection_matrix",
            tool_projection_matrix,
        )

        if P.size == 0:
            raise BooleanInputValidationError(
                "tool_projection_matrix no puede ser vacía."
            )

        tool_count, variable_dim = P.shape

        if tool_count == 0 or variable_dim == 0:
            raise BooleanInputValidationError(
                "tool_projection_matrix debe tener herramientas y variables."
            )

        if grobner_audit is not None:
            if grobner_audit.cols != variable_dim:
                raise ValueError(
                    "Inconsistencia dimensional entre Fase 1 y Fase 2. "
                    f"Fase 1 certificó cols={grobner_audit.cols}, pero "
                    f"tool_projection_matrix tiene variable_dim={variable_dim}."
                )

            if grobner_audit.rows != tool_count:
                logger.warning(
                    "Fase 1 certificó %d generadores, pero Fase 2 recibió %d herramientas.",
                    grobner_audit.rows,
                    tool_count,
                )

        try:
            gram = P @ P.T
        except Exception as exc:
            raise NonInterferenceViolationError(
                "No fue posible computar la matriz de Gram P Pᵀ."
            ) from exc

        if not np.all(np.isfinite(gram)):
            raise NonInterferenceViolationError(
                "La matriz de Gram P Pᵀ contiene valores NaN o infinitos."
            )

        diagonal = np.diag(gram).copy()

        if diagonal.size == 0:
            raise BooleanInputValidationError(
                "La matriz de Gram no posee diagonal; herramienta degenerada."
            )

        diagonal_deviation = float(np.max(np.abs(diagonal - 1.0)))

        if not math.isfinite(diagonal_deviation):
            raise NonInterferenceViolationError(
                "La desviación diagonal de P Pᵀ no es finita."
            )

        off_diagonal = gram.copy()
        np.fill_diagonal(off_diagonal, 0.0)

        if tool_count > 1:
            upper_indices = np.triu_indices(tool_count, k=1)
            off_diagonal_values = np.abs(off_diagonal[upper_indices])
            off_diagonal_conflict_norm = float(np.sum(off_diagonal_values))
        else:
            off_diagonal_values = np.empty(0, dtype=np.float64)
            off_diagonal_conflict_norm = 0.0

        if not math.isfinite(off_diagonal_conflict_norm):
            raise NonInterferenceViolationError(
                "La norma de interferencia cruzada no es finita."
            )

        diagonal_scale = float(np.sum(np.abs(diagonal)))

        if not math.isfinite(diagonal_scale):
            raise NonInterferenceViolationError(
                "La escala diagonal de la matriz de Gram no es finita."
            )

        scale = max(1.0, diagonal_scale)

        orthogonality_tolerance = max(
            _ORTHOGONALITY_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON * scale,
        )

        conflict_edges = int(
            np.count_nonzero(off_diagonal_values > orthogonality_tolerance)
        )

        if off_diagonal_conflict_norm > orthogonality_tolerance:
            raise NonInterferenceViolationError(
                "Violación del axioma Zero Side-Effects. "
                "La matriz de capacidades no es ortogonal. "
                f"Norma residual fuera de diagonal = {off_diagonal_conflict_norm:.6e} > "
                f"tolerancia = {orthogonality_tolerance:.6e}."
            )

        if diagonal_deviation > orthogonality_tolerance:
            raise NonInterferenceViolationError(
                "Violación de ortonormalidad. "
                f"La diagonal de P Pᵀ se desvía de I en {diagonal_deviation:.6e} > "
                f"tolerancia = {orthogonality_tolerance:.6e}."
            )

        if conflict_edges > 0:
            raise NonInterferenceViolationError(
                "UNSAT Core detectó aristas de interferencia cruzada. "
                f"Número de pares no ortogonales = {conflict_edges}."
            )

        return UnsatCoreCertifierData(
            tool_count=int(tool_count),
            variable_dim=int(variable_dim),
            off_diagonal_conflict_norm=float(off_diagonal_conflict_norm),
            diagonal_deviation_norm=float(diagonal_deviation),
            conflict_edges=conflict_edges,
            orthogonality_tolerance=float(orthogonality_tolerance),
            is_strictly_orthogonal=True,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3: ISOMORFISMO DE REDUCCIÓN ROBDD                                    ║
# ║                                                                             ║
# ║   Garantiza:                                                                ║
# ║       H(X_original) ≈ H(X_reduced)                                          ║
# ║                                                                             ║
# ║   Esta fase comienza consumiendo el certificado de Fase 2.                  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_ROBDDIsomorphismValidator(Phase2_UnsatCoreCertifier):
    r"""
    Asegura que el Diagrama de Decisión Binaria Reducido (ROBDD) mantenga
    la equivalencia homotópica con el árbol de sintaxis original.

    La condición de preservación es:

        |H_original - H_reduced| ≤ ε_H.

    Esta fase hereda de Fase 2 y su primer método recibe explícitamente el
    certificado de no-interferencia emitido por:

        Phase2_UnsatCoreCertifier._certify_non_interference_unsat(...)

    De este modo, la Fase 3 está anidada funcionalmente en la Fase 2.
    """

    @staticmethod
    def _shannon_entropy_bits(probabilities: NDArray[np.float64]) -> float:
        r"""
        Calcula la entropía de Shannon en bits:

            H(X) = -Σ p(x_i) log₂ p(x_i).
        """
        p = np.clip(probabilities, 0.0, 1.0)
        p = p[p > 0.0]

        if p.size == 0:
            return 0.0

        entropy = float(-np.sum(p * np.log2(p)))

        if not math.isfinite(entropy):
            raise ROBDDHomotopyError(
                "La entropía de Shannon no es finita."
            )

        return entropy

    def _validate_robdd_homotopy(
        self,
        original_truth_table_probs: NDArray[np.float64],
        reduced_robdd_probs: NDArray[np.float64],
        unsat_core_audit: Optional[UnsatCoreCertifierData] = None,
    ) -> ROBDDIsomorphismData:
        r"""
        Primer método de la Fase 3.

        Continuación formal de Fase 2.

        Compara la entropía de Shannon de la distribución booleana original y
        la distribución reducida por ROBDD.

        Si `unsat_core_audit` es provisto:
            - Verifica que la Fase 2 haya certificado ortogonalidad estricta.

        Retorna:
            ROBDDIsomorphismData, certificado final de equivalencia homotópica.
        """
        if unsat_core_audit is not None:
            if not unsat_core_audit.is_strictly_orthogonal:
                raise NonInterferenceViolationError(
                    "La Fase 3 no puede iniciarse: la Fase 2 no certificó "
                    "ortogonalidad estricta."
                )

        p_original = self._as_finite_probability_vector(
            "original_truth_table_probs",
            original_truth_table_probs,
        )

        p_reduced = self._as_finite_probability_vector(
            "reduced_robdd_probs",
            reduced_robdd_probs,
        )

        support_dimension = max(p_original.size, p_reduced.size)

        p_original_pad = np.pad(
            p_original,
            (0, support_dimension - p_original.size),
        )

        p_reduced_pad = np.pad(
            p_reduced,
            (0, support_dimension - p_reduced.size),
        )

        original_mass = float(np.sum(p_original_pad))
        reduced_mass = float(np.sum(p_reduced_pad))

        if (
            not math.isfinite(original_mass)
            or not math.isfinite(reduced_mass)
            or original_mass <= _PROBABILITY_TOLERANCE
            or reduced_mass <= _PROBABILITY_TOLERANCE
        ):
            raise BooleanInputValidationError(
                "Las distribuciones probabilísticas ROBDD tienen masa inválida."
            )

        p_original_pad = p_original_pad / original_mass
        p_reduced_pad = p_reduced_pad / reduced_mass

        H_original = self._shannon_entropy_bits(p_original_pad)
        H_reduced = self._shannon_entropy_bits(p_reduced_pad)

        entropy_loss = float(abs(H_original - H_reduced))

        if not math.isfinite(entropy_loss):
            raise ROBDDHomotopyError(
                "La pérdida entrópica ΔH no es finita."
            )

        entropy_tolerance = max(
            _MIN_ENTROPY_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, abs(H_original), abs(H_reduced)),
        )

        total_variation_distance = float(
            0.5 * np.sum(np.abs(p_original_pad - p_reduced_pad))
        )

        if not math.isfinite(total_variation_distance):
            raise ROBDDHomotopyError(
                "La distancia de variación total no es finita."
            )

        if entropy_loss > entropy_tolerance:
            raise ROBDDHomotopyError(
                "Ruptura homotópica en la reducción ROBDD. "
                f"La entropía booleana divergió: ΔH = {entropy_loss:.6e} bits > "
                f"tolerancia = {entropy_tolerance:.6e}. "
                "El minimizador mutiló ramas lógicas operativas de la MIC."
            )

        return ROBDDIsomorphismData(
            support_dimension=int(support_dimension),
            original_entropy=float(H_original),
            reduced_entropy=float(H_reduced),
            entropy_loss=float(entropy_loss),
            entropy_tolerance=float(entropy_tolerance),
            total_variation_distance=float(total_variation_distance),
            is_homotopically_equivalent=True,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: MIC MINIMIZER AGENT                                  ║
# ║                                                                             ║
# ║   Endofuntor Z_Minimizer = Φ₃ ∘ Φ₂ ∘ Φ₁                                    ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class MICMinimizerAgent(Morphism, Phase3_ROBDDIsomorphismValidator):
    r"""
    El Custodio de la Base Booleana.

    Gobierna incondicionalmente el módulo `mic_minimizer.py`, impidiendo que
    algoritmos heurísticos de minimización degraden el rango efectivo de la MIC
    en el estrato Γ-TACTICS.
    """

    def execute_boolean_topology_governance(
        self,
        boolean_polynomial_matrix: NDArray[np.uint8],
        tool_projection_matrix: NDArray[np.float64],
        original_truth_table_probs: NDArray[np.float64],
        reduced_robdd_probs: NDArray[np.float64],
    ) -> MinimizerGovernanceState:
        r"""
        Ejecuta la composición funtorial estricta:

            Φ₁ : Auditoría de independencia Gröbner sobre GF(2).
            Φ₂ : Certificación de no-interferencia y ortogonalidad.
            Φ₃ : Validación de isomorfismo entrópico ROBDD.

        Parámetros:
            boolean_polynomial_matrix:
                Matriz de coeficientes booleanos sobre GF(2).

            tool_projection_matrix:
                Matriz de proyección de herramientas P.

            original_truth_table_probs:
                Distribución de probabilidad original de la tabla de verdad.

            reduced_robdd_probs:
                Distribución de probabilidad reducida por ROBDD.

        Retorna:
            MinimizerGovernanceState con los tres certificados y validez
            topológica final.
        """
        # Fase 1: Certificar independencia en la base de Gröbner sobre GF(2).
        grobner_audit = self._audit_grobner_independence(
            boolean_polynomial_matrix
        )

        # Fase 2: Certificar principio Zero Side-Effects (ortogonalidad).
        unsat_core_audit = self._certify_non_interference_unsat(
            tool_projection_matrix,
            grobner_audit=grobner_audit,
        )

        # Fase 3: Certificar isomorfismo entrópico en la reducción ROBDD.
        robdd_audit = self._validate_robdd_homotopy(
            original_truth_table_probs,
            reduced_robdd_probs,
            unsat_core_audit=unsat_core_audit,
        )

        is_topologically_valid = bool(
            grobner_audit.is_minimally_independent
            and unsat_core_audit.is_strictly_orthogonal
            and robdd_audit.is_homotopically_equivalent
        )

        if not is_topologically_valid:
            raise MICMinimizerAgentError(
                "La composición funtorial no autorizó la minimización booleana."
            )

        logger.info(
            "Gobernanza de la base booleana certificada. "
            "Rango GF(2): %d | "
            "Ortogonalidad preservada | "
            "Entropía H(X): %.6f bits | "
            "ΔH: %.6e bits",
            grobner_audit.ideal_dimension,
            robdd_audit.original_entropy,
            robdd_audit.entropy_loss,
        )

        return MinimizerGovernanceState(
            grobner_audit=grobner_audit,
            unsat_core_audit=unsat_core_audit,
            robdd_audit=robdd_audit,
            is_topologically_valid=is_topologically_valid,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ═══════════════════════════════════════════════════════════════════════════════
__all__: List[str] = [
    "MICMinimizerAgentError",
    "BooleanInputValidationError",
    "GrobnerDegeneracyError",
    "NonInterferenceViolationError",
    "ROBDDHomotopyError",
    "GrobnerAuditData",
    "UnsatCoreCertifierData",
    "ROBDDIsomorphismData",
    "MinimizerGovernanceState",
    "Phase1_GrobnerBasisAuditor",
    "Phase2_UnsatCoreCertifier",
    "Phase3_ROBDDIsomorphismValidator",
    "MICMinimizerAgent",
]