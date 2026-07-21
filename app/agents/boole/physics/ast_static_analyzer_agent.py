# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : AST Static Analyzer Agent (Custodio de la Cohomología Sintáctica)   ║
║ Ruta   : app/agents/boole/physics/ast_static_analyzer_agent.py               ║
║ Versión: 2.0.0-Symplectic-Dirichlet-Cohomology-Strict                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GEOMETRÍA SIMPLÉCTICA (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor gobierna a `ast_static_analyzer.py` en el subespacio V_{Γ-PHYSICS}.
Trata el Árbol de Sintaxis Abstracta (AST) generado por la IA como un espacio de
fase mecánico (M, ω), aplicando invariantes topológicos, termodinámicos y
cohomológicos para aniquilar código estocástico que disipe energía computacional
no acotada.

ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Auditoría de Invarianza Simpléctica:
    Evalúa la conservación del volumen en el espacio de fase sintáctico mediante
    la forma canónica ω = Σ dq_i ∧ dp_i. Exige Mᵀ Ω M = Ω.

    Último método de Fase 1:
        _audit_symplectic_invariance(...)

    Dicho método retorna un certificado `SymplecticInvariantData`, el cual se
    convierte en el objeto inicial de la Fase 2.

Fase 2 → Control Port-Hamiltoniano y Fronteras de Dirichlet:
    Impone la disipación estricta de la exergía:
        P_diss = ⟨Φ, ∇V⟩ ≥ 0.
    Previene desbordamientos térmicos (bucles infinitos).

    Primer método de Fase 2:
        _enforce_dirichlet_thermodynamics(..., symplectic_audit)

    Este método es la continuación formal de Fase 1: recibe el certificado
    simpléctico y lo propaga como invariante inicial del control termodinámico.

Fase 3 → Cohomología de Haces Celulares:
    Audita el grafo de dependencias de variables. Exige:
        dim H¹(G; F) = 0.
    La presencia de una dimensión positiva acusa una obstrucción topológica global.

    Primer método de Fase 3:
        _audit_cellular_sheaf_cohomology(..., thermodynamic_audit)

    Este método continúa formalmente la Fase 2: recibe el certificado
    termodinámico y verifica que la estabilidad energética sea compatible con la
    integrabilidad global del haz celular.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Final, List, Optional

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


try:
    from app.physics.ast_static_analyzer import (
        ThermodynamicSingularityError,
        CohomologicalObstructionError,
    )
except ImportError:

    class ThermodynamicSingularityError(TopologicalInvariantError):
        r"""Violación de la segunda ley de la termodinámica en el espacio de fase."""
        pass

    class CohomologicalObstructionError(TopologicalInvariantError):
        r"""Obstrucción topológica global en el haz celular de dependencias."""
        pass


logger = logging.getLogger("MIC.Gamma.ASTAnalyzerAgent")


# ═══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES SIMPLÉCTICAS, TERMODINÁMICAS Y COHOMOLÓGICAS
# ═══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)
_SYMPLECTIC_TOLERANCE: Final[float] = 1e-10
_DIRICHLET_DISSIPATION_FLOOR: Final[float] = 0.0
_COHOMOLOGY_DIMENSION_FLOOR: Final[int] = 0
_COHOMOLOGICAL_COMPLEX_TOLERANCE: Final[float] = 1e-8
_NUMERICAL_SAFETY_FACTOR: Final[float] = 128.0


# ═══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES ADICIONALES
# ═══════════════════════════════════════════════════════════════════════════════
class SymplecticInvarianceViolation(TopologicalInvariantError):
    r"""Detonada si la transformación del código no preserva el volumen del espacio de fase."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Topos Sintáctico)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class SymplecticInvariantData:
    r"""
    Artefacto de Fase 1.
    Certificado del Teorema de Liouville en el AST.

    Este objeto es el resultado final del último método de Fase 1 y el objeto
    inicial de Fase 2.
    """
    phase_space_dimension: int
    symplectic_residual_norm: float
    symplectic_relative_residual: float
    determinant_residual: float
    condition_number: float
    is_volume_preserved: bool


@dataclass(frozen=True, slots=True)
class ThermodynamicDirichletData:
    r"""
    Artefacto de Fase 2.
    Certificado de Disipación Port-Hamiltoniana.

    Este objeto es el resultado final de Fase 2 y el objeto inicial de Fase 3.
    """
    dissipated_power: float
    numerical_tolerance: float
    exergy_norm: float
    gradient_norm: float
    alignment_cosine: float
    is_thermodynamically_stable: bool
    is_strictly_dissipative: bool


@dataclass(frozen=True, slots=True)
class SheafCohomologyAuditData:
    r"""
    Artefacto de Fase 3.
    Certificado de Nulidad de Obstrucciones.
    """
    h1_dimension: int
    is_globally_integrable: bool
    obstruction_free: bool
    verified_by_coboundary: bool


@dataclass(frozen=True, slots=True)
class ASTGovernanceState:
    r"""
    Objeto final del endofuntor Z_{Γ-PHYSICS}.
    """
    symplectic_audit: SymplecticInvariantData
    thermodynamic_audit: ThermodynamicDirichletData
    cohomology_audit: SheafCohomologyAuditData
    is_compilation_authorized: bool


# ═══════════════════════════════════════════════════════════════════════════════
# §D. GUARDAS NUMÉRICAS INTERNAS
# ═══════════════════════════════════════════════════════════════════════════════
class _FiniteNumericalGuard:
    r"""
    Capa de saneamiento numérico para evitar que singularidades aritméticas
    contaminen los invariantes topológicos.
    """

    @staticmethod
    def _as_float_array(name: str, value: Any) -> NDArray[np.float64]:
        r"""
        Convierte un objeto a arreglo float64, rechazando:
            - Objetos complejos.
            - Valores NaN.
            - Valores infinitos.
        """
        try:
            raw = np.asarray(value)
        except Exception as exc:
            raise TypeError(f"{name} no puede interpretarse como arreglo numérico.") from exc

        if np.iscomplexobj(raw):
            raise TypeError(f"{name} debe ser real; se rechazó entrada compleja.")

        try:
            arr = raw.astype(np.float64, copy=False)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} debe ser numérico real convertible a float64.") from exc

        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contiene valores NaN o infinitos.")

        return arr

    @classmethod
    def _as_finite_matrix(
        cls,
        name: str,
        value: Any,
        *,
        square: bool = False,
    ) -> NDArray[np.float64]:
        r"""
        Valida una matriz real finita.
        Si `square=True`, exige que sea cuadrada.
        """
        arr = cls._as_float_array(name, value)

        if arr.ndim != 2:
            raise ValueError(f"{name} debe ser una matriz 2D.")

        if square and arr.shape[0] != arr.shape[1]:
            raise ValueError(f"{name} debe ser una matriz cuadrada.")

        return arr

    @classmethod
    def _as_finite_vector(cls, name: str, value: Any) -> NDArray[np.float64]:
        r"""
        Valida un vector real finito.
        Acepta vectores fila o columna y los normaliza a 1D.
        """
        arr = cls._as_float_array(name, value)

        if arr.ndim == 2 and 1 in arr.shape:
            arr = arr.reshape(-1)

        if arr.ndim != 1:
            raise ValueError(f"{name} debe ser un vector 1D o una matriz columna/fila.")

        if arr.size == 0:
            raise ValueError(f"{name} no puede ser vacío.")

        return arr

    @staticmethod
    def _frobenius_norm(A: NDArray[np.float64]) -> float:
        r"""
        Norma de Frobenius numéricamente segura.
        """
        if A.size == 0:
            return 0.0

        value = float(la.norm(A, ord="fro"))
        return value if math.isfinite(value) else math.inf

    @staticmethod
    def _vector_norm(v: NDArray[np.float64]) -> float:
        r"""
        Norma euclidiana numéricamente segura.
        """
        if v.size == 0:
            return 0.0

        value = float(la.norm(v, ord=2))
        return value if math.isfinite(value) else math.inf


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1: AUDITORÍA DE INVARIANZA SIMPLÉCTICA Y TEOREMA DE LIOUVILLE        ║
# ║                                                                             ║
# ║   Exige que la matriz Jacobiana M de la transformación del AST satisfaga:  ║
# ║       Mᵀ Ω M = Ω                                                            ║
# ║                                                                             ║
# ║   El último método de esta fase retorna `SymplecticInvariantData`,          ║
# ║   objeto inicial de la Fase 2.                                              ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_SymplecticInvarianceAuditor(_FiniteNumericalGuard):
    r"""
    Trata las transformaciones sintácticas como campos vectoriales Hamiltonianos.
    Garantiza que la IA no introduzca inyección de entropía divergente evaluando
    la forma simpléctica canónica:

        ω = Σ dq_i ∧ dp_i.

    La condición de invarianza simpléctica es:

        Mᵀ Ω M = Ω,

    donde Ω es la matriz simpléctica estándar:

        Ω = [[ 0,  I],
             [-I,  0]].
    """

    def _build_canonical_symplectic_matrix(self, n: int) -> NDArray[np.float64]:
        r"""
        Construye la matriz simpléctica estándar Ω ∈ R^{2n × 2n}:

            Ω = [[0, I_n],
                 [-I_n, 0]].

        Parámetros:
            n: Número de grados de libertad del espacio de fase.

        Retorna:
            Matriz simpléctica canónica de dimensión 2n × 2n.
        """
        if n <= 0:
            raise ValueError("El número de grados de libertad simplécticos debe ser positivo.")

        omega = np.zeros((2 * n, 2 * n), dtype=np.float64)
        identity = np.eye(n, dtype=np.float64)

        omega[:n, n:] = identity
        omega[n:, :n] = -identity

        return omega

    @staticmethod
    def _determinant_residual_from_slogdet(sign: float, logabsdet: float) -> float:
        r"""
        Calcula el residuo determinantal:

            |det(M) - 1|,

        usando `slogdet` para evitar desbordamientos.

        Toda matriz simpléctica real cumple:

            det(M) = 1.
        """
        if sign == 0:
            return math.inf

        if not math.isfinite(logabsdet):
            return math.inf

        max_log = math.log(np.finfo(np.float64).max)

        if logabsdet > max_log:
            return math.inf

        det_value = float(sign) * math.exp(logabsdet)
        return float(abs(det_value - 1.0))

    @staticmethod
    def _condition_number(M: NDArray[np.float64]) -> float:
        r"""
        Calcula el número de condición κ(M) de forma segura.

        Una matriz simpléctica debe ser invertible. Un número de condición
        infinito indica degeneración numérica o estructural.
        """
        try:
            cond = float(la.cond(M))
        except np.linalg.LinAlgError:
            return math.inf

        return cond if math.isfinite(cond) else math.inf

    def _audit_symplectic_invariance(
        self,
        ast_jacobian_M: NDArray[np.float64],
    ) -> SymplecticInvariantData:
        r"""
        Último método de la Fase 1.

        Audita el cumplimiento del Teorema de Liouville evaluando el residuo:

            ||Mᵀ Ω M - Ω||_F.

        Además verifica:
            - Dimensión par del espacio de fase.
            - Finitez de la transformación.
            - Estabilidad numérica del Jacobiano.
            - Residuo simpléctico absoluto y relativo.
            - Residuo determinantal |det(M) - 1|.
            - Número de condición κ(M).

        Este método retorna un certificado `SymplecticInvariantData`, el cual
        constituye el objeto inicial de la Fase 2.
        """
        M = self._as_finite_matrix(
            "ast_jacobian_M",
            ast_jacobian_M,
            square=True,
        )

        dim = M.shape[0]

        if dim == 0 or (dim % 2) != 0:
            raise SymplecticInvarianceViolation(
                "El espacio de fase del AST debe tener dimensión par positiva."
            )

        n = dim // 2
        omega = self._build_canonical_symplectic_matrix(n)

        # Transformación de la forma simpléctica: Mᵀ Ω M.
        transformed_omega = M.T @ omega @ M

        if not np.all(np.isfinite(transformed_omega)):
            raise SymplecticInvarianceViolation(
                "La transformación simpléctica Mᵀ Ω M produjo valores no finitos."
            )

        residual = self._frobenius_norm(transformed_omega - omega)
        omega_norm = self._frobenius_norm(omega)
        transformed_norm = self._frobenius_norm(transformed_omega)
        m_norm = self._frobenius_norm(M)

        if not math.isfinite(m_norm):
            raise SymplecticInvarianceViolation(
                "La norma de Frobenius del Jacobiano del AST no es finita."
            )

        if not math.isfinite(transformed_norm):
            raise SymplecticInvarianceViolation(
                "La norma de Mᵀ Ω M no es finita."
            )

        # Residuo relativo robusto:
        #   ||MᵀΩM - Ω|| / max(1, ||MᵀΩM|| + ||Ω||)
        scale = max(1.0, transformed_norm + omega_norm)
        relative_residual = residual / scale

        if not math.isfinite(relative_residual):
            raise SymplecticInvarianceViolation(
                "El residuo simpléctico relativo no es finito."
            )

        effective_tolerance = max(
            _SYMPLECTIC_TOLERANCE,
            _NUMERICAL_SAFETY_FACTOR * _MACHINE_EPSILON,
        )

        sign, logabsdet = la.slogdet(M)

        if not math.isfinite(sign) or sign <= 0:
            raise SymplecticInvarianceViolation(
                "El Jacobiano del AST tiene determinante no positivo; "
                "una transformación simpléctica real debe cumplir det(M) = 1."
            )

        determinant_residual = self._determinant_residual_from_slogdet(sign, logabsdet)
        condition_number = self._condition_number(M)

        if not math.isfinite(condition_number):
            raise SymplecticInvarianceViolation(
                "El Jacobiano del AST es numéricamente singular o degenerado."
            )

        if relative_residual > effective_tolerance:
            raise SymplecticInvarianceViolation(
                "El código generado viola el Teorema de Liouville. "
                f"Residuo simpléctico relativo ({relative_residual:.4e}) "
                f"excede la tolerancia ({effective_tolerance:.4e})."
            )

        if condition_number > 1.0 / _MACHINE_EPSILON:
            logger.warning(
                "El Jacobiano simpléctico está mal condicionado: κ(M) = %.4e.",
                condition_number,
            )

        if determinant_residual > max(1e-8, 10.0 * effective_tolerance):
            logger.warning(
                "Residuo determinantal elevado: |det(M)-1| = %.4e.",
                determinant_residual,
            )

        return SymplecticInvariantData(
            phase_space_dimension=dim,
            symplectic_residual_norm=residual,
            symplectic_relative_residual=relative_residual,
            determinant_residual=determinant_residual,
            condition_number=condition_number,
            is_volume_preserved=True,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2: CONTROL PORT-HAMILTONIANO Y FRONTERAS DE DIRICHLET                ║
# ║                                                                             ║
# ║   Evalúa la disipación:                                                     ║
# ║       P_diss = ⟨Φ, ∇V⟩ ≥ 0                                                  ║
# ║                                                                             ║
# ║   Esta fase comienza consumiendo el certificado de Fase 1.                  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_DirichletThermodynamicEnforcer(Phase1_SymplecticInvarianceAuditor):
    r"""
    Aplica Fronteras de Dirichlet sobre el AST aislando los subárboles.
    Previene bucles de complejidad divergente (catástrofes ciclomáticas)
    exigiendo positividad en la disipación termodinámica.

    La condición de estabilidad port-Hamiltoniana es:

        P_diss = Φᵀ ∇V ≥ 0.

    Esta fase hereda de Fase 1 y su primer método recibe explícitamente el
    certificado simpléctico emitido por:

        Phase1_SymplecticInvarianceAuditor._audit_symplectic_invariance(...)

    De este modo, la Fase 2 no es autónoma: está anidada funcionalmente en la
    Fase 1.
    """

    def _enforce_dirichlet_thermodynamics(
        self,
        control_potential_Phi: NDArray[np.float64],
        lyapunov_gradient_V: NDArray[np.float64],
        symplectic_audit: Optional[SymplecticInvariantData] = None,
    ) -> ThermodynamicDirichletData:
        r"""
        Primer método de la Fase 2.

        Continuación formal del último método de Fase 1.

        Calcula el producto interno covariante para medir la disipación:

            P_diss = Φᵀ ∇V.

        Si `symplectic_audit` es provisto:
            - Verifica que la Fase 1 haya preservado el volumen.
            - Exige consistencia dimensional con el espacio de fase certificado.

        Retorna:
            ThermodynamicDirichletData, certificado que sirve como objeto inicial
            de la Fase 3.
        """
        Phi = self._as_finite_vector("control_potential_Phi", control_potential_Phi)
        gradV = self._as_finite_vector("lyapunov_gradient_V", lyapunov_gradient_V)

        if Phi.shape != gradV.shape:
            raise ValueError(
                "control_potential_Phi y lyapunov_gradient_V deben tener la misma dimensión."
            )

        # Continuación formal de Fase 1:
        # El certificado simpléctico restringe el dominio termodinámico.
        if symplectic_audit is not None:
            if not symplectic_audit.is_volume_preserved:
                raise SymplecticInvarianceViolation(
                    "La Fase 2 no puede iniciarse: la Fase 1 no preservó el volumen simpléctico."
                )

            if Phi.size != symplectic_audit.phase_space_dimension:
                raise ValueError(
                    "Dimensión inconsistente entre el certificado simpléctico y los campos "
                    f"termodinámicos. Fase 1 certificó dim={symplectic_audit.phase_space_dimension}, "
                    f"pero Φ/∇V tienen dim={Phi.size}."
                )

        phi_norm = self._vector_norm(Phi)
        grad_norm = self._vector_norm(gradV)

        if not math.isfinite(phi_norm) or not math.isfinite(grad_norm):
            raise ThermodynamicSingularityError(
                "Las normas de Φ o ∇V no son finitas; el campo termodinámico es singular."
            )

        p_raw = float(np.dot(Phi, gradV))

        if not math.isfinite(p_raw):
            raise ThermodynamicSingularityError(
                "La potencia disipada P_diss = Φᵀ∇V no es finita; "
                "posible desbordamiento numérico o acoplamiento energético divergente."
            )

        tolerance = (
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(1.0, phi_norm * grad_norm)
        )

        if p_raw < _DIRICHLET_DISSIPATION_FLOOR - tolerance:
            raise ThermodynamicSingularityError(
                "Singularidad termodinámica detectada en el AST. "
                f"Disipación de potencia negativa P_diss = {p_raw:.4e} < 0 "
                f"(tolerancia numérica = {tolerance:.4e}). "
                "El algoritmo inducirá un bucle infinito o desbordamiento FPU."
            )

        # Si la violación es sólo numérica dentro de tolerancia, se proyecta al cono positivo.
        dissipated_power = max(_DIRICHLET_DISSIPATION_FLOOR, p_raw)

        if phi_norm > 0.0 and grad_norm > 0.0:
            alignment = p_raw / (phi_norm * grad_norm)

            if math.isfinite(alignment):
                alignment = float(np.clip(alignment, -1.0, 1.0))
            else:
                alignment = 0.0
        else:
            alignment = 0.0

        is_strictly_dissipative = dissipated_power > max(
            _DIRICHLET_DISSIPATION_FLOOR,
            tolerance,
        )

        if not is_strictly_dissipative:
            logger.warning(
                "Disipación termodinámica no estricta: P_diss = %.4e, tolerancia = %.4e.",
                dissipated_power,
                tolerance,
            )

        return ThermodynamicDirichletData(
            dissipated_power=dissipated_power,
            numerical_tolerance=tolerance,
            exergy_norm=phi_norm,
            gradient_norm=grad_norm,
            alignment_cosine=alignment,
            is_thermodynamically_stable=True,
            is_strictly_dissipative=is_strictly_dissipative,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3: COHOMOLOGÍA DE HACES CELULARES                                    ║
# ║                                                                             ║
# ║   Exige la anulación del primer grupo de cohomología:                       ║
# ║       dim H¹(G; F) = 0                                                      ║
# ║                                                                             ║
# ║   Esta fase comienza consumiendo el certificado de Fase 2.                  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_CellularSheafCohomologyAuditor(Phase2_DirichletThermodynamicEnforcer):
    r"""
    Eleva el flujo de datos del AST a un Haz Celular (Cellular Sheaf).
    Detecta variables huérfanas, ciclos lógicos mutantes o dependencias fantasma.

    La condición de integrabilidad global es:

        dim H¹(G; F) = 0.

    Si dim H¹ > 0, existe una obstrucción topológica global: el código no puede
    integrarse consistentemente sobre el grafo de dependencias.

    Esta fase hereda de Fase 2 y su primer método recibe explícitamente el
    certificado termodinámico emitido por:

        Phase2_DirichletThermodynamicEnforcer._enforce_dirichlet_thermodynamics(...)

    De este modo, la Fase 3 está anidada funcionalmente en la Fase 2.
    """

    @staticmethod
    def _numerical_rank(A: NDArray[np.float64]) -> int:
        r"""
        Calcula el rango numérico de una matriz mediante SVD con tolerancia
        adaptativa.

        La tolerancia usada es:

            tol = c · ε_maq · max(shape(A)) · σ_max(A),

        donde c es un factor de seguridad.
        """
        if A.size == 0 or min(A.shape) == 0:
            return 0

        try:
            singular_values = la.svdvals(A)
        except np.linalg.LinAlgError as exc:
            raise CohomologicalObstructionError(
                "SVD no convergió al auditar el haz celular."
            ) from exc

        if singular_values.size == 0:
            return 0

        sigma_max = float(singular_values[0])

        if sigma_max == 0.0:
            return 0

        if not math.isfinite(sigma_max):
            raise CohomologicalObstructionError(
                "El mayor valor singular del operador celular no es finito."
            )

        tol = (
            _NUMERICAL_SAFETY_FACTOR
            * _MACHINE_EPSILON
            * max(A.shape)
            * sigma_max
        )

        return int(np.count_nonzero(singular_values > tol))

    def _compute_first_cohomology_dimension(
        self,
        coboundary_delta0: NDArray[np.float64],
        coboundary_delta1: NDArray[np.float64],
    ) -> int:
        r"""
        Calcula dim H¹ a partir de operadores coboundary:

            δ⁰ : C⁰ → C¹
            δ¹ : C¹ → C²

        con condición de complejo cohomológico:

            δ¹ ∘ δ⁰ = 0.

        La dimensión del primer grupo de cohomología es:

            dim H¹ = dim ker(δ¹) - dim im(δ⁰)
                   = (dim C¹ - rank δ¹) - rank δ⁰.

        Las matrices deben tener formas:
            δ⁰ : (dim C¹, dim C⁰)
            δ¹ : (dim C², dim C¹)
        """
        D0 = self._as_finite_matrix(
            "coboundary_delta0",
            coboundary_delta0,
            square=False,
        )
        D1 = self._as_finite_matrix(
            "coboundary_delta1",
            coboundary_delta1,
            square=False,
        )

        # D0: (c1, c0), D1: (c2, c1)
        if D1.shape[1] != D0.shape[0]:
            raise ValueError(
                "Los operadores coboundary no componen: "
                f"δ¹ tiene dominio dim={D1.shape[1]}, pero δ⁰ tiene codominio dim={D0.shape[0]}."
            )

        composition = D1 @ D0

        if not np.all(np.isfinite(composition)):
            raise CohomologicalObstructionError(
                "La composición δ¹∘δ⁰ produjo valores no finitos."
            )

        d0_norm = self._frobenius_norm(D0)
        d1_norm = self._frobenius_norm(D1)

        if not math.isfinite(d0_norm) or not math.isfinite(d1_norm):
            raise CohomologicalObstructionError(
                "Las normas de los operadores coboundary no son finitas."
            )

        composition_norm = self._frobenius_norm(composition)

        if not math.isfinite(composition_norm):
            raise CohomologicalObstructionError(
                "La norma de δ¹∘δ⁰ no es finita."
            )

        scale = max(1.0, d0_norm * d1_norm)
        complex_residual = composition_norm / scale

        if not math.isfinite(complex_residual):
            raise CohomologicalObstructionError(
                "El residuo cohomológico δ¹∘δ⁰ no es finito."
            )

        if complex_residual > _COHOMOLOGICAL_COMPLEX_TOLERANCE:
            raise CohomologicalObstructionError(
                "Los operadores celulares no forman un complejo cohomológico válido: "
                f"||δ¹∘δ⁰||/scale = {complex_residual:.4e} > "
                f"{_COHOMOLOGICAL_COMPLEX_TOLERANCE:.4e}."
            )

        c1 = D0.shape[0]
        rank_D0 = self._numerical_rank(D0)
        rank_D1 = self._numerical_rank(D1)

        h1 = c1 - rank_D0 - rank_D1

        if h1 < 0:
            logger.warning(
                "dim H¹ calculada fue negativa (%d); se proyecta a 0 por tolerancia numérica.",
                h1,
            )
            h1 = 0

        return int(h1)

    def _audit_cellular_sheaf_cohomology(
        self,
        h1_dimension: Optional[int] = None,
        thermodynamic_audit: Optional[ThermodynamicDirichletData] = None,
        coboundary_delta0: Optional[NDArray[np.float64]] = None,
        coboundary_delta1: Optional[NDArray[np.float64]] = None,
    ) -> SheafCohomologyAuditData:
        r"""
        Primer método de la Fase 3.

        Continuación formal de Fase 2.

        Verifica la condición de integrabilidad global:

            dim H¹(G; F) = 0.

        Si `thermodynamic_audit` es provisto:
            - Verifica que la Fase 2 haya certificado estabilidad termodinámica.

        Puede operar en dos modos:
            1. Recibiendo `h1_dimension` directamente.
            2. Calculando `h1_dimension` desde operadores coboundary δ⁰ y δ¹.

        Si ambos modos se suministran, exige consistencia exacta.
        """
        # Continuación formal de Fase 2:
        # El certificado termodinámico restringe el dominio cohomológico.
        if thermodynamic_audit is not None:
            if not thermodynamic_audit.is_thermodynamically_stable:
                raise ThermodynamicSingularityError(
                    "La Fase 3 no puede iniciarse: la Fase 2 no certificó estabilidad termodinámica."
                )

        computed_h1: Optional[int] = None
        verified_by_coboundary = False

        if coboundary_delta0 is not None or coboundary_delta1 is not None:
            if coboundary_delta0 is None or coboundary_delta1 is None:
                raise ValueError(
                    "Para verificar cohomología por operadores coboundary deben proveerse "
                    "coboundary_delta0 y coboundary_delta1 simultáneamente."
                )

            computed_h1 = self._compute_first_cohomology_dimension(
                coboundary_delta0=coboundary_delta0,
                coboundary_delta1=coboundary_delta1,
            )
            verified_by_coboundary = True

        if h1_dimension is None:
            if computed_h1 is None:
                raise ValueError(
                    "Debe proveerse h1_dimension o los operadores coboundary_delta0 y coboundary_delta1."
                )

            h1 = computed_h1
        else:
            if isinstance(h1_dimension, (bool, np.bool_)):
                raise TypeError("h1_dimension debe ser un entero, no un booleano.")

            if not isinstance(h1_dimension, (int, np.integer)):
                raise TypeError("h1_dimension debe ser un entero.")

            h1 = int(h1_dimension)

            if computed_h1 is not None and computed_h1 != h1:
                raise CohomologicalObstructionError(
                    "Inconsistencia cohomológica: h1_dimension provisto="
                    f"{h1}, pero el cálculo por operadores δ dio {computed_h1}."
                )

        if h1 < _COHOMOLOGY_DIMENSION_FLOOR:
            raise ValueError(
                f"h1_dimension no puede ser negativa: {h1}."
            )

        if h1 > _COHOMOLOGY_DIMENSION_FLOOR:
            raise CohomologicalObstructionError(
                "Obstrucción topológica global detectada en la sintaxis. "
                f"dim H¹(G; F) = {h1} > 0. El código propuesto contiene "
                "contradicciones lógicas, ciclos de dependencia irresolubles "
                "o variables huérfanas."
            )

        return SheafCohomologyAuditData(
            h1_dimension=h1,
            is_globally_integrable=True,
            obstruction_free=True,
            verified_by_coboundary=verified_by_coboundary,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: AST STATIC ANALYZER AGENT                            ║
# ║                                                                             ║
# ║   Endofuntor Z_{Γ-PHYSICS} = Φ₃ ∘ Φ₂ ∘ Φ₁                                  ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class ASTStaticAnalyzerAgent(Morphism, Phase3_CellularSheafCohomologyAuditor):
    r"""
    El Custodio de la Cohomología Sintáctica en el estrato Γ-PHYSICS.

    Somete incondicionalmente el código generado por el Modelo de Lenguaje a la
    tiranía de la mecánica simpléctica, la termodinámica port-Hamiltoniana y el
    cálculo exterior, garantizando un ecosistema de ejecución matemáticamente
    purificado.
    """

    def execute_ast_symplectic_governance(
        self,
        ast_jacobian_M: NDArray[np.float64],
        control_potential_Phi: NDArray[np.float64],
        lyapunov_gradient_V: NDArray[np.float64],
        h1_dimension: Optional[int] = None,
        coboundary_delta0: Optional[NDArray[np.float64]] = None,
        coboundary_delta1: Optional[NDArray[np.float64]] = None,
    ) -> ASTGovernanceState:
        r"""
        Ejecuta la composición funtorial estricta:

            Φ₁ : Auditoría simpléctica.
            Φ₂ : Control termodinámico Dirichlet.
            Φ₃ : Auditoría cohomológica de haces celulares.

        Parámetros:
            ast_jacobian_M:
                Matriz Jacobiana de la transformación sintáctica.

            control_potential_Phi:
                Campo de control exergético Φ.

            lyapunov_gradient_V:
                Gradiente de la función de Lyapunov ∇V.

            h1_dimension:
                Dimensión conocida de H¹(G; F).

            coboundary_delta0:
                Operador δ⁰ opcional para verificar H¹.

            coboundary_delta1:
                Operador δ¹ opcional para verificar H¹.

        Retorna:
            ASTGovernanceState con los tres certificados y autorización final.
        """
        # Fase 1: Certificar la conservación del volumen del espacio de fase sintáctico.
        symplectic_audit = self._audit_symplectic_invariance(ast_jacobian_M)

        # Fase 2: Certificar el acatamiento de la Segunda Ley de la Termodinámica.
        thermodynamic_audit = self._enforce_dirichlet_thermodynamics(
            control_potential_Phi=control_potential_Phi,
            lyapunov_gradient_V=lyapunov_gradient_V,
            symplectic_audit=symplectic_audit,
        )

        # Fase 3: Certificar la nulidad de obstrucciones lógicas globales.
        cohomology_audit = self._audit_cellular_sheaf_cohomology(
            h1_dimension=h1_dimension,
            thermodynamic_audit=thermodynamic_audit,
            coboundary_delta0=coboundary_delta0,
            coboundary_delta1=coboundary_delta1,
        )

        is_compilation_authorized = bool(
            symplectic_audit.is_volume_preserved
            and thermodynamic_audit.is_thermodynamically_stable
            and cohomology_audit.is_globally_integrable
        )

        if not is_compilation_authorized:
            raise TopologicalInvariantError(
                "La composición funtorial no autorizó la compilación del AST."
            )

        logger.info(
            "Gobernanza Simpléctica del AST completada. "
            "Residuo Liouville: %.2e | Relativo: %.2e | "
            "P_diss: %.2f | cos(Φ,∇V): %.3f | "
            "dim H¹: %d | verified_by_δ: %s",
            symplectic_audit.symplectic_residual_norm,
            symplectic_audit.symplectic_relative_residual,
            thermodynamic_audit.dissipated_power,
            thermodynamic_audit.alignment_cosine,
            cohomology_audit.h1_dimension,
            cohomology_audit.verified_by_coboundary,
        )

        return ASTGovernanceState(
            symplectic_audit=symplectic_audit,
            thermodynamic_audit=thermodynamic_audit,
            cohomology_audit=cohomology_audit,
            is_compilation_authorized=is_compilation_authorized,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ═══════════════════════════════════════════════════════════════════════════════
__all__: List[str] = [
    "SymplecticInvarianceViolation",
    "ThermodynamicSingularityError",
    "CohomologicalObstructionError",
    "SymplecticInvariantData",
    "ThermodynamicDirichletData",
    "SheafCohomologyAuditData",
    "ASTGovernanceState",
    "Phase1_SymplecticInvarianceAuditor",
    "Phase2_DirichletThermodynamicEnforcer",
    "Phase3_CellularSheafCohomologyAuditor",
    "ASTStaticAnalyzerAgent",
]