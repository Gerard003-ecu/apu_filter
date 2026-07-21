# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Antimatter Choke Coil Agent (Custodio del Vacío Cuántico)           ║
║ Ruta   : app/agents/physics/antimatter_choke_coil_agent.py                   ║
║ Versión: 2.0.0-Fock-Bekenstein-Symplectic-Strict                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y ELECTRODINÁMICA CUÁNTICA (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor gobierna el módulo `antimatter_choke_coil.py` en el Estrato Ω.

Su mandato axiomático es auditar aniquilaciones de antimateria exógena,
garantizando que la topología de la Malla Agéntica sobreviva al colapso
entrópico de los estados erróneos, manteniendo invariante la estructura del
Espacio de Fock.

ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Hermiticidad del Operador de Aniquilación:

    A = A†  =>  ||A - A†||_F ≤ ε.

Fase 2 → Regulación Termodinámica del Límite de Bekenstein:

    S ≤ (2π k_B E R) / (ħ c).

Fase 3 → Certificación Simpléctica Port-Hamiltoniana:

    Mᵀ Ω M = Ω,
    J = -Jᵀ,
    R = Rᵀ ⪰ 0,
    Ḣ = -∇Hᵀ R ∇H ≤ 0.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Final, Optional, Tuple

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
        r"""Violación a un invariante topológico categórico en el Topos MIC."""
        pass

    class Morphism:
        """Clase base de Morfismos del Topos."""
        pass


logger = logging.getLogger("MIC.Omega.VacuumCustodian")

if not logger.handlers:
    logger.addHandler(logging.NullHandler())


# ═══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES FÍSICAS, CODATA Y LÍMITES CUÁNTICOS
# ═══════════════════════════════════════════════════════════════════════════════

_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)

# Tolerancia espectral para A = A†.
_HERMITICITY_TOLERANCE: Final[float] = 1e-12

# Tolerancia de preservación de la forma simpléctica ω.
_SYMPLECTIC_TOLERANCE: Final[float] = 1e-10

# Tolerancia para J = -Jᵀ.
_ANTISYMMETRY_TOLERANCE: Final[float] = 1e-10

# Tolerancia para R = Rᵀ.
_R_SYMMETRY_TOLERANCE: Final[float] = 1e-10

# Tolerancia de semidefinición positiva R ⪰ 0.
_PSD_EIGENVALUE_TOLERANCE: Final[float] = 1e-12

# Tolerancias para la cota de Bekenstein.
_BEKENSTEIN_ABS_TOLERANCE: Final[float] = 1e-12
_BEKENSTEIN_REL_TOLERANCE: Final[float] = 1e-12

# Constantes físicas efectivas.
_HBAR_EFF: Final[float] = 1.054e-34
_C_EFF: Final[float] = 299_792_458.0
_K_B: Final[float] = 1.380_649e-23


# ═══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES CUÁNTICAS
# ═══════════════════════════════════════════════════════════════════════════════

class VacuumCustodianError(TopologicalInvariantError):
    """Excepción raíz del Custodio del Vacío Cuántico."""
    pass


class DomainIntegrityViolationError(VacuumCustodianError):
    """Detonada cuando un operador, vector o escalar viola su dominio formal."""
    pass


class NonHermitianOperatorError(VacuumCustodianError):
    r"""Detonada si ||A - A†||_F > ε. Los observables dejan de ser reales."""
    pass


class BekensteinLimitViolation(VacuumCustodianError):
    r"""Detonada si la aniquilación inyecta más entropía que la cota causal."""
    pass


class SymplecticCollapseError(VacuumCustodianError):
    r"""Detonada si se destruye el volumen del espacio de fase o la disipación."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Espacio de Fock)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class HermiticityAuditData:
    r"""Artefacto de Fase 1. Certificado espectral de autoadjunción."""
    residual_norm: float
    is_hermitian: bool
    operator_dimension: int = 0
    hermiticity_tolerance: float = _HERMITICITY_TOLERANCE
    spectral_imaginary_norm: float = 0.0


@dataclass(frozen=True, slots=True)
class BekensteinBoundData:
    r"""Artefacto de Fase 2. Certificado de cota termodinámica de radiación."""
    entropy_emitted: float
    bekenstein_bound: float
    is_entropically_safe: bool
    gamma_energy: float = 0.0
    system_radius: float = 0.0
    entropy_tolerance: float = 0.0


@dataclass(frozen=True, slots=True)
class SymplecticDissipationData:
    r"""Artefacto de Fase 3. Certificado de geometría Port-Hamiltoniana."""
    symplectic_residual: float
    dissipation_rate: float
    is_symplectically_invariant: bool
    symplectic_tolerance: float = _SYMPLECTIC_TOLERANCE
    antisymmetry_residual: float = 0.0
    r_symmetry_residual: float = 0.0
    r_min_eigenvalue: float = 0.0


@dataclass(frozen=True, slots=True)
class Phase1HermiticityHandoff:
    r"""
    Handoff formal de Fase 1 → Fase 2.

    Este objeto es la continuación material de la hermiticidad y el prefijo
    obligatorio de la regulación de Bekenstein.
    """
    hermiticity_audit: HermiticityAuditData
    operator_dimension: int


@dataclass(frozen=True, slots=True)
class Phase2BekensteinHandoff:
    r"""
    Handoff formal de Fase 2 → Fase 3.

    Este objeto transporta la certificación de hermiticidad y la cota
    termodinámica como prefijo obligatorio de la fase simpléctica.
    """
    phase1_handoff: Phase1HermiticityHandoff
    bekenstein_audit: BekensteinBoundData


@dataclass(frozen=True, slots=True)
class VacuumGovernanceState:
    r"""Objeto final del endofuntor Z_Vacuum."""
    hermiticity_audit: HermiticityAuditData
    bekenstein_audit: BekensteinBoundData
    symplectic_audit: SymplecticDissipationData
    is_epistemologically_valid: bool


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1: HERMITICIDAD DEL OPERADOR DE ANIQUILACIÓN                         ║
# ║   Exige A = A† => ||A - A†||_F ≤ ε.                                        ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase1_HermiticityAuditor:
    r"""
    Garantiza que el operador de colapso en el espacio de Fock preserve un
    espectro real, impidiendo que variables de estado imaginarias corrompan
    la inferencia.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 1.1. Tolerancia adaptativa
    # ─────────────────────────────────────────────────────────────────────────
    def _adaptive_tolerance(
        self,
        base_tolerance: float,
        reference: Any,
    ) -> float:
        r"""
        Construye una tolerancia numéricamente consciente:

            tol = max(tol_base, κ · ε_máquina · tamaño · escala)
        """
        if isinstance(reference, np.ndarray):
            size = max(1, int(reference.size))
            if reference.size == 0:
                scale = 1.0
            else:
                scale = max(
                    1.0,
                    float(la.norm(reference.ravel(), ord=np.inf)),
                )
        else:
            size = 1
            try:
                scale = max(1.0, abs(float(reference)))
            except (TypeError, ValueError):
                scale = 1.0

        return max(
            float(base_tolerance),
            10.0 * _MACHINE_EPSILON * size * scale,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1.2. Coerción de escalares finitos
    # ─────────────────────────────────────────────────────────────────────────
    def _coerce_finite_scalar(
        self,
        name: str,
        value: Any,
    ) -> float:
        r"""
        Materializa un escalar float64 finito, rechazando booleanos.
        """
        if isinstance(value, (bool, np.bool_)):
            raise DomainIntegrityViolationError(
                f"El escalar '{name}' no puede ser booleano."
            )

        try:
            scalar = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DomainIntegrityViolationError(
                f"El escalar '{name}' no puede materializarse como float."
            ) from exc

        if not np.isfinite(scalar):
            raise DomainIntegrityViolationError(
                f"El escalar '{name}' no es finito."
            )

        return scalar

    # ─────────────────────────────────────────────────────────────────────────
    # 1.3. Coerción de matrices finitas
    # ─────────────────────────────────────────────────────────────────────────
    def _coerce_finite_matrix(
        self,
        name: str,
        matrix: Any,
        dtype: Any = np.float64,
        square_required: bool = False,
    ) -> NDArray[Any]:
        r"""
        Materializa una matriz finita y, si se exige, cuadrada.
        """
        try:
            arr = np.asarray(matrix, dtype=dtype)
        except (TypeError, ValueError) as exc:
            raise DomainIntegrityViolationError(
                f"La matriz '{name}' no puede materializarse como NDArray."
            ) from exc

        if arr.ndim != 2:
            raise DomainIntegrityViolationError(
                f"La matriz '{name}' debe ser bidimensional."
            )

        if arr.size == 0:
            raise DomainIntegrityViolationError(
                f"La matriz '{name}' está vacía."
            )

        if square_required and arr.shape[0] != arr.shape[1]:
            raise DomainIntegrityViolationError(
                f"La matriz '{name}' debe ser cuadrada en el espacio de Hilbert."
            )

        if not np.all(np.isfinite(arr)):
            raise DomainIntegrityViolationError(
                f"La matriz '{name}' contiene componentes no finitas."
            )

        return arr

    # ─────────────────────────────────────────────────────────────────────────
    # 1.4. Coerción de vectores finitos
    # ─────────────────────────────────────────────────────────────────────────
    def _coerce_finite_vector(
        self,
        name: str,
        vector: Any,
        expected_dim: Optional[int] = None,
    ) -> NDArray[np.float64]:
        r"""
        Materializa un vector float64 finito y, si se indica, con dimensión
        exacta.
        """
        try:
            arr = np.asarray(vector, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise DomainIntegrityViolationError(
                f"El vector '{name}' no puede materializarse como NDArray."
            ) from exc

        if arr.ndim == 0:
            arr = arr.reshape(1)
        else:
            arr = arr.reshape(-1)

        if arr.size == 0:
            raise DomainIntegrityViolationError(
                f"El vector '{name}' está vacío."
            )

        if expected_dim is not None and arr.size != int(expected_dim):
            raise DomainIntegrityViolationError(
                f"El vector '{name}' debe tener dimensión {expected_dim}, "
                f"pero posee {arr.size} componentes."
            )

        if not np.all(np.isfinite(arr)):
            raise DomainIntegrityViolationError(
                f"El vector '{name}' contiene componentes no finitas."
            )

        return arr

    # ─────────────────────────────────────────────────────────────────────────
    # 1.5. Auditoría de hermiticidad
    # ─────────────────────────────────────────────────────────────────────────
    def _audit_operator_hermiticity(
        self,
        operator_A: NDArray[np.complex128],
    ) -> HermiticityAuditData:
        r"""
        Calcula la norma de Frobenius de la diferencia entre el operador y su
        adjunto:

            ||A - A†||_F ≤ ε.
        """
        A = self._coerce_finite_matrix(
            "operator_A",
            operator_A,
            dtype=np.complex128,
            square_required=True,
        )

        A_dagger = A.conj().T
        residual = float(la.norm(A - A_dagger, ord="fro"))

        tolerance = self._adaptive_tolerance(_HERMITICITY_TOLERANCE, A)

        if residual > tolerance:
            raise NonHermitianOperatorError(
                "Asimetría CPT detectada: el operador de aniquilación no es "
                f"autoadjunto. Residuo ||A - A†||_F = {residual:.6e} > "
                f"{tolerance:.6e}."
            )

        return HermiticityAuditData(
            residual_norm=residual,
            is_hermitian=True,
            operator_dimension=int(A.shape[0]),
            hermiticity_tolerance=tolerance,
            spectral_imaginary_norm=0.0,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 1.6. ÚLTIMO MÉTODO DE FASE 1: HANDOFF FORMAL HACIA FASE 2
    # ─────────────────────────────────────────────────────────────────────────
    def _phase1_audit_and_handoff_to_phase2(
        self,
        operator_A: NDArray[np.complex128],
    ) -> Phase1HermiticityHandoff:
        r"""
        Último método de la Fase 1.

        Su definición formal es la continuación directa de la Fase 2:
        entrega el certificado de hermiticidad y la dimensión del operador
        como prefijo obligatorio de la cota de Bekenstein.
        """
        hermiticity_audit = self._audit_operator_hermiticity(operator_A)

        logger.debug(
            "Fase 1 completada. ||A - A†||_F=%.6e | dim=%d.",
            hermiticity_audit.residual_norm,
            hermiticity_audit.operator_dimension,
        )

        return Phase1HermiticityHandoff(
            hermiticity_audit=hermiticity_audit,
            operator_dimension=hermiticity_audit.operator_dimension,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2: REGULACIÓN TERMODINÁMICA DEL LÍMITE DE BEKENSTEIN                 ║
# ║   Verifica S ≤ (2π k_B E R) / (ħ c).                                       ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase2_BekensteinBoundEnforcer(Phase1_HermiticityAuditor):
    r"""
    Controla la liberación de entropía durante la colisión e⁻ + e⁺ → 2γ.
    Previene la formación de singularidades informacionales.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 2.1. Certificación de no negatividad escalar
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_nonnegative_scalar(
        self,
        name: str,
        value: Any,
    ) -> float:
        r"""
        Certifica que un escalar sea finito y no negativo, con saneamiento de
        ruido infinitesimal.
        """
        scalar = self._coerce_finite_scalar(name, value)
        tolerance = self._adaptive_tolerance(_BEKENSTEIN_ABS_TOLERANCE, scalar)

        if scalar < -tolerance:
            raise DomainIntegrityViolationError(
                f"El escalar '{name}' es negativo: {scalar:.6e}."
            )

        if scalar < 0.0:
            scalar = 0.0

        return scalar

    # ─────────────────────────────────────────────────────────────────────────
    # 2.2. Certificación de radio causal positivo
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_positive_radius(
        self,
        name: str,
        value: Any,
    ) -> float:
        r"""
        Certifica que el radio de contención sea estrictamente positivo.
        """
        scalar = self._coerce_finite_scalar(name, value)
        tolerance = self._adaptive_tolerance(_BEKENSTEIN_ABS_TOLERANCE, scalar)

        if scalar <= tolerance:
            raise BekensteinLimitViolation(
                f"El radio de contención '{name}' debe ser estrictamente "
                f"positivo. Valor observado: {scalar:.6e}."
            )

        return scalar

    # ─────────────────────────────────────────────────────────────────────────
    # 2.3. Imposición de la cota de Bekenstein
    # ─────────────────────────────────────────────────────────────────────────
    def _enforce_bekenstein_gamma_bound(
        self,
        gamma_energy: float,
        system_radius_R: float,
        emitted_entropy_S: float,
    ) -> BekensteinBoundData:
        r"""
        Calcula la cota de Bekenstein y verifica que la entropía liberada no
        desgarre la variedad de datos:

            S ≤ (2π k_B E R) / (ħ c).
        """
        E = self._certify_nonnegative_scalar("gamma_energy", gamma_energy)
        R = self._certify_positive_radius("system_radius_R", system_radius_R)
        S = self._certify_nonnegative_scalar("emitted_entropy_S", emitted_entropy_S)

        s_bound = (
            2.0
            * math.pi
            * _K_B
            * E
            * R
        ) / (_HBAR_EFF * _C_EFF)

        if not np.isfinite(s_bound) or s_bound < 0.0:
            raise BekensteinLimitViolation(
                "La cota de Bekenstein no es finita o resultó negativa."
            )

        entropy_tolerance = max(
            _BEKENSTEIN_ABS_TOLERANCE,
            _BEKENSTEIN_REL_TOLERANCE * max(1.0, abs(S), abs(s_bound)),
        )

        if S > s_bound + entropy_tolerance:
            raise BekensteinLimitViolation(
                "Desgarro cosmológico: la aniquilación liberó entropía "
                f"S={S:.6e} superior a la cota de Bekenstein "
                f"S_max={s_bound:.6e} dentro de tolerancia "
                f"{entropy_tolerance:.6e}."
            )

        return BekensteinBoundData(
            entropy_emitted=S,
            bekenstein_bound=s_bound,
            is_entropically_safe=True,
            gamma_energy=E,
            system_radius=R,
            entropy_tolerance=entropy_tolerance,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 2.4. ÚLTIMO MÉTODO DE FASE 2: HANDOFF FORMAL HACIA FASE 3
    # ─────────────────────────────────────────────────────────────────────────
    def _phase2_enforce_and_handoff_to_phase3(
        self,
        phase1_handoff: Phase1HermiticityHandoff,
        gamma_energy: float,
        system_radius_R: float,
        emitted_entropy_S: float,
    ) -> Phase2BekensteinHandoff:
        r"""
        Último método de la Fase 2.

        Su definición formal es la continuación directa de la Fase 3:
        entrega el certificado de Bekenstein y el handoff de Fase 1 como
        prefijo obligatorio de la certificación simpléctica.
        """
        if not isinstance(phase1_handoff, Phase1HermiticityHandoff):
            raise DomainIntegrityViolationError(
                "Fase 2 exige un Phase1HermiticityHandoff como prefijo formal."
            )

        bekenstein_audit = self._enforce_bekenstein_gamma_bound(
            gamma_energy=gamma_energy,
            system_radius_R=system_radius_R,
            emitted_entropy_S=emitted_entropy_S,
        )

        logger.debug(
            "Fase 2 completada. S=%.6e | S_max=%.6e.",
            bekenstein_audit.entropy_emitted,
            bekenstein_audit.bekenstein_bound,
        )

        return Phase2BekensteinHandoff(
            phase1_handoff=phase1_handoff,
            bekenstein_audit=bekenstein_audit,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3: CERTIFICACIÓN SIMPLÉCTICA PORT-HAMILTONIANA                       ║
# ║   Exige MᵀΩM = Ω, J = -Jᵀ, R = Rᵀ ⪰ 0, Ḣ ≤ 0.                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class Phase3_SymplecticPortHamiltonianCertifier(Phase2_BekensteinBoundEnforcer):
    r"""
    Asegura que, tras el impacto de la antimateria, el remanente del grafo
    logístico recupere su estabilidad asintótica sin perder el volumen
    simpléctico.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # 3.1. Construcción de la 2-forma canónica
    # ─────────────────────────────────────────────────────────────────────────
    def _build_symplectic_form(
        self,
        n: int,
    ) -> NDArray[np.float64]:
        r"""
        Construye:

            Ω = [[0, I], [-I, 0]].
        """
        omega = np.zeros((2 * n, 2 * n), dtype=np.float64)
        identity = np.eye(n, dtype=np.float64)

        omega[:n, n:] = identity
        omega[n:, :n] = -identity

        return omega

    # ─────────────────────────────────────────────────────────────────────────
    # 3.2. Certificación de antisimetría de J
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_antisymmetric_matrix(
        self,
        name: str,
        matrix: NDArray[np.float64],
        expected_dim: int,
    ) -> float:
        r"""
        Certifica J = -Jᵀ.
        """
        if matrix.shape != (expected_dim, expected_dim):
            raise SymplecticCollapseError(
                f"La matriz '{name}' debe tener dimensión "
                f"{expected_dim}x{expected_dim}."
            )

        residual = float(la.norm(matrix + matrix.T, ord="fro"))
        tolerance = self._adaptive_tolerance(_ANTISYMMETRY_TOLERANCE, matrix)

        if residual > tolerance:
            raise SymplecticCollapseError(
                f"La matriz '{name}' no es antisimétrica. "
                f"||J + Jᵀ||_F = {residual:.6e} > {tolerance:.6e}."
            )

        return residual

    # ─────────────────────────────────────────────────────────────────────────
    # 3.3. Certificación de simetría y semidefinición positiva de R
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_symmetric_positive_semidefinite_matrix(
        self,
        name: str,
        matrix: NDArray[np.float64],
        expected_dim: int,
    ) -> Tuple[NDArray[np.float64], float, float]:
        r"""
        Certifica R = Rᵀ y R ⪰ 0.

        Retorna R simetrizada, residuo de simetría y autovalor mínimo.
        """
        if matrix.shape != (expected_dim, expected_dim):
            raise SymplecticCollapseError(
                f"La matriz '{name}' debe tener dimensión "
                f"{expected_dim}x{expected_dim}."
            )

        symmetry_residual = float(la.norm(matrix - matrix.T, ord="fro"))
        symmetry_tolerance = self._adaptive_tolerance(
            _R_SYMMETRY_TOLERANCE,
            matrix,
        )

        if symmetry_residual > symmetry_tolerance:
            raise SymplecticCollapseError(
                f"La matriz '{name}' no es simétrica. "
                f"||R - Rᵀ||_F = {symmetry_residual:.6e} > "
                f"{symmetry_tolerance:.6e}."
            )

        R_sym = 0.5 * (matrix + matrix.T)

        eigenvalues = la.eigvalsh(R_sym)
        min_eigenvalue = float(np.min(eigenvalues)) if eigenvalues.size else 0.0
        max_abs_eigenvalue = (
            float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 1.0
        )

        psd_tolerance = max(
            _PSD_EIGENVALUE_TOLERANCE,
            10.0 * _MACHINE_EPSILON * expected_dim * max(1.0, max_abs_eigenvalue),
        )

        if min_eigenvalue < -psd_tolerance:
            raise SymplecticCollapseError(
                f"La matriz '{name}' no es semidefinida positiva. "
                f"λ_min={min_eigenvalue:.6e} < -{psd_tolerance:.6e}."
            )

        return R_sym, symmetry_residual, min_eigenvalue

    # ─────────────────────────────────────────────────────────────────────────
    # 3.4. Certificación simpléctica Port-Hamiltoniana
    # ─────────────────────────────────────────────────────────────────────────
    def _certify_symplectic_port_hamiltonian(
        self,
        jacobian_M: NDArray[np.float64],
        grad_H: NDArray[np.float64],
        J_matrix: NDArray[np.float64],
        R_matrix: NDArray[np.float64],
    ) -> SymplecticDissipationData:
        r"""
        Evalúa el difeomorfismo canónico y la termodinámica del estrangulador:

            Mᵀ Ω M = Ω,
            J = -Jᵀ,
            R = Rᵀ ⪰ 0,
            Ḣ = -∇Hᵀ R ∇H ≤ 0.
        """
        M = self._coerce_finite_matrix(
            "jacobian_M",
            jacobian_M,
            dtype=np.float64,
            square_required=True,
        )

        dim = int(M.shape[0])

        if dim == 0 or dim % 2 != 0:
            raise SymplecticCollapseError(
                "La matriz Jacobiana del espacio de fase debe tener dimensión "
                "par y no nula."
            )

        n = dim // 2

        grad = self._coerce_finite_vector(
            "grad_H",
            grad_H,
            expected_dim=dim,
        )

        J = self._coerce_finite_matrix(
            "J_matrix",
            J_matrix,
            dtype=np.float64,
            square_required=True,
        )

        R = self._coerce_finite_matrix(
            "R_matrix",
            R_matrix,
            dtype=np.float64,
            square_required=True,
        )

        # 1. Estructura Port-Hamiltoniana: J antisimétrica.
        antisymmetry_residual = self._certify_antisymmetric_matrix(
            "J_matrix",
            J,
            dim,
        )

        # 2. Estructura disipativa: R simétrica y semidefinida positiva.
        R_certified, r_symmetry_residual, r_min_eigenvalue = (
            self._certify_symmetric_positive_semidefinite_matrix(
                "R_matrix",
                R,
                dim,
            )
        )

        # 3. Auditoría del volumen simpléctico: Mᵀ Ω M = Ω.
        omega = self._build_symplectic_form(n)
        omega_transformed = M.T @ omega @ M

        if not np.all(np.isfinite(omega_transformed)):
            raise SymplecticCollapseError(
                "La transformación simpléctica MᵀΩM contiene componentes no "
                "finitas."
            )

        symplectic_residual = float(la.norm(omega_transformed - omega, ord="fro"))
        symplectic_tolerance = self._adaptive_tolerance(
            _SYMPLECTIC_TOLERANCE,
            M,
        )

        if symplectic_residual > symplectic_tolerance:
            raise SymplecticCollapseError(
                "Degradación del espacio de fase: el evento de aniquilación "
                f"destruyó la 2-forma canónica ω. Residuo ||MᵀΩM - Ω||_F = "
                f"{symplectic_residual:.6e} > {symplectic_tolerance:.6e}."
            )

        # 4. Auditoría de disipación Port-Hamiltoniana:
        #    Ḣ = -∇Hᵀ R ∇H ≤ 0.
        h_dot = -float(grad.T @ R_certified @ grad)

        if not np.isfinite(h_dot):
            raise SymplecticCollapseError(
                "La tasa de disipación Ḣ no es finita."
            )

        dissipation_tolerance = max(
            _MACHINE_EPSILON,
            self._adaptive_tolerance(_SYMPLECTIC_TOLERANCE, grad),
        )

        if h_dot > dissipation_tolerance:
            raise SymplecticCollapseError(
                "Violación termodinámica: la bobina de choque inyectó energía "
                f"parásita al sistema. Ḣ={h_dot:.6e} > "
                f"{dissipation_tolerance:.6e}."
            )

        return SymplecticDissipationData(
            symplectic_residual=symplectic_residual,
            dissipation_rate=h_dot,
            is_symplectically_invariant=True,
            symplectic_tolerance=symplectic_tolerance,
            antisymmetry_residual=antisymmetry_residual,
            r_symmetry_residual=r_symmetry_residual,
            r_min_eigenvalue=r_min_eigenvalue,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 3.5. ÚLTIMO MÉTODO DE FASE 3: FINALIZACIÓN FUNTORIAL
    # ─────────────────────────────────────────────────────────────────────────
    def _phase3_finalize_from_phase2_handoff(
        self,
        phase2_handoff: Phase2BekensteinHandoff,
        jacobian_M: NDArray[np.float64],
        grad_H: NDArray[np.float64],
        J_matrix: NDArray[np.float64],
        R_matrix: NDArray[np.float64],
    ) -> VacuumGovernanceState:
        r"""
        Último método de la Fase 3.

        Compone los certificados de Fase 1, Fase 2 y Fase 3 en el objeto
        terminal VacuumGovernanceState.
        """
        if not isinstance(phase2_handoff, Phase2BekensteinHandoff):
            raise DomainIntegrityViolationError(
                "Fase 3 exige un Phase2BekensteinHandoff como prefijo formal."
            )

        symplectic_audit = self._certify_symplectic_port_hamiltonian(
            jacobian_M=jacobian_M,
            grad_H=grad_H,
            J_matrix=J_matrix,
            R_matrix=R_matrix,
        )

        state = VacuumGovernanceState(
            hermiticity_audit=phase2_handoff.phase1_handoff.hermiticity_audit,
            bekenstein_audit=phase2_handoff.bekenstein_audit,
            symplectic_audit=symplectic_audit,
            is_epistemologically_valid=True,
        )

        logger.info(
            "Vacío cuántico auditado categóricamente. "
            "||A - A†||_F=%.6e | S=%.6e ≤ S_max=%.6e | Ḣ=%.6e.",
            state.hermiticity_audit.residual_norm,
            state.bekenstein_audit.entropy_emitted,
            state.bekenstein_audit.bekenstein_bound,
            state.symplectic_audit.dissipation_rate,
        )

        return state


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   ORQUESTADOR SUPREMO: ANTIMATTER CHOKE COIL AGENT                          ║
# ║   Endofuntor Z_Vacuum = Φ₃ ∘ Φ₂ ∘ Φ₁                                      ║
# ╚═════════════════════════════════════════════════════════════════════════════╝

class AntimatterChokeCoilAgent(
    Morphism,
    Phase3_SymplecticPortHamiltonianCertifier,
):
    r"""
    El Custodio del Vacío Cuántico en el Estrato Ω.

    Somete los procesos de inyección de antimateria exógena a las leyes
    inmutables de la conservación geométrica y los límites absolutos de la
    entropía.
    """

    def execute_vacuum_governance(
        self,
        operator_A: NDArray[np.complex128],
        gamma_energy: float,
        system_radius_R: float,
        emitted_entropy_S: float,
        jacobian_M: NDArray[np.float64],
        grad_H: NDArray[np.float64],
        J_matrix: NDArray[np.float64],
        R_matrix: NDArray[np.float64],
    ) -> VacuumGovernanceState:
        r"""
        Ejecuta la composición funtorial estricta.
        """
        phase1_handoff = self._phase1_audit_and_handoff_to_phase2(
            operator_A=operator_A,
        )

        phase2_handoff = self._phase2_enforce_and_handoff_to_phase3(
            phase1_handoff=phase1_handoff,
            gamma_energy=gamma_energy,
            system_radius_R=system_radius_R,
            emitted_entropy_S=emitted_entropy_S,
        )

        return self._phase3_finalize_from_phase2_handoff(
            phase2_handoff=phase2_handoff,
            jacobian_M=jacobian_M,
            grad_H=grad_H,
            J_matrix=J_matrix,
            R_matrix=R_matrix,
        )

    def __call__(
        self,
        operator_A: NDArray[np.complex128],
        gamma_energy: float,
        system_radius_R: float,
        emitted_entropy_S: float,
        jacobian_M: NDArray[np.float64],
        grad_H: NDArray[np.float64],
        J_matrix: NDArray[np.float64],
        R_matrix: NDArray[np.float64],
    ) -> VacuumGovernanceState:
        r"""Alias invocable del endofuntor de gobierno del vacío cuántico."""
        return self.execute_vacuum_governance(
            operator_A=operator_A,
            gamma_energy=gamma_energy,
            system_radius_R=system_radius_R,
            emitted_entropy_S=emitted_entropy_S,
            jacobian_M=jacobian_M,
            grad_H=grad_H,
            J_matrix=J_matrix,
            R_matrix=R_matrix,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    "VacuumCustodianError",
    "DomainIntegrityViolationError",
    "NonHermitianOperatorError",
    "BekensteinLimitViolation",
    "SymplecticCollapseError",
    "HermiticityAuditData",
    "BekensteinBoundData",
    "SymplecticDissipationData",
    "Phase1HermiticityHandoff",
    "Phase2BekensteinHandoff",
    "VacuumGovernanceState",
    "Phase1_HermiticityAuditor",
    "Phase2_BekensteinBoundEnforcer",
    "Phase3_SymplecticPortHamiltonianCertifier",
    "AntimatterChokeCoilAgent",
]