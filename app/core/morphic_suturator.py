# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Morphic Suturator (Sutura e Integración de MIC y MAC)              ║
║  Ruta   : app/core/morphic_suturator.py                                      ║
║  Versión: 2.0.0-Galois-Adjunction-Spectral-Strict-Granular                   ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  NATURALEZA CIBER-FÍSICA Y RIGOR DOCTORAL:                                   ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Este módulo actúa como el morfismo de sutura supremo que unifica el         ║
║  espacio discreto de acción de la MIC (Matriz de Interacción Central) con    ║
║  el espacio de Hilbert continuo de la MAC (Matriz Atómica de Conocimiento).  ║
║                                                                              ║
║  Arquitectura de Fases Anidadas (continuación funtorial estricta):           ║
║    FASE 1 — Saneamiento IEEE-754, rango espectral (SVD/Gershgorin) y         ║
║             ortogonalidad O(n) de la MIC.                                    ║
║    FASE 2 — Postulados de Dirac–von Neumann sobre la MAC: hermiticidad,      ║
║             traza unitaria, positividad espectral, pureza y entropía.        ║
║    FASE 3 — Adjunción de Galois F ⊣ G: unidad η_X = X - G(F(X)), condición   ║
║             de Lipschitz cuantitativa y fidelidad de reconstrucción.         ║
║                                                                              ║
║  $$\text{Hom}_{\mathcal{D}}(F(X), Y) \cong \text{Hom}_{\mathcal{C}}(X, G(Y))$$║
║                                                                              ║
║  Cualquier violación a los invariantes algebraicos colapsa el estado de      ║
║  manera idempotente, activando el veto en el retículo distributivo Ω₃.       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Final, List, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Dependencias del ecosistema con fallbacks robustos
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
    from app.core.schemas import Stratum
except ImportError:  # pragma: no cover — entorno aislado / unit tests sin app
    class TopologicalInvariantError(Exception):
        """Excepción base del sistema para violaciones topológico-algebraicas."""

    class Morphism:
        """Clase base para morfismos categóricos en C_MIC."""

        def __init__(self, *args, **kwargs) -> None:
            pass

    class Stratum(Enum):
        """Estratos de la pirámide de información DIKW."""

        PHYSICS = 1
        TACTICS = 2
        STRATEGY = 3
        WISDOM = 4


logger = logging.getLogger("MIC.Wisdom.MorphicSuturator")

# ─────────────────────────────────────────────────────────────────────────────
# Tipos canónicos
# ─────────────────────────────────────────────────────────────────────────────
RealMatrix = NDArray[np.float64]
RealVector = NDArray[np.float64]
ComplexMatrix = NDArray[np.complex128]
RealScalar = float
BoolLattice = bool

# Constantes espectrales del silicio (tolerancia de Wilkinson)
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_DEFAULT_TOL: Final[float] = 1.0e-12
_SPECTRAL_PSD_FLOOR: Final[float] = -1.0e-13
_EPS_HERMITICITY: Final[float] = 1.0e-10
_EPS_ORTHOGONALITY: Final[float] = 1.0e-8
_EPS_ENTROPY_FLOOR: Final[float] = 1.0e-15


# ═══════════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES MORFOLÓGICAS (VETOS DE SUTURA — RETÍCULO Ω₃)
# ═══════════════════════════════════════════════════════════════════════════════
class MorphicSuturatorError(TopologicalInvariantError):
    """Excepción raíz del MorphicSuturator."""


class NonFiniteInputError(MorphicSuturatorError):
    """Lanzada si cualquier tensor de entrada contiene NaN/Inf (fuga IEEE-754)."""


class ShapeMismatchError(MorphicSuturatorError):
    """Lanzada ante inconsistencias dimensionales entre operadores/estados."""


class MicRankDeficiencyError(MorphicSuturatorError):
    """Lanzada si la MIC pierde rango completo o tiene columnas linealmente dependientes."""


class MicOrthogonalityBreachError(MorphicSuturatorError):
    """
    Reservada para políticas estrictas que exijan ortogonalidad O(n) como
    hard-gate. Por defecto la desviación de ortogonalidad es un invariante
    blando (se reporta, no aborta) salvo que ``strict_orthogonality=True``.
    """


class MacDensityAnomalyError(MorphicSuturatorError):
    """Excepción paraguas para violaciones de los postulados de Dirac-von Neumann."""


class MacHermiticityViolation(MacDensityAnomalyError):
    """La MAC viola el postulado autoadjunto: ρ ≠ ρ†."""


class MacTraceAnomalyError(MacDensityAnomalyError):
    """La MAC no conserva la probabilidad: Tr(ρ) ≠ 1."""


class MacPositivitySpectralError(MacDensityAnomalyError):
    """La MAC posee autovalores por debajo del floor PSD: ρ ⋡ 0."""


class LipschitzParameterError(MorphicSuturatorError):
    """La cota de Lipschitz L_max o la tolerancia de holgura es inválida."""


class GaloisAdjunctionBreachError(MorphicSuturatorError):
    """Lanzada si el residuo de la adjunción Hom(F(X), Y) ≄ Hom(X, G(Y)) supera la tolerancia."""


# ═══════════════════════════════════════════════════════════════════════════════
# DTOs INMUTABLES (Contratos entre Fases del Funtor)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class MicRankCertificate:
    r"""
    Certificado de rango y ortogonalidad de la Matriz de Interacción Central.

    Campos originales preservados; se añaden invariantes espectrales blandos
    de bajo costo (Gershgorin) y de estabilidad (radio espectral) que no
    requieren SVD completa para una primera estimación defensiva.
    """
    matrix_shape: Tuple[int, int]
    effective_rank: int
    condition_number: float
    is_full_rank: bool
    orthogonality_deviation: float
    is_orthogonal: bool
    is_finite: bool = True
    rank_deficiency_margin: int = 0
    spectral_radius: float = 0.0
    gershgorin_bound: float = 0.0


@dataclass(frozen=True, slots=True)
class MacHermiticityCertificate:
    r"""
    Certificado cuántico del operador de densidad de la MAC.

    Se añaden la dimensión de Hilbert, la certificación de la cota de pureza
    $\gamma\in[1/d,1]$ (mencionada mas nunca verificada en v1.0.0) y la
    entropía de von Neumann como telemetría espectral de segundo orden.
    """
    is_hermitian: bool
    hermitician_residual: float
    trace_value: float
    is_trace_normalized: bool
    minimum_eigenvalue: float
    is_positive_semidefinite: bool
    quantum_purity: float
    hilbert_dimension: int = 0
    is_purity_bounded: bool = True
    von_neumann_entropy: float = 0.0


@dataclass(frozen=True, slots=True)
class GaloisAdjunctionCertificate:
    r"""
    Certificado del residuo del isomorfismo del par funtorial adyacente F ⊣ G.

    ``adjunction_residual`` es precisamente la norma de la *unidad* de la
    adjunción $\eta_X = X - G(F(X))$. Se añade ``target_diff`` (norma del
    lado derecho de la desigualdad de Lipschitz) como telemetría explícita
    y un indicador de recorte de la fidelidad.
    """
    adjunction_residual: float
    is_adjunction_secured: bool
    lipschitz_bound: float
    reconstruction_fidelity: float
    target_diff: float = 0.0
    fidelity_was_clipped: bool = False


@dataclass(frozen=True, slots=True)
class MorphicSuturationState:
    """Estado final unificado y certificado de la sutura ciber-física MIC-MAC."""
    mic_audit: MicRankCertificate
    mac_audit: MacHermiticityCertificate
    galois_audit: GaloisAdjunctionCertificate
    is_sutured_coherent: bool
    timestamp_utc: str
    wilkinson_tolerance: float = _DEFAULT_TOL
    stratum: str = "WISDOM"


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1 — AUDITORÍA DE RANGO Y ORTOGONALIDAD DE LA MIC (Observe)
# Objetos: M ∈ Mat_n(ℝ), σ(M), discos de Gershgorin, Gram(M)
# ═══════════════════════════════════════════════════════════════════════════════
class Phase1_MicAuditor:
    r"""
    FASE 1: sanea, certifica rango completo y ortogonalidad de la MIC.

    Morfismo compuesto:

    \[
    \mathrm{ObserveMic}
    =\mathrm{Orthogonality}\circ\mathrm{Rank}\circ\mathrm{SVD}
    \circ\mathrm{Gershgorin}\circ\mathrm{Finite}\circ\mathrm{Shape}.
    \]
    """

    # ── FASE 1.1 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_validate_input_shape(mic_matrix: RealMatrix) -> int:
        r"""
        FASE 1.1 — Certificación de forma: \(M\in\mathrm{Mat}_n(\mathbb{R})\).

        Raises:
            ShapeMismatchError: Si M no es 2D, cuadrada o de dimensión ≥ 1.
        """
        if not isinstance(mic_matrix, np.ndarray) or mic_matrix.ndim != 2:
            raise ShapeMismatchError(
                "La matriz MIC debe ser un ndarray estrictamente bidimensional."
            )
        n, m = mic_matrix.shape
        if n != m:
            raise ShapeMismatchError(
                f"La matriz MIC debe ser cuadrada; forma recibida={mic_matrix.shape}."
            )
        if n < 1:
            raise ShapeMismatchError("La matriz MIC no puede tener dimensión nula.")
        return int(n)

    # ── FASE 1.2 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_validate_finite(mic_matrix: RealMatrix) -> bool:
        r"""
        FASE 1.2 — Saneamiento IEEE-754: \(M\) sin NaN/Inf.

        Raises:
            NonFiniteInputError: Ante cualquier entrada no finita.
        """
        if not np.all(np.isfinite(mic_matrix)):
            raise NonFiniteInputError(
                "La MIC contiene singularidades numéricas (NaN/Inf)."
            )
        return True

    # ── FASE 1.3 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_gershgorin_disc_radius(mic_matrix: RealMatrix) -> float:
        r"""
        FASE 1.3 — Cota espectral de Gershgorin (teoría espectral de grafos).

        Interpretando la MIC como matriz de adyacencia ponderada de un grafo
        dirigido de interacciones, por el Teorema del Círculo de Gershgorin:

        \[
        \forall\lambda\in\sigma(M)\ \exists i:\
        |\lambda-M_{ii}|\le R_i=\sum_{j\neq i}|M_{ij}|.
        \]

        Cota \(O(n^2)\) de bajo costo, previa a la SVD \(O(n^3)\), útil como
        chequeo defensivo temprano de estabilidad del operador de interacción.
        """
        diag = np.abs(np.diag(mic_matrix))
        off_diag_sum = np.sum(np.abs(mic_matrix), axis=1) - diag
        return float(np.max(diag + off_diag_sum))

    # ── FASE 1.4 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_spectral_radius(mic_matrix: RealMatrix) -> float:
        r"""
        FASE 1.4 — Radio espectral \(\rho(M)=\max_i|\lambda_i(M)|\).

        Análogo de circuitos: para un operador de transición/interacción,
        \(\rho(M)\le 1\) es condición necesaria de estabilidad asintótica
        bajo iteración (contractividad de Banach en norma espectral).
        """
        eigvals = la.eigvals(mic_matrix)
        return float(np.max(np.abs(eigvals)))

    # ── FASE 1.5 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_singular_value_decomposition(
        mic_matrix: RealMatrix,
    ) -> Tuple[RealMatrix, RealVector, RealMatrix]:
        r"""
        FASE 1.5 — SVD exacta: \(M=U\Sigma V^{T}\).

        Raises:
            MicRankDeficiencyError: Si la descomposición diverge (LinAlgError
                reencausado con contexto de dominio).
        """
        try:
            u, s, vh = la.svd(mic_matrix)
        except la.LinAlgError as err:  # pragma: no cover — defensivo
            raise MicRankDeficiencyError(
                f"La descomposición SVD de la MIC no convergió: {err}"
            ) from err
        return u, s, vh

    # ── FASE 1.6 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_condition_number(singular_values: RealVector) -> float:
        r"""
        FASE 1.6 — Número de condición espectral \(\kappa(M)=\sigma_{\max}/\sigma_{\min}\).
        """
        s_max = float(singular_values[0])
        s_min = float(singular_values[-1]) if singular_values[-1] > 0 else 0.0
        return s_max / s_min if s_min > 0 else float("inf")

    # ── FASE 1.7 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_certify_full_rank(
        singular_values: RealVector,
        n: int,
        condition_number: float,
    ) -> Tuple[int, bool]:
        r"""
        FASE 1.7 — Certificación de Higham del rango numérico efectivo.

        \[
        \sigma_i>\tau_{\mathrm{MIC}}=n\cdot\sigma_{\max}\cdot\varepsilon_{\mathrm{mach}}
        \;\Longrightarrow\;\mathrm{rank}_{\mathrm{eff}}=\bigl|\{i:\sigma_i>\tau\}\bigr|.
        \]

        Raises:
            MicRankDeficiencyError: Si el rango efectivo < n (hard-gate).
        """
        s_max = float(singular_values[0])
        rank_tol = n * s_max * _MACHINE_EPS
        effective_rank = int(np.sum(singular_values > rank_tol))
        is_full_rank = effective_rank == n

        if not is_full_rank:
            raise MicRankDeficiencyError(
                f"Degeneración de Base: La MIC no posee rango completo. "
                f"Rango efectivo: {effective_rank}/{n}. "
                f"Número de condición: {condition_number:.4e}."
            )
        return effective_rank, is_full_rank

    # ── FASE 1.8 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_orthogonality_residual(mic_matrix: RealMatrix, n: int) -> float:
        r"""
        FASE 1.8 — Residuo de ortogonalidad (grupo \(O(n)\)):

        \[
        \delta_{\perp}=\|MM^{T}-I_n\|_F.
        \]

        Nota de dominio: se asume \(M\in\mathrm{Mat}_n(\mathbb{R})\); para
        MIC de dominio unitario \(U(n)\) debería sustituirse \(M^T\to M^\dagger\).
        """
        gram = mic_matrix @ mic_matrix.T
        return float(la.norm(gram - np.eye(n), ord="fro"))

    # ── FASE 1.Ω · composición terminal Observe ───────────────────────────
    @staticmethod
    def audit_mic_rank(
        mic_matrix: RealMatrix,
        ortho_tolerance: float = 1e-8,
        strict_orthogonality: bool = False,
    ) -> MicRankCertificate:
        r"""
        FASE 1.Ω — Composición terminal de Observación de la MIC.

        Valida que la MIC posea rango numérico completo y, opcionalmente,
        que sea una base ortonormal estricta de \(O(n)\).

        **Contrato F1 → F2**: el certificado retornado no transporta estado
        mutable; FASE 2 opera sobre la MAC de forma independiente (los
        espacios discreto y continuo se auditan en paralelo, no en cadena
        de datos, a diferencia del pipeline Kraus→Choi del validador CPTP).

        Args:
            mic_matrix: Matriz de interacción central M.
            ortho_tolerance: Desviación máxima tolerada para ortogonalidad.
            strict_orthogonality: Si True, eleva ``MicOrthogonalityBreachError``
                cuando \(\delta_\perp>\) ``ortho_tolerance`` (hard-gate).
                Por defecto es un invariante blando (solo se reporta).

        Returns:
            MicRankCertificate: El certificado de integridad de la MIC.

        Raises:
            ShapeMismatchError, NonFiniteInputError, MicRankDeficiencyError,
            MicOrthogonalityBreachError (solo si strict_orthogonality=True).
        """
        n = Phase1_MicAuditor._phase1_validate_input_shape(mic_matrix)
        Phase1_MicAuditor._phase1_validate_finite(mic_matrix)

        gershgorin_bound = Phase1_MicAuditor._phase1_gershgorin_disc_radius(mic_matrix)
        spectral_radius = Phase1_MicAuditor._phase1_spectral_radius(mic_matrix)

        _u, s, _vh = Phase1_MicAuditor._phase1_singular_value_decomposition(mic_matrix)
        cond_num = Phase1_MicAuditor._phase1_condition_number(s)
        effective_rank, is_full_rank = Phase1_MicAuditor._phase1_certify_full_rank(
            s, n, cond_num
        )

        ortho_dev = Phase1_MicAuditor._phase1_orthogonality_residual(mic_matrix, n)
        is_orthogonal = ortho_dev <= ortho_tolerance

        if strict_orthogonality and not is_orthogonal:
            raise MicOrthogonalityBreachError(
                f"Ruptura de Ortogonalidad estricta: δ_⊥={ortho_dev:.4e} > "
                f"{ortho_tolerance:.4e}."
            )

        logger.debug(
            "FASE1.Ω MIC: n=%d, rank=%d, κ=%.4e, δ_⊥=%.4e, ρ(M)=%.4e, Gershgorin=%.4e",
            n, effective_rank, cond_num, ortho_dev, spectral_radius, gershgorin_bound,
        )

        return MicRankCertificate(
            matrix_shape=(n, n),
            effective_rank=effective_rank,
            condition_number=cond_num,
            is_full_rank=is_full_rank,
            orthogonality_deviation=ortho_dev,
            is_orthogonal=is_orthogonal,
            is_finite=True,
            rank_deficiency_margin=n - effective_rank,
            spectral_radius=spectral_radius,
            gershgorin_bound=gershgorin_bound,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2 — CERTIFICACIÓN CUÁNTICA DE LA MAC (Orient)
# Objetos: ρ ∈ Mat_d(ℂ), σ(ρ), Tr(ρ), Tr(ρ²), S(ρ)
# ═══════════════════════════════════════════════════════════════════════════════
class Phase2_MacCertifier(Phase1_MicAuditor):
    r"""
    FASE 2: certifica que la MAC satisfaga los postulados de Dirac-von Neumann.

    Morfismo compuesto:

    \[
    \mathrm{OrientMac}
    =\mathrm{Entropy}\circ\mathrm{Purity}\circ\mathrm{Positivity}
    \circ\mathrm{Trace}\circ\mathrm{Hermiticity}\circ\mathrm{Finite}\circ\mathrm{Shape}.
    \]
    """

    # ── FASE 2.1 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_validate_input_shape(mac_density: ComplexMatrix) -> int:
        r"""
        FASE 2.1 — Certificación de forma: \(\rho\in\mathrm{Mat}_d(\mathbb{C})\).

        Raises:
            ShapeMismatchError: Si ρ no es 2D, cuadrada o de dimensión ≥ 1.
        """
        if not isinstance(mac_density, np.ndarray) or mac_density.ndim != 2:
            raise ShapeMismatchError(
                "El operador densidad de la MAC debe ser ndarray bidimensional."
            )
        d, e = mac_density.shape
        if d != e:
            raise ShapeMismatchError(
                f"El operador densidad debe ser cuadrado; forma recibida={mac_density.shape}."
            )
        if d < 1:
            raise ShapeMismatchError("El operador densidad no puede tener dimensión nula.")
        return int(d)

    # ── FASE 2.2 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_validate_finite(mac_density: ComplexMatrix) -> bool:
        r"""
        FASE 2.2 — Saneamiento IEEE-754: ρ sin NaN/Inf.

        Raises:
            NonFiniteInputError: Ante cualquier entrada no finita.
        """
        if not np.all(np.isfinite(mac_density)):
            raise NonFiniteInputError(
                "La MAC contiene singularidades numéricas (NaN/Inf)."
            )
        return True

    # ── FASE 2.3 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_hermiticity_residual(mac_density: ComplexMatrix) -> float:
        r"""
        FASE 2.3 — Residuo hermítico: \(\|\rho-\rho^\dagger\|_F\).
        """
        return float(la.norm(mac_density - mac_density.conj().T, ord="fro"))

    # ── FASE 2.4 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_certify_hermiticity(residual: float, tolerance: float) -> bool:
        r"""
        FASE 2.4 — Certificación C*: \(\rho=\rho^\dagger\).

        Raises:
            MacHermiticityViolation: Si el residuo excede la tolerancia.
        """
        is_hermitian = residual <= tolerance
        if not is_hermitian:
            raise MacHermiticityViolation(
                f"Ruptura de Hermiticidad: La MAC viola el postulado autoadjunto. "
                f"Residuo: {residual:.4e} > {tolerance:.4e}."
            )
        return is_hermitian

    # ── FASE 2.5 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_weyl_symmetrize(mac_density: ComplexMatrix) -> ComplexMatrix:
        r"""
        FASE 2.5 — Proyección de Weyl al subespacio autoadjunto:

        \[
        \rho_{\mathrm{sym}}=\tfrac{1}{2}(\rho+\rho^\dagger).
        \]

        Garantiza que la diagonalización posterior (``eigvalsh``) reciba
        una entrada exactamente hermítica, evitando inconsistencias
        numéricas silenciosas de LAPACK ante defectos ínfimos.
        """
        return 0.5 * (mac_density + mac_density.conj().T)

    # ── FASE 2.6 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_trace_value(mac_density_sym: ComplexMatrix) -> float:
        r"""
        FASE 2.6 — Traza real: \(\operatorname{Tr}(\rho)\).
        """
        return float(np.real(np.trace(mac_density_sym)))

    # ── FASE 2.7 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_certify_trace_normalization(trace_val: float, tolerance: float) -> bool:
        r"""
        FASE 2.7 — Conservación de probabilidad: \(\operatorname{Tr}(\rho)=1\).

        Raises:
            MacTraceAnomalyError: Si |Tr(ρ) − 1| > tolerancia.
        """
        is_trace_ok = abs(trace_val - 1.0) <= tolerance
        if not is_trace_ok:
            raise MacTraceAnomalyError(
                f"Anomalía de Traza: Pérdida de probabilidad en la MAC. "
                f"Tr(ρ) = {trace_val:.6f} ≠ 1.0 (tol={tolerance:.4e})."
            )
        return is_trace_ok

    # ── FASE 2.8 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_spectral_positivity(
        mac_density_sym: ComplexMatrix,
    ) -> Tuple[RealVector, float, bool]:
        r"""
        FASE 2.8 — Positividad espectral: \(\rho\succeq 0\iff\lambda_{\min}\ge0\).

        Raises:
            MacPositivitySpectralError: Si λ_min < floor PSD.
        """
        eigvals = la.eigvalsh(mac_density_sym)
        min_eig = float(np.min(eigvals))
        is_psd = min_eig >= _SPECTRAL_PSD_FLOOR
        if not is_psd:
            raise MacPositivitySpectralError(
                f"Violación de Positividad: Autovalores negativos detectados en la MAC. "
                f"λ_min = {min_eig:.4e} < floor={_SPECTRAL_PSD_FLOOR:.4e}."
            )
        return eigvals, min_eig, is_psd

    # ── FASE 2.9 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_quantum_purity(
        mac_density_sym: ComplexMatrix,
        dimension_d: int,
        tolerance: float,
    ) -> Tuple[float, bool]:
        r"""
        FASE 2.9 — Pureza cuántica y su cota teórica.

        \[
        \gamma=\operatorname{Tr}(\rho^2)\in\Bigl[\tfrac{1}{d},1\Bigr],
        \]

        con \(\gamma=1/d\) para el estado maximalmente mixto y \(\gamma=1\)
        para estados puros. Se certifica la cota (invariante blando).
        """
        purity = float(np.real(np.trace(mac_density_sym @ mac_density_sym)))
        lower_bound = 1.0 / dimension_d
        is_bounded = (lower_bound - tolerance) <= purity <= (1.0 + tolerance)
        return purity, is_bounded

    # ── FASE 2.10 ─────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_von_neumann_entropy(eigvals: RealVector) -> float:
        r"""
        FASE 2.10 — Entropía de von Neumann:

        \[
        S(\rho)=-\sum_i\lambda_i\log\lambda_i,\qquad 0\log 0:=0.
        \]

        Autovalores negativos ínfimos (redondeo FPU) se recortan a 0 antes
        del logaritmo para evitar `NaN` por dominio.
        """
        clipped = np.clip(eigvals, 0.0, None)
        support = clipped[clipped > _EPS_ENTROPY_FLOOR]
        if support.size == 0:
            return 0.0
        return float(-np.sum(support * np.log(support)))

    # ── FASE 2.Ω · composición terminal Orient ────────────────────────────
    @staticmethod
    def audit_mac_density(
        mac_density: ComplexMatrix,
        tolerance: float = _DEFAULT_TOL,
    ) -> MacHermiticityCertificate:
        r"""
        FASE 2.Ω — Composición terminal de certificación cuántica de la MAC.

        Valida en cadena:
        \(\rho=\rho^\dagger\;\land\;\operatorname{Tr}(\rho)=1\;\land\;\rho\succeq0\),
        y reporta pureza, cota de pureza y entropía de von Neumann como
        telemetría espectral de segundo orden.

        Raises:
            ShapeMismatchError, NonFiniteInputError, MacHermiticityViolation,
            MacTraceAnomalyError, MacPositivitySpectralError.
        """
        d = Phase2_MacCertifier._phase2_validate_input_shape(mac_density)
        Phase2_MacCertifier._phase2_validate_finite(mac_density)

        hermitian_residual = Phase2_MacCertifier._phase2_hermiticity_residual(mac_density)
        is_hermitian = Phase2_MacCertifier._phase2_certify_hermiticity(
            hermitian_residual, tolerance
        )

        mac_sym = Phase2_MacCertifier._phase2_weyl_symmetrize(mac_density)

        trace_val = Phase2_MacCertifier._phase2_trace_value(mac_sym)
        is_trace_ok = Phase2_MacCertifier._phase2_certify_trace_normalization(
            trace_val, tolerance
        )

        eigvals, min_eig, is_psd = Phase2_MacCertifier._phase2_spectral_positivity(mac_sym)

        purity, is_purity_bounded = Phase2_MacCertifier._phase2_quantum_purity(
            mac_sym, d, tolerance
        )
        entropy = Phase2_MacCertifier._phase2_von_neumann_entropy(eigvals)

        logger.debug(
            "FASE2.Ω MAC: d=%d, herm_res=%.4e, Tr=%.6f, λ_min=%.4e, γ=%.6f, S=%.6f",
            d, hermitian_residual, trace_val, min_eig, purity, entropy,
        )

        return MacHermiticityCertificate(
            is_hermitian=is_hermitian,
            hermitician_residual=hermitian_residual,
            trace_value=trace_val,
            is_trace_normalized=is_trace_ok,
            minimum_eigenvalue=min_eig,
            is_positive_semidefinite=is_psd,
            quantum_purity=purity,
            hilbert_dimension=d,
            is_purity_bounded=is_purity_bounded,
            von_neumann_entropy=entropy,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 3 — SUTURA Y CERTIFICACIÓN DE LA ADJUNCIÓN DE GALOIS (Decide)
# Objetos: X∈ℝⁿ, Y∈ℝᵐ, F:ℝⁿ→ℝᵐ, G:ℝᵐ→ℝⁿ, η_X = X - G(F(X))
# ═══════════════════════════════════════════════════════════════════════════════
class Phase3_GaloisSuturator(Phase2_MacCertifier):
    r"""
    FASE 3: evalúa la consistencia cuantitativa de la adjunción \(F\dashv G\)
    sobre el Pasaporte, relajando la identidad triangular categórica exacta
    \(\eta_X=0\) a una desigualdad de Lipschitz tolerante al ruido retórico.

    Morfismo compuesto:

    \[
    \mathrm{DecideGalois}
    =\mathrm{Fidelity}\circ\mathrm{Lipschitz}\circ\mathrm{Unit}
    \circ\mathrm{Finite}\circ\mathrm{Shape}\circ\mathrm{Params}.
    \]
    """

    # ── FASE 3.1 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_validate_lipschitz_bound(lipschitz_bound: float, tolerance: float) -> None:
        r"""
        FASE 3.1 — Validación de parámetros escalares \(L_{\max}\ge0\), ε≥0, finitos.

        Raises:
            LipschitzParameterError: Si L_max o ε son negativos, NaN o Inf.
        """
        if not math.isfinite(lipschitz_bound) or lipschitz_bound < 0.0:
            raise LipschitzParameterError(
                f"lipschitz_bound debe ser real ≥ 0 y finito; recibido {lipschitz_bound}."
            )
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise LipschitzParameterError(
                f"tolerance debe ser real ≥ 0 y finito; recibido {tolerance}."
            )

    # ── FASE 3.2 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_validate_dimensional_compatibility(
        state_tactic_X: RealVector,
        state_wisdom_Y: RealVector,
        functor_F_projection: RealMatrix,
        functor_G_reconstruction: RealMatrix,
    ) -> None:
        r"""
        FASE 3.2 — Certificación de compatibilidad dimensional del par adjunto.

        Exige, para \(F:\mathbb{R}^n\to\mathbb{R}^m\) y \(G:\mathbb{R}^m\to\mathbb{R}^n\):

        \[
        X\in\mathbb{R}^n,\ Y\in\mathbb{R}^m,\
        F\in\mathrm{Mat}_{m\times n},\ G\in\mathrm{Mat}_{n\times m}.
        \]

        Raises:
            ShapeMismatchError: Ante cualquier incompatibilidad de forma.
        """
        for name, arr, ndim in (
            ("state_tactic_X", state_tactic_X, 1),
            ("state_wisdom_Y", state_wisdom_Y, 1),
            ("functor_F_projection", functor_F_projection, 2),
            ("functor_G_reconstruction", functor_G_reconstruction, 2),
        ):
            if not isinstance(arr, np.ndarray) or arr.ndim != ndim:
                raise ShapeMismatchError(
                    f"{name} debe ser ndarray de ndim={ndim}; recibido "
                    f"ndim={getattr(arr, 'ndim', None)}."
                )

        n = state_tactic_X.shape[0]
        m = state_wisdom_Y.shape[0]

        if functor_F_projection.shape != (m, n):
            raise ShapeMismatchError(
                f"functor_F_projection debe tener forma ({m},{n}); "
                f"recibido {functor_F_projection.shape}."
            )
        if functor_G_reconstruction.shape != (n, m):
            raise ShapeMismatchError(
                f"functor_G_reconstruction debe tener forma ({n},{m}); "
                f"recibido {functor_G_reconstruction.shape}."
            )

    # ── FASE 3.3 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_validate_finite(*arrays: NDArray) -> None:
        r"""
        FASE 3.3 — Saneamiento IEEE-754 conjunto de \((X,Y,F,G)\).

        Raises:
            NonFiniteInputError: Ante cualquier entrada no finita.
        """
        for arr in arrays:
            if not np.all(np.isfinite(arr)):
                raise NonFiniteInputError(
                    "Se detectó una singularidad numérica (NaN/Inf) en el "
                    "par de estados/funtores de la adjunción de Galois."
                )

    # ── FASE 3.4 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_reconstruct_state(
        state_tactic_X: RealVector,
        functor_F_projection: RealMatrix,
        functor_G_reconstruction: RealMatrix,
    ) -> RealVector:
        r"""
        FASE 3.4 — Reconstrucción \(G(F(X))\) vía composición de proyectores.
        """
        return functor_G_reconstruction @ (functor_F_projection @ state_tactic_X)

    # ── FASE 3.5 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_unit_residual(
        state_tactic_X: RealVector,
        reconstructed_X: RealVector,
    ) -> float:
        r"""
        FASE 3.5 — Norma de la *unidad* de la adjunción:

        \[
        \|\eta_X\|_2=\|X-G(F(X))\|_2.
        \]

        En una adjunción exacta \(\eta_X\equiv0\); el residuo mide la
        entropía retórica introducida por el ciclo elevación/olvido.
        """
        return float(la.norm(state_tactic_X - reconstructed_X, ord=2))

    # ── FASE 3.6 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_target_discrepancy(
        functor_F_projection: RealMatrix,
        state_tactic_X: RealVector,
        state_wisdom_Y: RealVector,
    ) -> float:
        r"""
        FASE 3.6 — Discrepancia del contra-dominio: \(\|F(X)-Y\|_2\).
        """
        return float(la.norm(functor_F_projection @ state_tactic_X - state_wisdom_Y, ord=2))

    # ── FASE 3.7 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_certify_lipschitz_condition(
        reconstruction_residual: float,
        target_diff: float,
        lipschitz_bound: float,
        tolerance: float,
    ) -> bool:
        r"""
        FASE 3.7 — Certificación cuantitativa de la adjunción (relajación
        métrica de la identidad triangular \(\varepsilon\circ F=\mathrm{id}_F\)):

        \[
        \|\eta_X\|_2\le L_{\max}\cdot\|F(X)-Y\|_2+\varepsilon.
        \]

        Raises:
            GaloisAdjunctionBreachError: Si la desigualdad se viola.
        """
        violated = reconstruction_residual > lipschitz_bound * target_diff + tolerance
        if violated:
            raise GaloisAdjunctionBreachError(
                f"Ruptura de Adjunción: El funtor de de-compresión introdujo "
                f"entropía retórica. Residuo de Reconstrucción: "
                f"{reconstruction_residual:.4e} > L·{target_diff:.4e} + {tolerance:.4e}."
            )
        return not violated

    # ── FASE 3.8 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_reconstruction_fidelity(
        state_tactic_X: RealVector,
        reconstructed_X: RealVector,
    ) -> Tuple[float, bool]:
        r"""
        FASE 3.8 — Fidelidad de reconstrucción (análogo al solapamiento
        cuántico \(|\langle\psi|\varphi\rangle|\)):

        \[
        F(X,\hat X)=\Bigl|\frac{\langle X,\hat X\rangle}{\|X\|_2\|\hat X\|_2}\Bigr|
        \in[0,1].
        \]

        Se toma valor absoluto (la orientación del vector no debe penalizar
        la fidelidad) y se recorta a \([0,1]\) por seguridad numérica ante
        overshoot de punto flotante.
        """
        norm_x = la.norm(state_tactic_X, ord=2)
        norm_rec_x = la.norm(reconstructed_X, ord=2)

        if norm_x <= _MACHINE_EPS or norm_rec_x <= _MACHINE_EPS:
            return 0.0, False

        cos_sim = float(np.dot(state_tactic_X, reconstructed_X) / (norm_x * norm_rec_x))
        raw_fidelity = abs(cos_sim)
        clipped = raw_fidelity > 1.0 + 1e-9
        fidelity = float(min(1.0, max(0.0, raw_fidelity)))
        return fidelity, clipped

    # ── FASE 3.Ω · composición terminal Decide ────────────────────────────
    @staticmethod
    def audit_galois_adjunction(
        state_tactic_X: RealVector,
        state_wisdom_Y: RealVector,
        functor_F_projection: RealMatrix,
        functor_G_reconstruction: RealMatrix,
        lipschitz_bound: float = 1.5,
        tolerance: float = 1e-7,
    ) -> GaloisAdjunctionCertificate:
        r"""
        FASE 3.Ω — Composición terminal de la certificación de adjunción.

        Certifica la reversibilidad tolerante de la traducción de la Malla
        mediante la desigualdad de Lipschitz cuantitativa sobre la unidad
        de la adjunción \(\eta_X\), y reporta fidelidad de reconstrucción.

        Raises:
            LipschitzParameterError, ShapeMismatchError, NonFiniteInputError,
            GaloisAdjunctionBreachError.
        """
        Phase3_GaloisSuturator._phase3_validate_lipschitz_bound(lipschitz_bound, tolerance)
        Phase3_GaloisSuturator._phase3_validate_dimensional_compatibility(
            state_tactic_X, state_wisdom_Y, functor_F_projection, functor_G_reconstruction
        )
        Phase3_GaloisSuturator._phase3_validate_finite(
            state_tactic_X, state_wisdom_Y, functor_F_projection, functor_G_reconstruction
        )

        reconstructed_X = Phase3_GaloisSuturator._phase3_reconstruct_state(
            state_tactic_X, functor_F_projection, functor_G_reconstruction
        )
        reconstruction_residual = Phase3_GaloisSuturator._phase3_unit_residual(
            state_tactic_X, reconstructed_X
        )
        target_diff = Phase3_GaloisSuturator._phase3_target_discrepancy(
            functor_F_projection, state_tactic_X, state_wisdom_Y
        )

        is_secured = Phase3_GaloisSuturator._phase3_certify_lipschitz_condition(
            reconstruction_residual, target_diff, lipschitz_bound, tolerance
        )

        fidelity, was_clipped = Phase3_GaloisSuturator._phase3_reconstruction_fidelity(
            state_tactic_X, reconstructed_X
        )

        logger.debug(
            "FASE3.Ω Galois: ‖η_X‖=%.4e, ‖F(X)-Y‖=%.4e, L=%.3f, fidelidad=%.6f",
            reconstruction_residual, target_diff, lipschitz_bound, fidelity,
        )

        return GaloisAdjunctionCertificate(
            adjunction_residual=reconstruction_residual,
            is_adjunction_secured=is_secured,
            lipschitz_bound=lipschitz_bound,
            reconstruction_fidelity=fidelity,
            target_diff=target_diff,
            fidelity_was_clipped=was_clipped,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SUTURADOR MÓRFICO SUPREMO (ORQUESTADOR CATEGORIAL)
# Observe (F1) ∥ Orient (F2) ⟶ Decide (F3) ⟶ Seal / Veto
# ═══════════════════════════════════════════════════════════════════════════════
class MorphicSuturator(Morphism, Phase3_GaloisSuturator):
    r"""
    Morfismo supremo encargado de unificar los espacios discretos y continuos
    de la Malla. Verifica de manera robusta la MIC, la MAC y la adjunción de
    Galois, sellando un certificado inmutable de coherencia ciber-física.

    .. code-block:: text

        ┌────────────────────────────────────────────────────────────┐
        │ FASE 1  Observe / MIC        │ FASE 2  Orient / MAC         │
        │   1.1 validate_input_shape   │   2.1 validate_input_shape   │
        │   1.2 validate_finite        │   2.2 validate_finite        │
        │   1.3 gershgorin_disc_radius │   2.3 hermiticity_residual   │
        │   1.4 spectral_radius        │   2.4 certify_hermiticity    │
        │   1.5 svd                    │   2.5 weyl_symmetrize        │
        │   1.6 condition_number       │   2.6 trace_value            │
        │   1.7 certify_full_rank      │   2.7 certify_trace_norm     │
        │   1.8 orthogonality_residual │   2.8 spectral_positivity    │
        │   1.Ω audit_mic_rank         │   2.9 quantum_purity         │
        │                              │   2.10 von_neumann_entropy   │
        │                              │   2.Ω audit_mac_density      │
        ├──────────────────────────────┴──────────────────────────────┤
        │ FASE 3  Decide / Galois                                     │
        │   3.1 validate_lipschitz_bound   3.5 unit_residual           │
        │   3.2 validate_dim_compat        3.6 target_discrepancy      │
        │   3.3 validate_finite            3.7 certify_lipschitz       │
        │   3.4 reconstruct_state          3.8 reconstruction_fidelity │
        │   3.Ω audit_galois_adjunction                                │
        ├───────────────────────────────────────────────────────────── ┤
        │ SEAL / VETO                                                  │
        │   Ω.1 coherence_conjunction   Ω.2 seal_state   Ω.Veto reraise│
        └───────────────────────────────────────────────────────────── ┘
    """

    def __init__(self, target_stratum: Stratum = Stratum.WISDOM) -> None:
        """Inicializa al orquestador en el estrato supremo."""
        super().__init__()
        self._target_stratum: Stratum = target_stratum

    # ── Ω.0 · validación de tolerancia global ───────────────────────────────
    @staticmethod
    def _validate_global_tolerance(tolerance: float) -> None:
        r"""
        FASE Ω.0 — Certificación del ε de Wilkinson global de la sutura.

        Raises:
            MorphicSuturatorError: Si tolerance no es real ≥ 0 y finito.
        """
        if not math.isfinite(tolerance) or tolerance < 0.0:
            raise MorphicSuturatorError(
                f"tolerance debe ser real ≥ 0 y finito; recibido {tolerance}."
            )

    # ── Ω.1 · conjunción de hard-gates ───────────────────────────────────────
    @staticmethod
    def _seal_coherence_conjunction(
        mic_audit: MicRankCertificate,
        mac_audit: MacHermiticityCertificate,
        galois_audit: GaloisAdjunctionCertificate,
    ) -> BoolLattice:
        r"""
        FASE Ω.1 — Clasificador de subobjetos \(\chi_{\mathrm{coherent}}\in\Omega\).

        \[
        \chi=\chi_{\mathrm{full\ rank}}\wedge\chi_{\mathrm{PSD}}\wedge\chi_{\mathrm{adjunction}}.
        \]

        Ortogonalidad de la MIC y cota de pureza de la MAC son invariantes
        blandos y no participan de χ (análogo a PPT/unitalidad en el
        validador CPTP).
        """
        return bool(
            mic_audit.is_full_rank
            and mac_audit.is_positive_semidefinite
            and galois_audit.is_adjunction_secured
        )

    # ── Ω.2 · sellado del certificado ────────────────────────────────────────
    def _seal_suturation_state(
        self,
        mic_audit: MicRankCertificate,
        mac_audit: MacHermiticityCertificate,
        galois_audit: GaloisAdjunctionCertificate,
        is_coherent: BoolLattice,
        tolerance: float,
    ) -> MorphicSuturationState:
        """FASE Ω.2 — Sellado del objeto terminal frozen del funtor de sutura."""
        timestamp_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        stratum_name = (
            self._target_stratum.name
            if hasattr(self._target_stratum, "name")
            else str(self._target_stratum)
        )
        return MorphicSuturationState(
            mic_audit=mic_audit,
            mac_audit=mac_audit,
            galois_audit=galois_audit,
            is_sutured_coherent=is_coherent,
            timestamp_utc=timestamp_utc,
            wilkinson_tolerance=tolerance,
            stratum=stratum_name,
        )

    # ── Ω.Veto · telemetría de colapso ───────────────────────────────────────
    @staticmethod
    def _veto_log_and_reraise(err: MorphicSuturatorError) -> None:
        """FASE Ω.Veto — Log CRITICAL y re-propagación hacia Crowbar."""
        logger.critical(
            "¡VETO DE SUTURA! Se ha detectado una ruptura de simetría en la "
            "aduana de control: %s",
            str(err),
        )
        raise err

    # ── Compositor público OODA ──────────────────────────────────────────────
    def execute_suturation(
        self,
        mic_matrix: RealMatrix,
        mac_density: ComplexMatrix,
        state_tactic_X: RealVector,
        state_wisdom_Y: RealVector,
        functor_F: RealMatrix,
        functor_G: RealMatrix,
        lipschitz_limit: float = 1.5,
        tolerance: float = _DEFAULT_TOL,
    ) -> MorphicSuturationState:
        r"""
        Resuelve la consistencia global del espacio de fase en un único acto
        de verificación categorial.

        Args:
            mic_matrix: Matriz de Interacción Central M.
            mac_density: Operador densidad de la MAC \(\rho\).
            state_tactic_X: Estado del presupuesto discreto X.
            state_wisdom_Y: Estado de opinión de la Sabiduría Y.
            functor_F: Operador proyección del funtor F.
            functor_G: Operador proyección del funtor G.
            lipschitz_limit: Cota de Lipschitz máxima permitida \(L_{\max}\).
            tolerance: Épsilon de Wilkinson base para auditoría espectral.

        Returns:
            MorphicSuturationState: Certificado inmutable de coherencia.

        Raises:
            MorphicSuturatorError: Si falla cualquiera de las 3 fases de la
                aduana (re-propagada tras log CRITICAL).
        """
        self._validate_global_tolerance(tolerance)

        try:
            # ── FASE 1 · Observe / MIC ──────────────────────────────────────
            mic_audit = self.audit_mic_rank(mic_matrix, ortho_tolerance=tolerance * 100)

            # ── FASE 2 · Orient / MAC ────────────────────────────────────────
            mac_audit = self.audit_mac_density(mac_density, tolerance=tolerance)

            # ── FASE 3 · Decide / Galois ─────────────────────────────────────
            galois_audit = self.audit_galois_adjunction(
                state_tactic_X=state_tactic_X,
                state_wisdom_Y=state_wisdom_Y,
                functor_F_projection=functor_F,
                functor_G_reconstruction=functor_G,
                lipschitz_bound=lipschitz_limit,
                tolerance=tolerance * 100,
            )

            # ── SEAL ──────────────────────────────────────────────────────────
            is_sutured_coherent = self._seal_coherence_conjunction(
                mic_audit, mac_audit, galois_audit
            )
            state = self._seal_suturation_state(
                mic_audit, mac_audit, galois_audit, is_sutured_coherent, tolerance
            )

            logger.info(
                "Sutura mórfica ciber-física completada con éxito. "
                "Pureza MAC=%.4f | S(ρ)=%.4f | Residuo Adjunción=%.4e | "
                "Fidelidad=%.4f | coherente=%s",
                mac_audit.quantum_purity,
                mac_audit.von_neumann_entropy,
                galois_audit.adjunction_residual,
                galois_audit.reconstruction_fidelity,
                str(is_sutured_coherent),
            )
            return state

        except MorphicSuturatorError as err:
            self._veto_log_and_reraise(err)
            raise  # unreachable; satisface type-checkers
        except la.LinAlgError as err:  # pragma: no cover — defensivo
            wrapped = MorphicSuturatorError(
                f"Colapso de álgebra lineal no categorizado durante la sutura: {err}"
            )
            self._veto_log_and_reraise(wrapped)
            raise  # unreachable

    # ─────────────────────────────────────────────────────────────────────
    # Fábricas de referencia (calibración / tests del suturador)
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def identity_mic(n: int) -> RealMatrix:
        """MIC de referencia: base canónica ortonormal \(I_n\in O(n)\)."""
        return np.eye(n, dtype=np.float64)

    @staticmethod
    def rotation_mic(n: int, theta: float) -> RealMatrix:
        r"""
        MIC de referencia rotacional (bloque 2×2, resto identidad):
        rango completo y ortogonal por construcción para cualquier θ.
        """
        m = np.eye(n, dtype=np.float64)
        if n >= 2:
            c, s = math.cos(theta), math.sin(theta)
            m[0, 0], m[0, 1] = c, -s
            m[1, 0], m[1, 1] = s, c
        return m

    @staticmethod
    def maximally_mixed_mac(dimension_d: int) -> ComplexMatrix:
        r"""MAC de referencia: estado maximalmente mixto \(I_d/d\)."""
        return (np.eye(dimension_d, dtype=np.complex128) / dimension_d)

    @staticmethod
    def pure_state_mac(psi: NDArray[np.complex128]) -> ComplexMatrix:
        r"""
        MAC de referencia: estado puro \(\rho=|\psi\rangle\langle\psi|\)
        a partir de un vector normalizado \(\psi\in\mathbb{C}^d\).
        """
        norm = float(la.norm(psi, ord=2))
        if norm <= _MACHINE_EPS:
            raise ShapeMismatchError("psi no puede ser el vector nulo.")
        psi_normalized = (psi / norm).astype(np.complex128)
        return np.outer(psi_normalized, psi_normalized.conj())

    @staticmethod
    def identity_adjunction_pair(n: int) -> Tuple[RealMatrix, RealMatrix]:
        r"""
        Par funtorial trivial \((F,G)=(I_n,I_n)\): la adjunción se reduce
        a la identidad, con residuo de unidad \(\eta_X\equiv0\) exacto.
        """
        identity = np.eye(n, dtype=np.float64)
        return identity, identity


# ═══════════════════════════════════════════════════════════════════════════════
# Exportación canónica del módulo
# ═══════════════════════════════════════════════════════════════════════════════
__all__ = [
    # Excepciones
    "MorphicSuturatorError",
    "NonFiniteInputError",
    "ShapeMismatchError",
    "MicRankDeficiencyError",
    "MicOrthogonalityBreachError",
    "MacDensityAnomalyError",
    "MacHermiticityViolation",
    "MacTraceAnomalyError",
    "MacPositivitySpectralError",
    "LipschitzParameterError",
    "GaloisAdjunctionBreachError",
    # DTOs
    "MicRankCertificate",
    "MacHermiticityCertificate",
    "GaloisAdjunctionCertificate",
    "MorphicSuturationState",
    # Fases
    "Phase1_MicAuditor",
    "Phase2_MacCertifier",
    "Phase3_GaloisSuturator",
    # Orquestador
    "MorphicSuturator",
    # Constantes útiles para tests
    "_DEFAULT_TOL",
    "_SPECTRAL_PSD_FLOOR",
    "_MACHINE_EPS",
    "_EPS_HERMITICITY",
    "_EPS_ORTHOGONALITY",
]