#-*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Morphic Suturator Agent (Soberano del Acoplamiento Categorial)     ║
║  Ruta   : app/agents/wisdom/morphic_suturator_agent.py                       ║
║  Versión: 2.0.0-Galois-Adjunction-OODA-Strict-FPU-Secure-Granular            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  NATURALEZA CIBER-FÍSICA Y RIGOR DOCTORAL:                                   ║
║  ──────────────────────────────────────────────────────────────────────────  ║
║  Este módulo materializa al Agente Soberano y Observador Activo que          ║
║  gobierna al morfismo de sutura 'morphic_suturator.py'.                      ║
║  Ejecuta un ciclo OODA covariante sobre el espacio de fase para acoplar      ║
║  la Matriz de Interacción Central (MIC) discreta con el espacio de           ║
║  Hilbert continuo de la Matriz Atómica de Conocimiento (MAC).                ║
║                                                                              ║
║  Arquitectura de Fases Anidadas (OODA espectral fail-secure):                ║
║    FASE 1 — Observe: saneamiento IEEE-754, estabilidad espectral de la MIC   ║
║             (SVD/condición/rango) y postulados de Dirac-von Neumann sobre    ║
║             la MAC (hermiticidad, traza, positividad, pureza, entropía).     ║
║    FASE 2 — Orient: validación de dominio escalar y residuo cuantitativo     ║
║             de la Adjunción de Galois frente a la cota de Lipschitz.         ║
║    FASE 3 — Decide + Act: clasificación en el retículo Ω₃ de tres valores,   ║
║             escalamiento por anomalías estructurales duras, activación del   ║
║             Crowbar físico y sellado del certificado de gobernanza.          ║
║                                                                              ║
║  Axioma de Consistencia de la Adjunción (Isomorfismo de Galois):             ║
║  $$\text{Hom}_{\mathcal{D}}(F(X), Y) \cong \text{Hom}_{\mathcal{C}}(X, G(Y))$$║
║                                                                              ║
║  Contrato de Seguridad (fail-secure): este agente **nunca propaga**          ║
║  excepciones al llamador — toda anomalía de dominio o colapso numérico       ║
║  colapsa determinísticamente a un veredicto ``VETOED`` sellado, salvo que    ║
║  se solicite explícitamente el modo ``raise_on_veto=True``.                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
from typing import Final, Optional, Tuple

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

    class Stratum(IntEnum):
        """Estratos de la pirámide de información DIKW."""

        PHYSICS = 1
        TACTICS = 2
        STRATEGY = 3
        WISDOM = 4


logger = logging.getLogger("MIC.Wisdom.MorphicSuturatorAgent")

# ─────────────────────────────────────────────────────────────────────────────
# Tipos canónicos
# ─────────────────────────────────────────────────────────────────────────────
RealMatrix = NDArray[np.float64]
ComplexMatrix = NDArray[np.complex128]
RealVector = NDArray[np.float64]
RealScalar = float
BoolLattice = bool

# Constantes espectrales del silicio (tolerancia de Wilkinson)
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_DEFAULT_TOL: Final[float] = 1.0e-12
_SPECTRAL_PSD_FLOOR: Final[float] = -1.0e-13
_EPS_HERMITICITY: Final[float] = 1.0e-9
_EPS_TRACE: Final[float] = 1.0e-6
_EPS_ENTROPY_FLOOR: Final[float] = 1.0e-15


# ═══════════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES (VETOS DE LA ADUANA DE SUTURA — RETÍCULO Ω₃)
# ═══════════════════════════════════════════════════════════════════════════════
class SuturatorAgentError(TopologicalInvariantError):
    """Excepción raíz del Agente Soberano de la Aduana de Sutura."""


class NonFiniteInputError(SuturatorAgentError):
    """Lanzada si cualquier tensor de entrada contiene NaN/Inf (fuga IEEE-754)."""


class ShapeMismatchError(SuturatorAgentError):
    """Lanzada ante inconsistencias dimensionales en la MIC o la MAC."""


class MicSpectralAnomalyError(SuturatorAgentError):
    """La descomposición espectral de la MIC divergió o es numéricamente inválida."""


class MacSpectralAnomalyError(SuturatorAgentError):
    """La descomposición espectral de la MAC divergió o es numéricamente inválida."""


class ScalarDomainError(SuturatorAgentError):
    """El residuo de reconstrucción o la constante de Lipschitz violan su dominio ℝ≥0."""


class ThresholdOrderingError(SuturatorAgentError):
    """coherence_threshold debe ser ≤ veto_threshold para una clasificación bien definida."""


class AdjunctionBreachVeto(SuturatorAgentError):
    """
    Detonada únicamente en modo estricto (``raise_on_veto=True``) cuando el
    veredicto legítimamente clasificado es ``VETOED``. Por defecto el veto
    se resuelve como estado sellado, no como excepción (contrato fail-secure).
    """


# ═══════════════════════════════════════════════════════════════════════════════
# RETÍCULO DE VEREDICTOS (CLASIFICADOR DE SUBOBJETOS Ω₃)
# ═══════════════════════════════════════════════════════════════════════════════
class EpistemicSuturatorVerdict(IntEnum):
    """
    Clasificador de subobjetos de tres valores en el topos de sutura.

    Valores:
        COHERENT (0): Consistencia categórica y estabilidad espectral absolutas.
        DEGRADED (1): Desviaciones menores tolerables bajo regularización.
        VETOED (2): Ruptura de simetría o alucinación severa; detonación de Crowbar.
    """

    COHERENT = 0
    DEGRADED = 1
    VETOED = 2


# ═══════════════════════════════════════════════════════════════════════════════
# DTOs INMUTABLES (Contratos entre Fases del Funtor OODA)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class Phase1SpectralObservation:
    r"""
    Artefacto terminal de la FASE 1 (Observe / auditoría espectral).

    Certifica la estabilidad de la MIC y los postulados de Dirac-von Neumann
    de la MAC. Es el *objeto inicial* de FASE 3 para el escalamiento de
    veredicto por anomalías estructurales duras.
    """
    mic_shape: Tuple[int, int]
    mic_is_finite: bool
    mic_condition_number: float
    mic_is_full_rank: bool
    mac_shape: Tuple[int, int]
    mac_is_finite: bool
    mac_is_hermitian: bool
    mac_hermiticity_residual: float
    mac_trace: float
    mac_trace_anomaly: bool
    mac_minimum_eigenvalue: float
    mac_is_psd: bool
    mac_purity: float
    mac_entropy: float


@dataclass(frozen=True, slots=True)
class Phase2AdjunctionOrientation:
    r"""
    Artefacto terminal de la FASE 2 (Orient / residuo de Galois).

    \[
    r_{\mathrm{adj}}=\max\bigl(0,\ \varepsilon_{\mathrm{rec}}-L_{\max}\bigr).
    \]
    """
    adjunction_residual: float
    reconstruction_error: float
    lipschitz_constant: float
    coherence_threshold: float
    veto_threshold: float


@dataclass(frozen=True, slots=True)
class SuturatorAgentVerdictState:
    r"""
    Certificado inmutable del veredicto final de sutura del agente soberano.

    Atributos originales (v1.0.0, preservados sin cambio de tipo/orden):
        verdict, adjunction_residual, mic_condition_number, mac_entropy,
        is_crowbar_active.

    Atributos añadidos (v2.0.0, todos con valor por defecto — retro-compatibles):
        mac_purity, mac_trace_anomaly, mic_rank_deficient, reconstruction_error,
        lipschitz_constant, coherence_threshold, veto_threshold, timestamp_utc.
    """
    verdict: EpistemicSuturatorVerdict
    adjunction_residual: float
    mic_condition_number: float
    mac_entropy: float
    is_crowbar_active: bool
    mac_purity: float = 0.0
    mac_trace_anomaly: bool = False
    mic_rank_deficient: bool = False
    reconstruction_error: float = 0.0
    lipschitz_constant: float = 0.0
    coherence_threshold: float = 0.0
    veto_threshold: float = 0.0
    timestamp_utc: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1 — OBSERVACIÓN ESPECTRAL DE LA MIC Y LA MAC (Observe)
# Objetos: M∈Mat_n(ℝ), σ(M), ρ∈Mat_d(ℂ), σ(ρ)
# ═══════════════════════════════════════════════════════════════════════════════
class Phase1_SpectralObserver:
    r"""
    FASE 1: sanea y audita la estabilidad espectral de la MIC y los
    postulados de Dirac-von Neumann de la MAC.

    Morfismo compuesto:

    \[
    \mathrm{ObserveSpectrum}
    =(\mathrm{MicRank},\,\mathrm{MacDirac})\circ(\mathrm{Finite},\,\mathrm{Shape}).
    \]
    """

    # ── FASE 1.1 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_validate_mic_shape(mic_matrix: RealMatrix) -> int:
        r"""
        FASE 1.1 — Certificación de forma: \(M\in\mathrm{Mat}_n(\mathbb{R})\).

        Raises:
            ShapeMismatchError: Si M no es 2D, cuadrada o de dimensión ≥ 1.
        """
        if not isinstance(mic_matrix, np.ndarray) or mic_matrix.ndim != 2:
            raise ShapeMismatchError(
                "La MIC debe ser un ndarray estrictamente bidimensional."
            )
        n, m = mic_matrix.shape
        if n != m or n < 1:
            raise ShapeMismatchError(
                f"La MIC debe ser cuadrada de dimensión ≥ 1; forma={mic_matrix.shape}."
            )
        return int(n)

    # ── FASE 1.2 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_validate_mic_finite(mic_matrix: RealMatrix) -> bool:
        r"""
        FASE 1.2 — Saneamiento IEEE-754 de la MIC.

        Raises:
            NonFiniteInputError: Ante NaN/Inf en la MIC.
        """
        if not np.all(np.isfinite(mic_matrix)):
            raise NonFiniteInputError("La MIC contiene singularidades numéricas (NaN/Inf).")
        return True

    # ── FASE 1.3 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_mic_singular_values(mic_matrix: RealMatrix) -> RealVector:
        r"""
        FASE 1.3 — Valores singulares \(\sigma(M)\) vía SVD sin reconstrucción.

        Raises:
            MicSpectralAnomalyError: Si la SVD no converge (LAPACK).
        """
        try:
            return la.svdvals(mic_matrix).astype(np.float64)
        except la.LinAlgError as err:
            raise MicSpectralAnomalyError(
                f"La descomposición espectral de la MIC no convergió: {err}"
            ) from err

    # ── FASE 1.4 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_mic_condition_number(singular_values: RealVector) -> float:
        r"""
        FASE 1.4 — Número de condición con suelo de Wilkinson:

        \[
        \kappa(M)=\frac{\sigma_{\max}}{\max(\sigma_{\min},\varepsilon_{\mathrm{mach}})}.
        \]
        """
        s_max = float(np.max(singular_values))
        s_min_floored = float(np.maximum(np.min(singular_values), _MACHINE_EPS))
        return s_max / s_min_floored

    # ── FASE 1.5 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_mic_rank_certificate(singular_values: RealVector, n: int) -> bool:
        r"""
        FASE 1.5 — Certificación de Higham del rango numérico efectivo (blando):

        \[
        \mathrm{rank}_{\mathrm{eff}}=\bigl|\{i:\sigma_i>n\sigma_{\max}\varepsilon_{\mathrm{mach}}\}\bigr|,
        \qquad
        \chi_{\mathrm{full\ rank}}\iff\mathrm{rank}_{\mathrm{eff}}=n.
        \]

        Invariante blando: no aborta el flujo, solo etiqueta telemetría que
        FASE 3 usará para escalar el veredicto.
        """
        s_max = float(np.max(singular_values))
        rank_tol = n * s_max * _MACHINE_EPS
        effective_rank = int(np.sum(singular_values > rank_tol))
        return effective_rank == n

    # ── FASE 1.6 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_validate_mac_shape(mac_density: ComplexMatrix) -> int:
        r"""
        FASE 1.6 — Certificación de forma: \(\rho\in\mathrm{Mat}_d(\mathbb{C})\).

        Raises:
            ShapeMismatchError: Si ρ no es 2D, cuadrada o de dimensión ≥ 1.
        """
        if not isinstance(mac_density, np.ndarray) or mac_density.ndim != 2:
            raise ShapeMismatchError(
                "La MAC debe ser un ndarray estrictamente bidimensional."
            )
        d, e = mac_density.shape
        if d != e or d < 1:
            raise ShapeMismatchError(
                f"La MAC debe ser cuadrada de dimensión ≥ 1; forma={mac_density.shape}."
            )
        return int(d)

    # ── FASE 1.7 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_validate_mac_finite(mac_density: ComplexMatrix) -> bool:
        r"""
        FASE 1.7 — Saneamiento IEEE-754 de la MAC.

        Raises:
            NonFiniteInputError: Ante NaN/Inf en la MAC.
        """
        if not np.all(np.isfinite(mac_density)):
            raise NonFiniteInputError("La MAC contiene singularidades numéricas (NaN/Inf).")
        return True

    # ── FASE 1.8 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_mac_hermiticity_residual(mac_density: ComplexMatrix) -> float:
        r"""
        FASE 1.8 — Residuo hermítico: \(\|\rho-\rho^\dagger\|_F\) (invariante blando).
        """
        return float(la.norm(mac_density - mac_density.conj().T, ord="fro"))

    # ── FASE 1.9 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_weyl_symmetrize_mac(mac_density: ComplexMatrix) -> ComplexMatrix:
        r"""
        FASE 1.9 — Proyección de Weyl: \(\rho_{\mathrm{sym}}=\tfrac{1}{2}(\rho+\rho^\dagger)\).

        Garantiza que la diagonalización reciba una entrada exactamente
        hermítica, evitando inconsistencias silenciosas de LAPACK.
        """
        return 0.5 * (mac_density + mac_density.conj().T)

    # ── FASE 1.10 ─────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_mac_spectrum(mac_density_sym: ComplexMatrix) -> RealVector:
        r"""
        FASE 1.10 — Espectro \(\sigma(\rho_{\mathrm{sym}})\) vía diagonalización hermítica.

        Raises:
            MacSpectralAnomalyError: Si la diagonalización no converge (LAPACK).
        """
        try:
            return la.eigvalsh(mac_density_sym).astype(np.float64)
        except la.LinAlgError as err:
            raise MacSpectralAnomalyError(
                f"La diagonalización espectral de la MAC no convergió: {err}"
            ) from err

    # ── FASE 1.11 ─────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_mac_trace_and_anomaly(eigvals: RealVector) -> Tuple[float, bool]:
        r"""
        FASE 1.11 — Traza real y anomalía de conservación de probabilidad:

        \[
        \operatorname{Tr}(\rho)=\sum_i\lambda_i,
        \qquad
        \chi_{\mathrm{trace\ anomaly}}\iff|\operatorname{Tr}(\rho)-1|>\varepsilon_{\mathrm{Tr}}.
        \]

        Se computa **antes** de recortar autovalores negativos, para no
        enmascarar la anomalía.
        """
        trace_val = float(np.sum(eigvals))
        anomaly = abs(trace_val - 1.0) > _EPS_TRACE
        return trace_val, anomaly

    # ── FASE 1.12 ─────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_mac_purity_and_entropy(eigvals: RealVector) -> Tuple[float, float]:
        r"""
        FASE 1.12 — Pureza y entropía de von Neumann, defensivamente normalizadas.

        \[
        p_i=\frac{\max(\lambda_i,0)}{\sum_j\max(\lambda_j,0)},
        \qquad
        \gamma=\sum_i p_i^2,
        \qquad
        S=-\sum_i p_i\log p_i.
        \]

        A diferencia de v1.0.0, el recorte de autovalores negativos ocurre
        **después** de registrar la anomalía de traza/positividad (FASE 1.11
        y 1.13), evitando enmascarar violaciones físicas silenciosamente.
        """
        clipped = np.clip(eigvals, 0.0, None)
        support_sum = float(np.sum(clipped))
        if support_sum <= _EPS_ENTROPY_FLOOR:
            return 0.0, 0.0
        probs = clipped / support_sum
        purity = float(np.sum(probs ** 2))
        support = probs[probs > _EPS_ENTROPY_FLOOR]
        entropy = float(-np.sum(support * np.log(support))) if support.size else 0.0
        return purity, entropy

    # ── FASE 1.13 ─────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_mac_positivity(eigvals: RealVector) -> Tuple[float, bool]:
        r"""
        FASE 1.13 — Positividad espectral (invariante blando):

        \[
        \lambda_{\min}=\min_i\lambda_i,
        \qquad
        \chi_{\mathrm{PSD}}\iff\lambda_{\min}\ge\text{floor}_{\mathrm{PSD}}.
        \]
        """
        min_eig = float(np.min(eigvals))
        is_psd = min_eig >= _SPECTRAL_PSD_FLOOR
        return min_eig, is_psd

    # ── FASE 1.Ω · composición terminal Observe ───────────────────────────
    @staticmethod
    def observe_spectral_state(
        mic_matrix: RealMatrix,
        mac_density: ComplexMatrix,
    ) -> Phase1SpectralObservation:
        r"""
        FASE 1.Ω — Composición terminal de Observación espectral.

        **Contrato F1 → F3**: el DTO ``Phase1SpectralObservation`` alimenta
        el escalamiento de veredicto por anomalías estructurales duras en
        FASE 3 (MAC no-PSD, MIC no full-rank).

        Raises:
            ShapeMismatchError, NonFiniteInputError, MicSpectralAnomalyError,
            MacSpectralAnomalyError.
        """
        n = Phase1_SpectralObserver._phase1_validate_mic_shape(mic_matrix)
        Phase1_SpectralObserver._phase1_validate_mic_finite(mic_matrix)
        mic_singular_values = Phase1_SpectralObserver._phase1_mic_singular_values(mic_matrix)
        mic_cond = Phase1_SpectralObserver._phase1_mic_condition_number(mic_singular_values)
        mic_full_rank = Phase1_SpectralObserver._phase1_mic_rank_certificate(
            mic_singular_values, n
        )

        d = Phase1_SpectralObserver._phase1_validate_mac_shape(mac_density)
        Phase1_SpectralObserver._phase1_validate_mac_finite(mac_density)
        herm_residual = Phase1_SpectralObserver._phase1_mac_hermiticity_residual(mac_density)
        is_hermitian = herm_residual <= _EPS_HERMITICITY
        mac_sym = Phase1_SpectralObserver._phase1_weyl_symmetrize_mac(mac_density)
        mac_eigvals = Phase1_SpectralObserver._phase1_mac_spectrum(mac_sym)
        trace_val, trace_anomaly = Phase1_SpectralObserver._phase1_mac_trace_and_anomaly(
            mac_eigvals
        )
        min_eig, is_psd = Phase1_SpectralObserver._phase1_mac_positivity(mac_eigvals)
        purity, entropy = Phase1_SpectralObserver._phase1_mac_purity_and_entropy(mac_eigvals)

        logger.debug(
            "FASE1.Ω observe: κ(M)=%.4e, full_rank=%s | Tr(ρ)=%.6f, λ_min=%.4e, "
            "γ=%.6f, S=%.6f",
            mic_cond, mic_full_rank, trace_val, min_eig, purity, entropy,
        )

        return Phase1SpectralObservation(
            mic_shape=(n, n),
            mic_is_finite=True,
            mic_condition_number=mic_cond,
            mic_is_full_rank=mic_full_rank,
            mac_shape=(d, d),
            mac_is_finite=True,
            mac_is_hermitian=is_hermitian,
            mac_hermiticity_residual=herm_residual,
            mac_trace=trace_val,
            mac_trace_anomaly=trace_anomaly,
            mac_minimum_eigenvalue=min_eig,
            mac_is_psd=is_psd,
            mac_purity=purity,
            mac_entropy=entropy,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2 — ORIENTACIÓN DEL RESIDUO DE ADJUNCIÓN (Orient)
# Continuación directa de FASE 1 (independiente de datos, paralela en OODA)
# ═══════════════════════════════════════════════════════════════════════════════
class Phase2_AdjunctionOrienter(Phase1_SpectralObserver):
    r"""
    FASE 2: valida el dominio escalar y computa el residuo cuantitativo de
    la Adjunción de Galois frente a la cota de Lipschitz.

    Morfismo compuesto:

    \[
    \mathrm{OrientAdjunction}
    =\mathrm{Residual}\circ\mathrm{ThresholdOrder}\circ\mathrm{ScalarDomain}.
    \]
    """

    # ── FASE 2.1 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_validate_scalar_domain(
        reconstruction_error: float,
        lipschitz_constant: float,
    ) -> None:
        r"""
        FASE 2.1 — Certificación de dominio \(\mathbb{R}_{\ge0}\) de los escalares.

        Normas (\(\varepsilon_{\mathrm{rec}}=\|X-G(F(X))\|\)) y constantes de
        Lipschitz son no-negativas por definición matemática.

        Raises:
            ScalarDomainError: Si algún escalar es negativo, NaN o Inf.
        """
        for name, value in (
            ("reconstruction_error", reconstruction_error),
            ("lipschitz_constant", lipschitz_constant),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ScalarDomainError(
                    f"{name} debe ser real ≥ 0 y finito; recibido {value}."
                )

    # ── FASE 2.2 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_validate_threshold_ordering(
        coherence_threshold: float,
        veto_threshold: float,
    ) -> None:
        r"""
        FASE 2.2 — Certificación de orden total de umbrales:

        \[
        0\le\tau_{\mathrm{coherent}}\le\tau_{\mathrm{veto}}.
        \]

        Raises:
            ThresholdOrderingError: Si los umbrales son negativos, no finitos
                o \(\tau_{\mathrm{coherent}}>\tau_{\mathrm{veto}}\).
        """
        for name, value in (
            ("coherence_threshold", coherence_threshold),
            ("veto_threshold", veto_threshold),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ThresholdOrderingError(
                    f"{name} debe ser real ≥ 0 y finito; recibido {value}."
                )
        if coherence_threshold > veto_threshold:
            raise ThresholdOrderingError(
                f"coherence_threshold={coherence_threshold:.4e} debe ser ≤ "
                f"veto_threshold={veto_threshold:.4e} para una clasificación bien definida."
            )

    # ── FASE 2.3 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase2_compute_adjunction_residual(
        reconstruction_error: float,
        lipschitz_constant: float,
    ) -> float:
        r"""
        FASE 2.3 — Residuo de adjunción relajado (semántica idéntica a v1.0.0):

        \[
        r_{\mathrm{adj}}=\max\bigl(0,\ \varepsilon_{\mathrm{rec}}-L_{\max}\bigr).
        \]

        En un topos perfecto el error de reconstrucción está acotado por
        \(L_{\max}\); el exceso mide la entropía retórica introducida por
        la alucinación del LLM.
        """
        return max(0.0, reconstruction_error - lipschitz_constant)

    # ── FASE 2.Ω · composición terminal Orient ────────────────────────────
    @staticmethod
    def orient_adjunction_residual(
        reconstruction_error: float,
        lipschitz_constant: float,
        coherence_threshold: float,
        veto_threshold: float,
    ) -> Phase2AdjunctionOrientation:
        r"""
        FASE 2.Ω — Composición terminal de Orientación del residuo de Galois.

        **Contrato F2 → F3**: el DTO ``Phase2AdjunctionOrientation`` alimenta
        la clasificación tri-valuada de FASE 3.

        Raises:
            ScalarDomainError, ThresholdOrderingError.
        """
        Phase2_AdjunctionOrienter._phase2_validate_scalar_domain(
            reconstruction_error, lipschitz_constant
        )
        Phase2_AdjunctionOrienter._phase2_validate_threshold_ordering(
            coherence_threshold, veto_threshold
        )
        residual = Phase2_AdjunctionOrienter._phase2_compute_adjunction_residual(
            reconstruction_error, lipschitz_constant
        )

        logger.debug(
            "FASE2.Ω orient: ε_rec=%.4e, L=%.4e, r_adj=%.4e, τ_coh=%.4e, τ_veto=%.4e",
            reconstruction_error, lipschitz_constant, residual,
            coherence_threshold, veto_threshold,
        )

        return Phase2AdjunctionOrientation(
            adjunction_residual=residual,
            reconstruction_error=reconstruction_error,
            lipschitz_constant=lipschitz_constant,
            coherence_threshold=coherence_threshold,
            veto_threshold=veto_threshold,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 3 — DECISIÓN, ESCALAMIENTO Y SELLADO (Decide + Act)
# Continuación directa de FASE 1 ∥ FASE 2
# ═══════════════════════════════════════════════════════════════════════════════
class Phase3_VerdictDecider(Phase2_AdjunctionOrienter):
    r"""
    FASE 3: clasifica el veredicto en el retículo Ω₃, lo escala por
    anomalías estructurales duras (MAC no-PSD, MIC no full-rank), activa el
    Crowbar físico y sella el certificado de gobernanza.

    Morfismo compuesto:

    \[
    \mathrm{DecideAndAct}
    =\mathrm{Seal}\circ\mathrm{Log}\circ\mathrm{Crowbar}\circ\mathrm{Escalate}\circ\mathrm{Classify}.
    \]
    """

    # ── FASE 3.1 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_classify_base_verdict(
        residual: float,
        coherence_threshold: float,
        veto_threshold: float,
    ) -> EpistemicSuturatorVerdict:
        r"""
        FASE 3.1 — Clasificación base en el retículo distributivo acotado Ω₃
        (semántica idéntica a v1.0.0):

        \[
        \mathrm{verdict}=
        \begin{cases}
        \mathrm{COHERENT} & r_{\mathrm{adj}}\le\tau_{\mathrm{coherent}}\\
        \mathrm{DEGRADED} & \tau_{\mathrm{coherent}}<r_{\mathrm{adj}}\le\tau_{\mathrm{veto}}\\
        \mathrm{VETOED} & r_{\mathrm{adj}}>\tau_{\mathrm{veto}}
        \end{cases}
        """
        if residual <= coherence_threshold:
            return EpistemicSuturatorVerdict.COHERENT
        if residual <= veto_threshold:
            return EpistemicSuturatorVerdict.DEGRADED
        return EpistemicSuturatorVerdict.VETOED

    # ── FASE 3.2 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_escalate_verdict(
        base_verdict: EpistemicSuturatorVerdict,
        mac_is_psd: bool,
        mic_is_full_rank: bool,
    ) -> EpistemicSuturatorVerdict:
        r"""
        FASE 3.2 — Escalamiento monótono por anomalías estructurales duras.

        Un \(\rho_{\mathrm{MAC}}\) no-PSD es físicamente imposible (firma
        fuerte de alucinación) y fuerza \(\mathrm{VETOED}\) independientemente
        del residuo de adjunción. Una MIC de rango deficiente degrada la
        base canónica y eleva el piso del veredicto a \(\mathrm{DEGRADED}\).

        El escalamiento es **monótono creciente**: nunca reduce la severidad
        determinada por el residuo puro (FASE 3.1), solo la incrementa.
        """
        verdict = base_verdict
        if not mac_is_psd:
            verdict = EpistemicSuturatorVerdict.VETOED
        elif not mic_is_full_rank and verdict == EpistemicSuturatorVerdict.COHERENT:
            verdict = EpistemicSuturatorVerdict.DEGRADED
        return verdict

    # ── FASE 3.3 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_determine_crowbar_activation(verdict: EpistemicSuturatorVerdict) -> bool:
        r"""
        FASE 3.3 — Activación del bypass de hardware perimetral (ESP32).

        \[
        \chi_{\mathrm{crowbar}}\iff\mathrm{verdict}=\mathrm{VETOED}.
        \]
        """
        return verdict == EpistemicSuturatorVerdict.VETOED

    # ── FASE 3.4 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_log_verdict_telemetry(
        verdict: EpistemicSuturatorVerdict,
        residual: float,
        lipschitz_constant: float,
        mac_entropy: float,
        is_crowbar_active: bool,
    ) -> None:
        r"""
        FASE 3.4 — Telemetría de bitácora graduada por severidad (ERROR /
        WARNING / INFO), a diferencia de la bifurcación binaria de v1.0.0.
        """
        if verdict == EpistemicSuturatorVerdict.VETOED:
            logger.error(
                "¡VETO DE SUTURA! La alucinación del LLM quebrantó el isomorfismo "
                "de Galois. Residuo de Adjunción = %.4e | L_max = %.4e. "
                "Activando Crowbar.",
                residual, lipschitz_constant,
            )
        elif verdict == EpistemicSuturatorVerdict.DEGRADED:
            logger.warning(
                "Sutura de Galois degradada bajo regularización tolerable. "
                "Residuo de Adjunción = %.4e | L_max = %.4e | S_vN = %.4f.",
                residual, lipschitz_constant, mac_entropy,
            )
        else:
            logger.info(
                "Sutura de Galois aprobada de forma asintótica. Veredicto: %s | S_vN = %.4f",
                verdict.name, mac_entropy,
            )

    # ── FASE 3.5 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase3_seal_verdict_state(
        observation: Phase1SpectralObservation,
        orientation: Phase2AdjunctionOrientation,
        verdict: EpistemicSuturatorVerdict,
        is_crowbar_active: bool,
    ) -> SuturatorAgentVerdictState:
        """FASE 3.5 — Sellado del objeto terminal frozen del funtor OODA."""
        timestamp_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return SuturatorAgentVerdictState(
            verdict=verdict,
            adjunction_residual=orientation.adjunction_residual,
            mic_condition_number=observation.mic_condition_number,
            mac_entropy=observation.mac_entropy,
            is_crowbar_active=is_crowbar_active,
            mac_purity=observation.mac_purity,
            mac_trace_anomaly=observation.mac_trace_anomaly,
            mic_rank_deficient=not observation.mic_is_full_rank,
            reconstruction_error=orientation.reconstruction_error,
            lipschitz_constant=orientation.lipschitz_constant,
            coherence_threshold=orientation.coherence_threshold,
            veto_threshold=orientation.veto_threshold,
            timestamp_utc=timestamp_utc,
        )

    # ── FASE 3.Ω · composición terminal Decide + Act ──────────────────────
    @staticmethod
    def decide_and_seal(
        observation: Phase1SpectralObservation,
        orientation: Phase2AdjunctionOrientation,
    ) -> SuturatorAgentVerdictState:
        r"""
        FASE 3.Ω — Composición terminal de Decisión y Actuación.

        **Continuación funtorial de FASE 1 ∥ FASE 2**: consume ambos DTOs
        para clasificar, escalar, loguear y sellar el certificado final.
        """
        base_verdict = Phase3_VerdictDecider._phase3_classify_base_verdict(
            orientation.adjunction_residual,
            orientation.coherence_threshold,
            orientation.veto_threshold,
        )
        verdict = Phase3_VerdictDecider._phase3_escalate_verdict(
            base_verdict, observation.mac_is_psd, observation.mic_is_full_rank
        )
        is_crowbar_active = Phase3_VerdictDecider._phase3_determine_crowbar_activation(verdict)

        Phase3_VerdictDecider._phase3_log_verdict_telemetry(
            verdict,
            orientation.adjunction_residual,
            orientation.lipschitz_constant,
            observation.mac_entropy,
            is_crowbar_active,
        )

        return Phase3_VerdictDecider._phase3_seal_verdict_state(
            observation, orientation, verdict, is_crowbar_active
        )


# ═══════════════════════════════════════════════════════════════════════════════
# AGENTE SOBERANO — MORPHIC SUTURATOR AGENT (ORQUESTADOR CATEGORIAL)
# Observe (F1) ∥ Orient (F2) ⟶ Decide+Act (F3) ⟶ Seal (fail-secure)
# ═══════════════════════════════════════════════════════════════════════════════
class MorphicSuturatorAgent(Morphism, Phase3_VerdictDecider):
    r"""
    Agente soberano de gobernanza y control de la sutura ciber-física MIC-MAC.

    .. code-block:: text

        ┌────────────────────────────────────────────────────────────┐
        │ FASE 1  Observe                │ FASE 2  Orient             │
        │   1.1 validate_mic_shape       │   2.1 validate_scalar_dom  │
        │   1.2 validate_mic_finite      │   2.2 validate_threshold   │
        │   1.3 mic_singular_values      │   2.3 compute_residual     │
        │   1.4 mic_condition_number     │   2.Ω orient_adjunction    │
        │   1.5 mic_rank_certificate     │                            │
        │   1.6-1.7 validate_mac         │                            │
        │   1.8 hermiticity_residual     │                            │
        │   1.9 weyl_symmetrize          │                            │
        │   1.10 mac_spectrum            │                            │
        │   1.11 trace_and_anomaly       │                            │
        │   1.12 purity_and_entropy      │                            │
        │   1.13 positivity              │                            │
        │   1.Ω observe_spectral_state   │                            │
        ├─────────────────────────────────┴─────────────────────────── ┤
        │ FASE 3  Decide + Act                                         │
        │   3.1 classify_base_verdict   3.4 log_verdict_telemetry      │
        │   3.2 escalate_verdict        3.5 seal_verdict_state         │
        │   3.3 determine_crowbar       3.Ω decide_and_seal            │
        ├───────────────────────────────────────────────────────────── ┤
        │ FAIL-SECURE WRAPPER (nunca propaga excepciones, salvo         │
        │ raise_on_veto=True sobre un veredicto VETOED legítimo)        │
        └───────────────────────────────────────────────────────────── ┘
    """

    def __init__(self, adjunction_tolerance: float = 1e-10) -> None:
        """Inicializa al centinela de sutura acoplando la tolerancia del isomorfismo."""
        super().__init__()
        self._target_stratum: Stratum = Stratum.WISDOM
        self._adjunction_tolerance: float = adjunction_tolerance

    # ── Fail-secure · construcción de estado de colapso ──────────────────────
    @staticmethod
    def _build_failure_state(reason: str) -> SuturatorAgentVerdictState:
        r"""
        Construye el certificado de colapso determinista ante un fallo
        catastrófico de la FPU o una violación de dominio no anticipada.
        Centraliza la lógica que en v1.0.0 estaba inlineada en el bloque
        ``except``, evitando duplicación entre las distintas rutas de captura.
        """
        timestamp_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
        logger.critical(
            "Colapso catastrófico durante la auditoría de sutura: %s. "
            "Forzando colapso de estado a VETOED.",
            reason,
        )
        return SuturatorAgentVerdictState(
            verdict=EpistemicSuturatorVerdict.VETOED,
            adjunction_residual=float("inf"),
            mic_condition_number=float("inf"),
            mac_entropy=float("inf"),
            is_crowbar_active=True,
            timestamp_utc=timestamp_utc,
        )

    # ── Compositor público OODA ────────────────────────────────────────────
    def execute_sutured_ooda_cycle(
        self,
        mic_matrix: RealMatrix,
        mac_density: ComplexMatrix,
        reconstruction_error: float,
        lipschitz_constant: float,
        coherence_threshold: Optional[float] = None,
        veto_threshold: float = 1e-6,
        raise_on_veto: bool = False,
    ) -> SuturatorAgentVerdictState:
        r"""
        Ejecuta el ciclo OODA covariante completo sobre el espacio de fase
        de acoplamiento MIC-MAC.

        Args:
            mic_matrix: Matriz de Interacción Central (\(M_{\mathrm{MIC}}\)).
            mac_density: Operador densidad de la MAC (\(\rho_{\mathrm{MAC}}\)).
            reconstruction_error: Residuo \(\|X-G(F(X))\|_F\) del isomorfismo.
            lipschitz_constant: Cota de Lipschitz conforme \(L_{\max}\).
            coherence_threshold: Límite superior para el estado nominal. Si
                ``None`` (default), se resuelve a ``self._adjunction_tolerance``
                — vincula el estado del constructor con el ciclo de decisión
                (corrige el código muerto de v1.0.0).
            veto_threshold: Límite superior para el estado degradado.
            raise_on_veto: Si ``True``, eleva ``AdjunctionBreachVeto`` cuando
                el veredicto legítimamente clasificado es ``VETOED`` (en vez
                de solo retornar el estado sellado). No afecta el
                comportamiento fail-secure ante fallos internos.

        Returns:
            SuturatorAgentVerdictState: El reporte inmutable de gobernanza
            categórica. **Contrato fail-secure**: ante cualquier error de
            dominio o colapso numérico interno, retorna un estado ``VETOED``
            sellado en vez de propagar la excepción (salvo ``raise_on_veto``
            combinado con clasificación legítima, ver arriba).
        """
        resolved_coherence_threshold = (
            self._adjunction_tolerance if coherence_threshold is None else coherence_threshold
        )

        try:
            # ── FASE 1 · Observe ─────────────────────────────────────────────
            observation = self.observe_spectral_state(mic_matrix, mac_density)

            # ── FASE 2 · Orient ──────────────────────────────────────────────
            orientation = self.orient_adjunction_residual(
                reconstruction_error,
                lipschitz_constant,
                resolved_coherence_threshold,
                veto_threshold,
            )

            # ── FASE 3 · Decide + Act ────────────────────────────────────────
            state = self.decide_and_seal(observation, orientation)

            if raise_on_veto and state.verdict == EpistemicSuturatorVerdict.VETOED:
                raise AdjunctionBreachVeto(
                    f"Veredicto VETOED bajo raise_on_veto=True: "
                    f"r_adj={state.adjunction_residual:.4e} > "
                    f"veto_threshold={veto_threshold:.4e}."
                )

            return state

        except AdjunctionBreachVeto:
            # Re-propagación intencional del veto estricto solicitado por el
            # llamador; no forma parte del contrato fail-secure por defecto.
            raise
        except SuturatorAgentError as err:
            return self._build_failure_state(str(err))
        except (la.LinAlgError, ValueError, TypeError, ArithmeticError) as err:
            return self._build_failure_state(
                f"Colapso de FPU/álgebra lineal no categorizado: {err}"
            )
        except Exception as err:  # pragma: no cover — red de seguridad final
            return self._build_failure_state(
                f"Excepción no anticipada durante la sutura OODA: {err}"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# Exportación canónica del módulo
# ═══════════════════════════════════════════════════════════════════════════════
__all__ = [
    # Excepciones
    "SuturatorAgentError",
    "NonFiniteInputError",
    "ShapeMismatchError",
    "MicSpectralAnomalyError",
    "MacSpectralAnomalyError",
    "ScalarDomainError",
    "ThresholdOrderingError",
    "AdjunctionBreachVeto",
    # Retículo de veredictos
    "EpistemicSuturatorVerdict",
    # DTOs
    "Phase1SpectralObservation",
    "Phase2AdjunctionOrientation",
    "SuturatorAgentVerdictState",
    # Fases
    "Phase1_SpectralObserver",
    "Phase2_AdjunctionOrienter",
    "Phase3_VerdictDecider",
    # Agente
    "MorphicSuturatorAgent",
    # Constantes útiles para tests
    "_DEFAULT_TOL",
    "_SPECTRAL_PSD_FLOOR",
    "_MACHINE_EPS",
    "_EPS_HERMITICITY",
    "_EPS_TRACE",
]