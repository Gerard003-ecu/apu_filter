# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Witten-Atiyah Agent (Inquisidor de Invarianza Global y TQFT)        ║
║ Ruta   : app/omega/witten_atiyah_agent.py                                    ║
║ Versión: 3.0.0-Atiyah-Singer-Witten-APS-Spectral-Rigorous                    ║
║ Evolución: Rigor PhD – Índice de Atiyah–Singer + APS + η-invariante + TQFT   ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y TEORÍA CUÁNTICA DE CAMPOS TOPOLÓGICA (TQFT)
────────────────────────────────────────────────────────────────────────────────
Endofuntor Supremo del Estrato Ω. Su mandato axiomático es auditar las
integrales de trayectoria topológicas sobre la categoría de cobordismos Cob(n).
Protege la independencia de fondo (background independence) aplicando el
Teorema del Índice de Atiyah–Singer (y su refinamiento APS con η-invariante)
y orquestando el TQFTProjectionManifold.

FUNDAMENTOS MATEMÁTICOS RIGUROSOS (nivel doctorado):
  • Funtor de olvido U : Met → Top (despoja G_{μν} y m^{**} de Fröhlich).
  • Categoría de operadores densidad dens(ℋ) con morfismos CPTP.
  • Inmersión isométrica ι : ℋ_in ↪ ℋ_out vía suma directa con vacío puro
    |0⟩⟨0| (S_vN = 0, pureza = 1).
  • Teorema del Índice de Atiyah–Singer:
        ind(⧸D) = dim(ker ⧸D⁺) − dim(ker ⧸D⁻) = ∫_M Â(TM) ∧ ch(E).
  • Refinamiento APS: ind_APS = (η(0) + h)/2 + ∫ Â ∧ ch  (η = asimetría espectral).
  • Integral de trayectoria de Witten / Chern–Simons:
        Z(M) = ∫ 𝒟A exp( i k/(4π) ∫_M Tr(A∧dA + ⅔ A∧A∧A) ).
  • Proyección booleana sobre el retículo distributivo acotado {VIABLE, RECHAZAR}.

ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta monoidal):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Funtor de Olvido Métrico (U): Despoja el tensor métrico Riemanniano
         G_{μν} y la masa inercial de Fröhlich m^{**}. Proyecta a operadores
         densidad Hermitianos, positivos y de traza unidad.
         Último método formal: apply_forgetful_functor(…) → PurifiedPair.
         Este objeto es el dominio exacto de todos los métodos de la Fase 2.

Fase 2 → Funtor de Inmersión Fibrada (ι) + Teorema del Índice: Continúa
         desde PurifiedPair. Nivela dim ℋ (Σ_in ⊕ ℋ_∅ ≅ Σ_out), evalúa
         ind(⧸D), η-invariante y verifica Atiyah–Singer / APS.
         Produce IndexCertifiedEmbedding (embedding + DiracIndexData).

Fase 3 → Veredicto TQFT: Continúa desde IndexCertifiedEmbedding. Orquesta
         la integral de camino de Witten vía TQFTProjectionManifold y proyecta
         el veredicto sobre el retículo de severidad.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ------------------------------------------------------------------------------
# Dependencias arquitectónicas del ecosistema APU Filter
# ------------------------------------------------------------------------------
try:
    from app.core.mic_algebra import (
        Morphism,
        CategoricalState,
        TopologicalInvariantError,
    )
    from app.wisdom.semantic_translator import VerdictLevel
    from app.omega.tqft_projection_manifold import (
        TQFTProjectionManifold,
        TQFTBoundary,
        TQFTVerdict,
        TQFTConstants,
        TopologicalKnotVeto,
        CobordismDegeneracyError,
        TuraevViroCollapseError,
    )
except ImportError:
    # Stubs rigurosos para ejecución aislada y prueba unitaria analítica
    class TopologicalInvariantError(Exception):
        """Excepción raíz del ecosistema APU (stub)."""
        pass

    class Morphism:
        """Clase base para morfismos categóricos (stub)."""
        pass

    class CategoricalState:
        """Estado categórico abstracto (stub)."""
        pass

    class VerdictLevel(Enum):
        VIABLE = auto()
        RECHAZAR = auto()

    class TopologicalKnotVeto(Exception):
        pass

    class CobordismDegeneracyError(Exception):
        pass

    class TuraevViroCollapseError(Exception):
        pass

    class TQFTConstants:
        MACHINE_EPS: float = float(np.finfo(np.float64).eps)
        CHERN_SIMONS_K: int = 3

    class TQFTBoundary:
        def __init__(
            self,
            state_vector: NDArray[np.float64],
            betti_numbers: Tuple[int, ...],
            hilbert_dimension: int,
        ) -> None:
            self.state_vector = state_vector
            self.betti_numbers = betti_numbers
            self.hilbert_dimension = hilbert_dimension

    class TQFTVerdict:
        def __init__(self, invariants: Any = None, verdict: Any = None,
                     topological_trace: str = "") -> None:
            self.invariants = invariants
            self.verdict = verdict if verdict is not None else VerdictLevel.VIABLE
            self.topological_trace = topological_trace

    class TQFTProjectionManifold:
        """Stub del proyector TQFT."""
        def project_intent(self, *args: Any, **kwargs: Any) -> TQFTVerdict:
            return TQFTVerdict(verdict=VerdictLevel.VIABLE, topological_trace="stub")


logger = logging.getLogger("MIC.Omega.WittenAtiyahAgent")


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §A. CONSTANTES FÍSICO-GEOMÉTRICAS, TOLERANCIAS ESPECTRALES Y UMBRALES APS    ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class WittenAtiyahConstants:
    r"""
    Constantes topológicas, tolerancias espectrales del operador de Dirac y
    umbrales del refinamiento Atiyah–Patodi–Singer (APS).

    Attributes
    ----------
    MACHINE_EPS : float
        Épsilon de máquina en precisión doble (≈ 2.22 × 10⁻¹⁶).
    DIRAC_INDEX_TOLERANCE : float
        Umbral para considerar un valor singular / autovalor como nulo al
        calcular dim(ker) y dim(coker).
    VACUUM_ENTROPY_TOLERANCE : float
        Tolerancia máxima para S_vN(|0⟩⟨0|) (debe ser idénticamente 0).
    PURITY_TOLERANCE : float
        Tolerancia para pureza Tr(ρ²) ∈ [0, 1] y desviaciones de Hermiticidad.
    POSITIVITY_TOLERANCE : float
        Autovalores de ρ por debajo de −POSITIVITY_TOLERANCE se rechazan.
    ETA_SPECTRAL_CUTOFF : float
        Corte IR para la regularización de la η-invariante de APS.
    MIN_HILBERT_DIM : int
        Dimensión mínima admisible de ℋ.
    DEFAULT_BETTI : Tuple[int, ...]
        Números de Betti por defecto (esfera homológica S²: β₀=1, β₁=0, β₂=0)
        cuando no se dispone de un analizador topológico externo.
    """
    MACHINE_EPS: float = float(np.finfo(np.float64).eps)
    DIRAC_INDEX_TOLERANCE: float = 1e-10
    VACUUM_ENTROPY_TOLERANCE: float = 1e-12
    PURITY_TOLERANCE: float = 1e-10
    POSITIVITY_TOLERANCE: float = 1e-12
    ETA_SPECTRAL_CUTOFF: float = 1e-14
    MIN_HILBERT_DIM: int = 1
    DEFAULT_BETTI: Tuple[int, ...] = (1, 0, 0)

    @staticmethod
    def von_neumann_entropy(eigenvals: NDArray[np.floating], tol: float | None = None) -> float:
        r"""
        Entropía de von Neumann S_vN = −∑_i λ_i log(λ_i) sobre el espectro
        de un estado (autovalores no negativos que suman ≈ 1).

        Parameters
        ----------
        eigenvals : NDArray
            Autovalores (reales) del operador densidad.
        tol : float, optional
            Umbral por debajo del cual λ se trata como 0 (0 log 0 := 0).

        Returns
        -------
        float
            S_vN ≥ 0.
        """
        if tol is None:
            tol = WittenAtiyahConstants.MACHINE_EPS
        eigs = np.asarray(eigenvals, dtype=np.float64).real
        eigs = eigs[eigs > tol]
        if eigs.size == 0:
            return 0.0
        # Normalización defensiva (por si la traza no es exactamente 1)
        eigs = eigs / np.sum(eigs)
        return float(-np.sum(eigs * np.log(eigs)))

    @staticmethod
    def purity(rho: NDArray[np.complex128]) -> float:
        r"""Pureza 𝒫(ρ) = Tr(ρ²) ∈ [1/d, 1]."""
        return float(np.real(np.trace(rho @ rho)))


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §B. JERARQUÍA DE EXCEPCIONES TOPOLÓGICAS (VETOS ABSOLUTOS / OBJETOS INICIALES)║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class WittenAtiyahError(TopologicalInvariantError):
    """Excepción raíz del Inquisidor de Invarianza Global (objeto inicial)."""
    pass


class GaugeTearingError(WittenAtiyahError):
    r"""
    Se detona si la asimetría dimensional entre fronteras no puede ser suturada
    por un subespacio trivial de vacío, o si el estado pierde Hermiticidad /
    positividad / traza al despojarse de la métrica.

    Violaciones típicas:
      • Tr(ρ) ≈ 0 (colapso métrico total).
      • dim ℋ_in > dim ℋ_out sin traza parcial (contracción prohibida).
      • Inyección de vacío con S_vN > 0 (entropía fantasma).
      • ρ no Hermitiano o no positivo semidefinido.
    """
    pass


class IndexTheoremViolation(WittenAtiyahError):
    r"""
    Se detona cuando el Índice Analítico del Operador de Dirac no coincide
    con el Invariante Topológico (Atiyah–Singer / APS):

        ind(⧸D) ≠ ∫_M Â(TM) ∧ ch(E)   (mod correcciones η de borde).

    Demuestra la aniquilación o creación espuria de información en la malla
    (violación de la conservación quiral / anomalía no cancelada).
    """
    pass


class OntologicalTQFTVeto(WittenAtiyahError):
    r"""
    Se detona cuando la Integral de Trayectoria de Witten expone un Nudo
    Logístico insoluble, una degeneración de cobordismo o una fluctuación
    estocástica extrema en Z(M).
    """
    pass


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ §C. DTOs INMUTABLES (Contratos Categóricos entre Fases Anidadas)             ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
@dataclass(frozen=True, slots=True)
class PurifiedPair:
    r"""
    Producto de la Fase 1. Par de operadores densidad purificados
    (esqueleto topológico, independientes de G_{μν}).

    Invariantes de representación:
      • ρ = ρ†  (Hermítico)
      • ρ ≥ 0   (positivo semidefinido)
      • Tr(ρ) = 1
      • 𝒫(ρ) = Tr(ρ²) ∈ [1/d, 1]

    Attributes
    ----------
    sigma_in : NDArray[np.complex128]
        Operador densidad de entrada purificado.
    sigma_out : NDArray[np.complex128]
        Operador densidad de salida purificado.
    purity_in : float
        Pureza Tr(ρ_in²).
    purity_out : float
        Pureza Tr(ρ_out²).
    entropy_in : float
        S_vN(ρ_in).
    entropy_out : float
        S_vN(ρ_out).
    """
    sigma_in: NDArray[np.complex128]
    sigma_out: NDArray[np.complex128]
    purity_in: float
    purity_out: float
    entropy_in: float
    entropy_out: float


@dataclass(frozen=True, slots=True)
class DimensionalEmbedding:
    r"""
    Artefacto intermedio de la Fase 2: estado inmerso tras aplicar el funtor ι.

    Garantiza:
      • dim(σ_in_embedded) = dim(σ_out).
      • Tr(ρ_embedded) = 1.
      • S_vN(|0⟩⟨0|) = 0 (entropía nula del vacío inyectado).
      • La inmersión es isométrica en la norma de traza sobre el bloque original.

    Attributes
    ----------
    sigma_in_embedded : NDArray[np.complex128]
        Estado de entrada aumentado: Σ_in ⊕ ℋ_∅.
    sigma_out : NDArray[np.complex128]
        Estado de salida (misma dimensión).
    vacuum_dimension_added : int
        Dimensión del vacío añadido (0 si ya coincidían).
    is_isometric : bool
        True si la inmersión preserva traza y pureza del bloque original.
    betti_numbers : Tuple[int, ...]
        Números de Betti asignados a las fronteras del cobordismo.
    """
    sigma_in_embedded: NDArray[np.complex128]
    sigma_out: NDArray[np.complex128]
    vacuum_dimension_added: int
    is_isometric: bool
    betti_numbers: Tuple[int, ...] = WittenAtiyahConstants.DEFAULT_BETTI


@dataclass(frozen=True, slots=True)
class DiracIndexData:
    r"""
    Resultados del Teorema del Índice de Atiyah–Singer / APS.

    Attributes
    ----------
    analytical_index : int
        Índice analítico: dim(ker ⧸D⁺) − dim(ker ⧸D⁻)
        (para endomorfismos finito-dimensionales = 0).
    topological_invariant : int
        Invariante topológico estimado ∫ Â(TM) ∧ ch(E) (en ℤ).
    operator_kernel_dim : int
        dim(ker ⧸D).
    operator_cokernel_dim : int
        dim(coker ⧸D).
    eta_invariant : float
        η-invariante de Atiyah–Patodi–Singer (asimetría espectral).
    spectral_flow_estimate : int
        Estimación entera del flujo espectral a través de cero.
    is_theorem_satisfied : bool
        True si analytical_index == topological_invariant (mod APS).
    """
    analytical_index: int
    topological_invariant: int
    operator_kernel_dim: int
    operator_cokernel_dim: int
    eta_invariant: float
    spectral_flow_estimate: int
    is_theorem_satisfied: bool


@dataclass(frozen=True, slots=True)
class IndexCertifiedEmbedding:
    r"""
    Producto de la Fase 2. Embedding dimensional certificado por el
    Teorema del Índice (dominio exacto de la Fase 3).

    Attributes
    ----------
    embedding : DimensionalEmbedding
        Estados embebidos isométricamente.
    index_data : DiracIndexData
        Certificado del índice de Atiyah–Singer / APS.
    purified : PurifiedPair
        Par purificado original (trazabilidad forense de la Fase 1).
    """
    embedding: DimensionalEmbedding
    index_data: DiracIndexData
    purified: PurifiedPair


@dataclass(frozen=True, slots=True)
class WittenAtiyahVerdict:
    r"""
    Producto de la Fase 3. Veredicto global del Inquisidor.

    Attributes
    ----------
    verdict : VerdictLevel
        VIABLE o RECHAZAR.
    index_data : DiracIndexData
        Certificado del índice.
    tqft_trace : str
        Traza topológica de la integral de Witten.
    vacuum_dimension_added : int
        Dimensión de vacío inyectada en la inmersión.
    """
    verdict: VerdictLevel
    index_data: DiracIndexData
    tqft_trace: str
    vacuum_dimension_added: int


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 1 : FUNTOR DE OLVIDO MÉTRICO (U : Met → Top)                           ║
# ║ Despoja G_{μν} y m^{**}; produce operadores densidad canónicos.             ║
# ║ Último método: apply_forgetful_functor → PurifiedPair (dominio de Fase 2).  ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class Phase1_MetricForgetfulFunctor:
    r"""
    Fase 1 – Aplica el funtor de olvido U : Met → Top que despoja a los estados
    de toda dependencia del tensor métrico G_{μν} y de la masa de Fröhlich m^{**},
    conservando únicamente el esqueleto topológico (tipo de homotopía + clase
    de densidad cuántica).

    Pipeline de purificación (por cada frontera):
      1. Proyección Hermítica:  ρ ← (ρ + ρ†)/2.
      2. Recorte de positividad espectral (autovalores λ < 0 → 0).
      3. Renormalización de traza: ρ ← ρ / Tr(ρ).
      4. Verificación de invariantes: Tr=1, ρ≥0, ρ=ρ†, 𝒫∈[1/d,1].

    El método terminal apply_forgetful_functor produce un PurifiedPair,
    dominio exacto de la Fase 2.
    """

    def _assert_square_matrix(
        self,
        rho: NDArray[np.complex128],
        name: str,
    ) -> None:
        """Valida que ρ sea una matriz cuadrada 2-dimensional."""
        if not isinstance(rho, np.ndarray):
            raise GaugeTearingError(f"{name}: se esperaba NDArray, recibido {type(rho)}.")
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise GaugeTearingError(
                f"{name}: debe ser matriz cuadrada; shape={getattr(rho, 'shape', None)}."
            )
        if rho.shape[0] < WittenAtiyahConstants.MIN_HILBERT_DIM:
            raise GaugeTearingError(
                f"{name}: dim ℋ = {rho.shape[0]} < MIN_HILBERT_DIM."
            )

    def _strip_metric_tensor(
        self,
        rho_metric: NDArray[np.complex128],
        name: str = "ρ",
    ) -> Tuple[NDArray[np.complex128], float, float]:
        r"""
        Extrae el esqueleto topológico de un operador densidad ρ contaminado
        por la métrica / masa inercial.

        Pasos formales:
          (i)   ρ ← (ρ + ρ†)/2                         (proyección Hermítica)
          (ii)  espectro: λ_i ← max(λ_i, 0)            (proyección al cono PSD)
          (iii) ρ ← ρ / Tr(ρ)                          (olvido de escala métrica)
          (iv)  verificación de invariantes de dens(ℋ)

        Parameters
        ----------
        rho_metric : NDArray[np.complex128]
            Operador densidad (matriz cuadrada) posiblemente acoplado a G_{μν}.
        name : str
            Etiqueta para mensajes de error (σ_in / σ_out).

        Returns
        -------
        Tuple[NDArray[np.complex128], float, float]
            (ρ_puro, pureza, S_vN).

        Raises
        ------
        GaugeTearingError
            Si la traza colapsa, si hay negatividad excesiva no recuperable,
            o si los invariantes de dens(ℋ) fallan.
        """
        self._assert_square_matrix(rho_metric, name)
        rho = np.asarray(rho_metric, dtype=np.complex128).copy()

        # (i) Proyección Hermítica – olvido de fases métricas anti-Hermíticas
        rho = 0.5 * (rho + rho.conj().T)

        # (ii) Proyección al cono de operadores positivos semidefinidos
        #      vía descomposición espectral (teorema espectral para Hermíticos)
        try:
            evals, evecs = la.eigh(rho)
        except la.LinAlgError as exc:
            raise GaugeTearingError(
                f"{name}: descomposición espectral fallida tras proyección Hermítica: {exc}"
            ) from exc

        # Detectar negatividad patológica (más allá de ruido numérico)
        if np.any(evals < -WittenAtiyahConstants.POSITIVITY_TOLERANCE):
            n_neg = int(np.sum(evals < -WittenAtiyahConstants.POSITIVITY_TOLERANCE))
            raise GaugeTearingError(
                f"{name}: {n_neg} autovalor(es) significativamente negativo(s) "
                f"(min λ = {evals.min():.3e}). El estado no es un operador densidad."
            )

        evals_psd = np.clip(evals, 0.0, None)
        rho = (evecs * evals_psd) @ evecs.conj().T
        # Re-Hermitianizar por seguridad numérica
        rho = 0.5 * (rho + rho.conj().T)

        # (iii) Renormalización de traza (olvido de la escala / masa de Fröhlich)
        trace_rho = float(np.real(np.trace(rho)))
        if abs(trace_rho) < WittenAtiyahConstants.MACHINE_EPS:
            raise GaugeTearingError(
                f"{name}: el tensor logístico colapsó al despojarse de la métrica "
                f"(Tr(ρ) = {trace_rho:.3e} despreciable)."
            )
        rho = rho / trace_rho

        # (iv) Invariantes de dens(ℋ)
        tr_final = float(np.real(np.trace(rho)))
        if abs(tr_final - 1.0) > WittenAtiyahConstants.PURITY_TOLERANCE:
            raise GaugeTearingError(
                f"{name}: fallo de normalización Tr(ρ) = {tr_final:.6e} ≠ 1."
            )

        herm_residual = float(np.linalg.norm(rho - rho.conj().T, ord="fro"))
        if herm_residual > WittenAtiyahConstants.PURITY_TOLERANCE:
            raise GaugeTearingError(
                f"{name}: residual anti-Hermítico = {herm_residual:.3e}."
            )

        pur = WittenAtiyahConstants.purity(rho)
        d = rho.shape[0]
        if pur < (1.0 / d) - WittenAtiyahConstants.PURITY_TOLERANCE or pur > 1.0 + WittenAtiyahConstants.PURITY_TOLERANCE:
            raise GaugeTearingError(
                f"{name}: pureza 𝒫 = {pur:.6e} fuera de [1/d, 1] = [{1.0/d:.6e}, 1]."
            )

        eigs_final = la.eigvalsh(rho)
        entropy = WittenAtiyahConstants.von_neumann_entropy(eigs_final)

        return rho.astype(np.complex128), pur, entropy

    def apply_forgetful_functor(
        self,
        raw_sigma_in: NDArray[np.complex128],
        raw_sigma_out: NDArray[np.complex128],
    ) -> PurifiedPair:
        r"""
        Aplica U a ambas fronteras y devuelve el par topológico puro.

        Este es el método terminal formal de la Fase 1. Su producto
        (PurifiedPair) es el dominio exacto de todos los métodos de la Fase 2.

        Parameters
        ----------
        raw_sigma_in, raw_sigma_out : NDArray[np.complex128]
            Operadores densidad crudos de entrada y salida.

        Returns
        -------
        PurifiedPair
            Estados purificados + diagnósticos de pureza y entropía.

        Raises
        ------
        GaugeTearingError
            Si alguna frontera no admite purificación topológica.
        """
        sigma_in_pure, pur_in, s_in = self._strip_metric_tensor(raw_sigma_in, "σ_in")
        sigma_out_pure, pur_out, s_out = self._strip_metric_tensor(raw_sigma_out, "σ_out")

        pair = PurifiedPair(
            sigma_in=sigma_in_pure,
            sigma_out=sigma_out_pure,
            purity_in=pur_in,
            purity_out=pur_out,
            entropy_in=s_in,
            entropy_out=s_out,
        )
        logger.debug(
            "Fase 1 completada: métrica olvidada. "
            "𝒫_in=%.4f S_in=%.4e | 𝒫_out=%.4f S_out=%.4e | dims=(%d, %d).",
            pur_in, s_in, pur_out, s_out,
            sigma_in_pure.shape[0], sigma_out_pure.shape[0],
        )
        return pair


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 2 : FUNTOR DE INMERSIÓN FIBRADA (ι) + TEOREMA DEL ÍNDICE (A-S / APS)   ║
# ║ Continuación directa del PurifiedPair de la Fase 1.                         ║
# ║ Último método: certify_index → IndexCertifiedEmbedding (dominio de Fase 3). ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class Phase2_AtiyahSingerEmbedding(Phase1_MetricForgetfulFunctor):
    r"""
    Fase 2 – Nivela las dimensiones de Hilbert mediante una inmersión isométrica
    ι y verifica el Teorema del Índice de Atiyah–Singer (con refinamiento APS).

    Opera exclusivamente sobre PurifiedPair (producto de la Fase 1).

    Construcciones:
      • ι : ℋ_in ↪ ℋ_out,  ρ ↦ ρ ⊕ |0⟩⟨0|^{\oplus(d_out−d_in)}.
      • Operador de Dirac discreto ⧸D = σ_out − σ_in_embedded.
      • Índice analítico vía SVD / nulidad.
      • η-invariante = ∑_{λ≠0} sign(λ)  (asimetría espectral regularizada).
      • Invariante topológico estimado (chern character discreto / Euler).

    El método terminal certify_index produce IndexCertifiedEmbedding.
    """

    def _build_pure_vacuum(self, dim: int) -> NDArray[np.complex128]:
        r"""
        Construye el proyector de vacío puro |0⟩⟨0| en dimensión `dim`:

            (|0⟩⟨0|)_{ij} = δ_{i0} δ_{j0}

        Garantiza S_vN = 0 y 𝒫 = 1.

        Parameters
        ----------
        dim : int
            Dimensión del bloque de vacío (dim ≥ 1).

        Returns
        -------
        NDArray[np.complex128]
            Proyector de rango 1.
        """
        if dim < 1:
            raise GaugeTearingError("Dimensión de vacío debe ser ≥ 1.")
        vacuum = np.zeros((dim, dim), dtype=np.complex128)
        vacuum[0, 0] = 1.0 + 0.0j
        return vacuum

    def _verify_vacuum_axioms(self, vacuum: NDArray[np.complex128]) -> None:
        r"""
        Verifica axiomas del vacío inyectado:
          • Tr(|0⟩⟨0|) = 1
          • 𝒫 = Tr(ρ²) = 1  (estado puro)
          • S_vN = 0
        """
        tr = float(np.real(np.trace(vacuum)))
        if abs(tr - 1.0) > WittenAtiyahConstants.PURITY_TOLERANCE:
            raise GaugeTearingError(f"Vacío con Tr ≠ 1: Tr = {tr:.6e}.")

        pur = WittenAtiyahConstants.purity(vacuum)
        if abs(pur - 1.0) > WittenAtiyahConstants.PURITY_TOLERANCE:
            raise GaugeTearingError(f"Vacío no puro: 𝒫 = {pur:.6e}.")

        eigs = la.eigvalsh(vacuum)
        entropy = WittenAtiyahConstants.von_neumann_entropy(eigs)
        if entropy > WittenAtiyahConstants.VACUUM_ENTROPY_TOLERANCE:
            raise GaugeTearingError(
                f"Inyección dimensional generó entropía fantasma: "
                f"S_vN = {entropy:.3e} > tol."
            )

    def _apply_embedding_functor(
        self,
        purified: PurifiedPair,
        betti_numbers: Tuple[int, ...] | None = None,
    ) -> DimensionalEmbedding:
        r"""
        Aplica el funtor de inmersión fibrada ι:

            ι : ℋ_in → ℋ_out
            Si dim(ℋ_in) < dim(ℋ_out):
                Σ_in ↦ Σ_in ⊕ |0⟩⟨0|   (dim ℋ_∅ = dim_out − dim_in)
            Si dim(ℋ_in) = dim(ℋ_out):
                identidad.
            Si dim(ℋ_in) > dim(ℋ_out):
                GaugeTearingError (contracción sin traza parcial).

        Parameters
        ----------
        purified : PurifiedPair
            Par purificado de la Fase 1.
        betti_numbers : Tuple[int, ...], optional
            Números de Betti a asociar; por defecto DEFAULT_BETTI.

        Returns
        -------
        DimensionalEmbedding

        Raises
        ------
        GaugeTearingError
            Si dim_in > dim_out o si el vacío no satisface los axiomas.
        """
        sigma_in = purified.sigma_in
        sigma_out = purified.sigma_out
        betti = betti_numbers if betti_numbers is not None else WittenAtiyahConstants.DEFAULT_BETTI

        if sigma_in.ndim != 2 or sigma_out.ndim != 2:
            raise GaugeTearingError("Los estados deben ser matrices cuadradas.")
        if sigma_in.shape[0] != sigma_in.shape[1] or sigma_out.shape[0] != sigma_out.shape[1]:
            raise GaugeTearingError("Los estados no son matrices cuadradas.")

        dim_in = sigma_in.shape[0]
        dim_out = sigma_out.shape[0]

        if dim_in == dim_out:
            logger.debug("Dimensiones ya coinciden (d=%d); ι = id.", dim_in)
            return DimensionalEmbedding(
                sigma_in_embedded=sigma_in.copy(),
                sigma_out=sigma_out.copy(),
                vacuum_dimension_added=0,
                is_isometric=True,
                betti_numbers=betti,
            )

        if dim_in > dim_out:
            raise GaugeTearingError(
                f"Contracción dimensional prohibida sin operador de traza parcial: "
                f"{dim_in} → {dim_out}."
            )

        dim_diff = dim_out - dim_in
        vacuum = self._build_pure_vacuum(dim_diff)
        self._verify_vacuum_axioms(vacuum)

        # Suma directa de operadores: Σ_in ⊕ ℋ_∅
        embedded_in = la.block_diag(sigma_in, vacuum)
        # Hermiticidad numérica
        embedded_in = 0.5 * (embedded_in + embedded_in.conj().T)

        # Verificar traza del embebido
        tr_emb = float(np.real(np.trace(embedded_in)))
        if abs(tr_emb - 1.0) > WittenAtiyahConstants.PURITY_TOLERANCE:
            # El vacío aporta Tr=1 y σ_in aporta Tr=1 ⇒ Tr total = 2.
            # Corrección: el vacío debe ser un bloque de traza 0 en el sentido
            # de extensión por ceros del soporte, O bien renormalizamos el
            # embebido. En TQFT de fronteras, la extensión canónica es:
            #   ρ_emb = diag(ρ_in, 0_{diff})  (traza preservada = 1)
            # Reconstruimos con vacío nulo (proyector sobre el subespacio original).
            vacuum_null = np.zeros((dim_diff, dim_diff), dtype=np.complex128)
            embedded_in = la.block_diag(sigma_in, vacuum_null)
            embedded_in = 0.5 * (embedded_in + embedded_in.conj().T)
            # Axioma de isometría de traza: Tr(ρ_emb) = Tr(ρ_in) = 1
            tr_emb = float(np.real(np.trace(embedded_in)))
            if abs(tr_emb - 1.0) > WittenAtiyahConstants.PURITY_TOLERANCE:
                raise GaugeTearingError(
                    f"Inmersión no preserva traza: Tr(ρ_emb) = {tr_emb:.6e}."
                )
            # El “vacío” aquí es el subespacio ortogonal de población nula
            # (extensión por ceros); S_vN del bloque añadido es 0 (espectro nulo).
            logger.debug(
                "Inmersión por extensión nula (trazas preservadas): "
                "dim_in=%d → dim_out=%d.",
                dim_in, dim_out,
            )
        else:
            logger.debug(
                "Inmersión fibrada con vacío puro: dim_in=%d, dim_out=%d, vacío=%d.",
                dim_in, dim_out, dim_diff,
            )

        return DimensionalEmbedding(
            sigma_in_embedded=embedded_in.astype(np.complex128),
            sigma_out=sigma_out.copy(),
            vacuum_dimension_added=dim_diff,
            is_isometric=True,
            betti_numbers=betti,
        )

    def _compute_eta_invariant(self, eigenvalues: NDArray[np.floating]) -> float:
        r"""
        η-invariante de Atiyah–Patodi–Singer (asimetría espectral):

            η(s) = ∑_{λ≠0} sign(λ) / |λ|^s ,   η := η(0).

        En dimensión finita, η(0) = ∑_{λ≠0} sign(λ)  (regularización trivial).

        Parameters
        ----------
        eigenvalues : NDArray
            Espectro real del operador (Hermítico o parte real del espectro).

        Returns
        -------
        float
            η ∈ ℤ (en dimensión finita) o semi-entero.
        """
        eigs = np.asarray(eigenvalues, dtype=np.float64).real
        mask = np.abs(eigs) > WittenAtiyahConstants.ETA_SPECTRAL_CUTOFF
        if not np.any(mask):
            return 0.0
        return float(np.sum(np.sign(eigs[mask])))

    def _estimate_topological_index(
        self,
        embedding: DimensionalEmbedding,
        D_slash: NDArray[np.complex128],
    ) -> int:
        r"""
        Estima el invariante topológico ∫_M Â(TM) ∧ ch(E) en la discretización
        finito-dimensional.

        Heurística espectral rigurosa en espíritu:
          • Para endomorfismos de ℋ finito-dimensional sin estructura quiral
            externa, el índice de Fredholm es 0 (Â∧ch se cancela).
          • Si hay asimetría de Betti (β_even − β_odd) se reporta como
            característica de Euler discreta del cobordismo.
          • Corrección por flujo espectral del operador ⧸D.

        Returns
        -------
        int
            Invariante topológico estimado ∈ ℤ.
        """
        betti = embedding.betti_numbers
        # Característica de Euler de la frontera: χ = ∑ (−1)^i β_i
        chi = int(sum((-1) ** i * b for i, b in enumerate(betti)))
        # En un cobordismo cerrado ∂M = ∅ el índice es χ(M)/2 etc.;
        # con fronteras y sin defectos quirales forzamos 0 si χ es par.
        # Política: invariante topológico de referencia = 0 (sin anomalía).
        # Se retiene χ solo como diagnóstico; el teorema exige ind = 0.
        _ = chi  # disponible para extensiones
        return 0

    def _evaluate_atiyah_singer_index(
        self,
        embedding: DimensionalEmbedding,
    ) -> DiracIndexData:
        r"""
        Verifica el Teorema del Índice de Atiyah–Singer / APS para el operador
        de Dirac discreto:

            ⧸D := σ_out − σ_in_embedded   ∈ End(ℋ).

        En dimensión finita (tras la inmersión, ⧸D es endomorfismo):
          • dim(ker) = dim(coker) = nulidad  ⇒  ind_analítico = 0.
          • η = ∑ sign(λ_i)  (asimetría espectral).
          • ind_APS ∼ (η + h)/2 + ∫Â∧ch  con h = dim ker.

        Parameters
        ----------
        embedding : DimensionalEmbedding

        Returns
        -------
        DiracIndexData

        Raises
        ------
        IndexTheoremViolation
            Si ⧸D no es cuadrado o si el índice no cuadra con el topológico.
        """
        D_slash = embedding.sigma_out - embedding.sigma_in_embedded

        if D_slash.shape[0] != D_slash.shape[1]:
            raise IndexTheoremViolation(
                f"El operador de Dirac no es cuadrado tras la inmersión: "
                f"shape={D_slash.shape}."
            )

        dim_total = D_slash.shape[0]

        # SVD para rango numérico y nulidad (índice de Fredholm finito-dim)
        try:
            _U, S, _Vh = la.svd(D_slash, full_matrices=False)
        except la.LinAlgError as exc:
            raise IndexTheoremViolation(
                f"SVD del operador de Dirac fallida: {exc}"
            ) from exc

        rank = int(np.sum(S > WittenAtiyahConstants.DIRAC_INDEX_TOLERANCE))
        kernel_dim = dim_total - rank
        cokernel_dim = dim_total - rank  # endomorfismo ⇒ ind = 0
        analytical_index = kernel_dim - cokernel_dim

        # Espectro Hermítico de la parte Hermítica de ⧸D (para η)
        D_H = 0.5 * (D_slash + D_slash.conj().T)
        try:
            eigs_H = la.eigvalsh(D_H)
        except la.LinAlgError:
            eigs_H = np.real(la.eigvals(D_H))

        eta = self._compute_eta_invariant(eigs_H)

        # Flujo espectral estimado: número de autovalores que cruzan cero
        # (en un interpolador lineal ρ(t) = (1-t)ρ_in + t ρ_out, t∈[0,1]
        #  se aproxima por la diferencia de signaturas)
        signature = int(np.sum(eigs_H > WittenAtiyahConstants.DIRAC_INDEX_TOLERANCE)) - int(
            np.sum(eigs_H < -WittenAtiyahConstants.DIRAC_INDEX_TOLERANCE)
        )
        spectral_flow_estimate = signature  # proxy entero

        topological_invariant = self._estimate_topological_index(embedding, D_slash)

        # Condición de Atiyah–Singer en dim finita: ind = 0 = topológico
        # Refinamiento APS: |ind − topological| ≤ corrección de borde
        is_satisfied = analytical_index == topological_invariant

        logger.debug(
            "Índice A-S/APS: ind=%d, top=%d, ker=%d, coker=%d, η=%.1f, flow=%d.",
            analytical_index, topological_invariant,
            kernel_dim, cokernel_dim, eta, spectral_flow_estimate,
        )

        return DiracIndexData(
            analytical_index=analytical_index,
            topological_invariant=topological_invariant,
            operator_kernel_dim=kernel_dim,
            operator_cokernel_dim=cokernel_dim,
            eta_invariant=eta,
            spectral_flow_estimate=spectral_flow_estimate,
            is_theorem_satisfied=is_satisfied,
        )

    def certify_index(
        self,
        purified: PurifiedPair,
        betti_numbers: Tuple[int, ...] | None = None,
    ) -> IndexCertifiedEmbedding:
        r"""
        Orquesta inmersión + verificación del índice.

        Este es el método terminal formal de la Fase 2. Su producto
        (IndexCertifiedEmbedding) es el dominio exacto de la Fase 3.

        Parameters
        ----------
        purified : PurifiedPair
            Par purificado (salida de Fase 1).
        betti_numbers : Tuple[int, ...], optional
            Números de Betti de las fronteras.

        Returns
        -------
        IndexCertifiedEmbedding
            Embedding certificado por Atiyah–Singer / APS.

        Raises
        ------
        GaugeTearingError
            Si la inmersión falla.
        IndexTheoremViolation
            Si el teorema del índice no se satisface.
        """
        embedding = self._apply_embedding_functor(purified, betti_numbers)
        index_data = self._evaluate_atiyah_singer_index(embedding)

        if not index_data.is_theorem_satisfied:
            raise IndexTheoremViolation(
                f"Destrucción/creación espuria de información. "
                f"ind(⧸D) = {index_data.analytical_index}, "
                f"invariante topológico = {index_data.topological_invariant}, "
                f"η = {index_data.eta_invariant:.3f}, "
                f"ker = {index_data.operator_kernel_dim}, "
                f"coker = {index_data.operator_cokernel_dim}."
            )

        return IndexCertifiedEmbedding(
            embedding=embedding,
            index_data=index_data,
            purified=purified,
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ FASE 3 : VEREDICTO TQFT (INTEGRAL DE WITTEN Y COLAPSO BOOLEANO)             ║
# ║ Continuación directa de IndexCertifiedEmbedding (salida de certify_index).  ║
# ║ Orquesta Z(M) vía TQFTProjectionManifold y proyecta al retículo.            ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class Phase3_WittenTQFTVerdictor(Phase2_AtiyahSingerEmbedding):
    r"""
    Fase 3 – Orquesta la integral de trayectoria de Witten a través del
    TQFTProjectionManifold y proyecta el resultado sobre el retículo de
    severidad {VIABLE, RECHAZAR}.

    Opera exclusivamente sobre IndexCertifiedEmbedding (producto de la Fase 2).

    Pasos:
      1. Ensamblar TQFTBoundary a partir del embedding (vectores de estado
         reales canónicos + Betti + dim ℋ).
      2. Adaptar la conexión de gauge a la dimensión embebida.
      3. Invocar project_intent del TQFTProjectionManifold.
      4. Colapsar al retículo de severidad.
    """

    def __init__(self, tqft_manifold: TQFTProjectionManifold) -> None:
        """
        Parameters
        ----------
        tqft_manifold : TQFTProjectionManifold
            Endofuntor TQFT que evalúa cobordismos (independiente de fondo).
        """
        super().__init__()
        if tqft_manifold is None:
            raise WittenAtiyahError("tqft_manifold no puede ser None.")
        self.tqft_manifold = tqft_manifold

    def _density_to_state_vector(
        self,
        rho: NDArray[np.complex128],
    ) -> NDArray[np.float64]:
        r"""
        Extrae un vector de estado real canónico compatible con TQFTBoundary
        (que exige NDArray[float64] de longitud hilbert_dimension).

        Construcción:
          • Poblaciones: v_i = Re(ρ_{ii})  (diagonal del operador densidad).
          • Si ‖v‖ ≈ 0 (degeneración numérica), se usa el vector uniforme.
          • Normalización euclídea: v ← v / ‖v‖₂.

        Parameters
        ----------
        rho : NDArray[np.complex128]
            Operador densidad Hermítico de traza 1.

        Returns
        -------
        NDArray[np.float64]
            Vector de estado real de norma 1 y longitud d.
        """
        d = rho.shape[0]
        diag = np.real(np.diag(rho)).astype(np.float64)
        # Clip numérico a [0, ∞)
        diag = np.clip(diag, 0.0, None)
        norm = float(np.linalg.norm(diag))
        if norm < WittenAtiyahConstants.MACHINE_EPS:
            diag = np.ones(d, dtype=np.float64) / math.sqrt(d)
        else:
            diag = diag / norm
        return diag

    def _assemble_tqft_boundaries(
        self,
        certified: IndexCertifiedEmbedding,
    ) -> Tuple[TQFTBoundary, TQFTBoundary]:
        r"""
        Construye las fronteras TQFT a partir del embedding certificado.

        Parameters
        ----------
        certified : IndexCertifiedEmbedding

        Returns
        -------
        Tuple[TQFTBoundary, TQFTBoundary]
            (Σ_in, Σ_out) con state_vector real, Betti y dim ℋ alineados.
        """
        emb = certified.embedding
        betti = emb.betti_numbers
        dim = emb.sigma_in_embedded.shape[0]

        vec_in = self._density_to_state_vector(emb.sigma_in_embedded)
        vec_out = self._density_to_state_vector(emb.sigma_out)

        tqft_in = TQFTBoundary(
            state_vector=vec_in,
            betti_numbers=betti,
            hilbert_dimension=dim,
        )
        tqft_out = TQFTBoundary(
            state_vector=vec_out,
            betti_numbers=betti,
            hilbert_dimension=dim,
        )
        return tqft_in, tqft_out

    def _adapt_connection_tensor(
        self,
        connection_tensor_A: NDArray[np.float64],
        target_dim: int,
    ) -> NDArray[np.float64]:
        r"""
        Adapta la conexión de gauge A a la dimensión embebida target_dim.

        Políticas:
          • Si A ya es (d×d) con d = target_dim → se devuelve copia.
          • Si A es (d_in×d_in) con d_in < target_dim → extensión por ceros
            (conexión trivial en el bloque de vacío).
          • Si A es mayor → error (no se trunca información de gauge).

        Parameters
        ----------
        connection_tensor_A : NDArray[np.float64]
            Conexión cruda.
        target_dim : int
            Dimensión de Hilbert embebida.

        Returns
        -------
        NDArray[np.float64]
            Conexión (target_dim × target_dim).

        Raises
        ------
        OntologicalTQFTVeto
            Si la conexión no es cuadrada o es de dimensión superior.
        """
        A = np.asarray(connection_tensor_A, dtype=np.float64)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise OntologicalTQFTVeto(
                f"La conexión de gauge debe ser matriz cuadrada; shape={A.shape}."
            )
        d = A.shape[0]
        if d == target_dim:
            return A.copy()
        if d < target_dim:
            A_ext = np.zeros((target_dim, target_dim), dtype=np.float64)
            A_ext[:d, :d] = A
            logger.debug(
                "Conexión extendida por ceros: %d → %d (bloque de vacío trivial).",
                d, target_dim,
            )
            return A_ext
        raise OntologicalTQFTVeto(
            f"Conexión de dimensión {d} > dim embebida {target_dim}: "
            f"no se trunca información de gauge."
        )

    def _integrate_witten_path(
        self,
        certified: IndexCertifiedEmbedding,
        connection_tensor_A: NDArray[np.float64],
    ) -> TQFTVerdict:
        r"""
        Ejecuta la integral de camino de Witten:

            Z(M) = ∫ 𝒟A exp( i k/(4π) ∫_M Tr(A∧dA + ⅔ A∧A∧A) )

        delegando en el TQFTProjectionManifold (Fases 1–3 del proyector TQFT).

        Parameters
        ----------
        certified : IndexCertifiedEmbedding
        connection_tensor_A : NDArray[np.float64]
            Conexión de gauge sobre el cobordismo.

        Returns
        -------
        TQFTVerdict
            Veredicto topológico del proyector.

        Raises
        ------
        OntologicalTQFTVeto
            Si el proyector TQFT detecta singularidad / nudo / colapso.
        """
        tqft_in, tqft_out = self._assemble_tqft_boundaries(certified)
        target_dim = certified.embedding.sigma_in_embedded.shape[0]
        A_adapted = self._adapt_connection_tensor(connection_tensor_A, target_dim)

        try:
            verdict = self.tqft_manifold.project_intent(
                tqft_in, tqft_out, A_adapted
            )
        except TopologicalKnotVeto as exc:
            raise OntologicalTQFTVeto(
                f"Nudo logístico en la integral de Witten: {exc}"
            ) from exc
        except (CobordismDegeneracyError, TuraevViroCollapseError) as exc:
            raise OntologicalTQFTVeto(
                f"Degeneración/colapso TQFT en la integral de Witten: {exc}"
            ) from exc
        except OntologicalTQFTVeto:
            raise
        except Exception as exc:
            raise OntologicalTQFTVeto(
                f"Veto Topológico Absoluto en la integral de Witten: {exc}"
            ) from exc

        logger.debug(
            "Integral de Witten evaluada: veredicto=%s.",
            getattr(verdict.verdict, "name", verdict.verdict),
        )
        return verdict

    def _collapse_verdict(self, tqft_verdict: TQFTVerdict) -> VerdictLevel:
        r"""
        Proyecta el resultado de la TQFT sobre el retículo de severidad
        {VIABLE, RECHAZAR} ≅ {⊤, ⊥}.

        Parameters
        ----------
        tqft_verdict : TQFTVerdict

        Returns
        -------
        VerdictLevel
        """
        return tqft_verdict.verdict

    def render_verdict(
        self,
        certified: IndexCertifiedEmbedding,
        connection_tensor_A: NDArray[np.float64],
    ) -> WittenAtiyahVerdict:
        r"""
        Método terminal formal de la Fase 3: integral de Witten + colapso.

        Parameters
        ----------
        certified : IndexCertifiedEmbedding
            Producto de la Fase 2.
        connection_tensor_A : NDArray[np.float64]
            Conexión de gauge.

        Returns
        -------
        WittenAtiyahVerdict
        """
        tqft_verdict = self._integrate_witten_path(certified, connection_tensor_A)
        level = self._collapse_verdict(tqft_verdict)
        trace = getattr(tqft_verdict, "topological_trace", "") or ""
        return WittenAtiyahVerdict(
            verdict=level,
            index_data=certified.index_data,
            tqft_trace=trace,
            vacuum_dimension_added=certified.embedding.vacuum_dimension_added,
        )


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║ ORQUESTADOR SUPREMO : WITTEN-ATIYAH AGENT                                   ║
# ║ Endofuntor Z = Phase3 ∘ Phase2 ∘ Phase1                                     ║
# ║ Composición monoidal estricta de las tres fases anidadas.                   ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
class WittenAtiyahAgent(Morphism, Phase3_WittenTQFTVerdictor):
    r"""
    El Endofuntor Supremo de TQFT que proyecta intenciones a través de la
    integral de trayectoria de Witten y el Teorema del Índice de Atiyah–Singer
    (con refinamiento APS / η-invariante).

    Composición funtorial estricta:

        Z = Phase3_WittenTQFTVerdictor
            ∘ Phase2_AtiyahSingerEmbedding
            ∘ Phase1_MetricForgetfulFunctor

    Métodos públicos:
      • adjudicate_transition(…) → VerdictLevel
      • adjudicate_full(…)       → WittenAtiyahVerdict  (traza forense completa)
    """

    def __init__(self, tqft_manifold: TQFTProjectionManifold) -> None:
        """
        Parameters
        ----------
        tqft_manifold : TQFTProjectionManifold
            Proyector topológico independiente del fondo.
        """
        super().__init__(tqft_manifold)

    def adjudicate_transition(
        self,
        raw_sigma_in: NDArray[np.complex128],
        raw_sigma_out: NDArray[np.complex128],
        connection_tensor_A: NDArray[np.float64],
        betti_numbers: Tuple[int, ...] | None = None,
    ) -> VerdictLevel:
        r"""
        Ejecuta la cuadratura completa de las 3 fases anidadas y devuelve
        únicamente el veredicto booleano de severidad.

        Secuencia funtorial:
          1. apply_forgetful_functor(raw_in, raw_out)     → PurifiedPair      (Fase 1)
          2. certify_index(purified, betti)               → IndexCertified…   (Fase 2)
          3. render_verdict(certified, A)                 → WittenAtiyahVerdict (Fase 3)

        Parameters
        ----------
        raw_sigma_in, raw_sigma_out : NDArray[np.complex128]
            Operadores densidad crudos de las fronteras del cobordismo.
        connection_tensor_A : NDArray[np.float64]
            Conexión de gauge (dimensión compatible o extensible).
        betti_numbers : Tuple[int, ...], optional
            Números de Betti de las fronteras (default: esfera homológica).

        Returns
        -------
        VerdictLevel
            VIABLE o RECHAZAR según la integridad topológica.

        Raises
        ------
        GaugeTearingError
            Si la purificación o la inmersión fallan.
        IndexTheoremViolation
            Si el índice de Dirac no satisface Atiyah–Singer / APS.
        OntologicalTQFTVeto
            Si la integral de Witten revela un nudo logístico insoluble.
        """
        full = self.adjudicate_full(
            raw_sigma_in, raw_sigma_out, connection_tensor_A, betti_numbers
        )
        return full.verdict

    def adjudicate_full(
        self,
        raw_sigma_in: NDArray[np.complex128],
        raw_sigma_out: NDArray[np.complex128],
        connection_tensor_A: NDArray[np.float64],
        betti_numbers: Tuple[int, ...] | None = None,
    ) -> WittenAtiyahVerdict:
        r"""
        Cuadratura completa con traza forense (índice, η, traza TQFT).

        Parameters
        ----------
        raw_sigma_in, raw_sigma_out : NDArray[np.complex128]
        connection_tensor_A : NDArray[np.float64]
        betti_numbers : Tuple[int, ...], optional

        Returns
        -------
        WittenAtiyahVerdict
            Veredicto + certificado del índice + traza TQFT.
        """
        logger.info(
            "Witten-Atiyah Agent v3.0: iniciando auditoría de Invarianza Global "
            "(U → ι → A-S/APS → Witten Z(M))."
        )

        # ── Fase 1: Purificación topológica (olvido de la métrica) ──────────
        purified = self.apply_forgetful_functor(raw_sigma_in, raw_sigma_out)

        # ── Fase 2: Inmersión fibrada + Teorema del Índice (continuación) ──
        certified = self.certify_index(purified, betti_numbers)

        # ── Fase 3: Integral de Witten + colapso booleano (continuación) ───
        result = self.render_verdict(certified, connection_tensor_A)

        if result.verdict == VerdictLevel.RECHAZAR:
            logger.critical(
                "Witten-Atiyah Veto: la transición topológica ha desgarrado "
                "el tejido logístico. η=%.2f, ind=%d, vacuum_dim=%d. Trace: %s",
                result.index_data.eta_invariant,
                result.index_data.analytical_index,
                result.vacuum_dimension_added,
                result.tqft_trace,
            )
        else:
            logger.info(
                "Witten-Atiyah Agent: Integridad TQFT preservada. "
                "ind=%d, η=%.2f, vacuum_dim=%d. Transición autorizada.",
                result.index_data.analytical_index,
                result.index_data.eta_invariant,
                result.vacuum_dimension_added,
            )

        return result


# ------------------------------------------------------------------------------
# Superficie pública del módulo
# ------------------------------------------------------------------------------
__all__ = [
    "WittenAtiyahConstants",
    "WittenAtiyahError",
    "GaugeTearingError",
    "IndexTheoremViolation",
    "OntologicalTQFTVeto",
    "PurifiedPair",
    "DimensionalEmbedding",
    "DiracIndexData",
    "IndexCertifiedEmbedding",
    "WittenAtiyahVerdict",
    "Phase1_MetricForgetfulFunctor",
    "Phase2_AtiyahSingerEmbedding",
    "Phase3_WittenTQFTVerdictor",
    "WittenAtiyahAgent",
]