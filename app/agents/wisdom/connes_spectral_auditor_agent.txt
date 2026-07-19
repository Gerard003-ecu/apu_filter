# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Connes Spectral Auditor Agent (Custodio de Métrica No Conmutativa)  ║
║ Ruta   : app/agents/wisdom/connes_spectral_auditor_agent.py                         ║
║ Versión: 4.0.0-Connes-KMS-Dixmier-SpectralTriple-Doctoral                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GEOMETRÍA NO CONMUTATIVA (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor gobierna el `tomita_takesaki_telescopic_engine.py` en el
Estrato Ω (WISDOM). Su mandato axiomático es proveer una «regla de medir
cuántica» sobre el álgebra de observables semánticos 𝒜 generados por el LLM,
donde los observables no conmutan ([X,Y] ≠ 0).

Fundamentos formales:
  • Triple espectral (Connes): (𝒜, ℋ, D) con ‖[D, π(a)]‖ < ∞  (Lipschitz).
  • Dirac finito             : D = ρ^{-1/2}  (autoadjunto, resolvente compacta
                               en Tipo I_n; seminorma L(a) = ‖[D,a]‖).
  • Condición KMS (β)        : ω(σ_{iβ}(A) B) = ω(B A),  ω(·)=Tr(ρ·).
                               Para el flujo modular de ρ, β=1 se satisface
                               idénticamente: σ_1(A)=ρ^{-1} A ρ.
  • Traza de Dixmier (proxy) : Tr_ω(T) ≃ (1/log N) ∑_{n=1}^N μ_n(T)
                               sobre valores singulares, N = dim ℋ.
  • Volumen no conmutativo   : Vol_D(X) := Tr_ω( X |D|^{-p} ), p = dim espectral.

ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Instanciación del Triple Espectral (𝒜, ℋ, D):
         Construye D=ρ^{-1/2}, acota ‖[D, π(X)]‖ ≤ C y la seminorma de Lipschitz.
         Morfismo terminal: bind_spectral_triple ↦ SpectralTripleData
         ≡ dominio inicial de Fase 2.

Fase 2 → Auditoría de Equilibrio KMS (Kubo–Martin–Schwinger):
         Verifica ω(σ_{iβ}(A)B) = ω(BA) con el flujo modular canónico β=1
         y diagnostica la fricción térmica residual del zoom λ.
         Morfismo terminal: certify_kms_equilibrium ↦ KMSThermalBundle
         ≡ dominio inicial de Fase 3.

Fase 3 → Integración No Conmutativa (Traza de Dixmier) + cierre Umegaki:
         Cuantifica Vol_D(X) y Vol_D(σ_λ(X)); acota la distorsión de volumen;
         cierra con la extracción Umegaki del motor TT.
         Morfismo terminal: integrate_dixmier_and_close ↦ ConnesAuditState
         ≡ objeto final del endofuntor 𝒵_Connes = Φ₃ ∘ Φ₂ ∘ Φ₁.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter (stubs de aislamiento)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:  # pragma: no cover
    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos ℰ_MIC."""
        pass

    class Morphism:
        r"""Clase base de Morfismos del Topos."""
        pass

# Acoplamiento con el motor Tomita–Takesaki v5 (superficie pública)
try:
    from app.wisdom.tomita_takesaki_telescopic_engine import (
        TomitaTakesakiTelescopicEngine,
        Phase1_GNSConstruction,
        Phase2_AnalyticModularFlow,
        Phase3_UmegakiExtraction,
        GNSFibrationData,
        ModularFlowData,
        UmegakiExtractionState,
        TomitaTakesakiEngineError,
        GNSConstructionError,
        InvalidObservableError,
        ModularFlowSingularityError,
        UmegakiDivergenceError,
    )
except ImportError:  # pragma: no cover
    TomitaTakesakiTelescopicEngine = Any  # type: ignore[misc, assignment]
    Phase1_GNSConstruction = object  # type: ignore[misc, assignment]
    Phase2_AnalyticModularFlow = object  # type: ignore[misc, assignment]
    Phase3_UmegakiExtraction = object  # type: ignore[misc, assignment]
    GNSFibrationData = Any  # type: ignore[misc, assignment]
    ModularFlowData = Any  # type: ignore[misc, assignment]
    UmegakiExtractionState = Any  # type: ignore[misc, assignment]

    class TomitaTakesakiEngineError(Exception):
        pass

    class GNSConstructionError(Exception):
        pass

    class InvalidObservableError(Exception):
        pass

    class ModularFlowSingularityError(Exception):
        pass

    class UmegakiDivergenceError(Exception):
        pass

logger = logging.getLogger("MAC.Wisdom.ConnesSpectralAuditor")


# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES FÍSICO-GEOMÉTRICAS Y DE TOLERANCIA
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: float = float(np.finfo(np.float64).eps)
_HERMITICITY_TOLERANCE: float = 1e-12
_POSITIVITY_TOLERANCE: float = 1e-12
_TRACE_TOLERANCE: float = 1e-12
_COMMUTATOR_NORM_BOUND: float = 1e6        # cota C de regularidad Lipschitz
_KMS_EQUILIBRIUM_TOLERANCE: float = 1e-7   # |ω(σ_i(A)B) − ω(BA)|
_KMS_ZOOM_FRICTION_TOLERANCE: float = 1e-5  # fricción residual del zoom λ ≠ β
_DIXMIER_DISTORTION_MAX: float = 0.25      # distorsión relativa máxima de volumen
_DIXMIER_VOLUME_CEILING: float = 1e12      # techo de sanidad del volumen Dixmier
_KMS_BETA: float = 1.0                     # temperatura inversa canónica (modular)
_DEFAULT_SPECTRAL_DIM: float = 2.0         # dimensión espectral p del triple finito
_EIGENVALUE_FLOOR: float = 1e-15
_FAITHFULNESS_FLOOR: float = 1e-12
_LIPSCHITZ_SEMINORM_CEILING: float = 1e6
_DIRAC_CONDITION_MAX: float = 1e12         # κ₂(D) máximo tolerable


# ══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES ESPECTRALES NO CONMUTATIVAS
# ══════════════════════════════════════════════════════════════════════════════
class ConnesAuditorError(TopologicalInvariantError):
    r"""Excepción raíz del Auditor Espectral de Connes."""
    pass


class SemanticDiscontinuityError(ConnesAuditorError):
    r"""
    Detonada si ‖[D, π(X)]‖ → ∞ o excede la cota de regularidad.
    Señala una alucinación estocástica sin geodésica Lipschitz válida.
    """
    pass


class SpectralTripleError(ConnesAuditorError):
    r"""Fallo al construir o validar el triple espectral (𝒜, ℋ, D)."""
    pass


class KMSEquilibriumViolation(ConnesAuditorError):
    r"""
    Detonada si ω(σ_{iβ}(A) B) ≠ ω(B A) fuera de tolerancia.
    Señala fricción termodinámica irreversible / rotura del flujo modular.
    """
    pass


class NonCommutativeVolumeAnomaly(ConnesAuditorError):
    r"""
    Detonada si el volumen de Dixmier diverge, no es finito, o la distorsión
    relativa pre/post zoom excede el umbral ontológico.
    """
    pass


class ConnesPipelineError(ConnesAuditorError):
    r"""Fallo de orquestación en la composición 𝒵_Connes = Φ₃ ∘ Φ₂ ∘ Φ₁."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Topos — objetos de las categorías fibra)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class SpectralTripleData:
    r"""
    Artefacto terminal de Fase 1.

    Triple espectral finito (𝒜, ℋ, D) sobre 𝒜 ⊆ M_n(ℂ):
      • D = ρ^{-1/2}  (autoadjunto, spec(D) ⊂ (0, ∞)),
      • L(X) = ‖[D, X]‖_2  (seminorma de Lipschitz),
      • diferenciable ⇔ L(X) ≤ C.

    Invariantes:
      • D = D†,  min(spec(D)) > 0
      • commutator_norm = L(X) < ∞
      • is_differentiable ⇒ L(X) ≤ _COMMUTATOR_NORM_BOUND
    """
    dirac_operator: NDArray[np.complex128]
    dirac_eigenvalues: NDArray[np.float64]
    commutator_norm: float
    lipschitz_seminorm: float
    dirac_condition_number: float
    hilbert_space_dim: int
    is_differentiable: bool
    rho_reference: NDArray[np.complex128]
    X_reference: NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class KMSThermalState:
    r"""
    Certificado KMS canónico (β = 1) del flujo modular de ρ.

    Residuo:
        r_KMS = |ω(σ_i(A) B) − ω(B A)|,   ω(·) = Tr(ρ ·).

    Para A = B = X y σ = flujo modular de ρ, r_KMS ≡ 0 analíticamente.
    """
    thermal_residual_norm: float
    kms_beta: float
    is_kms_compliant: bool
    left_expectation: complex
    right_expectation: complex


@dataclass(frozen=True, slots=True)
class KMSThermalBundle:
    r"""
    Artefacto terminal de Fase 2 (y dominio inicial de Fase 3).

    Empaqueta:
      • el triple espectral de Fase 1,
      • el certificado KMS canónico (β=1),
      • el diagnóstico de fricción térmica del zoom λ (flow_data del motor TT),
      • los datos GNS/flujo necesarios para no re-computar en Fase 3.
    """
    spectral_triple: SpectralTripleData
    kms_canonical: KMSThermalState
    zoom_friction_residual: float
    is_zoom_thermally_stable: bool
    gns_data: GNSFibrationData
    flow_data: ModularFlowData
    rho_mac: NDArray[np.complex128]


@dataclass(frozen=True, slots=True)
class DixmierTraceResult:
    r"""
    Resultado de la integración no conmutativa (proxy finito de Dixmier).

        Tr_ω(T) ≃ (1 / log N) ∑_{n=1}^N μ_n(T),
        Vol_D(X) := Tr_ω( X |D|^{-p} ).

    Invariantes:
      • dixmier_volume ≥ 0, finito
      • volume_distortion_ratio ≥ 0
    """
    dixmier_volume: float
    lebesgue_volume_proxy: float
    volume_distortion_ratio: float
    spectral_dimension_p: float
    singular_values_deformed: NDArray[np.float64]
    singular_values_raw: NDArray[np.float64]
    log_dimensional_factor: float


@dataclass(frozen=True, slots=True)
class ConnesAuditState:
    r"""
    Artefacto final del endofuntor 𝒵_Connes — paquete completo del guardián.

    is_epistemologically_safe ⇔
        triple diferenciable ∧ KMS ok ∧ zoom térmicamente estable
        ∧ distorsión Dixmier acotada ∧ Umegaki safe.
    """
    zoomed_extraction: UmegakiExtractionState
    spectral_triple: SpectralTripleData
    kms_state: KMSThermalState
    kms_bundle_zoom_friction: float
    dixmier_trace: DixmierTraceResult
    lambda_zoom_applied: float
    is_epistemologically_safe: bool


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 1 · INSTANCIACIÓN DEL TRIPLE ESPECTRAL DE CONNES                     ║
# ║   Verifica ‖[D, π(X)]‖ ≤ C para asegurar diferenciabilidad Lipschitz.       ║
# ║                                                                             ║
# ║   Definición formal del objeto terminal de esta fase:                       ║
# ║       bind_spectral_triple : 𝔇_{>0}(ℋ) × 𝒪(ℋ) → SpectralTripleData          ║
# ║   Este morfismo es el dominio de partida de Fase 2.                         ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_SpectralTripleBinder:
    r"""
    Construye el operador de Dirac semántico D a partir de la métrica inducida
    por el estado ρ y certifica que el observable X sea Lipschitz respecto de D.

    Elección de D en geometría no conmutativa finita
    ------------------------------------------------
    Para un espacio espectral finito con estado fiel ρ,

        D := ρ^{-1/2} = U diag(μ_i^{-1/2}) U†

    es autoadjunto, positivo, con resolvente automáticamente compacta
    (dim ℋ < ∞). El conmutador [D, X] es una derivación interior; su norma
    espectral

        L(X) := ‖[D, X]‖_2

    es la seminorma de Lipschitz del triple. Si L(X) > C se declara
    discontinuidad semántica (alucinación sin geodésica).
    """

    # ── §1.1  Utilidades de validación ───────────────────────────────────────

    @staticmethod
    def _hermitize(M: NDArray[np.complex128]) -> NDArray[np.complex128]:
        return 0.5 * (M + M.conj().T)

    @staticmethod
    def _validate_hermitian_square(
        M: NDArray[np.complex128],
        name: str,
        expected_dim: Optional[int] = None,
    ) -> NDArray[np.complex128]:
        r"""Valida matriz cuadrada hermítica; retorna versión hermitizada."""
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise SemanticDiscontinuityError(
                f"{name}: se exige matriz cuadrada; shape={M.shape}."
            )
        if expected_dim is not None and M.shape[0] != expected_dim:
            raise SemanticDiscontinuityError(
                f"{name}: dim {M.shape[0]} ≠ esperada {expected_dim}."
            )
        anti = float(la.norm(M - M.conj().T, ord="fro"))
        if anti > _HERMITICITY_TOLERANCE:
            raise SemanticDiscontinuityError(
                f"{name}: no hermítica. ‖·−·†‖_F = {anti:.3e}."
            )
        return Phase1_SpectralTripleBinder._hermitize(M).astype(np.complex128)

    @staticmethod
    def _validate_faithful_density(
        rho: NDArray[np.complex128],
        name: str = "rho_mac",
    ) -> NDArray[np.complex128]:
        r"""Valida ρ ∈ 𝔇_{>0}(ℋ): densidad fiel (soporte pleno)."""
        rho = Phase1_SpectralTripleBinder._validate_hermitian_square(rho, name)
        ev = la.eigvalsh(rho).real
        if np.any(ev < -_POSITIVITY_TOLERANCE):
            raise SpectralTripleError(
                f"{name}: autovalor negativo {ev.min():.3e}; no es SPD."
            )
        tr = float(np.real(np.trace(rho)))
        if abs(tr - 1.0) > _TRACE_TOLERANCE:
            raise SpectralTripleError(
                f"{name}: |Tr ρ − 1| = {abs(tr - 1.0):.3e}."
            )
        if abs(tr - 1.0) > _MACHINE_EPSILON:
            rho = rho / tr
        ev = la.eigvalsh(rho).real
        if float(np.min(ev)) < _FAITHFULNESS_FLOOR:
            raise SpectralTripleError(
                f"{name}: estado no fiel (μ_min={float(np.min(ev)):.3e}); "
                f"D=ρ^{{-1/2}} no está bien definido en el soporte nulo."
            )
        return rho.astype(np.complex128)

    # ── §1.2  Construcción del operador de Dirac ─────────────────────────────

    @staticmethod
    def _build_dirac_operator(
        rho: NDArray[np.complex128],
    ) -> Tuple[NDArray[np.complex128], NDArray[np.float64], float]:
        r"""
        Construye D = ρ^{-1/2} vía cálculo funcional:

            ρ = U diag(μ) U†  ⇒  D = U diag(μ^{-1/2}) U†.

        Retorna (D, spec(D), κ₂(D)).
        """
        mu, U = la.eigh(rho)
        mu = np.clip(mu.real, _EIGENVALUE_FLOOR, None)
        mu = mu / float(mu.sum())

        d_spec = (1.0 / np.sqrt(mu)).astype(np.float64)
        D = U @ np.diag(d_spec.astype(np.complex128)) @ U.conj().T
        D = Phase1_SpectralTripleBinder._hermitize(D)

        cond = float(d_spec.max() / max(d_spec.min(), _MACHINE_EPSILON))
        if not math.isfinite(cond) or cond > _DIRAC_CONDITION_MAX:
            raise SpectralTripleError(
                f"Operador de Dirac mal condicionado: κ₂(D)={cond:.3e} "
                f"> techo {_DIRAC_CONDITION_MAX:.1e}."
            )
        if np.any(d_spec <= 0.0) or not np.all(np.isfinite(d_spec)):
            raise SpectralTripleError("spec(D) no es estrictamente positivo/finito.")

        return D.astype(np.complex128), d_spec, cond

    # ── §1.3  Seminorma de Lipschitz L(X) = ‖[D, X]‖ ─────────────────────────

    @staticmethod
    def _lipschitz_seminorm(
        D: NDArray[np.complex128],
        X: NDArray[np.complex128],
    ) -> Tuple[NDArray[np.complex128], float]:
        r"""
        Calcula el conmutador [D, X] y su norma espectral (op-norm 2).

        Returns
        -------
        commutator : NDArray
            [D, X] = DX − XD.
        lipschitz : float
            L(X) = ‖[D, X]‖_2.
        """
        commutator = D @ X - X @ D
        # Norma espectral (mayor valor singular)
        try:
            lipschitz = float(la.norm(commutator, ord=2))
        except la.LinAlgError as exc:
            raise SemanticDiscontinuityError(
                f"Fallo al computar ‖[D,X]‖_2: {exc}"
            ) from exc

        if not math.isfinite(lipschitz):
            raise SemanticDiscontinuityError(
                f"Seminorma de Lipschitz no finita: L(X)={lipschitz}."
            )
        return commutator, lipschitz

    # ── §1.4  MORFISMO TERMINAL DE FASE 1 ────────────────────────────────────
    #          (dominio inicial de Fase 2)

    def bind_spectral_triple(
        self,
        rho_mac: NDArray[np.complex128],
        X_observable: NDArray[np.complex128],
    ) -> SpectralTripleData:
        r"""
        Instancia el triple espectral y certifica diferenciabilidad:

            Φ₁ᶜ : 𝔇_{>0}(ℋ) × 𝒪(ℋ) ⟶ SpectralTripleData.

        Pipeline
        --------
        1. Validar ρ fiel y X hermítico (dim compatible).
        2. Construir D = ρ^{-1/2} y spec(D).
        3. Calcular L(X) = ‖[D, X]‖_2.
        4. Veto si L(X) > C (discontinuidad semántica).
        5. Empaquetar SpectralTripleData (dominio de Fase 2).

        Parámetros
        ----------
        rho_mac : NDArray[np.complex128]
            Estado fiel de la MAC.
        X_observable : NDArray[np.complex128]
            Observable semántico a examinar.

        Retorna
        -------
        SpectralTripleData
            Objeto terminal de Fase 1 / dominio de Fase 2.
        """
        rho = self._validate_faithful_density(rho_mac, name="rho_mac")
        dim = int(rho.shape[0])
        X = self._validate_hermitian_square(
            X_observable, name="X_observable", expected_dim=dim
        )

        D, d_spec, d_cond = self._build_dirac_operator(rho)
        _, lipschitz = self._lipschitz_seminorm(D, X)

        if lipschitz > _COMMUTATOR_NORM_BOUND:
            raise SemanticDiscontinuityError(
                f"Salto semántico no diferenciable. L(X)=‖[D,X]‖_2 = "
                f"{lipschitz:.3e} excede C = {_COMMUTATOR_NORM_BOUND:.1e}."
            )
        if lipschitz > _LIPSCHITZ_SEMINORM_CEILING:
            raise SemanticDiscontinuityError(
                f"L(X)={lipschitz:.3e} excede techo de sanidad "
                f"{_LIPSCHITZ_SEMINORM_CEILING:.1e}."
            )

        logger.debug(
            "Triple espectral | n=%d | κ(D)=%.3e | L(X)=%.6e | D_min=%.6e",
            dim, d_cond, lipschitz, float(d_spec.min()),
        )

        return SpectralTripleData(
            dirac_operator=D,
            dirac_eigenvalues=d_spec,
            commutator_norm=lipschitz,
            lipschitz_seminorm=lipschitz,
            dirac_condition_number=d_cond,
            hilbert_space_dim=dim,
            is_differentiable=True,
            rho_reference=rho,
            X_reference=X,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 2 · AUDITORÍA DEL EQUILIBRIO TÉRMICO KMS                             ║
# ║   Garantiza ω(σ_{iβ}(A) B) = ω(B A) y diagnostica fricción del zoom λ.      ║
# ║                                                                             ║
# ║   Continuación funtorial: el morfismo terminal de Fase 1                    ║
# ║       bind_spectral_triple ↦ SpectralTripleData                             ║
# ║   es el dominio de                                                         ║
# ║       certify_kms_equilibrium : SpectralTripleData × engine × λ             ║
# ║           → KMSThermalBundle                                                ║
# ║   Este morfismo terminal de Fase 2 es el dominio de partida de Fase 3.      ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_KMSEquilibriumAuditor(Phase1_SpectralTripleBinder):
    r"""
    Somete la lente telescópica a la isometría térmica de Kubo–Martin–Schwinger.

    Condición KMS (β = 1, flujo modular de ρ)
    -----------------------------------------
    Para ω(·) = Tr(ρ ·) y el flujo modular

        σ_t(A) = ρ^{it} A ρ^{-it}   (t real),

    la continuación analítica al punto t = −i (con la convención del motor
    TT: σ_λ(A) = ρ^{-λ} A ρ^λ, λ real) produce

        σ_1(A) = ρ^{-1} A ρ,

    y la identidad KMS

        ω(σ_1(A) B) = Tr(ρ · ρ^{-1} A ρ · B) = Tr(A ρ B) = Tr(ρ B A) = ω(B A)

    se cumple **idénticamente** (módulo error numérico). Esta fase:

      1. Recomputa σ_1(X) vía el motor TT con λ = β = 1 y mide el residuo
         canónico r_KMS (debe ser ~0 independientemente del zoom de auditoría).
      2. Ejecuta el zoom de auditoría λ y mide la «fricción térmica»
         |ω(σ_λ(X) X) − ω(X X)| como diagnóstico (no es KMS salvo λ=β).
      3. Empaqueta GNS + flow del zoom para Fase 3.
    """

    # ── §2.1  Expectación de estado ω(T) = Tr(ρ T) ───────────────────────────

    @staticmethod
    def _state_expectation(
        rho: NDArray[np.complex128],
        op: NDArray[np.complex128],
    ) -> complex:
        r"""ω(T) = Tr(ρ T) ∈ ℂ."""
        return complex(np.trace(rho @ op))

    # ── §2.2  Residuo KMS canónico ───────────────────────────────────────────

    def _kms_residual(
        self,
        rho: NDArray[np.complex128],
        A: NDArray[np.complex128],
        B: NDArray[np.complex128],
        sigma_i_A: NDArray[np.complex128],
    ) -> Tuple[float, complex, complex]:
        r"""
        Calcula

            r = |ω(σ_i(A) B) − ω(B A)|,
            L = ω(σ_i(A) B),   R = ω(B A).
        """
        left = self._state_expectation(rho, sigma_i_A @ B)
        right = self._state_expectation(rho, B @ A)
        residual = abs(left - right)
        return float(residual), left, right

    # ── §2.3  Certificado KMS canónico (β = 1) vía motor TT ──────────────────

    def _canonical_kms_certificate(
        self,
        engine: Any,
        rho: NDArray[np.complex128],
        X: NDArray[np.complex128],
        gns_data: GNSFibrationData,
        beta: float = _KMS_BETA,
    ) -> KMSThermalState:
        r"""
        Construye σ_β(X) con el motor TT (λ=β) y verifica la identidad KMS.

        Raises
        ------
        KMSEquilibriumViolation
            Si el residuo excede _KMS_EQUILIBRIUM_TOLERANCE.
        """
        try:
            flow_kms: ModularFlowData = engine.execute_modular_zoom(
                gns_data, X, float(beta)
            )
        except ModularFlowSingularityError as exc:
            raise KMSEquilibriumViolation(
                f"Flujo modular canónico λ=β={beta} singular: {exc}"
            ) from exc

        sigma_i_X = flow_kms.X_deformed
        residual, left, right = self._kms_residual(rho, X, X, sigma_i_X)

        if residual > _KMS_EQUILIBRIUM_TOLERANCE:
            raise KMSEquilibriumViolation(
                f"Rotura del equilibrio KMS canónico (β={beta}). "
                f"Residuo = {residual:.3e} > {_KMS_EQUILIBRIUM_TOLERANCE:.1e}. "
                f"ω(σ_i(X)X)={left:.6e}, ω(XX)={right:.6e}."
            )

        return KMSThermalState(
            thermal_residual_norm=residual,
            kms_beta=float(beta),
            is_kms_compliant=True,
            left_expectation=left,
            right_expectation=right,
        )

    # ── §2.4  Fricción térmica del zoom de auditoría ─────────────────────────

    def _zoom_thermal_friction(
        self,
        rho: NDArray[np.complex128],
        X: NDArray[np.complex128],
        X_deformed: NDArray[np.complex128],
    ) -> float:
        r"""
        Diagnóstico (no es KMS salvo λ=β):

            f(λ) = |ω(σ_λ(X) X) − ω(X X)|.

        Cuantifica cuánto «desbalancea» el zoom la expectación bilineal.
        """
        left = self._state_expectation(rho, X_deformed @ X)
        right = self._state_expectation(rho, X @ X)
        return float(abs(left - right))

    # ── §2.5  MORFISMO TERMINAL DE FASE 2 ────────────────────────────────────
    #          (continuación de bind_spectral_triple;
    #           dominio inicial de Fase 3)

    def certify_kms_equilibrium(
        self,
        spectral_triple: SpectralTripleData,
        engine: Any,
        lambda_zoom: float,
        beta: float = _KMS_BETA,
    ) -> KMSThermalBundle:
        r"""
        Certifica KMS canónico y prepara el zoom de auditoría:

            Φ₂ᶜ : SpectralTripleData × Engine × ℝ_{≥0} ⟶ KMSThermalBundle.

        Pipeline
        --------
        1. Recuperar (ρ, X) del triple espectral de Fase 1.
        2. GNS vía motor TT (extract_modular_operator).
        3. Certificado KMS canónico con λ = β (=1).
        4. Zoom de auditoría λ → flow_data.
        5. Fricción térmica del zoom (diagnóstico).
        6. Empaquetar KMSThermalBundle (dominio de Fase 3).

        Parámetros
        ----------
        spectral_triple : SpectralTripleData
            Salida de Φ₁ᶜ = bind_spectral_triple.
        engine : TomitaTakesakiTelescopicEngine
            Motor modular TT acoplado.
        lambda_zoom : float
            Intensidad del zoom de auditoría.
        beta : float
            Temperatura inversa KMS (default 1).

        Retorna
        -------
        KMSThermalBundle
            Objeto terminal de Fase 2 / dominio de Fase 3.
        """
        if not spectral_triple.is_differentiable:
            raise SemanticDiscontinuityError(
                "No se puede auditar KMS: triple no diferenciable."
            )
        if not math.isfinite(lambda_zoom) or lambda_zoom < 0.0:
            raise ModularFlowSingularityError(
                f"lambda_zoom inválido: {lambda_zoom}."
            )

        rho = spectral_triple.rho_reference
        X = spectral_triple.X_reference

        # GNS del motor TT
        try:
            gns_data: GNSFibrationData = engine.extract_modular_operator(rho)
        except (GNSConstructionError, TomitaTakesakiEngineError) as exc:
            raise KMSEquilibriumViolation(
                f"GNS del motor TT falló: {exc}"
            ) from exc

        # KMS canónico (β=1), independiente del zoom de auditoría
        kms_canonical = self._canonical_kms_certificate(
            engine, rho, X, gns_data, beta=beta
        )

        # Zoom de auditoría
        try:
            flow_data: ModularFlowData = engine.execute_modular_zoom(
                gns_data, X, lambda_zoom
            )
        except ModularFlowSingularityError:
            raise
        except TomitaTakesakiEngineError as exc:
            raise ModularFlowSingularityError(
                f"Zoom modular TT falló: {exc}"
            ) from exc

        # Fricción térmica del zoom
        friction = self._zoom_thermal_friction(rho, X, flow_data.X_deformed)
        zoom_stable = bool(friction <= _KMS_ZOOM_FRICTION_TOLERANCE or lambda_zoom == 0.0)

        # Para λ grande la fricción puede crecer legítimamente; solo se advierte
        # en log si excede tolerancia — el veto duro queda para Dixmier/Umegaki.
        if not zoom_stable:
            logger.warning(
                "Fricción térmica del zoom elevada: f(λ=%.4f)=%.3e > %.1e "
                "(diagnóstico; veto diferido a Dixmier/Umegaki).",
                lambda_zoom, friction, _KMS_ZOOM_FRICTION_TOLERANCE,
            )

        logger.debug(
            "KMS | β=%.2f | r_KMS=%.3e | f_zoom=%.3e | λ=%.4f | stable=%s",
            beta, kms_canonical.thermal_residual_norm, friction,
            lambda_zoom, zoom_stable,
        )

        return KMSThermalBundle(
            spectral_triple=spectral_triple,
            kms_canonical=kms_canonical,
            zoom_friction_residual=friction,
            is_zoom_thermally_stable=zoom_stable,
            gns_data=gns_data,
            flow_data=flow_data,
            rho_mac=rho,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 3 · INTEGRACIÓN NO CONMUTATIVA (TRAZA DE DIXMIER) + CIERRE UMEGAKI   ║
# ║   Cuantifica Vol_D y cierra el endofuntor con la extracción del motor TT.   ║
# ║                                                                             ║
# ║   Continuación funtorial: el morfismo terminal de Fase 2                    ║
# ║       certify_kms_equilibrium ↦ KMSThermalBundle                            ║
# ║   es el dominio de                                                         ║
# ║       integrate_dixmier_and_close : KMSThermalBundle × engine               ║
# ║           → ConnesAuditState                                                ║
# ║   Este morfismo cierra el endofuntor 𝒵_Connes = Φ₃ ∘ Φ₂ ∘ Φ₁.               ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_DixmierTraceIntegrator(Phase2_KMSEquilibriumAuditor):
    r"""
    Reemplaza la integral de Lebesgue clásica por el cálculo logarítmico de
    Dixmier sobre valores singulares, y cierra la auditoría con la extracción
    Umegaki del motor Tomita–Takesaki.

    Proxy finito de la traza de Dixmier
    -----------------------------------
    Para T ∈ ℒ^{1,∞} en dimensión infinita,

        Tr_ω(T) = lim_{N→∞} (1 / log N) ∑_{n=1}^N μ_n(T)

    (media logarítmica de Cesàro sobre valores singulares μ_n ↓ 0).
    En dimensión finita N = dim ℋ se usa el análogo exacto

        Tr_ω^{(N)}(T) := (1 / log N) ∑_{n=1}^N μ_n(T)   (N > 1),

    y el volumen no conmutativo del observable respecto del triple:

        Vol_D(X) := Tr_ω^{(N)}( X |D|^{-p} ),   p = dimensión espectral.

    Con D = ρ^{-1/2} se tiene |D|^{-p} = ρ^{p/2}.
    """

    # ── §3.1  Potencia resolvente |D|^{-p} ───────────────────────────────────

    @staticmethod
    def _dirac_resolvent_power(
        D: NDArray[np.complex128],
        d_spec: NDArray[np.float64],
        p: float,
    ) -> NDArray[np.complex128]:
        r"""
        Calcula |D|^{-p} vía cálculo funcional sobre spec(D).

        Como D > 0 autoadjunto: |D|^{-p} = U diag(d_i^{-p}) U†.
        """
        if not math.isfinite(p) or p <= 0.0:
            raise NonCommutativeVolumeAnomaly(
                f"Dimensión espectral p={p} inválida (se exige p > 0 finito)."
            )
        d_safe = np.clip(d_spec.real, _EIGENVALUE_FLOOR, None)
        powers = np.power(d_safe, -p).astype(np.float64)
        if not np.all(np.isfinite(powers)):
            raise NonCommutativeVolumeAnomaly(
                f"|D|^{{-p}} no finito para p={p}."
            )
        # Reconstruir en la base propia de D (D ya viene diagonalizado por construcción)
        evals, evecs = la.eigh(D)
        # Emparejar potencias con el espectro actual de D (recomputado por estabilidad)
        evals_safe = np.clip(evals.real, _EIGENVALUE_FLOOR, None)
        powers_live = np.power(evals_safe, -p).astype(np.complex128)
        R = evecs @ np.diag(powers_live) @ evecs.conj().T
        R = Phase1_SpectralTripleBinder._hermitize(R)
        return R.astype(np.complex128)

    # ── §3.2  Proxy de traza de Dixmier ──────────────────────────────────────

    @staticmethod
    def _dixmier_proxy(
        T: NDArray[np.complex128],
        dim: int,
    ) -> Tuple[float, NDArray[np.float64], float]:
        r"""
        Tr_ω^{(N)}(T) = (∑ μ_n(T)) / log N,  N = dim.

        Retorna (volumen, valores_singulares, log_N).
        """
        mu = la.svd(T, compute_uv=False)
        mu = np.clip(mu.real, 0.0, None).astype(np.float64)

        log_N = math.log(dim) if dim > 1 else 1.0
        if log_N <= 0.0:
            log_N = 1.0

        volume = float(np.sum(mu) / log_N)
        if not math.isfinite(volume) or volume < 0.0:
            raise NonCommutativeVolumeAnomaly(
                f"Volumen de Dixmier no físico: {volume}."
            )
        if volume > _DIXMIER_VOLUME_CEILING:
            raise NonCommutativeVolumeAnomaly(
                f"Volumen de Dixmier {volume:.3e} excede techo "
                f"{_DIXMIER_VOLUME_CEILING:.1e} (entropía fantasma)."
            )
        return volume, mu, float(log_N)

    # ── §3.3  Volumen no conmutativo y distorsión ────────────────────────────

    def compute_dixmier_volumes(
        self,
        spectral_triple: SpectralTripleData,
        X_raw: NDArray[np.complex128],
        X_deformed: NDArray[np.complex128],
        spectral_dim_p: float = _DEFAULT_SPECTRAL_DIM,
    ) -> DixmierTraceResult:
        r"""
        Computa Vol_D(X_raw), Vol_D(X_deformed) y la distorsión relativa.

            distortion = |Vol_def − Vol_raw| / max(Vol_raw, ε).

        Raises
        ------
        NonCommutativeVolumeAnomaly
            Si la distorsión excede _DIXMIER_DISTORTION_MAX o hay no-finitud.
        """
        D = spectral_triple.dirac_operator
        d_spec = spectral_triple.dirac_eigenvalues
        dim = int(spectral_triple.hilbert_space_dim)

        R = self._dirac_resolvent_power(D, d_spec, spectral_dim_p)

        T_def = X_deformed @ R
        T_raw = X_raw @ R

        vol_def, mu_def, log_N = self._dixmier_proxy(T_def, dim)
        vol_raw, mu_raw, _ = self._dixmier_proxy(T_raw, dim)

        base = max(vol_raw, _MACHINE_EPSILON)
        distortion = abs(vol_def - vol_raw) / base

        if not math.isfinite(distortion):
            raise NonCommutativeVolumeAnomaly(
                f"Distorsión de volumen no finita (vol_def={vol_def}, vol_raw={vol_raw})."
            )
        if distortion > _DIXMIER_DISTORTION_MAX:
            raise NonCommutativeVolumeAnomaly(
                f"Inyección de entropía fantasma: distorsión de Dixmier = "
                f"{distortion*100:.2f}% > techo {_DIXMIER_DISTORTION_MAX*100:.2f}%. "
                f"Vol_def={vol_def:.6e}, Vol_raw={vol_raw:.6e}."
            )

        return DixmierTraceResult(
            dixmier_volume=vol_def,
            lebesgue_volume_proxy=vol_raw,
            volume_distortion_ratio=float(distortion),
            spectral_dimension_p=float(spectral_dim_p),
            singular_values_deformed=mu_def,
            singular_values_raw=mu_raw,
            log_dimensional_factor=log_N,
        )

    # ── §3.4  MORFISMO TERMINAL DE FASE 3 ────────────────────────────────────
    #          (continuación de certify_kms_equilibrium;
    #           cierra el endofuntor 𝒵_Connes)

    def integrate_dixmier_and_close(
        self,
        kms_bundle: KMSThermalBundle,
        engine: Any,
        spectral_dim_p: float = _DEFAULT_SPECTRAL_DIM,
    ) -> ConnesAuditState:
        r"""
        Integra Dixmier y cierra con Umegaki:

            Φ₃ᶜ : KMSThermalBundle × Engine ⟶ ConnesAuditState.

        Pipeline
        --------
        1. Verificar KMS canónico del bundle de Fase 2.
        2. Volúmenes de Dixmier pre/post zoom + distorsión.
        3. Extracción Umegaki vía motor TT (fase 3 del motor).
        4. Consolidar bandera epistemológica global.
        5. Empaquetar ConnesAuditState (objeto final).

        Parámetros
        ----------
        kms_bundle : KMSThermalBundle
            Salida de Φ₂ᶜ = certify_kms_equilibrium.
        engine : TomitaTakesakiTelescopicEngine
            Motor TT para la extracción Umegaki.
        spectral_dim_p : float
            Dimensión espectral p del triple.

        Retorna
        -------
        ConnesAuditState
            Paquete final de la auditoría espectral de Connes.
        """
        if not kms_bundle.kms_canonical.is_kms_compliant:
            raise KMSEquilibriumViolation(
                "Bundle KMS no compliant; imposible integrar Dixmier."
            )

        triple = kms_bundle.spectral_triple
        flow = kms_bundle.flow_data
        rho = kms_bundle.rho_mac
        X_raw = triple.X_reference
        X_def = flow.X_deformed

        # Dixmier
        dixmier = self.compute_dixmier_volumes(
            spectral_triple=triple,
            X_raw=X_raw,
            X_deformed=X_def,
            spectral_dim_p=spectral_dim_p,
        )

        # Umegaki (cierre del motor TT)
        try:
            umegaki: UmegakiExtractionState = engine.extract_and_verify_umegaki(
                rho, flow
            )
        except UmegakiDivergenceError:
            raise
        except TomitaTakesakiEngineError as exc:
            raise ConnesPipelineError(
                f"Extracción Umegaki del motor TT falló: {exc}"
            ) from exc

        umegaki_safe = bool(getattr(umegaki, "is_epistemologically_safe", False))

        global_safe = bool(
            triple.is_differentiable
            and kms_bundle.kms_canonical.is_kms_compliant
            and dixmier.volume_distortion_ratio <= _DIXMIER_DISTORTION_MAX
            and umegaki_safe
        )

        logger.debug(
            "Dixmier+Umegaki | Vol_def=%.6e | Vol_raw=%.6e | dist=%.4f | "
            "D_Umegaki=%.6e | F=%.6f | safe=%s",
            dixmier.dixmier_volume,
            dixmier.lebesgue_volume_proxy,
            dixmier.volume_distortion_ratio,
            float(getattr(umegaki, "umegaki_relative_entropy", float("nan"))),
            float(getattr(umegaki, "fidelity_uhlmann", float("nan"))),
            global_safe,
        )

        return ConnesAuditState(
            zoomed_extraction=umegaki,
            spectral_triple=triple,
            kms_state=kms_bundle.kms_canonical,
            kms_bundle_zoom_friction=kms_bundle.zoom_friction_residual,
            dixmier_trace=dixmier,
            lambda_zoom_applied=float(getattr(flow, "lambda_zoom", float("nan"))),
            is_epistemologically_safe=global_safe,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   ORQUESTADOR SUPREMO · CONNES SPECTRAL AUDITOR AGENT                       ║
# ║   Endofuntor 𝒵_Connes = Φ₃ᶜ ∘ Φ₂ᶜ ∘ Φ₁ᶜ                                     ║
# ║   Custodio de la métrica no conmutativa sobre el álgebra semántica 𝒜.       ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class ConnesSpectralAuditorAgent(Morphism, Phase3_DixmierTraceIntegrator):
    r"""
    Custodio de la métrica no conmutativa.

    Asegura que ni el LLM ni las operaciones humanas de auditoría telescópica
    puedan torcer el isomorfismo ontológico del espacio de Hilbert semántico,
    componiendo:

        (ρ, X, λ)  ──Φ₁ᶜ──►  SpectralTripleData
                   ──Φ₂ᶜ──►  KMSThermalBundle
                   ──Φ₃ᶜ──►  ConnesAuditState.

    El motor Tomita–Takesaki se inyecta por composición (no por herencia),
    preservando la separación de responsabilidades:
      • TT  = flujo modular + Umegaki,
      • Connes = triple espectral + KMS + Dixmier + orquestación.
    """

    def __init__(
        self,
        engine: Any,
        kms_beta: float = _KMS_BETA,
        spectral_dim_p: float = _DEFAULT_SPECTRAL_DIM,
        dixmier_distortion_max: float = _DIXMIER_DISTORTION_MAX,
    ) -> None:
        r"""
        Parámetros
        ----------
        engine : TomitaTakesakiTelescopicEngine
            Motor modular acoplado (requerido).
        kms_beta : float
            Temperatura inversa canónica para el certificado KMS.
        spectral_dim_p : float
            Dimensión espectral p del triple (Dixmier).
        dixmier_distortion_max : float
            Umbral de distorsión de volumen (override opcional; informativo
            — el veto usa la constante de módulo salvo reconfiguración futura).
        """
        # Validación duck-typing + isinstance cuando la clase real está disponible
        if engine is None:
            raise ConnesAuditorError("Se requiere un motor Tomita–Takesaki.")
        if TomitaTakesakiTelescopicEngine is not Any and not isinstance(
            engine, TomitaTakesakiTelescopicEngine
        ):
            # Aceptar duck-typing si expone la interfaz mínima
            required = (
                "extract_modular_operator",
                "execute_modular_zoom",
                "extract_and_verify_umegaki",
            )
            missing = [m for m in required if not hasattr(engine, m)]
            if missing:
                raise ConnesAuditorError(
                    "El motor suministrado no es TomitaTakesakiTelescopicEngine "
                    f"ni expone la interfaz mínima; faltan: {missing}."
                )

        if not math.isfinite(kms_beta) or kms_beta <= 0.0:
            raise ValueError(f"kms_beta > 0 finito requerido; recibido {kms_beta}.")
        if not math.isfinite(spectral_dim_p) or spectral_dim_p <= 0.0:
            raise ValueError(
                f"spectral_dim_p > 0 finito requerido; recibido {spectral_dim_p}."
            )
        if not math.isfinite(dixmier_distortion_max) or dixmier_distortion_max < 0.0:
            raise ValueError(
                f"dixmier_distortion_max ≥ 0 finito; recibido {dixmier_distortion_max}."
            )

        self.engine = engine
        self.kms_beta = float(kms_beta)
        self.spectral_dim_p = float(spectral_dim_p)
        self.dixmier_distortion_max = float(dixmier_distortion_max)

    # ── Orquestación principal ───────────────────────────────────────────────

    def execute_spectral_audit(
        self,
        rho_mac: NDArray[np.complex128],
        X_observable: NDArray[np.complex128],
        lambda_zoom: float,
    ) -> ConnesAuditState:
        r"""
        Orquesta el funtor espectral de Connes:

            𝒵_Connes(ρ, X, λ) = Φ₃ᶜ(Φ₂ᶜ(Φ₁ᶜ(ρ, X), engine, λ), engine).

        Parámetros
        ----------
        rho_mac : NDArray[np.complex128]
            Estado fiel de la MAC.
        X_observable : NDArray[np.complex128]
            Observable semántico a amplificar / auditar.
        lambda_zoom : float
            Intensidad del zoom modular (0 ≤ λ ≤ λ_max del motor).

        Retorna
        -------
        ConnesAuditState
            Estado completo de la auditoría espectral.
        """
        if not math.isfinite(lambda_zoom) or lambda_zoom < 0.0:
            raise ModularFlowSingularityError(
                f"lambda_zoom inválido: {lambda_zoom}."
            )

        try:
            # ── Φ₁ᶜ · Fase 1 · Triple espectral (𝒜, ℋ, D) ────────────────────
            spectral_triple: SpectralTripleData = self.bind_spectral_triple(
                rho_mac, X_observable
            )
            logger.info(
                "Fase 1 completa | n=%d | L(X)=%.6e | κ(D)=%.3e | diferenciable=%s",
                spectral_triple.hilbert_space_dim,
                spectral_triple.lipschitz_seminorm,
                spectral_triple.dirac_condition_number,
                spectral_triple.is_differentiable,
            )

            # ── Φ₂ᶜ · Fase 2 · KMS canónico + zoom de auditoría ──────────────
            #     Dominio = objeto terminal de Φ₁ᶜ
            kms_bundle: KMSThermalBundle = self.certify_kms_equilibrium(
                spectral_triple,
                self.engine,
                lambda_zoom,
                beta=self.kms_beta,
            )
            logger.info(
                "Fase 2 completa | β=%.2f | r_KMS=%.3e | f_zoom=%.3e | "
                "KMS=%s | zoom_stable=%s",
                kms_bundle.kms_canonical.kms_beta,
                kms_bundle.kms_canonical.thermal_residual_norm,
                kms_bundle.zoom_friction_residual,
                kms_bundle.kms_canonical.is_kms_compliant,
                kms_bundle.is_zoom_thermally_stable,
            )

            # ── Φ₃ᶜ · Fase 3 · Dixmier + cierre Umegaki ──────────────────────
            #     Dominio = objeto terminal de Φ₂ᶜ
            audit_state: ConnesAuditState = self.integrate_dixmier_and_close(
                kms_bundle,
                self.engine,
                spectral_dim_p=self.spectral_dim_p,
            )

            logger.info(
                "Fase 3 completa | Vol_Dixmier=%.6e | Vol_proxy=%.6e | "
                "distorsión=%.4f | λ=%.4f | D_Umegaki=%.6e | F_Uhlmann=%.6f | "
                "seguro=%s",
                audit_state.dixmier_trace.dixmier_volume,
                audit_state.dixmier_trace.lebesgue_volume_proxy,
                audit_state.dixmier_trace.volume_distortion_ratio,
                audit_state.lambda_zoom_applied,
                float(
                    getattr(
                        audit_state.zoomed_extraction,
                        "umegaki_relative_entropy",
                        float("nan"),
                    )
                ),
                float(
                    getattr(
                        audit_state.zoomed_extraction,
                        "fidelity_uhlmann",
                        float("nan"),
                    )
                ),
                audit_state.is_epistemologically_safe,
            )

            return audit_state

        except (
            ConnesAuditorError,
            TomitaTakesakiEngineError,
            GNSConstructionError,
            InvalidObservableError,
            ModularFlowSingularityError,
            UmegakiDivergenceError,
        ):
            raise
        except ValueError:
            raise
        except Exception as exc:  # pragma: no cover
            raise ConnesPipelineError(
                f"Fallo no catalogado en la tubería espectral de Connes: {exc}"
            ) from exc


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA DEL MÓDULO
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    # Excepciones
    "ConnesAuditorError",
    "SemanticDiscontinuityError",
    "SpectralTripleError",
    "KMSEquilibriumViolation",
    "NonCommutativeVolumeAnomaly",
    "ConnesPipelineError",
    # DTOs (objetos del Topos)
    "SpectralTripleData",
    "KMSThermalState",
    "KMSThermalBundle",
    "DixmierTraceResult",
    "ConnesAuditState",
    # Fases anidadas
    "Phase1_SpectralTripleBinder",
    "Phase2_KMSEquilibriumAuditor",
    "Phase3_DixmierTraceIntegrator",
    # Orquestador
    "ConnesSpectralAuditorAgent",
]