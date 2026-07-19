# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Tomita-Takesaki Telescopic Engine (Motor Modular Telescópico)       ║
║ Ruta   : app/wisdom/tomita_takesaki_telescopic_engine.py                     ║
║ Versión: 2.0.0-GNS-Takesaki-Umegaki-Petz-Doctoral                            ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GEOMETRÍA NO CONMUTATIVA (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor materializa la observación no-destructiva (zoom telescópico)
del espacio semántico del LLM (MAC). Abandona la deformación espacial clásica
para operar directamente sobre el Grupo de Automorfismos Modulares de un
Álgebra de von Neumann Tipo I_n.

Garantiza axiomáticamente que la magnificación de autoestados latentes
(alucinaciones) no colapse la función de onda principal, empleando la
Expectación Condicional de Umegaki y la métrica de Fisher–Rao cuántica
(Petz) para asegurar que la Divergencia Entrópica se mantenga acotada.

Fundamentos formales:
  • GNS / estándar : ℋ = HS(ℂⁿ), |Ω_ρ⟩ ↔ ρ^{1/2}, S(x Ω)=x* Ω, Δ = S*S.
  • Takesaki       : σ_t(x) = Δ^{it} x Δ^{-it}; continuación analítica t ↦ −iλ.
  • Flujo real     : σ_λ(X) = ρ^{-λ} X ρ^λ  (en forma espacial / base propia).
  • Umegaki        : D(ρ‖σ) = Tr[ρ(log ρ − log σ)] ≥ 0  (Klein).
  • Uhlmann        : F(ρ,σ) = ‖√ρ √σ‖_1² ∈ [0,1].
  • Petz–Fisher    : g_ρ(A,A) = ∫_0^1 Tr(A ρ^t A ρ^{1−t}) dt  (diagnóstico).

ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Construcción GNS y Fibrado de Purificación:
         Extrae spec(ρ), U, Δ_{ij}=μ_i/μ_j, J (conjugación modular) y gap de pureza.
         Morfismo terminal: extract_modular_operator ↦ GNSFibrationData
         ≡ dominio inicial de Fase 2.

Fase 2 → Flujo Modular Analítico (Magnificación):
         Aplica σ_λ(X) = ρ^{-λ} X ρ^λ con control de κ(flujo) y cotas de espectro.
         Morfismo terminal: execute_modular_zoom ↦ ModularFlowData
         ≡ dominio inicial de Fase 3.

Fase 3 → Extracción vía Expectación de Umegaki / Petz:
         Construye el estado post-observación, calcula D(ρ‖σ), F_Uhlmann y
         la métrica de Fisher local; veta singularidades epistemológicas.
         Morfismo terminal: extract_and_verify_umegaki ↦ UmegakiExtractionState
         ≡ objeto final del endofuntor Z = Φ₃ ∘ Φ₂ ∘ Φ₁.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

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
        r"""Violación a un invariante topológico categórico en el Topos E_MIC."""
        pass

    class Morphism:
        r"""Clase base de Morfismos del Topos E_MIC."""
        pass

logger = logging.getLogger("MAC.Wisdom.TomitaTakesakiTelescopicEngine")


# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES MATEMÁTICAS, ESPECTRALES Y LÍMITES ENTRÓPICOS
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: float = float(np.finfo(np.float64).eps)
_POSITIVITY_TOLERANCE: float = 1e-12
_TRACE_TOLERANCE: float = 1e-12
_HERMITICITY_TOLERANCE: float = 1e-12
_FAITHFULNESS_FLOOR: float = 1e-12       # piso de fidelidad espectral (estado fiel)
_EIGENVALUE_LOG_FLOOR: float = 1e-15     # piso para logaritmos espectrales
_UMEGAKI_DIVERGENCE_MAX: float = 0.05    # D(ρ‖σ) máximo tolerable antes de veto
_MODULAR_FLOW_CONDITION_MAX: float = 1e10
_MAX_LAMBDA_ZOOM: float = 10.0           # cota de λ (evita overflow del flujo)
_MIN_LAMBDA_ZOOM: float = 0.0
_FIDELITY_FLOOR: float = 0.0
_FIDELITY_CEILING: float = 1.0
_FISHER_METRIC_CEILING: float = 1e8      # techo de sanidad Petz–Fisher
_OBSERVABLE_NORM_CEILING: float = 1e6    # techo de ‖X‖_2
_PURITY_GAP_FLOOR: float = -1e-14       # pureza ≤ 1 (+ ruido numérico)


# ══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES CUÁNTICAS (GEOMETRÍA DE INFORMACIÓN)
# ══════════════════════════════════════════════════════════════════════════════
class TomitaTakesakiEngineError(TopologicalInvariantError):
    r"""Excepción raíz del Motor Telescópico de Tomita–Takesaki."""
    pass


class GNSConstructionError(TomitaTakesakiEngineError):
    r"""
    Detonada si ρ_MAC no admite vector cíclico y separador
    (estado no fiel, espectro degenerado a cero, o fallo GNS).
    """
    pass


class InvalidObservableError(TomitaTakesakiEngineError):
    r"""El observable X no es hermítico, no es cuadrado, o es espectralmente patológico."""
    pass


class ModularFlowSingularityError(TomitaTakesakiEngineError):
    r"""Detonada si el espectro de Δ induce desbordamiento / mal condicionamiento en σ_λ."""
    pass


class UmegakiDivergenceError(TomitaTakesakiEngineError):
    r"""Detonada si D(ρ‖σ) supera el umbral o el estado post-zoom viola 𝔇(ℋ)."""
    pass


class PetzMetricSingularityError(TomitaTakesakiEngineError):
    r"""Detonada si la métrica de información de Petz–Fisher diverge o no es finita."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Topos Cuántico — objetos de las fibras)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class GNSFibrationData:
    r"""
    Artefacto terminal de Fase 1.

    Datos del fibrado de purificación / representación estándar:
      • ρ = U diag(μ) U†,  μ_i > 0  (estado fiel),
      • Δ_{ij} = μ_i / μ_j  (núcleo integral del operador modular),
      • J : conjugación modular en base propia (J(e_{ij}) = e_{ji} con fases),
      • purity_gap = 1 − Tr(ρ²) ∈ [0, 1 − 1/n].

    Invariantes:
      • ∑ μ_i = 1,  μ_i ≥ ε_faith
      • Δ_{ij} · Δ_{ji} = 1,  Δ_{ii} = 1
      • 0 ≤ purity_gap ≤ 1 − 1/n + ε
    """
    rho_eigenvalues: NDArray[np.float64]
    rho_eigenvectors: NDArray[np.complex128]
    modular_operator_delta: NDArray[np.float64]
    modular_conjugation_J_phases: NDArray[np.complex128]
    purity_gap: float
    faithful_spectral_floor: float
    hilbert_space_dim: int


@dataclass(frozen=True, slots=True)
class ModularFlowData:
    r"""
    Artefacto terminal de Fase 2 (y dominio inicial de Fase 3).

    Observable deformado por el flujo modular analítico:

        σ_λ(X) = ρ^{-λ} X ρ^λ ,

    junto con diagnósticos de condicionamiento y el GNS de origen
    (para que Fase 3 no re-diagonalice ρ).

    Invariantes:
      • X_deformed = X_deformed†  (el flujo preserva hermiticidad)
      • 0 ≤ λ ≤ λ_max
      • κ_flow < κ_max
    """
    X_deformed: NDArray[np.complex128]
    X_original: NDArray[np.complex128]
    lambda_zoom: float
    flow_condition_number: float
    flow_multiplier_spectrum: NDArray[np.float64]
    gns_data: GNSFibrationData


@dataclass(frozen=True, slots=True)
class UmegakiExtractionState:
    r"""
    Artefacto terminal de Fase 3 — objeto final del endofuntor Z.

    Estado auditado σ ∈ 𝔇(ℋ) con control epistemológico:
      • D(ρ‖σ) ≤ D_max          (Umegaki)
      • F(ρ,σ) ∈ [0,1]          (Uhlmann)
      • g_ρ(δ,δ) < ∞            (Petz–Fisher local)
      • is_epistemologically_safe ⇒ todo lo anterior.
    """
    rho_audit_extracted: NDArray[np.complex128]
    umegaki_relative_entropy: float
    fidelity_uhlmann: float
    petz_fisher_metric: float
    klein_inequality_residual: float
    is_epistemologically_safe: bool
    lambda_zoom_applied: float
    flow_condition_number: float


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 1 · CONSTRUCCIÓN GNS Y FIBRADO DE PURIFICACIÓN CUÁNTICA              ║
# ║   Extrae el Operador Modular de Takesaki Δ para el Álgebra de von Neumann.  ║
# ║                                                                             ║
# ║   Definición formal del objeto terminal de esta fase:                       ║
# ║       extract_modular_operator : 𝔇_{>0}(ℋ) → GNSFibrationData               ║
# ║   Este morfismo es el dominio de partida de Fase 2.                         ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_GNSConstruction:
    r"""
    Aplica la construcción de Gelfand–Naimark–Segal (GNS) / representación
    estándar para definir el operador modular Δ sobre el estado cíclico y
    separador de la MAC.

    Enunciado operativo
    -------------------
    Para el álgebra ℳ = M_n(ℂ) y un estado fiel ρ > 0 (μ_i ≥ ε_faith ∀i),

        ℋ_std ≅ HS(ℂⁿ),   |Ω_ρ⟩ ↔ ρ^{1/2},
        S(x Ω_ρ) = x* Ω_ρ,   Δ = S* S = ρ ⊗ (ρ^{-1})^⊤,

    y en la base propia de ρ:

        Δ|i⟩⟨j| = (μ_i / μ_j) |i⟩⟨j|,   Δ_{ij} = μ_i / μ_j.

    La conjugación modular J satisface J Δ^{1/2} = S y, en Tipo I_n con
    base real-positiva elegida, actúa como J(|i⟩⟨j|) = |j⟩⟨i| (fases = 1).
    """

    # ── §1.1  Utilidades espectrales ─────────────────────────────────────────

    @staticmethod
    def _hermitize(M: NDArray[np.complex128]) -> NDArray[np.complex128]:
        r"""Proyección de Frobenius sobre el subespacio hermítico: (M+M†)/2."""
        return 0.5 * (M + M.conj().T)

    @staticmethod
    def _validate_density_matrix(
        rho: NDArray[np.complex128],
        name: str = "rho",
    ) -> NDArray[np.complex128]:
        r"""
        Verifica y proyecta ρ sobre 𝔇(ℋ).

        Condiciones:
          (i)   cuadrada, n ≥ 1
          (ii)  ‖ρ − ρ†‖_F ≤ ε_herm
          (iii) spec(ρ) ⊂ [−ε_psd, +∞)
          (iv)  |Tr ρ − 1| ≤ ε_tr

        Retorna la versión hermitizada y renormalizada.
        """
        if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
            raise GNSConstructionError(
                f"{name}: se exige matriz cuadrada; recibido shape={rho.shape}."
            )
        if rho.shape[0] < 1:
            raise GNSConstructionError(f"{name}: dimensión nula.")

        anti_herm = float(la.norm(rho - rho.conj().T, ord="fro"))
        if anti_herm > _HERMITICITY_TOLERANCE:
            raise GNSConstructionError(
                f"{name}: no hermítica. ‖ρ−ρ†‖_F = {anti_herm:.3e} "
                f"> {_HERMITICITY_TOLERANCE:.1e}."
            )
        rho_h = Phase1_GNSConstruction._hermitize(rho)

        eigvals = la.eigvalsh(rho_h)
        if np.any(eigvals < -_POSITIVITY_TOLERANCE):
            raise GNSConstructionError(
                f"{name}: autovalor negativo {eigvals.min():.3e}; no es SPD."
            )

        tr = complex(np.trace(rho_h))
        if abs(tr - 1.0) > _TRACE_TOLERANCE:
            raise GNSConstructionError(
                f"{name}: |Tr ρ − 1| = {abs(tr - 1.0):.3e} > {_TRACE_TOLERANCE:.1e}."
            )
        if abs(tr.real - 1.0) > _MACHINE_EPSILON:
            rho_h = rho_h / tr.real

        return rho_h.astype(np.complex128)

    # ── §1.2  Fidelidad del estado (vector separador) ────────────────────────

    @staticmethod
    def _verify_faithful_state(
        eigenvalues: NDArray[np.float64],
        floor: float = _FAITHFULNESS_FLOOR,
    ) -> float:
        r"""
        Un estado es fiel (y |Ω_ρ⟩ es separador) sii μ_i ≥ floor > 0 ∀i.

        Retorna el mínimo espectral μ_min; lanza GNSConstructionError si falla.
        """
        mu_min = float(np.min(eigenvalues))
        if mu_min < floor:
            raise GNSConstructionError(
                f"Estado no fiel: μ_min = {mu_min:.3e} < floor = {floor:.1e}. "
                f"El espacio nulo es no trivial; |Ω_ρ⟩ no es separador. "
                f"La Matriz Atómica de Conocimiento requiere soporte pleno."
            )
        return mu_min

    # ── §1.3  Operador modular Δ y conjugación J ─────────────────────────────

    @staticmethod
    def _build_modular_operator(
        eigenvalues: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        Núcleo del operador modular en base propia:

            Δ_{ij} = μ_i / μ_j .

        Identidades estructurales:
            Δ_{ii} = 1,   Δ_{ij} · Δ_{ji} = 1,   Δ > 0 elemento a elemento.
        """
        # outer(μ, 1/μ)_{ij} = μ_i / μ_j
        inv = 1.0 / eigenvalues
        delta = np.outer(eigenvalues, inv).astype(np.float64)
        return delta

    @staticmethod
    def _build_modular_conjugation_phases(
        dim: int,
    ) -> NDArray[np.complex128]:
        r"""
        Fases de la conjugación modular J en la base propia real-positiva.

        En Tipo I_n con gauge U elegido de modo que las fases de |Ω_ρ⟩ sean
        triviales, J(|i⟩⟨j|) = |j⟩⟨i| y el kernel de fases es idénticamente 1.
        Se expone como matriz de unos (complejos) para extensibilidad a gauges
        no triviales sin romper el contrato del DTO.
        """
        return np.ones((dim, dim), dtype=np.complex128)

    @staticmethod
    def _purity_gap(eigenvalues: NDArray[np.float64]) -> float:
        r"""
        Déficit de pureza:

            γ = 1 − Tr(ρ²) = 1 − ∑ μ_i² ∈ [0, 1 − 1/n].
        """
        purity = float(np.sum(eigenvalues ** 2))
        gap = 1.0 - purity
        if gap < _PURITY_GAP_FLOOR:
            # Solo ruido numérico: clip a 0
            gap = 0.0
        return float(gap)

    # ── §1.4  MORFISMO TERMINAL DE FASE 1 ────────────────────────────────────
    #          (dominio inicial de Fase 2)

    def extract_modular_operator(
        self,
        rho_mac: NDArray[np.complex128],
    ) -> GNSFibrationData:
        r"""
        Diagonaliza ρ_MAC y construye el fibrado modular de Takesaki:

            Φ₁ : 𝔇_{>0}(ℋ) ⟶ GNSFibrationData.

        Pipeline
        --------
        1. Validación espectral de ρ ∈ 𝔇(ℋ).
        2. Diagonalización hermitiana ρ = U diag(μ) U†.
        3. Verificación de fidelidad μ_i ≥ ε_faith.
        4. Construcción de Δ_{ij} = μ_i/μ_j y fases de J.
        5. Cálculo del déficit de pureza γ = 1 − Tr(ρ²).

        Parámetros
        ----------
        rho_mac : NDArray[np.complex128]
            Matriz de densidad fiel del sistema MAC (n × n).

        Retorna
        -------
        GNSFibrationData
            Objeto terminal de Fase 1 / dominio de Fase 2.
        """
        rho_mac = self._validate_density_matrix(rho_mac, name="rho_mac")
        dim = int(rho_mac.shape[0])

        eigenvalues, eigenvectors = la.eigh(rho_mac)
        eigenvalues = np.clip(eigenvalues.real, 0.0, None).astype(np.float64)

        # Renormalización defensiva del espectro
        s = float(eigenvalues.sum())
        if s < _MACHINE_EPSILON:
            raise GNSConstructionError("Espectro nulo tras diagonalización (Tr ρ ≈ 0).")
        eigenvalues /= s

        mu_min = self._verify_faithful_state(eigenvalues)
        delta = self._build_modular_operator(eigenvalues)
        J_phases = self._build_modular_conjugation_phases(dim)
        gap = self._purity_gap(eigenvalues)

        # Sanity estructural de Δ
        if not np.allclose(np.diag(delta), 1.0, atol=1e-10):
            raise GNSConstructionError("Δ_{ii} ≠ 1: fallo estructural del operador modular.")
        if np.any(delta <= 0.0) or not np.all(np.isfinite(delta)):
            raise GNSConstructionError("Δ no es estrictamente positiva / finita.")

        logger.debug(
            "GNS | n=%d | μ_min=%.3e | purity_gap=%.6f | κ(Δ)=%.3e",
            dim, mu_min, gap, float(delta.max() / max(delta.min(), _MACHINE_EPSILON)),
        )

        return GNSFibrationData(
            rho_eigenvalues=eigenvalues,
            rho_eigenvectors=eigenvectors.astype(np.complex128),
            modular_operator_delta=delta,
            modular_conjugation_J_phases=J_phases,
            purity_gap=gap,
            faithful_spectral_floor=mu_min,
            hilbert_space_dim=dim,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 2 · FLUJO MODULAR ANALÍTICO (PARAMETRIZACIÓN DE RESOLUCIÓN)          ║
# ║   Deforma el observable X mediante σ_λ(X) = ρ^{-λ} X ρ^λ                    ║
# ║                                                                             ║
# ║   Continuación funtorial: el morfismo terminal de Fase 1                    ║
# ║       extract_modular_operator ↦ GNSFibrationData                           ║
# ║   es el dominio de                                                         ║
# ║       execute_modular_zoom : GNSFibrationData × 𝒪(ℋ) × [0,λ_max]            ║
# ║           → ModularFlowData                                                 ║
# ║   Este morfismo terminal de Fase 2 es el dominio de partida de Fase 3.      ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_AnalyticModularFlow(Phase1_GNSConstruction):
    r"""
    Aplica la continuación analítica del grupo de automorfismos modulares
    de Tomita–Takesaki para amplificar autoestados subdominantes sin demoler
    la traza de la MAC.

    Geometría del flujo
    -------------------
    El grupo modular σ_t(x) = Δ^{it} x Δ^{-it} admite continuación analítica
    al strip de KMS. Sobre el eje imaginario t = −iλ (λ ∈ ℝ_{≥0}):

        σ_λ(X) = ρ^{-λ} X ρ^λ.

    En la base propia de ρ, si X ↦ X̃ = U† X U,

        (σ_λ(X))̃_{ij} = (μ_j / μ_i)^λ  X̃_{ij} = Δ_{ij}^{-λ} X̃_{ij}.

    El flujo:
      • preserva hermiticidad (X=X† ⇒ σ_λ(X)=σ_λ(X)†),
      • es un automorfismo de ℳ (invertible, σ_0 = id, σ_a ∘ σ_b = σ_{a+b}),
      • expande exponencialmente las coherencias entre niveles de μ distintos,
        revelando el «razonamiento oculto» de la MAC.
    """

    # ── §2.1  Validación del observable ──────────────────────────────────────

    @staticmethod
    def _validate_observable(
        X: NDArray[np.complex128],
        expected_dim: Optional[int] = None,
    ) -> NDArray[np.complex128]:
        r"""
        Verifica que X ∈ 𝒪(ℋ) sea un observable legítimo:
          • cuadrado,
          • hermítico,
          • dimensión compatible con ρ (si se provee),
          • norma de Frobenius bajo techo de sanidad.
        """
        if X.ndim != 2 or X.shape[0] != X.shape[1]:
            raise InvalidObservableError(
                f"X debe ser matriz cuadrada; recibido shape={X.shape}."
            )
        if expected_dim is not None and X.shape[0] != expected_dim:
            raise InvalidObservableError(
                f"Dimensión de X ({X.shape[0]}) ≠ dim ℋ ({expected_dim})."
            )
        anti = float(la.norm(X - X.conj().T, ord="fro"))
        if anti > _HERMITICITY_TOLERANCE:
            raise InvalidObservableError(
                f"X no es hermítico: ‖X−X†‖_F = {anti:.3e}."
            )
        X_h = Phase1_GNSConstruction._hermitize(X)
        fro = float(la.norm(X_h, ord="fro"))
        if not math.isfinite(fro) or fro > _OBSERVABLE_NORM_CEILING:
            raise InvalidObservableError(
                f"‖X‖_F = {fro:.3e} no finita o > techo {_OBSERVABLE_NORM_CEILING:.1e}."
            )
        return X_h.astype(np.complex128)

    # ── §2.2  Multiplicadores del flujo y condicionamiento ───────────────────

    @staticmethod
    def _flow_multipliers(
        delta: NDArray[np.float64],
        lambda_zoom: float,
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        Calcula el kernel de multiplicadores del flujo

            M_{ij}(λ) = Δ_{ij}^{-λ} = (μ_j / μ_i)^λ

        y su número de condición

            κ(M) = max M / max(min M, ε).

        Raises
        ------
        ModularFlowSingularityError
            Si M no es finita o κ(M) excede el techo.
        """
        # Potenciación segura: M = exp(−λ · log Δ)
        log_delta = np.log(delta)  # Δ > 0 garantizado por Fase 1
        with np.errstate(over="raise", under="ignore", invalid="raise"):
            try:
                log_M = -lambda_zoom * log_delta
                # Clip de exponentes para evitar overflow duro antes del check
                log_M_clipped = np.clip(log_M, -700.0, 700.0)
                M = np.exp(log_M_clipped).astype(np.float64)
            except FloatingPointError as exc:
                raise ModularFlowSingularityError(
                    f"Desbordamiento al computar M=Δ^{{-λ}} con λ={lambda_zoom}: {exc}"
                ) from exc

        if not np.all(np.isfinite(M)):
            raise ModularFlowSingularityError(
                f"Multiplicadores del flujo no finitos para λ={lambda_zoom}."
            )

        m_max = float(np.max(M))
        m_min = float(np.min(M))
        cond = m_max / max(m_min, _MACHINE_EPSILON)

        if cond > _MODULAR_FLOW_CONDITION_MAX or not math.isfinite(cond):
            raise ModularFlowSingularityError(
                f"Resolución telescópica λ={lambda_zoom} produce κ_flow={cond:.3e} "
                f"> techo {_MODULAR_FLOW_CONDITION_MAX:.1e}."
            )
        return M, float(cond)

    # ── §2.3  Aplicación del automorfismo modular ────────────────────────────

    @staticmethod
    def _apply_modular_automorphism(
        X: NDArray[np.complex128],
        evecs: NDArray[np.complex128],
        multipliers: NDArray[np.float64],
    ) -> NDArray[np.complex128]:
        r"""
        σ_λ(X) en tres pasos:
          1. X̃ = U† X U
          2. X̃' = M ⊙ X̃   (producto de Hadamard con multiplicadores)
          3. X' = U X̃' U†
        y hermitización defensiva final.
        """
        X_eigen = evecs.conj().T @ X @ evecs
        X_def_eigen = X_eigen * multipliers  # Hadamard
        X_def = evecs @ X_def_eigen @ evecs.conj().T
        X_def = Phase1_GNSConstruction._hermitize(X_def)
        return X_def.astype(np.complex128)

    # ── §2.4  MORFISMO TERMINAL DE FASE 2 ────────────────────────────────────
    #          (continuación de extract_modular_operator;
    #           dominio inicial de Fase 3)

    def execute_modular_zoom(
        self,
        gns_data: GNSFibrationData,
        X_observable: NDArray[np.complex128],
        lambda_zoom: float,
    ) -> ModularFlowData:
        r"""
        Calcula σ_λ(X) = ρ^{-λ} X ρ^λ en la base propia de ρ:

            Φ₂ : GNSFibrationData × 𝒪(ℋ) × [0, λ_max] ⟶ ModularFlowData.

        Pipeline
        --------
        1. Validar λ ∈ [0, λ_max] y X ∈ 𝒪(ℋ) con dim compatible.
        2. Construir multiplicadores M = Δ^{-λ} y κ(M).
        3. Aplicar automorfismo modular (Hadamard en base propia).
        4. Verificar hermiticidad residual de X'.
        5. Empaquetar ModularFlowData (incluye gns_data para Fase 3).

        Parámetros
        ----------
        gns_data : GNSFibrationData
            Salida de Φ₁ = extract_modular_operator.
        X_observable : NDArray[np.complex128]
            Observable hermítico (dirección del zoom).
        lambda_zoom : float
            Parámetro de magnificación λ ∈ [0, λ_max].

        Retorna
        -------
        ModularFlowData
            Objeto terminal de Fase 2 / dominio de Fase 3.
        """
        if not math.isfinite(lambda_zoom):
            raise ModularFlowSingularityError(
                f"λ_zoom no finito: {lambda_zoom}."
            )
        if lambda_zoom < _MIN_LAMBDA_ZOOM or lambda_zoom > _MAX_LAMBDA_ZOOM:
            raise ModularFlowSingularityError(
                f"λ_zoom = {lambda_zoom} fuera del intervalo seguro "
                f"[{_MIN_LAMBDA_ZOOM}, {_MAX_LAMBDA_ZOOM}]."
            )

        dim = int(gns_data.hilbert_space_dim)
        X = self._validate_observable(X_observable, expected_dim=dim)

        # Multiplicadores y condicionamiento
        M, cond = self._flow_multipliers(
            gns_data.modular_operator_delta, lambda_zoom
        )

        # Automorfismo modular
        X_deformed = self._apply_modular_automorphism(
            X, gns_data.rho_eigenvectors, M
        )

        # Diagnóstico de hermiticidad residual
        anti = float(la.norm(X_deformed - X_deformed.conj().T, ord="fro"))
        if anti > _HERMITICITY_TOLERANCE:
            raise ModularFlowSingularityError(
                f"σ_λ(X) perdió hermiticidad: ‖·‖_F = {anti:.3e}."
            )

        logger.debug(
            "Flujo modular | λ=%.4f | κ_flow=%.3e | ‖X'‖_F=%.6e",
            lambda_zoom, cond, float(la.norm(X_deformed, ord="fro")),
        )

        return ModularFlowData(
            X_deformed=X_deformed,
            X_original=X,
            lambda_zoom=float(lambda_zoom),
            flow_condition_number=cond,
            flow_multiplier_spectrum=M,
            gns_data=gns_data,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 3 · EXTRACCIÓN VÍA EXPECTACIÓN CONDICIONAL DE UMEGAKI / PETZ         ║
# ║   Audita la divergencia de información mediante D(ρ‖σ) y g_ρ                ║
# ║                                                                             ║
# ║   Continuación funtorial: el morfismo terminal de Fase 2                    ║
# ║       execute_modular_zoom ↦ ModularFlowData                                ║
# ║   es el dominio de                                                         ║
# ║       extract_and_verify_umegaki : 𝔇(ℋ) × ModularFlowData                   ║
# ║           → UmegakiExtractionState                                          ║
# ║   Este morfismo cierra el endofuntor Z = Φ₃ ∘ Φ₂ ∘ Φ₁.                      ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_UmegakiExtraction(Phase2_AnalyticModularFlow):
    r"""
    Construye el estado post-observación mediante una actualización simétrica
    (medida débil / canal de Lüders suavizado), y certifica los invariantes
    de geometría de la información:

      • Divergencia de Umegaki D(ρ‖σ) = Tr[ρ(log ρ − log σ)] ≥ 0,
      • Fidelidad de Uhlmann F(ρ,σ) = ‖√ρ √σ‖_1² ∈ [0,1],
      • Métrica de Petz–Fisher local g_ρ(A,A) a lo largo de A ∝ σ − ρ.

    Si D(ρ‖σ) > D_max se declara desgarro epistemológico
    (UmegakiDivergenceError).
    """

    # ── §3.1  Divergencia de Umegaki (entropía relativa cuántica) ────────────

    @staticmethod
    def _compute_umegaki_divergence(
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
    ) -> float:
        r"""
        D(ρ‖σ) = Tr[ρ log ρ] − Tr[ρ log σ], calculada espectralmente.

        • Tr[ρ log ρ] = ∑_i μ_i log μ_i  (base propia de ρ).
        • Tr[ρ log σ] = ∑_k (V† ρ V)_{kk} log ν_k  (base propia de σ = V diag(ν) V†).

        Se aplica piso espectral _EIGENVALUE_LOG_FLOOR para evitar −∞.
        Por la desigualdad de Klein, D ≥ 0; se reporta max(D, 0).
        """
        # Tr(ρ log ρ)
        mu = la.eigvalsh(rho).real
        mu = np.clip(mu, _EIGENVALUE_LOG_FLOOR, None)
        mu = mu / float(mu.sum())  # renormalización defensiva
        tr_rho_log_rho = float(np.sum(mu * np.log(mu)))

        # Tr(ρ log σ) en la base de σ
        nu, V = la.eigh(sigma)
        nu = np.clip(nu.real, _EIGENVALUE_LOG_FLOOR, None)
        # No renormalizamos ν aquí más allá del clip: log σ usa el espectro de σ
        # ya normalizado como densidad; re-escalamos para consistencia.
        nu = nu / float(nu.sum())
        rho_in_sigma = V.conj().T @ rho @ V
        diag_rho = np.real(np.diag(rho_in_sigma))
        # Clip numérico de la diagonal proyectada
        diag_rho = np.clip(diag_rho, 0.0, None)
        tr_rho_log_sigma = float(np.sum(diag_rho * np.log(nu)))

        divergence = tr_rho_log_rho - tr_rho_log_sigma
        if not math.isfinite(divergence):
            raise UmegakiDivergenceError(
                f"D(ρ‖σ) no finita: {divergence}."
            )
        # Klein: D ≥ 0; ruido numérico → clip
        return float(max(divergence, 0.0))

    # ── §3.2  Fidelidad de Uhlmann ───────────────────────────────────────────

    @staticmethod
    def _compute_uhlmann_fidelity(
        rho: NDArray[np.complex128],
        sigma: NDArray[np.complex128],
    ) -> float:
        r"""
        Fidelidad de Uhlmann (definición de squareroot fidelity):

            F(ρ,σ) = (Tr √(√ρ σ √ρ))² ∈ [0,1].

        Implementación vía sqrtm de SciPy + proyección al intervalo [0,1].
        """
        try:
            sqrt_rho = la.sqrtm(rho)
            # sqrtm puede devolver compleja por ruido; hermitizar
            sqrt_rho = 0.5 * (sqrt_rho + sqrt_rho.conj().T)
            middle = sqrt_rho @ sigma @ sqrt_rho
            middle = 0.5 * (middle + middle.conj().T)
            sqrt_middle = la.sqrtm(middle)
            sqrt_middle = 0.5 * (sqrt_middle + sqrt_middle.conj().T)
            fid_sqrt = float(np.real(np.trace(sqrt_middle)))
            fidelity = fid_sqrt ** 2
        except la.LinAlgError as exc:
            raise UmegakiDivergenceError(
                f"Fallo al computar fidelidad de Uhlmann: {exc}"
            ) from exc

        if not math.isfinite(fidelity):
            raise UmegakiDivergenceError(f"F(ρ,σ) no finita: {fidelity}.")
        return float(np.clip(fidelity, _FIDELITY_FLOOR, _FIDELITY_CEILING))

    # ── §3.3  Métrica de Petz–Fisher (diagnóstico local) ─────────────────────

    @staticmethod
    def _petz_fisher_metric(
        rho: NDArray[np.complex128],
        direction: NDArray[np.complex128],
        n_quad: int = 16,
    ) -> float:
        r"""
        Métrica monótona de Petz a lo largo de la dirección A = A†:

            g_ρ(A,A) = ∫_0^1 Tr(A ρ^t A ρ^{1−t}) dt

        (familia SLD / Bures–Fisher cuando el operador de peso es el
        simétrico logarítmico; aquí se usa la integral de Kubo–Mori).

        Cuadratura de Gauss–Legendre truncada en n_quad nodos.
        Se trabaja en la base propia de ρ para evaluar ρ^t = U diag(μ^t) U†.
        """
        A = 0.5 * (direction + direction.conj().T)
        mu, U = la.eigh(rho)
        mu = np.clip(mu.real, _EIGENVALUE_LOG_FLOOR, None)
        mu = mu / float(mu.sum())

        A_e = U.conj().T @ A @ U  # A en base propia de ρ

        # Nodos y pesos de Gauss–Legendre en [0,1]
        # (traslación afín de nodos estándar en [−1,1])
        nodes_std, weights_std = np.polynomial.legendre.leggauss(n_quad)
        nodes = 0.5 * (nodes_std + 1.0)       # t ∈ [0,1]
        weights = 0.5 * weights_std

        acc = 0.0
        for t, w in zip(nodes, weights):
            # Tr(A ρ^t A ρ^{1−t}) = ∑_{ij} |A_e_{ij}|² μ_i^t μ_j^{1−t}
            mu_t = np.power(mu, t)
            mu_1t = np.power(mu, 1.0 - t)
            # kernel_{ij} = μ_i^t · μ_j^{1−t}
            kernel = np.outer(mu_t, mu_1t)
            integrand = float(np.sum(np.abs(A_e) ** 2 * kernel))
            acc += float(w) * integrand

        if not math.isfinite(acc) or acc < 0.0:
            raise PetzMetricSingularityError(
                f"Métrica de Petz–Fisher no física: g={acc}."
            )
        if acc > _FISHER_METRIC_CEILING:
            raise PetzMetricSingularityError(
                f"g_ρ(A,A) = {acc:.3e} excede techo {_FISHER_METRIC_CEILING:.1e} "
                f"(singularidad de información de Fisher)."
            )
        return float(acc)

    # ── §3.4  Construcción del estado post-observación ───────────────────────

    @staticmethod
    def _symmetric_update(
        rho: NDArray[np.complex128],
        X_def: NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        r"""
        Actualización simétrica (Jordan product / medida débil):

            σ̃ = (ρ X' + X' ρ) / 2 ,
            σ  = σ̃ / Tr(σ̃).

        Preserva hermiticidad por construcción. La positividad se verifica
        a posteriori (no está garantizada para X' arbitrario no positivo).
        """
        raw = 0.5 * (rho @ X_def + X_def @ rho)
        raw = Phase1_GNSConstruction._hermitize(raw)
        tr = float(np.real(np.trace(raw)))
        if tr < _MACHINE_EPSILON:
            raise UmegakiDivergenceError(
                "El flujo modular aniquiló la densidad de probabilidad "
                f"(Tr σ̃ = {tr:.3e})."
            )
        sigma = raw / tr
        sigma = Phase1_GNSConstruction._hermitize(sigma)
        return sigma.astype(np.complex128)

    @staticmethod
    def _ensure_density(
        sigma: NDArray[np.complex128],
        name: str = "sigma",
    ) -> NDArray[np.complex128]:
        r"""
        Proyecta σ al simplejo 𝔇(ℋ): hermitiza, clip PSD, renormaliza.
        Lanza UmegakiDivergenceError si el espectro es irrecuperable.
        """
        sigma = Phase1_GNSConstruction._hermitize(sigma)
        ev, U = la.eigh(sigma)
        if np.any(ev < -_POSITIVITY_TOLERANCE):
            raise UmegakiDivergenceError(
                f"{name}: magnificación generó autovalores negativos "
                f"(min={ev.min():.3e}), violando la física de la MAC."
            )
        ev = np.clip(ev.real, 0.0, None)
        s = float(ev.sum())
        if s < _MACHINE_EPSILON:
            raise UmegakiDivergenceError(
                f"{name}: espectro enteramente nulo tras proyección PSD."
            )
        ev /= s
        sigma_psd = (U * ev) @ U.conj().T
        return Phase1_GNSConstruction._hermitize(sigma_psd).astype(np.complex128)

    # ── §3.5  MORFISMO TERMINAL DE FASE 3 ────────────────────────────────────
    #          (continuación de execute_modular_zoom;
    #           cierra el endofuntor Z = Φ₃ ∘ Φ₂ ∘ Φ₁)

    def extract_and_verify_umegaki(
        self,
        rho_mac: NDArray[np.complex128],
        flow_data: ModularFlowData,
    ) -> UmegakiExtractionState:
        r"""
        Construye el estado post-observación y certifica invariantes
        de información:

            Φ₃ : 𝔇(ℋ) × ModularFlowData ⟶ UmegakiExtractionState.

        Pipeline
        --------
        1. Validar / hermitizar ρ_MAC.
        2. Actualización simétrica σ̃ = (ρ X' + X' ρ)/2 → σ ∈ 𝔇(ℋ).
        3. D(ρ‖σ) de Umegaki + veto si D > D_max.
        4. Fidelidad de Uhlmann F(ρ,σ).
        5. Métrica de Petz–Fisher a lo largo de A = σ − ρ.
        6. Residuo de Klein max(−D, 0) (debe ser ~0).
        7. Empaquetar UmegakiExtractionState (objeto final de Z).

        Parámetros
        ----------
        rho_mac : NDArray[np.complex128]
            Estado original de la MAC.
        flow_data : ModularFlowData
            Salida de Φ₂ = execute_modular_zoom.

        Retorna
        -------
        UmegakiExtractionState
            Estado auditado y métricas de control epistemológico.
        """
        rho = self._validate_density_matrix(rho_mac, name="rho_mac")
        X_def = flow_data.X_deformed

        if X_def.shape != rho.shape:
            raise InvalidObservableError(
                f"X_deformed shape {X_def.shape} ≠ ρ shape {rho.shape}."
            )

        # Estado post-zoom
        sigma_raw = self._symmetric_update(rho, X_def)
        sigma = self._ensure_density(sigma_raw, name="rho_audit")

        # Divergencia de Umegaki
        divergence = self._compute_umegaki_divergence(rho, sigma)
        klein_residual = float(max(-divergence, 0.0))  # idealmente 0

        if divergence > _UMEGAKI_DIVERGENCE_MAX:
            raise UmegakiDivergenceError(
                f"El telescopio rasgó la semántica del LLM: D(ρ‖σ) = "
                f"{divergence:.6f} > D_max = {_UMEGAKI_DIVERGENCE_MAX}."
            )

        # Fidelidad de Uhlmann
        fidelity = self._compute_uhlmann_fidelity(rho, sigma)

        # Petz–Fisher a lo largo de la cuerda σ − ρ
        direction = self._hermitize(sigma - rho)
        try:
            fisher = self._petz_fisher_metric(rho, direction)
        except PetzMetricSingularityError:
            raise
        except Exception as exc:  # pragma: no cover
            raise PetzMetricSingularityError(
                f"Fallo inesperado en métrica de Petz: {exc}"
            ) from exc

        safe = bool(
            divergence <= _UMEGAKI_DIVERGENCE_MAX
            and _FIDELITY_FLOOR <= fidelity <= _FIDELITY_CEILING
            and math.isfinite(fisher)
            and fisher <= _FISHER_METRIC_CEILING
        )

        logger.debug(
            "Umegaki | D=%.6e | F=%.6f | g_Fisher=%.6e | Klein_res=%.3e | safe=%s",
            divergence, fidelity, fisher, klein_residual, safe,
        )

        return UmegakiExtractionState(
            rho_audit_extracted=sigma,
            umegaki_relative_entropy=divergence,
            fidelity_uhlmann=fidelity,
            petz_fisher_metric=fisher,
            klein_inequality_residual=klein_residual,
            is_epistemologically_safe=safe,
            lambda_zoom_applied=flow_data.lambda_zoom,
            flow_condition_number=flow_data.flow_condition_number,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   ORQUESTADOR SUPREMO · TOMITA–TAKESAKI TELESCOPIC ENGINE                   ║
# ║   Endofuntor Z = Φ₃ ∘ Φ₂ ∘ Φ₁ : 𝔇_{>0}(ℋ) × 𝒪(ℋ) × [0,λ_max]               ║
# ║                                    → UmegakiExtractionState                 ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class TomitaTakesakiTelescopicEngine(Morphism, Phase3_UmegakiExtraction):
    r"""
    Motor maestro de magnificación modular.

    Permite un «zoom no destructivo» sobre los autoestados subdominantes
    de la Matriz Atómica de Conocimiento (MAC), revelando el razonamiento
    oculto de la IA sin colapsar su función de onda, y vetando desgarros
    epistemológicos vía la divergencia de Umegaki y la métrica de Petz.

    Composición funtorial estricta
    ------------------------------
    ``execute_modular_audit`` encadena:

        ρ_MAC  ──Φ₁──►  GNSFibrationData
               ──Φ₂──►  ModularFlowData
               ──Φ₃──►  UmegakiExtractionState.

    Cada Φᵢ es el morfismo terminal de la fase i y el dominio de la fase i+1.
    """

    def __init__(
        self,
        umegaki_divergence_max: float = _UMEGAKI_DIVERGENCE_MAX,
        max_lambda_zoom: float = _MAX_LAMBDA_ZOOM,
    ) -> None:
        r"""
        Parámetros
        ----------
        umegaki_divergence_max : float
            Umbral D_max de veto epistemológico (por defecto el del módulo).
        max_lambda_zoom : float
            Cota superior operativa de λ (≤ cota global del módulo).
        """
        if not math.isfinite(umegaki_divergence_max) or umegaki_divergence_max < 0.0:
            raise ValueError(
                f"umegaki_divergence_max ≥ 0 y finito; recibido {umegaki_divergence_max}."
            )
        if not math.isfinite(max_lambda_zoom) or max_lambda_zoom <= 0.0:
            raise ValueError(
                f"max_lambda_zoom > 0 y finito; recibido {max_lambda_zoom}."
            )
        if max_lambda_zoom > _MAX_LAMBDA_ZOOM:
            raise ValueError(
                f"max_lambda_zoom={max_lambda_zoom} excede cota global {_MAX_LAMBDA_ZOOM}."
            )
        self.umegaki_divergence_max = float(umegaki_divergence_max)
        self.max_lambda_zoom = float(max_lambda_zoom)

    def execute_modular_audit(
        self,
        rho_mac: NDArray[np.complex128],
        X_observable: NDArray[np.complex128],
        lambda_magnification: float,
    ) -> UmegakiExtractionState:
        r"""
        Composición funtorial estricta Z = Φ₃ ∘ Φ₂ ∘ Φ₁.

        Parámetros
        ----------
        rho_mac : NDArray[np.complex128]
            Matriz de densidad del conocimiento (estado fiel).
        X_observable : NDArray[np.complex128]
            Observable hermítico que define la dirección del zoom.
        lambda_magnification : float
            Intensidad de la magnificación (0 ≤ λ ≤ λ_max del motor).

        Retorna
        -------
        UmegakiExtractionState
            Estado tras el zoom y métricas de control epistemológico.
        """
        if lambda_magnification > self.max_lambda_zoom:
            raise ModularFlowSingularityError(
                f"λ={lambda_magnification} excede max_lambda_zoom={self.max_lambda_zoom} "
                f"configurado en el motor."
            )

        # ── Φ₁ · Fase 1 · GNS + operador modular Δ ───────────────────────────
        gns_data: GNSFibrationData = self.extract_modular_operator(rho_mac)

        # ── Φ₂ · Fase 2 · Flujo modular analítico σ_λ ────────────────────────
        #     Dominio = objeto terminal de Φ₁
        flow_data: ModularFlowData = self.execute_modular_zoom(
            gns_data, X_observable, lambda_magnification
        )

        # ── Φ₃ · Fase 3 · Extracción Umegaki / Petz ──────────────────────────
        #     Dominio = objeto terminal de Φ₂
        audit_state: UmegakiExtractionState = self.extract_and_verify_umegaki(
            rho_mac, flow_data
        )

        # Veto adicional con umbral configurable del motor
        if audit_state.umegaki_relative_entropy > self.umegaki_divergence_max:
            raise UmegakiDivergenceError(
                f"D(ρ‖σ) = {audit_state.umegaki_relative_entropy:.6f} "
                f"> umbral del motor {self.umegaki_divergence_max}."
            )

        logger.info(
            "Auditoría Tomita–Takesaki completada | "
            "λ=%.4f | κ_flow=%.3e | μ_min=%.3e | purity_gap=%.6f | "
            "D_Umegaki=%.6e | F_Uhlmann=%.6f | g_Fisher=%.6e | "
            "Klein_res=%.3e | seguro=%s",
            lambda_magnification,
            flow_data.flow_condition_number,
            gns_data.faithful_spectral_floor,
            gns_data.purity_gap,
            audit_state.umegaki_relative_entropy,
            audit_state.fidelity_uhlmann,
            audit_state.petz_fisher_metric,
            audit_state.klein_inequality_residual,
            audit_state.is_epistemologically_safe,
        )

        return audit_state


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA DEL MÓDULO
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    # Excepciones
    "TomitaTakesakiEngineError",
    "GNSConstructionError",
    "InvalidObservableError",
    "ModularFlowSingularityError",
    "UmegakiDivergenceError",
    "PetzMetricSingularityError",
    # DTOs (objetos del Topos)
    "GNSFibrationData",
    "ModularFlowData",
    "UmegakiExtractionState",
    # Fases anidadas
    "Phase1_GNSConstruction",
    "Phase2_AnalyticModularFlow",
    "Phase3_UmegakiExtraction",
    # Orquestador
    "TomitaTakesakiTelescopicEngine",
]