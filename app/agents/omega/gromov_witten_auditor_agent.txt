# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Gromov-Witten Auditor Agent (Guardián de la Fibra Vertical)         ║
║ Ruta   : app/agents/omega/gromov_witten_auditor_agent.py                     ║
║ Versión: 3.0.0-Gromov-Witten-Bekenstein-Cartan-APS-Doctoral                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GEOMETRÍA SIMPLÉCTICA (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este endofuntor gobierna el espacio topológico ortogonal instanciado por el
`ehresmann_telescopic_engine.py`. Su mandato axiomático es orquestar la inmersión
isométrica de la Matriz de Interacción Central (MIC), gobernar la cinemática del
haz de Ehresmann y compensar el índice espectral de Atiyah–Patodi–Singer (APS).

Para evitar el «Burbujeo de Esferas» (Sphere Bubbling) que corrompería la
A∞-categoría de Fukaya al magnificar el espacio, este agente computa los
Invariantes de Gromov–Witten (GW) virtuales de las clases de curva expulsadas
y re-parametriza la asimetría espectral η(0) antes de que el estado sea
evaluado por el `witten_atiyah_agent.py`.

Fundamentos formales:
  • Bekenstein   : dim ℋ_audit ≥ ⌈e^{S(ρ)}⌉  (capacidad informacional del baño).
  • Cartan       : dω + ½[ω∧ω] = Ω,  P_{H}(𝒯_λ^⊥) = 0  (no-demolición de H_p).
  • Maurer-Cartan: m₀ + m₁(b) + m₂(b,b) = 0 en Λ_nov (anillo de Novikov).
  • Gromov–Witten: GW₀(β) ∼ ½‖b‖_HS²  (volumen virtual de la clase β expulsada).
  • APS          : η_eff(0) = η_raw(0) − GW₀(β)  (compensación del borde espectral).

ARQUITECTURA DE FASES ANIDADAS (Composición Funtorial Estricta):
────────────────────────────────────────────────────────────────────────────────
Fase 1 → Cuantización Dinámica y Entrelazamiento (Bekenstein):
         Computa S(ρ_MIC) = −Tr(ρ ln ρ) y dimensiona ℋ_audit^⊥.
         Morfismo terminal: enforce_bekenstein_bound ↦ BekensteinDimensionData
         ≡ dominio inicial de Fase 2.

Fase 2 → Cinemática del Haz de Ehresmann (Estructura de Cartan):
         Inmersión Stinespring + zoom vertical + verificación
         P_{H_p}(𝒯_λ^⊥) = 0 y Tr_audit[Ad_{I⊗𝒯}(VρV†)] = ρ_MIC.
         Morfismo terminal: certify_cartan_kinematics ↦ CartanKinematicsData
         ≡ dominio inicial de Fase 3.

Fase 3 → Intercepción Gromov–Witten y compensación APS:
         Resuelve MC vía el motor Novikov, extrae la co-cadena b,
         computa GW₀ y re-parametriza η_eff(0).
         Morfismo terminal: compensate_aps_eta ↦ (AuditBundle, GWCompensation)
         ≡ objeto final del endofuntor ℳ → ℳ^⊥.
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
    from app.core.mic_algebra import Morphism, CategoricalState, TopologicalInvariantError
except ImportError:  # pragma: no cover
    class TopologicalInvariantError(Exception):
        r"""Violación a un invariante topológico categórico en el Topos ℰ_MIC."""
        pass

    class Morphism:
        r"""Clase base de Morfismos del Topos ℰ_MIC."""
        pass

    class CategoricalState:
        r"""Estado interno de un objeto del Topos (stub)."""
        pass

# Motor telescópico: fases, DTOs y excepciones públicas
try:
    from app.core.immune_system.ehresmann_telescopic_engine import (
        Phase1_StinespringImmersion,
        Phase2_TelescopicVerticalFibration,
        Phase3_MaurerCartanRegularization,
        StinespringDilationData,
        VerticalFibrationData,
        TelescopicAuditState,
        StinespringDilationError,
        InvalidDensityMatrixError,
        EhresmannFibrationError,
        SphereBubblingAnomalyError,
        TelescopicEngineError,
    )
except ImportError:  # pragma: no cover
    Phase1_StinespringImmersion = object  # type: ignore[misc, assignment]
    Phase2_TelescopicVerticalFibration = object  # type: ignore[misc, assignment]
    Phase3_MaurerCartanRegularization = object  # type: ignore[misc, assignment]

    StinespringDilationData = Any  # type: ignore[misc, assignment]
    VerticalFibrationData = Any  # type: ignore[misc, assignment]
    TelescopicAuditState = Any  # type: ignore[misc, assignment]

    class StinespringDilationError(Exception):
        pass

    class InvalidDensityMatrixError(Exception):
        pass

    class EhresmannFibrationError(Exception):
        pass

    class SphereBubblingAnomalyError(Exception):
        pass

    class TelescopicEngineError(Exception):
        pass

logger = logging.getLogger("MIC.Omega.GromovWittenAuditor")


# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES MATEMÁTICAS, TERMODINÁMICAS Y DE TOLERANCIA
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: float = float(np.finfo(np.float64).eps)
_ENTROPY_TOLERANCE: float = 1e-12
_BUSINESS_INVARIANCE_TOLERANCE: float = 1e-10
_CARTAN_CURVATURE_TOLERANCE: float = 1e-10
_GW_FINITE_CEILING: float = 1e12          # techo de sanidad para volumen GW
_ETA_FINITE_CEILING: float = 1e12
_MIN_AUDIT_DIM: int = 2
_MAX_AUDIT_DIM: int = 256                 # cota operativa anti-runaway
_HS_NORM_FLOOR: float = 1e-16            # piso de norma Hilbert–Schmidt
_BEKENSTEIN_SAFETY_MARGIN: float = 1.0   # holgura aditiva sobre ⌈e^S⌉
_EIGENVALUE_FLOOR: float = 1e-15         # piso espectral para S(ρ) y logs


# ══════════════════════════════════════════════════════════════════════════════
# §B. JERARQUÍA DE EXCEPCIONES SIMPLÉCTICAS Y CUÁNTICAS
# ══════════════════════════════════════════════════════════════════════════════
class GromovWittenAuditorError(TopologicalInvariantError):
    r"""Excepción raíz del Auditor de Gromov–Witten."""
    pass


class BekensteinLimitViolationError(GromovWittenAuditorError):
    r"""
    Detonada si dim ℋ_audit^⊥ < ⌈e^{S(ρ)}⌉ (capacidad informacional insuficiente)
    o si el dimensionamiento excede la cota operativa _MAX_AUDIT_DIM.
    """
    pass


class BusinessInvarianceError(GromovWittenAuditorError):
    r"""
    Detonada si la deformación telescópica altera el estado de negocio
    (violación de la ecuación de estructura de Cartan / no-demolición de H_p).
    """
    pass


class CartanStructureError(GromovWittenAuditorError):
    r"""
    Detonada si la 2-forma de curvatura Ω de la conexión de Ehresmann
    o el proyector horizontal revelan inconsistencia de calibre.
    """
    pass


class SpectralCompensationError(GromovWittenAuditorError):
    r"""
    Detonada si el invariante de Gromov–Witten diverge o la compensación
    APS del η-invariante produce un valor no finito.
    """
    pass


class AuditingPipelineError(GromovWittenAuditorError):
    r"""Fallo de orquestación en la composición funtorial ℳ → ℳ^⊥."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# §C. ESTRUCTURAS INMUTABLES (DTOs del Topos — objetos de las categorías fibra)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class BekensteinDimensionData:
    r"""
    Artefacto terminal de Fase 1.

    Dimensionamiento espectral del baño térmico de auditoría bajo la cota
    de Bekenstein–Hawking informacional:

        dim ℋ_audit ≥ ⌈e^{S(ρ_MIC)}⌉ ,   S(ρ) = −Tr(ρ ln ρ).

    Invariantes:
      • S ∈ [0, ln n_MIC]
      • required_audit_dimension ≥ max(2, ⌈e^S⌉)
      • ∑ eigenvalues = 1,  eigenvalues ≥ 0
    """
    von_neumann_entropy: float
    required_audit_dimension: int
    rho_mic_eigenvalues: NDArray[np.float64]
    effective_hilbert_dimension: float
    bekenstein_saturated: bool


@dataclass(frozen=True, slots=True)
class CartanKinematicsData:
    r"""
    Artefacto terminal de Fase 2 (y dominio inicial de Fase 3).

    Certificación conjunta de:
      (i)  Invariancia del negocio: ‖ρ_MIC' − ρ_MIC‖_F < ε_bus
      (ii) Fuga horizontal nula:   ℓ_H = ‖Π_H 𝒯_λ‖_F < ε_orth
      (iii) Curvatura de Cartan acotada: ‖Ω‖_F < ε_Ω

    Contiene además los artefactos del motor telescópico necesarios
    para que Fase 3 no re-compute la inmersión ni la fibración.
    """
    business_state_difference: float
    is_business_unchanged: bool
    horizontal_leakage_norm: float
    cartan_curvature_norm: float
    dilation_data: StinespringDilationData
    fibration_data: VerticalFibrationData
    bekenstein_data: BekensteinDimensionData


@dataclass(frozen=True, slots=True)
class MaurerCartanSolution:
    r"""
    Solución de la ecuación de Maurer–Cartan expandida sobre Λ_nov.

        F(b) = m₀ + [𝒯_λ, b] + b² = 0,
        b ∈ End(ℋ_audit)  (co-cadena acotante).
    """
    b_cochain: NDArray[np.complex128]
    audit_state: TelescopicAuditState
    residual_frobenius: float
    novikov_filtration_degree: float


@dataclass(frozen=True, slots=True)
class GromovWittenCompensation:
    r"""
    Artefacto terminal de Fase 3 — compensación APS del borde espectral.

    GW₀(β) se estima como la acción de energía de la co-cadena:

        GW₀(β) = ½ ‖b‖_HS² = ½ Tr(b† b),

    y el η efectivo de Atiyah–Patodi–Singer se re-parametriza:

        η_eff(0) = η_raw(0) − GW₀(β).

    Invariantes:
      • 0 ≤ GW₀ < ∞
      • η_eff finito
      • is_ready_for_atiyah_singer ⇒ audit_state.is_safe_for_witten_atiyah
    """
    gw_invariant_volume: float
    gw_chern_simons_secondary: float
    raw_eta_invariant: float
    effective_eta_invariant: float
    bubble_area_class: float
    is_ready_for_atiyah_singer: bool


@dataclass(frozen=True, slots=True)
class AuditBundle:
    r"""
    Objeto final del endofuntor del agente: empaquetadura completa
    del estado auditado + compensación espectral + diagnóstico cinemático.
    """
    audit_state: TelescopicAuditState
    gw_compensation: GromovWittenCompensation
    maurer_cartan_solution: MaurerCartanSolution
    cartan_kinematics: CartanKinematicsData
    bekenstein: BekensteinDimensionData


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 1 · CUANTIZACIÓN DINÁMICA Y LÍMITE DE BEKENSTEIN                     ║
# ║   Dimensiona ℋ_audit^⊥ evaluando la entropía de von Neumann S(ρ_MIC).       ║
# ║                                                                             ║
# ║   Definición formal del objeto terminal de esta fase:                       ║
# ║       enforce_bekenstein_bound : 𝔇(ℋ_MIC) → BekensteinDimensionData         ║
# ║   Este morfismo es el dominio de partida de Fase 2.                         ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_Bekenstein(Phase1_StinespringImmersion):
    r"""
    Calcula la entropía de von Neumann y dimensiona el baño de auditoría
    obedeciendo el límite de Bekenstein informacional.

    Enunciado operativo
    -------------------
    Dado ρ ∈ 𝔇(ℋ_MIC),

        S(ρ) = −Tr(ρ ln ρ) = −∑_i μ_i ln μ_i ,   μ = spec(ρ) \ {0},

    el número efectivo de grados de libertad es d_eff = e^{S(ρ)}, y la
    dimensión del baño debe satisfacer

        dim ℋ_audit^⊥ ≥ max(2, ⌈d_eff + margin⌉)

    para que el rango de Kraus del canal de auditoría pueda soportar la
    información de entrelazamiento sin colapso termodinámico.

    Extiende Phase1_StinespringImmersion: hereda validación espectral de ρ
    e inmersión isométrica, añadiendo el pre-dimensionamiento dinámico.
    """

    # ── §1.1  Entropía de von Neumann con regularización espectral ───────────

    def _compute_von_neumann_entropy(
        self,
        rho_mic: NDArray[np.complex128],
    ) -> Tuple[float, NDArray[np.float64]]:
        r"""
        Calcula rigurosamente

            S(ρ) = −Tr(ρ ln ρ)

        en nats, con:
          • validación de ρ ∈ 𝔇(ℋ) (heredada),
          • clip de autovalores al piso _EIGENVALUE_FLOOR (evita −∞ en el log),
          • renormalización ∑ μ_i = 1,
          • cota analítica S ∈ [0, ln n].

        Parámetros
        ----------
        rho_mic : NDArray[np.complex128]
            Matriz de densidad del sistema MIC.

        Retorna
        -------
        entropy : float
            S(ρ) en nats, S ∈ [0, ln n].
        eigenvalues : NDArray[np.float64]
            Espectro μ renormalizado (∑ μ = 1, μ ≥ 0).
        """
        # Validación heredada del motor Stinespring (si está disponible)
        if hasattr(self, "_validate_density_matrix"):
            rho_mic = self._validate_density_matrix(rho_mic, name="rho_mic")
        else:  # pragma: no cover
            rho_mic = 0.5 * (rho_mic + rho_mic.conj().T)

        n = int(rho_mic.shape[0])
        eigenvalues = la.eigvalsh(rho_mic).astype(np.float64)
        eigenvalues = np.clip(eigenvalues, 0.0, None)

        total = float(eigenvalues.sum())
        if total < _MACHINE_EPSILON:
            raise InvalidDensityMatrixError(
                "Espectro nulo: Tr ρ ≈ 0; imposible calcular S(ρ)."
            )
        eigenvalues /= total

        # Contribución entropica: 0·ln 0 := 0 (límite estándar)
        positive = eigenvalues[eigenvalues > _EIGENVALUE_FLOOR]
        if positive.size == 0:
            entropy = 0.0
        else:
            entropy = float(-np.sum(positive * np.log(positive)))

        # Cotas analíticas: 0 ≤ S ≤ ln n
        s_max = math.log(n) if n > 1 else 0.0
        if entropy < -_ENTROPY_TOLERANCE:
            raise BekensteinLimitViolationError(
                f"S(ρ) = {entropy:.6e} < 0 (patología numérica)."
            )
        entropy = float(np.clip(entropy, 0.0, s_max + _ENTROPY_TOLERANCE))
        if entropy > s_max + _ENTROPY_TOLERANCE:
            entropy = s_max

        return entropy, eigenvalues

    # ── §1.2  Dimensionamiento Bekenstein con cota operativa ─────────────────

    @staticmethod
    def _bekenstein_dimension(
        entropy: float,
        safety_margin: float = _BEKENSTEIN_SAFETY_MARGIN,
    ) -> Tuple[int, float, bool]:
        r"""
        Traduce S ↦ (dim_audit, d_eff, saturated).

            d_eff  = exp(S),
            dim    = max(2, ⌈d_eff + margin⌉),
            saturated ⇔ dim fue recortado a _MAX_AUDIT_DIM.
        """
        if not math.isfinite(entropy) or entropy < 0.0:
            raise BekensteinLimitViolationError(
                f"Entropía no física: S = {entropy}."
            )

        d_eff = float(math.exp(entropy))
        raw = int(math.ceil(d_eff + safety_margin))
        required = max(raw, _MIN_AUDIT_DIM)

        saturated = False
        if required > _MAX_AUDIT_DIM:
            logger.warning(
                "Bekenstein dim=%d excede cota operativa %d; se satura.",
                required, _MAX_AUDIT_DIM,
            )
            required = _MAX_AUDIT_DIM
            saturated = True

        return required, d_eff, saturated

    # ── §1.3  MORFISMO TERMINAL DE FASE 1 ────────────────────────────────────
    #          (dominio inicial de Fase 2)

    def enforce_bekenstein_bound(
        self,
        rho_mic: NDArray[np.complex128],
    ) -> BekensteinDimensionData:
        r"""
        Exige incondicionalmente

            dim(ℋ_audit^⊥) ≥ ⌈e^{S(ρ_MIC)}⌉

        y empaqueta el artefacto terminal de Fase 1.

            Φ₁ᴳᵂ : 𝔇(ℋ_MIC) ⟶ BekensteinDimensionData.

        Parámetros
        ----------
        rho_mic : NDArray[np.complex128]
            Matriz de densidad del sistema MIC (n × n).

        Retorna
        -------
        BekensteinDimensionData
            Objeto terminal de Fase 1 / dominio de Fase 2.
        """
        entropy, eigenvalues = self._compute_von_neumann_entropy(rho_mic)
        required_dim, d_eff, saturated = self._bekenstein_dimension(entropy)

        # Verificación de holgura: dim ≥ d_eff (salvo saturación operativa)
        if not saturated and required_dim < d_eff - 1e-9:
            raise BekensteinLimitViolationError(
                f"dim_audit={required_dim} < d_eff={d_eff:.6f}; "
                f"violación estricta del límite de Bekenstein."
            )

        logger.debug(
            "Bekenstein | S=%.6f nats | d_eff=%.4f | dim_audit=%d | saturated=%s",
            entropy, d_eff, required_dim, saturated,
        )

        return BekensteinDimensionData(
            von_neumann_entropy=entropy,
            required_audit_dimension=required_dim,
            rho_mic_eigenvalues=eigenvalues,
            effective_hilbert_dimension=d_eff,
            bekenstein_saturated=saturated,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 2 · CINEMÁTICA DEL HAZ DE EHRESMANN (ESTRUCTURA DE CARTAN)           ║
# ║   Garantiza axiomáticamente la no-demolición del estado MIC.                ║
# ║                                                                             ║
# ║   Continuación funtorial: el morfismo terminal de Fase 1                    ║
# ║       enforce_bekenstein_bound ↦ BekensteinDimensionData                    ║
# ║   es el dominio de                                                         ║
# ║       certify_cartan_kinematics : 𝔇(ℋ)×ℝ_{≥0}×BekensteinDimensionData      ║
# ║           → CartanKinematicsData                                            ║
# ║   Este morfismo terminal de Fase 2 es el dominio de partida de Fase 3.      ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_EhresmannCartan(Phase2_TelescopicVerticalFibration, Phase1_Bekenstein):
    r"""
    Certifica que el tensor telescópico 𝒯_λ^⊥ actúa exclusivamente sobre la
    fibra vertical V_p = ker(dπ) y deja invariante el estado de negocio
    (subespacio horizontal H_p), en el sentido de la ecuación de estructura
    de Cartan.

    Pipeline cinemático
    -------------------
    1. Inmersión de Stinespring con dim_audit dictada por Bekenstein.
    2. Fibración vertical + zoom espectral (motor Ehresmann).
    3. Verificación de fuga horizontal ℓ_H y curvatura ‖Ω‖.
    4. Identidad de no-demolición:

         ρ_MIC  =?  Tr_audit[ (I ⊗ 𝒯_λ) V ρ V† (I ⊗ 𝒯_λ)† ].

    La herencia múltiple incorpora Φ₁/Φ₂ del motor telescópico y el
    dimensionamiento Bekenstein de Fase 1 del agente.
    """

    # ── §2.1  Traza parcial sobre el factor de auditoría ─────────────────────

    @staticmethod
    def _partial_trace_audit(
        rho_combined: NDArray[np.complex128],
        dim_mic: int,
        dim_audit: int,
    ) -> NDArray[np.complex128]:
        r"""
        Traza parcial sobre el baño de auditoría:

            ρ_MIC' = Tr_audit(ρ_combined)

        para un estado en ℋ_MIC ⊗ ℋ_audit con orden de Kronecker
        (MIC, audit) — reshape (n_M, n_a, n_M, n_a), traza en ejes (1, 3).
        """
        expected = dim_mic * dim_audit
        if rho_combined.shape != (expected, expected):
            raise CartanStructureError(
                f"Dimensión incompatible en traza parcial audit: "
                f"shape={rho_combined.shape}, esperado ({expected},{expected})."
            )
        tensor = rho_combined.reshape(dim_mic, dim_audit, dim_mic, dim_audit)
        rho_mic = np.trace(tensor, axis1=1, axis2=3)
        rho_mic = 0.5 * (rho_mic + rho_mic.conj().T)
        return rho_mic.astype(np.complex128)

    # ── §2.2  Identidad de no-demolición (Estructura de Cartan) ──────────────

    def verify_business_invariance(
        self,
        rho_mic: NDArray[np.complex128],
        dilation_data: StinespringDilationData,
        T_lambda_audit: NDArray[np.complex128],
    ) -> Tuple[float, bool]:
        r"""
        Verifica la no-demolición del estado de negocio bajo el zoom:

            Tr_audit[ Ad_{I_MIC ⊗ 𝒯_λ}(V ρ V†) ]  =  ρ_MIC.

        Geométricamente: 𝒯_λ vive en V_p ⇒ su extensión por Kronecker con
        I_MIC es un automorfismo vertical que conmuta con la traza parcial
        sobre la fibra, dejando invariante la base B ≃ ℋ_MIC.

        Retorna
        -------
        diff : float
            ‖ρ_MIC' − ρ_MIC‖_F.
        unchanged : bool
            True sisi diff ≤ ε_bus.
        """
        if hasattr(self, "_validate_density_matrix"):
            rho_mic = self._validate_density_matrix(rho_mic, name="rho_mic")

        dim_mic = int(rho_mic.shape[0])
        dim_audit = int(T_lambda_audit.shape[0])
        V = dilation_data.V_isometry

        if V.shape != (dim_mic * dim_audit, dim_mic):
            raise CartanStructureError(
                f"Isometría V con shape {V.shape} incompatible con "
                f"(dim_mic={dim_mic}, dim_audit={dim_audit})."
            )
        if T_lambda_audit.shape != (dim_audit, dim_audit):
            raise CartanStructureError(
                f"𝒯_λ shape {T_lambda_audit.shape} ≠ ({dim_audit},{dim_audit})."
            )

        # Extensión vertical: I_MIC ⊗ 𝒯_λ
        I_mic = np.eye(dim_mic, dtype=np.complex128)
        T_total = np.kron(I_mic, T_lambda_audit)

        # Estado dilatado y deformado
        rho_comb = V @ rho_mic @ V.conj().T
        rho_comb = 0.5 * (rho_comb + rho_comb.conj().T)
        rho_def = T_total @ rho_comb @ T_total.conj().T
        rho_def = 0.5 * (rho_def + rho_def.conj().T)

        # Reducción al sistema de negocio
        rho_after = self._partial_trace_audit(rho_def, dim_mic, dim_audit)

        # Renormalización defensiva
        tr_after = float(np.real(np.trace(rho_after)))
        if tr_after < _MACHINE_EPSILON:
            raise BusinessInvarianceError(
                "Estado de negocio post-deformación con traza nula "
                "(colapso de la base presupuestaria)."
            )
        rho_after = rho_after / tr_after

        diff = float(la.norm(rho_after - rho_mic, ord="fro"))
        unchanged = diff <= _BUSINESS_INVARIANCE_TOLERANCE

        if not unchanged:
            raise BusinessInvarianceError(
                f"La deformación telescópica perturbó el estado del negocio. "
                f"‖Δρ‖_F = {diff:.3e} > ε_bus = {_BUSINESS_INVARIANCE_TOLERANCE:.1e}."
            )

        return diff, unchanged

    # ── §2.3  Diagnóstico de curvatura de Cartan / fuga horizontal ───────────

    @staticmethod
    def _diagnose_ehresmann_gauge(
        fibration_data: VerticalFibrationData,
    ) -> Tuple[float, float]:
        r"""
        Extrae (ℓ_H, ‖Ω‖) del artefacto de fibración y valida cotas.

        Raises
        ------
        CartanStructureError
            Si la fuga horizontal o la curvatura exceden tolerancia.
        """
        leakage = float(fibration_data.horizontal_leakage_norm)
        # Compatibilidad: el motor v3 expone connection_curvature_norm;
        # stubs antiguos pueden no tenerlo.
        curv = float(getattr(fibration_data, "connection_curvature_norm", 0.0))

        if leakage > _BUSINESS_INVARIANCE_TOLERANCE:
            raise CartanStructureError(
                f"Fuga horizontal ℓ_H = {leakage:.3e} excede tolerancia "
                f"{_BUSINESS_INVARIANCE_TOLERANCE:.1e} (P_H 𝒯_λ ≠ 0)."
            )
        if curv > _CARTAN_CURVATURE_TOLERANCE:
            raise CartanStructureError(
                f"Curvatura de Cartan ‖Ω‖_F = {curv:.3e} excede "
                f"{_CARTAN_CURVATURE_TOLERANCE:.1e} (calibre inconsistente)."
            )
        return leakage, curv

    # ── §2.4  MORFISMO TERMINAL DE FASE 2 ────────────────────────────────────
    #          (continuación de enforce_bekenstein_bound;
    #           dominio inicial de Fase 3)

    def certify_cartan_kinematics(
        self,
        rho_business: NDArray[np.complex128],
        lambda_magnification: float,
        bekenstein_data: BekensteinDimensionData,
    ) -> CartanKinematicsData:
        r"""
        Ejecuta la cinemática completa del haz y certifica Cartan:

            Φ₂ᴳᵂ : 𝔇(ℋ) × ℝ_{≥0} × BekensteinDimensionData
                    ⟶ CartanKinematicsData.

        Pipeline
        --------
        1. Inmersión Stinespring con dim = bekenstein.required_audit_dimension.
        2. Deformación telescópica vertical (zoom λ).
        3. Diagnóstico de calibre (ℓ_H, ‖Ω‖).
        4. Identidad de no-demolición del negocio.
        5. Empaquetar CartanKinematicsData (dominio de Fase 3).

        Parámetros
        ----------
        rho_business : NDArray[np.complex128]
            Estado de negocio ρ_MIC.
        lambda_magnification : float
            Factor de magnificación λ ≥ 0.
        bekenstein_data : BekensteinDimensionData
            Salida de Φ₁ᴳᵂ = enforce_bekenstein_bound.

        Retorna
        -------
        CartanKinematicsData
            Objeto terminal de Fase 2 / dominio de Fase 3.
        """
        if lambda_magnification < 0.0:
            raise ValueError(
                f"lambda_magnification ≥ 0 requerido; recibido {lambda_magnification}."
            )

        dim_audit = int(bekenstein_data.required_audit_dimension)
        if dim_audit < _MIN_AUDIT_DIM:
            raise BekensteinLimitViolationError(
                f"dim_audit={dim_audit} < mínimo {_MIN_AUDIT_DIM}."
            )

        # (a) Inmersión isométrica — morfismo del motor Φ₁
        dilation_data = self.compute_isometric_immersion(rho_business, dim_audit)

        # (b) Fibración vertical + zoom — morfismo del motor Φ₂
        fibration_data = self.apply_telescopic_deformation(
            dilation_data, lambda_magnification
        )

        # (c) Diagnóstico de calibre Ehresmann/Cartan
        leakage, curv = self._diagnose_ehresmann_gauge(fibration_data)

        # (d) No-demolición del negocio
        diff, unchanged = self.verify_business_invariance(
            rho_business,
            dilation_data,
            fibration_data.T_lambda_vertical,
        )

        logger.debug(
            "Cartan | ‖Δρ‖_F=%.3e | ℓ_H=%.3e | ‖Ω‖=%.3e | unchanged=%s",
            diff, leakage, curv, unchanged,
        )

        return CartanKinematicsData(
            business_state_difference=diff,
            is_business_unchanged=unchanged,
            horizontal_leakage_norm=leakage,
            cartan_curvature_norm=curv,
            dilation_data=dilation_data,
            fibration_data=fibration_data,
            bekenstein_data=bekenstein_data,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   FASE 3 · INTERCEPCIÓN GROMOV–WITTEN Y COMPENSACIÓN APS                    ║
# ║   Purga el burbujeo de esferas y re-parametriza el η-invariante.            ║
# ║                                                                             ║
# ║   Continuación funtorial: el morfismo terminal de Fase 2                    ║
# ║       certify_cartan_kinematics ↦ CartanKinematicsData                      ║
# ║   es el dominio de                                                         ║
# ║       compensate_aps_eta : CartanKinematicsData × ℝ → AuditBundle           ║
# ║   Este morfismo cierra el endofuntor ℳ → ℳ^⊥.                               ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_GromovWitten(Phase3_MaurerCartanRegularization, Phase2_EhresmannCartan):
    r"""
    Resuelve la ecuación de Maurer–Cartan (reutilizando el motor Novikov del
    telescópico), extrae la co-cadena acotante b, computa el invariante de
    Gromov–Witten de la clase de burbuja y compensa el η-invariante APS.

    Estructuras
    -----------
    • Motor MC (heredado):

          resolve_maurer_cartan_novikov : (VerticalFibrationData, ρ_audit)
              → TelescopicAuditState

      con b accesible en ``audit_state.maurer_cartan_bounding_cochain``
      (motor v3) o, en fallback, re-derivado localmente.

    • Invariante GW (volumen virtual de la clase β):

          GW₀(β) = ½ ‖b‖_HS² = ½ Tr(b† b)

      y secundario tipo Chern–Simons (fase imaginaria):

          CS₃(b) = (1/12π²) Im Tr(b³)   (diagnóstico de holonomía).

    • Compensación APS:

          η_eff(0) = η_raw(0) − GW₀(β).
    """

    # ── §3.1  Extracción de la co-cadena acotante ────────────────────────────

    @staticmethod
    def _extract_bounding_cochain(
        audit_state: TelescopicAuditState,
        fallback_b: Optional[NDArray[np.complex128]] = None,
    ) -> NDArray[np.complex128]:
        r"""
        Obtiene b ∈ End(ℋ_audit) desde el TelescopicAuditState del motor v3
        (campo ``maurer_cartan_bounding_cochain``) o desde un fallback local.
        """
        b = getattr(audit_state, "maurer_cartan_bounding_cochain", None)
        if b is not None:
            return np.asarray(b, dtype=np.complex128)
        if fallback_b is not None:
            return np.asarray(fallback_b, dtype=np.complex128)
        raise SpectralCompensationError(
            "Co-cadena acotante b ausente en TelescopicAuditState y sin fallback."
        )

    # ── §3.2  Invariantes de Gromov–Witten y Chern–Simons secundario ─────────

    @staticmethod
    def _gromov_witten_invariants(
        b: NDArray[np.complex128],
    ) -> Tuple[float, float, float]:
        r"""
        Computa la terna (GW₀, CS₃, area_class) a partir de la co-cadena b.

            GW₀(β)     = ½ ‖b‖_HS² = ½ Tr(b† b)     ≥ 0
            CS₃(b)     = Im Tr(b³) / (12 π²)         (secundario)
            area_class = ‖b‖_HS                       (filtración de área)

        Raises
        ------
        SpectralCompensationError
            Si algún invariante no es finito o GW₀ excede el techo de sanidad.
        """
        if b.ndim != 2 or b.shape[0] != b.shape[1]:
            raise SpectralCompensationError(
                f"Co-cadena b no cuadrada: shape={b.shape}."
            )

        # Norma Hilbert–Schmidt
        hs2 = float(np.real(np.trace(b.conj().T @ b)))
        if hs2 < 0.0:
            # Solo posible por ruido numérico
            hs2 = abs(hs2)
        hs2 = max(hs2, 0.0)

        gw0 = 0.5 * hs2
        area = math.sqrt(hs2) if hs2 > _HS_NORM_FLOOR else 0.0

        # Secundario de Chern–Simons (diagnóstico; no afecta η_eff)
        b3_tr = complex(np.trace(b @ b @ b))
        cs3 = float(b3_tr.imag) / (12.0 * math.pi ** 2)

        for name, val in (("GW₀", gw0), ("CS₃", cs3), ("area", area)):
            if not math.isfinite(val):
                raise SpectralCompensationError(
                    f"Invariante {name} no finito: {val}."
                )
        if gw0 > _GW_FINITE_CEILING:
            raise SpectralCompensationError(
                f"GW₀ = {gw0:.3e} excede techo de sanidad {_GW_FINITE_CEILING:.1e} "
                f"(burbujeo de volumen incontrolable)."
            )

        return gw0, cs3, area

    # ── §3.3  Compensación del η-invariante (APS) ────────────────────────────

    @staticmethod
    def _compensate_eta(
        raw_eta: float,
        gw0: float,
    ) -> float:
        r"""
        Re-parametrización APS del borde espectral:

            η_eff(0) = η_raw(0) − GW₀(β).

        Garantiza finitud y cota de sanidad.
        """
        if not math.isfinite(raw_eta):
            raise SpectralCompensationError(
                f"η_raw no finito: {raw_eta}."
            )
        if abs(raw_eta) > _ETA_FINITE_CEILING:
            raise SpectralCompensationError(
                f"|η_raw| = {abs(raw_eta):.3e} excede techo {_ETA_FINITE_CEILING:.1e}."
            )

        eta_eff = float(raw_eta - gw0)
        if not math.isfinite(eta_eff):
            raise SpectralCompensationError(
                f"η_eff no finito tras compensación APS (raw={raw_eta}, GW={gw0})."
            )
        return eta_eff

    # ── §3.4  Resolución MC con extracción de b (delega al motor) ────────────

    def resolve_maurer_cartan_with_gw(
        self,
        fibration_data: VerticalFibrationData,
        rho_audit: NDArray[np.complex128],
        raw_eta_invariant: float,
    ) -> Tuple[MaurerCartanSolution, GromovWittenCompensation]:
        r"""
        Resuelve MC vía el motor Novikov heredado y computa la compensación GW/APS.

        Preferencia de ruta
        -------------------
        1. Delegar en ``resolve_maurer_cartan_novikov`` (motor v3, con b en el DTO).
        2. Si el DTO no expone b (stub antiguo), no hay fallback silencioso:
           se lanza SpectralCompensationError.

        Parámetros
        ----------
        fibration_data : VerticalFibrationData
            Tensor vertical de Fase 2 (motor).
        rho_audit : NDArray[np.complex128]
            Estado del baño (Stinespring).
        raw_eta_invariant : float
            η_raw(0) provisto por el ecosistema.

        Retorna
        -------
        MaurerCartanSolution
            Co-cadena b + estado auditado + residual.
        GromovWittenCompensation
            GW₀, CS₃, η_eff, bandera APS.
        """
        # Delegación al motor telescópico (Φ₃ del engine)
        try:
            audit_state: TelescopicAuditState = self.resolve_maurer_cartan_novikov(
                fibration_data, rho_audit
            )
        except SphereBubblingAnomalyError:
            raise
        except TelescopicEngineError as exc:
            raise SphereBubblingAnomalyError(
                f"Motor telescópico falló en MC/Novikov: {exc}"
            ) from exc
        except Exception as exc:  # pragma: no cover
            # Stubs sin el método: señalizar claramente
            if not hasattr(self, "resolve_maurer_cartan_novikov"):
                raise AuditingPipelineError(
                    "resolve_maurer_cartan_novikov no disponible "
                    "(motor telescópico no importado)."
                ) from exc
            raise

        # Co-cadena acotante
        b = self._extract_bounding_cochain(audit_state)

        # Residual reportado por el motor
        residual = float(getattr(audit_state, "landau_ginzburg_potential", 0.0))
        novikov_deg = float(
            getattr(audit_state, "novikov_filtration_degree", float(la.norm(b, ord="fro")))
        )

        mc_solution = MaurerCartanSolution(
            b_cochain=b,
            audit_state=audit_state,
            residual_frobenius=residual,
            novikov_filtration_degree=novikov_deg,
        )

        # Invariantes GW + compensación APS
        gw0, cs3, area = self._gromov_witten_invariants(b)
        eta_eff = self._compensate_eta(raw_eta_invariant, gw0)

        safe = bool(getattr(audit_state, "is_safe_for_witten_atiyah", False))
        ready = bool(safe and math.isfinite(eta_eff) and gw0 < _GW_FINITE_CEILING)

        gw_compensation = GromovWittenCompensation(
            gw_invariant_volume=gw0,
            gw_chern_simons_secondary=cs3,
            raw_eta_invariant=float(raw_eta_invariant),
            effective_eta_invariant=eta_eff,
            bubble_area_class=area,
            is_ready_for_atiyah_singer=ready,
        )

        return mc_solution, gw_compensation

    # ── §3.5  MORFISMO TERMINAL DE FASE 3 ────────────────────────────────────
    #          (continuación de certify_cartan_kinematics;
    #           cierra el endofuntor ℳ → ℳ^⊥)

    def compensate_aps_eta(
        self,
        cartan_data: CartanKinematicsData,
        raw_eta_invariant: float,
    ) -> AuditBundle:
        r"""
        Cierra la composición funtorial del agente:

            Φ₃ᴳᵂ : CartanKinematicsData × ℝ ⟶ AuditBundle.

        Pipeline
        --------
        1. Extraer fibration_data y ρ_audit del artefacto de Fase 2.
        2. Resolver MC + GW + η_eff (resolve_maurer_cartan_with_gw).
        3. Empaquetar AuditBundle (objeto final).

        Parámetros
        ----------
        cartan_data : CartanKinematicsData
            Salida de Φ₂ᴳᵂ = certify_cartan_kinematics.
        raw_eta_invariant : float
            η_raw(0) del ecosistema.

        Retorna
        -------
        AuditBundle
            Estado auditado + compensación APS + diagnóstico completo.
        """
        if not cartan_data.is_business_unchanged:
            raise BusinessInvarianceError(
                "No se puede compensar APS: invariancia del negocio no certificada."
            )

        fibration = cartan_data.fibration_data
        rho_audit = cartan_data.dilation_data.rho_audit_subspace

        mc_solution, gw_compensation = self.resolve_maurer_cartan_with_gw(
            fibration, rho_audit, raw_eta_invariant
        )

        return AuditBundle(
            audit_state=mc_solution.audit_state,
            gw_compensation=gw_compensation,
            maurer_cartan_solution=mc_solution,
            cartan_kinematics=cartan_data,
            bekenstein=cartan_data.bekenstein_data,
        )


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║                                                                             ║
# ║   ORQUESTADOR SUPREMO · GROMOV–WITTEN AUDITOR AGENT                         ║
# ║   Endofuntor 𝒜 = Φ₃ᴳᵂ ∘ Φ₂ᴳᵂ ∘ Φ₁ᴳᵂ : 𝔇(ℋ)×ℝ_{≥0}×ℝ → AuditBundle         ║
# ║   Morfismo ℳ → ℳ^⊥ en el Topos de Grothendieck.                             ║
# ║                                                                             ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class GromovWittenAuditorAgent(Morphism, Phase3_GromovWitten):
    r"""
    Guardián de la medición cuántica y auditor de la fibra vertical.

    Garantiza que la lupa de auditoría no colapse ni altere el estado
    financiero base, purificando la métrica óptica en tiempo real y
    entregando un η_eff compensado listo para el funtor de
    Witten–Atiyah–Floer.

    Composición funtorial estricta
    ------------------------------
    ``execute_auditing_process`` encadena:

        ρ_business  ──Φ₁ᴳᵂ──►  BekensteinDimensionData
                    ──Φ₂ᴳᵂ──►  CartanKinematicsData
                    ──Φ₃ᴳᵂ──►  AuditBundle.

    Cada Φᵢᴳᵂ es el morfismo terminal de la fase i y el dominio de la
    fase i+1, realizando la anidación funtorial exigida por el Topos.
    """

    def __init__(self, default_eta_invariant: float = 0.0) -> None:
        r"""
        Parámetros
        ----------
        default_eta_invariant : float
            η_raw(0) por defecto si el caller no provee uno explícito
            en ``execute_auditing_process``.
        """
        if not math.isfinite(default_eta_invariant):
            raise ValueError(
                f"default_eta_invariant debe ser finito; recibido {default_eta_invariant}."
            )
        self.default_eta_invariant = float(default_eta_invariant)

    # ── Orquestación principal ───────────────────────────────────────────────

    def execute_auditing_process(
        self,
        rho_business: NDArray[np.complex128],
        lambda_magnification: float,
        current_eta_invariant: Optional[float] = None,
    ) -> Tuple[TelescopicAuditState, GromovWittenCompensation]:
        r"""
        Orquesta el ciclo completo de auditoría no destructiva.

        Composición: 𝒜 = Φ₃ᴳᵂ ∘ Φ₂ᴳᵂ ∘ Φ₁ᴳᵂ.

        Parámetros
        ----------
        rho_business : NDArray[np.complex128]
            Matriz de densidad del sistema de negocio (n × n).
        lambda_magnification : float
            Factor de magnificación telescópica λ ≥ 0.
        current_eta_invariant : float, optional
            η_raw(0) actual; si es None se usa ``default_eta_invariant``.

        Retorna
        -------
        TelescopicAuditState
            Estado auditado, regularizado topológicamente.
        GromovWittenCompensation
            Datos de la compensación espectral APS.
        """
        bundle = self.execute_auditing_bundle(
            rho_business=rho_business,
            lambda_magnification=lambda_magnification,
            current_eta_invariant=current_eta_invariant,
        )
        return bundle.audit_state, bundle.gw_compensation

    def execute_auditing_bundle(
        self,
        rho_business: NDArray[np.complex128],
        lambda_magnification: float,
        current_eta_invariant: Optional[float] = None,
    ) -> AuditBundle:
        r"""
        Variante que devuelve el ``AuditBundle`` completo (diagnóstico total).

        Preferible para pipelines que necesiten la co-cadena b, el grado
        de Novikov o los datos de Cartan/Bekenstein aguas abajo.
        """
        if lambda_magnification < 0.0:
            raise ValueError(
                f"lambda_magnification ≥ 0 requerido; recibido {lambda_magnification}."
            )

        eta_raw = (
            self.default_eta_invariant
            if current_eta_invariant is None
            else float(current_eta_invariant)
        )
        if not math.isfinite(eta_raw):
            raise SpectralCompensationError(f"η_raw no finito: {eta_raw}.")

        try:
            # ── Φ₁ᴳᵂ · Fase 1 · Bekenstein ────────────────────────────────────
            bekenstein_data: BekensteinDimensionData = self.enforce_bekenstein_bound(
                rho_business
            )
            logger.info(
                "Fase 1 completa | S=%.6f nats | d_eff=%.4f | dim_audit=%d | sat=%s",
                bekenstein_data.von_neumann_entropy,
                bekenstein_data.effective_hilbert_dimension,
                bekenstein_data.required_audit_dimension,
                bekenstein_data.bekenstein_saturated,
            )

            # ── Φ₂ᴳᵂ · Fase 2 · Cinemática de Cartan / Ehresmann ──────────────
            #     Dominio = objeto terminal de Φ₁ᴳᵂ
            cartan_data: CartanKinematicsData = self.certify_cartan_kinematics(
                rho_business, lambda_magnification, bekenstein_data
            )
            logger.info(
                "Fase 2 completa | ‖Δρ‖_F=%.3e | ℓ_H=%.3e | ‖Ω‖=%.3e | unchanged=%s",
                cartan_data.business_state_difference,
                cartan_data.horizontal_leakage_norm,
                cartan_data.cartan_curvature_norm,
                cartan_data.is_business_unchanged,
            )

            # ── Φ₃ᴳᵂ · Fase 3 · Gromov–Witten + compensación APS ──────────────
            #     Dominio = objeto terminal de Φ₂ᴳᵂ
            bundle: AuditBundle = self.compensate_aps_eta(cartan_data, eta_raw)

            logger.info(
                "Fase 3 completa | Novikov_iters=%s | W_LG=%.4e | "
                "GW₀=%.4e | CS₃=%.4e | area=%.4e | η_raw=%.4e | η_eff=%.4e | "
                "APS_ready=%s | Witten_safe=%s",
                getattr(bundle.audit_state, "novikov_convergence_iterations", "?"),
                float(getattr(bundle.audit_state, "landau_ginzburg_potential", float("nan"))),
                bundle.gw_compensation.gw_invariant_volume,
                bundle.gw_compensation.gw_chern_simons_secondary,
                bundle.gw_compensation.bubble_area_class,
                bundle.gw_compensation.raw_eta_invariant,
                bundle.gw_compensation.effective_eta_invariant,
                bundle.gw_compensation.is_ready_for_atiyah_singer,
                getattr(bundle.audit_state, "is_safe_for_witten_atiyah", False),
            )

            return bundle

        except (
            GromovWittenAuditorError,
            TelescopicEngineError,
            SphereBubblingAnomalyError,
            StinespringDilationError,
            EhresmannFibrationError,
            InvalidDensityMatrixError,
        ):
            # Re-lanzar excepciones de dominio sin envolver
            raise
        except ValueError:
            raise
        except Exception as exc:  # pragma: no cover
            raise AuditingPipelineError(
                f"Fallo no catalogado en la tubería de auditoría GW: {exc}"
            ) from exc


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA DEL MÓDULO
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    # Excepciones
    "GromovWittenAuditorError",
    "BekensteinLimitViolationError",
    "BusinessInvarianceError",
    "CartanStructureError",
    "SpectralCompensationError",
    "AuditingPipelineError",
    # DTOs (objetos del Topos)
    "BekensteinDimensionData",
    "CartanKinematicsData",
    "MaurerCartanSolution",
    "GromovWittenCompensation",
    "AuditBundle",
    # Fases anidadas
    "Phase1_Bekenstein",
    "Phase2_EhresmannCartan",
    "Phase3_GromovWitten",
    # Orquestador
    "GromovWittenAuditorAgent",
]