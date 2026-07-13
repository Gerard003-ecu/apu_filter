# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo: Watcher Agent (Funtor Covariante de Propagación Tensorial)           ║
║ Ubicación: app/agents/core/immune_system/watcher_agent.py                    ║
║ Versión: 5.0.0-Nested-Spectral-Topos-Cauchy                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y TOPOLOGÍA ALGEBRAICA:
────────────────────────────────────────────────────────────────────────────────
Este módulo opera como el Funtor Covariante de Propagación Tensorial (𝒲_agent)
en el 2-categoría de haces de módulos sobre el sitio de Grothendieck de la
membrana p-Laplaciana. Transporta el Tensor de Estrés de Cauchy-Momentum
𝒯^{μν} (objeto en Γ(Sym² Tℳ)) hacia el Topos de Grothendieck (MICAgent) y el
Interferómetro de Holonomía (SheafCohomologyOrchestrator).

Arquitectura en tres fases anidadas (composición de morfismos funtoriales):

  Fase 1 → Validación espectral-covariante y extracción de invariantes
           (conservación ∇_μ 𝒯^{μν}=0, espectro de G, traza mixta, cond(G)).
  Fase 2 (anidada en Fase 1) → Pushforward cohomológico de Dirichlet
           (energía deformada ℰ_deform + cota de Rayleigh + radio espectral).
  Fase 3 (anidada en Fase 2) → Pullback al topos de Grothendieck
           (L_max por norma de operador, predicado booleano de paranoia).

AXIOMAS DE EJECUCIÓN (nivel PhD — geometría de Riemann + teoría espectral):
────────────────────────────────────────────────────────────────────────────────
§1. Conservación del Momento Ciber-Físico (conexión de Levi-Civita ∇):
    (∇_μ 𝒯)^{ν} := ∂_μ 𝒯^{μν} + Γ^μ_{μλ} 𝒯^{λν} + Γ^ν_{μλ} 𝒯^{μλ} = 0
    Residuo relativo: ‖∇𝒯‖₂ / (‖𝒯‖_F + ε_mach) < τ.

§2. Energía de Dirichlet Deformada (pushforward en H¹_sheaf):
    ℰ_deform(δx) = ⟨δx, 𝒯♭ δx⟩_G = 𝒯_{μν} (δx)^μ (δx)^ν
    Cota de Rayleigh: |ℰ| ≤ ‖𝒯‖_{op} ‖δx‖₂²
    donde ‖𝒯‖_{op} = ρ(|𝒯|) (radio espectral del valor absoluto).

§3. Restricción de Lipschitz Homomórfica (pullback en el topos ℰ = Sh(C,J)):
    L_max(𝒯) = κ₀ / ( √|Tr_G(𝒯)| + ε_mach · λ_max(G) · κ(G)^{1/2} )
    Predicado de lockdown (álgebra de Boole de predicados estables):
    ZeroTrust ⇔ (L_max < θ_crit) ∨ (cond(G) > κ_max) ∨ (ρ(𝒯) > ρ_crit).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Optional, Tuple, Final

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ══════════════════════════════════════════════════════════════════════════════
# DEPENDENCIAS ESTRUCTURALES DEL ECOSISTEMA APU FILTER
# ══════════════════════════════════════════════════════════════════════════════
try:
    from app.core.mic_algebra import Morphism, CategoricalState, TopologicalInvariantError
    from app.core.schemas import Stratum
except ImportError:
    class TopologicalInvariantError(Exception):
        """Violación a invariante topológico categórico."""
        pass

    class Morphism:
        """Morfismo base del 2-categoría de agentes."""
        pass

    class CategoricalState:
        pass

try:
    from app.core.immune_system.metric_tensors import G_PHYSICS
except ImportError:
    # Tensor métrico euclídeo de dimensión canónica 7 (fallback unitario).
    G_PHYSICS: NDArray[np.float64] = np.eye(7, dtype=np.float64)

logger = logging.getLogger("MIC.ImmuneSystem.WatcherAgent")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES NUMÉRICAS, ESPECTRALES Y TERMODINÁMICAS
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)
_KAPPA_0: Final[float] = 1.0                    # Escala de Lipschitz base
_DEFAULT_TOL: Final[float] = 1e-8               # Tolerancia absoluta de conservación
_COND_MAX: Final[float] = 1e12                  # Umbral de mal-condicionamiento de G
_RHO_CRIT: Final[float] = 1e6                   # Radio espectral crítico de 𝒯
_LOCKDOWN_EPS_FACTOR: Final[float] = 1e4        # Factor de umbral de paranoia
_SYMMETRY_REL_TOL: Final[float] = 1e-10         # Simetría relativa Frobenius


# ══════════════════════════════════════════════════════════════════════════════
# EXCEPCIONES TENSORIALES (jerarquía categórica de fallos de invariantes)
# ══════════════════════════════════════════════════════════════════════════════
class WatcherAgentError(TopologicalInvariantError):
    """Excepción raíz del Funtor Covariante de Propagación 𝒲_agent."""
    pass


class TensorConservationError(WatcherAgentError):
    """Divergencia covariante ∇_μ 𝒯^{μν} no nula (fuga de exergía)."""
    pass


class TensorSymmetryError(WatcherAgentError):
    """Violación de simetría 𝒯^{μν} = 𝒯^{νμ} (o de la métrica)."""
    pass


class MetricSignatureError(WatcherAgentError):
    """Métrica no definida positiva o mal condicionada (firma riemanniana rota)."""
    pass


class DeformationEnergyOverflowError(WatcherAgentError):
    """Energía de Dirichlet deformada supera la conductancia del haz celular."""
    pass


class DimensionMismatchError(WatcherAgentError):
    """Inconsistencia dimensional entre tensores, métrica y cofrontera δx."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# ENUMERACIONES AUXILIARES (álgebra de Boole de severidad espectral)
# ══════════════════════════════════════════════════════════════════════════════
class SpectralSeverity(Enum):
    """Severidad del espectro del par (𝒯, G) en el espacio de secciones."""
    NOMINAL = auto()
    ELEVATED = auto()
    CRITICAL = auto()
    SINGULAR = auto()


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS INMUTABLES (DTOs TENSORIALES / OBJETOS DEL TOPOS)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class StressTensorData:
    r"""
    Encapsula el Tensor de Estrés de Cauchy-Momentum evaluado localmente
    junto con la geometría de fondo (métrica + conexión de Levi-Civita).

    Convenciones de índices (notación abstracta de índices de Penrose):
      • T_mu_nu     : 𝒯^{μν}  — (2,0) simétrico, dim×dim
      • G_mu_nu     : G_{μν}  — (0,2) riemanniano, dim×dim
      • Christoffel : Γ^ρ_{μν} — (1,2), dim×dim×dim
      • partial_T   : ∂_ρ 𝒯^{μν} — dim×dim×dim  (ρ, μ, ν)
    """
    T_mu_nu: NDArray[np.float64]
    G_mu_nu: NDArray[np.float64]
    Christoffel: NDArray[np.float64]
    partial_T: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SpectralInvariants:
    r"""
    Invariantes espectrales del par (𝒯, G) extraídos en Fase 1.
    Forman el objeto de datos que viaja funtorialmente a las fases anidadas.
    """
    trace_mixed: float              # Tr_G(𝒯) = 𝒯^{μν} G_{μν}  (traza mixta)
    lambda_max_G: float             # λ_max(G)  (máximo autovalor de la métrica)
    lambda_min_G: float             # λ_min(G)  (mínimo autovalor — gap espectral)
    cond_G: float                   # κ(G) = λ_max / λ_min  (número de condición)
    rho_T: float                    # ρ(𝒯) = max |λ_i(𝒯)|  (radio espectral)
    op_norm_T: float                # ‖𝒯‖_{op} = σ_max(𝒯)  (norma de operador 2)
    frobenius_T: float              # ‖𝒯‖_F
    christoffel_contracted: NDArray[np.float64]  # Γ^μ_{μλ} ∈ ℝ^{dim}
    severity: SpectralSeverity


@dataclass(frozen=True, slots=True)
class ValidatedStressTensor:
    """
    Producto canónico de la Fase 1: tensor validado + invariantes espectrales.
    Constituye el dominio formal del morfismo de la Fase 2.
    """
    T_mu_nu: NDArray[np.float64]
    G_mu_nu: NDArray[np.float64]
    invariants: SpectralInvariants
    residual_relative: float        # Residuo relativo de ∇𝒯


@dataclass(frozen=True, slots=True)
class PushforwardResult:
    """
    Producto canónico de la Fase 2: energía deformada + cota de Rayleigh
    + invariantes heredados. Dominio formal del morfismo de la Fase 3.
    """
    deformed_energy: float
    rayleigh_bound: float
    energy_is_bounded: bool
    invariants: SpectralInvariants


@dataclass(frozen=True, slots=True)
class CohomologyPushforwardData:
    r"""Deformación inyectada al SheafCohomologyOrchestrator (H¹-push)."""
    deformed_dirichlet_energy: float
    stress_trace: float
    operator_norm: float
    spectral_radius: float


@dataclass(frozen=True, slots=True)
class ToposPullbackData:
    r"""Restricción inyectada al MICAgent (pullback en el topos de Grothendieck)."""
    L_max: float
    zero_trust_lockdown: bool
    severity: SpectralSeverity
    condition_number: float


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 – VALIDACIÓN TENSORIAL ESPECTRAL Y EXTRACCIÓN DE INVARIANTES
# ══════════════════════════════════════════════════════════════════════════════
class Phase1_StressTensorValidation:
    r"""
    Primera fase del pipeline tensorial (morfismo monádico de validación).

    Verifica:
      (i)   consistencia dimensional del 4-tuplo (𝒯, G, Γ, ∂𝒯);
      (ii)  simetría de 𝒯 y de G (álgebra de tensores simétricos);
      (iii) firma riemanniana de G (definida positiva vía Cholesky + espectro);
      (iv)  conservación covariante ∇_μ 𝒯^{μν} = 0 (residuo relativo);
      (v)   extracción del espectro completo (λ(G), ρ(𝒯), ‖𝒯‖_{op}, κ(G)).

    El método público terminal `validate_tensor` devuelve un
    `ValidatedStressTensor` que es el objeto de entrada formal de la
    Fase 2 anidada (`Phase2_CohomologyPushforward`).
    """

    def __init__(self, tolerance: float = _DEFAULT_TOL) -> None:
        if tolerance <= 0.0:
            raise ValueError("tolerance debe ser estrictamente positiva.")
        self._tol: float = float(tolerance)

    # ─── utilidades de álgebra lineal numérica ───────────────────────────────

    @staticmethod
    def _frobenius_norm(A: NDArray[np.float64]) -> float:
        """‖A‖_F = √(Σ_{ij} A_{ij}²) — norma de Hilbert-Schmidt."""
        return float(np.linalg.norm(A, ord="fro"))

    @staticmethod
    def _relative_symmetry_error(A: NDArray[np.float64]) -> float:
        r"""
        Error de simetría relativo: ‖A − Aᵀ‖_F / (‖A‖_F + ε_mach).
        Invariante bajo escalado global del tensor.
        """
        fro = float(np.linalg.norm(A, ord="fro"))
        skew = float(np.linalg.norm(A - A.T, ord="fro"))
        return skew / (fro + _MACHINE_EPSILON)

    @staticmethod
    def _safe_eigvalsh(A: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Autovalores de matriz hermitiana con re-simetrización previa
        (proyección de Frobenius al subespacio simétrico) para estabilidad.
        """
        A_sym = 0.5 * (A + A.T)
        return la.eigvalsh(A_sym)

    # ─── chequeos estructurales ──────────────────────────────────────────────

    def _assert_dimensions(self, data: StressTensorData) -> int:
        """
        Verifica que todos los tensores compartan la misma dimensión n
        y que los tensores de rango 3 tengan forma (n, n, n).
        Devuelve n.
        """
        T = data.T_mu_nu
        G = data.G_mu_nu
        Gamma = data.Christoffel
        dT = data.partial_T

        if T.ndim != 2 or T.shape[0] != T.shape[1]:
            raise DimensionMismatchError(
                f"𝒯 debe ser cuadrado; recibido shape={T.shape}."
            )
        n = T.shape[0]
        if G.shape != (n, n):
            raise DimensionMismatchError(
                f"G shape {G.shape} incompatible con dim(𝒯)={n}."
            )
        if Gamma.shape != (n, n, n):
            raise DimensionMismatchError(
                f"Christoffel shape {Gamma.shape} ≠ ({n},{n},{n})."
            )
        if dT.shape != (n, n, n):
            raise DimensionMismatchError(
                f"partial_T shape {dT.shape} ≠ ({n},{n},{n})."
            )
        if n == 0:
            raise DimensionMismatchError("Dimensión nula del fibrado tangente.")
        return n

    def _check_symmetry(self, T: NDArray[np.float64]) -> None:
        """Exige simetría de 𝒯 con tolerancia relativa Frobenius."""
        err = self._relative_symmetry_error(T)
        if err > max(self._tol, _SYMMETRY_REL_TOL):
            raise TensorSymmetryError(
                f"𝒯 no es simétrico: error relativo Frobenius = {err:.3e}."
            )

    def _check_metric(self, G: NDArray[np.float64]) -> Tuple[float, float, float]:
        r"""
        Verifica simetría y definidad positiva de G.
        Usa Cholesky (fallo ⇒ no PD) + espectro para λ_min, λ_max, κ(G).
        Devuelve (λ_min, λ_max, cond).
        """
        err = self._relative_symmetry_error(G)
        if err > max(self._tol, _SYMMETRY_REL_TOL):
            raise MetricSignatureError(
                f"G no es simétrica: error relativo Frobenius = {err:.3e}."
            )

        # Test de Cholesky (O(n³/3), más estable que solo autovalores para PD).
        try:
            la.cholesky(G, lower=True, overwrite_a=False, check_finite=True)
        except la.LinAlgError as exc:
            raise MetricSignatureError(
                "G no es definida positiva (fallo de Cholesky)."
            ) from exc

        eigvals = self._safe_eigvalsh(G)
        lam_min = float(eigvals[0])
        lam_max = float(eigvals[-1])

        if lam_min <= 0.0:
            # Redundante tras Cholesky, pero cierra el invariante espectral.
            raise MetricSignatureError(
                f"λ_min(G) = {lam_min:.3e} ≤ 0: firma no riemanniana."
            )

        cond = lam_max / lam_min
        if cond > _COND_MAX:
            logger.warning(
                "Métrica mal condicionada: κ(G)=%.3e > κ_max=%.1e",
                cond, _COND_MAX,
            )
        return lam_min, lam_max, cond

    # ─── divergencia covariante (conexión de Levi-Civita) ────────────────────

    def _compute_divergence(
        self, data: StressTensorData
    ) -> NDArray[np.float64]:
        r"""
        Calcula el vector (∇_μ 𝒯)^{ν} mediante contracciones de Einstein
        vectorizadas (einsum), respetando la fórmula de la conexión afín:

            (∇𝒯)^ν = ∂_μ 𝒯^{μν}
                     + Γ^μ_{μλ} 𝒯^{λν}
                     + Γ^ν_{μλ} 𝒯^{μλ}

        Índices de entrada:
          partial_T[ρ, μ, ν] = ∂_ρ 𝒯^{μν}
          Christoffel[ρ, μ, ν] = Γ^ρ_{μν}
          T[μ, ν] = 𝒯^{μν}
        """
        T = data.T_mu_nu
        Gamma = data.Christoffel
        partial_T = data.partial_T

        # ∂_μ 𝒯^{μν}  → contracción del índice de derivada con el primer índice
        # de 𝒯 (μ). Resultado: vector en ν.
        div_partial = np.einsum("mmn->n", partial_T, optimize=True)

        # Γ^μ_{μλ}  → contracción de los dos primeros índices de Γ (ρ=μ).
        # Resultado: covector en λ.
        gamma_contracted = np.einsum("mml->l", Gamma, optimize=True)

        # (Γ^μ_{μλ}) 𝒯^{λν}  → producto matriz-vector (λ→ν).
        term1 = gamma_contracted @ T

        # Γ^ν_{μλ} 𝒯^{μλ}  → contracción total de los dos índices de 𝒯
        # con los dos índices covariantes de Γ, libre en ν.
        term2 = np.einsum("nml,ml->n", Gamma, T, optimize=True)

        return div_partial + term1 + term2

    # ─── espectro completo del par (𝒯, G) ────────────────────────────────────

    def _compute_spectral_invariants(
        self,
        data: StressTensorData,
        lam_min_G: float,
        lam_max_G: float,
        cond_G: float,
    ) -> SpectralInvariants:
        r"""
        Extrae el paquete completo de invariantes espectrales:

          • Tr_G(𝒯) = 𝒯^{μν} G_{μν}          (traza mixta / contracción métrica)
          • ρ(𝒯)    = max |λ_i(𝒯)|            (radio espectral)
          • ‖𝒯‖_op  = σ_max(𝒯)                (norma de operador inducida por ℓ₂)
          • ‖𝒯‖_F                              (norma de Frobenius)
          • Γ^μ_{μλ}                           (contracción de Christoffel)
          • Severidad espectral (álgebra de Boole de umbrales)
        """
        T = data.T_mu_nu
        G = data.G_mu_nu

        # Traza mixta: 𝒯^{μν} G_{μν}. No requiere invertir G.
        # (Equivalente a Tr(𝒯 · G^♭) si 𝒯 es (2,0) y G es (0,2).)
        trace_mixed = float(np.einsum("ij,ij->", T, G, optimize=True))

        # Espectro de 𝒯 (re-simetrizado).
        eig_T = self._safe_eigvalsh(T)
        rho_T = float(np.max(np.abs(eig_T)))

        # Norma de operador 2 = mayor valor singular.
        # Para simétricas: σ_max = max |λ_i|; usamos SVD genérico por robustez.
        op_norm_T = float(np.linalg.norm(T, ord=2))

        frobenius_T = self._frobenius_norm(T)

        gamma_contracted = np.einsum("mml->l", data.Christoffel, optimize=True)

        # Predicado booleano de severidad (álgebra de Boole de umbrales).
        severity = self._classify_severity(cond_G, rho_T, lam_min_G)

        return SpectralInvariants(
            trace_mixed=trace_mixed,
            lambda_max_G=lam_max_G,
            lambda_min_G=lam_min_G,
            cond_G=cond_G,
            rho_T=rho_T,
            op_norm_T=op_norm_T,
            frobenius_T=frobenius_T,
            christoffel_contracted=gamma_contracted,
            severity=severity,
        )

    @staticmethod
    def _classify_severity(
        cond_G: float, rho_T: float, lam_min_G: float
    ) -> SpectralSeverity:
        """
        Clasificación booleana de severidad espectral:
          SINGULAR  ⇔ λ_min ≈ 0  (gap colapsado)
          CRITICAL  ⇔ κ(G) > κ_max  ∨  ρ(𝒯) > ρ_crit
          ELEVATED  ⇔ κ(G) > √κ_max  ∨  ρ(𝒯) > √ρ_crit
          NOMINAL   ⇔ en otro caso
        """
        if lam_min_G < _MACHINE_EPSILON * 10.0:
            return SpectralSeverity.SINGULAR
        if cond_G > _COND_MAX or rho_T > _RHO_CRIT:
            return SpectralSeverity.CRITICAL
        if cond_G > math.sqrt(_COND_MAX) or rho_T > math.sqrt(_RHO_CRIT):
            return SpectralSeverity.ELEVATED
        return SpectralSeverity.NOMINAL

    # ─── método terminal de la Fase 1 (contrato de entrada de la Fase 2) ─────

    def validate_tensor(self, data: StressTensorData) -> ValidatedStressTensor:
        r"""
        Ejecuta la validación completa del 4-tuplo (𝒯, G, Γ, ∂𝒯) y devuelve
        el objeto `ValidatedStressTensor` que constituye el dominio formal
        del morfismo de pushforward cohomológico de la Fase 2.

        Pipeline interno (composición de monomorfismos de validación):
          dim ✓ → simetría(𝒯) ✓ → métrica(G) ✓ → ∇𝒯 ≈ 0 ✓ → espectro(𝒯,G)

        Este es el último método público de la Fase 1; su tipo de retorno
        `ValidatedStressTensor` es exactamente el tipo de entrada del método
        principal de la clase anidada `Phase2_CohomologyPushforward`.
        """
        # (i) Dimensiones
        n = self._assert_dimensions(data)

        # (ii) Simetría de 𝒯
        self._check_symmetry(data.T_mu_nu)

        # (iii) Métrica riemanniana + espectro de G
        lam_min_G, lam_max_G, cond_G = self._check_metric(data.G_mu_nu)

        # (iv) Conservación covariante (residuo absoluto y relativo)
        div_cov = self._compute_divergence(data)
        residual_abs = float(np.linalg.norm(div_cov, ord=2))
        scale = self._frobenius_norm(data.T_mu_nu) + _MACHINE_EPSILON
        residual_rel = residual_abs / scale

        if residual_rel > self._tol:
            logger.critical(
                "Divergencia covariante no nula: ‖∇𝒯‖₂=%.3e, rel=%.3e (n=%d)",
                residual_abs, residual_rel, n,
            )
            raise TensorConservationError(
                f"Fuga de exergía: ∇_μ 𝒯^{{μν}} ≠ 0 "
                f"(residuo abs={residual_abs:.3e}, rel={residual_rel:.3e})."
            )

        # (v) Invariantes espectrales
        invariants = self._compute_spectral_invariants(
            data, lam_min_G, lam_max_G, cond_G
        )

        logger.debug(
            "Fase 1 OK | n=%d | Tr_G(𝒯)=%.4e | λ_max(G)=%.4e | κ(G)=%.2e | "
            "ρ(𝒯)=%.4e | ‖𝒯‖_op=%.4e | sev=%s | res_rel=%.3e",
            n,
            invariants.trace_mixed,
            invariants.lambda_max_G,
            invariants.cond_G,
            invariants.rho_T,
            invariants.op_norm_T,
            invariants.severity.name,
            residual_rel,
        )

        return ValidatedStressTensor(
            T_mu_nu=data.T_mu_nu,
            G_mu_nu=data.G_mu_nu,
            invariants=invariants,
            residual_relative=residual_rel,
        )

    # =========================================================================
    # FASE 2 (ANIDADA EN FASE 1) – PUSHFORWARD COHOMOLÓGICO DE DIRICHLET
    # =========================================================================
    class Phase2_CohomologyPushforward:
        r"""
        Segunda fase anidada: morfismo de pushforward en la cohomología de
        haces H¹(X; 𝒮_𝒯).

        Recibe el `ValidatedStressTensor` producido por
        `Phase1_StressTensorValidation.validate_tensor` y el vector
        cofrontera δx ∈ Γ(Tℳ), y produce la energía de Dirichlet deformada

            ℰ_deform(δx) = 𝒯_{μν} (δx)^μ (δx)^ν = ⟨δx, 𝒯♭ δx⟩

        junto con la cota de Rayleigh |ℰ| ≤ ‖𝒯‖_{op} ‖δx‖₂² y la verificación
        de acotación (conductancia del haz celular).

        El método terminal `compute_pushforward` devuelve un `PushforwardResult`
        que es el dominio formal de la Fase 3 anidada.
        """

        def __init__(
            self,
            energy_overflow_factor: float = 1e8,
        ) -> None:
            """
            energy_overflow_factor: múltiplo de ‖𝒯‖_op · ‖δx‖₂² a partir del
            cual se considera desbordamiento de la conductancia del haz.
            """
            if energy_overflow_factor <= 0.0:
                raise ValueError("energy_overflow_factor debe ser > 0.")
            self._overflow_factor = float(energy_overflow_factor)

        def _assert_delta_dimension(
            self,
            validated: ValidatedStressTensor,
            delta_x: NDArray[np.float64],
        ) -> None:
            """δx debe ser un vector de longitud n = dim(𝒯)."""
            n = validated.T_mu_nu.shape[0]
            if delta_x.ndim != 1 or delta_x.shape[0] != n:
                raise DimensionMismatchError(
                    f"δx shape {delta_x.shape} incompatible con dim={n}."
                )

        def _dirichlet_energy(
            self,
            T: NDArray[np.float64],
            delta_x: NDArray[np.float64],
        ) -> float:
            r"""
            Forma bilineal de Dirichlet deformada:
                ℰ = Σ_{μ,ν} 𝒯^{μν} (δx)_μ (δx)_ν = δxᵀ 𝒯 δx
            Evaluada con einsum de contracción total (estable y legible).
            """
            return float(np.einsum("i,ij,j->", delta_x, T, delta_x, optimize=True))

        def _rayleigh_bound(
            self,
            op_norm_T: float,
            delta_x: NDArray[np.float64],
        ) -> float:
            r"""
            Cota de Rayleigh-Ritz para la forma cuadrática:
                |δxᵀ 𝒯 δx| ≤ ‖𝒯‖_{op} ‖δx‖₂²
            donde ‖𝒯‖_{op} = σ_max(𝒯) es la norma de operador inducida por ℓ₂.
            """
            dx_sq = float(np.dot(delta_x, delta_x))
            return op_norm_T * dx_sq

        def compute_pushforward(
            self,
            validated: ValidatedStressTensor,
            delta_x: NDArray[np.float64],
        ) -> PushforwardResult:
            r"""
            Calcula ℰ_deform y la cota espectral de Rayleigh a partir del
            tensor validado de la Fase 1 y del vector cofrontera δx.

            Invariantes heredados (espectro de (𝒯, G)) se reenvían intactos
            al `PushforwardResult` para consumo exclusivo de la Fase 3.

            Este es el último método público de la Fase 2; su tipo de retorno
            `PushforwardResult` es exactamente el tipo de entrada del método
            principal de la clase anidada `Phase3_ToposPullback`.
            """
            self._assert_delta_dimension(validated, delta_x)

            inv = validated.invariants
            T = validated.T_mu_nu

            # Energía de Dirichlet deformada
            energy = self._dirichlet_energy(T, delta_x)

            # Cota de Rayleigh
            bound = self._rayleigh_bound(inv.op_norm_T, delta_x)

            # Verificación de acotación (conductancia del haz)
            # Si |ℰ| excede overflow_factor · cota, hay desbordamiento patológico
            # (posible singularidad del haz o δx fuera de la sección admisible).
            abs_energy = abs(energy)
            energy_is_bounded = abs_energy <= (self._overflow_factor * bound + _MACHINE_EPSILON)

            if not energy_is_bounded:
                logger.error(
                    "Desbordamiento de energía deformada: |ℰ|=%.3e > factor·bound=%.3e",
                    abs_energy, self._overflow_factor * bound,
                )
                raise DeformationEnergyOverflowError(
                    f"|ℰ_deform|={abs_energy:.3e} supera la conductancia del haz "
                    f"(bound={bound:.3e}, factor={self._overflow_factor:.1e})."
                )

            # Consistencia numérica débil: |ℰ| no debería superar bound + holgura.
            # Si lo hace levemente, lo registramos (redondeo / no-simetría residual).
            if abs_energy > bound * (1.0 + 1e-6) + _MACHINE_EPSILON:
                logger.warning(
                    "Rayleigh violado levemente: |ℰ|=%.6e > bound=%.6e "
                    "(posible residual de simetría o redondeo).",
                    abs_energy, bound,
                )

            logger.info(
                "Fase 2 OK | ℰ_deform=%.6e | Rayleigh_bound=%.6e | bounded=%s",
                energy, bound, energy_is_bounded,
            )

            return PushforwardResult(
                deformed_energy=energy,
                rayleigh_bound=bound,
                energy_is_bounded=energy_is_bounded,
                invariants=inv,
            )

        # =====================================================================
        # FASE 3 (ANIDADA EN FASE 2) – PULLBACK AL TOPOS DE GROTHENDIECK
        # =====================================================================
        class Phase3_ToposPullback:
            r"""
            Tercera fase anidada: morfismo de pullback en el topos de
            Grothendieck ℰ = Sh(𝒞, J) del MICAgent.

            Recibe el `PushforwardResult` de la Fase 2 y determina:

              L_max(𝒯) = κ₀ / ( √|Tr_G(𝒯)| + ε_mach · λ_max(G) · √κ(G) )

            El denominador combina:
              • la escala de traza mixta √|Tr_G(𝒯)|  (intensidad del estrés);
              • la regularización espectral ε·λ_max·√κ  (estabilidad numérica
                y penalización del mal-condicionamiento de la métrica).

            El predicado de lockdown (álgebra de Boole de predicados estables):

              ZeroTrust ⇔ (L_max < θ_crit)
                         ∨ (severity ∈ {CRITICAL, SINGULAR})
                         ∨ (κ(G) > κ_max)
                         ∨ (ρ(𝒯) > ρ_crit)

            cierra la cadena anidada Fase1 → Fase2 → Fase3.
            """

            def __init__(
                self,
                kappa0: float = _KAPPA_0,
                epsilon: float = _MACHINE_EPSILON,
                lockdown_eps_factor: float = _LOCKDOWN_EPS_FACTOR,
            ) -> None:
                if kappa0 <= 0.0:
                    raise ValueError("kappa0 debe ser estrictamente positivo.")
                if epsilon <= 0.0:
                    raise ValueError("epsilon debe ser estrictamente positivo.")
                self._kappa0 = float(kappa0)
                self._epsilon = float(epsilon)
                self._lockdown_eps_factor = float(lockdown_eps_factor)

            def _compute_L_max(self, inv: SpectralInvariants) -> float:
                r"""
                Estima la cota de Lipschitz homomórfica:

                    L_max = κ₀ / ( √|Tr_G(𝒯)| + ε · λ_max(G) · √κ(G) )

                El término √κ(G) penaliza métricas mal condicionadas,
                elevando el denominador y reduciendo L_max (más restrictivo),
                lo cual es coherente con la pérdida de control geométrico.
                """
                safe_trace = abs(inv.trace_mixed)
                sqrt_trace = math.sqrt(safe_trace) if safe_trace > 0.0 else 0.0
                sqrt_cond = math.sqrt(max(inv.cond_G, 1.0))
                denominator = (
                    sqrt_trace
                    + self._epsilon * inv.lambda_max_G * sqrt_cond
                )
                # Evitar división por cero patológica (aunque ε·λ_max·√κ > 0).
                denominator = max(denominator, self._epsilon)
                return self._kappa0 / denominator

            def _boolean_lockdown_predicate(
                self,
                L_max: float,
                inv: SpectralInvariants,
            ) -> bool:
                r"""
                Predicado de Zero-Trust en el álgebra de Boole de umbrales:

                    P₁ = (L_max < ε · lockdown_eps_factor)
                    P₂ = (severity ∈ {CRITICAL, SINGULAR})
                    P₃ = (κ(G) > κ_max)
                    P₄ = (ρ(𝒯) > ρ_crit)

                    ZeroTrust = P₁ ∨ P₂ ∨ P₃ ∨ P₄
                """
                theta_crit = self._epsilon * self._lockdown_eps_factor
                p1 = L_max < theta_crit
                p2 = inv.severity in (
                    SpectralSeverity.CRITICAL,
                    SpectralSeverity.SINGULAR,
                )
                p3 = inv.cond_G > _COND_MAX
                p4 = inv.rho_T > _RHO_CRIT
                return bool(p1 or p2 or p3 or p4)

            def compute_pullback(
                self, pushforward: PushforwardResult
            ) -> ToposPullbackData:
                r"""
                Evalúa L_max y el predicado de paranoia estructural a partir
                del `PushforwardResult` de la Fase 2, cerrando la cadena
                anidada de morfismos:

                    validate_tensor  →  compute_pushforward  →  compute_pullback

                Devuelve el DTO `ToposPullbackData` inyectable en el MICAgent.
                """
                inv = pushforward.invariants

                L_max = self._compute_L_max(inv)
                lockdown = self._boolean_lockdown_predicate(L_max, inv)

                if lockdown:
                    logger.warning(
                        "Estrés crítico → Paranoia Estructural | "
                        "L_max=%.3e | sev=%s | κ(G)=%.2e | ρ(𝒯)=%.3e",
                        L_max, inv.severity.name, inv.cond_G, inv.rho_T,
                    )
                else:
                    logger.info(
                        "Fase 3 OK | L_max=%.6e | lockdown=%s | sev=%s",
                        L_max, lockdown, inv.severity.name,
                    )

                return ToposPullbackData(
                    L_max=float(L_max),
                    zero_trust_lockdown=lockdown,
                    severity=inv.severity,
                    condition_number=inv.cond_G,
                )


# ══════════════════════════════════════════════════════════════════════════════
# FUNTOR COVARIANTE DE PROPAGACIÓN (WATCHER AGENT)
# ══════════════════════════════════════════════════════════════════════════════
class TopologicalWatcherAgent(Morphism):
    r"""
    Funtor Covariante de Propagación Tensorial (𝒲_agent).

    Asegura el difeomorfismo (a nivel de 2-morfismos) entre el escudo
    topológico (membrana p-Laplaciana) y el estrato generativo (MICAgent).

    Orquesta las tres fases anidadas como composición de morfismos:

        𝒲 = pullback ∘ pushforward ∘ validate

      1. Phase1_StressTensorValidation.validate_tensor
            → ValidatedStressTensor
      2. Phase2_CohomologyPushforward.compute_pushforward
            → PushforwardResult
      3. Phase3_ToposPullback.compute_pullback
            → ToposPullbackData

    La anidación de clases refleja la dependencia funtorial estricta:
    cada fase solo es accesible a través de la envoltura de la anterior,
    garantizando que no exista un camino que omita la validación espectral
    ni la cota de Rayleigh antes del pullback al topos.
    """

    def __init__(
        self,
        tolerance: float = _DEFAULT_TOL,
        kappa0: float = _KAPPA_0,
        energy_overflow_factor: float = 1e8,
    ) -> None:
        self._tol = float(tolerance)
        self._kappa0 = float(kappa0)
        self._energy_overflow_factor = float(energy_overflow_factor)

        # Instanciación única de la torre anidada (reutilizable, sin estado mutable
        # entre invocaciones más allá de los parámetros de construcción).
        self._phase1 = Phase1_StressTensorValidation(tolerance=self._tol)
        self._phase2 = (
            Phase1_StressTensorValidation.Phase2_CohomologyPushforward(
                energy_overflow_factor=self._energy_overflow_factor,
            )
        )
        self._phase3 = (
            Phase1_StressTensorValidation
            .Phase2_CohomologyPushforward
            .Phase3_ToposPullback(kappa0=self._kappa0)
        )

    def execute_propagation(
        self,
        tensor_data: StressTensorData,
        delta_x: NDArray[np.float64],
    ) -> Tuple[CohomologyPushforwardData, ToposPullbackData]:
        r"""
        Ejecución canónica del morfismo categórico 𝒲.

        Aplica la composición:

            tensor_data  ──validate──▶  ValidatedStressTensor
                         ──push──────▶  PushforwardResult
                         ──pull──────▶  ToposPullbackData

        y empaqueta la respuesta en los DTOs de interfaz
        (CohomologyPushforwardData, ToposPullbackData) consumibles por
        el SheafCohomologyOrchestrator y el MICAgent respectivamente.
        """
        # 1. Auditoría inercial espectral y extracción de invariantes
        validated = self._phase1.validate_tensor(tensor_data)

        # 2. Pushforward cohomológico (energía de Dirichlet deformada)
        pushforward_result = self._phase2.compute_pushforward(validated, delta_x)

        # 3. Pullback al topos de Grothendieck (L_max + predicado de paranoia)
        pullback_data = self._phase3.compute_pullback(pushforward_result)

        # Empaquetado de interfaz hacia el orquestador de cohomología de haces
        inv = pushforward_result.invariants
        cohomology_data = CohomologyPushforwardData(
            deformed_dirichlet_energy=pushforward_result.deformed_energy,
            stress_trace=inv.trace_mixed,
            operator_norm=inv.op_norm_T,
            spectral_radius=inv.rho_T,
        )

        return cohomology_data, pullback_data

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Compatibilidad con la interfaz Morphism del 2-categoría de agentes."""
        return self.execute_propagation(*args, **kwargs)

    def describe_pipeline(self) -> Dict[str, str]:
        """
        Metadatos del pipeline anidado (introspección categórica ligera).
        Útil para telemetría y auditorías del sistema inmune.
        """
        return {
            "functor": "𝒲_agent (TopologicalWatcherAgent)",
            "version": "5.0.0-Nested-Spectral-Topos-Cauchy",
            "phase_1": "Phase1_StressTensorValidation.validate_tensor",
            "phase_2": "Phase2_CohomologyPushforward.compute_pushforward",
            "phase_3": "Phase3_ToposPullback.compute_pullback",
            "axiom_1": "∇_μ 𝒯^{μν} = 0  (conservación covariante, residuo relativo)",
            "axiom_2": "ℰ = δxᵀ 𝒯 δx ≤ ‖𝒯‖_op ‖δx‖₂²  (Rayleigh)",
            "axiom_3": "L_max = κ₀ / (√|Tr_G 𝒯| + ε λ_max √κ)  (Lipschitz topos)",
        }<|eos|>