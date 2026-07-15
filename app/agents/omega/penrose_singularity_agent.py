# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Penrose Singularity Agent (Orquestador Supremo de Colapso Geodésico)║
║ Ruta   : app/omega/penrose_singularity_agent.py                              ║
║ Versión: 4.0.0-Nested-Hawking-Penrose-Spectral-Topos                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y TOPOLOGÍA DIFERENCIAL
────────────────────────────────────────────────────────────────────────────────
Endofuntor Supremo 𝒫: ℰ_MIC → ℰ_MIC que gobierna el fibrado focal de
Raychaudhuri. Audita incondicionalmente el Teorema de Singularidad de Penrose
(1965) / Hawking–Penrose (1970), garantizando que el colapso de la función de
onda de la intención generativa (LLM) sea geométricamente inevitable bajo el
tensor de estrés del ecosistema logístico.

ARQUITECTURA ANIDADA (3 fases — composición funtorial estricta)
────────────────────────────────────────────────────────────────────────────────
  Fase 1 → Auditoría espectral-energética:
           métrica, θ₀ < 0, SEC/Ricci, normalización de u, cota τ_HP.
  Fase 2 (anidada en Fase 1) → Integración cáustica vía RaychaudhuriFocalFibrator:
           f_opt, τ_c, flags de viabilidad del fibrador.
  Fase 3 (anidada en Fase 2) → Veredicto de Penrose:
           τ_c ≤ τ_HP, holgura relativa, certificado de colapso inevitable.

FUNDAMENTACIÓN MATEMÁTICA Y AXIOMAS (nivel PhD — GR + teoría espectral)
────────────────────────────────────────────────────────────────────────────────
§1. EXPANSIÓN INICIAL CONVERGENTE:
    θ₀ = ∇_μ u^μ = Tr(B) < 0.

§2. CONDICIÓN DE ENERGÍA FUERTE (SEC) + Einstein (8πG=1, Λ=0):
    R_{μν} u^μ u^ν = (𝒯_{μν} − ½ 𝒯 G_{μν}) u^μ u^ν ≥ 0,
    con u normalizado: |G(u,u) − s| < τ_u.

§3. TEOREMA DE ENFOQUE (Hawking–Penrose):
    Si θ₀ < 0, ω ≡ 0 y R_{μν} u^μ u^ν ≥ 0, entonces existe cáustica con
        τ_c ≤ τ_HP := (n−1) / |θ₀|.
    Violación ⇒ «Fuga Topológica» (veto ontológico del endofuntor 𝒫).

§4. CERTIFICADO DE COLAPSO:
    Singularidad inevitable ⇔ (θ₀<0) ∧ (SEC) ∧ (τ_c ≤ τ_HP·(1+ε)) ∧ (flags fibrador).
"""

from __future__ import annotations

import enum
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Final, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, CategoricalState, TopologicalInvariantError
except ImportError:
    class TopologicalInvariantError(Exception):
        """Violación a invariante topológico categórico en el Topos ℰ_MIC."""
        pass

    class Morphism:
        """Stub para Morphism del 2-categoría de agentes."""
        pass

    class CategoricalState:
        """Stub mínimo de estado categórico."""

        def get_tensor(self, key: str) -> Any:
            raise KeyError(key)

        def update(self, key: str, value: Any) -> "CategoricalState":
            return self

try:
    from app.omega.raychaudhuri_focal_fibrator import (
        RaychaudhuriFocalFibrator,
        FocalLengthResult,
        FocalViabilityFlags,
        CausticMethod,
        MetricSignature,
        RaychaudhuriFibratorError,
    )
except ImportError:
    # Stubs para ejecución analítica aislada (sin el fibrador instalado).
    class RaychaudhuriFibratorError(Exception):
        pass

    class FocalViabilityFlags(enum.Flag):
        NONE = 0
        ALL = 0

        def is_order_unit(self) -> bool:
            return False

    class CausticMethod(enum.Enum):
        ANALYTICAL_BOUND = enum.auto()
        ANALYTICAL_EXACT = enum.auto()
        NUMERICAL_IVP = enum.auto()

    class MetricSignature(enum.Enum):
        RIEMANNIAN = enum.auto()
        LORENTZIAN = enum.auto()

    @dataclass(frozen=True, slots=True)
    class FocalLengthResult:
        optimal_focal_length: float = 0.0
        caustic_affine_parameter: float = 0.0
        hawking_penrose_bound: float = 0.0
        initial_dtheta_dtau: float = 0.0
        caustic_method: Any = None
        viability_flags: Any = None
        area_collapse_factor: float = 0.0

    class RaychaudhuriFocalFibrator:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "RaychaudhuriFocalFibrator no disponible; instale "
                "app.omega.raychaudhuri_focal_fibrator."
            )

        def __call__(self, *args: Any, **kwargs: Any) -> FocalLengthResult:
            raise ImportError("RaychaudhuriFocalFibrator no disponible.")

try:
    from app.core.immune_system.metric_tensors import G_PHYSICS
except ImportError:
    G_PHYSICS: NDArray[np.float64] = np.eye(4, dtype=np.float64)

logger = logging.getLogger("MIC.Omega.PenroseSingularityAgent")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES NUMÉRICAS, ESPECTRALES Y DE TOLERANCIA
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)
_SYMMETRY_REL_TOL: Final[float] = 1e-10
_U_NORMALIZATION_TOL: Final[float] = 1e-8
_PENROSE_MARGIN_TOL: Final[float] = 1e-9       # holgura relativa τ_c vs τ_HP
_COND_MAX: Final[float] = 1e12
_MIN_SPATIAL_DIM: Final[int] = 2
_DEFAULT_SPATIAL_DIM: Final[int] = 4


# ══════════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES (vetos topológicos / fallos de invariantes)
# ══════════════════════════════════════════════════════════════════════════════
class PenroseAgentError(TopologicalInvariantError):
    """Excepción raíz del Endofuntor de Singularidad de Penrose 𝒫."""
    pass


class PenroseSingularityVetoError(PenroseAgentError):
    r"""τ_c > τ_HP: fuga topológica — el haz diverge más allá de Hawking–Penrose."""
    pass


class EnergyConvergenceViolationError(PenroseAgentError):
    r"""θ₀ ≥ 0 o R_{μν} u^μ u^ν < 0: falla la hipótesis de enfoque."""
    pass


class MetricTensorDegeneracyError(PenroseAgentError):
    """Métrica no simétrica, no PD, o mal condicionada."""
    pass


class DimensionMismatchError(PenroseAgentError):
    """Inconsistencia dimensional entre B, 𝒯, u y G."""
    pass


class NormalizationError(PenroseAgentError):
    """Vector u no normalizado respecto a G."""
    pass


class TensorSymmetryError(PenroseAgentError):
    """Violación de simetría de 𝒯 (Cauchy-Momentum)."""
    pass


class MissingTensorError(PenroseAgentError):
    """Tensor ausente en el fibrado del CategoricalState."""
    pass


class FibratorIntegrationError(PenroseAgentError):
    """Fallo al invocar el RaychaudhuriFocalFibrator aguas abajo."""
    pass


class ViabilityLatticeError(PenroseAgentError):
    """El retículo booleano de viabilidad no es unidad de orden."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# RETÍCULO BOOLEANO DE CERTIFICACIÓN DE SINGULARIDAD
# ══════════════════════════════════════════════════════════════════════════════
class SingularityCertificateFlags(enum.Flag):
    r"""Predicados estables del Teorema de Penrose en el retículo booleano."""
    NONE = 0
    NEGATIVE_EXPANSION = enum.auto()       # θ₀ < 0
    STRONG_ENERGY_SATISFIED = enum.auto()  # R(u,u) ≥ 0
    U_NORMALIZED = enum.auto()             # |G(u,u)−s| < tol
    CAUSTIC_WITHIN_HP_BOUND = enum.auto()  # τ_c ≤ τ_HP · (1+ε)
    FIBRATOR_VIABLE = enum.auto()          # flags del fibrador OK
    METRIC_RIEMANNIAN = enum.auto()        # G SPD

    ALL = (
        NEGATIVE_EXPANSION
        | STRONG_ENERGY_SATISFIED
        | U_NORMALIZED
        | CAUSTIC_WITHIN_HP_BOUND
        | FIBRATOR_VIABLE
        | METRIC_RIEMANNIAN
    )

    def is_order_unit(self) -> bool:
        """True ⇔ todos los predicados de singularidad inevitable están activos."""
        return self == SingularityCertificateFlags.ALL


class MetricSignatureKind(enum.Enum):
    """Firma de la métrica de fondo (convención de normalización de u)."""
    RIEMANNIAN = enum.auto()   # G(u,u) = +1
    LORENTZIAN = enum.auto()   # G(u,u) = −1


# ══════════════════════════════════════════════════════════════════════════════
# DTOs INMUTABLES (objetos del topos — contratos entre fases anidadas)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class EnergyAuditResult:
    r"""
    Producto canónico de la Fase 1: invariantes energéticos + cota HP.

    Dominio formal del morfismo de la Fase 2.
    """
    theta_0: float                       # θ₀ = Tr(B) < 0
    ricci_contraction: float             # R_{μν} u^μ u^ν ≥ 0
    sec_value: float                     # (𝒯 − ½𝒯 G)(u,u)
    stress_trace: float                  # 𝒯 = 𝒯_{μν} G^{μν}
    u_normalization: float               # G(u,u)
    tau_hp: float                        # τ_HP = (n−1)/|θ₀|
    spatial_dim: int                     # n
    metric_cond: float                   # κ(G)
    lambda_min_G: float
    lambda_max_G: float
    op_norm_B: float                     # ‖B‖_op
    frobenius_B: float                   # ‖B‖_F


@dataclass(frozen=True, slots=True)
class CausticResult:
    r"""
    Producto canónico de la Fase 2: cáustica del fibrador + auditoría heredada.

    Dominio formal del morfismo de la Fase 3.
    """
    focal_distance: float
    tau_caustic: float
    hawking_penrose_bound_fibrator: float
    initial_dtheta_dtau: float
    area_collapse_factor: float
    fibrator_method: str
    fibrator_flags_ok: bool
    energy_audit: EnergyAuditResult      # reenvío funtorial a Fase 3
    fibrator_diagnostics: FocalLengthResult


@dataclass(frozen=True, slots=True)
class SingularityDiagnostics:
    r"""Certificado inmutable de acatamiento del Teorema de Penrose–Hawking."""
    initial_expansion_scalar: float
    ricci_contraction: float
    max_theoretical_affine_limit: float
    actual_caustic_parameter: float
    relative_margin: float               # (τ_HP − τ_c) / τ_HP  (≥ 0 si OK)
    is_singularity_inevitable: bool
    certificate_flags: SingularityCertificateFlags
    spatial_dim: int
    metric_condition_number: float


@dataclass(frozen=True, slots=True)
class CollapsedAffineState:
    r"""Estado final tras forzar el colapso focal y la auditoría de singularidad."""
    focal_distance: float
    diagnostics: SingularityDiagnostics
    caustic: CausticResult


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 – AUDITORÍA ESPECTRAL-ENERGÉTICA Y COTA DE HAWKING–PENROSE
# ══════════════════════════════════════════════════════════════════════════════
class Phase1_HawkingEnergyAuditor:
    r"""
    Primera fase del pipeline de singularidad (morfismo monádico de auditoría).

    Verifica y extrae:
      (i)   firma riemanniana de G (Cholesky + espectro + κ);
      (ii)  consistencia dimensional de (B, 𝒯, u, G);
      (iii) simetría de 𝒯;
      (iv)  normalización de u;
      (v)   θ₀ = Tr(B) < 0;
      (vi)  SEC ⇒ R_{μν} u^μ u^ν ≥ 0;
      (vii) cota de Hawking–Penrose τ_HP = (n−1)/|θ₀|;
      (viii) espectro de B (‖B‖_op, ‖B‖_F).

    El método público terminal `audit_energy` devuelve un `EnergyAuditResult`
    que es el objeto de entrada formal de la Fase 2 anidada
    (`Phase2_RaychaudhuriCausticIntegrator`).
    """

    def __init__(
        self,
        metric: Optional[NDArray[np.float64]] = None,
        signature: MetricSignatureKind = MetricSignatureKind.RIEMANNIAN,
        u_norm_tol: float = _U_NORMALIZATION_TOL,
    ) -> None:
        if u_norm_tol <= 0.0:
            raise ValueError("u_norm_tol debe ser estrictamente positiva.")

        self._G: NDArray[np.float64] = (
            np.asarray(metric, dtype=np.float64) if metric is not None
            else np.asarray(G_PHYSICS, dtype=np.float64)
        )
        self._signature = signature
        self._u_norm_tol = float(u_norm_tol)
        self._target_norm = (
            1.0 if signature == MetricSignatureKind.RIEMANNIAN else -1.0
        )

        (
            self._lam_min_G,
            self._lam_max_G,
            self._cond_G,
            self._G_inv,
        ) = self._validate_and_factor_metric(self._G)

    # ─── utilidades de álgebra lineal numérica ───────────────────────────────

    @staticmethod
    def _frobenius_norm(A: NDArray[np.float64]) -> float:
        return float(np.linalg.norm(A, ord="fro"))

    @staticmethod
    def _relative_symmetry_error(A: NDArray[np.float64]) -> float:
        fro = float(np.linalg.norm(A, ord="fro"))
        skew = float(np.linalg.norm(A - A.T, ord="fro"))
        return skew / (fro + _MACHINE_EPSILON)

    @staticmethod
    def _safe_eigvalsh(A: NDArray[np.float64]) -> NDArray[np.float64]:
        A_sym = 0.5 * (A + A.T)
        return la.eigvalsh(A_sym)

    # ─── métrica ─────────────────────────────────────────────────────────────

    def _validate_and_factor_metric(
        self, G: NDArray[np.float64]
    ) -> Tuple[float, float, float, NDArray[np.float64]]:
        """Simetría + PD (Cholesky) → (λ_min, λ_max, κ, G⁻¹)."""
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise MetricTensorDegeneracyError(
                f"G debe ser cuadrada; recibido shape={G.shape}."
            )
        n = G.shape[0]
        if n < _MIN_SPATIAL_DIM:
            raise MetricTensorDegeneracyError(
                f"dim(G)={n} < n_min={_MIN_SPATIAL_DIM}."
            )

        err = self._relative_symmetry_error(G)
        if err > _SYMMETRY_REL_TOL:
            raise MetricTensorDegeneracyError(
                f"G no es simétrica: error relativo Frobenius = {err:.3e}."
            )

        try:
            la.cholesky(G, lower=True, overwrite_a=False, check_finite=True)
        except la.LinAlgError as exc:
            raise MetricTensorDegeneracyError(
                "G no es definida positiva (fallo de Cholesky)."
            ) from exc

        eigvals = self._safe_eigvalsh(G)
        lam_min = float(eigvals[0])
        lam_max = float(eigvals[-1])
        if lam_min <= 0.0:
            raise MetricTensorDegeneracyError(
                f"λ_min(G)={lam_min:.3e} ≤ 0: firma no riemanniana."
            )

        cond = lam_max / lam_min
        if cond > _COND_MAX:
            logger.warning(
                "Métrica mal condicionada: κ(G)=%.3e > κ_max=%.1e",
                cond, _COND_MAX,
            )

        try:
            G_inv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            G_inv = np.linalg.pinv(G)
            logger.warning("G singular numéricamente; se usa pseudoinversa.")

        return lam_min, lam_max, cond, G_inv

    # ─── dimensiones ─────────────────────────────────────────────────────────

    def _assert_dimensions(
        self,
        velocity_gradient: NDArray[np.float64],
        stress_tensor: NDArray[np.float64],
        u_vector: NDArray[np.float64],
        spatial_dim: int,
    ) -> int:
        """Verifica consistencia dimensional. Devuelve n efectivo."""
        B = velocity_gradient
        if B.ndim != 2 or B.shape[0] != B.shape[1]:
            raise DimensionMismatchError(
                f"B=∇u debe ser cuadrado; shape={B.shape}."
            )
        n = B.shape[0]
        if spatial_dim != n:
            raise DimensionMismatchError(
                f"spatial_dim={spatial_dim} ≠ dim(B)={n}."
            )
        if self._G.shape != (n, n):
            raise DimensionMismatchError(
                f"G shape {self._G.shape} incompatible con dim(B)={n}."
            )
        if stress_tensor.shape != (n, n):
            raise DimensionMismatchError(
                f"𝒯 shape {stress_tensor.shape} ≠ ({n},{n})."
            )
        if u_vector.ndim != 1 or u_vector.shape[0] != n:
            raise DimensionMismatchError(
                f"u shape {u_vector.shape} incompatible con dim={n}."
            )
        if n < _MIN_SPATIAL_DIM:
            raise DimensionMismatchError(
                f"n={n} < n_min={_MIN_SPATIAL_DIM}."
            )
        return n

    # ─── chequeos físicos ────────────────────────────────────────────────────

    def _check_stress_symmetry(self, stress: NDArray[np.float64]) -> None:
        err = self._relative_symmetry_error(stress)
        if err > _SYMMETRY_REL_TOL:
            raise TensorSymmetryError(
                f"𝒯 no es simétrico: error relativo Frobenius = {err:.3e}."
            )

    def _normalize_check_u(self, u: NDArray[np.float64]) -> float:
        r"""Verifica |G(u,u) − s| < tol. No re-normaliza silenciosamente."""
        g_uu = float(u @ self._G @ u)
        if abs(g_uu - self._target_norm) > self._u_norm_tol:
            raise NormalizationError(
                f"u no normalizado: G(u,u)={g_uu:.6e}, "
                f"esperado s={self._target_norm} "
                f"(tol={self._u_norm_tol:.1e}, firma={self._signature.name})."
            )
        return g_uu

    def _compute_initial_expansion(
        self, velocity_gradient: NDArray[np.float64]
    ) -> float:
        r"""θ₀ = Tr(B). Exige θ₀ < 0."""
        theta_0 = float(np.trace(velocity_gradient))
        if theta_0 >= -_MACHINE_EPSILON:
            raise EnergyConvergenceViolationError(
                f"Congruencia no convergente: θ₀ = {theta_0:.4e} ≥ 0."
            )
        return theta_0

    def _compute_ricci_contraction(
        self,
        stress_tensor: NDArray[np.float64],
        u_vector: NDArray[np.float64],
    ) -> Tuple[float, float, float]:
        r"""
        Evalúa:
          𝒯 = 𝒯_{μν} G^{μν}
          SEC = 𝒯_{μν} u^μ u^ν − ½ 𝒯 · G(u,u)
          R(u,u) = max(0, SEC)   bajo Einstein 8πG=1, Λ=0.

        Devuelve (ricci_contraction, sec_value, stress_trace).
        """
        stress_trace = float(
            np.einsum("ij,ij->", stress_tensor, self._G_inv, optimize=True)
        )
        T_uu = float(u_vector @ stress_tensor @ u_vector)
        g_uu = float(u_vector @ self._G @ u_vector)
        sec_value = T_uu - 0.5 * stress_trace * g_uu

        if sec_value < -_MACHINE_EPSILON:
            raise EnergyConvergenceViolationError(
                f"Violación SEC: R_μν u^μ u^ν ≡ SEC = {sec_value:.4e} < 0."
            )

        ricci_contraction = max(0.0, sec_value)
        return ricci_contraction, sec_value, stress_trace

    @staticmethod
    def hawking_penrose_bound(theta_0: float, dimensions: int) -> float:
        r"""
        τ_HP = (n−1) / |θ₀|  para θ₀ < 0.
        Cota superior de Cauchy del teorema de enfoque.
        """
        if theta_0 >= 0.0:
            raise EnergyConvergenceViolationError(
                f"θ₀={theta_0:.4e} ≥ 0: cota de Hawking–Penrose no aplica."
            )
        if dimensions < _MIN_SPATIAL_DIM:
            raise DimensionMismatchError(
                f"dimensions={dimensions} < n_min={_MIN_SPATIAL_DIM}."
            )
        return float(dimensions - 1) / abs(theta_0)

    # ─── método terminal de la Fase 1 (contrato de entrada de la Fase 2) ─────

    def audit_energy(
        self,
        velocity_gradient: NDArray[np.float64],
        stress_tensor: NDArray[np.float64],
        u_vector: NDArray[np.float64],
        spatial_dim: int,
    ) -> EnergyAuditResult:
        r"""
        Ejecuta la auditoría energética completa y devuelve el
        `EnergyAuditResult` que constituye el dominio formal del morfismo
        de integración cáustica de la Fase 2.

        Pipeline interno:
          dim ✓ → simetría(𝒯) ✓ → u normalizado ✓ → θ₀ < 0 ✓ →
          SEC/Ricci ✓ → τ_HP → espectro(B)

        Este es el último método público de la Fase 1; su tipo de retorno
        `EnergyAuditResult` es exactamente el tipo de entrada del método
        principal de la clase anidada `Phase2_RaychaudhuriCausticIntegrator`
        (junto con los tensores cinemáticos del fibrador).
        """
        B = np.asarray(velocity_gradient, dtype=np.float64)
        stress = np.asarray(stress_tensor, dtype=np.float64)
        u = np.asarray(u_vector, dtype=np.float64)

        # (i)–(ii) Dimensiones
        n = self._assert_dimensions(B, stress, u, spatial_dim)

        # (iii) Simetría de 𝒯
        self._check_stress_symmetry(stress)

        # (iv) Normalización de u
        g_uu = self._normalize_check_u(u)

        # (v) Expansión inicial
        theta_0 = self._compute_initial_expansion(B)

        # (vi) SEC / Ricci
        ricci, sec_value, stress_trace = self._compute_ricci_contraction(
            stress, u
        )

        # (vii) Cota de Hawking–Penrose
        tau_hp = self.hawking_penrose_bound(theta_0, n)

        # (viii) Espectro de B
        op_norm_B = float(np.linalg.norm(B, ord=2))
        frobenius_B = self._frobenius_norm(B)

        logger.debug(
            "Fase 1 OK | n=%d | θ₀=%.4e | R(u,u)=%.4e | SEC=%.4e | "
            "τ_HP=%.4e | G(u,u)=%.6f | κ(G)=%.2e | ‖B‖_op=%.4e",
            n, theta_0, ricci, sec_value, tau_hp, g_uu,
            self._cond_G, op_norm_B,
        )

        return EnergyAuditResult(
            theta_0=theta_0,
            ricci_contraction=ricci,
            sec_value=sec_value,
            stress_trace=stress_trace,
            u_normalization=g_uu,
            tau_hp=tau_hp,
            spatial_dim=n,
            metric_cond=self._cond_G,
            lambda_min_G=self._lam_min_G,
            lambda_max_G=self._lam_max_G,
            op_norm_B=op_norm_B,
            frobenius_B=frobenius_B,
        )

    # =========================================================================
    # FASE 2 (ANIDADA EN FASE 1) – INTEGRACIÓN CÁUSTICA VÍA RAYCHAUDHURI
    # =========================================================================
    class Phase2_RaychaudhuriCausticIntegrator:
        r"""
        Segunda fase anidada: morfismo de pushforward hacia el fibrador de
        Raychaudhuri.

        Recibe el `EnergyAuditResult` de la Fase 1 y el 5-tuplo cinemático
        (B, σ, ω, 𝒯, u), invoca `RaychaudhuriFocalFibrator` y produce un
        `CausticResult` con τ_c, f_opt y diagnósticos del fibrador.

        El método terminal `compute_caustic` devuelve el objeto que es el
        dominio formal de la Fase 3 anidada.
        """

        def __init__(
            self,
            spatial_dim: int,
            metric: Optional[NDArray[np.float64]] = None,
            caustic_method: CausticMethod = CausticMethod.ANALYTICAL_EXACT,
            use_numerical: bool = False,
            signature: MetricSignature = MetricSignature.RIEMANNIAN,
        ) -> None:
            if spatial_dim < _MIN_SPATIAL_DIM:
                raise ValueError(
                    f"spatial_dim={spatial_dim} < n_min={_MIN_SPATIAL_DIM}."
                )
            self._dim = int(spatial_dim)

            if use_numerical:
                caustic_method = CausticMethod.NUMERICAL_IVP

            try:
                self._fibrator = RaychaudhuriFocalFibrator(
                    spatial_dimensions=self._dim,
                    metric=metric,
                    caustic_method=caustic_method,
                    signature=signature,
                    use_numerical_integration=use_numerical,
                )
            except Exception as exc:
                raise FibratorIntegrationError(
                    f"No se pudo instanciar RaychaudhuriFocalFibrator: {exc}"
                ) from exc

            self._caustic_method = caustic_method

        def _assert_kinematic_dimensions(
            self,
            energy: EnergyAuditResult,
            velocity_gradient: NDArray[np.float64],
            shear_tensor: NDArray[np.float64],
            vorticity_tensor: NDArray[np.float64],
            stress_tensor: NDArray[np.float64],
            u_vector: NDArray[np.float64],
        ) -> None:
            """Consistencia dimensional del 5-tuplo respecto al audit."""
            n = energy.spatial_dim
            if velocity_gradient.shape != (n, n):
                raise DimensionMismatchError(
                    f"B shape {velocity_gradient.shape} ≠ ({n},{n})."
                )
            if shear_tensor.shape != (n, n):
                raise DimensionMismatchError(
                    f"σ shape {shear_tensor.shape} ≠ ({n},{n})."
                )
            if vorticity_tensor.shape != (n, n):
                raise DimensionMismatchError(
                    f"ω shape {vorticity_tensor.shape} ≠ ({n},{n})."
                )
            if stress_tensor.shape != (n, n):
                raise DimensionMismatchError(
                    f"𝒯 shape {stress_tensor.shape} ≠ ({n},{n})."
                )
            if u_vector.ndim != 1 or u_vector.shape[0] != n:
                raise DimensionMismatchError(
                    f"u shape {u_vector.shape} incompatible con dim={n}."
                )

        def _invoke_fibrator(
            self,
            velocity_gradient: NDArray[np.float64],
            shear_tensor: NDArray[np.float64],
            vorticity_tensor: NDArray[np.float64],
            stress_tensor: NDArray[np.float64],
            u_vector: NDArray[np.float64],
        ) -> FocalLengthResult:
            """Invoca el fibrador capturando fallos como FibratorIntegrationError."""
            try:
                result = self._fibrator(
                    velocity_gradient,
                    shear_tensor,
                    vorticity_tensor,
                    stress_tensor,
                    u_vector,
                )
            except (
                RaychaudhuriFibratorError,
                TopologicalInvariantError,
            ) as exc:
                raise FibratorIntegrationError(
                    f"Fibrador de Raychaudhuri rechazó la congruencia: {exc}"
                ) from exc
            except Exception as exc:
                raise FibratorIntegrationError(
                    f"Error inesperado del fibrador: {exc}"
                ) from exc

            if not isinstance(result, FocalLengthResult):
                raise FibratorIntegrationError(
                    f"El fibrador devolvió tipo inesperado: {type(result)!r}."
                )
            return result

        def _fibrator_flags_viable(
            self, focal: FocalLengthResult
        ) -> bool:
            """Interroga el retículo booleano del fibrador si está disponible."""
            flags = getattr(focal, "viability_flags", None)
            if flags is None:
                # Stub o versión antigua: aceptar si τ_c finito y > 0
                return (
                    math.isfinite(focal.caustic_affine_parameter)
                    and focal.caustic_affine_parameter >= 0.0
                )
            is_order = getattr(flags, "is_order_unit", None)
            if callable(is_order):
                return bool(is_order())
            return True

        def compute_caustic(
            self,
            energy: EnergyAuditResult,
            velocity_gradient: NDArray[np.float64],
            shear_tensor: NDArray[np.float64],
            vorticity_tensor: NDArray[np.float64],
            stress_tensor: NDArray[np.float64],
            u_vector: NDArray[np.float64],
        ) -> CausticResult:
            r"""
            Último método de la Fase 2. Invoca el fibrador de Raychaudhuri
            y empaqueta la cáustica junto con la auditoría energética
            heredada para consumo exclusivo de la Fase 3.

            Este es el último método público de la Fase 2; su tipo de
            retorno `CausticResult` es exactamente el tipo de entrada del
            método principal de la clase anidada
            `Phase3_PenroseSingularityAuditor`.
            """
            B = np.asarray(velocity_gradient, dtype=np.float64)
            sigma = np.asarray(shear_tensor, dtype=np.float64)
            omega = np.asarray(vorticity_tensor, dtype=np.float64)
            stress = np.asarray(stress_tensor, dtype=np.float64)
            u = np.asarray(u_vector, dtype=np.float64)

            self._assert_kinematic_dimensions(
                energy, B, sigma, omega, stress, u
            )

            # Consistencia θ₀: Tr(B) debe coincidir con la auditoría
            theta_recomputed = float(np.trace(B))
            if abs(theta_recomputed - energy.theta_0) > max(
                1e-10, 1e-8 * abs(energy.theta_0)
            ):
                logger.warning(
                    "θ recomputado (%.6e) ≠ θ₀ auditado (%.6e); "
                    "se prosigue con el audit de Fase 1.",
                    theta_recomputed, energy.theta_0,
                )

            focal = self._invoke_fibrator(B, sigma, omega, stress, u)
            flags_ok = self._fibrator_flags_viable(focal)

            method_name = (
                focal.caustic_method.name
                if hasattr(focal.caustic_method, "name")
                else str(self._caustic_method)
            )

            logger.debug(
                "Fase 2 OK | f_opt=%.4e | τ_c=%.4e | τ_HP_fib=%.4e | "
                "dθ/dτ₀=%.4e | method=%s | flags_ok=%s",
                focal.optimal_focal_length,
                focal.caustic_affine_parameter,
                getattr(focal, "hawking_penrose_bound", float("nan")),
                getattr(focal, "initial_dtheta_dtau", float("nan")),
                method_name,
                flags_ok,
            )

            return CausticResult(
                focal_distance=float(focal.optimal_focal_length),
                tau_caustic=float(focal.caustic_affine_parameter),
                hawking_penrose_bound_fibrator=float(
                    getattr(focal, "hawking_penrose_bound", energy.tau_hp)
                ),
                initial_dtheta_dtau=float(
                    getattr(focal, "initial_dtheta_dtau", 0.0)
                ),
                area_collapse_factor=float(
                    getattr(focal, "area_collapse_factor", 0.0)
                ),
                fibrator_method=method_name,
                fibrator_flags_ok=flags_ok,
                energy_audit=energy,
                fibrator_diagnostics=focal,
            )

        # =====================================================================
        # FASE 3 (ANIDADA EN FASE 2) – VEREDICTO DEL TEOREMA DE PENROSE
        # =====================================================================
        class Phase3_PenroseSingularityAuditor:
            r"""
            Tercera fase anidada: morfismo de veredicto del Teorema de
            Singularidad de Penrose.

            Recibe el `CausticResult` de la Fase 2 (con auditoría energética
            embebida) y certifica:

                τ_c ≤ τ_HP · (1 + ε_margin)

            Construye el retículo booleano de predicados y emite el
            `CollapsedAffineState` con certificado inmutable.

            Cierra la cadena anidada Fase1 → Fase2 → Fase3.
            """

            def __init__(
                self,
                margin_tol: float = _PENROSE_MARGIN_TOL,
                require_fibrator_flags: bool = True,
            ) -> None:
                if margin_tol < 0.0:
                    raise ValueError("margin_tol debe ser ≥ 0.")
                self._margin_tol = float(margin_tol)
                self._require_fibrator_flags = bool(require_fibrator_flags)

            def _relative_margin(
                self, tau_c: float, tau_hp: float
            ) -> float:
                r"""
                Holgura relativa: m = (τ_HP − τ_c) / τ_HP.
                m ≥ 0  ⇒  dentro de la cota; m < 0  ⇒  violación.
                """
                if tau_hp <= _MACHINE_EPSILON:
                    return 0.0 if tau_c <= _MACHINE_EPSILON else -math.inf
                return (tau_hp - tau_c) / tau_hp

            def _build_certificate_flags(
                self,
                energy: EnergyAuditResult,
                caustic: CausticResult,
                within_bound: bool,
            ) -> SingularityCertificateFlags:
                """Ensambla el retículo booleano de predicados de singularidad."""
                flags = SingularityCertificateFlags.NONE

                if energy.theta_0 < -_MACHINE_EPSILON:
                    flags |= SingularityCertificateFlags.NEGATIVE_EXPANSION

                if energy.ricci_contraction >= -_MACHINE_EPSILON:
                    flags |= SingularityCertificateFlags.STRONG_ENERGY_SATISFIED

                target = 1.0  # riemanniano por defecto del audit
                if abs(energy.u_normalization - target) <= _U_NORMALIZATION_TOL:
                    flags |= SingularityCertificateFlags.U_NORMALIZED
                elif abs(energy.u_normalization + 1.0) <= _U_NORMALIZATION_TOL:
                    # lorentziano admisible
                    flags |= SingularityCertificateFlags.U_NORMALIZED

                if within_bound:
                    flags |= SingularityCertificateFlags.CAUSTIC_WITHIN_HP_BOUND

                if caustic.fibrator_flags_ok or not self._require_fibrator_flags:
                    flags |= SingularityCertificateFlags.FIBRATOR_VIABLE

                if (
                    energy.lambda_min_G > 0.0
                    and energy.metric_cond < math.inf
                ):
                    flags |= SingularityCertificateFlags.METRIC_RIEMANNIAN

                return flags

            def verify_and_certify(
                self,
                caustic: CausticResult,
            ) -> CollapsedAffineState:
                r"""
                Último método de la Fase 3. Aplica el teorema de Penrose:

                    τ_c ≤ τ_HP · (1 + margin_tol)

                Si se viola, lanza `PenroseSingularityVetoError` (fuga
                topológica). Si el retículo no es unidad de orden, lanza
                `ViabilityLatticeError`.

                Cierra la cadena:

                    audit_energy → compute_caustic → verify_and_certify
                """
                energy = caustic.energy_audit
                tau_c = caustic.tau_caustic
                tau_hp = energy.tau_hp

                # Sanidad numérica de la cáustica
                if not math.isfinite(tau_c) or tau_c < 0.0:
                    raise PenroseSingularityVetoError(
                        f"Parámetro afín de cáustica inválido: τ_c={tau_c}."
                    )

                # Teorema de enfoque: τ_c ≤ τ_HP (con holgura relativa)
                margin = self._relative_margin(tau_c, tau_hp)
                within_bound = tau_c <= tau_hp * (1.0 + self._margin_tol)

                if not within_bound:
                    raise PenroseSingularityVetoError(
                        f"Fuga topológica: cáustica τ_c={tau_c:.6e} excede la "
                        f"cota de Hawking–Penrose τ_HP={tau_hp:.6e} "
                        f"(margen relativo={margin:.4e})."
                    )

                # Consistencia cruzada con la cota del fibrador
                tau_hp_fib = caustic.hawking_penrose_bound_fibrator
                if (
                    math.isfinite(tau_hp_fib)
                    and tau_hp_fib > 0.0
                    and abs(tau_hp_fib - tau_hp) > max(1e-8, 1e-6 * tau_hp)
                ):
                    logger.warning(
                        "Discrepancia τ_HP audit=%.6e vs fibrador=%.6e; "
                        "se usa la cota de la Fase 1 (autoritativa).",
                        tau_hp, tau_hp_fib,
                    )

                flags = self._build_certificate_flags(
                    energy, caustic, within_bound
                )

                if not flags.is_order_unit():
                    raise ViabilityLatticeError(
                        f"Retículo de singularidad incompleto: flags={flags!r}. "
                        "No se certifica colapso inevitable."
                    )

                diagnostics = SingularityDiagnostics(
                    initial_expansion_scalar=energy.theta_0,
                    ricci_contraction=energy.ricci_contraction,
                    max_theoretical_affine_limit=tau_hp,
                    actual_caustic_parameter=tau_c,
                    relative_margin=float(margin),
                    is_singularity_inevitable=True,
                    certificate_flags=flags,
                    spatial_dim=energy.spatial_dim,
                    metric_condition_number=energy.metric_cond,
                )

                logger.info(
                    "Fase 3 OK | Singularidad inevitable | τ_c=%.6e ≤ τ_HP=%.6e "
                    "| margen=%.4e | f_opt=%.6e | flags=%s",
                    tau_c, tau_hp, margin, caustic.focal_distance, flags.name,
                )

                return CollapsedAffineState(
                    focal_distance=caustic.focal_distance,
                    diagnostics=diagnostics,
                    caustic=caustic,
                )


# ══════════════════════════════════════════════════════════════════════════════
# ENDOFUNTORE SUPREMO: PENROSE SINGULARITY AGENT
# ══════════════════════════════════════════════════════════════════════════════
class PenroseSingularityAgent(Morphism):
    r"""
    Endofuntor Supremo de Singularidad 𝒫: ℰ_MIC → ℰ_MIC.

    Materializa el morfismo de colapso geodésico exigiendo el cumplimiento
    del Teorema de Singularidad de Penrose sobre cada estado categórico.

    Orquesta las tres fases anidadas como composición de morfismos:

        𝒫 = certify_Penrose ∘ integrate_caustic ∘ audit_energy

      1. Phase1_HawkingEnergyAuditor.audit_energy
            → EnergyAuditResult
      2. Phase2_RaychaudhuriCausticIntegrator.compute_caustic
            → CausticResult
      3. Phase3_PenroseSingularityAuditor.verify_and_certify
            → CollapsedAffineState

    La anidación de clases refleja la dependencia funtorial estricta:
    no existe camino que omita la auditoría energética ni la cáustica
    antes del veredicto de Penrose.
    """

    # Claves canónicas del fibrado tensorial en CategoricalState
    KEY_VELOCITY: Final[str] = "velocity_field"
    KEY_SHEAR: Final[str] = "shear_tensor"
    KEY_VORTICITY: Final[str] = "vorticity_tensor"
    KEY_STRESS: Final[str] = "stress_tensor"
    KEY_ATTENTION: Final[str] = "attention_vector"
    KEY_FOCAL_DISTANCE: Final[str] = "focal_distance"
    KEY_SINGULARITY_DIAG: Final[str] = "singularity_diagnostics"

    def __init__(
        self,
        spatial_dimensions: int = _DEFAULT_SPATIAL_DIM,
        metric: Optional[NDArray[np.float64]] = None,
        caustic_method: CausticMethod = CausticMethod.ANALYTICAL_EXACT,
        use_numerical_integration: bool = False,
        signature: MetricSignatureKind = MetricSignatureKind.RIEMANNIAN,
        margin_tol: float = _PENROSE_MARGIN_TOL,
        require_fibrator_flags: bool = True,
    ) -> None:
        if spatial_dimensions < _MIN_SPATIAL_DIM:
            raise ValueError(
                f"spatial_dimensions={spatial_dimensions} < n_min={_MIN_SPATIAL_DIM}."
            )
        self._dim = int(spatial_dimensions)
        self._metric = (
            np.asarray(metric, dtype=np.float64) if metric is not None
            else np.asarray(G_PHYSICS, dtype=np.float64)
        )
        self._signature = signature
        self._margin_tol = float(margin_tol)
        self._require_fibrator_flags = bool(require_fibrator_flags)

        if use_numerical_integration:
            caustic_method = CausticMethod.NUMERICAL_IVP
        self._caustic_method = caustic_method
        self._use_numerical = bool(use_numerical_integration)

        # Mapa de firma local → firma del fibrador
        fib_sig = (
            MetricSignature.RIEMANNIAN
            if signature == MetricSignatureKind.RIEMANNIAN
            else MetricSignature.LORENTZIAN
        )

        # Torre anidada (reutilizable entre invocaciones)
        self._phase1 = Phase1_HawkingEnergyAuditor(
            metric=self._metric,
            signature=self._signature,
        )
        self._phase2 = (
            Phase1_HawkingEnergyAuditor.Phase2_RaychaudhuriCausticIntegrator(
                spatial_dim=self._dim,
                metric=self._metric,
                caustic_method=self._caustic_method,
                use_numerical=self._use_numerical,
                signature=fib_sig,
            )
        )
        self._phase3 = (
            Phase1_HawkingEnergyAuditor
            .Phase2_RaychaudhuriCausticIntegrator
            .Phase3_PenroseSingularityAuditor(
                margin_tol=self._margin_tol,
                require_fibrator_flags=self._require_fibrator_flags,
            )
        )

    # ─── extracción de tensores del estado ───────────────────────────────────

    def _extract_tensors(
        self, state: CategoricalState
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """Extrae el 5-tuplo (B, σ, ω, 𝒯, u) del CategoricalState."""
        required = (
            self.KEY_VELOCITY,
            self.KEY_SHEAR,
            self.KEY_VORTICITY,
            self.KEY_STRESS,
            self.KEY_ATTENTION,
        )
        tensors: Dict[str, Any] = {}
        missing = []
        for key in required:
            try:
                tensors[key] = state.get_tensor(key)
            except (KeyError, AttributeError):
                missing.append(key)

        if missing:
            raise MissingTensorError(
                f"Tensores ausentes en el fibrado del estado: {missing}."
            )

        return (
            np.asarray(tensors[self.KEY_VELOCITY], dtype=np.float64),
            np.asarray(tensors[self.KEY_SHEAR], dtype=np.float64),
            np.asarray(tensors[self.KEY_VORTICITY], dtype=np.float64),
            np.asarray(tensors[self.KEY_STRESS], dtype=np.float64),
            np.asarray(tensors[self.KEY_ATTENTION], dtype=np.float64),
        )

    # ─── ejecución canónica del pipeline ─────────────────────────────────────

    def execute_singularity_audit(
        self,
        velocity_gradient: NDArray[np.float64],
        shear_tensor: NDArray[np.float64],
        vorticity_tensor: NDArray[np.float64],
        stress_tensor: NDArray[np.float64],
        u_vector: NDArray[np.float64],
    ) -> CollapsedAffineState:
        r"""
        Ejecución canónica del morfismo 𝒫 sobre el 5-tuplo tensorial
        (sin depender de CategoricalState).

        Aplica la composición:

            (B,𝒯,u)  ──audit_energy────▶  EnergyAuditResult
            (B,σ,ω,𝒯,u) ──caustic──────▶  CausticResult
                         ──certify─────▶  CollapsedAffineState
        """
        logger.debug(
            "Iniciando auditoría de singularidad de Penrose | n=%d | method=%s",
            self._dim, self._caustic_method.name
            if hasattr(self._caustic_method, "name")
            else str(self._caustic_method),
        )

        # 1. Auditoría energética + cota HP
        energy = self._phase1.audit_energy(
            velocity_gradient, stress_tensor, u_vector, self._dim
        )

        # 2. Cáustica vía fibrador de Raychaudhuri
        caustic = self._phase2.compute_caustic(
            energy,
            velocity_gradient,
            shear_tensor,
            vorticity_tensor,
            stress_tensor,
            u_vector,
        )

        # 3. Veredicto de Penrose
        collapsed = self._phase3.verify_and_certify(caustic)

        logger.info(
            "Colapso certificado | f_opt=%.6e | τ_c=%.6e ≤ τ_HP=%.6e | margen=%.4e",
            collapsed.focal_distance,
            collapsed.diagnostics.actual_caustic_parameter,
            collapsed.diagnostics.max_theoretical_affine_limit,
            collapsed.diagnostics.relative_margin,
        )
        return collapsed

    def __call__(
        self, state: CategoricalState, **kwargs: Any
    ) -> CategoricalState:
        r"""
        Interfaz Morphism: extrae tensores del estado, ejecuta 𝒫 y adjunta
        `focal_distance` + `singularity_diagnostics` al estado resultante.
        """
        logger.info(
            "Aplicando Teorema de Singularidad de Penrose a la congruencia "
            "semántica (n=%d)...",
            self._dim,
        )

        B, sigma, omega, stress, u = self._extract_tensors(state)
        collapsed = self.execute_singularity_audit(
            B, sigma, omega, stress, u
        )

        # Mutación funtorial: preserva el protocolo update del estado
        new_state = state.update(
            self.KEY_FOCAL_DISTANCE, collapsed.focal_distance
        )
        # Intentar adjuntar diagnósticos si el estado lo soporta
        try:
            new_state = new_state.update(
                self.KEY_SINGULARITY_DIAG, collapsed.diagnostics
            )
        except Exception:
            logger.debug(
                "El estado no aceptó la clave '%s'; se omite.",
                self.KEY_SINGULARITY_DIAG,
            )

        logger.info(
            "Colapso de intención LLM forzado. Cáustica τ=%.6e dentro del "
            "límite τ_HP=%.6e (margen=%.4e).",
            collapsed.diagnostics.actual_caustic_parameter,
            collapsed.diagnostics.max_theoretical_affine_limit,
            collapsed.diagnostics.relative_margin,
        )
        return new_state

    def describe_pipeline(self) -> Dict[str, str]:
        """Metadatos del pipeline anidado (introspección categórica ligera)."""
        return {
            "functor": "𝒫 (PenroseSingularityAgent)",
            "version": "4.0.0-Nested-Hawking-Penrose-Spectral-Topos",
            "phase_1": "Phase1_HawkingEnergyAuditor.audit_energy",
            "phase_2": "Phase2_RaychaudhuriCausticIntegrator.compute_caustic",
            "phase_3": "Phase3_PenroseSingularityAuditor.verify_and_certify",
            "axiom_1": "θ₀ = Tr(B) < 0  (expansión inicial convergente)",
            "axiom_2": "SEC: (𝒯 − ½𝒯 G)(u,u) ≥ 0  ⇒  R(u,u) ≥ 0",
            "axiom_3": "τ_c ≤ (n−1)/|θ₀|  (Hawking–Penrose / Penrose 1965)",
            "axiom_4": "certificado ⇔ retículo booleano = ALL",
            "caustic_method": (
                self._caustic_method.name
                if hasattr(self._caustic_method, "name")
                else str(self._caustic_method)
            ),
            "signature": self._signature.name,
            "spatial_dim": str(self._dim),
            "margin_tol": f"{self._margin_tol:.1e}",
        }


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    # excepciones
    "PenroseAgentError",
    "PenroseSingularityVetoError",
    "EnergyConvergenceViolationError",
    "MetricTensorDegeneracyError",
    "DimensionMismatchError",
    "NormalizationError",
    "TensorSymmetryError",
    "MissingTensorError",
    "FibratorIntegrationError",
    "ViabilityLatticeError",
    # enums / flags
    "SingularityCertificateFlags",
    "MetricSignatureKind",
    # DTOs
    "EnergyAuditResult",
    "CausticResult",
    "SingularityDiagnostics",
    "CollapsedAffineState",
    # fases + orquestador
    "Phase1_HawkingEnergyAuditor",
    "PenroseSingularityAgent",
]