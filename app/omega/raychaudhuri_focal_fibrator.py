# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Raychaudhuri Focal Fibrator (Colapso Geodésico y Límite Afín)        ║
║ Ruta   : app/omega/raychaudhuri_focal_fibrator.py                             ║
║ Versión: 3.0.0-Nested-Caustic-Spectral-Topos                                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GEOMETRÍA DIFERENCIAL
────────────────────────────────────────────────────────────────────────────────
Operador de Convergencia Geodésica entre la difracción del
`optical_riemann_lens.py` y la proyección catadióptrica de von Neumann del
`semantic_parabolic_mirror.py`.

Control Port-Hamiltoniano sobre el escalar de expansión θ de la congruencia
de geodésicas semánticas a lo largo del parámetro afín τ, con descomposición
espectral del endomorfismo de Jacobi B^μ_ν = ∇_ν u^μ y certificación de la
Condición de Energía Fuerte (SEC) vía el tensor de Cauchy-Momentum del
`watcher_agent`.

ARQUITECTURA ANIDADA (3 fases — composición funtorial estricta)
────────────────────────────────────────────────────────────────────────────────
  Fase 1 → Cinemática espectral de la congruencia:
           descomposición B = (θ/(n-1)) h + σ + ω, invariantes θ, σ², ω²,
           consistencia métrica y residual de descomposición.
  Fase 2 (anidada en Fase 1) → Condición de Energía Fuerte (SEC):
           (𝒯_{μν} − ½ 𝒯 G_{μν}) u^μ u^ν ≥ 0  ⇒  R_{μν} u^μ u^ν ≥ 0.
  Fase 3 (anidada en Fase 2) → Integración de Raychaudhuri:
           cota de Hawking–Penrose, cáustica τ_c, distancia focal f_opt.

AXIOMAS DE EJECUCIÓN (nivel PhD — GR + teoría espectral + topos)
────────────────────────────────────────────────────────────────────────────────
§1. VORTICIDAD NULA (isomorfismo de Hodge–Helmholtz / Frobenius):
    ω_{μν} ≡ 0  (el flujo fue purgado de ciclos solenoidales, β₁ = 0).
    Residuo relativo: ‖ω‖_G / (‖B‖_G + ε_mach) < τ_ω.

§2. CONDICIÓN DE ENERGÍA FUERTE (SEC) + Einstein (8πG = 1, Λ = 0):
    (𝒯_{μν} − ½ 𝒯 G_{μν}) u^μ u^ν ≥ 0
    ⇒  R_{μν} u^μ u^ν ≥ 0  (atracción geodésica del haz semántico).
    u normalizado: |G(u,u) − s| < τ_u  (s = +1 riemanniano / −1 lorentziano).

§3. TEOREMA DE ENFOQUE (Hawking–Penrose) Y DISTANCIA FOCAL:
    Si θ₀ < 0, ω ≡ 0 y R_{μν} u^μ u^ν ≥ 0, entonces existe cáustica en
        τ_c ≤ τ_HP := (n−1) / |θ₀|.
    Ecuación de Raychaudhuri (ω = 0):
        dθ/dτ = − θ²/(n−1) − σ² − R_{μν} u^μ u^ν.
    f_opt se define como la escala afín de colapso regularizada:
        f_opt = τ_c · exp(∫₀^{τ_c} θ(s)/(n−1) ds)   (área transversal → 0).
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
from scipy.integrate import solve_ivp

# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, CategoricalState, TopologicalInvariantError
except ImportError:
    class TopologicalInvariantError(Exception):
        """Violación a invariante topológico categórico en el Topos E_MIC."""
        pass

    class Morphism:
        """Stub para Morphism del 2-categoría de agentes."""
        pass

    class CategoricalState:
        pass

try:
    from app.core.immune_system.metric_tensors import G_PHYSICS
except ImportError:
    # Tensor métrico euclídeo de dimensión canónica 4 (fallback unitario).
    G_PHYSICS: NDArray[np.float64] = np.eye(4, dtype=np.float64)

logger = logging.getLogger("MIC.Omega.RaychaudhuriFibrator")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS, ESPECTRALES Y NUMÉRICAS (alta precisión)
# ══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPSILON: Final[float] = float(np.finfo(np.float64).eps)
_VORTICITY_TOLERANCE: Final[float] = 1e-12
_SYMMETRY_REL_TOL: Final[float] = 1e-10
_TRACELESS_TOL: Final[float] = 1e-10
_DECOMPOSITION_TOL: Final[float] = 1e-8
_U_NORMALIZATION_TOL: Final[float] = 1e-8
_CAUSTIC_THETA_THRESHOLD: Final[float] = -1.0e6
_MAX_AFFINE_PARAMETER: Final[float] = 1.0e6
_DEFAULT_RTOL: Final[float] = 1e-9
_DEFAULT_ATOL: Final[float] = 1e-12
_COND_MAX: Final[float] = 1e12
_MIN_SPATIAL_DIM: Final[int] = 2  # n≥2 para que n−1 ≥ 1 en Raychaudhuri


# ══════════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES (vetos topológicos / fallos de invariantes)
# ══════════════════════════════════════════════════════════════════════════════
class RaychaudhuriFibratorError(TopologicalInvariantError):
    """Excepción raíz del Operador de Convergencia Geodésica."""
    pass


class FocalDivergenceVetoError(RaychaudhuriFibratorError):
    r"""
    dθ/dτ ≱ 0 de forma convergente, o θ₀ ≥ 0:
    dispersión entrópica de la intención (alucinación divergente).
    """
    pass


class StrongEnergyViolationError(RaychaudhuriFibratorError):
    r"""
    (𝒯_{μν} − ½ 𝒯 G_{μν}) u^μ u^ν < 0: violación SEC → repulsión geodésica.
    """
    pass


class VorticityAnomalyError(RaychaudhuriFibratorError):
    r"""ω_{μν} ≠ 0 o residual solenoidal por encima de tolerancia (β₁ ≠ 0)."""
    pass


class ShearAnomalyError(RaychaudhuriFibratorError):
    """σ no simétrico, no traceless, o inconsistente con la descomposición de B."""
    pass


class MetricSignatureError(RaychaudhuriFibratorError):
    """Métrica no simétrica, no definida positiva, o mal condicionada."""
    pass


class DimensionMismatchError(RaychaudhuriFibratorError):
    """Inconsistencia dimensional entre B, σ, ω, 𝒯, u y G."""
    pass


class NormalizationError(RaychaudhuriFibratorError):
    """Vector u no normalizado respecto a G ( |G(u,u) − s| > tolerancia )."""
    pass


class CausticNotReachedError(RaychaudhuriFibratorError):
    """La integración afín no alcanzó cáustica dentro de τ_max."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# RETÍCULO BOOLEANO DE VIABILIDAD FOCAL (álgebra de Boole de predicados)
# ══════════════════════════════════════════════════════════════════════════════
class FocalViabilityFlags(enum.Flag):
    r"""Predicados estables de convergencia geodésica en el retículo booleano."""
    NONE = 0
    ZERO_VORTICITY = enum.auto()
    STRONG_ENERGY_SATISFIED = enum.auto()
    NEGATIVE_EXPANSION = enum.auto()
    CONVERGENT_RAYCHAUDHURI = enum.auto()
    CAUSTIC_REACHABLE = enum.auto()

    ALL = (
        ZERO_VORTICITY
        | STRONG_ENERGY_SATISFIED
        | NEGATIVE_EXPANSION
        | CONVERGENT_RAYCHAUDHURI
        | CAUSTIC_REACHABLE
    )

    def is_order_unit(self) -> bool:
        """True ⇔ todos los predicados de viabilidad están activos."""
        return self == FocalViabilityFlags.ALL


class CausticMethod(enum.Enum):
    """Estrategia de resolución del punto conjugado."""
    ANALYTICAL_BOUND = enum.auto()   # cota de Hawking–Penrose (+ solución exacta α=0)
    ANALYTICAL_EXACT = enum.auto()   # solución exacta con α = σ²+R constante
    NUMERICAL_IVP = enum.auto()      # integración adaptativa (solve_ivp)


class MetricSignature(enum.Enum):
    """Firma de la métrica de fondo (convención de normalización de u)."""
    RIEMANNIAN = enum.auto()   # G(u,u) = +1
    LORENTZIAN = enum.auto()   # G(u,u) = −1  (u tipo-tiempo)


# ══════════════════════════════════════════════════════════════════════════════
# DTOs INMUTABLES (objetos del topos — contratos entre fases anidadas)
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class KinematicExpansionData:
    r"""
    Producto canónico de la Fase 1: invariantes cinemáticos de la congruencia.

    Constituye el dominio formal del morfismo de la Fase 2.
    """
    expansion_scalar: float              # θ = ∇_μ u^μ = Tr(B)
    shear_magnitude: float               # σ² = σ_{μν} σ^{μν}  (≥ 0)
    vorticity_magnitude: float           # ω² = ω_{μν} ω^{μν}  (≈ 0)
    op_norm_B: float                     # ‖B‖_{op}
    frobenius_B: float                   # ‖B‖_F
    residual_decomposition: float        # ‖B − (θ/(n−1))I_h − σ − ω‖_F / (‖B‖_F+ε)
    spatial_dim: int                     # n = dim
    metric_cond: float                   # κ(G)


@dataclass(frozen=True, slots=True)
class EnergyConditionCertificate:
    r"""
    Producto canónico de la Fase 2: certificado SEC + contracción de Ricci.

    Dominio formal del morfismo de la Fase 3. Hereda cinemática intacta.
    """
    ricci_contraction: float             # R_{μν} u^μ u^ν  (≥ 0 bajo SEC + Einstein)
    sec_value: float                     # (𝒯 − ½ 𝒯 G)(u,u)
    stress_trace: float                  # 𝒯 = 𝒯_{μν} G^{μν}
    u_normalization: float               # G(u,u)
    is_strongly_attractive: bool
    kinematics: KinematicExpansionData   # reenvío funtorial a Fase 3


@dataclass(frozen=True, slots=True)
class FocalLengthResult:
    r"""
    Producto canónico de la Fase 3: distancia focal y cáustica afín.
    """
    optimal_focal_length: float          # f_opt
    caustic_affine_parameter: float      # τ_c
    hawking_penrose_bound: float         # τ_HP = (n−1)/|θ₀|
    initial_dtheta_dtau: float           # (dθ/dτ)|_{τ=0}
    caustic_method: CausticMethod
    viability_flags: FocalViabilityFlags
    area_collapse_factor: float          # exp(∫ θ/(n−1) dτ)  (→ 0 en cáustica)


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1 – CINEMÁTICA ESPECTRAL DE LA CONGRUENCIA GEODÉSICA
# ══════════════════════════════════════════════════════════════════════════════
class Phase1_RaychaudhuriKinematics:
    r"""
    Primera fase del pipeline geodésico (morfismo monádico de cinemática).

    Verifica y extrae:
      (i)   consistencia dimensional de (B, σ, ω, G);
      (ii)  firma riemanniana de G (Cholesky + espectro);
      (iii) simetría de σ, antisimetría de ω, traceless de σ;
      (iv)  vorticidad nula (axioma §1) con residual relativo;
      (v)   consistencia de la descomposición
            B ≈ (θ/(n−1)) P + σ + ω
            donde P es el proyector espacial (euclídeo: I);
      (vi)  invariantes espectrales ‖B‖_op, ‖B‖_F, κ(G).

    El método público terminal `compute_kinematics` devuelve un
    `KinematicExpansionData` que es el objeto de entrada formal de la
    Fase 2 anidada (`Phase2_StrongEnergyCondition`).
    """

    def __init__(
        self,
        metric: Optional[NDArray[np.float64]] = None,
        vorticity_tol: float = _VORTICITY_TOLERANCE,
        decomposition_tol: float = _DECOMPOSITION_TOL,
    ) -> None:
        if vorticity_tol <= 0.0:
            raise ValueError("vorticity_tol debe ser estrictamente positiva.")
        if decomposition_tol <= 0.0:
            raise ValueError("decomposition_tol debe ser estrictamente positiva.")

        self._G: NDArray[np.float64] = (
            np.asarray(metric, dtype=np.float64) if metric is not None
            else np.asarray(G_PHYSICS, dtype=np.float64)
        )
        self._vorticity_tol = float(vorticity_tol)
        self._decomposition_tol = float(decomposition_tol)

        self._lam_min_G: float
        self._lam_max_G: float
        self._cond_G: float
        self._G_inv: NDArray[np.float64]
        self._lam_min_G, self._lam_max_G, self._cond_G, self._G_inv = (
            self._validate_and_factor_metric(self._G)
        )

    # ─── utilidades de álgebra lineal numérica ───────────────────────────────

    @staticmethod
    def _frobenius_norm(A: NDArray[np.float64]) -> float:
        """‖A‖_F = √(Σ_{ij} A_{ij}²)."""
        return float(np.linalg.norm(A, ord="fro"))

    @staticmethod
    def _relative_symmetry_error(A: NDArray[np.float64]) -> float:
        """‖A − Aᵀ‖_F / (‖A‖_F + ε_mach)."""
        fro = float(np.linalg.norm(A, ord="fro"))
        skew = float(np.linalg.norm(A - A.T, ord="fro"))
        return skew / (fro + _MACHINE_EPSILON)

    @staticmethod
    def _relative_antisymmetry_error(A: NDArray[np.float64]) -> float:
        """‖A + Aᵀ‖_F / (‖A‖_F + ε_mach)."""
        fro = float(np.linalg.norm(A, ord="fro"))
        sym = float(np.linalg.norm(A + A.T, ord="fro"))
        return sym / (fro + _MACHINE_EPSILON)

    @staticmethod
    def _safe_eigvalsh(A: NDArray[np.float64]) -> NDArray[np.float64]:
        """Autovalores hermitianos con re-simetrización de Frobenius."""
        A_sym = 0.5 * (A + A.T)
        return la.eigvalsh(A_sym)

    def _metric_raise_both(
        self, T_low: NDArray[np.float64]
    ) -> float:
        r"""
        Contracción total T_{μν} T^{μν} = T_{μν} G^{μα} G^{νβ} T_{αβ}.
        Devuelve el escalar ‖T‖²_G (magnitud al cuadrado con métrica).
        """
        G_inv = self._G_inv
        return float(
            np.einsum("ij,ik,jl,kl->", T_low, G_inv, G_inv, T_low, optimize=True)
        )

    def _metric_trace(self, T_low: NDArray[np.float64]) -> float:
        r"""Tr_G(T) = T_{μν} G^{μν}."""
        return float(np.einsum("ij,ij->", T_low, self._G_inv, optimize=True))

    # ─── métrica ─────────────────────────────────────────────────────────────

    def _validate_and_factor_metric(
        self, G: NDArray[np.float64]
    ) -> Tuple[float, float, float, NDArray[np.float64]]:
        """
        Verifica simetría + PD (Cholesky) y devuelve (λ_min, λ_max, κ, G⁻¹).
        """
        if G.ndim != 2 or G.shape[0] != G.shape[1]:
            raise MetricSignatureError(
                f"G debe ser cuadrada; recibido shape={G.shape}."
            )
        if G.shape[0] < _MIN_SPATIAL_DIM:
            raise MetricSignatureError(
                f"dim(G)={G.shape[0]} < n_min={_MIN_SPATIAL_DIM}."
            )

        err = self._relative_symmetry_error(G)
        if err > _SYMMETRY_REL_TOL:
            raise MetricSignatureError(
                f"G no es simétrica: error relativo Frobenius = {err:.3e}."
            )

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
            raise MetricSignatureError(
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
        shear_tensor: NDArray[np.float64],
        vorticity_tensor: NDArray[np.float64],
    ) -> int:
        """Verifica que B, σ, ω y G compartan dimensión n×n. Devuelve n."""
        B = velocity_gradient
        if B.ndim != 2 or B.shape[0] != B.shape[1]:
            raise DimensionMismatchError(
                f"B=∇u debe ser cuadrado; shape={B.shape}."
            )
        n = B.shape[0]
        if self._G.shape != (n, n):
            raise DimensionMismatchError(
                f"G shape {self._G.shape} incompatible con dim(B)={n}."
            )
        if shear_tensor.shape != (n, n):
            raise DimensionMismatchError(
                f"σ shape {shear_tensor.shape} ≠ ({n},{n})."
            )
        if vorticity_tensor.shape != (n, n):
            raise DimensionMismatchError(
                f"ω shape {vorticity_tensor.shape} ≠ ({n},{n})."
            )
        if n < _MIN_SPATIAL_DIM:
            raise DimensionMismatchError(
                f"n={n} < n_min={_MIN_SPATIAL_DIM} (Raychaudhuri requiere n≥2)."
            )
        return n

    # ─── chequeos de σ y ω ───────────────────────────────────────────────────

    def _check_shear(self, sigma: NDArray[np.float64]) -> float:
        r"""
        Exige simetría y traceless de σ (con métrica).
        Devuelve σ² = σ_{μν} σ^{μν}.
        """
        err = self._relative_symmetry_error(sigma)
        if err > _SYMMETRY_REL_TOL:
            raise ShearAnomalyError(
                f"σ no es simétrico: error relativo Frobenius = {err:.3e}."
            )

        tr = self._metric_trace(sigma)
        if abs(tr) > _TRACELESS_TOL:
            raise ShearAnomalyError(
                f"σ no es traceless: Tr_G(σ) = {tr:.3e}."
            )

        sigma_sq = self._metric_raise_both(sigma)
        # σ² debe ser ≥ 0 por construcción (forma de Hilbert–Schmidt).
        if sigma_sq < -_MACHINE_EPSILON:
            raise ShearAnomalyError(
                f"σ² = {sigma_sq:.3e} < 0 (inconsistencia numérica)."
            )
        return max(0.0, sigma_sq)

    def _check_vorticity(self, omega: NDArray[np.float64]) -> float:
        r"""
        Exige antisimetría de ω y ω² ≈ 0 (axioma §1).
        Devuelve ω².
        """
        err = self._relative_antisymmetry_error(omega)
        if err > _SYMMETRY_REL_TOL:
            raise VorticityAnomalyError(
                f"ω no es antisimétrico: error relativo = {err:.3e}."
            )

        omega_sq = self._metric_raise_both(omega)
        omega_sq = max(0.0, omega_sq)

        if omega_sq > self._vorticity_tol:
            raise VorticityAnomalyError(
                f"Fuga solenoidal detectada: ω² = {omega_sq:.4e} "
                f"> tol = {self._vorticity_tol:.1e}."
            )
        return omega_sq

    def _check_decomposition(
        self,
        B: NDArray[np.float64],
        theta: float,
        sigma: NDArray[np.float64],
        omega: NDArray[np.float64],
        n: int,
    ) -> float:
        r"""
        Residuo relativo de la descomposición cinemática (caso riemanniano
        con proyector espacial ≈ I, válido cuando u está normalizado y se
        trabaja en el hiperplano ortogonal en coordenadas adaptadas):

            B ≟ (θ/(n−1)) I + σ + ω

        En GR completa el proyector es h^μ_ν = δ^μ_ν − s u^μ u_ν; aquí se
        usa la forma euclídea canónica del pipeline semántico (u espacial
        absorbido en la carta local del optical lens).

        Devuelve residual_rel = ‖B − B_rec‖_F / (‖B‖_F + ε).
        """
        expansion_part = (theta / float(n - 1)) * np.eye(n, dtype=np.float64)
        B_rec = expansion_part + sigma + omega
        residual_abs = self._frobenius_norm(B - B_rec)
        scale = self._frobenius_norm(B) + _MACHINE_EPSILON
        residual_rel = residual_abs / scale

        if residual_rel > self._decomposition_tol:
            logger.warning(
                "Descomposición cinemática inconsistente: residual_rel=%.3e "
                "(tol=%.1e). Se prosigue con θ=Tr(B) canónico.",
                residual_rel, self._decomposition_tol,
            )
        return residual_rel

    # ─── método terminal de la Fase 1 (contrato de entrada de la Fase 2) ─────

    def compute_kinematics(
        self,
        velocity_gradient: NDArray[np.float64],
        shear_tensor: NDArray[np.float64],
        vorticity_tensor: NDArray[np.float64],
    ) -> KinematicExpansionData:
        r"""
        Ejecuta la cinemática completa de la congruencia y devuelve el
        objeto `KinematicExpansionData` que constituye el dominio formal
        del morfismo de certificación energética de la Fase 2.

        Pipeline interno:
          dim ✓ → σ sim+traceless ✓ → ω anti+nulo ✓ → θ=Tr(B) →
          descomposición residual → espectro de B

        Este es el último método público de la Fase 1; su tipo de retorno
        `KinematicExpansionData` es exactamente el tipo de entrada del
        método principal de la clase anidada `Phase2_StrongEnergyCondition`.
        """
        B = np.asarray(velocity_gradient, dtype=np.float64)
        sigma = np.asarray(shear_tensor, dtype=np.float64)
        omega = np.asarray(vorticity_tensor, dtype=np.float64)

        # (i) Dimensiones
        n = self._assert_dimensions(B, sigma, omega)

        # (ii)–(iii) σ y ω
        sigma_sq = self._check_shear(sigma)
        omega_sq = self._check_vorticity(omega)

        # (iv) Expansión canónica θ = Tr(B)
        theta = float(np.trace(B))

        # (v) Residuo de descomposición
        residual_rel = self._check_decomposition(B, theta, sigma, omega, n)

        # (vi) Espectro de B
        op_norm_B = float(np.linalg.norm(B, ord=2))
        frobenius_B = self._frobenius_norm(B)

        logger.debug(
            "Fase 1 OK | n=%d | θ=%.4e | σ²=%.4e | ω²=%.4e | "
            "‖B‖_op=%.4e | res_dec=%.3e | κ(G)=%.2e",
            n, theta, sigma_sq, omega_sq, op_norm_B, residual_rel, self._cond_G,
        )

        return KinematicExpansionData(
            expansion_scalar=theta,
            shear_magnitude=sigma_sq,
            vorticity_magnitude=omega_sq,
            op_norm_B=op_norm_B,
            frobenius_B=frobenius_B,
            residual_decomposition=residual_rel,
            spatial_dim=n,
            metric_cond=self._cond_G,
        )

    # =========================================================================
    # FASE 2 (ANIDADA EN FASE 1) – CONDICIÓN DE ENERGÍA FUERTE (SEC)
    # =========================================================================
    class Phase2_StrongEnergyCondition:
        r"""
        Segunda fase anidada: morfismo de certificación de la Condición de
        Energía Fuerte acoplada al Tensor de Estrés Macroscópico 𝒯_{μν}
        emanado por el `watcher_agent`.

        Recibe el `KinematicExpansionData` de la Fase 1, el tensor 𝒯 y el
        vector de 4-velocidad (o n-velocidad) u, y produce un
        `EnergyConditionCertificate` con R_{μν} u^μ u^ν bajo Einstein
        (8πG = 1, Λ = 0).

        El método terminal `certify_energy` devuelve el certificado que es
        el dominio formal de la Fase 3 anidada.
        """

        def __init__(
            self,
            metric: NDArray[np.float64],
            metric_inv: NDArray[np.float64],
            signature: MetricSignature = MetricSignature.RIEMANNIAN,
            u_norm_tol: float = _U_NORMALIZATION_TOL,
        ) -> None:
            if u_norm_tol <= 0.0:
                raise ValueError("u_norm_tol debe ser estrictamente positiva.")
            self._G = np.asarray(metric, dtype=np.float64)
            self._G_inv = np.asarray(metric_inv, dtype=np.float64)
            self._signature = signature
            self._u_norm_tol = float(u_norm_tol)
            self._target_norm = (
                1.0 if signature == MetricSignature.RIEMANNIAN else -1.0
            )

        def _assert_dimensions(
            self,
            stress_tensor: NDArray[np.float64],
            u_vector: NDArray[np.float64],
            kinematics: KinematicExpansionData,
        ) -> None:
            n = kinematics.spatial_dim
            if stress_tensor.shape != (n, n):
                raise DimensionMismatchError(
                    f"𝒯 shape {stress_tensor.shape} ≠ ({n},{n})."
                )
            if u_vector.ndim != 1 or u_vector.shape[0] != n:
                raise DimensionMismatchError(
                    f"u shape {u_vector.shape} incompatible con dim={n}."
                )
            if self._G.shape != (n, n):
                raise DimensionMismatchError(
                    f"G shape {self._G.shape} incompatible con dim={n}."
                )

        def _check_stress_symmetry(
            self, stress: NDArray[np.float64]
        ) -> None:
            """𝒯 debe ser simétrico (tensor de Cauchy-Momentum)."""
            fro = float(np.linalg.norm(stress, ord="fro"))
            skew = float(np.linalg.norm(stress - stress.T, ord="fro"))
            err = skew / (fro + _MACHINE_EPSILON)
            if err > _SYMMETRY_REL_TOL:
                raise StrongEnergyViolationError(
                    f"𝒯 no es simétrico: error relativo = {err:.3e}."
                )

        def _normalize_check_u(
            self, u: NDArray[np.float64]
        ) -> float:
            r"""
            Verifica |G(u,u) − s| < tol y devuelve G(u,u).
            No re-normaliza silenciosamente: un u mal normalizado es un
            error de protocolo del optical lens.
            """
            g_uu = float(u @ self._G @ u)
            if abs(g_uu - self._target_norm) > self._u_norm_tol:
                raise NormalizationError(
                    f"u no normalizado: G(u,u)={g_uu:.6e}, "
                    f"esperado s={self._target_norm} "
                    f"(tol={self._u_norm_tol:.1e}, firma={self._signature.name})."
                )
            return g_uu

        def _compute_sec_value(
            self,
            stress: NDArray[np.float64],
            u: NDArray[np.float64],
        ) -> Tuple[float, float]:
            r"""
            Evalúa:
              𝒯 = 𝒯_{μν} G^{μν}
              SEC = (𝒯_{μν} − ½ 𝒯 G_{μν}) u^μ u^ν
                  = 𝒯_{μν} u^μ u^ν − ½ 𝒯 · G(u,u)

            Devuelve (sec_value, stress_trace).
            """
            stress_trace = float(
                np.einsum("ij,ij->", stress, self._G_inv, optimize=True)
            )
            T_uu = float(u @ stress @ u)
            g_uu = float(u @ self._G @ u)
            sec_value = T_uu - 0.5 * stress_trace * g_uu
            return sec_value, stress_trace

        def certify_energy(
            self,
            kinematics: KinematicExpansionData,
            stress_tensor: NDArray[np.float64],
            u_vector: NDArray[np.float64],
        ) -> EnergyConditionCertificate:
            r"""
            Certifica la SEC y produce el `EnergyConditionCertificate`
            que alimenta exclusivamente a la Fase 3.

            Bajo Einstein (8πG = 1, Λ = 0):
                R_{μν} u^μ u^ν = (𝒯_{μν} − ½ 𝒯 G_{μν}) u^μ u^ν = SEC(u).

            Este es el último método público de la Fase 2; su tipo de
            retorno `EnergyConditionCertificate` es exactamente el tipo
            de entrada del método principal de la clase anidada
            `Phase3_FocalCollapse` (junto con la cinemática embebida).
            """
            stress = np.asarray(stress_tensor, dtype=np.float64)
            u = np.asarray(u_vector, dtype=np.float64)

            self._assert_dimensions(stress, u, kinematics)
            self._check_stress_symmetry(stress)
            g_uu = self._normalize_check_u(u)

            sec_value, stress_trace = self._compute_sec_value(stress, u)

            if sec_value < -_MACHINE_EPSILON:
                raise StrongEnergyViolationError(
                    f"Violación SEC: valor = {sec_value:.4e}. "
                    "Repulsión geodésica inducida (R_{μν}u^μu^ν < 0)."
                )

            # Einstein: R(u,u) = SEC(u)  (unidades 8πG = 1, Λ = 0)
            ricci_contraction = max(0.0, sec_value)  # clip numérico residual

            logger.debug(
                "Fase 2 OK | SEC=%.4e | Tr_G(𝒯)=%.4e | G(u,u)=%.6f | "
                "R(u,u)=%.4e",
                sec_value, stress_trace, g_uu, ricci_contraction,
            )

            return EnergyConditionCertificate(
                ricci_contraction=ricci_contraction,
                sec_value=sec_value,
                stress_trace=stress_trace,
                u_normalization=g_uu,
                is_strongly_attractive=True,
                kinematics=kinematics,
            )

        # =====================================================================
        # FASE 3 (ANIDADA EN FASE 2) – COLAPSO FOCAL Y DISTANCIA ÓPTIMA
        # =====================================================================
        class Phase3_FocalCollapse:
            r"""
            Tercera fase anidada: morfismo de integración de la ecuación de
            Raychaudhuri y determinación del punto conjugado (cáustica).

            Recibe el `EnergyConditionCertificate` de la Fase 2 (que embebe
            la cinemática de la Fase 1) y resuelve:

                dθ/dτ = − θ²/(n−1) − σ² − R_{μν} u^μ u^ν

            Métodos de resolución:
              • ANALYTICAL_BOUND — cota de Hawking–Penrose τ_HP = (n−1)/|θ₀|
              • ANALYTICAL_EXACT — solución exacta con α = σ² + R constante
              • NUMERICAL_IVP    — solve_ivp con evento de cáustica

            Cierra la cadena anidada Fase1 → Fase2 → Fase3.
            """

            def __init__(
                self,
                spatial_dim: int,
                method: CausticMethod = CausticMethod.ANALYTICAL_EXACT,
                max_affine: float = _MAX_AFFINE_PARAMETER,
                caustic_theta: float = _CAUSTIC_THETA_THRESHOLD,
                rtol: float = _DEFAULT_RTOL,
                atol: float = _DEFAULT_ATOL,
            ) -> None:
                if spatial_dim < _MIN_SPATIAL_DIM:
                    raise ValueError(
                        f"spatial_dim={spatial_dim} < n_min={_MIN_SPATIAL_DIM}."
                    )
                if max_affine <= 0.0:
                    raise ValueError("max_affine debe ser > 0.")
                if caustic_theta >= 0.0:
                    raise ValueError("caustic_theta debe ser estrictamente negativo.")

                self._n = int(spatial_dim)
                self._method = method
                self._max_affine = float(max_affine)
                self._caustic_theta = float(caustic_theta)
                self._rtol = float(rtol)
                self._atol = float(atol)

            # ── lado derecho de Raychaudhuri ─────────────────────────────────

            def raychaudhuri_rhs(
                self,
                theta: float,
                sigma_sq: float,
                ricci_term: float,
            ) -> float:
                r"""
                dθ/dτ = − θ²/(n−1) − σ² − R_{μν} u^μ u^ν
                (ω ≡ 0 por axioma §1).
                """
                n_m1 = float(self._n - 1)
                return -(theta * theta) / n_m1 - sigma_sq - ricci_term

            # ── cota de Hawking–Penrose ──────────────────────────────────────

            @staticmethod
            def hawking_penrose_bound(theta0: float, n: int) -> float:
                r"""
                τ_HP = (n−1) / |θ₀|  para θ₀ < 0.
                Garantiza existencia de cáustica en τ ∈ (0, τ_HP] bajo SEC.
                """
                if theta0 >= 0.0:
                    raise FocalDivergenceVetoError(
                        f"θ₀={theta0:.4e} ≥ 0: el haz no es inicialmente "
                        "convergente; cota de Hawking–Penrose no aplica."
                    )
                return float(n - 1) / abs(theta0)

            # ── solución analítica exacta (α = σ²+R constante) ───────────────

            def _analytical_exact_caustic(
                self,
                theta0: float,
                alpha: float,
            ) -> Tuple[float, float, float]:
                r"""
                Resuelve dθ/dτ = −θ²/(n−1) − α  con α = σ² + R ≥ 0 constante.

                Casos:
                  α = 0:
                    θ(τ) = θ₀ / (1 + θ₀ τ/(n−1))
                    τ_c  = −(n−1)/θ₀
                    ∫ θ/(n−1) dτ = ln|1 + θ₀ τ/(n−1)|
                    → area_factor(τ_c) → 0

                  α > 0:
                    Sea β = √((n−1) α),  γ = √(α/(n−1)).
                    θ(τ) = −β cot(φ₀ + γ τ)   si |θ₀| configuración convergente,
                    con φ₀ = arccot(−θ₀/β) ∈ (0, π).
                    Cáustica en φ₀ + γ τ_c = π  ⇒  τ_c = (π − φ₀)/γ.

                Devuelve (τ_c, f_opt, area_collapse_factor).
                """
                n_m1 = float(self._n - 1)

                if alpha < 0.0:
                    # α < 0 violaría SEC+σ²≥0; clip defensivo
                    logger.warning(
                        "α=σ²+R=%.3e < 0; se recorta a 0 (inconsistencia SEC).",
                        alpha,
                    )
                    alpha = 0.0

                if alpha <= _MACHINE_EPSILON:
                    # ── caso α = 0 ──────────────────────────────────────────
                    tau_c = -n_m1 / theta0  # θ₀ < 0 garantizado por caller
                    # area ~ |1 + θ₀ τ / (n−1)| → 0 en τ_c
                    area_factor = 0.0
                    f_opt = tau_c  # escala afín pura
                    return float(tau_c), float(f_opt), float(area_factor)

                # ── caso α > 0 ──────────────────────────────────────────────
                beta = math.sqrt(n_m1 * alpha)          # √((n−1)α)
                gamma = math.sqrt(alpha / n_m1)         # √(α/(n−1))

                # φ₀ = arccot(−θ₀/β) ∈ (0, π)
                # arccot(x) = arctan(1/x) con ajuste de cuadrante
                ratio = -theta0 / beta
                phi0 = math.atan2(1.0, ratio)  # ∈ (0, π) para ratio ∈ ℝ
                if phi0 <= 0.0:
                    phi0 += math.pi

                # Cáustica: φ₀ + γ τ = π  (polo de cot)
                delta_phi = math.pi - phi0
                if delta_phi <= _MACHINE_EPSILON:
                    # Ya estamos en la cáustica (θ₀ → −∞)
                    return 0.0, 0.0, 0.0

                tau_c = delta_phi / gamma

                # Factor de área: para la solución cot,
                # A(τ)/A(0) = |sin(φ₀)/sin(φ₀+γτ)|  → 0 en τ_c.
                # f_opt ∝ τ_c · (amplitud residual); usamos escala afín
                # regularizada por la cota HP.
                area_factor = 0.0  # por definición en la cáustica
                f_opt = tau_c * abs(math.sin(phi0))  # peso por fase inicial

                return float(tau_c), float(f_opt), float(area_factor)

            # ── integración numérica ─────────────────────────────────────────

            def _numerical_caustic(
                self,
                theta0: float,
                sigma_sq: float,
                ricci_term: float,
            ) -> Tuple[float, float, float]:
                r"""
                Integra dθ/dτ con solve_ivp hasta θ ≤ θ_caustic o τ_max.

                Estima ∫ θ/(n−1) dτ por trapecio compuesto sobre la malla
                adaptativa y define:
                  area_factor = exp(∫ θ/(n−1) dτ)
                  f_opt       = τ_c · area_factor   (degenera → 0 en cáustica)

                Devuelve (τ_c, f_opt, area_factor).
                """
                n_m1 = float(self._n - 1)
                alpha = sigma_sq + ricci_term

                def ode(tau: float, y: NDArray[np.float64]) -> NDArray[np.float64]:
                    return np.array(
                        [self.raychaudhuri_rhs(y[0], sigma_sq, ricci_term)],
                        dtype=np.float64,
                    )

                def event_caustic(
                    tau: float, y: NDArray[np.float64]
                ) -> float:
                    # Se dispara cuando θ − θ_caustic = 0 (θ descendente).
                    return float(y[0] - self._caustic_theta)

                event_caustic.terminal = True  # type: ignore[attr-defined]
                event_caustic.direction = -1   # type: ignore[attr-defined]

                # Cota HP como guía de horizonte de integración
                try:
                    tau_hp = self.hawking_penrose_bound(theta0, self._n)
                except FocalDivergenceVetoError:
                    tau_hp = self._max_affine

                t_span_end = min(self._max_affine, max(tau_hp * 2.0, tau_hp + 1.0))

                sol = solve_ivp(
                    ode,
                    [0.0, t_span_end],
                    np.array([theta0], dtype=np.float64),
                    events=event_caustic,
                    rtol=self._rtol,
                    atol=self._atol,
                    dense_output=False,
                    method="RK45",
                )

                if sol.t_events is not None and len(sol.t_events) > 0 and sol.t_events[0].size > 0:
                    tau_c = float(sol.t_events[0][0])
                elif sol.status == 0:
                    # Integración completada sin evento: usar cota HP
                    logger.warning(
                        "Cáustica numérica no alcanzada en τ∈[0,%.2e]; "
                        "se usa cota de Hawking–Penrose.",
                        t_span_end,
                    )
                    tau_c = min(tau_hp, t_span_end)
                else:
                    raise CausticNotReachedError(
                        f"IVP falló (status={sol.status}): {sol.message}"
                    )

                # Integral de θ/(n−1) sobre la malla
                theta_vals = np.asarray(sol.y[0], dtype=np.float64)
                tau_vals = np.asarray(sol.t, dtype=np.float64)

                # Recortar a τ ≤ τ_c
                mask = tau_vals <= tau_c + _MACHINE_EPSILON
                if not np.any(mask):
                    integral_theta = 0.0
                else:
                    t_m = tau_vals[mask]
                    th_m = theta_vals[mask]
                    # np.trapezoid (NumPy ≥ 2.0) con fallback a trapz
                    try:
                        integral_theta = float(np.trapezoid(th_m, t_m))
                    except AttributeError:
                        integral_theta = float(np.trapz(th_m, t_m))  # type: ignore[attr-defined]

                exponent = integral_theta / n_m1
                # Underflow guard
                if exponent < -700.0:
                    area_factor = 0.0
                else:
                    area_factor = math.exp(exponent)

                f_opt = tau_c * area_factor
                return float(tau_c), float(f_opt), float(area_factor)

            # ── método terminal de la Fase 3 ─────────────────────────────────

            def compute_focal_length(
                self,
                energy_cert: EnergyConditionCertificate,
            ) -> FocalLengthResult:
                r"""
                Último método de la Fase 3. Recibe el certificado energético
                de la Fase 2 (con cinemática embebida) y retorna la distancia
                focal óptima, el parámetro afín de cáustica y los flags de
                viabilidad, cerrando la cadena anidada:

                    compute_kinematics → certify_energy → compute_focal_length
                """
                kin = energy_cert.kinematics
                theta0 = kin.expansion_scalar
                sigma_sq = max(0.0, kin.shear_magnitude)
                ricci = max(0.0, energy_cert.ricci_contraction)
                n = kin.spatial_dim

                # Alinear n del certificado con el de la fase
                if n != self._n:
                    logger.warning(
                        "spatial_dim cinemática (%d) ≠ Phase3 (%d); "
                        "se usa n cinemático=%d.",
                        n, self._n, n,
                    )
                    self._n = n

                flags = FocalViabilityFlags.NONE

                # §1 residual: ω² ya validado en Fase 1
                if kin.vorticity_magnitude <= _VORTICITY_TOLERANCE:
                    flags |= FocalViabilityFlags.ZERO_VORTICITY

                # §2 SEC
                if energy_cert.is_strongly_attractive and ricci >= 0.0:
                    flags |= FocalViabilityFlags.STRONG_ENERGY_SATISFIED

                # Expansión inicial convergente
                if theta0 >= -_MACHINE_EPSILON:
                    raise FocalDivergenceVetoError(
                        f"Expansión inicial no negativa: θ₀={theta0:.4e}. "
                        "El haz semántico ya diverge; imposible focalizar."
                    )
                flags |= FocalViabilityFlags.NEGATIVE_EXPANSION

                # dθ/dτ en τ=0 debe ser < 0 (refuerzo de convergencia)
                dtheta0 = self.raychaudhuri_rhs(theta0, sigma_sq, ricci)
                if dtheta0 >= -_MACHINE_EPSILON:
                    # Con θ₀ < 0 y σ²,R ≥ 0, dθ/dτ = −θ²/(n−1)−σ²−R < 0
                    # siempre; si no, hay inconsistencia numérica grave.
                    raise FocalDivergenceVetoError(
                        f"Haz semántico no convergente: dθ/dτ|₀ = {dtheta0:.4e}."
                    )
                flags |= FocalViabilityFlags.CONVERGENT_RAYCHAUDHURI

                # Cota de Hawking–Penrose (siempre disponible bajo las hipótesis)
                tau_hp = self.hawking_penrose_bound(theta0, self._n)

                alpha = sigma_sq + ricci
                method_used = self._method

                if self._method == CausticMethod.NUMERICAL_IVP:
                    tau_c, f_opt, area_factor = self._numerical_caustic(
                        theta0, sigma_sq, ricci
                    )
                elif self._method == CausticMethod.ANALYTICAL_BOUND:
                    tau_c = tau_hp
                    area_factor = 0.0
                    f_opt = tau_hp
                    method_used = CausticMethod.ANALYTICAL_BOUND
                else:
                    # ANALYTICAL_EXACT (default)
                    tau_c, f_opt, area_factor = self._analytical_exact_caustic(
                        theta0, alpha
                    )
                    method_used = CausticMethod.ANALYTICAL_EXACT

                # La cáustica no puede superar la cota HP (teorema)
                if tau_c > tau_hp * (1.0 + 1e-6):
                    logger.warning(
                        "τ_c=%.4e > τ_HP=%.4e; se satura a la cota de "
                        "Hawking–Penrose (teorema de enfoque).",
                        tau_c, tau_hp,
                    )
                    tau_c = tau_hp
                    f_opt = min(f_opt, tau_hp)

                if tau_c >= 0.0 and math.isfinite(tau_c):
                    flags |= FocalViabilityFlags.CAUSTIC_REACHABLE

                # Sanitizar f_opt
                if not math.isfinite(f_opt) or f_opt < 0.0:
                    f_opt = max(0.0, tau_c)

                result = FocalLengthResult(
                    optimal_focal_length=float(f_opt),
                    caustic_affine_parameter=float(tau_c),
                    hawking_penrose_bound=float(tau_hp),
                    initial_dtheta_dtau=float(dtheta0),
                    caustic_method=method_used,
                    viability_flags=flags,
                    area_collapse_factor=float(area_factor),
                )

                logger.info(
                    "Fase 3 OK | τ_c=%.6e | τ_HP=%.6e | f_opt=%.6e | "
                    "dθ/dτ₀=%.4e | method=%s | flags=%s",
                    result.caustic_affine_parameter,
                    result.hawking_penrose_bound,
                    result.optimal_focal_length,
                    result.initial_dtheta_dtau,
                    result.caustic_method.name,
                    result.viability_flags.name,
                )
                return result


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR CATEGÓRICO: Raychaudhuri Focal Fibrator
# ══════════════════════════════════════════════════════════════════════════════
class RaychaudhuriFocalFibrator(Morphism):
    r"""
    Funtor Supremo de Focalización (𝒲_Raychaudhuri).

    Intercepta el vector ψ_focus del `optical_riemann_lens.py` y contrae
    el espacio afín garantizando impacto puntual sobre el
    `semantic_parabolic_mirror.py`.

    Orquesta las tres fases anidadas como composición de morfismos:

        𝒲 = focal_collapse ∘ certify_SEC ∘ kinematics

      1. Phase1_RaychaudhuriKinematics.compute_kinematics
            → KinematicExpansionData
      2. Phase2_StrongEnergyCondition.certify_energy
            → EnergyConditionCertificate
      3. Phase3_FocalCollapse.compute_focal_length
            → FocalLengthResult

    La anidación de clases refleja la dependencia funtorial estricta:
    no existe camino que omita la cinemática ni la SEC antes del colapso.
    """

    def __init__(
        self,
        spatial_dimensions: int = 4,
        metric: Optional[NDArray[np.float64]] = None,
        caustic_method: CausticMethod = CausticMethod.ANALYTICAL_EXACT,
        signature: MetricSignature = MetricSignature.RIEMANNIAN,
        use_numerical_integration: bool = False,
        vorticity_tol: float = _VORTICITY_TOLERANCE,
    ) -> None:
        if spatial_dimensions < _MIN_SPATIAL_DIM:
            raise ValueError(
                f"spatial_dimensions={spatial_dimensions} < n_min={_MIN_SPATIAL_DIM}."
            )
        self.n_dim = int(spatial_dimensions)
        self._signature = signature
        self._vorticity_tol = float(vorticity_tol)

        # Compatibilidad con API v2
        if use_numerical_integration:
            caustic_method = CausticMethod.NUMERICAL_IVP
        self._caustic_method = caustic_method

        # Torre anidada (reutilizable; estado inmutable entre invocaciones
        # salvo los tensores de entrada).
        self._phase1 = Phase1_RaychaudhuriKinematics(
            metric=metric,
            vorticity_tol=self._vorticity_tol,
        )
        self._phase2 = (
            Phase1_RaychaudhuriKinematics.Phase2_StrongEnergyCondition(
                metric=self._phase1._G,
                metric_inv=self._phase1._G_inv,
                signature=self._signature,
            )
        )
        self._phase3 = (
            Phase1_RaychaudhuriKinematics
            .Phase2_StrongEnergyCondition
            .Phase3_FocalCollapse(
                spatial_dim=self.n_dim,
                method=self._caustic_method,
            )
        )

    def execute_focalization(
        self,
        velocity_gradient: NDArray[np.float64],
        shear_tensor: NDArray[np.float64],
        vorticity_tensor: NDArray[np.float64],
        stress_tensor: NDArray[np.float64],
        u_vector: NDArray[np.float64],
    ) -> FocalLengthResult:
        r"""
        Ejecución canónica del morfismo categórico 𝒲_Raychaudhuri.

        Aplica la composición:

            (B,σ,ω)  ──kinematics──▶  KinematicExpansionData
            (𝒯,u)    ──certify_SEC─▶  EnergyConditionCertificate
                     ──collapse────▶  FocalLengthResult
        """
        logger.debug(
            "Iniciando Fibración Focal de Raychaudhuri | n=%d | method=%s",
            self.n_dim, self._caustic_method.name,
        )

        # 1. Cinemática espectral de la congruencia
        kinematics = self._phase1.compute_kinematics(
            velocity_gradient, shear_tensor, vorticity_tensor
        )

        # 2. Certificación SEC + Ricci
        energy_cert = self._phase2.certify_energy(
            kinematics, stress_tensor, u_vector
        )

        # 3. Colapso focal / cáustica
        focal_result = self._phase3.compute_focal_length(energy_cert)

        if not focal_result.viability_flags.is_order_unit():
            raise RaychaudhuriFibratorError(
                f"El haz semántico no superó la retícula de viabilidad focal "
                f"(flags={focal_result.viability_flags!r})."
            )

        logger.info(
            "Distancia focal colapsada: f_opt=%.6e | τ_c=%.6e | τ_HP=%.6e",
            focal_result.optimal_focal_length,
            focal_result.caustic_affine_parameter,
            focal_result.hawking_penrose_bound,
        )
        return focal_result

    def __call__(
        self,
        velocity_gradient: NDArray[np.float64],
        shear_tensor: NDArray[np.float64],
        vorticity_tensor: NDArray[np.float64],
        stress_tensor: NDArray[np.float64],
        u_vector: NDArray[np.float64],
    ) -> FocalLengthResult:
        """Compatibilidad con interfaz Morphism / API v2."""
        return self.execute_focalization(
            velocity_gradient,
            shear_tensor,
            vorticity_tensor,
            stress_tensor,
            u_vector,
        )

    def describe_pipeline(self) -> Dict[str, str]:
        """Metadatos del pipeline anidado (introspección categórica ligera)."""
        return {
            "functor": "𝒲_Raychaudhuri (RaychaudhuriFocalFibrator)",
            "version": "3.0.0-Nested-Caustic-Spectral-Topos",
            "phase_1": "Phase1_RaychaudhuriKinematics.compute_kinematics",
            "phase_2": "Phase2_StrongEnergyCondition.certify_energy",
            "phase_3": "Phase3_FocalCollapse.compute_focal_length",
            "axiom_1": "ω_{μν} ≡ 0  (vorticidad nula, residual relativo)",
            "axiom_2": "SEC: (𝒯 − ½𝒯 G)(u,u) ≥ 0  ⇒  R(u,u) ≥ 0",
            "axiom_3": "τ_c ≤ (n−1)/|θ₀|  (Hawking–Penrose) + f_opt",
            "caustic_method": self._caustic_method.name,
            "signature": self._signature.name,
            "spatial_dim": str(self.n_dim),
        }


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    # excepciones
    "RaychaudhuriFibratorError",
    "FocalDivergenceVetoError",
    "StrongEnergyViolationError",
    "VorticityAnomalyError",
    "ShearAnomalyError",
    "MetricSignatureError",
    "DimensionMismatchError",
    "NormalizationError",
    "CausticNotReachedError",
    # enums / flags
    "FocalViabilityFlags",
    "CausticMethod",
    "MetricSignature",
    # DTOs
    "KinematicExpansionData",
    "EnergyConditionCertificate",
    "FocalLengthResult",
    # fases + orquestador
    "Phase1_RaychaudhuriKinematics",
    "RaychaudhuriFocalFibrator",
]