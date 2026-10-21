# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Riemannian Metric Tensors v6.0.0                                    ║
║          Tejido Conectivo Anisotrópico + RSI³ + Mecánica de Poincaré         ║
║ Ruta   : app/core/immune_system/metric_tensors.py                            ║
║ Evolución: v5.0.0 → v6.0.0 (1 ⊂ 2 ⊂ 3, CPS, KAM orbital, sin monkey-patch)   ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GEOMETRÍA ANISOTRÓPICA (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra la definición, validación espectral, endurecimiento
dinámico y precompilación de los tensores métricos de-confinados G_{μν}.
Los tensores actúan como métrica Riemanniana basal del hiperespacio de fase
M ⊂ ℝ⁷. Toda "distancia de decisión" (Mahalanobis), "fricción logística",
"disipación de exergía" y "energía de Dirichlet" depende de forma unívoca
de este campo métrico.

Sobre los tensores se impone, en tiempo de carga o bajo demanda
(`MetricTensorFactory.harden_tensor`), el siguiente teorema triple:

  (T1) Punto fijo geométrico: defecto de Einstein
       $$\delta_{\mathrm{Einstein}}(G)=\frac{\|\operatorname{Ric}-\frac{R}{n}G\|_F}{\|G\|_F}\to 0^+$$

  (T2) Automejora recursiva de nivel 3 (RSI³) en un álgebra de Banach:
       $$\Phi^{(1)}_{\Delta\tau}(G)=\Pi_{\mathrm{SPD}}(G-2\Delta\tau\,\mathrm{Ric}^\circ(G))$$
       $$\Phi^{(2)}_{\eta}=\Phi^{(1)}_{\Delta\tau\exp(-\eta|\nabla R|)}$$
       $$\Phi^{(3)}_{\zeta}:\eta\mapsto\eta\exp(\zeta(L_\Phi-L_\star)),\;
         \zeta\mapsto\zeta\cdot\sigma(\mathrm{osc}\,\eta)$$
       con contracción de Banach $L_\Phi<1$ (punto fijo único).

  (T3) Estabilidad de Poincaré–KAM sobre la órbita $\{G_k\}_{k=0}^N$:
       $$\lambda_{\max}=\lim_{N\to\infty}\tfrac{1}{N}\sum_k\log
         \frac{\|G_{k+1}-G_k\|_F}{\|G_k-G_{k-1}\|_F}$$
       $$|\omega\cdot k|\ge\gamma/|k|^\tau,\quad
         \omega=\mathrm{FFT}(R(\tau))\ \text{(ángulo-acción)}$$

AXIOMÁTICA RIEMANNIANA (preservada íntegra):
  [I1] Simetría bilateral G = Gᵀ
  [I2] Positividad definida G ≻ 0 (signatura de Sylvester estricta)
  [I3] Acotación espectral κ₂(G) ≤ 10⁸ (estabilidad Wilkinson)
  [I4] Isomorfismos musicales ♭/♯ (Riesz)
  [I5] Clausura topológica de Kuratowski

RESOLUCIÓN DE LA CONTRADICCIÓN Cholesky:
  La descomposición de Cholesky se emplea como certificación REDUNDANTE
  (confirmación independiente) DESPUÉS de la certificación espectral exacta.
  NO se admite Cholesky como certificación primaria (inestable ante matrices
  casi singulares), pero SÍ como verificación cruzada (defensa en profundidad).
  La inversa SPD se calcula SIEMPRE vía Cholesky (estabilidad numérica).

Tejido anidado (continuación estricta):
  FASE 1.Z `_handoff_to_phase2`  ──▶  FASE 2.A `_ingest_phase1_handoff`
  FASE 2.Z `_handoff_to_phase3`  ──▶  FASE 3.A `_ingest_phase2_handoff`
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations, product
from typing import Any, Dict, Final, List, NamedTuple, Optional, Tuple

import numpy as np
from numpy.linalg import LinAlgError
from scipy.linalg import block_diag, cho_factor, cho_solve, eigh

from app.core.immune_system.topological_watcher import (
    COND_NUM_TOL,
    MIN_EIGVAL_TOL,
    MetricTensorError,
)

logger = logging.getLogger("MIC.ImmuneSystem.MetricTensors")

__all__ = [
    "G_PHYSICS",
    "G_TOPOLOGY",
    "G_THERMODYNAMICS",
    "HeytingVerdict",
    "MetricTensorFactory",
    "SpectralProfile",
    "RegularizationReport",
    "RSI3Trace",
    "PoincareData",
    "HardenedCertificate",
    "Phase1Handoff",
    "Phase2Handoff",
    "get_tensor_diagnostics",
    "get_hardening_diagnostics",
]

__version__: Final[str] = "6.0.0"

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FUNDAMENTALES CON JUSTIFICACIÓN MATEMÁTICA
# ═══════════════════════════════════════════════════════════════════════════════

_FLOAT_DTYPE: Final[np.dtype] = np.dtype(np.float64)
_WILKINSON_EPS: Final[float] = float(np.finfo(np.float64).eps)

_SYMMETRY_RTOL: Final[float] = 1e-14
_SYMMETRY_ATOL_BASE: Final[float] = 1e-15
_SPD_INTERIOR_FACTOR: Final[float] = 1.1
_REGULARIZATION_WARN_THRESHOLD: Final[float] = 0.05
_REGULARIZATION_ABORT_THRESHOLD: Final[float] = 0.25
_TIKHONOV_DELTA_ATOL: Final[float] = 1e-16
_NEAR_ZERO_FROBENIUS_TOL: Final[float] = 1e-14
_GERSHGORIN_MARGIN: Final[float] = 1.0e-6

# ── Constantes RSI³ + Poincaré ──────────────────────────────────────────────
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_BANACH_TARGET: Final[float] = 0.95
_RSI3_CONV_EPS: Final[float] = 1.0e-9
_RSI3_MAX_ITER: Final[int] = 40
_RSI3_MIN_ITER: Final[int] = 8
_EINSTEIN_TOL: Final[float] = 1.0e-3
_LYAPUNOV_CHAOS: Final[float] = 1.0e-2
_KAM_GAMMA: Final[float] = 1.0e-3
_KAM_TAU: Final[float] = 2.0
_POINCARE_RETURN_TOL: Final[float] = 1.0e-4
_TAU_LOWER: Final[float] = 1.0e-5
_TAU_UPPER: Final[float] = 0.1
_ETA_LOWER: Final[float] = 1.0e-4
_ETA_UPPER: Final[float] = 10.0
_ZETA_LOWER: Final[float] = 1.0e-4
_ZETA_UPPER: Final[float] = 0.1
_FLOQUET_UNSTABLE: Final[float] = 1.0 + 1.0e-2
_LIPSCHITZ_PERT_REL: Final[float] = 1.0e-6
_LIPSCHITZ_PROBE_STRIDE: Final[int] = 5
_COMPAT_REL_TOL: Final[float] = 1.0e-4


# ═══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS DE DATOS (v4.1.0 / v5.0.0 preservadas + v6.0.0)
# ═══════════════════════════════════════════════════════════════════════════════

class HeytingVerdict(Enum):
    r"""Clasificador en el álgebra de Heyting trivalente $\Omega_3$.

    Orden de obstrucción: $\mathrm{COHERENT}=\bot < \mathrm{DEGRADED}=\tfrac12
    < \mathrm{VETOED}=\top$. El join $a\vee b=\max$ acumula obstrucciones.
    """

    COHERENT = "COHERENT"
    DEGRADED = "DEGRADED"
    VETOED = "VETOED"

    @property
    def omega(self) -> float:
        return {
            HeytingVerdict.COHERENT: 0.0,
            HeytingVerdict.DEGRADED: 0.5,
            HeytingVerdict.VETOED: 1.0,
        }[self]

    @classmethod
    def from_omega(cls, value: float) -> "HeytingVerdict":
        if value <= 0.0:
            return cls.COHERENT
        if value < 1.0:
            return cls.DEGRADED
        return cls.VETOED

    def join(self, other: "HeytingVerdict") -> "HeytingVerdict":
        return HeytingVerdict.from_omega(max(self.omega, other.omega))


class SpectralProfile(NamedTuple):
    """Perfil espectral completo de una matriz simétrica (v4.1.0 preservado)."""

    eigenvalues: np.ndarray
    lambda_min: float
    lambda_max: float
    condition_number: float
    spectral_gap: float
    frobenius_norm: float

    def is_strictly_positive(self, threshold: float = MIN_EIGVAL_TOL) -> bool:
        return bool(self.lambda_min >= threshold)

    def is_well_conditioned(self, max_kappa: float = COND_NUM_TOL) -> bool:
        return bool(np.isfinite(self.condition_number) and self.condition_number <= max_kappa)


@dataclass(frozen=True, slots=True)
class RegularizationReport:
    """Informe de regularización (v4.1.0 preservado)."""

    tensor_name: str
    original_profile: SpectralProfile
    regularized_profile: SpectralProfile
    tikhonov_delta: float
    frobenius_deformation: float
    required_regularization: bool
    input_asymmetry: float

    def log_summary(self) -> None:
        if self.required_regularization:
            log_fn = (
                logger.warning
                if self.frobenius_deformation > _REGULARIZATION_WARN_THRESHOLD
                else logger.info
            )
            log_fn(
                "Regularización aplicada a %s: δ=%.6e, deformación=%.4f%%, "
                "λ_min: %.6e → %.6e, κ: %.4e → %.4e, asimetría entrada: %.4e",
                self.tensor_name,
                self.tikhonov_delta,
                self.frobenius_deformation * 100,
                self.original_profile.lambda_min,
                self.regularized_profile.lambda_min,
                self.original_profile.condition_number,
                self.regularized_profile.condition_number,
                self.input_asymmetry,
            )
        else:
            logger.debug(
                "Tensor %s validado sin regularización: λ_min=%.6e, κ=%.4e, asimetría: %.4e",
                self.tensor_name,
                self.regularized_profile.lambda_min,
                self.regularized_profile.condition_number,
                self.input_asymmetry,
            )


@dataclass(frozen=True, slots=True)
class RSI3Trace:
    r"""Traza inmutable de la ejecución RSI³."""

    rsi_level_achieved: int
    banach_L: float
    delta_tau_history: Tuple[float, ...]
    eta_history: Tuple[float, ...]
    einstein_defect_history: Tuple[float, ...]
    converged: bool


@dataclass(frozen=True, slots=True)
class PoincareData:
    r"""Datos de mecánica celeste de Poincaré sobre la órbita del flujo."""

    return_periods: Tuple[float, ...]
    lyapunov_max: float
    kam_stable: bool
    homoclinic_tangle: bool
    hopf_index: float
    diophantine_gamma: float
    floquet_multiplier: float = 0.0
    dominant_frequencies: Tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class HardenedCertificate:
    r"""Certificado final de endurecimiento del tensor."""

    tensor_name: str
    spectral_profile: SpectralProfile
    rsi3_trace: RSI3Trace
    poincare_data: PoincareData
    einstein_defect: float
    lipschitz_connes_bound: float
    is_hardened: bool
    is_geometrically_stable: bool
    heyting_verdict: HeytingVerdict = HeytingVerdict.COHERENT
    log_volume: float = 0.0
    metric_compatibility_residual: float = 0.0
    used_phase1_fallback: bool = False


@dataclass(frozen=True, slots=True)
class Phase1Handoff:
    """Paquete de transferencia FASE 1 → FASE 2 (morfismo de continuación)."""

    tensor_name: str
    candidate: np.ndarray
    spectral_profile: SpectralProfile
    einstein_defect: float
    lipschitz_connes_bound: float
    input_asymmetry: float
    regularization_report: RegularizationReport
    log_volume: float = 0.0
    compatibility_residual: float = 0.0


@dataclass(frozen=True)
class Phase2Handoff:
    """Paquete de transferencia FASE 2 → FASE 3 (morfismo de continuación)."""

    phase1: Phase1Handoff
    orbit: Tuple[np.ndarray, ...]
    curvature_history: Tuple[float, ...]
    lyapunov_max: float
    poincare_data: Dict[str, Any]
    banach_L: float
    delta_tau_history: Tuple[float, ...]
    eta_history: Tuple[float, ...]
    einstein_defect_history: Tuple[float, ...]
    rsi3_level: int
    rsi3_converged: bool
    dominant_frequencies: Tuple[float, ...] = field(default_factory=tuple)


def _fro(matrix: np.ndarray) -> float:
    """Norma de Frobenius escalar segura."""
    return float(np.linalg.norm(np.asarray(matrix, dtype=_FLOAT_DTYPE), ord="fro"))


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1 — GEOMETRÍA ESPECTRAL, EINSTEIN DEFECT, CHRISTOFFEL Y HANDOFF
# ═══════════════════════════════════════════════════════════════════════════════
class MetricTensorFactory:
    r"""
    Fábrica determinista de tensores métricos con endurecimiento RSI³+Poincaré
    opcional. Cada tensor construido satisface rigurosamente:

    1. G ∈ ℝⁿˣⁿ (matriz cuadrada)
    2. G = Gᵀ (simetría exacta por proyección ortogonal)
    3. G ≻ 0 (definida positiva, certificada espectralmente y por Cholesky)
    4. λ_min(G) ≥ MIN_EIGVAL_TOL
    5. κ₂(G) ≤ COND_NUM_TOL
    6. ||G_reg − G_orig||_F / ||G_orig||_F ≤ 25 %
    7. Consistencia espectral post-Tikhonov
    8. [v5.0.0] Estabilidad KAM y ausencia de tangle homoclínico
    9. [v5.0.0] Defecto de Einstein acotado
    10. [v6.0.0] Compatibilidad métrica $\nabla G\approx 0$, KAM orbital (FFT),
        Floquet interpolado, fallback SPD si el flujo degrada la certificación.
    """

    _EXPECTED_DIMENSIONS: Final[Dict[str, int]] = {
        "G_phys": 3,
        "G_topo": 2,
        "G_thermo": 2,
        "G_global": 7,
    }

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 1.0 — Conversión, Cholesky SPD, volumen logarítmico
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _as_strict_array(G: np.ndarray) -> np.ndarray:
        return np.array(G, dtype=_FLOAT_DTYPE, copy=True)

    @staticmethod
    def _spd_inverse(G: np.ndarray) -> np.ndarray:
        r"""Inversa SPD vía Cholesky: $G^{-1}=(LL^\top)^{-1}$."""
        n = int(G.shape[0])
        shifted = np.asarray(G, dtype=_FLOAT_DTYPE) + _SPECTRAL_PSD_FLOOR * np.eye(
            n, dtype=_FLOAT_DTYPE
        )
        chol, lower = cho_factor(shifted, lower=True, check_finite=False)
        return cho_solve(
            (chol, lower), np.eye(n, dtype=_FLOAT_DTYPE), check_finite=False
        )

    @staticmethod
    def _spd_logdet(G: np.ndarray) -> float:
        r"""$\log\det G=2\sum_i\log L_{ii}$ (Cholesky)."""
        n = int(G.shape[0])
        shifted = np.asarray(G, dtype=_FLOAT_DTYPE) + _SPECTRAL_PSD_FLOOR * np.eye(
            n, dtype=_FLOAT_DTYPE
        )
        chol, _ = cho_factor(shifted, lower=True, check_finite=False)
        diag = np.abs(np.diag(chol))
        return float(2.0 * np.sum(np.log(np.maximum(diag, _WILKINSON_EPS))))

    @classmethod
    def _validate_structural_preconditions(cls, name: str, G: np.ndarray) -> np.ndarray:
        if G is None:
            raise MetricTensorError(f"Tensor {name}: entrada NoneType.")
        try:
            G_arr = cls._as_strict_array(G)
        except (TypeError, ValueError) as exc:
            raise MetricTensorError(
                f"Tensor {name}: no convertible a float64. Error: {exc}"
            ) from exc

        if G_arr.ndim != 2:
            raise MetricTensorError(f"Tensor {name}: ndim={G_arr.ndim} ≠ 2.")
        n_rows, n_cols = G_arr.shape
        if n_rows != n_cols:
            raise MetricTensorError(f"Tensor {name}: no cuadrada {n_rows}×{n_cols}.")
        if n_rows == 0:
            raise MetricTensorError(f"Tensor {name}: matriz vacía.")
        expected = cls._EXPECTED_DIMENSIONS.get(name)
        if expected is not None and n_rows != expected:
            raise MetricTensorError(
                f"Tensor {name}: dim esperada {expected}, recibida {n_rows}."
            )
        if not np.all(np.isfinite(G_arr)):
            n_nf = int(np.count_nonzero(~np.isfinite(G_arr)))
            idx = list(zip(*np.where(~np.isfinite(G_arr))))[:5]
            raise MetricTensorError(
                f"Tensor {name}: {n_nf} entradas no finitas. Índices: {idx}."
            )
        return G_arr

    @classmethod
    def _compute_symmetry_tolerance(cls, G: np.ndarray) -> float:
        return max(_SYMMETRY_ATOL_BASE, _SYMMETRY_RTOL * _fro(G))

    @classmethod
    def _assess_input_symmetry(cls, name: str, G: np.ndarray) -> float:
        f_norm = _fro(G)
        asym = _fro(G - G.T)
        rel = asym if f_norm < _NEAR_ZERO_FROBENIUS_TOL else asym / f_norm
        if asym > cls._compute_symmetry_tolerance(G):
            logger.warning(
                "Tensor %s: asimetría entrada. ||G-Gᵀ||_F=%.6e, relativa=%.6e",
                name,
                asym,
                rel,
            )
        return rel

    @staticmethod
    def _symmetrize_by_projection(G: np.ndarray) -> np.ndarray:
        return 0.5 * (G + G.T)

    @staticmethod
    def _project_psd(G: np.ndarray) -> np.ndarray:
        r"""Proyección espectral de Higham $\Pi_{\mathrm{SPD}}$ con re-simetrización."""
        G_sym = 0.5 * (np.asarray(G, dtype=_FLOAT_DTYPE) + np.asarray(G, dtype=_FLOAT_DTYPE).T)
        try:
            eigvals, eigvecs = eigh(G_sym, check_finite=False)
        except LinAlgError:
            return G_sym
        clipped = np.maximum(eigvals, _SPECTRAL_PSD_FLOOR)
        G_proj = (eigvecs * clipped) @ eigvecs.T
        return 0.5 * (G_proj + G_proj.T)

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 1.B — Análisis espectral canónico
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _compute_spectral_profile(cls, G: np.ndarray) -> SpectralProfile:
        tol = cls._compute_symmetry_tolerance(G)
        asym = _fro(G - G.T)
        if asym > tol:
            raise MetricTensorError(
                f"_compute_spectral_profile: matriz no simétrica "
                f"({asym:.3e} > tol={tol:.3e})."
            )
        eigenvalues = np.linalg.eigvalsh(G)
        eigenvalues_immutable = np.array(eigenvalues, dtype=_FLOAT_DTYPE, copy=True)
        eigenvalues_immutable.setflags(write=False)

        lam_min = float(eigenvalues[0])
        lam_max = float(eigenvalues[-1])
        if lam_min > 0.0:
            cond = lam_max / lam_min
        elif lam_min == 0.0:
            cond = np.inf
        else:
            cond = np.nan

        n = len(eigenvalues)
        gap = (
            float(eigenvalues[1] - eigenvalues[0]) / lam_max
            if n > 1 and lam_max > 0.0
            else 0.0
        )

        return SpectralProfile(
            eigenvalues=eigenvalues_immutable,
            lambda_min=lam_min,
            lambda_max=lam_max,
            condition_number=cond,
            spectral_gap=gap,
            frobenius_norm=_fro(G),
        )

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 1.C — Christoffel sintetizado por conmutador de Connes + Ricci
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _connes_metric_derivatives(G: np.ndarray) -> np.ndarray:
        r"""
        Derivada no conmutativa $\partial_\mu G_{\nu\sigma}=[X_\mu,G]_{\nu\sigma}$.

        En base canónica: $[E_{\mu\mu},G]=e_\mu\otimes G_{\mu\cdot}-G_{\cdot\mu}\otimes e_\mu$.
        Vectorizado por rebanadas (sin producto cartesiano sobre $\nu,\sigma$).
        """
        n = int(G.shape[0])
        dG = np.zeros((n, n, n), dtype=_FLOAT_DTYPE)
        for mu in range(n):
            dG[mu, mu, :] += G[mu, :]
            dG[mu, :, mu] -= G[:, mu]
        return dG

    @classmethod
    def _synthesize_christoffel_from_metric(cls, G: np.ndarray) -> np.ndarray:
        r"""
        Construye $\Gamma^{\rho}_{\mu\nu}$ desde $G$ con derivadas no
        conmutativas $[X_\mu,G]$ ($X_\mu=|e_\mu\rangle\langle e_\mu|$), fórmula
        de Koszul, y simetriza en $\mu\leftrightarrow\nu$ (torsión nula).
        """
        G_inv = cls._spd_inverse(G)
        dG = cls._connes_metric_derivatives(G)
        term1 = np.einsum("rs,mns->rmn", G_inv, dG)  # ∂_μ G_νσ
        term2 = np.einsum("rs,nms->rmn", G_inv, dG)  # ∂_ν G_μσ
        term3 = np.einsum("rs,smn->rmn", G_inv, dG)  # ∂_σ G_μν
        gamma = 0.5 * (term1 + term2 - term3)
        return 0.5 * (gamma + np.transpose(gamma, (0, 2, 1)))

    @classmethod
    def _metric_compatibility_residual(
        cls,
        G: np.ndarray,
        christoffel: Optional[np.ndarray] = None,
        dG: Optional[np.ndarray] = None,
    ) -> float:
        r"""
        Residuo de $\nabla G=0$:
        $(\nabla_\rho G)_{\mu\nu}=\partial_\rho G_{\mu\nu}
         -\Gamma^\lambda_{\rho\mu}G_{\lambda\nu}-\Gamma^\lambda_{\rho\nu}G_{\mu\lambda}$.
        """
        if christoffel is None:
            christoffel = cls._synthesize_christoffel_from_metric(G)
        if dG is None:
            dG = cls._connes_metric_derivatives(G)
        term_a = np.einsum("lrm,ln->rmn", christoffel, G)
        term_b = np.einsum("lrn,ml->rmn", christoffel, G)
        return _fro(dG - term_a - term_b)

    @classmethod
    def _compute_ricci_tensor(cls, G: np.ndarray) -> Tuple[np.ndarray, float]:
        r"""
        $\operatorname{Ric}_{\mu\nu}=\Gamma^{\rho}_{\mu\nu}\Gamma^{\sigma}_{\rho\sigma}
        -\Gamma^{\sigma}_{\mu\rho}\Gamma^{\rho}_{\nu\sigma}$,
        escalar $R=\mathrm{tr}(G^{-1}\mathrm{Ric})$ (0-jet, carta normal discreta).
        """
        gamma = cls._synthesize_christoffel_from_metric(G)
        trace_gamma = np.einsum("srs->r", gamma)
        quad1 = np.einsum("rmn,r->mn", gamma, trace_gamma)
        quad2 = np.einsum("smp,pns->mn", gamma, gamma)
        ricci = quad1 - quad2
        ricci = 0.5 * (ricci + ricci.T)
        G_inv = cls._spd_inverse(G)
        scalar_r = float(np.trace(G_inv @ ricci))
        return ricci, scalar_r

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 1.D — Defecto de Einstein y cota de Connes–Daleckii–Krein
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _compute_einstein_defect(cls, G: np.ndarray) -> float:
        r"""
        $$\delta_{\mathrm{Einstein}}(G)=\frac{\|\operatorname{Ric}-(R/n)G\|_F}{\|G\|_F}$$
        Mide la distancia al punto fijo geométrico (métrica de Einstein).
        """
        n = int(G.shape[0])
        ricci, scalar_r = cls._compute_ricci_tensor(G)
        target = ricci - (scalar_r / float(n)) * G
        denom = max(_fro(G), _NEAR_ZERO_FROBENIUS_TOL)
        return float(_fro(target) / denom)

    @classmethod
    def _compute_lipschitz_connes_bound(cls, G: np.ndarray) -> float:
        r"""Cota $L_{\max}\le 1/(2\lambda_{\min}^{3/2})$ sobre $D=\boldsymbol\rho^{-1/2}$."""
        lam_min = float(
            max(np.linalg.eigvalsh(0.5 * (G + G.T))[0], _SPECTRAL_PSD_FLOOR)
        )
        return float(0.5 / (lam_min ** 1.5))

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 1.E — SPD estricta y control de condicionamiento (doble vía)
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _verify_spd_by_cholesky(cls, tensor: np.ndarray) -> bool:
        try:
            np.linalg.cholesky(tensor)
            return True
        except LinAlgError:
            return False

    @classmethod
    def _assert_spd_strict(
        cls, name: str, G: np.ndarray, profile: SpectralProfile
    ) -> None:
        r"""
        Certificación SPD por doble vía:
          (a) Primaria — espectral: λ_min ≥ MIN_EIGVAL_TOL y λ_max > 0.
          (b) Redundante — Cholesky: confirmación independiente.

        La contradicción documental del v4.1.0 se resuelve: Cholesky NO se
        rechaza, se EXIGE como defensa en profundidad; pero la certificación
        PRIMARIA es espectral (la única robusta ante casi-singularidad).
        """
        if not np.all(np.isfinite(G)):
            raise MetricTensorError(
                f"Tensor {name}: valores no finitos post-regularización."
            )
        tol = cls._compute_symmetry_tolerance(G)
        if _fro(G - G.T) > tol:
            raise MetricTensorError(f"Tensor {name}: asimetría residual.")
        if profile.lambda_max <= 0.0:
            raise MetricTensorError(
                f"Tensor {name}: λ_max={profile.lambda_max:.6e} ≤ 0."
            )
        if profile.lambda_min <= MIN_EIGVAL_TOL:
            raise MetricTensorError(
                f"Tensor {name}: λ_min={profile.lambda_min:.6e} ≤ {MIN_EIGVAL_TOL:.3e}."
            )
        if not cls._verify_spd_by_cholesky(G):
            raise MetricTensorError(
                f"Tensor {name}: falló Cholesky (certificación redundante). "
                "Discrepancia espectral/Cholesky ⇒ bug numérico o patología."
            )

    @classmethod
    def _assert_well_conditioned(cls, name: str, profile: SpectralProfile) -> None:
        kappa = profile.condition_number
        if not np.isfinite(kappa):
            raise MetricTensorError(f"Tensor {name}: κ no finito.")
        if kappa > COND_NUM_TOL:
            raise MetricTensorError(
                f"Tensor {name}: κ₂={kappa:.4e} > {COND_NUM_TOL:.4e}."
            )

    @classmethod
    def _try_certify_spd(
        cls, name: str, G: np.ndarray
    ) -> Optional[Tuple[np.ndarray, SpectralProfile]]:
        """Certifica SPD+κ sin lanzar; `None` si falla."""
        try:
            G_sym = cls._symmetrize_by_projection(np.asarray(G, dtype=_FLOAT_DTYPE))
            profile = cls._compute_spectral_profile(G_sym)
            cls._assert_spd_strict(name, G_sym, profile)
            cls._assert_well_conditioned(name, profile)
            return G_sym, profile
        except (LinAlgError, MetricTensorError):
            return None

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 1.F — Tikhonov, Gershgorin y deformación
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _apply_tikhonov_regularization(
        cls, name: str, G: np.ndarray, original_profile: SpectralProfile
    ) -> Tuple[np.ndarray, float]:
        lam_min = original_profile.lambda_min
        target = MIN_EIGVAL_TOL * _SPD_INTERIOR_FACTOR
        if lam_min >= target:
            return G, 0.0
        delta = target - lam_min
        n = int(G.shape[0])
        return G + delta * np.eye(n, dtype=_FLOAT_DTYPE), float(delta)

    @classmethod
    def _compute_regularization_deformation(
        cls, G_orig: np.ndarray, G_reg: np.ndarray, delta: float
    ) -> float:
        orig_norm = _fro(G_orig)
        if orig_norm < _NEAR_ZERO_FROBENIUS_TOL:
            raise MetricTensorError(
                f"Deformación relativa no definida: ||G||_F={orig_norm:.3e}≈0."
            )
        deform = _fro(G_reg - G_orig)
        n = int(G_orig.shape[0])
        expected = delta * np.sqrt(n)
        rel_disc = abs(deform - expected)
        tol = max(1e-12, 1e-10 * expected)
        if rel_disc > tol:
            logger.warning(
                "Inconsistencia Tikhonov: numérica=%.3e, analítica=%.3e",
                deform,
                expected,
            )
        return float(deform / orig_norm)

    @classmethod
    def _verify_spectral_consistency(
        cls,
        name: str,
        orig: SpectralProfile,
        reg: SpectralProfile,
        delta: float,
    ) -> None:
        if delta < _TIKHONOV_DELTA_ATOL:
            return
        expected = orig.eigenvalues + delta
        actual = reg.eigenvalues
        scale = max(
            float(np.max(np.abs(expected))),
            float(np.max(np.abs(actual))),
            1e-15,
        )
        tol = scale * 1e-12
        disc = float(np.max(np.abs(actual - expected)))
        if disc > tol:
            logger.warning(
                "Tensor %s: inconsistencia espectral post-Tikhonov (disc=%.3e).",
                name,
                disc,
            )

    @classmethod
    def _enforce_gershgorin_spd_margin(
        cls, G: np.ndarray, margin: float = _GERSHGORIN_MARGIN
    ) -> np.ndarray:
        r"""
        Levantamiento diagonal mínimo que restaura dominancia de Gershgorin:

            $G_{ii}\leftarrow G_{ii}+\max(0,\sum_{j\neq i}|G_{ij}|+\varepsilon-G_{ii})$.

        Garantiza un disco de Gershgorin contenido en $\{\Re\lambda>0\}$.
        """
        G_out = np.array(G, dtype=_FLOAT_DTYPE, copy=True)
        diag = np.diag(G_out).copy()
        off = np.sum(np.abs(G_out), axis=1) - np.abs(diag)
        lift = np.maximum(off + float(margin) - diag, 0.0)
        if np.any(lift > 0.0):
            G_out = G_out + np.diag(lift)
            logger.info(
                "Gershgorin: levantamiento diagonal en filas %s (‖lift‖_∞=%.3e).",
                np.where(lift > 0.0)[0].tolist(),
                float(np.max(lift)),
            )
        return G_out

    @classmethod
    def _make_immutable(cls, G: np.ndarray) -> np.ndarray:
        G_imm = np.array(G, dtype=_FLOAT_DTYPE, copy=True)
        G_imm.setflags(write=False)
        return G_imm

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 1.G — Pipeline canónico de certificación (sustrato de FASE 1.Z)
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _certify_canonical(cls, name: str, G: np.ndarray) -> Phase1Handoff:
        """Valida, simetriza, Tikhonov, SPD dual y empaqueta el 0-jet geométrico."""
        G_validated = cls._validate_structural_preconditions(name, G)
        input_asym = cls._assess_input_symmetry(name, G_validated)
        G_sym = cls._symmetrize_by_projection(G_validated)

        if not np.all(np.isfinite(G_sym)):
            raise MetricTensorError(f"Tensor {name}: no finitos post-simetrización.")

        try:
            original_profile = cls._compute_spectral_profile(G_sym)
        except (LinAlgError, MetricTensorError) as exc:
            raise MetricTensorError(
                f"Tensor {name}: análisis espectral inicial fallido: {exc}"
            ) from exc

        G_reg, delta = cls._apply_tikhonov_regularization(name, G_sym, original_profile)
        required_reg = delta > _TIKHONOV_DELTA_ATOL

        if required_reg:
            deformation = cls._compute_regularization_deformation(G_sym, G_reg, delta)
            if deformation > _REGULARIZATION_ABORT_THRESHOLD:
                raise MetricTensorError(
                    f"Tensor {name}: deformación {deformation:.2%} > "
                    f"{_REGULARIZATION_ABORT_THRESHOLD:.0%}."
                )
        else:
            deformation = 0.0

        try:
            reg_profile = cls._compute_spectral_profile(G_reg)
        except (LinAlgError, MetricTensorError) as exc:
            raise MetricTensorError(
                f"Tensor {name}: análisis espectral post-reg fallido: {exc}"
            ) from exc

        cls._verify_spectral_consistency(name, original_profile, reg_profile, delta)
        cls._assert_spd_strict(name, G_reg, reg_profile)
        cls._assert_well_conditioned(name, reg_profile)

        report = RegularizationReport(
            tensor_name=name,
            original_profile=original_profile,
            regularized_profile=reg_profile,
            tikhonov_delta=delta,
            frobenius_deformation=deformation,
            required_regularization=required_reg,
            input_asymmetry=input_asym,
        )
        report.log_summary()

        einstein_defect = cls._compute_einstein_defect(G_reg)
        lip = cls._compute_lipschitz_connes_bound(G_reg)
        try:
            log_volume = cls._spd_logdet(G_reg)
        except LinAlgError:
            log_volume = float("-inf")
        try:
            compat = cls._metric_compatibility_residual(G_reg)
        except LinAlgError:
            compat = float("inf")

        return Phase1Handoff(
            tensor_name=name,
            candidate=np.array(G_reg, dtype=_FLOAT_DTYPE, copy=True),
            spectral_profile=reg_profile,
            einstein_defect=float(einstein_defect),
            lipschitz_connes_bound=float(lip),
            input_asymmetry=float(input_asym),
            regularization_report=report,
            log_volume=float(log_volume),
            compatibility_residual=float(compat),
        )

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 1.Z — HANDOFF: última definición de FASE 1 ≡ inicio de FASE 2
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _handoff_to_phase2(cls, name: str, G: np.ndarray) -> Phase1Handoff:
        r"""
        Última definición formal de FASE 1 y **continuación estricta** de FASE 2.

        Ejecuta la validación canónica y construye el morfismo `Phase1Handoff`.
        El objeto retornado **es** el dominio de `_ingest_phase1_handoff`
        (primera flecha de FASE 2):

            $\Phi_2\circ\Phi_1(G)=\texttt{\_ingest\_phase1\_handoff}
             (\texttt{\_handoff\_to\_phase2}(G))$.
        """
        handoff = cls._certify_canonical(name, G)
        logger.info(
            "[FASE1→FASE2] %s: κ=%.3e, λ_min=%.3e, δ_Einstein=%.3e, "
            "L_Connes=%.3e, logvol=%.3e, ∇G=%.3e",
            name,
            handoff.spectral_profile.condition_number,
            handoff.spectral_profile.lambda_min,
            handoff.einstein_defect,
            handoff.lipschitz_connes_bound,
            handoff.log_volume,
            handoff.compatibility_residual,
        )
        return handoff

    # ═══════════════════════════════════════════════════════════════════════════
    # FASE 2 — RSI³ Y MAPA DE POINCARÉ
    # (inicia exactamente donde termina FASE 1.Z)
    # ═══════════════════════════════════════════════════════════════════════════

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 2.A — INGESTA del handoff de FASE 1 (continuación directa)
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _ingest_phase1_handoff(
        cls, handoff: Phase1Handoff
    ) -> Tuple[np.ndarray, float]:
        r"""
        Consume el `Phase1Handoff` de `_handoff_to_phase2` (continuación 1→2).

        Extrae la semilla métrica certificada $G_\star$ y el defecto de Einstein
        inicial — sustrato de $\Phi^{(1)}\subset\Phi^{(2)}\subset\Phi^{(3)}$.
        """
        return (
            np.array(handoff.candidate, dtype=_FLOAT_DTYPE, copy=True),
            float(handoff.einstein_defect),
        )

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 2.B — Nivel 1: paso de Ricci normalizado (unidad atómica)
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _rsi_level1_ricci_step(cls, G: np.ndarray, delta_tau: float) -> np.ndarray:
        r"""
        RSI$^1$: $\Phi^{(1)}_{\Delta\tau}(G)=\Pi_{\mathrm{SPD}}\bigl(
        G-2\Delta\tau(\operatorname{Ric}-(R/n)G)\bigr)$.
        """
        n = int(G.shape[0])
        ricci, scalar_r = cls._compute_ricci_tensor(G)
        G_next = G - 2.0 * float(delta_tau) * (ricci - (scalar_r / float(n)) * G)
        return cls._project_psd(G_next)

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 2.C — Nivel 2: adaptación de Δτ por gradiente de curvatura
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _rsi_level2_adaptive_tau(
        cls,
        G: np.ndarray,
        delta_tau: float,
        eta: float,
        curvature_history: List[float],
    ) -> Tuple[np.ndarray, float]:
        r"""
        RSI$^2$: $\Phi^{(2)}_\eta$ adapta el paso según $\|\nabla R\|$:

            $\Delta\tau_{k+1}=\mathrm{sat}\bigl(\Delta\tau_k\exp(-\eta|\nabla R_k|)\bigr)$,

        saturado en $[10^{-5},10^{-1}]$ (CFL discreta). Anida $\Phi^{(1)}$.
        """
        if len(curvature_history) >= 2:
            grad_R = abs(curvature_history[-1] - curvature_history[-2])
            tau_new = float(
                np.clip(delta_tau * np.exp(-eta * grad_R), _TAU_LOWER, _TAU_UPPER)
            )
        else:
            tau_new = float(delta_tau)
        return cls._rsi_level1_ricci_step(G, tau_new), tau_new

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 2.D — Nivel 3: adaptación de η por contracción de Banach
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _rsi_level3_adaptive_eta(
        cls,
        G: np.ndarray,
        delta_tau: float,
        eta: float,
        zeta: float,
        contraction_history: List[float],
        curvature_history: List[float],
    ) -> Tuple[np.ndarray, float, float]:
        r"""
        RSI$^3$: $\Phi^{(3)}_\zeta$ adapta $\eta$ a la desviación de $L_\Phi$
        respecto de $L_\star=0.95$:

            $\eta_{k+1}=\mathrm{sat}\bigl(\eta_k\exp(\zeta(L_k-L_\star))\bigr)$.

        Anida $\Phi^{(2)}\supset\Phi^{(1)}$.
        """
        if len(contraction_history) >= 1:
            L_recent = float(contraction_history[-1])
            eta_new = float(
                np.clip(
                    eta * np.exp(zeta * (L_recent - _BANACH_TARGET)),
                    _ETA_LOWER,
                    _ETA_UPPER,
                )
            )
        else:
            eta_new = float(eta)
        G_next, tau_new = cls._rsi_level2_adaptive_tau(
            G, delta_tau, eta_new, curvature_history
        )
        return G_next, tau_new, eta_new

    @classmethod
    def _adapt_zeta_meta_meta(cls, zeta: float, eta_history: List[float]) -> float:
        r"""
        Recursión meta-meta de nivel 3: $\zeta$ se atenúa si $\eta$ oscila
        y se refuerza si $\eta$ estanca — automejora de la ley de adaptación.
        """
        if len(eta_history) < 4:
            return float(np.clip(zeta, _ZETA_LOWER, _ZETA_UPPER))
        diffs = np.diff(np.asarray(eta_history[-4:], dtype=_FLOAT_DTYPE))
        osc = float(np.std(diffs))
        eta_abs = max(abs(float(eta_history[-1])), 1.0e-12)
        if osc > 0.5 * eta_abs:
            zeta_new = 0.5 * zeta
        elif osc < 1.0e-6:
            zeta_new = 1.1 * zeta
        else:
            zeta_new = zeta
        return float(np.clip(zeta_new, _ZETA_LOWER, _ZETA_UPPER))

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 2.E — Constante de Banach empírica y sonda de Lipschitz
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _banach_contraction_constant(orbit: List[np.ndarray]) -> float:
        r"""
        $L_\Phi\approx\sup_k\|G_{k+1}-G_k\|_F/\|G_k-G_{k-1}\|_F$.
        Si $L_\Phi<1$ ⇒ flujo contractivo ⇒ punto fijo único (Banach).
        """
        if len(orbit) < 3:
            return 1.0
        diffs = [_fro(orbit[k + 1] - orbit[k]) for k in range(len(orbit) - 1)]
        ratios = [
            diffs[k + 1] / max(diffs[k], 1e-15) for k in range(len(diffs) - 1)
        ]
        if not ratios:
            return 1.0
        clipped = np.clip(np.asarray(ratios, dtype=_FLOAT_DTYPE), 0.0, 1.0e6)
        return float(np.max(clipped))

    @classmethod
    def _lipschitz_phi_empirical(
        cls,
        G: np.ndarray,
        delta_tau: float,
        eta: float,
        curvature_history: List[float],
    ) -> float:
        r"""
        Sonda determinista de $\mathrm{Lip}(\Phi^{(2)})$ en la dirección del
        Ricci sin traza (vector tangente canónico al flujo).
        """
        n = int(G.shape[0])
        ricci, scalar_r = cls._compute_ricci_tensor(G)
        tangent = ricci - (scalar_r / float(n)) * G
        vnorm = _fro(tangent)
        if vnorm < 1.0e-15:
            tangent = np.eye(n, dtype=_FLOAT_DTYPE)
            vnorm = float(np.sqrt(n))
        eps = _LIPSCHITZ_PERT_REL * max(_fro(G), 1.0)
        perturbation = (eps / vnorm) * tangent
        G_pert = cls._project_psd(G + perturbation)
        dummy = (
            list(curvature_history[-2:])
            if len(curvature_history) >= 2
            else [scalar_r, scalar_r]
        )
        phi_g, _ = cls._rsi_level2_adaptive_tau(G, delta_tau, eta, dummy)
        phi_p, _ = cls._rsi_level2_adaptive_tau(G_pert, delta_tau, eta, dummy)
        den = _fro(G_pert - G)
        return _fro(phi_p - phi_g) / max(den, 1.0e-15)

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 2.F — Driver RSI³ completo con anidamiento 1 ⊂ 2 ⊂ 3
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _rsi3_driver(
        cls,
        G0: np.ndarray,
        n_iterations: int = _RSI3_MAX_ITER,
        delta_tau0: float = 1e-2,
        eta0: float = 0.1,
        zeta0: float = 0.01,
    ) -> Dict[str, Any]:
        r"""
        Orquesta RSI$^3$: en cada iteración ejecuta
        $\Phi^{(3)}\supset\Phi^{(2)}\supset\Phi^{(1)}$ y adapta $\zeta$ (meta-meta).

        Cada `_LIPSCHITZ_PROBE_STRIDE` pasos se inyecta una sonda de Lipschitz
        determinista, más fiel a $L_\Phi$ que el cociente orbital crudo.
        Parada temprana si $\delta_E<\varepsilon$ tras al menos `_RSI3_MIN_ITER` pasos.
        """
        G = np.array(G0, dtype=_FLOAT_DTYPE, copy=True)
        delta_tau, eta, zeta = float(delta_tau0), float(eta0), float(zeta0)
        orbit: List[np.ndarray] = [np.array(G, copy=True)]
        _, r0 = cls._compute_ricci_tensor(G)
        curv_hist: List[float] = [float(r0)]
        einstein_hist: List[float] = [cls._compute_einstein_defect(G)]
        contraction_hist: List[float] = []
        tau_hist: List[float] = [delta_tau]
        eta_hist: List[float] = [eta]
        zeta_hist: List[float] = [zeta]
        n_iter = max(int(n_iterations), 1)

        for k in range(n_iter):
            zeta = cls._adapt_zeta_meta_meta(zeta, eta_hist)
            G_next, delta_tau, eta = cls._rsi_level3_adaptive_eta(
                G, delta_tau, eta, zeta, contraction_hist, curv_hist
            )
            orbit.append(np.array(G_next, copy=True))
            _, scalar_r = cls._compute_ricci_tensor(G_next)
            curv_hist.append(float(scalar_r))
            einstein_hist.append(cls._compute_einstein_defect(G_next))
            tau_hist.append(delta_tau)
            eta_hist.append(eta)
            zeta_hist.append(zeta)

            d_new = 0.0
            if k >= 1:
                d_new = _fro(orbit[-1] - orbit[-2])
                d_old = _fro(orbit[-2] - orbit[-3]) if k >= 2 else d_new
                if d_old > 1e-15:
                    contraction_hist.append(d_new / d_old)

            if k > 0 and (k % _LIPSCHITZ_PROBE_STRIDE == 0):
                try:
                    probe = cls._lipschitz_phi_empirical(
                        G_next, delta_tau, eta, curv_hist
                    )
                    probe = float(np.clip(probe, 0.0, 1.0e6))
                    if contraction_hist:
                        contraction_hist[-1] = probe
                    else:
                        contraction_hist.append(probe)
                except (LinAlgError, ValueError, MetricTensorError) as exc:
                    logger.debug("[RSI³] sonda Lipschitz omitida: %s", exc)

            if (
                k + 1 >= _RSI3_MIN_ITER
                and einstein_hist[-1] < _EINSTEIN_TOL
            ):
                logger.info(
                    "[RSI³] Convergencia temprana en k=%d, δ_E=%.3e",
                    k,
                    einstein_hist[-1],
                )
                G = G_next
                break

            if k >= 1 and d_new < _RSI3_CONV_EPS:
                logger.info("[RSI³] Punto fijo numérico en k=%d, ‖ΔG‖=%.3e", k, d_new)
                G = G_next
                break

            G = G_next

        L_final = cls._banach_contraction_constant(orbit)
        converged = bool(
            (L_final < _BANACH_TARGET) or (einstein_hist[-1] < _EINSTEIN_TOL)
        )
        logger.info(
            "[RSI³] Niveles=1⊂2⊂3, iter=%d, L_Φ=%.4f, converged=%s, "
            "Δτ_final=%.3e, η_final=%.3e, ζ_final=%.3e, δ_E=%.3e",
            len(orbit) - 1,
            L_final,
            converged,
            delta_tau,
            eta,
            zeta,
            einstein_hist[-1],
        )
        return {
            "orbit": orbit,
            "curvature_history": curv_hist,
            "contraction_history": contraction_hist,
            "delta_tau_history": tau_hist,
            "eta_history": eta_hist,
            "zeta_history": zeta_hist,
            "einstein_defect_history": einstein_hist,
            "banach_L": L_final,
            "rsi_level_achieved": 3,
            "converged": converged,
        }

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 2.G — Mapa de retorno de Poincaré (sección interpolada)
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _poincare_return_map(
        orbit: List[np.ndarray],
        energy_target: float,
        curvature_history: List[float],
    ) -> Dict[str, Any]:
        r"""
        Sección $\Sigma=\{R=\bar R\}$ con interpolación lineal del tiempo de
        vuelo (Poincaré, *Méthodes Nouvelles*, t. I):

            $\alpha=(\bar R-R_{k-1})/(R_k-R_{k-1}),\quad
             G_{\tau^\star}=(1-\alpha)G_{k-1}+\alpha G_k$.

        Estima periodos, periodicidad y el multiplicador de Floquet
        $\mu\approx\|G_{\tau_{i+1}}-G_{\tau_i}\|/\|G_{\tau_i}-G_{\tau_{i-1}}\|$.

        Nota v6: se elimina el predicado `abs(R_k-R̄)<tol` del v5, que
        contaminaba la sección con falsos cruces cerca de la mediana.
        """
        energies = np.asarray(curvature_history, dtype=_FLOAT_DTYPE)
        n_min = min(int(energies.size), len(orbit))
        energies = energies[:n_min]
        orbit_use = orbit[:n_min]

        crossings: List[int] = []
        taus: List[float] = []
        section_points: List[np.ndarray] = []

        for k in range(1, n_min):
            e0 = float(energies[k - 1])
            e1 = float(energies[k])
            if (e0 - energy_target) * (e1 - energy_target) < 0.0:
                denom = e1 - e0
                alpha = 0.5 if abs(denom) < 1.0e-15 else (energy_target - e0) / denom
                alpha = float(np.clip(alpha, 0.0, 1.0))
                crossings.append(k)
                taus.append((k - 1) + alpha)
                g_cross = (1.0 - alpha) * orbit_use[k - 1] + alpha * orbit_use[k]
                section_points.append(0.5 * (g_cross + g_cross.T))

        periods = (
            np.diff(np.asarray(taus, dtype=_FLOAT_DTYPE))
            if len(taus) > 1
            else np.array([], dtype=_FLOAT_DTYPE)
        )
        is_periodic = False
        if periods.size >= 2:
            mean_p = float(np.mean(periods))
            std_rel = float(np.std(periods) / max(mean_p, 1e-15))
            is_periodic = std_rel < 0.05

        floquet = 0.0
        if len(section_points) >= 3:
            d0 = _fro(section_points[1] - section_points[0])
            d1 = _fro(section_points[2] - section_points[1])
            floquet = float(d1 / max(d0, 1.0e-15))

        return {
            "crossings": crossings,
            "crossing_times": taus,
            "return_periods": periods,
            "is_periodic_orbit": is_periodic,
            "energies": energies,
            "energy_target": float(energy_target),
            "floquet_multiplier": floquet,
            "section_points": section_points,
        }

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 2.H — Espectro de Lyapunov y frecuencias orbitales (FFT)
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _lyapunov_exponent_spectrum(orbit: List[np.ndarray]) -> np.ndarray:
        r"""
        $\lambda_{\max}=\overline{\log(\|G_{k+1}-G_k\|_F/\|G_k-G_{k-1}\|_F)}$.
        $\lambda_{\max}>0$ ⇒ caos; $\lambda_{\max}<0$ ⇒ convergencia.
        """
        if len(orbit) < 3:
            return np.array([-np.inf], dtype=_FLOAT_DTYPE)
        norms = [_fro(orbit[k + 1] - orbit[k]) for k in range(len(orbit) - 1)]
        log_ratios = [
            float(np.log(max(norms[k + 1] / max(norms[k], 1e-15), 1e-15)))
            for k in range(len(norms) - 1)
        ]
        if not log_ratios:
            return np.array([-np.inf], dtype=_FLOAT_DTYPE)
        return np.array([float(np.mean(log_ratios))], dtype=_FLOAT_DTYPE)

    @staticmethod
    def _extract_orbital_frequencies(
        signal: np.ndarray, dt: float = 1.0
    ) -> np.ndarray:
        r"""
        Extrae frecuencias dominantes de $R(\tau)$ por FFT con ventana de Hann.

        Estas $\omega$ son las frecuencias de la mecánica celeste discreta
        (ángulo-acción) que alimentan el criterio diofántico KAM en FASE 3.
        """
        x = np.asarray(signal, dtype=_FLOAT_DTYPE)
        n = int(x.size)
        if n < 8:
            return np.array([], dtype=_FLOAT_DTYPE)
        x = x - float(np.mean(x))
        hann = 0.5 - 0.5 * np.cos(
            2.0 * np.pi * np.arange(n, dtype=_FLOAT_DTYPE) / max(n - 1, 1)
        )
        spec = np.abs(np.fft.rfft(x * hann))
        freqs = np.fft.rfftfreq(n, d=float(dt))
        spec[0] = 0.0
        thresh = float(np.median(spec) * 3.0)
        peaks: List[float] = []
        for i in range(1, len(spec) - 1):
            if spec[i] > spec[i - 1] and spec[i] > spec[i + 1] and spec[i] > thresh:
                peaks.append(float(freqs[i]))
        peaks.sort(key=lambda f: -float(spec[int(np.argmin(np.abs(freqs - f)))]))
        return np.asarray(peaks[:4], dtype=_FLOAT_DTYPE)

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 2.Z — HANDOFF: última definición de FASE 2 ≡ inicio de FASE 3
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _handoff_to_phase3(
        cls, phase1: Phase1Handoff, rsi_out: Dict[str, Any]
    ) -> Phase2Handoff:
        r"""
        Última definición formal de FASE 2 y **continuación estricta** de FASE 3.

        Empaqueta RSI$^3$ + Poincaré interpolado + Lyapunov + FFT y construye el
        morfismo `Phase2Handoff`, dominio de `_ingest_phase2_handoff`:

            $\Phi_3\circ\Phi_2(h_1)=\texttt{\_ingest\_phase2\_handoff}
             (\texttt{\_handoff\_to\_phase3}(h_1,\mathrm{RSI}^3))$.
        """
        orbit: List[np.ndarray] = list(rsi_out["orbit"])
        curv_hist = list(rsi_out["curvature_history"])
        energy_target = float(np.median(curv_hist)) if curv_hist else 0.0
        poincare = cls._poincare_return_map(orbit, energy_target, curv_hist)
        lyap = cls._lyapunov_exponent_spectrum(orbit)
        dt_mean = float(np.mean(rsi_out.get("delta_tau_history", [1.0])))
        omega = cls._extract_orbital_frequencies(
            np.asarray(curv_hist, dtype=_FLOAT_DTYPE),
            dt=max(dt_mean, 1.0e-12),
        )

        handoff = Phase2Handoff(
            phase1=phase1,
            orbit=tuple(orbit),
            curvature_history=tuple(float(x) for x in curv_hist),
            lyapunov_max=float(lyap[0]) if len(lyap) else -np.inf,
            poincare_data=poincare,
            banach_L=float(rsi_out["banach_L"]),
            delta_tau_history=tuple(float(x) for x in rsi_out["delta_tau_history"]),
            eta_history=tuple(float(x) for x in rsi_out["eta_history"]),
            einstein_defect_history=tuple(
                float(x) for x in rsi_out["einstein_defect_history"]
            ),
            rsi3_level=int(rsi_out["rsi_level_achieved"]),
            rsi3_converged=bool(rsi_out["converged"]),
            dominant_frequencies=tuple(float(x) for x in omega),
        )
        logger.info(
            "[FASE2→FASE3] %s: λ_max=%.3e, L_Φ=%.4f, crossings=%d, "
            "Floquet=%.3e, n_ω=%d, converged=%s",
            phase1.tensor_name,
            handoff.lyapunov_max,
            handoff.banach_L,
            len(poincare["crossings"]),
            float(poincare.get("floquet_multiplier", 0.0)),
            len(handoff.dominant_frequencies),
            handoff.rsi3_converged,
        )
        return handoff

    # ═══════════════════════════════════════════════════════════════════════════
    # FASE 3 — KAM, TANGLE HOMOCLÍNICO, POINCARÉ–HOPF Y CERTIFICADO
    # (inicia exactamente donde termina FASE 2.Z)
    # ═══════════════════════════════════════════════════════════════════════════

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 3.A — INGESTA del handoff de FASE 2 (continuación directa)
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _ingest_phase2_handoff(
        cls, handoff: Phase2Handoff
    ) -> Tuple[Phase1Handoff, List[np.ndarray], Dict[str, Any], float, Tuple[float, ...]]:
        r"""
        Consume el `Phase2Handoff` de `_handoff_to_phase3` (continuación 2→3).

        Extrae el sustrato geométrico, la órbita, Poincaré, Lyapunov y las
        frecuencias orbitales (FFT) para el criterio KAM.
        """
        return (
            handoff.phase1,
            list(handoff.orbit),
            handoff.poincare_data,
            float(handoff.lyapunov_max),
            tuple(handoff.dominant_frequencies),
        )

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 3.B — Estabilidad KAM vía criterio diofántico (pares + retículo)
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _diophantine_gamma(
        omega: np.ndarray,
        tau: float = _KAM_TAU,
        max_k_norm: int = 4,
    ) -> float:
        r"""
        Ínfimo $\gamma=\inf_k|\omega\cdot k|\,|k|^{\tau}$ sobre el retículo.

        Si $\dim\omega>4$ se reduce a chequeos *pares* (resonancias 2-toros),
        evitando la explosión $|\mathcal{L}|=(2K+1)^n$ (crítica en $G_{\mathrm{global}}\in\mathbb{R}^{7\times7}$).
        """
        omega = np.asarray(omega, dtype=_FLOAT_DTYPE).ravel()
        n = int(omega.size)
        if n == 0:
            return float("inf")
        if n == 1:
            return float(abs(omega[0]))

        min_gamma = np.inf
        if n > 4:
            for i, j in combinations(range(n), 2):
                pair = omega[[i, j]]
                for k in product(range(-max_k_norm, max_k_norm + 1), repeat=2):
                    if k[0] == 0 and k[1] == 0:
                        continue
                    k_arr = np.asarray(k, dtype=_FLOAT_DTYPE)
                    k_norm = float(np.linalg.norm(k_arr))
                    if k_norm > max_k_norm or k_norm < 1.0e-15:
                        continue
                    dot = abs(float(np.dot(pair, k_arr)))
                    candidate = dot * (k_norm ** tau)
                    if candidate < min_gamma:
                        min_gamma = candidate
            return float(min_gamma)

        for k in product(range(-max_k_norm, max_k_norm + 1), repeat=n):
            if all(ki == 0 for ki in k):
                continue
            k_arr = np.asarray(k, dtype=_FLOAT_DTYPE)
            k_norm = float(np.linalg.norm(k_arr))
            if k_norm > max_k_norm or k_norm < 1.0e-15:
                continue
            dot = abs(float(np.dot(omega, k_arr)))
            candidate = dot * (k_norm ** tau)
            if candidate < min_gamma:
                min_gamma = candidate
        return float(min_gamma)

    @classmethod
    def _kam_torus_stability(
        cls,
        eigenvalues: np.ndarray,
        tau: float = _KAM_TAU,
        max_k_norm: int = 4,
        orbital_frequencies: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        r"""
        Criterio diofántico KAM.

        Preferencia: frecuencias orbitales $\omega$ extraídas de $R(\tau)$
        (ángulo-acción). Respaldo: $\omega_i=\log\lambda_i(G)$.
        """
        if orbital_frequencies is not None and np.asarray(orbital_frequencies).size >= 1:
            omega = np.asarray(orbital_frequencies, dtype=_FLOAT_DTYPE)
            source = "orbital"
        else:
            ev = np.asarray(eigenvalues, dtype=_FLOAT_DTYPE)
            omega = np.log(np.maximum(ev, _SPECTRAL_PSD_FLOOR))
            source = "spectral"
            if omega.size < 2:
                return {
                    "diophantine_gamma": float("inf"),
                    "is_kam_stable": True,
                    "source": source,
                }

        gamma = cls._diophantine_gamma(omega, tau=tau, max_k_norm=max_k_norm)
        return {
            "diophantine_gamma": gamma,
            "is_kam_stable": bool(gamma >= _KAM_GAMMA),
            "source": source,
        }

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 3.C — Detección de tangle homoclínico (Poincaré–Smale)
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _detect_homoclinic_tangle(
        lyapunov_max: float, poincare_data: Dict[str, Any]
    ) -> bool:
        r"""
        Tangle homoclínico (Poincaré 1890; Smale horseshoe):
          (i)  $\lambda_{\max}>0$ (hiperbolicidad),
          (ii) $\ge 3$ cruces de $\Sigma$ con periodos *no* periódicos,
          (iii) $|\mu_{\mathrm{Floquet}}|>1$ (estiramiento de la sección).
        """
        if lyapunov_max <= _LYAPUNOV_CHAOS:
            return False
        crossings = poincare_data.get("crossings", [])
        if len(crossings) < 3:
            return False
        periods = np.asarray(poincare_data.get("return_periods", []), dtype=_FLOAT_DTYPE)
        if periods.size < 2:
            return False
        irregular = bool(
            np.std(periods) / max(float(np.mean(periods)), 1e-15) > 0.1
        )
        floquet = abs(float(poincare_data.get("floquet_multiplier", 0.0)))
        stretching = floquet > _FLOQUET_UNSTABLE
        return bool(irregular and stretching)

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 3.D — Índice de Poincaré–Hopf discreto
    # ───────────────────────────────────────────────────────────────────────────
    @staticmethod
    def _poincare_hopf_index(orbit: List[np.ndarray]) -> float:
        r"""
        Índice topológico discreto sobre mínimos locales de $\|v_k\|_F$:

            $\mathrm{ind}(p)=\mathrm{sign}\langle v_{\mathrm{in}},v_{\mathrm{out}}\rangle_F$.

        Aproxima $\sum_p\mathrm{sign}\det(J_p-I)$ a lo largo de la órbita.
        """
        if len(orbit) < 3:
            return 0.0
        velocities = [_fro(orbit[k + 1] - orbit[k]) for k in range(len(orbit) - 1)]
        if max(velocities) < 1e-15:
            return 0.0
        total = 0.0
        for k in range(1, len(velocities) - 1):
            if velocities[k] < velocities[k - 1] and velocities[k] < velocities[k + 1]:
                d_in = orbit[k] - orbit[k - 1]
                d_out = orbit[k + 1] - orbit[k]
                n_in = _fro(d_in)
                n_out = _fro(d_out)
                if n_in < 1e-15 or n_out < 1e-15:
                    continue
                cos_ang = float(np.sum(d_in * d_out) / (n_in * n_out))
                total += 1.0 if cos_ang > 0.0 else -1.0
        return float(total)

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 3.E — Combinatoria de Heyting Ω₃ y certificado
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def _heyting_combine(
        cls,
        is_stable: bool,
        kam_stable: bool,
        tangle: bool,
        floquet: float,
        banach_L: float,
        einstein_defect: float,
        used_fallback: bool,
    ) -> HeytingVerdict:
        """Join de átomos en $\Omega_3$: el veredicto es el supremo de obstrucciones."""
        v_stab = HeytingVerdict.COHERENT if is_stable else HeytingVerdict.DEGRADED
        v_kam = HeytingVerdict.COHERENT if kam_stable else HeytingVerdict.DEGRADED
        v_tangle = HeytingVerdict.VETOED if tangle else HeytingVerdict.COHERENT
        v_floquet = (
            HeytingVerdict.DEGRADED
            if abs(floquet) > _FLOQUET_UNSTABLE
            else HeytingVerdict.COHERENT
        )
        v_banach = (
            HeytingVerdict.COHERENT
            if banach_L < _BANACH_TARGET
            else HeytingVerdict.DEGRADED
        )
        v_ein = (
            HeytingVerdict.COHERENT
            if einstein_defect < _EINSTEIN_TOL
            else HeytingVerdict.DEGRADED
        )
        v_fb = HeytingVerdict.DEGRADED if used_fallback else HeytingVerdict.COHERENT
        return (
            v_stab.join(v_kam)
            .join(v_tangle)
            .join(v_floquet)
            .join(v_banach)
            .join(v_ein)
            .join(v_fb)
        )

    @classmethod
    def _finalize_hardened_certificate(
        cls, handoff: Phase2Handoff
    ) -> Tuple[np.ndarray, HardenedCertificate]:
        r"""
        Consume el `Phase2Handoff`, ejecuta KAM/tangle/Hopf sobre el estado
        final y emite `(G_certificado, HardenedCertificate)`.

        Si el estado terminal del flujo **degrada** la certificación SPD/κ,
        se retorna el candidato de FASE 1 (ya certificado) y se marca
        `used_phase1_fallback=True` — invariante de fábrica: jamás se emite
        un tensor no SPD.
        """
        phase1, orbit, poincare, lyap_max, omega_orb = cls._ingest_phase2_handoff(
            handoff
        )
        G_flow = np.array(orbit[-1], dtype=_FLOAT_DTYPE, copy=True)
        certified = cls._try_certify_spd(phase1.tensor_name, G_flow)
        used_fallback = False
        if certified is None:
            logger.warning(
                "[HARDEN] %s: el flujo RSI³ degradó SPD/κ; se restaura el "
                "candidato certificado de FASE 1.",
                phase1.tensor_name,
            )
            G_final = np.array(phase1.candidate, dtype=_FLOAT_DTYPE, copy=True)
            spectrum_final = phase1.spectral_profile
            used_fallback = True
        else:
            G_final, spectrum_final = certified

        kam = cls._kam_torus_stability(
            spectrum_final.eigenvalues,
            orbital_frequencies=np.asarray(omega_orb, dtype=_FLOAT_DTYPE),
        )
        tangle = cls._detect_homoclinic_tangle(lyap_max, poincare)
        hopf = cls._poincare_hopf_index(orbit)
        einstein_defect = cls._compute_einstein_defect(G_final)
        lipschitz = cls._compute_lipschitz_connes_bound(G_final)
        floquet = float(poincare.get("floquet_multiplier", 0.0))
        try:
            log_volume = cls._spd_logdet(G_final)
        except LinAlgError:
            log_volume = float("-inf")
        try:
            compat = cls._metric_compatibility_residual(G_final)
        except LinAlgError:
            compat = float("inf")

        stable = bool(
            handoff.rsi3_converged
            and lyap_max <= _LYAPUNOV_CHAOS
            and kam["is_kam_stable"]
            and not tangle
            and spectrum_final.lambda_min > MIN_EIGVAL_TOL
            and spectrum_final.condition_number <= COND_NUM_TOL
            and not used_fallback
        )
        verdict = cls._heyting_combine(
            is_stable=stable,
            kam_stable=bool(kam["is_kam_stable"]),
            tangle=bool(tangle),
            floquet=floquet,
            banach_L=float(handoff.banach_L),
            einstein_defect=float(einstein_defect),
            used_fallback=used_fallback,
        )

        rsi_trace = RSI3Trace(
            rsi_level_achieved=handoff.rsi3_level,
            banach_L=handoff.banach_L,
            delta_tau_history=handoff.delta_tau_history,
            eta_history=handoff.eta_history,
            einstein_defect_history=handoff.einstein_defect_history,
            converged=handoff.rsi3_converged,
        )
        periods = poincare.get("return_periods", np.array([]))
        poincare_dt = PoincareData(
            return_periods=tuple(float(x) for x in np.asarray(periods).ravel()),
            lyapunov_max=float(lyap_max),
            kam_stable=bool(kam["is_kam_stable"]),
            homoclinic_tangle=bool(tangle),
            hopf_index=float(hopf),
            diophantine_gamma=float(kam["diophantine_gamma"]),
            floquet_multiplier=floquet,
            dominant_frequencies=tuple(float(x) for x in omega_orb),
        )
        cert = HardenedCertificate(
            tensor_name=phase1.tensor_name,
            spectral_profile=spectrum_final,
            rsi3_trace=rsi_trace,
            poincare_data=poincare_dt,
            einstein_defect=float(einstein_defect),
            lipschitz_connes_bound=float(lipschitz),
            is_hardened=bool(stable and verdict is not HeytingVerdict.VETOED),
            is_geometrically_stable=stable,
            heyting_verdict=verdict,
            log_volume=float(log_volume),
            metric_compatibility_residual=float(compat),
            used_phase1_fallback=used_fallback,
        )
        return G_final, cert

    # ───────────────────────────────────────────────────────────────────────────
    # FASE 3.F — Endurecimiento público (punto de entrada RSI³+Poincaré)
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def harden_tensor(
        cls,
        name: str,
        G: np.ndarray,
        n_iterations: int = _RSI3_MAX_ITER,
    ) -> Tuple[np.ndarray, HardenedCertificate]:
        r"""
        Ejecuta el ciclo OODA completo anidando las tres fases:

          1. **FASE 1** (`_handoff_to_phase2`): validación canónica,
             Christoffel–Koszul, Ricci, defecto de Einstein, $\nabla G$.
          2. **FASE 2** (`_ingest_phase1_handoff` → `_rsi3_driver` →
             `_handoff_to_phase3`): RSI³ anidado 1⊂2⊂3 + Lyapunov +
             Poincaré interpolado + FFT.
          3. **FASE 3** (`_ingest_phase2_handoff` → `_finalize_hardened_certificate`):
             KAM diofántico + tangle + Floquet + Hopf + Heyting $\Omega_3$.

        Retorna la tupla `(tensor_endurecido_inmutable, certificado)`.
        Invariante: el tensor emitido es SPD (fallback a FASE 1 si el flujo degrada).
        """
        phase1 = cls._handoff_to_phase2(name, G)
        G_seed, _ = cls._ingest_phase1_handoff(phase1)
        rsi_out = cls._rsi3_driver(G_seed, n_iterations=int(n_iterations))
        phase2 = cls._handoff_to_phase3(phase1, rsi_out)
        G_final, cert = cls._finalize_hardened_certificate(phase2)
        G_frozen = cls._make_immutable(G_final)
        logger.info(
            "[HARDEN] %s: hardened=%s, stable=%s, Heyting=%s, δ_E=%.3e, "
            "λ_max=%.3e, γ_KAM=%.3e, tangle=%s, Floquet=%.3e, ind_Hopf=%.1f, "
            "fallback=%s",
            name,
            cert.is_hardened,
            cert.is_geometrically_stable,
            cert.heyting_verdict.value,
            cert.einstein_defect,
            cert.poincare_data.lyapunov_max,
            cert.poincare_data.diophantine_gamma,
            cert.poincare_data.homoclinic_tangle,
            cert.poincare_data.floquet_multiplier,
            cert.poincare_data.hopf_index,
            cert.used_phase1_fallback,
        )
        return G_frozen, cert

    # ───────────────────────────────────────────────────────────────────────────
    # CONSTRUCTORES (backward-compatible: harden es opt-in)
    # ───────────────────────────────────────────────────────────────────────────
    @classmethod
    def build_physics_tensor(cls) -> np.ndarray:
        """Tensor métrico $V_{\mathrm{phys}}\subset\mathbb{R}^3$ (v4.1.0 preservado)."""
        G_phys = np.array(
            [
                [2.50, 0.85, 0.30],
                [0.85, 1.50, 0.45],
                [0.30, 0.45, 1.00],
            ],
            dtype=_FLOAT_DTYPE,
        )
        return cls._make_immutable(cls._handoff_to_phase2("G_phys", G_phys).candidate)

    @classmethod
    def build_topology_tensor(cls) -> np.ndarray:
        """Tensor métrico $V_{\mathrm{topo}}\subset\mathbb{R}^2$ (v4.1.0 preservado)."""
        G_topo = np.array([[1.00, 0.60], [0.60, 3.00]], dtype=_FLOAT_DTYPE)
        return cls._make_immutable(cls._handoff_to_phase2("G_topo", G_topo).candidate)

    @classmethod
    def build_thermo_tensor(cls) -> np.ndarray:
        """Tensor métrico $V_{\mathrm{thermo}}\subset\mathbb{R}^2$ (v4.1.0 preservado)."""
        G_thermo = np.array([[1.80, 0.75], [0.75, 2.20]], dtype=_FLOAT_DTYPE)
        return cls._make_immutable(cls._handoff_to_phase2("G_thermo", G_thermo).candidate)

    @classmethod
    def build_global_tensor(
        cls,
        physics_tensor: np.ndarray,
        topology_tensor: np.ndarray,
        thermo_tensor: np.ndarray,
    ) -> np.ndarray:
        r"""
        Ensambla el tensor $7\times 7$. Inyecta fricciones cruzadas y **restaura
        dominancia diagonal de Gershgorin** post-inyección (levantamiento
        mínimo), garantizando discos espectrales en $\{\Re\lambda>0\}$ antes
        de la certificación espectral de FASE 1.
        """
        G_base = block_diag(physics_tensor, topology_tensor, thermo_tensor)
        G_global = np.array(G_base, dtype=_FLOAT_DTYPE, copy=True)

        dim_phys = int(physics_tensor.shape[0])
        dim_topo = int(topology_tensor.shape[0])

        idx_sat = 0
        idx_frag = dim_phys + 0
        idx_ciclos = dim_phys + 1
        idx_entropia = dim_phys + dim_topo + 0

        G_global[idx_ciclos, idx_entropia] = 0.50
        G_global[idx_entropia, idx_ciclos] = 0.50
        G_global[idx_sat, idx_frag] = 0.35
        G_global[idx_frag, idx_sat] = 0.35

        G_global = cls._enforce_gershgorin_spd_margin(G_global)
        return cls._make_immutable(cls._handoff_to_phase2("G_global", G_global).candidate)


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCIONES UTILITARIAS PARA DIAGNÓSTICO EXTERNO
# ═══════════════════════════════════════════════════════════════════════════════

def get_tensor_diagnostics(G: np.ndarray, name: str = "unnamed") -> dict:
    """Diagnósticos completos (v4.1.0 preservado + extensión RSI³ / v6)."""
    try:
        G_arr = np.asarray(G, dtype=_FLOAT_DTYPE)
        if G_arr.ndim != 2 or G_arr.shape[0] != G_arr.shape[1]:
            return {
                "name": name,
                "error": f"No es matriz cuadrada: shape={G_arr.shape}",
                "error_type": "StructuralError",
            }
        sym_tol = MetricTensorFactory._compute_symmetry_tolerance(G_arr)
        asym = _fro(G_arr - G_arr.T)
        G_sym = MetricTensorFactory._symmetrize_by_projection(G_arr)
        profile = MetricTensorFactory._compute_spectral_profile(G_sym)
        is_spd = MetricTensorFactory._verify_spd_by_cholesky(G_sym)
        einstein_defect = MetricTensorFactory._compute_einstein_defect(G_sym)
        lipschitz = MetricTensorFactory._compute_lipschitz_connes_bound(G_sym)
        try:
            log_volume = MetricTensorFactory._spd_logdet(G_sym)
        except LinAlgError:
            log_volume = float("-inf")
        try:
            compat = MetricTensorFactory._metric_compatibility_residual(G_sym)
        except LinAlgError:
            compat = float("inf")

        return {
            "name": name,
            "module_version": __version__,
            "shape": G_arr.shape,
            "eigenvalues": profile.eigenvalues.tolist(),
            "dtype": str(G_arr.dtype),
            "is_symmetric": asym <= sym_tol,
            "asymmetry_norm": asym,
            "symmetry_tolerance": sym_tol,
            "is_spd": is_spd,
            "lambda_min": profile.lambda_min,
            "lambda_max": profile.lambda_max,
            "condition_number": profile.condition_number,
            "spectral_gap": profile.spectral_gap,
            "frobenius_norm": profile.frobenius_norm,
            "is_well_conditioned": profile.is_well_conditioned(),
            "is_strictly_positive": profile.is_strictly_positive(),
            "einstein_defect": einstein_defect,
            "lipschitz_connes_bound": lipschitz,
            "log_volume": log_volume,
            "metric_compatibility_residual": compat,
        }
    except Exception as exc:
        return {"name": name, "error": str(exc), "error_type": type(exc).__name__}


def get_hardening_diagnostics(
    name: str, G: np.ndarray, n_iterations: int = _RSI3_MAX_ITER
) -> dict:
    """Ejecuta el ciclo RSI³+Poincaré y devuelve dict serializable."""
    try:
        _, cert = MetricTensorFactory.harden_tensor(name, G, n_iterations)
        return {
            "module_version": __version__,
            "tensor_name": cert.tensor_name,
            "is_hardened": cert.is_hardened,
            "is_geometrically_stable": cert.is_geometrically_stable,
            "heyting_verdict": cert.heyting_verdict.value,
            "heyting_omega": cert.heyting_verdict.omega,
            "einstein_defect": cert.einstein_defect,
            "lipschitz_connes_bound": cert.lipschitz_connes_bound,
            "log_volume": cert.log_volume,
            "metric_compatibility_residual": cert.metric_compatibility_residual,
            "used_phase1_fallback": cert.used_phase1_fallback,
            "spectral": {
                "lambda_min": cert.spectral_profile.lambda_min,
                "lambda_max": cert.spectral_profile.lambda_max,
                "condition_number": cert.spectral_profile.condition_number,
            },
            "rsi3": {
                "level": cert.rsi3_trace.rsi_level_achieved,
                "banach_L": cert.rsi3_trace.banach_L,
                "converged": cert.rsi3_trace.converged,
                "n_steps": len(cert.rsi3_trace.delta_tau_history),
                "einstein_defect_history": list(cert.rsi3_trace.einstein_defect_history),
            },
            "poincare": {
                "lyapunov_max": cert.poincare_data.lyapunov_max,
                "kam_stable": cert.poincare_data.kam_stable,
                "homoclinic_tangle": cert.poincare_data.homoclinic_tangle,
                "hopf_index": cert.poincare_data.hopf_index,
                "diophantine_gamma": cert.poincare_data.diophantine_gamma,
                "floquet_multiplier": cert.poincare_data.floquet_multiplier,
                "dominant_frequencies": list(cert.poincare_data.dominant_frequencies),
                "return_periods": list(cert.poincare_data.return_periods),
            },
        }
    except Exception as exc:
        return {"name": name, "error": str(exc), "error_type": type(exc).__name__}


# ═══════════════════════════════════════════════════════════════════════════════
# PRECOMPILACIÓN (preservada; endurecimiento RSI³ es opt-in vía harden_tensor)
# ═══════════════════════════════════════════════════════════════════════════════

G_PHYSICS: Final[np.ndarray] = MetricTensorFactory.build_physics_tensor()
G_TOPOLOGY: Final[np.ndarray] = MetricTensorFactory.build_topology_tensor()
G_THERMODYNAMICS: Final[np.ndarray] = MetricTensorFactory.build_thermo_tensor()

logger.info(
    "Tensores métricos inicializados (v%s): G_PHYSICS %s, G_TOPOLOGY %s, G_THERMODYNAMICS %s",
    __version__,
    G_PHYSICS.shape,
    G_TOPOLOGY.shape,
    G_THERMODYNAMICS.shape,
)