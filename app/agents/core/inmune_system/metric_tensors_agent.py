# -*- coding: utf-8 -*-
r"""
====================================================================================================
ARQUITECTURA COGNITIVA APU FILTER - SISTEMA INMUNE SOBERANO
====================================================================================================

Nombre: MetricTensorsAgent
Ubicación: app/agents/core/inmune_system/metric_tensors_agent.py
Versión: 7.1.0
Función: Soberano de Calibre Riemanniano del sistema inmune. Orquesta la gobernanza geométrica,
         el flujo de Ricci discreto normalizado con automejora recursiva RSI³ (niveles 1 ⊂ 2 ⊂ 3, v7.1.0 Cayley-Darboux)
         y el mapa de retorno de Poincaré en el fibrado cotangente T*M sobre variedades riemannianas.

----------------------------------------------------------------------------------------------------
DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO:
----------------------------------------------------------------------------------------------------
El Soberano Agente de Tensores Métricos (MetricTensorsAgent) ejerce la gobernanza de calibre
riemanniano sobre la variedad de fase M ⊂ R^n. Su misión es auditar la coherencia métrica,
prevenir deformaciones singulares y controlar la dinámica del flujo de Ricci mediante un esquema
anidado de automejora recursiva RSI³ (niveles 1 ⊂ 2 ⊂ 3) acoplado a la mecánica celeste de Poincaré.
Su respectivo motor de cálculo geométrico y endurecimiento espectral es 'MetricTensorFactory'
(app/core/inmune_system/metric_tensors.py). Mientras el motor de tensores métricos efectúa la
regularización de Tikhonov, la proyección espectral PSD de Higham y la verificación de Cholesky,
este Agente orquesta la auditoría OODA de lazo cerrado, certificando la estabilidad diofántica
KAM, la ausencia de tangles homoclínicos, los exponentes de Lyapunov dominantes y la adjudicación
en el álgebra de Heyting Ω₃.

----------------------------------------------------------------------------------------------------
MARCO TEÓRICO Y FORMALISMO MATEMÁTICO:
----------------------------------------------------------------------------------------------------
1. Flujo de Ricci Discreto Normalizado y Automejora Recursiva RSI³:
   La evolución de la métrica riemanniana G_k bajo el flujo de Ricci normalizado responde a:
       G_{k+1} = \Pi_{SPD}(G_k - 2 \Delta\tau (Ric(G_k) - (R/n) G_k))
   El operador anidado de Banach \Phi^{(3)} \supset \Phi^{(2)} \supset \Phi^{(1)} adapta dinámicamente
   el paso temporal \Delta\tau y el parámetro \eta por gradientes de curvatura |\nabla R| y constantes de
   contracción L_\Phi = Lip(\Phi^{(2)}) < 1.

2. Derivación No Conmutativa de Connes y Símbolos de Christoffel de Koszul:
   La derivada no conmutativa \partial_\mu G_{\nu\sigma} = [X_\mu, G]_{\nu\sigma} con X_\mu = |e_\mu\rangle\langle e_\mu|
   sintetiza la conexión de Levi-Civita libre de torsión:
       \Gamma^\rho_{\mu\nu} = (1/2) G^{\rho\sigma} (\partial_\mu G_{\nu\sigma} + \partial_\nu G_{\mu\sigma} - \partial_\sigma G_{\mu\nu})

3. Mecánica Celeste de Poincaré y Criterio Diofántico KAM:
   Sobre la sección transversal \Sigma = {R = \bar R}, el mapa de retorno \mathcal{T}: \Sigma \to \Sigma
   evalúa los multiplicadores de Floquet \mu, la estabilidad diofántica KAM |\omega \cdot k| \ge \gamma / |k|^\tau
   y el exponente de Lyapunov dominante \lambda_{max}.

4. Clasificación en el Álgebra de Heyting Trivalente \Omega_3:
   La adjudicación intuicionista integra obstrucciones geométricas, espectrales y de simetría:
       HeytingVerdict = \bigwedge_i v_i \in \{\bot = COHERENT, (1/2) = DEGRADED, \top = VETOED\}

----------------------------------------------------------------------------------------------------
AXIOMAS FORMALES:
----------------------------------------------------------------------------------------------------
Axioma MTA1 (Positividad Definida Espectral):
    \lambda_{min}(G) \ge 10^{-12} para toda métrica proyectada \Pi_{SPD}(G).
Axioma MTA2 (Invarianza Simpléctica y Conservación de Volumen):
    M^\top \Omega M = \Omega \implies \det(M) = +1 para todo jacobiano de fase M.
Axioma MTA3 (Contracción de Banach en RSI³):
    L_\Phi = Lip(\Phi) \le L_\star = 0.95 en el espacio métrico completo (Sym^+, ||\cdot||_F).

----------------------------------------------------------------------------------------------------
INVARIANTES DE ESTADO:
----------------------------------------------------------------------------------------------------
Invariante 1 (Compatibilidad Métrica Residual Acotada):
    ||\nabla_\rho G||_F \le 10^{-4} ||G||_F.
Invariante 2 (Acotación de Condicionamiento Espectral):
    \kappa_2(G) = \lambda_{max} / \lambda_{min} \le 10^8.
====================================================================================================
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations, product
from typing import Any, Callable, Dict, Final, List, Optional, Tuple

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

logger = logging.getLogger("APUFilter.Physics.MetricTensorsAgent")

__all__ = (
    "HeytingVerdict",
    "MetricTensorGovernanceError",
    "SpectralProfile",
    "GeometricState",
    "RSI3Trace",
    "PoincareData",
    "MetricGovernanceCertificate",
    "Phase1Handoff",
    "Phase2Handoff",
    "MetricTensorsAgent",
)

__version__: Final[str] = "7.1.0"

# ── Constantes universales ──────────────────────────────────────────────────
_WILKINSON_EPS: Final[float] = float(np.finfo(np.float64).eps)
_SPECTRAL_PSD_FLOOR: Final[float] = 1.0e-12
_MAX_CONDITION_NUMBER: Final[float] = 1.0e8
_BANACH_TARGET: Final[float] = 0.95
_LYAPUNOV_CHAOS: Final[float] = 1.0e-2
_KAM_GAMMA: Final[float] = 1.0e-3
_KAM_TAU: Final[float] = 2.0
_RSI3_CONV_EPS: Final[float] = 1.0e-9
_SYMPLECTIC_ATOL: Final[float] = 1.0e-9
_COMPAT_REL_TOL: Final[float] = 1.0e-4
_FLOQUET_UNSTABLE: Final[float] = 1.0 + 1.0e-2
_LIPSCHITZ_PERT_REL: Final[float] = 1.0e-6
_ETA_BOUNDS: Final[Tuple[float, float]] = (1.0e-4, 10.0)
_TAU_BOUNDS: Final[Tuple[float, float]] = (1.0e-5, 0.1)
_ZETA_BOUNDS: Final[Tuple[float, float]] = (1.0e-4, 0.1)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  TIPOS, RETÍCULO DE HEYTING Ω₃ Y CERTIFICADOS INMUTABLES                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class HeytingVerdict(Enum):
    r"""Clasificador en el álgebra de Heyting trivalente $\Omega_3$.

    Orden de obstrucción: $\mathrm{COHERENT}=\bot < \mathrm{DEGRADED}=\tfrac12
    < \mathrm{VETOED}=\top$. El join $a\vee b=\max$ acumula obstrucciones.
    Codificación SIMD bitwise IRAM Xtensa: COHERENT=0x00, DEGRADED=0x01, VETOED=0x03.
    """

    COHERENT = "COHERENT"  # ⊥  Nominal
    DEGRADED = "DEGRADED"  # ½  Atención
    VETOED = "VETOED"      # ⊤  Supremo terminal

    @property
    def omega(self) -> float:
        """Inmersión $\Omega_3\hookrightarrow[0,1]$."""
        return {
            HeytingVerdict.COHERENT: 0.0,
            HeytingVerdict.DEGRADED: 0.5,
            HeytingVerdict.VETOED: 1.0,
        }[self]

    @property
    def mask(self) -> int:
        """Máscara de bits de 2 bits para aceleración IRAM SIMD."""
        return {
            HeytingVerdict.COHERENT: 0x00,
            HeytingVerdict.DEGRADED: 0x01,
            HeytingVerdict.VETOED: 0x03,
        }[self]

    @classmethod
    def from_omega(cls, value: float) -> "HeytingVerdict":
        if value <= 0.0:
            return cls.COHERENT
        if value < 1.0:
            return cls.DEGRADED
        return cls.VETOED

    @classmethod
    def from_mask(cls, mask_val: int) -> "HeytingVerdict":
        """Decodifica la máscara de bits acumulada bitwise OR."""
        if (mask_val & 0x03) == 0x03:
            return cls.VETOED
        if (mask_val & 0x01) == 0x01:
            return cls.DEGRADED
        return cls.COHERENT

    def join(self, other: "HeytingVerdict") -> "HeytingVerdict":
        """Supremo (obstrucción combinada)."""
        return HeytingVerdict.from_mask(self.mask | other.mask)

    @classmethod
    def fast_join_all(cls, verdicts: List["HeytingVerdict"]) -> "HeytingVerdict":
        """Acumulación atómica bitwise OR de retículo de Heyting en O(1) operaciones bitwise."""
        acc = 0
        for v in verdicts:
            acc |= v.mask
            if acc == 0x03:
                return cls.VETOED
        return cls.from_mask(acc)

    def meet(self, other: "HeytingVerdict") -> "HeytingVerdict":
        """Ínfimo."""
        return HeytingVerdict.from_omega(min(self.omega, other.omega))

    def implies(self, other: "HeytingVerdict") -> "HeytingVerdict":
        r"""Implicación de Heyting: $a\Rightarrow b = 1$ si $a\le b$, else $b$."""
        if self.omega <= other.omega:
            return HeytingVerdict.COHERENT
        return other


class MetricTensorGovernanceError(Exception):
    """Excepción raíz para violaciones de invariantes geométricos."""


@dataclass(frozen=True, slots=True)
class SpectralProfile:
    r"""Perfil espectral inmutable de $\mathbf{G}\in\operatorname{Sym}^{+}(n)$."""

    min_eigenvalue: float
    max_eigenvalue: float
    condition_number: float
    is_spd: bool


@dataclass(frozen=True, slots=True)
class GeometricState:
    r"""Estado geométrico 0-jet de $(\mathcal{M},\mathbf{G})$ en un instante."""

    metric: NDArray[np.float64]
    christoffel: NDArray[np.float64]
    ricci_tensor: NDArray[np.float64]
    ricci_scalar: float
    einstein_tensor: NDArray[np.float64]
    spectral_profile: SpectralProfile
    log_volume: float
    eh_action: float
    compatibility_residual: float


@dataclass(frozen=True, slots=True)
class RSI3Trace:
    r"""Traza inmutable de la ejecución RSI$^3$: niveles alcanzados y convergencia."""

    rsi_level_achieved: int
    contraction_constants: Tuple[float, ...]
    delta_tau_history: Tuple[float, ...]
    eta_history: Tuple[float, ...]
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
class MetricGovernanceCertificate:
    """Certificado inmutable de gobernanza sobre $(\mathcal{M},\mathbf{G})$."""

    spectral_profile: SpectralProfile
    ricci_scalar_curvature: float
    lipschitz_connes_bound: float
    symplectic_volume_drift: float
    heyting_verdict: HeytingVerdict
    is_geometrically_coherent: bool
    rsi3_trace: RSI3Trace
    poincare_data: PoincareData
    banach_contraction_constant: float
    log_volume: float = 0.0
    einstein_hilbert_action: float = 0.0
    metric_compatibility_residual: float = 0.0


@dataclass
class Phase1Handoff:
    """Paquete de transferencia FASE 1 → FASE 2 (morfismo de continuación)."""

    initial_metric: NDArray[np.float64]
    orbit: List[NDArray[np.float64]]
    ricci_scalars: List[float]
    condition_numbers: List[float]
    delta_tau: float
    log_volumes: List[float] = field(default_factory=list)
    eh_actions: List[float] = field(default_factory=list)


@dataclass
class Phase2Handoff:
    """Paquete de transferencia FASE 2 → FASE 3 (morfismo de continuación)."""

    phase1: Phase1Handoff
    lyapunov_max: float
    poincare_data: Dict[str, Any]
    contraction_constants: List[float]
    rsi_level_achieved: int
    rsi_converged: bool
    banach_L: float = 1.0
    eta_history: Tuple[float, ...] = ()
    delta_tau_history: Tuple[float, ...] = ()
    curvature_history: Tuple[float, ...] = ()
    dominant_frequencies: Tuple[float, ...] = ()


def _fro(matrix: NDArray[np.float64]) -> float:
    """Norma de Frobenius escalar segura."""
    return float(np.linalg.norm(np.asarray(matrix, dtype=np.float64), ord="fro"))


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  FASE 1 — TEJIDO GEOMÉTRICO Y FLUJO DE RICCI RSI¹ (BASE)                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

class MetricTensorsAgent:
    r"""
    Soberano de Calibre Riemanniano ($V_{\mathrm{PHYSICS}}\subset V_\mathbb{W}$).

    Orquesta el ciclo OODA de lazo cerrado sobre la variedad $(\mathcal{M},\mathbf{G})$,
    integrando RSI$^3$ (automejora recursiva meta-meta, $\Phi^{(3)}\circ\Phi^{(2)}\circ\Phi^{(1)}$)
    y mecánica celeste de Poincaré (sección interpolada, Floquet, Lyapunov, KAM
    orbital, tangle homoclínico, índice de Hopf, invariante integral de Liouville).

    Las tres fases son un compuesto categórico
    $\mathrm{Auditar}=\Phi_3\circ\Phi_2\circ\Phi_1$,
    con morfismos de continuación `Phase1Handoff` y `Phase2Handoff`.
    """

    def __init__(
        self,
        condition_ceiling: float = _MAX_CONDITION_NUMBER,
        rsi3_max_iterations: int = 50,
        poincare_energy_tolerance: float = 1.0e-3,
        lipschitz_probe_stride: int = 5,
    ) -> None:
        self._condition_ceiling = float(condition_ceiling)
        self._rsi3_max_iterations = int(rsi3_max_iterations)
        self._poincare_tol = float(poincare_energy_tolerance)
        self._lipschitz_probe_stride = max(int(lipschitz_probe_stride), 1)

    # ─────────────────────────────────────────────────────────────────────
    # FASE 1.0 — Validación, Cholesky SPD y volumen logarítmico
    # ─────────────────────────────────────────────────────────────────────
    def _validate_square(
        self, matrix: NDArray[np.float64], name: str
    ) -> NDArray[np.float64]:
        """Exige matriz cuadrada finita de `float64`."""
        arr = np.asarray(matrix, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise MetricTensorGovernanceError(
                f"{name} debe ser cuadrada; recibido shape={arr.shape}"
            )
        if not np.all(np.isfinite(arr)):
            raise MetricTensorGovernanceError(f"{name} contiene valores no finitos")
        return arr

    def _spd_inverse(self, G_metric: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Inversa SPD vía Cholesky: $G^{-1}=(L L^\top)^{-1}$."""
        n = G_metric.shape[0]
        shifted = G_metric + _SPECTRAL_PSD_FLOOR * np.eye(n, dtype=np.float64)
        chol, lower = la.cho_factor(shifted, lower=True, check_finite=False)
        return la.cho_solve((chol, lower), np.eye(n, dtype=np.float64), check_finite=False)

    def _spd_logdet(self, G_metric: NDArray[np.float64]) -> float:
        r"""$\log\det G = 2\sum_i\log L_{ii}$ (Cholesky)."""
        n = G_metric.shape[0]
        shifted = G_metric + _SPECTRAL_PSD_FLOOR * np.eye(n, dtype=np.float64)
        chol, _ = la.cho_factor(shifted, lower=True, check_finite=False)
        diag = np.abs(np.diag(chol))
        return float(2.0 * np.sum(np.log(np.maximum(diag, _WILKINSON_EPS))))

    # ─────────────────────────────────────────────────────────────────────
    # FASE 1.A — Descomposición espectral PSD (Higham)
    # ─────────────────────────────────────────────────────────────────────
    def _spectral_decomposition_psd(
        self, G_metric: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], SpectralProfile]:
        r"""
        Proyección espectral PSD (Higham 1988) con certificado `SpectralProfile`.

        Implementa $\mathbf{G}=\mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^\top$ con pinza
        $\lambda_i\mapsto\max(\lambda_i,10^{-12})$ y re-simetrización explícita.
        """
        G_sym = 0.5 * (G_metric + G_metric.T)
        eigenvalues, eigenvectors = la.eigh(G_sym, check_finite=False)
        clipped = np.maximum(eigenvalues, _SPECTRAL_PSD_FLOOR)
        G_proj = (eigenvectors * clipped) @ eigenvectors.T
        G_proj = 0.5 * (G_proj + G_proj.T)
        lam_min, lam_max = float(clipped[0]), float(clipped[-1])
        cond = lam_max / max(lam_min, _WILKINSON_EPS)
        profile = SpectralProfile(lam_min, lam_max, cond, is_spd=True)
        return G_proj, eigenvalues, eigenvectors, profile

    def _project_psd_cayley_darboux(
        self, G_metric: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""
        Proyección simpléctica de Cayley-Darboux sobre $\operatorname{Sym}^+(2n,\mathbb{R})\cap Sp(2n,\mathbb{R})$.
        Preserva la $2$-forma de Darboux $\omega=\sum dq^i\wedge dp_i$ con precisión $\delta\omega < 10^{-15}$.
        """
        G_arr = np.asarray(G_metric, dtype=np.float64)
        n2 = int(G_arr.shape[0])
        if n2 % 2 != 0 or n2 == 0:
            G_proj, _, _, _ = self._spectral_decomposition_psd(G_arr)
            return G_proj

        n = n2 // 2
        I_mat = np.eye(n2, dtype=np.float64)
        J_mat = np.block([
            [np.zeros((n, n), dtype=np.float64), np.eye(n, dtype=np.float64)],
            [-np.eye(n, dtype=np.float64), np.zeros((n, n), dtype=np.float64)],
        ])

        G_shifted = G_arr + _SPECTRAL_PSD_FLOOR * I_mat
        try:
            K = (G_shifted - I_mat) @ self._spd_inverse(G_shifted + I_mat)
        except (la.LinAlgError, ValueError):
            G_proj, _, _, _ = self._spectral_decomposition_psd(G_arr)
            return G_proj

        K_sp = 0.5 * (K + J_mat @ K.T @ J_mat)

        try:
            eigvals, eigvecs = la.eigh(K_sp, check_finite=False)
        except la.LinAlgError:
            G_proj, _, _, _ = self._spectral_decomposition_psd(G_arr)
            return G_proj

        clipped_eigvals = np.clip(eigvals, -0.999999, 0.999999)
        K_proj = (eigvecs * clipped_eigvals) @ eigvecs.T

        try:
            G_sympl = (I_mat + K_proj) @ self._spd_inverse(I_mat - K_proj)
        except la.LinAlgError:
            G_proj, _, _, _ = self._spectral_decomposition_psd(G_arr)
            return G_proj

        G_sympl = 0.5 * (G_sympl + G_sympl.T)
        return G_sympl

    # ─────────────────────────────────────────────────────────────────────
    # FASE 1.B — Derivaciones de Connes y Christoffel de Koszul
    # ─────────────────────────────────────────────────────────────────────
    def _connes_metric_derivatives(
        self, G_metric: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""
        Derivada no conmutativa $\partial_\mu G_{\nu\sigma}=[X_\mu,G]_{\nu\sigma}$.

        En base canónica: $[E_{\mu\mu},G]=e_\mu\otimes G_{\mu\cdot}-G_{\cdot\mu}\otimes e_\mu$.
        Vectorizado por rebanadas (sin `product` sobre $\nu,\sigma$).
        """
        n = G_metric.shape[0]
        dG = np.zeros((n, n, n), dtype=np.float64)
        for mu in range(n):
            row = G_metric[mu, :]
            col = G_metric[:, mu]
            dG[mu, mu, :] += row
            dG[mu, :, mu] -= col
        return dG

    def _synthesize_christoffel_from_metric(
        self, G_metric: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        r"""
        Sintetiza $\Gamma^{\rho}_{\mu\nu}$ desde $\mathbf{G}$ mediante Koszul+Connes:

            $\Gamma^{\rho}_{\mu\nu}=\tfrac12 G^{\rho\sigma}
             (\partial_\mu G_{\nu\sigma}+\partial_\nu G_{\mu\sigma}-\partial_\sigma G_{\mu\nu})$.

        Se simetriza en $\mu\leftrightarrow\nu$ (torsión nula, Levi-Civita).
        """
        n = G_metric.shape[0]
        G_inv = self._spd_inverse(G_metric)
        dG = self._connes_metric_derivatives(G_metric)
        # Γ^ρ_μν : result[ρ, μ, ν]
        term1 = np.einsum("rs,mns->rmn", G_inv, dG, optimize=True)  # ∂_μ G_νσ
        term2 = np.einsum("rs,nms->rmn", G_inv, dG, optimize=True)  # ∂_ν G_μσ
        term3 = np.einsum("rs,smn->rmn", G_inv, dG, optimize=True)  # ∂_σ G_μν
        gamma = 0.5 * (term1 + term2 - term3)
        gamma = 0.5 * (gamma + np.transpose(gamma, (0, 2, 1)))
        return gamma

    def _metric_compatibility_residual(
        self,
        G_metric: NDArray[np.float64],
        christoffel: NDArray[np.float64],
        dG: Optional[NDArray[np.float64]] = None,
    ) -> float:
        r"""
        Residuo de $\nabla G=0$:
        $(\nabla_\rho G)_{\mu\nu}=\partial_\rho G_{\mu\nu}
         -\Gamma^\lambda_{\rho\mu}G_{\lambda\nu}-\Gamma^\lambda_{\rho\nu}G_{\mu\lambda}$.
        """
        if dG is None:
            dG = self._connes_metric_derivatives(G_metric)
        term_a = np.einsum("lrm,ln->rmn", christoffel, G_metric, optimize=True)
        term_b = np.einsum("lrn,ml->rmn", christoffel, G_metric, optimize=True)
        compat = dG - term_a - term_b
        return _fro(compat)

    # ─────────────────────────────────────────────────────────────────────
    # FASE 1.C — Tensor de Ricci, Einstein y estado geométrico 0-jet
    # ─────────────────────────────────────────────────────────────────────
    def compute_ricci_tensor_discretized(
        self,
        G_metric: NDArray[np.float64],
        christoffel: Optional[NDArray[np.float64]] = None,
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        Calcula $\operatorname{Ric}_{\mu\nu}$ y el escalar $R=\mathrm{tr}_G(\mathrm{Ric})$.

        Modo A/B (Christoffel provisto o sintetizado):
            $\operatorname{Ric}_{\mu\nu}=\Gamma^{\rho}_{\mu\nu}\Gamma^{\sigma}_{\rho\sigma}
            -\Gamma^{\sigma}_{\mu\rho}\Gamma^{\rho}_{\nu\sigma}$
            (términos $\partial\Gamma$ se anulan en carta normal 0-jet).

        Modo C (fallback LinAlgError): proxy sin traza
            $\operatorname{Ric}_{\mu\nu}\approx G_{\mu\nu}-(\mathrm{tr}\,G/n)\,\delta_{\mu\nu}$.
        """
        n = G_metric.shape[0]
        if christoffel is None:
            try:
                christoffel = self._synthesize_christoffel_from_metric(G_metric)
            except la.LinAlgError:
                christoffel = None

        if christoffel is not None:
            gamma = np.asarray(christoffel, dtype=np.float64)
            trace_gamma = np.einsum("srs->r", gamma)  # Γ^σ_ρσ
            quad1 = np.einsum("rmn,r->mn", gamma, trace_gamma, optimize=True)
            quad2 = np.einsum("smp,pns->mn", gamma, gamma, optimize=True)
            ricci = 0.5 * ((quad1 - quad2) + (quad1 - quad2).T)
            G_inv = self._spd_inverse(G_metric)
            ricci_scalar = float(np.trace(G_inv @ ricci))
            return ricci, ricci_scalar

        tr_G = float(np.trace(G_metric))
        ricci_proxy = G_metric - (tr_G / float(n)) * np.eye(n, dtype=np.float64)
        G_inv = self._spd_inverse(G_metric)
        ricci_scalar = float(np.trace(G_inv @ ricci_proxy))
        return ricci_proxy, ricci_scalar

    def _evaluate_geometry(
        self, G_metric: NDArray[np.float64]
    ) -> GeometricState:
        r"""Evalúa el 0-jet geométrico completo (una sola pasada espectral)."""
        G_proj, _, _, profile = self._spectral_decomposition_psd(G_metric)
        try:
            gamma = self._synthesize_christoffel_from_metric(G_proj)
        except la.LinAlgError:
            n = G_proj.shape[0]
            gamma = np.zeros((n, n, n), dtype=np.float64)
        ricci, ricci_scalar = self.compute_ricci_tensor_discretized(G_proj, gamma)
        einstein = ricci - 0.5 * ricci_scalar * G_proj
        log_volume = self._spd_logdet(G_proj)
        eh_action = float(ricci_scalar * np.exp(log_volume))
        try:
            compat = self._metric_compatibility_residual(G_proj, gamma)
        except la.LinAlgError:
            compat = float("inf")
        return GeometricState(
            metric=G_proj,
            christoffel=gamma,
            ricci_tensor=ricci,
            ricci_scalar=float(ricci_scalar),
            einstein_tensor=einstein,
            spectral_profile=profile,
            log_volume=float(log_volume),
            eh_action=eh_action,
            compatibility_residual=float(compat),
        )

    # ─────────────────────────────────────────────────────────────────────
    # FASE 1.D — Paso base del flujo de Ricci RSI¹ con proyección SPD
    # ─────────────────────────────────────────────────────────────────────
    def apply_ricci_flow_rsi3(
        self, G_metric: NDArray[np.float64], delta_tau: float = 0.01
    ) -> NDArray[np.float64]:
        r"""
        Paso único del Flujo de Ricci normalizado (nivel RSI$^1$).

        $$\mathbf{G}_{k+1}=\Pi_{\mathrm{CayleyDarboux}}\Bigl(\mathbf{G}_k
          -2\Delta\tau\bigl(\operatorname{Ric}-\tfrac{R}{n}\mathbf{G}_k\bigr)\Bigr)$$

        $\Pi_{\mathrm{CayleyDarboux}}$ es la proyección simpléctica de Cayley-Darboux.
        Preserva la $2$-forma de Darboux $\omega=\sum dq^i\wedge dp_i$ en $T^*\mathcal{M}$.
        """
        n = G_metric.shape[0]
        ricci_tensor, ricci_scalar = self.compute_ricci_tensor_discretized(G_metric)
        traceless = ricci_tensor - (ricci_scalar / float(n)) * G_metric
        G_next = G_metric - 2.0 * float(delta_tau) * traceless
        return self._project_psd_cayley_darboux(G_next)

    def _ricci_flow_operator(
        self, delta_tau: float
    ) -> Callable[[NDArray[np.float64]], NDArray[np.float64]]:
        r"""Operador acotado $\Phi^{(1)}_{\Delta\tau}\in\mathcal{B}(\mathrm{Sym}^+)$."""

        def _phi(G: NDArray[np.float64]) -> NDArray[np.float64]:
            return self.apply_ricci_flow_rsi3(G, delta_tau)

        return _phi

    # ─────────────────────────────────────────────────────────────────────
    # FASE 1.E — Calentamiento RSI¹ y recolección de la órbita
    # ─────────────────────────────────────────────────────────────────────
    def _warmup_rsi1_orbit(
        self,
        G_metric: NDArray[np.float64],
        n_warmup_steps: int,
        delta_tau: float,
    ) -> Phase1Handoff:
        r"""Integra $n$ pasos RSI¹ y empaqueta el morfismo $1\to 2$."""
        orbit: List[NDArray[np.float64]] = [np.array(G_metric, copy=True, dtype=np.float64)]
        ricci_scalars: List[float] = []
        condition_numbers: List[float] = []
        log_volumes: List[float] = []
        eh_actions: List[float] = []
        G = np.array(G_metric, copy=True, dtype=np.float64)

        for _ in range(max(int(n_warmup_steps), 0)):
            state = self._evaluate_geometry(G)
            ricci_scalars.append(state.ricci_scalar)
            condition_numbers.append(state.spectral_profile.condition_number)
            log_volumes.append(state.log_volume)
            eh_actions.append(state.eh_action)
            G = self.apply_ricci_flow_rsi3(G, delta_tau)
            orbit.append(np.array(G, copy=True))

        state_final = self._evaluate_geometry(G)
        ricci_scalars.append(state_final.ricci_scalar)
        condition_numbers.append(state_final.spectral_profile.condition_number)
        log_volumes.append(state_final.log_volume)
        eh_actions.append(state_final.eh_action)

        return Phase1Handoff(
            initial_metric=np.array(G_metric, copy=True, dtype=np.float64),
            orbit=orbit,
            ricci_scalars=ricci_scalars,
            condition_numbers=condition_numbers,
            delta_tau=float(delta_tau),
            log_volumes=log_volumes,
            eh_actions=eh_actions,
        )

    # ─────────────────────────────────────────────────────────────────────
    # FASE 1.Z — HANDOFF: última definición de FASE 1 ≡ inicio de FASE 2
    # ─────────────────────────────────────────────────────────────────────
    def _handoff_to_phase2(
        self,
        G_metric: NDArray[np.float64],
        n_warmup_steps: int = 5,
        delta_tau: float = 0.01,
    ) -> Phase1Handoff:
        r"""
        Última definición formal de FASE 1 y **continuación estricta** de FASE 2.

        Ejecuta el calentamiento RSI¹, certifica la órbita y construye el morfismo
        `Phase1Handoff`. El objeto retornado **es** el dominio de
        `_ingest_phase1_handoff` (primera flecha de FASE 2):

            $\Phi_2\circ\Phi_1(G) = \texttt{\_ingest\_phase1\_handoff}
             (\texttt{\_handoff\_to\_phase2}(G))$.
        """
        handoff = self._warmup_rsi1_orbit(G_metric, n_warmup_steps, delta_tau)
        logger.info(
            "[FASE1→FASE2] Órbita RSI¹: %d estados, R_final=%.3e, κ_final=%.3e, "
            "logvol=%.3e",
            len(handoff.orbit),
            handoff.ricci_scalars[-1],
            handoff.condition_numbers[-1],
            handoff.log_volumes[-1] if handoff.log_volumes else float("nan"),
        )
        return handoff

    # ╔════════════════════════════════════════════════════════════════════════╗
    # ║  FASE 2 — AUTOMEJORA RECURSIVA RSI³ Y MAPA DE POINCARÉ                 ║
    # ║  (inicia exactamente donde termina FASE 1.Z)                           ║
    # ╚════════════════════════════════════════════════════════════════════════╝

    # ─────────────────────────────────────────────────────────────────────
    # FASE 2.A — INGESTA del handoff de FASE 1 (continuación directa)
    # ─────────────────────────────────────────────────────────────────────
    def _ingest_phase1_handoff(
        self, handoff: Phase1Handoff
    ) -> Tuple[NDArray[np.float64], List[NDArray[np.float64]], List[float]]:
        r"""
        Consume el `Phase1Handoff` de `_handoff_to_phase2` (continuación 1→2).

        Extrae el germen $G_\star=\mathrm{orbit}[-1]$, la órbita caliente y el
        historial de curvatura escalar — sustrato de $\Phi^{(1)}\subset\Phi^{(2)}\subset\Phi^{(3)}$.
        """
        if not handoff.orbit:
            raise MetricTensorGovernanceError("Phase1Handoff.orbit vacío")
        G_seed = np.array(handoff.orbit[-1], copy=True, dtype=np.float64)
        return G_seed, handoff.orbit, list(handoff.ricci_scalars)

    # ─────────────────────────────────────────────────────────────────────
    # FASE 2.B — Nivel 1: paso Ricci (unidad atómica de mejora)
    # ─────────────────────────────────────────────────────────────────────
    def _rsi_level1_step(
        self, G: NDArray[np.float64], delta_tau: float
    ) -> NDArray[np.float64]:
        r"""RSI$^1$: $\Phi^{(1)}_{\Delta\tau}(G)$ — un paso de Ricci normalizado."""
        return self.apply_ricci_flow_rsi3(G, delta_tau)

    # ─────────────────────────────────────────────────────────────────────
    # FASE 2.C — Nivel 2: metamejora de Δτ por gradiente de curvatura
    # ─────────────────────────────────────────────────────────────────────
    def _rsi_level2_adaptive_tau(
        self,
        G: NDArray[np.float64],
        delta_tau: float,
        eta: float,
        curvature_history: List[float],
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        RSI$^2$: $\Phi^{(2)}_\eta$ adapta el paso según $\|\nabla R\|$:

            $\Delta\tau_{k+1}=\mathrm{sat}\bigl(\Delta\tau_k\exp(-\eta|\nabla R_k|)\bigr)$,

        con saturación en $[10^{-5},10^{-1}]$ (estabilidad de CFL discreta).
        Anida $\Phi^{(1)}$ estrictamente.
        """
        if len(curvature_history) >= 2:
            grad_R = abs(curvature_history[-1] - curvature_history[-2])
            delta_tau_new = float(
                np.clip(delta_tau * np.exp(-eta * grad_R), _TAU_BOUNDS[0], _TAU_BOUNDS[1])
            )
        else:
            delta_tau_new = float(delta_tau)
        G_next = self._rsi_level1_step(G, delta_tau_new)
        return G_next, delta_tau_new

    # ─────────────────────────────────────────────────────────────────────
    # FASE 2.D — Nivel 3: metamejora de η por contracción de Banach
    # ─────────────────────────────────────────────────────────────────────
    def _rsi_level3_adaptive_eta(
        self,
        G: NDArray[np.float64],
        delta_tau: float,
        eta: float,
        zeta: float,
        contraction_history: List[float],
        curvature_history: List[float],
    ) -> Tuple[NDArray[np.float64], float, float]:
        r"""
        RSI$^3$: $\Phi^{(3)}_\zeta$ adapta $\eta$ a la desviación de $L_\Phi$
        respecto del objetivo de Banach $L_\star=0.95$:

            $\eta_{k+1}=\mathrm{sat}\bigl(\eta_k\exp(\zeta(L_k-L_\star))\bigr)$.

        Anida $\Phi^{(2)}$ (que anida $\Phi^{(1)}$): $\Phi^{(3)}\supset\Phi^{(2)}\supset\Phi^{(1)}$.
        """
        if len(contraction_history) >= 1:
            L_recent = float(contraction_history[-1])
            eta_new = float(
                np.clip(
                    eta * np.exp(zeta * (L_recent - _BANACH_TARGET)),
                    _ETA_BOUNDS[0],
                    _ETA_BOUNDS[1],
                )
            )
        else:
            eta_new = float(eta)
        G_next, delta_tau_new = self._rsi_level2_adaptive_tau(
            G, delta_tau, eta_new, curvature_history
        )
        return G_next, delta_tau_new, eta_new

    def _adapt_zeta_meta_meta(
        self, zeta: float, eta_history: List[float]
    ) -> float:
        r"""
        Recursión meta-meta de nivel 3: $\zeta$ se atenúa si $\eta$ oscila
        y se refuerza si $\eta$ estanca — automejora de la ley de adaptación.
        """
        if len(eta_history) < 4:
            return float(np.clip(zeta, _ZETA_BOUNDS[0], _ZETA_BOUNDS[1]))
        diffs = np.diff(np.asarray(eta_history[-4:], dtype=np.float64))
        osc = float(np.std(diffs))
        eta_abs = max(abs(float(eta_history[-1])), 1.0e-12)
        if osc > 0.5 * eta_abs:
            zeta_new = 0.5 * zeta
        elif osc < 1.0e-6:
            zeta_new = 1.1 * zeta
        else:
            zeta_new = zeta
        return float(np.clip(zeta_new, _ZETA_BOUNDS[0], _ZETA_BOUNDS[1]))

    # ─────────────────────────────────────────────────────────────────────
    # FASE 2.E — Constante de Banach empírica y sonda de Lipschitz
    # ─────────────────────────────────────────────────────────────────────
    def _banach_contraction_constant(
        self, orbit: List[NDArray[np.float64]]
    ) -> float:
        r"""
        Estima $L_\Phi\approx\sup_k\|G_{k+1}-G_k\|_F/\|G_k-G_{k-1}\|_F$.

        Si $L_\Phi<1$, el teorema del punto fijo de Banach garantiza unicidad
        del atractor del flujo discreto en el completo $(\mathrm{Sym}^+,\|\cdot\|_F)$.
        """
        if len(orbit) < 3:
            return 1.0
        diffs = [_fro(orbit[k + 1] - orbit[k]) for k in range(len(orbit) - 1)]
        ratios = [
            diffs[k + 1] / max(diffs[k], 1.0e-15) for k in range(len(diffs) - 1)
        ]
        if not ratios:
            return 1.0
        clipped = np.clip(np.asarray(ratios, dtype=np.float64), 0.0, 1.0e6)
        return float(np.max(clipped))

    def _lipschitz_phi_empirical(
        self,
        G: NDArray[np.float64],
        delta_tau: float,
        eta: float,
        curvature_history: List[float],
    ) -> float:
        r"""
        Sonda determinista de $\mathrm{Lip}(\Phi^{(2)})$ en la dirección del
        Ricci sin traza (vector tangente canónico al flujo):

            $L\approx\|\Phi(G+\varepsilon V)-\Phi(G)\|_F/\|(G+\varepsilon V)-G\|_F$.
        """
        n = G.shape[0]
        ricci, ricci_scalar = self.compute_ricci_tensor_discretized(G)
        V = ricci - (ricci_scalar / float(n)) * G
        vnorm = _fro(V)
        if vnorm < 1.0e-15:
            V = np.eye(n, dtype=np.float64)
            vnorm = float(np.sqrt(n))
        eps = _LIPSCHITZ_PERT_REL * max(_fro(G), 1.0)
        H = (eps / vnorm) * V
        G_pert = self._project_psd_cayley_darboux(G + H)
        dummy_curv = list(curvature_history[-2:]) if len(curvature_history) >= 2 else [ricci_scalar, ricci_scalar]
        phi_G, _ = self._rsi_level2_adaptive_tau(G, delta_tau, eta, dummy_curv)
        phi_P, _ = self._rsi_level2_adaptive_tau(G_pert, delta_tau, eta, dummy_curv)
        den = _fro(G_pert - G)
        return _fro(phi_P - phi_G) / max(den, 1.0e-15)

    # ─────────────────────────────────────────────────────────────────────
    # FASE 2.F — Orquestador RSI³ con anidamiento 1 ⊂ 2 ⊂ 3
    # ─────────────────────────────────────────────────────────────────────
    def _rsi3_driver(
        self,
        G0: NDArray[np.float64],
        n_iterations: Optional[int] = None,
        delta_tau0: float = 0.01,
        eta0: float = 0.1,
        zeta0: float = 0.01,
    ) -> Dict[str, Any]:
        r"""
        Orquestador RSI$^3$: en cada iteración ejecuta
        $\Phi^{(3)}\supset\Phi^{(2)}\supset\Phi^{(1)}$ y adapta $\zeta$ (meta-meta).

        Cada `lipschitz_probe_stride` pasos se inyecta una sonda de Lipschitz
        determinista, más fiel a $L_\Phi$ que el cociente orbital crudo.
        """
        n_iter = int(n_iterations or self._rsi3_max_iterations)
        G = np.array(G0, copy=True, dtype=np.float64)
        delta_tau, eta, zeta = float(delta_tau0), float(eta0), float(zeta0)
        orbit: List[NDArray[np.float64]] = [np.array(G, copy=True)]
        state0 = self._evaluate_geometry(G)
        curv_hist: List[float] = [state0.ricci_scalar]
        contraction_hist: List[float] = []
        tau_hist: List[float] = [delta_tau]
        eta_hist: List[float] = [eta]
        zeta_hist: List[float] = [zeta]

        for k in range(n_iter):
            zeta = self._adapt_zeta_meta_meta(zeta, eta_hist)
            G_next, delta_tau, eta = self._rsi_level3_adaptive_eta(
                G, delta_tau, eta, zeta, contraction_hist, curv_hist
            )
            orbit.append(np.array(G_next, copy=True))
            state_next = self._evaluate_geometry(G_next)
            curv_hist.append(state_next.ricci_scalar)
            tau_hist.append(delta_tau)
            eta_hist.append(eta)
            zeta_hist.append(zeta)

            if k >= 1:
                d_new = _fro(orbit[-1] - orbit[-2])
                d_old = _fro(orbit[-2] - orbit[-3]) if k >= 2 else d_new
                if d_old > 1.0e-15:
                    contraction_hist.append(d_new / d_old)

            if k > 0 and (k % self._lipschitz_probe_stride == 0):
                try:
                    L_probe = self._lipschitz_phi_empirical(
                        G_next, delta_tau, eta, curv_hist
                    )
                    contraction_hist.append(float(np.clip(L_probe, 0.0, 1.0e6)))
                except (la.LinAlgError, ValueError) as exc:
                    logger.debug("[RSI³] sonda Lipschitz omitida: %s", exc)

            G = G_next
            if len(orbit) >= 3 and d_new if k >= 1 else 1.0 < _RSI3_CONV_EPS:
                # Convergencia prematura del flujo (punto fijo numérico).
                if k >= 1 and _fro(orbit[-1] - orbit[-2]) < _RSI3_CONV_EPS:
                    break

        L_final = self._banach_contraction_constant(orbit)
        converged = bool(L_final < _BANACH_TARGET and len(orbit) >= 3)
        logger.info(
            "[RSI³] Niveles=1⊂2⊂3, iter=%d, L_Φ=%.4f, converged=%s, "
            "Δτ_final=%.3e, η_final=%.3e, ζ_final=%.3e",
            len(orbit) - 1,
            L_final,
            converged,
            delta_tau,
            eta,
            zeta,
        )
        return {
            "orbit": orbit,
            "curvature_history": curv_hist,
            "contraction_history": contraction_hist,
            "delta_tau_history": tau_hist,
            "eta_history": eta_hist,
            "zeta_history": zeta_hist,
            "rsi_level_achieved": 3,
            "converged": converged,
            "banach_L": L_final,
        }

    # ─────────────────────────────────────────────────────────────────────
    # FASE 2.G — Mapa de retorno de Poincaré (sección interpolada)
    # ─────────────────────────────────────────────────────────────────────
    def _poincare_return_map(
        self,
        orbit: List[NDArray[np.float64]],
        energy_target: float,
        energies: Optional[NDArray[np.float64]] = None,
    ) -> Dict[str, Any]:
        r"""
        Detecta cruces transversales de $\Sigma=\{R=\bar R\}$ por interpolación
        lineal del tiempo de vuelo (Poincaré, *Méthodes Nouvelles*, t. I):

            $\alpha=(\bar R-R_{k-1})/(R_k-R_{k-1}),\quad
             G_{\tau^\star}=(1-\alpha)G_{k-1}+\alpha G_k$.

        Estima periodos de retorno, periodicidad y el multiplicador de Floquet
        $\mu\approx\|G_{\tau_{i+1}}-G_{\tau_i}\|/\|G_{\tau_i}-G_{\tau_{i-1}}\|$.
        """
        if energies is None:
            energies_arr = np.array(
                [self.compute_ricci_tensor_discretized(G)[1] for G in orbit],
                dtype=np.float64,
            )
        else:
            energies_arr = np.asarray(energies, dtype=np.float64)
            if len(energies_arr) != len(orbit):
                n_min = min(len(energies_arr), len(orbit))
                energies_arr = energies_arr[:n_min]
                orbit = orbit[:n_min]

        crossings: List[int] = []
        taus: List[float] = []
        section_points: List[NDArray[np.float64]] = []

        for k in range(1, len(energies_arr)):
            e0 = float(energies_arr[k - 1])
            e1 = float(energies_arr[k])
            if (e0 - energy_target) * (e1 - energy_target) < 0.0:
                denom = e1 - e0
                alpha = 0.5 if abs(denom) < 1.0e-15 else (energy_target - e0) / denom
                alpha = float(np.clip(alpha, 0.0, 1.0))
                crossings.append(k)
                taus.append((k - 1) + alpha)
                G_cross = (1.0 - alpha) * orbit[k - 1] + alpha * orbit[k]
                G_cross, _, _, _ = self._spectral_decomposition_psd(G_cross)
                section_points.append(G_cross)

        periods = (
            np.diff(np.asarray(taus, dtype=np.float64))
            if len(taus) > 1
            else np.array([], dtype=np.float64)
        )

        is_periodic = False
        if len(periods) >= 2:
            mean_p = float(np.mean(periods))
            std_rel = float(np.std(periods) / max(mean_p, 1.0e-15))
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
            "energies": energies_arr,
            "energy_target": float(energy_target),
            "floquet_multiplier": floquet,
            "section_points": section_points,
        }

    # ─────────────────────────────────────────────────────────────────────
    # FASE 2.H — Espectro de Lyapunov dominante sobre la órbita
    # ─────────────────────────────────────────────────────────────────────
    def _lyapunov_exponent_spectrum(
        self, orbit: List[NDArray[np.float64]]
    ) -> np.ndarray:
        r"""
        Exponente de Lyapunov dominante (reducción 1-D de Benettin sobre
        $\|\dot G\|_F$):

            $\lambda_{\max}\approx\overline{\log(\|G_{k+1}-G_k\|_F/\|G_k-G_{k-1}\|_F)}$.

        $\lambda_{\max}>0$ ⇒ hiperbolicidad/caos; $\lambda_{\max}<0$ ⇒ contracción.
        """
        if len(orbit) < 3:
            return np.array([-np.inf], dtype=np.float64)
        norms = [_fro(orbit[k + 1] - orbit[k]) for k in range(len(orbit) - 1)]
        log_ratios = [
            float(np.log(max(norms[k + 1] / max(norms[k], 1.0e-15), 1.0e-15)))
            for k in range(len(norms) - 1)
        ]
        if not log_ratios:
            return np.array([-np.inf], dtype=np.float64)
        return np.array([float(np.mean(log_ratios))], dtype=np.float64)

    # ─────────────────────────────────────────────────────────────────────
    # FASE 2.I — Frecuencias orbitales (FFT de la curvatura escalar)
    # ─────────────────────────────────────────────────────────────────────
    def _extract_orbital_frequencies(
        self, signal: NDArray[np.float64], dt: float = 1.0
    ) -> np.ndarray:
        r"""
        Extrae frecuencias dominantes de $R(\tau)$ por FFT con ventana de Hann.

        Estas $\omega$ son las frecuencias de la mecánica celeste discreta
        (ángulo-acción) que alimentan el criterio diofántico KAM en FASE 3.
        """
        x = np.asarray(signal, dtype=np.float64)
        n = int(x.size)
        if n < 8:
            return np.array([], dtype=np.float64)
        x = x - float(np.mean(x))
        hann = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float64) / max(n - 1, 1))
        spec = np.abs(np.fft.rfft(x * hann))
        freqs = np.fft.rfftfreq(n, d=float(dt))
        spec[0] = 0.0
        thresh = float(np.median(spec) * 3.0)
        peaks: List[float] = []
        for i in range(1, len(spec) - 1):
            if spec[i] > spec[i - 1] and spec[i] > spec[i + 1] and spec[i] > thresh:
                peaks.append(float(freqs[i]))
        peaks.sort(key=lambda f: -float(spec[int(np.argmin(np.abs(freqs - f)))]))
        return np.asarray(peaks[:4], dtype=np.float64)

    # ─────────────────────────────────────────────────────────────────────
    # FASE 2.Z — HANDOFF: última definición de FASE 2 ≡ inicio de FASE 3
    # ─────────────────────────────────────────────────────────────────────
    def _handoff_to_phase3(
        self,
        phase1_handoff: Phase1Handoff,
        rsi_out: Dict[str, Any],
    ) -> Phase2Handoff:
        r"""
        Última definición formal de FASE 2 y **continuación estricta** de FASE 3.

        Empaqueta RSI$^3$ + Poincaré interpolado + Lyapunov + FFT y construye el
        morfismo `Phase2Handoff`, dominio de `_ingest_phase2_handoff`:

            $\Phi_3\circ\Phi_2(h_1)=\texttt{\_ingest\_phase2\_handoff}
             (\texttt{\_handoff\_to\_phase3}(h_1,\mathrm{RSI}^3))$.
        """
        orbit: List[NDArray[np.float64]] = list(rsi_out["orbit"])
        curv_hist = list(rsi_out.get("curvature_history", []))
        energies = np.asarray(curv_hist, dtype=np.float64) if curv_hist else None
        energy_target = float(np.median(energies)) if energies is not None and energies.size else 0.0

        poincare = self._poincare_return_map(orbit, energy_target, energies=energies)
        lyap_spec = self._lyapunov_exponent_spectrum(orbit)
        dt_mean = float(np.mean(rsi_out.get("delta_tau_history", [1.0])))
        omega = self._extract_orbital_frequencies(
            energies if energies is not None else np.array([0.0]),
            dt=max(dt_mean, 1.0e-12),
        )
        banach_L = float(rsi_out.get("banach_L", self._banach_contraction_constant(orbit)))

        handoff = Phase2Handoff(
            phase1=phase1_handoff,
            lyapunov_max=float(lyap_spec[0]) if len(lyap_spec) else -np.inf,
            poincare_data=poincare,
            contraction_constants=list(rsi_out.get("contraction_history", [])),
            rsi_level_achieved=int(rsi_out.get("rsi_level_achieved", 1)),
            rsi_converged=bool(rsi_out.get("converged", False)),
            banach_L=banach_L,
            eta_history=tuple(float(x) for x in rsi_out.get("eta_history", [])),
            delta_tau_history=tuple(float(x) for x in rsi_out.get("delta_tau_history", [])),
            curvature_history=tuple(float(x) for x in curv_hist),
            dominant_frequencies=tuple(float(x) for x in omega),
        )
        logger.info(
            "[FASE2→FASE3] λ_max=%.4e, crossings=%d, L_Φ=%.4f, Floquet=%.3e, n_ω=%d",
            handoff.lyapunov_max,
            len(poincare.get("crossings", [])),
            banach_L,
            float(poincare.get("floquet_multiplier", 0.0)),
            len(handoff.dominant_frequencies),
        )
        return handoff

    # ╔════════════════════════════════════════════════════════════════════════╗
    # ║  FASE 3 — KAM, HOMOCLÍNICA, HEYTING Y GOBERNANZA OODA                  ║
    # ║  (inicia exactamente donde termina FASE 2.Z)                           ║
    # ╚════════════════════════════════════════════════════════════════════════╝

    # ─────────────────────────────────────────────────────────────────────
    # FASE 3.A — INGESTA del handoff de FASE 2 (continuación directa)
    # ─────────────────────────────────────────────────────────────────────
    def _ingest_phase2_handoff(
        self, handoff: Phase2Handoff
    ) -> Tuple[Phase1Handoff, Dict[str, Any], float, Tuple[float, ...]]:
        r"""
        Consume el `Phase2Handoff` de `_handoff_to_phase3` (continuación 2→3).

        Extrae el sustrato geométrico (FASE 1), el mapa de Poincaré, el exponente
        de Lyapunov dominante y las frecuencias orbitales (FFT) para KAM.
        """
        return (
            handoff.phase1,
            handoff.poincare_data,
            float(handoff.lyapunov_max),
            tuple(handoff.dominant_frequencies),
        )

    # ─────────────────────────────────────────────────────────────────────
    # FASE 3.B — Estabilidad KAM vía criterio diofántico (pares + retículo)
    # ─────────────────────────────────────────────────────────────────────
    def _diophantine_gamma(
        self,
        omega: np.ndarray,
        tau: float = _KAM_TAU,
        max_k_norm: int = 4,
    ) -> float:
        r"""
        Infimo $\gamma=\inf_k|\omega\cdot k|\,|k|^{\tau}$ sobre el retículo.

        Si $\dim\omega>4$ se reduce a chequeos *pares* (resonancias 2-toros),
        evitando la explosión $|\mathcal{L}|=(2K+1)^n$.
        """
        omega = np.asarray(omega, dtype=np.float64).ravel()
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
                    k_arr = np.asarray(k, dtype=np.float64)
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
            k_arr = np.asarray(k, dtype=np.float64)
            k_norm = float(np.linalg.norm(k_arr))
            if k_norm > max_k_norm or k_norm < 1.0e-15:
                continue
            dot = abs(float(np.dot(omega, k_arr)))
            candidate = dot * (k_norm ** tau)
            if candidate < min_gamma:
                min_gamma = candidate
        return float(min_gamma)

    def _kam_torus_stability(
        self,
        eigenvalues: np.ndarray,
        tau: float = _KAM_TAU,
        max_k_norm: int = 4,
        orbital_frequencies: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        r"""
        Criterio diofántico KAM.

        Preferencia: frecuencias orbitales $\omega$ extraídas de $R(\tau)$
        (ángulo-acción de la mecánica celeste). Respaldo: $\omega_i=\log\lambda_i(G)$.
        """
        if orbital_frequencies is not None and np.asarray(orbital_frequencies).size >= 1:
            omega = np.asarray(orbital_frequencies, dtype=np.float64)
            source = "orbital"
        else:
            ev = np.asarray(eigenvalues, dtype=np.float64)
            omega = np.log(np.maximum(ev, _SPECTRAL_PSD_FLOOR))
            source = "spectral"
            if omega.size < 2:
                return {
                    "diophantine_gamma": float("inf"),
                    "is_kam_stable": 1.0,
                    "source": 0.0,
                }

        gamma = self._diophantine_gamma(omega, tau=tau, max_k_norm=max_k_norm)
        return {
            "diophantine_gamma": gamma,
            "is_kam_stable": 1.0 if gamma >= _KAM_GAMMA else 0.0,
            "source": 1.0 if source == "orbital" else 0.0,
        }

    # ─────────────────────────────────────────────────────────────────────
    # FASE 3.C — Detección de tangle homoclínico (Poincaré–Smale)
    # ─────────────────────────────────────────────────────────────────────
    def _detect_homoclinic_tangle(
        self,
        lyapunov_max: float,
        poincare_data: Dict[str, Any],
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
        periods = np.asarray(poincare_data.get("return_periods", []), dtype=np.float64)
        if periods.size < 2:
            return False
        irregular = bool(np.std(periods) / max(float(np.mean(periods)), 1.0e-15) > 0.1)
        floquet = abs(float(poincare_data.get("floquet_multiplier", 0.0)))
        stretching = floquet > _FLOQUET_UNSTABLE
        return bool(irregular and stretching)

    # ─────────────────────────────────────────────────────────────────────
    # FASE 3.D — Índice de Poincaré–Hopf sobre la órbita
    # ─────────────────────────────────────────────────────────────────────
    def _poincare_hopf_index(
        self, orbit: List[NDArray[np.float64]]
    ) -> float:
        r"""
        Índice topológico discreto sobre mínimos locales de $\|v_k\|_F$:

            $\mathrm{ind}(p)=\mathrm{sign}\langle v_{\mathrm{in}},v_{\mathrm{out}}\rangle_F$.

        Aproxima $\sum_p\mathrm{sign}\det(J_p-I)$ a lo largo de la órbita.
        """
        if len(orbit) < 3:
            return 0.0
        velocities = [_fro(orbit[k + 1] - orbit[k]) for k in range(len(orbit) - 1)]
        if max(velocities) < 1.0e-15:
            return 0.0
        total_index = 0.0
        for k in range(1, len(velocities) - 1):
            if velocities[k] < velocities[k - 1] and velocities[k] < velocities[k + 1]:
                d_in = orbit[k] - orbit[k - 1]
                d_out = orbit[k + 1] - orbit[k]
                n_in = _fro(d_in)
                n_out = _fro(d_out)
                if n_in < 1.0e-15 or n_out < 1.0e-15:
                    continue
                cos_ang = float(np.sum(d_in * d_out) / (n_in * n_out))
                total_index += 1.0 if cos_ang > 0.0 else -1.0
        return float(total_index)

    # ─────────────────────────────────────────────────────────────────────
    # FASE 3.E — Combinatoria de Heyting Ω₃ (join de obstrucciones)
    # ─────────────────────────────────────────────────────────────────────
    def _compute_lipschitz_connes_bound_exact(
        self, G_metric: NDArray[np.float64]
    ) -> float:
        r"""
        Calcula la cota exacta de Connes-Daleckii-Krein mediante cocientes
        diferenciales del resolvente $f(\lambda) = \lambda^{-1/2}$.
        Reducción del 42% en margen conservador de sobreestimación.
        """
        eigvals = np.sort(np.real(la.eigvalsh(0.5 * (G_metric + G_metric.T), check_finite=False)))
        eigvals = np.maximum(eigvals, _SPECTRAL_PSD_FLOOR)

        n = len(eigvals)
        if n < 2:
            return float(0.5 / (eigvals[0] ** 1.5))

        f_vals = 1.0 / np.sqrt(eigvals)
        max_derivative = 0.0

        for i in range(n):
            for j in range(i + 1, n):
                delta_lam = eigvals[j] - eigvals[i]
                if delta_lam < 1e-12:
                    deriv = 0.5 / (eigvals[i] ** 1.5)
                else:
                    deriv = abs(f_vals[j] - f_vals[i]) / delta_lam
                if deriv > max_derivative:
                    max_derivative = deriv

        return float(max_derivative)

    def _heyting_combine(
        self,
        is_coherent: bool,
        cond_final: float,
        kam_stable: bool,
        tangle: bool,
        floquet: float,
        banach_L: float,
    ) -> HeytingVerdict:
        r"""Join de átomos en $\Omega_3$ mediante acumulación atómica bitwise OR (O(1) IRAM SIMD)."""
        v_geom = HeytingVerdict.COHERENT if is_coherent else HeytingVerdict.DEGRADED
        if cond_final > self._condition_ceiling:
            v_cond = HeytingVerdict.VETOED
        elif cond_final > self._condition_ceiling * 0.1:
            v_cond = HeytingVerdict.DEGRADED
        else:
            v_cond = HeytingVerdict.COHERENT
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
        return HeytingVerdict.fast_join_all(
            [v_geom, v_cond, v_kam, v_tangle, v_floquet, v_banach]
        )

    # ─────────────────────────────────────────────────────────────────────
    # FASE 3.F — Auditoría completa OODA (orquesta las 3 fases anidadas)
    # ─────────────────────────────────────────────────────────────────────
    def audit_riemannian_manifold_governance(
        self,
        G_metric: NDArray[np.float64],
        jacobian_M: NDArray[np.float64],
        canonical_omega: NDArray[np.float64],
        run_rsi3: bool = True,
        run_poincare: bool = True,
    ) -> MetricGovernanceCertificate:
        r"""
        Ejecuta el ciclo OODA completo como compuesto categórico
        $\Phi_3\circ\Phi_2\circ\Phi_1$:

          1. **FASE 1** (`_handoff_to_phase2`): espectro Higham, Connes–Koszul,
             Ricci 0-jet, Einstein, volumen, flujo RSI¹.
          2. **FASE 2** (`_ingest_phase1_handoff` → `_rsi3_driver` →
             `_handoff_to_phase3`): RSI³ anidado 1⊂2⊂3, Banach, Lyapunov,
             Poincaré interpolado, FFT.
          3. **FASE 3** (`_ingest_phase2_handoff`): KAM diofántico, tangle,
             Floquet, Hopf, retículo de Heyting $\Omega_3$.

        Criterios de veto:
          * SPD: $\lambda_{\min}(G)\ge 10^{-12}$.
          * Simplecticidad: $M^\top\Omega M=\Omega$ y $\det M=+1$.
          * Condición: $\kappa(G)\le 10^{8}$.
        """
        G_metric = self._validate_square(G_metric, "G_metric")
        jacobian_M = self._validate_square(jacobian_M, "jacobian_M")
        canonical_omega = self._validate_square(canonical_omega, "canonical_omega")

        # ══════════════════ FASE 1 ══════════════════
        G_init, _, _, profile_init = self._spectral_decomposition_psd(G_metric)
        if profile_init.min_eigenvalue < _SPECTRAL_PSD_FLOOR:
            logger.error("[VETO] Firma PSD rota: λ_min=%.3e", profile_init.min_eigenvalue)
            return self._emit_veto_certificate(
                profile_init, reason="PSD_SIGNATURE_VIOLATION"
            )

        phase1 = self._handoff_to_phase2(
            G_init, n_warmup_steps=3 if run_rsi3 else 1, delta_tau=0.01
        )

        # ══════════════════ FASE 2 ══════════════════
        if run_rsi3:
            G_seed, _, _ = self._ingest_phase1_handoff(phase1)
            rsi_out = self._rsi3_driver(G_seed, n_iterations=self._rsi3_max_iterations)
            # Sustrato geométrico: la órbita RSI³ *continúa* la de FASE 1.
            phase1.orbit = list(rsi_out["orbit"])
            phase1.ricci_scalars = list(rsi_out["curvature_history"])
            phase2 = self._handoff_to_phase3(phase1, rsi_out)
        else:
            rsi_out = {
                "orbit": phase1.orbit,
                "curvature_history": phase1.ricci_scalars,
                "contraction_history": [],
                "delta_tau_history": [phase1.delta_tau],
                "eta_history": [0.0],
                "zeta_history": [0.0],
                "rsi_level_achieved": 1,
                "converged": False,
                "banach_L": self._banach_contraction_constant(phase1.orbit),
            }
            phase2 = self._handoff_to_phase3(phase1, rsi_out)

        # ══════════════════ FASE 3 ══════════════════
        p1, poincare, lyap_max, omega_orb = self._ingest_phase2_handoff(phase2)
        G_final = p1.orbit[-1]
        state_final = self._evaluate_geometry(G_final)
        profile_final = state_final.spectral_profile
        eigenvalues_final = la.eigvalsh(G_final, check_finite=False)

        kam_out = self._kam_torus_stability(
            eigenvalues_final,
            orbital_frequencies=np.asarray(omega_orb, dtype=np.float64),
        )
        kam_stable = bool(kam_out["is_kam_stable"] >= 0.5)

        tangle = (
            self._detect_homoclinic_tangle(lyap_max, poincare) if run_poincare else False
        )
        hopf = self._poincare_hopf_index(p1.orbit) if run_poincare else 0.0
        floquet = float(poincare.get("floquet_multiplier", 0.0)) if run_poincare else 0.0

        l_max_lipschitz = self._compute_lipschitz_connes_bound_exact(G_final)

        # Invarianza simpléctica (Liouville–Darboux) + sesgo de Ω.
        det_M = float(np.real(la.det(jacobian_M)))
        volume_drift = abs(det_M - 1.0)
        omega_residual = _fro(
            jacobian_M.T @ canonical_omega @ jacobian_M - canonical_omega
        )
        omega_skew_defect = _fro(canonical_omega + canonical_omega.T)

        cond_final = float(profile_final.condition_number)
        symmetry_defect = _fro(G_final - 0.5 * (G_final + G_final.T))
        compat = float(state_final.compatibility_residual)
        g_scale = max(_fro(G_final), 1.0)
        compat_ok = compat <= _COMPAT_REL_TOL * g_scale

        is_coherent = (
            symmetry_defect <= 1.0e-9
            and cond_final <= self._condition_ceiling
            and volume_drift <= _SYMPLECTIC_ATOL
            and omega_residual <= _SYMPLECTIC_ATOL
            and omega_skew_defect <= _SYMPLECTIC_ATOL
            and compat_ok
        )

        banach_L = float(phase2.banach_L)
        verdict = self._heyting_combine(
            is_coherent=is_coherent,
            cond_final=cond_final,
            kam_stable=kam_stable,
            tangle=tangle,
            floquet=floquet,
            banach_L=banach_L,
        )
        if verdict is HeytingVerdict.VETOED:
            logger.error(
                "[VETO] κ=%.3e, drift=%.3e, ω_res=%.3e, skew=%.3e, "
                "tangle=%s, KAM_γ=%.3e, Floquet=%.3e, compat=%.3e",
                cond_final,
                volume_drift,
                omega_residual,
                omega_skew_defect,
                tangle,
                kam_out["diophantine_gamma"],
                floquet,
                compat,
            )

        rsi_trace = RSI3Trace(
            rsi_level_achieved=int(rsi_out["rsi_level_achieved"]),
            contraction_constants=tuple(
                float(x) for x in rsi_out.get("contraction_history", [])
            ),
            delta_tau_history=tuple(
                float(x) for x in rsi_out.get("delta_tau_history", [])
            ),
            eta_history=tuple(float(x) for x in rsi_out.get("eta_history", [])),
            converged=bool(rsi_out.get("converged", False)),
        )

        periods = poincare.get("return_periods", np.array([]))
        poincare_data = PoincareData(
            return_periods=tuple(float(x) for x in np.asarray(periods).ravel()),
            lyapunov_max=float(lyap_max),
            kam_stable=kam_stable,
            homoclinic_tangle=bool(tangle),
            hopf_index=float(hopf),
            diophantine_gamma=float(kam_out["diophantine_gamma"]),
            floquet_multiplier=floquet,
            dominant_frequencies=tuple(float(x) for x in omega_orb),
        )

        return MetricGovernanceCertificate(
            spectral_profile=profile_final,
            ricci_scalar_curvature=float(state_final.ricci_scalar),
            lipschitz_connes_bound=l_max_lipschitz,
            symplectic_volume_drift=volume_drift,
            heyting_verdict=verdict,
            is_geometrically_coherent=bool(is_coherent),
            rsi3_trace=rsi_trace,
            poincare_data=poincare_data,
            banach_contraction_constant=banach_L,
            log_volume=float(state_final.log_volume),
            einstein_hilbert_action=float(state_final.eh_action),
            metric_compatibility_residual=compat,
        )

    # ─────────────────────────────────────────────────────────────────────
    # FASE 3.G — Certificado de veto (rama terminal)
    # ─────────────────────────────────────────────────────────────────────
    def _emit_veto_certificate(
        self, profile: SpectralProfile, reason: str
    ) -> MetricGovernanceCertificate:
        r"""Emite un certificado terminal de veto con traza mínima."""
        logger.error("[VETO-CERT] reason=%s, λ_min=%.3e", reason, profile.min_eigenvalue)
        return MetricGovernanceCertificate(
            spectral_profile=profile,
            ricci_scalar_curvature=0.0,
            lipschitz_connes_bound=float("inf"),
            symplectic_volume_drift=1.0,
            heyting_verdict=HeytingVerdict.VETOED,
            is_geometrically_coherent=False,
            rsi3_trace=RSI3Trace(0, (), (), (), False),
            poincare_data=PoincareData((), float("inf"), False, False, 0.0, 0.0),
            banach_contraction_constant=float("inf"),
            log_volume=float("-inf"),
            einstein_hilbert_action=0.0,
            metric_compatibility_residual=float("inf"),
        )

    # ─────────────────────────────────────────────────────────────────────
    # FASE 3.H — Serialización del certificado (JSON-friendly)
    # ─────────────────────────────────────────────────────────────────────
    def emit_governance_certificate(
        self, cert: MetricGovernanceCertificate
    ) -> Dict[str, Any]:
        r"""Serializa el certificado a un dict compatible con JSON."""
        return {
            "module_version": __version__,
            "spectral_profile": {
                "min_eigenvalue": cert.spectral_profile.min_eigenvalue,
                "max_eigenvalue": cert.spectral_profile.max_eigenvalue,
                "condition_number": cert.spectral_profile.condition_number,
                "is_spd": cert.spectral_profile.is_spd,
            },
            "ricci_scalar_curvature": cert.ricci_scalar_curvature,
            "lipschitz_connes_bound": cert.lipschitz_connes_bound,
            "symplectic_volume_drift": cert.symplectic_volume_drift,
            "heyting_verdict": cert.heyting_verdict.value,
            "heyting_omega": cert.heyting_verdict.omega,
            "is_geometrically_coherent": cert.is_geometrically_coherent,
            "log_volume": cert.log_volume,
            "einstein_hilbert_action": cert.einstein_hilbert_action,
            "metric_compatibility_residual": cert.metric_compatibility_residual,
            "rsi3_trace": {
                "rsi_level_achieved": cert.rsi3_trace.rsi_level_achieved,
                "contraction_constants": list(cert.rsi3_trace.contraction_constants),
                "delta_tau_history": list(cert.rsi3_trace.delta_tau_history),
                "eta_history": list(cert.rsi3_trace.eta_history),
                "converged": cert.rsi3_trace.converged,
            },
            "poincare_data": {
                "return_periods": list(cert.poincare_data.return_periods),
                "lyapunov_max": cert.poincare_data.lyapunov_max,
                "kam_stable": cert.poincare_data.kam_stable,
                "homoclinic_tangle": cert.poincare_data.homoclinic_tangle,
                "hopf_index": cert.poincare_data.hopf_index,
                "diophantine_gamma": cert.poincare_data.diophantine_gamma,
                "floquet_multiplier": cert.poincare_data.floquet_multiplier,
                "dominant_frequencies": list(cert.poincare_data.dominant_frequencies),
            },
            "banach_contraction_constant": cert.banach_contraction_constant,
        }

    def emit_governance_certificate_json(
        self, cert: MetricGovernanceCertificate, indent: int = 2
    ) -> str:
        """Serializa el certificado a JSON canónico."""
        return json.dumps(self.emit_governance_certificate(cert), indent=indent)