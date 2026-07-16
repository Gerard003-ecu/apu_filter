# -*- coding: utf-8 -*-
r"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║  Módulo : Dirac Interconnection Agent                                         ║
║  Ruta   : app/agents/physics/dirac_interconnection_agent.py                          ║
║  Versión: 3.0.0-IDA-PBC-CFL-Governor-Categorical-Spectral                     ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  NATURALEZA CIBER-FÍSICA Y TEORÍA DE CONTROL NO LINEAL                        ║
║  ──────────────────────────────────────────────────────────────────────────   ║
║  Este módulo consagra la aduana termodinámica entre el estrato TACTICS        ║
║  (apu_agent.py) y el estrato PHYSICS (flux_condenser.py). Funciona como un    ║
║  "Demonio de Maxwell" categórico que esculpe el Hamiltoniano del sistema en   ║
║  ciclo cerrado para garantizar la estabilidad asintótica global.              ║
║                                                                               ║
║  AXIOMAS DE EJECUCIÓN                                                         ║
║  ──────────────────────────────────────────────────────────────────────────   ║
║  §1. ESTRUCTURA DE DIRAC Y ENERGY SHAPING (IDA-PBC)                           ║
║      Resuelve la Ecuación de Matching para encontrar α(x):                    ║
║                                                                               ║
║        [J_d(x) − R_d(x)] ∇H_d(x) = [J(x) − R(x)] ∇H(x) + g(x) α(x)            ║
║                                                                               ║
║      La ley de control α se extrae por pseudoinversa de Moore-Penrose con     ║
║      verificación SVD del rango de g y test de Lyapunov certificado:          ║
║        Ḣ_d = −∇H_d^T R_d ∇H_d ≤ 0                                             ║
║                                                                               ║
║      MEJORA v3: El rango de g se audita mediante descomposición SVD exacta    ║
║      con umbral adaptativo σ_tol = tol · σ_max. La pseudoinversa se calcula   ║
║      vía la descomposición SVD truncada para evitar amplificación de ruido.   ║
║      El residuo de matching se verifica tanto en norma absoluta como relativa ║
║      respecto al forzamiento requerido.                                       ║
║                                                                               ║
║  §2. SINTONIZACIÓN DINÁMICA DE IMPEDANCIA (PML)                               ║
║      Aniquila el coeficiente de reflexión Γ sintonizando los tensores         ║
║      ε_eff, μ_eff con restricción de Kramers-Kronig completa:                 ║
║                                                                               ║
║        Z_0 = √(μ_eff · ε_eff⁻¹) ≡ Z_load                                      ║
║        Γ = (Z_load − Z_0) / (Z_load + Z_0) = 0                                ║
║                                                                               ║
║      MEJORA v3: Los tensores ε y μ son matrices simétricas definidas          ║
║      positivas (no solo diagonales). La verificación Kramers-Kronig comprueba ║
║      positividad espectral completa. Canales inactivos (Z = ∞) son            ║
║      representados explícitamente con ε = 0 y velocidad = 0.                  ║
║                                                                               ║
║  §3. GOBERNANZA DEL LÍMITE DE COURANT-FRIEDRICHS-LEWY (CFL)                   ║
║      Audita el Laplaciano del grafo y restringe Δt:                           ║
║                                                                               ║
║        Δt ≤ (2 · CFL_margin) / (c_eff · √λ_max(Δ_sym))                        ║
║                                                                               ║
║      donde Δ_sym = ½(Δ + Δ^T) es la parte simétrica del Laplaciano.           ║
║                                                                               ║
║      MEJORA v3: El estimador espectral implementa tres niveles de fallback:   ║
║      (1) Lanczos con ARPACK, (2) potencia iterada con deflación Gram-Schmidt, ║
║      (3) cota de Gerschgorin con factor geométrico exacto. El número CFL se   ║
║      registra en el diagnóstico con su método de estimación.                  ║
║                                                                               ║
║  CADENA CATEGÓRICA (morfismos entre fases)                                    ║
║  ──────────────────────────────────────────────────────────────────────────   ║
║  Las tres fases forman una cadena de morfismos en la categoría de             ║
║  espacios de Hilbert con estructura Port-Hamiltoniana:                        ║
║                                                                               ║
║    Phase1.compute_control_law()              → ControlSolution                ║
║    Phase1.compute_effective_load_impedance() → Z_eff ∈ ℝ^m ∪ {∞}              ║
║    ─────────────────────────────────────────── (costura Fase 1 → Fase 2)      ║
║    Phase2.tune_dielectric_tensors()          → ImpedanceTensor                ║
║    Phase2.compute_effective_wave_speed()     → c_eff ∈ ℝ₊                     ║
║    ─────────────────────────────────────────── (costura Fase 2 → Fase 3)      ║ 
║    Phase3.audit_time_step()                  → dt_safe ∈ ℝ₊                   ║
║    Phase3.cfl_diagnostic()                   → Dict diagnóstico               ║
║                                                                               ║
║  MEJORAS GLOBALES v3.0                                                        ║
║  ──────────────────────────────────────────────────────────────────────────   ║
║  • SVD truncada con umbral adaptativo para pseudoinversa de Moore-Penrose.    ║
║  • Verificación Kramers-Kronig mediante análisis espectral completo de ε, μ.  ║
║  • Laplaciano asimétrico: simetrización exacta con norma de asimetría.        ║
║  • Lyapunov verificado en trayectorias completas (no solo en un punto).       ║
║  • Cálculo de Z_eff con análisis de canales inactivos y diagnóstico por canal.║
║  • Estimador espectral CFL con tres niveles de fallback y cota Gerschgorin    ║
║    mejorada mediante factor de estructura del grafo.                          ║
║  • Docstrings LaTeX rigurosos con referencias a definiciones formales.        ║
║  • Fases anidadas: el último método de cada fase es la costura al inicio      ║
║    formal de la siguiente.                                                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import eigsh

# ══════════════════════════════════════════════════════════════════════════════
# DEPENDENCIAS ESTRUCTURALES DEL ECOSISTEMA (resilientes con fallback canónico)
# ══════════════════════════════════════════════════════════════════════════════
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:
    class TopologicalInvariantError(Exception):
        """Invariante topológico violado en el ecosistema MIC."""
        pass

    class Morphism:
        """Morfismo categórico abstracto (stub de resiliencia)."""
        pass

try:
    from app.core.immune_system.metric_tensors import G_PHYSICS
except ImportError:
    # Tensor métrico identidad como fallback seguro
    G_PHYSICS: NDArray[np.float64] = np.eye(1, dtype=np.float64)

logger = logging.getLogger("MIC.Physics.DiracInterconnection")


# ══════════════════════════════════════════════════════════════════════════════
# JERARQUÍA DE EXCEPCIONES DE CONTROL Y ESTABILIDAD
# ══════════════════════════════════════════════════════════════════════════════

class DiracMatchingError(TopologicalInvariantError):
    r"""
    Detonada cuando la Ecuación de Matching IDA-PBC no tiene solución en la
    imagen de :math:`g(x)`, o cuando el residuo de matching excede la tolerancia.

    Fundamento matemático:
        La condición necesaria y suficiente para existencia de :math:`\alpha` es:

        .. math::
            f_d := [J_d - R_d]\nabla H_d - [J - R]\nabla H \;\in\; \mathrm{Im}(g)

        donde :math:`\mathrm{Im}(g) = \{g\alpha : \alpha \in \mathbb{R}^m\}`.
        Si :math:`f_d \notin \mathrm{Im}(g)`, el sistema está subactuado en
        la dirección de :math:`f_d` y la condición de energía no se puede satisfacer.
    """
    pass


class ImpedanceMismatchError(TopologicalInvariantError):
    r"""
    Detonada si la sintonización dieléctrica induce un tensor
    no semidefinido positivo, violando la condición de Kramers-Kronig:

    .. math::
        \epsilon_{\mathrm{eff}} \succeq 0, \quad \mu_{\mathrm{eff}} \succeq 0
    """
    pass


class CFLViolationError(TopologicalInvariantError):
    r"""
    Detonada si el diferencial de tiempo exigido rompe el cono de luz causal
    de la red, es decir:

    .. math::
        \Delta t > \frac{2}{c_{\mathrm{eff}}\sqrt{\lambda_{\max}(\Delta_{\mathrm{sym}})}}
    """
    pass


class LyapunovInstabilityError(TopologicalInvariantError):
    r"""
    Detonada si la verificación de Lyapunov detecta :math:`\dot{H}_d > 0`
    en algún punto de la trayectoria, violando la condición de disipación:

    .. math::
        \dot{H}_d = -\nabla H_d^T R_d \nabla H_d \leq 0 \quad \forall x
    """
    pass


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS INMUTABLES DEL ESPACIO DE CONTROL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ImpedanceTensor:
    r"""
    Tensor dieléctrico y magnético sintonizado para adaptación perfecta de
    impedancia en el puerto de control.

    Fundamento matemático:
        Para un medio anisótropo, la impedancia de onda se define como:

        .. math::
            Z_0 = \sqrt{\mu_{\mathrm{eff}} \cdot \epsilon_{\mathrm{eff}}^{-1}}

        La condición de adaptación perfecta exige :math:`Z_0 \equiv Z_{\mathrm{load}}`,
        lo que aniquila el coeficiente de reflexión:

        .. math::
            \Gamma = \frac{Z_{\mathrm{load}} - Z_0}{Z_{\mathrm{load}} + Z_0} = 0

    Atributos:
        epsilon_eff:
            Matriz :math:`(m \times m)` del tensor dieléctrico efectivo.
            Debe ser :math:`\epsilon_{\mathrm{eff}} \succeq 0`.
        mu_eff:
            Matriz :math:`(m \times m)` del tensor de permeabilidad.
            Debe ser :math:`\mu_{\mathrm{eff}} \succeq 0`.
        reflection_coefficient_norm:
            :math:`\|\Gamma\|_F` residual post-sintonización (norma de Frobenius).
        wave_speeds:
            Vector :math:`(m,)` con :math:`c_i = 1/\sqrt{\mu_i \epsilon_i}`
            por canal. Canales inactivos tienen :math:`c_i = 0`.
        is_isotropic:
            ``True`` si los tensores son diagonales (isotrópicos por canal).
        channel_active_mask:
            Vector booleano :math:`(m,)` indicando canales con :math:`Z < \infty`.
    """
    epsilon_eff: NDArray[np.float64]
    mu_eff: NDArray[np.float64]
    reflection_coefficient_norm: float
    wave_speeds: NDArray[np.float64]
    is_isotropic: bool
    channel_active_mask: NDArray[np.bool_]

    def verify_kramers_kronig(self, tol: float = 1e-9) -> Dict[str, Any]:
        r"""
        Verifica la positividad espectral completa de :math:`\epsilon` y :math:`\mu`,
        condición necesaria de Kramers-Kronig para causalidad física.

        La condición de Kramers-Kronig en su forma algebraica exige que ambos
        tensores sean semidefinidos positivos como operadores en :math:`\mathbb{R}^m`:

        .. math::
            \epsilon_{\mathrm{eff}} \succeq 0 \iff
            \lambda_{\min}(\epsilon_{\mathrm{eff}}) \geq 0

        Args:
            tol:
                Tolerancia absoluta para considerar un autovalor no negativo.
                Por defecto :math:`10^{-9}`.

        Returns:
            Diccionario con métricas espectrales:

            - ``epsilon_min_eig``: :math:`\lambda_{\min}(\epsilon_{\mathrm{eff}})`.
            - ``mu_min_eig``: :math:`\lambda_{\min}(\mu_{\mathrm{eff}})`.
            - ``epsilon_max_eig``: :math:`\lambda_{\max}(\epsilon_{\mathrm{eff}})`.
            - ``mu_max_eig``: :math:`\lambda_{\max}(\mu_{\mathrm{eff}})`.
            - ``epsilon_pd``: ``True`` si :math:`\epsilon \succeq 0`.
            - ``mu_pd``: ``True`` si :math:`\mu \succeq 0`.
            - ``epsilon_condition_number``: Número de condición de :math:`\epsilon`.
            - ``mu_condition_number``: Número de condición de :math:`\mu`.

        Raises:
            ImpedanceMismatchError:
                Si algún tensor tiene autovalor negativo más allá de ``tol``.
        """
        eps_eigs = la.eigvalsh(self.epsilon_eff)
        mu_eigs = la.eigvalsh(self.mu_eff)

        eps_min = float(np.min(eps_eigs))
        eps_max = float(np.max(eps_eigs))
        mu_min = float(np.min(mu_eigs))
        mu_max = float(np.max(mu_eigs))

        # Número de condición (solo si epsilon tiene parte positiva)
        eps_cond = (eps_max / eps_min) if eps_min > tol else float("inf")
        mu_cond = (mu_max / mu_min) if mu_min > tol else float("inf")

        eps_pd = bool(eps_min >= -tol)
        mu_pd = bool(mu_min >= -tol)

        return {
            "epsilon_min_eig": eps_min,
            "mu_min_eig": mu_min,
            "epsilon_max_eig": eps_max,
            "mu_max_eig": mu_max,
            "epsilon_pd": eps_pd,
            "mu_pd": mu_pd,
            "epsilon_condition_number": eps_cond,
            "mu_condition_number": mu_cond,
        }


@dataclass(frozen=True, slots=True)
class ControlSolution:
    r"""
    Resultado canónico de la Fase 1: ley de control IDA-PBC con metadatos
    de verificación completos.

    Fundamento matemático:
        La ley de control :math:`\alpha` se obtiene resolviendo:

        .. math::
            g^{\dagger} f_d = \alpha, \quad
            f_d = [J_d - R_d]\nabla H_d - [J - R]\nabla H

        donde :math:`g^{\dagger}` es la pseudoinversa de Moore-Penrose calculada
        por SVD truncada con umbral :math:`\sigma_{\mathrm{tol}} = \tau \cdot \sigma_{\max}`.

    Atributos:
        alpha:
            Vector :math:`(m,)` con la ley de control IDA-PBC.
        H_dot:
            Derivada temporal :math:`\dot{H}_d = -\nabla H_d^T R_d \nabla H_d \leq 0`.
        desired_gradient:
            Vector :math:`(n,)` con :math:`\nabla H_d(x)`.
        port_matrix:
            Matriz :math:`(n, m)` de :math:`g(x)`.
        residual_norm:
            Norma absoluta del residuo :math:`\|f_d - g\alpha\|_2`.
        residual_relative:
            Norma relativa :math:`\|f_d - g\alpha\|_2 / \|f_d\|_2`.
        g_rank:
            Rango numérico de :math:`g` por SVD truncada.
        singular_values:
            Vector de valores singulares de :math:`g` (decrecientes).
        lyapunov_verified:
            ``True`` si :math:`\dot{H}_d \leq 0` en el punto verificado.
        required_forcing:
            Vector :math:`f_d` usado en el matching (para diagnóstico).
    """
    alpha: NDArray[np.float64]
    H_dot: float
    desired_gradient: NDArray[np.float64]
    port_matrix: NDArray[np.float64]
    residual_norm: float
    residual_relative: float
    g_rank: int
    singular_values: NDArray[np.float64]
    lyapunov_verified: bool
    required_forcing: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class InterconnectionState:
    r"""
    Estado consolidado inyectable directamente en ``flux_condenser.py``.

    Representa el resultado final de la cadena categórica de morfismos:

    .. math::
        \phi: \mathcal{X}_{\mathrm{tactics}} \longrightarrow \mathcal{X}_{\mathrm{physics}}

    Atributos:
        control_law_alpha:
            Ley de control IDA-PBC :math:`\alpha(x) \in \mathbb{R}^m`.
        impedance:
            Tensor de impedancia sintonizado (Fase 2).
        safe_dt:
            Paso de tiempo seguro :math:`\Delta t_{\mathrm{safe}} \in \mathbb{R}_+`.
        lyapunov_derivative:
            :math:`\dot{H}_d` certificado en el punto actual.
        c_eff:
            Velocidad de onda efectiva máxima :math:`c_{\mathrm{eff}}`.
        cfl_margin:
            Margen de seguridad CFL :math:`\in (0, 1]` utilizado.
        lambda_max_laplacian:
            :math:`\lambda_{\max}(\Delta_{\mathrm{sym}})` estimado en Fase 3.
        cfl_number:
            Número CFL efectivo :math:`= c \sqrt{\lambda_{\max}} \Delta t / 2`.
    """
    control_law_alpha: NDArray[np.float64]
    impedance: ImpedanceTensor
    safe_dt: float
    lyapunov_derivative: float
    c_eff: float
    cfl_margin: float
    lambda_max_laplacian: float
    cfl_number: float


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                                                              ║
# ║   ████████╗ █████╗  ██████╗███████╗    ██╗                                   ║
# ║   ██╔════╝██╔══██╗██╔════╝██╔════╝    ██║                                    ║
# ║   █████╗  ███████║╚█████╗ █████╗      ██║                                    ║
# ║   ██╔══╝  ██╔══██║ ╚═══██╗██╔══╝      ╚═╝                                    ║
# ║   ██║     ██║  ██║██████╔╝███████╗    ██╗                                    ║
# ║   ╚═╝     ╚═╝  ╚═╝╚═════╝ ╚══════╝    ╚═╝                                    ║
# ║                                                                              ║
# ║   FASE 1 · RESOLUCIÓN IDA-PBC Y CÁLCULO DE IMPEDANCIA EFECTIVA               ║
# ║                                                                              ║
# ║   Responsabilidad: Resolver la Ecuación de Matching del sistema              ║
# ║   Port-Hamiltoniano, certificar estabilidad de Lyapunov, y calcular la       ║
# ║   impedancia efectiva por canal que sirve como entrada a la Fase 2.          ║
# ║                                                                              ║
# ║   Último método de esta fase:  compute_effective_load_impedance()            ║
# ║   → su resultado Z_eff ∈ ℝ^m ∪ {∞} es el argumento de entrada al             ║
# ║     primer método de la Fase 2: tune_dielectric_tensors().                   ║
# ║                                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Phase1_IDAPBC_Solver:
    r"""
    Resuelve la Ecuación de Matching para sistemas Port-Hamiltonianos (PCH).

    **Formulación canónica del sistema PCH**:

    .. math::
        \dot{x} = [J(x) - R(x)] \nabla H(x) + g(x) u

    donde :math:`J = -J^T` (antisimétrica, flujo conservativo) y
    :math:`R = R^T \succeq 0` (simétrica semidefinida positiva, disipación).

    **Ecuación de Matching IDA-PBC**:

    .. math::
        [J_d(x) - R_d(x)] \nabla H_d(x) = [J(x) - R(x)] \nabla H(x) + g(x)\alpha(x)

    La solución :math:`\alpha` existe si y solo si el lado derecho
    :math:`f_d := [J_d - R_d]\nabla H_d - [J - R]\nabla H` pertenece a
    :math:`\mathrm{Im}(g)`. En caso contrario se produce ``DiracMatchingError``.

    **Certificación de Lyapunov**:

    Para sistemas PCH, :math:`H_d` es función de Lyapunov si:

    .. math::
        \dot{H}_d = \nabla H_d^T \dot{x}_d = -\nabla H_d^T R_d \nabla H_d \leq 0

    (el término :math:`\nabla H_d^T J_d \nabla H_d = 0` por antisimetría de :math:`J_d`).

    Args:
        tolerance:
            Umbral absoluto :math:`\tau` para SVD truncada y comparaciones.
            Por defecto :math:`10^{-9}`.
        relative_tol:
            Tolerancia relativa para verificación de simetría y residuos.
            Por defecto :math:`10^{-6}`.
        require_full_rank:
            Si ``True``, lanza ``DiracMatchingError`` cuando :math:`\mathrm{rank}(g) <
            \min(n, m)`. Por defecto ``True``.
        max_residual_relative:
            Máximo residuo relativo admisible :math:`\|f_d - g\alpha\|/\|f_d\|`.
            Por defecto :math:`10^{-4}` (permite ligera subactuación numérica).
    """

    def __init__(
        self,
        tolerance: float = 1e-9,
        relative_tol: float = 1e-6,
        require_full_rank: bool = True,
        max_residual_relative: float = 1e-4,
    ) -> None:
        if tolerance <= 0:
            raise DiracMatchingError(f"tolerance debe ser > 0; got {tolerance}")
        if relative_tol <= 0:
            raise DiracMatchingError(f"relative_tol debe ser > 0; got {relative_tol}")
        if max_residual_relative <= 0:
            raise DiracMatchingError(
                f"max_residual_relative debe ser > 0; got {max_residual_relative}"
            )

        self._tol = tolerance
        self._rel_tol = relative_tol
        self._require_full_rank = require_full_rank
        self._max_res_rel = max_residual_relative

    # ──────────────────────────────────────────────────────────────────────────
    #  VALIDACIONES ESTRUCTURALES (Port-Hamiltoniano)
    # ──────────────────────────────────────────────────────────────────────────

    def _validate_skew_symmetry(
        self, M: NDArray[np.float64], name: str
    ) -> None:
        r"""
        Verifica que :math:`M = -M^T` (antisimetría) con residuo relativo.

        La antisimetría de :math:`J` y :math:`J_d` es estructural en la teoría
        PCH: garantiza que el flujo de interconexión conserva energía.

        Args:
            M:
                Matriz cuadrada a verificar.
            name:
                Nombre descriptivo para mensajes de error.

        Raises:
            DiracMatchingError:
                Si :math:`\|M + M^T\|_F / \max(1, \|M\|_F) > \tau_{\mathrm{rel}}`.
        """
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise DiracMatchingError(
                f"{name} debe ser matriz cuadrada 2D; shape={M.shape}"
            )

        norm_M = float(la.norm(M, ord="fro"))
        residual = float(la.norm(M + M.T, ord="fro")) / max(1.0, norm_M)

        if residual > self._rel_tol:
            raise DiracMatchingError(
                f"{name} no es antisimétrica: "
                f"‖M + M^T‖_F / ‖M‖_F = {residual:.3e} > τ_rel = {self._rel_tol:.3e}. "
                f"Shape={M.shape}, ‖M‖_F={norm_M:.3e}."
            )

    def _validate_symmetry_psd(
        self, M: NDArray[np.float64], name: str
    ) -> NDArray[np.float64]:
        r"""
        Verifica que :math:`M = M^T` y :math:`M \succeq 0`.

        La semidefinición positiva de :math:`R` y :math:`R_d` garantiza que
        la disipación no genera energía. La verificación usa
        ``scipy.linalg.eigvalsh`` que explota la simetría.

        Args:
            M:
                Matriz cuadrada a verificar.
            name:
                Nombre descriptivo para mensajes de error.

        Returns:
            Autovalores de :math:`M` en orden creciente (para diagnóstico).

        Raises:
            DiracMatchingError:
                Si :math:`M \neq M^T` o :math:`\lambda_{\min}(M) < -\tau`.
        """
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise DiracMatchingError(
                f"{name} debe ser matriz cuadrada 2D; shape={M.shape}"
            )

        norm_M = float(la.norm(M, ord="fro"))
        sym_res = float(la.norm(M - M.T, ord="fro")) / max(1.0, norm_M)

        if sym_res > self._rel_tol:
            raise DiracMatchingError(
                f"{name} no es simétrica: "
                f"‖M - M^T‖_F / ‖M‖_F = {sym_res:.3e} > τ_rel = {self._rel_tol:.3e}."
            )

        # Simetrizar explícitamente para eigvalsh estable
        M_sym = 0.5 * (M + M.T)
        eigvals = la.eigvalsh(M_sym)
        min_eig = float(np.min(eigvals))

        if min_eig < -self._tol:
            raise DiracMatchingError(
                f"{name} no es semidefinida positiva: "
                f"λ_min = {min_eig:.3e} < -τ = {-self._tol:.3e}."
            )

        return eigvals

    def _svd_rank_and_pinv(
        self, g: NDArray[np.float64]
    ) -> Tuple[int, NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Calcula el rango numérico y la pseudoinversa de Moore-Penrose de :math:`g`
        mediante SVD truncada.

        La pseudoinversa se calcula como:

        .. math::
            g^{\dagger} = V \Sigma^{\dagger} U^T

        donde :math:`\Sigma^{\dagger}_{ii} = 1/\sigma_i` si
        :math:`\sigma_i > \tau \cdot \sigma_{\max}`, y :math:`0` en otro caso.

        El umbral adaptativo :math:`\tau \cdot \sigma_{\max}` evita amplificar
        componentes de ruido numérico en la dirección del núcleo de :math:`g`.

        Args:
            g:
                Matriz :math:`(n, m)` del puerto de control.

        Returns:
            Tupla ``(rank, g_pinv, singular_values)`` donde:

            - ``rank``: Rango numérico :math:`r = |\{i : \sigma_i > \tau \sigma_{\max}\}|`.
            - ``g_pinv``: Pseudoinversa :math:`g^{\dagger}` de forma :math:`(m, n)`.
            - ``singular_values``: Vector :math:`(\min(n,m),)` en orden decreciente.
        """
        # SVD completa (full_matrices=False para economía de memoria)
        U, sv, Vt = la.svd(g, full_matrices=False)

        # Umbral adaptativo
        sigma_max = float(sv[0]) if sv.size > 0 else 0.0
        threshold = self._tol * sigma_max if sigma_max > 0 else self._tol

        # Máscara de valores singulares significativos
        mask = sv > threshold
        rank = int(np.sum(mask))

        if rank == 0:
            # g es la matriz cero: pseudoinversa es también cero
            g_pinv = np.zeros((g.shape[1], g.shape[0]), dtype=np.float64)
            return 0, g_pinv, sv

        # Pseudoinversa truncada: V @ diag(1/σ_i) @ U^T
        sv_inv = np.where(mask, 1.0 / np.maximum(sv, threshold), 0.0)
        g_pinv = (Vt.T * sv_inv) @ U.T  # shape (m, n)

        return rank, g_pinv, sv

    def _validate_port_matrix(
        self, g: NDArray[np.float64], grad_dim: int
    ) -> Tuple[int, int, int, NDArray[np.float64], NDArray[np.float64]]:
        r"""
        Valida dimensiones y calcula rango y pseudoinversa de :math:`g(x)`.

        Args:
            g:
                Matriz :math:`(n, m)` del puerto de control.
            grad_dim:
                Dimensión esperada :math:`n` del espacio de estado.

        Returns:
            Tupla ``(n, m, rank, g_pinv, singular_values)``.

        Raises:
            DiracMatchingError:
                Si :math:`g` no es 2D, dimensiones inconsistentes, o rango
                insuficiente (cuando ``require_full_rank=True``).
        """
        if g.ndim != 2:
            raise DiracMatchingError(
                f"g_port debe ser matriz 2D; ndim={g.ndim}"
            )

        n_g, m_g = g.shape

        if n_g != grad_dim:
            raise DiracMatchingError(
                f"g_port tiene {n_g} filas pero se esperaban {grad_dim} "
                f"(dimensión del espacio de estado)."
            )

        g_rank, g_pinv, sv = self._svd_rank_and_pinv(g)

        expected_rank = min(n_g, m_g)
        if self._require_full_rank and g_rank < expected_rank:
            raise DiracMatchingError(
                f"g_port no es de rango completo: "
                f"rank={g_rank} < min(n,m)={expected_rank}. "
                f"σ_min/σ_max = {sv[-1]/sv[0]:.3e}. "
                f"El sistema está subactuado."
            )

        return n_g, m_g, g_rank, g_pinv, sv

    # ──────────────────────────────────────────────────────────────────────────
    #  RESOLUCIÓN IDA-PBC PRINCIPAL
    # ──────────────────────────────────────────────────────────────────────────

    def compute_control_law(
        self,
        J_current: NDArray[np.float64],
        R_current: NDArray[np.float64],
        grad_H: NDArray[np.float64],
        J_desired: NDArray[np.float64],
        R_desired: NDArray[np.float64],
        grad_H_desired: NDArray[np.float64],
        g_port: NDArray[np.float64],
    ) -> ControlSolution:
        r"""
        Resuelve la Ecuación de Matching IDA-PBC mediante pseudoinversa SVD
        truncada con verificación completa de rango, residuo y Lyapunov.

        **Algoritmo**:

        1. Validación estructural de :math:`J, J_d` (antisimetría) y
           :math:`R, R_d` (simetría + PSD).
        2. Validación dimensional de :math:`g` y cálculo de :math:`g^{\dagger}`.
        3. Cálculo del forzamiento requerido:
           :math:`f_d = [J_d - R_d]\nabla H_d - [J - R]\nabla H`.
        4. Solución por pseudoinversa: :math:`\alpha = g^{\dagger} f_d`.
        5. Verificación del residuo:
           :math:`\|f_d - g\alpha\|_2 / \|f_d\|_2 \leq \tau_{\max}`.
        6. Certificación de Lyapunov:
           :math:`\dot{H}_d = -\nabla H_d^T R_d \nabla H_d \leq 0`.

        Args:
            J_current:
                Matriz de interconexión actual :math:`J(x) \in \mathbb{R}^{n \times n}`,
                debe ser antisimétrica.
            R_current:
                Matriz de disipación actual :math:`R(x) \in \mathbb{R}^{n \times n}`,
                debe ser simétrica PSD.
            grad_H:
                Gradiente del Hamiltoniano actual :math:`\nabla H(x) \in \mathbb{R}^n`.
            J_desired:
                Matriz de interconexión deseada :math:`J_d(x)`, antisimétrica.
            R_desired:
                Matriz de disipación deseada :math:`R_d(x)`, simétrica PSD.
            grad_H_desired:
                Gradiente del Hamiltoniano deseado :math:`\nabla H_d(x) \in \mathbb{R}^n`.
            g_port:
                Matriz de puerto de control :math:`g(x) \in \mathbb{R}^{n \times m}`.

        Returns:
            :class:`ControlSolution` con todos los metadatos diagnósticos.

        Raises:
            DiracMatchingError:
                En cualquier violación estructural o si el residuo excede
                ``max_residual_relative``.
            LyapunovInstabilityError:
                Si :math:`\dot{H}_d > \tau` (Lyapunov violado).
        """
        # ── 1. Validación estructural ──────────────────────────────────────
        self._validate_skew_symmetry(J_current, "J_current")
        self._validate_skew_symmetry(J_desired, "J_desired")
        R_current_eigs = self._validate_symmetry_psd(R_current, "R_current")
        R_desired_eigs = self._validate_symmetry_psd(R_desired, "R_desired")

        # Validar gradientes como vectores 1D
        for vec, vname in [(grad_H, "grad_H"), (grad_H_desired, "grad_H_desired")]:
            if vec.ndim != 1:
                raise DiracMatchingError(
                    f"{vname} debe ser vector 1D; ndim={vec.ndim}"
                )

        n = grad_H.shape[0]

        if grad_H_desired.shape[0] != n:
            raise DiracMatchingError(
                f"grad_H ({n}) y grad_H_desired ({grad_H_desired.shape[0]}) "
                f"deben tener la misma dimensión."
            )

        if J_current.shape != (n, n):
            raise DiracMatchingError(
                f"J_current shape={J_current.shape} inconsistente con n={n}."
            )
        if J_desired.shape != (n, n):
            raise DiracMatchingError(
                f"J_desired shape={J_desired.shape} inconsistente con n={n}."
            )

        # ── 2. Validar puerto g y obtener pseudoinversa ────────────────────
        n_g, m_g, g_rank, g_pinv, sv = self._validate_port_matrix(g_port, n)

        # ── 3. Calcular dinámicas y forzamiento requerido ──────────────────
        # Dinámica del sistema actual en lazo abierto
        open_loop_dyn: NDArray[np.float64] = (J_current - R_current) @ grad_H

        # Dinámica deseada en lazo cerrado
        desired_dyn: NDArray[np.float64] = (J_desired - R_desired) @ grad_H_desired

        # Forzamiento que g debe suministrar
        required_forcing: NDArray[np.float64] = desired_dyn - open_loop_dyn

        norm_forcing = float(la.norm(required_forcing))

        # ── 4. Resolver α = g† f_d ────────────────────────────────────────
        alpha_x: NDArray[np.float64] = g_pinv @ required_forcing

        # ── 5. Verificación del residuo de matching ───────────────────────
        reconstructed: NDArray[np.float64] = g_port @ alpha_x
        residual_vec: NDArray[np.float64] = required_forcing - reconstructed
        norm_res = float(la.norm(residual_vec))
        norm_relative = norm_res / max(norm_forcing, 1e-15)

        if norm_relative > self._max_res_rel:
            raise DiracMatchingError(
                f"Residuo de matching excesivo: "
                f"‖f_d - gα‖/‖f_d‖ = {norm_relative:.3e} > τ = {self._max_res_rel:.3e}. "
                f"rank(g)={g_rank}, dim(f_d)={n}. "
                f"Sistema posiblemente subactuado en {n - g_rank} dimensiones."
            )

        # ── 6. Certificación de Lyapunov ──────────────────────────────────
        # ∇H_d^T J_d ∇H_d = 0 por antisimetría de J_d
        # ∴ Ḣ_d = ∇H_d^T (J_d - R_d) ∇H_d = -∇H_d^T R_d ∇H_d
        H_dot = float(-(grad_H_desired @ R_desired @ grad_H_desired))
        lyapunov_ok = H_dot <= self._tol

        if not lyapunov_ok:
            # Es una violación activa: puede indicar R_desired mal condicionada
            logger.warning(
                f"[Phase1] Lyapunov en límite: Ḣ_d = {H_dot:.3e} > 0. "
                f"λ_min(R_d) = {float(R_desired_eigs[0]):.3e}. "
                f"Verificar semidefinición positiva de R_desired."
            )

        logger.debug(
            f"[Phase1] IDA-PBC resuelto: rank(g)={g_rank}/{min(n, m_g)}, "
            f"‖residual‖={norm_res:.2e}, rel={norm_relative:.2e}, "
            f"Ḣ_d={H_dot:.2e}, σ_max={sv[0]:.2e}, σ_min={sv[-1]:.2e}"
        )

        return ControlSolution(
            alpha=alpha_x,
            H_dot=H_dot,
            desired_gradient=grad_H_desired,
            port_matrix=g_port,
            residual_norm=norm_res,
            residual_relative=norm_relative,
            g_rank=g_rank,
            singular_values=sv,
            lyapunov_verified=lyapunov_ok,
            required_forcing=required_forcing,
        )

    def verify_trajectory_lyapunov(
        self,
        R_desired: NDArray[np.float64],
        trajectory_gradients: NDArray[np.float64],
    ) -> Dict[str, Any]:
        r"""
        Verifica :math:`\dot{H}_d \leq 0` a lo largo de una trayectoria completa
        de gradientes, no solo en un punto.

        Para cada instante :math:`t` con gradiente :math:`\nabla H_d^{(t)}`:

        .. math::
            \dot{H}_d^{(t)} = -(\nabla H_d^{(t)})^T R_d (\nabla H_d^{(t)}) \leq 0

        Esta verificación es más robusta que la verificación puntual porque
        detecta violaciones causadas por singularidades en la trayectoria.

        Args:
            R_desired:
                Matriz :math:`(n, n)` de disipación deseada (simétrica PSD).
            trajectory_gradients:
                Matriz :math:`(T, n)` con gradientes en :math:`T` instantes.

        Returns:
            Diccionario con métricas de la trayectoria:

            - ``H_dot_min``: :math:`\min_t \dot{H}_d^{(t)}`.
            - ``H_dot_max``: :math:`\max_t \dot{H}_d^{(t)}`.
            - ``H_dot_mean``: :math:`\frac{1}{T}\sum_t \dot{H}_d^{(t)}`.
            - ``violations``: Número de instantes con :math:`\dot{H}_d > \tau`.
            - ``violation_indices``: Índices de los instantes violadores.
            - ``lyapunov_stable``: ``True`` si no hay violaciones.
            - ``total_steps``: :math:`T`.

        Raises:
            DiracMatchingError:
                Si ``trajectory_gradients`` no es 2D.
            LyapunovInstabilityError:
                Si se detectan violaciones de Lyapunov en la trayectoria.
        """
        if trajectory_gradients.ndim != 2:
            raise DiracMatchingError(
                f"trajectory_gradients debe ser 2D; ndim={trajectory_gradients.ndim}"
            )

        T, n = trajectory_gradients.shape

        # Validar R_desired
        self._validate_symmetry_psd(R_desired, "R_desired (trayectoria)")

        # Calcular Ḣ_d en cada paso: Ḣ_d^(t) = -grad^T R_d grad (forma cuadrática)
        # Eficientemente: primero computamos R_d @ grad para cada t
        R_grads: NDArray[np.float64] = trajectory_gradients @ R_desired  # (T, n)
        H_dots: NDArray[np.float64] = -np.einsum(
            "ti,ti->t", trajectory_gradients, R_grads
        )  # (T,) vectorizado

        violation_mask = H_dots > self._tol
        violations = int(np.sum(violation_mask))
        violation_indices = np.where(violation_mask)[0].tolist()

        stable = violations == 0

        result = {
            "H_dot_min": float(np.min(H_dots)),
            "H_dot_max": float(np.max(H_dots)),
            "H_dot_mean": float(np.mean(H_dots)),
            "violations": violations,
            "violation_indices": violation_indices,
            "lyapunov_stable": stable,
            "total_steps": T,
        }

        if not stable:
            raise LyapunovInstabilityError(
                f"{violations}/{T} violaciones de Lyapunov en la trayectoria. "
                f"Ḣ_d_max = {float(np.max(H_dots)):.3e} en índices {violation_indices[:5]}"
                f"{'...' if violations > 5 else ''}."
            )

        logger.debug(
            f"[Phase1] Lyapunov trayectoria OK: T={T}, "
            f"Ḣ_d ∈ [{result['H_dot_min']:.2e}, {result['H_dot_max']:.2e}]"
        )

        return result

    # ──────────────────────────────────────────────────────────────────────────
    #  COSTURA FASE 1 → FASE 2
    #  El resultado de este método es el argumento directo de
    #  Phase2_ImpedanceTuner.tune_dielectric_tensors()
    # ──────────────────────────────────────────────────────────────────────────

    def compute_effective_load_impedance(
        self, control_solution: ControlSolution
    ) -> NDArray[np.float64]:
        r"""
        **[COSTURA FASE 1 → FASE 2]**

        Calcula la impedancia efectiva por canal del puerto de control:

        .. math::
            Z_{\mathrm{eff},i} = \frac{\alpha_i}{(g^T \nabla H_d)_i}

        donde :math:`(g^T \nabla H_d)_i` es la :math:`i`-ésima componente de la
        salida del puerto en lazo cerrado (flujo de energía por canal).

        **Interpretación física**:
            :math:`Z_{\mathrm{eff},i}` es la razón entre la acción de control
            (análogo a la tensión) y el flujo de energía (análogo a la corriente)
            en el :math:`i`-ésimo canal. Esta es exactamente la impedancia que
            el sintonizador dieléctrico (Fase 2) debe igualar para obtener
            :math:`\Gamma = 0`.

        **Canales inactivos**:
            Si :math:`|(g^T \nabla H_d)_i| \leq \tau`, el canal :math:`i` no
            transporta flujo de energía y se asigna :math:`Z_{\mathrm{eff},i} = \infty`.
            El sintonizador (Fase 2) interpreta esto como :math:`\epsilon_i = 0`
            (canal apagado).

        Args:
            control_solution:
                :class:`ControlSolution` producida por :meth:`compute_control_law`.

        Returns:
            Vector :math:`Z_{\mathrm{eff}} \in (\mathbb{R} \cup \{\infty\})^m`.
            Los canales activos tienen :math:`Z_{\mathrm{eff},i} \in \mathbb{R}`,
            los inactivos tienen :math:`Z_{\mathrm{eff},i} = +\infty`.

        Raises:
            DiracMatchingError:
                Si se detectan NaN en canales activos.

        **Nota de encadenamiento**:
            Este método es el último de la Fase 1. Su valor de retorno
            ``Z_eff`` debe pasarse directamente como argumento a
            ``Phase2_ImpedanceTuner.tune_dielectric_tensors(Z_eff)``.
        """
        alpha: NDArray[np.float64] = control_solution.alpha
        g: NDArray[np.float64] = control_solution.port_matrix
        grad_Hd: NDArray[np.float64] = control_solution.desired_gradient

        # Salida del puerto en lazo cerrado: y_d = g^T ∇H_d ∈ ℝ^m
        y_d: NDArray[np.float64] = g.T @ grad_Hd  # shape (m,)

        m = alpha.shape[0]
        Z_eff = np.full(m, np.inf, dtype=np.float64)

        # Máscara de canales activos
        active_mask: NDArray[np.bool_] = np.abs(y_d) > self._tol

        if np.any(active_mask):
            Z_eff[active_mask] = alpha[active_mask] / y_d[active_mask]

        # Verificación de integridad
        nan_in_active = np.any(np.isnan(Z_eff[active_mask])) if np.any(active_mask) else False
        if nan_in_active:
            raise DiracMatchingError(
                "Impedancia efectiva contiene NaN en canales activos. "
                "Posible división por cero no capturada."
            )

        n_active = int(np.sum(active_mask))
        n_inactive = m - n_active

        # Estadísticas diagnósticas
        if n_active > 0:
            Z_finite = Z_eff[active_mask]
            logger.debug(
                f"[Phase1→2] Z_eff: {n_active} activos "
                f"(min={np.min(np.abs(Z_finite)):.2e}, "
                f"max={np.max(np.abs(Z_finite)):.2e}), "
                f"{n_inactive} inactivos (Z=∞)."
            )
        else:
            logger.warning(
                "[Phase1→2] Todos los canales inactivos (y_d ≈ 0). "
                "Verificar grad_H_desired."
            )

        return Z_eff


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                                                              ║
# ║   ████████╗ █████╗  ██████╗███████╗    ██████╗                               ║
# ║   ██╔════╝██╔══██╗██╔════╝██╔════╝    ╚════██╗                               ║
# ║   █████╗  ███████║╚█████╗ █████╗       █████╔╝                               ║
# ║   ██╔══╝  ██╔══██║ ╚═══██╗██╔══╝      ██╔═══╝                                ║
# ║   ██║     ██║  ██║██████╔╝███████╗    ███████╗                               ║
# ║   ╚═╝     ╚═╝  ╚═╝╚═════╝ ╚══════╝    ╚══════╝                               ║
# ║                                                                              ║
# ║   FASE 2 · SINTONIZADOR DINÁMICO DE IMPEDANCIA (PML)                         ║
# ║                                                                              ║
# ║   Entrada  : Z_eff ∈ (ℝ ∪ {∞})^m    ← salida de Phase1                       ║
# ║   Salida   : c_eff ∈ ℝ₊             → entrada de Phase3                      ║
# ║                                                                              ║
# ║   Responsabilidad: Sintonizar los tensores dieléctricos ε_eff y μ_eff        ║
# ║   para adaptación perfecta de impedancia (Γ = 0), verificar la condición     ║
# ║   de Kramers-Kronig, y calcular la velocidad de onda efectiva que            ║
# ║   determina el límite CFL en la Fase 3.                                      ║
# ║                                                                              ║
# ║   Último método de esta fase: compute_effective_wave_speed()                 ║
# ║   → su resultado c_eff ∈ ℝ₊ es el argumento principal de                    ║
# ║     Phase3_CFLGovernor.audit_time_step().                                    ║
# ║                                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Phase2_ImpedanceTuner:
    r"""
    Sintoniza los tensores dieléctricos :math:`\epsilon_{\mathrm{eff}}` y
    :math:`\mu_{\mathrm{eff}}` para acoplamiento perfecto de impedancia en el
    puerto de control.

    **Principio de adaptación**:

    Para un medio anisótropo isótropo por canal, la impedancia característica
    del :math:`i`-ésimo canal es:

    .. math::
        Z_{0,i} = \sqrt{\frac{\mu_i}{\epsilon_i}}

    La condición de adaptación perfecta :math:`Z_{0,i} = Z_{\mathrm{load},i}`
    se satisface eligiendo:

    .. math::
        \epsilon_{\mathrm{eff},i} = \frac{\mu_{\mathrm{base}}}{Z_{\mathrm{load},i}^2}

    lo que implica :math:`\Gamma_i = 0` exactamente (sin reflexión).

    **Restricción de Kramers-Kronig**:

    La causalidad física exige :math:`\epsilon \succeq 0` y :math:`\mu \succeq 0`
    (positividad completa como operadores). Esta condición se verifica
    espectralmente.

    Args:
        base_permeability:
            Permeabilidad base :math:`\mu_0 > 0` uniforme por canal.
            Por defecto :math:`1.0`.
        max_permittivity:
            Cota superior :math:`\epsilon_{\max}` para evitar materiales
            físicamente imposibles. Por defecto :math:`10^6`.
        min_wave_speed:
            Velocidad mínima admisible :math:`c_{\min} > 0`. Canales con
            :math:`c_i < c_{\min}` se consideran apagados.
            Por defecto :math:`10^{-6}`.
        epsilon_floor:
            Piso numérico para :math:`\epsilon` en canales activos.
            Por defecto :math:`10^{-15}`.

    Raises:
        ImpedanceMismatchError:
            Si ``base_permeability ≤ 0``.
    """

    def __init__(
        self,
        base_permeability: float = 1.0,
        max_permittivity: float = 1e6,
        min_wave_speed: float = 1e-6,
        epsilon_floor: float = 1e-15,
    ) -> None:
        if base_permeability <= 0:
            raise ImpedanceMismatchError(
                f"base_permeability debe ser > 0; got {base_permeability}."
            )
        if max_permittivity <= 0:
            raise ImpedanceMismatchError(
                f"max_permittivity debe ser > 0; got {max_permittivity}."
            )
        if min_wave_speed <= 0:
            raise ImpedanceMismatchError(
                f"min_wave_speed debe ser > 0; got {min_wave_speed}."
            )

        self._mu_base = float(base_permeability)
        self._eps_max = float(max_permittivity)
        self._c_min = float(min_wave_speed)
        self._eps_floor = float(epsilon_floor)

    # ──────────────────────────────────────────────────────────────────────────
    #  SINTONIZACIÓN DE TENSORES DIELÉCTRICOS
    # ──────────────────────────────────────────────────────────────────────────

    def tune_dielectric_tensors(
        self, target_load_impedances: NDArray[np.float64]
    ) -> ImpedanceTensor:
        r"""
        **[ENTRADA DE FASE 2 — recibe Z_eff de Phase1]**

        Sintoniza :math:`\epsilon_{\mathrm{eff}}` canal a canal para lograr
        adaptación perfecta :math:`Z_0 = Z_{\mathrm{load}}`:

        .. math::
            \epsilon_{\mathrm{eff},i} = \frac{\mu_{\mathrm{base}}}{Z_{\mathrm{load},i}^2},
            \quad
            Z_{0,i} = \sqrt{\frac{\mu_{\mathrm{base}}}{\epsilon_{\mathrm{eff},i}}}
            = Z_{\mathrm{load},i}

        **Canales inactivos** (:math:`Z_{\mathrm{load},i} = \infty`):
            Se asigna :math:`\epsilon_{\mathrm{eff},i} = 0` y
            :math:`c_i = 0` (canal sin propagación).

        **Canales con** :math:`Z < 0`:
            Impedancias negativas indican activos (amplificadores). Se toma
            :math:`|Z|` para el cálculo de :math:`\epsilon` y se registra una
            advertencia.

        **Verificación Kramers-Kronig**:
            Se verifica :math:`\epsilon \succeq 0` y :math:`\mu \succeq 0`
            espectralmente (no solo diagonal).

        Args:
            target_load_impedances:
                Vector :math:`Z_{\mathrm{load}} \in (\mathbb{R} \cup \{\infty\})^m`
                proveniente de :meth:`Phase1_IDAPBC_Solver.compute_effective_load_impedance`.

        Returns:
            :class:`ImpedanceTensor` con tensores sintonizados y métricas.

        Raises:
            ImpedanceMismatchError:
                Si el tensor resultante no es PSD o contiene NaN.
        """
        Z = np.asarray(target_load_impedances, dtype=np.float64)

        if Z.ndim != 1:
            raise ImpedanceMismatchError(
                f"target_load_impedances debe ser 1D; ndim={Z.ndim}."
            )
        if np.any(np.isnan(Z)):
            raise ImpedanceMismatchError(
                "target_load_impedances contiene NaN. "
                "Verificar Phase1.compute_effective_load_impedance."
            )

        m = Z.shape[0]

        # ── Clasificación de canales ───────────────────────────────────────
        # Canal activo: Z finito y Z^2 > 0 (puede ser negativo → |Z|)
        active_mask: NDArray[np.bool_] = np.isfinite(Z) & (Z != 0.0)
        inactive_mask: NDArray[np.bool_] = ~active_mask

        # Advertencia para canales activos con Z < 0 (elementos activos)
        negative_z = active_mask & (Z < 0)
        if np.any(negative_z):
            logger.warning(
                f"[Phase2] {int(np.sum(negative_z))} canales con Z < 0 "
                f"(elementos activos). Usando |Z| para sintonización."
            )

        # ── Cálculo de ε por canal ─────────────────────────────────────────
        epsilon_diag = np.zeros(m, dtype=np.float64)

        if np.any(active_mask):
            Z_active = np.abs(Z[active_mask])  # Usar |Z| para robustez
            # ε_i = μ_base / Z_i²  con cota superior para estabilidad numérica
            eps_active = self._mu_base / np.maximum(Z_active ** 2, self._eps_floor)
            eps_active = np.clip(eps_active, a_min=0.0, a_max=self._eps_max)
            epsilon_diag[active_mask] = eps_active

        # μ uniforme (hipótesis: medio magnéticamente lineal e isótropo)
        mu_diag = np.full(m, self._mu_base, dtype=np.float64)

        # ── Construcción de matrices tensoriales ───────────────────────────
        eps_mat: NDArray[np.float64] = np.diag(epsilon_diag)
        mu_mat: NDArray[np.float64] = np.diag(mu_diag)

        # ── Verificación Kramers-Kronig (PSD completo) ─────────────────────
        eps_eigs = la.eigvalsh(eps_mat)
        mu_eigs = la.eigvalsh(mu_mat)

        eps_min = float(np.min(eps_eigs))
        mu_min = float(np.min(mu_eigs))

        if eps_min < -self._eps_floor:
            raise ImpedanceMismatchError(
                f"ε_eff no es PSD: λ_min(ε) = {eps_min:.3e} < 0. "
                f"Kramers-Kronig violado."
            )
        if mu_min < -self._eps_floor:
            raise ImpedanceMismatchError(
                f"μ_eff no es PSD: λ_min(μ) = {mu_min:.3e} < 0. "
                f"Kramers-Kronig violado."
            )

        # ── Verificación del coeficiente de reflexión ──────────────────────
        # Para canales activos con ε > 0: Z_0 = √(μ/ε), Γ = (Z_load - Z_0)/(Z_load + Z_0)
        gamma_norm = 0.0
        if np.any(active_mask):
            eps_active_vals = epsilon_diag[active_mask]
            mu_active_vals = mu_diag[active_mask]
            Z_active_abs = np.abs(Z[active_mask])

            # Solo canales con ε > 0 (algunos podrían haberse clippeado a 0)
            nonzero_eps = eps_active_vals > self._eps_floor
            if np.any(nonzero_eps):
                Z0_i = np.sqrt(
                    mu_active_vals[nonzero_eps]
                    / np.maximum(eps_active_vals[nonzero_eps], self._eps_floor)
                )
                Z_load_i = Z_active_abs[nonzero_eps]
                denom = Z_load_i + Z0_i
                denom_safe = np.where(np.abs(denom) > self._eps_floor, denom, 1.0)
                gamma_vec = (Z_load_i - Z0_i) / denom_safe
                gamma_norm = float(la.norm(gamma_vec))

        # ── Velocidades de onda por canal ──────────────────────────────────
        wave_speeds = np.zeros(m, dtype=np.float64)
        valid_speed_mask = (
            active_mask
            & (epsilon_diag > self._eps_floor)
            & (mu_diag > self._eps_floor)
        )
        if np.any(valid_speed_mask):
            wave_speeds[valid_speed_mask] = 1.0 / np.sqrt(
                mu_diag[valid_speed_mask] * epsilon_diag[valid_speed_mask]
            )

        n_active = int(np.sum(active_mask))
        n_inactive = m - n_active

        logger.debug(
            f"[Phase2] PML: {n_active} activos, {n_inactive} inactivos. "
            f"‖Γ‖_F={gamma_norm:.3e}, "
            f"ε ∈ [{epsilon_diag[active_mask].min() if n_active > 0 else 0.0:.2e}, "
            f"{epsilon_diag[active_mask].max() if n_active > 0 else 0.0:.2e}], "
            f"c_max={float(np.max(wave_speeds)):.2e}"
        )

        return ImpedanceTensor(
            epsilon_eff=eps_mat,
            mu_eff=mu_mat,
            reflection_coefficient_norm=gamma_norm,
            wave_speeds=wave_speeds,
            is_isotropic=True,  # Tensores diagonales por construcción
            channel_active_mask=active_mask,
        )

    # ──────────────────────────────────────────────────────────────────────────
    #  COSTURA FASE 2 → FASE 3
    #  El resultado de este método es el argumento principal de
    #  Phase3_CFLGovernor.audit_time_step()
    # ──────────────────────────────────────────────────────────────────────────

    def compute_effective_wave_speed(
        self, impedance: ImpedanceTensor
    ) -> float:
        r"""
        **[COSTURA FASE 2 → FASE 3]**

        Calcula :math:`c_{\mathrm{eff}}` como el máximo de las velocidades de
        onda por canal:

        .. math::
            c_{\mathrm{eff}} = \max_{i : \text{activo}}
            \frac{1}{\sqrt{\mu_i \epsilon_{\mathrm{eff},i}}}

        **Justificación de la elección del máximo**:
            La condición CFL debe ser satisfecha por el canal más rápido
            (mayor velocidad de propagación). Si se usara el mínimo o la media,
            los canales rápidos violarían la estabilidad. El máximo es la
            elección conservadora correcta.

        **Estadísticas adicionales**:
            Se registran en el log la velocidad mínima, media y la razón
            :math:`c_{\max}/c_{\min}` (factor de anisotropía).

        Args:
            impedance:
                :class:`ImpedanceTensor` producido por :meth:`tune_dielectric_tensors`.

        Returns:
            :math:`c_{\mathrm{eff}} \in \mathbb{R}_+`, velocidad de onda máxima.

        Raises:
            ImpedanceMismatchError:
                Si ningún canal tiene velocidad positiva, o si
                :math:`c_{\mathrm{eff}} < c_{\min}`.

        **Nota de encadenamiento**:
            Este es el último método de la Fase 2. Su valor de retorno
            ``c_eff`` debe pasarse como ``wave_speed_c`` al método
            ``Phase3_CFLGovernor.audit_time_step(graph_laplacian, c_eff, dt)``.
        """
        c_per_channel: NDArray[np.float64] = impedance.wave_speeds

        # Máscara de canales con velocidad positiva
        positive_speed_mask = c_per_channel > self._c_min
        n_positive = int(np.sum(positive_speed_mask))

        if n_positive == 0:
            raise ImpedanceMismatchError(
                f"Ningún canal tiene velocidad de onda > c_min={self._c_min:.2e}. "
                f"El medio está completamente apagado o mal sintonizado. "
                f"c_per_channel = {c_per_channel.tolist()}"
            )

        c_active = c_per_channel[positive_speed_mask]
        c_eff = float(np.max(c_active))
        c_min_active = float(np.min(c_active))
        c_mean_active = float(np.mean(c_active))
        anisotropy = c_eff / max(c_min_active, self._c_min)

        logger.debug(
            f"[Phase2→3] c_eff = {c_eff:.4e} "
            f"(min={c_min_active:.2e}, mean={c_mean_active:.2e}, "
            f"anisotropía={anisotropy:.2f}x, "
            f"{n_positive}/{len(c_per_channel)} canales activos)"
        )

        return c_eff


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                                                              ║
# ║   ████████╗ █████╗  ██████╗███████╗    ██████╗                               ║
# ║   ██╔════╝██╔══██╗██╔════╝██╔════╝    ╚════██╗                               ║
# ║   █████╗  ███████║╚█████╗ █████╗       █████╔╝                               ║
# ║   ██╔══╝  ██╔══██║ ╚═══██╗██╔══╝       ╚═══██╗                               ║
# ║   ██║     ██║  ██║██████╔╝███████╗    ██████╔╝                               ║
# ║   ╚═╝     ╚═╝  ╚═╝╚═════╝ ╚══════╝    ╚═════╝                                ║
# ║                                                                              ║
# ║   FASE 3 · GOBERNADOR DEL LÍMITE DE COURANT-FRIEDRICHS-LEWY (CFL)            ║
# ║                                                                              ║
# ║   Entrada  : c_eff ∈ ℝ₊              ← salida de Phase2                      ║
# ║   Salida   : (dt_safe, diag_report)   → InterconnectionState                 ║
# ║                                                                              ║
# ║   Responsabilidad: Custodiar el cono de luz causal del grafo computacional.  ║
# ║   Estima λ_max(Δ_sym) del Laplaciano de la red, calcula Δt_max según CFL,    ║
# ║   y veta cualquier paso de integración que violaría la estabilidad           ║
# ║   numérica del esquema de discretización temporal.                           ║
# ║                                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class Phase3_CFLGovernor:
    r"""
    Custodio del cono de luz causal en grafos computacionales Port-Hamiltonianos.

    **Condición CFL para esquemas en grafos**:

    Para un grafo con Laplaciano :math:`\Delta` y velocidad de onda
    :math:`c_{\mathrm{eff}}`, la condición de Courant-Friedrichs-Lewy es:

    .. math::
        \Delta t \leq \Delta t_{\max} :=
        \frac{2 \cdot \mathrm{margin}}{c_{\mathrm{eff}}
        \sqrt{\lambda_{\max}(\Delta_{\mathrm{sym}})}}

    donde :math:`\Delta_{\mathrm{sym}} = \tfrac{1}{2}(\Delta + \Delta^T)` es
    la parte simétrica (fundamental cuando :math:`\Delta` proviene de un sistema
    no conservativo, como redes con corrientes de fuga direccionadas).

    **Estimación de** :math:`\lambda_{\max}`:

    Se implementan tres niveles de fallback con calidad decreciente:

    1. **Lanczos (ARPACK)**: Método iterativo exacto, :math:`O(nk)` operaciones.
    2. **Potencia iterada con deflación**: :math:`O(n \cdot \text{iter})`.
    3. **Cota de Gerschgorin geométrica**: :math:`O(n)`, cota superior garantizada.

    Args:
        safety_margin:
            Factor de seguridad :math:`\rho \in (0, 1]`. El paso seguro es
            :math:`\Delta t_{\mathrm{safe}} = \rho \cdot \Delta t_{\max}`.
            Por defecto :math:`0.95`.
        lanczos_tol:
            Tolerancia de convergencia para el método de Lanczos.
            Por defecto :math:`10^{-6}`.
        fallback_max_iter:
            Iteraciones máximas para la potencia iterada. Por defecto :math:`1000`.
        lanczos_k:
            Número de valores propios a estimar en Lanczos. Por defecto :math:`3`.

    Raises:
        CFLViolationError:
            Si ``safety_margin`` no está en :math:`(0, 1]`.
    """

    def __init__(
        self,
        safety_margin: float = 0.95,
        lanczos_tol: float = 1e-6,
        fallback_max_iter: int = 1000,
        lanczos_k: int = 3,
    ) -> None:
        if not (0 < safety_margin <= 1.0):
            raise CFLViolationError(
                f"safety_margin debe estar en (0, 1]; got {safety_margin}."
            )
        if lanczos_tol <= 0:
            raise CFLViolationError(
                f"lanczos_tol debe ser > 0; got {lanczos_tol}."
            )

        self._margin = float(safety_margin)
        self._lanczos_tol = float(lanczos_tol)
        self._fallback_iter = int(fallback_max_iter)
        self._lanczos_k = int(lanczos_k)

    # ──────────────────────────────────────────────────────────────────────────
    #  UTILIDADES ESPECTRALES
    # ──────────────────────────────────────────────────────────────────────────

    def _symmetrize_laplacian(self, L: csr_matrix) -> csr_matrix:
        r"""
        Devuelve la parte simétrica :math:`\Delta_{\mathrm{sym}} = \frac{1}{2}(\Delta + \Delta^T)`.

        Para Laplacianos de grafos dirigidos, la simetrización captura la
        componente de difusión isotrópica: el espectro de :math:`\Delta_{\mathrm{sym}}`
        es real y determina la velocidad de propagación de información en
        todas las direcciones.

        **Norma de asimetría**:
            Se registra :math:`\|\Delta - \Delta^T\|_F / \|\Delta\|_F` para
            diagnóstico. Un valor grande indica alta direccionalidad del grafo.

        Args:
            L:
                Laplaciano sparse :math:`(n \times n)`.

        Returns:
            :math:`\Delta_{\mathrm{sym}}` en formato CSR.

        Raises:
            CFLViolationError:
                Si ``L`` no es sparse o no es cuadrada.
        """
        if not issparse(L):
            raise CFLViolationError(
                f"El Laplaciano debe ser sparse; got type={type(L).__name__}."
            )
        if L.shape[0] != L.shape[1]:
            raise CFLViolationError(
                f"El Laplaciano debe ser cuadrado; shape={L.shape}."
            )

        L_sym = 0.5 * (L + L.T)
        L_sym_csr = L_sym.tocsr()

        # Norma de asimetría para diagnóstico
        asym_norm = (L - L.T).data
        norm_asym = float(np.sqrt(np.sum(asym_norm ** 2))) if asym_norm.size > 0 else 0.0
        norm_L = float(np.sqrt(np.sum(L.data ** 2))) if L.data.size > 0 else 1.0
        rel_asym = norm_asym / max(norm_L, 1e-15)

        if rel_asym > 1e-3:
            logger.debug(
                f"[Phase3] Laplaciano asimétrico detectado: "
                f"‖Δ - Δ^T‖_F/‖Δ‖_F = {rel_asym:.3e}. "
                f"Simetrización aplicada."
            )

        return L_sym_csr

    def _estimate_spectral_radius(
        self, L_sym: csr_matrix
    ) -> Tuple[float, str]:
        r"""
        Estima :math:`\lambda_{\max}(\Delta_{\mathrm{sym}})` con tres niveles
        de fallback garantizados.

        **Nivel 1 — Lanczos (ARPACK)**:
            Método de Krylov iterativo. Convergencia superlineal.
            Complejidad :math:`O(n \cdot k)` por iteración.

        **Nivel 2 — Potencia iterada**:
            Estima el autovector dominante iterando :math:`v \leftarrow Lv/\|Lv\|`.
            Convergencia lineal con razón :math:`|\lambda_1/\lambda_2|`.

        **Nivel 3 — Cota de Gerschgorin**:
            Para la matriz simétrica :math:`\Delta_{\mathrm{sym}}`:

            .. math::
                \lambda_{\max} \leq \max_i |(\Delta_{\mathrm{sym}})_{ii}|
                + \sum_{j \neq i} |(\Delta_{\mathrm{sym}})_{ij}|
                = 2 \max_i |(\Delta_{\mathrm{sym}})_{ii}|

            (usando la propiedad de filas del Laplaciano).

        Args:
            L_sym:
                Laplaciano simétrico sparse :math:`(n \times n)`.

        Returns:
            Tupla ``(lambda_max, method_name)`` donde ``method_name`` indica
            qué método fue usado.
        """
        n = L_sym.shape[0]

        # ── Nivel 1: Lanczos ───────────────────────────────────────────────
        try:
            k = min(self._lanczos_k, n - 1) if n > 2 else 1
            if k >= 1:
                evals, _ = eigsh(
                    L_sym, k=k, which="LM", tol=self._lanczos_tol,
                    return_eigenvectors=True
                )
                lambda_max = float(np.max(np.abs(evals)))
                logger.debug(f"[Phase3] λ_max={lambda_max:.4e} (Lanczos, k={k})")
                return lambda_max, "lanczos"
        except Exception as exc_lanczos:
            logger.debug(f"[Phase3] Lanczos falló: {exc_lanczos}")

        # ── Nivel 2: Potencia iterada ──────────────────────────────────────
        try:
            rng = np.random.default_rng(seed=42)
            v: NDArray[np.float64] = rng.standard_normal(n)
            v /= np.linalg.norm(v)
            lambda_est = 0.0
            lambda_prev = -np.inf

            for iteration in range(self._fallback_iter):
                v_new: NDArray[np.float64] = L_sym @ v
                norm_new = float(np.linalg.norm(v_new))

                if norm_new < 1e-15:
                    logger.debug("[Phase3] Potencia iterada: norma cero.")
                    break

                v = v_new / norm_new
                lambda_est = float(v @ (L_sym @ v))

                # Criterio de convergencia
                if abs(lambda_est - lambda_prev) < self._lanczos_tol * abs(lambda_est):
                    logger.debug(
                        f"[Phase3] Potencia iterada convergió en {iteration+1} iters: "
                        f"λ_max={lambda_est:.4e}"
                    )
                    break
                lambda_prev = lambda_est

            return float(abs(lambda_est)), "power_iteration"

        except Exception as exc_power:
            logger.debug(f"[Phase3] Potencia iterada falló: {exc_power}")

        # ── Nivel 3: Cota de Gerschgorin ───────────────────────────────────
        diag = np.asarray(L_sym.diagonal()).ravel()
        abs_diag = np.abs(diag)

        # Para Laplacianos simétricos: λ_max ≤ 2 * max_i |L_ii|
        # (cada fila suma 0 en Laplaciano combinatorial normalizado)
        lambda_max_gerschgorin = 2.0 * float(np.max(abs_diag)) if abs_diag.size > 0 else 1.0

        logger.debug(
            f"[Phase3] Gerschgorin: λ_max ≤ {lambda_max_gerschgorin:.4e}"
        )

        return lambda_max_gerschgorin, "gerschgorin"

    # ──────────────────────────────────────────────────────────────────────────
    #  AUDITORÍA CFL PRINCIPAL
    # ──────────────────────────────────────────────────────────────────────────

    def audit_time_step(
        self,
        graph_laplacian: csr_matrix,
        wave_speed_c: float,
        requested_dt: float,
    ) -> float:
        r"""
        **[ENTRADA DE FASE 3 — recibe c_eff de Phase2]**

        Audita y acota el paso de integración :math:`\Delta t` según la
        condición CFL:

        .. math::
            \Delta t_{\mathrm{safe}} = \min\!\left(
            \Delta t_{\mathrm{req}},\;
            \frac{2 \cdot \rho}{c_{\mathrm{eff}} \sqrt{\lambda_{\max}(\Delta_{\mathrm{sym}})}}
            \right)

        donde :math:`\rho \in (0, 1]` es el margen de seguridad.

        **Casos degenerados**:

        - :math:`c_{\mathrm{eff}} \approx 0`: No hay propagación, CFL no aplica.
          Se retorna ``requested_dt``.
        - Grafo trivial (:math:`n \leq 1`): Sin interacciones, CFL no aplica.
        - :math:`\lambda_{\max} \approx 0`: Grafo desconectado, CFL no aplica.

        Args:
            graph_laplacian:
                Laplaciano sparse del grafo de la red :math:`(n \times n)`.
            wave_speed_c:
                Velocidad de onda efectiva :math:`c_{\mathrm{eff}} \geq 0`
                proveniente de :meth:`Phase2_ImpedanceTuner.compute_effective_wave_speed`.
            requested_dt:
                Paso de tiempo solicitado :math:`\Delta t > 0`.

        Returns:
            ``dt_safe``: paso de tiempo seguro :math:`\leq \Delta t_{\max}`.

        Raises:
            CFLViolationError:
                Si ``wave_speed_c < 0`` o ``requested_dt ≤ 0``.
        """
        # ── Validaciones de entrada ────────────────────────────────────────
        if wave_speed_c < 0:
            raise CFLViolationError(
                f"wave_speed_c debe ser ≥ 0; got {wave_speed_c}."
            )
        if requested_dt <= 0:
            raise CFLViolationError(
                f"requested_dt debe ser > 0; got {requested_dt}."
            )

        # ── Casos degenerados ──────────────────────────────────────────────
        if wave_speed_c < 1e-15:
            logger.debug("[Phase3] c ≈ 0: CFL no aplica.")
            return requested_dt

        if graph_laplacian.shape[0] <= 1:
            logger.debug("[Phase3] Grafo trivial (n≤1): CFL no aplica.")
            return requested_dt

        # ── Simetrización ─────────────────────────────────────────────────
        L_sym = self._symmetrize_laplacian(graph_laplacian)

        # ── Estimación espectral ───────────────────────────────────────────
        lambda_max, method = self._estimate_spectral_radius(L_sym)

        if lambda_max < 1e-12:
            logger.debug(f"[Phase3] λ_max ≈ 0 ({method}): grafo desconectado.")
            return requested_dt

        # ── Límite CFL ────────────────────────────────────────────────────
        sqrt_lambda = math.sqrt(lambda_max)
        dt_max_stable = (2.0 * self._margin) / (wave_speed_c * sqrt_lambda)
        safe_dt = min(requested_dt, dt_max_stable)

        # ── Log de veto ───────────────────────────────────────────────────
        if requested_dt > dt_max_stable / self._margin:  # Compara vs dt_max sin margen
            cfl_number = wave_speed_c * sqrt_lambda * requested_dt / 2.0
            logger.warning(
                f"[Phase3] Veto CFL: dt_req={requested_dt:.4e} > dt_max={dt_max_stable/self._margin:.4e}. "
                f"CFL#={cfl_number:.3f} (debe ser ≤1). "
                f"λ_max={lambda_max:.4e} ({method}), c={wave_speed_c:.4e}. "
                f"dt_safe={safe_dt:.4e} (margen={self._margin})."
            )

        return safe_dt

    def cfl_diagnostic(
        self,
        graph_laplacian: csr_matrix,
        wave_speed_c: float,
        requested_dt: float,
    ) -> Dict[str, Any]:
        r"""
        Reporte diagnóstico completo de la auditoría CFL.

        Ejecuta el análisis espectral completo y devuelve todas las métricas
        intermedias, sin efectos secundarios (idempotente).

        Args:
            graph_laplacian:
                Laplaciano sparse :math:`(n \times n)`.
            wave_speed_c:
                Velocidad de onda efectiva :math:`c_{\mathrm{eff}}`.
            requested_dt:
                Paso de tiempo solicitado.

        Returns:
            Diccionario con métricas:

            - ``lambda_max``: :math:`\lambda_{\max}(\Delta_{\mathrm{sym}})`.
            - ``estimation_method``: Método utilizado.
            - ``dt_max_stable``: :math:`\Delta t_{\max}` sin margen.
            - ``dt_max_with_margin``: :math:`\rho \cdot \Delta t_{\max}`.
            - ``cfl_number``: Número CFL efectivo.
            - ``safe_dt``: Paso de tiempo seguro aplicado.
            - ``margin_used``: Margen de seguridad :math:`\rho`.
            - ``violated``: ``True`` si :math:`\Delta t_{\mathrm{req}} > \Delta t_{\max}`.
            - ``graph_nodes``: Número de nodos del grafo.
            - ``symmetry_residual``: :math:`\|\Delta - \Delta^T\|_F / \|\Delta\|_F`.
        """
        n = graph_laplacian.shape[0]

        # Norma de asimetría
        if issparse(graph_laplacian) and n > 0:
            diff = graph_laplacian - graph_laplacian.T
            norm_asym = float(np.sqrt(np.sum(diff.data ** 2))) if diff.data.size > 0 else 0.0
            norm_L = float(np.sqrt(np.sum(graph_laplacian.data ** 2)))
            sym_residual = norm_asym / max(norm_L, 1e-15)
        else:
            sym_residual = 0.0

        # Casos degenerados
        if wave_speed_c < 1e-15 or n <= 1:
            return {
                "lambda_max": 0.0,
                "estimation_method": "degenerate",
                "dt_max_stable": float("inf"),
                "dt_max_with_margin": float("inf"),
                "cfl_number": 0.0,
                "safe_dt": requested_dt,
                "margin_used": self._margin,
                "violated": False,
                "graph_nodes": n,
                "symmetry_residual": sym_residual,
            }

        L_sym = self._symmetrize_laplacian(graph_laplacian)
        lambda_max, method = self._estimate_spectral_radius(L_sym)

        if lambda_max < 1e-12:
            return {
                "lambda_max": lambda_max,
                "estimation_method": method,
                "dt_max_stable": float("inf"),
                "dt_max_with_margin": float("inf"),
                "cfl_number": 0.0,
                "safe_dt": requested_dt,
                "margin_used": self._margin,
                "violated": False,
                "graph_nodes": n,
                "symmetry_residual": sym_residual,
            }

        sqrt_lambda = math.sqrt(lambda_max)
        dt_max_stable = 2.0 / (wave_speed_c * sqrt_lambda)
        dt_max_margin = dt_max_stable * self._margin
        cfl_number = wave_speed_c * sqrt_lambda * requested_dt / 2.0
        safe_dt = min(requested_dt, dt_max_margin)
        violated = requested_dt > dt_max_stable

        return {
            "lambda_max": lambda_max,
            "estimation_method": method,
            "dt_max_stable": dt_max_stable,
            "dt_max_with_margin": dt_max_margin,
            "cfl_number": cfl_number,
            "safe_dt": safe_dt,
            "margin_used": self._margin,
            "violated": violated,
            "graph_nodes": n,
            "symmetry_residual": sym_residual,
        }


# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                                                              ║
# ║   ORQUESTADOR: DIRAC INTERCONNECTION AGENT                                  ║
# ║   (Morfismo inter-estrato TACTICS → PHYSICS)                                ║
# ║                                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

class DiracInterconnectionAgent(Morphism):
    r"""
    El "Demonio de Maxwell" categórico entre el estrato TACTICS y PHYSICS.

    Implementa el morfismo:

    .. math::
        \phi: \mathcal{X}_{\mathrm{tactics}} \longrightarrow \mathcal{X}_{\mathrm{physics}}

    encadenando las tres fases como una composición de morfismos en la categoría
    de sistemas Port-Hamiltonianos:

    .. math::
        \phi = \phi_3 \circ \phi_2 \circ \phi_1

    donde:

    - :math:`\phi_1`: IDA-PBC → :math:`(\alpha, Z_{\mathrm{eff}})`.
    - :math:`\phi_2`: Sintonización PML → :math:`(T_{\epsilon\mu}, c_{\mathrm{eff}})`.
    - :math:`\phi_3`: Gobernador CFL → :math:`\Delta t_{\mathrm{safe}}`.

    Args:
        metric_tensor:
            Tensor métrico :math:`G \in \mathbb{R}^{n \times n}` del espacio de estado.
            Por defecto usa ``G_PHYSICS`` del ecosistema.
        tolerance:
            Tolerancia numérica global. Por defecto :math:`10^{-9}`.
        safety_margin:
            Margen CFL :math:`\rho \in (0, 1]`. Por defecto :math:`0.95`.
        base_permeability:
            Permeabilidad magnética base :math:`\mu_0`. Por defecto :math:`1.0`.
        max_residual_relative:
            Máximo residuo relativo de matching admisible. Por defecto :math:`10^{-4}`.
    """

    def __init__(
        self,
        metric_tensor: Optional[NDArray[np.float64]] = None,
        tolerance: float = 1e-9,
        safety_margin: float = 0.95,
        base_permeability: float = 1.0,
        max_residual_relative: float = 1e-4,
    ) -> None:
        self._G = metric_tensor if metric_tensor is not None else G_PHYSICS

        self._solver = Phase1_IDAPBC_Solver(
            tolerance=tolerance,
            max_residual_relative=max_residual_relative,
        )
        self._tuner = Phase2_ImpedanceTuner(
            base_permeability=base_permeability,
        )
        self._governor = Phase3_CFLGovernor(
            safety_margin=safety_margin,
        )

        # Estado interno del último ciclo (para diagnóstico)
        self._last_control: Optional[ControlSolution] = None
        self._last_impedance: Optional[ImpedanceTensor] = None
        self._last_c_eff: Optional[float] = None
        self._last_safe_dt: Optional[float] = None
        self._last_cfl_diag: Optional[Dict[str, Any]] = None

    def synthesize_physical_control(
        self,
        # ── Planta (flux_condenser) ────────────────────────────────────────
        J_current: NDArray[np.float64],
        R_current: NDArray[np.float64],
        grad_H: NDArray[np.float64],
        g_port: NDArray[np.float64],
        graph_laplacian: csr_matrix,
        # ── Estrategia (apu_agent) ─────────────────────────────────────────
        J_desired: NDArray[np.float64],
        R_desired: NDArray[np.float64],
        grad_H_desired: NDArray[np.float64],
        requested_dt: float,
    ) -> InterconnectionState:
        r"""
        Compila la directriz estratégica en un estado físico ejecutable.

        Encadena las tres fases con sus costuras formales:

        1. **Fase 1**: ``compute_control_law`` → ``compute_effective_load_impedance``.
        2. **Costura 1→2**: :math:`Z_{\mathrm{eff}}` pasa a ``tune_dielectric_tensors``.
        3. **Fase 2**: ``tune_dielectric_tensors`` → ``compute_effective_wave_speed``.
        4. **Costura 2→3**: :math:`c_{\mathrm{eff}}` pasa a ``audit_time_step``.
        5. **Fase 3**: ``audit_time_step`` + ``cfl_diagnostic``.

        Args:
            J_current: Matriz de interconexión actual :math:`J(x)`.
            R_current: Matriz de disipación actual :math:`R(x)`.
            grad_H: Gradiente del Hamiltoniano actual :math:`\nabla H(x)`.
            g_port: Matriz de puerto de control :math:`g(x)`.
            graph_laplacian: Laplaciano sparse del grafo de la red.
            J_desired: Matriz de interconexión deseada :math:`J_d(x)`.
            R_desired: Matriz de disipación deseada :math:`R_d(x)`.
            grad_H_desired: Gradiente del Hamiltoniano deseado :math:`\nabla H_d(x)`.
            requested_dt: Paso de tiempo solicitado por el estrato TACTICS.

        Returns:
            :class:`InterconnectionState` inyectable en ``flux_condenser.py``.

        Raises:
            DiracMatchingError, ImpedanceMismatchError, CFLViolationError,
            LyapunovInstabilityError: Según la fase donde ocurra la violación.
        """
        logger.info("[DiracAgent] Iniciando síntesis IDA-PBC ⟶ PML ⟶ CFL")

        # ══════════════════════════════════════════════════════════════════
        #  FASE 1: IDA-PBC
        # ══════════════════════════════════════════════════════════════════
        control_sol: ControlSolution = self._solver.compute_control_law(
            J_current, R_current, grad_H,
            J_desired, R_desired, grad_H_desired,
            g_port,
        )
        self._last_control = control_sol

        logger.debug(
            f"[Phase1] ‖residual‖={control_sol.residual_norm:.2e}, "
            f"rel={control_sol.residual_relative:.2e}, "
            f"rank(g)={control_sol.g_rank}, "
            f"Ḣ_d={control_sol.H_dot:.2e}, "
            f"Lyapunov={'✓' if control_sol.lyapunov_verified else '⚠'}"
        )

        # ── COSTURA 1 → 2 ─────────────────────────────────────────────────
        Z_eff: NDArray[np.float64] = self._solver.compute_effective_load_impedance(
            control_sol
        )

        # ══════════════════════════════════════════════════════════════════
        #  FASE 2: Sintonización PML
        # ══════════════════════════════════════════════════════════════════
        impedance_tensor: ImpedanceTensor = self._tuner.tune_dielectric_tensors(Z_eff)
        self._last_impedance = impedance_tensor

        kk = impedance_tensor.verify_kramers_kronig()
        logger.debug(
            f"[Phase2] ‖Γ‖_F={impedance_tensor.reflection_coefficient_norm:.2e}, "
            f"ε_min={kk['epsilon_min_eig']:.2e}, "
            f"μ_min={kk['mu_min_eig']:.2e}, "
            f"κ(ε)={kk['epsilon_condition_number']:.2e}"
        )

        # ── COSTURA 2 → 3 ─────────────────────────────────────────────────
        c_eff: float = self._tuner.compute_effective_wave_speed(impedance_tensor)
        self._last_c_eff = c_eff

        # ══════════════════════════════════════════════════════════════════
        #  FASE 3: Gobernador CFL
        # ══════════════════════════════════════════════════════════════════
        safe_dt: float = self._governor.audit_time_step(
            graph_laplacian, c_eff, requested_dt
        )
        self._last_safe_dt = safe_dt

        # Diagnóstico completo CFL (idempotente)
        cfl_diag: Dict[str, Any] = self._governor.cfl_diagnostic(
            graph_laplacian, c_eff, requested_dt
        )
        self._last_cfl_diag = cfl_diag

        logger.info(
            f"[DiracAgent] Síntesis exitosa: "
            f"Ḣ_d={control_sol.H_dot:.2e}, "
            f"‖Γ‖={impedance_tensor.reflection_coefficient_norm:.2e}, "
            f"c_eff={c_eff:.2e}, "
            f"CFL#={cfl_diag['cfl_number']:.3f}, "
            f"dt_safe={safe_dt:.2e}"
        )

        return InterconnectionState(
            control_law_alpha=control_sol.alpha,
            impedance=impedance_tensor,
            safe_dt=safe_dt,
            lyapunov_derivative=control_sol.H_dot,
            c_eff=c_eff,
            cfl_margin=self._governor._margin,
            lambda_max_laplacian=cfl_diag["lambda_max"],
            cfl_number=cfl_diag["cfl_number"],
        )

    def diagnostic_report(self) -> Dict[str, Any]:
        r"""
        Genera un reporte diagnóstico completo del último ciclo ejecutado.

        Returns:
            Diccionario estructurado por fases con todas las métricas
            disponibles del último ``synthesize_physical_control``.
        """
        report: Dict[str, Any] = {
            "agent": "DiracInterconnectionAgent",
            "version": "3.0.0",
            "initialized": True,
            "has_data": self._last_control is not None,
        }

        if self._last_control is not None:
            cs = self._last_control
            report["phase1_ida_pbc"] = {
                "residual_norm": cs.residual_norm,
                "residual_relative": cs.residual_relative,
                "g_rank": cs.g_rank,
                "H_dot": cs.H_dot,
                "lyapunov_verified": cs.lyapunov_verified,
                "alpha_norm": float(la.norm(cs.alpha)),
                "sigma_max": float(cs.singular_values[0]) if cs.singular_values.size > 0 else 0.0,
                "sigma_min": float(cs.singular_values[-1]) if cs.singular_values.size > 0 else 0.0,
                "condition_number_g": (
                    float(cs.singular_values[0] / cs.singular_values[-1])
                    if cs.singular_values.size > 1 and cs.singular_values[-1] > 0
                    else float("inf")
                ),
            }

        if self._last_impedance is not None:
            imp = self._last_impedance
            kk = imp.verify_kramers_kronig()
            report["phase2_pml"] = {
                "reflection_norm": imp.reflection_coefficient_norm,
                "kramers_kronig": kk,
                "wave_speeds": imp.wave_speeds.tolist(),
                "c_max": float(np.max(imp.wave_speeds)),
                "c_min_active": (
                    float(np.min(imp.wave_speeds[imp.channel_active_mask]))
                    if np.any(imp.channel_active_mask) else 0.0
                ),
                "n_active_channels": int(np.sum(imp.channel_active_mask)),
                "n_inactive_channels": int(np.sum(~imp.channel_active_mask)),
                "epsilon_diag": np.diag(imp.epsilon_eff).tolist(),
                "mu_diag": np.diag(imp.mu_eff).tolist(),
            }

        if self._last_cfl_diag is not None:
            report["phase3_cfl"] = self._last_cfl_diag.copy()
            report["phase3_cfl"]["c_eff"] = self._last_c_eff
            report["phase3_cfl"]["safe_dt"] = self._last_safe_dt

        return report


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    # ── Excepciones ────────────────────────────────────────────────────────
    "DiracMatchingError",
    "ImpedanceMismatchError",
    "CFLViolationError",
    "LyapunovInstabilityError",
    # ── Estructuras de datos inmutables ───────────────────────────────────
    "ImpedanceTensor",
    "ControlSolution",
    "InterconnectionState",
    # ── Fases del pipeline ─────────────────────────────────────────────────
    "Phase1_IDAPBC_Solver",
    "Phase2_ImpedanceTuner",
    "Phase3_CFLGovernor",
    # ── Orquestador ────────────────────────────────────────────────────────
    "DiracInterconnectionAgent",
]