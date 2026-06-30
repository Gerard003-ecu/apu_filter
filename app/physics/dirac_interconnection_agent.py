# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo: Dirac Interconnection Agent (Asignación de Amortiguamiento IDA-PBC)  ║
║ Ubicación: app/physics/dirac_interconnection_agent.py                        ║
║ Versión: 6.0.0-IDA-PBC-CFL-Governor-Rigorous-Spectral                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Naturaleza Ciber-Física y Teoría de Control No Lineal:
────────────────────────────────────────────────────────────────────────────────
Este módulo consagra la aduana termodinámica entre el estrato TACTICS (`apu_agent.py`) 
y el estrato PHYSICS (`flux_condenser.py`). Opera como un "Demonio de Maxwell" que 
esculpe el Hamiltoniano del sistema en ciclo cerrado para garantizar la estabilidad 
asintótica sin importar cuán entrópica sea la directriz estratégica.

Axiomas de Ejecución:
§1. ESTRUCTURA DE DIRAC Y ENERGY SHAPING (IDA-PBC):
    Resuelve la Ecuación de Matching para encontrar la ley de control $\alpha(x)$:
    $$ [J_d(x) - R_d(x)] \nabla H_d(x) = [J(x) - R(x)] \nabla H(x) + g(x)\alpha(x) $$
    Garantizando axiomáticamente la disipación: $\dot{H}_d \le 0$.

§2. SINTONIZACIÓN DINÁMICA DE IMPEDANCIA (PML):
    Aniquila el coeficiente de reflexión $\Gamma$ sintonizando los tensores
    $\mu, \epsilon$ con restricción de Kramers-Kronig:
    $$ Z_0 = \sqrt{\mu_{\text{eff}} \epsilon_{\text{eff}}^{-1}} \equiv Z_{\text{load}} $$
    $$ \Gamma = \frac{Z_{\text{load}} - Z_0}{Z_{\text{load}} + Z_0} = 0 $$

§3. GOBERNANZA DEL LÍMITE DE COURANT-FRIEDRICHS-LEWY (CFL):
    Audita el Laplaciano del grafo y restringe el paso de integración $\Delta t$:
    $$ \Delta t \le \frac{2}{c_{\text{eff}} \sqrt{\lambda_{\max}(\Delta_{\text{sym}})}} $$
    donde $\Delta_{\text{sym}} = \tfrac{1}{2}(\Delta + \Delta^T)$ es la parte simétrica.

Las tres fases se encadenan rigurosamente:
- **Fase 1** resuelve IDA-PBC y extrae la impedancia efectiva del puerto de control.
- **Fase 2** sintoniza los tensores dieléctricos para acoplar perfectamente esa impedancia.
- **Fase 3** calcula la velocidad de onda efectiva y aplica la restricción CFL.

Mejoras v6.0:
    • Validación de rango de g(x) con SVD y condición de matching verificable.
    • Restricción de Kramers-Kronig en tensores ε, μ (positividad completa).
    • Soporte para Laplacianos asimétricos mediante simetrización explícita.
    • Verificación de Lyapunov a lo largo de trayectorias (no solo en un punto).
    • Manejo de canales inactivos con asignación explícita de Z=∞.
    • Logging diagnóstico con residuales de cada axioma.
═══════════════════════════════════════════════════════════════════════════════
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
from scipy.sparse.linalg import eigsh, LinearOperator

# ══════════════════════════════════════════════════════════════════════════════
# DEPENDENCIAS ESTRUCTURALES DEL ECOSISTEMA (resilientes)
# ══════════════════════════════════════════════════════════════════════════════
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:
    class TopologicalInvariantError(Exception):
        pass

    class Morphism:
        pass

try:
    from app.core.immune_system.metric_tensors import G_PHYSICS
except ImportError:
    G_PHYSICS: NDArray[np.float64] = np.eye(1, dtype=np.float64)

logger = logging.getLogger("MIC.Physics.DiracInterconnection")

# ══════════════════════════════════════════════════════════════════════════════
# EXCEPCIONES DE CONTROL Y ESTABILIDAD
# ══════════════════════════════════════════════════════════════════════════════
class DiracMatchingError(TopologicalInvariantError):
    """Detonada si la Ecuación de Matching IDA-PBC no tiene solución exacta en la imagen de g(x)."""
    pass


class ImpedanceMismatchError(TopologicalInvariantError):
    """Detonada si la sintonización dieléctrica induce un tensor no definido positivo."""
    pass


class CFLViolationError(TopologicalInvariantError):
    """Detonada si el diferencial de tiempo exigido rompe el cono de luz causal de la red."""
    pass


class LyapunovInstabilityError(TopologicalInvariantError):
    """Detonada si la verificación de Lyapunov detecta crecimiento de H_d."""
    pass


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURAS INMUTABLES DEL ESPACIO DE CONTROL
# ══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class ImpedanceTensor:
    r"""
    Tensor dieléctrico y magnético sintonizado para adaptación perfecta.

    Atributos:
        epsilon_eff: Matriz $(m \times m)$ del tensor dieléctrico efectivo.
        mu_eff: Matriz $(m \times m)$ del tensor de permeabilidad.
        reflection_coefficient_norm: $\|\Gamma\|_F$ residual post-sintonización.
        wave_speeds: Vector $(m,)$ con $c_i = 1/\sqrt{\mu_i \epsilon_i}$ por canal.
        is_isotropic: Si True, los tensores son escalares por canal.
    """
    epsilon_eff: NDArray[np.float64]
    mu_eff: NDArray[np.float64]
    reflection_coefficient_norm: float
    wave_speeds: NDArray[np.float64]
    is_isotropic: bool

    def verify_kramers_kronig(self, tol: float = 1e-9) -> Dict[str, float]:
        r"""
        Verifica la positividad de los tensores dieléctricos (condición Kramers-Kronig).

        Returns:
            Dict con métricas:
                - 'epsilon_min_eig': Mínimo autovalor de $\epsilon$.
                - 'mu_min_eig': Mínimo autovalor de $\mu$.
                - 'epsilon_pd': Si $\epsilon \succeq 0$.
                - 'mu_pd': Si $\mu \succeq 0$.
        """
        eps_eigs = la.eigvalsh(self.epsilon_eff)
        mu_eigs = la.eigvalsh(self.mu_eff)
        return {
            "epsilon_min_eig": float(np.min(eps_eigs)),
            "mu_min_eig": float(np.min(mu_eigs)),
            "epsilon_pd": bool(np.min(eps_eigs) >= -tol),
            "mu_pd": bool(np.min(mu_eigs) >= -tol),
        }


@dataclass(frozen=True, slots=True)
class ControlSolution:
    r"""
    Resultado de la Fase 1: ley de control y derivada de Lyapunov.

    Atributos:
        alpha: Vector $(m,)$ con la ley de control.
        H_dot: Derivada temporal del Hamiltoniano deseado.
        desired_gradient: Vector $(n,)$ con $\nabla H_d$.
        port_matrix: Matriz $(n, m)$ de $g(x)$.
        residual_norm: Norma del residuo de matching $\|f_d - g\alpha\|$.
        g_rank: Rango numérico de $g$.
        lyapunov_verified: Si la condición de Lyapunov se certificó.
    """
    alpha: NDArray[np.float64]
    H_dot: float
    desired_gradient: NDArray[np.float64]
    port_matrix: NDArray[np.float64]
    residual_norm: float
    g_rank: int
    lyapunov_verified: bool


@dataclass(frozen=True, slots=True)
class InterconnectionState:
    r"""
    Estado consolidado inyectable directamente en `flux_condenser.py`.

    Atributos:
        control_law_alpha: Ley de control IDA-PBC.
        impedance: Tensor de impedancia sintonizado.
        safe_dt: Paso de tiempo seguro según CFL.
        lyapunov_derivative: $\dot{H}_d$ certificado.
        c_eff: Velocidad de onda efectiva máxima.
        cfl_margin: Margen de seguridad CFL usado.
    """
    control_law_alpha: NDArray[np.float64]
    impedance: ImpedanceTensor
    safe_dt: float
    lyapunov_derivative: float
    c_eff: float
    cfl_margin: float


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 1 · RESOLUCIÓN IDA-PBC Y CÁLCULO DE IMPEDANCIA EFECTIVA          ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase1_IDAPBC_Solver:
    r"""
    Resuelve la Ecuación de Matching para sistemas Port-Hamiltonianos.

    Ecuación central:
    $$
    [J_d - R_d] \nabla H_d = [J - R] \nabla H + g \alpha
    $$

    con:
    - $J = -J^T$ (antisimétrica, interconexión conservativa)
    - $R = R^T \succeq 0$ (simétrica PSD, disipación)
    - Análogas para $(J_d, R_d)$

    Validaciones:
    - Rango completo de $g$ (rango = número de actuadores).
    - Residuo de matching $\|f_d - g\alpha\|$ bajo tolerancia.
    - $\dot{H}_d \le 0$ en el gradiente verificado.
    """

    def __init__(
        self,
        tolerance: float = 1e-9,
        relative_tol: float = 1e-6,
        require_full_rank: bool = True,
    ):
        self._tol = tolerance
        self._rel_tol = relative_tol
        self._require_full_rank = require_full_rank

    # ──────────────────────────────────────────────────────────────────────────
    # Validaciones de matrices estructurales
    # ──────────────────────────────────────────────────────────────────────────
    def _validate_skew_symmetry(self, M: NDArray[np.float64], name: str) -> None:
        """Verifica $M = -M^T$."""
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise DiracMatchingError(f"{name} debe ser matriz cuadrada; shape={M.shape}")
        residual = la.norm(M + M.T, ord='fro') / max(1.0, la.norm(M, ord='fro'))
        if residual > self._rel_tol:
            raise DiracMatchingError(
                f"{name} no es antisimétrica; ‖M + M^T‖_F/‖M‖_F = {residual:.2e}"
            )

    def _validate_symmetry_psd(self, M: NDArray[np.float64], name: str) -> None:
        """Verifica $M = M^T$ y $M \succeq 0$."""
        if M.ndim != 2 or M.shape[0] != M.shape[1]:
            raise DiracMatchingError(f"{name} debe ser matriz cuadrada; shape={M.shape}")
        sym_res = la.norm(M - M.T, ord='fro') / max(1.0, la.norm(M, ord='fro'))
        if sym_res > self._rel_tol:
            raise DiracMatchingError(
                f"{name} no es simétrica; residual={sym_res:.2e}"
            )
        eigvals = la.eigvalsh(M)
        min_eig = float(np.min(eigvals))
        if min_eig < -self._tol:
            raise DiracMatchingError(
                f"{name} no es semidefinida positiva; λ_min = {min_eig:.2e}"
            )

    def _validate_port_matrix(self, g: NDArray[np.float64], grad_dim: int) -> Tuple[int, int]:
        """Valida dimensiones y calcula rango de g."""
        if g.ndim != 2:
            raise DiracMatchingError(f"g_port debe ser 2D; ndim={g.ndim}")
        n_g, m_g = g.shape
        if n_g != grad_dim:
            raise DiracMatchingError(
                f"g_port tiene {n_g} filas, se esperaban {grad_dim} (igual a dim(∇H))"
            )
        # Rango numérico via SVD
        sv = la.svdvals(g)
        g_rank = int(np.sum(sv > self._tol * sv[0] if sv.size > 0 else [0.0]))
        if self._require_full_rank and g_rank < min(n_g, m_g):
            raise DiracMatchingError(
                f"g_port no es de rango completo: rank={g_rank}, esperado={min(n_g, m_g)}"
            )
        return n_g, m_g

    # ──────────────────────────────────────────────────────────────────────────
    # Resolución IDA-PBC
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
        Resuelve la ecuación de matching mediante pseudoinversa de Moore-Penrose
        con verificación de rango, residuo y Lyapunov.

        Returns:
            ControlSolution con todos los metadatos diagnósticos.
        """
        # 1. Validación estructural
        self._validate_skew_symmetry(J_current, "J_current")
        self._validate_skew_symmetry(J_desired, "J_desired")
        self._validate_symmetry_psd(R_current, "R_current")
        self._validate_symmetry_psd(R_desired, "R_desired")

        if grad_H.ndim != 1:
            raise DiracMatchingError(f"grad_H debe ser vector; ndim={grad_H.ndim}")
        if grad_H_desired.ndim != 1:
            raise DiracMatchingError(f"grad_H_desired debe ser vector; ndim={grad_H_desired.ndim}")

        n = grad_H.shape[0]
        if grad_H.shape[0] != grad_H_desired.shape[0]:
            raise DiracMatchingError(
                f"grad_H y grad_H_desired difieren en dimensión: {n} vs {grad_H_desired.shape[0]}"
            )
        if J_current.shape[0] != n:
            raise DiracMatchingError(
                f"J_current ({J_current.shape}) inconsistente con grad_H ({n})"
            )

        n_g, m_g = self._validate_port_matrix(g_port, n)

        # 2. Dinámicas
        open_loop_dyn = (J_current - R_current) @ grad_H
        desired_dyn = (J_desired - R_desired) @ grad_H_desired
        required_forcing = desired_dyn - open_loop_dyn

        # 3. Resolución por pseudoinversa
        g_pinv = la.pinv(g_port, rcond=self._tol)
        alpha_x = g_pinv @ required_forcing

        # 4. Verificación del residuo
        reconstructed = g_port @ alpha_x
        residual = required_forcing - reconstructed
        norm_res = float(la.norm(residual))
        norm_required = float(la.norm(required_forcing))
        relative_res = norm_res / max(norm_required, 1e-15)

        if relative_res > self._rel_tol * 100:
            raise DiracMatchingError(
                f"Residuo de matching excesivo: ‖f_d - gα‖/‖f_d‖ = {relative_res:.2e}. "
                f"Posible subactuación."
            )

        # 5. Certificación de Lyapunov
        # Para sistemas PCH: dH_d/dt = ∇H_d^T (J_d - R_d) ∇H_d
        # Como J_d es antisimétrica: ∇H_d^T J_d ∇H_d = 0
        # Entonces: dH_d/dt = -∇H_d^T R_d ∇H_d ≤ 0
        H_dot = float(-grad_H_desired @ R_desired @ grad_H_desired)

        lyapunov_ok = H_dot <= self._tol
        if not lyapunov_ok:
            logger.warning(
                f"Lyapunov en el límite: Ḣ_d = {H_dot:.2e}. "
                "Verificar R_desired."
            )

        # 6. Cálculo de rango
        sv = la.svdvals(g_port)
        g_rank = int(np.sum(sv > self._tol * sv[0] if sv.size > 0 else [0.0]))

        return ControlSolution(
            alpha=alpha_x,
            H_dot=H_dot,
            desired_gradient=grad_H_desired,
            port_matrix=g_port,
            residual_norm=norm_res,
            g_rank=g_rank,
            lyapunov_verified=lyapunov_ok,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Cálculo de impedancia efectiva
    # ──────────────────────────────────────────────────────────────────────────
    def compute_effective_load_impedance(
        self, control_solution: ControlSolution
    ) -> NDArray[np.float64]:
        r"""
        Calcula la impedancia efectiva por canal del puerto de control:
        $$
        Z_{\text{eff},i} = \frac{\alpha_i}{(g^T \nabla H_d)_i}
        $$

        Canales con $(g^T \nabla H_d)_i \approx 0$ reciben $Z = \infty$ (sin acoplamiento).

        Returns:
            Vector $(m,)$ de impedancias efectivas (algunos pueden ser $\infty$).
        """
        alpha = control_solution.alpha
        g = control_solution.port_matrix
        grad_Hd = control_solution.desired_gradient

        # Salida del puerto en lazo cerrado
        y_d = g.T @ grad_Hd

        Z_eff = np.full(alpha.shape, np.inf, dtype=np.float64)
        active = np.abs(y_d) > self._tol

        if np.any(active):
            Z_eff[active] = alpha[active] / y_d[active]

        # Verificación: no debe haber NaN en canales activos
        if np.any(np.isnan(Z_eff[active])):
            raise DiracMatchingError(
                "Impedancia efectiva mal condicionada: NaN en canales activos."
            )

        n_active = int(np.sum(active))
        n_inf = int(np.sum(np.isinf(Z_eff)))

        logger.debug(
            f"Impedancia efectiva: {n_active} canales activos, "
            f"{n_inf} canales inactivos (Z=∞)"
        )

        return Z_eff

    def verify_trajectory_lyapunov(
        self,
        R_desired: NDArray[np.float64],
        trajectory_gradients: NDArray[np.float64],
        max_growth: float = 1e-6,
    ) -> Dict[str, float]:
        r"""
        Verifica $\dot{H}_d \le 0$ a lo largo de una trayectoria de gradientes,
        no solo en un punto.

        Args:
            R_desired: Matriz de disipación deseada.
            trajectory_gradients: Matriz $(T, n)$ con gradientes en $T$ instantes.
            max_growth: Crecimiento máximo permitido de $H_d$ entre instantes.

        Returns:
            Dict con métricas:
                - 'H_dot_min': Mínimo $\dot{H}_d$ en la trayectoria.
                - 'H_dot_max': Máximo $\dot{H}_d$ en la trayectoria.
                - 'violations': Número de instantes con $\dot{H}_d > 0$.
                - 'lyapunov_stable': Si la trayectoria es Lyapunov-estable.
        """
        if trajectory_gradients.ndim != 2:
            raise DiracMatchingError(
                f"trajectory_gradients debe ser 2D; ndim={trajectory_gradients.ndim}"
            )

        T = trajectory_gradients.shape[0]
        H_dots = np.zeros(T)
        for t in range(T):
            grad = trajectory_gradients[t, :]
            H_dots[t] = float(-grad @ R_desired @ grad)

        violations = int(np.sum(H_dots > self._tol))
        stable = violations == 0

        if not stable:
            raise LyapunovInstabilityError(
                f"{violations}/{T} violaciones de Lyapunov detectadas "
                f"(Ḣ_d max = {np.max(H_dots):.2e})"
            )

        return {
            "H_dot_min": float(np.min(H_dots)),
            "H_dot_max": float(np.max(H_dots)),
            "violations": violations,
            "lyapunov_stable": stable,
        }


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 2 · SINTONIZADOR DINÁMICO DE IMPEDANCIA (PML)                     ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase2_ImpedanceTuner:
    r"""
    Sintoniza los tensores dieléctricos $\epsilon$ y $\mu$ para acoplamiento perfecto
    de impedancia.

    Refactorización v6:
    • Anisotropía completa con matrices diagonales (no escalares).
    • Restricción de Kramers-Kronig: $\epsilon, \mu \succeq 0$.
    • Manejo riguroso de canales inactivos ($Z = \infty$).
    • Cálculo exacto de $c_{\text{eff}}$ por canal con estadísticas.
    """

    def __init__(
        self,
        base_permeability: float = 1.0,
        max_permittivity: float = 1e6,
        min_wave_speed: float = 1e-6,
    ):
        self._mu_base = base_permeability
        self._eps_max = max_permittivity
        self._c_min = min_wave_speed

        if base_permeability <= 0:
            raise ImpedanceMismatchError(
                f"base_permeability debe ser positivo; got {base_permeability}"
            )

    def tune_dielectric_tensors(
        self, target_load_impedances: NDArray[np.float64]
    ) -> ImpedanceTensor:
        r"""
        Sintoniza $\epsilon_{\text{eff}}$ tal que $Z_0 = \sqrt{\mu_{\text{base}}/\epsilon} = Z_{\text{load}}$:
        $$
        \epsilon_{\text{eff},i} = \frac{\mu_{\text{base}}}{Z_{\text{load},i}^2}
        $$

        Args:
            target_load_impedances: Vector $(m,)$ con $Z_{\text{load},i}$ (puede contener $\infty$).

        Returns:
            ImpedanceTensor con tensores diagonalizados y métricas.
        """
        Z = np.asarray(target_load_impedances, dtype=np.float64)
        if Z.ndim != 1:
            raise ImpedanceMismatchError(
                f"target_load_impedances debe ser 1D; ndim={Z.ndim}"
            )
        if np.any(np.isnan(Z)):
            raise ImpedanceMismatchError("target_load_impedances contiene NaN")

        m = Z.shape[0]

        # Manejo de canales inactivos (Z = ∞)
        active = np.isfinite(Z) & (Z > 0)
        inactive = ~active

        # Inicializar epsilon
        epsilon_diag = np.zeros(m, dtype=np.float64)

        # Solo calcular epsilon para canales activos
        if np.any(active):
            Z_active = Z[active]
            # Clip para evitar singularidades numéricas
            Z_active_clipped = np.clip(Z_active, a_min=1e-6, a_max=None)
            epsilon_active = self._mu_base / (Z_active_clipped ** 2)
            # Clip superior para evitar materiales físicamente imposibles
            epsilon_active = np.clip(epsilon_active, a_min=0, a_max=self._eps_max)
            epsilon_diag[active] = epsilon_active

        # Mu constante para todo el medio (hipótesis simplificada)
        mu_diag = np.full(m, self._mu_base, dtype=np.float64)

        # ── Validación Kramers-Kronig ────────────────────────────────────────
        eps_mat = np.diag(epsilon_diag)
        mu_mat = np.diag(mu_diag)

        eps_eigs = la.eigvalsh(eps_mat)
        mu_eigs = la.eigvalsh(mu_mat)

        if np.any(eps_eigs < -self._c_min):
            raise ImpedanceMismatchError(
                f"ε no PSD; λ_min = {eps_eigs.min():.2e}"
            )
        if np.any(mu_eigs < -self._c_min):
            raise ImpedanceMismatchError(
                f"μ no PSD; λ_min = {mu_eigs.min():.2e}"
            )

        # ── Verificación del coeficiente de reflexión ──────────────────────
        # Para canales activos, Γ debe ser exactamente 0
        # Para canales inactivos, Γ no aplica
        if np.any(active):
            Z0_active = np.sqrt(mu_diag[active] / np.maximum(epsilon_diag[active], 1e-15))
            gamma_per_channel = (Z[active] - Z0_active) / (Z[active] + Z0_active)
            gamma_norm = float(la.norm(gamma_per_channel))
        else:
            gamma_norm = 0.0

        # ── Velocidades por canal ──────────────────────────────────────────
        wave_speeds = np.zeros(m, dtype=np.float64)
        valid_speed = active & (epsilon_diag > 1e-15) & (mu_diag > 1e-15)
        wave_speeds[valid_speed] = 1.0 / np.sqrt(mu_diag[valid_speed] * epsilon_diag[valid_speed])
        # Canales inactivos tienen c = 0 (sin propagación útil)

        return ImpedanceTensor(
            epsilon_eff=eps_mat,
            mu_eff=mu_mat,
            reflection_coefficient_norm=gamma_norm,
            wave_speeds=wave_speeds,
            is_isotropic=True,  # Por construcción son diagonales
        )

    def compute_effective_wave_speed(self, impedance: ImpedanceTensor) -> float:
        r"""
        Calcula $c_{\text{eff}}$ como máximo de las velocidades por canal:
        $$
        c_{\text{eff}} = \max_i \frac{1}{\sqrt{\mu_i \epsilon_i}}
        $$

        Esta es la elección conservadora para la condición CFL: garantiza
        que ningún canal viole el límite de estabilidad.
        """
        c_per_channel = impedance.wave_speeds

        if not np.any(c_per_channel > 0):
            raise ImpedanceMismatchError(
                "Ningún canal tiene velocidad de onda positiva; "
                "el medio está completamente apagado."
            )

        c_eff = float(np.max(c_per_channel))

        if c_eff < self._c_min:
            raise ImpedanceMismatchError(
                f"c_eff = {c_eff:.2e} < c_min = {self._c_min}"
            )

        return c_eff


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║   FASE 3 · GOBERNADOR DEL LÍMITE DE COURANT-FRIEDRICHS-LEWY (CFL)       ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class Phase3_CFLGovernor:
    r"""
    Custodio del cono de luz causal en grafos computacionales.

    Para un grafo con Laplaciano $\Delta$, la condición CFL es:
    $$
    \Delta t_{\max} = \frac{2}{c_{\text{eff}} \sqrt{\lambda_{\max}(\Delta_{\text{sym}})}}
    $$

    donde $\Delta_{\text{sym}} = \frac{1}{2}(\Delta + \Delta^T)$ es la parte simétrica
    (importante cuando $\Delta$ proviene de un sistema no conservativo).
    """

    def __init__(
        self,
        safety_margin: float = 0.95,
        lanczos_tol: float = 1e-6,
        fallback_max_iter: int = 1000,
    ):
        if not 0 < safety_margin <= 1.0:
            raise CFLViolationError(
                f"safety_margin debe estar en (0, 1]; got {safety_margin}"
            )
        self._margin = safety_margin
        self._lanczos_tol = lanczos_tol
        self._fallback_iter = fallback_max_iter

    def _symmetrize_laplacian(self, L: csr_matrix) -> csr_matrix:
        r"""Devuelve $\Delta_{\text{sym}} = \frac{1}{2}(\Delta + \Delta^T)$."""
        if not issparse(L):
            raise CFLViolationError(f"L debe ser sparse; got type={type(L)}")
        L_sym = 0.5 * (L + L.T)
        return L_sym.tocsr()

    def _estimate_spectral_radius(
        self, L_sym: csr_matrix
    ) -> Tuple[float, str]:
        r"""
        Estima $\lambda_{\max}(\Delta_{\text{sym}})$ usando Lanczos con fallback
        a potencia iterada y finalmente cota de Gerschgorin.

        Returns:
            Tupla (lambda_max, método_usado).
        """
        n = L_sym.shape[0]

        # Método 1: Lanczos con shift-invert
        try:
            k = min(3, n - 1) if n > 2 else 1
            evals, _ = eigsh(L_sym, k=k, which='LM', tol=self._lanczos_tol)
            return float(np.max(np.abs(evals))), "lanczos"
        except Exception as e:
            logger.debug(f"Lanczos falló: {e}")

        # Método 2: Potencia iterada
        try:
            rng = np.random.default_rng(0)
            v = rng.standard_normal(n)
            v /= np.linalg.norm(v)
            lambda_est = 0.0
            for _ in range(self._fallback_iter):
                v_new = L_sym @ v
                norm = np.linalg.norm(v_new)
                if norm < 1e-15:
                    break
                v = v_new / norm
                lambda_est = float(v @ L_sym @ v)
            return lambda_est, "power_iteration"
        except Exception as e:
            logger.debug(f"Power iteration falló: {e}")

        # Método 3: Cota de Gerschgorin
        diag = np.asarray(L_sym.diagonal()).ravel()
        abs_diag = np.abs(diag)
        # Para Laplacianos: cada fila i tiene suma <= 0; cota = 2 * max(|diag|)
        lambda_max = 2.0 * float(np.max(abs_diag)) if abs_diag.size > 0 else 1.0
        return lambda_max, "gerschgorin"

    def audit_time_step(
        self,
        graph_laplacian: csr_matrix,
        wave_speed_c: float,
        requested_dt: float,
    ) -> float:
        r"""
        Audita y acota el paso de integración según CFL.

        Returns:
            safe_dt = $\min(dt_{\text{req}}, \text{margin} \cdot dt_{\max})$.
        """
        # Validaciones
        if wave_speed_c < 0:
            raise CFLViolationError(f"c debe ser ≥ 0; got {wave_speed_c}")
        if requested_dt <= 0:
            raise CFLViolationError(f"requested_dt debe ser > 0; got {requested_dt}")

        # Caso degenerado: c = 0 o grafo trivial
        if wave_speed_c < 1e-15:
            logger.debug("c ≈ 0; CFL no aplica, retornando dt solicitado")
            return requested_dt

        if graph_laplacian.shape[0] <= 1:
            return requested_dt

        # 1. Simetrizar el Laplaciano
        L_sym = self._symmetrize_laplacian(graph_laplacian)

        # 2. Estimar lambda_max
        lambda_max, method = self._estimate_spectral_radius(L_sym)

        if lambda_max < 1e-12:
            logger.debug("λ_max ≈ 0; CFL no aplica")
            return requested_dt

        # 3. Límite CFL estricto
        dt_max_stable = 2.0 / (wave_speed_c * math.sqrt(lambda_max))

        # 4. Aplicar margen de seguridad
        safe_dt = min(requested_dt, dt_max_stable * self._margin)

        if requested_dt > dt_max_stable:
            logger.warning(
                f"Veto CFL: dt_req={requested_dt:.4e} > dt_max={dt_max_stable:.4e} "
                f"(λ_max={lambda_max:.4e} via {method}). "
                f"Integración estrangulada a {safe_dt:.4e}."
            )

        return safe_dt

    def cfl_diagnostic(
        self,
        graph_laplacian: csr_matrix,
        wave_speed_c: float,
        requested_dt: float,
    ) -> Dict[str, Any]:
        r"""Reporte diagnóstico completo de la auditoría CFL."""
        L_sym = self._symmetrize_laplacian(graph_laplacian)
        lambda_max, method = self._estimate_spectral_radius(L_sym)

        if wave_speed_c < 1e-15 or lambda_max < 1e-12:
            return {
                "lambda_max": lambda_max,
                "estimation_method": method,
                "dt_max_stable": float('inf'),
                "cfl_number": 0.0,
                "safe_dt": requested_dt,
                "margin_used": self._margin,
                "violated": False,
            }

        dt_max_stable = 2.0 / (wave_speed_c * math.sqrt(lambda_max))
        cfl_number = wave_speed_c * math.sqrt(lambda_max) * requested_dt / 2.0
        safe_dt = min(requested_dt, dt_max_stable * self._margin)
        violated = requested_dt > dt_max_stable

        return {
            "lambda_max": lambda_max,
            "estimation_method": method,
            "dt_max_stable": dt_max_stable,
            "cfl_number": cfl_number,
            "safe_dt": safe_dt,
            "margin_used": self._margin,
            "violated": violated,
        }


# ╔═════════════════════════════════════════════════════════════════════════════╗
# ║  ORQUESTADOR: DIRAC INTERCONNECTION AGENT (Morfismo Inter-Estrato)       ║
# ╚═════════════════════════════════════════════════════════════════════════════╝
class DiracInterconnectionAgent(Morphism):
    r"""
    El "Demonio de Maxwell" entre TACTICS y PHYSICS.

    Encadena:
        Phase1.compute_control_law() → ControlSolution
        Phase1.compute_effective_load_impedance() → Z_eff
        Phase2.tune_dielectric_tensors() → ImpedanceTensor
        Phase2.compute_effective_wave_speed() → c_eff
        Phase3.audit_time_step() → dt_safe
        → InterconnectionState
    """

    def __init__(
        self,
        metric_tensor: Optional[NDArray[np.float64]] = None,
        tolerance: float = 1e-9,
        safety_margin: float = 0.95,
        base_permeability: float = 1.0,
    ):
        self._G = metric_tensor if metric_tensor is not None else G_PHYSICS
        self._solver = Phase1_IDAPBC_Solver(tolerance=tolerance)
        self._tuner = Phase2_ImpedanceTuner(base_permeability=base_permeability)
        self._governor = Phase3_CFLGovernor(safety_margin=safety_margin)

        # Estado interno para diagnóstico
        self._last_control: Optional[ControlSolution] = None
        self._last_impedance: Optional[ImpedanceTensor] = None
        self._last_c_eff: Optional[float] = None
        self._last_safe_dt: Optional[float] = None

    def synthesize_physical_control(
        self,
        # Planta (flux_condenser)
        J_current: NDArray[np.float64],
        R_current: NDArray[np.float64],
        grad_H: NDArray[np.float64],
        g_port: NDArray[np.float64],
        graph_laplacian: csr_matrix,
        # Estrategia (apu_agent)
        J_desired: NDArray[np.float64],
        R_desired: NDArray[np.float64],
        grad_H_desired: NDArray[np.float64],
        requested_dt: float,
    ) -> InterconnectionState:
        r"""
        Compila la directriz estratégica en un estado físico ejecutable.
        """
        logger.info("Dirac Interconnection Agent: IDA-PBC iniciado")

        # ── FASE 1: IDA-PBC ────────────────────────────────────────────────
        control_sol = self._solver.compute_control_law(
            J_current, R_current, grad_H,
            J_desired, R_desired, grad_H_desired,
            g_port,
        )
        self._last_control = control_sol
        logger.debug(
            f"IDA-PBC: ‖residual‖={control_sol.residual_norm:.2e}, "
            f"rank(g)={control_sol.g_rank}, Ḣ_d={control_sol.H_dot:.2e}"
        )

        # Impedancia efectiva (último método Fase 1 → Fase 2)
        Z_eff = self._solver.compute_effective_load_impedance(control_sol)

        # ── FASE 2: Sintonización PML ──────────────────────────────────────
        impedance_tensor = self._tuner.tune_dielectric_tensors(Z_eff)
        self._last_impedance = impedance_tensor
        kk_check = impedance_tensor.verify_kramers_kronig()
        logger.debug(
            f"PML: ‖Γ‖_F={impedance_tensor.reflection_coefficient_norm:.2e}, "
            f"ε_min={kk_check['epsilon_min_eig']:.2e}"
        )

        # Velocidad efectiva (último método Fase 2 → Fase 3)
        c_eff = self._tuner.compute_effective_wave_speed(impedance_tensor)
        self._last_c_eff = c_eff
        logger.debug(f"c_eff = {c_eff:.4e}")

        # ── FASE 3: CFL ───────────────────────────────────────────────────
        safe_dt = self._governor.audit_time_step(
            graph_laplacian, c_eff, requested_dt
        )
        self._last_safe_dt = safe_dt
        logger.debug(f"dt_safe = {safe_dt:.4e} (req={requested_dt:.4e})")

        logger.info(
            f"Compilación física exitosa: Ḣ={control_sol.H_dot:.2e}, "
            f"Γ={impedance_tensor.reflection_coefficient_norm:.2e}, "
            f"c_eff={c_eff:.2e}, dt={safe_dt:.2e}"
        )

        return InterconnectionState(
            control_law_alpha=control_sol.alpha,
            impedance=impedance_tensor,
            safe_dt=safe_dt,
            lyapunov_derivative=control_sol.H_dot,
            c_eff=c_eff,
            cfl_margin=self._governor._margin,
        )

    def diagnostic_report(self) -> Dict[str, Any]:
        r"""Genera reporte completo del último ciclo ejecutado."""
        report: Dict[str, Any] = {
            "agent_initialized": True,
        }

        if self._last_control is not None:
            cs = self._last_control
            report["phase1_ida_pbc"] = {
                "residual_norm": cs.residual_norm,
                "g_rank": cs.g_rank,
                "H_dot": cs.H_dot,
                "lyapunov_verified": cs.lyapunov_verified,
                "alpha_norm": float(la.norm(cs.alpha)),
            }

        if self._last_impedance is not None:
            imp = self._last_impedance
            report["phase2_pml"] = {
                "reflection_norm": imp.reflection_coefficient_norm,
                "kramers_kronig": imp.verify_kramers_kronig(),
                "wave_speeds": imp.wave_speeds.tolist(),
                "epsilon_min_diag": float(np.min(np.diag(imp.epsilon_eff))),
                "mu_min_diag": float(np.min(np.diag(imp.mu_eff))),
            }

        if self._last_c_eff is not None:
            report["phase3_cfl"] = {
                "c_eff": self._last_c_eff,
                "safe_dt": self._last_safe_dt,
                "margin": self._governor._margin,
            }

        return report


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    # Excepciones
    "DiracMatchingError",
    "ImpedanceMismatchError",
    "CFLViolationError",
    "LyapunovInstabilityError",
    # Estructuras
    "ImpedanceTensor",
    "ControlSolution",
    "InterconnectionState",
    # Fases
    "Phase1_IDAPBC_Solver",
    "Phase2_ImpedanceTuner",
    "Phase3_CFLGovernor",
    # Orquestador
    "DiracInterconnectionAgent",
]