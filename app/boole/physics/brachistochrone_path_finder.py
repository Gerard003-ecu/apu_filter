
# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Brachistochrone Path Finder (Motor de Descenso Rápido de Fermat)    ║
║ Ruta   : app/boole/physics/brachistochrone_path_finder.py                    ║
║ Versión: 2.0.0-Fermat-Jacobi-Koszul-CSMD-RK4-FPU-Strict-PhD                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GEOMETRÍA DE JACOBI-FERMAT (Rigor Doctoral):
────────────────────────────────────────────────────────────────────────────────
Motor de Descenso Rápido de Fermat-Jacobi dentro del Estrato Physics del
ecosistema APU Filter. Resuelve la trayectoria geodésica del flujo logístico
sobre una variedad Riemanniana anisotrópica deformada conformemente por el
potencial termodinámico-financiero.

El tiempo de tránsito mínimo se reduce a geodésicas sobre la métrica de Jacobi:

    G̃_μν = n²(q) G_μν,  n(q) = 1 / sqrt(2(H0 - V(q)))

INVARIANTES:
  [A1] Sylvester/Cholesky: G = Gᵀ ≻ 0, inversión precondicionada.
  [A2] Barrera de energía: H0 - V(q) > 0.
  [A3] Conexión de Koszul con derivadas por paso complejo (CSMD).
  [A4] Ecuación geodésica RK4 con conservación Hamiltoniana auditada.

ARQUITECTURA DE TRES FASES ANIDADAS:
────────────────────────────────────────────────────────────────────────────────
  Fase 1 ──► EXTRACCIÓN DE BARRERA POTENCIAL
  Fase 2 ──► SINTONÍA DE MÉTRICA DE JACOBI Y KOSZUL
  Fase 3 ──► INTEGRACIÓN DE TRAYECTORIA Y ACCIÓN
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Final, Callable
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# Dependencias arquitectónicas del ecosistema APU Filter
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:
    class TopologicalInvariantError(Exception):
        """Excepción base del sistema para violaciones topológico-algebraicas."""
        pass

    class Morphism:
        """Clase base para morfismos categóricos."""
        pass

logger = logging.getLogger("MIC.Physics.BrachistochronePathFinder")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS, NUMÉRICAS Y LÍMITES DE LA FPU
# ═══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_ENERGY_GAP_FLOOR: Final[float] = 1.0e-12       # Piso absoluto para H0 - V(q)
_CONDITION_NUMBER_MAX: Final[float] = 1.0e8      # Umbral espectral duro de Wilkinson
_CONDITION_NUMBER_SOFT: Final[float] = 1.0e6     # Umbral espectral de degradación
_SYMMETRY_REL_TOL: Final[float] = 1.0e-12        # Tolerancia relativa de simetría
_CSMD_PERTURBATION: Final[float] = 1.0e-20       # Paso imaginario para CSMD
_ENERGY_DRIFT_HARD_TOL: Final[float] = 1.0e-3    # Deriva Hamiltoniana dura
_ENERGY_DRIFT_SOFT_TOL: Final[float] = 1.0e-6    # Deriva Hamiltoniana blanda
_DEFAULT_RK4_STEPS: Final[int] = 500


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPCIONES DEL MOTOR FÍSICO CIEGO
# ═══════════════════════════════════════════════════════════════════════════════
class BrachistochroneEngineError(Exception):
    """Excepción raíz del motor físico de la braquistócrona."""
    pass


class EnergyWellColapsoError(BrachistochroneEngineError):
    """La barrera de energía es nula o negativa (H0 <= V(q))."""
    pass


class ConformalSingularityError(BrachistochroneEngineError):
    """El tensor métrico o su inversa presentan singularidades numéricas."""
    pass


class GeodesicIntegrationError(BrachistochroneEngineError):
    """Divergencia numérica catastrófica o caústica durante la integración RK4."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# DTOs INMUTABLES (Contratos Categóricos de Handoff entre Fases del Motor)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True, slots=True)
class ConformalPotentialDilation:
    """Artefacto terminal de la Fase 1. Precondición formal de la Fase 2."""
    g_base: NDArray[np.float64]
    potential_v: NDArray[np.float64]
    initial_h0: float
    energy_gap: NDArray[np.float64]
    is_safe: bool
    cholesky_factor: NDArray[np.float64]
    symmetry_residual_relative: float
    energy_gap_min: float


@dataclass(frozen=True, slots=True)
class ConformalManifoldBundle:
    """Artefacto terminal de la Fase 2. Precondición formal de la Fase 3."""
    g_base: NDArray[np.float64]
    g_conformal: NDArray[np.float64]
    g_conformal_inv: NDArray[np.float64]
    refractive_index: float
    christoffel_symbols: NDArray[np.float64]  # Forma (d, d, d)
    wilkinson_condition: float
    initial_h0: float


@dataclass(frozen=True, slots=True)
class BrachistochronePhysicalState:
    """Objeto terminal y certificado físico inmutable emitido por el Motor."""
    trajectory: NDArray[np.float64]           # Forma (steps, d)
    velocities: NDArray[np.float64]           # Forma (steps, d)
    transit_time_t: float                     # Tiempo de Fermat mínimo
    energy_drift_max: float                   # Máximo desvío del Hamiltoniano basal
    is_globally_stable: bool                  # Estabilidad global de fase


# ═══════════════════════════════════════════════════════════════════════════════
# GUARDIA NUMÉRICA Y VALIDACIÓN ESTRUCTURAL
# ═══════════════════════════════════════════════════════════════════════════════
class _BrachistochroneNumericalGuard:
    """Capa de saneamiento y validación para tensores y escalares."""

    @staticmethod
    def _assert_square_matrix(X: NDArray[np.float64], name: str) -> None:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError(f"{name} debe ser una matriz bidimensional.")
        if X.shape[0] != X.shape[1] or X.shape[0] == 0:
            raise ValueError(f"{name} debe ser cuadrada y de dimensión positiva: {X.shape}")
        if not np.all(np.isfinite(X)):
            raise ArithmeticError(f"{name} contiene valores no finitos (NaN/Inf).")

    @staticmethod
    def _assert_vector(x: NDArray[np.float64], dim: int, name: str) -> None:
        if not isinstance(x, np.ndarray) or x.ndim != 1:
            raise ValueError(f"{name} debe ser un vector unidimensional.")
        if x.shape[0] != dim:
            raise ValueError(f"{name} debe tener dimensión {dim}, pero tiene {x.shape[0]}.")
        if not np.all(np.isfinite(x)):
            raise ArithmeticError(f"{name} contiene valores no finitos (NaN/Inf).")

    @staticmethod
    def _assert_finite_scalar(x: float, name: str) -> None:
        if not np.isfinite(x):
            raise ArithmeticError(f"{name} debe ser un escalar finito.")

    @staticmethod
    def _assert_positive_scalar(x: float, name: str) -> None:
        _BrachistochroneNumericalGuard._assert_finite_scalar(x, name)
        if x <= 0.0:
            raise ValueError(f"{name} debe ser positivo.")


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1 — EXTRACCIÓN DE BARRERA POTENCIAL (Observe)
# ═══════════════════════════════════════════════════════════════════════════════
class Phase1_PotentialWellInquirer(_BrachistochroneNumericalGuard):
    r"""
    Fase 1 (Observe): Sanea y audita el pozo de potencial termodinámico.
    Verifica que G_{\mu\nu} sea SPD mediante Cholesky y que H0 - V(q) > 0.
    """

    def _validate_sylvester_criteria(
        self,
        g_base: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], float]:
        """
        Aplica factorización de Cholesky para certificar la firma SPD de Sylvester.

        Parámetros
        ----------
        g_base : tensor métrico basal (d,d).

        Retorna
        -------
        L : factor de Cholesky triangular inferior.
        symmetry_rel : residuo relativo de simetría.

        Excepciones
        -----------
        ConformalSingularityError : si no es simétrico o no es definido positivo.
        """
        self._assert_square_matrix(g_base, "g_base")

        # Simetría relativa en norma de Frobenius
        sym_residual = float(la.norm(g_base - g_base.T, ord="fro"))
        sym_scale = max(1.0, float(la.norm(g_base, ord="fro")))
        symmetry_rel = sym_residual / sym_scale
        if symmetry_rel > _SYMMETRY_REL_TOL:
            raise ConformalSingularityError(
                f"Ruptura de simetría en G_base. Residuo relativo: {symmetry_rel:.3e}"
            )

        try:
            L = la.cholesky(g_base, lower=True)
        except la.LinAlgError as exc:
            raise ConformalSingularityError(
                "Fallo del Criterio de Sylvester: G_base no es definido positivo."
            ) from exc

        return L, symmetry_rel

    def evaluate_potential_barrier(
        self,
        g_base: NDArray[np.float64],
        potential_v: NDArray[np.float64],
        initial_h0: float
    ) -> ConformalPotentialDilation:
        """
        Audita el gap de energía H0 - V(q) para prevenir singularidades en el divisor.

        Parámetros
        ----------
        g_base : tensor métrico SPD (d,d).
        potential_v : vector de potencial evaluado en los puntos de arranque (k,).
        initial_h0 : energía total inicial.

        Retorna
        -------
        ConformalPotentialDilation : artefacto terminal de Fase 1.
        """
        L, symmetry_rel = self._validate_sylvester_criteria(g_base)
        self._assert_finite_scalar(initial_h0, "initial_h0")

        potential_v = np.asarray(potential_v, dtype=np.float64)
        if potential_v.ndim != 1 or potential_v.shape[0] == 0:
            raise ValueError("potential_v debe ser un vector unidimensional no vacío.")
        if not np.all(np.isfinite(potential_v)):
            raise ArithmeticError("potential_v contiene valores no finitos (NaN/Inf).")

        energy_gap = initial_h0 - potential_v
        min_gap = float(np.min(energy_gap))

        if min_gap <= 0.0:
            raise EnergyWellColapsoError(
                f"La energía total H0 ({initial_h0:.3f}) no supera la barrera potencial "
                f"máxima V(q) ({float(np.max(potential_v)):.3f}). Gap mínimo: {min_gap:.3e}"
            )

        is_safe = min_gap >= _ENERGY_GAP_FLOOR

        return ConformalPotentialDilation(
            g_base=g_base,
            potential_v=potential_v,
            initial_h0=initial_h0,
            energy_gap=energy_gap,
            is_safe=is_safe,
            cholesky_factor=L,
            symmetry_residual_relative=symmetry_rel,
            energy_gap_min=min_gap,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2 — SINTONÍA DE MÉTRICA DE JACOBI Y KOSZUL (Orient)
# ═══════════════════════════════════════════════════════════════════════════════
class Phase2_ConformalKoszulSuturator(Phase1_PotentialWellInquirer):
    r"""
    Fase 2 (Orient): Construye el espacio métrico de Jacobi-Fermat G̃_{μν}
    e invierte el tensor mediante Cholesky para calcular los símbolos de Christoffel
    Γ^ρ_{μν} usando diferenciación por paso complejo (CSMD) con fallback robusto.
    """

    def _eval_conformal_metric(
        self,
        q: NDArray[np.complex128] | NDArray[np.float64],
        g_base: NDArray[np.float64],
        potential_v_fn: Callable[[NDArray[np.complex128] | NDArray[np.float64]], float],
        initial_h0: float,
    ) -> NDArray[np.complex128] | NDArray[np.float64]:
        """
        Evalúa G̃(q) = n²(q) G_base en un punto real o complejo.

        Parámetros
        ----------
        q : coordenada real o compleja.
        g_base : métrica basal SPD.
        potential_v_fn : función potencial V(q).
        initial_h0 : energía total inicial.

        Retorna
        -------
        g_tilde : métrica conforme evaluada, del tipo correspondiente a q.
        """
        val_v = potential_v_fn(q)
        gap = initial_h0 - val_v
        if np.iscomplexobj(gap):
            if np.real(gap) <= _ENERGY_GAP_FLOOR:
                gap = _ENERGY_GAP_FLOOR + 0j
        else:
            if gap <= _ENERGY_GAP_FLOOR:
                gap = _ENERGY_GAP_FLOOR
        n_sq = 1.0 / (2.0 * gap)
        return n_sq * g_base.astype(q.dtype)

    def _compute_koszul_christoffel_tensor(
        self,
        q_coords: NDArray[np.float64],
        g_base: NDArray[np.float64],
        g_base_inv: NDArray[np.float64],
        potential_v_fn: Callable[[NDArray[np.complex128] | NDArray[np.float64]], float],
        initial_h0: float,
    ) -> NDArray[np.float64]:
        r"""
        Calcula los símbolos de Christoffel de segunda especie:

            Γ^ρ_{μν} = ½ G̃^{ρλ} ( ∂_μ G̃_{λν} + ∂_ν G̃_{μλ} - ∂_λ G̃_{μν} )

        usando derivadas por paso complejo (CSMD) con fallback a diferencias
        centrales reales si la función potencial no admite entradas complejas.
        """
        d = g_base.shape[0]
        christoffel = np.zeros((d, d, d), dtype=np.float64)

        # Derivadas parciales ∂_k G̃_{ij}
        d_g_tilde = np.zeros((d, d, d), dtype=np.float64)
        h_csmd = _CSMD_PERTURBATION

        for k in range(d):
            q_pert = q_coords.astype(np.complex128)
            q_pert[k] += 1j * h_csmd
            try:
                g_pert = self._eval_conformal_metric(q_pert, g_base, potential_v_fn, initial_h0)
                d_g_tilde[k] = np.imag(g_pert) / h_csmd
            except (TypeError, ValueError, AttributeError):
                # Fallback robusto a diferencias centrales reales
                h_real = math.sqrt(_MACHINE_EPS) * max(1.0, float(np.linalg.norm(q_coords[k])))
                e_k = np.zeros(d, dtype=np.float64)
                e_k[k] = 1.0
                q_plus = q_coords + h_real * e_k
                q_minus = q_coords - h_real * e_k
                g_plus = self._eval_conformal_metric(q_plus, g_base, potential_v_fn, initial_h0)
                g_minus = self._eval_conformal_metric(q_minus, g_base, potential_v_fn, initial_h0)
                d_g_tilde[k] = (g_plus - g_minus) / (2.0 * h_real)

        # Fórmula de Koszul
        for r in range(d):
            for m in range(d):
                for n in range(d):
                    s = 0.0
                    for l in range(d):
                        s += g_base_inv[r, l] * (
                            d_g_tilde[m, l, n] + d_g_tilde[n, m, l] - d_g_tilde[l, m, n]
                        )
                    christoffel[r, m, n] = 0.5 * s

        return christoffel

    def sintonizar_metrica_conforme(
        self,
        dilation: ConformalPotentialDilation,
        q_eval: NDArray[np.float64],
        potential_v_fn: Callable[[NDArray[np.complex128] | NDArray[np.float64]], float],
    ) -> ConformalManifoldBundle:
        """
        Consume el DTO de Fase 1, resuelve la métrica de Jacobi y precomputa
        el tensor de Christoffel en q_eval.

        Parámetros
        ----------
        dilation : artefacto terminal de Fase 1.
        q_eval : punto de evaluación geodésico (d,).
        potential_v_fn : función potencial V(q).

        Retorna
        -------
        ConformalManifoldBundle : artefacto terminal de Fase 2.
        """
        if not isinstance(dilation, ConformalPotentialDilation):
            raise TypeError("dilation debe ser ConformalPotentialDilation")
        self._assert_vector(q_eval, dilation.g_base.shape[0], "q_eval")

        g_base = dilation.g_base
        d = g_base.shape[0]
        initial_h0 = dilation.initial_h0

        val_v = potential_v_fn(q_eval)
        if not np.isscalar(val_v):
            val_v = np.asarray(val_v).item()
        self._assert_finite_scalar(float(val_v), "potential_v_fn(q_eval)")

        gap = initial_h0 - float(val_v)
        if gap <= _ENERGY_GAP_FLOOR:
            raise EnergyWellColapsoError(
                "Colapso de barrera de energía en la coordenada geodésica de evaluación."
            )

        n_factor = 1.0 / math.sqrt(2.0 * gap)
        n_sq = n_factor * n_factor
        g_conformal = n_sq * g_base

        # Inversión precondicionada mediante el factor de Cholesky de la base
        L = dilation.cholesky_factor
        try:
            L_inv = la.solve_triangular(L, np.eye(d), lower=True)
            g_base_inv = L_inv.T @ L_inv
            g_conformal_inv = (1.0 / n_sq) * g_base_inv
        except la.LinAlgError as exc:
            raise ConformalSingularityError("Inversión Cholesky fallida en la FPU.") from exc

        # Auditoría de Wilkinson sobre el número de condición espectral
        condition_number = float(la.norm(g_conformal, 2) * la.norm(g_conformal_inv, 2))
        if condition_number > _CONDITION_NUMBER_MAX:
            raise ConformalSingularityError(
                f"Inestabilidad de Wilkinson detectada: cond(G̃) = {condition_number:.3e}"
            )

        christoffel = self._compute_koszul_christoffel_tensor(
            q_coords=q_eval,
            g_base=g_base,
            g_base_inv=g_base_inv,
            potential_v_fn=potential_v_fn,
            initial_h0=initial_h0,
        )

        return ConformalManifoldBundle(
            g_base=g_base,
            g_conformal=g_conformal,
            g_conformal_inv=g_conformal_inv,
            refractive_index=n_factor,
            christoffel_symbols=christoffel,
            wilkinson_condition=condition_number,
            initial_h0=initial_h0,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 3 — INTEGRACIÓN DE TRAYECTORIA Y ACCIÓN (Decide & Act)
# ═══════════════════════════════════════════════════════════════════════════════
class Phase3_FermatGeodesicSolver(Phase2_ConformalKoszulSuturator):
    r"""
    Fase 3 (Decide & Act): Integra la ecuación geodésica de Fermat con RK4,
    mide el tiempo de tránsito y audita el residuo de energía en la variedad.
    """

    def _evaluate_geodesic_rhs(
        self,
        state: NDArray[np.float64],
        g_base: NDArray[np.float64],
        g_base_inv: NDArray[np.float64],
        potential_v_fn: Callable[[NDArray[np.complex128] | NDArray[np.float64]], float],
        initial_h0: float,
    ) -> NDArray[np.float64]:
        """
        Calcula el campo vectorial geodésico:

            q̇ = v
            v̇^ρ = -Γ^ρ_{μν} v^μ v^ν

        Parámetros
        ----------
        state : vector de estado concatenado [q, v].
        g_base, g_base_inv : métrica basal y su inversa.
        potential_v_fn : función potencial.
        initial_h0 : energía total.

        Retorna
        -------
        rhs : vector derivada temporal [q̇, v̇].
        """
        d = g_base.shape[0]
        q = state[:d]
        v = state[d:]

        try:
            Gamma = self._compute_koszul_christoffel_tensor(
                q_coords=q,
                g_base=g_base,
                g_base_inv=g_base_inv,
                potential_v_fn=potential_v_fn,
                initial_h0=initial_h0,
            )
        except Exception as exc:
            raise GeodesicIntegrationError(
                f"La trayectoria geodésica colisionó con una singularidad en q={q}: {exc}"
            ) from exc

        v_dot = np.zeros(d, dtype=np.float64)
        for r in range(d):
            s = 0.0
            for m in range(d):
                for n in range(d):
                    s += Gamma[r, m, n] * v[m] * v[n]
            v_dot[r] = -s

        return np.concatenate([v, v_dot])

    def integrar_trayectoria_conforme(
        self,
        bundle: ConformalManifoldBundle,
        q_start: NDArray[np.float64],
        v_start: NDArray[np.float64],
        potential_v_fn: Callable[[NDArray[np.complex128] | NDArray[np.float64]], float],
        t_max: float = 2.0,
        dt: float = 0.005,
    ) -> BrachistochronePhysicalState:
        """
        Integra la geodésica de Fermat mediante Runge-Kutta de cuarto orden (RK4).

        Parámetros
        ----------
        bundle : artefacto terminal de Fase 2.
        q_start, v_start : condiciones iniciales en el espacio de fase.
        potential_v_fn : función potencial V(q).
        t_max : tiempo paramétrico máximo.
        dt : paso de integración.

        Retorna
        -------
        BrachistochronePhysicalState : trayectoria, tiempo de tránsito y drift energético.
        """
        if not isinstance(bundle, ConformalManifoldBundle):
            raise TypeError("bundle debe ser ConformalManifoldBundle")
        self._assert_vector(q_start, bundle.g_base.shape[0], "q_start")
        self._assert_vector(v_start, bundle.g_base.shape[0], "v_start")
        self._assert_positive_scalar(t_max, "t_max")
        self._assert_positive_scalar(dt, "dt")

        g_base = bundle.g_base
        d = g_base.shape[0]
        initial_h0 = bundle.initial_h0

        # Precomputar inversa de la métrica basal una sola vez
        try:
            L = la.cholesky(g_base, lower=True)
            L_inv = la.solve_triangular(L, np.eye(d), lower=True)
            g_base_inv = L_inv.T @ L_inv
        except la.LinAlgError as exc:
            raise ConformalSingularityError("Cholesky fallido en la fase de integración.") from exc

        steps = max(1, int(t_max / dt))
        trajectory = np.zeros((steps, d), dtype=np.float64)
        velocities = np.zeros((steps, d), dtype=np.float64)

        state = np.concatenate([q_start, v_start])
        energy_drift_max = 0.0
        transit_time_t = 0.0
        is_globally_stable = True

        for step in range(steps):
            q_curr = state[:d]
            v_curr = state[d:]
            trajectory[step] = q_curr
            velocities[step] = v_curr

            # Verificación de finitud
            if not np.all(np.isfinite(state)):
                is_globally_stable = False
                break

            # Conservación del Hamiltoniano basal: T + V(q) = H0
            val_v = potential_v_fn(q_curr)
            if not np.isscalar(val_v):
                val_v = np.asarray(val_v).item()
            if not np.isfinite(val_v):
                is_globally_stable = False
                break

            kinetic_t = 0.5 * float(v_curr @ g_base @ v_curr)
            hamiltonian_curr = kinetic_t + float(val_v)
            drift = abs(hamiltonian_curr - initial_h0)
            energy_drift_max = max(energy_drift_max, drift)

            # Tiempo de tránsito físico: dT = n(q) ||v||_G dt
            v_norm_g = math.sqrt(max(0.0, float(v_curr @ g_base @ v_curr)))
            gap = initial_h0 - float(val_v)
            if gap <= _ENERGY_GAP_FLOOR:
                gap = _ENERGY_GAP_FLOOR
            n_factor = 1.0 / math.sqrt(2.0 * gap)
            transit_time_t += n_factor * v_norm_g * dt

            # Paso RK4
            try:
                k1 = self._evaluate_geodesic_rhs(state, g_base, g_base_inv, potential_v_fn, initial_h0)
                k2 = self._evaluate_geodesic_rhs(state + 0.5 * dt * k1, g_base, g_base_inv, potential_v_fn, initial_h0)
                k3 = self._evaluate_geodesic_rhs(state + 0.5 * dt * k2, g_base, g_base_inv, potential_v_fn, initial_h0)
                k4 = self._evaluate_geodesic_rhs(state + dt * k3, g_base, g_base_inv, potential_v_fn, initial_h0)
            except GeodesicIntegrationError:
                is_globally_stable = False
                break

            state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        # Si la deriva energética es excesiva, la trayectoria no es globalmente estable
        if energy_drift_max > _ENERGY_DRIFT_HARD_TOL:
            is_globally_stable = False

        return BrachistochronePhysicalState(
            trajectory=trajectory[: step + 1] if not is_globally_stable else trajectory,
            velocities=velocities[: step + 1] if not is_globally_stable else velocities,
            transit_time_t=transit_time_t,
            energy_drift_max=energy_drift_max,
            is_globally_stable=is_globally_stable,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MOTOR DE RESOLUCIÓN COVARIANTE DE BRAQUISTÓCRONAS (Orquestador Ciego)
# ═══════════════════════════════════════════════════════════════════════════════
class BrachistochronePathFinder(Morphism, Phase3_FermatGeodesicSolver):
    r"""
    Motor físico puro de la braquistócrona de Fermat-Jacobi.
    Resolvedor ciego de-confinado y libre de realimentaciones tácticas exógenas.
    """

    def __init__(self) -> None:
        super().__init__()

    def compute_brachistochrone_path(
        self,
        g_base: NDArray[np.float64],
        q_start: NDArray[np.float64],
        v_start: NDArray[np.float64],
        potential_v_fn: Callable[[NDArray[np.complex128] | NDArray[np.float64]], float],
        initial_h0: float,
        t_max: float = 2.0,
        dt: float = 0.005,
    ) -> BrachistochronePhysicalState:
        """
        Orquesta de forma estrictamente funcional la resolución geodésica.

        Composición funtorial:
            Z_motor = Φ₃ ∘ Φ₂ ∘ Φ₁
        """
        # Fase 1: Observe — Evaluación del pozo de potencial
        val_start = potential_v_fn(q_start)
        if not np.isscalar(val_start):
            val_start = np.asarray(val_start).item()
        self._assert_finite_scalar(float(val_start), "potential_v_fn(q_start)")

        potential_vector = np.array([float(val_start)], dtype=np.float64)
        dilation = self.evaluate_potential_barrier(
            g_base=g_base,
            potential_v=potential_vector,
            initial_h0=initial_h0,
        )

        # Fase 2: Orient — Sintonía métrica conforme y símbolos de Christoffel
        bundle = self.sintonizar_metrica_conforme(
            dilation=dilation,
            q_eval=q_start,
            potential_v_fn=potential_v_fn,
        )

        # Fase 3: Decide & Act — RK4 e integración de acción de Fermat
        path_state = self.integrar_trayectoria_conforme(
            bundle=bundle,
            q_start=q_start,
            v_start=v_start,
            potential_v_fn=potential_v_fn,
            t_max=t_max,
            dt=dt,
        )

        return path_state


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ═══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "BrachistochroneEngineError",
    "EnergyWellColapsoError",
    "ConformalSingularityError",
    "GeodesicIntegrationError",
    "ConformalPotentialDilation",
    "ConformalManifoldBundle",
    "BrachistochronePhysicalState",
    "Phase1_PotentialWellInquirer",
    "Phase2_ConformalKoszulSuturator",
    "Phase3_FermatGeodesicSolver",
    "BrachistochronePathFinder",
]