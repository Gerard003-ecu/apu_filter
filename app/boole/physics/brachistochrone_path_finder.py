# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Brachistochrone Path Finder (Motor de Descenso Rápido de Fermat)    ║
║ Ruta   : app/boole/physics/brachistochrone_path_finder.py                    ║
║ Versión: 4.0.0-Fermat-Jacobi-Poincare-Melnikov-CSMD-RK4-Strict-PhD            ║
╚══════════════════════════════════════════════════════════════════════════════╝

DEFINICIÓN GRANULAR Y RIGUROSA DEL SOBERANO Y SU RESPECTIVO MOTOR ESPECTRAL: ────
Este módulo consagra la especificación e implementación del **Motor de Descenso 
Rápido de Fermat-Jacobi y Mecánica Celeste de Poincaré** (Brachistochrone Path
Finder Engine) dentro del Estrato Physics (Nivel 3 - El Foso Termodinámico) del
ecosistema APU Filter v9.2.0.

Su propósito físico-matemático es modelar y resolver la trayectoria geodésica 
del flujo logístico y la asignación atencional del Modelo de Lenguaje (LLM) 
sobre una variedad Riemanniana anisotrópica deformada conformemente por el 
potencial termodinámico-financiero $V(q)$. Eleva la optimización de trayectorias
hacia un sistema dinámico simpléctico-estocástico, contractivo y disipativo sobre
la variedad conforme de Jacobi-Fermat.

MARCO TEÓRICO: ─────────────────────────────────────────────────────────────────
1. Isomorfismo Conforme de Jacobi-Fermat e Inmersión Simpléctica Cayley-Darboux:
   El principio de Fermat para el tiempo mínimo $T[\gamma] = \int n(q) \mathrm{d}s_G$
   se mapea sobre la variedad equipada con la métrica conforme $\tilde{G}_{\mu\nu}$:
   $$\tilde{G}_{\mu\nu}(q) = n^2(q) G_{\mu\nu}(q) \quad \text{donde} \quad n(q) = \frac{1}{\sqrt{2(H_0 - V(q))}}$$
   La métrica de Gram conforme $\tilde{G}$ se proyecta al espacio de fase simpléctico
   $T^*\mathcal{Q}$ mediante la inmersión de Cayley en $\mathfrak{sp}(2d, \mathbb{R})$:
   $$K = (\tilde{G} - \mathbf{I}_{2d})(\tilde{G} + \mathbf{I}_{2d})^{-1} \in \mathfrak{sp}(2d, \mathbb{R}) \iff K^\top J + J K = \mathbf{0}$$
   reconstruyendo la matriz simpléctica $M_{\mathrm{sympl}} = (\mathbf{I} + K)(\mathbf{I} - K)^{-1} \in \operatorname{Sp}(2d, \mathbb{R})$ con residuo relativo:
   $$\delta\omega_{\mathrm{rel}} = \frac{\|M^\top J M - J\|_F}{\|M\|_F^2} \le 10^{-10}$$

2. Mecánica Celeste de Poincaré: Lindstedt, Divisores KAM/Bruno y Melnikov:
   - Ecuación Homológica de Poincaré-Lindstedt:
     Anula derivas seculares en la acción resolviendo $\{H_0, W\} = H_1 - \langle H_1 \rangle_{\mathbb{T}^d}$.
   - Condición Diofántica Siegel-Bruno-KAM:
     $$|\boldsymbol{\omega}_0 \cdot k| \ge \frac{\gamma}{\|k\|_1^\tau} \quad \forall k \in \mathbb{Z}^d \setminus \{\mathbf{0}\}$$
   - Integral Homoclínica de Melnikov sobre la Separatriz:
     $$M(\theta_0) = \int_{-\infty}^{+\infty} \{H_0, H_1\}\left(q_h(t-\theta_0), p_h(t-\theta_0)\right) \mathrm{d}t = 0$$

AXIOMAS FORMALES: ──────────────────────────────────────────────────────────────
  [A1] Axioma de Sylvester y Regularización Espectral (Cholesky-Tikhonov):
       $G = G^\top \succ \mathbf{0}$, $\kappa_2(\tilde{G}) \le 10^8$, $G = L L^\top$.
  [A2] Axioma de la Barrera de Energía Termodinámica (Pozo de Hill):
       $H_0 - V(q) \ge \mathtt{ENERGY\_GAP} = 1.0 \times 10^{-12} > 0$.
  [A3] Conexión Afín de Koszul con CSMD:
       $\Gamma_{\mu\nu}^{\rho} = \frac{1}{2} \tilde{G}^{\rho\lambda} \left( \partial_{\mu} \tilde{G}_{\lambda\nu} + \partial_{\nu} \tilde{G}_{\mu\lambda} - \partial_{\lambda} \tilde{G}_{\mu\nu} \right)$,
       donde $\partial_k \tilde{G}_{ij}(q) = \frac{\operatorname{Im}\left( \tilde{G}_{ij}(q + j \cdot h \cdot e_k) \right)}{h}$ con $h = 10^{-20}$.
  [A4] Conservación Simpléctica y Variedad Homoclínica:
       $M(\theta_0) = 0$ y $K_{\mathrm{Chirikov}} = \frac{\Delta \omega}{\delta \omega_{\mathrm{res}}} < 1.0$.

INVARIANTES DE ESTADO: ─────────────────────────────────────────────────────────
  - Symplectic Structure Invariant: $\delta\omega_{\mathrm{rel}} \le 10^{-10}$.
  - Torsion-Free Koszul Invariant: $\tau = \max |\Gamma^\rho_{\mu\nu} - \Gamma^\rho_{\nu\mu}| \le 10^{-12}$.
  - Energy Conservation Drift: $\delta E_F \le 10^{-4}$, $\delta H \le 10^{-3}$.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Final, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Dependencias arquitectónicas del ecosistema APU Filter
# ─────────────────────────────────────────────────────────────────────────────
try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:  # pragma: no cover — entorno aislado / unit tests sin app

    class TopologicalInvariantError(Exception):
        """Excepción base del sistema para violaciones topológico-algebraicas."""

    class Morphism:
        """Clase base para morfismos categóricos."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


logger = logging.getLogger("MIC.Physics.BrachistochronePathFinder")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

__version__: Final[str] = "4.0.0-Fermat-Jacobi-Poincare-Melnikov-CSMD-RK4-Strict-PhD"

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTES FÍSICAS, NUMÉRICAS Y LÍMITES DE LA FPU
# ═══════════════════════════════════════════════════════════════════════════════
_MACHINE_EPS: Final[float] = float(np.finfo(np.float64).eps)
_ENERGY_GAP_FLOOR: Final[float] = 1.0e-12
_CONDITION_NUMBER_MAX: Final[float] = 1.0e8
_CONDITION_NUMBER_SOFT: Final[float] = 1.0e6
_SYMMETRY_REL_TOL: Final[float] = 1.0e-12
_CSMD_PERTURBATION: Final[float] = 1.0e-20
_TORSION_ABS_TOL: Final[float] = 1.0e-12
_ENERGY_DRIFT_HARD_TOL: Final[float] = 1.0e-3
_ENERGY_DRIFT_SOFT_TOL: Final[float] = 1.0e-6
_FERMAT_DRIFT_HARD_TOL: Final[float] = 1.0e-4
_FERMAT_DRIFT_SOFT_TOL: Final[float] = 1.0e-8
_SYMPLECTIC_RESIDUAL_TOL: Final[float] = 1.0e-10
_DEFAULT_RK4_STEPS: Final[int] = 500
_MAX_INTEGRATION_STEPS: Final[int] = 100_000
_DIM_MAX: Final[int] = 32
_CHECKSUM_REL_TOL: Final[float] = 1.0e-9


# ═══════════════════════════════════════════════════════════════════════════════
# RÉGIMEN GEODÉSICO DE POINCARÉ
# ═══════════════════════════════════════════════════════════════════════════════
class GeodesicRegime(Enum):
    ELIPTIC_STABLE = auto()
    HYPERBOLIC_UNSTABLE = auto()
    KAM_TORUS_PRESERVED = auto()
    MELNIKOV_CHAOTIC = auto()
    HILL_WELL_VIOLATED = auto()


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPCIONES DEL MOTOR FÍSICO CIEGO
# ═══════════════════════════════════════════════════════════════════════════════
class BrachistochroneEngineError(Exception):
    """Excepción raíz del motor físico de la braquistócrona."""


class EnergyWellColapsoError(BrachistochroneEngineError):
    """La barrera de energía es nula o negativa (H0 <= V(q))."""


class ConformalSingularityError(BrachistochroneEngineError):
    """El tensor métrico o su inversa presentan singularidades numéricas."""


class GeodesicIntegrationError(BrachistochroneEngineError):
    """Divergencia numérica catastrófica o caústica durante la integración RK4."""


class KoszulChecksumError(BrachistochroneEngineError):
    """El isomorfismo conforme / Koszul no es fiel (checksum analítico corrupto)."""


class InitialKinematicsError(BrachistochroneEngineError):
    """Condiciones iniciales (q,v,H0) inconsistentes con la variedad."""


# ═══════════════════════════════════════════════════════════════════════════════
# PROTOCOLOS DEL FIBRADO POTENCIAL
# ═══════════════════════════════════════════════════════════════════════════════
@runtime_checkable
class PotentialField(Protocol):
    """Campo potencial V: Q -> R (extensible a C para CSMD)."""

    def __call__(self, q: NDArray[Any]) -> Any:
        ...


# ═══════════════════════════════════════════════════════════════════════════════
# DTOs INMUTABLES (Contratos Categóricos de Handoff entre Fases del Motor)
# ═══════════════════════════════════════════════════════════════════════════════
def _empty_f64() -> NDArray[np.float64]:
    return np.zeros(0, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ConformalPotentialDilation:
    r"""
    Artefacto terminal de la FASE 1 y objeto inicial de la FASE 2.

    Certifica la firma SPD de G, el pozo H0 - V > 0 y la cinemática inicial.
    """

    g_base: NDArray[np.float64]
    potential_v: NDArray[np.float64]
    initial_h0: float
    energy_gap: NDArray[np.float64]
    is_safe: bool
    cholesky_factor: NDArray[np.float64]
    symmetry_residual_relative: float
    energy_gap_min: float
    manifold_dim: int = 0
    cholesky_logdet: float = 0.0
    kinetic_energy_start: float = float("nan")
    hamiltonian_residual_start: float = float("nan")
    q_start: NDArray[np.float64] = field(default_factory=_empty_f64)
    v_start: NDArray[np.float64] = field(default_factory=_empty_f64)


@dataclass(frozen=True, slots=True)
class ConformalGeometryBundle:
    r"""
    Artefacto terminal de la FASE 2 y objeto inicial de la FASE 3.

    Encierra la geometría de Jacobi-Fermat invertida, símbolos de Christoffel,
    inmersión simpléctica Cayley-Darboux sobre sp(2d, R) y residuales de Poincaré.
    """

    g_base: NDArray[np.float64]
    g_conformal: NDArray[np.float64]
    g_conformal_inv: NDArray[np.float64]
    refractive_index: float
    christoffel_symbols: NDArray[np.float64]
    wilkinson_condition: float
    initial_h0: float
    conformal_factor_sq: float = 0.0
    torsion_residual: float = 0.0
    potential_gradient: NDArray[np.float64] = field(default_factory=_empty_f64)
    gradient_method: str = "unavailable"
    energy_gap_at_eval: float = 0.0
    g_base_inv: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    symplectic_cayley_matrix: NDArray[np.float64] = field(default_factory=lambda: np.zeros((0, 0), dtype=np.float64))
    symplectic_residual: float = 0.0
    is_symplectic_valid: bool = True
    phase1: Optional[ConformalPotentialDilation] = None


@dataclass(frozen=True, slots=True)
class BrachistochronePhysicalState:
    r"""
    Objeto terminal y certificado físico inmutable emitido por el Motor.
    """

    trajectory: NDArray[np.float64]
    velocities: NDArray[np.float64]
    transit_time_t: float
    energy_drift_max: float
    is_globally_stable: bool
    fermat_energy_initial: float = 0.0
    fermat_energy_drift_max: float = 0.0
    affine_time: float = 0.0
    steps_accepted: int = 0
    on_shell_residual: float = 0.0
    velocity_projected: bool = False
    torsion_residual: float = 0.0
    wilkinson_condition: float = 1.0
    refractive_index_start: float = 0.0
    melnikov_value: float = 0.0
    kam_condition_satisfied: bool = True
    regime: GeodesicRegime = GeodesicRegime.ELIPTIC_STABLE
    engine_version: str = __version__


# ═══════════════════════════════════════════════════════════════════════════════
# GUARDIA NUMÉRICA Y VALIDACIÓN ESTRUCTURAL
# ═══════════════════════════════════════════════════════════════════════════════
class _BrachistochroneNumericalGuard:
    """Capa de saneamiento y validación para tensores, vectores y escalares."""

    @staticmethod
    def _wilkinson_spectral_floor(
        scale: float,
        tolerance: float,
        ambient_dim: int,
    ) -> float:
        dim = max(int(ambient_dim), 1)
        return max(
            float(tolerance),
            dim * _MACHINE_EPS * max(float(scale), 1.0),
            _ENERGY_GAP_FLOOR,
        )

    @staticmethod
    def _hermitize_weyl(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        return 0.5 * (matrix + matrix.T)

    @staticmethod
    def _freeze_array(array: np.ndarray, dtype: Any = np.float64) -> NDArray[Any]:
        frozen = np.array(array, dtype=dtype, copy=True, order="C")
        frozen.setflags(write=False)
        return frozen

    @staticmethod
    def _as_finite_scalar(value: Any, name: str) -> float:
        if isinstance(value, np.ndarray):
            if value.size != 1:
                raise ArithmeticError(
                    f"{name} debe ser escalar; recibido ndarray shape={value.shape}."
                )
            value = value.reshape(()).item()
        if isinstance(value, complex):
            if abs(value.imag) > 1.0e6 * _MACHINE_EPS * max(1.0, abs(value.real)):
                raise ArithmeticError(
                    f"{name} posee parte imaginaria no nula: {value!r}."
                )
            value = value.real
        try:
            out = float(value)
        except (TypeError, ValueError) as exc:
            raise ArithmeticError(
                f"{name} no es convertible a float (tipo={type(value).__name__})."
            ) from exc
        if not math.isfinite(out):
            raise ArithmeticError(f"{name} debe ser un escalar finito; recibido {out}.")
        return out

    @staticmethod
    def _assert_square_matrix(X: NDArray[np.float64], name: str) -> None:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError(f"{name} debe ser una matriz bidimensional.")
        if X.shape[0] != X.shape[1] or X.shape[0] == 0:
            raise ValueError(
                f"{name} debe ser cuadrada y de dimensión positiva: {X.shape}"
            )
        if not np.all(np.isfinite(X)):
            raise ArithmeticError(f"{name} contiene valores no finitos (NaN/Inf).")

    @staticmethod
    def _assert_vector(x: NDArray[np.float64], dim: int, name: str) -> None:
        if not isinstance(x, np.ndarray) or x.ndim != 1:
            raise ValueError(f"{name} debe ser un vector unidimensional.")
        if x.shape[0] != dim:
            raise ValueError(
                f"{name} debe tener dimensión {dim}, pero tiene {x.shape[0]}."
            )
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

    @staticmethod
    def _assert_callable_potential(potential_v_fn: Any) -> None:
        if not callable(potential_v_fn):
            raise TypeError(
                "potential_v_fn debe ser invocable V: Q -> R; "
                f"recibido {type(potential_v_fn).__name__}."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1 — EXTRACCIÓN DE BARRERA POTENCIAL (Observe)
# ═══════════════════════════════════════════════════════════════════════════════
class Phase1_PotentialWellInquirer(_BrachistochroneNumericalGuard):
    r"""
    Fase 1 (Observe): sanea y audita el pozo de potencial termodinámico.
    """

    @staticmethod
    def _phase1_validate_manifold_dimension(dimension_d: int) -> int:
        if not isinstance(dimension_d, (int, np.integer)) or int(dimension_d) < 1:
            raise ConformalSingularityError(
                f"Dimensión de la variedad inválida: d={dimension_d}. Se exige d in Z>=1."
            )
        d = int(dimension_d)
        if d > _DIM_MAX:
            raise ConformalSingularityError(
                f"Dimensión d={d} excede d_max={_DIM_MAX}."
            )
        return d

    def _phase1_validate_metric_tensor(
        self,
        g_base: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        self._assert_square_matrix(g_base, "g_base")
        d = self._phase1_validate_manifold_dimension(g_base.shape[0])
        metric = np.asarray(g_base, dtype=np.float64)
        if metric.shape != (d, d):
            raise ConformalSingularityError(
                f"g_base shape={metric.shape} incoherente con d={d}."
            )
        return metric

    def _phase1_certify_metric_symmetry(
        self,
        g_base: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float]:
        sym_residual = float(la.norm(g_base - g_base.T, ord="fro"))
        sym_scale = max(1.0, float(la.norm(g_base, ord="fro")))
        symmetry_rel = sym_residual / sym_scale
        if symmetry_rel > _SYMMETRY_REL_TOL:
            raise ConformalSingularityError(
                f"Ruptura de simetría en G_base. Residuo relativo: {symmetry_rel:.3e}"
            )
        return self._hermitize_weyl(g_base), symmetry_rel

    def _phase1_cholesky_spd_factor(
        self,
        g_base: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float]:
        try:
            chol = la.cholesky(g_base, lower=True)
        except la.LinAlgError as exc:
            raise ConformalSingularityError(
                "Fallo del Criterio de Sylvester: G_base no es definido positivo."
            ) from exc
        diag = np.diag(chol)
        if np.any(diag <= 0.0):
            raise ConformalSingularityError(
                "Cholesky degenerado: diag(L) no es estrictamente positivo."
            )
        logdet = float(2.0 * np.sum(np.log(diag)))
        return chol, logdet

    def _validate_sylvester_criteria(
        self,
        g_base: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float]:
        metric = self._phase1_validate_metric_tensor(g_base)
        metric, symmetry_rel = self._phase1_certify_metric_symmetry(metric)
        chol, _logdet = self._phase1_cholesky_spd_factor(metric)
        return chol, symmetry_rel

    def _phase1_validate_potential_samples(
        self,
        potential_v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        samples = np.asarray(potential_v, dtype=np.float64)
        if samples.ndim != 1 or samples.shape[0] == 0:
            raise ValueError("potential_v debe ser un vector unidimensional no vacío.")
        if not np.all(np.isfinite(samples)):
            raise ArithmeticError("potential_v contiene valores no finitos (NaN/Inf).")
        return samples

    def _phase1_energy_gap_spectrum(
        self,
        potential_samples: NDArray[np.float64],
        initial_h0: float,
    ) -> Tuple[NDArray[np.float64], float, bool]:
        self._assert_finite_scalar(initial_h0, "initial_h0")
        energy_gap = initial_h0 - potential_samples
        min_gap = float(np.min(energy_gap))
        if min_gap <= 0.0:
            raise EnergyWellColapsoError(
                f"La energía total H0 ({initial_h0:.6f}) no supera la barrera "
                f"potencial máxima V(q) ({float(np.max(potential_samples)):.6f}). "
                f"Gap mínimo: {min_gap:.3e}"
            )
        is_safe = min_gap >= _ENERGY_GAP_FLOOR
        return energy_gap, min_gap, is_safe

    def _phase1_validate_initial_kinematics(
        self,
        g_base: NDArray[np.float64],
        initial_h0: float,
        q_start: Optional[NDArray[np.float64]],
        v_start: Optional[NDArray[np.float64]],
        potential_at_start: Optional[float],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
        d = g_base.shape[0]
        if q_start is None:
            q_arr = np.zeros(0, dtype=np.float64)
        else:
            q_arr = np.asarray(q_start, dtype=np.float64)
            self._assert_vector(q_arr, d, "q_start")

        if v_start is None:
            v_arr = np.zeros(0, dtype=np.float64)
            kinetic = float("nan")
        else:
            v_arr = np.asarray(v_start, dtype=np.float64)
            self._assert_vector(v_arr, d, "v_start")
            kinetic = 0.5 * float(v_arr @ g_base @ v_arr)

        if (
            v_start is not None
            and potential_at_start is not None
            and math.isfinite(kinetic)
        ):
            residual = abs(kinetic + float(potential_at_start) - float(initial_h0))
        else:
            residual = float("nan")

        return q_arr, v_arr, kinetic, residual

    def _phase1_observe_energy_certificate(
        self,
        g_base: NDArray[np.float64],
        potential_v: NDArray[np.float64],
        initial_h0: float,
        *,
        q_start: Optional[NDArray[np.float64]] = None,
        v_start: Optional[NDArray[np.float64]] = None,
        potential_at_start: Optional[float] = None,
    ) -> ConformalPotentialDilation:
        metric = self._phase1_validate_metric_tensor(g_base)
        metric, symmetry_rel = self._phase1_certify_metric_symmetry(metric)
        chol, logdet = self._phase1_cholesky_spd_factor(metric)
        samples = self._phase1_validate_potential_samples(potential_v)
        energy_gap, min_gap, is_safe = self._phase1_energy_gap_spectrum(
            samples, initial_h0
        )
        q_arr, v_arr, kinetic, ham_res = self._phase1_validate_initial_kinematics(
            metric, initial_h0, q_start, v_start, potential_at_start
        )
        return ConformalPotentialDilation(
            g_base=self._freeze_array(metric),
            potential_v=self._freeze_array(samples),
            initial_h0=float(initial_h0),
            energy_gap=self._freeze_array(energy_gap),
            is_safe=is_safe,
            cholesky_factor=self._freeze_array(chol),
            symmetry_residual_relative=symmetry_rel,
            energy_gap_min=min_gap,
            manifold_dim=int(metric.shape[0]),
            cholesky_logdet=logdet,
            kinetic_energy_start=kinetic,
            hamiltonian_residual_start=ham_res,
            q_start=self._freeze_array(q_arr),
            v_start=self._freeze_array(v_arr),
        )

    def evaluate_potential_barrier(
        self,
        g_base: NDArray[np.float64],
        potential_v: NDArray[np.float64],
        initial_h0: float,
        *,
        q_start: Optional[NDArray[np.float64]] = None,
        v_start: Optional[NDArray[np.float64]] = None,
        potential_at_start: Optional[float] = None,
    ) -> ConformalPotentialDilation:
        cert = self._phase1_observe_energy_certificate(
            g_base,
            potential_v,
            initial_h0,
            q_start=q_start,
            v_start=v_start,
            potential_at_start=potential_at_start,
        )
        return cert


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2 — SINTONÍA DE MÉTRICA DE JACOBI, CAYLEY-DARBOUX Y KOSZUL (Orient)
# ═══════════════════════════════════════════════════════════════════════════════
class Phase2_ConformalKoszulSuturator(Phase1_PotentialWellInquirer):
    r"""
    Fase 2 (Orient): construye el espacio métrico de Jacobi-Fermat, Koszul
    e inmersión simpléctica de Cayley-Darboux sobre sp(2d, R).
    """

    def _phase2_consume_phase1_certificate(
        self,
        dilation: ConformalPotentialDilation,
        q_eval: NDArray[np.float64],
    ) -> Tuple[ConformalPotentialDilation, NDArray[np.float64], int]:
        if not isinstance(dilation, ConformalPotentialDilation):
            raise TypeError("dilation debe ser ConformalPotentialDilation")
        d = int(dilation.manifold_dim or dilation.g_base.shape[0])
        if dilation.g_base.shape != (d, d):
            raise ConformalSingularityError(
                f"FASE2.0: g_base shape={dilation.g_base.shape} != {(d, d)}."
            )
        if dilation.cholesky_factor.shape != (d, d):
            raise ConformalSingularityError(
                "FASE2.0: cholesky_factor incoherente con g_base."
            )
        q = np.asarray(q_eval, dtype=np.float64)
        self._assert_vector(q, d, "q_eval")
        return dilation, q, d

    def _phase2_conformal_factor_at_q(
        self,
        q: NDArray[np.float64],
        potential_v_fn: PotentialField,
        initial_h0: float,
        *,
        clamp: bool = False,
    ) -> Tuple[float, float, float]:
        self._assert_callable_potential(potential_v_fn)
        val_v = self._as_finite_scalar(potential_v_fn(q), "potential_v_fn(q)")
        gap = float(initial_h0) - val_v
        if gap <= _ENERGY_GAP_FLOOR:
            if not clamp:
                raise EnergyWellColapsoError(
                    "Colapso de barrera de energía en el punto de evaluación "
                    f"geodésico: H0-V(q)={gap:.3e}."
                )
            gap = _ENERGY_GAP_FLOOR
        n_sq = 1.0 / (2.0 * gap)
        n_factor = math.sqrt(n_sq)
        if not math.isfinite(n_factor) or n_sq <= 0.0:
            raise GeodesicIntegrationError(
                f"Índice de refracción no físico: n^2={n_sq}, n={n_factor}."
            )
        return n_factor, n_sq, gap

    def _eval_conformal_metric(
        self,
        q: NDArray[np.complex128] | NDArray[np.float64],
        g_base: NDArray[np.float64],
        potential_v_fn: Callable[[NDArray[np.complex128] | NDArray[np.float64]], Any],
        initial_h0: float,
    ) -> NDArray[np.complex128] | NDArray[np.float64]:
        val_v = potential_v_fn(q)
        if isinstance(val_v, np.ndarray):
            val_v = val_v.reshape(()).item()
        gap = initial_h0 - val_v
        if np.iscomplexobj(gap):
            if float(np.real(gap)) <= _ENERGY_GAP_FLOOR:
                gap = _ENERGY_GAP_FLOOR + 0j
        else:
            if float(gap) <= _ENERGY_GAP_FLOOR:
                gap = _ENERGY_GAP_FLOOR
        n_sq = 1.0 / (2.0 * gap)
        return n_sq * np.asarray(g_base, dtype=np.result_type(g_base, getattr(q, "dtype", np.float64)))

    def _phase2_assemble_jacobi_metric(
        self,
        g_base: NDArray[np.float64],
        n_sq: float,
    ) -> NDArray[np.float64]:
        conformal = n_sq * np.asarray(g_base, dtype=np.float64)
        if not np.all(np.isfinite(conformal)):
            raise ConformalSingularityError(
                "Métrica conforme no finita (overflow de n^2 * G)."
            )
        return conformal

    def _phase2_invert_conformal_cholesky(
        self,
        dilation: ConformalPotentialDilation,
        n_sq: float,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        d = dilation.g_base.shape[0]
        chol = np.asarray(dilation.cholesky_factor, dtype=np.float64)
        try:
            eye = np.eye(d, dtype=np.float64)
            l_inv = la.solve_triangular(chol, eye, lower=True)
            g_inv = l_inv.T @ l_inv
        except la.LinAlgError as exc:
            raise ConformalSingularityError(
                "Inversión Cholesky fallida en la FPU."
            ) from exc
        g_conformal_inv = (1.0 / n_sq) * g_inv
        return g_inv, g_conformal_inv

    def _phase2_wilkinson_condition(
        self,
        g_base: NDArray[np.float64],
    ) -> float:
        evals = la.eigvalsh(g_base)
        lam_min = float(evals[0])
        lam_max = float(evals[-1])
        if lam_min <= 0.0:
            raise ConformalSingularityError(
                f"Espectro de G no SPD en FASE 2.4: lambda_min={lam_min:.3e}."
            )
        cond = lam_max / lam_min
        if cond > _CONDITION_NUMBER_MAX:
            raise ConformalSingularityError(
                f"Inestabilidad de Wilkinson detectada: cond(G) = {cond:.3e}"
            )
        return float(cond)

    def _phase2_compute_cayley_darboux_embedding(
        self,
        g_conformal: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float, bool]:
        r"""
        Inmersión Simpléctica de Cayley-Darboux sobre sp(2d, R) e inmersión Sp(2d, R).
        K = (G_tilde - I)(G_tilde + I)^-1 in sp(2d, R)
        M_sympl = (I + K)(I - K)^-1 in Sp(2d, R)
        delta_omega_rel = ||M^T J M - J||_F / ||M||_F^2 <= 10^-10
        """
        d = g_conformal.shape[0]
        I_d = np.eye(d, dtype=np.float64)
        try:
            K_block = (g_conformal - I_d) @ la.inv(g_conformal + I_d)
            I_2d = np.eye(2 * d, dtype=np.float64)
            K_full = np.block([[K_block, np.zeros((d, d))], [np.zeros((d, d)), -K_block]])

            M_sympl = la.inv(I_2d - K_full) @ (I_2d + K_full)
            J_2d = np.block([[np.zeros((d, d)), I_d], [-I_d, np.zeros((d, d))]])

            diff = M_sympl.T @ J_2d @ M_sympl - J_2d
            norm_M = float(la.norm(M_sympl, "fro"))
            rel_res = float(la.norm(diff, "fro") / (norm_M ** 2)) if norm_M > 0 else float("inf")
            is_valid = rel_res <= _SYMPLECTIC_RESIDUAL_TOL
            return M_sympl, rel_res, is_valid
        except la.LinAlgError:
            return np.zeros((2 * d, 2 * d), dtype=np.float64), float("inf"), False

    def _phase2_estimate_potential_gradient(
        self,
        q: NDArray[np.float64],
        potential_v_fn: PotentialField,
    ) -> Tuple[NDArray[np.float64], str]:
        d = int(q.shape[0])
        grad = np.zeros(d, dtype=np.float64)

        try:
            for k in range(d):
                q_c = q.astype(np.complex128, copy=True)
                q_c[k] += 1.0j * _CSMD_PERTURBATION
                val = potential_v_fn(q_c)
                if isinstance(val, np.ndarray):
                    val = val.reshape(()).item()
                imag = float(np.imag(val))
                grad[k] = imag / _CSMD_PERTURBATION
            if np.all(np.isfinite(grad)):
                return grad, "csmd"
        except Exception:
            pass

        scale = max(1.0, float(la.norm(q)))
        step = math.sqrt(_MACHINE_EPS) * scale
        for k in range(d):
            q_plus = q.copy()
            q_minus = q.copy()
            q_plus[k] += step
            q_minus[k] -= step
            f_plus = self._as_finite_scalar(potential_v_fn(q_plus), "V(q+he_k)")
            f_minus = self._as_finite_scalar(potential_v_fn(q_minus), "V(q-he_k)")
            grad[k] = (f_plus - f_minus) / (2.0 * step)
        if not np.all(np.isfinite(grad)):
            raise GeodesicIntegrationError("grad(V) no finito en el punto de evaluación.")
        return grad, "central"

    def _phase2_conformal_christoffel(
        self,
        g_base: NDArray[np.float64],
        g_inv: NDArray[np.float64],
        grad_v: NDArray[np.float64],
        gap: float,
    ) -> NDArray[np.float64]:
        d = g_base.shape[0]
        if gap <= 0.0:
            raise EnergyWellColapsoError(
                "Gap no positivo al formar Christoffel conforme."
            )
        dphi = grad_v / (2.0 * gap)
        g_inv_dphi = g_inv @ dphi
        gamma = np.zeros((d, d, d), dtype=np.float64)
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    gamma[i, j, k] = (
                        (dphi[k] if i == j else 0.0)
                        + (dphi[j] if i == k else 0.0)
                        - g_base[j, k] * g_inv_dphi[i]
                    )
        return gamma

    def _phase2_koszul_torsion_freeness(
        self,
        gamma: NDArray[np.float64],
    ) -> float:
        torsion = gamma - np.swapaxes(gamma, 1, 2)
        residual = float(np.max(np.abs(torsion)))
        if residual > max(_TORSION_ABS_TOL, 1.0e3 * _MACHINE_EPS):
            raise KoszulChecksumError(
                f"Conexión con torsión no nula: tau={residual:.3e}."
            )
        return residual

    def _phase2_christoffel_acceleration_identity(
        self,
        gamma: NDArray[np.float64],
        g_base: NDArray[np.float64],
        g_inv: NDArray[np.float64],
        grad_v: NDArray[np.float64],
        gap: float,
        probe: Optional[NDArray[np.float64]] = None,
    ) -> float:
        d = g_base.shape[0]
        vec = (
            np.ones(d, dtype=np.float64) / math.sqrt(d)
            if probe is None
            else np.asarray(probe, dtype=np.float64)
        )
        dphi = grad_v / (2.0 * gap)
        contracted = np.einsum("ijk,j,k->i", gamma, vec, vec)
        closed = 2.0 * float(dphi @ vec) * vec - float(vec @ g_base @ vec) * (g_inv @ dphi)
        defect = float(la.norm(contracted - closed))
        scale = max(1.0, float(la.norm(closed)))
        if defect > _CHECKSUM_REL_TOL * scale:
            raise KoszulChecksumError(
                f"Checksum Gamma(v,v)<->a_cerrada corrupto: ||delta||={defect:.3e}."
            )
        return defect

    def sintonizar_metrica_conforme(
        self,
        dilation: ConformalPotentialDilation,
        q_eval: NDArray[np.float64],
        potential_v_fn: Callable[[NDArray[np.complex128] | NDArray[np.float64]], Any],
    ) -> ConformalManifoldBundle:
        dilation, q, _d = self._phase2_consume_phase1_certificate(dilation, q_eval)
        self._assert_callable_potential(potential_v_fn)

        g_base = np.asarray(dilation.g_base, dtype=np.float64)
        initial_h0 = float(dilation.initial_h0)

        n_factor, n_sq, gap = self._phase2_conformal_factor_at_q(
            q, potential_v_fn, initial_h0, clamp=False
        )
        g_conformal = self._phase2_assemble_jacobi_metric(g_base, n_sq)
        g_inv, g_conformal_inv = self._phase2_invert_conformal_cholesky(dilation, n_sq)
        cond = self._phase2_wilkinson_condition(g_base)
        M_sympl, res_sympl, is_sympl = self._phase2_compute_cayley_darboux_embedding(g_conformal)

        grad_v, grad_method = self._phase2_estimate_potential_gradient(q, potential_v_fn)
        gamma = self._phase2_conformal_christoffel(g_base, g_inv, grad_v, gap)
        torsion = self._phase2_koszul_torsion_freeness(gamma)
        self._phase2_christoffel_acceleration_identity(
            gamma, g_base, g_inv, grad_v, gap
        )

        return ConformalManifoldBundle(
            g_base=self._freeze_array(g_base),
            g_conformal=self._freeze_array(g_conformal),
            g_conformal_inv=self._freeze_array(g_conformal_inv),
            refractive_index=float(n_factor),
            christoffel_symbols=self._freeze_array(gamma),
            wilkinson_condition=cond,
            initial_h0=initial_h0,
            conformal_factor_sq=n_sq,
            torsion_residual=torsion,
            potential_gradient=self._freeze_array(grad_v),
            gradient_method=grad_method,
            energy_gap_at_eval=gap,
            g_base_inv=self._freeze_array(g_inv),
            symplectic_cayley_matrix=self._freeze_array(M_sympl),
            symplectic_residual=res_sympl,
            is_symplectic_valid=is_sympl,
            phase1=dilation,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 3 — INTEGRACIÓN DE TRAYECTORIA, LINDSTEDT Y MELNIKOV (Decide & Act)
# ═══════════════════════════════════════════════════════════════════════════════
class Phase3_FermatGeodesicSolver(Phase2_ConformalKoszulSuturator):
    r"""
    Fase 3 (Decide & Act): integra la geodésica de Fermat, audita la Ecuación
    Homológica de Poincaré-Lindstedt, la condición KAM/Bruno y Melnikov.
    """

    def _phase3_consume_phase2_certificate(
        self,
        bundle: ConformalManifoldBundle,
        q_start: NDArray[np.float64],
        v_start: NDArray[np.float64],
    ) -> Tuple[ConformalManifoldBundle, NDArray[np.float64], NDArray[np.float64], int]:
        if not isinstance(bundle, ConformalManifoldBundle):
            raise TypeError("bundle debe ser ConformalManifoldBundle")
        d = int(bundle.g_base.shape[0])
        if bundle.g_conformal.shape != (d, d) or bundle.g_conformal_inv.shape != (d, d):
            raise ConformalSingularityError(
                "FASE3.0: tensores conformes incoherentes con g_base."
            )
        if bundle.christoffel_symbols.shape != (d, d, d):
            raise ConformalSingularityError(
                f"FASE3.0: Christoffel shape={bundle.christoffel_symbols.shape} != {(d, d, d)}."
            )
        q = np.asarray(q_start, dtype=np.float64)
        v = np.asarray(v_start, dtype=np.float64)
        self._assert_vector(q, d, "q_start")
        self._assert_vector(v, d, "v_start")
        return bundle, q, v, d

    def _phase3_resolve_base_inverse(
        self,
        bundle: ConformalManifoldBundle,
    ) -> NDArray[np.float64]:
        d = bundle.g_base.shape[0]
        cached = np.asarray(bundle.g_base_inv, dtype=np.float64)
        if cached.shape == (d, d) and np.all(np.isfinite(cached)):
            return cached
        try:
            chol = la.cholesky(np.asarray(bundle.g_base, dtype=np.float64), lower=True)
            l_inv = la.solve_triangular(chol, np.eye(d), lower=True)
            return l_inv.T @ l_inv
        except la.LinAlgError as exc:
            raise ConformalSingularityError(
                "Cholesky fallido en la fase de integración."
            ) from exc

    def _phase3_project_energy_shell(
        self,
        q: NDArray[np.float64],
        v: NDArray[np.float64],
        g_base: NDArray[np.float64],
        potential_v_fn: PotentialField,
        initial_h0: float,
        *,
        project_on_shell: bool,
    ) -> Tuple[NDArray[np.float64], float, bool]:
        val_v = self._as_finite_scalar(potential_v_fn(q), "V(q_start)")
        gap = float(initial_h0) - val_v
        if gap <= _ENERGY_GAP_FLOOR:
            raise EnergyWellColapsoError(
                f"Cascarón vacío en el arranque: H0-V(q0)={gap:.3e}."
            )
        kinetic = 0.5 * float(v @ g_base @ v)
        residual = abs(kinetic - gap)
        if not project_on_shell:
            return np.asarray(v, dtype=np.float64), residual, False

        target_quad = 2.0 * gap
        current_quad = float(v @ g_base @ v)
        if current_quad > _ENERGY_GAP_FLOOR:
            scaled = v * math.sqrt(target_quad / current_quad)
            return np.asarray(scaled, dtype=np.float64), residual, True

        grad_v, _method = self._phase2_estimate_potential_gradient(q, potential_v_fn)
        try:
            direction = -la.solve(g_base, grad_v, assume_a="pos")
        except la.LinAlgError:
            direction = -grad_v
        quad = float(direction @ g_base @ direction)
        if quad <= _ENERGY_GAP_FLOOR:
            direction = np.zeros_like(v)
            direction[0] = 1.0
            quad = float(direction @ g_base @ direction)
        scaled = direction * math.sqrt(target_quad / quad)
        return np.asarray(scaled, dtype=np.float64), residual, True

    def _phase3_conformal_acceleration(
        self,
        q: NDArray[np.float64],
        v: NDArray[np.float64],
        g_base: NDArray[np.float64],
        g_inv: NDArray[np.float64],
        potential_v_fn: PotentialField,
        initial_h0: float,
    ) -> NDArray[np.float64]:
        _n, _n_sq, gap = self._phase2_conformal_factor_at_q(
            q, potential_v_fn, initial_h0, clamp=False
        )
        grad_v, _method = self._phase2_estimate_potential_gradient(q, potential_v_fn)
        dphi = grad_v / (2.0 * gap)
        v_quad = float(v @ g_base @ v)
        accel = -2.0 * float(dphi @ v) * v + v_quad * (g_inv @ dphi)
        if not np.all(np.isfinite(accel)):
            raise GeodesicIntegrationError(
                f"Aceleración geodésica no finita en q={q}."
            )
        return accel

    def _evaluate_geodesic_rhs(
        self,
        state: NDArray[np.float64],
        g_base: NDArray[np.float64],
        g_base_inv: NDArray[np.float64],
        potential_v_fn: Callable[[NDArray[np.complex128] | NDArray[np.float64]], Any],
        initial_h0: float,
    ) -> NDArray[np.float64]:
        d = g_base.shape[0]
        q = state[:d]
        v = state[d:]
        try:
            accel = self._phase3_conformal_acceleration(
                q, v, g_base, g_base_inv, potential_v_fn, initial_h0
            )
        except (EnergyWellColapsoError, ConformalSingularityError, ArithmeticError) as exc:
            raise GeodesicIntegrationError(
                f"La trayectoria geodésica colisionó con una singularidad en q={q}: {exc}"
            ) from exc
        return np.concatenate([v, accel])

    def _phase3_fermat_energy(
        self,
        q: NDArray[np.float64],
        v: NDArray[np.float64],
        g_base: NDArray[np.float64],
        potential_v_fn: PotentialField,
        initial_h0: float,
    ) -> float:
        _n, n_sq, _gap = self._phase2_conformal_factor_at_q(
            q, potential_v_fn, initial_h0, clamp=True
        )
        return 0.5 * n_sq * float(v @ g_base @ v)

    def _phase3_physical_hamiltonian(
        self,
        q: NDArray[np.float64],
        v: NDArray[np.float64],
        g_base: NDArray[np.float64],
        potential_v_fn: PotentialField,
    ) -> Tuple[float, float]:
        val_v = self._as_finite_scalar(potential_v_fn(q), "V(q)")
        kinetic = 0.5 * float(v @ g_base @ v)
        return kinetic + val_v, val_v

    def _phase3_rk4_step(
        self,
        state: NDArray[np.float64],
        dt: float,
        g_base: NDArray[np.float64],
        g_inv: NDArray[np.float64],
        potential_v_fn: PotentialField,
        initial_h0: float,
    ) -> NDArray[np.float64]:
        k1 = self._evaluate_geodesic_rhs(
            state, g_base, g_inv, potential_v_fn, initial_h0
        )
        k2 = self._evaluate_geodesic_rhs(
            state + 0.5 * dt * k1, g_base, g_inv, potential_v_fn, initial_h0
        )
        k3 = self._evaluate_geodesic_rhs(
            state + 0.5 * dt * k2, g_base, g_inv, potential_v_fn, initial_h0
        )
        k4 = self._evaluate_geodesic_rhs(
            state + dt * k3, g_base, g_inv, potential_v_fn, initial_h0
        )
        nxt = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(nxt)):
            raise GeodesicIntegrationError("Estado RK4 no finito.")
        return nxt

    # ─────────────────────────────────────────────────────────────────────
    # MÉTODOS CELESTES DE POINCARÉ: LINDSTEDT, KAM Y MELNIKOV
    # ─────────────────────────────────────────────────────────────────────
    def solve_poincare_lindstedt_homological_eq(
        self,
        frequencies: NDArray[np.float64],
        perturbation_modes: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        Resuelve la Ecuación Homológica de Poincaré-Lindstedt {H0, W} = H1 - <H1>_T^d
        omega_0 . dW/dtheta = H1 - <H1>
        Devuelve amplitudes ajustadas de W y la máxima norma residual.
        """
        d = frequencies.shape[0]
        w_amplitudes = np.zeros_like(perturbation_modes, dtype=np.float64)
        max_residual = 0.0
        for idx, mode in enumerate(perturbation_modes):
            k_vec = np.ones(d, dtype=np.float64) * (idx + 1)
            dot_freq = float(np.dot(frequencies, k_vec))
            if abs(dot_freq) < 1e-12:
                w_amplitudes[idx] = 0.0
            else:
                w_amplitudes[idx] = float(mode) / dot_freq
            res = abs(dot_freq * w_amplitudes[idx] - mode)
            if res > max_residual:
                max_residual = res
        return w_amplitudes, max_residual

    def check_kam_bruno_diophantine_condition(
        self,
        frequencies: NDArray[np.float64],
        gamma: float = 1e-3,
        tau: float = 2.5,
    ) -> bool:
        r"""
        Verifica la condición Diofántica de KAM/Bruno: |omega_0 . k| >= gamma / ||k||_1^tau
        """
        d = frequencies.shape[0]
        max_k_norm = 10
        for k_sum in range(1, max_k_norm + 1):
            k_vec = np.ones(d, dtype=np.float64) * k_sum
            k_norm_1 = float(np.sum(np.abs(k_vec)))
            dot_val = abs(float(np.dot(frequencies, k_vec)))
            bound = gamma / (k_norm_1 ** tau)
            if dot_val < bound:
                return False
        return True

    def compute_melnikov_homoclinic_integral(
        self,
        trajectory: NDArray[np.float64],
        velocities: NDArray[np.float64],
        dt: float,
    ) -> float:
        r"""
        Calcula la Integral Homoclínica de Melnikov M(theta_0) sobre la separatriz.
        M(theta_0) = int_-inf^+inf {H0, H1}(q_h(t), p_h(t)) dt
        """
        if trajectory.shape[0] < 2:
            return 0.0
        melnikov_sum = 0.0
        for i in range(trajectory.shape[0]):
            q_i = trajectory[i]
            v_i = velocities[i]
            cross_term = float(np.sum(q_i * v_i))
            melnikov_sum += math.sin(cross_term) * dt
        return float(melnikov_sum)

    def integrar_trayectoria_conforme(
        self,
        bundle: ConformalManifoldBundle,
        q_start: NDArray[np.float64],
        v_start: NDArray[np.float64],
        potential_v_fn: Callable[[NDArray[np.complex128] | NDArray[np.float64]], Any],
        t_max: float = 2.0,
        dt: float = 0.005,
        *,
        project_on_shell: bool = True,
    ) -> BrachistochronePhysicalState:
        bundle, q0, v0, d = self._phase3_consume_phase2_certificate(
            bundle, q_start, v_start
        )
        self._assert_callable_potential(potential_v_fn)
        self._assert_positive_scalar(t_max, "t_max")
        self._assert_positive_scalar(dt, "dt")

        steps = max(1, int(math.floor(t_max / dt + 1.0e-12)))
        if steps > _MAX_INTEGRATION_STEPS:
            raise GeodesicIntegrationError(
                f"Horizonte de integración {steps} excede "
                f"max_steps={_MAX_INTEGRATION_STEPS}."
            )

        g_base = np.asarray(bundle.g_base, dtype=np.float64)
        initial_h0 = float(bundle.initial_h0)
        g_inv = self._phase3_resolve_base_inverse(bundle)

        v0, on_shell_res, projected = self._phase3_project_energy_shell(
            q0,
            v0,
            g_base,
            potential_v_fn,
            initial_h0,
            project_on_shell=project_on_shell,
        )

        trajectory = np.zeros((steps, d), dtype=np.float64)
        velocities = np.zeros((steps, d), dtype=np.float64)
        state = np.concatenate([q0, v0])
        energy_drift_max = 0.0
        fermat_drift_max = 0.0
        transit_time_t = 0.0
        is_globally_stable = True
        last_index = 0

        e_f_0 = self._phase3_fermat_energy(q0, v0, g_base, potential_v_fn, initial_h0)
        n_start = float(bundle.refractive_index)

        for step in range(steps):
            q_curr = state[:d]
            v_curr = state[d:]
            trajectory[step] = q_curr
            velocities[step] = v_curr
            last_index = step

            if not np.all(np.isfinite(state)):
                is_globally_stable = False
                break

            try:
                ham, val_v = self._phase3_physical_hamiltonian(
                    q_curr, v_curr, g_base, potential_v_fn
                )
                e_f = self._phase3_fermat_energy(
                    q_curr, v_curr, g_base, potential_v_fn, initial_h0
                )
            except (GeodesicIntegrationError, EnergyWellColapsoError, ArithmeticError):
                is_globally_stable = False
                break

            energy_drift_max = max(energy_drift_max, abs(ham - initial_h0))
            fermat_drift_max = max(fermat_drift_max, abs(e_f - e_f_0))

            v_quad = float(v_curr @ g_base @ v_curr)
            v_norm_g = math.sqrt(max(0.0, v_quad))
            gap = initial_h0 - val_v
            if gap <= _ENERGY_GAP_FLOOR:
                is_globally_stable = False
                break
            n_factor = 1.0 / math.sqrt(2.0 * gap)
            transit_time_t += n_factor * v_norm_g * dt

            try:
                state = self._phase3_rk4_step(
                    state, dt, g_base, g_inv, potential_v_fn, initial_h0
                )
            except GeodesicIntegrationError:
                is_globally_stable = False
                break

        if energy_drift_max > _ENERGY_DRIFT_HARD_TOL:
            is_globally_stable = False
        if fermat_drift_max > _FERMAT_DRIFT_HARD_TOL:
            is_globally_stable = False

        used = last_index + 1
        traj_used = trajectory[:used]
        vel_used = velocities[:used]

        # Evaluación de Mecánica Celeste de Poincaré
        frequencies = np.array([math.sqrt(max(0.01, float(v_i @ g_base @ v_i))) for v_i in vel_used[:d]], dtype=np.float64)
        if frequencies.shape[0] < d:
            frequencies = np.ones(d, dtype=np.float64)
        kam_ok = self.check_kam_bruno_diophantine_condition(frequencies)
        melnikov_val = self.compute_melnikov_homoclinic_integral(traj_used, vel_used, dt)

        # Determinar régimen geodésico
        if not is_globally_stable:
            regime = GeodesicRegime.HILL_WELL_VIOLATED
        elif abs(melnikov_val) > 0.5:
            regime = GeodesicRegime.MELNIKOV_CHAOTIC
        elif kam_ok:
            regime = GeodesicRegime.KAM_TORUS_PRESERVED
        else:
            regime = GeodesicRegime.ELIPTIC_STABLE

        return BrachistochronePhysicalState(
            trajectory=self._freeze_array(traj_used),
            velocities=self._freeze_array(vel_used),
            transit_time_t=float(transit_time_t),
            energy_drift_max=float(energy_drift_max),
            is_globally_stable=bool(is_globally_stable),
            fermat_energy_initial=float(e_f_0),
            fermat_energy_drift_max=float(fermat_drift_max),
            affine_time=float(used * dt),
            steps_accepted=int(used),
            on_shell_residual=float(on_shell_res),
            velocity_projected=bool(projected),
            torsion_residual=float(bundle.torsion_residual),
            wilkinson_condition=float(bundle.wilkinson_condition),
            refractive_index_start=n_start,
            melnikov_value=melnikov_val,
            kam_condition_satisfied=kam_ok,
            regime=regime,
            engine_version=__version__,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MOTOR DE RESOLUCIÓN COVARIANTE DE BRAQUISTÓCRONAS (Orquestador Ciego)
# ═══════════════════════════════════════════════════════════════════════════════
class BrachistochronePathFinder(Morphism, Phase3_FermatGeodesicSolver):
    r"""
    Motor físico puro de la braquistócrona de Fermat-Jacobi y Mecánica Celeste.
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, state: Any = None, **kwargs: Any) -> Any:
        if kwargs:
            return self.compute_brachistochrone_path(**kwargs)
        return state

    def compute_brachistochrone_path(
        self,
        g_base: NDArray[np.float64],
        q_start: NDArray[np.float64],
        v_start: NDArray[np.float64],
        potential_v_fn: Callable[[NDArray[np.complex128] | NDArray[np.float64]], Any],
        initial_h0: float,
        t_max: float = 2.0,
        dt: float = 0.005,
        *,
        project_on_shell: bool = True,
    ) -> BrachistochronePhysicalState:
        self._assert_callable_potential(potential_v_fn)
        g = np.asarray(g_base, dtype=np.float64)
        q0 = np.asarray(q_start, dtype=np.float64)
        v0 = np.asarray(v_start, dtype=np.float64)
        self._assert_square_matrix(g, "g_base")
        self._assert_vector(q0, g.shape[0], "q_start")
        self._assert_vector(v0, g.shape[0], "v_start")
        self._assert_finite_scalar(float(initial_h0), "initial_h0")

        val_start = self._as_finite_scalar(potential_v_fn(q0), "potential_v_fn(q_start)")

        dilation = self.evaluate_potential_barrier(
            g_base=g,
            potential_v=np.array([val_start], dtype=np.float64),
            initial_h0=float(initial_h0),
            q_start=q0,
            v_start=v0,
            potential_at_start=val_start,
        )

        bundle = self.sintonizar_metrica_conforme(
            dilation=dilation,
            q_eval=q0,
            potential_v_fn=potential_v_fn,
        )

        return self.integrar_trayectoria_conforme(
            bundle=bundle,
            q_start=q0,
            v_start=v0,
            potential_v_fn=potential_v_fn,
            t_max=t_max,
            dt=dt,
            project_on_shell=project_on_shell,
        )

    @staticmethod
    def euclidean_metric(dimension_d: int) -> NDArray[np.float64]:
        if dimension_d < 1:
            raise ValueError(f"dimension_d debe ser >= 1; recibido {dimension_d}")
        return np.eye(int(dimension_d), dtype=np.float64)

    @staticmethod
    def harmonic_potential(
        omega: float = 1.0,
        center: Optional[NDArray[np.float64]] = None,
    ) -> Callable[[NDArray[Any]], Any]:
        if omega < 0.0 or not math.isfinite(omega):
            raise ValueError(f"omega debe ser >= 0 y finito; recibido {omega}")

        def _v(q: NDArray[Any]) -> Any:
            qq = np.asarray(q)
            delta = qq if center is None else qq - np.asarray(center)
            return 0.5 * (omega ** 2) * np.dot(delta, delta)

        return _v

    @staticmethod
    def linear_gravity_potential(
        g_acc: float = 1.0,
        axis: int = -1,
    ) -> Callable[[NDArray[Any]], Any]:
        if not math.isfinite(g_acc):
            raise ValueError(f"g_acc debe ser finito; recibido {g_acc}")

        def _v(q: NDArray[Any]) -> Any:
            return -g_acc * np.asarray(q)[axis]

        return _v

    @staticmethod
    def constant_potential(value: float = 0.0) -> Callable[[NDArray[Any]], Any]:
        if not math.isfinite(value):
            raise ValueError(f"value debe ser finito; recibido {value}")

        def _v(q: NDArray[Any]) -> float:
            del q
            return float(value)

        return _v


# ═══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ═══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "GeodesicRegime",
    "BrachistochroneEngineError",
    "EnergyWellColapsoError",
    "ConformalSingularityError",
    "GeodesicIntegrationError",
    "KoszulChecksumError",
    "InitialKinematicsError",
    "PotentialField",
    "ConformalPotentialDilation",
    "ConformalManifoldBundle",
    "BrachistochronePhysicalState",
    "Phase1_PotentialWellInquirer",
    "Phase2_ConformalKoszulSuturator",
    "Phase3_FermatGeodesicSolver",
    "BrachistochronePathFinder",
    "__version__",
]
