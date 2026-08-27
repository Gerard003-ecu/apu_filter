# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : Brachistochrone Path Finder (Motor de Descenso Rápido de Fermat)    ║
║ Ruta   : app/physics/brachistochrone_path_finder.py                          ║
║ Versión: 3.1.0-Fermat-Jacobi-Koszul-CSMD-RK4-FPU-Strict-PhD                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

NATURALEZA CIBER-FÍSICA Y GEOMETRÍA DE JACOBI-FERMAT (Rigor Doctoral): ──────────
Este módulo consagra la especificación e implementación del **Motor de Descenso 
Rápido de Fermat-Jacobi** (Brachistochrone Path Finder) dentro del Estrato 
Physics (Nivel 3 - El Foso Termodinámico) del ecosistema APU Filter. 
Su propósito físico-matemático es modelar y resolver la trayectoria geodésica 
del flujo logístico y la asignación atencional del Modelo de Lenguaje (LLM) 
sobre una variedad Riemanniana anisotrópica deformada conformemente por el 
potencial termodinámico-financiero.

El motor elude de raíz las heurísticas euclidianas planas, tratando la evolución 
de los costos y recursos como geodésicas en el espacio de fase simpléctico 
$$\mathcal{M}$$. Mediante el Isomorfismo de Jacobi, el problema de 
tiempo de tránsito mínimo (braquistócrona clásica de Bernoulli) se reduce a encontrar 
las geodésicas sobre la variedad equipada con la métrica conforme de Fermat:

$$\tilde{G}_{\mu\nu} = n^2(q) G_{\mu\nu} \quad \text{donde} \quad n(q) = \frac{1}{\sqrt{2(H_0 - V(q))}} \quad\big[117, 128\big]$$

Donde:
  · $$G_{\mu\nu}$$ es el tensor métrico Riemanniano basal (inercia estructural).
  · $$V(q)$$ es el potencial generalizado de restricciones (costos/BOM).
  · $$H_0$$ es la energía total de de-confinamiento (presupuesto inicial).
  · $$n(q)$$ es el índice de refracción semántica conforme.

AXIOMÁTICA FÍSICA Y GEOMÉTRICA (Invariantes del Lazo): ──────────────────────────

  [A1] Axioma de Sylvester y Regularización Espectral (Cholesky-Tikhonov):
       El tensor métrico basal $$G_{\mu\nu}$$ debe ser estrictamente simétrico 
       y definido positivo (SPD) para inducir un producto de Riesz estable:
       $$G = G^\top \succ \mathbf{0} \quad\big[123, 131\big]$$
       Para blindar la Unit de Punto Flotante (FPU) frente al mal condicionamiento 
       espectral de Wilkinson ($$\kappa_2 \ge 10^8$$), la inversión se opera 
       exclusivamente a través de la factorización de Cholesky precondicionada:
       $$G = L L^\top \quad\big[125, 137\big]$$

  [A2] Axioma de la Barrera de Energía Termodinámica:
       El descenso geodésico solo es físicamente viable si la energía total de 
       referencia $$H_0$$ supera estrictamente el potencial local $$V(q)$$, confinando 
       la trayectoria dentro de la cubeta de pasividad:
       $$H_0 - V(q) \ge \mathtt{ENERGY\_GAP} > 0 \quad\big[119, 131\big]$$
       Cualquier violación rompe el carácter holomorfo de la refracción, detonando 
       síncronamente la interrupción por hardware de-confinada.

  [A3] Conexión Afín de Koszul y Derivación por Paso Complejo (CSMD):
       La aceleración del frente de onda atencional se calcula libre de torsión 
       topológica ($$T(X,Y) = 0$$) evaluando los símbolos de Christoffel de segunda 
       especie asociados a la conexión de Levi-Civita:
       $$\Gamma_{\mu\nu}^{\rho} = \frac{1}{2} \tilde{G}^{\rho\lambda} \left( \partial_{\mu} \tilde{G}_{\lambda\nu} + \partial_{\nu} \tilde{G}_{\mu\lambda} - \partial_{\lambda} \tilde{G}_{\mu\nu} \right) \quad\big[12, 124, 137\big]$$
       Donde las derivadas parciales se extraen con exactitud analítica al nivel 
       de FPU ($$\mathcal{O}(h^2)$$) inyectando una perturbación imaginaria pura infinitesimal:
       $$\partial_k \tilde{G}_{ij}(q) = \frac{\operatorname{Im}\left( \tilde{G}_{ij}(q + j \cdot h \cdot e_k) \right)}{h} \quad\big[124, 137\big]$$

  [A4] Ecuación de Movimiento Geodésico y Conservación Hamiltoniana:
       El momentum covariante $$p_\mu = \tilde{G}_{\mu\nu} \dot{q}^\nu$$ obedece el 
       transporte paralelo de Levi-Civita, satisfaciendo la ecuación geodésica:
       $$\ddot{q}^\rho + \Gamma_{\mu\nu}^{\rho} \dot{q}^\mu \dot{q}^\nu \equiv 0 \quad\big[126, 138\big]$$
       Verificando síncronamente la conservación de la energía mecánica total en 
       cada paso temporal RK4 para anular derivas ficticias:
       $$\left| T(p) + V(q) - H_0 \right| \le \mathtt{\varepsilon_{\mathrm{FPU}}} \quad\big[126, 138\big]$$

ARQUITECTURA DE TRES FASES ANIDADAS (Funtor de Inferencia Inercial): ────────────
El motor se orquesta mediante un acoplamiento monoidal covariante estricto, donde 
el objeto terminal de cada fase constituye la precondición formal de la siguiente:

  Fase 1 ──► EXTRACCIÓN DE BARRERA POTENCIAL (Phase1_PotentialWellInquirer)
             Ingiere el tensor basal, valida la firma SPD de Sylvester mediante 
             Cholesky, y certifica que el gap de energía $$H_0 - V(q)$$ esté a salvo 
             de cancelaciones catastróficas.
             Entrega: ConformalPotentialDilation.

  Fase 2 ──► SINTONÍA DE MÉTRICA DE JACOBI Y KOSZUL (Phase2_ConformalKoszulSuturator)
             Hereda la ConformalPotentialDilation. Computa el tensor de Fermat 
             $$\tilde{G}_{\mu\nu}$$ y su inversa precondicionada, y calcula los 
             símbolos de Christoffel exactos vía CSMD en el punto de fase.
             Entrega: ConformalManifoldBundle.

  Fase 3 ──► INTEGRACIÓN DE TRAYECTORIA Y ACCIÓN (Phase3_FermatGeodesicSolver)
             Hereda la ConformalManifoldBundle. Integra la ecuación geodésica 
             mediante Runge-Kutta de 4to Orden (RK4), evalúa el tiempo de tránsito 
             de Fermat, y certifica el residuo de deriva del Hamiltoniano.
             Entrega: BrachistochronePhysicalState.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Final, Optional, Protocol, Tuple, runtime_checkable

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

__version__: Final[str] = "3.0.0-Fermat-Jacobi-Koszul-ClosedForm-RK4-FPU-Strict-PhD"

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
_DEFAULT_RK4_STEPS: Final[int] = 500
_MAX_INTEGRATION_STEPS: Final[int] = 100_000
_DIM_MAX: Final[int] = 32
_CHECKSUM_REL_TOL: Final[float] = 1.0e-9


# ═══════════════════════════════════════════════════════════════════════════════
# EXCEPCIONES DEL MOTOR FÍSICO CIEGO
# ═══════════════════════════════════════════════════════════════════════════════
class BrachistochroneEngineError(Exception):
    """Excepción raíz del motor físico de la braquistócrona."""


class EnergyWellColapsoError(BrachistochroneEngineError):
    """La barrera de energía es nula o negativa (\(H_0\le V(q)\))."""


class ConformalSingularityError(BrachistochroneEngineError):
    """El tensor métrico o su inversa presentan singularidades numéricas."""


class GeodesicIntegrationError(BrachistochroneEngineError):
    """Divergencia numérica catastrófica o caústica durante la integración RK4."""


class KoszulChecksumError(BrachistochroneEngineError):
    """El isomorfismo conforme / Koszul no es fiel (checksum analítico corrupto)."""


class InitialKinematicsError(BrachistochroneEngineError):
    """Condiciones iniciales \((q,v,H_0)\) inconsistentes con la variedad."""


# ═══════════════════════════════════════════════════════════════════════════════
# PROTOCOLOS DEL FIBRADO POTENCIAL
# ═══════════════════════════════════════════════════════════════════════════════
@runtime_checkable
class PotentialField(Protocol):
    """Campo potencial \(V:\mathcal{Q}\to\mathbb{R}\) (extensible a \(\mathbb{C}\) para CSMD)."""

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

    Certifica la firma SPD de \(G\), el pozo \(H_0-V>0\) y la cinemática
    inicial opcional. ``Phase2_ConformalKoszulSuturator._phase2_consume_phase1_certificate``
    lo ingiere sin re-tipar la métrica ni re-factorizar Cholesky.

    Campos:
        g_base: \(G\in\mathrm{SPD}(d)\) write-protected.
        potential_v: muestras de \(V\) usadas en el saneamiento.
        initial_h0: energía total \(H_0\).
        energy_gap: \(H_0-V\) punto a punto.
        is_safe: \(\min(H_0-V)\ge\varepsilon_{\mathrm{gap}}\).
        cholesky_factor: \(L\) triangular inferior, \(G=LL^\top\).
        symmetry_residual_relative: \(\|G-G^\top\|_F/\|G\|_F\).
        energy_gap_min: \(\min(H_0-V)\).
        manifold_dim: \(d=\dim\mathcal{Q}\).
        cholesky_logdet: \(\log\det G=2\sum_i\log L_{ii}\).
        kinetic_energy_start: \(\tfrac12 v^\top G v\) (NaN si no hubo \(v\)).
        hamiltonian_residual_start: \(|T+V-H_0|\) en el arranque.
        q_start / v_start: cinemática congelada (vacío si no se aportó).
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
class ConformalManifoldBundle:
    r"""
    Artefacto terminal de la FASE 2 y objeto inicial de la FASE 3.

    Encierra la geometría de Jacobi–Fermat ya invertida, el tensor de
    Christoffel en \(q_{\mathrm{eval}}\) y los residuales Koszul.
    ``Phase3_FermatGeodesicSolver._phase3_consume_phase2_certificate``
    lo ingiere sin reconstruir \(\widetilde{G}\).

    Campos:
        g_base / g_conformal / g_conformal_inv: tensores write-protected.
        refractive_index: \(n(q_{\mathrm{eval}})\).
        christoffel_symbols: \(\widetilde{\Gamma}^\rho_{\mu\nu}\) en \(q_{\mathrm{eval}}\),
            shape \((d,d,d)\).
        wilkinson_condition: \(\kappa_2(G)=\kappa_2(\widetilde{G})\).
        initial_h0: \(H_0\).
        conformal_factor_sq: \(n^2(q_{\mathrm{eval}})\).
        torsion_residual: \(\max|\Gamma^\rho_{\mu\nu}-\Gamma^\rho_{\nu\mu}|\).
        potential_gradient: \(\nabla V(q_{\mathrm{eval}})\).
        gradient_method: ``csmd`` | ``central`` | ``unavailable``.
        energy_gap_at_eval: \(H_0-V(q_{\mathrm{eval}})\).
        g_base_inv: \(G^{-1}\) write-protected (reutilizado por el integrador).
        phase1: certificado de FASE 1 anidado (continuidad funtorial).
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
    phase1: Optional[ConformalPotentialDilation] = None


@dataclass(frozen=True, slots=True)
class BrachistochronePhysicalState:
    r"""
    Objeto terminal y certificado físico inmutable emitido por el Motor.

    Campos:
        trajectory / velocities: geodésica muestreada, shape \((N,d)\).
        transit_time_t: tiempo de Fermat \(T=\int n\|dq\|_G\).
        energy_drift_max: \(\max|T+V-H_0|\) (diagnóstico de cascarón).
        is_globally_stable: finitud + deriva Fermat/física bajo umbral duro.
        fermat_energy_initial: \(E_F(\lambda_0)\).
        fermat_energy_drift_max: \(\max|E_F-E_F(\lambda_0)|\) (invariante afín).
        affine_time: horizonte del parámetro afín \(N\cdot\Delta\lambda\).
        steps_accepted: muestras efectivamente integradas.
        on_shell_residual: \(|T- (H_0-V)|\) pre-proyección.
        velocity_projected: si \(v_0\) fue reescalado al cascarón.
        torsion_residual / wilkinson_condition: invariantes heredados de FASE 2.
        refractive_index_start: \(n(q_0)\).
        engine_version: semver del motor.
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
        r"""
        Suelo espectral de Weyl–Wilkinson

        \[
        \varepsilon_W
        =\max\bigl(\varepsilon,\;
        n\cdot\varepsilon_{\mathrm{mach}}\cdot\max(\mathrm{scale},1),\;
        \varepsilon_{\mathrm{gap}}\bigr).
        \]
        """
        dim = max(int(ambient_dim), 1)
        return max(
            float(tolerance),
            dim * _MACHINE_EPS * max(float(scale), 1.0),
            _ENERGY_GAP_FLOOR,
        )

    @staticmethod
    def _hermitize_weyl(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Proyección de Weyl/Cartan \(H\mapsto\tfrac12(H+H^\top)\) sobre \(\mathrm{Sym}_d\)."""
        return 0.5 * (matrix + matrix.T)

    @staticmethod
    def _freeze_array(array: np.ndarray, dtype: Any = np.float64) -> NDArray[Any]:
        """Copia C-contigua write-protected (invariante de certificado)."""
        frozen = np.array(array, dtype=dtype, copy=True, order="C")
        frozen.setflags(write=False)
        return frozen

    @staticmethod
    def _as_finite_scalar(value: Any, name: str) -> float:
        """Coacciona un escalar (o 0-d array) a ``float`` finito."""
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
                "potential_v_fn debe ser invocable V: Q → ℝ; "
                f"recibido {type(potential_v_fn).__name__}."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 1 — EXTRACCIÓN DE BARRERA POTENCIAL (Observe)
# Objetos: G ∈ SPD(d), V(q), H0 − V > 0, factor de Cholesky
# Funtores: tipado C∞, Sylvester, pozo de energía
# Terminal: ConformalPotentialDilation → objeto inicial FASE 2
# ═══════════════════════════════════════════════════════════════════════════════
class Phase1_PotentialWellInquirer(_BrachistochroneNumericalGuard):
    r"""
    Fase 1 (Observe): sanea y audita el pozo de potencial termodinámico.

    Morfismo compuesto:

    \[
    \mathrm{ObserveEnergy}
    =\mathrm{Well}\circ\mathrm{Kinematics}\circ\mathrm{Samples}
    \circ\mathrm{Chol}\circ\mathrm{Sym}\circ\mathrm{Type}\circ\mathrm{Dim}.
    \]

    El certificado ``ConformalPotentialDilation`` es el objeto inicial
    exacto de
    ``Phase2_ConformalKoszulSuturator._phase2_consume_phase1_certificate``.
    """

    # ── FASE 1.1 ──────────────────────────────────────────────────────────
    @staticmethod
    def _phase1_validate_manifold_dimension(dimension_d: int) -> int:
        r"""
        FASE 1.1 — Certificación de \(d=\dim\mathcal{Q}\).

        Exige \(d\in\mathbb{Z}_{\ge 1}\) y \(d\le d_{\max}\) (el tensor de
        Christoffel es \(d^3\); el régimen Weyl-estable del estrato Physics
        se acota a \(d_{\max}=32\)).
        """
        if not isinstance(dimension_d, (int, np.integer)) or int(dimension_d) < 1:
            raise ConformalSingularityError(
                f"Dimensión de la variedad inválida: d={dimension_d}. Se exige d ∈ ℤ≥1."
            )
        d = int(dimension_d)
        if d > _DIM_MAX:
            raise ConformalSingularityError(
                f"Dimensión d={d} excede d_max={_DIM_MAX}. "
                "El fibrado de Christoffel (d³) abandonaría el régimen de "
                "auditoría geodésica estable del estrato Physics."
            )
        return d

    # ── FASE 1.2 ──────────────────────────────────────────────────────────
    def _phase1_validate_metric_tensor(
        self,
        g_base: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        FASE 1.2 — Tipado de \(G\in\mathrm{Mat}_d(\mathbb{R})\), finitud IEEE-754.

        Coacciona a ``float64`` y certifica squareness. No proyecta aún:
        la simetría se audita en 1.3 (un Weyl prematuro ocultaría un bug).
        """
        self._assert_square_matrix(g_base, "g_base")
        d = self._phase1_validate_manifold_dimension(g_base.shape[0])
        metric = np.asarray(g_base, dtype=np.float64)
        if metric.shape != (d, d):
            raise ConformalSingularityError(
                f"g_base shape={metric.shape} incoherente con d={d}."
            )
        logger.debug(
            "FASE1.2 métrica: d=%d, ‖G‖_F=%.6e",
            d,
            float(la.norm(metric, ord="fro")),
        )
        return metric

    # ── FASE 1.3 ──────────────────────────────────────────────────────────
    def _phase1_certify_metric_symmetry(
        self,
        g_base: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        FASE 1.3 — Simetría de \(G\) en norma de Frobenius relativa.

        \[
        \|G-G^\top\|_F
        \le
        \varepsilon_{\mathrm{sym}}\,\|G\|_F.
        \]

        Si el defecto es tolerable se devuelve la proyección de Weyl
        (el residuo *pre-proyección* se reporta).
        """
        sym_residual = float(la.norm(g_base - g_base.T, ord="fro"))
        sym_scale = max(1.0, float(la.norm(g_base, ord="fro")))
        symmetry_rel = sym_residual / sym_scale
        if symmetry_rel > _SYMMETRY_REL_TOL:
            raise ConformalSingularityError(
                f"Ruptura de simetría en G_base. Residuo relativo: {symmetry_rel:.3e}"
            )
        return self._hermitize_weyl(g_base), symmetry_rel

    # ── FASE 1.4 ──────────────────────────────────────────────────────────
    def _phase1_cholesky_spd_factor(
        self,
        g_base: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        FASE 1.4 — Factorización de Cholesky / criterio de Sylvester.

        \[
        G=LL^\top,
        \qquad
        \log\det G=2\sum_i\log L_{ii}.
        \]

        Certifica \(G\succ 0\). El factor \(L\) se reutiliza en FASE 2
        para invertir \(\widetilde{G}=n^2 G\) sin refactorizar.
        """
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
        logger.debug(
            "FASE1.4 Cholesky: logdet(G)=%.6e, min L_ii=%.6e",
            logdet,
            float(np.min(diag)),
        )
        return chol, logdet

    def _validate_sylvester_criteria(
        self,
        g_base: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float]:
        """
        Fachada de compatibilidad: simetría + Cholesky (FASES 1.2–1.4).

        Retorna
        -------
        L : factor de Cholesky triangular inferior.
        symmetry_rel : residuo relativo de simetría.
        """
        metric = self._phase1_validate_metric_tensor(g_base)
        metric, symmetry_rel = self._phase1_certify_metric_symmetry(metric)
        chol, _logdet = self._phase1_cholesky_spd_factor(metric)
        return chol, symmetry_rel

    # ── FASE 1.5 ──────────────────────────────────────────────────────────
    def _phase1_validate_potential_samples(
        self,
        potential_v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        r"""
        FASE 1.5 — Tipado y finitud del vector de muestras de \(V\).

        Se admite un vector unidimensional no vacío (p.ej. \(\{V(q_0)\}\)
        o una malla de sondeo del pozo). El campo vivo llega como
        ``potential_v_fn`` en FASE 2.
        """
        samples = np.asarray(potential_v, dtype=np.float64)
        if samples.ndim != 1 or samples.shape[0] == 0:
            raise ValueError("potential_v debe ser un vector unidimensional no vacío.")
        if not np.all(np.isfinite(samples)):
            raise ArithmeticError("potential_v contiene valores no finitos (NaN/Inf).")
        return samples

    # ── FASE 1.6 ──────────────────────────────────────────────────────────
    def _phase1_energy_gap_spectrum(
        self,
        potential_samples: NDArray[np.float64],
        initial_h0: float,
    ) -> Tuple[NDArray[np.float64], float, bool]:
        r"""
        FASE 1.6 — Espectro del gap \(H_0-V\) y veredicto de pozo.

        \[
        \Delta_i=H_0-V_i,
        \qquad
        \Delta_{\min}=\min_i\Delta_i.
        \]

        \(\Delta_{\min}\le 0\) es una barrera clásica (caústica / turning
        point): el índice de refracción deja de ser real y se eleva veto.
        Un gap positivo pero inferior a \(\varepsilon_{\mathrm{gap}}\) se
        marca ``is_safe=False`` (pozo numéricamente rasante).
        """
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
        logger.debug(
            "FASE1.6 pozo: Δ_min=%.6e, H0=%.6f, safe=%s",
            min_gap,
            initial_h0,
            is_safe,
        )
        return energy_gap, min_gap, is_safe

    # ── FASE 1.7 ──────────────────────────────────────────────────────────
    def _phase1_validate_initial_kinematics(
        self,
        g_base: NDArray[np.float64],
        initial_h0: float,
        q_start: Optional[NDArray[np.float64]],
        v_start: Optional[NDArray[np.float64]],
        potential_at_start: Optional[float],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], float, float]:
        r"""
        FASE 1.7 — Cinemática inicial opcional y residuo hamiltoniano.

        Si se aportan \((q_0,v_0)\) se verifica

        \[
        r_H=\bigl|\tfrac12 v_0^\top G v_0+V(q_0)-H_0\bigr|
        \]

        como invariante blando (la proyección al cascarón se decide en FASE 3).
        """
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

        logger.debug(
            "FASE1.7 cinemática: T0=%.6e, r_H=%.3e, has_q=%s, has_v=%s",
            kinetic,
            residual,
            q_arr.size > 0,
            v_arr.size > 0,
        )
        return q_arr, v_arr, kinetic, residual

    # ── FASE 1.Ω · composición terminal Observe ───────────────────────────
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
        r"""
        FASE 1.Ω — Composición terminal de Observación energética.

        \[
        \mathrm{ObserveEnergy}
        =\mathrm{Kinematics}\circ\mathrm{Well}\circ\mathrm{Samples}
        \circ\mathrm{Chol}\circ\mathrm{Sym}\circ\mathrm{Type}.
        \]

        **Contrato funtorial F1 → F2**: el DTO
        ``ConformalPotentialDilation`` es el objeto inicial exacto de
        ``_phase2_consume_phase1_certificate``. Ningún re-tipado de \(G\)
        ni re-factorización de Cholesky se aplica aguas abajo.
        """
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
        """
        Audita el gap de energía \(H_0-V(q)\) para prevenir singularidades
        en el divisor del índice de refracción.

        Fachada pública de FASE 1.Ω. Véase ``_phase1_observe_energy_certificate``.

        **Este certificado es el objeto inicial de la FASE 2.**
        """
        cert = self._phase1_observe_energy_certificate(
            g_base,
            potential_v,
            initial_h0,
            q_start=q_start,
            v_start=v_start,
            potential_at_start=potential_at_start,
        )
        logger.debug(
            "FASE1.Ω observe: d=%d, Δ_min=%.6e, logdet=%.6e, safe=%s",
            cert.manifold_dim,
            cert.energy_gap_min,
            cert.cholesky_logdet,
            cert.is_safe,
        )
        return cert


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 2 — SINTONÍA DE MÉTRICA DE JACOBI Y KOSZUL (Orient)
# Continuación directa de evaluate_potential_barrier (FASE 1.Ω) vía FASE 2.0
# Objetos: n(q), G̃ = n² G, G̃⁻¹, Γ̃, torsión, κ₂(G)
# Teorías: Jacobi–Fermat, Koszul, Weyl–Wilkinson, CSMD de ∇V
# Terminal: ConformalManifoldBundle → objeto inicial FASE 3
# ═══════════════════════════════════════════════════════════════════════════════
class Phase2_ConformalKoszulSuturator(Phase1_PotentialWellInquirer):
    r"""
    Fase 2 (Orient): construye el espacio métrico de Jacobi–Fermat y Koszul.

    Morfismo compuesto:

    \[
    \mathrm{OrientConformal}
    =(\mathrm{Torsion},\,\widetilde{\Gamma},\,\nabla V,\,\kappa,\,
    \widetilde{G}^{-1},\,\widetilde{G},\,n)
    \circ\mathrm{Consume}\circ\mathrm{ObserveEnergy}^*.
    \]

    El primer morfismo, ``_phase2_consume_phase1_certificate``, *es* la
    continuación estricta de ``evaluate_potential_barrier``.
    """

    # ── FASE 2.0 · ingesta funtorial del certificado de FASE 1.Ω ──────────
    def _phase2_consume_phase1_certificate(
        self,
        dilation: ConformalPotentialDilation,
        q_eval: NDArray[np.float64],
    ) -> Tuple[ConformalPotentialDilation, NDArray[np.float64], int]:
        r"""
        FASE 2.0 — Ingesta funtorial del certificado de FASE 1.Ω.

        **Continuación estricta de**
        ``Phase1_PotentialWellInquirer.evaluate_potential_barrier``.
        Verifica la coherencia \((G,L,d)\) y entrega el objeto de trabajo
        de Orient *sin re-tipado ni re-Cholesky*.

        Raises:
            TypeError, ConformalSingularityError.
        """
        if not isinstance(dilation, ConformalPotentialDilation):
            raise TypeError("dilation debe ser ConformalPotentialDilation")
        d = int(dilation.manifold_dim or dilation.g_base.shape[0])
        if dilation.g_base.shape != (d, d):
            raise ConformalSingularityError(
                f"FASE2.0: g_base shape={dilation.g_base.shape} ≠ {(d, d)}."
            )
        if dilation.cholesky_factor.shape != (d, d):
            raise ConformalSingularityError(
                "FASE2.0: cholesky_factor incoherente con g_base."
            )
        q = np.asarray(q_eval, dtype=np.float64)
        self._assert_vector(q, d, "q_eval")
        logger.debug(
            "FASE2.0 consume F1: d=%d, H0=%.6f, Δ_min=%.6e, safe=%s",
            d,
            dilation.initial_h0,
            dilation.energy_gap_min,
            dilation.is_safe,
        )
        return dilation, q, d

    # ── FASE 2.1 ──────────────────────────────────────────────────────────
    def _phase2_conformal_factor_at_q(
        self,
        q: NDArray[np.float64],
        potential_v_fn: PotentialField,
        initial_h0: float,
        *,
        clamp: bool = False,
    ) -> Tuple[float, float, float]:
        r"""
        FASE 2.1 — Factor conforme / índice de refracción en \(q\).

        \[
        \Delta=H_0-V(q),
        \qquad
        n=\frac{1}{\sqrt{2\Delta}},
        \qquad
        n^2=\frac{1}{2\Delta}.
        \]

        Si ``clamp`` y \(\Delta\le\varepsilon_{\mathrm{gap}}\) se regulariza
        el turning point (uso interno del integrador). En sintonía (F2.Ω)
        ``clamp=False`` y la barrera se eleva como veto.
        """
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
                f"Índice de refracción no físico: n²={n_sq}, n={n_factor}."
            )
        return n_factor, n_sq, gap

    def _eval_conformal_metric(
        self,
        q: NDArray[np.complex128] | NDArray[np.float64],
        g_base: NDArray[np.float64],
        potential_v_fn: Callable[[NDArray[np.complex128] | NDArray[np.float64]], Any],
        initial_h0: float,
    ) -> NDArray[np.complex128] | NDArray[np.float64]:
        r"""
        Fachada de compatibilidad: \(\widetilde{G}(q)=n^2(q)\,G\).

        Conservada para inspecciones / tests que evalúen la métrica en un
        punto real. El camino principal de FASE 2–3 usa la forma cerrada
        (2.6 / 3.2) y **no** diferencia el tensor completo.
        """
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

    # ── FASE 2.2 ──────────────────────────────────────────────────────────
    def _phase2_assemble_jacobi_metric(
        self,
        g_base: NDArray[np.float64],
        n_sq: float,
    ) -> NDArray[np.float64]:
        r"""
        FASE 2.2 — Ensamblaje de la métrica de Jacobi–Fermat.

        \[
        \widetilde{G}=n^2\,G.
        \]
        """
        conformal = n_sq * np.asarray(g_base, dtype=np.float64)
        if not np.all(np.isfinite(conformal)):
            raise ConformalSingularityError(
                "Métrica conforme no finita (overflow de n²·G)."
            )
        return conformal

    # ── FASE 2.3 ──────────────────────────────────────────────────────────
    def _phase2_invert_conformal_cholesky(
        self,
        dilation: ConformalPotentialDilation,
        n_sq: float,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        r"""
        FASE 2.3 — Inversión estable \(\widetilde{G}^{-1}=n^{-2}(LL^\top)^{-1}\).

        Reutiliza el factor de Cholesky de FASE 1.4 (sin refactorizar):

        \[
        G^{-1}=(L^{-1})^\top L^{-1},
        \qquad
        \widetilde{G}^{-1}=n^{-2}G^{-1}.
        \]
        """
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

    # ── FASE 2.4 ──────────────────────────────────────────────────────────
    def _phase2_wilkinson_condition(
        self,
        g_base: NDArray[np.float64],
    ) -> float:
        r"""
        FASE 2.4 — Número de condición de Wilkinson.

        Como \(n^2\) es escalar, \(\kappa_2(\widetilde{G})=\kappa_2(G)\).
        Se evalúa sobre el espectro simétrico de \(G\):

        \[
        \kappa_2(G)=\lambda_{\max}(G)/\lambda_{\min}(G).
        \]

        (Más estable que \(\|G\|_2\|G^{-1}\|_2\) por dos normas-2 sucesivas.)
        """
        evals = la.eigvalsh(g_base)
        lam_min = float(evals[0])
        lam_max = float(evals[-1])
        if lam_min <= 0.0:
            raise ConformalSingularityError(
                f"Espectro de G no SPD en FASE 2.4: λ_min={lam_min:.3e}."
            )
        cond = lam_max / lam_min
        if cond > _CONDITION_NUMBER_MAX:
            raise ConformalSingularityError(
                f"Inestabilidad de Wilkinson detectada: cond(G) = {cond:.3e}"
            )
        logger.debug(
            "FASE2.4 Wilkinson: κ₂=%.6e, λ∈[%.3e, %.3e]",
            cond,
            lam_min,
            lam_max,
        )
        return float(cond)

    # ── FASE 2.5 ──────────────────────────────────────────────────────────
    def _phase2_estimate_potential_gradient(
        self,
        q: NDArray[np.float64],
        potential_v_fn: PotentialField,
    ) -> Tuple[NDArray[np.float64], str]:
        r"""
        FASE 2.5 — Gradiente \(\nabla V(q)\) por CSMD con fallback central.

        * Si \(V\) admite extensión holomorfa, el paso complejo
          \(q_k\leftarrow q_k+ih\) extrae \(\partial_k V=\Im V(q+ihe_k)/h\)
          con error \(\mathcal{O}(h^2)\) sin cancelación sustractiva.
        * Si \(V\) no es holomorfa / no acepta \(\mathbb{C}\), se cae a
          diferencias centrales reales con \(h\sim\sqrt{\varepsilon_{\mathrm{mach}}}\).

        **No** se diferencia el tensor \(\widetilde{G}\) completo: \(G\)
        es constante y \(\partial\widetilde{G}\) se reduce a \(\nabla V\).
        """
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
                logger.debug(
                    "FASE2.5 ∇V: método=csmd, ‖∇V‖=%.6e",
                    float(la.norm(grad)),
                )
                return grad, "csmd"
        except Exception as exc:  # noqa: BLE001 — fallback deliberado a diferencias reales
            logger.debug(
                "FASE2.5 CSMD no aplicable (%s); se usa diferencias centrales.",
                exc,
            )

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
            raise GeodesicIntegrationError("∇V no finito en el punto de evaluación.")
        logger.debug(
            "FASE2.5 ∇V: método=central, ‖∇V‖=%.6e",
            float(la.norm(grad)),
        )
        return grad, "central"

    # ── FASE 2.6 ──────────────────────────────────────────────────────────
    def _phase2_conformal_christoffel(
        self,
        g_base: NDArray[np.float64],
        g_inv: NDArray[np.float64],
        grad_v: NDArray[np.float64],
        gap: float,
    ) -> NDArray[np.float64]:
        r"""
        FASE 2.6 — Christoffel conforme en forma cerrada (\(G\) constante).

        Con \(\phi=\ln n=-\tfrac12\ln\bigl(2(H_0-V)\bigr)\) se tiene
        \(\nabla\phi=\nabla V\big/\,2\Delta\) y

        \[
        \widetilde{\Gamma}^i_{jk}
        =\delta^i_j\partial_k\phi+\delta^i_k\partial_j\phi
         -G_{jk}\,G^{il}\partial_l\phi.
        \]

        Complejidad \(\mathcal{O}(d^3)\) en el tensor (se materializa para
        el certificado). **Corrección respecto de v2**: la fórmula de Koszul
        debe contraer con \(\widetilde{G}^{-1}=n^{-2}G^{-1}\), no con
        \(G^{-1}\); la forma cerrada ya incorpora ese factor a través de
        \(\nabla\phi\).
        """
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

    def _compute_koszul_christoffel_tensor(
        self,
        q_coords: NDArray[np.float64],
        g_base: NDArray[np.float64],
        g_base_inv: NDArray[np.float64],
        potential_v_fn: Callable[[NDArray[np.complex128] | NDArray[np.float64]], Any],
        initial_h0: float,
    ) -> NDArray[np.float64]:
        r"""
        Fachada de compatibilidad: Christoffel conforme en \(q\) (forma cerrada).

        Sustituye el CSMD matricial de la v2, que (i) diferenciaba
        \(\widetilde{G}\) completo en lugar de \(\nabla V\), y (ii) contraía
        Koszul con \(G^{-1}\) en vez de \(\widetilde{G}^{-1}\).
        """
        d = g_base.shape[0]
        self._assert_vector(q_coords, d, "q_coords")
        self._assert_square_matrix(g_base, "g_base")
        _n, _n_sq, gap = self._phase2_conformal_factor_at_q(
            q_coords, potential_v_fn, initial_h0, clamp=True
        )
        grad_v, _method = self._phase2_estimate_potential_gradient(
            q_coords, potential_v_fn
        )
        return self._phase2_conformal_christoffel(g_base, g_base_inv, grad_v, gap)

    # ── FASE 2.7 ──────────────────────────────────────────────────────────
    def _phase2_koszul_torsion_freeness(
        self,
        gamma: NDArray[np.float64],
    ) -> float:
        r"""
        FASE 2.7 — Testigo de torsión nula (Levi-Civita).

        \[
        \tau=\max_{i,j,k}\bigl|\Gamma^i_{jk}-\Gamma^i_{kj}\bigr|.
        \]

        La forma cerrada es idénticamente libre de torsión; un \(\tau\)
        por encima de tolerancia delata corrupción de índices / redondeo
        patológico.
        """
        torsion = gamma - np.swapaxes(gamma, 1, 2)
        residual = float(np.max(np.abs(torsion)))
        if residual > max(_TORSION_ABS_TOL, 1.0e3 * _MACHINE_EPS):
            raise KoszulChecksumError(
                f"Conexión con torsión no nula: τ={residual:.3e}."
            )
        logger.debug("FASE2.7 Koszul: τ=%.3e (libre de torsión)", residual)
        return residual

    # ── FASE 2.8 ──────────────────────────────────────────────────────────
    def _phase2_christoffel_acceleration_identity(
        self,
        gamma: NDArray[np.float64],
        g_base: NDArray[np.float64],
        g_inv: NDArray[np.float64],
        grad_v: NDArray[np.float64],
        gap: float,
        probe: Optional[NDArray[np.float64]] = None,
    ) -> float:
        r"""
        FASE 2.8 — Checksum \(\Gamma(v,v)\) ↔ aceleración contraída \(\mathcal{O}(d^2)\).

        \[
        \bigl(\Gamma(v,v)\bigr)^i
        =2(\nabla\phi\cdot v)\,v^i-\|v\|_G^2\,(G^{-1}\nabla\phi)^i.
        \]
        """
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
                f"Checksum Γ(v,v)↔a_cerrada corrupto: ‖Δ‖={defect:.3e}."
            )
        return defect

    # ── FASE 2.Ω · composición terminal Orient ────────────────────────────
    def sintonizar_metrica_conforme(
        self,
        dilation: ConformalPotentialDilation,
        q_eval: NDArray[np.float64],
        potential_v_fn: Callable[[NDArray[np.complex128] | NDArray[np.float64]], Any],
    ) -> ConformalManifoldBundle:
        r"""
        FASE 2.Ω — Composición terminal Orient (métrica + Koszul + κ).

        **Continuación funtorial de FASE 1.Ω**: consume
        ``ConformalPotentialDilation`` vía FASE 2.0.

        **Contrato funtorial F2 → F3**: el DTO
        ``ConformalManifoldBundle`` es el objeto inicial exacto de
        ``_phase3_consume_phase2_certificate``.
        """
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
        grad_v, grad_method = self._phase2_estimate_potential_gradient(q, potential_v_fn)
        gamma = self._phase2_conformal_christoffel(g_base, g_inv, grad_v, gap)
        torsion = self._phase2_koszul_torsion_freeness(gamma)
        self._phase2_christoffel_acceleration_identity(
            gamma, g_base, g_inv, grad_v, gap
        )

        logger.debug(
            "FASE2.Ω conforme: n=%.6e, Δ=%.6e, κ=%.3e, τ=%.3e, ∇V=%s",
            n_factor,
            gap,
            cond,
            torsion,
            grad_method,
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
            phase1=dilation,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FASE 3 — INTEGRACIÓN DE TRAYECTORIA Y ACCIÓN (Decide & Act)
# Continuación directa de sintonizar_metrica_conforme (FASE 2.Ω) vía FASE 3.0
# Objetos: cascarón H0, geodésica RK4, E_F, T_Fermat, estado físico
# Teorías: geodésicas afines, conservación Fermat, Hamiltoniano diagnóstico
# ═══════════════════════════════════════════════════════════════════════════════
class Phase3_FermatGeodesicSolver(Phase2_ConformalKoszulSuturator):
    r"""
    Fase 3 (Decide & Act): integra la geodésica de Fermat y audita invariantes.

    Morfismo compuesto:

    \[
    \mathrm{DecideFermat}
    =(E_F,\,\mathrm{RK4},\,a_{\widetilde{\Gamma}},\,\mathrm{Shell})
    \circ\mathrm{Consume}\circ\mathrm{OrientConformal}^*.
    \]

    El primer morfismo, ``_phase3_consume_phase2_certificate``, *es* la
    continuación estricta de ``sintonizar_metrica_conforme``.
    """

    # ── FASE 3.0 · ingesta funtorial del certificado de FASE 2.Ω ──────────
    def _phase3_consume_phase2_certificate(
        self,
        bundle: ConformalManifoldBundle,
        q_start: NDArray[np.float64],
        v_start: NDArray[np.float64],
    ) -> Tuple[ConformalManifoldBundle, NDArray[np.float64], NDArray[np.float64], int]:
        r"""
        FASE 3.0 — Ingesta funtorial del certificado de FASE 2.Ω.

        **Continuación estricta de**
        ``Phase2_ConformalKoszulSuturator.sintonizar_metrica_conforme``.
        Verifica la coherencia geométrica de \(\widetilde{G}\) y
        \(\widetilde{\Gamma}\) (shape \(d^3\)) y entrega el objeto de
        trabajo de Decide *sin reconstruir la métrica conforme*.
        """
        if not isinstance(bundle, ConformalManifoldBundle):
            raise TypeError("bundle debe ser ConformalManifoldBundle")
        d = int(bundle.g_base.shape[0])
        if bundle.g_conformal.shape != (d, d) or bundle.g_conformal_inv.shape != (d, d):
            raise ConformalSingularityError(
                "FASE3.0: tensores conformes incoherentes con g_base."
            )
        if bundle.christoffel_symbols.shape != (d, d, d):
            raise ConformalSingularityError(
                f"FASE3.0: Christoffel shape={bundle.christoffel_symbols.shape} "
                f"≠ {(d, d, d)}."
            )
        q = np.asarray(q_start, dtype=np.float64)
        v = np.asarray(v_start, dtype=np.float64)
        self._assert_vector(q, d, "q_start")
        self._assert_vector(v, d, "v_start")
        logger.debug(
            "FASE3.0 consume F2: d=%d, κ=%.3e, n=%.6e, τ=%.3e",
            d,
            bundle.wilkinson_condition,
            bundle.refractive_index,
            bundle.torsion_residual,
        )
        return bundle, q, v, d

    # ── FASE 3.1 ──────────────────────────────────────────────────────────
    def _phase3_resolve_base_inverse(
        self,
        bundle: ConformalManifoldBundle,
    ) -> NDArray[np.float64]:
        r"""
        Recupera \(G^{-1}\) del bundle (FASE 2.3) o la reconstruye por Cholesky
        si un caller construyó el DTO a mano (compatibilidad).
        """
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
        r"""
        FASE 3.1 — Proyección al cascarón de Maupertuis \(T=H_0-V\).

        Una braquistócrona de energía \(H_0\) vive en

        \[
        \tfrac12 v^\top G v=H_0-V(q).
        \]

        Si \(v=0\), se toma la dirección de descenso \(-G^{-1}\nabla V\)
        (o \(e_1\) si el gradiente es nulo). El residuo pre-proyección se
        reporta como diagnóstico.
        """
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
        logger.debug(
            "FASE3.1 cascarón: r_H=%.3e, proyectado=%s, ‖v‖_G²→%.6e",
            residual,
            True,
            target_quad,
        )
        return np.asarray(scaled, dtype=np.float64), residual, True

    # ── FASE 3.2 ──────────────────────────────────────────────────────────
    def _phase3_conformal_acceleration(
        self,
        q: NDArray[np.float64],
        v: NDArray[np.float64],
        g_base: NDArray[np.float64],
        g_inv: NDArray[np.float64],
        potential_v_fn: PotentialField,
        initial_h0: float,
    ) -> NDArray[np.float64]:
        r"""
        FASE 3.2 — Aceleración geodésica conforme en forma cerrada \(\mathcal{O}(d^2)\).

        \[
        a
        =-2(\nabla\phi\cdot v)\,v
         +\|v\|_G^2\,G^{-1}\nabla\phi,
        \qquad
        \nabla\phi=\frac{\nabla V}{2(H_0-V)}.
        \]

        Equivale a \(a^\rho=-\widetilde{\Gamma}^\rho_{\mu\nu}v^\mu v^\nu\)
        sin materializar el tensor \(d^3\) ni invocar Koszul en cada etapa RK4.
        """
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
        r"""
        Campo vectorial geodésico (fachada):

        \[
        \dot q=v,
        \qquad
        \dot v=a_{\widetilde{\Gamma}}(q,v).
        \]
        """
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

    # ── FASE 3.3 ──────────────────────────────────────────────────────────
    def _phase3_fermat_energy(
        self,
        q: NDArray[np.float64],
        v: NDArray[np.float64],
        g_base: NDArray[np.float64],
        potential_v_fn: PotentialField,
        initial_h0: float,
    ) -> float:
        r"""
        FASE 3.3 — Invariante Fermat \(E_F=\tfrac12 n^2\|v\|_G^2\).

        Es la energía cinética de la métrica conforme y se conserva a
        lo largo de geodésicas afines. En el cascarón \(E_F=\tfrac12\).
        """
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
        r"""
        Hamiltoniano físico \(H=T+V\) (diagnóstico; no es el invariante afín).

        El flujo RK4 *no* es canónico para \(H\): conserva \(E_F\). \(H\)
        se reporta para auditar la adhesión al cascarón de Maupertuis.
        """
        val_v = self._as_finite_scalar(potential_v_fn(q), "V(q)")
        kinetic = 0.5 * float(v @ g_base @ v)
        return kinetic + val_v, val_v

    # ── FASE 3.4 ──────────────────────────────────────────────────────────
    def _phase3_rk4_step(
        self,
        state: NDArray[np.float64],
        dt: float,
        g_base: NDArray[np.float64],
        g_inv: NDArray[np.float64],
        potential_v_fn: PotentialField,
        initial_h0: float,
    ) -> NDArray[np.float64]:
        """FASE 3.4 — Un paso de Runge–Kutta 4 para \(\dot q=v\), \(\dot v=a(q,v)\)."""
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

    # ── FASE 3.Ω · composición terminal Decide (integración) ──────────────
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
        r"""
        FASE 3.Ω — Integra la geodésica de Fermat mediante RK4.

        **Continuación funtorial de FASE 2.Ω**: consume
        ``ConformalManifoldBundle`` vía FASE 3.0.

        El parámetro de integración es el parámetro *afín* \(\lambda\) de
        \(\widetilde{G}\). El tiempo de tránsito físico/óptico es la
        longitud de Fermat

        \[
        T=\int n(q)\,\|dq\|_G
         =\int\sqrt{\widetilde{G}(\dot q,\dot q)}\,d\lambda.
        \]

        El invariante de conservación es \(E_F\), no \(H=T+V\) (este último
        se reporta como diagnóstico de cascarón). La estabilidad global
        exige finitud, no-caústica y

        \[
        \delta E_F\le\varepsilon_F^{\mathrm{hard}},
        \qquad
        \delta H\le\varepsilon_H^{\mathrm{hard}}.
        \]

        **Contrato funtorial F3 → objeto terminal**:
        ``BrachistochronePhysicalState``.
        """
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
            except (GeodesicIntegrationError, EnergyWellColapsoError, ArithmeticError) as exc:
                logger.warning("Singularidad de observación en paso %d: %s", step, exc)
                is_globally_stable = False
                break

            energy_drift_max = max(energy_drift_max, abs(ham - initial_h0))
            fermat_drift_max = max(fermat_drift_max, abs(e_f - e_f_0))

            v_quad = float(v_curr @ g_base @ v_curr)
            v_norm_g = math.sqrt(max(0.0, v_quad))
            gap = initial_h0 - val_v
            if gap <= _ENERGY_GAP_FLOOR:
                logger.warning(
                    "Barrera de energía alcanzada en paso %d (Δ=%.3e).",
                    step,
                    gap,
                )
                is_globally_stable = False
                break
            n_factor = 1.0 / math.sqrt(2.0 * gap)
            transit_time_t += n_factor * v_norm_g * dt

            try:
                state = self._phase3_rk4_step(
                    state, dt, g_base, g_inv, potential_v_fn, initial_h0
                )
            except GeodesicIntegrationError as exc:
                logger.warning("Singularidad geodésica en paso %d: %s", step, exc)
                is_globally_stable = False
                break

        if energy_drift_max > _ENERGY_DRIFT_HARD_TOL:
            is_globally_stable = False
        if fermat_drift_max > _FERMAT_DRIFT_HARD_TOL:
            is_globally_stable = False

        used = last_index + 1
        logger.debug(
            "FASE3.Ω integrar: steps=%d/%d, T=%.6e, δH=%.3e, δE_F=%.3e, stable=%s",
            used,
            steps,
            transit_time_t,
            energy_drift_max,
            fermat_drift_max,
            is_globally_stable,
        )
        return BrachistochronePhysicalState(
            trajectory=self._freeze_array(trajectory[:used]),
            velocities=self._freeze_array(velocities[:used]),
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
            engine_version=__version__,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MOTOR DE RESOLUCIÓN COVARIANTE DE BRAQUISTÓCRONAS (Orquestador Ciego)
# Observe (F1) ⟶ Orient (F2) ⟶ Decide (F3) ⟶ Estado físico
# ═══════════════════════════════════════════════════════════════════════════════
class BrachistochronePathFinder(Morphism, Phase3_FermatGeodesicSolver):
    r"""
    Motor físico puro de la braquistócrona de Fermat–Jacobi.

    Resolvedor ciego de-confinado y libre de realimentaciones tácticas
    exógenas (sin retículo de Heyting ni Crowbar: esos viven en el agente
    soberano). Endofuntor

    \[
    \mathcal{Z}_{\mathrm{motor}}
    :
    (G,q_0,v_0,V,H_0)
    \longrightarrow
    \mathbf{PhysicalState}
    \]

    compuesto como

    \[
    \mathrm{Integrate}\circ\mathrm{Conformal}\circ\mathrm{Observe}.
    \]
    """

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, state: Any = None, **kwargs: Any) -> Any:
        r"""Invocación como morfismo categórico."""
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
        r"""
        Orquesta de forma estrictamente funcional la resolución geodésica.

        El objeto terminal de cada fase es el objeto inicial de la siguiente:

        .. code-block:: text

            ┌────────────────────────────────────────────────────────────┐
            │ FASE 1  Observe / Energy                                   │
            │   1.1 validate_manifold_dimension                          │
            │   1.2 validate_metric_tensor                               │
            │   1.3 certify_metric_symmetry                              │
            │   1.4 cholesky_spd_factor                                  │
            │   1.5 validate_potential_samples                           │
            │   1.6 energy_gap_spectrum                                  │
            │   1.7 validate_initial_kinematics                          │
            │   1.Ω evaluate_potential_barrier  ──► Dilation ──┐         │
            ├──────────────────────────────────────────────────┼─────────┤
            │ FASE 2  Orient / Conformal  ◄────────────────────┘         │
            │   2.0 consume_phase1_certificate                           │
            │   2.1 conformal_factor_at_q                                │
            │   2.2 assemble_jacobi_metric                               │
            │   2.3 invert_conformal_cholesky                            │
            │   2.4 wilkinson_condition                                  │
            │   2.5 estimate_potential_gradient                          │
            │   2.6 conformal_christoffel  (forma cerrada)               │
            │   2.7 koszul_torsion_freeness                              │
            │   2.8 christoffel_acceleration_identity                    │
            │   2.Ω sintonizar_metrica_conforme  ──► Bundle ──┐          │
            ├─────────────────────────────────────────────────┼──────────┤
            │ FASE 3  Decide / Integrate  ◄───────────────────┘          │
            │   3.0 consume_phase2_certificate                           │
            │   3.1 project_energy_shell                                 │
            │   3.2 conformal_acceleration                               │
            │   3.3 fermat_energy                                        │
            │   3.4 rk4_step                                             │
            │   3.Ω integrar_trayectoria_conforme  ──► PhysicalState     │
            └────────────────────────────────────────────────────────────┘

        Args:
            g_base: tensor métrico SPD \((d,d)\), constante.
            q_start, v_start: condiciones iniciales en \(T\mathcal{Q}\).
            potential_v_fn: campo \(V(q)\to\mathbb{R}\) (holomorfo ⇒ CSMD).
            initial_h0: energía total de Maupertuis.
            t_max, dt: horizonte y paso del parámetro afín.
            project_on_shell: reescala \(v_0\) al cascarón \(T=H_0-V\).

        Returns:
            BrachistochronePhysicalState: certificado físico inmutable.

        Raises:
            EnergyWellColapsoError, ConformalSingularityError,
            GeodesicIntegrationError, KoszulChecksumError,
            InitialKinematicsError.
        """
        self._assert_callable_potential(potential_v_fn)
        g = np.asarray(g_base, dtype=np.float64)
        q0 = np.asarray(q_start, dtype=np.float64)
        v0 = np.asarray(v_start, dtype=np.float64)
        self._assert_square_matrix(g, "g_base")
        self._assert_vector(q0, g.shape[0], "q_start")
        self._assert_vector(v0, g.shape[0], "v_start")
        self._assert_finite_scalar(float(initial_h0), "initial_h0")

        val_start = self._as_finite_scalar(potential_v_fn(q0), "potential_v_fn(q_start)")

        # ── FASE 1 · Observe ──────────────────────────────────────────────
        dilation = self.evaluate_potential_barrier(
            g_base=g,
            potential_v=np.array([val_start], dtype=np.float64),
            initial_h0=float(initial_h0),
            q_start=q0,
            v_start=v0,
            potential_at_start=val_start,
        )

        # ── FASE 2 · Orient  (continúa certificado F1.Ω) ──────────────────
        bundle = self.sintonizar_metrica_conforme(
            dilation=dilation,
            q_eval=q0,
            potential_v_fn=potential_v_fn,
        )

        # ── FASE 3 · Decide  (continúa certificado F2.Ω) ──────────────────
        return self.integrar_trayectoria_conforme(
            bundle=bundle,
            q_start=q0,
            v_start=v0,
            potential_v_fn=potential_v_fn,
            t_max=t_max,
            dt=dt,
            project_on_shell=project_on_shell,
        )

    # ─────────────────────────────────────────────────────────────────────
    # Fábricas de referencia (calibración / tests del motor)
    # ─────────────────────────────────────────────────────────────────────
    @staticmethod
    def euclidean_metric(dimension_d: int) -> NDArray[np.float64]:
        r"""Métrica euclídea \(G=I_d\)."""
        if dimension_d < 1:
            raise ValueError(f"dimension_d debe ser ≥ 1; recibido {dimension_d}")
        return np.eye(int(dimension_d), dtype=np.float64)

    @staticmethod
    def harmonic_potential(
        omega: float = 1.0,
        center: Optional[NDArray[np.float64]] = None,
    ) -> Callable[[NDArray[Any]], Any]:
        r"""
        Pozo armónico \(V(q)=\tfrac12\omega^2\|q-q_\star\|^2\).

        Holomorfo en \(q\) (usa \(q\cdot q\), no \(|q|^2\)), por lo que
        admite CSMD.
        """
        if omega < 0.0 or not math.isfinite(omega):
            raise ValueError(f"omega debe ser ≥ 0 y finito; recibido {omega}")

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
        r"""
        Gravedad lineal \(V(q)=-g\,q_{\mathrm{axis}}\) (braquistócrona clásica
        de Bernoulli cuando \(G=I_2\) y el eje es el vertical).
        """
        if not math.isfinite(g_acc):
            raise ValueError(f"g_acc debe ser finito; recibido {g_acc}")

        def _v(q: NDArray[Any]) -> Any:
            return -g_acc * np.asarray(q)[axis]

        return _v

    @staticmethod
    def constant_potential(value: float = 0.0) -> Callable[[NDArray[Any]], Any]:
        """Potencial constante (geodésicas = rectas de \(G\))."""
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