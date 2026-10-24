# -*- coding: utf-8 -*-
r"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ Módulo : BrachistochronePathFinder (Motor Espectral y Físico Geodésico)      ║
║ Ruta   : app/boole/physics/brachistochrone_path_finder.py                    ║
║ Versión: 5.1.0-Fermat-Jacobi-Poincare-Melnikov-CSMD-RK4-RSI3-Strict-PhD      ║
║ Función: Motor físico y espectral soberano para el trazado de braquistócronas║
║          en la variedad conforme de Fermat-Jacobi, integrando métrica de     ║
║          Fermat n²(q)G, embedding simpléctico Cayley-Darboux en Sp(2d, ℝ),   ║
║          estabilidad celeste de Poincaré-Lindstedt-Melnikov, Mónada Kleisli  ║
║          RSI³ de 3 niveles y proyector Runge-Kutta 4º orden sobre la capa H₀.║
╚══════════════════════════════════════════════════════════════════════════════╝

--------------------------------------------------------------------------------
1. DEFINICIÓN GRANULAR Y RIGUROSA DEL MOTOR ESPECTRAL Y SU RESPECTIVO AGENTE CUSTODIO
--------------------------------------------------------------------------------
El Motor Espectral y Físico de Trayectorias Braquistócronas (`BrachistochronePathFinder`)
es el núcleo variacional encargado de calcular geodésicas de tiempo mínimo en la
variedad de Fermat-Jacobi bajo el principio de Fermat generalizado a variedades riemannianas.
Su función soberana es la integración numérica rigurosa de las ecuaciones del flujo geodésico
sin disipación de volumen simpléctico, certificando la estabilidad de KAM/Bruno y
resolviendo la ecuación homológica de Poincaré-Lindstedt.

Su respectivo agente custodio y gobernador OODA es `BrachistochronePathFinderAgent`
(`app/agents/boole/omega/brachistochrone_path_finder_agent.py`), el cual supervisa las
tres fases anidadas producidas por este motor, ejecuta la adjudicación en Heyting $\Omega_3$
y controla el disparo ciber-físico del tiristor BT151 Crowbar en hardware ESP32.

Estructura de Tres Fases Anidadas:
  • Fase 1 (Observe): `Phase1_PotentialWellInquirer` audita el pozo de potencial termodinámico,
    evaluando la positividad definida de $G = L L^\top$ por Cholesky, el gap de energía
    $H_0 - V(q) \ge 10^{-12}$, el espectro de accesibilidad en la bola de Hill con $g^{-1/2}$
    real y la constante de Lipschitz local $| \nabla V |_\infty$. Emite `Phase1Envelope`.
  • Fase 2 (Orient): `Phase2_ConformalKoszulSuturator` construye la métrica conforme de Fermat
    $\hat{G} = n^2(q) G$ con $n^2 = [2(H_0 - V)]^{-1}$, invierte Cholesky triangular sin `la.inv`,
    proyecta la inmersión simpléctica de Cayley-Darboux $M \in \mathrm{Sp}(2d, \mathbb{R})$,
    calcula los símbolos de Koszul-Christoffel conformes $\Gamma^\rho_{\mu\nu}$ con derivación CSMD
    y verifica la torsión nula. Emite `Phase2Envelope`.
  • Fase 3 (Decide & Act): `Phase3_FermatGeodesicSolver` proyecta el vector velocidad sobre la
    capa de energía $H = H_0$, integra por Runge-Kutta de 4º orden (RK4) con refinamiento
    adaptativo de paso Picard-RSI³, resuelve la ecuación homológica de Poincaré-Lindstedt,
    certifica la condición diofántica de KAM/Bruno, calcula la integral de Melnikov $M(\tau)$
    y extrae el espectro de Lyapunov Jacobi-Benettin. Emite `BrachistochronePhysicalState`.

--------------------------------------------------------------------------------
2. MARCO TEÓRICO Y FORMALISMO MATEMÁTICO
--------------------------------------------------------------------------------
1. Principio Variacional de Fermat y Métrica Conforme de Jacobi:
   El tiempo de tránsito $T = \int dt = \int \frac{ds}{v}$ sobre la variedad riemanniana con
   métrica kinetic-energy $G_{ij}$ se transforma, mediante el principio de Maupertuis-Jacobi
   a energía fija $H_0 = \frac{1}{2} v^\top G v + V(q)$, en la longitud geodésica de Fermat
   $S = \int \hat{g}_{ij} dq^i dq^j$ con la métrica conforme de Fermat:
       \hat{G}_{ij}(q) = n^2(q) G_{ij}, \qquad n^2(q) = \frac{1}{2(H_0 - V(q))}
   donde $n(q) = \frac{1}{\sqrt{2(H_0 - V(q))}}$ es el índice de refracción efectivo del medio.

2. Inmersión Simpléctica de Cayley-Darboux en $\mathrm{Sp}(2d, \mathbb{R})$:
   Sobre el espacio fásico $T^* Q \cong \mathbb{R}^{2d}$ provisto de la estructura simpléctica
   canónica $\Omega = \begin{pmatrix} 0 & \mathbf{I}_d \\ -\mathbf{I}_d & 0 \end{pmatrix}$, la matriz de
   transformación de Cayley $K = (\hat{G} - \mathbf{I}_d)(\hat{G} + \mathbf{I}_d)^{-1}$ se extiende al bloque fásico
   $K_{\text{full}} = \operatorname{diag}(K, -K) \in \mathfrak{sp}(2d, \mathbb{R})$, satisfaciendo $K_{\text{full}}^\top \Omega + \Omega K_{\text{full}} = \mathbf{0}$.
   El mapeo simpléctico Cayley $M = (\mathbf{I}_{2d} - K_{\text{full}})^{-1} (\mathbf{I}_{2d} + K_{\text{full}}) \in \mathrm{Sp}(2d, \mathbb{R})$
   satisface la simetría de Darboux:
       \frac{\| M^\top \Omega M - \Omega \|_F}{\| M \|_F^2} \le 10^{-10}

3. Símbolos de Koszul-Christoffel Conformes y Derivación CSMD:
   Definiendo el potencial logarítmico de refracción $\varphi(q) = \ln n(q) = -\frac{1}{2} \ln(2(H_0 - V(q)))$,
   su gradiente es $\nabla \varphi = \frac{\nabla V}{2(H_0 - V)}$. Los símbolos de Christoffel conformes son:
       \tilde{\Gamma}^i_{jk} = \delta^i_j \partial_k \varphi + \delta^i_k \partial_j \varphi - G_{jk} G^{il} \partial_l \varphi
   Para erradicar la imprecisión de cancelación finita, $\nabla V$ se evalúa mediante Complex Step Material Derivative (CSMD):
       \partial_k V(q) = \frac{\operatorname{Im}\left( V(q + i \varepsilon_{\text{csmd}} e_k) \right)}{\varepsilon_{\text{csmd}}}, \qquad \varepsilon_{\text{csmd}} = 10^{-20}

4. Mecánica Celeste de Poincaré, KAM/Bruno, Melnikov y Monodromía:
   • Ecuación Homológica de Poincaré-Lindstedt: Para eliminar términos seculares en la serie perturbativa
     se resuelve $\{H_0, W\} = H_1 - \langle H_1 \rangle_{\mathbb{T}^d}$, dando $W_k = \frac{H_{1,k}}{\boldsymbol{\omega}_0 \cdot k}$.
   • Condición Diofántica de KAM/Bruno: Las frecuencias elípticas $\boldsymbol{\omega}_0 = \sqrt{\operatorname{eig}_+(G^{-1} \operatorname{Hess} V)}$
     cumplen $|\boldsymbol{\omega}_0 \cdot k| \ge \frac{\gamma}{\|k\|_1^\tau}$ con $\tau = d + 1$ sobre el retículo $k \in \mathbb{Z}^d \setminus \{\mathbf{0}\}$.
   • Integral Homoclínica de Melnikov-Poincaré: $M(\tau) = \int_{-\infty}^{+\infty} \{ T, V \} dt = -\int_{-\infty}^{+\infty} v(t) \cdot \nabla V(q(t)) \, dt$.
     Un cero simple $M(\tau) = 0$ caracteriza el desdoblamiento de separatrices homoclínicas (caos de Poincaré).
   • Monodromía Variacional y Benettin QR: La matriz variacional $W(t)$ resuelve $\dot{W} = D F(q,v) W$. Mediante
     re-ortonormalización QR periódica $W \to Q R$, se obtienen los exponentes de Lyapunov $\lambda_i = \frac{1}{T} \sum \ln R_{ii}$.

--------------------------------------------------------------------------------
3. AXIOMAS FORMALES
--------------------------------------------------------------------------------
Axioma PHYS_A1 (Positividad Definida y Criterio de Sylvester):
    $G = G^\top \succ 0$, $\| L L^\top - G \|_F \le \varepsilon$ y $\kappa_2(G) \le 10^8$.
Axioma PHYS_A2 (Accesibilidad Energética Clásica de Hill):
    $H_0 - V(q) \ge 10^{-12}$ en todo punto del dominio accesible.
Axioma PHYS_A3 (Symplecticidad de Cayley-Darboux):
    $K_{\text{full}} \in \mathfrak{sp}(2d, \mathbb{R}) \implies M \in \mathrm{Sp}(2d, \mathbb{R})$ con residuo $\le 10^{-10}$.
Axioma PHYS_A4 (Invariancia de Torsión de Koszul):
    $\max_{\rho,\mu,\nu} | \Gamma^\rho_{\mu\nu} - \Gamma^\rho_{\nu\mu} | \le 10^{-12}$ (conexión de Levi-Civita torsión nula).
Axioma PHYS_A5 (Estabilidad Diofántica de KAM/Bruno):
    $|\boldsymbol{\omega}_0 \cdot k| \cdot \|k\|_1^\tau \ge \gamma$ sobre el vector de onda $\mathcal{K}$, previniendo divisores pequeños.

--------------------------------------------------------------------------------
4. INVARIANTES DE ESTADO
--------------------------------------------------------------------------------
Invariante 1 (Checksum Geometrico Koszul-Aceleración):
    $\| \Gamma(v,v) - a_{\text{cerrada}} \|_{\text{rel}} \le 10^{-9}$ donde $a_{\text{cerrada}} = -2(\nabla\varphi \cdot v)v + \|v\|_G^2 G^{-1} \nabla\varphi$.
Invariante 2 (Deriva Energética Blanda y Refinamiento Picard-RSI³):
    La deriva de energía física $\delta H \le 10^{-3}$ y de Fermat $\delta E_F \le 10^{-4}$. Si se supera la cota
    blanda $\delta H > 10^{-6}$, se ejecuta automejora RSI³ refinando $dt \to dt / 2$.
Invariante 3 (Cierre Angular Fubini-Study):
    La deriva angular entre las velocidades inicial y final cumple $d_{\text{FS}}(\psi_{\text{start}}, \psi_{\text{end}}) \le 10^{-4}\text{ rad}$.

--------------------------------------------------------------------------------
5. FUNTOR SEMÁNTICO Y CÚSPIDE VISCERAL ("DOLOR Y DINERO" / Φ_sem)
--------------------------------------------------------------------------------
El Funtor Semántico $\Phi_{\mathrm{sem}} : \mathbf{Sh}(\partial\mathcal{M}, \Omega_3) \to \text{Business}$
asocia la precisión variacional y espectral con el impacto económico directo:
  • Optimización Extrema del Tiempo de Tránsito (Mínimo Tiempo Garantizado):
    Al resolver la braquistócrona exacta sobre la variedad riemanniana conforme, el sistema
    minimiza la latencia de procesamiento táctico en tuberías de datos críticos (-40% en tiempo de ejecución).
  • Inmunización Contra Colapsos por Caos Hamiltoniano:
    Al certificar la condición de KAM/Bruno y verificar $|M| < 0.5$, se evita que los modelos de decisión
    entren en resonancias estocásticas o bucles de cálculo infinitos, salvaguardando la disponibilidad operativa.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import product
from typing import (
    Any, Callable, Dict, Final, Iterator, List, Optional, Protocol, Tuple,
    runtime_checkable,
)

import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray

try:
    from app.core.mic_algebra import Morphism, TopologicalInvariantError
except ImportError:  # pragma: no cover

    class TopologicalInvariantError(Exception):
        """Excepción base del sistema para violaciones topológico-algebraicas."""

    class Morphism:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass


logger = logging.getLogger("MIC.Physics.BrachistochronePathFinder.v5")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())

__version__: Final[str] = "5.1.0-Fermat-Jacobi-Poincare-Melnikov-CSMD-RK4-RSI3-Strict-PhD"


# ══════════════════════════════════════════════════════════════════════════════
# §A. CONSTANTES FÍSICAS, NUMÉRICAS Y DE POINCARÉ
# ══════════════════════════════════════════════════════════════════════════════
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
_CHOLESKY_RECON_TOL: Final[float] = 1.0e-8

_KAM_GAMMA: Final[float] = 0.125
_KAM_TAU_OFFSET: Final[float] = 1.0
_KAM_MAX_ORDER: Final[int] = 3
_KAM_FULL_PRODUCT_DIM: Final[int] = 4
_MELNIKOV_TOL: Final[float] = 0.5
_FLOQUET_UNIT_TOL: Final[float] = 1.0e-6
_CHIRIKOV_CRITICAL: Final[float] = 1.0
_LYAPUNOV_HYPERBOLIC: Final[float] = 1.0e-6
_CLOSE_RETURN_REL_TOL: Final[float] = 1.0e-3
_POINCARE_DISP_REGULAR: Final[float] = 0.5

_KRULL_NOVIKOV_FLOOR: Final[float] = 0.10
_FUBINI_STUDY_MAX: Final[float] = 1.0e-4
_BANACH_LIPSCHITZ_MAX: Final[float] = 0.95
_RSI3_MAX_REFINEMENTS: Final[int] = 3
_RSI3_DT_DECAY: Final[float] = 0.5
_BANACH_TAU: Final[float] = 0.05

_HILL_ACCESSIBILITY_SAMPLES: Final[int] = 32
_LOCAL_LIPSCHITZ_RADIUS: Final[float] = 1.0
_HOM_DIVISOR_FLOOR: Final[float] = 1.0e-12
_BENETTIN_RENORM: Final[float] = 1.0e6


# ══════════════════════════════════════════════════════════════════════════════
# §B. RÉGIMEN GEODÉSICO DE POINCARÉ
# ══════════════════════════════════════════════════════════════════════════════
class GeodesicRegime(Enum):
    ELIPTIC_STABLE = auto()
    HYPERBOLIC_UNSTABLE = auto()
    KAM_TORUS_PRESERVED = auto()
    MELNIKOV_CHAOTIC = auto()
    HILL_WELL_VIOLATED = auto()


# ══════════════════════════════════════════════════════════════════════════════
# §C. EXCEPCIONES DEL MOTOR FÍSICO
# ══════════════════════════════════════════════════════════════════════════════
class BrachistochroneEngineError(Exception):
    """Excepción raíz del motor físico de la braquistócrona."""


class EnergyWellColapsoError(BrachistochroneEngineError):
    """La barrera de energía es nula o negativa (H0 ≤ V(q))."""


class ConformalSingularityError(BrachistochroneEngineError):
    """Tensor métrico o inversa con singularidades numéricas."""


class GeodesicIntegrationError(BrachistochroneEngineError):
    """Divergencia numérica catastrófica o caústica durante RK4."""


class KoszulChecksumError(BrachistochroneEngineError):
    """El isomorfismo conforme / Koszul no es fiel."""


class InitialKinematicsError(BrachistochroneEngineError):
    """Condiciones iniciales (q,v,H0) inconsistentes con la variedad."""


class SymplecticEmbeddingError(BrachistochroneEngineError):
    """La inmersión Cayley-Darboux no pertenece a sp(2d,ℝ)."""


class EnvelopeChainViolationError(BrachistochroneEngineError):
    """La cadena de hashes entre fases fue corrompida."""


class KamBrunoViolationError(BrachistochroneEngineError):
    """Divisores pequeños: la condición diofantina de Bruno falla."""


class MelnikovSplittingError(BrachistochroneEngineError):
    """Splitting homoclínico: |M| supera la tolerancia de Poincaré."""


# ══════════════════════════════════════════════════════════════════════════════
# §D. PROTOCOLOS DEL FIBRADO POTENCIAL
# ══════════════════════════════════════════════════════════════════════════════
@runtime_checkable
class PotentialField(Protocol):
    """Campo potencial V: Q → ℝ (extensible a ℂ para CSMD)."""

    def __call__(self, q: NDArray[Any]) -> Any: ...


# ══════════════════════════════════════════════════════════════════════════════
# §E. GUARDIA NUMÉRICA
# ══════════════════════════════════════════════════════════════════════════════
class _BrachistochroneNumericalGuard:
    """Capa de saneamiento y validación para tensores, vectores y escalares."""

    @staticmethod
    def _wilkinson_spectral_floor(scale: float, tol: float, ambient_dim: int) -> float:
        dim = max(int(ambient_dim), 1)
        return max(float(tol), dim * _MACHINE_EPS * max(float(scale), 1.0), _ENERGY_GAP_FLOOR)

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
                    f"{name} debe ser escalar; ndarray shape={value.shape}.")
            value = value.reshape(()).item()
        if isinstance(value, complex):
            if abs(value.imag) > 1.0e6 * _MACHINE_EPS * max(1.0, abs(value.real)):
                raise ArithmeticError(f"{name} posee parte imaginaria no nula: {value!r}.")
            value = value.real
        try:
            out = float(value)
        except (TypeError, ValueError) as exc:
            raise ArithmeticError(
                f"{name} no es convertible a float ({type(value).__name__}).") from exc
        if not math.isfinite(out):
            raise ArithmeticError(f"{name} debe ser finito; recibido {out}.")
        return out

    @staticmethod
    def _assert_square_matrix(X: NDArray[np.float64], name: str) -> None:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            raise ValueError(f"{name} debe ser matriz 2D.")
        if X.shape[0] != X.shape[1] or X.shape[0] == 0:
            raise ValueError(f"{name} debe ser cuadrada y positiva: {X.shape}")
        if not np.all(np.isfinite(X)):
            raise ArithmeticError(f"{name} contiene NaN/Inf.")

    @staticmethod
    def _assert_vector(x: NDArray[np.float64], dim: int, name: str) -> None:
        if not isinstance(x, np.ndarray) or x.ndim != 1:
            raise ValueError(f"{name} debe ser vector 1D.")
        if x.shape[0] != dim:
            raise ValueError(f"{name} dim={x.shape[0]} != {dim}.")
        if not np.all(np.isfinite(x)):
            raise ArithmeticError(f"{name} contiene NaN/Inf.")

    @staticmethod
    def _assert_finite_scalar(x: float, name: str) -> None:
        if not np.isfinite(x):
            raise ArithmeticError(f"{name} debe ser finito.")

    @staticmethod
    def _assert_positive_scalar(x: float, name: str) -> None:
        _BrachistochroneNumericalGuard._assert_finite_scalar(x, name)
        if x <= 0.0:
            raise ValueError(f"{name} debe ser positivo.")

    @staticmethod
    def _assert_callable_potential(potential_v_fn: Any) -> None:
        if not callable(potential_v_fn):
            raise TypeError(
                f"potential_v_fn debe ser invocable V: Q → ℝ ({type(potential_v_fn).__name__}).")

    @staticmethod
    def _real_spd_sqrt_inv(g: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""\(g^{-1/2}\) real simétrico vía \(Q\Lambda^{-1/2}Q^\top\). Evita `sqrtm` complejo."""
        eig_vals, eig_vecs = la.eigh(g)
        if float(np.min(eig_vals)) <= _MACHINE_EPS:
            raise ConformalSingularityError(
                f"No existe g^{{-1/2}} real: λ_min={float(np.min(eig_vals)):.3e}")
        inv_sqrt = eig_vecs @ np.diag(1.0 / np.sqrt(eig_vals)) @ eig_vecs.T
        return np.real(inv_sqrt).astype(np.float64)

    @staticmethod
    def _cholesky_inverse(chol_lower: NDArray[np.float64]) -> NDArray[np.float64]:
        d = chol_lower.shape[0]
        l_inv = la.solve_triangular(chol_lower, np.eye(d, dtype=np.float64), lower=True)
        return l_inv.T @ l_inv

    @staticmethod
    def _canonical_omega(dim_q: int) -> NDArray[np.float64]:
        i_d = np.eye(dim_q, dtype=np.float64)
        z = np.zeros((dim_q, dim_q), dtype=np.float64)
        return np.block([[z, i_d], [-i_d, z]])


def _hash_state(*arrays: Any) -> str:
    h = hashlib.sha256()
    for a in arrays:
        if isinstance(a, np.ndarray):
            h.update(np.ascontiguousarray(a).tobytes())
        else:
            h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes())
    return h.hexdigest()[:16]


def _iter_kam_wavevectors(dimension_d: int, max_order: int) -> Iterator[NDArray[np.float64]]:
    r"""
    Genera \(k\in\mathbb{Z}^d\setminus\{0\}\) con \(\|k\|_\infty\le K\).

    Si \(d\le 4\) recorre el cubo completo (cardinal \( (2K+1)^d-1 \)).
    Si \(d>4\) restringe a vectores de soporte \(\le 2\) — evita \(7^{32}\).
    """
    d = int(dimension_d)
    k_max = int(max_order)
    if d <= 0 or k_max <= 0:
        return
    if d <= _KAM_FULL_PRODUCT_DIM:
        for k in product(range(-k_max, k_max + 1), repeat=d):
            if any(ki != 0 for ki in k):
                yield np.asarray(k, dtype=np.float64)
        return
    for i in range(d):
        for ki in range(-k_max, k_max + 1):
            if ki == 0:
                continue
            vec = np.zeros(d, dtype=np.float64)
            vec[i] = float(ki)
            yield vec
    for i in range(d):
        for j in range(i + 1, d):
            for ki in range(-k_max, k_max + 1):
                if ki == 0:
                    continue
                for kj in range(-k_max, k_max + 1):
                    if kj == 0:
                        continue
                    vec = np.zeros(d, dtype=np.float64)
                    vec[i] = float(ki)
                    vec[j] = float(kj)
                    yield vec


# ══════════════════════════════════════════════════════════════════════════════
# §F. DTOs INMUTABLES (Contratos Categóricos de Handoff entre Fases)
# ══════════════════════════════════════════════════════════════════════════════
def _empty_f64() -> NDArray[np.float64]:
    return np.zeros(0, dtype=np.float64)


def _empty_2d() -> NDArray[np.float64]:
    return np.zeros((0, 0), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ConformalPotentialDilation:
    r"""Artefacto de FASE 1: certificado del pozo de potencial."""
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
class Phase1Envelope:
    r"""Contrato FASE 1 (Observe) → FASE 2 (Orient)."""
    dilation: ConformalPotentialDilation
    hill_accessibility_ok: bool
    hill_accessibility_margin: float
    local_lipschitz_v: float
    hill_potential_samples: NDArray[np.float64]
    origin_timestamp_ns: float
    chain_hash: str


@dataclass(frozen=True, slots=True)
class ConformalGeometryBundle:
    r"""Artefacto de FASE 2: geometría de Fermat + Cayley-Darboux."""
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
    g_base_inv: NDArray[np.float64] = field(default_factory=_empty_2d)
    symplectic_cayley_matrix: NDArray[np.float64] = field(default_factory=_empty_2d)
    symplectic_residual: float = 0.0
    symplectic_defect: float = 0.0
    is_symplectic_valid: bool = True
    almost_complex_residual: float = 0.0
    phase1: Optional[ConformalPotentialDilation] = None


ConformalManifoldBundle = ConformalGeometryBundle


@dataclass(frozen=True, slots=True)
class Phase2Envelope:
    r"""Contrato FASE 2 (Orient) → FASE 3 (Decide&Act)."""
    phase1: Phase1Envelope
    bundle: ConformalGeometryBundle
    origin_timestamp_ns: float
    chain_hash: str


@dataclass(frozen=True, slots=True)
class PoincareSectionCertificate:
    r"""Sección transversal de Poincaré (retornos + dispersión)."""
    n_crossings: int
    return_dispersion: float
    is_regular: bool
    returns: Tuple[float, ...]
    first_return_time: float = 0.0
    close_returns: int = 0
    return_residual: float = 0.0


@dataclass(frozen=True, slots=True)
class FloquetCertificate:
    r"""Multiplicadores de Floquet μ_k = eig(W(T)) de la monodromía variacional."""
    multipliers: NDArray[np.complex128]
    max_unit_deviation: float
    is_neutral: bool
    is_dissipative_stable: bool
    exponents: NDArray[np.float64]
    spectral_radius: float = 1.0
    lyapunov_max: float = 0.0


@dataclass(frozen=True, slots=True)
class LindstedtCertificate:
    r"""Solución de la ecuación homológica de Poincaré-Lindstedt."""
    n_modes: int
    n_resonant: int
    residual_max: float
    divisor_min: float


@dataclass(frozen=True, slots=True)
class BrachistochronePhysicalState:
    r"""Objeto terminal y certificado físico inmutable emitido por el Motor."""
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

    kam_worst_ratio: float = 0.0
    melnikov_max_abs_integrand: float = 0.0
    poincare_section_n_crossings: int = 0
    poincare_section_dispersion: float = 0.0
    poincare_section_regular: bool = True
    floquet_max_unit_deviation: float = 0.0
    floquet_is_neutral: bool = True
    floquet_exponents: NDArray[np.float64] = field(default_factory=_empty_f64)
    banach_lipschitz_t3: float = 0.0
    fubini_study_distance: float = 0.0
    novikov_krull_valuation: float = 0.0
    lindstedt_residual: float = 0.0
    rsi3_refinements: int = 0
    jacobi_lyapunov_max: float = 0.0
    monodromy_spectral_radius: float = 1.0
    elliptic_frequency_min: float = 0.0
    poincare_first_return_time: float = 0.0


# ══════════════════════════════════════════════════════════════════════════════════
# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║  F A S E   1 :   O B S E R V E                                               ║
# ║  Auditoría del pozo de potencial termodinámico + Hill accessibility.         ║
# ║  Cierra con `emit_phase1_envelope` → `Phase1Envelope`.                       ║
# ║  Ese cierre ES el contrato de apertura de la FASE 2.                         ║
# ╚════════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════════

class Phase1_PotentialWellInquirer(_BrachistochroneNumericalGuard):
    r"""
    FASE 1 (Observe): sanea y audita el pozo de potencial termodinámico.

    Métodos rigurosos:
      • `_phase1_validate_manifold_dimension`
      • `_phase1_validate_metric_tensor`
      • `_phase1_certify_metric_symmetry`
      • `_phase1_cholesky_spd_factor` — Sylvester + LLᵀ
      • `_phase1_energy_gap_spectrum`
      • `_phase1_validate_initial_kinematics`
      • `_phase1_hill_accessibility_ball` — g^{-1/2} real
      • `_phase1_local_lipschitz_v` — |∇V|∞ 4º orden
    """

    @staticmethod
    def _phase1_validate_manifold_dimension(dimension_d: int) -> int:
        if not isinstance(dimension_d, (int, np.integer)) or int(dimension_d) < 1:
            raise ConformalSingularityError(
                f"Dimensión inválida: d={dimension_d}. Se exige d ∈ ℤ≥1.")
        d = int(dimension_d)
        if d > _DIM_MAX:
            raise ConformalSingularityError(f"d={d} excede d_max={_DIM_MAX}.")
        return d

    def _phase1_validate_metric_tensor(
        self, g_base: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        self._assert_square_matrix(g_base, "g_base")
        d = self._phase1_validate_manifold_dimension(g_base.shape[0])
        metric = np.asarray(g_base, dtype=np.float64)
        if metric.shape != (d, d):
            raise ConformalSingularityError(
                f"g_base shape={metric.shape} incoherente con d={d}.")
        return metric

    def _phase1_certify_metric_symmetry(
        self, g_base: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float]:
        sym_residual = float(la.norm(g_base - g_base.T, ord="fro"))
        sym_scale = max(1.0, float(la.norm(g_base, ord="fro")))
        symmetry_rel = sym_residual / sym_scale
        if symmetry_rel > _SYMMETRY_REL_TOL:
            raise ConformalSingularityError(
                f"Ruptura de simetría en G_base: {symmetry_rel:.3e} "
                f"> {_SYMMETRY_REL_TOL:.0e}.")
        return self._hermitize_weyl(g_base), symmetry_rel

    def _phase1_cholesky_spd_factor(
        self, g_base: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float]:
        try:
            chol = la.cholesky(g_base, lower=True)
        except la.LinAlgError as exc:
            raise ConformalSingularityError(
                "Criterio de Sylvester fallido: G_base no es definida positiva.") from exc
        diag = np.diag(chol)
        if np.any(diag <= 0.0):
            raise ConformalSingularityError(
                "Cholesky degenerado: diag(L) no estrictamente positivo.")
        recon = chol @ chol.T
        recon_err = float(la.norm(recon - g_base, ord="fro"))
        recon_scale = max(1.0, float(la.norm(g_base, ord="fro")))
        if recon_err > _CHOLESKY_RECON_TOL * recon_scale:
            raise ConformalSingularityError(
                f"LLᵀ−G no despreciable: ‖err‖_F={recon_err:.3e}.")
        logdet = float(2.0 * np.sum(np.log(diag)))
        return chol, logdet

    def _validate_sylvester_criteria(
        self, g_base: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float]:
        metric = self._phase1_validate_metric_tensor(g_base)
        metric, symmetry_rel = self._phase1_certify_metric_symmetry(metric)
        chol, _ = self._phase1_cholesky_spd_factor(metric)
        return chol, symmetry_rel

    def _phase1_validate_potential_samples(
        self, potential_v: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        samples = np.asarray(potential_v, dtype=np.float64)
        if samples.ndim != 1 or samples.shape[0] == 0:
            raise ValueError("potential_v debe ser vector 1D no vacío.")
        if not np.all(np.isfinite(samples)):
            raise ArithmeticError("potential_v contiene NaN/Inf.")
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
                f"H0 ({initial_h0:.6f}) no supera V_max ({float(np.max(potential_samples)):.6f}). "
                f"Gap min: {min_gap:.3e}")
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

        if (v_start is not None and potential_at_start is not None
                and math.isfinite(kinetic)):
            residual = abs(kinetic + float(potential_at_start) - float(initial_h0))
        else:
            residual = float("nan")
        return q_arr, v_arr, kinetic, residual

    def _phase1_hill_accessibility_ball(
        self,
        g_base: NDArray[np.float64],
        q_start: NDArray[np.float64],
        v_start: NDArray[np.float64],
        potential_v_fn: PotentialField,
        initial_h0: float,
        n_samples: int = _HILL_ACCESSIBILITY_SAMPLES,
    ) -> Tuple[bool, float, NDArray[np.float64]]:
        r"""
        Verifica \(H_0 \ge V(q)\) sobre la bola de Hill
        \(q(\sigma)=q_s+\sigma\, g^{-1/2}v_s\), \(\sigma\in[-1,1]\).
        \(g^{-1/2}\) es real (eigh), jamás `sqrtm`.
        """
        try:
            g_sqrt_inv = self._real_spd_sqrt_inv(g_base)
        except (ConformalSingularityError, la.LinAlgError):
            return False, -float("inf"), np.zeros(0, dtype=np.float64)

        direction = g_sqrt_inv @ v_start
        samples_v = np.zeros(n_samples, dtype=np.float64)
        margin = float("inf")
        for i, sigma in enumerate(np.linspace(-1.0, 1.0, n_samples)):
            q_test = q_start + sigma * direction
            try:
                v_val = self._as_finite_scalar(potential_v_fn(q_test), "V(q_test)")
            except Exception:
                v_val = float("inf")
            samples_v[i] = v_val
            margin = min(margin, float(initial_h0) - v_val)
        return (margin >= 0.0), float(margin), samples_v

    def _phase1_local_lipschitz_v(
        self,
        q_start: NDArray[np.float64],
        potential_v_fn: PotentialField,
        radius: float = _LOCAL_LIPSCHITZ_RADIUS,
    ) -> float:
        r"""
        Lipschitz local de \(V\) en \(B(q_s,r)\) por stencil 4º orden
        \(\bigl[-f(q+2he_i)+8f(q+he_i)-8f(q-he_i)+f(q-2he_i)\bigr]/(12h)\).
        """
        d = int(q_start.shape[0])
        scale = max(1.0, float(la.norm(q_start)))
        h = max(radius * 1.0e-3, math.sqrt(math.sqrt(_MACHINE_EPS)) * scale)
        max_slope = 0.0
        for i in range(d):
            e_i = np.zeros(d, dtype=np.float64)
            e_i[i] = 1.0
            try:
                f_p2 = self._as_finite_scalar(potential_v_fn(q_start + 2.0 * h * e_i), "V+")
                f_p1 = self._as_finite_scalar(potential_v_fn(q_start + h * e_i), "V+")
                f_m1 = self._as_finite_scalar(potential_v_fn(q_start - h * e_i), "V-")
                f_m2 = self._as_finite_scalar(potential_v_fn(q_start - 2.0 * h * e_i), "V-")
                deriv = abs((-f_p2 + 8.0 * f_p1 - 8.0 * f_m1 + f_m2) / (12.0 * h))
                max_slope = max(max_slope, deriv)
            except Exception:
                continue
        return float(max_slope)

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
            samples, initial_h0)
        q_arr, v_arr, kinetic, ham_res = self._phase1_validate_initial_kinematics(
            metric, initial_h0, q_start, v_start, potential_at_start)
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
        return self._phase1_observe_energy_certificate(
            g_base, potential_v, initial_h0,
            q_start=q_start, v_start=v_start, potential_at_start=potential_at_start)

    # ── CIERRE FASE 1 → FASE 2 ────────────────────────────────────────────────
    def emit_phase1_envelope(
        self,
        g_base: NDArray[np.float64],
        q_start: NDArray[np.float64],
        v_start: NDArray[np.float64],
        potential_v_fn: PotentialField,
        initial_h0: float,
    ) -> Phase1Envelope:
        r"""
        **Cierre formal de la FASE 1 / contrato de apertura de la FASE 2.**

        Audita el pozo (Sylvester + Cholesky + Hill + Lipschitz) y emite
        `Phase1Envelope`. El hash se calcula sobre G ya hermitizada para
        que `from_phase1_envelope` verifique la misma cadena.
        """
        t0 = time.perf_counter_ns()
        self._assert_callable_potential(potential_v_fn)
        g = np.asarray(g_base, dtype=np.float64)
        q0 = np.asarray(q_start, dtype=np.float64)
        v0 = np.asarray(v_start, dtype=np.float64)

        val_start = self._as_finite_scalar(potential_v_fn(q0), "V(q_start)")
        dilation = self._phase1_observe_energy_certificate(
            g_base=g,
            potential_v=np.array([val_start], dtype=np.float64),
            initial_h0=float(initial_h0),
            q_start=q0, v_start=v0, potential_at_start=val_start,
        )
        hill_ok, hill_margin, hill_samples = self._phase1_hill_accessibility_ball(
            g_base=dilation.g_base, q_start=dilation.q_start, v_start=dilation.v_start,
            potential_v_fn=potential_v_fn, initial_h0=float(initial_h0))
        lip = self._phase1_local_lipschitz_v(dilation.q_start, potential_v_fn)

        chain = _hash_state(
            dilation.g_base, dilation.q_start, dilation.v_start,
            np.array([
                dilation.initial_h0,
                float(dilation.potential_v[0]) if dilation.potential_v.size else 0.0,
                float(hill_margin),
                float(lip),
            ]),
        )
        return Phase1Envelope(
            dilation=dilation,
            hill_accessibility_ok=bool(hill_ok),
            hill_accessibility_margin=float(hill_margin),
            local_lipschitz_v=float(lip),
            hill_potential_samples=self._freeze_array(hill_samples),
            origin_timestamp_ns=float(time.perf_counter_ns() - t0),
            chain_hash=chain,
        )


# ══════════════════════════════════════════════════════════════════════════════════
# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║  F A S E   2 :   O R I E N T                                                 ║
# ║  Sintonía de la métrica de Fermat y embedding Cayley-Darboux en Sp(2d).      ║
# ║  Apertura = `from_phase1_envelope` (continuación del cierre de FASE 1).      ║
# ║  Cierre   = `emit_phase2_envelope` → `Phase2Envelope`.                       ║
# ╚════════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════════

class Phase2_ConformalKoszulSuturator(Phase1_PotentialWellInquirer):
    r"""
    FASE 2 (Orient): métrica de Fermat, Koszul conforme y Cayley-Darboux.
    """

    # ── APERTURA FASE 2  (continuación contractual del cierre de FASE 1) ──────
    @classmethod
    def from_phase1_envelope(
        cls, env: Phase1Envelope,
    ) -> "Phase2_ConformalKoszulSuturator":
        r"""
        **Apertura formal de la FASE 2.**

        Continúa `emit_phase1_envelope`: valida la cadena de hashes y
        autoriza la sintonía conforme.
        """
        dlt = env.dilation
        expected = _hash_state(
            dlt.g_base, dlt.q_start, dlt.v_start,
            np.array([
                dlt.initial_h0,
                float(dlt.potential_v[0]) if dlt.potential_v.size else 0.0,
                float(env.hill_accessibility_margin),
                float(env.local_lipschitz_v),
            ]),
        )
        if expected != env.chain_hash:
            raise EnvelopeChainViolationError(
                f"Cadena FASE 1 corrupta: {expected} ≠ {env.chain_hash}.")
        inst = cls()
        inst._phase1_envelope = env  # type: ignore[attr-defined]
        return inst

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
                f"g_base shape={dilation.g_base.shape} != {(d, d)}.")
        if dilation.cholesky_factor.shape != (d, d):
            raise ConformalSingularityError("cholesky_factor incoherente con g_base.")
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
        r"""Índice de Fermat: \(n^2=[2(H_0-V)]^{-1}\), \(n=\sqrt{n^2}\)."""
        self._assert_callable_potential(potential_v_fn)
        val_v = self._as_finite_scalar(potential_v_fn(q), "potential_v_fn(q)")
        gap = float(initial_h0) - val_v
        if gap <= _ENERGY_GAP_FLOOR:
            if not clamp:
                raise EnergyWellColapsoError(
                    f"Colapso de barrera en q de evaluación: H0−V(q)={gap:.3e}.")
            gap = _ENERGY_GAP_FLOOR
        n_sq = 1.0 / (2.0 * gap)
        n_factor = math.sqrt(n_sq)
        if not math.isfinite(n_factor) or n_sq <= 0.0:
            raise GeodesicIntegrationError(
                f"Índice de refracción no físico: n²={n_sq}, n={n_factor}.")
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
        return n_sq * np.asarray(
            g_base, dtype=np.result_type(g_base, getattr(q, "dtype", np.float64)))

    def _phase2_assemble_jacobi_metric(
        self,
        g_base: NDArray[np.float64],
        n_sq: float,
    ) -> NDArray[np.float64]:
        conformal = n_sq * np.asarray(g_base, dtype=np.float64)
        if not np.all(np.isfinite(conformal)):
            raise ConformalSingularityError("Métrica conforme no finita (overflow de n²·G).")
        return conformal

    def _phase2_invert_conformal_cholesky(
        self,
        dilation: ConformalPotentialDilation,
        n_sq: float,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        chol = np.asarray(dilation.cholesky_factor, dtype=np.float64)
        try:
            g_inv = self._cholesky_inverse(chol)
        except la.LinAlgError as exc:
            raise ConformalSingularityError("Inversión Cholesky fallida.") from exc
        g_conformal_inv = (1.0 / n_sq) * g_inv
        return g_inv, g_conformal_inv

    def _phase2_wilkinson_condition(
        self, g_base: NDArray[np.float64],
    ) -> float:
        evals = la.eigvalsh(g_base)
        lam_min = float(evals[0])
        lam_max = float(evals[-1])
        if lam_min <= 0.0:
            raise ConformalSingularityError(
                f"Espectro de G no SPD: λ_min={lam_min:.3e}.")
        cond = lam_max / lam_min
        if cond > _CONDITION_NUMBER_MAX:
            raise ConformalSingularityError(
                f"Inestabilidad de Wilkinson: κ(G) = {cond:.3e}.")
        return float(cond)

    def _phase2_compute_cayley_darboux_embedding(
        self,
        g_conformal: NDArray[np.float64],
        g_base: NDArray[np.float64],
        g_inv: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float, bool, float, float]:
        r"""
        Inmersión simpléctica de Cayley-Darboux sobre \(T^*Q\cong\mathbb{R}^{2d}\).

        (i) \(K=(Ĝ-I)(Ĝ+I)^{-1}\) (solve triangular-equivalente, G simétrica),
            \(K_{\mathrm{full}}=\mathrm{diag}(K,-K)\). Simétrica ⇒ \(K_{\mathrm{full}}\in\mathfrak{sp}(2d)\).
        (ii) \(M=(I-K_{\mathrm{full}})^{-1}(I+K_{\mathrm{full}})\in\mathrm{Sp}(2d)\).
        (iii) \(J_G=\begin{pmatrix}0&-G^{-1}\\G&0\end{pmatrix}\), \(J_G^2=-I\), \(J_G\in\mathrm{Sp}(2d)\).

        Retorna `(M, rel_residual, is_valid, sp_defect, J²+I residual)`.
        """
        d = int(g_conformal.shape[0])
        i_d = np.eye(d, dtype=np.float64)
        omega = self._canonical_omega(d)
        try:
            k_block = la.solve(g_conformal + i_d, g_conformal - i_d, assume_a="sym")
            k_block = 0.5 * (k_block + k_block.T)
            z = np.zeros((d, d), dtype=np.float64)
            k_full = np.block([[k_block, z], [z, -k_block]])
            sp_defect = float(la.norm(k_full.T @ omega + omega @ k_full, "fro"))

            i_2d = np.eye(2 * d, dtype=np.float64)
            m_sympl = la.solve(i_2d - k_full, i_2d + k_full)
            diff = m_sympl.T @ omega @ m_sympl - omega
            norm_m = float(la.norm(m_sympl, "fro"))
            rel_res = (float(la.norm(diff, "fro") / (norm_m ** 2))
                       if norm_m > 0.0 else float("inf"))

            j_g = np.block([[z, -g_inv], [g_base, z]])
            j_sq_res = float(la.norm(j_g @ j_g + i_2d, "fro"))
            j_sp_res = float(la.norm(j_g.T @ omega @ j_g - omega, "fro"))

            is_valid = bool(
                rel_res <= _SYMPLECTIC_RESIDUAL_TOL
                and sp_defect <= _SYMPLECTIC_RESIDUAL_TOL
                and j_sq_res <= 1.0e-8
                and j_sp_res <= 1.0e-8
            )
            return m_sympl, rel_res, is_valid, sp_defect, j_sq_res
        except la.LinAlgError:
            z2 = np.zeros((2 * d, 2 * d), dtype=np.float64)
            return z2, float("inf"), False, float("inf"), float("inf")

    def _phase2_estimate_potential_gradient(
        self,
        q: NDArray[np.float64],
        potential_v_fn: PotentialField,
    ) -> Tuple[NDArray[np.float64], str]:
        r"""
        \(\nabla V\) por CSMD (\(h=10^{-20}\)) con fallback a stencil 4º orden.
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
                grad[k] = float(np.imag(val)) / _CSMD_PERTURBATION
            if np.all(np.isfinite(grad)):
                return grad, "csmd"
        except Exception:
            pass

        scale = max(1.0, float(la.norm(q)))
        step = math.sqrt(math.sqrt(_MACHINE_EPS)) * scale
        for k in range(d):
            e_k = np.zeros(d, dtype=np.float64)
            e_k[k] = 1.0
            f_p2 = self._as_finite_scalar(potential_v_fn(q + 2.0 * step * e_k), "V+2")
            f_p1 = self._as_finite_scalar(potential_v_fn(q + step * e_k), "V+1")
            f_m1 = self._as_finite_scalar(potential_v_fn(q - step * e_k), "V-1")
            f_m2 = self._as_finite_scalar(potential_v_fn(q - 2.0 * step * e_k), "V-2")
            grad[k] = (-f_p2 + 8.0 * f_p1 - 8.0 * f_m1 + f_m2) / (12.0 * step)
        if not np.all(np.isfinite(grad)):
            raise GeodesicIntegrationError("∇V no finito en el punto de evaluación.")
        return grad, "central-4"

    def _phase2_hessian_v(
        self,
        q: NDArray[np.float64],
        potential_v_fn: PotentialField,
    ) -> NDArray[np.float64]:
        r"""Hessiano simétrico de \(V\) por diferencias mixtas de 2º orden."""
        d = int(q.shape[0])
        scale = max(1.0, float(la.norm(q)))
        h = math.sqrt(_MACHINE_EPS) * scale
        hess = np.zeros((d, d), dtype=np.float64)
        v0 = self._as_finite_scalar(potential_v_fn(q), "V(q)")

        def _v(delta: NDArray[np.float64]) -> float:
            return self._as_finite_scalar(potential_v_fn(q + delta), "V(q+δ)")

        for i in range(d):
            e_i = np.zeros(d, dtype=np.float64)
            e_i[i] = h
            hess[i, i] = (_v(e_i) - 2.0 * v0 + _v(-e_i)) / (h * h)
            for j in range(i + 1, d):
                e_j = np.zeros(d, dtype=np.float64)
                e_j[j] = h
                mixed = (_v(e_i + e_j) - _v(e_i - e_j) - _v(-e_i + e_j) + _v(-e_i - e_j)) / (4.0 * h * h)
                hess[i, j] = hess[j, i] = mixed
        return 0.5 * (hess + hess.T)

    def _phase2_conformal_christoffel(
        self,
        g_base: NDArray[np.float64],
        g_inv: NDArray[np.float64],
        grad_v: NDArray[np.float64],
        gap: float,
    ) -> NDArray[np.float64]:
        r"""
        Christoffel de Fermat para \(\tilde g = n^2 G\), \(n^2=[2(H_0-V)]^{-1}\),
        \(\varphi=\log n\), \(\nabla\varphi=\nabla V/(2\,\mathrm{gap})\):

        \[\tilde\Gamma^i_{jk}=\delta^i_j\partial_k\varphi+\delta^i_k\partial_j\varphi
          -G_{jk}G^{il}\partial_l\varphi.\]
        """
        d = g_base.shape[0]
        if gap <= 0.0:
            raise EnergyWellColapsoError("Gap no positivo al formar Christoffel conforme.")
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
        self, gamma: NDArray[np.float64],
    ) -> float:
        torsion = gamma - np.swapaxes(gamma, 1, 2)
        residual = float(np.max(np.abs(torsion))) if torsion.size else 0.0
        if residual > max(_TORSION_ABS_TOL, 1.0e3 * _MACHINE_EPS):
            raise KoszulChecksumError(f"Conexión con torsión no nula: τ={residual:.3e}.")
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
        vec = (np.ones(d) / math.sqrt(d) if probe is None
               else np.asarray(probe, dtype=np.float64))
        dphi = grad_v / (2.0 * gap)
        contracted = np.einsum("ijk,j,k->i", gamma, vec, vec)
        closed = (2.0 * float(dphi @ vec) * vec
                  - float(vec @ g_base @ vec) * (g_inv @ dphi))
        defect = float(la.norm(contracted - closed))
        scale = max(1.0, float(la.norm(closed)))
        if defect > _CHECKSUM_REL_TOL * scale:
            raise KoszulChecksumError(
                f"Checksum Γ(v,v)↔a_cerrada corrupto: ‖δ‖={defect:.3e}.")
        return defect

    def sintonizar_metrica_conforme(
        self,
        dilation: ConformalPotentialDilation,
        q_eval: NDArray[np.float64],
        potential_v_fn: Callable[[NDArray[np.complex128] | NDArray[np.float64]], Any],
    ) -> ConformalGeometryBundle:
        dilation, q, _d = self._phase2_consume_phase1_certificate(dilation, q_eval)
        self._assert_callable_potential(potential_v_fn)

        g_base = np.asarray(dilation.g_base, dtype=np.float64)
        initial_h0 = float(dilation.initial_h0)

        n_factor, n_sq, gap = self._phase2_conformal_factor_at_q(
            q, potential_v_fn, initial_h0, clamp=False)
        g_conformal = self._phase2_assemble_jacobi_metric(g_base, n_sq)
        g_inv, g_conformal_inv = self._phase2_invert_conformal_cholesky(dilation, n_sq)
        cond = self._phase2_wilkinson_condition(g_base)
        m_sympl, res_sympl, is_sympl, sp_defect, j_res = (
            self._phase2_compute_cayley_darboux_embedding(g_conformal, g_base, g_inv))

        grad_v, grad_method = self._phase2_estimate_potential_gradient(q, potential_v_fn)
        gamma = self._phase2_conformal_christoffel(g_base, g_inv, grad_v, gap)
        torsion = self._phase2_koszul_torsion_freeness(gamma)
        self._phase2_christoffel_acceleration_identity(gamma, g_base, g_inv, grad_v, gap)

        return ConformalGeometryBundle(
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
            symplectic_cayley_matrix=self._freeze_array(m_sympl),
            symplectic_residual=res_sympl,
            symplectic_defect=sp_defect,
            is_symplectic_valid=is_sympl,
            almost_complex_residual=float(j_res),
            phase1=dilation,
        )

    # ── CIERRE FASE 2 → FASE 3 ────────────────────────────────────────────────
    def emit_phase2_envelope(
        self,
        phase1_env: Phase1Envelope,
        potential_v_fn: Callable[[NDArray[np.complex128] | NDArray[np.float64]], Any],
    ) -> Phase2Envelope:
        r"""
        **Cierre formal de la FASE 2 / contrato de apertura de la FASE 3.**
        """
        t0 = time.perf_counter_ns()
        dlt = phase1_env.dilation
        bundle = self.sintonizar_metrica_conforme(
            dilation=dlt, q_eval=dlt.q_start, potential_v_fn=potential_v_fn)
        chain = _hash_state(
            bundle.g_base, bundle.g_conformal, bundle.christoffel_symbols,
            np.array([
                bundle.symplectic_residual, bundle.symplectic_defect,
                bundle.torsion_residual, bundle.wilkinson_condition,
                bundle.almost_complex_residual,
            ]),
        )
        return Phase2Envelope(
            phase1=phase1_env, bundle=bundle,
            origin_timestamp_ns=float(time.perf_counter_ns() - t0),
            chain_hash=chain,
        )


# ══════════════════════════════════════════════════════════════════════════════════
# ╔════════════════════════════════════════════════════════════════════════════════╗
# ║  F A S E   3 :   D E C I D E   &   A C T                                      ║
# ║  RK4 + Poincaré-Lindstedt + KAM/Bruno + Melnikov + sección + Floquet + RSI³. ║
# ║  Apertura = `from_phase2_envelope` (continuación del cierre de FASE 2).      ║
# ║  Cierre   = `emit_governance_state` → `BrachistochronePhysicalState`.        ║
# ╚════════════════════════════════════════════════════════════════════════════════╝
# ══════════════════════════════════════════════════════════════════════════════════

class Phase3_FermatGeodesicSolver(Phase2_ConformalKoszulSuturator):
    r"""
    FASE 3 (Decide & Act): geodésica de Fermat, Poincaré y RSI³ Picard.
    """

    # ── APERTURA FASE 3  (continuación contractual del cierre de FASE 2) ──────
    @classmethod
    def from_phase2_envelope(
        cls, env: Phase2Envelope,
    ) -> "Phase3_FermatGeodesicSolver":
        r"""
        **Apertura formal de la FASE 3.**

        Continúa `emit_phase2_envelope`: valida la cadena y autoriza RK4.
        """
        b = env.bundle
        expected = _hash_state(
            b.g_base, b.g_conformal, b.christoffel_symbols,
            np.array([
                b.symplectic_residual, b.symplectic_defect,
                b.torsion_residual, b.wilkinson_condition,
                b.almost_complex_residual,
            ]),
        )
        if expected != env.chain_hash:
            raise EnvelopeChainViolationError(
                f"Cadena FASE 2 corrupta: {expected} ≠ {env.chain_hash}.")
        inst = cls()
        inst._phase2_envelope = env  # type: ignore[attr-defined]
        return inst

    def _phase3_consume_phase2_certificate(
        self,
        bundle: ConformalGeometryBundle,
        q_start: NDArray[np.float64],
        v_start: NDArray[np.float64],
    ) -> Tuple[ConformalGeometryBundle, NDArray[np.float64], NDArray[np.float64], int]:
        if not isinstance(bundle, ConformalGeometryBundle):
            raise TypeError("bundle debe ser ConformalGeometryBundle")
        d = int(bundle.g_base.shape[0])
        if bundle.g_conformal.shape != (d, d) or bundle.g_conformal_inv.shape != (d, d):
            raise ConformalSingularityError(
                "FASE3.0: tensores conformes incoherentes con g_base.")
        if bundle.christoffel_symbols.shape != (d, d, d):
            raise ConformalSingularityError(
                f"FASE3.0: Christoffel shape={bundle.christoffel_symbols.shape} "
                f"!= {(d, d, d)}.")
        q = np.asarray(q_start, dtype=np.float64)
        v = np.asarray(v_start, dtype=np.float64)
        self._assert_vector(q, d, "q_start")
        self._assert_vector(v, d, "v_start")
        return bundle, q, v, d

    def _phase3_resolve_base_inverse(
        self, bundle: ConformalGeometryBundle,
    ) -> NDArray[np.float64]:
        d = bundle.g_base.shape[0]
        cached = np.asarray(bundle.g_base_inv, dtype=np.float64)
        if cached.shape == (d, d) and np.all(np.isfinite(cached)) and cached.size > 0:
            return cached
        try:
            chol = la.cholesky(np.asarray(bundle.g_base, dtype=np.float64), lower=True)
            return self._cholesky_inverse(chol)
        except la.LinAlgError as exc:
            raise ConformalSingularityError(
                "Cholesky fallido en la fase de integración.") from exc

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
                f"Cascarón vacío en el arranque: H0−V(q0)={gap:.3e}.")
        kinetic = 0.5 * float(v @ g_base @ v)
        residual = abs(kinetic - gap)
        if not project_on_shell:
            return np.asarray(v, dtype=np.float64), residual, False

        target_quad = 2.0 * gap
        current_quad = float(v @ g_base @ v)
        if current_quad > _ENERGY_GAP_FLOOR:
            scaled = v * math.sqrt(target_quad / current_quad)
            return np.asarray(scaled, dtype=np.float64), residual, True

        grad_v, _ = self._phase2_estimate_potential_gradient(q, potential_v_fn)
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
            q, potential_v_fn, initial_h0, clamp=False)
        grad_v, _ = self._phase2_estimate_potential_gradient(q, potential_v_fn)
        dphi = grad_v / (2.0 * gap)
        v_quad = float(v @ g_base @ v)
        accel = -2.0 * float(dphi @ v) * v + v_quad * (g_inv @ dphi)
        if not np.all(np.isfinite(accel)):
            raise GeodesicIntegrationError(f"Aceleración geodésica no finita en q={q}.")
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
                q, v, g_base, g_base_inv, potential_v_fn, initial_h0)
        except (EnergyWellColapsoError, ConformalSingularityError, ArithmeticError) as exc:
            raise GeodesicIntegrationError(
                f"Colisión con singularidad en q={q}: {exc}") from exc
        return np.concatenate([v, accel])

    def _phase3_fermat_energy(
        self,
        q: NDArray[np.float64],
        v: NDArray[np.float64],
        g_base: NDArray[np.float64],
        potential_v_fn: PotentialField,
        initial_h0: float,
    ) -> float:
        _n, n_sq, _ = self._phase2_conformal_factor_at_q(
            q, potential_v_fn, initial_h0, clamp=True)
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
        k1 = self._evaluate_geodesic_rhs(state, g_base, g_inv, potential_v_fn, initial_h0)
        k2 = self._evaluate_geodesic_rhs(
            state + 0.5 * dt * k1, g_base, g_inv, potential_v_fn, initial_h0)
        k3 = self._evaluate_geodesic_rhs(
            state + 0.5 * dt * k2, g_base, g_inv, potential_v_fn, initial_h0)
        k4 = self._evaluate_geodesic_rhs(
            state + dt * k3, g_base, g_inv, potential_v_fn, initial_h0)
        nxt = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if not np.all(np.isfinite(nxt)):
            raise GeodesicIntegrationError("Estado RK4 no finito.")
        return nxt

    def _phase3_flow_jacobian(
        self,
        state: NDArray[np.float64],
        g_base: NDArray[np.float64],
        g_inv: NDArray[np.float64],
        potential_v_fn: PotentialField,
        initial_h0: float,
    ) -> Optional[NDArray[np.float64]]:
        r"""Jacobiano \(DF\) del campo \((v,a)\) por diferencias centradas 2º orden."""
        n = int(state.size)
        scale = max(1.0, float(la.norm(state)))
        h = math.sqrt(_MACHINE_EPS) * scale
        jac = np.zeros((n, n), dtype=np.float64)
        try:
            for i in range(n):
                e_i = np.zeros(n, dtype=np.float64)
                e_i[i] = h
                fp = self._evaluate_geodesic_rhs(
                    state + e_i, g_base, g_inv, potential_v_fn, initial_h0)
                fm = self._evaluate_geodesic_rhs(
                    state - e_i, g_base, g_inv, potential_v_fn, initial_h0)
                jac[:, i] = (fp - fm) / (2.0 * h)
        except GeodesicIntegrationError:
            return None
        if not np.all(np.isfinite(jac)):
            return None
        return jac

    # ══════════════════════════════════════════════════════════════════════════
    # MECÁNICA CELESTE DE POINCARÉ
    # ══════════════════════════════════════════════════════════════════════════

    def solve_poincare_lindstedt_homological_eq(
        self,
        frequencies: NDArray[np.float64],
        perturbation_modes: NDArray[np.float64],
        max_order: int = _KAM_MAX_ORDER,
    ) -> Tuple[NDArray[np.float64], float]:
        r"""
        Ecuación homológica \(\{H_0,W\}=H_1-\langle H_1\rangle\):
        \(W_k=H_{1,k}/(\omega\cdot k)\) para \(k\in\mathcal{K}\setminus\{0\}\).

        Si `perturbation_modes` es una serie temporal (len ≫ d), se toma su
        FFT 1D y los divisores \(k\,\bar\omega\). En caso contrario se recorre
        el retículo de KAM y se asignan modos cíclicamente.
        Conserva la firma pública (frecuencias, modos) → (W, residual).
        """
        freqs = np.asarray(frequencies, dtype=np.float64).ravel()
        modes = np.asarray(perturbation_modes, dtype=np.float64).ravel()
        w_amp = np.zeros_like(modes)
        max_residual = 0.0
        if freqs.size == 0 or modes.size == 0:
            return w_amp, 0.0

        omega_bar = float(np.mean(np.abs(freqs))) + _HOM_DIVISOR_FLOOR
        if modes.size >= 8:
            spec = np.fft.rfft(modes)
            for k_idx, coeff in enumerate(spec):
                if k_idx == 0:
                    w_amp[0] = 0.0
                    continue
                divisor = float(k_idx) * omega_bar
                h1 = float(np.real(coeff) / max(modes.size, 1))
                if k_idx < w_amp.size:
                    w_amp[k_idx] = h1 / divisor if abs(divisor) > _HOM_DIVISOR_FLOOR else 0.0
                    max_residual = max(max_residual, abs(divisor * w_amp[k_idx] - h1))
            return w_amp, float(max_residual)

        k_list = list(_iter_kam_wavevectors(int(freqs.size), max_order))
        if not k_list:
            return w_amp, 0.0
        for idx, mode in enumerate(modes):
            k_vec = k_list[idx % len(k_list)]
            if k_vec.shape[0] != freqs.shape[0]:
                k_use = np.zeros_like(freqs)
                n_copy = min(k_vec.shape[0], freqs.shape[0])
                k_use[:n_copy] = k_vec[:n_copy]
            else:
                k_use = k_vec
            divisor = float(np.dot(freqs, k_use))
            if abs(divisor) < _HOM_DIVISOR_FLOOR:
                w_amp[idx] = 0.0
                max_residual = max(max_residual, abs(float(mode)))
            else:
                w_amp[idx] = float(mode) / divisor
                max_residual = max(max_residual, abs(divisor * w_amp[idx] - float(mode)))
        return w_amp, float(max_residual)

    def _phase3_lindstedt_certificate(
        self,
        frequencies: NDArray[np.float64],
        h1_series: NDArray[np.float64],
    ) -> LindstedtCertificate:
        _w, residual = self.solve_poincare_lindstedt_homological_eq(frequencies, h1_series)
        n_res = 0
        div_min = float("inf")
        omega_bar = float(np.mean(np.abs(frequencies))) + _HOM_DIVISOR_FLOOR
        spec = np.fft.rfft(np.asarray(h1_series, dtype=np.float64).ravel()) if h1_series.size else np.zeros(1)
        for k_idx in range(1, spec.size):
            div = abs(float(k_idx) * omega_bar)
            div_min = min(div_min, div)
            if div < _HOM_DIVISOR_FLOOR:
                n_res += 1
        return LindstedtCertificate(
            n_modes=int(spec.size),
            n_resonant=int(n_res),
            residual_max=float(residual),
            divisor_min=float(div_min if math.isfinite(div_min) else 0.0),
        )

    def check_kam_bruno_diophantine_condition(
        self,
        frequencies: NDArray[np.float64],
        gamma: float = _KAM_GAMMA,
        max_order: int = _KAM_MAX_ORDER,
    ) -> bool:
        r"""
        \(|\omega\cdot k|\ge\gamma/\|k\|_1^\tau\) para todo \(k\in\mathcal{K}\setminus\{0\}\),
        \(\tau=d+1\). El retículo \(\mathcal{K}\) es el cubo si \(d\le 4\) y de
        soporte \(\le 2\) si \(d>4\).
        """
        freqs = np.asarray(frequencies, dtype=np.float64).ravel()
        d = int(freqs.shape[0])
        if d == 0:
            return False
        tau = float(d) + _KAM_TAU_OFFSET
        for k_vec in _iter_kam_wavevectors(d, max_order):
            k_norm_1 = float(np.sum(np.abs(k_vec)))
            if k_norm_1 <= 0.0:
                continue
            dot = abs(float(np.dot(freqs, k_vec)))
            bound = gamma / (k_norm_1 ** tau)
            if dot < bound:
                return False
        return True

    def _phase3_kam_worst_ratio(
        self,
        frequencies: NDArray[np.float64],
        gamma: float = _KAM_GAMMA,
        max_order: int = _KAM_MAX_ORDER,
    ) -> float:
        r"""\(\min_k |\omega\cdot k|\cdot\|k\|_1^\tau/\gamma\)."""
        freqs = np.asarray(frequencies, dtype=np.float64).ravel()
        d = int(freqs.shape[0])
        if d == 0:
            return 0.0
        tau = float(d) + _KAM_TAU_OFFSET
        worst = float("inf")
        for k_vec in _iter_kam_wavevectors(d, max_order):
            k_norm_1 = float(np.sum(np.abs(k_vec)))
            if k_norm_1 <= 0.0:
                continue
            dot = abs(float(np.dot(freqs, k_vec)))
            ratio = dot * (k_norm_1 ** tau) / max(gamma, 1e-15)
            worst = min(worst, ratio)
        return float(worst) if math.isfinite(worst) else 0.0

    def compute_melnikov_homoclinic_integral(
        self,
        trajectory: NDArray[np.float64],
        velocities: NDArray[np.float64],
        dt: float,
    ) -> float:
        r"""
        Legacy (firma preservada). Usa el proxy \(\int\sin(q\cdot v)\,dt\).
        El cálculo físico vive en `_phase3_melnikov_from_trajectory`.
        """
        if trajectory.shape[0] < 2:
            return 0.0
        acc = 0.0
        n = min(trajectory.shape[0], velocities.shape[0])
        for i in range(n):
            cross_term = float(np.sum(trajectory[i] * velocities[i]))
            acc += math.sin(cross_term) * dt
        return float(acc)

    def _phase3_melnikov_from_trajectory(
        self,
        trajectory: NDArray[np.float64],
        velocities: NDArray[np.float64],
        g_base: NDArray[np.float64],
        potential_v_fn: PotentialField,
        dt: float,
    ) -> Tuple[float, float]:
        r"""
        Integral de Melnikov-Poincaré con el corchete de Poisson canónico:

        \[
          \{T,V\}=-v\cdot\nabla V,\qquad
          M=\int\{T,V\}\,dt=-\int v(t)\cdot\nabla V\bigl(q(t)\bigr)\,dt.
        \]

        (El peso \(Gv\) es \(p\cdot\nabla V\), incorrecto salvo \(G=I\).)
        \(g_base\) se acepta por simetría de firma / diagnósticos futuros.
        """
        del g_base
        n = min(int(trajectory.shape[0]), int(velocities.shape[0]))
        if n < 2:
            return 0.0, 0.0
        acc = 0.0
        max_abs = 0.0
        for i in range(n):
            try:
                grad_v, _ = self._phase2_estimate_potential_gradient(
                    trajectory[i], potential_v_fn)
            except Exception:
                continue
            integrand = -float(np.dot(velocities[i], grad_v))
            max_abs = max(max_abs, abs(integrand))
            acc += integrand * dt
        return float(acc), float(max_abs)

    def _phase3_poincare_section(
        self,
        trajectory: NDArray[np.float64],
        velocities: NDArray[np.float64],
        *,
        section_point: Optional[NDArray[np.float64]] = None,
        section_normal: Optional[NDArray[np.float64]] = None,
        dt: float = 0.005,
    ) -> PoincareSectionCertificate:
        r"""
        Sección transversal \(\Sigma=\{n\cdot(q-q_\Sigma)=0\}\) con interpolación
        lineal del cruce y filtro de orientación \(n\cdot v>0\).
        """
        traj = np.asarray(trajectory, dtype=np.float64)
        vel = np.asarray(velocities, dtype=np.float64)
        if traj.ndim != 2 or traj.shape[0] < 3:
            return PoincareSectionCertificate(0, 0.0, True, ())

        q0 = traj[0]
        q_sigma = q0 if section_point is None else np.asarray(section_point, dtype=np.float64)
        if section_normal is None:
            if vel.shape[0] > 0 and float(la.norm(vel[0])) > 1e-15:
                nrm = vel[0] / float(la.norm(vel[0]))
            else:
                delta = traj[1] - q0
                nrm = delta / max(float(la.norm(delta)), 1e-15)
        else:
            nrm = np.asarray(section_normal, dtype=np.float64)
            nrm = nrm / max(float(la.norm(nrm)), 1e-15)

        signs = np.array([float(nrm @ (q - q_sigma)) for q in traj], dtype=np.float64)
        crossings_q: List[NDArray[np.float64]] = []
        crossings_t: List[float] = []
        for k in range(len(signs) - 1):
            if signs[k] == 0.0:
                continue
            if signs[k] * signs[k + 1] >= 0.0:
                continue
            denom = signs[k] - signs[k + 1]
            alpha = signs[k] / denom if abs(denom) > _MACHINE_EPS else 0.0
            if vel.shape[0] > k and float(nrm @ vel[k]) <= 0.0:
                continue
            crossings_q.append(traj[k] + alpha * (traj[k + 1] - traj[k]))
            crossings_t.append((k + alpha) * dt)

        n_cross = len(crossings_q)
        if n_cross == 0:
            return PoincareSectionCertificate(0, 0.0, True, ())

        coords = np.array([float(q[0]) for q in crossings_q], dtype=np.float64)
        disp = float(np.std(coords) / max(float(np.mean(np.abs(coords))), 1e-15)) if n_cross > 1 else 0.0
        first_return = 0.0
        close = 0
        residual = 0.0
        if n_cross >= 2:
            first_return = float(crossings_t[1] - crossings_t[0])
            residual = float(la.norm(crossings_q[1] - crossings_q[0]))
            scale = max(float(la.norm(crossings_q[0])), 1e-15)
            if residual / scale <= _CLOSE_RETURN_REL_TOL:
                close += 1
            for a, b in zip(crossings_q, crossings_q[1:]):
                rel = float(la.norm(b - a)) / max(float(la.norm(a)), 1e-15)
                if rel <= _CLOSE_RETURN_REL_TOL:
                    close += 1
        return PoincareSectionCertificate(
            n_crossings=n_cross,
            return_dispersion=disp,
            is_regular=bool(disp < _POINCARE_DISP_REGULAR),
            returns=tuple(float(x) for x in coords),
            first_return_time=first_return,
            close_returns=close,
            return_residual=residual,
        )

    def _phase3_floquet_from_monodromy(
        self,
        monodromy: NDArray[np.float64],
        period: float,
        lyapunov: NDArray[np.float64],
    ) -> FloquetCertificate:
        r"""Multiplicadores \(\mu=\mathrm{eig}(W)\) o \(\mathrm{e}^{\lambda T}\) si hay QR."""
        t_per = max(float(period), 1e-15)
        try:
            mu = la.eigvals(np.asarray(monodromy, dtype=np.complex128))
            moduli = np.abs(mu)
            unit_dev = float(np.max(np.abs(moduli - 1.0)))
            dissipative = bool(np.max(moduli) <= 1.0 + _FLOQUET_UNIT_TOL)
            exponents = np.log(np.maximum(moduli, 1e-30)) / t_per
            rho = float(np.max(moduli))
        except Exception:
            mu = np.exp(np.asarray(lyapunov, dtype=np.complex128) * t_per)
            moduli = np.abs(mu)
            unit_dev = float(np.max(np.abs(moduli - 1.0))) if moduli.size else 0.0
            dissipative = bool(np.max(moduli) <= 1.0 + _FLOQUET_UNIT_TOL) if moduli.size else True
            exponents = np.asarray(lyapunov, dtype=np.float64)
            rho = float(np.max(moduli)) if moduli.size else 1.0
        lyap_max = float(np.max(lyapunov)) if lyapunov.size else float(np.max(exponents)) if exponents.size else 0.0
        return FloquetCertificate(
            multipliers=np.asarray(mu, dtype=np.complex128),
            max_unit_deviation=unit_dev,
            is_neutral=bool(unit_dev < _FLOQUET_UNIT_TOL),
            is_dissipative_stable=dissipative,
            exponents=np.asarray(exponents, dtype=np.float64),
            spectral_radius=rho,
            lyapunov_max=lyap_max,
        )

    def _phase3_floquet_multipliers(
        self,
        generator_matrix: NDArray[np.float64],
        period: float,
    ) -> FloquetCertificate:
        r"""Legacy: \(\mu=\mathrm{eig}(\mathrm{e}^{A T})\). Preferir monodromía variacional."""
        try:
            m_t = la.expm(np.asarray(generator_matrix, dtype=np.complex128) * period)
            mu = la.eigvals(m_t)
            moduli = np.abs(mu)
            unit_dev = float(np.max(np.abs(moduli - 1.0)))
            dissipative = bool(np.max(moduli) <= 1.0 + _FLOQUET_UNIT_TOL)
            exponents = np.log(np.maximum(moduli, 1e-30)) / max(period, 1e-15)
            rho = float(np.max(moduli))
        except Exception:
            mu = np.array([1.0 + 0j])
            unit_dev, dissipative, exponents, rho = 0.0, True, np.array([0.0]), 1.0
        return FloquetCertificate(
            multipliers=mu,
            max_unit_deviation=unit_dev,
            is_neutral=bool(unit_dev < _FLOQUET_UNIT_TOL),
            is_dissipative_stable=dissipative,
            exponents=np.asarray(exponents, dtype=np.float64),
            spectral_radius=rho,
            lyapunov_max=float(np.max(exponents)) if np.size(exponents) else 0.0,
        )

    def _phase3_small_oscillation_frequencies(
        self,
        q: NDArray[np.float64],
        g_inv: NDArray[np.float64],
        potential_v_fn: PotentialField,
    ) -> NDArray[np.float64]:
        r"""Frecuencias elípticas \(\omega=\sqrt{\mathrm{eig}_+(G^{-1}\mathrm{Hess}\,V)}\)."""
        try:
            hess = self._phase2_hessian_v(q, potential_v_fn)
            matrix = g_inv @ hess
            evals = np.real(la.eigvals(matrix))
            omega = np.sqrt(np.maximum(evals, 0.0))
            if float(np.max(omega)) < 1e-12:
                return np.zeros(0, dtype=np.float64)
            return np.sort(omega)[::-1].astype(np.float64)
        except Exception:
            return np.zeros(0, dtype=np.float64)

    def _phase3_rsi3_audit(
        self,
        transit_time: float,
        g_base: NDArray[np.float64],
        v_start: NDArray[np.float64],
        v_end: NDArray[np.float64],
    ) -> Dict[str, float]:
        r"""
        RSI³:
          S1 Novikov \(v_{\mathrm{Krull}}(T)=-\log(T/T_{\mathrm{ref}})\ge 0.10\).
          S2 Harness: SPD de \(G\) (ya en FASE 1/2).
          S3 Model: \(d_{\mathrm{FS}}(\hat v_s,\hat v_e)\le 10^{-4}\) rad.
        """
        del g_base
        t_val = max(float(transit_time), _MACHINE_EPS)
        v_krull = -math.log(t_val / max(1.0, _MACHINE_EPS))
        v_s = np.asarray(v_start, dtype=np.complex128).ravel()
        v_e = np.asarray(v_end, dtype=np.complex128).ravel()
        n_s = float(np.linalg.norm(v_s))
        n_e = float(np.linalg.norm(v_e))
        if n_s < 1e-15 or n_e < 1e-15:
            d_fs = 0.0
        else:
            num = abs(np.vdot(v_s, v_e))
            d_fs = float(np.arccos(np.clip(num / (n_s * n_e), 0.0, 1.0)))
        return {
            "novikov_krull_valuation": float(v_krull),
            "fubini_study_distance": float(d_fs),
            "rsi3_valid": float(v_krull >= _KRULL_NOVIKOV_FLOOR and d_fs <= _FUBINI_STUDY_MAX),
        }

    # ══════════════════════════════════════════════════════════════════════════
    # INTEGRACIÓN PRINCIPAL + PICARD RSI³
    # ══════════════════════════════════════════════════════════════════════════
    def _phase3_rk4_rollout(
        self,
        bundle: ConformalGeometryBundle,
        q0: NDArray[np.float64],
        v0: NDArray[np.float64],
        potential_v_fn: PotentialField,
        t_max: float,
        dt: float,
        *,
        project_on_shell: bool,
        with_variational: bool,
    ) -> Dict[str, Any]:
        r"""Núcleo RK4 (+ monodromía Benettin opcional). No emite el DTO terminal."""
        d = int(q0.shape[0])
        g_base = np.asarray(bundle.g_base, dtype=np.float64)
        initial_h0 = float(bundle.initial_h0)
        g_inv = self._phase3_resolve_base_inverse(bundle)
        v0, on_shell_res, projected = self._phase3_project_energy_shell(
            q0, v0, g_base, potential_v_fn, initial_h0,
            project_on_shell=project_on_shell)

        steps = max(1, int(math.floor(t_max / dt + 1.0e-12)))
        if steps > _MAX_INTEGRATION_STEPS:
            raise GeodesicIntegrationError(
                f"Pasos {steps} exceden max={_MAX_INTEGRATION_STEPS}.")

        trajectory = np.zeros((steps, d), dtype=np.float64)
        velocities = np.zeros((steps, d), dtype=np.float64)
        h1_series = np.zeros(steps, dtype=np.float64)
        state = np.concatenate([q0, v0])
        energy_drift_max = 0.0
        fermat_drift_max = 0.0
        transit_time_t = 0.0
        is_globally_stable = True
        last_index = 0
        e_f_0 = self._phase3_fermat_energy(q0, v0, g_base, potential_v_fn, initial_h0)

        dim_w = 2 * d
        w_mat = np.eye(dim_w, dtype=np.float64)
        log_norms = np.zeros(dim_w, dtype=np.float64)
        n_renorm = 0

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
                    q_curr, v_curr, g_base, potential_v_fn)
                e_f = self._phase3_fermat_energy(
                    q_curr, v_curr, g_base, potential_v_fn, initial_h0)
            except (GeodesicIntegrationError, EnergyWellColapsoError, ArithmeticError):
                is_globally_stable = False
                break
            energy_drift_max = max(energy_drift_max, abs(ham - initial_h0))
            fermat_drift_max = max(fermat_drift_max, abs(e_f - e_f_0))
            h1_series[step] = val_v

            v_quad = float(v_curr @ g_base @ v_curr)
            v_norm_g = math.sqrt(max(0.0, v_quad))
            gap = initial_h0 - val_v
            if gap <= _ENERGY_GAP_FLOOR:
                is_globally_stable = False
                break
            n_factor = 1.0 / math.sqrt(2.0 * gap)
            transit_time_t += n_factor * v_norm_g * dt

            if with_variational:
                jac = self._phase3_flow_jacobian(
                    state, g_base, g_inv, potential_v_fn, initial_h0)
                if jac is not None:
                    w_mat = w_mat + dt * (jac @ w_mat)
                    col_n = np.linalg.norm(w_mat, axis=0)
                    if np.any(col_n > _BENETTIN_RENORM) or np.any(col_n < 1.0 / _BENETTIN_RENORM):
                        q_f, r_f = np.linalg.qr(w_mat)
                        log_norms += np.log(np.maximum(np.abs(np.diag(r_f)), _MACHINE_EPS))
                        w_mat = q_f
                        n_renorm += 1

            try:
                state = self._phase3_rk4_step(
                    state, dt, g_base, g_inv, potential_v_fn, initial_h0)
            except GeodesicIntegrationError:
                is_globally_stable = False
                break

        if energy_drift_max > _ENERGY_DRIFT_HARD_TOL:
            is_globally_stable = False
        if fermat_drift_max > _FERMAT_DRIFT_HARD_TOL:
            is_globally_stable = False

        used = last_index + 1
        t_aff = max(used * dt, _MACHINE_EPS)
        if n_renorm == 0:
            col_n = np.maximum(np.linalg.norm(w_mat, axis=0), _MACHINE_EPS)
            lyap = np.log(col_n) / t_aff
        else:
            lyap = log_norms / t_aff

        return {
            "trajectory": trajectory[:used],
            "velocities": velocities[:used],
            "h1_series": h1_series[:used],
            "transit_time_t": float(transit_time_t),
            "energy_drift_max": float(energy_drift_max),
            "fermat_drift_max": float(fermat_drift_max),
            "is_globally_stable": bool(is_globally_stable),
            "e_f_0": float(e_f_0),
            "on_shell_res": float(on_shell_res),
            "projected": bool(projected),
            "affine_time": float(used * dt),
            "steps": int(used),
            "v0": v0,
            "monodromy": w_mat,
            "lyapunov": lyap,
            "g_inv": g_inv,
        }

    def integrar_trayectoria_conforme(
        self,
        bundle: ConformalGeometryBundle,
        q_start: NDArray[np.float64],
        v_start: NDArray[np.float64],
        potential_v_fn: Callable[[NDArray[np.complex128] | NDArray[np.float64]], Any],
        t_max: float = 2.0,
        dt: float = 0.005,
        *,
        project_on_shell: bool = True,
    ) -> BrachistochronePhysicalState:
        r"""
        Integra la geodésica de Fermat y certifica Poincaré + RSI³.

        Automejora de nivel 3: si \(\delta H\) o \(\delta E_F\) superan la cota
        blanda, \(dt\leftarrow dt/2\) y se re-integra (Picard, máx. 3 veces).
        """
        bundle, q0, v0_in, d = self._phase3_consume_phase2_certificate(
            bundle, q_start, v_start)
        self._assert_callable_potential(potential_v_fn)
        self._assert_positive_scalar(t_max, "t_max")
        self._assert_positive_scalar(dt, "dt")

        dt_work = float(dt)
        roll: Dict[str, Any] = {}
        refinements = 0
        for r in range(_RSI3_MAX_REFINEMENTS + 1):
            roll = self._phase3_rk4_rollout(
                bundle, q0, v0_in, potential_v_fn, t_max, dt_work,
                project_on_shell=project_on_shell,
                with_variational=(r == 0 or r == _RSI3_MAX_REFINEMENTS),
            )
            refinements = r
            soft_ok = (
                float(roll["energy_drift_max"]) <= _ENERGY_DRIFT_SOFT_TOL
                and float(roll["fermat_drift_max"]) <= _FERMAT_DRIFT_SOFT_TOL
            )
            if bool(roll["is_globally_stable"]) and soft_ok:
                break
            if not bool(roll["is_globally_stable"]) and float(roll["energy_drift_max"]) == 0.0:
                break
            dt_work = max(dt_work * _RSI3_DT_DECAY, 1.0e-6)

        if refinements > 0 and "lyapunov" in roll:
            roll_final = self._phase3_rk4_rollout(
                bundle, q0, v0_in, potential_v_fn, t_max, dt_work,
                project_on_shell=project_on_shell, with_variational=True)
            roll_final["is_globally_stable"] = roll["is_globally_stable"] and roll_final["is_globally_stable"]
            roll = roll_final

        traj = np.asarray(roll["trajectory"], dtype=np.float64)
        vel = np.asarray(roll["velocities"], dtype=np.float64)
        g_base = np.asarray(bundle.g_base, dtype=np.float64)
        g_inv = roll["g_inv"]
        v0 = roll["v0"]
        dt_eff = dt_work

        freqs = self._phase3_small_oscillation_frequencies(q0, g_inv, potential_v_fn)
        if freqs.size == 0:
            if vel.shape[0] > 0:
                proxy = math.sqrt(max(0.01, float(vel[0] @ g_base @ vel[0])))
            else:
                proxy = 1.0
            freqs = np.full(d, proxy, dtype=np.float64)
        elliptic_min = float(np.min(freqs)) if freqs.size else 0.0

        kam_ok = self.check_kam_bruno_diophantine_condition(freqs)
        kam_worst = self._phase3_kam_worst_ratio(freqs)
        lindstedt = self._phase3_lindstedt_certificate(
            freqs, np.asarray(roll["h1_series"], dtype=np.float64))

        melnikov_val, melnikov_max = self._phase3_melnikov_from_trajectory(
            traj, vel, g_base, potential_v_fn, dt_eff)
        poincare_sec = self._phase3_poincare_section(
            traj, vel, section_point=q0, section_normal=v0, dt=dt_eff)

        period = max(float(roll["transit_time_t"]), 1e-3)
        floquet = self._phase3_floquet_from_monodromy(
            np.asarray(roll["monodromy"], dtype=np.float64),
            period,
            np.asarray(roll["lyapunov"], dtype=np.float64),
        )

        v_end_eff = vel[-1] if vel.shape[0] > 0 else v0
        rsi3 = self._phase3_rsi3_audit(float(roll["transit_time_t"]), g_base, v0, v_end_eff)
        lip_proxy = min(
            float(np.linalg.norm(v_end_eff - v0)) / max(float(np.linalg.norm(v0)), 1e-15),
            1.0)
        banach_t3 = float(lip_proxy ** 3)

        is_globally_stable = bool(roll["is_globally_stable"])
        lyap_max = float(floquet.lyapunov_max)
        if not is_globally_stable:
            regime = GeodesicRegime.HILL_WELL_VIOLATED
        elif abs(melnikov_val) > _MELNIKOV_TOL:
            regime = GeodesicRegime.MELNIKOV_CHAOTIC
        elif lyap_max > _LYAPUNOV_HYPERBOLIC:
            regime = GeodesicRegime.HYPERBOLIC_UNSTABLE
        elif kam_ok and poincare_sec.is_regular:
            regime = GeodesicRegime.KAM_TORUS_PRESERVED
        else:
            regime = GeodesicRegime.ELIPTIC_STABLE

        return BrachistochronePhysicalState(
            trajectory=self._freeze_array(traj),
            velocities=self._freeze_array(vel),
            transit_time_t=float(roll["transit_time_t"]),
            energy_drift_max=float(roll["energy_drift_max"]),
            is_globally_stable=is_globally_stable,
            fermat_energy_initial=float(roll["e_f_0"]),
            fermat_energy_drift_max=float(roll["fermat_drift_max"]),
            affine_time=float(roll["affine_time"]),
            steps_accepted=int(roll["steps"]),
            on_shell_residual=float(roll["on_shell_res"]),
            velocity_projected=bool(roll["projected"]),
            torsion_residual=float(bundle.torsion_residual),
            wilkinson_condition=float(bundle.wilkinson_condition),
            refractive_index_start=float(bundle.refractive_index),
            melnikov_value=float(melnikov_val),
            kam_condition_satisfied=bool(kam_ok),
            regime=regime,
            engine_version=__version__,
            kam_worst_ratio=float(kam_worst),
            melnikov_max_abs_integrand=float(melnikov_max),
            poincare_section_n_crossings=int(poincare_sec.n_crossings),
            poincare_section_dispersion=float(poincare_sec.return_dispersion),
            poincare_section_regular=bool(poincare_sec.is_regular),
            floquet_max_unit_deviation=float(floquet.max_unit_deviation),
            floquet_is_neutral=bool(floquet.is_neutral),
            floquet_exponents=self._freeze_array(np.asarray(floquet.exponents, dtype=np.float64)),
            banach_lipschitz_t3=float(banach_t3),
            fubini_study_distance=float(rsi3["fubini_study_distance"]),
            novikov_krull_valuation=float(rsi3["novikov_krull_valuation"]),
            lindstedt_residual=float(lindstedt.residual_max),
            rsi3_refinements=int(refinements),
            jacobi_lyapunov_max=lyap_max,
            monodromy_spectral_radius=float(floquet.spectral_radius),
            elliptic_frequency_min=elliptic_min,
            poincare_first_return_time=float(poincare_sec.first_return_time),
        )

    # ── CIERRE FASE 3 (governance) ────────────────────────────────────────────
    def emit_governance_state(
        self,
        phase2_env: Phase2Envelope,
        potential_v_fn: Callable[[NDArray[np.complex128] | NDArray[np.float64]], Any],
        t_max: float = 2.0,
        dt: float = 0.005,
        *,
        project_on_shell: bool = True,
    ) -> BrachistochronePhysicalState:
        r"""
        **Cierre formal de la FASE 3.**

        Consume `Phase2Envelope`, integra con auditoría Poincaré/RSI³ y
        emite `BrachistochronePhysicalState`.
        """
        b = phase2_env.bundle
        d = int(b.g_base.shape[0])
        q_s = b.phase1.q_start if b.phase1 is not None and b.phase1.q_start.size == d else np.zeros(d)
        v_s = b.phase1.v_start if b.phase1 is not None and b.phase1.v_start.size == d else np.zeros(d)
        return self.integrar_trayectoria_conforme(
            bundle=b, q_start=q_s, v_start=v_s,
            potential_v_fn=potential_v_fn,
            t_max=t_max, dt=dt, project_on_shell=project_on_shell,
        )


# ══════════════════════════════════════════════════════════════════════════════
# ORQUESTADOR (backward-compatible)
# ══════════════════════════════════════════════════════════════════════════════

class BrachistochronePathFinder(Morphism, Phase3_FermatGeodesicSolver):
    r"""
    Motor físico puro de la braquistócrona de Fermat.

    Orquesta las tres fases anidadas:
      FASE 1 (Observe)    → `emit_phase1_envelope` → `Phase1Envelope`
      FASE 2 (Orient)     → `from_phase1_envelope` + `emit_phase2_envelope`
                          → `Phase2Envelope`
      FASE 3 (Decide&Act) → `from_phase2_envelope` + `emit_governance_state`
                          → `BrachistochronePhysicalState`

    API pública `compute_brachistochrone_path(...)` idéntica a v4/v5.0.
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
        r"""
        Pipeline completo de tres fases. Firma idéntica a v4.0.0 / v5.0.0.
        """
        self._assert_callable_potential(potential_v_fn)
        g = np.asarray(g_base, dtype=np.float64)
        q0 = np.asarray(q_start, dtype=np.float64)
        v0 = np.asarray(v_start, dtype=np.float64)
        self._assert_square_matrix(g, "g_base")
        self._assert_vector(q0, g.shape[0], "q_start")
        self._assert_vector(v0, g.shape[0], "v_start")
        self._assert_finite_scalar(float(initial_h0), "initial_h0")

        env1 = self.emit_phase1_envelope(
            g_base=g, q_start=q0, v_start=v0,
            potential_v_fn=potential_v_fn, initial_h0=float(initial_h0))
        _ = self.from_phase1_envelope(env1)

        env2 = self.emit_phase2_envelope(env1, potential_v_fn=potential_v_fn)
        _ = self.from_phase2_envelope(env2)

        return self.emit_governance_state(
            env2, potential_v_fn=potential_v_fn,
            t_max=t_max, dt=dt, project_on_shell=project_on_shell)

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
        g_acc: float = 1.0, axis: int = -1,
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


# ══════════════════════════════════════════════════════════════════════════════
# EXPORTACIÓN CANÓNICA
# ══════════════════════════════════════════════════════════════════════════════
__all__ = [
    "GeodesicRegime",
    "BrachistochroneEngineError", "EnergyWellColapsoError",
    "ConformalSingularityError", "GeodesicIntegrationError",
    "KoszulChecksumError", "InitialKinematicsError",
    "SymplecticEmbeddingError", "EnvelopeChainViolationError",
    "KamBrunoViolationError", "MelnikovSplittingError",
    "PotentialField",
    "ConformalPotentialDilation", "Phase1Envelope",
    "ConformalGeometryBundle", "ConformalManifoldBundle", "Phase2Envelope",
    "PoincareSectionCertificate", "FloquetCertificate", "LindstedtCertificate",
    "BrachistochronePhysicalState",
    "Phase1_PotentialWellInquirer",
    "Phase2_ConformalKoszulSuturator",
    "Phase3_FermatGeodesicSolver",
    "BrachistochronePathFinder",
    "__version__",
]